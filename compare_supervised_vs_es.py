"""Compare supervised inverse training against ES-trained inverse training.

Supervised inverse:
    spectrum -> generator -> design logits
    loss = MSE(sigmoid(logits), known dataset design)

ES inverse:
    spectrum -> perturbed generator -> hard threshold -> scorer -> fitness

Both methods can be evaluated with a differentiable surrogate or with the
external MATLAB EM scorer because neither evaluation path requires gradients
through the scorer for ES, and supervised training does not use the scorer at
all during training.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from pixelant_eggroll.antenna_ops import force_feed_pixels, hard_threshold_design, mask_uniqueness, metal_fill_ratio
from pixelant_eggroll.config import DEFAULT_LAYOUT, ScorerConfig, parse_hidden_dims
from pixelant_eggroll.data import load_paired_antenna_mat
from pixelant_eggroll.fitness import FitnessConfig, compute_fitness
from pixelant_eggroll.models_jax import InverseGenerator, bn_eval, leaky_relu, linear_eval
from pixelant_eggroll.scorers import build_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="antenna_dataset.mat")
    parser.add_argument("--forward-checkpoint", default="Forward_model_for_tandem.pth")
    parser.add_argument("--output-dir", default="eggroll_runs/supervised_vs_es")
    parser.add_argument("--scorer", choices=("surrogate", "external-em"), default="surrogate")
    parser.add_argument("--solver-mode", choices=("air", "substrate"), default="air")
    parser.add_argument("--scorer-command", help="External scorer command with optional {input_mat}/{output_mat} placeholders.")
    parser.add_argument("--scorer-work-dir", default=".")
    parser.add_argument("--scorer-cache-dir", help="External scorer cache directory.")
    parser.add_argument("--scorer-timeout", type=float)
    parser.add_argument("--bad-spectrum-value", type=float, default=1.0e6)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--supervised-steps", type=int, default=200)
    parser.add_argument("--es-steps", type=int, default=50)
    parser.add_argument("--policy-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--population-size", type=int, default=128)
    parser.add_argument("--hidden-dims", default="128,128")
    parser.add_argument("--supervised-lr", type=float, default=1e-3)
    parser.add_argument("--es-lr", type=float, default=1e-2)
    parser.add_argument("--es-sigma", type=float, default=0.2)
    parser.add_argument("--es-rank", type=int, default=1)
    parser.add_argument("--policy-lr", type=float, default=1e-3)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--policy-entropy", type=float, default=0.0)
    parser.add_argument("--representative-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-conn", type=float, default=0.0)
    parser.add_argument("--lambda-area", type=float, default=0.0)
    parser.add_argument("--lambda-frag", type=float, default=0.0)
    parser.add_argument("--lambda-rule", type=float, default=0.0)
    return parser.parse_args()


def pure_generator_forward(params, frozen_params, spectrum: jnp.ndarray) -> jnp.ndarray:
    x = spectrum
    for idx in range(len(frozen_params["hidden_dims"])):
        x = linear_eval(x, params[f"fc{idx}"])
        bn = {**frozen_params[f"bn{idx}"], **params[f"bn{idx}"]}
        x = leaky_relu(bn_eval(x, bn))
    return linear_eval(x, params["out"])


def design_mse_loss(params, frozen_params, spectra: jnp.ndarray, designs: jnp.ndarray) -> jnp.ndarray:
    logits = pure_generator_forward(params, frozen_params, spectra)
    pred_design = jax.nn.sigmoid(logits).reshape((-1, 1, 12, 12))
    pred_design = pred_design.at[:, 0, 5:7, 0].set(1.0)
    return jnp.mean((pred_design - designs) ** 2)


def hard_designs(params, frozen_params, spectra: jnp.ndarray) -> jnp.ndarray:
    logits = pure_generator_forward(params, frozen_params, spectra)
    return hard_threshold_design(logits, layout=DEFAULT_LAYOUT)


def sample_policy_designs(params, frozen_params, spectra: jnp.ndarray, key, temperature: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    logits = pure_generator_forward(params, frozen_params, spectra)
    probs = jax.nn.sigmoid(logits / temperature)
    flat = jax.random.bernoulli(key, probs).astype(jnp.float32)
    designs = flat.reshape((-1, 1, 12, 12))
    designs = force_feed_pixels(designs, layout=DEFAULT_LAYOUT)
    return logits, designs


def policy_gradient_loss(
    params,
    frozen_params,
    spectra: jnp.ndarray,
    sampled_designs: jnp.ndarray,
    advantages: jnp.ndarray,
    temperature: float,
    entropy_coef: float,
) -> jnp.ndarray:
    logits = pure_generator_forward(params, frozen_params, spectra) / temperature
    sampled_flat = sampled_designs.reshape((sampled_designs.shape[0], -1))
    log_prob = sampled_flat * jax.nn.log_sigmoid(logits) + (1.0 - sampled_flat) * jax.nn.log_sigmoid(-logits)
    log_prob = jnp.mean(log_prob, axis=-1)
    probs = jax.nn.sigmoid(logits)
    entropy = -probs * jax.nn.log_sigmoid(logits) - (1.0 - probs) * jax.nn.log_sigmoid(-logits)
    entropy = jnp.mean(entropy, axis=-1)
    return -jnp.mean(jax.lax.stop_gradient(advantages) * log_prob) - entropy_coef * jnp.mean(entropy)


def make_scorer(args: argparse.Namespace):
    if args.scorer == "surrogate":
        scorer_config = ScorerConfig(kind="surrogate", checkpoint_path=Path(args.forward_checkpoint))
    else:
        scorer_config = ScorerConfig(
            kind="external-em",
            command=args.scorer_command,
            work_dir=Path(args.scorer_work_dir),
            solver_mode=args.solver_mode,
            cache_dir=Path(args.scorer_cache_dir) if args.scorer_cache_dir else None,
            timeout_seconds=args.scorer_timeout,
            bad_spectrum_value=args.bad_spectrum_value,
        )
    return build_scorer(scorer_config)


def evaluate_model(params, frozen_params, spectra, targets, scorer, fitness_config):
    designs = hard_designs(params, frozen_params, spectra)
    pred = scorer.score(designs)
    fitness, metrics = compute_fitness(pred, targets, designs, fitness_config, layout=DEFAULT_LAYOUT)
    return {
        "spectrum_mse": float(jax.device_get(jnp.mean(metrics["spectrum_mse"]))),
        "fitness_mean": float(jax.device_get(jnp.mean(fitness))),
        "mask_uniqueness": float(jax.device_get(mask_uniqueness(designs, layout=DEFAULT_LAYOUT))),
        "metal_fill": float(jax.device_get(jnp.mean(metal_fill_ratio(designs, layout=DEFAULT_LAYOUT)))),
    }


def main() -> None:
    args = parse_args()
    if args.policy_steps is None:
        args.policy_steps = args.es_steps
    if args.population_size % 2 != 0:
        raise SystemExit("--population-size must be even for antithetic EGGROLL pairs")

    import hyperscalees as hs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(args.seed)
    model_key, es_key, data_key = jax.random.split(key, 3)
    spectra, designs = load_paired_antenna_mat(args.dataset)
    total_needed = args.train_size + args.val_size + args.test_size
    if spectra.shape[0] < total_needed:
        total_needed = int(spectra.shape[0])
        args.test_size = min(args.test_size, total_needed // 5)
        args.val_size = min(args.val_size, total_needed // 5)
        args.train_size = total_needed - args.val_size - args.test_size
    train_spectra = spectra[: args.train_size]
    train_designs = designs[: args.train_size]
    val_spectra = spectra[args.train_size : args.train_size + args.val_size]
    val_designs = designs[args.train_size : args.train_size + args.val_size]
    test_start = args.train_size + args.val_size
    test_spectra = spectra[test_start : test_start + args.test_size]
    test_designs = designs[test_start : test_start + args.test_size]

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    init = InverseGenerator.rand_init(model_key, hidden_dims=hidden_dims, input_dim=81, output_dim=144)
    frozen_params = init.frozen_params
    supervised_params = jax.tree.map(lambda x: x.copy(), init.params)
    es_params = jax.tree.map(lambda x: x.copy(), init.params)
    policy_params = jax.tree.map(lambda x: x.copy(), init.params)
    scorer = make_scorer(args)
    fitness_config = FitnessConfig(
        lambda_conn=args.lambda_conn,
        lambda_area=args.lambda_area,
        lambda_frag=args.lambda_frag,
        lambda_rule=args.lambda_rule,
    )

    supervised_opt = optax.adam(args.supervised_lr)
    supervised_state = supervised_opt.init(supervised_params)
    supervised_step = jax.jit(jax.value_and_grad(design_mse_loss))
    policy_opt = optax.adam(args.policy_lr)
    policy_state = policy_opt.init(policy_params)
    policy_step = jax.jit(jax.value_and_grad(policy_gradient_loss))

    noiser = hs.noiser.eggroll.EggRoll
    es_tree_key = hs.models.common.simple_es_tree_key(es_params, es_key, init.scan_map)
    frozen_noiser_params, noiser_params = noiser.init_noiser(
        es_params,
        args.es_sigma,
        args.es_lr,
        solver=optax.adam,
        rank=args.es_rank,
        freeze_nonlora=False,
    )

    def es_member(noiser_params_, params_, iterinfo_, spectrum_):
        return InverseGenerator.forward(
            noiser,
            frozen_noiser_params,
            noiser_params_,
            frozen_params,
            params_,
            es_tree_key,
            iterinfo_,
            spectrum_,
        )

    batched_es = jax.jit(jax.vmap(es_member, in_axes=(None, None, 0, 0)))
    es_update = jax.jit(lambda n, p, f, i: noiser.do_updates(frozen_noiser_params, n, p, es_tree_key, f, i, init.es_map))

    def es_score_population(noiser_params_, params_, iterinfos_, targets_):
        logits = batched_es(noiser_params_, params_, iterinfos_, targets_)
        pop_designs = hard_threshold_design(logits, layout=DEFAULT_LAYOUT)
        pred = scorer.score(pop_designs)
        raw_fitness, _metrics = compute_fitness(pred, targets_, pop_designs, fitness_config, layout=DEFAULT_LAYOUT)
        return raw_fitness

    rows = []
    max_steps = max(args.supervised_steps, args.es_steps, args.policy_steps)
    eval_steps = {0, args.supervised_steps, args.es_steps, args.policy_steps, max_steps}
    eval_steps.update(range(0, max_steps + 1, max(1, args.eval_every)))
    for step in range(max_steps + 1):
        if step in eval_steps:
            sup_eval = evaluate_model(supervised_params, frozen_params, val_spectra, val_spectra, scorer, fitness_config)
            es_eval = evaluate_model(es_params, frozen_params, val_spectra, val_spectra, scorer, fitness_config)
            policy_eval = evaluate_model(policy_params, frozen_params, val_spectra, val_spectra, scorer, fitness_config)
            sup_test_eval = evaluate_model(supervised_params, frozen_params, test_spectra, test_spectra, scorer, fitness_config)
            es_test_eval = evaluate_model(es_params, frozen_params, test_spectra, test_spectra, scorer, fitness_config)
            policy_test_eval = evaluate_model(policy_params, frozen_params, test_spectra, test_spectra, scorer, fitness_config)
            sup_design_loss = float(jax.device_get(design_mse_loss(supervised_params, frozen_params, val_spectra, val_designs)))
            es_design_loss = float(jax.device_get(design_mse_loss(es_params, frozen_params, val_spectra, val_designs)))
            policy_design_loss = float(jax.device_get(design_mse_loss(policy_params, frozen_params, val_spectra, val_designs)))
            sup_test_design_loss = float(jax.device_get(design_mse_loss(supervised_params, frozen_params, test_spectra, test_designs)))
            es_test_design_loss = float(jax.device_get(design_mse_loss(es_params, frozen_params, test_spectra, test_designs)))
            policy_test_design_loss = float(jax.device_get(design_mse_loss(policy_params, frozen_params, test_spectra, test_designs)))
            row = {
                "step": step,
                "supervised_design_mse": sup_design_loss,
                "supervised_spectrum_mse": sup_eval["spectrum_mse"],
                "supervised_test_design_mse": sup_test_design_loss,
                "supervised_test_spectrum_mse": sup_test_eval["spectrum_mse"],
                "supervised_uniqueness": sup_eval["mask_uniqueness"],
                "supervised_fill": sup_eval["metal_fill"],
                "es_design_mse": es_design_loss,
                "es_spectrum_mse": es_eval["spectrum_mse"],
                "es_test_design_mse": es_test_design_loss,
                "es_test_spectrum_mse": es_test_eval["spectrum_mse"],
                "es_uniqueness": es_eval["mask_uniqueness"],
                "es_fill": es_eval["metal_fill"],
                "policy_design_mse": policy_design_loss,
                "policy_spectrum_mse": policy_eval["spectrum_mse"],
                "policy_test_design_mse": policy_test_design_loss,
                "policy_test_spectrum_mse": policy_test_eval["spectrum_mse"],
                "policy_uniqueness": policy_eval["mask_uniqueness"],
                "policy_fill": policy_eval["metal_fill"],
            }
            rows.append(row)
            print(
                f"step={step:05d} "
                f"supervised val_mse={sup_eval['spectrum_mse']:.6g} test_mse={sup_test_eval['spectrum_mse']:.6g}; "
                f"es val_mse={es_eval['spectrum_mse']:.6g} test_mse={es_test_eval['spectrum_mse']:.6g}; "
                f"policy val_mse={policy_eval['spectrum_mse']:.6g} test_mse={policy_test_eval['spectrum_mse']:.6g}"
            )

        if step == max_steps:
            break

        data_key, sup_key, es_key_step, policy_key = jax.random.split(data_key, 4)
        if step < args.supervised_steps:
            idx = jax.random.randint(sup_key, (args.batch_size,), 0, train_spectra.shape[0])
            loss, grads = supervised_step(supervised_params, frozen_params, train_spectra[idx], train_designs[idx])
            updates, supervised_state = supervised_opt.update(grads, supervised_state, supervised_params)
            supervised_params = optax.apply_updates(supervised_params, updates)

        if step < args.es_steps:
            pair_count = args.population_size // 2
            idx = jax.random.randint(es_key_step, (pair_count,), 0, train_spectra.shape[0])
            targets = jnp.repeat(train_spectra[idx], 2, axis=0)
            iterinfos = (jnp.full((args.population_size,), step, dtype=jnp.int32), jnp.arange(args.population_size))
            raw_fitness = es_score_population(noiser_params, es_params, iterinfos, targets)
            fitnesses = noiser.convert_fitnesses(frozen_noiser_params, noiser_params, raw_fitness)
            noiser_params, es_params = es_update(noiser_params, es_params, fitnesses, iterinfos)

        if step < args.policy_steps:
            policy_key, idx_key, sample_key = jax.random.split(policy_key, 3)
            idx = jax.random.randint(idx_key, (args.population_size,), 0, train_spectra.shape[0])
            targets = train_spectra[idx]
            _logits, sampled_designs = sample_policy_designs(
                policy_params,
                frozen_params,
                targets,
                sample_key,
                args.policy_temperature,
            )
            pred = scorer.score(sampled_designs)
            raw_fitness, _metrics = compute_fitness(pred, targets, sampled_designs, fitness_config, layout=DEFAULT_LAYOUT)
            advantages = (raw_fitness - jnp.mean(raw_fitness)) / (jnp.std(raw_fitness) + 1e-6)
            _loss, grads = policy_step(
                policy_params,
                frozen_params,
                targets,
                sampled_designs,
                advantages,
                args.policy_temperature,
                args.policy_entropy,
            )
            updates, policy_state = policy_opt.update(grads, policy_state, policy_params)
            policy_params = optax.apply_updates(policy_params, updates)

    csv_path = output_dir / "supervised_vs_es.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    steps = [row["step"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [row["supervised_spectrum_mse"] for row in rows], label="Supervised inverse")
    ax.plot(steps, [row["es_spectrum_mse"] for row in rows], label="ES inverse")
    ax.plot(steps, [row["policy_spectrum_mse"] for row in rows], label="Policy gradient")
    ax.set_xlabel("Step")
    ax.set_ylabel(f"{args.scorer} hard-mask spectrum MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "supervised_vs_es.png", dpi=160)
    plt.close(fig)

    _save_metric_plot(
        output_dir / "design_mse.png",
        steps,
        [row["supervised_design_mse"] for row in rows],
        [row["es_design_mse"] for row in rows],
        [row["policy_design_mse"] for row in rows],
        "Design-label MSE",
    )
    _save_metric_plot(
        output_dir / "mask_uniqueness.png",
        steps,
        [row["supervised_uniqueness"] for row in rows],
        [row["es_uniqueness"] for row in rows],
        [row["policy_uniqueness"] for row in rows],
        "Unique hard masks / validation batch",
    )
    _save_metric_plot(
        output_dir / "metal_fill.png",
        steps,
        [row["supervised_fill"] for row in rows],
        [row["es_fill"] for row in rows],
        [row["policy_fill"] for row in rows],
        "Mean metal fill ratio",
    )
    _save_metric_plot(
        output_dir / "test_spectrum_mse.png",
        steps,
        [row["supervised_test_spectrum_mse"] for row in rows],
        [row["es_test_spectrum_mse"] for row in rows],
        [row["policy_test_spectrum_mse"] for row in rows],
        f"{args.scorer} hard-mask TEST spectrum MSE",
    )
    _save_metric_plot(
        output_dir / "test_design_mse.png",
        steps,
        [row["supervised_test_design_mse"] for row in rows],
        [row["es_test_design_mse"] for row in rows],
        [row["policy_test_design_mse"] for row in rows],
        "TEST design-label MSE",
    )
    _save_representative_s11_cuts(
        output_dir=output_dir,
        count=args.representative_count,
        spectra=test_spectra,
        supervised_params=supervised_params,
        es_params=es_params,
        policy_params=policy_params,
        frozen_params=frozen_params,
        scorer=scorer,
    )


def _save_metric_plot(
    path: Path,
    steps: list[int],
    supervised: list[float],
    es: list[float],
    policy: list[float],
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, supervised, marker="o", label="Supervised inverse")
    ax.plot(steps, es, marker="o", label="ES inverse")
    ax.plot(steps, policy, marker="o", label="Policy gradient")
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_representative_s11_cuts(
    output_dir: Path,
    count: int,
    spectra: jnp.ndarray,
    supervised_params,
    es_params,
    policy_params,
    frozen_params,
    scorer,
) -> None:
    count = max(0, min(int(count), int(spectra.shape[0])))
    if count == 0:
        return

    targets = spectra[:count]
    methods = {
        "Supervised": supervised_params,
        "ES": es_params,
        "Policy gradient": policy_params,
    }
    predicted = {}
    masks = {}
    for label, params in methods.items():
        designs = hard_designs(params, frozen_params, targets)
        masks[label] = jax.device_get(designs[:, 0])
        predicted[label] = jax.device_get(scorer.score(designs))

    targets_np = jax.device_get(targets)
    freq = getattr(scorer, "freq", None)
    if freq is None:
        x = jnp.linspace(1.0, 5.0, targets_np.shape[1])
        xlabel = "Frequency (GHz, dataset grid)"
    else:
        x = jnp.asarray(freq) / 1.0e9
        xlabel = "Frequency (GHz)"
    x = jax.device_get(x)

    cols = min(3, count)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.1 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= count:
            ax.axis("off")
            continue
        ax.plot(x, targets_np[idx], color="black", linewidth=2.0, label="Target")
        ax.plot(x, predicted["Supervised"][idx], linewidth=1.4, label="Supervised")
        ax.plot(x, predicted["ES"][idx], linewidth=1.4, label="ES")
        ax.plot(x, predicted["Policy gradient"][idx], linewidth=1.4, label="Policy")
        ax.set_title(f"Test target {idx}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("S11 (dB)")
        ax.grid(True, alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "representative_s11_cuts.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(count, 3, figsize=(5.4, 1.9 * count), squeeze=False)
    for row in range(count):
        for col, label in enumerate(methods):
            ax = axes[row][col]
            ax.imshow(masks[label][row], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label)
    fig.tight_layout()
    fig.savefig(output_dir / "representative_hard_masks.png", dpi=180)
    plt.close(fig)

    npz_payload = {"target": targets_np, "frequency_ghz": x}
    for label, values in predicted.items():
        key = label.lower().replace(" ", "_")
        npz_payload[f"{key}_s11"] = values
        npz_payload[f"{key}_mask"] = masks[label]
    import numpy as np

    np.savez_compressed(output_dir / "representative_s11_cuts.npz", **npz_payload)


if __name__ == "__main__":
    main()
