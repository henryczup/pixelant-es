"""Train a target-conditioned hard-binary PNGF antenna generator.

This path uses no neural surrogate.  Fitness comes from an external PNGF scorer
command that consumes `input.npz` and writes `output.npz`; see `PNGFScorer`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from pixelant_eggroll.antenna_ops import mask_uniqueness, metal_fill_ratio
from pixelant_eggroll.checkpoints import save_inverse_npz
from pixelant_eggroll.config import GeneratorConfig, ScorerConfig
from pixelant_eggroll.models_jax import DirectCNNGenerator, direct_cnn_eval
from pixelant_eggroll.pngf import (
    PNGF_FREQ_HZ,
    PNGF_LAYOUT,
    PNGF_TARGET_DIM,
    hard_project_pngf_center_fed,
    pack_pngf_targets,
    pngf_target_errors,
    project_pngf_center_fed_mask_np,
    unpack_pngf_targets,
)
from pixelant_eggroll.scorers import PNGFScorer, build_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", required=True, help="PNGF dataset with masks/designs and 15-value targets.")
    parser.add_argument("--output-dir", default="eggroll_runs/pngf_es", help="Directory for logs and checkpoints.")
    parser.add_argument("--mode", choices=("pretrain", "es", "both"), default="both")
    parser.add_argument("--accepted-only", action="store_true", help="Use only accepted DBS records from exported datasets.")
    parser.add_argument("--pngf-command", help="External PNGF batch scorer command with {input_npz} and {output_npz}.")
    parser.add_argument("--pngf-work-dir", default=r"C:\Users\hczupryna\dev\PNGF", help="External PNGF repo/build directory.")
    parser.add_argument("--pngf-cache-dir", default="eggroll_runs/pngf_cache")
    parser.add_argument("--scorer-timeout", type=float, default=3600.0)
    parser.add_argument("--bad-spectrum-value", type=float, default=1.0e6)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma-decay", type=float, default=1.0)
    parser.add_argument("--noise-reuse", type=int, default=0)
    parser.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0, help="Directivity target-match weight.")
    parser.add_argument("--directivity-scale", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze-nonlora", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def optimizer_factory(name: str):
    return {"sgd": optax.sgd, "adam": optax.adam, "adamw": optax.adamw}[name]


def load_pngf_dataset(path: str | Path, *, accepted_only: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "masks" in data:
            masks = data["masks"]
        elif "designs" in data:
            masks = data["designs"]
        else:
            raise ValueError(f"{path} must contain `masks` or `designs`")
        if "targets" in data:
            targets = data["targets"].astype(np.float32)
        elif {"s11_re", "s11_im", "directivity"}.issubset(set(data.files)):
            targets = pack_pngf_targets(s11_re=data["s11_re"], s11_im=data["s11_im"], directivity=data["directivity"])
        elif {"s11_complex", "directivity"}.issubset(set(data.files)):
            targets = pack_pngf_targets(s11=data["s11_complex"], directivity=data["directivity"])
        else:
            raise ValueError(f"{path} must contain `targets` or S11/directivity fields")
        if accepted_only and "accepted" in data:
            keep = np.asarray(data["accepted"], dtype=bool)
            masks = masks[keep]
            targets = targets[keep]
    masks = project_pngf_center_fed_mask_np(masks)[:, None, :, :]
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, PNGF_TARGET_DIM)
    if masks.shape[0] != targets.shape[0]:
        raise ValueError(f"Dataset mask/target count mismatch: {masks.shape[0]} vs {targets.shape[0]}")
    if masks.shape[0] == 0:
        raise ValueError("PNGF dataset is empty after filtering")
    return jnp.asarray(targets, dtype=jnp.float32), jnp.asarray(masks, dtype=jnp.float32)


def _sample_indices(key, dataset_size: int, batch_size: int) -> jnp.ndarray:
    return jax.random.randint(key, (batch_size,), 0, dataset_size)


def _sample_antithetic_targets(key, targets: jnp.ndarray, population_size: int) -> jnp.ndarray:
    if population_size % 2 != 0:
        raise ValueError("population-size must be even for antithetic PNGF ES")
    pair_count = population_size // 2
    idx = _sample_indices(key, targets.shape[0], pair_count)
    return jnp.repeat(targets[idx], 2, axis=0)


def pretrain_generator(
    params,
    frozen_params,
    targets: jnp.ndarray,
    masks: jnp.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    key,
) -> tuple[dict, list[dict[str, float]]]:
    tx = optax.adamw(lr)
    opt_state = tx.init(params)

    def loss_fn(params_, x_, y_):
        logits = direct_cnn_eval(params_, frozen_params, x_)
        loss = optax.sigmoid_binary_cross_entropy(logits, y_).mean()
        projected = hard_project_pngf_center_fed(logits)
        recon_mse = jnp.mean(jnp.square(projected - y_))
        return loss, recon_mse

    @jax.jit
    def train_step(params_, opt_state_, x_, y_):
        (loss, recon_mse), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_, x_, y_)
        updates, opt_state_ = tx.update(grads, opt_state_, params_)
        params_ = optax.apply_updates(params_, updates)
        return params_, opt_state_, loss, recon_mse

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        key, batch_key = jax.random.split(key)
        idx = _sample_indices(batch_key, targets.shape[0], min(batch_size, targets.shape[0]))
        params, opt_state, loss, recon_mse = train_step(params, opt_state, targets[idx], masks[idx])
        row = {"epoch": float(epoch), "bce": float(jax.device_get(loss)), "hard_recon_mse": float(jax.device_get(recon_mse))}
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"pretrain epoch={epoch:04d} bce={row['bce']:.6g} hard_recon_mse={row['hard_recon_mse']:.6g}")
    return params, history


def save_best_pngf_design(path: Path, mask: jnp.ndarray, target: jnp.ndarray, prediction: jnp.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_np = np.asarray(jax.device_get(mask[0, 0]))
    target_np = np.asarray(jax.device_get(target))
    pred_np = np.asarray(jax.device_get(prediction))
    target_re, target_im, target_d = (np.asarray(x) for x in unpack_pngf_targets(target_np))
    pred_re, pred_im, pred_d = (np.asarray(x) for x in unpack_pngf_targets(pred_np))
    target_db = 20.0 * np.log10(np.maximum(np.sqrt(target_re[0] ** 2 + target_im[0] ** 2), 1e-12))
    pred_db = 20.0 * np.log10(np.maximum(np.sqrt(pred_re[0] ** 2 + pred_im[0] ** 2), 1e-12))
    freq_ghz = PNGF_FREQ_HZ / 1e9

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    axes[0].imshow(mask_np, cmap="gray_r", interpolation="nearest")
    axes[0].set_title("Hard PNGF mask")
    axes[0].axis("off")
    axes[1].plot(freq_ghz, target_db, marker="o", label="target")
    axes[1].plot(freq_ghz, pred_db, marker="o", label="PNGF")
    axes[1].set_title("S11 dB")
    axes[1].set_xlabel("GHz")
    axes[1].legend()
    axes[2].plot(freq_ghz, target_d[0], marker="o", label="target")
    axes[2].plot(freq_ghz, pred_d[0], marker="o", label="PNGF")
    axes[2].set_title("Directivity")
    axes[2].set_xlabel("GHz")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_es(
    params,
    frozen_params,
    scan_map,
    es_map,
    targets: jnp.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    key,
) -> dict:
    try:
        import hyperscalees as hs
    except ImportError as exc:
        raise SystemExit("Install EGGROLL dependencies first: `pip install -r requirements-eggroll.txt`.") from exc

    scorer = build_scorer(
        ScorerConfig(
            kind="pngf",
            command=args.pngf_command,
            work_dir=Path(args.pngf_work_dir),
            cache_dir=Path(args.pngf_cache_dir),
            timeout_seconds=args.scorer_timeout,
            bad_spectrum_value=args.bad_spectrum_value,
        )
    )
    if not isinstance(scorer, PNGFScorer):
        raise TypeError("Expected PNGFScorer")
    missing = scorer.missing_center_fed_matrices(args.pngf_work_dir)
    if missing:
        print("warning: missing paper center-fed PNGF matrices:")
        for path in missing:
            print(f"  {path}")
        print("the scorer can still run from cache or a custom command, but uncached real PNGF calls need these files.")

    noiser_cls = hs.noiser.eggroll.EggRoll
    es_key, data_key = jax.random.split(key)
    es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser_params, noiser_params = noiser_cls.init_noiser(
        params,
        args.sigma,
        args.lr,
        solver=optimizer_factory(args.optimizer),
        solver_kwargs={},
        rank=args.rank,
        noise_reuse=args.noise_reuse,
        freeze_nonlora=args.freeze_nonlora,
        use_batched_update=True,
    )

    def generator_apply(noiser_params_, params_, iterinfo_, target_):
        return DirectCNNGenerator.forward(
            noiser_cls,
            frozen_noiser_params,
            noiser_params_,
            frozen_params,
            params_,
            es_tree_key,
            iterinfo_,
            target_,
            None,
        )

    batched_generator = jax.jit(jax.vmap(generator_apply, in_axes=(None, None, 0, 0)))
    update = jax.jit(lambda n, p, f, i: noiser_cls.do_updates(frozen_noiser_params, n, p, es_tree_key, f, i, es_map))

    @jax.jit
    def generate_masks(noiser_params_, params_, iterinfos_, targets_):
        logits = batched_generator(noiser_params_, params_, iterinfos_, targets_)
        return hard_project_pngf_center_fed(logits)

    log_path = output_dir / "pngf_es_metrics.csv"
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "step",
                "sigma",
                "fitness_mean",
                "fitness_best",
                "target_error_mean",
                "s11_error_mean",
                "directivity_error_mean",
                "mask_uniqueness",
                "metal_fill_ratio",
                "valid_ratio",
            ],
        )
        writer.writeheader()
        best_fitness = -jnp.inf
        best_payload = None
        for step in range(args.steps):
            data_key, target_key = jax.random.split(data_key)
            target_batch = _sample_antithetic_targets(target_key, targets, args.population_size)
            iterinfos = (jnp.full((args.population_size,), step, dtype=jnp.int32), jnp.arange(args.population_size))
            masks = generate_masks(noiser_params, params, iterinfos, target_batch)
            predictions = scorer.score(masks)
            errors = pngf_target_errors(
                predictions,
                target_batch,
                beta=args.beta,
                directivity_scale=args.directivity_scale,
            )
            raw_fitness = -errors.total
            fitnesses = noiser_cls.convert_fitnesses(frozen_noiser_params, noiser_params, raw_fitness)
            noiser_params, params = update(noiser_params, params, fitnesses, iterinfos)
            if args.sigma_decay != 1.0:
                noiser_params["sigma"] = noiser_params["sigma"] * args.sigma_decay

            best_idx = int(jax.device_get(jnp.argmax(raw_fitness)))
            row = {
                "step": step,
                "sigma": float(jax.device_get(noiser_params["sigma"])),
                "fitness_mean": float(jax.device_get(jnp.mean(raw_fitness))),
                "fitness_best": float(jax.device_get(jnp.max(raw_fitness))),
                "target_error_mean": float(jax.device_get(jnp.mean(errors.total))),
                "s11_error_mean": float(jax.device_get(jnp.mean(errors.s11))),
                "directivity_error_mean": float(jax.device_get(jnp.mean(errors.directivity))),
                "mask_uniqueness": float(jax.device_get(mask_uniqueness(masks, layout=PNGF_LAYOUT))),
                "metal_fill_ratio": float(jax.device_get(jnp.mean(metal_fill_ratio(masks, layout=PNGF_LAYOUT)))),
                "valid_ratio": float(np.mean(scorer.last_valid)) if len(scorer.last_valid) else 0.0,
            }
            writer.writerow(row)
            fh.flush()
            print(
                f"step={step:04d} fitness_best={row['fitness_best']:.6g} "
                f"err={row['target_error_mean']:.6g} s11={row['s11_error_mean']:.6g} "
                f"dir={row['directivity_error_mean']:.6g} unique={row['mask_uniqueness']:.3f} "
                f"valid={row['valid_ratio']:.2f}"
            )

            if raw_fitness[best_idx] > best_fitness:
                best_fitness = raw_fitness[best_idx]
                best_payload = (masks[best_idx : best_idx + 1], target_batch[best_idx : best_idx + 1], predictions[best_idx : best_idx + 1])
                save_best_pngf_design(output_dir / "best_pngf_design.png", *best_payload)

            if (step + 1) % args.save_every == 0 or step == args.steps - 1:
                save_inverse_npz(
                    output_dir / f"pngf_generator_step_{step + 1}.npz",
                    frozen_params,
                    params,
                    {
                        "step": step + 1,
                        "generator_config": asdict(GeneratorConfig(kind="cnn", layout=PNGF_LAYOUT, hidden_dims=(args.channels,), latent_dim=0, spectrum_dim=PNGF_TARGET_DIM)),
                        "scorer": "pngf",
                        "rank": args.rank,
                        "lr": args.lr,
                        "sigma": float(jax.device_get(noiser_params["sigma"])),
                        "beta": args.beta,
                        "directivity_scale": args.directivity_scale,
                    },
                )
    return params


def main() -> None:
    args = parse_args()
    if args.population_size % 2 != 0:
        raise SystemExit("--population-size must be even for antithetic ES")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets, masks = load_pngf_dataset(args.dataset_npz, accepted_only=args.accepted_only)
    generator_config = GeneratorConfig(kind="cnn", layout=PNGF_LAYOUT, hidden_dims=(args.channels,), latent_dim=0, spectrum_dim=PNGF_TARGET_DIM)
    key = jax.random.key(args.seed)
    init_key, pretrain_key, es_key = jax.random.split(key, 3)
    init = DirectCNNGenerator.rand_init(init_key, generator_config, channels=args.channels)
    frozen_params, params, scan_map, es_map = init.frozen_params, init.params, init.scan_map, init.es_map

    if args.mode in {"pretrain", "both"}:
        params, history = pretrain_generator(
            params,
            frozen_params,
            targets,
            masks,
            epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch_size,
            lr=args.pretrain_lr,
            key=pretrain_key,
        )
        with (output_dir / "pngf_pretrain_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["epoch", "bce", "hard_recon_mse"])
            writer.writeheader()
            writer.writerows(history)
        save_inverse_npz(
            output_dir / "pngf_generator_pretrained.npz",
            frozen_params,
            params,
            {
                "stage": "pretrain",
                "generator_config": asdict(generator_config),
                "pretrain_epochs": args.pretrain_epochs,
                "pretrain_lr": args.pretrain_lr,
            },
        )

    if args.mode in {"es", "both"}:
        params = run_es(params, frozen_params, scan_map, es_map, targets, output_dir, args, key=es_key)
        save_inverse_npz(
            output_dir / "pngf_generator_final.npz",
            frozen_params,
            params,
            {
                "stage": "final",
                "generator_config": asdict(generator_config),
                "es_steps": args.steps,
                "rank": args.rank,
                "sigma": args.sigma,
                "lr": args.lr,
            },
        )


if __name__ == "__main__":
    main()
