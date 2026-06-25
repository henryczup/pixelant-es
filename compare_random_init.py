"""Head-to-head random-init comparison: EGGROLL hard binary vs ST gradient.

This is a lightweight synthetic benchmark for environments where the real
antenna `.mat` data and `.pth` surrogate checkpoints are not present. Both
methods start from the same random inverse-generator weights and optimize
against the same frozen toy surrogate:

    target spectrum -> generator -> design -> frozen surrogate -> MSE

EGGROLL uses hard thresholding. The gradient baseline uses a smooth tanh
straight-through-style activation during training and is evaluated with the
same hard thresholding used by EGGROLL.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from pixelant_eggroll.antenna_ops import hard_threshold_design, mask_uniqueness, metal_fill_ratio
from pixelant_eggroll.models_jax import InverseGenerator, bn_eval, leaky_relu, linear_eval


@dataclass
class History:
    eggroll_val_mse: list[float]
    gradient_val_mse: list[float]
    eggroll_uniqueness: list[float]
    gradient_uniqueness: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--population-size", type=int, default=128)
    parser.add_argument("--dataset-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--hidden-dims", default="64,64")
    parser.add_argument("--eggroll-rank", type=int, default=1)
    parser.add_argument("--eggroll-sigma", type=float, default=0.2)
    parser.add_argument("--eggroll-lr", type=float, default=0.03)
    parser.add_argument("--gradient-lr", type=float, default=0.003)
    parser.add_argument("--st-slope", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="eggroll_runs/random_init_compare")
    return parser.parse_args()


def force_feed_flat(flat: jnp.ndarray) -> jnp.ndarray:
    flat = flat.reshape((-1, 1, 12, 12))
    flat = flat.at[:, 0, 5, 0].set(1.0)
    flat = flat.at[:, 0, 6, 0].set(1.0)
    return flat.reshape((-1, 144))


def toy_forward(design_nchw: jnp.ndarray, surrogate: dict[str, jnp.ndarray]) -> jnp.ndarray:
    flat = design_nchw.reshape((design_nchw.shape[0], -1))
    return 10.0 * jnp.tanh(flat @ surrogate["weight"] + surrogate["bias"])


def make_dataset(key, n: int, surrogate: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    mask_key, = jax.random.split(key, 1)
    masks = jax.random.bernoulli(mask_key, p=0.45, shape=(n, 144)).astype(jnp.float32)
    masks = force_feed_flat(masks)
    spectra = toy_forward(masks.reshape((-1, 1, 12, 12)), surrogate)
    return spectra, masks


def make_surrogate(key) -> dict[str, jnp.ndarray]:
    weight_key, bias_key = jax.random.split(key)
    return {
        "weight": jax.random.normal(weight_key, (144, 81), dtype=jnp.float32) / jnp.sqrt(144.0),
        "bias": 0.05 * jax.random.normal(bias_key, (81,), dtype=jnp.float32),
    }


def pure_generator_forward(params, frozen_params, spectrum: jnp.ndarray) -> jnp.ndarray:
    x = linear_eval(spectrum, params["fc0"])
    bn0 = {**frozen_params["bn0"], **params["bn0"]}
    x = leaky_relu(bn_eval(x, bn0))
    x = linear_eval(x, params["fc1"])
    bn1 = {**frozen_params["bn1"], **params["bn1"]}
    x = leaky_relu(bn_eval(x, bn1))
    return linear_eval(x, params["out"])


def st_design(logits: jnp.ndarray, slope: float) -> jnp.ndarray:
    soft = 0.5 + 0.5 * jnp.tanh(slope * logits)
    soft = soft.reshape((-1, 1, 12, 12))
    soft = soft.at[:, 0, 5, 0].set(1.0)
    soft = soft.at[:, 0, 6, 0].set(1.0)
    return soft


def evaluate_hard(params, frozen_params, spectra: jnp.ndarray, surrogate: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    logits = pure_generator_forward(params, frozen_params, spectra)
    designs = hard_threshold_design(logits)
    pred = toy_forward(designs, surrogate)
    return jnp.mean((pred - spectra) ** 2), designs


def main() -> None:
    args = parse_args()
    if args.population_size % 2 != 0:
        raise SystemExit("--population-size must be even for antithetic EGGROLL pairs")

    import hyperscalees as hs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(","))
    key = jax.random.key(args.seed)
    model_key, es_key, surrogate_key, train_key, val_key = jax.random.split(key, 5)

    surrogate = make_surrogate(surrogate_key)
    train_spectra, _ = make_dataset(train_key, args.dataset_size, surrogate)
    val_spectra, _ = make_dataset(val_key, args.val_size, surrogate)

    init = InverseGenerator.rand_init(model_key, hidden_dims=hidden_dims)
    frozen_params = init.frozen_params
    eggroll_params = init.params
    gradient_params = jax.tree.map(lambda x: x.copy(), init.params)

    noiser = hs.noiser.eggroll.EggRoll
    es_tree_key = hs.models.common.simple_es_tree_key(eggroll_params, es_key, init.scan_map)
    frozen_noiser_params, noiser_params = noiser.init_noiser(
        eggroll_params,
        args.eggroll_sigma,
        args.eggroll_lr,
        solver=optax.adam,
        rank=args.eggroll_rank,
        freeze_nonlora=False,
    )

    gradient_optimizer = optax.adam(args.gradient_lr)
    gradient_opt_state = gradient_optimizer.init(gradient_params)

    def eggroll_member(noiser_params_, params_, iterinfo_, spectrum_):
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

    batched_eggroll = jax.jit(jax.vmap(eggroll_member, in_axes=(None, None, 0, 0)))
    eggroll_update = jax.jit(
        lambda n, p, f, i: noiser.do_updates(frozen_noiser_params, n, p, es_tree_key, f, i, init.es_map)
    )

    @jax.jit
    def eggroll_step(noiser_params_, params_, spectra_, iterinfos_):
        logits = batched_eggroll(noiser_params_, params_, iterinfos_, spectra_)
        designs = hard_threshold_design(logits)
        pred = toy_forward(designs, surrogate)
        raw_scores = -jnp.mean((pred - spectra_) ** 2, axis=-1)
        fitnesses = noiser.convert_fitnesses(frozen_noiser_params, noiser_params_, raw_scores)
        return raw_scores, designs, fitnesses

    def gradient_loss(params_, spectra_):
        logits = pure_generator_forward(params_, frozen_params, spectra_)
        designs = st_design(logits, args.st_slope)
        pred = toy_forward(designs, surrogate)
        return jnp.mean((pred - spectra_) ** 2)

    gradient_step = jax.jit(jax.value_and_grad(gradient_loss))
    eval_step = jax.jit(evaluate_hard)

    history = History([], [], [], [])
    data_key = jax.random.fold_in(key, 999)
    for step in range(args.steps + 1):
        egg_mse, egg_designs = eval_step(eggroll_params, frozen_params, val_spectra, surrogate)
        grad_mse, grad_designs = eval_step(gradient_params, frozen_params, val_spectra, surrogate)
        history.eggroll_val_mse.append(float(jax.device_get(egg_mse)))
        history.gradient_val_mse.append(float(jax.device_get(grad_mse)))
        history.eggroll_uniqueness.append(float(jax.device_get(mask_uniqueness(egg_designs))))
        history.gradient_uniqueness.append(float(jax.device_get(mask_uniqueness(grad_designs))))

        if step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(
                f"step={step:04d} "
                f"eggroll_mse={history.eggroll_val_mse[-1]:.6f} "
                f"gradient_mse={history.gradient_val_mse[-1]:.6f} "
                f"egg_unique={history.eggroll_uniqueness[-1]:.3f} "
                f"grad_unique={history.gradient_uniqueness[-1]:.3f} "
                f"egg_fill={float(jax.device_get(jnp.mean(metal_fill_ratio(egg_designs)))):.3f} "
                f"grad_fill={float(jax.device_get(jnp.mean(metal_fill_ratio(grad_designs)))):.3f}"
            )

        if step == args.steps:
            break

        data_key, egg_key, grad_key = jax.random.split(data_key, 3)
        pair_count = args.population_size // 2
        pair_indices = jax.random.randint(egg_key, (pair_count,), 0, train_spectra.shape[0])
        egg_batch = jnp.repeat(train_spectra[pair_indices], 2, axis=0)
        iterinfos = (jnp.full((args.population_size,), step, dtype=jnp.int32), jnp.arange(args.population_size))
        _raw_scores, _designs, fitnesses = eggroll_step(noiser_params, eggroll_params, egg_batch, iterinfos)
        noiser_params, eggroll_params = eggroll_update(noiser_params, eggroll_params, fitnesses, iterinfos)

        grad_indices = jax.random.randint(grad_key, (args.population_size,), 0, train_spectra.shape[0])
        grad_batch = train_spectra[grad_indices]
        loss, grads = gradient_step(gradient_params, grad_batch)
        updates, gradient_opt_state = gradient_optimizer.update(grads, gradient_opt_state, gradient_params)
        gradient_params = optax.apply_updates(gradient_params, updates)

    steps = list(range(args.steps + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, history.eggroll_val_mse, label="EGGROLL hard binary")
    ax.plot(steps, history.gradient_val_mse, label="Gradient ST")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation hard-mask MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mse_comparison.png", dpi=160)
    plt.close(fig)

    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"steps={args.steps}",
                f"population_size={args.population_size}",
                f"hidden_dims={hidden_dims}",
                f"eggroll_initial_mse={history.eggroll_val_mse[0]:.8f}",
                f"eggroll_final_mse={history.eggroll_val_mse[-1]:.8f}",
                f"gradient_initial_mse={history.gradient_val_mse[0]:.8f}",
                f"gradient_final_mse={history.gradient_val_mse[-1]:.8f}",
                f"eggroll_final_uniqueness={history.eggroll_uniqueness[-1]:.8f}",
                f"gradient_final_uniqueness={history.gradient_uniqueness[-1]:.8f}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
