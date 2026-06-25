"""Real-data random-init comparison: EGGROLL hard binary vs ST gradient.

This compares the two inverse-training mechanisms against the downloaded
antenna dataset and frozen forward surrogate:

    target S11 -> random generator -> design -> Forward_model_for_tandem -> MSE

EGGROLL evaluates hard binary masks during training. The gradient baseline
trains through the paper's smooth tanh threshold and is evaluated with the same
hard thresholding used by EGGROLL.
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
from pixelant_eggroll.checkpoints import forward_surrogate_from_torch_checkpoint
from pixelant_eggroll.config import DEFAULT_LAYOUT, parse_hidden_dims
from pixelant_eggroll.data import load_spectra_mat
from pixelant_eggroll.models_jax import InverseGenerator, bn_eval, forward_surrogate, leaky_relu, linear_eval


@dataclass
class History:
    eggroll_val_mse: list[float]
    gradient_val_mse: list[float]
    eggroll_train_score: list[float]
    gradient_train_loss: list[float]
    eggroll_uniqueness: list[float]
    gradient_uniqueness: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-mat", default="antenna_dataset.mat")
    parser.add_argument("--forward-checkpoint", default="Forward_model_for_tandem.pth")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--hidden-dims", default="128,128")
    parser.add_argument("--eggroll-rank", type=int, default=1)
    parser.add_argument("--eggroll-sigma", type=float, default=0.2)
    parser.add_argument("--eggroll-lr", type=float, default=0.01)
    parser.add_argument("--gradient-lr", type=float, default=3e-4)
    parser.add_argument("--st-slope", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="eggroll_runs/real_random_init_compare")
    return parser.parse_args()


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
    soft = soft.reshape((-1, *DEFAULT_LAYOUT.mask_shape))
    for pixel in DEFAULT_LAYOUT.feed_pixels:
        row, col = pixel
        soft = soft.at[:, 0, row, col].set(1.0)
    return soft


def evaluate_hard(params, frozen_params, surrogate_params, spectra: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    logits = pure_generator_forward(params, frozen_params, spectra)
    designs = hard_threshold_design(logits)
    pred = forward_surrogate(surrogate_params, designs)
    return jnp.mean((pred - spectra) ** 2), designs


def save_plot(output_dir: Path, history: History) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = list(range(len(history.eggroll_val_mse)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, history.eggroll_val_mse, label="EGGROLL hard binary")
    ax.plot(steps, history.gradient_val_mse, label="Gradient ST")
    ax.set_xlabel("Step")
    ax.set_ylabel("Real surrogate validation MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mse_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.population_size % 2 != 0:
        raise SystemExit("--population-size must be even for antithetic EGGROLL pairs")

    import hyperscalees as hs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(args.seed)
    model_key, es_key, sample_key = jax.random.split(key, 3)
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    spectra_all = load_spectra_mat(args.spectra_mat)
    surrogate_params = forward_surrogate_from_torch_checkpoint(args.forward_checkpoint)
    val_spectra = spectra_all[: args.val_size]
    train_spectra = spectra_all[args.val_size :]

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
        pred = forward_surrogate(surrogate_params, designs)
        raw_scores = -jnp.mean((pred - spectra_) ** 2, axis=-1)
        fitnesses = noiser.convert_fitnesses(frozen_noiser_params, noiser_params_, raw_scores)
        return raw_scores, designs, fitnesses

    def gradient_loss(params_, spectra_):
        logits = pure_generator_forward(params_, frozen_params, spectra_)
        designs = st_design(logits, args.st_slope)
        pred = forward_surrogate(surrogate_params, designs)
        return jnp.mean((pred - spectra_) ** 2)

    gradient_step = jax.jit(jax.value_and_grad(gradient_loss))
    eval_step = jax.jit(evaluate_hard)

    history = History([], [], [], [], [], [])
    data_key = sample_key
    for step in range(args.steps + 1):
        egg_mse, egg_designs = eval_step(eggroll_params, frozen_params, surrogate_params, val_spectra)
        grad_mse, grad_designs = eval_step(gradient_params, frozen_params, surrogate_params, val_spectra)
        history.eggroll_val_mse.append(float(jax.device_get(egg_mse)))
        history.gradient_val_mse.append(float(jax.device_get(grad_mse)))
        history.eggroll_uniqueness.append(float(jax.device_get(mask_uniqueness(egg_designs))))
        history.gradient_uniqueness.append(float(jax.device_get(mask_uniqueness(grad_designs))))

        print(
            f"step={step:04d} "
            f"eggroll_val_mse={history.eggroll_val_mse[-1]:.6f} "
            f"gradient_val_mse={history.gradient_val_mse[-1]:.6f} "
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
        raw_scores, _designs, fitnesses = eggroll_step(noiser_params, eggroll_params, egg_batch, iterinfos)
        noiser_params, eggroll_params = eggroll_update(noiser_params, eggroll_params, fitnesses, iterinfos)
        history.eggroll_train_score.append(float(jax.device_get(jnp.mean(raw_scores))))

        grad_indices = jax.random.randint(grad_key, (args.population_size,), 0, train_spectra.shape[0])
        grad_batch = train_spectra[grad_indices]
        loss, grads = gradient_step(gradient_params, grad_batch)
        updates, gradient_opt_state = gradient_optimizer.update(grads, gradient_opt_state, gradient_params)
        gradient_params = optax.apply_updates(gradient_params, updates)
        history.gradient_train_loss.append(float(jax.device_get(loss)))

    save_plot(output_dir, history)
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
