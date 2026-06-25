"""Compare direct ES over mask logits against the pretrained inverse NN."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from pixelant_eggroll.antenna_ops import hard_threshold_design, metal_fill_ratio
from pixelant_eggroll.checkpoints import forward_surrogate_from_torch_checkpoint, inverse_from_torch_checkpoint
from pixelant_eggroll.config import DEFAULT_LAYOUT
from pixelant_eggroll.data import load_spectra_mat
from pixelant_eggroll.models_jax import bn_eval, forward_surrogate, leaky_relu, linear_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-mat", default="antenna_dataset.mat")
    parser.add_argument("--forward-checkpoint", default="Forward_model_for_tandem.pth")
    parser.add_argument("--inverse-checkpoint", default="inverse_tandem_model.pth")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--output-dir", default="eggroll_runs/direct_design_es_vs_nn")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-on-beat", action="store_true")
    return parser.parse_args()


def pure_inverse_forward(params, frozen_params, spectra: jnp.ndarray) -> jnp.ndarray:
    x = spectra
    for idx in range(len(frozen_params["hidden_dims"])):
        x = linear_eval(x, params[f"fc{idx}"])
        bn = {**frozen_params[f"bn{idx}"], **params[f"bn{idx}"]}
        x = leaky_relu(bn_eval(x, bn))
    return linear_eval(x, params["out"])


def save_design(path: Path, design: jnp.ndarray, target: jnp.ndarray, prediction: jnp.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].imshow(jax.device_get(design[0, 0]), cmap="gray_r", interpolation="nearest")
    axes[0].set_title(title)
    axes[0].axis("off")
    axes[1].plot(jax.device_get(target[0]), label="target")
    axes[1].plot(jax.device_get(prediction[0]), label="surrogate")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.population_size % 2 != 0:
        raise SystemExit("--population-size must be even for antithetic EGGROLL pairs")

    import hyperscalees as hs
    from hyperscalees.models.common import PARAM

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra = load_spectra_mat(args.spectra_mat)
    target = spectra[args.target_index : args.target_index + 1]
    surrogate_params = forward_surrogate_from_torch_checkpoint(args.forward_checkpoint)
    inv_frozen, inv_params, _scan, _es = inverse_from_torch_checkpoint(args.inverse_checkpoint)

    @jax.jit
    def nn_baseline(target_):
        logits = pure_inverse_forward(inv_params, inv_frozen, target_)
        design = hard_threshold_design(logits)
        prediction = forward_surrogate(surrogate_params, design)
        mse = jnp.mean((prediction - target_) ** 2)
        return mse, design, prediction

    baseline_mse, baseline_design, baseline_prediction = nn_baseline(target)
    baseline_mse = float(jax.device_get(baseline_mse))
    save_design(output_dir / "nn_baseline.png", baseline_design, target, baseline_prediction, "NN baseline")

    noiser = hs.noiser.eggroll.EggRoll
    key = jax.random.key(args.seed)
    param_key, es_key = jax.random.split(key)
    params = {"logits": 0.01 * jax.random.normal(param_key, (DEFAULT_LAYOUT.flat_size,), dtype=jnp.float32)}
    scan_map = {"logits": ()}
    es_map = {"logits": PARAM}
    es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser_params, noiser_params = noiser.init_noiser(
        params,
        args.sigma,
        args.lr,
        solver=optax.adam,
        solver_kwargs={},
        rank=args.rank,
        use_batched_update=True,
    )

    @jax.jit
    def score_population(noiser_params_, params_, step_):
        iterinfos = (jnp.full((args.population_size,), step_, dtype=jnp.int32), jnp.arange(args.population_size))
        logits = jax.vmap(
            lambda ii: noiser.get_noisy_standard(
                frozen_noiser_params,
                noiser_params_,
                params_["logits"],
                es_tree_key["logits"],
                ii,
            )
        )(iterinfos)
        designs = hard_threshold_design(logits)
        predictions = forward_surrogate(surrogate_params, designs)
        mse = jnp.mean((predictions - target) ** 2, axis=-1)
        raw_fitness = -mse
        fitnesses = noiser.convert_fitnesses(frozen_noiser_params, noiser_params_, raw_fitness)
        return raw_fitness, mse, designs, predictions, fitnesses, iterinfos

    update = jax.jit(lambda n, p, f, i: noiser.do_updates(frozen_noiser_params, n, p, es_tree_key, f, i, es_map))

    metrics_path = output_dir / "metrics.csv"
    start = time.perf_counter()
    first_beat_step = None
    first_beat_seconds = None
    best_mse = float("inf")
    best_step = None

    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "step",
                "elapsed_seconds",
                "baseline_nn_mse",
                "best_population_mse",
                "best_so_far_mse",
                "beats_baseline",
                "metal_fill_ratio",
            ],
        )
        writer.writeheader()

        for step in range(args.steps + 1):
            raw_fitness, mse, designs, predictions, fitnesses, iterinfos = score_population(noiser_params, params, step)
            best_idx = int(jax.device_get(jnp.argmin(mse)))
            step_best_mse = float(jax.device_get(mse[best_idx]))
            step_fill = float(jax.device_get(metal_fill_ratio(designs[best_idx : best_idx + 1])[0]))
            elapsed = time.perf_counter() - start

            if step_best_mse < best_mse:
                best_mse = step_best_mse
                best_step = step
                jnp.savez(
                    output_dir / "final_logits.npz",
                    logits=jax.device_get(designs[best_idx].reshape(-1) * 2.0 - 1.0),
                    step=jnp.asarray(step),
                    mse=jnp.asarray(step_best_mse),
                )
                save_design(
                    output_dir / "best_design.png",
                    designs[best_idx : best_idx + 1],
                    target,
                    predictions[best_idx : best_idx + 1],
                    f"Direct ES best step {step}",
                )

            beats = step_best_mse < baseline_mse
            if beats and first_beat_step is None:
                first_beat_step = step
                first_beat_seconds = elapsed

            writer.writerow(
                {
                    "step": step,
                    "elapsed_seconds": elapsed,
                    "baseline_nn_mse": baseline_mse,
                    "best_population_mse": step_best_mse,
                    "best_so_far_mse": best_mse,
                    "beats_baseline": beats,
                    "metal_fill_ratio": step_fill,
                }
            )
            fh.flush()
            print(
                f"step={step:05d} elapsed={elapsed:.2f}s "
                f"pop_best_mse={step_best_mse:.6f} best={best_mse:.6f} "
                f"baseline={baseline_mse:.6f} beats={beats} fill={step_fill:.3f}"
            )

            if beats and args.stop_on_beat:
                break
            if step == args.steps:
                break

            noiser_params, params = update(noiser_params, params, fitnesses, iterinfos)

    summary = "\n".join(
        [
            f"target_index={args.target_index}",
            f"baseline_nn_mse={baseline_mse:.8f}",
            f"first_beat_step={first_beat_step}",
            f"first_beat_seconds={first_beat_seconds}",
            f"best_step={best_step}",
            f"best_mse={best_mse:.8f}",
            f"population_size={args.population_size}",
            f"sigma={args.sigma}",
            f"lr={args.lr}",
            f"rank={args.rank}",
        ]
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
