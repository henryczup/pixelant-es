"""Compare scratch CNN EGGROLL direct search against the downloaded inverse NN.

The comparison uses the real antenna spectra and frozen forward surrogate. The
pretrained inverse NN produces the baseline hard masks. A scratch CNN generator
is trained with EGGROLL and periodically evaluated on the same validation
spectra; the run reports when the CNN's surrogate MSE beats the NN baseline.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from pixelant_eggroll.antenna_ops import hard_threshold_design, mask_uniqueness, metal_fill_ratio
from pixelant_eggroll.checkpoints import forward_surrogate_from_torch_checkpoint, inverse_from_torch_checkpoint
from pixelant_eggroll.config import GeneratorConfig, LayoutSpec
from pixelant_eggroll.data import load_spectra_mat
from pixelant_eggroll.fitness import FitnessConfig, compute_fitness
from pixelant_eggroll.models_jax import DirectCNNGenerator, bn_eval, forward_surrogate, leaky_relu, linear_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-mat", default="antenna_dataset.mat")
    parser.add_argument("--forward-checkpoint", default="Forward_model_for_tandem.pth")
    parser.add_argument("--inverse-checkpoint", default="inverse_tandem_model.pth")
    parser.add_argument("--output-dir", default="eggroll_runs/cnn_direct_vs_nn")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--val-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--eval-latent-samples", type=int, default=4)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-conn", type=float, default=0.0)
    parser.add_argument("--lambda-area", type=float, default=0.0)
    parser.add_argument("--lambda-frag", type=float, default=0.0)
    parser.add_argument("--lambda-rule", type=float, default=0.0)
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


def sample_paired(key, spectra: jnp.ndarray, population_size: int) -> jnp.ndarray:
    pair_count = population_size // 2
    indices = jax.random.randint(key, (pair_count,), 0, spectra.shape[0])
    return jnp.repeat(spectra[indices], 2, axis=0)


def sample_paired_latents(key, population_size: int, latent_dim: int) -> jnp.ndarray:
    if latent_dim == 0:
        return jnp.zeros((population_size, 0), dtype=jnp.float32)
    pair_count = population_size // 2
    paired = jax.random.normal(key, (pair_count, latent_dim), dtype=jnp.float32)
    return jnp.repeat(paired, 2, axis=0)


def main() -> None:
    args = parse_args()
    if args.population_size % 2 != 0:
        raise SystemExit("--population-size must be even for antithetic EGGROLL pairs")

    import hyperscalees as hs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(args.seed)
    model_key, es_key, data_key, eval_key = jax.random.split(key, 4)
    layout = LayoutSpec(height=12, width=12)
    generator_config = GeneratorConfig(
        kind="cnn",
        layout=layout,
        hidden_dims=(args.channels,),
        latent_dim=args.latent_dim,
        spectrum_dim=81,
    )

    spectra = load_spectra_mat(args.spectra_mat)
    val_spectra = spectra[: args.val_size]
    train_spectra = spectra[args.val_size :]
    surrogate_params = forward_surrogate_from_torch_checkpoint(args.forward_checkpoint)
    baseline_frozen, baseline_params, _baseline_scan, _baseline_es = inverse_from_torch_checkpoint(args.inverse_checkpoint)

    @jax.jit
    def baseline_eval(targets):
        logits = pure_inverse_forward(baseline_params, baseline_frozen, targets)
        designs = hard_threshold_design(logits, layout=layout)
        predictions = forward_surrogate(surrogate_params, designs)
        mse_per_target = jnp.mean((predictions - targets) ** 2, axis=-1)
        return jnp.mean(mse_per_target), designs

    baseline_mse, baseline_designs = baseline_eval(val_spectra)
    baseline_mse = float(jax.device_get(baseline_mse))
    baseline_unique = float(jax.device_get(mask_uniqueness(baseline_designs, layout=layout)))
    baseline_fill = float(jax.device_get(jnp.mean(metal_fill_ratio(baseline_designs, layout=layout))))

    noiser_cls = hs.noiser.eggroll.EggRoll
    init = DirectCNNGenerator.rand_init(model_key, generator_config)
    frozen_params, params, scan_map, es_map = init.frozen_params, init.params, init.scan_map, init.es_map
    es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser_params, noiser_params = noiser_cls.init_noiser(
        params,
        args.sigma,
        args.lr,
        solver=optax.adamw,
        solver_kwargs={},
        rank=args.rank,
        use_batched_update=True,
    )
    fitness_config = FitnessConfig(
        lambda_conn=args.lambda_conn,
        lambda_area=args.lambda_area,
        lambda_frag=args.lambda_frag,
        lambda_rule=args.lambda_rule,
    )

    def generator_apply(noiser_params_, params_, iterinfo_, spectrum_, latent_):
        return DirectCNNGenerator.forward(
            noiser_cls,
            frozen_noiser_params,
            noiser_params_,
            frozen_params,
            params_,
            es_tree_key,
            iterinfo_,
            spectrum_,
            latent_,
        )

    batched_generator = jax.jit(jax.vmap(generator_apply, in_axes=(None, None, 0, 0, 0)))
    update = jax.jit(lambda n, p, f, i: noiser_cls.do_updates(frozen_noiser_params, n, p, es_tree_key, f, i, es_map))

    @jax.jit
    def train_step(noiser_params_, params_, step_, targets_, latents_):
        iterinfos = (jnp.full((targets_.shape[0],), step_, dtype=jnp.int32), jnp.arange(targets_.shape[0]))
        logits = batched_generator(noiser_params_, params_, iterinfos, targets_, latents_)
        designs = hard_threshold_design(logits, layout=layout)
        predictions = forward_surrogate(surrogate_params, designs)
        raw_fitness, metrics = compute_fitness(predictions, targets_, designs, fitness_config, layout=layout)
        fitnesses = noiser_cls.convert_fitnesses(frozen_noiser_params, noiser_params_, raw_fitness)
        return raw_fitness, metrics, designs, fitnesses, iterinfos

    @jax.jit
    def cnn_eval(noiser_params_, params_, targets_, latents_):
        eval_noiser_params = dict(noiser_params_)
        eval_noiser_params["sigma"] = 0.0
        sample_count = latents_.shape[0]
        repeated_targets = jnp.repeat(targets_, sample_count, axis=0)
        tiled_latents = jnp.tile(latents_, (targets_.shape[0], 1))
        iterinfos = (jnp.zeros((repeated_targets.shape[0],), dtype=jnp.int32), jnp.arange(repeated_targets.shape[0]))
        logits = batched_generator(eval_noiser_params, params_, iterinfos, repeated_targets, tiled_latents)
        designs = hard_threshold_design(logits, layout=layout)
        predictions = forward_surrogate(surrogate_params, designs)
        mse = jnp.mean((predictions - repeated_targets) ** 2, axis=-1).reshape((targets_.shape[0], sample_count))
        best_indices = jnp.argmin(mse, axis=1)
        best_mse = mse[jnp.arange(targets_.shape[0]), best_indices]
        best_designs = designs.reshape((targets_.shape[0], sample_count, *layout.mask_shape))[
            jnp.arange(targets_.shape[0]), best_indices
        ]
        return jnp.mean(best_mse), best_designs

    if args.latent_dim == 0:
        eval_latents = jnp.zeros((1, 0), dtype=jnp.float32)
    else:
        eval_key, subkey = jax.random.split(eval_key)
        random_latents = jax.random.normal(
            subkey,
            (max(0, args.eval_latent_samples - 1), args.latent_dim),
            dtype=jnp.float32,
        )
        zero_latent = jnp.zeros((1, args.latent_dim), dtype=jnp.float32)
        eval_latents = jnp.concatenate([zero_latent, random_latents], axis=0)

    metrics_path = output_dir / "metrics.csv"
    start = time.perf_counter()
    best_step: int | None = None
    best_cnn_mse = float("inf")
    last_train_fitness = float("nan")
    last_unique = float("nan")
    last_fill = float("nan")

    print(
        f"baseline_nn_mse={baseline_mse:.6f} "
        f"baseline_unique={baseline_unique:.3f} baseline_fill={baseline_fill:.3f}"
    )

    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "step",
                "elapsed_seconds",
                "baseline_nn_mse",
                "cnn_val_mse",
                "cnn_best_so_far",
                "cnn_beats_baseline",
                "train_fitness_mean",
                "mask_uniqueness",
                "metal_fill_ratio",
            ],
        )
        writer.writeheader()

        for step in range(args.steps + 1):
            should_eval = step == 0 or step == args.steps or step % args.eval_every == 0
            if should_eval:
                cnn_mse, cnn_designs = cnn_eval(noiser_params, params, val_spectra, eval_latents)
                cnn_mse = float(jax.device_get(cnn_mse))
                best_cnn_mse = min(best_cnn_mse, cnn_mse)
                last_unique = float(jax.device_get(mask_uniqueness(cnn_designs, layout=layout)))
                last_fill = float(jax.device_get(jnp.mean(metal_fill_ratio(cnn_designs, layout=layout))))
                elapsed = time.perf_counter() - start
                beats = cnn_mse < baseline_mse
                if beats and best_step is None:
                    best_step = step

                row = {
                    "step": step,
                    "elapsed_seconds": elapsed,
                    "baseline_nn_mse": baseline_mse,
                    "cnn_val_mse": cnn_mse,
                    "cnn_best_so_far": best_cnn_mse,
                    "cnn_beats_baseline": beats,
                    "train_fitness_mean": last_train_fitness,
                    "mask_uniqueness": last_unique,
                    "metal_fill_ratio": last_fill,
                }
                writer.writerow(row)
                fh.flush()
                print(
                    f"step={step:05d} elapsed={elapsed:.1f}s "
                    f"cnn_mse={cnn_mse:.6f} baseline={baseline_mse:.6f} "
                    f"best={best_cnn_mse:.6f} beats={beats} "
                    f"unique={last_unique:.3f} fill={last_fill:.3f}"
                )

                if beats and args.stop_on_beat:
                    break

            if step == args.steps:
                break

            data_key, target_key, latent_key = jax.random.split(data_key, 3)
            targets = sample_paired(target_key, train_spectra, args.population_size)
            latents = sample_paired_latents(latent_key, args.population_size, args.latent_dim)
            raw_fitness, _metrics, designs, fitnesses, iterinfos = train_step(noiser_params, params, step, targets, latents)
            noiser_params, params = update(noiser_params, params, fitnesses, iterinfos)
            last_train_fitness = float(jax.device_get(jnp.mean(raw_fitness)))

    elapsed = time.perf_counter() - start
    summary_lines = [
        f"baseline_nn_mse={baseline_mse:.8f}",
        f"baseline_unique={baseline_unique:.8f}",
        f"baseline_fill={baseline_fill:.8f}",
        f"best_cnn_mse={best_cnn_mse:.8f}",
        f"beat_step={best_step}",
        f"elapsed_seconds={elapsed:.3f}",
        f"steps_run={step}",
        f"population_size={args.population_size}",
        f"val_size={args.val_size}",
        f"channels={args.channels}",
        f"latent_dim={args.latent_dim}",
        f"eval_latent_samples={eval_latents.shape[0]}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
