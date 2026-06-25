"""Train a hard-binary inverse antenna generator with HyperscaleES EGGROLL."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from pixelant_eggroll.antenna_ops import hard_threshold_design, mask_uniqueness, metal_fill_ratio
from pixelant_eggroll.checkpoints import inverse_from_torch_checkpoint, save_inverse_npz
from pixelant_eggroll.config import (
    DEFAULT_LAYOUT,
    GeneratorConfig,
    ScorerConfig,
    parse_hidden_dims,
    parse_layout,
)
from pixelant_eggroll.data import load_spectra_mat, sample_antithetic_spectra
from pixelant_eggroll.fitness import FitnessConfig, compute_fitness
from pixelant_eggroll.models_jax import DirectMLPGenerator, generator_class
from pixelant_eggroll.scorers import build_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-mat", required=True, help="Path to `.mat` file containing target spectra.")
    parser.add_argument("--forward-checkpoint", help="Frozen PyTorch Net_big `.pth` checkpoint for surrogate scoring.")
    parser.add_argument("--inverse-checkpoint", help="Optional ST-trained PyTorch Net_inverse `.pth` warm start.")
    parser.add_argument("--output-dir", default="eggroll_runs/run", help="Directory for logs and checkpoints.")
    parser.add_argument("--layout", default="12x12", help="Mask size as `HxW`, e.g. `12x12`, `32x32`, or `64`.")
    parser.add_argument("--layers", type=int, default=1, help="Number of mask layers.")
    parser.add_argument("--generator", choices=("mlp", "cnn"), help="Generator family. Defaults to MLP for 12x12, CNN otherwise.")
    parser.add_argument("--latent-dim", type=int, default=None, help="Latent noise dimension. Defaults to 0 for MLP, 16 for CNN.")
    parser.add_argument("--scorer", choices=("surrogate", "external-em"), default="surrogate")
    parser.add_argument("--solver-mode", choices=("air", "substrate"), default="air", help="External EM solver mode.")
    parser.add_argument("--scorer-command", help="External scorer command. Supports {input_mat} and {output_mat} placeholders.")
    parser.add_argument("--scorer-work-dir", help="External EM scorer working directory.")
    parser.add_argument("--scorer-cache-dir", help="External EM scorer cache directory.")
    parser.add_argument("--scorer-timeout", type=float, help="External EM scorer timeout in seconds.")
    parser.add_argument("--bad-spectrum-value", type=float, default=1.0e6, help="Spectrum fill value for failed EM solves.")
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma-decay", type=float, default=1.0)
    parser.add_argument("--noise-reuse", type=int, default=0)
    parser.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    parser.add_argument("--hidden-dims", default="1054,512", help="MLP hidden dims or CNN channel seed, e.g. `1054,512`.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-conn", type=float, default=0.0)
    parser.add_argument("--lambda-area", type=float, default=0.0)
    parser.add_argument("--lambda-frag", type=float, default=0.0)
    parser.add_argument("--lambda-rule", type=float, default=0.0)
    parser.add_argument("--target-fill", type=float, default=0.50)
    parser.add_argument("--freeze-nonlora", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    return parser.parse_args()


def optimizer_factory(name: str):
    return {"sgd": optax.sgd, "adam": optax.adam, "adamw": optax.adamw}[name]


def save_best_design(path: Path, design: jnp.ndarray, target: jnp.ndarray, prediction: jnp.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].imshow(jax.device_get(design[0, 0]), cmap="gray_r", interpolation="nearest")
    axes[0].set_title("Best binary mask")
    axes[0].axis("off")
    axes[1].plot(jax.device_get(target[0]), label="target")
    axes[1].plot(jax.device_get(prediction[0]), label="scorer")
    axes[1].set_title("S11")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _resolve_generator_config(args: argparse.Namespace, spectrum_dim: int) -> GeneratorConfig:
    layout = parse_layout(args.layout, layers=args.layers)
    kind = args.generator or ("mlp" if layout == DEFAULT_LAYOUT else "cnn")
    latent_dim = args.latent_dim if args.latent_dim is not None else (16 if kind == "cnn" else 0)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    return GeneratorConfig(kind=kind, layout=layout, hidden_dims=hidden_dims, latent_dim=latent_dim, spectrum_dim=spectrum_dim)


def _sample_paired_latents(key, population_size: int, latent_dim: int) -> jnp.ndarray:
    if latent_dim == 0:
        return jnp.zeros((population_size, 0), dtype=jnp.float32)
    pair_count = population_size // 2
    paired = jax.random.normal(key, (pair_count, latent_dim), dtype=jnp.float32)
    return jnp.repeat(paired, 2, axis=0)


def _validate_args(args: argparse.Namespace, generator_config: GeneratorConfig) -> None:
    if args.scorer == "surrogate" and not args.forward_checkpoint:
        raise SystemExit("--scorer surrogate requires --forward-checkpoint")
    if args.scorer == "external-em":
        if generator_config.layout != DEFAULT_LAYOUT:
            raise SystemExit("--scorer external-em currently supports only --layout 12x12")
    if args.inverse_checkpoint:
        if generator_config.kind != "mlp" or generator_config.layout != DEFAULT_LAYOUT or generator_config.latent_dim != 0:
            raise SystemExit("--inverse-checkpoint is only compatible with --generator mlp --layout 12x12 --latent-dim 0")


def _scorer_metadata(config: ScorerConfig) -> dict[str, str | None]:
    return {
        "kind": config.kind,
        "checkpoint_path": str(config.checkpoint_path) if config.checkpoint_path else None,
        "command": config.command,
        "work_dir": str(config.work_dir) if config.work_dir else None,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import hyperscalees as hs
    except ImportError as exc:
        raise SystemExit("Install EGGROLL dependencies first: `pip install -r requirements-eggroll.txt`.") from exc

    noiser_cls = hs.noiser.eggroll.EggRoll
    key = jax.random.key(args.seed)
    model_key, es_key, data_key = jax.random.split(key, 3)
    spectra = load_spectra_mat(args.spectra_mat)
    generator_config = _resolve_generator_config(args, spectrum_dim=int(spectra.shape[-1]))
    _validate_args(args, generator_config)

    scorer_config = ScorerConfig(
        kind=args.scorer,
        checkpoint_path=Path(args.forward_checkpoint) if args.forward_checkpoint else None,
        command=args.scorer_command,
        work_dir=Path(args.scorer_work_dir) if args.scorer_work_dir else None,
        solver_mode=args.solver_mode,
        cache_dir=Path(args.scorer_cache_dir) if args.scorer_cache_dir else None,
        timeout_seconds=args.scorer_timeout,
        bad_spectrum_value=args.bad_spectrum_value,
    )
    scorer = build_scorer(scorer_config)

    if args.inverse_checkpoint:
        frozen_params, params, scan_map, es_map = inverse_from_torch_checkpoint(args.inverse_checkpoint)
        generator = DirectMLPGenerator
        init_mode = "warm_start"
        sigma = 0.02 if args.sigma is None else args.sigma
    else:
        generator = generator_class(generator_config.kind)
        init = generator.rand_init(model_key, generator_config)
        frozen_params, params, scan_map, es_map = init.frozen_params, init.params, init.scan_map, init.es_map
        init_mode = f"random_{generator_config.kind}"
        sigma = 0.20 if args.sigma is None else args.sigma

    es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser_params, noiser_params = noiser_cls.init_noiser(
        params,
        sigma,
        args.lr,
        solver=optimizer_factory(args.optimizer),
        solver_kwargs={},
        rank=args.rank,
        noise_reuse=args.noise_reuse,
        freeze_nonlora=args.freeze_nonlora,
        use_batched_update=True,
    )
    fitness_config = FitnessConfig(
        lambda_conn=args.lambda_conn,
        lambda_area=args.lambda_area,
        lambda_frag=args.lambda_frag,
        lambda_rule=args.lambda_rule,
        target_fill=args.target_fill,
    )

    def generator_apply(noiser_params_, params_, iterinfo_, spectrum_, latent_):
        return generator.forward(
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
    def generate_designs(noiser_params_, params_, iterinfos_, targets_, latents_):
        logits = batched_generator(noiser_params_, params_, iterinfos_, targets_, latents_)
        return hard_threshold_design(logits, layout=generator_config.layout)

    if args.scorer == "surrogate":
        @jax.jit
        def score_population(noiser_params_, params_, iterinfos_, targets_, latents_):
            designs = generate_designs(noiser_params_, params_, iterinfos_, targets_, latents_)
            predictions = scorer.score(designs)
            raw_fitness, metrics = compute_fitness(
                predictions,
                targets_,
                designs,
                fitness_config,
                layout=generator_config.layout,
            )
            return raw_fitness, metrics, designs, predictions
    else:
        def score_population(noiser_params_, params_, iterinfos_, targets_, latents_):
            designs = generate_designs(noiser_params_, params_, iterinfos_, targets_, latents_)
            predictions = scorer.score(designs)
            raw_fitness, metrics = compute_fitness(
                predictions,
                targets_,
                designs,
                fitness_config,
                layout=generator_config.layout,
            )
            return raw_fitness, metrics, designs, predictions

    log_path = output_dir / "metrics.csv"
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "step",
                "sigma",
                "fitness_mean",
                "fitness_best",
                "spectrum_mse_mean",
                "connectivity_penalty_mean",
                "area_penalty_mean",
                "fragmentation_penalty_mean",
                "fabrication_rule_penalty_mean",
                "mask_uniqueness",
                "metal_fill_ratio",
            ],
        )
        writer.writeheader()

        best_fitness = -jnp.inf
        for step in range(args.steps):
            data_key, target_key, latent_key = jax.random.split(data_key, 3)
            targets = sample_antithetic_spectra(target_key, spectra, args.population_size)
            latents = _sample_paired_latents(latent_key, args.population_size, generator_config.latent_dim)
            iterinfos = (jnp.full((args.population_size,), step, dtype=jnp.int32), jnp.arange(args.population_size))
            raw_fitness, metrics, designs, predictions = score_population(noiser_params, params, iterinfos, targets, latents)
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
                "spectrum_mse_mean": float(jax.device_get(jnp.mean(metrics["spectrum_mse"]))),
                "connectivity_penalty_mean": float(jax.device_get(jnp.mean(metrics["connectivity_penalty"]))),
                "area_penalty_mean": float(jax.device_get(jnp.mean(metrics["area_penalty"]))),
                "fragmentation_penalty_mean": float(jax.device_get(jnp.mean(metrics["fragmentation_penalty"]))),
                "fabrication_rule_penalty_mean": float(jax.device_get(jnp.mean(metrics["fabrication_rule_penalty"]))),
                "mask_uniqueness": float(jax.device_get(mask_uniqueness(designs, layout=generator_config.layout))),
                "metal_fill_ratio": float(jax.device_get(jnp.mean(metal_fill_ratio(designs, layout=generator_config.layout)))),
            }
            writer.writerow(row)
            fh.flush()
            print(
                f"step={step:05d} fitness_mean={row['fitness_mean']:.6g} "
                f"fitness_best={row['fitness_best']:.6g} mse={row['spectrum_mse_mean']:.6g} "
                f"unique={row['mask_uniqueness']:.3f} fill={row['metal_fill_ratio']:.3f}"
            )

            if raw_fitness[best_idx] > best_fitness:
                best_fitness = raw_fitness[best_idx]
                save_best_design(
                    output_dir / "best_design.png",
                    designs[best_idx : best_idx + 1],
                    targets[best_idx : best_idx + 1],
                    predictions[best_idx : best_idx + 1],
                )

            if (step + 1) % args.save_every == 0 or step == args.steps - 1:
                save_inverse_npz(
                    output_dir / f"inverse_generator_step_{step + 1}.npz",
                    frozen_params,
                    params,
                    {
                        "step": step + 1,
                        "init_mode": init_mode,
                        "generator_config": asdict(generator_config),
                        "scorer_config": _scorer_metadata(scorer_config),
                        "rank": args.rank,
                        "lr": args.lr,
                        "sigma": float(jax.device_get(noiser_params["sigma"])),
                        "fitness_config": asdict(fitness_config),
                    },
                )


if __name__ == "__main__":
    main()
