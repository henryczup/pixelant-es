"""Scale hard-discrete EGGROLL vs smooth-threshold gradient training.

The downloaded antenna surrogate is fixed to 12x12 masks, so it cannot answer
the design-space scaling question directly. This benchmark uses a frozen,
layout-scalable random-feature surrogate and sweeps mask size plus generator
capacity:

    target spectrum -> generator -> design -> frozen scalable surrogate -> MSE

EGGROLL trains and evaluates hard masks. The gradient baseline trains through a
smooth tanh threshold and is evaluated with hard masks, matching the practical
question of whether hard-discrete training can replace ST gradients as masks and
generators grow.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from pixelant_eggroll.antenna_ops import force_feed_pixels, hard_threshold_design, mask_uniqueness, metal_fill_ratio
from pixelant_eggroll.config import LayoutSpec, parse_layout
from pixelant_eggroll.models_jax import InverseGenerator, bn_eval, leaky_relu, linear_eval


@dataclass(frozen=True)
class RunConfig:
    layout: LayoutSpec
    hidden_dims: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", default="12x12,24x24,32x32")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width-factor", type=float, default=8.0)
    parser.add_argument("--min-width", type=int, default=64)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=128)
    parser.add_argument("--dataset-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--spectrum-dim", type=int, default=81)
    parser.add_argument("--surrogate-features", type=int, default=256)
    parser.add_argument("--eggroll-rank", type=int, default=1)
    parser.add_argument("--eggroll-sigma", type=float, default=0.2)
    parser.add_argument("--eggroll-lr", type=float, default=0.02)
    parser.add_argument("--gradient-lr", type=float, default=0.003)
    parser.add_argument("--st-slope", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="eggroll_runs/scaling_hard_vs_st")
    return parser.parse_args()


def layout_with_center_feed(value: str) -> LayoutSpec:
    parsed = parse_layout(value)
    row0 = max(0, parsed.height // 2 - 1)
    row1 = min(parsed.height - 1, parsed.height // 2)
    return LayoutSpec(height=parsed.height, width=parsed.width, feed_pixels=((row0, 0), (row1, 0)))


def hidden_dims_for(layout: LayoutSpec, depth: int, width_factor: float, min_width: int, max_width: int) -> tuple[int, ...]:
    width = int(width_factor * jnp.sqrt(float(layout.flat_size)))
    width = max(min_width, min(max_width, width))
    return tuple(width for _ in range(depth))


def make_surrogate(key, flat_size: int, features: int, spectrum_dim: int) -> dict[str, jnp.ndarray]:
    k1, k2, kb1, kb2 = jax.random.split(key, 4)
    return {
        "w1": jax.random.normal(k1, (flat_size, features), dtype=jnp.float32) / jnp.sqrt(float(flat_size)),
        "b1": 0.05 * jax.random.normal(kb1, (features,), dtype=jnp.float32),
        "w2": jax.random.normal(k2, (features, spectrum_dim), dtype=jnp.float32) / jnp.sqrt(float(features)),
        "b2": 0.05 * jax.random.normal(kb2, (spectrum_dim,), dtype=jnp.float32),
    }


def scalable_surrogate(designs: jnp.ndarray, surrogate: dict[str, jnp.ndarray]) -> jnp.ndarray:
    flat = designs.reshape((designs.shape[0], -1))
    centered = flat - 0.5
    hidden = jax.nn.gelu(centered @ surrogate["w1"] + surrogate["b1"])
    return 8.0 * jnp.tanh(hidden @ surrogate["w2"] + surrogate["b2"])


def make_dataset(key, n: int, layout: LayoutSpec, surrogate: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    masks = jax.random.bernoulli(key, p=0.45, shape=(n, *layout.mask_shape)).astype(jnp.float32)
    masks = force_feed_pixels(masks, layout=layout)
    spectra = scalable_surrogate(masks, surrogate)
    return spectra, masks


def pure_generator_forward(params, frozen_params, spectrum: jnp.ndarray) -> jnp.ndarray:
    x = spectrum
    for idx in range(len(frozen_params["hidden_dims"])):
        x = linear_eval(x, params[f"fc{idx}"])
        bn = {**frozen_params[f"bn{idx}"], **params[f"bn{idx}"]}
        x = leaky_relu(bn_eval(x, bn))
    return linear_eval(x, params["out"])


def st_design(logits: jnp.ndarray, slope: float, layout: LayoutSpec) -> jnp.ndarray:
    soft = 0.5 + 0.5 * jnp.tanh(slope * logits)
    soft = soft.reshape((-1, *layout.mask_shape))
    return force_feed_pixels(soft, layout=layout)


def run_one(args: argparse.Namespace, config: RunConfig, seed: int, output_dir: Path) -> dict[str, float | int | str]:
    import hyperscalees as hs

    key = jax.random.key(seed)
    model_key, es_key, surrogate_key, train_key, val_key, loop_key = jax.random.split(key, 6)
    layout = config.layout
    surrogate = make_surrogate(surrogate_key, layout.flat_size, args.surrogate_features, args.spectrum_dim)
    train_spectra, _ = make_dataset(train_key, args.dataset_size, layout, surrogate)
    val_spectra, _ = make_dataset(val_key, args.val_size, layout, surrogate)

    init = InverseGenerator.rand_init(
        model_key,
        hidden_dims=config.hidden_dims,
        input_dim=args.spectrum_dim,
        output_dim=layout.flat_size,
    )
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
        designs = hard_threshold_design(logits, layout=layout)
        pred = scalable_surrogate(designs, surrogate)
        raw_scores = -jnp.mean((pred - spectra_) ** 2, axis=-1)
        fitnesses = noiser.convert_fitnesses(frozen_noiser_params, noiser_params_, raw_scores)
        return raw_scores, designs, fitnesses

    def gradient_loss(params_, spectra_):
        logits = pure_generator_forward(params_, frozen_params, spectra_)
        designs = st_design(logits, args.st_slope, layout)
        pred = scalable_surrogate(designs, surrogate)
        return jnp.mean((pred - spectra_) ** 2)

    gradient_step = jax.jit(jax.value_and_grad(gradient_loss))

    @jax.jit
    def eval_hard(params_, spectra_):
        logits = pure_generator_forward(params_, frozen_params, spectra_)
        designs = hard_threshold_design(logits, layout=layout)
        pred = scalable_surrogate(designs, surrogate)
        return jnp.mean((pred - spectra_) ** 2), designs

    egg_mse_curve: list[float] = []
    grad_mse_curve: list[float] = []
    start = time.perf_counter()
    data_key = loop_key
    for step in range(args.steps + 1):
        egg_mse, egg_designs = eval_hard(eggroll_params, val_spectra)
        grad_mse, grad_designs = eval_hard(gradient_params, val_spectra)
        egg_mse_curve.append(float(jax.device_get(egg_mse)))
        grad_mse_curve.append(float(jax.device_get(grad_mse)))

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
        _loss, grads = gradient_step(gradient_params, grad_batch)
        updates, gradient_opt_state = gradient_optimizer.update(grads, gradient_opt_state, gradient_params)
        gradient_params = optax.apply_updates(gradient_params, updates)

    elapsed = time.perf_counter() - start
    final_egg_mse, final_egg_designs = eval_hard(eggroll_params, val_spectra)
    final_grad_mse, final_grad_designs = eval_hard(gradient_params, val_spectra)
    result = {
        "layout": f"{layout.height}x{layout.width}",
        "flat_size": layout.flat_size,
        "hidden_dims": "-".join(str(dim) for dim in config.hidden_dims),
        "param_count": int(sum(x.size for x in jax.tree.leaves(init.params))),
        "steps": args.steps,
        "population_size": args.population_size,
        "eggroll_initial_mse": egg_mse_curve[0],
        "eggroll_final_mse": float(jax.device_get(final_egg_mse)),
        "eggroll_best_mse": min(egg_mse_curve),
        "gradient_initial_mse": grad_mse_curve[0],
        "gradient_final_mse": float(jax.device_get(final_grad_mse)),
        "gradient_best_mse": min(grad_mse_curve),
        "eggroll_final_uniqueness": float(jax.device_get(mask_uniqueness(final_egg_designs))),
        "gradient_final_uniqueness": float(jax.device_get(mask_uniqueness(final_grad_designs))),
        "eggroll_fill": float(jax.device_get(jnp.mean(metal_fill_ratio(final_egg_designs, layout=layout)))),
        "gradient_fill": float(jax.device_get(jnp.mean(metal_fill_ratio(final_grad_designs, layout=layout)))),
        "elapsed_seconds": elapsed,
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(args.steps + 1), egg_mse_curve, label="EGGROLL hard binary")
    ax.plot(range(args.steps + 1), grad_mse_curve, label="Gradient ST")
    ax.set_xlabel("Step")
    ax.set_ylabel("Hard-mask validation MSE")
    ax.set_title(f"{result['layout']} params={result['param_count']:,}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"curve_{layout.height}x{layout.width}.png", dpi=160)
    plt.close(fig)
    return result


def main() -> None:
    args = parse_args()
    if args.population_size % 2 != 0:
        raise SystemExit("--population-size must be even for antithetic EGGROLL pairs")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layouts = [layout_with_center_feed(item.strip()) for item in args.layouts.split(",") if item.strip()]
    configs = [
        RunConfig(
            layout=layout,
            hidden_dims=hidden_dims_for(layout, args.depth, args.width_factor, args.min_width, args.max_width),
        )
        for layout in layouts
    ]

    rows = []
    for index, config in enumerate(configs):
        print(
            f"running layout={config.layout.height}x{config.layout.width} "
            f"hidden={config.hidden_dims} pop={args.population_size} steps={args.steps}"
        )
        row = run_one(args, config, seed=args.seed + index * 1000, output_dir=output_dir)
        rows.append(row)
        print(
            f"  eggroll {row['eggroll_initial_mse']:.6f}->{row['eggroll_final_mse']:.6f} "
            f"(best {row['eggroll_best_mse']:.6f}); "
            f"gradient {row['gradient_initial_mse']:.6f}->{row['gradient_final_mse']:.6f} "
            f"(best {row['gradient_best_mse']:.6f}); "
            f"params={row['param_count']:,} elapsed={row['elapsed_seconds']:.1f}s"
        )

    csv_path = output_dir / "scaling_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
