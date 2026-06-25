"""Benchmark direct antenna design optimizers on frozen surrogate score."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pixelant_eggroll.antenna_ops import hard_threshold_design, metal_fill_ratio
from pixelant_eggroll.checkpoints import forward_surrogate_from_torch_checkpoint, inverse_from_torch_checkpoint
from pixelant_eggroll.config import DEFAULT_LAYOUT
from pixelant_eggroll.data import load_spectra_mat
from pixelant_eggroll.models_jax import bn_eval, forward_surrogate, leaky_relu, linear_eval


@dataclass
class EvalResult:
    mse: np.ndarray
    predictions: np.ndarray
    designs: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-mat", default="antenna_dataset.mat")
    parser.add_argument("--forward-checkpoint", default="Forward_model_for_tandem.pth")
    parser.add_argument("--inverse-checkpoint", default="inverse_tandem_model.pth")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--output-dir", default="eggroll_runs/direct_optimizer_benchmark_target0")
    parser.add_argument("--methods", default="cma,cem,gaussian-es,bpso")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--population-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--cem-elite-frac", type=float, default=0.2)
    parser.add_argument("--cem-alpha", type=float, default=0.7)
    parser.add_argument("--gaussian-lr", type=float, default=0.5)
    parser.add_argument("--bpso-w", type=float, default=0.72)
    parser.add_argument("--bpso-c1", type=float, default=1.49)
    parser.add_argument("--bpso-c2", type=float, default=1.49)
    return parser.parse_args()


def pure_inverse_forward(params, frozen_params, spectra: jnp.ndarray) -> jnp.ndarray:
    x = spectra
    for idx in range(len(frozen_params["hidden_dims"])):
        x = linear_eval(x, params[f"fc{idx}"])
        bn = {**frozen_params[f"bn{idx}"], **params[f"bn{idx}"]}
        x = leaky_relu(bn_eval(x, bn))
    return linear_eval(x, params["out"])


def build_evaluator(surrogate_params, target: jnp.ndarray):
    @jax.jit
    def evaluate_logits(logits: jnp.ndarray):
        designs = hard_threshold_design(logits, layout=DEFAULT_LAYOUT)
        predictions = forward_surrogate(surrogate_params, designs)
        mse = jnp.mean((predictions - target) ** 2, axis=-1)
        return mse, predictions, designs

    def evaluate(logits_np: np.ndarray) -> EvalResult:
        logits = jnp.asarray(np.atleast_2d(logits_np).astype(np.float32))
        mse, predictions, designs = evaluate_logits(logits)
        return EvalResult(np.asarray(mse), np.asarray(predictions), np.asarray(designs))

    return evaluate


def nn_baseline(inv_frozen, inv_params, surrogate_params, target: jnp.ndarray):
    logits = pure_inverse_forward(inv_params, inv_frozen, target)
    designs = hard_threshold_design(logits)
    predictions = forward_surrogate(surrogate_params, designs)
    mse = float(jax.device_get(jnp.mean((predictions - target) ** 2)))
    return mse, np.asarray(predictions), np.asarray(designs)


def log_row(writer, method, iteration, elapsed, baseline_mse, eval_result, best):
    idx = int(np.argmin(eval_result.mse))
    step_mse = float(eval_result.mse[idx])
    if step_mse < best["mse"]:
        best.update(
            mse=step_mse,
            iteration=iteration,
            prediction=eval_result.predictions[idx : idx + 1],
            design=eval_result.designs[idx : idx + 1],
        )
    writer.writerow(
        {
            "method": method,
            "iteration": iteration,
            "elapsed_seconds": elapsed,
            "baseline_nn_mse": baseline_mse,
            "step_best_mse": step_mse,
            "best_mse": best["mse"],
            "beats_baseline": step_mse < baseline_mse,
            "best_iteration": best["iteration"],
        }
    )
    return step_mse


def run_cem(args, evaluate, writer, baseline_mse, rng):
    dim = DEFAULT_LAYOUT.flat_size
    mean = np.zeros(dim, dtype=np.float32)
    std = np.full(dim, args.sigma, dtype=np.float32)
    elite_count = max(2, int(args.population_size * args.cem_elite_frac))
    best = {"mse": float("inf"), "iteration": None, "prediction": None, "design": None}
    start = time.perf_counter()
    for iteration in range(args.iterations + 1):
        samples = mean + rng.normal(size=(args.population_size, dim)).astype(np.float32) * std
        result = evaluate(samples)
        elapsed = time.perf_counter() - start
        log_row(writer, "cem", iteration, elapsed, baseline_mse, result, best)
        elite_idx = np.argsort(result.mse)[:elite_count]
        elite = samples[elite_idx]
        mean = args.cem_alpha * mean + (1.0 - args.cem_alpha) * elite.mean(axis=0)
        std = args.cem_alpha * std + (1.0 - args.cem_alpha) * (elite.std(axis=0) + 1e-3)
    return best


def run_gaussian_es(args, evaluate, writer, baseline_mse, rng):
    dim = DEFAULT_LAYOUT.flat_size
    center = np.zeros(dim, dtype=np.float32)
    best = {"mse": float("inf"), "iteration": None, "prediction": None, "design": None}
    start = time.perf_counter()
    half = args.population_size // 2
    for iteration in range(args.iterations + 1):
        eps = rng.normal(size=(half, dim)).astype(np.float32)
        noise = np.concatenate([eps, -eps], axis=0)
        samples = center + args.sigma * noise
        result = evaluate(samples)
        elapsed = time.perf_counter() - start
        log_row(writer, "gaussian-es", iteration, elapsed, baseline_mse, result, best)
        scores = -result.mse
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        grad = (scores[:, None] * noise).mean(axis=0) / args.sigma
        center = center + args.gaussian_lr * grad.astype(np.float32)
    return best


def run_bpso(args, evaluate, writer, baseline_mse, rng):
    dim = DEFAULT_LAYOUT.flat_size
    positions = rng.integers(0, 2, size=(args.population_size, dim)).astype(np.float32)
    velocities = np.zeros_like(positions)
    pbest = positions.copy()
    pbest_scores = np.full(args.population_size, np.inf)
    gbest = positions[0].copy()
    gbest_score = float("inf")
    best = {"mse": float("inf"), "iteration": None, "prediction": None, "design": None}
    start = time.perf_counter()
    for iteration in range(args.iterations + 1):
        logits = positions * 2.0 - 1.0
        result = evaluate(logits)
        elapsed = time.perf_counter() - start
        log_row(writer, "bpso", iteration, elapsed, baseline_mse, result, best)
        improved = result.mse < pbest_scores
        pbest[improved] = positions[improved]
        pbest_scores[improved] = result.mse[improved]
        idx = int(np.argmin(pbest_scores))
        if float(pbest_scores[idx]) < gbest_score:
            gbest_score = float(pbest_scores[idx])
            gbest = pbest[idx].copy()
        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = args.bpso_w * velocities + args.bpso_c1 * r1 * (pbest - positions) + args.bpso_c2 * r2 * (gbest - positions)
        probs = 1.0 / (1.0 + np.exp(-velocities))
        positions = (rng.random(size=positions.shape) < probs).astype(np.float32)
    return best


def run_cma(args, evaluate, writer, baseline_mse, rng):
    import cma

    dim = DEFAULT_LAYOUT.flat_size
    x0 = np.zeros(dim, dtype=np.float64)
    es = cma.CMAEvolutionStrategy(
        x0,
        args.sigma,
        {"popsize": args.population_size, "seed": int(rng.integers(0, 2**31 - 1)), "verbose": -9},
    )
    best = {"mse": float("inf"), "iteration": None, "prediction": None, "design": None}
    start = time.perf_counter()
    for iteration in range(args.iterations + 1):
        samples = np.asarray(es.ask(), dtype=np.float32)
        result = evaluate(samples)
        elapsed = time.perf_counter() - start
        log_row(writer, "cma", iteration, elapsed, baseline_mse, result, best)
        es.tell(samples, result.mse.tolist())
    return best


def save_overlay(path: Path, target, nn_pred, baseline_mse, best_by_method):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.asarray(target[0]), label="target", linewidth=2)
    ax.plot(nn_pred[0], label=f"NN generator MSE={baseline_mse:.3f}")
    for method, best in best_by_method.items():
        if best["prediction"] is not None:
            ax.plot(best["prediction"][0], label=f"{method} MSE={best['mse']:.3f}")
    ax.set_xlabel("Frequency index")
    ax.set_ylabel("S11")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.population_size % 2 != 0:
        raise SystemExit("--population-size must be even")

    spectra = load_spectra_mat(args.spectra_mat)
    target = spectra[args.target_index : args.target_index + 1]
    surrogate_params = forward_surrogate_from_torch_checkpoint(args.forward_checkpoint)
    inv_frozen, inv_params, _scan, _es = inverse_from_torch_checkpoint(args.inverse_checkpoint)
    baseline_mse, nn_pred, _nn_design = nn_baseline(inv_frozen, inv_params, surrogate_params, target)
    evaluate = build_evaluator(surrogate_params, target)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    runners = {"cma": run_cma, "cem": run_cem, "gaussian-es": run_gaussian_es, "bpso": run_bpso}

    best_by_method = {}
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "iteration",
                "elapsed_seconds",
                "baseline_nn_mse",
                "step_best_mse",
                "best_mse",
                "beats_baseline",
                "best_iteration",
            ],
        )
        writer.writeheader()
        for offset, method in enumerate(methods):
            if method not in runners:
                raise SystemExit(f"Unknown method {method!r}; choose from {sorted(runners)}")
            print(f"running {method} baseline_nn_mse={baseline_mse:.6f}")
            rng = np.random.default_rng(args.seed + offset * 1000)
            best_by_method[method] = runners[method](args, evaluate, writer, baseline_mse, rng)
            fh.flush()
            print(
                f"{method}: best_mse={best_by_method[method]['mse']:.6f} "
                f"iteration={best_by_method[method]['iteration']}"
            )

    save_overlay(output_dir / "s11_overlay.png", np.asarray(target), nn_pred, baseline_mse, best_by_method)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "baseline_nn_mse", "best_mse", "best_iteration"])
        writer.writeheader()
        for method, best in best_by_method.items():
            writer.writerow(
                {
                    "method": method,
                    "baseline_nn_mse": baseline_mse,
                    "best_mse": best["mse"],
                    "best_iteration": best["iteration"],
                }
            )


if __name__ == "__main__":
    main()
