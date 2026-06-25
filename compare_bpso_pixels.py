"""Compare trained inverse generators against direct BPSO over antenna pixels.

This script takes representative S11 cuts exported by ``compare_supervised_vs_es.py``
and runs one independent binary particle swarm per target. BPSO optimizes the
hard 12x12 mask directly, so it is a useful "no generator" baseline for asking
whether one-shot inverse networks are actually helping.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pixelant_eggroll.antenna_ops import force_feed_pixels, metal_fill_ratio
from pixelant_eggroll.checkpoints import forward_surrogate_from_torch_checkpoint
from pixelant_eggroll.models_jax import forward_surrogate


FEED_FLAT_INDICES = (5 * 12, 6 * 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representative-npz", default="eggroll_runs/supervised_es_policy_200_cuts/representative_s11_cuts.npz")
    parser.add_argument("--forward-checkpoint", default="Forward_model_for_tandem.pth")
    parser.add_argument("--output-dir", default="eggroll_runs/bpso_pixels_vs_inverse_cuts")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bpso-w", type=float, default=0.72)
    parser.add_argument("--bpso-c1", type=float, default=1.49)
    parser.add_argument("--bpso-c2", type=float, default=1.49)
    return parser.parse_args()


def _enforce_feed_flat(positions: np.ndarray) -> np.ndarray:
    positions[..., FEED_FLAT_INDICES[0]] = 1.0
    positions[..., FEED_FLAT_INDICES[1]] = 1.0
    return positions


def _save_s11_overlay(path: Path, frequency_ghz: np.ndarray, targets: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    count = targets.shape[0]
    cols = min(3, count)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= count:
            ax.axis("off")
            continue
        ax.plot(frequency_ghz, targets[idx], color="black", linewidth=2.0, label="Target")
        for label, pred in predictions.items():
            ax.plot(frequency_ghz, pred[idx], linewidth=1.3, label=label)
        ax.set_title(f"Test target {idx}")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("S11 (dB)")
        ax.grid(True, alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_mask_grid(path: Path, masks: dict[str, np.ndarray]) -> None:
    labels = list(masks)
    count = next(iter(masks.values())).shape[0]
    fig, axes = plt.subplots(count, len(labels), figsize=(1.8 * len(labels), 1.9 * count), squeeze=False)
    for row in range(count):
        for col, label in enumerate(labels):
            ax = axes[row][col]
            ax.imshow(masks[label][row], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.representative_npz)
    targets = np.asarray(data["target"], dtype=np.float32)
    frequency_ghz = np.asarray(data["frequency_ghz"], dtype=np.float32)
    target_count, spectrum_dim = targets.shape
    population = int(args.population_size)
    dim = 144

    surrogate_params = forward_surrogate_from_torch_checkpoint(args.forward_checkpoint)

    @jax.jit
    def evaluate_positions(positions_: jnp.ndarray, targets_: jnp.ndarray):
        target_count_, population_, _ = positions_.shape
        designs = positions_.reshape((target_count_ * population_, 1, 12, 12))
        designs = force_feed_pixels(designs)
        pred = forward_surrogate(surrogate_params, designs).reshape((target_count_, population_, spectrum_dim))
        mse = jnp.mean((pred - targets_[:, None, :]) ** 2, axis=-1)
        fill = metal_fill_ratio(designs).reshape((target_count_, population_))
        return mse, pred, designs.reshape((target_count_, population_, 1, 12, 12)), fill

    rng = np.random.default_rng(args.seed)
    positions = rng.integers(0, 2, size=(target_count, population, dim)).astype(np.float32)
    positions = _enforce_feed_flat(positions)
    velocities = np.zeros_like(positions, dtype=np.float32)
    pbest = positions.copy()
    pbest_scores = np.full((target_count, population), np.inf, dtype=np.float32)
    gbest = positions[:, 0, :].copy()
    gbest_scores = np.full((target_count,), np.inf, dtype=np.float32)

    best_pred = np.zeros((target_count, spectrum_dim), dtype=np.float32)
    best_mask = np.zeros((target_count, 12, 12), dtype=np.float32)
    best_fill = np.zeros((target_count,), dtype=np.float32)
    rows = []

    for iteration in range(args.iterations + 1):
        mse, pred, designs, fill = evaluate_positions(jnp.asarray(positions), jnp.asarray(targets))
        mse_np = np.asarray(mse)
        pred_np = np.asarray(pred)
        designs_np = np.asarray(designs)
        fill_np = np.asarray(fill)

        improved = mse_np < pbest_scores
        pbest[improved] = positions[improved]
        pbest_scores[improved] = mse_np[improved]

        current_best_idx = np.argmin(pbest_scores, axis=1)
        for target_idx, particle_idx in enumerate(current_best_idx):
            candidate_score = float(pbest_scores[target_idx, particle_idx])
            if candidate_score < float(gbest_scores[target_idx]):
                gbest_scores[target_idx] = candidate_score
                gbest[target_idx] = pbest[target_idx, particle_idx].copy()

        for target_idx in range(target_count):
            step_idx = int(np.argmin(mse_np[target_idx]))
            if float(mse_np[target_idx, step_idx]) <= float(gbest_scores[target_idx]) + 1e-7:
                best_pred[target_idx] = pred_np[target_idx, step_idx]
                best_mask[target_idx] = designs_np[target_idx, step_idx, 0]
                best_fill[target_idx] = fill_np[target_idx, step_idx]

        rows.append(
            {
                "iteration": iteration,
                "mean_best_mse": float(np.mean(gbest_scores)),
                "median_best_mse": float(np.median(gbest_scores)),
                "max_best_mse": float(np.max(gbest_scores)),
                "min_best_mse": float(np.min(gbest_scores)),
            }
        )
        print(
            f"iteration={iteration:04d} mean_best_mse={np.mean(gbest_scores):.6g} "
            f"max_best_mse={np.max(gbest_scores):.6g}"
        )

        if iteration == args.iterations:
            break

        r1 = rng.random(size=positions.shape, dtype=np.float32)
        r2 = rng.random(size=positions.shape, dtype=np.float32)
        velocities = (
            args.bpso_w * velocities
            + args.bpso_c1 * r1 * (pbest - positions)
            + args.bpso_c2 * r2 * (gbest[:, None, :] - positions)
        )
        probs = 1.0 / (1.0 + np.exp(-velocities))
        positions = (rng.random(size=positions.shape, dtype=np.float32) < probs).astype(np.float32)
        positions = _enforce_feed_flat(positions)

    with (output_dir / "bpso_progress.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    comparison_rows = []
    predictions = {
        "Supervised": np.asarray(data["supervised_s11"], dtype=np.float32),
        "ES": np.asarray(data["es_s11"], dtype=np.float32),
        "Policy": np.asarray(data["policy_gradient_s11"], dtype=np.float32),
        "BPSO pixels": best_pred,
    }
    masks = {
        "Supervised": np.asarray(data["supervised_mask"], dtype=np.float32),
        "ES": np.asarray(data["es_mask"], dtype=np.float32),
        "Policy": np.asarray(data["policy_gradient_mask"], dtype=np.float32),
        "BPSO": best_mask,
    }
    for target_idx in range(target_count):
        row = {"target": target_idx, "bpso_fill": float(best_fill[target_idx])}
        for label, pred in predictions.items():
            row[f"{label.lower().replace(' ', '_')}_mse"] = float(np.mean((pred[target_idx] - targets[target_idx]) ** 2))
        comparison_rows.append(row)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    _save_s11_overlay(output_dir / "s11_cuts_with_bpso.png", frequency_ghz, targets, predictions)
    _save_mask_grid(output_dir / "hard_masks_with_bpso.png", masks)
    np.savez_compressed(
        output_dir / "bpso_pixels_results.npz",
        target=targets,
        frequency_ghz=frequency_ghz,
        bpso_s11=best_pred,
        bpso_mask=best_mask,
        bpso_mse=gbest_scores,
        bpso_fill=best_fill,
    )


if __name__ == "__main__":
    main()
