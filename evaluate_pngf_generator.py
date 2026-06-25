"""Evaluate saved PNGF target-conditioned generator checkpoints with PNGF scoring."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pixelant_eggroll.antenna_ops import mask_uniqueness, metal_fill_ratio
from pixelant_eggroll.checkpoints import load_inverse_npz
from pixelant_eggroll.models_jax import direct_cnn_eval
from pixelant_eggroll.pngf import PNGF_LAYOUT, hard_project_pngf_center_fed, pngf_target_errors, project_pngf_center_fed_mask_np
from pixelant_eggroll.scorers import PNGFScorer
from train_pngf_es_generator import load_pngf_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", required=True)
    parser.add_argument("--checkpoints", nargs="*", default=[], help="Generator `.npz` checkpoints to evaluate.")
    parser.add_argument("--output-csv", default="eggroll_runs/pngf_eval.csv")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--random-baseline", type=int, default=0, help="Number of projected random-mask batches to score.")
    parser.add_argument("--pngf-command", help="External PNGF batch scorer command with {input_npz} and {output_npz}.")
    parser.add_argument("--pngf-work-dir", default=r"C:\Users\hczupryna\dev\PNGF")
    parser.add_argument("--pngf-cache-dir", default="eggroll_runs/pngf_cache")
    parser.add_argument("--scorer-timeout", type=float, default=3600.0)
    parser.add_argument("--bad-spectrum-value", type=float, default=1.0e6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--directivity-scale", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _summarize(name: str, masks: jnp.ndarray, predictions: jnp.ndarray, targets: jnp.ndarray, scorer: PNGFScorer, args) -> dict[str, float | str]:
    errors = pngf_target_errors(predictions, targets, beta=args.beta, directivity_scale=args.directivity_scale)
    return {
        "method": name,
        "target_error_mean": float(jax.device_get(jnp.mean(errors.total))),
        "s11_error_mean": float(jax.device_get(jnp.mean(errors.s11))),
        "directivity_error_mean": float(jax.device_get(jnp.mean(errors.directivity))),
        "mask_uniqueness": float(jax.device_get(mask_uniqueness(masks, layout=PNGF_LAYOUT))),
        "metal_fill_ratio": float(jax.device_get(jnp.mean(metal_fill_ratio(masks, layout=PNGF_LAYOUT)))),
        "valid_ratio": float(np.mean(scorer.last_valid)) if len(scorer.last_valid) else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    targets, _masks = load_pngf_dataset(args.dataset_npz)
    targets = targets[: args.limit]
    scorer = PNGFScorer(
        command=args.pngf_command,
        work_dir=Path(args.pngf_work_dir),
        cache_dir=Path(args.pngf_cache_dir),
        timeout_seconds=args.scorer_timeout,
        bad_spectrum_value=args.bad_spectrum_value,
    )
    rows: list[dict[str, float | str]] = []
    for checkpoint in args.checkpoints:
        frozen, params, metadata = load_inverse_npz(checkpoint)
        logits = direct_cnn_eval(params, frozen, targets)
        masks = hard_project_pngf_center_fed(logits)
        predictions = scorer.score(masks)
        name = str(metadata.get("stage") or Path(checkpoint).stem)
        rows.append(_summarize(name, masks, predictions, targets, scorer, args))

    key = jax.random.key(args.seed)
    for idx in range(args.random_baseline):
        key, subkey = jax.random.split(key)
        random_masks = (jax.random.uniform(subkey, (targets.shape[0], 21, 21)) > 0.5).astype(jnp.float32)
        masks_np = project_pngf_center_fed_mask_np(np.asarray(random_masks))[:, None, :, :]
        masks = jnp.asarray(masks_np, dtype=jnp.float32)
        predictions = scorer.score(masks)
        rows.append(_summarize(f"random_projected_{idx}", masks, predictions, targets, scorer, args))

    if not rows:
        raise SystemExit("Nothing to evaluate; provide --checkpoints or --random-baseline")
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['method']}: err={row['target_error_mean']:.6g} "
            f"s11={row['s11_error_mean']:.6g} dir={row['directivity_error_mean']:.6g} "
            f"unique={row['mask_uniqueness']:.3f} valid={row['valid_ratio']:.2f}"
        )


if __name__ == "__main__":
    main()
