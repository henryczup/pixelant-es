"""Bridge `PNGFScorer` NPZ batches to the AGPL fixed-design PNGF evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


FREQS = [
    ("Gmat_sub_01.bin", 25_000_000_000.0),
    ("Gmat_sub_02.bin", 27_500_000_000.0),
    ("Gmat_sub_03.bin", 30_000_000_000.0),
    ("Gmat_sub_04.bin", 32_500_000_000.0),
    ("Gmat_sub_05.bin", 35_000_000_000.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input `.npz` from PNGFScorer.")
    parser.add_argument("--output", required=True, help="Output `.npz` for PNGFScorer.")
    parser.add_argument("--evaluator", default="./evaluate-fixed-design-batch")
    parser.add_argument("--matrix-dir", default=".")
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def _normalize_masks(masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(masks, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim == 2 and arr.shape == (21, 21):
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[1:] != (21, 21):
        raise ValueError(f"expected masks shaped [N,21,21] or [N,1,21,21], got {arr.shape}")
    return (arr > 0.5).astype(np.uint8)


def _write_design(path: Path, mask: np.ndarray) -> None:
    lines = []
    for row in mask:
        lines.append("".join("x" if int(value) else "." for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_batch(tmp_dir: Path, masks: np.ndarray) -> Path:
    batch_dir = tmp_dir / "batch"
    design_dir = batch_dir / "designs"
    design_dir.mkdir(parents=True)
    candidates = []
    for idx, mask in enumerate(masks):
        candidate_id = f"candidate_{idx:05d}"
        rel_design = f"designs/{candidate_id}.txt"
        _write_design(batch_dir / rel_design, mask)
        bits = "".join(str(int(value)) for value in mask.reshape(-1))
        candidates.append({"candidate_id": candidate_id, "design_path": rel_design, "bits": bits})
    (batch_dir / "candidates.jsonl").write_text(
        "\n".join(json.dumps(candidate) for candidate in candidates) + "\n",
        encoding="utf-8",
    )
    return batch_dir


def _parse_metrics(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    s11_re = np.full((n, 5), np.nan, dtype=np.float32)
    s11_im = np.full((n, 5), np.nan, dtype=np.float32)
    directivity = np.full((n, 5), np.nan, dtype=np.float32)
    seen = np.zeros((n, 5), dtype=bool)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate_id = row["candidate_id"]
            idx = int(candidate_id.rsplit("_", 1)[-1])
            freq_ghz = float(row["freq_ghz"])
            freq_idx = min(range(5), key=lambda ii: abs(freq_ghz - (25.0 + 2.5 * ii)))
            s11_re[idx, freq_idx] = float(row["s11_real"])
            s11_im[idx, freq_idx] = float(row["s11_imag"])
            directivity[idx, freq_idx] = float(row["directivity"])
            seen[idx, freq_idx] = True
    valid = np.all(seen, axis=1) & np.all(np.isfinite(s11_re), axis=1) & np.all(np.isfinite(s11_im), axis=1) & np.all(np.isfinite(directivity), axis=1)
    errors = ["" if bool(ok) else "missing or nonfinite PNGF metrics" for ok in valid]
    return s11_re, s11_im, directivity, valid, errors


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        masks = _normalize_masks(data["masks"])
    matrix_dir = Path(args.matrix_dir).resolve()
    evaluator = Path(args.evaluator).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pngf_npz_score_") as tmp:
        tmp_dir = Path(tmp)
        batch_dir = _build_batch(tmp_dir, masks)
        metrics_csv = tmp_dir / "fixed_design_batch_metrics.csv"
        command = [
            str(evaluator),
            str(metrics_csv),
            str(batch_dir / "candidates.jsonl"),
            str(batch_dir),
        ]
        for matrix_name, freq in FREQS:
            command.extend([str(matrix_dir / matrix_name), str(freq)])
        completed = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
        if completed.returncode != 0:
            np.savez_compressed(
                output,
                s11_re=np.zeros((masks.shape[0], 5), dtype=np.float32),
                s11_im=np.zeros((masks.shape[0], 5), dtype=np.float32),
                directivity=np.zeros((masks.shape[0], 5), dtype=np.float32),
                valid=np.zeros((masks.shape[0],), dtype=bool),
                error_message=np.asarray([completed.stderr[:2000] or completed.stdout[:2000]] * masks.shape[0]),
            )
            return
        s11_re, s11_im, directivity, valid, errors = _parse_metrics(metrics_csv, masks.shape[0])
        np.savez_compressed(
            output,
            s11_re=s11_re,
            s11_im=s11_im,
            directivity=directivity,
            valid=valid,
            error_message=np.asarray(errors),
        )


if __name__ == "__main__":
    main()
