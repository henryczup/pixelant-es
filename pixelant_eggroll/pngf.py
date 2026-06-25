"""PNGF center-fed antenna utilities.

The PNGF paper substrate antenna is a 21x21 single-layer mask with two-axis
mirror symmetry and a fixed center lumped feed.  This module keeps those
geometry rules separate from the earlier 12x12 edge-fed antenna utilities.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from .config import LayoutSpec


PNGF_FREQ_HZ = np.asarray([25.0e9, 27.5e9, 30.0e9, 32.5e9, 35.0e9], dtype=np.float64)
PNGF_TARGET_DIM = 15
PNGF_LAYOUT = LayoutSpec(height=21, width=21, feed_pixels=((10, 9), (10, 11)))
PNGF_FORCED_METAL: tuple[tuple[int, int], ...] = ((10, 9), (10, 11))
PNGF_FORCED_EMPTY: tuple[tuple[int, int], ...] = ((10, 10), (9, 10), (11, 10))


@dataclass(frozen=True)
class PNGFTargetErrors:
    total: jnp.ndarray
    s11: jnp.ndarray
    directivity: jnp.ndarray


def _to_nchw_jax(values: jnp.ndarray) -> jnp.ndarray:
    values = jnp.asarray(values, dtype=jnp.float32)
    h, w = PNGF_LAYOUT.height, PNGF_LAYOUT.width
    if values.ndim == 1:
        if values.shape[0] != h * w:
            raise ValueError(f"Expected flat PNGF mask/logits of length {h * w}, got {values.shape}")
        return values.reshape(1, 1, h, w)
    if values.ndim == 2:
        if values.shape == (h, w):
            return values.reshape(1, 1, h, w)
        if values.shape[1] == h * w:
            return values.reshape(values.shape[0], 1, h, w)
    if values.ndim == 3 and values.shape[-2:] == (h, w):
        return values.reshape(values.shape[0], 1, h, w)
    if values.ndim == 4 and values.shape[1:] == (1, h, w):
        return values
    raise ValueError(f"Expected PNGF values shaped [N,21,21], [N,1,21,21], or [N,441], got {values.shape}")


def _to_nhw_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    h, w = PNGF_LAYOUT.height, PNGF_LAYOUT.width
    if values.ndim == 2 and values.shape == (h, w):
        return values.reshape(1, h, w)
    if values.ndim == 3 and values.shape[-2:] == (h, w):
        if values.shape[0] == 1 and values.shape[1:] == (h, w):
            return values.reshape(1, h, w)
        return values.reshape(values.shape[0], h, w)
    if values.ndim == 4 and values.shape[1:] == (1, h, w):
        return values[:, 0, :, :]
    if values.ndim == 2 and values.shape[1] == h * w:
        return values.reshape(values.shape[0], h, w)
    if values.ndim == 1 and values.shape[0] == h * w:
        return values.reshape(1, h, w)
    raise ValueError(f"Expected PNGF masks shaped [N,21,21], [N,1,21,21], or [N,441], got {values.shape}")


def symmetrize_two_axis_logits(logits: jnp.ndarray) -> jnp.ndarray:
    """Average logits over the four two-axis mirror copies."""

    logits = _to_nchw_jax(logits)
    return 0.25 * (
        logits
        + jnp.flip(logits, axis=-1)
        + jnp.flip(logits, axis=-2)
        + jnp.flip(jnp.flip(logits, axis=-1), axis=-2)
    )


def enforce_pngf_center_feed(mask: jnp.ndarray) -> jnp.ndarray:
    """Force the paper center-feed metal/open pixels on an NCHW mask."""

    mask = _to_nchw_jax(mask)
    for row, col in PNGF_FORCED_EMPTY:
        mask = mask.at[:, 0, row, col].set(0.0)
    for row, col in PNGF_FORCED_METAL:
        mask = mask.at[:, 0, row, col].set(1.0)
    return mask


def hard_project_pngf_center_fed(logits: jnp.ndarray, threshold: float = 0.0) -> jnp.ndarray:
    """Project generator logits to hard two-axis-symmetric center-fed masks."""

    sym_logits = symmetrize_two_axis_logits(logits)
    mask = (sym_logits > threshold).astype(jnp.float32)
    return enforce_pngf_center_feed(mask)


def project_pngf_center_fed_mask_np(mask: np.ndarray) -> np.ndarray:
    """Project numpy masks to hard two-axis-symmetric center-fed masks.

    Existing metal is preserved by OR-ing the four mirror copies.  This is the
    right behavior for cleaning scorer inputs because it does not silently remove
    metal away from the feed constraints.
    """

    masks = (_to_nhw_np(mask) > 0.5).astype(np.float32)
    masks = np.maximum.reduce(
        [
            masks,
            np.flip(masks, axis=-1),
            np.flip(masks, axis=-2),
            np.flip(np.flip(masks, axis=-1), axis=-2),
        ]
    )
    for row, col in PNGF_FORCED_EMPTY:
        masks[:, row, col] = 0.0
    for row, col in PNGF_FORCED_METAL:
        masks[:, row, col] = 1.0
    return masks.astype(np.float32)


def pngf_masks_are_projected(mask: np.ndarray) -> np.ndarray:
    masks = (_to_nhw_np(mask) > 0.5).astype(np.float32)
    projected = project_pngf_center_fed_mask_np(masks)
    return np.all(projected == masks, axis=(1, 2))


def pack_pngf_targets(
    s11: np.ndarray | None = None,
    directivity: np.ndarray | None = None,
    *,
    s11_re: np.ndarray | None = None,
    s11_im: np.ndarray | None = None,
) -> np.ndarray:
    """Pack five-frequency PNGF metrics as [Re(S11), Im(S11), D] per frequency."""

    if s11 is not None:
        s11 = np.asarray(s11)
        s11_re_arr = np.real(s11)
        s11_im_arr = np.imag(s11)
    else:
        if s11_re is None or s11_im is None:
            raise ValueError("Provide either complex s11 or both s11_re and s11_im")
        s11_re_arr = np.asarray(s11_re)
        s11_im_arr = np.asarray(s11_im)
    if directivity is None:
        raise ValueError("directivity is required")
    directivity_arr = np.asarray(directivity)
    if s11_re_arr.ndim == 1:
        s11_re_arr = s11_re_arr.reshape(1, -1)
        s11_im_arr = s11_im_arr.reshape(1, -1)
        directivity_arr = directivity_arr.reshape(1, -1)
    if s11_re_arr.shape != s11_im_arr.shape or s11_re_arr.shape != directivity_arr.shape:
        raise ValueError("s11_re, s11_im, and directivity must have matching shapes")
    if s11_re_arr.shape[1] != 5:
        raise ValueError(f"Expected five PNGF frequency samples, got shape {s11_re_arr.shape}")
    packed = np.stack([s11_re_arr, s11_im_arr, directivity_arr], axis=-1)
    return packed.reshape(s11_re_arr.shape[0], PNGF_TARGET_DIM).astype(np.float32)


def unpack_pngf_targets(targets):
    """Return `(s11_re, s11_im, directivity)` arrays with shape `[N, 5]`."""

    arr = jnp.asarray(targets, dtype=jnp.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, PNGF_TARGET_DIM)
    if arr.shape[-1] != PNGF_TARGET_DIM:
        raise ValueError(f"Expected PNGF target dimension {PNGF_TARGET_DIM}, got {arr.shape}")
    arr = arr.reshape(arr.shape[0], 5, 3)
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]


def pngf_target_errors(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    beta: float = 1.0,
    directivity_scale: float = 15.0,
) -> PNGFTargetErrors:
    pred_re, pred_im, pred_d = unpack_pngf_targets(predictions)
    target_re, target_im, target_d = unpack_pngf_targets(targets)
    s11_error = jnp.mean(jnp.square(pred_re - target_re) + jnp.square(pred_im - target_im), axis=1)
    directivity_error = jnp.mean(jnp.square((pred_d - target_d) / directivity_scale), axis=1)
    total = s11_error + beta * directivity_error
    return PNGFTargetErrors(total=total, s11=s11_error, directivity=directivity_error)


def pngf_paper_objective(targets: np.ndarray) -> np.ndarray:
    arr = np.asarray(targets, dtype=np.float32).reshape(-1, 5, 3)
    s11_mag2 = arr[:, :, 0] ** 2 + arr[:, :, 1] ** 2
    directivity = arr[:, :, 2]
    return np.sum(200.0 * s11_mag2 + np.square(15.0 - directivity), axis=1).astype(np.float32)


def read_pngf_design_csv(path: str | Path) -> np.ndarray:
    path = Path(path)
    mask = np.zeros((PNGF_LAYOUT.height, PNGF_LAYOUT.width), dtype=np.float32)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x = int(row["x_index"])
            y = int(row["y_index"])
            mask[y, x] = float(row["metal"])
    return project_pngf_center_fed_mask_np(mask)[0]


def read_pngf_sparams_csv(path: str | Path) -> np.ndarray:
    path = Path(path)
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if len(rows) != 5:
        raise ValueError(f"Expected five PNGF frequency rows in {path}, got {len(rows)}")
    s11_re = np.asarray([float(row["s11_re"]) for row in rows], dtype=np.float32)
    s11_im = np.asarray([float(row["s11_im"]) for row in rows], dtype=np.float32)
    directivity = np.asarray([float(row["directivity"]) for row in rows], dtype=np.float32)
    return pack_pngf_targets(s11_re=s11_re, s11_im=s11_im, directivity=directivity)[0]


def center_fed_symmetric_flip_indices(base_index: int, *, height: int = 21, width: int = 21) -> tuple[int, ...]:
    """Return paper two-axis symmetric tile indices for a reduced DBS flip."""

    mid_x = (width - 1) // 2
    mid_y = (height - 1) // 2
    ty = base_index // width
    tx = base_index - ty * width
    if (tx, ty) in {(mid_x, mid_y), (mid_x - 1, mid_y)}:
        return tuple()
    coords = {(tx, ty), (width - 1 - tx, ty), (tx, height - 1 - ty), (width - 1 - tx, height - 1 - ty)}
    return tuple(sorted(x + y * width for x, y in coords))


def _mask_from_logger_chars(chars: str) -> np.ndarray:
    chars = chars.strip()
    expected = PNGF_LAYOUT.height * PNGF_LAYOUT.width
    if len(chars) != expected:
        raise ValueError(f"Expected {expected} tile chars in PNGF log, got {len(chars)}")
    flat = np.fromiter((1.0 if ch == "x" else 0.0 for ch in chars), dtype=np.float32, count=expected)
    return project_pngf_center_fed_mask_np(flat.reshape(PNGF_LAYOUT.height, PNGF_LAYOUT.width))[0]


def parse_pngf_dbs_log(path: str | Path, *, run_id: str | None = None) -> list[dict[str, object]]:
    """Parse a PNGF optimizer log into accepted/rejected candidate records."""

    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"PNGF log {path} is too short")
    seed, num_flips = (int(value) for value in lines[0].split()[:2])
    current = _mask_from_logger_chars(lines[1])
    records: list[dict[str, object]] = []
    for row in lines[2 : 2 + num_flips]:
        parts = row.split()
        if len(parts) < 20:
            raise ValueError(f"Malformed PNGF log record in {path}: {row}")
        step = int(parts[0])
        flip_index = int(parts[1])
        accepted = int(parts[2]) == 1
        perf = np.asarray([float(value) for value in parts[3:20]], dtype=np.float32)
        if flip_index < 0:
            candidate = current.copy()
        else:
            candidate = current.copy()
            for idx in center_fed_symmetric_flip_indices(flip_index):
                y = idx // PNGF_LAYOUT.width
                x = idx - y * PNGF_LAYOUT.width
                candidate[y, x] = 1.0 - candidate[y, x]
            candidate = project_pngf_center_fed_mask_np(candidate)[0]
        records.append(
            {
                "mask": candidate,
                "target": perf[2:17].copy(),
                "objective": float(perf[0]),
                "previous_best_objective": float(perf[1]),
                "accepted": bool(accepted),
                "seed": seed,
                "run_id": run_id or path.stem,
                "step": step,
                "flip_index": flip_index,
            }
        )
        if accepted:
            current = candidate
    return records


def export_pngf_dbs_records(records: Iterable[dict[str, object]], output_npz: str | Path) -> None:
    records = list(records)
    if not records:
        raise ValueError("No PNGF DBS records to export")
    masks = np.stack([np.asarray(record["mask"], dtype=np.float32) for record in records], axis=0)
    targets = np.stack([np.asarray(record["target"], dtype=np.float32) for record in records], axis=0)
    objective = np.asarray([record["objective"] for record in records], dtype=np.float32)
    accepted = np.asarray([record["accepted"] for record in records], dtype=bool)
    seed = np.asarray([record["seed"] for record in records], dtype=np.int64)
    step = np.asarray([record["step"] for record in records], dtype=np.int64)
    flip_index = np.asarray([record["flip_index"] for record in records], dtype=np.int64)
    run_id = np.asarray([str(record["run_id"]) for record in records])
    np.savez_compressed(
        output_npz,
        masks=project_pngf_center_fed_mask_np(masks),
        targets=targets.astype(np.float32),
        objective=objective,
        accepted=accepted,
        seed=seed,
        step=step,
        flip_index=flip_index,
        run_id=run_id,
        freq_hz=PNGF_FREQ_HZ,
    )
