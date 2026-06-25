"""Hard-binary antenna masks and physical penalty terms."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from .config import DEFAULT_LAYOUT, FeedPixel, LayoutSpec

FEED_ROWS: tuple[int, int] = (5, 6)
FEED_COL: int = 0
AREA_TARGET: float = 0.50


def _legacy_layout(feed_rows: Sequence[int] = FEED_ROWS, feed_col: int = FEED_COL) -> LayoutSpec:
    return LayoutSpec(
        height=12,
        width=12,
        layers=1,
        feed_pixels=tuple((int(row), int(feed_col)) for row in feed_rows),
    )


def _infer_layout(design: jnp.ndarray) -> LayoutSpec:
    if design.ndim == 0:
        raise ValueError("Expected an antenna mask array, got scalar input")
    if design.shape[-1] == DEFAULT_LAYOUT.flat_size and design.ndim <= 2:
        return DEFAULT_LAYOUT
    if design.ndim >= 2 and design.shape[-2:] == (12, 12):
        return DEFAULT_LAYOUT
    if design.ndim == 4:
        return LayoutSpec(height=int(design.shape[-2]), width=int(design.shape[-1]), layers=int(design.shape[-3]), feed_pixels=())
    raise ValueError(f"Could not infer layout from mask shape {design.shape}; pass layout=LayoutSpec(...)")


def _as_nlhw(design: jnp.ndarray, layout: LayoutSpec | None = None) -> tuple[jnp.ndarray, bool]:
    """Return masks as [batch, layers, height, width] and whether input was unbatched."""

    design = jnp.asarray(design)
    layout = layout or _infer_layout(design)
    flat_size = layout.flat_size
    unbatched = False

    if design.ndim == 1 and design.shape[0] == flat_size:
        design = design.reshape(1, *layout.mask_shape)
        unbatched = True
    elif design.ndim == 2 and design.shape == (layout.height, layout.width) and layout.layers == 1:
        design = design[None, None, :, :]
        unbatched = True
    elif design.ndim == 2 and design.shape[-1] == flat_size:
        design = design.reshape(design.shape[0], *layout.mask_shape)
    elif design.ndim == 3 and design.shape == layout.mask_shape:
        design = design[None, :, :, :]
        unbatched = True
    elif design.ndim == 3 and design.shape[-2:] == (layout.height, layout.width) and layout.layers == 1:
        design = design[:, None, :, :]

    if design.ndim != 4 or design.shape[1:] != layout.mask_shape:
        raise ValueError(
            "Expected mask shaped [N, flat], [N, layers, height, width], "
            f"or {layout.mask_shape}; got {design.shape} for layout {layout}"
        )
    return design, unbatched


def _restore_unbatched(design: jnp.ndarray, unbatched: bool) -> jnp.ndarray:
    if not unbatched:
        return design
    return design[0, 0] if design.shape[1] == 1 else design[0]


def force_feed_pixels(
    design: jnp.ndarray,
    feed_rows: Sequence[int] = FEED_ROWS,
    feed_col: int = FEED_COL,
    layout: LayoutSpec | None = None,
) -> jnp.ndarray:
    """Force feed-adjacent pixels to metal.

    The default matches the MATLAB scripts, which use one-based rows 6 and 7.
    These are zero-based rows 5 and 6 in Python. For generalized layouts,
    pass a LayoutSpec with explicit feed pixels.
    """

    layout = layout or _legacy_layout(feed_rows=feed_rows, feed_col=feed_col)
    design, unbatched = _as_nlhw(design, layout)
    for pixel in layout.feed_pixels:
        layer, row, col = LayoutSpec.normalize_feed_pixel(pixel)
        design = design.at[:, layer, row, col].set(1.0)
    return _restore_unbatched(design, unbatched)


def hard_threshold_design(
    logits: jnp.ndarray,
    feed_rows: Sequence[int] = FEED_ROWS,
    feed_col: int = FEED_COL,
    layout: LayoutSpec | None = None,
) -> jnp.ndarray:
    """Convert generator logits to a true binary [N, layers, height, width] mask."""

    layout = layout or _legacy_layout(feed_rows=feed_rows, feed_col=feed_col)
    logits = jnp.asarray(logits)
    flat_size = layout.flat_size
    if logits.ndim >= 1 and logits.shape[-1] == flat_size:
        design = (logits > 0).astype(jnp.float32).reshape(-1, *layout.mask_shape)
    elif logits.ndim == 4 and logits.shape[1:] == layout.mask_shape:
        design = (logits > 0).astype(jnp.float32)
    else:
        raise ValueError(
            f"Expected logits with final dimension {flat_size} or shape [N, {layout.layers}, "
            f"{layout.height}, {layout.width}], got {logits.shape} for layout {layout}"
        )
    return force_feed_pixels(design, layout=layout)


def metal_fill_ratio(design: jnp.ndarray, layout: LayoutSpec | None = None) -> jnp.ndarray:
    design, _ = _as_nlhw(design, layout)
    return jnp.mean(design, axis=(1, 2, 3))


def area_penalty(
    design: jnp.ndarray,
    target_fill: float = AREA_TARGET,
    layout: LayoutSpec | None = None,
) -> jnp.ndarray:
    fill = metal_fill_ratio(design, layout=layout)
    return (fill - target_fill) ** 2


def _neighbor_count(mask: jnp.ndarray) -> jnp.ndarray:
    up = jnp.pad(mask[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
    down = jnp.pad(mask[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
    left = jnp.pad(mask[:, :, :-1], ((0, 0), (0, 0), (1, 0)))
    right = jnp.pad(mask[:, :, 1:], ((0, 0), (0, 0), (0, 1)))
    return up + down + left + right


def feed_connectivity_penalty(
    design: jnp.ndarray,
    feed_rows: Sequence[int] = FEED_ROWS,
    feed_col: int = FEED_COL,
    iterations: int | None = None,
    layout: LayoutSpec | None = None,
) -> jnp.ndarray:
    """Penalize metal cells not connected to feed pixels by 4-neighbor paths.

    Connectivity is evaluated independently per layer and then aggregated.
    """

    layout = layout or _legacy_layout(feed_rows=feed_rows, feed_col=feed_col)
    design, _ = _as_nlhw(design, layout)
    metal = design > 0.5
    seed = jnp.zeros_like(metal, dtype=bool)
    for pixel in layout.feed_pixels:
        layer, row, col = LayoutSpec.normalize_feed_pixel(pixel)
        seed = seed.at[:, layer, row, col].set(True)
    connected0 = seed & metal
    scan_steps = iterations or (layout.height + layout.width + layout.layers)

    def step(connected, _):
        grown_layers = []
        for layer in range(layout.layers):
            grown = (_neighbor_count(connected[:, layer].astype(jnp.float32)) > 0) | connected[:, layer]
            grown_layers.append(grown)
        grown_all = jnp.stack(grown_layers, axis=1)
        return grown_all & metal, None

    connected, _ = jax.lax.scan(step, connected0, xs=None, length=scan_steps)
    metal_count = jnp.maximum(jnp.sum(metal, axis=(1, 2, 3)), 1)
    disconnected = jnp.sum(metal & ~connected, axis=(1, 2, 3))
    return disconnected / metal_count


def fragmentation_penalty(design: jnp.ndarray, layout: LayoutSpec | None = None) -> jnp.ndarray:
    """Penalize isolated metal pixels that have no 4-neighbor metal contact."""

    layout = layout or _infer_layout(jnp.asarray(design))
    design, _ = _as_nlhw(design, layout)
    metal = design > 0.5
    isolated_layers = []
    for layer in range(layout.layers):
        neighbors = _neighbor_count(metal[:, layer].astype(jnp.float32))
        isolated_layers.append(metal[:, layer] & (neighbors == 0))
    isolated = jnp.stack(isolated_layers, axis=1)
    metal_count = jnp.maximum(jnp.sum(metal, axis=(1, 2, 3)), 1)
    return jnp.sum(isolated, axis=(1, 2, 3)) / metal_count


def fabrication_rule_penalty(design: jnp.ndarray, layout: LayoutSpec | None = None) -> jnp.ndarray:
    """Penalize single-pixel-width notches/bridges using local transitions."""

    design, _ = _as_nlhw(design, layout)
    horizontal_changes = jnp.mean(jnp.abs(design[:, :, :, 1:] - design[:, :, :, :-1]), axis=(1, 2, 3))
    vertical_changes = jnp.mean(jnp.abs(design[:, :, 1:, :] - design[:, :, :-1, :]), axis=(1, 2, 3))
    return 0.5 * (horizontal_changes + vertical_changes)


def mask_uniqueness(design: jnp.ndarray, layout: LayoutSpec | None = None) -> jnp.ndarray:
    """Return the fraction of unique binary masks in a batch."""

    design, _ = _as_nlhw(design, layout)
    flat = design.reshape(design.shape[0], -1).astype(jnp.int32)
    unique = jnp.unique(flat, axis=0, size=flat.shape[0], fill_value=-1)
    valid = jnp.any(unique >= 0, axis=1)
    return jnp.sum(valid) / flat.shape[0]
