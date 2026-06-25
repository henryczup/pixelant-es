import pytest

jnp = pytest.importorskip("jax.numpy")

from pixelant_eggroll.antenna_ops import (
    area_penalty,
    fabrication_rule_penalty,
    feed_connectivity_penalty,
    force_feed_pixels,
    fragmentation_penalty,
    hard_threshold_design,
    mask_uniqueness,
    metal_fill_ratio,
)
from pixelant_eggroll.config import LayoutSpec


def test_hard_threshold_forces_feed_pixels():
    logits = jnp.full((2, 144), -1.0)
    logits = logits.at[1, 0].set(2.0)

    design = hard_threshold_design(logits)

    assert design.shape == (2, 1, 12, 12)
    assert design[0, 0, 5, 0] == 1.0
    assert design[0, 0, 6, 0] == 1.0
    assert design[0, 0, 0, 1] == 0.0
    assert design[1, 0, 0, 0] == 1.0


def test_force_feed_pixels_accepts_unbatched_mask():
    mask = jnp.zeros((12, 12), dtype=jnp.float32)
    forced = force_feed_pixels(mask)
    assert forced.shape == (12, 12)
    assert forced[5, 0] == 1.0
    assert forced[6, 0] == 1.0


def test_penalties_distinguish_connected_and_fragmented_masks():
    connected = jnp.zeros((1, 1, 12, 12), dtype=jnp.float32)
    connected = connected.at[0, 0, 5:8, 0].set(1.0)
    fragmented = connected.at[0, 0, 11, 11].set(1.0)

    assert feed_connectivity_penalty(connected)[0] == 0.0
    assert feed_connectivity_penalty(fragmented)[0] > 0.0
    assert fragmentation_penalty(fragmented)[0] > fragmentation_penalty(connected)[0]
    assert area_penalty(connected)[0] >= 0.0
    assert fabrication_rule_penalty(fragmented)[0] >= 0.0


def test_batch_metrics():
    masks = jnp.zeros((2, 1, 12, 12), dtype=jnp.float32)
    masks = masks.at[1, 0, :, :].set(1.0)

    fill = metal_fill_ratio(masks)
    assert fill.shape == (2,)
    assert fill[0] == 0.0
    assert fill[1] == 1.0
    assert mask_uniqueness(masks) == 1.0


def test_hard_threshold_accepts_32x32_layout():
    layout = LayoutSpec(height=32, width=32, feed_pixels=((15, 0), (16, 0)))
    logits = jnp.full((2, layout.flat_size), -1.0)

    design = hard_threshold_design(logits, layout=layout)

    assert design.shape == (2, 1, 32, 32)
    assert design[0, 0, 15, 0] == 1.0
    assert design[0, 0, 16, 0] == 1.0
    assert design[0, 0, 0, 0] == 0.0


def test_multilayer_feed_pixels_and_metrics():
    layout = LayoutSpec(height=8, width=8, layers=2, feed_pixels=((0, 3, 0), (1, 4, 0)))
    logits = jnp.full((1, 2, 8, 8), -1.0)

    design = hard_threshold_design(logits, layout=layout)
    fill = metal_fill_ratio(design, layout=layout)

    assert design.shape == (1, 2, 8, 8)
    assert design[0, 0, 3, 0] == 1.0
    assert design[0, 1, 4, 0] == 1.0
    assert fill.shape == (1,)
    assert fill[0] > 0.0


def test_shape_validation_mentions_layout():
    layout = LayoutSpec(height=32, width=32)
    with pytest.raises(ValueError, match="layout"):
        hard_threshold_design(jnp.zeros((1, 144)), layout=layout)
