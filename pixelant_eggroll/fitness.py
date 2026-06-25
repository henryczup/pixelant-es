"""Fitness calculation for surrogate-scored inverse antenna training."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .antenna_ops import (
    area_penalty,
    fabrication_rule_penalty,
    feed_connectivity_penalty,
    fragmentation_penalty,
)
from .config import LayoutSpec


@dataclass(frozen=True)
class FitnessConfig:
    lambda_conn: float = 0.0
    lambda_area: float = 0.0
    lambda_frag: float = 0.0
    lambda_rule: float = 0.0
    target_fill: float = 0.50


def compute_fitness(
    pred_spectrum: jnp.ndarray,
    target_spectrum: jnp.ndarray,
    design: jnp.ndarray,
    config: FitnessConfig,
    layout: LayoutSpec | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return scalar fitness per population member and metric components."""

    spectrum_mse = jnp.mean((pred_spectrum - target_spectrum) ** 2, axis=-1)
    conn = feed_connectivity_penalty(design, layout=layout)
    area = area_penalty(design, target_fill=config.target_fill, layout=layout)
    frag = fragmentation_penalty(design, layout=layout)
    rule = fabrication_rule_penalty(design, layout=layout)

    total_penalty = (
        config.lambda_conn * conn
        + config.lambda_area * area
        + config.lambda_frag * frag
        + config.lambda_rule * rule
    )
    fitness = -spectrum_mse - total_penalty
    metrics = {
        "fitness": fitness,
        "spectrum_mse": spectrum_mse,
        "connectivity_penalty": conn,
        "area_penalty": area,
        "fragmentation_penalty": frag,
        "fabrication_rule_penalty": rule,
        "total_penalty": total_penalty,
    }
    return fitness, metrics
