"""EGGROLL hard-binary inverse-design utilities for pixelated antennas."""

from .antenna_ops import (
    AREA_TARGET,
    FEED_COL,
    FEED_ROWS,
    area_penalty,
    fabrication_rule_penalty,
    feed_connectivity_penalty,
    force_feed_pixels,
    fragmentation_penalty,
    hard_threshold_design,
    mask_uniqueness,
    metal_fill_ratio,
)
from .config import DEFAULT_LAYOUT, GeneratorConfig, LayoutSpec, ScorerConfig, parse_layout
from .fitness import FitnessConfig, compute_fitness
from .checkpoints import load_inverse_npz, save_inverse_npz
from .pngf import (
    PNGF_FREQ_HZ,
    PNGF_LAYOUT,
    PNGF_TARGET_DIM,
    hard_project_pngf_center_fed,
    pack_pngf_targets,
    pngf_target_errors,
    project_pngf_center_fed_mask_np,
)
from .scorers import ExternalEMScorer, PNGFScorer, SurrogateScorer

__all__ = [
    "AREA_TARGET",
    "FEED_COL",
    "FEED_ROWS",
    "DEFAULT_LAYOUT",
    "FitnessConfig",
    "GeneratorConfig",
    "LayoutSpec",
    "PNGF_FREQ_HZ",
    "PNGF_LAYOUT",
    "PNGFScorer",
    "PNGF_TARGET_DIM",
    "ScorerConfig",
    "ExternalEMScorer",
    "SurrogateScorer",
    "area_penalty",
    "compute_fitness",
    "fabrication_rule_penalty",
    "feed_connectivity_penalty",
    "force_feed_pixels",
    "fragmentation_penalty",
    "hard_threshold_design",
    "hard_project_pngf_center_fed",
    "load_inverse_npz",
    "mask_uniqueness",
    "metal_fill_ratio",
    "pack_pngf_targets",
    "parse_layout",
    "pngf_target_errors",
    "project_pngf_center_fed_mask_np",
    "save_inverse_npz",
]
