"""Typed configuration for layout-agnostic EGGROLL antenna search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, Union


FeedPixel = Union[tuple[int, int], tuple[int, int, int]]


@dataclass(frozen=True)
class LayoutSpec:
    """Binary mask geometry and fixed-feed constraints.

    Feed pixels are `(row, col)` pairs applied to layer 0, or explicit
    `(layer, row, col)` triples for multilayer layouts.
    """

    height: int
    width: int
    layers: int = 1
    feed_pixels: tuple[FeedPixel, ...] = ((5, 0), (6, 0))
    output_channels: int = 1

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.output_channels <= 0:
            raise ValueError("output_channels must be positive")
        for pixel in self.feed_pixels:
            layer, row, col = self.normalize_feed_pixel(pixel)
            if not (0 <= layer < self.layers and 0 <= row < self.height and 0 <= col < self.width):
                raise ValueError(f"feed pixel {pixel} is outside layout {self}")

    @property
    def flat_size(self) -> int:
        return self.layers * self.height * self.width

    @property
    def mask_shape(self) -> tuple[int, int, int]:
        return (self.layers, self.height, self.width)

    @staticmethod
    def normalize_feed_pixel(pixel: FeedPixel) -> tuple[int, int, int]:
        if len(pixel) == 2:
            row, col = pixel
            return (0, int(row), int(col))
        if len(pixel) == 3:
            layer, row, col = pixel
            return (int(layer), int(row), int(col))
        raise ValueError(f"feed pixels must be row/col pairs or layer/row/col triples, got {pixel}")


DEFAULT_LAYOUT = LayoutSpec(height=12, width=12)


@dataclass(frozen=True)
class GeneratorConfig:
    kind: Literal["mlp", "cnn"]
    layout: LayoutSpec = DEFAULT_LAYOUT
    hidden_dims: tuple[int, ...] = (1054, 512)
    latent_dim: int = 0
    spectrum_dim: int = 81


@dataclass(frozen=True)
class ScorerConfig:
    kind: Literal["surrogate", "external-em", "pngf"]
    checkpoint_path: Path | None = None
    command: str | None = None
    work_dir: Path | None = None
    solver_mode: Literal["air", "substrate"] = "air"
    cache_dir: Path | None = None
    timeout_seconds: float | None = None
    bad_spectrum_value: float = 1.0e6


def parse_layout(value: str, layers: int = 1) -> LayoutSpec:
    """Parse `HxW`, `H,W`, or a square size like `32` into a LayoutSpec."""

    normalized = value.lower().replace(",", "x")
    parts = normalized.split("x")
    if len(parts) == 1:
        height = width = int(parts[0])
    elif len(parts) == 2:
        height, width = (int(part) for part in parts)
    else:
        raise ValueError(f"Expected layout like 12x12, 32x32, or 64; got {value!r}")
    return LayoutSpec(height=height, width=width, layers=layers, output_channels=layers)


def parse_hidden_dims(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        dims = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    else:
        dims = tuple(int(x) for x in value)
    if not dims:
        raise ValueError("hidden_dims must contain at least one layer size")
    return dims
