"""Attention map — the primary output type.

An AttentionMap wraps a float32 heatmap array and provides methods for
overlay rendering, region scoring, and focal-point extraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class AttentionMap:
    """A predicted visual attention heatmap.

    Wraps a float32 (H, W) array in [0, 1] along with metadata about the
    original image dimensions.
    """

    def __init__(
        self,
        heatmap: np.ndarray,
        original_size: tuple[int, int],
        config: dict[str, Any] | None = None,
    ) -> None:
        self._heatmap = heatmap.astype(np.float32)
        self._original_size = original_size
        self._config: dict[str, Any] = config or {}

    @property
    def heatmap(self) -> np.ndarray:
        """The attention heatmap as float32 (H, W) in [0, 1]."""
        return self._heatmap

    @property
    def original_size(self) -> tuple[int, int]:
        """Original image dimensions as (width, height)."""
        return self._original_size

    @property
    def config(self) -> dict[str, Any]:
        """Engine configuration used to produce this map."""
        return self._config

    def overlay(
        self,
        original: Image.Image,
        alpha: float = 0.6,
        colormap: str = "jet",
    ) -> Image.Image:
        """Overlay the heatmap on the original image.

        Args:
            original: Original PIL image.
            alpha: Blend factor (0 = original, 1 = heatmap only).
            colormap: Palette name (``"jet"`` or ``"turbo"``).

        Returns:
            PIL Image in RGB mode with the heatmap overlaid.
        """
        from .render import render_overlay

        return render_overlay(
            self._heatmap, original, self._original_size, alpha=alpha, colormap=colormap
        )

    def score(self, regions: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Score named regions on this attention map.

        Args:
            regions: Region strings (e.g. ``"cta:100,200,300,80"``).

        Returns:
            Tuple of ``(scored_regions, focal_points)``.
        """
        from .scoring import score_regions

        return score_regions(self, regions)

    def focal_points(self, n: int = 5) -> list[dict[str, Any]]:
        """Extract top-N focal points (local maxima).

        Args:
            n: Maximum number of focal points to return.

        Returns:
            List of dicts with x, y, value, rank.
        """
        from .scoring import find_focal_points

        return find_focal_points(self, n)

    def __repr__(self) -> str:
        h, w = self._heatmap.shape
        ow, oh = self._original_size
        return f"AttentionMap(size={w}x{h}, orig={ow}x{oh})"
