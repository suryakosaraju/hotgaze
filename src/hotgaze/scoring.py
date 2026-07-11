"""Region scoring and focal point extraction (T2.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .attention_map import AttentionMap


def score_regions(attention_map: AttentionMap, regions: list[Any]) -> list[Any]:
    """Score named regions on an attention map. Placeholder for T2.1."""
    raise NotImplementedError("Region scoring will be implemented in T2.1")


def find_focal_points(attention_map: AttentionMap, n: int = 5) -> list[Any]:
    """Find top-N focal points. Placeholder for T2.1."""
    raise NotImplementedError("Focal point extraction will be implemented in T2.1")
