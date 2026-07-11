"""Pydantic configuration models for HotGaze engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LayerWeights(BaseModel):
    """Weights for each attention signal layer in a backend."""

    saliency: float = 0.5
    contrast: float = 0.2
    center_bias: float = 0.2
    gaze_flow: float = 0.1
    text: float = 0.0
    faces: float = 0.0


class EngineConfig(BaseModel):
    """Configuration for the attention engine."""

    backend: Literal["fast", "deep"] = "fast"
    working_long_edge: int = Field(default=1024, ge=256, le=4096)
    smooth_sigma: float = Field(default=5.0, ge=0.0, le=50.0)
    weights: LayerWeights = Field(default_factory=LayerWeights)

    @classmethod
    def fast_default(cls) -> EngineConfig:
        """Default config for the fast (heuristic) backend."""
        return cls(
            backend="fast",
            weights=LayerWeights(
                saliency=0.5,
                contrast=0.2,
                center_bias=0.2,
                gaze_flow=0.1,
            ),
        )

    @classmethod
    def deep_default(cls) -> EngineConfig:
        """Default config for the deep backend."""
        return cls(
            backend="deep",
            weights=LayerWeights(
                saliency=0.7,
                contrast=0.0,
                center_bias=0.2,
                gaze_flow=0.1,
            ),
        )
