"""Pydantic configuration models for HotGaze engine."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LayerWeights(BaseModel):
    """Weights for each attention signal layer in a backend."""

    model_config = ConfigDict(validate_assignment=True)

    saliency: float = 0.5
    contrast: float = 0.2
    center_bias: float = 0.2
    gaze_flow: float = 0.1
    faces: float = 0.0

    @field_validator("saliency", "contrast", "center_bias", "gaze_flow", "faces")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        """Require every layer weight to be finite and non-negative."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("layer weights must be finite and non-negative")
        return value


class EngineConfig(BaseModel):
    """Configuration for the attention engine."""

    backend: Literal["fast", "deep"] = "fast"
    working_long_edge: int = Field(default=1024, ge=256, le=4096)
    smooth_sigma: float = Field(default=5.0, ge=0.0, le=50.0)
    weights: LayerWeights = Field(default_factory=LayerWeights)
    extra_layers: list[str] = Field(default_factory=list)

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
