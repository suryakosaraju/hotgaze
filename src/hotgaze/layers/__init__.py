"""Attention signal layers for HotGaze."""

from .base import SignalLayer
from .center_bias import CenterBias
from .contrast import Contrast
from .faces import Faces
from .gaze_flow import GazeFlow
from .saliency_fast import SaliencyFast

__all__ = [
    "SignalLayer",
    "SaliencyFast",
    "Contrast",
    "CenterBias",
    "GazeFlow",
    "Faces",
]

# SaliencyDeep is lazy-imported to keep torch optional
try:
    from .saliency_deep import SaliencyDeep  # noqa: F401

    __all__.append("SaliencyDeep")
except ImportError:
    pass
