"""Base class for attention signal layers."""

from abc import ABC, abstractmethod

import numpy as np


class SignalLayer(ABC):
    """Abstract base for an attention signal layer.

    Each layer receives an RGB image at working resolution and returns
    a float32 (H, W) array in [0, 1] representing predicted attention.

    Layers must be deterministic: same input → same output.
    """

    @abstractmethod
    def compute(self, img: np.ndarray) -> np.ndarray:
        """Compute attention signal from an image.

        Args:
            img: RGB image as uint8 (H, W, 3) at working resolution.

        Returns:
            Float32 (H, W) array with values in [0, 1].
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
