"""Local contrast / edge-energy attention layer.

High-contrast edges (text, borders, icons) attract visual attention.
Uses Sobel gradient magnitude as a proxy for local contrast energy.
"""

import numpy as np

from .._imageops import conv2d, to_grayscale
from .._imageops import gaussian_blur as _gaussian_blur
from .base import SignalLayer


class Contrast(SignalLayer):
    """Local contrast / edge-energy layer.

    Computes Sobel gradient magnitude as a proxy for visual contrast
    energy, which correlates with attention-drawing power.
    """

    def compute(self, img: np.ndarray) -> np.ndarray:
        gray = to_grayscale(img, dtype=np.uint8)
        gx = _sobel_x(gray)
        gy = _sobel_y(gray)
        magnitude = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)

        # Smooth to spread edge energy into regions
        h, w = magnitude.shape
        sigma = max(h, w) / 80.0
        magnitude = _gaussian_blur(magnitude, sigma)

        # Normalize to [0, 1]
        mn, mx = magnitude.min(), magnitude.max()
        if mx - mn < 1e-10:
            return np.zeros_like(magnitude, dtype=np.float32)
        result = (magnitude - mn) / (mx - mn)
        return result.astype(np.float32)


_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def _sobel_x(img: np.ndarray) -> np.ndarray:
    """Sobel X gradient via convolution."""
    return conv2d(img.astype(np.float32), _SOBEL_X)


def _sobel_y(img: np.ndarray) -> np.ndarray:
    """Sobel Y gradient via convolution."""
    return conv2d(img.astype(np.float32), _SOBEL_Y)
