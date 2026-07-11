"""Anisotropic Gaussian center-bias attention prior.

Viewers tend to look near the center of a screen first. This layer
creates a 2D anisotropic Gaussian (wider than tall, since screens are
landscape-oriented) centered on the image.
"""

import numpy as np

from .base import SignalLayer


class CenterBias(SignalLayer):
    """Anisotropic Gaussian center-bias prior.

    Creates a 2D Gaussian attention prior centered on the image.
    The Gaussian is wider than tall (sigma_x > sigma_y) to reflect
    the typical landscape aspect ratio of screens.
    """

    def __init__(
        self,
        sigma_x_ratio: float = 0.35,
        sigma_y_ratio: float = 0.25,
    ) -> None:
        """Args:
        sigma_x_ratio: Horizontal sigma as fraction of image width.
        sigma_y_ratio: Vertical sigma as fraction of image height.
        """
        self._sigma_x_ratio = sigma_x_ratio
        self._sigma_y_ratio = sigma_y_ratio

    def compute(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        sigma_x = w * self._sigma_x_ratio
        sigma_y = h * self._sigma_y_ratio

        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        y, x = np.mgrid[0:h, 0:w]
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        gauss = np.exp(-(((x - cx) ** 2) / (2 * sigma_x**2) + ((y - cy) ** 2) / (2 * sigma_y**2)))

        # Normalize to [0, 1]
        gauss = (gauss - gauss.min()) / (gauss.max() - gauss.min())
        return gauss.astype(np.float32)
