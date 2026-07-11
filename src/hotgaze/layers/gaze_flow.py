"""F/Z-pattern reading attention prior.

In left-to-right reading cultures, visual attention follows an F-shaped
or Z-shaped scanning pattern: top row gets the most attention, then a
horizontal band below, then the left edge. This layer creates a static
prior that models this reading pattern.
"""

import numpy as np

from .base import SignalLayer


class GazeFlow(SignalLayer):
    """F/Z-pattern reading prior.

    Models the top-left-weighted scanning pattern typical in LTR reading
    cultures. Attention is highest at the top-left and decays toward the
    bottom-right, with a secondary horizontal band in the upper portion.
    """

    def __init__(self, decay_rate: float = 1.5) -> None:
        """Args:
        decay_rate: How quickly attention falls off from top-left.
            Higher = faster decay.
        """
        self._decay_rate = decay_rate

    def compute(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        y, x = np.mgrid[0:h, 0:w]
        x_norm = x.astype(np.float32) / max(w - 1, 1)
        y_norm = y.astype(np.float32) / max(h - 1, 1)

        # Diagonal distance from top-left (Euclidean in normalized coords)
        dist = np.sqrt(x_norm**2 + y_norm**2)

        # Exponential decay from top-left
        prior = np.exp(-self._decay_rate * dist)

        # Add F-pattern horizontal band: boost the top ~30% of the image
        f_band = np.exp(-((y_norm - 0.15) ** 2) / (2 * 0.1**2))
        prior = 0.7 * prior + 0.3 * f_band

        # Normalize to [0, 1]
        prior = (prior - prior.min()) / (prior.max() - prior.min())
        return prior.astype(np.float32)
