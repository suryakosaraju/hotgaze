"""Spectral residual saliency — hand-implemented in numpy FFT.

Based on Hou & Zhang (2007): "Saliency Detection: A Spectral Residual Approach."
Implemented directly with numpy FFT — NOT using cv2.saliency (requires the
contrib package we don't ship).
"""

import numpy as np

from .._imageops import conv2d as _conv2d
from .._imageops import gaussian_blur as _gaussian_blur
from .._imageops import to_grayscale
from .base import SignalLayer


class SaliencyFast(SignalLayer):
    """Spectral residual saliency layer.

    Computes the spectral residual of the log-amplitude spectrum, which
    highlights visually salient (non-redundant) regions in an image.
    """

    def __init__(self, avg_filter_size: int = 3) -> None:
        self._avg_filter_size = avg_filter_size

    def compute(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        gray = to_grayscale(img)
        gray_f = gray.astype(np.float32) / 255.0

        # 2D FFT
        fft = np.fft.fft2(gray_f)
        fft_shifted = np.fft.fftshift(fft)

        # Log amplitude spectrum
        log_amplitude = np.log(np.abs(fft_shifted) + 1e-12)

        # Local average of log amplitude (mean filter)
        kernel = np.ones((self._avg_filter_size, self._avg_filter_size), dtype=np.float32)
        kernel /= kernel.sum()
        avg_log_amplitude = _conv2d(log_amplitude, kernel)

        # Spectral residual
        spectral_residual = log_amplitude - avg_log_amplitude

        # Reconstruct: combine residual phase with original phase
        phase = np.angle(fft_shifted)
        reconstructed = np.exp(spectral_residual + 1j * phase)
        ifft_shifted = np.fft.ifftshift(reconstructed)
        saliency_map = np.abs(np.fft.ifft2(ifft_shifted))

        # Gaussian smooth
        sigma = max(h, w) / 50.0
        saliency_map = _gaussian_blur(saliency_map, sigma)

        # Normalize to [0, 1]
        saliency_map = _minmax_norm(saliency_map)

        return saliency_map.astype(np.float32)


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] range."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)
