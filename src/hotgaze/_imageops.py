"""Shared image-processing utilities.

Single-source implementations used across layers and the engine.
No cross-module private imports — everything imports from here.
"""

from __future__ import annotations

import numpy as np


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur via FFT convolution.

    Args:
        arr: 2D float32 array.
        sigma: Gaussian sigma. Values < 0.5 return arr unchanged.

    Returns:
        Blurred array, same shape and dtype as input.
    """
    if sigma < 0.5:
        return arr

    kernel = _gaussian_kernel(sigma)
    from numpy.fft import fft2, ifft2

    h, w = arr.shape

    # Convolve rows
    k1d = np.zeros(w, dtype=np.float32)
    k1d[: len(kernel)] = kernel
    k1d = np.roll(k1d, -(len(kernel) // 2))
    k1d_f = fft2(k1d.reshape(1, -1), s=(h, w))
    arr_f = fft2(arr)
    arr = np.real(ifft2(arr_f * k1d_f))

    # Convolve columns
    k1d_h = np.zeros(h, dtype=np.float32)
    k1d_h[: len(kernel)] = kernel
    k1d_h = np.roll(k1d_h, -(len(kernel) // 2))
    arr_f = fft2(arr)
    k1d_f = fft2(k1d_h.reshape(-1, 1), s=(h, w))
    arr = np.real(ifft2(arr_f * k1d_f))

    return arr


def conv2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution with 'same' padding via FFT.

    Args:
        arr: 2D float32 array.
        kernel: 2D float32 kernel.

    Returns:
        Convolved array, same shape as arr.
    """
    from numpy.fft import fft2, ifft2

    kh, kw = kernel.shape
    h, w = arr.shape
    kernel_padded = np.zeros_like(arr)
    kernel_padded[:kh, :kw] = kernel
    kernel_padded = np.roll(kernel_padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    result = np.real(ifft2(fft2(arr) * fft2(kernel_padded)))
    return result


def to_grayscale(img: np.ndarray, dtype: type = np.float32) -> np.ndarray:
    """Convert an RGB uint8 image to grayscale using BT.601 weights.

    Args:
        img: RGB uint8 image (H, W, 3) or already-grayscale (H, W).
        dtype: Output dtype — ``np.float32`` (default) or ``np.uint8``.

    Returns:
        Grayscale array of the requested dtype.
    """
    if img.ndim == 2:
        return img.astype(dtype)
    gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    return gray.astype(dtype)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """1D Gaussian kernel."""
    size = int(4 * sigma + 1) | 1
    x = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()
