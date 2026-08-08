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
    row_kernel = _fit_1d_kernel(kernel, w)
    k1d = np.zeros(w, dtype=np.float32)
    k1d[: len(row_kernel)] = row_kernel
    k1d = np.roll(k1d, -(len(row_kernel) // 2))
    k1d_f = fft2(k1d.reshape(1, -1), s=(h, w))
    arr_f = fft2(arr)
    arr = np.real(ifft2(arr_f * k1d_f))

    # Convolve columns
    column_kernel = _fit_1d_kernel(kernel, h)
    k1d_h = np.zeros(h, dtype=np.float32)
    k1d_h[: len(column_kernel)] = column_kernel
    k1d_h = np.roll(k1d_h, -(len(column_kernel) // 2))
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

    h, w = arr.shape
    kernel = _fit_2d_kernel(kernel, (h, w))
    kh, kw = kernel.shape
    kernel_padded = np.zeros_like(arr)
    kernel_padded[:kh, :kw] = kernel
    kernel_padded = np.roll(kernel_padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    result = np.real(ifft2(fft2(arr) * fft2(kernel_padded)))
    return result


def _fit_1d_kernel(kernel: np.ndarray, size: int) -> np.ndarray:
    """Center-crop a one-dimensional kernel when an image axis is smaller."""
    if len(kernel) <= size:
        return kernel
    start = (len(kernel) - size) // 2
    cropped = kernel[start : start + size]
    total = float(cropped.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("Cannot normalize a non-positive Gaussian kernel")
    normalized = (cropped / total).astype(np.float32, copy=False)
    # Correct the final representable bit so the float32 kernel still sums to 1.
    normalized[-1] = np.float32(1.0 - normalized[:-1].sum(dtype=np.float32))
    return normalized


def _fit_2d_kernel(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Center-crop a two-dimensional kernel to fit an image shape."""
    h, w = shape
    kh, kw = kernel.shape
    y0 = max((kh - h) // 2, 0)
    x0 = max((kw - w) // 2, 0)
    return kernel[y0 : y0 + min(kh, h), x0 : x0 + min(kw, w)]


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
