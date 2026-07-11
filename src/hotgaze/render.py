"""Colormap overlay PNG rendering.

Handles palette construction (jet, turbo) and compositing heatmaps
onto original images. Called by AttentionMap.overlay().
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# ── Public API ───────────────────────────────────────────────────────────────


def render_overlay(
    heatmap: np.ndarray,
    original: Image.Image,
    original_size: tuple[int, int],
    alpha: float = 0.6,
    colormap: str = "jet",
) -> Image.Image:
    """Overlay a heatmap on an original image with a colormap.

    Args:
        heatmap: Float32 (H, W) array in [0, 1].
        original: Original PIL image.
        original_size: Target (width, height) for output.
        alpha: Blend factor (0 = original only, 1 = heatmap only).
        colormap: Palette name — ``"jet"`` or ``"turbo"``.

    Returns:
        PIL Image in RGB mode.
    """
    w, h = original_size

    # Resize heatmap to original dimensions if needed
    if heatmap.shape != (h, w):
        hm_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        hm_img = hm_img.resize((w, h), Image.Resampling.LANCZOS)
        hm = np.array(hm_img, dtype=np.float32) / 255.0
    else:
        hm = heatmap

    gray_img = Image.fromarray((hm * 255).astype(np.uint8), mode="L")
    colored = _apply_colormap(gray_img, colormap)

    original_rgb = original.convert("RGB").resize((w, h))
    colored_rgb = colored.convert("RGB") if colored.mode != "RGB" else colored

    return Image.blend(original_rgb, colored_rgb, alpha)


# ── Palette cache ────────────────────────────────────────────────────────────

_JET: list[int] | None = None
_TURBO: list[int] | None = None


def _apply_colormap(gray: Image.Image, name: str) -> Image.Image:
    """Apply a named colormap to a grayscale image."""
    palette = _get_turbo_palette() if name == "turbo" else _get_jet_palette()
    result = gray.convert("P")
    result.putpalette(palette)
    return result.convert("RGB")


def _get_jet_palette() -> list[int]:
    global _JET
    if _JET is None:
        _JET = _build_jet_palette()
    return _JET


def _get_turbo_palette() -> list[int]:
    global _TURBO
    if _TURBO is None:
        _TURBO = _build_turbo_palette()
    return _TURBO


# ── JET palette ──────────────────────────────────────────────────────────────


def _build_jet_palette() -> list[int]:
    palette: list[int] = []
    for i in range(256):
        v = i / 255.0
        r = int(min(255, max(0, _jet_red(v) * 255)))
        g = int(min(255, max(0, _jet_green(v) * 255)))
        b = int(min(255, max(0, _jet_blue(v) * 255)))
        palette.extend([r, g, b])
    return palette


def _jet_red(v: float) -> float:
    if v < 0.25:
        return 0.0
    if v < 0.55:
        return (v - 0.25) / 0.3
    if v < 0.85:
        return 1.0
    return max(0.0, (1.0 - v) / 0.15)


def _jet_green(v: float) -> float:
    if v < 0.1:
        return 0.0
    if v < 0.4:
        return (v - 0.1) / 0.3
    if v < 0.7:
        return 1.0
    if v < 0.9:
        return 1.0 - (v - 0.7) / 0.2
    return 0.0


def _jet_blue(v: float) -> float:
    if v < 0.15:
        return 0.5 + v / 0.3
    if v < 0.45:
        return 1.0
    if v < 0.65:
        return 1.0 - (v - 0.45) / 0.2
    return 0.0


# ── TURBO palette ────────────────────────────────────────────────────────────


def _build_turbo_palette() -> list[int]:
    """Build a 256-entry Turbo colormap palette.

    Turbo is perceptually uniform and colorblind-friendly.
    Based on the reference implementation by Anton Mikhailov (Google).
    """
    # Turbo control points (R, G, B) × 256 positions
    palette: list[int] = []
    for i in range(256):
        t = i / 255.0
        r = int(min(255, max(0, _turbo_red(t) * 255)))
        g = int(min(255, max(0, _turbo_green(t) * 255)))
        b = int(min(255, max(0, _turbo_blue(t) * 255)))
        palette.extend([r, g, b])
    return palette


def _turbo_red(t: float) -> float:
    """Turbo red channel, piecewise polynomial."""
    x = t
    if x < 0.125:
        return 0.0
    if x < 0.375:
        return (x - 0.125) / 0.25
    if x < 0.625:
        return 1.0
    if x < 0.875:
        return 1.0 - (x - 0.625) / 0.25
    return 0.0


def _turbo_green(t: float) -> float:
    x0, y0 = 0.005, 0.135
    x1, y1 = 0.500, 0.865
    x2, y2 = 0.125, 1.0
    x3, y3 = 0.420, 1.0
    x4, y4 = 0.500, 0.865
    x5, y5 = 0.875, 0.0

    if t < x0:
        return y0
    if t < x1:
        return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
    if t < x2:
        return y1
    if t < x3:
        return y1 + (y2 - y1) * (t - x2) / (x3 - x2) if x3 > x2 else y2
    if t < x4:
        return y3 + (y4 - y3) * (t - x3) / (x4 - x3) if x4 > x3 else y4
    if t < x5:
        return y5
    return 0.0


def _turbo_blue(t: float) -> float:
    _, y0 = 0.500, 0.135
    x1, y1 = 0.125, 0.865
    x2, y2 = 0.375, 1.0
    x3, y3 = 0.500, 0.865
    x4, y4 = 0.625, 0.0

    if t < x1:
        return y0 + (y1 - y0) * t / x1
    if t < x2:
        return y1
    if t < x3:
        return y1 + (y2 - y1) * (t - x2) / (x3 - x2) if x3 > x2 else y2
    if t < x4:
        return y3 + (y4 - y3) * (t - x3) / (x4 - x3) if x4 > x3 else y4
    return 0.0
