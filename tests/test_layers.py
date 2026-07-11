"""Tests for signal layers."""

import numpy as np
import pytest

from hotgaze.layers.center_bias import CenterBias
from hotgaze.layers.contrast import Contrast
from hotgaze.layers.gaze_flow import GazeFlow
from hotgaze.layers.saliency_fast import SaliencyFast

# ── Test helpers ──────────────────────────────────────────────────────────


def _structured_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Create a non-constant structured RGB test image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # White square in top-left
    img[20:80, 20:80] = 255
    # Gray bar across middle
    img[120:140, 50:200] = 128
    # Grid pattern
    for i in range(0, w, 32):
        img[:, i : i + 4] = 200
    return img


def _uniform_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Create a uniform gray image."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ── SaliencyFast ──────────────────────────────────────────────────────────


class TestSaliencyFast:
    def test_shape(self) -> None:
        img = _structured_image(128, 256)
        result = SaliencyFast().compute(img)
        assert result.shape == (128, 256)

    def test_dtype(self) -> None:
        img = _structured_image()
        result = SaliencyFast().compute(img)
        assert result.dtype == np.float32

    def test_range(self) -> None:
        img = _structured_image()
        result = SaliencyFast().compute(img)
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_deterministic(self) -> None:
        img = _structured_image()
        r1 = SaliencyFast().compute(img)
        r2 = SaliencyFast().compute(img)
        np.testing.assert_array_equal(r1, r2)

    def test_non_constant_on_structured(self) -> None:
        img = _structured_image()
        result = SaliencyFast().compute(img)
        assert result.max() - result.min() > 0.01, "Saliency map is nearly constant"

    def test_grayscale_input(self) -> None:
        """Grayscale (2D) input should work."""
        img = _structured_image()[:, :, 0]  # 2D
        result = SaliencyFast().compute(img)
        assert result.shape == (256, 256)


# ── Contrast ──────────────────────────────────────────────────────────────


class TestContrast:
    def test_shape(self) -> None:
        img = _structured_image(128, 256)
        result = Contrast().compute(img)
        assert result.shape == (128, 256)

    def test_dtype(self) -> None:
        img = _structured_image()
        result = Contrast().compute(img)
        assert result.dtype == np.float32

    def test_range(self) -> None:
        img = _structured_image()
        result = Contrast().compute(img)
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_deterministic(self) -> None:
        img = _structured_image()
        r1 = Contrast().compute(img)
        r2 = Contrast().compute(img)
        np.testing.assert_array_equal(r1, r2)

    def test_non_constant_on_structured(self) -> None:
        img = _structured_image()
        result = Contrast().compute(img)
        assert result.max() - result.min() > 0.01, "Contrast map is nearly constant"

    def test_uniform_zero(self) -> None:
        """Uniform image should produce near-zero contrast."""
        img = _uniform_image()
        result = Contrast().compute(img)
        assert result.max() < 0.01


# ── CenterBias ────────────────────────────────────────────────────────────


class TestCenterBias:
    def test_shape(self) -> None:
        img = _structured_image(128, 256)
        result = CenterBias().compute(img)
        assert result.shape == (128, 256)

    def test_dtype(self) -> None:
        img = _structured_image()
        result = CenterBias().compute(img)
        assert result.dtype == np.float32

    def test_range(self) -> None:
        img = _structured_image()
        result = CenterBias().compute(img)
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_deterministic(self) -> None:
        img = _structured_image()
        r1 = CenterBias().compute(img)
        r2 = CenterBias().compute(img)
        np.testing.assert_array_equal(r1, r2)

    def test_center_is_max(self) -> None:
        """Center pixel should have the highest value."""
        img = _structured_image()
        result = CenterBias().compute(img)
        h, w = result.shape
        cy, cx = h // 2, w // 2
        assert result[cy, cx] == pytest.approx(result.max(), abs=0.01)

    def test_anisotropic(self) -> None:
        """Gaussian should be wider than tall (horizontal spread > vertical)."""
        img = _structured_image(128, 256)
        result = CenterBias().compute(img)
        h, w = result.shape
        cy = h // 2
        # Find horizontal and vertical spread at half-max
        half_max = result[cy, w // 2] / 2
        h_spread = (result[cy] > half_max).sum()
        v_spread = (result[:, w // 2] > half_max).sum()
        assert h_spread > v_spread, f"h_spread={h_spread}, v_spread={v_spread}"


# ── GazeFlow ───────────────────────────────────────────────────────────────


class TestGazeFlow:
    def test_shape(self) -> None:
        img = _structured_image(128, 256)
        result = GazeFlow().compute(img)
        assert result.shape == (128, 256)

    def test_dtype(self) -> None:
        img = _structured_image()
        result = GazeFlow().compute(img)
        assert result.dtype == np.float32

    def test_range(self) -> None:
        img = _structured_image()
        result = GazeFlow().compute(img)
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_deterministic(self) -> None:
        img = _structured_image()
        r1 = GazeFlow().compute(img)
        r2 = GazeFlow().compute(img)
        np.testing.assert_array_equal(r1, r2)

    def test_top_left_is_max(self) -> None:
        """Top-left corner should be among the highest values."""
        img = _structured_image()
        result = GazeFlow().compute(img)
        # The F-band may shift the absolute peak slightly, but top-left
        # should be very close to the maximum.
        assert result[0, 0] >= result.max() * 0.85

    def test_image_independent(self) -> None:
        """GazeFlow is a static prior — same output regardless of image content."""
        img1 = _structured_image()
        img2 = _uniform_image()
        r1 = GazeFlow().compute(img1)
        r2 = GazeFlow().compute(img2)
        np.testing.assert_array_equal(r1, r2)


# ── Shared interface tests ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "layer",
    [
        SaliencyFast(),
        Contrast(),
        CenterBias(),
        GazeFlow(),
    ],
)
def test_layer_interface(layer: object) -> None:
    """Every layer satisfies the SignalLayer contract."""
    img = _structured_image(128, 200)
    result = layer.compute(img)  # type: ignore[union-attr]
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (128, 200)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() <= result.max() <= 1.0
