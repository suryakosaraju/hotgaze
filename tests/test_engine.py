"""Tests for engine and AttentionMap."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from hotgaze.attention_map import AttentionMap
from hotgaze.config import EngineConfig, LayerWeights
from hotgaze.engine import run_engine


def _make_test_image(w: int = 200, h: int = 150) -> Image.Image:
    """Create a structured test image."""
    img = Image.new("RGB", (w, h), (128, 128, 128))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 80, 80], fill=(255, 255, 255))
    draw.rectangle([120, 50, 180, 100], fill=(0, 0, 0))
    draw.text((10, 110), "Test UI", fill=(255, 0, 0))
    return img


def _save_temp(img: Image.Image) -> str:
    """Save image to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        return tmp.name


# ── Engine ─────────────────────────────────────────────────────────────────


class TestEngine:
    def test_run_produces_attention_map(self) -> None:
        img = _make_test_image()
        path = _save_temp(img)
        try:
            result = run_engine(path)
            assert result.heatmap.shape == (150, 200)
            assert result.heatmap.dtype == np.float32
            assert result.original_size == (200, 150)
        finally:
            Path(path).unlink()

    def test_heatmap_range(self) -> None:
        img = _make_test_image()
        path = _save_temp(img)
        try:
            result = run_engine(path)
            hm = result.heatmap
            assert 0.0 <= hm.min() <= hm.max() <= 1.0
        finally:
            Path(path).unlink()

    def test_rgba_input(self) -> None:
        """RGBA images should be flattened over white."""
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.rectangle([30, 30, 70, 70], fill=(255, 0, 0, 255))
        path = _save_temp(img)
        try:
            result = run_engine(path)
            assert result.heatmap.shape == (100, 100)
        finally:
            Path(path).unlink()

    def test_grayscale_input(self) -> None:
        """Grayscale (L mode) images should work."""
        img = Image.new("L", (100, 100), 128)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.rectangle([30, 30, 70, 70], fill=255)
        path = _save_temp(img)
        try:
            result = run_engine(path)
            assert result.heatmap.shape == (100, 100)
        finally:
            Path(path).unlink()

    def test_downscale_large_image(self) -> None:
        """Image wider than working_long_edge should be downscaled."""
        img = _make_test_image(3000, 2000)
        path = _save_temp(img)
        try:
            config = EngineConfig.fast_default()
            config.working_long_edge = 1024
            result = run_engine(path, config=config)
            # Output should match original size
            assert result.original_size == (3000, 2000)
            assert result.heatmap.shape == (2000, 3000)
        finally:
            Path(path).unlink()

    def test_no_upscale_small_image(self) -> None:
        """Image smaller than working_long_edge should not be upscaled."""
        img = _make_test_image(100, 80)
        path = _save_temp(img)
        try:
            config = EngineConfig.fast_default()
            config.working_long_edge = 1024
            result = run_engine(path, config=config)
            assert result.original_size == (100, 80)
            assert result.heatmap.shape == (80, 100)
        finally:
            Path(path).unlink()

    def test_deterministic(self) -> None:
        """Same input + config → identical output."""
        img = _make_test_image()
        path = _save_temp(img)
        try:
            r1 = run_engine(path)
            r2 = run_engine(path)
            np.testing.assert_array_equal(r1.heatmap, r2.heatmap)
        finally:
            Path(path).unlink()


# ── AttentionMap ───────────────────────────────────────────────────────────


class TestAttentionMap:
    def test_overlay_size_matches_original(self) -> None:
        hm = np.random.rand(100, 200).astype(np.float32)
        am = AttentionMap(hm, (200, 100))
        original = Image.new("RGB", (200, 100), (128, 128, 128))
        overlay = am.overlay(original)
        assert overlay.size == (200, 100)
        assert overlay.mode == "RGB"

    def test_overlay_resizes_heatmap(self) -> None:
        """Overlay should resize heatmap to original size if different."""
        hm = np.random.rand(50, 100).astype(np.float32)
        am = AttentionMap(hm, (200, 100))
        original = Image.new("RGB", (200, 100), (128, 128, 128))
        overlay = am.overlay(original)
        assert overlay.size == (200, 100)

    def test_properties(self) -> None:
        hm = np.ones((100, 200), dtype=np.float32) * 0.5
        am = AttentionMap(hm, (200, 100), {"backend": "fast"})
        assert am.original_size == (200, 100)
        assert am.config == {"backend": "fast"}
        np.testing.assert_array_equal(am.heatmap, hm)


# ── Config ─────────────────────────────────────────────────────────────────


class TestConfig:
    def test_fast_default(self) -> None:
        cfg = EngineConfig.fast_default()
        assert cfg.backend == "fast"
        assert cfg.weights.saliency == 0.5
        assert cfg.weights.contrast == 0.2
        assert cfg.weights.center_bias == 0.2
        assert cfg.weights.gaze_flow == 0.1

    def test_deep_default(self) -> None:
        cfg = EngineConfig.deep_default()
        assert cfg.backend == "deep"
        assert cfg.weights.saliency == 0.7
        assert cfg.weights.center_bias == 0.2
        assert cfg.weights.gaze_flow == 0.1
        # Contrast should be zero in deep mode
        assert cfg.weights.contrast == 0.0

    def test_custom_weights(self) -> None:
        cfg = EngineConfig(
            backend="fast",
            weights=LayerWeights(saliency=0.8, contrast=0.1, gaze_flow=0.1),
        )
        assert cfg.weights.saliency == 0.8
        assert cfg.weights.contrast == 0.1
