"""Tests for engine and AttentionMap."""

import logging
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps
from pydantic import ValidationError

from hotgaze._imageops import _fit_1d_kernel, gaussian_blur
from hotgaze.attention_map import AttentionMap
from hotgaze.config import EngineConfig, LayerWeights
from hotgaze.engine import _load_image, run_engine
from hotgaze.layers.base import SignalLayer
from hotgaze.scoring import scores_to_json


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


class _PatternLayer(SignalLayer):
    """Valid deterministic layer used to test engine failure isolation."""

    def compute(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)


class _FailingLayer(SignalLayer):
    def compute(self, img: np.ndarray) -> np.ndarray:
        raise RuntimeError("synthetic layer failure")


class _OutputLayer(SignalLayer):
    """Test layer that can return a deliberately invalid contract value."""

    def __init__(self, output: object) -> None:
        self._output = output

    def compute(self, img: np.ndarray) -> np.ndarray:
        return self._output  # type: ignore[return-value]


class _ConstantLayer(SignalLayer):
    def __init__(self, value: float) -> None:
        self._value = value

    def compute(self, img: np.ndarray) -> np.ndarray:
        return np.full(img.shape[:2], self._value, dtype=np.float32)


# ── Engine ─────────────────────────────────────────────────────────────────


class TestEngine:
    def test_center_cropped_gaussian_kernel_remains_normalized(self) -> None:
        """Cropping a Gaussian for a tiny axis preserves constant maps."""
        kernel = np.ones(21, dtype=np.float32) / 21.0
        cropped = _fit_1d_kernel(kernel, 10)

        assert cropped.shape == (10,)
        assert np.sum(cropped, dtype=np.float64) == pytest.approx(1.0, abs=1e-7)
        constant = np.full((10, 20), 0.5, dtype=np.float32)
        np.testing.assert_allclose(gaussian_blur(constant, 5.0), constant, atol=1e-6)

    def test_failed_layer_is_diagnosed_without_changing_valid_output(self, caplog) -> None:
        """A failed layer is logged and skipped while valid output is preserved."""
        path = _save_temp(_make_test_image(20, 10))
        try:
            config = EngineConfig.fast_default()
            expected = run_engine(path, config=config, layers={"contrast": _PatternLayer()})
            with caplog.at_level(logging.WARNING, logger="hotgaze.engine"):
                actual = run_engine(
                    path,
                    config=config,
                    layers={"saliency": _FailingLayer(), "contrast": _PatternLayer()},
                )

            np.testing.assert_array_equal(actual.heatmap, expected.heatmap)
            expected_regions, expected_focal = expected.score(["all:0,0,20,10"])
            actual_regions, actual_focal = actual.score(["all:0,0,20,10"])
            expected_json = scores_to_json(
                "score",
                path,
                expected.original_size,
                expected.working_size,
                config.model_dump(),
                expected_regions,
                expected_focal,
            )
            actual_json = scores_to_json(
                "score",
                path,
                actual.original_size,
                actual.working_size,
                config.model_dump(),
                actual_regions,
                actual_focal,
            )
            assert actual_json == expected_json
            assert "saliency" in caplog.text
            assert "synthetic layer failure" in caplog.text
        finally:
            Path(path).unlink()

    @pytest.mark.parametrize(
        "output_factory",
        [
            pytest.param(lambda shape: np.zeros(shape, dtype=np.float64), id="float64"),
            pytest.param(lambda shape: np.zeros(shape, dtype=np.uint8), id="uint8"),
            pytest.param(lambda shape: np.zeros(shape, dtype=bool), id="bool"),
            pytest.param(
                lambda shape: np.zeros((shape[0], shape[1] + 1), dtype=np.float32),
                id="wrong-shape",
            ),
            pytest.param(lambda shape: [[0.0]], id="non-array"),
            pytest.param(lambda shape: np.full(shape, np.nan, dtype=np.float32), id="nan"),
            pytest.param(lambda shape: np.full(shape, np.inf, dtype=np.float32), id="inf"),
            pytest.param(lambda shape: np.full(shape, -0.01, dtype=np.float32), id="below-zero"),
            pytest.param(lambda shape: np.full(shape, 1.01, dtype=np.float32), id="above-one"),
        ],
    )
    def test_invalid_layer_maps_are_logged_and_skipped(self, output_factory, caplog) -> None:
        """Every layer-contract violation is diagnosed and excluded from blending."""
        path = _save_temp(_make_test_image(20, 10))
        try:
            shape = (10, 20)
            expected = run_engine(
                path,
                config=EngineConfig.fast_default(),
                layers={"contrast": _PatternLayer()},
            )
            with caplog.at_level(logging.WARNING, logger="hotgaze.engine"):
                result = run_engine(
                    path,
                    config=EngineConfig.fast_default(),
                    layers={
                        "saliency": _OutputLayer(output_factory(shape)),
                        "contrast": _PatternLayer(),
                    },
                )

            np.testing.assert_array_equal(result.heatmap, expected.heatmap)
            warning_messages = [
                record.getMessage()
                for record in caplog.records
                if record.name == "hotgaze.engine" and record.levelno >= logging.WARNING
            ]
            assert any(
                "saliency" in message and "skipped" in message for message in warning_messages
            )
            assert len(warning_messages) == 1
        finally:
            Path(path).unlink()

    @pytest.mark.parametrize("size", [(1, 1), (10, 10), (20, 10)])
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_final_heatmap_contract_for_constant_float32_layer(
        self, size: tuple[int, int], value: float
    ) -> None:
        """Blending and FFT smoothing cannot leak precision outside [0, 1]."""
        path = _save_temp(Image.new("RGB", size, (128, 64, 32)))
        try:
            result = run_engine(
                path,
                config=EngineConfig.fast_default(),
                layers={"contrast": _ConstantLayer(value)},
            )
        finally:
            Path(path).unlink()

        heatmap = result.heatmap
        assert heatmap.dtype == np.float32
        assert np.isfinite(heatmap).all()
        assert np.all(heatmap >= 0.0)
        assert np.all(heatmap <= 1.0)
        assert np.allclose(heatmap, value, atol=1e-6)

    @pytest.mark.parametrize("case", ["no-resize", "downscale"])
    def test_internal_nonfinite_heatmap_raises_before_conversion(self, case, monkeypatch) -> None:
        """Internal NaN raises before normalization or lossy uint8 conversion."""
        if case == "no-resize":
            path = _save_temp(_make_test_image(20, 10))
            config = EngineConfig.fast_default()
        else:
            path = "tests/fixtures/landing.png"
            config = EngineConfig.fast_default()
            config.working_long_edge = 256

        def broken_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
            result = arr.copy()
            result[0, 0] = np.nan
            return result

        monkeypatch.setattr("hotgaze.engine._gaussian_blur", broken_blur)
        monkeypatch.setattr(
            "hotgaze.engine.Image.fromarray",
            lambda *args, **kwargs: pytest.fail("lossy uint8 conversion was reached"),
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pytest.raises(FloatingPointError, match="Non-finite heatmap"):
                    run_engine(path, config=config)
            assert not any("invalid value encountered in cast" in str(w.message) for w in caught)
        finally:
            if case == "no-resize":
                Path(path).unlink()

    def test_all_failed_layers_raise_instead_of_returning_zero_map(self) -> None:
        """A numeric layer failure cannot masquerade as a valid all-zero map."""
        path = _save_temp(_make_test_image(20, 10))
        invalid = np.full((10, 20), np.nan, dtype=np.float32)
        config = EngineConfig(weights=LayerWeights(saliency=1.0))
        try:
            with pytest.raises(FloatingPointError, match="all enabled layers failed"):
                run_engine(path, config=config, layers={"saliency": _OutputLayer(invalid)})
        finally:
            Path(path).unlink()

    def test_palette_transparency_flattens_over_white(self, tmp_path: Path) -> None:
        """A transparent palette entry becomes white instead of black."""
        img = Image.new("P", (2, 1))
        img.putdata([0, 1])
        img.putpalette([255, 0, 0, 0, 0, 255] + [0] * (256 * 3 - 6))
        img.info["transparency"] = 0
        path = tmp_path / "palette-transparent.png"
        img.save(path)

        loaded = _load_image(str(path))

        assert loaded.mode == "RGB"
        assert loaded.getpixel((0, 0)) == (255, 255, 255)
        assert loaded.getpixel((1, 0)) == (0, 0, 255)

    def test_jpeg_exif_orientation_precedes_dimensions(self, tmp_path: Path) -> None:
        """Orientation 6 is applied before engine dimensions and attention."""
        img = Image.new("RGB", (4, 3))
        img.putdata(
            [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (32, 64, 96),
                (64, 96, 128),
                (96, 128, 160),
                (128, 160, 192),
                (160, 192, 224),
                (192, 224, 255),
            ]
        )
        exif = Image.Exif()
        exif[274] = 6  # Rotate 90° clockwise.
        path = tmp_path / "oriented.jpg"
        img.save(path, format="JPEG", quality=100, subsampling=0, exif=exif)

        with Image.open(path) as decoded:
            decoded.load()
            expected = ImageOps.exif_transpose(decoded).convert("RGB")
        loaded = _load_image(str(path))
        result = run_engine(str(path))

        np.testing.assert_array_equal(np.asarray(loaded), np.asarray(expected))
        assert loaded.size == (3, 4)
        assert result.original_size == (3, 4)
        assert result.heatmap.shape == (4, 3)

    def test_existing_fixture_scores_are_preserved(self) -> None:
        """Input normalization must not change the established fast scores."""
        result = run_engine("tests/fixtures/landing.png", config=EngineConfig.fast_default())
        scored, _ = result.score(["headline:130,110,330,130", "cta:250,200,200,35"])

        by_name = {entry["name"]: entry for entry in scored}
        assert by_name["headline"]["share"] == pytest.approx(0.191018, abs=1e-6)
        assert by_name["headline"]["peak_value"] == pytest.approx(0.989533, abs=1e-6)
        assert by_name["cta"]["share"] == pytest.approx(0.035241, abs=1e-6)
        assert by_name["cta"]["peak_value"] == pytest.approx(0.768788, abs=1e-6)

    @pytest.mark.parametrize("size", [(1, 1), (10, 10), (20, 10)])
    def test_tiny_images_are_finite_and_warning_free(self, size: tuple[int, int], caplog) -> None:
        """The fast pipeline handles tiny images without numerical noise."""
        path = _save_temp(Image.new("RGB", size, (128, 64, 32)))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with caplog.at_level(logging.WARNING, logger="hotgaze.engine"):
                    result = run_engine(path)

            assert not caught
            assert not [
                record
                for record in caplog.records
                if record.name == "hotgaze.engine" and record.levelno >= logging.WARNING
            ]
            assert result.heatmap.shape == (size[1], size[0])
            assert result.heatmap.dtype == np.float32
            assert np.isfinite(result.heatmap).all()
            assert 0.0 <= result.heatmap.min() <= result.heatmap.max() <= 1.0
            scored, _ = result.score([f"box:0,0,{size[0]},{size[1]}"])
            assert np.isfinite(scored[0]["share"])
            assert all(np.isfinite(point["value"]) for point in result.focal_points())
        finally:
            Path(path).unlink()

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
            assert result.working_size == (1024, 683)
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
        assert am.working_size == (200, 100)
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

    @pytest.mark.parametrize("field", ["saliency", "contrast", "center_bias", "gaze_flow", "faces"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.1])
    def test_invalid_layer_weights_fail_validation(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError, match="finite and non-negative"):
            LayerWeights(**{field: value})

    def test_valid_relative_finite_weights_are_preserved(self) -> None:
        weights = LayerWeights(
            saliency=2.0,
            contrast=0.0,
            center_bias=0.5,
            gaze_flow=0.25,
            faces=0.0,
        )
        assert weights.model_dump() == {
            "saliency": 2.0,
            "contrast": 0.0,
            "center_bias": 0.5,
            "gaze_flow": 0.25,
            "faces": 0.0,
        }

    @pytest.mark.parametrize("field", ["saliency", "contrast", "center_bias", "gaze_flow", "faces"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.1])
    def test_invalid_layer_weight_assignment_is_rejected(self, field: str, value: float) -> None:
        weights = LayerWeights()
        with pytest.raises(ValidationError, match="finite and non-negative"):
            setattr(weights, field, value)

    def test_valid_layer_weight_assignment_is_allowed(self) -> None:
        weights = LayerWeights()
        weights.saliency = 2.0
        weights.contrast = 0.0
        assert weights.saliency == 2.0
        assert weights.contrast == 0.0
