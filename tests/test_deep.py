"""Tests for the deep saliency layer (T3.2)."""

from __future__ import annotations

import numpy as np
import pytest


def _require_torch():
    """Skip the test if torch is not installed."""
    pytest.importorskip("torch")


def _fake_unisal_model():
    """Create a tiny fake torch module that returns a known tensor."""
    import torch

    class FakeUNISAL(torch.nn.Module):
        def forward(self, x, target_size=None, source="SALICON", static=True):  # noqa: ARG002
            b, t, c, h, w = x.shape
            y, xc = torch.meshgrid(
                torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
            )
            gauss = torch.exp(-(y**2 + xc**2) / 0.1)
            gauss = gauss / gauss.sum()
            log_prob = torch.log(gauss + 1e-12)
            return log_prob.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    return FakeUNISAL()


# ── Layer with fake model ────────────────────────────────────────────────────


class TestSaliencyDeepFake:
    def test_non_stride_input_is_padded_and_cropped(self) -> None:
        _require_torch()
        import torch

        from hotgaze.layers.saliency_deep import SaliencyDeep

        class ShapeCheckingFakeUNISAL(torch.nn.Module):
            def forward(self, x, target_size=None, source="SALICON", static=True):  # noqa: ARG002
                _, _, _, h, w = x.shape
                assert h % 32 == 0
                assert w % 32 == 0
                return torch.zeros((1, 1, 1, h, w), dtype=x.dtype)

        img = np.zeros((600, 800, 3), dtype=np.uint8)
        result = SaliencyDeep(ShapeCheckingFakeUNISAL()).compute(img)

        assert result.shape == (600, 800)
        assert result.dtype == np.float32

    def test_output_shape_matches_input(self) -> None:
        _require_torch()
        from hotgaze.layers.saliency_deep import SaliencyDeep

        model = _fake_unisal_model()
        layer = SaliencyDeep(model)
        img = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
        result = layer.compute(img)
        assert result.shape == (128, 256)

    def test_output_dtype_float32(self) -> None:
        _require_torch()
        from hotgaze.layers.saliency_deep import SaliencyDeep

        layer = SaliencyDeep(_fake_unisal_model())
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = layer.compute(img)
        assert result.dtype == np.float32

    def test_output_range(self) -> None:
        _require_torch()
        from hotgaze.layers.saliency_deep import SaliencyDeep

        layer = SaliencyDeep(_fake_unisal_model())
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = layer.compute(img)
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_deterministic(self) -> None:
        _require_torch()
        from hotgaze.layers.saliency_deep import SaliencyDeep

        layer = SaliencyDeep(_fake_unisal_model())
        img = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
        r1 = layer.compute(img)
        r2 = layer.compute(img)
        np.testing.assert_array_equal(r1, r2)

    def test_non_square_input(self) -> None:
        _require_torch()
        from hotgaze.layers.saliency_deep import SaliencyDeep

        layer = SaliencyDeep(_fake_unisal_model())
        img = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
        result = layer.compute(img)
        assert result.shape == (100, 300)

    def test_odd_dimensions(self) -> None:
        _require_torch()
        from hotgaze.layers.saliency_deep import SaliencyDeep

        layer = SaliencyDeep(_fake_unisal_model())
        img = np.random.randint(0, 255, (127, 253, 3), dtype=np.uint8)
        result = layer.compute(img)
        assert result.shape == (127, 253)


# ── CLI error paths ──────────────────────────────────────────────────────────


class TestDeepCLIErrors:
    def test_deep_without_torch_actionable(self) -> None:
        """--backend deep raises actionable ImportError when torch missing.

        When torch is absent, _default_deep_layers raises ImportError with
        the pip install hotgaze[deep] message. When torch is present, this
        test skips (the real CI matrix covers the no-torch path naturally).
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            from hotgaze.engine import _default_deep_layers

            with pytest.raises(ImportError, match="pip install hotgaze"):
                _default_deep_layers()
        else:
            pytest.skip("torch is installed — CI covers the no-torch path")

    def test_deep_with_torch_no_weights(self) -> None:
        """--backend deep with torch but no published weights → actionable error."""
        pytest.importorskip("torch")
        from unittest.mock import patch

        from hotgaze.layers.saliency_deep import load_unisal

        with patch(  # noqa: SIM117
            "hotgaze.layers.saliency_deep.download_weight",
            side_effect=FileNotFoundError("weights not published"),
        ):
            with pytest.raises(FileNotFoundError, match="not yet published"):
                load_unisal()


# ── Real-weight tests (skip without weights) ─────────────────────────────────


@pytest.mark.deep
class TestSaliencyDeepReal:
    def test_real_weights_differ_from_fast_and_repeat_scores(self) -> None:
        """The real deep backend changes the map and repeats canonical scores."""
        pytest.importorskip("torch")
        try:
            from hotgaze.layers.saliency_deep import load_unisal

            load_unisal()
        except FileNotFoundError:
            pytest.skip("UNISAL weights not yet published")

        from hotgaze.config import EngineConfig
        from hotgaze.engine import run_engine
        from hotgaze.scoring import scores_to_json

        path = "tests/fixtures/landing.png"
        fast = run_engine(path, config=EngineConfig.fast_default())
        deep_a = run_engine(path, config=EngineConfig.deep_default())
        deep_b = run_engine(path, config=EngineConfig.deep_default())

        mean_difference = float(np.abs(fast.heatmap - deep_a.heatmap).mean())
        assert mean_difference > 1e-3

        regions_a, focal_a = deep_a.score(["headline:130,110,330,130", "cta:250,200,200,35"])
        regions_b, focal_b = deep_b.score(["headline:130,110,330,130", "cta:250,200,200,35"])
        json_a = scores_to_json(
            "score",
            path,
            deep_a.original_size,
            deep_a.working_size,
            deep_a.config,
            regions_a,
            focal_a,
        )
        json_b = scores_to_json(
            "score",
            path,
            deep_b.original_size,
            deep_b.working_size,
            deep_b.config,
            regions_b,
            focal_b,
        )
        assert json_a == json_b

    def test_real_weights_handle_non_stride_landing_fixture(self) -> None:
        """UNISAL handles the 800×600 fixture without a skip or shape error."""
        pytest.importorskip("torch")
        try:
            from hotgaze.layers.saliency_deep import SaliencyDeep, load_unisal

            model = load_unisal()
        except FileNotFoundError:
            pytest.skip("UNISAL weights not yet published")

        from PIL import Image

        image = np.asarray(Image.open("tests/fixtures/landing.png").convert("RGB"))
        result = SaliencyDeep(model).compute(image)

        assert result.shape == image.shape[:2]
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_real_weights_astronaut_face(self) -> None:
        """Load real UNISAL, run on scikit-image astronaut, verify face detected.

        The astronaut image (512×512) has Eileen Collins' face at roughly
        x ∈ [180, 320], y ∈ [60, 170].  The top focal point from UNISAL
        should land in that region.
        """
        pytest.importorskip("torch")
        try:
            from hotgaze.layers.saliency_deep import SaliencyDeep, load_unisal

            model = load_unisal()
        except FileNotFoundError:
            pytest.skip("UNISAL weights not yet published")

        from skimage import data

        astronaut = data.astronaut()  # (512, 512, 3) uint8 RGB
        layer = SaliencyDeep(model)
        result = layer.compute(astronaut)

        from hotgaze.scoring import find_focal_points

        am = type("AM", (), {"original_size": (512, 512), "heatmap": result})()
        focal = find_focal_points(am, n=5)
        assert len(focal) >= 1, "No focal points found"
        top = focal[0]
        assert 180 <= top["x"] <= 320, f"Focal point x={top['x']} outside face region"
        assert 60 <= top["y"] <= 170, f"Focal point y={top['y']} outside face region"
