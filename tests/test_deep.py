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
    def test_real_weights_load_and_predict(self) -> None:
        """Full pipeline with real UNISAL weights."""
        pytest.importorskip("torch")
        try:
            from hotgaze.layers.saliency_deep import SaliencyDeep, load_unisal

            model = load_unisal()
        except FileNotFoundError:
            pytest.skip("UNISAL weights not yet published")

        layer = SaliencyDeep(model)
        img = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
        result = layer.compute(img)

        assert result.shape == (128, 256)
        assert result.dtype == np.float32
        assert result.max() - result.min() > 0.01
