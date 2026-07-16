"""Tests for optional layers: faces (YuNet)."""

from __future__ import annotations

import numpy as np
import pytest


def _require_yunet():
    """Skip the test if the YuNet ONNX weight is not cached."""
    try:
        from hotgaze.weights import download_weight

        download_weight("yunet")
    except FileNotFoundError:
        pytest.skip("YuNet weight not cached")


def _astronaut() -> np.ndarray:
    from skimage import data

    return data.astronaut()


def _blank(h: int = 200, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ── Faces (YuNet) ────────────────────────────────────────────────────────────


class TestFaces:
    def test_astronaut_face_detected(self) -> None:
        _require_yunet()
        from hotgaze.layers.faces import Faces
        from hotgaze.scoring import find_focal_points

        layer = Faces()
        astronaut = _astronaut()
        result = layer.compute(astronaut)

        am = type("AM", (), {"original_size": (512, 512), "heatmap": result})()
        focal = find_focal_points(am, n=3)
        assert len(focal) >= 1, "No focal points — face not detected"
        top = focal[0]
        assert 180 <= top["x"] <= 320, f"x={top['x']} outside face region [180,320]"
        assert 60 <= top["y"] <= 170, f"y={top['y']} outside face region [60,170]"

    def test_no_face_blank_near_zero(self) -> None:
        _require_yunet()
        from hotgaze.layers.faces import Faces

        layer = Faces()
        result = layer.compute(_blank())
        assert result.max() < 0.01, f"Blank image produced non-zero map: max={result.max():.4f}"

    def test_output_contract(self) -> None:
        _require_yunet()
        from hotgaze.layers.faces import Faces

        layer = Faces()
        result = layer.compute(_blank())
        assert result.dtype == np.float32
        assert result.shape == (200, 200)
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_deterministic(self) -> None:
        _require_yunet()
        from hotgaze.layers.faces import Faces

        layer = Faces()
        img = _astronaut()
        r1 = layer.compute(img)
        r2 = layer.compute(img)
        np.testing.assert_array_equal(r1, r2)


# ── info without torch ───────────────────────────────────────────────────────


class TestInfoNoTorch:
    def test_info_runs_without_torch(self, monkeypatch) -> None:
        """hotgaze info exits 0 even when torch is not installed."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"torch": None}, clear=False):
            from click.testing import CliRunner

            from hotgaze.cli import main

            runner = CliRunner()
            result = runner.invoke(main, ["info"])
            assert result.exit_code == 0
