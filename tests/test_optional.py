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


# ── regression: --layers faces actually enables the layer ────────────────────


class TestLayersFlagEnablesFaces:
    def test_layers_faces_actually_enabled(self) -> None:
        """--layers faces flag actually adds the layer with weight 0.15.

        Regression test for bug: line 118 re-assigned w = config.weights,
        discarding the renormalized copy.  Fixed by removing that line.
        """
        _require_yunet()

        import numpy as np

        from hotgaze.config import EngineConfig, LayerWeights
        from hotgaze.engine import run_engine

        # Build a config with faces enabled
        config = EngineConfig(
            backend="fast",
            weights=LayerWeights(saliency=0.5, contrast=0.2, center_bias=0.2, gaze_flow=0.1),
            extra_layers=["faces"],
        )

        # Run without faces first
        config_no_faces = EngineConfig(
            backend="fast",
            weights=LayerWeights(saliency=0.5, contrast=0.2, center_bias=0.2, gaze_flow=0.1),
            extra_layers=[],
        )

        astronaut = _astronaut()
        import tempfile
        from pathlib import Path

        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(astronaut).save(tmp.name)
            tmp_path = tmp.name

        try:
            result_faces = run_engine(tmp_path, config=config)
            result_no_faces = run_engine(tmp_path, config=config_no_faces)

            # The two maps should differ — faces layer adds Gaussian blobs
            diff = np.abs(result_faces.heatmap - result_no_faces.heatmap).max()
            assert diff > 1e-6, (
                f"Faces layer had no effect on attention map (max diff={diff:.10f}). "
                "The --layers faces flag is not actually enabling the layer."
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_faces_weight_hits_face_on_astronaut(self) -> None:
        """At default weight 0.50, the top focal point lands in the face region."""
        _require_yunet()

        from hotgaze.config import EngineConfig, LayerWeights
        from hotgaze.engine import run_engine
        from hotgaze.scoring import find_focal_points

        config = EngineConfig(
            backend="fast",
            weights=LayerWeights(saliency=0.5, contrast=0.2, center_bias=0.2, gaze_flow=0.1),
            extra_layers=["faces"],
        )

        astronaut = _astronaut()
        import tempfile
        from pathlib import Path

        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(astronaut).save(tmp.name)
            tmp_path = tmp.name

        try:
            result = run_engine(tmp_path, config=config)
            focal = find_focal_points(result, n=3)
            assert len(focal) >= 1, "No focal points"
            top = focal[0]
            assert 180 <= top["x"] <= 320, (
                f"Focal point x={top['x']} outside face region [180,320] "
                f"— faces weight may be too low"
            )
            assert 60 <= top["y"] <= 170, (
                f"Focal point y={top['y']} outside face region [60,170] "
                f"— faces weight may be too low"
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_faces_no_effect_on_faceless_image(self) -> None:
        """Enabling faces on a faceless image does not shift focal points."""
        _require_yunet()

        from hotgaze.config import EngineConfig, LayerWeights
        from hotgaze.engine import run_engine
        from hotgaze.scoring import find_focal_points

        config_faces = EngineConfig(
            backend="fast",
            weights=LayerWeights(saliency=0.5, contrast=0.2, center_bias=0.2, gaze_flow=0.1),
            extra_layers=["faces"],
        )
        config_none = EngineConfig(
            backend="fast",
            weights=LayerWeights(saliency=0.5, contrast=0.2, center_bias=0.2, gaze_flow=0.1),
            extra_layers=[],
        )

        result_faces = run_engine("tests/fixtures/landing.png", config=config_faces)
        result_none = run_engine("tests/fixtures/landing.png", config=config_none)

        focal_faces = find_focal_points(result_faces, n=3)
        focal_none = find_focal_points(result_none, n=3)

        assert len(focal_faces) >= 1 and len(focal_none) >= 1
        dx = abs(focal_faces[0]["x"] - focal_none[0]["x"])
        dy = abs(focal_faces[0]["y"] - focal_none[0]["y"])
        assert dx <= 2 and dy <= 2, (
            f"Faces layer shifted focal point by ({dx},{dy})px on faceless image"
        )
