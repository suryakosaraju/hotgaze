"""Tests for optional layers: faces (YuNet) and text (MSER)."""

from __future__ import annotations

import numpy as np

# ── Helpers ──────────────────────────────────────────────────────────────────


def _astronaut() -> np.ndarray:
    from skimage import data

    return data.astronaut()


def _blank(h: int = 200, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ── Faces (YuNet) ────────────────────────────────────────────────────────────


class TestFaces:
    def test_astronaut_face_detected(self) -> None:
        """YuNet finds the astronaut's face and creates a strong attention blob."""
        from hotgaze.layers.faces import Faces
        from hotgaze.scoring import find_focal_points

        layer = Faces()
        astronaut = _astronaut()
        result = layer.compute(astronaut)

        am = type("AM", (), {"original_size": (512, 512), "heatmap": result})()
        focal = find_focal_points(am, n=3)
        assert len(focal) >= 1, "No focal points — face not detected"
        top = focal[0]
        # Face region of astronaut: approx x 180-320, y 60-170
        assert 180 <= top["x"] <= 320, f"x={top['x']} outside face region [180,320]"
        assert 60 <= top["y"] <= 170, f"y={top['y']} outside face region [60,170]"

    def test_no_face_blank_near_zero(self) -> None:
        """Blank/no-face image → near-zero attention map."""
        from hotgaze.layers.faces import Faces

        layer = Faces()
        result = layer.compute(_blank())
        assert result.max() < 0.01, f"Blank image produced non-zero map: max={result.max():.4f}"

    def test_output_contract(self) -> None:
        from hotgaze.layers.faces import Faces

        layer = Faces()
        result = layer.compute(_blank())
        assert result.dtype == np.float32
        assert result.shape == (200, 200)
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_deterministic(self) -> None:
        from hotgaze.layers.faces import Faces

        layer = Faces()
        img = _astronaut()
        r1 = layer.compute(img)
        r2 = layer.compute(img)
        np.testing.assert_array_equal(r1, r2)
