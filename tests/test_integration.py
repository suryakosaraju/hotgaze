"""Integration quality test — machine-checkable Phase 1 demo.

Per PLAN.md T1.4 item 6: on the landing fixture, assert the heatmap is
non-uniform and that attention over the headline/CTA region significantly
exceeds the uniform baseline.
"""

from __future__ import annotations

import numpy as np

from hotgaze.config import EngineConfig
from hotgaze.engine import run_engine


def _region_share(heatmap: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Compute attention share of a region: sum(region) / sum(total)."""
    hmap_h, hmap_w = heatmap.shape
    x = max(0, x)
    y = max(0, y)
    w = min(w, hmap_w - x)
    h = min(h, hmap_h - y)
    if w <= 0 or h <= 0:
        return 0.0
    region_sum = float(heatmap[y : y + h, x : x + w].sum())
    total_sum = float(heatmap.sum())
    if total_sum == 0:
        return 0.0
    return region_sum / total_sum


def _uniform_baseline_share(heatmap: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Share a region would get if attention were perfectly uniform."""
    hmap_h, hmap_w = heatmap.shape
    x = max(0, x)
    y = max(0, y)
    w = min(w, hmap_w - x)
    h = min(h, hmap_h - y)
    return (w * h) / (hmap_w * hmap_h)


class TestIntegrationQuality:
    def test_heatmap_is_non_uniform(self) -> None:
        """The heatmap on a structured UI should have meaningful variance."""
        config = EngineConfig.fast_default()
        result = run_engine("tests/fixtures/landing.png", config=config)
        hm = result.heatmap
        assert hm.std() > 0.05, f"Heatmap too uniform: std={hm.std():.4f}"

    def test_headline_cta_region_above_baseline(self) -> None:
        """The headline + CTA region grabs ≥ 2× the uniform-baseline share."""
        config = EngineConfig.fast_default()
        result = run_engine("tests/fixtures/landing.png", config=config)
        hm = result.heatmap

        # Headline text + CTA button core: (x=130, y=110, w=330, h=130)
        # "Predict Visual Attention" at (150, 130), subtitle at (150, 160),
        # "Get Started" button at (250, 200, 450, 235)
        region = (130, 110, 330, 130)

        actual_share = _region_share(hm, *region)
        baseline_share = _uniform_baseline_share(hm, *region)

        ratio = actual_share / baseline_share if baseline_share > 0 else float("inf")

        assert ratio >= 2.0, (
            f"Headline/CTA region attention share ({actual_share:.4f}) "
            f"is only {ratio:.1f}× the uniform baseline ({baseline_share:.4f}); "
            f"need ≥ 2.0×"
        )

    def test_jet_and_turbo_produce_different_output(self) -> None:
        """--colormap jet vs turbo must produce visually distinct overlays
        (assert the output PNG bytes differ)."""
        import io

        from PIL import Image

        config = EngineConfig.fast_default()
        result = run_engine("tests/fixtures/landing.png", config=config)

        original = Image.open("tests/fixtures/landing.png").convert("RGB")

        jet_overlay = result.overlay(original, alpha=0.6, colormap="jet")
        turbo_overlay = result.overlay(original, alpha=0.6, colormap="turbo")

        jet_buf = io.BytesIO()
        turbo_buf = io.BytesIO()
        jet_overlay.save(jet_buf, format="PNG")
        turbo_overlay.save(turbo_buf, format="PNG")

        assert jet_buf.getvalue() != turbo_buf.getvalue(), (
            "Jet and turbo colormaps produced identical PNG output"
        )
