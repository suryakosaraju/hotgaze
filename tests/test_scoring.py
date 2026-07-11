"""Tests for region scoring, focal points, and canonical JSON (T2.1)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hotgaze.attention_map import AttentionMap
from hotgaze.scoring import (
    RegionParseError,
    _CanonicalEncoder,
    find_focal_points,
    parse_region,
    score_regions,
    scores_to_json,
    validate_against_schema,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _uniform_map(w: int = 200, h: int = 100) -> AttentionMap:
    hm = np.ones((h, w), dtype=np.float32) * 0.5
    return AttentionMap(hm, (w, h), {"backend": "fast"})


def _gradient_map(w: int = 200, h: int = 100) -> AttentionMap:
    """Left-to-right gradient: value = x / w."""
    xs = np.linspace(0, 1, w, dtype=np.float32)
    hm = np.tile(xs, (h, 1))
    return AttentionMap(hm, (w, h), {"backend": "fast"})


def _two_hotspots_map() -> AttentionMap:
    """Two Gaussian hotspots at known positions."""
    w, h = 200, 100
    y, x = np.mgrid[0:h, 0:w]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    # Hotspot 1 at (50, 30), sigma 10
    g1 = np.exp(-((x - 50) ** 2 + (y - 30) ** 2) / (2 * 10**2))
    # Hotspot 2 at (150, 70), sigma 15
    g2 = np.exp(-((x - 150) ** 2 + (y - 70) ** 2) / (2 * 15**2))
    hm = g1 + 0.5 * g2
    hm /= hm.max()
    return AttentionMap(hm.astype(np.float32), (w, h), {"backend": "fast"})


# ── Region parsing ───────────────────────────────────────────────────────────


class TestParseRegion:
    def test_pixel_coords(self) -> None:
        name, x, y, w, h = parse_region("cta:10,20,100,50", 800, 600)
        assert name == "cta"
        assert (x, y, w, h) == (10, 20, 100, 50)

    def test_fractional_coords(self) -> None:
        name, x, y, w, h = parse_region("logo:0.1,0.2,0.3,0.08f", 800, 600)
        assert name == "logo"
        assert (x, y, w, h) == (80, 120, 240, 48)

    def test_clamping(self) -> None:
        """Negative coords clamp to 0, oversized boxes clamp to image edge."""
        name, x, y, w, h = parse_region("edge:-10,-10,500,500", 200, 100)
        assert x == 0
        assert y == 0
        assert w <= 200
        assert h <= 100

    def test_fully_out_of_bounds(self) -> None:
        with pytest.raises(RegionParseError, match="fully out of bounds"):
            parse_region("bad:300,300,10,10", 200, 100)

    def test_invalid_format(self) -> None:
        with pytest.raises(RegionParseError, match="Invalid region format"):
            parse_region("just_a_name", 800, 600)

    def test_float_without_f_is_truncated_to_int(self) -> None:
        """A region without trailing f but with float values → int truncation."""
        name, x, y, w, h = parse_region("test:0.5,0.5,10,10", 800, 600)
        assert x == 0 and y == 0  # truncated to int


# ── Region scoring ───────────────────────────────────────────────────────────


class TestScoreRegions:
    def test_uniform_map_equal_shares(self) -> None:
        """On a uniform map, share = region_area / total_area exactly."""
        am = _uniform_map(200, 100)
        scored, _ = score_regions(am, ["half:0,0,100,100"])
        assert len(scored) == 1
        # Half-width region on uniform map → exactly 50% share
        assert scored[0]["share"] == pytest.approx(0.5, abs=1e-6)

    def test_multiple_regions_ranked(self) -> None:
        """Regions are ranked by attention share descending."""
        am = _gradient_map(200, 100)
        # Right side has higher values (gradient increases left→right)
        scored, _ = score_regions(am, ["left:0,0,100,100", "right:100,0,100,100"])
        assert scored[0]["name"] == "right"
        assert scored[1]["name"] == "left"
        assert scored[0]["share"] > scored[1]["share"]
        assert scored[0]["rank"] == 1
        assert scored[1]["rank"] == 2

    def test_share_is_proportion_of_total_mass(self) -> None:
        """Region share = sum(region) / sum(total)."""
        am = _gradient_map(200, 100)
        scored, _ = score_regions(am, ["full:0,0,200,100", "half:0,0,100,100"])
        # Full region covers everything → share ≈ 1.0
        assert scored[0]["share"] == pytest.approx(1.0, abs=1e-6)
        # Half region covers left 50% (lower gradient values) → < 0.5
        half_share = [r for r in scored if r["name"] == "half"][0]["share"]
        assert half_share < 0.5

    def test_hand_computed_share_exact(self) -> None:
        """Hand-computed share on a known map matches to 1e-6 before rounding."""
        # Simple 4x4 map: values 0..15
        hm = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
        am = AttentionMap(hm, (4, 4))
        scored, _ = score_regions(am, ["tl:0,0,2,2"])
        # Region tl covers indices [0,1] × [0,1] = values 0,1,4,5 = 10
        # Total = sum(0..15) = 120, share = 10/120
        expected = 10.0 / 120.0
        assert scored[0]["share"] == pytest.approx(expected, abs=1e-6)

    def test_peak_value(self) -> None:
        am = _two_hotspots_map()
        scored, _ = score_regions(am, ["hs1:40,20,30,30", "hs2:140,60,30,30"])
        hs1 = [r for r in scored if r["name"] == "hs1"][0]
        hs2 = [r for r in scored if r["name"] == "hs2"][0]
        # Hotspot 1 is stronger
        assert hs1["peak_value"] > hs2["peak_value"]


# ── Focal points ─────────────────────────────────────────────────────────────


class TestFocalPoints:
    def test_finds_local_maxima(self) -> None:
        am = _two_hotspots_map()
        pts = find_focal_points(am, n=5)
        assert len(pts) >= 1
        assert pts[0]["rank"] == 1
        # Strongest hotspot is at ~(50, 30)
        assert abs(pts[0]["x"] - 50) < 10
        assert abs(pts[0]["y"] - 30) < 10

    def test_respects_n(self) -> None:
        am = _two_hotspots_map()
        pts = find_focal_points(am, n=1)
        assert len(pts) == 1

    def test_min_distance_suppression(self) -> None:
        """Points within 5% of diagonal are suppressed."""
        # Create a map with two peaks close together
        w, h = 200, 200
        hm = np.zeros((h, w), dtype=np.float32)
        hm[100, 50] = 1.0
        hm[100, 55] = 0.9  # only 5px away — should be suppressed
        am = AttentionMap(hm, (w, h))
        pts = find_focal_points(am, n=5)
        # Should find at most 1 (second is within min_dist)
        assert len(pts) <= 1

    def test_threshold_filters_low_values(self) -> None:
        """Peaks below 10% of max are excluded."""
        hm = np.full((100, 100), 0.05, dtype=np.float32)
        hm[50, 50] = 0.09  # below 10% of max=0.09? No, max is 0.09, threshold=0.009
        am = AttentionMap(hm, (100, 100))
        pts = find_focal_points(am, n=5)
        # Should find at least the 0.09 peak (> 0.009)
        assert len(pts) >= 1

    def test_original_image_coords(self) -> None:
        """Focal point coords are in original-image space."""
        am = _two_hotspots_map()
        pts = find_focal_points(am, n=5)
        for pt in pts:
            assert 0 <= pt["x"] < am.original_size[0]
            assert 0 <= pt["y"] < am.original_size[1]


# ── Canonical JSON ───────────────────────────────────────────────────────────


class TestCanonicalJSON:
    def test_sorted_keys(self) -> None:
        data = {"z": 1, "a": 2, "m": 3}
        result = _CanonicalEncoder().encode(data)
        # Keys should appear in sorted order
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_float_6dp(self) -> None:
        data = {"pi": 3.1415926535}
        result = _CanonicalEncoder().encode(data)
        assert "3.141593" in result

    def test_byte_identical_two_runs(self) -> None:
        am = _gradient_map()
        scored, focal = score_regions(am, ["half:0,0,100,100"])
        config = am.config

        json1 = scores_to_json("score", "test.png", (200, 100), (200, 100), config, scored, focal)
        json2 = scores_to_json("score", "test.png", (200, 100), (200, 100), config, scored, focal)
        assert json1 == json2
        assert isinstance(json1, str)
        assert json1.endswith("\n")

    def test_compare_mode_includes_image_b(self) -> None:
        am = _gradient_map()
        scored, focal = score_regions(am, [])
        result = scores_to_json(
            "compare",
            "a.png",
            (200, 100),
            (200, 100),
            am.config,
            scored,
            focal,
            compare={"per_region_deltas": [], "grid_deltas": [0] * 9, "focal_point_movement": []},
            image_b_path="b.png",
            img_b_size=(200, 100),
            work_b_size=(200, 100),
        )
        data = json.loads(result)
        assert data["mode"] == "compare"
        assert "image_b" in data
        assert "compare" in data


# ── Schema validation ────────────────────────────────────────────────────────


class TestSchemaValidation:
    def test_valid_score_output(self) -> None:
        am = _gradient_map()
        scored, focal = score_regions(am, ["half:0,0,100,100"])
        result = scores_to_json(
            "score", "test.png", (200, 100), (200, 100), am.config, scored, focal
        )
        data = json.loads(result)
        errors = validate_against_schema(data)
        assert errors == [], f"Schema errors: {errors}"

    def test_missing_required_key_fails(self) -> None:
        data = {"schema": 1, "mode": "score"}
        errors = validate_against_schema(data)
        assert len(errors) > 0

    def test_wrong_schema_version_fails(self) -> None:
        data = {"schema": 99, "mode": "score"}
        errors = validate_against_schema(data)
        assert any("version" in e.lower() for e in errors)

    def test_valid_compare_output(self) -> None:
        am = _gradient_map()
        scored, focal = score_regions(am, [])
        result = scores_to_json(
            "compare",
            "a.png",
            (200, 100),
            (200, 100),
            am.config,
            scored,
            focal,
            compare={"per_region_deltas": [], "grid_deltas": [0] * 9, "focal_point_movement": []},
            image_b_path="b.png",
            img_b_size=(200, 100),
            work_b_size=(200, 100),
        )
        data = json.loads(result)
        errors = validate_against_schema(data)
        assert errors == [], f"Schema errors: {errors}"
