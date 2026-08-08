"""Tests for the `hotgaze compare` CLI command (T2.3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from hotgaze.attention_map import AttentionMap
from hotgaze.cli import main
from hotgaze.scoring import validate_against_schema


def _fixture(name: str) -> str:
    return str(Path(__file__).parent / "fixtures" / name)


class TestCompareCLI:
    def test_region_deltas_match_names_when_rankings_swap(self, tmp_path, monkeypatch) -> None:
        """Compare pairs each region with its namesake, not its rank."""
        image_a = tmp_path / "a.png"
        image_b = tmp_path / "b.png"
        image_a.touch()
        image_b.touch()

        heatmap_a = np.ones((10, 20), dtype=np.float32)
        heatmap_a[:, :10] = 10.0
        heatmap_b = np.ones((10, 20), dtype=np.float32)
        heatmap_b[:, 10:] = 10.0
        maps = {
            str(image_a): AttentionMap(heatmap_a, (20, 10)),
            str(image_b): AttentionMap(heatmap_b, (20, 10)),
        }

        def fake_run_engine(path: str, config=None) -> AttentionMap:
            return maps[path]

        monkeypatch.setattr("hotgaze.cli.run_engine", fake_run_engine)
        result = CliRunner().invoke(
            main,
            [
                "compare",
                str(image_a),
                str(image_b),
                "--region",
                "left:0,0,10,10",
                "--region",
                "right:10,0,10,10",
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        deltas = {
            entry["name"]: entry
            for entry in json.loads(result.output)["compare"]["per_region_deltas"]
        }
        assert deltas["left"]["delta"] == -0.818182
        assert deltas["right"]["delta"] == 0.818182

    # ── AC 1: A==B → all deltas exactly 0 ─────────────────────────────────

    def test_same_image_zero_deltas(self) -> None:
        """Comparing an image to itself yields zero deltas."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "compare",
                _fixture("landing.png"),
                _fixture("landing.png"),
                "--region",
                "cta:250,200,200,35",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["mode"] == "compare"
        for d in data["compare"]["per_region_deltas"]:
            assert d["delta"] == 0.0
            assert d["share_a"] == d["share_b"]

    def test_duplicate_region_names_error(self) -> None:
        """Distinct boxes must not share a name in compare mode."""
        result = CliRunner().invoke(
            main,
            [
                "compare",
                _fixture("landing.png"),
                _fixture("landing_variant.png"),
                "--region",
                "target:0,0,100,100",
                "--region",
                "target:700,500,100,100",
                "--json",
            ],
        )
        assert result.exit_code != 0
        assert "duplicate region name" in result.output.lower()
        assert "unique name" in result.output.lower()

    # ── AC 2: region relocated onto hotter area → positive delta ──────────

    def test_relocated_region_positive_delta(self, tmp_path, monkeypatch) -> None:
        """A controlled relocation onto a hotter area produces a positive delta."""
        image_a = tmp_path / "a.png"
        image_b = tmp_path / "b.png"
        image_a.touch()
        image_b.touch()

        heatmap_a = np.ones((10, 20), dtype=np.float32)
        heatmap_b = np.ones((10, 20), dtype=np.float32)
        heatmap_b[:, :10] = 10.0
        maps = {
            str(image_a): AttentionMap(heatmap_a, (20, 10)),
            str(image_b): AttentionMap(heatmap_b, (20, 10)),
        }

        def fake_run_engine(path: str, config=None) -> AttentionMap:
            return maps[path]

        monkeypatch.setattr("hotgaze.cli.run_engine", fake_run_engine)
        result = CliRunner().invoke(
            main,
            [
                "compare",
                str(image_a),
                str(image_b),
                "--region",
                "target:0,0,10,10",
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        delta = json.loads(result.output)["compare"]["per_region_deltas"][0]["delta"]
        assert delta == 0.409091
        assert delta > 0

    # ── AC 3: JSON includes absolute shares AND deltas ────────────────────

    def test_json_has_absolute_shares_and_deltas(self) -> None:
        """Compare JSON includes share_a, share_b, and delta for each region."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "compare",
                _fixture("landing.png"),
                _fixture("landing_variant.png"),
                "--region",
                "cta:250,200,200,35",
                "--region",
                "hero:100,100,600,150",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["regions"]) == 2
        for r in data["regions"]:
            assert "share" in r
            assert "peak_value" in r
        deltas = data["compare"]["per_region_deltas"]
        assert len(deltas) == 2
        for d in deltas:
            assert "name" in d
            assert "share_a" in d
            assert "share_b" in d
            assert "delta" in d
        # Region mode: grid_deltas and focal_point_movement are empty
        assert data["compare"]["grid_deltas"] == []
        assert data["compare"]["focal_point_movement"] == []
        assert validate_against_schema(data) == []

    def test_grid_deltas_non_empty_in_no_region_mode(self) -> None:
        """In no-region mode, grid_deltas has 9 values and focal_movement is populated."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["compare", _fixture("landing.png"), _fixture("landing_variant.png"), "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["compare"]["grid_deltas"]) == 9
        assert len(data["compare"]["focal_point_movement"]) > 0

    def test_size_mismatch_without_regions_errors(self) -> None:
        """No-region movement metrics require a shared pixel coordinate system."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["compare", _fixture("landing.png"), _fixture("1440x900.png"), "--json"],
        )
        assert result.exit_code != 0
        assert "different sizes" in result.output.lower()
        assert "fractional --region" in result.output.lower()

    # ── AC 4: no-region mode → 3×3 grid + focal movement ─────────────────

    def test_no_region_mode_grid_and_focal(self) -> None:
        """Without --region, output includes grid_deltas and focal_point_movement."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "compare",
                _fixture("landing.png"),
                _fixture("landing_variant.png"),
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        compare = data["compare"]
        assert len(compare["grid_deltas"]) == 9
        assert len(compare["focal_point_movement"]) > 0

    # ── Size mismatch + pixel region error ────────────────────────────────

    def test_size_mismatch_pixel_region_error(self) -> None:
        """Pixel regions with mismatched image sizes → actionable error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "compare",
                _fixture("landing.png"),
                _fixture("1440x900.png"),
                "--region",
                "cta:250,200,200,35",
                "--json",
            ],
        )
        assert result.exit_code != 0
        assert "fractional coords" in result.output.lower()

    def test_size_mismatch_fractional_region_works(self) -> None:
        """Fractional regions work across different image sizes."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "compare",
                _fixture("landing.png"),
                _fixture("1440x900.png"),
                "--region",
                "cta:0.3,0.3,0.25,0.06f",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["mode"] == "compare"

    # ── Human-readable modes ──────────────────────────────────────────────

    def test_human_readable_region_mode(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "compare",
                _fixture("landing.png"),
                _fixture("landing_variant.png"),
                "--region",
                "cta:250,200,200,35",
            ],
        )
        assert result.exit_code == 0
        assert "Δ=" in result.output

    def test_human_readable_no_region_mode(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "compare",
                _fixture("landing.png"),
                _fixture("landing_variant.png"),
            ],
        )
        assert result.exit_code == 0
        assert "3×3 grid" in result.output
        assert "Focal-point movement" in result.output

    # ── Determinism ───────────────────────────────────────────────────────

    def test_deterministic(self) -> None:
        runner = CliRunner()
        args = [
            "compare",
            _fixture("landing.png"),
            _fixture("landing_variant.png"),
            "--region",
            "cta:250,200,200,35",
            "--json",
        ]
        r1 = runner.invoke(main, args)
        r2 = runner.invoke(main, args)
        assert r1.exit_code == 0 and r2.exit_code == 0
        assert r1.output == r2.output
