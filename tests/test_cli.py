"""Tests for the HotGaze CLI — run, score, info."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from hotgaze.cli import main


def _fixture(name: str) -> str:
    return str(Path(__file__).parent / "fixtures" / name)


# ── T1.3: run ────────────────────────────────────────────────────────────────


class TestRun:
    def test_run_produces_png(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "out.png")
            result = runner.invoke(main, ["run", _fixture("landing.png"), "-o", out])
            assert result.exit_code == 0
            assert Path(out).exists()
            img = Image.open(out)
            assert img.size == (800, 600)
            assert img.mode == "RGB"

    def test_bad_path_exits_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "/nonexistent/path.png"])
        assert result.exit_code != 0

    def test_help_includes_usage(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--alpha" in result.output

    def test_default_output_naming(self) -> None:
        """Without -o, output defaults to IMG_overlay.png alongside input."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            import shutil

            src = _fixture("landing.png")
            dst = str(Path(tmp) / "landing.png")
            shutil.copy(src, dst)
            result = runner.invoke(main, ["run", dst])
            assert result.exit_code == 0
            assert Path(tmp, "landing_overlay.png").exists()


# ── T1.3: info ───────────────────────────────────────────────────────────────


class TestInfo:
    def test_info_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "HotGaze" in result.output
        assert "Cache:" in result.output
        assert "fast" in result.output

    def test_version_option(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "hotgaze" in result.output


# ── T2.2: score ──────────────────────────────────────────────────────────────


class TestScoreCLI:
    def test_json_pipes_cleanly(self) -> None:
        """--json output validates as JSON and parses cleanly."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["score", _fixture("landing.png"), "--region", "cta:250,200,200,35", "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["schema"] == 1
        assert data["mode"] == "score"
        assert len(data["regions"]) == 1
        assert "focal_points" in data

    def test_deterministic_json(self) -> None:
        """Two runs produce byte-identical JSON."""
        runner = CliRunner()
        args = [
            "score",
            _fixture("landing.png"),
            "--region",
            "cta:250,200,200,35",
            "--json",
        ]
        r1 = runner.invoke(main, args)
        r2 = runner.invoke(main, args)
        assert r1.exit_code == 0
        assert r2.exit_code == 0
        assert r1.output == r2.output

    def test_region_parse_error_actionable(self) -> None:
        """Decimal values without trailing f produce actionable error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "score",
                _fixture("landing.png"),
                "--region",
                "bad:0.5,0.5,10,10",
                "--json",
            ],
        )
        assert result.exit_code != 0
        assert "looks fractional" in result.output

    def test_out_of_bounds_error_actionable(self) -> None:
        """Fully out-of-bounds region produces actionable error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "score",
                _fixture("landing.png"),
                "--region",
                "nowhere:900,900,100,100",
                "--json",
            ],
        )
        assert result.exit_code != 0
        assert "fully out of bounds" in result.output

    def test_human_readable_output(self) -> None:
        """Without --json, outputs a human-readable table."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["score", _fixture("landing.png"), "--region", "cta:250,200,200,35"],
        )
        assert result.exit_code == 0
        assert "Region scores" in result.output
        assert "Focal points" in result.output

    def test_multiple_regions(self) -> None:
        """Multiple --region flags work."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "score",
                _fixture("landing.png"),
                "--region",
                "tl:0,0,100,100",
                "--region",
                "br:700,500,100,100",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["regions"]) == 2

    def test_fractional_region_works(self) -> None:
        """Fractional region with trailing f parses correctly."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "score",
                _fixture("landing.png"),
                "--region",
                "half:0.25,0.25,0.5,0.5f",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["regions"][0]["name"] == "half"

    def test_help_shows_score_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["score", "--help"])
        assert result.exit_code == 0
        assert "--region" in result.output
        assert "--json" in result.output
