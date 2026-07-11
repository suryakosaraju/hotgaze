"""Tests for the HotGaze CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from hotgaze.cli import main


def _fixture(name: str) -> str:
    return str(Path(__file__).parent / "fixtures" / name)


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
            # Copy fixture into tmp so we can write alongside it
            import shutil

            src = _fixture("landing.png")
            dst = str(Path(tmp) / "landing.png")
            shutil.copy(src, dst)
            result = runner.invoke(main, ["run", dst])
            assert result.exit_code == 0
            assert Path(tmp, "landing_overlay.png").exists()


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
