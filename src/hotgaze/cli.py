"""CLI entry point for HotGaze."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from PIL import Image

from . import __version__
from .config import EngineConfig
from .engine import run_engine

_SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}


def _validate_image_format(path: str) -> str:
    """Validate the image has a supported format (PNG/JPEG/WebP)."""
    suffix = Path(path).suffix.lower()
    if suffix not in _SUPPORTED_FORMATS:
        msg = (
            f"Unsupported image format: {suffix}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_FORMATS))}"
        )
        raise click.BadParameter(msg, param_hint="IMAGE")
    return path


@click.group()
@click.version_option(version=__version__, prog_name="hotgaze")
def main() -> None:
    """HotGaze — predict visual attention on UI screenshots.

    Produces attention heatmap overlays and machine-readable scores.
    """


@main.command()
@click.argument(
    "image",
    type=click.Path(exists=True),
    callback=lambda ctx, param, value: _validate_image_format(value),
)
@click.option("-o", "--output", default=None, help="Output PNG path (default: IMG_overlay.png)")
@click.option("--backend", default="fast", type=click.Choice(["fast"]), help="Saliency backend")
@click.option(
    "--alpha",
    default=0.6,
    type=click.FloatRange(0.0, 1.0),
    help="Heatmap overlay opacity (0-1)",
)
@click.option(
    "--colormap",
    default="jet",
    type=click.Choice(["jet", "turbo"]),
    help="Heatmap colormap palette",
)
def run(image: str, output: str | None, backend: str, alpha: float, colormap: str) -> None:
    """Generate an attention heatmap overlay for IMAGE."""
    if output is None:
        p = Path(image)
        output = str(p.parent / f"{p.stem}_overlay.png")

    try:
        config = EngineConfig.fast_default()
        attn = run_engine(image, config=config)

        original = _open_original(image)
        overlay = attn.overlay(original, alpha=alpha, colormap=colormap)
        overlay.save(output)

        click.echo(f"Overlay saved to {output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
def info() -> None:
    """Show version, available backends, and cache location."""
    import os

    cache_dir = os.environ.get("HOTGAZE_CACHE", str(Path.home() / ".cache" / "hotgaze"))

    click.echo(f"HotGaze v{__version__}")
    click.echo(f"Cache: {cache_dir}")
    click.echo()
    click.echo("Available backends:")
    click.echo("  fast   — heuristic (spectral residual + contrast + center bias + gaze flow)")
    click.echo("  deep   — pretrained saliency model (requires hotgaze[deep], Phase 3)")
    click.echo()
    click.echo("Run 'hotgaze run --help' for usage.")


def _open_original(path: str) -> Image.Image:
    """Open original image without alpha flattening (for overlay)."""
    img = Image.open(path)
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    return img.convert("RGB")
