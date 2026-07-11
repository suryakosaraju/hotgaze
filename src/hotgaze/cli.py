"""CLI entry point for HotGaze."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
from PIL import Image

from . import __version__
from .config import EngineConfig
from .engine import run_engine
from .scoring import (
    RegionParseError,
    compute_focal_movement,
    compute_grid_deltas,
    parse_region,
    score_regions,
    scores_to_json,
)

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
@click.argument(
    "image",
    type=click.Path(exists=True),
    callback=lambda ctx, param, value: _validate_image_format(value),
)
@click.option(
    "--region",
    multiple=True,
    help="Region to score (name:x,y,w,h or name:0.1,0.2,0.3,0.08f). Repeatable.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output canonical JSON (sorted keys, 6 dp floats).",
)
@click.option("--backend", default="fast", type=click.Choice(["fast"]), help="Saliency backend")
def score(image: str, region: tuple[str, ...], json_output: bool, backend: str) -> None:
    """Score attention on IMAGE by named regions.

    Without --json, prints a human-readable table. With --json, outputs
    canonical machine-readable JSON to stdout.
    """
    try:
        config = EngineConfig.fast_default()
        attn = run_engine(image, config=config)

        scored, focal = score_regions(attn, list(region))

        if json_output:
            result = scores_to_json(
                "score",
                image,
                attn.original_size,
                (attn.heatmap.shape[1], attn.heatmap.shape[0]),
                config.model_dump(),
                scored,
                focal,
            )
            click.echo(result, nl=False)
        else:
            _print_score_table(scored, focal)
    except RegionParseError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _print_score_table(regions: list[dict[str, Any]], focal_points: list[dict[str, Any]]) -> None:
    """Print a human-readable score table."""
    click.echo()
    click.echo("Region scores (ranked by attention share):")
    click.echo("─" * 60)
    for r in regions:
        click.echo(
            f"  #{r['rank']:<3} {r['name']:<20} share={r['share']:.4f}  peak={r['peak_value']:.4f}"
        )
    if not regions:
        click.echo("  (no regions specified)")

    click.echo()
    click.echo("Focal points:")
    click.echo("─" * 60)
    for fp in focal_points:
        click.echo(f"  #{fp['rank']:<3} ({fp['x']:>4}, {fp['y']:>4})  value={fp['value']:.4f}")
    if not focal_points:
        click.echo("  (none found)")
    click.echo()


@main.command()
@click.argument(
    "image_a",
    type=click.Path(exists=True),
    callback=lambda ctx, p, v: _validate_image_format(v),
)
@click.argument(
    "image_b",
    type=click.Path(exists=True),
    callback=lambda ctx, p, v: _validate_image_format(v),
)
@click.option(
    "--region",
    multiple=True,
    help="Region applied to both images (name:x,y,w,h or name:0.1,0.2,0.3,0.08f). Repeatable.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output canonical JSON (sorted keys, 6 dp floats).",
)
@click.option("--backend", default="fast", type=click.Choice(["fast"]), help="Saliency backend")
def compare(
    image_a: str,
    image_b: str,
    region: tuple[str, ...],
    json_output: bool,
    backend: str,
) -> None:
    """Compare attention between two images A and B.

    With --region, shows per-region attention-share deltas. Without regions,
    compares focal-point movement and a 3×3 grid of attention-share deltas.
    """
    try:
        config = EngineConfig.fast_default()
        attn_a = run_engine(image_a, config=config)
        attn_b = run_engine(image_b, config=config)

        wa, ha = attn_a.original_size
        wb, hb = attn_b.original_size

        # Check size mismatch + pixel regions
        if (wa, ha) != (wb, hb):
            for rstr in region:
                _, x, y, w, h = parse_region(rstr, wa, ha)
                # Re-parse to check if it had trailing f — parse_region converts both,
                # so check the raw string for 'f' suffix
                if not rstr.rstrip().endswith("f"):
                    raise RegionParseError(
                        f"Images have different sizes ({wa}×{ha} vs {wb}×{hb}). "
                        f"Region {rstr!r} uses pixel coordinates. "
                        "Use fractional coords with trailing f "
                        "(e.g. name:0.1,0.2,0.3,0.08f) for cross-size comparisons."
                    )

        if region:
            # Region mode: score both images with the same regions
            scored_a, _ = score_regions(attn_a, list(region))
            scored_b, _ = score_regions(attn_b, list(region))

            deltas: list[dict[str, Any]] = []
            for ra, rb in zip(scored_a, scored_b, strict=False):
                delta = round(rb["share"] - ra["share"], 6)
                deltas.append(
                    {
                        "name": ra["name"],
                        "share_a": ra["share"],
                        "share_b": rb["share"],
                        "delta": delta,
                    }
                )

            if json_output:
                result = scores_to_json(
                    "compare",
                    image_a,
                    attn_a.original_size,
                    (attn_a.heatmap.shape[1], attn_a.heatmap.shape[0]),
                    config.model_dump(),
                    scored_a,
                    [],
                    compare={
                        "per_region_deltas": deltas,
                        "grid_deltas": [],
                        "focal_point_movement": [],
                    },
                    image_b_path=image_b,
                    img_b_size=attn_b.original_size,
                    work_b_size=(attn_b.heatmap.shape[1], attn_b.heatmap.shape[0]),
                )
                click.echo(result, nl=False)
            else:
                _print_compare_table(deltas)
        else:
            # No-region mode: focal points + 3×3 grid
            _, focal_a = score_regions(attn_a, [])
            _, focal_b = score_regions(attn_b, [])
            grid_deltas = compute_grid_deltas(attn_a.heatmap, attn_b.heatmap)
            focal_movement = compute_focal_movement(focal_a, focal_b)

            if json_output:
                result = scores_to_json(
                    "compare",
                    image_a,
                    attn_a.original_size,
                    (attn_a.heatmap.shape[1], attn_a.heatmap.shape[0]),
                    config.model_dump(),
                    [],
                    [],
                    compare={
                        "per_region_deltas": [],
                        "grid_deltas": grid_deltas,
                        "focal_point_movement": focal_movement,
                    },
                    image_b_path=image_b,
                    img_b_size=attn_b.original_size,
                    work_b_size=(attn_b.heatmap.shape[1], attn_b.heatmap.shape[0]),
                )
                click.echo(result, nl=False)
            else:
                _print_grid_compare(grid_deltas, focal_movement)

    except RegionParseError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _print_compare_table(deltas: list[dict[str, Any]]) -> None:
    """Print a human-readable compare table for region mode."""
    click.echo()
    click.echo("Per-region attention-share deltas (B − A):")
    click.echo("─" * 70)
    for d in deltas:
        direction = "↑" if d["delta"] > 0 else ("↓" if d["delta"] < 0 else "=")
        click.echo(
            f"  {d['name']:<20}  A={d['share_a']:.4f}  B={d['share_b']:.4f}  "
            f"Δ={d['delta']:+.4f} {direction}"
        )
    click.echo()


def _print_grid_compare(grid_deltas: list[float], focal_movement: list[dict[str, Any]]) -> None:
    """Print a human-readable grid + focal compare."""
    click.echo()
    click.echo("3×3 grid attention-share deltas (B − A):")
    click.echo("─" * 30)
    for row in range(3):
        cells = grid_deltas[row * 3 : (row + 1) * 3]
        line = "  ".join(f"{c:+.4f}" for c in cells)
        click.echo(f"  {line}")

    click.echo()
    click.echo("Focal-point movement:")
    click.echo("─" * 60)
    for fp in focal_movement:
        click.echo(
            f"  #{fp['rank']}  A=({fp['x_a']:>4},{fp['y_a']:>4})  "
            f"B=({fp['x_b']:>4},{fp['y_b']:>4})  "
            f"dist={fp['distance']:.1f}px"
        )
    click.echo()


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
