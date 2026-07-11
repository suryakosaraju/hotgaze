"""Weight file download, checksum verification, and cache management.

Core-package code — imports only stdlib + click. No torch, no onnxruntime.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import NamedTuple

import click

# ── Placeholder URLs ─────────────────────────────────────────────────────────
#
# TODO(orchestrator): replace with real GitHub Release URLs before first publish.
# These must point to the HotGaze GitHub Releases page, not the upstream repos.
# We re-host weight files on our own GitHub Release per CLAUDE.md hosting plan
# (stable URL, checksum-pinnable, no Google Drive).
#
_PLACEHOLDER_BASE = "https://github.com/hotgaze/hotgaze/releases/download/v0.1.0"

_UNISAL_URL = f"{_PLACEHOLDER_BASE}/weights_best.pth"
_UNISAL_MIT1003_URL = f"{_PLACEHOLDER_BASE}/weights_ft_mit1003.pth"
_YUNET_URL = f"{_PLACEHOLDER_BASE}/face_detection_yunet_2023mar.onnx"

# ── Registry ─────────────────────────────────────────────────────────────────


class WeightSpec(NamedTuple):
    """A registered weight file."""

    name: str  # registry key
    filename: str  # basename on disk
    sha256: str  # expected SHA-256 hex digest
    url: str  # download URL


_WEIGHTS: dict[str, WeightSpec] = {
    "unisal": WeightSpec(
        name="unisal",
        filename="weights_best.pth",
        sha256="4a9157411f1741d588b15670d15295e998805648b8b6348599fe447298338481",
        url=_UNISAL_URL,
    ),
    "unisal_mit1003": WeightSpec(
        name="unisal_mit1003",
        filename="weights_ft_mit1003.pth",
        sha256="a6de8ea27d812cfc3fbc2b8cab59862a8e48aaf4d512e670468d3b6972a81262",
        url=_UNISAL_MIT1003_URL,
    ),
    "yunet": WeightSpec(
        name="yunet",
        filename="face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        url=_YUNET_URL,
    ),
}


def get_weight_spec(name: str) -> WeightSpec:
    """Look up a weight by registry name."""
    if name not in _WEIGHTS:
        raise KeyError(f"Unknown weight: {name!r}. Registered: {list(_WEIGHTS.keys())}")
    return _WEIGHTS[name]


def get_cache_dir() -> Path:
    """Return the cache directory (respects HOTGAZE_CACHE env var)."""
    env = os.environ.get("HOTGAZE_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "hotgaze"


# ── Download ─────────────────────────────────────────────────────────────────


def download_weight(
    name: str,
    cache_dir: Path | None = None,
    progress: bool = True,
) -> Path:
    """Download a weight file to the cache, verifying its SHA-256.

    If already cached (file exists at the final path), returns immediately
    with zero network calls.  Otherwise, downloads to a temp file in the
    cache directory, verifies the checksum, then atomically renames into
    place.  A partial download is never left at the final path.

    Args:
        name: Registry key (e.g. ``"unisal"``, ``"yunet"``).
        cache_dir: Cache directory.  Defaults to ``~/.cache/hotgaze/``.
        progress: Show a ``click.progressbar`` during download.

    Returns:
        Path to the cached file.

    Raises:
        KeyError: Unknown weight name.
        FileNotFoundError: Network unreachable / URL not found.
        ValueError: Checksum mismatch.
    """
    spec = get_weight_spec(name)
    if cache_dir is None:
        cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    dest = cache_dir / spec.filename

    # Cache hit — skip download
    if dest.exists():
        return dest

    # Download to temp file, verify, then atomically rename
    tmp = cache_dir / f".{spec.filename}.tmp"

    try:
        _download_to(spec.url, tmp, progress=progress)
        _verify_checksum(tmp, spec.sha256, spec.name)
        os.replace(tmp, dest)
    except Exception:
        # Clean up temp file on any failure
        if tmp.exists():
            tmp.unlink()
        raise

    return dest


def _download_to(url: str, dest: Path, progress: bool) -> None:
    """Download a URL to a file, optionally showing a progress bar."""
    import urllib.request

    try:
        response = urllib.request.urlopen(url, timeout=30)  # noqa: S310
    except OSError as e:
        raise FileNotFoundError(
            f"Cannot download weight from {url}\n"
            f"Network error: {e}\n"
            f"Is the machine online? The deep backend requires a one-time download.\n"
            f"File: {dest.name}"
        ) from e

    total = response.headers.get("Content-Length")
    total_bytes = int(total) if total else None

    label = f"Downloading {dest.name}"

    if progress and total_bytes:
        with click.progressbar(length=total_bytes, label=label) as bar, dest.open("wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    else:
        with dest.open("wb") as f:
            shutil.copyfileobj(response, f)


def _verify_checksum(path: Path, expected: str, name: str) -> None:
    """Verify the SHA-256 digest of a file. Raise ValueError on mismatch."""
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        path.unlink()
        raise ValueError(
            f"Checksum mismatch for {name}:\n"
            f"  Expected: {expected}\n"
            f"  Got:      {actual}\n"
            f"  File deleted: {path}"
        )
