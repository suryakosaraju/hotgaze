"""HotGaze web demo — try the attention engine from a browser.

A thin HTTP layer over the same engine, region scoring and canonical schema-v1
JSON that the CLI and the GitHub Action use. Nothing is stored: uploads live in
a temporary directory for the duration of one request.

    GET  /              single-page UI (web/static/index.html)
    GET  /healthz       liveness probe for the hosting platform
    GET  /api/info      version, available backends and server limits
    POST /api/score     one screenshot  -> overlay + score JSON    (hotgaze run + score)
    POST /api/compare   two screenshots -> overlays + compare JSON (hotgaze compare)
    GET  /samples/NAME  synthetic landing pages used by the "Try a sample" button

Run locally from the repository root:

    pip install . -r web/requirements.txt
    uvicorn app:app --app-dir web --reload
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from hotgaze import __version__
from hotgaze.attention_map import AttentionMap
from hotgaze.cli import _get_config
from hotgaze.config import EngineConfig
from hotgaze.engine import _default_deep_layers, _load_image, run_engine
from hotgaze.layers.base import SignalLayer
from hotgaze.scoring import (
    RegionParseError,
    compare_attention_maps,
    parse_region,
    score_regions,
    scores_to_json,
)


def _int_setting(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


# Defaults fit a 512 MB instance: a 10 MP compare peaks around 390 MB RSS with the
# malloc settings in render.yaml.
MAX_PIXELS = _int_setting("HOTGAZE_WEB_MAX_PIXELS", 10_000_000)
MAX_UPLOAD_BYTES = _int_setting("HOTGAZE_WEB_MAX_UPLOAD_MB", 20) * 1024 * 1024
MAX_JOBS = _int_setting("HOTGAZE_WEB_MAX_JOBS", 1)
QUEUE_TIMEOUT_SECONDS = 120.0

DEEP_AVAILABLE = importlib.util.find_spec("torch") is not None

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_SCRIPT = _REPO_ROOT / "scripts" / "generate_fixtures.py"
_SAMPLE_DIR = _REPO_ROOT / "tests" / "fixtures"
SAMPLES = ("landing.png", "landing_variant.png")

# Formats accepted by the CLI, keyed by the name Pillow reports.
_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
# Large enough that only malformed regions fail; real bounds are checked once
# the image is loaded.
_SYNTAX_CHECK_SIZE = 1 << 30

_REGION_HELP = (
    "Region to score: name:x,y,w,h in pixels or name:0.1,0.2,0.3,0.08f as fractions. "
    "Repeat the field (or separate with newlines) for several regions."
)

Backend = Literal["fast", "deep"]
Colormap = Literal["jet", "turbo"]

_job_slots = threading.BoundedSemaphore(MAX_JOBS)
_deep_lock = threading.Lock()
_deep_layers: dict[str, SignalLayer] | None = None
_sample_lock = threading.Lock()

app = FastAPI(
    title="HotGaze web demo",
    version=__version__,
    description=(
        "Attention heatmaps for UI screenshots, with numeric scores and A/B compare. "
        "Responses embed the same canonical schema-v1 JSON as `hotgaze score --json` "
        "and `hotgaze compare --json`."
    ),
)


@dataclass(frozen=True)
class _Upload:
    path: Path
    name: str


# ── Routes ───────────────────────────────────────────────────────────────────


# The lightweight routes are async so they are answered on the event loop even
# while queued analyses occupy the threadpool.


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/info")
async def info() -> dict[str, Any]:
    """Version, available backends and the limits this server enforces."""
    return {
        "version": __version__,
        "backends": ["fast", "deep"] if DEEP_AVAILABLE else ["fast"],
        "layers": ["faces"],
        "limits": {
            "max_pixels": MAX_PIXELS,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_concurrent_jobs": MAX_JOBS,
        },
        "samples": list(SAMPLES) if _SAMPLE_SCRIPT.exists() else [],
    }


@app.post("/api/score")
def score(
    image: Annotated[UploadFile, File(description="PNG, JPEG or WebP screenshot.")],
    region: Annotated[list[str] | None, Form(description=_REGION_HELP)] = None,
    backend: Annotated[Backend, Form()] = "fast",
    faces: Annotated[bool, Form(description="Enable the YuNet faces layer.")] = False,
    alpha: Annotated[float, Form(ge=0.0, le=1.0, description="Overlay opacity.")] = 0.6,
    colormap: Annotated[Colormap, Form()] = "jet",
) -> dict[str, Any]:
    """Score one screenshot: overlay image plus the schema-v1 score document."""
    regions = _parse_regions(region)
    config = _engine_config(backend, faces)
    with tempfile.TemporaryDirectory(prefix="hotgaze-") as tmp:
        upload = _save_upload(image, Path(tmp), "image")
        with _job_slot(), _analysis_errors():
            started = time.perf_counter()
            attn = _run_engine(upload.path, config)
            document = _score_document(upload.name, attn, config, regions)
            overlay = _overlay_data_url(attn, upload.path, alpha, colormap)
            elapsed = time.perf_counter() - started
    return {"score": document, "overlay": overlay, "elapsed_seconds": round(elapsed, 3)}


@app.post("/api/compare")
def compare(
    image_a: Annotated[UploadFile, File(description="Baseline screenshot (A).")],
    image_b: Annotated[UploadFile, File(description="Candidate screenshot (B).")],
    region: Annotated[list[str] | None, Form(description=_REGION_HELP)] = None,
    backend: Annotated[Backend, Form()] = "fast",
    faces: Annotated[bool, Form(description="Enable the YuNet faces layer.")] = False,
    alpha: Annotated[float, Form(ge=0.0, le=1.0, description="Overlay opacity.")] = 0.6,
    colormap: Annotated[Colormap, Form()] = "jet",
) -> dict[str, Any]:
    """Compare two screenshots: both overlays, both score documents and the compare document.

    With regions, the compare document holds per-region share deltas (B − A); without
    regions, a 3×3 grid of share deltas and focal-point movement. Mirrors the files the
    GitHub Action writes.
    """
    regions = _parse_regions(region)
    config = _engine_config(backend, faces)
    with tempfile.TemporaryDirectory(prefix="hotgaze-") as tmp:
        upload_a = _save_upload(image_a, Path(tmp), "a")
        upload_b = _save_upload(image_b, Path(tmp), "b")
        with _job_slot(), _analysis_errors():
            started = time.perf_counter()
            attn_a = _run_engine(upload_a.path, config)
            overlay_a = _overlay_data_url(attn_a, upload_a.path, alpha, colormap)
            attn_b = _run_engine(upload_b.path, config)
            overlay_b = _overlay_data_url(attn_b, upload_b.path, alpha, colormap)
            compare_regions, _, comparison = compare_attention_maps(attn_a, attn_b, regions)
            compare_document = scores_to_json(
                "compare",
                upload_a.name,
                attn_a.original_size,
                attn_a.working_size,
                config.model_dump(),
                compare_regions,
                [],
                compare=comparison,
                image_b_path=upload_b.name,
                img_b_size=attn_b.original_size,
                work_b_size=attn_b.working_size,
            )
            score_a = _score_document(upload_a.name, attn_a, config, regions)
            score_b = _score_document(upload_b.name, attn_b, config, regions)
            elapsed = time.perf_counter() - started
    return {
        "compare": json.loads(compare_document),
        "score_a": score_a,
        "score_b": score_b,
        "overlay_a": overlay_a,
        "overlay_b": overlay_b,
        "elapsed_seconds": round(elapsed, 3),
    }


@app.get("/samples/{name}", include_in_schema=False)
def sample(name: str) -> FileResponse:
    if name not in SAMPLES or not _SAMPLE_SCRIPT.exists():
        raise HTTPException(404, "Unknown sample")
    path = _SAMPLE_DIR / name
    with _sample_lock:
        if not path.exists():
            # Same generator the test suite uses; the Render build runs it ahead of time.
            subprocess.run(
                [sys.executable, str(_SAMPLE_SCRIPT)],
                cwd=_REPO_ROOT,
                check=True,
                capture_output=True,
                timeout=60,
            )
    return FileResponse(path, media_type="image/png")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_regions(values: list[str] | None) -> list[str]:
    """Flatten repeated/newline-separated region fields and reject malformed ones early."""
    regions = [line.strip() for value in values or [] for line in value.splitlines()]
    regions = [region for region in regions if region]
    for region in regions:
        try:
            parse_region(region, _SYNTAX_CHECK_SIZE, _SYNTAX_CHECK_SIZE)
        except RegionParseError as exc:
            raise HTTPException(400, str(exc)) from exc
    return regions


def _engine_config(backend: str, faces: bool) -> EngineConfig:
    if backend == "deep" and not DEEP_AVAILABLE:
        raise HTTPException(
            400,
            "The deep backend is not installed on this server (PyTorch is missing). "
            "Use the fast backend, or redeploy with HOTGAZE_ENABLE_DEEP=true.",
        )
    return _get_config(backend, ["faces"] if faces else None)


def _save_upload(upload: UploadFile, directory: Path, stem: str) -> _Upload:
    """Write an upload to disk after checking its size, format and pixel count."""
    name = Path(upload.filename or "").name[:120] or f"{stem}.png"
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{name}: file exceeds {MAX_UPLOAD_BYTES // 2**20} MB.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            image_format, (width, height) = img.format, img.size
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, f"{name}: {exc}") from exc
    except Exception as exc:
        raise HTTPException(400, f"{name} is not a readable PNG, JPEG or WebP image.") from exc
    if image_format not in _FORMATS:
        raise HTTPException(
            400, f"{name}: unsupported image format {image_format}. Supported: PNG, JPEG, WebP."
        )
    if width * height > MAX_PIXELS:
        raise HTTPException(
            413,
            f"{name} is {width}×{height} ({width * height / 1e6:.1f} MP); this server accepts "
            f"up to {MAX_PIXELS / 1e6:g} MP. Crop or downscale the screenshot.",
        )
    path = directory / f"{stem}{_FORMATS[image_format]}"
    path.write_bytes(data)
    return _Upload(path, name)


@contextmanager
def _job_slot() -> Iterator[None]:
    """Bound concurrent analyses so a burst of uploads cannot exhaust memory."""
    if not _job_slots.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
        raise HTTPException(503, "The server is busy with other analyses. Try again shortly.")
    try:
        yield
    finally:
        _job_slots.release()


@contextmanager
def _analysis_errors() -> Iterator[None]:
    """Report HotGaze's own error messages with a matching HTTP status."""
    try:
        yield
    except ValueError as exc:  # bad regions, mismatched sizes, invalid image data
        raise HTTPException(400, str(exc)) from exc
    except (ImportError, FileNotFoundError) as exc:  # deep backend or model weights unavailable
        raise HTTPException(503, str(exc)) from exc
    except OSError as exc:  # Pillow could not decode the upload
        raise HTTPException(400, f"Could not decode the image: {exc}") from exc


def _run_engine(path: Path, config: EngineConfig) -> AttentionMap:
    layers = _load_deep_layers() if config.backend == "deep" else None
    return run_engine(str(path), config=config, layers=layers)


def _load_deep_layers() -> dict[str, SignalLayer]:
    """Load UNISAL once per process; run_engine would otherwise rebuild it per image."""
    global _deep_layers
    with _deep_lock:
        if _deep_layers is None:
            _deep_layers = _default_deep_layers()
        return _deep_layers


def _score_document(
    name: str, attn: AttentionMap, config: EngineConfig, regions: list[str]
) -> dict[str, Any]:
    scored, focal = score_regions(attn, regions)
    document = scores_to_json(
        "score", name, attn.original_size, attn.working_size, config.model_dump(), scored, focal
    )
    result: dict[str, Any] = json.loads(document)
    return result


def _overlay_data_url(attn: AttentionMap, path: Path, alpha: float, colormap: str) -> str:
    """Render the overlay exactly like `hotgaze run`, as JPEG to keep responses small."""
    overlay = attn.overlay(_load_image(str(path)), alpha=alpha, colormap=colormap)
    buffer = io.BytesIO()
    overlay.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
