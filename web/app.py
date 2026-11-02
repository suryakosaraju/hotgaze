"""焦点 Focal: attention heatmaps, background removal and erase in one web app.

The heatmap tool runs the HotGaze engine in this process. Background removal and
erase are forwarded over the private network to the IOPaint service in
services/iopaint. Nothing is stored: uploads live in memory or a temporary
directory for the duration of one request.

    GET  /, /heatmap, /cutout, /erase   single-page UI (web/static/index.html)
    GET  /healthz                       liveness probe for the hosting platform
    GET  /api/info                      version, available tools and server limits
    POST /api/score                     one screenshot  -> overlay + score JSON (hotgaze score)
    POST /api/compare                   two screenshots -> overlays + compare JSON
    POST /api/cutout                    image -> PNG with a transparent background (RMBG-1.4)
    POST /api/erase                     image + mask -> image with the painted area filled (LaMa)
    GET  /samples/NAME                  synthetic landing pages for the heatmap samples

Errors are JSON, {"detail": {"code": ..., "message": ..., "params": {...}}}, so the
UI can explain them in Chinese or English.

Run locally from the repository root:

    pip install . -r web/requirements.txt
    uvicorn app:app --app-dir web --reload

Cutout and erase also need IOPAINT_URL pointing at a running services/iopaint container.
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
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from hotgaze import __version__
from hotgaze.attention_map import AttentionMap
from hotgaze.cli import _get_config
from hotgaze.config import EngineConfig
from hotgaze.engine import _default_deep_layers, _load_image, run_engine
from hotgaze.layers.base import SignalLayer
from hotgaze.render import render_overlay
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


def _service_url(raw: str) -> str:
    """Accept a full URL or the bare host:port Render gives for a private service."""
    url = raw.strip().rstrip("/")
    if url and "://" not in url:
        url = f"http://{url}"
    return url


# Defaults fit a 512 MB instance: a 10 MP compare peaks around 390 MB RSS with the
# malloc settings in render.yaml.
MAX_PIXELS = _int_setting("HOTGAZE_WEB_MAX_PIXELS", 10_000_000)
MAX_UPLOAD_BYTES = _int_setting("HOTGAZE_WEB_MAX_UPLOAD_MB", 20) * 1024 * 1024
MAX_JOBS = _int_setting("HOTGAZE_WEB_MAX_JOBS", 1)
QUEUE_TIMEOUT_SECONDS = 120.0

# Cutout and erase accept larger photos and downscale them before IOPaint sees them;
# the IOPaint service is sized for at most 2048 px on the long side.
EDIT_MAX_PIXELS = 25_000_000
EDIT_MAX_EDGE = 2048
IOPAINT_URL = _service_url(os.environ.get("IOPAINT_URL", ""))
# After a deploy IOPaint loads its first model for over a minute on one CPU, and a
# large erase takes up to ~40 s, so requests may legitimately wait a while.
IOPAINT_TIMEOUT_SECONDS = 300.0

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
# HotGaze raises plain ValueErrors; these phrases identify the ones the UI explains.
_LIBRARY_ERRORS = (
    ("no regions were supplied", "size_mismatch"),
    ("uses pixel coordinates", "pixel_region_size_mismatch"),
    ("Duplicate region name", "duplicate_region"),
    ("fully out of bounds", "region_out_of_bounds"),
)

_REGION_HELP = (
    "Region to score: name:x,y,w,h in pixels or name:0.1,0.2,0.3,0.08f as fractions. "
    "Repeat the field (or separate with newlines) for several regions."
)

Backend = Literal["fast", "deep"]
Colormap = Literal["jet", "turbo"]
OverlayStyle = Literal["focus", "classic"]

# In the focus overlay, attention below this fraction of the peak leaves the page
# untinted, so text in quiet areas stays readable.
_FOCUS_FLOOR = 0.2
_OVERLAY_HELP = (
    "focus: the colormap is blended in proportion to attention and `alpha` is its "
    "opacity at the peak; classic: exactly `hotgaze run`, `alpha` everywhere."
)

_job_slots = threading.BoundedSemaphore(MAX_JOBS)
# The IOPaint service keeps one model in memory and serves one request at a time.
_iopaint_slots = threading.BoundedSemaphore(1)
# Private-network calls must never go through an HTTP proxy from the environment.
_iopaint_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_deep_lock = threading.Lock()
_deep_layers: dict[str, SignalLayer] | None = None
_sample_lock = threading.Lock()

app = FastAPI(
    title="焦点 Focal",
    version=__version__,
    description=(
        "Attention heatmaps (HotGaze), background removal and erase (IOPaint). "
        "Heatmap responses embed the same canonical schema-v1 JSON as `hotgaze score --json` "
        "and `hotgaze compare --json`; cutout and erase return the image itself."
    ),
)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@dataclass(frozen=True)
class _Upload:
    data: bytes
    name: str
    format: str


@dataclass(frozen=True)
class _Saved:
    path: Path
    name: str


# ── Routes ───────────────────────────────────────────────────────────────────


# The lightweight routes are async so they are answered on the event loop even
# while queued analyses occupy the threadpool.


@app.get("/", include_in_schema=False)
@app.get("/heatmap", include_in_schema=False)
@app.get("/cutout", include_in_schema=False)
@app.get("/erase", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/info")
async def info() -> dict[str, Any]:
    """Version, available tools and the limits this server enforces."""
    return {
        "version": __version__,
        "tools": {"heatmap": True, "cutout": bool(IOPAINT_URL), "erase": bool(IOPAINT_URL)},
        "backends": ["fast", "deep"] if DEEP_AVAILABLE else ["fast"],
        "layers": ["faces"],
        "limits": {
            "max_pixels": MAX_PIXELS,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_concurrent_jobs": MAX_JOBS,
            "edit_max_pixels": EDIT_MAX_PIXELS,
            "edit_max_edge": EDIT_MAX_EDGE,
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
    overlay: Annotated[OverlayStyle, Form(description=_OVERLAY_HELP)] = "focus",
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
            overlay_url = _overlay_data_url(attn, upload.path, alpha, colormap, overlay)
            elapsed = time.perf_counter() - started
    return {"score": document, "overlay": overlay_url, "elapsed_seconds": round(elapsed, 3)}


@app.post("/api/compare")
def compare(
    image_a: Annotated[UploadFile, File(description="Baseline screenshot (A).")],
    image_b: Annotated[UploadFile, File(description="Candidate screenshot (B).")],
    region: Annotated[list[str] | None, Form(description=_REGION_HELP)] = None,
    backend: Annotated[Backend, Form()] = "fast",
    faces: Annotated[bool, Form(description="Enable the YuNet faces layer.")] = False,
    alpha: Annotated[float, Form(ge=0.0, le=1.0, description="Overlay opacity.")] = 0.6,
    colormap: Annotated[Colormap, Form()] = "jet",
    overlay: Annotated[OverlayStyle, Form(description=_OVERLAY_HELP)] = "focus",
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
            overlay_a = _overlay_data_url(attn_a, upload_a.path, alpha, colormap, overlay)
            attn_b = _run_engine(upload_b.path, config)
            overlay_b = _overlay_data_url(attn_b, upload_b.path, alpha, colormap, overlay)
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


@app.post("/api/cutout", response_class=Response)
def cutout(
    image: Annotated[UploadFile, File(description="PNG, JPEG or WebP photo.")],
) -> Response:
    """Remove the background with BRIA RMBG-1.4 (non-commercial use only); returns a PNG.

    Photos larger than 2048 px on the long side are downscaled first.
    """
    upload = _read_upload(image, "image", EDIT_MAX_PIXELS)
    with _job_slot():
        picture = _fit(_decode(upload, "RGB"), Image.Resampling.LANCZOS)
        payload = {"name": "RemoveBG", "image": _b64(picture, "JPEG")}
    started = time.perf_counter()
    body = _iopaint("/api/v1/run_plugin_gen_image", payload)
    return _image_response(body, "image/png", picture.size, started)


@app.post("/api/erase", response_class=Response)
def erase(
    image: Annotated[UploadFile, File(description="PNG, JPEG or WebP photo.")],
    mask: Annotated[
        UploadFile, File(description="Same size as the image; white marks what to erase.")
    ],
) -> Response:
    """Fill the painted area with LaMa; returns JPEG for JPEG input, PNG otherwise.

    Photos larger than 2048 px on the long side are downscaled first.
    """
    upload = _read_upload(image, "image", EDIT_MAX_PIXELS)
    painted = _read_upload(mask, "mask", EDIT_MAX_PIXELS)
    output = "JPEG" if upload.format == "JPEG" else "PNG"
    with _job_slot():
        picture = _decode(upload, "RGB")
        stroke = _decode(painted, "L")
        if stroke.size != picture.size:
            raise _error(
                400,
                "mask_size_mismatch",
                f"The mask is {_dims(stroke.size)} but the image is {_dims(picture.size)}.",
                image=_dims(picture.size),
                mask=_dims(stroke.size),
            )
        picture = _fit(picture, Image.Resampling.LANCZOS)
        stroke = _fit(stroke, Image.Resampling.NEAREST).point(lambda v: 255 if v >= 128 else 0)
        if stroke.getbbox() is None:
            raise _error(400, "empty_mask", "Paint over the area to erase first.")
        payload = {"image": _b64(picture, output), "mask": _b64(stroke, "PNG")}
    started = time.perf_counter()
    body = _iopaint("/api/v1/inpaint", payload)
    media_type = "image/jpeg" if output == "JPEG" else "image/png"
    return _image_response(body, media_type, picture.size, started)


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


def _error(status: int, code: str, message: str, **params: Any) -> HTTPException:
    """An HTTP error the UI can translate: a stable code, an English message, its values."""
    return HTTPException(status, {"code": code, "message": message, "params": params})


def _parse_regions(values: list[str] | None) -> list[str]:
    """Flatten repeated/newline-separated region fields and reject malformed ones early."""
    regions = [line.strip() for value in values or [] for line in value.splitlines()]
    regions = [region for region in regions if region]
    for region in regions:
        try:
            parse_region(region, _SYNTAX_CHECK_SIZE, _SYNTAX_CHECK_SIZE)
        except RegionParseError as exc:
            raise _error(400, "invalid_region", str(exc), region=region) from exc
    return regions


def _engine_config(backend: str, faces: bool) -> EngineConfig:
    if backend == "deep" and not DEEP_AVAILABLE:
        raise _error(
            400,
            "deep_unavailable",
            "The deep backend is not installed on this server (PyTorch is missing). "
            "Use the fast backend, or redeploy with HOTGAZE_ENABLE_DEEP=true.",
        )
    return _get_config(backend, ["faces"] if faces else None)


def _read_upload(upload: UploadFile, stem: str, max_pixels: int) -> _Upload:
    """Read an upload after checking its size, format and pixel count."""
    name = Path(upload.filename or "").name[:120] or f"{stem}.png"
    limit_mb = MAX_UPLOAD_BYTES // 2**20
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise _error(
            413,
            "file_too_large",
            f"{name}: file exceeds {limit_mb} MB.",
            name=name,
            limit_mb=limit_mb,
        )
    limit_mp = round(max_pixels / 1e6, 1)
    try:
        with Image.open(io.BytesIO(data)) as img:
            image_format, (width, height) = img.format, img.size
    except Image.DecompressionBombError as exc:
        raise _error(
            413, "too_many_pixels", f"{name}: {exc}", name=name, limit_megapixels=limit_mp
        ) from exc
    except Exception as exc:
        raise _error(
            400, "unreadable_image", f"{name} is not a readable PNG, JPEG or WebP image.", name=name
        ) from exc
    if image_format not in _FORMATS:
        raise _error(
            400,
            "unsupported_format",
            f"{name}: unsupported image format {image_format}. Supported: PNG, JPEG, WebP.",
            name=name,
            format=image_format,
        )
    if width * height > max_pixels:
        megapixels = round(width * height / 1e6, 1)
        raise _error(
            413,
            "too_many_pixels",
            f"{name} is {width}×{height} ({megapixels} MP); this server accepts up to "
            f"{max_pixels / 1e6:g} MP. Crop or downscale the image.",
            name=name,
            width=width,
            height=height,
            megapixels=megapixels,
            limit_megapixels=limit_mp,
        )
    return _Upload(data, name, image_format)


def _save_upload(upload: UploadFile, directory: Path, stem: str) -> _Saved:
    """Write a heatmap upload to disk, where the engine reads it."""
    checked = _read_upload(upload, stem, MAX_PIXELS)
    path = directory / f"{stem}{_FORMATS[checked.format]}"
    path.write_bytes(checked.data)
    return _Saved(path, checked.name)


@contextmanager
def _job_slot() -> Iterator[None]:
    """Bound concurrent work in this process so a burst of uploads cannot exhaust memory."""
    if not _job_slots.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
        raise _error(503, "busy", "The server is busy with other images. Try again shortly.")
    try:
        yield
    finally:
        _job_slots.release()


@contextmanager
def _analysis_errors() -> Iterator[None]:
    """Report HotGaze's own error messages with a matching HTTP status and code."""
    try:
        yield
    except ValueError as exc:  # bad regions, mismatched sizes, invalid image data
        message = str(exc)
        code = next((c for phrase, c in _LIBRARY_ERRORS if phrase in message), "invalid_input")
        raise _error(400, code, message) from exc
    except (ImportError, FileNotFoundError) as exc:  # deep backend or model weights unavailable
        raise _error(503, "model_unavailable", str(exc)) from exc
    except OSError as exc:  # Pillow could not decode the upload
        raise _error(
            400, "unreadable_image", f"Could not decode the image: {exc}", name=""
        ) from exc


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


def _overlay_data_url(
    attn: AttentionMap, path: Path, alpha: float, colormap: str, style: str
) -> str:
    """Render the overlay as a JPEG data URL (JPEG keeps responses small)."""
    original = _load_image(str(path))
    if style == "classic":
        overlay = attn.overlay(original, alpha=alpha, colormap=colormap)
    else:
        overlay = _focus_overlay(attn, original, alpha, colormap)
    buffer = io.BytesIO()
    overlay.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _focus_overlay(
    attn: AttentionMap, original: Image.Image, alpha: float, colormap: str
) -> Image.Image:
    """`hotgaze run`'s colormap, blended in proportion to attention.

    Where attention is low the screenshot shows through untouched; the colour reaches
    `alpha` at the peak. Everything stays 8-bit, so memory matches the classic overlay.
    """
    size = attn.original_size
    colored = render_overlay(attn.heatmap, original, size, alpha=1.0, colormap=colormap)
    ramp = np.clip((attn.heatmap - _FOCUS_FLOOR) / (1.0 - _FOCUS_FLOOR), 0.0, 1.0) * alpha
    weight = Image.fromarray(np.round(ramp * 255).astype(np.uint8))
    weight = weight.resize(size, Image.Resampling.BILINEAR)
    return Image.composite(colored, original.convert("RGB").resize(size), weight)


def _decode(upload: _Upload, mode: str) -> Image.Image:
    """Decode an upload the way the browser shows it: EXIF-oriented, in `mode`."""
    try:
        with Image.open(io.BytesIO(upload.data)) as img:
            oriented = ImageOps.exif_transpose(img)
            return oriented.convert(mode)
    except OSError as exc:
        raise _error(
            400, "unreadable_image", f"Could not decode {upload.name}: {exc}", name=upload.name
        ) from exc


def _fit(image: Image.Image, resample: Image.Resampling) -> Image.Image:
    """Downscale so the long side is at most EDIT_MAX_EDGE; never upscale."""
    width, height = image.size
    scale = EDIT_MAX_EDGE / max(width, height)
    if scale >= 1:
        return image
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), resample)


def _b64(image: Image.Image, image_format: str) -> str:
    buffer = io.BytesIO()
    if image_format == "JPEG":
        image.save(buffer, format="JPEG", quality=95)
    else:
        # Fast compression: on a 0.1 CPU instance the default level costs seconds.
        image.save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _dims(size: tuple[int, int]) -> str:
    return f"{size[0]}×{size[1]}"


def _iopaint(path: str, payload: dict[str, Any]) -> bytes:
    """POST to the IOPaint service, one request at a time, and return the image it sends."""
    if not IOPAINT_URL:
        raise _error(
            503,
            "tool_unavailable",
            "Cutout and erase are not configured on this server (IOPAINT_URL is not set).",
        )
    request = urllib.request.Request(
        IOPAINT_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if not _iopaint_slots.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
        raise _error(503, "busy", "The image service is busy with other images. Try again shortly.")
    try:
        with _iopaint_opener.open(request, timeout=IOPAINT_TIMEOUT_SECONDS) as response:
            body: bytes = response.read()
            return body
    except urllib.error.HTTPError as exc:
        reason = _service_detail(exc.read(2000))
        if exc.code < 500:
            raise _error(400, "tool_rejected", reason, reason=reason) from exc
        raise _error(502, "tool_failed", reason, reason=reason) from exc
    except OSError as exc:  # refused, unreachable or timed out
        raise _error(503, "tool_unavailable", f"The image service did not respond: {exc}") from exc
    finally:
        _iopaint_slots.release()


def _service_detail(body: bytes) -> str:
    """The `detail` of a FastAPI error body, or the raw text."""
    text = body.decode("utf-8", "replace")
    try:
        detail = json.loads(text).get("detail", text)
    except (ValueError, AttributeError):
        return text
    return detail if isinstance(detail, str) else json.dumps(detail)


def _image_response(
    body: bytes, media_type: str, size: tuple[int, int], started: float
) -> Response:
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Focal-Size": f"{size[0]}x{size[1]}",
            "X-Focal-Elapsed": f"{time.perf_counter() - started:.3f}",
        },
    )
