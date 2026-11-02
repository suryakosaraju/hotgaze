"""IOPaint for a 2 GB instance: one model in memory at a time.

`iopaint start` keeps LaMa and the RemoveBG model loaded together, which peaks
above 2 GB under a hard memory limit. This entry point serves the two IOPaint
endpoints the Focal site calls, with IOPaint's own request schemas, model code
and image helpers, but holds only the model the current request needs:

    POST /api/v1/inpaint               erase with LaMa (InpaintRequest)
    POST /api/v1/run_plugin_gen_image  remove the background with BRIA RMBG-1.4
                                       (RunPluginRequest, name "RemoveBG")
    GET  /healthz

Requests run one at a time. Measured peaks under a 2 GB limit: about 1.75 GB
for LaMa and 1.87 GB for RMBG-1.4. On an instance with 4 GB or more,
`iopaint start` serves the same endpoints with both models resident.
"""

from __future__ import annotations

import ctypes
import gc
import math
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from iopaint.helper import concat_alpha_channel, decode_base64_to_image, pil_to_bytes
from iopaint.model_manager import ModelManager
from iopaint.plugins.remove_bg import RemoveBG
from iopaint.schema import HDStrategy, InpaintRequest, RunPluginRequest
from PIL import Image

DEVICE = torch.device("cpu")
REMOVE_BG_MODEL = os.environ.get("REMOVE_BG_MODEL", "briaai/RMBG-1.4")
MAX_EDGE = int(os.environ.get("IOPAINT_MAX_EDGE", "2048"))
# LaMa's memory grows with the area it processes. Past this many pixels the
# erase runs on a downscaled copy and only the painted area is pasted back;
# 0.75 MP keeps the erase peak near 1.7 GB, leaving headroom under 2 GB.
MAX_LAMA_PIXELS = 768 * 1024
JPEG_QUALITY = 95
# Loaded in the background at startup: the first load after a cold start reads
# the weights from disk and takes over a minute on one CPU; later swaps take seconds.
PRELOAD = os.environ.get("IOPAINT_PRELOAD", "removebg")

_lock = threading.Lock()
_loaded: dict[str, Any] = {}


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    if PRELOAD in ("lama", "removebg"):
        threading.Thread(target=_preload, daemon=True).start()
    yield


app = FastAPI(
    title="Focal IOPaint", docs_url=None, redoc_url=None, openapi_url=None, lifespan=_lifespan
)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "loaded": next(iter(_loaded), None)}


@app.post("/api/v1/inpaint")
def inpaint(req: InpaintRequest) -> Response:
    image, alpha_channel, infos, ext = _decode(req.image)
    mask, _, _, _ = _decode(req.mask, gray=True)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    if image.shape[:2] != mask.shape[:2]:
        raise HTTPException(
            400, f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match."
        )
    _bound_lama_area(req, mask)
    with _lock:
        bgr = _model("lama")(image, mask, req)
        _release_memory()
    rgb = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    rgb = concat_alpha_channel(rgb, alpha_channel)
    body = pil_to_bytes(Image.fromarray(rgb), ext=ext, quality=JPEG_QUALITY, infos=infos)
    return Response(content=body, media_type=f"image/{ext}")


@app.post("/api/v1/run_plugin_gen_image")
def run_plugin_gen_image(req: RunPluginRequest) -> Response:
    if req.name != "RemoveBG":
        raise HTTPException(422, "Plugin not found")
    rgb, alpha_channel, infos, _ = _decode(req.image)
    with _lock:
        result = _model("removebg").gen_image(rgb, req)
        _release_memory()
    if result.shape[2] != 4:
        result = concat_alpha_channel(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), alpha_channel)
    body = pil_to_bytes(Image.fromarray(result), ext="png", quality=JPEG_QUALITY, infos=infos)
    return Response(content=body, media_type="image/png")


def _decode(encoding: str | None, gray: bool = False) -> tuple[np.ndarray, Any, dict, str]:
    if not encoding:
        raise HTTPException(400, "An image is required.")
    try:
        image, alpha_channel, infos, ext = decode_base64_to_image(encoding, gray=gray)
    except Exception as exc:
        raise HTTPException(400, f"Could not decode the image: {exc}") from exc
    height, width = image.shape[:2]
    if max(height, width) > MAX_EDGE:
        raise HTTPException(
            413, f"Image is {width}×{height}; the longest side may be at most {MAX_EDGE} px."
        )
    return image, alpha_channel, infos, ext


def _bound_lama_area(req: InpaintRequest, mask: np.ndarray) -> None:
    """Switch to the Resize strategy when LaMa would otherwise process too many pixels."""
    height, width = mask.shape[:2]
    long_edge = max(height, width)
    ys, xs = np.nonzero(mask)
    if req.hd_strategy == HDStrategy.CROP and long_edge > req.hd_strategy_crop_trigger_size:
        if xs.size == 0:
            return
        margin = 2 * req.hd_strategy_crop_margin
        area = min(width, int(np.ptp(xs)) + 1 + margin) * min(height, int(np.ptp(ys)) + 1 + margin)
    elif req.hd_strategy == HDStrategy.RESIZE and long_edge > req.hd_strategy_resize_limit:
        area = height * width * (req.hd_strategy_resize_limit / long_edge) ** 2
    else:
        area = height * width
    if area > MAX_LAMA_PIXELS:
        req.hd_strategy = HDStrategy.RESIZE
        req.hd_strategy_resize_limit = int(
            long_edge * math.sqrt(MAX_LAMA_PIXELS / (height * width))
        )


def _preload() -> None:
    with _lock:
        _model(PRELOAD)


def _model(kind: str) -> Any:
    """Return the model for `kind`, unloading the other one first."""
    if kind not in _loaded:
        _loaded.clear()
        _release_memory()
        if kind == "lama":
            _loaded[kind] = ModelManager(name="lama", device=DEVICE)
        else:
            _loaded[kind] = RemoveBG(REMOVE_BG_MODEL, DEVICE)
    return _loaded[kind]


def _release_memory() -> None:
    gc.collect()
    with suppress(OSError):
        ctypes.CDLL("libc.so.6").malloc_trim(0)
