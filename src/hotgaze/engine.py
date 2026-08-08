"""Attention prediction engine — weighted layer blending.

The engine is a pipeline of independent signal layers blended by weights.
It owns image loading, resolution normalization, layer execution, blending,
and final heatmap generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

from ._imageops import gaussian_blur as _gaussian_blur
from .attention_map import AttentionMap
from .config import EngineConfig

if TYPE_CHECKING:
    from .layers.base import SignalLayer

logger = logging.getLogger(__name__)


def _load_image(path: str) -> Image.Image:
    """Load an image, applying EXIF orientation and flattening alpha over white."""
    with Image.open(path) as opened:
        img = ImageOps.exif_transpose(opened)
        img.load()

        # Palette PNGs can carry transparency in ``info`` rather than an A
        # channel. Converting them to RGBA first makes Pillow apply the tRNS
        # table before the same white-background compositing used elsewhere.
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")

        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            return background
        if img.mode == "LA":
            background = Image.new("L", img.size, 255)
            background.paste(img.getchannel("L"), mask=img.getchannel("A"))
            return background.convert("RGB")
        return img.convert("RGB")


def _working_size(w: int, h: int, long_edge: int) -> tuple[int, int]:
    """Compute working resolution: long edge ≤ `long_edge`, never upscale."""
    max_dim = max(w, h)
    if max_dim <= long_edge:
        return w, h
    scale = long_edge / max_dim
    return int(round(w * scale)), int(round(h * scale))


def run_engine(
    image_path: str,
    config: EngineConfig | None = None,
    layers: dict[str, SignalLayer] | None = None,
) -> AttentionMap:
    """Run the attention prediction engine on an image.

    Args:
        image_path: Path to PNG/JPEG/WebP image.
        config: Engine configuration. Defaults to fast backend defaults.
        layers: Dict of layer_name -> SignalLayer. If None, uses default
            fast layers.

    Returns:
        AttentionMap with the blended heatmap.
    """
    if config is None:
        config = EngineConfig.fast_default()

    # Load and prepare image
    img = _load_image(image_path)
    orig_w, orig_h = img.size
    work_w, work_h = _working_size(orig_w, orig_h, config.working_long_edge)

    # Resize to working resolution (never upscale — _working_size guarantees this)
    if (work_w, work_h) != (orig_w, orig_h):
        img_work = img.resize((work_w, work_h), Image.Resampling.LANCZOS)
    else:
        img_work = img

    img_array = np.array(img_work)  # (H, W, 3) uint8

    # Build default layers if not provided
    if layers is None:
        layers = _default_deep_layers() if config.backend == "deep" else _default_fast_layers()

    # Add optional extra layers if requested
    layers = dict(layers)  # shallow copy
    for lname in config.extra_layers:
        if lname == "faces" and "faces" not in layers:
            from .layers.faces import Faces

            layers["faces"] = Faces()

    # Renormalize weights if extra layers enabled
    w = config.weights
    if config.extra_layers:
        base_keys = ["saliency", "contrast", "center_bias", "gaze_flow"]
        base_weight = sum(getattr(w, k, 0.0) for k in base_keys)
        extra_weight = sum(
            0.50 if getattr(w, ln, 0.0) == 0 and ln in config.extra_layers else getattr(w, ln, 0.0)
            for ln in config.extra_layers
        )
        if base_weight > 0 and extra_weight > 0:
            # Renormalize: scale base weights so total = 1.0
            scale = (1.0 - extra_weight) / base_weight
            w = w.model_copy()
            for k in base_keys:
                current = getattr(w, k, 0.0)
                if current > 0:
                    setattr(w, k, round(current * scale, 6))
            for ln in config.extra_layers:
                if getattr(w, ln, 0.0) == 0:
                    setattr(w, ln, 0.50)

    # Run each enabled layer
    layer_maps: list[np.ndarray] = []
    total_weight = 0.0
    failed_layers: list[str] = []

    for name, layer in layers.items():
        weight = getattr(w, name, 0.0)
        if weight <= 0:
            continue
        try:
            layer_map = layer.compute(img_array)
            _validate_layer_map(layer_map, (work_h, work_w), name)
        except Exception as exc:
            failed_layers.append(name)
            logger.warning(
                "Layer %r failed and was skipped (%s): %s",
                name,
                type(exc).__name__,
                exc,
            )
            continue
        layer_maps.append(layer_map * weight)
        total_weight += weight

    if total_weight == 0:
        if failed_layers:
            raise FloatingPointError(
                f"Cannot produce a heatmap: all enabled layers failed ({', '.join(failed_layers)})"
            )
        # All layers are disabled — return a deliberate uniform map.
        blended: np.ndarray = np.full((work_h, work_w), 0.0, dtype=np.float32)
    else:
        blended = sum(layer_maps) / total_weight  # type: ignore[assignment]

    _ensure_finite_heatmap(blended, "after blending")

    # Smooth
    if config.smooth_sigma > 0:
        blended = _gaussian_blur(blended, config.smooth_sigma)

    _ensure_finite_heatmap(blended, "before normalization")

    # Renormalize to [0, 1]
    mn, mx = blended.min(), blended.max()
    if mx - mn > 1e-10:
        blended = (blended - mn) / (mx - mn)

    # Resize back to original dimensions
    if (work_w, work_h) != (orig_w, orig_h):
        _ensure_finite_heatmap(blended, "before uint8 conversion")
        hm_img = Image.fromarray((blended * 255).astype(np.uint8))
        hm_img = hm_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        blended = np.array(hm_img, dtype=np.float32) / 255.0

    # Final public heatmap contract: finite float32 values exactly in [0, 1].
    _ensure_finite_heatmap(blended, "after resizing")
    blended = np.clip(blended, 0.0, 1.0).astype(np.float32, copy=False)

    config_dict = config.model_dump()
    return AttentionMap(
        blended,
        (orig_w, orig_h),
        config_dict,
        working_size=(work_w, work_h),
    )


def _ensure_finite_heatmap(heatmap: np.ndarray, stage: str) -> None:
    """Raise before non-finite values can be normalized, cast, or returned."""
    if not np.isfinite(heatmap).all():
        raise FloatingPointError(f"Non-finite heatmap {stage}")


def _validate_layer_map(layer_map: object, expected_shape: tuple[int, int], name: str) -> None:
    """Validate a layer's output before it enters the weighted blend."""
    if not isinstance(layer_map, np.ndarray):
        raise TypeError(f"layer {name!r} returned {type(layer_map).__name__}, expected ndarray")
    if layer_map.shape != expected_shape:
        raise ValueError(
            f"layer {name!r} returned shape {layer_map.shape}, expected {expected_shape}"
        )
    if layer_map.dtype != np.float32:
        raise TypeError(f"layer {name!r} returned dtype {layer_map.dtype}, expected float32")
    if not np.isfinite(layer_map).all():
        raise ValueError(f"layer {name!r} returned non-finite values")
    if np.any(layer_map < 0.0) or np.any(layer_map > 1.0):
        raise ValueError(f"layer {name!r} returned values outside the [0, 1] range")


def _default_fast_layers() -> dict[str, SignalLayer]:
    """Build the default set of fast (heuristic) layers."""
    from .layers.center_bias import CenterBias
    from .layers.contrast import Contrast
    from .layers.gaze_flow import GazeFlow
    from .layers.saliency_fast import SaliencyFast

    return {
        "saliency": SaliencyFast(),
        "contrast": Contrast(),
        "center_bias": CenterBias(),
        "gaze_flow": GazeFlow(),
    }


def _default_deep_layers() -> dict[str, SignalLayer]:
    """Build the default set of deep-backend layers.

    Uses the pretrained UNISAL model with center_bias + gaze_flow priors.
    Lazy-imports torch — raises ImportError with actionable message if missing.
    """
    try:
        from .layers.saliency_deep import SaliencyDeep, load_unisal
    except ImportError:
        raise ImportError(
            "The deep backend requires PyTorch. Install with: pip install hotgaze[deep]"
        ) from None

    from .layers.center_bias import CenterBias
    from .layers.gaze_flow import GazeFlow

    try:
        model = load_unisal()
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load deep backend model: {e}") from e
    return {
        "saliency": SaliencyDeep(model),
        "center_bias": CenterBias(),
        "gaze_flow": GazeFlow(),
    }
