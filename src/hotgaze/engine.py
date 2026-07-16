"""Attention prediction engine — weighted layer blending.

The engine is a pipeline of independent signal layers blended by weights.
It owns image loading, resolution normalization, layer execution, blending,
and final heatmap generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ._imageops import gaussian_blur as _gaussian_blur
from .attention_map import AttentionMap
from .config import EngineConfig

if TYPE_CHECKING:
    from .layers.base import SignalLayer


def _load_image(path: str) -> Image.Image:
    """Load an image from a file path, converting to RGB."""
    img = Image.open(path)
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    if img.mode == "LA":
        background = Image.new("L", img.size, 255)
        background.paste(img, mask=img.split()[1])
        return background.convert("RGB")
    if img.mode in ("L", "P"):
        return img.convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


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

    # Add optional extra layers (faces, text) if requested
    layers = dict(layers)  # shallow copy
    for lname in config.extra_layers:
        if lname == "faces" and "faces" not in layers:
            from .layers.faces import Faces

            layers["faces"] = Faces()
        elif lname == "text" and "text" not in layers:
            from .layers.text import Text

            layers["text"] = Text()

    # Renormalize weights if extra layers enabled
    w = config.weights
    if config.extra_layers:
        base_keys = ["saliency", "contrast", "center_bias", "gaze_flow"]
        base_weight = sum(getattr(w, k, 0.0) for k in base_keys)
        extra_weight = sum(
            0.15 if getattr(w, ln, 0.0) == 0 and ln in config.extra_layers else getattr(w, ln, 0.0)
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
                    setattr(w, ln, 0.15)

    # Run each enabled layer
    layer_maps: list[np.ndarray] = []
    total_weight = 0.0
    w = config.weights

    for name, layer in layers.items():
        weight = getattr(w, name, 0.0)
        if weight <= 0:
            continue
        try:
            layer_map = layer.compute(img_array)
        except Exception:
            # Layer failed — skip it silently
            continue
        layer_maps.append(layer_map * weight)
        total_weight += weight

    if total_weight == 0:
        # All layers disabled or failed — return uniform map
        blended: np.ndarray = np.full((work_h, work_w), 0.0, dtype=np.float32)
    else:
        blended = sum(layer_maps) / total_weight  # type: ignore[assignment]

    # Smooth
    if config.smooth_sigma > 0:
        blended = _gaussian_blur(blended, config.smooth_sigma)

    # Renormalize to [0, 1]
    mn, mx = blended.min(), blended.max()
    if mx - mn > 1e-10:
        blended = (blended - mn) / (mx - mn)

    # Resize back to original dimensions
    if (work_w, work_h) != (orig_w, orig_h):
        hm_img = Image.fromarray((blended * 255).astype(np.uint8))
        hm_img = hm_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        blended = np.array(hm_img, dtype=np.float32) / 255.0

    config_dict = config.model_dump()
    return AttentionMap(blended, (orig_w, orig_h), config_dict)


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
