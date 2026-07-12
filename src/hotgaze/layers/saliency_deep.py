"""Deep saliency layer — pretrained UNISAL model wrapper.

Uses the vendored UNISAL model from ``hotgaze._unisal``. Weights are
downloaded on first use via ``hotgaze.weights`` and cached locally.
Requires ``hotgaze[deep]`` (torch).
"""

from __future__ import annotations

import numpy as np
import torch

from ..weights import download_weight
from .base import SignalLayer

# ImageNet normalization — from unisal/unisal/data.py lines 64-65
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SaliencyDeep(SignalLayer):
    """Deep saliency layer using a pretrained UNISAL model.

    The model is dependency-injected via ``model`` — pass an already-loaded
    ``torch.nn.Module``. Use ``load_unisal()`` to build the real model
    (downloads weights, constructs the vendored architecture).

    Args:
        model: A callable that takes a 5-D tensor ``[B,T,C,H,W]`` and returns
            a log-probability map of shape ``[B,T,1,H,W]``.
    """

    def __init__(self, model) -> None:  # type: ignore[no-untyped-def]
        self._model = model

    def compute(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        # Convert to tensor, ImageNet-normalize
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # (3, H, W)
        t = t / 255.0
        for c in range(3):
            t[c] = (t[c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]

        # UNISAL expects 5-D: [batch=1, time=1, channels=3, H, W]
        t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)

        with torch.no_grad():
            output = self._model(t, source="SALICON", static=True)
            # output: (1, 1, 1, H, W) — log-probability map
            prob = torch.exp(output[0, 0, 0])  # (H, W) probability map

        result = prob.cpu().numpy().astype(np.float32)

        # Normalize to [0, 1]
        mn, mx = result.min(), result.max()
        if mx - mn > 1e-10:
            result = (result - mn) / (mx - mn)

        return result


def load_unisal(device: str = "cpu") -> UNISAL:  # type: ignore[name-defined] # noqa: F821
    """Load the UNISAL model with pretrained weights.

    Downloads weights via ``hotgaze.weights``, constructs the vendored
    model architecture, loads the state dict, and sets eval mode.

    Args:
        device: Torch device string (``"cpu"`` only in v1 — determinism).

    Returns:
        A ``UNISAL`` model in eval mode.

    Raises:
        ImportError: If torch is not installed (with actionable message).
        FileNotFoundError: If weights are not yet published.
    """

    from .._unisal._model import UNISAL

    # Download weights (or use cache)
    try:
        weight_path = download_weight("unisal")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "UNISAL weights are not yet published. "
            "See PROGRESS.md Orchestrator TODOs for the publish checklist.\n"
            f"Original error: {e}"
        ) from e

    model = UNISAL()  # type: ignore[no-untyped-call]  # vendored Apache code, untyped
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Determinism per CLAUDE.md
    if device == "cpu":
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

    return model
