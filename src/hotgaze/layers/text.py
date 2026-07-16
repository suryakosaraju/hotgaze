"""Text region layer — MSER-based text detection heuristic.

Uses OpenCV's MSER (Maximally Stable Extremal Regions) to find text-like
regions, then boosts attention over them.  No pretrained model, no
download — fully offline, base OpenCV only.  EAST is rejected
(unofficial re-hosted weights, murky provenance).
"""

from __future__ import annotations

import cv2
import numpy as np

from .._imageops import gaussian_blur
from .base import SignalLayer

# MSER parameters
_MSER_DELTA = 2
_MSER_MIN_AREA = 10
_MSER_MAX_AREA = 5000

# Text-like filtering
_MIN_ASPECT_RATIO = 0.1
_MAX_ASPECT_RATIO = 10.0
_MIN_WIDTH = 5
_MIN_HEIGHT = 5
_MERGE_DISTANCE = 15


class Text(SignalLayer):
    """Attention layer that boosts regions containing text.

    Uses MSER to find stable text-like regions, filters them by aspect
    ratio and size, merges nearby boxes, and creates an attention boost
    proportional to region area — weighted toward the top of the image
    (where headings typically live).
    """

    def compute(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # MSER detection — OpenCV 5.x uses positional args
        mser = cv2.MSER_create(_MSER_DELTA, _MSER_MIN_AREA, _MSER_MAX_AREA)  # type: ignore[attr-defined]
        regions, _ = mser.detectRegions(gray)

        # Convert to bounding boxes and filter
        boxes: list[tuple[int, int, int, int]] = []
        for region in regions:
            rx, ry, rw, rh = cv2.boundingRect(region)
            if rw < _MIN_WIDTH or rh < _MIN_HEIGHT:
                continue
            ar = rw / max(rh, 1)
            if ar < _MIN_ASPECT_RATIO or ar > _MAX_ASPECT_RATIO:
                continue
            boxes.append((rx, ry, rw, rh))

        if not boxes:
            return np.zeros((h, w), dtype=np.float32)

        # Merge nearby boxes (simple greedy merge)
        merged = _merge_boxes(boxes, _MERGE_DISTANCE)

        # Build attention map
        attn = np.zeros((h, w), dtype=np.float32)
        for mx, my, mw, mh in merged:
            # Boost proportional to area, weighted toward top of image
            area_ratio = (mw * mh) / (w * h)
            vertical_weight = 1.0 - (my / h) * 0.7  # higher = more boost
            boost = area_ratio * vertical_weight * 10.0

            y0, y1 = max(0, my), min(h, my + mh)
            x0, x1 = max(0, mx), min(w, mx + mw)
            attn[y0:y1, x0:x1] += boost

        # Blur to spread text attention into surrounding regions
        sigma = max(w, h) / 60.0
        attn = gaussian_blur(attn, sigma)

        # Normalize
        mn, mx = attn.min(), attn.max()
        if mx - mn > 1e-10:
            attn = (attn - mn) / (mx - mn)

        return attn.astype(np.float32)


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]], dist: int
) -> list[tuple[int, int, int, int]]:
    """Greedy merge of nearby bounding boxes."""
    if len(boxes) <= 1:
        return boxes

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: list[tuple[int, int, int, int]] = []

    current: list[int] = list(boxes[0])
    for bx, by, bw, bh in boxes[1:]:
        c_right = current[0] + current[2]
        c_bottom = current[1] + current[3]
        if (
            abs(current[0] - bx) < dist
            and abs(current[1] - by) < dist
            or abs(c_right - bx) < dist
            and abs(c_bottom - by) < dist
        ):
            new_x = min(current[0], bx)
            new_y = min(current[1], by)
            new_r = max(c_right, bx + bw)
            new_b = max(c_bottom, by + bh)
            current = [new_x, new_y, new_r - new_x, new_b - new_y]
        else:
            merged.append((current[0], current[1], current[2], current[3]))
            current = [bx, by, bw, bh]

    merged.append((current[0], current[1], current[2], current[3]))
    return merged
