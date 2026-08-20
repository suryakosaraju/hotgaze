"""Region scoring, focal point extraction, and canonical JSON serialization.

T2.1 — the numeric differentiator for HotGaze.
"""

from __future__ import annotations

import json
import math
import re
from importlib.resources import files
from typing import Any

import cv2
import numpy as np

# ── Region parsing ───────────────────────────────────────────────────────────

_REGION_RE = re.compile(r"^(?P<name>[^:]+):(?P<coords>.*)$")
_PIXEL_INTEGER_RE = re.compile(r"^-?\d+$")


class RegionParseError(ValueError):
    """Raised when a region string cannot be parsed."""


def parse_region(region_str: str, img_w: int, img_h: int) -> tuple[str, int, int, int, int]:
    """Parse a region string into (name, x, y, w, h) in pixel coordinates.

    Formats:
        ``name:x,y,w,h``     — pixel coordinates
        ``name:0.1,0.2,0.3,0.08f`` — fractional (trailing ``f`` marks entire tuple)

    Fractional and pixel values mixed in one region is invalid. Boxes are clamped
    to image bounds. A fully out-of-bounds box is an error.
    """
    m = _REGION_RE.match(region_str.strip())
    if not m or not m.group("name").strip():
        raise RegionParseError(
            f"Invalid region format: {region_str!r}. "
            f"Expected name:x,y,w,h or name:0.1,0.2,0.3,0.08f"
        )

    name = m.group("name").strip()
    coords = m.group("coords").strip()
    is_frac = coords.endswith("f")
    if is_frac:
        coords = coords[:-1]

    raw_values = [value.strip() for value in coords.split(",")]
    labels = ("x", "y", "w", "h")
    if len(raw_values) != 4 or any(not value for value in raw_values):
        raise RegionParseError(
            f"Invalid region format: {region_str!r}. "
            f"Expected name:x,y,w,h or name:0.1,0.2,0.3,0.08f"
        )

    if is_frac:
        fractional_values: list[float] = []
        for raw, label in zip(raw_values, labels, strict=True):
            try:
                value = float(raw)
            except ValueError as exc:
                raise RegionParseError(
                    f"Region {region_str!r}: {label}={raw!r} is not a valid number. "
                    "Fractional coordinates must be finite numbers between 0 and 1."
                ) from exc
            if not math.isfinite(value):
                raise RegionParseError(
                    f"Region {region_str!r}: {label}={raw!r} must be finite. "
                    "Use a number between 0 and 1, followed by the f suffix."
                )
            if not 0.0 <= value <= 1.0:
                raise RegionParseError(
                    f"Region {region_str!r}: {label}={raw!r} is outside the allowed 0–1 range. "
                    "Fractional coordinates must be between 0 and 1."
                )
            fractional_values.append(value)
        x, y, w, h = (
            int(round(fractional_values[0] * img_w)),
            int(round(fractional_values[1] * img_h)),
            int(round(fractional_values[2] * img_w)),
            int(round(fractional_values[3] * img_h)),
        )
    else:
        # Pixel: reject values with decimal points (ambiguous without trailing f)
        pixel_values: list[int] = []
        for raw, label in zip(raw_values, labels, strict=True):
            try:
                numeric_value = float(raw)
            except ValueError as exc:
                raise RegionParseError(
                    f"Region {region_str!r}: {label}={raw!r} is not a valid number. "
                    "Use whole-pixel values or a finite 0–1 value with the f suffix."
                ) from exc
            if not math.isfinite(numeric_value):
                raise RegionParseError(
                    f"Region {region_str!r}: {label}={raw!r} must be finite. "
                    "Use whole-pixel values or a finite 0–1 value with the f suffix."
                )
            if not _PIXEL_INTEGER_RE.fullmatch(raw):
                raise RegionParseError(
                    f"Region {region_str!r}: {label}={raw!r} looks fractional. "
                    "Add the f suffix for fractional coords (name:0.1,0.2,0.3,0.08f) "
                    "or use whole-pixel values (name:10,20,100,50)."
                )
            pixel_values.append(int(raw))
        x, y, w, h = pixel_values

    return _clamp_region(name, x, y, w, h, img_w, img_h)


def _clamp_region(
    name: str, x: int, y: int, w: int, h: int, img_w: int, img_h: int
) -> tuple[str, int, int, int, int]:
    """Clamp region to image bounds. Raise if fully out-of-bounds."""
    # Clamp top-left
    x1 = max(0, x)
    y1 = max(0, y)
    # Clamp bottom-right
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    cw = x2 - x1
    ch = y2 - y1
    if cw <= 0 or ch <= 0:
        raise RegionParseError(
            f"Region {name!r} is fully out of bounds: ({x},{y},{w},{h}) for image {img_w}×{img_h}"
        )
    return name, x1, y1, cw, ch


# ── Region scoring ───────────────────────────────────────────────────────────


def score_regions(
    attention_map: Any, regions: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score named regions on an attention map.

    Args:
        attention_map: An ``AttentionMap`` instance.
        regions: List of region strings (e.g. ``"cta:100,200,300,80"``).

    Returns:
        Tuple of (scored_regions, focal_points). Each scored region dict has
        keys: name, box (x/y/w/h), share (0-1), peak_value (0-1), rank.
    """
    ow, oh = attention_map.original_size
    hm = attention_map.heatmap

    # Parse and score each region
    scored: list[dict[str, Any]] = []
    for rstr in regions:
        name, x, y, w, h = parse_region(rstr, ow, oh)
        region_hm = hm[y : y + h, x : x + w]
        share = float(region_hm.sum()) / max(float(hm.sum()), 1e-12)
        peak = float(region_hm.max())
        scored.append(
            {
                "name": name,
                "box": {"x": x, "y": y, "w": w, "h": h},
                "share": share,
                "peak_value": peak,
                "rank": 0,
            }
        )

    # Rank by share (descending)
    scored.sort(key=lambda r: r["share"], reverse=True)
    for i, r in enumerate(scored):
        r["rank"] = i + 1

    return scored, find_focal_points(attention_map)


# ── Focal points ─────────────────────────────────────────────────────────────


def find_focal_points(attention_map: Any, n: int = 5) -> list[dict[str, Any]]:
    """Find top-N focal points (local maxima) on the attention map.

    Uses ``cv2.dilate``-based max-filter comparison. No scikit-image or scipy.

    Args:
        attention_map: An ``AttentionMap`` instance.
        n: Maximum number of focal points to return.

    Returns:
        List of dicts with keys: x, y (original-image coords), value, rank.
    """
    ow, oh = attention_map.original_size
    hm = attention_map.heatmap

    diag = math.sqrt(ow**2 + oh**2)
    min_dist_px = max(1, int(0.05 * diag))
    threshold = 0.10 * float(hm.max()) if hm.max() > 0 else 0.0

    # Dilate-based max filter: a pixel is a local maximum if it equals the
    # dilated value and exceeds threshold.
    kernel_size = 2 * min_dist_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(hm, kernel)

    # Find local maxima
    is_max = (hm == dilated) & (hm >= threshold)
    ys, xs = np.where(is_max)
    values = hm[ys, xs]

    # Sort by value descending
    order = np.argsort(values)[::-1]
    xs = xs[order]
    ys = ys[order]
    values = values[order]

    # Greedy non-maximum suppression: keep strongest, suppress neighbors within min_dist
    kept: list[dict[str, Any]] = []
    suppressed = np.zeros(len(xs), dtype=bool)

    for i in range(len(xs)):
        if suppressed[i]:
            continue
        kept.append(
            {
                "x": int(xs[i]),
                "y": int(ys[i]),
                "value": float(values[i]),
                "rank": len(kept),
            }
        )
        if len(kept) >= n:
            break
        # Suppress nearby points
        dx = xs[i + 1 :] - xs[i]
        dy = ys[i + 1 :] - ys[i]
        dist = np.sqrt(dx**2 + dy**2)
        suppressed[i + 1 :] |= dist < min_dist_px

    for i, pt in enumerate(kept):
        pt["rank"] = i + 1

    return kept


# ── Compare helpers ──────────────────────────────────────────────────────────


def compute_grid_deltas(hm_a: np.ndarray, hm_b: np.ndarray) -> list[float]:
    """Compute 3×3 grid of attention-share deltas (B - A).

    Each cell's share is its fraction of total map mass. Returns 9 floats
    in row-major order (top-left to bottom-right). Positive = B has more
    attention in that cell.
    """
    if hm_a.shape != hm_b.shape:
        raise ValueError(
            f"Cannot compare grid deltas for different heatmap shapes: {hm_a.shape} vs {hm_b.shape}"
        )

    h, w = hm_a.shape
    dw = w // 3
    dh = h // 3
    total_a = max(float(hm_a.sum()), 1e-12)
    total_b = max(float(hm_b.sum()), 1e-12)

    deltas: list[float] = []
    for row in range(3):
        y0 = row * dh
        y1 = (row + 1) * dh if row < 2 else h
        for col in range(3):
            x0 = col * dw
            x1 = (col + 1) * dw if col < 2 else w
            share_a = float(hm_a[y0:y1, x0:x1].sum()) / total_a
            share_b = float(hm_b[y0:y1, x0:x1].sum()) / total_b
            deltas.append(round(share_b - share_a, 6))
    return deltas


def compute_focal_movement(
    focal_a: list[dict[str, Any]], focal_b: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compute focal-point movement between two sets.

    Pairs focal points by rank and computes Euclidean distance.
    """
    import math

    movement: list[dict[str, Any]] = []
    max_n = max(len(focal_a), len(focal_b))
    for i in range(min(max_n, 5)):
        a = focal_a[i] if i < len(focal_a) else None
        b = focal_b[i] if i < len(focal_b) else None
        entry: dict[str, Any] = {"rank": i + 1}
        if a:
            entry["x_a"] = a["x"]
            entry["y_a"] = a["y"]
        else:
            entry["x_a"] = -1
            entry["y_a"] = -1
        if b:
            entry["x_b"] = b["x"]
            entry["y_b"] = b["y"]
        else:
            entry["x_b"] = -1
            entry["y_b"] = -1
        if a and b:
            entry["distance"] = round(math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2), 6)
        else:
            entry["distance"] = -1.0
        movement.append(entry)
    return movement


def compare_attention_maps(
    attention_map_a: Any,
    attention_map_b: Any,
    regions: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compare two attention maps using the public compare-mode contract.

    Returns baseline scores, candidate scores, and the compare payload used by
    canonical schema-v1 JSON. Region comparisons are matched by unique name;
    without regions, both images must share a pixel coordinate system.
    """
    size_a = attention_map_a.original_size
    size_b = attention_map_b.original_size

    if not regions and size_a != size_b:
        wa, ha = size_a
        wb, hb = size_b
        raise ValueError(
            f"Images have different sizes ({wa}×{ha} vs {wb}×{hb}) and no regions were "
            "supplied. Use a fractional --region for cross-size comparison, or resize "
            "the images to the same dimensions."
        )

    if size_a != size_b:
        wa, ha = size_a
        wb, hb = size_b
        for region in regions:
            if not region.rstrip().endswith("f"):
                raise RegionParseError(
                    f"Images have different sizes ({wa}×{ha} vs {wb}×{hb}). "
                    f"Region {region!r} uses pixel coordinates. "
                    "Use fractional coords with trailing f "
                    "(e.g. name:0.1,0.2,0.3,0.08f) for cross-size comparisons."
                )

    if regions:
        seen_names: set[str] = set()
        for region in regions:
            name = parse_region(region, *size_a)[0]
            if name in seen_names:
                raise RegionParseError(
                    f"Duplicate region name {name!r} in compare input. "
                    "Use a unique name for each --region so per-region deltas are unambiguous."
                )
            seen_names.add(name)

        scored_a, _ = score_regions(attention_map_a, regions)
        scored_b, _ = score_regions(attention_map_b, regions)
        scores_b_by_name = {entry["name"]: entry for entry in scored_b}
        deltas = [
            {
                "name": entry_a["name"],
                "share_a": entry_a["share"],
                "share_b": scores_b_by_name[entry_a["name"]]["share"],
                "delta": round(scores_b_by_name[entry_a["name"]]["share"] - entry_a["share"], 6),
            }
            for entry_a in scored_a
        ]
        compare: dict[str, Any] = {
            "per_region_deltas": deltas,
            "grid_deltas": [],
            "focal_point_movement": [],
        }
        return scored_a, scored_b, compare

    _, focal_a = score_regions(attention_map_a, [])
    _, focal_b = score_regions(attention_map_b, [])
    compare = {
        "per_region_deltas": [],
        "grid_deltas": compute_grid_deltas(attention_map_a.heatmap, attention_map_b.heatmap),
        "focal_point_movement": compute_focal_movement(focal_a, focal_b),
    }
    return [], [], compare


# ── Canonical JSON ───────────────────────────────────────────────────────────


def _format_float(value: float) -> str:
    """Format a float to fixed 6 decimal places.

    Raises ValueError on NaN or Inf — these are invalid JSON and would
    break downstream parsers.
    """
    if not math.isfinite(value):
        raise ValueError(f"Non-finite float in canonical JSON: {value!r}")
    return f"{value:.6f}"


class _CanonicalEncoder(json.JSONEncoder):
    """JSON encoder that sorts keys and formats floats to 6 dp."""

    def encode(self, o: Any) -> str:
        return self._encode(o)

    def _encode(self, o: Any) -> str:
        if isinstance(o, dict):
            items = sorted(o.items(), key=lambda kv: kv[0])
            parts = [f"{json.dumps(k)}: {self._encode(v)}" for k, v in items]
            return "{" + ", ".join(parts) + "}"
        if isinstance(o, list):
            parts = [self._encode(v) for v in o]
            return "[" + ", ".join(parts) + "]"
        if isinstance(o, bool):
            return "true" if o else "false"
        if isinstance(o, int) and not isinstance(o, bool):
            return str(o)
        if isinstance(o, float):
            return _format_float(o)
        if o is None:
            return "null"
        return json.dumps(o)


def scores_to_json(
    mode: str,
    image_path: str,
    img_size: tuple[int, int],
    work_size: tuple[int, int],
    config: dict[str, Any],
    regions: list[dict[str, Any]],
    focal_points: list[dict[str, Any]],
    compare: dict[str, Any] | None = None,
    image_b_path: str | None = None,
    img_b_size: tuple[int, int] | None = None,
    work_b_size: tuple[int, int] | None = None,
) -> str:
    """Serialize a score or compare result to canonical JSON.

    Returns a string with sorted keys and floats rounded to 6 dp.
    Two runs with identical inputs produce byte-identical output.
    """
    output: dict[str, Any] = {
        "schema": 1,
        "mode": mode,
        "image": {
            "path": image_path,
            "size": {"width": img_size[0], "height": img_size[1]},
            "working_size": {"width": work_size[0], "height": work_size[1]},
        },
        "config": config,
        "regions": regions,
        "focal_points": focal_points,
    }

    if mode == "compare":
        output["image_b"] = {
            "path": image_b_path or "",
            "size": {"width": (img_b_size or (0, 0))[0], "height": (img_b_size or (0, 0))[1]},
            "working_size": {
                "width": (work_b_size or (0, 0))[0],
                "height": (work_b_size or (0, 0))[1],
            },
        }
        output["compare"] = compare or {}

    return _CanonicalEncoder().encode(output) + "\n"


# ── Schema validation ────────────────────────────────────────────────────────

_SCHEMA_CACHE: dict[str, Any] | None = None


def _load_schema() -> dict[str, Any]:
    """Load the packaged score schema via importlib.resources."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        text = files("hotgaze.schemas").joinpath("score.schema.json").read_text()
        _SCHEMA_CACHE = json.loads(text)
    return _SCHEMA_CACHE


def validate_against_schema(data: dict[str, Any]) -> list[str]:
    """Validate a score output dict against the packaged schema.

    Returns a list of error messages (empty = valid). This is a minimal
    structural validator — no external jsonschema dependency needed.
    """
    schema = _load_schema()
    errors: list[str] = []

    required = schema.get("required", [])
    for key in required:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    props = schema.get("properties", {})
    for key, spec in props.items():
        if key not in data:
            continue
        value = data[key]
        errors.extend(_validate_value(value, spec, key))

    # Schema version check
    if data.get("schema") != schema.get("properties", {}).get("schema", {}).get("const"):
        errors.append(f"Schema version mismatch: got {data.get('schema')}, expected 1")

    return errors


def _validate_value(value: Any, spec: dict[str, Any], path: str) -> list[str]:
    """Recursively validate a value against a schema spec."""
    errors: list[str] = []

    if "anyOf" in spec:
        branch_errors = [_validate_value(value, branch, path) for branch in spec["anyOf"]]
        if not any(not branch for branch in branch_errors):
            errors.append(f"{path}: does not match any allowed schema")
        return errors

    typ = spec.get("type")

    if typ == "object":
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object, got {type(value).__name__}")
            return errors
        for key, prop_spec in spec.get("properties", {}).items():
            if key not in value:
                if key in spec.get("required", []):
                    errors.append(f"{path}.{key}: missing required key")
                continue
            errors.extend(_validate_value(value[key], prop_spec, f"{path}.{key}"))

    elif typ == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array, got {type(value).__name__}")
            return errors
        if "minItems" in spec and len(value) < spec["minItems"]:
            errors.append(f"{path}: expected at least {spec['minItems']} items")
        if "maxItems" in spec and len(value) > spec["maxItems"]:
            errors.append(f"{path}: expected at most {spec['maxItems']} items")
        items_spec = spec.get("items", {})
        for i, item in enumerate(value):
            errors.extend(_validate_value(item, items_spec, f"{path}[{i}]"))

    elif typ == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string, got {type(value).__name__}")
    elif typ == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}: expected integer, got {type(value).__name__}")
    elif typ == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path}: expected number, got {type(value).__name__}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"{path}: expected at least {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            errors.append(f"{path}: expected at most {spec['maximum']}")

    if "enum" in spec and value not in spec["enum"]:
        errors.append(f"{path}: {value!r} not in {spec['enum']}")
    if "const" in spec and value != spec["const"]:
        errors.append(f"{path}: expected {spec['const']}, got {value!r}")

    return errors
