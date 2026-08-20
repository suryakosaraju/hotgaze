"""GitHub Actions runner for deterministic HotGaze screenshot comparisons."""

from __future__ import annotations

import json
import math
import os
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .attention_map import AttentionMap
from .cli import _validate_image_format
from .config import EngineConfig
from .engine import _load_image, run_engine
from .scoring import (
    compare_attention_maps,
    score_regions,
    scores_to_json,
    validate_against_schema,
)

_DEFAULT_OUTPUT_DIRECTORY = "hotgaze-results"


@dataclass(frozen=True)
class ActionInputs:
    """Validated inputs supplied by the composite GitHub Action."""

    baseline: Path
    candidate: Path
    regions: tuple[str, ...]
    backend: str
    failure_threshold: float | None
    output_directory: Path
    alpha: float
    colormap: str

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> ActionInputs:
        """Parse and validate Action inputs from a mapping."""
        values = os.environ if environment is None else environment
        baseline = _required_path(values, "HOTGAZE_ACTION_BASELINE", "baseline")
        candidate = _required_path(values, "HOTGAZE_ACTION_CANDIDATE", "candidate")
        regions = _parse_regions(values.get("HOTGAZE_ACTION_REGIONS", ""))
        backend = values.get("HOTGAZE_ACTION_BACKEND", "fast") or "fast"
        if backend not in {"fast", "deep"}:
            raise ValueError("backend must be exactly 'fast' or 'deep'")

        threshold = _parse_optional_non_negative_float(
            values.get("HOTGAZE_ACTION_FAILURE_THRESHOLD", ""), "failure-threshold"
        )
        if threshold is not None and not regions:
            raise ValueError("failure-threshold requires at least one named region")

        output_raw = values.get("HOTGAZE_ACTION_OUTPUT_DIRECTORY", _DEFAULT_OUTPUT_DIRECTORY)
        output_directory = Path(output_raw.strip() or _DEFAULT_OUTPUT_DIRECTORY)
        alpha = _parse_bounded_float(values.get("HOTGAZE_ACTION_ALPHA", "0.6"), "alpha", 0, 1)
        colormap = values.get("HOTGAZE_ACTION_COLORMAP", "jet").strip() or "jet"
        if colormap not in {"jet", "turbo"}:
            raise ValueError("colormap must be 'jet' or 'turbo'")

        for path in (baseline, candidate):
            if not path.is_file():
                raise ValueError(f"image does not exist or is not a file: {path}")
            _validate_image_format(str(path))
        if output_directory.exists() and not output_directory.is_dir():
            raise ValueError(f"output-directory exists and is not a directory: {output_directory}")

        return cls(
            baseline=baseline,
            candidate=candidate,
            regions=regions,
            backend=backend,
            failure_threshold=threshold,
            output_directory=output_directory,
            alpha=alpha,
            colormap=colormap,
        )

    def engine_config(self) -> EngineConfig:
        """Build the existing HotGaze backend configuration without changing defaults."""
        if self.backend == "deep":
            return EngineConfig.deep_default()
        return EngineConfig.fast_default()


@dataclass(frozen=True)
class ActionResult:
    """Files and threshold result produced by one Action invocation."""

    baseline_score: Path
    candidate_score: Path
    compare_json: Path
    baseline_overlay: Path
    candidate_overlay: Path
    summary: str
    violations: tuple[dict[str, object], ...]

    @property
    def threshold_exceeded(self) -> bool:
        """Whether at least one configured region exceeded the loss threshold."""
        return bool(self.violations)


def _required_path(values: Mapping[str, str], key: str, label: str) -> Path:
    raw = values.get(key, "").strip()
    if not raw:
        raise ValueError(f"missing required input: {label}")
    return Path(raw)


def _parse_regions(raw: str) -> tuple[str, ...]:
    """Parse newline-delimited region definitions without splitting coordinates."""
    return tuple(line.strip() for line in raw.splitlines() if line.strip())


def _parse_optional_non_negative_float(raw: str, label: str) -> float | None:
    value = raw.strip()
    if not value:
        return None
    return _parse_bounded_float(value, label, 0, None)


def _parse_bounded_float(
    raw: str, label: str, minimum: float | None, maximum: float | None
) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum:g}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be at most {maximum:g}")
    return value


def evaluate_threshold(
    comparison: Mapping[str, object], threshold: float | None
) -> tuple[dict[str, object], ...]:
    """Return regions whose candidate attention loss is greater than the threshold."""
    if threshold is None:
        return ()
    raw_deltas = comparison.get("per_region_deltas", [])
    if not isinstance(raw_deltas, list):
        raise ValueError("compare payload has invalid per_region_deltas")
    violations: list[dict[str, object]] = []
    for entry in raw_deltas:
        if not isinstance(entry, dict) or not isinstance(entry.get("delta"), (int, float)):
            raise ValueError("compare payload contains an invalid region delta")
        if float(entry["delta"]) < -threshold:
            violations.append(entry)
    return tuple(violations)


def run_action(inputs: ActionInputs) -> ActionResult:
    """Run one baseline/candidate comparison and write all Action artifacts."""
    inputs.output_directory.mkdir(parents=True, exist_ok=True)
    config = inputs.engine_config()
    baseline_map = run_engine(str(inputs.baseline), config=config)
    candidate_map = run_engine(str(inputs.candidate), config=config)

    baseline_regions, baseline_focal = score_regions(baseline_map, list(inputs.regions))
    candidate_regions, candidate_focal = score_regions(candidate_map, list(inputs.regions))
    compare_regions, _, comparison = compare_attention_maps(
        baseline_map, candidate_map, list(inputs.regions)
    )

    baseline_json = scores_to_json(
        "score",
        str(inputs.baseline),
        baseline_map.original_size,
        baseline_map.working_size,
        config.model_dump(),
        baseline_regions,
        baseline_focal,
    )
    candidate_json = scores_to_json(
        "score",
        str(inputs.candidate),
        candidate_map.original_size,
        candidate_map.working_size,
        config.model_dump(),
        candidate_regions,
        candidate_focal,
    )
    compare_json = scores_to_json(
        "compare",
        str(inputs.baseline),
        baseline_map.original_size,
        baseline_map.working_size,
        config.model_dump(),
        compare_regions,
        [],
        compare=comparison,
        image_b_path=str(inputs.candidate),
        img_b_size=candidate_map.original_size,
        work_b_size=candidate_map.working_size,
    )

    for label, payload in (
        ("baseline score", baseline_json),
        ("candidate score", candidate_json),
        ("comparison", compare_json),
    ):
        errors = validate_against_schema(json.loads(payload))
        if errors:
            details = "; ".join(errors)
            raise ValueError(f"generated {label} JSON failed schema validation: {details}")

    baseline_score_path = inputs.output_directory / "baseline-score.json"
    candidate_score_path = inputs.output_directory / "candidate-score.json"
    compare_path = inputs.output_directory / "compare.json"
    baseline_overlay_path = inputs.output_directory / "baseline-overlay.png"
    candidate_overlay_path = inputs.output_directory / "candidate-overlay.png"

    baseline_score_path.write_text(baseline_json, encoding="utf-8")
    candidate_score_path.write_text(candidate_json, encoding="utf-8")
    compare_path.write_text(compare_json, encoding="utf-8")
    _write_overlay(baseline_map, inputs.baseline, baseline_overlay_path, inputs)
    _write_overlay(candidate_map, inputs.candidate, candidate_overlay_path, inputs)

    violations = evaluate_threshold(comparison, inputs.failure_threshold)
    summary = build_summary(inputs, comparison, violations)
    return ActionResult(
        baseline_score=baseline_score_path,
        candidate_score=candidate_score_path,
        compare_json=compare_path,
        baseline_overlay=baseline_overlay_path,
        candidate_overlay=candidate_overlay_path,
        summary=summary,
        violations=violations,
    )


def _write_overlay(
    attention_map: AttentionMap, source: Path, destination: Path, inputs: ActionInputs
) -> None:
    original = _load_image(str(source))
    overlay = attention_map.overlay(original, alpha=inputs.alpha, colormap=inputs.colormap)
    overlay.save(destination)


def build_summary(
    inputs: ActionInputs,
    comparison: Mapping[str, object],
    violations: tuple[dict[str, object], ...],
) -> str:
    """Build the Markdown written to the GitHub Actions job summary."""
    status = "❌ Threshold exceeded" if violations else "✅ Comparison completed"
    lines = [
        "# HotGaze attention comparison",
        "",
        f"**{status}**",
        "",
        f"- Backend: `{inputs.backend}`",
        f"- Baseline: `{_markdown_text(str(inputs.baseline))}`",
        f"- Candidate: `{_markdown_text(str(inputs.candidate))}`",
    ]
    if inputs.failure_threshold is None:
        lines.append("- Failure threshold: not configured")
    else:
        lines.append(f"- Maximum allowed region loss: `{inputs.failure_threshold:.6f}`")

    raw_deltas = comparison.get("per_region_deltas", [])
    if isinstance(raw_deltas, list) and raw_deltas:
        lines.extend(
            [
                "",
                "## Region attention share",
                "",
                "| Region | Baseline | Candidate | Delta | Result |",
                "|---|---:|---:|---:|---|",
            ]
        )
        violation_names = {str(entry["name"]) for entry in violations}
        for entry in raw_deltas:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", ""))
            result = "Fail" if name in violation_names else "Pass"
            lines.append(
                f"| {_markdown_text(name)} | {float(entry['share_a']):.6f} | "
                f"{float(entry['share_b']):.6f} | {float(entry['delta']):+.6f} | {result} |"
            )
    else:
        raw_grid = comparison.get("grid_deltas", [])
        if isinstance(raw_grid, list) and len(raw_grid) == 9:
            lines.extend(["", "## 3×3 attention-share deltas", ""])
            for row in range(3):
                cells = raw_grid[row * 3 : (row + 1) * 3]
                lines.append(" | ".join(f"`{float(value):+.6f}`" for value in cells))

    lines.extend(
        [
            "",
            "Artifacts contain canonical schema-v1 JSON and both overlay images.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_text(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _write_github_outputs(path: Path, result: ActionResult) -> None:
    values = {
        "result": "failed" if result.threshold_exceeded else "passed",
        "output-directory": str(result.compare_json.parent.resolve()),
        "baseline-score": str(result.baseline_score.resolve()),
        "candidate-score": str(result.candidate_score.resolve()),
        "compare-json": str(result.compare_json.resolve()),
        "baseline-overlay": str(result.baseline_overlay.resolve()),
        "candidate-overlay": str(result.candidate_overlay.resolve()),
    }
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            delimiter = f"hotgaze_{uuid.uuid4().hex}"
            output.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")


def main(environment: Mapping[str, str] | None = None) -> int:
    """Run from GitHub Actions environment variables and return a process code."""
    values = os.environ if environment is None else environment
    summary_path_raw = values.get("GITHUB_STEP_SUMMARY", "")
    try:
        inputs = ActionInputs.from_environment(values)
        result = run_action(inputs)
        if summary_path_raw:
            Path(summary_path_raw).write_text(result.summary, encoding="utf-8")
        output_path_raw = values.get("GITHUB_OUTPUT", "")
        if output_path_raw:
            _write_github_outputs(Path(output_path_raw), result)
        if result.threshold_exceeded:
            names = ", ".join(str(entry["name"]) for entry in result.violations)
            print(f"HotGaze threshold exceeded for: {names}", file=sys.stderr)
            return 1
        print(f"HotGaze artifacts written to {inputs.output_directory}")
        return 0
    except Exception as exc:
        message = f"HotGaze Action failed: {exc}"
        if summary_path_raw:
            Path(summary_path_raw).write_text(f"# HotGaze attention comparison\n\n❌ {message}\n")
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
