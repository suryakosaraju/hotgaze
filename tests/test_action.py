"""Tests for the read-only HotGaze GitHub Action runner."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from hotgaze.action import (
    ActionInputs,
    ActionResult,
    build_summary,
    evaluate_threshold,
    main,
)
from hotgaze.scoring import _load_schema


def _image(path: Path, color: tuple[int, int, int] = (30, 60, 90)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color).save(path)
    return path


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "HOTGAZE_ACTION_BASELINE": str(_image(tmp_path / "baseline.png")),
        "HOTGAZE_ACTION_CANDIDATE": str(_image(tmp_path / "candidate.png", (90, 60, 30))),
        "HOTGAZE_ACTION_OUTPUT_DIRECTORY": str(tmp_path / "results"),
    }


class TestActionInputs:
    def test_defaults_to_offline_fast_backend(self, tmp_path: Path) -> None:
        inputs = ActionInputs.from_environment(_environment(tmp_path))

        assert inputs.backend == "fast"
        assert inputs.failure_threshold is None
        assert inputs.regions == ()
        assert inputs.engine_config().backend == "fast"

    def test_parses_newline_regions_and_deep_only_when_selected(self, tmp_path: Path) -> None:
        environment = _environment(tmp_path)
        environment.update(
            {
                "HOTGAZE_ACTION_REGIONS": "headline:0.1,0.1,0.5,0.2f\n\ncta:1,2,3,4\n",
                "HOTGAZE_ACTION_BACKEND": "deep",
                "HOTGAZE_ACTION_FAILURE_THRESHOLD": "0.025",
            }
        )

        inputs = ActionInputs.from_environment(environment)

        assert inputs.regions == ("headline:0.1,0.1,0.5,0.2f", "cta:1,2,3,4")
        assert inputs.backend == "deep"
        assert inputs.engine_config().backend == "deep"
        assert inputs.failure_threshold == 0.025

    def test_paths_with_spaces_are_preserved(self, tmp_path: Path) -> None:
        directory = tmp_path / "screenshots with spaces"
        environment = {
            "HOTGAZE_ACTION_BASELINE": str(_image(directory / "baseline image.png")),
            "HOTGAZE_ACTION_CANDIDATE": str(_image(directory / "candidate image.png")),
            "HOTGAZE_ACTION_OUTPUT_DIRECTORY": str(tmp_path / "output with spaces"),
        }

        inputs = ActionInputs.from_environment(environment)

        assert inputs.baseline.name == "baseline image.png"
        assert inputs.output_directory.name == "output with spaces"

    @pytest.mark.parametrize("backend", [" deep ", "DEEP", " Fast ", "FAST"])
    def test_backend_requires_exact_lowercase_value(self, tmp_path: Path, backend: str) -> None:
        environment = _environment(tmp_path)
        environment["HOTGAZE_ACTION_BACKEND"] = backend

        with pytest.raises(ValueError, match="exactly 'fast' or 'deep'"):
            ActionInputs.from_environment(environment)

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            ({"HOTGAZE_ACTION_BACKEND": "gpu"}, "backend"),
            ({"HOTGAZE_ACTION_FAILURE_THRESHOLD": "-0.1"}, "at least 0"),
            ({"HOTGAZE_ACTION_FAILURE_THRESHOLD": "nan"}, "finite"),
            ({"HOTGAZE_ACTION_FAILURE_THRESHOLD": "0.1"}, "requires at least one"),
            ({"HOTGAZE_ACTION_ALPHA": "1.1"}, "at most 1"),
            ({"HOTGAZE_ACTION_COLORMAP": "rainbow"}, "colormap"),
        ],
    )
    def test_invalid_values_are_actionable(
        self, tmp_path: Path, updates: dict[str, str], message: str
    ) -> None:
        environment = _environment(tmp_path)
        environment.update(updates)

        with pytest.raises(ValueError, match=message):
            ActionInputs.from_environment(environment)

    def test_missing_and_invalid_images_fail(self, tmp_path: Path) -> None:
        environment = _environment(tmp_path)
        environment["HOTGAZE_ACTION_BASELINE"] = str(tmp_path / "missing.png")
        with pytest.raises(ValueError, match="does not exist"):
            ActionInputs.from_environment(environment)

        invalid = tmp_path / "baseline.txt"
        invalid.write_text("not an image", encoding="utf-8")
        environment["HOTGAZE_ACTION_BASELINE"] = str(invalid)
        with pytest.raises(Exception, match="Unsupported image format"):
            ActionInputs.from_environment(environment)


class TestThresholdsAndSummary:
    def test_only_losses_greater_than_threshold_fail(self) -> None:
        comparison = {
            "per_region_deltas": [
                {"name": "cta", "share_a": 0.3, "share_b": 0.2, "delta": -0.1},
                {"name": "logo", "share_a": 0.2, "share_b": 0.1, "delta": -0.1},
                {"name": "hero", "share_a": 0.1, "share_b": 0.3, "delta": 0.2},
            ]
        }

        assert evaluate_threshold(comparison, None) == ()
        assert evaluate_threshold(comparison, 0.1) == ()
        violations = evaluate_threshold(comparison, 0.099)
        assert [entry["name"] for entry in violations] == ["cta", "logo"]

    def test_summary_reports_metrics_and_failure(self, tmp_path: Path) -> None:
        environment = _environment(tmp_path)
        environment.update(
            {
                "HOTGAZE_ACTION_REGIONS": "cta:1,2,3,4",
                "HOTGAZE_ACTION_FAILURE_THRESHOLD": "0.05",
            }
        )
        inputs = ActionInputs.from_environment(environment)
        comparison = {
            "per_region_deltas": [{"name": "cta", "share_a": 0.3, "share_b": 0.2, "delta": -0.1}]
        }
        violations = evaluate_threshold(comparison, inputs.failure_threshold)

        summary = build_summary(inputs, comparison, violations)

        assert "Threshold exceeded" in summary
        assert "| cta | 0.300000 | 0.200000 | -0.100000 | Fail |" in summary
        assert "canonical schema-v1 JSON" in summary


class TestActionExecution:
    def test_invalid_execution_returns_two_and_writes_summary(self, tmp_path: Path) -> None:
        summary = tmp_path / "summary.md"
        exit_code = main({"GITHUB_STEP_SUMMARY": str(summary)})

        assert exit_code == 2
        assert "missing required input: baseline" in summary.read_text(encoding="utf-8")

    def test_threshold_failure_returns_one_after_writing_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        environment = _environment(tmp_path)
        environment.update(
            {
                "HOTGAZE_ACTION_REGIONS": "cta:1,2,3,4",
                "HOTGAZE_ACTION_FAILURE_THRESHOLD": "0.01",
                "GITHUB_OUTPUT": str(tmp_path / "github-output.txt"),
                "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
            }
        )
        output_directory = Path(environment["HOTGAZE_ACTION_OUTPUT_DIRECTORY"])
        result = ActionResult(
            baseline_score=output_directory / "baseline-score.json",
            candidate_score=output_directory / "candidate-score.json",
            compare_json=output_directory / "compare.json",
            baseline_overlay=output_directory / "baseline-overlay.png",
            candidate_overlay=output_directory / "candidate-overlay.png",
            summary="# summary\n",
            violations=({"name": "cta", "share_a": 0.2, "share_b": 0.1, "delta": -0.1},),
        )
        monkeypatch.setattr("hotgaze.action.run_action", lambda inputs: result)

        exit_code = main(environment)

        assert exit_code == 1
        assert Path(environment["GITHUB_STEP_SUMMARY"]).read_text(encoding="utf-8") == "# summary\n"
        outputs = Path(environment["GITHUB_OUTPUT"]).read_text(encoding="utf-8")
        assert "result<<" in outputs
        assert "failed" in outputs

    def test_fixture_integration_produces_schema_valid_artifacts_with_spaces(
        self, tmp_path: Path
    ) -> None:
        screenshot_directory = tmp_path / "screenshots with spaces"
        screenshot_directory.mkdir()
        baseline = screenshot_directory / "baseline image.png"
        candidate = screenshot_directory / "candidate image.png"
        shutil.copyfile("tests/fixtures/landing.png", baseline)
        shutil.copyfile("tests/fixtures/landing_variant.png", candidate)
        environment = {
            "HOTGAZE_ACTION_BASELINE": str(baseline),
            "HOTGAZE_ACTION_CANDIDATE": str(candidate),
            "HOTGAZE_ACTION_REGIONS": ("headline:0.16,0.18,0.42,0.22f\ncta:0.31,0.33,0.25,0.08f"),
            "HOTGAZE_ACTION_BACKEND": "fast",
            "HOTGAZE_ACTION_OUTPUT_DIRECTORY": str(tmp_path / "results with spaces"),
            "GITHUB_OUTPUT": str(tmp_path / "github output.txt"),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "job summary.md"),
        }
        process_environment = os.environ.copy()
        process_environment.update(environment)

        completed = subprocess.run(
            [sys.executable, "-m", "hotgaze.action"],
            env=process_environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        output_directory = Path(environment["HOTGAZE_ACTION_OUTPUT_DIRECTORY"])
        baseline_score = output_directory / "baseline-score.json"
        candidate_score = output_directory / "candidate-score.json"
        compare_json = output_directory / "compare.json"
        baseline_overlay = output_directory / "baseline-overlay.png"
        candidate_overlay = output_directory / "candidate-overlay.png"
        expected_files = (
            baseline_score,
            candidate_score,
            compare_json,
            baseline_overlay,
            candidate_overlay,
        )
        assert all(path.is_file() for path in expected_files)
        validator = Draft202012Validator(_load_schema())
        for path in (baseline_score, candidate_score, compare_json):
            validator.validate(json.loads(path.read_text(encoding="utf-8")))
        assert json.loads(compare_json.read_text(encoding="utf-8"))["schema"] == 1
        assert Image.open(baseline_overlay).size == (800, 600)
        assert Image.open(candidate_overlay).size == (800, 600)
        assert "result<<" in Path(environment["GITHUB_OUTPUT"]).read_text(encoding="utf-8")
        assert "Region attention share" in Path(environment["GITHUB_STEP_SUMMARY"]).read_text(
            encoding="utf-8"
        )


class TestActionMetadata:
    def test_defaults_and_safe_command_construction(self) -> None:
        metadata = Path("action.yml").read_text(encoding="utf-8")

        assert "default: fast" in metadata
        assert "run: python3 -m hotgaze.action" in metadata
        assert 'pip install "$GITHUB_ACTION_PATH"' in metadata
        assert 'pip install "$GITHUB_ACTION_PATH[deep]"' in metadata
        assert "if: ${{ inputs.backend == 'deep' }}" in metadata
        assert "HOTGAZE_ACTION_BASELINE: ${{ inputs.baseline }}" in metadata
        assert "HOTGAZE_ACTION_FAILURE_THRESHOLD: ${{ inputs['failure-threshold'] }}" in metadata
        assert "steps.compare.outputs['compare-json']" in metadata
        assert "run: python3 -m hotgaze.action ${{ inputs.baseline }}" not in metadata

    def test_artifacts_are_uploaded_without_token_or_write_permissions(self) -> None:
        metadata = Path("action.yml").read_text(encoding="utf-8")

        assert "uses: actions/upload-artifact@v4" in metadata
        assert "if: ${{ always() && steps.compare.outputs['output-directory'] != '' }}" in metadata
        assert "path: ${{ inputs['output-directory'] }}" not in metadata
        assert "steps.compare.outputs['baseline-score']" in metadata
        assert "steps.compare.outputs['candidate-overlay']" in metadata
        assert "github-token" not in metadata.lower()
        assert "permissions:" not in metadata


class TestActionIntegrationWorkflow:
    def test_workflow_dogfoods_local_action_and_checks_both_runs(self) -> None:
        workflow = Path(".github/workflows/action-integration.yml").read_text(encoding="utf-8")

        assert "pull_request:" in workflow
        assert "branches: [main]" in workflow
        assert 'python-version: "3.12"' in workflow
        assert workflow.count("uses: ./") == 2
        first_action = workflow.index("uses: ./")
        second_action = workflow.index("uses: ./", first_action + 1)
        assert "python3 -m pip install pillow" in workflow[:first_action]
        assert "pip install -e ." not in workflow[:second_action]
        assert "landing.png" in workflow
        assert "landing_variant.png" in workflow
        assert workflow.count("headline:0.16,0.18,0.42,0.22f") == 2
        assert workflow.count("cta:0.31,0.33,0.25,0.08f") == 2
        assert "output-directory: .github-action results/pass run" in workflow
        assert "output-directory: .github-action results/threshold run" in workflow
        assert "continue-on-error: true" in workflow
        assert 'test "$OUTCOME" = failure' in workflow
        assert workflow.count("artifact-name:") == 2
        for output_name in (
            "result",
            "output-directory",
            "baseline-score",
            "candidate-score",
            "compare-json",
            "baseline-overlay",
            "candidate-overlay",
        ):
            assert f"steps.pass.outputs['{output_name}']" in workflow
            assert f"steps.threshold.outputs['{output_name}']" in workflow
