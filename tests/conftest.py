"""Pytest configuration — auto-generate test fixtures on demand.

Per PLAN.md T1.4 item 1: do NOT commit binary PNGs. This session-scoped
fixture runs scripts/generate_fixtures.py before any test when the
fixture directory is empty or missing, so a fresh clone just works.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_EXPECTED_FILES = ["landing.png", "landing_variant.png", "1440x900.png"]


def _fixtures_exist() -> bool:
    return all((_FIXTURE_DIR / f).exists() for f in _EXPECTED_FILES)


@pytest.fixture(scope="session", autouse=True)
def _ensure_fixtures() -> None:
    """Generate test fixtures if they don't exist (fresh clone)."""
    if _fixtures_exist():
        return

    script = Path(__file__).parent.parent / "scripts" / "generate_fixtures.py"
    if not script.exists():
        pytest.skip(f"Fixture generator not found: {script}")

    result = subprocess.run(
        ["python", str(script)],
        capture_output=True,
        text=True,
        cwd=str(script.parent.parent),
        timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(f"Fixture generation failed:\n{result.stderr}")

    assert _fixtures_exist(), "Fixtures not created after generation"
