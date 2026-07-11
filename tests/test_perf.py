"""Performance test for the fast backend.

Per CLAUDE.md: end-to-end under 10 seconds for 1440×900 on CPU.
This test is marked @pytest.mark.perf and excluded from CI (shared runners flake).
"""

import time
from pathlib import Path

import pytest

from hotgaze.config import EngineConfig
from hotgaze.engine import run_engine


@pytest.mark.perf
def test_fast_backend_performance() -> None:
    """Fast backend completes a 1440×900 image in under 10 seconds."""
    fixture = Path(__file__).parent / "fixtures" / "1440x900.png"
    assert fixture.exists(), f"Fixture not found: {fixture}"

    config = EngineConfig.fast_default()

    start = time.perf_counter()
    result = run_engine(str(fixture), config=config)
    elapsed = time.perf_counter() - start

    assert result.heatmap.shape == (900, 1440)
    assert elapsed < 10.0, f"Fast backend took {elapsed:.1f}s, expected < 10s"
