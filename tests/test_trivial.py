"""Trivial test to confirm the scaffold works."""

import hotgaze


def test_version():
    """Package is importable and has a version string."""
    assert isinstance(hotgaze.__version__, str)
    assert hotgaze.__version__ == "0.1.0"
