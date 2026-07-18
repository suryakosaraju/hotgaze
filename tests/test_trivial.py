"""Trivial test to confirm the scaffold works."""

import hotgaze


def test_version():
    """Package is importable and has a non-empty version string."""
    assert isinstance(hotgaze.__version__, str)
    assert len(hotgaze.__version__) > 0
    assert hotgaze.__version__[0].isdigit()
