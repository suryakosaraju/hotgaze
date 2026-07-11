"""Tests for weight download and cache management (T3.1)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hotgaze.weights import (
    WeightSpec,
    _verify_checksum,
    download_weight,
    get_cache_dir,
    get_weight_spec,
)

# Sample weight data (not a real model — just bytes for testing)
_SAMPLE_DATA = b"mock-weight-data" * 100  # ~1.7 KB reproducible
_SAMPLE_SHA256 = hashlib.sha256(_SAMPLE_DATA).hexdigest()
_SAMPLE_SPEC = WeightSpec(
    name="test", filename="test.pth", sha256=_SAMPLE_SHA256, url="https://example.com/test.pth"
)

_WRONG_DATA = b"tampered-data" * 100
_WRONG_SHA256 = hashlib.sha256(_WRONG_DATA).hexdigest()


def _fake_urlopen(data: bytes, total: int | None = None):
    """Create a mock urlopen that returns a BytesIO."""

    def opener(url, timeout=30):
        resp = MagicMock()
        resp.headers = {"Content-Length": str(total or len(data))}
        resp.read.side_effect = [data, b""]
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: None
        return resp

    return opener


# ── Registry ─────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_get_weight_spec_returns_spec(self) -> None:
        spec = get_weight_spec("unisal")
        assert isinstance(spec, WeightSpec)
        assert spec.name == "unisal"

    def test_unknown_weight_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown weight"):
            get_weight_spec("nonexistent")

    def test_all_three_registered(self) -> None:
        for name in ("unisal", "unisal_mit1003", "yunet"):
            spec = get_weight_spec(name)
            assert len(spec.sha256) == 64
            assert spec.filename
            assert spec.url.startswith("https://")


# ── Cache ────────────────────────────────────────────────────────────────────


class TestCache:
    def test_default_cache_dir(self) -> None:
        d = get_cache_dir()
        assert str(d).endswith(".cache/hotgaze")

    def test_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("HOTGAZE_CACHE", "/tmp/my-hotgaze-cache")
        assert str(get_cache_dir()) == "/tmp/my-hotgaze-cache"


# ── Download (mocked) ────────────────────────────────────────────────────────


class TestDownload:
    def test_cache_hit_zero_network(self, tmp_path: Path) -> None:  # noqa: SIM117
        """Existing cached file → zero network calls, returns immediately."""
        dest = tmp_path / "test.pth"
        dest.write_bytes(_SAMPLE_DATA)

        with patch("urllib.request.urlopen") as mock_open:  # noqa: SIM117
            with patch("hotgaze.weights.get_weight_spec", return_value=_SAMPLE_SPEC):
                result = download_weight("test", cache_dir=tmp_path)
                assert result == dest
                mock_open.assert_not_called()

    def test_downloads_and_caches(self, tmp_path: Path) -> None:
        """First download → fetches, verifies, caches."""
        with (
            patch(
                "urllib.request.urlopen",
                _fake_urlopen(_SAMPLE_DATA),
            ),
            patch("hotgaze.weights.get_weight_spec", return_value=_SAMPLE_SPEC),
        ):
            result = download_weight("test", cache_dir=tmp_path, progress=False)

        assert result.name == "test.pth"
        assert result.exists()
        assert result.read_bytes() == _SAMPLE_DATA
        # No temp file left behind
        assert not list(tmp_path.glob("*.tmp"))

    def test_checksum_mismatch_cleans_up(self, tmp_path: Path) -> None:
        """Checksum mismatch → hard error, temp file deleted, final path empty."""
        with (  # noqa: SIM117
            patch(  # noqa: SIM117
                "urllib.request.urlopen",
                _fake_urlopen(_WRONG_DATA),
            ),
            patch("hotgaze.weights.get_weight_spec", return_value=_SAMPLE_SPEC),
        ):
            with pytest.raises(ValueError, match="Checksum mismatch"):
                download_weight("test", cache_dir=tmp_path, progress=False)

        # No final file
        assert not (tmp_path / "test.pth").exists()
        # No temp file left behind
        assert not list(tmp_path.glob("*.tmp"))

    def test_atomic_rename_no_partial_file(self, tmp_path: Path) -> None:
        """Simulate interrupt: temp file exists but final doesn't."""
        # Write a temp file manually, then call download — it should overwrite
        tmp_file = tmp_path / ".test.pth.tmp"
        tmp_file.write_bytes(b"partial")
        dest = tmp_path / "test.pth"

        with (
            patch(
                "urllib.request.urlopen",
                _fake_urlopen(_SAMPLE_DATA),
            ),
            patch("hotgaze.weights.get_weight_spec", return_value=_SAMPLE_SPEC),
        ):
            result = download_weight("test", cache_dir=tmp_path, progress=False)

        assert result == dest
        assert dest.read_bytes() == _SAMPLE_DATA

    def test_env_override_respected(self, tmp_path: Path, monkeypatch) -> None:
        """HOTGAZE_CACHE env var controls cache location."""
        monkeypatch.setenv("HOTGAZE_CACHE", str(tmp_path / "custom-cache"))

        with (
            patch(
                "urllib.request.urlopen",
                _fake_urlopen(_SAMPLE_DATA),
            ),
            patch("hotgaze.weights.get_weight_spec", return_value=_SAMPLE_SPEC),
        ):
            result = download_weight("test", progress=False)

        assert str(tmp_path / "custom-cache") in str(result)


# ── Checksum verification ────────────────────────────────────────────────────


class TestChecksum:
    def test_matching_checksum_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "ok.bin"
        f.write_bytes(_SAMPLE_DATA)
        _verify_checksum(f, _SAMPLE_SHA256, "test")  # no exception

    def test_mismatch_raises_and_deletes(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.bin"
        f.write_bytes(_WRONG_DATA)
        with pytest.raises(ValueError, match="Checksum mismatch"):
            _verify_checksum(f, _SAMPLE_SHA256, "test")
        assert not f.exists()


# ── Network error ────────────────────────────────────────────────────────────


class TestNetworkError:
    def test_offline_error_actionable(self, tmp_path: Path) -> None:
        """Unreachable URL → actionable error naming the file and URL."""
        import urllib.error

        def _raise(*a, **kw):
            raise urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", _raise):  # noqa: SIM117
            with patch("hotgaze.weights.get_weight_spec", return_value=_SAMPLE_SPEC):
                with pytest.raises(FileNotFoundError, match="Cannot download"):
                    download_weight("test", cache_dir=tmp_path, progress=False)
