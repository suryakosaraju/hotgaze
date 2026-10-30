"""Tests for the web demo in web/app.py.

Skipped unless the web dependencies and Starlette's test client are installed:
pip install -r web/requirements.txt httpx2
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from PIL import Image

from hotgaze.config import EngineConfig
from hotgaze.engine import run_engine
from hotgaze.scoring import score_regions, scores_to_json, validate_against_schema

_WEB_APP = Path(__file__).parent.parent / "web" / "app.py"


def _fixture(name: str) -> Path:
    return Path(__file__).parent / "fixtures" / name


def _upload(path: Path) -> tuple[str, bytes, str]:
    return (path.name, path.read_bytes(), "image/png")


def _image_bytes(fmt: str, size: tuple[int, int] = (40, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


def _decode_overlay(data_url: str) -> Image.Image:
    header, payload = data_url.split(",", 1)
    assert header == "data:image/jpeg;base64"
    return Image.open(io.BytesIO(base64.b64decode(payload)))


@pytest.fixture(scope="module")
def web() -> ModuleType:
    pytest.importorskip("fastapi")
    pytest.importorskip("python_multipart")
    if importlib.util.find_spec("httpx2") is None:  # older Starlette test clients use httpx
        pytest.importorskip("httpx")
    spec = importlib.util.spec_from_file_location("hotgaze_web_app", _WEB_APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(web: ModuleType) -> Any:
    from fastapi.testclient import TestClient

    return TestClient(web.app)


class TestPages:
    def test_healthz(self, client: Any) -> None:
        assert client.get("/healthz").json() == {"status": "ok"}

    def test_index_is_served(self, client: Any) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "<title>HotGaze" in response.text

    def test_info_lists_fast_backend_and_samples(self, client: Any) -> None:
        info = client.get("/api/info").json()
        assert "fast" in info["backends"]
        assert info["layers"] == ["faces"]
        assert info["samples"] == ["landing.png", "landing_variant.png"]
        assert info["limits"]["max_pixels"] > 0

    def test_sample_is_a_png(self, client: Any) -> None:
        response = client.get("/samples/landing.png")
        assert response.status_code == 200
        assert Image.open(io.BytesIO(response.content)).size == (800, 600)

    def test_only_known_samples_are_served(self, client: Any) -> None:
        assert client.get("/samples/1440x900.png").status_code == 404


class TestScore:
    def test_matches_library_output(self, client: Any) -> None:
        """The API embeds exactly the document `hotgaze score --json` would print."""
        regions = ["cta:250,200,200,35", "headline:140,120,300,60"]
        response = client.post(
            "/api/score",
            files={"image": _upload(_fixture("landing.png"))},
            data={"region": regions},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        config = EngineConfig.fast_default()
        attn = run_engine(str(_fixture("landing.png")), config=config)
        scored, focal = score_regions(attn, regions)
        expected = scores_to_json(
            "score",
            "landing.png",
            attn.original_size,
            attn.working_size,
            config.model_dump(),
            scored,
            focal,
        )
        assert body["score"] == json.loads(expected)
        assert validate_against_schema(body["score"]) == []
        assert _decode_overlay(body["overlay"]).size == (800, 600)

    def test_newline_separated_regions(self, client: Any) -> None:
        response = client.post(
            "/api/score",
            files={"image": _upload(_fixture("landing.png"))},
            data={"region": "cta:250,200,200,35\n\nheadline:0.1,0.1,0.5,0.2f\n"},
        )
        assert response.status_code == 200, response.text
        names = {region["name"] for region in response.json()["score"]["regions"]}
        assert names == {"cta", "headline"}

    def test_malformed_region_is_rejected_before_analysis(
        self, client: Any, web: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*args: object, **kwargs: object) -> None:
            raise AssertionError("engine should not run for a malformed region")

        monkeypatch.setattr(web, "run_engine", fail)
        response = client.post(
            "/api/score",
            files={"image": _upload(_fixture("landing.png"))},
            data={"region": "cta:1.5,2,3,4"},
        )
        assert response.status_code == 400
        assert "looks fractional" in response.json()["detail"]

    def test_out_of_bounds_region_reports_library_message(self, client: Any) -> None:
        response = client.post(
            "/api/score",
            files={"image": _upload(_fixture("landing.png"))},
            data={"region": "cta:5000,5000,10,10"},
        )
        assert response.status_code == 400
        assert "out of bounds" in response.json()["detail"]

    def test_invalid_alpha_is_a_validation_error(self, client: Any) -> None:
        response = client.post(
            "/api/score",
            files={"image": _upload(_fixture("landing.png"))},
            data={"alpha": "1.5"},
        )
        assert response.status_code == 422


class TestCompare:
    def test_region_deltas(self, client: Any) -> None:
        response = client.post(
            "/api/compare",
            files={
                "image_a": _upload(_fixture("landing.png")),
                "image_b": _upload(_fixture("landing_variant.png")),
            },
            data={"region": ["old_cta:250,200,200,35", "new_cta:550,110,130,35"]},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert validate_against_schema(body["compare"]) == []
        assert validate_against_schema(body["score_a"]) == []
        assert validate_against_schema(body["score_b"]) == []
        assert body["compare"]["image_b"]["path"] == "landing_variant.png"
        deltas = {d["name"]: d["delta"] for d in body["compare"]["compare"]["per_region_deltas"]}
        assert deltas["old_cta"] < 0 < deltas["new_cta"]
        assert _decode_overlay(body["overlay_b"]).size == (800, 600)

    def test_grid_without_regions(self, client: Any) -> None:
        response = client.post(
            "/api/compare",
            files={
                "image_a": _upload(_fixture("landing.png")),
                "image_b": _upload(_fixture("landing_variant.png")),
            },
        )
        assert response.status_code == 200, response.text
        compare = response.json()["compare"]["compare"]
        assert len(compare["grid_deltas"]) == 9
        assert compare["focal_point_movement"]

    def test_size_mismatch_reports_library_message(self, client: Any) -> None:
        response = client.post(
            "/api/compare",
            files={
                "image_a": _upload(_fixture("landing.png")),
                "image_b": _upload(_fixture("1440x900.png")),
            },
        )
        assert response.status_code == 400
        assert "different sizes" in response.json()["detail"]


class TestLimits:
    def test_unsupported_format(self, client: Any) -> None:
        response = client.post("/api/score", files={"image": ("x.gif", _image_bytes("GIF"))})
        assert response.status_code == 400
        assert "unsupported image format GIF" in response.json()["detail"]

    def test_not_an_image(self, client: Any) -> None:
        response = client.post("/api/score", files={"image": ("x.png", b"hello")})
        assert response.status_code == 400
        assert "not a readable" in response.json()["detail"]

    def test_pixel_limit(
        self, client: Any, web: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(web, "MAX_PIXELS", 1000)
        response = client.post("/api/score", files={"image": ("x.png", _image_bytes("PNG"))})
        assert response.status_code == 413
        assert "40×30" in response.json()["detail"]

    def test_upload_limit(
        self, client: Any, web: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(web, "MAX_UPLOAD_BYTES", 10)
        response = client.post("/api/score", files={"image": ("x.png", _image_bytes("PNG"))})
        assert response.status_code == 413

    def test_deep_backend_unavailable(
        self, client: Any, web: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(web, "DEEP_AVAILABLE", False)
        response = client.post(
            "/api/score",
            files={"image": _upload(_fixture("landing.png"))},
            data={"backend": "deep"},
        )
        assert response.status_code == 400
        assert "deep backend is not installed" in response.json()["detail"]

    def test_busy_server(
        self, client: Any, web: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slots = threading.BoundedSemaphore(1)
        slots.acquire()
        monkeypatch.setattr(web, "_job_slots", slots)
        monkeypatch.setattr(web, "QUEUE_TIMEOUT_SECONDS", 0.01)
        response = client.post("/api/score", files={"image": _upload(_fixture("landing.png"))})
        assert response.status_code == 503


class TestSettings:
    def test_faces_flag_enables_layer(self, web: ModuleType) -> None:
        assert web._engine_config("fast", True).extra_layers == ["faces"]
        assert web._engine_config("fast", False).extra_layers == []

    def test_deep_config_when_available(
        self, web: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(web, "DEEP_AVAILABLE", True)
        assert web._engine_config("deep", False).backend == "deep"

    @pytest.mark.parametrize("raw", ["abc", "0", "-5"])
    def test_int_setting_rejects_invalid(
        self, web: ModuleType, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("HOTGAZE_TEST_SETTING", raw)
        with pytest.raises(ValueError, match="HOTGAZE_TEST_SETTING"):
            web._int_setting("HOTGAZE_TEST_SETTING", 5)

    def test_int_setting_default_and_value(
        self, web: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HOTGAZE_TEST_SETTING", raising=False)
        assert web._int_setting("HOTGAZE_TEST_SETTING", 5) == 5
        monkeypatch.setenv("HOTGAZE_TEST_SETTING", " 7 ")
        assert web._int_setting("HOTGAZE_TEST_SETTING", 5) == 7
