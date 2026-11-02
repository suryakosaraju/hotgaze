"""Tests for the web app in web/app.py.

Skipped unless the web dependencies and Starlette's test client are installed:
pip install -r web/requirements.txt httpx2
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import re
import sys
import threading
import urllib.error
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


def _image_bytes(fmt: str, size: tuple[int, int] = (40, 30), **save: Any) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, format=fmt, **save)
    return buffer.getvalue()


def _mask_bytes(size: tuple[int, int], painted: bool = True) -> bytes:
    mask = Image.new("L", size, 0)
    if painted:
        mask.paste(255, (size[0] // 4, size[1] // 4, size[0] // 2, size[1] // 2))
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_overlay(data_url: str) -> Image.Image:
    header, payload = data_url.split(",", 1)
    assert header == "data:image/jpeg;base64"
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def _detail(response: Any) -> dict[str, Any]:
    detail: dict[str, Any] = response.json()["detail"]
    return detail


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


@pytest.fixture
def iopaint(web: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Replace the IOPaint call; records (path, payload) and answers with a 4×4 PNG."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake(path: str, payload: dict[str, Any]) -> bytes:
        calls.append((path, payload))
        return _image_bytes("PNG", (4, 4))

    monkeypatch.setattr(web, "_iopaint", fake)
    return calls


def _sent_image(payload: dict[str, Any], key: str = "image") -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(payload[key])))


class TestPages:
    def test_healthz(self, client: Any) -> None:
        assert client.get("/healthz").json() == {"status": "ok"}

    @pytest.mark.parametrize("route", ["/", "/heatmap", "/cutout", "/erase"])
    def test_every_route_serves_the_app(self, client: Any, route: str) -> None:
        response = client.get(route)
        assert response.status_code == 200
        assert "<title>" in response.text

    def test_static_files_are_served(self, client: Any) -> None:
        assert client.get("/static/index.html").status_code == 200

    def test_info_lists_tools_backends_and_samples(self, client: Any, web: ModuleType) -> None:
        info = client.get("/api/info").json()
        assert info["tools"]["heatmap"] is True
        assert info["tools"]["cutout"] is bool(web.IOPAINT_URL)
        assert "fast" in info["backends"]
        assert info["samples"] == ["landing.png", "landing_variant.png"]
        assert info["limits"]["edit_max_edge"] == 2048

    def test_sample_is_a_png(self, client: Any) -> None:
        response = client.get("/samples/landing.png")
        assert response.status_code == 200
        assert Image.open(io.BytesIO(response.content)).size == (800, 600)

    def test_only_known_samples_are_served(self, client: Any) -> None:
        assert client.get("/samples/1440x900.png").status_code == 404

    def test_every_asset_the_page_references_is_served(self, client: Any, web: ModuleType) -> None:
        html = client.get("/").text
        scripts = "".join(path.read_text() for path in (web._STATIC_DIR / "js").glob("*.js"))
        paths = set(re.findall(r'(?:src|href)="(/static/[^"#]+)"', html))
        paths |= set(re.findall(r'"(/static/[^"]+)"', scripts))
        assert "/static/js/app.js" in paths
        for path in sorted(paths):
            assert client.get(path).status_code == 200, path

    def test_both_languages_define_every_ui_string(self, web: ModuleType) -> None:
        source = (web._STATIC_DIR / "js" / "i18n.js").read_text()

        def keys(lang: str) -> set[str]:
            start = source.index(f"  {lang}: {{")
            block = source[start : source.index("\n  },", start)]
            return set(re.findall(r'^\s+"([^"]+)":', block, re.MULTILINE))

        chinese, english = keys("zh"), keys("en")
        assert chinese == english
        html = (web._STATIC_DIR / "index.html").read_text()
        used = set(re.findall(r'data-i18n="([^"]+)"', html))
        for pairs in re.findall(r'data-i18n-attr="([^"]+)"', html):
            used |= {pair.split(":")[1].strip() for pair in pairs.split(";")}
        assert used <= chinese, sorted(used - chinese)

    def test_every_api_error_code_has_a_translation(self, web: ModuleType) -> None:
        source = (web._STATIC_DIR / "js" / "i18n.js").read_text()
        app_source = Path(web.__file__).read_text()
        codes = set(re.findall(r'_error\(\s*\d+,\s*"([a-z_]+)"', app_source))
        codes |= {code for _, code in web._LIBRARY_ERRORS} | {"invalid_input"}
        assert {"tool_failed", "empty_mask", "file_too_large"} <= codes
        for code in sorted(codes):
            assert source.count(f'"errors.{code}":') == 2, code


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

    def test_classic_overlay_is_exactly_hotgaze_run(self, client: Any) -> None:
        response = client.post(
            "/api/score",
            files={"image": _upload(_fixture("landing.png"))},
            data={"overlay": "classic", "alpha": "0.6"},
        )
        assert response.status_code == 200, response.text
        attn = run_engine(str(_fixture("landing.png")), config=EngineConfig.fast_default())
        expected = attn.overlay(Image.open(_fixture("landing.png")), alpha=0.6)
        buffer = io.BytesIO()
        expected.save(buffer, format="JPEG", quality=90)
        assert response.json()["overlay"].split(",", 1)[1] == base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

    def test_focus_overlay_leaves_quiet_areas_untinted(self, client: Any, web: ModuleType) -> None:
        """The default overlay keeps low-attention pixels as they were in the screenshot."""
        response = client.post("/api/score", files={"image": _upload(_fixture("landing.png"))})
        assert response.status_code == 200, response.text
        overlay = _decode_overlay(response.json()["overlay"]).convert("RGB")
        original = Image.open(_fixture("landing.png")).convert("RGB")
        attn = run_engine(str(_fixture("landing.png")), config=EngineConfig.fast_default())
        heat = Image.fromarray((attn.heatmap * 255).astype("uint8")).resize(original.size)
        quiet = [
            (x, y)
            for x in range(0, original.width, 37)
            for y in range(0, original.height, 37)
            if heat.getpixel((x, y)) < 255 * web._FOCUS_FLOOR * 0.5
        ]
        assert quiet, "the fixture should have low-attention areas"
        # JPEG noise only: within a few levels of the original.
        for point in quiet:
            pairs = zip(overlay.getpixel(point), original.getpixel(point), strict=True)
            assert max(abs(a - b) for a, b in pairs) <= 12, point

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
        detail = _detail(response)
        assert detail["code"] == "invalid_region"
        assert detail["params"]["region"] == "cta:1.5,2,3,4"
        assert "looks fractional" in detail["message"]

    def test_out_of_bounds_region(self, client: Any) -> None:
        response = client.post(
            "/api/score",
            files={"image": _upload(_fixture("landing.png"))},
            data={"region": "cta:5000,5000,10,10"},
        )
        assert response.status_code == 400
        assert _detail(response)["code"] == "region_out_of_bounds"

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

    def test_size_mismatch(self, client: Any) -> None:
        response = client.post(
            "/api/compare",
            files={
                "image_a": _upload(_fixture("landing.png")),
                "image_b": _upload(_fixture("1440x900.png")),
            },
        )
        assert response.status_code == 400
        assert _detail(response)["code"] == "size_mismatch"

    def test_duplicate_region_names(self, client: Any) -> None:
        response = client.post(
            "/api/compare",
            files={
                "image_a": _upload(_fixture("landing.png")),
                "image_b": _upload(_fixture("landing_variant.png")),
            },
            data={"region": ["a:1,1,9,9", "a:2,2,9,9"]},
        )
        assert response.status_code == 400
        assert _detail(response)["code"] == "duplicate_region"


class TestCutout:
    def test_returns_the_service_png(self, client: Any, iopaint: list[Any]) -> None:
        response = client.post("/api/cutout", files={"image": _upload(_fixture("landing.png"))})
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.headers["x-focal-size"] == "800x600"
        path, payload = iopaint[0]
        assert path == "/api/v1/run_plugin_gen_image"
        assert payload["name"] == "RemoveBG"
        assert _sent_image(payload).size == (800, 600)

    def test_large_photos_are_downscaled(self, client: Any, iopaint: list[Any]) -> None:
        big = _image_bytes("JPEG", (3000, 2000))
        response = client.post("/api/cutout", files={"image": ("big.jpg", big, "image/jpeg")})
        assert response.status_code == 200, response.text
        assert _sent_image(iopaint[0][1]).size == (2048, 1365)
        assert response.headers["x-focal-size"] == "2048x1365"

    def test_unsupported_format(self, client: Any, iopaint: list[Any]) -> None:
        response = client.post("/api/cutout", files={"image": ("x.gif", _image_bytes("GIF"))})
        assert response.status_code == 400
        assert _detail(response)["code"] == "unsupported_format"
        assert iopaint == []


class TestErase:
    def test_sends_image_and_mask(self, client: Any, iopaint: list[Any]) -> None:
        response = client.post(
            "/api/erase",
            files={
                "image": ("photo.png", _image_bytes("PNG", (64, 48)), "image/png"),
                "mask": ("mask.png", _mask_bytes((64, 48)), "image/png"),
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        path, payload = iopaint[0]
        assert path == "/api/v1/inpaint"
        assert _sent_image(payload).size == (64, 48)
        mask = _sent_image(payload, "mask")
        assert mask.mode == "L" and {value for _, value in mask.getcolors()} == {0, 255}

    def test_jpeg_in_jpeg_out(self, client: Any, iopaint: list[Any]) -> None:
        response = client.post(
            "/api/erase",
            files={
                "image": ("photo.jpg", _image_bytes("JPEG", (64, 48)), "image/jpeg"),
                "mask": ("mask.png", _mask_bytes((64, 48)), "image/png"),
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/jpeg"
        assert _sent_image(iopaint[0][1]).format == "JPEG"

    def test_mask_is_compared_with_the_oriented_image(
        self, client: Any, iopaint: list[Any]
    ) -> None:
        """A phone photo stored sideways (EXIF orientation 6) is edited as the browser shows it."""
        exif = Image.Exif()
        exif[274] = 6
        photo = _image_bytes("JPEG", (64, 48), exif=exif)
        response = client.post(
            "/api/erase",
            files={
                "image": ("photo.jpg", photo, "image/jpeg"),
                "mask": ("mask.png", _mask_bytes((48, 64)), "image/png"),
            },
        )
        assert response.status_code == 200, response.text
        assert _sent_image(iopaint[0][1]).size == (48, 64)

    def test_mask_size_mismatch(self, client: Any, iopaint: list[Any]) -> None:
        response = client.post(
            "/api/erase",
            files={
                "image": ("photo.png", _image_bytes("PNG", (64, 48)), "image/png"),
                "mask": ("mask.png", _mask_bytes((32, 24)), "image/png"),
            },
        )
        assert response.status_code == 400
        detail = _detail(response)
        assert detail["code"] == "mask_size_mismatch"
        assert detail["params"] == {"image": "64×48", "mask": "32×24"}
        assert iopaint == []

    def test_empty_mask(self, client: Any, iopaint: list[Any]) -> None:
        response = client.post(
            "/api/erase",
            files={
                "image": ("photo.png", _image_bytes("PNG", (64, 48)), "image/png"),
                "mask": ("mask.png", _mask_bytes((64, 48), painted=False), "image/png"),
            },
        )
        assert response.status_code == 400
        assert _detail(response)["code"] == "empty_mask"


class TestIOPaintService:
    def test_service_url_accepts_render_hostport(self, web: ModuleType) -> None:
        assert web._service_url("focal-iopaint:10000") == "http://focal-iopaint:10000"
        assert web._service_url(" http://127.0.0.1:8080/ ") == "http://127.0.0.1:8080"
        assert web._service_url("") == ""

    def test_unconfigured(
        self, client: Any, web: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(web, "IOPAINT_URL", "")
        response = client.post("/api/cutout", files={"image": _upload(_fixture("landing.png"))})
        assert response.status_code == 503
        assert _detail(response)["code"] == "tool_unavailable"

    @pytest.mark.parametrize(
        ("error", "status", "code"),
        [
            (
                urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b'{"detail":"x"}')),
                502,
                "tool_failed",
            ),
            (
                urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b'{"detail":"x"}')),
                400,
                "tool_rejected",
            ),
            (urllib.error.URLError("refused"), 503, "tool_unavailable"),
            (TimeoutError("slow"), 503, "tool_unavailable"),
        ],
    )
    def test_service_errors(
        self,
        client: Any,
        web: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
        status: int,
        code: str,
    ) -> None:
        def fail(*args: object, **kwargs: object) -> None:
            raise error

        monkeypatch.setattr(web, "IOPAINT_URL", "http://iopaint.invalid")
        monkeypatch.setattr(web._iopaint_opener, "open", fail)
        response = client.post("/api/cutout", files={"image": _upload(_fixture("landing.png"))})
        assert response.status_code == status
        assert _detail(response)["code"] == code


class TestLimits:
    def test_not_an_image(self, client: Any) -> None:
        response = client.post("/api/score", files={"image": ("x.png", b"hello")})
        assert response.status_code == 400
        assert _detail(response)["code"] == "unreadable_image"

    def test_pixel_limit(
        self, client: Any, web: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(web, "MAX_PIXELS", 1000)
        response = client.post("/api/score", files={"image": ("x.png", _image_bytes("PNG"))})
        assert response.status_code == 413
        detail = _detail(response)
        assert detail["code"] == "too_many_pixels"
        assert (detail["params"]["width"], detail["params"]["height"]) == (40, 30)

    def test_upload_limit(
        self, client: Any, web: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(web, "MAX_UPLOAD_BYTES", 10)
        response = client.post("/api/score", files={"image": ("x.png", _image_bytes("PNG"))})
        assert response.status_code == 413
        assert _detail(response)["code"] == "file_too_large"

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
        assert _detail(response)["code"] == "deep_unavailable"

    def test_busy_server(
        self, client: Any, web: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slots = threading.BoundedSemaphore(1)
        slots.acquire()
        monkeypatch.setattr(web, "_job_slots", slots)
        monkeypatch.setattr(web, "QUEUE_TIMEOUT_SECONDS", 0.01)
        response = client.post("/api/score", files={"image": _upload(_fixture("landing.png"))})
        assert response.status_code == 503
        assert _detail(response)["code"] == "busy"


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
