# HotGaze web demo

A small browser front end and JSON API for trying HotGaze without installing
anything: upload a screenshot to get the attention overlay, region scores and
focal points, or compare two variants A/B. It wraps the same engine and
canonical schema-v1 JSON as the CLI — the `score` document returned by
`POST /api/score` is exactly what `hotgaze score --json` prints.

Uploads live in a temporary directory for the duration of one request and are
deleted afterwards; nothing is stored. The service has no login, so anyone with
the URL can use it.

## Deploy to Render

The repository root holds a Render Blueprint, [`render.yaml`](../render.yaml).

1. Push the branch that contains `render.yaml` to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com), choose
   **New → Blueprint**, connect the repository, pick that branch and click
   **Apply**.
3. When the first deploy finishes, open the service URL
   (`https://<name>.onrender.com`) and click **Try a sample**.

For a public repository you can also use the button below, which deploys the
Blueprint from the default branch:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/DietCokewithSugar/hotgaze)

What the Blueprint sets up:

| Setting | Value | Notes |
|---|---|---|
| Plan | `free` | 512 MB RAM, 0.1 CPU. Sleeps after 15 minutes without traffic; the next visit waits about a minute while it wakes. |
| Region | `singapore` | Closest to East Asia. Change it before the first deploy — Render cannot move an existing service. |
| Python | `3.12.13` | Via `PYTHON_VERSION` (Render's default is newer than the versions this project tests). |
| Build | `bash web/render-build.sh` | Installs HotGaze and the web server, prefetches the YuNet face weights into `HOTGAZE_CACHE` and generates the sample screenshots, so the running service never downloads anything. |
| Start | `uvicorn app:app --app-dir web --host 0.0.0.0 --port $PORT` | One worker (`WEB_CONCURRENCY=1`). |
| Health check | `/healthz` | Answered even while an analysis is running. |

On the free plan's 0.1 CPU, expect roughly 5–15 s for a typical screenshot and
twice that for a comparison — about ten times as long as on one full CPU core,
and longer for very large images. A paid instance type such as `0.5c-512mb`
gets five times the CPU.

### Settings

| Variable | Default | Purpose |
|---|---|---|
| `HOTGAZE_WEB_MAX_PIXELS` | `10000000` | Largest accepted image (width × height). A 10 MP comparison peaks around 390 MB of memory; raise it on larger instances. |
| `HOTGAZE_WEB_MAX_UPLOAD_MB` | `20` | Largest accepted upload. |
| `HOTGAZE_WEB_MAX_JOBS` | `1` | Analyses that may run at once. Others wait up to two minutes, then get HTTP 503. |
| `HOTGAZE_ENABLE_DEEP` | `false` | Build time: install CPU-only PyTorch and the UNISAL weights for the deep backend. |
| `HOTGAZE_CACHE` | project dir | HotGaze's weight cache; `render.yaml` keeps it inside the deployed checkout. |

`render.yaml` also sets `MALLOC_MMAP_THRESHOLD_` and `MALLOC_ARENA_MAX`, which
keep glibc from holding on to freed image buffers (without them the same 10 MP
comparison peaks around 460 MB), and `OPENCV_LOG_LEVEL=ERROR`.

### Enabling the deep backend

The deep (UNISAL) backend needs PyTorch and about 0.75 GB of memory per
analysis, more than the free plan's 512 MB. In `render.yaml`, set
`plan: 1c-2g` (or larger) and `HOTGAZE_ENABLE_DEEP` to `"true"`, then push.
The build installs the CPU-only PyTorch wheel and prefetches the weights, and
**deep — UNISAL** becomes selectable in the UI. Paid instance types are billed
by Render.

## Run locally

From the repository root:

```bash
pip install . -r web/requirements.txt
uvicorn app:app --app-dir web --reload
# open http://127.0.0.1:8000
```

For the deep backend, also install PyTorch, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## API

Interactive documentation is served at `/docs`. Both endpoints take
multipart form data; `region` is repeatable (or newline-separated) and uses the
CLI's `name:x,y,w,h` / `name:0.1,0.2,0.3,0.08f` syntax.

```bash
curl -F image=@screenshot.png -F region=cta:250,200,200,35 \
  https://<name>.onrender.com/api/score

curl -F image_a=@baseline.png -F image_b=@candidate.png \
  -F region=cta:0.35,0.55,0.30,0.12f \
  https://<name>.onrender.com/api/compare
```

| Endpoint | Response fields |
|---|---|
| `POST /api/score` | `score` (schema-v1 score document), `overlay` (JPEG data URL), `elapsed_seconds` |
| `POST /api/compare` | `compare`, `score_a`, `score_b` (the same three documents the GitHub Action writes), `overlay_a`, `overlay_b`, `elapsed_seconds` |
| `GET /api/info` | version, available backends, limits |

Optional fields: `backend` (`fast` or `deep`), `faces` (`true` adds the YuNet
faces layer), `alpha` (overlay opacity, 0–1) and `colormap` (`jet` or `turbo`).
Errors come back as `{"detail": ...}` with HTTP 400 for invalid input, 413 for
images over the limits and 503 when the server is busy.

## Tests

```bash
pip install -e ".[dev]" -r web/requirements.txt httpx2
pytest tests/test_web.py
```

The tests are skipped automatically when the web dependencies are not
installed, so the regular CI run is unaffected.
