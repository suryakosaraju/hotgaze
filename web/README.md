# 焦点 Focal — web app

A free online image tool site in Chinese and English, built on this repository:

| Page | What it does | Engine |
|---|---|---|
| `/cutout` 智能抠图 | Removes the background and exports a transparent PNG; swap in white, black or any colour before downloading. | BRIA RMBG-1.4 via [IOPaint](https://github.com/Sanster/IOPaint) |
| `/erase` 涂抹消除 | Paint over people, clutter, watermarks or text; the painted area is filled from its surroundings. Erase repeatedly, undo the last erase, hold to compare. | LaMa via IOPaint |
| `/heatmap` 注意力热力图 | Predicts where people look on a UI screenshot, scores regions, compares two versions A/B. | HotGaze (this repository) |

The home page takes an upload, a drop or a paste and asks which tool to use. The
UI is a single page (`static/index.html`, `static/css/focal.css`,
`static/js/*.js`, no build step) served by FastAPI (`app.py`). Chinese is the
default; the 中/EN switch in the header is remembered per browser, and every API
error comes back as a stable code the page translates.

Nothing is stored. Uploads live in memory or a temporary directory for the
duration of one request. There is no login, so anyone with the URL can use it.

**Non-commercial.** BRIA RMBG-1.4's licence allows non-commercial use only, and
the site says so in the cutout panel and the footer. For commercial use, switch
`REMOVE_BG_MODEL` in `services/iopaint` to a model whose licence allows it (or
license RMBG from BRIA).

## Architecture on Render

```
browser ──HTTPS──▶ hotgaze (web, free, 0.1 CPU / 512 MB)
                     ├─ heatmap: HotGaze runs in-process
                     └─ cutout / erase ──private network──▶ iopaint (pserv, 1 CPU / 2 GB)
                                                             LaMa + RMBG-1.4, one at a time
```

The web service downsizes photos to at most 2048 px on the long side, sends them
to the IOPaint private service as JSON, and streams the image back. IOPaint has
no public URL. Its image, models included, is described in
[`services/iopaint/README.md`](../services/iopaint/README.md).

## Deploy to Render

The repository root holds a Render Blueprint, [`render.yaml`](../render.yaml),
with both services.

1. Push the branch that contains `render.yaml` to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com), choose
   **New → Blueprint**, connect the repository, pick that branch and click
   **Apply**. If the first version of this Blueprint (the `hotgaze` web service
   alone) is already deployed, open that Blueprint and sync it instead: the web
   service updates in place and the `iopaint` private service is added.
3. Wait for both deploys. The IOPaint image downloads PyTorch and bakes in the
   models, so its first build takes several minutes.
4. Open the web service URL (`https://<name>.onrender.com`) and try the samples.
   Right after IOPaint starts it loads its first model, so the first cutout can
   take a minute or two; later ones take about 10–15 s.

| Service | Plan | Notes |
|---|---|---|
| `hotgaze` (web) | `free` | 512 MB RAM, 0.1 CPU. Sleeps after 15 minutes without traffic; the next visit waits about a minute while it wakes. |
| `iopaint` (private) | `1c-2g` | Paid: 1 CPU, 2 GB RAM, billed by Render while it exists. Private services do not sleep. A free web service may call a paid private service in the same region and workspace. |

Both services use region `singapore` (closest to East Asia). They must share a
region to reach each other, and Render cannot move a service after it is
created, so change both before the first deploy if you want another region.

Measured on one CPU: a cutout takes about 10–15 s; an erase 10–40 s depending on
the painted area. The private service runs one request at a time; others wait
(up to two minutes) in the web service's queue.

### Web service settings

| Variable | Default | Purpose |
|---|---|---|
| `IOPAINT_URL` | from `iopaint` | `host:port` (or a full URL) of the IOPaint service. Unset, the cutout and erase pages say the tools are unavailable. |
| `HOTGAZE_WEB_MAX_PIXELS` | `10000000` | Largest heatmap image (width × height). A 10 MP comparison peaks around 390 MB; raise it on larger instances. |
| `HOTGAZE_WEB_MAX_UPLOAD_MB` | `20` | Largest accepted upload, all tools. |
| `HOTGAZE_WEB_MAX_JOBS` | `1` | Heatmap analyses (and cutout/erase image preparation) that may run at once. Others wait up to two minutes, then get HTTP 503. |
| `HOTGAZE_ENABLE_DEEP` | `false` | Build time: install CPU-only PyTorch and the UNISAL weights for the deep heatmap backend. |
| `HOTGAZE_CACHE` | project dir | HotGaze's weight cache; `render.yaml` keeps it inside the deployed checkout. |

Cutout and erase accept photos up to 25 MP and downscale them to 2048 px on the
long side before IOPaint sees them.

`render.yaml` also sets `MALLOC_MMAP_THRESHOLD_` and `MALLOC_ARENA_MAX`, which
keep glibc from holding on to freed image buffers, and `OPENCV_LOG_LEVEL=ERROR`.

### Enabling the deep heatmap backend

The deep (UNISAL) backend needs PyTorch and about 0.75 GB of memory per
analysis, more than the free plan's 512 MB. In `render.yaml`, set the web
service's `plan: 1c-2g` (or larger) and `HOTGAZE_ENABLE_DEEP` to `"true"`, then
push. The build installs the CPU-only PyTorch wheel and prefetches the weights,
and the deep model becomes selectable on the heatmap page.

## Run locally

From the repository root:

```bash
pip install . -r web/requirements.txt
uvicorn app:app --app-dir web --reload
# open http://127.0.0.1:8000
```

The heatmap works on its own. For cutout and erase, run the IOPaint image and
point the app at it:

```bash
docker build -t focal-iopaint services/iopaint
docker run --rm -p 127.0.0.1:8080:8080 --memory 2g focal-iopaint
IOPAINT_URL=127.0.0.1:8080 uvicorn app:app --app-dir web --reload
```

## API

Interactive documentation is served at `/docs`. All endpoints take multipart
form data.

```bash
curl -F image=@photo.jpg -o cutout.png https://<name>.onrender.com/api/cutout
curl -F image=@photo.jpg -F mask=@mask.png -o erased.jpg https://<name>.onrender.com/api/erase
curl -F image=@screenshot.png -F region=cta:250,200,200,35 https://<name>.onrender.com/api/score
curl -F image_a=@baseline.png -F image_b=@candidate.png \
  -F region=cta:0.35,0.55,0.30,0.12f https://<name>.onrender.com/api/compare
```

| Endpoint | Response |
|---|---|
| `POST /api/cutout` | PNG with a transparent background. |
| `POST /api/erase` | `mask` is the same size as `image`, white where to erase. JPEG for JPEG input, PNG otherwise. |
| `POST /api/score` | `score` (schema-v1 score document, as `hotgaze score --json`), `overlay` (JPEG data URL), `elapsed_seconds` |
| `POST /api/compare` | `compare`, `score_a`, `score_b` (the documents the GitHub Action writes), `overlay_a`, `overlay_b`, `elapsed_seconds` |
| `GET /api/info` | version, available tools and backends, limits |

Image responses carry `X-Focal-Size` (the processed size) and `X-Focal-Elapsed`.
Heatmap fields: `region` is repeatable (or newline-separated) in the CLI's
`name:x,y,w,h` / `name:0.1,0.2,0.3,0.08f` syntax (any name without a colon,
Chinese included); optional `backend` (`fast` or `deep`), `faces`, `alpha` (0–1),
`colormap` (`jet` or `turbo`) and `overlay`: `focus` (the default) blends the colormap
in proportion to attention, so quiet parts of the page stay readable and `alpha` is the
opacity at the peak; `classic` is exactly the overlay `hotgaze run` writes.

Errors are JSON, `{"detail": {"code": ..., "message": ..., "params": {...}}}`:
HTTP 400 for invalid input (`unreadable_image`, `unsupported_format`,
`invalid_region`, `mask_size_mismatch`, `empty_mask`, `tool_rejected`, …), 413
for files or images over the limits, 502 when IOPaint failed (`tool_failed`),
and 503 when a service is busy or unreachable (`busy`, `tool_unavailable`). The
UI's translations live in `static/js/i18n.js`.

## Tests

```bash
pip install -e ".[dev]" -r web/requirements.txt httpx2
pytest tests/test_web.py
```

The tests replace the IOPaint call, so they run without the service. They are
skipped automatically when the web dependencies are not installed.

## Design

The site's design system is recorded in [`DESIGN.md`](../DESIGN.md); product
intent in [`PRODUCT.md`](../PRODUCT.md). Image credits and provenance:
[`static/img/PROVENANCE.md`](static/img/PROVENANCE.md).
