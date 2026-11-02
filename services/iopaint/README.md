# IOPaint service (cutout and erase)

The image service behind Focal's 智能抠图 (cutout) and 涂抹消除 (erase) pages:
[IOPaint](https://github.com/Sanster/IOPaint) 1.6.0 (Apache-2.0) on CPU, with

- **LaMa** (Apache-2.0) to fill in what the user painted over, and
- **BRIA RMBG-1.4** to remove backgrounds. Its licence allows **non-commercial use
  only**; the site says so where the tool is used.

It runs as a Render **private service** (`iopaint` in [`render.yaml`](../../render.yaml)):
no public URL, reached only by the web service over the private network. The web app
([`web/app.py`](../../web/app.py)) does the upload handling, downsizing and error
translation; this service only runs the models.

## Why a custom entry point

`iopaint start` keeps LaMa and the RemoveBG model in memory together. Under a hard
2 GB limit that is killed by the kernel. [`server.py`](server.py) serves the same two
IOPaint endpoints, with IOPaint's own request schemas, model code and image helpers, but
holds one model at a time and handles one request at a time:

| Endpoint | Body | Returns |
|---|---|---|
| `POST /api/v1/inpaint` | IOPaint `InpaintRequest`: base64 `image` and `mask` (white = erase) | the filled image, same format as the input |
| `POST /api/v1/run_plugin_gen_image` | IOPaint `RunPluginRequest` with `name: "RemoveBG"` | PNG with a transparent background |
| `GET /healthz` | | `{"status": "ok", "loaded": "lama" \| "removebg" \| null}` |

Measured under a 2 GB, 1 CPU limit (Docker `--memory 2g --cpus 1`):

| | Peak memory | Time |
|---|---|---|
| Cutout (RMBG-1.4), 2048 px photo | ~1.87 GB | 10–15 s |
| Erase (LaMa), large painted area | ~1.75 GB | 10–40 s |
| Switching models | | a few seconds |
| First model load after a deploy | | ~70 s (RMBG-1.4 is preloaded in the background at startup) |

Two guards keep the peaks there: images may be at most 2048 px on the long side
(`IOPAINT_MAX_EDGE`), and when LaMa would process more than 0.75 MP the erase runs on a
downscaled copy with only the painted area pasted back. glibc is told to return freed
memory (`MALLOC_MMAP_THRESHOLD_`, `MALLOC_ARENA_MAX`, `malloc_trim` after each request).

On an instance with 4 GB or more you can drop `server.py` and run the stock server
instead (`iopaint start --model lama --device cpu --enable-remove-bg --remove-bg-model
briaai/RMBG-1.4 --host 0.0.0.0 --port 8080`); the web app calls the same endpoints.

## Image

[`Dockerfile`](Dockerfile): Python 3.11 slim, CPU-only PyTorch, IOPaint and every
transitive dependency pinned in [`constraints.txt`](constraints.txt). The model weights
are downloaded at build time by [`fetch_models.py`](fetch_models.py) and baked into the
image, and the service runs with `HF_HUB_OFFLINE=1`, so it never downloads while serving.
It runs as a non-root user. The image is about 4 GB.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` (`10000` in `render.yaml`) | Listening port. |
| `IOPAINT_PRELOAD` | `removebg` | Model to load at startup: `removebg`, `lama`, or anything else for none. |
| `IOPAINT_MAX_EDGE` | `2048` | Largest accepted long side, in pixels. |
| `REMOVE_BG_MODEL` | `briaai/RMBG-1.4` | Background-removal model IOPaint's RemoveBG plugin loads. A different model must also be fetched in `fetch_models.py`. |

## Run locally

```bash
docker build -t focal-iopaint services/iopaint
docker run --rm -p 127.0.0.1:8080:8080 --memory 2g --cpus 1 focal-iopaint
curl -s 127.0.0.1:8080/healthz
```

Then start the web app with `IOPAINT_URL=127.0.0.1:8080` (see
[`web/README.md`](../../web/README.md)).
