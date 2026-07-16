# Changelog

All notable changes to HotGaze will be documented in this file.

## v0.1.0 (2026-07-16)

Initial release.

### Features
- **Fast heuristic backend**: spectral-residual saliency + Sobel contrast + center-bias prior + F-pattern gaze flow. Zero downloads, fully offline, sub-second on CPU.
- **Deep saliency backend** (`--backend deep`): UNISAL pretrained model (Apache 2.0), CPU-only, deterministic per-machine. Requires `pip install hotgaze[deep]` and one-time weight download.
- **Region scoring**: `hotgaze score IMG --region name:x,y,w,h --json` → canonical JSON with attention share, peak value, and rank per region.
- **A/B comparison**: `hotgaze compare A.png B.png [--region ...]` → per-region attention-share deltas + 3×3 spatial grid + focal-point movement.
- **Focal points**: local maxima via OpenCV dilate-based max-filter, ranked by attention value.
- **Faces layer** (`--layers faces`): YuNet face detector (MIT) adds Gaussian attention blobs over detected faces.
- **Canonical JSON**: schema v1, sorted keys, floats rounded to 6 dp, byte-identical across runs on the same machine.
- **CLI**: `hotgaze run`, `hotgaze score`, `hotgaze compare`, `hotgaze info`.
- **Weight manager**: download-on-first-use with SHA-256 verification, atomic cache, `click.progressbar`.
- **Offline-first**: default pipeline works with zero network access; deep backend caches weights after one fetch.
