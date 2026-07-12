# Third-Party Licenses

This file records every third-party dependency bundled or used by HotGaze,
as required by the project's license-hygiene policy (see CLAUDE.md).

## Core runtime dependencies

| Package | Version (min) | URL | License | Notes |
|---------|--------------|-----|---------|-------|
| numpy | 1.24 | https://github.com/numpy/numpy | BSD-3-Clause | Array computation |
| opencv-python-headless | 4.8 | https://github.com/opencv/opencv-python | Apache-2.0 | Image I/O, base OpenCV (no contrib) |
| pillow | 10.0 | https://github.com/python-pillow/Pillow | HPND (historical) | Image loading |
| click | 8.1 | https://github.com/pallets/click | BSD-3-Clause | CLI framework |
| pydantic | 2.0 | https://github.com/pydantic/pydantic | MIT | Config models, validation |

## Optional runtime dependencies

| Package | Version (min) | URL | License | Notes |
|---------|--------------|-----|---------|-------|
| torch | 2.0 | https://github.com/pytorch/pytorch | BSD-3-Clause | Deep saliency backend (`hotgaze[deep]`) |

## Development dependencies

| Package | Version (min) | URL | License | Notes |
|---------|--------------|-----|---------|-------|
| ruff | 0.4 | https://github.com/astral-sh/ruff | MIT | Lint + format |
| pytest | 8.0 | https://github.com/pytest-dev/pytest | MIT | Test framework |
| pytest-cov | 5.0 | https://github.com/pytest-dev/pytest-cov | MIT | Coverage reporting |
| mypy | 1.8 | https://github.com/python/mypy | MIT | Static type checking |
| scikit-image | 0.22 | https://github.com/scikit-image/scikit-image | BSD-3-Clause | **TEST-ONLY** — astronaut face fixture (BSD). Must never be imported from `src/` |
| actionlint-py | 1.7 | https://github.com/Mateusz-Grzelinski/actionlint-py | MIT | CI workflow linting |

## Pretrained models

No pretrained model weights are bundled in the repository. The deep backend
downloads weights on first use (cached in `~/.cache/hotgaze/`). Model license
audit is conducted in T0.3 and will be appended below.

### Model audit (T0.3 — completed 2026-07-11)

#### Saliency backends (deep backend candidates)

##### 1. UNISAL — ✅ SELECTED as default deep backend

| Field | Value |
|-------|-------|
| URL | https://github.com/rdroste/unisal |
| Code license | Apache License 2.0 |
| Weights license | Apache License 2.0 (weights are part of the repo, same license) |
| Weights file 1 | `weights_best.pth` (14.7 MB) — trained on SALICON |
| Weights file 2 | `weights_ft_mit1003.pth` (14.7 MB) — fine-tuned on MIT1003 |
| Hosting | GitHub repo (`raw.githubusercontent.com`) — stable URL, not Google Drive |
| SHA-256 (weights_best) | `4a9157411f1741d588b15670d15295e998805648b8b6348599fe447298338481` |
| SHA-256 (weights_ft_mit1003) | `a6de8ea27d812cfc3fbc2b8cab59862a8e48aaf4d512e670468d3b6972a81262` |
| Framework | PyTorch (native) |
| Redistribution | ✅ YES — Apache 2.0 Section 4 permits reproduction + distribution |
| **Final verdict** | **OK** — permissive license, stable URL, redistribution-allowed |

##### 2. DeepGaze IIE — ❌ REJECTED (no license)

| Field | Value |
|-------|-------|
| URL | https://github.com/matthias-k/DeepGaze |
| Code license | **None** — no LICENSE file; `setup.py` license fields commented out; defaults to "all rights reserved" |
| Weights license | None (same as code) |
| Weights URL | GitHub Releases (`v1.0.0` tag) — stable URL |
| Weights files | `deepgaze2e.pth` (~400 MB), `centerbias_mit1003.npy` (~8 MB) |
| Framework | PyTorch (native) |
| Redistribution | ❌ NO — no license grants redistribution rights |
| **Final verdict** | **needs-author-contact** — contact matthias.kuemmerer@bethgelab.org; GitHub issue #15 (commercial use) unanswered since Oct 2023 |

##### 3. MSI-Net — ✅ OK (fallback, TF → ONNX path)

| Field | Value |
|-------|-------|
| URL | https://github.com/alexanderkroner/saliency |
| Code license | MIT |
| Weights license | MIT (HuggingFace model card confirms) |
| Weights URL | https://huggingface.co/alexanderkroner/MSI-Net |
| Framework | TensorFlow (original); foveacast-training provides PyTorch port + ONNX export |
| Redistribution | ✅ YES — MIT permits use, copy, modify, merge, publish, distribute |
| **Final verdict** | **OK** — MIT-licensed, but TF dependency adds complexity; eligible only via ONNX export per CLAUDE.md |

#### Face detection candidates

##### 1. YuNet — ✅ SELECTED

| Field | Value |
|-------|-------|
| URL | https://github.com/opencv/opencv_zoo (models/face_detection_yunet) |
| Code license | MIT |
| Weights license | MIT |
| Weights URL | https://huggingface.co/opencv-zoo/face-detection-yunet (227 KB ONNX) |
| SHA-256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Integration | Native `cv2.FaceDetectorYN` API (OpenCV ≥ 4.5) |
| Redistribution | ✅ YES — MIT-licensed |
| **Final verdict** | **OK** — tiny (232 KB), native OpenCV API, MIT |

##### 2. UltraFace — ✅ OK (fallback)

| Field | Value |
|-------|-------|
| URL | https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB |
| Code license | MIT |
| Weights license | MIT |
| Weights URL | ONNX weights available (~1.2 MB) |
| Integration | `cv2.dnn.readNetFromONNX` (standard OpenCV DNN) |
| Redistribution | ✅ YES — MIT-licensed |
| **Final verdict** | **OK** — lightweight, standard DNN integration |

#### Exclusion notes

- **RetinaFace / InsightFace / SCRFD**: Code MIT, but pretrained models carry **non-commercial** restriction explicitly in README → **REJECTED**
- **DBFace**: No license specified → **REJECTED** (all rights reserved)
- **BlazeFace**: Apache 2.0 but TFLite format → excluded (no TFLite in deps)

#### Hosting plan

- UNISAL weights will be re-hosted from our own GitHub Release with pinned SHA-256 checksums (required by CLAUDE.md — original hosting is not Google Drive, but we re-host to control checksumming and availability)
- YuNet weights will be re-hosted from our own GitHub Release (232 KB, trivial)
- Download-on-first-use, cached in `~/.cache/hotgaze/` with `HOTGAZE_CACHE` env override

#### Default deep backend

**UNISAL** — best quality among PyTorch-native models with permissive license (Apache 2.0) AND redistribution rights. PyTorch-native (no TF/ONNX complexity). Two weight variants available (SALICON-trained and MIT1003-fine-tuned).

#### Vendored code

| Source | Files | License | Commit | Notes |
|--------|-------|---------|--------|-------|
| https://github.com/rdroste/unisal | `_unisal/_model.py`, `_unisal/_mobilenet.py`, `_unisal/_cgru.py` | Apache 2.0 | HEAD (2026-07-12) | Inference-only subset; training/dataset code removed. `_model.py` modified to remove training methods and KwConfigClass; `_mobilenet.py` and `_cgru.py` are unmodified except for the license header.
