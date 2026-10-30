#!/usr/bin/env bash
# Build step for the HotGaze web demo on Render (see render.yaml).
#
# Installs HotGaze and the web server, optionally CPU-only PyTorch for the deep
# backend, then fetches model weights and sample screenshots ahead of time so
# the running service never downloads anything while handling a request.
set -euo pipefail

python -m pip install . -r web/requirements.txt

case "${HOTGAZE_ENABLE_DEEP:-false}" in
  true | 1)
    # CPU-only wheel: the default PyPI torch wheel pulls in several GB of CUDA libraries.
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    ;;
esac

# Weights go to $HOTGAZE_CACHE, which render.yaml points inside the project
# directory so they ship with the build. A failed fetch is not fatal: HotGaze
# downloads missing weights (SHA-256 verified) on first use.
python - <<'PY' || echo "warning: could not prefetch model weights; they will be downloaded on first use"
import importlib.util

from hotgaze.weights import download_weight

names = ["yunet"]
if importlib.util.find_spec("torch") is not None:
    names += ["unisal", "mobilenet_backbone"]
for name in names:
    print(f"cached {download_weight(name, progress=False)}")
PY

# Synthetic landing pages behind the UI's "Try a sample" button.
python scripts/generate_fixtures.py
