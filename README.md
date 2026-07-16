# HotGaze ⏿

[![CI](https://github.com/suryakosaraju/hotgaze/actions/workflows/ci.yml/badge.svg)](https://github.com/suryakosaraju/hotgaze/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Attention heatmaps for UI screenshots — with **numeric scores and A/B compare**, not just pretty pictures.

Predict where users' eyes land on a design, get a machine-readable attention share for any region, and diff two variants to see which one wins and by how much. Local, offline, MIT-licensed.

![Original design next to its HotGaze attention overlay — headline, CTA, and sidebar items glow hot](docs/demo.png)

> **Status: v0.1 alpha.** The fast heuristic backend, region scoring, A/B compare, and deep UNISAL backend are shipping. Faces layer (YuNet) is available via `--layers faces`. API and CLI may change before v1.

## Why this exists

Every existing attention-prediction tool — paid ([HeatScope](https://heatscope.space), Attention Insight, EyeQuant) and free ([Foveacast](https://www.allaboutken.com/posts/20260503-foveacast/)) — stops at a colored overlay for a human to eyeball. That's fine for a designer squinting at a mockup. It's useless when you want to:

- Compare two design variants and *quantify* which one draws more attention to the CTA.
- Wire attention checks into CI ("your button just lost 23% of its attention").
- Script or automate any part of the design-review loop.

HotGaze outputs the picture too, but the picture isn't the point. The **numbers** are.

## Install

Not on PyPI yet (v0.1 is a working-directory install):

```bash
git clone https://github.com/suryakosaraju/hotgaze
cd hotgaze
pip install -e .
```

Python 3.10+. Runs on macOS and Linux. No cloud, no API keys, no telemetry.

## Quickstart

```bash
# Generate an attention overlay
hotgaze run screenshot.png -o overlay.png

# Score a specific region — how much attention does the CTA get?
hotgaze score screenshot.png --region cta:250,200,200,35 --json

# Compare two variants — which one wins?
hotgaze compare landing_a.png landing_b.png --region cta:250,200,200,35
```

## The differentiator: numbers, not just pictures

**Score any region.** Attention share, peak value, rank — canonical JSON, deterministic on the same machine:

```bash
$ hotgaze score design.png --region cta:250,200,200,35 --json
{
  "schema": 1,
  "regions": [
    {"name": "cta", "share": 0.035, "peak_value": 0.769, "rank": 1}
  ],
  "focal_points": [...]
}
```

**Compare two variants.** Per-region deltas, plus a 3×3 spatial grid showing where attention *moved*:

```bash
$ hotgaze compare landing_a.png landing_b.png --region cta:250,200,200,35 --json
{
  "compare": {
    "per_region_deltas": [
      {"name": "cta", "share_a": 0.035, "share_b": 0.022, "delta": -0.013}
    ]
  }
}
```

Variant B lost 37% of the CTA's attention share. That's an actionable number, not "the heatmap looks about the same."

## What's in the box

- **CLI**: `hotgaze run`, `hotgaze score`, `hotgaze compare`, `hotgaze info`.
- **Fast heuristic backend** (default): spectral-residual saliency + contrast + center bias + F-pattern reading prior. No downloads, works offline, sub-second on CPU.
- **Deep saliency backend** (`--backend deep`): [UNISAL](https://github.com/rdroste/unisal) (Apache-2.0), CPU-only, deterministic per-machine. Install: `pip install hotgaze[deep]`. Weights download on first use (one-time, ~30 MB).
- **Faces layer** (`--layers faces`): [YuNet](https://github.com/opencv/opencv_zoo) (MIT) face detection adds attention blobs over detected faces.
- **Versioned JSON output**: schema v1 covers both score and compare modes so downstream tools don't break on new features.
- **Deterministic**: same image + same config + same machine → byte-identical JSON.

## Backends

| Backend | Default | What it uses | Quality |
|---------|---------|-------------|---------|
| `fast` | ✅ | Spectral residual + contrast + center bias + gaze flow | Strong on flat UI screenshots; zero downloads |
| `deep` | | UNISAL pretrained saliency model (PyTorch) | Better on natural images; on flat UIs the fast backend often matches or exceeds it — the domain gap is real. |

`--backend deep` requires `pip install hotgaze[deep]` and a one-time weight download on first use. Both backends are fully offline after the initial fetch.

## What HotGaze isn't

- **Not real eye-tracking.** It's a prediction from computer-vision priors. Useful for early design review; not a substitute for a user study.
- **Not a conversion oracle.** Attention share correlates with visibility, not with conversion — high attention on a bad CTA still doesn't sell.
- **Not a designer GUI.** It's a scriptable tool for developers. A GUI/Figma plugin is roadmap, not v1.

## Roadmap

- **v0.2** — PyPI release, GitHub Action for attention regression testing on PR screenshots.
- **v0.3+** — UI-tuned text/saliency models, Figma plugin, macOS wrapper, benchmarking against public saliency datasets.


## License

MIT. See [LICENSE](LICENSE).

Third-party models and dependencies are recorded with their licenses and redistribution status in [LICENSES-THIRD-PARTY.md](LICENSES-THIRD-PARTY.md).

## Credits

Predictive saliency stands on decades of vision research. The deep backend builds on [UNISAL](https://github.com/rdroste/unisal) (Droste et al.). The fast backend implements [Hou & Zhang's spectral-residual approach (2007)](https://www.cv-foundation.org/openaccess/content_cvpr_2007/papers/Hou_Saliency_Detection_A_Spectral_Residual_Approach_CVPR_2007_paper.pdf). Face detection uses [YuNet](https://github.com/opencv/opencv_zoo).
