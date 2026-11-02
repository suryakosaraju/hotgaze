# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Members of the public who work with images but are not machine-learning people: designers checking whether a layout pulls the eye where they intend, marketers and operators preparing campaign visuals, and e-commerce sellers cleaning up product photos. They arrive with one image (or two variants of a design) and want a usable result in about a minute, mostly in a desktop browser, sometimes on a phone. Simplified Chinese is their first language; some prefer English.

## Product Purpose

焦点 Focal is a free, public, non-commercial website that puts three open-source image tools behind one Chinese-first interface:

- **注意力热力图 · Attention heatmap** (the HotGaze library in this repository): predicts where viewers look on a UI screenshot, reports attention share and focal points per region, and compares two variants A/B.
- **自动抠图 · Background removal** (IOPaint's RemoveBG plugin with BRIA RMBG-1.4): one click to a transparent PNG.
- **涂抹消除 · Erase** (IOPaint with the LaMa model): paint over a person, object or watermark and the background is filled in.

Success means a first-time visitor understands what each tool does from the home page, gets a usable result on their own image without reading documentation, and downloads it.

## Positioning

Every tool runs on self-hosted open-source models; uploads are processed in memory and deleted after each request, with no accounts or history. The heatmap tool returns numbers (attention share per region, A/B deltas), not just a picture.

## Operating Context

- Each tool follows the same loop: upload or drag in an image, optionally mark regions (heatmap) or paint a mask (erase), run, inspect, download.
- Hosting is Render. The main site runs on the free plan (0.1 CPU, sleeps after 15 minutes without traffic, about a minute to wake). IOPaint runs as a separate private service on a 1 CPU / 2 GB instance that only the main site can reach.
- Processing takes seconds to tens of seconds. Measured on one CPU core: background removal about 8 s; erase about 13 s at 1024 px and 30 s at 2048 px. The heatmap tool is estimated at 5–15 s per screenshot on the free plan.

## Capabilities and Constraints

- Bilingual: Simplified Chinese by default and English, switchable and remembered; error messages from the API appear in the chosen language.
- Limits: heatmap images up to 10 MP and 20 MB; background removal and erase inputs are downscaled to at most 2048 px on the long edge; IOPaint handles one job at a time.
- Non-commercial use only, because BRIA RMBG-1.4 is licensed for non-commercial use; the site must say so.
- CPU-only inference. No Stable Diffusion features (replace object, outpainting, text), which need a GPU.
- Nothing is stored server-side; results live in the browser until downloaded.

## Brand Commitments

- Name: 焦点 Focal, Chinese and English shown together. The idea behind it: every tool is about the focus of an image — see where attention lands, keep the subject, remove what distracts.
- HotGaze stays the name of the underlying open-source library and is credited, as are IOPaint, LaMa and BRIA RMBG-1.4 with their licences.
- Visual direction: the category standard for online image tools, played straight, at the craft level of remove.bg and Canva (可画). The user chose it over two distinctive directions (a lightbox eye chart and a darkroom contact sheet); familiar is the commitment, executed without irony or smuggled quirks.

## Evidence on Hand

- Real outputs of all three tools can be generated on demand from the running services.
- Sample imagery, all real photos or real tool outputs (`web/static/img/PROVENANCE.md`): NASA's public-domain portrait of Eileen Collins, a CC0 teapot photo from the Cleveland Museum of Art's open access collection, the scikit-image coffee photo (CC0), and a fictional teaware product page (`scripts/focal-assets/shop.html`, labelled fictional on the page) with a promo-badge variant for A/B.
- The CLI's synthetic landing pages (`scripts/generate_fixtures.py`) remain the API and test fixtures.
- No testimonials, usage numbers, benchmarks or press exist; none may be invented.

## Product Principles

1. Result first: every tool has one obvious path from upload to download.
2. Honest about speed and limits: say when the server is waking up or a job is slow, and show errors plainly.
3. Private by default: nothing is kept, and the site says so in plain words.
4. Chinese first, English equal: every string and message exists in both languages.
5. Credit the open-source work the site stands on.
