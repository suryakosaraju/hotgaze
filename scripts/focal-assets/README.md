# Focal site imagery: sources

How the rasters in `web/static/img/` were made. Every one is a real photograph or a
real output of the site's own tools; none is generated from a text prompt. Licences are
listed in [`web/static/img/PROVENANCE.md`](../../web/static/img/PROVENANCE.md).

| Asset | Made from |
| --- | --- |
| `portrait.webp` | NASA JSC photo S99-00858 (public domain), `~large.jpg` from images.nasa.gov, cropped to 4:3 (top 70 px off, full width) and resized to 1200×900. |
| `portrait-cutout.webp` | `POST /api/cutout` with `portrait.webp`. |
| `portrait-erased.webp` | `POST /api/erase` with `portrait.webp` and `portrait-erase-mask.png` (this folder). |
| `teapot.webp`, `teapot-white.webp` | Cleveland Museum of Art Open Access 2022.203 (CC0), `_print.jpg` downscaled to 1600 px; `POST /api/cutout`; both cropped to the same square around the object and resized to 1000×1000; the white version composites the cutout onto #ffffff. |
| `samples/shop-a.webp`, `samples/shop-b.webp` | `shop.html` (this folder) captured by headless Chromium at 1280×800; B adds `class="b"` to `<body>`. The shop and product are fictional, and the page says so. |
| `shop-a-heatmap.webp`, `shop-b-heatmap.webp` | The `overlay_a` / `overlay_b` images of `POST /api/compare` for the two screenshots (fast backend, jet, alpha 0.6, focus overlay). |
| `coffee.webp`, `coffee-erased.webp` | scikit-image `data.coffee()` (CC0) and its `POST /api/erase` result with the spoon painted over. |

The heatmap sample regions in `web/static/js/heatmap.js` are the bounding boxes of the
title, price, buy button and badge elements in `shop.html`, as fractions of 1280×800.
