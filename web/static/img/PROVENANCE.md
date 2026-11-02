# Image provenance

Every raster on the Focal site is a real photograph or a real output of the site's own
tools. None is generated from a text prompt. Each file's origin is also embedded in the
`<name>.webp.json` sidecar next to it (`impeccable embed-prompt --read <file>`); how each
one was made is in [`scripts/focal-assets/README.md`](../../../scripts/focal-assets/README.md).

| File | Origin | Licence |
| --- | --- | --- |
| `portrait.webp` | NASA official portrait of astronaut Eileen M. Collins (JSC photo S99-00858, 1998), via images.nasa.gov | Public domain |
| `portrait-cutout.webp` | BRIA RMBG-1.4 background removal of `portrait.webp`, run through `services/iopaint` | Output of a model licensed for non-commercial use only |
| `portrait-erased.webp` | LaMa inpainting of `portrait.webp` with the shuttle model masked, run through `services/iopaint` | Public-domain source; LaMa is Apache-2.0 |
| `teapot.webp` | *Teapot with Bamboo*, Seifū Yohei IV, Cleveland Museum of Art, accession 2022.203 (Open Access) | CC0 |
| `teapot-white.webp` | RMBG-1.4 cutout of the same photo on a white background | CC0 source; RMBG-1.4 non-commercial |
| `coffee.webp` | Espresso cup photographed by Rachel Michetti, courtesy of Pikolo Espresso Bar, as distributed with scikit-image (`skimage.data.coffee`) | CC0 |
| `coffee-erased.webp` | LaMa inpainting of `coffee.webp` with the spoon masked | CC0 source; LaMa is Apache-2.0 |
| `samples/shop-a.webp`, `samples/shop-b.webp` | Screenshots of a fictional product page, `scripts/focal-assets/shop.html`, drawn for this site | Same as this repository (MIT) |
| `shop-a-heatmap.webp`, `shop-b-heatmap.webp` | HotGaze focus overlays of the two screenshots | Same as this repository (MIT) |

`favicon.svg` is the Focal mark, drawn for this site. `fonts/figtree-latin.woff2` is a Latin
subset of Figtree (SIL Open Font License 1.1, see `fonts/Figtree-OFL.txt`).
