---
name: 焦点 Focal
description: Free online background removal, erase, and attention heatmaps; the familiar image tool, done properly.
colors:
  page-white: "#ffffff"
  cool-surface: "#f5f7fb"
  cool-surface-deep: "#eceff6"
  hairline: "#e2e6ee"
  hairline-strong: "#c9cfdb"
  ink: "#10131a"
  ink-secondary: "#454c5a"
  ink-tertiary: "#626a79"
  focal-blue: "#1f4fe0"
  focal-blue-deep: "#1a43c4"
  focal-blue-wash: "#e9effe"
  focal-blue-ink: "#1c3a99"
  focus-ring: "rgba(31, 79, 224, 0.38)"
  checker-grey: "#e7eaf1"
  success: "#157f3c"
  success-wash: "#e8f5ec"
  danger: "#c42b2b"
  danger-wash: "#fdecec"
  danger-ink: "#7d1717"
  warning: "#a35a00"
  warning-wash: "#fdf3e3"
  warning-ink: "#6b3b00"
  overlay-ink: "rgba(16, 19, 26, 0.72)"
typography:
  display:
    fontFamily: "Figtree, PingFang SC, HarmonyOS Sans SC, MiSans, Microsoft YaHei UI, Microsoft YaHei, Noto Sans SC, Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, system-ui, sans-serif"
    fontSize: "46px"
    fontWeight: 800
    lineHeight: 1.18
    letterSpacing: "-0.015em"
  headline:
    fontFamily: "Figtree, PingFang SC, HarmonyOS Sans SC, MiSans, Microsoft YaHei UI, Microsoft YaHei, Noto Sans SC, Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, system-ui, sans-serif"
    fontSize: "32px"
    fontWeight: 800
    lineHeight: 1.25
    letterSpacing: "-0.01em"
  title:
    fontFamily: "Figtree, PingFang SC, HarmonyOS Sans SC, MiSans, Microsoft YaHei UI, Microsoft YaHei, Noto Sans SC, Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, system-ui, sans-serif"
    fontSize: "30px"
    fontWeight: 800
    lineHeight: 1.25
    letterSpacing: "-0.01em"
  title-sm:
    fontFamily: "Figtree, PingFang SC, HarmonyOS Sans SC, MiSans, Microsoft YaHei UI, Microsoft YaHei, Noto Sans SC, Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, system-ui, sans-serif"
    fontSize: "22px"
    fontWeight: 800
    lineHeight: 1.25
  lead:
    fontFamily: "Figtree, PingFang SC, HarmonyOS Sans SC, MiSans, Microsoft YaHei UI, Microsoft YaHei, Noto Sans SC, Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.6
  body:
    fontFamily: "Figtree, PingFang SC, HarmonyOS Sans SC, MiSans, Microsoft YaHei UI, Microsoft YaHei, Noto Sans SC, Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.6
  button:
    fontFamily: "Figtree, PingFang SC, HarmonyOS Sans SC, MiSans, Microsoft YaHei UI, Microsoft YaHei, Noto Sans SC, Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1.2
  label:
    fontFamily: "Figtree, PingFang SC, HarmonyOS Sans SC, MiSans, Microsoft YaHei UI, Microsoft YaHei, Noto Sans SC, Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 600
    lineHeight: 1.6
  meta:
    fontFamily: "Figtree, PingFang SC, HarmonyOS Sans SC, MiSans, Microsoft YaHei UI, Microsoft YaHei, Noto Sans SC, Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, system-ui, sans-serif"
    fontSize: "13.5px"
    fontWeight: 400
    lineHeight: 1.55
  numeric:
    fontFamily: "Figtree, PingFang SC, HarmonyOS Sans SC, MiSans, Microsoft YaHei UI, Microsoft YaHei, Noto Sans SC, Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, system-ui, sans-serif"
    fontSize: "14.5px"
    fontWeight: 400
    fontFeature: "tnum"
  mono:
    fontFamily: "ui-monospace, SF Mono, JetBrains Mono, Menlo, Consolas, monospace"
    fontSize: "13.5px"
    fontWeight: 400
    lineHeight: 1.6
rounded:
  sm: "8px"
  control: "10px"
  md: "12px"
  lg: "16px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  panel: "20px"
  gutter: "24px"
  gutter-mobile: "16px"
  block: "40px"
  hero: "56px"
  section: "96px"
components:
  button-primary:
    backgroundColor: "{colors.focal-blue}"
    textColor: "{colors.page-white}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "0 18px"
    height: "44px"
  button-primary-hover:
    backgroundColor: "{colors.focal-blue-deep}"
    textColor: "{colors.page-white}"
  button-primary-disabled:
    backgroundColor: "{colors.cool-surface-deep}"
    textColor: "{colors.ink-tertiary}"
  button-primary-lg:
    backgroundColor: "{colors.focal-blue}"
    textColor: "{colors.page-white}"
    rounded: "{rounded.md}"
    padding: "0 26px"
    height: "52px"
  button-secondary:
    backgroundColor: "{colors.page-white}"
    textColor: "{colors.ink}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "0 18px"
    height: "44px"
  button-secondary-hover:
    backgroundColor: "{colors.cool-surface}"
    textColor: "{colors.ink}"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.ink-secondary}"
    rounded: "{rounded.control}"
  button-quiet-hover:
    backgroundColor: "{colors.cool-surface-deep}"
  icon-button:
    backgroundColor: "{colors.page-white}"
    textColor: "{colors.ink-secondary}"
    rounded: "{rounded.control}"
    size: "40px"
  segmented:
    backgroundColor: "{colors.cool-surface-deep}"
    rounded: "{rounded.pill}"
    padding: "4px"
  segmented-selected:
    backgroundColor: "{colors.page-white}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    height: "36px"
  sample-chip:
    backgroundColor: "{colors.page-white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "3px 12px 3px 3px"
    height: "36px"
  input:
    backgroundColor: "{colors.page-white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "9px 12px"
    height: "42px"
  stage:
    backgroundColor: "{colors.cool-surface}"
    rounded: "{rounded.lg}"
    padding: "24px"
  stage-drop-over:
    backgroundColor: "{colors.focal-blue-wash}"
  panel:
    backgroundColor: "{colors.page-white}"
    rounded: "{rounded.lg}"
    padding: "{spacing.panel}"
    width: "320px"
  alert-error:
    backgroundColor: "{colors.danger-wash}"
    textColor: "{colors.danger-ink}"
    rounded: "{rounded.md}"
    padding: "12px 14px"
  alert-info:
    backgroundColor: "{colors.focal-blue-wash}"
    textColor: "{colors.focal-blue-ink}"
    rounded: "{rounded.md}"
  alert-warning:
    backgroundColor: "{colors.warning-wash}"
    textColor: "{colors.warning-ink}"
    rounded: "{rounded.md}"
  image-tag:
    backgroundColor: "{colors.overlay-ink}"
    textColor: "{colors.page-white}"
    rounded: "{rounded.pill}"
    padding: "3px 10px"
  toast:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.page-white}"
    rounded: "{rounded.pill}"
    padding: "10px 16px"
---

# Design System: 焦点 Focal

## Overview

**Creative North Star: "The Honest Workbench"**

Focal is the familiar online image tool, played straight and finished well: a white page, cool grey working surfaces, near-black ink and one confident blue. Nothing pretends to be magic. The only pictures are real photographs and real outputs of the site's own tools, and the checkerboard is the one pattern in the system because it means something: transparency. The craft bar is remove.bg and Canva (可画), and the system meets it through restraint. Hairline borders, generous rounded containers, and a single soft lift on the comparison stage.

Two modes share one world. Home persuades: a two-column hero with a working dropzone on the left and a live before/after stage on the right, then a band of plain promises and alternating feature rows of real output pairs. The three tools (/cutout, /erase, /heatmap) operate: the same tokens arranged into one fixed grammar of stage, panel and results. Density stays comfortable rather than compact. Chinese leads, English is equal, and every layout holds both.

The system rejects "AI magic" gradients, hype styling, dark-mode theatrics, and walls of icon tiles with no real output behind them.

**Key Characteristics:**
- White ground, cool grey surfaces, near-black ink; one blue, used for action, focus, selection and the brand mark.
- Checkerboard means transparency, and is used for nothing else.
- Rounded containers (12–16px), 10px controls, pills for chips, tags and toggles.
- Hairline borders carry structure; one soft shadow lifts the comparison stage.
- Figtree for Latin, system CJK sans for Chinese, tabular numerals for every reading.
- Every state is visible: empty, working with elapsed seconds, done, error with retry.

## Colors

A cool, nearly colourless palette in which a single saturated blue does all the pointing.

### Primary
- **Focal Blue** (focal-blue): the primary action button, link text, focus rings (as focus-ring at 38% alpha), the active nav underline, the selected swatch ring, slider and checkbox accent, line icons that announce a capability (dropzone image icon, promise icons, tool-chooser icons), heatmap share bars, region boxes, the erase paint layer, and the brand mark tile.
- **Focal Blue Deep** (focal-blue-deep): hover for the primary button and links only.
- **Focal Blue Wash** (focal-blue-wash): the drag-over state of every drop target, and the info alert ground, with **Focal Blue Ink** (focal-blue-ink) as its text.

### Neutral
- **Page White** (page-white): the page, the header, the panel, secondary buttons, chips, inputs.
- **Cool Surface** (cool-surface): working surfaces: the stage, the home dropzone, the promise band, the footer, hover on white controls.
- **Cool Surface Deep** (cool-surface-deep): the image well inside compare and editor, the segmented-control track, the disabled primary button, inline code, the 3×3 attention grid cells.
- **Hairline** (hairline): borders of the panel, stage, toolbar and tables; section dividers.
- **Hairline Strong** (hairline-strong): borders on interactive controls (buttons, inputs, toggles, swatches) and the dashed edge of an empty drop target.
- **Ink** (ink): headings and body text, the toast ground.
- **Ink Secondary** (ink-secondary): lead copy, nav links at rest, quiet buttons, status text.
- **Ink Tertiary** (ink-tertiary): metadata, fineprint, table headers, elapsed seconds, disabled-primary text.
- **Checker Grey** (checker-grey): the dark square of the 20px transparency checkerboard over white.
- **Overlay Ink** (overlay-ink): pill tags laid over imagery (原图 / 处理后, A · 原版本), and at 0.84 alpha the on-stage working pill.

### Status
- **Success** (success) with **Success Wash**: done-state check icon, feature check lists, gains in A/B tables and the positive tint of the attention grid.
- **Danger** (danger) with **Danger Wash** and **Danger Ink**: the error banner, its icon, losses in A/B tables and the negative tint of the attention grid.
- **Warning** (warning) with **Warning Wash** and **Warning Ink**: non-blocking cautions (e.g. a stale heatmap after regions change).

### Named Rules
**The One Blue Rule.** Focal Blue is the only chromatic accent in the interface. A solid blue fill means "the next step"; at most one solid blue button is visible in a panel at a time. Every other blue is a line, a ring, a wash, or a data mark.

**The Checkerboard Means Transparency Rule.** The checker pattern (checker-grey on white, 20px tiles) appears only behind transparent pixels or as the "transparent" background swatch. It is never decoration.

**The Real Pixels Rule.** Imagery is real photographs or real outputs of the three tools, each with recorded provenance. No illustration, stock mood imagery, or generated art.

## Typography

**Display Font:** Figtree (Latin subset, 300–900, self-hosted, OFL) over the system CJK sans
**Body Font:** the same stack
**Label/Mono Font:** ui-monospace stack, only for JSON, CSV and code

**Character:** One stack serves everything. Figtree is limited by unicode-range to Latin, punctuation and arrows, so Latin words and numerals render in Figtree while Chinese falls through to PingFang SC, HarmonyOS Sans SC, MiSans, Microsoft YaHei UI, Noto Sans SC / Source Han Sans SC, and WenQuanYi Zen Hei. Mixed lines like "上传图片 PNG 20 MB" read as one family with a geometric Latin voice.

### Hierarchy
- **Display** (800, 46px; 42px when `lang="en"`; 34px under 768px; line-height 1.18; -0.015em): the home hero headline only. Chinese is set as two deliberate lines; English drops the break and balances.
- **Headline** (800, 32px; 26px mobile; -0.01em): home feature rows.
- **Title** (800, 30px; 26px mobile; -0.01em): the tool page heading. **Title Small** (800, 22px) heads the results block.
- **Lead** (400, 18px; 16.5px mobile; 1.6): home lead, max 34em. Tool intros use body at Ink Secondary, max 62ch.
- **Body** (400, 16px, 1.6): default text. Feature paragraphs run at 17px, max 36em.
- **Button** (600, 15px, 1.2): all buttons; 17px on large buttons, 14px on small.
- **Label** (600, 14px): field labels, segmented options, language toggle, disclosure summaries.
- **Meta** (400, 13.5px): file dimensions, captions, fineprint (13px), status hints (13px).
- **Image tag** (600, 12.5px): pill labels over imagery.

### Named Rules
**The Tabular Readings Rule.** Every number a user reads as a measurement (elapsed seconds, pixel dimensions, brush size, attention share, A/B deltas, grid cells) is set with tabular numerals.

**The Bilingual Fit Rule.** Every heading and control is sized for both languages. Where English runs longer, it gets its own size or wrapping (hero display steps down to 42px, `<br>` is hidden); it never gets truncated.

## Layout

A centred page column of 1200px max with 24px gutters (16px under 768px). The sticky header is 64px tall (56px on mobile) with a hairline bottom border; on mobile the tool nav wraps onto its own horizontally scrolling 44px row and drops its icons.

**Home** is a 5:6 two-column hero (56px gap) that stacks under 1024px, a full-bleed Cool Surface promise band of three icon-plus-phrase items, and feature rows in a 1:1 grid (64px gap, 96px between rows) that alternate media left and right and stack on mobile with media first.

**Tool pages** share one grammar:
1. A heading block: Title plus one line of Ink Secondary.
2. A workspace grid: the main column (`minmax(0, 1fr)`) holds an error slot, an optional toolbar, then the **stage**; a 320px **action panel** sits on the right, sticky below the header and scrolling inside itself when taller than the viewport. The heatmap panel grows with its region list and so scrolls with the page instead.
3. **Results** below the workspace, after a hairline rule with 40px above and 32px below it.

Under 1024px the panel drops below the stage and stops being sticky, and a working pill appears on the stage itself so progress stays visible. Under 768px the stage padding drops to 12px, the stage loses its 300px minimum once an image is loaded, and paired heatmap shots stack.

Spacing runs on a 4px base, used mostly as 8, 10, 12, 16, 20 and 24px inside components, and 40, 56, 72 and 96px between page blocks.

## Elevation & Depth

Focal is flat and bordered. Structure comes from hairlines and the step from white to Cool Surface; nothing at rest floats except the comparison stage and small controls laid over imagery.

### Shadow Vocabulary
- **Stage lift** (`0 22px 44px -22px rgba(16,19,26,0.32), 0 2px 6px rgba(16,19,26,0.06)`): the home before/after comparison stage only.
- **Pop** (`0 10px 24px -8px rgba(16,19,26,0.24), 0 1px 3px rgba(16,19,26,0.08)`): small controls floating over content: the compare knob and the toast.
- **Selected thumb** (`0 1px 2px rgba(16,19,26,0.14)`): the selected option inside a segmented control.
- **Image hairline** (`inset 0 0 0 1px rgba(16,19,26,0.1)`): an edge drawn over feature images and heatmap shots so light pictures keep their boundary on white. This is a border, not elevation.

### Named Rules
**The One Lift Rule.** One soft shadow per page, on the comparison stage. Panels, cards, toolbars and tool stages stay flat and bordered.

## Shapes

Soft, consistent rounding. Containers are 16px (stage, panel, chooser, dropzone, feature media), inner image wells and alerts 12px, controls 10px (buttons, icon buttons, selects, textareas), thumbnails and screenshot shots 8px. Large buttons step up to 12px. Pills (999px) are reserved for things that read as tokens: sample chips, the segmented control, the language toggle, image tags, the toast and the working pill. Circles are swatches, the compare knob, sample thumbnails and focal-point markers.

Borders are 1px; interactive controls take Hairline Strong, containers take Hairline. An empty drop target is the one dashed edge in the system (2px dashed Hairline Strong), and it turns solid blue with a blue wash on drag-over. Pending heatmap regions use a 2px blue dash as well, for the same reason: not yet committed.

Icons are 24px-grid line icons drawn inline as SVG symbols: 1.8px round-capped strokes at 20px (40px with a 1.5px stroke for the drop-target image icon), coloured by `currentColor`.

## Components

### Buttons
Confident and plain, sized for touch.
- **Shape:** gently rounded (10px; 12px on large).
- **Primary:** Focal Blue fill, white text, 44px tall with 18px side padding; large is 52px with 26px padding at 17px. Leading line icon where it names the action (upload, download, erase), trailing arrow on home feature CTAs.
- **Hover / Focus / Active:** fill deepens to Focal Blue Deep over 150ms; focus is a 3px focus-ring outline at 2px offset; press nudges down 1px with the ease-out curve.
- **Disabled primary:** grey, not faded. Cool Surface Deep fill, Hairline border, Ink Tertiary text, full opacity. It reads "not yet", not "faint call to action".
- **Busy primary:** stays blue, its icon replaced by a 16px spinning ring, cursor `progress`.
- **Secondary:** white with a Hairline Strong border and Ink text; hover to Cool Surface. Disabled secondaries fade to 50%.
- **Quiet:** transparent, Ink Secondary; hover to Cool Surface Deep. For reversible side actions (revert, choose another image).
- **Icon button:** 40px square (32px small), 10px radius, white with Hairline Strong border, Ink Secondary icon; carries `aria-label` and a title with its shortcut.

### Chips and toggles
- **Sample chip:** white pill, Hairline border, 28px round thumbnail of the real sample, 14px medium label; hover strengthens the border and grounds to Cool Surface.
- **Segmented control:** Cool Surface Deep pill track with 4px padding; the selected option becomes a white pill with the selected-thumb shadow. Used for the showcase tabs and heatmap mode.
- **Language toggle:** 36px pill, globe icon plus the target language label (EN or 中).
- **Swatches:** 36px circles with a Hairline Strong edge; selected gets a 2px white gap and a 2px Focal Blue ring. The first swatch is the checkerboard (transparent); the last is a rainbow conic swatch wrapping the native colour picker.

### Stage (signature)
The stage is the drop target. Empty, it is a Cool Surface field with a 2px dashed edge, minimum 420px tall (300px mobile), holding a 40px blue image icon, a large primary upload button, a drag-or-paste hint, the accepted formats, and real sample chips. Drag-over turns the edge solid blue over a blue wash. With an image loaded, the edge goes to a 1px hairline and the stage still accepts a replacement drop. While working, a white sheen sweeps across it (1.6s linear) and, on narrow layouts, a dark pill with a spinner and elapsed seconds sits at its bottom centre.

### Compare slider (signature)
Before and after stacked in a 12px-rounded well (16px with the stage lift on home), the after layer clipped at `--pos`. A 2px white divider with a faint ink edge carries a 40px white knob with the pop shadow and a horizontal drag icon. "Before" and "after" tags sit as overlay-ink pills in the top corners. It is a keyboard slider with live `aria-valuenow`; on home it sweeps on tab change, animating `--pos` over 0.9s with the ease-out curve.

### Action panel
A 320px white container, 16px radius, Hairline border, 20px padding, sections divided by hairlines with 20px between. Order: status and metadata, settings, then a primary action block (one full-width large primary, then secondary rows split 50/50), then fineprint for licences and tips. Keyboard shortcuts appear as `kbd` keys with a 2px bottom border and are hidden on touch screens. A quiet panel collapses its empty status section rather than showing a blank line.

### Status and states
- **Empty:** the stage's drop prompt; the panel status shows an info icon and one sentence of what happens next. Downloads are disabled grey.
- **Working:** an 18px spinner (Hairline Strong ring, blue arc), the state sentence, elapsed seconds in Ink Tertiary tabular numerals updating each second (hidden from screen readers so the live region speaks once), a typical-duration hint, and after a threshold a plain "slower than usual, the server may be waking" hint.
- **Done:** a green check icon and a one-line result; metadata (dimensions) appears beneath in Meta.
- **Error:** a banner in the error slot above the stage: Danger Wash ground, Danger Ink text, alert icon in Danger, 12px radius, plain-language message in the chosen language, and a small secondary "retry" button inside it. Info and warning banners share the shape with their own washes.
- **Toast:** bottom-centre Ink pill, white 14px text, pop shadow, for brief confirmations (copied, unsupported file).

### Inputs / Fields
- **Style:** white, 1px Hairline Strong, 10px radius (8px in the compact region list), 42px tall; labels in Label above, help text in 13px Ink Tertiary.
- **Hover / Focus:** border to Ink Tertiary on hover; focus swaps the outline for a blue border plus a 3px focus-ring halo.
- **Range and checkbox:** native controls tinted with `accent-color` Focal Blue.
- **Textarea:** the mono stack at 13.5px for JSON region input.

### Navigation
Sticky white header, hairline bottom. Brand left: the blue focus-mark tile (30px), 焦点 at 20px bold, Focal at 17px semibold in Ink Tertiary. Tool links right, each a line icon plus label in Ink Secondary 500; hover to Ink; the current page gets Ink text and a 2px Focal Blue underline sitting on the header's hairline. Footer is Cool Surface with a hairline top, two columns of brand note and credits.

### Results
Heading in Title Small with Meta alongside, then tables: 14.5px, hairline row rules, 13px semibold Ink Tertiary headers, right-aligned tabular numerals, gains in Success and losses in Danger, 6px blue share bars inline. The 3×3 attention grid tints cells with Success or Danger at 6–32% alpha by magnitude. Raw JSON sits in a disclosure at the end.

## Do's and Don'ts

### Do:
- **Do** keep exactly one solid Focal Blue button per panel or block: the next step.
- **Do** grey out a disabled primary (Cool Surface Deep, Ink Tertiary text) instead of fading the blue.
- **Do** make the stage the drop target and keep it a drop target after an image is loaded.
- **Do** put errors in a banner above the stage, in plain words, with a retry button.
- **Do** show elapsed seconds while working, in tabular numerals, and say plainly when the server is slow.
- **Do** use the checkerboard behind every transparent result and as the transparent swatch.
- **Do** use inline line-icon SVGs at a 1.8px stroke, coloured by `currentColor`.
- **Do** set every string for both languages and check both widths.
- **Do** honour `prefers-reduced-motion`; all transitions are 150ms colour changes or ease-out (`cubic-bezier(0.16, 1, 0.3, 1)`) movements.

### Don't:
- **Don't** add a second accent hue; status colours are for status and data only.
- **Don't** use decorative gradients, glows or "AI magic" colour washes. Gradients appear only where they carry meaning: the checkerboard, the working sheen, the colour-picker swatch.
- **Don't** put shadows on panels, cards, toolbars or tool stages; the comparison stage is the one lift.
- **Don't** use illustrations, stock photography or generated imagery; show real outputs with provenance.
- **Don't** use emoji or Unicode glyphs as icons.
- **Don't** add eyebrow labels or kickers above headings.
- **Don't** introduce a dark theme; the page is light (`color-scheme: light`). Dark appears only as translucent ink pills over imagery and the toast.
- **Don't** invent social proof, usage numbers or testimonials.
