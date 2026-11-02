// Home: a working dropzone that hands the image to a tool, and the before/after showcase.

import { onLangChange, t } from "./i18n.js";
import { $, $$, bindDropzone, compareSlider, h, isImageFile, reducedMotion, toast } from "./core.js";

const EXAMPLES = {
  cutout: {
    before: "/static/img/portrait.webp",
    after: "/static/img/portrait-cutout.webp",
    checker: true,
    focus: "50% 50%",
    rest: 50,
    alt: ["showcase.cutout_before", "showcase.cutout_after"],
  },
  erase: {
    before: "/static/img/portrait.webp",
    after: "/static/img/portrait-erased.webp",
    checker: false,
    focus: "50% 50%",
    // The divider rests across the rocket model, half of it already erased.
    rest: 78,
    alt: ["showcase.erase_before", "showcase.erase_after"],
  },
  heatmap: {
    before: "/static/img/samples/shop-a.webp",
    after: "/static/img/shop-a-heatmap.webp",
    checker: false,
    focus: "50% 50%",
    rest: 50,
    alt: ["showcase.heatmap_before", "showcase.heatmap_after"],
  },
};

let root;
let ctx;
let slider;
let example = "cutout";
let chosen = null;
let chosenUrl = null;

export function init(section, context) {
  root = section;
  ctx = context;

  const drop = $("#home-drop", root);
  const input = $("input[type=file]", drop);
  $("[data-browse]", drop).addEventListener("click", () => input.click());
  bindDropzone(drop, input, receive);

  const chooser = $("#home-chooser", root);
  for (const button of $$("[data-tool]", chooser)) {
    button.addEventListener("click", () => {
      if (chosen) ctx.navigate(button.dataset.tool, { file: chosen });
    });
  }
  $("[data-reset]", chooser).addEventListener("click", () => {
    resetChooser();
    $("[data-browse]", drop).focus();
  });

  for (const chip of $$("[data-sample]", root)) {
    chip.addEventListener("click", () => ctx.navigate(chip.dataset.sample, { sample: true }));
  }

  slider = compareSlider();
  slider.root.style.setProperty("--ar", String(4 / 3));
  $("#showcase-stage", root).append(slider.root);
  const tabs = $$("[data-example]", root);
  for (const tab of tabs) {
    tab.addEventListener("click", () => selectExample(tab.dataset.example, { reveal: true }));
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      const next = tabs[(tabs.indexOf(tab) + step + tabs.length) % tabs.length];
      next.focus();
      selectExample(next.dataset.example, { reveal: true });
    });
  }
  selectExample("cutout", { reveal: true });
  onLangChange(() => applyAlt());
}

/** An image arrived (upload, drop or paste): ask which tool to use. */
export function receive(file) {
  if (!isImageFile(file)) {
    toast(t("errors.local_file"));
    return;
  }
  chosen = file;
  if (chosenUrl) URL.revokeObjectURL(chosenUrl);
  chosenUrl = URL.createObjectURL(file);

  const chooser = $("#home-chooser", root);
  const head = $(".chooser-head", chooser);
  $("img", head)?.remove();
  head.prepend(h("img", { src: chosenUrl, alt: "", width: "64", height: "64" }));
  $(".chooser-file", chooser).textContent = `${file.name || "image"} · ${formatSize(file.size)}`;
  $("#home-drop", root).hidden = true;
  $(".hero-samples", root).hidden = true;
  chooser.hidden = false;
  $("[data-tool]", chooser).focus();
}

function resetChooser() {
  chosen = null;
  if (chosenUrl) URL.revokeObjectURL(chosenUrl);
  chosenUrl = null;
  $("#home-chooser", root).hidden = true;
  $("#home-drop", root).hidden = false;
  $(".hero-samples", root).hidden = false;
}

function selectExample(name, { reveal = false } = {}) {
  example = name;
  for (const tab of $$("[data-example]", root)) {
    const selected = tab.dataset.example === name;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  }
  const item = EXAMPLES[name];
  slider.root.style.setProperty("--focus", item.focus);
  slider.setBackground(item.checker ? "" : "transparent");
  slider.set({ beforeSrc: item.before, afterSrc: item.after });
  applyAlt();
  if (!reveal || reducedMotion()) {
    slider.position(item.rest);
    return;
  }
  // Start on the original, then sweep the divider across to reveal the result.
  slider.position(100);
  const after = $(".compare-after img", slider.root);
  const sweep = () => requestAnimationFrame(() => requestAnimationFrame(() => slider.position(item.rest, { sweep: true })));
  if (after.complete) sweep();
  else after.addEventListener("load", sweep, { once: true });
}

function applyAlt() {
  const [beforeKey, afterKey] = EXAMPLES[example].alt;
  const [before, after] = $$("img", slider.root);
  before.alt = t(beforeKey);
  after.alt = t(afterKey);
}

function formatSize(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}
