// Attention heatmap (HotGaze): score one screenshot or compare two. Regions are
// drawn on the screenshot (or typed as coordinates); the heatmap and focal points
// show on the stage, the numbers in the panel and below.

import { onLangChange, t } from "./i18n.js";
import {
  $,
  $$,
  ApiError,
  bindDropzone,
  clearNotice,
  download,
  fetchSample,
  h,
  icon,
  isImageFile,
  post,
  reducedMotion,
  serverInfo,
  showNotice,
  statusLine,
  stem,
  toast,
} from "./core.js";

const SAMPLE = {
  a: { url: "/static/img/samples/shop-a.webp", name: "shop-a.webp" },
  b: { url: "/static/img/samples/shop-b.webp", name: "shop-b.webp" },
  // Fractions of the page, measured from the sample's own layout.
  regions: [
    { key: "title", box: [0.5344, 0.1675, 0.4156, 0.0585] },
    { key: "price", box: [0.5344, 0.389, 0.4156, 0.06] },
    { key: "buy", box: [0.5344, 0.5565, 0.1719, 0.07] },
    { key: "badge", box: [0.0719, 0.1625, 0.1226, 0.0965], compareOnly: true },
  ],
};
const MIN_BOX_PX = 8;

let root;
let els;
let status;
let info = null;
const state = {
  mode: "score",
  files: { a: null, b: null },
  images: { a: null, b: null }, // {url, width, height}
  // {id, name, sampleKey?, box: [x, y, w, h] as fractions}. An empty name means the
  // label follows the language: the sample's own name, or "Region n".
  regions: [],
  nextId: 1,
  target: "a",
  busy: false,
  result: null, // {mode, body, regionsKey, settingsKey, names: Map(sent name -> region id)}
  show: "overlay",
  marks: true,
  error: null,
  retry: null,
};

export function init(section) {
  root = section;
  els = {
    stage: $("[data-stage]", root),
    empty: $("[data-empty]", root),
    shots: $("[data-shots]", root),
    notices: $(".notice-slot", root),
    file: $("[data-file]", root),
    meta: $("[data-meta]", root),
    list: $("[data-region-list]", root),
    listEmpty: $("[data-regions-empty]", root),
    typed: $("textarea[name=regions]", root),
    viewOptions: $("[data-view-options]", root),
    marks: $("input[name=marks]", root),
    backend: $("select[name=backend]", root),
    colormap: $("select[name=colormap]", root),
    alpha: $("input[name=alpha]", root),
    alphaOut: $("[data-alpha-out]", root),
    faces: $("input[name=faces]", root),
    run: $("[data-run]", root),
    runLabel: $("[data-run-label]", root),
    results: $("[data-results]", root),
    resultsBody: $("[data-results-body]", root),
    resultsMeta: $("[data-results-meta]", root),
  };
  status = statusLine($("[data-status]", root));

  for (const button of $$("[data-browse]", root)) {
    button.addEventListener("click", () => browse(button.dataset.browse));
  }
  els.file.addEventListener("change", () => {
    const file = els.file.files?.[0];
    els.file.value = "";
    if (file) setFile(state.target, file);
  });
  bindDropzone(els.stage, null, receive);
  for (const button of $$("[data-sample]", root)) button.addEventListener("click", sample);

  const tabs = $$("[data-mode]", root);
  for (const tab of tabs) {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      const next = tabs[(tabs.indexOf(tab) + step + tabs.length) % tabs.length];
      next.focus();
      setMode(next.dataset.mode);
    });
  }
  for (const button of $$("[data-show]", root)) {
    button.addEventListener("click", () => {
      state.show = button.dataset.show;
      renderStage();
    });
  }
  els.marks.addEventListener("change", () => {
    state.marks = els.marks.checked;
    renderStage();
  });
  els.alpha.addEventListener("input", () => {
    els.alphaOut.textContent = Number(els.alpha.value).toFixed(2);
  });
  for (const control of [els.backend, els.colormap, els.alpha, els.faces]) {
    control.addEventListener("change", renderStatus);
  }
  els.typed.addEventListener("input", renderStatus);
  els.run.addEventListener("click", run);
  root.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !els.run.disabled) run();
  });

  serverInfo().then((value) => {
    info = value;
    labelBackends();
  });
  onLangChange(() => {
    labelBackends();
    renderAll();
    if (state.error) showNotice(els.notices, state.error, state.retry);
  });
  renderAll();
}

/** A pasted or dropped image fills A, or B in compare mode once A is set. */
export function receive(file) {
  setFile(state.mode === "compare" && state.images.a && !state.images.b ? "b" : "a", file);
}

export async function sample() {
  if (state.busy) return;
  const compare = state.mode === "compare";
  try {
    const [a, b] = await Promise.all([
      fetchSample(SAMPLE.a.url, SAMPLE.a.name),
      compare ? fetchSample(SAMPLE.b.url, SAMPLE.b.name) : null,
    ]);
    await setFile("a", a, { quiet: true });
    if (b) await setFile("b", b, { quiet: true });
    els.typed.value = "";
    state.regions = SAMPLE.regions
      .filter((region) => compare || !region.compareOnly)
      .map((region) => ({ id: state.nextId++, name: "", sampleKey: region.key, box: [...region.box] }));
    renderAll();
    run();
  } catch (error) {
    fail(error, sample);
  }
}

// ── Files and mode ────────────────────────────────────────────────────────────

function browse(slot) {
  state.target = slot === "b" ? "b" : "a";
  els.file.click();
}

async function setFile(slot, file, { quiet = false } = {}) {
  if (!isImageFile(file)) {
    toast(t("errors.local_file"));
    return;
  }
  const limit = info?.limits?.max_upload_bytes ?? 20 * 1024 * 1024;
  if (file.size > limit) {
    fail(new ApiError("file_too_large", { name: file.name, limit_mb: Math.round(limit / 1048576) }));
    return;
  }
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.src = url;
  try {
    await image.decode();
  } catch {
    URL.revokeObjectURL(url);
    fail(new ApiError("local_file"));
    return;
  }
  if (state.images[slot]) URL.revokeObjectURL(state.images[slot].url);
  state.files[slot] = file;
  state.images[slot] = { url, width: image.naturalWidth, height: image.naturalHeight };
  state.result = null;
  clearError();
  if (!quiet) renderAll();
}

function setMode(mode) {
  if (state.busy || mode === state.mode) return;
  state.mode = mode;
  renderAll();
}

// ── Regions ───────────────────────────────────────────────────────────────────

function cleanName(name) {
  return name.replace(/[:：\n\r]/g, " ").trim().slice(0, 40);
}

/** The name shown and sent for a region: what the user typed, or a label in the current language. */
function label(region) {
  const typed = cleanName(region.name || "");
  if (typed) return typed;
  if (region.sampleKey) return t(`heatmap.sample_region.${region.sampleKey}`);
  return t("heatmap.region_default", { n: state.regions.indexOf(region) + 1 });
}

/** The display name for a region name in a result document. */
function resultLabel(name) {
  const id = currentResult()?.names?.get(name);
  const region = state.regions.find((item) => item.id === id);
  return region ? label(region) : name;
}

function addRegion(box) {
  const id = state.nextId++;
  state.regions.push({ id, name: "", box });
  renderAll();
  // Put the cursor in the new region's name so it can be renamed right away.
  const input = $(`[data-region-id="${id}"] input`, els.list);
  input?.focus();
  input?.select();
}

function removeRegion(id) {
  state.regions = state.regions.filter((region) => region.id !== id);
  renderAll();
}

const typedRegions = () => els.typed.value.split("\n").map((line) => line.trim()).filter(Boolean);

function regionStrings() {
  const drawn = state.regions.map((region) => {
    const [x, y, w, h] = region.box.map((value) => value.toFixed(4));
    return `${label(region)}:${x},${y},${w},${h}f`;
  });
  return [...drawn, ...typedRegions()];
}

const settingsKey = () => [els.backend.value, els.colormap.value, els.alpha.value, els.faces.checked].join("|");
// Independent of the UI language, so switching languages never makes a result stale.
const regionsKey = () =>
  [
    ...state.regions.map((region) => `${region.id}|${cleanName(region.name || "") || `#${region.sampleKey ?? ""}`}|${region.box}`),
    ...typedRegions(),
  ].join("\n");

/** The result on screen, if it belongs to the current mode and images. */
function currentResult() {
  return state.result && state.result.mode === state.mode ? state.result : null;
}

function resultIsFresh() {
  const result = currentResult();
  return Boolean(result) && result.regionsKey === regionsKey() && result.settingsKey === settingsKey();
}

// ── Drawing ───────────────────────────────────────────────────────────────────

function bindDrawing(frame, marks) {
  let start = null;
  let draft = null;
  const point = (event) => {
    const box = frame.getBoundingClientRect();
    return [
      Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)),
      Math.min(1, Math.max(0, (event.clientY - box.top) / box.height)),
    ];
  };
  const rect = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])];
  const place = (node, [x, y, w, hgt]) =>
    Object.assign(node.style, { left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${hgt * 100}%` });

  frame.addEventListener("pointerdown", (event) => {
    if (state.busy || (event.pointerType === "mouse" && event.button !== 0)) return;
    event.preventDefault();
    try {
      frame.setPointerCapture(event.pointerId);
    } catch {
      // The pointer is already gone; draw without capture.
    }
    start = point(event);
    draft = h("div", { class: "mark-box is-draft" });
    place(draft, [...start, 0, 0]);
    marks.append(draft);
  });
  frame.addEventListener("pointermove", (event) => {
    if (start) place(draft, rect(start, point(event)));
  });
  const finish = (event) => {
    if (!start) return;
    const box = rect(start, point(event));
    const size = frame.getBoundingClientRect();
    start = null;
    draft.remove();
    draft = null;
    if (box[2] * size.width >= MIN_BOX_PX && box[3] * size.height >= MIN_BOX_PX) addRegion(box.map((v) => +v.toFixed(4)));
  };
  frame.addEventListener("pointerup", finish);
  frame.addEventListener("pointercancel", () => {
    start = null;
    draft?.remove();
    draft = null;
  });
}

// ── Rendering ─────────────────────────────────────────────────────────────────

function renderAll() {
  renderModeControls();
  renderStage();
  renderRegionList();
  renderResults();
  renderStatus();
  renderMeta();
}

function renderModeControls() {
  for (const tab of $$("[data-mode]", root)) {
    const selected = tab.dataset.mode === state.mode;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    tab.disabled = state.busy;
  }
  els.runLabel.textContent = t(state.mode === "compare" ? "heatmap.run_compare" : "heatmap.run");
  const ready = state.images.a && (state.mode === "score" || state.images.b);
  els.run.disabled = state.busy || !ready;
  els.run.setAttribute("aria-busy", String(state.busy));
  for (const button of $$("[data-browse], [data-sample]", root)) button.disabled = state.busy;
}

function paneTitle(slot) {
  if (state.mode === "score") return t("heatmap.screenshot");
  return t(slot === "a" ? "cmp.a" : "cmp.b");
}

function resultDoc(slot) {
  const result = currentResult();
  if (!result) return null;
  if (result.mode === "score") return slot === "a" ? { doc: result.body.score, overlay: result.body.overlay } : null;
  return { doc: result.body[`score_${slot}`], overlay: result.body[`overlay_${slot}`] };
}

function renderStage() {
  const slots = state.mode === "compare" ? ["a", "b"] : ["a"];
  const empty = !state.images.a && !(state.mode === "compare" && state.images.b);
  els.empty.hidden = !empty;
  els.shots.hidden = empty;
  els.stage.classList.toggle("is-empty", empty);
  els.stage.classList.toggle("stage-busy", state.busy);
  if (state.busy) els.stage.setAttribute("aria-busy", "true");
  else els.stage.removeAttribute("aria-busy");
  // Landscape screenshots stack, each at the stage's full width; portrait ones sit side by side.
  const portrait = slots.every((slot) => !state.images[slot] || state.images[slot].height > state.images[slot].width);
  els.shots.classList.toggle("is-pair", state.mode === "compare" && portrait);
  if (empty) {
    els.shots.replaceChildren();
    return;
  }
  els.shots.replaceChildren(...slots.map(renderPane));
  els.viewOptions.hidden = !currentResult();
  for (const button of $$("[data-show]", root)) button.setAttribute("aria-pressed", String(button.dataset.show === state.show));
  els.marks.checked = state.marks;
}

function renderPane(slot) {
  const image = state.images[slot];
  const head = h("figcaption", { class: "shot-head" }, h("span", { class: "shot-title", text: paneTitle(slot) }));
  if (!image) {
    return h(
      "figure",
      { class: "shot shot-missing", "data-pane": slot },
      head,
      h(
        "div",
        { class: "shot-placeholder" },
        h("div", { class: "drop-prompt" }, h(
          "button",
          { type: "button", class: "btn btn-primary", onclick: () => browse(slot), disabled: state.busy },
          icon("upload"),
          h("span", { text: t(slot === "b" ? "heatmap.upload_b" : "heatmap.upload") }),
        ), h("p", { class: "dropzone-hint", text: t("heatmap.drop_hint") })),
      ),
    );
  }
  const found = resultDoc(slot);
  head.append(h("span", { class: "meta num", text: `${state.files[slot]?.name ?? ""} · ${image.width}×${image.height}` }));
  if (found) {
    head.append(
      h(
        "button",
        {
          type: "button",
          class: "icon-btn icon-btn-sm",
          title: t("res.download_overlay"),
          "aria-label": `${t("res.download_overlay")} (${paneTitle(slot)})`,
          onclick: () => download(found.overlay, `${stem(state.files[slot]?.name)}_heatmap.jpg`),
        },
        icon("download"),
      ),
    );
  }
  const src = found && state.show === "overlay" ? found.overlay : image.url;
  const picture = h("img", {
    src,
    alt: found && state.show === "overlay" ? `${paneTitle(slot)}: ${t("res.overlay")}` : paneTitle(slot),
    width: image.width,
    height: image.height,
    draggable: "false",
  });
  const marks = h("div", { class: "marks" });
  if (state.marks) marks.append(...renderMarks(slot, image, found));
  const frame = h("div", { class: "shot-frame", "data-frame": slot, "aria-label": t("heatmap.draw_label") }, picture, marks);
  bindDrawing(frame, marks);
  return h("figure", { class: "shot", "data-pane": slot }, head, h("div", { class: "shot-body" }, frame));
}

function renderMarks(slot, image, found) {
  const nodes = [];
  const fresh = resultIsFresh();
  const place = (x, y, w, hgt) => ({
    left: `${x * 100}%`,
    top: `${y * 100}%`,
    width: `${w * 100}%`,
    height: `${hgt * 100}%`,
  });
  if (found && fresh) {
    const { width, height } = found.doc.image.size;
    for (const region of found.doc.regions) {
      const id = currentResult()?.names?.get(region.name);
      nodes.push(
        h(
          "div",
          {
            class: "mark-box",
            "data-region-id": id === undefined ? null : String(id),
            style: place(region.box.x / width, region.box.y / height, region.box.w / width, region.box.h / height),
          },
          h("span", {
            class: region.box.y < height * 0.05 ? "mark-tag inside" : "mark-tag",
            text: `${resultLabel(region.name)} · ${pct(region.share)}`,
          }),
        ),
      );
    }
  } else {
    for (const region of state.regions) {
      const [x, y, w, hgt] = region.box;
      nodes.push(
        h(
          "div",
          { class: "mark-box is-pending", "data-region-id": String(region.id), style: place(x, y, w, hgt) },
          h("span", { class: y < 0.05 ? "mark-tag inside" : "mark-tag", text: label(region) }),
        ),
      );
    }
  }
  if (found) {
    const { width, height } = found.doc.image.size;
    for (const point of found.doc.focal_points) {
      nodes.push(
        h("div", {
          class: "mark-focal",
          style: { left: `${(point.x / width) * 100}%`, top: `${(point.y / height) * 100}%` },
          title: t("res.focal_title", { rank: point.rank, value: point.value.toFixed(3) }),
          text: String(point.rank),
        }),
      );
    }
  }
  return nodes;
}

function regionNumbers() {
  const result = currentResult();
  if (!result || !resultIsFresh()) return new Map();
  const numbers = new Map();
  if (result.mode === "score") {
    for (const region of result.body.score.regions) numbers.set(result.names.get(region.name), { text: pct(region.share) });
  } else {
    for (const row of result.body.compare.compare.per_region_deltas) {
      numbers.set(result.names.get(row.name), { text: pp(row.delta), cls: trend(row.delta) });
    }
  }
  return numbers;
}

function renderRegionList() {
  const numbers = regionNumbers();
  els.list.hidden = !state.regions.length;
  els.listEmpty.hidden = Boolean(state.regions.length);
  els.list.replaceChildren(
    ...state.regions.map((region, index) => {
      const input = h("input", {
        type: "text",
        value: label(region),
        maxlength: "40",
        "aria-label": t("heatmap.region_name", { n: index + 1 }),
        disabled: state.busy,
      });
      input.addEventListener("input", () => {
        region.name = input.value;
        renderStatus();
      });
      input.addEventListener("change", () => {
        // Clearing the field goes back to the automatic, language-following name.
        region.name = cleanName(input.value);
        renderAll();
      });
      const number = numbers.get(region.id);
      const row = h(
        "li",
        { "data-region-id": String(region.id) },
        h("span", { class: "region-index num", text: String(index + 1) }),
        input,
        h(
          "button",
          {
            type: "button",
            class: "icon-btn icon-btn-sm",
            "aria-label": t("heatmap.region_remove", { name: label(region) }),
            title: t("heatmap.region_remove", { name: label(region) }),
            disabled: state.busy,
            onclick: () => removeRegion(region.id),
          },
          icon("trash"),
        ),
        number ? h("span", { class: `region-number num ${number.cls ?? ""}`, text: number.text }) : null,
      );
      row.addEventListener("pointerenter", () => highlight(region.id));
      row.addEventListener("pointerleave", () => highlight(null));
      row.addEventListener("focusin", () => highlight(region.id));
      row.addEventListener("focusout", () => highlight(null));
      return row;
    }),
  );
}

/** Outline a region's box on the stage while its row in the list is hovered or focused. */
function highlight(id) {
  for (const box of $$(".mark-box[data-region-id]", els.shots)) {
    box.classList.toggle("is-hot", id !== null && box.dataset.regionId === String(id));
  }
}

function renderStatus() {
  if (state.busy) return;
  if (state.error) {
    status.clear();
    return;
  }
  const result = currentResult();
  if (result && !resultIsFresh()) {
    status.info("heatmap.stale");
  } else if (result) {
    status.done("heatmap.done", { seconds: result.body.elapsed_seconds.toFixed(1) });
  } else if (!state.images.a) {
    status.info("heatmap.idle");
  } else if (state.mode === "compare" && !state.images.b) {
    status.info("heatmap.need_b");
  } else {
    status.info("heatmap.ready");
  }
}

function renderMeta() {
  const result = currentResult();
  if (!result) {
    els.meta.hidden = true;
    return;
  }
  els.meta.textContent = runMeta(result.mode === "score" ? result.body.score : result.body.score_a, result.body.elapsed_seconds);
  els.meta.hidden = false;
}

function labelBackends() {
  const deep = $('option[value="deep"]', els.backend);
  const available = info?.backends?.includes("deep") ?? true;
  deep.disabled = !available;
  deep.textContent = t("heatmap.backend_deep") + (available ? "" : t("heatmap.backend_missing"));
}

// ── Run ───────────────────────────────────────────────────────────────────────

async function run() {
  if (state.busy) return;
  const compare = state.mode === "compare";
  if (!state.files.a || (compare && !state.files.b)) {
    status.info(compare ? "heatmap.need_ab" : "heatmap.need_a");
    return;
  }
  const data = new FormData();
  if (compare) {
    data.append("image_a", state.files.a);
    data.append("image_b", state.files.b);
  } else {
    data.append("image", state.files.a);
  }
  for (const region of regionStrings()) data.append("region", region);
  const names = new Map(state.regions.map((region) => [label(region), region.id]));
  data.append("backend", els.backend.value);
  data.append("colormap", els.colormap.value);
  data.append("alpha", els.alpha.value);
  data.append("faces", els.faces.checked ? "true" : "false");
  data.append("overlay", "focus");
  const keys = { regionsKey: regionsKey(), settingsKey: settingsKey(), names };

  clearError();
  state.busy = true;
  renderAll();
  status.working("heatmap.working", { slowAfter: 6, slowKey: "heatmap.slow" });
  try {
    const body = await post(`/api/${state.mode}`, data);
    state.result = { mode: compare ? "compare" : "score", body, ...keys };
    state.show = "overlay";
  } catch (error) {
    state.error = error;
    state.retry = run;
  } finally {
    state.busy = false;
    renderAll();
    if (state.error) showNotice(els.notices, state.error, state.retry);
  }
  if (state.result && !state.error && !reducedMotion()) {
    // Bring the heatmap into view on small screens, where the panel sits below it.
    if (els.stage.getBoundingClientRect().top < 0) els.stage.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function fail(error, retry = null) {
  state.error = error;
  state.retry = retry;
  showNotice(els.notices, error, retry);
  renderStatus();
}

function clearError() {
  state.error = null;
  state.retry = null;
  clearNotice(els.notices);
}

// ── Results below the workspace ──────────────────────────────────────────────

const pct = (share) => `${(share * 100).toFixed(2)}%`;
const signed = (value, digits, unit) =>
  `${value > 0 ? "+" : value < 0 ? "−" : "±"}${Math.abs(value).toFixed(digits)}${unit}`;
const pp = (delta) => signed(delta * 100, 2, t("cmp.pp"));
const trend = (delta) => (delta > 1e-6 ? "gain" : delta < -1e-6 ? "loss" : "");

function runMeta(doc, elapsed) {
  const work = doc.image.working_size;
  const backend = t(doc.config.backend === "deep" ? "heatmap.backend_deep" : "heatmap.backend_fast");
  return t("res.meta", {
    backend,
    layers: doc.config.extra_layers.length ? t("res.layers_faces") : "",
    width: work.width,
    height: work.height,
    seconds: elapsed.toFixed(1),
  });
}

function renderResults() {
  const result = currentResult();
  els.results.hidden = !result;
  if (!result) {
    els.resultsBody.replaceChildren();
    return;
  }
  const { body } = result;
  if (result.mode === "score") {
    els.resultsMeta.textContent = body.score.image.path;
    els.resultsBody.replaceChildren(
      h("div", { class: "tables" }, regionTable(body.score), focalTable(body.score)),
      developer(body.score, `${stem(body.score.image.path)}_score.json`, "hotgaze score --json"),
    );
    return;
  }
  const compare = body.compare.compare;
  const hasRegions = compare.per_region_deltas.length > 0;
  els.resultsMeta.textContent = `${body.score_a.image.path} → ${body.score_b.image.path}`;
  els.resultsBody.replaceChildren(
    hasRegions
      ? h("div", { class: "tables" }, deltaTable(compare))
      : h("div", { class: "tables" }, gridView(compare.grid_deltas), movementTable(compare)),
    h("p", { class: "help", text: t(hasRegions ? "cmp.hint_regions" : "cmp.hint_grid") }),
    developer(body.compare, "compare.json", "hotgaze compare --json"),
  );
}

function table(caption, headers, rows, empty) {
  if (!rows.length) {
    return h("div", { class: "table-empty" }, h("h3", { text: caption }), h("p", { class: "help", text: empty }));
  }
  return h(
    "div",
    { class: "table-wrap" },
    h(
      "table",
      {},
      h("caption", { text: caption }),
      h("thead", {}, h("tr", {}, headers.map(([label, cls]) => h("th", { class: cls, scope: "col", text: label })))),
      h("tbody", {}, rows.map((cells) => h("tr", {}, cells))),
    ),
  );
}

function regionTable(doc) {
  const top = Math.max(...doc.regions.map((region) => region.share), 1e-9);
  return table(
    t("res.regions"),
    [[t("res.rank"), "n"], [t("res.region")], [t("res.share"), "n"], [t("res.peak"), "n"], [t("res.box"), "n col-box"]],
    doc.regions.map((region) => [
      h("td", { class: "n", text: String(region.rank) }),
      h("td", { class: "name", text: resultLabel(region.name) }),
      h(
        "td",
        { class: "n" },
        pct(region.share),
        h("span", { class: "share-bar", style: { width: `${Math.max(2, (region.share / top) * 60)}px` } }),
      ),
      h("td", { class: "n", text: region.peak_value.toFixed(3) }),
      h("td", { class: "n col-box", text: `${region.box.x}, ${region.box.y}, ${region.box.w}, ${region.box.h}` }),
    ]),
    t("res.no_regions"),
  );
}

function focalTable(doc) {
  return table(
    t("res.focal"),
    [[t("res.rank"), "n"], ["x", "n"], ["y", "n"], [t("res.value"), "n"]],
    doc.focal_points.map((point) => [
      h("td", { class: "n", text: String(point.rank) }),
      h("td", { class: "n", text: String(point.x) }),
      h("td", { class: "n", text: String(point.y) }),
      h("td", { class: "n", text: point.value.toFixed(3) }),
    ]),
    t("res.none"),
  );
}

function deltaTable(compare) {
  return table(
    t("cmp.by_region"),
    [[t("res.region")], ["A", "n"], ["B", "n"], [t("cmp.delta"), "n"], [t("cmp.relative"), "n"]],
    compare.per_region_deltas.map((row) => {
      const relative = row.share_a > 0 ? signed((row.delta / row.share_a) * 100, 0, "%") : "—";
      return [
        h("td", { class: "name", text: resultLabel(row.name) }),
        h("td", { class: "n", text: pct(row.share_a) }),
        h("td", { class: "n", text: pct(row.share_b) }),
        h("td", { class: `n ${trend(row.delta)}`, text: pp(row.delta) }),
        h("td", { class: `n ${trend(row.delta)}`, text: relative }),
      ];
    }),
    t("res.none"),
  );
}

function gridView(grid) {
  const peak = Math.max(...grid.map(Math.abs), 1e-9);
  const shade = (value) => {
    const rgb = value >= 0 ? "21, 127, 60" : "196, 43, 43";
    return `rgba(${rgb}, ${(0.06 + (Math.abs(value) / peak) * 0.26).toFixed(3)})`;
  };
  return h(
    "div",
    {},
    h("h3", { text: t("cmp.grid") }),
    h("p", { class: "help", text: t("cmp.grid_help") }),
    h(
      "div",
      { class: "grid9", role: "img", "aria-label": t("cmp.grid") },
      grid.map((value) => h("div", { class: "num", style: { background: shade(value) }, text: pp(value) })),
    ),
  );
}

function movementTable(compare) {
  const xy = (x, y) => (x < 0 ? "—" : `${x}, ${y}`);
  return table(
    t("cmp.movement"),
    [[t("res.rank"), "n"], ["A (x, y)", "n"], ["B (x, y)", "n"], [t("cmp.moved"), "n"]],
    compare.focal_point_movement.map((row) => [
      h("td", { class: "n", text: String(row.rank) }),
      h("td", { class: "n", text: xy(row.x_a, row.y_a) }),
      h("td", { class: "n", text: xy(row.x_b, row.y_b) }),
      h("td", { class: "n", text: row.distance < 0 ? "—" : `${row.distance.toFixed(0)} px` }),
    ]),
    t("res.none"),
  );
}

/** The raw schema-v1 document, for people automating this; closed by default. */
function developer(doc, filename, command) {
  const text = JSON.stringify(doc, null, 2);
  const copy = h("button", { type: "button", class: "btn btn-sm", text: t("res.copy") });
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      copy.textContent = t("res.copied");
      setTimeout(() => {
        copy.textContent = t("res.copy");
      }, 1500);
    } catch {
      toast(t("tool.copy_failed"));
    }
  });
  const save = h("button", {
    type: "button",
    class: "btn btn-sm btn-quiet",
    text: t("res.download_json", { name: filename }),
    onclick: () => download(new Blob([`${text}\n`], { type: "application/json" }), filename),
  });
  return h(
    "details",
    { class: "dev" },
    h("summary", { text: t("res.developer") }),
    h("p", { class: "help" }, t("res.developer_help", { command })),
    h("div", { class: "json-tools" }, copy, save),
    h("pre", { text }),
  );
}
