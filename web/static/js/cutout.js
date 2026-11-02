// Background removal: upload → IOPaint RemoveBG (RMBG-1.4) → compare, recolour, download.

import { onLangChange, t } from "./i18n.js";
import {
  $,
  $$,
  ApiError,
  bindDropzone,
  canvasBlob,
  clearNotice,
  compareSlider,
  download,
  fetchSample,
  h,
  isImageFile,
  post,
  prepareImage,
  serverInfo,
  showNotice,
  statusLine,
  stem,
  toast,
} from "./core.js";

const SAMPLES = {
  portrait: { url: "/static/img/portrait.webp", name: "portrait.webp" },
  product: { url: "/static/img/teapot.webp", name: "teapot.webp" },
};

let root;
let els;
let status;
let slider;
const state = {
  token: 0,
  name: "",
  width: 0,
  height: 0,
  scaled: false,
  sourceUrl: null,
  result: null,
  resultUrl: null,
  background: "",
  error: null,
  retry: null,
};

export function init(section) {
  root = section;
  els = {
    stage: $(".stage", root),
    empty: $("[data-empty]", root),
    result: $("[data-result]", root),
    notices: $(".notice-slot", root),
    meta: $("[data-meta]", root),
    file: $("[data-file]", root),
    download: $("[data-download]", root),
    copy: $("[data-copy]", root),
    custom: $("[data-bg-custom]", root),
    pill: h("div", { class: "stage-status", hidden: true }),
    preview: h("img", { class: "preview", alt: "", hidden: true }),
  };
  els.stage.append(els.preview, els.pill);
  status = statusLine($("[data-status]", root), els.pill);

  for (const button of $$("[data-browse]", root)) button.addEventListener("click", () => els.file.click());
  // The whole stage takes a dropped image, empty or not.
  bindDropzone(els.stage, els.file, receive);
  for (const chip of $$("[data-sample]", root)) chip.addEventListener("click", () => sample(chip.dataset.sample));

  slider = compareSlider({ checker: true });
  els.result.append(slider.root);

  for (const swatch of $$("[data-bg]", root)) {
    swatch.addEventListener("click", () => setBackground(swatch.dataset.bg, swatch));
  }
  const picker = $("input[type=color]", els.custom);
  picker.addEventListener("input", () => setBackground(picker.value, els.custom));
  picker.addEventListener("click", () => setBackground(picker.value, els.custom));

  els.download.addEventListener("click", save);
  if ("ClipboardItem" in window && navigator.clipboard?.write) {
    els.copy.addEventListener("click", copy);
  } else {
    els.copy.hidden = true;
  }

  onLangChange(() => {
    status.refresh();
    renderMeta();
    if (state.error) showNotice(els.notices, state.error, state.retry);
  });
}

export function receive(file) {
  load(file);
}

export async function sample(kind = "portrait") {
  const { url, name } = SAMPLES[kind] ?? SAMPLES.portrait;
  try {
    load(await fetchSample(url, name));
  } catch (error) {
    fail(error, () => sample(kind));
  }
}

async function load(file) {
  if (!isImageFile(file)) {
    fail(new ApiError("local_file"));
    return;
  }
  const token = ++state.token;
  clearError();
  resetResult();
  showEmpty(false);
  status.working("status.prepare");

  const info = await serverInfo();
  let prepared;
  try {
    prepared = await prepareImage(file, info.limits.edit_max_edge);
  } catch (error) {
    if (token === state.token) fail(error, null, { empty: true });
    return;
  }
  if (token !== state.token) return;
  prepared.bitmap.close?.();
  Object.assign(state, {
    name: file.name || "image.png",
    width: prepared.width,
    height: prepared.height,
    scaled: prepared.scaled,
  });
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.sourceUrl = URL.createObjectURL(prepared.file);

  els.preview.src = state.sourceUrl;
  els.preview.style.setProperty("--ar", String(state.width / state.height));
  els.preview.hidden = false;
  renderMeta();
  run(prepared.file, token);
}

async function run(file, token) {
  clearError();
  els.stage.classList.add("stage-busy");
  els.stage.setAttribute("aria-busy", "true");
  status.working("status.cutout", { hintKey: "status.cutout_hint", slowAfter: 30 });
  const body = new FormData();
  body.append("image", file, file.name);
  try {
    const { blob } = await post("/api/cutout", body);
    if (token !== state.token) return;
    const url = URL.createObjectURL(blob);
    await decode(url);
    if (token !== state.token) {
      URL.revokeObjectURL(url);
      return;
    }
    state.result = blob;
    state.resultUrl = url;
    slider.set({
      beforeSrc: state.sourceUrl,
      afterSrc: url,
      beforeAlt: t("cutout.alt_before"),
      afterAlt: t("cutout.alt_after"),
      width: state.width,
      height: state.height,
    });
    applyBackground();
    els.preview.hidden = true;
    els.result.hidden = false;
    slider.position(50);
    status.done("status.done", { seconds: status.elapsed().toFixed(1) });
    els.download.disabled = false;
    els.copy.disabled = false;
  } catch (error) {
    if (token !== state.token) return;
    fail(error, () => run(file, token));
  } finally {
    if (token === state.token) {
      els.stage.classList.remove("stage-busy");
      els.stage.removeAttribute("aria-busy");
    }
  }
}

function decode(url) {
  const image = new Image();
  image.src = url;
  return image.decode().catch(() => {
    throw new ApiError("tool_failed", { reason: "invalid image" });
  });
}

function fail(error, retry = null, { empty = false } = {}) {
  state.error = error;
  state.retry = retry;
  // The error and its retry live above the stage; the panel stays quiet.
  status.clear();
  showNotice(els.notices, error, retry);
  if (empty) {
    els.preview.hidden = true;
    showEmpty(true);
  }
}

function showEmpty(empty) {
  els.empty.hidden = !empty;
  els.stage.classList.toggle("is-empty", empty);
}

function clearError() {
  state.error = null;
  state.retry = null;
  clearNotice(els.notices);
}

function resetResult() {
  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  state.result = null;
  state.resultUrl = null;
  els.result.hidden = true;
  els.download.disabled = true;
  els.copy.disabled = true;
}

function renderMeta() {
  if (!state.width) {
    els.meta.hidden = true;
    return;
  }
  const size = { width: state.width, height: state.height };
  els.meta.textContent = state.scaled ? t("tool.scaled", size) : t("tool.size", size);
  els.meta.hidden = false;
}

function setBackground(color, control) {
  state.background = color;
  for (const swatch of $$("[data-bg]", root)) swatch.setAttribute("aria-pressed", String(swatch === control));
  els.custom.classList.toggle("is-selected", control === els.custom);
  applyBackground();
}

function applyBackground() {
  slider.setBackground(state.background);
}

/** The result as a PNG, flattened onto the chosen colour unless it is transparent. */
async function exportBlob() {
  if (!state.background) return state.result;
  const bitmap = await createImageBitmap(state.result);
  const canvas = h("canvas", { width: bitmap.width, height: bitmap.height });
  const context = canvas.getContext("2d");
  context.fillStyle = state.background;
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(bitmap, 0, 0);
  bitmap.close?.();
  return canvasBlob(canvas, "image/png");
}

async function save() {
  if (!state.result) return;
  download(await exportBlob(), `${stem(state.name)}-cutout.png`);
}

async function copy() {
  if (!state.result) return;
  try {
    // Safari needs the ClipboardItem created inside the click, with a promise for the data.
    await navigator.clipboard.write([new ClipboardItem({ "image/png": exportBlob() })]);
    toast(t("tool.copied"));
  } catch {
    toast(t("tool.copy_failed"));
  }
}
