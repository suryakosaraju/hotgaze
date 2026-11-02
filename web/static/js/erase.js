// Erase: paint over what to remove, IOPaint's LaMa fills it in. Erase again,
// undo the last erase, hold to compare with the original, download.

import { onLangChange, t } from "./i18n.js";
import {
  $,
  $$,
  ApiError,
  bindDropzone,
  canvasBlob,
  clearNotice,
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

const SAMPLE = {
  url: "/static/img/coffee.webp",
  name: "coffee.webp",
  // The spoon on the saucer, in image pixels, so the sample erases in one click.
  strokes: [
    {
      size: 30,
      points: [
        [412, 66],
        [415, 98],
        [408, 132],
        [396, 170],
        [384, 206],
        [372, 238],
      ],
    },
    {
      size: 92,
      points: [
        [367, 280],
        [367, 286],
      ],
    },
  ],
};
const PAINT = "#1f4fe0";
const BRUSH = { min: 8, max: 160, step: 8 };

let root;
let ctx;
let els;
let status;
let base;
let paint;
let active = null;
const state = {
  token: 0,
  name: "",
  versions: [], // {blob, bitmap}; versions[0] is the image as uploaded
  strokes: [],
  past: [],
  future: [],
  brush: 48,
  busy: false,
  comparing: false,
  error: null,
  retry: null,
};

export function init(section, context) {
  root = section;
  ctx = context;
  els = {
    stage: $(".stage", root),
    empty: $("[data-empty]", root),
    editor: $("[data-editor]", root),
    base: $("[data-base]", root),
    paint: $("[data-paint]", root),
    cursor: $("[data-cursor]", root),
    toolbar: $("[data-toolbar]", root),
    notices: $(".notice-slot", root),
    meta: $("[data-meta]", root),
    file: $("[data-file]", root),
    brush: $("[data-brush]", root),
    brushOut: $("[data-brush-out]", root),
    undo: $("[data-undo]", root),
    redo: $("[data-redo]", root),
    clear: $("[data-clear]", root),
    run: $("[data-run]", root),
    compare: $("[data-compare]", root),
    revert: $("[data-revert]", root),
    download: $("[data-download]", root),
    toCutout: $("[data-to-cutout]", root),
    pill: h("div", { class: "stage-status", hidden: true }),
  };
  els.stage.append(els.pill);
  status = statusLine($("[data-status]", root), els.pill);
  base = els.base.getContext("2d");
  paint = els.paint.getContext("2d");

  for (const button of $$("[data-browse]", root)) button.addEventListener("click", () => els.file.click());
  // The whole stage takes a dropped image, empty or not.
  bindDropzone(els.stage, els.file, receive);
  $("[data-sample]", root).addEventListener("click", sample);

  els.brush.addEventListener("input", () => setBrush(Number(els.brush.value)));
  els.undo.addEventListener("click", undo);
  els.redo.addEventListener("click", redo);
  els.clear.addEventListener("click", clearStrokes);
  els.run.addEventListener("click", run);
  els.revert.addEventListener("click", revert);
  els.download.addEventListener("click", save);
  els.toCutout.addEventListener("click", toCutout);

  // Hold to compare with the original, by pointer or by keyboard.
  els.compare.addEventListener("pointerdown", startCompare);
  for (const type of ["pointerup", "pointerleave", "pointercancel", "blur"]) {
    els.compare.addEventListener(type, endCompare);
  }
  els.compare.addEventListener("keydown", (event) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) {
      event.preventDefault();
      startCompare();
    }
  });
  els.compare.addEventListener("keyup", (event) => {
    if (event.key === " " || event.key === "Enter") endCompare();
  });

  els.paint.addEventListener("pointerdown", strokeStart);
  els.paint.addEventListener("pointermove", strokeMove);
  els.paint.addEventListener("pointerup", strokeEnd);
  els.paint.addEventListener("pointercancel", strokeEnd);
  els.paint.addEventListener("pointerenter", moveCursor);
  els.paint.addEventListener("pointerleave", () => {
    els.cursor.hidden = true;
  });

  document.addEventListener("keydown", shortcuts);
  onLangChange(() => {
    status.refresh();
    renderMeta();
    if (state.error) showNotice(els.notices, state.error, state.retry);
  });
  setBrush(state.brush);
  updateControls();
}

export function receive(file) {
  load(file);
}

export async function sample() {
  try {
    load(await fetchSample(SAMPLE.url, SAMPLE.name), { strokes: SAMPLE.strokes });
  } catch (error) {
    fail(error, sample);
  }
}

// ── Loading ───────────────────────────────────────────────────────────────────

async function load(file, { strokes = [] } = {}) {
  if (!isImageFile(file)) {
    fail(new ApiError("local_file"));
    return;
  }
  const token = ++state.token;
  clearError();
  status.working("status.prepare");
  const info = await serverInfo();
  let prepared;
  try {
    prepared = await prepareImage(file, info.limits.edit_max_edge);
  } catch (error) {
    if (token === state.token) fail(error);
    return;
  }
  if (token !== state.token) {
    prepared.bitmap.close?.();
    return;
  }
  for (const version of state.versions) version.bitmap.close?.();
  state.name = file.name || "image.png";
  state.versions = [{ blob: prepared.file, bitmap: prepared.bitmap }];
  state.strokes = strokes.map((stroke) => ({ size: stroke.size, points: stroke.points.map((point) => [...point]) }));
  state.past = [];
  state.future = [];
  state.busy = false;
  resize(prepared.width, prepared.height);
  els.empty.hidden = true;
  els.stage.classList.remove("is-empty");
  els.editor.hidden = false;
  els.toolbar.hidden = false;
  drawBase();
  redrawPaint();
  status.info(strokes.length ? "erase.sample_ready" : "erase.ready");
  renderMeta();
  updateControls();
}

function resize(width, height) {
  for (const canvas of [els.base, els.paint]) {
    canvas.width = width;
    canvas.height = height;
  }
  els.editor.style.setProperty("--ar", String(width / height));
}

const current = () => state.versions.at(-1);

// ── Painting ──────────────────────────────────────────────────────────────────

function toImage(event) {
  const box = els.paint.getBoundingClientRect();
  return [
    ((event.clientX - box.left) * els.paint.width) / box.width,
    ((event.clientY - box.top) * els.paint.height) / box.height,
  ];
}

function strokeStart(event) {
  if (state.busy || state.comparing || !state.versions.length) return;
  if (event.pointerType === "mouse" && event.button !== 0) return;
  event.preventDefault();
  try {
    els.paint.setPointerCapture(event.pointerId);
  } catch {
    // The pointer is already gone (or was synthesized); paint without capture.
  }
  const scale = els.paint.width / els.paint.getBoundingClientRect().width;
  active = { size: Math.max(2, Math.round(state.brush * scale)), points: [toImage(event)] };
  dot(paint, active.points[0], active.size, PAINT);
  moveCursor(event);
}

function strokeMove(event) {
  moveCursor(event);
  if (!active) return;
  for (const sample of event.getCoalescedEvents?.() ?? [event]) {
    const point = toImage(sample);
    const last = active.points.at(-1);
    if (Math.hypot(point[0] - last[0], point[1] - last[1]) < 1) continue;
    active.points.push(point);
    segment(paint, last, point, active.size, PAINT);
  }
}

function strokeEnd() {
  if (!active) return;
  commit([...state.strokes, active]);
  active = null;
}

function moveCursor(event) {
  if (event.pointerType === "touch" || state.busy || state.comparing) {
    els.cursor.hidden = true;
    return;
  }
  const box = els.editor.getBoundingClientRect();
  els.cursor.style.left = `${event.clientX - box.left}px`;
  els.cursor.style.top = `${event.clientY - box.top}px`;
  els.cursor.hidden = false;
}

function setBrush(size) {
  state.brush = Math.min(BRUSH.max, Math.max(BRUSH.min, Math.round(size)));
  els.brush.value = String(state.brush);
  els.brushOut.textContent = String(state.brush);
  els.cursor.style.width = `${state.brush}px`;
  els.cursor.style.height = `${state.brush}px`;
}

function style(context, size, color) {
  context.lineCap = "round";
  context.lineJoin = "round";
  context.lineWidth = size;
  context.strokeStyle = color;
  context.fillStyle = color;
}

function dot(context, [x, y], size, color) {
  style(context, size, color);
  context.beginPath();
  context.arc(x, y, size / 2, 0, Math.PI * 2);
  context.fill();
}

function segment(context, from, to, size, color) {
  style(context, size, color);
  context.beginPath();
  context.moveTo(...from);
  context.lineTo(...to);
  context.stroke();
}

function drawStroke(context, stroke, color) {
  const [first, ...rest] = stroke.points;
  if (!rest.length) {
    dot(context, first, stroke.size, color);
    return;
  }
  style(context, stroke.size, color);
  context.beginPath();
  context.moveTo(...first);
  for (const point of rest) context.lineTo(...point);
  context.stroke();
}

function redrawPaint() {
  paint.clearRect(0, 0, els.paint.width, els.paint.height);
  for (const stroke of state.strokes) drawStroke(paint, stroke, PAINT);
}

function drawBase() {
  const version = state.comparing ? state.versions[0] : current();
  if (!version) return;
  base.clearRect(0, 0, els.base.width, els.base.height);
  base.drawImage(version.bitmap, 0, 0, els.base.width, els.base.height);
}

// ── History ───────────────────────────────────────────────────────────────────

function commit(strokes) {
  state.past.push(state.strokes);
  state.strokes = strokes;
  state.future = [];
  updateControls();
}

function undo() {
  if (state.busy || !state.past.length) return;
  state.future.push(state.strokes);
  state.strokes = state.past.pop();
  redrawPaint();
  updateControls();
}

function redo() {
  if (state.busy || !state.future.length) return;
  state.past.push(state.strokes);
  state.strokes = state.future.pop();
  redrawPaint();
  updateControls();
}

function clearStrokes() {
  if (state.busy || !state.strokes.length) return;
  commit([]);
  redrawPaint();
}

// ── Erase ─────────────────────────────────────────────────────────────────────

async function maskBlob() {
  const canvas = h("canvas", { width: els.paint.width, height: els.paint.height });
  const context = canvas.getContext("2d");
  context.fillStyle = "#000";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (const stroke of state.strokes) drawStroke(context, stroke, "#fff");
  return canvasBlob(canvas, "image/png");
}

async function run() {
  if (state.busy || !state.versions.length) return;
  if (!state.strokes.length) {
    status.info("erase.paint_first");
    toast(t("erase.paint_first"));
    return;
  }
  const token = state.token;
  const source = current();
  const mask = await maskBlob();
  clearError();
  setBusy(true);
  status.working("status.erase", { hintKey: "status.erase_hint", slowAfter: 60 });
  const body = new FormData();
  body.append("image", source.blob, `${stem(state.name)}.${extension(source.blob)}`);
  body.append("mask", mask, "mask.png");
  try {
    const { blob } = await post("/api/erase", body);
    if (token !== state.token) return;
    const bitmap = await createImageBitmap(blob).catch(() => {
      throw new ApiError("tool_failed", { reason: "invalid image" });
    });
    if (token !== state.token) {
      bitmap.close?.();
      return;
    }
    state.versions.push({ blob, bitmap });
    if (bitmap.width !== els.base.width || bitmap.height !== els.base.height) resize(bitmap.width, bitmap.height);
    state.strokes = [];
    state.past = [];
    state.future = [];
    drawBase();
    redrawPaint();
    status.done("status.done", { seconds: status.elapsed().toFixed(1) });
    renderMeta();
  } catch (error) {
    if (token === state.token) fail(error, run);
  } finally {
    if (token === state.token) setBusy(false);
  }
}

function revert() {
  if (state.busy || state.versions.length < 2) return;
  state.versions.pop().bitmap.close?.();
  const { bitmap } = current();
  if (bitmap.width !== els.base.width || bitmap.height !== els.base.height) {
    resize(bitmap.width, bitmap.height);
    redrawPaint();
  }
  drawBase();
  status.info("erase.reverted");
  renderMeta();
  updateControls();
}

function startCompare() {
  if (state.busy || state.versions.length < 2 || state.comparing) return;
  state.comparing = true;
  els.compare.setAttribute("aria-pressed", "true");
  els.paint.style.visibility = "hidden";
  els.cursor.hidden = true;
  drawBase();
}

function endCompare() {
  if (!state.comparing) return;
  state.comparing = false;
  els.compare.setAttribute("aria-pressed", "false");
  els.paint.style.visibility = "";
  drawBase();
}

function setBusy(busy) {
  state.busy = busy;
  els.stage.classList.toggle("stage-busy", busy);
  if (busy) els.stage.setAttribute("aria-busy", "true");
  else els.stage.removeAttribute("aria-busy");
  els.run.setAttribute("aria-busy", String(busy));
  if (busy) els.cursor.hidden = true;
  updateControls();
}

function updateControls() {
  const ready = state.versions.length > 0 && !state.busy;
  const edited = state.versions.length > 1;
  els.run.disabled = !ready;
  els.undo.disabled = !ready || !state.past.length;
  els.redo.disabled = !ready || !state.future.length;
  els.clear.disabled = !ready || !state.strokes.length;
  els.revert.disabled = !ready || !edited;
  els.compare.disabled = !ready || !edited;
  els.download.disabled = !ready || !edited;
  els.toCutout.disabled = !ready || !edited;
  els.brush.disabled = !ready;
  for (const button of $$("[data-browse]", root)) button.disabled = state.busy;
}

function renderMeta() {
  const version = current();
  if (!version) {
    els.meta.hidden = true;
    return;
  }
  const size = t("tool.size", { width: version.bitmap.width, height: version.bitmap.height });
  const count = state.versions.length - 1;
  els.meta.textContent = count ? `${size} · ${t("erase.applied", { count })}` : size;
  els.meta.hidden = false;
}

function fail(error, retry = null) {
  state.error = error;
  state.retry = retry;
  // The error and its retry live above the stage; the panel stays quiet.
  status.clear();
  showNotice(els.notices, error, retry);
}

function clearError() {
  state.error = null;
  state.retry = null;
  clearNotice(els.notices);
}

// ── Output ────────────────────────────────────────────────────────────────────

function extension(blob) {
  return { "image/png": "png", "image/webp": "webp" }[blob.type] ?? "jpg";
}

function save() {
  const { blob } = current() ?? {};
  if (blob) download(blob, `${stem(state.name)}-erased.${extension(blob)}`);
}

function toCutout() {
  const { blob } = current() ?? {};
  if (!blob) return;
  ctx.navigate("cutout", {
    file: new File([blob], `${stem(state.name)}-erased.${extension(blob)}`, { type: blob.type }),
  });
}

function shortcuts(event) {
  if (root.hidden || !state.versions.length) return;
  if (event.target.closest?.("textarea, select, input:not([type=range])")) return;
  const key = event.key.toLowerCase();
  if ((event.ctrlKey || event.metaKey) && key === "z") {
    event.preventDefault();
    if (event.shiftKey) redo();
    else undo();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && key === "y") {
    event.preventDefault();
    redo();
    return;
  }
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === "[" || event.key === "]") {
    event.preventDefault();
    setBrush(state.brush + (event.key === "]" ? BRUSH.step : -BRUSH.step));
  } else if (event.key === "Enter" && !event.target.closest?.("button, a, input")) {
    event.preventDefault();
    run();
  }
}
