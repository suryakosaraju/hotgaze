// Shared helpers: DOM building, API calls with translated errors, file intake,
// the before/after compare slider, toasts and downloads.

import { has, t } from "./i18n.js";

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

/** Build an element. All user-provided text goes through textContent. */
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "style") Object.assign(node.style, value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  node.append(...children.flat().filter((child) => child !== null && child !== undefined && child !== false));
  return node;
}

/** An icon from the sprite in index.html. */
export function icon(name, className = "icon") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", className);
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

// ── Server info ───────────────────────────────────────────────────────────────

const fallbackInfo = {
  tools: { heatmap: true, cutout: true, erase: true },
  backends: ["fast"],
  limits: {
    max_pixels: 10_000_000,
    max_upload_bytes: 20 * 1024 * 1024,
    edit_max_pixels: 25_000_000,
    edit_max_edge: 2048,
  },
};

let infoPromise = null;

/** /api/info, fetched once. Falls back to defaults so the page still works offline. */
export function serverInfo() {
  infoPromise ??= fetch("/api/info")
    .then((response) => (response.ok ? response.json() : fallbackInfo))
    .catch(() => fallbackInfo);
  return infoPromise;
}

// ── API calls ─────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(code, params = {}, message = "") {
    super(message || code);
    this.code = code;
    this.params = params;
  }
}

/**
 * POST form data. Resolves to the parsed JSON, or to {blob, headers} for image
 * responses. Rejects with an ApiError carrying a translatable code.
 */
export async function post(path, body, { signal } = {}) {
  let response;
  try {
    response = await fetch(path, { method: "POST", body, signal });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("network", {}, error.message);
  }
  const type = response.headers.get("content-type") || "";
  if (response.ok) {
    if (type.startsWith("image/")) return { blob: await response.blob(), headers: response.headers };
    return response.json();
  }
  const payload = type.includes("json") ? await response.json().catch(() => null) : null;
  const detail = payload?.detail;
  if (detail && typeof detail === "object" && !Array.isArray(detail) && detail.code) {
    throw new ApiError(detail.code, detail.params || {}, detail.message);
  }
  if (Array.isArray(detail)) {
    const message = detail
      .map((item) => `${(item.loc || []).slice(1).join(".") || "request"}: ${item.msg}`)
      .join("; ");
    throw new ApiError("validation", { message }, message);
  }
  if (typeof detail === "string") throw new ApiError("generic", { message: detail }, detail);
  // A platform proxy answers 502/503/504 with an HTML page while the app is asleep or restarting.
  if ([502, 503, 504].includes(response.status)) throw new ApiError("network", {}, response.statusText);
  throw new ApiError("server", { status: response.status }, response.statusText);
}

/** A localized sentence for any error thrown by post() or the file helpers. */
export function errorMessage(error) {
  if (error instanceof ApiError) {
    const key = `errors.${error.code}`;
    if (has(key)) return t(key, { message: error.message, ...error.params });
    return t("errors.generic", { message: error.message });
  }
  return t("errors.generic", { message: error?.message || String(error) });
}

// ── Files ─────────────────────────────────────────────────────────────────────

const ACCEPTED = ["image/png", "image/jpeg", "image/webp"];

export function isImageFile(file) {
  return Boolean(file) && (ACCEPTED.includes(file.type) || /\.(png|jpe?g|webp)$/i.test(file.name || ""));
}

/** The first image in a paste or drop, if any. */
export function imageFromTransfer(transfer) {
  if (!transfer) return null;
  for (const file of transfer.files || []) if (isImageFile(file)) return file;
  for (const item of transfer.items || []) {
    if (item.kind === "file") {
      const file = item.getAsFile();
      if (isImageFile(file)) return file.name ? file : new File([file], "pasted.png", { type: file.type });
    }
  }
  return null;
}

/**
 * Decode a file as the browser displays it (EXIF orientation applied) and, when
 * its long side exceeds maxEdge, downscale it. Returns the bitmap, its size, a
 * file to upload and whether it was scaled. Throws ApiError("local_file").
 */
export async function prepareImage(file, maxEdge) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    throw new ApiError("local_file");
  }
  const longEdge = Math.max(bitmap.width, bitmap.height);
  if (longEdge <= maxEdge && file.size <= 8 * 1024 * 1024) {
    return { bitmap, width: bitmap.width, height: bitmap.height, file, scaled: false };
  }
  const scale = Math.min(1, maxEdge / longEdge);
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = h("canvas", { width, height });
  const context = canvas.getContext("2d");
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close?.();
  const type = file.type === "image/png" ? "image/png" : "image/jpeg";
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, type, 0.95));
  const name = renameExtension(file.name || "image", type === "image/png" ? "png" : "jpg");
  const resized = new File([blob], name, { type });
  return {
    bitmap: await createImageBitmap(resized),
    width,
    height,
    file: resized,
    scaled: scale < 1,
  };
}

export function renameExtension(name, extension) {
  const stem = name.replace(/\.[^.]+$/, "") || "image";
  return `${stem}.${extension}`;
}

export function stem(name) {
  return (name || "image").replace(/\.[^.]+$/, "") || "image";
}

/** Fetch a same-origin sample as a File. */
export async function fetchSample(url, name) {
  const response = await fetch(url);
  if (!response.ok) throw new ApiError("network", {}, response.statusText);
  const blob = await response.blob();
  return new File([blob], name, { type: blob.type || "image/png" });
}

/**
 * Wire a dropzone: drag and drop, plus click-to-browse through its <input type=file>
 * when one is given. `onFile` receives the first image dropped or chosen.
 */
export function bindDropzone(zone, input, onFile) {
  input?.addEventListener("change", () => {
    const file = input.files?.[0];
    input.value = "";
    if (file) onFile(file);
  });
  let depth = 0;
  zone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    depth += 1;
    zone.classList.add("is-over");
  });
  zone.addEventListener("dragover", (event) => event.preventDefault());
  zone.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) zone.classList.remove("is-over");
  });
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    depth = 0;
    zone.classList.remove("is-over");
    const file = imageFromTransfer(event.dataTransfer);
    if (file) onFile(file);
    else if (event.dataTransfer?.files?.length) toast(t("errors.local_file"));
  });
}

// ── Compare slider ────────────────────────────────────────────────────────────

/**
 * Before/after slider. The returned element is focusable (role=slider): drag it,
 * click anywhere, or use the arrow keys. `set(before, after, ratio)` swaps images.
 */
export function compareSlider({ label = t("compare.label"), position = 50, checker = false } = {}) {
  const before = h("img", { alt: "", draggable: "false", decoding: "async" });
  const after = h("img", { alt: "", draggable: "false", decoding: "async" });
  const afterLayer = h("div", { class: checker ? "compare-after checker" : "compare-after" }, after);
  const tagBefore = h("span", { class: "compare-tag before", "data-i18n": "compare.before", text: t("compare.before") });
  const tagAfter = h("span", { class: "compare-tag after", "data-i18n": "compare.after", text: t("compare.after") });
  const knob = h("span", { class: "compare-knob" }, icon("drag"));
  const handle = h("span", { class: "compare-handle", "aria-hidden": "true" }, knob);
  const root = h(
    "div",
    {
      class: "compare",
      role: "slider",
      tabindex: "0",
      "aria-label": label,
      "aria-valuemin": "0",
      "aria-valuemax": "100",
      "data-i18n-attr": "aria-label:compare.label",
    },
    before,
    afterLayer,
    tagBefore,
    tagAfter,
    handle,
  );

  let current = position;
  const apply = (value, { sweep = false } = {}) => {
    current = Math.min(100, Math.max(0, value));
    root.classList.toggle("is-sweeping", sweep);
    root.style.setProperty("--pos", `${current}%`);
    root.setAttribute("aria-valuenow", String(Math.round(current)));
    root.setAttribute("aria-valuetext", `${Math.round(current)}%`);
  };
  apply(current);

  const fromPointer = (event) => {
    const box = root.getBoundingClientRect();
    apply(((event.clientX - box.left) / box.width) * 100);
  };
  root.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    try {
      root.setPointerCapture(event.pointerId);
    } catch {
      // The pointer is already gone; follow it without capture.
    }
    fromPointer(event);
    root.dataset.dragging = "true";
  });
  root.addEventListener("pointermove", (event) => {
    if (root.dataset.dragging) fromPointer(event);
  });
  const stop = () => delete root.dataset.dragging;
  root.addEventListener("pointerup", stop);
  root.addEventListener("pointercancel", stop);
  root.addEventListener("keydown", (event) => {
    const steps = { ArrowLeft: -5, ArrowDown: -5, ArrowRight: 5, ArrowUp: 5, PageDown: -20, PageUp: 20 };
    if (event.key in steps) apply(current + steps[event.key]);
    else if (event.key === "Home") apply(0);
    else if (event.key === "End") apply(100);
    else return;
    event.preventDefault();
  });

  return {
    root,
    set({ beforeSrc, afterSrc, beforeAlt = "", afterAlt = "", width, height }) {
      if (width && height) root.style.setProperty("--ar", String(width / height));
      before.src = beforeSrc;
      after.src = afterSrc;
      before.alt = beforeAlt;
      after.alt = afterAlt;
    },
    setBackground(color) {
      afterLayer.classList.toggle("checker", !color);
      afterLayer.style.backgroundColor = color || "";
    },
    position: (value, options) => apply(value, options),
  };
}

// ── Feedback ──────────────────────────────────────────────────────────────────

let toastTimer = null;

export function toast(message) {
  let node = $(".toast");
  if (!node) {
    node = h("div", { class: "toast", role: "status" });
    document.body.append(node);
  }
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.hidden = true;
  }, 2600);
}

/** An inline error with an optional retry button. */
export function alertBox(message, { kind = "", retry } = {}) {
  const body = h("div", { class: "alert-body" }, h("p", { text: message }));
  if (retry) {
    body.append(
      h("div", {}, h("button", { type: "button", class: "btn btn-sm", onclick: retry, text: t("tool.retry") })),
    );
  }
  return h(
    "div",
    { class: kind ? `alert alert-${kind}` : "alert", role: kind === "info" ? "status" : "alert" },
    icon(kind === "info" ? "info" : "alert"),
    body,
  );
}

/**
 * The status line of a tool (an aria-live region) plus an optional pill on the
 * stage for narrow screens. While working it counts seconds in an aria-hidden
 * span, so screen readers hear each state once rather than every second, and it
 * adds a "slower than usual" hint after `slowAfter` seconds.
 */
export function statusLine(node, pill = null) {
  let state = null;
  let timer = null;
  let parts = null;

  const seconds = () => Math.round((performance.now() - state.started) / 1000);

  const tick = () => {
    const elapsed = t("status.elapsed", { seconds: seconds() });
    parts.elapsed.textContent = elapsed;
    if (parts.pillElapsed) parts.pillElapsed.textContent = elapsed;
    if (state.slowAfter !== null && seconds() >= state.slowAfter) parts.hint.textContent = t(state.slowKey);
  };

  const render = () => {
    clearInterval(timer);
    if (pill) pill.hidden = !state || state.kind !== "working";
    if (!state) {
      node.replaceChildren();
      return;
    }
    if (state.kind === "working") {
      parts = {
        elapsed: h("span", { class: "status-elapsed num", "aria-hidden": "true" }),
        hint: h("small", { class: "status-hint", text: state.hintKey ? t(state.hintKey) : "" }),
        pillElapsed: pill ? h("span", { class: "num", "aria-hidden": "true" }) : null,
      };
      node.className = "status";
      node.replaceChildren(
        h("span", { class: "spinner", "aria-hidden": "true" }),
        h("span", { class: "status-text" }, h("span", { text: t(state.key) }), " ", parts.elapsed, parts.hint),
      );
      pill?.replaceChildren(h("span", { class: "spinner", "aria-hidden": "true" }), h("span", { text: t(state.key) }), parts.pillElapsed);
      tick();
      timer = setInterval(tick, 1000);
      return;
    }
    node.className = state.kind === "done" ? "status status-done" : "status";
    node.replaceChildren(icon(state.kind === "done" ? "check" : "info"), h("span", { text: t(state.key, state.params) }));
  };

  return {
    working(key, { hintKey = null, slowAfter = null, slowKey = "status.slow" } = {}) {
      state = { kind: "working", key, hintKey, slowAfter, slowKey, started: performance.now() };
      render();
    },
    done(key, params = {}) {
      state = { kind: "done", key, params };
      render();
    },
    info(key, params = {}) {
      state = { kind: "info", key, params };
      render();
    },
    clear() {
      state = null;
      render();
    },
    /** Seconds since working() was called. */
    elapsed() {
      return state?.started ? (performance.now() - state.started) / 1000 : 0;
    },
    /** Re-render in the current language. */
    refresh() {
      render();
    },
  };
}

/** Show an error above the stage of a tool, with an optional retry. */
export function showNotice(slot, error, retry) {
  clearNotice(slot);
  slot.append(alertBox(errorMessage(error), { retry }));
}

export function clearNotice(slot) {
  slot.querySelector(".alert:not(.alert-warning)")?.remove();
}

// ── Downloads ─────────────────────────────────────────────────────────────────

export function download(blobOrUrl, filename) {
  const href = typeof blobOrUrl === "string" ? blobOrUrl : URL.createObjectURL(blobOrUrl);
  const link = h("a", { href, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  if (typeof blobOrUrl !== "string") setTimeout(() => URL.revokeObjectURL(href), 10_000);
}

export function canvasBlob(canvas, type = "image/png", quality = 0.95) {
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("encode"))), type, quality),
  );
}

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new ApiError("local_file"));
    image.src = src;
  });
}

export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
