// Module palette: category tabs, search, lazy thumbnails, brush arming.

import { getCatalogue } from "./kit.js";
import { request as requestThumb, requestTurntable, TURN_FRAMES } from "./thumbs.js";
import { state, emit } from "./editor.js";
import { armGhost, cancelGhost } from "./interact.js";

const listEl = document.getElementById("palette-list");
const tabsEl = document.getElementById("palette-tabs");
const searchEl = document.getElementById("palette-search");
const brushEl = document.getElementById("brush-label");

let activeCategory = "All";
let observer = null;

export function initPalette() {
  const cat = getCatalogue();

  const cats = ["All", ...cat.categories.map((c) => c.name)];
  tabsEl.innerHTML = "";
  for (const name of cats) {
    const b = document.createElement("button");
    b.textContent = name;
    b.dataset.cat = name;
    if (name === activeCategory) b.classList.add("active");
    b.addEventListener("click", () => {
      activeCategory = name;
      [...tabsEl.children].forEach((x) => x.classList.toggle("active", x.dataset.cat === name));
      render();
    });
    tabsEl.appendChild(b);
  }

  searchEl.addEventListener("input", render);

  observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const img = e.target;
      observer.unobserve(img);
      requestThumb(JSON.parse(img.dataset.mod), img);
    }
  }, { root: listEl, rootMargin: "220px" });

  render();
}

function visibleModules() {
  const cat = getCatalogue();
  const q = searchEl.value.trim().toLowerCase();
  const out = [];
  for (const c of cat.categories) {
    if (activeCategory !== "All" && c.name !== activeCategory) continue;
    for (const m of c.modules) {
      if (q && !m.name.toLowerCase().includes(q)) continue;
      out.push(m);
    }
  }
  return out;
}

function render() {
  const mods = visibleModules();
  listEl.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const m of mods) {
    const el = document.createElement("div");
    el.className = "item" + (state.brush === m.id ? " active" : "");
    el.dataset.id = m.id;
    el.title = m.id;

    const img = document.createElement("img");
    img.className = "thumb";
    img.alt = m.name;
    img.loading = "lazy";
    img.dataset.mod = JSON.stringify(m);

    const cap = document.createElement("span");
    cap.className = "cap";
    cap.textContent = m.name;

    el.append(img, cap);
    el.addEventListener("click", () => setBrush(state.brush === m.id ? null : m.id));
    el.addEventListener("pointerenter", () => startTurntable(el, m));
    el.addEventListener("pointerleave", () => stopTurntable(el));
    frag.appendChild(el);
    observer.observe(img);
  }
  listEl.appendChild(frag);
  document.getElementById("status-text").textContent =
    `${mods.length} modules shown of ${getCatalogue().byId.size}`;
}

export function setBrush(id, opts = {}) {
  state.brush = id;
  brushEl.textContent = id ? `Placing: ${id}` : "No module selected";
  [...listEl.children].forEach((el) => el.classList.toggle("active", el.dataset.id === id));
  // Transient guidance only, and no idle text: the wheel no longer turns
  // anything, which is exactly how the old string went stale.
  document.getElementById("hint").textContent = id
    ? "Click to place · E turns · Shift+wheel scales · Esc or right-click to stop"
    : "";
  if (id) armGhost(id, opts); else cancelGhost();
  emit("brush");
}

export function refreshPalette() { render(); }

// ------------------------------------------------------------- turntables
//
// A still tile cannot tell you which way a corner turns or which side a cable
// run is on. Hovering spins the module through a full turn, which answers both
// at a glance. Only the hovered tile ever animates, so a single timer is
// enough - and it is driven in JS rather than a CSS steps() animation because
// stepping whole pixels keeps every frame exactly aligned.

const TURN_MS = 180;
let turnTile = null;
let turnTimer = 0;
let turnToken = 0;

async function startTurntable(el, mod) {
  stopTurntable(turnTile);
  const token = ++turnToken;
  el.classList.add("turn-wait");

  let url;
  try {
    url = await requestTurntable(mod);
  } catch {
    el.classList.remove("turn-wait");
    return;
  }
  // The pointer may well have moved on while 12 frames were rendering.
  if (token !== turnToken || !el.isConnected) { el.classList.remove("turn-wait"); return; }

  el.classList.remove("turn-wait");
  const film = document.createElement("div");
  film.className = "turn";
  film.style.backgroundImage = `url(${url})`;
  film.style.backgroundSize = `${TURN_FRAMES * 100}% 100%`;
  el.appendChild(film);

  turnTile = el;
  let frame = 0;
  const step = () => {
    // Percentage background-position is measured against (box - image), so the
    // step between frames is 1/(n-1) of the range, not 1/n.
    film.style.backgroundPositionX = `${(frame / (TURN_FRAMES - 1)) * 100}%`;
    frame = (frame + 1) % TURN_FRAMES;
  };
  step();
  turnTimer = setInterval(step, TURN_MS);
}

function stopTurntable(el) {
  turnToken++;
  if (turnTimer) { clearInterval(turnTimer); turnTimer = 0; }
  const tile = el || turnTile;
  if (tile) {
    tile.classList.remove("turn-wait");
    tile.querySelector(".turn")?.remove();
  }
  turnTile = null;
}

/** Exposed so tests can assert the animation state without racing the timer. */
export function turntableState() {
  return { tile: turnTile?.dataset.id || null, running: !!turnTimer, frames: TURN_FRAMES };
}
