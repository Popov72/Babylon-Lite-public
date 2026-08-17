// Module palette: kit picker, category tabs, search, lazy thumbnails, brush arming.

import { getCatalogue } from "./kit.js";
import { request as requestThumb, cachedUrl, requestTurntable, TURN_FRAMES } from "./thumbs.js";
import { state, emit, on, hooks } from "./editor.js";
import { armGhost, cancelGhost } from "./interact.js";

const listEl = document.getElementById("palette-list");
const kitEl = document.getElementById("palette-kit");
const tabsEl = document.getElementById("palette-tabs");
const searchEl = document.getElementById("palette-search");
const brushEl = document.getElementById("brush-label");

// Which kit the palette is showing. Always exactly one: the library spans five
// unrelated packs, and merging their categories gives a strip of 27 tabs where
// Walls, Rocks and Potions sit side by side, which groups nothing. The choice
// is a view preference, not ship data - a manifest may name a module from any
// kit whatever the palette happens to be showing - so it is remembered in
// localStorage beside the palette width rather than written to the ship.
const KIT_STORE = "paletteKit";
let activeKit = null;
let activeCategory = "All";
let observer = null;

export function initPalette() {
  const kits = (getCatalogue().kits || []).map((k) => k.name);
  const saved = localStorage.getItem(KIT_STORE);
  activeKit = kits.includes(saved) ? saved : (kits[0] || null);

  kitEl.innerHTML = "";
  for (const name of kits) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    kitEl.appendChild(o);
  }
  kitEl.value = activeKit || "";
  kitEl.addEventListener("change", () => {
    activeKit = kitEl.value;
    localStorage.setItem(KIT_STORE, activeKit);
    buildTabs();
    render();
  });

  buildTabs();
  searchEl.addEventListener("input", render);

  // Saving or deleting a compound rewrites the catalogue in place, and the
  // palette is the thing that shows it. Re-tabbing as well as re-rendering:
  // the first compound filed under a category creates that category.
  on("catalogue", () => { buildTabs(); render(); });

  observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      observer.unobserve(e.target);
      fillThumb(e.target);
    }
  }, { root: listEl, rootMargin: "220px" });

  render();
}

function kitInfo() {
  return (getCatalogue().kits || []).find((k) => k.name === activeKit) || null;
}

/**
 * The category tabs of the kit on show.
 *
 * Rebuilt per kit rather than filtered, because a category belongs to the kit
 * it came from: "Rocks" means nothing in the MegaKit, and a stale selection
 * would leave the palette empty with no clue why. A category that survives the
 * switch - both sci-fi kits have Props - keeps its place.
 */
function buildTabs() {
  const cats = ["All", ...(kitInfo()?.categories || [])];
  if (!cats.includes(activeCategory)) activeCategory = "All";
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
}

function visibleModules() {
  const cat = getCatalogue();
  const q = searchEl.value.trim().toLowerCase();
  const out = [];
  for (const c of cat.categories) {
    if (activeCategory !== "All" && c.name !== activeCategory) continue;
    for (const m of c.modules) {
      // The catalogue's categories span the whole library - Props holds both
      // sci-fi kits' props - so the kit is filtered here rather than in the
      // tabs.
      if (activeKit && m.kit !== activeKit) continue;
      if (q && !m.name.toLowerCase().includes(q)) continue;
      out.push(m);
    }
  }
  return out;
}

/**
 * Fill a tile's thumbnail, saying so while it is being made.
 *
 * A module that has never been previewed has to be loaded and rendered before
 * there is a picture at all, and switching to a fresh kit queues a hundred and
 * fifty of them. The tile used to sit blank for the whole wait, which reads as
 * a broken image rather than as work in progress.
 */
function fillThumb(img) {
  const mod = JSON.parse(img.dataset.mod);
  const tile = img.closest(".item");
  // A cached thumbnail is set synchronously, so only a render that has to
  // happen now is worth announcing - otherwise every tile flashes the message.
  if (!cachedUrl(mod.id)) tile?.classList.add("thumb-pending");
  requestThumb(mod, img).finally(() => tile?.classList.remove("thumb-pending"));
}

function render() {
  const mods = visibleModules();
  listEl.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const m of mods) {
    const el = document.createElement("div");
    el.className = "item" + (state.brush === m.id ? " active" : "");
    el.dataset.id = m.id;

    const img = document.createElement("img");
    img.className = "thumb";
    img.alt = m.name;
    img.loading = "lazy";
    img.dataset.mod = JSON.stringify(m);

    const cap = document.createElement("span");
    cap.className = "cap";
    cap.textContent = m.name;

    el.append(img, cap);
    markCollision(el, m.id);
    // A compound is the only kind of tile the editor itself made, so it is the
    // only kind it can throw away. The cross lives on the tile rather than
    // behind a menu because that is where you are when you decide.
    if (m.compound) {
      el.classList.add("compound");
      const del = document.createElement("button");
      del.type = "button";
      del.className = "tile-del";
      del.textContent = "\u00d7";
      del.title = `Delete the "${m.name}" compound`;
      del.addEventListener("click", (ev) => {
        ev.stopPropagation();          // not also a click on the tile beneath
        emit("deletecompound", m.name);
      });
      el.appendChild(del);
    }
    el.addEventListener("click", () => {
      // On the collision staging area the palette *stages* modules: there is no
      // ship on screen to arm a brush against, and staging is what you came for.
      // A compound has no model of its own to fit a hull to - its members carry
      // their own - so there is nothing there to stage.
      if (state.mode === "collision") {
        if (!m.compound) emit("stagemodule", m.id);
        return;
      }
      // On the compound bench a compound tile *opens* rather than places. It is
      // the only way back to a saved recipe, and placing one inside another
      // would fold a whole compound into the next save as loose members - which
      // is what "break apart" is for and not what a click on its own tile
      // should ever mean.
      if (state.mode === "compound" && m.compound) { emit("editcompound", m.name); return; }
      setBrush(state.brush === m.id ? null : m.id);
    });
    el.addEventListener("pointerenter", () => startTurntable(el, m));
    el.addEventListener("pointerleave", () => stopTurntable(el));
    frag.appendChild(el);
    observer.observe(img);
  }
  listEl.appendChild(frag);
  document.getElementById("status-text").textContent =
    `${mods.length} modules shown of ${kitInfo()?.count ?? getCatalogue().byId.size}`;
}

/**
 * Mark a tile whose module carries collision.
 *
 * The green dot is the same green the shapes are drawn in, so "this one is
 * done" reads the same on the tile as it does in the viewport. Without it the
 * only way to tell which of 277 modules had been fitted was to stage each one
 * and look - and fitting a kit is precisely a job you do a few at a time and
 * come back to.
 */
function markCollision(el, moduleId) {
  const n = state.moduleCollision.get(moduleId)?.length || 0;
  el.classList.toggle("has-collision", n > 0);
  el.title = n
    ? `${moduleId} — ${n} collision shape${n === 1 ? "" : "s"}`
    : moduleId;
}

/** Re-mark every visible tile, without rebuilding the list. */
export function refreshCollisionMarks() {
  if (!listEl) return;
  for (const el of listEl.children) {
    if (el.dataset.id) markCollision(el, el.dataset.id);
  }
}

export function setBrush(id, opts = {}) {
  state.brush = id;
  brushEl.textContent = id ? `Placing: ${id}` : "No module selected";
  [...listEl.children].forEach((el) => el.classList.toggle("active", el.dataset.id === id));
  // Transient guidance only, and no idle text: the old string named keys that
  // have since moved, which is exactly how it went stale.
  document.getElementById("hint").textContent = id
    ? (state.mode === "collision"
      ? "Click to put it on the bench · Shift+wheel turns · Ctrl+wheel scales · Esc to stop"
      : "Click to place · B brings it to you · Shift+wheel turns · Ctrl+wheel scales · Esc or right-click to stop")
    : "";
  if (id) armGhost(id, opts); else cancelGhost();
  emit("brush");
}

hooks.clearBrush = () => setBrush(null);

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

