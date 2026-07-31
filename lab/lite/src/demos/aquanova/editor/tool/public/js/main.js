// Wiring: toolbar, inspector, keyboard, autoload.

import { loadCatalogue, getCatalogue } from "./kit.js";
import { initThumbs } from "./thumbs.js";
import { initPalette, setBrush } from "./palette.js";
import { saveLayout, loadLayout, exportGlb, resolveDoorChunks } from "./manifest.js";
import { addDoor, doorFromSelection, resizeDoor } from "./markers.js";
import {
  initInteract, cancelGhost, cancelDrag, isDragging, currentElement,
  ghostActive, ghostModule, hoveredId, cycleRotAxis, cycleScaleAxis, rotateCurrent, flipCurrent,
  toggleDragAxis, setDragAxis, cancelMarquee,
} from "./interact.js";
import {
  state, on, emit, initScene, setGridVisible, setGridElevation,
  nudgeGridElevation, select, removeSelected, duplicateSelected, focusSelection,
  addChunk, assignSelectionToChunk, applyVisibility, undo, redo, pushUndo,
  renameChunk, renamePlacement, hideSelected, unhideAll, hiddenCount, veilCounts,
  setVeilAlpha,
  getBehaviorDef, setBehaviorDef, renameBehaviorDef, deleteBehaviorDef, behaviorNames,
  entityBehaviors, addEntityBehavior, removeEntityBehavior, setEntityLinked,
  isLiquefiable, defaultDirection, setEntityDirection, nodeNamesInChunk, nodesNamed,
  setEntityExcludeSDF, dynamicNodeNamesInChunk,
  isBusy, busyLabel, whileBusy,
  toggleAxes, nearestToCursor, hideAxes, GHOST_AXES,
  eulerOf, setEuler, worldBounds, entryOf, nudgeSelection,
  noteKey, releaseAllKeys, setUnlit, setExposure, EXPOSURE_DEFAULT,
  setWalk, EYE_HEIGHT, setEnvIntensity, ENV_INTENSITY_DEFAULT, setSelectMode,
  setRuntimeLighting, activeLightSet,
} from "./editor.js";

const $ = (id) => document.getElementById(id);
const statusText = $("status-text");
const statusCounts = $("status-counts");

/**
 * No browser context menu anywhere in the editor.
 *
 * The right button is a camera control here, so the menu its release raises is
 * never wanted - and guarding the canvas was never enough. Measured order for a
 * right-drag that ends off-canvas: pointerdown@CANVAS, pointerup@CANVAS (the
 * canvas holds pointer capture), then contextmenu@<whatever is under the
 * cursor>. Capture does not redirect contextmenu, and it arrives *after* the
 * release, so neither a canvas handler nor a "button still down" flag can catch
 * it: releasing over a palette thumbnail served up "Save image as".
 *
 * Registered here rather than with the scene, and at module scope, because the
 * loading overlay is up before there is a scene at all - and that was the one
 * place a menu could still be raised.
 *
 * Ctrl+C/Ctrl+V still work in the text fields; only the menu is gone.
 */
document.addEventListener("contextmenu", (e) => e.preventDefault(), true);

function setStatus(msg) { statusText.textContent = msg; }

// -------------------------------------------------------------- inspector

const posIn = ["pos-x", "pos-y", "pos-z"].map($);
const rotIn = ["rot-x", "rot-y", "rot-z"].map($);
const sclIn = ["scl-x", "scl-y", "scl-z"].map($);
let syncing = false;

function refreshInspector() {
  const n = state.selection.length;
  $("insp-empty").hidden = n > 0;
  $("insp-body").hidden = n === 0;
  if (!n) return;

  const e = entryOf(state.selection[0]);
  if (!e) return;
  const isDoor = e.type === "door";
  const isMarker = !!e.type;

  syncing = true;
  // Always the element whose values fill the rest of the panel, even in a
  // multi-selection - Module says "3 selected" but the transform below is one
  // element's, and until now nothing said which.
  $("insp-id").textContent = e.id;
  $("insp-id").title = e.id;
  // Markers have no module, and used to fall back to the id - which the Id row
  // above now shows anyway. Their type is the useful thing instead.
  $("insp-module").textContent = n > 1 ? `${n} selected` : (e.module || e.type || e.id);
  $("insp-name").parentElement.hidden = n > 1 || isMarker;
  // Written even when the row is hidden: a stale name sitting in there is one
  // CSS rule away from being on screen against the wrong element, and it is
  // what the behaviour panel would key off if it ever read the field.
  setField($("insp-name"), n === 1 && !isMarker ? (e.name || "") : "");
  $("insp-chunk").parentElement.hidden = isMarker;
  if (!isMarker) $("insp-chunk").value = e.chunk;
  const p = e.node.position, r = eulerOf(e.node), s = e.node.scaling;
  posIn.forEach((el, i) => setField(el, round(p.asArray()[i])));
  rotIn.forEach((el, i) => setField(el, round(r[i])));
  sclIn.forEach((el, i) => setField(el, round(s.asArray()[i])));

  $("door-fields").hidden = !isDoor;
  if (isDoor) {
    setField($("door-w"), e.width);
    setField($("door-h"), e.height);
    fillChunkSelect($("door-a"), e.chunkA);
    fillChunkSelect($("door-b"), e.chunkB);
    setField($("door-trig"), e.triggerRadius);
    setField($("door-slide"), e.slideDistance);
    $("door-leaves").textContent = e.leaves.length ? e.leaves.join(", ") : "none";
  }
  refreshDimensions();
  refreshBehavior();
  syncing = false;
}

/**
 * The behaviour panel, driven by the selected element's *name*.
 *
 * Behaviours attach to a node name, not to an element, so this panel is really
 * an editor for the name currently in the Name field - which is why the element
 * count is shown: attaching one here can be governing one crate or thirty, and
 * there is no other way to tell. Zero means the entry is orphaned.
 *
 * A behaviour with no name could never be matched to an element, so nothing can
 * be attached until it has one.
 */
function refreshBehavior() {
  const single = state.selection.length === 1 ? entryOf(state.selection[0]) : null;
  const placement = single && !single.type ? single : null;
  $("behavior-fields").hidden = !placement;
  if (!placement) return;

  const name = (placement.name || "").trim();
  const count = nodesNamed(name);
  const applied = name ? entityBehaviors(name) : [];
  const library = behaviorNames();

  $("bhv-count").textContent = name
    ? `"${name}" — ${count} element${count === 1 ? "" : "s"}`
    : "";
  renderApplied(name, applied);

  const free = library.filter((b) => !applied.some((a) => a.name === b));
  $("bhv-add").innerHTML = free.length
    ? free.map((b) => `<option value="${esc(b)}">${esc(b)}</option>`).join("")
    : `<option disabled>${library.length ? "(all attached)" : "(none defined)"}</option>`;
  $("bhv-add").disabled = !name || !free.length;
  $("btn-bhv-add").disabled = !name || !free.length;

  $("bhv-hint").textContent = !name
    ? "Name the element first — behaviours attach to the node name."
    : library.length ? "" : "No behaviours defined yet.";
}

const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/**
 * The behaviours on this node, each with a Remove button - and, for the ones
 * that liquefy, the `linked` picker. Only liquefaction needs it: linking is for
 * pieces that melt as one, which is what a pair of door halves is.
 *
 * Candidates come from the **current room**, since linked pieces are neighbours
 * in practice and a ship-wide list would be hundreds of entries long.
 */
function renderApplied(nodeName, applied) {
  const host = $("bhv-applied");
  // Never rebuild under a field being typed in: it destroys the caret, the
  // same trap the inspector's number fields fell into. A focused *button* is
  // not editing, so Remove still redraws the list it just changed.
  const el = document.activeElement;
  if (host.contains(el) && /^(INPUT|SELECT)$/.test(el.tagName)) return;

  if (!applied.length) {
    host.innerHTML = nodeName ? '<div class="muted">none attached</div>' : "";
    return;
  }
  const chunk = entryOf(state.selection[0])?.chunk;
  const candidates = nodeNamesInChunk(chunk, nodeName);
  const dynamics = dynamicNodeNamesInChunk(chunk, nodeName);
  const picker = (kind, label, options, chosen, name) => {
    const opts = options.length
      ? options.map((n) =>
        `<option value="${esc(n)}"${chosen.includes(n) ? " selected" : ""}>${esc(n)}</option>`)
        .join("")
      : `<option disabled>(${kind === "linked"
        ? "nothing else named in this room"
        : "nothing dynamic in this room"})</option>`;
    return `<div class="linked"><div class="lbl">${label}</div>`
      + `<select multiple size="4" data-${kind}="${esc(name)}">${opts}</select></div>`;
  };
  host.innerHTML = applied.map((b) => {
    const rows = [
      `<div class="item"><span class="n">${esc(b.name)}</span>`
      + `<button data-remove="${esc(b.name)}">Remove</button></div>`,
    ];
    if (isLiquefiable(b.name)) {
      rows.push(picker("linked", "linked — melts together", candidates, b.linked, b.name));
      // Only a body with an SDF in the simulation is worth dropping from it,
      // so the list is the dynamic nodes rather than every named one.
      rows.push(picker("sdf", "excludeSDF — kept out of the fluid sim",
        dynamics, b.excludeSDF || [], b.name));
    }
    // Optional on every applied behaviour - the definition only supplies a
    // starting value. Gating this on the definition declaring it made an
    // optional parameter invisible until you knew to declare it.
    const d = b.direction || defaultDirection(b.name) || [];
    const f = (i) => `<input type="number" step="0.1" data-dir="${esc(b.name)}" `
      + `data-axis="${i}" value="${Number.isFinite(d[i]) ? d[i] : ""}">`;
    rows.push('<div class="linked"><div class="lbl">direction — the way it faces (optional)</div>'
      + `<div class="vec">${f(0)}${f(1)}${f(2)}</div></div>`);
    return rows.join("");
  }).join("");

  for (const btn of host.querySelectorAll("[data-remove]")) {
    btn.addEventListener("click", () => {
      removeEntityBehavior(nodeName, btn.dataset.remove);
      refreshBehavior();
    });
  }
  for (const sel of host.querySelectorAll("[data-linked]")) {
    sel.addEventListener("change", () => {
      setEntityLinked(nodeName, sel.dataset.linked,
        [...sel.selectedOptions].map((o) => o.value));
      refreshBehavior();
    });
  }
  for (const sel of host.querySelectorAll("[data-sdf]")) {
    sel.addEventListener("change", () => {
      setEntityExcludeSDF(nodeName, sel.dataset.sdf,
        [...sel.selectedOptions].map((o) => o.value));
      refreshBehavior();
    });
  }
  // `change`, not `input`: committing on every keystroke would store a
  // half-typed "-" or "0." as the direction.
  for (const input of host.querySelectorAll("[data-dir]")) {
    input.addEventListener("change", () => {
      const name = input.dataset.dir;
      const fields = [...host.querySelectorAll(`[data-dir="${CSS.escape(name)}"]`)];
      const vec = [0, 1, 2].map((i) => {
        const f = fields.find((x) => +x.dataset.axis === i);
        return f && f.value.trim() !== "" ? parseFloat(f.value) : 0;
      });
      setEntityDirection(nodeName, name, vec.every((v) => v === 0) ? null : vec);
      refreshBehavior();
    });
  }
}

function selectedName() {
  const e = state.selection.length === 1 ? entryOf(state.selection[0]) : null;
  return e && !e.type ? (e.name || "").trim() : "";
}

$("btn-bhv-add").addEventListener("click", () => {
  const name = selectedName();
  const pick = $("bhv-add").value;
  if (!name || !pick) return;
  addEntityBehavior(name, pick);
  refreshBehavior();
});

// ------------------------------------------------- behaviour library dialog

let libSelected = null;

function openLibrary() {
  $("bhv-modal").hidden = false;
  refreshLibrary(libSelected || behaviorNames()[0] || null);
}

function closeLibrary() {
  $("bhv-modal").hidden = true;
  refreshBehavior();
}

function refreshLibrary(pick) {
  const names = behaviorNames();
  libSelected = pick && names.includes(pick) ? pick : null;
  $("bhv-list").innerHTML = names
    .map((n) => `<option value="${esc(n)}"${n === libSelected ? " selected" : ""}>${esc(n)}</option>`)
    .join("");
  $("bhv-name").value = libSelected || "";
  $("bhv-json").value = libSelected
    ? JSON.stringify(getBehaviorDef(libSelected), null, 2)
    : "";
  $("bhv-error").textContent = "";
  $("btn-bhv-delete").disabled = !libSelected;
}

$("btn-bhv-library").addEventListener("click", openLibrary);
$("btn-bhv-close").addEventListener("click", closeLibrary);
$("bhv-list").addEventListener("change", (e) => refreshLibrary(e.target.value));
$("btn-bhv-new").addEventListener("click", () => {
  libSelected = null;
  $("bhv-list").value = "";
  $("bhv-name").value = "";
  $("bhv-json").value = "{\n  \n}";
  $("bhv-error").textContent = "";
  $("btn-bhv-delete").disabled = true;
  $("bhv-name").focus();
});

$("btn-bhv-save").addEventListener("click", () => {
  const name = $("bhv-name").value.trim();
  if (!name) { $("bhv-error").textContent = "A behaviour needs a name."; return; }
  let body;
  try {
    body = JSON.parse($("bhv-json").value || "{}");
  } catch (err) {
    // the parser's own message says where, which is the whole value of showing it
    $("bhv-error").textContent = `Not valid JSON — ${err.message}`;
    return;
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    $("bhv-error").textContent = "The body must be a JSON object, like { \"dynamic\": true }.";
    return;
  }
  // renaming in place carries every entity reference with it, rather than
  // leaving them pointing at a name that no longer exists
  if (libSelected && libSelected !== name) renameBehaviorDef(libSelected, name);
  setBehaviorDef(name, body);
  refreshLibrary(name);
  setStatus(`saved behaviour "${name}"`);
});

$("btn-bhv-delete").addEventListener("click", () => {
  if (!libSelected) return;
  const gone = libSelected;
  deleteBehaviorDef(gone);
  refreshLibrary(behaviorNames()[0] || null);
  setStatus(`deleted behaviour "${gone}" and every use of it`);
});

/**
 * Write a value into an inspector field - unless you are typing in it.
 *
 * Editing a field applies the value and re-selects, which refreshes the whole
 * inspector; rewriting the focused field with the parsed number then destroys
 * whatever is half-typed. "1." parses as 1 and comes back as "1", so the
 * decimal point vanishes as fast as you type it, and the same goes for a
 * trailing zero in "0.05".
 */
function setField(el, v) {
  if (el === document.activeElement) return;
  el.value = v;
}

/**
 * World-space bounding box of the selection, in metres. It is the *world* box,
 * so it changes as things are rotated - which is the number you want when
 * checking whether a piece still fits its tile.
 */
function refreshDimensions() {
  let min = null, max = null;
  let counted = 0;
  for (const id of state.selection) {
    const e = entryOf(id);
    const b = e && worldBounds(e.node);
    if (!b) continue;
    counted++;
    min = min ? BABYLON.Vector3.Minimize(min, b.min) : b.min.clone();
    max = max ? BABYLON.Vector3.Maximize(max, b.max) : b.max.clone();
  }
  const fmt = (v) => (v < 0.1 ? v.toFixed(3) : v.toFixed(2));
  if (!min) {
    for (const id of ["dim-x", "dim-y", "dim-z"]) $(id).textContent = "—";
    $("dim-note").textContent = "";
    return;
  }
  const size = max.subtract(min);
  $("dim-x").textContent = fmt(size.x);
  $("dim-y").textContent = fmt(size.y);
  $("dim-z").textContent = fmt(size.z);
  $("dim-x").title = `X ${size.x} m`;
  $("dim-y").title = `Y ${size.y} m`;
  $("dim-z").title = `Z ${size.z} m`;
  $("dim-note").textContent = counted > 1 ? `— combined, ${counted} objects` : "— metres, X/Y/Z";
}

function fillChunkSelect(sel, value) {
  sel.innerHTML = "";
  for (const c of ["", ...state.chunks]) {
    const o = document.createElement("option");
    o.value = c;
    o.textContent = c || "(auto)";
    sel.appendChild(o);
  }
  sel.value = state.chunks.includes(value) ? value : "";
}

/**
 * One undo entry per visit to a field, not one per keystroke.
 *
 * `input` fires on every character, so typing "12.5" used to push four
 * snapshots and cost four `Ctrl+Z` to take back, which is also how the old
 * 80-deep stack quietly dropped real edits. Armed on focus, spent on the first
 * edit.
 */
let inspectorPushed = false;

function applyInspector(source) {
  if (syncing || !state.selection.length) return;
  const e = entryOf(state.selection[0]);
  if (!e) return;
  if (!inspectorPushed) { pushUndo(); inspectorPushed = true; }

  if (source === "scale" && $("scl-uniform").checked) {
    const changed = sclIn.find((el) => el === document.activeElement);
    const v = changed && num(changed, null);
    // only mirror onto the *other* fields: rewriting the one being typed in
    // would fight the caret and make a leading "-" impossible to enter
    if (v !== null && v !== undefined) {
      sclIn.forEach((el) => { if (el !== changed) el.value = v; });
    }
  }
  const p = e.node.position, s = e.node.scaling, r = eulerOf(e.node);
  e.node.position.set(
    num(posIn[0], p.x), num(posIn[1], p.y), num(posIn[2], p.z));
  setEuler(e.node, [num(rotIn[0], r[0]), num(rotIn[1], r[1]), num(rotIn[2], r[2])]);
  // Doors scale too: their exported width/height fold the node scale in, and
  // portalOf() already reads the world matrix, so the portal follows.
  e.node.scaling.set(
    num(sclIn[0], s.x), num(sclIn[1], s.y), num(sclIn[2], s.z));
  select([e.id]);                       // keeps the outline in step
  emit("transform");
}

/**
 * Read a numeric field, keeping the old value while the entry is incomplete.
 * A bare "-" parses as NaN, and coercing that to a default is what used to make
 * negative (mirrored) scales impossible to type.
 */
function num(el, fallback) {
  const v = parseFloat(el.value);
  return Number.isFinite(v) ? v : fallback;
}

for (const el of [...posIn, ...rotIn]) el.addEventListener("input", () => applyInspector("t"));
for (const el of sclIn) el.addEventListener("input", () => applyInspector("scale"));
for (const el of [...posIn, ...rotIn, ...sclIn]) {
  el.addEventListener("focus", () => { inspectorPushed = false; });
}

$("insp-chunk").addEventListener("change", (ev) => assignSelectionToChunk(ev.target.value));
$("btn-duplicate").addEventListener("click", () => duplicateCurrent());
$("btn-delete").addEventListener("click", () => removeSelected());
$("btn-focus").addEventListener("click", () => focusSelection());
$("btn-ground").addEventListener("click", () => {
  if (!state.selection.length) return;
  pushUndo();
  // "floor" is wherever the build plane currently is, so this doubles as
  // "rest this on the ceiling" once the plane has been raised
  for (const id of state.selection) {
    const e = entryOf(id);
    if (!e || e.type) continue;
    const b = worldBounds(e.node);
    if (b) e.node.position.y += state.gridY - b.min.y;
  }
  refreshInspector();
  emit("transform");
  setStatus(`dropped ${state.selection.length} object(s) onto y = ${state.gridY.toFixed(2)} m`);
});

// ---------------------------------------------------------------- markers

function armMarker(kind, btn) {
  setBrush(null);
  cancelGhost();
  const already = state.markerBrush === kind;
  state.markerBrush = already ? null : kind;
  for (const b of document.querySelectorAll("#marker-tools button")) b.classList.remove("active");
  if (!already) btn.classList.add("active");
  // The hint carries transient "you are mid-gesture" guidance only - there is
  // no idle text, because a permanent instruction just goes stale.
  $("hint").textContent = state.markerBrush
    ? `Click the grid to drop the ${kind.replace(":", " ")} marker. Esc to stop.`
    : "";
}

$("btn-door").addEventListener("click", (e) => armMarker("door", e.currentTarget));

$("btn-door-sel").addEventListener("click", () => {
  const d = doorFromSelection();
  if (d) { select([d.id]); setStatus(`created ${d.id} from ${d.leaves.length} placement(s)`); }
  else setStatus("select the door geometry first");
});

on("markerdrop", ({ position }) => {
  select([addDoor(position).id]);
  clearMarkerBrush();
});

function clearMarkerBrush() {
  state.markerBrush = null;
  for (const b of document.querySelectorAll("#marker-tools button")) b.classList.remove("active");
}

for (const [id, key] of [["door-trig", "triggerRadius"], ["door-slide", "slideDistance"]]) {
  $(id).addEventListener("input", () => {
    const e = entryOf(state.selection[0]);
    if (!e || e.type !== "door" || syncing) return;
    pushUndo();
    e[key] = parseFloat($(id).value) || 0;
  });
}
for (const id of ["door-w", "door-h"]) {
  $(id).addEventListener("change", () => {
    const e = entryOf(state.selection[0]);
    if (!e || e.type !== "door" || syncing) return;
    pushUndo();
    resizeDoor(e, parseFloat($("door-w").value) || 1, parseFloat($("door-h").value) || 1);
  });
}
for (const [id, key] of [["door-a", "chunkA"], ["door-b", "chunkB"]]) {
  $(id).addEventListener("change", () => {
    const e = entryOf(state.selection[0]);
    if (!e || e.type !== "door" || syncing) return;
    pushUndo();
    e[key] = $(id).value;
    validate();
  });
}
$("btn-door-leaves").addEventListener("click", () => {
  const door = entryOf(state.selection[0]);
  if (!door || door.type !== "door") return;
  const leaves = state.selection.filter((id) => state.placements.has(id));
  if (!leaves.length) { setStatus("select the leaf placements too, then click again"); return; }
  pushUndo();
  door.leaves = leaves;
  refreshInspector();
  setStatus(`${door.id}: ${leaves.length} leaf/leaves assigned`);
});

// ----------------------------------------------------------------- chunks

function refreshChunks() {
  for (const sel of [$("chunk-select"), $("insp-chunk")]) {
    const prev = sel.value;
    sel.innerHTML = "";
    for (const c of state.chunks) {
      const o = document.createElement("option");
      o.value = o.textContent = c;
      sel.appendChild(o);
    }
    if (state.chunks.includes(prev)) sel.value = prev;
  }
  $("chunk-select").value = state.activeChunk;
  refreshStats();
}

function refreshStats() {
  const counts = new Map(state.chunks.map((c) => [c, 0]));
  for (const p of state.placements.values()) {
    counts.set(p.chunk, (counts.get(p.chunk) || 0) + 1);
  }
  const rows = [...counts.entries()].map(([c, n]) =>
    `<tr class="${c === state.activeChunk ? "active" : ""}"><td>${c}</td><td class="n">${n}</td></tr>`);
  $("chunk-stats").innerHTML = `<table>${rows.join("")}</table>`;
  const doors = [...state.markers.values()].filter((m) => m.type === "door").length;
  const veiled = veilCounts();
  const pct = Math.round(state.veilAlpha * 100);
  statusCounts.textContent =
    `${state.placements.size} objects · ${state.chunks.length} chunks · ${doors} doors · ${state.selection.length} selected`
    + (veiled.ghost ? ` · ${veiled.ghost} at ${pct}%` : "")
    + (veiled.hidden ? ` · ${veiled.hidden} hidden` : "");
}

$("chunk-select").addEventListener("change", (ev) => {
  state.activeChunk = ev.target.value;
  applyVisibility();
  refreshStats();
});
$("btn-chunk-add").addEventListener("click", () => {
  const n = String(state.chunks.length).padStart(2, "0");
  const name = prompt("New chunk id", `CH${n}_New`);
  if (name && addChunk(name.trim())) {
    state.activeChunk = name.trim();
    refreshChunks();
  }
});
$("btn-chunk-rename").addEventListener("click", () => {
  const from = state.activeChunk;
  const to = prompt(`Rename "${from}" to`, from);
  if (to === null) return;
  if (renameChunk(from, to)) {
    refreshChunks();
    refreshInspector();
    setStatus(`renamed ${from} → ${to.trim()}`);
  } else {
    setStatus(`could not rename ${from} — "${to.trim()}" is empty or already used`);
  }
});
$("btn-chunk-assign").addEventListener("click", () => assignSelectionToChunk(state.activeChunk));

$("insp-name").addEventListener("change", (e) => {
  if (syncing || !state.selection.length) return;
  renamePlacement(state.selection[0], e.target.value);
});
// Behaviours are keyed by name, so leaving the field is the moment to go and
// look: type a name another element already uses and its flags appear here.
$("insp-name").addEventListener("blur", () => refreshBehavior());

// ---------------------------------------------------------------- toolbar

$("snap-pos").addEventListener("change", (e) => { state.snap.pos = +e.target.value; refreshHud(); });

/**
 * Walk one of the snap dropdowns.
 *
 * The options are read off the select rather than duplicated here, so adding
 * one to the markup is the whole change - which is exactly how `0.1` came to
 * exist. Shared by all three, because all three are now on a key.
 */
function cycleSnap(id, key, dir) {
  const el = $(id);
  const values = [...el.options].map((o) => o.value);
  const i = values.indexOf(String(state.snap[key]));
  const next = values[((i < 0 ? 0 : i) + dir + values.length) % values.length];
  el.value = next;
  state.snap[key] = +next;
  refreshHud();
  return +next;
}

function cycleRotOrScaleAxis(which) {
  if (which === "rot") {
    cycleRotAxis(1);
    setStatus(`rotate about ${state.rotAxis.toUpperCase()}`);
  } else {
    cycleScaleAxis(1);
    setStatus(`scale on ${state.scaleAxis === "all" ? "all axes" : state.scaleAxis.toUpperCase()}`);
  }
}

function cycleMoveSnap(dir) {
  const v = cycleSnap("snap-pos", "pos", dir);
  setStatus(v ? `move step ${v} m` : "move step off — free positioning");
}

function cycleRotSnap(dir) {
  setStatus(`rotate by ${cycleSnap("snap-rot", "rot", dir)}°`);
}

function cycleScaleSnap(dir) {
  setStatus(`scale step ${cycleSnap("snap-scale", "scale", dir)}`);
}
$("snap-rot").addEventListener("change", (e) => { state.snap.rot = +e.target.value; refreshHud(); });
$("snap-scale").addEventListener("change", (e) => { state.snap.scale = +e.target.value; refreshHud(); });
$("rot-axis").addEventListener("change", (e) => { state.rotAxis = e.target.value; refreshHud(); });
$("scale-axis").addEventListener("change", (e) => { state.scaleAxis = e.target.value; refreshHud(); });

$("big-palette").addEventListener("change", (e) => setBigPalette(e.target.checked));

// Keyboard shortcuts are ignored while a form control has focus, so a toolbar
// control that keeps focus after being changed silently kills the numpad keys.
for (const el of document.querySelectorAll("#toolbar select, #toolbar input")) {
  el.addEventListener("change", () => el.blur());
}

function setBigPalette(on) {
  document.body.classList.toggle("big-palette", on);
  $("big-palette").checked = on;
  localStorage.setItem("bigPalette", on ? "1" : "0");
  state.engine?.resize();
}

$("env-intensity").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  setEnvIntensity(v);
  $("env-intensity-val").textContent = v.toFixed(1);
  localStorage.setItem("envIntensity", String(v));
});

$("exposure").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  setExposure(v);
  showExposure(v);
  localStorage.setItem("exposure", String(v));
});

/**
 * How see-through Shift+H makes things.
 *
 * Kept in localStorage rather than the manifest, like the exposure: it says
 * nothing about the ship and everything about how this person likes to look at
 * it, and writing a view preference into the ship's data would be wrong.
 */
$("veil-alpha").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  setVeilAlpha(v);
  showVeilAlpha(v);
  refreshStats();                     // the status bar quotes the percentage
  localStorage.setItem("veilAlpha", String(v));
});

function showVeilAlpha(v) {
  $("veil-alpha-val").textContent = `${Math.round(v * 100)}%`;
}

/**
 * Both readings of the exposure at once.
 *
 * Just the linear multiplier now: it is what the slider sets, what the manifest
 * stores and what the demos apply, so there is only one number to know.
 */
function showExposure(v) {
  $("exposure-val").textContent = v.toFixed(2);
}

/**
 * Follow the lighting when a load, or the light-mode switch, changes it.
 *
 * The sliders are normally the source of truth and persist to localStorage, but
 * a manifest's environment block outranks them: exposure and IBL strength are
 * properties of the ship, not of this browser. Only the *active* pair is
 * remembered locally - the other one lives in the manifest.
 */
function refreshLighting() {
  $("env-intensity").value = String(state.envIntensity);
  $("env-intensity-val").textContent = state.envIntensity.toFixed(1);
  localStorage.setItem("envIntensity", String(state.envIntensity));
  $("exposure").value = String(state.exposure);
  showExposure(state.exposure);
  localStorage.setItem("exposure", String(state.exposure));
}

$("show-grid").addEventListener("change", (e) => setGridVisible(e.target.checked));

// The chunk label doubles as the isolate toggle: pressed, everything outside
// the active chunk is hidden. State lives in aria-pressed so the button cannot
// look pressed while the scene says otherwise.
$("btn-isolate").addEventListener("click", () => {
  setIsolate(!state.isolate);
  setStatus(state.isolate
    ? `isolated — showing only ${state.activeChunk}`
    : "showing every chunk");
});

function setIsolate(on) {
  state.isolate = !!on;
  $("btn-isolate").setAttribute("aria-pressed", state.isolate ? "true" : "false");
  applyVisibility();
}

$("unlit").addEventListener("change", (e) => {
  setUnlit(e.target.checked);
  localStorage.setItem("unlit", e.target.checked ? "1" : "0");
  setStatus(e.target.checked
    ? "unlit — raw albedo, no lighting"
    : "lit — hemi + key/fill + IBL");
});

$("runtime-light").addEventListener("change", (e) => {
  setRuntimeLighting(e.target.checked);
  localStorage.setItem("runtimeLight", e.target.checked ? "1" : "0");
  // the mode swaps in its own Env/Exposure pair, so the sliders must follow
  refreshLighting();
  refreshHud();
  setStatus(e.target.checked
    ? "runtime light — HDRI only, with the values the demos read"
    : "editor light — the authoring rig and its own Env/Exposure are back");
});

$("walk").addEventListener("change", (e) => {
  setWalk(e.target.checked);
  setStatus(e.target.checked
    ? `walking at ${EYE_HEIGHT} m — WASD only, height follows the floor`
    : "flying — Space/C change height");
});

/** How each drag axis reads in the toolbar, the HUD and the status line. */
const DRAG_AXIS_LABEL = {
  xz: "X/Z (floor)",
  y: "Y (up/down)",
  x: "X only",
  z: "Z only",
};

/**
 * V cycles the drag axis. The combo and the status line follow - a mode you
 * cannot see is a mode you will forget you are in, and this one silently
 * changes what every drag does.
 */
function setDragAxisFromKey() {
  const axis = toggleDragAxis();
  refreshDragAxis();
  setStatus(`drag moves ${DRAG_AXIS_LABEL[axis]} — press V for the next axis`);
}

function refreshDragAxis() {
  $("drag-axis").value = state.dragAxis;
}

$("drag-axis").addEventListener("change", (e) => {
  setDragAxis(e.target.value);
  refreshDragAxis();
  setStatus(`drag moves ${DRAG_AXIS_LABEL[state.dragAxis]}`);
});

$("btn-select-rect").addEventListener("click", () => {
  const on = setSelectMode(!state.selectMode);
  $("btn-select-rect").setAttribute("aria-pressed", on ? "true" : "false");
  setStatus(on
    ? "rectangle select — drag to select, Ctrl/Shift adds"
    : "drag moves elements again");
});

$("btn-save").addEventListener("click", doSave);$("btn-load").addEventListener("click", doLoad);
$("btn-export").addEventListener("click", doExport);

async function doSave() {
  try {
    setStatus("saving…");
    const r = await saveLayout();
    setStatus(r.previous
      ? `saved ${r.bytes} bytes → ${r.path} (previous kept as ${r.previous})`
      : `saved ${r.bytes} bytes → ${r.path}`);
  } catch (e) { setStatus("save failed: " + e.message); }
}

async function doLoad() {
  try {
    setStatus("loading…");
    const data = await loadLayout();
    setStatus(data
      ? `loaded ${data.instances.length} instances`
      : "nothing to load (no tool-written manifest yet)");
    refreshChunks();
  } catch (e) { setStatus("load failed: " + e.message); }
}

async function doExport() {
  try {
    setStatus("exporting glb…");
    const r = await exportGlb();
    setStatus(`exported ${(r.bytes / 1048576).toFixed(1)} MB → ${r.path}`);
  } catch (e) {
    console.error(e);
    setStatus("export failed: " + e.message);
  }
}

// --------------------------------------------------------------- keyboard

window.addEventListener("keydown", async (e) => {
  // Every shortcut is an edit or a mode change, and the scene is mid-rebuild.
  if (isBusy()) { e.preventDefault(); return; }
  // The behaviour library is modal: nothing behind it should be editable, and
  // a stray X or Del while a button in it holds focus would act on the ship.
  if (!$("bhv-modal").hidden) {
    if (e.key === "Escape") { e.preventDefault(); closeLibrary(); }
    return;
  }
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) {
    if (e.key === "Escape") t.blur();
    return;
  }
  const mod = e.ctrlKey || e.metaKey;

  // Windows/Chromium hands Alt to the browser menu bar, which then swallows the
  // next keystrokes and silently kills WASD. Alt is ours (eyedropper, pivot
  // rotate), so claim it outright.
  if (e.key === "Alt") { e.preventDefault(); noteKey(e.code, true, e); return; }

  if (mod && e.key.toLowerCase() === "s") { e.preventDefault(); return doSave(); }
  if (mod && e.key.toLowerCase() === "d") { e.preventDefault(); return duplicateCurrent(); }
  if (mod && e.key.toLowerCase() === "z") { e.preventDefault(); return e.shiftKey ? redo() : undo(); }
  if (mod && e.key.toLowerCase() === "y") { e.preventDefault(); return redo(); }
  // Ctrl + the key that uses a setting cycles that setting's *value*; Shift +
  // the same key cycles its *axis*. R turns, F mirrors, V drags, and each keeps
  // its own settings behind it - so everything about rotation is on R, and
  // everything about translation on V.
  // preventDefault is not optional on these: Ctrl+F is the browser's find bar,
  // Ctrl+R reloads the page, and Ctrl+V pastes. All three *can* be claimed -
  // Ctrl+R confirmed by hand, which is the only way to check: Playwright
  // injects keys through CDP, so a genuinely reserved shortcut still looks
  // claimed to a test.
  if (mod && e.key.toLowerCase() === "r") {
    e.preventDefault();
    return cycleRotSnap(e.shiftKey ? -1 : 1);
  }
  if (mod && e.key.toLowerCase() === "f") {
    e.preventDefault();
    return cycleScaleSnap(e.shiftKey ? -1 : 1);
  }
  // Ctrl+V walks the move step back, the direction Shift+V does not go.
  // Ctrl+T would have been the tidier pair with V's neighbour T, but it is a
  // *reserved* browser shortcut - Chrome and Edge open a new tab before the
  // page sees the key, and preventDefault has no effect on it.
  if (mod && e.key.toLowerCase() === "v") {
    e.preventDefault();
    return cycleMoveSnap(-1);
  }
  if (mod) return;

  // WASD drives the orbit camera; the fly camera binds them itself. Shift is
  // read from every key event so the 2x boost can be toggled mid-slide.
  if (noteKey(e.code, true, e)) { e.preventDefault(); return; }

  // numpad drives the build plane, so the mouse never has to leave the model.
  // The main-row keys are accepted too, for keyboards without a numpad.
  switch (e.code) {
    case "NumpadAdd": case "Equal":
      e.preventDefault(); return nudgeGridElevation(1);
    case "NumpadSubtract": case "Minus":
      e.preventDefault(); return nudgeGridElevation(-1);
    case "NumpadDecimal": e.preventDefault(); return elevationFromHover();
  }

  const step = state.snap.pos || 1;
  switch (e.key) {
    case "ArrowLeft": e.preventDefault(); return nudgeSelection(new BABYLON.Vector3(-step, 0, 0));
    case "ArrowRight": e.preventDefault(); return nudgeSelection(new BABYLON.Vector3(step, 0, 0));
    case "ArrowUp": e.preventDefault(); return nudgeSelection(new BABYLON.Vector3(0, 0, step));
    case "ArrowDown": e.preventDefault(); return nudgeSelection(new BABYLON.Vector3(0, 0, -step));
    case "PageUp": e.preventDefault(); return nudgeSelection(new BABYLON.Vector3(0, step, 0));
    case "PageDown": e.preventDefault(); return nudgeSelection(new BABYLON.Vector3(0, -step, 0));
    // F mirrors; Shift+F picks the axis it mirrors on (and that Shift+wheel
    // resizes on). The action and its two settings share one letter.
    case "f": case "F": {
      e.preventDefault();
      if (e.shiftKey) {
        cycleRotOrScaleAxis("scale");
        break;
      }
      const r = flipCurrent();
      setStatus(r
        ? `mirrored ${r.count} object(s) on ${r.axis.toUpperCase()}`
        : "nothing to mirror — select or hover an element first");
      break;
    }
    // R turns the current element, Shift+R picks the axis it turns about, and
    // Alt+R swings the whole selection about a shared pivot. Everything about
    // rotation is on one key, the way everything about translation is on V.
    case "r": case "R":
      e.preventDefault();
      if (e.shiftKey) cycleRotOrScaleAxis("rot");
      else rotateCurrent(1, e.altKey);
      break;
    // V is the drag axis; Shift+V walks the move step, Ctrl+V walks it back.
    //
    // Not Q: these letter cases match e.key (the label), while WASD matches
    // e.code (the physical position, so the keys stay under the same fingers on
    // any layout). On AZERTY those two collide exactly on Q - the key labelled
    // Q sits where QWERTY has A, so it arrives as code "KeyA", gets claimed as
    // strafe-left, and never reaches this switch. V is in the same place and
    // carries the same label on both layouts, and reads as "vertical".
    case "v": case "V": {
      e.preventDefault();
      if (e.shiftKey) cycleMoveSnap(1);
      else setDragAxisFromKey();
      break;
    }
    // X shows one element's axes in world space; Shift+X in its own local
    // space, which is the one that matters for scaling - scaling is local, so
    // on anything that has been turned a world gizmo cannot say which way X
    // grows. The armed ghost counts as an element: you set its rotation and
    // mirroring before dropping it, which is exactly when you need to see them.
    // With several selected the nearest to the cursor wins - "the one I am
    // looking at" is the only reading of a multi-selection that does not need a
    // second key to disambiguate.
    case "x": case "X": {
      e.preventDefault();
      const space = e.shiftKey ? "local" : "world";
      const ids = state.selection.filter((id) => entryOf(id));
      const target = ghostActive() ? GHOST_AXES
        : ids.length ? nearestToCursor(ids) : hoveredId();
      if (!target) {
        // Nothing to point at means nothing to show. Leaving the previous
        // element's gizmo up would leave it hanging off something you are no
        // longer working on, with no key that clears it.
        setStatus(hideAxes()
          ? "axes hidden — nothing selected or hovered"
          : "no element to show axes for — select or point at one");
        break;
      }
      const shown = toggleAxes(target, space);
      const label = target === GHOST_AXES
        ? (ghostModule() || "ghost")
        : (entryOf(target)?.name || entryOf(target)?.module || target);
      setStatus(shown
        ? `${space} axes on ${label} — ${space === "local" ? "Shift+X" : "X"} hides them`
        : "axes hidden");
      break;
    }
    // Shift+H parks the selection out of sight so you can reach what is behind
    // it; plain H is the way back. That way round because an accidental H on a
    // large selection is expensive and an accidental unhide costs nothing.
    // Either way it reports the count - a hide you forgot about looks exactly
    // like an element you deleted by mistake.
    case "h": case "H": {
      e.preventDefault();
      if (e.shiftKey) {
        const r = hideSelected();
        setStatus(r
          ? (r.level === "ghost"
            ? `${r.count} element(s) at 50% and click-through — Shift+H again to hide outright`
            : `hid ${r.count} element(s) — Shift+H returns them to 50%, H brings everything back`)
          : "nothing to hide — select something first");
      } else {
        const n = unhideAll();
        setStatus(n ? `unhid ${n} element(s)` : "nothing was hidden");
      }
      break;
    }
    case "Delete": case "Backspace": deleteCurrent(); break;
    case "Escape": cancelEverything(); break;
    case "g": case "G": {
      const c = $("show-grid");
      c.checked = !c.checked;
      setGridVisible(c.checked);
      break;
    }
  }
});

window.addEventListener("keyup", (e) => {
  // menu activation happens on the Alt *release*, so it has to be claimed here too
  if (e.key === "Alt") e.preventDefault();
  noteKey(e.code, false, e);
});
// a lost focus never delivers keyup, which would leave the camera drifting
window.addEventListener("blur", () => releaseAllKeys());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) releaseAllKeys();
});

/** Right mouse button does the same as Escape. */
on("escape", () => cancelEverything());

function cancelEverything() {
  if (cancelMarquee()) return;     // an in-flight rectangle goes first
  if (cancelDrag()) return;        // then an in-flight drag
  cancelGhost();
  setBrush(null);
  clearMarkerBrush();
  select([]);
}

/**
 * Ctrl+D arms a ghost holding a copy of the current element, rather than
 * dropping one beside it: you almost always want the copy somewhere specific,
 * and an in-place duplicate then has to be dragged there anyway. The ghost
 * keeps the source's rotation and scale, so a mirrored or turned piece copies
 * as it looks.
 *
 * A ghost can only hold one module, so a multi-selection still duplicates in
 * place - there is nothing sensible to attach to the cursor.
 */
function duplicateCurrent() {
  if (ghostActive() || isDragging()) return;
  const cur = currentElement();
  if (!cur || cur.ids.length !== 1) return duplicateSelected();

  const entry = entryOf(cur.ids[0]);
  if (!entry?.module) return duplicateSelected();   // markers have no module

  // The ghost sits on the build plane, so without this the copy of something on
  // an upper deck would appear back down at ground level. Moving the plane
  // rather than giving the ghost its own height keeps one source of truth, and
  // the grid visibly follows so it is obvious what happened.
  const y = entry.node.position.y;
  if (Math.abs(y - state.gridY) > 1e-6) setGridElevation(y);

  setBrush(entry.module, {
    rotation: eulerOf(entry.node),
    scaling: entry.node.scaling.asArray(),
  });
  setStatus(`copy of ${entry.module} on the cursor at ${y.toFixed(2)} m — click to place`);
}

/**
 * Delete whatever the cursor is over; with nothing under it, delete the
 * selection. Deleting a hovered element leaves an unrelated selection intact -
 * pointing at one thing is no reason to forget the others. Nothing mid-gesture
 * is deletable - Esc is the way out of those.
 */
function deleteCurrent() {
  if (ghostActive() || isDragging()) return;
  const id = hoveredId();
  if (id) {
    const keep = state.selection.filter((s) => s !== id);
    select([id]);
    removeSelected();
    if (keep.length) select(keep);
    return;
  }
  if (state.selection.length) removeSelected();
}

/** Put the build plane on top of whatever the cursor is over. */
function elevationFromHover() {
  const cur = currentElement();
  if (!cur || cur.kind !== "hover") { setStatus("hover an element first"); return; }
  const entry = entryOf(cur.ids[0]);
  const b = entry && worldBounds(entry.node);
  if (!b) return;
  setGridElevation(b.max.y);
  setStatus(`build plane raised to ${b.max.y.toFixed(2)} m`);
}

/** Delete under the cursor if there is something there, else the selection. */
// ------------------------------------------------------------------- HUD

/**
 * Two rows only.
 *
 * Everything else the HUD used to carry - rotation axis, scale axis, fly speed,
 * drag axis, lighting mode - is already on screen in the toolbar, whose combos
 * *are* those readouts. The build plane is not: it is a mode with no control of
 * its own, changed only by the numpad, and invisible until something lands at
 * the wrong height. The current element is not either: it is derived from the
 * ghost, the hover and the selection, and answers "what will the next key act
 * on".
 */
function refreshHud() {
  $("hud-elev").textContent = `${state.gridY.toFixed(2)} m`;
  refreshDragAxis();
  $("rot-axis").value = state.rotAxis;
  $("scale-axis").value = state.scaleAxis;

  const cur = currentElement();
  $("hud-current").textContent = cur
    ? `${cur.kind === "ghost" ? "◆" : cur.kind === "selection" ? "■" : "▸"} ${short(cur.module)}`
    : "—";
  $("hud-current").title = cur?.module || "";
}

function short(s) {
  if (!s) return "—";
  const name = s.includes("/") ? s.slice(s.indexOf("/") + 1) : s;
  return name.length > 22 ? name.slice(0, 21) + "…" : name;
}

// ------------------------------------------------------------- validation

function validate() {
  const out = [];
  const boxes = [];
  for (const c of state.chunks) {
    const members = [...state.placements.values()].filter((p) => p.chunk === c);
    if (!members.length) continue;
    let min = null, max = null;
    for (const m of members) {
      const b = worldBounds(m.node);
      if (!b) continue;
      min = min ? BABYLON.Vector3.Minimize(min, b.min) : b.min.clone();
      max = max ? BABYLON.Vector3.Maximize(max, b.max) : b.max.clone();
    }
    if (min) boxes.push({ id: c, min, max });
  }

  // Portal rendering needs unambiguous chunk membership, so overlapping chunk
  // volumes are a real defect rather than a cosmetic one.
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      const ov = ["x", "y", "z"].every((k) =>
        Math.min(a.max[k], b.max[k]) - Math.max(a.min[k], b.min[k]) > 0.05);
      if (ov) out.push(["warn", `${a.id} overlaps ${b.id}`]);
    }
  }

  const sunk = [...state.placements.values()].filter((p) => {
    const b = worldBounds(p.node);
    return b && b.min.y < -0.05;
  });
  if (sunk.length) out.push(["warn", `${sunk.length} object(s) below y = 0`]);

  const offGrid = [...state.placements.values()].filter((p) => {
    const s = state.snap.pos || 0;
    if (!s) return false;
    return ["x", "z"].some((k) => Math.abs(p.node.position[k] / s - Math.round(p.node.position[k] / s)) > 1e-3);
  });
  if (offGrid.length) out.push(["warn", `${offGrid.length} object(s) off the ${state.snap.pos} m grid`]);

  // A portal is only usable if both of its sides are known, so an unresolved
  // door is an error rather than a warning. Check the *resolved* sides: most
  // doors are left on "(auto)".
  const doors = [...state.markers.values()].filter((m) => m.type === "door");
  const linked = new Set();
  for (const d of doors) {
    const [a, b] = resolveDoorChunks(d, boxes);
    linked.add(a); linked.add(b);
    if (!a || !b) out.push(["err", `${d.id}: only one side resolves (${a || b || "none"})`]);
    else if (a === b) out.push(["err", `${d.id}: both sides resolve to ${a}`]);
    if (!d.leaves.length) out.push(["warn", `${d.id}: no leaves assigned`]);
  }
  const orphans = state.chunks.filter((c) =>
    !linked.has(c) && [...state.placements.values()].some((p) => p.chunk === c));
  if (orphans.length && state.chunks.length > 1) {
    out.push(["warn", `unreachable: ${orphans.join(", ")}`]);
  }
  if (!doors.length && state.chunks.length > 1) out.push(["warn", "no doors placed yet"]);

  const el = $("validation");
  el.innerHTML = out.length
    ? out.map(([k, m]) => `<div class="${k}">${m}</div>`).join("")
    : `<div class="ok">All checks pass.</div>`;
}

// ------------------------------------------------------------------- boot

on("selection", () => { refreshInspector(); refreshStats(); });
on("placements", () => { refreshChunks(); refreshStats(); validate(); });
on("chunks", refreshChunks);
on("markers", () => { refreshStats(); validate(); });
on("transform", () => { refreshInspector(); validate(); });
on("grid", refreshHud);
on("modes", refreshHud);
on("current", refreshHud);
on("environment", refreshLighting);
on("behaviors", () => {
  refreshBehavior();
  if (!$("bhv-modal").hidden) refreshLibrary(libSelected);
});
on("busy", refreshBusy);

/**
 * Lock the UI while the scene is being rebuilt.
 *
 * Two layers, because either alone leaks. The overlay is what you see and it
 * swallows the pointer, but a keyboard shortcut or a Tab into a toolbar select
 * would sail straight past it — so the panels are also marked `inert`, which
 * takes them out of hit-testing *and* out of the focus order in one attribute.
 * The keydown handler bails as well, since window-level shortcuts never had a
 * target inside the inert subtree to begin with.
 */
function refreshBusy() {
  const busy = isBusy();
  $("busy").hidden = !busy;
  $("busy-msg").textContent = busyLabel() || "working…";
  for (const id of ["toolbar", "palette", "viewport", "inspector"]) {
    const el = $(id);
    if (el) el.inert = busy;
  }
  if (busy) document.activeElement?.blur();
}
on("pickmodule", (moduleId) => { setBrush(moduleId); setStatus(`armed ${moduleId}`); });
on("status", (msg) => setStatus(msg));
on("focus", () => focusSelection());

// The overlay is up in the markup already, so there is never a frame in which
// a half-built editor looks ready to use. whileBusy() takes it from here.
(function boot() {
  return whileBusy("loading…", bootstrap);
})();

async function bootstrap() {
  setStatus("loading catalogue…");
  setBigPalette(localStorage.getItem("bigPalette") !== "0");
  if (localStorage.getItem("unlit") === "1") {
    $("unlit").checked = true;
  }
  if (localStorage.getItem("runtimeLight") === "1") {
    $("runtime-light").checked = true;
    state.runtimeLight = true;    // read by initScene when it builds the rig
  }
  {
    const saved = parseFloat(localStorage.getItem("envIntensity"));
    const v = Number.isFinite(saved) ? saved : ENV_INTENSITY_DEFAULT;
    $("env-intensity").value = String(v);
    $("env-intensity-val").textContent = v.toFixed(1);
    state.envIntensity = v;
    activeLightSet().strength = v;
  }
  {
    const saved = parseFloat(localStorage.getItem("exposure"));
    const v = Number.isFinite(saved) ? saved : EXPOSURE_DEFAULT;
    $("exposure").value = String(v);
    showExposure(v);
    state.exposure = v;
    activeLightSet().exposure = v;
  }
  {
    const saved = parseFloat(localStorage.getItem("veilAlpha"));
    const v = Number.isFinite(saved) ? saved : state.veilAlpha;
    $("veil-alpha").value = String(v);
    showVeilAlpha(v);
    setVeilAlpha(v);
  }
  if (localStorage.getItem("lockPan") === "1") {
    // the setting is gone: right-drag looks now, and nothing pans
    localStorage.removeItem("lockPan");
  }
  // RMB+WASD no longer does anything special either
  localStorage.removeItem("rmbSteer");
  await loadCatalogue();
  state.kitDir = getCatalogue().kitDir;
  state.fluidSim = getCatalogue().fluidSim || [];

  await initScene($("render-canvas"));
  // Returning to the viewport must restore keyboard control: a toolbar select
  // that still holds focus swallows every shortcut, numpad included.
  $("render-canvas").addEventListener("pointerdown", () => {
    const a = document.activeElement;
    if (a && $("toolbar").contains(a)) a.blur();
  });
  initInteract();
  if ($("unlit").checked) setUnlit(true);
  await initThumbs();
  initPalette();
  refreshChunks();
  refreshHud();

  const data = await loadLayout().catch(() => null);
  if (data) {
    refreshChunks();
    setStatus(data.legacySpawns
      ? `restored ${data.instances.length} instances — the old player/weapon spawns were `
        + "dropped: give a dummy element a start-position behaviour instead"
      : `restored ${data.instances.length} instances from ship_manifest.json`);
  } else {
    setStatus(`ready — ${getCatalogue().byId.size} modules`);
  }
  validate();
}

function round(v) { return Math.round(v * 1000) / 1000; }
