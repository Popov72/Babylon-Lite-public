// Wiring: toolbar, inspector, keyboard, autoload.

import { loadCatalogue, getCatalogue, moduleBounds, instantiate } from "./kit.js";
import { initThumbs } from "./thumbs.js";
import { initPalette, setBrush, refreshCollisionMarks } from "./palette.js";
import {
  saveLayout, loadLayout, loadCollision, saveAutosave, exportGlb, resolveDoorChunks, nodeNameOf,
} from "./manifest.js";
import { addDoor, doorFromSelection, resizeDoor, normalizeDoorSides } from "./markers.js";
import {
  removeCollider, COLLIDER_KINDS, COLLIDER_LABEL, SCALE_RULE, COLLIDER_DEFAULT_SCALE,
  reconcileCollider, colliderDims,
  enterCollisionMode, exitCollisionMode, fitBoxToSelection, fitHullToSelection,
  stageModule, unstageModule, harvestStage, orphanCount,
} from "./colliders.js";
// Side-effect import for the hooks; the named ones drive the inspector.
import {
  LIGHT_TYPES, addLight, duplicateLight, setLightOwner, setLightPart,
} from "./lights.js";
import {
  initInteract, cancelGhost, cancelDrag, isDragging, currentElement,
  ghostActive, ghostModule, ghostCollider, armColliderGhost, colliderHalf, hoveredId,
  cycleRotAxis, cycleScaleAxis, rotateCurrent, flipCurrent,
  toggleDragAxis, setDragAxis, toggleAxisSpace, setAxisSpace, cancelMarquee, grabSelection,
  bringToCamera,
} from "./interact.js";
import {
  state, on, emit, initScene, setGridVisible, setGridElevation,
  nudgeGridElevation, select, removeSelected, duplicateSelected, focusSelection, focusNodes,
  shipPlacements, loadModuleCollision,
  addChunk, assignSelectionToChunk, applyVisibility, undo, redo, pushUndo,
  renameChunk, removeChunk, chunkUsers,
  environmentProbeIds, environmentProbeOf, nextEnvironmentProbeId,
  setEnvironmentProbe, removeEnvironmentProbe, syncEnvironmentProbeTransform,
  syncEnvironmentProbeInfluence, syncEnvironmentProbeInnerSize,
  environmentProbePartOf, environmentProbePartId,
  validEnvironmentProbeId, environmentProbeIdAvailable,
  renamePlacement, hideSelected, unhideAll, hiddenCount, veilCounts,
  setVeilAlpha, SKYBOX_CHUNK,
  getBehaviorDef, setBehaviorDef, renameBehaviorDef, deleteBehaviorDef, behaviorNames,
  entityBehaviors, addEntityBehavior, removeEntityBehavior, setEntityLinked,
  isLiquefiable, setEntityParams, nodeNamesInChunk, nodesNamed,
  isBusy, busyLabel, whileBusy, serialize, cursorOnGrid, hooks,
  toggleAxes, nearestToCursor, hideAxes, GHOST_AXES,
  eulerOf, setEuler, worldBounds, entryOf, nudgeSelection,
  noteKey, releaseAllKeys, setUnlit, EXPOSURE_DEFAULT,
  setConfig, resetConfig, CONFIG_DEFAULTS,
  setWalk, EYE_HEIGHT, ENV_INTENSITY_DEFAULT, setSelectMode,
  setLightSetting, viewMode, viewModeFlags, VIEW_MODES,
  TONE_MAPPING_DEFAULT, RUNTIME_SPECULAR_AA_DEFAULT, RUNTIME_ROUGHNESS_FACTOR_DEFAULT,
  VEIL_ALPHA_DEFAULT, BIG_PALETTE_DEFAULT,
  setShowLayer, SHOW_LAYERS,
  resolveToneMapping, setRuntimeSpecularAA, setRuntimeRoughnessFactor,
} from "./editor.js";
import {
  setRuntimePreview, runtimePreview,
  localEnvironmentProbeOf, showEnvironmentProbe, refreshEnvironmentProbeAssets,
} from "./runtime.js";
import { generateLocalEnvironments } from "./local-environments.js";

const $ = (id) => document.getElementById(id);
const alphabetical = (values) => [...values].sort((a, b) => a.localeCompare(b));
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
// Read from the markup rather than restated here, so the tooltip a door onto
// space temporarily replaces has exactly one author.
const SEALED_TITLE = $("door-sealed").parentElement.title;
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
  const isLight = e.type === "light";
  const isProbe = e.type === "environment-probe";

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
  // A module's primitive belongs to a kit prototype, not a room, so the chunk
  // row is meaningless for it - and letting it be set would silently re-home
  // the shape into a room where its local-space transform means nothing.
  const isModuleCollider = e.type === "collider" && !!e.stage;
  $("insp-chunk").parentElement.hidden = isMarker || isModuleCollider;
  if (!isMarker && !isModuleCollider) $("insp-chunk").value = e.chunk;
  // A light's node hangs off the element it rides, so its "position" is really
  // an offset within that element - and a lamp has no size of its own: an engine
  // light is a point, a direction and a falloff.
  $("insp-h-pos").textContent = isLight ? "Offset (local)" : "Position";
  // A probe's inner blend box has no centre of its own - it is concentric with
  // the outer one - so there is nothing for the row to edit.
  $("position-fields").hidden = e.canMove === false;
  $("scale-fields").hidden = isLight;
  $("rotation-fields").hidden = isProbe;
  const p = e.node.position, r = eulerOf(e.node), s = e.node.scaling;
  posIn.forEach((el, i) => setField(el, round(p.asArray()[i])));
  rotIn.forEach((el, i) => setField(el, round(r[i])));
  sclIn.forEach((el, i) => setField(el, round(s.asArray()[i])));

  $("door-fields").hidden = !isDoor;
  if (isDoor) {
    setField($("door-w"), e.width);
    setField($("door-h"), e.height);
    fillChunkSelect($("door-a"), e.chunkA);
    fillChunkSelect($("door-b"), e.chunkB, { skybox: true });
    setField($("door-trig"), e.triggerRadius);
    setField($("door-slide"), e.slideDistance);
    $("door-enabled").checked = e.enabled !== false;
    // A window onto space is sealed and cannot be anything else, so the box
    // shows the truth but stops being an argument.
    const toSpace = e.chunkB === SKYBOX_CHUNK;
    $("door-sealed").checked = !!e.sealed;
    $("door-sealed").disabled = toSpace;
    $("door-sealed").parentElement.title = toSpace
      ? "Forced: a door onto the skybox opens onto space, which cannot be walked into."
      : SEALED_TITLE;
    $("door-leaves").textContent = e.leaves.length ? e.leaves.join(", ") : "none";
  }

  refreshDimensions();
  refreshBehavior();
  refreshLight(isLight ? e : null);
  syncing = false;
}

/** #rrggbb from a linear-ish [0..1] triple, and back. The picker is 8-bit, so
 *  a round trip quantises - which is why the field is only written from the
 *  record, never from itself. */
function hexOf(c) {
  const b = (v) => Math.max(0, Math.min(255, Math.round((v ?? 0) * 255)));
  return `#${[0, 1, 2].map((i) => b(c?.[i]).toString(16).padStart(2, "0")).join("")}`;
}
function rgbOf(hex) {
  const m = /^#?([\da-f]{6})$/i.exec(hex || "");
  if (!m) return [1, 1, 1];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => Math.round((v / 255) * 1e4) / 1e4);
}

/**
 * The light panel.
 *
 * Rows that cannot apply are disabled rather than hidden, with #lgt-hint saying
 * why: a disabled Cone row still tells you a spot light is the thing that has
 * one. The rules mirror normalizeLight() exactly - this panel must never be
 * able to author a light the model would rewrite behind your back.
 */
function refreshLight(light) {
  $("light-fields").hidden = !light;
  if (!light) return;
  const ownerSelect = $("lgt-owner");
  ownerSelect.replaceChildren();
  for (const owner of shipPlacements()) {
    const option = document.createElement("option");
    option.value = owner.id;
    option.textContent = `${owner.id} — ${owner.module}`;
    ownerSelect.appendChild(option);
  }
  ownerSelect.value = light.owner;
  ownerSelect.title = light.owner;

  const rt = light.runtime;
  setField($("lgt-type"), rt.type);
  setField($("lgt-intensity"), rt.intensity);
  setField($("lgt-range"), rt.range);
  setField($("lgt-angle"), rt.angle);
  $("lgt-run-color").value = hexOf(rt.color);
  $("lgt-clustered").checked = !!rt.clustered;
  $("lgt-shadows").checked = !!rt.castsShadows;

  const live = rt.type !== "none";
  for (const id of ["lgt-intensity", "lgt-run-color"]) $(id).disabled = !live;
  // Range is a real setting on every positioned lamp, and only because the
  // preview materials no longer use the PBR *physical* falloff. Physical is a
  // plain inverse square with no cut-off, so it ignored Range outright - and on
  // a clustered lamp it was worse than ignoring it, because the cluster still
  // sizes and culls the light proxy by Range, leaving the light to stop dead in
  // a straight line where the proxy ended. The preview now asks for the glTF
  // falloff (see materialFor in runtime.js), which windows the light down to
  // nothing at Range, so the field means here what it means in the game.
  //
  // A directional light is the one exception: it has no position, so no
  // distance, so nothing for Range to attenuate.
  $("lgt-range").disabled = !live || rt.type === "directional";
  $("lgt-clustered").disabled = !(rt.type === "point" || rt.type === "spot");
  $("lgt-angle").disabled = rt.type !== "spot";
  $("lgt-shadows").disabled = rt.clustered
    || !(rt.type === "spot" || rt.type === "directional");

  const why = [];
  if (!live) why.push("Switched off: no light is created at runtime.");
  else if (rt.type === "directional") why.push("A directional light has no position to bin, so it cannot be clustered.");
  else if (rt.clustered) why.push("A clustered light carries no shadow map. Uncluster it to cast.");
  else if (rt.type === "point") why.push("A point light cannot cast: there is no cube shadow map.");
  if (live && rt.type === "directional") {
    why.push("A directional light has no distance falloff, so Range does nothing.");
  } else if (live && !rt.clustered) {
    why.push("An unclustered lamp is cut off at Range in the game too, but along a straight ramp rather than this preview's curve.");
  }
  $("lgt-hint").textContent = why.join(" ");
}

/** One undo entry per visit to the panel, matching the transform fields. */
function editLight(part, patch) {
  if (syncing) return;
  const light = entryOf(state.selection[0]);
  if (light?.type !== "light") return;
  if (!inspectorPushed) { pushUndo(); inspectorPushed = true; }
  // No refresh here: setLightPart emits "lights", which the panel follows - and
  // this streams on every keystroke and every tick of the colour picker, so
  // refreshing twice a stroke is a rebuilt owner list for nothing.
  setLightPart(light.id, part, patch);
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

  // The whole library, every time: a behaviour may be attached more than once,
  // so "already attached" is no longer a reason to leave it out of the list.
  $("bhv-add").innerHTML = library.length
    ? library.map((b) => `<option value="${esc(b)}">${esc(b)}</option>`).join("")
    : `<option disabled>(none defined)</option>`;
  $("bhv-add").disabled = !name || !library.length;
  $("btn-bhv-add").disabled = !name || !library.length;

  $("bhv-hint").textContent = !name
    ? "Name the element first — behaviours attach to the node name."
    : library.length ? "" : "No behaviours defined yet.";
}

const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/**
 * The behaviours on this node, each with a Remove button, a JSON editor for its
 * parameters - and, for the ones that liquefy, the `linked` picker.
 *
 * Rows are keyed by **position**, not by behaviour name. A node may carry the
 * same behaviour twice - the runtime builds one instance per entry in the list
 * - and keying by name would have the second row's Remove take the first one
 * away and both parameter boxes write to the same assignment. Where a name
 * appears more than once the rows are numbered, since otherwise nothing on
 * screen would tell them apart.
 *
 * The parameters are edited as raw JSON rather than as named fields because the
 * runtime owns which parameters a behaviour understands: a form built here
 * would list whatever this tool happened to know about on the day it was
 * written, and hide the rest. The old three `direction` boxes were exactly that
 * - one runtime parameter promoted to a widget, with no way to author a second.
 *
 * The `linked` picker survives on top of the JSON because its value is a list
 * of node names from the current room, which is knowledge the editor has and
 * the person typing does not. It edits the same key, and that key is shown in
 * the JSON too, so neither view can silently contradict the other.
 */
function renderApplied(nodeName, applied) {
  const host = $("bhv-applied");
  // Never rebuild under a field being typed in: it destroys the caret, the
  // same trap the inspector's number fields fell into. A focused *button* is
  // not editing, so Remove still redraws the list it just changed.
  const el = document.activeElement;
  if (host.contains(el) && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) return;

  if (!applied.length) {
    host.innerHTML = nodeName ? '<div class="muted">none attached</div>' : "";
    return;
  }
  const chunk = entryOf(state.selection[0])?.chunk;
  const candidates = nodeNamesInChunk(chunk, nodeName);
  const picker = (kind, label, options, chosen, at) => {
    const opts = options.length
      ? options.map((n) =>
        `<option value="${esc(n)}"${chosen.includes(n) ? " selected" : ""}>${esc(n)}</option>`)
        .join("")
      : `<option disabled>(nothing else named in this room)</option>`;
    return `<div class="linked"><div class="lbl">${label}</div>`
      + `<select multiple size="4" data-${kind}="${at}">${opts}</select></div>`;
  };
  const total = new Map();
  for (const b of applied) total.set(b.name, (total.get(b.name) || 0) + 1);
  const seen = new Map();
  host.innerHTML = applied.map((b, at) => {
    const ordinal = (seen.get(b.name) || 0) + 1;
    seen.set(b.name, ordinal);
    const rows = [
      `<div class="item"><span class="n">${esc(b.name)}</span>`
      + (total.get(b.name) > 1 ? `<span class="muted">#${ordinal}</span>` : "")
      + `<button data-remove="${at}">Remove</button></div>`,
    ];
    if (isLiquefiable(b.name)) {
      rows.push(picker("linked", "linked — melts together", candidates, b.linked, at));
    }
    // Empty rather than "{}" when there is nothing set, so the placeholder can
    // show what this behaviour's definition suggests - the hint the old
    // direction fields gave by pre-filling, which a filled-in "{}" would hide.
    const text = behaviorParamsText(b);
    const lines = text ? text.split("\n").length : 3;
    rows.push('<div class="linked"><div class="lbl">parameters — JSON, editor space</div>'
      + `<textarea data-params="${at}" spellcheck="false" `
      + `rows="${Math.min(Math.max(lines, 3), 14)}" `
      + `placeholder="${esc(behaviorParamsHint(b.name))}">${esc(text)}</textarea>`
      + `<div class="bhv-err" data-err="${at}"></div></div>`);
    return rows.join("");
  }).join("");

  for (const btn of host.querySelectorAll("[data-remove]")) {
    btn.addEventListener("click", () => {
      removeEntityBehavior(nodeName, Number(btn.dataset.remove));
      refreshBehavior();
    });

  }
  for (const sel of host.querySelectorAll("[data-linked]")) {
    sel.addEventListener("change", () => {
      setEntityLinked(nodeName, Number(sel.dataset.linked),
        [...sel.selectedOptions].map((o) => o.value));
      refreshBehavior();
    });
  }
  // `change`, not `input`: JSON is invalid for most of the time it takes to
  // type, so committing on every keystroke would be one long error message.
  for (const area of host.querySelectorAll("[data-params]")) {
    area.addEventListener("change", () => {
      const at = Number(area.dataset.params);
      const err = host.querySelector(`[data-err="${at}"]`);
      const text = area.value.trim();
      let params;
      try {
        params = text ? JSON.parse(text) : {};
      } catch (e) {
        // Left as typed, and NOT redrawn: the text is wrong but it is the
        // user's, and replacing it with the stored value would throw away the
        // edit they are in the middle of making.
        if (err) err.textContent = String(e.message || e);
        return;
      }
      if (!params || typeof params !== "object" || Array.isArray(params)) {
        if (err) err.textContent = "expected a JSON object, like { \"direction\": [0, 0, 1] }";
        return;
      }
      setEntityParams(nodeName, at, params);
      refreshBehavior();
    });
  }
}

/** An applied behaviour as the JSON the panel shows: its parameters, no name. */
function behaviorParamsText(b) {
  const out = {};
  for (const [key, value] of Object.entries(b)) {
    if (key === "name") continue;
    if (key === "linked" && !value.length) continue;   // absent means "stands alone"
    out[key] = value;
  }
  return Object.keys(out).length ? JSON.stringify(out, null, 2) : "";
}

/**
 * The placeholder for an empty parameter box: what this behaviour's definition
 * suggests. `direction` is the one the kit's definitions carry, and the old
 * fields pre-filled from it; the definition body is not otherwise a template
 * for the assignment, so nothing else is offered.
 */
function behaviorParamsHint(behaviorName) {
  const d = getBehaviorDef(behaviorName)?.direction;
  return Array.isArray(d) && d.length === 3 && d.every(Number.isFinite)
    ? `{ "direction": [${d.map(Number).join(", ")}] }`
    : "{ }";
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
  // A collision primitive reports what Havok is actually given, not a bounding
  // box: a sphere is one radius and a capsule a radius plus a height, and the
  // whole point of constraining the scale is that those numbers are the truth.
  // A world AABB would show a sphere as three identical sides and a *turned*
  // capsule as something with no relation to its radius at all.
  const only = state.selection.length === 1 ? entryOf(state.selection[0]) : null;
  if (only?.type === "collider") {
    const d = colliderDims(only);
    const f = (v) => (v < 0.1 ? v.toFixed(3) : v.toFixed(2));
    if (d.kind === "box") {
      $("dim-x").textContent = f(d.size[0]);
      $("dim-y").textContent = f(d.size[1]);
      $("dim-z").textContent = f(d.size[2]);
      $("dim-note").textContent = "— metres, X/Y/Z";
    } else if (d.kind === "sphere") {
      $("dim-x").textContent = f(d.radius);
      $("dim-y").textContent = "—";
      $("dim-z").textContent = "—";
      $("dim-note").textContent = "— radius, in metres";
    } else {
      $("dim-x").textContent = f(d.radius);
      $("dim-y").textContent = f(d.height);
      $("dim-z").textContent = "—";
      $("dim-note").textContent = d.kind === "capsule"
        ? "— radius / height, caps included"
        : "— radius / height, in metres";
    }
    for (const id of ["dim-x", "dim-y", "dim-z"]) $(id).title = "";
    return;
  }

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

/**
 * The chunk list as a dropdown, plus the two entries that are not chunks.
 *
 * "(auto)" is the empty value - let the manifest infer the side. `skybox: true`
 * adds the reserved id that means "space", offered on Chunk B only: a portal
 * has to lead *from* somewhere, and a door with space on both sides would be a
 * hole in nothing.
 */
function fillChunkSelect(sel, value, { skybox = false } = {}) {
  sel.innerHTML = "";
  const entries = [["", "(auto)"], ...alphabetical(state.chunks).map((c) => [c, c])];
  if (skybox) entries.push([SKYBOX_CHUNK, "Skybox (outer space)"]);
  for (const [v, label] of entries) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = label;
    sel.appendChild(o);
  }
  const known = state.chunks.includes(value) || (skybox && value === SKYBOX_CHUNK);
  sel.value = known ? value : "";
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
  if (e.canMove !== false) {
    e.node.position.set(
      num(posIn[0], p.x), num(posIn[1], p.y), num(posIn[2], p.z));
  }
  if (e.type !== "environment-probe") {
    setEuler(e.node, [num(rotIn[0], r[0]), num(rotIn[1], r[1]), num(rotIn[2], r[2])]);
  }
  // Doors scale too: their exported width/height fold the node scale in, and
  // portalOf() already reads the world matrix, so the portal follows.
  const scale = [
    num(sclIn[0], s.x), num(sclIn[1], s.y), num(sclIn[2], s.z),
  ];
  e.node.scaling.set(...(e.type === "environment-probe"
    ? scale.map((value) => Math.max(0.01, Math.abs(value)))
    : scale));
  // A collider's kind decides what scales are representable at all.
  if (e.type === "collider" && reconcileCollider(e)) syncScaleFields(e);
  select([e.id]);                       // keeps the outline in step
  emit("transform");
}

/**
 * Read a numeric field, keeping the old value while the entry is incomplete.
 * A bare "-" parses as NaN, and coercing that to a default is what used to make
 * negative (mirrored) scales impossible to type.
 */
/**
 * Write a constrained scale back into the fields the user is not typing in.
 *
 * Rewriting the focused one would fight the caret, exactly as the uniform-scale
 * mirroring does - so a sphere typed into X shows the same number appear in Y
 * and Z, but only once the caret leaves.
 */
function syncScaleFields(e) {
  const s = e.node.scaling.asArray();
  sclIn.forEach((el, i) => { if (el !== document.activeElement) el.value = round(s[i]); });
  refreshDimensions();
}

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

// ------------------------------------------------------------------ lights

$("lgt-type").innerHTML = LIGHT_TYPES
  .map((v) => `<option value="${v}">${v}</option>`).join("");

$("lgt-owner").addEventListener("change", (ev) => {
  if (syncing) return;
  const light = entryOf(state.selection[0]);
  if (light?.type !== "light") return;
  inspectorPushed = false;
  pushUndo();
  const owner = setLightOwner(light.id, ev.target.value);
  if (!owner) {
    // Nothing changed, so nothing emitted: put the select back by hand.
    refreshInspector();
    setStatus("that placement cannot own a light");
    return;
  }
  setStatus(`${light.id} now rides ${owner.owner}`);
});

// Number fields fire on every keystroke like the transform ones, so they share
// the arm-on-focus undo; a select or a checkbox is a single decision and pushes
// on its own.
const LIGHT_NUM = {
  "lgt-intensity": ["runtime", "intensity"], "lgt-range": ["runtime", "range"],
  "lgt-angle": ["runtime", "angle"],
};
for (const [id, [part, key]] of Object.entries(LIGHT_NUM)) {
  const el = $(id);
  el.addEventListener("focus", () => { inspectorPushed = false; });
  el.addEventListener("input", () => {
    const v = num(el, null);
    if (v !== null) editLight(part, { [key]: v });
  });
}
for (const [id, part, key] of [
  ["lgt-type", "runtime", "type"],
]) {
  $(id).addEventListener("change", (ev) => {
    inspectorPushed = false;
    editLight(part, { [key]: ev.target.value });
  });
}
for (const [id, part, key] of [
  ["lgt-clustered", "runtime", "clustered"], ["lgt-shadows", "runtime", "castsShadows"],
]) {
  $(id).addEventListener("change", (ev) => {
    inspectorPushed = false;
    editLight(part, { [key]: ev.target.checked });
  });
}
for (const [id, part] of [["lgt-run-color", "runtime"]]) {
  // A colour picker streams while the user drags around the wheel, so it gets
  // the same one-undo-per-visit treatment as a number field.
  $(id).addEventListener("focus", () => { inspectorPushed = false; });
  $(id).addEventListener("input", (ev) => editLight(part, { color: rgbOf(ev.target.value) }));
}

$("btn-add-light").addEventListener("click", () => {
  const e = state.selection.length === 1 ? entryOf(state.selection[0]) : null;
  // A light rides a placement. Adding one to the light already selected is the
  // obvious second click, so that resolves to its owner rather than doing
  // nothing - but nothing else can own one.
  const owner = e && (e.type === "light" ? e.owner : (e.type ? null : e.id));
  if (!owner) {
    setStatus("select one module or prop to attach a light to");
    return;
  }
  const light = addLight(owner);
  if (!light) { setStatus("that element cannot hold a light"); return; }
  select([light.id]);
  setStatus(`added ${light.id} to ${owner}`);
});

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
    // Choosing space as the far side seals the door; the inspector is redrawn
    // so the checkbox shows it and greys out, rather than sitting there stale.
    normalizeDoorSides(e);
    if (key === "chunkB") {
      setStatus(e.chunkB === SKYBOX_CHUNK
        ? `${e.id} opens onto space — sealed, see through only`
        : `${e.id}: chunk B is ${e.chunkB || "(auto)"}`);
    }
    refreshInspector();
    validate();
  });
}
$("door-enabled").addEventListener("change", () => {
  const e = entryOf(state.selection[0]);
  if (!e || e.type !== "door" || syncing) return;
  pushUndo();
  e.enabled = $("door-enabled").checked;
  setStatus(`${e.id} portal ${e.enabled ? "enabled" : "disabled"}`);
  validate();
});
$("door-sealed").addEventListener("change", () => {
  const e = entryOf(state.selection[0]);
  if (!e || e.type !== "door" || syncing) return;
  pushUndo();
  e.sealed = $("door-sealed").checked;
  setStatus(e.sealed
    ? `${e.id} sealed — visible through, not walkable`
    : `${e.id} is a doorway again`);
  validate();
});

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
  const chunks = alphabetical(state.chunks);
  for (const sel of [$("chunk-select"), $("insp-chunk")]) {
    const prev = sel.value;
    sel.innerHTML = "";
    for (const c of chunks) {
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
  const counts = new Map(alphabetical(state.chunks).map((c) => [c, 0]));
  for (const p of shipPlacements()) {
    counts.set(p.chunk, (counts.get(p.chunk) || 0) + 1);
  }
  const rows = [...counts.entries()].map(([c, n]) =>
    `<tr class="${c === state.activeChunk ? "active" : ""}"><td>${c}</td><td class="n">${n}</td></tr>`);
  $("chunk-stats").innerHTML = `<table>${rows.join("")}</table>`;
  const doors = [...state.markers.values()].filter((m) => m.type === "door").length;
  const veiled = veilCounts();
  const pct = Math.round(state.veilAlpha * 100);
  statusCounts.textContent =
    `${shipPlacements().length} objects · ${state.chunks.length} chunks · ${state.environmentProbes.size} probes · ${doors} doors · ${state.selection.length} selected`
    + (veiled.ghost ? ` · ${veiled.ghost} at ${pct}%` : "")
    + (veiled.hidden ? ` · ${veiled.hidden} hidden` : "");
}

$("chunk-select").addEventListener("change", (ev) => {
  state.activeChunk = ev.target.value;
  applyVisibility();
  refreshStats();
});
$("btn-chunk-assign").addEventListener("click", () => assignSelectionToChunk(state.activeChunk));

// ----------------------------------------------------------- chunks window
//
// A window rather than a toolbar button, because renaming a room, seeing what
// is in it and deleting an empty one are three halves of one job. The toolbar
// keeps only the two things you reach for while building: which chunk is
// active, and Assign.

/** Which chunk the pane is editing. Not `state.activeChunk`: browsing the
 *  list to look at a far room should not move where new placements land. */
let chunkSelected = null;

function placeChunkWindow(left, top) {
  const win = $("chunk-modal");
  const rect = win.getBoundingClientRect();
  const edge = 8;
  const maxLeft = Math.max(edge, window.innerWidth - rect.width - edge);
  const maxTop = Math.max(edge, window.innerHeight - rect.height - edge);
  win.style.left = `${Math.min(Math.max(edge, left), maxLeft)}px`;
  win.style.top = `${Math.min(Math.max(edge, top), maxTop)}px`;
}

function keepChunkWindowOnScreen() {
  const win = $("chunk-modal");
  if (win.hidden) return;
  const rect = win.getBoundingClientRect();
  placeChunkWindow(rect.left, rect.top);
}

{
  const win = $("chunk-modal");
  const handle = $("chunk-window-handle");
  let drag = null;
  handle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.target.closest("button")) return;
    const rect = win.getBoundingClientRect();
    drag = { pointerId: e.pointerId, dx: e.clientX - rect.left, dy: e.clientY - rect.top };
    handle.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  handle.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.pointerId) return;
    placeChunkWindow(e.clientX - drag.dx, e.clientY - drag.dy);
  });
  const finishDrag = (e) => {
    if (!drag || e.pointerId !== drag.pointerId) return;
    if (handle.hasPointerCapture(e.pointerId)) handle.releasePointerCapture(e.pointerId);
    drag = null;
  };
  handle.addEventListener("pointerup", finishDrag);
  handle.addEventListener("pointercancel", finishDrag);
  window.addEventListener("resize", keepChunkWindowOnScreen);
}

function openChunks() {
  $("chunk-modal").hidden = false;
  refreshChunkPane(chunkSelected || state.activeChunk);
  requestAnimationFrame(keepChunkWindowOnScreen);
}

function closeChunks() {
  $("chunk-modal").hidden = true;
}

function refreshChunkPane(pick) {
  const chunks = alphabetical(state.chunks);
  chunkSelected = pick && state.chunks.includes(pick) ? pick : chunks[0];
  $("chunk-list").innerHTML = chunks.map((c) => {
    const held = chunkUsers(c);
    // The count is the affordance: an empty room is either the one you are
    // about to fill or one you forgot to delete, and there is no other way to
    // tell them apart without clicking every entry.
    return `<option value="${esc(c)}"${c === chunkSelected ? " selected" : ""}>${esc(c)} (${held.placements.length})</option>`;
  }).join("");
  $("chunk-name").value = chunkSelected || "";
  const users = chunkSelected ? chunkUsers(chunkSelected) : { placements: [], doors: [] };
  $("chunk-holds").textContent = chunkSelected
    ? `holds ${users.placements.length} object(s), ${users.doors.length} door(s)`
    : "";  $("chunk-error").textContent = "";
  $("btn-chunk-delete").disabled = state.chunks.length < 2;
}

$("btn-chunks").addEventListener("click", openChunks);
$("btn-chunk-close").addEventListener("click", closeChunks);
$("chunk-list").addEventListener("change", (e) => refreshChunkPane(e.target.value));

$("btn-chunk-new").addEventListener("click", () => {
  const n = String(state.chunks.length).padStart(2, "0");
  let name = `CH${n}_New`;
  for (let i = 1; state.chunks.includes(name); i++) name = `CH${n}_New${i}`;
  if (!addChunk(name)) return;
  refreshChunkPane(name);
  $("chunk-name").select();
  setStatus(`added chunk ${name}`);
});

$("btn-chunk-apply").addEventListener("click", () => {
  if (!chunkSelected) return;
  const was = chunkSelected;
  const wanted = $("chunk-name").value.trim();
  if (!wanted || wanted === was) {
    setStatus(`${was}: nothing to apply`);
    return;
  }
  if (!renameChunk(was, wanted)) {
    $("chunk-error").textContent =
      `Cannot rename to "${wanted}" — it is empty or already used.`;
    return;
  }
  refreshChunkPane(wanted);
  refreshChunks();
  setStatus(`renamed ${was} → ${wanted}`);
});
$("btn-chunk-delete").addEventListener("click", () => {
  if (!chunkSelected) return;
  const gone = chunkSelected;
  const res = removeChunk(gone);
  if (res.ok) {
    refreshChunkPane(state.chunks[0]);
    refreshChunks();
    setStatus(`deleted chunk ${gone}`);
    return;
  }
  // Say what is in the way, not just that something is: "in use" with no names
  // leaves you clicking through the ship looking for it.
  if (res.reason === "last") {
    $("chunk-error").textContent = "A ship needs at least one chunk.";
  } else if (res.reason === "in use") {
    const bits = [];
    if (res.users.placements.length) bits.push(`${res.users.placements.length} object(s)`);
    if (res.users.doors.length) bits.push(`${res.users.doors.length} door(s)`);
    $("chunk-error").textContent =
      `"${gone}" still holds ${bits.join(" and ")} — move them out with Assign first`
      + ` (${[...res.users.placements, ...res.users.doors].slice(0, 6).join(", ")}`
      + `${res.users.placements.length + res.users.doors.length > 6 ? ", …" : ""}).`;
  } else {
    $("chunk-error").textContent = `Cannot delete "${gone}".`;
  }
});

// ------------------------------------------------ environment probe window

let probeSelected = null;
let probeRefreshRequest = 0;
const PROBE_BOX_FIELDS = ["x", "y", "z"].map((axis) => `probe-box-${axis}`);
const PROBE_SIZE_FIELDS = ["x", "y", "z"].map((axis) => `probe-size-${axis}`);
const PROBE_CAMERA_FIELDS = ["x", "y", "z"].map((axis) => `probe-camera-${axis}`);
const PROBE_INFLUENCE_FIELDS = ["x", "y", "z"].map((axis) => `probe-influence-${axis}`);
const PROBE_INFLUENCE_SIZE_FIELDS = ["x", "y", "z"].map((axis) => `probe-influence-size-${axis}`);
const PROBE_INNER_SIZE_FIELDS = ["x", "y", "z"].map((axis) => `probe-inner-size-${axis}`);
const PROBE_VALUE_FIELDS = [
  ...PROBE_BOX_FIELDS, ...PROBE_SIZE_FIELDS, ...PROBE_CAMERA_FIELDS,
  ...PROBE_INFLUENCE_FIELDS, ...PROBE_INFLUENCE_SIZE_FIELDS, ...PROBE_INNER_SIZE_FIELDS,
];

function placeProbeWindow(left, top) {
  const win = $("probe-modal");
  const rect = win.getBoundingClientRect();
  const edge = 8;
  const maxLeft = Math.max(edge, window.innerWidth - rect.width - edge);
  const maxTop = Math.max(edge, window.innerHeight - rect.height - edge);
  win.style.left = `${Math.min(Math.max(edge, left), maxLeft)}px`;
  win.style.top = `${Math.min(Math.max(edge, top), maxTop)}px`;
}

function keepProbeWindowOnScreen() {
  const win = $("probe-modal");
  if (win.hidden) return;
  const rect = win.getBoundingClientRect();
  placeProbeWindow(rect.left, rect.top);
}

{
  const win = $("probe-modal");
  const handle = $("probe-window-handle");
  let drag = null;
  handle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.target.closest("button")) return;
    const rect = win.getBoundingClientRect();
    drag = { pointerId: e.pointerId, dx: e.clientX - rect.left, dy: e.clientY - rect.top };
    handle.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  handle.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.pointerId) return;
    placeProbeWindow(e.clientX - drag.dx, e.clientY - drag.dy);
  });
  const finishDrag = (e) => {
    if (!drag || e.pointerId !== drag.pointerId) return;
    if (handle.hasPointerCapture(e.pointerId)) handle.releasePointerCapture(e.pointerId);
    drag = null;
  };
  handle.addEventListener("pointerup", finishDrag);
  handle.addEventListener("pointercancel", finishDrag);
  window.addEventListener("resize", keepProbeWindowOnScreen);
}

function defaultProbeVolume() {
  let min = null;
  let max = null;
  for (const placement of shipPlacements()) {
    if (placement.chunk !== state.activeChunk) continue;
    const bounds = worldBounds(placement.node);
    if (!bounds) continue;
    min = min ? BABYLON.Vector3.Minimize(min, bounds.min) : bounds.min.clone();
    max = max ? BABYLON.Vector3.Maximize(max, bounds.max) : bounds.max.clone();
  }
  const centre = min && max
    ? min.add(max).scale(0.5)
    : (state.camera?.globalPosition || state.camera?.position
      || new BABYLON.Vector3(0, 2.5, 0)).clone();
  const size = min && max ? max.subtract(min) : new BABYLON.Vector3(8, 5, 8);
  // The influence volumes are left out on purpose: setEnvironmentProbe derives
  // them from the box, which is the one place that default is written down.
  return {
    boxPosition: centre.asArray(),
    boxSize: size.asArray().map((n) => Math.max(0.01, n)),
    capturePosition: centre.asArray(),
    resolution: 512,
  };
}

function probeDraft() {
  const raw = PROBE_VALUE_FIELDS.map((id) => $(id).value.trim());
  const resolution = Number($("probe-resolution").value);
  if (raw.some((value) => value === "")) return null;
  const values = raw.map(Number);
  const boxPosition = values.slice(0, 3);
  const boxSize = values.slice(3, 6);
  const capturePosition = values.slice(6, 9);
  const influenceBoxPosition = values.slice(9, 12);
  const influenceBoxSize = values.slice(12, 15);
  const influenceInnerBoxSize = values.slice(15, 18);
  if (!values.every(Number.isFinite) || !boxSize.every((n) => n > 0)
    || !influenceBoxSize.every((n) => n > 0)
    || influenceInnerBoxSize.some((n, axis) => n < 0 || n > influenceBoxSize[axis])
    || !Number.isFinite(resolution) || resolution < 16 || resolution > 4096) return null;
  return {
    id: validEnvironmentProbeId($("probe-id").value),
    boxPosition,
    boxSize,
    capturePosition,
    influenceBoxPosition,
    influenceBoxSize,
    influenceInnerBoxSize,
    resolution: Math.round(resolution),
  };
}

async function refreshProbeWindow(pick = probeSelected) {
  const request = ++probeRefreshRequest;
  const ids = alphabetical(environmentProbeIds());
  probeSelected = pick && ids.includes(pick) ? pick : (ids[0] || null);
  $("probe-list").innerHTML = ids.map((id) => {
    const probe = environmentProbeOf(id);
    return `<option value="${esc(id)}"${id === probeSelected ? " selected" : ""}>`
      + `${esc(id)} · ${probe.resolution}px</option>`;
  }).join("");
  const probe = probeSelected ? environmentProbeOf(probeSelected) : null;
  $("probe-id").value = probe?.id || "";
  const values = probe
    ? [...probe.boxPosition, ...probe.boxSize, ...probe.capturePosition,
      ...probe.influenceBoxPosition, ...probe.influenceBoxSize,
      ...probe.influenceInnerBoxSize] : [];
  PROBE_VALUE_FIELDS.forEach((id, index) => { $(id).value = values[index] ?? ""; });
  $("probe-resolution").value = probe?.resolution ?? "";
  $("btn-probe-delete").disabled = !probe;
  $("btn-probe-apply").disabled = !probe;
  // Capture takes the selected probe, so it goes with Delete and Apply rather
  // than with Capture all, which never needs one.
  $("btn-capture-one").disabled = !probe;
  $("probe-error").textContent = "";
  if (!probe) {
    $("probe-resolved").textContent = "No probe volumes. Press New to create one.";
    $("probe-show").checked = false;
    $("probe-env").checked = false;
    await showEnvironmentProbe(null, false, null, false);
    return;
  }
  const info = await localEnvironmentProbeOf(probe.id);
  if (request !== probeRefreshRequest || probe.id !== probeSelected) return;
  $("probe-resolved").textContent =
    `box ${probe.boxSize.map((n) => Number(n).toFixed(2)).join(" × ")} m`
    + ` · camera ${probe.capturePosition.map((n) => Number(n).toFixed(2)).join(", ")}`
    + ` · ${info.generated?.env ? "generated asset available" : "not generated yet"}`;
  if ($("probe-show").checked) {
    await showEnvironmentProbe(
      probe.id, true, probe, $("probe-env").checked);
  }
}

async function showAndSelectProbe(id) {
  if (!id) return;
  $("probe-show").checked = true;
  await refreshProbeWindow(id);
  select([id]);
}

function openProbes() {
  $("probe-modal").hidden = false;
  const id = probeSelected || environmentProbeIds()[0];
  // With nothing to select there is still a pane to put in order: the empty
  // state is what says to press New, and the buttons that act on a selection
  // have to open inert rather than pointing at a probe that is not there.
  if (id) void showAndSelectProbe(id); else void refreshProbeWindow(null);
  requestAnimationFrame(keepProbeWindowOnScreen);
}

function closeProbes() {
  $("probe-modal").hidden = true;
  $("probe-show").checked = false;
  $("probe-env").checked = false;
  dropProbeSelection();
  void showEnvironmentProbe(null, false, null, false);
}

/**
 * Drop every gizmo of `id` - or of all probes - from the selection.
 *
 * The gizmos go with the window, and a selection holding an element that no
 * longer exists is a selection whose inspector, wheel and arrow keys all point
 * at nothing. `probe` names the capture box; its two blend volumes are parts of
 * the same id and have to go with it.
 */
function dropProbeSelection(id = null) {
  const doomed = (sel) => (id
    ? sel === id || environmentProbePartOf(sel)?.probe === id
    : state.environmentProbes.has(sel) || !!environmentProbePartOf(sel));
  if (!state.selection.some(doomed)) return;
  select(state.selection.filter((sel) => !doomed(sel)));
}

$("btn-probes").addEventListener("click", openProbes);
$("btn-probe-close").addEventListener("click", closeProbes);
$("probe-list").addEventListener("change", (e) => {
  probeSelected = e.target.value;
  void showAndSelectProbe(probeSelected);
});

for (const id of [...PROBE_VALUE_FIELDS, "probe-resolution"]) {
  $(id).addEventListener("input", () => {
    if (!$("probe-show").checked || !probeSelected) return;
    const draft = probeDraft();
    if (draft) {
      void showEnvironmentProbe(
        probeSelected, true, draft, $("probe-env").checked);
    }
  });
}

$("probe-show").addEventListener("change", (e) => {
  if (!e.target.checked) $("probe-env").checked = false;
  if (!e.target.checked) {
    dropProbeSelection(probeSelected);
    void showEnvironmentProbe(null, false, null, false);
    return;
  }
  void showEnvironmentProbe(
    probeSelected, true, probeDraft(), $("probe-env").checked).then(() => {
    if (probeSelected) select([probeSelected]);
  });
});

$("probe-env").addEventListener("change", (e) => {
  if (e.target.checked) $("probe-show").checked = true;
  void showEnvironmentProbe(
    probeSelected, $("probe-show").checked, probeDraft(), e.target.checked);
});

$("btn-probe-new").addEventListener("click", () => {
  const id = nextEnvironmentProbeId();
  setEnvironmentProbe(id, defaultProbeVolume());
  probeSelected = id;
  void showAndSelectProbe(id);
  setStatus(`created environment probe ${id}`);
});

$("btn-probe-apply").addEventListener("click", () => {
  if (!probeSelected) return;
  const draft = probeDraft();
  const nextId = validEnvironmentProbeId($("probe-id").value);
  if (!nextId) {
    $("probe-error").textContent =
      "ID must be 1–128 letters, digits, dots, underscores, or hyphens.";
    return;
  }
  if (!environmentProbeIdAvailable(nextId, probeSelected)) {
    $("probe-error").textContent = `ID "${nextId}" is already used by another editor entry.`;
    return;
  }
  if (!draft) {
    $("probe-error").textContent =
      "Enter numeric box/camera positions, positive box sizes, an inner size from"
      + " 0 up to the influence size, and a texture size from 16 to 4096.";
    return;
  }
  const previousId = probeSelected;
  probeSelected = nextId;
  const changed = setEnvironmentProbe(nextId, draft, previousId);
  $("probe-error").textContent = "";
  void showAndSelectProbe(probeSelected);
  setStatus(changed
    ? `${previousId === probeSelected ? probeSelected : `${previousId} → ${probeSelected}`}: environment probe updated`
    : `${probeSelected}: environment probe unchanged`);
});

$("btn-probe-delete").addEventListener("click", () => {
  if (!probeSelected) return;
  const gone = probeSelected;
  if (!removeEnvironmentProbe(gone)) return;
  probeSelected = null;
  void refreshProbeWindow();
  setStatus(`deleted environment probe ${gone}`);
});

on("environment-probes", () => {
  refreshStats();
  if (!$("probe-modal").hidden) void refreshProbeWindow(probeSelected);
});

on("selection", () => {
  // A blend volume is part of its probe: clicking one in the viewport should
  // bring that probe up in the window, exactly as clicking its capture box does.
  const selectedProbe = state.selection
    .map((id) => (state.environmentProbes.has(id) ? id : environmentProbePartOf(id)?.probe))
    .find(Boolean);
  if (!selectedProbe || $("probe-modal").hidden || selectedProbe === probeSelected) return;
  probeSelected = selectedProbe;
  $("probe-show").checked = true;
  void refreshProbeWindow(selectedProbe);
});

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

/**
 * Palette tile size.
 *
 * Saved with the ship (manifest `editorPrefs`) *and* mirrored to localStorage:
 * the manifest wins when it carries a value, and the local copy is what a ship
 * written before the block came in falls back to. Same arrangement as the
 * editor's Env/Exposure, and for the same reason - reopening a ship should look
 * the way you left it, on any machine.
 */
function setBigPalette(on) {
  state.bigPalette = !!on;
  document.body.classList.toggle("big-palette", on);
  $("big-palette").checked = on;
  localStorage.setItem("bigPalette", on ? "1" : "0");
  applyPanelWidth(PANELS[0]);
  state.engine?.resize();
}

/**
 * Follow the editor's view preferences when a load or a reset moves them.
 *
 * applyEditorPrefs() only touches state - what a preference *means* on screen
 * is the UI's business - so this is where the manifest's values become a body
 * class and a slider position.
 */
function refreshEditorPrefs() {
  $("veil-alpha").value = String(state.veilAlpha);
  showVeilAlpha(state.veilAlpha);
  localStorage.setItem("veilAlpha", String(state.veilAlpha));
  setBigPalette(state.bigPalette);
  refreshStats();                     // the status bar quotes the ghost percentage
}
on("prefs", refreshEditorPrefs);

// ------------------------------------------------------- resizable panels
//
// Both side panels are dragged by the strip between them and the viewport.
// Widths are kept in localStorage only, unlike the settings on the Settings
// pane: a drag is not a setting anyone opened a pane to make, and there are two
// remembered widths per panel (see `key` below) rather than one value to author.

const PANELS = [
  {
    id: "resize-palette", panel: "palette", prop: "--palette-w",
    min: 180, edge: "right",
    // The palette grid keeps its column count, so its width *is* the tile size
    // and the two icon sizes want genuinely different widths. One remembered
    // number would mean dragging it once permanently defeated Big icons.
    key: () => (document.body.classList.contains("big-palette")
      ? "paletteWidth.big" : "paletteWidth"),
    fallback: () => (document.body.classList.contains("big-palette") ? 520 : 260),
  },
  {
    id: "resize-inspector", panel: "inspector", prop: "--inspector-w",
    min: 180, edge: "left",
    key: () => "inspectorWidth",
    fallback: () => 232,
  },
];

/** The width this panel should have now, from storage or from its default. */
function panelWidth(p) {
  const saved = parseFloat(localStorage.getItem(p.key()));
  return Number.isFinite(saved) ? clampPanel(p, saved) : p.fallback();
}

/** Wide enough to be usable, and never so wide the viewport disappears. */
function clampPanel(p, w) {
  const room = Math.max(p.min, Math.round(window.innerWidth * 0.45));
  return Math.round(Math.min(Math.max(w, p.min), room));
}

function applyPanelWidth(p, w = panelWidth(p)) {
  // On body, not :root — `body.big-palette` sets --palette-w too, and a rule on
  // the nearer ancestor would win over one set further up.
  document.body.style.setProperty(p.prop, `${w}px`);
}

for (const p of PANELS) {
  applyPanelWidth(p);
  const grip = $(p.id);
  grip.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const startX = e.clientX;
    const startW = $(p.panel).getBoundingClientRect().width;
    // Capture, so a drag that outruns the 5 px strip - which every drag does -
    // keeps being reported here instead of stopping dead over the canvas.
    grip.setPointerCapture(e.pointerId);
    grip.classList.add("dragging");
    document.body.classList.add("resizing");
    const move = (ev) => {
      const dx = p.edge === "right" ? ev.clientX - startX : startX - ev.clientX;
      applyPanelWidth(p, clampPanel(p, startW + dx));
      state.engine?.resize();
    };
    const up = () => {
      grip.removeEventListener("pointermove", move);
      grip.removeEventListener("pointerup", up);
      grip.removeEventListener("pointercancel", up);
      grip.releasePointerCapture(e.pointerId);
      grip.classList.remove("dragging");
      document.body.classList.remove("resizing");
      localStorage.setItem(p.key(), String($(p.panel).getBoundingClientRect().width));
    };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", up);
    grip.addEventListener("pointercancel", up);
  });
  grip.addEventListener("dblclick", () => {
    localStorage.removeItem(p.key());
    applyPanelWidth(p);
    state.engine?.resize();
  });
  // Keyboard, because a 5 px target is not one everybody can hit.
  grip.addEventListener("keydown", (e) => {
    const step = e.key === "ArrowLeft" ? -16 : e.key === "ArrowRight" ? 16 : 0;
    if (!step) return;
    e.preventDefault();
    const w = clampPanel(p, $(p.panel).getBoundingClientRect().width
      + (p.edge === "right" ? step : -step));
    applyPanelWidth(p, w);
    localStorage.setItem(p.key(), String(w));
    state.engine?.resize();
  });
}

// A narrower window can leave a stored width covering most of it.
addEventListener("resize", () => { for (const p of PANELS) applyPanelWidth(p); });

/**
 * Remember which palette panes are folded.
 *
 * localStorage rather than the manifest, like the exposure and the big-palette
 * switch: whether *you* keep Settings rolled up says nothing about the ship,
 * and putting it in the layout would make folding a pane count as unsaved work.
 */
for (const pane of document.querySelectorAll("#palette-panes details")) {
  const key = `pane.${pane.id}`;
  if (localStorage.getItem(key) === "0") pane.open = false;
  pane.addEventListener("toggle", () => {
    localStorage.setItem(key, pane.open ? "1" : "0");
  });
}

/**
 * Flash a button orange when it is pressed.
 *
 * Save, Load and Export glb all do their work somewhere else - a file on disk,
 * a line in the status bar - so the button gave no sign it had been hit, and a
 * press that missed looked exactly like one that worked.
 *
 * One delegated listener rather than a line in every handler: a button added
 * later is covered without anyone remembering to, and no handler can forget.
 * It runs on the capture phase so a handler that stops propagation - or throws
 * - still gets its flash.
 *
 * Toggle buttons are skipped: they go solid orange and *stay* there, which says
 * the same thing for longer, and flashing a slightly different orange first
 * only muddies it.
 */
const TAP_MS = 260;
const tapTimers = new WeakMap();
document.addEventListener("click", (e) => {
  const b = e.target?.closest?.("button");
  if (!b || b.disabled || b.classList.contains("toggle")) return;
  clearTimeout(tapTimers.get(b));      // a second press restarts the flash
  b.classList.add("tapped");
  tapTimers.set(b, setTimeout(() => b.classList.remove("tapped"), TAP_MS));
}, true);

/**
 * One undo entry per slider *gesture*, not per pixel of travel.
 *
 * `input` fires continuously while a range is dragged, so a single sweep would
 * otherwise bury the stack in near-identical snapshots. Armed on pointerdown
 * and on the first keyboard nudge, spent on the first change - the same shape
 * as the inspector fields, but with no focus event to hang it on, because a
 * range keeps focus across separate drags.
 */
let lightPushed = false;
function pushLightUndoOnce() {
  if (lightPushed) return;
  pushUndo();
  lightPushed = true;
}

/** Arm the once-per-gesture guard on a range input, however it is being moved. */
function armSliderUndo(el) {
  el.addEventListener("pointerdown", () => { lightPushed = false; });
  el.addEventListener("keydown", () => { lightPushed = false; });
  // A wheel over a focused range also moves it, and fires neither of the above.
  el.addEventListener("wheel", () => { lightPushed = false; }, { passive: true });
}

/**
 * The two lighting rigs, control by control.
 *
 * `which` names the set in state.lightSets. Both are saved with the ship, in
 * `environment` and `editorEnvironment`, but only the runtime's is read by the
 * game - which is the whole point of showing them apart: one says how you like
 * to look at the ship, the other is part of the ship. `store` is the machine's
 * fallback copy, kept for the editor rig so a manifest without one still opens
 * at the brightness you last built in.
 */
const LIGHT_CONTROLS = [
  { id: "editor-env", which: "editor", key: "strength", digits: 1, store: "editorEnv" },
  { id: "editor-exposure", which: "editor", key: "exposure", digits: 2, store: "editorExposure" },
  { id: "runtime-env", which: "runtime", key: "strength", digits: 1 },
  { id: "runtime-exposure", which: "runtime", key: "exposure", digits: 2 },
];

for (const c of LIGHT_CONTROLS) {
  const el = $(c.id);
  armSliderUndo(el);
  el.addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    // Only the ship's own rig is an edit to the ship. Undoing a change to how
    // *you* light the editor would spend the undo stack on nothing.
    if (c.which === "runtime") pushLightUndoOnce();
    setLightSetting(c.which, c.key, v);
    $(`${c.id}-val`).textContent = v.toFixed(c.digits);
    if (c.store) localStorage.setItem(c.store, String(v));
  });
}

/**
 * The view transforms.
 *
 * The runtime's is undoable, and its only copy is the manifest: the demos read
 * `environment.toneMapping` from there, so a value kept on this machine would
 * be one the runtime never sees. The editor's is saved too, in
 * `editorEnvironment`, but keeps a localStorage fallback and stays off the undo
 * stack - it is not an edit to the ship.
 */
$("editor-tone").addEventListener("change", (e) => {
  setLightSetting("editor", "toneMapping", e.target.value);
  localStorage.setItem("editorTone", e.target.value);
});

$("runtime-tone").addEventListener("change", (e) => {
  pushUndo();
  setLightSetting("runtime", "toneMapping", e.target.value);
});

/**
 * How see-through Shift+H makes things.
 *
 * Saved with the ship (manifest `editorPrefs`) and mirrored to localStorage,
 * like the editor's Env/Exposure above it. Not undoable, for the same reason
 * they are not: it says how you are looking at the ship, not what the ship is,
 * and spending an undo entry on it would take back a change no edit made.
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
 * Follow the lighting when a load, an undo, or a change of view mode moves it.
 *
 * Both rigs are shown at once and always read out of state, so the pane never
 * has to guess which one is live. Only the editor's is mirrored to
 * localStorage: the runtime's belongs to the manifest, and copying it here is
 * what used to let a session spent in the Runtime view quietly overwrite the
 * brightness the editor came up in.
 */
function refreshLighting() {
  for (const c of LIGHT_CONTROLS) {
    const v = state.lightSets[c.which][c.key];
    $(c.id).value = String(v);
    $(`${c.id}-val`).textContent = v.toFixed(c.digits);
    if (c.store) localStorage.setItem(c.store, String(v));
  }
  showToneMapping($("editor-tone"), state.lightSets.editor.toneMapping);
  showToneMapping($("runtime-tone"), state.lightSets.runtime.toneMapping);
  localStorage.setItem("editorTone", $("editor-tone").value);
  $("view-mode").value = viewMode();
}

/** Put a tone-mapping combo on `name`, however loosely the manifest spelled it. */
function showToneMapping(el, name) {
  el.value = name;
  if (el.value) return;
  // The manifest may name the transform loosely ("aces", "KHR_PBR_NEUTRAL").
  // The scene resolves those fine, but the combo box only holds the four
  // canonical spellings, so match on what they resolve to rather than
  // leaving the control blank and lying about the state.
  const want = resolveToneMapping(name);
  el.value = [...el.options]
    .find((o) => resolveToneMapping(o.value) === want)?.value ?? "Khronos PBR Neutral";
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

/**
 * Put the viewport in one of the three named modes.
 *
 * Switching in deliberately does NOT capture anything. It used to: entering the
 * Runtime view brought every stale probe up to date first, which meant that
 * looking at the ship the way the game lights it could cost a minute of
 * rendering nobody asked for, at the one moment you wanted a quick look. The
 * capture is a deliberate act now, and it lives with the boxes it belongs to,
 * in the Probes window. What you see on the way in is whatever .env files are
 * on disk, which is exactly what the game would load right now.
 *
 * On failure the combo is put back on the mode actually in force rather than
 * on the one asked for, so it never claims to be showing something it is not.
 */
async function applyViewMode(mode) {
  const sel = $("view-mode");
  const flags = viewModeFlags(mode);
  try {
    sel.disabled = true;
    setUnlit(flags.unlit);
    if (flags.runtime !== state.runtime) {
      setStatus(flags.runtime ? "lighting the ship the way the game does…" : "back to the editor's rig");
      await setRuntimePreview(flags.runtime);
    }
    setStatus(viewModeStatus(mode));
  } catch (err) {
    console.error(err);
    await setRuntimePreview(false).catch(() => {});
    setStatus("view mode: " + err.message);
  } finally {
    sel.disabled = false;
    refreshLighting();
    localStorage.setItem("viewMode", viewMode());
    refreshHud();
  }
}

/** Drop the preview and build it again, so it picks up freshly written .env files. */
async function reloadRuntimePreview() {
  await setRuntimePreview(false);
  await setRuntimePreview(true);
}

/** What just happened, counted off the preview where there is one. */
function viewModeStatus(mode) {
  if (mode === "editor") return "editor — the ship you are building, under the editor's own rig";
  if (mode === "editor-unlit") return "editor unlit — raw albedo, no lighting";
  const p = runtimePreview();
  return `runtime — ${p.meshes.length} mesh(es) lit by ${p.lights.length} lamp(s), `
    + `${p.inProbe} in a probe box and ${p.outsideProbe} outside one`;
}

$("view-mode").addEventListener("change", (e) => applyViewMode(e.target.value));
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

/**
 * Y switches between the world's axes and the element's own. The combo and the
 * status line follow, for the same reason V's do: a mode you cannot see is one
 * you will forget you are in, and this one silently changes where every drag,
 * carry, arrow key and turn goes.
 */
function setAxisSpaceFromKey() {
  toggleAxisSpace();
  refreshAxisSpace();
  setStatus(state.axisSpace === "local"
    ? "moving and turning about the element's own axes — press Y for the world's"
    : "moving and turning about the world's axes — press Y for the element's own");
}

function refreshAxisSpace() {
  $("axis-space").value = state.axisSpace;
}

$("axis-space").addEventListener("change", (e) => {
  setAxisSpace(e.target.value);
  refreshAxisSpace();
  setStatus(`moving and turning in ${state.axisSpace} space`);
});

$("btn-select-rect").addEventListener("click", () => {
  const on = setSelectMode(!state.selectMode);
  $("btn-select-rect").setAttribute("aria-pressed", on ? "true" : "false");
  setStatus(on
    ? "rectangle select — drag to select, Ctrl/Shift adds"
    : "drag moves elements again");
});

/**
 * Whether anything has changed since the last save or load.
 *
 * Compared against a snapshot rather than tracked with a flag, so undoing back
 * to the saved state correctly counts as *clean* - a flag would keep claiming
 * unsaved work that no longer exists.
 *
 * `hidden` is stripped: it never reaches the manifest, so it can never be
 * saved, and leaving it in would make hiding one wall enough to prompt for the
 * rest of the session.
 */
let savedState = null;

function dirtyKey() {
  const s = serialize();
  delete s.hidden;
  return JSON.stringify(s);
}

export function markSaved() { savedState = dirtyKey(); autoState = savedState; }
function isDirty() { return savedState !== null && savedState !== dirtyKey(); }

// --------------------------------------------------------------- auto-save
//
// A recovery copy, written beside the ship rather than over it. It keeps its
// *own* baseline: a background write must not clear "you have unsaved work",
// because the ship you last chose to save still does not have those changes.
// Using one baseline for both would mean an auto-save quietly disarmed the
// guard that stops you closing the tab on an hour of work.

let autoState = null;
let autoTimer = null;
let autoBusy = false;

async function autoSaveTick() {
  if (autoBusy || isBusy()) return;
  if (!(Number(state.config.autoSaveMinutes) > 0)) return;   // off is off
  const now = dirtyKey();
  if (autoState !== null && now === autoState) return;   // nothing has changed
  autoBusy = true;
  try {
    const r = await saveAutosave();
    autoState = dirtyKey();
    const at = new Date().toLocaleTimeString();
    setStatus(`auto-saved ${r.bytes} bytes → ${r.path.split(/[\\/]/).pop()} at ${at}`);
  } catch (e) {
    setStatus(`auto-save failed: ${e.message}`);
  } finally {
    autoBusy = false;
  }
}

/** Restart the timer from the current setting. 0 minutes turns it off. */
function rearmAutoSave() {
  if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
  const mins = Number(state.config.autoSaveMinutes) || 0;
  if (mins > 0) autoTimer = setInterval(autoSaveTick, mins * 60 * 1000);
}
on("config", rearmAutoSave);

/** For tests: run one tick now, and report whether it wrote anything. */
export async function autoSaveNow() {
  const before = autoState;
  await autoSaveTick();
  return { wrote: autoState !== before, armed: !!autoTimer };
}

$("btn-save").addEventListener("click", doSave);
$("btn-load").addEventListener("click", doLoad);

/**
 * The same guard for closing or reloading the tab.
 *
 * The browser decides the wording - custom text has been ignored since 2016 -
 * so this only chooses *whether* to ask. Chrome additionally requires the page
 * to have been interacted with before it will show the prompt at all, which is
 * the behaviour we want anyway: a tab you only ever looked at closes silently.
 *
 * `returnValue` as well as `preventDefault()`, because browsers disagree about
 * which one arms it and setting both is the only combination that works
 * everywhere.
 */
addEventListener("beforeunload", (e) => {
  if (!isDirty()) return;
  e.preventDefault();
  e.returnValue = "";
});

/**
 * Write the ship: the manifest, and the glb beside it.
 *
 * One button, because the two files are one thing: the runtime loads the glb
 * for geometry and the manifest for everything else, so a manifest saved
 * without its glb is a ship whose description and geometry disagree.
 *
 * The glb is reported separately and never takes the manifest down with it:
 * losing the export is a nuisance, and being told the layout was not saved
 * when it was would be worse.
 */
async function doSave() {
  let saved;
  try {
    setStatus("saving…");
    // The staging area is a working copy: read it back before writing, or
    // whatever is on it right now would not be in the file.
    if (state.collisionMode) harvestStage();
    const r = await saveLayout();
    markSaved();
    const coll = r.collisionError
      ? ` — collision file NOT written: ${r.collisionError}`
      : (r.collision ? `, collision → ${r.collision.path.split(/[\\/]/).pop()}` : "");
    saved = r.previous
      ? `saved ${r.bytes} bytes → ${r.path} (previous kept as ${r.previous})${coll}`
      : `saved ${r.bytes} bytes → ${r.path}${coll}`;
  } catch (e) { setStatus("save failed: " + e.message); return; }
  try {
    setStatus(`${saved} — exporting glb…`);
    const r = await exportGlb();
    setStatus(`${saved}, ${(r.bytes / 1048576).toFixed(1)} MB → ${r.path}`);
  } catch (e) {
    console.error(e);
    setStatus(`${saved} — glb NOT written: ${e.message}`);
  }
}

async function doLoad() {
  // A load throws away everything in the scene, and it is one button away from
  // Save. Nothing else in the tool destroys unsaved work in a single click.
  if (isDirty()
    && !confirm("Load will discard your unsaved changes.\n\nLoad the saved ship anyway?")) {
    setStatus("load cancelled — your changes are still here");
    return;
  }
  try {
    setStatus("loading…");
    const data = await loadLayout();
    markSaved();
    setStatus(data
      ? `loaded ${data.instances.length} instances`
      : "nothing to load (no tool-written manifest yet)");
    refreshChunks();
  } catch (e) { setStatus("load failed: " + e.message); }
}

// -------------------------------------------------- the environment probes
//
// A capture renders the live scene six times per probe, so it has to happen in
// this window - there is nothing on disk to hand to a background process. The
// server is told the whole authored probe list and answers with the ones whose
// room has actually changed, so the usual click costs one request.

/**
 * Take the probes and say what came of it.
 *
 * `only` names a single probe and is what the Probes window's **Capture**
 * sends; without it the whole ship is checked and the stale rooms are taken,
 * which is **Capture all**. Both buttons go inert for the duration: the capture
 * owns the scene, and a second one started on top of it would be photographing
 * a viewport the first has already taken over.
 */
async function doProbes({ force = false, only = null } = {}) {
  const buttons = [$("btn-capture-all"), $("btn-capture-one")];
  try {
    for (const b of buttons) b.disabled = true;
    setStatus(only ? `capturing ${only}…` : "checking environment probes…");
    const result = await generateLocalEnvironments((done, total, probe) => {
      if (probe) setStatus(`capturing environment probe ${probe} (${done + 1}/${total})…`);
    }, { force, only });
    const size = `${(result.bytes / 1048576).toFixed(1)} MB`;
    setStatus(only
      ? `captured ${only}, ${size} → export/environments/`
      : (result.converted
        ? `captured ${result.converted} probe(s), ${size} → export/environments/`
        : "every environment probe is already up to date"));
    // The preview is wearing the cubemaps from before the capture.
    if (result.converted && state.runtime) await reloadRuntimePreview();
  } catch (e) {
    console.error(e);
    setStatus("probe capture failed: " + e.message);
  } finally {
    // The selected-probe button follows the list, not the lock: with no probe
    // to point at there is nothing for it to capture.
    for (const b of buttons) b.disabled = false;
    $("btn-capture-one").disabled = !probeSelected;
  }
}

// Shift-click re-captures everything, for when a probe has to be re-taken after
// a change the digest cannot see - a texture edited on disk, say.
$("btn-capture-all").addEventListener("click", (e) => doProbes({ force: e.shiftKey }));
// The selected probe is always re-taken: asking for one by name is an
// instruction, not a question about whether it has gone stale.
$("btn-capture-one").addEventListener("click", () => {
  if (probeSelected) void doProbes({ only: probeSelected });
});

/** For tests: capture the probes and wait for it. */
export async function captureProbes(force = false, only = null) {
  return await generateLocalEnvironments(() => {}, { force, only });
}
// --------------------------------------------------------------- keyboard

/**
 * Keys that belong to the browser, not to the editor.
 *
 * The function keys are the reload, the dev tools and the rest of the chrome's
 * own row, and F5 above all is how you get out of a tool that has wedged. The
 * busy guard below is exactly the state you want it in - so it is the one place
 * that must never be allowed to swallow them.
 */
function isBrowserKey(e) {
  return /^F\d{1,2}$/.test(e.key);
}

/**
 * Which controls capture the keyboard, and which keys they actually use.
 *
 * A focused control owning the keyboard is what typing *is*, so the shortcuts
 * stand aside for one. But a Position or Intensity field is `type="number"`,
 * and a number field cannot spell a letter: the browser drops it on the floor.
 * So pressing R after typing a value did nothing at all, and nothing visible
 * said why - press it again, still nothing, until something was clicked. That
 * is the whole of "sometimes R takes several presses".
 *
 * A control keeps every key it can use. A control that cannot use a letter -
 * number, range, colour, checkbox - hands that letter back to the editor, which
 * costs the field nothing: it was discarding it. `e` is the exception, because
 * a number field reads it as an exponent, and no shortcut is spelled with it.
 *
 * A `select` is not in that set either: a letter jumps it to the matching
 * option, which is a real use.
 */
const BLIND_INPUTS = new Set(["number", "range", "color", "checkbox", "radio"]);

function isFormControl(t) {
  return !!t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA");
}

function fieldKeepsKey(t, e) {
  if (t.tagName !== "INPUT") return true;
  if (!BLIND_INPUTS.has((t.type || "text").toLowerCase())) return true;
  const letter = e.key.length === 1 && /[a-z]/i.test(e.key) && !/^e$/i.test(e.key);
  return !letter || e.ctrlKey || e.metaKey;
}

window.addEventListener("keydown", async (e) => {
  // Every shortcut is an edit or a mode change, and the scene is mid-rebuild.
  // Dropped, not claimed: preventDefault here used to take F5 with it, which
  // turned any stall under the lock into a page that could not even be
  // reloaded.
  if (isBusy()) {
    if (!isBrowserKey(e)) e.preventDefault();
    return;
  }
  // The behaviour library is modal: nothing behind it should be editable, and
  // a stray X or Del while a button in it holds focus would act on the ship.
  if (!$("bhv-modal").hidden) {
    if (e.key === "Escape") { e.preventDefault(); closeLibrary(); }
    return;
  }
  const t = e.target;
  if (isFormControl(t)) {
    // Escape abandons a field and Enter says "done with it" - both hand the
    // keyboard back, so the next shortcut lands on the ship instead of being
    // typed into a control nobody is looking at any more. A textarea keeps
    // Enter: there it is a new line.
    if (e.key === "Escape" || (e.key === "Enter" && t.tagName !== "TEXTAREA")) {
      t.blur();
      return;
    }
    if (fieldKeepsKey(t, e)) return;
  }
  if (t && !$("chunk-modal").hidden && $("chunk-modal").contains(t)) {
    if (e.key === "Escape") { e.preventDefault(); closeChunks(); }
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
  // The bare letter picks the *axis*, Shift walks that setting's value and Ctrl
  // walks it back. V for translation, R for rotation, F for scale - one letter
  // each, the same three modifiers on every one. The edits themselves live on
  // the wheel (Shift turns, Ctrl resizes) and on Alt+F for a mirror.
  //
  // preventDefault is not optional on these: Ctrl+F is the browser's find bar,
  // Ctrl+R reloads the page, and Ctrl+V pastes. All three *can* be claimed -
  // Ctrl+R confirmed by hand, which is the only way to check: Playwright
  // injects keys through CDP, so a genuinely reserved shortcut still looks
  // claimed to a test.
  if (mod && e.key.toLowerCase() === "r") {
    e.preventDefault();
    return cycleRotSnap(-1);
  }
  if (mod && e.key.toLowerCase() === "f") {
    e.preventDefault();
    return cycleScaleSnap(-1);
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
    // F is the scale axis, Shift+F walks the scale step and Ctrl+F walks it
    // back - the same shape V has for translation and R for rotation. Mirroring
    // is the odd one out with no setting of its own, so it takes Alt+F.
    case "f": case "F": {
      e.preventDefault();
      if (e.altKey) {
        const r = flipCurrent();
        setStatus(r
          ? `mirrored ${r.count} object(s) on ${r.axis.toUpperCase()}`
          : "nothing to mirror — select an element first");
      } else if (e.shiftKey) {
        cycleScaleSnap(1);
      } else {
        cycleRotOrScaleAxis("scale");
      }
      break;
    }
    // R is the rotation axis, Shift+R walks the angle and Ctrl+R walks it back.
    // The turn itself is Shift+wheel, and Alt+Shift+wheel swings the whole
    // selection about a shared pivot.
    case "r": case "R":
      e.preventDefault();
      if (e.shiftKey) cycleRotSnap(1);
      else cycleRotOrScaleAxis("rot");
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
    // Y says whose axes V's choice means: the world's, or the element's own.
    // Next to V on both layouts, and the pair reads as "which axis, whose".
    // Ctrl+Y is already redo and is claimed above, before this switch.
    case "y": case "Y": {
      e.preventDefault();
      setAxisSpaceFromKey();
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
      // Showing the axes says which space you are thinking in, so the move and
      // turn axes follow. It is what you meant 99 times in 100 - you press
      // Shift+X to see which way the element's own X grows *because* you are
      // about to work along it - and Y still overrides it either way.
      // Only on the way up: hiding a gizmo says nothing about intent.
      if (shown && state.axisSpace !== space) {
        setAxisSpace(space);
        refreshAxisSpace();
      }
      const label = target === GHOST_AXES
        ? (ghostModule() || "ghost")
        : (entryOf(target)?.name || entryOf(target)?.module || target);
      setStatus(shown
        ? `${space} axes on ${label} — moving and turning in ${space} space too,`
          + ` Y switches · ${space === "local" ? "Shift+X" : "X"} hides them`
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
    // M picks the selection up onto the cursor (see grabCurrent). G was the
    // Blender-idiomatic key for this, but it already toggles the grid here.
    case "m": case "M": e.preventDefault(); grabCurrent(); break;
    // B brings whatever is in hand to your feet. The physical key is KeyB on
    // both QWERTY and AZERTY, so matching the label costs nothing here.
    case "b": case "B": e.preventDefault(); bringCurrentToCamera(); break;
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

/**
 * Right mouse button does the same as Escape - except it never closes the
 * collision area.
 *
 * RMB is also the camera button. Letting a stray right-click tear down the
 * bench you were working on, when its only job here is to put down whatever is
 * in your hand, is far too much to hang off a button you press to look around.
 */
on("escape", () => cancelEverything({ closeModes: false }));

function cancelEverything({ closeModes = true } = {}) {
  if (cancelMarquee()) return;     // an in-flight rectangle goes first
  if (cancelDrag()) return;        // then an in-flight drag
  if (ghostActive() || state.brush) { cancelGhost(); setBrush(null); clearMarkerBrush(); select([]); return; }
  // Only once there is nothing in hand does Escape close the collision area -
  // otherwise cancelling an armed shape would throw you back to the ship.
  if (closeModes && state.collisionMode) { closeCollisionArea(); return; }
  cancelGhost();
  setBrush(null);
  clearMarkerBrush();
  select([]);
}

/**
 * Whether arming a ghost is about to move the drag axis, so the status line can
 * say so. The move itself happens in interact.js, where every arming path goes
 * through one of three functions and none can be forgotten - this only reports
 * it, because a mode that changes itself quietly is worse than one you change
 * by hand.
 */
function axisNoteForGhost() {
  return state.dragAxis === "xz" ? "" : " · drag axis set back to X/Z";
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
  if (!cur) return;

  // Several selected: carry copies of the whole set. This used to fall back to
  // duplicating in place, because a ghost could only hold one module.
  const many = cur.ids.map(entryOf)
    .filter((e) => e?.module || e?.type === "collider")
    // A *module* can only be on the bench once - the association rule needs one
    // answer to "which element is this shape on", and two instances give two.
    // Shapes are a different matter: a hull is often several boxes.
    .filter((e) => !(state.collisionMode && e.stage && e.type !== "collider"));
  if (!many.length
      && cur.ids.some((id) => { const e = entryOf(id); return e?.stage && e.type !== "collider"; })) {
    setStatus("a module can only be on the bench once — copy its shapes instead");
    return;
  }
  const lights = cur.ids.map(entryOf).filter((e) => e?.type === "light");
  if (lights.length === cur.ids.length) {
    pushUndo();
    const made = lights.map((light) => duplicateLight(light.id)).filter(Boolean);
    if (!made.length) return;
    emit("lights");
    select(made.map((light) => light.id));
    setStatus(made.length === 1
      ? `copy of light ${lights[0].id} created beside the original`
      : `copies of ${made.length} lights created beside the originals`);
    return;
  }
  if (many.length > 1) {
    const axisNote = axisNoteForGhost();
    grabSelection({ copy: true }).then((g) => {
      if (g) setStatus(`copy of ${many.length} elements on the cursor — click to place${axisNote}`);
    });
    return;
  }

  const entry = many[0];
  if (!entry) return duplicateSelected();          // markers have no module
  const axisNote = axisNoteForGhost();

  // The ghost sits on the build plane, so without this the copy of something on
  // an upper deck would appear back down at ground level. Moving the plane
  // rather than giving the ghost its own height keeps one source of truth, and
  // the grid visibly follows so it is obvious what happened.
  const y = entry.node.position.y;
  if (entry.type === "collider") {
    // the plane is the primitive's base, to match the corner-first drop
    const half = colliderHalf(entry.node.scaling.asArray(), entry.node.rotationQuaternion).y;
    armColliderGhost(entry.kind, {
      rotation: eulerOf(entry.node),
      scaling: entry.node.scaling.asArray(),
      baseY: y - half,
    });
    refreshColliderButtons();
    setStatus(`copy of ${COLLIDER_LABEL[entry.kind]} on the cursor — click to place${axisNote}`);
    return;
  }

  setBrush(entry.module, {
    rotation: eulerOf(entry.node),
    scaling: entry.node.scaling.asArray(),
    // The copy keeps the source's height without dragging the build plane up
    // to it. Moving the plane was the old way, and it meant Ctrl+D silently
    // changed where *everything placed afterwards* would land.
    baseY: y,
  });
  setStatus(`copy of ${entry.module} on the cursor at ${y.toFixed(2)} m — click to place${axisNote}`);
}

/**
 * Pick the selection up so it follows the cursor with no button held.
 *
 * The other half of the answer to "why are there two ways to move something".
 * A drag is still a drag - press, move, release, and the elements stay solid
 * the whole way. This is the hands-free version: the elements go translucent,
 * the mouse is free, and a click lands them. Esc puts them back.
 */
function grabCurrent() {
  if (ghostActive() || isDragging()) return;
  const cur = currentElement();
  const ids = (cur?.ids || []).map(entryOf)
    .filter((e) => e?.module || e?.type === "collider");
  if (!ids.length) {
    setStatus("nothing to pick up — select an element first");
    return;
  }
  grabSelection().then((g) => {
    if (g) {
      setStatus(ids.length > 1
        ? `carrying ${ids.length} elements — click to drop, Esc to put them back`
        : `carrying ${ids[0].module || COLLIDER_LABEL[ids[0].kind]}`
          + " — click to drop, Esc to put it back");
    }
  });
}

/**
 * Move what is in hand to a grid spot just in front of the camera.
 *
 * The answer to "I clicked a module in the palette and the ghost went
 * somewhere I cannot find". Arming does not choose a position - the ghost is
 * wherever the cursor ray happens to cross the build plane - so from inside a
 * finished room it is routinely behind you, or across the map. Rather than
 * teach the palette to guess, `B` fetches it, and takes the build plane with
 * it so it stays fetched.
 */
function bringCurrentToCamera() {
  if (isDragging()) return;
  const r = bringToCamera();
  if (!r) {
    setStatus("nothing in hand — arm a module or select an element first");
    return;
  }
  const what = r.kind === "ghost"
    ? (r.count > 1 ? `${r.count} ghosts` : "the ghost")
    : (r.count > 1 ? `${r.count} elements` : "the selection");
  setStatus(`brought ${what} here — build plane`
    + `${r.floor ? " on the floor" : ""} at y ${r.y.toFixed(2)}`);
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
  // Straight off the hover, not through currentElement(): edits follow the
  // selection now, and this one is deliberately about what you are pointing at.
  // It reads an element rather than changing one, so it has none of the "the
  // wrong thing moved" problem that took hover off the editing keys.
  const id = hoveredId();
  if (!id) { setStatus("hover an element first"); return; }
  const entry = entryOf(id);
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
  refreshAxisSpace();
  $("rot-axis").value = state.rotAxis;
  $("scale-axis").value = state.scaleAxis;

  const cur = currentElement();
  $("hud-current").textContent = cur
    ? `${cur.kind === "ghost" ? "◆" : "■"} ${short(cur.module)}`
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
    const members = shipPlacements().filter((p) => p.chunk === c);
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

  const sunk = shipPlacements().filter((p) => {
    const b = worldBounds(p.node);
    return b && b.min.y < -0.05;
  });
  if (sunk.length) out.push(["warn", `${sunk.length} object(s) below y = 0`]);

  const offGrid = shipPlacements().filter((p) => {
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
    // A window onto space has nothing to slide: it is a hole in the hull, not a
    // doorway, so the missing-leaves warning would be permanent noise.
    if (!d.leaves.length && b !== SKYBOX_CHUNK) out.push(["warn", `${d.id}: no leaves assigned`]);
  }
  const orphans = state.chunks.filter((c) =>
    !linked.has(c) && shipPlacements().some((p) => p.chunk === c));
  if (orphans.length && state.chunks.length > 1) {
    out.push(["warn", `unreachable: ${orphans.join(", ")}`]);
  }
  if (!doors.length && state.chunks.length > 1) out.push(["warn", "no doors placed yet"]);

  // A room with no probe reflects nothing at all in the game, which on a kit
  // this metallic reads as black panels rather than as a missing feature.
  const covered = new Set();
  for (const probe of state.environmentProbes.values()) covered.add(probe.id);
  if (!covered.size && shipPlacements().length) {
    out.push(["warn", "no environment probes — nothing in the ship will reflect anything"]);
  }
  const el = $("validation");
  el.innerHTML = out.length
    ? out.map(([k, m]) => `<div class="${k}">${m}</div>`).join("")
    : `<div class="ok">All checks pass.</div>`;
}

// ------------------------------------------------------------------- boot

on("selection", () => { refreshInspector(); refreshStats(); });
// The wheel tunes a lamp without going through the panel, so the panel has to
// follow the record rather than only ever writing to it.
on("lights", () => { refreshInspector(); refreshStats(); });
on("placements", () => { refreshChunks(); refreshStats(); validate(); });
on("chunks", () => {
  refreshChunks();
  refreshSettings();
  // An undo can add, remove, rename or retune a chunk while the pane is open.
  if (!$("chunk-modal").hidden) refreshChunkPane(chunkSelected);
});
on("markers", () => { refreshStats(); validate(); });
on("transform", () => {
  for (const id of state.selection) {
    const isProbe = state.environmentProbes.has(id);
    const part = isProbe ? null : environmentProbePartOf(id);
    if (!isProbe && !part) continue;
    const entry = entryOf(id);
    if (!entry) continue;
    // The gizmo is the truth for the length of a gesture, and the record is
    // written from it - but a box dragged down to nothing is no box at all, so
    // a floor goes on first and is written back to the node too, or the next
    // notch would resume from the value the wheel reached rather than the one
    // that was kept.
    const size = entry.node.scaling.asArray().map((value) => Math.max(0.01, Math.abs(value)));
    entry.node.scaling.set(...size);
    if (isProbe) {
      syncEnvironmentProbeTransform(id, entry.node.position.asArray(), size);
    } else if (part.part === "influence") {
      // The blend volumes ride the same kind of gizmo but write other fields,
      // and the inner one has no centre of its own to write at all.
      syncEnvironmentProbeInfluence(part.probe, entry.node.position.asArray(), size);
    } else {
      syncEnvironmentProbeInnerSize(part.probe, size);
    }
  }
  refreshInspector();
  validate();
});
on("grid", refreshHud);
on("modes", () => { refreshHud(); refreshLighting(); });
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
 *
 * The floating tool windows are in that list too, and the Probes window is why:
 * it is where a capture is started from, so it is guaranteed to be open while
 * the longest lock the editor takes is held, with a New and a Delete button on
 * it that would otherwise still take the keyboard.
 */
function refreshBusy() {
  const busy = isBusy();
  $("busy").hidden = !busy;
  $("busy-msg").textContent = busyLabel() || "working…";
  for (const id of ["toolbar", "palette", "viewport", "inspector",
    "probe-modal", "chunk-modal", "bhv-modal"]) {
    const el = $(id);
    if (el) el.inert = busy;
  }
  if (busy) document.activeElement?.blur();
}
on("pickmodule", (moduleId) => {
  // Arming anything puts the drag axis back on the floor, or the ghost drives
  // the build plane away from wherever you are looking.
  const axisNote = axisNoteForGhost();
  // On the collision area the palette *stages* modules: arming a brush there
  // would drop real kit geometry into the ship you cannot see.
  if (state.collisionMode) { stageFromPalette(moduleId, axisNote); return; }
  setBrush(moduleId); setStatus(`armed ${moduleId}${axisNote}`);
});
on("stagemodule", (moduleId) => stageFromPalette(moduleId, axisNoteForGhost()));
on("status", (msg) => setStatus(msg));
on("deletecurrent", () => deleteCurrent());
on("colliders", () => { refreshStats(); validate(); refreshCollisionMarks(); });

// ------------------------------------------------------------- collision

/**
 * The Collision pane: one button per primitive. A button *arms* the ghost the
 * same way a palette module does - nothing exists until you click in the view,
 * and the ghost stays armed so a run of boxes along a wall is repeated clicks.
 */
for (const kind of COLLIDER_KINDS) {
  const b = document.createElement("button");
  b.textContent = COLLIDER_LABEL[kind];
  b.dataset.kind = kind;
  b.title = SCALE_RULE[kind] === "free"
    ? "Any scale — Havok takes a box with a quaternion"
    : SCALE_RULE[kind] === "uniform"
      ? "Scales uniformly: Havok's sphere is a single radius"
      : kind === "capsule"
        ? "X and Z scale together as the radius; Y is the height, caps included"
        : "X and Z scale together as the radius; Y is the height";
  b.addEventListener("click", async () => {
    setBrush(null);                       // the two ghosts are the same slot
    const axisNote = axisNoteForGhost();
    await armColliderGhost(kind, { scaling: COLLIDER_DEFAULT_SCALE[kind] });
    refreshColliderButtons();
    setStatus(`${COLLIDER_LABEL[kind]} — click to place, Esc to cancel${axisNote}`);
  });
  $("collider-buttons").appendChild(b);
}

/** Light the button whose primitive the ghost is holding. */
function refreshColliderButtons() {
  const armed = ghostCollider();
  for (const b of $("collider-buttons").children) {
    b.classList.toggle("active", b.dataset.kind === armed);
  }
}
on("current", refreshColliderButtons);


// ----------------------------------------------- the collision staging area
//
// A mode, not a property of the selection: open it, stage whatever modules you
// want to fit shapes to, and close it again. The ship is hidden while it is
// open and put back untouched afterwards.

async function openCollisionArea() {
  cancelGhost();
  setBrush(null);
  await whileBusy("opening the collision area…", async () => {
    const restored = await enterCollisionMode(instantiate, moduleBounds);
    setGridElevation(0);
    // Only frame the bench when there is no viewpoint to come back to -
    // otherwise the focus throws away the view the mode just restored.
    const back = [...state.placements.values()].filter((p) => p.stage);
    if (back.length && !restored?.viewRestored) focusNodes(back.map((p) => p.node));
  });
  const n = [...state.placements.values()].filter((p) => p.stage).length;
  setStatus(n
    ? `collision area — ${n} module(s) back on the bench`
    : "collision area — pick modules from the left to stage them");
}

function closeCollisionArea() {
  cancelGhost();
  setBrush(null);
  const r = harvestStage();
  exitCollisionMode();
  setStatus(`back to the ship — ${state.moduleCollision.size} module(s) carry collision`
    + (r.orphans ? `, ${r.orphans} shape(s) belonged to nothing and were dropped` : ""));
}

$("btn-edit-module").addEventListener("click", async () => {
  if (state.collisionMode) closeCollisionArea(); else await openCollisionArea();
});
$("btn-module-done").addEventListener("click", closeCollisionArea);

/**
 * Put a module on the bench.
 *
 * One already there is focused rather than duplicated - a second instance would
 * give the association rule two equally good answers. A new one comes up on the
 * cursor as a ghost, the same as placing anything else, so you choose where it
 * goes instead of being handed a spot.
 */
async function stageFromPalette(moduleId, axisNote = "") {
  const already = [...state.placements.values()]
    .find((p) => p.stage && p.module === moduleId);
  if (already) {
    cancelGhost();
    select([already.id]);
    focusNodes([already.node]);
    setStatus(`${moduleId} is already staged`);
    return;
  }
  setBrush(moduleId);
  setStatus(`${moduleId} — click to put it on the bench${axisNote}`);
}

$("btn-module-fit").addEventListener("click", async () => {
  const r = await fitBoxToSelection(moduleBounds);
  if (!r.ok) { setStatus(r.error); return; }
  refreshModuleBanner();
  setStatus(`${r.module}: fitted one box at the collision shell`
    + ` (${state.config.shellThickness} m minimum) — scale and split it as you like`);
});

$("btn-module-fit-hull").addEventListener("click", async () => {
  setStatus("fitting a hull…");
  const r = await fitHullToSelection();
  if (!r.ok) { setStatus(r.error); return; }
  refreshModuleBanner();
  const how = { box: "one box", slabs: "a slab per face", split: "a split" }[r.how] || r.how;
  setStatus(`${r.module}: ${r.colliders.length} box(es) by ${how}`
    + ` at ${state.config.hullTolerance} m tolerance`
    + ` — covers ${(100 * r.coverage).toFixed(0)}%, ${(100 * r.solid).toFixed(0)}% of it on surface`
    + (r.confident ? "" : " — worth checking by eye, this is a shape better drawn by hand"));
});

function refreshModuleBanner() {
  const on = state.collisionMode;
  $("module-banner").hidden = !on;
  if (on) {
    const staged = [...state.placements.values()].filter((p) => p.stage).length;
    const orphans = orphanCount();
    $("module-banner-text").textContent = `Collision area — ${staged} staged`
      + (orphans ? `, ${orphans} shape(s) belong to nothing` : "");
    $("module-banner-text").classList.toggle("warn", orphans > 0);
  }
  $("btn-edit-module").textContent = on ? "Back to the ship" : "Edit collision";
  $("btn-edit-module").classList.toggle("active", on);
}
on("collisionMode", refreshModuleBanner);
on("colliders", refreshModuleBanner);
on("placements", refreshModuleBanner);

$("show-layer").addEventListener("change", (ev) => {
  setShowLayer(ev.target.value);
  setStatus(ev.target.value === "both" ? "showing the ship and its collision"
    : ev.target.value === "geometry" ? "showing the ship only"
      : "showing collision only");
});
on("modes", () => { $("show-layer").value = state.showLayer; });

on("focus", () => focusSelection());

// -------------------------------------------------------------- settings
//
// One row per ship-wide constant. Writes go through setConfig(), so they land
// on the undo stack and in the saved layout like any other edit.

function refreshSettings() {
  const shell = $("cfg-shell");
  if (document.activeElement !== shell) shell.value = state.config.shellThickness;
  const auto = $("cfg-autosave");
  if (document.activeElement !== auto) auto.value = state.config.autoSaveMinutes;
  const tol = $("cfg-hull-tol");
  if (document.activeElement !== tol) tol.value = state.config.hullTolerance;
  const thick = $("cfg-hull-thick");
  if (document.activeElement !== thick) thick.value = state.config.hullThickness;
  $("cfg-hull-offset").value = state.config.hullOffset;
  $("runtime-specular-aa").checked = state.runtimeSpecularAA;
  const roughness = $("runtime-roughness");
  if (document.activeElement !== roughness) roughness.value = state.runtimeRoughnessFactor;
  $("runtime-roughness-val").textContent = `${state.runtimeRoughnessFactor.toFixed(2)}×`;
}

$("cfg-hull-tol").addEventListener("change", () => {
  if (setConfig("hullTolerance", $("cfg-hull-tol").value)) {
    setStatus(`hull tolerance ${state.config.hullTolerance} m`
      + " — refit a module to apply it");
  }
  refreshSettings();
});

$("cfg-hull-thick").addEventListener("change", () => {
  if (setConfig("hullThickness", $("cfg-hull-thick").value)) {
    setStatus(`hull thickness ${state.config.hullThickness} m`
      + " — refit a module to apply it");
  }
  refreshSettings();
});

$("cfg-hull-offset").addEventListener("change", () => {
  if (setConfig("hullOffset", $("cfg-hull-offset").value)) {
    const o = state.config.hullOffset;
    setStatus(`hull offset ${o}`
      + (o === "centered" ? " — thickness split either side of the art"
        : o === "negative" ? " — hull tucked behind the art, clear of the play space"
          : " — hull stood in front of the art")
      + " — refit a module to apply it");
  }
  refreshSettings();
});

/**
 * The two material dials in the Runtime section.
 *
 * Authored ship values like the rig above them - saved in `environment` and
 * read by the game - so both are undoable, the checkbox per click and the
 * slider once per drag.
 */
$("runtime-specular-aa").addEventListener("change", (e) => {
  pushUndo();
  setRuntimeSpecularAA(e.target.checked);
  setStatus(e.target.checked ? "specular AA on" : "specular AA off");
});

armSliderUndo($("runtime-roughness"));
$("runtime-roughness").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  pushLightUndoOnce();
  setRuntimeRoughnessFactor(v);
  $("runtime-roughness-val").textContent = `${state.runtimeRoughnessFactor.toFixed(2)}×`;
});

on("reflection", refreshSettings);

$("cfg-shell").addEventListener("change", () => {
  if (setConfig("shellThickness", $("cfg-shell").value)) {
    setStatus(`collision shell ${state.config.shellThickness} m`
      + " — refit a module to apply it");
  }
  refreshSettings();
});

$("cfg-autosave").addEventListener("change", () => {
  if (setConfig("autoSaveMinutes", $("cfg-autosave").value)) {
    const m = state.config.autoSaveMinutes;
    setStatus(m ? `auto-saving every ${m} min to ship_autosave.json` : "auto-save off");
  }
  refreshSettings();
});

/**
 * Every row on the pane back to the value the editor ships with.
 *
 * All of them, not just the collision constants: since the whole pane is saved
 * with the ship, a button labelled "Reset to defaults" that quietly skipped the
 * lighting and the ghost alpha would be lying about what it did.
 *
 * The ship-side rows go under one undo entry, taken before anything moves and
 * only when something will actually move, so the button never leaves a no-op on
 * the stack. resetConfig() keeps pushing its own per key, as a single edit to
 * one of those rows does.
 */
const RIG_DEFAULTS = {
  strength: ENV_INTENSITY_DEFAULT,
  exposure: EXPOSURE_DEFAULT,
  toneMapping: TONE_MAPPING_DEFAULT,
};

$("btn-cfg-reset").addEventListener("click", () => {
  let changed = false;
  const rigsStale = ["editor", "runtime"].some((which) =>
    Object.entries(RIG_DEFAULTS).some(([key, v]) => state.lightSets[which][key] !== v));
  const dialsStale = state.runtimeSpecularAA !== RUNTIME_SPECULAR_AA_DEFAULT
    || state.runtimeRoughnessFactor !== RUNTIME_ROUGHNESS_FACTOR_DEFAULT;
  if (rigsStale || dialsStale) {
    pushUndo();
    for (const which of ["editor", "runtime"]) {
      for (const [key, v] of Object.entries(RIG_DEFAULTS)) setLightSetting(which, key, v);
    }
    setRuntimeSpecularAA(RUNTIME_SPECULAR_AA_DEFAULT);
    setRuntimeRoughnessFactor(RUNTIME_ROUGHNESS_FACTOR_DEFAULT);
    changed = true;
  }
  // The editor's own view preferences stay off the stack, the same rule their
  // rows follow when you move them by hand.
  if (state.veilAlpha !== VEIL_ALPHA_DEFAULT || state.bigPalette !== BIG_PALETTE_DEFAULT) {
    setVeilAlpha(VEIL_ALPHA_DEFAULT);
    state.bigPalette = BIG_PALETTE_DEFAULT;
    changed = true;
  }
  refreshEditorPrefs();
  for (const key of Object.keys(CONFIG_DEFAULTS)) changed = resetConfig(key) || changed;
  refreshLighting();
  refreshSettings();
  setStatus(changed ? "settings back to defaults" : "settings were already default");
});

on("config", refreshSettings);
on("placements", refreshSettings);      // a load or an undo can change them

// The overlay is up in the markup already, so there is never a frame in which
// a half-built editor looks ready to use. whileBusy() takes it from here.
(function boot() {
  return whileBusy("loading…", bootstrap).catch(showBootFailure);
})();

/**
 * Say why the editor did not start, on screen.
 *
 * Boot is all-or-nothing: fail it and there is no scene, no palette and no
 * inspector, just an empty viewport. whileBusy() takes its overlay down on the
 * way out regardless, so put one back up carrying the reason — otherwise the
 * only account of it is an unhandled rejection in a console nobody has open.
 */
function showBootFailure(err) {
  console.error(err);
  setStatus(`could not start: ${err?.message || err}`);
  const busy = $("busy");
  if (!busy) return;
  busy.hidden = false;
  busy.classList.add("failed");
  $("busy-msg").textContent = `The editor could not start.\n\n${err?.message || err}`;
  for (const id of ["toolbar", "palette", "viewport", "inspector"]) {
    const el = $(id);
    if (el) el.inert = true;
  }
}

async function bootstrap() {
  setStatus("loading catalogue…");
  refreshSettings();
  refreshModuleBanner();
  rearmAutoSave();
  // This machine's fallbacks, for a ship whose manifest carries no editorPrefs
  // or editorEnvironment block. Anything the manifest does carry overwrites
  // them when it lands, so these only ever show on an old or a brand-new ship.
  setBigPalette(localStorage.getItem("bigPalette") !== "0");
  localStorage.removeItem("taa");
  // The old rig keys held whichever set happened to be live when they were
  // written, so they are dropped rather than read: half of them are runtime
  // values.
  localStorage.removeItem("envIntensity");
  localStorage.removeItem("exposure");
  localStorage.removeItem("unlit");
  {
    const env = parseFloat(localStorage.getItem("editorEnv"));
    const exp = parseFloat(localStorage.getItem("editorExposure"));
    const tone = localStorage.getItem("editorTone");
    const set = state.lightSets.editor;
    set.strength = Number.isFinite(env) ? env : ENV_INTENSITY_DEFAULT;
    set.exposure = Number.isFinite(exp) ? exp : EXPOSURE_DEFAULT;
    if (tone) set.toneMapping = tone;
    // The editor set is the live one until a runtime view is picked, so the
    // mirrors state reads from have to start on it.
    state.envIntensity = set.strength;
    state.exposure = set.exposure;
    state.toneMapping = set.toneMapping;
  }
  // Only the editor modes are restored. The Runtime view re-dresses every mesh
  // in the ship - a clone per material per probe, a stand-in per instance - and
  // opening the tool straight into that is not what opening it should do.
  const mode = localStorage.getItem("viewMode");
  if (!viewModeFlags(mode).runtime && VIEW_MODES[mode]) $("view-mode").value = mode;
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
  state.kits = (getCatalogue().kits || []).map((k) => k.name);
  state.kitsSource = getCatalogue().kitsSource || null;
  state.kitsBase = getCatalogue().kitsBase || null;
  state.fluidSim = getCatalogue().fluidSim || [];

  await initScene($("render-canvas"));
  refreshLighting();
  // Returning to the viewport must restore keyboard control: a toolbar select
  // that still holds focus swallows every shortcut, numpad included.
  $("render-canvas").addEventListener("pointerdown", () => {
    const a = document.activeElement;
    if (a && $("toolbar").contains(a)) a.blur();
  });
  initInteract();
  if ($("view-mode").value === "editor-unlit") setUnlit(true);
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
    // No ship yet, but the kit's collision file may still be there - that is
    // the whole point of it living apart from any one ship.
    const coll = await loadCollision().catch(() => null);
    setStatus(`ready — ${getCatalogue().byId.size} modules`
      + (coll ? `, collision for ${Object.keys(coll).length} of them` : ""));
  }
  // Whatever we booted with - a restored ship or an empty grid - is the
  // baseline "unsaved changes" is measured against.
  markSaved();
  refreshCollisionMarks();     // the palette is only built by now
  validate();
}

function round(v) { return Math.round(v * 1000) / 1000; }
