// The authoring interaction model.
//
// There are no gizmos and no transform modes. A "ghost" - a translucent clone
// of a module - follows the cursor snapped to the build plane; clicking drops
// it. A placed element is moved by dragging it directly. Whatever the ghost
// holds, or whatever the cursor hovers, or failing both the selection, is the
// *current element*: Q/E turn it, F flips it, Shift + wheel resizes it.

import { getProto } from "./kit.js";
import {
  state, emit, on, pushUndo, placeAt, select, toggleSelect, entryOf,
  pickUnderCursor, setGridElevation, eulerOf, setEuler, cursorOnGrid, cursorOnPlane,
  worldBounds, dollyCamera, isRmbDown, nudgeMoveSpeed, cursorOnVerticalPlane,
  elementsInRect, isBusy, ghostMaterialFor, hooks,
} from "./editor.js";

const {
  TransformNode, Color4, Vector3, Quaternion, Matrix,
} = BABYLON;

export const ROT_AXES = ["y", "x", "z"];export const SCALE_AXES = ["all", "x", "y", "z"];

const HOVER_COLOR = new Color4(1.0, 0.55, 0.15, 1);
const SELECT_COLOR = new Color4(0.25, 0.75, 1.0, 1);
const EDGE_WIDTH = 5;
const EDGE_EPSILON = 0.96;      // higher draws more interior edges
const GHOST_ALPHA = 0.7;

let ghost = null;          // see armGhost()
let ghostToken = 0;        // guards against a slow load landing after a cancel
let hoverId = null;
let outlined = new Map();  // meshUniqueId -> mesh, everything currently edged
let lastWheelAt = 0;

// ------------------------------------------------------------------ setup

export function initInteract() {
  on("pointermove", onPointerMove);
  on("pointerdown", onPointerDown);
  on("pointerup", onPointerUp);
  on("click", onClick);
  on("dblclick", onDoubleClick);
  on("cancel", () => emit("escape"));
  // A selection change can both create and remove a hover: selecting the
  // element under the cursor has to drop it, and deselecting has to bring it
  // back without waiting for the pointer to move - otherwise Esc followed by a
  // key does nothing until you jiggle the mouse.
  on("selection", () => {
    if (drag || ghost) {
      if (hoverId && state.selection.includes(hoverId)) hoverId = null;
    } else {
      const hit = pickUnderCursor();
      const id = hit.kind === "entry" ? hit.id : null;
      hoverId = id && !state.selection.includes(id) ? id : null;
    }
    refreshOutlines();
    emit("current");
  });
  // Looking around sweeps the cursor across the whole scene, so hovering while
  // the right button is held would strobe the outline and keep changing what a
  // key would act on. Suppress it for the gesture, then re-evaluate on release
  // rather than waiting for the next mouse move.
  on("rmbdown", () => { setHover(null); setCursorHidden(true); });
  on("rmbup", () => {
    // the ghost owns the cursor while placing, so do not reveal it under one
    if (!ghost && !drag) setCursorHidden(false);
    if (drag || ghost) return;
    refreshHover();
  });
  on("placements", () => { hoverId = null; refreshOutlines(); });

  const viewport = document.getElementById("viewport");
  // Capture on the parent so this runs before Babylon's canvas handler; without
  // stopPropagation the camera would also zoom, and Ctrl+wheel is browser zoom.
  viewport.addEventListener("wheel", onWheel, { capture: true, passive: false });
}

// --------------------------------------------------------------- materials

// ------------------------------------------------------------------ ghost

function disposeClones(list) {
  for (const m of list) m.dispose(false, false);
  list.length = 0;
}

/**
 * Clone a prototype's parts under `root`. Clones share geometry with the
 * prototype and re-use a cached translucent copy of each real material, so the
 * ghost is textured - you can tell which face of a module you are looking at.
 */
function cloneParts(parts, root, prefix) {
  const out = [];
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    const c = part.mesh.clone(`${prefix}${i}`, root, true);
    c.setEnabled(true);
    c.isVisible = true;
    c.isPickable = false;
    c.material = ghostMaterialFor(part.mesh.material, "GHOST", GHOST_ALPHA);
    c.position.copyFrom(part.position);
    c.rotationQuaternion = part.rotationQuaternion.clone();
    c.scaling.copyFrom(part.scaling);
    out.push(c);
  }
  return out;
}

/** Local-space AABB centre of a prototype, cached on it. */
function protoCentre(proto) {
  if (proto._centre) return proto._centre;
  let min = null, max = null;
  for (const part of proto.parts) {
    const bb = part.mesh.getBoundingInfo().boundingBox;
    const m = Matrix.Compose(part.scaling, part.rotationQuaternion, part.position);
    for (const v of bb.vectors) {
      const p = Vector3.TransformCoordinates(v, m);
      min = min ? Vector3.Minimize(min, p) : p.clone();
      max = max ? Vector3.Maximize(max, p) : p.clone();
    }
  }
  proto._centre = min ? min.add(max).scale(0.5) : Vector3.Zero();
  return proto._centre;
}

export function ghostActive() { return !!ghost; }
export function ghostModule() { return ghost?.module || null; }
// so editor.js can hang the axis gizmo on the ghost without importing this
// module back and closing a cycle
hooks.ghostNode = () => (ghost && !ghost.root.isDisposed() ? ghost.root : null);

/**
 * Arm the ghost with a catalogue module. The ghost exists only for *placing*
 * new modules; moving something already placed is a drag (see beginDragCandidate).
 *
 * `opts.rotation` / `opts.scaling` seed the ghost, which is what makes Ctrl+D
 * feel like a duplicate rather than a fresh placement - the copy keeps the
 * orientation and any mirroring of the thing it came from.
 */
export async function armGhost(moduleId, opts = {}) {
  cancelGhost();
  if (!moduleId) { emit("current"); return null; }
  const token = ++ghostToken;
  const proto = await getProto(moduleId);
  if (token !== ghostToken) return null;          // cancelled while loading

  const root = new TransformNode("GHOST", state.scene);
  root.rotationQuaternion = Quaternion.Identity();
  const meshes = cloneParts(proto.parts, root, "GHOST_");

  ghost = {
    module: moduleId, root, meshes,
    centre: protoCentre(proto),
    quat: opts.rotation
      ? Quaternion.FromEulerAngles(
        opts.rotation[0] * Math.PI / 180,
        opts.rotation[1] * Math.PI / 180,
        opts.rotation[2] * Math.PI / 180)
      : Quaternion.Identity(),
    scaling: opts.scaling ? [...opts.scaling] : [1, 1, 1],
  };
  applyGhostTransform();
  moveGhostToCursor();
  setCursorHidden(true);
  emit("current");
  return ghost;
}

function applyGhostTransform() {
  if (!ghost) return;
  ghost.root.rotationQuaternion = ghost.quat.clone();
  ghost.root.scaling.set(...ghost.scaling);
}

/**
 * Turn `node` by `rad` about a **world** axis.
 *
 * Accumulating the angle in Euler space instead - read the triple, add a step,
 * write it back - looks equivalent and is not. `toEulerAngles()` decomposes
 * YXZ with X clamped to +/-90 deg, so once an element is yawed, stepping the X
 * component walks into gimbal lock: from a 90 deg yaw, four 90 deg X steps give
 * (90,90,0) -> (0,-90,180) -> (90,-270,0) -> (0,-90,180), which oscillates
 * instead of coming back round. Composing quaternions has no such failure.
 *
 * Order matters: Babylon's `a.multiply(b)` applies b first, so the incremental
 * rotation goes on the *left* to act in world space.
 */
function spinNode(node, unit, rad) {
  const q = Quaternion.RotationAxis(unit, rad);
  node.rotationQuaternion = q.multiply(node.rotationQuaternion || Quaternion.Identity());
}

const AXIS_UNITS = [Vector3.Right(), Vector3.Up(), Vector3.Forward()];

/**
 * Take the ghost's vertical reference: where the cursor sits on a plane through
 * it, and the build-plane height at that moment. Without re-taking this on
 * every axis switch the plane would jump by however far the cursor had moved
 * since the ghost was armed.
 */
function rebaseGhostVertical() {
  if (!ghost) return;
  const anchor = ghost.root.position.add(centreOffset());
  ghost.vFrom = cursorOnVerticalPlane(anchor);
  ghost.vGrid = state.gridY;
}

/**
 * Offset from the module's origin to the centre of its body, in world space.
 * Modules are not modelled around their origin - a 4 m wall sits 2 m to the
 * side of it, and a ceiling piece 4 m above it - so this is what keeps the
 * geometry under the pointer rather than the invisible origin.
 */
function centreOffset() {
  const m = Matrix.Compose(
    Vector3.FromArray(ghost.scaling), ghost.root.rotationQuaternion, Vector3.Zero());
  return Vector3.TransformCoordinates(ghost.centre, m);
}

function moveGhostToCursor() {
  if (!ghost) return;
  const off = centreOffset();

  // In Y mode the cursor drives the *build plane* rather than the ghost's own
  // height. Everything already hangs off the build plane - the grid draws
  // there, the ghost sits there, numpad +/- moves it - so raising the plane
  // keeps all of that in step, and switching back to X/Z simply resumes at the
  // new height instead of snapping the ghost back down.
  if (state.dragAxis === "y") {
    if (!ghost.vFrom) rebaseGhostVertical();
    const anchor = ghost.root.position.add(off);
    const now = cursorOnVerticalPlane(anchor);
    if (now && ghost.vFrom) {
      const s = state.snap.pos || 0;
      const raw = now.y - ghost.vFrom.y;
      const dy = s ? Math.round(raw / s) * s : raw;
      setGridElevation(ghost.vGrid + dy);
    }
    ghost.root.position.y = state.gridY;
    return;
  }

  // Track the cursor on a plane through the *body*, not the build plane. A
  // ceiling module sits metres above its origin, and following the build plane
  // would leave it floating far up-screen from the pointer - far enough that
  // you cannot reach the bottom of the viewport before running out of window.
  const p = cursorOnPlane(state.gridY + off.y);
  if (!p) return;
  const s = state.snap.pos || 0;
  const snap = (v) => (s ? Math.round(v / s) * s : v);
  // A single-axis mode leaves the other coordinate wherever the ghost already
  // is, so it slides along one line from where you put it.
  const x = state.dragAxis === "z" ? ghost.root.position.x : snap(p.x - off.x);
  const z = state.dragAxis === "x" ? ghost.root.position.z : snap(p.z - off.z);
  ghost.root.position.set(x, state.gridY, z);
}

function setCursorHidden(hidden) {
  const canvas = state.engine?.getRenderingCanvas();
  if (canvas) canvas.style.cursor = hidden ? "none" : "";
}

export function cancelGhost() {
  ghostToken++;
  setCursorHidden(false);
  if (!ghost) return;
  disposeClones(ghost.meshes);
  ghost.root.dispose();
  ghost = null;
  emit("current");
}

/** Commit the ghost: create a placement where it stands. */
export async function dropGhost() {
  if (!ghost) return null;
  const g = ghost;
  const position = g.root.position.clone();
  const quat = g.quat.clone();

  // The manifest stores Euler triples, so the orientation is decomposed only
  // here, once, at the boundary - never accumulated in that form.
  const made = await placeAt(g.module, position, {
    rotation: eulerOf(g.root), scale: g.scaling,
  });
  // keep the same module armed so a run of tiles is just repeated clicks
  if (ghost) {
    ghost.quat = quat;
    ghost.scaling = g.scaling.slice();
    applyGhostTransform();
  }
  emit("current");
  return made;
}

// ---------------------------------------------------------------- marquee
//
// A rubber band drawn with the left button. The overlay is a plain DOM element
// rather than anything in the scene: it has to be crisp at exactly one pixel
// and must never be pickable, and a div is both for free.

let marquee = null;

export function marqueeRect() {
  if (!marquee) return null;
  return {
    x0: Math.min(marquee.x0, marquee.x1), y0: Math.min(marquee.y0, marquee.y1),
    x1: Math.max(marquee.x0, marquee.x1), y1: Math.max(marquee.y0, marquee.y1),
  };
}

function canvasPoint(ev) {
  const r = state.engine.getRenderingCanvas().getBoundingClientRect();
  return { x: ev.clientX - r.left, y: ev.clientY - r.top };
}

function beginMarquee(ev) {
  const p = canvasPoint(ev);
  const el = document.createElement("div");
  el.className = "marquee";
  document.getElementById("viewport").appendChild(el);
  marquee = {
    x0: p.x, y0: p.y, x1: p.x, y1: p.y, el, moved: false,
    additive: ev.ctrlKey || ev.metaKey || ev.shiftKey,
  };
  drawMarquee();
}

function drawMarquee() {
  const r = marqueeRect();
  const box = state.engine.getRenderingCanvas().getBoundingClientRect();
  const host = document.getElementById("viewport").getBoundingClientRect();
  const dx = box.left - host.left, dy = box.top - host.top;
  Object.assign(marquee.el.style, {
    left: `${r.x0 + dx}px`, top: `${r.y0 + dy}px`,
    width: `${r.x1 - r.x0}px`, height: `${r.y1 - r.y0}px`,
  });
}

function updateMarquee(ev) {
  if (!marquee) return;
  const p = canvasPoint(ev);
  marquee.x1 = p.x; marquee.y1 = p.y;
  if (Math.hypot(marquee.x1 - marquee.x0, marquee.y1 - marquee.y0) > 3) marquee.moved = true;
  drawMarquee();
}

function endMarquee(commit) {
  if (!marquee) return false;
  const { moved, additive } = marquee;
  const rect = marqueeRect();
  marquee.el.remove();
  marquee = null;
  if (!commit || !moved) return false;      // that was a click, not a rectangle

  const hits = elementsInRect(rect);
  select(additive ? [...new Set([...state.selection, ...hits])] : hits);
  emit("current");
  return true;
}

export function cancelMarquee() { return endMarquee(false); }
export function isMarqueeing() { return !!marquee?.moved; }

// ------------------------------------------------------------------- drag
//
// Dragging moves the real elements rather than a ghost: it has to work for a
// whole selection at once, and seeing the actual geometry land is the point.
// The delta is measured on a horizontal plane at the selection's own height so
// the motion tracks the cursor instead of being skewed by perspective, and the
// *delta* is snapped rather than each element - that keeps a group's internal
// spacing exactly as authored even if it was placed off-grid.

let drag = null;

export function isDragging() { return !!drag?.active; }

/** Which mapping the current drag is using - for tests and diagnostics. */
export function dragMode() { return drag?.mode || null; }

function beginDragCandidate(id, ev, pickedPoint) {
  const alreadySelected = state.selection.includes(id);
  const ids = alreadySelected ? [...state.selection] : [id];

  const entries = ids.map(entryOf).filter(Boolean);
  if (!entries.length) return false;

  // Anchor on the point actually clicked, not the element's origin.
  //
  // Origins sit at the base, so a column or door frame grabbed near the top,
  // from a camera near the floor, needs a ray that goes *up* - and a horizontal
  // plane down at the base is then behind the camera (t <= 0), so there was no
  // reference point and the piece simply refused to move.
  const centre = entries
    .reduce((a, e) => a.addInPlace(e.node.position), Vector3.Zero())
    .scale(1 / entries.length);
  const anchor = pickedPoint ? pickedPoint.clone() : centre;

  drag = {
    active: false,
    // Selecting here would make a press that never moves - a long hold - select
    // the element anyway. It is deferred to whichever the gesture turns out to
    // be: a real drag selects what it moves, and a press that stays put is only
    // a selection if it was quick enough to count as a click.
    pendingSelect: alreadySelected ? null : ids,
    startX: ev.clientX, startY: ev.clientY,
    entries,
    origins: entries.map((e) => e.node.position.clone()),
  };
  anchorDrag(anchor);
  return true;
}

/**
 * Decide how a horizontal drag maps the mouse, and take its reference points.
 *
 * Two mappings, because one is not enough:
 *
 * - **plane** - intersect the cursor ray with the horizontal plane through the
 *   grabbed point. The grabbed point stays exactly under the cursor, which is
 *   what you want looking down at the floor.
 * - **screen** - move along the camera's own right/forward axes by the pixel
 *   delta, scaled to world units at the grabbed point's distance.
 *
 * The plane mapping *inverts* once the plane is above the camera: looking up at
 * a column, higher on screen means nearer, so pushing the mouse forward pulls
 * the piece towards you. It also runs to infinity as the ray approaches
 * parallel. Neither is fixable by choosing a different height - at eye level
 * the horizontal plane is simply edge-on. The mapping is chosen once, at the
 * start of the gesture, so it can never switch mid-drag and jump.
 */
function anchorDrag(anchor) {
  const cam = state.camera;
  const engine = state.engine;
  const ray = state.scene.createPickingRay(
    state.scene.pointerX, state.scene.pointerY, Matrix.Identity(), cam);

  const belowCamera = anchor.y < cam.position.y - 0.05;
  const steepEnough = Math.abs(ray.direction.y) > 0.25;   // ~15 deg off the plane

  drag.planeY = anchor.y;
  drag.centre = anchor;
  drag.fromV = cursorOnVerticalPlane(anchor);
  drag.mode = belowCamera && steepEnough ? "plane" : "screen";

  if (drag.mode === "plane") {
    drag.from = cursorOnPlane(anchor.y);
    if (drag.from) return;
    drag.mode = "screen";                      // plane turned out to be unusable
  }

  const right = cam.getDirection(Vector3.Right());
  const fwd = cam.getDirection(Vector3.Forward());
  right.y = 0; fwd.y = 0;
  if (right.lengthSquared() > 1e-6) right.normalize();
  if (fwd.lengthSquared() > 1e-6) fwd.normalize();
  drag.right = right;
  drag.fwd = fwd;
  drag.fromPx = { x: state.scene.pointerX, y: state.scene.pointerY };
  // world units per pixel at the grabbed point's distance
  const dist = Vector3.Distance(cam.position, anchor);
  drag.unitsPerPx = (2 * dist * Math.tan(cam.fov / 2)) / engine.getRenderHeight();
}

/**
 * Which way a drag moves things.
 *
 * `xz` is the floor plane and the default - modular kits are laid out on the
 * horizontal - with `y` for stacking decks and hanging ceiling parts, and `x`
 * and `z` for sliding a piece along one axis without the other drifting.
 *
 * V cycles rather than toggles, in the order the combo lists them.
 *
 * Safe to change mid-drag: the drag re-anchors on the new axis so the element
 * does not jump.
 */
export const DRAG_AXES = ["xz", "y", "x", "z"];

export function setDragAxis(axis) {
  if (!DRAG_AXES.includes(axis)) return state.dragAxis;
  state.dragAxis = axis;
  rebaseDrag();
  if (ghost) {
    rebaseGhostVertical();
    moveGhostToCursor();
  }
  emit("modes");
  return state.dragAxis;
}

export function toggleDragAxis() {
  const i = DRAG_AXES.indexOf(state.dragAxis);
  return setDragAxis(DRAG_AXES[(i + 1) % DRAG_AXES.length]);
}
/** Re-anchor a drag in progress, so switching axis does not jump the element. */
export function rebaseDrag() {
  if (!drag) return;
  const live = drag.entries
    .reduce((a, e) => a.addInPlace(e.node.position), Vector3.Zero())
    .scale(1 / drag.entries.length);
  drag.origins = drag.entries.map((e) => e.node.position.clone());
  anchorDrag(live);
}

function updateDrag(ev) {
  if (!drag) return;
  if (!drag.active) {
    if (Math.hypot(ev.clientX - drag.startX, ev.clientY - drag.startY) <= 4) return;
    drag.active = true;
    if (drag.pendingSelect) { select(drag.pendingSelect); drag.pendingSelect = null; }
    pushUndo();
    setCursorHidden(true);          // same as placing: the element is the cursor
    emit("current");
  }
  const s = state.snap.pos || 0;
  const snap = (v) => (s ? Math.round(v / s) * s : v);

  if (state.dragAxis === "y") {
    const now = cursorOnVerticalPlane(drag.centre);
    if (!now) return;
    // The reference can be missing if the ray was unusable at press time (see
    // beginDragCandidate); take it at the first frame it works instead of
    // refusing to move for the rest of the gesture.
    if (!drag.fromV) { drag.fromV = now; return; }
    const dy = snap(now.y - drag.fromV.y);
    for (let i = 0; i < drag.entries.length; i++) {
      const o = drag.origins[i];
      drag.entries[i].node.position.set(o.x, o.y + dy, o.z);
    }
    emit("transform");
    return;
  }

  const now = cursorOnPlane(drag.planeY);
  if (drag.mode === "plane") {
    if (!now) return;
    if (!drag.from) { drag.from = now; return; }
    applyFlatDelta(now.x - drag.from.x, now.z - drag.from.z, snap);
    return;
  }

  const dpx = state.scene.pointerX - drag.fromPx.x;
  const dpy = state.scene.pointerY - drag.fromPx.y;
  // screen up (negative dpy) pushes the element away, which is the direction
  // the plane mapping inverts when the plane sits above the camera
  const move = drag.right.scale(dpx * drag.unitsPerPx)
    .add(drag.fwd.scale(-dpy * drag.unitsPerPx));
  applyFlatDelta(move.x, move.z, snap);
}

function applyFlatDelta(rawX, rawZ, snap) {
  // A single-axis mode simply drops the other component: the cursor still
  // tracks on the floor plane, but only one coordinate is allowed through.
  const dx = state.dragAxis === "z" ? 0 : snap(rawX);
  const dz = state.dragAxis === "x" ? 0 : snap(rawZ);
  for (let i = 0; i < drag.entries.length; i++) {
    const o = drag.origins[i];
    drag.entries[i].node.position.set(o.x + dx, o.y, o.z + dz);
  }
  emit("transform");
}

function endDrag(commit) {
  if (!drag) return false;
  const wasActive = drag.active;
  if (!commit && wasActive) {
    for (let i = 0; i < drag.entries.length; i++) {
      drag.entries[i].node.position.copyFrom(drag.origins[i]);
    }
    emit("transform");
  }
  drag = null;
  if (wasActive) { setCursorHidden(false); emit("current"); }
  return wasActive;
}

export function cancelDrag() { return endDrag(false); }

// ---------------------------------------------------- hover and selection
// Outlines use per-instance edges rendering. HighlightLayer refuses an
// InstancedMesh outright, and renderOutline is read off the source mesh so it
// would outline every copy of the module at once; edges rendering is the only
// one that follows a single instance - and it tracks its world matrix, so
// nothing has to be re-synced after an edit.

function edgeMeshes(entry) {
  return entry?.node ? entry.node.getChildMeshes() : [];
}

function applyEdges(mesh, color) {
  if (!mesh.edgesRenderer) mesh.enableEdgesRendering(EDGE_EPSILON);
  mesh.edgesWidth = EDGE_WIDTH;
  mesh.edgesColor = color;
  outlined.set(mesh.uniqueId, mesh);
}

function refreshOutlines() {
  const wanted = new Map();
  // A selected element is never also hovered (see setHover), so these two can
  // no longer disagree about the same mesh.
  if (hoverId) {
    for (const m of edgeMeshes(entryOf(hoverId))) wanted.set(m.uniqueId, [m, HOVER_COLOR]);
  }
  for (const id of state.selection) {
    for (const m of edgeMeshes(entryOf(id))) wanted.set(m.uniqueId, [m, SELECT_COLOR]);
  }

  for (const [key, mesh] of outlined) {
    if (wanted.has(key)) continue;
    if (!mesh.isDisposed()) mesh.disableEdgesRendering();
    outlined.delete(key);
  }
  for (const [, [mesh, color]] of wanted) applyEdges(mesh, color);
}

/**
 * Selection wins outright: an element that is selected is never reported as
 * hovered, even with the cursor on it. One element, one state - otherwise the
 * outline colour and "what will this key act on?" disagree with each other.
 */
function setHover(id) {
  const next = id && entryOf(id) && !state.selection.includes(id) ? id : null;
  if (next === hoverId) return;
  hoverId = next;
  refreshOutlines();
  emit("current");
}

export function hoveredId() { return hoverId; }

/** Re-pick under a cursor that has not moved - after a right-drag. */
function refreshHover() {
  if (drag || ghost || isRmbDown()) return;
  const hit = pickUnderCursor();
  setHover(hit.kind === "entry" ? hit.id : null);
}

/**
 * What an edit acts on, in priority order: the ghost being placed, then
 * whatever the cursor is over, then the selection.
 *
 * Hover outranks the selection deliberately - pointing at something is a more
 * immediate statement of intent than a selection made earlier, and it saves
 * clearing the selection before acting on a different element.
 */
export function currentElement() {
  if (ghost) return { kind: "ghost", ids: [], module: ghost.module };
  if (hoverId) {
    const e = entryOf(hoverId);
    if (e) return { kind: "hover", ids: [hoverId], module: e.name || e.module || e.type };
  }
  if (state.selection.length) {
    const first = entryOf(state.selection[0]);
    return {
      kind: "selection",
      ids: [...state.selection],
      module: state.selection.length > 1
        ? `${state.selection.length} selected`
        : (first?.name || first?.module || first?.type || state.selection[0]),
    };
  }
  return null;
}

/** Placement entries a wheel edit should touch - hover first, as above. */
function wheelTargets() {
  const hovered = hoverId ? entryOf(hoverId) : null;
  if (hovered) return [hovered];
  return state.selection.map(entryOf).filter(Boolean);
}

/**
 * Shared pivot for an Alt+wheel group rotation: the centre of the selection's
 * bounding box, snapped to the move grid so a rotated group stays on-grid.
 */
export function sharedPivot(entries) {
  let min = null, max = null;
  for (const e of entries) {
    const b = worldBounds(e.node);
    if (!b) continue;
    min = min ? Vector3.Minimize(min, b.min) : b.min.clone();
    max = max ? Vector3.Maximize(max, b.max) : b.max.clone();
  }
  if (!min) return Vector3.Zero();
  const c = min.add(max).scale(0.5);
  const s = state.snap.pos || 0;
  if (!s) return c;
  return new Vector3(
    Math.round(c.x / s) * s, Math.round(c.y / s) * s, Math.round(c.z / s) * s);
}

// --------------------------------------------------------------- handlers

function onPointerDown(req) {
  if (isBusy()) return;
  if (ghost || state.markerBrush) return;
  const ev = req.event;
  if (ev.altKey) return;                               // eyedropper
  const hit = pickUnderCursor();

  // A drag that starts on nothing can only be a rectangle; starting on a module
  // is the ambiguous case, which is what the mode is for.
  if (state.selectMode || hit.kind !== "entry") {
    beginMarquee(ev);
    req.capture = true;
    return;
  }
  if (ev.ctrlKey || ev.metaKey || ev.shiftKey) return; // multi-select click
  if (beginDragCandidate(hit.id, ev, hit.pick?.pickedPoint)) req.capture = true;
}

function onPointerUp() {
  if (endMarquee(true)) return;
  endDrag(true);
}

function onPointerMove(ev) {
  if (isBusy()) return;
  if (marquee) { updateMarquee(ev); return; }
  if (drag) { updateDrag(ev); return; }
  if (ghost) {
    if (!ghost.following) {
      // first real movement after a grab: start following and hide the pointer
      ghost.following = true;
      setCursorHidden(true);
    }
    moveGhostToCursor();
    return;
  }
  // No hover while the camera is being driven: the picking would be wasted
  // work on every frame of a look, and the outline would flicker across
  // everything the cursor sweeps past.
  if (isRmbDown()) return;
  const hit = pickUnderCursor();
  setHover(hit.kind === "entry" ? hit.id : null);
}

async function onClick(ev) {
  if (isBusy()) return;
  if (ghost) { await dropGhost(); return; }

  const hit = pickUnderCursor();
  if (hit.kind === "entry") {
    // Alt-click is an eyedropper: arm the module you are pointing at.
    if (ev.altKey && hit.entry.module) {
      emit("pickmodule", hit.entry.module);
      return;
    }
    // Both modifiers add. Shift lost that job for a while to mean "see through
    // the door portals"; Shift+H replaced that, so it has it back.
    if (ev.ctrlKey || ev.metaKey || ev.shiftKey) { toggleSelect(hit.id); return; }
    select([hit.id]);                       // double-click picks it up instead
    return;
  }

  if (hit.kind === "ground") {
    if (state.markerBrush) {
      const s = state.snap.pos || 0;
      const p = new Vector3(
        s ? Math.round(hit.point.x / s) * s : hit.point.x,
        state.gridY,
        s ? Math.round(hit.point.z / s) * s : hit.point.z);
      emit("markerdrop", { kind: state.markerBrush, position: p });
      return;
    }
    select([]);
  }
}

/** Double-click frames the element under the cursor. */
async function onDoubleClick(ev) {
  if (isBusy() || ghost || ev.altKey || ev.ctrlKey || ev.metaKey) return;
  const hit = pickUnderCursor();
  if (hit.kind === "entry") { select([hit.id]); emit("focus"); }
}

/**
 * Wheel bindings:
 *   right button + wheel  adjust the fly speed
 *   Shift + wheel         resize the current element
 *   anything else         dolly the camera (and stop the browser page-zooming)
 *
 * The wheel no longer rotates. It collided with the one thing the wheel is
 * expected to do in a 3D view - zoom - and every rotation it could do is on
 * `Q`/`E` anyway, which act on the hovered element without needing a selection
 * first.
 */
function onWheel(ev) {
  const dir = ev.deltaY < 0 ? 1 : -1;
  ev.preventDefault();
  ev.stopPropagation();
  if (isBusy()) return;                 // and the page must still not scroll

  // The right button already means "I am driving the camera", so the wheel
  // adjusting how fast reads naturally and cannot collide with editing.
  if (isRmbDown()) {
    // The HUD no longer carries a speed row, so the status line is the only
    // feedback this control has - and a fly speed you cannot see is one you
    // will wonder about the next time the camera feels wrong.
    emit("status", `fly speed ${nudgeMoveSpeed(dir)} m/s`);
    return;
  }

  const editable = !!ghost || wheelTargets().length > 0;
  if (ev.shiftKey && !ev.ctrlKey && !ev.metaKey && editable) {
    scaleCurrent(dir);
    return;
  }
  dollyCamera(dir);
}

function beginWheelEdit() {
  // one undo entry per gesture rather than per wheel notch
  const now = performance.now();
  const fresh = now - lastWheelAt > 400;
  lastWheelAt = now;
  return fresh;
}

export function rotateCurrent(dir, aboutPivot = false) {
  const step = (state.snap.rot || 90) * dir;
  // the axis name is a cycle label, not a component index - map it or "y"
  // would turn about X
  const axis = "xyz".indexOf(state.rotAxis);
  if (axis < 0) return;
  const rad = step * Math.PI / 180;
  const unit = AXIS_UNITS[axis];

  if (ghost) {
    ghost.quat = Quaternion.RotationAxis(unit, rad).multiply(ghost.quat);
    applyGhostTransform();
    moveGhostToCursor();          // keep the body centred as the offset rotates
    emit("current");
    return;
  }

  const targets = wheelTargets();
  if (!targets.length) return;
  if (beginWheelEdit()) pushUndo();

  if (aboutPivot) {
    const pivot = sharedPivot(targets);
    const q = Quaternion.RotationAxis(unit, rad);
    const m = Matrix.Identity();
    q.toRotationMatrix(m);
    for (const e of targets) {
      const rel = e.node.position.subtract(pivot);
      e.node.position.copyFrom(pivot.add(Vector3.TransformCoordinates(rel, m)));
      spinNode(e.node, unit, rad);
    }
  } else {
    // each element turns about its own origin, which is what a row of props wants
    for (const e of targets) spinNode(e.node, unit, rad);
  }
  emit("transform");
  emit("current");
}

export function scaleCurrent(dir) {
  const step = (state.snap.scale || 0.1) * dir;
  // Work on the magnitude and keep the sign, so a mirrored (negative) element
  // can still be resized instead of being snapped back to +0.05 on the first
  // notch.
  const bump = (v) => {
    const sign = v < 0 ? -1 : 1;
    return sign * Math.max(0.05, Math.abs(v) + step);
  };
  const apply = (arr) => {
    if (state.scaleAxis === "all") {
      for (let i = 0; i < 3; i++) arr[i] = bump(arr[i]);
    } else {
      const i = "xyz".indexOf(state.scaleAxis);
      arr[i] = bump(arr[i]);
    }
    return arr;
  };

  if (ghost) {
    ghost.scaling = apply(ghost.scaling.slice());
    applyGhostTransform();
    moveGhostToCursor();
    emit("current");
    return;
  }

  const targets = wheelTargets();
  if (!targets.length) return;
  if (beginWheelEdit()) pushUndo();
  for (const e of targets) e.node.scaling.set(...apply(e.node.scaling.asArray()));
  emit("transform");
  emit("current");
}

// ------------------------------------------------------------------ modes

/**
 * Mirror the current element(s) by negating one scale component. The axis comes
 * from the Scale-axis setting; "all" would be a point inversion rather than a
 * mirror, so it is treated as X - the horizontal flip a modular kit almost
 * always wants. Returns the axis actually used, for the status line.
 */
export function flipCurrent() {
  const axis = state.scaleAxis === "all" ? "x" : state.scaleAxis;
  const i = "xyz".indexOf(axis);
  if (i < 0) return null;

  if (ghost) {
    ghost.scaling[i] = -ghost.scaling[i];
    applyGhostTransform();
    moveGhostToCursor();
    emit("current");
    return { axis, count: 1, kind: "ghost" };
  }

  // while a drag is in flight the dragged set *is* the selection, so this
  // covers "flip what I am dragging" without a separate branch
  const targets = wheelTargets().filter((e) => !e.type);
  if (!targets.length) return null;
  pushUndo();
  for (const e of targets) {
    const s = e.node.scaling.asArray();
    s[i] = -s[i];
    e.node.scaling.set(...s);
  }
  emit("transform");
  emit("current");
  return { axis, count: targets.length, kind: isDragging() ? "drag" : "selection" };
}

export function cycleRotAxis(dir = 1) {
  const i = ROT_AXES.indexOf(state.rotAxis);
  state.rotAxis = ROT_AXES[(i + dir + ROT_AXES.length) % ROT_AXES.length];
  emit("modes");
}

export function cycleScaleAxis(dir = 1) {
  const i = SCALE_AXES.indexOf(state.scaleAxis);
  state.scaleAxis = SCALE_AXES[(i + dir + SCALE_AXES.length) % SCALE_AXES.length];
  emit("modes");
}
