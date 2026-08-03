// The authoring interaction model.
//
// There are no gizmos and no transform modes. A "ghost" - a translucent clone
// of a module - follows the cursor snapped to the build plane; clicking drops
// it. A placed element is moved by dragging it directly. Whatever the ghost
// holds, or whatever the cursor hovers, or failing both the selection, is the
// *current element*: Shift+wheel turns it, Ctrl+wheel resizes it, Alt+F flips it.

import { getProto } from "./kit.js";
import {
  state, emit, on, pushUndo, placeAt, select, toggleSelect, entryOf,
  pickUnderCursor, setGridElevation, eulerOf, setEuler, cursorOnGrid, cursorOnPlane,
  worldBounds, dollyCamera, isRmbDown, nudgeMoveSpeed, cursorOnVerticalPlane,
  elementsInRect, isBusy, ghostMaterialFor, hooks, nearestToCursor, applyVisibility,
  constrainMove, axisBasis,
} from "./editor.js";

const {
  TransformNode, Color4, Vector3, Quaternion, Matrix,
} = BABYLON;

export const ROT_AXES = ["y", "x", "z"];export const SCALE_AXES = ["all", "x", "y", "z"];

const HOVER_COLOR = new Color4(1.0, 0.55, 0.15, 1);
const SELECT_COLOR = new Color4(0.25, 0.75, 1.0, 1);
const EDGE_EPSILON = 0.96;      // higher draws more interior edges
const EDGE_PX = 3.5;            // how thick an outline should look, in pixels
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
  // The camera moves without emitting anything, and an outline drawn for where
  // the camera *was* is the whole problem, so this rides the render loop.
  state.scene?.onBeforeRenderObservable.add(resizeOutlines);

  const viewport = document.getElementById("viewport");
  // Capture on the parent so this runs before Babylon's canvas handler; without
  // stopPropagation the camera would also zoom, and Ctrl+wheel is browser zoom.
  viewport.addEventListener("wheel", onWheel, { capture: true, passive: false });

  // Middle button deletes what Del would delete. `preventDefault` on the
  // *mousedown* is what suppresses Windows' autoscroll, which otherwise drops
  // a scroll anchor on the page and swallows the following mouse moves.
  viewport.addEventListener("mousedown", (e) => {
    if (e.button !== 1) return;
    e.preventDefault();
  }, { capture: true });
  viewport.addEventListener("auxclick", (e) => {
    if (e.button !== 1 || isBusy()) return;
    e.preventDefault();
    emit("deletecurrent");
  }, { capture: true });
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

/**
 * Local-space AABB of a prototype's parts, cached on it.
 *
 * Modules are not modelled around their origin - a 4 m wall sits 2 m to the
 * side of it - so this is what lets the geometry sit under the pointer rather
 * than the invisible origin.
 */
function protoBounds(proto) {
  if (proto._bounds) return proto._bounds;
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
  proto._bounds = { min: min || Vector3.Zero(), max: max || Vector3.Zero() };
  return proto._bounds;
}

/** Local-space AABB centre of a prototype, cached on it. */
function protoCentre(proto) {
  const b = protoBounds(proto);
  return b.min.add(b.max).scale(0.5);
}

/**
 * Build the ghost from a list of items, each a module with its own offset,
 * rotation and scale inside the group.
 *
 * One item is a palette placement; several are a grabbed selection or a
 * multi-element duplicate. The group root carries the *shared* rotation and
 * scale, so R and F turn the whole set about its own centre, exactly as they
 * turn a single module about its origin.
 */
async function buildGhost(specs, opts = {}) {
  const root = new TransformNode("GHOST", state.scene);
  root.rotationQuaternion = Quaternion.Identity();

  const items = [];
  const meshes = [];
  let min = null, max = null;
  for (const spec of specs) {
    const node = new TransformNode(`GHOST_ITEM_${items.length}`, state.scene);
    node.parent = root;
    node.position.copyFrom(spec.offset || Vector3.Zero());
    node.rotationQuaternion = spec.rotation
      ? Quaternion.FromEulerAngles(
        spec.rotation[0] * Math.PI / 180,
        spec.rotation[1] * Math.PI / 180,
        spec.rotation[2] * Math.PI / 180)
      : Quaternion.Identity();
    if (spec.scaling) node.scaling.set(...spec.scaling);

    // A collision primitive has no kit prototype: it is generated geometry, so
    // the ghost builds the same unit shape and wears the ghost material over
    // it. Everything downstream - offsets, group turns, the drop - is identical.
    let b;
    if (spec.collider) {
      const parts = hooks.buildColliderMesh(spec.collider, `GHOST_${items.length}`);
      if (!parts?.length) continue;
      for (const shape of parts) {
        shape.parent = node;
        shape.isPickable = false;
        shape.material = ghostMaterialFor(shape.material, "GHOST", GHOST_ALPHA)
          || shape.material;
        meshes.push(shape);
      }
      b = { min: new Vector3(-0.5, -0.5, -0.5), max: new Vector3(0.5, 0.5, 0.5) };
    } else {
      const proto = await getProto(spec.module);
      const parts = cloneParts(proto.parts, node, `GHOST_${items.length}_`);
      meshes.push(...parts);
      b = protoBounds(proto);
    }
    items.push({
      module: spec.module || null, collider: spec.collider || null,
      node, sourceId: spec.sourceId || null,
      // Kept alongside the node so the drop can *compose* the world transform
      // rather than decompose a matrix. A mirrored element has a negative
      // determinant, which has no unique rotation/scale split - Babylon's
      // decompose() legitimately moved a scale of [-1,1,1] onto Y with a
      // compensating rotation, which looks identical and reads as a different
      // ship in the manifest.
      quat: node.rotationQuaternion.clone(),
      scaling: node.scaling.asArray(),
    });

    // group AABB, in root-local space, so the body sits under the cursor
    const m = Matrix.Compose(node.scaling, node.rotationQuaternion, node.position);
    for (const v of [b.min, b.max, new Vector3(b.min.x, b.min.y, b.max.z),
      new Vector3(b.min.x, b.max.y, b.min.z), new Vector3(b.max.x, b.min.y, b.min.z),
      new Vector3(b.max.x, b.max.y, b.min.z), new Vector3(b.max.x, b.min.y, b.max.z),
      new Vector3(b.min.x, b.max.y, b.max.z)]) {
      const p = Vector3.TransformCoordinates(v, m);
      min = min ? Vector3.Minimize(min, p) : p.clone();
      max = max ? Vector3.Maximize(max, p) : p.clone();
    }
  }

  return {
    // the module a single-item ghost holds, for the HUD and repeat placement
    module: items.length === 1 ? items[0].module : null,
    // likewise for a collision primitive, which has a kind instead of a module
    collider: items.length === 1 ? items[0].collider : null,
    root, items, meshes,
    centre: min ? min.add(max).scale(0.5) : Vector3.Zero(),
    mode: opts.mode || "place",
    // A copy keeps its source's height without the build plane being dragged
    // up to it - which used to change where everything placed afterwards went.
    baseY: Number.isFinite(opts.baseY) ? opts.baseY : null,
    quat: opts.rotation
      ? Quaternion.FromEulerAngles(
        opts.rotation[0] * Math.PI / 180,
        opts.rotation[1] * Math.PI / 180,
        opts.rotation[2] * Math.PI / 180)
      : Quaternion.Identity(),
    scaling: opts.scaling ? [...opts.scaling] : [1, 1, 1],
  };
}

export function ghostActive() { return !!ghost; }
export function ghostModule() { return ghost?.module || null; }
/** The collision primitive a single-item ghost holds, if it is one. */
export function ghostCollider() { return ghost?.collider || null; }
/** How many elements the ghost is carrying - 1 for a palette placement. */
export function ghostCount() { return ghost?.items.length || 0; }
/** "place" a new module, "copy" a duplicate, or "move" a grabbed selection. */
export function ghostMode() { return ghost?.mode || null; }
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
  floorDragAxisForGhost();
  const token = ++ghostToken;
  const built = await buildGhost([{ module: moduleId }], opts);
  if (token !== ghostToken) { disposeGhost(built); return null; }  // cancelled while loading

  ghost = built;
  applyGhostTransform();
  moveGhostToCursor();
  setCursorHidden(true);
  emit("current");
  return ghost;
}

/**
 * Arming a ghost puts the drag axis back on the floor plane.
 *
 * In Y mode the cursor drives the *build plane* rather than the ghost's own
 * height: the plane follows however far the cursor has travelled vertically,
 * and the ghost goes with it. That is the point of the mode when you are moving
 * something that already exists, and useless when you are arming something new
 * - the ghost leaves the top or bottom of the screen before you have chosen
 * where to put it. `Ctrl+D` has it worse still: a copy is given its source's
 * height so it appears beside it, and Y mode clears exactly that.
 *
 * It lives here rather than in the click handlers because there are four ways
 * to arm one - a palette tile, the eyedropper, a collision shape button, a
 * duplicate - and one of them was missed the first time. `setDragAxis` emits
 * `modes`, so the combo follows on its own.
 *
 * Deliberately *not* applied to an `M` carry: raising something is a perfectly
 * good reason to be in Y mode, and that gesture moves what already exists.
 */
function floorDragAxisForGhost() {
  if (state.dragAxis !== "xz") setDragAxis("xz");
}

/**
 * Arm the ghost with a collision primitive. Same contract as armGhost: nothing
 * exists until you click, and the ghost stays armed afterwards so a run of
 * boxes along a wall is just repeated clicks.
 */
export async function armColliderGhost(kind, opts = {}) {
  cancelGhost();
  if (!kind) { emit("current"); return null; }
  floorDragAxisForGhost();
  const token = ++ghostToken;
  const built = await buildGhost([{ collider: kind }], opts);
  if (token !== ghostToken) { disposeGhost(built); return null; }

  ghost = built;
  applyGhostTransform();
  moveGhostToCursor();
  setCursorHidden(true);
  emit("current");
  return ghost;
}

/**
 * Pick the selection up into the ghost, so it follows the cursor with no button
 * held. `G`, the way Blender's grab works.
 *
 * The originals are hidden rather than moved: the ghost *is* the preview, and
 * leaving solid copies behind would read as "these have been duplicated". `Esc`
 * brings them back untouched, which is why they are hidden and not deleted.
 *
 * Markers are skipped - a door has no kit prototype to clone - and a selection
 * of nothing but markers simply does not grab, leaving the drag to handle it.
 */
export async function grabSelection(opts = {}) {
  const entries = state.selection.map(entryOf)
    .filter((e) => e && (e.module || e.type === "collider"))
    // A *module* can only be on the bench once: the association rule needs one
    // answer to "which element is this shape on", and two instances give two.
    // Shapes are a different matter - a hull is often several boxes. The rule
    // belongs here rather than only at the Ctrl+D key, which filtered a list it
    // then did not pass on, so a module selected alongside a shape was copied
    // anyway. Carrying one with `M` is still fine; it is copying that is not.
    .filter((e) => !(opts.copy && state.collisionMode && e.stage && e.type !== "collider"));
  if (!entries.length) return null;
  // A copy is a new thing being placed, so it wants the floor plane; a carry is
  // moving what is already there, and raising it is a fair reason to be in Y.
  if (opts.copy) floorDragAxisForGhost();
  cancelGhost();
  const token = ++ghostToken;

  // The anchor is the element nearest the cursor, so the set keeps its shape
  // around the piece you were pointing at rather than jumping by its centroid.
  const anchorId = nearestToCursor(entries.map((e) => e.id)) || entries[0].id;
  const anchor = entries.find((e) => e.id === anchorId) || entries[0];
  const base = anchor.node.position.clone();

  const specs = entries.map((e) => ({
    module: e.module || null,
    collider: e.type === "collider" ? e.kind : null,
    offset: e.node.position.subtract(base),
    rotation: eulerOf(e.node),
    scaling: e.node.scaling.asArray(),
    sourceId: opts.copy ? null : e.id,
  }));
  const built = await buildGhost(specs, { mode: opts.copy ? "copy" : "move" });
  if (token !== ghostToken) { disposeGhost(built); return null; }

  ghost = built;
  if (!opts.copy) for (const e of entries) e.node.setEnabled(false);
  // The build plane follows, or a grab from an upper deck would drop back to
  // the floor - the same reason Ctrl+D moves it. For a lone primitive the
  // plane means its *base*, matching the corner-first drop: referencing the
  // centre instead would raise it by half its height on every grab.
  const ref = base.y - (entries.length === 1 && entries[0].type === "collider"
    ? colliderHalf(entries[0].node.scaling.asArray(), entries[0].node.rotationQuaternion).y
    : 0);
  if (Math.abs(ref - state.gridY) > 1e-6) setGridElevation(ref);

  // A grab is *relative*: the elements stay exactly where they are and then
  // follow how far the mouse moves, the way Blender's G works. Teleporting them
  // onto the cursor instead threw an element halfway across the room the moment
  // you pressed the key - and re-snapped anything deliberately placed off the
  // grid. Anchoring here, before the first move, is what makes "stay in place"
  // the starting state rather than a special case.
  ghost.root.position.copyFrom(base);
  ghost.anchor = { cursor: cursorOnPlane(state.gridY + centreOffset().y), base: base.clone() };
  // Taken from the anchor element, and taken once: it is the piece you grabbed,
  // and re-reading it would let a turn mid-carry swing the axes about.
  ghost.anchorNode = anchor.node;
  ghost.basis = axisBasis(anchor.node);
  applyGhostTransform();
  moveGhostToCursor();
  setCursorHidden(true);
  emit("current");
  return ghost;
}

function disposeGhost(g) {
  if (!g) return;
  disposeClones(g.meshes);
  for (const it of g.items) it.node.dispose();
  g.root.dispose();
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

/**
 * World half-extents of a unit shape carrying this scale and turn - the sum of
 * the absolute basis components, so a turned box reports the box that actually
 * contains it.
 */
function unitHalf(m) {
  return new Vector3(
    (Math.abs(m.m[0]) + Math.abs(m.m[4]) + Math.abs(m.m[8])) / 2,
    (Math.abs(m.m[1]) + Math.abs(m.m[5]) + Math.abs(m.m[9])) / 2,
    (Math.abs(m.m[2]) + Math.abs(m.m[6]) + Math.abs(m.m[10])) / 2);
}

/** Half the world size of a lone collision primitive on the ghost, else zero. */
export function colliderHalf(scaling, quat) {
  return unitHalf(Matrix.Compose(
    Vector3.FromArray(scaling), quat || Quaternion.Identity(), Vector3.Zero()));
}

/**
 * How far to shift the ghost so a collision primitive lands *corner* first.
 *
 * A kit module is modelled from its origin, so dropping it with the origin on
 * the build plane leaves it resting on the floor and filling whole grid cells.
 * A collision primitive is a unit shape centred on its origin, because Havok
 * wants a centre and not a corner - so the same drop buries half of it under
 * the plane and puts its faces through the middle of a cell. Shifting by the
 * half size puts the corner where the origin was snapped to.
 *
 * Derived from the live transforms, so it stays right after the wheel has
 * scaled the ghost or `R` has turned it. Only a lone primitive gets this: for
 * a set, the offsets are measured from an anchor whose own half size is not
 * the group's, and the two references would fight and drift.
 */
function colliderLift() {
  if (ghost?.items.length !== 1 || !ghost.items[0].collider) return Vector3.Zero();
  const it = ghost.items[0];
  return unitHalf(Matrix.Compose(Vector3.FromArray(it.scaling), it.quat, Vector3.Zero())
    .multiply(Matrix.Compose(Vector3.FromArray(ghost.scaling), ghost.quat, Vector3.Zero())));
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
    ghost.baseY = null;              // V takes over the height from here
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
  const planeY = Number.isFinite(ghost.baseY) ? ghost.baseY : state.gridY;
  const p = cursorOnPlane(planeY + off.y);
  if (!p) return;
  const s = state.snap.pos || 0;
  const snap = (v) => (s ? Math.round(v / s) * s : v);
  const lift = colliderLift();

  // A grab moves things by how far the cursor has travelled since you pressed
  // the key, so they start where they already were and keep whatever sub-grid
  // offset they were placed with. A palette ghost has no "where it already was"
  // and simply sits on the cursor.
  if (ghost.anchor?.cursor) {
    const a = ghost.anchor;
    const to = a.base.add(constrainMove(p.subtract(a.cursor), snap, ghost.basis));
    // In local space the element's own axes may lean, so a move along one of
    // them is allowed to change height. In world space they never do, and the
    // build plane stays in charge of Y as it does everywhere else.
    ghost.root.position.set(to.x, ghost.basis ? to.y : planeY + lift.y, to.z);
    return;
  }

  // A single-axis mode leaves the other coordinate wherever the ghost already
  // is, so it slides along one line from where you put it. Always the world's
  // axes: a module still on the palette has no place of its own yet for a local
  // one to be measured from.
  const x = state.dragAxis === "z" ? ghost.root.position.x : snap(p.x - off.x);
  const z = state.dragAxis === "x" ? ghost.root.position.z : snap(p.z - off.z);
  ghost.root.position.set(x + lift.x, planeY + lift.y, z + lift.z);
}

function setCursorHidden(hidden) {
  const canvas = state.engine?.getRenderingCanvas();
  if (canvas) canvas.style.cursor = hidden ? "none" : "";
}

export function cancelGhost() {
  ghostToken++;
  setCursorHidden(false);
  if (!ghost) return;
  // A cancelled grab must put back exactly what it picked up.
  if (ghost.mode === "move") {
    for (const it of ghost.items) {
      const e = it.sourceId && entryOf(it.sourceId);
      if (e) e.node.setEnabled(true);
    }
    applyVisibility();          // isolation and Shift+H get the last word
  }
  disposeGhost(ghost);
  ghost = null;
  emit("current");
}

/**
 * Commit the ghost: create placements where it stands, or land the originals
 * there if it was a grab.
 */
export async function dropGhost() {
  if (!ghost) return null;
  const g = ghost;
  const quat = g.quat.clone();
  const scaling = g.scaling.slice();

  // Read every item's landing spot *before* touching anything, and compose the
  // world transform rather than decomposing the matrix - see the note in
  // buildGhost about mirrored elements.
  const landed = g.items.map((it) => {
    it.node.computeWorldMatrix(true);
    const pos = it.node.getAbsolutePosition().clone();
    // Babylon's a.multiply(b) applies b first, which is what parenting does:
    // the item's own turn, then the group's.
    const rot = g.quat.multiply(it.quat);
    const scl = new Vector3(
      g.scaling[0] * it.scaling[0],
      g.scaling[1] * it.scaling[1],
      g.scaling[2] * it.scaling[2]);
    return { module: it.module, collider: it.collider, sourceId: it.sourceId, pos, rot, scl };
  });

  let made = null;
  if (g.mode === "move") {
    pushUndo();
    for (const l of landed) {
      const e = l.sourceId && entryOf(l.sourceId);
      if (!e) continue;
      e.node.position.copyFrom(l.pos);
      e.node.rotationQuaternion = l.rot.clone();
      e.node.scaling.copyFrom(l.scl);
      // a sphere has one radius and a capsule one, so an ellipsoid arriving
      // from a free-scaled ghost is snapped back to something Havok can hold
      if (e.type === "collider") hooks.reconcileCollider(e);
      e.node.setEnabled(true);
      made = e;
    }
    applyVisibility();
    cancelGhost();                       // nothing left to carry
    emit("transform");
    emit("placements");
    return made;
  }

  // The manifest stores Euler triples, so the orientation is decomposed only
  // here, once, at the boundary - never accumulated in that form.
  const ids = [];
  for (const l of landed) {
    const node = new TransformNode("TMP", state.scene);
    node.rotationQuaternion = l.rot.clone();
    const euler = eulerOf(node);
    node.dispose();
    const p = l.collider
      ? hooks.addCollider(l.collider, {
        position: l.pos.asArray(), rotation: euler, scale: l.scl.asArray(),
        silent: ids.length > 0,
      })
      : await placeAt(l.module, l.pos, {
        rotation: euler, scale: l.scl.asArray(), silent: ids.length > 0,
        // On the collision bench a dropped module is a stand-in, not ship
        // geometry - and it arrives carrying whatever hull it already has.
        stage: state.collisionMode,
      });
    if (p && state.collisionMode && !l.collider) hooks.attachModuleShapes(p);
    if (p) { ids.push(p.id); made = p; }
  }
  if (ids.length > 1) select(ids);
  // A duplicate is done once dropped; a palette module stays armed so a run of
  // tiles is just repeated clicks. Not on the collision bench: a module goes
  // there once, to have a hull fitted to it, and staging the same one twice is
  // refused anyway - so staying armed only ever produced a second stand-in
  // nobody asked for. Collision primitives still repeat, because a hull really
  // is a run of boxes.
  const oneShot = state.collisionMode && landed.some((l) => !l.collider);
  if (g.mode === "copy" || oneShot) {
    cancelGhost();
    if (oneShot) hooks.clearBrush();
  } else if (ghost) {
    ghost.quat = quat;
    ghost.scaling = scaling;
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
    // The frame is taken once, from the element under the cursor rather than
    // the first of the selection: it is the one you grabbed. Taking it once
    // also means turning a piece mid-drag cannot make its own axes run away
    // from underneath the gesture.
    refNode: entryOf(id)?.node || null,
    basis: axisBasis(entryOf(id)?.node),
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

/**
 * Whose axes the move and rotation axes mean: the world's, or the element's own.
 *
 * Local is what a modular kit wants half the time - every second wall is turned
 * 90 degrees, and "slide it along its length" is world Z on one and world X on
 * the next. It is a *space*, not a fifth axis, which is why it is its own combo
 * rather than four more entries in the axis one.
 *
 * Safe to change mid-gesture, like the axis: a drag re-anchors and re-takes its
 * frame, so the element carries on from where it is instead of jumping.
 */
export const AXIS_SPACES = ["world", "local"];

export function setAxisSpace(space) {
  if (!AXIS_SPACES.includes(space)) return state.axisSpace;
  state.axisSpace = space;
  if (drag) { rebaseDrag(); drag.basis = axisBasis(drag.refNode); }
  if (ghost) {
    if (ghost.anchor) {
      // Re-anchoring on the cursor's *current* position is what keeps the carry
      // still: without it the travel so far would be re-measured in the new
      // frame and the element would jump to wherever that lands.
      ghost.anchor.base = ghost.root.position.clone();
      ghost.anchor.cursor = cursorOnPlane(state.gridY + centreOffset().y);
      ghost.basis = axisBasis(ghost.anchorNode);
    }
    rebaseGhostVertical();
    moveGhostToCursor();
  }
  emit("modes");
  return state.axisSpace;
}

export function toggleAxisSpace() {
  return setAxisSpace(state.axisSpace === "local" ? "world" : "local");
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
    applyDelta(now.subtract(drag.fromV), snap);
    return;
  }

  const now = cursorOnPlane(drag.planeY);
  if (drag.mode === "plane") {
    if (!now) return;
    if (!drag.from) { drag.from = now; return; }
    applyDelta(now.subtract(drag.from), snap);
    return;
  }

  const dpx = state.scene.pointerX - drag.fromPx.x;
  const dpy = state.scene.pointerY - drag.fromPx.y;
  // screen up (negative dpy) pushes the element away, which is the direction
  // the plane mapping inverts when the plane sits above the camera
  const move = drag.right.scale(dpx * drag.unitsPerPx)
    .add(drag.fwd.scale(-dpy * drag.unitsPerPx));
  applyDelta(move, snap);
}

/**
 * Move everything in the drag by however much of this world-space travel the
 * axis and space settings let through.
 *
 * The whole gesture is re-derived from the origins each frame rather than
 * accumulated, so a snapped drag never drifts and cancelling is exact.
 */
function applyDelta(raw, snap) {
  const d = constrainMove(raw, snap, drag.basis);
  for (let i = 0; i < drag.entries.length; i++) {
    drag.entries[i].node.position.copyFrom(drag.origins[i].add(d));
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
  mesh.edgesColor = color;
  outlined.set(mesh.uniqueId, mesh);
  sizeEdges(mesh);
}

/**
 * Keep an outline the same thickness on screen, wherever the camera is.
 *
 * `edgesWidth` is not a pixel width. The line shader offsets the vertex in
 * *clip* space, before the perspective divide, and the renderer hands it
 * `edgesWidth / 50` — so what you see is
 *
 *     pixels = edgesWidth * renderHeight / (100 * viewDepth)
 *
 * which doubles every time you halve your distance. A fixed 5 read as a fine
 * line across a room and as a slab of colour with your nose against a crate,
 * hiding the very thing it was drawn to point at. Turning that around gives
 * the width to ask for, and the outline then reads the same at any zoom.
 *
 * The depth is the *view* depth, not the distance: it is what the shader
 * divides by, and using the plain distance would fatten the outline on
 * anything off to the side of the screen.
 */
function sizeEdges(mesh) {
  const scene = state.scene;
  const cam = scene?.activeCamera;
  if (!cam || mesh.isDisposed()) return;
  const c = mesh.getBoundingInfo().boundingSphere.centerWorld;
  const m = scene.getViewMatrix().m;
  const depth = Math.max(0.05, c.x * m[2] + c.y * m[6] + c.z * m[10] + m[14]);
  mesh.edgesWidth = (100 * EDGE_PX * depth) / scene.getEngine().getRenderHeight();
}

/** Re-size every outline for where the camera is now. Cheap: it is the selection. */
export function resizeOutlines() {
  for (const mesh of outlined.values()) {
    if (mesh.isDisposed()) continue;
    sizeEdges(mesh);
  }
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
 * What an edit acts on: the ghost being placed, otherwise the selection.
 *
 * Hover used to outrank the selection, on the reasoning that pointing at
 * something is a more immediate statement of intent. In practice it made every
 * edit conditional on where the mouse happened to be resting - turn a wall,
 * drift the cursor a pixel onto its neighbour, press `R` again and the
 * neighbour turns. You end up parking the pointer over empty space before
 * touching the keyboard, which is no way to work.
 *
 * `Del` is the one exception, and it keeps its own hover rule: "get rid of that
 * one" is a complete thought on its own, and it needs no selection to survive
 * afterwards.
 */
export function currentElement() {
  if (ghost) return { kind: "ghost", ids: [], module: ghost.module || ghost.collider };
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

/** Placement entries a wheel or key edit should touch: the selection. */
function wheelTargets() {
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

  // Shift turns, Ctrl resizes. The two edits a wheel can do, on the two
  // modifiers, so the bare wheel is always the camera - and the letter keys are
  // left free to carry the *settings* those edits use. Alt adds the pivot
  // variant of a turn: the whole selection about one point rather than each
  // element about its own.
  //
  // preventDefault above is what makes Ctrl+wheel usable at all: it is the
  // browser's zoom, and the listener is registered non-passive on the viewport
  // in capture, before Babylon's own handler sees it.
  const editable = !!ghost || wheelTargets().length > 0;
  const shift = ev.shiftKey && !ev.ctrlKey && !ev.metaKey;
  const ctrl = (ev.ctrlKey || ev.metaKey) && !ev.shiftKey;
  if (editable && shift) {
    rotateCurrent(dir, ev.altKey);
    return;
  }
  if (editable && ctrl) {
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

/**
 * Turn the current element(s) by one step about the chosen axis.
 *
 * Whose axis that is comes from the same World/Local setting as moving: in
 * world space a turn goes about the world's Y, in local space about the
 * element's own - which on anything already turned is a different axis, and
 * the one you mean when you say "tilt this panel back a bit".
 *
 * The axis is taken per element, matching the origin: each turns about its own
 * origin, so each turns about its own axis too, and a row of props tilts the
 * same way relative to each piece rather than fanning out. A pivot turn is the
 * exception - one axis has to serve the whole group, so it comes from the
 * element the gesture is aimed at.
 */
export function rotateCurrent(dir, aboutPivot = false) {
  const step = (state.snap.rot || 90) * dir;
  // the axis name is a cycle label, not a component index - map it or "y"
  // would turn about X
  const axis = "xyz".indexOf(state.rotAxis);
  if (axis < 0) return;
  const rad = step * Math.PI / 180;
  const worldUnit = AXIS_UNITS[axis];
  const unitFor = (node) => axisBasis(node)?.[state.rotAxis] || worldUnit;

  if (ghost) {
    // The ghost root already wears the turn and the mirroring, so its own axes
    // are read from it the same way an element's are.
    const unit = axisBasis(ghost.root)?.[state.rotAxis] || worldUnit;
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
    const unit = unitFor(targets[0].node);
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
    for (const e of targets) spinNode(e.node, unitFor(e.node), rad);
  }
  emit("transform");
  emit("current");
}

export function scaleCurrent(dir) {
  const step = (state.snap.scale || 0.1) * dir;
  // Work on the magnitude and keep the sign, so a mirrored (negative) element
  // can still be resized instead of being snapped back to +0.05 on the first
  // notch.
  //
  // The floor is the smaller of 5 cm and one step, so the wheel can never take
  // something down to nothing but the `free` step can still reach the sizes it
  // exists for: a fixed 5 cm floor sat at exactly five times a 0.01 step, and a
  // collision shell fitted to the kit's 7.5 mm walls could not be nudged at all.
  const floor = Math.min(0.05, Math.abs(step));
  const bump = (v) => {
    const sign = v < 0 ? -1 : 1;
    return sign * Math.max(floor, Math.abs(v) + step);
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
  for (const e of targets) {
    e.node.scaling.set(...apply(e.node.scaling.asArray()));
    // A sphere has one radius and a capsule one too: pull any shape Havok
    // cannot build back onto something it can, as it is being made.
    if (e.type === "collider") hooks.reconcileCollider(e);
  }
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

