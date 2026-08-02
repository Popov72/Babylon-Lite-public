// Collision primitives.
//
// These are not kit geometry and never reach ship.glb: they are the shapes the
// runtime hands to Havok, carried in the manifest as a kind plus a transform.
//
// Every primitive is authored as a UNIT shape and sized by scaling, so the data
// is nothing but a transform - but only the box can take an arbitrary one.
// Havok's sphere is a radius; its capsule and cylinder are a radius and a pair
// of endpoints. There is no ellipsoid, so a sphere scaled [2,1,1] describes
// something Havok cannot build, and the runtime would have to silently pick an
// axis and resize the collider behind your back. constrainScale() below keeps
// the editor honest instead: what you see is what Havok gets.
//
// Orientation: a box carries a quaternion, a sphere needs none, and a capsule
// or cylinder gets its axis from its two endpoints - which the runtime derives
// from this transform, so any orientation works for all four.

import {
  state, emit, pushUndo, hooks, applyVisibility, select,
  placeAt, removePlacement, worldBounds, shipPlacements,
} from "./editor.js";

const { MeshBuilder, StandardMaterial, Color3, Vector3, Quaternion, TransformNode, Matrix } = BABYLON;

/** Smallest extent a collider may have: enough to exist, small enough to hide. */
const MIN_EXTENT = 1e-4;

/**
 * An extent, never thinner than the configured collision shell.
 *
 * The shell is a *minimum*, not a filler for zero-depth axes. It started as the
 * latter, which made it look broken: a barrel has real depth on all three axes,
 * so nothing you typed ever changed anything. A minimum is both what the name
 * implies and what is actually useful - the kit's walls measure 7.5 mm, and a
 * collider that thin is something a fast-moving body tunnels straight through.
 */
function shellThick(v) {
  return Math.max(Math.abs(v) || 0, state.config.shellThickness);
}

/** The kinds Havok can take directly, in the order the palette lists them. */
export const COLLIDER_KINDS = ["box", "sphere", "capsule", "cylinder"];

export const COLLIDER_LABEL = {
  box: "Box (OBB)",
  sphere: "Sphere",
  capsule: "Capsule",
  cylinder: "Cylinder",
};

/**
 * How each kind may be scaled.
 *
 * "free"    every axis independent            - box
 * "uniform" one number, a radius              - sphere
 * "radial"  X and Z together, Y on its own    - capsule, cylinder
 */
export const SCALE_RULE = {
  box: "free", sphere: "uniform", capsule: "radial", cylinder: "radial",
};

let colliderMat = null;
let colliderMatSel = null;

function materials(scene) {
  if (!colliderMat) {
    colliderMat = new StandardMaterial("COLLIDER_mat", scene);
    colliderMat.emissiveColor = new Color3(0.15, 0.9, 0.35);
    colliderMat.diffuseColor = Color3.Black();
    colliderMat.specularColor = Color3.Black();
    colliderMat.alpha = 0.3;
    colliderMat.backFaceCulling = false;
    colliderMat.disableLighting = true;

    // Wireframe over the fill: a translucent green box inside a dark room reads
    // as a smudge, and the edges are what tell you where it actually stops.
    colliderMatSel = new StandardMaterial("COLLIDER_wire", scene);
    colliderMatSel.emissiveColor = new Color3(0.3, 1.0, 0.45);
    colliderMatSel.diffuseColor = Color3.Black();
    colliderMatSel.specularColor = Color3.Black();
    colliderMatSel.disableLighting = true;
    colliderMatSel.wireframe = true;
  }
  return { colliderMat, colliderMatSel };
}

function nextColliderId() {
  let n = 1;
  while (state.colliders.has(`C${String(n).padStart(4, "0")}`)) n++;
  return `C${String(n).padStart(4, "0")}`;
}

/**
 * Unit geometry, one metre across, so a scale of 1 is a 1 m shape and the
 * numbers in the inspector read as metres.
 *
 * The capsule is built at height 1 *including* its caps, so scaling Y by 3
 * gives a 3 m capsule rather than 3 m plus two hemispheres.
 */
function buildMesh(kind, id, scene) {
  switch (kind) {
    case "sphere":
      return MeshBuilder.CreateSphere(`${id}_shape`, { diameter: 1, segments: 12 }, scene);
    case "capsule":
      return MeshBuilder.CreateCapsule(`${id}_shape`,
        { height: 1, radius: 0.5, tessellation: 12, subdivisions: 1 }, scene);
    case "cylinder":
      return MeshBuilder.CreateCylinder(`${id}_shape`,
        { height: 1, diameter: 1, tessellation: 16 }, scene);
    default:
      return MeshBuilder.CreateBox(`${id}_shape`, { size: 1 }, scene);
  }
}

export function addCollider(kind, position, opts = {}) {
  if (!COLLIDER_KINDS.includes(kind)) return null;
  if (!opts.silent) pushUndo();
  const scene = state.scene;
  const { colliderMat: mat, colliderMatSel: wire } = materials(scene);
  const id = opts.id || nextColliderId();
  const root = new TransformNode(id, scene);
  root.position.copyFrom(position);
  root.rotationQuaternion = opts.rotation
    ? Quaternion.FromEulerAngles(
      opts.rotation[0] * Math.PI / 180,
      opts.rotation[1] * Math.PI / 180,
      opts.rotation[2] * Math.PI / 180)
    : Quaternion.Identity();
  if (opts.scale) root.scaling.set(...constrainScale(kind, opts.scale));

  const mesh = buildMesh(kind, id, scene);
  mesh.parent = root;
  mesh.material = mat;
  mesh.isPickable = true;
  mesh.metadata = { colliderRoot: root };

  const edges = buildMesh(kind, `${id}_edge`, scene);
  edges.parent = root;
  edges.material = wire;
  edges.isPickable = false;

  const data = {
    id,
    type: "collider",          // entryOf() and the inspector branch on this
    kind,
    // A primitive belongs to a room, or to the collision staging area.
    //
    // A staged one is a *working copy* of what some module carries: which
    // module is decided by where it sits (see harvestStage), not by a tag, so
    // that dragging a copy of one hull onto a similar element simply makes it
    // that element's. It is never serialized into the layout - the per-module
    // record is what persists.
    stage: !!opts.stage,
    chunk: opts.stage ? null : (opts.chunk || state.activeChunk),
    node: root,
    mesh,
    edges,
  };
  root.metadata = { collider: data };
  state.colliders.set(id, data);
  applyVisibility();
  if (!opts.silent) { emit("colliders"); }
  return data;
}

export function removeCollider(id, silent = false) {
  const c = state.colliders.get(id);
  if (!c) return;
  c.mesh?.dispose();
  c.edges?.dispose();
  c.node.dispose();
  state.colliders.delete(id);
  if (!silent) emit("colliders");
}

/**
 * Force a scale the collider's kind can actually represent.
 *
 * Called on every write rather than only in the inspector, so nothing - a
 * wheel resize, a carried duplicate, a loaded manifest - can slip an
 * unrepresentable shape past.
 */
export function constrainScale(kind, scale) {
  // A zero extent has no shape at all, but inflating it to 1 m - which is what
  // `|| 1` used to do - silently turned the kit's flat floor planes into metre
  // thick slabs. Clamp to something merely tiny instead, and let the fitter
  // give a flat module a real shell thickness before it ever gets here.
  const s = scale.map((v) => Math.max(MIN_EXTENT, Math.abs(Number(v)) || 0));
  switch (SCALE_RULE[kind]) {
    case "uniform": {
      const r = (s[0] + s[1] + s[2]) / 3;
      return [r, r, r];
    }
    case "radial": {
      const r = (s[0] + s[2]) / 2;
      return [r, s[1], r];
    }
    default:
      return s;
  }
}

/** Re-apply the kind's rule to a collider that has just been scaled. */
export function reconcileCollider(c) {
  if (!c || c.type !== "collider") return false;
  const before = c.node.scaling.asArray();
  const after = constrainScale(c.kind, before);
  if (before.every((v, i) => Math.abs(v - after[i]) < 1e-9)) return false;
  c.node.scaling.set(...after);
  return true;
}

/**
 * What the shape measures, in metres, for the inspector.
 *
 * A box reports its three sides; everything else reports the radius and height
 * Havok is actually given - which is the point of constraining the scale.
 */
export function colliderDims(c) {
  const s = c.node.scaling;
  if (c.kind === "box") return { kind: "box", size: [s.x, s.y, s.z] };
  if (c.kind === "sphere") return { kind: "sphere", radius: s.x / 2 };
  return { kind: c.kind, radius: s.x / 2, height: s.y };
}

export function serializeColliders() {
  // The staging area is a working copy; the per-module record is what persists.
  return [...state.colliders.values()].filter((c) => !c.stage).map((c) => ({
    id: c.id,
    kind: c.kind,
    chunk: c.chunk,
    position: round(c.node.position.asArray()),
    rotation: round(eulerDeg(c.node)),
    scale: round(c.node.scaling.asArray()),
  }));
}

export function deserializeColliders(list) {
  for (const id of [...state.colliders.keys()]) removeCollider(id, true);
  for (const c of list || []) {
    if (!COLLIDER_KINDS.includes(c.kind)) continue;
    addCollider(c.kind, Vector3.FromArray(c.position),
      { ...c, stage: false, silent: true });
  }
  emit("colliders");
}

function round(a) { return a.map((v) => Math.round(v * 1e4) / 1e4); }

function eulerDeg(node) {
  const q = node.rotationQuaternion || Quaternion.FromEulerVector(node.rotation);
  const e = q.toEulerAngles();
  return [e.x, e.y, e.z].map((r) => Math.round((r * 180 / Math.PI) * 1e4) / 1e4);
}

// --------------------------------------------- the collision staging area
//
// A mode, not a property of the selection. It opens empty; you stage whatever
// modules you want to work on, one instance of each, and fit shapes to them.
//
// Staged elements are real placements carrying `stage: true`, which is what
// makes every existing tool work on them unchanged - selection, the axis
// gizmo, hiding, dragging, Ctrl+D, the marquee, the inspector. They are
// filtered out of the manifest and the .glb, the only two places that walk
// every placement, so they can never reach the ship.
//
// Which element a shape belongs to is decided by *where it is*, not by a tag:
// a shape is owned by the staged element whose bounding box, grown by
// ASSOCIATION_MARGIN, it overlaps most. That is what makes copying work - drag
// a duplicate of one barrel's hull onto another and it simply becomes that
// one's. It is only safe because the stage lays elements out with more than
// twice that margin between them, so no two grown boxes can ever touch.

/** How far outside an element's own bounds a shape still counts as its own. */
export const ASSOCIATION_MARGIN = 0.5;

/** Clear ground between staged elements, so their grown boxes cannot meet. */
const STAGE_GAP = ASSOCIATION_MARGIN * 2 + 1;

function stagedElements() {
  return [...state.placements.values()].filter((p) => p.stage);
}

export function stageColliders() {
  return [...state.colliders.values()].filter((c) => c.stage);
}

/** The world AABB of a node, grown by the association margin. */
function grownBounds(node) {
  const b = worldBounds(node);
  if (!b) return null;
  const m = new Vector3(ASSOCIATION_MARGIN, ASSOCIATION_MARGIN, ASSOCIATION_MARGIN);
  return { min: b.min.subtract(m), max: b.max.add(m) };
}

function overlapVolume(a, b) {
  const dx = Math.min(a.max.x, b.max.x) - Math.max(a.min.x, b.min.x);
  const dy = Math.min(a.max.y, b.max.y) - Math.max(a.min.y, b.min.y);
  const dz = Math.min(a.max.z, b.max.z) - Math.max(a.min.z, b.min.z);
  if (dx <= 0 || dy <= 0 || dz <= 0) return 0;
  return dx * dy * dz;
}

/**
 * Read the staging area back into the per-module record.
 *
 * Only modules currently staged are rewritten. Anything else keeps what it had
 * - staging one barrel must not wipe the collision of every other module.
 *
 * Run after every change rather than only on the way out, so the record is
 * always current and leaving, saving or removing an element need no special
 * handling. A shape that belongs to nothing is left where it is and counted, so
 * it can be reported rather than silently dropped.
 */
export function harvestStage() {
  const elements = stagedElements();
  if (!elements.length) return { assigned: 0, orphans: stageColliders().length };

  const boxes = elements.map((e) => {
    e.node.computeWorldMatrix(true);
    return { entry: e, grown: grownBounds(e.node) };
  }).filter((x) => x.grown);

  const claimed = new Map(boxes.map((b) => [b.entry.id, []]));
  let orphans = 0;
  for (const c of stageColliders()) {
    c.node.computeWorldMatrix(true);
    const cb = worldBounds(c.node);
    if (!cb) continue;
    let best = null, bestVol = 0;
    for (const b of boxes) {
      const v = overlapVolume(cb, b.grown);
      if (v > bestVol) { bestVol = v; best = b; }
    }
    // Remembered on the shape as well as counted: moving or turning a staged
    // element has to carry its own shapes with it, and that has to know whose
    // they are without re-deciding ownership half way through the move.
    c.host = best ? best.entry.id : null;
    if (!best) { orphans++; continue; }
    claimed.get(best.entry.id).push({ collider: c, host: best.entry });
  }

  let assigned = 0;
  for (const b of boxes) {
    const mine = claimed.get(b.entry.id);
    // The shapes are stored relative to the element, so a module staged at a
    // different spot next time gets them back in the right place.
    const toLocal = b.entry.node.getWorldMatrix().clone().invert();
    const shapes = mine.map(({ collider }) => {
      const local = Matrix.Compose(
        collider.node.scaling,
        collider.node.rotationQuaternion || Quaternion.Identity(),
        collider.node.position).multiply(toLocal);
      const pos = new Vector3(), rot = new Quaternion(), scl = new Vector3();
      local.decompose(scl, rot, pos);
      const e = rot.toEulerAngles();
      return {
        kind: collider.kind,
        position: round([pos.x, pos.y, pos.z]),
        rotation: round([e.x, e.y, e.z].map((r) => r * 180 / Math.PI)),
        scale: round([scl.x, scl.y, scl.z]),
      };
    });
    assigned += shapes.length;
    if (shapes.length) state.moduleCollision.set(b.entry.module, shapes);
    else state.moduleCollision.delete(b.entry.module);
  }
  return { assigned, orphans };
}

/** How many staged shapes belong to nothing, for the banner to report. */
export function orphanCount() {
  const boxes = stagedElements().map((e) => grownBounds(e.node)).filter(Boolean);
  let n = 0;
  for (const c of stageColliders()) {
    const cb = worldBounds(c.node);
    if (!cb) continue;
    if (!boxes.some((b) => overlapVolume(cb, b) > 0)) n++;
  }
  return n;
}

// ------------------------------------------- inherited collision, on the ship
//
// A module's shapes are stored once and instanced onto every placement of it at
// export time, which means the ship carried collision you could not see: the
// only place it was ever drawn was the staging area. "Collision only" on a ship
// whose collision is all inherited showed an empty room, which reads exactly
// like a broken switch.
//
// So it is drawn here too - as a preview, not as data. These meshes are not in
// state.colliders, are not pickable and are rebuilt from the record, so there
// is nothing to keep in sync and nothing to accidentally edit: to change them
// you open the staging area, which is the one place they are editable.

let previewRoot = null;
let previewPending = false;

function disposePreview() {
  if (!previewRoot) return;
  for (const m of previewRoot.getChildMeshes()) m.dispose();
  previewRoot.dispose();
  previewRoot = null;
}

/** Whether a placement is on screen for reasons other than the layer switch. */
function visibleIgnoringLayer(p) {
  if (p.stage) return false;
  if (state.hidden.get(p.id) === "hidden") return false;
  return !state.isolate || p.chunk === state.activeChunk;
}

function buildPreview() {
  disposePreview();
  if (state.collisionMode) return;              // the bench draws its own
  if (state.showLayer === "geometry") return;   // collision is off screen
  if (!state.moduleCollision.size) return;

  const scene = state.scene;
  if (!scene) return;
  const { colliderMat: mat, colliderMatSel: wire } = materials(scene);
  previewRoot = new TransformNode("__COLLISION_PREVIEW", scene);

  let n = 0;
  for (const p of shipPlacements()) {
    const shapes = state.moduleCollision.get(p.module);
    if (!shapes?.length || !visibleIgnoringLayer(p)) continue;
    p.node.computeWorldMatrix(true);
    const placement = p.node.getWorldMatrix();
    for (const s of shapes) {
      const world = Matrix.Compose(
        Vector3.FromArray(s.scale),
        Quaternion.FromEulerAngles(
          s.rotation[0] * Math.PI / 180,
          s.rotation[1] * Math.PI / 180,
          s.rotation[2] * Math.PI / 180),
        Vector3.FromArray(s.position)).multiply(placement);
      const pos = new Vector3(), rot = new Quaternion(), scl = new Vector3();
      world.decompose(scl, rot, pos);
      for (const [material, name] of [[mat, "fill"], [wire, "edge"]]) {
        const mesh = buildMesh(s.kind, `PREVIEW_${n}_${name}`, scene);
        mesh.parent = previewRoot;
        mesh.position.copyFrom(pos);
        mesh.rotationQuaternion = rot.clone();
        mesh.scaling.copyFrom(scl);
        mesh.material = material;
        mesh.isPickable = false;
      }
      n++;
    }
  }
}

/**
 * Rebuild the preview once, after the current burst of changes.
 *
 * applyVisibility() runs on every single collider added, so fitting a room of
 * eighty boxes would otherwise rebuild the whole ship's preview eighty times.
 */
export function refreshCollisionPreview() {
  if (previewPending) return;
  previewPending = true;
  Promise.resolve().then(() => { previewPending = false; buildPreview(); });
}

/** How many inherited shapes are currently drawn, for tests. */
export function previewCount() {
  return previewRoot ? previewRoot.getChildMeshes().length / 2 : 0;
}

/** Where to park the next staged element, clear of everything already there. */
async function nextStageSpot(moduleId, boundsOf) {
  const bounds = await boundsOf(moduleId);
  const half = bounds ? Math.max(
    Math.abs(bounds.max.x - bounds.min.x),
    Math.abs(bounds.max.z - bounds.min.z)) / 2 : 2;
  let x = null;
  for (const e of stagedElements()) {
    const b = worldBounds(e.node);
    if (b) x = x === null ? b.max.x : Math.max(x, b.max.x);
  }
  // A gap wider than twice the association margin, so no two grown boxes touch
  return new Vector3(x === null ? 0 : x + STAGE_GAP + half, 0, 0);
}

/**
 * Put a module on the stage, with whatever shapes it already carries.
 *
 * One instance of each: a second would give the association rule two equally
 * good answers. Staging one that is already there focuses it instead, which is
 * what you actually wanted when you clicked it.
 */
export async function stageModule(moduleId, instantiate, boundsOf, at = null) {
  const existing = stagedElements().find((e) => e.module === moduleId);
  if (existing) return { entry: existing, added: false };

  const spot = at ? Vector3.FromArray(at) : await nextStageSpot(moduleId, boundsOf);
  const entry = await placeAt(moduleId, spot, { stage: true, silent: true });

  for (const s of state.moduleCollision.get(moduleId) || []) {
    const world = Matrix.Compose(
      Vector3.FromArray(s.scale),
      Quaternion.FromEulerAngles(
        s.rotation[0] * Math.PI / 180,
        s.rotation[1] * Math.PI / 180,
        s.rotation[2] * Math.PI / 180),
      Vector3.FromArray(s.position)).multiply(entry.node.getWorldMatrix());
    const pos = new Vector3(), rot = new Quaternion(), scl = new Vector3();
    world.decompose(scl, rot, pos);
    const e = rot.toEulerAngles();
    addCollider(s.kind, pos, {
      stage: true, silent: true,
      rotation: [e.x, e.y, e.z].map((r) => r * 180 / Math.PI),
      scale: [scl.x, scl.y, scl.z],
    });
  }
  harvestStage();            // the new shapes need to know their host
  applyVisibility();
  emit("placements");
  emit("colliders");
  return { entry, added: true };
}
/**
 * Take a module off the stage, keeping what was fitted to it.
 *
 * The record is read back first, so removing an element is not a way to lose
 * its collision - staging it again brings the shapes straight back.
 */
export function unstageModule(entryOrId) {
  const entry = typeof entryOrId === "string"
    ? state.placements.get(entryOrId) : entryOrId;
  if (!entry?.stage) return false;

  harvestStage();
  const grown = grownBounds(entry.node);
  if (grown) {
    for (const c of stageColliders()) {
      const cb = worldBounds(c.node);
      if (cb && overlapVolume(cb, grown) > 0) removeCollider(c.id, true);
    }
  }
  removePlacement(entry.id);
  state.selection = state.selection.filter((id) => id !== entry.id);
  applyVisibility();
  emit("placements");
  emit("colliders");
  emit("selection");
  return true;
}

/**
 * Shapes follow the element they belong to.
 *
 * Moving or turning a stand-in on the bench is a *view* operation - the hull is
 * authored in the module's own frame, so nothing about it should change. Left
 * to itself the element slid out from under its shapes, which then belonged to
 * nothing, or worse, to whichever neighbour they had drifted into.
 *
 * Done by watching the result rather than by hooking the causes. There are at
 * least six ways an element's transform changes - a drag, an M carry, the
 * inspector, an arrow-key nudge, R, F - and a watcher catches all of them,
 * including any added later.
 */
let stageWatch = null;

function watchStagedElements() {
  unwatchStagedElements();
  const seen = new Map();
  stageWatch = state.scene.onBeforeRenderObservable.add(() => {
    if (!state.collisionMode) return;
    let moved = false;
    for (const e of stagedElements()) {
      e.node.computeWorldMatrix(true);
      const now = e.node.getWorldMatrix();
      const before = seen.get(e.id);
      if (before && !before.equals(now)) {
        const delta = before.clone().invert().multiply(now);
        for (const c of stageColliders()) {
          if (c.host !== e.id) continue;
          c.node.computeWorldMatrix(true);
          const world = c.node.getWorldMatrix().multiply(delta);
          const p = new Vector3(), q = new Quaternion(), s = new Vector3();
          world.decompose(s, q, p);
          c.node.position.copyFrom(p);
          c.node.rotationQuaternion = q;
          c.node.scaling.copyFrom(s);
          c.node.computeWorldMatrix(true);
        }
        moved = true;
      }
      seen.set(e.id, now.clone());
    }
    // Drop anything that has left the bench, so a re-staged module does not
    // inherit a stale matrix and jump.
    for (const id of [...seen.keys()]) {
      if (!state.placements.get(id)?.stage) seen.delete(id);
    }
    if (moved) harvestStage();
  });
}

function unwatchStagedElements() {
  if (!stageWatch) return;
  state.scene.onBeforeRenderObservable.remove(stageWatch);
  stageWatch = null;
}

/**
 * Open the staging area, restoring whatever was on it when it was last closed.
 *
 * Coming back to a blank stage after stepping out to look at the ship was the
 * wrong default: the area is a workbench, and a workbench keeps what you left
 * on it. The roster travels in the collision file, so it survives a reload too.
 */
export async function enterCollisionMode(instantiate, boundsOf) {
  if (state.collisionMode) return false;
  state.collisionMode = true;
  select([]);
  applyVisibility();
  emit("collisionMode");

  for (const s of state.stageLayout) {
    if (!s?.module) continue;
    await stageModule(s.module, instantiate, boundsOf, s.position);
  }
  harvestStage();            // so every shape knows its host before anything moves
  watchStagedElements();
  applyVisibility();
  emit("placements");
  emit("colliders");
  return true;
}

/** Close it, keeping everything that was fitted and where it all stood. */
export function exitCollisionMode() {
  if (!state.collisionMode) return false;
  unwatchStagedElements();
  harvestStage();
  // Remember the bench before clearing it, so re-opening finds the same
  // modules in the same places.
  state.stageLayout = stagedElements().map((e) => ({
    module: e.module,
    position: round(e.node.position.asArray()),
  }));
  for (const c of stageColliders()) removeCollider(c.id, true);
  for (const e of stagedElements()) removePlacement(e.id);
  state.collisionMode = false;
  select([]);
  applyVisibility();
  emit("collisionMode");
  emit("placements");
  emit("colliders");
  return true;
}

/**
 * Fit a box to one staged element.
 *
 * Acts on the selection, and says so plainly when the selection is not one
 * element: fitting "the current module" was ambiguous the moment the stage
 * could hold more than one.
 */
export async function fitBoxToSelection(boundsOf) {
  if (!state.collisionMode) return { ok: false, error: "open the collision area first" };
  if (state.selection.length !== 1) {
    return { ok: false,
      error: state.selection.length
        ? `select a single element — ${state.selection.length} are selected`
        : "select the element to fit a box to" };
  }
  const entry = state.placements.get(state.selection[0]);
  if (!entry) {
    return { ok: false, error: "that is a collision shape — select the element itself" };
  }
  if (!entry.stage) return { ok: false, error: "that element is not on the staging area" };

  const bounds = await boundsOf(entry.module);
  if (!bounds) return { ok: false, error: `no bounds for ${entry.module}` };

  pushUndo();
  const grown = grownBounds(entry.node);
  if (grown) {
    for (const c of stageColliders()) {
      const cb = worldBounds(c.node);
      if (cb && overlapVolume(cb, grown) > 0) removeCollider(c.id, true);
    }
  }
  entry.node.computeWorldMatrix(true);
  const size = bounds.max.subtract(bounds.min);
  const centre = bounds.min.add(bounds.max).scale(0.5);
  const world = Matrix.Compose(Vector3.One(), Quaternion.Identity(), centre)
    .multiply(entry.node.getWorldMatrix());
  const made = addCollider("box",
    new Vector3(world.m[12], world.m[13], world.m[14]), {
      stage: true, silent: true,
      rotation: eulerDeg(entry.node),
      scale: [
        shellThick(size.x * entry.node.scaling.x),
        shellThick(size.y * entry.node.scaling.y),
        shellThick(size.z * entry.node.scaling.z)],
    });
  harvestStage();
  applyVisibility();
  emit("colliders");
  return { ok: true, collider: made, module: entry.module };
}

hooks.serializeColliders = serializeColliders;
hooks.deserializeColliders = deserializeColliders;
hooks.removeCollider = (id) => removeCollider(id, true);
hooks.reconcileCollider = reconcileCollider;
// Used by the ghost, which has to draw a collision primitive without having a
// kit prototype to clone, and to land one when the carry is dropped.
hooks.buildColliderMesh = (kind, id) => buildMesh(kind, id, state.scene);
hooks.addCollider = (kind, opts) => {
  // a shape dropped while the staging area is open is a staged shape
  const c = addCollider(kind, Vector3.FromArray(opts.position),
    { ...opts, stage: opts.stage !== undefined ? opts.stage : state.collisionMode });
  // and needs a host straight away, or the element it was dropped on would
  // move out from under it the first time anything nudged that element
  if (c?.stage) harvestStage();
  return c;
};
// Removing a staged element must keep what was fitted to it, so the delete path
// goes through unstageModule rather than disposing the placement outright.
hooks.unstageModule = (id) => unstageModule(id);
// Any change to a staged shape re-reads the area into the per-module record,
// so leaving, saving or removing an element need no special handling.
hooks.harvestStage = () => { if (state.collisionMode) harvestStage(); };
// Inherited collision is drawn from the record rather than stored as elements,
// so applyVisibility - the one place that decides what is on screen - asks for
// it to be rebuilt whenever anything it depends on has moved.
hooks.refreshCollisionPreview = refreshCollisionPreview;