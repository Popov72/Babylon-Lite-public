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
  placeAt, removePlacement, worldBounds, shipPlacements, resetStageHistory,
  serializeView, applyView,
} from "./editor.js";

const {
  MeshBuilder, StandardMaterial, Color3, Vector3, Quaternion, TransformNode, Matrix,
  Mesh, CreateCapsuleVertexData,
} = BABYLON;

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

/**
 * What a primitive is when you first arm it.
 *
 * A capsule is the only one that needs saying: at a scale of 1 it is one metre
 * wide and one metre tall, which is a capsule whose caps meet in the middle -
 * a sphere. Arming it as a 1 x 2 m pill makes the brush look like the thing it
 * places.
 */
export const COLLIDER_DEFAULT_SCALE = {
  box: [1, 1, 1], sphere: [1, 1, 1], capsule: [1, 2, 1], cylinder: [1, 1, 1],
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
 * The capsule is the exception and gets built by shapeCapsule() below: it is
 * the one kind whose *proportions* change its geometry rather than just its
 * size.
 */
function buildMesh(kind, id, scene) {
  switch (kind) {
    case "sphere":
      return MeshBuilder.CreateSphere(`${id}_shape`, { diameter: 1, segments: 12 }, scene);
    case "capsule": {
      const m = new Mesh(`${id}_shape`, scene);
      shapeCapsule(m, 2);              // a pill until its owner's scale says otherwise
      return m;
    }
    case "cylinder":
      return MeshBuilder.CreateCylinder(`${id}_shape`,
        { height: 1, diameter: 1, tessellation: 16 }, scene);
    default:
      return MeshBuilder.CreateBox(`${id}_shape`, { size: 1 }, scene);
  }
}

// Every capsule currently drawn, and the proportion each was last built at.
const capsules = new Map();
let capsuleWatcher = null;

/**
 * Draw a capsule as a capsule: a tube closed by two hemispheres.
 *
 * A box, a sphere and a cylinder are each one unit mesh under a scale. A
 * capsule is not, because its caps are hemispheres of the *tube's* radius:
 * stretch the mesh in Y and they stretch with it into an ellipsoid. The old
 * unit mesh was worse still - `height: 1, radius: 0.5`, and Babylon's capsule
 * height includes the caps, so `height - 2 x radius` left no tube at all. Every
 * capsule in the editor was a sphere pulled into a lozenge, while Havok
 * collided with a proper pill.
 *
 * What keeps it to a single mesh: build the geometry at the right *ratio* -
 * radius 0.5, total height h/d - and counter-scale Y by d/h. The mesh's own
 * world scale then comes out uniform (d in every axis), so the caps are true
 * hemispheres and the normals stay correct, and the geometry only has to be
 * rebuilt when the proportions change, not when the shape is merely resized.
 */
function shapeCapsule(mesh, ratio) {
  CreateCapsuleVertexData({
    height: ratio, radius: 0.5, tessellation: 12, subdivisions: 1, capSubdivisions: 6,
  }).applyToMesh(mesh);
  mesh.scaling.set(1, 1 / ratio, 1);
  capsules.set(mesh, ratio);
  watchCapsules(mesh.getScene());
}

/**
 * How long the capsule is in units of its own width, read off what its owner
 * actually resolves to rather than its local scale - so a shape under a ghost,
 * which carries its size on the ghost's root, comes out the same as one under a
 * collider root.
 *
 * Floored at 1: at h == d the caps meet and the capsule *is* a sphere. Shorter
 * than that is not a capsule, and Havok agrees - its capsule is a segment plus
 * a radius, and there is no segment of negative length.
 */
function capsuleRatio(mesh) {
  const p = mesh.parent;
  if (!p) return 1;
  const w = p.computeWorldMatrix(true).m;
  // The mean of X and Z, not X alone: a capsule has one radius, and an owner
  // that is wider one way than the other has no capsule to speak of. Averaging
  // is what constrainScale does with the same problem.
  const d = (Math.hypot(w[0], w[1], w[2]) + Math.hypot(w[8], w[9], w[10])) / 2;
  const h = Math.hypot(w[4], w[5], w[6]);
  return d > 1e-9 ? Math.max(h / d, 1) : 1;
}

/**
 * One observer keeps every capsule in shape.
 *
 * A collider's scale changes from the gizmo, the inspector, the wheel, a
 * carried duplicate, a fit, an undo and a load, and a capsule has to be
 * redrawn after any of them. Watching the result catches all seven, and
 * whatever is added next, for the cost of comparing a number a frame.
 */
function watchCapsules(scene) {
  if (capsuleWatcher || !scene) return;
  capsuleWatcher = scene.onBeforeRenderObservable.add(() => {
    for (const [mesh, ratio] of capsules) {
      if (mesh.isDisposed()) { capsules.delete(mesh); continue; }
      const want = capsuleRatio(mesh);
      if (Math.abs(want - ratio) > 1e-4) shapeCapsule(mesh, want);
    }
  });
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

  // A capsule's geometry depends on how tall it is against how wide, so it can
  // only be drawn once it knows its owner - one frame late would show a sphere.
  if (kind === "capsule") for (const m of [mesh, edges]) shapeCapsule(m, capsuleRatio(m));

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
      // A capsule's height counts its caps, the same way a box's side counts
      // the whole box - so it cannot be shorter than it is wide. At h == d the
      // hemispheres meet and it is exactly a sphere; below that there is no
      // shape left, and Havok says the same, its capsule being a segment plus
      // a radius. A cylinder has flat ends and no such floor.
      return [r, kind === "capsule" ? Math.max(s[1], r) : s[1], r];
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
 * Havok is actually given - which is the point of constraining the scale. A
 * capsule's height is the whole pill, caps included, so it reads like a box's
 * side rather than like Havok's inner segment.
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
  for (const t of previewRoot.getChildTransformNodes()) t.dispose();
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
      // Through the same rule the authoring path uses. A placement may be
      // scaled unevenly - this ship has doors at [0.65, 0.85, 1] - and a
      // capsule or sphere composed with that is an ellipsoid, which Havok has
      // no shape for. Drawing the ellipsoid would promise something the runtime
      // cannot deliver, so the preview shows the shape Havok actually gets.
      const size = constrainScale(s.kind, [scl.x, scl.y, scl.z]);
      for (const [material, name] of [[mat, "fill"], [wire, "edge"]]) {
        // The size goes on a holder and the shape hangs off it, exactly as a
        // real collider is built. Putting it on the mesh looked equivalent and
        // was not: a capsule counter-scales its own Y so its caps stay round,
        // and writing over that left the preview drawing spheres - the ship
        // disagreeing with the bench about the same hull.
        const holder = new TransformNode(`PREVIEW_${n}_${name}`, scene);
        holder.parent = previewRoot;
        holder.position.copyFrom(pos);
        holder.rotationQuaternion = rot.clone();
        holder.scaling.set(...size);
        const mesh = buildMesh(s.kind, `PREVIEW_${n}_${name}`, scene);
        mesh.parent = holder;
        mesh.material = material;
        mesh.isPickable = false;
        if (s.kind === "capsule") shapeCapsule(mesh, capsuleRatio(mesh));
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
  watchShipPlacements();
  if (previewPending) return;
  previewPending = true;
  Promise.resolve().then(() => { previewPending = false; buildPreview(); });
}

/**
 * Keep the preview under the elements it belongs to.
 *
 * It is drawn from each placement's world matrix, and rebuilt only through
 * applyVisibility() - which a move, a turn or a delete never calls. So the
 * inherited hulls sat where the elements used to be, and outlived the elements
 * entirely. Watching the matrices catches every way an element can move,
 * including the ones that only emit "transform" at the end of a drag.
 *
 * Only placements whose module actually carries collision are looked at, so on
 * a ship where nothing is authored yet this costs one map lookup per element.
 */
let shipWatch = null;
const shipSeen = new Map();

function watchShipPlacements() {
  if (shipWatch || !state.scene) return;
  shipWatch = state.scene.onBeforeRenderObservable.add(() => {
    if (state.collisionMode || !state.moduleCollision.size) return;
    if (state.showLayer === "geometry") return;
    let changed = false;
    const live = new Set();
    for (const p of shipPlacements()) {
      if (!state.moduleCollision.get(p.module)?.length) continue;
      live.add(p.id);
      p.node.computeWorldMatrix(true);
      const now = p.node.getWorldMatrix();
      const before = shipSeen.get(p.id);
      if (!before || !before.equals(now)) { changed = true; shipSeen.set(p.id, now.clone()); }
    }
    for (const id of [...shipSeen.keys()]) {
      if (!live.has(id)) { shipSeen.delete(id); changed = true; }
    }
    if (changed) refreshCollisionPreview();
  });
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
  attachModuleShapes(entry);
  harvestStage();            // the new shapes need to know their host
  applyVisibility();
  emit("placements");
  emit("colliders");
  return { entry, added: true };
}

/**
 * Lay a module's stored shapes onto a stand-in that has just been placed.
 *
 * Shared by staging from the palette and by dropping a stand-in from the ghost,
 * so a module brings its hull with it however it arrives on the bench.
 */
export function attachModuleShapes(entry) {
  if (!entry?.stage) return 0;
  entry.node.computeWorldMatrix(true);
  let n = 0;
  for (const s of state.moduleCollision.get(entry.module) || []) {
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
    n++;
  }
  return n;
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
  const shapeSeen = new Map();
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

    // A shape moved or scaled on its own is an edit of the hull, and has to
    // reach the record - otherwise "you have unsaved work" never notices it,
    // and neither does the next save.
    let shapesChanged = false;
    const live = new Set();
    for (const c of stageColliders()) {
      live.add(c.id);
      c.node.computeWorldMatrix(true);
      const now = c.node.getWorldMatrix();
      const before = shapeSeen.get(c.id);
      if (!before || !before.equals(now)) shapesChanged = true;
      shapeSeen.set(c.id, now.clone());
    }
    for (const id of [...shapeSeen.keys()]) {
      if (!live.has(id)) { shapeSeen.delete(id); shapesChanged = true; }
    }

    if (moved || shapesChanged) harvestStage();
  });
}

function unwatchStagedElements() {
  if (!stageWatch) return;
  state.scene.onBeforeRenderObservable.remove(stageWatch);
  stageWatch = null;
}

/**
 * The staging area as plain data, for its own undo history.
 *
 * Deliberately not part of serialize(): the bench must never reach the ship.
 * That is also why it needs a snapshot of its own - a ship snapshot restores as
 * "no bench at all", which is how Ctrl+Z used to wipe it.
 */
export function serializeStage() {
  return {
    elements: stagedElements().map((e) => ({
      id: e.id,
      module: e.module,
      position: round(e.node.position.asArray()),
      rotation: round(eulerDeg(e.node)),
      scale: round(e.node.scaling.asArray()),
    })),
    shapes: stageColliders().map((c) => ({
      kind: c.kind,
      position: round(c.node.position.asArray()),
      rotation: round(eulerDeg(c.node)),
      scale: round(c.node.scaling.asArray()),
    })),
  };
}

/** Put the staging area back exactly as a snapshot found it. */
export async function restoreStage(data, instantiate) {
  const make = instantiate || stageInstantiate;
  for (const c of stageColliders()) removeCollider(c.id, true);
  for (const e of stagedElements()) removePlacement(e.id);
  state.selection = [];

  for (const e of data?.elements || []) {
    if (!e?.module) continue;
    await placeAt(e.module, Vector3.FromArray(e.position), {
      stage: true, silent: true, id: e.id,
      rotation: e.rotation, scale: e.scale,
    });
  }
  for (const s of data?.shapes || []) {
    if (!COLLIDER_KINDS.includes(s.kind)) continue;
    addCollider(s.kind, Vector3.FromArray(s.position), {
      stage: true, silent: true, rotation: s.rotation, scale: s.scale,
    });
  }
  harvestStage();
  applyVisibility();
  emit("placements");
  emit("colliders");
  emit("selection");
}

/** Remembered on the way in, so a restore can rebuild stand-ins on its own. */
let stageInstantiate = null;

/**
 * Open the staging area, restoring whatever was on it when it was last closed.
 *
 * Coming back to a blank stage after stepping out to look at the ship was the
 * wrong default: the area is a workbench, and a workbench keeps what you left
 * on it. The roster travels in the collision file, so it survives a reload too.
 */
export async function enterCollisionMode(instantiate, boundsOf) {
  if (state.collisionMode) return false;
  // Each side keeps its own viewpoint. Coming back to the ship from the bench
  // pointing at a barrel, or to the bench pointing across the ship, means
  // finding your bearings again on every switch.
  shipView = serializeView();
  stageInstantiate = instantiate;
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
  resetStageHistory();       // the bench's history starts here, not in the ship's
  const viewRestored = !!stageView;
  if (stageView) applyView(stageView);
  applyVisibility();
  emit("placements");
  emit("colliders");
  return { viewRestored };
}

let shipView = null;
let stageView = null;

/**
 * Where you are standing on each side of the switch, whichever side is open.
 *
 * The live camera is *one* camera serving two rooms. Saving from the bench
 * therefore wrote the bench's viewpoint into the ship's `view` and a stale one
 * into `stageView`, so a save made without closing the bench moved the ship's
 * saved viewpoint to wherever the bench happened to be - the same shape of bug
 * in both directions at once.
 *
 * These say which of the two the live camera currently *is*, so both records
 * come out right no matter where the save was made from. The other side's is
 * stashed and cannot have moved: nothing can drive a camera that is not on
 * screen.
 */
export function shipViewpoint() {
  return state.collisionMode ? (shipView && { ...shipView }) : serializeView();
}

/** The bench's own viewpoint, so it rides in the collision file. */
export function stageViewpoint() {
  if (state.collisionMode) return serializeView();
  return stageView ? { ...stageView } : null;
}
export function setStageViewpoint(v) {
  stageView = v && Array.isArray(v.position) && Array.isArray(v.rotation) ? { ...v } : null;
}

/**
 * What is on the bench, read live while it is open.
 *
 * `state.stageLayout` is only written when the bench closes, so it has exactly
 * the same problem as the viewpoints: a save made without closing wrote the
 * roster from the *previous* session and quietly lost everything staged since.
 */
export function stageLayoutNow() {
  if (!state.collisionMode) return state.stageLayout;
  return stagedElements().map((e) => ({
    module: e.module,
    position: round(e.node.position.asArray()),
  }));
}

/**
 * Both sides store the camera as the *camera* holds it - position and rotation,
 * via serializeView/applyView - rather than a position and a target.
 *
 * `getTarget()` on a FreeCamera reports the last point something explicitly
 * aimed it at, and free look never updates that. Capturing a stale target and
 * then aiming at it put the camera back in the right place looking the wrong
 * way, which reads as "the view was not restored" - because it wasn't.
 */

/** Close it, keeping everything that was fitted and where it all stood. */
export function exitCollisionMode() {
  if (!state.collisionMode) return false;
  unwatchStagedElements();
  harvestStage();
  stageView = serializeView();
  // Remember the bench before clearing it, so re-opening finds the same
  // modules in the same places.
  state.stageLayout = stageLayoutNow();
  for (const c of stageColliders()) removeCollider(c.id, true);
  for (const e of stagedElements()) removePlacement(e.id);
  state.collisionMode = false;
  select([]);
  resetStageHistory();       // the bench's history does not outlive the bench
  if (shipView) applyView(shipView);
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

/**
 * Fit a whole hull to one staged element.
 *
 * Where "fit a box" gives you a starting block to carve up by hand, this reads
 * the element's own triangles and lays out as many boxes as the shape asks for.
 * It declines rather than guess: a curved corner is still better drawn by a
 * person, and saying so is more use than a hull that looks plausible and leaks.
 */
export async function fitHullToSelection() {
  if (!state.collisionMode) return { ok: false, error: "open the collision area first" };
  if (state.selection.length !== 1) {
    return { ok: false,
      error: state.selection.length
        ? `select a single element — ${state.selection.length} are selected`
        : "select the element to fit a hull to" };
  }
  const entry = state.placements.get(state.selection[0]);
  if (!entry) {
    return { ok: false, error: "that is a collision shape — select the element itself" };
  }
  if (!entry.stage) return { ok: false, error: "that element is not on the staging area" };

  const tris = localTriangles(entry.node);
  if (!tris.length) return { ok: false, error: `no geometry to read on ${entry.module}` };

  const { fit } = await import("./hullfit.js");
  const result = fit(tris);
  if (!result.boxes.length) return { ok: false, error: `could not fit ${entry.module}` };

  pushUndo();
  const grown = grownBounds(entry.node);
  if (grown) {
    for (const c of stageColliders()) {
      const cb = worldBounds(c.node);
      if (cb && overlapVolume(cb, grown) > 0) removeCollider(c.id, true);
    }
  }
  entry.node.computeWorldMatrix(true);
  const parent = entry.node.getWorldMatrix();
  const made = [];
  for (const b of result.boxes) {
    // the fitter hands back a frame, not angles, so compose the box's own
    // matrix and let Babylon do the conversion it is already trusted for
    const rot = Matrix.FromValues(
      b.basis[0][0], b.basis[0][1], b.basis[0][2], 0,
      b.basis[1][0], b.basis[1][1], b.basis[1][2], 0,
      b.basis[2][0], b.basis[2][1], b.basis[2][2], 0,
      0, 0, 0, 1);
    const world = Matrix.Scaling(2 * b.half[0], 2 * b.half[1], 2 * b.half[2])
      .multiply(rot)
      .multiply(Matrix.Translation(b.centre[0], b.centre[1], b.centre[2]))
      .multiply(parent);
    const scale = new Vector3(), quat = new Quaternion(), pos = new Vector3();
    world.decompose(scale, quat, pos);
    const e = quat.toEulerAngles();
    made.push(addCollider("box", pos, {
      stage: true, silent: true,
      rotation: [e.x, e.y, e.z].map((r) => Math.round((r * 180 / Math.PI) * 1e4) / 1e4),
      // a box is the same box under an axis flip, so a mirrored placement's
      // negative scale is noise here - the rotation already carries the flip
      scale: [shellThick(scale.x), shellThick(scale.y), shellThick(scale.z)],
    }));
  }
  harvestStage();
  applyVisibility();
  emit("colliders");
  return { ok: true, colliders: made, module: entry.module,
    coverage: result.coverage, solid: result.solid,
    how: result.how, confident: result.confident };
}

/** Every triangle under a node, in that node's own space. */
function localTriangles(node) {
  node.computeWorldMatrix(true);
  const inv = node.getWorldMatrix().clone().invert();
  const tris = [];
  for (const m of node.getChildMeshes()) {
    const pos = m.getVerticesData && m.getVerticesData("position");
    const idx = m.getIndices && m.getIndices();
    if (!pos || !idx) continue;
    m.computeWorldMatrix(true);
    const w = m.getWorldMatrix().multiply(inv);
    const at = (i) => {
      const q = Vector3.TransformCoordinates(
        new Vector3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]), w);
      return [q.x, q.y, q.z];
    };
    for (let i = 0; i < idx.length; i += 3) {
      tris.push([at(idx[i]), at(idx[i + 1]), at(idx[i + 2])]);
    }
  }
  return tris;
}

hooks.serializeColliders = serializeColliders;
hooks.deserializeColliders = deserializeColliders;
hooks.removeCollider = (id) => removeCollider(id, true);
hooks.reconcileCollider = reconcileCollider;
// Used by the ghost, which has to draw a collision primitive without having a
// kit prototype to clone. The materials go on here, not at the call site:
// ghostMaterialFor() clones whatever it is given and returns null for nothing,
// so a bare mesh came out wearing the scene's default grey.
hooks.buildColliderMesh = (kind, id) => {
  const scene = state.scene;
  const { colliderMat, colliderMatSel } = materials(scene);
  const fill = buildMesh(kind, id, scene);
  fill.material = colliderMat;
  const wire = buildMesh(kind, `${id}_edge`, scene);
  wire.material = colliderMatSel;
  // The edges are what make a translucent green shape readable - the fill alone
  // is a smudge, and it is the edges that say where it actually stops.
  return [fill, wire];
};
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
// The staging area has a history of its own - see pushUndo in editor.js.
hooks.attachModuleShapes = attachModuleShapes;
hooks.stageViewpoint = stageViewpoint;
hooks.setStageViewpoint = setStageViewpoint;
hooks.shipViewpoint = shipViewpoint;
hooks.stageLayoutNow = stageLayoutNow;
hooks.serializeStage = serializeStage;
hooks.restoreStage = (data) => restoreStage(data);