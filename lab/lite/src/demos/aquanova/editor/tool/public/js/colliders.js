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
    // set by generateForChunk, so regenerating replaces its own output and
    // leaves anything you placed or adjusted by hand alone
    generated: !!opts.generated,
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
    generated: !!c.generated,
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

// ------------------------------------------------------- auto-generation

/**
 * Categories that get no collider.
 *
 * Decals are stickers - grilles, signage, painted panels - lying flat on a
 * surface that already has one, so a box round them would jut into the room by
 * their whole thickness for nothing. Everything else is solid: props included,
 * since the barriers and pods are things you walk into.
 */
const NO_COLLIDER = new Set(["Decals"]);

function categoryOf(moduleId) {
  return String(moduleId).split("/")[0];
}

/**
 * A world-space OBB hugging one placement, from its module's local bounds.
 *
 * Fitted per *module* rather than per room: the kit has 277 of them and a room
 * has dozens of instances, so the bounds are cached and the placement's own
 * transform does the rest - which is what reproduces corners and inclines
 * without any special handling.
 */
function obbForPlacement(entry, bounds) {
  const size = bounds.max.subtract(bounds.min);
  const centre = bounds.min.add(bounds.max).scale(0.5);
  const scale = entry.node.scaling;
  const quat = entry.node.rotationQuaternion || Quaternion.Identity();
  // the module-local centre, scaled and turned the way the placement is
  const offset = Vector3.TransformCoordinates(
    new Vector3(centre.x * scale.x, centre.y * scale.y, centre.z * scale.z),
    Matrix.Compose(Vector3.One(), quat, Vector3.Zero()));
  // The kit models its floors and ceilings as single planes with no depth at
  // all, and its walls only 7.5 mm thick. Both are given at least the shell
  // thickness, so a fitted room is nowhere thinner than you asked for.
  return {
    centre: entry.node.position.add(offset),
    quat: quat.clone(),
    half: new Vector3(
      shellThick(size.x * scale.x) / 2,
      shellThick(size.y * scale.y) / 2,
      shellThick(size.z * scale.z) / 2),
  };
}

/** True when `q` maps every axis onto an axis - a multiple of 90 degrees. */
function isAxisAligned(m) {
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const v = Math.abs(m.getRow(row).asArray()[col]);
      if (v > 1e-3 && v < 1 - 1e-3) return false;
    }
  }
  return true;
}

/**
 * Cut a doorway out of a box.
 *
 * Works in the door's own frame, where the opening is the rectangle
 * x in [-w/2, w/2], y in [0, h] on the plane z = 0. A wall crossing that plane
 * is split into the pieces around the hole - left, right, sill below, lintel
 * above - and anything not reaching the opening is returned untouched.
 *
 * Only attempted when the box's axes line up with the door's. Every rotation in
 * the kit is a multiple of 90 degrees so that is the normal case; a door at an
 * odd angle leaves an honest solid wall to fix by hand, rather than a cutout
 * that is subtly in the wrong place.
 */
function subtractOpening(box, door) {
  const dq = door.node.rotationQuaternion || Quaternion.Identity();
  const toDoor = Matrix.Compose(Vector3.One(), dq, door.node.getAbsolutePosition()).invert();
  const rel = Matrix.Compose(Vector3.One(), dq, Vector3.Zero()).invert()
    .multiply(Matrix.Compose(Vector3.One(), box.quat, Vector3.Zero()));
  if (!isAxisAligned(rel)) return null;

  // the box, as an AABB in door space (exact, because the axes line up)
  const c = Vector3.TransformCoordinates(box.centre, toDoor);
  const h = new Vector3(
    Math.abs(rel.getRow(0).x) * box.half.x + Math.abs(rel.getRow(1).x) * box.half.y
      + Math.abs(rel.getRow(2).x) * box.half.z,
    Math.abs(rel.getRow(0).y) * box.half.x + Math.abs(rel.getRow(1).y) * box.half.y
      + Math.abs(rel.getRow(2).y) * box.half.z,
    Math.abs(rel.getRow(0).z) * box.half.x + Math.abs(rel.getRow(1).z) * box.half.y
      + Math.abs(rel.getRow(2).z) * box.half.z);

  const w = (door.width * Math.abs(door.node.scaling.x)) / 2;
  const top = door.height * Math.abs(door.node.scaling.y);
  const lo = { x: c.x - h.x, y: c.y - h.y, z: c.z - h.z };
  const hi = { x: c.x + h.x, y: c.y + h.y, z: c.z + h.z };

  // Must straddle the door plane, or a wall on the far side of the room that
  // happens to line up in X and Y would be cut for no reason.
  if (hi.z < -1e-3 || lo.z > 1e-3) return null;
  if (hi.x <= -w + 1e-4 || lo.x >= w - 1e-4) return null;   // clear of the hole
  if (hi.y <= 1e-4 || lo.y >= top - 1e-4) return null;

  const pieces = [];
  const push = (x0, x1, y0, y1) => {
    if (x1 - x0 < 1e-3 || y1 - y0 < 1e-3) return;
    pieces.push({
      centre: Vector3.TransformCoordinates(
        new Vector3((x0 + x1) / 2, (y0 + y1) / 2, c.z), toDoor.clone().invert()),
      quat: dq.clone(),
      half: new Vector3((x1 - x0) / 2, (y1 - y0) / 2, h.z),
    });
  };
  const bandX0 = Math.max(lo.x, -w);
  const bandX1 = Math.min(hi.x, w);
  push(lo.x, Math.min(hi.x, -w), lo.y, hi.y);          // left of the opening
  push(Math.max(lo.x, w), hi.x, lo.y, hi.y);           // right of it
  push(bandX0, bandX1, lo.y, Math.min(hi.y, 0));       // sill below
  push(bandX0, bandX1, Math.max(lo.y, top), hi.y);     // lintel above
  return pieces;
}

/**
 * Fit collision boxes to every solid module in a chunk.
 *
 * Regenerating replaces only what was generated before: colliders you placed or
 * adjusted by hand carry `generated: false` and are left alone, so the button
 * is safe to press twice.
 */
export async function generateForChunk(chunkId, boundsOf) {
  pushUndo();
  for (const c of [...state.colliders.values()]) {
    if (c.chunk === chunkId && c.generated) removeCollider(c.id, true);
  }

  const doors = [...state.markers.values()]
    .filter((m) => m.type === "door" && !m.sealed);

  let made = 0, skipped = 0, cut = 0, inherited = 0; const cutModules = [];
  for (const e of state.placements.values()) {
    if (e.stage || e.chunk !== chunkId) continue;
    if (NO_COLLIDER.has(categoryOf(e.module))) { skipped++; continue; }
    // A module that carries its own collision is covered everywhere, always -
    // the manifest instances those shapes onto every placement of it. Fitting a
    // box here as well would give it collision twice, and editing the module
    // would silently stop matching the room until you pressed the button again.
    if (state.moduleCollision.get(e.module)?.length) { inherited++; continue; }
    const bounds = await boundsOf(e.module);
    if (!bounds) continue;

    let boxes = [obbForPlacement(e, bounds)];
    for (const d of doors) {
      const next = [];
      for (const b of boxes) {
        const split = subtractOpening(b, d);
        if (split) { next.push(...split); cut++; cutModules.push(e.module); } else next.push(b);
      }
      boxes = next;
    }
    for (const b of boxes) {
      const node = new TransformNode("TMP_OBB", state.scene);
      node.rotationQuaternion = b.quat;
      const euler = eulerDeg(node);
      node.dispose();
      addCollider("box", b.centre, {
        chunk: chunkId, stage: false, rotation: euler,
        scale: [b.half.x * 2, b.half.y * 2, b.half.z * 2],
        generated: true, silent: true,
      });
      made++;
    }
  }
  applyVisibility();
  emit("colliders");
  return { made, skipped, cut, inherited, cutModules };
}

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
  applyVisibility();
  emit("placements");
  emit("colliders");
  return true;
}

/** Close it, keeping everything that was fitted and where it all stood. */
export function exitCollisionMode() {
  if (!state.collisionMode) return false;
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
hooks.addCollider = (kind, opts) =>
  addCollider(kind, Vector3.FromArray(opts.position),
    // a shape dropped while the staging area is open is a staged shape
    { ...opts, stage: opts.stage !== undefined ? opts.stage : state.collisionMode });
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
