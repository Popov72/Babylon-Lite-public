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

import { state, emit, pushUndo, hooks, applyVisibility, select } from "./editor.js";

const { MeshBuilder, StandardMaterial, Color3, Vector3, Quaternion, TransformNode, Matrix } = BABYLON;

/** Smallest extent a collider may have: enough to exist, small enough to hide. */
const MIN_EXTENT = 1e-4;

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
  const mod = opts.module !== undefined ? (opts.module || null) : (state.editModule || null);
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
    // A primitive belongs to a room *or* to a kit module, never both.
    //
    // A module's primitives are authored once, in the module's own local space,
    // and every placement of it inherits them - which is the only sane way to
    // give a barrier or a pod a decent hull, since its AABB is a poor fit and
    // it may be placed twenty times. A room's primitives are world space and
    // belong to that room alone.
    //
    // Undefined means "whatever the editor is doing", so dropping a shape while
    // editing a module joins that module. A restore passes it explicitly, so a
    // saved layout can never be re-homed by the mode that happens to be open.
    module: mod,
    chunk: mod ? null : (opts.chunk || state.activeChunk),
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
  return [...state.colliders.values()].map((c) => ({
    id: c.id,
    kind: c.kind,
    ...(c.module ? { module: c.module } : { chunk: c.chunk }),
    position: round(c.node.position.asArray()),
    rotation: round(eulerDeg(c.node)),
    scale: round(c.node.scaling.asArray()),
    generated: !!c.generated,
  }));
}

/** Every primitive authored on `moduleId`, in the module's own local space. */
export function moduleColliders(moduleId) {
  return [...state.colliders.values()].filter((c) => c.module === moduleId);
}

/** Module ids that carry authored collision, so the fitter can skip them. */
export function modulesWithCollision() {
  const out = new Set();
  for (const c of state.colliders.values()) if (c.module) out.add(c.module);
  return out;
}

export function deserializeColliders(list) {
  for (const id of [...state.colliders.keys()]) removeCollider(id, true);
  for (const c of list || []) {
    if (!COLLIDER_KINDS.includes(c.kind)) continue;
    addCollider(c.kind, Vector3.FromArray(c.position),
      { ...c, module: c.module || null, silent: true });
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
function obbForPlacement(entry, bounds, shell) {
  const size = bounds.max.subtract(bounds.min);
  const centre = bounds.min.add(bounds.max).scale(0.5);
  const scale = entry.node.scaling;
  const quat = entry.node.rotationQuaternion || Quaternion.Identity();
  // the module-local centre, scaled and turned the way the placement is
  const offset = Vector3.TransformCoordinates(
    new Vector3(centre.x * scale.x, centre.y * scale.y, centre.z * scale.z),
    Matrix.Compose(Vector3.One(), quat, Vector3.Zero()));
  // The kit models its floors and ceilings as single planes with no depth at
  // all, so their AABB is degenerate on one axis. Give it the shell thickness
  // instead, centred on the plane, and a room comes out the same thickness all
  // the way round rather than with metre thick slabs top and bottom.
  const thick = (v) => (Math.abs(v) < 1e-6 ? shell : Math.abs(v));
  return {
    centre: entry.node.position.add(offset),
    quat: quat.clone(),
    half: new Vector3(
      thick(size.x * scale.x) / 2,
      thick(size.y * scale.y) / 2,
      thick(size.z * scale.z) / 2),
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
  const shell = state.config.shellThickness;
  const authored = modulesWithCollision();
  for (const e of state.placements.values()) {
    if (e.chunk !== chunkId) continue;
    if (NO_COLLIDER.has(categoryOf(e.module))) { skipped++; continue; }
    // A module that carries its own collision is covered everywhere, always -
    // the manifest instances those shapes onto every placement of it. Fitting a
    // box here as well would give it collision twice, and editing the module
    // would silently stop matching the room until you pressed the button again.
    if (authored.has(e.module)) { inherited++; continue; }
    const bounds = await boundsOf(e.module);
    if (!bounds) continue;

    let boxes = [obbForPlacement(e, bounds, shell)];
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
        chunk: chunkId, module: null, rotation: euler,
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

// ------------------------------------------------- editing a module's shapes

/**
 * A stand-in of the module being edited, so you can see what you are fitting.
 *
 * Deliberately *not* a placement: it is never in state.placements, so it cannot
 * reach the layout, the manifest or the .glb, and it cannot be selected, moved
 * or deleted by accident. It sits at the origin with an identity transform,
 * which is what makes the module's local space and the world agree while you
 * are editing - so every existing tool (the ghost, dragging, scaling, the axis
 * gizmo, the inspector) works on module primitives with no changes at all.
 */
let moduleRef = null;

async function showModuleRef(moduleId, instantiate) {
  hideModuleRef();
  const root = await instantiate(moduleId, "__MODULE_REF");
  root.position.set(0, 0, 0);
  root.rotationQuaternion = Quaternion.Identity();
  root.scaling.set(1, 1, 1);
  for (const m of root.getChildMeshes()) {
    m.isPickable = false;         // the shapes are what you are here to click
    m.metadata = null;            // and it must never resolve as a placement
  }
  moduleRef = root;
  return root;
}

function hideModuleRef() {
  if (!moduleRef) return;
  for (const m of moduleRef.getChildMeshes()) m.dispose();
  moduleRef.dispose();
  moduleRef = null;
}

/** The stand-in's node, so the camera can frame it. */
export function moduleRefNode() { return moduleRef; }

/**
 * Edit the collision of one kit module.
 *
 * The ship is not touched: it is hidden, and put back untouched on the way out.
 * Nothing here is undoable in itself - entering and leaving a mode is not an
 * edit - but everything you do *inside* it is, because module primitives are
 * ordinary colliders that happen to carry a module instead of a chunk.
 */
export async function enterModuleCollision(moduleId, instantiate) {
  if (!moduleId) return false;
  state.editModule = moduleId;
  // through select(), not by clearing the array, or the inspector keeps showing
  // an element that is no longer on screen and no longer acted on
  select([]);
  await showModuleRef(moduleId, instantiate);
  applyVisibility();
  emit("editModule");
  emit("colliders");
  return true;
}

export function exitModuleCollision() {
  if (!state.editModule) return false;
  state.editModule = null;
  select([]);
  hideModuleRef();
  applyVisibility();
  emit("editModule");
  emit("colliders");
  return true;
}

/**
 * Fit a starting box to the module being edited, so there is something to
 * adjust rather than a blank stage. Uses the same AABB the room fitter would,
 * which is exactly the shape authoring a module is meant to improve on.
 */
export async function fitModuleCollision(moduleId, boundsOf) {
  const bounds = await boundsOf(moduleId);
  if (!bounds) return null;
  pushUndo();
  for (const c of moduleColliders(moduleId)) removeCollider(c.id, true);
  const size = bounds.max.subtract(bounds.min);
  const centre = bounds.min.add(bounds.max).scale(0.5);
  const shell = state.config.shellThickness;
  const thick = (v) => (Math.abs(v) < 1e-6 ? shell : Math.abs(v));
  const made = addCollider("box", centre, {
    module: moduleId, silent: true,
    scale: [thick(size.x), thick(size.y), thick(size.z)],
  });
  applyVisibility();
  emit("colliders");
  return made;
}

hooks.serializeColliders = serializeColliders;
hooks.deserializeColliders = deserializeColliders;
hooks.removeCollider = (id) => removeCollider(id, true);
hooks.reconcileCollider = reconcileCollider;
// Used by the ghost, which has to draw a collision primitive without having a
// kit prototype to clone, and to land one when the carry is dropped.
hooks.buildColliderMesh = (kind, id) => buildMesh(kind, id, state.scene);
hooks.addCollider = (kind, opts) =>
  addCollider(kind, Vector3.FromArray(opts.position), opts);
