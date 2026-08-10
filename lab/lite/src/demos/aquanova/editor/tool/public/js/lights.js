// Authored lights.
//
// A light rides a placement: its node is a CHILD of that placement's node, so
// turning or moving the module carries its light with it and the authored
// offset/rotation are read in the module's own local space. That is what makes
// per-module defaults possible at all - "the panel's lamp sits 5 cm under its
// face" is true of every copy of the panel, wherever it ends up.
//
// Each light carries two halves that describe the SAME lamp to two different
// consumers:
//
//   bake    - a Blender Cycles area light, rebuilt by the bake script from the
//             glTF extras. Shape, size, spread, colour and watts are Blender's
//             own units, so what is authored here is what Cycles gets. Power is
//             TOTAL watts over the surface, so resizing changes the brightness.
//   runtime - the Babylon-Lite light the game creates for what a lightmap
//             cannot cover: dynamic props, the player, specular highlights.
//
// Either half may be "none". A flickering lamp is runtime-only - baking it
// would freeze one frame of the flicker into the wall. A bounce fill that only
// exists to lift a dark corner is bake-only, and costs the runtime nothing.
//
// EMISSION AXIS: a light emits along its own local -Y, so a default rotation of
// [0,0,0] is a ceiling panel shining at the floor. glTF export rotates the ship
// +90 degrees about X on the way into Blender, which lands that -Y on Blender's
// -Z - the axis an Area light emits along. The two conventions meet with no
// fix-up, which is why -Y was chosen over the more obvious -Z.

import { state, emit, pushUndo, hooks, eulerOf, setEuler } from "./editor.js";
import { getKitLights } from "./kit.js";

const { Vector3, Quaternion, Color3, TransformNode, MeshBuilder, StandardMaterial } = BABYLON;

/** Blender's Area light shapes, plus "none" for a light the bake ignores. */
export const LIGHT_SHAPES = ["none", "square", "rectangle", "disk", "ellipse"];

/** The runtime light kinds, plus "none" for a light that only ever bakes. */
export const LIGHT_TYPES = ["none", "point", "spot", "directional"];

/** Shapes whose second dimension is authored; the others are square/circular. */
const TWO_SIZED = ["rectangle", "ellipse"];

/**
 * A ceiling panel, because that is what almost every light in the kit is.
 *
 * 40 W over a 50 cm square reads as a bright utility panel in Cycles, and the
 * runtime half is a clustered point light: the ship has enough lamps that the
 * cluster is the only affordable way to run them, and a point light needs no
 * aiming to match an area light pointing straight down.
 */
export const DEFAULT_LIGHT = {
  offset: [0, 0, 0],
  rotation: [0, 0, 0],
  bake: {
    shape: "square",
    sizeX: 0.5,
    sizeY: 0.5,
    spread: 180,
    color: [1, 1, 1],
    watts: 40,
  },
  runtime: {
    type: "point",
    clustered: true,
    color: [1, 1, 1],
    intensity: 1,
    range: 8,
    angle: 90,
    castsShadows: false,
  },
};

function nextLightId() {
  let n = 1;
  while (state.lights.has(`L${String(n).padStart(4, "0")}`)) n++;
  return `L${String(n).padStart(4, "0")}`;
}

function num(v, fallback, min, max) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(n, min), max);
}

function colorOf(v, fallback) {
  if (!Array.isArray(v) || v.length !== 3) return [...fallback];
  return v.map((c, i) => num(c, fallback[i], 0, 1));
}

function vec3Of(v, fallback) {
  if (!Array.isArray(v) || v.length !== 3) return [...fallback];
  return v.map((c, i) => num(c, fallback[i], -1e4, 1e4));
}

/**
 * Settle the fields that cannot disagree, on the way in.
 *
 * Every one of these is a combination the engine or Blender has no meaning for,
 * so rather than trusting the inspector, a loaded manifest and the per-module
 * defaults to each get it right, they are resolved in one place - the same way
 * normalizeDoorSides() settles a skybox door's `sealed` flag.
 */
export function normalizeLight(light) {
  const b = light.bake;
  const r = light.runtime;

  if (!LIGHT_SHAPES.includes(b.shape)) b.shape = DEFAULT_LIGHT.bake.shape;
  b.sizeX = num(b.sizeX, DEFAULT_LIGHT.bake.sizeX, 1e-3, 100);
  // Blender reads one size for a square or a disk, so a stored sizeY would be a
  // value the inspector shows, the bake ignores, and the next shape change
  // silently resurrects. Mirroring it keeps the record honest.
  b.sizeY = TWO_SIZED.includes(b.shape)
    ? num(b.sizeY, DEFAULT_LIGHT.bake.sizeY, 1e-3, 100)
    : b.sizeX;
  b.spread = num(b.spread, DEFAULT_LIGHT.bake.spread, 0, 180);
  b.watts = num(b.watts, DEFAULT_LIGHT.bake.watts, 0, 1e6);
  b.color = colorOf(b.color, DEFAULT_LIGHT.bake.color);

  if (!LIGHT_TYPES.includes(r.type)) r.type = DEFAULT_LIGHT.runtime.type;
  r.intensity = num(r.intensity, DEFAULT_LIGHT.runtime.intensity, 0, 1e4);
  r.range = num(r.range, DEFAULT_LIGHT.runtime.range, 1e-3, 1e4);
  r.angle = num(r.angle, DEFAULT_LIGHT.runtime.angle, 1, 179);
  r.color = colorOf(r.color, DEFAULT_LIGHT.runtime.color);
  // A directional light has no position to cluster and no falloff to bin, so
  // the cluster cannot hold one.
  r.clustered = !!r.clustered && (r.type === "point" || r.type === "spot");
  // Two separate walls, both engine limits rather than taste: Babylon-Lite has
  // no cube shadow generator, so a point light can never cast; and a clustered
  // light is packed into a data texture that carries no shadow map, whatever
  // its kind.
  r.castsShadows = !!r.castsShadows && !r.clustered
    && (r.type === "spot" || r.type === "directional");
  return light;
}

/**
 * The plain-data half of a light, defaults filled in and settled.
 *
 * `offset` and `rotation` are deliberately NOT stored on the record: the scene
 * node is where they live, so a viewport drag, an arrow-key nudge and the
 * inspector all edit the same thing, and none of them has to remember to write
 * a copy back. serializeLights reads them off the node, exactly as
 * serializeMarkers does for a door.
 */
function makeLight(id, owner, opts = {}) {
  return normalizeLight({
    id,
    owner,
    // What the inspector keys off to know it is not looking at a placement -
    // the same field a marker and a collision primitive carry.
    type: "light",
    bake: { ...DEFAULT_LIGHT.bake, ...(opts.bake || {}) },
    runtime: { ...DEFAULT_LIGHT.runtime, ...(opts.runtime || {}) },
  });
}

/** Where a light sits and which way it points, in its owner's local space. */
export function lightOffset(light) {
  return light.node.position.asArray().map((v) => Math.round(v * 1e4) / 1e4);
}
export function lightRotation(light) {
  return eulerOf(light.node).map((v) => Math.round(v * 1e4) / 1e4);
}

/**
 * Attach a light to a placement.
 *
 * @param placementId - the element the light rides. The staging bench is a
 *   working copy that never reaches the ship, so it cannot own one.
 * @returns the light, or null when there is nothing to attach it to.
 */
export function addLight(placementId, opts = {}) {
  const owner = state.placements.get(placementId);
  if (!owner || owner.stage) return null;
  if (!opts.silent) pushUndo();

  const light = makeLight(opts.id || nextLightId(), placementId, opts);
  const node = new TransformNode(light.id, state.scene);
  node.parent = owner.node;
  light.node = node;
  node.position.copyFrom(Vector3.FromArray(vec3Of(opts.offset, DEFAULT_LIGHT.offset)));
  setEuler(node, vec3Of(opts.rotation, DEFAULT_LIGHT.rotation));
  node.metadata = { light };
  state.lights.set(light.id, light);
  rebuildGizmo(light);
  if (!opts.silent) emit("lights");
  return light;
}

export function removeLight(id, silent = false) {
  const light = state.lights.get(id);
  if (!light) return;
  light.node.dispose();
  state.lights.delete(id);
  if (!silent) emit("lights");
}

/**
 * Drop every light riding a placement.
 *
 * Called from removePlacement: the nodes go with the parent either way, so
 * without this the entries would outlive their scene nodes and the next
 * serialize() would write lights that point at an element that is gone.
 */
export function removeLightsOf(placementId) {
  for (const [id, l] of [...state.lights]) {
    if (l.owner === placementId) state.lights.delete(id);
  }
}

/** The lights riding a placement, in creation order. */
export function lightsOf(placementId) {
  return [...state.lights.values()].filter((l) => l.owner === placementId);
}

/** Give a copied element the same lights as the one it was copied from. */
export function copyLightsTo(fromPlacementId, toPlacementId) {
  for (const l of lightsOf(fromPlacementId)) {
    addLight(toPlacementId, {
      offset: lightOffset(l),
      rotation: lightRotation(l),
      bake: { ...l.bake, color: [...l.bake.color] },
      runtime: { ...l.runtime, color: [...l.runtime.color] },
      silent: true,
    });
  }
}

/** Duplicate one authored light beside its source in the owner's local X. */
export function duplicateLight(id) {
  const light = state.lights.get(id);
  if (!light) return null;
  const offset = lightOffset(light);
  offset[0] += state.snap.pos || 1;
  return addLight(light.owner, {
    offset,
    rotation: lightRotation(light),
    bake: { ...light.bake, color: [...light.bake.color] },
    runtime: { ...light.runtime, color: [...light.runtime.color] },
    silent: true,
  });
}

/**
 * Reattach a light to another placement without moving it in the viewport.
 *
 * The light's authored transform is local to its owner, so changing the
 * parent requires converting the current world matrix into the new owner's
 * frame. Light scale is intentionally not authored; any scale found while
 * decomposing the conversion belongs to the placement hierarchy and is left
 * out of the light record.
 */
export function setLightOwner(id, placementId) {
  const light = state.lights.get(id);
  const owner = state.placements.get(placementId);
  if (!light || !owner || owner.stage) return null;
  if (light.owner === placementId) return light;

  light.node.computeWorldMatrix(true);
  owner.node.computeWorldMatrix(true);
  const local = light.node.getWorldMatrix().multiply(owner.node.getWorldMatrix().clone().invert());
  const scale = new Vector3();
  const rotation = new Quaternion();
  const position = new Vector3();
  local.decompose(scale, rotation, position);

  light.node.parent = owner.node;
  light.node.position.copyFrom(position);
  light.node.rotationQuaternion = rotation;
  light.node.scaling.set(1, 1, 1);
  light.owner = placementId;
  emit("lights");
  return light;
}

/**
 * Give a freshly placed element the lights its module comes with.
 *
 * A seed, not a derivation: it fires once, from placeAt, and from then on the
 * lights belong to that element. That is what lets an authored tweak survive a
 * save, and it is why the two paths that already carry lights of their own -
 * duplicateSelected, which copies them, and restoreFrom, which loads them -
 * place with `noLights` rather than being seeded and then corrected.
 *
 * @returns the lights created, so a caller can report or select them.
 */
export function seedKitLights(placementId) {
  const owner = state.placements.get(placementId);
  if (!owner || owner.stage) return [];
  const made = [];
  for (const spec of getKitLights(owner.module)) {
    if (!spec || typeof spec !== "object") continue;
    const light = addLight(placementId, { ...spec, id: undefined, silent: true });
    if (light) made.push(light);
  }
  return made;
}

export function findLightByNode(node) {
  for (const l of state.lights.values()) if (l.node === node) return l;
  return null;
}

/** Move or turn a light within its owner, in the owner's local space. */
export function setLightTransform(id, patch) {
  const light = state.lights.get(id);
  if (!light) return null;
  if (patch.offset) {
    light.node.position.copyFrom(
      Vector3.FromArray(vec3Of(patch.offset, lightOffset(light))));
  }
  if (patch.rotation) setEuler(light.node, vec3Of(patch.rotation, lightRotation(light)));
  emit("lights");
  return light;
}

/**
 * Edit one half of a light.
 *
 * Both halves go back through normalizeLight, not just the one that changed:
 * switching the runtime type to "point" has to clear `castsShadows`, and
 * switching the shape to "square" has to fold sizeY back onto sizeX.
 */
export function setLightPart(id, part, patch) {
  const light = state.lights.get(id);
  if (!light || (part !== "bake" && part !== "runtime")) return null;
  Object.assign(light[part], patch);
  normalizeLight(light);
  // The gizmo IS the bake shape, drawn - so it has to be rebuilt, not nudged.
  rebuildGizmo(light);
  emit("lights");
  return light;
}

export function serializeLights() {
  return [...state.lights.values()].map((l) => ({
    id: l.id,
    owner: l.owner,
    offset: lightOffset(l),
    rotation: lightRotation(l),
    bake: { ...l.bake, color: round(l.bake.color) },
    runtime: { ...l.runtime, color: round(l.runtime.color) },
  }));
}

export function deserializeLights(list) {
  for (const id of [...state.lights.keys()]) removeLight(id, true);
  for (const l of list || []) {
    // The owner is written first by both serialize() and the manifest, so a
    // miss means the element was deleted under it - drop the light rather than
    // stranding it at the world origin.
    if (!state.placements.has(l.owner)) continue;
    addLight(l.owner, { ...l, silent: true });
  }
  emit("lights");
}

function round(a) { return a.map((v) => Math.round(v * 1e4) / 1e4); }

/**
 * The lights that reach ship.glb, each with the `extras` its node will carry.
 *
 * Two consumers read this from the file and neither can see the editor: the
 * Blender bake script rebuilds a Cycles Area light from `bake`, and the runtime
 * builds a Babylon light from `runtime`. So the whole record goes out, not a
 * summary of it - a light that has to be re-derived on the far side is a light
 * that will drift.
 *
 * `kind` is what tells the two apart from a placement's extras, which carry a
 * `module` instead. `chunk` rides along so the bake can work one chunk at a
 * time without walking back up the hierarchy.
 *
 * The staging bench never reaches the ship, so neither do its lights - and a
 * light whose owner has been deleted is skipped rather than exported pointing
 * at nothing.
 */
export function lightsForExport() {
  const out = [];
  for (const l of state.lights.values()) {
    const owner = state.placements.get(l.owner);
    if (!owner || owner.stage) continue;
    out.push({
      id: l.id,
      node: l.node,
      extras: {
        id: l.id,
        kind: "light",
        owner: l.owner,
        chunk: owner.chunk,
        bake: { ...l.bake, color: round(l.bake.color) },
        runtime: { ...l.runtime, color: round(l.runtime.color) },
      },
    });
  }
  return out;
}

// ----------------------------------------------------------------- gizmo
//
// A light has no mesh of its own, so without this it would be invisible and
// unpickable - authored only through ids typed into a console. The gizmo is
// the bake shape drawn at its real size, plus a stub along the emission axis,
// which makes "is this panel pointing at the floor or into the ceiling?" a
// thing you can see rather than a rotation you have to read.
//
// It is furniture, not art: `metadata.gizmo` keeps it out of the .glb, out of
// the bounds a placement reports, and out of the veil clones - see isGizmoMesh.

/** How far the emission stub reaches, in metres. */
const STUB = 0.6;

let gizmoMats = null;

function materials(scene) {
  if (!gizmoMats) {
    const lit = new StandardMaterial("GIZMO_light", scene);
    lit.emissiveColor = new Color3(1, 0.85, 0.35);
    lit.diffuseColor = Color3.Black();
    lit.specularColor = Color3.Black();
    lit.alpha = 0.35;
    lit.backFaceCulling = false;
    lit.disableLighting = true;
    // A lamp the bake ignores is still a real light at runtime, and telling the
    // two apart at a glance is the whole reason either half may be "none".
    const off = lit.clone("GIZMO_light_off");
    off.emissiveColor = new Color3(0.45, 0.55, 0.7);
    gizmoMats = { lit, off };
  }
  return gizmoMats;
}

function disposeGizmo(light) {
  light.gizmo?.dispose();
  light.stub?.dispose();
  light.gizmo = null;
  light.stub = null;
}

function rebuildGizmo(light) {
  disposeGizmo(light);
  const scene = state.scene;
  if (!scene) return;
  const { lit, off } = materials(scene);
  const b = light.bake;
  const baking = b.shape !== "none";
  // A bake-only shape still needs something to click on, so a light the bake
  // ignores falls back to a fixed 25 cm plate rather than vanishing.
  const sx = baking ? b.sizeX : 0.25;
  const sy = baking ? b.sizeY : 0.25;

  const round2 = b.shape === "disk" || b.shape === "ellipse";
  const face = round2
    ? MeshBuilder.CreateDisc(`${light.id}_gizmo`,
      { radius: 0.5, tessellation: 24, sideOrientation: 2 }, scene)
    : MeshBuilder.CreatePlane(`${light.id}_gizmo`,
      { size: 1, sideOrientation: 2 }, scene);
  // Both are built in the XY plane facing +Z; a quarter turn about X sends that
  // +Z onto -Y, which is the axis the light emits along.
  face.rotation.x = Math.PI / 2;
  face.scaling.set(sx, sy, 1);
  face.parent = light.node;
  face.material = baking ? lit : off;
  face.isPickable = true;
  face.metadata = { lightRoot: light.node, gizmo: true };
  light.gizmo = face;

  const stub = MeshBuilder.CreateLines(`${light.id}_stub`, {
    points: [Vector3.Zero(), new Vector3(0, -STUB, 0)],
  }, scene);
  stub.color = (baking ? lit : off).emissiveColor;
  stub.parent = light.node;
  stub.isPickable = false;
  stub.metadata = { lightRoot: light.node, gizmo: true };
  light.stub = stub;
}

hooks.serializeLights = serializeLights;
hooks.deserializeLights = deserializeLights;
hooks.removeLightsOf = removeLightsOf;
hooks.removeLight = removeLight;
hooks.copyLightsTo = copyLightsTo;
hooks.seedKitLights = seedKitLights;
hooks.lightsForExport = lightsForExport;
