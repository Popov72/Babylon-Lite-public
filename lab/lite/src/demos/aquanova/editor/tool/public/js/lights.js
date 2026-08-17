// Authored lights.
//
// A light rides a placement: its node is a CHILD of that placement's node, so
// turning or moving the module carries its light with it and the authored
// offset/rotation are read in the module's own local space. That is what makes
// per-module defaults possible at all - "the panel's lamp sits 5 cm under its
// face" is true of every copy of the panel, wherever it ends up.
//
// A light is one record: the Babylon-Lite light the game creates. There is no
// second, pre-computed half - the lamps are the ship's whole direct lighting -
// so what is authored here is exactly what is rendered, in the editor's Runtime
// view and in the game alike.
//
// `type` may be "none", which switches a lamp off without deleting it: its
// position, colour and settings survive, which is what makes trying a room with
// one fewer light a two-click experiment rather than an edit you have to undo.
//
// EMISSION AXIS: a light emits along its own local -Y, so a default rotation of
// [0,0,0] is a ceiling panel shining at the floor. Down is the direction almost
// every lamp in the kit points, and -Y is the only axis that needs no rotation
// to get there - which is why it was chosen over the more obvious -Z.

import { state, emit, pushUndo, hooks, eulerOf, setEuler, benchOf } from "./editor.js";
import { getKitLights } from "./kit.js";

const { Vector3, Quaternion, Color3, TransformNode, MeshBuilder, StandardMaterial } = BABYLON;

/** The light kinds, plus "none" for a lamp that is switched off. */
export const LIGHT_TYPES = ["none", "point", "spot", "directional"];

/**
 * A ceiling panel, because that is what almost every light in the kit is.
 *
 * Clustered, because the ship has far more lamps than a forward pass can hold
 * in uniform slots; and a point light, because a ceiling panel shines straight
 * down over a whole room and needs no aiming to do it.
 */
export const DEFAULT_LIGHT = {
  offset: [0, 0, 0],
  rotation: [0, 0, 0],
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

/**
 * What each kind of lamp starts at, where that differs from the panel above.
 *
 * Intensity does not mean the same thing from kind to kind - a point light
 * fills a small room from the inside, where 1 is already bright, while a spot
 * is aimed at a surface metres away through a cone and needs two orders of
 * magnitude more before the wall it is pointed at looks lit at all. Carrying
 * one number across a type change is how a spot ends up looking broken.
 *
 * Only the fields a kind actually disagrees about are listed; everything else
 * comes from DEFAULT_LIGHT, so there is still one place a colour or the
 * clustering default is written down.
 */
export const LIGHT_TYPE_DEFAULTS = {
  spot: { intensity: 80, range: 6, angle: 120 },
};

/** The starting runtime record for one kind of lamp. */
export function defaultsFor(type) {
  return { ...DEFAULT_LIGHT.runtime, ...(LIGHT_TYPE_DEFAULTS[type] || {}), type };
}

/** The fields whose meaning changes with the kind of lamp. */
const TYPED_FIELDS = ["intensity", "range", "angle"];

/**
 * The values a lamp should take on when its kind changes.
 *
 * A field still sitting at the *outgoing* kind's default was never chosen, so
 * it follows the lamp to its new kind; a number that was typed in is kept,
 * whichever kind it was typed for. That is what makes "add a light, choose
 * spot" land on a usable spot while leaving a tuned one alone across a switch
 * to `none` and back.
 */
function retypeDefaults(runtime, type) {
  const was = defaultsFor(runtime.type);
  const now = defaultsFor(type);
  const out = {};
  for (const key of TYPED_FIELDS) {
    if (runtime[key] === was[key] && now[key] !== was[key]) out[key] = now[key];
  }
  return out;
}

function nextLightId() {
  let n = 1;
  while (state.lights.has(`L${String(n).padStart(4, "0")}`)
    || state.environmentProbes.has(`L${String(n).padStart(4, "0")}`)) n++;
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
 * Every one of these is a combination the engine has no meaning for, so rather
 * than trusting the inspector, a loaded manifest and the per-module defaults to
 * each get it right, they are resolved in one place - the same way
 * normalizeDoorSides() settles a skybox door's `sealed` flag.
 */
export function normalizeLight(light) {
  const r = light.runtime;

  if (!LIGHT_TYPES.includes(r.type)) r.type = DEFAULT_LIGHT.runtime.type;
  // Per kind, so a record that arrives without a cone - a kit seed, or a
  // manifest written before the field existed - gets the cone its kind wants
  // rather than the one a point light would have had.
  const d = defaultsFor(r.type);
  r.intensity = num(r.intensity, d.intensity, 0, 1e4);
  r.range = num(r.range, d.range, 1e-3, 1e4);
  r.angle = num(r.angle, d.angle, 1, 179);
  r.color = colorOf(r.color, d.color);
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
  const wanted = opts.runtime || {};
  return normalizeLight({
    id,
    owner,
    // What the inspector keys off to know it is not looking at a placement -
    // the same field a marker and a collision primitive carry.
    type: "light",
    // The kind is settled before its defaults are read, so a kit seed that asks
    // for a spot and nothing else gets a spot's numbers, not a point's.
    runtime: { ...defaultsFor(wanted.type ?? DEFAULT_LIGHT.runtime.type), ...wanted },
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
  // The collision bench holds stand-ins whose only purpose is to have a hull
  // fitted to them, and a lamp on one would be authored into nothing. The
  // compound bench is the opposite case - a wall *with* its lamp is the thing
  // being built - so only the collision bench refuses.
  if (!owner || benchOf(owner) === "collision") return null;
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
  const made = [];
  for (const l of lightsOf(fromPlacementId)) {
    const copy = addLight(toPlacementId, {
      offset: lightOffset(l),
      rotation: lightRotation(l),
      runtime: { ...l.runtime, color: [...l.runtime.color] },
      silent: true,
    });
    if (copy) made.push(copy);
  }
  // Returned so a caller that copied silently knows whether it has anything to
  // announce - the Runtime view rebuilds its lights off that announcement.
  return made;
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
 * Edit a light's settings.
 *
 * The patch goes back through normalizeLight rather than being assigned and
 * trusted: switching the type to "point" has to clear `castsShadows`, and
 * clustering one has to clear it too.
 *
 * A change of kind also brings that kind's defaults with it, for the fields
 * that were never chosen - see retypeDefaults. The patch is applied last, so a
 * caller that names a value in the same breath as the type still wins.
 */
export function setLightPart(id, part, patch) {
  const light = state.lights.get(id);
  if (!light || part !== "runtime") return null;
  const wasLive = light.runtime.type !== "none";
  const retyped = patch.type && patch.type !== light.runtime.type
    ? retypeDefaults(light.runtime, patch.type)
    : null;
  Object.assign(light[part], retyped, patch);
  normalizeLight(light);
  // The gizmo carries the on/off colour, so switching a lamp off - or back on -
  // has to rebuild it rather than nudge it. Nothing else on the record reaches
  // the gizmo, and rebuilding regardless would dispose the mesh on every wheel
  // notch, taking the selection outline attached to it along.
  if ((light.runtime.type !== "none") !== wasLive) rebuildGizmo(light);
  emit("lights");
  return light;
}

/**
 * Every authored light in the ship, as plain data.
 *
 * A bench lamp is skipped: the compound bench builds real lights on real
 * placements, and this feeds `serialize()`, which is the ship's undo snapshot
 * and the manifest's `lights` block. A lamp being fitted to a compound has no
 * business in either, and a snapshot carrying one would restore it onto an
 * owner the ship has never heard of.
 */
export function serializeLights() {
  return [...state.lights.values()]
    .filter((l) => !state.placements.get(l.owner)?.stage)
    .map((l) => ({
      id: l.id,
      owner: l.owner,
      offset: lightOffset(l),
      rotation: lightRotation(l),
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
 * The runtime reads these from the file and cannot see the editor, so the whole
 * record goes out, not a summary of it - a light that has to be re-derived on
 * the far side is a light that will drift.
 *
 * `kind` is what tells a light apart from a placement's extras, which carry a
 * `module` instead. `chunk` rides along so the runtime can light one room at
 * a time without walking back up the hierarchy.
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
        runtime: { ...l.runtime, color: round(l.runtime.color) },
      },
    });
  }
  return out;
}

// ----------------------------------------------------------------- gizmo
//
// A light has no mesh of its own, so without this it would be invisible and
// unpickable - authored only through ids typed into a console. The gizmo is a
// small plate at the lamp's position plus a stub along the emission axis, which
// makes "is this panel pointing at the floor or into the ceiling?" a thing you
// can see rather than a rotation you have to read.
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
    // A lamp switched off is still an authored lamp you have to be able to find
    // and click, and telling it apart at a glance is why "none" is a type.
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
  const on = light.runtime.type !== "none";

  // A fixed 25 cm plate: the lamp has no authored size, and a gizmo that grew
  // with range or intensity would read as geometry rather than as a handle.
  const face = MeshBuilder.CreatePlane(`${light.id}_gizmo`,
    { size: 0.25, sideOrientation: 2 }, scene);
  // Built in the XY plane facing +Z; a quarter turn about X sends that +Z onto
  // -Y, which is the axis the light emits along.
  face.rotation.x = Math.PI / 2;
  face.parent = light.node;
  face.material = on ? lit : off;
  face.isPickable = true;
  face.metadata = { lightRoot: light.node, gizmo: true };
  light.gizmo = face;
  const stub = MeshBuilder.CreateLines(`${light.id}_stub`, {
    points: [Vector3.Zero(), new Vector3(0, -STUB, 0)],
  }, scene);
  stub.color = (on ? lit : off).emissiveColor;
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
