// Door, portal and spawn markers.
//
// These are not kit modules: they are the metadata the portal renderer needs.
// A door marker owns the portal rectangle between two chunks and, optionally,
// the placements that act as its sliding leaves.

import { state, emit, pushUndo, worldBounds, hooks, refreshVeil } from "./editor.js";

const { MeshBuilder, StandardMaterial, Color3, Vector3, Quaternion, TransformNode } = BABYLON;

export const DEFAULT_DOOR = { width: 3.02, height: 4.05, triggerRadius: 3.5, slideDistance: 1.51 };

// Deleting a door must not make the next one reuse a live id, so probe for the
// first free slot rather than counting the markers.
function nextDoorId() {
  let n = 0;
  while (state.markers.has(`Door_D${String(n).padStart(2, "0")}`)) n++;
  return `Door_D${String(n).padStart(2, "0")}`;
}

let doorMat = null;

function materials(scene) {
  if (!doorMat) {
    doorMat = new StandardMaterial("MARKER_door", scene);
    doorMat.emissiveColor = new Color3(1.0, 0.45, 0.1);
    doorMat.diffuseColor = Color3.Black();
    doorMat.specularColor = Color3.Black();
    doorMat.alpha = 0.28;
    doorMat.backFaceCulling = false;
    doorMat.disableLighting = true;
  }
  return { doorMat };
}

// ------------------------------------------------------------------ doors

export function addDoor(position, opts = {}) {
  if (!opts.silent) pushUndo();
  const scene = state.scene;
  const { doorMat: dm } = materials(scene);
  const id = opts.id || nextDoorId();

  const data = {
    id,
    type: "door",
    width: opts.width ?? DEFAULT_DOOR.width,
    height: opts.height ?? DEFAULT_DOOR.height,
    // empty means "auto": the manifest resolves it from the chunk volumes on
    // either side. Defaulting to the active chunk would bias that guess.
    chunkA: opts.chunkA ?? "",
    chunkB: opts.chunkB ?? "",
    triggerRadius: opts.triggerRadius ?? DEFAULT_DOOR.triggerRadius,
    slideDistance: opts.slideDistance ?? DEFAULT_DOOR.slideDistance,
    // A portal you can see through but not walk through - a window onto space.
    // The renderer still draws the far chunk; collision generation keeps the
    // opening solid. Default false, so every manifest written before this reads
    // back as an ordinary doorway.
    sealed: !!opts.sealed,
    leaves: opts.leaves ? [...opts.leaves] : [],
  };

  const root = new TransformNode(id, scene);
  root.position.copyFrom(position);
  root.rotationQuaternion = opts.rotation
    ? Quaternion.FromEulerAngles(
        opts.rotation[0] * Math.PI / 180,
        opts.rotation[1] * Math.PI / 180,
        opts.rotation[2] * Math.PI / 180)
    : Quaternion.Identity();
  if (opts.scale) root.scaling.copyFrom(Vector3.FromArray(opts.scale));

  data.node = root;
  rebuildDoorMesh(data);
  root.metadata = { marker: data };
  state.markers.set(id, data);
  if (!opts.silent) emit("markers");
  return data;
}

function rebuildDoorMesh(data) {
  data.mesh?.dispose();
  const quad = MeshBuilder.CreatePlane(`${data.id}_portal`,
    { width: data.width, height: data.height, sideOrientation: 2 }, state.scene);
  quad.material = doorMat;
  quad.parent = data.node;
  quad.position.y = data.height / 2;
  quad.isPickable = true;
  quad.metadata = { markerRoot: data.node };
  data.mesh = quad;
}

export function resizeDoor(data, width, height) {
  data.width = width;
  data.height = height;
  rebuildDoorMesh(data);
  // the stand-in was cloned from the mesh that just got disposed
  refreshVeil(data.id);
}

/**
 * Build a door from the current selection: the marker is centred on the
 * selected placements and sized to their combined opening, and those
 * placements become its animated leaves.
 */
export function doorFromSelection() {
  if (!state.selection.length) return null;
  pushUndo();
  let min = null, max = null;
  for (const id of state.selection) {
    const b = worldBounds(state.placements.get(id).node);
    if (!b) continue;
    min = min ? Vector3.Minimize(min, b.min) : b.min.clone();
    max = max ? Vector3.Maximize(max, b.max) : b.max.clone();
  }
  if (!min) return null;

  const size = max.subtract(min);
  // the thin axis is the door's normal; the wide horizontal axis is its span
  const alongZ = size.x >= size.z;
  const centre = new Vector3((min.x + max.x) / 2, min.y, (min.z + max.z) / 2);
  return addDoor(centre, {
    width: alongZ ? size.x : size.z,
    height: size.y,
    rotation: [0, alongZ ? 0 : 90, 0],
    leaves: [...state.selection],
    silent: true,
  });
}

// ---------------------------------------------------------------- generic

export function removeMarker(id, silent = false) {
  const m = state.markers.get(id);
  if (!m) return;
  m.mesh?.dispose();
  m.node.dispose();
  state.markers.delete(id);
  if (!silent) emit("markers");
}

export function findMarkerByNode(node) {
  for (const m of state.markers.values()) if (m.node === node) return m;
  return null;
}

export function serializeMarkers() {
  return [...state.markers.values()].map((m) => ({
    id: m.id,
    type: m.type,
    position: round(m.node.position.asArray()),
    rotation: round(eulerDeg(m.node)),
    // Doors are scalable like anything else, so their scale has to survive a
    // reload - without this a resized door would spring back to its authored
    // width on the next Load, and the manifest would quietly change with it.
    scale: round(m.node.scaling.asArray()),
    width: m.width, height: m.height,
    chunkA: m.chunkA, chunkB: m.chunkB,
    triggerRadius: m.triggerRadius, slideDistance: m.slideDistance,
    sealed: !!m.sealed,
    leaves: [...m.leaves],
  }));
}

export function deserializeMarkers(list) {
  for (const id of [...state.markers.keys()]) removeMarker(id, true);
  for (const m of list || []) {
    // Spawn markers are gone: a start position is now a dummy element carrying
    // a "player_startingpos" behaviour, so an old one is dropped rather than
    // resurrected as a marker nothing can edit.
    if (m.type !== "door") continue;
    addDoor(Vector3.FromArray(m.position), { ...m, silent: true });
  }
  emit("markers");
}

/** Portal rectangle in world space, derived from a door marker. */
export function portalOf(door) {
  const m = door.node.getWorldMatrix();
  const hw = door.width / 2;
  const local = [
    new Vector3(-hw, 0, 0), new Vector3(hw, 0, 0),
    new Vector3(hw, door.height, 0), new Vector3(-hw, door.height, 0),
  ];
  const corners = local.map((v) => Vector3.TransformCoordinates(v, m));
  const normal = Vector3.TransformNormal(new Vector3(0, 0, 1), m).normalize();
  const centre = corners.reduce((a, c) => a.add(c), Vector3.Zero()).scale(0.25);
  return {
    id: `Portal_${door.id}`,
    chunkA: door.chunkA,
    chunkB: door.chunkB,
    door: door.id,
    centre: round(centre.asArray()),
    normal: round(normal.asArray()),
    corners: corners.map((c) => round(c.asArray())),
  };
}

function eulerDeg(node) {
  const q = node.rotationQuaternion || Quaternion.FromEulerVector(node.rotation);
  const e = q.toEulerAngles();
  return [e.x * 180 / Math.PI, e.y * 180 / Math.PI, e.z * 180 / Math.PI];
}

function round(a) { return a.map((v) => Math.round(v * 1e4) / 1e4); }

hooks.serializeMarkers = serializeMarkers;
hooks.deserializeMarkers = deserializeMarkers;
