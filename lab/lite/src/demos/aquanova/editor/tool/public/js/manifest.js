// Manifest I/O and glb export.
//
// ship_manifest.json is the source of truth: it stores every placement as a
// module id plus a transform, so the editor can be restored exactly. ship.glb
// is a derived runtime artefact and is never read back.

import {
  state, serialize, deserialize, worldBounds, withAuthoredMaterials, shipPlacements,
  loadModuleCollision, serializeModuleCollision, emit, hooks,
  serializeView, applyView, serializeEnvironment, serializeEditorEnvironment,
  applyEnvironment, whileBusy, withVeilSuspended, isVeilClone,
} from "./editor.js";
import { portalOf } from "./markers.js";

const { TransformNode, Vector3, Quaternion, Matrix } = BABYLON;

export const SCHEMA = 2;

// A door marker only knows which two chunks it joins if the user said so;
// otherwise infer it from the chunk volumes the doorway sits between. Distance
// is measured to the box, not to its centre, because corridors are long and
// their centres can be far from any doorway.
function distanceToBox(p, box) {
  const dx = Math.max(box.min.x - p.x, 0, p.x - box.max.x);
  const dy = Math.max(box.min.y - p.y, 0, p.y - box.max.y);
  const dz = Math.max(box.min.z - p.z, 0, p.z - box.max.z);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function resolveDoorChunks(door, boxes) {
  const c = door.node.getAbsolutePosition();
  const near = boxes
    .map((b) => ({ id: b.id, d: distanceToBox(c, b) }))
    .sort((x, y) => x.d - y.d);

  let a = door.chunkA || "";
  let b = door.chunkB || "";
  // never let the fallback pick the side that is already taken
  if (!a) a = near.find((n) => n.id !== b)?.id || "";
  if (!b) b = near.find((n) => n.id !== a)?.id || "";
  return [a, b];
}

const inferChunks = resolveDoorChunks;

/**
 * Editor space to glTF space: negate X.
 *
 * The editor is a left-handed Babylon scene; glTF is right-handed, and the
 * exporter mirrors X on the way out - measured, an element at editor `[7,3,5]`
 * lands in the .glb at `[-7,3,5]`. The runtime loads that .glb (its loader
 * mirrors X back) and reads this manifest as **glTF space**, negating X in a
 * dozen places: colliders, portal openings, room tests, the player's facing.
 *
 * So every field the runtime consumes has to be written in glTF space, or the
 * whole ship is mirrored against its own geometry. The fields only the tool
 * reads back - `instances`, `markers`, `view` - stay in editor space, because
 * they exist to rebuild the editor and never leave it.
 */
function toGltf(v) { return [-v[0], v[1], v[2]]; }

/** An axis-aligned box mirrors on X, which swaps its own min and max. */
function boxToGltf(min, max) {
  return { min: [-max[0], min[1], min[2]], max: [-min[0], max[1], max[2]] };
}

/**
 * What an element is called in ship.glb: its own name, or its id when it has
 * none. Everything the runtime resolves - behaviours, door leaves - goes
 * through this one function, so the manifest can never name a node the export
 * does not.
 */
export function nodeNameOf(placement) {
  return String(placement?.name || "").trim() || placement?.id || "";
}

/**
 * The `behaviors` library and the `entities` that carry them.
 *
 * Definitions are written through untouched: the body is arbitrary JSON because
 * the runtime owns which flags exist, and a tool that normalised the ones it
 * happened to know about would quietly drop the rest.
 *
 * `linked` is omitted when empty rather than written as `[]` - the absence is
 * what "this one stands alone" means - and entries pointing at a behaviour that
 * no longer exists are dropped, since the runtime would only ignore them.
 */
function serializeBehaviors() {
  return Object.fromEntries(
    [...state.behaviors].map(([k, v]) => [k, JSON.parse(JSON.stringify(v))]));
}

function serializeEntities() {
  const out = {};
  for (const [node, list] of state.entities) {
    const kept = list
      .filter((b) => state.behaviors.has(b.name))
      .map((b) => ({
        name: b.name,
        ...(b.linked.length ? { linked: [...b.linked] } : {}),
        ...(b.excludeSDF?.length ? { excludeSDF: [...b.excludeSDF] } : {}),
        ...(b.direction ? { direction: toGltf(b.direction) } : {}),
      }));
    if (kept.length) out[node] = { behaviors: kept };
  }
  return out;
}

/**
 * One Havok record from a world matrix.
 *
 * Everything is read off the matrix's own basis rows rather than decomposed.
 * `Matrix.decompose()` is ambiguous the moment a transform is mirrored - a
 * negative determinant has no unique rotation/scale split - and a placement
 * scaled [-1,1,1] is completely ordinary in this kit. Reading the rows is
 * exact: their lengths are the extents and their directions are the axes.
 *
 * A box is symmetric under an axis flip, so when the composed basis comes out
 * left-handed one axis is negated to describe the identical box with a rotation
 * that actually exists. A capsule or cylinder needs no such care: its two
 * endpoints carry its axis, which is the Havok signature.
 */
function shapeRecord(id, kind, world) {
  const centre = new Vector3(world.m[12], world.m[13], world.m[14]);
  const ax = new Vector3(world.m[0], world.m[1], world.m[2]);
  const ay = new Vector3(world.m[4], world.m[5], world.m[6]);
  const az = new Vector3(world.m[8], world.m[9], world.m[10]);
  const shape = {
    id, kind,
    centre: toGltf(r(centre.asArray())),
  };

  if (kind === "box") {
    shape.halfExtents = r([ax.length() / 2, ay.length() / 2, az.length() / 2]);
    const bx = ax.clone().normalize();
    const by = ay.clone().normalize();
    const bz = az.clone().normalize();
    if (Vector3.Dot(Vector3.Cross(bx, by), bz) < 0) bx.scaleInPlace(-1);
    const m = new Matrix();
    Matrix.FromXYZAxesToRef(bx, by, bz, m);
    const q = Quaternion.FromRotationMatrix(m);
    // mirroring X flips the handedness of the turn as well as the position
    shape.rotation = r([-q.x, q.y, q.z, -q.w]);
  } else if (kind === "sphere") {
    shape.radius = r([ax.length() / 2])[0];
  } else {
    const radius = ax.length() / 2;
    shape.radius = r([radius])[0];
    shape.height = r([ay.length()])[0];
    // The segment endpoints, so the runtime needs no quaternion at all: the
    // local Y axis turned into world space, half a *segment* either way.
    //
    // A capsule's height is the whole pill, caps included - the same reading as
    // a box's side or a sphere's diameter, and what the editor draws. Havok's
    // capsule is that segment grown by the radius in every direction, so the
    // segment is one diameter shorter than the height. A cylinder has flat
    // ends: its segment is its full height. Getting this wrong makes every
    // capsule a diameter taller in play than it looks in the editor.
    const span = kind === "capsule" ? Math.max(ay.length() - radius * 2, 0) : ay.length();
    const half = ay.length() > 1e-9 ? ay.scale(0.5 * span / ay.length()) : Vector3.Zero();
    shape.pointA = toGltf(r(centre.subtract(half).asArray()));
    shape.pointB = toGltf(r(centre.add(half).asArray()));
  }
  return shape;
}

/**
 * The collision shapes, per chunk, in the form Havok's constructors take.
 *
 * A box gets a centre, a quaternion and half-extents. A sphere gets a centre
 * and a radius. A capsule or cylinder gets the two endpoints of its segment,
 * because that is literally the Havok signature - and it is how an arbitrarily
 * oriented capsule is expressed, since those shapes take no quaternion. Note
 * that a capsule's `height` is the whole pill and its segment is a diameter
 * shorter, while a cylinder's segment *is* its height.
 *
 * **Room shapes only.** A module's hull is written once in `moduleCollision`
 * and instanced by the runtime, which already has every placement's module,
 * chunk and transform in `instances`. Writing it out per placement as well was
 * pure duplication, and the kind that grows: placements x shapes rather than
 * modules x shapes.
 *
 * All of it is mirrored on X, like every other runtime-facing field.
 */
function collisionByChunk() {
  const out = {};
  for (const c of state.colliders.values()) {
    if (c.stage) continue;             // a working copy, not part of the ship
    c.node.computeWorldMatrix(true);
    (out[c.chunk] ||= []).push(shapeRecord(c.id, c.kind, c.node.getWorldMatrix()));
  }
  return out;
}

/**
 * The shapes authored on each kit module, in the module's own local space.
 *
 * What the runtime instances: for each entry in `instances`, look its module up
 * here and compose with the placement's transform. One Havok shape per module,
 * reused across every body that needs it.
 *
 * `moduleShapes` beside it is the same hulls in the editor's own coordinates -
 * the authoring source both this and the editor's reload come from. The two are
 * one transform apart, which is why they are two keys and not one: they were
 * one key once, and a reload silently threw every shape away because the reader
 * expected the other form.
 */
function moduleCollision() {
  const out = {};
  for (const [moduleId, shapes] of state.moduleCollision) {
    if (!shapes?.length) continue;
    out[moduleId] = shapes.map((s, i) => shapeRecord(`${moduleId}:${i}`, s.kind,
      Matrix.Compose(
        Vector3.FromArray(s.scale),
        Quaternion.FromEulerAngles(
          s.rotation[0] * Math.PI / 180,
          s.rotation[1] * Math.PI / 180,
          s.rotation[2] * Math.PI / 180),
        Vector3.FromArray(s.position))));
  }
  return out;
}

export function buildManifest() {  const layout = serialize();

  // Every derived figure below - chunk bounds, portal corners - is read off a
  // world matrix, and Babylon only refreshes those at render time. An export
  // fired straight after a scale (a wheel notch, an inspector keystroke) would
  // otherwise measure the ship as it was one frame ago, and the door's own
  // width, taken from `scaling` directly, would disagree with its portal.
  for (const p of shipPlacements()) p.node.computeWorldMatrix(true);
  for (const m of state.markers.values()) m.node.computeWorldMatrix(true);

  const boxes = [];
  const chunks = state.chunks.map((id) => {
    const members = shipPlacements().filter((p) => p.chunk === id);
    let min = null, max = null;
    let meshCount = 0;
    for (const m of members) {
      meshCount += m.node.getChildMeshes().filter((x) => !isVeilClone(x)).length;
      const b = worldBounds(m.node);
      if (!b) continue;
      min = min ? Vector3.Minimize(min, b.min) : b.min.clone();
      max = max ? Vector3.Maximize(max, b.max) : b.max.clone();
    }
    if (min) boxes.push({ id, min, max });
    return {
      id,
      node: `CHUNK_${id}`,
      aabb: min ? boxToGltf(r(min.asArray()), r(max.asArray())) : null,
      instanceCount: members.length,
      meshCount,
    };
  });

  const doors = [];
  const portals = [];
  for (const m of state.markers.values()) {
    if (m.type !== "door") continue;
    const [a, b] = inferChunks(m, boxes);
    const resolved = { ...m, chunkA: a, chunkB: b };
    const p = portalOf(resolved);
    portals.push({
      ...p,
      centre: toGltf(p.centre),
      normal: toGltf(p.normal),
      corners: p.corners.map(toGltf),
    });
    doors.push({
      id: m.id,
      node: m.id,
      chunkA: a,
      chunkB: b,
      position: toGltf(r(m.node.getAbsolutePosition().asArray())),
      // The node's own scale counts: portalOf() reads the world matrix, so the
      // portal already grows with it, and a raw width here would disagree with
      // the corners the runtime gets. Magnitude only - a mirrored door is still
      // a door of the same size.
      width: r([m.width * Math.abs(m.node.scaling.x)])[0],
      height: r([m.height * Math.abs(m.node.scaling.y)])[0],
      triggerRadius: m.triggerRadius,
      // The far side can be seen but not reached - a window onto space rather
      // than a doorway. Written on the door only for now; collision generation
      // will read it when that work happens. Default false, so every manifest
      // written before this reads back as an ordinary doorway.
      sealed: !!m.sealed,
      leaves: m.leaves
        .filter((id) => state.placements.has(id))
        .map((id) => ({
          node: nodeNameOf(state.placements.get(id)),
          openDistance: m.slideDistance,
        })),
    });
  }

  const adjacency = Object.fromEntries(state.chunks.map((c) => [c, []]));
  for (const p of portals) {
    if (adjacency[p.chunkA]) adjacency[p.chunkA].push({ to: p.chunkB, portal: p.id });
    if (adjacency[p.chunkB]) adjacency[p.chunkB].push({ to: p.chunkA, portal: p.id });
  }

  return {
    generator: "SciFiShip layout tool",
    schema: SCHEMA,
    savedAt: new Date().toISOString(),
    units: "metres",
    up: "Y (glTF)",
    grid: { tile: 4 },
    // Ship-wide authoring constants - see state.config. They have to travel
    // with the ship: refitting collision with a different shell thickness
    // produces different geometry, so a manifest that dropped them would come
    // back looking identical and then fit differently the next time you
    // pressed Generate.
    config: layout.config,
    kitDir: state.kitDir || null,
    chunks,
    // "node" is what the element is called in ship.glb - its own name, or its
    // id when it has none - and is the only handle the runtime needs. "id" is
    // the editor's, and is what the tool reloads from.
    instances: layout.instances.map((i) => ({
      ...i, node: nodeNameOf(state.placements.get(i.id)) || i.id,
    })),
    markers: layout.markers,
    // The editor's own record of every collision primitive: kind plus a plain
    // transform, which is what it reloads from. `collision` below is the same
    // information turned into what Havok's constructors take, and is derived -
    // this is the source. Without it a saved ship came back with no collision
    // at all, because restoreFrom() reads this key and nothing wrote it.
    colliders: layout.colliders,
    // A room's own one-off shapes, in glTF space, in the form Havok's
    // constructors take. Grouped by chunk because collision is streamed per
    // room, and a flat list would make every room filter the whole ship.
    //
    // Sizes are the *effective* ones, not the editor's raw scale: a unit box
    // scaled 4x2x0.2 is written as half-extents, and a capsule as the radius
    // and the two endpoints Havok's constructor actually takes.
    //
    // A module's hull is NOT expanded into here. It is written once below and
    // instanced by the runtime, which already knows every placement's module,
    // chunk and transform from `instances`. Expanding it as well was pure
    // duplication of the sort that grows: placements x shapes, against
    // modules x shapes.
    collision: collisionByChunk(),
    // What each kit module carries, in its own local space, in Havok's terms -
    // one shape per module, for the runtime to instance and to share.
    moduleCollision: moduleCollision(),
    // The same hulls in the editor's own coordinates. The authoring source both
    // of the above and the editor's own reload come from.
    moduleShapes: layout.moduleShapes,
    stageLayout: layout.stageLayout,
    activeChunk: layout.activeChunk,
    // The sims a liquefied element may use. Seeded from the tool's config.json,
    // but a loaded ship's own list wins - see restoreFrom - so the two cannot
    // drift apart behind your back.
    fluidSim: [...state.fluidSim],
    behaviors: serializeBehaviors(),
    entities: serializeEntities(),
    // where you were standing when you saved, so a reload puts you back - the
    // ship's own viewpoint, which is not the live camera while the bench is open
    view: hooks.shipViewpoint?.() || serializeView(),
    // and what it was lit by, so a reload looks the same
    environment: serializeEnvironment(),
    // the editor's own Env/Exposure, which the demos must not read
    editorEnvironment: serializeEditorEnvironment(),
    portals,
    doors,
    adjacency,
  };
}

export async function saveLayout(name) {
  const body = JSON.stringify(buildManifest(), null, 2);
  const url = name ? `/api/layout?name=${encodeURIComponent(name)}` : "/api/layout";
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  if (!res.ok) throw new Error(await res.text());
  // Collision goes to its own file as well as into the manifest. The manifest
  // is this ship; the file is the kit's, and is what you carry to the next one.
  //
  // A failure here must not fail the save: the ship is already written, and
  // throwing would leave the editor believing it had unsaved work - which is
  // exactly what happened against a server too old to know this route.
  let collision = null, collisionError = null;
  try {
    collision = await saveCollision();
  } catch (e) {
    collisionError = e.message || String(e);
  }
  return { ...(await res.json()), collision, collisionError };
}

/** Write the per-module collision to its own file. */
export async function saveCollision() {
  const body = JSON.stringify({
    generator: "SciFiShip layout tool",
    schema: 1,
    savedAt: new Date().toISOString(),
    units: "metres",
    note: "Collision authored per kit module, in each module's local space."
      + " Editor space: the manifest's own collision block is the mirrored,"
      + " runtime-facing copy.",
    moduleShapes: serializeModuleCollision(),
    // What is on the collision staging area - read live if it is open, and from
    // the last time it closed if it is not. Purely an authoring convenience,
    // and no part of the ship.
    stageLayout: (hooks.stageLayoutNow?.() || state.stageLayout)
      .map((s) => ({ module: s.module, position: [...s.position] })),
    // and where you were standing on the bench, so a reload puts you back
    stageView: hooks.stageViewpoint?.() || null,
  }, null, 2);
  const res = await fetch("/api/collision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

/** Write a recovery copy, apart from the ship you last chose to save. */
export async function saveAutosave() {
  if (state.collisionMode) hooks.harvestStage?.();
  const body = JSON.stringify({
    ...buildManifest(),
    generator: "SciFiShip layout tool (auto-save)",
    autoSaved: true,
  }, null, 2);
  const res = await fetch("/api/autosave", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

/**
 * Read the per-module collision back from its own file.
 *
 * The file wins over whatever the ship manifest carries: it is the one you
 * shipped with the kit, and the point of having it is that a new ship starts
 * with every hull already fitted.
 */
export async function loadCollision() {
  const res = await fetch("/api/collision");
  if (!res.ok) return null;
  const data = await res.json();
  const shapes = data?.moduleShapes || data?.moduleCollision;
  if (!shapes || !Object.keys(shapes).length) return null;
  loadModuleCollision(shapes, data.stageLayout);
  hooks.setStageViewpoint?.(data.stageView);
  emit("colliders");
  return shapes;
}

export async function loadLayout(name) {
  const url = name ? `/api/layout?name=${encodeURIComponent(name)}` : "/api/layout";
  return whileBusy("loading ship…", async () => {
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    if (!data.instances) return null;    // a Blender-era manifest, not reloadable
    await deserialize(data);
    // The shipped collision file wins over whatever this ship's manifest
    // carries, so a new ship built from the same kit starts fully fitted.
    await loadCollision();
    applyView(data.view);                // older manifests simply have none
    applyEnvironment(data.environment, data.editorEnvironment);
    // Start positions moved to behaviours, so an old block is dropped rather
    // than silently kept and re-saved. Said out loud, because losing where the
    // player starts without being told is exactly the kind of thing you notice
    // three sessions later.
    if (data.spawns && Object.keys(data.spawns).length) {
      console.warn("legacy spawns dropped:", data.spawns);
      data.legacySpawns = data.spawns;
    }
    return data;
  });
}

/**
 * Export one ship.glb with every chunk as a named parent node, matching the
 * runtime contract the portal renderer already expects.
 *
 * An element's **parent node** carries its name; its primitives are numbered
 * off it as `<name>_primitive0`, `_primitive1`… The editor never deals in
 * primitives, and neither does the manifest: everything - behaviours, door
 * leaves - is keyed by node name, and the runtime walks down to the primitives
 * itself.
 *
 * Names are deliberately **not** unique. Naming six crates "crate" gives six
 * nodes called "crate", which is exactly how one behaviour entry comes to
 * govern all six. Unnamed elements fall back to their id.
 *
 * The renames last only for the export and are put back afterwards.
 */
export async function exportGlb() {
  return withVeilSuspended(() => exportGlbInner());
}

async function exportGlbInner() {
  const holders = new Map();
  const restore = [];
  const renamed = [];
  const tagged = [];

  for (const id of state.chunks) {
    const t = new TransformNode(`CHUNK_${id}`, state.scene);
    holders.set(id, t);
  }
  for (const p of shipPlacements()) {
    restore.push([p.node, p.node.parent]);
    p.node.parent = holders.get(p.chunk) || null;

    const name = nodeNameOf(p);
    renamed.push([p.node, p.node.name]);
    p.node.name = name;
    p.node.getChildMeshes().forEach((m, i) => {
      renamed.push([m, m.name]);
      m.name = `${name}_primitive${i}`;
    });

    // Who this node is, written into the glTF node's `extras`.
    //
    // A name is not an identity: two placements may carry the same one - this
    // ship has `crate4` twice - so a runtime that matched by name could not
    // tell which body belongs to which mesh, and taking one mesh out of the
    // scene would be a guess as to which Havok shape to drop with it.
    //
    // `id` is the manifest's own key and unique by construction; `module` and
    // `chunk` ride along so a mesh resolves to its hull in `moduleCollision`
    // without a lookup table. Babylon's exporter takes `metadata.gltf.extras`
    // and both Babylon and Babylon-Lite hand it back at the same address.
    tagged.push([p.node, p.node.metadata]);
    p.node.metadata = {
      ...(p.node.metadata || {}),
      gltf: { extras: { id: p.id, module: p.module, chunk: p.chunk } },
    };
  }

  const exportable = new Set();
  for (const t of holders.values()) exportable.add(t);
  for (const p of shipPlacements()) {
    exportable.add(p.node);
    for (const m of p.node.getChildMeshes()) exportable.add(m);
  }

  try {
    const glb = await withAuthoredMaterials(() =>
      BABYLON.GLTF2Export.GLBAsync(state.scene, "ship", {
        shouldExportNode: (node) => exportable.has(node),
        exportWithoutWaitingForScene: false,
      }));
    const blob = glb.glTFFiles["ship.glb"];
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "model/gltf-binary" },
      body: blob,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  } finally {
    for (const [node, parent] of restore) node.parent = parent;
    for (const [node, name] of renamed) node.name = name;
    // Put the metadata back exactly, undefined included: the editor's own
    // picking reads this object, and a stray `gltf` key left on it would be a
    // quiet lie about what the node is.
    for (const [node, meta] of tagged) node.metadata = meta;
    for (const t of holders.values()) t.dispose();
  }
}

function r(a) { return a.map((v) => Math.round(v * 1e4) / 1e4); }
