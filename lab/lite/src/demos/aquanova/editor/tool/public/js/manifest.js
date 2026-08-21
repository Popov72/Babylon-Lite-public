// Manifest I/O and glb export.
//
// ship_manifest.json is the source of truth: it stores every placement as a
// module id plus a transform, so the editor can be restored exactly. ship.glb
// is a derived runtime artefact and is never read back.

import {
  state, serialize, deserialize, worldBounds, withAuthoredMaterials, shipPlacements,
  loadModuleCollision, serializeModuleCollision, emit, hooks,
  serializeView, applyView, serializeEnvironment, serializeEditorEnvironment,
  serializeEditorPrefs, applyEditorPrefs,
  applyEnvironment, whileBusy, withVeilSuspended,
  isVeilClone, isGizmoMesh, isRuntimeStandIn, SKYBOX_CHUNK,
  environmentProbeIds, environmentProbeOf, writeBehaviorExtras, nodeNameOf, pushUndo,
} from "./editor.js";
import { portalOf } from "./markers.js";

const { TransformNode, Vector3, Quaternion, Matrix } = BABYLON;

export const SCHEMA = 3;

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
 *
 * It lives in editor.js, next to the behaviour store that keys off it, and is
 * re-exported here because this is the module the rest of the tool has always
 * asked for it by.
 */
export { nodeNameOf };

/**
 * The `behaviors` library and the `entities` that carry them.
 *
 * Definitions are written through untouched. The authoring metadata controls
 * which fields the UI edits, while pass-through serialization preserves unknown
 * legacy fields instead of quietly deleting them. Applied parameters use
 * writeBehaviorExtras for the same reason - one helper, shared with the undo
 * snapshot, so the two can never disagree about what an assignment may carry.
 *
 * `linked` is omitted when empty rather than written as `[]` - the absence is
 * what "this one stands alone" means - and entries pointing at a behaviour that
 * no longer exists are dropped, since the runtime would only ignore them.
 *
 */
function serializeBehaviors() {
  return Object.fromEntries(
    [...state.behaviors].map(([k, v]) => [k, JSON.parse(JSON.stringify(v))]));
}

function serializeEntities() {
  const out = {};
  const nodes = [...state.entities.keys()];
  for (const node of nodes) {
    const kept = (state.entities.get(node) || [])
      .filter((b) => state.behaviors.has(b.name))
      .map((b) => ({ name: b.name, ...writeBehaviorExtras(b) }));
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
 * The shapes authored on each kit module, in the module's own local space,
 * **mirrored to glTF space** like everything else the runtime reads.
 *
 * What the runtime instances: for each placement, look its module up here and
 * compose with the placement's transform. One Havok shape per module, reused
 * across every body that needs it.
 *
 * Two things to get right, both silent when wrong:
 *
 * - Take the placement's transform from the **loaded glTF node**, not from
 *   `instances`. `instances` is editor space - it is the tool's own reload
 *   source - and composing a glTF-space hull onto it puts the collider on the
 *   wrong side of the prop, where it still looks perfectly plausible. The glTF
 *   node's `extras` carries `id`, `module` and `chunk` for exactly this.
 * - Converting a *local* transform between the two spaces needs the **rotation
 *   as well as the centre** - `[-x,y,z]` for the point and `[-x,y,z,-w]` for
 *   the quaternion. Centre only leaves a turned box mirrored, which looks
 *   right on anything symmetrical and wrong on everything else.
 *
 * The manifest's `space` block names every field's space, so this can be
 * asserted rather than remembered.
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
      meshCount += m.node.getChildMeshes().filter((x) => !isVeilClone(x) && !isRuntimeStandIn(x)).length;
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

  const environmentProbes = environmentProbeIds().map((id) => {
    const probe = environmentProbeOf(id);
    return {
      id,
      boxPosition: toGltf(probe.boxPosition),
      boxSize: r(probe.boxSize),
      capturePosition: toGltf(probe.capturePosition),
      angle: -probe.angle,
      // The volume the runtime blends this probe over, which is a different
      // question from the volume it projects onto: the box above is the room's
      // walls, these two are where the cubemap starts and stops being the one
      // to use. Sizes are full extents, and the inner box shares the outer
      // one's centre.
      influenceBoxPosition: toGltf(probe.influenceBoxPosition),
      influenceBoxSize: r(probe.influenceBoxSize),
      influenceInnerBoxSize: r(probe.influenceInnerBoxSize),
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
      enabled: m.enabled !== false,
      // The far side can be seen but not reached - a window onto space rather
      // than a doorway. Written on the door only for now; collision generation
      // will read it when that work happens. Default false, so every manifest
      // written before this reads back as an ordinary doorway.
      //
      // A skybox side implies it: `normalizeDoorSides` already settles the pair
      // whenever a door is made or loaded, and this restates it at the point
      // the contract is actually written, so a hand-edited manifest cannot
      // describe a window onto space you could walk out of.
      sealed: !!m.sealed || b === SKYBOX_CHUNK,
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
    // The skybox is a destination, not a room: it is deliberately absent from
    // `chunks`, so the guards below leave it out of the reverse direction. The
    // forward edge is kept - a portal renderer standing in the room has to know
    // this opening leads to space and not to another chunk - and the door's
    // `sealed` flag says nobody can walk it.
    if (adjacency[p.chunkA]) adjacency[p.chunkA].push({ to: p.chunkB, portal: p.id });
    if (adjacency[p.chunkB]) adjacency[p.chunkB].push({ to: p.chunkA, portal: p.id });
  }

  return {
    generator: "SciFiShip layout tool",
    schema: SCHEMA,
    savedAt: new Date().toISOString(),
    units: "metres",
    up: "Y (glTF)",
    // Which half of this file is in which space, said out loud and by name.
    //
    // The manifest has two halves. Runtime-facing fields are written in **glTF
    // space**, matching ship.glb: an element at editor [7,3,5] is at [-7,3,5]
    // in both. Tool-facing fields stay in **editor space**, because they exist
    // to rebuild the editor and never leave it - `instances` is what the tool
    // reloads the ship from.
    //
    // That split is deliberate but it was only ever written in a source
    // comment, which is no use to a runtime: compose a glTF-space module hull
    // onto an editor-space instance and the collider lands on the *wrong side*
    // of the prop, still looking perfectly plausible. Naming the fields here
    // lets a reader assert instead of remember.
    space: {
      gltf: ["chunks[].aabb", "environmentProbes[].boxPosition",
        "environmentProbes[].capturePosition",
        "environmentProbes[].angle",
        "environmentProbes[].influenceBoxPosition",
        "collision", "moduleCollision", "portals", "doors"],
      editor: ["instances", "markers", "colliders", "lights", "moduleShapes", "stageLayout", "view"],
      none: ["generator", "schema", "savedAt", "units", "up", "grid", "config", "kits",
        "activeChunk", "fluidSim", "behaviors", "entities", "environment",
        "editorEnvironment", "editorPrefs", "adjacency", "space",
        "environmentProbes[].boxSize",
        "environmentProbes[].influenceBoxSize",
        "environmentProbes[].influenceInnerBoxSize"],
      convert: {
        note: "editor <-> glTF is its own inverse: negate X.",
        point: "[-x, y, z]",
        yaw: "-angle",
        quaternion: "[-x, y, z, -w] — mirroring flips the handedness of the turn as well",
        aabb: "min.x and max.x swap as well as negate",
        // The one that bites. moduleCollision is a *local* transform written in
        // glTF space, so converting it back needs the rotation as well as the
        // centre - a centre-only conversion leaves a turned box mirrored, and
        // looks right on anything symmetrical.
        moduleCollision: "local to its module, in glTF space:"
          + " convert centre AND rotation before composing onto a placement",
      },
    },
    grid: { tile: 4 },
    // Ship-wide authoring constants - see state.config. They have to travel
    // with the ship: refitting collision with a different shell thickness
    // produces different geometry, so a manifest that dropped them would come
    // back looking identical and then fit differently the next time you
    // pressed Generate.
    config: layout.config,
    // Which kits the ship was laid out from, and where they were read. A module
    // id is `<kit>/<category>/<name>`, so recording the kit list makes an id in
    // `instances` resolvable by anyone who did not build the ship. `source` is
    // provenance only - it says nothing about where a later session should read
    // them from, which is that session's own config.
    kits: state.kits?.length ? { source: state.kitsSource || null, base: state.kitsBase || null, folders: state.kits } : null,
    chunks,
    environmentProbes,
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
    // Authored lights, each riding a placement. Editor space and local to the
    // owner, like `moduleShapes`: this is what the tool reloads from. What the
    // RUNTIME reads is the exported TransformNode's own extras, not this - so
    // the two never have to agree about handedness.
    lights: layout.lights,
    // A room's own one-off shapes, in glTF space, in the form Havok's
    // constructors take. Grouped by chunk because collision is streamed per
    // room, and a flat list would make every room filter the whole ship.
    //
    // Sizes are the *effective* ones, not the editor's raw scale: a unit box
    // scaled 4x2x0.2 is written as half-extents, and a capsule as the radius
    // and the two endpoints Havok's constructor actually takes.
    //
    // A module's hull is NOT expanded into here. It is written once below and
    // instanced by the runtime, which already knows every placement's module
    // and chunk - from the glTF node's own `extras`, or from `instances`.
    // Expanding it as well was pure duplication of the sort that grows:
    // placements x shapes, against modules x shapes.
    //
    // Take the *transform* from the loaded glTF node, not from `instances`:
    // the node is in the same space as this block, and `instances` is not.
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
    // and the editor's own view preferences, which they must not read either
    editorPrefs: serializeEditorPrefs(),
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
  if (state.mode === "collision") hooks.harvestStage?.();
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
    if (!data.instances) return null;    // a pre-schema-1 manifest, not reloadable
    await deserialize(data);
    // The shipped collision file wins over whatever this ship's manifest
    // carries, so a new ship built from the same kit starts fully fitted.
    await loadCollision();
    applyView(data.view);                // older manifests simply have none
    applyEnvironment(data.environment, data.editorEnvironment);
    applyEditorPrefs(data.editorPrefs);   // older manifests simply have none
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
 * Copy what is on disk into the demo, by running `sync-ship.ts` server-side.
 *
 * This publishes the **saved** ship: the manifest, collision hulls, probes and
 * glb the export folder holds right now. Unsaved edits in the viewport are not
 * part of it, which is why the caller says so before starting.
 *
 * The script's own success is `ok`; a `false` with no `output` means the server
 * could not run it at all. Either way the whole transcript comes back, because
 * a publish that failed halfway is only diagnosable from its log.
 */
export async function syncShip(optimize) {
  const res = await fetch("/api/sync-ship", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ optimize: !!optimize }),
  });
  let data = null;
  try { data = await res.json(); } catch { /* answered with something that is not JSON */ }
  if (!data) return { ok: false, error: `server answered ${res.status}` };
  return data;
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
    // Gizmos are skipped, not just left unexported: they hang off the same node
    // as the module's parts, so counting them here would shift every primitive
    // number after the one they sit next to.
    artMeshes(p.node).forEach((m, i) => {
      renamed.push([m, m.name]);
      m.name = `${name}_primitive${i}`;
    });
    for (const animationNode of p.node._shipAnimationNodes ?? []) {
      renamed.push([animationNode, animationNode.name]);
      animationNode.name = `${name}_${animationNode._shipAnimationSourceName}`;
    }

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

  // Lights ride out as bare nodes carrying their whole record.
  //
  // There is nothing to draw: the runtime builds a Babylon light from `extras`,
  // so what has to survive the trip is the transform and the record, not
  // geometry. Being a child of the element it rides means the glTF hierarchy
  // carries the offset for free, in exactly the space it was authored in.
  //
  // The gizmo meshes hanging off the same node are left out of `exportable`, so
  // the node reaches the file with no children at all - which the exporter is
  // fine with: a node with a name, a transform and extras is still a node.
  const lights = hooks.lightsForExport();
  for (const l of lights) {
    renamed.push([l.node, l.node.name]);
    l.node.name = `LIGHT_${l.id}`;
    tagged.push([l.node, l.node.metadata]);
    l.node.metadata = {
      ...(l.node.metadata || {}),
      gltf: { extras: l.extras },
    };
  }

  const exportable = new Set();
  for (const t of holders.values()) exportable.add(t);
  for (const p of shipPlacements()) {
    exportable.add(p.node);
    for (const m of artMeshes(p.node)) exportable.add(m);
    for (const animationNode of p.node._shipAnimationNodes ?? []) exportable.add(animationNode);
  }
  for (const l of lights) exportable.add(l.node);

  // The exporter reads skins and animation groups from the whole scene rather
  // than through `shouldExportNode`. Hidden kit prototypes must stay out: their
  // joints are disabled, not exportable, and shared by every placement.
  // Animated placements instead own private skeletons, targets, and groups;
  // expose only those for the duration so each exported clip targets its own
  // entity and static modules remain ordinary hardware instances.
  const skeletons = state.scene.skeletons;
  const hiddenSkeletons = skeletons.splice(0, skeletons.length);
  skeletons.push(...shipPlacements().flatMap((placement) => placement.node._shipSkeletons ?? []));
  const animationGroups = state.scene.animationGroups;
  const hiddenAnimationGroups = animationGroups.splice(0, animationGroups.length);
  animationGroups.push(...shipPlacements().flatMap((placement) => placement.node._shipAnimationGroups ?? []));

  // The exporter writes each node's transform as it stands. A ship exported
  // while the behaviour preview is running would record a fan halfway round as
  // its rest pose, so playback is held at frame 0 for the duration.
  const resumeBehaviorAnimations = hooks.pauseBehaviorAnimations();

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
    animationGroups.splice(0, animationGroups.length, ...hiddenAnimationGroups);
    skeletons.splice(0, skeletons.length, ...hiddenSkeletons);
    for (const [node, parent] of restore) node.parent = parent;
    for (const [node, name] of renamed) node.name = name;
    // Put the metadata back exactly, undefined included: the editor's own
    // picking reads this object, and a stray `gltf` key left on it would be a
    // quiet lie about what the node is.
    for (const [node, meta] of tagged) node.metadata = meta;
    for (const t of holders.values()) t.dispose();
    resumeBehaviorAnimations();
  }
}

function r(a) { return a.map((v) => Math.round(v * 1e4) / 1e4); }

/** A placement's own primitives - its module's parts, and nothing else. */
function artMeshes(node) {
  return node.getChildMeshes().filter((m) => !isGizmoMesh(m) && !isRuntimeStandIn(m));
}

/**
 * Every name an entity entry can legitimately be keyed by.
 *
 * The runtime resolves an entity by looking its key up among the nodes of
 * `ship.glb`, so this is that set - built by the same rules `exportGlbInner`
 * renames by, and deliberately sitting next to them so the two cannot drift:
 * placements under their node name, their primitives under
 * `<name>_primitive<i>`, their animation nodes under `<name>_<clip>`, the chunk
 * holders under the chunk id and the lights under `LIGHT_<id>`. Door ids join
 * them: a door is not in the glb, but its id is a key the manifest carries.
 *
 * The derived names matter as much as the plain ones. A hand-written entry on
 * `Fan_primitive0` is a perfectly good way to give one part of a module a
 * behaviour of its own, and anything that only knew about placement names would
 * read it as rubbish and throw it away.
 *
 * **Every** placement, not only the ship's: something on the bench has not been
 * exported yet, but it exists, and work in progress is not stale data.
 */
export function liveEntityNames() {
  const live = new Set(state.chunks);
  for (const p of state.placements.values()) {
    for (const name of namesOfPlacement(p)) live.add(name);
  }
  for (const m of state.markers.values()) if (m.type === "door") live.add(m.id);
  for (const l of state.lights.values()) live.add(`LIGHT_${l.id}`);
  return live;
}

/**
 * The same names again, filed under the room each one is in.
 *
 * What the behaviour panel's entity pickers narrow by, so it has to answer for
 * exactly the set above - a name the filter cannot place would vanish from
 * every room and only reappear under "anywhere on the ship". Beside its twin
 * for the same reason that one sits beside the exporter.
 *
 * A door is filed under **both** rooms it joins: either side is a fair place to
 * reach one from. Each chunk is filed under itself, because the holder is
 * addressable too, and a lamp under the room of the placement it hangs on.
 */
export function liveEntitiesByChunk() {
  const out = {};
  const add = (chunk, name) => {
    if (!chunk || !name) return;
    (out[chunk] ??= new Set()).add(name);
  };
  for (const chunk of state.chunks) add(chunk, chunk);
  for (const p of state.placements.values()) {
    for (const name of namesOfPlacement(p)) add(p.chunk, name);
  }
  for (const m of state.markers.values()) {
    if (m.type !== "door") continue;
    add(m.chunkA, m.id);
    add(m.chunkB, m.id);
  }
  for (const l of state.lights.values()) {
    add(state.placements.get(l.owner)?.chunk, `LIGHT_${l.id}`);
  }
  for (const [chunk, names] of Object.entries(out)) {
    out[chunk] = [...names].sort((a, b) => a.localeCompare(b));
  }
  return out;
}

function namesOfPlacement(p) {
  const name = nodeNameOf(p);
  const names = [name];
  artMeshes(p.node).forEach((m, i) => names.push(`${name}_primitive${i}`));
  for (const a of p.node._shipAnimationNodes ?? []) {
    names.push(`${name}_${a._shipAnimationSourceName}`);
  }
  return names;
}

/**
 * Drop the behaviour entries no node answers to any more.
 *
 * Behaviours outlive the element that carried them **by design**: deleting the
 * last crate to put a better one down must not throw away how crates behave,
 * and renaming leaves the old name's entry alone in case something else is
 * meant to pick it up. That is the right rule while you are working, and the
 * wrong one for a file - the entries pile up under names that were typos, or
 * were renamed years ago, and the runtime looks every one of them up and finds
 * nothing.
 *
 * So the tidy happens at the one moment the working state becomes the ship:
 * an explicit **Save**. Not on auto-save, which is a recovery copy and must
 * never be the thing that destroys what you were hoping to recover, and not on
 * every edit, which would make the working rule above impossible.
 *
 * Returns what it dropped, so the save can name them - and pushes an undo entry
 * first, and only when there is something to drop, so an entry that was still
 * wanted is one Ctrl+Z away rather than gone.
 */
export function pruneOrphanEntities() {
  const live = liveEntityNames();
  const dropped = [...state.entities.keys()]
    .filter((k) => !live.has(k))
    .sort((a, b) => a.localeCompare(b));
  if (!dropped.length) return dropped;
  pushUndo();
  for (const k of dropped) state.entities.delete(k);
  emit("behaviors");
  return dropped;
}
