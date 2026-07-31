// Manifest I/O and glb export.
//
// ship_manifest.json is the source of truth: it stores every placement as a
// module id plus a transform, so the editor can be restored exactly. ship.glb
// is a derived runtime artefact and is never read back.

import {
  state, serialize, deserialize, worldBounds, withAuthoredMaterials,
  serializeView, applyView, serializeEnvironment, serializeEditorEnvironment,
  applyEnvironment, whileBusy, withVeilSuspended, isVeilClone,
} from "./editor.js";
import { portalOf } from "./markers.js";

const { TransformNode, Vector3 } = BABYLON;

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

export function buildManifest() {
  const layout = serialize();

  // Every derived figure below - chunk bounds, portal corners - is read off a
  // world matrix, and Babylon only refreshes those at render time. An export
  // fired straight after a scale (a wheel notch, an inspector keystroke) would
  // otherwise measure the ship as it was one frame ago, and the door's own
  // width, taken from `scaling` directly, would disagree with its portal.
  for (const p of state.placements.values()) p.node.computeWorldMatrix(true);
  for (const m of state.markers.values()) m.node.computeWorldMatrix(true);

  const boxes = [];
  const chunks = state.chunks.map((id) => {
    const members = [...state.placements.values()].filter((p) => p.chunk === id);
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
    kitDir: state.kitDir || null,
    chunks,
    // "node" is what the element is called in ship.glb - its own name, or its
    // id when it has none - and is the only handle the runtime needs. "id" is
    // the editor's, and is what the tool reloads from.
    instances: layout.instances.map((i) => ({
      ...i, node: nodeNameOf(state.placements.get(i.id)) || i.id,
    })),
    markers: layout.markers,
    activeChunk: layout.activeChunk,
    // The sims a liquefied element may use. Seeded from the tool's config.json,
    // but a loaded ship's own list wins - see restoreFrom - so the two cannot
    // drift apart behind your back.
    fluidSim: [...state.fluidSim],
    behaviors: serializeBehaviors(),
    entities: serializeEntities(),
    // where you were standing when you saved, so a reload puts you back
    view: serializeView(),
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
  return res.json();
}

export async function loadLayout(name) {
  const url = name ? `/api/layout?name=${encodeURIComponent(name)}` : "/api/layout";
  return whileBusy("loading ship…", async () => {
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    if (!data.instances) return null;    // a Blender-era manifest, not reloadable
    await deserialize(data);
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

  for (const id of state.chunks) {
    const t = new TransformNode(`CHUNK_${id}`, state.scene);
    holders.set(id, t);
  }
  for (const p of state.placements.values()) {
    restore.push([p.node, p.node.parent]);
    p.node.parent = holders.get(p.chunk) || null;

    const name = nodeNameOf(p);
    renamed.push([p.node, p.node.name]);
    p.node.name = name;
    p.node.getChildMeshes().forEach((m, i) => {
      renamed.push([m, m.name]);
      m.name = `${name}_primitive${i}`;
    });
  }

  const exportable = new Set();
  for (const t of holders.values()) exportable.add(t);
  for (const p of state.placements.values()) {
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
    for (const t of holders.values()) t.dispose();
  }
}

function r(a) { return a.map((v) => Math.round(v * 1e4) / 1e4); }
