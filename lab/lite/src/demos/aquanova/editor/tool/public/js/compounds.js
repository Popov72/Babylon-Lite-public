// The compound editor: one reusable object built out of several kit modules.
//
// The ship is full of pieces that are never placed alone. A ceiling panel wants
// its lamp, a wall wants its trim and its cable run, a doorway wants the frame
// on both sides. Placing each part separately is three or four operations for
// something that is conceptually one thing, repeated a few hundred times across
// a ship - and every repetition is a chance to get an offset slightly wrong.
//
// A compound is that group, authored once and then dropped as a unit.
//
// **It is a recipe, not a mesh.** Saving one records what its members are and
// where they sit; placing one expands that into ordinary placements which share
// a `group` id. Three consequences, all of them the reason for the choice:
//
// - A lamp is a `state.lights` entry rather than geometry, so a baked .glb
//   could never have carried one - and "a wall with its lamp" is the case the
//   feature exists for.
// - Each member keeps its own authored collision hull, which baking would have
//   flattened into one anonymous shape.
// - Deleting one component, or breaking the whole thing apart, needs no rebuild
//   path: the parts were never anything other than ordinary placements.
//
// The bench works the way the collision staging area does, for the same reason:
// its contents are real placements carrying `stage: true`, so selection, the
// gizmo, hiding, dragging, `Ctrl+D` and the inspector all work on them
// unchanged - and `shipPlacements()` filters them out at the two boundaries
// that walk every placement, so they can never reach the ship.

import {
  state, emit, hooks, select, placeAt, removePlacement, eulerOf,
  applyVisibility, resetModeHistory, serializeView, applyView,
  COMPOUND_CHUNK, pushUndo, pushUndoFor, shipPlacements,
} from "./editor.js";
import { lightsOf, addLight, lightOffset, lightRotation } from "./lights.js";
import { getCatalogue, getModule, reloadCatalogue } from "./kit.js";
import { warm, forgetCached } from "./thumbs.js";

const { Vector3, Quaternion, Matrix, TransformNode } = BABYLON;

/**
 * Where the bench's contents live between visits.
 *
 * A workbench keeps what you left on it - the same decision the collision area
 * made, and for the same reason: stepping out to look at the ship and coming
 * back to a blank stage is not a fresh start, it is lost work. It is kept in
 * `localStorage` rather than in the ship, because a half-built compound is no
 * more ship data than the palette's kit choice is: it belongs to this browser,
 * and writing it into the manifest would put a workbench in everybody's ship.
 */
const BENCH_STORE = "compoundBench";

const round = (a) => a.map((v) => Math.round(v * 1e4) / 1e4);

// -------------------------------------------------------------- the bench

/** The placements currently on the compound bench. */
export function benchMembers() {
  return [...state.placements.values()].filter((p) => p.chunk === COMPOUND_CHUNK);
}

/**
 * The bench as plain data - its own undo snapshot, and what is remembered
 * between visits.
 *
 * Lights are part of a member rather than a separate list: on the bench a lamp
 * only ever exists as something fitted to one of the pieces, and recording them
 * together means a restore cannot put a light back before its owner exists.
 */
export function serializeBench() {
  return {
    members: benchMembers().map((e) => ({
      id: e.id,
      module: e.module,
      name: e.name || "",
      position: round(e.node.position.asArray()),
      rotation: round(eulerOf(e.node)),
      scale: round(e.node.scaling.asArray()),
      lights: lightsOf(e.id).map((l) => ({
        offset: lightOffset(l),
        rotation: lightRotation(l),
        runtime: { ...l.runtime, color: round(l.runtime.color) },
      })),
    })),
    editing: editing || "",
  };
}

/** Put the bench back exactly as a snapshot found it. */
export async function restoreBench(data) {
  for (const e of benchMembers()) removePlacement(e.id);
  state.selection = [];
  for (const m of data?.members || []) {
    if (!m?.module || !getModule(m.module)) continue;
    const entry = await placeAt(m.module, Vector3.FromArray(m.position || [0, 0, 0]), {
      stage: true, stageChunk: COMPOUND_CHUNK, silent: true, id: m.id,
      rotation: m.rotation, scale: m.scale, name: m.name,
      // The snapshot carries the lamps the member actually had, which is not
      // the same list as the kit's defaults the moment one has been deleted or
      // moved. Seeding would put the deleted one back on every undo.
      noLights: true,
    });
    for (const l of m.lights || []) addLight(entry.id, { ...l, silent: true });
  }
  editing = data?.editing || "";
  applyVisibility();
  emit("placements");
  emit("lights");
  emit("selection");
  emit("compound");
}

/** Which saved compound the bench was opened from, "" for a new one. */
let editing = "";
export function editingCompound() { return editing; }

let shipView = null;
let benchView = null;

export function isCompoundMode() { return state.mode === "compound"; }

/**
 * Open the bench, restoring whatever was on it when it was last closed.
 *
 * Refuses while the collision area is open rather than swapping straight from
 * one bench to the other: each closes by harvesting what is on it, and running
 * two harvests into each other is exactly the kind of thing that works until
 * the day it silently does not.
 */
export async function enterCompoundMode() {
  if (state.mode !== "ship") return false;
  // Each side keeps its own viewpoint. Coming back to the ship pointing at a
  // bench, or to the bench pointing across the ship, means finding your
  // bearings again on every switch.
  shipView = serializeView();
  state.mode = "compound";
  select([]);
  applyVisibility();
  emit("mode");

  await restoreBench(readBench());
  resetModeHistory("compound");   // the bench's history starts here, not in the ship's
  const viewRestored = !!benchView;
  if (benchView) applyView(benchView);
  applyVisibility();
  emit("placements");
  emit("compound");
  return { viewRestored };
}

/** Close it, keeping everything that is on it for next time. */
export function exitCompoundMode() {
  if (state.mode !== "compound") return false;
  benchView = serializeView();
  writeBench(serializeBench());
  for (const e of benchMembers()) removePlacement(e.id);
  state.mode = "ship";
  select([]);
  resetModeHistory("compound");   // the bench's history does not outlive the bench
  if (shipView) applyView(shipView);
  applyVisibility();
  emit("mode");
  emit("placements");
  emit("lights");
  emit("compound");
  return true;
}

/** Clear the bench and start again. Undoable, like any other bench edit. */
export async function newCompound() {
  if (state.mode !== "compound") return false;
  pushUndo();
  for (const e of benchMembers()) removePlacement(e.id);
  editing = "";
  select([]);
  applyVisibility();
  emit("placements");
  emit("lights");
  emit("compound");
  return true;
}

/** Write the bench to its store without leaving it - see the ship save. */
export function persistBench() {
  if (state.mode !== "compound") return false;
  writeBench(serializeBench());
  return true;
}

function readBench() {
  try {
    const raw = localStorage.getItem(BENCH_STORE);
    return raw ? JSON.parse(raw) : { members: [] };
  } catch {
    return { members: [] };
  }
}

function writeBench(data) {
  try {
    localStorage.setItem(BENCH_STORE, JSON.stringify(data));
  } catch { /* a full or blocked store is not worth failing a mode switch over */ }
}

// ------------------------------------------------------------ definitions

/** Every saved compound, read off the catalogue the palette is showing. */
export function compoundTiles() {
  const out = [];
  for (const c of getCatalogue()?.categories || []) {
    for (const m of c.modules) if (m.compound) out.push(m);
  }
  return out;
}

export function compoundTile(name) {
  return compoundTiles().find((m) => m.name === name) || null;
}

/**
 * A definition's members, measured from the first one.
 *
 * The bench writes them that way now, but compounds saved before the anchor
 * moved measured from the bounding box's bottom centre instead - which is what
 * made a compound land on fractional coordinates. Shifting the whole list so
 * its first member sits at the origin changes nothing about the shape and
 * brings those older recipes under the same rule, rather than leaving two kinds
 * of definition for every consumer to know about.
 */
export function definitionMembers(tile) {
  const list = tile?.members || [];
  const lead = list[0]?.position || [0, 0, 0];
  if (!lead[0] && !lead[1] && !lead[2]) return list;
  return list.map((m) => ({
    ...m,
    position: [
      (m.position?.[0] || 0) - lead[0],
      (m.position?.[1] || 0) - lead[1],
      (m.position?.[2] || 0) - lead[2],
    ],
  }));
}

/**
 * The bench as a definition, with its members measured from a shared origin.
 *
 * **The origin is the first member's own origin.** The bench lays a compound
 * out around whichever piece you started with, and that piece is the thing you
 * are really placing - a wall with a lamp on it is a wall, and it should land
 * where a wall lands.
 *
 * It began as the bottom centre of the bench's bounding box, on the reasoning
 * that a compound should drop onto the build plane the way a single module
 * does. It does not: a module's origin is on the grid because the artist put it
 * there, whereas a bounding-box centre lands wherever the arithmetic of two
 * differently sized pieces happens to put it. A 4 m wall beside a 12 m platform
 * centres at 5, so every member came out offset by a half-metre and a compound
 * dropped with `Move` at 1 m produced positions like `-0.5` and `3.5`. Anchoring
 * on a real module's origin instead means integer offsets in, integer positions
 * out - and gives the group rotation a pivot that is a real thing rather than
 * an average.
 *
 * "First" is the order the pieces were put on the bench, which is also the
 * order they are listed in and the order they come back in.
 */
export function benchDefinition(name, kit, category) {
  const members = benchMembers();
  if (!members.length) return null;
  const origin = members[0].node.position.clone();

  return {
    name,
    kit,
    category,
    savedAt: new Date().toISOString(),
    members: members.map((e) => ({
      module: e.module,
      ...(e.name ? { name: e.name } : {}),
      position: round(e.node.position.subtract(origin).asArray()),
      rotation: round(eulerOf(e.node)),
      scale: round(e.node.scaling.asArray()),
      lights: lightsOf(e.id).map((l) => ({
        offset: lightOffset(l),
        rotation: lightRotation(l),
        runtime: { ...l.runtime, color: round(l.runtime.color) },
      })),
    })),
  };
}

/** POST the whole list back, then re-read the catalogue it feeds. */
async function writeDefinitions(list) {
  const res = await fetch("/api/compounds", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ compounds: list }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body.ok) {
    throw new Error(body.error || `POST /api/compounds → ${res.status}`);
  }
  await reloadCatalogue();
  return body;
}

/** The saved definitions, straight from the server rather than the catalogue. */
async function readDefinitions() {
  const res = await fetch("/api/compounds");
  if (!res.ok) throw new Error(`GET /api/compounds → ${res.status}`);
  const body = await res.json();
  return Array.isArray(body?.compounds) ? body.compounds : [];
}

/**
 * Save what is on the bench under a name, filed in a kit and category.
 *
 * Re-saving under an existing name replaces it, which is how a compound is
 * edited: open it, change it, save it again. Instances already placed in the
 * ship are *not* touched - they are ordinary placements and have been since the
 * moment they were dropped. That is macro semantics rather than prefab
 * semantics, and it is the honest reading of a recipe: changing a recipe does
 * not reach into meals already cooked.
 */
export async function saveCompound({ name, kit, category, updateInstances = false }) {
  const clean = String(name || "").trim();
  if (!clean) return { ok: false, error: "a compound needs a name" };
  if (/[/\\]/.test(clean)) return { ok: false, error: "a name cannot contain a slash" };
  if (!kit || !category) return { ok: false, error: "choose a kit and a category" };
  const def = benchDefinition(clean, kit, category);
  if (!def) return { ok: false, error: "the bench is empty" };

  const list = await readDefinitions();
  const at = list.findIndex((c) => c.name === clean);
  if (at >= 0) list[at] = def; else list.push(def);
  try {
    await writeDefinitions(list);
  } catch (e) {
    return { ok: false, error: e.message };
  }
  editing = clean;
  // Both pictures now, while the members are certainly loadable: a compound is
  // only ever saved from a bench that is showing it.
  const tile = compoundTile(clean);
  if (tile) await warm(tile).catch(() => {});
  // After the catalogue has the new recipe, never before: the rebuild reads the
  // definition back out of it rather than trusting what was posted.
  const synced = updateInstances ? await syncInstances(clean) : null;
  emit("compound");
  emit("catalogue");
  return {
    ok: true, name: clean, members: def.members.length, replaced: at >= 0,
    instances: synced?.instances || 0,
  };
}

/**
 * Re-save under the name the bench is already editing.
 *
 * The dialog exists to answer three questions - name, kit, category - and after
 * the first save all three are settled. Asking them again on every tweak makes
 * "adjust the lamp, save, look" a five-click loop, and the one place a wrong
 * click there is expensive is the name field, where it silently forks the
 * compound into two.
 */
export async function quickSaveCompound(opts = {}) {
  if (!editing) return { ok: false, error: "this bench has no name yet - use Save as\u2026" };
  const tile = compoundTile(editing);
  if (!tile) return { ok: false, error: `"${editing}" is no longer in the palette - use Save as\u2026` };
  return saveCompound({ name: editing, kit: tile.kit, category: tile.category, ...opts });
}

/**
 * Forget a definition.
 *
 * Placed instances survive, for the same reason re-saving does not rewrite
 * them: they are ordinary placements. Deleting a compound removes the *tile*,
 * not the ship.
 */
export async function deleteCompound(name) {
  const list = await readDefinitions();
  const keep = list.filter((c) => c.name !== name);
  if (keep.length === list.length) return { ok: false, error: "no such compound" };
  try {
    await writeDefinitions(keep);
  } catch (e) {
    return { ok: false, error: e.message };
  }
  forgetCached(`@compound/${name}`);
  if (editing === name) editing = "";
  emit("compound");
  emit("catalogue");
  return { ok: true, name };
}

/**
 * Put a saved compound back on the bench, so it can be edited and re-saved.
 *
 * The bench is cleared first: it is one compound at a time by definition, and
 * merging the two would silently fold whatever was already there into the next
 * save.
 */
export async function editCompound(name) {
  const tile = compoundTile(name);
  if (!tile) return { ok: false, error: "no such compound" };
  if (state.mode !== "compound") await enterCompoundMode();
  pushUndo();
  for (const e of benchMembers()) removePlacement(e.id);
  select([]);
  for (const m of definitionMembers(tile)) {
    if (!getModule(m.module)) continue;
    const entry = await placeAt(m.module, Vector3.FromArray(m.position || [0, 0, 0]), {
      stage: true, stageChunk: COMPOUND_CHUNK, silent: true,
      rotation: m.rotation, scale: m.scale, name: m.name, noLights: true,
    });
    for (const l of m.lights || []) addLight(entry.id, { ...l, silent: true });
  }
  editing = name;
  applyVisibility();
  emit("placements");
  emit("lights");
  emit("compound");
  return { ok: true, name, members: (tile.members || []).length };
}

// ------------------------------------------------------- placed instances

/** The ship's copies of one compound, grouped by instance, in placement order. */
export function compoundInstances(name) {
  const groups = new Map();
  for (const p of shipPlacements()) {
    if (p.compound !== name || !p.group) continue;
    if (!groups.has(p.group)) groups.set(p.group, []);
    groups.get(p.group).push(p);
  }
  return groups;
}

const quatOf = (node) => (node.rotationQuaternion
  || Quaternion.FromEulerVector(node.rotation)).clone();

const eulerQuat = (deg) => Quaternion.FromEulerAngles(
  (deg?.[0] || 0) * Math.PI / 180,
  (deg?.[1] || 0) * Math.PI / 180,
  (deg?.[2] || 0) * Math.PI / 180);

/** A vector turned by a quaternion, without borrowing a node to do it. */
const turn = (v, q) => Vector3.TransformCoordinates(v, q.toRotationMatrix(Matrix.Identity()));

/**
 * Rewrite every copy of a compound in the ship from its current definition.
 *
 * A compound is a recipe, so re-saving one deliberately leaves the copies
 * alone - see saveCompound. That is right for a macro and wrong for the job it
 * is usually doing: fit a lamp to a wall, place forty of them, then find the
 * lamp is a foot too low. This is the way to push that fix through, asked for
 * rather than assumed, because the same semantics that make it useful here make
 * it destructive elsewhere: a copy is rebuilt from the recipe, so anything done
 * to it *as a copy* - a renamed member, a deleted component, a behaviour hung
 * on one of its pieces - does not survive.
 *
 * **Each copy is re-laid around its own anchor**, the first member of the
 * group, which is the same piece the definition measures itself from. The
 * anchor keeps its exact world transform and everything else is rebuilt around
 * it, so a fleet of walls does not shift by a millimetre when the lamp above
 * them moves. The group id is kept as well, so a copy stays the same copy: only
 * its parts are new.
 *
 * One undo step for the lot - pushing an update to forty instances is one
 * decision, and undoing it forty times would be a punishment for making it.
 */
export async function syncInstances(name) {
  const tile = compoundTile(name);
  if (!tile?.members?.length) return { ok: false, error: "no such compound" };
  const groups = compoundInstances(name);
  if (!groups.size) return { ok: true, name, instances: 0, members: 0 };

  const def = definitionMembers(tile);
  const def0 = def[0];
  // The ship's stack, not the bench's: this rewrites ship data, and it is
  // usually done from the bench, where pushUndo() would file it somewhere that
  // is thrown away on the way out.
  pushUndoFor("ship");
  let members = 0;
  for (const [group, list] of groups) {
    const anchor = list[0];
    const chunk = anchor.chunk;
    // The instance's own transform, recovered from the anchor: the definition
    // expresses every member relative to its first one, so undoing that first
    // member's share of the anchor's transform leaves the instance's. Its
    // position needs no such undoing - measured from itself, the first member
    // sits at the origin.
    const rootQuat = quatOf(anchor.node).multiply(Quaternion.Inverse(eulerQuat(def0.rotation)));
    const rootScale = new Vector3(
      anchor.node.scaling.x / (def0.scale?.[0] || 1),
      anchor.node.scaling.y / (def0.scale?.[1] || 1),
      anchor.node.scaling.z / (def0.scale?.[2] || 1));
    const rootPos = anchor.node.position.clone();

    for (const p of list) removePlacement(p.id);

    for (const m of def) {
      if (!getModule(m.module)) continue;
      const offset = Vector3.FromArray(m.position || [0, 0, 0]).multiply(rootScale);
      const pos = rootPos.add(turn(offset, rootQuat));
      const node = new TransformNode("TMP_COMPOUND", state.scene);
      node.rotationQuaternion = rootQuat.multiply(eulerQuat(m.rotation));
      const rotation = eulerOf(node);
      node.dispose();
      const entry = await placeAt(m.module, pos, {
        stage: false, chunk, silent: true, group, compound: name,
        rotation, name: m.name || "", noLights: true,
        scale: [
          rootScale.x * (m.scale?.[0] ?? 1),
          rootScale.y * (m.scale?.[1] ?? 1),
          rootScale.z * (m.scale?.[2] ?? 1)],
      });
      for (const l of m.lights || []) addLight(entry.id, { ...l, silent: true });
      members++;
    }
  }
  // Only the copies that were rebuilt: syncing from the bench must not clear a
  // bench selection, and `removePlacement` leaves the id behind in it.
  const gone = new Set([...groups.values()].flat().map((p) => p.id));
  if (state.selection.some((id) => gone.has(id))) {
    select(state.selection.filter((id) => !gone.has(id)));
  }
  applyVisibility();
  emit("placements");
  emit("lights");
  return { ok: true, name, instances: groups.size, members };
}

// ---------------------------------------------------------------- placing

/**
 * The ghost specs for one compound tile.
 *
 * Everything a drop needs travels with the spec - the lamps included - because
 * the ghost is the only thing that survives between arming a tile and clicking
 * to place it, and re-reading the definition at drop time would use a list that
 * may have been re-saved in between.
 */
export function compoundSpecs(tile) {
  const specs = [];
  for (const m of definitionMembers(tile)) {
    if (!getModule(m.module)) continue;
    specs.push({
      module: m.module,
      offset: Vector3.FromArray(m.position || [0, 0, 0]),
      rotation: m.rotation || [0, 0, 0],
      scaling: m.scale || [1, 1, 1],
      name: m.name || "",
      lights: m.lights || [],
      compound: tile.name,
    });
  }
  return specs;
}

hooks.serializeBench = serializeBench;
hooks.restoreBench = (data) => restoreBench(data);
