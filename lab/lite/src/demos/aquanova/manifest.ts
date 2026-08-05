// Loading and typing of `ship_manifest.json` — the Blender exporter's companion to `ship.glb`.
//
// The manifest is the single source of truth for the ship's structure: chunk AABBs and their glTF
// root nodes, portals, per-entity behaviours and spawns. Anything that needs to know "which room is
// this" or "what does this prop do" resolves it from here rather than inferring it from geometry.

import type { ShipEnvironment } from "../ship-manifest.js";
import type { BehaviorLibrary, Entities } from "./behaviors/index.js";
import type { ShipCollisionShape, ShipInstance } from "./collision-shapes.js";
import { MANIFEST_URL, type Vec3 } from "./constants.js";

export interface Aabb {
    min: Vec3;
    max: Vec3;
}

/** One authored room. `node` is the glTF node every mesh of the chunk hangs under (`CHUNK_<id>`),
 *  which is the authoritative mesh→chunk mapping — a mesh's centre is NOT, since chunk AABBs
 *  overlap. */
export interface ShipChunk {
    id: string;
    node?: string;
    aabb: Aabb;
}

// `behaviors` is a library of named behaviour definitions; `entities` assigns them to MESH NAMES,
// optionally overriding parameters per entity. Mesh names are shared across rooms, so an entity
// applies to every mesh carrying that name. Assignments remain separate so several strongly typed
// behavior instances can coexist on one mesh. A liquefaction behavior without `fluidSim` inherits
// the manifest's global list.
export interface ShipManifest {
    fluidSim?: string[]; // fluid-simulation setting files (in aquanova/fluidSim/) a liquefied mesh may use
    behaviors?: BehaviorLibrary; // behaviour name → definition
    entities?: Entities; // mesh name → assigned behaviours
    chunks: ShipChunk[];
    portals: { centre: Vec3; normal?: Vec3; corners?: Vec3[] }[];
    environment?: ShipEnvironment;
    spawns?: { player?: Vec3; weapon?: Vec3; startChunk?: string };
    /** Every placed kit module. Together with `moduleCollision` this is the ship's collision data. */
    instances?: ShipInstance[];
    /** Kit module path → its collision primitive(s), in module-local space. One shape or several. */
    moduleCollision?: Record<string, ShipCollisionShape | ShipCollisionShape[]>;
}

export async function fetchManifest(): Promise<ShipManifest | undefined> {
    try {
        const res = await fetch(MANIFEST_URL);
        if (!res.ok) {
            return undefined;
        }
        const manifest = (await res.json()) as ShipManifest;
        // Drop chunks the editor left without bounds. A room the author created but never placed
        // anything in exports as `aabb: null, meshCount: 0`, and every consumer here (the collider
        // shell, roomAt, the per-room SDF bake) reads `aabb.min` — one empty chunk otherwise takes
        // the whole demo down before it renders a frame.
        manifest.chunks = (manifest.chunks ?? []).filter((c) => c.aabb?.min?.length === 3 && c.aabb?.max?.length === 3);
        return manifest;
    } catch {
        return undefined;
    }
}
