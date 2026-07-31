// Loading and typing of `ship_manifest.json` — the Blender exporter's companion to `ship.glb`.
//
// The manifest is the single source of truth for the ship's structure: chunk AABBs and their glTF
// root nodes, portals, per-entity behaviours and spawns. Anything that needs to know "which room is
// this" or "what does this prop do" resolves it from here rather than inferring it from geometry.

import type { ShipBehaviorLibrary, ShipEntities, ShipEnvironment } from "../ship-manifest.js";
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
// optionally overriding parameters per entity (see resolveBehavior). Mesh names are shared across
// rooms, so an entity applies to every mesh carrying that name. `liquefiable` implies `dynamic`; a
// behaviour without `fluidSim` inherits the manifest's global list.
export interface ShipManifest {
    fluidSim?: string[]; // fluid-simulation setting files (in aquanova/fluidSim/) a liquefied mesh may use
    behaviors?: ShipBehaviorLibrary; // behaviour name → definition
    entities?: ShipEntities; // mesh name → assigned behaviours
    chunks: ShipChunk[];
    portals: { centre: Vec3; normal?: Vec3; corners?: Vec3[] }[];
    environment?: ShipEnvironment;
    spawns?: { player?: Vec3; weapon?: Vec3; startChunk?: string };
}

export async function fetchManifest(): Promise<ShipManifest | undefined> {
    try {
        const res = await fetch(MANIFEST_URL);
        if (!res.ok) {
            return undefined;
        }
        return (await res.json()) as ShipManifest;
    } catch {
        return undefined;
    }
}
