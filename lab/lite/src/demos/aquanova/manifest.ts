// Loading and typing of `ship_manifest.json` — the ship editor's companion to `ship.glb`.
//
// The manifest is the single source of truth for the ship's structure: chunk AABBs and their glTF
// root nodes, portals, per-entity behaviours and gameplay placements. Anything that needs to know "which room is
// this" or "what does this prop do" resolves it from here rather than inferring it from geometry.

import type { ShipEnvironment } from "../ship-manifest.js";
import type { BehaviorLibrary, BehaviorReference, Entities } from "./behaviors/index.js";
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

interface ShipEnvironmentProbeBase {
    id: string;
    capturePosition: Vec3;
}

export type ShipEnvironmentProbe = ShipEnvironmentProbeBase &
    (
        | {
              shape?: "box";
              boxPosition: Vec3;
              boxSize: Vec3;
              /** Probe yaw in degrees. Optional until the editor exports oriented probes. */
              angle?: number;
              influenceBoxPosition?: Vec3;
              influenceBoxSize?: Vec3;
              influenceInnerBoxSize?: Vec3;
          }
        | {
              shape: "sphere";
              spherePosition: Vec3;
              sphereRadius: number;
              influenceSpherePosition?: Vec3;
              influenceSphereRadius?: number;
              influenceInnerSphereRadius?: number;
          }
    );

/**
 * Resolve an XZ position to the most specific containing chunk.
 *
 * Chunk bounds come from their meshes, so doorway pieces and modules spanning a split can make
 * neighboring AABBs overlap. Choosing the smallest containing footprint makes the narrower split
 * chunk win instead of whichever chunk happens to appear first in the manifest.
 */
export function chunkAt(chunks: readonly ShipChunk[], x: number, z: number): ShipChunk | undefined {
    let best: ShipChunk | undefined;
    let bestArea = Number.POSITIVE_INFINITY;
    for (const chunk of chunks) {
        const [x0, , z0] = chunk.aabb.min;
        const [x1, , z1] = chunk.aabb.max;
        if (x < x0 || x > x1 || z < z0 || z > z1) continue;
        const area = (x1 - x0) * (z1 - z0);
        if (area < bestArea) {
            best = chunk;
            bestArea = area;
        }
    }
    return best;
}

export interface ShipRuntimeLight {
    type: "none" | "point" | "spot" | "directional";
    clustered?: boolean;
    color: Vec3;
    intensity: number;
    range: number;
    angle: number;
    castsShadows?: boolean;
}

export interface ShipLight {
    id: string;
    owner: string;
    runtime?: ShipRuntimeLight;
}

export interface ShipPortal {
    id: string;
    chunkA: string;
    chunkB: string;
    door?: string;
    centre: Vec3;
    normal?: Vec3;
    corners?: Vec3[];
    /** False closes the visibility link without changing collision or door geometry. */
    enabled?: boolean;
}

export interface ShipDoor {
    id: string;
    enabled?: boolean;
    behaviors?: BehaviorReference[];
}

// `behaviors` is a library of named behaviour definitions; `entities` assigns them to MESH NAMES,
// optionally overriding parameters per entity. Mesh names are shared across rooms, so an entity
// applies to every mesh carrying that name. Assignments remain separate so several strongly typed
// behavior instances can coexist on one mesh. A liquefaction behavior without `fluidSim` inherits
// the manifest's global list.
export interface ShipManifest {
    fluidSim?: string[]; // extensionless fluid-setting names under aquanova/fluidSim/ that a liquefied mesh may use
    behaviors?: BehaviorLibrary; // behaviour name → definition
    entities?: Entities; // mesh name → assigned behaviours
    chunks: ShipChunk[];
    /** Spatial reflection volumes, independent from rendering chunks. */
    environmentProbes?: ShipEnvironmentProbe[];
    portals: ShipPortal[];
    doors?: ShipDoor[];
    environment?: ShipEnvironment;
    /** Every placed kit module. Together with `moduleCollision` this is the ship's collision data. */
    instances?: ShipInstance[];
    /** Kit module path → its collision primitive(s), in module-local space. One shape or several. */
    moduleCollision?: Record<string, ShipCollisionShape | ShipCollisionShape[]>;
    /** Runtime parameters keyed by the matching `LIGHT_<id>` transform node in the exported glTF. */
    lights?: ShipLight[];
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
