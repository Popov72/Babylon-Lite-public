import {
    enablePbrLocalCubemap,
    isPbrMaterial,
    loadEnvironment,
    markMaterialUboDirty,
    type EnvironmentTextures,
    type Mesh,
    type PbrMaterialProps,
    type SceneContext,
} from "babylon-lite";
import { LOCAL_ENVIRONMENTS_URL, toLite, type Vec3 } from "./constants.js";

export interface LocalEnvironmentProbe {
    url: string;
    position: Vec3;
    boxPosition: Vec3;
    boxSize: Vec3;
    resolution: number;
    bytes: number;
}

export interface LocalEnvironmentIndex {
    /** Explicit spatial probes. */
    probes?: Record<string, LocalEnvironmentProbe>;
    /** Pre-probe runtime indexes, retained as a migration fallback. */
    chunks?: Record<string, LocalEnvironmentProbe>;
}

export interface LocalEnvironmentStats {
    loaded: number;
    assigned: number;
    bytes: number;
    missing: string[];
}

export interface LocalEnvironmentController extends LocalEnvironmentStats {
    /** Probe for a point — the degenerate case of `probeForBounds`. */
    probeAt(position: readonly number[]): string | undefined;
    /** Probe holding the largest share of a world-space AABB. */
    probeForBounds(min: readonly number[], max: readonly number[]): string | undefined;
    /** Loaded environment for a resolved probe. */
    environment(probeId: string | undefined): EnvironmentTextures | undefined;
    /** Swap meshes to one resolved probe, or to a material with no IBL outside every probe. */
    update(meshes: readonly Mesh[], probeId: string | undefined): number;
}

interface LoadedProbe {
    id: string;
    centre: Vec3;
    size: Vec3;
    volume: number;
}

/** Two overlap shares closer than this count as equal, so the tie goes to the tighter probe. */
const SHARE_EPSILON = 1e-6;

async function fetchLocalEnvironmentIndex(): Promise<LocalEnvironmentIndex | null> {
    try {
        const response = await fetch(LOCAL_ENVIRONMENTS_URL);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return (await response.json()) as LocalEnvironmentIndex;
    } catch (err) {
        console.warn("[aquanova] no local environments —", err);
        return null;
    }
}

/**
 * Grow `min`/`max` to cover one mesh's world-space AABB.
 *
 * All eight corners are transformed rather than just the local centre, because probe membership is
 * about where the element's surfaces are, not where its middle is.
 */
function accumulateWorldBounds(mesh: Mesh, min: [number, number, number], max: [number, number, number]): void {
    const lo = mesh.boundMin;
    const hi = mesh.boundMax;
    if (!lo || !hi) {
        return;
    }
    const w = mesh.worldMatrix;
    for (let corner = 0; corner < 8; corner++) {
        const x = (corner & 1) === 0 ? lo[0]! : hi[0]!;
        const y = (corner & 2) === 0 ? lo[1]! : hi[1]!;
        const z = (corner & 4) === 0 ? lo[2]! : hi[2]!;
        const p = [
            w[0]! * x + w[4]! * y + w[8]! * z + w[12]!,
            w[1]! * x + w[5]! * y + w[9]! * z + w[13]!,
            w[2]! * x + w[6]! * y + w[10]! * z + w[14]!,
        ] as const;
        for (let i = 0; i < 3; i++) {
            if (p[i]! < min[i]!) min[i] = p[i]!;
            if (p[i]! > max[i]!) max[i] = p[i]!;
        }
    }
}

/**
 * World-space AABB of a whole element — every primitive of one placement unioned.
 *
 * A probe is a property of the element, not of a primitive: a wall whose trim is a second primitive
 * cannot have its band lit by the corridor and its face by the room, so all of its primitives are
 * resolved together against one shared box.
 */
function elementWorldBounds(meshes: readonly Mesh[]): { min: [number, number, number]; max: [number, number, number] } {
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (const mesh of meshes) accumulateWorldBounds(mesh, min, max);
    return { min, max };
}

/**
 * Load generated probe environments, assign static elements once from their world-space bounds, and
 * retain material variants for moving meshes.
 *
 * Each local probe supplies both diffuse SH and specular radiance. Materials outside every probe
 * retain no environment lighting.
 * @param scene - Scene the ship was loaded into.
 * @param meshes - Every mesh that may ever be assigned a probe, static or not.
 * @param staticElements - Never-moving geometry grouped by placement: one entry per element, holding
 *   all of its primitives.
 */
export async function applyLocalEnvironmentProbes(
    scene: SceneContext,
    meshes: readonly Mesh[],
    staticElements: readonly (readonly Mesh[])[]
): Promise<LocalEnvironmentController | null> {
    const index = await fetchLocalEnvironmentIndex();
    if (!index) return null;

    await enablePbrLocalCubemap();
    const savedEnvironment = scene._envTextures;
    const savedImageProcessing = { ...scene.imageProcessing };
    const base = new URL(LOCAL_ENVIRONMENTS_URL, location.href);
    const records = index.probes ?? index.chunks ?? {};
    const environments = new Map<string, EnvironmentTextures>();
    const loadedProbes: LoadedProbe[] = [];
    const missing: string[] = [];

    try {
        for (const [probeId, probe] of Object.entries(records)) {
            try {
                const environment = await loadEnvironment(scene, new URL(probe.url, base).href, {
                    brdfUrl: "/brdf-lut.png",
                    skipSkybox: true,
                    skipGround: true,
                });
                const centre = toLite(probe.boxPosition);
                const size: Vec3 = [...probe.boxSize];
                environment.boundingBoxPosition = [...centre];
                environment.boundingBoxSize = [...size];
                environments.set(probeId, environment);
                loadedProbes.push({
                    id: probeId,
                    centre,
                    size,
                    volume: size[0] * size[1] * size[2],
                });
            } catch (err) {
                missing.push(probeId);
                console.warn(`[aquanova] local environment ${probeId} failed to load`, err);
            }
        }
    } finally {
        scene._envTextures = savedEnvironment;
        Object.assign(scene.imageProcessing, savedImageProcessing);
    }

    // Membership is INTERSECTION, not containment. A probe box is drawn around the room it captures,
    // so an element flush with a wall — a skirting band, a door frame, a ceiling trim — routinely
    // pokes a centimetre or two through a face, and a containment test would leave it with no
    // environment at all rather than the obvious one. The winner is the probe holding the largest
    // SHARE of the element, which is a fraction rather than an overlap volume because ship trim is
    // frequently a zero-thickness sliver whose volume is exactly zero on every probe.
    //
    // An element sitting entirely inside several boxes scores 1 against all of them, and that tie is
    // broken by the tightest box — so a small room nested inside a corridor's probe still wins its
    // own geometry, exactly as it did when this was a containment test.
    const probeForBounds = (min: readonly number[], max: readonly number[]): string | undefined => {
        let best: LoadedProbe | undefined;
        let bestShare = 0;
        for (const probe of loadedProbes) {
            let share = 1;
            for (let i = 0; i < 3; i++) {
                const half = probe.size[i]! * 0.5;
                const lo = Math.max(min[i]!, probe.centre[i]! - half);
                const hi = Math.min(max[i]!, probe.centre[i]! + half);
                if (hi < lo) {
                    share = 0;
                    break;
                }
                const extent = max[i]! - min[i]!;
                if (extent > 0) share *= (hi - lo) / extent;
            }
            if (share <= 0) {
                continue;
            }
            if (!best || share > bestShare + SHARE_EPSILON || (share > bestShare - SHARE_EPSILON && probe.volume < best.volume)) {
                best = probe;
                bestShare = share;
            }
        }
        return best?.id;
    };

    const probeAt = (position: readonly number[]): string | undefined => probeForBounds(position, position);

    const sourceByMesh = new Map<Mesh, PbrMaterialProps>();
    for (const mesh of meshes) {
        const source = mesh.material;
        if (source && isPbrMaterial(source)) sourceByMesh.set(mesh, source);
    }

    const clones = new Map<string, Map<PbrMaterialProps, PbrMaterialProps>>();
    const materialFor = (source: PbrMaterialProps, probeId: string | undefined): PbrMaterialProps => {
        const environment = probeId ? environments.get(probeId) : undefined;
        if (!environment) return source;
        let byMaterial = clones.get(probeId!);
        if (!byMaterial) {
            byMaterial = new Map();
            clones.set(probeId!, byMaterial);
        }
        let clone = byMaterial.get(source);
        if (!clone) {
            clone = { ...source, localEnvironment: environment };
            delete (clone as { _renderFeatures?: unknown })._renderFeatures;
            byMaterial.set(source, clone);
        }
        return clone;
    };

    const controller: LocalEnvironmentController = {
        loaded: environments.size,
        assigned: 0,
        bytes: Object.values(records).reduce((sum, probe) => sum + probe.bytes, 0),
        missing,
        probeAt,
        probeForBounds,
        environment(probeId) {
            return probeId ? environments.get(probeId) : undefined;
        },
        update(updateMeshes, probeId) {
            let changed = 0;
            for (const mesh of updateMeshes) {
                const source = sourceByMesh.get(mesh);
                if (!source) continue;
                const currentIntensity = isPbrMaterial(mesh.material) ? mesh.material.environmentIntensity : undefined;
                const next = materialFor(source, probeId);
                if (currentIntensity !== undefined && next.environmentIntensity !== currentIntensity) {
                    next.environmentIntensity = currentIntensity;
                    markMaterialUboDirty(next);
                }
                if (mesh.material === next) continue;
                mesh.material = next;
                changed++;
            }
            return changed;
        },
    };

    for (const element of staticElements) {
        const { min, max } = elementWorldBounds(element);
        if (min[0] > max[0]) {
            continue;
        }
        const probeId = probeForBounds(min, max);
        if (!probeId) {
            continue;
        }
        for (const mesh of element) {
            const source = sourceByMesh.get(mesh);
            if (!source) continue;
            mesh.material = materialFor(source, probeId);
            controller.assigned++;
        }
    }
    return controller;
}
