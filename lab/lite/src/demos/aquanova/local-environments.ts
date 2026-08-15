import {
    createPbrLocalEnvironmentBlend,
    enablePbrLocalCubemap,
    isPbrMaterial,
    loadEnvironment,
    markMaterialBindingsDirty,
    updatePbrLocalEnvironmentBlend,
    type EnvironmentTextures,
    type Mesh,
    type PbrLocalEnvironmentBlend,
    type PbrMaterialProps,
    type SceneContext,
} from "babylon-lite";
import { LOCAL_ENVIRONMENTS_URL, toLite, type Vec3 } from "./constants.js";
import { boxProbeNdf, selectContainingBoxProbe, selectPoiProbeBlend, type BoxProbeInfluence, type BoxProbeRegion, type ProbeBlendWeight } from "./probe-blending.js";

export interface LocalEnvironmentProbe {
    url: string;
    position: Vec3;
    boxPosition: Vec3;
    boxSize: Vec3;
    /** Optional POI influence volume. Defaults to the parallax box expanded for overlap. */
    influenceBoxPosition?: Vec3;
    influenceBoxSize?: Vec3;
    /** Full size of the 100%-influence inner box. */
    influenceInnerBoxSize?: Vec3;
    resolution: number;
    bytes: number;
}

export interface LocalEnvironmentIndex {
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

export interface LocalEnvironmentBlendInfo {
    readonly probes: readonly ProbeBlendWeight[];
    readonly dominantProbeId: string | undefined;
}

export interface LocalEnvironmentProbeVolume extends BoxProbeInfluence, BoxProbeRegion {
    readonly capturePosition: Vec3;
}

export interface LocalEnvironmentController extends LocalEnvironmentStats {
    /** Recompute the scene-wide two-probe blend from the player/camera point of interest. */
    updatePoi(position: readonly [number, number, number]): LocalEnvironmentBlendInfo;
    /** Switch between camera-driven two-probe blending and immutable per-mesh assignments. */
    setBlendingEnabled(enabled: boolean): void;
    blendingEnabled(): boolean;
    blendInfo(): LocalEnvironmentBlendInfo;
    /** Authored POI volumes in Lite scene coordinates, for editor/debug visualization. */
    probeVolumes(): readonly LocalEnvironmentProbeVolume[];
    environment(probeId: string | undefined): EnvironmentTextures | undefined;
    dominantEnvironment(): EnvironmentTextures | undefined;
}

interface LoadedProbe extends LocalEnvironmentProbeVolume {
    environment: EnvironmentTextures;
}

/** Default transition width on each side of a probe's parallax box. */
const DEFAULT_BLEND_DISTANCE = 1.5;
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

function halfSize(size: readonly number[]): [number, number, number] {
    return [size[0]! * 0.5, size[1]! * 0.5, size[2]! * 0.5];
}

function defaultInfluenceHalfSizes(boxSize: Vec3): {
    inner: [number, number, number];
    outer: [number, number, number];
} {
    const half = halfSize(boxSize);
    return {
        inner: [Math.max(0, half[0] - DEFAULT_BLEND_DISTANCE), Math.max(0, half[1] - DEFAULT_BLEND_DISTANCE), Math.max(0, half[2] - DEFAULT_BLEND_DISTANCE)],
        outer: [half[0] + DEFAULT_BLEND_DISTANCE, half[1] + DEFAULT_BLEND_DISTANCE, half[2] + DEFAULT_BLEND_DISTANCE],
    };
}

function accumulateWorldBounds(mesh: Mesh, min: [number, number, number], max: [number, number, number]): void {
    const lo = mesh.boundMin;
    const hi = mesh.boundMax;
    if (!lo || !hi) return;
    const world = mesh.worldMatrix;
    for (let corner = 0; corner < 8; corner++) {
        const x = (corner & 1) === 0 ? lo[0]! : hi[0]!;
        const y = (corner & 2) === 0 ? lo[1]! : hi[1]!;
        const z = (corner & 4) === 0 ? lo[2]! : hi[2]!;
        const point: Vec3 = [
            world[0]! * x + world[4]! * y + world[8]! * z + world[12]!,
            world[1]! * x + world[5]! * y + world[9]! * z + world[13]!,
            world[2]! * x + world[6]! * y + world[10]! * z + world[14]!,
        ];
        for (let axis = 0; axis < 3; axis++) {
            min[axis] = Math.min(min[axis]!, point[axis]!);
            max[axis] = Math.max(max[axis]!, point[axis]!);
        }
    }
}

function elementWorldBounds(meshes: readonly Mesh[]): { min: [number, number, number]; max: [number, number, number] } | undefined {
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (const mesh of meshes) accumulateWorldBounds(mesh, min, max);
    return min[0] <= max[0] ? { min, max } : undefined;
}

/**
 * Load generated probes and prepare both local-environment modes.
 *
 * Blended mode drives every mesh from one camera/POI blend. Static mode resolves each authored
 * element once from its initial world-space bounds, then never changes that mesh's probe.
 */
export async function applyLocalEnvironmentProbes(
    scene: SceneContext,
    meshes: readonly Mesh[],
    options: {
        readonly blendingEnabled?: boolean;
        /** Mesh primitives grouped by authored element, so one element cannot straddle probes. */
        readonly staticElements?: readonly (readonly Mesh[])[];
        /** Camera-attached meshes use the POI's single dominant probe when blending is disabled. */
        readonly poiMeshes?: readonly Mesh[];
    } = {}
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
                const boxCentre = toLite(probe.boxPosition);
                environment.boundingBoxPosition = [...boxCentre];
                environment.boundingBoxSize = [...probe.boxSize];
                environments.set(probeId, environment);

                const defaults = defaultInfluenceHalfSizes(probe.boxSize);
                loadedProbes.push({
                    id: probeId,
                    environment,
                    capturePosition: toLite(probe.position),
                    projectionCentre: boxCentre,
                    projectionHalfSize: halfSize(probe.boxSize),
                    centre: probe.influenceBoxPosition ? toLite(probe.influenceBoxPosition) : boxCentre,
                    innerHalfSize: probe.influenceInnerBoxSize ? halfSize(probe.influenceInnerBoxSize) : defaults.inner,
                    outerHalfSize: probe.influenceBoxSize ? halfSize(probe.influenceBoxSize) : defaults.outer,
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

    const first = loadedProbes[0];
    if (!first) {
        return {
            loaded: 0,
            assigned: 0,
            bytes: Object.values(records).reduce((sum, probe) => sum + probe.bytes, 0),
            missing,
            updatePoi: () => ({ probes: [], dominantProbeId: undefined }),
            setBlendingEnabled: () => {},
            blendingEnabled: () => options.blendingEnabled !== false,
            blendInfo: () => ({ probes: [], dominantProbeId: undefined }),
            probeVolumes: () => [],
            environment: () => undefined,
            dominantEnvironment: () => undefined,
        };
    }

    const blendState: PbrLocalEnvironmentBlend = createPbrLocalEnvironmentBlend(scene, {
        primary: first.environment,
        secondary: first.environment,
        weight: 0,
        parallaxCorrection: true,
    });
    const staticBlendByProbe = new Map(
        loadedProbes.map((probe) => [
            probe.id,
            createPbrLocalEnvironmentBlend(scene, {
                primary: probe.environment,
                secondary: probe.environment,
                weight: 0,
                parallaxCorrection: true,
            }),
        ])
    );

    const probeForBounds = (min: readonly number[], max: readonly number[]): LoadedProbe | undefined => {
        let best: LoadedProbe | undefined;
        let bestShare = 0;
        let bestVolume = Infinity;
        for (const probe of loadedProbes) {
            let share = 1;
            for (let axis = 0; axis < 3; axis++) {
                const half = probe.projectionHalfSize[axis]!;
                const lo = Math.max(min[axis]!, probe.projectionCentre[axis]! - half);
                const hi = Math.min(max[axis]!, probe.projectionCentre[axis]! + half);
                if (hi < lo) {
                    share = 0;
                    break;
                }
                const extent = max[axis]! - min[axis]!;
                if (extent > 0) share *= (hi - lo) / extent;
            }
            if (share <= 0) continue;
            const volume = probe.projectionHalfSize[0] * probe.projectionHalfSize[1] * probe.projectionHalfSize[2];
            if (!best || share > bestShare + SHARE_EPSILON || (share > bestShare - SHARE_EPSILON && volume < bestVolume)) {
                best = probe;
                bestShare = share;
                bestVolume = volume;
            }
        }
        return best;
    };

    const staticProbeByMesh = new Map<Mesh, LoadedProbe>();
    const poiMeshes = new Set(options.poiMeshes ?? []);
    for (const element of options.staticElements ?? []) {
        const bounds = elementWorldBounds(element);
        const probe = bounds ? probeForBounds(bounds.min, bounds.max) : undefined;
        if (!probe) continue;
        for (const mesh of element) staticProbeByMesh.set(mesh, probe);
    }
    for (const mesh of meshes) {
        if (staticProbeByMesh.has(mesh)) continue;
        const bounds = elementWorldBounds([mesh]);
        staticProbeByMesh.set(mesh, (bounds && probeForBounds(bounds.min, bounds.max)) || first);
    }

    let blendingEnabled = options.blendingEnabled !== false;
    const cloneBySource = new Map<PbrMaterialProps, Map<string, PbrMaterialProps>>();
    let assigned = 0;
    for (const mesh of meshes) {
        const source = mesh.material;
        if (!source || !isPbrMaterial(source)) continue;
        const staticProbe = staticProbeByMesh.get(mesh) ?? first;
        let byProbe = cloneBySource.get(source);
        if (!byProbe) {
            byProbe = new Map();
            cloneBySource.set(source, byProbe);
        }
        let clone = byProbe.get(staticProbe.id);
        if (!clone) {
            clone = {
                ...source,
                localEnvironmentBlend: blendingEnabled ? blendState : staticBlendByProbe.get(poiMeshes.has(mesh) ? first.id : staticProbe.id)!,
            };
            delete (clone as { _renderFeatures?: unknown })._renderFeatures;
            byProbe.set(staticProbe.id, clone);
        }
        mesh.material = clone;
        assigned++;
    }

    let currentProbeIds = `${first.id}|${first.id}`;
    let info: LocalEnvironmentBlendInfo = {
        probes: [{ id: first.id, weight: 1, ndf: Number.NEGATIVE_INFINITY }],
        dominantProbeId: first.id,
    };
    let selectedWeights: readonly ProbeBlendWeight[] = info.probes;
    let poiPosition: Vec3 | undefined;

    const applySelection = (weights: readonly ProbeBlendWeight[], position: Vec3 | undefined): LocalEnvironmentBlendInfo => {
        const primaryWeight = weights[0];
        const secondaryWeight = weights[1];
        const selectedPrimary = loadedProbes.find((probe) => probe.id === primaryWeight?.id) ?? first;
        const selectedSecondary = loadedProbes.find((probe) => probe.id === secondaryWeight?.id) ?? selectedPrimary;
        const weightedDominant = secondaryWeight && secondaryWeight.weight > (primaryWeight?.weight ?? 0) ? secondaryWeight : primaryWeight;
        const contained = !blendingEnabled && position ? selectContainingBoxProbe(loadedProbes, position) : undefined;
        const dominant = contained ?? loadedProbes.find((probe) => probe.id === weightedDominant?.id) ?? selectedPrimary;
        const dominantNdf = weights.find((weight) => weight.id === dominant.id)?.ndf ?? (position ? boxProbeNdf(dominant, position) : Number.NEGATIVE_INFINITY);
        const nextWeight = secondaryWeight?.weight ?? 0;
        const ids = `${selectedPrimary.id}|${selectedSecondary.id}`;
        const pairChanged = ids !== currentProbeIds;
        const weightChanged = Math.abs(nextWeight - blendState.weight) > 1e-5;
        if (blendingEnabled && (pairChanged || weightChanged)) {
            updatePbrLocalEnvironmentBlend(blendState, {
                primary: selectedPrimary.environment,
                secondary: selectedSecondary.environment,
                weight: nextWeight,
            });
            currentProbeIds = ids;
        }
        info = {
            probes: blendingEnabled ? weights : [{ id: dominant.id, weight: 1, ndf: dominantNdf }],
            dominantProbeId: dominant.id,
        };
        return info;
    };

    const applyMeshMode = (modeMeshes: readonly Mesh[] = meshes): void => {
        for (const mesh of modeMeshes) {
            const material = mesh.material;
            if (!material || !isPbrMaterial(material)) continue;
            const staticProbe = staticProbeByMesh.get(mesh) ?? first;
            const probeId = poiMeshes.has(mesh) ? (info.dominantProbeId ?? first.id) : staticProbe.id;
            const nextBlend = blendingEnabled ? blendState : staticBlendByProbe.get(probeId)!;
            if (material.localEnvironmentBlend === nextBlend) continue;
            material.localEnvironmentBlend = nextBlend;
            markMaterialBindingsDirty(material);
        }
    };

    const controller: LocalEnvironmentController = {
        loaded: environments.size,
        assigned,
        bytes: Object.values(records).reduce((sum, probe) => sum + probe.bytes, 0),
        missing,
        updatePoi(position) {
            const previousDominantProbeId = info.dominantProbeId;
            poiPosition = position;
            selectedWeights = selectPoiProbeBlend(loadedProbes, position, 2);
            const nextInfo = applySelection(selectedWeights, position);
            if (!blendingEnabled && nextInfo.dominantProbeId !== previousDominantProbeId) {
                applyMeshMode(options.poiMeshes ?? []);
            }
            return nextInfo;
        },
        setBlendingEnabled(enabled) {
            if (blendingEnabled === enabled) return;
            blendingEnabled = enabled;
            applySelection(selectedWeights, poiPosition);
            applyMeshMode();
        },
        blendingEnabled: () => blendingEnabled,
        blendInfo: () => info,
        probeVolumes: () => loadedProbes,
        environment: (probeId) => (probeId ? environments.get(probeId) : undefined),
        dominantEnvironment: () => environments.get(info.dominantProbeId ?? ""),
    };

    return controller;
}
