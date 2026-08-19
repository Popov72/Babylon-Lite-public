import {
    createPbrLocalEnvironmentProbeSet,
    enablePbrLocalCubemap,
    getPbrLocalEnvironmentProbeGridCell,
    isPbrMaterial,
    loadEnvironment,
    MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES,
    setPbrLocalEnvironmentProbeDebug,
    type EnvironmentTextures,
    type Mesh,
    type PbrLocalEnvironmentProbeSet,
    type PbrMaterialProps,
    type SceneContext,
} from "babylon-lite";
import { LOCAL_ENVIRONMENTS_URL, toLite, type Vec3 } from "./constants.js";
import {
    boxProbeNdf,
    selectContainingBoxProbe,
    selectPoiProbeBlend,
    selectStaticBoxProbe,
    type BoxProbeInfluence,
    type BoxProbeRegion,
    type ProbeBlendWeight,
    type ProbeWorldBounds,
} from "./probe-blending.js";

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
    /** Probe yaw in exported degrees. Defaults to zero for legacy exports. */
    angle?: number;
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
    /** Probe set addressed by the camera's current world-space voxel. */
    readonly cameraVoxelProbeIds: readonly string[];
}

export interface LocalEnvironmentProbeVolume extends BoxProbeInfluence, BoxProbeRegion {
    readonly capturePosition: Vec3;
    readonly debugColor: Vec3;
}

export interface LocalEnvironmentController extends LocalEnvironmentStats {
    /** Update gameplay/debug POI metadata and report the camera's current voxel probes. */
    updatePoi(position: readonly [number, number, number]): LocalEnvironmentBlendInfo;
    /** Enable per-fragment multi-probe blending or immutable per-mesh single-probe assignments. */
    setBlendingEnabled(enabled: boolean): void;
    blendingEnabled(): boolean;
    /** Replace blended PBR output with per-probe diagnostic colors. Effective only while blending. */
    setDebugEnabled(enabled: boolean): void;
    debugEnabled(): boolean;
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
const PROBE_VOXEL_SIZE = 2;

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

function fullSize(size: readonly number[]): [number, number, number] {
    return [size[0]! * 2, size[1]! * 2, size[2]! * 2];
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

function toLiteYawRadians(angleDegrees: number | undefined): number {
    const angle = angleDegrees ?? 0;
    if (!Number.isFinite(angle)) {
        throw new Error(`[aquanova] environment probe angle must be finite, received ${String(angleDegrees)}`);
    }
    // Mirroring exported glTF X into Lite reverses yaw.
    return angle === 0 ? 0 : (-angle * Math.PI) / 180;
}

function probeDebugColor(index: number): Vec3 {
    const phase = index * 2.399963229728653;
    const channel = (offset: number): number => Math.round((0.55 + Math.cos(phase + offset) * 0.45) * 255) / 255;
    return [channel(0), channel((Math.PI * 2) / 3), channel((Math.PI * 4) / 3)];
}

function meshWorldBounds(mesh: Mesh): ProbeWorldBounds {
    const world = mesh.worldMatrix;
    const minimum = mesh.boundMin;
    const maximum = mesh.boundMax;
    if (!minimum || !maximum) {
        return { centre: [world[12]!, world[13]!, world[14]!], halfSize: [0, 0, 0] };
    }
    const localCentre = [(minimum[0] + maximum[0]) * 0.5, (minimum[1] + maximum[1]) * 0.5, (minimum[2] + maximum[2]) * 0.5];
    const localHalf = [(maximum[0] - minimum[0]) * 0.5, (maximum[1] - minimum[1]) * 0.5, (maximum[2] - minimum[2]) * 0.5];
    const centre: [number, number, number] = [0, 0, 0];
    const halfSize: [number, number, number] = [0, 0, 0];
    for (let row = 0; row < 3; row++) {
        centre[row] = world[12 + row]!;
        for (let column = 0; column < 3; column++) {
            const coefficient = world[column * 4 + row]!;
            centre[row] = centre[row]! + coefficient * localCentre[column]!;
            halfSize[row] = halfSize[row]! + Math.abs(coefficient) * localHalf[column]!;
        }
    }
    return { centre, halfSize };
}

function probeVoxelGrid(probes: readonly LocalEnvironmentProbeVolume[]): {
    minimum: [number, number, number];
    maximum: [number, number, number];
    cellSize: number;
} {
    const minimum: [number, number, number] = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
    const maximum: [number, number, number] = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    for (const probe of probes) {
        const angle = probe.angleRadians ?? 0;
        const cosine = Math.abs(Math.cos(angle));
        const sine = Math.abs(Math.sin(angle));
        const extent: [number, number, number] = [
            cosine * probe.outerHalfSize[0] + sine * probe.outerHalfSize[2],
            probe.outerHalfSize[1],
            sine * probe.outerHalfSize[0] + cosine * probe.outerHalfSize[2],
        ];
        for (let axis = 0; axis < 3; axis++) {
            minimum[axis] = Math.min(minimum[axis]!, probe.centre[axis]! - extent[axis]!);
            maximum[axis] = Math.max(maximum[axis]!, probe.centre[axis]! + extent[axis]!);
        }
    }
    for (let axis = 0; axis < 3; axis++) {
        minimum[axis] = Math.floor(minimum[axis]! / PROBE_VOXEL_SIZE) * PROBE_VOXEL_SIZE - PROBE_VOXEL_SIZE;
        maximum[axis] = Math.ceil(maximum[axis]! / PROBE_VOXEL_SIZE) * PROBE_VOXEL_SIZE + PROBE_VOXEL_SIZE;
    }
    return { minimum, maximum, cellSize: PROBE_VOXEL_SIZE };
}

function emptyController(records: Record<string, LocalEnvironmentProbe>, missing: string[], blendingEnabled: boolean): LocalEnvironmentController {
    return {
        loaded: 0,
        assigned: 0,
        bytes: Object.values(records).reduce((sum, probe) => sum + probe.bytes, 0),
        missing,
        updatePoi: () => ({ probes: [], dominantProbeId: undefined, cameraVoxelProbeIds: [] }),
        setBlendingEnabled: () => {},
        blendingEnabled: () => blendingEnabled,
        setDebugEnabled: () => {},
        debugEnabled: () => false,
        blendInfo: () => ({ probes: [], dominantProbeId: undefined, cameraVoxelProbeIds: [] }),
        probeVolumes: () => [],
        environment: () => undefined,
        dominantEnvironment: () => undefined,
    };
}

/**
 * Load generated probes and prepare both material modes: one shared fragment-blended array and one
 * immutable intersecting single-probe assignment per PBR mesh.
 */
export async function applyLocalEnvironmentProbes(
    scene: SceneContext,
    meshes: readonly Mesh[],
    options: {
        readonly blendingEnabled?: boolean;
    } = {}
): Promise<LocalEnvironmentController | null> {
    const index = await fetchLocalEnvironmentIndex();
    if (!index) return null;

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
                    angleRadians: toLiteYawRadians(probe.angle),
                    debugColor: probeDebugColor(loadedProbes.length),
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
    let blendingEnabled = options.blendingEnabled !== false;
    if (!first) {
        return emptyController(records, missing, blendingEnabled);
    }

    await enablePbrLocalCubemap();
    const probeSet: PbrLocalEnvironmentProbeSet = createPbrLocalEnvironmentProbeSet(scene, {
        probes: loadedProbes.map((probe) => ({
            environment: probe.environment,
            capturePosition: probe.capturePosition,
            projectionPosition: probe.projectionCentre,
            projectionSize: fullSize(probe.projectionHalfSize),
            influencePosition: probe.centre,
            influenceInnerSize: fullSize(probe.innerHalfSize),
            influenceOuterSize: fullSize(probe.outerHalfSize),
            angleRadians: probe.angleRadians,
            debugColor: probe.debugColor,
        })),
        voxelGrid: probeVoxelGrid(loadedProbes),
    });

    const meshAssignments: Array<{ mesh: Mesh; probeIndex: number }> = [];
    for (const mesh of meshes) {
        const source = mesh.material;
        if (!source || !isPbrMaterial(source)) {
            continue;
        }
        const probe = selectStaticBoxProbe(loadedProbes, meshWorldBounds(mesh)) ?? first;
        meshAssignments.push({ mesh, probeIndex: loadedProbes.indexOf(probe) });
    }

    let info: LocalEnvironmentBlendInfo = {
        probes: [{ id: first.id, weight: 1, ndf: Number.NEGATIVE_INFINITY }],
        dominantProbeId: first.id,
        cameraVoxelProbeIds: blendingEnabled ? [first.id] : [],
    };
    let poiPosition: Vec3 | undefined;
    let debugEnabled = false;

    const applyMaterialMode = (): void => {
        const variants = new Map<PbrMaterialProps, Map<number, PbrMaterialProps>>();
        for (const { mesh, probeIndex } of meshAssignments) {
            const source = mesh.material;
            if (!source || !isPbrMaterial(source)) {
                continue;
            }
            const variantKey = blendingEnabled ? -1 : probeIndex;
            let sourceVariants = variants.get(source);
            if (!sourceVariants) {
                sourceVariants = new Map();
                variants.set(source, sourceVariants);
            }
            let variant = sourceVariants.get(variantKey);
            if (!variant) {
                variant = { ...source };
                if (blendingEnabled) {
                    variant.localEnvironment = null;
                    variant.localEnvironmentProbes = probeSet;
                } else {
                    variant.localEnvironment = loadedProbes[probeIndex]!.environment;
                    variant.localEnvironmentProbes = null;
                }
                delete (variant as { _renderFeatures?: unknown })._renderFeatures;
                sourceVariants.set(variantKey, variant);
            }
            mesh.material = variant;
        }
    };

    const applyDebugMode = (): void => {
        setPbrLocalEnvironmentProbeDebug(probeSet, blendingEnabled && debugEnabled);
    };

    const applyPoi = (position: Vec3): LocalEnvironmentBlendInfo => {
        const voxelProbeIndices = getPbrLocalEnvironmentProbeGridCell(probeSet, position).probeIndices;
        const voxelProbes = voxelProbeIndices.map((index) => loadedProbes[index]!);
        const debugWeights = selectPoiProbeBlend(voxelProbes, position, MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES);
        const contained = selectContainingBoxProbe(loadedProbes, position);
        const weightedDominant = debugWeights.reduce<ProbeBlendWeight | undefined>((best, candidate) => (!best || candidate.weight > best.weight ? candidate : best), undefined);
        const dominant = contained ?? loadedProbes.find((probe) => probe.id === weightedDominant?.id) ?? first;
        info = {
            probes: blendingEnabled
                ? debugWeights
                : [
                      {
                          id: dominant.id,
                          weight: 1,
                          ndf: boxProbeNdf(dominant, position),
                      },
                  ],
            dominantProbeId: dominant.id,
            cameraVoxelProbeIds: blendingEnabled ? voxelProbes.map((probe) => probe.id) : [],
        };
        return info;
    };

    applyMaterialMode();
    applyPoi(first.capturePosition);

    const controller: LocalEnvironmentController = {
        loaded: environments.size,
        assigned: meshAssignments.length,
        bytes: Object.values(records).reduce((sum, probe) => sum + probe.bytes, 0),
        missing,
        updatePoi(position) {
            poiPosition = [...position];
            return applyPoi(poiPosition);
        },
        setBlendingEnabled(enabled) {
            if (blendingEnabled === enabled) {
                return;
            }
            blendingEnabled = enabled;
            applyMaterialMode();
            applyDebugMode();
            applyPoi(poiPosition ?? first.capturePosition);
        },
        blendingEnabled: () => blendingEnabled,
        setDebugEnabled(enabled) {
            if (debugEnabled === enabled) {
                return;
            }
            debugEnabled = enabled;
            applyDebugMode();
        },
        debugEnabled: () => debugEnabled,
        blendInfo: () => info,
        probeVolumes: () => loadedProbes,
        environment: (probeId) => (probeId ? environments.get(probeId) : undefined),
        dominantEnvironment: () => environments.get(info.dominantProbeId ?? ""),
    };

    return controller;
}
