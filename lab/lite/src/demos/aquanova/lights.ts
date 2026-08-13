// Runtime lights for the Aquanova ship.
//
// Every enabled ship mesh receives the authored runtime lights. Clustered lights cover the whole
// ship, while regular lights are scoped to the chunk where they were authored.
//
// ── Where the records come from ───────────────────────────────────────────────────────────────
// The editor authors one record per lamp and the exporter writes it BOTH into `ship_manifest.json`
// and onto a `LIGHT_<id>` node in the glb, as `extras = { id, kind: "light", owner, chunk,
// runtime }`. The glb node supplies the transform because it is parented under the placement that
// owns the lamp: its world matrix already carries the placement transform AND the loader's
// glTF→Lite mirror. The manifest record with the same id overrides runtime parameters, so lighting
// can be tuned without rebuilding the glTF.
//
// ── The emission axis is local −Y ─────────────────────────────────────────────────────────────
// Not −Z. The editor chose −Y so that the +90° X rotation glTF picks up on the way into Blender
// lands it exactly on Blender's −Z, the axis an Area lamp emits along, with no fix-up at either
// end. See the header of `editor/tool/public/js/lights.js`.
//
// ── Clustered vs. not ─────────────────────────────────────────────────────────────────────────
// `runtime.clustered` decides which of Lite's two lighting paths a lamp takes:
//
//   • clustered — binned on the GPU per screen tile, so the cost is per-light-per-tile rather than
//     per-light-per-fragment. Hundreds of them are affordable, which is the point: a corridor ship
//     wants a lamp in every ceiling panel. The container is SCENE-GLOBAL, so these light every
//     chunk. That is deliberate and agreed: a prop near a doorway picking up the neighbouring
//     room's lamps is what you want, and the alternative is a per-chunk container the renderer has
//     no way to switch between mid-frame.
//   • not clustered — a regular scene light in the shared 16-slot lights UBO, and therefore scarce.
//     These are scoped to their own chunk via `includedOnlyMeshIds`, which is the only per-light
//     mesh filter Lite has.
//
// ── Chunk membership is resolved once ─────────────────────────────────────────────────────────
// A ship mesh is assigned the chunk it was authored in and keeps it. Re-assigning as a prop is
// carried through a doorway is deferred by design — it needs the light UBO's mesh selection to be
// rebuilt at runtime, and nothing in the demo moves a prop between chunks yet.

import {
    addClusteredLightContainer,
    addToScene,
    createClusteredLightContainer,
    createClusteredPointLight,
    createClusteredSpotLight,
    createDirectionalLight,
    createPointLight,
    createSpotLight,
    markClusteredLightContainerDirty,
    setMaxLights,
    MAX_LIGHTS,
    type Mesh,
    type SceneContext,
    type SceneNode,
} from "babylon-lite";
import type { ShipLight, ShipRuntimeLight } from "./manifest.js";

/** The runtime half of an authored light record. Mirrors `DEFAULT_LIGHT.runtime` in the editor. */
export type AuthoredLightRuntime = ShipRuntimeLight;

interface LightExtras {
    id: string;
    kind: string;
    owner: string;
    chunk?: string;
    runtime?: AuthoredLightRuntime;
}

export interface RuntimeLightStats {
    clusteredPoint: number;
    clusteredSpot: number;
    scoped: number;
    /** Records whose `runtime.type` is `none`. */
    disabled: number;
    /** Non-clustered lights dropped because the shared UBO has no room for them. */
    overflow: number;
    /** The effective runtime records, including manifest overrides, used by the debug overlay. */
    lights: RuntimeLightDebug[];
    /** Toggle the closest point/spot light to a world-space position. Directional lights are ignored. */
    toggleNearest(position: readonly [number, number, number]): RuntimeLightToggleResult | null;
}

export interface RuntimeLightToggleResult {
    id: string;
    enabled: boolean;
    intensity: number;
    distance: number;
}

export interface RuntimeLightDebug {
    id: string;
    chunk: string;
    type: AuthoredLightRuntime["type"];
    clustered: boolean;
    color: [number, number, number];
    intensity: number;
    range: number;
    angle: number;
    direction: [number, number, number];
    position: [number, number, number];
    /** The authored request. Runtime shadow generators are not currently created in Aquanova. */
    shadowRequested: boolean;
    castsShadows: boolean;
}

interface RuntimeLightControl {
    debug: RuntimeLightDebug;
    disabled: boolean;
    restoreIntensity: number;
    getIntensity(): number;
    setIntensity(value: number): void;
}

/** Column-major 4×4 → world position. */
function positionOf(w: ArrayLike<number>): [number, number, number] {
    return [w[12]!, w[13]!, w[14]!];
}

/** Column-major 4×4 → the world direction of the node's local −Y, normalised.
 *  Column 1 IS local +Y in world space, so the emission axis is its negation. A mirrored parent
 *  (the loader's `__root__` has scale x = −1) is handled for free: mapping the axis through the
 *  same matrix as the geometry is what keeps the cone pointing at the floor it was aimed at. */
function directionOf(w: ArrayLike<number>): [number, number, number] {
    const x = -w[4]!;
    const y = -w[5]!;
    const z = -w[6]!;
    const len = Math.hypot(x, y, z) || 1;
    return [x / len, y / len, z / len];
}

/** Collect every `LIGHT_*` node the exporter wrote, with its record. */
function collectLightNodes(root: SceneNode): Array<{ node: SceneNode; extras: LightExtras }> {
    const out: Array<{ node: SceneNode; extras: LightExtras }> = [];
    const walk = (node: SceneNode): void => {
        const extras = node.metadata?.gltf?.extras as LightExtras | undefined;
        if (extras?.kind === "light") {
            out.push({ node, extras });
        }
        for (const child of node.children) walk(child as SceneNode);
    };
    walk(root);
    return out;
}

/**
 * Build the ship's runtime lights and attach them to the scene. Call **before `registerScene`**:
 * `setMaxLights` and `addClusteredLightContainer` both feed shader/UBO layout that is baked when the
 * first pipeline is compiled.
 *
 * @param scene - Scene the ship was added to.
 * @param shipRoot - Loaded ship root, walked for the exporter's `LIGHT_*` nodes.
 * @param litMeshes - Ship meshes that receive runtime diffuse and specular lighting.
 * @param chunkOfMesh - Which chunk each mesh sits in, for the non-clustered scoping.
 * @param authoredLights - Manifest runtime values. Transforms still come from the exported glTF nodes.
 */
export function buildRuntimeLights(
    scene: SceneContext,
    shipRoot: SceneNode,
    litMeshes: Iterable<Mesh>,
    chunkOfMesh: ReadonlyMap<Mesh, string>,
    authoredLights?: readonly ShipLight[]
): RuntimeLightStats {
    const controls: RuntimeLightControl[] = [];
    const stats: RuntimeLightStats = {
        clusteredPoint: 0,
        clusteredSpot: 0,
        scoped: 0,
        disabled: 0,
        overflow: 0,
        lights: [],
        toggleNearest(position) {
            let nearest: RuntimeLightControl | null = null;
            let nearestDistanceSquared = Number.POSITIVE_INFINITY;
            for (const control of controls) {
                const dx = control.debug.position[0] - position[0];
                const dy = control.debug.position[1] - position[1];
                const dz = control.debug.position[2] - position[2];
                const distanceSquared = dx * dx + dy * dy + dz * dz;
                if (distanceSquared < nearestDistanceSquared) {
                    nearest = control;
                    nearestDistanceSquared = distanceSquared;
                }
            }
            if (!nearest) return null;
            if (nearest.disabled) {
                nearest.setIntensity(nearest.restoreIntensity);
                nearest.disabled = false;
            } else {
                nearest.restoreIntensity = nearest.getIntensity();
                nearest.setIntensity(0);
                nearest.disabled = true;
            }
            nearest.debug.intensity = nearest.getIntensity();
            return {
                id: nearest.debug.id,
                enabled: !nearest.disabled,
                intensity: nearest.debug.intensity,
                distance: Math.sqrt(nearestDistanceSquared),
            };
        },
    };
    const litSet = new Set(litMeshes);

    // glTF kit instances share material objects. Clustered lighting is material-gated, so clone the
    // ship materials before the scene-global container stamps them; otherwise an unrelated scene
    // mesh sharing the same source material would receive the lamps.
    const dynamicMaterialClones = new Map<object, object>();
    for (const mesh of litSet) {
        const source = mesh.material;
        if (!source) continue;
        let clone = dynamicMaterialClones.get(source);
        if (!clone) {
            clone = { ...source };
            delete (clone as { _renderFeatures?: unknown })._renderFeatures;
            delete (clone as { _clusteredLightState?: unknown })._clusteredLightState;
            dynamicMaterialClones.set(source, clone);
        }
        mesh.material = clone as typeof source;
    }

    // `includedOnlyMeshIds` matches on `mesh.id`, which the glTF loader does not set (it is a
    // Babylon `.babylon`-loader concept). Give the affected ship meshes a stable id here so the
    // filter has something to match, and index them by chunk in the same pass.
    const idsByChunk = new Map<string, Set<string>>();
    let n = 0;
    for (const mesh of litSet) {
        const chunk = chunkOfMesh.get(mesh);
        if (!chunk) {
            continue;
        }
        mesh.id ??= `ship${n++}`;
        const set = idsByChunk.get(chunk);
        if (set) set.add(mesh.id);
        else idsByChunk.set(chunk, new Set([mesh.id]));
    }

    const runtimeById = new Map((authoredLights ?? []).map((light) => [light.id, light.runtime] as const));
    const records = collectLightNodes(shipRoot).map((record) => ({
        ...record,
        runtime: runtimeById.has(record.extras.id) ? runtimeById.get(record.extras.id) : record.extras.runtime,
    }));
    const container = createClusteredLightContainer();

    // The shared lights UBO is a fixed-size array whose length is compiled into the WGSL, so the
    // non-clustered lights have to be counted before any of them is created.
    const scopedWanted = records.filter((r) => r.runtime && r.runtime.type !== "none" && (!r.runtime.clustered || r.runtime.type === "directional")).length;
    if (scopedWanted > MAX_LIGHTS) {
        setMaxLights(scopedWanted);
    }

    for (const { node, extras, runtime: r } of records) {
        if (!r || r.type === "none") {
            stats.disabled++;
            continue;
        }
        const w = node.worldMatrix;
        const position = positionOf(w);
        const direction = directionOf(w);
        const diffuse: [number, number, number] = [...r.color];
        const angle = (r.angle * Math.PI) / 180;
        const debug: RuntimeLightDebug = {
            id: extras.id,
            chunk: extras.chunk ?? "",
            type: r.type,
            clustered: !!r.clustered && r.type !== "directional",
            color: diffuse,
            intensity: r.intensity,
            range: r.range,
            angle: r.angle,
            direction,
            position,
            shadowRequested: !!r.castsShadows,
            castsShadows: false,
        };
        stats.lights.push(debug);

        // A directional light has no position to cluster by, so it is always a scene light.
        if (r.clustered && r.type !== "directional") {
            const light =
                r.type === "spot"
                    ? createClusteredSpotLight(container, { position, direction, diffuse, range: r.range, intensity: r.intensity, angle })
                    : createClusteredPointLight(container, { position, diffuse, range: r.range, intensity: r.intensity });
            controls.push({
                debug,
                disabled: false,
                restoreIntensity: light.intensity,
                getIntensity: () => light.intensity,
                setIntensity: (value) => {
                    light.intensity = value;
                    markClusteredLightContainerDirty(container);
                },
            });
            if (r.type === "spot") {
                stats.clusteredSpot++;
            } else {
                stats.clusteredPoint++;
            }
            continue;
        }

        if (stats.scoped >= MAX_LIGHTS) {
            stats.overflow++;
            continue;
        }
        // exponent 0: Lite's spot falloff is the glTF smooth cone ramp, matching the clustered path
        // above, so the two kinds of spot light look the same.
        const light =
            r.type === "spot"
                ? createSpotLight(position, direction, angle, 0, r.intensity)
                : r.type === "directional"
                  ? createDirectionalLight(direction, r.intensity)
                  : createPointLight(position, r.intensity);
        light.diffuse = diffuse;
        light.specular = diffuse;
        if ("range" in light) {
            light.range = r.range;
        }
        // Chunk scoping. An EMPTY set would mean "lights nothing", which is right: a chunk with no
        // dynamic props has nothing for this light to do, and leaving the filter unset would instead
        // mean "lights everything" — the one behaviour the ship must not have.
        light.includedOnlyMeshIds = idsByChunk.get(extras.chunk ?? "") ?? new Set<string>();
        addToScene(scene, light);
        if (r.type !== "directional") {
            controls.push({
                debug,
                disabled: false,
                restoreIntensity: light.intensity,
                getIntensity: () => light.intensity,
                setIntensity: (value) => {
                    light.intensity = value;
                    light._bumpLightVersion?.();
                },
            });
        }
        stats.scoped++;
    }

    if (container.pointLights.length || container.spotLights.length) {
        addClusteredLightContainer(scene, container);
        // `addClusteredLightContainer` initially stamps every material already in the scene. Remove
        // that state everywhere except the isolated runtime-lit ship clones.
        for (const mesh of scene.meshes) {
            if (litSet.has(mesh) || !mesh.material) continue;
            const material = mesh.material as { _clusteredLightState?: unknown; _renderFeatures?: unknown };
            delete material._clusteredLightState;
            material._renderFeatures = undefined;
        }
    }
    return stats;
}
