// Aquanova demo — first-person sci-fi shooter tech demo built on Lite.
//
// The player traverses a confined spaceship wielding the "Liquefactor": melt flying alien foes
// and ship props into GPU fluid. This slice LOADS THE SHIP: a chunked modular interior authored
// in Blender from the CC0 Quaternius "Modular SciFi MegaKit" and exported to a single glTF
// (ship.glb) with a companion ship_manifest.json describing chunks, portals, doors and spawns
// (see lab/public/aquanova/ASSET-LICENSES.md and SciFiShip/README.md).
//
// What this slice does:
//   • loads ship.glb and lights it with an HDR environment (IBL);
//   • builds one static trimesh collider from the whole ship so the player collides with every
//     wall, floor and prop — no hand-authored boxes;
//   • spawns a Havok first-person character-controller capsule at the manifest player spawn, the
//     camera riding at eye height (WASD/arrows walk in the horizontal plane, mouse-drag looks).
//
// The ship runs back → front along glTF +X (storage → corridor → junction → cargo bay). Lite's
// glTF loader mirrors handedness on the __root__ (scale x = -1), so glTF (x,y,z) renders at Lite
// (-x, y, z); every manifest coordinate is converted through `toLite()` before use.
//

import HavokPhysics from "@babylonjs/havok";
import {
    addTask,
    addTaskAfter,
    addTaskBefore,
    addToScene,
    createDepthResolveTask,
    createEngine,
    createFreeCamera,
    createGpuPicker,
    createHavokWorld,
    createCopyToTextureTask,
    createPhysicsBody,
    createPhysicsCharacterController,
    createPhysicsShape,
    createRenderTarget,
    createRenderTask,
    createSceneContext,
    createSmaaPostProcessTask,
    createTaaPostProcessTask,
    createTransformNode,
    createUtilityLayer,
    AcesToneMapping,
    NeutralToneMapping,
    StandardToneMapping,
    setSceneImageProcessing,
    enableMaterialPlugins,
    getMeshGeometry,
    getFrameGraph,
    getPhysicsBodyLinearVelocity,
    getProjectionMatrix,
    getViewMatrix,
    getViewProjectionMatrix,
    isPbrMaterial,
    loadGltf,
    loadHdrEnvironment,
    loadSkybox,
    PhysicsMotionType,
    PhysicsShapeType,
    physicsRaycast,
    rebuildScenePbrPipelines,
    registerScene,
    registerUtilityLayer,
    removeFromScene,
    removePhysicsBody,
    setMeshVisible,
    setParent,
    setPhysicsBodyShape,
    setPhysicsTimestepMs,
    startEngine,
} from "babylon-lite";
import type { Material, Mesh, PbrMaterialProps, PhysicsBody, RenderTask, SceneNode, Task, ToneMapping } from "babylon-lite";
import { fillMeshParticles } from "../particle-fill.js";
import { createMlsMpmSim } from "babylon-lite/fluid/mls-mpm-sim.js";
import { createPbMpmSim } from "babylon-lite/fluid/pbmpm-sim.js";
import { createFluidSurfaceTask } from "babylon-lite/fluid/fluid-surface-render.js";
import type { FluidSim, ForceFieldSpec, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import { createLiquefyPlugin } from "../liquefy-plugin.js";
import { buildLitParticleColors } from "../particle-lit-colors.js";
import type { LitColorScene } from "../particle-lit-colors.js";
import { DEFAULT_SHIP_IBL_STRENGTH, resolveExposure, resolveToneMapping } from "../ship-manifest.js";
import type { LiquefyState } from "../liquefy-plugin.js";
import { gridFloorY, gridTopY } from "../fluid/grid-bounds.js";
import { CEIL_Y, ENV_URL, FLOOR_Y, SHIP_URL, SKYBOX_EXT, SKYBOX_SIZE, SKYBOX_URL, toLite, type Vec3 } from "./constants.js";
import { applyBakedLightmaps, fetchLightmapIndex, isBakeExcluded } from "./lightmaps.js";
import { buildRuntimeLights } from "./lights.js";
import { buildManifestColliders, createWorldCollisionShape } from "./colliders.js";
import { worldShapesForMatrix, type WorldCollisionShape } from "./collision-shapes.js";
import {
    localizePrimitive,
    packPrimitive,
    packPrimitives,
    primBufferBytes,
    PRIMITIVES_WGSL,
    PRIM_ACTIVE_OFFSET,
    PRIM_HEADER,
    PRIM_STRIDE,
    setPackedPrimitiveActive,
    type FluidPrimitive,
} from "./collision-field.js";
import { chunkAt, fetchManifest } from "./manifest.js";
import { createInspectOverlay } from "./debug/inspect-overlay.js";
import { createPerfOverlay } from "./debug/perf-overlay.js";
import { createFluidProfiler, type FluidProfilerImpl } from "../fluid/gpu-profiler.js";
import { createColliderOverlay } from "./debug/collider-overlay.js";
import { createLightOverlay } from "./debug/light-overlay.js";
import { LAB_DEBUG } from "./debug-flag.js";
import { GRAPHICS_SETTING_DEFS, loadGraphicsSettings, saveGraphicsSettings } from "./settings.js";
import { meshGroupBounds, type MeshGroupBounds } from "./mesh-bounds.js";
import { canonicalSettingName, fetchFluidSetting, hexToRgb, type FluidFoamSetting, type FluidRenderSetting, type FluidSimSetting } from "./fluid-setting.js";
import { BehaviorManager, PlayerBehavior, type LiquefiableBehaviorConfig, type MeshBehaviorAvailability } from "./behaviors/index.js";

export async function main(): Promise<void> {
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    /** Reflect a toggle's state in the HUD hint, e.g. "M: MSAA (on)". No-op if the span is absent. */
    const setHudFlag = (id: string, on: boolean): void => {
        const el = document.getElementById(id);
        if (el) el.textContent = on ? "on" : "off";
    };
    /**
     * SSAA: render the whole frame at `scale`× the display resolution and let the compositor
     * downscale it.
     *
     * Done through CSS rather than by resizing render targets, for two reasons. Every canvas-sized
     * target already derives from `canvas.width` — including the fluid task's screen-space depth and
     * thickness buffers — so scaling the backing store scales the entire pipeline at once with no
     * chance of two passes disagreeing about size. And the engine calls `resizeEngine` every frame,
     * recomputing the backing store from `clientWidth × devicePixelRatio`; setting the size directly
     * would be undone immediately, whereas growing the ELEMENT makes the engine compute the size we
     * want by itself. The transform shrinks it back to the layout box for display.
     *
     * Picking is unaffected: it uses `clientWidth / 2`, which is still the centre of the crosshair.
     */
    const applySsaa = (scale: number): void => {
        const pct = `${scale * 100}%`;
        canvas.style.width = pct;
        canvas.style.height = pct;
        canvas.style.transformOrigin = "0 0";
        canvas.style.transform = scale === 1 ? "" : `scale(${1 / scale})`;
        const el = document.getElementById("ssaaState");
        if (el) el.textContent = `${scale}×`;
        canvas.dataset.ssaa = String(scale);
    };
    // Player-tunable graphics (defaults ← localStorage ← ?msaa= query override). Resolved before the
    // engine so a future setting that has to be chosen at engine-creation time can be read here too.
    const graphics = loadGraphicsSettings();
    applySsaa(graphics.ssaa);
    // msaaSamples: 1 — the ship renders into an offscreen target the fluid surface samples, never
    // straight to the swapchain, so the engine's own MSAA would never apply. The scene pass does its
    // own multisampling instead; see the MSAA block further down.
    const engine = await createEngine(canvas, { msaaSamples: 1, srgb: graphics.srgb });
    const scene = createSceneContext(engine, { defaultRenderTask: false });

    const cam = createFreeCamera({ x: 0, y: 1.5, z: 0 }, { x: -1, y: 1.5, z: 0 });
    cam.nearPlane = 0.1;
    cam.farPlane = 400;
    scene.camera = cam;

    // Fetch the ship manifest up front — it drives the IBL strength, tone mapping, colliders, and spawn.
    const manifest = await fetchManifest();

    // Preload the fluid-simulation settings the manifest lists (parsed once). At liquefy time a mesh
    // uses its pinned override if it has one, else a random pick from this global list.
    //
    // The union of the global list AND every behaviour's own `fluidSim` is loaded, keyed by canonical
    // name. Loading only the global list meant a behaviour naming its own setting (the crates carry
    // `fluidSim: ["liquid-slow"]`) missed the map at liquefy time and silently fell back to
    // DEFAULT_SAMPLE_RADIUS + FLUID_SETTING: the crates sampled at 0.03 instead of the file's 0.08,
    // giving ~19x the particles (6260 vs 330) and default physics instead of the file's.
    const fluidSettingNames = (manifest?.fluidSim ?? []).map(canonicalSettingName);
    const behaviorSettingNames = new Set<string>();
    for (const b of Object.values(manifest?.behaviors ?? {})) {
        if ("fluidSim" in b) {
            for (const n of b.fluidSim ?? []) behaviorSettingNames.add(canonicalSettingName(n));
        }
    }
    for (const e of Object.values(manifest?.entities ?? {})) {
        for (const ref of e?.behaviors ?? []) {
            if ("fluidSim" in ref) {
                for (const n of ref.fluidSim ?? []) behaviorSettingNames.add(canonicalSettingName(n));
            }
        }
    }
    const fluidSettings = new Map<string, FluidSimSetting>();
    await Promise.all(
        [...new Set([...fluidSettingNames, ...behaviorSettingNames])].map(async (name) => {
            const s = await fetchFluidSetting(name);
            if (s) fluidSettings.set(name, s);
        })
    );

    // ── Load the ship + image-based lighting ──────────────────────────────────────────────
    // Lighting is fully image-based (the export carries no punctual lights): the bank_vault HDRI
    // supplies ambient + specular, and the trim/fixtures are HDR-emissive geometry (M_Light etc.,
    // via KHR_materials_emissive_strength). entities[0] is the __root__ (RH→LH mirror baked in)
    // with all geometry beneath it. Note: Lite applies IBL per-surface with no occlusion, so the
    // HDRI lights even these sealed compartments (Blender needs baked GI for the equivalent look).
    const ship = await loadGltf(engine, SHIP_URL);
    const shipRoot = ship.entities[0] as SceneNode;
    addToScene(scene, shipRoot);

    // Build the HDR IBL (prefiltered specular cubemap + SH irradiance + BRDF LUT) from the .hdr.
    // 512-pixel cube faces rather than the 256 default: the prefiltered specular cube is what the
    // ship's metal reflects, and at 256 those reflections are visibly softer.
    const env = await loadHdrEnvironment(scene, ENV_URL, { faceSize: 512, skipSkybox: true, skipGround: true });

    // Backdrop seen through the ship's openings. Kept separate from the IBL above: the HDRI is what
    // lights the metal, this is only what you see. Non-fatal — the cube faces are gitignored like
    // the ship and the HDR, so a fresh clone without them still runs, just against the clear colour.
    try {
        await loadSkybox(scene, SKYBOX_URL, SKYBOX_EXT, SKYBOX_SIZE);
    } catch (err) {
        console.warn("[aquanova] skybox not loaded (missing cube faces?)", err);
    }

    // Honour the manifest's IBL strength: environmentIntensity scales only the IBL contribution
    // (diffuse SH + specular) per material, leaving the emissive fixtures at full brightness.
    const iblStrength = manifest?.environment?.strength ?? DEFAULT_SHIP_IBL_STRENGTH;
    const seenMats = new Set<object>();
    const applyIblStrength = (node: SceneNode): void => {
        const mat = (node as { material?: { environmentIntensity?: number } }).material;
        if (mat && !seenMats.has(mat)) {
            seenMats.add(mat);
            mat.environmentIntensity = iblStrength;
        }
        for (const child of node.children) applyIblStrength(child as SceneNode);
    };
    applyIblStrength(shipRoot);

    // ── Classify ship meshes: collect all, plus global dynamic / liquefiable and per-room static sets ─
    // Entities are keyed by the glTF NODE name (what the Babylon sandbox shows, e.g. "Door_D00_L"),
    // NOT the glTF mesh name Lite puts on `mesh.name`. The loader parents each node's primitives under
    // a TransformNode carrying the node name, so the walk below records that name per mesh and groups
    // one node's primitives together.
    //
    // `_primitiveN` WRAPPERS: an asset round-tripped through Babylon carries its loader's split
    // convention baked in as real nodes — a mesh-less parent "player" holding a child node
    // "player_primitive0" that owns the geometry (195 of 292 nodes in the current ship). Those
    // wrappers are collapsed into their parent so the author's node name is what the manifest keys,
    // and so a node's primitives regroup into ONE entity instead of N single-primitive ones. The test
    // is exact — the child's name must be the parent's plus `_primitive<digits>` — so a genuinely
    // authored node is never absorbed.
    //
    // `chunkStaticMeshes` (everything NOT dynamic/liquefiable, grouped by the chunk whose AABB contains
    // the mesh centre) feeds the per-room SDF bake; manifest AABBs are glTF-space, so the Lite-space
    // mesh centre is converted back.
    const PRIMITIVE_WRAPPER = /_primitive\d+$/;
    // BLENDER DE-DUPLICATION: the baked ship is a Blender round-trip, and Blender's importer makes
    // object names unique by appending `.001`, `.002`… Two crates were placed from the same module,
    // so glTF's two sibling `crate4` nodes come back as `crate4` and `crate4.001` (the only such
    // collision in the current ship). Entities are keyed by node name deliberately — one manifest
    // entry is meant to drive every mesh sharing that name — so the suffix is stripped to restore
    // the authored name. Exactly three digits, anchored at the end, is Blender's own format; no kit
    // module is named that way.
    const BLENDER_DEDUP = /\.\d{3}$/;
    const authoredName = (name: string): string => name.replace(BLENDER_DEDUP, "");
    const allShipMeshes: Mesh[] = [];
    const nodeNameOfMesh = new Map<Mesh, string>(); // mesh → owning glTF node name
    const ownerOfMesh = new Map<Mesh, SceneNode>(); // mesh → the glTF node that owns it (carries extras)
    const nodePrimitives = new Map<Mesh, Mesh[]>(); // mesh → every primitive of the SAME node (incl. itself)
    const meshesByNodeName = new Map<string, Mesh[]>(); // node name → all its primitives, ship-wide
    const primitivesByOwner = new Map<SceneNode, Mesh[]>();
    // glTF node name of each chunk root ("CHUNK_CH00_Storage") → chunk id, straight from the manifest.
    // A mesh's chunk is its CHUNK_* ancestor, tracked on the way down — the exporter parents every
    // mesh under exactly one chunk root, and `chunks[].meshCount` agrees with that walk exactly.
    const chunkIdByRootNode = new Map<string, string>();
    for (const c of manifest?.chunks ?? []) if (c.node) chunkIdByRootNode.set(c.node, c.id);
    const chunkOfMesh = new Map<Mesh, string>();
    /** Instance id → its placement node, from the exporter's `extras = { id, module, chunk }`. The
     *  node is what carries the placement transform in the scene's own space. */
    const placementNodes = new Map<string, { module: string; node: SceneNode }>();
    const collectMeshes = (node: SceneNode, owner: SceneNode, chunk: string | undefined): void => {
        for (const child of node.children) {
            const c = child as SceneNode;
            const cName = authoredName(c.name);
            const inChunkNow = chunkIdByRootNode.get(cName) ?? chunk;
            const ex = c.metadata?.gltf?.extras as { id?: string; module?: string } | undefined;
            if (ex?.id && ex.module && !placementNodes.has(ex.id)) placementNodes.set(ex.id, { module: ex.module, node: c });
            if ((c as Mesh).material) {
                const m = c as Mesh;
                const ownerName = authoredName(owner.name);
                allShipMeshes.push(m);
                nodeNameOfMesh.set(m, ownerName);
                ownerOfMesh.set(m, owner);
                if (inChunkNow) chunkOfMesh.set(m, inChunkNow);
                const group = primitivesByOwner.get(owner);
                if (group) group.push(m);
                else primitivesByOwner.set(owner, [m]);
                const byName = meshesByNodeName.get(ownerName);
                if (byName) byName.push(m);
                else meshesByNodeName.set(ownerName, [m]);
                continue;
            }
            // A wrapper keeps the parent as owner; any other node owns its own subtree.
            const wrapper = PRIMITIVE_WRAPPER.test(cName) && cName.replace(PRIMITIVE_WRAPPER, "") === authoredName(node.name);
            collectMeshes(c, wrapper ? owner : c, inChunkNow);
        }
    };
    collectMeshes(shipRoot, shipRoot, undefined);
    for (const group of primitivesByOwner.values()) for (const m of group) nodePrimitives.set(m, group);
    const shipPbrMaterials = (): PbrMaterialProps[] => {
        const materials = new Set<PbrMaterialProps>();
        for (const mesh of allShipMeshes) {
            if (mesh.material && isPbrMaterial(mesh.material)) materials.add(mesh.material);
        }
        return [...materials];
    };
    const setShipSpecularAA = (on: boolean): void => {
        for (const material of shipPbrMaterials()) {
            material.enableSpecularAA = on;
            material._renderFeatures = undefined;
        }
    };
    setShipSpecularAA(graphics.specularAA);
    canvas.dataset.specularAa = String(graphics.specularAA);
    setHudFlag("specularAaState", graphics.specularAA);
    const behaviorManager = new BehaviorManager({
        library: manifest?.behaviors,
        entities: manifest?.entities,
        meshesByEntityName: meshesByNodeName,
        entityNameOf: (mesh) => nodeNameOfMesh.get(mesh) ?? mesh.name,
    });

    // Portal meshes ("Portal_*") are doorway markers the exporter emits for culling / door pairing —
    // NOT real geometry. They render as a visible pane spanning the doorway (seen from the corridor)
    // AND sit coplanar with the door leaves, so the weapon pick can hit the portal instead of the door
    // behind it (the door then won't liquefy). Hide them, make them non-pickable, and skip them in the
    // mesh classification below so they never enter the SDF bake either.
    const isPortalMesh = (m: Mesh): boolean => m.name.startsWith("Portal_");
    for (const m of allShipMeshes) {
        if (isPortalMesh(m)) {
            setMeshVisible(m, false);
            (m as { pickable?: boolean }).pickable = false;
        }
    }

    // ── Placement markers (player / weapon start) ────────────────────────────────────────
    // An entity carrying a `*_startpos` behaviour is a PLACEMENT MARKER, not scenery: something
    // spawns at its node and the marker itself is DISABLED — hidden, non-pickable, and skipped by
    // every classification pass below so it never enters physics, the SDF bake or the target sets.
    // Its `direction` parameter is the glTF-space vector the spawned thing initially faces. The
    // library definitions are empty, so a marker is matched by NAME on the entity's reference.
    const markerMeshes = new Set<Mesh>();
    interface Marker {
        entity: string;
        min: number[] | null;
        max: number[] | null;
        direction?: number[];
    }
    const resolveMarker = (behaviorName: string): Marker | null => {
        const found = behaviorManager.findEntityWithBehavior(behaviorName);
        if (!found) return null;
        const meshes = found.meshes;
        if (!meshes.length) {
            // eslint-disable-next-line no-console
            console.warn(`[aquanova] ${behaviorName} entity "${found.entityName}" matches no ship node — ignoring it`);
            return null;
        }
        // World AABB — already Lite space, since it comes from the loaded ship (mirror applied).
        let min: number[] | null = null;
        let max: number[] | null = null;
        for (const m of meshes) {
            markerMeshes.add(m);
            setMeshVisible(m, false);
            (m as { pickable?: boolean }).pickable = false;
            const a = m.boundMin;
            const b = m.boundMax;
            if (!a || !b) continue;
            min = min ? [Math.min(min[0]!, a[0]!), Math.min(min[1]!, a[1]!), Math.min(min[2]!, a[2]!)] : [a[0]!, a[1]!, a[2]!];
            max = max ? [Math.max(max[0]!, b[0]!), Math.max(max[1]!, b[1]!), Math.max(max[2]!, b[2]!)] : [b[0]!, b[1]!, b[2]!];
        }
        return { entity: found.entityName, min, max, direction: found.assignment.direction };
    };
    const playerMarker = resolveMarker("player");
    const weaponMarker = resolveMarker("weapon");
    // Disabled marker geometry is treated exactly like a portal: excluded everywhere below.
    const isDisabledMesh = (m: Mesh): boolean => isPortalMesh(m) || markerMeshes.has(m);

    // Weapon pickup spot: the marker's centre, else the legacy glTF-space spawn. Exposed for the
    // pickup prop (and QA) rather than being read from the removed `spawns.weapon`.
    const weaponSpawn: Vec3 =
        weaponMarker?.min && weaponMarker.max
            ? [(weaponMarker.min[0]! + weaponMarker.max[0]!) / 2, weaponMarker.max[1]!, (weaponMarker.min[2]! + weaponMarker.max[2]!) / 2]
            : toLite(manifest?.spawns?.weapon ?? [7, 0.95, 2.8]);
    canvas.dataset.weaponSpawn = weaponSpawn.map((v) => v.toFixed(3)).join(",");

    // Disable KHR_materials_transmission on the ship: the fluid surface composites into the swapchain
    // AFTER the scene renders, but transmission retargets the scene task to an offscreen HDR buffer and
    // appends a final tonemap pass that would overwrite (hide) the fluid. Dropping the refraction
    // sub-feature keeps the scene on our sceneColorRT (like the Liquefactor demo) so the water shows;
    // the glass then renders as a plain surface instead of refracting.
    for (const m of allShipMeshes) {
        const mat = m.material;
        if (mat && isPbrMaterial(mat)) {
            const ss = (mat as unknown as { subsurface?: { refraction?: unknown } }).subsurface;
            if (ss?.refraction) ss.refraction = undefined;
        }
    }

    // Which chunk (room) a mesh belongs to. Taken from the exporter's own scene-graph grouping (the
    // CHUNK_* ancestor recorded during collectMeshes), NOT from a centre-in-AABB test: chunk AABBs
    // overlap and a mesh's centre says nothing reliable about which room authored it. The guess put
    // 165 meshes in CH00_Storage, a chunk the manifest says has 48 — so ~117 corridor meshes were being
    // baked into the storage room's SDF, which is what put a phantom wall across the middle of it.
    const roomOfMesh = (m: Mesh): string => chunkOfMesh.get(m) ?? "unknown";

    // Node name → the kit module it was placed from, so collision policy can be decided per module.
    const moduleOfNode = new Map<string, string>();
    for (const inst of manifest?.instances ?? []) {
        const node = inst.node ?? inst.name;
        if (node && !moduleOfNode.has(node)) moduleOfNode.set(node, inst.module);
    }

    // The exporter stamps every PLACEMENT's glTF node with `extras = { id, module, chunk }`, so a mesh
    // reaches its own instance in one hop. That is the only reliable key: node NAMES are not unique
    // (both stacked crates are called `crate4`), and matching by name handed the second crate the
    // first one's shape — its collider ended up a full crate below itself.
    const instanceIdOfMesh = (m: Mesh): string | undefined => {
        const extras = ownerOfMesh.get(m)?.metadata?.gltf?.extras as { id?: string } | undefined;
        return extras?.id;
    };

    // Instance id → its authored collision primitives, resolved into world space from the LOADED
    // node's matrix. Not from `instances[]`: the manifest's own `space` block says `moduleCollision`
    // is glTF space while `instances[]` is editor space, so composing those two mixes spaces — it
    // looks right on anything symmetrical and is wrong on everything turned. The loaded node already
    // carries the glTF→scene mirror, so node.worldMatrix · shape is correct with no conversion.
    const placementById = new Map<string, { node: string; shapes: WorldCollisionShape[] }>();
    for (const [id, pn] of placementNodes) {
        const shapes = manifest?.moduleCollision?.[pn.module];
        if (!shapes) continue;
        placementById.set(id, { node: pn.node.name, shapes: worldShapesForMatrix(pn.node.worldMatrix, shapes) });
    }
    /** Flattened view for the debug overlay and QA, tagged with the instance that produced each. */
    const manifestPlacements: Array<{ id: string; node: string; shape: WorldCollisionShape }> = [];
    for (const [id, { node, shapes }] of placementById) for (const shape of shapes) manifestPlacements.push({ id, node, shape });

    const chunkStaticMeshes = new Map<string, Mesh[]>();
    behaviorManager.classifyMeshes(allShipMeshes, {
        isDisabled: isDisabledMesh,
        instanceIdOf: instanceIdOfMesh,
    });

    // Pass 2 — per-room STATIC geometry (everything that neither moves nor melts) for the SDF bake.
    for (const c of manifest?.chunks ?? []) {
        const staticList: Mesh[] = [];
        for (const m of allShipMeshes) {
            if (isDisabledMesh(m)) continue;
            if (behaviorManager.dynamicMeshes.has(m) || behaviorManager.dissolvableMeshes.has(m)) continue;
            if (chunkOfMesh.get(m) !== c.id) continue;
            staticList.push(m);
        }
        chunkStaticMeshes.set(c.id, staticList);
    }
    canvas.dataset.dynamicCount = String(behaviorManager.dynamicMeshes.size);
    canvas.dataset.liquefiableCount = String(behaviorManager.liquefiableMeshes.size);

    // ── Baked lighting ────────────────────────────────────────────────────────────────────────
    // The ship is primarily lit by a Blender bake. Static geometry gets its chunk's KTX2 irradiance
    // atlas as a multiplier over the normal PBR result (`lightmaps.ts`), so the environment still
    // supplies metallic reflections. The authored runtime lamps cover the complement: every mesh
    // the bake left out, which `isBakeExcluded` reads straight off the geometry. Scoping them by
    // the `dynamic` behaviour instead used to leave the liquefiable-only doors with no lightmap AND
    // no lamp, lit by the IBL alone.
    //
    // Both run BEFORE registerScene: `enablePbrLightmap` has to register its PBR extension, and the
    // clustered-light container and light-UBO size have to be known, before the first pipeline is
    // composed. Lights go FIRST because `addClusteredLightContainer` stamps every material already
    // in the scene; running it first means the lightmap clones are made afterwards and can drop that
    // stamp, keeping authored runtime lights off static geometry while retaining its environment
    // lighting path.
    // A missing lightmaps.json is non-fatal for the same reason the skybox is — the atlases are
    // gitignored, so a fresh clone renders the ship IBL-lit rather than not at all.
    const runtimeLitMeshes = new Set(allShipMeshes.filter((m) => !isDisabledMesh(m) && isBakeExcluded(m)));
    canvas.dataset.runtimeLitCount = String(runtimeLitMeshes.size);
    // Keep dynamic/unbaked materials independent from the room surfaces. A
    // shared source material is copied before runtime-light stamping so a
    // static mesh can retain the baked environment strength.
    const dynamicIblStrength = manifest?.environment?.dynamicStrength ?? iblStrength;
    const dynamicMaterialClones = new Map<PbrMaterialProps, PbrMaterialProps>();
    for (const mesh of runtimeLitMeshes) {
        const source = mesh.material;
        if (!source || !isPbrMaterial(source)) continue;
        let clone = dynamicMaterialClones.get(source);
        if (!clone) {
            clone = { ...source };
            delete (clone as { _renderFeatures?: unknown })._renderFeatures;
            delete (clone as { _clusteredLightState?: unknown })._clusteredLightState;
            clone.environmentIntensity = dynamicIblStrength;
            dynamicMaterialClones.set(source, clone);
        }
        mesh.material = clone;
    }
    const lights = buildRuntimeLights(scene, shipRoot, runtimeLitMeshes, chunkOfMesh, manifest?.lights);
    canvas.dataset.clusteredLightCount = String(lights.clusteredPoint + lights.clusteredSpot);
    if (lights.overflow) console.warn(`[aquanova] ${lights.overflow} non-clustered light(s) dropped: the shared lights UBO is full`);
    const lightmapIndex = await fetchLightmapIndex();
    if (lightmapIndex) {
        const lm = await applyBakedLightmaps(engine, lightmapIndex, allShipMeshes, chunkOfMesh);
        canvas.dataset.lightmappedCount = String(lm.lit);
        if (lm.missing.length) console.warn("[aquanova] baked chunks with no meshes loaded:", lm.missing.join(", "));
        console.log(`[aquanova] lightmaps: ${lm.lit} meshes lit, ${lm.skipped} dynamic/unmapped, ${(lm.bytes / 1048576).toFixed(2)} MB`);
    }

    // Match the Blender view transform the ship was authored against, straight from the manifest:
    // tone mapping (Khronos PBR Neutral keeps the strength-boosted emissive trim from clipping to
    // white and losing its colour) and exposure, converted from Blender stops to a linear multiplier.
    // Set before registerScene so the first PBR build + the deferred skybox snapshot pick it up.
    const tone = resolveToneMapping(manifest?.environment?.toneMapping);
    scene.imageProcessing.toneMappingEnabled = tone !== null;
    if (tone) scene.imageProcessing.toneMapping = tone;

    /** Assigned just below; declared here because the key handler further down closes over it. */
    let cycleTone: () => void = () => {};

    // ── Tone-mapping cycle (O) ───────────────────────────────────────────────────────────────────    // Every algorithm Babylon-Lite ships, plus "None" (no curve at all). The manifest still chooses
    // the startup value — this is a look-dev comparison tool, so it deliberately does NOT persist:
    // reloading returns to what `environment.toneMapping` authored.
    //
    // The curve is a compile-time PBR shader feature (it is injected WGSL, not a uniform), so
    // switching recompiles the affected pipelines. setSceneImageProcessing does that for us; it is
    // async, hence the in-flight guard — hammering the key would otherwise stack rebuilds.
    const TONE_MAPPINGS: ReadonlyArray<{ name: string; tm: ToneMapping | null }> = [
        { name: "Neutral", tm: NeutralToneMapping },
        { name: "ACES", tm: AcesToneMapping },
        { name: "Standard", tm: StandardToneMapping },
        { name: "None", tm: null },
    ];
    let toneIndex = Math.max(
        0,
        TONE_MAPPINGS.findIndex((t) => (tone === null ? t.tm === null : t.tm?.id === tone.id))
    );
    let toneBusy = false;
    const showTone = (): void => {
        const el = document.getElementById("toneState");
        if (el) el.textContent = TONE_MAPPINGS[toneIndex]!.name;
        canvas.dataset.tonemap = TONE_MAPPINGS[toneIndex]!.name;
    };
    showTone();
    cycleTone = (): void => {
        if (toneBusy) return;
        toneBusy = true;
        toneIndex = (toneIndex + 1) % TONE_MAPPINGS.length;
        const next = TONE_MAPPINGS[toneIndex]!;
        showTone();
        void setSceneImageProcessing(scene, next.tm ? { toneMappingEnabled: true, toneMapping: next.tm } : { toneMappingEnabled: false })
            .catch((err: unknown) => {
                // eslint-disable-next-line no-console
                console.warn("[aquanova] tone mapping switch failed", err);
            })
            .finally(() => {
                toneBusy = false;
            });
    };
    scene.imageProcessing.exposure = resolveExposure(manifest?.environment?.exposure);

    // ── Havok physics: clean per-chunk box-shell colliders (see buildShipColliders) ───────
    const hknp = await HavokPhysics({ locateFile: () => "/HavokPhysics.wasm" });
    const world = createHavokWorld(scene, hknp, { x: 0, y: -9.8, z: 0 });
    // FIXED simulation step. With the default variable step the world advances by the real frame
    // delta, so how far the player capsule sinks before the solver stops it depends on how slow the
    // first frames after load happen to be — and it stays wherever it came to rest. Measured on the
    // same build: the capsule settles at y 0.909 at full speed but 1.036 under 3x CPU throttling, i.e.
    // a 13 cm eye-height difference between machines. Babylon.js fixes its step at 1/60 for the same
    // reason. This also keeps the dynamic props' rest poses reproducible.
    setPhysicsTimestepMs(world, 1000 / 60);
    /** Manifest-authored collision primitives that became static bodies (QA/debug). */
    let manifestShapes: WorldCollisionShape[] = [];
    // The raw ship geometry is one glTF hierarchy of 568 overlapping modules; a trimesh collider
    // built from it pins the character controller on countless coplanar/overlapping triangles.
    // Instead we collide against clean per-chunk box shells from the manifest — smooth for the
    // capsule, and a natural fit since each chunk is already an axis-aligned room.
    if (manifest) {
        // Per-element collision from the manifest: every placed module that has an authored shape
        // becomes a solid static body, so props are collidable without anyone marking them
        // `dynamic` — that flag now means only "Havok may MOVE this", nothing else.
        //
        // Dissolvable placements are skipped: they already get their own body in the dynamic-prop
        // pass below (from the same manifest shape), and that body is removed when the prop melts.
        // A second static body here would outlive the melt as an invisible wall. Matched by INSTANCE
        // ID from the glTF node's extras, never by name — names repeat across placements.
        manifestShapes = buildManifestColliders(
            world,
            [...placementById].filter(([id]) => !behaviorManager.dissolvableInstanceIds.has(id)).flatMap(([, p]) => p.shapes)
        );
        canvas.dataset.manifestColliders = String(manifestShapes.length);
    } else {
        // Fallback floor so the player at least stands if the manifest failed to load.
        const node = createTransformNode("shipFloor", -30, -0.2, 0);
        const shape = createPhysicsShape(world, { type: PhysicsShapeType.BOX, parameters: { extents: { x: 140, y: 0.4, z: 24 } } });
        setPhysicsBodyShape(world, createPhysicsBody(world, node, PhysicsMotionType.STATIC), shape);
    }

    // ── First-person player at the manifest spawn ─────────────────────────────────────────
    const CAP_H = 1.8;
    const CAP_R = 0.4;
    const FLUID_CAP_R = CAP_R * 2;
    const EYE = 0.62; // camera offset above the capsule centre → ~1.5 m eye level
    // Marker present → stand on top of it; otherwise fall back to the legacy glTF-space spawn.
    const fallback = toLite(manifest?.spawns?.player ?? [-0.5, 0, 0]);
    const pMin = playerMarker?.min;
    const pMax = playerMarker?.max;
    const sx = pMin && pMax ? (pMin[0]! + pMax[0]!) / 2 : fallback[0];
    const sz = pMin && pMax ? (pMin[2]! + pMax[2]!) / 2 : fallback[2];
    const sy = pMax ? pMax[1]! + CAP_H / 2 + 0.1 : CAP_H / 2 + 0.1;
    const character = createPhysicsCharacterController(world, { x: sx, y: sy, z: sz }, { capsuleHeight: CAP_H, capsuleRadius: CAP_R });
    const capsuleHeight = (): number => character.shapeOptions.capsuleHeight ?? CAP_H;
    const canStand = (): boolean => {
        const currentHeight = capsuleHeight();
        if (currentHeight >= CAP_H - 1e-4) return true;
        const position = character.getPosition();
        const footY = position.y - currentHeight * 0.5;
        const fromY = footY + currentHeight + 1e-3;
        const toY = footY + CAP_H;
        const ring = CAP_R * 0.85;
        const diagonal = ring / Math.SQRT2;
        const offsets: ReadonlyArray<readonly [number, number]> = [
            [0, 0],
            [ring, 0],
            [-ring, 0],
            [0, ring],
            [0, -ring],
            [diagonal, diagonal],
            [diagonal, -diagonal],
            [-diagonal, diagonal],
            [-diagonal, -diagonal],
        ];
        return offsets.every(([dx, dz]) => !physicsRaycast(world, { x: position.x + dx, y: fromY, z: position.z + dz }, { x: position.x + dx, y: toY, z: position.z + dz }).hasHit);
    };

    // ── Dynamic (dissolvable) props ──────────────────────────────────────────────────────
    // A liquefiable prop needs three things: a Havok body so it is solid, a display root it can be
    // posed through while it melts, and its own bounds for mass/inertia and the particle fill.
    //
    // It no longer needs a BAKED SDF. The fluid collides against the ship's authored collision
    // primitives (see buildCollisionSet), so there are no distance grids anywhere in this demo — no
    // bake at load, no per-room grid, no arena to size, and none of the sign problems that came with
    // baking kit-bashed geometry (an open shell has no reliable inside).
    const MIRROR: [number, number, number] = [-1, 1, 1];

    // One body per NODE, not per mesh: a glTF node with several primitives is ONE object to the
    // player, so treating each primitive separately gave it that many independent rigid bodies and
    // the parts drifted apart the moment physics ran.
    const dynGroups: Mesh[][] = [];
    const grouped = new Set<Mesh>();
    for (const m of behaviorManager.dynamicMeshes) {
        if (grouped.has(m)) {
            continue;
        }
        const group = (nodePrimitives.get(m) ?? [m]).filter((p) => behaviorManager.dynamicMeshes.has(p));
        for (const p of group) {
            grouped.add(p);
        }
        dynGroups.push(group.length ? group : [m]);
    }
    const dynBounds = new Map<Mesh, MeshGroupBounds>(); // keyed by the group's FIRST primitive
    for (const group of dynGroups) {
        const b = meshGroupBounds(group);
        if (b) {
            dynBounds.set(group[0]!, b);
        }
    }

    interface DynBody {
        proxy: SceneNode;
        /** Null for a decal: it dissolves and bounds the fluid, but the surface it sits on collides. */
        body: PhysicsBody | null;
        /** Representative primitive (the group's first) — what call sites that want "the" mesh use. */
        mesh: Mesh;
        /** EVERY primitive of the node, all reparented under `disp` so they move as one rigid body. */
        meshes: Mesh[];
        /** World bounds of meshes at rest. */
        bounds: MeshGroupBounds;
        /** Display-root offset in its own frame, applied when the body is posed. */
        dispOffset: [number, number, number];
        /** Whether Havok may move this body (manifest `dynamic` behavior). Immovable ones are STATIC. */
        movable: boolean;
        /** The manifest placement this prop came from, so its debug shape can be dropped on melt. */
        instanceId: string | undefined;
        disp: SceneNode;
        wriggling?: boolean; // true while THIS body is dissolving: its Havok body still collides/supports,
        // but the wriggle owns the display node — so skip the pose sync below.
    }
    const dynBodies: DynBody[] = [];
    const dynBodyByMesh = new Map<Mesh, DynBody>();
    /** Placements whose prop has melted — their colliders and debug shapes are gone. */
    const removedPlacements = new Set<string>();

    // ── The fluid's collision set ────────────────────────────────────────────────────────────────
    // The same authored primitives the player collides against, handed to the fluid as an analytic
    // field instead of a baked SDF grid. `worldAabb` is only used to pick which ones a given
    // simulation needs; the solver evaluates the exact primitive.
    const shapeToPrimitive = (s: WorldCollisionShape): FluidPrimitive =>
        s.kind === "box"
            ? { kind: "box", a: s.centre, b: s.halfExtents ?? [0.05, 0.05, 0.05], rotation: s.rotation ?? [0, 0, 0, 1] }
            : s.kind === "sphere"
              ? { kind: "sphere", a: s.centre, radius: s.radius ?? 0.05 }
              : { kind: s.kind, a: s.pointA ?? s.centre, b: s.pointB ?? s.centre, radius: s.radius ?? 0.05 };
    const primAabb = (p: FluidPrimitive): { min: [number, number, number]; max: [number, number, number] } => {
        if (p.kind === "box") {
            // A rotated box's AABB is bounded by the sum of its half extents — coarse but never too
            // small, which is what matters for a selection test.
            const r = Math.hypot(...(p.b ?? [0, 0, 0]));
            return { min: [p.a[0] - r, p.a[1] - r, p.a[2] - r], max: [p.a[0] + r, p.a[1] + r, p.a[2] + r] };
        }
        const r = p.radius ?? 0;
        const b = p.b ?? p.a;
        const lo = (i: number): number => Math.min(p.a[i]!, b[i]!) - r;
        const hi = (i: number): number => Math.max(p.a[i]!, b[i]!) + r;
        return { min: [lo(0), lo(1), lo(2)], max: [hi(0), hi(1), hi(2)] };
    };
    for (const group of dynGroups) {
        const bounds = dynBounds.get(group[0]!);
        if (!bounds) {
            continue;
        }
        const { centre } = bounds;
        // Display root at the prop's rest pose (the glTF root mirror preserved via `scaling`);
        // reparent EVERY primitive of the node under it (world-preserving) so the whole node is posed
        // through one root and the parts keep their relative placement.
        const r = createTransformNode(`dyn_disp_${dynBodies.length}`);
        r.scaling.set(MIRROR[0], MIRROR[1], MIRROR[2]);
        r.position.set(centre[0] - MIRROR[0] * centre[0], centre[1] - MIRROR[1] * centre[1], centre[2] - MIRROR[2] * centre[2]);
        addToScene(scene, r);
        for (const m of group) {
            setParent(m, r);
        }
        // Havok proxy. Its shape comes from the manifest — that is the ONLY thing that decides
        // whether a prop is solid. A prop with no authored collision (the "Caution" sticker on the
        // door) simply gets no rigid body: the surface it is applied to already collides. Nothing
        // is inferred from the element's kind or module path.
        // STATIC unless the manifest assigns the `dynamic` behavior — a fixture stays put no matter
        // how hard the player runs into it, while a genuinely loose prop still falls and can be shoved.
        const movable = group.some((m) => behaviorManager.movableMeshes.has(m));
        const proxy = createTransformNode(`dyn_proxy_${dynBodies.length}`, centre[0], centre[1], centre[2]);
        const instanceId = instanceIdOfMesh(group[0]!);
        const authored = instanceId ? placementById.get(instanceId)?.shapes[0] : undefined;
        // The proxy sits at the prop's bounds centre and the authored shape is offset WITHIN the body,
        // so the display root (posed from the same pose) and the collider stay in agreement.
        let body: PhysicsBody | null = null;
        if (authored) {
            body = createPhysicsBody(world, proxy, movable ? PhysicsMotionType.DYNAMIC : PhysicsMotionType.STATIC, true);
            setPhysicsBodyShape(world, body, createWorldCollisionShape(world, authored, centre));
        }
        // The display root's offset in its OWN frame, so a moved body poses it as
        // `position = pose ∘ (−displayScale · centre)` — the transform the floating-body system used
        // to apply, kept identical so a melting prop still lines up with its water.
        const dispOffset: [number, number, number] = [-MIRROR[0] * centre[0], -MIRROR[1] * centre[1], -MIRROR[2] * centre[2]];
        const dyn: DynBody = { proxy, body, mesh: group[0]!, meshes: group, bounds, movable, instanceId, disp: r, dispOffset };
        dynBodies.push(dyn);
        for (const m of group) {
            dynBodyByMesh.set(m, dyn);
        }
    }

    // Each physics step, pose a MOVABLE prop's display root from its Havok body, and refresh the
    // primitive the fluid collides against. An immovable prop never moves, so it costs nothing.
    behaviorManager.events.on("physicsStep", () => {
        for (const d of dynBodies) {
            if (!d.movable) continue; // fixture: posed once at load
            if (!d.body) continue; // decal: no rigid body to read a pose from
            if (d.wriggling) continue; // dissolving: the wriggle owns the display
            const p = d.proxy.position;
            const q = d.proxy.rotationQuaternion;
            const off = qRot(q, d.dispOffset);
            d.disp.position.set(p.x + off[0], p.y + off[1], p.z + off[2]);
            d.disp.rotationQuaternion.set(q.x, q.y, q.z, q.w);
        }
        // Movable props also move the fluid's boundary: rewrite just their slots in each running
        // sim's primitive buffer. Only the changed primitives are uploaded, not the whole set.
        for (const a of activeSims) {
            for (const m of a.collision.moving) {
                if (a.collision.prims[m.slot]?.active === false) continue;
                const live = livePrim(m.body);
                if (!live) continue;
                a.collision.prims[m.slot] = live;
                packPrimitive(a.collision.scratch, m.slot, live);
                device.queue.writeBuffer(a.collision.buffer, (PRIM_HEADER + m.slot * PRIM_STRIDE) * 4, a.collision.scratch, PRIM_HEADER + m.slot * PRIM_STRIDE, PRIM_STRIDE);
            }
        }
    });
    canvas.dataset.dynBodies = String(dynBodies.length);

    // Static collision primitives for the fluid: every authored placement EXCEPT the dissolvable
    // ones, which move (and later vanish) and so are tracked per body below.
    const staticPrims: Array<{ id: string; prim: FluidPrimitive; min: [number, number, number]; max: [number, number, number] }> = [];
    for (const [id, pl] of placementById) {
        if (behaviorManager.dissolvableInstanceIds.has(id)) continue;
        for (const s of pl.shapes) {
            const prim = shapeToPrimitive(s);
            staticPrims.push({ id, prim, ...primAabb(prim) });
        }
    }
    canvas.dataset.fluidStaticPrims = String(staticPrims.length);

    /** A dissolvable prop's primitive, kept in its proxy's REST frame so a moved body can re-pose it. */
    const dynPrims = new Map<DynBody, { local: FluidPrimitive; rest: [number, number, number] }>();
    const dynPrimsSkipped: string[] = [];
    for (const d of dynBodies) {
        const s = d.instanceId ? placementById.get(d.instanceId)?.shapes[0] : undefined;
        if (!s) {
            // No authored collision → nothing for the fluid to collide with either. Reported because
            // it is otherwise invisible: the prop still blocks the PLAYER (Havok) while water pours
            // straight through it, which reads as a fluid bug rather than missing authoring.
            dynPrimsSkipped.push(`${nodeNameOfMesh.get(d.mesh) ?? d.mesh.name}(${d.instanceId ?? "no-id"})`);
            continue;
        }
        const c = d.bounds.centre;
        const p = shapeToPrimitive(s);
        // Express relative to the proxy's rest position; the proxy starts unrotated at `centre`.
        // localizePrimitive knows a box's `b` is half extents and must not be re-based — see its doc.
        dynPrims.set(d, { local: localizePrimitive(p, c), rest: [c[0], c[1], c[2]] });
    }
    if (dynPrimsSkipped.length) {
        // eslint-disable-next-line no-console
        console.warn(`[aquanova] ${dynPrimsSkipped.length} dynamic prop(s) have NO fluid collision primitive: ${dynPrimsSkipped.join(", ")}`);
    }
    canvas.dataset.fluidDynPrims = `${dynPrims.size}/${dynBodies.length}`;

    /** Hamilton product — compose two rotations (`a` applied after `b`). */
    const quatMul = (a: readonly [number, number, number, number], b: readonly [number, number, number, number]): [number, number, number, number] => [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ];
    /** Rotate `v` by quaternion `q`. */
    const qRot = (q: { x: number; y: number; z: number; w: number }, v: readonly [number, number, number]): [number, number, number] => {
        const tx = 2 * (q.y * v[2] - q.z * v[1]);
        const ty = 2 * (q.z * v[0] - q.x * v[2]);
        const tz = 2 * (q.x * v[1] - q.y * v[0]);
        return [v[0] + q.w * tx + (q.y * tz - q.z * ty), v[1] + q.w * ty + (q.z * tx - q.x * tz), v[2] + q.w * tz + (q.x * ty - q.y * tx)];
    };
    /** A dissolvable prop's primitive at its CURRENT pose, with the velocity the solver needs. */
    const livePrim = (d: DynBody): FluidPrimitive | null => {
        const e = dynPrims.get(d);
        if (!e) return null;
        const p = d.proxy.position;
        const q = d.proxy.rotationQuaternion;
        const put = (v: readonly [number, number, number]): [number, number, number] => {
            const r = qRot(q, v);
            return [p.x + r[0], p.y + r[1], p.z + r[2]];
        };
        const vel = d.movable && d.body ? getPhysicsBodyLinearVelocity(world, d.body) : { x: 0, y: 0, z: 0 };
        const base = e.local;
        const out: FluidPrimitive = { ...base, a: put(base.a), velocity: [vel.x, vel.y, vel.z] };
        if (base.b) out.b = base.b && base.kind !== "box" ? put(base.b) : base.b; // a box's `b` is half extents, not a point
        if (base.kind === "box") out.rotation = quatMul([q.x, q.y, q.z, q.w], base.rotation ?? [0, 0, 0, 1]);
        return out;
    };

    let playerBehavior: PlayerBehavior | null = null;
    const livePlayerPrimitive = (): FluidPrimitive => {
        const p = character.getPosition();
        const velocity = character.getVelocity();
        const axisHalf = capsuleHeight() * 0.5 - CAP_R;
        return {
            kind: "capsule",
            a: [p.x, p.y + axisHalf, p.z],
            // Extend only the fluid boundary below the physical capsule so walking through shallow
            // fluid displaces it instead of merely grazing its surface at floor level.
            b: [p.x, p.y - axisHalf - CAP_R * 0.5, p.z],
            radius: FLUID_CAP_R,
            velocity: [velocity.x, velocity.y, velocity.z],
            active: !(playerBehavior?.isNoclip ?? false),
        };
    };

    // Which chunk (room) contains the camera. Camera is Lite-space; convert X back to glTF to test
    // against the manifest AABBs (which are glTF-space).
    const roomAt = (): string => {
        const gx = -cam.position.x;
        const gz = cam.position.z;
        return chunkAt(manifest?.chunks ?? [], gx, gz)?.id ?? "—";
    };

    // ── Debug overlays (I inspect, B colliders, L lights, F/Shift+F SDF) — see ./debug/ ─────────
    // Gated on LAB_DEBUG so a release bundle folds these to `null` and drops the modules entirely.
    let picker: ReturnType<typeof createGpuPicker> | null = null;
    const getPicker = (): ReturnType<typeof createGpuPicker> => (picker ??= createGpuPicker(scene));
    const inspectOverlay = !LAB_DEBUG
        ? null
        : createInspectOverlay({
              engine,
              scene,
              canvas,
              world,
              cam,
              character,
              getCapsuleHeight: capsuleHeight,
              roomAt,
              getPicker,
              nodeNameOf: (m: Mesh) => nodeNameOfMesh.get(m),
              nodePrimitivesOf: (m: Mesh) => nodePrimitives.get(m),
              isNoclip: () => playerBehavior?.isNoclip ?? false,
          });
    // Fluid GPU profiling. The sim is stepped directly onto the frame encoder (not as a frame-graph
    // task), so the engine's per-task timer structurally cannot see it; the fluid's own profiler is
    // the only way to attribute that time. Created once if the device supports timestamp queries,
    // but only ATTACHED while the perf panel is open — detached, `profiler?.pass()` returns
    // undefined and no pass requests timestamp writes, so it costs nothing.
    let fluidProfiler: FluidProfilerImpl | null = null;
    try {
        fluidProfiler = engine._device.features.has("timestamp-query") ? createFluidProfiler(engine._device) : null;
    } catch {
        fluidProfiler = null; // never worth failing the demo over a profiler
    }
    let fluidProfilerOn = false;
    const perfOverlay = createPerfOverlay({
        engine,
        fluidWorkload: () => ({
            simulations: activeSims.length,
            particles: activeSims.reduce((total, active) => total + active.sim.count, 0),
        }),
        fluidStages: () => (fluidProfilerOn ? (fluidProfiler?.results() ?? null) : null),
        onToggle: (on) => {
            fluidProfilerOn = on && fluidProfiler !== null;
            attachFluidProfiler();
        },
    });
    const colliderOverlay = !LAB_DEBUG
        ? null
        : createColliderOverlay({
              engine,
              scene,
              canvas,
              manifestShapes: manifestPlacements,
              isRemoved: (id) => removedPlacements.has(id),
              injectedPrims: () =>
                  activeSims.map((a) => ({
                      sim: nodeNameOfMesh.get(a.mesh) ?? a.mesh.name,
                      prims: a.collision.prims.filter((primitive) => primitive.active !== false),
                  })),
              dynBodies: () => dynBodies.map((d) => ({ name: nodeNameOfMesh.get(d.mesh) ?? d.mesh.name, position: d.proxy.position, half: d.bounds.half })),
              roomAt,
          });
    const lightUtility = !LAB_DEBUG ? null : createUtilityLayer(engine, scene, { addDefaultLight: false });
    const lightOverlay = !LAB_DEBUG
        ? null
        : createLightOverlay({
              engine,
              scene: lightUtility!.scene,
              canvas,
              lights: lights.lights,
              roomAt,
          });
    const inspectOn = (): boolean => inspectOverlay?.isOn() ?? false;

    // Assigned by the Milestone D fluid block below. Liquefiable behaviors call this service after
    // accepting a typed `hitWithWeapon` event addressed to their mesh.
    let liquefyMesh: (mesh: Mesh, hitPoint?: readonly [number, number, number] | null, config?: LiquefiableBehaviorConfig) => void = () => {};
    let requestFusionResume: () => number | null = () => null;
    let resolveFusionResume: (token: number, mesh: Mesh | null) => "resumed" | "start-new" | "continue" = () => "continue";
    let reverseFusion: () => boolean = () => false;
    // G key — swap the fluid's collision between the full ship SDF and a bare ground plane (the
    // Liquefactor demo's collision), so the two can be compared with only that variable changed.
    // Assigned by the same block, which owns both SDF specs.
    let toggleGroundOnly: () => void = () => {};
    // M key — toggle 4× MSAA on the scene pass. Assigned by the render-path block below (which owns
    // the MSAA target and its tasks); declared here because the key handler is registered first.
    let toggleMsaa: () => void = () => {};
    let toggleSmaa: () => void = () => {};
    let toggleTaa: () => void = () => {};
    let toggleSpecularAA: () => void = () => {};
    const toggleNearestLight = (): void => {
        const position = character.getPosition();
        const result = lights.toggleNearest([position.x, position.y, position.z]);
        canvas.dataset.nearestLight = result?.id ?? "none";
        canvas.dataset.nearestLightEnabled = result ? String(result.enabled) : "";
        const state = document.getElementById("nearestLightState");
        if (state) state.textContent = result ? `${result.id} ${result.enabled ? "on" : "off"}` : "none";
    };
    let cycleSsaa: () => void = () => {};
    /** Re-applies the fluid profiler to the surface task and every live sim. Assigned once they exist. */
    let attachFluidProfiler: () => void = () => {};
    let retargetTaa: (task: RenderTask) => void = () => {};
    // Assigned once each subsystem exists. Split from the toggles so the three modes can switch each
    // other off without recursing: a setter never touches another mode, only the toggles do.
    let setMsaaEnabled: (on: boolean) => void = () => {};
    let setSmaaEnabled: (on: boolean) => void = () => {};
    let setTaaEnabled: (on: boolean) => void = () => {};

    behaviorManager.start({
        canvas,
        camera: cam,
        character,
        capsuleHeight: CAP_H,
        eyeHeight: EYE,
        canStand,
        getPicker,
        nodeNameOf: (mesh) => nodeNameOfMesh.get(mesh) ?? mesh.name,
        isLiquefiable: (mesh) => behaviorManager.isLiquefiable(mesh),
        isInspecting: inspectOn,
        inspectAt: (x, y) => inspectOverlay?.pickAt(x, y),
        requestFusionResume: () => requestFusionResume(),
        resolveFusionResume: (token, mesh) => resolveFusionResume(token, mesh),
        reverseFusion: () => {
            reverseFusion();
        },
        liquefy: (mesh, point, config) => liquefyMesh(mesh, point, config),
    });
    playerBehavior = behaviorManager.player;
    if (!playerBehavior) {
        // eslint-disable-next-line no-console
        console.warn("[aquanova] manifest has no player behavior; first-person controls are disabled");
    }

    window.addEventListener("keydown", (e) => {
        if (e.code === "KeyI" && !e.repeat) inspectOverlay?.toggle(); // I → toggle inspect overlay
        if (e.code === "KeyB" && !e.repeat) colliderOverlay?.cycle(); // B → toggle collision-box overlay
        if (e.code === "KeyL" && !e.repeat) lightOverlay?.toggle(); // L → runtime lights for current chunk
        if (e.code === "KeyG" && !e.repeat) toggleGroundOnly(); // G → fluid collides with ground only
        if (e.code === "KeyM" && !e.repeat) toggleMsaa(); // M → toggle 4x MSAA (anti-aliasing)
        if (e.code === "KeyN" && !e.repeat) toggleSmaa(); // N → toggle SMAA post-process
        if (e.code === "KeyT" && !e.repeat) toggleTaa(); // T → toggle TAA (only accumulates while still)
        if (e.code === "KeyK" && !e.repeat) toggleSpecularAA(); // K → toggle PBR specular AA
        if (e.code === "KeyH" && !e.repeat) toggleNearestLight(); // H → toggle nearest positional light
        if (e.code === "KeyR" && !e.repeat) cycleSsaa(); // R → cycle SSAA render scale
        if (e.code === "KeyP" && !e.repeat) perfOverlay.toggle(); // P → FPS + per-task GPU timing
        if (e.code === "KeyO" && !e.repeat) cycleTone(); // O → cycle tone mapping
    });

    // Test hook (QA): read the player position + nudge look/movement programmatically.
    (window as unknown as { __aquanova?: unknown }).__aquanova = {
        getPos: (): { x: number; y: number; z: number } => playerBehavior?.getPosition() ?? character.getPosition(),
        // Both SNAP (target and current together) rather than easing like the mouse does: a test that
        // aims and then immediately fires must not have the shot land wherever the smoothing had got to.
        setYaw: (y: number): void => {
            playerBehavior?.setYaw(y);
        },
        look: (dx: number, dy: number): void => {
            playerBehavior?.look(dx, dy);
        },
        press: (code: string): void => {
            playerBehavior?.press(code);
        },
        release: (code: string): void => {
            playerBehavior?.release(code);
        },
        isCrouched: (): boolean => playerBehavior?.isCrouched ?? false,
        capsuleHeight,
        toggleNoclip: (): void => playerBehavior?.toggleNoclip(),
        behaviors: (): Array<{ name: string; mesh: string }> => behaviorManager.describeInstances(),
        /** Graphics settings (for the future config page + QA). `setMsaa` is idempotent. */
        graphics: (): Record<string, unknown> => ({ ...graphics, msaaActive: msaaOn, smaaActive: smaaOn, taaActive: taaOn }),
        setMsaa: (on: boolean): void => {
            if (on !== msaaOn) toggleMsaa();
        },
        setSmaa: (on: boolean): void => {
            if (on !== smaaOn) toggleSmaa();
        },
        setTaa: (on: boolean): void => {
            if (on !== taaOn) toggleTaa();
        },
        setSpecularAA: (on: boolean): void => {
            if (on !== graphics.specularAA) toggleSpecularAA();
        },
        pbrMaterials: (): Array<{ name: string; roughnessFactor: number; specularAA: boolean }> =>
            shipPbrMaterials().map((material) => ({
                name: material.name ?? "",
                roughnessFactor: material.roughnessFactor ?? 1,
                specularAA: material.enableSpecularAA === true,
            })),
        /** SSAA render scale (1 = off). Applied immediately; the engine picks up the size next frame. */
        setSsaa: (scale: number): void => {
            graphics.ssaa = scale;
            saveGraphicsSettings(graphics);
            applySsaa(scale);
        },
        /** SMAA edge threshold — lower catches more edges. Live, for tuning and the config page. */
        setSmaaThreshold: (t: number): void => {
            smaaTask.threshold = t;
            smaaTask.updateUniforms();
        },
        /** SMAA pattern-search length in pixels — longer reconstructs shallower edges. */
        setSmaaSearchSteps: (n: number): void => {
            smaaTask.maxSearchSteps = n;
            smaaTask.updateUniforms();
        },
        /** SMAA blending rule: dominant-axis (canonical) vs all four neighbours. For A/B. */
        setSmaaDominantAxis: (on: boolean): void => {
            smaaTask.dominantAxisBlend = on;
            smaaTask.updateUniforms();
        },
        /** Effective SMAA parameters after clamping — lets QA assert hostile inputs were rejected. */
        smaaParams: (): Record<string, number | boolean> => ({
            threshold: smaaTask.threshold,
            maxSearchSteps: smaaTask.maxSearchSteps,
            diagonalDetection: smaaTask.diagonalDetection,
            minDiagonalRun: smaaTask.minDiagonalRun,
            sourceIsSrgb: smaaTask.sourceIsSrgb,
            dominantAxisBlend: smaaTask.dominantAxisBlend,
        }),
        /** SMAA 45-degree pattern detection on/off. Live, for A/B comparison. */
        setSmaaDiagonal: (on: boolean, minRun?: number): void => {
            smaaTask.diagonalDetection = on;
            if (minRun !== undefined) smaaTask.minDiagonalRun = minRun;
            smaaTask.updateUniforms();
        },
        toggleInspect: (): void => inspectOverlay?.toggle(),
        /** FPS + per-task GPU timing overlay (P). */
        togglePerf: (): void => perfOverlay.toggle(),
        /** Cycle tone mapping (O). Recompiles PBR pipelines, so it settles a frame or two later. */
        cycleTone: (): void => cycleTone(),
        /** Active tone-mapping algorithm name. */
        toneMapping: (): string => canvas.dataset.tonemap ?? "?",
        toggleColliders: (): void => colliderOverlay?.cycle(),
        toggleLights: (): void => lightOverlay?.toggle(),
        toggleNearestLight,
        runtimeLights: () => lights.lights,
        runtimeLitMeshes: (): string[] =>
            allShipMeshes
                .filter((mesh) => !!(mesh.material as { _clusteredLightState?: unknown } | undefined)?._clusteredLightState)
                .map((mesh) => nodeNameOfMesh.get(mesh) ?? mesh.name),
        /** Manifest-authored collision primitives that became static bodies. */
        manifestColliders: (): Array<{ kind: string; c: number[]; r?: number; he?: number[] }> =>
            manifestShapes.map((s) => ({
                kind: s.kind,
                c: [...s.centre],
                ...(s.radius !== undefined ? { r: s.radius } : {}),
                ...(s.halfExtents ? { he: [...s.halfExtents] } : {}),
            })),
        /** Node name → the authored shape used for its rigid body (empty when it has none). */
        shapedNodes: (): Array<{ node: string; kind: string; c: number[] }> => manifestPlacements.map(({ node, shape }) => ({ node, kind: shape.kind, c: [...shape.centre] })),
        /** What each RUNNING simulation was handed: the primitives packed into its shader's buffer. */
        injectedPrims: (): Array<{ sim: string; n: number; kinds: Record<string, number>; moving: number }> =>
            activeSims.map((a) => {
                const kinds: Record<string, number> = {};
                const activePrims = a.collision.prims.filter((primitive) => primitive.active !== false);
                for (const p of activePrims) kinds[p.kind] = (kinds[p.kind] ?? 0) + 1;
                const moving = a.collision.moving.filter(({ slot }) => a.collision.prims[slot]?.active !== false).length;
                return { sim: nodeNameOfMesh.get(a.mesh) ?? a.mesh.name, n: activePrims.length, kinds, moving };
            }),
        /** Instance ids whose prop has melted — their colliders and debug shapes are gone. */
        removedPlacements: (): string[] => [...removedPlacements],
        /** Full primitive data each running sim collides against — the exact values packed for the
         *  shader. Verbose, so kept separate from the `injectedPrims` summary. */
        fluidPrims: (): Array<{ sim: string; prims: FluidPrimitive[] }> => activeSims.map((a) => ({ sim: nodeNameOfMesh.get(a.mesh) ?? a.mesh.name, prims: a.collision.prims })),
        /** Particles currently inside a world-space AABB. The direct test of whether the fluid is
         *  respecting a collider: a solid box should contain (almost) none. */
        waterInAabb: async (min: [number, number, number], max: [number, number, number]): Promise<{ total: number; inside: number }> => {
            let total = 0;
            let inside = 0;
            for (const a of activeSims) {
                const n = a.sim.count;
                const bytes = n * 16;
                const rb = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
                const enc = device.createCommandEncoder();
                enc.copyBufferToBuffer(a.sim.positionBuffer, 0, rb, 0, bytes);
                device.queue.submit([enc.finish()]);
                await rb.mapAsync(GPUMapMode.READ);
                const f = new Float32Array(rb.getMappedRange().slice(0));
                rb.unmap();
                rb.destroy();
                for (let i = 0; i < n; i++) {
                    const x = f[i * 4]!;
                    const y = f[i * 4 + 1]!;
                    const z = f[i * 4 + 2]!;
                    total++;
                    if (x >= min[0] && x <= max[0] && y >= min[1] && y <= max[1] && z >= min[2] && z <= max[2]) inside++;
                }
            }
            return { total, inside };
        },
        roomAt: (): string => roomAt(),
        camState: (): Record<string, number> => ({
            px: cam.position.x,
            py: cam.position.y,
            pz: cam.position.z,
            tx: cam.target.x,
            ty: cam.target.y,
            tz: cam.target.z,
            fov: (cam as unknown as { fov?: number }).fov ?? -1,
            near: cam.nearPlane,
            far: cam.farPlane,
        }),
        grading: (): Record<string, unknown> => ({
            exposure: scene.imageProcessing.exposure,
            contrast: scene.imageProcessing.contrast,
            toneMapping: scene.imageProcessing.toneMapping,
            toneMappingEnabled: scene.imageProcessing.toneMappingEnabled,
            ibl: (allShipMeshes.find((m) => m.material)?.material as unknown as { environmentIntensity?: number } | undefined)?.environmentIntensity,
        }),
        fire: (): void => {
            void playerBehavior?.fire();
        },
        setFusionPressed: (pressed: boolean, targetName?: string): boolean => {
            if (!pressed) return reverseFusion();
            const token = requestFusionResume();
            if (token === null) return false;
            if (token === 0) return true;
            const target = targetName ? (meshesByNodeName.get(targetName)?.[0] ?? null) : null;
            const result = resolveFusionResume(token, target);
            if (result === "start-new" && target) {
                liquefyMesh(target, null, behaviorManager.getLiquefiableConfig(target));
            }
            return result !== "continue";
        },
        liqTargets: (): Array<{ name: string; c: number[] | null }> =>
            [...behaviorManager.liquefiableMeshes].map((m) => ({
                name: nodeNameOfMesh.get(m) ?? m.name,
                c: m.boundMin && m.boundMax ? [(m.boundMin[0] + m.boundMax[0]) / 2, (m.boundMin[1] + m.boundMax[1]) / 2, (m.boundMin[2] + m.boundMax[2]) / 2] : null,
            })),
        dynPos: (): Array<{ name: string; pos: number[]; half: number[]; movable: boolean; hasBody: boolean }> =>
            dynBodies.map((d) => ({
                name: nodeNameOfMesh.get(d.mesh) ?? d.mesh.name,
                pos: [d.proxy.position.x, d.proxy.position.y, d.proxy.position.z],
                half: d.bounds.half,
                movable: d.movable,
                hasBody: d.body !== null,
            })),
        liquefyNamed: (name: string, hit?: [number, number, number]): string | null => {
            // When a hit point is given, pick the nearest match: several props legitimately share a
            // node name (e.g. a stack of identical crates), so name alone is ambiguous.
            let best: Mesh | null = null;
            let bestD = Infinity;
            for (const m of behaviorManager.liquefiableMeshes) {
                if ((nodeNameOfMesh.get(m) ?? m.name) !== name) continue;
                if (!hit) return (liquefyMesh(m, null), name);
                const c: [number, number, number] =
                    m.boundMin && m.boundMax ? [(m.boundMin[0] + m.boundMax[0]) / 2, (m.boundMin[1] + m.boundMax[1]) / 2, (m.boundMin[2] + m.boundMax[2]) / 2] : [0, 0, 0];
                const d = (c[0] - hit[0]) ** 2 + (c[1] - hit[1]) ** 2 + (c[2] - hit[2]) ** 2;
                if (d < bestD) [bestD, best] = [d, m];
            }
            if (!best) return null;
            liquefyMesh(best, hit ?? null);
            return name;
        },
        warp: (x: number, y: number, z: number): void => playerBehavior?.warp(x, y, z),
        freeze: (): void => {
            playerBehavior?.freeze();
        },
        meltState: (): Array<{ name: string; phase: string; frontR: number; maxR: number }> =>
            activeSims.flatMap((a) =>
                a.members.map((member) => ({
                    name: nodeNameOfMesh.get(member.mesh) ?? member.mesh.name,
                    phase: a.phase,
                    frontR: member.state ? member.state.frontR : -1,
                    maxR: member.maxR,
                }))
            ),
        /** Read every live sim's particle positions back and report how the water sits relative to the
         *  floor. `below` counting anything but ~0 means the scene SDF has a hole the particles fell
         *  through, since the solver has no floor of its own. (Positions only — `velocityBuffer` is
         *  not COPY_SRC, and a rejected copy would invalidate the whole command buffer.) */
        waterStats: async (): Promise<Array<{ n: number; minY: number; below: number; spread: number }>> => {
            const out: Array<{ n: number; minY: number; below: number; spread: number }> = [];
            for (const a of activeSims) {
                const n = a.sim.count;
                const bytes = n * 16;
                const rb = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
                const enc = device.createCommandEncoder();
                enc.copyBufferToBuffer(a.sim.positionBuffer, 0, rb, 0, bytes);
                device.queue.submit([enc.finish()]);
                await rb.mapAsync(GPUMapMode.READ);
                const f = new Float32Array(rb.getMappedRange().slice(0));
                rb.unmap();
                rb.destroy();
                let minY = Infinity,
                    below = 0,
                    minX = Infinity,
                    maxX = -Infinity,
                    minZ = Infinity,
                    maxZ = -Infinity;
                for (let i = 0; i < n; i++) {
                    const x = f[i * 4]!,
                        y = f[i * 4 + 1]!,
                        z = f[i * 4 + 2]!;
                    if (y < -1e4) continue; // dormant particle parked off-screen
                    if (y < minY) minY = y;
                    if (y < FLOOR_Y - 1e-3) below++;
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (z < minZ) minZ = z;
                    if (z > maxZ) maxZ = z;
                }
                out.push({ n, minY, below, spread: Math.max(maxX - minX, maxZ - minZ) });
            }
            return out;
        },
        /** G-key state: true when the fluid collides with a bare ground plane instead of the ship. */
        groundOnly: (): boolean => groundOnly,
        setGroundOnly: (on: boolean): void => {
            if (on !== groundOnly) toggleGroundOnly();
        },
        /** What the last liquefaction actually took from its fluidSim file: the physics the solver
         *  was built with, the render block pushed onto the shared surface pass, and the foam block
         *  (parsed and held, but not consumed until a foam pass exists). */
        fluidSetting: (): Record<string, unknown> => ({
            name: activeSettingName ?? null,
            render: activeRender ?? null,
            foam: activeFoam ?? null,
            impulse: activeImpulse ?? null,
            grid: activeGrid ?? null,
            physics: activePhysics ?? null,
        }),
        camTo: (px: number, py: number, pz: number, tx: number, ty: number, tz: number): void => {
            cam.position.set(px, py, pz);
            cam.target.set(tx, ty, tz);
        },
    };

    // ── Fluid render path: scene → sceneColorRT, then the fluid surface composites to the swapchain ─
    // (Liquefactor pattern. Single-sample scene target → MSAA is off on this path.) The scene colour
    // target bakes its OWN depth (dFormat) rather than taking an external depth attachment: with
    // transmissive materials + a skybox, the transmission base-task's opaque bundle reads its depth
    // format from the colour RT descriptor, so an externally-supplied depth would desync the bundle
    // from the pipelines and drop every draw. The surface samples this same baked depth for compositing.
    const device = engine._device;
    const sceneColorRT = createRenderTarget({ lbl: "aq-scene-color", format: engine.format, dFormat: "depth24plus", samples: 1, size: engine });
    const sceneTask = createRenderTask({ name: "scene", rt: sceneColorRT, clr: true, clrColor: scene.clearColor }, engine, scene);
    addTask(scene, sceneTask);

    // The fluid composite's output. It cannot go straight to the swapchain any more: SMAA has to
    // SAMPLE the finished image, and a swapchain texture is renderable but not sampleable. So the
    // composite lands here and one of two presenting tasks copies it to the screen — SMAA when it is
    // on, a plain blit when it is off. Both are always in the graph and gated, so toggling the
    // setting never rebuilds the frame graph.
    const presentRT = createRenderTarget({ lbl: "aq-present", format: engine.format, samples: 1, size: engine });

    // ── Optional 4× MSAA on the scene (ship geometry) pass ───────────────────────────────
    // The ship renders into an OFFSCREEN single-sample target because the fluid surface has to
    // SAMPLE it when compositing, so it gets none of the engine's own MSAA (`createEngine(...,
    // { msaaSamples: 1 })`) — every panel seam, railing and door frame is hard-aliased.
    //
    // Only the SCENE pass is multisampled. The fluid surface is reconstructed in screen space from
    // a depth/thickness buffer rather than rasterised, so MSAA would do nothing for it.
    // Multisampling the one pass that draws hard-edged triangles is where all the benefit is.
    //
    // Wiring: an MSAA colour+depth target is rendered instead of `sceneColorRT` and RESOLVES into
    // it (`rst`), so every downstream consumer (fluid surface, lit-particle colours) keeps reading
    // the same single-sample texture. Depth needs an explicit pass — WebGPU has no hardware depth
    // resolve — so `createDepthResolveTask` copies sample 0 of the MSAA depth into sceneColorRT's
    // own baked depth, which the surface composites against.
    //
    // Task order is [scene-msaa, scene, depth-resolve]: the plain scene task must record LAST of
    // the two so the colour view it bakes into its pass descriptor belongs to the live
    // `sceneColorRT` texture (both tasks build that target — one as `rt`, one as `rst`). Exactly
    // one of the two executes per frame, so the plain task never clears the depth the resolve wrote.
    //
    // Built LAZILY on first enable: the MSAA colour + depth pair is ~4× a normal canvas-sized pair,
    // so a player who leaves it off never pays for it.
    const MSAA_SAMPLES = 4;
    let msaaOn = false;
    let msaaSceneTask: Task | null = null;
    let pendingFrameGraphRebuild = false;
    const gateExistingTask = (task: Task, active: () => boolean): void => {
        const run = task.execute!.bind(task);
        task.execute = (): number => (active() ? run() : 0);
    };
    gateExistingTask(sceneTask, () => !msaaOn);
    const setMsaa = (on: boolean): void => {
        if (on === msaaOn) return;
        if (on && !msaaSceneTask) {
            const sceneMsaaRT = createRenderTarget({
                lbl: "aq-scene-msaa",
                format: engine.format,
                // Same depth format as sceneColorRT: the resolve copies between them, and the PBR
                // pipelines are keyed on the depth format they were built against.
                dFormat: "depth24plus",
                samples: MSAA_SAMPLES,
                size: engine,
            });
            msaaSceneTask = createRenderTask({ name: "scene-msaa", rt: sceneMsaaRT, rst: sceneColorRT, clr: true, clrColor: scene.clearColor }, engine, scene);
            const depthResolveTask = createDepthResolveTask({ name: "aq-depth-resolve", sourceTexture: sceneMsaaRT, targetTexture: sceneColorRT }, engine, scene);
            gateExistingTask(msaaSceneTask, () => msaaOn);
            gateExistingTask(depthResolveTask, () => msaaOn);
            addTaskBefore(scene, msaaSceneTask, sceneTask);
            addTaskAfter(scene, depthResolveTask, sceneTask);
            // Rebuild so the new tasks record and every canvas-sized target is re-allocated in the
            // right order. This destroys and re-creates textures that other tasks' bind groups and
            // pass descriptors point at, so it must not run from a UI/key event mid-frame — it is
            // deferred to the next frameStart event.
            pendingFrameGraphRebuild = true;
        }
        msaaOn = on;
        // TAA jitters the UBO of the task it points at, so it has to follow the active scene task.
        retargetTaa(on && msaaSceneTask ? (msaaSceneTask as RenderTask) : sceneTask);
    };
    setMsaaEnabled = (on: boolean): void => {
        if (on === msaaOn) return;
        graphics.msaa = on;
        setMsaa(on);
        saveGraphicsSettings(graphics);
        canvas.dataset.msaa = String(on);
        setHudFlag("msaaState", on);
    };
    toggleMsaa = (): void => {
        const next = !msaaOn;
        if (next) setTaaEnabled(false); // TAA is an alternative to MSAA, not a companion
        setMsaaEnabled(next);
    };
    // Apply the resolved setting (defaults ← saved ← ?msaa= override). Enabling here rather than at
    // target-creation time keeps a single code path: the first frame does the same deferred graph
    // rebuild a mid-session toggle does.
    setMsaa(graphics.msaa);
    canvas.dataset.msaa = String(graphics.msaa);
    setHudFlag("msaaState", graphics.msaa);

    // Where the ship was last drawn, plus that draw's matrices — the source for per-particle LIT
    // colours (see particle-lit-colors.ts). The scene target is written every frame before the fluid
    // composite, so at liquefy time it still holds the solid prop we are about to melt.
    const litScene = (): LitColorScene | null => {
        const rt = sceneColorRT as unknown as { _colorView: GPUTextureView | null; _depthView: GPUTextureView | null };
        // The scene colour/depth textures are allocated lazily on first render. Until then there is
        // no rendered image to promote the albedo against, so skip the lit pass for that shot rather
        // than binding null views (which throws inside createBindGroup).
        if (!rt._colorView || !rt._depthView) return null;
        const aspect = engine.canvas.width / Math.max(1, engine.canvas.height);
        const proj = getProjectionMatrix(cam, aspect);
        return {
            colorView: rt._colorView,
            depthView: rt._depthView,
            view: getViewMatrix(cam),
            viewProj: getViewProjectionMatrix(cam, aspect),
            projZW: proj[14]!,
            projZZ: proj[10]!,
        };
    };

    const MAX_TOTAL = 200000;
    const combinedPos = device.createBuffer({ label: "aq-combined-pos", size: MAX_TOTAL * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const combinedDebug = device.createBuffer({ label: "aq-combined-debug", size: MAX_TOTAL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const combinedAlpha = device.createBuffer({ label: "aq-combined-alpha", size: MAX_TOTAL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const alphaScratch = new Float32Array(MAX_TOTAL).fill(1);
    device.queue.writeBuffer(combinedAlpha, 0, alphaScratch);
    // Per-particle RGBA colour — filled per shot from the liquefied mesh's base-colour texture when its
    // fluidSim setting has useMeshColors (else the flat water tint below). Aggregated like combinedPos
    // each frame; the surface tints the water by it.
    const combinedColor = device.createBuffer({ label: "aq-combined-color", size: MAX_TOTAL * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // Non-mesh-colour fallback fill, kept in step with the active setting's `render.waterColor` by
    // applyRenderSetting. Only used before any setting has been applied (i.e. never on screen, since
    // water exists only after a liquefaction).
    const waterRgb: [number, number, number] = [0x16 / 255, 0xa3 / 255, 0xc3 / 255];
    const colorScratch = new Float32Array(MAX_TOTAL * 4);
    const fillColorScratch = (): void => {
        for (let i = 0; i < MAX_TOTAL; i++) {
            colorScratch[i * 4] = waterRgb[0];
            colorScratch[i * 4 + 1] = waterRgb[1];
            colorScratch[i * 4 + 2] = waterRgb[2];
            colorScratch[i * 4 + 3] = 1;
        }
    };
    fillColorScratch();
    const virtualSim = {
        count: 0,
        particleRadius: 0.08,
        surfaceSizeScale: 1,
        positionBuffer: combinedPos,
        velocityBuffer: combinedPos,
        debugBuffer: combinedDebug,
        debugNorm: 1,
        gpuBytes: 0,
        step: (): void => {},
        reset: (): void => {},
        setParam: (): void => {},
        setSceneSdf: (): void => {},
        setEmitters: (): void => {},
        setSpawn: (): void => {},
        setForceField: (): void => {},
        dispose: (): void => {},
    };
    const surfaceTask = createFluidSurfaceTask(engine, scene, { bgRT: sceneColorRT, outRT: presentRT, depthRT: sceneColorRT, camera: cam, sim: virtualSim as unknown as FluidSim });
    surfaceTask.setSim(virtualSim as unknown as FluidSim);
    surfaceTask.setParticleAlpha(combinedAlpha);
    surfaceTask.setParticleColor(combinedColor);
    // Scene wiring only — the sun direction the water is lit by, and the HDR cube it reflects. Without
    // setEnvMap the surface samples garbage for reflections and renders opaque green. Every look
    // parameter (colour, absorption, blur, filter, impostor size…) comes from the active fluidSim
    // file's `render` block instead; see applyRenderSetting.
    surfaceTask.setDirLight([-0.4, -0.82, -0.45]);
    surfaceTask.setEnvMap({ view: env._specularCubeView, sampler: env._cubeSampler });
    addTask(scene, surfaceTask);

    // ── Presenting: exactly one pass, chosen by which AA mode is active ─────────────────────────
    // MSAA above only supersamples POLYGON coverage, so it cannot touch the aliasing that dominates
    // this content: the hard straight lines painted into the panel, grate and hazard-stripe
    // textures. SMAA works on the finished image and treats those the same as a silhouette. TAA
    // jitters the camera by a sub-pixel Halton offset each frame and accumulates, so it supersamples
    // EVERYTHING including texture interiors — but core TAA resets to the raw frame whenever the
    // camera moves (disableOnCameraMove), so today it only pays off standing still. Fixing that
    // needs motion-vector reprojection, which core TAA does not have yet.
    //
    // SMAA and TAA are mutually exclusive (see applyExclusivity in settings.ts), so these are
    // PARALLEL alternatives rather than a chain: all three read presentRT and write the swapchain,
    // and gating picks exactly one. A chained design was tried first and cost an extra fullscreen
    // blit on every path, which measurably degraded the image — MSAA scored worse than no AA.
    let smaaOn = false;
    let taaOn = false;
    // TAA must not accumulate while the view is actually changing: with no motion vectors it
    // reprojects a moved camera onto stale history, which is the ghosting you see. Core TAA's own
    // `disableOnCameraMove` cannot do this job here — it keys off the camera's version, and the
    // first-person controller smooths yaw asymptotically (`yaw += (yawTarget - yaw) * k`), so the
    // camera technically changes EVERY frame forever and the flag would fire permanently, leaving
    // TAA resetting every frame (measured: worse than no AA at all). Compare the actual values with
    // an epsilon instead: real walking moves centimetres per frame, the smoothing residual decays to
    // ~1e-9, so 0.1 mm cleanly separates them.
    const TAA_STILL_EPS = 1e-4;
    const TAA_FACTOR = 0.05; // core default: blend weight of the current frame once accumulating
    let taaMoving = true;
    let taaResetPending = false;
    const taaPrevCam = new Float64Array(6);
    const smaaTask = createSmaaPostProcessTask(
        {
            name: "aq-smaa",
            sourceTexture: presentRT,
            targetTexture: engine.scRT,
            // presentRT carries engine.format, which is an sRGB VIEW when the engine was created
            // with srgb: true. Sampling that decodes to linear, so edge detection must be told or its
            // fixed threshold silently means something different (and misses most dark-region edges).
            sourceIsSrgb: graphics.srgb,
        },
        engine,
        scene
    );
    // TAA writes its jitter into the source render task's own scene-uniform block, so it has to
    // point at the task that is actually drawing — which flips when MSAA is toggled (setMsaa
    // re-points it). Otherwise the jitter lands in an unused UBO and TAA accumulates identical
    // frames, doing nothing at all.
    // disableOnCameraMove: false — the character controller re-writes the camera transform every
    // frame (physics runs even when the player is stationary), so the default `true` sees "moved"
    // on every single frame and resets the history forever: TAA then costs three passes and, because
    // the jitter is still applied, hands back a sub-pixel-shifted frame that measures WORSE than no
    // AA at all. With it off, history accumulates and a still view converges; the cost is ghosting
    // while actually moving, which is the trade this demo wants.
    const taaTask = createTaaPostProcessTask(
        { name: "aq-taa", sourceTexture: presentRT, targetTexture: engine.scRT, sourceRenderTask: sceneTask, disableOnCameraMove: false },
        engine,
        scene
    );
    const presentCopyTask = createCopyToTextureTask({ name: "aq-present-copy", sourceTexture: presentRT, targetTexture: engine.scRT }, engine, scene);
    // While moving, TAA is skipped entirely and the plain blit presents the raw frame: no history,
    // no jitter, so no ghosting and no sub-pixel wobble. The scene re-packs a clean (unjittered)
    // matrix on its own whenever the camera changes, so nothing stale is left behind.
    gateExistingTask(smaaTask, () => smaaOn);
    gateExistingTask(taaTask, () => taaOn && !taaMoving);
    gateExistingTask(presentCopyTask, () => !smaaOn && !(taaOn && !taaMoving));
    addTask(scene, smaaTask);
    addTask(scene, taaTask);
    addTask(scene, presentCopyTask);

    // Closes the whole-frame timing envelope and resolves the fluid profiler's queries. Draws
    // nothing, and MUST be the last task added — resolveInto has to be recorded after every timed
    // pass, before the frame's command buffer is submitted.
    if (fluidProfiler) {
        const prof = fluidProfiler;
        addTask(scene, {
            name: "aq-timing-resolve",
            engine,
            scene,
            _passes: [],
            record: (): void => {},
            execute: (): number => {
                if (!fluidProfilerOn) {
                    return 0;
                }
                prof.frameStop(engine._currentEncoder);
                prof.resolveInto(engine._currentEncoder);
                return 0;
            },
            dispose: (): void => {},
        });
    }
    attachFluidProfiler = (): void => {
        const p = fluidProfilerOn ? fluidProfiler : null;
        surfaceTask.setProfiler(p);
        for (const a of activeSims) {
            (a.sim as { setProfiler?: (x: typeof p) => void }).setProfiler?.(p);
        }
    };
    retargetTaa = (task: RenderTask): void => {
        // `_sourceRenderTask` is internal, but it is the only way to follow the MSAA toggle.
        (taaTask as unknown as { _sourceRenderTask: RenderTask })._sourceRenderTask = task;
    };
    // setMsaa already ran during startup, when retargetTaa was still a no-op, so point TAA at the
    // task that MSAA actually left active.
    retargetTaa(msaaOn && msaaSceneTask ? (msaaSceneTask as RenderTask) : sceneTask);

    smaaOn = graphics.smaa;
    canvas.dataset.smaa = String(smaaOn);
    setHudFlag("smaaState", smaaOn);
    setSmaaEnabled = (on: boolean): void => {
        if (on === smaaOn) return;
        smaaOn = on;
        graphics.smaa = on;
        saveGraphicsSettings(graphics);
        canvas.dataset.smaa = String(on);
        setHudFlag("smaaState", on);
    };
    toggleSmaa = (): void => {
        const next = !smaaOn;
        if (next) setTaaEnabled(false); // exclusive with TAA
        setSmaaEnabled(next);
    };
    taaOn = graphics.taa;
    canvas.dataset.taa = String(taaOn);
    setHudFlag("taaState", taaOn);
    setTaaEnabled = (on: boolean): void => {
        if (on === taaOn) return;
        taaOn = on;
        graphics.taa = on;
        saveGraphicsSettings(graphics);
        canvas.dataset.taa = String(on);
        setHudFlag("taaState", on);
    };
    toggleTaa = (): void => {
        const next = !taaOn;
        if (next) {
            // TAA supersamples by jittering the camera; running it over MSAA/SMAA would accumulate
            // already-filtered frames and the jitter would fight their edge detection.
            setMsaaEnabled(false);
            setSmaaEnabled(false);
        }
        setTaaEnabled(next);
    };
    // SSAA is deliberately NOT exclusive with the others: it is a different axis (how many pixels
    // you render) and composes with any of them, which is exactly what makes it a useful baseline.
    cycleSsaa = (): void => {
        const opts = GRAPHICS_SETTING_DEFS.find((d) => d.key === "ssaa")?.options ?? [1];
        const next = opts[(Math.max(0, opts.indexOf(graphics.ssaa)) + 1) % opts.length] ?? 1;
        graphics.ssaa = next;
        saveGraphicsSettings(graphics);
        applySsaa(next);
    };
    let specularAABusy = false;
    toggleSpecularAA = (): void => {
        if (specularAABusy) return;
        specularAABusy = true;
        const next = !graphics.specularAA;
        setShipSpecularAA(next);
        graphics.specularAA = next;
        saveGraphicsSettings(graphics);
        canvas.dataset.specularAa = String(next);
        setHudFlag("specularAaState", next);
        void rebuildScenePbrPipelines(scene)
            .catch((err: unknown) => {
                setShipSpecularAA(!next);
                graphics.specularAA = !next;
                saveGraphicsSettings(graphics);
                canvas.dataset.specularAa = String(!next);
                setHudFlag("specularAaState", !next);
                console.warn("[aquanova] specular AA switch failed", err);
            })
            .finally(() => {
                specularAABusy = false;
            });
    };

    // The setting that most recently drove the shared surface pass, for the QA hook below.
    let activeRender: FluidRenderSetting | undefined;
    let activeFoam: FluidFoamSetting | undefined;
    let activeSettingName: string | undefined;
    /** The impulse the last liquefaction actually fired, with `direction` already resolved to a unit
     *  vector (so a (0,0,0) setting shows the shot ray it was replaced with). */
    let activeImpulse: Record<string, unknown> | undefined;
    let activeGrid: Record<string, unknown> | undefined;
    /** The physics record the last liquefaction's solver was actually built with. */
    let activePhysics: Record<string, unknown> | undefined;

    // Push a setting file's `render` block onto the surface pass. That pass is shared by every active
    // blob, so this is global state and the most recent liquefaction wins — the same compromise the
    // per-particle colour toggle already makes. Nothing here is defaulted locally: a key the file omits
    // simply keeps whatever the surface task already had, so the file is the single source of truth.
    // Previously these were hardcoded (absorption 0.4, size 0.7, refraction 0.06, specular 41), which
    // silently overrode the file — Liquefactor drives the identical surface task from its UI and did
    // honour them, so the same setting file produced visibly different water in the two demos.
    const applyRenderSetting = (r: FluidRenderSetting | undefined): void => {
        activeRender = r;
        if (!r) return;
        const rgb = hexToRgb(r.waterColor);
        if (rgb) {
            surfaceTask.setFluidColor(rgb);
            // Keep the flat (non-mesh-colour) particle fill in step with the surface tint.
            waterRgb[0] = rgb[0];
            waterRgb[1] = rgb[1];
            waterRgb[2] = rgb[2];
            fillColorScratch();
        }
        if (r.absorption !== undefined) surfaceTask.setAbsorption(r.absorption);
        if (r.particleSize !== undefined) surfaceTask.setSizeScale(r.particleSize);
        if (r.refractionStrength !== undefined) surfaceTask.setRefractionStrength(r.refractionStrength);
        if (r.specularPower !== undefined) surfaceTask.setSpecularPower(r.specularPower);
        if (r.surfaceDepthBlur !== undefined) surfaceTask.setDepthBlur(r.surfaceDepthBlur, r.depthBlurEdgeThreshold ?? 0);
        if (r.surfaceThicknessBlur !== undefined) surfaceTask.setThicknessBlur(r.surfaceThicknessBlur);
        if (r.halfRendering !== undefined) surfaceTask.setHalfRender(r.halfRendering);
        if (r.thicknessDownscale !== undefined) surfaceTask.setThicknessDownscale(r.thicknessDownscale);
        if (r.surfaceFilter === "bilateral" || r.surfaceFilter === "narrowRange") surfaceTask.setSurfaceFilter(r.surfaceFilter);
        if (r.narrowRangeDelta !== undefined || r.narrowRangeMu !== undefined) surfaceTask.setNarrowRange(r.narrowRangeDelta ?? 10, r.narrowRangeMu ?? 1);
    };

    // ── Per-particle mesh-colour path (honours a fluidSim setting's useMeshColors) ─────────────────
    // Colour each water particle from the liquefied mesh's base-colour texture: nearest mesh vertex →
    // its UV → textureLoad, gamma-encoded (the sRGB view decodes to linear on load) to match the
    // display-space fluid composite. Ported from the Liquefactor demo's per-particle colour render.
    const COLOR_SAMPLE_WGSL = /* wgsl */ `
struct P { count: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> uvs: array<vec2<f32>>;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outCol: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) { return; }
    let dims = vec2<f32>(textureDimensions(tex, 0));
    let w = fract(uvs[i]);                          // repeat-wrap
    let coord = vec2<i32>(clamp(w * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));
    let c = textureLoad(tex, coord, 0);             // sRGB view → linear
    outCol[i] = vec4<f32>(pow(max(c.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}`;
    const colorPipe = device.createComputePipeline({
        label: "aq-color-sample",
        layout: "auto",
        compute: { module: device.createShaderModule({ label: "aq-color-sample", code: COLOR_SAMPLE_WGSL }), entryPoint: "main" },
    });

    // Build a per-particle RGBA colour buffer by texture-sampling each particle's UV. The UVs are
    // computed alongside the particles (in the worker when one is available), so nothing here touches
    // the mesh geometry. Falls back to the base water tint when the mesh lacks UVs or a base-colour
    // texture.
    const buildMeshColorBuffer = (count: number, particleUvs: Float32Array | null, texView: GPUTextureView | undefined): GPUBuffer => {
        const buf = device.createBuffer({ label: "aq-shot-color", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(buf, 0, colorScratch, 0, count * 4); // water-tint fallback (overwritten below when textured)
        if (!particleUvs || particleUvs.length < count * 2 || !texView) return buf;
        const uvBuf = device.createBuffer({ label: "aq-shot-uv", size: count * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(uvBuf, 0, particleUvs, 0, count * 2);
        const cbuf = device.createBuffer({ label: "aq-shot-ccount", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(cbuf, 0, new Uint32Array([count, 0, 0, 0]));
        const enc = device.createCommandEncoder();
        const bg = device.createBindGroup({
            layout: colorPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: cbuf } },
                { binding: 1, resource: { buffer: uvBuf } },
                { binding: 2, resource: texView },
                { binding: 3, resource: { buffer: buf } },
            ],
        });
        const pass = enc.beginComputePass();
        pass.setPipeline(colorPipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(count / 64));
        pass.end();
        device.queue.submit([enc.finish()]);
        uvBuf.destroy();
        cbuf.destroy();
        return buf;
    };

    // ── Milestone D: Liquefaction ────────────────────────────────────────────────────────────────
    // Fluid collision is the SHIP'S OWN AUTHORED PRIMITIVES, evaluated analytically — no baked SDF
    // grids anywhere. Each simulation gets the subset of primitives whose bounds meet its domain,
    // packed into a storage buffer; the WGSL is identical for every sim, so they all share one
    // compiled pipeline (the solvers cache on shader source — a per-shot shader would pay a ~450 ms
    // compile every time).
    //
    // A liquefied prop's own primitive is left out of its sim: it has just become the water, so
    // colliding the water against it would trap every particle inside a solid.
    //
    // The ground plane is a BACKSTOP for a domain that reaches past the authored floor slabs. It is a
    // single plane at FLOOR_Y, so it is right for the main deck only — a liquefiable prop on an upper
    // deck would rest on it rather than falling. None are tagged today.
    const sceneSdfUbo = device.createBuffer({ label: "aq-scene-sdf", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(sceneSdfUbo, 0, new Float32Array([FLOOR_Y, 0, 0, 0]));
    const SCENE_SDF_STRUCT = "struct SceneSdfParams { ground: vec4<f32>, };";
    const SCENE_SDF_BODY = `${PRIMITIVES_WGSL}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return min(primitivesSdf(pt, dt), pt.y - sceneSdfParams.ground.x); }`;

    /** Buffer + spec for one sim's collision set, with the slots that need refreshing each step. */
    interface CollisionSet {
        buffer: GPUBuffer;
        scratch: Float32Array;
        /** The selected primitives, kept live so the debug overlay can draw exactly what the shader sees. */
        prims: FluidPrimitive[];
        /** Movable props in this set and the slot each occupies — only these are rewritten per step. */
        moving: Array<{ body: DynBody; slot: number }>;
        /** Packed slots owned by a manifest placement, including static primitives. */
        slotsByInstanceId: Map<string, number[]>;
        /** Packed slots owned by a dynamic body, including bodies without an instance id. */
        slotsByBody: Map<DynBody, number[]>;
        /** Player capsule slot, reserved in every non-ground-only set regardless of its initial domain. */
        playerSlot: number | null;
        spec: SceneSdfSpec;
    }
    interface SimulationGridAabb {
        min: [number, number, number];
        max: [number, number, number];
    }
    /** Select, pack and upload the primitives one simulation collides against. */
    const buildCollisionSet = (gridAabb: SimulationGridAabb, exclude: ReadonlySet<string>): CollisionSet => {
        const hit = (lo: readonly [number, number, number], hi: readonly [number, number, number]): boolean =>
            lo[0] <= gridAabb.max[0] && hi[0] >= gridAabb.min[0] && lo[1] <= gridAabb.max[1] && hi[1] >= gridAabb.min[1] && lo[2] <= gridAabb.max[2] && hi[2] >= gridAabb.min[2];
        const prims: FluidPrimitive[] = [];
        const moving: Array<{ body: DynBody; slot: number }> = [];
        const slotsByInstanceId = new Map<string, number[]>();
        const slotsByBody = new Map<DynBody, number[]>();
        let playerSlot: number | null = null;
        const addSlot = <Key>(map: Map<Key, number[]>, key: Key, slot: number): void => {
            const slots = map.get(key);
            if (slots) slots.push(slot);
            else map.set(key, [slot]);
        };
        const addPrimitive = (primitive: FluidPrimitive, instanceId?: string, body?: DynBody): number => {
            const slot = prims.length;
            prims.push(primitive);
            if (instanceId) addSlot(slotsByInstanceId, instanceId, slot);
            if (body) addSlot(slotsByBody, body, slot);
            return slot;
        };
        if (!groundOnly) {
            for (const s of staticPrims) {
                if (!exclude.has(s.id) && !removedPlacements.has(s.id) && hit(s.min, s.max)) addPrimitive(s.prim, s.id);
            }
            for (const d of dynBodies) {
                // The melting prop is excluded: it has just BECOME this water, so colliding against it
                // would trap every particle inside a solid.
                if (d.instanceId && (exclude.has(d.instanceId) || removedPlacements.has(d.instanceId))) continue;
                const p = livePrim(d);
                if (!p) continue;
                const bb = primAabb(p);
                if (!hit(bb.min, bb.max)) continue;
                const slot = addPrimitive(p, d.instanceId, d);
                if (d.movable) moving.push({ body: d, slot });
            }
            playerSlot = addPrimitive(livePlayerPrimitive());
        }
        const capacity = Math.max(prims.length, 1);
        const buffer = device.createBuffer({ label: "aq-fluid-prims", size: primBufferBytes(capacity), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const scratch = new Float32Array(PRIM_HEADER + capacity * PRIM_STRIDE);
        packPrimitives(scratch, prims);
        device.queue.writeBuffer(buffer, 0, scratch);
        return {
            buffer,
            scratch,
            prims,
            moving,
            slotsByInstanceId,
            slotsByBody,
            playerSlot,
            spec: { struct: SCENE_SDF_STRUCT, sdf: SCENE_SDF_BODY, buffer: sceneSdfUbo, sdfGrid: buffer },
        };
    };
    const setCollisionSlotsActive = (set: CollisionSet, slots: readonly number[], active: boolean): void => {
        for (const slot of slots) {
            const primitive = set.prims[slot];
            if (!primitive || (primitive.active !== false) === active) continue;
            set.prims[slot] = { ...primitive, active };
            setPackedPrimitiveActive(set.scratch, slot, active);
            const flagOffset = PRIM_HEADER + slot * PRIM_STRIDE + PRIM_ACTIVE_OFFSET;
            device.queue.writeBuffer(set.buffer, flagOffset * 4, new Float32Array([active ? 1 : 0]));
        }
    };
    const setPlacementCollisionActive = (instanceId: string, active: boolean): void => {
        if (active) removedPlacements.delete(instanceId);
        else removedPlacements.add(instanceId);
        for (const sim of activeSims) setCollisionSlotsActive(sim.collision, sim.collision.slotsByInstanceId.get(instanceId) ?? [], active);
    };
    const setDynBodyCollisionActive = (body: DynBody, active: boolean): void => {
        if (body.instanceId) {
            setPlacementCollisionActive(body.instanceId, active);
            return;
        }
        for (const sim of activeSims) setCollisionSlotsActive(sim.collision, sim.collision.slotsByBody.get(body) ?? [], active);
    };
    let groundOnly = false;
    toggleGroundOnly = (): void => {
        groundOnly = !groundOnly;
        // Rebuild every live sim's primitive set: ground-only simply packs ZERO primitives, so the
        // shader (and its compiled pipeline) is unchanged — only the buffer's count header moves.
        for (const a of activeSims) {
            const set = buildCollisionSet(a.gridAabb, a.group.excluded);
            a.collision.buffer.destroy();
            a.collision = set;
            a.sim.setSceneSdf(set.spec);
        }
        canvas.dataset.groundOnly = groundOnly ? "on" : "off";
        // eslint-disable-next-line no-console
        console.log(`[aquanova] fluid collision: ${groundOnly ? "GROUND PLANE ONLY (Liquefactor-equivalent)" : "ship collision primitives + ground"}`);
    };

    // A radial burst from the volume centre plus a uniform directional push, so a liquefied prop
    // erupts before settling (kept in sync with the Liquefactor demo — see IMPULSE_* below).
    //
    // push.xyz is the uniform push (already scaled: unit direction × base × intensity) and push.w is
    // the radial magnitude. push.xyz used to be a fixed +Y lift; it is now an arbitrary direction so a
    // setting file can aim the burst.
    const IMPULSE_WGSL = /* wgsl */ `
fn externalForce(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> vec3<f32> {
    let center = forceFieldParams.center.xyz;
    let radius = max(forceFieldParams.center.w, 1.0e-4);
    let toParticle = pos - center;
    let dist = length(toParticle);
    var f = forceFieldParams.push.xyz;
    if (dist < radius) {
        let dir = select(vec3<f32>(0.0, 1.0, 0.0), toParticle / max(dist, 1.0e-4), dist > 1.0e-4);
        let core = smoothstep(0.0, 0.15, dist / radius);
        f += dir * forceFieldParams.push.w * core;
    }
    return f * dt;
}`;
    const LIFETIME = 5.0; // seconds a settled blob lives before it fades out
    const FADE_DUR = 1.2; // seconds the alpha fade-out takes before dispose + mesh removal
    // Impulse bases. These MUST match Liquefactor's: a prop auditioned there has to behave the same
    // here, and a setting file only carries an intensity + direction, not these magnitudes. (Aquanova
    // used 8 / 10 against Liquefactor's 18 / 6, which is why the same file erupted far more violently
    // in one demo than the other.)
    const IMPULSE_RADIAL_BASE = 18; // outward burst from the volume centre (× intensity)
    const IMPULSE_DIR_BASE = 6; // uniform push along the setting's direction (× intensity)
    const IMPULSE_DEFAULT_DIR: readonly [number, number, number] = [0, 1, 0]; // straight up, the old behaviour
    const IMPULSE_DEFAULT_INTENSITY = 1;
    const SPREAD_MARGIN = 1.2; // extra half-width (world units) around the mesh footprint for the sim grid
    const SPREAD_SCALE = 3; // widen the X/Z half-extent so a puddle has room to spread instead of piling against the grid wall
    // Fallback MLS-MPM water used when the manifest lists no fluidSim settings (or a name fails to load).
    // `maxSubDtMs` mirrors the solver default (1/120 s): the largest slice one sub-step may integrate,
    // which decides how many sub-steps a frame needs — never how much time it advances.
    const FLUID_SETTING = {
        gravity: 9.8,
        stiffness: 350,
        viscosity: 0.3,
        restDensity: 3,
        damping: 0.995,
        affineDamping: 0.9,
        groundDamp: 0.85,
        groundDampHeight: 1.5,
        restitution: 0.3,
        substeps: 3,
        maxSubDtMs: 1000 / 120,
    };
    // Build the solver a liquefied prop erupts into, honouring the chosen setting's method + params. The
    // settings can be MLS-MPM or PB-MPM. Render (colour/absorption/…) is a SHARED surface pass across all
    // active blobs, so it stays global — only the simulation varies per setting.
    const buildFluidSim = (
        setting: FluidSimSetting | undefined,
        o: { count: number; particleRadius: number; initialPositions: Float32Array; boundsMin: [number, number, number]; boundsMax: [number, number, number]; dx: number }
    ): FluidSim => {
        const sim = buildFluidSimRaw(setting, o);
        // Sims are created per shot, so a newly-spawned blob has to pick up the current profiler
        // state or its passes would go untimed while the panel is open.
        (sim as { setProfiler?: (p: FluidProfilerImpl | null) => void }).setProfiler?.(fluidProfilerOn ? fluidProfiler : null);
        return sim;
    };
    const buildFluidSimRaw = (
        setting: FluidSimSetting | undefined,
        o: { count: number; particleRadius: number; initialPositions: Float32Array; boundsMin: [number, number, number]; boundsMax: [number, number, number]; dx: number }
    ): FluidSim => {
        const base = {
            count: o.count,
            particleRadius: o.particleRadius,
            initialPositions: o.initialPositions,
            boundsMin: o.boundsMin,
            boundsMax: o.boundsMax,
            dx: o.dx,
            groundY: FLOOR_Y,
        };
        if (setting?.method === "PB-MPM") {
            const p = setting.physics;
            return createPbMpmSim(engine, {
                ...base,
                material: setting.material ?? 0,
                gravity: p.gravity,
                substeps: p.substeps,
                iterations: p.iterations,
                ...(p.maxSubDtMs ? { maxSubDt: p.maxSubDtMs / 1000 } : {}),
                liquidRelaxation: p.liquidRelaxation,
                liquidViscosity: p.liquidViscosity,
                elasticityRatio: p.elasticityRatio,
                elasticRelaxation: p.elasticRelaxation,
                frictionAngle: p.frictionAngle,
                plasticity: p.plasticity,
                restitution: p.restitution,
            });
        }
        const p = setting?.method === "MLS-MPM" ? setting.physics : FLUID_SETTING;
        return createMlsMpmSim(engine, {
            ...base,
            gravity: p.gravity,
            stiffness: p.stiffness,
            viscosity: p.viscosity,
            restDensity: p.restDensity,
            substeps: p.substeps,
            damping: p.damping,
            affineDamping: p.affineDamping,
            ...(p.maxSubDtMs ? { maxSubDt: p.maxSubDtMs / 1000 } : {}),
            groundDamp: p.groundDamp,
            groundDampHeight: p.groundDampHeight,
            restitution: p.restitution,
        });
    };
    /**
     * Compile every fluid pipeline once, at load, on a throwaway one-particle sim.
     *
     * The first liquefaction otherwise stalls for ~480 ms — measured, and measured to be a ONE-time
     * cost: the second shot (and a shot using a different setting) has no frame over 30 ms. It is not
     * JS (the main thread profiles 93% idle through the stall) and not the WebGPU API calls (12 shader
     * modules + 12 compute pipelines return in under a millisecond); the cost lands in the GPU process
     * when those pipelines are first used. `g2pPipeCache` is per-sim so it cannot help across shots —
     * what saves the second shot is the browser's own cache, keyed on the WGSL source. Compiling that
     * same source here, before the demo reports ready, moves the stall into the load screen.
     *
     * Warmed per METHOD, since MLS-MPM and PB-MPM are different shaders, and with the real scene SDF,
     * because the G2P and update-grid variants are keyed on that WGSL too.
     */
    async function warmUpFluidPipelines(): Promise<void> {
        const methods = new Set<string>();
        for (const s of fluidSettings.values()) {
            if (s) {
                methods.add(s.method);
            }
        }
        methods.add("MLS-MPM"); // the fallback FLUID_SETTING, used when a mesh names no setting
        const t0 = performance.now();
        for (const method of methods) {
            const setting = [...fluidSettings.values()].find((s) => s?.method === method);
            const sim = buildFluidSim(setting, {
                count: 1,
                particleRadius: 0.05,
                initialPositions: new Float32Array([0, -1e5, 0]),
                boundsMin: [-1, -1, -1],
                boundsMax: [1, 1, 1],
                dx: 0.18,
            });
            const warmSet = buildCollisionSet({ min: [-1, -1, -1], max: [1, 1, 1] }, new Set());
            sim.setSceneSdf(warmSet.spec);
            const enc = device.createCommandEncoder({ label: "aq-fluid-warmup" });
            sim.step(enc, 1 / 60);
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
            sim.dispose();
            warmSet.buffer.destroy();
        }
        // eslint-disable-next-line no-console
        console.log(`[aquanova] fluid pipelines warmed (${[...methods].join(", ")}) in ${(performance.now() - t0).toFixed(0)} ms`);
    }

    const WRIGGLE_AMP = 0.016; // per-frame "pain" jitter amplitude (world units)    // Dissolve-front growth (world units / s). Matched to Liquefactor's SHIP-mode front (its gallery
    // speed of 6 divided by 3): ship modules read as architecture rather than a single prop, so a faster
    // sweep pops instead of melting. Liquefactor loads this same ship.glb to audition liquefaction, so
    // the two must agree or what the audition shows is not what the game does.
    const LIQUEFY_SPEED = 2;
    const DEFAULT_SAMPLE_RADIUS = 0.03; // volume-sampling spacing when a fluidSim setting omits demoParams.particleRadius
    const LIQUEFY_EDGE = 0.6; // fire-glow band width at the dissolving boundary

    // Dissolve clip: give each liquefiable mesh its OWN material instance (a shallow clone — textures
    // and build group are shared by reference) carrying a private liquefy plugin + state. The ship's
    // materials are shared level-wide (many walls/props reuse e.g. MI_Trim_02), so attaching the clip
    // to the shared material would make EVERY mesh using it clip/glow around the same hit point. Cloning
    // per mesh isolates the dissolve to the shot target. Done BEFORE registerScene so each clone's
    // pipeline is compiled with the plugin; disabled (no-op) until a shot fires.
    type PluginMat = { plugins?: unknown[]; _uboVersion: number };
    const liquefyStates = new Map<Mesh, LiquefyState>();
    for (const m of behaviorManager.dissolvableMeshes) {
        const mat = m.material;
        if (!mat || !isPbrMaterial(mat)) continue;
        const st: LiquefyState = { hit: [0, 0, 0], frontR: 0, edge: LIQUEFY_EDGE, enabled: false };
        const src = mat as unknown as PluginMat;
        const clone = { ...(mat as object), _uboVersion: 0, plugins: [...(src.plugins ?? []), createLiquefyPlugin(() => st, "pbr")] } as unknown as Material;
        m.material = clone;
        liquefyStates.set(m, st);
    }
    const bumpMat = (mat: Material): void => {
        (mat as unknown as PluginMat)._uboVersion++;
    };
    // Front radius that fully engulfs the mesh AABB from the hit point (+ edge band + margin).
    const computeMaxR = (hit: readonly [number, number, number], bMin: readonly [number, number, number], bMax: readonly [number, number, number]): number => {
        let maxD = 0;
        for (const px of [bMin[0], bMax[0]])
            for (const py of [bMin[1], bMax[1]]) for (const pz of [bMin[2], bMax[2]]) maxD = Math.max(maxD, Math.hypot(px - hit[0]!, py - hit[1]!, pz - hit[2]!));
        return maxD + LIQUEFY_EDGE + 0.5;
    };

    interface DissolveMember {
        mesh: Mesh;
        disp: SceneNode | null; // dynamic-mesh display root (dispose the whole subtree on fade-out)
        dyn: DynBody | null; // dynamic-body handle: its collision is disabled when the fluid phase begins (null for a static prop)
        state: LiquefyState | null; // material clip state (null if the mesh had no PBR material)
        material: Material | null;
        maxR: number;
        wriggleNode: SceneNode; // node jittered during the dissolve (display root, or the mesh itself)
        wriggleBase: [number, number, number];
        particleOffset: number;
        particleCount: number;
        colorBuffer: GPUBuffer | null;
        behaviorAvailability: MeshBehaviorAvailability;
        dissolved: boolean;
    }
    interface ActiveSim {
        sim: FluidSim;
        mesh: Mesh; // primary shot mesh, used only to name this grouped simulation
        members: DissolveMember[];
        phase: "dissolving" | "fluid" | "fading";
        fluidElapsed: number;
        fadeElapsed: number;
        impulseRemaining: number;
        impulseBuffer: GPUBuffer;
        impulseSpec: ForceFieldSpec; // applied when the fluid phase begins, not during the visual dissolve
        /** False when the setting's intensity is 0: the burst would be exactly zero, so the force pass is
         *  never installed (the engine compiles it lazily on first use — see beginFluidPhase). */
        impulseActive: boolean;
        waterBase: Float32Array; // all members' sampled particle positions (xyzw)
        waterScratch: Float32Array;
        useMeshColors: boolean;
        group: ShotGroup;
        /** This sim's collision primitives and the exact solver-grid AABB they were picked for. */
        collision: CollisionSet;
        gridAabb: SimulationGridAabb;
    }
    const activeSims: ActiveSim[] = [];

    interface CollectedSample {
        entry: PendingSample;
        sample: SampleResult;
    }

    // One shot = one GROUP (the melted node's primitives + every linked node's, transitively).
    // Every mesh is sampled independently in the worker pool, then all samples are concatenated into
    // ONE solver. Per-mesh state survives only for visual dissolve, color, Havok teardown and cleanup.
    interface ShotGroup {
        sampling: number;
        dispatchComplete: boolean;
        samples: CollectedSample[];
        sim: ActiveSim | null;
        direction: 1 | -1;
        meshes: ReadonlySet<Mesh>;
        resumeGeneration: number;
        resumePending: number | null;
        primaryMesh: Mesh;
        hit: readonly [number, number, number] | null;
        setting: FluidSimSetting | undefined;
        settingName: string | undefined;
        /** Instance ids this shot's water must NOT collide against (its own props). */
        excluded: ReadonlySet<string>;
        /** Unit direction from the player to the crosshair when the shot was fired. Used for a
         *  setting whose `impulse.direction` is (0,0,0) — "push it the way I shot it". */
        shotDir: [number, number, number];
    }

    // ── Sampling worker pool ─────────────────────────────────────────────────────────────
    // The CPU particle fill is 100–400 ms for a typical prop and scales with the group size (a
    // 4-primitive node samples four times, a linked door twice), so running it inline froze the
    // frame on every shot. It now runs in workers: `requestSample` posts the WORLD-space geometry
    // and `collectSample` gathers the result. Once every linked mesh is ready, the main thread builds
    // one shared simulation and its per-mesh colour buffers.
    type SampleResult = { positions: Float32Array; count: number; radius: number; min: [number, number, number]; max: [number, number, number]; uvs: Float32Array | null };
    interface PendingSample {
        mesh: Mesh;
        dyn: DynBody | null;
        wriggleNode: SceneNode; // jittered from the moment the shot lands, before the seed arrives
        wriggleBase: [number, number, number];
        group: ShotGroup;
        behaviorAvailability: MeshBehaviorAvailability;
    }
    type SampleMsg = {
        id: number;
        positions: Float32Array;
        uvs: Float32Array | null;
        count: number;
        radius: number;
        boundsMin: [number, number, number];
        boundsMax: [number, number, number];
    };
    interface PoolWorker {
        worker: Worker;
        pending: number;
    }
    const workerPool: PoolWorker[] = [];
    const pendingSamples = new Map<number, PendingSample>();
    // Shots whose seed is still being sampled. The frame loop pain-shakes these so the hit reads as
    // instant feedback instead of the prop sitting still until the worker replies.
    const samplingShots = new Set<PendingSample>();
    let controlledGroup: ShotGroup | null = null;
    let sampleSeq = 0;

    const setSamplingGroupWriggle = (group: ShotGroup, active: boolean): void => {
        const restoredNodes = new Set<SceneNode>();
        for (const entry of samplingShots) {
            if (entry.group !== group) continue;
            // Keep a paused dynamic body visually pinned to the pose whose geometry was sent to the
            // worker. If the trigger is pressed again, the returned particles and display still line
            // up; cancellation re-enables Havok pose sync once the pending samples have settled.
            if (active && entry.dyn) entry.dyn.wriggling = true;
            if (!active && !restoredNodes.has(entry.wriggleNode)) {
                restoredNodes.add(entry.wriggleNode);
                entry.wriggleNode.position.set(entry.wriggleBase[0], entry.wriggleBase[1], entry.wriggleBase[2]);
            }
        }
    };

    requestFusionResume = (): number | null => {
        const group = controlledGroup;
        if (!group) return null;
        if (group.direction > 0) return 0;
        const token = ++group.resumeGeneration;
        group.resumePending = token;
        return token;
    };

    resolveFusionResume = (token: number, mesh: Mesh | null): "resumed" | "start-new" | "continue" => {
        const group = controlledGroup;
        if (!group || group.resumePending !== token) return "continue";
        group.resumePending = null;
        if (mesh && group.meshes.has(mesh)) {
            group.direction = 1;
            setSamplingGroupWriggle(group, true);
            finalizeShotIfReady(group);
            return "resumed";
        }
        if (mesh && behaviorManager.isLiquefiable(mesh)) {
            releaseControlledGroup(group);
            finalizeShotIfReady(group);
            return "start-new";
        }
        finalizeShotIfReady(group);
        return "continue";
    };

    reverseFusion = (): boolean => {
        const group = controlledGroup;
        if (!group) return false;
        group.direction = -1;
        group.resumePending = null;
        group.resumeGeneration++;
        setSamplingGroupWriggle(group, false);
        finalizeShotIfReady(group);
        return true;
    };

    const releaseControlledGroup = (group: ShotGroup): void => {
        if (controlledGroup === group) controlledGroup = null;
    };

    const abortSample = (entry: PendingSample): void => {
        samplingShots.delete(entry);
        entry.group.sampling--;
        entry.wriggleNode.position.set(entry.wriggleBase[0], entry.wriggleBase[1], entry.wriggleBase[2]);
        if (entry.dyn && ![...samplingShots].some((pending) => pending.dyn === entry.dyn)) entry.dyn.wriggling = false;
        // Nothing sampled — hand the mesh back so it stays shootable rather than becoming an inert
        // solid that can never be liquefied.
        behaviorManager.restoreMesh(entry.mesh, entry.behaviorAvailability);
        finalizeShotIfReady(entry.group);
    };

    const onSampleMessage = (ev: MessageEvent<SampleMsg>): void => {
        const { id, positions, uvs, count, radius, boundsMin, boundsMax } = ev.data;
        const entry = pendingSamples.get(id);
        pendingSamples.delete(id);
        if (!entry) return;
        if (!count) {
            abortSample(entry);
            return;
        }
        collectSample(entry, { positions, count, radius, min: boundsMin, max: boundsMax, uvs });
    };
    try {
        if (typeof Worker !== "undefined") {
            const poolSize = Math.min(3, Math.max(1, (navigator.hardwareConcurrency || 4) - 1));
            for (let i = 0; i < poolSize; i++) {
                const worker = new Worker(new URL("../particle-fill-worker.ts", import.meta.url), { type: "module" });
                const pw: PoolWorker = { worker, pending: 0 };
                worker.addEventListener("message", (ev: MessageEvent<SampleMsg>) => {
                    pw.pending = Math.max(0, pw.pending - 1);
                    onSampleMessage(ev);
                });
                worker.addEventListener("error", (e) => {
                    // eslint-disable-next-line no-console
                    console.warn("[aquanova] sample worker error", e.message);
                    const idx = workerPool.indexOf(pw);
                    if (idx >= 0) workerPool.splice(idx, 1); // drop the faulty worker; sync fallback once the pool empties
                });
                workerPool.push(pw);
            }
        }
    } catch {
        workerPool.length = 0;
    }
    // Least-loaded worker, so a multi-primitive node's primitives sample in parallel.
    const pickWorker = (): PoolWorker | null => {
        let best: PoolWorker | null = null;
        for (const pw of workerPool) if (!best || pw.pending < best.pending) best = pw;
        return best;
    };
    canvas.dataset.sampleWorkers = String(workerPool.length);

    const requestSample = (mesh: Mesh, group: ShotGroup): void => {
        const g = getMeshGeometry(mesh);
        if (!g) return;
        // Transform the mesh's LOCAL geometry into WORLD space (the fluid lives in world coords).
        // Captured NOW, so a Havok-posed prop is sampled in the pose it had when it was shot.
        const w = mesh.worldMatrix;
        const src = g.positions;
        const worldPos = new Float32Array(src.length);
        for (let i = 0; i < src.length; i += 3) {
            const lx = src[i]!,
                ly = src[i + 1]!,
                lz = src[i + 2]!;
            worldPos[i] = w[0]! * lx + w[4]! * ly + w[8]! * lz + w[12]!;
            worldPos[i + 1] = w[1]! * lx + w[5]! * ly + w[9]! * lz + w[13]!;
            worldPos[i + 2] = w[2]! * lx + w[6]! * ly + w[10]! * lz + w[14]!;
        }
        // The setting is chosen ONCE per shot by the caller and shared across the whole group; its
        // demoParams.particleRadius drives the volume-sampling spacing (and therefore the particle
        // count), so the same setting file yields the same count here as in the Liquefactor demo.
        const radius = group.setting?.particleRadius ?? DEFAULT_SAMPLE_RADIUS;
        // Retire the mesh as a target IMMEDIATELY: sampling is async now, so leaving it in the sets
        // would let a second shot (or a linked cascade) start a duplicate sim for the same mesh.
        const behaviorAvailability = behaviorManager.retireMesh(mesh);
        // Start the pain shake NOW, on the click, rather than when the seed arrives — the sample takes
        // 100–400 ms in the worker and the prop would otherwise stand still through it. `wriggling`
        // decouples the display node from its Havok pose so the shake doesn't fight physics; the base
        // is captured here and reused when the grouped sim is finalized, which must not read the
        // already-jittered pose.
        const dyn = dynBodyByMesh.get(mesh) ?? null;
        if (dyn) dyn.wriggling = true;
        const wriggleNode = dyn ? dyn.disp : (mesh as unknown as SceneNode);
        const entry: PendingSample = {
            mesh,
            dyn,
            wriggleNode,
            wriggleBase: [wriggleNode.position.x, wriggleNode.position.y, wriggleNode.position.z],
            group,
            behaviorAvailability,
        };
        group.sampling++;
        samplingShots.add(entry);
        const uvs = g.uvs ? g.uvs.slice() : null;
        const pw = pickWorker();
        if (pw) {
            const id = ++sampleSeq;
            pendingSamples.set(id, entry);
            pw.pending++;
            const indices = g.indices.slice();
            const transfer: Transferable[] = [worldPos.buffer, indices.buffer];
            if (uvs) transfer.push(uvs.buffer);
            pw.worker.postMessage({ id, positions: worldPos, indices, uvs, texIndices: null, radius, mode: "dense", surfaceOnly: false, ox: 0, oy: 0, oz: 0 }, transfer);
            return;
        }
        // No worker available — sample synchronously on the main thread (blocks).
        const s = fillMeshParticles({ positions: worldPos, indices: g.indices, uvs, radius, mode: "dense" });
        if (!s.count) {
            abortSample(entry);
            return;
        }
        collectSample(entry, { positions: s.positions, count: s.count, radius: s.radius, min: s.bounds.min, max: s.bounds.max, uvs: s.uvs });
    };

    function collectSample(entry: PendingSample, sample: SampleResult): void {
        entry.group.samples.push({ entry, sample });
        entry.group.sampling--;
        finalizeShotIfReady(entry.group);
    }

    function finalizeShotIfReady(group: ShotGroup): void {
        if (!group.dispatchComplete || group.sampling > 0 || group.sim) return;
        if (group.samples.length === 0) {
            releaseControlledGroup(group);
            return;
        }
        if (group.direction < 0 && group.resumePending !== null) return;
        for (const { entry } of group.samples) samplingShots.delete(entry);
        if (group.direction < 0) {
            const restoredNodes = new Set<SceneNode>();
            for (const { entry } of group.samples) {
                if (!restoredNodes.has(entry.wriggleNode)) {
                    restoredNodes.add(entry.wriggleNode);
                    entry.wriggleNode.position.set(entry.wriggleBase[0], entry.wriggleBase[1], entry.wriggleBase[2]);
                }
                if (entry.dyn) entry.dyn.wriggling = false;
                behaviorManager.restoreMesh(entry.mesh, entry.behaviorAvailability);
            }
            group.samples.length = 0;
            releaseControlledGroup(group);
            return;
        }
        const setting = group.setting;
        const settingName = group.settingName;
        // A member whose geometry could not be sampled was restored as a solid target. Keep its
        // collider in the shared simulation; only successful members become water and exclude their
        // own authored collision placement.
        group.excluded = excludedIds(group.samples.map(({ entry }) => entry.mesh));
        applyRenderSetting(setting?.render);
        activeFoam = setting?.foam;
        activeSettingName = settingName;
        const totalCount = group.samples.reduce((total, collected) => total + collected.sample.count, 0);
        const positions = new Float32Array(totalCount * 3);
        const waterBase = new Float32Array(totalCount * 4);
        const min: [number, number, number] = [Infinity, Infinity, Infinity];
        const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
        let particleOffset = 0;
        for (const { sample } of group.samples) {
            positions.set(sample.positions, particleOffset * 3);
            for (let axis = 0; axis < 3; axis++) {
                min[axis] = Math.min(min[axis]!, sample.min[axis]!);
                max[axis] = Math.max(max[axis]!, sample.max[axis]!);
            }
            for (let i = 0; i < sample.count; i++) {
                const dst = (particleOffset + i) * 4;
                waterBase[dst] = sample.positions[i * 3]!;
                waterBase[dst + 1] = sample.positions[i * 3 + 1]!;
                waterBase[dst + 2] = sample.positions[i * 3 + 2]!;
                waterBase[dst + 3] = 1;
            }
            particleOffset += sample.count;
        }
        const cx = (min[0] + max[0]) / 2,
            cy = (min[1] + max[1]) / 2,
            cz = (min[2] + max[2]) / 2;
        // Domain: the file's explicit `grid` when it pins an axis, else the automatic size from the
        // prop's own footprint. The wall is a hard boundary, so this is what decides how far a puddle
        // may spread — and the reason a prop tuned in Liquefactor only matches here when it carries
        // its grid across.
        const g = setting?.grid;
        const particleRadius = group.samples[0]!.sample.radius;
        const dx = Math.max(particleRadius * 2.4, 0.18);
        const autoHalf = (Math.max(max[0] - min[0], max[2] - min[2]) / 2 + SPREAD_MARGIN) * SPREAD_SCALE;
        const halfX = g?.x && g.x > 0 ? g.x / 2 : autoHalf;
        const halfZ = g?.z && g.z > 0 ? g.z / 2 : autoHalf;
        const floorY = gridFloorY(FLOOR_Y, dx);
        const topY = gridTopY(floorY, g?.y, max[1], dx, Math.max(max[1] + 3, CEIL_Y));
        const gridAabb: SimulationGridAabb = {
            min: [cx - halfX, floorY, cz - halfZ],
            max: [cx + halfX, topY, cz + halfZ],
        };
        activeGrid = { x: halfX * 2, y: topY - floorY, z: halfZ * 2, auto: !g?.x && !g?.y && !g?.z };
        // eslint-disable-next-line no-console
        console.log(
            `[aquanova] liquefy — room: ${roomOfMesh(group.primaryMesh)}, node: ${nodeNameOfMesh.get(group.primaryMesh) ?? group.primaryMesh.name}, meshes: ${group.samples.length}, particles: ${totalCount}, fluidSim: ${settingName ?? "(default)"}, ` +
                `grid: ${(halfX * 2).toFixed(2)}×${(topY - floorY).toFixed(2)}×${(halfZ * 2).toFixed(2)} m ` +
                `(${Math.ceil((halfX * 2) / dx)}×${Math.ceil((topY - floorY) / dx)}×${Math.ceil((halfZ * 2) / dx)} cells @ dx ${dx.toFixed(3)})`
        );
        const sim = buildFluidSim(setting, {
            count: totalCount,
            particleRadius,
            initialPositions: positions,
            boundsMin: gridAabb.min,
            boundsMax: gridAabb.max,
            dx,
        });
        // Select authored primitives against the exact AABB passed to the solver, never the sampled prop AABB.
        const collision = buildCollisionSet(gridAabb, group.excluded);
        sim.setSceneSdf(collision.spec);
        // eslint-disable-next-line no-console
        console.log(`[aquanova]   collision: ${collision.scratch[0]} primitive(s), ${collision.moving.length} movable`);
        // Render impostor radius, taken from the sample spacing the setting file asked for
        // (`demoParams.particleRadius`) exactly as Liquefactor does. Hardcoding it (it was 0.05) made
        // the splats thinner than the file specifies, which shortens the path light travels through
        // the water and washes out `render.absorption` — the same file looked far more transparent
        // here than in Liquefactor.
        virtualSim.particleRadius = particleRadius;
        virtualSim.surfaceSizeScale = sim.surfaceSizeScale ?? 1;

        // Explosion force field — PREPARED now, APPLIED when the fluid phase begins so the
        // water only erupts once the solid has fully melted.
        const impulseBuffer = device.createBuffer({ label: "aq-impulse", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const impulseSpec: ForceFieldSpec = { struct: "struct ForceFieldParams { center: vec4<f32>, push: vec4<f32>, };", wgsl: IMPULSE_WGSL, buffer: impulseBuffer };
        // Impulse from the setting file: a unit direction (normalised here — its length is meaningless)
        // scaled by the base and the file's intensity, plus the radial burst. `push.xyz` is the uniform
        // push and `push.w` the radial magnitude.
        const imp = setting?.impulse;
        const intensity = imp?.intensity ?? IMPULSE_DEFAULT_INTENSITY;
        const rawDir = imp?.direction ?? IMPULSE_DEFAULT_DIR;
        const dirLen = Math.hypot(rawDir[0], rawDir[1], rawDir[2]);
        // (0,0,0) is not "no direction" — it means "the way the shot was travelling", i.e. from the
        // player through the crosshair, captured when the trigger was pulled.
        const unit = dirLen > 1e-6 ? ([rawDir[0] / dirLen, rawDir[1] / dirLen, rawDir[2] / dirLen] as const) : group.shotDir;
        const dirMag = IMPULSE_DIR_BASE * intensity;

        // The dissolve proper. Every member grows from the shared shot origin; the combined sim is
        // frozen during this phase while each particle segment wriggles with its source mesh.
        const hitPoint = group.hit;
        const hit: [number, number, number] = hitPoint ? [hitPoint[0]!, hitPoint[1]!, hitPoint[2]!] : [cx, cy, cz];
        const maxR = computeMaxR(hit, min, max);

        // The blast is centred on the LIQUEFACTION POINT, not the mesh's bounding-box centre: the water
        // should be thrown away from where the shot landed. `maxR` reaches the farthest corner of the
        // whole linked group's union, so one force field covers every particle in the shared sim.
        // A `render`-style explicit `impulse.radius` overrides it, pinning the reach in world units.
        const blastR = imp?.radius && imp.radius > 0 ? imp.radius : Math.max(maxR, particleRadius * 8, 1);
        device.queue.writeBuffer(
            impulseBuffer,
            0,
            new Float32Array([hit[0], hit[1], hit[2], blastR, unit[0] * dirMag, unit[1] * dirMag, unit[2] * dirMag, IMPULSE_RADIAL_BASE * intensity])
        );
        activeImpulse = { intensity, direction: [unit[0], unit[1], unit[2]], radius: blastR, fromShotRay: dirLen <= 1e-6 };
        activePhysics = { method: setting?.method ?? "(fallback)", ...(setting?.method === "MLS-MPM" || setting?.method === "PB-MPM" ? setting.physics : FLUID_SETTING) };

        // Per-member visual and color state. The Havok bodies remain until the shared fluid phase
        // begins, so linked props continue supporting one another throughout the dissolve.
        const members: DissolveMember[] = [];
        const wantColor = !!setting?.useMeshColors;
        const ls = wantColor ? litScene() : null;
        particleOffset = 0;
        for (const { entry, sample } of group.samples) {
            const memberMin = sample.min;
            const memberMax = sample.max;
            const state = liquefyStates.get(entry.mesh) ?? null;
            const material = state ? (entry.mesh.material ?? null) : null;
            if (state) {
                state.hit = hit;
                state.frontR = 0;
                state.enabled = true;
            }
            if (material) bumpMat(material);

            const texView = wantColor ? (entry.mesh.material as unknown as { baseColorTexture?: { view?: GPUTextureView } } | undefined)?.baseColorTexture?.view : undefined;
            let colorBuffer = wantColor ? buildMeshColorBuffer(sample.count, sample.uvs, texView) : null;
            if (colorBuffer && ls) {
                const lit = buildLitParticleColors(device, sample.count, sample.positions, colorBuffer, ls);
                colorBuffer.destroy();
                colorBuffer = lit;
            }
            members.push({
                mesh: entry.mesh,
                disp: entry.dyn ? entry.dyn.disp : null,
                dyn: entry.dyn,
                state,
                material,
                maxR: computeMaxR(hit, memberMin, memberMax),
                wriggleNode: entry.wriggleNode,
                wriggleBase: entry.wriggleBase,
                particleOffset,
                particleCount: sample.count,
                colorBuffer,
                behaviorAvailability: entry.behaviorAvailability,
                dissolved: false,
            });
            particleOffset += sample.count;
        }

        const active: ActiveSim = {
            sim,
            mesh: group.primaryMesh,
            members,
            phase: "dissolving",
            fluidElapsed: 0,
            fadeElapsed: 0,
            impulseRemaining: 0,
            impulseBuffer,
            impulseSpec,
            impulseActive: intensity > 0,
            waterBase,
            waterScratch: new Float32Array(totalCount * 4),
            useMeshColors: wantColor,
            group,
            collision,
            gridAabb,
        };
        group.sim = active;
        activeSims.push(active);
    }
    liquefyMesh = (mesh: Mesh, hitPoint?: readonly [number, number, number] | null, sourceConfig?: LiquefiableBehaviorConfig): void => {
        if (controlledGroup) return;
        // Melting a node melts it WHOLE: every primitive of that node goes at once (a node with 4
        // primitives is one object to the player). A liquefiable behaviour may also name LINKED
        // nodes that have to melt at the same moment (e.g. a door's two leaves); linked names are
        // matched SHIP-WIDE and the cascade is transitive, following each linked node's own links
        // and their primitives in turn.
        //
        // The group is enumerated FIRST, then dispatched, so the whole shot shares one decision:
        //   • one melt ORIGIN — the weapon hit point, or (for a programmatic fire with no pick
        //     point) the shot mesh's centre. Letting each member start at its own centre made a
        //     door's two leaves burn from two separate spots.
        //   • one fluid SETTING, drawn once from the shot mesh's candidates. Picking per mesh meant
        //     a 4-primitive pod could melt into four different fluids.
        // Enumeration is synchronous; only the sampling itself is async, so the members reach the
        // worker pool together and melt in step.
        const group: Mesh[] = [];
        const seen = new Set<Mesh>();
        const queue: Mesh[] = [mesh];
        while (queue.length) {
            const next = queue.shift()!;
            if (seen.has(next)) continue;
            if (next !== mesh && !behaviorManager.isDissolvable(next)) continue; // already melting / not dissolvable
            seen.add(next);
            group.push(next);
            for (const s of nodePrimitives.get(next) ?? []) queue.push(s);
            for (const name of behaviorManager.getLinkedEntityNames(next)) {
                for (const other of meshesByNodeName.get(name) ?? []) queue.push(other);
            }
        }

        const bMin = mesh.boundMin;
        const bMax = mesh.boundMax;
        const origin: readonly [number, number, number] | null = hitPoint
            ? [hitPoint[0]!, hitPoint[1]!, hitPoint[2]!]
            : bMin && bMax
              ? [(bMin[0]! + bMax[0]!) / 2, (bMin[1]! + bMax[1]!) / 2, (bMin[2]! + bMax[2]!) / 2]
              : null;
        // Candidates come from the SHOT mesh's behaviour (its own `fluidSim` list when it has one,
        // otherwise the manifest's global list); the winner then applies to every member.
        const ownSettings = sourceConfig?.fluidSim ?? behaviorManager.getLiquefiableConfig(mesh)?.fluidSim;
        const candidates = ownSettings?.length ? ownSettings : fluidSettingNames;
        const settingName = candidates.length ? candidates[Math.floor(Math.random() * candidates.length)]! : undefined;
        const setting = settingName ? fluidSettings.get(canonicalSettingName(settingName)) : undefined;
        if (settingName && !setting) {
            // Never degrade quietly: falling back changes both the particle count and the physics.
            // eslint-disable-next-line no-console
            console.warn(`[aquanova] fluidSim "${settingName}" was requested but never loaded — using the default water and sample radius`);
        }

        const shot: ShotGroup = {
            sampling: 0,
            dispatchComplete: false,
            samples: [],
            sim: null,
            direction: 1,
            meshes: new Set(group),
            resumeGeneration: 0,
            resumePending: null,
            primaryMesh: mesh,
            hit: origin,
            setting,
            settingName,
            excluded: new Set(),
            shotDir: shotDirection(origin),
        };
        controlledGroup = shot;
        for (const m of group) requestSample(m, shot);
        shot.dispatchComplete = true;
        finalizeShotIfReady(shot);
    };

    /** Direction from the player to the crosshair, captured AT FIRE TIME — the shot ray. Used when a
     *  setting's `impulse.direction` is (0,0,0), i.e. "blow it the way I shot it". Taken here rather
     *  than at eruption because the melt takes over a second, during which the player is free to move.
     *  Falls back to straight up when the shot has no hit point (a programmatic fire) or the player is
     *  standing exactly on it. */
    function shotDirection(origin: readonly [number, number, number] | null): [number, number, number] {
        if (!origin) return [...IMPULSE_DEFAULT_DIR] as [number, number, number];
        const dx = origin[0] - cam.position.x;
        const dy = origin[1] - cam.position.y;
        const dz = origin[2] - cam.position.z;
        const len = Math.hypot(dx, dy, dz);
        return len > 1e-6 ? [dx / len, dy / len, dz / len] : ([...IMPULSE_DEFAULT_DIR] as [number, number, number]);
    }

    // A shot's own props are kept out of its water's collision set: the water is seeded in the
    // prop's own volume, so leaving it in would eject the seeded particles immediately. Keyed by
    // INSTANCE ID, so this covers static placements too.
    function excludedIds(group: readonly Mesh[]): ReadonlySet<string> {
        const out = new Set<string>();
        for (const m of group) {
            const id = instanceIdOfMesh(m); // the prop that is becoming this water
            if (id) out.add(id);
        }
        return out;
    }

    function cancelDissolve(a: ActiveSim): void {
        const restoredNodes = new Set<SceneNode>();
        for (const member of a.members) {
            if (!restoredNodes.has(member.wriggleNode)) {
                restoredNodes.add(member.wriggleNode);
                member.wriggleNode.position.set(member.wriggleBase[0], member.wriggleBase[1], member.wriggleBase[2]);
            }
            if (member.dyn) member.dyn.wriggling = false;
            if (member.state) {
                member.state.frontR = 0;
                member.state.enabled = false;
            }
            if (member.material) bumpMat(member.material);
            member.colorBuffer?.destroy();
            behaviorManager.restoreMesh(member.mesh, member.behaviorAvailability);
        }
        a.sim.setForceField(null);
        a.sim.dispose();
        a.impulseBuffer.destroy();
        a.collision.buffer.destroy();
        a.group.sim = null;
        releaseControlledGroup(a.group);
    }

    // Transition from the frozen visual dissolve to the running fluid: hide the solid, disable its
    // collision everywhere, remove its Havok body, then mark the simulation ready to step.
    function beginFluidPhase(a: ActiveSim): void {
        const restoredNodes = new Set<SceneNode>();
        const removedBodies = new Set<DynBody>();
        for (const member of a.members) {
            if (!restoredNodes.has(member.wriggleNode)) {
                restoredNodes.add(member.wriggleNode);
                member.wriggleNode.position.set(member.wriggleBase[0], member.wriggleBase[1], member.wriggleBase[2]);
            }
            if (member.state) {
                member.state.enabled = false;
                member.state.frontR = member.maxR;
            }
            if (member.material) bumpMat(member.material);
            setMeshVisible(member.mesh, false);
            if (!member.dyn || removedBodies.has(member.dyn) || !dynBodies.includes(member.dyn)) continue;
            removedBodies.add(member.dyn);
            // Disable collision BEFORE the first fluid step. Any prop resting on this body drops into
            // the erupting water, and every already-running simulation skips the deactivated slots.
            setDynBodyCollisionActive(member.dyn, false);
            if (member.dyn.body) removePhysicsBody(world, member.dyn.body);
            for (const m of member.dyn.meshes) {
                dynBodyByMesh.delete(m);
            }
            const di = dynBodies.indexOf(member.dyn);
            if (di >= 0) dynBodies.splice(di, 1);
        }
        // A zero intensity zeroes both the radial and directional terms, so the pass would evaluate to
        // exactly 0 for every particle. Skip it: the engine builds the external-force compute pass
        // lazily on the first non-null injection, so installing one would compile a shader and add a
        // dispatch per substep for 0.35 s to achieve nothing.
        if (a.impulseActive) {
            a.sim.setForceField(a.impulseSpec);
            a.impulseRemaining = 0.35;
        }
        a.phase = "fluid";
        a.fluidElapsed = 0;
        releaseControlledGroup(a.group);
    }

    // A shot erupts as ONE event: nothing may start simulating until every member has been sampled
    // AND has finished melting. Without this a door's first leaf would burst while the second was
    // still dissolving.
    function eruptIfReady(group: ShotGroup): void {
        const active = group.sim;
        if (!active || active.phase !== "dissolving" || active.members.some((member) => !member.dissolved)) return;
        // The sim starts STEPPING now (it is frozen while dissolving), so this is the first moment the
        // scene SDF matters — and the last moment every linked body is still resolvable.
        beginFluidPhase(active);
    }

    // Per-frame: advance each shot. DISSOLVING — wriggle the solid + water and grow the clip front
    // (the sim does NOT step; the water stays mesh-shaped and is revealed as the solid clips away; the
    // Havok body stays live so props on top keep their support); at full front → beginFluidPhase
    // (hide the solid, disable fluid collision, remove its Havok collider, fire the impulse).
    // FLUID/FADING — step the sim, drive the impulse + lifetime, then fade out and dispose. Encoded
    // BEFORE the surface task reads combinedPos.
    behaviorManager.events.on("frameStart", ({ deltaMs }) => {
        // A newly-inserted task (the MSAA scene pass + its depth resolve) needs the whole graph
        // re-recorded, which re-allocates canvas-sized targets other tasks' bind groups point at.
        // Do it here, at the very top of the frame, before anything is encoded against them.
        if (pendingFrameGraphRebuild) {
            pendingFrameGraphRebuild = false;
            getFrameGraph(scene).build();
        }
        // Open the fluid profiler's frame BEFORE anything is encoded: beginFrame resets the query
        // cursor, frameStart stamps the whole-frame envelope that "aq-timing-resolve" closes.
        if (fluidProfilerOn && fluidProfiler) {
            fluidProfiler.beginFrame();
            fluidProfiler.frameStart(engine._currentEncoder);
        }
        // Has the view actually changed since last frame? Position and target together cover both
        // walking and looking. On the first still frame after moving, burn one frame at factor 1 to
        // throw away the history accumulated at the old viewpoint — otherwise it bleeds in as a
        // slowly-fading ghost for about a second.
        const camMoved =
            Math.abs(cam.position.x - taaPrevCam[0]!) > TAA_STILL_EPS ||
            Math.abs(cam.position.y - taaPrevCam[1]!) > TAA_STILL_EPS ||
            Math.abs(cam.position.z - taaPrevCam[2]!) > TAA_STILL_EPS ||
            Math.abs(cam.target.x - taaPrevCam[3]!) > TAA_STILL_EPS ||
            Math.abs(cam.target.y - taaPrevCam[4]!) > TAA_STILL_EPS ||
            Math.abs(cam.target.z - taaPrevCam[5]!) > TAA_STILL_EPS;
        taaPrevCam[0] = cam.position.x;
        taaPrevCam[1] = cam.position.y;
        taaPrevCam[2] = cam.position.z;
        taaPrevCam[3] = cam.target.x;
        taaPrevCam[4] = cam.target.y;
        taaPrevCam[5] = cam.target.z;
        taaMoving = camMoved;
        if (camMoved) {
            taaResetPending = true;
        } else if (taaResetPending) {
            taaResetPending = false;
            taaTask.factor = 1;
        } else {
            taaTask.factor = TAA_FACTOR;
        }
        canvas.dataset.taaAccum = String(taaOn && !taaMoving);
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
        const growDt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 30);
        // The player can enter a simulation domain long after it was built, so every set reserves a
        // capsule slot and refreshes it from the character controller immediately before fluid steps.
        for (const a of activeSims) {
            const slot = a.collision.playerSlot;
            if (slot === null) continue;
            const live = livePlayerPrimitive();
            a.collision.prims[slot] = live;
            packPrimitive(a.collision.scratch, slot, live);
            device.queue.writeBuffer(a.collision.buffer, (PRIM_HEADER + slot * PRIM_STRIDE) * 4, a.collision.scratch, PRIM_HEADER + slot * PRIM_STRIDE, PRIM_STRIDE);
        }
        // Shots still waiting on their seed: shake the solid so the hit reads instantly. There is no
        // water to shake in lock-step yet — that starts with the ActiveSim below.
        for (const s of samplingShots) {
            if (s.group.direction < 0) continue;
            const ox = (Math.random() * 2 - 1) * WRIGGLE_AMP,
                oy = (Math.random() * 2 - 1) * WRIGGLE_AMP,
                oz = (Math.random() * 2 - 1) * WRIGGLE_AMP;
            s.wriggleNode.position.set(s.wriggleBase[0] + ox, s.wriggleBase[1] + oy, s.wriggleBase[2] + oz);
        }
        canvas.dataset.sampling = String(samplingShots.size);
        inspectOverlay?.onFrame();
        colliderOverlay?.onFrame(); // dynamic proxies move; the chunk changes as you walk
        lightOverlay?.onFrame();
        perfOverlay.onFrame(deltaMs); // FPS + GPU timing readout (P)
        // The room voxels cover EVERY baked room and are world-anchored, so neither walking around nor
        // crossing a chunk boundary changes them. Only the dynamic bodies actually move.
        for (let k = activeSims.length - 1; k >= 0; k--) {
            const a = activeSims[k]!;
            if (a.phase === "dissolving") {
                // Pain-shake each source node and its segment of the combined particle buffer in
                // lock-step. Several render primitives can share one display root, so they must also
                // share one random offset rather than fighting over the node's final pose.
                const wb = a.waterBase,
                    ws = a.waterScratch;
                const offsets = new Map<SceneNode, [number, number, number]>();
                const rate = LIQUEFY_SPEED * growDt * (a.group.direction > 0 ? 1 : -2);
                for (const member of a.members) {
                    let offset = offsets.get(member.wriggleNode);
                    if (!offset) {
                        offset = [(Math.random() * 2 - 1) * WRIGGLE_AMP, (Math.random() * 2 - 1) * WRIGGLE_AMP, (Math.random() * 2 - 1) * WRIGGLE_AMP];
                        offsets.set(member.wriggleNode, offset);
                        member.wriggleNode.position.set(member.wriggleBase[0] + offset[0], member.wriggleBase[1] + offset[1], member.wriggleBase[2] + offset[2]);
                    }
                    const start = member.particleOffset * 4;
                    const end = (member.particleOffset + member.particleCount) * 4;
                    for (let i = start; i < end; i += 4) {
                        ws[i] = wb[i]! + offset[0];
                        ws[i + 1] = wb[i + 1]! + offset[1];
                        ws[i + 2] = wb[i + 2]! + offset[2];
                        ws[i + 3] = 1;
                    }
                    if (member.state) member.state.frontR = Math.max(0, Math.min(member.state.frontR + rate, member.maxR));
                    if (member.material) bumpMat(member.material);
                    member.dissolved = a.group.direction > 0 && (!member.state || member.state.frontR >= member.maxR);
                }
                device.queue.writeBuffer(a.sim.positionBuffer, 0, ws);
                if (a.group.direction < 0 && a.group.resumePending === null && a.members.every((member) => !member.state || member.state.frontR <= 0)) {
                    cancelDissolve(a);
                    activeSims.splice(k, 1);
                    continue;
                }
                if (a.group.direction > 0) eruptIfReady(a.group);
                continue; // do not step the sim while dissolving
            }
            if (a.phase === "fading") {
                a.fadeElapsed += dt;
                if (a.fadeElapsed >= FADE_DUR) {
                    // Dispose BEFORE stepping so we never free a sim's buffers after encoding its step.
                    a.sim.setForceField(null);
                    a.sim.dispose();
                    a.impulseBuffer.destroy();
                    a.collision.buffer.destroy();
                    const removedRoots = new Set<SceneNode>();
                    for (const member of a.members) {
                        member.colorBuffer?.destroy();
                        const root = member.disp ?? (member.mesh as unknown as SceneNode);
                        if (!removedRoots.has(root)) {
                            removedRoots.add(root);
                            removeFromScene(scene, root);
                        }
                    }
                    activeSims.splice(k, 1);
                    // Last of this shot's water gone → put any SDF bodies it excluded back in the union.
                    continue;
                }
            }
            a.sim.step(engine._currentEncoder, dt);
            if (a.impulseRemaining > 0) {
                a.impulseRemaining = Math.max(0, a.impulseRemaining - dt);
                if (a.impulseRemaining === 0) a.sim.setForceField(null);
            }
            if (a.phase === "fluid") {
                a.fluidElapsed += dt;
                if (a.fluidElapsed >= LIFETIME) {
                    a.phase = "fading";
                    a.fadeElapsed = 0;
                    a.impulseRemaining = 0;
                    a.sim.setForceField(null);
                }
            }
        }
        // Aggregate live particles into the shared surface buffer + per-particle alpha (fading ramps 1→0)
        // + per-particle colour (mesh-colour blobs copy their texture colours; others get the water tint).
        let off = 0;
        let anyColor = false;
        for (const a of activeSims) {
            const n = a.sim.count;
            if (off + n <= MAX_TOTAL) {
                engine._currentEncoder.copyBufferToBuffer(a.sim.positionBuffer, 0, combinedPos, off * 16, n * 16);
                if (a.useMeshColors) {
                    for (const member of a.members) {
                        if (member.colorBuffer) {
                            engine._currentEncoder.copyBufferToBuffer(member.colorBuffer, 0, combinedColor, (off + member.particleOffset) * 16, member.particleCount * 16);
                        } else {
                            device.queue.writeBuffer(combinedColor, (off + member.particleOffset) * 16, colorScratch, 0, member.particleCount * 4);
                        }
                    }
                    anyColor = true;
                } else {
                    device.queue.writeBuffer(combinedColor, off * 16, colorScratch, 0, n * 4); // plain water tint
                }
                const alpha = a.phase === "fading" ? Math.max(0, 1 - a.fadeElapsed / FADE_DUR) : 1;
                alphaScratch.fill(alpha, off, off + n);
                off += n;
            }
        }
        surfaceTask.setUseParticleColor(anyColor); // shared toggle: on when any active blob carries mesh colours
        if (off > 0) {
            device.queue.writeBuffer(combinedAlpha, 0, alphaScratch, 0, off);
            virtualSim.count = off;
        } else {
            virtualSim.count = 0;
        }
        canvas.dataset.particleCount = String(off);
        canvas.dataset.activeSims = String(activeSims.length);
    });
    behaviorManager.bindSystemEvents(scene, world);

    enableMaterialPlugins(scene);
    await registerScene(scene);
    if (lightUtility) await registerUtilityLayer(lightUtility);
    await startEngine(engine);
    await warmUpFluidPipelines();
    canvas.dataset.ready = "true";
}
