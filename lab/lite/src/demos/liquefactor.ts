// Liquefactor demo — MULTI-TARGET shooting gallery with concurrent GPU fluid sims.
//
// Three "enemy" meshes float over the ground. Click one to LIQUEFY it: its surface
// dissolves radially from the hit point (the material clips inside a growing world-space
// "front"), the interior is CPU volume-sampled into particles, and an independent GPU
// fluid sim (grid centred on THAT mesh, at ground level) launches the blob upward and
// melts it into a puddle. Multiple meshes can be dissolving / running at the same time —
// each mesh owns its own sim (grid, particle count, physics). A few seconds after a mesh
// liquefies, its sim fades away (sinks through the floor) and the mesh is gone for good.
// "Restart" stops every sim and restores all three solid meshes.
//
// Rendering keeps demo-fluid's screen-space surface but composites ALL live sims in ONE
// pass: every frame each sim's render positions are copied into a shared "combined" buffer
// and a single fluid-surface task reconstructs the water from it. The scene (ground + HDR
// skybox, minus any dissolving foe) draws into `sceneColorRT`; the surface pass composites
// that into the swapchain; then a single foe-overlay pass draws every dissolving foe on top,
// each clipping inside its own front to reveal the one water render beneath.

import {
    addAnimationGroups,
    addTask,
    addToScene,
    computeDeformedPositions,
    createAnimationManager,
    createFreeCamera,
    createHavokWorld,
    createPhysicsBody,
    createPhysicsShape,
    createDirectionalLight,
    createEngine,
    createGpuPicker,
    createGround,
    createHemisphericLight,
    createRenderTarget,
    createRenderTask,
    createSceneContext,
    createStandardMaterial,
    createTransformNode,
    enableMaterialPlugins,
    getPhysicsBodyLinearVelocity,
    getProjectionMatrix,
    getViewMatrix,
    getViewProjectionMatrix,
    isPbrMaterial,
    onPhysicsAfterStep,
    PhysicsMotionType,
    PhysicsShapeType,
    setParent,
    setPhysicsBodyAngularVelocity,
    setPhysicsBodyLinearVelocity,
    setPhysicsBodyShape,
    setPhysicsBodyTransform,
    setPhysicsTimestepMs,
    loadEnvironment,
    loadHdrEnvironment,
    loadGltf,
    onBeforeRender,
    pauseAnimation,
    pickAsync,
    playAnimation,
    registerScene,
    setMeshVisible,
    startEngine,
    updateAnimationManager,
} from "babylon-lite";
import type { AnimationGroup, EnvironmentTextures, Material, Mesh, PhysicsBody, PhysicsWorld, Renderable, SceneNode } from "babylon-lite";
import HavokPhysics from "@babylonjs/havok";
import { createLiquefyPlugin } from "./liquefy-plugin.js";
import { buildLitParticleColors } from "./particle-lit-colors.js";
import type { LitColorScene } from "./particle-lit-colors.js";
import {
    DEFAULT_SHIP_IBL_STRENGTH,
    findEntityWithBehavior,
    isDynamicBehavior,
    isLiquefiableBehavior,
    linkedMeshNames,
    PLAYER_START_BEHAVIOR,
    resolveBehavior,
    resolveExposure,
    resolveToneMapping,
    WEAPON_START_BEHAVIOR,
} from "./ship-manifest.js";
import type { ShipBehavior, ShipBehaviorLibrary, ShipEntities, ShipEnvironment } from "./ship-manifest.js";
import type { LiquefyState } from "./liquefy-plugin.js";
import { createMlsMpmSim } from "babylon-lite/fluid/mls-mpm-sim.js";
import { createPbfSim } from "babylon-lite/fluid/pbf-sim.js";
import { createPbMpmSim, pbmpmParamKeysForMaterial } from "babylon-lite/fluid/pbmpm-sim.js";
import type { FluidSim, ForceFieldSpec, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import { createFluidSurfaceTask } from "babylon-lite/fluid/fluid-surface-render.js";
import { fillMeshParticles } from "./particle-fill.js";
import type { MeshFillStrategy, MeshParticleFill } from "./particle-fill.js";
import type { VolumeSamplingMode } from "babylon-lite/fluid/volume-sampling/index.js";
import { buildHdrSkyboxRenderable } from "babylon-lite/material/pbr/background-hdr-skybox.js";
import { createFluidControlsPanel, DEFAULT_FLUID_SCHEMAS } from "babylon-lite/fluid/controls-panel.js";
import type { PhysSchemaEntry } from "babylon-lite/fluid/controls-panel.js";
import { createFluidProfiler } from "./fluid/gpu-profiler.js";
import type { FluidProfilerImpl } from "./fluid/gpu-profiler.js";
import { exportJsonFromPairState, presetFromExportJson } from "./fluid/preset-io.js";
import type { FluidExportJson } from "./fluid/preset-io.js";
import type { PairState } from "./fluid/demo.js";
import { gridFloorY, gridTopY } from "./fluid/grid-bounds.js";
import { demoAssetUrl } from "./demo-asset-url.js";

// Box-preset render defaults (fluid/scenes/box.ts MLS-MPM preset).
const DEF_COLOR = "#16a3c3";

const DEF_ABSORPTION = 0.4;
const DEF_SIZE = 0.7;
const DEF_REFRACTION = 0.06;
const DEF_SPECULAR = 41;
const DEF_DEPTH_BLUR = 40;
const DEF_DEPTH_BLUR_THRESHOLD = 41;
const DEF_THICKNESS_BLUR = 16;
const DEF_HALF = true;
const DEF_THICKNESS_DOWNSCALE = 6;
const DEF_SURFACE_FILTER: "bilateral" | "narrowRange" = "narrowRange";
const DEF_NARROW_DELTA = 10;
const DEF_NARROW_MU = 1;

// Which foe set the demo auditions, read from the URL. Declared up here because scene sizing below
// branches on it.
const FOE_SET: string = new URLSearchParams(location.search).get("foes") === "ship" ? "ship" : "props";
/** Ship mode auditions the Aquanova meshes, and deliberately reproduces that demo's scene setup
 *  (grading, lighting, environment build, camera clip range, puddle spread) so the two can be
 *  compared frame by frame. The gallery keeps its own presentation. */
const SHIP_MODE = FOE_SET === "ship";

// World layout. Foes rest on the ground plane at y = 0; each sim grid drops one unit below
// ground so the ground BC isn't fighting the grid's own border.
const GROUND_Y = 0;
const SPREAD_MARGIN = 9; // extra half-width (world units) around a mesh footprint for puddle spread
const LIQUEFY_SPEED = 6; // dissolve front growth (world units / s)
// Ship modules are dense and read as architecture rather than a single prop, so the dissolve front
// sweeping across them at gallery speed looks like a pop rather than a melt — run it a third as fast.
const LIQUEFY_SPEED_SHIP = LIQUEFY_SPEED / 3;
const LIFETIME = 5.0; // seconds a running sim lives before it starts fading
const FADE_DUR = 1.2; // seconds the alpha fade-out takes before dispose
const WRIGGLE_AMP = 0.04; // cartoon "pain" jitter amplitude (world units)
const IMPULSE_RADIAL_BASE = 18; // outward explosion accel from the volume centre, all directions (× impulse intensity)
// Uniform push along the configurable impulse direction (× impulse intensity). Kept in sync with
// Aquanova, which reads the direction + intensity from the setting file: only these two bases live in
// code, so a prop auditioned here bursts identically there.
const IMPULSE_DIR_BASE = 6;
const IMPULSE_DEFAULT_DIR: readonly [number, number, number] = [0, 1, 0]; // straight up — the original lift
const MAX_TOTAL = 600000; // combined render-buffer capacity (particles across all live sims)

// Studio HDR environment — drives the fluid-surface reflections + the skybox background.
const ENV_STUDIO_URL = "https://playground.babylonjs.com/textures/environment.env";
const SUN_DIR: [number, number, number] = [-0.4, -0.82, -0.45];

// Textured glTF foes — each is loaded, auto-fit to a common size, sat on the ground, and (when the
// model is skinned/animated) played + sampled in its CURRENT deformed pose. Their diffuse textures
// drive the per-particle "Use mesh colours" render. Alien + CesiumMan are animated (skinned).
const MODEL_FOES: { key: string; url: string; x: number; z?: number; ry?: number; surfaceOnly?: boolean; scale?: number }[] = [
    { key: "alien", url: "https://playground.babylonjs.com/scenes/Alien/Alien.gltf", x: -15, z: 6 },
    { key: "barrel", url: "https://assets.babylonjs.com/meshes/ExplodingBarrel.glb", x: -5, z: 6, ry: -Math.PI / 2, surfaceOnly: true },
    { key: "house", url: "https://assets.babylonjs.com/meshes/haunted_house.glb", x: 5, z: 6, surfaceOnly: true, scale: 4 },
    { key: "cesium", url: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMan/glTF-Binary/CesiumMan.glb", x: 15, z: 6 },
];
const MODEL_TARGET_SIZE = 7; // auto-fit: scale each model so its largest dimension ≈ this many world units

// ── Foe set: the sample models, or the Aquanova ship's own liquefiable meshes ──────────────────
// `?foes=ship` swaps the gallery for the EXACT ship the Aquanova demo liquefies (ship.glb at its
// authored scale, every mesh made a foe). Same geometry, materials and trim atlas as Aquanova, but
// under Liquefactor's lighting, camera and controls — so the water look can be compared directly
// between the two demos.
//
// This is a URL param rather than a live toggle on purpose: the radial-clip liquefy plugin only
// materialises for meshes present in the INITIAL scene build (see loadModelInstance), so foes
// cannot be swapped after boot without leaving them invisible. The picker below reloads the page.
const SHIP_URL = "/aquanova/ship.glb";
const SHIP_MANIFEST_URL = "/aquanova/ship_manifest.json";
const FOE_SETS = [
    { key: "props", label: "Sample props (default)" },
    { key: "ship", label: "Aquanova ship meshes" },
] as const;

// PB-MPM material (0 liquid, 1 elastic, 2 sand, 3 viscoelastic). Only PB-MPM branches on it.
const PBMPM_MATERIAL_LABELS: [string, number][] = [
    ["Liquid", 0],
    ["Elastic", 1],
    ["Sand", 2],
    ["Viscoelastic", 3],
];

// Impulse force field: an EXPLOSION FROM THE INSIDE. Every particle inside the blast radius is
// pushed radially OUTWARD from the volume centre (push.w, ~uniform through the core), plus a uniform
// push along a configurable direction (push.xyz, already scaled: unit direction × base × intensity)
// so the burst can be aimed. center.xyz = volume centre, center.w = blast radius (set well beyond the
// blob so the whole volume is in the flat region).
const IMPULSE_WGSL = /* wgsl */ `
fn externalForce(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> vec3<f32> {
    let center = forceFieldParams.center.xyz;
    let radius = max(forceFieldParams.center.w, 1.0e-4);
    let toParticle = pos - center;
    let dist = length(toParticle);
    var f = forceFieldParams.push.xyz; // uniform push along the configured direction
    if (dist < radius) {
        let dir = select(vec3<f32>(0.0, 1.0, 0.0), toParticle / max(dist, 1.0e-4), dist > 1.0e-4);
        // ~uniform outward blast through the core; ramp up over the innermost 15% to avoid a hard
        // direction flip on particles sitting right at the centre.
        let core = smoothstep(0.0, 0.15, dist / radius);
        f += dir * forceFieldParams.push.w * core;
    }
    return f * dt;
}`;

function hexToRgb(hex: string): [number, number, number] {
    return [parseInt(hex.slice(1, 3), 16) / 255, parseInt(hex.slice(3, 5), 16) / 255, parseInt(hex.slice(5, 7), 16) / 255];
}

type InstancePhase = "solid" | "dissolving" | "fluid" | "fading" | "gone";

type TexInfo = { view: GPUTextureView; width: number; height: number };

interface Instance {
    readonly key: string;
    readonly meshes: Mesh[]; // display sub-meshes (1 procedural, N for the skinned model)
    readonly materials: Material[]; // clip-plugin hosts + per-frame UBO bump
    readonly root: SceneNode; // wriggle / placement target (the mesh itself, or a parent transform)
    readonly deformable: boolean; // sample the CURRENT animated pose (skinned model) instead of static geometry
    readonly surfaceOnly: boolean; // skip volume sampling (hollow prop) → dense barycentric-UV surface shell
    readonly x: number;
    readonly baseY: number; // world Y the mesh rests at (used for the static-sample world offset)
    homePos: [number, number, number]; // resting root position (auto-fit centred); Restart resets here. Rewritten if the root is later reparented under a physics display root.
    readonly diffuseTexs: TexInfo[]; // distinct base-colour textures across the sub-meshes (per-particle colour source)
    readonly meshTexIndex: number[]; // per display sub-mesh: its index into diffuseTexs (-1 if untextured)
    readonly baseColor: [number, number, number]; // fallback per-particle colour when there's no texture
    colorBuffer: GPUBuffer | null; // per-particle RGBA colour (filled once at shot time), fed to the surface renderer
    animation: AnimationGroup | null; // looping walk (paused on shot, resumed on Restart)
    readonly liquefyState: LiquefyState;
    readonly impulseBuffer: GPUBuffer;
    readonly impulseSpec: ForceFieldSpec;
    sim: FluidSim | null;
    phase: InstancePhase;
    sampling: boolean; // true while the worker is volume-sampling this foe (before dissolve starts)
    shell: boolean; // which path produced the CURRENT particles: surface shell (true) or volume fill (false)
    sampleId: number; // id of the in-flight sample request (stale results are ignored)
    maxR: number;
    /** Members of the shot this instance belongs to (its node's primitives + every linked node). One
     *  shared array per shot, so the whole group can erupt on the same frame. Null when solid. */
    shot: Instance[] | null;
    /** True once this member's clip front has swept its whole volume. It then HOLDS — fully clipped,
     *  water still frozen — until every member of `shot` has also finished, so a door's two panes
     *  erupt together instead of the nearer one (smaller maxR) bursting first. */
    dissolved: boolean;
    /** Unit direction from the camera to the click point, captured when the shot was fired. Used for
     *  an impulse direction of (0,0,0) — "push it the way I shot it". */
    shotDir: [number, number, number];
    count: number;
    radius: number;
    impulseRemaining: number;
    fluidElapsed: number;
    fadeElapsed: number;
    // wriggle
    wriggling: boolean; // mesh pain-shake active (starts on click, before the sim exists)
    wriggleBase: [number, number, number];
    wriggleWaterBase: Float32Array | null;
    wriggleWaterScratch: Float32Array | null;
}

async function main(): Promise<void> {
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;

    let requiredLimits: Record<string, number> | undefined;
    try {
        const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
        if (adapter) {
            requiredLimits = {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
            };
        }
    } catch {
        // Fall back to default limits.
    }

    const engine = await createEngine(canvas, { msaaSamples: 1, requiredLimits });
    const scene = createSceneContext(engine, { defaultRenderTask: false });

    // Free-fly camera: WASD strafes in the view plane, Space/C rise and fall, and RIGHT-drag looks
    // around — LMB stays free for shooting. An arc-rotate camera made the ship interior a chore to
    // navigate (orbiting around a target you cannot see past). Shift is deliberately NOT the descend
    // key: ship mode uses Shift+LMB to liquefy.
    const cam = createFreeCamera({ x: -3, y: 31.6, z: -40.7 }, { x: -3, y: 2.5, z: 0 }); // the old arc-rotate vantage (alpha -PI/2, beta 0.95, radius 50)
    cam.nearPlane = 0.1;
    // Ship mode matches Aquanova's clip range so depth precision (and therefore any depth-derived
    // shading) is identical between the two demos.
    cam.farPlane = SHIP_MODE ? 400 : 200;
    cam.speed = FOE_SET === "ship" ? 8.8 : 36; // the ship interior is metres across, the gallery tens
    scene.camera = cam;

    // Yaw/pitch are the camera's own state; we drive them from RMB drag and rebuild `target` each
    // frame. `position`/`target` are ObservableVec3 — assign COMPONENTS, never replace the objects,
    // or the view matrix stops being flagged dirty.
    let camYaw = 0;
    let camPitch = 0;
    {
        const dx = cam.target.x - cam.position.x;
        const dy = cam.target.y - cam.position.y;
        const dz = cam.target.z - cam.position.z;
        camYaw = Math.atan2(dx, dz);
        camPitch = Math.atan2(dy, Math.hypot(dx, dz));
    }
    const camKeys = new Set<string>();
    let looking = false;
    const LOOK_SENS = 1 / 350;
    canvas.addEventListener("contextmenu", (e) => e.preventDefault()); // RMB is the look button
    canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 2) return;
        looking = true;
        canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointerup", (e) => {
        if (e.button !== 2) return;
        looking = false;
        if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
        if (!looking) return;
        camYaw += e.movementX * LOOK_SENS;
        camPitch = Math.max(-1.45, Math.min(1.45, camPitch - e.movementY * LOOK_SENS));
    });
    window.addEventListener("keydown", (e) => {
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
        camKeys.add(e.code);
    });
    window.addEventListener("keyup", (e) => camKeys.delete(e.code));
    window.addEventListener("blur", () => camKeys.clear());
    const updateCamera = (deltaMs: number): void => {
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 30);
        const cy = Math.cos(camYaw),
            sy = Math.sin(camYaw);
        const cp = Math.cos(camPitch),
            sp = Math.sin(camPitch);
        const fwdZ = (camKeys.has("KeyW") ? 1 : 0) - (camKeys.has("KeyS") ? 1 : 0);
        const strafe = (camKeys.has("KeyD") ? 1 : 0) - (camKeys.has("KeyA") ? 1 : 0);
        const rise = (camKeys.has("Space") ? 1 : 0) - (camKeys.has("KeyC") ? 1 : 0);
        if (fwdZ || strafe || rise) {
            const spd = cam.speed * (camKeys.has("ShiftLeft") || camKeys.has("ShiftRight") ? 4 : 1) * dt;
            cam.position.x += (sy * cp * fwdZ + cy * strafe) * spd;
            cam.position.y += (sp * fwdZ + rise) * spd;
            cam.position.z += (cy * cp * fwdZ - sy * strafe) * spd;
        }
        cam.target.x = cam.position.x + sy * cp;
        cam.target.y = cam.position.y + sp;
        cam.target.z = cam.position.z + cy * cp;
    };

    // Ship mode is lit purely by the HDRI, exactly as Aquanova is. The lights are not merely dimmed —
    // they are never registered, because a PBR material compiled with lights present renders brighter
    // than one compiled without, regardless of intensity.
    const hemiLight = createHemisphericLight([0.2, 1, 0.3], 0.8);
    if (FOE_SET !== "ship") addToScene(scene, hemiLight);
    const sun = createDirectionalLight(SUN_DIR, 2.0);
    sun.position.set(12, 20, 10);
    if (FOE_SET !== "ship") addToScene(scene, sun);

    const ground = createGround(engine, { width: 60, height: 60, subdivisions: 1 });
    const groundMat = createStandardMaterial();
    groundMat.diffuseColor = [0.22, 0.24, 0.28];
    groundMat.specularColor = [0.04, 0.04, 0.05];
    ground.material = groundMat;
    addToScene(scene, ground);

    const device = engine._device;

    // Opt-in GPU timing (timestamp-query). Null when the host GPU lacks the feature — the panel's
    // GPU section then shows "unavailable". Wired onto every sim + the surface task.
    let profiler: FluidProfilerImpl | null = null;
    try {
        profiler = createFluidProfiler(device);
    } catch {
        profiler = null;
    }

    // ── Enemy instances (loaded glTF models) ─────────────────────────────────
    // Each instance owns its OWN material + liquefy state so its dissolve clip is independent.
    function makeImpulse(key: string): { impulseBuffer: GPUBuffer; impulseSpec: ForceFieldSpec } {
        const impulseBuffer = device.createBuffer({ label: `liq-impulse-${key}`, size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const impulseSpec: ForceFieldSpec = { struct: "struct ForceFieldParams { center: vec4<f32>, push: vec4<f32>, };", wgsl: IMPULSE_WGSL, buffer: impulseBuffer };
        return { impulseBuffer, impulseSpec };
    }

    const isMeshNode = (node: SceneNode): node is Mesh => "_gpu" in node && "material" in node;

    const instances: Instance[] = [];
    const meshToInstance = new Map<Mesh, Instance>();
    const instanceOf = (m: unknown): Instance | undefined => meshToInstance.get(m as Mesh);
    const animManager = createAnimationManager({ engine });

    // Load a textured glTF model as a foe: parent it under an auto-fit root (scaled so its largest
    // dimension ≈ MODEL_TARGET_SIZE, centred on cfg.x and sat on the ground), attach the radial-clip
    // liquefy plugin per PBR material, play its first animation if any, and capture its diffuse
    // texture for per-particle colouring. Sampled in its CURRENT deformed pose on a shot.
    async function loadModelInstance(cfg: { key: string; url: string; x: number; z?: number; ry?: number; surfaceOnly?: boolean; scale?: number }): Promise<void> {
        let asset;
        try {
            asset = await loadGltf(engine, cfg.url);
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn(`[liquefactor] ${cfg.key} load failed`, err);
            return;
        }
        const liquefyState: LiquefyState = { hit: [0, 0, 0], frontR: 0, edge: 0.6, enabled: false };
        const ry = cfg.ry ?? 0; // optional Y spin so a labelled face (e.g. the barrel logo) points at the camera
        const root = createTransformNode(`${cfg.key}_root`, 0, 0, 0, 0, Math.sin(ry / 2), 0, Math.cos(ry / 2), 1, 1, 1);
        const meshes: Mesh[] = [];
        const materials = new Set<Material>();
        for (const e of asset.entities) {
            if (!("position" in e)) continue; // skip non-node entities (e.g. lights)
            e.parent = root;
            root.children.push(e);
            collectFoeMeshes(e, meshes, materials, liquefyState);
        }
        if (meshes.length === 0) {
            // eslint-disable-next-line no-console
            console.warn(`[liquefactor] ${cfg.key} has no CPU-geometry meshes`);
            return;
        }
        const anim = asset.animationGroups?.find((g) => /walk|run|idle/i.test(g.name)) ?? asset.animationGroups?.[0] ?? null;
        registerFoeInstance(cfg, root, meshes, materials, liquefyState, anim);
    }

    // Walk a loaded subtree, gathering its CPU-geometry meshes + materials and attaching the
    // radial-clip liquefy plugin. Shared by the per-model loader and the ship-mesh loader.
    function collectFoeMeshes(node: SceneNode, meshes: Mesh[], materials: Set<Material>, liquefyState: LiquefyState): void {
        if (isMeshNode(node) && node._cpuPositions && node._cpuIndices) {
            meshes.push(node);
            // Attach the radial-clip liquefy plugin (PBR host). This works on SKINNED models
            // because they are materialized in the INITIAL scene build (loaded before registerScene);
            // a dynamic post-boot add leaves the plugin UBO uninitialised and the mesh invisible.
            if (node.material && isPbrMaterial(node.material) && !node.material.plugins?.some((p) => p.name === "liquefy")) {
                node.material.plugins = [...(node.material.plugins ?? []), createLiquefyPlugin(() => liquefyState, "pbr")];
            }
            if (node.material) materials.add(node.material);
        }
        for (const c of node.children ?? []) collectFoeMeshes(c, meshes, materials, liquefyState);
    }

    // Fit an assembled foe root onto the gallery row and register it as a live Instance.
    function registerFoeInstance(
        cfg: { key: string; x: number; z?: number; surfaceOnly?: boolean; scale?: number; inPlace?: boolean },
        root: SceneNode,
        meshes: Mesh[],
        materials: Set<Material>,
        liquefyState: LiquefyState,
        anim: AnimationGroup | null
    ): void {
        // Add to the scene FIRST so the world-matrix state is wired up — reading worldMatrix before
        // this yields a partial (pre-hierarchy) transform and mis-fits the model. `inPlace` foes are
        // already inside an added hierarchy (the ship), so adding again would double-register them.
        if (!cfg.inPlace) addToScene(scene, root);
        // Auto-fit: world AABB at scale 1 (root at origin) → uniform scale so the largest extent ≈
        // MODEL_TARGET_SIZE, then centre on cfg.x and sit the model's base on the ground.
        let minx = Infinity,
            miny = Infinity,
            minz = Infinity,
            maxx = -Infinity,
            maxy = -Infinity,
            maxz = -Infinity;
        const measure = (): void => {
            minx = miny = minz = Infinity;
            maxx = maxy = maxz = -Infinity;
            for (const m of meshes) {
                const p = m._cpuPositions!;
                const w = m.worldMatrix as unknown as ArrayLike<number>;
                for (let i = 0; i < p.length; i += 3) {
                    const lx = p[i]!,
                        ly = p[i + 1]!,
                        lz = p[i + 2]!;
                    const wx = w[0]! * lx + w[4]! * ly + w[8]! * lz + w[12]!;
                    const wy = w[1]! * lx + w[5]! * ly + w[9]! * lz + w[13]!;
                    const wz = w[2]! * lx + w[6]! * ly + w[10]! * lz + w[14]!;
                    if (wx < minx) minx = wx;
                    if (wx > maxx) maxx = wx;
                    if (wy < miny) miny = wy;
                    if (wy > maxy) maxy = wy;
                    if (wz < minz) minz = wz;
                    if (wz > maxz) maxz = wz;
                }
            }
        };
        measure();
        const extent = Math.max(maxx - minx, maxy - miny, maxz - minz, 1e-3);
        // `inPlace` foes are already posed by their own hierarchy — their root is an identity node that
        // exists only so wriggle/restart can jitter them without touching the surrounding ship.
        const scale = cfg.inPlace ? 1 : (MODEL_TARGET_SIZE / extent) * (cfg.scale ?? 1);
        const cx = (minx + maxx) / 2;
        const cz = (minz + maxz) / 2;
        const homePos: [number, number, number] = cfg.inPlace ? [0, 0, 0] : [cfg.x - cx * scale, GROUND_Y - miny * scale, (cfg.z ?? 0) - cz * scale];
        root.scaling.set(scale, scale, scale);
        root.position.set(homePos[0], homePos[1], homePos[2]);
        if (anim) {
            anim.loopAnimation = true;
            playAnimation(anim);
            addAnimationGroups(animManager, [anim]);
        }
        const { impulseBuffer, impulseSpec } = makeImpulse(cfg.key);
        // Collect the DISTINCT base-colour textures across sub-meshes (a model may use several — the
        // haunted house has two) and record which texture each sub-mesh uses, so every particle can be
        // coloured from ITS OWN mesh's texture. Only real images (>1x1) count; a 1x1 is a flat factor.
        const diffuseTexs: TexInfo[] = [];
        const meshTexIndex: number[] = [];
        for (const m of meshes) {
            const t = (m.material as unknown as { baseColorTexture?: TexInfo } | undefined)?.baseColorTexture;
            if (!t?.view || t.width <= 1 || t.height <= 1) {
                meshTexIndex.push(-1);
                continue;
            }
            let idx = diffuseTexs.findIndex((d) => d.view === t.view);
            if (idx < 0) {
                idx = diffuseTexs.length;
                diffuseTexs.push({ view: t.view, width: t.width, height: t.height });
            }
            meshTexIndex.push(idx);
        }
        const inst: Instance = {
            key: cfg.key,
            meshes,
            materials: [...materials],
            root,
            deformable: true,
            surfaceOnly: cfg.surfaceOnly ?? false,
            x: cfg.x,
            baseY: GROUND_Y,
            homePos,
            diffuseTexs,
            meshTexIndex,
            baseColor: [0.8, 0.8, 0.8],
            colorBuffer: null,
            animation: anim,
            liquefyState,
            impulseBuffer,
            impulseSpec,
            sim: null,
            phase: "solid",
            sampling: false,
            shell: false,
            sampleId: 0,
            maxR: 0,
            shot: null,
            dissolved: false,
            shotDir: [...IMPULSE_DEFAULT_DIR] as [number, number, number],
            count: 0,
            radius: 0.08,
            impulseRemaining: 0,
            fluidElapsed: 0,
            fadeElapsed: 0,
            wriggling: false,
            wriggleBase: [0, 0, 0],
            wriggleWaterBase: null,
            wriggleWaterScratch: null,
        };
        instances.push(inst);
        for (const m of meshes) meshToInstance.set(m, inst);
        invalidateFilteredSceneTasks();
        setStatus();
    }

    // Ship mode: primitives of one glTF node are one object to the player, so a shot melts the whole
    // group (mirroring Aquanova). Empty for the gallery, where each foe is already a single instance.
    const nodeGroups = new Map<SceneNode, Instance[]>();
    const groupOfInstance = new Map<Instance, Instance[]>();
    // Ship mode also mirrors Aquanova's MANIFEST semantics, so the same click melts the same group:
    // a node's `behaviors` supply its linked partners. The FLUID itself always comes from the demo's
    // own panel — auditioning settings is the point of this demo.
    const behaviorOfInstance = new Map<Instance, ShipBehavior>();
    const instancesByNodeName = new Map<string, Instance[]>();
    // Display roots driven by a Havok body (one per `dynamic` node). `rest` is the spawn pose, which
    // Restart teleports the body back to.
    const dynDisplays: { proxy: SceneNode; body: PhysicsBody; disp: SceneNode; rest: [number, number, number] }[] = [];
    let physWorld: PhysicsWorld | null = null;

    // ── Aquanova ship foes (?foes=ship) ──────────────────────────────────────
    // Load the FULL ship.glb exactly as Aquanova does and make every ship mesh a foe, IN PLACE inside
    // the ship. Geometry, materials, trim atlas, world transforms and the particle-fill path are all
    // identical to Aquanova, so the only variables left are Liquefactor's lighting, camera and fluid
    // controls. ship_manifest.json's `entities` names are used only to pick which room to frame.
    async function loadShipFoes(): Promise<void> {
        await shipManifestReady;
        const chunkFocus = (c: { aabb: { min: number[]; max: number[] } }): [number, number, number] => [
            -(c.aabb.min[0]! + c.aabb.max[0]!) / 2, // glTF space → Lite negates X
            1.5,
            (c.aabb.min[2]! + c.aabb.max[2]!) / 2,
        ];
        const iblStrength = shipManifest?.environment?.dynamicStrength ?? shipManifest?.environment?.strength ?? DEFAULT_SHIP_IBL_STRENGTH;
        // A chunk is a room the ship editor laid out, and an EMPTY one (no meshes placed yet) is
        // exported with `aabb: null` — it has no spatial extent to describe. Every use here is
        // geometric (framing a room, hit-testing a point against one, building its collision shell),
        // so those are filtered out once rather than guarded at each site. Dereferencing them is
        // what broke this demo when the editor gained an empty room.
        //
        // The component check matches Aquanova's own loader (aquanova/manifest.ts), which reads the
        // same file: a truthy-but-malformed AABB would not crash, it would quietly yield `undefined`
        // corner components and propagate NaN into the room test and the collision box extents.
        const hasBounds = (c: { aabb: { min: number[]; max: number[] } | null }): c is { aabb: { min: number[]; max: number[] } } =>
            c.aabb?.min?.length === 3 && c.aabb?.max?.length === 3;
        const chunks = (shipManifest?.chunks ?? []).filter(hasBounds);
        const liqNames = new Set(
            Object.keys(shipManifest?.entities ?? {}).filter((name) => isLiquefiableBehavior(resolveBehavior(shipManifest?.behaviors, shipManifest?.entities, name)))
        );
        let focus: [number, number, number] = chunks[0] ? chunkFocus(chunks[0]) : [0, 1.5, 0];
        let asset;
        try {
            asset = await loadGltf(engine, SHIP_URL);
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[liquefactor] ship.glb load failed", err);
            return;
        }
        const shipRoot = createTransformNode("ship_root", 0, 0, 0, 0, 0, 0, 1, 1, 1);
        for (const e of asset.entities) {
            if (!("position" in e)) continue; // skip non-node entities (e.g. lights)
            e.parent = shipRoot;
            shipRoot.children.push(e);
        }
        addToScene(scene, shipRoot);

        // Match Aquanova's material setup on the ship:
        //  • Drop KHR_materials_transmission. Both demos composite the fluid surface into the swapchain
        //    AFTER the scene renders, but transmission retargets the scene task to an offscreen HDR
        //    buffer and appends a tonemap pass that would hide the water. (It also keeps the refraction
        //    fragment out of the composed shader, which otherwise collides with Liquefactor's punctual
        //    lights and aborts the material build.)
        //  • environmentIntensity from the manifest's dynamic value — this demo loads the authored
        //    unbaked ship, so it should match Aquanova's dynamic/unbaked materials.
        const seenMats = new Set<object>();
        const prepMaterials = (node: SceneNode): void => {
            const mat = (node as Mesh).material;
            if (mat && isPbrMaterial(mat) && !seenMats.has(mat)) {
                seenMats.add(mat);
                const ss = (mat as unknown as { subsurface?: { refraction?: unknown } }).subsurface;
                if (ss?.refraction) ss.refraction = undefined;
                (mat as unknown as { environmentIntensity?: number }).environmentIntensity = iblStrength;
            }
            for (const c of node.children ?? []) prepMaterials(c);
        };
        prepMaterials(shipRoot);

        // Every ship mesh is a foe — click anything to liquefy it. `owner` is the glTF node the mesh
        // belongs to, collapsing the `_primitiveN` wrapper nodes a Babylon re-export bakes in (a
        // mesh-less "storageDoorL" holding "storageDoorL_primitive0"), so a node's primitives group
        // under one entity exactly as they do in Aquanova.
        const PRIMITIVE_WRAPPER = /_primitive\d+$/;
        const found: { mesh: Mesh; parent: SceneNode; owner: SceneNode }[] = [];
        const seen = new Set<Mesh>();
        const walk = (node: SceneNode, owner: SceneNode): void => {
            for (const c of [...(node.children ?? [])]) {
                if (isMeshNode(c) && c._cpuPositions && c._cpuIndices && !seen.has(c)) {
                    seen.add(c);
                    found.push({ mesh: c, parent: node, owner });
                    continue;
                }
                const wrapper = PRIMITIVE_WRAPPER.test(c.name) && c.name.replace(PRIMITIVE_WRAPPER, "") === node.name;
                walk(c, wrapper ? owner : c);
            }
        };
        walk(shipRoot, shipRoot);
        if (!found.length) {
            // eslint-disable-next-line no-console
            console.warn("[liquefactor] ship.glb contained no sampleable meshes");
            return;
        }

        // Frame the room that actually holds a node Aquanova marks liquefiable. The manifest's
        // `entities` map is keyed by glTF NODE name (what the sandbox shows), so the room is
        // resolved from a listed node's position rather than read off a chunk entry.
        const marked = found.find(({ owner }) => liqNames.has(owner.name));
        const mn = marked?.mesh.boundMin;
        const mx = marked?.mesh.boundMax;
        if (mn && mx) {
            const gx = -(mn[0]! + mx[0]!) / 2; // Lite → glTF (negate X)
            const gz = (mn[2]! + mx[2]!) / 2;
            const room = chunks.find((c) => gx >= c.aabb.min[0]! && gx <= c.aabb.max[0]! && gz >= c.aabb.min[2]! && gz <= c.aabb.max[2]!);
            if (room) focus = chunkFocus(room);
        }

        let n = 0;
        const behaviorOfNode = (name: string): ShipBehavior | undefined => resolveBehavior(shipManifest?.behaviors, shipManifest?.entities, name);
        // Placement markers are not scenery: Aquanova hides them and spawns on them, so do the same
        // here rather than letting the player shoot an invisible floor strip.
        const markerNames = new Set(
            [PLAYER_START_BEHAVIOR, WEAPON_START_BEHAVIOR].map((bh) => findEntityWithBehavior(shipManifest?.entities, bh)?.name).filter((v): v is string => !!v)
        );
        for (const { mesh, parent, owner } of found) {
            if (markerNames.has(owner.name)) {
                setMeshVisible(mesh, false);
                (mesh as { pickable?: boolean }).pickable = false;
                continue;
            }
            const liquefyState: LiquefyState = { hit: [0, 0, 0], frontR: 0, edge: LIQUEFY_EDGE, enabled: false };
            // Splice an identity root ABOVE the mesh: the wriggle/restart logic moves that node, which
            // jitters the foe locally without disturbing the rest of the ship hierarchy.
            const root = createTransformNode(`${mesh.name}_${n}_root`, 0, 0, 0, 0, 0, 0, 1, 1, 1);
            const at = parent.children.indexOf(mesh);
            parent.children[at] = root;
            root.parent = parent;
            root.children.push(mesh);
            mesh.parent = root;
            // Ship materials are SHARED across hundreds of kit modules, so the liquefy plugin has to go
            // on a per-foe CLONE — mutating the shared material would clip every door in the ship at once.
            const materials = new Set<Material>();
            if (mesh.material && isPbrMaterial(mesh.material) && !mesh.material.plugins?.some((p) => p.name === "liquefy")) {
                const src = mesh.material as unknown as { plugins?: { name: string }[] };
                mesh.material = {
                    ...(mesh.material as object),
                    _uboVersion: 0,
                    plugins: [...(src.plugins ?? []), createLiquefyPlugin(() => liquefyState, "pbr")],
                } as unknown as Material;
            }
            if (mesh.material) materials.add(mesh.material);
            registerFoeInstance({ key: `${owner.name}#${n}`, x: 0, inPlace: true }, root, [mesh], materials, liquefyState, null);
            // Group the instance with the other primitives of its node so a shot melts the node whole,
            // and index it by node name so the manifest's `linked` lists resolve.
            const inst = instances[instances.length - 1]!;
            const siblings = nodeGroups.get(owner);
            if (siblings) siblings.push(inst);
            else nodeGroups.set(owner, [inst]);
            groupOfInstance.set(inst, nodeGroups.get(owner)!);
            const byName = instancesByNodeName.get(owner.name);
            if (byName) byName.push(inst);
            else instancesByNodeName.set(owner.name, [inst]);
            const bh = behaviorOfNode(owner.name);
            if (bh) behaviorOfInstance.set(inst, bh);
            n++;
        }

        // ── Havok: chunk shells + a rigid body per `dynamic` node ────────────────────────
        // Aquanova collides against clean per-chunk BOXES from the manifest rather than a trimesh of
        // the raw ship (568 overlapping kit modules make a trimesh riddled with coplanar triangles).
        // Same here: each chunk becomes a floor + ceiling + 4 walls, which is what the dynamic props
        // rest on and rattle around inside. There is no character controller in this demo — the free
        // camera flies through everything — so the portal cut-outs Aquanova needs are not required.
        const dynNodes = [...nodeGroups.entries()].filter(([node]) => isDynamicBehavior(behaviorOfNode(node.name)));
        if (chunks.length || dynNodes.length) {
            try {
                const hknp = await HavokPhysics({ locateFile: () => "/HavokPhysics.wasm" });
                physWorld = createHavokWorld(scene, hknp, { x: 0, y: -9.8, z: 0 });
                // Fixed step, as Aquanova does: the default variable step advances physics by the real
                // frame delta, so a prop's resting pose depends on how slow the frames around it were.
                setPhysicsTimestepMs(physWorld, 1000 / 60);
            } catch (err) {
                // A missing/broken WASM must not cost the demo its ship: fall through with everything
                // static instead of aborting the rest of the ship setup (ground, lights, camera).
                // eslint-disable-next-line no-console
                console.warn("[liquefactor] Havok unavailable — dynamic behaviours stay static", err);
            }
        }
        if (physWorld) {
            const world = physWorld;
            let cn = 0;
            // glTF-space centre + full extents; X is negated into Lite scene space. Dynamic props are
            // added ASLEEP, exactly as Aquanova does: a prop must hold the pose the artist authored until
            // something actually disturbs it. Awake, each door immediately sags onto the chunk floor
            // (measured: 8.8 cm) because its collider is a coarse AABB, not the real door geometry.
            const addBox = (gx: number, gy: number, gz: number, ex: number, ey: number, ez: number, motion: PhysicsMotionType): { node: SceneNode; body: PhysicsBody } | null => {
                if (ex <= 1e-3 || ey <= 1e-3 || ez <= 1e-3) return null;
                const node = createTransformNode(`liqCol_${cn++}`, -gx, gy, gz);
                const shape = createPhysicsShape(world, { type: PhysicsShapeType.BOX, parameters: { extents: { x: ex, y: ey, z: ez } } });
                const body = createPhysicsBody(world, node, motion, motion === PhysicsMotionType.DYNAMIC);
                setPhysicsBodyShape(world, body, shape);
                return { node, body };
            };
            const WALL_T = 0.4;
            for (const c of chunks) {
                const [x0, y0, z0] = c.aabb.min as [number, number, number];
                const [x1, y1, z1] = c.aabb.max as [number, number, number];
                const cx = (x0 + x1) / 2;
                const cz = (z0 + z1) / 2;
                const sx = x1 - x0;
                const sz = z1 - z0;
                const h = y1 - y0;
                const midY = (y0 + y1) / 2;
                addBox(cx, y0 - WALL_T / 2, cz, sx, WALL_T, sz, PhysicsMotionType.STATIC); // floor
                addBox(cx, y1 + WALL_T / 2, cz, sx, WALL_T, sz, PhysicsMotionType.STATIC); // ceiling
                addBox(x0 - WALL_T / 2, midY, cz, WALL_T, h, sz, PhysicsMotionType.STATIC);
                addBox(x1 + WALL_T / 2, midY, cz, WALL_T, h, sz, PhysicsMotionType.STATIC);
                addBox(cx, midY, z0 - WALL_T / 2, sx, h, WALL_T, PhysicsMotionType.STATIC);
                addBox(cx, midY, z1 + WALL_T / 2, sx, h, WALL_T, PhysicsMotionType.STATIC);
            }

            // One body per NODE (not per primitive): a node's primitives are one object, so they are
            // reparented under a single display root that the body drives. `setParent` preserves each
            // one's world transform.
            //
            // The props are ANIMATED (kinematic), not DYNAMIC. A door's collider is its coarse world
            // AABB, and that box floats ~8.8 cm above the deck in the authored art, so a free rigid body
            // immediately falls onto the chunk floor: measured asleep at spawn (vel 0) but awake by
            // t = 0.08 s, dropping 1.796 -> 1.708 by t = 0.48 s. `startsAsleep` does not prevent it —
            // something wakes the body within the first frames. Aquanova's equivalent bodies never move,
            // so kinematic reproduces the reference exactly: the prop holds the pose the artist gave it,
            // still collides, and never sags. Switch back to DYNAMIC if props should be knockable.
            for (const [node, group] of dynNodes) {
                let mn: number[] | null = null;
                let mx: number[] | null = null;
                for (const inst of group) {
                    for (const m of inst.meshes) {
                        const a = m.boundMin;
                        const bb = m.boundMax;
                        if (!a || !bb) continue;
                        mn = mn ? [Math.min(mn[0]!, a[0]!), Math.min(mn[1]!, a[1]!), Math.min(mn[2]!, a[2]!)] : [a[0]!, a[1]!, a[2]!];
                        mx = mx ? [Math.max(mx[0]!, bb[0]!), Math.max(mx[1]!, bb[1]!), Math.max(mx[2]!, bb[2]!)] : [bb[0]!, bb[1]!, bb[2]!];
                    }
                }
                if (!mn || !mx) continue;
                const centre: [number, number, number] = [(mn[0]! + mx[0]!) / 2, (mn[1]! + mx[1]!) / 2, (mn[2]! + mx[2]!) / 2];
                const half: [number, number, number] = [Math.max((mx[0]! - mn[0]!) / 2, 0.03), Math.max((mx[1]! - mn[1]!) / 2, 0.03), Math.max((mx[2]! - mn[2]!) / 2, 0.03)];
                const disp = createTransformNode(`dyn_disp_${node.name}`, centre[0], centre[1], centre[2]);
                addToScene(scene, disp);
                for (const inst of group) {
                    setParent(inst.root, disp);
                    // `setParent` rewrites the root's LOCAL transform to preserve its world pose, so the
                    // home captured at registration (identity, pre-reparenting) is now stale. Without this,
                    // Restart collapses every primitive of the node onto the display root's origin.
                    inst.homePos = [inst.root.position.x, inst.root.position.y, inst.root.position.z];
                }
                const box = addBox(-centre[0], centre[1], centre[2], half[0] * 2, half[1] * 2, half[2] * 2, PhysicsMotionType.ANIMATED);
                if (box) dynDisplays.push({ proxy: box.node, body: box.body, disp, rest: centre });
            }
            // Each step, copy the body pose onto the display root so the visible primitives follow.
            onPhysicsAfterStep(world, () => {
                for (const d of dynDisplays) {
                    d.disp.position.set(d.proxy.position.x, d.proxy.position.y, d.proxy.position.z);
                    d.disp.rotationQuaternion.set(d.proxy.rotationQuaternion.x, d.proxy.rotationQuaternion.y, d.proxy.rotationQuaternion.z, d.proxy.rotationQuaternion.w);
                }
            });
            // eslint-disable-next-line no-console
            console.warn(`[liquefactor] havok: ${cn} colliders, ${dynDisplays.length} dynamic node(s)`);
        }

        // The gallery ground would slice through the ship's decks — the ship brings its own floors.
        setMeshVisible(ground, false);
        // Aquanova is lit purely by the HDRI (its glTF carries no punctual lights). The lights are kept
        // OUT of the scene entirely in ship mode (see createHemisphericLight above) rather than merely
        // zeroed: a material compiled with >= 1 light takes a different PBR permutation than one
        // compiled with none, and that permutation renders the ship ~1.2x brighter even when every
        // light's contribution is zero. Measured against Aquanova at a matched camera, muting by
        // intensity left the side panels 1.23x / 1.60x too bright; not registering them lands at 1.00x.
        // Stand where Aquanova spawns the player — the `player_startpos` marker node — facing its
        // `direction` (glTF space → Lite negates X), so both demos open on the same view. Falls back
        // to the marked room's centre. `position` is an ObservableVec3 — assign the COMPONENTS, never
        // the object, or the view matrix stops being flagged dirty.
        const startEntity = findEntityWithBehavior(shipManifest?.entities, PLAYER_START_BEHAVIOR);
        const startMeshes = startEntity ? found.filter(({ owner }) => owner.name === startEntity.name) : [];
        let eye: [number, number, number] = [focus[0] + 3, focus[1], focus[2]];
        if (startMeshes.length) {
            let mn: number[] | null = null;
            let mx: number[] | null = null;
            for (const { mesh } of startMeshes) {
                const a = mesh.boundMin;
                const bb = mesh.boundMax;
                if (!a || !bb) continue;
                mn = mn ? [Math.min(mn[0]!, a[0]!), Math.min(mn[1]!, a[1]!), Math.min(mn[2]!, a[2]!)] : [a[0]!, a[1]!, a[2]!];
                mx = mx ? [Math.max(mx[0]!, bb[0]!), Math.max(mx[1]!, bb[1]!), Math.max(mx[2]!, bb[2]!)] : [bb[0]!, bb[1]!, bb[2]!];
            }
            // Match Aquanova's SETTLED eye height. Aquanova drops a Havok capsule on top of the marker;
            // the marker is a thin non-collidable slab, so gravity pulls the capsule past it down to the
            // chunk floor, which Aquanova pins at FLOOR_Y = 0. The capsule comes to rest with a 0.1333
            // stand-off (the character controller's keepDistance solve at the fixed 1/60 step), so the
            // eye lands at 0 + 0.1333 stand-off + 0.9 capsule half-height + 0.62 eye offset = 1.6533.
            // Both demos fix the physics step, so that stand-off is reproducible rather than depending
            // on how slow the first frames after load happen to be.
            if (mn && mx) eye = [(mn[0]! + mx[0]!) / 2, 1.6533, (mn[2]! + mx[2]!) / 2];
        }
        cam.position.x = eye[0];
        cam.position.y = eye[1];
        cam.position.z = eye[2];
        const dir = startEntity?.ref.direction;
        camYaw = dir && (dir[0] || dir[2]) ? Math.atan2(-dir[0]!, dir[2]!) : -Math.PI / 2;
        camPitch = 0;
        // eslint-disable-next-line no-console
        console.warn(`[liquefactor] ship foes: ${found.length} mesh(es), ibl ${iblStrength}`);
    }

    const picker = createGpuPicker(scene);

    // ── Render pipeline ──────────────────────────────────────────────────────
    const depthRT = createRenderTarget({ lbl: "liq-depth", dFormat: "depth24plus", samples: 1, size: engine });
    const sceneColorRT = createRenderTarget({ lbl: "liq-scene-color", format: engine.format, samples: 1, size: engine });
    // Scene bg excludes any DISSOLVING foe (drawn later as a clipping overlay); solid instances
    // render normally so the water refracts them; fluid/fading/gone instances have hidden meshes.
    const sceneTask = createRenderTask(
        {
            name: "scene",
            rt: sceneColorRT,
            depth: depthRT,
            clr: true,
            clrColor: { r: 0.05, g: 0.06, b: 0.1, a: 1 },
            _filterRenderable: (renderable) => instanceOf(renderable.mesh)?.phase !== "dissolving",
        },
        engine,
        scene
    );
    addTask(scene, sceneTask);

    // Ground-plane scene SDF for every sim: positive above the floor, negative below.
    const groundSdfBuffer = device.createBuffer({ label: "liq-scene-sdf", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(groundSdfBuffer, 0, new Float32Array([GROUND_Y, 0, 0, 0]));
    const groundSdf: SceneSdfSpec = {
        struct: "struct SceneSdfParams { ground: vec4<f32>, };",
        sdf: "fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { return pt.y - sceneSdfParams.ground.x; }",
        buffer: groundSdfBuffer,
    };

    // Combined render buffer aggregating every live sim's particles into ONE surface pass.
    const combinedPos = device.createBuffer({ label: "liq-combined-pos", size: MAX_TOTAL * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const combinedDebug = device.createBuffer({ label: "liq-combined-debug", size: MAX_TOTAL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // Per-particle alpha (opt-in on the surface renderer) — 1 for running blobs, ramped 1→0
    // for a blob that is fading out, so ONLY the fading blob's water fades (the shared render
    // can't fade globally). Seeded to 1 so untouched slots stay opaque.
    const combinedAlpha = device.createBuffer({ label: "liq-combined-alpha", size: MAX_TOTAL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const alphaScratch = new Float32Array(MAX_TOTAL).fill(1);
    device.queue.writeBuffer(combinedAlpha, 0, alphaScratch);
    // Per-particle RGBA colour aggregated across sims (opt-in "Use mesh colours" render toggle).
    const combinedColor = device.createBuffer({ label: "liq-combined-color", size: MAX_TOTAL * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    let useMeshColors = false; // UI toggle: tint the water by the liquefied mesh's texture/vertex colours
    // A virtual sim the surface renderer reads live (count + positionBuffer refreshed each frame).
    // When count is 0 (no active sims) the surface task skips all fluid passes and just presents
    // the scene, so the GPU "Surface" timing is 0 while idle.
    const virtualSim = {
        count: 0,
        particleRadius: 0.08,
        surfaceSizeScale: 1,
        positionBuffer: combinedPos,
        velocityBuffer: combinedPos,
        debugBuffer: combinedDebug,
        debugNorm: 1,
        gpuBytes: 0,
        step: () => {},
        reset: () => {},
        setParam: () => {},
        setSceneSdf: () => {},
        setEmitters: () => {},
        setSpawn: () => {},
        setForceField: () => {},
        dispose: () => {},
    };

    const surfaceTask = createFluidSurfaceTask(engine, scene, { bgRT: sceneColorRT, outRT: engine.scRT, depthRT, camera: cam, sim: virtualSim as unknown as FluidSim });
    surfaceTask.setSim(virtualSim as unknown as FluidSim);
    surfaceTask.setProfiler(profiler);
    surfaceTask.setParticleAlpha(combinedAlpha);
    surfaceTask.setParticleColor(combinedColor);
    addTask(scene, surfaceTask);

    // Per-particle colour: for each DISTINCT sub-mesh texture, a compute pass samples that texture at
    // the particle's UV (from the worker) for the particles that belong to it (texIdx == T), gamma-
    // encoding into the instance colour buffer. Particles with no texture keep a baseColor fill. The
    // buffers are aggregated into combinedColor each frame (same layout as combinedPos).
    const COLOR_SAMPLE_WGSL = /* wgsl */ `
struct P { count: u32, tsel: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> uvs: array<vec2<f32>>;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outCol: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> texIdx: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) { return; }
    if (texIdx[i] != p.tsel) { return; }          // this pass only fills particles using texture tsel
    let dims = vec2<f32>(textureDimensions(tex, 0));
    let w = fract(uvs[i]);                         // repeat-wrap
    let coord = vec2<i32>(clamp(w * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));
    let c = textureLoad(tex, coord, 0);            // sRGB views decode to linear on load
    // Gamma-ENCODE back to display space: the fluid composite outputs gamma-encoded colour, but the
    // Beer-Lambert tint + overlay use this value directly, so a linear sample renders too dark.
    outCol[i] = vec4<f32>(pow(max(c.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}`;
    const colorPipe = device.createComputePipeline({
        label: "liq-color-sample",
        layout: "auto",
        compute: { module: device.createShaderModule({ label: "liq-color-sample", code: COLOR_SAMPLE_WGSL }), entryPoint: "main" },
    });

    function fillInstanceColor(inst: Instance, uvs: Float32Array | null, texIndices: Uint32Array | null, count: number): void {
        inst.colorBuffer?.destroy();
        const buf = device.createBuffer({ label: `liq-color-${inst.key}`, size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        inst.colorBuffer = buf;
        // baseColor fill first: untextured particles (texIdx == NO_TEX) and any not overwritten below.
        const [r, g, b] = inst.baseColor;
        const scratch = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) {
            scratch[i * 4] = r;
            scratch[i * 4 + 1] = g;
            scratch[i * 4 + 2] = b;
            scratch[i * 4 + 3] = 1;
        }
        device.queue.writeBuffer(buf, 0, scratch);
        if (!inst.diffuseTexs.length || !uvs || !texIndices || uvs.length < count * 2 || texIndices.length < count) return;
        const uvBuf = device.createBuffer({ label: `liq-uv-${inst.key}`, size: count * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(uvBuf, 0, uvs, 0, count * 2);
        const tiBuf = device.createBuffer({ label: `liq-ti-${inst.key}`, size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(tiBuf, 0, texIndices, 0, count);
        const enc = device.createCommandEncoder();
        const cbufs: GPUBuffer[] = [];
        for (let t = 0; t < inst.diffuseTexs.length; t++) {
            const cbuf = device.createBuffer({ label: `liq-colcount-${inst.key}-${t}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(cbuf, 0, new Uint32Array([count, t, 0, 0]));
            cbufs.push(cbuf);
            const bg = device.createBindGroup({
                layout: colorPipe.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: cbuf } },
                    { binding: 1, resource: { buffer: uvBuf } },
                    { binding: 2, resource: inst.diffuseTexs[t]!.view },
                    { binding: 3, resource: { buffer: buf } },
                    { binding: 4, resource: { buffer: tiBuf } },
                ],
            });
            const pass = enc.beginComputePass();
            pass.setPipeline(colorPipe);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(count / 64));
            pass.end();
        }
        device.queue.submit([enc.finish()]);
        uvBuf.destroy();
        tiBuf.destroy();
        for (const c of cbufs) c.destroy();
    }

    // Where the scene was last drawn, plus that draw's matrices — the source for per-particle LIT
    // colours (see particle-lit-colors.ts). The scene target still holds the solid foe at the moment
    // it is liquefied, so the water inherits how the foe actually looked rather than a raw (unlit)
    // texture read.
    const litScene = (): LitColorScene | null => {
        // The scene colour/depth textures are allocated lazily on first render. Until then there is
        // no rendered image to promote the albedo against, so skip the lit pass for that shot rather
        // than binding null views (which throws inside createBindGroup).
        if (!sceneColorRT._colorView || !depthRT._depthView) return null;
        const aspect = engine.canvas.width / Math.max(1, engine.canvas.height);
        const proj = getProjectionMatrix(cam, aspect);
        return {
            colorView: sceneColorRT._colorView,
            depthView: depthRT._depthView,
            view: getViewMatrix(cam),
            viewProj: getViewProjectionMatrix(cam, aspect),
            projZW: proj[14]!,
            projZZ: proj[10]!,
        };
    };

    // Foe overlay: after the single water render presents into scRT, draw ALL dissolving foes
    // on top (each clips inside its own front). Depth-aliases the scene-minus-foes depth.
    const foeDepth = createRenderTarget({ lbl: "liq-foe-depth", dFormat: depthRT._descriptor.dFormat, samples: 1, size: engine });
    foeDepth._eager = true;
    foeDepth._ownsDepthTexture = false;
    const foeTask = createRenderTask(
        { name: "liq-foe", rt: engine.scRT, depth: foeDepth, clr: false, _filterRenderable: (renderable) => instanceOf(renderable.mesh)?.phase === "dissolving" },
        engine,
        scene
    );
    const foeRecord = foeTask.record.bind(foeTask);
    foeTask.record = (): void => {
        foeDepth._depthTexture = depthRT._depthTexture;
        foeDepth._depthView = depthRT._depthView;
        foeDepth._width = depthRT._width;
        foeDepth._height = depthRT._height;
        foeRecord();
    };
    addTask(scene, foeTask);

    // Encoder-level GPU-timing resolve: after every other pass, close the whole-frame envelope and
    // resolve the timestamp queries. Draws nothing. Must be the LAST task added.
    if (profiler) {
        const prof = profiler;
        addTask(scene, {
            name: "liq-timing-resolve",
            engine,
            scene,
            _passes: [],
            record: (): void => {},
            execute: (): number => {
                prof.frameStop(engine._currentEncoder);
                prof.resolveInto(engine._currentEncoder);
                return 0;
            },
            dispose: (): void => {},
        });
    }

    const invalidateFilteredSceneTasks = (): void => {
        for (const task of [sceneTask, foeTask]) {
            task._lastVersion = -1;
            task._ob.length = 0;
        }
    };

    surfaceTask.setDirLight(SUN_DIR);
    surfaceTask.setFluidColor(hexToRgb(DEF_COLOR));
    surfaceTask.setAbsorption(DEF_ABSORPTION);
    surfaceTask.setSizeScale(DEF_SIZE);
    surfaceTask.setRefractionStrength(DEF_REFRACTION);
    surfaceTask.setSpecularPower(DEF_SPECULAR);
    surfaceTask.setDepthBlur(DEF_DEPTH_BLUR, DEF_DEPTH_BLUR_THRESHOLD);
    surfaceTask.setThicknessBlur(DEF_THICKNESS_BLUR);
    surfaceTask.setHalfRender(DEF_HALF);
    surfaceTask.setThicknessDownscale(DEF_THICKNESS_DOWNSCALE);
    surfaceTask.setSurfaceFilter(DEF_SURFACE_FILTER);
    surfaceTask.setNarrowRange(DEF_NARROW_DELTA, DEF_NARROW_MU);
    surfaceTask.setMode("surface");

    enableMaterialPlugins(scene);

    const brdfUrl = demoAssetUrl("./brdf-lut.png", import.meta.url);

    // ── Environment picker ────────────────────────────────────────────────────
    // ── Ship manifest (ship mode only) ───────────────────────────────────────
    // Fetched ONCE, before the environment slots are graded: the manifest carries the Blender view
    // transform the ship was authored with (tone mapping + exposure in stops) and the IBL strength,
    // and Aquanova reads exactly the same values — so the two demos render the interior identically.
    interface ShipManifestData {
        environment?: ShipEnvironment;
        behaviors?: ShipBehaviorLibrary; // behaviour name → definition
        entities?: ShipEntities; // mesh name → assigned behaviours
        chunks?: { aabb: { min: number[]; max: number[] } | null }[];
    }
    let shipManifest: ShipManifestData | null = null;
    const shipManifestReady: Promise<void> =
        FOE_SET === "ship"
            ? fetch(SHIP_MANIFEST_URL)
                  .then((r) => r.json() as Promise<ShipManifestData>)
                  .then((j) => {
                      shipManifest = j;
                  })
                  .catch((err: unknown) => {
                      // eslint-disable-next-line no-console
                      console.warn("[liquefactor] ship manifest load failed", err);
                  })
            : Promise.resolve();

    // Audition backdrops for the fluid-surface reflections + skybox. Switching environments updates
    // diffuse IBL + fluid reflections live; foe specular reflections stay on the startup cube.
    interface EnvSlot {
        env: EnvironmentTextures;
        /** Null in ship mode, which uses the solid fallback skybox `loadHdrEnvironment` builds
         *  itself (as Aquanova does) rather than this demo's own HDR cubemap sky. */
        sky: Renderable | null;
        exposure: number;
        contrast: number;
    }
    const ENV_CHOICES = [{ key: "studio", label: "Studio (default)", url: ENV_STUDIO_URL, hdr: false, exposure: 1.0, contrast: 1.1 }] as const;
    const envSlots = new Map<string, EnvSlot | null>();
    let activeSky: Renderable | null = null;
    let currentEnvKey = "studio";
    // Ship mode retains its manifest grading instead of the environment-gallery grading.
    const SHIP_CONTRAST = 1.05;
    const gradedExposure = (v: number): number => (FOE_SET === "ship" ? resolveExposure(shipManifest?.environment?.exposure) : v);
    const gradedContrast = (v: number): number => (FOE_SET === "ship" ? SHIP_CONTRAST : v);
    const makeEnvSlot = (env: EnvironmentTextures, exposure: number, contrast: number): EnvSlot => {
        // The skybox snapshots scene.imageProcessing at build time, so grade FIRST.
        scene.imageProcessing.exposure = gradedExposure(exposure);
        scene.imageProcessing.contrast = gradedContrast(contrast);
        // Ship mode has no sky of its own — loadHdrEnvironment already pushed the solid fallback
        // skybox, matching Aquanova.
        return { env, sky: SHIP_MODE ? null : buildHdrSkyboxRenderable(scene, env, 10, [0, 0, 0], [0, 0, 0]), exposure, contrast };
    };
    const loadEnvSlot = async (c: (typeof ENV_CHOICES)[number]): Promise<EnvSlot | null> => {
        if (envSlots.has(c.key)) return envSlots.get(c.key) ?? null;
        try {
            const env = c.hdr
                ? // Both demos build the IBL at 512-pixel cube faces (sharper specular reflections on
                  // the ship's metal than the 256 default). Ship mode additionally mirrors Aquanova's
                  // background: the solid fallback skybox `loadHdrEnvironment` builds, rather than this
                  // demo's own HDR cubemap sky, which only the gallery's environment dropdown needs.
                  await loadHdrEnvironment(scene, c.url, SHIP_MODE ? { faceSize: 512, skipGround: true } : { faceSize: 512, skipGround: true, skipSkybox: true })
                : await loadEnvironment(scene, c.url, { brdfUrl, skipGround: true, skipSkybox: true });
            scene.imageProcessing.toneMappingEnabled = true;
            // Ship mode uses the view transform named in the manifest (Aquanova reads the same
            // field); the gallery keeps the default transform.
            if (FOE_SET === "ship") {
                const tone = resolveToneMapping(shipManifest?.environment?.toneMapping);
                scene.imageProcessing.toneMappingEnabled = tone !== null;
                if (tone) scene.imageProcessing.toneMapping = tone;
            }
            const slot = makeEnvSlot(env, c.exposure, c.contrast);
            envSlots.set(c.key, slot);
            return slot;
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn(`[liquefactor] env "${c.key}" failed to load (${c.url})`, err);
            envSlots.set(c.key, null);
            return null;
        }
    };
    const installEnvSlot = (slot: EnvSlot): void => {
        scene.imageProcessing.exposure = gradedExposure(slot.exposure);
        scene.imageProcessing.contrast = gradedContrast(slot.contrast);
        scene._envTextures = slot.env; // diffuse IBL (repacked into the scene UBO each frame)
        if (activeSky !== slot.sky) {
            if (activeSky) {
                const i = scene._renderables.indexOf(activeSky);
                if (i >= 0) scene._renderables.splice(i, 1);
            }
            if (slot.sky) {
                scene._renderables.push(slot.sky);
            }
            scene._renderableVersion++;
            activeSky = slot.sky;
        }
        surfaceTask.setEnvMap({ view: slot.env._specularCubeView, sampler: slot.env._cubeSampler });
    };
    const applyEnv = async (key: string): Promise<void> => {
        await shipManifestReady; // ship grading comes from the manifest — read it before building a slot
        currentEnvKey = key;
        const c = ENV_CHOICES.find((x) => x.key === key);
        if (!c) return;
        const slot = await loadEnvSlot(c);
        if (slot && currentEnvKey === key) installEnvSlot(slot);
    };
    const envReady = applyEnv(currentEnvKey);

    // ── Global tuning state (applied to newly built sims) ────────────────────
    let radiusValue = 0.08;
    let modeValue: VolumeSamplingMode = "dense";
    // Volume lattice vs surface shell. `auto` is the shipped behaviour (openness + thickness gates,
    // plus the per-prop hollow hint in MODEL_FOES); the other two force the choice so the difference
    // can be auditioned on the same prop.
    let fillStrategy: MeshFillStrategy = "auto";
    // Point spacing shared by BOTH fill paths, in units of the particle radius. Fixed at one particle
    // diameter (neighbours just touch) — the convention the volume lattice uses and the only value the
    // two demos agree on, so it is no longer exposed as a control.
    const FILL_SPACING = 2;
    let impulseIntensity = 1.0;
    // Direction of the uniform part of the eruption impulse. Normalised at use, so these are raw
    // editable components rather than a unit vector. Serialised with the intensity into the setting
    // file so Aquanova bursts the same way.
    const impulseDir: [number, number, number] = [...IMPULSE_DEFAULT_DIR] as [number, number, number];
    /** Blast-sphere radius in world units around the impact point. 0 = derive it per mesh from the
     *  sampled volume (the old behaviour), which makes the falloff scale with the prop instead of
     *  being a fixed distance. */
    let impulseRadius = 0;
    /** Simulation domain size in world units (FULL extent, not a half-width), per axis. 0 = derive it
     *  from the prop's footprint via {@link SPREAD_MARGIN}. Exported so Aquanova uses the same box —
     *  the domain wall is a hard boundary, so a prop tuned here only behaves the same there if the
     *  grid matches. */
    const gridSize: [number, number, number] = [0, 0, 0];
    let currentMethod = "MLS-MPM";
    let currentMaterial = 0;

    const LIQ_SCHEMAS: Record<string, PhysSchemaEntry[]> = Object.fromEntries(
        Object.entries(DEFAULT_FLUID_SCHEMAS).map(([m, entries]) => [m, entries.map((e) => (m === "PBF" && e.key === "viscosity" ? { ...e, value: 0.35 } : { ...e }))])
    );
    const SCHEMA_DEFAULTS: Record<string, Record<string, number>> = {};
    const physValues: Record<string, Record<string, number>> = {};
    for (const m of Object.keys(LIQ_SCHEMAS)) {
        SCHEMA_DEFAULTS[m] = {};
        physValues[m] = {};
        for (const p of LIQ_SCHEMAS[m]!) {
            SCHEMA_DEFAULTS[m]![p.key] = p.value;
            physValues[m]![p.key] = p.value;
        }
    }

    // ── Per-instance sim construction (grid centred on the sampled foe, at ground) ──
    function buildInstanceSim(
        inst: Instance,
        positions: Float32Array,
        count: number,
        radius: number,
        wMin: readonly [number, number, number],
        wMax: readonly [number, number, number]
    ): void {
        const dx = Math.max(radius * 2.4, 0.18);
        const phys = physValues[currentMethod]!;
        const cx = (wMin[0] + wMax[0]) / 2;
        const cz = (wMin[2] + wMax[2]) / 2;
        const autoHalf = Math.max(wMax[0] - wMin[0], wMax[2] - wMin[2]) / 2 + SPREAD_MARGIN;
        const halfX = gridSize[0] > 0 ? gridSize[0] / 2 : autoHalf;
        const halfZ = gridSize[2] > 0 ? gridSize[2] / 2 : autoHalf;
        const floorY = gridFloorY(GROUND_Y, dx);
        const topY = gridTopY(floorY, gridSize[1], wMax[1], dx, Math.max(wMax[1] + 3, 8));
        const boundsMin: [number, number, number] = [cx - halfX, floorY, cz - halfZ];
        const boundsMax: [number, number, number] = [cx + halfX, topY, cz + halfZ];
        // eslint-disable-next-line no-console
        console.log(
            `[liquefactor] sim grid: ${(halfX * 2).toFixed(2)}×${(topY - floorY).toFixed(2)}×${(halfZ * 2).toFixed(2)} m ` +
                `(${Math.ceil((halfX * 2) / dx)}×${Math.ceil((topY - floorY) / dx)}×${Math.ceil((halfZ * 2) / dx)} cells @ dx ${dx.toFixed(3)}), particles: ${count}`
        );
        let sim: FluidSim;
        if (currentMethod === "PBF") {
            const pbfH = Math.max(radius * 4.0, 0.3);
            sim = createPbfSim(engine, {
                count,
                particleRadius: radius,
                initialPositions: positions,
                smoothingRadius: pbfH,
                boundsMin,
                boundsMax,
                groundY: GROUND_Y,
                maxPerCell: 48,
                gravity: phys.gravity,
                viscosity: phys.viscosity,
                relaxation: phys.relaxation,
                scorr: phys.scorr,
                iterations: phys.iterations,
                restDensity: phys.restDensity,
                boundaryDensity: phys.boundaryDensity,
            });
        } else if (currentMethod === "PB-MPM") {
            sim = createPbMpmSim(engine, {
                count,
                particleRadius: radius,
                initialPositions: positions,
                boundsMin,
                boundsMax,
                dx,
                groundY: GROUND_Y,
                material: currentMaterial,
                gravity: phys.gravity,
                substeps: phys.substeps,
                ...(phys.maxSubDtMs ? { maxSubDt: phys.maxSubDtMs / 1000 } : {}),
                iterations: phys.iterations,
                liquidRelaxation: phys.liquidRelaxation,
                liquidViscosity: phys.liquidViscosity,
                elasticityRatio: phys.elasticityRatio,
                elasticRelaxation: phys.elasticRelaxation,
                frictionAngle: phys.frictionAngle,
                plasticity: phys.plasticity,
                restitution: phys.restitution,
            });
        } else {
            sim = createMlsMpmSim(engine, {
                count,
                particleRadius: radius,
                initialPositions: positions,
                boundsMin,
                boundsMax,
                dx,
                groundY: GROUND_Y,
                gravity: phys.gravity,
                stiffness: phys.stiffness,
                viscosity: phys.viscosity,
                restDensity: phys.restDensity,
                substeps: phys.substeps,
                ...(phys.maxSubDtMs ? { maxSubDt: phys.maxSubDtMs / 1000 } : {}),
                damping: phys.damping,
                affineDamping: phys.affineDamping,
                groundDamp: phys.groundDamp,
                groundDampHeight: phys.groundDampHeight,
                restitution: phys.restitution,
            });
        }
        sim.setSceneSdf(groundSdf);
        sim.setProfiler?.(profiler);
        inst.sim = sim;
        inst.count = count;
        inst.radius = radius;
        virtualSim.particleRadius = radius;
        virtualSim.surfaceSizeScale = sim.surfaceSizeScale ?? 1;
    }

    type SampleGeom = { positions: Float32Array; indices: Uint32Array; uvs: Float32Array | null; texIndices: Uint32Array | null; ox: number; oy: number; oz: number };
    type SampledFill = {
        positions: Float32Array;
        count: number;
        radius: number;
        boundsMin: [number, number, number];
        boundsMax: [number, number, number];
        uvs?: Float32Array | null;
        texIndices?: Uint32Array | null;
        shell?: boolean;
    };
    const LIQUEFY_EDGE = 0.6;
    const NO_TEX = 0xffffffff; // per-vertex/particle texIndex sentinel: no texture → baseColor fill

    // Geometry to feed the volume sampler: LOCAL + a world offset (procedural), or the CURRENT
    // deformed pose already in WORLD space with a zero offset (skinned model). Also emits per-vertex
    // UVs + a per-vertex texture index (which sub-mesh texture to sample) when the instance is textured.
    function sampleGeometry(inst: Instance): SampleGeom | null {
        const wantUv = inst.diffuseTexs.length > 0;
        if (!inst.deformable) {
            const m = inst.meshes[0]!;
            if (!m._cpuPositions || !m._cpuIndices) return null;
            const uvs = wantUv && m._cpuUvs ? m._cpuUvs.slice() : null;
            return { positions: m._cpuPositions.slice(), indices: (m._cpuIndices as Uint32Array).slice(), uvs, texIndices: null, ox: inst.x, oy: inst.baseY, oz: 0 };
        }
        let totalV = 0;
        let totalI = 0;
        let allHaveUv = wantUv;
        const parts: { pos: Float32Array; idx: Uint32Array; uv: Float32Array | null; ti: number; w: ArrayLike<number> }[] = [];
        for (let mi = 0; mi < inst.meshes.length; mi++) {
            const m = inst.meshes[mi]!;
            if (!m._cpuPositions || !m._cpuIndices) continue;
            const local = computeDeformedPositions(m) ?? m._cpuPositions;
            const uv = m._cpuUvs ?? null;
            if (!uv) allHaveUv = false;
            parts.push({ pos: local, idx: m._cpuIndices as Uint32Array, uv, ti: inst.meshTexIndex[mi] ?? -1, w: m.worldMatrix as unknown as ArrayLike<number> });
            totalV += local.length / 3;
            totalI += m._cpuIndices.length;
        }
        if (totalV === 0) return null;
        const positions = new Float32Array(totalV * 3);
        const indices = new Uint32Array(totalI);
        const uvs = allHaveUv ? new Float32Array(totalV * 2) : null;
        const texIndices = wantUv ? new Uint32Array(totalV) : null;
        let vb = 0;
        let ic = 0;
        for (const p of parts) {
            const w = p.w;
            const n = p.pos.length / 3;
            const ti = p.ti >= 0 ? p.ti >>> 0 : NO_TEX;
            for (let v = 0; v < n; v++) {
                const s = v * 3;
                const lx = p.pos[s]!,
                    ly = p.pos[s + 1]!,
                    lz = p.pos[s + 2]!;
                const o = (vb + v) * 3;
                positions[o] = w[0]! * lx + w[4]! * ly + w[8]! * lz + w[12]!;
                positions[o + 1] = w[1]! * lx + w[5]! * ly + w[9]! * lz + w[13]!;
                positions[o + 2] = w[2]! * lx + w[6]! * ly + w[10]! * lz + w[14]!;
                if (uvs && p.uv) {
                    uvs[(vb + v) * 2] = p.uv[v * 2]!;
                    uvs[(vb + v) * 2 + 1] = p.uv[v * 2 + 1]!;
                }
                if (texIndices) texIndices[vb + v] = ti;
            }
            for (let k = 0; k < p.idx.length; k++) indices[ic + k] = vb + p.idx[k]!;
            vb += n;
            ic += p.idx.length;
        }
        return { positions, indices, uvs, texIndices, ox: 0, oy: 0, oz: 0 };
    }

    // Bake a sampler result's world offset into the points → world-space seed + world AABB.
    function bakeFill(ox: number, oy: number, oz: number, result: MeshParticleFill): SampledFill {
        const p = result.positions;
        for (let i = 0; i < p.length; i += 3) {
            p[i] = p[i]! + ox;
            p[i + 1] = p[i + 1]! + oy;
            p[i + 2] = p[i + 2]! + oz;
        }
        return {
            positions: p,
            count: result.count,
            radius: result.radius,
            boundsMin: [result.bounds.min[0] + ox, result.bounds.min[1] + oy, result.bounds.min[2] + oz],
            boundsMax: [result.bounds.max[0] + ox, result.bounds.max[1] + oy, result.bounds.max[2] + oz],
            shell: result.shell,
        };
    }

    function computeMaxR(hit: readonly [number, number, number], bMin: readonly [number, number, number], bMax: readonly [number, number, number]): number {
        let maxD = 0;
        for (const cx of [bMin[0], bMax[0]])
            for (const cy of [bMin[1], bMax[1]]) for (const cz of [bMin[2], bMax[2]]) maxD = Math.max(maxD, Math.hypot(cx - hit[0], cy - hit[1], cz - hit[2]));
        return maxD + LIQUEFY_EDGE + 0.5;
    }

    // ── Wriggle (cartoon pain shake) ─────────────────────────────────────────
    // The MESH jitter starts the instant the foe is clicked (before the async volume-sample
    // finishes) for immediate feedback; the WATER jitter is added once the sim exists.
    function startWriggle(inst: Instance): void {
        inst.wriggleBase[0] = inst.root.position.x;
        inst.wriggleBase[1] = inst.root.position.y;
        inst.wriggleBase[2] = inst.root.position.z;
        inst.wriggleWaterBase = null;
        inst.wriggleWaterScratch = null;
        inst.wriggling = true;
    }

    function setWaterWriggle(inst: Instance, sampled: Float32Array, count: number): void {
        const base = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) {
            base[i * 4] = sampled[i * 3]!;
            base[i * 4 + 1] = sampled[i * 3 + 1]!;
            base[i * 4 + 2] = sampled[i * 3 + 2]!;
            base[i * 4 + 3] = 1;
        }
        inst.wriggleWaterBase = base;
        inst.wriggleWaterScratch = new Float32Array(count * 4);
    }

    function stopWriggle(inst: Instance): void {
        if (inst.wriggling) inst.root.position.set(inst.wriggleBase[0], inst.wriggleBase[1], inst.wriggleBase[2]);
        inst.wriggling = false;
        inst.wriggleWaterBase = null;
        inst.wriggleWaterScratch = null;
    }

    function applyWriggle(inst: Instance): void {
        const ox = (Math.random() * 2 - 1) * WRIGGLE_AMP;
        const oy = (Math.random() * 2 - 1) * WRIGGLE_AMP;
        const oz = (Math.random() * 2 - 1) * WRIGGLE_AMP;
        inst.root.position.set(inst.wriggleBase[0] + ox, inst.wriggleBase[1] + oy, inst.wriggleBase[2] + oz);
        const base = inst.wriggleWaterBase;
        const scratch = inst.wriggleWaterScratch;
        if (base && scratch && inst.sim) {
            for (let i = 0; i < base.length; i += 4) {
                scratch[i] = base[i]! + ox;
                scratch[i + 1] = base[i + 1]! + oy;
                scratch[i + 2] = base[i + 2]! + oz;
                scratch[i + 3] = 1;
            }
            device.queue.writeBuffer(inst.sim.positionBuffer, 0, scratch);
        }
    }

    // ── Impulse / fade force fields ──────────────────────────────────────────
    const impulseData = new Float32Array(8);
    function startImpulse(inst: Instance): void {
        if (!inst.sim) return;
        // A zero intensity zeroes both the radial and the directional term, so the force pass would
        // evaluate to exactly 0 for every particle. Skip it entirely: the engine builds the
        // external-force compute pass lazily on the first non-null injection, so installing one here
        // would compile a shader and add a dispatch per substep for 0.35 s to achieve nothing.
        if (impulseIntensity <= 0) return;
        // Explode from the LIQUEFACTION POINT — where the shot landed — not the volume centre, so the
        // water is thrown away from the impact. A configured radius wins; at 0 it falls back to
        // `maxR`, the distance from that point to the farthest corner of the sampled volume, which
        // reaches every particle even though the centre sits off to one side.
        impulseData[0] = inst.liquefyState.hit[0];
        impulseData[1] = inst.liquefyState.hit[1];
        impulseData[2] = inst.liquefyState.hit[2];
        impulseData[3] = impulseRadius > 0 ? impulseRadius : Math.max(inst.maxR, inst.radius * 8, 1);
        // Uniform push along the configured direction, normalised here so the vector's length is
        // irrelevant and only `impulseIntensity` scales it; the radial burst rides in .w.
        // (0,0,0) means "the way I shot it": the camera→click ray captured when the melt started.
        const len = Math.hypot(impulseDir[0], impulseDir[1], impulseDir[2]);
        const u = len > 1e-6 ? ([impulseDir[0] / len, impulseDir[1] / len, impulseDir[2] / len] as const) : inst.shotDir;
        const dirMag = IMPULSE_DIR_BASE * impulseIntensity;
        impulseData[4] = u[0] * dirMag;
        impulseData[5] = u[1] * dirMag;
        impulseData[6] = u[2] * dirMag;
        impulseData[7] = IMPULSE_RADIAL_BASE * impulseIntensity;
        device.queue.writeBuffer(inst.impulseBuffer, 0, impulseData);
        inst.sim.setForceField(inst.impulseSpec);
        inst.impulseRemaining = 0.35;
    }

    function beginFade(inst: Instance): void {
        // Fade the blob out in place by ramping its per-particle alpha 1→0 (handled in the
        // per-frame loop); the sim keeps settling under gravity meanwhile. No sink force.
        inst.phase = "fading";
        inst.fadeElapsed = 0;
        inst.impulseRemaining = 0;
        inst.sim?.setForceField(null);
    }

    const bumpUbo = (inst: Instance): void => {
        for (const m of inst.materials) m._uboVersion++;
    };
    const setVisible = (inst: Instance, v: boolean): void => {
        for (const m of inst.meshes) setMeshVisible(m, v);
    };

    function disposeInstanceSim(inst: Instance): void {
        inst.sim?.setForceField(null);
        inst.sim?.dispose();
        inst.sim = null;
        inst.colorBuffer?.destroy();
        inst.colorBuffer = null;
        inst.phase = "gone";
        inst.count = 0;
    }

    // ── Shot lifecycle ───────────────────────────────────────────────────────
    // Volume-sampling runs in a worker (see particle-fill-worker.ts) so the ~1 s dense sample
    // doesn't freeze the frame at the shot. requestSample() fires the job (foe stays solid,
    // marked `sampling`); applySample() builds the sim + starts the dissolve when it returns.
    let liveShots = 0;
    let sampleSeq = 0;
    const pendingSamples = new Map<number, { inst: Instance; hit: [number, number, number] }>();

    // A small pool of sampling workers so several foes can convert to particles IN PARALLEL: a
    // second shot while one conversion is in flight goes to a free worker instead of queueing behind
    // it. `pendingSamples` (keyed by id) routes each reply to the right foe regardless of worker.
    type WorkerMsg = {
        id: number;
        positions: Float32Array;
        uvs: Float32Array | null;
        texIndices: Uint32Array | null;
        count: number;
        radius: number;
        shell: boolean;
        boundsMin: [number, number, number];
        boundsMax: [number, number, number];
    };
    interface PoolWorker {
        worker: Worker;
        pending: number;
    }
    const workerPool: PoolWorker[] = [];
    const onSampleMessage = (ev: MessageEvent<WorkerMsg>): void => {
        const { id, positions, uvs, texIndices, count, radius, shell, boundsMin, boundsMax } = ev.data;
        const entry = pendingSamples.get(id);
        pendingSamples.delete(id);
        if (!entry) return;
        if (count === 0) {
            entry.inst.sampling = false;
            leaveShot(entry.inst);
            stopWriggle(entry.inst);
            setStatus();
            return;
        }
        applySample(entry.inst, id, entry.hit, { positions, count, radius, shell, boundsMin, boundsMax, uvs, texIndices });
    };
    try {
        if (typeof Worker !== "undefined") {
            const poolSize = Math.min(3, Math.max(1, (navigator.hardwareConcurrency || 4) - 1));
            for (let i = 0; i < poolSize; i++) {
                const worker = new Worker(new URL("./particle-fill-worker.ts", import.meta.url), { type: "module" });
                const pw: PoolWorker = { worker, pending: 0 };
                worker.addEventListener("message", (ev: MessageEvent<WorkerMsg>) => {
                    pw.pending = Math.max(0, pw.pending - 1);
                    onSampleMessage(ev);
                });
                worker.addEventListener("error", (e) => {
                    // eslint-disable-next-line no-console
                    console.warn("[liquefactor] sample worker error", e.message);
                    const idx = workerPool.indexOf(pw);
                    if (idx >= 0) workerPool.splice(idx, 1); // drop the faulty worker; sync fallback once the pool empties
                });
                workerPool.push(pw);
            }
        }
    } catch {
        workerPool.length = 0;
    }
    // Least-loaded worker (fewest in-flight jobs) so concurrent conversions spread across the pool.
    const pickWorker = (): PoolWorker | null => {
        let best: PoolWorker | null = null;
        for (const pw of workerPool) if (!best || pw.pending < best.pending) best = pw;
        return best;
    };

    function requestSample(inst: Instance, hit: readonly [number, number, number]): boolean {
        if (inst.phase !== "solid" || inst.sampling) return false;
        const id = ++sampleSeq;
        inst.sampling = true;
        inst.sampleId = id;
        const h: [number, number, number] = [hit[0], hit[1], hit[2]];
        pendingSamples.set(id, { inst, hit: h });
        startWriggle(inst); // pain shake begins immediately on click, before the async sample finishes
        setStatus();
        const geom = sampleGeometry(inst);
        if (!geom) {
            pendingSamples.delete(id);
            inst.sampling = false;
            leaveShot(inst);
            stopWriggle(inst);
            setStatus();
            return false;
        }
        // Freeze a walking foe so the dissolve shows the exact pose we just sampled.
        if (inst.animation) pauseAnimation(inst.animation);
        const pw = pickWorker();
        if (pw) {
            const transfer: Transferable[] = [geom.positions.buffer, geom.indices.buffer];
            if (geom.uvs) transfer.push(geom.uvs.buffer);
            if (geom.texIndices) transfer.push(geom.texIndices.buffer);
            pw.pending++;
            pw.worker.postMessage(
                {
                    id,
                    positions: geom.positions,
                    indices: geom.indices,
                    uvs: geom.uvs,
                    texIndices: geom.texIndices,
                    radius: radiusValue,
                    mode: modeValue,
                    surfaceOnly: inst.surfaceOnly,
                    strategy: fillStrategy,
                    spacing: FILL_SPACING,
                    ox: geom.ox,
                    oy: geom.oy,
                    oz: geom.oz,
                },
                transfer
            );
        } else {
            // No worker available — sample synchronously on the main thread (blocks). No per-particle
            // UVs are computed here, so colours fall back to the instance baseColor.
            pendingSamples.delete(id);
            const result = fillMeshParticles({
                positions: geom.positions,
                indices: geom.indices,
                radius: radiusValue,
                mode: modeValue,
                surfaceOnly: inst.surfaceOnly,
                strategy: fillStrategy,
                spacing: FILL_SPACING,
            });
            if (result.count > 0) {
                applySample(inst, id, h, { ...bakeFill(geom.ox, geom.oy, geom.oz, result), uvs: null, texIndices: null });
            } else {
                inst.sampling = false;
                leaveShot(inst);
                stopWriggle(inst);
                setStatus();
            }
        }
        return true;
    }

    function applySample(inst: Instance, id: number, hit: readonly [number, number, number], fill: SampledFill): void {
        // Ignore stale results (a Restart or re-shape happened while sampling).
        if (inst.sampleId !== id || !inst.sampling || inst.phase !== "solid") {
            inst.sampling = false;
            leaveShot(inst);
            return;
        }
        inst.sampling = false;
        inst.shell = fill.shell ?? false;
        buildInstanceSim(inst, fill.positions, fill.count, fill.radius, fill.boundsMin, fill.boundsMax);
        fillInstanceColor(inst, fill.uvs ?? null, fill.texIndices ?? null, fill.count); // per-particle mesh colours (per-mesh texture sample or baseColor)
        // Promote the albedo to the foe's CURRENTLY RENDERED colours (lit + tone-mapped).
        if (inst.colorBuffer) {
            const ls = litScene();
            if (ls) {
                const lit = buildLitParticleColors(device, fill.count, fill.positions, inst.colorBuffer, ls);
                inst.colorBuffer.destroy();
                inst.colorBuffer = lit;
            }
        }
        inst.liquefyState.hit = [hit[0], hit[1], hit[2]];
        inst.liquefyState.frontR = 0;
        inst.liquefyState.enabled = true;
        // Distance from the hit point to the farthest corner of the sampled volume. Drives both the
        // dissolve front and (at finishShot) the explosion reach, so the blast centred on that same
        // point still covers every particle. The sim does NOT step during "dissolving", so this still
        // matches the particles at explosion time.
        inst.maxR = computeMaxR(hit, fill.boundsMin, fill.boundsMax);
        inst.phase = "dissolving";
        setWaterWriggle(inst, fill.positions, fill.count); // add the water jitter; the mesh shake is already running
        bumpUbo(inst);
        liveShots++;
        invalidateFilteredSceneTasks();
        setStatus();
    }

    function finishShot(inst: Instance): void {
        stopWriggle(inst);
        setVisible(inst, false);
        inst.liquefyState.enabled = false;
        inst.liquefyState.frontR = inst.maxR;
        bumpUbo(inst);
        inst.phase = "fluid";
        inst.fluidElapsed = 0;
        inst.shot = null; // the shot has erupted; membership no longer gates anything
        startImpulse(inst);
        invalidateFilteredSceneTasks();
        setStatus();
    }

    function restart(): void {
        pendingSamples.clear();
        for (const inst of instances) {
            inst.sim?.setForceField(null);
            inst.sim?.dispose();
            inst.sim = null;
            inst.colorBuffer?.destroy();
            inst.colorBuffer = null;
            inst.phase = "solid";
            inst.sampling = false;
            inst.shot = null;
            inst.dissolved = false;
            inst.count = 0;
            inst.shell = false;
            inst.liquefyState.enabled = false;
            inst.liquefyState.frontR = 0;
            inst.impulseRemaining = 0;
            inst.fluidElapsed = 0;
            inst.fadeElapsed = 0;
            inst.wriggling = false;
            inst.wriggleWaterBase = null;
            inst.wriggleWaterScratch = null;
            inst.root.position.set(inst.homePos[0], inst.homePos[1], inst.homePos[2]);
            setVisible(inst, true);
            bumpUbo(inst);
            if (inst.animation) playAnimation(inst.animation); // resume the walk on the restored foe
        }
        // Restoring the meshes is not enough for a `dynamic` prop: its Havok body keeps whatever pose it
        // toppled into and would re-impose it on the next step. Teleport each body home, kill its
        // velocities, and sync the display root so there's no one-frame flash before the next step.
        // Props that never moved are skipped — they are asleep, and teleporting them would wake them for
        // nothing (an awake door immediately sags onto the chunk floor).
        if (physWorld) {
            for (const d of dynDisplays) {
                const q = d.disp.rotationQuaternion;
                const off = Math.max(Math.abs(d.disp.position.x - d.rest[0]), Math.abs(d.disp.position.y - d.rest[1]), Math.abs(d.disp.position.z - d.rest[2]));
                if (off < 1e-4 && Math.abs(q.x) < 1e-4 && Math.abs(q.y) < 1e-4 && Math.abs(q.z) < 1e-4) continue;
                setPhysicsBodyTransform(physWorld, d.body, { x: d.rest[0], y: d.rest[1], z: d.rest[2] }, { x: 0, y: 0, z: 0, w: 1 });
                setPhysicsBodyLinearVelocity(physWorld, d.body, { x: 0, y: 0, z: 0 });
                setPhysicsBodyAngularVelocity(physWorld, d.body, { x: 0, y: 0, z: 0 });
                d.disp.position.set(d.rest[0], d.rest[1], d.rest[2]);
                d.disp.rotationQuaternion.set(0, 0, 0, 1);
            }
        }
        liveShots = 0;
        virtualSim.count = 0;
        invalidateFilteredSceneTasks();
        setStatus();
    }

    // ── Control panel ────────────────────────────────────────────────────────
    const PANEL_STYLE =
        "position:fixed;top:12px;left:12px;z-index:10;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;" +
        "font-size:0.8rem;color:#dfe6ee;background:rgba(12,16,24,0.82);padding:12px 14px;border-radius:10px;" +
        "width:232px;max-height:92vh;overflow-y:auto;box-shadow:0 6px 22px rgba(0,0,0,0.4);";

    const title = document.createElement("div");
    title.textContent = "Liquefactor";
    title.style.cssText = "font-weight:700;font-size:0.95rem;margin-bottom:2px;";
    const subtitle = document.createElement("div");
    subtitle.textContent =
        FOE_SET === "ship"
            ? "WASD/Space/C to fly · RMB-drag to look · click a node → its own fluid sim"
            : "WASD/Space/C to fly · RMB-drag to look · click a foe → its own fluid sim";
    subtitle.style.cssText = "color:#8fa4bc;margin-bottom:10px;";

    function labelledRow(text: string, control: HTMLElement): HTMLDivElement {
        const row = document.createElement("div");
        row.style.cssText = "margin-bottom:8px;";
        const lab = document.createElement("div");
        lab.textContent = text;
        lab.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
        row.append(lab, control);
        return row;
    }
    function styleSelect(sel: HTMLSelectElement): void {
        sel.style.cssText = "width:100%;padding:3px;background:#1a2230;color:#dfe6ee;border:1px solid #33415a;border-radius:4px;";
    }

    // Sampling mode selector.
    const modeSelect = document.createElement("select");
    styleSelect(modeSelect);
    for (const m of ["dense", "regular", "kugelstadt2021"] as VolumeSamplingMode[]) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m === "kugelstadt2021" ? "kugelstadt2021 (SPH, slow)" : m;
        modeSelect.append(opt);
    }
    modeSelect.value = modeValue;
    modeSelect.onchange = () => {
        modeValue = modeSelect.value as VolumeSamplingMode;
    };

    // Environment picker — swaps the skybox + fluid-surface reflection cube (audition backdrops;
    // "Bank vault" is the Aquanova HDRI, for comparing the water look under identical lighting).
    const envSelect = document.createElement("select");
    styleSelect(envSelect);
    for (const c of ENV_CHOICES) {
        const opt = document.createElement("option");
        opt.value = c.key;
        opt.textContent = c.label;
        envSelect.append(opt);
    }
    envSelect.value = currentEnvKey;
    envSelect.onchange = () => {
        void applyEnv(envSelect.value);
    };
    const envRow = labelledRow("Environment", envSelect);

    // Foe-set picker — see FOE_SET: swaps between the sample models and the Aquanova ship's own
    // liquefiable meshes. Reloads the page because the liquefy plugin only materialises for meshes
    // present in the initial scene build.
    const foeSelect = document.createElement("select");
    styleSelect(foeSelect);
    for (const f of FOE_SETS) {
        const opt = document.createElement("option");
        opt.value = f.key;
        opt.textContent = f.label;
        foeSelect.append(opt);
    }
    foeSelect.value = FOE_SET;
    foeSelect.onchange = () => {
        const url = new URL(location.href);
        if (foeSelect.value === "ship") {
            url.searchParams.set("foes", "ship");
        } else {
            url.searchParams.delete("foes");
        }
        location.href = url.toString();
    };
    const foeRow = labelledRow("Foes", foeSelect);

    // PB-MPM material selector (only meaningful for PB-MPM; applied to the next shot).
    const materialSelect = document.createElement("select");
    styleSelect(materialSelect);
    for (const [label, value] of PBMPM_MATERIAL_LABELS) {
        const opt = document.createElement("option");
        opt.value = String(value);
        opt.textContent = label;
        materialSelect.append(opt);
    }
    materialSelect.value = String(currentMaterial);
    const materialRow = labelledRow("PB-MPM material", materialSelect);
    materialSelect.onchange = () => {
        currentMaterial = parseInt(materialSelect.value, 10) || 0;
        for (const inst of instances) inst.sim?.setMaterial?.(currentMaterial);
        refreshPhysicsParamVisibility();
    };
    const refreshMaterialRow = (): void => {
        materialRow.style.display = currentMethod === "PB-MPM" ? "" : "none";
    };

    // Particle radius slider.
    const radiusInput = document.createElement("input");
    radiusInput.type = "range";
    radiusInput.min = "0.01";
    radiusInput.max = "0.2";
    radiusInput.step = "0.005";
    radiusInput.value = String(radiusValue);
    radiusInput.style.cssText = "width:100%;";
    const radiusVal = document.createElement("span");
    radiusVal.style.cssText = "color:#9fb4cc;float:right;";
    radiusVal.textContent = radiusValue.toFixed(3);
    radiusInput.oninput = () => {
        radiusValue = parseFloat(radiusInput.value);
        radiusVal.textContent = radiusValue.toFixed(3);
    };
    const radiusLabelWrap = document.createElement("div");
    radiusLabelWrap.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
    radiusLabelWrap.append(document.createTextNode("Particle radius"), radiusVal);
    const radiusRow = document.createElement("div");
    radiusRow.style.cssText = "margin-bottom:8px;";
    radiusRow.append(radiusLabelWrap, radiusInput);

    // Fill strategy — how a mesh becomes particles: a filled interior, or points scattered over the
    // surface. `auto` picks per mesh (most ship geometry is open single-sided panels with no interior
    // to fill); the forced options make the difference visible on the same prop.
    const fillSelect = document.createElement("select");
    fillSelect.id = "liq-fill";
    styleSelect(fillSelect);
    for (const [value, label] of [
        ["auto", "auto (per mesh)"],
        ["volume", "volume (fill interior)"],
        ["surface", "surface (shell only)"],
    ] as [MeshFillStrategy, string][]) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = label;
        fillSelect.append(opt);
    }
    fillSelect.value = fillStrategy;
    fillSelect.onchange = () => {
        fillStrategy = fillSelect.value as MeshFillStrategy;
        setStatus();
    };
    const fillRow = labelledRow("Fill strategy", fillSelect);

    // Impulse intensity slider — scales the hand-off launch (0 = none, 1 = tuned default).
    const impulseInput = document.createElement("input");
    impulseInput.type = "range";
    impulseInput.id = "liq-impulse";
    impulseInput.min = "0";
    impulseInput.max = "20";
    impulseInput.step = "0.05";
    impulseInput.value = String(impulseIntensity);
    impulseInput.style.cssText = "width:100%;";
    const impulseVal = document.createElement("span");
    impulseVal.style.cssText = "color:#9fb4cc;float:right;";
    impulseVal.textContent = `${impulseIntensity.toFixed(2)}×`;
    impulseInput.oninput = () => {
        impulseIntensity = parseFloat(impulseInput.value);
        impulseVal.textContent = `${impulseIntensity.toFixed(2)}×`;
    };
    const impulseLabelWrap = document.createElement("div");
    impulseLabelWrap.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
    impulseLabelWrap.append(document.createTextNode("Impulse intensity"), impulseVal);
    const impulseRow = document.createElement("div");
    impulseRow.style.cssText = "margin-bottom:8px;";
    impulseRow.append(impulseLabelWrap, impulseInput);

    // Impulse direction — the uniform part of the burst. Free-form components (normalised at use), so
    // [0,1,0] is straight up, [-1,0,0] blows the prop back down the corridor, and the vector's length
    // is ignored. (0,0,0) is special: it means the shot ray — camera to click point — so the water is
    // thrown the way you fired. Exported with the intensity so Aquanova reproduces the same burst.
    const dirInputs: HTMLInputElement[] = [];
    const dirFieldRow = document.createElement("div");
    dirFieldRow.style.cssText = "display:flex;gap:4px;";
    for (let axis = 0; axis < 3; axis++) {
        const box = document.createElement("input");
        box.type = "number";
        box.id = `liq-impulse-dir-${"xyz"[axis]}`;
        box.step = "0.1";
        box.value = String(impulseDir[axis]);
        box.title = `Impulse direction ${"XYZ"[axis]} (normalised before use)`;
        box.style.cssText = "width:100%;min-width:0;background:#1b2430;color:#dbe6f2;border:1px solid #33465c;border-radius:3px;padding:2px 4px;";
        box.oninput = () => {
            const v = parseFloat(box.value);
            impulseDir[axis] = isFinite(v) ? v : 0;
            showDirHint();
        };
        dirInputs.push(box);
        dirFieldRow.append(box);
    }
    /** Push `impulseDir` back into the three boxes (after an import). */
    const showImpulseDir = (): void => {
        for (let axis = 0; axis < 3; axis++) dirInputs[axis]!.value = String(impulseDir[axis]);
        showDirHint();
    };
    const dirLabelWrap = document.createElement("div");
    dirLabelWrap.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
    dirLabelWrap.append(document.createTextNode("Impulse direction (x, y, z)"));
    dirLabelWrap.title = "Normalised before use, so only the direction matters. (0, 0, 0) = the shot ray (camera → click point).";
    const dirHint = document.createElement("span");
    dirHint.style.cssText = "color:#9fb4cc;float:right;";
    const showDirHint = (): void => {
        dirHint.textContent = Math.hypot(impulseDir[0], impulseDir[1], impulseDir[2]) > 1e-6 ? "" : "shot ray";
    };
    showDirHint();
    dirLabelWrap.append(dirHint);
    const impulseDirRow = document.createElement("div");
    impulseDirRow.style.cssText = "margin-bottom:8px;";
    impulseDirRow.append(dirLabelWrap, dirFieldRow);

    // Impulse radius — the blast sphere around the impact point. At 0 the radius is derived per mesh
    // from the sampled volume, so the falloff scales with the prop; any other value pins it in world
    // units, which is what you want when several props should burst with the same reach.
    const radiusImpInput = document.createElement("input");
    radiusImpInput.type = "range";
    radiusImpInput.id = "liq-impulse-radius";
    radiusImpInput.min = "0";
    radiusImpInput.max = "20";
    radiusImpInput.step = "0.1";
    radiusImpInput.value = String(impulseRadius);
    radiusImpInput.style.cssText = "width:100%;";
    const radiusImpVal = document.createElement("span");
    radiusImpVal.style.cssText = "color:#9fb4cc;float:right;";
    const showImpulseRadius = (): void => {
        radiusImpVal.textContent = impulseRadius > 0 ? `${impulseRadius.toFixed(1)} m` : "auto (per mesh)";
    };
    showImpulseRadius();
    radiusImpInput.oninput = () => {
        impulseRadius = parseFloat(radiusImpInput.value);
        showImpulseRadius();
    };
    const radiusImpLabelWrap = document.createElement("div");
    radiusImpLabelWrap.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
    radiusImpLabelWrap.append(document.createTextNode("Impulse radius"), radiusImpVal);
    const impulseRadiusRow = document.createElement("div");
    impulseRadiusRow.style.cssText = "margin-bottom:8px;";
    impulseRadiusRow.append(radiusImpLabelWrap, radiusImpInput);

    // Grid size — the simulation domain, in world units, as a FULL extent per axis. The domain wall is
    // a hard boundary, so this decides how far a puddle may spread (X/Z) and how much headroom a burst
    // has (Y). 0 on an axis keeps the automatic size derived from the prop's footprint. Exported so
    // Aquanova builds the same box; without it a prop tuned here spreads differently there.
    const gridInputs: HTMLInputElement[] = [];
    const gridFieldRow = document.createElement("div");
    gridFieldRow.style.cssText = "display:flex;gap:4px;";
    for (let axis = 0; axis < 3; axis++) {
        const box = document.createElement("input");
        box.type = "number";
        box.id = `liq-grid-${"xyz"[axis]}`;
        box.min = "0";
        box.step = "0.5";
        box.value = String(gridSize[axis]);
        box.title = `Simulation domain size on ${"XYZ"[axis]} in world units (full extent). 0 = automatic.${axis === 1 ? " Y is a minimum — raised when needed to clear the prop." : ""}`;
        box.style.cssText = "width:100%;min-width:0;background:#1b2430;color:#dbe6f2;border:1px solid #33465c;border-radius:3px;padding:2px 4px;";
        box.oninput = () => {
            const v = parseFloat(box.value);
            gridSize[axis] = isFinite(v) && v > 0 ? v : 0;
            showGridHint();
        };
        gridInputs.push(box);
        gridFieldRow.append(box);
    }
    /** Push `gridSize` back into the three boxes (after an import). */
    const showGridSize = (): void => {
        for (let axis = 0; axis < 3; axis++) gridInputs[axis]!.value = String(gridSize[axis]);
        showGridHint();
    };
    const gridLabelWrap = document.createElement("div");
    gridLabelWrap.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
    gridLabelWrap.append(document.createTextNode("Grid size (x, y, z) m"));
    gridLabelWrap.title =
        "Simulation domain, full extent in world units. Centred on the prop in X/Z, resting on the ground in Y. Y is a minimum — it is raised when needed to clear the prop. 0 on an axis = automatic (from the prop's footprint). Takes effect on the next liquefaction.";
    const gridHint = document.createElement("span");
    gridHint.style.cssText = "color:#9fb4cc;float:right;";
    const showGridHint = (): void => {
        const autoAxes = ["x", "y", "z"].filter((_, i) => gridSize[i]! <= 0);
        gridHint.textContent = autoAxes.length === 3 ? "auto" : autoAxes.length ? `auto: ${autoAxes.join(", ")}` : "";
    };
    showGridHint();
    gridLabelWrap.append(gridHint);
    const gridRow = document.createElement("div");
    gridRow.style.cssText = "margin-bottom:8px;";
    gridRow.append(gridLabelWrap, gridFieldRow);

    // "Use mesh colours" — tint the water by the liquefied mesh's texture/vertex colours
    // (per-particle) instead of the uniform water colour.
    const meshColorInput = document.createElement("input");
    meshColorInput.type = "checkbox";
    meshColorInput.id = "liq-mesh-colors";
    meshColorInput.checked = useMeshColors;
    meshColorInput.style.cssText = "margin-right:6px;vertical-align:middle;";
    meshColorInput.onchange = () => {
        useMeshColors = meshColorInput.checked;
        surfaceTask.setUseParticleColor(useMeshColors);
    };
    const meshColorRow = document.createElement("label");
    meshColorRow.style.cssText = "display:block;margin-bottom:8px;color:#b6c4d6;cursor:pointer;";
    meshColorRow.append(meshColorInput, document.createTextNode("Use mesh colours"));

    const restartBtn = document.createElement("button");
    restartBtn.id = "liq-restart";
    restartBtn.textContent = "Restart (reset all foes)";
    restartBtn.style.cssText = "width:100%;padding:6px;margin-top:4px;border:0;border-radius:6px;cursor:pointer;background:#4a5568;color:#fff;font-weight:600;";
    restartBtn.onclick = () => restart();

    const status = document.createElement("div");
    status.style.cssText = "margin-top:8px;color:#8fa4bc;min-height:1.1em;";
    const partCount = document.createElement("div");
    partCount.style.cssText = "margin-top:2px;color:#8fa4bc;min-height:1.1em;";
    partCount.textContent = "0 particles";
    function setStatus(): void {
        const solid = instances.filter((i) => i.phase === "solid" && !i.sampling).length;
        const active = instances.filter((i) => i.sampling || i.phase === "dissolving" || i.phase === "fluid" || i.phase === "fading").length;
        // Which path the live particles actually came from — the interesting readout when `auto` is on,
        // since it decides per mesh and most ship geometry has no interior to fill.
        const live = instances.filter((i) => i.phase === "dissolving" || i.phase === "fluid" || i.phase === "fading");
        const shells = live.filter((i) => i.shell).length;
        const fillNote = live.length ? ` · ${live.length - shells} volume / ${shells} surface` : "";
        status.textContent = `${solid} solid · ${active} liquefying${fillNote} · ${FOE_SET === "ship" ? "click a node" : "click a foe"}`;
        canvas.dataset.solid = String(solid);
        canvas.dataset.active = String(active);
        canvas.dataset.shells = String(shells);
        canvas.dataset.volumes = String(live.length - shells);
    }

    function refreshPhysicsParamVisibility(): void {
        controls.setVisiblePhysicsParams(currentMethod === "PB-MPM" ? pbmpmParamKeysForMaterial(currentMaterial) : null);
    }

    function switchMethod(method: string): void {
        if (method === currentMethod) return;
        currentMethod = method;
        canvas.dataset.method = method;
        controls.setMethod(method);
        controls.rebuildPhysics(method);
        refreshMaterialRow();
        refreshPhysicsParamVisibility();
    }

    const controls = createFluidControlsPanel({
        hideParticles: true,
        hideMethod: false,
        hideContainerToggle: true,
        hideFoam: true,
        hidePhysics: false,
        hidePhysScale: true,
        hideDebug: true,
        hideGpuTiming: false,
        panelStyle: PANEL_STYLE,
        schemas: LIQ_SCHEMAS,
        methods: ["PBF", "MLS-MPM", "PB-MPM"],
        particleCounts: [],
        initial: {
            method: currentMethod,
            count: 0,
            physScale: 1,
            color: DEF_COLOR,
            absorption: DEF_ABSORPTION,
            size: DEF_SIZE,
            refraction: DEF_REFRACTION,
            specular: DEF_SPECULAR,
            depthBlur: DEF_DEPTH_BLUR,
            depthBlurThreshold: DEF_DEPTH_BLUR_THRESHOLD,
            thicknessBlur: DEF_THICKNESS_BLUR,
            half: DEF_HALF,
            thicknessDownscale: DEF_THICKNESS_DOWNSCALE,
            surfaceFilter: DEF_SURFACE_FILTER,
            narrowDelta: DEF_NARROW_DELTA,
            narrowMu: DEF_NARROW_MU,
            anisotropic: false,
            renderMode: "surface",
            debug: "none",
            showContainer: true,
            foam: {
                enabled: false,
                kTa: 40,
                kWc: 40,
                kb: 0.8,
                kd: 0.5,
                tMin: 0.3,
                tMax: 2.0,
                poolScale: 3,
                size: 1,
                blurRadius: 4,
                lightIntensity: 0.9,
                ambient: 0.5,
                aoStrength: 0.5,
                normalStrength: 6,
                debugTexture: "off",
                softness: 0.25,
                density: 1.6,
                subsurfaceStrength: 0.4,
                subsurfaceColor: "#b8d1f2",
            },
        },
        gpu: { stages: ["Simulation", "Surface"], supported: profiler !== null },
        on: {
            onMethod: (m) => switchMethod(m),
            onRenderMode: () => {}, // surface only in the multi-target demo
            onColor: (rgb) => surfaceTask.setFluidColor(rgb),
            onAbsorption: (v) => surfaceTask.setAbsorption(v),
            onParticleSize: (s) => surfaceTask.setSizeScale(s),
            onRefraction: (v) => surfaceTask.setRefractionStrength(v),
            onSpecular: (v) => surfaceTask.setSpecularPower(v),
            onDepthBlur: (size, threshold) => surfaceTask.setDepthBlur(size, threshold),
            onThicknessBlur: (v) => surfaceTask.setThicknessBlur(v),
            onHalf: (on) => surfaceTask.setHalfRender(on),
            onSurfaceFilter: (m) => surfaceTask.setSurfaceFilter(m),
            onNarrowRange: (delta, mu) => surfaceTask.setNarrowRange(delta, mu),
            onThicknessDownscale: (v) => surfaceTask.setThicknessDownscale(v),
            // Physics sliders apply LIVE to every running sim AND seed the next shot.
            onPhysicsParam: (key, value) => {
                physValues[currentMethod]![key] = value;
                for (const inst of instances) inst.sim?.setParam(key, value);
            },
            onReset: () => {
                const defaults = SCHEMA_DEFAULTS[currentMethod]!;
                physValues[currentMethod] = { ...defaults };
                for (const inst of instances) for (const [k, v] of Object.entries(defaults)) inst.sim?.setParam(k, v);
                controls.setPhysics(defaults);
                controls.rebuildPhysics(currentMethod);
            },
        },
    });

    // ── Export / import fluid settings ───────────────────────────────────────
    // Reuse the fluid demo's grouped JSON shape (fluid/preset-io.js): serialise the live
    // physics + surface-render params so a setting authored here can be dropped straight
    // into another demo (e.g. aquanova's FLUID_SETTING) or re-imported below to iterate.
    const IO_BTN_STYLE = "flex:1;padding:6px;border:0;border-radius:6px;cursor:pointer;background:#4a5568;color:#fff;font-weight:600;";
    function liqPairState(): PairState {
        const v = controls.getValues();
        return {
            schema: { ...physValues[currentMethod]! },
            demoParams: { particleRadius: radiusValue, useMeshColors: useMeshColors ? 1 : 0 },
            color: v.color,
            half: v.half,
            thicknessDownscale: v.thicknessDownscale,
            absorption: v.absorption,
            size: v.size,
            physScale: 1,
            count: 0,
            material: currentMethod === "PB-MPM" ? currentMaterial : undefined,
            renderMode: "surface",
            refraction: v.refraction,
            specular: v.specular,
            depthBlur: v.depthBlur,
            depthBlurThreshold: v.depthBlurThreshold,
            thicknessBlur: v.thicknessBlur,
            surfaceFilter: v.surfaceFilter,
            narrowDelta: v.narrowDelta,
            narrowMu: v.narrowMu,
            anisotropic: v.anisotropic,
            anisoSurfScale: v.anisoSurfScale,
            foam: v.foam,
            demoState: {},
            showContainer: v.showContainer,
        };
    }
    function applyImportedPreset(j: FluidExportJson): void {
        // Switch to the file's method first so the physics sliders target the right block.
        const method = j.meta?.method;
        if (method && LIQ_SCHEMAS[method]) switchMethod(method);
        const p = presetFromExportJson(j);
        // Physics: update the seed values, the sliders AND every running sim.
        if (p.schema) {
            const merged = { ...SCHEMA_DEFAULTS[currentMethod], ...p.schema };
            physValues[currentMethod] = merged;
            controls.setPhysics(merged);
            controls.rebuildPhysics(currentMethod);
            for (const inst of instances) for (const [k, val] of Object.entries(merged)) inst.sim?.setParam(k, val);
        }
        if (typeof p.material === "number" && currentMethod === "PB-MPM") {
            currentMaterial = p.material;
            materialSelect.value = String(currentMaterial);
            for (const inst of instances) inst.sim?.setMaterial?.(currentMaterial);
        }
        // Particle radius rides in demoParams. It's a construction-time param (like the slider, it can't
        // change a running sim) so restore the value + its slider; it takes effect on the next spawn.
        const pr = p.demoParams?.particleRadius;
        if (typeof pr === "number" && isFinite(pr)) {
            radiusValue = pr;
            radiusInput.value = String(radiusValue);
            radiusVal.textContent = radiusValue.toFixed(3);
        }
        // "Use mesh colours" also rides in demoParams (encoded 0/1). Restore state + checkbox and apply LIVE.
        const umc = p.demoParams?.useMeshColors;
        if (typeof umc === "number") {
            useMeshColors = umc !== 0;
            meshColorInput.checked = useMeshColors;
            surfaceTask.setUseParticleColor(useMeshColors);
        }
        // Surface render — these setters fire their host callbacks, applying LIVE to surfaceTask.
        if (p.color !== undefined) controls.setColor(p.color);
        if (p.absorption !== undefined) controls.setAbsorption(p.absorption);
        if (p.size !== undefined) controls.setParticleSize(p.size);
        if (p.refraction !== undefined) controls.setRefraction(p.refraction);
        if (p.specular !== undefined) controls.setSpecular(p.specular);
        if (p.depthBlur !== undefined && p.depthBlurThreshold !== undefined) controls.setDepthBlur(p.depthBlur, p.depthBlurThreshold);
        if (p.thicknessBlur !== undefined) controls.setThicknessBlur(p.thicknessBlur);
        if (p.half !== undefined) controls.setHalf(p.half);
        if (p.surfaceFilter !== undefined) controls.setSurfaceFilter(p.surfaceFilter);
        if (p.narrowDelta !== undefined && p.narrowMu !== undefined) controls.setNarrowRange(p.narrowDelta, p.narrowMu);
        if (p.thicknessDownscale !== undefined) controls.setThicknessDownscale(p.thicknessDownscale);
        // Impulse — intensity + direction ride in their own block (absent in files written before it
        // existed, and in fluid-demo presets, which leaves the current values alone).
        const imp = j.impulse;
        if (imp) {
            if (typeof imp.intensity === "number" && isFinite(imp.intensity)) {
                impulseIntensity = imp.intensity;
                impulseInput.value = String(impulseIntensity);
                impulseVal.textContent = `${impulseIntensity.toFixed(2)}×`;
            }
            const d = imp.direction;
            if (Array.isArray(d) && d.length === 3 && d.every((v) => typeof v === "number" && isFinite(v))) {
                impulseDir[0] = d[0];
                impulseDir[1] = d[1];
                impulseDir[2] = d[2];
                showImpulseDir();
            }
            if (typeof imp.radius === "number" && isFinite(imp.radius) && imp.radius >= 0) {
                impulseRadius = imp.radius;
                radiusImpInput.value = String(impulseRadius);
                showImpulseRadius();
            }
        }
        // Grid size — same story: its own optional block, absent in older files, which keeps whatever
        // the sliders currently hold rather than silently snapping back to automatic.
        const g = j.grid;
        if (g) {
            const axes = [g.x, g.y, g.z];
            for (let axis = 0; axis < 3; axis++) {
                const v = axes[axis];
                if (typeof v === "number" && isFinite(v) && v >= 0) gridSize[axis] = v;
            }
            showGridSize();
        }
        refreshMaterialRow();
        refreshPhysicsParamVisibility();
    }
    const exportBtn = document.createElement("button");
    exportBtn.textContent = "Export";
    exportBtn.style.cssText = IO_BTN_STYLE;
    exportBtn.onclick = () => {
        const data = exportJsonFromPairState("liquefactor", currentMethod, liqPairState());
        // Impulse is a liquefaction concept the shared PairState has no slot for, so it is attached
        // here. Aquanova reads the same block straight out of the setting file.
        data.impulse = { intensity: impulseIntensity, direction: [impulseDir[0], impulseDir[1], impulseDir[2]], radius: impulseRadius };
        data.grid = { x: gridSize[0], y: gridSize[1], z: gridSize[2] };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `liquefactor-${currentMethod}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    };
    const importInput = document.createElement("input");
    importInput.type = "file";
    importInput.accept = "application/json,.json";
    importInput.style.display = "none";
    importInput.onchange = () => {
        const file = importInput.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
            try {
                applyImportedPreset(JSON.parse(String(reader.result)) as FluidExportJson);
                status.textContent = `Imported ${file.name}`;
            } catch (e) {
                status.textContent = `Import failed: ${(e as Error).message}`;
            }
        };
        reader.readAsText(file);
        importInput.value = ""; // allow re-importing the same file
    };
    const importBtn = document.createElement("button");
    importBtn.textContent = "Import";
    importBtn.style.cssText = IO_BTN_STYLE;
    importBtn.onclick = () => importInput.click();
    const ioRow = document.createElement("div");
    ioRow.style.cssText = "display:flex;gap:6px;margin-top:6px;";
    ioRow.append(exportBtn, importBtn, importInput);

    controls.demoSlot.append(
        title,
        subtitle,
        foeRow,
        labelledRow("Sampling mode", modeSelect),
        envRow,
        materialRow,
        radiusRow,
        fillRow,
        impulseRow,
        impulseDirRow,
        impulseRadiusRow,
        gridRow,
        meshColorRow,
        restartBtn,
        ioRow,
        status,
        partCount
    );
    document.body.append(controls.root);
    if (controls.gpu) {
        // The demo's own panel is top-left, so pin the GPU pane top-right to avoid overlap.
        controls.gpu.panel.style.left = "auto";
        controls.gpu.panel.style.right = "12px";
        document.body.appendChild(controls.gpu.panel);
    }
    canvas.dataset.timing = profiler ? "on" : "unavailable";
    canvas.dataset.method = currentMethod;
    refreshMaterialRow();
    refreshPhysicsParamVisibility();
    setStatus();

    // ── Programmatic hooks for headless QA ───────────────────────────────────
    (window as unknown as { __liquefactor?: unknown }).__liquefactor = {
        getInstances: () =>
            instances.map((i) => {
                const w = i.meshes[0]!.worldMatrix as unknown as ArrayLike<number>;
                return { key: i.key, phase: i.phase, count: i.count, sampling: i.sampling, tex: i.diffuseTexs.map((t) => [t.width, t.height]), pos: [w[12]!, w[13]!, w[14]!] };
            }),
        usesWorker: () => workerPool.length > 0,
        workerCount: () => workerPool.length,
        getTotalParticles: () => virtualSim.count,
        getInstancePos: (key: string) => {
            const inst = instances.find((i) => i.key === key);
            return inst ? ([inst.root.position.x, inst.root.position.y, inst.root.position.z] as [number, number, number]) : null;
        },
        restart: () => restart(),
        camState: () => ({ px: cam.position.x, py: cam.position.y, pz: cam.position.z, tx: cam.target.x, ty: cam.target.y, tz: cam.target.z, yaw: camYaw, pitch: camPitch }),
        dynBodies: () =>
            dynDisplays.map((d) => ({
                name: d.disp.name,
                rest: d.rest,
                pos: [d.disp.position.x, d.disp.position.y, d.disp.position.z] as [number, number, number],
                rot: [d.disp.rotationQuaternion.x, d.disp.rotationQuaternion.y, d.disp.rotationQuaternion.z, d.disp.rotationQuaternion.w] as [number, number, number, number],
                vel: physWorld
                    ? (() => {
                          const v = getPhysicsBodyLinearVelocity(physWorld, d.body);
                          return [v.x, v.y, v.z] as [number, number, number];
                      })()
                    : null,
            })),
        camTo: (px: number, py: number, pz: number, tx: number, ty: number, tz: number) => {
            cam.position.set(px, py, pz);
            const dx = tx - px;
            const dy = ty - py;
            const dz = tz - pz;
            camYaw = Math.atan2(dx, dz);
            camPitch = Math.atan2(dy, Math.hypot(dx, dz));
        },
        liquefyKey: (key: string): boolean => {
            const inst = instances.find((i) => i.key === key);
            if (!inst) return false;
            const w = inst.meshes[0]!.worldMatrix as unknown as ArrayLike<number>;
            liquefyGroup(inst, [w[12]!, w[13]!, w[14]!]);
            return true;
        },
        grading: (): Record<string, unknown> => ({
            exposure: scene.imageProcessing.exposure,
            contrast: scene.imageProcessing.contrast,
            toneMapping: scene.imageProcessing.toneMapping,
            toneMappingEnabled: scene.imageProcessing.toneMappingEnabled,
            ibl: (instances[0]?.materials[0] as unknown as { environmentIntensity?: number } | undefined)?.environmentIntensity,
            hemi: hemiLight.intensity,
            sun: sun.intensity,
            env: currentEnvKey,
        }),
        setFillStrategy: (s: MeshFillStrategy) => {
            fillStrategy = s;
            fillSelect.value = s;
        },
        fillOf: (key: string): boolean | null => {
            const inst = instances.find((i) => i.key === key);
            return inst && inst.count > 0 ? inst.shell : null;
        },
        setImpulse: (v: number) => {
            impulseIntensity = v;
            impulseInput.value = String(v);
            impulseVal.textContent = `${v.toFixed(2)}×`;
        },
        setImpulseDir: (x: number, y: number, z: number) => {
            impulseDir[0] = x;
            impulseDir[1] = y;
            impulseDir[2] = z;
            showImpulseDir();
        },
        setImpulseRadius: (r: number) => {
            impulseRadius = r;
            radiusImpInput.value = String(r);
            showImpulseRadius();
        },
        impulseState: (): Record<string, unknown> => ({ intensity: impulseIntensity, direction: [...impulseDir], radius: impulseRadius }),
        setGridSize: (x: number, y: number, z: number) => {
            gridSize[0] = x;
            gridSize[1] = y;
            gridSize[2] = z;
            showGridSize();
        },
        gridState: (): number[] => [...gridSize],
        exportPreset: (): unknown => {
            const data = exportJsonFromPairState("liquefactor", currentMethod, liqPairState());
            data.impulse = { intensity: impulseIntensity, direction: [impulseDir[0], impulseDir[1], impulseDir[2]], radius: impulseRadius };
            data.grid = { x: gridSize[0], y: gridSize[1], z: gridSize[2] };
            return data;
        },
        importPreset: (j: FluidExportJson): void => applyImportedPreset(j),
        setUseMeshColors: (on: boolean) => {
            useMeshColors = on;
            meshColorInput.checked = on;
            surfaceTask.setUseParticleColor(on);
        },
        readColors: async (key: string, n = 12, stride = 1): Promise<number[] | null> => {
            const inst = instances.find((i) => i.key === key);
            if (!inst || !inst.colorBuffer || inst.count === 0) return null;
            const step = Math.max(1, Math.floor(stride));
            const count = Math.min(n, Math.floor(inst.count / step) || 1);
            const rb = device.createBuffer({ label: "liq-color-readback", size: inst.count * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const enc = device.createCommandEncoder();
            enc.copyBufferToBuffer(inst.colorBuffer, 0, rb, 0, inst.count * 16);
            device.queue.submit([enc.finish()]);
            await rb.mapAsync(GPUMapMode.READ);
            const all = new Float32Array(rb.getMappedRange().slice(0));
            rb.unmap();
            rb.destroy();
            const out: number[] = [];
            for (let i = 0; i < count; i++) {
                const p = i * step * 4;
                out.push(all[p]!, all[p + 1]!, all[p + 2]!, all[p + 3]!);
            }
            return out;
        },
        shootAt: (px: number, py: number) =>
            pickAsync(picker, px, py, { filter: (m) => instanceOf(m)?.phase === "solid" && !instanceOf(m)?.sampling }).then((info) => {
                const inst = info.pickedMesh ? instanceOf(info.pickedMesh) : undefined;
                const onEnemy = !!info.hit && !!info.pickedPoint && !!inst && inst.phase === "solid" && !inst.sampling;
                let dissolveStarted = false;
                if (onEnemy && info.pickedPoint) dissolveStarted = requestSample(inst!, info.pickedPoint);
                return { hit: info.hit, onEnemy, dissolveStarted, key: inst?.key ?? null };
            }),
    };

    // Click a foe to liquefy it from the hit point. RIGHT-drag looks around, so LMB is free for
    // shooting in both modes — the old Shift gate existed only because left-drag used to orbit.
    //
    // One shot = the picked node's primitives + every node its manifest behaviour links to
    // (transitively), all sampled from the SAME hit point, so the two demos melt the same group. A
    // node the manifest says nothing about simply melts its own primitives, as before. The FLUID is
    // always the demo's current panel settings — this demo exists to audition them.
    function liquefyGroup(inst: Instance, hit: readonly [number, number, number]): void {
        // Direction of the click ray, captured NOW: the melt runs for over a second and the camera is
        // free to move, so resolving it at eruption would use a stale-feeling direction. Used when the
        // impulse direction is (0,0,0) — "push it the way I shot it".
        const dx = hit[0] - cam.position.x;
        const dy = hit[1] - cam.position.y;
        const dz = hit[2] - cam.position.z;
        const dl = Math.hypot(dx, dy, dz);
        const shotDir: [number, number, number] = dl > 1e-6 ? [dx / dl, dy / dl, dz / dl] : ([...IMPULSE_DEFAULT_DIR] as [number, number, number]);
        const group: Instance[] = [];
        const seen = new Set<Instance>();
        const queue: Instance[] = [inst];
        while (queue.length) {
            const next = queue.shift()!;
            if (seen.has(next)) continue;
            seen.add(next);
            group.push(next);
            for (const s of groupOfInstance.get(next) ?? []) if (!seen.has(s)) queue.push(s);
            for (const name of linkedMeshNames(behaviorOfInstance.get(next))) for (const o of instancesByNodeName.get(name) ?? []) if (!seen.has(o)) queue.push(o);
        }
        for (const m of group) if (m.phase === "solid" && !m.sampling) requestSample(m, hit);
        // Everything actually taking part shares ONE member list, so the group can erupt on the same
        // frame. Each member's clip front reaches its own maxR at a different time (the pane you hit is
        // nearer the impact than its twin), and erupting on that alone made the near pane burst first.
        const shot = group.filter((m) => m.sampling || m.phase === "dissolving");
        for (const m of shot) {
            m.shot = shot;
            m.dissolved = false;
            m.shotDir = shotDir;
        }
    }

    /** Erupt every member of a shot at once, once none are still sampling and all have fully melted. */
    function eruptIfReady(shot: Instance[] | null): void {
        if (!shot) return;
        for (const m of shot) if (m.sampling) return;
        for (const m of shot) if (m.phase === "dissolving" && !m.dissolved) return;
        for (const m of shot) if (m.phase === "dissolving") finishShot(m);
    }

    /** Drop an instance out of its shot — its sample failed, or it was reset — and let whatever is left
     *  erupt. Without this a member that never reaches "dissolving" would strand its group forever. */
    function leaveShot(inst: Instance): void {
        const shot = inst.shot;
        if (!shot) return;
        inst.shot = null;
        inst.dissolved = false;
        const i = shot.indexOf(inst);
        if (i >= 0) shot.splice(i, 1);
        eruptIfReady(shot);
    }

    canvas.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        const rect = canvas.getBoundingClientRect();
        const px = (ev.clientX - rect.left) * (canvas.width / rect.width);
        const py = (ev.clientY - rect.top) * (canvas.height / rect.height);
        void pickAsync(picker, px, py, { filter: (m) => instanceOf(m)?.phase === "solid" && !instanceOf(m)?.sampling }).then((info) => {
            if (!info.hit || !info.pickedPoint || !info.pickedMesh) return;
            const inst = instanceOf(info.pickedMesh);
            if (!inst || inst.phase !== "solid" || inst.sampling) return;
            liquefyGroup(inst, info.pickedPoint);
        });
    });

    // ── Per-frame loop ───────────────────────────────────────────────────────
    let lastRenderableCount = -1;
    let fpsAccumMs = 0;
    let fpsFrames = 0;
    onBeforeRender(scene, (deltaMs: number) => {
        updateCamera(deltaMs);
        // Open the whole-frame GPU-timing envelope BEFORE any pass is encoded this frame.
        if (profiler) {
            profiler.beginFrame();
            profiler.frameStart(engine._currentEncoder);
        }
        // The model is a dynamic (post-boot) add; its renderables materialize a few frames later.
        // Rebuild the filtered scene/foe draw lists whenever the renderable set changes so the
        // model appears in sceneColorRT (not just in the picker) once it drains in.
        if (scene._renderables.length !== lastRenderableCount) {
            lastRenderableCount = scene._renderables.length;
            invalidateFilteredSceneTasks();
        }
        updateAnimationManager(animManager, deltaMs); // advance the model's walk (skeleton pose)
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
        const growDt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 30);
        for (const inst of instances) {
            if (inst.wriggling) applyWriggle(inst); // pain shake — active from click through the dissolve
            if (inst.phase === "dissolving") {
                inst.liquefyState.frontR = Math.min(inst.liquefyState.frontR + (FOE_SET === "ship" ? LIQUEFY_SPEED_SHIP : LIQUEFY_SPEED) * growDt, inst.maxR);
                bumpUbo(inst);
                if (!inst.dissolved && inst.liquefyState.frontR >= inst.maxR) {
                    // Solid gone, but hold here (fully clipped, water still frozen) until every member
                    // of the shot has melted, so the whole group erupts on one frame.
                    inst.dissolved = true;
                    eruptIfReady(inst.shot);
                }
            } else if (inst.phase === "fluid" || inst.phase === "fading") {
                // Dispose BEFORE stepping so we never destroy a sim's buffers after having
                // already encoded a step into this frame's (not-yet-submitted) encoder.
                if (inst.phase === "fading") {
                    inst.fadeElapsed += dt;
                    if (inst.fadeElapsed >= FADE_DUR) {
                        disposeInstanceSim(inst);
                        liveShots = Math.max(0, liveShots - 1);
                        setStatus();
                        continue;
                    }
                }
                inst.sim?.step(engine._currentEncoder, dt);
                if (inst.impulseRemaining > 0) {
                    inst.impulseRemaining = Math.max(0, inst.impulseRemaining - dt);
                    if (inst.impulseRemaining === 0) inst.sim?.setForceField(null);
                }
                if (inst.phase === "fluid") {
                    inst.fluidElapsed += dt;
                    if (inst.fluidElapsed >= LIFETIME) {
                        beginFade(inst);
                        setStatus();
                    }
                }
            }
        }

        // Aggregate every live sim's render positions into the combined buffer (after stepping),
        // and fill the matching per-particle alpha (1 for running blobs, ramped for fading ones).
        // When "Use mesh colours" is on, also aggregate each sim's per-particle colour buffer.
        let off = 0;
        for (const inst of instances) {
            if (inst.sim && inst.phase !== "solid" && inst.phase !== "gone") {
                const n = inst.sim.count;
                if (off + n <= MAX_TOTAL) {
                    engine._currentEncoder.copyBufferToBuffer(inst.sim.positionBuffer, 0, combinedPos, off * 16, n * 16);
                    if (useMeshColors && inst.colorBuffer) {
                        engine._currentEncoder.copyBufferToBuffer(inst.colorBuffer, 0, combinedColor, off * 16, n * 16);
                    }
                    const a = inst.phase === "fading" ? Math.max(0, 1 - inst.fadeElapsed / FADE_DUR) : 1;
                    alphaScratch.fill(a, off, off + n);
                    off += n;
                }
            }
        }
        if (off > 0) {
            // Keep the whole active range in sync so a slot reused after a fade isn't left dim.
            device.queue.writeBuffer(combinedAlpha, 0, alphaScratch, 0, off);
            virtualSim.count = off;
        } else {
            virtualSim.count = 0;
        }
        canvas.dataset.particleCount = String(off);
        partCount.textContent = `${off.toLocaleString()} particles`;

        // GPU timing + memory read-outs on a ~2 Hz cadence (same as the FPS counter).
        fpsAccumMs += deltaMs;
        fpsFrames++;
        if (fpsAccumMs >= 500) {
            const gpu = controls.gpu;
            if (gpu) {
                gpu.fpsLabel.textContent = `${Math.round((fpsFrames * 1000) / fpsAccumMs)}`;
                gpu.refreshTiming(profiler ? profiler.results() : null);
                let simBytes = 0;
                for (const inst of instances) if (inst.sim) simBytes += inst.sim.gpuBytes;
                gpu.refreshMemory(simBytes, engine.canvas.width, engine.canvas.height);
            }
            fpsAccumMs = 0;
            fpsFrames = 0;
        }
    });

    await envReady;
    // Load all glTF foes BEFORE the scene build so they materialize reliably (a dynamic post-boot
    // add leaves the clip-plugin UBO uninitialised). The 39 MB haunted house dominates this wait.
    if (FOE_SET === "ship") {
        await loadShipFoes();
    } else {
        await Promise.all(MODEL_FOES.map((cfg) => loadModelInstance(cfg)));
    }
    await registerScene(scene);
    await startEngine(engine);

    canvas.dataset.ready = "true";
}

main().catch((err) => {
    // eslint-disable-next-line no-console
    console.error(err);
    const c = document.getElementById("renderCanvas") as HTMLCanvasElement | null;
    if (c) c.dataset.error = String(err);
});
