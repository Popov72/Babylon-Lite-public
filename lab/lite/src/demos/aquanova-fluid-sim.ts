// AquanovaFluidSim — author and audition mesh-liquefaction and fluid simulations in the Aquanova ship.
//
// Three "enemy" meshes float over the ground. Click one to LIQUEFY it: its surface
// dissolves radially from the hit point (the material clips inside a growing world-space
// "front"), the interior is CPU volume-sampled into particles, and an independent GPU
// fluid sim in the shared world-space domain launches the blob upward and melts it into
// a puddle. Multiple meshes can be dissolving / running at the same time —
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
    attachPositionGizmoToNode,
    attachRotationGizmoToNode,
    attachScaleGizmoToNode,
    createAnimationManager,
    createFreeCamera,
    createHavokWorld,
    createLineMaterial,
    createLineSystem,
    createPhysicsBody,
    createPhysicsShape,
    createPositionGizmo,
    createDirectionalLight,
    createEngine,
    createGpuPicker,
    createGround,
    createHemisphericLight,
    createRenderTarget,
    createRenderTask,
    createRotationGizmo,
    createSceneContext,
    createScaleGizmo,
    createStandardMaterial,
    createTransformNode,
    createUtilityLayer,
    enableMaterialPlugins,
    getPhysicsBodyLinearVelocity,
    getProjectionMatrix,
    getViewMatrix,
    getViewProjectionMatrix,
    isPbrMaterial,
    isGizmoInteracting,
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
    loadSkybox,
    markMaterialUboDirty,
    onBeforeRender,
    pauseAnimation,
    pickAsync,
    playAnimation,
    registerScene,
    registerUtilityLayer,
    setMeshVisible,
    setPositionGizmoLocalCoordinates,
    setRotationGizmoLocalCoordinates,
    setScaleGizmoLocalCoordinates,
    startEngine,
    updateLineSystem,
    updateAnimationManager,
} from "babylon-lite";
import type { AnimationGroup, EnvironmentTextures, FluidFlowConfig, FluidShape, Material, Mesh, PhysicsBody, PhysicsWorld, Renderable, SceneNode } from "babylon-lite";
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
import { createFlipSim, estimateFlipGpuBytes } from "babylon-lite/fluid/flip-sim.js";
import { createMlsMpmSim } from "babylon-lite/fluid/mls-mpm-sim.js";
import { createPbfSim } from "babylon-lite/fluid/pbf-sim.js";
import { createPbMpmSim, pbmpmParamKeysForMaterial } from "babylon-lite/fluid/pbmpm-sim.js";
import { fluidShapeVolume, MAX_FLUID_POLYGON_POINTS } from "babylon-lite/fluid/sim-common.js";
import type { FluidSim, FoamConfig, ForceFieldSpec, SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import { createRayForce } from "babylon-lite/fluid/ray-force.js";
import { createFluidSurfaceTask } from "babylon-lite/fluid/fluid-surface-render.js";
import { createFoamRenderTask } from "babylon-lite/fluid/foam-render.js";
import type { FoamDebugTexture } from "babylon-lite/fluid/foam-render.js";
import { fillMeshParticles } from "./particle-fill.js";
import type { MeshFillStrategy, MeshParticleFill } from "./particle-fill.js";
import type { VolumeSamplingMode } from "babylon-lite/fluid/volume-sampling/index.js";
import { createFluidControlsPanel, DEFAULT_FLUID_SCHEMAS } from "babylon-lite/fluid/controls-panel.js";
import type { PhysSchemaEntry } from "babylon-lite/fluid/controls-panel.js";
import { createFluidProfiler } from "./fluid/gpu-profiler.js";
import type { FluidProfilerImpl } from "./fluid/gpu-profiler.js";
import { exportJsonFromPairState, presetFromExportJson } from "./fluid/preset-io.js";
import type { FluidExportJson } from "./fluid/preset-io.js";
import type { PairState, PendingForce } from "./fluid/demo.js";
import { createFluidFlowEditor } from "./fluid/flow-editor.js";
import type { FluidFlowEditor, FluidFlowObjectKind } from "./fluid/flow-editor.js";
import { flipParticleCountForVolume } from "./fluid/grid-settings.js";
import { screenRay } from "./fluid/pick.js";
import { demoAssetUrl } from "./demo-asset-url.js";
import { collisionShapesForModule, worldShapesForMatrix } from "./aquanova/collision-shapes.js";
import type { ShipCollisionShape, WorldCollisionShape } from "./aquanova/collision-shapes.js";
import { packPrimitives, primBufferBytes, PRIMITIVES_WGSL } from "./aquanova/collision-field.js";
import type { FluidPrimitive } from "./aquanova/collision-field.js";
import { SKYBOX_EXT, SKYBOX_SIZE, SKYBOX_URL } from "./aquanova/constants.js";
import { applyLocalEnvironmentProbes, type LocalEnvironmentController } from "./aquanova/local-environments.js";
import { buildRuntimeLights } from "./aquanova/lights.js";
import type { ShipLight } from "./aquanova/manifest.js";
import { getMeshPoseGeometry } from "./mesh-pose-geometry.js";

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

/** Ship mode auditions the Aquanova meshes, and deliberately reproduces that demo's scene setup
 *  (grading, lighting, environment build, camera clip range, puddle spread) so the two can be
 *  compared frame by frame. The gallery keeps its own presentation. */
const SHIP_MODE = true;

// World layout. Foes rest on the ground plane at y = 0; each sim grid drops one unit below
// ground so the ground BC isn't fighting the grid's own border.
const GROUND_Y = 0;
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
const DEFAULT_FORCE_STRENGTH = 0.5;
const DEFAULT_FORCE_RADIUS = 3.5;

// Studio HDR environment — drives the fluid-surface reflections + the skybox background.
const ENV_STUDIO_URL = "https://playground.babylonjs.com/textures/environment.env";
const SUN_DIR: [number, number, number] = [-0.4, -0.82, -0.45];

const MODEL_TARGET_SIZE = 7; // auto-fit: scale each model so its largest dimension ≈ this many world units

// Use the exact Aquanova ship, manifest, and authored scale so results can be compared directly
// with the game runtime.
const SHIP_URL = "/aquanova/ship.glb";
const SHIP_MANIFEST_URL = "/aquanova/ship_manifest.json";
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
    readonly surfaceOnly: boolean; // skip volume sampling (hollow prop) → dense barycentric-UV surface shell
    homePos: [number, number, number]; // resting root position (auto-fit centred); Restart resets here. Rewritten if the root is later reparented under a physics display root.
    readonly diffuseTexs: TexInfo[]; // distinct base-colour textures across the sub-meshes (per-particle colour source)
    readonly meshTexIndex: number[]; // per display sub-mesh: its index into diffuseTexs (-1 if untextured)
    readonly baseColor: [number, number, number]; // fallback per-particle colour when there's no texture
    colorBuffer: GPUBuffer | null; // per-particle RGBA colour (filled once at shot time), fed to the surface renderer
    animation: AnimationGroup | null; // looping walk, paused while liquefaction owns this instance
    resumeAnimationOnCancel: boolean;
    readonly liquefyState: LiquefyState;
    readonly impulseBuffer: GPUBuffer;
    readonly impulseSpec: ForceFieldSpec;
    readonly collisionOwnerId: string | null;
    sim: FluidSim | null;
    collisionBuffer: GPUBuffer | null;
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

interface ManualRun {
    sim: FluidSim;
    collisionBuffer: GPUBuffer | null;
    colorBuffer: GPUBuffer;
    center: [number, number, number];
    method: string;
    configurationSignature: string;
}

async function main(): Promise<void> {
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    const renderLikeAquanova = new URL(location.href).searchParams.get("aquanovaRendering") === "1";
    canvas.dataset.aquanovaRendering = String(renderLikeAquanova);
    canvas.dataset.portalRendering = "false";

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
    const gridPosition: [number, number, number] = [0, 10, 0];

    // Free-fly camera: WASD strafes in the view plane, Space/C rise and fall, and RIGHT-drag looks
    // around — LMB stays free for shooting. An arc-rotate camera made the ship interior a chore to
    // navigate (orbiting around a target you cannot see past). Shift is deliberately NOT the descend
    // key: ship mode uses Shift+LMB to liquefy.
    const cam = createFreeCamera({ x: -3, y: 31.6, z: -40.7 }, { x: -3, y: 2.5, z: 0 }); // the old arc-rotate vantage (alpha -PI/2, beta 0.95, radius 50)
    cam.nearPlane = 0.1;
    // Ship mode matches Aquanova's clip range so depth precision (and therefore any depth-derived
    // shading) is identical between the two demos.
    cam.farPlane = SHIP_MODE ? 400 : 200;
    cam.speed = 8.8;
    scene.camera = cam;

    const cameraPositionHud = document.createElement("div");
    cameraPositionHud.id = "liq-camera-position";
    cameraPositionHud.style.cssText =
        "position:fixed;right:12px;bottom:12px;z-index:20;pointer-events:none;padding:5px 8px;border:1px solid rgba(143,164,188,0.45);border-radius:5px;" +
        "background:rgba(12,16,24,0.78);color:#dfe6ee;font:12px/1.4 ui-monospace,SFMono-Regular,Consolas,'Liberation Mono',monospace;white-space:pre;";
    const cameraPositionLine = document.createElement("div");
    const cameraGridRelativeLine = document.createElement("div");
    cameraPositionHud.append(cameraPositionLine, cameraGridRelativeLine);
    document.body.appendChild(cameraPositionHud);
    const updateCameraPositionHud = (): void => {
        const camera = [cam.position.x, cam.position.y, cam.position.z];
        const relative = camera.map((value, axis) => value - gridPosition[axis]!) as [number, number, number];
        cameraPositionLine.textContent = `Camera ${camera.map((value) => value.toFixed(2)).join(", ")}`;
        cameraGridRelativeLine.textContent = `Grid relative ${relative.map((value) => value.toFixed(2)).join(", ")}`;
        canvas.dataset.cameraPosition = `${cam.position.x},${cam.position.y},${cam.position.z}`;
        canvas.dataset.cameraGridRelative = relative.join(",");
    };
    updateCameraPositionHud();

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
    let camYawTarget = camYaw;
    let camPitchTarget = camPitch;
    const camVelocity = { x: 0, y: 0, z: 0 };
    const camKeys = new Set<string>();
    let looking = false;
    let forceDragging = false;
    let forcePointerId = -1;
    let forceLastX = 0;
    let forceLastY = 0;
    let forceLastT = 0;
    let pendingForce: PendingForce | null = null;
    let paused = false;
    canvas.dataset.paused = "false";
    const LOOK_SENS = 1 / 350;
    const LOOK_ACCELERATION = 28;
    const MOVE_ACCELERATION = 11;
    const releaseLook = (): void => {
        looking = false;
        if (document.pointerLockElement === canvas) document.exitPointerLock();
    };
    canvas.addEventListener("contextmenu", (e) => e.preventDefault()); // RMB is the look button
    canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 2) return;
        if (beginManualForceDrag(e)) return;
        looking = true;
        if (document.pointerLockElement !== canvas) void canvas.requestPointerLock();
    });
    window.addEventListener("pointerup", (e) => {
        if (e.button !== 2) return;
        if (forceDragging) {
            endManualForceDrag(e.pointerId);
        } else {
            releaseLook();
        }
    });
    canvas.addEventListener("pointercancel", (e) => {
        endManualForceDrag(e.pointerId);
        releaseLook();
    });
    document.addEventListener("pointerlockchange", () => {
        if (document.pointerLockElement !== canvas) looking = false;
    });
    document.addEventListener("pointermove", (e) => {
        if (forceDragging) {
            updateManualForceDrag(e);
            return;
        }
        if (!looking) return;
        camYawTarget += e.movementX * LOOK_SENS;
        camPitchTarget = Math.max(-1.45, Math.min(1.45, camPitchTarget - e.movementY * LOOK_SENS));
    });
    window.addEventListener("keydown", (e) => {
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
        if (e.code === "KeyP") {
            if (!e.repeat) {
                paused = !paused;
                canvas.dataset.paused = paused ? "true" : "false";
            }
            return;
        }
        if (e.code === "KeyR") {
            if (!e.repeat) {
                runPrimarySimulationAction();
            }
            return;
        }
        camKeys.add(e.code);
    });
    window.addEventListener("keyup", (e) => camKeys.delete(e.code));
    window.addEventListener("blur", () => {
        camKeys.clear();
        endManualForceDrag();
        releaseLook();
        camVelocity.x = 0;
        camVelocity.y = 0;
        camVelocity.z = 0;
    });
    const updateCamera = (deltaMs: number): void => {
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 30);
        const lookFactor = 1 - Math.exp(-dt * LOOK_ACCELERATION);
        camYaw += (camYawTarget - camYaw) * lookFactor;
        camPitch += (camPitchTarget - camPitch) * lookFactor;
        const cy = Math.cos(camYaw),
            sy = Math.sin(camYaw);
        const cp = Math.cos(camPitch),
            sp = Math.sin(camPitch);
        const fwdZ = (camKeys.has("KeyW") ? 1 : 0) - (camKeys.has("KeyS") ? 1 : 0);
        const strafe = (camKeys.has("KeyD") ? 1 : 0) - (camKeys.has("KeyA") ? 1 : 0);
        const rise = (camKeys.has("Space") ? 1 : 0) - (camKeys.has("KeyC") ? 1 : 0);
        const speed = cam.speed * (camKeys.has("ShiftLeft") || camKeys.has("ShiftRight") ? 4 : 1);
        const moveFactor = 1 - Math.exp(-dt * MOVE_ACCELERATION);
        camVelocity.x += ((sy * cp * fwdZ + cy * strafe) * speed - camVelocity.x) * moveFactor;
        camVelocity.y += ((sp * fwdZ + rise) * speed - camVelocity.y) * moveFactor;
        camVelocity.z += ((cy * cp * fwdZ - sy * strafe) * speed - camVelocity.z) * moveFactor;
        cam.position.x += camVelocity.x * dt;
        cam.position.y += camVelocity.y * dt;
        cam.position.z += camVelocity.z * dt;
        cam.target.x = cam.position.x + sy * cp;
        cam.target.y = cam.position.y + sp;
        cam.target.z = cam.position.z + cy * cp;
        updateCameraPositionHud();
    };
    const applyFreeCameraPose = (position: readonly [number, number, number], target: readonly [number, number, number]): void => {
        cam.position.set(position[0], position[1], position[2]);
        cam.target.set(target[0], target[1], target[2]);
        const dx = target[0] - position[0];
        const dy = target[1] - position[1];
        const dz = target[2] - position[2];
        if (Math.hypot(dx, dy, dz) > 1e-6) {
            camYaw = Math.atan2(dx, dz);
            camPitch = Math.atan2(dy, Math.hypot(dx, dz));
            camYawTarget = camYaw;
            camPitchTarget = camPitch;
        }
        camVelocity.x = 0;
        camVelocity.y = 0;
        camVelocity.z = 0;
        updateCameraPositionHud();
    };

    // Ship mode is lit purely by the HDRI, exactly as Aquanova is. The lights are not merely dimmed —
    // they are never registered, because a PBR material compiled with lights present renders brighter
    // than one compiled without, regardless of intensity.
    const hemiLight = createHemisphericLight([0.2, 1, 0.3], 0.8);
    // Aquanova relies on environment and authored lights rather than gallery fill lights.
    const sun = createDirectionalLight(SUN_DIR, 2.0);
    sun.position.set(12, 20, 10);

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
    const editableShipMeshes = new Set<Mesh>();
    const instanceOf = (m: unknown): Instance | undefined => meshToInstance.get(m as Mesh);
    const animManager = createAnimationManager({ engine });

    // Fit an assembled foe root onto the gallery row and register it as a live Instance.
    function registerFoeInstance(
        cfg: { key: string; x: number; z?: number; surfaceOnly?: boolean; scale?: number; inPlace?: boolean; collisionOwnerId?: string | null },
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
            surfaceOnly: cfg.surfaceOnly ?? false,
            homePos,
            diffuseTexs,
            meshTexIndex,
            baseColor: [0.8, 0.8, 0.8],
            colorBuffer: null,
            animation: anim,
            resumeAnimationOnCancel: false,
            liquefyState,
            impulseBuffer,
            impulseSpec,
            collisionOwnerId: cfg.collisionOwnerId ?? null,
            sim: null,
            collisionBuffer: null,
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
        invalidatePhaseTaskBundles();
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
    const shipCollisionPlacements: Array<{ id: string; node: SceneNode; shapes: ShipCollisionShape | readonly ShipCollisionShape[] }> = [];
    // Display roots driven by a Havok body (one per `dynamic` node). `rest` is the spawn pose, which
    // Restart teleports the body back to.
    const dynDisplays: { proxy: SceneNode; body: PhysicsBody; disp: SceneNode; rest: [number, number, number] }[] = [];
    let physWorld: PhysicsWorld | null = null;
    let localEnvironmentController: LocalEnvironmentController | null = null;

    // ── Aquanova ship meshes ─────────────────────────────────────────────────
    // Load the FULL ship.glb exactly as Aquanova does and make every ship mesh a foe, IN PLACE inside
    // the ship. Geometry, materials, trim atlas, world transforms and the particle-fill path are all
    // identical to Aquanova, so the only variables left are this demo's lighting, camera and fluid
    // controls. ship_manifest.json's `entities` names are used only to pick which room to frame.
    async function loadShipFoes(): Promise<void> {
        await shipManifestReady;
        const chunkFocus = (c: { aabb: { min: number[]; max: number[] } }): [number, number, number] => [
            -(c.aabb.min[0]! + c.aabb.max[0]!) / 2, // glTF space → Lite negates X
            1.5,
            (c.aabb.min[2]! + c.aabb.max[2]!) / 2,
        ];
        const iblStrength = shipManifest?.environment?.strength ?? DEFAULT_SHIP_IBL_STRENGTH;
        // A chunk is a room the ship editor laid out, and an EMPTY one (no meshes placed yet) is
        // exported with `aabb: null` — it has no spatial extent to describe. Every use here is
        // geometric (framing a room, hit-testing a point against one, building its collision shell),
        // so those are filtered out once rather than guarded at each site. Dereferencing them is
        // what broke this demo when the editor gained an empty room.
        //
        // The component check matches Aquanova's own loader (aquanova/manifest.ts), which reads the
        // same file: a truthy-but-malformed AABB would not crash, it would quietly yield `undefined`
        // corner components and propagate NaN into the room test and the collision box extents.
        type ShipChunkData = NonNullable<ShipManifestData["chunks"]>[number];
        const hasBounds = (c: ShipChunkData): c is ShipChunkData & { aabb: { min: number[]; max: number[] } } => c.aabb?.min?.length === 3 && c.aabb?.max?.length === 3;
        const chunks = (shipManifest?.chunks ?? []).filter(hasBounds);
        const chunkIdByRootNode = new Map<string, string>();
        for (const chunk of chunks) {
            if (chunk.node && chunk.id) {
                chunkIdByRootNode.set(chunk.node, chunk.id);
            }
        }
        const chunkOfMesh = new Map<Mesh, string>();
        const liqNames = new Set(
            Object.keys(shipManifest?.entities ?? {}).filter((name) => isLiquefiableBehavior(resolveBehavior(shipManifest?.behaviors, shipManifest?.entities, name)))
        );
        let focus: [number, number, number] = chunks[0] ? chunkFocus(chunks[0]) : [0, 1.5, 0];
        let asset;
        try {
            asset = await loadGltf(engine, SHIP_URL);
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[aquanova-fluid-sim] ship.glb load failed", err);
            return;
        }
        const shipRoot = createTransformNode("ship_root", 0, 0, 0, 0, 0, 0, 1, 1, 1);
        for (const e of asset.entities) {
            if (!("position" in e)) continue; // skip non-node entities (e.g. lights)
            e.parent = shipRoot;
            shipRoot.children.push(e);
        }
        addToScene(scene, shipRoot);

        const collectCollisionPlacements = (node: SceneNode): void => {
            const extras = node.metadata?.gltf?.extras as { id?: string; module?: string } | undefined;
            if (extras?.id && extras.module) {
                const shapes = collisionShapesForModule(shipManifest?.moduleCollision, extras.module);
                if (shapes) {
                    shipCollisionPlacements.push({ id: extras.id, node, shapes });
                }
            }
            for (const child of node.children ?? []) {
                collectCollisionPlacements(child);
            }
        };
        collectCollisionPlacements(shipRoot);
        canvas.dataset.collisionPlacements = String(shipCollisionPlacements.length);

        // Match Aquanova's material setup on the ship:
        //  • Drop KHR_materials_transmission. Both demos composite the fluid surface into the swapchain
        //    AFTER the scene renders, but transmission retargets the scene task to an offscreen HDR
        //    buffer and appends a tonemap pass that would hide the water. (It also keeps the refraction
        //    fragment out of the composed shader, which otherwise collides with punctual
        //    lights and aborts the material build.)
        //  • environmentIntensity from the manifest's dynamic value — this demo loads the authored
        //    unbaked ship, so it should match Aquanova's dynamic/unbaked materials.
        const seenMats = new Set<object>();
        const specularAA = shipManifest?.environment?.specularAA ?? false;
        const roughnessScale =
            typeof shipManifest?.environment?.reflectionRoughness === "number" && Number.isFinite(shipManifest.environment.reflectionRoughness)
                ? shipManifest.environment.reflectionRoughness
                : 1;
        const prepMaterials = (node: SceneNode): void => {
            const mat = (node as Mesh).material;
            if (mat && isPbrMaterial(mat) && !seenMats.has(mat)) {
                seenMats.add(mat);
                const ss = (mat as unknown as { subsurface?: { refraction?: unknown } }).subsurface;
                if (ss?.refraction) ss.refraction = undefined;
                mat.environmentIntensity = iblStrength;
                mat.enableSpecularAA = specularAA;
                mat.roughnessFactor = (mat.roughnessFactor ?? 1) * roughnessScale;
                markMaterialUboDirty(mat);
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
        const shipMeshes: Array<{ mesh: Mesh; owner: SceneNode }> = [];
        const seen = new Set<Mesh>();
        const walk = (node: SceneNode, owner: SceneNode, chunk: string | undefined): void => {
            for (const c of [...(node.children ?? [])]) {
                const childChunk = chunkIdByRootNode.get(c.name) ?? chunk;
                if (isMeshNode(c) && !seen.has(c)) {
                    seen.add(c);
                    shipMeshes.push({ mesh: c, owner });
                    if (childChunk) {
                        chunkOfMesh.set(c, childChunk);
                    }
                    if (c._cpuPositions && c._cpuIndices) {
                        found.push({ mesh: c, parent: node, owner });
                    }
                    continue;
                }
                const wrapper = PRIMITIVE_WRAPPER.test(c.name) && c.name.replace(PRIMITIVE_WRAPPER, "") === node.name;
                walk(c, wrapper ? owner : c, childChunk);
            }
        };
        walk(shipRoot, shipRoot, undefined);
        if (!found.length) {
            // eslint-disable-next-line no-console
            console.warn("[aquanova-fluid-sim] ship.glb contained no sampleable meshes");
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
        const disabledShipMeshes = new Set<Mesh>();
        for (const { mesh, owner } of shipMeshes) {
            if (markerNames.has(owner.name) || mesh.name.startsWith("Portal_")) {
                setMeshVisible(mesh, false);
                (mesh as { pickable?: boolean }).pickable = false;
                disabledShipMeshes.add(mesh);
            } else {
                editableShipMeshes.add(mesh);
            }
        }
        const placementIdForNode = (node: SceneNode): string | null => {
            let current: SceneNode | null = node;
            while (current) {
                const id = (current.metadata?.gltf?.extras as { id?: string } | undefined)?.id;
                if (id) {
                    return id;
                }
                current = current.parent as SceneNode | null;
            }
            return null;
        };
        const shipInstanceStart = instances.length;
        for (const { mesh, parent, owner } of found) {
            if (disabledShipMeshes.has(mesh)) {
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
            registerFoeInstance({ key: `${owner.name}#${n}`, x: 0, inPlace: true, collisionOwnerId: placementIdForNode(owner) }, root, [mesh], materials, liquefyState, null);
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

        if (renderLikeAquanova) {
            const runtimeLitMeshes = new Set(shipMeshes.map(({ mesh }) => mesh).filter((mesh) => !disabledShipMeshes.has(mesh)));
            const lights = buildRuntimeLights(scene, shipRoot, runtimeLitMeshes, chunkOfMesh, shipManifest?.lights);
            canvas.dataset.clusteredLightCount = String(lights.clusteredPoint + lights.clusteredSpot);
            canvas.dataset.runtimeLitCount = String(runtimeLitMeshes.size);
            if (lights.overflow) {
                console.warn(`[aquanova-fluid-sim] ${lights.overflow} non-clustered light(s) dropped: the shared lights UBO is full`);
            }

            // Fixed mode assigns one immutable, intersecting box-projected cubemap to each mesh.
            // This deliberately does not enable Aquanova's per-fragment cubemap blending.
            localEnvironmentController = await applyLocalEnvironmentProbes(
                scene,
                shipMeshes.map(({ mesh }) => mesh),
                { blendingEnabled: false }
            );
            if (localEnvironmentController) {
                canvas.dataset.localEnvironmentCount = String(localEnvironmentController.loaded);
                canvas.dataset.localCubemapBlending = "false";
                if (localEnvironmentController.missing.length) {
                    console.warn("[aquanova-fluid-sim] local environments not loaded:", localEnvironmentController.missing.join(", "));
                }
                localEnvironmentController.updatePoi([cam.position.x, cam.position.y, cam.position.z]);
                const fluidEnvironment = localEnvironmentController.dominantEnvironment();
                if (fluidEnvironment) {
                    surfaceTask.setEnvMap({ view: fluidEnvironment._specularCubeView, sampler: fluidEnvironment._cubeSampler });
                }
            }

            // Lighting and local-environment setup replace material objects. Keep each liquefaction
            // instance pointed at the final materials so its clip UBO is dirtied every frame.
            for (const instance of instances.slice(shipInstanceStart)) {
                const materials = new Set(instance.meshes.map((mesh) => mesh.material).filter((material): material is Material => material !== undefined));
                instance.materials.splice(0, instance.materials.length, ...materials);
            }
            // No portal visibility controller is created here: the existing scene task renders every
            // visible ship mesh every frame, which is intentional for fluid performance testing.
            canvas.dataset.fullShipRendering = "true";
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
                console.warn("[aquanova-fluid-sim] Havok unavailable — dynamic behaviours stay static", err);
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
            console.warn(`[aquanova-fluid-sim] havok: ${cn} colliders, ${dynDisplays.length} dynamic node(s)`);
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
        camYawTarget = camYaw;
        camPitchTarget = camPitch;
        camVelocity.x = 0;
        camVelocity.y = 0;
        camVelocity.z = 0;
        // eslint-disable-next-line no-console
        console.warn(`[aquanova-fluid-sim] ship meshes: ${found.length} mesh(es), ibl ${iblStrength}`);
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

    const foamTask = createFoamRenderTask(engine, scene, {
        colorRT: engine.scRT,
        depthRT,
        camera: cam,
        sim: virtualSim as unknown as FluidSim,
        getSurfaceDepth: () => surfaceTask.surfaceDepthView(),
    });
    const executeFoam = foamTask.execute?.bind(foamTask);
    foamTask.execute = (): number => {
        if (!controls.getValues().foam.enabled) {
            return 0;
        }
        let passes = 0;
        for (const sim of runningSims()) {
            if (!sim.diffuse) {
                continue;
            }
            foamTask.setSim(sim);
            passes += executeFoam?.() ?? 0;
        }
        return passes;
    };
    foamTask.setProfiler(profiler);
    addTask(scene, foamTask);

    function runningSims(): FluidSim[] {
        const sims = instances.flatMap((inst) => (inst.sim ? [inst.sim] : []));
        if (manualRun) {
            sims.push(manualRun.sim);
        }
        return sims;
    }

    function currentFoamConfig(): FoamConfig {
        const f = controls.getValues().foam;
        return {
            activeParticles: true,
            generateSpray: f.generateSpray,
            generateFoam: f.generateFoam,
            generateBubbles: f.generateBubbles,
            kTa: f.kTa,
            kWc: f.kWc,
            kTurb: f.kTurb,
            energySpeedMin: f.energySpeedMin,
            energySpeedMax: f.energySpeedMax,
            curvatureMin: f.curvatureMin,
            curvatureMax: f.curvatureMax,
            turbulenceMin: f.turbulenceMin,
            turbulenceMax: f.turbulenceMax,
            foamLayerDepth: f.foamLayerDepth,
            sprayDrag: f.sprayDrag,
            kb: f.kb,
            kd: f.kd,
            tMin: f.tMin,
            tMax: f.tMax,
            poolScale: f.poolScale,
        };
    }

    function applyFoamToSim(sim: FluidSim): void {
        const f = controls.getValues().foam;
        sim.setFoam?.(f.enabled ? currentFoamConfig() : null);
    }

    function pushFoam(): void {
        for (const sim of runningSims()) {
            applyFoamToSim(sim);
        }
        foamTask.setEnabled(controls.getValues().foam.enabled);
    }

    function fillManualColor(run: ManualRun, color = controls.getValues().color): void {
        const rgb = hexToRgb(color);
        const capacity = run.colorBuffer.size / 16;
        const colors = new Float32Array(capacity * 4);
        for (let i = 0; i < capacity; i++) {
            const offset = i * 4;
            colors[offset] = rgb[0];
            colors[offset + 1] = rgb[1];
            colors[offset + 2] = rgb[2];
            colors[offset + 3] = 1;
        }
        device.queue.writeBuffer(run.colorBuffer, 0, colors);
    }

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
    const foeTask = createRenderTask({ name: "liq-foe", rt: engine.scRT, depth: foeDepth, clr: false, autoMirror: false }, engine, scene);
    const foeRecord = foeTask.record.bind(foeTask);
    foeTask.record = (): void => {
        foeDepth._depthTexture = depthRT._depthTexture;
        foeDepth._depthView = depthRT._depthView;
        foeDepth._width = depthRT._width;
        foeDepth._height = depthRT._height;
        foeRecord();
    };
    addTask(scene, foeTask);

    const gateTaskMeshes = (task: typeof sceneTask, shouldDraw: (instance: Instance) => boolean): void => {
        const executeTask = task.execute;
        if (!executeTask) {
            throw new Error(`Render task "${task.name}" does not provide a direct execute path.`);
        }
        const execute = executeTask.bind(task);
        task.execute = (): number => {
            const hidden: Mesh[] = [];
            for (const instance of instances) {
                if (shouldDraw(instance)) continue;
                for (const mesh of instance.meshes) {
                    if (mesh.visible === false) continue;
                    mesh.visible = false;
                    hidden.push(mesh);
                }
            }
            try {
                return execute();
            } finally {
                for (const mesh of hidden) mesh.visible = true;
            }
        };
    };
    gateTaskMeshes(sceneTask, (instance) => instance.phase !== "dissolving");
    gateTaskMeshes(foeTask, (instance) => instance.phase === "dissolving");

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

    const invalidatePhaseTaskBundles = (): void => {
        for (const task of [sceneTask, foeTask]) {
            task._ob.length = 0;
        }
        foeTask.enabled = instances.some((instance) => instance.phase === "dissolving");
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
        chunks?: { id?: string; node?: string; aabb: { min: number[]; max: number[] } | null }[];
        moduleCollision?: Readonly<Record<string, ShipCollisionShape | readonly ShipCollisionShape[]>>;
        fluidSim?: string[];
        lights?: ShipLight[];
    }
    let shipManifest: ShipManifestData | null = null;
    const shipManifestReady = fetch(SHIP_MANIFEST_URL)
        .then((r) => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.json() as Promise<ShipManifestData>;
        })
        .then((j) => {
            shipManifest = j;
        })
        .catch((err: unknown) => {
            // eslint-disable-next-line no-console
            console.warn("[aquanova-fluid-sim] ship manifest load failed", err);
        });

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
    const gradedExposure = (_v: number): number => resolveExposure(shipManifest?.environment?.exposure);
    const gradedContrast = (_v: number): number => SHIP_CONTRAST;
    const makeEnvSlot = (env: EnvironmentTextures, exposure: number, contrast: number): EnvSlot => {
        // The skybox snapshots scene.imageProcessing at build time, so grade FIRST.
        scene.imageProcessing.exposure = gradedExposure(exposure);
        scene.imageProcessing.contrast = gradedContrast(contrast);
        // AquanovaFluidSim is ship-only. Its environment loader or authored Aquanova skybox owns
        // the background, so there is no gallery sky renderable to install here.
        return { env, sky: null, exposure, contrast };
    };
    const loadEnvSlot = async (c: (typeof ENV_CHOICES)[number]): Promise<EnvSlot | null> => {
        if (envSlots.has(c.key)) return envSlots.get(c.key) ?? null;
        try {
            const env = c.hdr
                ? // Both demos build the IBL at 512-pixel cube faces (sharper specular reflections on
                  // the ship's metal than the 256 default). Aquanova rendering uses its authored
                  // backdrop instead of the HDR loader's fallback skybox.
                  await loadHdrEnvironment(
                      scene,
                      c.url,
                      renderLikeAquanova
                          ? { faceSize: 512, skipGround: true, skipSkybox: true }
                          : SHIP_MODE
                            ? { faceSize: 512, skipGround: true }
                            : { faceSize: 512, skipGround: true, skipSkybox: true }
                  )
                : await loadEnvironment(scene, c.url, { brdfUrl, skipGround: true, skipSkybox: true });
            scene.imageProcessing.toneMappingEnabled = true;
            // Ship mode uses the view transform named in the manifest (Aquanova reads the same
            // field); the gallery keeps the default transform.
            const tone = resolveToneMapping(shipManifest?.environment?.toneMapping);
            scene.imageProcessing.toneMappingEnabled = tone !== null;
            if (tone) scene.imageProcessing.toneMapping = tone;
            const slot = makeEnvSlot(env, c.exposure, c.contrast);
            envSlots.set(c.key, slot);
            return slot;
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn(`[aquanova-fluid-sim] env "${c.key}" failed to load (${c.url})`, err);
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
    const envReady = (async (): Promise<void> => {
        await applyEnv(currentEnvKey);
        if (!renderLikeAquanova) {
            return;
        }
        try {
            await loadSkybox(scene, SKYBOX_URL, SKYBOX_EXT, SKYBOX_SIZE);
        } catch (err) {
            console.warn("[aquanova-fluid-sim] Aquanova skybox not loaded", err);
        }
    })();

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
    /** Shared world-space simulation domain used by both mesh and manual starts. */
    const gridSize: [number, number, number] = [40, 20, 40];
    let showGridBounds = false;
    let showGridGizmo = false;
    let currentMethod = "MLS-MPM";
    let currentMaterial = 0;
    let flipGridResolution = 160;
    let flipMarkersPerCell = 8;
    let flipParticleCapacityRequest: number | null = null;
    let mpmActiveBlocks = false;
    let mpmPagedGrid = false;
    const maxPagedGridPages = Math.max(1, Math.floor(engine._device.limits.maxStorageBufferBindingSize / 1024) - 1);
    let mpmPagedGridMaxPages = Math.min(maxPagedGridPages, 40000);
    let mpmFusedBlockDiscovery = false;
    let collisionNeighborhoodRadius = 12;
    type SimulationType = "mesh" | "fluid";
    let simulationType: SimulationType = "mesh";
    let manualRun: ManualRun | null = null;
    let manualForceStrength = DEFAULT_FORCE_STRENGTH;
    let manualForceRadius = DEFAULT_FORCE_RADIUS;
    const rayForce = createRayForce(device);
    let activeFlow: FluidFlowConfig = {
        emitters: [
            {
                id: "manual-source-1",
                name: "Manual source",
                enabled: true,
                behavior: "initial",
                transform: { position: [0, 3, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                shape: { type: "box", size: [3, 3, 3] },
                sampling: "volume",
                velocity: [0, 0, 0],
                velocitySpace: "world",
                spread: 0,
            },
        ],
        sinks: [],
    };
    let refreshFlowControls = (): void => {};
    let refreshFlipParticleCapacity = (): void => {};
    const simulationCellSize = (particleRadius = radiusValue): number =>
        currentMethod === "FLIP" ? Math.max(...gridSize) / flipGridResolution : Math.max(particleRadius * 2.4, 0.18);
    const simulationBounds = (center: readonly [number, number, number] = gridPosition): { min: [number, number, number]; max: [number, number, number] } => ({
        min: [center[0] - gridSize[0] * 0.5, center[1] - gridSize[1] * 0.5, center[2] - gridSize[2] * 0.5],
        max: [center[0] + gridSize[0] * 0.5, center[1] + gridSize[1] * 0.5, center[2] + gridSize[2] * 0.5],
    });
    const flowInWorldSpace = (flow: FluidFlowConfig): FluidFlowConfig => {
        const result = structuredClone(flow);
        for (const emitter of result.emitters) {
            emitter.transform.position = emitter.transform.position.map((value, axis) => value + gridPosition[axis]!) as [number, number, number];
        }
        for (const sink of result.sinks) {
            sink.transform.position = sink.transform.position.map((value, axis) => value + gridPosition[axis]!) as [number, number, number];
        }
        return result;
    };
    const initialFlowSignature = (flow: FluidFlowConfig): string =>
        JSON.stringify({
            initialEmittersFillCapacity: flow.initialEmittersFillCapacity ?? false,
            emitters: flow.emitters.filter((emitter) => emitter.behavior === "initial"),
        });
    const manualConfigurationSignature = (): string =>
        JSON.stringify({
            method: currentMethod,
            gridPosition,
            gridSize,
            flipGridResolution,
            flipMarkersPerCell,
            flipParticleCapacityRequest,
            initialFlow: initialFlowSignature(activeFlow),
        });

    function beginManualForceDrag(event: PointerEvent): boolean {
        if (event.button !== 2 || !event.shiftKey || simulationType !== "fluid" || !manualRun) {
            return false;
        }
        forceDragging = true;
        forcePointerId = event.pointerId;
        forceLastX = event.clientX;
        forceLastY = event.clientY;
        forceLastT = performance.now();
        canvas.setPointerCapture(event.pointerId);
        canvas.dataset.forceDragging = "true";
        return true;
    }

    function updateManualForceDrag(event: PointerEvent): void {
        if (!forceDragging || event.pointerId !== forcePointerId || !manualRun) {
            return;
        }
        const dx = event.clientX - forceLastX;
        const dy = event.clientY - forceLastY;
        const now = performance.now();
        const dtMs = Math.max(now - forceLastT, 1);
        forceLastX = event.clientX;
        forceLastY = event.clientY;
        forceLastT = now;
        const speed = (Math.hypot(dx, dy) / dtMs) * 1000;
        if (speed < 1) {
            return;
        }
        const rect = canvas.getBoundingClientRect();
        const aspect = engine.canvas.width / Math.max(1, engine.canvas.height);
        const ray = screenRay(getViewProjectionMatrix(cam, aspect), event.clientX - rect.left, event.clientY - rect.top, rect.width, rect.height);
        if (!ray) {
            return;
        }
        const world = cam.worldMatrix;
        let pushX = world[0]! * dx - world[4]! * dy;
        let pushY = world[1]! * dx - world[5]! * dy;
        let pushZ = world[2]! * dx - world[6]! * dy;
        const length = Math.hypot(pushX, pushY, pushZ) || 1;
        pushX /= length;
        pushY /= length;
        pushZ /= length;
        pendingForce = {
            origin: ray.origin,
            dir: ray.dir,
            push: [pushX, pushY, pushZ],
            radius: manualForceRadius,
            accel: speed * manualForceStrength,
        };
    }

    function endManualForceDrag(pointerId = forcePointerId): void {
        if (!forceDragging) {
            return;
        }
        forceDragging = false;
        pendingForce = null;
        rayForce.clear();
        manualRun?.sim.setForceField(null);
        if (pointerId >= 0 && canvas.hasPointerCapture(pointerId)) {
            canvas.releasePointerCapture(pointerId);
        }
        forcePointerId = -1;
        canvas.dataset.forceDragging = "false";
    }

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

    const worldShapeToFluidPrimitive = (shape: WorldCollisionShape): FluidPrimitive => {
        if (shape.kind === "box") {
            return {
                kind: "box",
                a: shape.centre,
                b: shape.halfExtents ?? [0, 0, 0],
                rotation: shape.rotation,
            };
        }
        if (shape.kind === "sphere") {
            return { kind: "sphere", a: shape.centre, radius: shape.radius ?? 0 };
        }
        return {
            kind: shape.kind,
            a: shape.pointA ?? shape.centre,
            b: shape.pointB ?? shape.centre,
            radius: shape.radius ?? 0,
        };
    };
    const shapeIntersectsSphere = (shape: WorldCollisionShape, center: readonly [number, number, number], radius: number): boolean => {
        let sx = shape.centre[0];
        let sy = shape.centre[1];
        let sz = shape.centre[2];
        let ex = shape.radius ?? 0;
        let ey = ex;
        let ez = ex;
        if (shape.kind === "box") {
            const h = shape.halfExtents ?? [0, 0, 0];
            const q = shape.rotation ?? [0, 0, 0, 1];
            const [x, y, z, w] = q;
            const m00 = 1 - 2 * (y * y + z * z);
            const m01 = 2 * (x * y - z * w);
            const m02 = 2 * (x * z + y * w);
            const m10 = 2 * (x * y + z * w);
            const m11 = 1 - 2 * (x * x + z * z);
            const m12 = 2 * (y * z - x * w);
            const m20 = 2 * (x * z - y * w);
            const m21 = 2 * (y * z + x * w);
            const m22 = 1 - 2 * (x * x + y * y);
            ex = Math.abs(m00) * h[0] + Math.abs(m01) * h[1] + Math.abs(m02) * h[2];
            ey = Math.abs(m10) * h[0] + Math.abs(m11) * h[1] + Math.abs(m12) * h[2];
            ez = Math.abs(m20) * h[0] + Math.abs(m21) * h[1] + Math.abs(m22) * h[2];
        } else if (shape.kind === "capsule" || shape.kind === "cylinder") {
            const a = shape.pointA ?? shape.centre;
            const b = shape.pointB ?? shape.centre;
            sx = (a[0] + b[0]) * 0.5;
            sy = (a[1] + b[1]) * 0.5;
            sz = (a[2] + b[2]) * 0.5;
            ex += Math.abs(b[0] - a[0]) * 0.5;
            ey += Math.abs(b[1] - a[1]) * 0.5;
            ez += Math.abs(b[2] - a[2]) * 0.5;
        }
        const dx = Math.max(Math.abs(center[0] - sx) - ex, 0);
        const dy = Math.max(Math.abs(center[1] - sy) - ey, 0);
        const dz = Math.max(Math.abs(center[2] - sz) - ez, 0);
        return dx * dx + dy * dy + dz * dz <= radius * radius;
    };
    const createNeighborhoodSdf = (
        center: readonly [number, number, number],
        excludePlacementId: string | null
    ): { spec: SceneSdfSpec; buffer: GPUBuffer | null; count: number } => {
        if (collisionNeighborhoodRadius <= 0) {
            return { spec: groundSdf, buffer: null, count: 0 };
        }
        const primitives = shipCollisionPlacements
            .filter((placement) => placement.id !== excludePlacementId)
            .flatMap((placement) => worldShapesForMatrix(placement.node.worldMatrix, placement.shapes))
            .filter((shape) => shapeIntersectsSphere(shape, center, collisionNeighborhoodRadius))
            .map(worldShapeToFluidPrimitive);
        if (primitives.length === 0) {
            return { spec: groundSdf, buffer: null, count: 0 };
        }
        const packed = new Float32Array(primBufferBytes(primitives.length) / 4);
        packPrimitives(packed, primitives);
        const buffer = device.createBuffer({
            label: "liq-neighborhood-collision",
            size: packed.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, packed);
        return {
            spec: {
                struct: groundSdf.struct,
                sdf: `${PRIMITIVES_WGSL}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    return min(pt.y - sceneSdfParams.ground.x, primitivesSdf(pt, dt));
}`,
                buffer: groundSdfBuffer,
                sdfGrid: buffer,
            },
            buffer,
            count: primitives.length,
        };
    };

    function createConfiguredSim(count: number, radius: number, boundsMin: [number, number, number], boundsMax: [number, number, number], positions?: Float32Array): FluidSim {
        const dx = simulationCellSize(radius);
        const phys = physValues[currentMethod]!;
        if (currentMethod === "PBF") {
            return createPbfSim(engine, {
                count,
                particleRadius: radius,
                ...(positions ? { initialPositions: positions } : {}),
                smoothingRadius: Math.max(radius * 4.0, 0.3),
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
        }
        if (currentMethod === "FLIP") {
            return createFlipSim(engine, {
                count,
                particleRadius: radius,
                ...(positions ? { initialPositions: positions } : {}),
                boundsMin,
                boundsMax,
                dx,
                markersPerCell: flipMarkersPerCell,
                groundY: GROUND_Y,
                gravity: phys.gravity,
                flipRatio: phys.flipRatio,
                pressureIterations: phys.pressureIterations,
                pressureRelaxation: phys.pressureRelaxation,
                pressureSolver: (phys.pressureSolver ?? 0) >= 0.5 ? "multigrid" : "jacobi",
                multigridCycles: phys.multigridCycles,
                velocityDamping: phys.velocityDamping,
                kinematicViscosity: phys.kinematicViscosity,
                viscosityIterations: phys.viscosityIterations,
                surfaceTension: phys.surfaceTension,
                liquidSdf: (phys.liquidSdf ?? 0) >= 0.5,
                ghostFluid: (phys.ghostFluid ?? 0) >= 0.5,
                fractionalSolids: (phys.fractionalSolids ?? 0) >= 0.5,
                movingSolidBoundaries: (phys.movingSolidBoundaries ?? 0) >= 0.5,
                restitution: phys.restitution,
                minSubsteps: phys.minSubsteps,
                maxSubsteps: phys.maxSubsteps,
                cflNumber: phys.cflNumber,
                ...(phys.maxSubDtMs ? { maxSubDt: phys.maxSubDtMs / 1000 } : {}),
            });
        }
        if (currentMethod === "PB-MPM") {
            return createPbMpmSim(engine, {
                count,
                particleRadius: radius,
                ...(positions ? { initialPositions: positions } : {}),
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
        }
        return createMlsMpmSim(engine, {
            count,
            particleRadius: radius,
            ...(positions ? { initialPositions: positions } : {}),
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
            activeBlocks: mpmActiveBlocks,
            pagedGrid: mpmPagedGrid,
            pagedGridMaxPages: mpmPagedGridMaxPages,
            fusedBlockDiscovery: mpmFusedBlockDiscovery,
            onPagedGridOverflow: (requiredPages, capacity) => {
                const message = `Page capacity exceeded: ${requiredPages.toLocaleString()} required, ${capacity.toLocaleString()} allocated.`;
                controls.setPagedGridStatus(message, true);
                console.error(`[AquanovaFluidSim MLS-MPM] ${message}`);
            },
        });
    }

    // ── Per-instance sim construction ──
    function buildInstanceSim(
        inst: Instance,
        positions: Float32Array,
        count: number,
        radius: number,
        wMin: readonly [number, number, number],
        wMax: readonly [number, number, number]
    ): void {
        const dx = simulationCellSize(radius);
        const meshCenter: [number, number, number] = [(wMin[0] + wMax[0]) * 0.5, (wMin[1] + wMax[1]) * 0.5, (wMin[2] + wMax[2]) * 0.5];
        const gridCenter: [number, number, number] = [meshCenter[0] + gridPosition[0], meshCenter[1] + gridPosition[1], meshCenter[2] + gridPosition[2]];
        const { min: boundsMin, max: boundsMax } = simulationBounds(gridCenter);
        // eslint-disable-next-line no-console
        console.log(
            `[aquanova-fluid-sim] sim grid: ${gridSize[0].toFixed(2)}×${gridSize[1].toFixed(2)}×${gridSize[2].toFixed(2)} m ` +
                `(${Math.ceil(gridSize[0] / dx)}×${Math.ceil(gridSize[1] / dx)}×${Math.ceil(gridSize[2] / dx)} cells @ dx ${dx.toFixed(3)}), particles: ${count}`
        );
        const sim = createConfiguredSim(count, radius, boundsMin, boundsMax, positions);
        const collision = createNeighborhoodSdf(meshCenter, inst.collisionOwnerId);
        inst.collisionBuffer?.destroy();
        inst.collisionBuffer = collision.buffer;
        sim.setSceneSdf(collision.spec);
        sim.setProfiler?.(profiler);
        applyFoamToSim(sim);
        inst.sim = sim;
        inst.count = count;
        inst.radius = radius;
        virtualSim.particleRadius = radius;
        virtualSim.surfaceSizeScale = sim.surfaceSizeScale ?? 1;
        canvas.dataset.lastLiquefiedMeshCenter = meshCenter.join(",");
        canvas.dataset.lastSimulationGridPosition = gridCenter.join(",");
        canvas.dataset.lastCollisionPrimitiveCount = String(collision.count);
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

    // Merge each mesh's current rendered pose into one world-space triangle soup.
    // Also emits per-vertex UVs and a texture index when the instance is textured.
    function sampleGeometry(inst: Instance): SampleGeom | null {
        const wantUv = inst.diffuseTexs.length > 0;
        let totalV = 0;
        let totalI = 0;
        let allHaveUv = wantUv;
        const parts: { pos: Float32Array; idx: Uint32Array; uv: Float32Array | null; ti: number }[] = [];
        for (let mi = 0; mi < inst.meshes.length; mi++) {
            const m = inst.meshes[mi]!;
            const geometry = getMeshPoseGeometry(m);
            if (!geometry) continue;
            if (!geometry.uvs) allHaveUv = false;
            parts.push({ pos: geometry.positions, idx: geometry.indices, uv: geometry.uvs, ti: inst.meshTexIndex[mi] ?? -1 });
            totalV += geometry.positions.length / 3;
            totalI += geometry.indices.length;
        }
        if (totalV === 0) return null;
        const positions = new Float32Array(totalV * 3);
        const indices = new Uint32Array(totalI);
        const uvs = allHaveUv ? new Float32Array(totalV * 2) : null;
        const texIndices = wantUv ? new Uint32Array(totalV) : null;
        let vb = 0;
        let ic = 0;
        for (const p of parts) {
            const n = p.pos.length / 3;
            const ti = p.ti >= 0 ? p.ti >>> 0 : NO_TEX;
            positions.set(p.pos, vb * 3);
            for (let v = 0; v < n; v++) {
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
        inst.collisionBuffer?.destroy();
        inst.collisionBuffer = null;
        inst.colorBuffer?.destroy();
        inst.colorBuffer = null;
        inst.phase = "gone";
        inst.count = 0;
    }

    function manualParticleCapacity(worldFlow = flowInWorldSpace(activeFlow)): number {
        const enabled = worldFlow.emitters.filter((emitter) => emitter.enabled);
        const particleVolume = Math.max((radiusValue * 2) ** 3, 1e-6);
        const initialDemand = enabled
            .filter((emitter) => emitter.behavior === "initial")
            .reduce((sum, emitter) => sum + Math.ceil(fluidShapeVolume(emitter.shape, emitter.transform) / particleVolume), 0);
        const inflowDemand = enabled.some((emitter) => emitter.behavior === "inflow") ? Math.max(initialDemand * 2, 20000) : 0;
        const automatic = Math.max(initialDemand + inflowDemand, initialDemand || 20000);
        const requested = currentMethod === "FLIP" ? (flipParticleCapacityRequest ?? automatic) : automatic;
        return Math.max(1, Math.min(MAX_TOTAL, Math.max(initialDemand, requested)));
    }

    function projectedFlipParticleUsage(): { active: number; capacity: number; gpuBytes: number } {
        const worldFlow = flowInWorldSpace(activeFlow);
        const capacity = manualParticleCapacity(worldFlow);
        const initial = worldFlow.emitters.filter((emitter) => emitter.enabled && emitter.behavior === "initial");
        const active =
            worldFlow.initialEmittersFillCapacity || initial.some((emitter) => emitter.sampling !== "volume")
                ? capacity
                : Math.min(
                      capacity,
                      flipParticleCountForVolume(
                          initial.reduce((sum, emitter) => sum + fluidShapeVolume(emitter.shape, emitter.transform), 0),
                          simulationCellSize(),
                          flipMarkersPerCell
                      )
                  );
        const dx = simulationCellSize();
        const gridDim = gridSize.map((size) => Math.max(1, Math.ceil(size / dx))) as [number, number, number];
        const flipPhysics = physValues["FLIP"]!;
        const gpuBytes = estimateFlipGpuBytes(capacity, gridDim, (flipPhysics.pressureSolver ?? 0) >= 0.5 ? "multigrid" : "jacobi", {
            pressureDiagnostics: (flipPhysics.pressureDiagnostics ?? 0) >= 0.5 || (flipPhysics.pressureTolerance ?? 0) > 0,
            liquidSdf: (flipPhysics.liquidSdf ?? 0) >= 0.5,
            fractionalSolids: (flipPhysics.fractionalSolids ?? 0) >= 0.5,
            reseedParticles: (flipPhysics.reseedParticles ?? 0) >= 0.5,
        });
        return { active, capacity, gpuBytes };
    }

    function refreshManualParticleUsage(): void {
        if (simulationType !== "fluid") {
            return;
        }
        const active = manualRun?.sim.activeCount ?? manualRun?.sim.count ?? 0;
        const total = manualRun?.sim.count ?? 0;
        const gpuBytes = manualRun?.sim.gpuBytes ?? 0;
        const projection = currentMethod === "FLIP" ? projectedFlipParticleUsage() : null;
        const pending =
            projection !== null &&
            (!manualRun ||
                manualRun.method !== currentMethod ||
                manualRun.configurationSignature !== manualConfigurationSignature() ||
                projection.capacity !== total ||
                projection.gpuBytes !== gpuBytes);
        controls.setParticleUsage(
            active,
            total,
            gpuBytes,
            pending ? projection!.active : undefined,
            pending ? projection!.capacity : undefined,
            pending ? projection!.gpuBytes : undefined
        );
        canvas.dataset.activeParticleCount = String(active);
        canvas.dataset.simulationGpuBytes = String(gpuBytes);
        canvas.dataset.gridRestartPending = String(pending);
        if (projection) {
            canvas.dataset.restartParticleCount = String(projection.active);
            canvas.dataset.restartParticleCapacity = String(projection.capacity);
            canvas.dataset.restartSimulationGpuBytes = String(projection.gpuBytes);
        } else {
            delete canvas.dataset.restartParticleCount;
            delete canvas.dataset.restartParticleCapacity;
            delete canvas.dataset.restartSimulationGpuBytes;
        }
    }

    function startManualSimulation(): void {
        const worldFlow = flowInWorldSpace(activeFlow);
        const enabled = worldFlow.emitters.filter((emitter) => emitter.enabled);
        if (enabled.length === 0) {
            status.textContent = "Enable at least one emitter before starting.";
            return;
        }
        manualRun?.sim.dispose();
        manualRun?.collisionBuffer?.destroy();
        manualRun?.colorBuffer.destroy();
        manualRun = null;
        const center: [number, number, number] = [...gridPosition];
        const { min: boundsMin, max: boundsMax } = simulationBounds();
        const capacity = manualParticleCapacity(worldFlow);
        const sim = createConfiguredSim(capacity, radiusValue, boundsMin, boundsMax);
        sim.setFlow(worldFlow);
        const collision = createNeighborhoodSdf(center, null);
        sim.setSceneSdf(collision.spec);
        sim.setProfiler?.(profiler);
        applyFoamToSim(sim);
        sim.reset();
        const colorBuffer = device.createBuffer({
            label: "liq-manual-color",
            size: capacity * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        manualRun = {
            sim,
            collisionBuffer: collision.buffer,
            colorBuffer,
            center,
            method: currentMethod,
            configurationSignature: manualConfigurationSignature(),
        };
        fillManualColor(manualRun);
        virtualSim.particleRadius = radiusValue;
        virtualSim.surfaceSizeScale = sim.surfaceSizeScale ?? 1;
        canvas.dataset.lastCollisionPrimitiveCount = String(collision.count);
        setStatus();
        refreshManualParticleUsage();
    }

    // ── Shot lifecycle ───────────────────────────────────────────────────────
    // Volume-sampling runs in a worker (see particle-fill-worker.ts) so the ~1 s dense sample
    // doesn't freeze the frame at the shot. requestSample() fires the job (foe stays solid,
    // marked `sampling`); applySample() builds the sim + starts the dissolve when it returns.
    let liveShots = 0;
    let sampleSeq = 0;
    const pendingSamples = new Map<number, { inst: Instance; hit: [number, number, number] }>();

    const resumeCancelledAnimation = (inst: Instance): void => {
        if (inst.resumeAnimationOnCancel && inst.animation) {
            playAnimation(inst.animation);
        }
        inst.resumeAnimationOnCancel = false;
    };

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
            resumeCancelledAnimation(entry.inst);
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
                    console.warn("[aquanova-fluid-sim] sample worker error", e.message);
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
        const animation = inst.animation;
        inst.resumeAnimationOnCancel = !!animation?.isPlaying;
        if (animation?.isPlaying) {
            pauseAnimation(animation);
        }
        startWriggle(inst); // pain shake begins immediately on click, before the async sample finishes
        setStatus();
        const geom = sampleGeometry(inst);
        if (!geom) {
            pendingSamples.delete(id);
            inst.sampling = false;
            leaveShot(inst);
            stopWriggle(inst);
            resumeCancelledAnimation(inst);
            setStatus();
            return false;
        }
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
                resumeCancelledAnimation(inst);
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
            resumeCancelledAnimation(inst);
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
        invalidatePhaseTaskBundles();
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
        inst.resumeAnimationOnCancel = false;
        startImpulse(inst);
        invalidatePhaseTaskBundles();
        setStatus();
    }

    function restart(): void {
        endManualForceDrag();
        pendingSamples.clear();
        manualRun?.sim.dispose();
        manualRun?.collisionBuffer?.destroy();
        manualRun?.colorBuffer.destroy();
        manualRun = null;
        for (const inst of instances) {
            inst.sim?.setForceField(null);
            inst.sim?.dispose();
            inst.sim = null;
            inst.collisionBuffer?.destroy();
            inst.collisionBuffer = null;
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
            inst.resumeAnimationOnCancel = false;
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
        invalidatePhaseTaskBundles();
        setStatus();
        refreshManualParticleUsage();
    }

    function runPrimarySimulationAction(): void {
        if (simulationType === "fluid") {
            if (manualRun) {
                restart();
            } else {
                startManualSimulation();
            }
            return;
        }
        restart();
    }

    // ── Control panel ────────────────────────────────────────────────────────
    const PANEL_STYLE =
        "position:fixed;top:12px;left:12px;z-index:10;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;" +
        "font-size:0.8rem;color:#dfe6ee;background:rgba(12,16,24,0.82);padding:12px 14px;border-radius:10px;" +
        "width:280px;height:calc(100vh - 48px);min-width:232px;min-height:240px;max-width:calc(100vw - 24px);max-height:calc(100vh - 24px);" +
        "box-shadow:0 6px 22px rgba(0,0,0,0.4);";

    const title = document.createElement("div");
    title.textContent = "AquanovaFluidSim";
    title.style.cssText = "font-weight:700;font-size:0.95rem;margin-bottom:8px;";
    const helperText = document.querySelector<HTMLElement>(".hint");
    const updateHelperText = (): void => {
        if (!helperText) return;
        helperText.textContent =
            simulationType === "mesh"
                ? "WASD + Space/C: fly · RMB-drag: look · Shift+LMB: toggle mesh gizmo · LMB: liquefy · P: pause · R: restart"
                : "WASD + Space/C: fly · RMB-drag: look · Shift+LMB: toggle mesh gizmo · Shift+RMB-drag: push fluid · P: pause · R: stop/start";
    };
    updateHelperText();

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
    const simulationTypeSelect = document.createElement("select");
    styleSelect(simulationTypeSelect);
    for (const [value, label] of [
        ["mesh", "Mesh liquefaction"],
        ["fluid", "Fluid"],
    ] as [SimulationType, string][]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        simulationTypeSelect.append(option);
    }
    simulationTypeSelect.value = simulationType;
    let setSimulationType = (value: SimulationType): void => {
        simulationType = value;
        simulationTypeSelect.value = value;
        updateHelperText();
    };
    simulationTypeSelect.onchange = () => setSimulationType(simulationTypeSelect.value as SimulationType);
    const simulationTypeRow = labelledRow("Simulation type", simulationTypeSelect);

    const aquanovaRenderingInput = document.createElement("input");
    aquanovaRenderingInput.type = "checkbox";
    aquanovaRenderingInput.id = "aquanova-rendering";
    aquanovaRenderingInput.checked = renderLikeAquanova;
    aquanovaRenderingInput.style.cssText = "margin-right:6px;vertical-align:middle;";
    aquanovaRenderingInput.onchange = () => {
        const url = new URL(location.href);
        if (aquanovaRenderingInput.checked) {
            url.searchParams.set("aquanovaRendering", "1");
        } else {
            url.searchParams.delete("aquanovaRendering");
        }
        location.assign(url);
    };
    const aquanovaRenderingRow = document.createElement("label");
    aquanovaRenderingRow.style.cssText = "display:block;margin-bottom:8px;color:#b6c4d6;cursor:pointer;";
    aquanovaRenderingRow.title =
        "Reload with Aquanova's authored clustered lights and one fixed local cubemap per ship mesh. The full ship remains rendered every frame; portal culling and cubemap blending stay disabled.";
    aquanovaRenderingRow.append(aquanovaRenderingInput, document.createTextNode("Aquanova rendering"));

    const fluidSimSelect = document.createElement("select");
    fluidSimSelect.id = "aquanova-fluid-sim-preset";
    styleSelect(fluidSimSelect);
    const fluidSimRow = labelledRow("Simulation", fluidSimSelect);
    const fluidSimNameInput = document.createElement("input");
    fluidSimNameInput.type = "text";
    fluidSimNameInput.placeholder = "new-simulation";
    fluidSimNameInput.style.cssText = "box-sizing:border-box;width:100%;padding:4px;background:#1a2230;color:#dfe6ee;border:1px solid #33415a;border-radius:4px;margin-bottom:5px;";
    const fluidSimButtons = document.createElement("div");
    fluidSimButtons.style.cssText = "display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:8px;";
    const makePresetButton = (text: string): HTMLButtonElement => {
        const button = document.createElement("button");
        button.textContent = text;
        button.style.cssText = "padding:5px 3px;border:0;border-radius:5px;cursor:pointer;background:#33465c;color:#fff;font-weight:600;";
        return button;
    };
    const fluidSimCreateButton = makePresetButton("Create");
    const fluidSimUpdateButton = makePresetButton("Update");
    const fluidSimDeleteButton = makePresetButton("Delete");
    fluidSimDeleteButton.style.background = "#693747";
    fluidSimButtons.append(fluidSimCreateButton, fluidSimUpdateButton, fluidSimDeleteButton);
    const fluidSimStatus = document.createElement("div");
    fluidSimStatus.style.cssText = "min-height:1.1em;margin-bottom:8px;color:#8fa4bc;";
    const fluidSimAuthoring = document.createElement("div");
    fluidSimAuthoring.append(fluidSimNameInput, fluidSimButtons, fluidSimStatus);

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
        controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
        refreshFlipParticleCapacity();
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

    const collisionRadiusInput = document.createElement("input");
    collisionRadiusInput.type = "range";
    collisionRadiusInput.min = "0";
    collisionRadiusInput.max = "40";
    collisionRadiusInput.step = "0.5";
    collisionRadiusInput.value = String(collisionNeighborhoodRadius);
    collisionRadiusInput.style.cssText = "width:100%;";
    const collisionRadiusValue = document.createElement("span");
    collisionRadiusValue.style.cssText = "color:#9fb4cc;float:right;";
    const showCollisionRadius = (): void => {
        collisionRadiusValue.textContent = collisionNeighborhoodRadius > 0 ? `${collisionNeighborhoodRadius.toFixed(1)} m` : "off";
    };
    showCollisionRadius();
    collisionRadiusInput.oninput = () => {
        collisionNeighborhoodRadius = parseFloat(collisionRadiusInput.value);
        showCollisionRadius();
    };
    const collisionRadiusLabel = document.createElement("div");
    collisionRadiusLabel.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
    collisionRadiusLabel.append(document.createTextNode("Collision neighborhood"), collisionRadiusValue);
    collisionRadiusLabel.title = "Radius of the sphere used to select authored ship collision primitives whenever a simulation starts.";
    const collisionRadiusRow = document.createElement("div");
    collisionRadiusRow.style.cssText = "margin-bottom:8px;";
    collisionRadiusRow.append(collisionRadiusLabel, collisionRadiusInput);

    const forceStrengthInput = document.createElement("input");
    forceStrengthInput.type = "range";
    forceStrengthInput.id = "liq-force-strength";
    forceStrengthInput.min = "0";
    forceStrengthInput.max = "2";
    forceStrengthInput.step = "0.05";
    forceStrengthInput.value = String(manualForceStrength);
    forceStrengthInput.style.cssText = "width:100%;";
    const forceStrengthValue = document.createElement("span");
    forceStrengthValue.style.cssText = "color:#9fb4cc;float:right;";
    const showForceStrength = (): void => {
        forceStrengthValue.textContent = `${manualForceStrength.toFixed(2)}×`;
    };
    showForceStrength();
    forceStrengthInput.oninput = () => {
        manualForceStrength = parseFloat(forceStrengthInput.value);
        showForceStrength();
    };
    const forceStrengthLabel = document.createElement("div");
    forceStrengthLabel.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
    forceStrengthLabel.append(document.createTextNode("Force strength"), forceStrengthValue);
    forceStrengthLabel.title = "Multiplier applied to mouse speed while Shift+RMB dragging.";
    const forceStrengthRow = document.createElement("div");
    forceStrengthRow.style.cssText = "margin-bottom:8px;";
    forceStrengthRow.append(forceStrengthLabel, forceStrengthInput);

    const forceRadiusInput = document.createElement("input");
    forceRadiusInput.type = "range";
    forceRadiusInput.id = "liq-force-radius";
    forceRadiusInput.min = "0.1";
    forceRadiusInput.max = "20";
    forceRadiusInput.step = "0.1";
    forceRadiusInput.value = String(manualForceRadius);
    forceRadiusInput.style.cssText = "width:100%;";
    const forceRadiusValue = document.createElement("span");
    forceRadiusValue.style.cssText = "color:#9fb4cc;float:right;";
    const showForceRadius = (): void => {
        forceRadiusValue.textContent = `${manualForceRadius.toFixed(1)} m`;
    };
    showForceRadius();
    forceRadiusInput.oninput = () => {
        manualForceRadius = parseFloat(forceRadiusInput.value);
        showForceRadius();
    };
    const forceRadiusLabel = document.createElement("div");
    forceRadiusLabel.style.cssText = "margin-bottom:3px;color:#b6c4d6;";
    forceRadiusLabel.append(document.createTextNode("Force influence radius"), forceRadiusValue);
    forceRadiusLabel.title = "World-space radius around the cursor ray affected by Shift+RMB dragging.";
    const forceRadiusRow = document.createElement("div");
    forceRadiusRow.style.cssText = "margin-bottom:8px;";
    forceRadiusRow.append(forceRadiusLabel, forceRadiusInput);

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
    restartBtn.textContent = "Reset liquefaction";
    restartBtn.style.cssText = "width:100%;padding:6px;margin-top:4px;border:0;border-radius:6px;cursor:pointer;background:#4a5568;color:#fff;font-weight:600;";
    restartBtn.onclick = () => runPrimarySimulationAction();

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
        status.textContent =
            simulationType === "mesh"
                ? `${solid} solid · ${active} liquefying${fillNote} · click a ship mesh`
                : manualRun
                  ? "fluid simulation running"
                  : "fluid simulation stopped";
        restartBtn.textContent = simulationType === "fluid" ? (manualRun ? "Stop simulation" : "Start simulation") : "Reset liquefaction";
        restartBtn.style.background = simulationType === "fluid" && !manualRun ? "#245a79" : "#4a5568";
        canvas.dataset.solid = String(solid);
        canvas.dataset.active = String(active);
        canvas.dataset.shells = String(shells);
        canvas.dataset.volumes = String(live.length - shells);
        canvas.dataset.manualRunning = manualRun ? "true" : "false";
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
        controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
        refreshFlipParticleCapacity();
        refreshPhysicsParamVisibility();
    }

    type FlowPoint = { x: number; y: number; z: number };
    type FlowWireframeSegment = readonly [FlowPoint, FlowPoint];
    const flowPoint = (x: number, y: number, z: number): FlowPoint => ({ x, y, z });
    const boxWireframeSegments = (min: readonly [number, number, number], max: readonly [number, number, number]): FlowWireframeSegment[] => {
        const corners = [
            flowPoint(min[0], min[1], min[2]),
            flowPoint(max[0], min[1], min[2]),
            flowPoint(max[0], min[1], max[2]),
            flowPoint(min[0], min[1], max[2]),
            flowPoint(min[0], max[1], min[2]),
            flowPoint(max[0], max[1], min[2]),
            flowPoint(max[0], max[1], max[2]),
            flowPoint(min[0], max[1], max[2]),
        ];
        return [
            [corners[0]!, corners[1]!],
            [corners[1]!, corners[2]!],
            [corners[2]!, corners[3]!],
            [corners[3]!, corners[0]!],
            [corners[4]!, corners[5]!],
            [corners[5]!, corners[6]!],
            [corners[6]!, corners[7]!],
            [corners[7]!, corners[4]!],
            [corners[0]!, corners[4]!],
            [corners[1]!, corners[5]!],
            [corners[2]!, corners[6]!],
            [corners[3]!, corners[7]!],
        ];
    };
    const gridBoundsSegments = (size: readonly [number, number, number]): FlowWireframeSegment[] =>
        boxWireframeSegments([-size[0] * 0.5, -size[1] * 0.5, -size[2] * 0.5], [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5]);
    const gridBoundsWireframe = createLineSystem(engine, {
        name: "aquanova-fluid-sim-grid-bounds",
        lines: gridBoundsSegments(gridSize),
        material: createLineMaterial({
            name: "aquanova-fluid-sim-grid-bounds-material",
            color: { r: 0.45, g: 1, b: 0.35, a: 0.85 },
            useVertexAlpha: true,
            depthWrite: false,
            depthCompare: "always",
        }),
    });
    gridBoundsWireframe.pickable = false;
    gridBoundsWireframe.renderOrder = 9_999;
    addToScene(scene, gridBoundsWireframe);

    const appendPolyline = (segments: FlowWireframeSegment[], points: readonly FlowPoint[], closed = true): void => {
        for (let i = 1; i < points.length; i++) {
            segments.push([points[i - 1]!, points[i]!]);
        }
        if (closed && points.length > 2) {
            segments.push([points[points.length - 1]!, points[0]!]);
        }
    };
    const appendRing = (segments: FlowWireframeSegment[], plane: "xy" | "xz" | "yz", radius: number, center: FlowPoint = flowPoint(0, 0, 0), steps = 32): void => {
        const points: FlowPoint[] = [];
        for (let i = 0; i < steps; i++) {
            const angle = (i / steps) * Math.PI * 2;
            const a = Math.cos(angle) * radius;
            const b = Math.sin(angle) * radius;
            points.push(
                plane === "xy"
                    ? flowPoint(center.x + a, center.y + b, center.z)
                    : plane === "xz"
                      ? flowPoint(center.x + a, center.y, center.z + b)
                      : flowPoint(center.x, center.y + a, center.z + b)
            );
        }
        appendPolyline(segments, points);
    };
    const flowShapeWireframe = (shape: FluidShape): FlowWireframeSegment[] => {
        const segments: FlowWireframeSegment[] = [];
        if (shape.type === "box") {
            const [hx, hy, hz] = shape.size.map((value) => value * 0.5);
            segments.push(...boxWireframeSegments([-hx!, -hy!, -hz!], [hx!, hy!, hz!]));
        } else if (shape.type === "sphere") {
            appendRing(segments, "xy", shape.radius);
            appendRing(segments, "xz", shape.radius);
            appendRing(segments, "yz", shape.radius);
        } else if (shape.type === "cylinder") {
            const halfHeight = shape.height * 0.5;
            for (const y of [-halfHeight, halfHeight]) {
                appendRing(segments, "xz", shape.radius, flowPoint(0, y, 0));
                if ((shape.innerRadius ?? 0) > 0) {
                    appendRing(segments, "xz", shape.innerRadius!, flowPoint(0, y, 0));
                }
            }
            const radii = shape.innerRadius && shape.innerRadius > 0 ? [shape.radius, shape.innerRadius] : [shape.radius];
            for (const radius of radii) {
                for (let i = 0; i < 8; i++) {
                    const angle = (i / 8) * Math.PI * 2;
                    const x = Math.cos(angle) * radius;
                    const z = Math.sin(angle) * radius;
                    segments.push([flowPoint(x, -halfHeight, z), flowPoint(x, halfHeight, z)]);
                }
            }
        } else if (shape.type === "cone") {
            const halfHeight = shape.height * 0.5;
            if (shape.bottomRadius > 0) {
                appendRing(segments, "xz", shape.bottomRadius, flowPoint(0, -halfHeight, 0));
            }
            if (shape.topRadius > 0) {
                appendRing(segments, "xz", shape.topRadius, flowPoint(0, halfHeight, 0));
            }
            for (let i = 0; i < 8; i++) {
                const angle = (i / 8) * Math.PI * 2;
                const x = Math.cos(angle);
                const z = Math.sin(angle);
                segments.push([flowPoint(x * shape.bottomRadius, -halfHeight, z * shape.bottomRadius), flowPoint(x * shape.topRadius, halfHeight, z * shape.topRadius)]);
            }
        } else if (shape.type === "capsule") {
            const bodyHalfHeight = Math.max(0, shape.height * 0.5 - shape.radius);
            appendRing(segments, "xz", shape.radius, flowPoint(0, -bodyHalfHeight, 0));
            appendRing(segments, "xz", shape.radius, flowPoint(0, bodyHalfHeight, 0));
            const meridian: FlowPoint[] = [];
            for (let i = 0; i <= 16; i++) {
                const angle = Math.PI - (i / 16) * Math.PI;
                meridian.push(flowPoint(Math.cos(angle) * shape.radius, bodyHalfHeight + Math.sin(angle) * shape.radius, 0));
            }
            meridian.push(flowPoint(shape.radius, -bodyHalfHeight, 0));
            for (let i = 1; i <= 16; i++) {
                const angle = -(i / 16) * Math.PI;
                meridian.push(flowPoint(Math.cos(angle) * shape.radius, -bodyHalfHeight + Math.sin(angle) * shape.radius, 0));
            }
            appendPolyline(segments, meridian);
            appendPolyline(
                segments,
                meridian.map((point) => flowPoint(0, point.y, point.x))
            );
        } else {
            const halfThickness = shape.thickness * 0.5;
            for (let i = 0; i < shape.points.length; i++) {
                const point = shape.points[i]!;
                const next = shape.points[(i + 1) % shape.points.length]!;
                const bottom = flowPoint(point[0], -halfThickness, point[1]);
                const top = flowPoint(point[0], halfThickness, point[1]);
                segments.push([bottom, flowPoint(next[0], -halfThickness, next[1])], [top, flowPoint(next[0], halfThickness, next[1])], [bottom, top]);
            }
        }
        return segments;
    };
    const maxFlowWireframeSegments = MAX_FLUID_POLYGON_POINTS * 3;
    const paddedFlowWireframe = (shape?: FluidShape): FlowWireframeSegment[] => {
        const segments = shape ? flowShapeWireframe(shape).slice(0, maxFlowWireframeSegments) : [];
        while (segments.length < maxFlowWireframeSegments) {
            segments.push([flowPoint(0, 0, 0), flowPoint(0, 0, 0)]);
        }
        return segments;
    };
    const createFlowWireframe = (name: string, color: { r: number; g: number; b: number; a: number }): Mesh => {
        const mesh = createLineSystem(engine, {
            name,
            lines: paddedFlowWireframe(),
            material: createLineMaterial({ name: `${name}-material`, color, useVertexAlpha: true, depthWrite: false, depthCompare: "always" }),
        });
        mesh.pickable = false;
        mesh.renderOrder = 10_000;
        addToScene(scene, mesh);
        setMeshVisible(mesh, false);
        return mesh;
    };
    const emitterFlowWireframe = createFlowWireframe("aquanova-fluid-sim-emitter-wireframe", { r: 0.1, g: 0.9, b: 1, a: 0.9 });
    const sinkFlowWireframe = createFlowWireframe("aquanova-fluid-sim-sink-wireframe", { r: 1, g: 0.55, b: 0.1, a: 0.9 });
    let showEmitterWireframe = false;
    let showSinkWireframe = false;
    let flowGizmoOwner: FluidFlowObjectKind | null = null;
    let flowEditor: FluidFlowEditor | null = null;

    const gridGizmoLayer = createUtilityLayer(engine, scene);
    const flowPositionGizmo = createPositionGizmo(engine, gridGizmoLayer, { planarEnabled: true });
    const flowRotationGizmo = createRotationGizmo(engine, gridGizmoLayer);
    const flowScaleGizmo = createScaleGizmo(engine, gridGizmoLayer);
    const gridPositionGizmo = createPositionGizmo(engine, gridGizmoLayer, { planarEnabled: true });
    const gridScaleGizmo = createScaleGizmo(engine, gridGizmoLayer);
    const meshPositionGizmo = createPositionGizmo(engine, gridGizmoLayer, { planarEnabled: true });
    const meshRotationGizmo = createRotationGizmo(engine, gridGizmoLayer);
    setPositionGizmoLocalCoordinates(flowPositionGizmo, false);
    setRotationGizmoLocalCoordinates(flowRotationGizmo, true);
    setScaleGizmoLocalCoordinates(flowScaleGizmo, true);
    setPositionGizmoLocalCoordinates(gridPositionGizmo, false);
    setScaleGizmoLocalCoordinates(gridScaleGizmo, true);
    setPositionGizmoLocalCoordinates(meshPositionGizmo, false);
    setRotationGizmoLocalCoordinates(meshRotationGizmo, true);
    const flowPositionSubGizmos = [
        flowPositionGizmo.xGizmo,
        flowPositionGizmo.yGizmo,
        flowPositionGizmo.zGizmo,
        ...(flowPositionGizmo.xPlaneGizmo ? [flowPositionGizmo.xPlaneGizmo] : []),
        ...(flowPositionGizmo.yPlaneGizmo ? [flowPositionGizmo.yPlaneGizmo] : []),
        ...(flowPositionGizmo.zPlaneGizmo ? [flowPositionGizmo.zPlaneGizmo] : []),
    ];
    const flowRotationSubGizmos = [flowRotationGizmo.xGizmo, flowRotationGizmo.yGizmo, flowRotationGizmo.zGizmo];
    const flowScaleSubGizmos = [flowScaleGizmo.xGizmo, flowScaleGizmo.yGizmo, flowScaleGizmo.zGizmo, flowScaleGizmo.uniformScaleGizmo];
    const gridPositionSubGizmos = [
        gridPositionGizmo.xGizmo,
        gridPositionGizmo.yGizmo,
        gridPositionGizmo.zGizmo,
        ...(gridPositionGizmo.xPlaneGizmo ? [gridPositionGizmo.xPlaneGizmo] : []),
        ...(gridPositionGizmo.yPlaneGizmo ? [gridPositionGizmo.yPlaneGizmo] : []),
        ...(gridPositionGizmo.zPlaneGizmo ? [gridPositionGizmo.zPlaneGizmo] : []),
    ];
    const gridScaleSubGizmos = [gridScaleGizmo.xGizmo, gridScaleGizmo.yGizmo, gridScaleGizmo.zGizmo, gridScaleGizmo.uniformScaleGizmo];
    const meshPositionSubGizmos = [
        meshPositionGizmo.xGizmo,
        meshPositionGizmo.yGizmo,
        meshPositionGizmo.zGizmo,
        ...(meshPositionGizmo.xPlaneGizmo ? [meshPositionGizmo.xPlaneGizmo] : []),
        ...(meshPositionGizmo.yPlaneGizmo ? [meshPositionGizmo.yPlaneGizmo] : []),
        ...(meshPositionGizmo.zPlaneGizmo ? [meshPositionGizmo.zPlaneGizmo] : []),
    ];
    const meshRotationSubGizmos = [meshRotationGizmo.xGizmo, meshRotationGizmo.yGizmo, meshRotationGizmo.zGizmo];
    const setGizmoMeshesVisible = (visible: boolean, gizmos: ReadonlyArray<{ _visibleMeshes: Mesh[] }>): void => {
        for (const gizmo of gizmos) {
            for (const mesh of gizmo._visibleMeshes) {
                setMeshVisible(mesh, visible);
            }
        }
    };
    const selectedFlowObject = (kind: FluidFlowObjectKind) => flowEditor?.getSelected(kind);
    let meshGizmoTarget: Mesh | null = null;
    const meshGizmoId = (mesh: Mesh | null): string | null => (mesh ? (instanceOf(mesh)?.key ?? mesh.name) : null);
    const setMeshGizmoTarget = (target: Mesh | null): void => {
        meshGizmoTarget = target;
        attachPositionGizmoToNode(meshPositionGizmo, target);
        attachRotationGizmoToNode(meshRotationGizmo, target);
        setGizmoMeshesVisible(!!target, meshPositionSubGizmos);
        setGizmoMeshesVisible(!!target, meshRotationSubGizmos);
        canvas.dataset.meshGizmo = meshGizmoId(target) ?? "";
    };
    setMeshGizmoTarget(null);
    const syncFlowWireframe = (kind: FluidFlowObjectKind): void => {
        const isEmitter = kind === "emitter";
        const mesh = isEmitter ? emitterFlowWireframe : sinkFlowWireframe;
        const visible = isEmitter ? showEmitterWireframe : showSinkWireframe;
        const object = selectedFlowObject(kind);
        if (!object) {
            setMeshVisible(mesh, false);
            return;
        }
        updateLineSystem(engine, mesh, { lines: paddedFlowWireframe(object.shape) });
        const { position, rotation, scale } = object.transform;
        mesh.position.set(position[0] + gridPosition[0], position[1] + gridPosition[1], position[2] + gridPosition[2]);
        mesh.rotationQuaternion.set(rotation[0], rotation[1], rotation[2], rotation[3]);
        mesh.scaling.set(scale[0], scale[1], scale[2]);
        setMeshVisible(mesh, visible || flowGizmoOwner === kind);
    };
    const syncFlowTransformFromGizmo = (): void => {
        if (!flowGizmoOwner) {
            return;
        }
        const object = selectedFlowObject(flowGizmoOwner);
        if (!object) {
            return;
        }
        const mesh = flowGizmoOwner === "emitter" ? emitterFlowWireframe : sinkFlowWireframe;
        object.transform.position = [mesh.position.x - gridPosition[0], mesh.position.y - gridPosition[1], mesh.position.z - gridPosition[2]];
        object.transform.rotation = [mesh.rotationQuaternion.x, mesh.rotationQuaternion.y, mesh.rotationQuaternion.z, mesh.rotationQuaternion.w];
        object.transform.scale = [mesh.scaling.x, mesh.scaling.y, mesh.scaling.z];
        manualRun?.sim.setFlow(flowInWorldSpace(activeFlow));
    };
    const finishFlowGizmoDrag = (): void => {
        syncFlowTransformFromGizmo();
        flowEditor?.refresh();
    };
    for (const gizmo of flowPositionSubGizmos) {
        gizmo.onPositionChanged.add(syncFlowTransformFromGizmo);
        gizmo.drag.onDragEnd.add(finishFlowGizmoDrag);
    }
    for (const gizmo of flowRotationSubGizmos) {
        gizmo.onRotationChanged.add(syncFlowTransformFromGizmo);
        gizmo.drag.onDragEnd.add(finishFlowGizmoDrag);
    }
    for (const gizmo of flowScaleSubGizmos) {
        gizmo.onScaleChanged.add(syncFlowTransformFromGizmo);
        gizmo.drag.onDragEnd.add(finishFlowGizmoDrag);
    }
    const syncFlowGizmo = (): void => {
        const owner = flowGizmoOwner;
        const object = owner ? selectedFlowObject(owner) : undefined;
        const target = owner && object ? (owner === "emitter" ? emitterFlowWireframe : sinkFlowWireframe) : null;
        attachPositionGizmoToNode(flowPositionGizmo, target);
        attachRotationGizmoToNode(flowRotationGizmo, target);
        attachScaleGizmoToNode(flowScaleGizmo, target);
        setGizmoMeshesVisible(!!target, flowPositionSubGizmos);
        setGizmoMeshesVisible(!!target, flowRotationSubGizmos);
        setGizmoMeshesVisible(!!target, flowScaleSubGizmos);
        if (owner) {
            syncFlowWireframe(owner);
        }
    };
    const syncGridBoundsWireframe = (): void => {
        updateLineSystem(engine, gridBoundsWireframe, { lines: gridBoundsSegments(gridSize) });
        gridBoundsWireframe.position.set(gridPosition[0], gridPosition[1], gridPosition[2]);
        gridBoundsWireframe.scaling.set(1, 1, 1);
        setMeshVisible(gridBoundsWireframe, showGridBounds || showGridGizmo);
        syncFlowWireframe("emitter");
        syncFlowWireframe("sink");
        syncFlowGizmo();
        canvas.dataset.gridPosition = gridPosition.join(",");
        canvas.dataset.gridSize = gridSize.join(",");
        canvas.dataset.showGridBounds = String(showGridBounds);
    };
    const syncGridGizmo = (): void => {
        attachPositionGizmoToNode(gridPositionGizmo, showGridGizmo ? gridBoundsWireframe : null);
        attachScaleGizmoToNode(gridScaleGizmo, showGridGizmo ? gridBoundsWireframe : null);
        setGizmoMeshesVisible(showGridGizmo, gridPositionSubGizmos);
        setGizmoMeshesVisible(showGridGizmo, gridScaleSubGizmos);
    };
    function setGridSettings(position: readonly number[], size: readonly number[]): void {
        const changed = position.some((value, axis) => value !== gridPosition[axis]) || size.some((value, axis) => value !== gridSize[axis]);
        for (let axis = 0; axis < 3; axis++) {
            gridPosition[axis] = position[axis]!;
            gridSize[axis] = Math.max(0.1, size[axis]!);
        }
        canvas.dataset.gridPosition = gridPosition.join(",");
        canvas.dataset.gridSize = gridSize.join(",");
        controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
        syncGridBoundsWireframe();
        if (changed) {
            if (simulationType === "mesh") {
                restart();
            } else {
                refreshManualParticleUsage();
            }
        }
    }
    function setGridGizmoVisible(visible: boolean): void {
        showGridGizmo = visible;
        canvas.dataset.gridGizmo = String(visible);
        syncGridBoundsWireframe();
        syncGridGizmo();
    }
    const finishGridGizmoDrag = (): void => {
        const position: [number, number, number] = [
            Math.round(gridBoundsWireframe.position.x * 10) / 10,
            Math.round(gridBoundsWireframe.position.y * 10) / 10,
            Math.round(gridBoundsWireframe.position.z * 10) / 10,
        ];
        const size: [number, number, number] = [
            Math.max(0.1, Math.round(Math.abs(gridSize[0] * gridBoundsWireframe.scaling.x) * 10) / 10),
            Math.max(0.1, Math.round(Math.abs(gridSize[1] * gridBoundsWireframe.scaling.y) * 10) / 10),
            Math.max(0.1, Math.round(Math.abs(gridSize[2] * gridBoundsWireframe.scaling.z) * 10) / 10),
        ];
        setGridSettings(position, size);
        syncGridGizmo();
    };
    for (const gizmo of gridPositionSubGizmos) {
        gizmo.drag.onDragEnd.add(finishGridGizmoDrag);
    }
    for (const gizmo of gridScaleSubGizmos) {
        gizmo.drag.onDragEnd.add(finishGridGizmoDrag);
    }
    syncGridBoundsWireframe();
    syncGridGizmo();

    const controls = createFluidControlsPanel({
        hideParticles: true,
        hideMethod: false,
        hideContainerToggle: true,
        hideFoam: false,
        hidePhysics: false,
        hidePhysScale: true,
        hideDebug: false,
        hideGpuTiming: false,
        showGridControls: true,
        panelStyle: PANEL_STYLE,
        schemas: LIQ_SCHEMAS,
        methods: Object.keys(LIQ_SCHEMAS),
        particleCounts: [],
        flipParticleCapacityMax: MAX_TOTAL,
        showActiveBlocks: true,
        initial: {
            method: currentMethod,
            material: currentMaterial,
            count: 0,
            physScale: 1,
            color: DEF_COLOR,
            absorption: DEF_ABSORPTION,
            size: DEF_SIZE,
            refraction: DEF_REFRACTION,
            specular: DEF_SPECULAR,
            reflectionExposure: 1,
            reflectionContrast: 1.1,
            reflectivity: 0.02,
            depthBlur: DEF_DEPTH_BLUR,
            depthBlurThreshold: DEF_DEPTH_BLUR_THRESHOLD,
            thicknessBlur: DEF_THICKNESS_BLUR,
            half: DEF_HALF,
            thicknessDownscale: DEF_THICKNESS_DOWNSCALE,
            surfaceFilter: DEF_SURFACE_FILTER,
            narrowDelta: DEF_NARROW_DELTA,
            narrowMu: DEF_NARROW_MU,
            anisotropic: false,
            anisoSurfScale: 0.5,
            activeBlocks: mpmActiveBlocks,
            pagedGrid: mpmPagedGrid,
            pagedGridMaxPages: mpmPagedGridMaxPages,
            fusedBlockDiscovery: mpmFusedBlockDiscovery,
            gridResolution: flipGridResolution,
            markersPerCell: flipMarkersPerCell,
            gridPosition: [...gridPosition],
            gridSize: [...gridSize],
            cellSize: simulationCellSize(),
            showGridBounds,
            renderMode: "surface",
            debug: "none",
            showContainer: true,
            foam: {
                enabled: false,
                activeParticles: true,
                generateSpray: true,
                generateFoam: true,
                generateBubbles: true,
                kTa: 40,
                kWc: 40,
                kTurb: 0,
                energySpeedMin: Math.sqrt(0.5),
                energySpeedMax: Math.sqrt(40),
                curvatureMin: 0.05,
                curvatureMax: 1.5,
                turbulenceMin: 0.1,
                turbulenceMax: 2.5,
                foamLayerDepth: 0,
                sprayDrag: 0,
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
        gpu: { stages: ["Simulation", "Surface", "Foam generate", "Foam render"], supported: profiler !== null },
        on: {
            onMethod: (m) => switchMethod(m),
            onMaterial: (material) => {
                currentMaterial = material;
                for (const inst of instances) inst.sim?.setMaterial?.(currentMaterial);
                if (manualRun?.method === currentMethod) manualRun.sim.setMaterial?.(currentMaterial);
                refreshPhysicsParamVisibility();
            },
            onRenderMode: () => {}, // surface only in the multi-target demo
            onDebug: (mode) => surfaceTask.setDebug(mode),
            onColor: (rgb) => {
                surfaceTask.setFluidColor(rgb);
                if (manualRun) {
                    fillManualColor(manualRun);
                }
            },
            onAbsorption: (v) => surfaceTask.setAbsorption(v),
            onParticleSize: (s) => surfaceTask.setSizeScale(s),
            onRefraction: (v) => surfaceTask.setRefractionStrength(v),
            onSpecular: (v) => surfaceTask.setSpecularPower(v),
            onReflection: (exposure, contrast) => surfaceTask.setEnvReflection(exposure, contrast),
            onReflectivity: (v) => surfaceTask.setFresnelF0(v),
            onDepthBlur: (size, threshold) => surfaceTask.setDepthBlur(size, threshold),
            onThicknessBlur: (v) => surfaceTask.setThicknessBlur(v),
            onHalf: (on) => surfaceTask.setHalfRender(on),
            onSurfaceFilter: (m) => surfaceTask.setSurfaceFilter(m),
            onNarrowRange: (delta, mu) => surfaceTask.setNarrowRange(delta, mu),
            onAnisotropic: (enabled) => surfaceTask.setAnisotropic(enabled),
            onAnisotropySurfScale: (share) => surfaceTask.setAnisotropySurfScale(share),
            onThicknessDownscale: (v) => surfaceTask.setThicknessDownscale(v),
            onActiveBlocks: (enabled) => {
                mpmActiveBlocks = enabled;
            },
            onPagedGrid: (enabled) => {
                mpmPagedGrid = enabled;
                controls.setPagedGridStatus("");
            },
            onPagedGridMaxPages: (pages) => {
                mpmPagedGridMaxPages = Math.min(maxPagedGridPages, pages);
                controls.setPagedGridMaxPages(mpmPagedGridMaxPages);
                controls.setPagedGridStatus("");
            },
            onFusedBlockDiscovery: (enabled) => {
                mpmFusedBlockDiscovery = enabled;
            },
            onGridResolution: (resolution) => {
                flipGridResolution = Math.max(16, Math.min(400, Math.round(resolution)));
                controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
                refreshManualParticleUsage();
            },
            onMarkersPerCell: (markersPerCell) => {
                flipMarkersPerCell = Math.max(1, Math.min(64, Math.round(markersPerCell)));
                refreshManualParticleUsage();
            },
            onFlipParticleCapacity: (capacity) => {
                flipParticleCapacityRequest = Math.max(1, Math.min(MAX_TOTAL, Math.round(capacity)));
                refreshManualParticleUsage();
            },
            onGridSettings: (position, size) => setGridSettings(position, size),
            onGridGizmo: (visible) => setGridGizmoVisible(visible),
            onShowGridBounds: (visible) => {
                showGridBounds = visible;
                canvas.dataset.showGridBounds = String(visible);
                syncGridBoundsWireframe();
            },
            // Physics sliders apply LIVE to every running sim AND seed the next shot.
            onPhysicsParam: (key, value) => {
                physValues[currentMethod]![key] = value;
                for (const inst of instances) inst.sim?.setParam(key, value);
                if (manualRun?.method === currentMethod) manualRun.sim.setParam(key, value);
                refreshManualParticleUsage();
            },
            onReset: () => restart(),
            onFoamEnable: () => pushFoam(),
            onFoamKinds: () => pushFoam(),
            onFoamKta: () => pushFoam(),
            onFoamKwc: () => pushFoam(),
            onFoamAdvanced: () => pushFoam(),
            onFoamLifetime: () => pushFoam(),
            onFoamBuoyancy: () => pushFoam(),
            onFoamDrag: () => pushFoam(),
            onFoamPool: () => pushFoam(),
            onFoamThresholds: (t0, t1) => foamTask.setThresholds(t0, t1),
            onFoamSubsurface: (v) => foamTask.setSubsurfaceStrength(v),
            onFoamSurfaceFiltering: (enabled) => foamTask.setSurfaceFiltering(enabled),
            onFoamSubColor: (rgb) => foamTask.setSubsurfaceColor(rgb),
            onFoamSize: (v) => foamTask.setSizeScale(v),
            onFoamBlur: (v) => foamTask.setBlurRadius(v),
            onFoamLight: (v) => foamTask.setLightIntensity(v),
            onFoamAmbient: (v) => foamTask.setAmbient(v),
            onFoamAO: (v) => foamTask.setAOStrength(v),
            onFoamNormal: (v) => foamTask.setNormalStrength(v),
            onFoamDebugByKind: (on) => foamTask.setDebugByKind(on),
            onFoamDebugTexture: (v) => foamTask.setDebugTexture(v),
        },
    });
    refreshFlipParticleCapacity = () => controls.setFlipParticleCapacity(manualParticleCapacity());
    refreshFlipParticleCapacity();

    // ── Export / import fluid settings ───────────────────────────────────────
    // Reuse the fluid demo's grouped JSON shape (fluid/preset-io.js): serialise the live
    // physics + surface-render params so a setting authored here can be dropped straight
    // into another demo (e.g. aquanova's FLUID_SETTING) or re-imported below to iterate.
    const IO_BTN_STYLE = "flex:1;padding:6px;border:0;border-radius:6px;cursor:pointer;background:#4a5568;color:#fff;font-weight:600;";
    function liqPairState(): PairState {
        const v = controls.getValues();
        return {
            schema: { ...physValues[currentMethod]! },
            demoParams: {
                particleRadius: radiusValue,
                useMeshColors: useMeshColors ? 1 : 0,
                collisionNeighborhoodRadius,
                forceStrength: manualForceStrength,
                forceInfluenceRadius: manualForceRadius,
            },
            color: v.color,
            half: v.half,
            thicknessDownscale: v.thicknessDownscale,
            absorption: v.absorption,
            size: v.size,
            physScale: 1,
            count: currentMethod === "FLIP" ? (flipParticleCapacityRequest ?? 0) : 0,
            gridResolution: currentMethod === "FLIP" ? flipGridResolution : undefined,
            markersPerCell: currentMethod === "FLIP" ? flipMarkersPerCell : undefined,
            material: currentMethod === "PB-MPM" ? v.material : undefined,
            freeCamera: {
                position: [cam.position.x, cam.position.y, cam.position.z],
                target: [cam.target.x, cam.target.y, cam.target.z],
            },
            renderMode: v.renderMode,
            refraction: v.refraction,
            specular: v.specular,
            reflectionExposure: v.reflectionExposure,
            reflectionContrast: v.reflectionContrast,
            reflectivity: v.reflectivity,
            depthBlur: v.depthBlur,
            depthBlurThreshold: v.depthBlurThreshold,
            thicknessBlur: v.thicknessBlur,
            surfaceFilter: v.surfaceFilter,
            narrowDelta: v.narrowDelta,
            narrowMu: v.narrowMu,
            anisotropic: v.anisotropic,
            anisoSurfScale: v.anisoSurfScale,
            activeBlocks: currentMethod === "MLS-MPM" ? v.activeBlocks : undefined,
            pagedGrid: currentMethod === "MLS-MPM" ? v.pagedGrid : undefined,
            pagedGridMaxPages: currentMethod === "MLS-MPM" ? v.pagedGridMaxPages : undefined,
            fusedBlockDiscovery: currentMethod === "MLS-MPM" ? v.fusedBlockDiscovery : undefined,
            grid: { position: [...gridPosition], size: [...gridSize] },
            showGridBounds: v.showGridBounds,
            foam: v.foam,
            emitters: structuredClone(activeFlow.emitters),
            sinks: structuredClone(activeFlow.sinks),
            initialEmittersFillCapacity: activeFlow.initialEmittersFillCapacity,
            demoState: { simulationType, samplingMode: modeValue, fillStrategy },
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
            controls.setMaterial(currentMaterial);
            for (const inst of instances) inst.sim?.setMaterial?.(currentMaterial);
        }
        if (currentMethod === "FLIP") {
            flipGridResolution = Math.max(16, Math.min(400, Math.round(p.gridResolution ?? flipGridResolution)));
            flipMarkersPerCell = Math.max(1, Math.min(64, Math.round(p.markersPerCell ?? flipMarkersPerCell)));
            flipParticleCapacityRequest = typeof p.count === "number" && p.count > 0 ? Math.min(MAX_TOTAL, Math.round(p.count)) : null;
            controls.setGridResolution(flipGridResolution);
            controls.setMarkersPerCell(flipMarkersPerCell);
        }
        if (currentMethod === "MLS-MPM") {
            mpmPagedGrid = p.pagedGrid ?? mpmPagedGrid;
            mpmActiveBlocks = mpmPagedGrid || (p.activeBlocks ?? mpmActiveBlocks);
            mpmPagedGridMaxPages = Math.min(maxPagedGridPages, p.pagedGridMaxPages ?? mpmPagedGridMaxPages);
            mpmFusedBlockDiscovery = p.fusedBlockDiscovery ?? mpmFusedBlockDiscovery;
            controls.setActiveBlocks(mpmActiveBlocks);
            controls.setPagedGrid(mpmPagedGrid);
            controls.setPagedGridMaxPages(mpmPagedGridMaxPages);
            controls.setFusedBlockDiscovery(mpmFusedBlockDiscovery);
            controls.setPagedGridStatus("");
        }
        if (p.emitters || p.sinks) {
            activeFlow = {
                emitters: structuredClone(p.emitters ?? []),
                sinks: structuredClone(p.sinks ?? []),
                ...(p.initialEmittersFillCapacity !== undefined ? { initialEmittersFillCapacity: p.initialEmittersFillCapacity } : {}),
            };
            refreshFlowControls();
        }
        const importedType = p.demoState?.simulationType;
        setSimulationType(importedType === "fluid" || importedType === "mesh" ? importedType : (p.emitters?.length ?? 0) > 0 || (p.sinks?.length ?? 0) > 0 ? "fluid" : "mesh");
        const importedSamplingMode = p.demoState?.samplingMode;
        if (importedSamplingMode === "dense" || importedSamplingMode === "regular" || importedSamplingMode === "kugelstadt2021") {
            modeValue = importedSamplingMode;
            modeSelect.value = importedSamplingMode;
        }
        const importedFillStrategy = p.demoState?.fillStrategy;
        if (importedFillStrategy === "auto" || importedFillStrategy === "volume" || importedFillStrategy === "surface") {
            fillStrategy = importedFillStrategy;
            fillSelect.value = importedFillStrategy;
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
        const collisionRadius = p.demoParams?.collisionNeighborhoodRadius;
        if (typeof collisionRadius === "number" && isFinite(collisionRadius) && collisionRadius >= 0) {
            collisionNeighborhoodRadius = collisionRadius;
            collisionRadiusInput.value = String(collisionNeighborhoodRadius);
            showCollisionRadius();
        }
        const forceStrength = p.demoParams?.forceStrength;
        manualForceStrength = typeof forceStrength === "number" && isFinite(forceStrength) ? Math.max(0, Math.min(2, forceStrength)) : DEFAULT_FORCE_STRENGTH;
        forceStrengthInput.value = String(manualForceStrength);
        showForceStrength();
        const forceInfluenceRadius = p.demoParams?.forceInfluenceRadius;
        manualForceRadius = typeof forceInfluenceRadius === "number" && isFinite(forceInfluenceRadius) ? Math.max(0.1, Math.min(20, forceInfluenceRadius)) : DEFAULT_FORCE_RADIUS;
        forceRadiusInput.value = String(manualForceRadius);
        showForceRadius();
        // Surface render — these setters fire their host callbacks, applying LIVE to surfaceTask.
        if (p.color !== undefined) controls.setColor(p.color);
        if (p.absorption !== undefined) controls.setAbsorption(p.absorption);
        if (p.size !== undefined) controls.setParticleSize(p.size);
        if (p.renderMode !== undefined) controls.setRenderMode(p.renderMode === "spheres");
        if (p.refraction !== undefined) controls.setRefraction(p.refraction);
        if (p.specular !== undefined) controls.setSpecular(p.specular);
        if (p.reflectionExposure !== undefined || p.reflectionContrast !== undefined) {
            const current = controls.getValues();
            controls.setReflection(p.reflectionExposure ?? current.reflectionExposure, p.reflectionContrast ?? current.reflectionContrast);
        }
        if (p.reflectivity !== undefined) controls.setReflectivity(p.reflectivity);
        if (p.depthBlur !== undefined || p.depthBlurThreshold !== undefined) {
            const current = controls.getValues();
            controls.setDepthBlur(p.depthBlur ?? current.depthBlur, p.depthBlurThreshold ?? current.depthBlurThreshold);
        }
        if (p.thicknessBlur !== undefined) controls.setThicknessBlur(p.thicknessBlur);
        if (p.half !== undefined) controls.setHalf(p.half);
        if (p.surfaceFilter !== undefined) controls.setSurfaceFilter(p.surfaceFilter);
        if (p.narrowDelta !== undefined || p.narrowMu !== undefined) {
            const current = controls.getValues();
            controls.setNarrowRange(p.narrowDelta ?? current.narrowDelta, p.narrowMu ?? current.narrowMu);
        }
        if (p.anisotropic !== undefined) controls.setAnisotropic(p.anisotropic);
        if (p.anisoSurfScale !== undefined) controls.setAnisotropySurfScale(p.anisoSurfScale);
        if (p.thicknessDownscale !== undefined) controls.setThicknessDownscale(p.thicknessDownscale);
        if (p.foam !== undefined) {
            controls.setFoam({ ...controls.getValues().foam, ...p.foam });
            pushFoam();
        }
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
        // Prefer the shared grid payload, while retaining the legacy Liquefactor/Aquanova block.
        const nextGridPosition: [number, number, number] = p.grid ? [...p.grid.position] : [...gridPosition];
        const nextGridSize: [number, number, number] = p.grid ? [...p.grid.size] : [...gridSize];
        const g = j.grid;
        if (g) {
            const axes = [g.x, g.y, g.z];
            for (let axis = 0; axis < 3; axis++) {
                const v = axes[axis];
                if (typeof v === "number" && isFinite(v) && v > 0) nextGridSize[axis] = v;
            }
            if (Array.isArray(g.position) && g.position.length === 3 && g.position.every((value) => typeof value === "number" && isFinite(value))) {
                nextGridPosition.splice(0, 3, ...g.position);
            }
        }
        setGridSettings(nextGridPosition, nextGridSize);
        if (p.showGridBounds !== undefined) {
            showGridBounds = p.showGridBounds;
            controls.setShowGridBounds(showGridBounds);
            syncGridBoundsWireframe();
        }
        if (p.freeCamera && p.freeCamera.position.every(isFinite) && p.freeCamera.target.every(isFinite)) {
            applyFreeCameraPose(p.freeCamera.position, p.freeCamera.target);
        }
        refreshFlipParticleCapacity();
        refreshPhysicsParamVisibility();
    }
    const buildCurrentPreset = (): FluidExportJson => {
        const data = exportJsonFromPairState("aquanova-fluid-sim", currentMethod, liqPairState());
        data.impulse = { intensity: impulseIntensity, direction: [impulseDir[0], impulseDir[1], impulseDir[2]], radius: impulseRadius };
        data.grid = { x: gridSize[0], y: gridSize[1], z: gridSize[2], position: [...gridPosition] };
        return data;
    };
    const exportBtn = document.createElement("button");
    exportBtn.textContent = "Export";
    exportBtn.style.cssText = IO_BTN_STYLE;
    exportBtn.onclick = () => {
        const data = buildCurrentPreset();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `aquanova-fluid-sim-${currentMethod}.json`;
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
                const preset = JSON.parse(String(reader.result)) as FluidExportJson;
                restart();
                applyImportedPreset(preset);
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

    const normalizeFluidSimName = (name: string): string => name.replace(/\.json$/i, "").toLowerCase();
    const isValidFluidSimName = (name: string): boolean => /^[a-z0-9][a-z0-9_-]{0,63}$/i.test(name);
    let fluidSimNames: string[] = [];
    const setFluidSimNames = (names: readonly string[], selected?: string): void => {
        fluidSimNames = Array.from(new Set(names.map(normalizeFluidSimName).filter(isValidFluidSimName)));
        const next = selected && fluidSimNames.includes(selected) ? selected : (fluidSimNames[0] ?? "");
        fluidSimSelect.replaceChildren();
        for (const name of fluidSimNames) {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            fluidSimSelect.append(option);
        }
        fluidSimSelect.value = next;
        fluidSimNameInput.value = next;
        fluidSimSelect.disabled = fluidSimNames.length === 0;
        fluidSimUpdateButton.disabled = fluidSimNames.length === 0;
        fluidSimDeleteButton.disabled = fluidSimNames.length === 0;
        canvas.dataset.fluidSimNames = fluidSimNames.join(",");
    };
    const readApiResponse = async (response: Response): Promise<{ name?: string; names?: string[] }> => {
        const payload = (await response.json().catch(() => ({}))) as { error?: string; name?: string; names?: string[] };
        if (!response.ok) {
            throw new Error(payload.error ?? (response.status === 404 ? "Preset authoring requires the local lab development server." : `HTTP ${response.status}`));
        }
        return payload;
    };
    const loadFluidSim = async (name: string): Promise<void> => {
        const normalized = normalizeFluidSimName(name);
        if (!isValidFluidSimName(normalized)) throw new Error(`Invalid simulation name "${name}".`);
        restart();
        const response = await fetch(`/aquanova/fluidSim/${encodeURIComponent(normalized)}.json`, { cache: "no-store" });
        if (!response.ok) throw new Error(`Could not load "${normalized}" (HTTP ${response.status}).`);
        applyImportedPreset((await response.json()) as FluidExportJson);
        fluidSimSelect.value = normalized;
        fluidSimNameInput.value = normalized;
        fluidSimStatus.textContent = `Loaded ${normalized}`;
        canvas.dataset.fluidSim = normalized;
    };
    const saveFluidSim = async (name: string, create: boolean): Promise<void> => {
        const normalized = normalizeFluidSimName(name.trim());
        if (!isValidFluidSimName(normalized)) {
            throw new Error("Use 1-64 letters, numbers, hyphens, or underscores.");
        }
        if (create && fluidSimNames.includes(normalized)) {
            throw new Error(`"${normalized}" already exists; use Update instead.`);
        }
        const result = await readApiResponse(
            await fetch(`/lab-api/aquanova-fluid-sims/${encodeURIComponent(normalized)}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(buildCurrentPreset()),
            })
        );
        setFluidSimNames(result.names ?? [...fluidSimNames, normalized], normalized);
        if (shipManifest) shipManifest.fluidSim = [...fluidSimNames];
        fluidSimStatus.textContent = `${create ? "Created" : "Updated"} ${normalized}`;
        canvas.dataset.fluidSim = normalized;
    };
    const showFluidSimError = (error: unknown): void => {
        fluidSimStatus.textContent = error instanceof Error ? error.message : String(error);
        fluidSimStatus.style.color = "#f29b9b";
    };
    const runFluidSimAction = async (action: () => Promise<void>): Promise<void> => {
        fluidSimStatus.style.color = "#8fa4bc";
        try {
            await action();
        } catch (error) {
            showFluidSimError(error);
        }
    };
    fluidSimSelect.onchange = () => {
        void runFluidSimAction(() => loadFluidSim(fluidSimSelect.value));
    };
    fluidSimCreateButton.onclick = () => {
        void runFluidSimAction(() => saveFluidSim(fluidSimNameInput.value, true));
    };
    fluidSimUpdateButton.onclick = () => {
        void runFluidSimAction(() => saveFluidSim(fluidSimSelect.value, false));
    };
    fluidSimDeleteButton.onclick = () => {
        const name = fluidSimSelect.value;
        if (!name || !window.confirm(`Delete fluid simulation "${name}"?`)) return;
        void runFluidSimAction(async () => {
            const result = await readApiResponse(await fetch(`/lab-api/aquanova-fluid-sims/${encodeURIComponent(name)}`, { method: "DELETE" }));
            const nextNames = result.names ?? fluidSimNames.filter((entry) => entry !== name);
            setFluidSimNames(nextNames);
            if (shipManifest) shipManifest.fluidSim = [...fluidSimNames];
            fluidSimStatus.textContent = `Deleted ${name}`;
            if (fluidSimSelect.value) await loadFluidSim(fluidSimSelect.value);
        });
    };

    const initialFoam = controls.getValues().foam;
    foamTask.setThresholds(initialFoam.softness, initialFoam.density);
    foamTask.setSubsurfaceStrength(initialFoam.subsurfaceStrength);
    foamTask.setSubsurfaceColor(hexToRgb(initialFoam.subsurfaceColor));
    foamTask.setSizeScale(initialFoam.size);
    foamTask.setBlurRadius(initialFoam.blurRadius);
    foamTask.setLightIntensity(initialFoam.lightIntensity);
    foamTask.setAmbient(initialFoam.ambient);
    foamTask.setAOStrength(initialFoam.aoStrength);
    foamTask.setNormalStrength(initialFoam.normalStrength);
    foamTask.setDebugTexture(initialFoam.debugTexture as FoamDebugTexture);
    foamTask.setEnabled(initialFoam.enabled);

    const demoSection = controls.makeSection("Demo", [
        title,
        simulationTypeRow,
        aquanovaRenderingRow,
        fluidSimRow,
        fluidSimAuthoring,
        collisionRadiusRow,
        forceStrengthRow,
        forceRadiusRow,
        restartBtn,
        ioRow,
        status,
        partCount,
    ]);
    const meshLiquefactionSection = controls.makeSection("Mesh liquefaction", [
        labelledRow("Sampling mode", modeSelect),
        radiusRow,
        fillRow,
        impulseRow,
        impulseDirRow,
        impulseRadiusRow,
        meshColorRow,
    ]);
    controls.demoSlot.append(...demoSection, ...meshLiquefactionSection);
    const emitterFlowHost = document.createElement("div");
    const sinkFlowHost = document.createElement("div");
    const emitterSection = controls.makeSection("Emitters", [emitterFlowHost]);
    const sinkSection = controls.makeSection("Sinks", [sinkFlowHost]);
    controls.root.append(...emitterSection, ...sinkSection);
    const setHostSectionVisible = (section: HTMLElement[], visible: boolean): void => {
        for (const element of section) element.style.display = visible ? "" : "none";
    };
    setSimulationType = (value: SimulationType): void => {
        const changed = simulationType !== value;
        simulationType = value;
        simulationTypeSelect.value = value;
        setHostSectionVisible(meshLiquefactionSection, value === "mesh");
        forceStrengthRow.style.display = value === "fluid" ? "" : "none";
        forceRadiusRow.style.display = value === "fluid" ? "" : "none";
        setHostSectionVisible(emitterSection, value === "fluid");
        setHostSectionVisible(sinkSection, value === "fluid");
        controls.setSectionVisible("Foam", value === "fluid");
        updateHelperText();
        canvas.dataset.simulationType = value;
        if (changed) restart();
        setStatus();
    };
    flowEditor = createFluidFlowEditor({
        emittersHost: emitterFlowHost,
        sinksHost: sinkFlowHost,
        flow: activeFlow,
        onChange: (flow) => {
            activeFlow = flow;
            manualRun?.sim.setFlow(flowInWorldSpace(activeFlow));
            refreshFlipParticleCapacity();
            refreshManualParticleUsage();
            canvas.dataset.emitterCount = String(activeFlow.emitters.length);
            canvas.dataset.sinkCount = String(activeFlow.sinks.length);
        },
        onRefresh: () => {
            syncFlowWireframe("emitter");
            syncFlowWireframe("sink");
            syncFlowGizmo();
        },
        visuals: {
            getWireframeVisible: (kind) => (kind === "emitter" ? showEmitterWireframe : showSinkWireframe),
            setWireframeVisible: (kind, visible) => {
                if (kind === "emitter") {
                    showEmitterWireframe = visible;
                } else {
                    showSinkWireframe = visible;
                }
            },
            getGizmoVisible: (kind) => flowGizmoOwner === kind,
            setGizmoVisible: (kind, visible) => {
                flowGizmoOwner = visible ? kind : flowGizmoOwner === kind ? null : flowGizmoOwner;
            },
        },
    });
    refreshFlowControls = () => flowEditor?.setFlow(activeFlow);
    refreshFlowControls();
    setSimulationType(simulationType);
    document.body.append(controls.root);
    await shipManifestReady;
    const loadedShipManifest = shipManifest as ShipManifestData | null;
    setFluidSimNames(loadedShipManifest?.fluidSim ?? []);
    if (fluidSimSelect.value) {
        await runFluidSimAction(() => loadFluidSim(fluidSimSelect.value));
    } else {
        fluidSimStatus.textContent = "No authored fluid simulations.";
    }
    if (controls.gpu) {
        // The demo's own panel is top-left, so pin the GPU pane top-right to avoid overlap.
        controls.gpu.panel.style.left = "auto";
        controls.gpu.panel.style.right = "12px";
        document.body.appendChild(controls.gpu.panel);
    }
    canvas.dataset.timing = profiler ? "on" : "unavailable";
    canvas.dataset.method = currentMethod;
    refreshPhysicsParamVisibility();
    setStatus();

    const pickEditableMeshAt = async (px: number, py: number): Promise<Mesh | null> => {
        const info = await pickAsync(picker, px, py, { filter: (mesh) => editableShipMeshes.has(mesh) && mesh.visible !== false });
        return info.hit && info.pickedMesh ? (info.pickedMesh as Mesh) : null;
    };
    const toggleMeshGizmoAt = async (px: number, py: number): Promise<string | null> => {
        const target = await pickEditableMeshAt(px, py);
        if (!target) return null;
        setMeshGizmoTarget(meshGizmoTarget === target ? null : target);
        return meshGizmoId(meshGizmoTarget);
    };

    // ── Programmatic hooks for headless QA ───────────────────────────────────
    const qa = {
        getInstances: () =>
            instances.map((i) => {
                const w = i.meshes[0]!.worldMatrix as unknown as ArrayLike<number>;
                return { key: i.key, phase: i.phase, count: i.count, sampling: i.sampling, tex: i.diffuseTexs.map((t) => [t.width, t.height]), pos: [w[12]!, w[13]!, w[14]!] };
            }),
        usesWorker: () => workerPool.length > 0,
        workerCount: () => workerPool.length,
        getTotalParticles: () => virtualSim.count,
        paused: () => paused,
        setPaused: (value: boolean) => {
            paused = value;
            canvas.dataset.paused = paused ? "true" : "false";
        },
        getLifecycle: (key: string) => {
            const inst = instances.find((i) => i.key === key);
            return inst
                ? {
                      phase: inst.phase,
                      frontR: inst.liquefyState.frontR,
                      fluidElapsed: inst.fluidElapsed,
                      fadeElapsed: inst.fadeElapsed,
                      impulseRemaining: inst.impulseRemaining,
                  }
                : null;
        },
        getInstancePos: (key: string) => {
            const inst = instances.find((i) => i.key === key);
            return inst ? ([inst.root.position.x, inst.root.position.y, inst.root.position.z] as [number, number, number]) : null;
        },
        restart: () => restart(),
        startManual: () => startManualSimulation(),
        manualRunning: () => manualRun !== null,
        flow: () => structuredClone(activeFlow),
        flowVisualState: () => ({
            emitterWireframe: showEmitterWireframe,
            sinkWireframe: showSinkWireframe,
            gizmoOwner: flowGizmoOwner,
            emitterVisible: emitterFlowWireframe.visible,
            sinkVisible: sinkFlowWireframe.visible,
        }),
        setFlow: (flow: FluidFlowConfig) => {
            activeFlow = structuredClone(flow);
            refreshFlowControls();
        },
        collisionNeighborhoodRadius: () => collisionNeighborhoodRadius,
        setCollisionNeighborhoodRadius: (radius: number) => {
            collisionNeighborhoodRadius = Math.max(0, Math.min(40, radius));
            collisionRadiusInput.value = String(collisionNeighborhoodRadius);
            showCollisionRadius();
        },
        forceSettings: () => ({ strength: manualForceStrength, radius: manualForceRadius }),
        setForceSettings: (strength: number, radius: number) => {
            manualForceStrength = Math.max(0, Math.min(2, strength));
            manualForceRadius = Math.max(0.1, Math.min(20, radius));
            forceStrengthInput.value = String(manualForceStrength);
            forceRadiusInput.value = String(manualForceRadius);
            showForceStrength();
            showForceRadius();
        },
        currentPreset: () => buildCurrentPreset(),
        setFoamEnabled: (enabled: boolean) => {
            controls.setFoam({ ...controls.getValues().foam, enabled });
            pushFoam();
        },
        foamEnabled: () => controls.getValues().foam.enabled,
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
            camYawTarget = camYaw;
            camPitchTarget = camPitch;
            camVelocity.x = 0;
            camVelocity.y = 0;
            camVelocity.z = 0;
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
            specularAA: (instances[0]?.materials[0] as unknown as { enableSpecularAA?: boolean } | undefined)?.enableSpecularAA,
            reflectionRoughness: shipManifest?.environment?.reflectionRoughness,
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
        meshAt: async (px: number, py: number): Promise<string | null> => meshGizmoId(await pickEditableMeshAt(px, py)),
        meshGizmo: (): string | null => meshGizmoId(meshGizmoTarget),
        toggleMeshGizmoAt,
        setImpulseRadius: (r: number) => {
            impulseRadius = r;
            radiusImpInput.value = String(r);
            showImpulseRadius();
        },
        impulseState: (): Record<string, unknown> => ({ intensity: impulseIntensity, direction: [...impulseDir], radius: impulseRadius }),
        setGridSize: (x: number, y: number, z: number) => {
            setGridSettings(gridPosition, [x, y, z]);
        },
        setGridPosition: (x: number, y: number, z: number) => {
            setGridSettings([x, y, z], gridSize);
        },
        gridState: (): Record<string, number[]> => ({ position: [...gridPosition], size: [...gridSize] }),
        simulationType: (): SimulationType => simulationType,
        setSimulationType: (value: SimulationType): void => setSimulationType(value),
        fluidSimNames: (): string[] => [...fluidSimNames],
        exportPreset: (): unknown => buildCurrentPreset(),
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
                const onEnemy = simulationType === "mesh" && !!info.hit && !!info.pickedPoint && !!inst && inst.phase === "solid" && !inst.sampling;
                let dissolveStarted = false;
                if (onEnemy && info.pickedPoint) dissolveStarted = requestSample(inst!, info.pickedPoint);
                return { hit: info.hit, onEnemy, dissolveStarted, key: inst?.key ?? null };
            }),
    };
    (window as unknown as { __aquanovaFluidSim?: unknown }).__aquanovaFluidSim = qa;
    (window as unknown as { __liquefactor?: unknown }).__liquefactor = qa;

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
        if (meshGizmoTarget && group.includes(instanceOf(meshGizmoTarget)!)) {
            setMeshGizmoTarget(null);
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
        if (ev.shiftKey) {
            // A gizmo collider press is handled by the utility-layer dispatcher. Its hover state is
            // synchronous here even though GPU picking on pointer-down is asynchronous.
            if (isGizmoInteracting(canvas)) {
                return;
            }
            void toggleMeshGizmoAt(px, py);
            return;
        }
        if (simulationType !== "mesh") return;
        void pickAsync(picker, px, py, { filter: (m) => instanceOf(m)?.phase === "solid" && !instanceOf(m)?.sampling }).then((info) => {
            if (!info.hit || !info.pickedPoint || !info.pickedMesh) return;
            const inst = instanceOf(info.pickedMesh);
            if (!inst || inst.phase !== "solid" || inst.sampling) return;
            liquefyGroup(inst, info.pickedPoint);
        });
    });

    // ── Per-frame loop ───────────────────────────────────────────────────────
    let fpsAccumMs = 0;
    let fpsFrames = 0;
    onBeforeRender(scene, (deltaMs: number) => {
        updateCamera(deltaMs);
        if (localEnvironmentController) {
            const previousProbe = localEnvironmentController.blendInfo().dominantProbeId;
            const blend = localEnvironmentController.updatePoi([cam.position.x, cam.position.y, cam.position.z]);
            if (blend.dominantProbeId !== previousProbe) {
                const fluidEnvironment = localEnvironmentController.dominantEnvironment();
                if (fluidEnvironment) {
                    surfaceTask.setEnvMap({ view: fluidEnvironment._specularCubeView, sampler: fluidEnvironment._cubeSampler });
                }
            }
        }
        // Open the whole-frame GPU-timing envelope BEFORE any pass is encoded this frame.
        if (profiler) {
            profiler.beginFrame();
            profiler.frameStart(engine._currentEncoder);
        }
        updateAnimationManager(animManager, deltaMs); // advance the model's walk (skeleton pose)
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
        const growDt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 30);
        if (!paused) {
            for (const inst of instances) {
                if (inst.wriggling) applyWriggle(inst); // pain shake — active from click through the dissolve
                if (inst.phase === "dissolving") {
                    inst.liquefyState.frontR = Math.min(inst.liquefyState.frontR + LIQUEFY_SPEED_SHIP * growDt, inst.maxR);
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
            if (manualRun) {
                if (pendingForce) {
                    rayForce.setRay(pendingForce.origin, pendingForce.dir, pendingForce.push, pendingForce.radius, pendingForce.accel);
                    manualRun.sim.setForceField(rayForce.spec);
                    pendingForce = null;
                } else {
                    manualRun.sim.setForceField(null);
                }
                manualRun.sim.step(engine._currentEncoder, dt);
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
        if (manualRun) {
            const n = manualRun.sim.count;
            if (off + n <= MAX_TOTAL) {
                engine._currentEncoder.copyBufferToBuffer(manualRun.sim.positionBuffer, 0, combinedPos, off * 16, n * 16);
                if (useMeshColors) {
                    engine._currentEncoder.copyBufferToBuffer(manualRun.colorBuffer, 0, combinedColor, off * 16, n * 16);
                }
                alphaScratch.fill(1, off, off + n);
                off += n;
            }
        }
        if (off > 0) {
            // Keep the whole active range in sync so a slot reused after a fade isn't left dim.
            device.queue.writeBuffer(combinedAlpha, 0, alphaScratch, 0, off);
            virtualSim.count = off;
        } else {
            virtualSim.count = 0;
        }
        const displayedCount = simulationType === "fluid" ? (manualRun?.sim.activeCount ?? manualRun?.sim.count ?? 0) : off;
        canvas.dataset.particleCount = String(displayedCount);
        partCount.textContent = `${displayedCount.toLocaleString()} particles`;
        refreshManualParticleUsage();

        // GPU timing + memory read-outs on a ~2 Hz cadence (same as the FPS counter).
        fpsAccumMs += deltaMs;
        fpsFrames++;
        if (fpsAccumMs >= 500) {
            const gpu = controls.gpu;
            if (gpu) {
                gpu.fpsLabel.textContent = paused ? "paused" : `${Math.round((fpsFrames * 1000) / fpsAccumMs)}`;
                gpu.refreshTiming(profiler ? profiler.results() : null);
                let simBytes = 0;
                for (const inst of instances) {
                    if (inst.sim) simBytes += inst.sim.gpuBytes;
                    simBytes += inst.collisionBuffer?.size ?? 0;
                }
                if (manualRun) simBytes += manualRun.sim.gpuBytes + (manualRun.collisionBuffer?.size ?? 0) + manualRun.colorBuffer.size;
                gpu.refreshMemory(simBytes, engine.canvas.width, engine.canvas.height);
            }
            const foamPools = runningSims()
                .map((sim) => sim.diffuse)
                .filter((pool) => !!pool);
            const foamCounts = foamPools.reduce(
                (sum, pool) => {
                    const counts = pool.counts;
                    sum.total += counts?.total ?? 0;
                    sum.spray += counts?.spray ?? 0;
                    sum.foam += counts?.foam ?? 0;
                    sum.bubble += counts?.bubble ?? 0;
                    sum.capacity += pool.capacity;
                    return sum;
                },
                { total: 0, spray: 0, foam: 0, bubble: 0, capacity: 0 }
            );
            controls.setFoamParticleCounts(foamPools.length ? foamCounts : undefined, controls.getValues().foam.enabled, foamCounts.capacity);
            fpsAccumMs = 0;
            fpsFrames = 0;
        }
    });

    await envReady;
    // Load the full Aquanova ship before scene registration so the liquefy plugin materializes for
    // every ship mesh.
    await loadShipFoes();
    // The overlay is an explicit demo-owned list. It contains only liquefiable ship meshes, while
    // task-local visibility gating decides which dissolving meshes draw on a given frame.
    for (const instance of instances) {
        for (const mesh of instance.meshes) {
            foeTask.addMesh(mesh);
        }
    }
    invalidatePhaseTaskBundles();
    await registerScene(scene);
    await registerUtilityLayer(gridGizmoLayer);
    await startEngine(engine);

    canvas.dataset.ready = "true";
}

main().catch((err) => {
    // eslint-disable-next-line no-console
    console.error(err);
    const c = document.getElementById("renderCanvas") as HTMLCanvasElement | null;
    if (c) c.dataset.error = String(err);
});
