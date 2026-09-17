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
    addMeshToTask,
    addAnimationGroups,
    addTask,
    addToScene,
    attachPositionGizmoToNode,
    attachRotationGizmoToNode,
    attachScaleGizmoToNode,
    createAnimationManager,
    createBox,
    createCapsule,
    createCylinder,
    createFreeCamera,
    createHavokWorld,
    createLineMaterial,
    createLineSystem,
    createMeshFromData,
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
    createShaderMaterial,
    createSphere,
    createStandardMaterial,
    createTransformNode,
    createUtilityLayer,
    enableMaterialPlugins,
    getPhysicsBodyLinearVelocity,
    getProjectionMatrix,
    getViewMatrix,
    getViewProjectionMatrix,
    fluidSimulationCellSize,
    fluidSimulationParticleCapacity,
    fluidSimulationParticleRadius,
    fluidMaximumPageCapacity,
    formatFluidPageDiagnostics,
    normalizeFluidFlipDiscretization,
    resolveFluidSimulationConfig,
    fluidMeshSamplingErrorInfo,
    sampleFluidMeshParticles,
    CURRENT_FLUID_SIMULATION_SEMANTICS,
    aquanovaCombinedParticleCapacity,
    isPbrMaterial,
    isGizmoInteracting,
    isGizmoPickPending,
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
    mat4Invert,
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
import type {
    AnimationGroup,
    EnvironmentTextures,
    FluidFlowConfig,
    FluidShape,
    FluidSimulationSemantics,
    Material,
    Mesh,
    PhysicsBody,
    PhysicsWorld,
    Renderable,
    SceneNode,
} from "babylon-lite";
import { wgsl } from "babylon-lite/shader/wgsl.js";
import HavokPhysics from "@babylonjs/havok";
import { createLiquefyPlugin } from "./liquefy-plugin.js";
import { buildLitParticleColors } from "./particle-lit-colors.js";
import type { LitColorScene } from "./particle-lit-colors.js";
import { DEFAULT_SHIP_IBL_STRENGTH, PLAYER_START_BEHAVIOR, resolveExposure, resolveToneMapping, WEAPON_START_BEHAVIOR } from "./ship-manifest.js";
import type { ShipEnvironment } from "./ship-manifest.js";
import type { LiquefyState } from "./liquefy-plugin.js";
import {
    planFluidInitialState,
    adoptFluidForceField,
    adoptFluidParticleChannel,
    adoptFluidSceneSdf,
    attachFluidSimulationCollectionRenderLayer,
    applyFluidControls,
    applyFluidGridSettings,
    configureFluidSimulationRenderLayer,
    createFluidImpulseForce,
    createFluidParticleChannel,
    createFluidRenderEnvironment,
    createFluidSimulationProfiler,
    createFluidSimulation,
    createFluidSimulationCollection,
    createFluidControlsPanel,
    createFluidFlowEditor,
    createRayForce,
    createSolidGridBounds,
    DEFAULT_FLUID_SCHEMAS,
    disposeFluidControlsBinding,
    editFluidPresetSession,
    exportFluidPresetSession,
    FLIP_DEFAULT_PAGE_CAPACITY,
    MAX_FLUID_POLYGON_POINTS,
    pbmpmParamKeysForMaterial,
    PHYS_MAX_SCALE,
    PHYS_MIN_SCALE,
    importFluidPresetSession,
    disposeFluidParticleChannel,
    disposeFluidSimulation,
    fillFluidParticleChannel,
    beginFluidSimulationProfilerFrame,
    bindFluidControls,
    endFluidSimulationProfilerFrame,
    getFluidSimulationCollectionDiagnostics,
    readFluidParticleChannel,
    readFluidSimulationProfiler,
    refreshFluidSimulationCollectionPolygonSurfaces,
    resetFluidSimulation,
    prepareFluidReconfiguration,
    prepareFluidCollectionReconfiguration,
    projectFluidControlsMemory,
    commitFluidReconfiguration,
    commitFluidCollectionReconfiguration,
    cancelFluidCollectionReconfiguration,
    cancelFluidReconfiguration,
    retireGpuResources,
    setFluidSimulationCollectionFoam,
    setFluidSimulationCollectionSources,
    setFluidSimulationFlow,
    setFluidSimulationForceField,
    setFluidSimulationFoam,
    setFluidSimulationParameter,
    setFluidSimulationProfiler,
    setFluidSimulationSceneSdf,
    stepFluidSimulation,
    syncFluidControls,
    transformFluidFlow,
    updateFluidImpulseForce,
    writeFluidSimulationPositions,
} from "babylon-lite";
import type {
    FluidExportJson,
    FluidControlValues,
    FluidControlsBinding,
    FluidControlsBindingTarget,
    FluidFlowEditor,
    FluidFlowObjectKind,
    FluidDebug,
    FluidParticleChannel,
    FluidForceField,
    FluidMeshParticleSamplingResult,
    FluidMeshSamplingErrorInfo,
    FluidMeshSamplingStrategy,
    FluidMeshSamplingWarning,
    FluidSimulation,
    FluidSimulationReconfigurationRequest,
    FluidSimulationOptions,
    FluidSimulationProfiler,
    FluidPresetSession,
    FoamConfig,
    FoamDebugTexture,
    ForceFieldSpec,
    PhysSchemaEntry,
    SceneSdfSpec,
    VolumeSamplingMode,
} from "babylon-lite";
import type { ParticleFillWorkerResponse } from "./particle-fill-worker.js";
import { INTERACTIVE_FORCE_SAMPLE_HOLD_MS, type PairState, type PendingForce } from "./fluid/demo.js";
import { screenRay } from "./fluid/pick.js";
import { demoAssetUrl } from "./demo-asset-url.js";
import { collisionShapesForModule, worldShapesForMatrix } from "./aquanova/collision-shapes.js";
import type { ShipCollisionShape, WorldCollisionShape } from "./aquanova/collision-shapes.js";
import { hollowCylinderPrimitiveForMatrix, packPrimitives, primBufferBytes, PRIMITIVES_WGSL } from "./aquanova/collision-field.js";
import type { FluidPrimitive } from "./aquanova/collision-field.js";
import { normalizeFluidSimShape } from "./aquanova/behaviors/set-collision-shape.js";
import { AquanovaBehaviorManager } from "./aquanova/behaviors/aquanova-behavior-manager.js";
import type { BehaviorPresets, Entities, FluidSimShape } from "./aquanova/behaviors/types.js";
import { SKYBOX_EXT, SKYBOX_SIZE, SKYBOX_URL } from "./aquanova/constants.js";
import { applyLocalEnvironmentProbes, type LocalEnvironmentController } from "./aquanova/local-environments.js";
import { buildRuntimeLights } from "./aquanova/lights.js";
import type { ShipLight } from "./aquanova/manifest.js";
import { getMeshPoseGeometry } from "./mesh-pose-geometry.js";

// Both FLIP and MLS-MPM support the paged sparse grid; the other backends do not. This host-local
// capability check drives which method the shared "Paged grid" control reads/writes.
const isPagingCapableMethod = (method: string): boolean => method === "FLIP" || method === "MLS-MPM";

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
const IMPULSE_DEFAULT_DIR: readonly [number, number, number] = [0, 1, 0]; // straight up — the original lift
const DEFAULT_FORCE_STRENGTH = 0.15;
const DEFAULT_FORCE_RADIUS = 0.6;

// Studio HDR environment — drives the fluid-surface reflections + the skybox background.
const ENV_STUDIO_URL = "https://playground.babylonjs.com/textures/environment.env";
const SUN_DIR: [number, number, number] = [-0.4, -0.82, -0.45];

const MODEL_TARGET_SIZE = 7; // auto-fit: scale each model so its largest dimension ≈ this many world units

// Use the exact Aquanova ship, manifest, and authored scale so results can be compared directly
// with the game runtime.
const SCENE_URL = "/aquanova/scene.glb";
const SCENE_MANIFEST_URL = "/aquanova/scene.json";
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
    colorChannel: FluidParticleChannel | null;
    animation: AnimationGroup | null; // looping walk, paused while liquefaction owns this instance
    resumeAnimationOnCancel: boolean;
    readonly liquefyState: LiquefyState;
    readonly impulseForce: FluidForceField;
    readonly collisionOwnerId: string | null;
    sim: FluidSimulation | null;
    simulationCenter: [number, number, number] | null;
    collisionBuffer: GPUBuffer | null;
    collisionRefresh: ((placementIds: ReadonlySet<string>) => boolean) | null;
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
    sim: FluidSimulation;
    collisionBuffer: GPUBuffer | null;
    collisionRefresh: (placementIds: ReadonlySet<string>) => boolean;
    colorChannel: FluidParticleChannel;
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
    // key: ship mode uses Shift+LMB to liquefy. Ctrl provides precise, half-speed movement.
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
    const meshGizmoInfo = document.createElement("div");
    meshGizmoInfo.hidden = true;
    meshGizmoInfo.style.cssText = "margin-top:5px;padding-top:5px;border-top:1px solid rgba(143,164,188,0.3);";
    const meshGizmoNameLine = document.createElement("div");
    const meshGizmoPositionLine = document.createElement("div");
    const meshGizmoSizeLine = document.createElement("div");
    meshGizmoInfo.append(meshGizmoNameLine, meshGizmoPositionLine, meshGizmoSizeLine);
    cameraPositionHud.append(cameraPositionLine, cameraGridRelativeLine, meshGizmoInfo);
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
        camPitch = Math.atan2(dy, Math.sqrt(dx * dx + dz * dz));
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
    const setPausedState = (value: boolean): void => {
        paused = value;
        canvas.dataset.paused = paused ? "true" : "false";
    };
    setPausedState(false);
    const LOOK_SENS = 1 / 350;
    const LOOK_ACCELERATION = 28;
    const MOVE_ACCELERATION = 11;
    const cameraMovementSpeedMultiplier = (): number =>
        (camKeys.has("ShiftLeft") || camKeys.has("ShiftRight") ? 4 : 1) * (camKeys.has("ControlLeft") || camKeys.has("ControlRight") ? 0.5 : 1);
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
            endManualForceDrag(e.pointerId, true);
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
                setPausedState(!paused);
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
        const speed = cam.speed * cameraMovementSpeedMultiplier();
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
        if (dx * dx + dy * dy + dz * dz > 1e-12) {
            camYaw = Math.atan2(dx, dz);
            camPitch = Math.atan2(dy, Math.sqrt(dx * dx + dz * dz));
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
    let profiler: FluidSimulationProfiler | null = null;
    try {
        profiler = createFluidSimulationProfiler(engine);
    } catch {
        profiler = null;
    }

    // ── Enemy instances (loaded glTF models) ─────────────────────────────────
    // Each instance owns its OWN material + liquefy state so its dissolve clip is independent.
    function makeImpulse(): FluidForceField {
        return createFluidImpulseForce(engine, {
            center: [0, 0, 0],
            radius: 1,
            intensity: 0,
            direction: IMPULSE_DEFAULT_DIR,
        });
    }

    const isMeshNode = (node: SceneNode): node is Mesh => "_gpu" in node && "material" in node;

    const instances: Instance[] = [];
    const meshToInstance = new Map<Mesh, Instance>();
    const editableShipMeshes = new Set<Mesh>();
    const editableTargetByMesh = new Map<Mesh, SceneNode>();
    const instancesByEditableTarget = new Map<SceneNode, Instance[]>();
    const positionTargetByEditableTarget = new Map<SceneNode, SceneNode>();
    const instanceOf = (m: unknown): Instance | undefined => meshToInstance.get(m as Mesh);
    const assignEditableTarget = (mesh: Mesh, instance: Instance, target: SceneNode, positionTarget: SceneNode = target): void => {
        const previous = editableTargetByMesh.get(mesh);
        if (previous && previous !== target) {
            const previousInstances = instancesByEditableTarget.get(previous);
            if (previousInstances) {
                const index = previousInstances.indexOf(instance);
                if (index >= 0) previousInstances.splice(index, 1);
                if (previousInstances.length === 0) {
                    instancesByEditableTarget.delete(previous);
                    positionTargetByEditableTarget.delete(previous);
                }
            }
        }
        editableTargetByMesh.set(mesh, target);
        positionTargetByEditableTarget.set(target, positionTarget);
        const targetInstances = instancesByEditableTarget.get(target);
        if (targetInstances) {
            if (!targetInstances.includes(instance)) targetInstances.push(instance);
        } else {
            instancesByEditableTarget.set(target, [instance]);
        }
    };
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
        const impulseForce = makeImpulse();
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
            colorChannel: null,
            animation: anim,
            resumeAnimationOnCancel: false,
            liquefyState,
            impulseForce,
            collisionOwnerId: cfg.collisionOwnerId ?? null,
            sim: null,
            simulationCenter: null,
            collisionBuffer: null,
            collisionRefresh: null,
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
    const instancesByNodeName = new Map<string, Instance[]>();
    interface ShipCollisionPlacement {
        id: string;
        node: SceneNode;
        shapes: ShipCollisionShape | readonly ShipCollisionShape[];
        fluidSimShape?: { mesh: Mesh; shape: FluidSimShape };
    }
    const shipCollisionPlacements: ShipCollisionPlacement[] = [];
    let collisionDebugVisible = false;
    let collisionDebugSceneRegistered = false;
    let updateCollisionDebugOverlay = (): void => {};
    let initializeCollisionDebugOverlay = (): void => {};
    // Display roots driven by a Havok body (one per `dynamic` node). `rest` is the spawn pose, which
    // Restart teleports the body back to.
    interface DynamicDisplay {
        proxy: SceneNode;
        body: PhysicsBody;
        disp: SceneNode;
        rest: [number, number, number];
        restRotation: [number, number, number, number];
    }
    const dynDisplays: DynamicDisplay[] = [];
    const dynamicDisplayByTarget = new Map<SceneNode, DynamicDisplay>();
    let physWorld: PhysicsWorld | null = null;
    let localEnvironmentController: LocalEnvironmentController | null = null;
    let shipBehaviorManager: AquanovaBehaviorManager | null = null;

    // ── Aquanova ship meshes ─────────────────────────────────────────────────
    // Load the full scene.glb exactly as Aquanova does and make every scene mesh a foe, in place inside
    // the ship. Geometry, materials, trim atlas, world transforms and the particle-fill path are all
    // identical to Aquanova, so the only variables left are this demo's lighting, camera and fluid
    // controls. scene.json's `entities` names are used only to pick which room to frame.
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
        let focus: [number, number, number] = chunks[0] ? chunkFocus(chunks[0]) : [0, 1.5, 0];
        let asset;
        try {
            asset = await loadGltf(engine, SCENE_URL);
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[aquanova-fluid-sim] scene.glb load failed", err);
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
            console.warn("[aquanova-fluid-sim] scene.glb contained no sampleable meshes");
            return;
        }

        const placementNodeForNode = (node: SceneNode): SceneNode | null => {
            let current: SceneNode | null = node;
            while (current) {
                const id = (current.metadata?.gltf?.extras as { id?: string } | undefined)?.id;
                if (id) {
                    return current;
                }
                current = current.parent as SceneNode | null;
            }
            return null;
        };
        const placementIdForNode = (node: SceneNode): string | null => (placementNodeForNode(node)?.metadata?.gltf?.extras as { id?: string } | undefined)?.id ?? null;
        const meshesByEntityName = new Map<string, Mesh[]>();
        const entityNameByMesh = new Map<Mesh, string>();
        for (const { mesh, owner } of shipMeshes) {
            entityNameByMesh.set(mesh, owner.name);
            const meshes = meshesByEntityName.get(owner.name);
            if (meshes) meshes.push(mesh);
            else meshesByEntityName.set(owner.name, [mesh]);
        }
        shipBehaviorManager = new AquanovaBehaviorManager({
            presets: shipManifest?.behaviorPresets,
            entities: shipManifest?.entities,
            meshesByEntityName,
            entityNameOf: (mesh) => entityNameByMesh.get(mesh) ?? mesh.name,
        });
        const behaviorManager = shipBehaviorManager;
        // Placement markers are not scenery: Aquanova hides them and spawns on them, so do the same
        // here rather than letting the player shoot an invisible floor strip.
        const markerNames = new Set(
            [PLAYER_START_BEHAVIOR, WEAPON_START_BEHAVIOR]
                .map((behaviorName) => behaviorManager.findEntityWithBehavior(behaviorName)?.entityName)
                .filter((value): value is string => !!value)
        );
        const disabledShipMeshes = new Set<Mesh>();
        const hiddenAreaBoxEntities = new Set<string>();
        for (const { mesh, owner } of shipMeshes) {
            const placementModule = (placementNodeForNode(owner)?.metadata?.gltf?.extras as { module?: string } | undefined)?.module;
            const unconditionalHide =
                renderLikeAquanova &&
                placementModule === "Aquanova/Props/AreaBox" &&
                behaviorManager.assignmentsOf(owner.name).some((assignment) => assignment.name === "hideEntity" && (!assignment.events || assignment.events.length === 0));
            if (unconditionalHide) {
                hiddenAreaBoxEntities.add(owner.name);
            }
            if (markerNames.has(owner.name) || mesh.name.startsWith("Portal_") || unconditionalHide) {
                setMeshVisible(mesh, false);
                (mesh as { pickable?: boolean }).pickable = false;
                disabledShipMeshes.add(mesh);
            } else {
                editableShipMeshes.add(mesh);
            }
        }
        canvas.dataset.hiddenAreaBoxEntityCount = String(hiddenAreaBoxEntities.size);
        canvas.dataset.hiddenAreaBoxMeshCount = String(shipMeshes.reduce((count, { owner }) => count + (hiddenAreaBoxEntities.has(owner.name) ? 1 : 0), 0));
        behaviorManager.classifyMeshes(
            shipMeshes.map(({ mesh }) => mesh),
            {
                isDisabled: (mesh) => disabledShipMeshes.has(mesh),
                instanceIdOf: () => undefined,
            }
        );

        // Frame the room that actually holds a node Aquanova marks liquefiable. The manifest's
        // `entities` map is keyed by glTF NODE name (what the sandbox shows), so the room is
        // resolved from a listed node's position rather than read off a chunk entry.
        const marked = found.find(({ mesh }) => behaviorManager.isLiquefiable(mesh));
        const mn = marked?.mesh.boundMin;
        const mx = marked?.mesh.boundMax;
        if (mn && mx) {
            const gx = -(mn[0]! + mx[0]!) / 2; // Lite → glTF (negate X)
            const gz = (mn[2]! + mx[2]!) / 2;
            const room = chunks.find((c) => gx >= c.aabb.min[0]! && gx <= c.aabb.max[0]! && gz >= c.aabb.min[2]! && gz <= c.aabb.max[2]!);
            if (room) focus = chunkFocus(room);
        }

        let n = 0;
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
            assignEditableTarget(mesh, inst, placementNodeForNode(owner) ?? owner);
            const siblings = nodeGroups.get(owner);
            if (siblings) siblings.push(inst);
            else nodeGroups.set(owner, [inst]);
            groupOfInstance.set(inst, nodeGroups.get(owner)!);
            const byName = instancesByNodeName.get(owner.name);
            if (byName) byName.push(inst);
            else instancesByNodeName.set(owner.name, [inst]);
            n++;
        }

        let fluidSimShapePlacements = 0;
        for (const [owner, ownerInstances] of nodeGroups) {
            const collisionAssignment = behaviorManager.assignmentsOf(owner.name).find((assignment) => assignment.name === "setCollisionShape");
            const shape = normalizeFluidSimShape(collisionAssignment?.fluidSimShape);
            if (!shape) continue;
            const mesh = ownerInstances[0]?.meshes[0];
            const placementNode = placementNodeForNode(owner);
            const placementId = placementIdForNode(owner);
            if (!mesh || !placementNode || !placementId) {
                throw new Error(`[aquanova-fluid-sim] setCollisionShape entity "${owner.name}" is not attached to a ship placement`);
            }
            let placement = shipCollisionPlacements.find((candidate) => candidate.id === placementId);
            if (!placement) {
                placement = { id: placementId, node: placementNode, shapes: [] };
                shipCollisionPlacements.push(placement);
            }
            placement.fluidSimShape = { mesh, shape };
            fluidSimShapePlacements++;
        }
        canvas.dataset.collisionPlacements = String(shipCollisionPlacements.length);
        canvas.dataset.fluidSimShapePlacements = String(fluidSimShapePlacements);

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
                    const environment = createFluidRenderEnvironment(fluidEnvironment);
                    configureFluidSimulationRenderLayer(surfaceTask, { environment });
                    configureFluidSimulationRenderLayer(polygonSurfaceTask, { environment });
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
        const dynNodes = [...nodeGroups.entries()].filter(([, group]) => group.some((instance) => instance.meshes.some((mesh) => behaviorManager.dynamicMeshes.has(mesh))));
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
                const ownedTargets = new Set<SceneNode>();
                for (const inst of group) {
                    for (const mesh of inst.meshes) {
                        const target = editableTargetByMesh.get(mesh);
                        if (target) ownedTargets.add(target);
                    }
                    setParent(inst.root, disp);
                    // `setParent` rewrites the root's LOCAL transform to preserve its world pose, so the
                    // home captured at registration (identity, pre-reparenting) is now stale. Without this,
                    // Restart collapses every primitive of the node onto the display root's origin.
                    inst.homePos = [inst.root.position.x, inst.root.position.y, inst.root.position.z];
                }
                // Keep the authored placement node under the same display root as its detached
                // liquefaction primitives. Its world matrix is also the source for module collision,
                // so visuals and collision continue to share one transform.
                for (const target of ownedTargets) {
                    if (target !== disp) setParent(target, disp);
                }
                const positionTarget = ownedTargets.size === 1 ? ownedTargets.values().next().value : undefined;
                for (const inst of group) {
                    for (const mesh of inst.meshes) assignEditableTarget(mesh, inst, disp, positionTarget ?? disp);
                }
                const box = addBox(-centre[0], centre[1], centre[2], half[0] * 2, half[1] * 2, half[2] * 2, PhysicsMotionType.ANIMATED);
                if (box) {
                    const display: DynamicDisplay = {
                        proxy: box.node,
                        body: box.body,
                        disp,
                        rest: centre,
                        restRotation: [0, 0, 0, 1],
                    };
                    dynDisplays.push(display);
                    dynamicDisplayByTarget.set(disp, display);
                }
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
        const startEntity = behaviorManager.findEntityWithBehavior(PLAYER_START_BEHAVIOR);
        const startMeshes = startEntity ? found.filter(({ owner }) => owner.name === startEntity.entityName) : [];
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
        const dir = startEntity?.assignment.direction;
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

    // Combined render buffer aggregating every live sim's particles into ONE surface pass. The
    // capacity comes from the shared core authority so authoring and production agree exactly
    // (the intended 600k, clamped to what the device's storage buffers can address).
    const MAX_TOTAL = aquanovaCombinedParticleCapacity(engine._device.limits);
    let useMeshColors = false; // UI toggle: tint the water by the liquefied mesh's texture/vertex colours
    const simulationCollection = createFluidSimulationCollection(engine);
    const surfaceTask = attachFluidSimulationCollectionRenderLayer(simulationCollection, {
        scene,
        camera: cam,
        mode: "surface",
        backgroundTarget: sceneColorRT,
        outputTarget: engine.scRT,
        depthTarget: depthRT,
        particleCapacity: MAX_TOTAL,
        profiler,
        useParticleColor: useMeshColors,
    });
    const polygonSurfaceTask = attachFluidSimulationCollectionRenderLayer(simulationCollection, {
        scene,
        camera: cam,
        mode: "polygon",
        backgroundTarget: sceneColorRT,
        outputTarget: engine.scRT,
        depthTarget: depthRT,
        profiler,
    });
    const foamTask = attachFluidSimulationCollectionRenderLayer(simulationCollection, {
        scene,
        camera: cam,
        mode: "foam",
        colorTarget: engine.scRT,
        depthTarget: depthRT,
        profiler,
        surfaceLayer: surfaceTask,
        polygonSurfaceLayer: polygonSurfaceTask,
    });

    function runningSims(): FluidSimulation[] {
        const sims = instances.flatMap((inst) => (inst.sim ? [inst.sim] : []));
        if (manualRun) {
            sims.push(manualRun.sim);
        }
        return sims;
    }

    function syncSimulationCollection(): void {
        setFluidSimulationCollectionSources(simulationCollection, [
            ...instances.flatMap((inst) =>
                inst.sim
                    ? [
                          {
                              simulation: inst.sim,
                              count: inst.count,
                              opacity: inst.phase === "fading" ? Math.max(0, 1 - inst.fadeElapsed / FADE_DUR) : 1,
                              ...(useMeshColors && inst.colorChannel ? { color: inst.colorChannel } : {}),
                          },
                      ]
                    : []
            ),
            ...(manualRun
                ? [
                      {
                          simulation: manualRun.sim,
                          count: manualRun.sim.activeCount ?? manualRun.sim.count,
                          opacity: 1,
                          ...(useMeshColors ? { color: manualRun.colorChannel } : {}),
                      },
                  ]
                : []),
        ]);
    }

    function polygonSurfaceRenderingEnabled(): boolean {
        return currentMethod === "FLIP" && (physValues.FLIP?.polygonSurface ?? 0) >= 0.5;
    }

    function syncPolygonSurfaceRendering(): void {
        const enabled = polygonSurfaceRenderingEnabled();
        configureFluidSimulationRenderLayer(polygonSurfaceTask, { enabled });
        configureFluidSimulationRenderLayer(foamTask, { foam: { polygonSurfaceDepth: enabled } });
        canvas.dataset.render = enabled ? "polygon" : "surface";
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

    function applyFoamToSim(sim: FluidSimulation): void {
        const f = controls.getValues().foam;
        setFluidSimulationFoam(sim, f.enabled ? currentFoamConfig() : null);
    }

    function fillManualColor(run: ManualRun, color = controls.getValues().color): void {
        const rgb = hexToRgb(color);
        fillFluidParticleChannel(run.colorChannel, [rgb[0], rgb[1], rgb[2], 1]);
    }

    // Per-particle colour: for each DISTINCT sub-mesh texture, a compute pass samples that texture at
    // the particle's UV (from the worker) for the particles that belong to it (texIdx == T), gamma-
    // encoding into the instance colour buffer. Particles with no texture keep a baseColor fill. The
    // channels are aggregated by the shared simulation collection each frame.
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

    function fillInstanceColor(inst: Instance, uvs: Float32Array | null, texIndices: Uint32Array | null, count: number): GPUBuffer {
        if (inst.colorChannel) {
            const previous = inst.colorChannel;
            inst.colorChannel = null;
            syncSimulationCollection();
            disposeFluidParticleChannel(previous);
        }
        const buf = device.createBuffer({ label: `liq-color-${inst.key}`, size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
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
        if (!inst.diffuseTexs.length || !uvs || !texIndices || uvs.length < count * 2 || texIndices.length < count) return buf;
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
        return buf;
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
    let gridBoundsOverlayVisible = false;

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
                endFluidSimulationProfilerFrame(prof);
                return 0;
            },
            dispose: (): void => {},
        });
    }

    const invalidatePhaseTaskBundles = (): void => {
        for (const task of [sceneTask, foeTask]) {
            task._ob.length = 0;
        }
        foeTask.enabled = gridBoundsOverlayVisible || instances.some((instance) => instance.phase === "dissolving");
    };

    const defaultRenderProfile = {
        waterColor: DEF_COLOR,
        absorption: DEF_ABSORPTION,
        particleSize: DEF_SIZE,
        refractionStrength: DEF_REFRACTION,
        specularPower: DEF_SPECULAR,
        surfaceDepthBlur: DEF_DEPTH_BLUR,
        depthBlurEdgeThreshold: DEF_DEPTH_BLUR_THRESHOLD,
        surfaceThicknessBlur: DEF_THICKNESS_BLUR,
        halfRendering: DEF_HALF,
        thicknessDownscale: DEF_THICKNESS_DOWNSCALE,
        surfaceFilter: DEF_SURFACE_FILTER,
        narrowRangeDelta: DEF_NARROW_DELTA,
        narrowRangeMu: DEF_NARROW_MU,
    };
    configureFluidSimulationRenderLayer(surfaceTask, { direction: SUN_DIR, profile: defaultRenderProfile, surfaceMode: "surface" });
    configureFluidSimulationRenderLayer(polygonSurfaceTask, { direction: SUN_DIR, profile: defaultRenderProfile, enabled: false });

    enableMaterialPlugins(scene);

    const brdfUrl = demoAssetUrl("./brdf-lut.png", import.meta.url);

    // ── Environment picker ────────────────────────────────────────────────────
    // ── Ship manifest (ship mode only) ───────────────────────────────────────
    // Fetched ONCE, before the environment slots are graded: the manifest carries the Blender view
    // transform the ship was authored with (tone mapping + exposure in stops) and the IBL strength,
    // and Aquanova reads exactly the same values — so the two demos render the interior identically.
    interface ShipManifestData {
        environment?: ShipEnvironment;
        behaviorPresets?: BehaviorPresets;
        entities?: Entities;
        chunks?: { id?: string; node?: string; aabb: { min: number[]; max: number[] } | null }[];
        moduleCollision?: Readonly<Record<string, ShipCollisionShape | readonly ShipCollisionShape[]>>;
        fluidSim?: string[];
        lights?: ShipLight[];
    }
    let shipManifest: ShipManifestData | null = null;
    const shipManifestReady = fetch(SCENE_MANIFEST_URL)
        .then((r) => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.json() as Promise<ShipManifestData>;
        })
        .then((j) => {
            shipManifest = j;
        })
        .catch((err: unknown) => {
            // eslint-disable-next-line no-console
            console.warn("[aquanova-fluid-sim] scene metadata load failed", err);
        });
    let publishedFluidSimNames: string[] = [];
    const behaviorOptionsReady = fetch("/aquanova/behaviors.json")
        .then((response) => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json() as Promise<{ options?: Record<string, unknown> }>;
        })
        .then((metadata) => {
            const names = metadata.options?.["Fluid Simulations"];
            publishedFluidSimNames = Array.isArray(names) ? names.filter((name): name is string => typeof name === "string") : [];
        })
        .catch((err: unknown) => {
            // eslint-disable-next-line no-console
            console.warn("[aquanova-fluid-sim] behavior options load failed", err);
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
        const environment = createFluidRenderEnvironment(slot.env);
        configureFluidSimulationRenderLayer(surfaceTask, { environment });
        configureFluidSimulationRenderLayer(polygonSurfaceTask, { environment });
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
    let radiusValue = fluidSimulationParticleRadius({ physicsParticleSize: 1, samplingType: "fluid" });
    let fluidPhysicsScale = 1;
    let modeValue: VolumeSamplingMode = "dense";
    // Volume lattice vs surface shell. `auto` is the shipped behaviour (openness + thickness gates,
    // plus the per-prop hollow hint in MODEL_FOES); the other two force the choice so the difference
    // can be auditioned on the same prop.
    let fillStrategy: FluidMeshSamplingStrategy = "auto";
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
    let showGridBoundsSolid = false;
    let showGridGizmo = false;
    let currentMethod: "PBF" | "FLIP" | "MLS-MPM" | "PB-MPM" = "MLS-MPM";
    let controlsBinding: FluidControlsBinding | null = null;
    let simulationSemantics: FluidSimulationSemantics = CURRENT_FLUID_SIMULATION_SEMANTICS;
    let currentMaterial = 0;
    let flipGridResolution = 160;
    let flipMarkersPerCell = 8;
    let flipParticleCapacityRequest: number | null = null;
    const particleCapacityRequestByMethod = new Map<string, number>();
    let mpmActiveBlocks = false;
    let mpmPagedGrid = false;
    const fluidDeviceLimits = engine._device.limits;
    const maxPagedGridPages = fluidMaximumPageCapacity("MLS-MPM", [2048, 2048, 2048], fluidDeviceLimits);
    let mpmPagedGridMaxPages = Math.min(maxPagedGridPages, 40000);
    let mpmFusedBlockDiscovery = false;
    // FLIP owns its own paging state — the 8³ paged grid is a distinct backend from MLS-MPM's, so a
    // single shared flag would leak between methods and turn the FLIP checkbox into a no-op.
    let flipPagedGrid = false;
    let flipPagedGridMaxPages = FLIP_DEFAULT_PAGE_CAPACITY;
    let collisionNeighborhoodRadius = 12;
    type SimulationType = "mesh" | "fluid";
    let simulationType: SimulationType = "mesh";
    let manualRun: ManualRun | null = null;
    let manualStepCount = 0;
    let manualForceStrength = DEFAULT_FORCE_STRENGTH;
    let manualForceRadius = DEFAULT_FORCE_RADIUS;
    const rayForce = createRayForce(device);
    const forceFieldHandles = new WeakMap<ForceFieldSpec, ReturnType<typeof adoptFluidForceField>>();
    const forceFieldHandle = (spec: ForceFieldSpec | null): ReturnType<typeof adoptFluidForceField> | null => {
        if (!spec) return null;
        let handle = forceFieldHandles.get(spec);
        if (!handle) {
            handle = adoptFluidForceField(engine, spec);
            forceFieldHandles.set(spec, handle);
        }
        return handle;
    };
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
    let committedGridPosition: [number, number, number] = [...gridPosition];
    let refreshFlowControls = (): void => {};
    let refreshFlipParticleCapacity = (): void => {};
    let refreshParticleCountControl = (): void => {};
    const simulationBounds = (center: readonly [number, number, number] = gridPosition): { min: [number, number, number]; max: [number, number, number] } => ({
        min: [center[0] - gridSize[0] * 0.5, center[1] - gridSize[1] * 0.5, center[2] - gridSize[2] * 0.5],
        max: [center[0] + gridSize[0] * 0.5, center[1] + gridSize[1] * 0.5, center[2] + gridSize[2] * 0.5],
    });
    // Resolve solver discretization (radius, cell size, particle volume, grid dims) for the active
    // method through the shared "aquanova" compatibility profile — the single home for this policy.
    // FLIP uses the authored grid resolution; non-FLIP honors an authored mesh radius.
    const manualFluidConfigForMethod = (method: "PBF" | "FLIP" | "MLS-MPM" | "PB-MPM", center: readonly [number, number, number] = gridPosition) => {
        const bounds = simulationBounds(center);
        return resolveFluidSimulationConfig("aquanova", {
            method,
            physicsScale: fluidPhysicsScale,
            samplingType: simulationType,
            semantics: simulationSemantics,
            physics: physValues[method],
            ...(method !== "FLIP" && radiusValue !== undefined ? { particleRadius: radiusValue } : {}),
            bounds: { min: bounds.min, max: bounds.max },
            ...(method === "FLIP" ? { gridResolution: flipGridResolution, markersPerCell: flipMarkersPerCell } : {}),
        });
    };
    const manualFluidConfig = (center: readonly [number, number, number] = gridPosition) => manualFluidConfigForMethod(currentMethod, center);
    const flipDiscretization = (center: readonly [number, number, number] = gridPosition) => {
        const bounds = simulationBounds(center);
        return resolveFluidSimulationConfig("aquanova", {
            method: "FLIP",
            physicsScale: fluidPhysicsScale,
            bounds: { min: bounds.min, max: bounds.max },
            gridResolution: flipGridResolution,
            markersPerCell: flipMarkersPerCell,
        }).flip!;
    };
    const flipPageLimit = (): number => fluidMaximumPageCapacity("FLIP", flipDiscretization().gridDim, fluidDeviceLimits);
    const mpmPageLimit = (): number => {
        const bounds = simulationBounds();
        const config = resolveFluidSimulationConfig("aquanova", {
            method: "MLS-MPM",
            physicsScale: fluidPhysicsScale,
            samplingType: simulationType,
            ...(radiusValue !== undefined ? { particleRadius: radiusValue } : {}),
            bounds: { min: bounds.min, max: bounds.max },
        });
        return fluidMaximumPageCapacity("MLS-MPM", config.gridDim!, fluidDeviceLimits);
    };
    // Per-method paging accessors. FLIP and MLS-MPM keep independent paged-grid state, so the shared
    // "Paged grid" control must read and write whichever method is active.
    const currentPagedGrid = (): boolean => (currentMethod === "FLIP" ? flipPagedGrid : mpmPagedGrid);
    const currentPagedGridMaxPages = (): number => (currentMethod === "FLIP" ? flipPagedGridMaxPages : mpmPagedGridMaxPages);
    const currentPagedGridLimit = (): number => (currentMethod === "FLIP" ? flipPageLimit() : mpmPageLimit());
    // Push the active method's paging state into the shared panel (on method switch / preset import).
    const syncPagingControls = (): void => {
        if (currentMethod === "MLS-MPM") {
            controls.setActiveBlocks(mpmActiveBlocks);
            controls.setFusedBlockDiscovery(mpmFusedBlockDiscovery);
        }
        controls.setPagedGrid(currentPagedGrid());
        controls.setPagedGridMaxPages(currentPagedGridMaxPages(), currentPagedGridLimit());
        controls.setPagedGridStatus("");
    };
    const fluidParticleRadius = (): number => manualFluidConfig().particleRadius;
    const simulationCellSize = (): number => manualFluidConfig().cellSize;
    const flowInWorldSpace = (flow: FluidFlowConfig): FluidFlowConfig => {
        return transformFluidFlow(flow, { translation: gridPosition });
    };
    const manualInitialStateKey = (): string => {
        const worldFlow = flowInWorldSpace(activeFlow);
        const config = manualFluidConfig();
        return planFluidInitialState({
            particleCapacity: manualParticleCapacity(worldFlow),
            particleVolume: config.particleVolume,
            flow: worldFlow,
            bounds: simulationBounds(),
            deriveInitialCount: currentMethod === "FLIP",
        }).initialStateKey;
    };
    const manualConfigurationSignature = (): string =>
        JSON.stringify({
            method: currentMethod,
            fluidPhysicsScale,
            gridPosition,
            gridSize,
            flipGridResolution,
            flipMarkersPerCell,
            flipParticleCapacityRequest,
            flipPagedGrid,
            flipPagedGridMaxPages,
            particleCapacityRequest: particleCapacityRequestByMethod.get(currentMethod),
            initialStateKey: manualInitialStateKey(),
        });

    function beginManualForceDrag(event: PointerEvent): boolean {
        if (event.button !== 2 || !event.shiftKey || simulationType !== "fluid" || !manualRun) {
            return false;
        }
        forceDragging = true;
        pendingForce = null;
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
        const speed = (Math.sqrt(dx * dx + dy * dy) / dtMs) * 1000;
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
        const length = Math.sqrt(pushX * pushX + pushY * pushY + pushZ * pushZ) || 1;
        pushX /= length;
        pushY /= length;
        pushZ /= length;
        pendingForce = {
            origin: ray.origin,
            dir: ray.dir,
            push: [pushX, pushY, pushZ],
            radius: manualForceRadius,
            accel: speed * manualForceStrength,
            expiresAt: now + INTERACTIVE_FORCE_SAMPLE_HOLD_MS,
        };
    }

    function endManualForceDrag(pointerId = forcePointerId, preservePendingForce = false): void {
        if (!forceDragging) {
            return;
        }
        forceDragging = false;
        if (!preservePendingForce) {
            pendingForce = null;
            rayForce.clear();
            if (manualRun) setFluidSimulationForceField(manualRun.sim, null);
        }
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
    const fluidPrimitivesForPlacement = (placement: ShipCollisionPlacement): FluidPrimitive[] => {
        const replacement = placement.fluidSimShape;
        if (replacement) {
            const { shape, mesh } = replacement;
            return [hollowCylinderPrimitiveForMatrix(mesh.worldMatrix, shape.start, shape.height, shape.innerRadius, shape.outerRadius)];
        }
        return worldShapesForMatrix(placement.node.worldMatrix, placement.shapes).map(worldShapeToFluidPrimitive);
    };
    const primitiveIntersectsSphere = (primitive: FluidPrimitive, center: readonly [number, number, number], radius: number): boolean => {
        let sx = primitive.a[0];
        let sy = primitive.a[1];
        let sz = primitive.a[2];
        let ex = primitive.radius ?? 0;
        let ey = ex;
        let ez = ex;
        if (primitive.kind === "box") {
            const h = primitive.b ?? [0, 0, 0];
            const q = primitive.rotation ?? [0, 0, 0, 1];
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
        } else if (primitive.kind === "capsule" || primitive.kind === "cylinder" || primitive.kind === "hollowCylinder") {
            const a = primitive.a;
            const b = primitive.b ?? primitive.a;
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
    const quaternionFromYTo = (direction: readonly [number, number, number]): [number, number, number, number] => {
        const length = Math.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
        if (length < 1e-9) return [0, 0, 0, 1];
        const x = direction[0] / length;
        const y = direction[1] / length;
        const z = direction[2] / length;
        if (y > 0.999999) return [0, 0, 0, 1];
        if (y < -0.999999) return [1, 0, 0, 0];
        const scale = Math.sqrt((1 + y) * 2);
        return [z / scale, 0, -x / scale, scale * 0.5];
    };
    const createHollowCylinderDebugMesh = (innerRatio: number): Mesh => {
        const segments = 32;
        const positions: number[] = [];
        const indices: number[] = [];
        const vertex = (radius: number, y: number, angle: number): number => {
            positions.push(Math.cos(angle) * radius, y, Math.sin(angle) * radius);
            return positions.length / 3 - 1;
        };
        const quad = (a: number, b: number, c: number, d: number): void => {
            indices.push(a, b, c, a, c, d);
        };
        for (let i = 0; i < segments; i++) {
            const a0 = (i / segments) * Math.PI * 2;
            const a1 = ((i + 1) / segments) * Math.PI * 2;
            const outerBottom0 = vertex(1, -0.5, a0);
            const outerBottom1 = vertex(1, -0.5, a1);
            const outerTop0 = vertex(1, 0.5, a0);
            const outerTop1 = vertex(1, 0.5, a1);
            const innerBottom0 = vertex(innerRatio, -0.5, a0);
            const innerBottom1 = vertex(innerRatio, -0.5, a1);
            const innerTop0 = vertex(innerRatio, 0.5, a0);
            const innerTop1 = vertex(innerRatio, 0.5, a1);
            quad(outerBottom0, outerBottom1, outerTop1, outerTop0);
            quad(innerBottom1, innerBottom0, innerTop0, innerTop1);
            quad(innerBottom0, innerBottom1, outerBottom1, outerBottom0);
            quad(innerTop0, outerTop0, outerTop1, innerTop1);
        }
        const positionData = new Float32Array(positions);
        return createMeshFromData(engine, "collision-debug-hollow-cylinder", positionData, new Float32Array(positionData.length), new Uint32Array(indices));
    };
    initializeCollisionDebugOverlay = (): void => {
        const material = createShaderMaterial({
            name: "aquanova-fluid-collision-debug",
            vertexSource: wgsl`struct VertexOutput{@builtin(position) position:vec4<f32>};@vertex fn mainVertex(input:VertexInput)->VertexOutput{var out:VertexOutput;out.position=shaderSystem.viewProjection*(shaderSystem.world*vec4<f32>(input.position,1.0));return out;}`,
            fragmentSource: wgsl`@fragment fn mainFragment()->@location(0) vec4<f32>{return vec4<f32>(0.1,0.9,1.0,0.08);}`,
            attributes: ["position"],
            uniforms: ["world", "viewProjection"],
            needAlphaBlending: true,
            blendMode: "alpha",
            depthWrite: false,
            depthCompare: "always",
            backFaceCulling: false,
        });
        const entries: Array<{ placement: ShipCollisionPlacement; primitiveIndex: number; mesh: Mesh }> = [];
        for (const placement of shipCollisionPlacements) {
            const primitives = fluidPrimitivesForPlacement(placement);
            for (let primitiveIndex = 0; primitiveIndex < primitives.length; primitiveIndex++) {
                const primitive = primitives[primitiveIndex]!;
                const radius = Math.max(primitive.radius ?? 0.1, 1e-3);
                const b = primitive.b ?? primitive.a;
                const axisX = b[0] - primitive.a[0];
                const axisY = b[1] - primitive.a[1];
                const axisZ = b[2] - primitive.a[2];
                const axisLength = Math.max(Math.sqrt(axisX * axisX + axisY * axisY + axisZ * axisZ), 1e-3);
                let mesh: Mesh;
                if (primitive.kind === "box") {
                    mesh = createBox(engine, 1);
                } else if (primitive.kind === "sphere") {
                    mesh = createSphere(engine, { diameter: 1 });
                } else if (primitive.kind === "capsule") {
                    mesh = createCapsule(engine, { radius, height: axisLength + radius * 2 });
                } else if (primitive.kind === "hollowCylinder") {
                    mesh = createHollowCylinderDebugMesh(Math.max(0, Math.min(1, (primitive.innerRadius ?? 0) / radius)));
                } else {
                    mesh = createCylinder(engine, { height: 1, diameter: 1, tessellation: 32 });
                }
                mesh.name = `collision-debug-${placement.id}-${primitiveIndex}-${primitive.kind}`;
                mesh.material = material;
                mesh.pickable = false;
                mesh.renderOrder = 9_997;
                mesh.position.set(0, -1e6, 0);
                addToScene(scene, mesh);
                entries.push({ placement, primitiveIndex, mesh });
            }
        }
        canvas.dataset.collisionDebugShapeCount = String(entries.length);
        updateCollisionDebugOverlay = (): void => {
            for (const entry of entries) {
                if (!collisionDebugVisible) {
                    if (collisionDebugSceneRegistered) {
                        if (entry.mesh.visible !== false) setMeshVisible(entry.mesh, false);
                    } else {
                        entry.mesh.position.set(0, -1e6, 0);
                    }
                    continue;
                }
                if (entry.mesh.visible === false) setMeshVisible(entry.mesh, true);
                const primitive = fluidPrimitivesForPlacement(entry.placement)[entry.primitiveIndex];
                if (!primitive) {
                    entry.mesh.position.set(0, -1e6, 0);
                    continue;
                }
                const radius = Math.max(primitive.radius ?? 0.1, 1e-3);
                if (primitive.kind === "box") {
                    const half = primitive.b ?? [0.1, 0.1, 0.1];
                    const rotation = primitive.rotation ?? [0, 0, 0, 1];
                    entry.mesh.position.set(primitive.a[0], primitive.a[1], primitive.a[2]);
                    entry.mesh.scaling.set(Math.max(half[0] * 2, 1e-3), Math.max(half[1] * 2, 1e-3), Math.max(half[2] * 2, 1e-3));
                    entry.mesh.rotationQuaternion.set(rotation[0], rotation[1], rotation[2], rotation[3]);
                } else if (primitive.kind === "sphere") {
                    entry.mesh.position.set(primitive.a[0], primitive.a[1], primitive.a[2]);
                    entry.mesh.scaling.set(radius * 2, radius * 2, radius * 2);
                    entry.mesh.rotationQuaternion.set(0, 0, 0, 1);
                } else {
                    const b = primitive.b ?? primitive.a;
                    const axis: [number, number, number] = [b[0] - primitive.a[0], b[1] - primitive.a[1], b[2] - primitive.a[2]];
                    const axisLength = Math.max(Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]), 1e-3);
                    const rotation = quaternionFromYTo(axis);
                    entry.mesh.position.set((primitive.a[0] + b[0]) * 0.5, (primitive.a[1] + b[1]) * 0.5, (primitive.a[2] + b[2]) * 0.5);
                    entry.mesh.rotationQuaternion.set(rotation[0], rotation[1], rotation[2], rotation[3]);
                    if (primitive.kind !== "capsule") {
                        entry.mesh.scaling.set(radius, axisLength, radius);
                    }
                }
            }
            canvas.dataset.collisionDebug = String(collisionDebugVisible);
        };
        updateCollisionDebugOverlay();
    };
    interface NeighborhoodSdf {
        spec: SceneSdfSpec;
        buffer: GPUBuffer | null;
        count: number;
        refreshPlacements: (placementIds: ReadonlySet<string>) => boolean;
    }
    const createNeighborhoodSdf = (center: readonly [number, number, number], excludePlacementId: string | null): NeighborhoodSdf => {
        if (collisionNeighborhoodRadius <= 0) {
            return { spec: groundSdf, buffer: null, count: 0, refreshPlacements: () => false };
        }
        const primitives: FluidPrimitive[] = [];
        const placementRanges: Array<{ placement: (typeof shipCollisionPlacements)[number]; shapeIndices: number[]; primitiveOffset: number }> = [];
        for (const placement of shipCollisionPlacements) {
            if (placement.id === excludePlacementId) continue;
            const placementPrimitives = fluidPrimitivesForPlacement(placement);
            const shapeIndices: number[] = [];
            const primitiveOffset = primitives.length;
            for (let i = 0; i < placementPrimitives.length; i++) {
                const primitive = placementPrimitives[i]!;
                if (!primitiveIntersectsSphere(primitive, center, collisionNeighborhoodRadius)) continue;
                shapeIndices.push(i);
                primitives.push(primitive);
            }
            if (shapeIndices.length > 0) {
                placementRanges.push({ placement, shapeIndices, primitiveOffset });
            }
        }
        if (primitives.length === 0) {
            return { spec: groundSdf, buffer: null, count: 0, refreshPlacements: () => false };
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
            refreshPlacements: (placementIds) => {
                let changed = false;
                for (const { placement, shapeIndices, primitiveOffset } of placementRanges) {
                    if (!placementIds.has(placement.id)) continue;
                    const placementPrimitives = fluidPrimitivesForPlacement(placement);
                    for (let i = 0; i < shapeIndices.length; i++) {
                        primitives[primitiveOffset + i] = placementPrimitives[shapeIndices[i]!]!;
                    }
                    changed = true;
                }
                if (changed) {
                    packPrimitives(packed, primitives);
                    device.queue.writeBuffer(buffer, 0, packed);
                }
                return changed;
            },
        };
    };

    function configuredSimulationOptions(
        count: number,
        radius: number,
        boundsMin: [number, number, number],
        boundsMax: [number, number, number],
        positions?: Float32Array
    ): FluidSimulationOptions {
        const overflow = (backend: "FLIP" | "MLS-MPM") => (requiredPages: number, capacity: number) => {
            const message = `Page capacity exceeded: ${requiredPages.toLocaleString()} required, ${capacity.toLocaleString()} allocated.`;
            if (controlsBinding) {
                syncFluidControls(controlsBinding, { pageDiagnostics: { method: backend, requiredPages, capacity, overflow: true } });
            } else {
                controls.setPagedGridStatus(message, true);
            }
            console.error(`[AquanovaFluidSim ${backend}] ${message}`);
        };
        return {
            method: currentMethod,
            particleCount: count,
            bounds: { min: boundsMin, max: boundsMax },
            physicsScale: fluidPhysicsScale,
            samplingType: simulationType,
            particleRadius: radius,
            semantics: simulationSemantics,
            physics: physValues[currentMethod],
            initialPositions: positions,
            groundY: GROUND_Y,
            gridResolution: currentMethod === "FLIP" ? flipGridResolution : undefined,
            markersPerCell: currentMethod === "FLIP" ? flipMarkersPerCell : undefined,
            material: currentMethod === "PB-MPM" ? currentMaterial : undefined,
            activeBlocks: mpmActiveBlocks,
            pagedGrid: currentMethod === "FLIP" ? flipPagedGrid : mpmPagedGrid,
            pagedGridMaxPages: currentMethod === "FLIP" ? flipPagedGridMaxPages : mpmPagedGridMaxPages,
            fusedBlockDiscovery: mpmFusedBlockDiscovery,
            onPagedGridPages: (requiredPages, capacity) => {
                if (controlsBinding && (currentMethod === "FLIP" || currentMethod === "MLS-MPM")) {
                    syncFluidControls(controlsBinding, { pageDiagnostics: { method: currentMethod, requiredPages, capacity } });
                } else {
                    controls.setPagedGridStatus(formatFluidPageDiagnostics(requiredPages, capacity), false);
                }
            },
            onPagedGridOverflow: currentMethod === "FLIP" ? overflow("FLIP") : currentMethod === "MLS-MPM" ? overflow("MLS-MPM") : undefined,
        };
    }

    function createConfiguredSim(
        count: number,
        radius: number,
        boundsMin: [number, number, number],
        boundsMax: [number, number, number],
        positions?: Float32Array
    ): FluidSimulation {
        return createFluidSimulation(engine, configuredSimulationOptions(count, radius, boundsMin, boundsMax, positions));
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
        const dx =
            currentMethod === "FLIP"
                ? simulationCellSize()
                : fluidSimulationCellSize(currentMethod, { physicsParticleSize: fluidPhysicsScale, samplingType: "mesh", particleRadius: radius });
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
        inst.collisionRefresh = collision.refreshPlacements;
        setFluidSimulationSceneSdf(sim, adoptFluidSceneSdf(engine, collision.spec));
        setFluidSimulationProfiler(sim, profiler);
        applyFoamToSim(sim);
        inst.sim = sim;
        inst.simulationCenter = gridCenter;
        inst.count = count;
        inst.radius = sim.particleRadius;
        syncSimulationCollection();
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
    function bakeFill(ox: number, oy: number, oz: number, result: FluidMeshParticleSamplingResult): SampledFill {
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
        for (const cx of [bMin[0], bMax[0]]) {
            for (const cy of [bMin[1], bMax[1]]) {
                for (const cz of [bMin[2], bMax[2]]) {
                    const dx = cx - hit[0];
                    const dy = cy - hit[1];
                    const dz = cz - hit[2];
                    maxD = Math.max(maxD, Math.sqrt(dx * dx + dy * dy + dz * dz));
                }
            }
        }
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
            writeFluidSimulationPositions(inst.sim, scratch);
        }
    }

    // ── Impulse / fade force fields ──────────────────────────────────────────
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
        updateFluidImpulseForce(inst.impulseForce, {
            center: inst.liquefyState.hit,
            radius: impulseRadius > 0 ? impulseRadius : Math.max(inst.maxR, inst.radius * 8, 1),
            intensity: impulseIntensity,
            direction: impulseDir,
            fallbackDirection: inst.shotDir,
        });
        setFluidSimulationForceField(inst.sim, inst.impulseForce);
        inst.impulseRemaining = 0.35;
    }

    function beginFade(inst: Instance): void {
        // Fade the blob out in place by ramping its per-particle alpha 1→0 (handled in the
        // per-frame loop); the sim keeps settling under gravity meanwhile. No sink force.
        inst.phase = "fading";
        inst.fadeElapsed = 0;
        inst.impulseRemaining = 0;
        if (inst.sim) setFluidSimulationForceField(inst.sim, null);
    }

    const bumpUbo = (inst: Instance): void => {
        for (const m of inst.materials) m._uboVersion++;
    };
    const setVisible = (inst: Instance, v: boolean): void => {
        for (const m of inst.meshes) setMeshVisible(m, v);
    };

    function disposeInstanceSim(inst: Instance): void {
        if (inst.sim) setFluidSimulationForceField(inst.sim, null);
        const simulation = inst.sim;
        inst.sim = null;
        inst.simulationCenter = null;
        syncSimulationCollection();
        if (simulation) {
            disposeFluidSimulation(simulation);
        }
        inst.collisionBuffer?.destroy();
        inst.collisionBuffer = null;
        inst.collisionRefresh = null;
        if (inst.colorChannel) {
            disposeFluidParticleChannel(inst.colorChannel);
            inst.colorChannel = null;
        }
        inst.phase = "gone";
        inst.count = 0;
    }

    function manualParticleCapacityForMethod(method: "PBF" | "FLIP" | "MLS-MPM" | "PB-MPM", worldFlow = flowInWorldSpace(activeFlow)): number {
        // Per-particle volume from the shared resolver (FLIP marker volume or particle diameter
        // cubed). Capacity floors it to 1e-6, so the profile's 0 floor is equivalent here.
        const particleVolume = manualFluidConfigForMethod(method).particleVolume;
        const requested = method === "FLIP" ? (flipParticleCapacityRequest ?? 0) : (particleCapacityRequestByMethod.get(method) ?? 0);
        return Math.min(MAX_TOTAL, fluidSimulationParticleCapacity(method, requested, worldFlow, particleVolume));
    }
    const manualParticleCapacity = (worldFlow = flowInWorldSpace(activeFlow)): number => manualParticleCapacityForMethod(currentMethod, worldFlow);

    function projectedParticleUsage(): { active: number; capacity: number; gpuBytes: number } {
        const worldFlow = flowInWorldSpace(activeFlow);
        const capacity = manualParticleCapacity(worldFlow);
        const config = manualFluidConfig();
        const initialPlan = planFluidInitialState({
            particleCapacity: capacity,
            flow: worldFlow,
            particleVolume: config.particleVolume,
            bounds: simulationBounds(),
            deriveInitialCount: currentMethod === "FLIP",
        });
        const active = initialPlan.activeCount;
        const bounds = simulationBounds();
        const options: FluidSimulationOptions = {
            ...configuredSimulationOptions(capacity, config.particleRadius, bounds.min, bounds.max),
            flow: worldFlow,
            foam: controls.getValues().foam.enabled ? currentFoamConfig() : null,
        };
        const memory = projectFluidControlsMemory([{ options, activeCount: active }], fluidDeviceLimits);
        return { active, capacity, gpuBytes: memory.steadyBytes };
    }

    function flipInitialEmitterParticleCounts(): ReadonlyMap<string, number> {
        const counts = new Map(activeFlow.emitters.map((emitter) => [emitter.id, 0]));
        if (currentMethod !== "FLIP") {
            return counts;
        }
        const worldFlow = flowInWorldSpace(activeFlow);
        const capacity = manualParticleCapacity(worldFlow);
        const resetCounts =
            manualRun?.method === "FLIP" && manualRun.configurationSignature === manualConfigurationSignature() && manualRun.sim.count === capacity
                ? manualRun.sim.initialEmitterParticleCounts
                : undefined;
        const projectedCounts =
            resetCounts ??
            planFluidInitialState({
                particleCapacity: capacity,
                flow: worldFlow,
                particleVolume: flipDiscretization().markerVolume,
                bounds: simulationBounds(),
                deriveInitialCount: true,
            }).emitterCounts;
        if (projectedCounts) {
            for (const emitter of activeFlow.emitters) {
                counts.set(emitter.id, projectedCounts.get(emitter.id) ?? 0);
            }
        }
        return counts;
    }

    function refreshManualParticleUsage(): void {
        if (simulationType !== "fluid") {
            return;
        }
        const active = manualRun?.sim.activeCount ?? manualRun?.sim.count ?? 0;
        const total = manualRun?.sim.count ?? 0;
        const gpuBytes = manualRun?.sim.gpuBytes ?? 0;
        const projection = projectedParticleUsage();
        const pending =
            !manualRun ||
            manualRun.method !== currentMethod ||
            manualRun.configurationSignature !== manualConfigurationSignature() ||
            projection.capacity !== total ||
            projection.gpuBytes !== gpuBytes;
        controls.setParticleUsage(
            active,
            total,
            gpuBytes,
            pending ? projection.active : undefined,
            pending ? projection.capacity : undefined,
            pending ? projection.gpuBytes : undefined
        );
        const polygonSurfaces = polygonSurfaceRenderingEnabled() ? getFluidSimulationCollectionDiagnostics(simulationCollection).polygons : [];
        const polygonTriangleCount =
            polygonSurfaces.length > 0 && polygonSurfaces.every((surface) => surface.triangleCount !== undefined)
                ? polygonSurfaces.reduce((sum, surface) => sum + (surface.triangleCount ?? 0), 0)
                : undefined;
        controls.setPolygonTriangleCount(polygonTriangleCount, polygonSurfaces.length > 0);
        if (polygonSurfaces[0]) {
            canvas.dataset.polygonReconstructionMultiplier = String(polygonSurfaces[0].reconstructionMultiplier);
        } else {
            delete canvas.dataset.polygonReconstructionMultiplier;
        }
        canvas.dataset.activeParticleCount = String(active);
        canvas.dataset.simulationGpuBytes = String(gpuBytes);
        if (polygonTriangleCount === undefined) {
            delete canvas.dataset.polygonTriangleCount;
        } else {
            canvas.dataset.polygonTriangleCount = String(polygonTriangleCount);
        }
        canvas.dataset.gridRestartPending = String(pending);
        canvas.dataset.restartParticleCount = String(projection.active);
        canvas.dataset.restartParticleCapacity = String(projection.capacity);
        canvas.dataset.restartSimulationGpuBytes = String(projection.gpuBytes);
        flowEditor?.refreshComputedValues();
    }

    function startManualSimulation(): void {
        const worldFlow = flowInWorldSpace(activeFlow);
        const enabled = worldFlow.emitters.filter((emitter) => emitter.enabled);
        if (enabled.length === 0) {
            status.textContent = "Enable at least one emitter before starting.";
            return;
        }
        manualStepCount = 0;
        const center: [number, number, number] = [...gridPosition];
        const { min: boundsMin, max: boundsMax } = simulationBounds();
        const capacity = manualParticleCapacity(worldFlow);
        const collision = createNeighborhoodSdf(center, null);
        const previous = manualRun;
        const sceneSdf = adoptFluidSceneSdf(engine, collision.spec);
        const options: FluidSimulationOptions = {
            ...configuredSimulationOptions(capacity, fluidParticleRadius(), boundsMin, boundsMax),
            flow: worldFlow,
            sceneSdf,
            profiler,
            foam: controls.getValues().foam.enabled ? currentFoamConfig() : null,
        };
        let prepared: ReturnType<typeof prepareFluidReconfiguration> | null = null;
        let colorChannel: FluidParticleChannel | null = null;
        let sim: FluidSimulation;
        try {
            prepared = previous ? prepareFluidReconfiguration(previous.sim, options, false) : null;
            colorChannel = createFluidParticleChannel(engine, {
                label: "liq-manual-color",
                capacity,
                components: 4,
            });
            sim = prepared ? (commitFluidReconfiguration(prepared), previous!.sim) : createFluidSimulation(engine, options);
            resetFluidSimulation(sim);
        } catch (error) {
            if (prepared) cancelFluidReconfiguration(prepared);
            if (colorChannel) disposeFluidParticleChannel(colorChannel);
            collision.buffer?.destroy();
            throw error;
        }
        manualRun = {
            sim,
            collisionBuffer: collision.buffer,
            collisionRefresh: collision.refreshPlacements,
            colorChannel,
            center,
            method: currentMethod,
            configurationSignature: manualConfigurationSignature(),
        };
        fillManualColor(manualRun);
        syncSimulationCollection();
        if (previous) {
            if (previous.collisionBuffer) retireGpuResources(engine, () => previous.collisionBuffer!.destroy());
            disposeFluidParticleChannel(previous.colorChannel);
        }
        canvas.dataset.lastCollisionPrimitiveCount = String(collision.count);
        setStatus();
        refreshManualParticleUsage();
    }

    function reconfigureRunningSimulations(): void {
        const requests: FluidSimulationReconfigurationRequest[] = [];
        const gridDelta: [number, number, number] = [
            gridPosition[0] - committedGridPosition[0],
            gridPosition[1] - committedGridPosition[1],
            gridPosition[2] - committedGridPosition[2],
        ];
        const nextInstanceCenters = new Map<Instance, [number, number, number]>();
        const foam = controls.getValues().foam.enabled ? currentFoamConfig() : null;
        const prepare = (simulation: FluidSimulation, center: readonly [number, number, number], particleRadius: number, particleCount?: number, flow?: FluidFlowConfig): void => {
            requests.push({
                simulation,
                preserveState: true,
                updates: {
                    method: currentMethod as FluidSimulationOptions["method"],
                    ...(particleCount !== undefined ? { particleCount } : {}),
                    bounds: simulationBounds(center),
                    physicsScale: fluidPhysicsScale,
                    compatibilityProfile: "aquanova",
                    samplingType: simulationType,
                    particleRadius,
                    semantics: simulationSemantics,
                    physics: physValues[currentMethod],
                    gridResolution: currentMethod === "FLIP" ? flipGridResolution : undefined,
                    markersPerCell: currentMethod === "FLIP" ? flipMarkersPerCell : undefined,
                    material: currentMethod === "PB-MPM" ? currentMaterial : undefined,
                    activeBlocks: mpmActiveBlocks,
                    pagedGrid: currentMethod === "FLIP" ? flipPagedGrid : mpmPagedGrid,
                    pagedGridMaxPages: currentMethod === "FLIP" ? flipPagedGridMaxPages : mpmPagedGridMaxPages,
                    fusedBlockDiscovery: mpmFusedBlockDiscovery,
                    foam,
                    ...(flow ? { flow } : {}),
                },
            });
        };
        let prepared: ReturnType<typeof prepareFluidCollectionReconfiguration> | null = null;
        try {
            for (const instance of instances) {
                if (instance.sim && instance.simulationCenter) {
                    const center: [number, number, number] = [
                        instance.simulationCenter[0] + gridDelta[0],
                        instance.simulationCenter[1] + gridDelta[1],
                        instance.simulationCenter[2] + gridDelta[2],
                    ];
                    nextInstanceCenters.set(instance, center);
                    prepare(instance.sim, center, instance.radius);
                }
            }
            if (manualRun) {
                const flow = flowInWorldSpace(activeFlow);
                prepare(manualRun.sim, gridPosition, fluidParticleRadius(), manualParticleCapacity(flow), flow);
            }
            prepared = prepareFluidCollectionReconfiguration(requests);
            canvas.dataset.simulationTransitionBytes = String(prepared.transitionPeakBytes);
            canvas.dataset.simulationSteadyBytes = String(prepared.steadyBytes);
            commitFluidCollectionReconfiguration(prepared);
        } catch (error) {
            if (prepared) {
                cancelFluidCollectionReconfiguration(prepared);
            }
            throw error;
        }
        if (manualRun) {
            manualRun.center = [...gridPosition];
            manualRun.method = currentMethod;
            manualRun.configurationSignature = manualConfigurationSignature();
        }
        for (const [instance, center] of nextInstanceCenters) {
            instance.simulationCenter = center;
            instance.radius = instance.sim?.particleRadius ?? instance.radius;
        }
        committedGridPosition = [...gridPosition];
        syncSimulationCollection();
    }

    function finishAquanovaControlPlan(plan: ReturnType<typeof applyFluidControls>): void {
        if (plan.reconfigure) {
            const gridDelta: [number, number, number] = [
                gridPosition[0] - committedGridPosition[0],
                gridPosition[1] - committedGridPosition[1],
                gridPosition[2] - committedGridPosition[2],
            ];
            if (manualRun) {
                manualRun.center = [...gridPosition];
                manualRun.method = currentMethod;
                manualRun.configurationSignature = manualConfigurationSignature();
            }
            for (const instance of instances) {
                if (instance.sim && instance.simulationCenter) {
                    instance.simulationCenter = [
                        instance.simulationCenter[0] + gridDelta[0],
                        instance.simulationCenter[1] + gridDelta[1],
                        instance.simulationCenter[2] + gridDelta[2],
                    ];
                    instance.radius = instance.sim.particleRadius;
                }
            }
            committedGridPosition = [...gridPosition];
            canvas.dataset.simulationTransitionBytes = String(controlsBinding?.memory.transitionPeakBytes ?? 0);
            canvas.dataset.simulationSteadyBytes = String(controlsBinding?.memory.steadyBytes ?? 0);
            syncSimulationCollection();
        }
        if (plan.restartRequired) {
            refreshManualParticleUsage();
        }
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
    type WorkerMsg = ParticleFillWorkerResponse;
    interface PoolWorker {
        worker: Worker;
        pending: number;
    }
    const workerPool: PoolWorker[] = [];
    const failSample = (entry: { inst: Instance }, error: FluidMeshSamplingErrorInfo): void => {
        // eslint-disable-next-line no-console
        console.error("[aquanova-fluid-sim] mesh particle sampling failed", error);
        entry.inst.sampling = false;
        leaveShot(entry.inst);
        stopWriggle(entry.inst);
        resumeCancelledAnimation(entry.inst);
        setStatus();
    };
    const reportSampleWarnings = (warnings: readonly FluidMeshSamplingWarning[]): void => {
        for (const warning of warnings) {
            // eslint-disable-next-line no-console
            console.warn("[aquanova-fluid-sim] mesh particle sampling warning", warning);
        }
    };
    const onSampleMessage = (ev: MessageEvent<WorkerMsg>): void => {
        const { id } = ev.data;
        const entry = pendingSamples.get(id);
        pendingSamples.delete(id);
        if (!entry) return;
        if (ev.data.error) {
            failSample(entry, ev.data.error);
            return;
        }
        const { positions, uvs, texIndices, count, radius, shell, boundsMin, boundsMax, warnings } = ev.data;
        if (count === 0) {
            failSample(entry, { code: "EMPTY_RESULT", message: "Sampling worker returned an empty success result.", requestedStrategy: fillStrategy });
            return;
        }
        reportSampleWarnings(warnings);
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
            try {
                const result = sampleFluidMeshParticles({
                    positions: geom.positions,
                    indices: geom.indices,
                    radius: radiusValue,
                    mode: modeValue,
                    surfaceOnly: inst.surfaceOnly,
                    strategy: fillStrategy,
                    spacing: FILL_SPACING,
                });
                reportSampleWarnings(result.warnings);
                applySample(inst, id, h, { ...bakeFill(geom.ox, geom.oy, geom.oz, result), uvs: null, texIndices: null });
            } catch (error) {
                failSample({ inst }, fluidMeshSamplingErrorInfo(error, fillStrategy));
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
        let colorBuffer = fillInstanceColor(inst, fill.uvs ?? null, fill.texIndices ?? null, fill.count);
        // Promote the albedo to the foe's CURRENTLY RENDERED colours (lit + tone-mapped).
        const ls = litScene();
        if (ls) {
            const lit = buildLitParticleColors(device, fill.count, fill.positions, colorBuffer, ls);
            colorBuffer.destroy();
            colorBuffer = lit;
        }
        inst.colorChannel = adoptFluidParticleChannel(engine, colorBuffer, {
            capacity: fill.count,
            components: 4,
            ownsBuffer: true,
            label: `liq-color-${inst.key}`,
        });
        syncSimulationCollection();
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
        const retiredSimulations = runningSims();
        const retiredChannels = [...instances.flatMap((inst) => (inst.colorChannel ? [inst.colorChannel] : [])), ...(manualRun ? [manualRun.colorChannel] : [])];
        const retiredCollisionBuffers = [
            ...instances.flatMap((inst) => (inst.collisionBuffer ? [inst.collisionBuffer] : [])),
            ...(manualRun?.collisionBuffer ? [manualRun.collisionBuffer] : []),
        ];
        manualRun = null;
        for (const inst of instances) {
            inst.sim = null;
            inst.collisionBuffer = null;
            inst.collisionRefresh = null;
            inst.colorChannel = null;
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
        syncSimulationCollection();
        committedGridPosition = [...gridPosition];
        for (const simulation of retiredSimulations) disposeFluidSimulation(simulation);
        for (const channel of retiredChannels) disposeFluidParticleChannel(channel);
        for (const buffer of retiredCollisionBuffers) buffer.destroy();
        // Restoring the meshes is not enough for a `dynamic` prop: its Havok body keeps whatever pose it
        // toppled into and would re-impose it on the next step. Teleport each body home, kill its
        // velocities, and sync the display root so there's no one-frame flash before the next step.
        // Props that never moved are skipped — they are asleep, and teleporting them would wake them for
        // nothing (an awake door immediately sags onto the chunk floor).
        if (physWorld) {
            for (const d of dynDisplays) {
                const q = d.disp.rotationQuaternion;
                const off = Math.max(Math.abs(d.disp.position.x - d.rest[0]), Math.abs(d.disp.position.y - d.rest[1]), Math.abs(d.disp.position.z - d.rest[2]));
                const rotationOff = Math.min(
                    Math.max(Math.abs(q.x - d.restRotation[0]), Math.abs(q.y - d.restRotation[1]), Math.abs(q.z - d.restRotation[2]), Math.abs(q.w - d.restRotation[3])),
                    Math.max(Math.abs(q.x + d.restRotation[0]), Math.abs(q.y + d.restRotation[1]), Math.abs(q.z + d.restRotation[2]), Math.abs(q.w + d.restRotation[3]))
                );
                if (off < 1e-4 && rotationOff < 1e-4) continue;
                setPhysicsBodyTransform(
                    physWorld,
                    d.body,
                    { x: d.rest[0], y: d.rest[1], z: d.rest[2] },
                    { x: d.restRotation[0], y: d.restRotation[1], z: d.restRotation[2], w: d.restRotation[3] }
                );
                setPhysicsBodyLinearVelocity(physWorld, d.body, { x: 0, y: 0, z: 0 });
                setPhysicsBodyAngularVelocity(physWorld, d.body, { x: 0, y: 0, z: 0 });
                d.disp.position.set(d.rest[0], d.rest[1], d.rest[2]);
                d.disp.rotationQuaternion.set(d.restRotation[0], d.restRotation[1], d.restRotation[2], d.restRotation[3]);
            }
        }
        liveShots = 0;
        syncSimulationCollection();
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
    const installClickFeedback = (buttons: readonly HTMLButtonElement[]): void => {
        for (const button of buttons) {
            button.dataset.clickFeedback = "true";
            button.addEventListener("click", () => {
                button.animate(
                    [
                        { boxShadow: "0 0 0 2px rgba(76, 220, 255, 0.95), 0 0 14px rgba(76, 220, 255, 0.8)", filter: "brightness(1.55)" },
                        { boxShadow: "0 0 0 1px rgba(76, 220, 255, 0.4), 0 0 5px rgba(76, 220, 255, 0.3)", filter: "brightness(1.15)", offset: 0.65 },
                        { boxShadow: "none", filter: "brightness(1)" },
                    ],
                    { duration: 360, easing: "ease-out" }
                );
            });
        }
    };
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
    fluidSimNameInput.id = "aquanova-fluid-sim-name";
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
    fluidSimUpdateButton.id = "aquanova-fluid-sim-update";
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
        refreshParticleCountControl();
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
    ] as [FluidMeshSamplingStrategy, string][]) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = label;
        fillSelect.append(opt);
    }
    fillSelect.value = fillStrategy;
    fillSelect.onchange = () => {
        fillStrategy = fillSelect.value as FluidMeshSamplingStrategy;
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
        dirHint.textContent = impulseDir[0] * impulseDir[0] + impulseDir[1] * impulseDir[1] + impulseDir[2] * impulseDir[2] > 1e-12 ? "" : "shot ray";
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

    const collisionDebugInput = document.createElement("input");
    collisionDebugInput.type = "checkbox";
    collisionDebugInput.id = "liq-collision-debug";
    collisionDebugInput.checked = collisionDebugVisible;
    collisionDebugInput.style.cssText = "margin-right:6px;vertical-align:middle;";
    collisionDebugInput.onchange = () => {
        collisionDebugVisible = collisionDebugInput.checked;
        updateCollisionDebugOverlay();
    };
    const collisionDebugRow = document.createElement("label");
    collisionDebugRow.style.cssText = "display:block;margin-bottom:8px;color:#b6c4d6;cursor:pointer;";
    collisionDebugRow.title = "Draw every effective fluid collision primitive as a translucent overlay through the ship.";
    collisionDebugRow.append(collisionDebugInput, document.createTextNode("Show fluid collision shapes"));

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
        configureFluidSimulationRenderLayer(surfaceTask, { useParticleColor: useMeshColors });
        syncSimulationCollection();
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

    function switchMethod(method: string, forceUiSync = false): void {
        if (method === currentMethod && !forceUiSync) return;
        if (method !== "PBF" && method !== "FLIP" && method !== "MLS-MPM" && method !== "PB-MPM") {
            throw new Error(`Unsupported fluid method "${method}".`);
        }
        currentMethod = method;
        canvas.dataset.method = method;
        controls.setMethod(method);
        controls.rebuildPhysics(method);
        controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
        // Reflect the newly-selected method's own paging state (FLIP and MLS-MPM are independent).
        if (isPagingCapableMethod(method)) {
            syncPagingControls();
        }
        refreshFlipParticleCapacity();
        refreshParticleCountControl();
        refreshFlowControls();
        refreshPhysicsParamVisibility();
        syncPolygonSurfaceRendering();
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
    const gridBoundsSolid = createSolidGridBounds(engine, "aquanova-fluid-sim-grid-bounds-solid");
    for (const face of gridBoundsSolid) {
        addMeshToTask(foeTask, face);
        setMeshVisible(face, false);
    }

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
    let meshGizmoTarget: SceneNode | null = null;
    let meshGizmoSelectionPending = false;
    let meshGizmoAnchorLocal: [number, number, number] | null = null;
    let meshGizmoPivotWorld: [number, number, number] | null = null;
    let meshGizmoRoots: SceneNode[] = [];
    let meshPositionGizmoRootCount = 0;
    const meshGizmoScale = 0.5;
    const meshPositionGizmoScale = meshGizmoScale * 1.35;
    const transformPoint = (matrix: ArrayLike<number>, point: readonly [number, number, number]): [number, number, number] => [
        matrix[0]! * point[0] + matrix[4]! * point[1] + matrix[8]! * point[2] + matrix[12]!,
        matrix[1]! * point[0] + matrix[5]! * point[1] + matrix[9]! * point[2] + matrix[13]!,
        matrix[2]! * point[0] + matrix[6]! * point[1] + matrix[10]! * point[2] + matrix[14]!,
    ];
    onBeforeRender(gridGizmoLayer.scene, () => {
        if (!meshGizmoTarget || !meshGizmoAnchorLocal) return;
        const [worldX, worldY, worldZ] = transformPoint(meshGizmoTarget.worldMatrix, meshGizmoAnchorLocal);
        meshGizmoPivotWorld = [worldX, worldY, worldZ];
        for (let i = 0; i < meshGizmoRoots.length; i++) {
            const root = meshGizmoRoots[i]!;
            root.position.set(worldX, worldY, worldZ);
            if (i < meshPositionGizmoRootCount) {
                // Keep translation arrow tips outside the rotation rings so both controls stay pickable.
                root.scaling.set(root.scaling.x * meshPositionGizmoScale, root.scaling.y * meshPositionGizmoScale, root.scaling.z * meshPositionGizmoScale);
            } else {
                root.scaling.set(root.scaling.x * meshGizmoScale, root.scaling.y * meshGizmoScale, root.scaling.z * meshGizmoScale);
            }
        }
        updateMeshGizmoInfo();
    });
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
    meshPositionGizmoRootCount = meshPositionSubGizmos.length;
    meshGizmoRoots = [...meshPositionSubGizmos, ...meshRotationSubGizmos].map((gizmo) => gizmo.root);
    const setGizmoMeshesVisible = (visible: boolean, gizmos: ReadonlyArray<{ _visibleMeshes: Mesh[] }>): void => {
        for (const gizmo of gizmos) {
            for (const mesh of gizmo._visibleMeshes) {
                setMeshVisible(mesh, visible);
            }
        }
    };
    const selectedFlowObject = (kind: FluidFlowObjectKind) => flowEditor?.getSelected(kind);
    const meshGizmoId = (target: SceneNode | null): string | null =>
        target ? (instancesByEditableTarget.get(target)?.[0]?.key ?? (isMeshNode(target) ? instanceOf(target)?.key : undefined) ?? target.name) : null;
    const editableTargetBounds = (target: SceneNode, worldToSpace?: ArrayLike<number>): { min: [number, number, number]; max: [number, number, number] } | null => {
        const targetInstances = instancesByEditableTarget.get(target);
        const seen = new Set<Mesh>();
        const min: [number, number, number] = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
        const max: [number, number, number] = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
        const targetMeshes = targetInstances?.length ? targetInstances.flatMap((instance) => instance.meshes) : isMeshNode(target) ? [target] : [];
        for (const mesh of targetMeshes) {
            if (seen.has(mesh) || !mesh.boundMin || !mesh.boundMax) continue;
            seen.add(mesh);
            const matrix = mesh.worldMatrix;
            for (const x of [mesh.boundMin[0], mesh.boundMax[0]]) {
                for (const y of [mesh.boundMin[1], mesh.boundMax[1]]) {
                    for (const z of [mesh.boundMin[2], mesh.boundMax[2]]) {
                        const world = transformPoint(matrix, [x, y, z]);
                        const point = worldToSpace ? transformPoint(worldToSpace, world) : world;
                        for (let axis = 0; axis < 3; axis++) {
                            min[axis] = Math.min(min[axis]!, point[axis]!);
                            max[axis] = Math.max(max[axis]!, point[axis]!);
                        }
                    }
                }
            }
        }
        return seen.size > 0 ? { min, max } : null;
    };
    const editableTargetLocalCenter = (target: SceneNode): [number, number, number] | null => {
        const targetInverse = mat4Invert(target.worldMatrix);
        if (!targetInverse) return null;
        const bounds = editableTargetBounds(target, targetInverse);
        return bounds ? [(bounds.min[0] + bounds.max[0]) * 0.5, (bounds.min[1] + bounds.max[1]) * 0.5, (bounds.min[2] + bounds.max[2]) * 0.5] : null;
    };
    const formatMeshGizmoVector = (values: readonly number[]): string => values.map((value) => value.toFixed(2)).join(", ");
    const meshGizmoWorldPosition = (target: SceneNode | null): [number, number, number] | null =>
        target ? transformPoint((positionTargetByEditableTarget.get(target) ?? target).worldMatrix, [0, 0, 0]) : null;
    const updateMeshGizmoInfo = (): void => {
        const target = meshGizmoTarget;
        if (!target) {
            meshGizmoInfo.hidden = true;
            return;
        }
        const position = meshGizmoWorldPosition(target)!;
        const bounds = editableTargetBounds(target);
        meshGizmoNameLine.textContent = `Object ${meshGizmoId(target) ?? target.name}`;
        meshGizmoPositionLine.textContent = `World position (m): ${formatMeshGizmoVector(position)}`;
        meshGizmoSizeLine.textContent = bounds
            ? `AABB size (m): ${[bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]].map((value) => value.toFixed(2)).join(" x ")}`
            : "AABB size unavailable";
        meshGizmoInfo.hidden = false;
    };
    const syncMeshTransformSideEffects = (): void => {
        if (!meshGizmoTarget) return;
        const display = dynamicDisplayByTarget.get(meshGizmoTarget);
        if (display && physWorld) {
            const { position, rotationQuaternion } = display.disp;
            setPhysicsBodyTransform(
                physWorld,
                display.body,
                { x: position.x, y: position.y, z: position.z },
                { x: rotationQuaternion.x, y: rotationQuaternion.y, z: rotationQuaternion.z, w: rotationQuaternion.w }
            );
            display.rest = [position.x, position.y, position.z];
            display.restRotation = [rotationQuaternion.x, rotationQuaternion.y, rotationQuaternion.z, rotationQuaternion.w];
        }
        const movedPlacementIds = new Set<string>();
        for (const placement of shipCollisionPlacements) {
            let node: SceneNode | null = placement.node;
            while (node) {
                if (node === meshGizmoTarget) {
                    movedPlacementIds.add(placement.id);
                    break;
                }
                node = node.parent as SceneNode | null;
            }
        }
        if (movedPlacementIds.size === 0) return;
        for (const instance of instances) {
            if (instance.sim) instance.collisionRefresh?.(movedPlacementIds);
        }
        manualRun?.collisionRefresh(movedPlacementIds);
    };
    const syncMeshPositionFromGizmo = (): void => {
        if (meshGizmoTarget && meshGizmoAnchorLocal) {
            meshGizmoPivotWorld = transformPoint(meshGizmoTarget.worldMatrix, meshGizmoAnchorLocal);
        }
        syncMeshTransformSideEffects();
    };
    const syncMeshRotationFromGizmo = (): void => {
        if (meshGizmoTarget && meshGizmoAnchorLocal && meshGizmoPivotWorld) {
            const currentPivot = transformPoint(meshGizmoTarget.worldMatrix, meshGizmoAnchorLocal);
            const parentInverse = meshGizmoTarget.parent ? mat4Invert(meshGizmoTarget.parent.worldMatrix) : null;
            if (parentInverse) {
                const desiredLocal = transformPoint(parentInverse, meshGizmoPivotWorld);
                const currentLocal = transformPoint(parentInverse, currentPivot);
                meshGizmoTarget.position.set(
                    meshGizmoTarget.position.x + desiredLocal[0] - currentLocal[0],
                    meshGizmoTarget.position.y + desiredLocal[1] - currentLocal[1],
                    meshGizmoTarget.position.z + desiredLocal[2] - currentLocal[2]
                );
            } else if (!meshGizmoTarget.parent) {
                meshGizmoTarget.position.set(
                    meshGizmoTarget.position.x + meshGizmoPivotWorld[0] - currentPivot[0],
                    meshGizmoTarget.position.y + meshGizmoPivotWorld[1] - currentPivot[1],
                    meshGizmoTarget.position.z + meshGizmoPivotWorld[2] - currentPivot[2]
                );
            }
        }
        syncMeshTransformSideEffects();
    };
    for (const gizmo of meshPositionSubGizmos) {
        gizmo.onPositionChanged.add(syncMeshPositionFromGizmo);
    }
    for (const gizmo of meshRotationSubGizmos) {
        gizmo.onRotationChanged.add(syncMeshRotationFromGizmo);
    }
    const setMeshGizmoTarget = (target: SceneNode | null, anchorWorld?: readonly [number, number, number]): void => {
        meshGizmoTarget = target;
        const inverse = target && anchorWorld ? mat4Invert(target.worldMatrix) : null;
        meshGizmoAnchorLocal =
            (target ? editableTargetLocalCenter(target) : null) ??
            (inverse && anchorWorld
                ? [
                      inverse[0]! * anchorWorld[0] + inverse[4]! * anchorWorld[1] + inverse[8]! * anchorWorld[2] + inverse[12]!,
                      inverse[1]! * anchorWorld[0] + inverse[5]! * anchorWorld[1] + inverse[9]! * anchorWorld[2] + inverse[13]!,
                      inverse[2]! * anchorWorld[0] + inverse[6]! * anchorWorld[1] + inverse[10]! * anchorWorld[2] + inverse[14]!,
                  ]
                : null);
        meshGizmoPivotWorld = target && meshGizmoAnchorLocal ? transformPoint(target.worldMatrix, meshGizmoAnchorLocal) : null;
        attachPositionGizmoToNode(meshPositionGizmo, target);
        attachRotationGizmoToNode(meshRotationGizmo, target);
        setGizmoMeshesVisible(!!target, meshPositionSubGizmos);
        setGizmoMeshesVisible(!!target, meshRotationSubGizmos);
        canvas.dataset.meshGizmo = meshGizmoId(target) ?? "";
        updateMeshGizmoInfo();
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
        if (manualRun) setFluidSimulationFlow(manualRun.sim, flowInWorldSpace(activeFlow));
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
        const showSolidFaces = showGridBounds && showGridBoundsSolid;
        for (const face of gridBoundsSolid) {
            face.position.set(gridPosition[0], gridPosition[1], gridPosition[2]);
            face.scaling.set(gridSize[0], gridSize[1], gridSize[2]);
            setMeshVisible(face, showSolidFaces);
        }
        if (gridBoundsOverlayVisible !== showSolidFaces) {
            gridBoundsOverlayVisible = showSolidFaces;
            invalidatePhaseTaskBundles();
        }
        setMeshVisible(gridBoundsWireframe, showGridGizmo || (showGridBounds && !showGridBoundsSolid));
        syncFlowWireframe("emitter");
        syncFlowWireframe("sink");
        syncFlowGizmo();
        canvas.dataset.gridPosition = gridPosition.join(",");
        canvas.dataset.gridSize = gridSize.join(",");
        canvas.dataset.showGridBounds = String(showGridBounds);
        canvas.dataset.showGridBoundsSolid = String(showGridBoundsSolid);
    };
    const syncGridGizmo = (): void => {
        attachPositionGizmoToNode(gridPositionGizmo, showGridGizmo ? gridBoundsWireframe : null);
        attachScaleGizmoToNode(gridScaleGizmo, showGridGizmo ? gridBoundsWireframe : null);
        setGizmoMeshesVisible(showGridGizmo, gridPositionSubGizmos);
        setGizmoMeshesVisible(showGridGizmo, gridScaleSubGizmos);
    };
    function setGridSettings(position: readonly number[], size: readonly number[]): string | void {
        if (controlsBinding) {
            const nextPosition: [number, number, number] = [position[0]!, position[1]!, position[2]!];
            const nextSize: [number, number, number] = [size[0]!, size[1]!, size[2]!];
            try {
                const plan = applyFluidGridSettings(controlsBinding, nextPosition, nextSize);
                finishAquanovaControlPlan(plan);
                canvas.dataset.gridPosition = gridPosition.join(",");
                canvas.dataset.gridSize = gridSize.join(",");
                controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
                syncGridBoundsWireframe();
                syncGridGizmo();
                refreshManualParticleUsage();
                return;
            } catch (error) {
                controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
                syncGridBoundsWireframe();
                syncGridGizmo();
                return error instanceof Error ? error.message : String(error);
            }
        }
        const changed = position.some((value, axis) => value !== gridPosition[axis]) || size.some((value, axis) => value !== gridSize[axis]);
        const previousPosition: [number, number, number] = [...gridPosition];
        const previousSize: [number, number, number] = [...gridSize];
        for (let axis = 0; axis < 3; axis++) {
            gridPosition[axis] = position[axis]!;
            gridSize[axis] = Math.max(0.1, size[axis]!);
        }
        canvas.dataset.gridPosition = gridPosition.join(",");
        canvas.dataset.gridSize = gridSize.join(",");
        controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
        syncGridBoundsWireframe();
        if (changed) {
            if (!applyingImportedPreset) {
                try {
                    reconfigureRunningSimulations();
                } catch (error) {
                    gridPosition.splice(0, 3, ...previousPosition);
                    gridSize.splice(0, 3, ...previousSize);
                    controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
                    syncGridBoundsWireframe();
                    throw error;
                }
            }
            refreshManualParticleUsage();
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

    function aquanovaBindingValues(values: Readonly<FluidControlValues>, changedKeys: readonly (keyof FluidControlValues)[]): Readonly<FluidControlValues> {
        if (!changedKeys.includes("method") || changedKeys.some((key) => key !== "method")) {
            return values;
        }
        if (values.method !== "PBF" && values.method !== "FLIP" && values.method !== "MLS-MPM" && values.method !== "PB-MPM") {
            return values;
        }
        const method = values.method;
        return {
            ...values,
            count: manualParticleCapacityForMethod(method),
            activeBlocks: method === "MLS-MPM" ? mpmActiveBlocks || mpmPagedGrid : false,
            pagedGrid: method === "FLIP" ? flipPagedGrid : method === "MLS-MPM" ? mpmPagedGrid : false,
            pagedGridMaxPages: method === "FLIP" ? flipPagedGridMaxPages : method === "MLS-MPM" ? mpmPagedGridMaxPages : values.pagedGridMaxPages,
            fusedBlockDiscovery: method === "MLS-MPM" ? mpmFusedBlockDiscovery : false,
        };
    }

    const controls = createFluidControlsPanel({
        hideParticles: false,
        hideMethod: false,
        hideRenderAsSpheres: true,
        hideContainerToggle: true,
        hideFoam: false,
        hidePhysics: false,
        hideDebug: false,
        hideGpuTiming: false,
        hideSimulationTiming: true,
        panelStyle: PANEL_STYLE,
        schemas: LIQ_SCHEMAS,
        methods: Object.keys(LIQ_SCHEMAS),
        particleCounts: [],
        particleCountInput: true,
        physScaleMin: PHYS_MIN_SCALE,
        physScaleMax: PHYS_MAX_SCALE,
        flipParticleCapacityMax: MAX_TOTAL,
        initial: {
            method: currentMethod,
            material: currentMaterial,
            count: manualParticleCapacity(),
            physScale: fluidPhysicsScale,
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
            showGridBoundsSolid,
            renderMode: "surface",
            polygonShader: "physical",
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
        onApply: (values, changedKeys) => {
            if (controlsBinding) {
                const bindingPlan = applyFluidControls(controlsBinding, aquanovaBindingValues(values, changedKeys), changedKeys);
                finishAquanovaControlPlan(bindingPlan);
                const changed = (key: keyof typeof values): boolean => changedKeys.includes(key);
                if (changed("method") || changed("schema") || changed("renderMode") || changed("anisotropic") || changed("debug")) {
                    canvas.dataset.render = controlsBinding.renderMode.polygonEnabled ? "polygon" : "surface";
                }
                if (changed("method")) {
                    canvas.dataset.method = currentMethod;
                    controls.rebuildPhysics(currentMethod);
                    if (isPagingCapableMethod(currentMethod)) {
                        syncPagingControls();
                    }
                    refreshFlowControls();
                }
                if (changed("method") || changed("material")) {
                    refreshPhysicsParamVisibility();
                }
                if (changed("physScale") || changed("gridPosition") || changed("gridSize") || changed("gridResolution")) {
                    controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
                }
                if (changed("gridPosition") || changed("gridSize")) {
                    canvas.dataset.gridPosition = gridPosition.join(",");
                    canvas.dataset.gridSize = gridSize.join(",");
                    syncGridBoundsWireframe();
                    syncGridGizmo();
                }
                if (
                    changed("method") ||
                    changed("count") ||
                    changed("physScale") ||
                    changed("gridPosition") ||
                    changed("gridSize") ||
                    changed("gridResolution") ||
                    changed("markersPerCell") ||
                    changed("schema") ||
                    changed("foam") ||
                    changed("pagedGrid") ||
                    changed("pagedGridMaxPages")
                ) {
                    refreshFlipParticleCapacity();
                    refreshParticleCountControl();
                    refreshManualParticleUsage();
                }
                if (changed("color") && manualRun) {
                    fillManualColor(manualRun, values.color);
                }
                if (changed("showGridBounds")) {
                    showGridBounds = values.showGridBounds;
                    canvas.dataset.showGridBounds = String(showGridBounds);
                    syncGridBoundsWireframe();
                }
                if (changed("showGridBoundsSolid")) {
                    showGridBoundsSolid = values.showGridBoundsSolid;
                    canvas.dataset.showGridBoundsSolid = String(showGridBoundsSolid);
                    syncGridBoundsWireframe();
                }
                return;
            }
            const changed = (key: keyof typeof values): boolean => changedKeys.includes(key);
            const rebuildKeys: (keyof typeof values)[] = [
                "method",
                "count",
                "physScale",
                "material",
                "activeBlocks",
                "pagedGrid",
                "pagedGridMaxPages",
                "fusedBlockDiscovery",
                "gridResolution",
                "markersPerCell",
                "schema",
                "foam",
            ];
            // The shared binding owns preparation/commit. The remaining code synchronizes
            // Aquanova-only authoring state and scene UI after that commit succeeds.
            let requiresReconfiguration = controlsBinding ? false : rebuildKeys.some(changed);
            const previousConfiguration = {
                currentMethod,
                currentMaterial,
                fluidPhysicsScale,
                flipParticleCapacityRequest,
                particleCapacityRequestByMethod: new Map(particleCapacityRequestByMethod),
                mpmActiveBlocks,
                mpmPagedGrid,
                mpmPagedGridMaxPages,
                mpmFusedBlockDiscovery,
                flipPagedGrid,
                flipPagedGridMaxPages,
                flipGridResolution,
                flipMarkersPerCell,
                gridPosition: [...gridPosition] as [number, number, number],
                gridSize: [...gridSize] as [number, number, number],
                physValues: structuredClone(physValues),
            };
            const applyVisualState = (): void => {
                if (changed("debug")) {
                    configureFluidSimulationRenderLayer(surfaceTask, {
                        debug: values.debug === "polygonWireframe" ? "none" : (values.debug as FluidDebug),
                    });
                    configureFluidSimulationRenderLayer(polygonSurfaceTask, {
                        polygonWireframe: values.debug === "polygonWireframe",
                    });
                }
                if (renderProfileChanged) {
                    const profile = {
                        waterColor: values.color,
                        polygonShader: values.polygonShader,
                        absorption: values.absorption,
                        particleSize: values.size,
                        refractionStrength: values.refraction,
                        specularPower: values.specular,
                        reflectionExposure: values.reflectionExposure,
                        reflectionContrast: values.reflectionContrast,
                        waterReflectivity: values.reflectivity,
                        surfaceDepthBlur: values.depthBlur,
                        depthBlurEdgeThreshold: values.depthBlurThreshold,
                        surfaceThicknessBlur: values.thicknessBlur,
                        halfRendering: values.half,
                        surfaceFilter: values.surfaceFilter,
                        narrowRangeDelta: values.narrowDelta,
                        narrowRangeMu: values.narrowMu,
                        anisotropicSurface: values.anisotropic,
                        anisoRadiusDamping: values.anisoSurfScale,
                        thicknessDownscale: values.thicknessDownscale,
                    };
                    configureFluidSimulationRenderLayer(surfaceTask, { profile });
                    configureFluidSimulationRenderLayer(polygonSurfaceTask, { profile });
                    canvas.dataset.surfaceShader = values.polygonShader;
                    canvas.dataset.polygonShader = values.polygonShader;
                    if (manualRun) {
                        fillManualColor(manualRun, values.color);
                    }
                }
                if (changed("foam")) {
                    const foam = values.foam;
                    configureFluidSimulationRenderLayer(foamTask, {
                        enabled: foam.enabled,
                        foam: {
                            softness: foam.softness,
                            density: foam.density,
                            subsurfaceStrength: foam.subsurfaceStrength,
                            surfaceFiltering: foam.surfaceFiltering ?? false,
                            subsurfaceColor: foam.subsurfaceColor,
                            sizeScale: foam.size,
                            spraySize: foam.spraySize ?? 0.55,
                            sprayIntensity: foam.sprayIntensity ?? 1.4,
                            spraySeparation: foam.spraySeparation ?? 1,
                            blurRadius: foam.blurRadius,
                            lightIntensity: foam.lightIntensity,
                            ambient: foam.ambient,
                            aoStrength: foam.aoStrength,
                            normalStrength: foam.normalStrength,
                            debugByKind: foam.debugByKind ?? false,
                            debugTexture: foam.debugTexture as FoamDebugTexture,
                        },
                    });
                }
                if (changed("schema")) {
                    syncPolygonSurfaceRendering();
                }
            };

            if (changed("method")) {
                switchMethod(values.method, controlsBinding !== null);
            }
            if (changed("count")) {
                const count = Math.max(1, Math.min(MAX_TOTAL, Math.round(values.count)));
                if (currentMethod === "FLIP") {
                    flipParticleCapacityRequest = count;
                } else {
                    particleCapacityRequestByMethod.set(currentMethod, count);
                }
                refreshManualParticleUsage();
                refreshParticleCountControl();
            }
            if (changed("physScale")) {
                fluidPhysicsScale = Math.max(PHYS_MIN_SCALE, Math.min(PHYS_MAX_SCALE, values.physScale));
                controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
                refreshParticleCountControl();
                refreshManualParticleUsage();
            }
            if (changed("material")) {
                currentMaterial = values.material;
                refreshPhysicsParamVisibility();
            }
            const renderProfileChanged = [
                "polygonShader",
                "color",
                "absorption",
                "size",
                "refraction",
                "specular",
                "reflectionExposure",
                "reflectionContrast",
                "reflectivity",
                "depthBlur",
                "depthBlurThreshold",
                "thicknessBlur",
                "half",
                "surfaceFilter",
                "narrowDelta",
                "narrowMu",
                "anisotropic",
                "anisoSurfScale",
                "thicknessDownscale",
            ].some((key) => changed(key as keyof typeof values));
            if (changed("activeBlocks")) {
                mpmActiveBlocks = values.activeBlocks;
            }
            if (changed("pagedGrid")) {
                if (currentMethod === "FLIP") {
                    flipPagedGrid = values.pagedGrid;
                    refreshManualParticleUsage();
                } else {
                    mpmPagedGrid = values.pagedGrid;
                }
                controls.setPagedGridStatus("");
            }
            if (changed("pagedGridMaxPages")) {
                if (currentMethod === "FLIP") {
                    flipPagedGridMaxPages = Math.min(flipPageLimit(), values.pagedGridMaxPages);
                    controls.setPagedGridMaxPages(flipPagedGridMaxPages, flipPageLimit());
                    refreshManualParticleUsage();
                } else {
                    const limit = mpmPageLimit();
                    mpmPagedGridMaxPages = Math.min(limit, values.pagedGridMaxPages);
                    controls.setPagedGridMaxPages(mpmPagedGridMaxPages, limit);
                }
                controls.setPagedGridStatus("");
            }
            if (changed("fusedBlockDiscovery")) {
                mpmFusedBlockDiscovery = values.fusedBlockDiscovery;
            }
            if (changed("gridResolution")) {
                flipGridResolution = normalizeFluidFlipDiscretization(values.gridResolution, flipMarkersPerCell).gridResolution;
                controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
                refreshManualParticleUsage();
            }
            if (changed("markersPerCell")) {
                flipMarkersPerCell = normalizeFluidFlipDiscretization(flipGridResolution, values.markersPerCell).markersPerCell;
                refreshManualParticleUsage();
            }
            if (changed("showGridBounds")) {
                showGridBounds = values.showGridBounds;
                canvas.dataset.showGridBounds = String(showGridBounds);
                syncGridBoundsWireframe();
            }
            if (changed("showGridBoundsSolid")) {
                showGridBoundsSolid = values.showGridBoundsSolid;
                canvas.dataset.showGridBoundsSolid = String(showGridBoundsSolid);
                syncGridBoundsWireframe();
            }
            if (changed("schema")) {
                const previousSchema = previousConfiguration.physValues[currentMethod] ?? {};
                const changedSchemaKeys = Object.keys(values.schema).filter((key) => values.schema[key] !== previousSchema[key]);
                physValues[currentMethod] = { ...values.schema };
                refreshManualParticleUsage();
                if (
                    !controlsBinding &&
                    !rebuildKeys.some((key) => key !== "schema" && changed(key)) &&
                    changedSchemaKeys.length > 0 &&
                    changedSchemaKeys.every((key) => key === "polygonReconstructionMultiplier")
                ) {
                    for (const simulation of runningSims()) {
                        setFluidSimulationParameter(simulation, "polygonReconstructionMultiplier", values.schema.polygonReconstructionMultiplier ?? 1);
                    }
                    requiresReconfiguration = false;
                }
            }
            if (requiresReconfiguration && !applyingImportedPreset) {
                try {
                    reconfigureRunningSimulations();
                } catch (error) {
                    currentMethod = previousConfiguration.currentMethod;
                    currentMaterial = previousConfiguration.currentMaterial;
                    fluidPhysicsScale = previousConfiguration.fluidPhysicsScale;
                    flipParticleCapacityRequest = previousConfiguration.flipParticleCapacityRequest;
                    particleCapacityRequestByMethod.clear();
                    for (const [method, count] of previousConfiguration.particleCapacityRequestByMethod) {
                        particleCapacityRequestByMethod.set(method, count);
                    }
                    mpmActiveBlocks = previousConfiguration.mpmActiveBlocks;
                    mpmPagedGrid = previousConfiguration.mpmPagedGrid;
                    mpmPagedGridMaxPages = previousConfiguration.mpmPagedGridMaxPages;
                    mpmFusedBlockDiscovery = previousConfiguration.mpmFusedBlockDiscovery;
                    flipPagedGrid = previousConfiguration.flipPagedGrid;
                    flipPagedGridMaxPages = previousConfiguration.flipPagedGridMaxPages;
                    flipGridResolution = previousConfiguration.flipGridResolution;
                    flipMarkersPerCell = previousConfiguration.flipMarkersPerCell;
                    for (const method of Object.keys(previousConfiguration.physValues)) {
                        physValues[method] = previousConfiguration.physValues[method]!;
                    }
                    controls.setMethod(currentMethod);
                    controls.rebuildPhysics(currentMethod);
                    controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
                    syncPagingControls();
                    refreshFlipParticleCapacity();
                    refreshParticleCountControl();
                    refreshPhysicsParamVisibility();
                    syncPolygonSurfaceRendering();
                    throw error;
                }
            }
            applyVisualState();
        },
        on: {
            onGridSettings: (position, size) => (controlsBinding ? undefined : setGridSettings(position, size)),
            onGridGizmo: (visible) => setGridGizmoVisible(visible),
            onReset: () => restart(),
        },
    });
    const aquanovaControlHostState = () => ({
        currentMethod,
        currentMaterial,
        fluidPhysicsScale,
        flipParticleCapacityRequest,
        particleCapacityRequestByMethod: new Map(particleCapacityRequestByMethod),
        mpmActiveBlocks,
        mpmPagedGrid,
        mpmPagedGridMaxPages,
        mpmFusedBlockDiscovery,
        flipPagedGrid,
        flipPagedGridMaxPages,
        flipGridResolution,
        flipMarkersPerCell,
        gridPosition: [...gridPosition] as [number, number, number],
        gridSize: [...gridSize] as [number, number, number],
        physValues: structuredClone(physValues),
    });
    const restoreAquanovaControlHostState = (state: ReturnType<typeof aquanovaControlHostState>): void => {
        currentMethod = state.currentMethod;
        currentMaterial = state.currentMaterial;
        fluidPhysicsScale = state.fluidPhysicsScale;
        flipParticleCapacityRequest = state.flipParticleCapacityRequest;
        particleCapacityRequestByMethod.clear();
        for (const [method, count] of state.particleCapacityRequestByMethod) {
            particleCapacityRequestByMethod.set(method, count);
        }
        mpmActiveBlocks = state.mpmActiveBlocks;
        mpmPagedGrid = state.mpmPagedGrid;
        mpmPagedGridMaxPages = state.mpmPagedGridMaxPages;
        mpmFusedBlockDiscovery = state.mpmFusedBlockDiscovery;
        flipPagedGrid = state.flipPagedGrid;
        flipPagedGridMaxPages = state.flipPagedGridMaxPages;
        flipGridResolution = state.flipGridResolution;
        flipMarkersPerCell = state.flipMarkersPerCell;
        gridPosition.splice(0, 3, ...state.gridPosition);
        gridSize.splice(0, 3, ...state.gridSize);
        for (const method of Object.keys(state.physValues)) {
            physValues[method] = state.physValues[method]!;
        }
    };
    const applyAquanovaControlHostState = (snapshot: Readonly<FluidControlValues>, changedKeys: readonly (keyof FluidControlValues)[]): void => {
        if (snapshot.method !== "PBF" && snapshot.method !== "FLIP" && snapshot.method !== "MLS-MPM" && snapshot.method !== "PB-MPM") {
            throw new RangeError(`[aquanova-fluid-sim] unsupported fluid method "${snapshot.method}".`);
        }
        currentMethod = snapshot.method;
        currentMaterial = snapshot.material;
        fluidPhysicsScale = snapshot.physScale;
        gridPosition.splice(0, 3, ...snapshot.gridPosition);
        gridSize.splice(0, 3, ...snapshot.gridSize);
        flipGridResolution = snapshot.gridResolution;
        flipMarkersPerCell = snapshot.markersPerCell;
        physValues[currentMethod] = { ...snapshot.schema };
        if (changedKeys.includes("count")) {
            if (currentMethod === "FLIP") {
                flipParticleCapacityRequest = snapshot.count;
            } else {
                particleCapacityRequestByMethod.set(currentMethod, snapshot.count);
            }
        }
        if (currentMethod === "MLS-MPM") {
            if (changedKeys.includes("activeBlocks") || changedKeys.includes("pagedGrid")) {
                mpmActiveBlocks = snapshot.activeBlocks || snapshot.pagedGrid;
            }
            if (changedKeys.includes("pagedGrid")) {
                mpmPagedGrid = snapshot.pagedGrid;
            }
            if (changedKeys.includes("pagedGridMaxPages")) {
                mpmPagedGridMaxPages = snapshot.pagedGridMaxPages;
            }
            if (changedKeys.includes("fusedBlockDiscovery")) {
                mpmFusedBlockDiscovery = snapshot.fusedBlockDiscovery;
            }
        } else if (currentMethod === "FLIP") {
            if (changedKeys.includes("pagedGrid")) {
                flipPagedGrid = snapshot.pagedGrid;
            }
            if (changedKeys.includes("pagedGridMaxPages")) {
                flipPagedGridMaxPages = snapshot.pagedGridMaxPages;
            }
        }
    };
    const aquanovaControlTarget = (simulation: FluidSimulation, snapshot: Readonly<FluidControlValues>): FluidControlsBindingTarget => {
        const instance = instances.find((candidate) => candidate.sim === simulation);
        const gridDelta: [number, number, number] = [
            snapshot.gridPosition[0] - committedGridPosition[0],
            snapshot.gridPosition[1] - committedGridPosition[1],
            snapshot.gridPosition[2] - committedGridPosition[2],
        ];
        const center =
            simulation === manualRun?.sim
                ? snapshot.gridPosition
                : instance?.simulationCenter
                  ? ([instance.simulationCenter[0] + gridDelta[0], instance.simulationCenter[1] + gridDelta[1], instance.simulationCenter[2] + gridDelta[2]] as [
                        number,
                        number,
                        number,
                    ])
                  : snapshot.gridPosition;
        const options: FluidSimulationOptions = {
            ...simulation.options,
            bounds: simulationBounds(center),
            particleCount: simulation === manualRun?.sim ? snapshot.count : simulation.count,
            particleRadius:
                simulation === manualRun?.sim ? manualFluidConfigForMethod(currentMethod, snapshot.gridPosition).particleRadius : (instance?.radius ?? simulation.particleRadius),
            flow: simulation === manualRun?.sim ? flowInWorldSpace(activeFlow) : simulation.options.flow,
            sceneSdf: simulation.sceneSdf,
            forceField: simulation.forceField,
            profiler: simulation.profiler,
        };
        return {
            simulation,
            options,
            preserveState: true,
            activeCount: simulation.activeCount ?? simulation.count,
        };
    };
    controlsBinding = bindFluidControls({
        controls,
        target: simulationCollection,
        deviceLimits: fluidDeviceLimits,
        maxParticleCount: MAX_TOTAL,
        renderLayers: {
            surface: surfaceTask,
            polygon: polygonSurfaceTask,
            foam: foamTask,
        },
        resolveTarget: aquanovaControlTarget,
        captureHostState: aquanovaControlHostState,
        applyHostState: applyAquanovaControlHostState,
        restoreHostState: restoreAquanovaControlHostState,
        onPageDiagnostics: (diagnostics) => {
            canvas.dataset.pagedGridPages = String(diagnostics.requiredPages);
            canvas.dataset.pagedGridPageCapacity = String(diagnostics.capacity);
            canvas.dataset.pagedGridOverflow = String(diagnostics.overflow);
        },
    });
    window.addEventListener("pagehide", () => controlsBinding && disposeFluidControlsBinding(controlsBinding), { once: true });
    configureFluidSimulationRenderLayer(surfaceTask, { profile: { polygonShader: controls.getValues().polygonShader } });
    configureFluidSimulationRenderLayer(polygonSurfaceTask, { profile: { polygonShader: controls.getValues().polygonShader } });
    canvas.dataset.surfaceShader = controls.getValues().polygonShader;
    canvas.dataset.polygonShader = controls.getValues().polygonShader;
    syncPolygonSurfaceRendering();
    refreshFlipParticleCapacity = () => controls.setFlipParticleCapacity(manualParticleCapacity());
    refreshParticleCountControl = () => controls.setParticleCount(manualParticleCapacity());
    refreshFlipParticleCapacity();
    refreshParticleCountControl();

    // ── Export / import fluid settings ───────────────────────────────────────
    // Reuse the shared grouped JSON shape: serialise the live
    // physics + surface-render params so a setting authored here can be dropped straight
    // into another demo (e.g. aquanova's FLUID_SETTING) or re-imported below to iterate.
    const IO_BTN_STYLE = "flex:1;padding:6px;border:0;border-radius:6px;cursor:pointer;background:#4a5568;color:#fff;font-weight:600;";
    function liqPairState(): PairState {
        const v = controls.getValues();
        return {
            simulationSemantics,
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
            physScale: fluidPhysicsScale,
            count: currentMethod === "FLIP" ? (flipParticleCapacityRequest ?? 0) : (particleCapacityRequestByMethod.get(currentMethod) ?? 0),
            gridResolution: currentMethod === "FLIP" ? flipGridResolution : undefined,
            markersPerCell: currentMethod === "FLIP" ? flipMarkersPerCell : undefined,
            material: currentMethod === "PB-MPM" ? v.material : undefined,
            freeCamera: {
                position: [cam.position.x, cam.position.y, cam.position.z],
                target: [cam.target.x, cam.target.y, cam.target.z],
            },
            renderMode: v.renderMode,
            polygonShader: v.polygonShader,
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
            // Paging is generic across every paging-capable backend (FLIP and MLS-MPM), so it is
            // serialized for whichever method authored the preset rather than MLS-MPM only.
            pagedGrid: isPagingCapableMethod(currentMethod) ? v.pagedGrid : undefined,
            pagedGridMaxPages: isPagingCapableMethod(currentMethod) ? v.pagedGridMaxPages : undefined,
            fusedBlockDiscovery: currentMethod === "MLS-MPM" ? v.fusedBlockDiscovery : undefined,
            grid: { position: [...gridPosition], size: [...gridSize] },
            showGridBounds: v.showGridBounds,
            showGridBoundsSolid: v.showGridBoundsSolid,
            foam: v.foam,
            emitters: structuredClone(activeFlow.emitters),
            sinks: structuredClone(activeFlow.sinks),
            initialEmittersFillCapacity: activeFlow.initialEmittersFillCapacity,
            demoState: { simulationType, samplingMode: modeValue, fillStrategy },
            showContainer: v.showContainer,
        };
    }
    let presetSession: FluidPresetSession | null = null;
    let applyingImportedPreset = false;
    function applyImportedPreset(j: FluidExportJson, restoreOnFailure = true, reconfigure = true): void {
        const previousPreset = restoreOnFailure ? buildCurrentPreset() : null;
        const nextSession = importFluidPresetSession(j, liqPairState());
        applyingImportedPreset = true;
        try {
            // Switch to the file's method first so the physics sliders target the right block.
            const method = j.meta?.method;
            if (method && LIQ_SCHEMAS[method]) switchMethod(method);
            const p = nextSession.state;
            simulationSemantics = p.simulationSemantics ?? CURRENT_FLUID_SIMULATION_SEMANTICS;
            fluidPhysicsScale = Math.max(PHYS_MIN_SCALE, Math.min(PHYS_MAX_SCALE, p.physScale ?? 1));
            controls.setPhysScale(fluidPhysicsScale);
            // Physics: update the seed values, the sliders AND every running sim.
            if (p.schema) {
                const merged = { ...SCHEMA_DEFAULTS[currentMethod], ...p.schema };
                physValues[currentMethod] = merged;
                controls.setPhysics(merged);
                controls.rebuildPhysics(currentMethod);
                syncPolygonSurfaceRendering();
            }
            if (typeof p.material === "number" && currentMethod === "PB-MPM") {
                currentMaterial = p.material;
                controls.setMaterial(currentMaterial);
            }
            if (currentMethod === "FLIP") {
                const discretization = normalizeFluidFlipDiscretization(p.gridResolution ?? flipGridResolution, p.markersPerCell ?? flipMarkersPerCell);
                flipGridResolution = discretization.gridResolution;
                flipMarkersPerCell = discretization.markersPerCell;
                flipParticleCapacityRequest = typeof p.count === "number" && p.count > 0 ? Math.min(MAX_TOTAL, Math.round(p.count)) : null;
                controls.setGridResolution(flipGridResolution);
                controls.setMarkersPerCell(flipMarkersPerCell);
                // Import FLIP paging generically (grid resolution above defines the page-block ceiling).
                flipPagedGrid = p.pagedGrid ?? flipPagedGrid;
                flipPagedGridMaxPages = Math.min(flipPageLimit(), p.pagedGridMaxPages ?? flipPagedGridMaxPages);
                controls.setPagedGrid(flipPagedGrid);
                controls.setPagedGridMaxPages(flipPagedGridMaxPages, flipPageLimit());
                controls.setPagedGridStatus("");
            } else if (typeof p.count === "number" && p.count > 0) {
                particleCapacityRequestByMethod.set(currentMethod, Math.min(MAX_TOTAL, Math.round(p.count)));
            } else {
                particleCapacityRequestByMethod.delete(currentMethod);
            }
            if (currentMethod === "MLS-MPM") {
                mpmPagedGrid = p.pagedGrid ?? mpmPagedGrid;
                mpmActiveBlocks = mpmPagedGrid || (p.activeBlocks ?? mpmActiveBlocks);
                const limit = mpmPageLimit();
                mpmPagedGridMaxPages = Math.min(limit, p.pagedGridMaxPages ?? mpmPagedGridMaxPages);
                mpmFusedBlockDiscovery = p.fusedBlockDiscovery ?? mpmFusedBlockDiscovery;
                controls.setActiveBlocks(mpmActiveBlocks);
                controls.setPagedGrid(mpmPagedGrid);
                controls.setPagedGridMaxPages(mpmPagedGridMaxPages, limit);
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
            refreshParticleCountControl();
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
                configureFluidSimulationRenderLayer(surfaceTask, { useParticleColor: useMeshColors });
                syncSimulationCollection();
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
            manualForceRadius =
                typeof forceInfluenceRadius === "number" && isFinite(forceInfluenceRadius) ? Math.max(0.1, Math.min(20, forceInfluenceRadius)) : DEFAULT_FORCE_RADIUS;
            forceRadiusInput.value = String(manualForceRadius);
            showForceRadius();
            // Surface and foam setters publish one normalized snapshot so the render and simulation
            // effects observe the fully committed preset rather than intermediate per-setter values.
            controls.runTransaction(() => {
                if (p.color !== undefined) controls.setColor(p.color);
                if (p.absorption !== undefined) controls.setAbsorption(p.absorption);
                if (p.size !== undefined) controls.setParticleSize(p.size);
                if (p.renderMode !== undefined) controls.setRenderMode(p.renderMode === "spheres");
                if (p.polygonShader !== undefined) controls.setPolygonShader(p.polygonShader);
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
                }
            });
            // Impulse — intensity + direction ride in their own block (absent in files written before it
            // existed, and in fluid-demo presets, which leaves the current values alone).
            const imp = nextSession.application.impulse;
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
            const g = nextSession.application.grid;
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
            }
            showGridBoundsSolid = p.showGridBoundsSolid ?? false;
            controls.setShowGridBoundsSolid(showGridBoundsSolid);
            syncGridBoundsWireframe();
            if (p.freeCamera && p.freeCamera.position.every(isFinite) && p.freeCamera.target.every(isFinite)) {
                applyFreeCameraPose(p.freeCamera.position, p.freeCamera.target);
            }
            refreshFlipParticleCapacity();
            refreshPhysicsParamVisibility();
            if (reconfigure) {
                reconfigureRunningSimulations();
            }
            presetSession = nextSession;
        } catch (error) {
            if (previousPreset) {
                applyingImportedPreset = false;
                applyImportedPreset(previousPreset, false, false);
            }
            throw error;
        } finally {
            applyingImportedPreset = false;
        }
    }
    const buildCurrentPreset = (): FluidExportJson => {
        const current = presetSession ?? { state: liqPairState(), application: {} };
        presetSession = editFluidPresetSession(current, {
            state: liqPairState(),
            application: {
                impulse: { intensity: impulseIntensity, direction: [impulseDir[0], impulseDir[1], impulseDir[2]], radius: impulseRadius },
                grid: { x: gridSize[0], y: gridSize[1], z: gridSize[2], position: [...gridPosition] },
            },
        });
        return exportFluidPresetSession(presetSession, {
            demo: "aquanova-fluid-sim",
            method: currentMethod,
        });
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
    installClickFeedback([fluidSimCreateButton, fluidSimUpdateButton, fluidSimDeleteButton, restartBtn, exportBtn, importBtn]);

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
        const response = await fetch(`/aquanova/fluidSim/${encodeURIComponent(normalized)}.json`, { cache: "no-store" });
        if (!response.ok) throw new Error(`Could not load "${normalized}" (HTTP ${response.status}).`);
        applyImportedPreset((await response.json()) as FluidExportJson);
        fluidSimSelect.value = normalized;
        fluidSimNameInput.value = normalized;
        fluidSimStatus.textContent = `Loaded ${normalized}`;
        canvas.dataset.fluidSim = normalized;
    };
    const saveFluidSim = async (name: string, create: boolean, previousName?: string): Promise<void> => {
        const normalized = normalizeFluidSimName(name.trim());
        if (!isValidFluidSimName(normalized)) {
            throw new Error("Use 1-64 letters, numbers, hyphens, or underscores.");
        }
        if (create && fluidSimNames.includes(normalized)) {
            throw new Error(`"${normalized}" already exists; use Update instead.`);
        }
        const previous = normalizeFluidSimName(previousName ?? normalized);
        if (!create && !fluidSimNames.includes(previous)) {
            throw new Error(`"${previous}" does not exist.`);
        }
        const renameQuery = !create && previous !== normalized ? `?renameTo=${encodeURIComponent(normalized)}` : "";
        const result = await readApiResponse(
            await fetch(`/lab-api/aquanova-fluid-sims/${encodeURIComponent(create ? normalized : previous)}${renameQuery}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(buildCurrentPreset()),
            })
        );
        setFluidSimNames(result.names ?? [...fluidSimNames, normalized], normalized);
        if (shipManifest) shipManifest.fluidSim = [...fluidSimNames];
        fluidSimStatus.textContent = create ? `Created ${normalized}` : previous === normalized ? `Updated ${normalized}` : `Renamed ${previous} to ${normalized}`;
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
        void runFluidSimAction(() => saveFluidSim(fluidSimNameInput.value, false, fluidSimSelect.value));
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
    configureFluidSimulationRenderLayer(foamTask, {
        foam: {
            softness: initialFoam.softness,
            density: initialFoam.density,
            subsurfaceStrength: initialFoam.subsurfaceStrength,
            surfaceFiltering: initialFoam.surfaceFiltering ?? false,
            subsurfaceColor: initialFoam.subsurfaceColor,
            sizeScale: initialFoam.size,
            blurRadius: initialFoam.blurRadius,
            lightIntensity: initialFoam.lightIntensity,
            ambient: initialFoam.ambient,
            aoStrength: initialFoam.aoStrength,
            normalStrength: initialFoam.normalStrength,
            spraySize: initialFoam.spraySize ?? 0.55,
            sprayIntensity: initialFoam.sprayIntensity ?? 1.4,
            spraySeparation: initialFoam.spraySeparation ?? 1,
            debugByKind: initialFoam.debugByKind ?? false,
            debugTexture: initialFoam.debugTexture as FoamDebugTexture,
        },
    });
    setFluidSimulationCollectionFoam(simulationCollection, initialFoam);

    const demoSection = controls.makeSection("Demo", [
        title,
        simulationTypeRow,
        aquanovaRenderingRow,
        fluidSimRow,
        fluidSimAuthoring,
        collisionRadiusRow,
        collisionDebugRow,
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
        const previous = simulationType;
        const changed = previous !== value;
        simulationType = value;
        simulationTypeSelect.value = value;
        setHostSectionVisible(meshLiquefactionSection, value === "mesh");
        forceStrengthRow.style.display = value === "fluid" ? "" : "none";
        forceRadiusRow.style.display = value === "fluid" ? "" : "none";
        setHostSectionVisible(emitterSection, value === "fluid");
        setHostSectionVisible(sinkSection, value === "fluid");
        controls.setSectionVisible("Foam", value === "fluid");
        controls.setParticleCountVisible(value === "fluid");
        controls.setPhysScaleVisible(value === "fluid");
        controls.setGridSettings(gridPosition, gridSize, simulationCellSize());
        refreshParticleCountControl();
        updateHelperText();
        canvas.dataset.simulationType = value;
        if (changed && !applyingImportedPreset) {
            try {
                reconfigureRunningSimulations();
            } catch (error) {
                simulationType = previous;
                simulationTypeSelect.value = previous;
                setHostSectionVisible(meshLiquefactionSection, previous === "mesh");
                setHostSectionVisible(emitterSection, previous === "fluid");
                setHostSectionVisible(sinkSection, previous === "fluid");
                controls.setSectionVisible("Foam", previous === "fluid");
                controls.setParticleCountVisible(previous === "fluid");
                controls.setPhysScaleVisible(previous === "fluid");
                updateHelperText();
                canvas.dataset.simulationType = previous;
                throw error;
            }
        }
        setStatus();
    };
    flowEditor = createFluidFlowEditor({
        emittersHost: emitterFlowHost,
        sinksHost: sinkFlowHost,
        flow: activeFlow,
        onChange: (flow) => {
            activeFlow = flow;
            if (manualRun) setFluidSimulationFlow(manualRun.sim, flowInWorldSpace(activeFlow));
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
        getEmitterRateMode: () => (currentMethod === "FLIP" ? "occupancy-refill" : "unlimited-toggle"),
        getInitialEmitterParticleCount: (emitter) =>
            currentMethod === "FLIP" && emitter.behavior === "initial" ? (flipInitialEmitterParticleCounts().get(emitter.id) ?? 0) : undefined,
        onInitialEmitterParticleCountDisplayed: (count) => {
            if (count === undefined) {
                delete canvas.dataset.selectedInitialEmitterParticleCount;
            } else {
                canvas.dataset.selectedInitialEmitterParticleCount = String(count);
            }
        },
    });
    refreshFlowControls = () => flowEditor?.setFlow(activeFlow);
    refreshFlowControls();
    setSimulationType(simulationType);
    document.body.append(controls.root);
    await Promise.all([shipManifestReady, behaviorOptionsReady]);
    const loadedShipManifest = shipManifest as ShipManifestData | null;
    setFluidSimNames(publishedFluidSimNames.length ? publishedFluidSimNames : (loadedShipManifest?.fluidSim ?? []));
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

    const pickEditableMeshAt = async (px: number, py: number): Promise<{ mesh: Mesh; point: [number, number, number] } | null> => {
        const info = await pickAsync(picker, px, py, { filter: (mesh) => editableShipMeshes.has(mesh) && mesh.visible !== false });
        return info.hit && info.pickedMesh && info.pickedPoint ? { mesh: info.pickedMesh as Mesh, point: [info.pickedPoint[0], info.pickedPoint[1], info.pickedPoint[2]] } : null;
    };
    const toggleMeshGizmoAt = async (px: number, py: number): Promise<string | null> => {
        const hit = await pickEditableMeshAt(px, py);
        if (!hit) return null;
        const target = editableTargetByMesh.get(hit.mesh) ?? hit.mesh;
        setMeshGizmoTarget(meshGizmoTarget === target ? null : target, hit.point);
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
        getTotalParticles: () => surfaceTask.particleStream?.count ?? 0,
        paused: () => paused,
        setPaused: setPausedState,
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
        manualStepCount: () => manualStepCount,
        renderMode: () => canvas.dataset.render ?? "surface",
        polygonSurfaceCount: () => getFluidSimulationCollectionDiagnostics(simulationCollection).polygons.length,
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
            refreshFlipParticleCapacity();
            refreshManualParticleUsage();
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
        },
        foamEnabled: () => controls.getValues().foam.enabled,
        camState: () => ({ px: cam.position.x, py: cam.position.y, pz: cam.position.z, tx: cam.target.x, ty: cam.target.y, tz: cam.target.z, yaw: camYaw, pitch: camPitch }),
        cameraMovementSpeedMultiplier,
        flipDiscretization: () => flipDiscretization(),
        manualParticleRadius: () => manualRun?.sim.particleRadius ?? null,
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
            camPitch = Math.atan2(dy, Math.sqrt(dx * dx + dz * dz));
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
        setFillStrategy: (s: FluidMeshSamplingStrategy) => {
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
        meshAt: async (px: number, py: number): Promise<string | null> => {
            const hit = await pickEditableMeshAt(px, py);
            return hit ? meshGizmoId(editableTargetByMesh.get(hit.mesh) ?? hit.mesh) : null;
        },
        meshGizmo: (): string | null => meshGizmoId(meshGizmoTarget),
        meshGizmoPosition: (): [number, number, number] | null => meshGizmoWorldPosition(meshGizmoTarget),
        meshGizmoPositionText: (): string | null => meshGizmoPositionLine.textContent,
        selectMeshGizmo: (entityName: string): boolean => {
            const instance = instances.find((candidate) => candidate.key.startsWith(`${entityName}#`));
            const mesh = instance?.meshes[0];
            if (!mesh) return false;
            setMeshGizmoTarget(editableTargetByMesh.get(mesh) ?? mesh);
            return true;
        },
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
        setCollisionDebug: (visible: boolean): void => {
            collisionDebugVisible = visible;
            collisionDebugInput.checked = visible;
            updateCollisionDebugOverlay();
        },
        fluidCollisionOverrides: () =>
            shipCollisionPlacements
                .filter((placement) => placement.fluidSimShape !== undefined)
                .map((placement) => ({ id: placement.id, primitive: fluidPrimitivesForPlacement(placement)[0] })),
        fluidSimNames: (): string[] => [...fluidSimNames],
        exportPreset: (): unknown => buildCurrentPreset(),
        importPreset: (j: FluidExportJson): void => applyImportedPreset(j),
        setUseMeshColors: (on: boolean) => {
            useMeshColors = on;
            meshColorInput.checked = on;
            configureFluidSimulationRenderLayer(surfaceTask, { useParticleColor: on });
            syncSimulationCollection();
        },
        readColors: async (key: string, n = 12, stride = 1): Promise<number[] | null> => {
            const inst = instances.find((i) => i.key === key);
            if (!inst || !inst.colorChannel || inst.count === 0) return null;
            const step = Math.max(1, Math.floor(stride));
            const count = Math.min(n, Math.floor(inst.count / step) || 1);
            const all = await readFluidParticleChannel(inst.colorChannel, { particleCount: inst.count });
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
        const dl = Math.sqrt(dx * dx + dy * dy + dz * dz);
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
            const behaviorMesh = next.meshes[0];
            for (const name of behaviorMesh ? (shipBehaviorManager?.getLinkedEntityNames(behaviorMesh) ?? []) : []) {
                for (const linked of instancesByNodeName.get(name) ?? []) {
                    if (!seen.has(linked)) queue.push(linked);
                }
            }
        }
        const selectedInstances = meshGizmoTarget ? instancesByEditableTarget.get(meshGizmoTarget) : undefined;
        if (selectedInstances?.some((instance) => group.includes(instance))) {
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
            if (isGizmoInteracting(canvas)) {
                return;
            }
            // The utility layer identifies a gizmo press asynchronously. Wait for that pick before
            // selecting the ship mesh behind the handle, otherwise this handler can detach the
            // gizmo just before its drag starts.
            meshGizmoSelectionPending = true;
            void (async () => {
                try {
                    while (isGizmoPickPending(canvas)) {
                        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
                    }
                    if (!isGizmoInteracting(canvas)) {
                        await toggleMeshGizmoAt(px, py);
                    }
                } finally {
                    meshGizmoSelectionPending = false;
                }
            })();
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
        if (collisionDebugVisible) {
            updateCollisionDebugOverlay();
        }
        if (localEnvironmentController) {
            const previousProbe = localEnvironmentController.blendInfo().dominantProbeId;
            const blend = localEnvironmentController.updatePoi([cam.position.x, cam.position.y, cam.position.z]);
            if (blend.dominantProbeId !== previousProbe) {
                const fluidEnvironment = localEnvironmentController.dominantEnvironment();
                if (fluidEnvironment) {
                    const environment = createFluidRenderEnvironment(fluidEnvironment);
                    configureFluidSimulationRenderLayer(surfaceTask, { environment });
                    configureFluidSimulationRenderLayer(polygonSurfaceTask, { environment });
                }
            }
        }
        // Open the whole-frame GPU-timing envelope BEFORE any pass is encoded this frame.
        if (profiler) {
            beginFluidSimulationProfilerFrame(profiler);
        }
        updateAnimationManager(animManager, deltaMs); // advance the model's walk (skeleton pose)
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
        const growDt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 30);
        if (!paused && !meshGizmoSelectionPending && !isGizmoPickPending(canvas)) {
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
                    if (inst.sim) stepFluidSimulation(inst.sim, dt);
                    if (inst.impulseRemaining > 0) {
                        inst.impulseRemaining = Math.max(0, inst.impulseRemaining - dt);
                        if (inst.impulseRemaining === 0 && inst.sim) setFluidSimulationForceField(inst.sim, null);
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
                if (pendingForce && pendingForce.expiresAt >= performance.now()) {
                    rayForce.setRay(pendingForce.origin, pendingForce.dir, pendingForce.push, pendingForce.radius, pendingForce.accel);
                    setFluidSimulationForceField(manualRun.sim, forceFieldHandle(rayForce.spec));
                } else {
                    pendingForce = null;
                    setFluidSimulationForceField(manualRun.sim, null);
                }
                stepFluidSimulation(manualRun.sim, dt);
                manualStepCount++;
            }
        }
        if (paused) {
            refreshFluidSimulationCollectionPolygonSurfaces(simulationCollection);
        }

        syncSimulationCollection();
        const total = getFluidSimulationCollectionDiagnostics(simulationCollection).count;
        const displayedCount = simulationType === "fluid" ? (manualRun?.sim.activeCount ?? manualRun?.sim.count ?? 0) : total;
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
                gpu.refreshTiming(profiler ? readFluidSimulationProfiler(profiler) : null);
                let simBytes = getFluidSimulationCollectionDiagnostics(simulationCollection).gpuBytes;
                for (const inst of instances) {
                    simBytes += inst.collisionBuffer?.size ?? 0;
                }
                if (manualRun) simBytes += (manualRun.collisionBuffer?.size ?? 0) + manualRun.colorChannel.capacity * 16;
                gpu.refreshMemory(simBytes, engine.canvas.width, engine.canvas.height);
            }
            if (controlsBinding) {
                syncFluidControls(controlsBinding);
            }
            fpsAccumMs = 0;
            fpsFrames = 0;
        }
    });

    await envReady;
    // Load the full Aquanova ship before scene registration so the liquefy plugin materializes for
    // every ship mesh.
    await loadShipFoes();
    initializeCollisionDebugOverlay();
    // The overlay is an explicit demo-owned list. It contains only liquefiable ship meshes, while
    // task-local visibility gating decides which dissolving meshes draw on a given frame.
    for (const instance of instances) {
        for (const mesh of instance.meshes) {
            addMeshToTask(foeTask, mesh);
        }
    }
    invalidatePhaseTaskBundles();
    await registerScene(scene);
    collisionDebugSceneRegistered = true;
    updateCollisionDebugOverlay();
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
