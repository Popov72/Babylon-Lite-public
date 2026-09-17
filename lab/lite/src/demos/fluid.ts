// Fluid demo — GPU fluid simulation (PBF / FLIP / MLS-MPM / PB-MPM) with SDF scene collision.
//
// This module is the GENERIC CORE. It owns the engine/scene/camera, the shared
// scene-SDF UBO + hole ring, both sims + their lifecycle, the render tasks
// (particle / fluid-surface / sky), the whole panel UI, the per-(demo, method)
// parameter store, the frame loop and input dispatch — all DEMO-AGNOSTIC. The
// three demos (capsule / box / fountain) live under ./fluid/scenes and are driven
// through the `FluidDemo` interface; the core contains no per-demo branches.
//
// The simulation runs as a WebGPU compute pass encoded into the frame's command
// encoder from onBeforeRender (engine._currentEncoder is valid there); the
// particles render via a custom frame-graph task that shares the scene depth
// buffer.

import {
    addMeshToTask,
    addTask,
    addTaskAfter,
    addTaskBefore,
    addToScene,
    attachPositionGizmoToNode,
    attachRotationGizmoToNode,
    attachScaleGizmoToNode,
    createArcRotateCamera,
    createDepthResolveTask,
    createDirectionalLight,
    createEngine,
    createGround,
    createHemisphericLight,
    createLineMaterial,
    createLineSystem,
    createMeshFromData,
    createPbrMaterial,
    createPcfDirectionalShadowGenerator,
    createPositionGizmo,
    createRenderTarget,
    createRenderTask,
    createRotationGizmo,
    createScaleGizmo,
    createSceneContext,
    createStandardMaterial,
    createUtilityLayer,
    enableMirroredMeshes,
    fluidSimulationCellSize,
    getEffectiveAspectRatio,
    getFrameGraph,
    getViewProjectionMatrix,
    getContainerMeshes,
    isGizmoDragging,
    isGizmoInteracting,
    isGizmoPickPending,
    goToFrame,
    loadGltf,
    loadEnvironment,
    loadHdrEnvironment,
    mat4Decompose,
    mat4Multiply,
    mat4Translation,
    createBlurPostProcessTask,
    markMaterialUboDirty,
    onBeforeRender,
    onSceneDispose,
    playAnimation,
    registerSceneWithShadowSupport,
    registerUtilityLayer,
    removeFromScene,
    renderFrame,
    setPositionGizmoLocalCoordinates,
    setRotationGizmoLocalCoordinates,
    setScaleGizmoLocalCoordinates,
    setMeshVisible,
    setEnvironmentRotation,
    setShadowTaskCasterMeshes,
    startEngine,
    stopEngine,
    updateLineSystem,
    waitForGpuIdle,
} from "babylon-lite";
import {
    resolveFluidSimulationConfig,
    resolveFluidAllocationPlan,
    CURRENT_FLUID_SIMULATION_SEMANTICS,
    fluidParticleBytesPerSlot,
    fluidParticleBufferLimitBytes,
    fluidDeviceParticleCapacity,
    resolveFluidGridCompatibility,
    fluidMaximumPageCapacity,
    fitFluidGridResolution,
    formatFluidPageDiagnostics,
    FLUID_GRID_MAX_AXIS_CELLS,
    mlsMpmDefaultPageCapacity,
    resolveFluidRenderMode,
    transformFluidFlow,
} from "babylon-lite";
import {
    carryMethodIndependentState,
    adoptFluidForceField,
    adoptFluidSceneSdf,
    applyFluidControls,
    applyFluidGridSettings,
    attachFluidSimulationCollectionRenderLayer,
    attachFluidSimulationRenderLayer,
    beginFluidSimulationProfilerFrame,
    bindFluidControls,
    configureFluidSimulationRenderLayer,
    commitFluidReconfiguration,
    createFluidCompositeSceneSdf,
    createFluidInitialStatePlanCache,
    createFluidSimulation,
    loadFlipReferenceBackend,
    createFluidSimulationCollection,
    createFluidControlsPanel,
    createFluidFlowEditor,
    createFluidRenderEnvironment,
    createFluidSimulationProfiler,
    createRayForce,
    createSolidGridBounds,
    DEFAULT_FLUID_SCHEMAS,
    disposeFluidControlsBinding,
    disposeFluidSceneSdf,
    exportJsonFromPairState,
    FLIP_DEFAULT_PAGE_CAPACITY,
    fluidCaptureCompletionTime,
    fluidSimulationLifecycle,
    fluidSimulationStepDelta,
    MAX_FLUID_POLYGON_POINTS,
    parseBlenderFluidJson,
    pbmpmParamKeysForMaterial,
    presetFromExportJson,
    resolveFlipDiscretization,
    scenePayloadFromBlenderFluidJson,
    endFluidSimulationProfilerFrame,
    getFluidSimulationDiagnostics,
    importFluidPresetSession,
    editFluidPresetSession,
    exportFluidPresetSession,
    fluidInitialEmitterVolume,
    prepareFluidReconfigurationUpdate,
    readFluidSimulationPositions,
    readFluidSimulationProfiler,
    refreshFluidSimulationPolygonSurface,
    resetFluidSimulation,
    cancelFluidReconfiguration,
    setFluidSimulationFlow,
    setFluidSimulationFoam,
    setFluidSimulationForceField,
    setFluidSimulationMaterial,
    setFluidSimulationParameter,
    setFluidSimulationProfiler,
    setFluidSimulationSceneSdf,
    settleFluidSimulation,
    stepFluidSimulation,
    submitFluidSimulationStep,
    submitFluidSimulationSteps,
    submitFluidSimulationProfiler,
    syncFluidControls,
    updateFluidSimulationEmitter,
    updateFluidSceneSdfContainer,
    updateFluidSceneSdfGridSettings,
    updateFluidSceneSdfStaticOffset,
    updateFluidSceneSdfStaticScale,
    updateFluidSceneSdfTransforms,
} from "babylon-lite";
import type { FluidEmitter, FluidFlowConfig, FluidSceneSdf, FluidShape, FluidSimulationSemantics, FluidSink, ForceFieldSpec, Mat4 } from "babylon-lite";
import { wgsl } from "babylon-lite/shader/wgsl.js";
import type {
    BlenderFluidCollision,
    BlenderFluidInitialState,
    BlenderFluidScene,
    FluidExportJson,
    FluidFlowEditor,
    FluidFlowObjectKind,
    FluidDebug,
    FluidForceField,
    FluidSimulation,
    FluidSimulationBackend,
    FluidSimulationOptions,
    FluidSimulationProfiler,
    FluidControlsBinding,
    FluidControlsApplicationPlan,
    FluidControlsBindingTarget,
    FluidControlValues,
    FluidPresetSession,
    FoamConfig,
    FoamDebugTexture,
    SceneSdfSpec,
} from "babylon-lite";
import type { AssetContainer, Mesh, Task, EnvironmentTextures, Renderable, Material, PbrMaterialProps, SceneNode, Vec3 } from "babylon-lite";
import { buildHdrSkyboxRenderable } from "babylon-lite";
// Plain source→target blit used as the no-bloom presentation pass. Only the TYPE is
// re-exported from the package root, so the factory comes from its own module (the same
// deep-import convention the fluid sim + HDR skybox already use here).
import { createPostProcessTask } from "babylon-lite";
import { demoAssetUrl } from "./demo-asset-url.js";
import type { DemoParam, FluidCtx, FluidDemo, FluidDomainBounds, FluidGridSettings, PairState, PendingForce } from "./fluid/demo.js";
import { getQualityPreset, QUALITIES, DEFAULT_QUALITY, loadQualityPresets, type Quality } from "./fluid/quality-presets.js";
import { createFluidReferencePump, queueFluidReferenceChange, startFluidReferencePump, stopFluidReferencePump } from "./fluid/reference-pump.js";
import type { FluidReferencePump } from "./fluid/reference-pump.js";
import { fluidReferenceTiming } from "./fluid/reference-timing.js";
import { createEmitterSourceTransform, resolveEmitterSourceTransform } from "./fluid/emitter-source-transform.js";
import type { EmitterSourceTransform } from "./fluid/emitter-source-transform.js";
import { ENV_STUDIO_URL, INTERACTIVE_FORCE_SAMPLE_HOLD_MS } from "./fluid/demo.js";
import {
    cellSizeForPhysicsScale,
    FLIP_DEFAULT_MARKERS_PER_CELL,
    FLIP_HIGH_MARKERS_PER_CELL,
    GRID_RESOLUTION_MIN,
    gridBounds,
    gridCellsForSize,
    gridPositionForBounds,
    gridResolutionForScale,
    gridSizeForBounds,
    MPM_MAX_SCALE,
    MPM_MIN_SCALE,
    PBF_MAX_SCALE,
    PBF_MIN_SCALE,
    PBMPM_MAX_SCALE,
    PBMPM_MIN_SCALE,
    PHYS_MAX_SCALE,
    PHYS_MIN_SCALE,
    scaleLimitsForMethod,
} from "babylon-lite";
import { screenRay } from "./fluid/pick.js";
import { attachFluidCameraControls } from "./fluid/camera-controls.js";
import { createCapsuleDemo } from "./fluid/scenes/capsule.js";
import { createBoxDemo } from "./fluid/scenes/box.js";
import { createFountainDemo } from "./fluid/scenes/fountain.js";
import { createMarbleTowerDemo } from "./fluid/scenes/marbleTower.js";
import { createJumpingWhaleDemo } from "./fluid/scenes/jumpingWhale.js";
import { createWaterfallDemo, WATERFALL_ENV_URL, WATERFALL_BELFAST_ENV_URL } from "./fluid/scenes/waterfall.js";
import { createWhiteboardDemo } from "./fluid/scenes/whiteboard.js";

// Particle count is chosen at runtime via the panel dropdown. The PBF rest
// density is pinned (see below) so the count scales the liquid VOLUME, not the
// packing density; MLS-MPM uses the same count so the two methods fill the tank
// comparably. Recreating the sim (createSim) is the only way to resize the
// GPU particle buffers, so the dropdown disposes and rebuilds the active backend.
const PARTICLE_COUNTS = [40000, 80000, 120000, 150000, 200000, 300000, 500000, 750000, 1000000, 1200000, 1500000, 1800000, 2000000];
const DEFAULT_PARTICLE_COUNT = 80000;
const SHARED_FLUID_BINDING_KEYS: readonly (keyof FluidControlValues)[] = [
    "count",
    "renderMode",
    "polygonShader",
    "color",
    "independentRendering",
    "half",
    "thicknessDownscale",
    "absorption",
    "size",
    "physScale",
    "gridPosition",
    "gridSize",
    "gridResolution",
    "markersPerCell",
    "refraction",
    "specular",
    "reflectionExposure",
    "reflectionContrast",
    "reflectivity",
    "depthBlur",
    "depthBlurThreshold",
    "thicknessBlur",
    "surfaceFilter",
    "narrowDelta",
    "narrowMu",
    "anisotropic",
    "anisoSurfScale",
    "activeBlocks",
    "pagedGrid",
    "pagedGridMaxPages",
    "fusedBlockDiscovery",
    "debug",
    "schema",
    "foam",
];

// World-space radius of the interactive Shift+RMB push force (mouse-stir). Shared
// by every demo now the force is core-owned.
const FORCE_RADIUS = 3.5;
const SHARED_HELPER_TEXT = "Drag rotate · RMB slide · Shift+RMB push fluid · wheel zoom · R refill · M switch method · P pause · F8 hide UI";
const clampScale = (s: number, lo: number, hi: number): number => Math.min(Math.max(s, lo), hi);
const applyLatestOnAnimationFrame = <T>(apply: (value: T) => void): ((value: T) => void) => {
    let scheduled = false;
    let latest: T;
    return (value: T): void => {
        latest = value;
        if (scheduled) {
            return;
        }
        scheduled = true;
        requestAnimationFrame(() => {
            scheduled = false;
            apply(latest);
        });
    };
};
// Material 2 = sand. Sand renders as opaque grainy spheres (no water surface) with no velocity
// brightening (uniform grains); its colour comes from the per-material sand preset.
const PBMPM_SAND_MATERIAL = 2;

async function main(): Promise<void> {
    const __initStart = performance.now();
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    const helperText = document.querySelector<HTMLElement>(".hint");
    const syncHelperText = (demo: FluidDemo): void => {
        if (helperText) {
            helperText.textContent = referenceSelected()
                ? "FLIP Reference: " + SHARED_HELPER_TEXT
                : demo.helperText
                  ? `${SHARED_HELPER_TEXT} — ${demo.helperText}`
                  : SHARED_HELPER_TEXT;
        }
    };
    const params = new URLSearchParams(location.search);
    const captureSeconds = Number(params.get("captureSeconds"));
    const captureFixedDt = Number(params.get("fixedDt"));
    const captureMode = Number.isFinite(captureSeconds) && captureSeconds > 0 && Number.isFinite(captureFixedDt) && captureFixedDt > 0;
    const offlineMode = params.get("offline") === "1";
    const offlineFixedDt = Number(params.get("fixedDt"));
    if (offlineMode && (!Number.isFinite(offlineFixedDt) || offlineFixedDt <= 0)) {
        throw new RangeError("[fluid] offline rendering requires a finite positive fixedDt query parameter.");
    }
    const deterministicMode = captureMode || offlineMode;
    const captureTargetSteps = captureMode ? Math.ceil(captureSeconds / captureFixedDt) : 0;
    const captureDemoKey = params.get("demo");
    const captureMethod = params.get("method");
    const captureQuality = params.get("quality") as Quality | null;

    // Single-sample so the swapchain RT is the direct render target and the
    // particle task can share one single-sample depth buffer with the scene.
    // Request the adapter's max storage-buffer limits so the very fine neighbour
    // grids used by small physics particle sizes fit (the defaults cap at 128 MB).
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
        // Fall back to default limits; small physics sizes will be unavailable.
    }
    const engine = await createEngine(canvas, { msaaSamples: 1, requiredLimits });

    // We own the frame graph: scene render task → particle render task, both
    // writing the swapchain colour and sharing a depth buffer we control.
    const scene = createSceneContext(engine, { defaultRenderTask: false });
    await enableMirroredMeshes(scene);

    const DEFAULT_CAMERA = { alpha: -Math.PI / 2, beta: 1.1, radius: 30, target: [0, 6, 0] as const, fov: 0.8, mirrorX: false };
    const cam = createArcRotateCamera(DEFAULT_CAMERA.alpha, DEFAULT_CAMERA.beta, DEFAULT_CAMERA.radius, {
        x: DEFAULT_CAMERA.target[0],
        y: DEFAULT_CAMERA.target[1],
        z: DEFAULT_CAMERA.target[2],
    });
    cam.nearPlane = 0.1;
    cam.farPlane = 200;
    let cameraMirrorX = false;
    let updateCameraMirrorPresentation: (() => void) | null = null;
    const setCameraMirrorX = (enabled: boolean): void => {
        if (cameraMirrorX === enabled) {
            canvas.dataset.cameraMirrorX = String(enabled);
            return;
        }
        cameraMirrorX = enabled;
        canvas.dataset.cameraMirrorX = String(enabled);
        updateCameraMirrorPresentation?.();
    };
    scene.camera = cam;
    // Shift+RMB is the "push the fluid" gesture in every demo; the camera must
    // ignore it (no pan) so the core's force-drag can own it. A demo may also claim
    // a plain pointerdown (capsule: LMB over the tank punches a hole) via
    // claimsPointer — the camera ignores those too. Everything else rotates (LMB) /
    // slides (RMB) / zooms (wheel) through the built-in arc control.
    const isForceGesture = (e: PointerEvent): boolean => e.button === 2 && e.shiftKey;
    onSceneDispose(
        scene,
        attachFluidCameraControls(cam, canvas, scene, () => cameraMirrorX, {
            shouldHandlePointerDown: (e) => !isGizmoInteracting(canvas) && !isForceGesture(e) && !activeDemo?.claimsPointer?.(e),
            isExternalDragActive: () => isGizmoDragging(canvas),
            isExternalPickPending: () => isGizmoPickPending(canvas),
        })
    );

    // Ambient fill. It exists for the STANDARD-material demos (capsule / box / fountain), which
    // sample no environment map at all and would otherwise be lit by the sun alone. The waterfall
    // is PBR under a full HDR IBL, so for that demo this is a second ambient term stacked on the
    // first — and one that no shadow can touch, which is what used to flatten its shadows. It is
    // exposed on the ctx so that demo can dim it while it is on screen (see waterfall.ts).
    const ambient = createHemisphericLight([0.3, 1, 0.4], 0.75);
    addToScene(scene, ambient);
    // Directional "sun". The PBR meshes + outdoor HDR IBL already carry most of the
    // lighting, so its intensity is tuned to add shape without blowing the scene out.
    // Only the waterfall casts shadows, and only when its "Sun shadows" toggle is on — see
    // `setSunShadows` below for how that toggle is wired (the generator stays attached; the
    // CASTER LIST is what switches).
    const sun = createDirectionalLight([-0.5, -0.72, -0.48], 2.4);
    sun.position.set(16, 24, 15);
    addToScene(scene, sun);
    // PCF rather than CSM: this scene is a single compact formation framed by one camera,
    // so a single ortho slice covers it — cascades would buy nothing for the extra maps.
    //
    // The ortho depth range is SYMMETRIC about the light's own position, which is the convention
    // `computeDirectionalLightMatrix` is built for (see scene66: shadowMinZ -10 / shadowMaxZ 10
    // with the light left at the origin). The light camera sits AT `light.position` looking along
    // `direction`, and this range is measured from there — so a symmetric range lets the light sit
    // in the middle of the scene and still cover geometry on both sides of it. A one-sided range
    // like [1, 400] only works if the light is first stood off outside the scene, and getting that
    // stand-off wrong silently clips casters out of the map.
    // `normalBias` is not read by the PCF path, so it is not passed.
    // The bias suits an ortho box that has to hold the whole island (~100 world units) once the
    // oasis ring is a caster: at 2048² that is ~0.05 world units per texel. Expressed in NDC, so
    // the world-space slack is bias/2 × the depth range — here 0.0005 × 500 ≈ 0.25 units.
    const sunShadow = createPcfDirectionalShadowGenerator(engine, sun, { mapSize: 2048, bias: 0.001, orthoMinZ: -250, orthoMaxZ: 250 });
    // The generator stays attached for the whole session, and the TOGGLE is the caster list.
    // That matters because `receiveShadows` is folded into a material variant at renderable
    // build time (pbr-renderable) and gated on "does any light have a generator right now" —
    // so detaching would permanently build the rock without its shadow-receiving path, and
    // re-attaching later would do nothing. Leaving it attached keeps that variant stable.
    //
    // Costing nothing when off relies on the task's own dirty check: swapping the caster set
    // re-renders the map once, after which every frame early-outs without encoding a pass.
    //
    // The "off" set is a single degenerate triangle rather than an EMPTY array, and that is not
    // cosmetic: a RenderTask with no meshes explicitly added auto-mirrors the whole scene, so an
    // empty caster list makes the shadow pass try to draw every scene mesh with its normal
    // colour material into a depth-only target — which fails outright with "Failed to read the
    // 'format' property from 'GPUColorTargetState'". One zero-area caster keeps the pass
    // explicit, draws nothing, and leaves the map cleared.
    //
    // No startup warm-up is needed: setShadowTaskCasterMeshes parks each new caster set on the
    // generator (`_preloadPending`) and the task skips it until the no-colour material import
    // for that caster family resolves, so enabling shadows mid-session cannot race the import.
    sun.shadowGenerator = sunShadow;
    const nullCaster = createMeshFromData(
        engine,
        "fluid-null-shadow-caster",
        new Float32Array([0, 0, 0, 0.01, 0, 0, 0, 0.01, 0]),
        new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        new Uint32Array([0, 1, 2]),
        new Float32Array(6)
    );
    nullCaster.material = createPbrMaterial({ baseColorFactor: [1, 1, 1, 1] });
    nullCaster.receiveShadows = true; // see waterfall.ts: keeps the PBR group's receiver path compiled
    addToScene(scene, nullCaster);
    setMeshVisible(nullCaster, false);
    setShadowTaskCasterMeshes(sunShadow, [nullCaster]);
    /** Turn the sun's shadow map on/off by swapping the caster list. */
    const setSunShadows = (on: boolean, casters: Mesh[]): void => {
        setShadowTaskCasterMeshes(sunShadow, on && casters.length > 0 ? casters : [nullCaster]);
    };

    // Ground plane the escaping liquid falls onto (hidden by the box demo, whose
    // floor replaces it).
    const ground = createGround(engine, { width: 40, height: 40, subdivisions: 1 });
    const groundMat = createStandardMaterial();
    groundMat.diffuseColor = [0.2, 0.22, 0.26];
    groundMat.specularColor = [0.05, 0.05, 0.05];
    ground.material = groundMat;
    addToScene(scene, ground);

    // Depth buffer owned by the scene task and re-used (loaded) by the particle
    // task so particles depth-test against the ground.
    const depthRT = createRenderTarget({ lbl: "fluid-depth", dFormat: "depth24plus", samples: 1, size: engine });

    // The scene renders to an offscreen colour target (not directly the
    // swapchain) so the fluid surface pass can SAMPLE it for refraction. The
    // fluid task then presents to the swapchain (blit in sphere mode, or a
    // refraction composite in surface mode).
    const sceneColorRT = createRenderTarget({ lbl: "fluid-scene-color", format: engine.format, samples: 1, size: engine });

    // The HDR skybox (loaded via loadEnvironment below) is pushed into the scene
    // as an order-0 renderable, so the scene task draws it FIRST. It fills the
    // whole view (an infinite-distance box re-centred on the camera) and does NOT
    // write depth, so the ground + fluid draw on top of it. The skybox therefore
    // covers every pixel; the colour is never cleared (clr: false) — the skybox
    // overwrites the target.
    const sceneTask = createRenderTask({ name: "scene", rt: sceneColorRT, depth: depthRT, clr: false }, engine, scene);
    addTask(scene, sceneTask);

    // Per-demo scene SDF. Each demo owns the sceneSdfParams UBO layout + packing
    // (offset 0 region) and the matching WGSL (`fn sceneSdf(pt, dt)`, positive
    // inside the fluid); the sims inject the code and bind the buffer. The shared
    // buffer holds the demo params (offset 0) plus the drain-hole ring (offset 32)
    // managed here by addSceneHole/clearSceneHoles.
    const sceneSdfBuffer = engine._device.createBuffer({
        label: "fluid-scene-sdf",
        size: 160, // capsule = a + b + 8 holes (10 × vec4); box = lo/hi + paddle (4 × vec4)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const whiteboardMlsContainerBuffer = engine._device.createBuffer({
        label: "fluid-whiteboard-mls-container",
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const whiteboardMlsContainerSdf: SceneSdfSpec = {
        struct: "struct SceneSdfParams { lo: vec4<f32>, hi: vec4<f32>, };",
        sdf: `fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    let fromLo = pt - sceneSdfParams.lo.xyz;
    let fromHi = sceneSdfParams.hi.xyz - pt;
    return min(min(min(fromLo.x, fromLo.y), fromLo.z), min(min(fromHi.x, fromHi.y), fromHi.z));
}`,
        buffer: whiteboardMlsContainerBuffer,
    };

    // Drain-hole ring buffer (capsule demo). Holes live in the capsule SDF UBO
    // (after a, b) as sphere subtracts that carve the solid wall shell, opening a
    // through-passage the fluid drains out of (the unified SDF then confines it on
    // the outside — no per-particle escaped state). Owned here so holes survive
    // PBF↔MPM switches.
    const MAX_SCENE_HOLES = 8;
    const sceneHoles = new Float32Array(MAX_SCENE_HOLES * 4); // (cx, cy, cz, radius) × 8
    let sceneHoleSlot = 0;
    function addSceneHole(center: [number, number, number], radius: number): void {
        const o = sceneHoleSlot * 4;
        sceneHoles[o] = center[0];
        sceneHoles[o + 1] = center[1];
        sceneHoles[o + 2] = center[2];
        sceneHoles[o + 3] = radius;
        sceneHoleSlot = (sceneHoleSlot + 1) % MAX_SCENE_HOLES;
        engine._device.queue.writeBuffer(sceneSdfBuffer, 32, sceneHoles); // holes start after a, b
    }
    function clearSceneHoles(): void {
        sceneHoles.fill(0);
        sceneHoleSlot = 0;
        engine._device.queue.writeBuffer(sceneSdfBuffer, 32, sceneHoles);
    }

    // Shared scene geometry for both backends so switching is apples-to-apples.
    // The domain spans both containers (capsule drain spread + the taller box).
    const BOUNDS_MIN: [number, number, number] = [-20, 0, -20];
    const BOUNDS_MAX: [number, number, number] = [20, 20, 20];
    const defaultDomainBounds = (method: string, scale = 1): FluidDomainBounds => ({
        min: [-20 * scale, (method === "PBF" || method === "FLIP" ? 0 : -1) * scale, -20 * scale],
        max: BOUNDS_MAX.map((value) => value * scale) as [number, number, number],
    });
    const defaultGridSettings = (method: string, scale = 1): FluidGridSettings => {
        const bounds = defaultDomainBounds(method, scale);
        return { position: gridPositionForBounds(bounds), size: gridSizeForBounds(bounds) };
    };
    const cloneGridSettings = (grid: FluidGridSettings): FluidGridSettings => ({ position: [...grid.position], size: [...grid.size] });
    const validGridSettings = (grid: FluidGridSettings): boolean => grid.position.every(Number.isFinite) && grid.size.every((value) => Number.isFinite(value) && value > 0);
    const gridSettingsEqual = (a: FluidGridSettings | undefined, b: FluidGridSettings | undefined): boolean =>
        a === undefined || b === undefined ? a === b : a.position.every((value, index) => value === b.position[index]) && a.size.every((value, index) => value === b.size[index]);
    // Particle-buffer / device-limit sizing policy lives in the core (allocation-plan); the host only
    // supplies its plain-number device limits and formats the user-facing copy.
    const deviceLimits = { maxStorageBufferBindingSize: engine._device.limits.maxStorageBufferBindingSize, maxBufferSize: engine._device.limits.maxBufferSize };
    const particleBufferLimit = fluidParticleBufferLimitBytes(deviceLimits);
    const deviceParticleCapacityForMethod = (method: string): number => fluidDeviceParticleCapacity(deviceLimits, method);
    const syncDeviceParticleCapacity = (method: string): void => {
        canvas.dataset.deviceParticleCapacity = String(deviceParticleCapacityForMethod(method));
        canvas.dataset.deviceParticleBytesPerSlot = String(fluidParticleBytesPerSlot(method));
        canvas.dataset.deviceParticleBufferLimitBytes = String(particleBufferLimit);
    };
    syncDeviceParticleCapacity("PBF");
    const particleAllocationError = (count: number, method: string): string | undefined => {
        const bytesPerSlot = fluidParticleBytesPerSlot(method);
        const deviceParticleCapacity = deviceParticleCapacityForMethod(method);
        return count > deviceParticleCapacity
            ? "Particle allocation requires " +
                  ((count * bytesPerSlot) / (1024 * 1024)).toFixed(1) +
                  " MiB for " +
                  method +
                  " particle buffers; this WebGPU device supports at most " +
                  (particleBufferLimit / (1024 * 1024)).toFixed(1) +
                  " MiB (" +
                  deviceParticleCapacity.toLocaleString() +
                  " particles)."
            : undefined;
    };
    const flipDiscretizationForGrid = (grid: FluidGridSettings, resolution: number, markersPerCell = flipMarkersPerCell) => {
        const bounds = gridBounds(grid.position, grid.size);
        return resolveFlipDiscretization({
            boundsMin: bounds.min,
            boundsMax: bounds.max,
            gridResolution: resolution,
            markersPerCell,
        });
    };
    const fluidDiscretization = (physicsParticleSize: number) => ({ physicsParticleSize, samplingType: "fluid" as const });
    const gridCellsForSettings = (grid: FluidGridSettings, method: string, physicsSize: number, flipResolution = flipGridResolution): [number, number, number] =>
        method === "FLIP"
            ? flipDiscretizationForGrid(grid, flipResolution).gridDim
            : gridCellsForSize(grid.size, fluidSimulationCellSize(method, fluidDiscretization(physicsSize)));
    const gridAllocationError = (
        grid: FluidGridSettings,
        method: string,
        physicsSize: number,
        paging: { enabled: boolean; maxPages: number } = method === "MLS-MPM"
            ? { enabled: mpmPagedGrid, maxPages: mpmPagedGridMaxPages }
            : { enabled: flipPagedGrid, maxPages: flipPagedGridMaxPages },
        flipResolution = flipGridResolution
    ): string | undefined => {
        const cells = gridCellsForSettings(grid, method, physicsSize, flipResolution);
        let allocationMethod: FluidSimulationOptions["method"];
        if (method === "PBF" || method === "FLIP" || method === "MLS-MPM" || method === "PB-MPM") {
            allocationMethod = method;
        } else {
            throw new RangeError(`[fluid] unsupported allocation method ${method}.`);
        }
        return resolveFluidGridCompatibility(allocationMethod, cells, deviceLimits, allocationMethod === "FLIP" || allocationMethod === "MLS-MPM" ? paging : undefined).message;
    };

    // Gridless presets retain the historical hidden domain multiplier. Once a pair has an
    // explicit grid, its position/size are exact world units and Physics particle size alone
    // controls particle radius and cubic cell size.
    let domainScale = 1;
    let builtDomainScale = 1;
    let methodName = "PBF";
    let flipBackendId: "flip-reference" | undefined;
    let referenceBackend: FluidSimulationBackend | undefined;
    let referenceBackendLoading: Promise<FluidSimulationBackend> | undefined;
    let referencePump: FluidReferencePump | null = null;
    let referencePausedByError = false;
    let engineLoopReady = false;
    let normalEngineLoopRunning = false;
    let insideReferenceSubstep = false;
    let stagingBackendSwitch = false;
    const referenceSelected = (): boolean => methodName === "FLIP" && flipBackendId === "flip-reference";
    const activeBackendId = (): "flip-reference" | undefined => (activeSim.options.backend?.id === "flip-reference" ? "flip-reference" : undefined);
    const sharedStateKey = (demoKey: string, backendId?: "flip-reference"): string => (backendId ? demoKey + ":b" + backendId : demoKey);
    function selectedPhysics(values: Readonly<Record<string, number>>): Record<string, number> {
        if (!referenceSelected()) {
            return { ...values };
        }
        const result: Record<string, number> = {};
        for (const key of referenceBackend!.physicsParameters) {
            if (values[key] !== undefined) {
                result[key] = values[key]!;
            }
        }
        return result;
    }
    async function ensureReferenceBackend(): Promise<void> {
        referenceBackendLoading ??= loadFlipReferenceBackend();
        referenceBackend = await referenceBackendLoading;
    }
    function deferReferenceMutation(action: () => void | Promise<void>): boolean {
        if (!referencePump?.pending || insideReferenceSubstep) {
            return false;
        }
        void queueFluidReferenceChange(referencePump, action);
        return true;
    }
    let controlsBinding: FluidControlsBinding | null = null;
    let simulationSemantics: FluidSimulationSemantics = CURRENT_FLUID_SIMULATION_SEMANTICS;
    let gridSettings: FluidGridSettings | undefined;
    let builtGridSettings: FluidGridSettings | undefined;
    let builtGridMethod = methodName;
    let builtPhysicsScale = 1;
    let activeDemo: FluidDemo | null = null;
    let importedCollisionActive = false;
    let builtWithGridFloor = false;
    let showGridBounds = false;
    let showGridBoundsSolid = false;
    let showGridGizmo = false;
    let flipGridResolution = 160;
    let builtFlipGridResolution = flipGridResolution;
    let flipMarkersPerCell = FLIP_DEFAULT_MARKERS_PER_CELL;
    let builtFlipMarkersPerCell = flipMarkersPerCell;
    let importedInitialState: BlenderFluidInitialState | null = null;
    let builtImportedInitialState: BlenderFluidInitialState | null = null;

    // ── Single active fluid backend (review finding 11) ──────────────────────
    // Only the SELECTED solver is ever constructed. Capacity / grid / paging /
    // method changes rebuild that one backend and dispose the outgoing one, so no
    // interaction path synchronously allocates the three unrelated backends — which
    // previously cost multi-second stalls and up to 4x peak GPU memory. These
    // per-method build tallies plus the active-backend / transition-byte read-outs
    // make that guarantee observable to the memory diagnostics and the regressions.
    const backendBuildCounts: Record<string, number> = { PBF: 0, FLIP: 0, "MLS-MPM": 0, "PB-MPM": 0 };
    let totalBackendBuilds = 0;
    let activeSimMethod = methodName;
    let lastTransitionPeakBytes = 0;
    function publishBackendDiagnostics(): void {
        canvas.dataset.simulationActiveBackend = activeSimMethod;
        canvas.dataset.fluidBackend = referenceSelected() ? "flip-reference" : "production";
        canvas.dataset.simulationBackendBuilds = String(totalBackendBuilds);
        canvas.dataset.backendBuildsPbf = String(backendBuildCounts.PBF ?? 0);
        canvas.dataset.backendBuildsFlip = String(backendBuildCounts.FLIP ?? 0);
        canvas.dataset.backendBuildsMlsMpm = String(backendBuildCounts["MLS-MPM"] ?? 0);
        canvas.dataset.backendBuildsPbMpm = String(backendBuildCounts["PB-MPM"] ?? 0);
    }
    function trackSimBuild(method: string): void {
        backendBuildCounts[method] = (backendBuildCounts[method] ?? 0) + 1;
        totalBackendBuilds++;
        activeSimMethod = method;
        publishBackendDiagnostics();
    }

    // The selected backend is (re)built by createSim so the particle-count dropdown
    // can resize the GPU buffers (the only way to change count is to reallocate). The
    // capsule tank geometry seeds the sim's built-in fallback confinement (a legacy
    // default; the demo's injected sceneSdf always overrides it). Exactly one of the
    // four candidate solvers is instantiated — the others short-circuit to null.
    function simulationOptions(count: number, scale: number): FluidSimulationOptions {
        const allocationError = particleAllocationError(count, methodName);
        if (allocationError) {
            throw new RangeError(allocationError);
        }
        const inactiveCount = Math.min(count, DEFAULT_PARTICLE_COUNT);
        const pbfCount = methodName === "PBF" ? count : inactiveCount;
        const flipCount = methodName === "FLIP" ? count : inactiveCount;
        const mpmCount = methodName === "MLS-MPM" ? count : inactiveCount;
        const pbmpmCount = methodName === "PB-MPM" ? count : inactiveCount;
        // `scale` is the physics particle-size multiplier. The particle COUNT is
        // the user's choice and stays fixed, so each particle is a bigger (or
        // smaller) blob of fluid and the liquid VOLUME scales with the size: the
        // tank overfills at large sizes and shrinks to a small puddle at tiny
        // sizes. Implemented as a uniform rescaling of one particle's footprint —
        // the smoothing radius / grid cell and the rest spacing both grow by the
        // scale, so the per-particle volume grows ∝ scale³:
        //   • PBF: restDensity ∝ 1/scale³ (the poly6 kernel sum of the scaled
        //     radius shrinks by scale³); relaxation ∝ 1/scale² (the additive ε in
        //     λ = -C/(Σ|∇C|² + ε) must track the rescaling or the liquid collapses).
        //   • MLS-MPM: cell size dx grows by scale; restDensity (particles/cell)
        //     stays fixed so the per-particle volume = dx³/restDensity grows ∝ scale³.
        // Extreme scale values remain opt-in because timestep and density settings
        // may also need adjustment.
        const pbfScale = methodName === "PBF" ? clampScale(scale, PBF_MIN_SCALE, PBF_MAX_SCALE) : 1;
        const mpmScale = methodName === "MLS-MPM" ? clampScale(scale, MPM_MIN_SCALE, MPM_MAX_SCALE) : 1;
        const pbmpmScale = methodName === "PB-MPM" ? clampScale(scale, PBMPM_MIN_SCALE, PBMPM_MAX_SCALE) : 1;
        const ds = domainScale;
        const scaleTriple = (t: [number, number, number]): [number, number, number] => [t[0] * ds, t[1] * ds, t[2] * ds];
        const explicitGrid = gridSettings !== undefined;
        const explicitBounds = gridSettings ? gridBounds(gridSettings.position, gridSettings.size) : undefined;
        // The "fluid" compatibility profile is the single home for this host's discretization policy:
        // it reproduces the previous explicit-grid (shared fluidSimulation* discretization) and the
        // gridless legacy fallback (0.09*scale radius, cellSizeForPhysicsScale cell size, both x domain
        // scale) exactly. FLIP derives its own grid from the resolution, so only non-FLIP backends use it.
        const activeScale = methodName === "PBF" ? pbfScale : methodName === "MLS-MPM" ? mpmScale : methodName === "PB-MPM" ? pbmpmScale : 1;
        const nonFlipConfig =
            methodName === "FLIP"
                ? null
                : resolveFluidSimulationConfig("fluid", {
                      method: methodName,
                      physicsScale: explicitGrid ? scale : activeScale,
                      explicitGrid,
                      domainScale: ds,
                      semantics: simulationSemantics,
                      ...(explicitBounds ? { bounds: explicitBounds } : {}),
                  });
        if (gridSettings) {
            const allocationError = gridAllocationError(gridSettings, methodName, scale);
            if (allocationError) {
                throw new Error(allocationError);
            }
        }
        const pbfBoundsMin = explicitBounds?.min ?? scaleTriple(BOUNDS_MIN);
        const pbfBoundsMax = explicitBounds?.max ?? scaleTriple(BOUNDS_MAX);
        const mpmBoundsMin = explicitBounds?.min ?? [pbfBoundsMin[0], -1 * ds, pbfBoundsMin[2]];
        const mpmBoundsMax = explicitBounds?.max ?? pbfBoundsMax;
        const useGridFloor = importedCollisionActive || activeDemo?.useGridFloor === true;
        const pbfGroundY = useGridFloor ? pbfBoundsMin[1] : 0;
        const mpmGroundY = useGridFloor ? mpmBoundsMin[1] : 0;
        canvas.dataset.simulationGroundY = String(methodName === "PBF" || methodName === "FLIP" ? pbfGroundY : mpmGroundY);
        // Backend 1 — Position Based Fluids (the original solver).
        const pbf =
            methodName === "PBF"
                ? ({
                      method: "PBF",
                      particleCount: pbfCount,
                      bounds: { min: pbfBoundsMin, max: pbfBoundsMax },
                      physicsScale: explicitGrid ? scale : pbfScale,
                      compatibilityProfile: "fluid",
                      semantics: simulationSemantics,
                      explicitGrid,
                      domainScale: ds,
                      particleRadius: nonFlipConfig!.particleRadius,
                      groundY: pbfGroundY,
                      physics: {
                          restDensity: 341,
                          relaxation: 50,
                      },
                  } satisfies FluidSimulationOptions)
                : null;

        // Backend 2 — FLIP (marker particles + incompressible staggered MAC grid).
        const flip =
            methodName === "FLIP"
                ? ({
                      method: "FLIP",
                      particleCount: flipCount,
                      bounds: { min: pbfBoundsMin, max: pbfBoundsMax },
                      physicsScale: 1,
                      compatibilityProfile: "fluid",
                      semantics: simulationSemantics,
                      explicitGrid,
                      domainScale: ds,
                      gridResolution: flipGridResolution,
                      markersPerCell: flipMarkersPerCell,
                      ...(importedInitialState
                          ? {
                                initialPositions: importedInitialState.positions,
                                initialVelocities: importedInitialState.velocities,
                            }
                          : {}),
                      groundY: pbfGroundY,
                      physics: {
                          gravity: 9.8,
                          flipRatio: 0.95,
                          pressureIterations: 40,
                          pressureRelaxation: 0.8,
                          velocityDamping: 0,
                          kinematicViscosity: 0,
                          viscosityIterations: 12,
                          surfaceTension: 0,
                          restitution: 0,
                          minSubsteps: 1,
                          maxSubsteps: 8,
                          cflNumber: 2,
                      },
                      pagedGrid: flipPagedGrid,
                      pagedGridMaxPages: flipPagedGridMaxPages,
                      onPagedGridPages: (requiredPages, capacity) => {
                          if (controlsBinding) {
                              syncFluidControls(controlsBinding, { pageDiagnostics: { method: "FLIP", requiredPages, capacity } });
                          } else {
                              controls.setPagedGridStatus(formatFluidPageDiagnostics(requiredPages, capacity));
                              canvas.dataset.pagedGridPages = String(requiredPages);
                              canvas.dataset.pagedGridPageCapacity = String(capacity);
                          }
                      },
                      onPagedGridOverflow: (requiredPages, capacity) => {
                          const message = `Page capacity exceeded: ${requiredPages.toLocaleString()} required, ${capacity.toLocaleString()} allocated. Increase Page capacity.`;
                          if (controlsBinding) {
                              syncFluidControls(controlsBinding, { pageDiagnostics: { method: "FLIP", requiredPages, capacity, overflow: true } });
                          } else {
                              controls.setPagedGridStatus(message, true);
                              canvas.dataset.pagedGridOverflow = "true";
                          }
                          console.error(`[FLIP] ${message}`);
                      },
                  } satisfies FluidSimulationOptions)
                : null;

        // Backend 3 — MLS-MPM (grid-transfer; scales to far more particles).
        const mpm =
            methodName === "MLS-MPM"
                ? ({
                      method: "MLS-MPM",
                      particleCount: mpmCount,
                      bounds: { min: mpmBoundsMin, max: mpmBoundsMax },
                      physicsScale: explicitGrid ? scale : mpmScale,
                      compatibilityProfile: "fluid",
                      semantics: simulationSemantics,
                      explicitGrid,
                      domainScale: ds,
                      particleRadius: nonFlipConfig!.particleRadius,
                      groundY: mpmGroundY,
                      // Drop the MLS grid floor below every demo floor (box/fountain at y=0,
                      // capsule bottom at y=2) so a demo floor is confined by its own scene
                      // SDF (like the side walls) rather than the grid's 2-cell domain-border
                      // v.y=0 zone — which coincided with y=0 and cancelled gravity there,
                      // leaving the fluid hovering a row above the floor (PBF hard-clamps, so
                      // it sat flush). The border now sits harmlessly below all demo floors.
                      // The -1 floor offset scales with the domain too so the grid dims stay constant.
                      physics: {
                          restDensity: 3,
                          stiffness: 350,
                          gravity: 9.8,
                          viscosity: 0.3,
                          substeps: 3,
                          damping: 0.995,
                          affineDamping: 0.9,
                          groundDamp: 0.85,
                          groundDampHeight: 1.5,
                      },
                      activeBlocks: mpmActiveBlocks,
                      pagedGrid: mpmPagedGrid,
                      pagedGridMaxPages: mpmPagedGridMaxPages,
                      fusedBlockDiscovery: mpmFusedBlockDiscovery,
                      onPagedGridPages: (requiredPages, capacity) => {
                          if (controlsBinding) {
                              syncFluidControls(controlsBinding, { pageDiagnostics: { method: "MLS-MPM", requiredPages, capacity } });
                          } else {
                              controls.setPagedGridStatus(formatFluidPageDiagnostics(requiredPages, capacity));
                          }
                      },
                      onPagedGridOverflow: (requiredPages, capacity) => {
                          const message = `Page capacity exceeded: ${requiredPages.toLocaleString()} required, ${capacity.toLocaleString()} allocated. Increase Page capacity.`;
                          if (controlsBinding) {
                              syncFluidControls(controlsBinding, { pageDiagnostics: { method: "MLS-MPM", requiredPages, capacity, overflow: true } });
                          } else {
                              controls.setPagedGridStatus(message, true);
                              canvas.dataset.pagedGridOverflow = "true";
                          }
                          console.error(`[MLS-MPM] ${message}`);
                      },
                  } satisfies FluidSimulationOptions)
                : null;

        // Backend 4 — Position-Based MPM.
        const pbmpm =
            methodName === "PB-MPM"
                ? ({
                      method: "PB-MPM",
                      particleCount: pbmpmCount,
                      bounds: { min: mpmBoundsMin, max: mpmBoundsMax },
                      physicsScale: explicitGrid ? scale : pbmpmScale,
                      compatibilityProfile: "fluid",
                      semantics: simulationSemantics,
                      explicitGrid,
                      domainScale: ds,
                      particleRadius: nonFlipConfig!.particleRadius,
                      groundY: mpmGroundY,
                      physics: {
                          gravity: 9.8,
                          substeps: 3,
                          iterations: 5,
                          liquidRelaxation: 1.5,
                          liquidViscosity: 0.01,
                          elasticityRatio: 0.3,
                          elasticRelaxation: 0.3,
                          frictionAngle: 35,
                          plasticity: 0.8,
                          restitution: 0,
                      },
                      material: pbmpmMaterial,
                  } satisfies FluidSimulationOptions)
                : null;

        // Exactly one candidate matched methodName; the rest stayed null and cost no
        // GPU allocation. Fail loudly rather than silently on an unknown method.
        const built: FluidSimulationOptions | null = pbf ?? flip ?? mpm ?? pbmpm;
        if (!built) {
            throw new Error(`Cannot build fluid backend for unknown method "${methodName}".`);
        }
        built.backend = undefined;
        if (referenceSelected()) {
            if (!referenceBackend) {
                throw new Error("[fluid] FLIP Reference must finish loading before creating its simulation.");
            }
            built.backend = referenceBackend;
            built.physics = selectedPhysics(built.physics ?? {});
            built.pagedGrid = false;
        }
        return built;
    }

    function createSim(count: number, scale: number): FluidSimulation {
        const simulation = createFluidSimulation(engine, simulationOptions(count, scale));
        trackSimBuild(methodName);
        return simulation;
    }

    let particleCount = DEFAULT_PARTICLE_COUNT;
    // FLIP may allocate only the particles needed by `initial` emitters, but an
    // inflow still needs the authored capacity restored when its behavior is
    // switched back. Keep that request separate from the current solver allocation.
    let flipParticleCapacityRequest = DEFAULT_PARTICLE_COUNT;
    let physicsScale = 1; // physics particle-size multiplier (rebuilds sims)
    let pbmpmMaterial = 0;
    let flipPagedGrid = false;
    const fluidDeviceLimits = engine._device.limits;
    const maxFlipPagedGridPagesFor = (grid: FluidGridSettings, resolution: number): number => {
        const dim = flipDiscretizationForGrid(grid, resolution).gridDim;
        return fluidMaximumPageCapacity("FLIP", dim, fluidDeviceLimits);
    };
    const maxMlsPagedGridPagesFor = (grid: FluidGridSettings, scale: number): number =>
        fluidMaximumPageCapacity("MLS-MPM", gridCellsForSettings(grid, "MLS-MPM", scale), fluidDeviceLimits);
    const maxFlipPagedGridPages = fluidMaximumPageCapacity("FLIP", [FLUID_GRID_MAX_AXIS_CELLS, FLUID_GRID_MAX_AXIS_CELLS, FLUID_GRID_MAX_AXIS_CELLS], fluidDeviceLimits);
    let flipPagedGridMaxPages = Math.min(maxFlipPagedGridPages, FLIP_DEFAULT_PAGE_CAPACITY);
    let mpmActiveBlocks = false;
    let mpmPagedGrid = false;
    const maxPagedGridPages = fluidMaximumPageCapacity("MLS-MPM", [FLUID_GRID_MAX_AXIS_CELLS, FLUID_GRID_MAX_AXIS_CELLS, FLUID_GRID_MAX_AXIS_CELLS], fluidDeviceLimits);
    let mpmPagedGridMaxPages = Math.min(maxPagedGridPages, mlsMpmDefaultPageCapacity(DEFAULT_PARTICLE_COUNT));
    let mpmFusedBlockDiscovery = false;
    const activeSim = createSim(particleCount, physicsScale);
    let quality: Quality = DEFAULT_QUALITY; // low/middle/high preset tier (panel dropdown)
    /** Demos the user has already opened once, so FluidDemo.defaultMethod/defaultQuality are
     *  honoured on the first visit only. Seeded with the start-up demo below. */
    const visitedDemos = new Set<string>();
    // Interactive push force (Shift+RMB mouse-stir): a ready-made injectable force
    // field shared by both backends. The frame loop drives it via setRay + toggles
    // it on the ACTIVE sim via setForceField, so it dispatches its own dedicated
    // compute pass ONLY while a push is active (and compiles lazily on first use).
    const rayForce = createRayForce(engine._device);
    const forceFieldHandles = new WeakMap<ForceFieldSpec, FluidForceField>();
    function forceFieldHandle(spec: ForceFieldSpec | null): FluidForceField | null {
        if (!spec) {
            return null;
        }
        let handle = forceFieldHandles.get(spec);
        if (!handle) {
            handle = adoptFluidForceField(engine, spec);
            forceFieldHandles.set(spec, handle);
        }
        return handle;
    }
    let activeFlow: FluidFlowConfig = { emitters: [], sinks: [] };
    let installedFlow: FluidFlowConfig = { emitters: [], sinks: [] };
    let flowEditor: FluidFlowEditor | null = null;
    let builtInitialStateKey = "";
    interface ImportedEmitterSourceBinding {
        emitterId: string;
        node: SceneNode;
        transform: EmitterSourceTransform;
        lastPosition: [number, number, number];
    }
    interface ImportedAnimatedCollisionBinding {
        id: string;
        node: SceneNode;
    }
    interface ImportedCollisionResources {
        sceneSdf: FluidSceneSdf;
        collisionOrigin: [number, number, number];
    }
    interface ImportedFluidScene extends ImportedCollisionResources {
        asset: AssetContainer;
        assetRoot: SceneNode;
        assetRootPosition: [number, number, number];
        assetRootScaling: [number, number, number];
        referenceGridPosition: [number, number, number];
        gridOffset: [number, number, number];
        groundWasVisible: boolean;
        bundle: BlenderFluidScene;
        sourceBindings: ImportedEmitterSourceBinding[];
        animatedCollisionBindings: ImportedAnimatedCollisionBinding[];
        collisionDebugMeshes: Map<string, Mesh>;
        collisionOffsets: Map<string, [number, number, number]>;
    }
    let importedScene: ImportedFluidScene | null = null;
    let transientSceneMeshScale = 1;
    let collisionControlsHost: HTMLDivElement | null = null;
    let externalSceneExportRow: HTMLLabelElement | null = null;
    let embedExternalSceneChk: HTMLInputElement | null = null;
    let suppressPairSnapshot = false;
    let importGeneration = 0;

    const scaleGridSettingsAroundPivot = (grid: FluidGridSettings, scale: number, pivot: readonly [number, number, number]): FluidGridSettings => ({
        position: [pivot[0] + (grid.position[0] - pivot[0]) * scale, pivot[1] + (grid.position[1] - pivot[1]) * scale, pivot[2] + (grid.position[2] - pivot[2]) * scale],
        size: [grid.size[0] * scale, grid.size[1] * scale, grid.size[2] * scale],
    });
    const scaleFlowTransforms = (flow: FluidFlowConfig, scale: number): FluidFlowConfig => {
        const scaled = structuredClone(flow);
        for (const object of [...scaled.emitters, ...scaled.sinks]) {
            object.transform.position = object.transform.position.map((value) => value * scale) as [number, number, number];
            object.transform.scale = object.transform.scale.map((value) => value * scale) as [number, number, number];
        }
        return scaled;
    };

    // Composite target for the whole fluid chain. The surface / foam / container-overlay
    // passes write HERE instead of straight to the swapchain, so a post-process stage can
    // read the finished frame and present it. `size: engine` keeps it canvas-sized across
    // resizes, and the format matches the swapchain so the final pass is a 1:1 resample.
    const postRT = createRenderTarget({ lbl: "fluid-post-color", format: engine.format, samples: 1, size: engine });

    const simulationCollection = createFluidSimulationCollection(engine, [activeSim]);
    const particleTask = attachFluidSimulationRenderLayer(activeSim, {
        scene,
        camera: cam,
        mode: "spheres",
        colorTarget: sceneColorRT,
        depthTarget: depthRT,
    });
    // Fluid surface renderer + frame compositor: reads the offscreen scene colour
    // and writes the composite target. In sphere mode it just blits the scene (with the
    // impostors already drawn into it); in surface mode it reconstructs and
    // shades the liquid surface (refraction of the scene).
    const surfaceTask = attachFluidSimulationRenderLayer(activeSim, {
        scene,
        camera: cam,
        mode: "surface",
        backgroundTarget: sceneColorRT,
        outputTarget: postRT,
        depthTarget: depthRT,
    });
    const polygonSurfaceTask = attachFluidSimulationRenderLayer(activeSim, {
        scene,
        camera: cam,
        mode: "polygon",
        backgroundTarget: sceneColorRT,
        outputTarget: postRT,
        depthTarget: depthRT,
    });

    // Foam (diffuse-particle) renderer — draws the active sim's spray/foam/bubble pool
    // as sprites OVER the composited fluid surface (added after surfaceTask), depth-
    // tested against the shared scene depth so opaque geometry occludes it. Only the
    // for every backend that exposes setFoam()/diffuse.
    const foamTask = attachFluidSimulationCollectionRenderLayer(simulationCollection, {
        scene,
        camera: cam,
        mode: "foam",
        colorTarget: postRT,
        depthTarget: depthRT,
        surfaceLayer: surfaceTask,
        polygonSurfaceLayer: polygonSurfaceTask,
    });
    let simulationDuration = 0;
    let simulationAlphaDecay = 2;
    let simulationTimeScale = 1;
    let simulationElapsed = 0;
    let simulationOpacity = 1;
    let simulationStopped = false;

    // Container-glass overlay — draws each demo's TRANSLUCENT container mesh (capsule
    // pill, box tank glass) AFTER the fluid surface + foam, straight into the composite
    // target `postRT`, depth-tested (compare, no write) against the shared opaque
    // `depthRT`. Those glass meshes have alpha < 1 and skip depth writes, so if they
    // were drawn (as usual) into the offscreen scene-colour target the fluid surface
    // pass would composite the liquid OVER them. Drawing them last instead means:
    // interior fluid (already in postRT) shows THROUGH the translucent glass;
    //   • ground liquid drained BEHIND the pill is correctly occluded/tinted by the
    //     glass in front of it (the reported bug — liquid was drawn over the capsule);
    //   • opaque geometry in front of the glass still occludes it via the depth test.
    // The container meshes are stripped out of the scene-colour pass below (after
    // registerScene) so they render exactly once, here.
    //
    // A `createRenderTask` reuses the standard-material pipeline/bind-groups (the glass
    // material owns its shader) and the transparent depth state (depthWriteEnabled=false,
    // reverse-Z depthCompare) — no bespoke WGSL. It targets `engine.scRT` (loaded, not
    // cleared: clr=false) with a BORROWED depth: an `_eager` alias of `depthRT` so the
    // task loads (never clears/builds/disposes) the opaque depth the scene pass owns.
    const overlayDepth = createRenderTarget({ lbl: "fluid-overlay-depth", dFormat: depthRT._descriptor.dFormat, samples: 1, size: engine });
    overlayDepth._eager = true; // task loads (loadOp "load") — never builds/clears/disposes it
    overlayDepth._ownsDepthTexture = false; // the shared depth texture is owned by the scene pass
    const overlayTask = createRenderTask({ name: "container-overlay", rt: postRT, depth: overlayDepth, clr: false }, engine, scene);
    // Re-point the borrowed depth view at the scene pass's (possibly resize-rebuilt)
    // depth texture before the task bakes it into its pass descriptor. The scene task
    // is recorded first (added first), so `depthRT._depthView` is already current here.
    const overlayRecord = overlayTask.record.bind(overlayTask);
    overlayTask.record = (): void => {
        overlayDepth._depthTexture = depthRT._depthTexture;
        overlayDepth._depthView = depthRT._depthView;
        overlayDepth._width = depthRT._width;
        overlayDepth._height = depthRT._height;
        overlayRecord();
    };
    addTask(scene, overlayTask);

    // ── Presentation stage: postRT → swapchain, either through BLOOM or a plain blit ──
    // Both paths stay registered in the frame graph for their whole life so each is recorded
    // (and re-recorded on canvas resize) even while inactive; only the active one does any
    // GPU work per frame. That is cheaper and far less fragile than splicing tasks in and out
    // of a live graph.
    //
    // The bloom chain is hand-rolled rather than `createBloomPostProcessTask` because the glow
    // must come from the FLUID ONLY — a stock bloom reads the finished frame, so the bright
    // HDR sky blooms as hard as the water and the whole image hazes over. The mask falls out
    // of two targets the demo already owns: `sceneColorRT` is the background (sky + rock +
    // ground) as it was BEFORE the fluid pass, and `postRT` is the same frame WITH the water,
    // foam and glass composited in. Any pixel the fluid did not touch is byte-identical in
    // both, so `|postRT - sceneColorRT|` is a free, exact "is this the waterfall?" mask —
    // no extra pass, no G-buffer, and it costs one texture fetch in the extract shader.
    // Each intermediate is the target of EXACTLY ONE task — no ping-ponging. `buildRenderTarget`
    // destroys and recreates the texture on every `record()`, so if two tasks targeted the same
    // RT the second one's record would invalidate the view the first one's bind group already
    // captured; sampling that destroyed texture kills the whole command encoder and the frame
    // presents black. Hence three targets: extract → A, blur X → B, blur Y → C, merge(postRT, C).
    const bloomA = createRenderTarget({ lbl: "fluid-bloom-extract", format: engine.format, samples: 1, size: engine });
    const bloomB = createRenderTarget({ lbl: "fluid-bloom-blur-x", format: engine.format, samples: 1, size: engine });
    const bloomC = createRenderTarget({ lbl: "fluid-bloom-blur-y", format: engine.format, samples: 1, size: engine });
    const bloomParams = { intensity: 0.6, threshold: 0.75 };
    /** Luma difference above which a pixel counts as "the fluid drew here". Just above 8-bit
     *  quantisation noise (1/255 ≈ 0.004) so untouched background reliably reads as 0. */
    const BLOOM_FLUID_EPS = 0.02;
    const bloomExtract = createPostProcessTask(
        {
            name: "fluid-bloom-extract",
            sourceTexture: postRT,
            targetTexture: bloomA,
            _shader: {
                extraTextures: [sceneColorRT],
                extraTextureWGSL: wgsl`@group(0) @binding(2) var bloomBackground:texture_2d<f32>;`,
                uniformWGSL: wgsl`struct P{threshold:f32,fluidEps:f32,p0:f32,p1:f32}
@group(0) @binding(3) var<uniform> bloomExtractParams:P;`,
                uniformBinding: 3,
                uniformByteLength: 16,
                writeUniforms(data) {
                    // Match the stock extract pass: the threshold slider is authored in linear
                    // space but compared against a gamma-space luma.
                    data[0] = Math.pow(bloomParams.threshold, 1 / 2.2);
                    data[1] = BLOOM_FLUID_EPS;
                },
                // Gate the usual luminance threshold on the fluid mask. Thresholding the
                // DIFFERENCE itself would be wrong: white water over a bright sky has a small
                // difference and would stop blooming exactly where it is brightest.
                fragmentWGSL: wgsl`fn applyPostProcess(color:vec4f, uv:vec2f)->vec4f{
let bg=textureSampleLevel(bloomBackground,sourceSampler,clamp(uv,vec2f(0),vec2f(1)),0).rgb;
let d=abs(color.rgb-bg);
let isFluid=step(bloomExtractParams.fluidEps,max(d.r,max(d.g,d.b)));
let luma=dot(vec3f(0.2126,0.7152,0.0722),color.rgb);
return vec4f(isFluid*step(bloomExtractParams.threshold,luma)*color.rgb,color.a);}`,
            },
        },
        engine,
        scene
    );
    const bloomBlurX = createBlurPostProcessTask(
        { name: "fluid-bloom-blur-x", sourceTexture: bloomA, sourceSamplingMode: "linear", targetTexture: bloomB, direction: { x: 1, y: 0 }, kernel: 48 },
        engine,
        scene
    );
    const bloomBlurY = createBlurPostProcessTask(
        { name: "fluid-bloom-blur-y", sourceTexture: bloomB, sourceSamplingMode: "linear", targetTexture: bloomC, direction: { x: 0, y: 1 }, kernel: 48 },
        engine,
        scene
    );
    const bloomMerge = createPostProcessTask(
        {
            name: "fluid-bloom-merge",
            sourceTexture: postRT,
            targetTexture: engine.scRT,
            _shader: {
                extraTextures: [bloomC],
                extraTextureWGSL: wgsl`@group(0) @binding(2) var bloomBlur:texture_2d<f32>;`,
                vertexMainWGSL: wgsl`out.uv.x=mix(out.uv.x,1.0-out.uv.x,bloomMergeParams.mirrorX);`,
                uniformWGSL: wgsl`struct M{weight:f32,mirrorX:f32,p1:f32,p2:f32}
@group(0) @binding(3) var<uniform> bloomMergeParams:M;`,
                uniformBinding: 3,
                uniformByteLength: 16,
                writeUniforms(data) {
                    // Weight 0 makes the merge an exact passthrough, which is what "bloom off"
                    // is: this task ALWAYS runs because it is the single owner of the swapchain.
                    data[0] = bloomEnabled ? bloomParams.intensity : 0;
                    data[1] = cameraMirrorX ? 1 : 0;
                },
                fragmentWGSL: wgsl`fn applyPostProcess(color:vec4f, uv:vec2f)->vec4f{
let b=textureSampleLevel(bloomBlur,sourceSampler,clamp(uv,vec2f(0),vec2f(1)),0).rgb;
return vec4f(color.rgb+b*bloomMergeParams.weight,color.a);}`,
            },
        },
        engine,
        scene
    );
    updateCameraMirrorPresentation = () => bloomMerge.updateUniforms();
    let bloomEnabled = false;
    /** Register `task` but let it execute only while `active()` holds. Keeps it in the graph
     *  (so it is recorded + resized normally) while costing nothing when switched off. */
    const gateTask = (task: Task, active: () => boolean): void => {
        const run = task.execute!.bind(task);
        task.execute = (): number => (active() ? run() : 0);
        addTask(scene, task);
    };
    /** Wrap an ALREADY-registered task's execute so it only runs while `active()` holds. */
    const gateExistingTask = (task: Task, active: () => boolean): void => {
        const run = task.execute!.bind(task);
        task.execute = (): number => (active() ? run() : 0);
    };
    // Only the highlight extraction and the two blurs are switchable. The MERGE is always
    // registered and always executes: it is the one and only task that writes the swapchain,
    // and with weight 0 it is a plain blit. (Handing the swapchain to two alternating
    // post-process tasks does NOT work — the inactive one still owns a recorded pass against
    // the same surface and the visible result is a black frame.)
    for (const t of [bloomExtract, bloomBlurX, bloomBlurY]) {
        gateTask(t, () => bloomEnabled);
    }
    addTask(scene, bloomMerge);

    // ── Optional MSAA on the scene (background geometry) pass ────────────────────────────
    // The demo renders the world into an OFFSCREEN single-sample target because the fluid
    // surface pass has to SAMPLE it for refraction, so it gets none of the engine's own MSAA
    // (`createEngine(..., { msaaSamples: 1 })` above) — every polygon edge is hard-aliased.
    // That is invisible on the simple demos, but the waterfall's oasis backdrop is a
    // million-triangle model whose leaf blades land at or below one pixel: each frame a
    // different sub-pixel sliver wins the coverage test and the foliage crawls with white
    // speckle. Rendering that model on its own, 4× MSAA removed 92% of its isolated bright
    // fringe pixels — because at sub-pixel triangle sizes each MSAA sample hits a DIFFERENT
    // triangle, so MSAA degenerates into supersampling exactly where the aliasing is worst.
    // (In the full waterfall frame the headline number is smaller, ~20%, simply because most of
    // the remaining bright fringe is the water itself, which MSAA cannot touch — see below.)
    //
    // Only the SCENE pass is multisampled. The fluid surface is reconstructed in screen space
    // from a depth/thickness buffer rather than rasterised, so MSAA would do nothing for it,
    // and the particle/foam sprites are soft-edged impostors. Multisampling just the one pass
    // that draws hard-edged triangles is where all of the benefit is and a fraction of the cost.
    //
    // Wiring: an MSAA colour+depth target is rendered instead of `sceneColorRT`, and the pass
    // RESOLVES into `sceneColorRT` at the end (`rst`), so every downstream consumer keeps
    // reading the same single-sample texture it always did. Depth needs an explicit pass —
    // WebGPU has no hardware depth resolve — so `createDepthResolveTask` copies sample 0 of the
    // MSAA depth into the shared `depthRT` that the particle, surface, foam and container-glass
    // passes all depth-test against.
    //
    // Task order is [scene-msaa, scene, depth-resolve, particle, …]: the plain scene task must
    // record LAST of the two so the colour view it bakes into its pass descriptor belongs to the
    // live `sceneColorRT` texture (both tasks build that target — one as `rt`, one as `rst`), and
    // the depth resolve must record after whoever (re)builds `depthRT`. Exactly one of the two
    // scene tasks executes per frame, so the plain task never clears the depth the resolve wrote.
    //
    // Built LAZILY on first enable: off is the default, and the MSAA colour + depth textures are
    // ~4× a normal canvas-sized pair, so a user who never asks for it never pays for it.
    const MSAA_SAMPLES = 4;
    let msaaOn = false;
    let msaaSceneTask: ReturnType<typeof createRenderTask> | null = null;
    let pendingFrameGraphRebuild = false;
    gateExistingTask(sceneTask, () => !msaaOn);
    const setMsaa = (on: boolean): void => {
        if (on && !msaaSceneTask) {
            const sceneMsaaRT = createRenderTarget({
                lbl: "fluid-scene-msaa",
                format: engine.format,
                dFormat: depthRT._descriptor.dFormat,
                samples: MSAA_SAMPLES,
                size: engine,
            });
            // No `depth`: the MSAA target owns its own depth attachment (matching the engine's
            // own MSAA scene task), which is what the resolve task then reads.
            msaaSceneTask = createRenderTask({ name: "scene-msaa", rt: sceneMsaaRT, rst: sceneColorRT, clr: hostSceneSuspended }, engine, scene);
            const depthResolveTask = createDepthResolveTask({ name: "fluid-depth-resolve", sourceTexture: sceneMsaaRT, targetTexture: depthRT }, engine, scene);
            gateExistingTask(msaaSceneTask, () => msaaOn);
            gateExistingTask(depthResolveTask, () => msaaOn);
            addTaskBefore(scene, msaaSceneTask, sceneTask);
            addTaskAfter(scene, depthResolveTask, sceneTask);
            // Rebuild so the new tasks record and every canvas-sized target is re-allocated in
            // the right order. Same call the engine makes on a canvas resize, so it is a
            // supported mid-session operation — but it destroys and re-creates textures that
            // other tasks' bind groups and pass descriptors point at, so it must not run from a
            // UI event while a frame is being encoded. Deferred to the top of the next
            // onBeforeRender, ahead of everything this frame encodes against them.
            pendingFrameGraphRebuild = true;
        }
        msaaOn = on;
    };

    // Foam (diffuse-particle) config lives in the shared controls panel (component-owned,
    // per-(demo, method)). `pushFoam` reads its live snapshot and (re)applies it to the
    // ACTIVE sim (null disables). Cheap: re-writes the foam UBO (and reallocates only when
    // poolScale changed). Also gates the renderer so a disabled pool costs nothing to draw.
    // Foam sprites are composited OVER the fluid surface straight into the swapchain, so while
    // a fluid-surface DEBUG texture (depth / normals / thickness) is being visualised they would
    // draw on top of it. Track that mode and suppress the foam render whenever it is active.
    let surfaceDebugActive = false;
    function foamRenderVisible(): boolean {
        return simulationOpacity > 0 && controls.getValues().foam.enabled && !surfaceDebugActive;
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
    function pushFoam(): void {
        const f = controls.getValues().foam;
        const cfg = currentFoamConfig();
        setFluidSimulationFoam(activeSim, f.enabled ? cfg : null);
        configureFluidSimulationRenderLayer(foamTask, { enabled: foamRenderVisible() });
    }
    // Re-apply after any sim rebuild / backend switch: (re)enable foam on the new active
    // sim (allocates its pool) THEN rebind the renderer to that pool.
    function applyFoam(): void {
        pushFoam();
        configureFluidSimulationRenderLayer(foamTask, {
            foam: {
                surfaceFiltering: controls.getValues().foam.surfaceFiltering ?? methodName === "FLIP",
            },
        });
    }

    // ── Opt-in GPU timing ────────────────────────────────────────────────────
    // The profiler is created lazily the first time the user enables timing; if the
    // headless/host GPU lacks the "timestamp-query" feature we never create it and
    // the UI shows a graceful fallback. When enabled it is wired to BOTH sims and all
    // three render tasks so every fluid pass is tagged with timestampWrites; when off
    // we wire null everywhere so there is zero timing cost.
    const timingSupported = engine._device.features.has("timestamp-query");
    let profiler: FluidSimulationProfiler | null = null;
    let referenceProfiler: FluidSimulationProfiler | null = null;
    let timingEnabled = false;
    function currentGpuProfiler(): FluidSimulationProfiler | null {
        if (!timingSupported || !profiler) {
            return null;
        }
        if (referenceSelected()) {
            referenceProfiler ??= createFluidSimulationProfiler(engine);
            return referenceProfiler;
        }
        return profiler;
    }
    // Push the profiler (or null) to both backends + every render task. Re-called after
    // a sim rebuild / backend switch (the sims are recreated; the tasks persist).
    function applyProfiler(): void {
        const activeProfiler = currentGpuProfiler();
        timingEnabled = activeProfiler !== null;
        canvas.dataset.timing = timingEnabled ? "on" : "off";
        setFluidSimulationProfiler(activeSim, activeProfiler);
        configureFluidSimulationRenderLayer(particleTask, { profiler: activeProfiler });
        configureFluidSimulationRenderLayer(surfaceTask, { profiler: activeProfiler });
        configureFluidSimulationRenderLayer(polygonSurfaceTask, { profiler: activeProfiler });
        configureFluidSimulationRenderLayer(foamTask, { profiler: activeProfiler });
    }
    // Timing is always on when the GPU supports timestamp-query: create + wire the
    // profiler unconditionally at startup (no opt-in checkbox). If the feature is
    // missing (or profiler creation throws) timing stays off and the GPU panel shows
    // a fallback note instead of the per-stage rows.
    if (timingSupported) {
        try {
            profiler = createFluidSimulationProfiler(engine);
            timingEnabled = true;
        } catch (err) {
            console.warn("[fluid] GPU timing unavailable", err);
            profiler = null;
            timingEnabled = false;
        }
        applyProfiler();
    }
    // Resolve task: recorded LAST in the frame (added after every timed task) so its
    // encoder-level resolve/copy runs after all timestampWrites, in the same encoder,
    // just before the engine submits. It draws nothing.
    const timingResolveTask: Task = {
        name: "fluid-timing-resolve",
        engine,
        scene,
        _passes: [],
        record(): void {
            /* nothing to record — encoder-level work happens in execute() */
        },
        execute(): number {
            const activeProfiler = currentGpuProfiler();
            if (timingEnabled && activeProfiler) {
                // Close the whole-frame envelope (frameStop) AFTER every other pass, then
                // resolve. The envelope's end-to-start span is the frame's total GPU time.
                endFluidSimulationProfilerFrame(activeProfiler, { deferResolve: referenceShouldStep() });
            }
            return 0;
        },
        dispose(): void {
            /* profiler lifetime is owned by the demo, not the frame graph */
        },
    };
    addTask(scene, timingResolveTask);

    // Load the HDR environment via babylon-lite. loadEnvironment uploads the
    // linear-HDR specular cube (fed to the fluid surface reflections) and, because
    // skyboxUrl matches the .env URL, pushes an HDR skybox renderable (order 0)
    // that the scene task draws as the background into sceneColorRT. The skybox
    // snapshots scene.imageProcessing at registerScene time, so exposure/contrast
    // are set here (matching the old baked exposure 2.0 / contrast 1.2 look). We
    // await `envReady` before registerScene so the deferred skybox builder and the
    // imageProcessing values are in place before the scene is built.
    // Per-demo HDR environments. box/capsule/fountain/marble-tower reflect the neutral studio
    // env (environment.env); the waterfall — and ONLY the waterfall — gets an open-sky
    // panorama (Poly Haven "quarry_04_puresky"), declared with that demo's own assets as
    // WATERFALL_ENV_URL. The studio map is a pre-filtered `.env` (loadEnvironment); the sky is
    // a raw Radiance `.hdr`, so it goes through loadHdrEnvironment, which parses RGBE and
    // prefilters the GGX cube on the GPU. Both are loaded with NO internal skybox (skipSkybox)
    // — the core instead builds one HDR-skybox renderable per cube (identical output to either
    // loader's own deferred builder: the skybox shader normalizes the cube direction, so
    // size/position/primaryColor are irrelevant) and, on demo switch, swaps which skybox is in
    // scene._renderables (bumping the renderable version so the scene task rebuilds its opaque
    // bundle), which cube feeds the fluid surface reflections, and which exposure the scene is
    // graded at. We await `envReady` before registerScene so the initial (box) skybox is in
    // the scene set and rendered from frame 0.
    //
    // The two maps are loaded INDEPENDENTLY and a failure of either is survivable: the scene
    // colour target is deliberately never cleared (the skybox is guaranteed to cover every
    // pixel), so a missing background is not a missing sky — it is a pure BLACK frame. A
    // rejected Promise.all would take both skyboxes down and blank every demo, so each load
    // has its own catch and the survivor is shared.
    const brdfUrl = demoAssetUrl("./brdf-lut.png", import.meta.url);
    /** One loaded environment: its cube, its background skybox and the grade it is viewed at.
     *  The open-sky HDR is far brighter than the studio .env (bare sun + sky vs. an interior
     *  probe), so they cannot share one exposure. Materials read exposure/contrast from the
     *  scene UBO, which is repacked every frame — so applyDemoEnv just re-points them. The
     *  SKYBOX is the exception: it snapshots imageProcessing into its per-mesh UBO at build
     *  time, hence each one is built under its own grade below. */
    interface EnvSlot {
        env: EnvironmentTextures;
        sky: Renderable;
        exposure: number;
        contrast: number;
    }
    let studioSlot: EnvSlot | null = null;
    let skySlot: EnvSlot | null = null;
    let activeSky: Renderable | null = null;
    let hostSceneSuspended = false;
    let suspendedHostSkyVisible = false;
    let suspendedAmbientIntensity = ambient.intensity;
    let suspendedSunIntensity = sun.intensity;
    let suspendedClearColor = { ...scene.clearColor };

    // ── Environment picker (TEMPORARY: a shortlist to audition backdrops) ────────────────
    // Every demo can be viewed under any of these. The first two are the demos' own defaults
    // and are bundled; the rest are pulled straight from Poly Haven's CDN at runtime (it
    // serves `Access-Control-Allow-Origin: *`), so auditioning them costs the repo nothing and
    // deleting the entries below is the whole removal. `exposure`/`contrast` are the grade the
    // scene is viewed at — the night maps need a much higher exposure than the midday skies,
    // and these are starting points rather than tuned values.
    // NB: built by concatenation, NOT a template literal — the WGSL minifier that runs over
    // emitted chunks treats every backtick template as shader source and strips `//` to
    // end-of-line, which silently truncates a URL at the scheme separator. Quoted strings are
    // skipped by that pass, so they are the safe way to hold a URL here.
    const POLY_HAVEN_2K = (slug: string): string => "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/" + slug + "_2k.hdr";
    interface EnvChoice {
        key: string;
        label: string;
        url: string;
        /** Pre-filtered `.env` (loadEnvironment) vs raw Radiance `.hdr` (loadHdrEnvironment). */
        hdr: boolean;
        exposure: number;
        contrast: number;
    }
    const ENV_CHOICES: EnvChoice[] = [
        { key: "studio", label: "Studio (default)", url: ENV_STUDIO_URL, hdr: false, exposure: 1.0, contrast: 1.1 },
        { key: "industrial", label: "Industrial sunset sky", url: WATERFALL_ENV_URL, hdr: true, exposure: 0.9, contrast: 1.15 },
        { key: "belfast", label: "Belfast sunset sky", url: WATERFALL_BELFAST_ENV_URL, hdr: true, exposure: 0.8, contrast: 1.15 },
        { key: "quarry", label: "Quarry pure sky", url: POLY_HAVEN_2K("quarry_04_puresky"), hdr: true, exposure: 0.55, contrast: 1.15 },
        { key: "driveway", label: "Tree-lined driveway", url: POLY_HAVEN_2K("tree_lined_driveway"), hdr: true, exposure: 1.0, contrast: 1.1 },
        { key: "drackenstein", label: "Drackenstein quarry sky", url: POLY_HAVEN_2K("drackenstein_quarry_puresky"), hdr: true, exposure: 0.6, contrast: 1.15 },
        { key: "dikhololo", label: "Dikhololo night", url: POLY_HAVEN_2K("dikhololo_night"), hdr: true, exposure: 1.2, contrast: 1.1 },
        { key: "rogland", label: "Rogland clear night", url: POLY_HAVEN_2K("rogland_clear_night"), hdr: true, exposure: 1.2, contrast: 1.1 },
        { key: "minedump", label: "Minedump flats", url: POLY_HAVEN_2K("minedump_flats"), hdr: true, exposure: 0.7, contrast: 1.15 },
        { key: "qwantani", label: "Qwantani night sky", url: POLY_HAVEN_2K("qwantani_night_puresky"), hdr: true, exposure: 1.2, contrast: 1.1 },
    ];
    /** The picker entry whose map is loaded EAGERLY into `skySlot` — the waterfall's own default
     *  backdrop. Derived from the URL rather than spelled out, so that pointing WATERFALL_ENV_URL
     *  at a different sky cannot leave the eager slot filed under the PREVIOUS default's key.
     *  Doing exactly that is what made "Belfast sunset sky" silently install the industrial map:
     *  the shortcut in `applyDemoEnv` still answered "belfast" with this slot, so the Belfast HDR
     *  was never fetched at all. */
    const EAGER_SKY = ENV_CHOICES.find((c) => c.url === WATERFALL_ENV_URL);
    const EAGER_SKY_KEY = EAGER_SKY?.key ?? "industrial";
    /** Loaded slots by choice key. `null` marks a load that FAILED, so it is not retried. */
    const envSlots = new Map<string, EnvSlot | null>();
    /** Picker override; null = follow the active demo's own `envUrl`. */
    let envOverride: string | null = null;
    const envKeyFor = (demo: FluidDemo): string => envOverride ?? demo.envKey ?? "studio";

    // Rotation is an opt-in skybox shader feature. Register it before makeSlot builds any
    // visible environment so the later UI updates can rotate existing skyboxes dynamically.
    setEnvironmentRotation(scene, 0);
    const envReady = Promise.all([
        loadEnvironment(scene, ENV_STUDIO_URL, { brdfUrl, skipGround: true, skipSkybox: true }).catch((err: unknown) => {
            console.warn("[fluid] studio env load failed", err);
            return null;
        }),
        // 512² faces (vs the 256 default): the panorama is the waterfall's full-screen
        // backdrop, not just an IBL source, so mip 0 has to hold up at viewport resolution.
        loadHdrEnvironment(scene, WATERFALL_ENV_URL, { faceSize: 512, skipGround: true, skipSkybox: true }).catch((err: unknown) => {
            console.warn(`[fluid] waterfall sky HDR load failed (${WATERFALL_ENV_URL}) — falling back to the studio environment`, err);
            return null;
        }),
    ]).then(async ([studio, sky]) => {
        // The loaders disagree on tone mapping (loadEnvironment enables it, loadHdrEnvironment
        // disables it) and Promise.all gives no ordering guarantee, so pin it explicitly.
        scene.imageProcessing.toneMappingEnabled = true;
        studioSlot = studio ? await makeSlot(studio, 1.0, 1.1) : null;
        // `sky` is the waterfall's own env (WATERFALL_ENV_URL). Take its grade from the matching
        // picker entry rather than repeating the numbers, so the eager slot and the picker can
        // never disagree about how the same map is exposed.
        skySlot = sky ? await makeSlot(sky, EAGER_SKY?.exposure ?? 0.8, EAGER_SKY?.contrast ?? 1.15) : null;
        studioSlot ??= skySlot;
        skySlot ??= studioSlot;
        // Cache AFTER the cross-fallback, so a failed map resolves to its survivor rather than
        // being remembered as "load failed" and leaving that demo with no sky at all.
        envSlots.set("studio", studioSlot);
        envSlots.set(EAGER_SKY_KEY, skySlot);
        if (!skySlot) {
            console.warn("[fluid] no environment loaded — the scene has no skybox and will render black");
            return;
        }
        // ONE global IBL cube: the PBR group builder captures scene._envTextures at
        // registerScene and bakes it into every material's bind group, so it cannot be swapped
        // per demo. The waterfall's rock is the only PBR geometry whose lighting must match
        // its background, so the sky cube wins (as country.env used to).
        scene._envTextures = skySlot.env;
    });

    /** Build a slot for an already-loaded cube. The skybox snapshots scene.imageProcessing at
     *  build time, so the grade must be set FIRST — that is why this is not just an object
     *  literal, and why every slot carries the grade it was built under. */
    async function makeSlot(env: EnvironmentTextures, exposure: number, contrast: number): Promise<EnvSlot> {
        scene.imageProcessing.exposure = exposure;
        scene.imageProcessing.contrast = contrast;
        const sky = await buildHdrSkyboxRenderable(scene, env, 10, [0, 0, 0], [0, 0, 0]);
        return { env, sky, exposure, contrast };
    }

    /** Fetch + prefilter a choice the first time it is picked, then cache it (including a
     *  failure, as null, so a dead URL is not re-fetched on every switch). */
    async function loadEnvChoice(choice: EnvChoice): Promise<EnvSlot | null> {
        if (envSlots.has(choice.key)) {
            return envSlots.get(choice.key) ?? null;
        }
        try {
            const env = choice.hdr
                ? await loadHdrEnvironment(scene, choice.url, { faceSize: 512, skipGround: true, skipSkybox: true })
                : await loadEnvironment(scene, choice.url, { brdfUrl, skipGround: true, skipSkybox: true });
            scene.imageProcessing.toneMappingEnabled = true;
            const slot = await makeSlot(env, choice.exposure, choice.contrast);
            envSlots.set(choice.key, slot);
            return slot;
        } catch (err: unknown) {
            console.warn(`[fluid] environment "${choice.key}" failed to load (${choice.url})`, err);
            envSlots.set(choice.key, null);
            return null;
        }
    }

    // Install the active demo's environment: swap the background skybox renderable in
    // the scene render set, re-grade the scene, and point the fluid surface reflections at
    // the matching cube. No-op until the envs finish loading (the first switchPair runs
    // before `await envReady`; the initial install then happens right after it, below).
    function applyDemoEnv(demo: FluidDemo): void {
        const key = envKeyFor(demo);
        if (envSlots.has(key)) {
            // Already attempted. A cached `null` means the load failed — leave whatever sky is
            // currently up rather than blanking the scene.
            const cached = envSlots.get(key) ?? null;
            if (cached) {
                installEnvSlot(cached);
            }
            return;
        }
        // The two built-ins are reachable through their eager slots before envReady has
        // populated the cache (the very first switchPair runs before it resolves). Only those
        // two: any other key must go through `loadEnvChoice`, or it would be answered with a
        // map that is not its own.
        const builtin = key === "studio" ? studioSlot : key === EAGER_SKY_KEY ? skySlot : null;
        if (builtin) {
            installEnvSlot(builtin);
            return;
        }
        // A picker choice on its first use: fetch it, then install — but only if it is still
        // the current choice by the time it lands.
        const choice = ENV_CHOICES.find((c) => c.key === key);
        if (!choice) {
            return;
        }
        void loadEnvChoice(choice).then((loaded) => {
            if (loaded && envKeyFor(activeDemo ?? demo) === key) {
                installEnvSlot(loaded);
            }
        });
    }

    /** Swap the background skybox renderable in the scene render set, re-grade the scene, and
     *  point the fluid surface reflections at the matching cube. */
    function installEnvSlot(slot: EnvSlot): void {
        scene.imageProcessing.exposure = slot.exposure;
        scene.imageProcessing.contrast = slot.contrast;
        // Diffuse IBL: the scene UBO is repacked every frame from scene._envTextures, so
        // re-pointing it makes ambient lighting follow the picker. The SPECULAR cube is a
        // different story — the PBR group builder bakes it into each material's bind group at
        // registerScene, so already-built materials keep the startup cube's reflections.
        scene._envTextures = slot.env;
        if (activeSky !== slot.sky) {
            if (activeSky) {
                const i = scene._renderables.indexOf(activeSky);
                if (i >= 0) {
                    scene._renderables.splice(i, 1);
                }
            }
            if (!hostSceneSuspended || suspendedHostSkyVisible) {
                scene._renderables.push(slot.sky);
            }
            scene._renderableVersion++;
            activeSky = slot.sky;
        }
        const environment = createFluidRenderEnvironment(slot.env);
        configureFluidSimulationRenderLayer(surfaceTask, { environment });
        configureFluidSimulationRenderLayer(polygonSurfaceTask, { environment });
    }

    function setHostSkyVisible(visible: boolean): void {
        if (!activeSky) {
            return;
        }
        const index = scene._renderables.indexOf(activeSky);
        if (visible ? index < 0 : index >= 0) {
            if (visible) {
                scene._renderables.push(activeSky);
            } else {
                scene._renderables.splice(index, 1);
            }
            scene._renderableVersion++;
        }
    }

    // On-screen FPS accumulators (smoothed over ~0.5 s windows). The FPS read-out element
    // itself lives in the shared controls panel's top-left GPU section (controls.gpu).
    let fpsAccumMs = 0;
    let fpsFrames = 0;

    // Mouse-force state. The Shift+RMB drag sets `pendingForce`; the frame loop
    // retains the latest sample briefly so a deferred solver step cannot lose it.
    let pendingForce: PendingForce | null = null;

    // Shift+RMB "push the fluid" drag state (core-owned; works in every demo).
    // While this is active the camera is disengaged — its pointerdown gate rejected
    // the gesture (isForceGesture) — so only the force applies, no camera pan.
    let forceDragging = false;
    let forceLastX = 0;
    let forceLastY = 0;
    let forceLastT = 0;

    const effectiveGridSettings = (method = methodName, scale = domainScale): FluidGridSettings => gridSettings ?? defaultGridSettings(method, scale);
    function mlsContainerBounds(): { min: [number, number, number]; max: [number, number, number] } {
        const grid = effectiveGridSettings();
        const dx = gridSettings ? fluidSimulationCellSize("MLS-MPM", fluidDiscretization(physicsScale)) : cellSizeForPhysicsScale("MLS-MPM", physicsScale);
        const dims = gridCellsForSize(grid.size, dx);
        const minimum = gridBounds(grid.position, grid.size).min;
        const lo: [number, number, number] = [minimum[0] + dx * 2.5, minimum[1] + dx * 2.5, minimum[2] + dx * 2.5];
        const hi: [number, number, number] = [minimum[0] + (dims[0] - 3.5) * dx, minimum[1] + (dims[1] - 3.5) * dx, minimum[2] + (dims[2] - 3.5) * dx];
        return { min: lo, max: hi };
    }
    function writeWhiteboardMlsContainer(buffer: GPUBuffer, byteOffset: number): void {
        const { min: lo, max: hi } = mlsContainerBounds();
        engine._device.queue.writeBuffer(buffer, byteOffset, new Float32Array([lo[0], lo[1], lo[2], 0, hi[0], hi[1], hi[2], 0]));
        canvas.dataset.mlsContainerLo = lo.join(",");
        canvas.dataset.mlsContainerHi = hi.join(",");
    }
    const gridLocalToWorld = (local: readonly [number, number, number], position: readonly [number, number, number]): [number, number, number] => [
        local[0] + position[0],
        local[1] + position[1],
        local[2] + position[2],
    ];
    const worldToGridLocal = (world: readonly [number, number, number], position: readonly [number, number, number]): [number, number, number] => [
        world[0] - position[0],
        world[1] - position[1],
        world[2] - position[2],
    ];
    const flowToWorld = (flow: FluidFlowConfig, position = effectiveGridSettings().position): FluidFlowConfig => transformFluidFlow(flow, { translation: position });
    const flowToGridLocal = (flow: FluidFlowConfig, position = effectiveGridSettings().position): FluidFlowConfig =>
        transformFluidFlow(flow, { translation: [-position[0], -position[1], -position[2]] });
    const flipInitialStatePlans = createFluidInitialStatePlanCache();
    const calculateFlipParticlePlan = (
        requested: number,
        resolution: number,
        markersPerCell: number,
        flow: FluidFlowConfig,
        grid: FluidGridSettings
    ): { active: number; total: number; required: number; initialStateKey: string; emitterCounts: ReadonlyMap<string, number> } => {
        const total = Math.max(1, Math.min(Number.MAX_SAFE_INTEGER, Math.round(requested)));
        const particleVolume = flipDiscretizationForGrid(grid, resolution, markersPerCell).markerVolume;
        const initialState = flipInitialStatePlans.resolve({
            particleCapacity: total,
            particleVolume,
            flow: flowToWorld(flow, grid.position),
            bounds: gridBounds(grid.position, grid.size),
            deriveInitialCount: true,
        });
        return {
            active: initialState.activeCount,
            total,
            required: initialState.requiredCount,
            initialStateKey: initialState.initialStateKey,
            emitterCounts: initialState.emitterCounts,
        };
    };
    const flipParticlePlan = (
        requested: number,
        flow = activeFlow,
        resolution = flipGridResolution
    ): { active: number; total: number; required: number; initialStateKey: string; emitterCounts: ReadonlyMap<string, number> } => {
        if (methodName !== "FLIP") {
            const requestedCapacity = Math.max(1, Math.min(Number.MAX_SAFE_INTEGER, Math.round(requested)));
            return {
                active: requestedCapacity,
                total: requestedCapacity,
                required: requestedCapacity,
                initialStateKey: `legacy:${requestedCapacity}`,
                emitterCounts: new Map(),
            };
        }
        return calculateFlipParticlePlan(requested, resolution, flipMarkersPerCell, flow, effectiveGridSettings("FLIP"));
    };
    function flipInitialEmitterParticleCounts(): Map<string, number> {
        const counts = new Map(activeFlow.emitters.map((emitter) => [emitter.id, 0]));
        if (methodName !== "FLIP") {
            return counts;
        }
        const resetCounts = activeSim.initialEmitterParticleCounts;
        const pendingInitialPlan =
            !gridSettingsEqual(gridSettings, builtGridSettings) ||
            builtGridMethod !== "FLIP" ||
            builtFlipGridResolution !== flipGridResolution ||
            builtDomainScale !== domainScale ||
            builtFlipMarkersPerCell !== flipMarkersPerCell ||
            activeSim.count !== flipParticleCapacityRequest;
        if (resetCounts && pendingInitialPlan) {
            for (const emitter of activeFlow.emitters) {
                counts.set(emitter.id, resetCounts.get(emitter.id) ?? 0);
            }
            return counts;
        }
        const plan = flipParticlePlan(flipParticleCapacityRequest);
        if (resetCounts && plan.initialStateKey === builtInitialStateKey) {
            for (const emitter of activeFlow.emitters) {
                counts.set(emitter.id, resetCounts.get(emitter.id) ?? 0);
            }
            return counts;
        }
        for (const [id, count] of plan.emitterCounts) {
            counts.set(id, count);
        }
        return counts;
    }
    function refreshInitialEmitterParticleCount(): void {
        flowEditor?.refreshComputedValues();
    }
    const flipParticleCapacity = (requested: number, _flow = activeFlow, _resolution = flipGridResolution): number =>
        Math.max(1, Math.min(Number.MAX_SAFE_INTEGER, Math.round(requested)));
    const requestedParticleCount = (): number => (methodName === "FLIP" ? flipParticleCapacityRequest : particleCount);
    function fitFlipResolutionToDevice(): { requestedResolution: number; fittedResolution: number } {
        const grid = effectiveGridSettings("FLIP");
        const requestedResolution = flipGridResolution;
        const deviceParticleCapacity = deviceParticleCapacityForMethod("FLIP");
        if (flipParticleCapacityRequest > deviceParticleCapacity) {
            throw new RangeError(
                "Particle capacity is " +
                    flipParticleCapacityRequest.toLocaleString() +
                    "; this WebGPU device supports at most " +
                    deviceParticleCapacity.toLocaleString() +
                    " FLIP particles."
            );
        }
        const fits = (resolution: number): boolean => {
            const plan = flipParticlePlan(flipParticleCapacityRequest, activeFlow, resolution);
            return plan.required <= plan.total && gridAllocationError(grid, "FLIP", physicsScale, undefined, resolution) === undefined;
        };
        const fit = fitFluidGridResolution(requestedResolution, GRID_RESOLUTION_MIN, fits);
        if (!fit) {
            const minimumPlan = flipParticlePlan(flipParticleCapacityRequest, activeFlow, GRID_RESOLUTION_MIN);
            const gridError = gridAllocationError(grid, "FLIP", physicsScale, undefined, GRID_RESOLUTION_MIN);
            throw new RangeError(
                gridError ??
                    "The minimum Resolution divisions value still requires " +
                        minimumPlan.required.toLocaleString() +
                        " initial particles; Particle capacity is " +
                        minimumPlan.total.toLocaleString() +
                        " particles."
            );
        }
        return {
            requestedResolution,
            fittedResolution: fit.fittedResolution,
        };
    }
    function flipPendingCapacityStatus(): string {
        if (methodName !== "FLIP") {
            return "";
        }
        const grid = effectiveGridSettings("FLIP");
        const deviceParticleCapacity = deviceParticleCapacityForMethod("FLIP");
        if (flipParticleCapacityRequest > deviceParticleCapacity) {
            return (
                "Particle capacity is " +
                flipParticleCapacityRequest.toLocaleString() +
                "; this WebGPU device supports at most " +
                deviceParticleCapacity.toLocaleString() +
                " FLIP particles."
            );
        }
        const gridError = gridAllocationError(grid, "FLIP", physicsScale);
        if (gridError) {
            return `${gridError}\u00a0Reset simulation will fit Resolution divisions to the available device limits.`;
        }
        return "";
    }
    function refreshFoamParticleCounts(): void {
        const enabled = controls.getValues().foam.enabled;
        const diffuse = enabled ? getFluidSimulationDiagnostics(activeSim).diffuse : null;
        controls.setFoamParticleCounts(diffuse ?? undefined, enabled, diffuse?.capacity);
        canvas.dataset.diffuseParticlesEnabled = String(enabled);
        canvas.dataset.diffuseParticleCapacity = String(diffuse?.capacity ?? 0);
        if (!diffuse) {
            delete canvas.dataset.diffuseParticleCount;
            delete canvas.dataset.sprayParticleCount;
            delete canvas.dataset.foamParticleCount;
            delete canvas.dataset.bubbleParticleCount;
            return;
        }
        canvas.dataset.diffuseParticleCount = String(diffuse.total);
        canvas.dataset.sprayParticleCount = String(diffuse.spray);
        canvas.dataset.foamParticleCount = String(diffuse.foam);
        canvas.dataset.bubbleParticleCount = String(diffuse.bubble);
    }
    function refreshParticleUsageStatus(activeCount = activeSim.activeCount ?? activeSim.count): void {
        if (controlsBinding) {
            syncFluidControls(controlsBinding);
        }
        refreshInitialEmitterParticleCount();
        refreshFoamParticleCounts();
        const pressureDiagnostics = methodName === "FLIP" ? activeSim.pressureDiagnostics : undefined;
        controls.setPressureDiagnostics(pressureDiagnostics);
        if (pressureDiagnostics) {
            canvas.dataset.pressureRelativeResidual = String(pressureDiagnostics.relativeResidual);
            canvas.dataset.pressureMaxResidual = String(pressureDiagnostics.maxResidual);
            canvas.dataset.postProjectionDivergence = String(pressureDiagnostics.maxDivergence);
            canvas.dataset.pressureFluidCellCount = String(pressureDiagnostics.fluidCellCount);
            canvas.dataset.pressureIterationsUsed = String(pressureDiagnostics.pressureIterations);
        } else {
            delete canvas.dataset.pressureRelativeResidual;
            delete canvas.dataset.pressureMaxResidual;
            delete canvas.dataset.postProjectionDivergence;
            delete canvas.dataset.pressureFluidCellCount;
            delete canvas.dataset.pressureIterationsUsed;
        }
        canvas.dataset.simulationGpuBytes = String(activeSim.gpuBytes);
        const polygonSurface = methodName === "FLIP" ? getFluidSimulationDiagnostics(activeSim).polygon : null;
        controls.setPolygonTriangleCount(polygonSurface?.triangleCount, polygonSurface !== null);
        if (polygonSurface) {
            canvas.dataset.polygonReconstructionMultiplier = String(polygonSurface.reconstructionMultiplier);
        } else {
            delete canvas.dataset.polygonReconstructionMultiplier;
        }
        if (polygonSurface?.triangleCount === undefined) {
            delete canvas.dataset.polygonTriangleCount;
        } else {
            canvas.dataset.polygonTriangleCount = String(polygonSurface.triangleCount);
        }
        if (methodName !== "FLIP") {
            controls.setParticleUsage(activeCount, activeSim.count, activeSim.gpuBytes);
            delete canvas.dataset.restartParticleCount;
            delete canvas.dataset.restartParticleCapacity;
            delete canvas.dataset.restartParticleRequired;
            delete canvas.dataset.restartSimulationGpuBytes;
            return;
        }
        if (controlsBinding?.plan.restartRequired) {
            const memory = controlsBinding.memory;
            controls.setParticleUsage(
                activeCount,
                activeSim.count,
                activeSim.gpuBytes,
                memory.restartActiveCount,
                memory.restartParticleCount,
                memory.steadyBytes,
                memory.restartActiveCountEstimated
            );
            canvas.dataset.gridRestartPending = "true";
            canvas.dataset.restartParticleCount = String(memory.restartActiveCount);
            canvas.dataset.restartParticleCapacity = String(memory.restartParticleCount);
            canvas.dataset.restartParticleRequired = String(memory.restartActiveCount);
            canvas.dataset.restartSimulationGpuBytes = String(memory.steadyBytes);
            return;
        }
        const plan = flipParticlePlan(flipParticleCapacityRequest);
        const effectiveGrid = effectiveGridSettings();
        const gridDim = flipDiscretizationForGrid(effectiveGrid, flipGridResolution).gridDim;
        const restartGpuBytes =
            controlsBinding?.memory.steadyBytes ??
            resolveFluidAllocationPlan({
                method: "FLIP",
                particleCount: plan.total,
                gridDim,
                flipWarmup: {
                    initialLiveCount: plan.active,
                    initialTargetCount: plan.active,
                },
                pressureSolver: (controls.getPhysicsValues("FLIP").pressureSolver ?? 0) >= 0.5 ? "multigrid" : "jacobi",
                quality: {
                    pagedGrid: flipPagedGrid,
                    pagedGridMaxPages: flipPagedGridMaxPages,
                    pressureDiagnostics: (controls.getPhysicsValues("FLIP").pressureDiagnostics ?? 0) >= 0.5 || (controls.getPhysicsValues("FLIP").pressureTolerance ?? 0) > 0,
                    liquidSdf: (controls.getPhysicsValues("FLIP").liquidSdf ?? 0) >= 0.5,
                    fractionalSolids: (controls.getPhysicsValues("FLIP").fractionalSolids ?? 0) >= 0.5,
                    reseedParticles: (controls.getPhysicsValues("FLIP").reseedParticles ?? 0) >= 0.5,
                    particleSheeting: (controls.getPhysicsValues("FLIP").particleSheeting ?? 0) >= 0.5,
                    polygonSurface: (controls.getPhysicsValues("FLIP").polygonSurface ?? 0) >= 0.5,
                    polygonReconstructionMultiplier: controls.getPhysicsValues("FLIP").polygonReconstructionMultiplier ?? 1,
                },
                foam: {
                    enabled: controls.getValues().foam.enabled,
                    activeParticles: true,
                    poolScale: controls.getValues().foam.poolScale,
                },
                limits: deviceLimits,
            }).steadyBytes;
        const gridRestartPending =
            controlsBinding?.plan.restartRequired === true ||
            flipGridResolution !== builtFlipGridResolution ||
            !gridSettingsEqual(gridSettings, builtGridSettings) ||
            builtGridMethod !== methodName ||
            flipMarkersPerCell !== builtFlipMarkersPerCell;
        const restartPreviewPending =
            gridRestartPending || plan.initialStateKey !== builtInitialStateKey || plan.total !== activeSim.count || restartGpuBytes !== activeSim.gpuBytes;
        controls.setParticleUsage(
            activeCount,
            activeSim.count,
            activeSim.gpuBytes,
            restartPreviewPending ? plan.active : undefined,
            restartPreviewPending ? plan.total : undefined,
            restartPreviewPending ? restartGpuBytes : undefined
        );
        canvas.dataset.gridRestartPending = String(gridRestartPending);
        canvas.dataset.restartParticleCount = String(plan.active);
        canvas.dataset.restartParticleCapacity = String(plan.total);
        canvas.dataset.restartParticleRequired = String(plan.required);
        canvas.dataset.restartSimulationGpuBytes = String(restartGpuBytes);
    }
    function setInstalledFlow(): void {
        // Only the active backend exists, so flow installs land on it directly. A method
        // switch always rebuilds the target backend (see rebuildSims) and re-applies the
        // installed flow through applyMethod, so no dormant backend can drift out of sync.
        setFluidSimulationFlow(activeSim, installedFlow);
    }
    function applyFlow(): void {
        if (methodName === "FLIP") {
            const capacity = flipParticleCapacity(flipParticleCapacityRequest);
            if (capacity !== activeSim.count) {
                rebuildSims(flipParticleCapacityRequest, physicsScale);
                return;
            }
        }
        installedFlow = flowToWorld(activeFlow);
        setInstalledFlow();
        if (!importedScene) {
            activeDemo?.onFlowChanged?.(installedFlow);
        }
        canvas.dataset.emitterCount = String(activeFlow.emitters.length);
        canvas.dataset.sinkCount = String(activeFlow.sinks.length);
    }
    let lastMarkerDensityWarning = "";
    function updateFlipMarkerDensityWarning(activeCount: number): void {
        let message = "";
        if (methodName === "FLIP" && !installedFlow.emitters.some((emitter) => emitter.enabled && emitter.behavior === "inflow")) {
            const discretization = flipDiscretizationForGrid(effectiveGridSettings("FLIP"), flipGridResolution);
            const authoredVolume = fluidInitialEmitterVolume(installedFlow);
            const markersPerCell = authoredVolume > 0 ? (activeCount * discretization.dx ** 3) / authoredVolume : 0;
            if (markersPerCell > FLIP_HIGH_MARKERS_PER_CELL) {
                message =
                    "High FLIP marker density:\u00a0about\u00a0" +
                    markersPerCell.toFixed(1) +
                    "\u00a0markers per authored MAC cell.\u00a0Above\u00a0" +
                    FLIP_HIGH_MARKERS_PER_CELL +
                    ",\u00a0extra markers mostly add GPU cost and Beer-Lambert darkness rather than simulation detail. Reduce Markers per cell.";
            }
        }
        if (message !== lastMarkerDensityWarning) {
            lastMarkerDensityWarning = message;
            controls.setMarkerDensityWarning(message);
            canvas.dataset.flipMarkerDensityWarning = message ? "true" : "false";
        }
    }
    function updateInstalledFlowObject(kind: "emitter" | "sink", object: FluidEmitter | FluidSink): void {
        const position = effectiveGridSettings().position;
        if (kind === "emitter") {
            const index = installedFlow.emitters.findIndex((candidate) => candidate.id === object.id);
            if (index < 0) {
                return;
            }
            installedFlow.emitters[index] = transformFluidFlow({ emitters: [object as FluidEmitter], sinks: [] }, { translation: position }).emitters[0]!;
        } else {
            const index = installedFlow.sinks.findIndex((candidate) => candidate.id === object.id);
            if (index < 0) {
                return;
            }
            installedFlow.sinks[index] = transformFluidFlow({ emitters: [], sinks: [object as FluidSink] }, { translation: position }).sinks[0]!;
        }
        setInstalledFlow();
        if (!importedScene) {
            activeDemo?.onFlowChanged?.(installedFlow);
        }
    }
    function syncImportedMeshAnimations(rewind: boolean): void {
        const imported = importedScene;
        if (!imported) {
            return;
        }
        if (rewind) {
            for (const group of imported.asset.animationGroups ?? []) {
                goToFrame(group, 0, engine);
                playAnimation(group);
            }
        }
        updateImportedSceneBindings(imported, 0, true);
    }

    function resetActiveFlow(clearHoles: boolean, preserveSceneAnimations = false): void {
        if (stagingBackendSwitch || deferReferenceMutation(() => resetActiveFlow(clearHoles, preserveSceneAnimations))) {
            return;
        }
        let resolutionAdjustment = "";
        if (methodName === "FLIP") {
            const fit = fitFlipResolutionToDevice();
            if (fit.fittedResolution < fit.requestedResolution) {
                flipGridResolution = fit.fittedResolution;
                controls.setGridResolution(fit.fittedResolution);
                resolutionAdjustment =
                    "Resolution divisions adjusted:\u00a0" +
                    fit.requestedResolution.toLocaleString() +
                    "\u00a0→\u00a0" +
                    fit.fittedResolution.toLocaleString() +
                    " to fit this WebGPU device.";
                syncGridControls();
                refreshParticleUsageStatus();
            }
        }
        const pendingBackendRebuild =
            controlsBinding?.plan.restartRequired === true ||
            (methodName === "FLIP" && importedInitialState !== builtImportedInitialState) ||
            !gridSettingsEqual(gridSettings, builtGridSettings) ||
            builtGridMethod !== methodName ||
            activeBackendId() !== (referenceSelected() ? flipBackendId : undefined) ||
            (methodName === "FLIP" &&
                (flipParticleCapacity(flipParticleCapacityRequest) !== activeSim.count ||
                    flipGridResolution !== builtFlipGridResolution ||
                    flipMarkersPerCell !== builtFlipMarkersPerCell));
        if (pendingBackendRebuild) {
            syncImportedMeshAnimations(!preserveSceneAnimations);
            rebuildSims(requestedParticleCount(), physicsScale);
            if (resolutionAdjustment) {
                controls.setGridStatus(resolutionAdjustment);
            } else {
                controls.setGridStatus("");
            }
            if (clearHoles) {
                clearSceneHoles();
            }
            completeReferenceReset();
            if (!preserveSceneAnimations && !importedScene) {
                restartOrStabilizeActiveDemo();
            }
            return;
        }
        applyFlow();
        syncImportedMeshAnimations(!preserveSceneAnimations);
        resetFluidSimulation(activeSim);
        builtInitialStateKey = flipParticlePlan(requestedParticleCount()).initialStateKey;
        restartSimulationLifecycle();
        if (clearHoles) {
            clearSceneHoles();
        }
        completeReferenceReset();
        if (!preserveSceneAnimations && !importedScene) {
            restartOrStabilizeActiveDemo();
        }
    }

    function suspendHostScenePresentation(): void {
        if (hostSceneSuspended) {
            return;
        }
        suspendedHostSkyVisible = activeDemo?.key === "whiteboard";
        activeDemo?.onLeave();
        setMeshVisible(ground, false);
        suspendedAmbientIntensity = ambient.intensity;
        suspendedSunIntensity = sun.intensity;
        suspendedClearColor = { ...scene.clearColor };
        ambient.intensity = 0;
        sun.intensity = 0;
        ambient._bumpLightVersion?.();
        sun._bumpLightVersion?.();
        setHostSkyVisible(suspendedHostSkyVisible);
        sceneTask._config.clr = !suspendedHostSkyVisible;
        if (msaaSceneTask) {
            msaaSceneTask._config.clr = !suspendedHostSkyVisible;
        }
        hostSceneSuspended = true;
        canvas.dataset.importedHostSceneCleared = "true";
        canvas.dataset.importedHostLightsDisabled = "true";
        canvas.dataset.importedHostSkyHidden = String(!suspendedHostSkyVisible);
    }

    function restoreHostScenePresentation(restoreDemo: boolean): void {
        if (!hostSceneSuspended) {
            return;
        }
        hostSceneSuspended = false;
        suspendedHostSkyVisible = false;
        ambient.intensity = suspendedAmbientIntensity;
        sun.intensity = suspendedSunIntensity;
        scene.clearColor = suspendedClearColor;
        setHostSkyVisible(true);
        sceneTask._config.clr = false;
        if (msaaSceneTask) {
            msaaSceneTask._config.clr = false;
        }
        if (restoreDemo) {
            activeDemo?.onEnter();
        }
        ambient._bumpLightVersion?.();
        sun._bumpLightVersion?.();
        delete canvas.dataset.importedHostSceneCleared;
        delete canvas.dataset.importedHostLightsDisabled;
        delete canvas.dataset.importedHostSkyHidden;
    }

    function currentSceneSdf(): FluidSceneSdf {
        if (!activeDemo) {
            throw new Error("[fluid] cannot resolve the scene SDF before the initial demo is active.");
        }
        const demo = activeDemo;
        if (importedScene) {
            syncImportedSceneGridTransform(importedScene);
            if (demo.key === "whiteboard" && methodName === "MLS-MPM") {
                const bounds = mlsContainerBounds();
                updateFluidSceneSdfContainer(importedScene.sceneSdf, bounds);
                canvas.dataset.mlsContainerLo = bounds.min.join(",");
                canvas.dataset.mlsContainerHi = bounds.max.join(",");
            } else {
                updateFluidSceneSdfContainer(importedScene.sceneSdf, null);
                delete canvas.dataset.mlsContainerLo;
                delete canvas.dataset.mlsContainerHi;
            }
            return importedScene.sceneSdf;
        } else {
            demo.writeSdfParams();
        }
        const demoMlsContainer = methodName === "MLS-MPM" ? mlsContainerBounds() : null;
        demo.setMlsContainer?.(demoMlsContainer);
        if (demo.setMlsContainer) {
            if (demoMlsContainer) {
                canvas.dataset.mlsContainerLo = demoMlsContainer.min.join(",");
                canvas.dataset.mlsContainerHi = demoMlsContainer.max.join(",");
            } else {
                delete canvas.dataset.mlsContainerLo;
                delete canvas.dataset.mlsContainerHi;
            }
        }
        const liveSceneSdf = demo.sceneSdf?.();
        if (liveSceneSdf) {
            return liveSceneSdf;
        }
        const sdf = demo.sdf;
        let mlsSdf = sdf;
        if (demo.key === "whiteboard") {
            writeWhiteboardMlsContainer(whiteboardMlsContainerBuffer, 0);
            mlsSdf = whiteboardMlsContainerSdf;
        } else {
            delete canvas.dataset.mlsContainerLo;
            delete canvas.dataset.mlsContainerHi;
        }
        // MLS-MPM consumes the whiteboard container SDF; every other backend uses the
        // demo/imported SDF. Only the active backend is resident, so pick its variant.
        return adoptFluidSceneSdf(engine, methodName === "MLS-MPM" ? mlsSdf : sdf);
    }
    function applySceneSdf(): void {
        if (stagingBackendSwitch) {
            return;
        }
        setFluidSimulationSceneSdf(activeSim, currentSceneSdf());
        applyFlow();
    }

    function clearImportedScene(restoreDemo: boolean): void {
        const previous = importedScene;
        if (!previous) {
            return;
        }
        importedScene = null;
        importedInitialState = null;
        refreshImportedCollisionControls(null);
        if (externalSceneExportRow) {
            externalSceneExportRow.style.display = "none";
        }
        importedCollisionActive = false;
        canvas.dataset.importedBundle = "false";
        canvas.dataset.importedMeshCount = "0";
        delete canvas.dataset.importedCollisionDims;
        delete canvas.dataset.importedCollisionOrigin;
        delete canvas.dataset.importedCollisionCellSize;
        delete canvas.dataset.importedSceneOffset;
        delete canvas.dataset.importedAnimationCount;
        delete canvas.dataset.importedAnimationTime;
        delete canvas.dataset.importedPlayingAnimationCount;
        delete canvas.dataset.importedAnimatedCollisionCount;
        delete canvas.dataset.importedAnimatedCollisionPosition;
        delete canvas.dataset.importedBoundEmitterCount;
        delete canvas.dataset.importedEmitterPosition;
        delete canvas.dataset.importedEmitterSourceVelocity;
        delete canvas.dataset.importedMissingEmitterSourceCount;
        delete canvas.dataset.importedHostGroundHidden;
        delete canvas.dataset.importedLightCount;
        delete canvas.dataset.importedFrameClearing;
        delete canvas.dataset.importedCameraFramed;
        delete canvas.dataset.importedCameraAlpha;
        delete canvas.dataset.importedCameraRadius;
        delete canvas.dataset.importedCameraTarget;
        delete canvas.dataset.importedInitialStateCount;
        delete canvas.dataset.importedInitialStateFrame;
        setMeshVisible(ground, previous.groundWasVisible);
        setFluidSimulationSceneSdf(activeSim, null);
        for (const mesh of previous.collisionDebugMeshes.values()) {
            removeFromScene(scene, mesh);
        }
        removeFromScene(scene, previous.asset);
        disposeFluidSceneSdf(previous.sceneSdf);
        restoreHostScenePresentation(restoreDemo);
        if (restoreDemo && activeDemo) {
            const usesGridFloor = activeDemo.useGridFloor === true;
            if (builtWithGridFloor !== usesGridFloor) {
                rebuildSims(requestedParticleCount(), physicsScale);
            } else {
                applySceneSdf();
            }
        }
    }

    function createImportedCollision(bundle: BlenderFluidScene): ImportedCollisionResources {
        return {
            sceneSdf: createFluidCompositeSceneSdf(engine, {
                staticSdf: {
                    ...bundle.collision,
                    enabled: bundle.collisionEnabled,
                    trilinear: bundle.collisionTrilinear,
                },
                localSdfs: bundle.animatedCollisions.map((entry) => ({
                    id: entry.id,
                    ...entry.collision,
                    enabled: entry.enabled,
                    trilinear: entry.trilinear,
                })),
            }),
            collisionOrigin: [...bundle.collision.origin],
        };
    }

    function refreshImportedCollisionControls(imported: ImportedFluidScene | null): void {
        if (!collisionControlsHost) {
            return;
        }
        collisionControlsHost.replaceChildren();
        controls.setSectionVisible("Collision", imported !== null);
        if (!imported) {
            return;
        }
        const payload = imported.bundle.preset.scene;
        if (!payload) {
            return;
        }
        const addGrid = (
            title: string,
            resolution: number,
            dims: readonly [number, number, number],
            settings: { enabled: boolean; trilinear: boolean },
            debugId: string,
            apply: (settings: { enabled: boolean; trilinear: boolean }) => void
        ): void => {
            const row = document.createElement("div");
            row.style.cssText = "padding:7px 0;border-bottom:1px solid #253247;";
            const heading = document.createElement("div");
            heading.style.cssText = "display:flex;justify-content:space-between;gap:8px;margin-bottom:5px;";
            const name = document.createElement("strong");
            name.textContent = title;
            const info = document.createElement("span");
            info.style.cssText = "color:#8fa4ba;font-size:11px;";
            info.textContent = `${resolution} (${dims.join(" × ")})`;
            const show = document.createElement("button");
            show.textContent = "Show";
            show.style.cssText = "padding:2px 8px;cursor:pointer;background:#26415f;color:#eef3f8;border:1px solid #3a567a;border-radius:4px;";
            show.onclick = () => {
                const mesh = importedCollisionDebugMesh(imported, debugId);
                const visible = mesh.visible === false;
                setMeshVisible(mesh, visible);
                show.textContent = visible ? "Hide" : "Show";
            };
            heading.append(name, info, show);
            const toggles = document.createElement("div");
            toggles.style.cssText = "display:flex;gap:14px;flex-wrap:wrap;";
            const checkbox = (label: string, checked: boolean, onchange: (checked: boolean) => void): HTMLLabelElement => {
                const wrapper = document.createElement("label");
                wrapper.style.cssText = "display:flex;align-items:center;gap:5px;cursor:pointer;";
                const input = document.createElement("input");
                input.type = "checkbox";
                input.checked = checked;
                input.onchange = () => onchange(input.checked);
                wrapper.append(input, document.createTextNode(label));
                return wrapper;
            };
            toggles.append(
                checkbox("Enabled", settings.enabled, (enabled) => {
                    settings.enabled = enabled;
                    apply(settings);
                }),
                checkbox("Trilinear filtering", settings.trilinear, (trilinear) => {
                    settings.trilinear = trilinear;
                    apply(settings);
                })
            );
            const position = document.createElement("div");
            position.style.cssText = "display:grid;grid-template-columns:auto repeat(3,minmax(0,1fr));gap:6px;align-items:center;margin-top:7px;";
            const positionLabel = document.createElement("span");
            positionLabel.textContent = "Position";
            positionLabel.style.cssText = "color:#9fb4cc;font-size:11px;";
            position.appendChild(positionLabel);
            const currentOffset = importedCollisionOffset(imported, debugId);
            for (let axis = 0; axis < 3; axis++) {
                const field = document.createElement("label");
                field.style.cssText = "display:flex;align-items:center;gap:3px;min-width:0;";
                const axisLabel = document.createElement("span");
                axisLabel.textContent = "XYZ"[axis]!;
                axisLabel.style.cssText = "color:#7c8aa0;font-size:10px;";
                const input = document.createElement("input");
                input.type = "number";
                input.step = "0.01";
                input.value = String(currentOffset[axis]);
                input.style.cssText = "width:100%;min-width:0;box-sizing:border-box;";
                input.onchange = () => {
                    const value = Number.parseFloat(input.value);
                    if (!Number.isFinite(value)) {
                        input.value = String(importedCollisionOffset(imported, debugId)[axis]);
                        return;
                    }
                    const next = [...importedCollisionOffset(imported, debugId)] as [number, number, number];
                    next[axis] = value;
                    imported.collisionOffsets.set(debugId, next);
                    if (debugId === STATIC_COLLISION_DEBUG_ID) {
                        syncImportedSceneGridTransform(imported);
                    } else {
                        updateImportedCollisionTransforms(imported, 0, true);
                    }
                };
                field.append(axisLabel, input);
                position.appendChild(field);
            }
            row.append(heading, toggles, position);
            collisionControlsHost!.appendChild(row);
        };

        const staticSettings = {
            enabled: payload.collisionEnabled ?? true,
            trilinear: payload.collisionTrilinear ?? true,
        };
        addGrid("Static scene", Math.max(...imported.bundle.collision.dims), imported.bundle.collision.dims, staticSettings, STATIC_COLLISION_DEBUG_ID, (settings) => {
            payload.collisionEnabled = settings.enabled;
            payload.collisionTrilinear = settings.trilinear;
            imported.bundle.collisionEnabled = settings.enabled;
            imported.bundle.collisionTrilinear = settings.trilinear;
            updateFluidSceneSdfGridSettings(imported.sceneSdf, [{ enabled: settings.enabled, trilinear: settings.trilinear }]);
        });
        for (const collision of imported.bundle.animatedCollisions) {
            const manifest = payload.animatedCollisions?.find((entry) => entry.id === collision.id);
            const settings = { enabled: collision.enabled, trilinear: collision.trilinear };
            addGrid(collision.node, collision.resolution, collision.collision.dims, settings, collision.id, (next) => {
                collision.enabled = next.enabled;
                collision.trilinear = next.trilinear;
                if (manifest) {
                    manifest.enabled = next.enabled;
                    manifest.trilinear = next.trilinear;
                }
                updateFluidSceneSdfGridSettings(imported.sceneSdf, [{ id: collision.id, enabled: next.enabled, trilinear: next.trilinear }]);
            });
        }
    }

    function importedSceneNodes(asset: AssetContainer): Map<string, SceneNode[]> {
        const nodes = new Map<string, SceneNode[]>();
        const seen = new Set<SceneNode>();
        const visit = (node: SceneNode): void => {
            if (seen.has(node)) {
                return;
            }
            seen.add(node);
            if (!("_gpu" in node)) {
                const named = nodes.get(node.name);
                if (named) {
                    named.push(node);
                } else {
                    nodes.set(node.name, [node]);
                }
            }
            for (const child of node.children) {
                visit(child);
            }
        };
        for (const entity of asset.entities) {
            if (!("lightType" in (entity as object))) {
                visit(entity as SceneNode);
            }
        }
        return nodes;
    }

    function setImportedNodePresentation(node: SceneNode, visible: boolean): void {
        for (const child of node.children) {
            if ("_gpu" in child) {
                setMeshVisible(child as Mesh, visible);
            }
            setImportedNodePresentation(child, visible);
        }
    }

    function createImportedAnimatedCollisionBindings(asset: AssetContainer, bundle: BlenderFluidScene): ImportedAnimatedCollisionBinding[] {
        const nodes = importedSceneNodes(asset);
        return bundle.animatedCollisions.map((entry) => {
            const matches = nodes.get(entry.node);
            if (!matches || matches.length !== 1) {
                throw new Error(`Animated fluid collision "${entry.id}" must resolve to exactly one GLB node named "${entry.node}".`);
            }
            const node = matches[0]!;
            if (!entry.presentation) {
                setImportedNodePresentation(node, false);
            }
            return { id: entry.id, node };
        });
    }

    function useImportedFluidBundleBasis(asset: AssetContainer): SceneNode {
        const root = asset.entities[0];
        if (!root || "lightType" in root || root.name !== "__root__") {
            throw new Error("Imported fluid GLB has no transform root.");
        }
        root.scaling.x = Math.abs(root.scaling.x);
        return root;
    }

    function applyTransientSceneScale(imported: ImportedFluidScene): void {
        imported.assetRoot.scaling.set(
            imported.assetRootScaling[0] * transientSceneMeshScale,
            imported.assetRootScaling[1] * transientSceneMeshScale,
            imported.assetRootScaling[2] * transientSceneMeshScale
        );
        updateFluidSceneSdfStaticScale(imported.sceneSdf, transientSceneMeshScale, imported.assetRootPosition, { resetMotion: true });
        canvas.dataset.whiteboardMeshScale = String(transientSceneMeshScale);
        canvas.dataset.importedCollisionCellSize = String(imported.bundle.collision.cellSize * transientSceneMeshScale);
        syncImportedStaticCollisionDebug(imported);
    }

    const STATIC_COLLISION_DEBUG_ID = "__static__";
    const importedCollisionOffset = (imported: ImportedFluidScene, id: string): [number, number, number] => imported.collisionOffsets.get(id) ?? [0, 0, 0];
    function collisionIsoLines(collision: BlenderFluidCollision, offset: readonly [number, number, number] = [0, 0, 0]): Vec3[][] {
        const [dimX, dimY, dimZ] = collision.dims;
        const stride = Math.max(1, Math.ceil(Math.max(dimX, dimY, dimZ) / 48));
        const half = collision.cellSize * stride * 0.2;
        const lines: Vec3[][] = [];
        const maximumCrossings = 12_000;
        const value = (i: number, j: number, k: number): number => collision.distances[i + dimX * (j + dimY * k)]!;
        const point = (i: number, j: number, k: number): [number, number, number] => [
            collision.origin[0] + i * collision.cellSize + offset[0],
            collision.origin[1] + j * collision.cellSize + offset[1],
            collision.origin[2] + k * collision.cellSize + offset[2],
        ];
        const appendCrossing = (a: readonly [number, number, number], b: readonly [number, number, number], va: number, vb: number): boolean => {
            if (va < 0 === vb < 0) {
                return true;
            }
            const t = va / (va - vb);
            const x = a[0] + (b[0] - a[0]) * t;
            const y = a[1] + (b[1] - a[1]) * t;
            const z = a[2] + (b[2] - a[2]) * t;
            lines.push(
                [flowPoint(x - half, y, z), flowPoint(x + half, y, z)],
                [flowPoint(x, y - half, z), flowPoint(x, y + half, z)],
                [flowPoint(x, y, z - half), flowPoint(x, y, z + half)]
            );
            return lines.length < maximumCrossings * 3;
        };
        for (let k = 0; k < dimZ; k += stride) {
            for (let j = 0; j < dimY; j += stride) {
                for (let i = 0; i < dimX; i += stride) {
                    const a = point(i, j, k);
                    const va = value(i, j, k);
                    if (i + stride < dimX && !appendCrossing(a, point(i + stride, j, k), va, value(i + stride, j, k))) {
                        return lines;
                    }
                    if (j + stride < dimY && !appendCrossing(a, point(i, j + stride, k), va, value(i, j + stride, k))) {
                        return lines;
                    }
                    if (k + stride < dimZ && !appendCrossing(a, point(i, j, k + stride), va, value(i, j, k + stride))) {
                        return lines;
                    }
                }
            }
        }
        return lines;
    }

    function syncImportedStaticCollisionDebug(imported: ImportedFluidScene): void {
        const mesh = imported.collisionDebugMeshes.get(STATIC_COLLISION_DEBUG_ID);
        if (!mesh) {
            return;
        }
        mesh.position.set(
            imported.assetRootPosition[0] + imported.gridOffset[0] + importedCollisionOffset(imported, STATIC_COLLISION_DEBUG_ID)[0],
            imported.assetRootPosition[1] + imported.gridOffset[1] + importedCollisionOffset(imported, STATIC_COLLISION_DEBUG_ID)[1],
            imported.assetRootPosition[2] + imported.gridOffset[2] + importedCollisionOffset(imported, STATIC_COLLISION_DEBUG_ID)[2]
        );
        mesh.scaling.set(transientSceneMeshScale, transientSceneMeshScale, transientSceneMeshScale);
    }

    function animatedCollisionTransform(imported: ImportedFluidScene, binding: ImportedAnimatedCollisionBinding): Mat4 {
        const offset = importedCollisionOffset(imported, binding.id);
        return offset[0] === 0 && offset[1] === 0 && offset[2] === 0
            ? (binding.node.worldMatrix as Mat4)
            : mat4Multiply(mat4Translation(offset[0], offset[1], offset[2]), binding.node.worldMatrix);
    }

    function syncImportedAnimatedCollisionDebug(
        imported: ImportedFluidScene,
        binding: ImportedAnimatedCollisionBinding,
        transform = animatedCollisionTransform(imported, binding)
    ): void {
        const mesh = imported.collisionDebugMeshes.get(binding.id);
        if (!mesh) {
            return;
        }
        const decomposed = mat4Decompose(transform);
        mesh.position.set(decomposed.translation.x, decomposed.translation.y, decomposed.translation.z);
        mesh.rotationQuaternion.set(decomposed.rotation.x, decomposed.rotation.y, decomposed.rotation.z, decomposed.rotation.w);
        mesh.scaling.set(decomposed.scale.x, decomposed.scale.y, decomposed.scale.z);
    }

    function importedCollisionDebugMesh(imported: ImportedFluidScene, id: string): Mesh {
        const existing = imported.collisionDebugMeshes.get(id);
        if (existing) {
            return existing;
        }
        const animated = imported.bundle.animatedCollisions.find((entry) => entry.id === id);
        const collision = animated?.collision ?? imported.bundle.collision;
        const mesh = createLineSystem(engine, {
            name: `fluid-collision-sdf-${id}`,
            lines: collisionIsoLines(collision, animated ? [0, 0, 0] : [-imported.assetRootPosition[0], -imported.assetRootPosition[1], -imported.assetRootPosition[2]]),
            material: createLineMaterial({
                name: `fluid-collision-sdf-${id}-material`,
                color: animated ? { r: 1, g: 0.25, b: 0.85, a: 0.9 } : { r: 0.15, g: 1, b: 0.85, a: 0.9 },
                useVertexAlpha: true,
                depthWrite: false,
                depthCompare: "always",
            }),
        });
        mesh.pickable = false;
        mesh.renderOrder = 10_000;
        addToScene(scene, mesh);
        imported.collisionDebugMeshes.set(id, mesh);
        if (animated) {
            const node = imported.animatedCollisionBindings.find((binding) => binding.id === id)?.node;
            if (!node) {
                removeFromScene(scene, mesh);
                throw new Error(`Animated fluid collision debug mesh "${id}" has no driving node.`);
            }
            syncImportedAnimatedCollisionDebug(imported, { id, node });
        } else {
            syncImportedStaticCollisionDebug(imported);
        }
        setMeshVisible(mesh, false);
        return mesh;
    }

    function syncImportedSceneGridTransform(imported: ImportedFluidScene): void {
        const gridPosition = effectiveGridSettings().position;
        const scaledReferenceGridPosition: [number, number, number] = [
            imported.assetRootPosition[0] + (imported.referenceGridPosition[0] - imported.assetRootPosition[0]) * transientSceneMeshScale,
            imported.assetRootPosition[1] + (imported.referenceGridPosition[1] - imported.assetRootPosition[1]) * transientSceneMeshScale,
            imported.assetRootPosition[2] + (imported.referenceGridPosition[2] - imported.assetRootPosition[2]) * transientSceneMeshScale,
        ];
        const nextOffset: [number, number, number] = [
            gridPosition[0] - scaledReferenceGridPosition[0],
            gridPosition[1] - scaledReferenceGridPosition[1],
            gridPosition[2] - scaledReferenceGridPosition[2],
        ];
        const offsetDelta: [number, number, number] = [nextOffset[0] - imported.gridOffset[0], nextOffset[1] - imported.gridOffset[1], nextOffset[2] - imported.gridOffset[2]];
        const moved = offsetDelta.some((component) => Math.abs(component) > 1e-8);
        imported.assetRoot.position.set(
            imported.assetRootPosition[0] + nextOffset[0],
            imported.assetRootPosition[1] + nextOffset[1],
            imported.assetRootPosition[2] + nextOffset[2]
        );
        for (const binding of imported.sourceBindings) {
            binding.lastPosition[0] += offsetDelta[0];
            binding.lastPosition[1] += offsetDelta[1];
            binding.lastPosition[2] += offsetDelta[2];
        }
        imported.gridOffset = nextOffset;
        const staticOffset = importedCollisionOffset(imported, STATIC_COLLISION_DEBUG_ID);
        const origin: [number, number, number] = [
            imported.assetRootPosition[0] + (imported.collisionOrigin[0] - imported.assetRootPosition[0]) * transientSceneMeshScale + nextOffset[0] + staticOffset[0],
            imported.assetRootPosition[1] + (imported.collisionOrigin[1] - imported.assetRootPosition[1]) * transientSceneMeshScale + nextOffset[1] + staticOffset[1],
            imported.assetRootPosition[2] + (imported.collisionOrigin[2] - imported.assetRootPosition[2]) * transientSceneMeshScale + nextOffset[2] + staticOffset[2],
        ];
        updateFluidSceneSdfStaticOffset(imported.sceneSdf, [nextOffset[0] + staticOffset[0], nextOffset[1] + staticOffset[1], nextOffset[2] + staticOffset[2]], {
            resetMotion: true,
        });
        if (moved) {
            updateImportedCollisionTransforms(imported, 0, true);
        }
        syncImportedStaticCollisionDebug(imported);
        canvas.dataset.importedCollisionOrigin = origin.join(",");
        canvas.dataset.importedSceneOffset = nextOffset.join(",");
    }

    function createImportedEmitterSourceBindings(asset: AssetContainer): ImportedEmitterSourceBinding[] {
        const nodes = importedSceneNodes(asset);
        const bindings: ImportedEmitterSourceBinding[] = [];
        let missing = 0;
        for (const emitter of installedFlow.emitters) {
            if (!emitter.sourceNode) {
                continue;
            }
            const matches = nodes.get(emitter.sourceNode);
            if (!matches) {
                missing++;
                console.warn(`Fluid emitter "${emitter.id}" references missing GLB source node "${emitter.sourceNode}"; using its static analytical transform.`);
                continue;
            }
            if (matches.length !== 1) {
                throw new Error(`Fluid emitter "${emitter.id}" references ambiguous GLB source node "${emitter.sourceNode}".`);
            }
            const node = matches[0]!;
            if (emitter.sourcePresentation === false) {
                setImportedNodePresentation(node, false);
            }
            const transform = createEmitterSourceTransform(node.worldMatrix, emitter.transform);
            bindings.push({ emitterId: emitter.id, node, transform, lastPosition: [...emitter.transform.position] });
        }
        canvas.dataset.importedMissingEmitterSourceCount = String(missing);
        return bindings;
    }

    function flowShapeHalfExtents(shape: FluidShape): [number, number, number] {
        if (shape.type === "box") {
            return [shape.size[0] * 0.5, shape.size[1] * 0.5, shape.size[2] * 0.5];
        }
        if (shape.type === "sphere") {
            return [shape.radius, shape.radius, shape.radius];
        }
        if (shape.type === "cylinder") {
            return [shape.radius, shape.height * 0.5, shape.radius];
        }
        if (shape.type === "cone") {
            const radius = Math.max(shape.bottomRadius, shape.topRadius);
            return [radius, shape.height * 0.5, radius];
        }
        if (shape.type === "capsule") {
            return [shape.radius, shape.height * 0.5, shape.radius];
        }
        let halfX = 0;
        let halfZ = 0;
        for (const point of shape.points) {
            halfX = Math.max(halfX, Math.abs(point[0]));
            halfZ = Math.max(halfZ, Math.abs(point[1]));
        }
        return [halfX, shape.thickness * 0.5, halfZ];
    }

    function includeFlowShapeBounds(minimum: [number, number, number], maximum: [number, number, number], object: FluidEmitter | FluidSink): void {
        const local = flowShapeHalfExtents(object.shape);
        const scaled: [number, number, number] = [
            local[0] * Math.abs(object.transform.scale[0]),
            local[1] * Math.abs(object.transform.scale[1]),
            local[2] * Math.abs(object.transform.scale[2]),
        ];
        const [rawX, rawY, rawZ, rawW] = object.transform.rotation;
        const invLength = 1 / Math.max(Math.sqrt(rawX * rawX + rawY * rawY + rawZ * rawZ + rawW * rawW), 1e-12);
        const x = rawX * invLength;
        const y = rawY * invLength;
        const z = rawZ * invLength;
        const w = rawW * invLength;
        const xx = x * x;
        const yy = y * y;
        const zz = z * z;
        const xy = x * y;
        const xz = x * z;
        const yz = y * z;
        const wx = w * x;
        const wy = w * y;
        const wz = w * z;
        const worldHalf: [number, number, number] = [
            Math.abs(1 - 2 * (yy + zz)) * scaled[0] + Math.abs(2 * (xy - wz)) * scaled[1] + Math.abs(2 * (xz + wy)) * scaled[2],
            Math.abs(2 * (xy + wz)) * scaled[0] + Math.abs(1 - 2 * (xx + zz)) * scaled[1] + Math.abs(2 * (yz - wx)) * scaled[2],
            Math.abs(2 * (xz - wy)) * scaled[0] + Math.abs(2 * (yz + wx)) * scaled[1] + Math.abs(1 - 2 * (xx + yy)) * scaled[2],
        ];
        for (let axis = 0; axis < 3; axis++) {
            minimum[axis] = Math.min(minimum[axis]!, object.transform.position[axis]! - worldHalf[axis]!);
            maximum[axis] = Math.max(maximum[axis]!, object.transform.position[axis]! + worldHalf[axis]!);
        }
    }

    function frameImportedWhiteboardScene(): void {
        const grid = effectiveGridSettings();
        const minimum: [number, number, number] = [grid.position[0] - grid.size[0] * 0.5, grid.position[1] - grid.size[1] * 0.5, grid.position[2] - grid.size[2] * 0.5];
        const maximum: [number, number, number] = [grid.position[0] + grid.size[0] * 0.5, grid.position[1] + grid.size[1] * 0.5, grid.position[2] + grid.size[2] * 0.5];
        for (const object of [...installedFlow.emitters, ...installedFlow.sinks]) {
            includeFlowShapeBounds(minimum, maximum, object);
        }
        const center: [number, number, number] = [(minimum[0] + maximum[0]) * 0.5, (minimum[1] + maximum[1]) * 0.5, (minimum[2] + maximum[2]) * 0.5];
        const halfX = (maximum[0] - minimum[0]) * 0.5;
        const halfY = (maximum[1] - minimum[1]) * 0.5;
        const halfZ = (maximum[2] - minimum[2]) * 0.5;
        const radius = Math.max(0.1, Math.sqrt(halfX * halfX + halfY * halfY + halfZ * halfZ));
        const aspect = Math.max(getEffectiveAspectRatio(cam, canvas.width, canvas.height), 1e-6);
        const horizontalFov = 2 * Math.atan(Math.tan(cam.fov * 0.5) * aspect);
        const limitingFov = Math.min(cam.fov, horizontalFov);
        cam.alpha = Math.PI / 2;
        cam.target.x = center[0];
        cam.target.y = center[1];
        cam.target.z = center[2];
        cam.radius = (radius / Math.sin(limitingFov * 0.5)) * 1.1;
        cam.farPlane = Math.max(cam.farPlane, cam.radius + radius * 2);
        canvas.dataset.importedCameraFramed = "true";
        canvas.dataset.importedCameraAlpha = String(cam.alpha);
        canvas.dataset.importedCameraRadius = String(cam.radius);
        canvas.dataset.importedCameraTarget = center.join(",");
    }

    function updateImportedEmitterSources(imported: ImportedFluidScene, deltaMs: number): void {
        const deltaSeconds = deltaMs / 1000;
        for (const binding of imported.sourceBindings) {
            const emitter = installedFlow.emitters.find((candidate) => candidate.id === binding.emitterId);
            if (!emitter) {
                continue;
            }
            emitter.transform = resolveEmitterSourceTransform(binding.transform, binding.node.worldMatrix);
            const position = emitter.transform.position;
            if (emitter.sourceVelocityFactor !== undefined && deltaSeconds > 0) {
                emitter.sourceVelocity = [
                    (position[0] - binding.lastPosition[0]) / deltaSeconds,
                    (position[1] - binding.lastPosition[1]) / deltaSeconds,
                    (position[2] - binding.lastPosition[2]) / deltaSeconds,
                ];
            } else {
                delete emitter.sourceVelocity;
            }
            updateFluidSimulationEmitter(activeSim, emitter);
            binding.lastPosition = position;
        }
        const first = imported.sourceBindings[0];
        canvas.dataset.importedAnimationTime = String(imported.asset.animationGroups?.[0]?.currentTime ?? 0);
        if (first) {
            const emitter = installedFlow.emitters.find((candidate) => candidate.id === first.emitterId);
            if (emitter) {
                canvas.dataset.importedEmitterPosition = emitter.transform.position.join(",");
                canvas.dataset.importedEmitterSourceVelocity = (emitter.sourceVelocity ?? [0, 0, 0]).join(",");
            }
        }
    }

    function updateImportedCollisionTransforms(imported: ImportedFluidScene, deltaMs: number, resetMotion = false): void {
        const transforms = imported.animatedCollisionBindings.map((binding) => ({
            id: binding.id,
            localToWorld: animatedCollisionTransform(imported, binding),
        }));
        updateFluidSceneSdfTransforms(imported.sceneSdf, transforms, deltaMs / 1000, { resetMotion });
        for (let index = 0; index < imported.animatedCollisionBindings.length; index++) {
            syncImportedAnimatedCollisionDebug(imported, imported.animatedCollisionBindings[index]!, transforms[index]!.localToWorld);
        }
        canvas.dataset.importedAnimatedCollisionCount = String(imported.animatedCollisionBindings.length);
        const first = transforms[0]?.localToWorld;
        if (first) {
            canvas.dataset.importedAnimatedCollisionPosition = [first[12], first[13], first[14]].join(",");
        }
    }

    function updateImportedSceneBindings(imported: ImportedFluidScene, deltaMs: number, resetMotion = false): void {
        updateImportedEmitterSources(imported, deltaMs);
        updateImportedCollisionTransforms(imported, deltaMs, resetMotion);
    }

    // ── Live tuning UI ───────────────────────────────────────────────
    // A combo box to switch method, per-method parameter sliders (applied live),
    // and a reset button. The per-method schema (each entry mirrors the sim's
    // current default and remembers the last value the user set) is shared with the
    // other fluid apps via `DEFAULT_FLUID_SCHEMAS` (fluid/controls/controls-panel).

    // The shared control panel (GENERAL / RENDER / FOAM / PHYSICS sections + the
    // pinned top-left GPU-timing panel) is built by the reusable component
    // (./fluid/controls/controls-panel.ts); this demo owns only the scene-specific "Demo"
    // section (scene dropdown + demo params + the container-visibility toggle) and the
    // "Export" section, mounted into the component's slots. Every on* callback below
    // applies the SAME effect the old inline handler did; the component's programmatic
    // setters drive the controls on pair-state restore WITHOUT re-firing side effects.
    const containerSel = document.createElement("select");
    const DEMO_SEL_CSS = "flex:1;min-width:0;padding:3px;background:#1a2230;color:#dfe6ee;border:1px solid #33415a;border-radius:4px;";
    containerSel.style.cssText = DEMO_SEL_CSS;
    // Options are populated from the demo registry once it is built (below).
    containerSel.onchange = () => {
        const demo = demos.find((d) => d.key === containerSel.value);
        if (demo) {
            // A demo may ask to open on its own method/quality the first time it is picked
            // (see FluidDemo.defaultMethod). Only ever on the first visit, so coming back to a
            // demo keeps whatever the user last set it to.
            const first = !visitedDemos.has(demo.key);
            visitedDemos.add(demo.key);
            requestPairSwitch(demo, (first && demo.defaultMethod) || methodName, (first && demo.defaultQuality) || quality);
            // switchPair adopts the pair it actually loaded — mirror that back into the two
            // selectors so the panel never disagrees with the running sim.
            qualitySel.value = quality;
            controls.setMethod(methodName);
        }
    };
    // Quality tier (low / middle / high) — sits NEXT TO the demo dropdown. Each
    // (demo, method, quality) is its own pair, seeded from the matching preset file, so
    // picking a tier loads that tier's settings (and remembers per-tier tweaks).
    const qualitySel = document.createElement("select");
    qualitySel.style.cssText = DEMO_SEL_CSS;
    for (const q of QUALITIES) {
        const opt = document.createElement("option");
        opt.value = q;
        opt.textContent = q.charAt(0).toUpperCase() + q.slice(1);
        qualitySel.appendChild(opt);
    }
    qualitySel.value = quality;
    qualitySel.onchange = () => {
        if (activeDemo) {
            requestPairSwitch(activeDemo, methodName, qualitySel.value as Quality);
        }
    };
    // Demo + quality dropdowns, side by side, at the top of the "Demo" section.
    const demoQualityRow = document.createElement("div");
    demoQualityRow.style.cssText = "display:flex;gap:6px;margin-bottom:8px;";
    demoQualityRow.append(containerSel, qualitySel);
    // Environment picker — applies to EVERY demo, overriding its own `envUrl` default.
    const envRow = document.createElement("div");
    envRow.style.cssText = "margin-bottom:8px;";
    const envLabel = document.createElement("div");
    envLabel.textContent = "Environment";
    envLabel.style.cssText = "font-weight:600;margin-bottom:4px;";
    const envSel = document.createElement("select");
    envSel.style.cssText = DEMO_SEL_CSS;
    {
        const auto = document.createElement("option");
        auto.value = "";
        auto.textContent = "Auto (per demo)";
        envSel.appendChild(auto);
    }
    for (const c of ENV_CHOICES) {
        const opt = document.createElement("option");
        opt.value = c.key;
        opt.textContent = c.label;
        envSel.appendChild(opt);
    }
    envSel.value = "";
    envSel.onchange = () => {
        envOverride = envSel.value || null;
        if (activeDemo) {
            applyDemoEnv(activeDemo);
        }
    };
    envRow.append(envLabel, envSel);

    // Environment yaw. Turns the backdrop, the PBR image-based lighting and the fluid's own
    // reflections together: `setEnvironmentRotation` updates the scene UBO every frame
    // (the skybox and the PBR IBL read it from there) and mirrored into the fluid surface
    // pass, which owns a separate uniform block.
    let envRotationDeg = 0;
    const envRotRow = document.createElement("div");
    const envRotHead = document.createElement("div");
    envRotHead.style.cssText = "display:flex;justify-content:space-between;";
    const envRotLab = document.createElement("span");
    envRotLab.textContent = "Environment rotation";
    const envRotVal = document.createElement("span");
    envRotVal.style.cssText = "color:#9fb4cc;";
    envRotVal.textContent = "0\u00b0";
    envRotHead.append(envRotLab, envRotVal);
    const envRotInput = document.createElement("input");
    envRotInput.type = "range";
    envRotInput.min = "0";
    envRotInput.max = "360";
    envRotInput.step = "1";
    envRotInput.value = "0";
    envRotInput.style.cssText = "width:100%;";
    const applyEnvRotation = (deg: number): void => {
        envRotationDeg = deg;
        envRotVal.textContent = `${Math.round(deg)}\u00b0`;
        const rad = (deg * Math.PI) / 180;
        setEnvironmentRotation(scene, rad);
        configureFluidSimulationRenderLayer(surfaceTask, { environmentRotationY: rad });
        configureFluidSimulationRenderLayer(polygonSurfaceTask, { environmentRotationY: rad });
    };
    const applyEnvironmentRotationInput = applyLatestOnAnimationFrame(applyEnvRotation);
    envRotInput.oninput = () => applyEnvironmentRotationInput(parseFloat(envRotInput.value));
    envRotRow.append(envRotHead, envRotInput);
    void envRotationDeg;

    // Environment INTENSITY. babylon-lite has no scene-wide equivalent of BJS
    // `scene.environmentIntensity`: the multiplier lives on each PBR material's own UBO slice
    // (`material.environmentIntensity`, read by the IBL / clearcoat / sheen / refraction
    // fragments), so the slider fans the value out over every PBR material in the scene and
    // bumps their UBO version — that version bump is what makes the renderer re-upload the slice.
    //
    // This is NOT exposure. It scales only the image-based LIGHTING that lands on surfaces: the
    // skybox snapshots its own grade into its mesh UBO at build time, and the fluid surface
    // samples the reflection cube directly, so both keep their brightness. That is deliberate —
    // the sky stays the fixed reference and the slider balances how hard it lights the geometry
    // against the direct sun and the baked AO, instead of just making the whole frame darker.
    //
    // Global rather than per-demo (unlike the rotation, which each demo declares): every demo is
    // authored to look right at 1.0, so there is no per-demo value to restore.
    //
    // It is a no-op on the capsule / box / fountain demos, and that is correct rather than a gap:
    // their container and nozzle meshes are STANDARD materials, which have no image-based
    // lighting term at all. Only the glTF-backed demos (the waterfall's rock + oasis) are PBR.
    let envIntensity = 1;
    /** Until the slider is touched, every material is already at the shader's own 1.0 default,
     *  so the fan-out is skipped entirely — the common case costs nothing. */
    let envIntensityTouched = false;
    /** Last value pushed to each material, so a re-scan only dirties what actually changed. */
    const envIntensityApplied = new WeakMap<Material, number>();
    /** `scene.meshes` grows as demos resolve their glTFs, and a mesh that arrives after the
     *  slider was moved would otherwise keep the default. Re-fanned out when the count changes. */
    let envIntensityMeshCount = -1;
    const pushEnvIntensity = (): void => {
        for (const mesh of scene.meshes) {
            const mat = mesh.material;
            // Standard materials carry no IBL term at all. `specularPower` is the same
            // discriminator babylon-lite's own material tracking uses to tell the two apart.
            if (!mat || "specularPower" in mat || envIntensityApplied.get(mat) === envIntensity) {
                continue;
            }
            (mat as PbrMaterialProps).environmentIntensity = envIntensity;
            envIntensityApplied.set(mat, envIntensity);
            markMaterialUboDirty(mat);
        }
        envIntensityMeshCount = scene.meshes.length;
    };
    const envIntRow = document.createElement("div");
    const envIntHead = document.createElement("div");
    envIntHead.style.cssText = "display:flex;justify-content:space-between;";
    const envIntLab = document.createElement("span");
    envIntLab.textContent = "Environment intensity";
    const envIntVal = document.createElement("span");
    envIntVal.style.cssText = "color:#9fb4cc;";
    envIntVal.textContent = "1.00\u00d7";
    envIntHead.append(envIntLab, envIntVal);
    const envIntInput = document.createElement("input");
    envIntInput.type = "range";
    envIntInput.min = "0";
    envIntInput.max = "3";
    envIntInput.step = "0.05";
    envIntInput.value = "1";
    envIntInput.style.cssText = "width:100%;";
    const applyEnvIntensity = (v: number): void => {
        envIntensity = v;
        envIntensityTouched = true;
        envIntVal.textContent = `${v.toFixed(2)}\u00d7`;
        pushEnvIntensity();
    };
    const applyEnvironmentIntensityInput = applyLatestOnAnimationFrame(applyEnvIntensity);
    envIntInput.oninput = () => applyEnvironmentIntensityInput(parseFloat(envIntInput.value));
    envIntRow.append(envIntHead, envIntInput);

    // Anti-aliasing toggle — applies to EVERY demo. Off by default: it is only worth its cost
    // on scenes with dense hard-edged geometry (the waterfall's oasis backdrop), and the first
    // enable allocates the multisampled colour + depth pair. See `setMsaa` for the wiring.
    const msaaRow = document.createElement("label");
    msaaRow.style.cssText = "display:flex;align-items:center;gap:6px;margin:8px 0;cursor:pointer;";
    const msaaChk = document.createElement("input");
    msaaChk.type = "checkbox";
    const msaaTxt = document.createElement("span");
    // Concatenation, NOT a template literal: the WGSL minifier that runs over emitted chunks
    // treats every backtick template as shader source and collapses its whitespace, which eats
    // the space before "(MSAA" in the built bundle. (Same trap as the Poly Haven URLs above.)
    msaaTxt.textContent = "Anti-aliasing (MSAA " + MSAA_SAMPLES + "\u00d7)";
    msaaRow.append(msaaChk, msaaTxt);
    msaaChk.onchange = () => setMsaa(msaaChk.checked);
    // Host for the active demo's live tunables ("Demo parameters") + its demo-specific
    // panel controls.
    const demoParamsHost = document.createElement("div");
    const initialGridSettings = defaultGridSettings(methodName, domainScale);
    const initialFlipDiscretization = flipDiscretizationForGrid(initialGridSettings, flipGridResolution);

    function finishFluidControlPlan(bindingPlan: FluidControlsApplicationPlan, changedKeys: readonly (keyof FluidControlValues)[]): void {
        if (bindingPlan.reconfigure) {
            particleCount = activeSim.count;
            builtImportedInitialState = importedInitialState;
            lastTransitionPeakBytes = controlsBinding!.memory.transitionPeakBytes;
            trackSimBuild(methodName);
            builtGridSettings = gridSettings ? cloneGridSettings(gridSettings) : undefined;
            builtGridMethod = methodName;
            builtPhysicsScale = physicsScale;
            builtFlipGridResolution = flipGridResolution;
            builtFlipMarkersPerCell = flipMarkersPerCell;
            builtWithGridFloor = importedCollisionActive || activeDemo?.useGridFloor === true;
            builtDomainScale = domainScale;
            canvas.dataset.simulationTransitionBytes = String(lastTransitionPeakBytes);
            canvas.dataset.particleCount = String(particleCount);
            canvas.dataset.flipParticleCapacityRequest = String(flipParticleCapacityRequest);
            canvas.dataset.activeBlocks = String(mpmActiveBlocks);
            canvas.dataset.pagedGrid = String(methodName === "FLIP" ? flipPagedGrid : mpmPagedGrid);
            canvas.dataset.pagedGridMaxPages = String(methodName === "FLIP" ? flipPagedGridMaxPages : mpmPagedGridMaxPages);
            canvas.dataset.fusedBlockDiscovery = String(mpmFusedBlockDiscovery);
            builtInitialStateKey = flipParticlePlan(requestedParticleCount()).initialStateKey;
            syncGridControls();
            syncGridBoundsWireframe();
            refreshParticleUsageStatus();
            if (changedKeys.some((key) => key === "count" || key === "physScale" || key === "gridResolution" || key === "markersPerCell")) {
                restartSimulationLifecycle();
            }
        }
        if (bindingPlan.restartRequired) {
            syncGridControls();
            refreshParticleUsageStatus();
            const deferInitialPlan = methodName === "FLIP" && changedKeys.includes("markersPerCell");
            controls.setGridStatus((deferInitialPlan ? "" : flipPendingCapacityStatus()) || "Simulation changes pending. Reset simulation to apply.");
        }
    }

    const controls = createFluidControlsPanel({
        schemas: DEFAULT_FLUID_SCHEMAS,
        methods: Object.keys(DEFAULT_FLUID_SCHEMAS),
        particleCounts: PARTICLE_COUNTS,
        physScaleMin: PHYS_MIN_SCALE,
        physScaleMax: PHYS_MAX_SCALE,
        flipParticleCapacityMax: deviceParticleCapacityForMethod("FLIP"),
        initial: {
            method: methodName,
            material: pbmpmMaterial,
            count: DEFAULT_PARTICLE_COUNT,
            simulationDuration,
            alphaDecay: simulationAlphaDecay,
            physScale: physicsScale,
            gridPosition: [...initialGridSettings.position],
            gridSize: [...initialGridSettings.size],
            cellSize: methodName === "FLIP" ? initialFlipDiscretization.dx : cellSizeForPhysicsScale(methodName, physicsScale) * domainScale,
            gridResolution: methodName === "FLIP" ? flipGridResolution : gridResolutionForScale(methodName, physicsScale, Math.max(...initialGridSettings.size)),
            markersPerCell: flipMarkersPerCell,
            showGridBounds,
            showGridBoundsSolid,
            color: "#16a3c3", // matches the default FLUID_COLOR
            independentRendering: false,
            absorption: 1,
            size: 1,
            refraction: 0.1,
            specular: 250,
            // The reflection tonemap wants to match the scene's own image processing, but the HDR
            // loader sets that per environment, so seed with the shader's historical constants and
            // let each demo's env load correct them (see the loadHdrEnvironment call sites).
            reflectionExposure: 1,
            reflectionContrast: 1.1,
            reflectivity: 0.02,
            depthBlur: 20,
            depthBlurThreshold: 10,
            thicknessBlur: 10,
            half: false,
            thicknessDownscale: 2,
            surfaceFilter: "narrowRange",
            narrowDelta: 10,
            narrowMu: 1,
            anisotropic: false,
            anisoSurfScale: 0.5,
            renderMode: "surface",
            polygonShader: "physical",
            debug: "none",
            showContainer: true,
            activeBlocks: false,
            pagedGrid: false,
            pagedGridMaxPages: mpmPagedGridMaxPages,
            fusedBlockDiscovery: false,
            foam: {
                enabled: false,
                activeParticles: true,
                generateSpray: true,
                generateFoam: true,
                generateBubbles: true,
                surfaceFiltering: false,
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
        gpu: { stages: ["Simulation", "Foam gen", "Surface", "Foam render", "Particles"], supported: profiler !== null },
        onApply: function applyControlSnapshot(values, changedKeys) {
            if (deferReferenceMutation(() => applyControlSnapshot(structuredClone(values), [...changedKeys]))) {
                return;
            }
            const applyingPairState = loadingPairState;
            const pairChange = changedKeys.includes("method") || changedKeys.includes("material");
            const bindingPlan =
                controlsBinding && (applyingPairState || !pairChange) ? applyFluidControls(controlsBinding, values, applyingPairState ? undefined : changedKeys) : null;
            if (bindingPlan) {
                const appliedKeys = applyingPairState ? bindingPlan.changedKeys : changedKeys;
                finishFluidControlPlan(bindingPlan, appliedKeys);
                if (appliedKeys.includes("gridPosition") || appliedKeys.includes("gridSize")) {
                    syncGridBoundsWireframe();
                    syncGridGizmo();
                }
            }
            const changed = (key: keyof typeof values): boolean => changedKeys.includes(key) && (!controlsBinding || !SHARED_FLUID_BINDING_KEYS.includes(key));

            // Pair changes load another complete control snapshot, so run them after this snapshot
            // callback returns rather than nesting a controls transaction inside onApply.
            if (!applyingPairState && (changed("method") || changed("material"))) {
                const nextMethod = values.method;
                const nextMaterial = values.material;
                queueMicrotask(() => requestPairSwitch(activeDemo!, nextMethod, quality, nextMaterial));
            }
            if (changed("count")) {
                if (methodName === "FLIP") {
                    setFlipParticleCapacity(values.count);
                } else {
                    setParticleCount(values.count);
                }
            }
            if (changed("simulationDuration")) {
                simulationDuration = values.simulationDuration;
                syncSimulationLifecycle();
            }
            if (changed("alphaDecay")) {
                simulationAlphaDecay = values.alphaDecay;
                syncSimulationLifecycle();
            }
            if (changed("renderMode")) {
                applyRenderMode(values.renderMode === "spheres");
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
                configureFluidSimulationRenderLayer(particleTask, { profile });
                canvas.dataset.surfaceShader = values.polygonShader;
                canvas.dataset.polygonShader = values.polygonShader;
                if (changed("anisotropic")) {
                    renderAnisotropic = values.anisotropic;
                    applyEffectiveRenderMode();
                }
            }
            if (changed("showContainer")) {
                activeDemo?.setContainerVisible?.(importedScene ? false : values.showContainer);
            }
            if (changed("debug")) {
                configureFluidSimulationRenderLayer(surfaceTask, {
                    debug: values.debug === "polygonWireframe" ? "none" : (values.debug as FluidDebug),
                });
                configureFluidSimulationRenderLayer(polygonSurfaceTask, {
                    polygonWireframe: values.debug === "polygonWireframe",
                });
                surfaceDebugActive = values.debug !== "none";
                configureFluidSimulationRenderLayer(foamTask, { enabled: foamRenderVisible() });
            }
            if (changed("schema")) {
                for (const [key, value] of Object.entries(values.schema)) {
                    applyParam(activeSim, key, value);
                }
                renderPolygonSurface = methodName === "FLIP" && (values.schema.polygonSurface ?? 0) >= 0.5;
                applyEffectiveRenderMode();
            }
            if (changed("physScale")) {
                setPhysicsScale(values.physScale);
            }
            if (changed("gridResolution")) {
                setGridResolution(values.gridResolution);
            }
            if (changed("markersPerCell")) {
                setMarkersPerCell(values.markersPerCell);
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

            const pagingChanged = changed("activeBlocks") || changed("pagedGrid") || changed("pagedGridMaxPages") || changed("fusedBlockDiscovery");
            if (pagingChanged) {
                const previousPaging = {
                    mpmActiveBlocks,
                    mpmPagedGrid,
                    mpmPagedGridMaxPages,
                    mpmFusedBlockDiscovery,
                    flipPagedGrid,
                    flipPagedGridMaxPages,
                };
                let rebuild = false;
                try {
                    if (methodName === "FLIP") {
                        if (changed("pagedGrid") && values.pagedGrid !== flipPagedGrid) {
                            flipPagedGrid = values.pagedGrid;
                            rebuild = true;
                        }
                        if (changed("pagedGridMaxPages")) {
                            const pages = Math.min(values.pagedGridMaxPages, maxFlipPagedGridPagesFor(effectiveGridSettings(), flipGridResolution));
                            if (pages !== flipPagedGridMaxPages) {
                                flipPagedGridMaxPages = pages;
                                rebuild ||= flipPagedGrid;
                            }
                            controls.setPagedGridMaxPages(flipPagedGridMaxPages, maxFlipPagedGridPagesFor(effectiveGridSettings(), flipGridResolution));
                        }
                    } else {
                        if (changed("activeBlocks") && values.activeBlocks !== mpmActiveBlocks) {
                            mpmActiveBlocks = values.activeBlocks;
                            rebuild = true;
                        }
                        if (changed("pagedGrid") && values.pagedGrid !== mpmPagedGrid) {
                            mpmPagedGrid = values.pagedGrid;
                            rebuild = true;
                        }
                        if (changed("pagedGridMaxPages")) {
                            const limit = maxMlsPagedGridPagesFor(effectiveGridSettings(), physicsScale);
                            const pages = Math.min(values.pagedGridMaxPages, limit);
                            if (pages !== mpmPagedGridMaxPages) {
                                mpmPagedGridMaxPages = pages;
                                rebuild ||= mpmPagedGrid;
                            }
                            controls.setPagedGridMaxPages(mpmPagedGridMaxPages, limit);
                        }
                        if (changed("fusedBlockDiscovery") && values.fusedBlockDiscovery !== mpmFusedBlockDiscovery) {
                            mpmFusedBlockDiscovery = values.fusedBlockDiscovery;
                            rebuild = true;
                        }
                    }
                    controls.setPagedGridStatus("");
                    canvas.dataset.pagedGridOverflow = "false";
                    if (rebuild) {
                        rebuildSims(requestedParticleCount(), physicsScale, methodName === "FLIP");
                    }
                } catch (error) {
                    ({ mpmActiveBlocks, mpmPagedGrid, mpmPagedGridMaxPages, mpmFusedBlockDiscovery, flipPagedGrid, flipPagedGridMaxPages } = previousPaging);
                    throw error;
                }
            }
            if (changed("foam")) {
                const foam = values.foam;
                pushFoam();
                configureFluidSimulationRenderLayer(foamTask, {
                    foam: {
                        surfaceFiltering: foam.surfaceFiltering ?? methodName === "FLIP",
                        softness: foam.softness,
                        density: foam.density,
                        subsurfaceStrength: foam.subsurfaceStrength,
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
        },
        on: {
            onGridSettings: (position, size) => (controlsBinding ? undefined : setGridSettings({ position, size })),
            onGridGizmo: (visible) => setGridGizmoVisible(visible),
            onReset: (preserveSceneAnimations) => resetActiveFlow(true, preserveSceneAnimations),
        },
    });
    const fluidControlHostState = () => ({
        flipBackendId,
        particleCount,
        physicsScale,
        flipParticleCapacityRequest,
        pbmpmMaterial,
        mpmActiveBlocks,
        mpmPagedGrid,
        mpmPagedGridMaxPages,
        mpmFusedBlockDiscovery,
        flipPagedGrid,
        flipPagedGridMaxPages,
        flipGridResolution,
        flipMarkersPerCell,
        gridSettings: gridSettings ? cloneGridSettings(gridSettings) : undefined,
        renderSpheres,
        renderAnisotropic,
        renderPolygonSurface,
        surfaceDebugActive,
    });
    const restoreFluidControlHostState = (state: ReturnType<typeof fluidControlHostState>): void => {
        ({
            flipBackendId,
            particleCount,
            physicsScale,
            flipParticleCapacityRequest,
            pbmpmMaterial,
            mpmActiveBlocks,
            mpmPagedGrid,
            mpmPagedGridMaxPages,
            mpmFusedBlockDiscovery,
            flipPagedGrid,
            flipPagedGridMaxPages,
            flipGridResolution,
            flipMarkersPerCell,
        } = state);
        gridSettings = state.gridSettings ? cloneGridSettings(state.gridSettings) : undefined;
        ({ renderSpheres, renderAnisotropic, renderPolygonSurface, surfaceDebugActive } = state);
    };
    const applyFluidControlHostState = (snapshot: Readonly<FluidControlValues>, changedKeys: readonly (keyof FluidControlValues)[]): void => {
        physicsScale = snapshot.physScale;
        if (!loadingPairState && (changedKeys.includes("gridPosition") || changedKeys.includes("gridSize"))) {
            gridSettings = { position: [...snapshot.gridPosition], size: [...snapshot.gridSize] };
        }
        flipGridResolution = snapshot.gridResolution;
        flipMarkersPerCell = snapshot.markersPerCell;
        renderSpheres = snapshot.renderMode === "spheres";
        renderAnisotropic = snapshot.anisotropic;
        renderPolygonSurface = methodName === "FLIP" && (snapshot.schema.polygonSurface ?? 0) >= 0.5;
        surfaceDebugActive = snapshot.debug !== "none";
        if (methodName === "FLIP") {
            flipParticleCapacityRequest = snapshot.count;
            flipPagedGrid = snapshot.pagedGrid;
            flipPagedGridMaxPages = snapshot.pagedGridMaxPages;
        } else {
            particleCount = snapshot.count;
            if (methodName === "MLS-MPM") {
                mpmActiveBlocks = snapshot.activeBlocks || snapshot.pagedGrid;
                mpmPagedGrid = snapshot.pagedGrid;
                mpmPagedGridMaxPages = snapshot.pagedGridMaxPages;
                mpmFusedBlockDiscovery = snapshot.fusedBlockDiscovery;
            }
        }
    };
    const fluidControlTarget = (simulation: FluidSimulation, snapshot: Readonly<FluidControlValues>): FluidControlsBindingTarget => {
        const requested = methodName === "FLIP" ? flipParticleCapacity(snapshot.count, activeFlow, snapshot.gridResolution) : snapshot.count;
        const base = simulationOptions(requested, snapshot.physScale);
        return {
            simulation,
            preserveState: !loadingPairState && !base.backend && !simulation.options.backend,
            activeCount: simulation.activeCount ?? simulation.count,
            options: {
                ...base,
                flow: flowToWorld(activeFlow),
                sceneSdf: currentSceneSdf(),
                forceField: base.backend?.supportsForces === false ? null : simulation.forceField,
                profiler: currentGpuProfiler(),
            },
        };
    };
    function initializeFluidControlsBinding(): void {
        if (controlsBinding) {
            return;
        }
        controlsBinding = bindFluidControls({
            controls,
            target: activeSim,
            deviceLimits,
            maxParticleCount: Math.max(...Object.keys(DEFAULT_FLUID_SCHEMAS).map(deviceParticleCapacityForMethod)),
            renderLayers: {
                particles: particleTask,
                surface: surfaceTask,
                polygon: polygonSurfaceTask,
                foam: foamTask,
            },
            surfaceDebugActive: () => surfaceDebugActive,
            simulationOpacity: () => simulationOpacity,
            resolveTarget: fluidControlTarget,
            captureHostState: fluidControlHostState,
            applyHostState: applyFluidControlHostState,
            restoreHostState: restoreFluidControlHostState,
            onPageDiagnostics: (diagnostics) => {
                canvas.dataset.pagedGridPages = String(diagnostics.requiredPages);
                canvas.dataset.pagedGridPageCapacity = String(diagnostics.capacity);
                canvas.dataset.pagedGridOverflow = String(diagnostics.overflow);
            },
        });
        window.addEventListener("pagehide", () => controlsBinding && disposeFluidControlsBinding(controlsBinding), { once: true });
    }
    configureFluidSimulationRenderLayer(surfaceTask, { profile: { polygonShader: controls.getValues().polygonShader } });
    configureFluidSimulationRenderLayer(polygonSurfaceTask, { profile: { polygonShader: controls.getValues().polygonShader } });
    canvas.dataset.surfaceShader = controls.getValues().polygonShader;
    canvas.dataset.polygonShader = controls.getValues().polygonShader;

    const implementationRow = document.createElement("label");
    implementationRow.style.cssText = "display:none;gap:6px;flex-direction:column;margin:10px 0;font-size:12px;";
    const implementationLabel = document.createElement("span");
    implementationLabel.textContent = "FLIP implementation";
    const implementationSelect = document.createElement("select");
    implementationSelect.dataset.fluidBackend = "true";
    implementationSelect.style.cssText = "width:100%;padding:5px;background:#1a2230;color:#eef3f8;border:1px solid #33415a;border-radius:4px;";
    for (const [value, label] of [
        ["production", "Production FLIP"],
        ["flip-reference", "FLIP Reference (experimental)"],
    ]) {
        const option = document.createElement("option");
        option.value = value!;
        option.textContent = label!;
        implementationSelect.appendChild(option);
    }
    const referenceNotice = document.createElement("div");
    referenceNotice.dataset.fluidReferenceInfo = "true";
    referenceNotice.style.cssText = "display:none;color:#a8bed3;font-size:11px;line-height:1.4;margin-bottom:8px;";
    const referenceError = document.createElement("div");
    referenceError.style.cssText = "color:#ffb7a0;font-size:11px;line-height:1.4;";
    const referenceBudgetWarning = document.createElement("div");
    referenceBudgetWarning.dataset.fluidReferenceBudgetWarning = "true";
    referenceBudgetWarning.style.cssText = "color:#f2ce7d;font-size:11px;line-height:1.4;";
    implementationRow.append(implementationLabel, implementationSelect, referenceBudgetWarning, referenceError);
    controls.implementationSlot.append(implementationRow, referenceNotice);

    function updateBackendControls(): void {
        const reference = referenceSelected();
        implementationRow.style.display = methodName === "FLIP" ? "flex" : "none";
        implementationSelect.value = reference ? "flip-reference" : "production";
        implementationSelect.disabled = !engineLoopReady;
        referenceNotice.style.display = reference ? "" : "none";
        referenceNotice.textContent = reference
            ? referenceBackend!.description + " Use modest resolution (40-80 to start). GPU-resident stepping; timestep-budget warnings do not stop playback."
            : "";
        refreshReferenceBudgetWarning();
        controls.setCapabilities(
            reference
                ? {
                      pagedGrid: false,
                      polygonSurface: false,
                      foam: referenceBackend!.supportsFoam,
                      timing: timingSupported,
                      pressureDiagnostics: false,
                      physicsParameters: referenceBackend!.physicsParameters,
                  }
                : {}
        );
        controls.setVisiblePhysicsParams(reference ? [...referenceBackend!.physicsParameters] : methodName === "PB-MPM" ? pbmpmParamKeysForMaterial(pbmpmMaterial) : null);
        flowEditor?.setInitialOnly(reference);
        if (controlsBinding) {
            syncFluidControls(controlsBinding);
        }
        canvas.dataset.fluidBackend = reference ? "flip-reference" : "production";
        canvas.dataset.referenceStatus = reference ? (paused ? "paused" : "ready") : "off";
        if (activeDemo) {
            syncHelperText(activeDemo);
        }
    }

    function reportReferenceError(error: unknown): void {
        if (error instanceof Error && error.name === "AbortError") {
            return;
        }
        const message = error instanceof Error ? error.message : String(error);
        referenceError.textContent = message;
        canvas.dataset.referenceError = message;
        canvas.dataset.referenceStatus = "error";
        if (referenceSelected()) {
            referencePausedByError ||= !paused;
            paused = true;
            canvas.dataset.paused = "true";
        }
        implementationSelect.value = activeBackendId() ?? "production";
        containerSel.value = activeDemo?.key ?? containerSel.value;
        controls.setMethod(methodName);
        console.error("[fluid] Reference operation failed", error);
    }

    function refreshReferenceBudgetWarning(): void {
        const timestep = referenceSelected() ? getFluidSimulationDiagnostics(activeSim).timestep : null;
        const dropped = timestep?.droppedSeconds ?? 0;
        const warning =
            dropped > 0
                ? "Substep budget reached. Simulation continues with stable steps; " +
                  dropped.toFixed(3) +
                  " s of fluid time skipped since reset. Increase Maximum substeps or reduce the timestep/time scale."
                : "";
        if (referenceBudgetWarning.textContent !== warning) {
            referenceBudgetWarning.textContent = warning;
        }
        canvas.dataset.referenceBudgetLimited = String(timestep?.saturated ?? false);
        canvas.dataset.referenceDroppedSeconds = String(dropped);
    }

    function completeReferenceReset(): void {
        if (!referenceSelected()) {
            return;
        }
        referenceError.textContent = "";
        delete canvas.dataset.referenceError;
        if (referencePump) {
            referencePump.error = null;
        }
        refreshReferenceBudgetWarning();
        if (referencePausedByError) {
            paused = false;
            canvas.dataset.paused = "false";
        }
        referencePausedByError = false;
        canvas.dataset.referenceStatus = paused ? "paused" : "ready";
    }

    function assertReferenceState(state: PairState, demo: FluidDemo): void {
        const flow = state.legacyFlow ? demo.flow() : { emitters: state.emitters ?? [], sinks: state.sinks ?? [] };
        if (flow.emitters.some((emitter) => emitter.enabled && emitter.behavior !== "initial") || flow.sinks.some((sink) => sink.enabled)) {
            throw new Error("[FLIP Reference] only initial fluid is supported. Disable active inflows and sinks before selecting this implementation.");
        }
        if ((state.schema.kinematicViscosity ?? 0) !== 0 || (state.schema.surfaceTension ?? 0) !== 0) {
            throw new Error("[FLIP Reference] viscosity and surface tension are not supported. Set them to zero in Production FLIP first.");
        }
    }

    function referencePairState(state: PairState, demo: FluidDemo): PairState {
        assertReferenceState(state, demo);
        return {
            ...structuredClone(state),
            backendId: "flip-reference",
            schema: { ...state.schema, polygonSurface: 0, reseedParticles: 0, particleSheeting: 0, kinematicViscosity: 0, surfaceTension: 0 },
            pagedGrid: false,
            activeBlocks: false,
            fusedBlockDiscovery: false,
            renderMode: state.renderMode === "spheres" ? "spheres" : "surface",
        };
    }

    function carryBackendScene(target: PairState, current: PairState): PairState {
        const schema = { ...target.schema };
        for (const key of referenceBackend!.physicsParameters) {
            if (current.schema[key] !== undefined) {
                schema[key] = current.schema[key]!;
            }
        }
        return {
            ...structuredClone(target),
            schema,
            simulationSemantics: current.simulationSemantics,
            demoParams: { ...current.demoParams },
            demoState: current.demoState ? structuredClone(current.demoState) : undefined,
            emitters: structuredClone(current.emitters ?? []),
            sinks: structuredClone(current.sinks ?? []),
            initialEmittersFillCapacity: current.initialEmittersFillCapacity,
            legacyFlow: false,
            grid: current.grid ? cloneGridSettings(current.grid) : undefined,
            gridResolution: current.gridResolution,
            markersPerCell: current.markersPerCell,
            physScale: current.physScale,
            count: current.count,
            camera: current.camera ? structuredClone(current.camera) : undefined,
            simulationDuration: current.simulationDuration,
            alphaDecay: current.alphaDecay,
            simulationTimeScale: current.simulationTimeScale,
        };
    }

    function pairKeyFor(demo: FluidDemo, method: string, tier: Quality, material: number, backendId?: "flip-reference"): string {
        const tierKey = demo.methodIndependentAuthoring ? "" : ":" + tier;
        const materialKey = method === "PB-MPM" ? ":m" + material : "";
        return demo.key + ":" + method + tierKey + materialKey + (method === "FLIP" && backendId ? ":b" + backendId : "");
    }

    async function selectImplementation(value: "production" | "flip-reference"): Promise<void> {
        if (!referencePump || !activeDemo || methodName !== "FLIP") {
            return;
        }
        await queueFluidReferenceChange(referencePump, async () => {
            if ((activeBackendId() ?? "production") === value) {
                return;
            }
            const previous = readLivePairState(methodName);
            const previousId = flipBackendId;
            const previousKey = currentPairKey;
            if (value === "flip-reference") {
                assertReferenceState(previous, activeDemo!);
                referenceTiming();
                await ensureReferenceBackend();
            }
            const targetId = value === "flip-reference" ? value : undefined;
            const targetKey = pairKeyFor(activeDemo!, "FLIP", quality, pbmpmMaterial, targetId);
            const previousTarget = pairStates.get(targetKey);
            const targetSharedKey = sharedStateKey(activeDemo!.key, targetId);
            const previousShared = methodIndependentStates.get(targetSharedKey);
            const target = previousTarget ?? (targetId ? previous : presetOrDefault(activeDemo!, "FLIP", quality, pbmpmMaterial));
            let next: PairState = { ...carryBackendScene(target, previous), backendId: targetId };
            if (targetId) {
                next = referencePairState(next, activeDemo!);
            }
            pairStates.set(targetKey, next);
            methodIndependentStates.set(targetSharedKey, next);
            try {
                switchPair(activeDemo!, "FLIP", quality, pbmpmMaterial, true, value);
                referenceError.textContent = "";
                delete canvas.dataset.referenceError;
            } catch (error) {
                if (previousTarget) {
                    pairStates.set(targetKey, previousTarget);
                } else {
                    pairStates.delete(targetKey);
                }
                if (previousShared) {
                    methodIndependentStates.set(targetSharedKey, previousShared);
                } else {
                    methodIndependentStates.delete(targetSharedKey);
                }
                flipBackendId = previousId;
                currentPairKey = previousKey;
                loadPairState(previous);
                throw error;
            }
        });
    }
    implementationSelect.onchange = () => {
        const value = implementationSelect.value;
        if (value !== "production" && value !== "flip-reference") {
            reportReferenceError(new Error("Unknown FLIP implementation."));
            return;
        }
        void selectImplementation(value);
    };

    function requestPairSwitch(demo: FluidDemo, method: string, tier: Quality = quality, material: number = pbmpmMaterial): void {
        if (referencePump && (referencePump.pending || activeSim.steppingMode === "async" || (method === "FLIP" && flipBackendId === "flip-reference"))) {
            void queueFluidReferenceChange(referencePump, () => switchPair(demo, method, tier, material));
        } else {
            switchPair(demo, method, tier, material);
        }
    }

    // Prepend the scene-specific "Demo" section (scene dropdown + demo params + the
    // container-visibility toggle) into the component's demo slot.
    controls.demoSlot.append(...controls.makeSection("Demo", [demoQualityRow, envRow, envRotRow, envIntRow, msaaRow, demoParamsHost, controls.containerToggleRow!]));

    const emitterFlowHost = document.createElement("div");
    const sinkFlowHost = document.createElement("div");
    controls.root.append(...controls.makeSection("Emitters", [emitterFlowHost]), ...controls.makeSection("Sinks", [sinkFlowHost]));
    collisionControlsHost = document.createElement("div");
    controls.root.append(...controls.makeSection("Collision", [collisionControlsHost]));
    controls.setSectionVisible("Collision", false);

    // ── Preset / self-contained JSON import and parameter export ─────────────
    const exportBtn = document.createElement("button");
    exportBtn.textContent = "Export parameters";
    exportBtn.style.cssText = "width:100%;padding:5px;cursor:pointer;background:#26415f;color:#eef3f8;border:1px solid #3a567a;border-radius:4px;";
    externalSceneExportRow = document.createElement("label");
    externalSceneExportRow.style.cssText = "display:none;align-items:center;gap:6px;margin:8px 0;cursor:pointer;";
    embedExternalSceneChk = document.createElement("input");
    embedExternalSceneChk.type = "checkbox";
    const embedExternalSceneText = document.createElement("span");
    embedExternalSceneText.textContent = "Embed external GLB/SDF in JSON";
    externalSceneExportRow.append(embedExternalSceneChk, embedExternalSceneText);

    function syncExternalSceneExportChoice(reset = false): void {
        if (!externalSceneExportRow || !embedExternalSceneChk) {
            return;
        }
        const external = importedScene?.bundle.preset.scene?.encoding === "external";
        externalSceneExportRow.style.display = external ? "flex" : "none";
        if (reset || !external) {
            embedExternalSceneChk.checked = false;
        }
    }

    function exportParameters(): void {
        // Serialise the live UI into the shared grouped shape (the same format the on-disk
        // quality presets use), so an exported file can be dropped straight into presets/.
        const state = readLivePairState(methodName);
        const retained =
            (currentPairKey ? presetSessions.get(currentPairKey) : undefined) ?? importFluidPresetSession(exportJsonFromPairState(activeDemo!.key, methodName, state), state);
        const application: Parameters<typeof exportFluidPresetSession>[1]["application"] = {};
        if (importedScene) {
            const preserveExternal = importedScene.bundle.preset.scene?.encoding === "external" && embedExternalSceneChk?.checked !== true;
            application.scene = {
                ...scenePayloadFromBlenderFluidJson(importedScene.bundle, { preserveExternal }),
                anchorPosition: [...importedScene.referenceGridPosition],
            };
            if (importedScene.bundle.preset.source) {
                application.source = structuredClone(importedScene.bundle.preset.source);
            }
        }
        const data = exportFluidPresetSession(retained, {
            demo: importedScene ? "blender" : activeDemo!.key,
            method: methodName,
            state,
            application,
        });
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `fluid-${activeDemo!.key}-${methodName}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    }
    exportBtn.onclick = exportParameters;
    const supportedMethods = new Set(["PBF", "FLIP", "MLS-MPM", "PB-MPM"]);
    function applyImportedPreset(json: FluidExportJson, switchMethod: boolean): void {
        const importedMethod = json.meta?.method;
        if (!importedMethod || !supportedMethods.has(importedMethod)) {
            throw new Error(`Unsupported fluid method "${importedMethod}"`);
        }
        presetFromExportJson(json);
        if (json.backendId && !referenceBackend) {
            throw new Error("[fluid] FLIP Reference must be loaded before importing its preset.");
        }
        if (switchMethod && (importedMethod !== methodName || activeBackendId() !== json.backendId)) {
            switchPair(activeDemo!, importedMethod, quality, pbmpmMaterial, true, json.backendId ?? "production");
        }
        const current = readLivePairState(methodName);
        const previousDomainScale = domainScale;
        const previousSession = currentPairKey ? presetSessions.get(currentPairKey) : undefined;
        const session = importFluidPresetSession(json, current);
        try {
            if (currentPairKey) {
                presetSessions.set(currentPairKey, session);
            }
            const imported = session.state;
            const currentDomainScale = typeof current.demoParams.meshScale === "number" ? current.demoParams.meshScale : domainScale;
            const importedDomainScale = typeof imported.demoParams.meshScale === "number" ? imported.demoParams.meshScale : currentDomainScale;
            domainScale = importedDomainScale;
            loadPairState(imported);
        } catch (error) {
            domainScale = previousDomainScale;
            if (currentPairKey) {
                if (previousSession) {
                    presetSessions.set(currentPairKey, previousSession);
                } else {
                    presetSessions.delete(currentPairKey);
                }
            }
            try {
                loadPairState(current);
            } catch (rollbackError) {
                throw new AggregateError([error, rollbackError], "Fluid preset preparation failed and its previous control snapshot could not be restored.");
            }
            throw error;
        }
    }
    async function installImportedBundle(bundle: BlenderFluidScene, generation: number, targetDemo: FluidDemo): Promise<void> {
        const importedMethod = bundle.preset.meta?.method;
        if (!importedMethod || !supportedMethods.has(importedMethod)) {
            throw new Error(`Unsupported fluid method "${importedMethod}"`);
        }
        let asset: AssetContainer;
        try {
            asset = await loadGltf(engine, bundle.sceneGlb);
        } catch (error) {
            if (generation !== importGeneration || activeDemo !== targetDemo) {
                return;
            }
            throw error;
        }
        if (generation !== importGeneration || activeDemo !== targetDemo) {
            removeFromScene(scene, asset);
            return;
        }
        const assetRoot = useImportedFluidBundleBasis(asset);
        let collision: ImportedCollisionResources;
        try {
            collision = createImportedCollision(bundle);
        } catch (error) {
            removeFromScene(scene, asset);
            throw error;
        }
        const groundWasVisible = importedScene?.groundWasVisible ?? ground.visible !== false;
        const referenceGridPosition = bundle.preset.scene?.anchorPosition ?? presetFromExportJson(bundle.preset).grid?.position ?? effectiveGridSettings().position;
        const nextImported: ImportedFluidScene = {
            asset,
            assetRoot,
            assetRootPosition: [assetRoot.position.x, assetRoot.position.y, assetRoot.position.z],
            assetRootScaling: [assetRoot.scaling.x, assetRoot.scaling.y, assetRoot.scaling.z],
            referenceGridPosition: [...referenceGridPosition],
            gridOffset: [0, 0, 0],
            ...collision,
            groundWasVisible,
            bundle,
            sourceBindings: [],
            animatedCollisionBindings: [],
            collisionDebugMeshes: new Map(),
            collisionOffsets: new Map(),
        };
        const replacingImported = importedScene !== null;
        let assetAdded = false;
        let primedPairKey: string | null = null;
        let previousPrimedState: PairState | undefined;
        try {
            clearImportedScene(false);
            if (!replacingImported && importedMethod === methodName && currentPairKey !== null) {
                pairStates.set(currentPairKey, readLivePairState(methodName));
            }
            if (importedMethod !== methodName || activeBackendId() !== bundle.preset.backendId) {
                const incoming = importFluidPresetSession(bundle.preset, defaultPairState(targetDemo, importedMethod, bundle.preset.material ?? pbmpmMaterial)).state;
                primedPairKey = pairKeyFor(targetDemo, importedMethod, quality, incoming.material ?? pbmpmMaterial, bundle.preset.backendId);
                previousPrimedState = pairStates.get(primedPairKey);
                pairStates.set(primedPairKey, incoming);
                importedInitialState = bundle.initialState;
                importedCollisionActive = true;
                suppressPairSnapshot = replacingImported;
                try {
                    switchPair(activeDemo!, importedMethod, quality, pbmpmMaterial, false, bundle.preset.backendId ?? "production");
                } finally {
                    suppressPairSnapshot = false;
                }
            }
            suspendHostScenePresentation();
            applyTransientSceneScale(nextImported);
            assetAdded = true;
            addToScene(scene, asset);
            for (const group of asset.animationGroups ?? []) {
                group.loopAnimation = true;
                playAnimation(group);
            }
            nextImported.animatedCollisionBindings = createImportedAnimatedCollisionBindings(asset, bundle);
            updateImportedCollisionTransforms(nextImported, 0, true);
            importedScene = nextImported;
            importedInitialState = bundle.initialState;
            if (bundle.initialState) {
                canvas.dataset.importedInitialStateCount = String(bundle.initialState.positions.length / 3);
                canvas.dataset.importedInitialStateFrame = String(bundle.initialState.frame);
            } else {
                delete canvas.dataset.importedInitialStateCount;
                delete canvas.dataset.importedInitialStateFrame;
            }
            refreshImportedCollisionControls(nextImported);
            syncExternalSceneExportChoice(true);
            importedCollisionActive = true;
            setMeshVisible(ground, false);
            canvas.dataset.importedHostGroundHidden = "true";
            canvas.dataset.importedBundle = "true";
            canvas.dataset.importedMeshCount = String(getContainerMeshes(asset).length);
            canvas.dataset.importedLightCount = String(asset.entities.filter((entity) => "lightType" in entity).length);
            canvas.dataset.importedCollisionDims = bundle.collision.dims.join(",");
            canvas.dataset.importedCollisionCellSize = String(bundle.collision.cellSize);
            canvas.dataset.importedAnimationCount = String(asset.animationGroups?.length ?? 0);
            canvas.dataset.importedPlayingAnimationCount = String(asset.animationGroups?.filter((group) => group.isPlaying).length ?? 0);
            applyImportedPreset(bundle.preset, false);
            if (targetDemo.key === "whiteboard") {
                const importedCamera = bundle.preset.camera;
                if (!importedCamera?.target) {
                    const authoredOrbit = importedCamera ? { alpha: cam.alpha, beta: cam.beta, radius: cam.radius } : null;
                    frameImportedWhiteboardScene();
                    if (authoredOrbit) {
                        cam.alpha = authoredOrbit.alpha;
                        cam.beta = authoredOrbit.beta;
                        cam.radius = authoredOrbit.radius;
                    }
                }
            }
            nextImported.sourceBindings = createImportedEmitterSourceBindings(asset);
            updateImportedSceneBindings(nextImported, 0, true);
            const animationHook = asset._beforeRenderHook;
            if (animationHook) {
                const animationIndex = scene._beforeRender.indexOf(animationHook);
                if (animationIndex >= 0) {
                    scene._beforeRender.splice(animationIndex, 1);
                }
            }
            canvas.dataset.importedBoundEmitterCount = String(nextImported.sourceBindings.length);
            resetActiveFlow(false);
            canvas.dataset.importedFrameClearing = String(sceneTask._config.clr && (!msaaOn || msaaSceneTask?._config.clr === true));
        } catch (error) {
            if (primedPairKey) {
                if (previousPrimedState) {
                    pairStates.set(primedPairKey, previousPrimedState);
                } else {
                    pairStates.delete(primedPairKey);
                }
            }
            if (importedScene === nextImported) {
                clearImportedScene(true);
            } else {
                if (assetAdded) {
                    removeFromScene(scene, asset);
                }
                disposeFluidSceneSdf(nextImported.sceneSdf);
                restoreHostScenePresentation(true);
            }
            throw error;
        }
    }
    const importInput = document.createElement("input");
    importInput.type = "file";
    importInput.accept = ".json,.glb,.sdf,application/json,model/gltf-binary,application/octet-stream";
    importInput.multiple = true;
    importInput.hidden = true;
    const importBtn = document.createElement("button");
    importBtn.textContent = "Import fluid files";
    importBtn.style.cssText = exportBtn.style.cssText;
    importBtn.onclick = () => importInput.click();
    const completePresetImport = (): void => {
        if (offlineMode) {
            const values = controls.getValues();
            const diagnostics = getFluidSimulationDiagnostics(activeSim);
            const grid = effectiveGridSettings();
            const flipDiscretization = methodName === "FLIP" ? flipDiscretizationForGrid(grid, flipGridResolution) : undefined;
            const gridCellSize = flipDiscretization?.dx ?? fluidSimulationCellSize(methodName, fluidDiscretization(physicsScale));
            const gridCells = flipDiscretization?.gridDim ?? gridCellsForSize(grid.size, gridCellSize);
            const pageDiagnostics = controlsBinding?.pageDiagnostics;
            canvas.dataset.offlineSummary = JSON.stringify({
                method: activeSim.options.backend?.name ?? diagnostics.method,
                gridResolution: flipDiscretization?.gridResolution ?? gridResolutionForScale(methodName, physicsScale, Math.max(...grid.size)),
                gridCells,
                gridSize: grid.size,
                gridCellSize,
                markersPerCell: methodName === "FLIP" ? flipMarkersPerCell : null,
                activeParticles: diagnostics.activeCount,
                particleCapacity: diagnostics.count,
                simulationGpuBytes: diagnostics.gpuBytes,
                deviceBufferLimitBytes: particleBufferLimit,
                deviceParticleCapacity: deviceParticleCapacityForMethod(methodName),
                deviceParticleBytesPerSlot: fluidParticleBytesPerSlot(methodName),
                pagedGrid: (methodName === "FLIP" || methodName === "MLS-MPM") && values.pagedGrid,
                residentPages: pageDiagnostics?.requiredPages ?? null,
                pageCapacity: values.pagedGridMaxPages,
                pressureSolver: referenceSelected() ? "pcg" : methodName === "FLIP" ? ((values.schema.pressureSolver ?? 0) >= 0.5 ? "multigrid" : "jacobi") : null,
                renderMode: values.renderMode,
                polygonSurface: diagnostics.polygon !== null,
                foamEnabled: values.foam.enabled,
                foamCapacity: diagnostics.diffuse?.capacity ?? 0,
            });
        }
        canvas.dataset.presetImportStatus = "ready";
    };
    importInput.onchange = async () => {
        const files = [...(importInput.files ?? [])];
        const file = files.find((candidate) => candidate.name.toLowerCase().endsWith(".json")) ?? files[0];
        importInput.value = "";
        const targetDemo = activeDemo;
        if (!file || !targetDemo) {
            return;
        }
        canvas.dataset.presetImportStatus = "loading";
        delete canvas.dataset.presetImportError;
        const generation = ++importGeneration;
        const importFiles = async (): Promise<void> => {
            try {
                if (generation !== importGeneration || activeDemo !== targetDemo) {
                    return;
                }
                const contents = await file.text();
                if (generation !== importGeneration || activeDemo !== targetDemo) {
                    return;
                }
                const parsed = JSON.parse(contents) as Partial<FluidExportJson>;
                if (parsed.backendId === "flip-reference") {
                    await ensureReferenceBackend();
                    if (generation !== importGeneration || activeDemo !== targetDemo) {
                        return;
                    }
                }
                if (parsed.scene) {
                    const externalResources = new Map<string, ArrayBuffer>();
                    if (parsed.scene.encoding === "external") {
                        await Promise.all(
                            files
                                .filter((candidate) => candidate !== file)
                                .map(async (candidate) => {
                                    externalResources.set(candidate.name.replaceAll("\\", "/"), await candidate.arrayBuffer());
                                })
                        );
                    }
                    const bundle = parseBlenderFluidJson(contents, externalResources);
                    await installImportedBundle(bundle, generation, targetDemo);
                    if (generation === importGeneration && activeDemo === targetDemo) {
                        completePresetImport();
                    }
                    return;
                }
                if (
                    referenceSelected() &&
                    Array.isArray(parsed.emitters) &&
                    !parsed.render &&
                    (parsed.emitters.some((emitter) => emitter.enabled && emitter.behavior !== "initial") || parsed.sinks?.some((sink) => sink.enabled))
                ) {
                    throw new Error("[FLIP Reference] flow imports must contain only initial emitters and no enabled sinks.");
                }
                clearImportedScene(true);
                if (Array.isArray(parsed.emitters) && !parsed.render) {
                    const importedFlow = {
                        emitters: structuredClone(parsed.emitters),
                        sinks: structuredClone(parsed.sinks ?? []).map((sink) => ({
                            ...sink,
                            mode: sink.mode ?? ((parsed.formatVersion ?? 0) <= 6 ? "recycle" : "delete"),
                        })),
                    };
                    activeFlow = (parsed.formatVersion ?? 0) < 4 ? flowToGridLocal(importedFlow) : importedFlow;
                    resetActiveFlow(false);
                    refreshFlowUI();
                    completePresetImport();
                    return;
                }
                applyImportedPreset(parsed as FluidExportJson, true);
                restartOrStabilizeActiveDemo();
                completePresetImport();
            } catch (error) {
                if (generation === importGeneration && activeDemo === targetDemo) {
                    canvas.dataset.presetImportStatus = "error";
                    canvas.dataset.presetImportError = error instanceof Error ? error.message : String(error);
                    console.error("[fluid] failed to import preset", error);
                }
            }
        };
        if (referencePump) {
            await queueFluidReferenceChange(referencePump, importFiles);
        } else {
            await importFiles();
        }
    };
    controls.root.append(...controls.makeSection("Presets", [importBtn, importInput, externalSceneExportRow, exportBtn]));

    // Mount the shared panel (right side) + the GPU-timing panel (top-left). The
    // `canvas.dataset.timing` flag lets tests read whether per-stage timing is active.
    document.body.appendChild(controls.root);
    // F8 hides/shows every overlay so the demo can be looked at (or captured) unobstructed:
    // the control panel, the GPU-timing panel and the page's own key hint. Collected lazily
    // because the GPU panel only exists when timestamp queries are available.
    let uiHidden = false;
    const uiOverlays = (): HTMLElement[] => {
        const list: HTMLElement[] = [controls.root];
        if (controls.gpu) {
            list.push(controls.gpu.panel);
        }
        const hintEl = document.querySelector<HTMLElement>(".hint");
        if (hintEl) {
            list.push(hintEl);
        }
        return list;
    };
    const setUiHidden = (hidden: boolean): void => {
        uiHidden = hidden;
        for (const el of uiOverlays()) {
            el.style.display = hidden ? "none" : "";
        }
    };
    if (controls.gpu) {
        document.body.appendChild(controls.gpu.panel);
    }
    canvas.dataset.timing = profiler ? "on" : "unavailable";

    // ── Generic per-demo parameter UI ────────────────────────────────────
    // A demo exposes a typed parameter list; render a control per type
    // (number → slider, boolean → checkbox, color → picker) and call the demo's
    // handler on change.
    function buildDemoParamsUI(host: HTMLElement, params: DemoParam[], onChange: (k: string, v: number | boolean | string) => void): void {
        host.replaceChildren();
        for (const p of params) {
            if (p.hidden) {
                continue; // still snapshotted + preset-driven, just not exposed as a control
            }
            if (p.type === "number") {
                const row = document.createElement("div");
                row.style.cssText = "margin:6px 0;";
                const head = document.createElement("div");
                head.style.cssText = "display:flex;justify-content:space-between;";
                const lab = document.createElement("span");
                lab.textContent = p.label;
                const val = document.createElement("span");
                val.style.cssText = "color:#9fb4cc;";
                val.textContent = String(p.value);
                head.append(lab, val);
                const input = document.createElement("input");
                input.type = "range";
                input.min = String(p.min);
                input.max = String(p.max);
                input.step = String(p.step);
                input.value = String(p.value);
                input.style.cssText = "width:100%;";
                const applyInput = applyLatestOnAnimationFrame((value: number) => onChange(p.key, value));
                input.oninput = () => {
                    const v = parseFloat(input.value);
                    val.textContent = String(v);
                    applyInput(v);
                };
                row.append(head, input);
                host.appendChild(row);
            } else if (p.type === "boolean") {
                const row = document.createElement("label");
                row.style.cssText = "display:flex;align-items:center;gap:6px;margin:6px 0;cursor:pointer;";
                const chk = document.createElement("input");
                chk.type = "checkbox";
                chk.checked = p.value;
                const txt = document.createElement("span");
                txt.textContent = p.label;
                chk.onchange = () => onChange(p.key, chk.checked);
                row.append(chk, txt);
                host.appendChild(row);
            } else {
                const row = document.createElement("label");
                row.style.cssText = "display:flex;align-items:center;gap:8px;margin:6px 0;cursor:pointer;";
                const lab = document.createElement("span");
                lab.textContent = p.label;
                const input = document.createElement("input");
                input.type = "color";
                input.value = p.value;
                input.style.cssText = "width:36px;height:22px;padding:0;border:1px solid #33415a;border-radius:4px;background:#1a2230;cursor:pointer;";
                const applyInput = applyLatestOnAnimationFrame((value: string) => onChange(p.key, value));
                input.oninput = () => applyInput(input.value);
                row.append(lab, input);
                host.appendChild(row);
            }
        }
    }
    let demoParamsGeneration = 0;
    function refreshDemoParams(): void {
        const generation = ++demoParamsGeneration;
        const demo = activeDemo!;
        const params = demo.demoParams();
        const extras = demo.extraControls();
        if (!params.some((p) => !p.hidden) && extras.length === 0) {
            const none = document.createElement("div");
            none.textContent = "No tunable parameters for this demo.";
            none.style.cssText = "color:#7c8aa0;font-size:12px;margin:2px 0;";
            demoParamsHost.replaceChildren(none);
            return;
        }
        buildDemoParamsUI(demoParamsHost, params, (k, v) => {
            if (generation === demoParamsGeneration && activeDemo === demo) {
                demo.applyParam(k, v);
            }
        });
        for (const el of extras) {
            demoParamsHost.appendChild(el);
        }
    }

    let showEmitterWireframe = false;
    let showSinkWireframe = false;
    let flowGizmoOwner: FluidFlowObjectKind | null = null;

    type FlowWireframeSegment = readonly [Vec3, Vec3];
    const FLOW_WIREFRAME_SEGMENTS = MAX_FLUID_POLYGON_POINTS * 3;
    const flowPoint = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
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
    const initialEffectiveGrid = effectiveGridSettings();
    const gridBoundsWireframe = createLineSystem(engine, {
        name: "fluid-grid-bounds",
        lines: gridBoundsSegments(initialEffectiveGrid.size),
        material: createLineMaterial({
            name: "fluid-grid-bounds-material",
            color: { r: 0.45, g: 1, b: 0.35, a: 0.85 },
            useVertexAlpha: true,
            depthWrite: false,
            depthCompare: "always",
        }),
    });
    gridBoundsWireframe.pickable = false;
    gridBoundsWireframe.renderOrder = 9_999;
    addToScene(scene, gridBoundsWireframe);
    setMeshVisible(gridBoundsWireframe, false);
    const gridBoundsSolid = createSolidGridBounds(engine, "fluid-grid-bounds-solid");
    for (const face of gridBoundsSolid) {
        addMeshToTask(overlayTask, face);
        setMeshVisible(face, false);
    }
    const syncGridBoundsWireframe = (): void => {
        const grid = effectiveGridSettings();
        updateLineSystem(engine, gridBoundsWireframe, { lines: gridBoundsSegments(grid.size) });
        gridBoundsWireframe.position.set(grid.position[0], grid.position[1], grid.position[2]);
        gridBoundsWireframe.scaling.set(1, 1, 1);
        for (const face of gridBoundsSolid) {
            face.position.set(grid.position[0], grid.position[1], grid.position[2]);
            face.scaling.set(grid.size[0], grid.size[1], grid.size[2]);
            setMeshVisible(face, showGridBounds && showGridBoundsSolid);
        }
        setMeshVisible(gridBoundsWireframe, showGridGizmo || (showGridBounds && !showGridBoundsSolid));
    };
    const appendPolyline = (segments: FlowWireframeSegment[], points: readonly Vec3[], closed = true): void => {
        for (let i = 1; i < points.length; i++) {
            segments.push([points[i - 1]!, points[i]!]);
        }
        if (closed && points.length > 2) {
            segments.push([points[points.length - 1]!, points[0]!]);
        }
    };
    const appendRing = (segments: FlowWireframeSegment[], plane: "xy" | "xz" | "yz", radius: number, center: Vec3 = flowPoint(0, 0, 0), steps = 32): void => {
        const points: Vec3[] = [];
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
            const meridian: Vec3[] = [];
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
    const paddedFlowWireframe = (shape?: FluidShape): FlowWireframeSegment[] => {
        const segments = shape ? flowShapeWireframe(shape).slice(0, FLOW_WIREFRAME_SEGMENTS) : [];
        while (segments.length < FLOW_WIREFRAME_SEGMENTS) {
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
    const emitterFlowWireframe = createFlowWireframe("fluid-emitter-wireframe", { r: 0.1, g: 0.9, b: 1, a: 0.9 });
    const sinkFlowWireframe = createFlowWireframe("fluid-sink-wireframe", { r: 1, g: 0.55, b: 0.1, a: 0.9 });
    const selectedFlowObject = (kind: FluidFlowObjectKind): FluidEmitter | FluidSink | undefined => flowEditor?.getSelected(kind);
    const syncFlowWireframe = (kind: FluidFlowObjectKind): void => {
        const isEmitter = kind === "emitter";
        const mesh = isEmitter ? emitterFlowWireframe : sinkFlowWireframe;
        const visible = isEmitter ? showEmitterWireframe : showSinkWireframe;
        const flowObject = selectedFlowObject(kind);
        if (!flowObject) {
            setMeshVisible(mesh, false);
            return;
        }
        updateLineSystem(engine, mesh, { lines: paddedFlowWireframe(flowObject.shape) });
        const { position, rotation, scale } = flowObject.transform;
        const worldPosition = gridLocalToWorld(position, effectiveGridSettings().position);
        mesh.position.set(worldPosition[0], worldPosition[1], worldPosition[2]);
        mesh.rotationQuaternion.set(rotation[0], rotation[1], rotation[2], rotation[3]);
        mesh.scaling.set(scale[0], scale[1], scale[2]);
        setMeshVisible(mesh, visible || flowGizmoOwner === kind);
    };

    const flowGizmoLayer = createUtilityLayer(engine, scene);
    const flowPositionGizmo = createPositionGizmo(engine, flowGizmoLayer, { planarEnabled: true });
    const flowRotationGizmo = createRotationGizmo(engine, flowGizmoLayer);
    const flowScaleGizmo = createScaleGizmo(engine, flowGizmoLayer);
    const gridPositionGizmo = createPositionGizmo(engine, flowGizmoLayer, { planarEnabled: true });
    const gridScaleGizmo = createScaleGizmo(engine, flowGizmoLayer);
    setPositionGizmoLocalCoordinates(flowPositionGizmo, false);
    setRotationGizmoLocalCoordinates(flowRotationGizmo, true);
    setScaleGizmoLocalCoordinates(flowScaleGizmo, true);
    setPositionGizmoLocalCoordinates(gridPositionGizmo, false);
    setScaleGizmoLocalCoordinates(gridScaleGizmo, true);
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
    const setGizmoMeshesVisible = (visible: boolean, gizmos: ReadonlyArray<{ _visibleMeshes: Mesh[] }>): void => {
        for (const gizmo of gizmos) {
            for (const mesh of gizmo._visibleMeshes) {
                setMeshVisible(mesh, visible);
            }
        }
    };
    const syncFlowTransformFromGizmo = (): void => {
        if (!flowGizmoOwner) {
            return;
        }
        const flowObject = selectedFlowObject(flowGizmoOwner);
        if (!flowObject) {
            return;
        }
        const mesh = flowGizmoOwner === "emitter" ? emitterFlowWireframe : sinkFlowWireframe;
        flowObject.transform.position = worldToGridLocal([mesh.position.x, mesh.position.y, mesh.position.z], effectiveGridSettings().position);
        flowObject.transform.rotation = [mesh.rotationQuaternion.x, mesh.rotationQuaternion.y, mesh.rotationQuaternion.z, mesh.rotationQuaternion.w];
        flowObject.transform.scale = [mesh.scaling.x, mesh.scaling.y, mesh.scaling.z];
        updateInstalledFlowObject(flowGizmoOwner, flowObject);
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
        const flowObject = owner ? selectedFlowObject(owner) : undefined;
        const target = owner && flowObject ? (owner === "emitter" ? emitterFlowWireframe : sinkFlowWireframe) : null;
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
    const finishGridGizmoDrag = (): void => {
        const base = effectiveGridSettings();
        const next: FluidGridSettings = {
            position: [
                Math.round(gridBoundsWireframe.position.x * 10) / 10,
                Math.round(gridBoundsWireframe.position.y * 10) / 10,
                Math.round(gridBoundsWireframe.position.z * 10) / 10,
            ],
            size: [
                Math.max(0.1, Math.round(Math.abs(base.size[0] * gridBoundsWireframe.scaling.x) * 10) / 10),
                Math.max(0.1, Math.round(Math.abs(base.size[1] * gridBoundsWireframe.scaling.y) * 10) / 10),
                Math.max(0.1, Math.round(Math.abs(base.size[2] * gridBoundsWireframe.scaling.z) * 10) / 10),
            ],
        };
        const error = setGridSettings(next);
        if (typeof error === "string") {
            controls.setGridStatus(error);
            syncGridBoundsWireframe();
        }
        syncGridGizmo();
    };
    for (const gizmo of gridPositionSubGizmos) {
        gizmo.drag.onDragEnd.add(finishGridGizmoDrag);
    }
    for (const gizmo of gridScaleSubGizmos) {
        gizmo.drag.onDragEnd.add(finishGridGizmoDrag);
    }
    const syncGridGizmo = (): void => {
        attachPositionGizmoToNode(gridPositionGizmo, showGridGizmo ? gridBoundsWireframe : null);
        attachScaleGizmoToNode(gridScaleGizmo, showGridGizmo ? gridBoundsWireframe : null);
        setGizmoMeshesVisible(showGridGizmo, gridPositionSubGizmos);
        setGizmoMeshesVisible(showGridGizmo, gridScaleSubGizmos);
    };
    function setGridGizmoVisible(visible: boolean): void {
        showGridGizmo = visible;
        canvas.dataset.gridGizmo = visible ? "true" : "false";
        syncGridBoundsWireframe();
        syncGridGizmo();
    }
    syncGridBoundsWireframe();
    syncGridGizmo();

    flowEditor = createFluidFlowEditor({
        emittersHost: emitterFlowHost,
        sinksHost: sinkFlowHost,
        flow: activeFlow,
        onChange: (_flow, change) => {
            if (change.type === "flow") {
                applyFlow();
            } else if (change.type === "object") {
                updateInstalledFlowObject(change.kind, change.object);
            }
        },
        onRefresh: () => {
            syncFlowWireframe("emitter");
            syncFlowWireframe("sink");
            syncFlowGizmo();
            refreshParticleUsageStatus();
            controls.setGridStatus(flipPendingCapacityStatus());
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
        getEmitterRateMode: () => (methodName === "FLIP" ? "occupancy-refill" : "unlimited-toggle"),
        getInitialEmitterParticleCount: (emitter) => {
            if (methodName !== "FLIP" || emitter.behavior !== "initial") {
                return undefined;
            }
            if (referenceSelected() && importedInitialState && !activeSim.initialEmitterParticleCounts.has(emitter.id)) {
                return undefined;
            }
            return flipInitialEmitterParticleCounts().get(emitter.id) ?? 0;
        },
        onInitialEmitterParticleCountDisplayed: (count) => {
            if (count === undefined) {
                delete canvas.dataset.selectedInitialEmitterParticleCount;
            } else {
                canvas.dataset.selectedInitialEmitterParticleCount = String(count);
            }
        },
    });
    function refreshFlowUI(): void {
        flowEditor?.setFlow(activeFlow);
    }
    function applyParam(sim: FluidSimulation, key: string, value: number): void {
        if (sim.options.backend && !sim.options.backend.physicsParameters.includes(key)) {
            return;
        }
        setFluidSimulationParameter(sim, key, value);
        if (key === "gravity" && sim === activeSim) {
            canvas.dataset.gravity = String(value);
        }
        if (key === "pressureSolver" && sim === activeSim) {
            canvas.dataset.pressureSolver = value >= 0.5 ? "multigrid" : "jacobi";
        }
        if (
            sim === activeSim &&
            (key === "pressureDiagnostics" ||
                key === "liquidSdf" ||
                key === "ghostFluid" ||
                key === "fractionalSolids" ||
                key === "movingSolidBoundaries" ||
                key === "reseedParticles" ||
                key === "particleSheeting" ||
                key === "polygonSurface")
        ) {
            canvas.dataset[key] = value >= 0.5 ? "true" : "false";
        }
    }

    function applyMethod(name: string, resetSimulation = true): void {
        // activeSim already holds the backend for `name`: rebuildSims constructs it before
        // calling here, and the no-rebuild path in loadPairState only re-enters with the
        // current method. There is no dormant per-method solver to swap in.
        methodName = name;
        syncDeviceParticleCapacity(name);
        if (name === "PB-MPM") {
            setFluidSimulationMaterial(activeSim, pbmpmMaterial);
        }
        // Sand renders as uniformly-coloured grains (no velocity brightening); water/jelly keep the
        // speed-based highlight. This is a render characteristic derived from the material — NOT forced
        // here on the colour or render MODE (those come from the per-material preset / pair state, so a
        // count-change re-applyMethod doesn't reset a user's colour).
        configureFluidSimulationRenderLayer(particleTask, {
            particleVelocityBrighten: name === "PB-MPM" && pbmpmMaterial === PBMPM_SAND_MATERIAL ? 0 : 1,
        });
        for (const [key, value] of Object.entries(controls.getPhysicsValues(name))) {
            applyParam(activeSim, key, value);
        }
        applyFlow();
        if (resetSimulation) {
            resetFluidSimulation(activeSim);
        }
        builtInitialStateKey = flipParticlePlan(requestedParticleCount()).initialStateKey;
        if (resetSimulation) {
            restartSimulationLifecycle();
            clearSceneHoles();
        }
        renderPolygonSurface = name === "FLIP" && !referenceSelected() && (controls.getPhysicsValues("FLIP").polygonSurface ?? 0) >= 0.5;
        applyEffectiveRenderMode();
        applyFoam(); // rebind the foam renderer + (re)enable foam on the new active sim
        applyProfiler(); // re-wire the GPU timing hook onto the rebuilt sims (tasks persist)
        controls.setMethod(name); // sync the method dropdown + component's current method (no side effect)
        controls.rebuildPhysics(name); // rebuild the physics-slider block for the new method
        // Show only the physics sliders the current method/material uses (PB-MPM branches on material).
        updateBackendControls();
        controls.setMaterial(pbmpmMaterial);
        canvas.dataset.method = methodName;
        canvas.dataset.activeParticleCount = String(activeSim.activeCount ?? activeSim.count);
        if (referenceSelected()) {
            canvas.dataset.pressureSolver = "pcg";
        }
        const pageCapacityLimit =
            name === "FLIP" ? maxFlipPagedGridPagesFor(effectiveGridSettings(), flipGridResolution) : maxMlsPagedGridPagesFor(effectiveGridSettings(), physicsScale);
        if (name === "FLIP") {
            flipPagedGridMaxPages = Math.min(flipPagedGridMaxPages, pageCapacityLimit);
        } else {
            mpmPagedGridMaxPages = Math.min(mpmPagedGridMaxPages, pageCapacityLimit);
        }
        const currentPagedGrid = name === "FLIP" ? flipPagedGrid : mpmPagedGrid;
        const currentPageCapacity = name === "FLIP" ? flipPagedGridMaxPages : mpmPagedGridMaxPages;
        controls.setPagedGrid(currentPagedGrid);
        controls.setPagedGridMaxPages(currentPageCapacity, pageCapacityLimit);
        controls.setPagedGridStatus("");
        canvas.dataset.activeBlocks = mpmActiveBlocks ? "true" : "false";
        canvas.dataset.pagedGrid = currentPagedGrid ? "true" : "false";
        canvas.dataset.pagedGridMaxPages = String(currentPageCapacity);
        canvas.dataset.fusedBlockDiscovery = mpmFusedBlockDiscovery ? "true" : "false";
        refreshFlowUI();
    }
    // Toggle between the sphere-impostor renderer and the screen-space surface.
    // Default is the fluid surface; the checkbox switches to spheres.
    //
    // "Render as spheres" and "Anisotropic surface" are independent toggles whose COMBINATION
    // selects what gets drawn, so the effective state is resolved in ONE place (called from
    // both applyRenderMode and onAnisotropic) to keep the sphere task and surface mode in sync.
    let renderSpheres = false;
    let renderAnisotropic = false;
    let renderPolygonSurface = false;
    function applyEffectiveRenderMode(): void {
        const mode = resolveFluidRenderMode({
            method: methodName as FluidSimulationOptions["method"],
            renderSpheres,
            anisotropicSurface: renderAnisotropic,
            polygonSurface: !referenceSelected() && renderPolygonSurface,
            foamEnabled: controls.getValues().foam.enabled,
            surfaceDebugActive,
        });
        configureFluidSimulationRenderLayer(particleTask, { enabled: mode.particleEnabled });
        configureFluidSimulationRenderLayer(surfaceTask, { surfaceMode: mode.surfaceMode });
        configureFluidSimulationRenderLayer(polygonSurfaceTask, { enabled: mode.polygonEnabled });
        configureFluidSimulationRenderLayer(foamTask, {
            enabled: simulationOpacity > 0 && mode.foamEnabled,
            foam: { polygonSurfaceDepth: mode.foamPolygonSurfaceDepth },
        });
        canvas.dataset.render = mode.diagnosticMode;
    }
    function applyRenderMode(spheres: boolean): void {
        renderSpheres = spheres;
        applyEffectiveRenderMode();
    }

    // Resize the particle buffers by disposing and rebuilding both backends at
    // the new count, then re-applying the current demo and method (which re-seeds
    // and rebinds the renderer).
    function setParticleCount(n: number): void {
        if (n === requestedParticleCount()) {
            return;
        }
        rebuildSims(n, physicsScale);
    }

    function syncGridControls(): void {
        const effectiveGrid = gridSettings ?? defaultGridSettings(methodName, domainScale);
        const flipDiscretization = methodName === "FLIP" ? flipDiscretizationForGrid(effectiveGrid, flipGridResolution) : undefined;
        const cellSize =
            flipDiscretization?.dx ??
            (gridSettings ? fluidSimulationCellSize(methodName, fluidDiscretization(physicsScale)) : cellSizeForPhysicsScale(methodName, physicsScale) * domainScale);
        const cells = flipDiscretization?.gridDim ?? gridCellsForSize(effectiveGrid.size, cellSize);
        const gridResolution = flipDiscretization?.gridResolution ?? gridResolutionForScale(methodName, physicsScale, Math.max(...effectiveGrid.size));
        controls.setGridSettings([...effectiveGrid.position], [...effectiveGrid.size], cellSize);
        controls.setGridResolution(gridResolution);
        controls.setMarkersPerCell(flipMarkersPerCell);
        controls.setShowGridBounds(showGridBounds);
        controls.setShowGridBoundsSolid(showGridBoundsSolid);
        canvas.dataset.gridPosition = effectiveGrid.position.join(",");
        canvas.dataset.gridSize = effectiveGrid.size.join(",");
        canvas.dataset.gridCells = cells.join(",");
        canvas.dataset.gridCellSize = String(cellSize);
        canvas.dataset.gridResolution = String(gridResolution);
        canvas.dataset.flipMarkersPerCell = String(flipMarkersPerCell);
        canvas.dataset.gridExplicit = String(gridSettings !== undefined);
        if (methodName === "FLIP") {
            delete canvas.dataset.physicsParticleSize;
        } else {
            canvas.dataset.physicsParticleSize = String(physicsScale);
        }
        canvas.dataset.showGridBounds = String(showGridBounds);
        canvas.dataset.showGridBoundsSolid = String(showGridBoundsSolid);
    }

    function setPhysicsScale(s: number): void {
        if (methodName === "FLIP") {
            return;
        }
        const [minScale, maxScale] = scaleLimitsForMethod(methodName);
        const nextScale = Math.min(maxScale, Math.max(minScale, Math.round(s * 100) / 100));
        if (nextScale === physicsScale) {
            return;
        }
        if (gridSettings) {
            const allocationError = gridAllocationError(gridSettings, methodName, nextScale);
            if (allocationError) {
                controls.setPhysScale(physicsScale);
                controls.setGridStatus(allocationError);
                return;
            }
        }
        rebuildSims(requestedParticleCount(), nextScale);
    }

    function setGridResolution(resolution: number): void {
        const nextResolution = Math.max(GRID_RESOLUTION_MIN, Math.round(resolution));
        if (nextResolution === flipGridResolution) {
            return;
        }
        flipGridResolution = nextResolution;
        syncGridControls();
        refreshParticleUsageStatus();
        controls.setGridStatus(flipPendingCapacityStatus());
    }

    function setMarkersPerCell(value: number): void {
        const next = Math.max(1, Math.min(64, Math.round(value)));
        if (next === flipMarkersPerCell) {
            return;
        }
        flipMarkersPerCell = next;
        syncGridControls();
        refreshParticleUsageStatus();
        controls.setGridStatus(flipPendingCapacityStatus());
    }

    function setFlipParticleCapacity(value: number): void {
        const next = Math.max(1, Math.min(deviceParticleCapacityForMethod("FLIP"), Math.round(value)));
        if (next === flipParticleCapacityRequest) {
            controls.setFlipParticleCapacity(next);
            return;
        }
        flipParticleCapacityRequest = next;
        controls.setFlipParticleCapacity(next);
        refreshParticleUsageStatus();
        controls.setGridStatus(flipPendingCapacityStatus());
    }

    function setGridSettings(next: FluidGridSettings): string | void {
        if (!validGridSettings(next)) {
            return "Grid position must be finite and Grid size must contain positive finite world-space dimensions.";
        }
        if (controlsBinding) {
            try {
                const bindingPlan = applyFluidGridSettings(controlsBinding, next.position, next.size);
                finishFluidControlPlan(bindingPlan, ["gridPosition", "gridSize"]);
                syncGridControls();
                syncGridBoundsWireframe();
                syncGridGizmo();
                refreshParticleUsageStatus();
                return bindingPlan.restartRequired ? flipPendingCapacityStatus() || undefined : undefined;
            } catch (error) {
                syncGridControls();
                syncGridBoundsWireframe();
                syncGridGizmo();
                return error instanceof Error ? error.message : String(error);
            }
        }
        if (methodName !== "FLIP") {
            const allocationError = gridAllocationError(next, methodName, physicsScale);
            if (allocationError) {
                return allocationError;
            }
        }
        if (gridSettingsEqual(next, gridSettings)) {
            syncGridControls();
            return;
        }
        gridSettings = cloneGridSettings(next);
        if (methodName !== "FLIP") {
            rebuildSims(particleCount, physicsScale);
            return;
        }
        syncGridControls();
        syncGridBoundsWireframe();
        syncGridGizmo();
        refreshParticleUsageStatus();
        return flipPendingCapacityStatus() || undefined;
    }

    // Rebuild the ACTIVE backend at a new particle count, physics size and grid. Only the
    // selected solver is (re)constructed and only the outgoing selected solver is disposed,
    // so a capacity / grid / paging change never allocates an unrelated backend. The
    // The outgoing simulation stays live until the replacement has validated and allocated.
    // After the render tasks are retargeted, its facade-owned resources retire behind the
    // next submission fence. FLIP can additionally copy compatible state before retirement.
    function rebuildSims(count: number, scale: number, preserveFlipState = false): void {
        if (stagingBackendSwitch || deferReferenceMutation(() => rebuildSims(count, scale, preserveFlipState))) {
            return;
        }
        const previousParticleCount = particleCount;
        const previousPhysicsScale = physicsScale;
        const previousCapacityRequest = flipParticleCapacityRequest;
        if (methodName === "FLIP") {
            flipParticleCapacityRequest = Math.max(1, Math.round(count));
        }
        particleCount = flipParticleCapacity(count);
        physicsScale = scale;
        let prepared: ReturnType<typeof prepareFluidReconfigurationUpdate> | null = null;
        try {
            const base = simulationOptions(particleCount, scale);
            const physics = { ...(base.physics ?? {}) };
            for (const [key, value] of Object.entries(selectedPhysics(controls.getPhysicsValues(methodName)))) {
                physics[key] = value;
            }
            const foamValues = controls.getValues().foam;
            prepared = prepareFluidReconfigurationUpdate(
                activeSim,
                {
                    ...base,
                    physics,
                    flow: flowToWorld(activeFlow),
                    foam: base.backend?.supportsFoam !== false && foamValues.enabled ? currentFoamConfig() : null,
                    sceneSdf: currentSceneSdf(),
                    forceField: base.backend?.supportsForces === false ? null : activeSim.forceField,
                    profiler: currentGpuProfiler(),
                },
                preserveFlipState && !base.backend && !activeSim.options.backend
            );
            lastTransitionPeakBytes = prepared.transitionPeakBytes;
            commitFluidReconfiguration(prepared);
        } catch (error) {
            if (prepared) {
                cancelFluidReconfiguration(prepared);
            }
            particleCount = previousParticleCount;
            physicsScale = previousPhysicsScale;
            flipParticleCapacityRequest = previousCapacityRequest;
            throw error;
        }
        trackSimBuild(methodName);
        builtImportedInitialState = importedInitialState;
        canvas.dataset.simulationTransitionBytes = String(lastTransitionPeakBytes);
        builtGridSettings = gridSettings ? cloneGridSettings(gridSettings) : undefined;
        builtGridMethod = methodName;
        builtPhysicsScale = scale;
        builtFlipGridResolution = flipGridResolution;
        builtFlipMarkersPerCell = flipMarkersPerCell;
        builtWithGridFloor = importedCollisionActive || activeDemo?.useGridFloor === true;
        builtDomainScale = domainScale; // sims are now built at the current domain scale
        applyMethod(methodName, !preserveFlipState);
        syncGridControls();
        syncGridBoundsWireframe();
        syncFlowWireframe("emitter");
        syncFlowWireframe("sink");
        syncFlowGizmo();
        canvas.dataset.particleCount = String(particleCount);
        canvas.dataset.flipParticleCapacityRequest = String(flipParticleCapacityRequest);
        controls.setParticleCount(particleCount); // sync the Particles dropdown (no rebuild re-entry)
        controls.setFlipParticleCapacity(flipParticleCapacityRequest);
        controls.setPhysScale(scale); // sync the physics-size slider + its read-out (no side effect)
    }

    // ── Per-(demo, simulation) parameter state ───────────────────────────────
    // Every tunable UI parameter (physics sliders, demo params, render settings,
    // particle count, physics scale) is stored PER (demo, method) pair, so e.g.
    // SPH-fountain and MLS-fountain keep independent values. On leaving a pair we
    // snapshot the live UI into its slot; on entering we restore it (seeding from a
    // preset or the built-in defaults on first visit).

    // Snapshot the untouched defaults (before any edit) to seed preset-less pairs.
    const SCHEMA_DEFAULTS: Record<string, Record<string, number>> = {};
    for (const m of Object.keys(DEFAULT_FLUID_SCHEMAS)) {
        SCHEMA_DEFAULTS[m] = {};
        for (const p of DEFAULT_FLUID_SCHEMAS[m]!) SCHEMA_DEFAULTS[m]![p.key] = p.value;
    }
    // Snapshot the component's pristine control values (before any interaction) to seed
    // preset-less pairs. physScale/count are the host's committed values (not the live
    // DOM), matching the original per-pair capture.
    const initialValues = controls.getValues();
    const RENDER_DEFAULTS = {
        color: initialValues.color,
        independentRendering: initialValues.independentRendering,
        half: initialValues.half,
        thicknessDownscale: initialValues.thicknessDownscale,
        absorption: initialValues.absorption,
        size: initialValues.size,
        physScale: physicsScale,
        gridResolution: initialValues.gridResolution,
        markersPerCell: initialValues.markersPerCell,
        count: particleCount,
        renderMode: initialValues.renderMode,
        polygonShader: initialValues.polygonShader,
        refraction: initialValues.refraction,
        specular: initialValues.specular,
        reflectionExposure: initialValues.reflectionExposure,
        reflectionContrast: initialValues.reflectionContrast,
        reflectivity: initialValues.reflectivity,
        depthBlur: initialValues.depthBlur,
        depthBlurThreshold: initialValues.depthBlurThreshold,
        thicknessBlur: initialValues.thicknessBlur,
        surfaceFilter: initialValues.surfaceFilter,
        narrowDelta: initialValues.narrowDelta,
        narrowMu: initialValues.narrowMu,
        anisotropic: initialValues.anisotropic,
        anisoSurfScale: initialValues.anisoSurfScale ?? 0.5,
        simulationDuration: initialValues.simulationDuration,
        alphaDecay: initialValues.alphaDecay,
    };
    // Pristine foam look, snapshotted before any interaction; seeds the foam block of
    // every preset-less pair so foam becomes per-(demo, method) (restored on switch).
    const FOAM_DEFAULTS: NonNullable<PairState["foam"]> = { ...initialValues.foam };
    // Per-demo default demo-param bag (numeric params only), snapshotted from each
    // demo's initial descriptors below once the registry is built.
    const DEMO_PARAM_DEFAULTS: Record<string, Record<string, number>> = {};

    const pairStates = new Map<string, PairState>();
    const presetSessions = new Map<string, FluidPresetSession>();
    const methodIndependentStates = new Map<string, PairState>();
    let currentPairKey: string | null = null;

    function defaultPairState(demo: FluidDemo, method: string, material = 0, backendId?: "flip-reference"): PairState {
        const flowPosition = defaultGridSettings(method, demo.getDomainScale?.() ?? 1).position;
        const flow = flowToGridLocal(demo.flow(), flowPosition);
        const sand = method === "PB-MPM" && material === PBMPM_SAND_MATERIAL;
        const state: PairState = {
            simulationSemantics: CURRENT_FLUID_SIMULATION_SEMANTICS,
            schema: { ...SCHEMA_DEFAULTS[method]! },
            demoParams: { ...(DEMO_PARAM_DEFAULTS[demo.key] ?? {}) },
            simulationDuration: RENDER_DEFAULTS.simulationDuration,
            alphaDecay: RENDER_DEFAULTS.alphaDecay,
            simulationTimeScale: 1,
            emitters: flow.emitters,
            sinks: flow.sinks,
            initialEmittersFillCapacity: flow.initialEmittersFillCapacity,
            color: sand ? "#c2b280" : RENDER_DEFAULTS.color,
            independentRendering: RENDER_DEFAULTS.independentRendering,
            half: RENDER_DEFAULTS.half,
            thicknessDownscale: RENDER_DEFAULTS.thicknessDownscale,
            absorption: RENDER_DEFAULTS.absorption,
            size: RENDER_DEFAULTS.size,
            physScale: RENDER_DEFAULTS.physScale,
            gridResolution: method === "FLIP" ? RENDER_DEFAULTS.gridResolution : undefined,
            markersPerCell: method === "FLIP" ? RENDER_DEFAULTS.markersPerCell : undefined,
            showGridBounds: false,
            showGridBoundsSolid: false,
            count: RENDER_DEFAULTS.count,
            material: method === "PB-MPM" ? material : undefined,
            camera: demo.defaultCamera
                ? {
                      alpha: demo.defaultCamera.alpha,
                      beta: demo.defaultCamera.beta,
                      radius: demo.defaultCamera.radius,
                      ...(demo.defaultCamera.target ? { target: [...demo.defaultCamera.target] as [number, number, number] } : {}),
                  }
                : undefined,
            renderMode: sand ? "spheres" : RENDER_DEFAULTS.renderMode,
            polygonShader: RENDER_DEFAULTS.polygonShader,
            refraction: RENDER_DEFAULTS.refraction,
            specular: RENDER_DEFAULTS.specular,
            reflectionExposure: RENDER_DEFAULTS.reflectionExposure,
            reflectionContrast: RENDER_DEFAULTS.reflectionContrast,
            reflectivity: RENDER_DEFAULTS.reflectivity,
            depthBlur: RENDER_DEFAULTS.depthBlur,
            depthBlurThreshold: RENDER_DEFAULTS.depthBlurThreshold,
            thicknessBlur: RENDER_DEFAULTS.thicknessBlur,
            surfaceFilter: RENDER_DEFAULTS.surfaceFilter,
            narrowDelta: RENDER_DEFAULTS.narrowDelta,
            narrowMu: RENDER_DEFAULTS.narrowMu,
            anisotropic: RENDER_DEFAULTS.anisotropic,
            anisoSurfScale: RENDER_DEFAULTS.anisoSurfScale,
            activeBlocks: method === "MLS-MPM" ? false : undefined,
            pagedGrid: method === "MLS-MPM" || method === "FLIP" ? false : undefined,
            pagedGridMaxPages: method === "FLIP" ? FLIP_DEFAULT_PAGE_CAPACITY : method === "MLS-MPM" ? mlsMpmDefaultPageCapacity(RENDER_DEFAULTS.count) : undefined,
            fusedBlockDiscovery: method === "MLS-MPM" ? false : undefined,
            foam: { ...FOAM_DEFAULTS, surfaceFiltering: method === "FLIP" },
            showContainer: true,
        };
        return backendId ? referencePairState(state, demo) : state;
    }
    // First-visit state: the on-disk quality preset for this (demo, method, quality),
    // if any, merged over the core defaults. Pairs with no file use pure defaults.
    function presetOrDefault(demo: FluidDemo, method: string, q: Quality, material = 0, backendId?: "flip-reference"): PairState {
        const base = defaultPairState(demo, method, material, backendId);
        if (backendId || demo.usesQualityPresets === false) {
            return base;
        }
        const p = getQualityPreset(demo.key, method, q, material);
        if (!p) {
            return base;
        }
        const presetGrid = p.grid ? cloneGridSettings(p.grid) : base.grid ? cloneGridSettings(base.grid) : undefined;
        if (presetGrid && !validGridSettings(presetGrid)) {
            throw new Error(`Invalid fluid grid in ${demo.key}/${method}/${q}: position must be finite and size must be positive.`);
        }
        return {
            schema: { ...base.schema, ...(p.schema ?? {}) },
            demoParams: { ...base.demoParams, ...(p.demoParams ?? {}) },
            simulationDuration: p.simulationDuration ?? base.simulationDuration,
            alphaDecay: p.alphaDecay ?? base.alphaDecay,
            simulationTimeScale: p.simulationTimeScale ?? base.simulationTimeScale,
            emitters: structuredClone(p.emitters ?? base.emitters ?? []),
            sinks: structuredClone(p.sinks ?? base.sinks ?? []),
            initialEmittersFillCapacity: p.initialEmittersFillCapacity ?? base.initialEmittersFillCapacity,
            legacyFlow: p.legacyFlow ?? (p.emitters === undefined && p.sinks === undefined),
            color: p.color ?? base.color,
            half: p.half ?? base.half,
            thicknessDownscale: p.thicknessDownscale ?? base.thicknessDownscale,
            absorption: p.absorption ?? base.absorption,
            size: p.size ?? base.size,
            physScale: p.physScale ?? base.physScale,
            gridResolution: p.gridResolution ?? base.gridResolution,
            markersPerCell: p.markersPerCell ?? base.markersPerCell,
            grid: presetGrid,
            showGridBounds: p.showGridBounds ?? base.showGridBounds,
            showGridBoundsSolid: p.showGridBoundsSolid ?? base.showGridBoundsSolid,
            count: p.count ?? base.count,
            // The requested PB-MPM material identifies the pair being loaded. A missing
            // material-specific preset may fall back to liquid look/physics defaults, but
            // its serialized material must not switch the selected pair back to liquid.
            material: method === "PB-MPM" ? base.material : (p.material ?? base.material),
            camera: p.camera ?? base.camera,
            renderMode: p.renderMode ?? base.renderMode,
            polygonShader: p.polygonShader ?? base.polygonShader,
            refraction: p.refraction ?? base.refraction,
            specular: p.specular ?? base.specular,
            reflectionExposure: p.reflectionExposure ?? base.reflectionExposure,
            reflectionContrast: p.reflectionContrast ?? base.reflectionContrast,
            reflectivity: p.reflectivity ?? base.reflectivity,
            depthBlur: p.depthBlur ?? base.depthBlur,
            depthBlurThreshold: p.depthBlurThreshold ?? base.depthBlurThreshold,
            thicknessBlur: p.thicknessBlur ?? base.thicknessBlur,
            surfaceFilter: p.surfaceFilter ?? base.surfaceFilter,
            narrowDelta: p.narrowDelta ?? base.narrowDelta,
            narrowMu: p.narrowMu ?? base.narrowMu,
            anisotropic: p.anisotropic ?? base.anisotropic,
            anisoSurfScale: p.anisoSurfScale ?? base.anisoSurfScale,
            activeBlocks: p.activeBlocks ?? base.activeBlocks,
            pagedGrid: p.pagedGrid ?? base.pagedGrid,
            pagedGridMaxPages: p.pagedGridMaxPages ?? base.pagedGridMaxPages,
            fusedBlockDiscovery: p.fusedBlockDiscovery ?? base.fusedBlockDiscovery,
            foam: p.foam ? { ...base.foam!, ...p.foam } : base.foam,
            demoState: p.demoState ?? base.demoState,
            showContainer: p.showContainer ?? base.showContainer,
            // Core-owned look settings. These are OPTIONAL on both sides — a preset written before
            // they existed omits them, and `base` never sets them — so they must fall through as
            // `undefined` rather than be defaulted here: `loadPairState` reads `undefined` as
            // "leave the live value alone", which is what a legacy file should do. Being absent
            // from this list at all is different, and was the bug: the merge enumerates every key
            // explicitly, so an unlisted one is silently dropped and the preset's value never
            // reached the UI.
            envIntensity: p.envIntensity ?? base.envIntensity,
            msaa: p.msaa ?? base.msaa,
        };
    }
    // Snapshot the current live UI/params for the given method into a PairState.
    function readLivePairState(method: string): PairState {
        const v = controls.getValues();
        const retainedState = currentPairKey ? pairStates.get(currentPairKey) : undefined;
        const transientScale = importedScene && activeDemo?.key === "whiteboard" ? transientSceneMeshScale : 1;
        const snapshotFlow = transientScale === 1 ? activeFlow : scaleFlowTransforms(activeFlow, 1 / transientScale);
        const snapshotGrid =
            gridSettings && importedScene && transientScale !== 1
                ? scaleGridSettingsAroundPivot(gridSettings, 1 / transientScale, importedScene.assetRootPosition)
                : gridSettings
                  ? cloneGridSettings(gridSettings)
                  : undefined;
        const demoParams: Record<string, number> = { ...(retainedState?.demoParams ?? {}) };
        for (const p of activeDemo!.demoParams()) {
            if (p.type === "number") demoParams[p.key] = p.value;
        }
        return {
            simulationSemantics,
            backendId: method === "FLIP" ? activeBackendId() : undefined,
            schema: controls.getPhysicsValues(method),
            demoParams,
            simulationDuration: v.simulationDuration,
            alphaDecay: v.alphaDecay,
            simulationTimeScale,
            emitters: structuredClone(snapshotFlow.emitters),
            sinks: structuredClone(snapshotFlow.sinks),
            initialEmittersFillCapacity: snapshotFlow.initialEmittersFillCapacity,
            color: v.color,
            independentRendering: v.independentRendering,
            half: v.half,
            thicknessDownscale: v.thicknessDownscale,
            absorption: v.absorption,
            size: v.size,
            physScale: method === "FLIP" ? 1 : physicsScale,
            gridResolution: method === "FLIP" ? flipGridResolution : undefined,
            markersPerCell: method === "FLIP" ? flipMarkersPerCell : undefined,
            grid: snapshotGrid,
            showGridBounds,
            showGridBoundsSolid,
            count: method === "FLIP" ? flipParticleCapacityRequest : particleCount,
            material: method === "PB-MPM" ? pbmpmMaterial : undefined,
            camera: {
                alpha: cam.alpha,
                beta: cam.beta,
                radius: cam.radius,
                target: [cam.target.x, cam.target.y, cam.target.z],
                fov: cam.fov,
                ...(cameraMirrorX ? { mirrorX: true } : {}),
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
            activeBlocks: method === "MLS-MPM" ? v.activeBlocks : undefined,
            pagedGrid: method === "MLS-MPM" || method === "FLIP" ? v.pagedGrid : undefined,
            pagedGridMaxPages: method === "MLS-MPM" || method === "FLIP" ? v.pagedGridMaxPages : undefined,
            fusedBlockDiscovery: method === "MLS-MPM" ? v.fusedBlockDiscovery : undefined,
            foam: v.foam,
            demoState: { ...(retainedState?.demoState ?? {}), ...(activeDemo!.snapshotState?.() ?? {}) },
            showContainer: v.showContainer,
            envIntensity,
            msaa: msaaOn,
        };
    }
    // Push a PairState into the live params + UI, then apply it (rebuild the sims
    // when the count/scale differ from what's currently built; otherwise re-apply
    // in place). The component's setters re-render the controls to match; refreshDemoParams
    // re-renders the demo section.
    let loadingPairState = false;
    function loadPairState(st: PairState): void {
        const demo = activeDemo!;
        if (methodName === "FLIP") {
            if (st.backendId && !referenceBackend) {
                throw new Error("[fluid] load FLIP Reference before restoring its preset.");
            }
            flipBackendId = st.backendId;
            if (st.backendId) {
                assertReferenceState(st, demo);
            }
        }
        controls.setCapabilities({});
        simulationSemantics = st.simulationSemantics ?? CURRENT_FLUID_SIMULATION_SEMANTICS;
        const transientScale = importedScene && demo.key === "whiteboard" ? transientSceneMeshScale : 1;
        const nextGridSettings =
            st.grid && importedScene && transientScale !== 1
                ? scaleGridSettingsAroundPivot(st.grid, transientScale, importedScene.assetRootPosition)
                : st.grid
                  ? cloneGridSettings(st.grid)
                  : undefined;
        if (nextGridSettings && !validGridSettings(nextGridSettings)) {
            throw new Error("Invalid fluid grid: position must be finite and size must be positive.");
        }
        const [minScale, maxScale] = scaleLimitsForMethod(methodName);
        const nextGridResolution =
            methodName === "FLIP"
                ? Math.max(GRID_RESOLUTION_MIN, Math.round(st.gridResolution ?? 160))
                : gridResolutionForScale(methodName, st.physScale, Math.max(...(nextGridSettings ?? defaultGridSettings(methodName, domainScale)).size));
        const nextPhysicsScale = methodName === "FLIP" ? 1 : Math.min(maxScale, Math.max(minScale, Math.round(st.physScale * 100) / 100));
        const nextMarkersPerCell = methodName === "FLIP" ? Math.max(1, Math.min(64, Math.round(st.markersPerCell ?? FLIP_DEFAULT_MARKERS_PER_CELL))) : flipMarkersPerCell;
        const markersPerCellChanged = methodName === "FLIP" && nextMarkersPerCell !== flipMarkersPerCell;
        const pagedMethod = methodName === "MLS-MPM" || methodName === "FLIP";
        const nextPagedGrid = pagedMethod ? (st.pagedGrid ?? false) : methodName === "FLIP" ? flipPagedGrid : mpmPagedGrid;
        const nextActiveBlocks = methodName === "MLS-MPM" ? nextPagedGrid || (st.activeBlocks ?? false) : mpmActiveBlocks;
        const targetGrid = nextGridSettings ?? defaultGridSettings(methodName, domainScale);
        const pageCapacityLimit = methodName === "FLIP" ? maxFlipPagedGridPagesFor(targetGrid, nextGridResolution) : maxMlsPagedGridPagesFor(targetGrid, nextPhysicsScale);
        const defaultPageCapacity = methodName === "FLIP" ? FLIP_DEFAULT_PAGE_CAPACITY : mlsMpmDefaultPageCapacity(st.count);
        const nextPagedGridMaxPages = Math.min(
            pageCapacityLimit,
            pagedMethod ? (st.pagedGridMaxPages ?? defaultPageCapacity) : methodName === "FLIP" ? flipPagedGridMaxPages : mpmPagedGridMaxPages
        );
        const nextFusedBlockDiscovery = methodName === "MLS-MPM" ? (st.fusedBlockDiscovery ?? false) : mpmFusedBlockDiscovery;
        if (nextGridSettings) {
            const allocationError = gridAllocationError(
                nextGridSettings,
                methodName,
                nextPhysicsScale,
                {
                    enabled: (methodName === "FLIP" || methodName === "MLS-MPM") && nextPagedGrid,
                    maxPages: nextPagedGridMaxPages,
                },
                nextGridResolution
            );
            if (allocationError) {
                throw new Error(allocationError);
            }
        }
        simulationTimeScale = Math.min(100, Math.max(0.01, st.simulationTimeScale ?? 1));
        canvas.dataset.simulationTimeScale = String(simulationTimeScale);
        if (typeof st.material === "number") {
            pbmpmMaterial = st.material;
        }
        const previousGridSettings = gridSettings ? cloneGridSettings(gridSettings) : undefined;
        loadingPairState = true;
        try {
            // Preserve the target pair's authored grid semantics while the shared binding
            // resolves its allocation. A gridless preset scales both bounds and cell size
            // through domainScale; publishing its displayed bounds as an explicit grid would
            // incorrectly drop that cell-size scale.
            gridSettings = nextGridSettings ? cloneGridSettings(nextGridSettings) : undefined;
            for (const k of Object.keys(st.demoParams)) {
                demo.applyParam(k, st.demoParams[k]!);
            }
            if (st.demoState) {
                demo.restoreState?.(st.demoState);
            }
            demo.commitRestoredParams?.();
            const targetCellSize =
                methodName === "FLIP"
                    ? flipDiscretizationForGrid(targetGrid, nextGridResolution).dx
                    : nextGridSettings
                      ? fluidSimulationCellSize(methodName, fluidDiscretization(nextPhysicsScale))
                      : cellSizeForPhysicsScale(methodName, nextPhysicsScale) * domainScale;
            const restoredFlow = st.legacyFlow
                ? flowToGridLocal(demo.flow(), targetGrid.position)
                : {
                      emitters: structuredClone(st.emitters ?? []),
                      sinks: structuredClone(st.sinks ?? []),
                      ...(st.initialEmittersFillCapacity !== undefined ? { initialEmittersFillCapacity: st.initialEmittersFillCapacity } : {}),
                  };
            activeFlow = transientScale === 1 ? restoredFlow : scaleFlowTransforms(restoredFlow, transientScale);
            stagingBackendSwitch = false;
            // Publish one complete target snapshot. During pair loading, onApply lets the
            // shared binding derive every changed key from this snapshot so host state is
            // committed before resolveTarget evaluates the new method, scale, and grid.
            controls.runTransaction(() => {
                controls.setMethod(methodName);
                controls.setPhysics(st.schema);
                controls.setMaterial(pbmpmMaterial);
                if (methodName === "FLIP") {
                    controls.setFlipParticleCapacity(st.count);
                } else {
                    controls.setParticleCount(st.count);
                }
                controls.setPhysScale(nextPhysicsScale);
                controls.setGridSettings([...targetGrid.position], [...targetGrid.size], targetCellSize);
                controls.setGridResolution(nextGridResolution);
                controls.setMarkersPerCell(nextMarkersPerCell);
                controls.setSimulationDuration(st.simulationDuration ?? 0);
                controls.setAlphaDecay(st.alphaDecay ?? 2);
                controls.setColor(st.color);
                controls.setIndependentRendering(st.independentRendering ?? false);
                controls.setHalf(st.half);
                controls.setThicknessDownscale(st.thicknessDownscale);
                controls.setParticleSize(st.size);
                controls.setAbsorption(st.absorption);
                // Surface render mode + shading sliders (each optional; only apply if present).
                if (st.renderMode !== undefined) {
                    controls.setRenderMode(st.renderMode === "spheres");
                }
                if (st.polygonShader !== undefined) {
                    controls.setPolygonShader(st.polygonShader);
                }
                if (st.refraction !== undefined) {
                    controls.setRefraction(st.refraction);
                }
                if (st.specular !== undefined) {
                    controls.setSpecular(st.specular);
                }
                // Reflection exposure/contrast share one host callback, so restore both together and fall
                // back to the live value when a preset carries only one — otherwise the pair's second
                // value would be pushed as whatever the previous demo left behind.
                if (st.reflectionExposure !== undefined || st.reflectionContrast !== undefined) {
                    const cur = controls.getValues();
                    controls.setReflection(st.reflectionExposure ?? cur.reflectionExposure, st.reflectionContrast ?? cur.reflectionContrast);
                }
                if (st.reflectivity !== undefined) {
                    controls.setReflectivity(st.reflectivity);
                }
                // The two depth sliders share one setter — restore both (fall back to the current
                // value when a preset carries only one) so the final surface-filter call matches.
                if (st.depthBlur !== undefined || st.depthBlurThreshold !== undefined) {
                    const cur = controls.getValues();
                    controls.setDepthBlur(st.depthBlur ?? cur.depthBlur, st.depthBlurThreshold ?? cur.depthBlurThreshold);
                }
                if (st.thicknessBlur !== undefined) {
                    controls.setThicknessBlur(st.thicknessBlur);
                }
                if (st.surfaceFilter !== undefined) {
                    controls.setSurfaceFilter(st.surfaceFilter);
                }
                if (st.narrowDelta !== undefined || st.narrowMu !== undefined) {
                    const cur = controls.getValues();
                    controls.setNarrowRange(st.narrowDelta ?? cur.narrowDelta, st.narrowMu ?? cur.narrowMu);
                }
                if (st.anisotropic !== undefined) {
                    controls.setAnisotropic(st.anisotropic);
                }
                if (st.anisoSurfScale !== undefined) {
                    controls.setAnisotropySurfScale(st.anisoSurfScale);
                }
                controls.setActiveBlocks(nextActiveBlocks);
                controls.setPagedGrid(nextPagedGrid);
                controls.setPagedGridMaxPages(nextPagedGridMaxPages, pageCapacityLimit);
                controls.setPagedGridStatus("");
                canvas.dataset.pagedGridOverflow = "false";
                controls.setFusedBlockDiscovery(nextFusedBlockDiscovery);
                // Missing softness/density/subsurface values in older presets retain the current look.
                if (st.foam) {
                    const f = st.foam;
                    const cur = controls.getValues().foam;
                    controls.setFoam({
                        enabled: f.enabled,
                        activeParticles: true,
                        generateSpray: f.generateSpray ?? true,
                        generateFoam: f.generateFoam ?? true,
                        generateBubbles: f.generateBubbles ?? true,
                        surfaceFiltering: f.surfaceFiltering ?? methodName === "FLIP",
                        kTa: f.kTa,
                        kWc: f.kWc,
                        kTurb: f.kTurb ?? cur.kTurb,
                        energySpeedMin: f.energySpeedMin ?? cur.energySpeedMin,
                        energySpeedMax: f.energySpeedMax ?? cur.energySpeedMax,
                        curvatureMin: f.curvatureMin ?? cur.curvatureMin,
                        curvatureMax: f.curvatureMax ?? cur.curvatureMax,
                        turbulenceMin: f.turbulenceMin ?? cur.turbulenceMin,
                        turbulenceMax: f.turbulenceMax ?? cur.turbulenceMax,
                        foamLayerDepth: f.foamLayerDepth ?? cur.foamLayerDepth,
                        sprayDrag: f.sprayDrag ?? cur.sprayDrag,
                        kb: f.kb,
                        kd: f.kd,
                        tMin: f.tMin,
                        tMax: f.tMax,
                        poolScale: f.poolScale,
                        size: f.size ?? cur.size,
                        blurRadius: f.blurRadius,
                        lightIntensity: f.lightIntensity,
                        ambient: f.ambient,
                        aoStrength: f.aoStrength,
                        normalStrength: f.normalStrength,
                        spraySize: f.spraySize ?? cur.spraySize ?? 0.55,
                        sprayIntensity: f.sprayIntensity ?? cur.sprayIntensity ?? 1.4,
                        spraySeparation: f.spraySeparation ?? cur.spraySeparation ?? 1,
                        debugTexture: f.debugTexture,
                        softness: f.softness ?? cur.softness,
                        density: f.density ?? cur.density,
                        subsurfaceStrength: f.subsurfaceStrength ?? cur.subsurfaceStrength,
                        subsurfaceColor: f.subsurfaceColor ?? cur.subsurfaceColor,
                    });
                }
                // Container/nozzle-mesh visibility (applied to the demo by switchPair's post-load
                // setContainerVisible(...) below).
                if (st.showContainer !== undefined) {
                    controls.setShowContainer(st.showContainer);
                }
                showGridBounds = st.showGridBounds ?? false;
                showGridBoundsSolid = st.showGridBoundsSolid ?? false;
                controls.setShowGridBounds(showGridBounds);
                controls.setShowGridBoundsSolid(showGridBoundsSolid);
            });
            if (controlsBinding && activeSim.options.backend !== (referenceSelected() ? referenceBackend : undefined)) {
                const plan = applyFluidControls(controlsBinding, controls.getValues());
                finishFluidControlPlan(plan, plan.changedKeys);
            }
        } catch (error) {
            gridSettings = previousGridSettings;
            throw error;
        } finally {
            loadingPairState = false;
        }
        // Core-owned viewing options the pair pins. Both are optional so files written before
        // they existed restore the defaults rather than turning themselves off/on at random.
        if (st.envIntensity !== undefined && st.envIntensity !== envIntensity) {
            envIntInput.value = String(st.envIntensity);
            applyEnvIntensity(st.envIntensity);
        }
        if (st.msaa !== undefined && st.msaa !== msaaOn) {
            msaaChk.checked = st.msaa;
            setMsaa(st.msaa);
        }
        const activeBlocksChanged =
            (methodName === "MLS-MPM" &&
                (nextActiveBlocks !== mpmActiveBlocks ||
                    nextPagedGrid !== mpmPagedGrid ||
                    nextPagedGridMaxPages !== mpmPagedGridMaxPages ||
                    nextFusedBlockDiscovery !== mpmFusedBlockDiscovery)) ||
            (methodName === "FLIP" && (nextPagedGrid !== flipPagedGrid || nextPagedGridMaxPages !== flipPagedGridMaxPages));
        mpmActiveBlocks = nextActiveBlocks;
        if (methodName === "FLIP") {
            flipPagedGrid = nextPagedGrid;
            flipPagedGridMaxPages = nextPagedGridMaxPages;
        } else if (methodName === "MLS-MPM") {
            mpmPagedGrid = nextPagedGrid;
            mpmPagedGridMaxPages = nextPagedGridMaxPages;
        }
        mpmFusedBlockDiscovery = nextFusedBlockDiscovery;
        const gridResolutionChanged = methodName === "FLIP" && nextGridResolution !== flipGridResolution;
        flipGridResolution = nextGridResolution;
        flipMarkersPerCell = nextMarkersPerCell;
        const gridChanged = !gridSettingsEqual(nextGridSettings, gridSettings) || !gridSettingsEqual(nextGridSettings, builtGridSettings);
        gridSettings = nextGridSettings;
        const allocationChanged = methodName === "FLIP" && flipParticleCapacity(st.count, activeFlow, nextGridResolution) !== particleCount;
        if (
            st.count !== requestedParticleCount() ||
            allocationChanged ||
            nextPhysicsScale !== physicsScale ||
            nextPhysicsScale !== builtPhysicsScale ||
            domainScale !== builtDomainScale ||
            gridChanged ||
            builtGridMethod !== methodName ||
            activeSim.options.backend !== (referenceSelected() ? referenceBackend : undefined) ||
            (methodName === "FLIP" && importedInitialState !== builtImportedInitialState) ||
            builtWithGridFloor !== (importedCollisionActive || demo.useGridFloor === true) ||
            activeBlocksChanged ||
            gridResolutionChanged ||
            (methodName === "FLIP" && nextGridResolution !== builtFlipGridResolution) ||
            markersPerCellChanged ||
            nextMarkersPerCell !== builtFlipMarkersPerCell
        ) {
            rebuildSims(st.count, nextPhysicsScale);
        } else {
            applySceneSdf(); // refresh emitters/spawn for the loaded demo params
            applyMethod(methodName);
            syncGridControls();
        }
        controls.setParticleCount(particleCount);
        controls.setFlipParticleCapacity(flipParticleCapacityRequest);
        controls.setPhysScale(nextPhysicsScale);
        controls.setGridResolution(nextGridResolution);
        controls.setMarkersPerCell(nextMarkersPerCell);
        refreshDemoParams();
        refreshFlowUI();
        // Apply the pair's camera framing (preset default on first visit, or the
        // viewpoint captured when this pair was last left). ArcRotate self-clamps.
        setCameraMirrorX(st.camera?.mirrorX === true);
        if (st.camera) {
            cam.alpha = st.camera.alpha;
            cam.beta = st.camera.beta;
            cam.radius = st.camera.radius;
            if (st.camera.target) {
                cam.target.x = st.camera.target[0];
                cam.target.y = st.camera.target[1];
                cam.target.z = st.camera.target[2];
            }
            cam.fov = st.camera.fov ?? DEFAULT_CAMERA.fov;
            canvas.dataset.cameraAlpha = String(cam.alpha);
            canvas.dataset.cameraBeta = String(cam.beta);
            canvas.dataset.cameraRadius = String(cam.radius);
            canvas.dataset.cameraTarget = [cam.target.x, cam.target.y, cam.target.z].join(",");
            canvas.dataset.cameraFov = String(cam.fov);
        }
        updateBackendControls();
    }
    // Switch to a (demo, method) pair: snapshot the pair we're leaving, set up the
    // demo visuals if the demo changed, then load the target pair's state.
    function resetCameraBase(): void {
        cam.alpha = DEFAULT_CAMERA.alpha;
        cam.beta = DEFAULT_CAMERA.beta;
        cam.radius = DEFAULT_CAMERA.radius;
        cam.target.x = DEFAULT_CAMERA.target[0];
        cam.target.y = DEFAULT_CAMERA.target[1];
        cam.target.z = DEFAULT_CAMERA.target[2];
        cam.fov = DEFAULT_CAMERA.fov;
        setCameraMirrorX(DEFAULT_CAMERA.mirrorX);
    }

    function switchPair(
        nextDemo: FluidDemo,
        nextMethod: string,
        nextQuality: Quality = quality,
        nextMaterial: number = pbmpmMaterial,
        invalidateImport = true,
        implementation?: "production" | "flip-reference"
    ): void {
        if (deferReferenceMutation(() => switchPair(nextDemo, nextMethod, nextQuality, nextMaterial, invalidateImport, implementation))) {
            return;
        }
        initialStabilizationGeneration++;
        const nextBackendId = nextMethod === "FLIP" ? (implementation === "production" ? undefined : (implementation ?? flipBackendId)) : undefined;
        if (nextBackendId) {
            if (!referenceBackend) {
                throw new Error("[fluid] FLIP Reference has not been loaded.");
            }
            const key = pairKeyFor(nextDemo, nextMethod, nextQuality, nextMaterial, nextBackendId);
            assertReferenceState(pairStates.get(key) ?? presetOrDefault(nextDemo, nextMethod, nextQuality, nextMaterial, nextBackendId), nextDemo);
        }
        const previousStaging = stagingBackendSwitch;
        stagingBackendSwitch ||= activeBackendId() !== undefined || nextBackendId !== undefined;
        try {
            const demoChanged = activeDemo !== nextDemo || currentPairKey === null;
            const qualityChanged = !demoChanged && quality !== nextQuality;
            const preservingImportedScene = importedScene !== null && !demoChanged && nextDemo.methodIndependentAuthoring === true;
            const leavingImportedScene = importedScene !== null && !preservingImportedScene;
            if (invalidateImport) {
                importGeneration++;
            }
            // Switching demo or fluid method resumes the sim if it was paused.
            if (paused) {
                paused = false;
                canvas.dataset.paused = "false";
            }
            if (currentPairKey !== null && (!importedScene || preservingImportedScene) && !suppressPairSnapshot) {
                const snapshot = readLivePairState(methodName);
                pairStates.set(currentPairKey, snapshot);
                const session = presetSessions.get(currentPairKey);
                if (session) {
                    presetSessions.set(currentPairKey, editFluidPresetSession(session, { state: snapshot }));
                }
                if (activeDemo?.methodIndependentAuthoring) {
                    methodIndependentStates.set(sharedStateKey(activeDemo.key, snapshot.backendId), snapshot);
                }
            }
            if (leavingImportedScene) {
                resetCameraBase();
                clearImportedScene(true);
            }
            if (preservingImportedScene && (activeBackendId() !== undefined || nextBackendId)) {
                syncImportedMeshAnimations(true);
            }
            // Adopt the target method before applying scene state.
            methodName = nextMethod;
            if (nextMethod === "FLIP") {
                flipBackendId = nextBackendId;
            }
            if (demoChanged) {
                if (activeDemo) {
                    activeDemo.onLeave();
                }
                flowEditor?.clearSelection(false);
                showEmitterWireframe = false;
                showSinkWireframe = false;
                flowGizmoOwner = null;
                setMeshVisible(emitterFlowWireframe, false);
                setMeshVisible(sinkFlowWireframe, false);
                syncFlowGizmo();
                activeDemo = nextDemo;
                activeFlow = flowToGridLocal(nextDemo.flow(), defaultGridSettings(nextMethod, nextDemo.getDomainScale?.() ?? 1).position);
                resetCameraBase();
                activeDemo.onEnter(); // meshes, camera mode
                applyDemoEnv(activeDemo); // swap skybox background + surface-reflection cube
                applyEnvRotation(activeDemo.envRotationDeg ?? 0); // aim the backdrop the way this demo wants it
                envRotInput.value = String(activeDemo.envRotationDeg ?? 0);
                applySceneSdf();
                refreshDemoParams();
                pendingForce = null;
                if (!stagingBackendSwitch) {
                    resetFluidSimulation(activeSim);
                }
                restartSimulationLifecycle();
                clearSceneHoles();
            }
            quality = nextQuality;
            const usesQualityPresets = nextDemo.usesQualityPresets !== false && !nextBackendId;
            qualitySel.style.display = usesQualityPresets ? "" : "none";
            canvas.dataset.qualityPresets = String(usesQualityPresets);
            controls.containerToggleRow!.style.display = nextDemo.setContainerVisible ? "" : "none";
            canvas.dataset.demo = nextDemo.key;
            syncHelperText(nextDemo);
            // PB-MPM keeps a separate physics/render/colour pair per material for every fluid demo.
            const withMaterial = nextMethod === "PB-MPM";
            if (nextMethod === "PB-MPM") {
                pbmpmMaterial = withMaterial ? nextMaterial : 0;
            }
            const key = pairKeyFor(nextDemo, nextMethod, nextQuality, pbmpmMaterial, nextBackendId);
            const defaultState = nextBackendId && pairStates.has(key) ? pairStates.get(key)! : presetOrDefault(nextDemo, nextMethod, nextQuality, pbmpmMaterial, nextBackendId);
            const targetState = pairStates.get(key) ?? defaultState;
            const sharedState = nextDemo.methodIndependentAuthoring ? methodIndependentStates.get(sharedStateKey(nextDemo.key, nextBackendId)) : undefined;
            const st = sharedState ? carryMethodIndependentState(targetState, sharedState, { retainTargetPresentation: withMaterial }) : targetState;
            currentPairKey = key;
            domainScale = typeof st.demoParams.meshScale === "number" ? st.demoParams.meshScale : (nextDemo.getDomainScale?.() ?? 1);
            loadPairState(st);
            pairStates.set(key, structuredClone(st));
            if ((demoChanged || qualityChanged) && !importedScene) {
                restartOrStabilizeActiveDemo();
            }
            if (demoChanged || leavingImportedScene) {
                const authoredCamera = defaultState.camera ?? DEFAULT_CAMERA;
                cam.alpha = authoredCamera.alpha;
                cam.beta = authoredCamera.beta;
                cam.radius = authoredCamera.radius;
                cam.fov = authoredCamera.fov ?? DEFAULT_CAMERA.fov;
                setCameraMirrorX(authoredCamera.mirrorX === true);
            }
            canvas.dataset.cameraAlpha = String(cam.alpha);
            canvas.dataset.cameraBeta = String(cam.beta);
            canvas.dataset.cameraRadius = String(cam.radius);
            canvas.dataset.cameraTarget = [cam.target.x, cam.target.y, cam.target.z].join(",");
            restartSimulationLifecycle();
            // Re-apply the container-mesh visibility choice (onEnter shows it by default).
            nextDemo.setContainerVisible?.(importedScene ? false : controls.getValues().showContainer);
            updateBackendControls();
            reconcileRenderingLoop();
        } finally {
            stagingBackendSwitch = previousStaging;
        }
    }

    // ── Services handed to each demo ──────────────────────────────────────
    const ctx: FluidCtx = {
        engine,
        scene,
        canvas,
        camera: cam,
        ground,
        sceneSdfBuffer,
        getActiveSimulation: () => activeSim,
        rebindSceneSdf: applySceneSdf,
        resetActiveSim: () => resetActiveFlow(false),
        refreshFlow: () => {
            if (loadingPairState || stagingBackendSwitch) {
                return;
            }
            activeFlow = flowToGridLocal(activeDemo!.flow());
            applyFlow();
            resetFluidSimulation(activeSim);
            builtInitialStateKey = flipParticlePlan(requestedParticleCount()).initialStateKey;
            restartSimulationLifecycle();
            refreshFlowUI();
        },
        addSceneHole,
        clearSceneHoles,
        sun,
        ambient,
        setSunShadows,
        get simHalfExtentXZ(): number {
            const bounds = gridSettings ? gridBounds(gridSettings.position, gridSettings.size) : defaultDomainBounds(methodName, domainScale);
            return Math.max(Math.abs(bounds.min[0]), Math.abs(bounds.max[0]), Math.abs(bounds.min[2]), Math.abs(bounds.max[2]));
        },
        viewProjection: () => getViewProjectionMatrix(cam, getEffectiveAspectRatio(cam, canvas.width, canvas.height)),
        setDomainScale: (s: number) => {
            if (stagingBackendSwitch) {
                domainScale = s;
                return false;
            }
            if (s === domainScale) {
                return false;
            }
            if (gridSettings) {
                const scaleRatio = s / Math.max(domainScale, 1e-6);
                const nextGridSettings: FluidGridSettings = {
                    position: gridSettings.position.map((value) => value * scaleRatio) as [number, number, number],
                    size: gridSettings.size.map((value) => value * scaleRatio) as [number, number, number],
                };
                const [minScale, maxScale] = scaleLimitsForMethod(methodName);
                const nextPhysicsScale = clampScale(physicsScale * scaleRatio, minScale, maxScale);
                const allocationError = gridAllocationError(nextGridSettings, methodName, nextPhysicsScale);
                if (allocationError) {
                    controls.setGridStatus(allocationError);
                    return false;
                }
                gridSettings = nextGridSettings;
                domainScale = s;
                rebuildSims(requestedParticleCount(), nextPhysicsScale);
                return true;
            }
            // Gridless presets retain the historical hidden scale on bounds, cell size,
            // particle radius and spawn so existing demos (notably Waterfall) are unchanged.
            domainScale = s;
            rebuildSims(requestedParticleCount(), physicsScale);
            return true;
        },
        setTransientSceneMeshScale: (scale: number) => {
            if (!Number.isFinite(scale) || scale < 0.01 || scale > 100) {
                throw new RangeError("[fluid] transient scene mesh scale must be between 0.01 and 100.");
            }
            const previousScale = transientSceneMeshScale;
            if (scale === previousScale) {
                return;
            }
            const imported = importedScene;
            if (imported) {
                const ratio = scale / previousScale;
                gridSettings = scaleGridSettingsAroundPivot(effectiveGridSettings(), ratio, imported.assetRootPosition);
                activeFlow = scaleFlowTransforms(activeFlow, ratio);
            }
            transientSceneMeshScale = scale;
            if (imported) {
                applyTransientSceneScale(imported);
                syncImportedSceneGridTransform(imported);
                updateImportedSceneBindings(imported, 0, true);
                syncGridControls();
                syncGridBoundsWireframe();
                syncGridGizmo();
                refreshFlowUI();
                syncFlowWireframe("emitter");
                syncFlowWireframe("sink");
                syncFlowGizmo();
                controls.setGridStatus("Simulation changes pending. Reset simulation to apply.");
            } else {
                canvas.dataset.whiteboardMeshScale = String(scale);
            }
        },
        setBloom: (cfg: { enabled: boolean; intensity: number; threshold: number }) => {
            bloomEnabled = cfg.enabled;
            bloomParams.intensity = cfg.intensity;
            bloomParams.threshold = cfg.threshold;
            // Both live in per-pass uniform buffers that are only written on demand —
            // without this the sliders would do nothing.
            bloomExtract.updateUniforms();
            bloomMerge.updateUniforms();
        },
    };

    // Build the demo registry (capsule default) and populate the demo dropdown.
    const demos: FluidDemo[] = [
        createWhiteboardDemo(ctx),
        createCapsuleDemo(ctx),
        await createBoxDemo(ctx),
        createFountainDemo(ctx),
        createWaterfallDemo(ctx),
        await createJumpingWhaleDemo(ctx),
        createMarbleTowerDemo(ctx),
    ];
    for (const d of demos) {
        const opt = document.createElement("option");
        opt.value = d.key;
        opt.textContent = d.label;
        containerSel.appendChild(opt);
        const bag: Record<string, number> = {};
        for (const p of d.demoParams()) {
            if (p.type === "number") bag[p.key] = p.value;
        }
        DEMO_PARAM_DEFAULTS[d.key] = bag;
    }

    // Collect every demo's translucent container mesh and route it through the
    // container-glass overlay task (created above). `addMesh` only queues the mesh;
    // its renderable is resolved from the material's registered pipeline at
    // registerScene time (below), so this must run before registerScene. The same
    // meshes are stripped out of the scene-colour pass after registerScene so they
    // are drawn exactly once — as the post-fluid overlay.
    const overlayMeshes: Mesh[] = [];
    for (const d of demos) {
        for (const m of d.containerMeshes?.() ?? []) {
            overlayMeshes.push(m);
            addMeshToTask(overlayTask, m);
        }
    }

    let paused = false;
    let captureDemoActivated = !deterministicMode;
    let captureWarmupFrames = 0;
    let captureStarted = false;
    let captureStep = 0;
    let captureReadyPending = false;
    let captureReadyTime = 0;
    let offlineCompletedSteps = 0;
    let initialStabilizationGeneration = 0;
    let initialStabilizationRunning = false;
    let initialStabilizationTask: Promise<void> = Promise.resolve();
    function syncSimulationLifecycle(): void {
        const lifecycle = fluidSimulationLifecycle(simulationElapsed, simulationDuration, simulationAlphaDecay);
        simulationOpacity = lifecycle.opacity;
        simulationStopped = lifecycle.stopped;
        configureFluidSimulationRenderLayer(particleTask, { opacity: simulationOpacity });
        configureFluidSimulationRenderLayer(surfaceTask, { opacity: simulationOpacity });
        configureFluidSimulationRenderLayer(polygonSurfaceTask, { opacity: simulationOpacity });
        configureFluidSimulationRenderLayer(foamTask, { opacity: simulationOpacity, enabled: foamRenderVisible() });
        canvas.dataset.simulationOpacity = simulationOpacity.toFixed(3);
        canvas.dataset.simulationStopped = simulationStopped ? "true" : "false";
    }

    function restartSimulationLifecycle(): void {
        simulationElapsed = 0;
        captureStep = 0;
        captureReadyPending = false;
        captureReadyTime = 0;
        delete canvas.dataset.captureReady;
        delete canvas.dataset.captureTime;
        if (offlineMode) {
            offlineCompletedSteps = 0;
            canvas.dataset.offlineStep = "0";
            canvas.dataset.offlineSimulationTime = "0";
        }
        if (referenceSelected()) {
            canvas.dataset.referenceSimulationTime = "0";
        }
        syncSimulationLifecycle();
    }

    async function readInitialStabilizationSample(activeCount: number): Promise<Float32Array> {
        const blockSize = Math.min(2048, activeCount);
        if (blockSize <= 0) {
            return new Float32Array(0);
        }
        const maximumOffset = Math.max(0, activeCount - blockSize);
        const offsets = [...new Set([0, Math.floor(maximumOffset / 3), Math.floor((maximumOffset * 2) / 3), maximumOffset])];
        const sample = new Float32Array(offsets.length * blockSize * 3);
        let cursor = 0;
        for (const particleOffset of offsets) {
            const positions = await readFluidSimulationPositions(activeSim, { particleOffset, particleCount: blockSize });
            for (let index = 0; index < positions.length; index += 4) {
                sample[cursor++] = positions[index]!;
                sample[cursor++] = positions[index + 1]!;
                sample[cursor++] = positions[index + 2]!;
            }
        }
        return sample;
    }

    function initialStabilizationMotion(previous: Float32Array, current: Float32Array): { rms: number; p95: number } {
        if (previous.length !== current.length || current.length === 0) {
            return { rms: Infinity, p95: Infinity };
        }
        const previousHeights = new Float32Array(previous.length / 3);
        const currentHeights = new Float32Array(current.length / 3);
        for (let index = 0; index < currentHeights.length; index++) {
            previousHeights[index] = previous[index * 3 + 1]!;
            currentHeights[index] = current[index * 3 + 1]!;
        }
        previousHeights.sort();
        currentHeights.sort();
        let squared = 0;
        const differences = new Float32Array(currentHeights.length);
        for (let index = 0; index < currentHeights.length; index++) {
            const difference = Math.abs(currentHeights[index]! - previousHeights[index]!);
            squared += difference * difference;
            differences[index] = difference;
        }
        differences.sort();
        return {
            rms: Math.sqrt(squared / differences.length),
            p95: differences[Math.floor((differences.length - 1) * 0.95)]!,
        };
    }

    async function runInitialStabilization(generation: number, demo: FluidDemo): Promise<void> {
        if (generation !== initialStabilizationGeneration || activeDemo !== demo || methodName === "FLIP" || demo.initialStabilizationEnabled?.() !== true) {
            return;
        }
        initialStabilizationRunning = true;
        demo.restartAnimation?.(true);
        demo.onInitialStabilizationProgress?.({ running: true, simulatedSeconds: 0, converged: false });
        canvas.dataset.initialStabilization = "running";
        configureFluidSimulationRenderLayer(particleTask, { opacity: 0 });
        configureFluidSimulationRenderLayer(surfaceTask, { opacity: 0 });
        configureFluidSimulationRenderLayer(polygonSurfaceTask, { opacity: 0 });
        configureFluidSimulationRenderLayer(foamTask, { opacity: 0, enabled: false });
        suspendRenderingLoop();
        let simulatedSeconds = 0;
        let converged = false;
        let failed = false;
        try {
            await demo.prepareInitialStabilization?.();
            if (generation !== initialStabilizationGeneration || activeDemo !== demo) {
                return;
            }
            await engine._device.queue.onSubmittedWorkDone();
            const sampledParticleCount = activeSim.activeCount ?? activeSim.count;
            let previous = await readInitialStabilizationSample(sampledParticleCount);
            let stableSamples = 0;
            const stepSeconds = 1 / 30;
            const stepsPerBatch = 6;
            const batchesPerSample = 5;
            const maximumSeconds = 30;
            while (
                generation === initialStabilizationGeneration &&
                activeDemo === demo &&
                demo.initialStabilizationEnabled?.() === true &&
                simulatedSeconds + stepSeconds * stepsPerBatch * batchesPerSample <= maximumSeconds + 1e-6
            ) {
                for (let batch = 0; batch < batchesPerSample; batch++) {
                    await submitFluidSimulationSteps(activeSim, stepSeconds, stepsPerBatch);
                    simulatedSeconds += stepSeconds * stepsPerBatch;
                    if (generation !== initialStabilizationGeneration || activeDemo !== demo) {
                        return;
                    }
                }
                const current = await readInitialStabilizationSample(sampledParticleCount);
                const motion = initialStabilizationMotion(previous, current);
                previous = current;
                stableSamples = motion.rms < 0.02 && motion.p95 < 0.03 ? stableSamples + 1 : 0;
                demo.onInitialStabilizationProgress?.({ running: true, simulatedSeconds, converged: false });
                canvas.dataset.initialStabilizationSeconds = simulatedSeconds.toFixed(1);
                canvas.dataset.initialStabilizationRms = motion.rms.toFixed(4);
                canvas.dataset.initialStabilizationP95 = motion.p95.toFixed(4);
                if (simulatedSeconds >= 2 && stableSamples >= 2) {
                    converged = true;
                    break;
                }
            }
            if (generation === initialStabilizationGeneration && activeDemo === demo && demo.initialStabilizationEnabled?.() === true) {
                await settleFluidSimulation(activeSim);
            }
        } catch (error) {
            failed = true;
            console.error("[fluid] initial stabilization failed", error);
        } finally {
            initialStabilizationRunning = false;
            if (generation === initialStabilizationGeneration && activeDemo === demo) {
                demo.restartAnimation?.(paused);
                demo.onInitialStabilizationProgress?.({ running: false, simulatedSeconds, converged });
                canvas.dataset.initialStabilization = failed ? "error" : converged ? "converged" : "complete";
            }
            syncSimulationLifecycle();
            reconcileRenderingLoop();
        }
    }

    function restartOrStabilizeActiveDemo(): void {
        const demo = activeDemo;
        const generation = ++initialStabilizationGeneration;
        if (!demo) {
            return;
        }
        if (methodName === "FLIP" || demo.initialStabilizationEnabled?.() !== true) {
            initialStabilizationRunning = false;
            canvas.dataset.initialStabilization = methodName === "FLIP" ? "skipped" : "disabled";
            delete canvas.dataset.initialStabilizationSeconds;
            delete canvas.dataset.initialStabilizationRms;
            delete canvas.dataset.initialStabilizationP95;
            demo.restartAnimation?.(paused);
            return;
        }
        initialStabilizationTask = initialStabilizationTask
            .catch(() => {
                // The preceding run already surfaced its error.
            })
            .then(() => runInitialStabilization(generation, demo));
    }

    function advanceSimulationLifecycle(dt: number): void {
        if (simulationStopped) {
            return;
        }
        simulationElapsed += dt;
        syncSimulationLifecycle();
    }

    let referenceTimingCache: { scene: ImportedFluidScene | null; timing: ReturnType<typeof fluidReferenceTiming> } | null = null;
    function referenceTiming(): ReturnType<typeof fluidReferenceTiming> {
        if (!referenceTimingCache || referenceTimingCache.scene !== importedScene) {
            referenceTimingCache = { scene: importedScene, timing: fluidReferenceTiming(importedScene?.bundle.preset) };
        }
        return referenceTimingCache.timing;
    }

    function updateSceneForSimulationStep(deltaSeconds: number): void {
        insideReferenceSubstep = true;
        try {
            if (importedScene) {
                const animationDelta = deltaSeconds * (referenceSelected() ? referenceTiming().animationRate : 1);
                importedScene.asset._beforeRenderHook?.(animationDelta * 1000);
                updateImportedSceneBindings(importedScene, deltaSeconds * 1000);
            } else {
                activeDemo?.update(deltaSeconds);
            }
        } finally {
            insideReferenceSubstep = false;
        }
    }

    function completeInteractiveStep(deltaSeconds: number): void {
        advanceSimulationLifecycle(deltaSeconds);
        if (captureMode) {
            captureStep++;
            const completedAt = fluidCaptureCompletionTime(simulationElapsed, captureStep >= captureTargetSteps, simulationStopped);
            if (completedAt !== null) {
                paused = true;
                canvas.dataset.paused = "true";
                captureReadyTime = completedAt;
                captureReadyPending = true;
            }
        }
    }

    function suspendRenderingLoop(): void {
        stopEngine(engine);
        normalEngineLoopRunning = false;
    }

    function reconcileRenderingLoop(): void {
        if (!engineLoopReady || !referencePump || referencePump.transitions > 0 || initialStabilizationRunning) {
            return;
        }
        if (offlineMode) {
            if (!captureStarted) {
                return;
            }
            stopFluidReferencePump(referencePump);
            suspendRenderingLoop();
        } else if (activeSim.steppingMode === "async") {
            suspendRenderingLoop();
            startFluidReferencePump(referencePump);
        } else {
            stopFluidReferencePump(referencePump);
            if (!normalEngineLoopRunning) {
                normalEngineLoopRunning = true;
                void startEngine(engine).catch((error: unknown) => {
                    normalEngineLoopRunning = false;
                    reportReferenceError(error);
                });
            }
        }
    }

    function referenceShouldStep(): boolean {
        return (
            !offlineMode &&
            activeSim.steppingMode === "async" &&
            !paused &&
            !initialStabilizationRunning &&
            !simulationStopped &&
            captureDemoActivated &&
            (!deterministicMode || captureStarted) &&
            activeDemo?.isReady?.() !== false
        );
    }
    referencePump = createFluidReferencePump({
        render: (deltaMs) => renderFrame(engine, deltaMs),
        shouldStep: referenceShouldStep,
        step: async () => {
            const fixedDelta = captureMode ? captureFixedDt : referenceTiming().frameDelta;
            const stepDt = fluidSimulationStepDelta(simulationElapsed, fixedDelta * simulationTimeScale, simulationDuration, simulationAlphaDecay);
            if (stepDt <= 0) {
                syncSimulationLifecycle();
                const activeProfiler = currentGpuProfiler();
                if (activeProfiler) {
                    await submitFluidSimulationProfiler(activeProfiler);
                }
                return;
            }
            canvas.dataset.referenceStatus = "stepping";
            await submitFluidSimulationStep(activeSim, stepDt, { beforeSubstep: updateSceneForSimulationStep });
            const activeProfiler = currentGpuProfiler();
            if (activeProfiler) {
                await submitFluidSimulationProfiler(activeProfiler);
            }
            completeInteractiveStep(stepDt);
            canvas.dataset.activeParticleCount = String(activeSim.activeCount ?? activeSim.count);
            canvas.dataset.referenceStatus = paused ? "paused" : "ready";
            canvas.dataset.referenceSimulationTime = String(simulationElapsed);
        },
        onError: reportReferenceError,
        suspend: suspendRenderingLoop,
        resume: reconcileRenderingLoop,
    });

    // DOM controls may change while a GPU solve is pending; replay mutations after its fence.
    const deferPanelEvent = (event: Event): void => {
        if (!referencePump?.pending || !(event.target instanceof Element)) {
            return;
        }
        const target = event.target;
        if (event.type === "click" && (!target.closest("button") || target.closest("button") === importBtn)) {
            return;
        }
        event.stopImmediatePropagation();
        if (event.type === "click") {
            event.preventDefault();
        }
        const replay =
            event instanceof MouseEvent
                ? new MouseEvent(event.type, { bubbles: true, cancelable: true, shiftKey: event.shiftKey, ctrlKey: event.ctrlKey, altKey: event.altKey, metaKey: event.metaKey })
                : new Event(event.type, { bubbles: true, cancelable: event.cancelable });
        void queueFluidReferenceChange(referencePump, () => {
            if (target.isConnected) {
                target.dispatchEvent(replay);
            }
        });
    };
    controls.root.addEventListener("input", deferPanelEvent, true);
    controls.root.addEventListener("change", deferPanelEvent, true);
    controls.root.addEventListener("click", deferPanelEvent, true);

    interface FluidOfflineRunDetail {
        token: string;
        steps: number;
        render: boolean;
    }
    function updateOfflineSceneForSimulationStep(deltaSeconds: number): void {
        updateSceneForSimulationStep(deltaSeconds);
        setFluidSimulationForceField(activeSim, forceFieldHandle(importedScene ? null : (activeDemo?.forceField?.() ?? null)));
    }
    let offlineOperation = Promise.resolve();
    const advanceOfflineStep = async (): Promise<void> => {
        if (simulationStopped) {
            return;
        }
        const stepDt = fluidSimulationStepDelta(simulationElapsed, offlineFixedDt * simulationTimeScale, simulationDuration, simulationAlphaDecay);
        if (referenceSelected()) {
            const activeProfiler = currentGpuProfiler();
            if (activeProfiler) {
                beginFluidSimulationProfilerFrame(activeProfiler, { captureEnvelope: false });
            }
            await submitFluidSimulationStep(activeSim, stepDt, { beforeSubstep: updateOfflineSceneForSimulationStep });
            if (activeProfiler) {
                await submitFluidSimulationProfiler(activeProfiler);
            }
        } else {
            await submitFluidSimulationStep(activeSim, stepDt, { beforeSubstep: updateOfflineSceneForSimulationStep });
        }
        advanceSimulationLifecycle(stepDt);
        offlineCompletedSteps++;
        canvas.dataset.offlineStep = String(offlineCompletedSteps);
        canvas.dataset.offlineSimulationTime = String(simulationElapsed);
        canvas.dataset.activeParticleCount = String(activeSim.activeCount ?? activeSim.count);
        refreshReferenceBudgetWarning();
    };
    canvas.addEventListener("fluid-offline-run", (event) => {
        if (!offlineMode) {
            return;
        }
        const detail = (event as CustomEvent<FluidOfflineRunDetail>).detail;
        if (!detail || typeof detail.token !== "string" || !detail.token || !Number.isInteger(detail.steps) || detail.steps < 0 || typeof detail.render !== "boolean") {
            canvas.dataset.offlineStatus = "error";
            canvas.dataset.offlineError = "Invalid fluid-offline-run request.";
            return;
        }
        offlineOperation = offlineOperation
            .then(async () => {
                canvas.dataset.offlineStatus = "running";
                delete canvas.dataset.offlineError;
                delete canvas.dataset.offlineErrorToken;
                for (let step = 0; step < detail.steps && !simulationStopped; step++) {
                    await advanceOfflineStep();
                }
                if (detail.render) {
                    renderFrame(engine, 0);
                    await waitForGpuIdle(engine);
                }
                canvas.dataset.offlineCompletedToken = detail.token;
                canvas.dataset.offlineStatus = "idle";
            })
            .catch((error: unknown) => {
                const message = error instanceof Error ? error.message : String(error);
                canvas.dataset.offlineStatus = "error";
                canvas.dataset.offlineErrorToken = detail.token;
                canvas.dataset.offlineError = message;
                console.error("[fluid] offline rendering failed", error);
            });
    });

    onBeforeRender(scene, (deltaMs: number) => {
        // A newly-inserted task (the MSAA scene pass + its depth resolve) needs the whole graph
        // re-recorded, which re-allocates canvas-sized targets other tasks' bind groups point
        // at. Do it here, at the very top of the frame, before anything is encoded against them.
        if (pendingFrameGraphRebuild) {
            pendingFrameGraphRebuild = false;
            getFrameGraph(scene).build();
        }
        if (captureReadyPending) {
            captureReadyPending = false;
            canvas.dataset.captureTime = String(captureReadyTime);
            canvas.dataset.captureReady = "true";
        }
        if (!captureDemoActivated) {
            if (captureWarmupFrames++ === 0) {
                return;
            }
            const captureDemo = demos.find((demo) => demo.key === captureDemoKey);
            if (!captureDemo || !captureMethod || !Object.hasOwn(DEFAULT_FLUID_SCHEMAS, captureMethod) || !captureQuality || !QUALITIES.includes(captureQuality)) {
                throw new Error("Invalid deterministic fluid capture parameters");
            }
            switchPair(captureDemo, captureMethod, captureQuality);
            visitedDemos.add(captureDemo.key);
            containerSel.value = captureDemo.key;
            qualitySel.value = captureQuality;
            controls.setMethod(captureMethod);
            captureDemoActivated = true;
            if (pendingFrameGraphRebuild) {
                pendingFrameGraphRebuild = false;
                getFrameGraph(scene).build();
            }
            return;
        }
        // A demo's glTF resolves long after its switchPair, so meshes keep arriving. Re-fan the
        // environment intensity over them whenever the scene's mesh set grows, or a model loaded
        // after the slider was moved would render at the default 1.0.
        if (envIntensityTouched && scene.meshes.length !== envIntensityMeshCount) {
            pushEnvIntensity();
        }
        // Opt-in GPU timing: reset the profiler's per-frame query cursor BEFORE any
        // timed pass (sim.step below + the render tasks) is encoded this frame, then
        // open the whole-frame envelope (frameStart) so the "Total" row reports the real
        // total GPU time of the frame, not just the sum of the itemised passes.
        const activeProfiler = currentGpuProfiler();
        if (timingEnabled && activeProfiler) {
            beginFluidSimulationProfilerFrame(activeProfiler);
        }
        // Smoothed FPS: average the frame interval over ~0.5 s, then refresh.
        fpsAccumMs += deltaMs;
        fpsFrames++;
        if (fpsAccumMs >= 500) {
            const gpu = controls.gpu;
            if (gpu) {
                gpu.fpsLabel.textContent = paused ? "paused" : simulationStopped ? "stopped" : `${Math.round((fpsFrames * 1000) / fpsAccumMs)}`;
            }
            fpsAccumMs = 0;
            fpsFrames = 0;
            refreshReferenceBudgetWarning();
            // Refresh the timing + memory read-outs on the same ~2 Hz cadence as the FPS counter.
            if (gpu) {
                gpu.refreshTiming(timingEnabled && activeProfiler ? readFluidSimulationProfiler(activeProfiler) : null);
                gpu.refreshMemory(activeSim.gpuBytes, engine.canvas.width, engine.canvas.height);
            }
        }
        // Clamp dt so a hitch / first frame can't blow the integration up.
        const dt = captureMode ? captureFixedDt : offlineMode ? offlineFixedDt : referenceSelected() ? referenceTiming().frameDelta : Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
        if (deterministicMode && !captureStarted) {
            if (activeDemo?.isReady?.() === false) {
                return;
            }
            if (!importedScene) {
                activeDemo?.update(0);
            }
            resetFluidSimulation(activeSim);
            restartSimulationLifecycle();
            captureStarted = true;
            canvas.dataset.captureStarted = "true";
            if (offlineMode) {
                canvas.dataset.offlinePrepared = "true";
            }
        }
        // Keep the latest pointer sample alive briefly: MLS-MPM may defer a whole step
        // while asynchronous status buffers are unavailable.
        if (pendingForce && pendingForce.expiresAt >= performance.now()) {
            rayForce.setRay(pendingForce.origin, pendingForce.dir, pendingForce.push, pendingForce.radius, pendingForce.accel);
            setFluidSimulationForceField(activeSim, forceFieldHandle(rayForce.spec));
        } else {
            pendingForce = null;
            setFluidSimulationForceField(activeSim, forceFieldHandle(importedScene ? null : (activeDemo?.forceField?.() ?? null)));
        }
        // "P" pauses: freeze the obstacles + the solver so the fluid stops advancing.
        // Rendering and the camera keep running, so you can inspect the frozen state;
        // forces / holes resume on unpause. A reconstruction-only refresh is still
        // allowed so polygon quality changes remain visible on the frozen particles.
        if (paused || simulationStopped) {
            refreshFluidSimulationPolygonSurface(activeSim);
        }
        if (!offlineMode && activeSim.steppingMode === "frame" && !paused && !simulationStopped && !initialStabilizationRunning) {
            const stepDt = fluidSimulationStepDelta(simulationElapsed, dt * simulationTimeScale, simulationDuration, simulationAlphaDecay);
            updateSceneForSimulationStep(stepDt);
            if (referenceSelected()) {
                try {
                    stepFluidSimulation(activeSim, stepDt);
                    completeInteractiveStep(stepDt);
                    canvas.dataset.referenceStatus = paused ? "paused" : "ready";
                    canvas.dataset.referenceSimulationTime = String(simulationElapsed);
                } catch (error) {
                    reportReferenceError(error);
                }
            } else {
                stepFluidSimulation(activeSim, stepDt);
                completeInteractiveStep(stepDt);
            }
        }
        const activeParticleCount = activeSim.activeCount ?? activeSim.count;
        refreshParticleUsageStatus(activeParticleCount);
        canvas.dataset.activeParticleCount = String(activeParticleCount);
        updateFlipMarkerDensityWarning(activeParticleCount);
    });

    // ── Input dispatch ────────────────────────────────────────────────────
    // The built-in arc control (attached above) owns LMB-rotate / RMB-slide /
    // wheel-zoom in every demo. These listeners add the two gestures it was told
    // to ignore: Shift+RMB pushes the fluid (core-owned), and a plain pointerdown
    // is forwarded to the active demo (capsule: LMB over the tank punches a hole).
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) => {
        if (isForceGesture(e)) {
            forceDragging = true;
            pendingForce = null;
            canvas.setPointerCapture(e.pointerId);
            forceLastX = e.clientX;
            forceLastY = e.clientY;
            forceLastT = performance.now();
        } else if (!importedScene) {
            if (!deferReferenceMutation(() => activeDemo?.onPointerDown?.(e))) {
                activeDemo?.onPointerDown?.(e);
            }
        }
    });
    canvas.addEventListener("pointermove", (e) => {
        if (forceDragging) {
            // Push the fluid: direction = screen motion mapped into world space via
            // the camera basis; magnitude ∝ mouse speed (px/s); origin/dir = the
            // cursor ray. The loop retains the latest sample briefly across deferred steps.
            const dx = e.clientX - forceLastX;
            const dy = e.clientY - forceLastY;
            const now = performance.now();
            const dtMs = Math.max(now - forceLastT, 1);
            forceLastX = e.clientX;
            forceLastY = e.clientY;
            forceLastT = now;
            const speed = (Math.sqrt(dx * dx + dy * dy) / dtMs) * 1000;
            if (speed < 1) {
                return;
            }
            const rect = canvas.getBoundingClientRect();
            const pointerX = e.clientX - rect.left;
            const rayX = cameraMirrorX ? rect.width - pointerX : pointerX;
            const ray = screenRay(ctx.viewProjection(), rayX, e.clientY - rect.top, rect.width, rect.height);
            if (!ray) {
                return;
            }
            const wm = cam.worldMatrix;
            // World-space mouse motion: camera right * dx − camera up * dy (screen-y down).
            const horizontalDx = cameraMirrorX ? -dx : dx;
            let px = wm[0]! * horizontalDx - wm[4]! * dy;
            let py = wm[1]! * horizontalDx - wm[5]! * dy;
            let pz = wm[2]! * horizontalDx - wm[6]! * dy;
            const plen = Math.sqrt(px * px + py * py + pz * pz) || 1;
            px /= plen;
            py /= plen;
            pz /= plen;
            pendingForce = {
                origin: ray.origin,
                dir: ray.dir,
                push: [px, py, pz],
                radius: FORCE_RADIUS,
                accel: speed * 0.5 * (activeDemo?.interactiveForceScale ?? 1),
                expiresAt: now + INTERACTIVE_FORCE_SAMPLE_HOLD_MS,
            };
        } else if (!importedScene) {
            if (!deferReferenceMutation(() => activeDemo?.onPointerMove?.(e))) {
                activeDemo?.onPointerMove?.(e);
            }
        }
    });
    const onPointerEnd = (e: PointerEvent): void => {
        if (forceDragging) {
            forceDragging = false;
            if (e.type === "pointercancel") {
                pendingForce = null;
            }
            canvas.releasePointerCapture(e.pointerId);
        } else if (!importedScene) {
            if (!deferReferenceMutation(() => activeDemo?.onPointerUp?.(e))) {
                activeDemo?.onPointerUp?.(e);
            }
        }
    };
    canvas.addEventListener("pointerup", onPointerEnd);
    canvas.addEventListener("pointercancel", onPointerEnd);

    window.addEventListener("keydown", (e) => {
        if (e.repeat) {
            return;
        }
        if (!importedScene) {
            if (!deferReferenceMutation(() => activeDemo?.onKey?.(e))) {
                activeDemo?.onKey?.(e);
            }
        }
        if (e.code === "Space") {
            e.preventDefault(); // prevent page scroll in every demo
            return;
        }
        if (e.key === "F8") {
            e.preventDefault(); // some browsers bind F8 to the debugger
            setUiHidden(!uiHidden);
            return;
        }
        // Global shortcuts: R resets (refills), M toggles the backend, P pauses, F8 hides the UI.
        if (e.key === "r" || e.key === "R") {
            resetActiveFlow(true, e.shiftKey);
        } else if (e.key === "m" || e.key === "M") {
            const methods = Object.keys(DEFAULT_FLUID_SCHEMAS);
            const nextMethod = methods[(methods.indexOf(methodName) + 1) % methods.length] ?? "PBF";
            requestPairSwitch(activeDemo!, nextMethod);
        } else if (e.key === "p" || e.key === "P") {
            referencePausedByError = false;
            paused = !paused;
            canvas.dataset.paused = paused ? "true" : "false";
            activeDemo?.onPauseChanged?.(paused);
            if (referenceSelected()) {
                canvas.dataset.referenceStatus = referencePump?.pending ? (paused ? "pausing" : "stepping") : paused ? "paused" : "ready";
            }
            if (paused) {
                if (controls.gpu) {
                    controls.gpu.fpsLabel.textContent = "paused";
                }
            }
        }
    });

    const boxDemo = demos.find((d) => d.key === "box") ?? demos[0]!;
    // Load the on-disk quality presets (served from lab/public/fluid-presets) BEFORE the first
    // switchPair, so first-visit lookups see them. Runtime fetch → editing a preset + reloading
    // the page applies it with no bundle rebuild.
    await loadQualityPresets(
        demos.filter((demo) => demo.usesQualityPresets !== false).map((demo) => demo.key),
        demos.filter((demo) => demo.usesQualityPresets !== false).map((demo) => demo.key)
    );
    switchPair(boxDemo, "MLS-MPM", quality); // box + MLS-MPM at the default quality
    visitedDemos.add(boxDemo.key); // the start-up demo counts as visited
    containerSel.value = boxDemo.key;
    qualitySel.value = quality;
    controls.setMethod("MLS-MPM");
    applyRenderMode(false); // fluid surface by default
    initializeFluidControlsBinding();
    if (deterministicMode) {
        setUiHidden(true);
    }

    // Ensure the environment finished loading (skybox builder registered + specular
    // cube wired into the surface pass + imageProcessing set) before we build the
    // scene, so the HDR skybox renders as the sceneColorRT background from frame 0.
    await envReady;
    applyDemoEnv(activeDemo!); // install the initial (box) skybox + surface env before frame 0
    await registerSceneWithShadowSupport(scene);
    await registerUtilityLayer(flowGizmoLayer);

    // Strip the translucent container meshes out of the auto-mirrored scene-colour
    // pass: they are now drawn only by the container-glass overlay task (after the
    // fluid). Remove their scene renderables and bump the renderable version so the
    // scene task re-syncs its binding lists without them next frame. The overlay task
    // holds its OWN renderables (built via addMesh → material._rebuildSingle) that
    // share the same mesh GPU buffers, so the glass still renders — once, on top.
    if (overlayMeshes.length > 0) {
        for (const m of overlayMeshes) {
            const i = scene._renderables.findIndex((r) => r.mesh === m);
            if (i >= 0) {
                scene._renderables.splice(i, 1);
            }
        }
        scene._renderableVersion++;
    }

    normalEngineLoopRunning = true;
    await startEngine(engine);
    engineLoopReady = true;
    updateBackendControls();
    if (offlineMode) {
        await new Promise<void>((resolve) => {
            const waitForPreparation = (): void => {
                if (captureStarted) {
                    resolve();
                } else {
                    requestAnimationFrame(waitForPreparation);
                }
            };
            requestAnimationFrame(waitForPreparation);
        });
        suspendRenderingLoop();
        await waitForGpuIdle(engine);
        canvas.dataset.offlineReady = "true";
        canvas.dataset.offlineStatus = "idle";
        canvas.dataset.offlineStep = "0";
        canvas.dataset.offlineSimulationTime = "0";
    }
    reconcileRenderingLoop();
    canvas.dataset.drawCalls = String(engine.drawCallCount);
    canvas.dataset.particleCount = String(particleCount);
    canvas.dataset.initMs = String(performance.now() - __initStart);
    canvas.dataset.ready = "true";
}

main().catch((err) => {
    console.error(err);
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement | null;
    if (canvas) {
        canvas.dataset.error = String(err);
    }
});
