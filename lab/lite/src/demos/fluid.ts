// Fluid demo — GPU fluid simulation (PBF / MLS-MPM / PB-MPM) with SDF scene collision.
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
    addTask,
    addTaskAfter,
    addTaskBefore,
    addToScene,
    attachControl,
    createArcRotateCamera,
    createDepthResolveTask,
    createDirectionalLight,
    createEngine,
    createGround,
    createHemisphericLight,
    createMeshFromData,
    createPbrMaterial,
    createPcfDirectionalShadowGenerator,
    createRenderTarget,
    createRenderTask,
    createSceneContext,
    createStandardMaterial,
    getEffectiveAspectRatio,
    getFrameGraph,
    getViewProjectionMatrix,
    loadEnvironment,
    loadHdrEnvironment,
    createBlurPostProcessTask,
    markMaterialUboDirty,
    onBeforeRender,
    registerSceneWithShadowSupport,
    setMeshVisible,
    setShadowTaskCasterMeshes,
    startEngine,
} from "babylon-lite";
import { createPbfSim } from "babylon-lite/fluid/pbf-sim.js";
import type { FluidSim } from "babylon-lite/fluid/sim-common.js";
import { createMlsMpmSim } from "babylon-lite/fluid/mls-mpm-sim.js";
import { createPbMpmSim, pbmpmParamKeysForMaterial } from "babylon-lite/fluid/pbmpm-sim.js";
import { createRayForce } from "babylon-lite/fluid/ray-force.js";
import { createParticleRenderTask } from "babylon-lite/fluid/particle-render.js";
import { createFluidSurfaceTask } from "babylon-lite/fluid/fluid-surface-render.js";
import { createFoamRenderTask } from "babylon-lite/fluid/foam-render.js";
import type { FoamConfig } from "babylon-lite/fluid/sim-common.js";
import type { Mesh, Task, EnvironmentTextures, Renderable, Material, PbrMaterialProps } from "babylon-lite";
import { buildHdrSkyboxRenderable } from "babylon-lite/material/pbr/background-hdr-skybox.js";
// Plain source→target blit used as the no-bloom presentation pass. Only the TYPE is
// re-exported from the package root, so the factory comes from its own module (the same
// deep-import convention the fluid sim + HDR skybox already use here).
import { createPostProcessTask } from "babylon-lite/frame-graph/post-process-task.js";
import { createFluidProfiler } from "./fluid/gpu-profiler.js";
import type { FluidProfilerImpl } from "./fluid/gpu-profiler.js";
import { createFluidControlsPanel, DEFAULT_FLUID_SCHEMAS } from "babylon-lite/fluid/controls-panel.js";
import { demoAssetUrl } from "./demo-asset-url.js";
import type { DemoParam, FluidCtx, FluidDemo, FluidDomainBounds, FluidGridSettings, PairState, PendingForce } from "./fluid/demo.js";
import { exportJsonFromPairState } from "./fluid/preset-io.js";
import { getQualityPreset, QUALITIES, DEFAULT_QUALITY, loadQualityPresets, type Quality } from "./fluid/quality-presets.js";
import { ENV_STUDIO_URL } from "./fluid/demo.js";
import {
    cellSizeForPhysicsScale,
    gridBounds,
    gridCellsForSize,
    gridPositionForBounds,
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
} from "./fluid/grid-settings.js";
import { screenRay } from "./fluid/pick.js";
import { CAP_A, CAP_B, CAP_R, createCapsuleDemo } from "./fluid/scenes/capsule.js";
import { createBoxDemo } from "./fluid/scenes/box.js";
import { createFountainDemo } from "./fluid/scenes/fountain.js";
import { createMarbleTowerDemo } from "./fluid/scenes/marbleTower.js";
import { createWaterfallDemo, WATERFALL_ENV_URL, WATERFALL_BELFAST_ENV_URL } from "./fluid/scenes/waterfall.js";

// Particle count is chosen at runtime via the panel dropdown. The PBF rest
// density is pinned (see below) so the count scales the liquid VOLUME, not the
// packing density; MLS-MPM uses the same count so the two methods fill the tank
// comparably. Recreating the sims (createSims) is the only way to resize the
// GPU particle buffers, so the dropdown disposes and rebuilds both backends.
const PARTICLE_COUNTS = [40000, 80000, 120000, 150000, 200000, 300000, 500000, 750000, 1000000, 1200000, 1500000, 1800000, 2000000];
const DEFAULT_PARTICLE_COUNT = 80000;

// World-space radius of the interactive Shift+RMB push force (mouse-stir). Shared
// by every demo now the force is core-owned.
const FORCE_RADIUS = 3.5;
const clampScale = (s: number, lo: number, hi: number): number => Math.min(Math.max(s, lo), hi);
const PBMPM_MATERIALS = [
    { value: 0, label: "Liquid" },
    { value: 1, label: "Elastic (jelly)" },
    { value: 2, label: "Sand" },
    { value: 3, label: "Viscoelastic" },
];
// Material 2 = sand. Sand renders as opaque grainy spheres (no water surface) with no velocity
// brightening (uniform grains); its colour comes from the per-material sand preset.
const PBMPM_SAND_MATERIAL = 2;
// Only these demos expose the PB-MPM material selector (sand / jelly / viscoelastic need a container to
// hold their shape); every other demo is a liquid flow/jet showcase and is liquid-only.
const MATERIAL_DEMO_KEYS = ["box"];

async function main(): Promise<void> {
    const __initStart = performance.now();
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    const params = new URLSearchParams(location.search);
    const captureSeconds = Number(params.get("captureSeconds"));
    const captureFixedDt = Number(params.get("fixedDt"));
    const captureMode = Number.isFinite(captureSeconds) && captureSeconds > 0 && Number.isFinite(captureFixedDt) && captureFixedDt > 0;
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

    const cam = createArcRotateCamera(-Math.PI / 2, 1.1, 30, { x: 0, y: 6, z: 0 });
    cam.nearPlane = 0.1;
    cam.farPlane = 200;
    scene.camera = cam;
    // Shift+RMB is the "push the fluid" gesture in every demo; the camera must
    // ignore it (no pan) so the core's force-drag can own it. A demo may also claim
    // a plain pointerdown (capsule: LMB over the tank punches a hole) via
    // claimsPointer — the camera ignores those too. Everything else rotates (LMB) /
    // slides (RMB) / zooms (wheel) through the built-in arc control.
    const isForceGesture = (e: PointerEvent): boolean => e.button === 2 && e.shiftKey;
    attachControl(cam, canvas, scene, { shouldHandlePointerDown: (e) => !isForceGesture(e) && !activeDemo?.claimsPointer?.(e) });

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
    const SPAWN_MIN: [number, number, number] = [-2, 4, -2];
    const SPAWN_MAX: [number, number, number] = [2, 12, 2];
    const BOUNDS_MIN: [number, number, number] = [-20, 0, -20];
    const BOUNDS_MAX: [number, number, number] = [20, 20, 20];
    const defaultDomainBounds = (method: string, scale = 1): FluidDomainBounds => ({
        min: [-20 * scale, (method === "PBF" ? 0 : -1) * scale, -20 * scale],
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
    const GRID_CELLS_MAX = 2048;
    const GRID_CELL_COUNT_MAX = Math.floor(engine._device.limits.maxStorageBufferBindingSize / 16);
    const gridCellsForSettings = (grid: FluidGridSettings, method: string, physicsSize: number): [number, number, number] =>
        gridCellsForSize(grid.size, cellSizeForPhysicsScale(method, physicsSize));
    const gridAllocationError = (grid: FluidGridSettings, method: string, physicsSize: number): string | undefined => {
        const cells = gridCellsForSettings(grid, method, physicsSize);
        const oversizedAxis = cells.findIndex((value) => value > GRID_CELLS_MAX);
        if (oversizedAxis >= 0) {
            return `Grid size requires ${cells[oversizedAxis]!.toLocaleString()} cells on ${"XYZ"[oversizedAxis]} at the current Physics particle size; maximum is ${GRID_CELLS_MAX.toLocaleString()}.`;
        }
        const totalCells = cells[0] * cells[1] * cells[2];
        return totalCells > GRID_CELL_COUNT_MAX
            ? `Grid size requires ${totalCells.toLocaleString()} cells at the current Physics particle size; this device supports at most ${GRID_CELL_COUNT_MAX.toLocaleString()}.`
            : undefined;
    };

    // Gridless presets retain the historical hidden domain multiplier. Once a pair has an
    // explicit grid, its position/size are exact world units and Physics particle size alone
    // controls particle radius and cubic cell size.
    let domainScale = 1;
    let builtDomainScale = 1;
    let methodName = "PBF";
    let gridSettings: FluidGridSettings | undefined;
    let builtGridSettings: FluidGridSettings | undefined;
    let builtGridMethod = methodName;

    // Seed box scaled with the physics particle size. The fixed spawn box only
    // matches the rest density at 1×; at other sizes the seed is far under-dense
    // (small particles → violent collapse) or over-dense (big particles → eruption),
    // which shows up as white spray. Scaling the box half-extents with the size
    // holds the seed-to-rest density ratio ~constant so the fluid settles calmly at
    // every size. Clamped to stay inside the base tank footprint and above ground.
    function scaledSpawn(s: number): { min: [number, number, number]; max: [number, number, number] } {
        const cx = (SPAWN_MIN[0] + SPAWN_MAX[0]) / 2;
        const cy = (SPAWN_MIN[1] + SPAWN_MAX[1]) / 2;
        const cz = (SPAWN_MIN[2] + SPAWN_MAX[2]) / 2;
        const hx = Math.min(((SPAWN_MAX[0] - SPAWN_MIN[0]) / 2) * s, 4);
        const hz = Math.min(((SPAWN_MAX[2] - SPAWN_MIN[2]) / 2) * s, 4);
        const hy = Math.min(((SPAWN_MAX[1] - SPAWN_MIN[1]) / 2) * s, 7);
        const yMin = Math.max(1, cy - hy);
        const yMax = Math.min(17, cy + hy);
        return { min: [cx - hx, yMin, cz - hz], max: [cx + hx, yMax, cz + hz] };
    }

    // Backends are (re)built by createSims so the particle-count dropdown can
    // resize the GPU buffers (the only way to change count is to reallocate). The
    // capsule tank geometry seeds the sims' built-in fallback confinement (a legacy
    // default; the demo's injected sceneSdf always overrides it).
    function createSims(count: number, scale: number): { pbf: FluidSim; mpm: FluidSim; pbmpm: FluidSim } {
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
        const pbfScale = clampScale(scale, PBF_MIN_SCALE, PBF_MAX_SCALE);
        const mpmScale = clampScale(scale, MPM_MIN_SCALE, MPM_MAX_SCALE);
        const pbmpmScale = clampScale(scale, PBMPM_MIN_SCALE, PBMPM_MAX_SCALE);
        const ds = domainScale;
        const scaleTriple = (t: [number, number, number]): [number, number, number] => [t[0] * ds, t[1] * ds, t[2] * ds];
        const explicitGrid = gridSettings !== undefined;
        const explicitBounds = gridSettings ? gridBounds(gridSettings.position, gridSettings.size) : undefined;
        const cellSize = explicitGrid ? cellSizeForPhysicsScale(methodName, scale) : undefined;
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
        const pbfSpawn = scaledSpawn(pbfScale);
        const mpmSpawn = scaledSpawn(mpmScale);
        const pbmpmSpawn = scaledSpawn(pbmpmScale);
        // Backend 1 — Position Based Fluids (the original solver).
        const pbf = createPbfSim(engine, {
            count,
            particleRadius: 0.09 * pbfScale * (explicitGrid ? 1 : ds),
            smoothingRadius: cellSize ?? 0.4 * pbfScale * ds,
            spawnMin: explicitGrid ? pbfSpawn.min : scaleTriple(pbfSpawn.min),
            spawnMax: explicitGrid ? pbfSpawn.max : scaleTriple(pbfSpawn.max),
            capsuleA: CAP_A,
            capsuleB: CAP_B,
            capsuleRadius: CAP_R,
            groundY: 0,
            restDensity: 341 / (pbfScale * pbfScale * pbfScale),
            boundsMin: pbfBoundsMin,
            boundsMax: pbfBoundsMax,
            maxPerCell: 48,
            relaxation: 50 / (pbfScale * pbfScale),
        });

        // Backend 2 — MLS-MPM (grid-transfer; scales to far more particles).
        const mpm = createMlsMpmSim(engine, {
            count,
            particleRadius: 0.09 * mpmScale * (explicitGrid ? 1 : ds),
            spawnMin: explicitGrid ? mpmSpawn.min : scaleTriple(mpmSpawn.min),
            spawnMax: explicitGrid ? mpmSpawn.max : scaleTriple(mpmSpawn.max),
            capsuleA: CAP_A,
            capsuleB: CAP_B,
            capsuleRadius: CAP_R,
            groundY: 0,
            // Drop the MLS grid floor below every demo floor (box/fountain at y=0,
            // capsule bottom at y=2) so a demo floor is confined by its own scene
            // SDF (like the side walls) rather than the grid's 2-cell domain-border
            // v.y=0 zone — which coincided with y=0 and cancelled gravity there,
            // leaving the fluid hovering a row above the floor (PBF hard-clamps, so
            // it sat flush). The border now sits harmlessly below all demo floors.
            // The -1 floor offset scales with the domain too so the grid dims stay constant.
            boundsMin: mpmBoundsMin,
            boundsMax: mpmBoundsMax,
            dx: cellSize ?? 0.22 * mpmScale * ds,
            restDensity: 3,
            stiffness: 350,
            gravity: 9.8,
            viscosity: 0.3,
            substeps: 3,
            damping: 0.995,
            affineDamping: 0.9,
            groundDamp: 0.85,
            groundDampHeight: 1.5,
            activeBlocks: mpmActiveBlocks,
            pagedGrid: mpmPagedGrid,
            pagedGridMaxPages: mpmPagedGridMaxPages,
            fusedBlockDiscovery: mpmFusedBlockDiscovery,
            onPagedGridOverflow: (requiredPages, capacity) => {
                const message = `Page capacity exceeded: ${requiredPages.toLocaleString()} required, ${capacity.toLocaleString()} allocated. Increase Page capacity.`;
                controls.setPagedGridStatus(message, true);
                canvas.dataset.pagedGridOverflow = "true";
                console.error(`[MLS-MPM] ${message}`);
            },
        });

        // Backend 3 — Position-Based MPM (liquid-only PB-MPM phase 1).
        const pbmpm = createPbMpmSim(engine, {
            count,
            particleRadius: 0.09 * pbmpmScale * (explicitGrid ? 1 : ds),
            spawnMin: explicitGrid ? pbmpmSpawn.min : scaleTriple(pbmpmSpawn.min),
            spawnMax: explicitGrid ? pbmpmSpawn.max : scaleTriple(pbmpmSpawn.max),
            groundY: 0,
            boundsMin: mpmBoundsMin,
            boundsMax: mpmBoundsMax,
            dx: cellSize ?? 0.22 * pbmpmScale * ds,
            gravity: 9.8,
            substeps: 3,
            iterations: 5,
            liquidRelaxation: 1.5,
            liquidViscosity: 0.01,
            elasticityRatio: 0.3,
            elasticRelaxation: 0.3,
            frictionAngle: 35,
            plasticity: 0.8,
            material: pbmpmMaterial,
            restitution: 0,
        });

        return { pbf, mpm, pbmpm };
    }

    let particleCount = DEFAULT_PARTICLE_COUNT;
    let physicsScale = 1; // physics particle-size multiplier (rebuilds sims)
    let pbmpmMaterial = 0;
    let mpmActiveBlocks = false;
    let mpmPagedGrid = false;
    const maxPagedGridPages = Math.max(1, Math.floor(engine._device.limits.maxStorageBufferBindingSize / 1024) - 1);
    let mpmPagedGridMaxPages = Math.min(maxPagedGridPages, Math.max(1000, Math.round((DEFAULT_PARTICLE_COUNT * 27 * 1.5) / 64000) * 1000));
    let mpmFusedBlockDiscovery = false;
    let { pbf: pbfSim, mpm: mpmSim, pbmpm: pbmpmSim } = createSims(particleCount, physicsScale);
    let activeSim: FluidSim = pbfSim;
    let quality: Quality = DEFAULT_QUALITY; // low/middle/high preset tier (panel dropdown)
    /** Demos the user has already opened once, so FluidDemo.defaultMethod/defaultQuality are
     *  honoured on the first visit only. Seeded with the start-up demo below. */
    const visitedDemos = new Set<string>();
    function simForMethod(name: string): FluidSim {
        if (name === "PBF") {
            return pbfSim;
        }
        if (name === "PB-MPM") {
            return pbmpmSim;
        }
        return mpmSim;
    }
    let pbmpmMaterialRow: HTMLElement | null = null;
    let pbmpmMaterialSel: HTMLSelectElement | null = null;
    function refreshPbMpmMaterialUi(): void {
        if (pbmpmMaterialRow) {
            // Material selector only for PB-MPM on a material-capable demo (the box container).
            const show = methodName === "PB-MPM" && !!activeDemo && MATERIAL_DEMO_KEYS.includes(activeDemo.key);
            pbmpmMaterialRow.style.display = show ? "block" : "none";
        }
        if (pbmpmMaterialSel) {
            pbmpmMaterialSel.value = String(pbmpmMaterial);
        }
    }
    // Interactive push force (Shift+RMB mouse-stir): a ready-made injectable force
    // field shared by both backends. The frame loop drives it via setRay + toggles
    // it on the ACTIVE sim via setForceField, so it dispatches its own dedicated
    // compute pass ONLY while a push is active (and compiles lazily on first use).
    const rayForce = createRayForce(engine._device);
    // The active demo (capsule / box / fountain). Assigned by the first switchPair;
    // every consumer below runs only after that, so the `!` reads are safe.
    let activeDemo: FluidDemo | null = null;

    // Composite target for the whole fluid chain. The surface / foam / container-overlay
    // passes write HERE instead of straight to the swapchain, so a post-process stage can
    // read the finished frame and present it. `size: engine` keeps it canvas-sized across
    // resizes, and the format matches the swapchain so the final pass is a 1:1 resample.
    const postRT = createRenderTarget({ lbl: "fluid-post-color", format: engine.format, samples: 1, size: engine });

    const particleTask = createParticleRenderTask(engine, scene, { colorRT: sceneColorRT, depthRT, camera: cam, sim: activeSim });
    addTask(scene, particleTask);
    // Fluid surface renderer + frame compositor: reads the offscreen scene colour
    // and writes the composite target. In sphere mode it just blits the scene (with the
    // impostors already drawn into it); in surface mode it reconstructs and
    // shades the liquid surface (refraction of the scene).
    const surfaceTask = createFluidSurfaceTask(engine, scene, { bgRT: sceneColorRT, outRT: postRT, depthRT, camera: cam, sim: activeSim });
    addTask(scene, surfaceTask);

    // Foam (diffuse-particle) renderer — draws the active sim's spray/foam/bubble pool
    // as sprites OVER the composited fluid surface (added after surfaceTask), depth-
    // tested against the shared scene depth so opaque geometry occludes it. Only the
    // PBF backend generates foam for now (setFoam is a no-op / undefined on MLS-MPM).
    const foamTask = createFoamRenderTask(engine, scene, {
        colorRT: postRT,
        depthRT,
        camera: cam,
        sim: activeSim,
        // Screen-space foam reads the fluid-surface eye-Z for surface occlusion +
        // submerged classification. Fetched per-frame (surfaceTask reallocates it
        // on resize / half-res), null in sphere/blit mode → foam falls back to "no
        // water" (spray/foam on top, no submerged bubbles).
        getSurfaceDepth: () => surfaceTask.surfaceDepthView(),
    });
    addTask(scene, foamTask);

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
                extraTextureWGSL: "@group(0) @binding(2) var bloomBackground:texture_2d<f32>;",
                uniformWGSL: "struct P{threshold:f32,fluidEps:f32,p0:f32,p1:f32}\n@group(0) @binding(3) var<uniform> bloomExtractParams:P;",
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
                fragmentWGSL: `fn applyPostProcess(color:vec4f, uv:vec2f)->vec4f{
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
                extraTextureWGSL: "@group(0) @binding(2) var bloomBlur:texture_2d<f32>;",
                uniformWGSL: "struct M{weight:f32,p0:f32,p1:f32,p2:f32}\n@group(0) @binding(3) var<uniform> bloomMergeParams:M;",
                uniformBinding: 3,
                uniformByteLength: 16,
                writeUniforms(data) {
                    // Weight 0 makes the merge an exact passthrough, which is what "bloom off"
                    // is: this task ALWAYS runs because it is the single owner of the swapchain.
                    data[0] = bloomEnabled ? bloomParams.intensity : 0;
                },
                fragmentWGSL: `fn applyPostProcess(color:vec4f, uv:vec2f)->vec4f{
let b=textureSampleLevel(bloomBlur,sourceSampler,clamp(uv,vec2f(0),vec2f(1)),0).rgb;
return vec4f(color.rgb+b*bloomMergeParams.weight,color.a);}`,
            },
        },
        engine,
        scene
    );
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
    let msaaSceneTask: Task | null = null;
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
            msaaSceneTask = createRenderTask({ name: "scene-msaa", rt: sceneMsaaRT, rst: sceneColorRT, clr: false }, engine, scene);
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
        return controls.getValues().foam.enabled && !!activeSim.setFoam && !surfaceDebugActive;
    }
    function pushFoam(): void {
        const f = controls.getValues().foam;
        const cfg: FoamConfig = {
            activeParticles: f.activeParticles,
            kTa: f.kTa,
            kWc: f.kWc,
            kb: f.kb,
            kd: f.kd,
            tMin: f.tMin,
            tMax: f.tMax,
            poolScale: f.poolScale,
        };
        activeSim.setFoam?.(f.enabled ? cfg : null);
        foamTask.setEnabled(foamRenderVisible());
    }
    // Re-apply after any sim rebuild / backend switch: (re)enable foam on the new active
    // sim (allocates its pool) THEN rebind the renderer to that pool.
    function applyFoam(): void {
        pushFoam();
        foamTask.setSim(activeSim);
    }

    // ── Opt-in GPU timing (see ./fluid/gpu-profiler.ts) ──────────────────────
    // The profiler is created lazily the first time the user enables timing; if the
    // headless/host GPU lacks the "timestamp-query" feature we never create it and
    // the UI shows a graceful fallback. When enabled it is wired to BOTH sims and all
    // three render tasks so every fluid pass is tagged with timestampWrites; when off
    // we wire null everywhere so there is zero timing cost.
    const timingSupported = engine._device.features.has("timestamp-query");
    let profiler: FluidProfilerImpl | null = null;
    let timingEnabled = false;
    // Push the profiler (or null) to both backends + every render task. Re-called after
    // a sim rebuild / backend switch (the sims are recreated; the tasks persist).
    function applyProfiler(): void {
        const p = timingEnabled ? profiler : null;
        pbfSim.setProfiler?.(p);
        mpmSim.setProfiler?.(p);
        pbmpmSim.setProfiler?.(p);
        particleTask.setProfiler(p);
        surfaceTask.setProfiler(p);
        foamTask.setProfiler(p);
    }
    // Timing is always on when the GPU supports timestamp-query: create + wire the
    // profiler unconditionally at startup (no opt-in checkbox). If the feature is
    // missing (or profiler creation throws) timing stays off and the GPU panel shows
    // a fallback note instead of the per-stage rows.
    if (timingSupported) {
        try {
            profiler = createFluidProfiler(engine._device);
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
            if (timingEnabled && profiler) {
                // Close the whole-frame envelope (frameStop) AFTER every other pass, then
                // resolve. The envelope's end-to-start span is the frame's total GPU time.
                profiler.frameStop(engine._currentEncoder);
                profiler.resolveInto(engine._currentEncoder);
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
    ]).then(([studio, sky]) => {
        // The loaders disagree on tone mapping (loadEnvironment enables it, loadHdrEnvironment
        // disables it) and Promise.all gives no ordering guarantee, so pin it explicitly.
        scene.imageProcessing.toneMappingEnabled = true;
        studioSlot = studio ? makeSlot(studio, 1.0, 1.1) : null;
        // `sky` is the waterfall's own env (WATERFALL_ENV_URL). Take its grade from the matching
        // picker entry rather than repeating the numbers, so the eager slot and the picker can
        // never disagree about how the same map is exposed.
        skySlot = sky ? makeSlot(sky, EAGER_SKY?.exposure ?? 0.8, EAGER_SKY?.contrast ?? 1.15) : null;
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
    function makeSlot(env: EnvironmentTextures, exposure: number, contrast: number): EnvSlot {
        scene.imageProcessing.exposure = exposure;
        scene.imageProcessing.contrast = contrast;
        return { env, sky: buildHdrSkyboxRenderable(scene, env, 10, [0, 0, 0], [0, 0, 0]), exposure, contrast };
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
            const slot = makeSlot(env, choice.exposure, choice.contrast);
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
            scene._renderables.push(slot.sky);
            scene._renderableVersion++;
            activeSky = slot.sky;
        }
        surfaceTask.setEnvMap({ view: slot.env._specularCubeView, sampler: slot.env._cubeSampler });
    }

    // On-screen FPS accumulators (smoothed over ~0.5 s windows). The FPS read-out element
    // itself lives in the shared controls panel's top-left GPU section (controls.gpu).
    let fpsAccumMs = 0;
    let fpsFrames = 0;

    // Mouse-force state. The Shift+RMB drag sets `pendingForce`; the frame loop
    // applies it for one frame then clears it, so the force only acts while the
    // mouse is actually moving.
    let pendingForce: PendingForce | null = null;

    // Shift+RMB "push the fluid" drag state (core-owned; works in every demo).
    // While this is active the camera is disengaged — its pointerdown gate rejected
    // the gesture (isForceGesture) — so only the force applies, no camera pan.
    let forceDragging = false;
    let forceLastX = 0;
    let forceLastY = 0;
    let forceLastT = 0;

    // Push the active demo's emitters to BOTH backends (fountain live-tuning).
    function applyEmitters(): void {
        const emit = activeDemo!.emitters();
        pbfSim.setEmitters(emit);
        mpmSim.setEmitters(emit);
        pbmpmSim.setEmitters(emit);
    }

    // Inject the active demo's scene SDF, emitters and spawn into both sims.
    // Re-applied after any sim rebuild. The demo packs its own params into the UBO
    // (offset 0); the hole ring (offset 32) is managed here.
    // Push the active demo's seed volume + warm-up to every backend. Split out of
    // applySceneSdf so a demo whose spawn volume resolves asynchronously can re-push it
    // (see ctx.refreshSpawn) without redoing the whole SDF/emitter apply.
    function applySpawn(): void {
        const s = activeDemo!.spawn();
        pbfSim.setSpawn(s.min, s.max, s.accept);
        mpmSim.setSpawn(s.min, s.max, s.accept);
        pbmpmSim.setSpawn(s.min, s.max, s.accept);
        // Per-demo start-of-sim warm-up (PB-MPM no-ops it). Applied on the next
        // reset()/seed() — switchPair resets the active sim right after this.
        pbfSim.setWarmup?.(s.warmupFrames ?? 0);
        mpmSim.setWarmup?.(s.warmupFrames ?? 0);
        pbmpmSim.setWarmup?.(s.warmupFrames ?? 0);
    }

    function applySceneSdf(): void {
        const demo = activeDemo!;
        demo.writeSdfParams();
        pbfSim.setSceneSdf(demo.sdf);
        mpmSim.setSceneSdf(demo.sdf);
        pbmpmSim.setSceneSdf(demo.sdf);
        const emit = demo.emitters();
        pbfSim.setEmitters(emit);
        mpmSim.setEmitters(emit);
        pbmpmSim.setEmitters(emit);
        applySpawn();
    }

    // ── Live tuning UI ───────────────────────────────────────────────
    // A combo box to switch method, per-method parameter sliders (applied live),
    // and a reset button. The per-method schema (each entry mirrors the sim's
    // current default and remembers the last value the user set) is shared with the
    // other fluid apps via `DEFAULT_FLUID_SCHEMAS` (fluid/controls-panel).

    // The shared control panel (GENERAL / RENDER / FOAM / PHYSICS sections + the
    // pinned top-left GPU-timing panel) is built by the reusable component
    // (./fluid/controls-panel.ts); this demo owns only the scene-specific "Demo"
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
            switchPair(demo, (first && demo.defaultMethod) || methodName, (first && demo.defaultQuality) || quality);
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
            switchPair(activeDemo, methodName, qualitySel.value as Quality);
        }
    };
    // Demo + quality dropdowns, side by side, at the top of the "Demo" section.
    const demoQualityRow = document.createElement("div");
    demoQualityRow.style.cssText = "display:flex;gap:6px;margin-bottom:8px;";
    demoQualityRow.append(containerSel, qualitySel);
    pbmpmMaterialRow = document.createElement("div");
    pbmpmMaterialRow.style.cssText = "display:none;margin:2px 0 8px;";
    const pbmpmMaterialLabel = document.createElement("div");
    pbmpmMaterialLabel.textContent = "Material";
    pbmpmMaterialLabel.style.cssText = "font-weight:600;margin-bottom:6px;";
    pbmpmMaterialSel = document.createElement("select");
    pbmpmMaterialSel.style.cssText = DEMO_SEL_CSS;
    for (const mat of PBMPM_MATERIALS) {
        const opt = document.createElement("option");
        opt.value = String(mat.value);
        opt.textContent = mat.label;
        pbmpmMaterialSel.appendChild(opt);
    }
    pbmpmMaterialSel.value = String(pbmpmMaterial);
    pbmpmMaterialSel.onchange = () => {
        // Each PB-MPM material is its own pair (physics + render + colour differ), so switching material
        // snapshots the current material's state and loads the target material's preset/state — exactly
        // like switching quality. The render mode + colour then come from that material's preset.
        const m = parseInt(pbmpmMaterialSel!.value, 10);
        switchPair(activeDemo!, methodName, quality, m);
    };
    pbmpmMaterialRow.append(pbmpmMaterialLabel, pbmpmMaterialSel);

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
    // reflections together: `scene.envRotationY` is repacked into the scene UBO every frame
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
        scene.envRotationY = rad;
        surfaceTask.setEnvRotationY(rad);
    };
    envRotInput.oninput = () => applyEnvRotation(parseFloat(envRotInput.value));
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
    envIntInput.oninput = () => applyEnvIntensity(parseFloat(envIntInput.value));
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

    const controls = createFluidControlsPanel({
        schemas: DEFAULT_FLUID_SCHEMAS,
        methods: Object.keys(DEFAULT_FLUID_SCHEMAS),
        particleCounts: PARTICLE_COUNTS,
        showActiveBlocks: true,
        showGridControls: true,
        physScaleMin: PHYS_MIN_SCALE,
        physScaleMax: PHYS_MAX_SCALE,
        initial: {
            method: methodName,
            count: DEFAULT_PARTICLE_COUNT,
            physScale: physicsScale,
            gridPosition: [...initialGridSettings.position],
            gridSize: [...initialGridSettings.size],
            cellSize: cellSizeForPhysicsScale(methodName, physicsScale) * domainScale,
            color: "#16a3c3", // matches the default FLUID_COLOR
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
            debug: "none",
            showContainer: true,
            activeBlocks: false,
            pagedGrid: false,
            pagedGridMaxPages: mpmPagedGridMaxPages,
            fusedBlockDiscovery: false,
            foam: {
                enabled: false,
                activeParticles: false,
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
        gpu: { stages: ["Simulation", "Foam gen", "Surface", "Foam render", "Particles"], supported: profiler !== null },
        on: {
            onMethod: (name) => switchPair(activeDemo!, name),
            onParticleCount: (n) => setParticleCount(n),
            onRenderMode: (spheres) => applyRenderMode(spheres),
            onColor: (rgb) => {
                surfaceTask.setFluidColor(rgb);
                particleTask.setTint(rgb);
            },
            onAbsorption: (v) => surfaceTask.setAbsorption(v),
            onParticleSize: (s) => {
                surfaceTask.setSizeScale(s);
                particleTask.setSizeScale(s);
            },
            onRefraction: (v) => surfaceTask.setRefractionStrength(v),
            onSpecular: (v) => surfaceTask.setSpecularPower(v),
            onReflection: (exposure, contrast) => surfaceTask.setEnvReflection(exposure, contrast),
            onReflectivity: (v) => surfaceTask.setFresnelF0(v),
            onDepthBlur: (size, threshold) => surfaceTask.setDepthBlur(size, threshold),
            onThicknessBlur: (v) => surfaceTask.setThicknessBlur(v),
            onHalf: (on) => surfaceTask.setHalfRender(on),
            onSurfaceFilter: (m) => surfaceTask.setSurfaceFilter(m),
            onNarrowRange: (delta, mu) => surfaceTask.setNarrowRange(delta, mu),
            onAnisotropic: (v) => {
                surfaceTask.setAnisotropic(v);
                renderAnisotropic = v;
                applyEffectiveRenderMode();
            },
            onAnisotropySurfScale: (v) => surfaceTask.setAnisotropySurfScale(v),
            onThicknessDownscale: (v) => surfaceTask.setThicknessDownscale(v),
            onShowContainer: (visible) => activeDemo?.setContainerVisible?.(visible),
            onDebug: (mode) => {
                surfaceTask.setDebug(mode);
                // Hide foam sprites while a surface debug texture is shown (they composite over it).
                surfaceDebugActive = mode !== "none";
                foamTask.setEnabled(foamRenderVisible());
            },
            onPhysicsParam: (k, v) => applyParam(activeSim, k, v),
            onPhysScale: (s) => setPhysicsScale(s),
            onGridSettings: (position, size) => setGridSettings({ position, size }),
            onActiveBlocks: (enabled) => {
                if (enabled === mpmActiveBlocks) return;
                mpmActiveBlocks = enabled;
                rebuildSims(particleCount, physicsScale);
            },
            onPagedGrid: (enabled) => {
                if (enabled === mpmPagedGrid) return;
                mpmPagedGrid = enabled;
                controls.setPagedGridStatus("");
                canvas.dataset.pagedGridOverflow = "false";
                rebuildSims(particleCount, physicsScale);
            },
            onPagedGridMaxPages: (pages) => {
                const clampedPages = Math.min(pages, maxPagedGridPages);
                controls.setPagedGridMaxPages(clampedPages);
                if (clampedPages === mpmPagedGridMaxPages) return;
                mpmPagedGridMaxPages = clampedPages;
                controls.setPagedGridStatus("");
                canvas.dataset.pagedGridOverflow = "false";
                if (mpmPagedGrid) {
                    rebuildSims(particleCount, physicsScale);
                }
            },
            onFusedBlockDiscovery: (enabled) => {
                if (enabled === mpmFusedBlockDiscovery) return;
                mpmFusedBlockDiscovery = enabled;
                rebuildSims(particleCount, physicsScale);
            },
            onReset: () => {
                activeSim.reset();
                clearSceneHoles();
            },
            onFoamEnable: () => pushFoam(),
            onFoamActiveParticles: () => pushFoam(),
            onFoamKta: () => {
                if (controls.getValues().foam.enabled) {
                    pushFoam();
                }
            },
            onFoamKwc: () => {
                if (controls.getValues().foam.enabled) {
                    pushFoam();
                }
            },
            onFoamLifetime: () => {
                if (controls.getValues().foam.enabled) {
                    pushFoam();
                }
            },
            onFoamBuoyancy: () => {
                if (controls.getValues().foam.enabled) {
                    pushFoam();
                }
            },
            onFoamDrag: () => {
                if (controls.getValues().foam.enabled) {
                    pushFoam();
                }
            },
            onFoamPool: () => {
                if (controls.getValues().foam.enabled) {
                    pushFoam(); // reallocates the ring on change
                }
            },
            onFoamThresholds: (t0, t1) => foamTask.setThresholds(t0, t1),
            onFoamSubsurface: (v) => foamTask.setSubsurfaceStrength(v),
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

    // Prepend the scene-specific "Demo" section (scene dropdown + demo params + the
    // container-visibility toggle) into the component's demo slot.
    controls.demoSlot.append(
        ...controls.makeSection("Demo", [demoQualityRow, envRow, envRotRow, envIntRow, msaaRow, pbmpmMaterialRow, demoParamsHost, controls.containerToggleRow!])
    );

    // ── Export parameters ────────────────────────────────────────────────────
    // Serialise the FULL current parameter set (pair state + render mode + surface +
    // foam) to a pretty-printed JSON download. Meant to seed a default (demo, method)
    // preset, so it must be complete + self-describing. Import is intentionally NOT
    // implemented yet (export only).
    const exportBtn = document.createElement("button");
    exportBtn.textContent = "Export parameters";
    exportBtn.style.cssText = "width:100%;padding:5px;cursor:pointer;background:#26415f;color:#eef3f8;border:1px solid #3a567a;border-radius:4px;";
    function exportParameters(): void {
        // Serialise the live UI into the shared grouped shape (the same format the on-disk
        // quality presets use), so an exported file can be dropped straight into presets/.
        const data = exportJsonFromPairState(activeDemo!.key, methodName, readLivePairState(methodName));
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
    controls.root.append(...controls.makeSection("Export", [exportBtn]));

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
                input.oninput = () => {
                    const v = parseFloat(input.value);
                    val.textContent = String(v);
                    onChange(p.key, v);
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
                input.oninput = () => onChange(p.key, input.value);
                row.append(lab, input);
                host.appendChild(row);
            }
        }
    }
    function refreshDemoParams(): void {
        const params = activeDemo!.demoParams();
        const extras = activeDemo!.extraControls();
        if (!params.some((p) => !p.hidden) && extras.length === 0) {
            const none = document.createElement("div");
            none.textContent = "No tunable parameters for this demo.";
            none.style.cssText = "color:#7c8aa0;font-size:12px;margin:2px 0;";
            demoParamsHost.replaceChildren(none);
            return;
        }
        buildDemoParamsUI(demoParamsHost, params, (k, v) => activeDemo!.applyParam(k, v));
        for (const el of extras) {
            demoParamsHost.appendChild(el);
        }
    }

    // Apply a solver parameter, folding in the physics particle-size coupling.
    // The sliders expose the base (1×) values; PBF's restDensity and constraint
    // relaxation must additionally track the particle scale (restDensity ∝
    // 1/scale³, relaxation ∝ 1/scale²) — using the same floored scale createSims
    // built the grid with — or the rescaled fluid leaves its stable regime. MLS-MPM
    // couples purely through its cell size dx (set at creation), so no fold-in.
    function applyParam(sim: FluidSim, key: string, value: number): void {
        let v = value;
        if (methodName === "PBF") {
            const s = clampScale(physicsScale, PBF_MIN_SCALE, PBF_MAX_SCALE);
            if (key === "restDensity") v = value / (s * s * s);
            else if (key === "relaxation") v = value / (s * s);
        }
        sim.setParam(key, v);
    }

    function applyMethod(name: string): void {
        activeSim = simForMethod(name);
        methodName = name;
        if (name === "PB-MPM") {
            pbmpmSim.setMaterial?.(pbmpmMaterial);
        }
        // Sand renders as uniformly-coloured grains (no velocity brightening); water/jelly keep the
        // speed-based highlight. This is a render characteristic derived from the material — NOT forced
        // here on the colour or render MODE (those come from the per-material preset / pair state, so a
        // count-change re-applyMethod doesn't reset a user's colour).
        particleTask.setVelocityBrighten(name === "PB-MPM" && pbmpmMaterial === PBMPM_SAND_MATERIAL ? 0 : 1);
        for (const [key, value] of Object.entries(controls.getPhysicsValues(name))) {
            applyParam(activeSim, key, value);
        }
        activeSim.reset();
        clearSceneHoles();
        particleTask.setSim(activeSim);
        surfaceTask.setSim(activeSim);
        applyFoam(); // rebind the foam renderer + (re)enable foam on the new active sim
        applyProfiler(); // re-wire the GPU timing hook onto the rebuilt sims (tasks persist)
        controls.setMethod(name); // sync the method dropdown + component's current method (no side effect)
        controls.rebuildPhysics(name); // rebuild the physics-slider block for the new method
        // Show only the physics sliders the current method/material uses (PB-MPM branches on material).
        controls.setVisiblePhysicsParams(name === "PB-MPM" ? pbmpmParamKeysForMaterial(pbmpmMaterial) : null);
        refreshPbMpmMaterialUi();
        canvas.dataset.method = methodName;
        canvas.dataset.activeBlocks = mpmActiveBlocks ? "true" : "false";
        canvas.dataset.pagedGrid = mpmPagedGrid ? "true" : "false";
        canvas.dataset.fusedBlockDiscovery = mpmFusedBlockDiscovery ? "true" : "false";
    }
    // Toggle between the sphere-impostor renderer and the screen-space surface.
    // Default is the fluid surface; the checkbox switches to spheres.
    //
    // "Render as spheres" and "Anisotropic surface" are independent toggles whose COMBINATION
    // selects what gets drawn, so the effective state is resolved in ONE place (called from
    // both applyRenderMode and onAnisotropic) to keep the sphere task and surface mode in sync.
    let renderSpheres = false;
    let renderAnisotropic = false;
    function applyEffectiveRenderMode(): void {
        if (renderSpheres && renderAnisotropic) {
            // Both ON: inspection view — show the true anisotropic ellipsoids as opaque lit
            // splats (the opaque sphere task is disabled so it doesn't overlap them).
            particleTask.setEnabled(false);
            surfaceTask.setMode("ellipsoidDebug");
            canvas.dataset.render = "ellipsoids";
        } else if (renderSpheres) {
            particleTask.setEnabled(true);
            surfaceTask.setMode("blit");
            canvas.dataset.render = "spheres";
        } else {
            // Surface view. Anisotropy (when on) still shapes the surface itself, as before.
            particleTask.setEnabled(false);
            surfaceTask.setMode("surface");
            canvas.dataset.render = "surface";
        }
    }
    function applyRenderMode(spheres: boolean): void {
        renderSpheres = spheres;
        applyEffectiveRenderMode();
    }

    // Resize the particle buffers by disposing and rebuilding both backends at
    // the new count, then re-applying the current demo and method (which re-seeds
    // and rebinds the renderer).
    function setParticleCount(n: number): void {
        if (n === particleCount) {
            return;
        }
        rebuildSims(n, physicsScale);
    }

    function syncGridControls(): void {
        const effectiveGrid = gridSettings ?? defaultGridSettings(methodName, domainScale);
        const cellSize = cellSizeForPhysicsScale(methodName, physicsScale) * (gridSettings ? 1 : domainScale);
        const cells = gridCellsForSize(effectiveGrid.size, cellSize);
        controls.setGridSettings([...effectiveGrid.position], [...effectiveGrid.size], cellSize);
        canvas.dataset.gridPosition = effectiveGrid.position.join(",");
        canvas.dataset.gridSize = effectiveGrid.size.join(",");
        canvas.dataset.gridCells = cells.join(",");
        canvas.dataset.gridCellSize = String(cellSize);
        canvas.dataset.gridExplicit = String(gridSettings !== undefined);
    }

    function setPhysicsScale(s: number): void {
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
        rebuildSims(particleCount, nextScale);
    }

    function setGridSettings(next: FluidGridSettings): string | void {
        if (!validGridSettings(next)) {
            return "Grid position must be finite and Grid size must contain positive finite world-space dimensions.";
        }
        const allocationError = gridAllocationError(next, methodName, physicsScale);
        if (allocationError) {
            return allocationError;
        }
        if (gridSettingsEqual(next, gridSettings)) {
            syncGridControls();
            return;
        }
        gridSettings = cloneGridSettings(next);
        rebuildSims(particleCount, physicsScale);
    }

    // Rebuild all sims at a new particle count, physics size and active grid.
    function rebuildSims(count: number, scale: number): void {
        particleCount = count;
        physicsScale = scale;
        pbfSim.dispose();
        mpmSim.dispose();
        pbmpmSim.dispose();
        ({ pbf: pbfSim, mpm: mpmSim, pbmpm: pbmpmSim } = createSims(count, scale));
        builtGridSettings = gridSettings ? cloneGridSettings(gridSettings) : undefined;
        builtGridMethod = methodName;
        builtDomainScale = domainScale; // sims are now built at the current domain scale
        applySceneSdf();
        applyMethod(methodName);
        syncGridControls();
        canvas.dataset.particleCount = String(count);
        controls.setParticleCount(count); // sync the Particles dropdown (no rebuild re-entry)
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
        half: initialValues.half,
        thicknessDownscale: initialValues.thicknessDownscale,
        absorption: initialValues.absorption,
        size: initialValues.size,
        physScale: physicsScale,
        count: particleCount,
        renderMode: initialValues.renderMode,
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
    };
    // Pristine foam look, snapshotted before any interaction; seeds the foam block of
    // every preset-less pair so foam becomes per-(demo, method) (restored on switch).
    const FOAM_DEFAULTS: NonNullable<PairState["foam"]> = { ...initialValues.foam };
    // Per-demo default demo-param bag (numeric params only), snapshotted from each
    // demo's initial descriptors below once the registry is built.
    const DEMO_PARAM_DEFAULTS: Record<string, Record<string, number>> = {};

    const pairStates = new Map<string, PairState>();
    let currentPairKey: string | null = null;

    function defaultPairState(demo: FluidDemo, method: string): PairState {
        return {
            schema: { ...SCHEMA_DEFAULTS[method]! },
            demoParams: { ...(DEMO_PARAM_DEFAULTS[demo.key] ?? {}) },
            color: RENDER_DEFAULTS.color,
            half: RENDER_DEFAULTS.half,
            thicknessDownscale: RENDER_DEFAULTS.thicknessDownscale,
            absorption: RENDER_DEFAULTS.absorption,
            size: RENDER_DEFAULTS.size,
            physScale: RENDER_DEFAULTS.physScale,
            count: RENDER_DEFAULTS.count,
            material: method === "PB-MPM" ? 0 : undefined,
            renderMode: RENDER_DEFAULTS.renderMode,
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
            pagedGrid: method === "MLS-MPM" ? false : undefined,
            pagedGridMaxPages: method === "MLS-MPM" ? Math.max(1000, Math.round((RENDER_DEFAULTS.count * 27 * 1.5) / 64000) * 1000) : undefined,
            fusedBlockDiscovery: method === "MLS-MPM" ? false : undefined,
            foam: { ...FOAM_DEFAULTS },
            showContainer: true,
        };
    }
    // First-visit state: the on-disk quality preset for this (demo, method, quality),
    // if any, merged over the core defaults. Pairs with no file use pure defaults.
    function presetOrDefault(demo: FluidDemo, method: string, q: Quality, material = 0): PairState {
        const base = defaultPairState(demo, method);
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
            color: p.color ?? base.color,
            half: p.half ?? base.half,
            thicknessDownscale: p.thicknessDownscale ?? base.thicknessDownscale,
            absorption: p.absorption ?? base.absorption,
            size: p.size ?? base.size,
            physScale: p.physScale ?? base.physScale,
            grid: presetGrid,
            count: p.count ?? base.count,
            material: p.material ?? base.material,
            camera: p.camera ?? base.camera,
            renderMode: p.renderMode ?? base.renderMode,
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
        const demoParams: Record<string, number> = {};
        for (const p of activeDemo!.demoParams()) {
            if (p.type === "number") demoParams[p.key] = p.value;
        }
        return {
            schema: controls.getPhysicsValues(method),
            demoParams,
            color: v.color,
            half: v.half,
            thicknessDownscale: v.thicknessDownscale,
            absorption: v.absorption,
            size: v.size,
            physScale: physicsScale,
            grid: gridSettings ? cloneGridSettings(gridSettings) : undefined,
            count: particleCount,
            material: method === "PB-MPM" ? pbmpmMaterial : undefined,
            camera: { alpha: cam.alpha, beta: cam.beta, radius: cam.radius },
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
            activeBlocks: method === "MLS-MPM" ? v.activeBlocks : undefined,
            pagedGrid: method === "MLS-MPM" ? v.pagedGrid : undefined,
            pagedGridMaxPages: method === "MLS-MPM" ? v.pagedGridMaxPages : undefined,
            fusedBlockDiscovery: method === "MLS-MPM" ? v.fusedBlockDiscovery : undefined,
            foam: v.foam,
            demoState: activeDemo!.snapshotState?.() ?? {},
            showContainer: v.showContainer,
            envIntensity,
            msaa: msaaOn,
        };
    }
    // Push a PairState into the live params + UI, then apply it (rebuild the sims
    // when the count/scale differ from what's currently built; otherwise re-apply
    // in place). The component's setters re-render the controls to match; refreshDemoParams
    // re-renders the demo section.
    function loadPairState(st: PairState): void {
        const demo = activeDemo!;
        const nextGridSettings = st.grid ? cloneGridSettings(st.grid) : undefined;
        if (nextGridSettings && !validGridSettings(nextGridSettings)) {
            throw new Error("Invalid fluid grid: position must be finite and size must be positive.");
        }
        const [minScale, maxScale] = scaleLimitsForMethod(methodName);
        const nextPhysicsScale = Math.min(maxScale, Math.max(minScale, Math.round(st.physScale * 100) / 100));
        if (nextGridSettings) {
            const allocationError = gridAllocationError(nextGridSettings, methodName, nextPhysicsScale);
            if (allocationError) {
                throw new Error(allocationError);
            }
        }
        // Sync the component's current method BEFORE pushing the physics values so
        // setPhysics targets the target method's slider block (methodName is already the
        // target method here — switchPair set it before calling loadPairState).
        controls.setMethod(methodName);
        controls.setPhysics(st.schema);
        if (typeof st.material === "number") {
            pbmpmMaterial = st.material;
        }
        refreshPbMpmMaterialUi();
        for (const k of Object.keys(st.demoParams)) {
            demo.applyParam(k, st.demoParams[k]!);
        }
        controls.setColor(st.color);
        controls.setHalf(st.half);
        controls.setThicknessDownscale(st.thicknessDownscale);
        controls.setParticleSize(st.size);
        controls.setAbsorption(st.absorption);
        // Surface render mode + shading sliders (each optional; only apply if present).
        if (st.renderMode !== undefined) {
            controls.setRenderMode(st.renderMode === "spheres");
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
        const nextPagedGrid = methodName === "MLS-MPM" ? (st.pagedGrid ?? false) : mpmPagedGrid;
        const nextActiveBlocks = methodName === "MLS-MPM" ? nextPagedGrid || (st.activeBlocks ?? false) : mpmActiveBlocks;
        const nextPagedGridMaxPages = Math.min(
            maxPagedGridPages,
            methodName === "MLS-MPM" ? (st.pagedGridMaxPages ?? Math.max(1000, Math.round((st.count * 27 * 1.5) / 64000) * 1000)) : mpmPagedGridMaxPages
        );
        const nextFusedBlockDiscovery = methodName === "MLS-MPM" ? (st.fusedBlockDiscovery ?? false) : mpmFusedBlockDiscovery;
        controls.setActiveBlocks(nextActiveBlocks);
        controls.setPagedGrid(nextPagedGrid);
        controls.setPagedGridMaxPages(nextPagedGridMaxPages);
        controls.setPagedGridStatus("");
        controls.setFusedBlockDiscovery(nextFusedBlockDiscovery);
        // Foam block (optional). The component sets the enable state + config + UI here;
        // the authoritative push to the active sim happens via applyFoam() inside
        // applyMethod() below. Missing softness/density/subsurface (older presets) keep
        // the current values.
        if (st.foam) {
            const f = st.foam;
            const cur = controls.getValues().foam;
            controls.setFoam({
                enabled: f.enabled,
                activeParticles: f.activeParticles ?? false,
                kTa: f.kTa,
                kWc: f.kWc,
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
                debugTexture: f.debugTexture,
                softness: f.softness ?? cur.softness,
                density: f.density ?? cur.density,
                subsurfaceStrength: f.subsurfaceStrength ?? cur.subsurfaceStrength,
                subsurfaceColor: f.subsurfaceColor ?? cur.subsurfaceColor,
            });
        }
        // Demo extra-control state (box size / paddle) — before applySceneSdf so the
        // restored box bounds are written into the scene SDF.
        if (st.demoState) {
            demo.restoreState?.(st.demoState);
        }
        // Container/nozzle-mesh visibility (applied to the demo by switchPair's post-load
        // setContainerVisible(...) below).
        if (st.showContainer !== undefined) {
            controls.setShowContainer(st.showContainer);
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
            methodName === "MLS-MPM" &&
            (nextActiveBlocks !== mpmActiveBlocks ||
                nextPagedGrid !== mpmPagedGrid ||
                nextPagedGridMaxPages !== mpmPagedGridMaxPages ||
                nextFusedBlockDiscovery !== mpmFusedBlockDiscovery);
        mpmActiveBlocks = nextActiveBlocks;
        mpmPagedGrid = nextPagedGrid;
        mpmPagedGridMaxPages = nextPagedGridMaxPages;
        mpmFusedBlockDiscovery = nextFusedBlockDiscovery;
        const gridChanged = !gridSettingsEqual(nextGridSettings, gridSettings) || !gridSettingsEqual(nextGridSettings, builtGridSettings);
        gridSettings = nextGridSettings;
        if (
            st.count !== particleCount ||
            nextPhysicsScale !== physicsScale ||
            domainScale !== builtDomainScale ||
            gridChanged ||
            (gridSettings !== undefined && builtGridMethod !== methodName) ||
            activeBlocksChanged
        ) {
            rebuildSims(st.count, nextPhysicsScale);
        } else {
            applySceneSdf(); // refresh emitters/spawn for the loaded demo params
            applyMethod(methodName);
            syncGridControls();
        }
        controls.setParticleCount(st.count);
        controls.setPhysScale(nextPhysicsScale);
        refreshDemoParams();
        // Apply the pair's camera framing (preset default on first visit, or the
        // viewpoint captured when this pair was last left). ArcRotate self-clamps.
        if (st.camera) {
            cam.alpha = st.camera.alpha;
            cam.beta = st.camera.beta;
            cam.radius = st.camera.radius;
        }
    }
    // Switch to a (demo, method) pair: snapshot the pair we're leaving, set up the
    // demo visuals if the demo changed, then load the target pair's state.
    function switchPair(nextDemo: FluidDemo, nextMethod: string, nextQuality: Quality = quality, nextMaterial: number = pbmpmMaterial): void {
        // Switching demo or fluid method resumes the sim if it was paused.
        if (paused) {
            paused = false;
            canvas.dataset.paused = "false";
        }
        if (currentPairKey !== null) {
            pairStates.set(currentPairKey, readLivePairState(methodName));
        }
        // Adopt the target method BEFORE anything reads the demo back (the demo-change branch
        // below calls applySceneSdf, which asks the demo for its spawn and emitters). Those can
        // legitimately depend on the method, so they must not be answering for the one we are
        // leaving.
        methodName = nextMethod;
        if (activeDemo !== nextDemo || currentPairKey === null) {
            if (activeDemo) {
                activeDemo.onLeave();
            }
            activeDemo = nextDemo;
            activeDemo.onEnter(); // meshes, camera mode
            applyDemoEnv(activeDemo); // swap skybox background + surface-reflection cube
            applyEnvRotation(activeDemo.envRotationDeg ?? 0); // aim the backdrop the way this demo wants it
            envRotInput.value = String(activeDemo.envRotationDeg ?? 0);
            applySceneSdf(); // scene bounds + emitters + spawn
            refreshDemoParams();
            pendingForce = null;
            activeSim.reset();
            clearSceneHoles();
        }
        quality = nextQuality;
        // PB-MPM keeps a SEPARATE pair (physics/render/colour) per material — but ONLY on material-capable
        // demos (the box). Every other demo is liquid-only, so material is forced to 0 there and the key
        // carries no material axis.
        const withMaterial = nextMethod === "PB-MPM" && MATERIAL_DEMO_KEYS.includes(nextDemo.key);
        if (nextMethod === "PB-MPM") {
            pbmpmMaterial = withMaterial ? nextMaterial : 0;
        }
        const key = withMaterial ? `${nextDemo.key}:${nextMethod}:${nextQuality}:m${pbmpmMaterial}` : `${nextDemo.key}:${nextMethod}:${nextQuality}`;
        const st = pairStates.get(key) ?? presetOrDefault(nextDemo, nextMethod, nextQuality, pbmpmMaterial);
        currentPairKey = key;
        // Propagate the target demo's domain (world) scale BEFORE loading the pair state, so the
        // pair-state rebuild guard fires when it differs from the built scale — switching AWAY from a
        // scaled demo resets it to 1 (base bounds), switching TO one grows the sim domain.
        domainScale = nextDemo.getDomainScale?.() ?? 1;
        loadPairState(st);
        // Re-apply the container-mesh visibility choice (onEnter shows it by default).
        nextDemo.setContainerVisible?.(controls.getValues().showContainer);
    }

    // ── Services handed to each demo ──────────────────────────────────────
    const ctx: FluidCtx = {
        engine,
        scene,
        canvas,
        camera: cam,
        ground,
        sceneSdfBuffer,
        getActiveSim: () => activeSim,
        resetActiveSim: () => activeSim.reset(),
        refreshEmitters: applyEmitters,
        refreshSpawn: applySpawn,
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
        getProfiler: () => (timingEnabled ? profiler : null),
        setDomainScale: (s: number) => {
            if (s === domainScale) {
                return;
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
                    return;
                }
                gridSettings = nextGridSettings;
                domainScale = s;
                rebuildSims(particleCount, nextPhysicsScale);
                return;
            }
            // Gridless presets retain the historical hidden scale on bounds, cell size,
            // particle radius and spawn so existing demos (notably Waterfall) are unchanged.
            domainScale = s;
            rebuildSims(particleCount, physicsScale);
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
    const demos: FluidDemo[] = [createCapsuleDemo(ctx), await createBoxDemo(ctx), createFountainDemo(ctx), createWaterfallDemo(ctx), createMarbleTowerDemo(ctx)];
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
            overlayTask.addMesh(m);
        }
    }

    let paused = false;
    let captureDemoActivated = !captureMode;
    let captureWarmupFrames = 0;
    let captureStarted = false;
    let captureStep = 0;
    let captureReadyPending = false;
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
            canvas.dataset.captureTime = String(captureStep * captureFixedDt);
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
        if (timingEnabled && profiler) {
            profiler.beginFrame();
            profiler.frameStart(engine._currentEncoder);
        }
        // Smoothed FPS: average the frame interval over ~0.5 s, then refresh.
        fpsAccumMs += deltaMs;
        fpsFrames++;
        if (fpsAccumMs >= 500) {
            const gpu = controls.gpu;
            if (gpu) {
                gpu.fpsLabel.textContent = paused ? "paused" : `${Math.round((fpsFrames * 1000) / fpsAccumMs)}`;
            }
            fpsAccumMs = 0;
            fpsFrames = 0;
            // Refresh the timing + memory read-outs on the same ~2 Hz cadence as the FPS counter.
            if (gpu) {
                gpu.refreshTiming(timingEnabled && profiler ? profiler.results() : null);
                gpu.refreshMemory(activeSim.gpuBytes, engine.canvas.width, engine.canvas.height);
            }
        }
        // Clamp dt so a hitch / first frame can't blow the integration up.
        const dt = captureMode ? captureFixedDt : Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
        if (captureMode && !captureStarted) {
            if (activeDemo?.isReady?.() === false) {
                return;
            }
            activeDemo?.update(0);
            activeSim.reset();
            captureStarted = true;
            canvas.dataset.captureStarted = "true";
        }
        // Interactive push force: enabled ONLY while a push is pending, so the
        // dedicated force compute pass is dispatched (and first-compiled) only during
        // an active Shift+RMB drag. Idle frames disable it — nothing force-related runs.
        if (pendingForce) {
            rayForce.setRay(pendingForce.origin, pendingForce.dir, pendingForce.push, pendingForce.radius, pendingForce.accel);
            activeSim.setForceField(rayForce.spec);
            pendingForce = null;
        } else {
            activeSim.setForceField(null);
        }
        // "P" pauses: freeze the obstacles + the solver so the fluid stops advancing.
        // Rendering and the camera keep running, so you can inspect the frozen state;
        // forces / holes resume on unpause.
        if (!paused) {
            activeDemo?.update(dt); // box: spin paddle + write paddle SDF block
            activeSim.step(engine._currentEncoder, dt);
            if (captureMode && ++captureStep >= captureTargetSteps) {
                paused = true;
                canvas.dataset.paused = "true";
                captureReadyPending = true;
            }
        }
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
            canvas.setPointerCapture(e.pointerId);
            forceLastX = e.clientX;
            forceLastY = e.clientY;
            forceLastT = performance.now();
        } else {
            activeDemo?.onPointerDown?.(e);
        }
    });
    canvas.addEventListener("pointermove", (e) => {
        if (forceDragging) {
            // Push the fluid: direction = screen motion mapped into world space via
            // the camera basis; magnitude ∝ mouse speed (px/s); origin/dir = the
            // cursor ray. Applied for one frame by the loop (see pendingForce).
            const dx = e.clientX - forceLastX;
            const dy = e.clientY - forceLastY;
            const now = performance.now();
            const dtMs = Math.max(now - forceLastT, 1);
            forceLastX = e.clientX;
            forceLastY = e.clientY;
            forceLastT = now;
            const speed = (Math.hypot(dx, dy) / dtMs) * 1000;
            if (speed < 1) {
                return;
            }
            const rect = canvas.getBoundingClientRect();
            const ray = screenRay(ctx.viewProjection(), e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height);
            if (!ray) {
                return;
            }
            const wm = cam.worldMatrix;
            // World-space mouse motion: camera right * dx − camera up * dy (screen-y down).
            let px = wm[0]! * dx - wm[4]! * dy;
            let py = wm[1]! * dx - wm[5]! * dy;
            let pz = wm[2]! * dx - wm[6]! * dy;
            const plen = Math.hypot(px, py, pz) || 1;
            px /= plen;
            py /= plen;
            pz /= plen;
            pendingForce = { origin: ray.origin, dir: ray.dir, push: [px, py, pz], radius: FORCE_RADIUS, accel: speed * 0.5 };
        } else {
            activeDemo?.onPointerMove?.(e);
        }
    });
    const onPointerEnd = (e: PointerEvent): void => {
        if (forceDragging) {
            forceDragging = false;
            pendingForce = null;
            canvas.releasePointerCapture(e.pointerId);
        } else {
            activeDemo?.onPointerUp?.(e);
        }
    };
    canvas.addEventListener("pointerup", onPointerEnd);
    canvas.addEventListener("pointercancel", onPointerEnd);

    window.addEventListener("keydown", (e) => {
        if (e.repeat) {
            return;
        }
        activeDemo?.onKey?.(e); // demo-specific (capsule: Space punches a hole)
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
            activeSim.reset();
            clearSceneHoles();
        } else if (e.key === "m" || e.key === "M") {
            const methods = Object.keys(DEFAULT_FLUID_SCHEMAS);
            const nextMethod = methods[(methods.indexOf(methodName) + 1) % methods.length] ?? "PBF";
            switchPair(activeDemo!, nextMethod);
        } else if (e.key === "p" || e.key === "P") {
            paused = !paused;
            canvas.dataset.paused = paused ? "true" : "false";
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
        demos.map((d) => d.key),
        MATERIAL_DEMO_KEYS
    );
    switchPair(boxDemo, "MLS-MPM", quality); // box + MLS-MPM at the default quality
    visitedDemos.add(boxDemo.key); // the start-up demo counts as visited
    containerSel.value = boxDemo.key;
    qualitySel.value = quality;
    controls.setMethod("MLS-MPM");
    applyRenderMode(false); // fluid surface by default
    if (captureMode) {
        setUiHidden(true);
    }

    const hint = document.querySelector(".hint");
    if (hint) {
        hint.textContent =
            "Drag rotate · RMB slide · Shift+RMB push fluid · wheel zoom — Capsule: LMB on tank punches hole · Space random hole · R refill · M switch method · P pause · F8 hide UI";
    }

    // Ensure the environment finished loading (skybox builder registered + specular
    // cube wired into the surface pass + imageProcessing set) before we build the
    // scene, so the HDR skybox renders as the sceneColorRT background from frame 0.
    await envReady;
    applyDemoEnv(activeDemo!); // install the initial (box) skybox + surface env before frame 0
    await registerSceneWithShadowSupport(scene);

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

    await startEngine(engine);
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
