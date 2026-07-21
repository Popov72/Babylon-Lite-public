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
    addToScene,
    attachControl,
    createArcRotateCamera,
    createCsmDirectionalShadowGenerator,
    createDirectionalLight,
    createEngine,
    createGround,
    createHemisphericLight,
    createRenderTarget,
    createRenderTask,
    createSceneContext,
    createStandardMaterial,
    getEffectiveAspectRatio,
    getViewProjectionMatrix,
    loadEnvironment,
    onBeforeRender,
    registerSceneWithShadowSupport,
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
import type { Mesh, Task, EnvironmentTextures, Renderable } from "babylon-lite";
import { buildHdrSkyboxRenderable } from "babylon-lite/material/pbr/background-hdr-skybox.js";
import { createFluidProfiler } from "./fluid/gpu-profiler.js";
import type { FluidProfilerImpl } from "./fluid/gpu-profiler.js";
import { createFluidControlsPanel, DEFAULT_FLUID_SCHEMAS } from "babylon-lite/fluid/controls-panel.js";
import { demoAssetUrl } from "./demo-asset-url.js";
import type { DemoParam, FluidCtx, FluidDemo, PairState, PendingForce } from "./fluid/demo.js";
import { exportJsonFromPairState } from "./fluid/preset-io.js";
import { getQualityPreset, QUALITIES, DEFAULT_QUALITY, loadQualityPresets, type Quality } from "./fluid/quality-presets.js";
import { ENV_COUNTRY_URL, ENV_STUDIO_URL } from "./fluid/demo.js";
import { screenRay } from "./fluid/pick.js";
import { CAP_A, CAP_B, CAP_R, createCapsuleDemo } from "./fluid/scenes/capsule.js";
import { createBoxDemo } from "./fluid/scenes/box.js";
import { createFountainDemo } from "./fluid/scenes/fountain.js";
import { createMarbleTowerDemo } from "./fluid/scenes/marbleTower.js";
import { createWaterfallDemo } from "./fluid/scenes/waterfall.js";

// Particle count is chosen at runtime via the panel dropdown. The PBF rest
// density is pinned (see below) so the count scales the liquid VOLUME, not the
// packing density; MLS-MPM uses the same count so the two methods fill the tank
// comparably. Recreating the sims (createSims) is the only way to resize the
// GPU particle buffers, so the dropdown disposes and rebuilds both backends.
const PARTICLE_COUNTS = [40000, 80000, 120000, 150000, 200000, 300000, 500000, 750000, 1000000];
const DEFAULT_PARTICLE_COUNT = 80000;

// Physics particle-size range (matches the spirit of the visual "Particle size"
// slider). Min is 0.5×; going lower makes the fluid stiff for a 60 fps timestep
// (CFL) and spray-prone. PBF over-compresses the closed box above ~2×, so PBF is
// capped at 2× (MLS-MPM handles the larger overfill gracefully up to 3×).
const PHYS_MIN_SCALE = 0.5;
const PHYS_MAX_SCALE = 3;
const PBF_MIN_SCALE = 0.5;
const PBF_MAX_SCALE = 2;
const MPM_MIN_SCALE = 0.5;
const MPM_MAX_SCALE = 3;
const PBMPM_MIN_SCALE = 0.5;
const PBMPM_MAX_SCALE = 3;
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

    addToScene(scene, createHemisphericLight([0.3, 1, 0.4], 0.75));

    // Directional "sun" — casts CSM (cascaded) shadows so the Waterfall demo's
    // terrain + boulders read with real depth. The PBR terrain/boulders + outdoor
    // HDR IBL already carry the lighting, so the sun's intensity is tuned NOT to
    // blow the scene out; its main job is the cascaded shadow map. The sun stays a
    // scene light for every demo, but the shadow GENERATOR is attached only while
    // the Waterfall demo is active (it attaches on enter, detaches on leave) so the
    // box/capsule/fountain demos never render a shadow map — keeping their custom
    // frame graph untouched.
    const sun = createDirectionalLight([-0.5, -0.72, -0.48], 2.4);
    sun.position.set(16, 24, 15);
    addToScene(scene, sun);
    const sunShadow = createCsmDirectionalShadowGenerator(engine, sun, { mapSize: 2048, numCascades: 4, lambda: 0.6, cascadeBlendPercentage: 0.1, bias: 0.00006 });

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

    // Domain (world) scale for the sim bounds. Base 1× keeps the tank at ±20; a demo can grow
    // the whole simulated domain (marble tower "Mesh scale") by scaling the BOUNDS, the grid
    // cell `dx`, the particle/smoothing radius and the spawn box together — so the grid
    // dimensions (bounds/dx) and therefore the GPU memory stay CONSTANT while the domain
    // physically grows. `domainScale` is the desired value (set by switchPair from the active
    // demo's getDomainScale, or by setDomainScale); `builtDomainScale` is what the live sims were
    // last created with (rebuild fires when they differ).
    let domainScale = 1;
    let builtDomainScale = 1;

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
        // Each backend's scale is clamped to its own visually-clean range: below
        // ~0.8× the fluid is too stiff for the real-time timestep and sprays
        // (CFL), and PBF over-compresses the closed box above ~2×.
        const pbfScale = clampScale(scale, PBF_MIN_SCALE, PBF_MAX_SCALE);
        const mpmScale = clampScale(scale, MPM_MIN_SCALE, MPM_MAX_SCALE);
        const pbmpmScale = clampScale(scale, PBMPM_MIN_SCALE, PBMPM_MAX_SCALE);
        // Domain scale grows the WORLD (bounds + dx + particle/smoothing radius + spawn) uniformly,
        // leaving the grid dimensions (bounds/dx) — and hence GPU memory — constant. It is a
        // separate axis from the physics particle-size `scale` above (which changes per-particle
        // density). restDensity/relaxation stay keyed off pbfScale/mpmScale ONLY: they encode the
        // per-particle size ratio, not the world size.
        const ds = domainScale;
        const scaleTriple = (t: [number, number, number]): [number, number, number] => [t[0] * ds, t[1] * ds, t[2] * ds];
        const boundsMin = scaleTriple(BOUNDS_MIN);
        const boundsMax = scaleTriple(BOUNDS_MAX);
        const pbfSpawn = scaledSpawn(pbfScale);
        const mpmSpawn = scaledSpawn(mpmScale);
        const pbmpmSpawn = scaledSpawn(pbmpmScale);
        // Backend 1 — Position Based Fluids (the original solver).
        const pbf = createPbfSim(engine, {
            count,
            particleRadius: 0.09 * pbfScale * ds,
            smoothingRadius: 0.4 * pbfScale * ds,
            spawnMin: scaleTriple(pbfSpawn.min),
            spawnMax: scaleTriple(pbfSpawn.max),
            capsuleA: CAP_A,
            capsuleB: CAP_B,
            capsuleRadius: CAP_R,
            groundY: 0,
            restDensity: 341 / (pbfScale * pbfScale * pbfScale),
            boundsMin,
            boundsMax,
            maxPerCell: 48,
            relaxation: 50 / (pbfScale * pbfScale),
        });

        // Backend 2 — MLS-MPM (grid-transfer; scales to far more particles).
        const mpm = createMlsMpmSim(engine, {
            count,
            particleRadius: 0.09 * mpmScale * ds,
            spawnMin: scaleTriple(mpmSpawn.min),
            spawnMax: scaleTriple(mpmSpawn.max),
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
            boundsMin: [boundsMin[0], -1 * ds, boundsMin[2]],
            boundsMax,
            dx: 0.22 * mpmScale * ds,
            restDensity: 3,
            stiffness: 350,
            gravity: 9.8,
            viscosity: 0.3,
            substeps: 3,
            damping: 0.995,
            affineDamping: 0.9,
            groundDamp: 0.85,
            groundDampHeight: 1.5,
        });

        // Backend 3 — Position-Based MPM (liquid-only PB-MPM phase 1).
        const pbmpm = createPbMpmSim(engine, {
            count,
            particleRadius: 0.09 * pbmpmScale * ds,
            spawnMin: scaleTriple(pbmpmSpawn.min),
            spawnMax: scaleTriple(pbmpmSpawn.max),
            groundY: 0,
            boundsMin: [boundsMin[0], -1 * ds, boundsMin[2]],
            boundsMax,
            dx: 0.22 * pbmpmScale * ds,
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
    let { pbf: pbfSim, mpm: mpmSim, pbmpm: pbmpmSim } = createSims(particleCount, physicsScale);
    let activeSim: FluidSim = pbfSim;
    let methodName = "PBF";
    let quality: Quality = DEFAULT_QUALITY; // low/middle/high preset tier (panel dropdown)
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

    const particleTask = createParticleRenderTask(engine, scene, { colorRT: sceneColorRT, depthRT, camera: cam, sim: activeSim });
    addTask(scene, particleTask);
    // Fluid surface renderer + frame presenter: reads the offscreen scene colour
    // and writes the swapchain. In sphere mode it just blits the scene (with the
    // impostors already drawn into it); in surface mode it reconstructs and
    // shades the liquid surface (refraction of the scene).
    const surfaceTask = createFluidSurfaceTask(engine, scene, { bgRT: sceneColorRT, outRT: engine.scRT, depthRT, camera: cam, sim: activeSim });
    addTask(scene, surfaceTask);

    // Foam (diffuse-particle) renderer — draws the active sim's spray/foam/bubble pool
    // as sprites OVER the composited fluid surface (added after surfaceTask), depth-
    // tested against the shared scene depth so opaque geometry occludes it. Only the
    // PBF backend generates foam for now (setFoam is a no-op / undefined on MLS-MPM).
    const foamTask = createFoamRenderTask(engine, scene, {
        colorRT: engine.scRT,
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
    // pill, box tank glass) AFTER the fluid surface + foam, straight into the swapchain
    // `engine.scRT`, depth-tested (compare, no write) against the shared opaque
    // `depthRT`. Those glass meshes have alpha < 1 and skip depth writes, so if they
    // were drawn (as usual) into the offscreen scene-colour target the fluid surface
    // pass would composite the liquid OVER them. Drawing them last instead means:
    //   • interior fluid (already in scRT) shows THROUGH the translucent glass;
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
    const overlayTask = createRenderTask({ name: "container-overlay", rt: engine.scRT, depth: overlayDepth, clr: false }, engine, scene);
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
        const cfg: FoamConfig = { kTa: f.kTa, kWc: f.kWc, kb: f.kb, kd: f.kd, tMin: f.tMin, tMax: f.tMax, poolScale: f.poolScale };
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
    // Per-demo HDR environments. box/capsule/fountain reflect the neutral studio
    // env (environment.env); the waterfall keeps the green countryside (country.env),
    // which ALSO feeds its PBR terrain/boulder IBL. Both cubes are loaded up front
    // with NO internal skybox (skipSkybox) — the core instead builds one HDR-skybox
    // renderable per cube (identical output to loadEnvironment's own deferred builder:
    // the skybox shader normalizes the cube direction, so size/position/primaryColor
    // are irrelevant) and, on demo switch, swaps which skybox is in scene._renderables
    // (bumping the renderable version so the scene task rebuilds its opaque bundle) and
    // which cube feeds the fluid surface reflections. country.env is left as
    // scene._envTextures so the waterfall terrain IBL is unchanged. The skybox
    // snapshots scene.imageProcessing at build time, so exposure/contrast are set
    // (matching the old baked look) BEFORE the renderables are built. We await
    // `envReady` before registerScene so the initial (box) skybox is in the scene set
    // and rendered from frame 0.
    const brdfUrl = demoAssetUrl("./brdf-lut.png", import.meta.url);
    let studioEnv: EnvironmentTextures | null = null;
    let countryEnv: EnvironmentTextures | null = null;
    let studioSky: Renderable | null = null;
    let countrySky: Renderable | null = null;
    let activeSky: Renderable | null = null;
    const envReady = Promise.all([
        loadEnvironment(scene, ENV_STUDIO_URL, { brdfUrl, skipGround: true, skipSkybox: true }),
        loadEnvironment(scene, ENV_COUNTRY_URL, { brdfUrl, skipGround: true, skipSkybox: true }),
    ])
        .then(([studio, country]) => {
            studioEnv = studio;
            countryEnv = country;
            // country.env drives the waterfall's PBR terrain/boulder IBL — make it the
            // scene env (the last-loaded env otherwise wins non-deterministically).
            scene._envTextures = country;
            scene.imageProcessing.exposure = 1.0;
            scene.imageProcessing.contrast = 1.1;
            studioSky = buildHdrSkyboxRenderable(scene, studio, 10, [0, 0, 0], [0, 0, 0]);
            countrySky = buildHdrSkyboxRenderable(scene, country, 10, [0, 0, 0], [0, 0, 0]);
        })
        .catch((err) => console.warn("[fluid] env load failed", err));

    // Install the active demo's environment: swap the background skybox renderable in
    // the scene render set + point the fluid surface reflections at the matching cube.
    // No-op until the envs finish loading (the first switchPair runs before
    // `await envReady`; the initial install then happens right after it, below).
    function applyDemoEnv(demo: FluidDemo): void {
        if (!studioSky || !countrySky || !studioEnv || !countryEnv) {
            return;
        }
        const useCountry = demo.envUrl === ENV_COUNTRY_URL;
        const nextSky = useCountry ? countrySky : studioSky;
        const nextEnv = useCountry ? countryEnv : studioEnv;
        if (activeSky !== nextSky) {
            if (activeSky) {
                const i = scene._renderables.indexOf(activeSky);
                if (i >= 0) {
                    scene._renderables.splice(i, 1);
                }
            }
            scene._renderables.push(nextSky);
            scene._renderableVersion++;
            activeSky = nextSky;
        }
        surfaceTask.setEnvMap({ view: nextEnv.specularCubeView, sampler: nextEnv.cubeSampler });
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
        const s = demo.spawn();
        pbfSim.setSpawn(s.min, s.max, s.accept);
        mpmSim.setSpawn(s.min, s.max, s.accept);
        pbmpmSim.setSpawn(s.min, s.max, s.accept);
        // Per-demo start-of-sim warm-up (MLS-MPM only; PBF no-ops via ?.). Applied on
        // the next reset()/seed() — switchPair resets the active sim right after this.
        pbfSim.setWarmup?.(s.warmupFrames ?? 0);
        mpmSim.setWarmup?.(s.warmupFrames ?? 0);
        pbmpmSim.setWarmup?.(s.warmupFrames ?? 0);
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
            switchPair(demo, methodName, quality);
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
    // Host for the active demo's live tunables ("Demo parameters") + its demo-specific
    // panel controls.
    const demoParamsHost = document.createElement("div");

    const controls = createFluidControlsPanel({
        schemas: DEFAULT_FLUID_SCHEMAS,
        methods: Object.keys(DEFAULT_FLUID_SCHEMAS),
        particleCounts: PARTICLE_COUNTS,
        physScaleMin: PHYS_MIN_SCALE,
        physScaleMax: PHYS_MAX_SCALE,
        initial: {
            method: methodName,
            count: DEFAULT_PARTICLE_COUNT,
            physScale: physicsScale,
            color: "#16a3c3", // matches the default FLUID_COLOR
            absorption: 1,
            size: 1,
            refraction: 0.1,
            specular: 250,
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
            onReset: () => {
                activeSim.reset();
                clearSceneHoles();
            },
            onFoamEnable: () => pushFoam(),
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
    controls.demoSlot.append(...controls.makeSection("Demo", [demoQualityRow, pbmpmMaterialRow, demoParamsHost, controls.containerToggleRow!]));

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
        if (params.length === 0 && extras.length === 0) {
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
        if (n === particleCount) return;
        particleCount = n;
        pbfSim.dispose();
        mpmSim.dispose();
        pbmpmSim.dispose();
        ({ pbf: pbfSim, mpm: mpmSim, pbmpm: pbmpmSim } = createSims(n, physicsScale));
        applySceneSdf();
        applyMethod(methodName);
        canvas.dataset.particleCount = String(n);
    }

    // Rebuild both backends at a new physics particle-size scale (couples the
    // smoothing radius / grid cell + rest spacing). Like a count change it
    // reallocates + re-seeds, so it is wired to the slider's release (change).
    function setPhysicsScale(s: number): void {
        if (s === physicsScale) return;
        physicsScale = s;
        pbfSim.dispose();
        mpmSim.dispose();
        pbmpmSim.dispose();
        ({ pbf: pbfSim, mpm: mpmSim, pbmpm: pbmpmSim } = createSims(particleCount, physicsScale));
        applySceneSdf();
        applyMethod(methodName);
    }

    // Rebuild both sims at a new particle count + physics scale in one shot.
    // Re-applies demo + method so the new params, emitters and spawn all land
    // before the re-seed.
    function rebuildSims(count: number, scale: number): void {
        particleCount = count;
        physicsScale = scale;
        pbfSim.dispose();
        mpmSim.dispose();
        pbmpmSim.dispose();
        ({ pbf: pbfSim, mpm: mpmSim, pbmpm: pbmpmSim } = createSims(count, scale));
        builtDomainScale = domainScale; // sims are now built at the current domain scale
        applySceneSdf();
        applyMethod(methodName);
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
            depthBlur: RENDER_DEFAULTS.depthBlur,
            depthBlurThreshold: RENDER_DEFAULTS.depthBlurThreshold,
            thicknessBlur: RENDER_DEFAULTS.thicknessBlur,
            surfaceFilter: RENDER_DEFAULTS.surfaceFilter,
            narrowDelta: RENDER_DEFAULTS.narrowDelta,
            narrowMu: RENDER_DEFAULTS.narrowMu,
            anisotropic: RENDER_DEFAULTS.anisotropic,
            anisoSurfScale: RENDER_DEFAULTS.anisoSurfScale,
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
        return {
            schema: { ...base.schema, ...(p.schema ?? {}) },
            demoParams: { ...base.demoParams, ...(p.demoParams ?? {}) },
            color: p.color ?? base.color,
            half: p.half ?? base.half,
            thicknessDownscale: p.thicknessDownscale ?? base.thicknessDownscale,
            absorption: p.absorption ?? base.absorption,
            size: p.size ?? base.size,
            physScale: p.physScale ?? base.physScale,
            count: p.count ?? base.count,
            material: p.material ?? base.material,
            camera: p.camera ?? base.camera,
            renderMode: p.renderMode ?? base.renderMode,
            refraction: p.refraction ?? base.refraction,
            specular: p.specular ?? base.specular,
            depthBlur: p.depthBlur ?? base.depthBlur,
            depthBlurThreshold: p.depthBlurThreshold ?? base.depthBlurThreshold,
            thicknessBlur: p.thicknessBlur ?? base.thicknessBlur,
            surfaceFilter: p.surfaceFilter ?? base.surfaceFilter,
            narrowDelta: p.narrowDelta ?? base.narrowDelta,
            narrowMu: p.narrowMu ?? base.narrowMu,
            anisotropic: p.anisotropic ?? base.anisotropic,
            anisoSurfScale: p.anisoSurfScale ?? base.anisoSurfScale,
            foam: p.foam ? { ...base.foam!, ...p.foam } : base.foam,
            demoState: p.demoState ?? base.demoState,
            showContainer: p.showContainer ?? base.showContainer,
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
            count: particleCount,
            material: method === "PB-MPM" ? pbmpmMaterial : undefined,
            camera: { alpha: cam.alpha, beta: cam.beta, radius: cam.radius },
            renderMode: v.renderMode,
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
            demoState: activeDemo!.snapshotState?.() ?? {},
            showContainer: v.showContainer,
        };
    }
    // Push a PairState into the live params + UI, then apply it (rebuild the sims
    // when the count/scale differ from what's currently built; otherwise re-apply
    // in place). The component's setters re-render the controls to match; refreshDemoParams
    // re-renders the demo section.
    function loadPairState(st: PairState): void {
        const demo = activeDemo!;
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
        // Foam block (optional). The component sets the enable state + config + UI here;
        // the authoritative push to the active sim happens via applyFoam() inside
        // applyMethod() below. Missing softness/density/subsurface (older presets) keep
        // the current values.
        if (st.foam) {
            const f = st.foam;
            const cur = controls.getValues().foam;
            controls.setFoam({
                enabled: f.enabled,
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
        if (st.count !== particleCount || st.physScale !== physicsScale || domainScale !== builtDomainScale) {
            rebuildSims(st.count, st.physScale); // re-does demo + sceneSdf + method (at the current domain scale)
        } else {
            applySceneSdf(); // refresh emitters/spawn for the loaded demo params
            applyMethod(methodName);
        }
        controls.setParticleCount(st.count);
        controls.setPhysScale(st.physScale);
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
        if (activeDemo !== nextDemo || currentPairKey === null) {
            if (activeDemo) {
                activeDemo.onLeave();
            }
            activeDemo = nextDemo;
            activeDemo.onEnter(); // meshes, camera mode
            applyDemoEnv(activeDemo); // swap skybox background + surface-reflection cube
            applySceneSdf(); // scene bounds + emitters + spawn
            refreshDemoParams();
            pendingForce = null;
            activeSim.reset();
            clearSceneHoles();
        }
        methodName = nextMethod;
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
        addSceneHole,
        clearSceneHoles,
        sun,
        sunShadow,
        viewProjection: () => getViewProjectionMatrix(cam, getEffectiveAspectRatio(cam, canvas.width, canvas.height)),
        getProfiler: () => (timingEnabled ? profiler : null),
        setDomainScale: (s: number) => {
            // Rebuild both backends with bounds/dx/radius/spawn scaled by `s` (grid dims — and GPU
            // memory — stay constant) and re-apply the active demo's scene SDF. builtDomainScale is
            // set inside rebuildSims.
            domainScale = s;
            rebuildSims(particleCount, physicsScale);
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

    // Warm up the sun's CSM shadow-caster pipeline for every demo that casts shadows
    // (only the waterfall does). The no-color caster material view is loaded via a
    // dynamic import kicked off by the shadow task's _preload, which registerScene
    // awaits — but ONLY for a light whose generator is attached with casters at that
    // point. So we temporarily attach the generator + register the union of caster
    // meshes here (before registerScene) to force that preload; otherwise the first
    // frame after switching to a shadow demo at runtime would race the import and the
    // shadow task's execute() would throw. Reset back to the default (box) demo's
    // no-shadow state right after registerScene (below); each shadow demo re-attaches
    // the generator + its casters in onEnter.
    const shadowCasters: Mesh[] = [];
    for (const d of demos) {
        for (const m of d.shadowCasters?.() ?? []) {
            shadowCasters.push(m);
        }
    }
    if (shadowCasters.length > 0) {
        sun.shadowGenerator = sunShadow;
        setShadowTaskCasterMeshes(sunShadow, shadowCasters);
    }

    let paused = false;
    onBeforeRender(scene, (deltaMs: number) => {
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
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
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
        // Global shortcuts: R resets (refills), M toggles the backend, P pauses.
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
    await loadQualityPresets(demos.map((d) => d.key), MATERIAL_DEMO_KEYS);
    switchPair(boxDemo, "MLS-MPM", quality); // box + MLS-MPM at the default quality
    containerSel.value = boxDemo.key;
    qualitySel.value = quality;
    controls.setMethod("MLS-MPM");
    applyRenderMode(false); // fluid surface by default

    const hint = document.querySelector(".hint");
    if (hint) {
        hint.textContent =
            "Drag rotate · RMB slide · Shift+RMB push fluid · wheel zoom — Capsule: LMB on tank punches hole · Space random hole · R refill · M switch method · P pause";
    }

    // Ensure the environment finished loading (skybox builder registered + specular
    // cube wired into the surface pass + imageProcessing set) before we build the
    // scene, so the HDR skybox renders as the sceneColorRT background from frame 0.
    await envReady;
    applyDemoEnv(activeDemo!); // install the initial (box) skybox + surface env before frame 0
    await registerSceneWithShadowSupport(scene);

    // Shadow warmup done — the no-color caster module is now loaded (awaited by the
    // registration above). Detach the generator + clear casters so the default (box)
    // demo starts with no shadow map; each shadow-casting demo re-attaches the
    // generator + its own casters in onEnter.
    if (shadowCasters.length > 0) {
        sun.shadowGenerator = undefined;
        setShadowTaskCasterMeshes(sunShadow, []);
    }

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
