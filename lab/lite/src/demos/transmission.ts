// Transmission demo — GPU fluid simulation (Phase 1).
//
// Phase 1 scope: ~30k GPU particles fall under gravity and pile on a ground
// plane, proving the compute-sim → storage-buffer → instanced-impostor render
// pipeline end-to-end, with correct depth compositing against the ground.
//
// The simulation runs as a WebGPU compute pass encoded into the frame's command
// encoder from onBeforeRender (engine._currentEncoder is valid there); the
// particles render via a custom frame-graph task that shares the scene depth
// buffer. Later phases add the spatial-hash grid, the PBF incompressibility
// solver, the capsule tank + ground SDF boundaries, and the LMB-driven hole.

import {
    addTask,
    addTaskAtStart,
    addToScene,
    attachControl,
    createArcRotateCamera,
    createBox,
    createCylinder,
    createEngine,
    createGround,
    createHemisphericLight,
    createRenderTarget,
    createRenderTask,
    createSceneContext,
    createSphere,
    createStandardMaterial,
    getEffectiveAspectRatio,
    getViewProjectionMatrix,
    onBeforeRender,
    registerScene,
    setMeshVisible,
    startEngine,
} from "babylon-lite";
import type { Mesh } from "babylon-lite";
import { createFluidSim } from "./transmission/fluid/pbf-sim.js";
import type { FluidSim } from "./transmission/fluid/pbf-sim.js";
import { createMlsMpmSim } from "./transmission/fluid/mls-mpm-sim.js";
import { createParticleRenderTask } from "./transmission/fluid/particle-render.js";
import { createFluidSurfaceTask } from "./transmission/fluid/fluid-surface-render.js";
import type { FluidDebug } from "./transmission/fluid/fluid-surface-render.js";
import { createSkyTask, loadEnvCubeFromEnv } from "./transmission/fluid/sky-render.js";import { pickCapsuleHole, screenRay } from "./transmission/fluid/pick.js";

// Particle count is chosen at runtime via the panel dropdown. The PBF rest
// density is pinned (see below) so the count scales the liquid VOLUME, not the
// packing density; MLS-MPM uses the same count so the two methods fill the tank
// comparably. Recreating the sims (createSims) is the only way to resize the
// GPU particle buffers, so the dropdown disposes and rebuilds both backends.
const PARTICLE_COUNTS = [40000, 80000, 120000, 150000, 200000, 300000, 500000];
const DEFAULT_PARTICLE_COUNT = 80000;

// Physics particle-size range (matches the spirit of the visual "Particle size"
// slider). The range is clamped to where the real-time solvers stay visually
// clean: below ~0.8× the fluid is too stiff for a 60 fps timestep (CFL) and
// sprays, and PBF over-compresses the closed box above ~2×. PBF is additionally
// capped at 2× (MLS-MPM handles the larger overfill gracefully up to 3×).
const PHYS_MIN_SCALE = 0.8;
const PHYS_MAX_SCALE = 3;
const PBF_MIN_SCALE = 0.8;
const PBF_MAX_SCALE = 2;
const MPM_MIN_SCALE = 0.8;
const MPM_MAX_SCALE = 3;
const clampScale = (s: number, lo: number, hi: number): number => Math.min(Math.max(s, lo), hi);

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
    let detachCam: (() => void) | null = attachControl(cam, canvas, scene);

    addToScene(scene, createHemisphericLight([0.3, 1, 0.4], 1.1));

    // Ground plane the escaping liquid falls onto.
    const ground = createGround(engine, { width: 40, height: 40, subdivisions: 1 });
    const groundMat = createStandardMaterial();
    groundMat.diffuseColor = [0.42, 0.45, 0.5];
    groundMat.specularColor = [0.05, 0.05, 0.05];
    ground.material = groundMat;
    addToScene(scene, ground);

    // Depth buffer owned by the scene task and re-used (loaded) by the particle
    // task so particles depth-test against the ground.
    const depthRT = createRenderTarget({ lbl: "fluid-depth", dFormat: "depth24plus", samples: 1, size: "canvas" });

    // The scene renders to an offscreen colour target (not directly the
    // swapchain) so the fluid surface pass can SAMPLE it for refraction. The
    // fluid task then presents to the swapchain (blit in sphere mode, or a
    // refraction composite in surface mode).
    const sceneColorRT = createRenderTarget({ lbl: "fluid-scene-color", format: engine.format, samples: 1, size: "canvas" });

    // A procedural sky is drawn into sceneColorRT first (sky task, added at the
    // start below), so the scene task must NOT clear the colour (clr: false) —
    // it loads the sky and renders the geometry on top (depth is still cleared).
    const sceneTask = createRenderTask(
        { name: "scene", rt: sceneColorRT, depth: depthRT, clr: false },
        engine,
        scene,
    );
    addTask(scene, sceneTask);
    const skyTask = createSkyTask(engine, scene, { targetRT: sceneColorRT, camera: cam });
    addTaskAtStart(scene, skyTask);

    // Capsule tank: a vertical pill floating just above the ground (bottom
    // hemisphere centre at y=5, radius 3 → bottom at y=2). The liquid is
    // constrained to its interior; the rounded boundary avoids the flat-wall
    // lattice artefacts of a box.
    const CAP_A: [number, number, number] = [0, 5, 0];
    const CAP_B: [number, number, number] = [0, 11, 0];
    const CAP_R = 3;

    // Transparent glass shell so the tank is visible (cylinder body + two
    // hemispherical caps). Alpha < 1 makes the standard material blend and skip
    // depth writes, so the liquid particles (drawn afterwards) stay visible
    // through it.
    const glass = createStandardMaterial();
    glass.diffuseColor = [0.5, 0.66, 0.82];
    glass.specularColor = [0.8, 0.85, 0.95];
    glass.alpha = 0.15;
    const capsuleMeshes: Mesh[] = [];
    const addShellPart = (mesh: Mesh, x: number, y: number, z: number): void => {
        mesh.material = glass;
        mesh.position.set(x, y, z);
        addToScene(scene, mesh);
        capsuleMeshes.push(mesh);
    };
    addShellPart(createCylinder(engine, { height: CAP_B[1] - CAP_A[1], diameter: 2 * CAP_R, tessellation: 48 }), 0, (CAP_A[1] + CAP_B[1]) / 2, 0);
    addShellPart(createSphere(engine, { diameter: 2 * CAP_R, segments: 32 }), CAP_A[0], CAP_A[1], CAP_A[2]);
    addShellPart(createSphere(engine, { diameter: 2 * CAP_R, segments: 32 }), CAP_B[0], CAP_B[1], CAP_B[2]);

    // Box container: a closed box sitting on the ground (no holes). The X/Z
    // footprint is resizable at runtime via the "Box size" slider (0.5×–3×) by
    // mutating BOX_MIN/BOX_MAX in place; the height is fixed so the box always
    // stays inside the simulation grid bounds (BOUNDS) and needs no realloc. The
    // box floor IS the ground, so the ground plane is hidden in box mode.
    const BOX_HALF_BASE = 4.5; // half-footprint at scale 1
    const BOX_MIN: [number, number, number] = [-BOX_HALF_BASE, 0, -BOX_HALF_BASE];
    const BOX_MAX: [number, number, number] = [BOX_HALF_BASE, 18, BOX_HALF_BASE];
    const boxMesh = createBox(engine, 1);
    boxMesh.material = glass;
    boxMesh.scaling.set(BOX_MAX[0] - BOX_MIN[0], BOX_MAX[1] - BOX_MIN[1], BOX_MAX[2] - BOX_MIN[2]);
    boxMesh.position.set(
        (BOX_MIN[0] + BOX_MAX[0]) / 2,
        (BOX_MIN[1] + BOX_MAX[1]) / 2,
        (BOX_MIN[2] + BOX_MAX[2]) / 2,
    );
    addToScene(scene, boxMesh);

    // Rotating paddle obstacle (box mode only): a thin, tall vertical slab that
    // spins about the box's vertical axis to stir the fluid. `OBS_HALF_WIDTH`
    // leaves a gap to the walls so fluid can flow around the blade ends.
    const OBS_HALF_WIDTH_BASE = 3.0;
    let obsHalfWidth = OBS_HALF_WIDTH_BASE; // scales with the box footprint
    const OBS_HALF_THICK = 0.25;
    const OBS_CENTER: [number, number] = [(BOX_MIN[0] + BOX_MAX[0]) / 2, (BOX_MIN[2] + BOX_MAX[2]) / 2];
    const paddleMat = createStandardMaterial();
    paddleMat.diffuseColor = [0.86, 0.5, 0.2];
    paddleMat.specularColor = [0.35, 0.35, 0.35];
    const paddleMesh = createBox(engine, 1);
    paddleMesh.material = paddleMat;
    paddleMesh.scaling.set(2 * OBS_HALF_THICK, BOX_MAX[1] - BOX_MIN[1], 2 * obsHalfWidth);
    paddleMesh.position.set(OBS_CENTER[0], (BOX_MIN[1] + BOX_MAX[1]) / 2, OBS_CENTER[1]);
    addToScene(scene, paddleMesh);

    // Shared scene geometry for both backends so switching is apples-to-apples.
    // The domain spans both containers (capsule drain spread + the taller box).
    const SPAWN_MIN: [number, number, number] = [-2, 4, -2];
    const SPAWN_MAX: [number, number, number] = [2, 12, 2];
    const BOUNDS_MIN: [number, number, number] = [-20, 0, -20];
    const BOUNDS_MAX: [number, number, number] = [20, 20, 20];

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
    // resize the GPU buffers (the only way to change count is to reallocate).
    function createSims(count: number, scale: number): { pbf: FluidSim; mpm: FluidSim } {
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
        const pbfSpawn = scaledSpawn(pbfScale);
        const mpmSpawn = scaledSpawn(mpmScale);
        // Backend 1 — Position Based Fluids (the original solver).
        const pbf = createFluidSim(engine, {
            count,
            particleRadius: 0.09 * pbfScale,
            smoothingRadius: 0.4 * pbfScale,
            spawnMin: pbfSpawn.min,
            spawnMax: pbfSpawn.max,
            capsuleA: CAP_A,
            capsuleB: CAP_B,
            capsuleRadius: CAP_R,
            groundY: 0,
            restDensity: 341 / (pbfScale * pbfScale * pbfScale),
            boundsMin: BOUNDS_MIN,
            boundsMax: BOUNDS_MAX,
            maxPerCell: 48,
            relaxation: 50 / (pbfScale * pbfScale),
        });

        // Backend 2 — MLS-MPM (grid-transfer; scales to far more particles).
        const mpm = createMlsMpmSim(engine, {
            count,
            particleRadius: 0.09 * mpmScale,
            spawnMin: mpmSpawn.min,
            spawnMax: mpmSpawn.max,
            capsuleA: CAP_A,
            capsuleB: CAP_B,
            capsuleRadius: CAP_R,
            groundY: 0,
            boundsMin: BOUNDS_MIN,
            boundsMax: BOUNDS_MAX,
            dx: 0.22 * mpmScale,
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

        return { pbf, mpm };
    }

    let particleCount = DEFAULT_PARTICLE_COUNT;
    let physicsScale = 1; // physics particle-size multiplier (rebuilds sims)
    let { pbf: pbfSim, mpm: mpmSim } = createSims(particleCount, physicsScale);
    let activeSim: FluidSim = pbfSim;
    let methodName = "PBF";
    let containerMode = 1; // 1 = capsule, 2 = box

    // Rotating-paddle obstacle state (box mode). `obstacleAngle` accumulates from
    // `obstacleSpeed` (rad/s) each frame; the mesh and both sims read it.
    let obstacleOn = false;
    let obstacleSpeed = 1.2;
    let obstacleAngle = 0;
    function updatePaddleVisibility(): void {
        setMeshVisible(paddleMesh, containerMode === 2 && obstacleOn);
    }

    // Mouse-force state (box mode). `forcePending` holds the force computed from
    // the latest pointer move; it's applied for one frame then cleared, so the
    // force only acts while the mouse is actually moving.
    interface PendingForce {
        origin: [number, number, number];
        dir: [number, number, number];
        push: [number, number, number];
        accel: number;
    }
    let forcePending: PendingForce | null = null;
    const FORCE_RADIUS = 3.5;

    const particleTask = createParticleRenderTask(engine, scene, { colorRT: sceneColorRT, depthRT, camera: cam, sim: activeSim });
    addTask(scene, particleTask);
    // Fluid surface renderer + frame presenter: reads the offscreen scene colour
    // and writes the swapchain. In sphere mode it just blits the scene (with the
    // impostors already drawn into it); in surface mode it reconstructs and
    // shades the liquid surface (refraction of the scene + speed foam).
    const surfaceTask = createFluidSurfaceTask(engine, scene, { bgRT: sceneColorRT, outRT: engine.scRT, depthRT, camera: cam, sim: activeSim });
    addTask(scene, surfaceTask);

    // Load the environment cube map and wire it into the sky + the fluid's
    // reflections (a sky-blue placeholder is used until it arrives).
    loadEnvCubeFromEnv(engine, "https://playground.babylonjs.com/textures/environment.env")
        .then((env) => {
            skyTask.setEnvMap(env);
            surfaceTask.setEnvMap(env);
        })
        .catch((err) => console.warn("[transmission] skybox load failed:", err));

    // On-screen FPS read-out (smoothed over ~0.5 s windows). The element lives in
    // the control panel (appended below); it's created here so the render loop can
    // update it without a use-before-define.
    const fpsLabel = document.createElement("div");
    fpsLabel.textContent = "— fps";
    fpsLabel.style.cssText = "font-weight:600;color:#7fd68a;margin-bottom:6px;font-variant-numeric:tabular-nums;";
    let fpsAccumMs = 0;
    let fpsFrames = 0;

    onBeforeRender(scene, (deltaMs: number) => {
        // Smoothed FPS: average the frame interval over ~0.5 s, then refresh.
        fpsAccumMs += deltaMs;
        fpsFrames++;
        if (fpsAccumMs >= 500) {
            fpsLabel.textContent = `${Math.round((fpsFrames * 1000) / fpsAccumMs)} fps`;
            fpsAccumMs = 0;
            fpsFrames = 0;
        }
        // Clamp dt so a hitch / first frame can't blow the integration up.
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
        if (forcePending) {
            activeSim.setForce(forcePending.origin, forcePending.dir, forcePending.push, FORCE_RADIUS, forcePending.accel);
            forcePending = null;
        } else {
            activeSim.setForce([0, 0, 0], [0, 0, 1], [0, 0, 0], FORCE_RADIUS, 0);
        }
        // Rotating paddle (box mode only): advance the angle and feed it to the
        // visible mesh and the active sim's collision.
        if (containerMode === 2 && obstacleOn) {
            obstacleAngle += obstacleSpeed * dt;
            paddleMesh.rotation.y = obstacleAngle;
            activeSim.setObstacle(true, OBS_CENTER, obsHalfWidth, OBS_HALF_THICK, obstacleAngle, obstacleSpeed);
        } else {
            activeSim.setObstacle(false, OBS_CENTER, obsHalfWidth, OBS_HALF_THICK, obstacleAngle, 0);
        }
        activeSim.step(engine._currentEncoder, dt);
    });

    // Apply the chosen container to both sims and swap the visible shell/ground.
    function applyContainer(mode: number): void {
        containerMode = mode;
        const isBox = mode === 2;
        pbfSim.setContainer(mode, BOX_MIN, BOX_MAX);
        mpmSim.setContainer(mode, BOX_MIN, BOX_MAX);
        for (const m of capsuleMeshes) {
            setMeshVisible(m, !isBox);
        }
        setMeshVisible(boxMesh, isBox);
        setMeshVisible(ground, !isBox); // the box floor replaces the ground
        updatePaddleVisibility();
        // In box mode LMB stirs the fluid, so the built-in camera control (which
        // uses LMB to rotate) is detached and replaced by RMB-rotate / wheel-zoom.
        if (isBox && detachCam) {
            detachCam();
            detachCam = null;
        } else if (!isBox && !detachCam) {
            detachCam = attachControl(cam, canvas, scene);
        }
        activeSim.setForce([0, 0, 0], [0, 0, 1], [0, 0, 0], 1, 0);
        activeSim.reset();
    }

    // Resize the box footprint live (X/Z only; height fixed). Mutates BOX_MIN/MAX
    // in place so every later setContainer call (incl. the particle-count rebuild)
    // picks up the current size, rescales the box mesh + paddle, and pushes the new
    // bounds to both sims. No reset: the solver's per-step clamp lets the fluid
    // flow into the new shape, and the grid bounds are unchanged so no realloc.
    function applyBoxScale(scale: number): void {
        const half = BOX_HALF_BASE * scale;
        BOX_MIN[0] = -half;
        BOX_MIN[2] = -half;
        BOX_MAX[0] = half;
        BOX_MAX[2] = half;
        boxMesh.scaling.set(BOX_MAX[0] - BOX_MIN[0], BOX_MAX[1] - BOX_MIN[1], BOX_MAX[2] - BOX_MIN[2]);
        obsHalfWidth = OBS_HALF_WIDTH_BASE * scale;
        paddleMesh.scaling.set(2 * OBS_HALF_THICK, BOX_MAX[1] - BOX_MIN[1], 2 * obsHalfWidth);
        pbfSim.setContainer(containerMode, BOX_MIN, BOX_MAX);
        mpmSim.setContainer(containerMode, BOX_MIN, BOX_MAX);
    }

    // ── Live tuning UI ───────────────────────────────────────────────
    // A combo box to switch method, per-method parameter sliders (applied live),
    // and a reset button. Each schema entry mirrors the sim's current default and
    // remembers the last value the user set.
    interface ParamDef {
        key: string;
        label: string;
        min: number;
        max: number;
        step: number;
        value: number;
    }
    const SCHEMAS: Record<string, ParamDef[]> = {
        PBF: [
            { key: "gravity", label: "Gravity", min: 0, max: 50, step: 0.1, value: 9.8 },
            { key: "viscosity", label: "Viscosity (XSPH)", min: 0, max: 0.3, step: 0.005, value: 0.08 },
            { key: "relaxation", label: "Relaxation ε", min: 1, max: 300, step: 1, value: 50 },
            { key: "scorr", label: "Artificial pressure", min: 0, max: 0.1, step: 0.001, value: 0.02 },
            { key: "iterations", label: "Solver iterations", min: 1, max: 8, step: 1, value: 3 },
            { key: "restDensity", label: "Rest density", min: 100, max: 600, step: 10, value: 341 },
            { key: "boundaryDensity", label: "Boundary density", min: 0, max: 1, step: 0.05, value: 0 },
        ],
        "MLS-MPM": [
            { key: "gravity", label: "Gravity", min: 0, max: 50, step: 0.1, value: 9.8 },
            { key: "stiffness", label: "Stiffness (EOS)", min: 10, max: 5000, step: 10, value: 350 },
            { key: "viscosity", label: "Viscosity", min: 0, max: 1, step: 0.01, value: 0.3 },
            { key: "restDensity", label: "Rest density (/cell)", min: 1, max: 16, step: 0.5, value: 3 },
            { key: "damping", label: "Velocity damping", min: 0.9, max: 1, step: 0.001, value: 0.995 },
            { key: "affineDamping", label: "Affine damping (→PIC)", min: 0.7, max: 1, step: 0.005, value: 0.9 },
            { key: "groundDamp", label: "Ground damping", min: 0.7, max: 1, step: 0.01, value: 0.85 },
            { key: "groundDampHeight", label: "Ground damp height", min: 0, max: 3, step: 0.1, value: 1.5 },
            { key: "restitution", label: "Restitution (bounce)", min: 0, max: 1, step: 0.05, value: 0.3 },
            { key: "substeps", label: "Substeps / frame", min: 1, max: 8, step: 1, value: 3 },
        ],
    };

    const panel = document.createElement("div");
    panel.style.cssText =
        "position:fixed;top:12px;right:12px;z-index:20;width:248px;font:12px system-ui,-apple-system,'Segoe UI',sans-serif;" +
        "color:#dfe6ee;background:rgba(10,14,20,0.85);padding:10px 12px;border-radius:8px;pointer-events:auto;user-select:none;";
    const title = document.createElement("div");
    title.textContent = "Fluid method";
    title.style.cssText = "font-weight:600;margin-bottom:6px;";
    const methodSel = document.createElement("select");
    methodSel.style.cssText = "width:100%;margin-bottom:8px;padding:3px;background:#1a2230;color:#dfe6ee;border:1px solid #33415a;border-radius:4px;";
    for (const name of Object.keys(SCHEMAS)) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name === "PBF" ? "SPH (PBF)" : name;
        methodSel.appendChild(opt);
    }
    const renderTitle = document.createElement("div");
    renderTitle.textContent = "Render";
    renderTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const renderRow = document.createElement("label");
    renderRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:8px;cursor:pointer;";
    const renderChk = document.createElement("input");
    renderChk.type = "checkbox"; // unchecked = fluid surface (default), checked = spheres
    const renderChkText = document.createElement("span");
    renderChkText.textContent = "Render as spheres";
    renderRow.append(renderChk, renderChkText);
    renderChk.onchange = () => applyRenderMode(renderChk.checked);
    const debugTitle = document.createElement("div");
    debugTitle.textContent = "Debug (feature)";
    debugTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const debugSel = document.createElement("select");
    debugSel.style.cssText = "width:100%;margin-bottom:8px;padding:3px;background:#1a2230;color:#dfe6ee;border:1px solid #33415a;border-radius:4px;";
    const debugOpts: { value: string; label: string }[] = [
        { value: "none", label: "None (final render)" },
        { value: "depth", label: "Depth" },
        { value: "depthBlur", label: "Depth (blurred)" },
        { value: "thickness", label: "Thickness" },
        { value: "thicknessBlur", label: "Thickness (blurred)" },
        { value: "normals", label: "Normals" },
    ];
    for (const o of debugOpts) {
        const opt = document.createElement("option");
        opt.value = o.value;
        opt.textContent = o.label;
        debugSel.appendChild(opt);
    }
    debugSel.onchange = () => surfaceTask.setDebug(debugSel.value as FluidDebug);

    // Foam threshold (speed at which the surface goes fully white) + a global
    // half-resolution toggle for the fluid textures (perf).
    const foamRow = document.createElement("div");
    foamRow.style.cssText = "margin:2px 0 6px;";
    const foamHead = document.createElement("div");
    foamHead.style.cssText = "display:flex;justify-content:space-between;";
    const foamLab = document.createElement("span");
    foamLab.textContent = "Foam threshold (speed)";
    const foamVal = document.createElement("span");
    foamVal.style.cssText = "color:#9fb4cc;";
    foamVal.textContent = "6";
    foamHead.append(foamLab, foamVal);
    const foamInput = document.createElement("input");
    foamInput.type = "range";
    foamInput.min = "1";
    foamInput.max = "30";
    foamInput.step = "0.5";
    foamInput.value = "6";
    foamInput.style.cssText = "width:100%;";
    foamInput.oninput = () => {
        foamVal.textContent = foamInput.value;
        surfaceTask.setFoamThreshold(parseFloat(foamInput.value));
    };
    foamRow.append(foamHead, foamInput);
    const halfRow = document.createElement("label");
    halfRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:8px;cursor:pointer;";
    const halfChk = document.createElement("input");
    halfChk.type = "checkbox";
    const halfText = document.createElement("span");
    halfText.textContent = "Half rendering (perf)";
    halfRow.append(halfChk, halfText);
    halfChk.onchange = () => surfaceTask.setHalfRender(halfChk.checked);

    // Particle size: a visual multiplier for both the sphere impostors and the
    // fluid-surface splats (does not change the physics / particle spacing).
    const sizeRow = document.createElement("div");
    sizeRow.style.cssText = "margin:2px 0 8px;";
    const sizeHead = document.createElement("div");
    sizeHead.style.cssText = "display:flex;justify-content:space-between;";
    const sizeLab = document.createElement("span");
    sizeLab.textContent = "Particle size";
    const sizeVal = document.createElement("span");
    sizeVal.style.cssText = "color:#9fb4cc;";
    sizeVal.textContent = "1.0×";
    sizeHead.append(sizeLab, sizeVal);
    const sizeInput = document.createElement("input");
    sizeInput.type = "range";
    sizeInput.min = "0.3";
    sizeInput.max = "3";
    sizeInput.step = "0.1";
    sizeInput.value = "1";
    sizeInput.style.cssText = "width:100%;";
    sizeInput.oninput = () => {
        const s = parseFloat(sizeInput.value);
        sizeVal.textContent = `${s.toFixed(1)}×`;
        surfaceTask.setSizeScale(s);
        particleTask.setSizeScale(s);
    };
    sizeRow.append(sizeHead, sizeInput);

    const containerTitle = document.createElement("div");
    containerTitle.textContent = "Container";
    containerTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const containerSel = document.createElement("select");
    containerSel.style.cssText = "width:100%;margin-bottom:8px;padding:3px;background:#1a2230;color:#dfe6ee;border:1px solid #33415a;border-radius:4px;";
    const containerOpts: { value: string; label: string }[] = [
        { value: "1", label: "Capsule (drainable)" },
        { value: "2", label: "Box (closed)" },
    ];
    for (const o of containerOpts) {
        const opt = document.createElement("option");
        opt.value = o.value;
        opt.textContent = o.label;
        containerSel.appendChild(opt);
    }
    containerSel.onchange = () => applyContainer(parseInt(containerSel.value, 10));

    // Box-size slider (box mode): scales the box X/Z footprint from 0.5× to 3×.
    const boxSizeRow = document.createElement("div");
    boxSizeRow.style.cssText = "margin:2px 0 8px;";
    const boxSizeHead = document.createElement("div");
    boxSizeHead.style.cssText = "display:flex;justify-content:space-between;";
    const boxSizeLab = document.createElement("span");
    boxSizeLab.textContent = "Box size (box only)";
    const boxSizeVal = document.createElement("span");
    boxSizeVal.style.cssText = "color:#9fb4cc;";
    boxSizeVal.textContent = "1.0×";
    boxSizeHead.append(boxSizeLab, boxSizeVal);
    const boxSizeInput = document.createElement("input");
    boxSizeInput.type = "range";
    boxSizeInput.min = "0.5";
    boxSizeInput.max = "3";
    boxSizeInput.step = "0.1";
    boxSizeInput.value = "1";
    boxSizeInput.style.cssText = "width:100%;";
    boxSizeInput.oninput = () => {
        const s = parseFloat(boxSizeInput.value);
        boxSizeVal.textContent = `${s.toFixed(1)}×`;
        applyBoxScale(s);
    };
    boxSizeRow.append(boxSizeHead, boxSizeInput);
    const particlesTitle = document.createElement("div");
    particlesTitle.textContent = "Particles";
    particlesTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const particlesSel = document.createElement("select");
    particlesSel.style.cssText = "width:100%;margin-bottom:8px;padding:3px;background:#1a2230;color:#dfe6ee;border:1px solid #33415a;border-radius:4px;";
    for (const c of PARTICLE_COUNTS) {
        const opt = document.createElement("option");
        opt.value = String(c);
        opt.textContent = `${(c / 1000).toFixed(0)}k`;
        if (c === DEFAULT_PARTICLE_COUNT) opt.selected = true;
        particlesSel.appendChild(opt);
    }
    const sliderHost = document.createElement("div");

    // Physics particle size: rebuilds the sims at a larger/smaller smoothing
    // radius / grid cell so each particle is a bigger or smaller blob of fluid.
    // The count stays the user's choice, so the water volume scales with the size
    // (overfills the tank at large sizes, a small puddle at small sizes). The
    // range is clamped to where the real-time solvers stay visually clean. The
    // rebuild is expensive, so it fires on release (change), not while dragging
    // (input only updates the read-out).
    const physTitle = document.createElement("div");
    physTitle.textContent = "Physics particle size";
    physTitle.style.cssText = "font-weight:600;margin:4px 0 6px;";
    const physRow = document.createElement("div");
    physRow.style.cssText = "margin:2px 0 8px;";
    const physHead = document.createElement("div");
    physHead.style.cssText = "display:flex;justify-content:flex-end;";
    const physVal = document.createElement("span");
    physVal.style.cssText = "color:#9fb4cc;";
    physVal.textContent = "1.0×";
    physHead.append(physVal);
    const physInput = document.createElement("input");
    physInput.type = "range";
    physInput.min = String(PHYS_MIN_SCALE);
    physInput.max = String(PHYS_MAX_SCALE);
    physInput.step = "0.1";
    physInput.value = "1";
    physInput.style.cssText = "width:100%;";
    physInput.oninput = () => {
        physVal.textContent = `${parseFloat(physInput.value).toFixed(1)}×`;
    };
    physRow.append(physHead, physInput);
    // Obstacle (box mode): a checkbox to toggle the rotating paddle and a slider
    // for its angular speed. Inactive in capsule mode (the paddle is box-only).
    const obstacleTitle = document.createElement("div");
    obstacleTitle.textContent = "Obstacle (box only)";
    obstacleTitle.style.cssText = "font-weight:600;margin:8px 0 6px;";
    const obstacleRow = document.createElement("label");
    obstacleRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:6px;cursor:pointer;";
    const obstacleChk = document.createElement("input");
    obstacleChk.type = "checkbox";
    const obstacleChkText = document.createElement("span");
    obstacleChkText.textContent = "Rotating paddle";
    obstacleRow.append(obstacleChk, obstacleChkText);
    const speedRow = document.createElement("div");
    speedRow.style.cssText = "margin:2px 0 4px;";
    const speedHead = document.createElement("div");
    speedHead.style.cssText = "display:flex;justify-content:space-between;";
    const speedLab = document.createElement("span");
    speedLab.textContent = "Paddle speed (rad/s)";
    const speedVal = document.createElement("span");
    speedVal.style.cssText = "color:#9fb4cc;";
    speedVal.textContent = String(obstacleSpeed);
    speedHead.append(speedLab, speedVal);
    const speedInput = document.createElement("input");
    speedInput.type = "range";
    speedInput.min = "0";
    speedInput.max = "4";
    speedInput.step = "0.1";
    speedInput.value = String(obstacleSpeed);
    speedInput.style.cssText = "width:100%;";
    speedInput.oninput = () => {
        obstacleSpeed = parseFloat(speedInput.value);
        speedVal.textContent = speedInput.value;
    };
    obstacleChk.onchange = () => {
        obstacleOn = obstacleChk.checked;
        updatePaddleVisibility();
    };
    speedRow.append(speedHead, speedInput);

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset simulation";
    resetBtn.style.cssText = "width:100%;margin-top:8px;padding:5px;cursor:pointer;background:#26415f;color:#eef3f8;border:1px solid #3a567a;border-radius:4px;";
    resetBtn.onclick = () => activeSim.reset();
    panel.append(title, fpsLabel, methodSel, renderTitle, renderRow, debugTitle, debugSel, foamRow, halfRow, sizeRow, containerTitle, containerSel, boxSizeRow, particlesTitle, particlesSel, physTitle, physRow, sliderHost, obstacleTitle, obstacleRow, speedRow, resetBtn);
    document.body.appendChild(panel);

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

    function buildSliders(name: string): void {
        sliderHost.replaceChildren();
        for (const p of SCHEMAS[name]!) {
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
                p.value = v;
                val.textContent = String(v);
                applyParam(activeSim, p.key, v);
            };
            row.append(head, input);
            sliderHost.appendChild(row);
        }
    }

    function applyMethod(name: string): void {
        activeSim = name === "PBF" ? pbfSim : mpmSim;
        methodName = name;
        for (const p of SCHEMAS[name]!) {
            applyParam(activeSim, p.key, p.value);
        }
        activeSim.reset();
        particleTask.setSim(activeSim);
        surfaceTask.setSim(activeSim);
        methodSel.value = name;
        buildSliders(name);
        canvas.dataset.method = methodName;
    }
    methodSel.onchange = () => applyMethod(methodSel.value);

    // Toggle between the sphere-impostor renderer and the screen-space surface.
    // Default is the fluid surface; the checkbox switches to spheres.
    function applyRenderMode(spheres: boolean): void {
        particleTask.setEnabled(spheres);
        surfaceTask.setMode(spheres ? "blit" : "surface");
        canvas.dataset.render = spheres ? "spheres" : "surface";
    }

    // Resize the particle buffers by disposing and rebuilding both backends at
    // the new count, then re-applying the current container and method (which
    // re-seeds and rebinds the renderer).
    function setParticleCount(n: number): void {
        if (n === particleCount) return;
        particleCount = n;
        pbfSim.dispose();
        mpmSim.dispose();
        ({ pbf: pbfSim, mpm: mpmSim } = createSims(n, physicsScale));
        pbfSim.setContainer(containerMode, BOX_MIN, BOX_MAX);
        mpmSim.setContainer(containerMode, BOX_MIN, BOX_MAX);
        applyMethod(methodName);
        canvas.dataset.particleCount = String(n);
    }
    particlesSel.onchange = () => setParticleCount(parseInt(particlesSel.value, 10));

    // Rebuild both backends at a new physics particle-size scale (couples the
    // smoothing radius / grid cell + rest spacing). Like a count change it
    // reallocates + re-seeds, so it is wired to the slider's release (change).
    function setPhysicsScale(s: number): void {
        if (s === physicsScale) return;
        physicsScale = s;
        pbfSim.dispose();
        mpmSim.dispose();
        ({ pbf: pbfSim, mpm: mpmSim } = createSims(particleCount, physicsScale));
        pbfSim.setContainer(containerMode, BOX_MIN, BOX_MAX);
        mpmSim.setContainer(containerMode, BOX_MIN, BOX_MAX);
        applyMethod(methodName);
    }
    physInput.onchange = () => setPhysicsScale(parseFloat(physInput.value));

    applyMethod("PBF");
    applyContainer(1); // capsule by default (also hides the box mesh)
    applyRenderMode(false); // fluid surface by default

    const hint = document.querySelector(".hint");
    if (hint) {
        hint.textContent = "Capsule: drag rotate · RMB/Space hole · R refill — Box: LMB drag rotate, RMB drag to push fluid, wheel zoom — M: switch method";
    }

    // Input. Capsule mode: the built-in arc camera owns LMB-rotate; RMB / Space
    // punch holes; R refills. Box mode: the camera is detached, LMB drag rotates,
    // RMB drag pushes the fluid (force ∝ mouse speed, along the mouse direction),
    // wheel zooms.
    const HOLE_RADIUS = 0.4;
    let dragBtn = -1;
    let lastX = 0;
    let lastY = 0;
    let lastT = 0;

    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    canvas.addEventListener("pointerdown", (e) => {
        if (containerMode === 1) {
            // Capsule: RMB punches a hole where the cursor hits the tank.
            if (e.button === 2) {
                const rect = canvas.getBoundingClientRect();
                const vp = getViewProjectionMatrix(cam, getEffectiveAspectRatio(cam, canvas.width, canvas.height));
                const hit = pickCapsuleHole(vp, e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height, CAP_A, CAP_B, CAP_R);
                if (hit) {
                    activeSim.addHole(hit, HOLE_RADIUS);
                }
            }
            return;
        }
        // Box mode: start an LMB (rotate) or RMB (force) drag.
        if (e.button === 0 || e.button === 2) {
            dragBtn = e.button;
            lastX = e.clientX;
            lastY = e.clientY;
            lastT = performance.now();
            canvas.setPointerCapture(e.pointerId);
        }
    });

    canvas.addEventListener("pointermove", (e) => {
        if (containerMode !== 2 || dragBtn < 0) {
            return;
        }
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        const now = performance.now();
        const dtMs = Math.max(now - lastT, 1);
        lastX = e.clientX;
        lastY = e.clientY;
        lastT = now;

        if (dragBtn === 0) {
            // LMB: orbit the (detached) arc camera.
            cam.alpha -= dx / 200;
            cam.beta = Math.min(Math.max(cam.beta - dy / 200, 0.05), Math.PI - 0.05);
            return;
        }
        // RMB: push the fluid. Direction = mouse motion mapped into world space
        // via the camera basis; magnitude ∝ mouse speed (px/s).
        const speed = (Math.hypot(dx, dy) / dtMs) * 1000;
        if (speed < 1) {
            return;
        }
        const rect = canvas.getBoundingClientRect();
        const vp = getViewProjectionMatrix(cam, getEffectiveAspectRatio(cam, canvas.width, canvas.height));
        const ray = screenRay(vp, e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height);
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
        forcePending = { origin: ray.origin, dir: ray.dir, push: [px, py, pz], accel: speed * 0.5 };
    });

    const endDrag = (e: PointerEvent): void => {
        if (dragBtn >= 0) {
            dragBtn = -1;
            forcePending = null;
            canvas.releasePointerCapture(e.pointerId);
        }
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    canvas.addEventListener("wheel", (e) => {
        if (containerMode !== 2) {
            return; // capsule mode: the arc camera handles the wheel
        }
        e.preventDefault();
        cam.radius = Math.min(Math.max(cam.radius * (1 + Math.sign(e.deltaY) * 0.1), 6), 120);
    });

    window.addEventListener("keydown", (e) => {
        if (e.repeat) {
            return;
        }
        if (e.code === "Space") {
            e.preventDefault();
            if (containerMode !== 1) {
                return;
            }
            const theta = Math.random() * Math.PI * 2;
            const y = CAP_A[1] + Math.random() * 1.5;
            activeSim.addHole([CAP_R * Math.cos(theta), y, CAP_R * Math.sin(theta)], HOLE_RADIUS);
        } else if (e.key === "r" || e.key === "R") {
            activeSim.reset();
        } else if (e.key === "m" || e.key === "M") {
            applyMethod(activeSim === pbfSim ? "MLS-MPM" : "PBF");
        }
    });

    await registerScene(engine, scene);
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
