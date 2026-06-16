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
    addToScene,
    attachControl,
    createArcRotateCamera,
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
    startEngine,
} from "babylon-lite";
import type { Mesh } from "babylon-lite";
import { createFluidSim } from "./transmission/fluid/pbf-sim.js";
import type { FluidSim } from "./transmission/fluid/pbf-sim.js";
import { createMlsMpmSim } from "./transmission/fluid/mls-mpm-sim.js";
import { createParticleRenderTask } from "./transmission/fluid/particle-render.js";
import { pickCapsuleHole } from "./transmission/fluid/pick.js";

const PARTICLE_COUNT = 60000;
// MLS-MPM scales far better (no neighbour search). Kept equal to the PBF count
// here so the two methods fill the tank to a comparable level (more particles
// would only pack denser in this fixed-volume capsule, not fill higher).
const MPM_PARTICLE_COUNT = 60000;

async function main(): Promise<void> {
    const __initStart = performance.now();
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;

    // Single-sample so the swapchain RT is the direct render target and the
    // particle task can share one single-sample depth buffer with the scene.
    const engine = await createEngine(canvas, { msaaSamples: 1 });

    // We own the frame graph: scene render task → particle render task, both
    // writing the swapchain colour and sharing a depth buffer we control.
    const scene = createSceneContext(engine, { defaultRenderTask: false });

    const cam = createArcRotateCamera(-Math.PI / 2, 1.1, 30, { x: 0, y: 6, z: 0 });
    cam.nearPlane = 0.1;
    cam.farPlane = 200;
    scene.camera = cam;
    attachControl(cam, canvas, scene);

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

    const sceneTask = createRenderTask(
        { name: "scene", rt: engine.scRT, depth: depthRT, clrColor: { r: 0.05, g: 0.06, b: 0.09, a: 1 } },
        engine,
        scene,
    );
    addTask(scene, sceneTask);

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
    const addShellPart = (mesh: Mesh, x: number, y: number, z: number): void => {
        mesh.material = glass;
        mesh.position.set(x, y, z);
        addToScene(scene, mesh);
    };
    addShellPart(createCylinder(engine, { height: CAP_B[1] - CAP_A[1], diameter: 2 * CAP_R, tessellation: 48 }), 0, (CAP_A[1] + CAP_B[1]) / 2, 0);
    addShellPart(createSphere(engine, { diameter: 2 * CAP_R, segments: 32 }), CAP_A[0], CAP_A[1], CAP_A[2]);
    addShellPart(createSphere(engine, { diameter: 2 * CAP_R, segments: 32 }), CAP_B[0], CAP_B[1], CAP_B[2]);

    // Shared scene geometry for both backends so switching is apples-to-apples.
    const SPAWN_MIN: [number, number, number] = [-2, 4, -2];
    const SPAWN_MAX: [number, number, number] = [2, 12, 2];
    const BOUNDS_MIN: [number, number, number] = [-20, 0, -20];
    const BOUNDS_MAX: [number, number, number] = [20, CAP_B[1] + CAP_R + 0.5, 20];

    // Backend 1 — Position Based Fluids (the original solver).
    const pbfSim = createFluidSim(engine, {
        count: PARTICLE_COUNT,
        particleRadius: 0.09,
        spawnMin: SPAWN_MIN,
        spawnMax: SPAWN_MAX,
        capsuleA: CAP_A,
        capsuleB: CAP_B,
        capsuleRadius: CAP_R,
        groundY: 0,
        // Pin the rest density (independent of count) so the liquid volume — not
        // the packing density — scales with count, keeping grid memory bounded.
        restDensity: 341,
        boundsMin: BOUNDS_MIN,
        boundsMax: BOUNDS_MAX,
        maxPerCell: 48,
    });

    // Backend 2 — MLS-MPM (grid-transfer; scales to far more particles).
    const mpmSim = createMlsMpmSim(engine, {
        count: MPM_PARTICLE_COUNT,
        particleRadius: 0.09,
        spawnMin: SPAWN_MIN,
        spawnMax: SPAWN_MAX,
        capsuleA: CAP_A,
        capsuleB: CAP_B,
        capsuleRadius: CAP_R,
        groundY: 0,
        boundsMin: BOUNDS_MIN,
        boundsMax: BOUNDS_MAX,
        dx: 0.22,
        // Fill control: a LOW rest density (few particles per cell) spaces the
        // particles out. Stiffness + gravity are kept in proportion so the fluid
        // still fills ~half the tank, but both are LOW (vs a stiff 400) so the
        // weakly-compressible EOS doesn't ring — high stiffness makes a springy
        // fluid whose impact/drain energy becomes standing waves that never damp.
        restDensity: 3,
        stiffness: 350,
        gravity: 9.8,
        // Dissipation so it actually comes to rest: a strong APIC→PIC affine
        // damping kills the bulk convection and the divergence (compression)
        // waves, modest viscosity smooths the floor pool, light velocity damping
        // bleeds residual bulk motion. (Bulk flow/draining stays lively because
        // the PIC velocity itself is only lightly damped.)
        viscosity: 0.3,
        substeps: 5,
        subDt: 1 / 300,
        damping: 0.995,
        affineDamping: 0.9,
        // Each new hole sends a stream onto the thin, wide floor pool; without
        // this the impact launches a ripple that travels forever. Damp velocity
        // hard only within 1.5 units of the ground so the pool settles on contact
        // while the capsule fluid and the falling streams stay lively.
        groundDamp: 0.85,
        groundDampHeight: 1.5,
    });

    let activeSim: FluidSim = pbfSim;
    let methodName = "PBF";

    const particleTask = createParticleRenderTask(engine, scene, { colorRT: engine.scRT, depthRT, camera: cam, sim: activeSim });
    addTask(scene, particleTask);

    const hint = document.querySelector(".hint");
    function refreshHud(): void {
        canvas.dataset.method = methodName;
        if (hint) {
            hint.textContent = `Method: ${methodName} (M to switch) · Drag: rotate · Right-click: hole at cursor · Space: random hole · R: refill`;
        }
    }
    refreshHud();

    onBeforeRender(scene, (deltaMs: number) => {
        // Clamp dt so a hitch / first frame can't blow the integration up.
        const dt = Math.min(Math.max(deltaMs, 0) / 1000, 1 / 60);
        activeSim.step(engine._currentEncoder, dt);
    });

    // Controls: drag (LMB) rotates the camera. Space punches a hole at a random
    // spot; RMB punches a hole exactly where the cursor hits the tank. Each press
    // adds another hole. R reseals + refills. M switches simulation backend.
    const HOLE_RADIUS = 0.4;
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 2) {
            return;
        }
        const rect = canvas.getBoundingClientRect();
        const vp = getViewProjectionMatrix(cam, getEffectiveAspectRatio(cam, canvas.width, canvas.height));
        const hit = pickCapsuleHole(vp, e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height, CAP_A, CAP_B, CAP_R);
        if (hit) {
            activeSim.addHole(hit, HOLE_RADIUS);
        }
    });
    window.addEventListener("keydown", (e) => {
        if (e.repeat) {
            return;
        }
        if (e.code === "Space") {
            e.preventDefault();
            const theta = Math.random() * Math.PI * 2;
            const y = CAP_A[1] + Math.random() * 1.5;
            activeSim.addHole([CAP_R * Math.cos(theta), y, CAP_R * Math.sin(theta)], HOLE_RADIUS);
        } else if (e.key === "r" || e.key === "R") {
            activeSim.reset();
        } else if (e.key === "m" || e.key === "M") {
            activeSim = activeSim === pbfSim ? mpmSim : pbfSim;
            methodName = activeSim === pbfSim ? "PBF" : "MLS-MPM";
            activeSim.reset();
            particleTask.setSim(activeSim);
            refreshHud();
        }
    });

    await registerScene(engine, scene);
    await startEngine(engine);
    canvas.dataset.drawCalls = String(engine.drawCallCount);
    canvas.dataset.particleCount = String(PARTICLE_COUNT);
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
