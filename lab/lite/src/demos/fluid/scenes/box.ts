// Box demo — a closed glass tank sitting on the ground (its floor replaces the
// ground plane). The X/Z footprint is resizable live; a rotating paddle can stir
// the fluid. Camera + interaction are core-owned: LMB orbits, RMB slides,
// Shift+RMB pushes the fluid, and the wheel zooms.

import HavokPhysics from "@babylonjs/havok";
import {
    addToScene,
    applyPhysicsBodyForce,
    createBox,
    createHavokWorld,
    createPhysicsBody,
    createPhysicsShape,
    createStandardMaterial,
    createTransformNode,
    getPhysicsBodyAngularVelocity,
    getPhysicsBodyLinearVelocity,
    loadGltf,
    onPhysicsAfterStep,
    PhysicsMotionType,
    PhysicsPrestepType,
    PhysicsShapeType,
    releasePhysicsShape,
    removePhysicsBody,
    setMeshVisible,
    setPhysicsBodyAngularVelocity,
    setPhysicsBodyLinearVelocity,
    setPhysicsBodyMassProperties,
    setPhysicsBodyMotionType,
    setPhysicsBodyPrestepType,
    setPhysicsBodyShape,
    setPhysicsBodyTransform,
    setPhysicsShapeMaterial,
    setPhysicsTimestepMs,
    setPhysicsVelocityLimits,
} from "babylon-lite";
import type { Mesh, PhysicsBody, PhysicsShape, SceneNode } from "babylon-lite";
import type { SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import { createFloatingBodySystem } from "babylon-lite/fluid/floating-body.js";
import { generateMeshSdf } from "babylon-lite/fluid/volume-sampling/index.js";
import type { FluidCtx, FluidDemo } from "../demo.js";
import { ENV_STUDIO_URL } from "../demo.js";
import { demoAssetUrl } from "../../demo-asset-url.js";

// glTF Mesh leaves carry CPU geometry copies (positions/indices) used to bake the body SDF.
type CpuMeshNode = SceneNode & { _cpuPositions?: Float32Array; _cpuIndices?: Uint32Array };

// A recognizable glTF to float: the Khronos rubber duck (self-contained GLB, CORS-enabled).
const DUCK_URL = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Duck/glTF-Binary/Duck.glb";

// Box footprint (X/Z resizable via the size slider; height fixed so the box
// stays inside the sim grid bounds and needs no realloc).
const BOX_HALF_BASE = 4.5; // half-footprint at scale 1
// The MLS grid domain is fixed at ±20 (fluid.ts BOUNDS). Keep the box walls a margin
// inside it: past the grid edge, fluid pressed against the SDF wall stacks at the grid's
// 2-cell domain border. The box-size slider max (4.2×) is set so 4.5×max ≈ 18.9 < this.
const BOX_HALF_LIMIT = 19;
const BOX_MIN: [number, number, number] = [-BOX_HALF_BASE, 0, -BOX_HALF_BASE];
const BOX_MAX: [number, number, number] = [BOX_HALF_BASE, 22.5, BOX_HALF_BASE];

// Seed column sized to fall inside the (closed) box height.
const BOX_SPAWN_MIN: [number, number, number] = [-2, 1.5, -2];
const BOX_SPAWN_MAX: [number, number, number] = [2, 7.5, 2];

// Rotating paddle obstacle: a thin, tall vertical slab spinning about the tank's
// vertical axis. OBS_HALF_WIDTH leaves a gap to the walls so fluid flows around
// the blade ends. The tank is centred on the XZ origin.
const OBS_HALF_WIDTH_BASE = 3.0;
const OBS_HALF_THICK = 0.25;
const OBS_CENTER: [number, number] = [(BOX_MIN[0] + BOX_MAX[0]) / 2, (BOX_MIN[2] + BOX_MAX[2]) / 2];

export async function createBoxDemo(ctx: FluidCtx): Promise<FluidDemo> {
    const { engine } = ctx;

    // ── Floating rigid bodies (generic mesh → SDF → two-way fluid coupling) ──────────
    // Delegated to the reusable createFloatingBodySystem (packages/.../fluid/floating-body.ts): ANY mesh
    // baked into a LOCAL-space signed-distance grid (generateMeshSdf) is unioned into the fluid's
    // sceneSdf as a MOVING boundary that pushes the water; here Havok consumes the GPU reduction as
    // buoyancy/drag forces for a rubber DUCK loaded async from glTF (registered once it loads).
    const BODY_HALF = 1.3; // cube half-extent of the bake grid (must contain the scaled duck)
    const BODY_CELL = 0.1; // SDF bake resolution
    const DUCK_TARGET = 2.2; // the duck is uniformly scaled so its LARGEST dimension = this (fits the cube)
    // The glTF loader mirrors geometry to convert RH→LH (__root__ scale (-1,1,1)); the duck display node
    // must keep that mirror or it renders inside-out. Bake + registration both use this frame.
    const DUCK_MIRROR: [number, number, number] = [-1, 1, 1];
    const WATER_DENSITY = 1; // relative; buoyancy uses this × displaced volume
    const BODY_REL_DENSITY = 0.45; // < 1 → floats, settling partially submerged
    // Fixed bake bounds (a cube): both the placeholder and the duck bake with these, so the grid dims —
    // and hence the GPU buffer — are identical and the duck can be written into the pre-allocated buffer.
    const BAKE = { min: [-BODY_HALF, -BODY_HALF, -BODY_HALF] as [number, number, number], max: [BODY_HALF, BODY_HALF, BODY_HALF] as [number, number, number], cellSize: BODY_CELL, padding: 3, sweepPasses: 1 };
    const buildBoxMesh = (h: [number, number, number]): { pos: Float32Array; idx: Uint32Array } => {
        const [hx, hy, hz] = h;
        const pos = new Float32Array([-hx, -hy, -hz, hx, -hy, -hz, hx, hy, -hz, -hx, hy, -hz, -hx, -hy, hz, hx, -hy, hz, hx, hy, hz, -hx, hy, hz]);
        const idx = new Uint32Array([0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7]);
        return { pos, idx };
    };
    // Volume from the grid interior (voxels < 0) + a box-diagonal inertia — GENERIC for any mesh.
    const volFromGrid = (g: { data: Float32Array; cellSize: number }): number => {
        let v = 0;
        const c3 = g.cellSize ** 3;
        for (let i = 0; i < g.data.length; i++) if (g.data[i]! < 0) v += c3;
        return v;
    };
    const boxInertia = (m: number, h: [number, number, number]): [number, number, number] => {
        const ex = (2 * h[0]) ** 2, ey = (2 * h[1]) ** 2, ez = (2 * h[2]) ** 2;
        return [(m / 12) * (ey + ez), (m / 12) * (ex + ez), (m / 12) * (ex + ey)];
    };
    const collectMeshes = (node: SceneNode, out: Mesh[]): void => {
        if ("_gpu" in node) out.push(node as unknown as Mesh);
        for (const c of node.children) collectMeshes(c, out);
    };

    // Havok owns rigid-body motion/collisions for the ducks. The floating-body system below remains
    // physics-agnostic: it only measures submersion on the GPU and receives Havok poses via setBodyPose.
    const hknp = await HavokPhysics({ locateFile: () => demoAssetUrl("./HavokPhysics.wasm", import.meta.url) });
    const physicsWorld = createHavokWorld(ctx.scene, hknp, { x: 0, y: -9.8, z: 0 });
    setPhysicsTimestepMs(physicsWorld, 1000 / 60);
    setPhysicsVelocityLimits(physicsWorld, 18, 10);

    // Placeholder bake — ONLY to fix the grid footprint (dims → storage-buffer size) so the system's
    // sdfBuffer exists up-front for the scene SDF. It is never floated; the duck (baked with the same
    // BAKE opts, so identical dims) is registered once it loads.
    const ph = buildBoxMesh([0.9, 0.7, 0.9]);
    const phGrid = generateMeshSdf(ph.pos, ph.idx, BAKE);

    // Reusable floating-body system (up to a handful of bodies). The box walls/floor clamp is fed live
    // via getBounds so it tracks the resizable footprint.
    const bodySystem = createFloatingBodySystem(engine._device, {
        maxBodies: 6,
        gridFloats: phGrid.data.length,
        band: 0.3,
        buoyK: 0.02,
        linDrag: 6,
        angDrag: 3.5,
        buoyEma: 8,
        dragCarry: 0.004,
        getBounds: () => ({ min: BOX_MIN, max: BOX_MAX }),
        externalPose: true,
    });

    interface PhysicsCollider {
        body: PhysicsBody;
        shape: PhysicsShape;
    }

    const WALL_THICK = 0.6;
    let wallColliders: PhysicsCollider[] = [];
    const createStaticBoxCollider = (name: string, center: [number, number, number], extents: [number, number, number], friction = 0.65, restitution = 0.15): PhysicsCollider => {
        const node = createTransformNode(name, center[0], center[1], center[2]);
        const shape = createPhysicsShape(physicsWorld, { type: PhysicsShapeType.BOX, parameters: { extents: { x: extents[0], y: extents[1], z: extents[2] } } });
        const body = createPhysicsBody(physicsWorld, node, PhysicsMotionType.STATIC);
        setPhysicsBodyShape(physicsWorld, body, shape);
        setPhysicsShapeMaterial(physicsWorld, shape, friction, restitution);
        return { body, shape };
    };
    const rebuildBoxColliders = (): void => {
        for (const c of wallColliders) {
            removePhysicsBody(physicsWorld, c.body);
            releasePhysicsShape(physicsWorld, c.shape);
        }
        const width = BOX_MAX[0] - BOX_MIN[0];
        const depth = BOX_MAX[2] - BOX_MIN[2];
        const height = BOX_MAX[1] - BOX_MIN[1];
        const cx = (BOX_MIN[0] + BOX_MAX[0]) * 0.5;
        const cy = (BOX_MIN[1] + BOX_MAX[1]) * 0.5;
        const cz = (BOX_MIN[2] + BOX_MAX[2]) * 0.5;
        wallColliders = [
            createStaticBoxCollider("box-floor-physics", [cx, BOX_MIN[1] - WALL_THICK * 0.5, cz], [width + 2 * WALL_THICK, WALL_THICK, depth + 2 * WALL_THICK], 0.8, 0.05),
            createStaticBoxCollider("box-wall-x-min-physics", [BOX_MIN[0] - WALL_THICK * 0.5, cy, cz], [WALL_THICK, height, depth + 2 * WALL_THICK]),
            createStaticBoxCollider("box-wall-x-max-physics", [BOX_MAX[0] + WALL_THICK * 0.5, cy, cz], [WALL_THICK, height, depth + 2 * WALL_THICK]),
            createStaticBoxCollider("box-wall-z-min-physics", [cx, cy, BOX_MIN[2] - WALL_THICK * 0.5], [width + 2 * WALL_THICK, height, WALL_THICK]),
            createStaticBoxCollider("box-wall-z-max-physics", [cx, cy, BOX_MAX[2] + WALL_THICK * 0.5], [width + 2 * WALL_THICK, height, WALL_THICK]),
        ];
    };
    rebuildBoxColliders();

    // Duck display state (filled by the async load). The system poses each duck root; the demo only
    // toggles the meshes' visibility.
    let bodyOn = false;
    let demoActive = false; // true only while the Box demo is the active demo (gates the kinematic paddle)
    let duckReady = false;
    let duckMeshes: Mesh[] = [];
    // Tuning read-out (scraped by QA + handy for the user).
    const raftReadout = document.createElement("div");
    raftReadout.style.cssText = "color:#9fb4cc;font-size:11px;margin:0 0 6px;";
    raftReadout.textContent = "ducks: off";
    let raftReadoutFrames = 0;

    interface DuckPhysics {
        index: number;
        body: PhysicsBody;
        shape: PhysicsShape;
        proxy: SceneNode;
        start: [number, number, number];
        smCount: number;
    }

    const duckBodies: DuckPhysics[] = [];
    const BUOY_FORCE_K = 0.022;
    const BUOY_EMA = 8;
    const DRAG_CARRY_FORCE = 0.004;
    const LINEAR_DAMP = 0.55;
    const VERTICAL_DAMP = 5.5;
    const ANGULAR_DAMP = 3.0;
    const RIGHTING_K = 9.0; // upright self-righting: mimics a low centre of mass so ducks settle upright
    const MAX_BUOY_ACCEL = 45;
    const MAX_SPEED = 14;

    const setDuckBodyEnabled = (duck: DuckPhysics, on: boolean): void => {
        setPhysicsBodyMotionType(physicsWorld, duck.body, on ? PhysicsMotionType.DYNAMIC : PhysicsMotionType.STATIC);
        if (!on) {
            setPhysicsBodyLinearVelocity(physicsWorld, duck.body, { x: 0, y: 0, z: 0 });
            setPhysicsBodyAngularVelocity(physicsWorld, duck.body, { x: 0, y: 0, z: 0 });
        }
    };
    const pushDuckPoseToSdf = (duck: DuckPhysics): void => {
        const p = duck.proxy.position;
        const q = duck.proxy.rotationQuaternion;
        const v = getPhysicsBodyLinearVelocity(physicsWorld, duck.body);
        const w = getPhysicsBodyAngularVelocity(physicsWorld, duck.body);
        bodySystem.setBodyPose(duck.index, [p.x, p.y, p.z], [q.x, q.y, q.z, q.w], [v.x, v.y, v.z], [w.x, w.y, w.z]);
    };
    const resetDuckBody = (duck: DuckPhysics): void => {
        duck.smCount = 0;
        setPhysicsBodyTransform(physicsWorld, duck.body, { x: duck.start[0], y: duck.start[1], z: duck.start[2] }, { x: 0, y: 0, z: 0, w: 1 });
        setPhysicsBodyLinearVelocity(physicsWorld, duck.body, { x: 0, y: 0, z: 0 });
        setPhysicsBodyAngularVelocity(physicsWorld, duck.body, { x: 0, y: 0, z: 0 });
        bodySystem.setBodyPose(duck.index, duck.start, [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]);
    };
    const setDucksEnabled = (on: boolean, reset = false): void => {
        bodySystem.setEnabled(on);
        for (const duck of duckBodies) {
            if (reset) {
                resetDuckBody(duck);
            }
            setDuckBodyEnabled(duck, on);
        }
    };
    const createDuckPhysics = (index: number, start: [number, number, number], mass: number, inertia: [number, number, number], half: [number, number, number]): DuckPhysics => {
        const proxy = createTransformNode(`duck-physics-${index}`, start[0], start[1], start[2]);
        const shape = createPhysicsShape(physicsWorld, {
            type: PhysicsShapeType.BOX,
            parameters: { extents: { x: 2 * half[0], y: 2 * half[1], z: 2 * half[2] } },
        });
        const body = createPhysicsBody(physicsWorld, proxy, PhysicsMotionType.DYNAMIC, true);
        setPhysicsBodyShape(physicsWorld, body, shape);
        setPhysicsShapeMaterial(physicsWorld, shape, 0.55, 0.18);
        setPhysicsBodyMassProperties(physicsWorld, body, { mass, inertia: { x: inertia[0], y: inertia[1], z: inertia[2] } });
        const duck: DuckPhysics = { index, body, shape, proxy, start: [...start], smCount: 0 };
        resetDuckBody(duck);
        setDuckBodyEnabled(duck, bodyOn);
        return duck;
    };
    const applyDuckForces = (dt: number): void => {
        if (!bodyOn) {
            return;
        }
        for (const duck of duckBodies) {
            const m = bodySystem.bodyMeasure(duck.index);
            if (!m) {
                continue;
            }
            duck.smCount += (m.submergedCount - duck.smCount) * Math.min(BUOY_EMA * dt, 1);
            const bodyPos = duck.proxy.position;
            const v = getPhysicsBodyLinearVelocity(physicsWorld, duck.body);
            if (duck.smCount > 0.1) {
                const buoy = Math.min(BUOY_FORCE_K * duck.smCount, m.mass * MAX_BUOY_ACCEL);
                applyPhysicsBodyForce(physicsWorld, duck.body, { x: 0, y: buoy, z: 0 }, { x: m.submergedCentroid[0], y: m.submergedCentroid[1], z: m.submergedCentroid[2] });
            }
            const dampY = m.submergedCount > 0 ? -v.y * m.mass * VERTICAL_DAMP : 0;
            applyPhysicsBodyForce(physicsWorld, duck.body, { x: -v.x * m.mass * LINEAR_DAMP, y: dampY, z: -v.z * m.mass * LINEAR_DAMP }, { x: bodyPos.x, y: bodyPos.y, z: bodyPos.z });
            if (m.submergedCount > 0) {
                const carry = Math.min(DRAG_CARRY_FORCE * m.submergedCount, 4);
                applyPhysicsBodyForce(
                    physicsWorld,
                    duck.body,
                    { x: (m.fluidVelocity[0] - v.x) * m.mass * carry, y: 0, z: (m.fluidVelocity[2] - v.z) * m.mass * carry },
                    { x: bodyPos.x, y: bodyPos.y, z: bodyPos.z }
                );
            }
            const w = getPhysicsBodyAngularVelocity(physicsWorld, duck.body);
            let wx = w.x,
                wy = w.y,
                wz = w.z;
            // Upright self-righting while in the water: rotate the duck's local up-axis back toward
            // world-up. cross(localUp, worldUp) is the restoring axis (∝ sin(tilt)); the angular damping
            // below settles the resulting oscillation. Yaw (wy) is left free.
            if (m.submergedCount > 0) {
                const q = duck.proxy.rotationQuaternion;
                const ux = 2 * (q.x * q.y - q.w * q.z); // localUp.x in world (2nd column of R(q))
                const uz = 2 * (q.y * q.z + q.w * q.x); // localUp.z in world
                wx += RIGHTING_K * -uz * dt;
                wz += RIGHTING_K * ux * dt;
            }
            const ad = Math.max(0, 1 - ANGULAR_DAMP * dt);
            setPhysicsBodyAngularVelocity(physicsWorld, duck.body, { x: wx * ad, y: wy * ad, z: wz * ad });
            const speed = Math.hypot(v.x, v.y, v.z);
            if (speed > MAX_SPEED) {
                const s = MAX_SPEED / speed;
                setPhysicsBodyLinearVelocity(physicsWorld, duck.body, { x: v.x * s, y: v.y * s, z: v.z * s });
            }
        }
    };
    onPhysicsAfterStep(physicsWorld, () => {
        for (const duck of duckBodies) {
            pushDuckPoseToSdf(duck);
        }
    });

    // Box + rotating paddle. The paddle is a subtracted vertical slab whose angle
    // advances by omega·dt inside the SDF, so the moving-boundary velocity comes
    // from -∂sceneSdf/∂t (the sim's resolve handles it). obs0 = (pivotX, pivotZ,
    // halfWidth, halfThickness); obs1 = (cosA, sinA, omega, active).
    const sdf: SceneSdfSpec = {
        struct: `struct SceneSdfParams { lo: vec4<f32>, hi: vec4<f32>, obs0: vec4<f32>, obs1: vec4<f32> };`,
        sdf: `${bodySystem.sdfWgsl}
fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    let d1 = pt - sceneSdfParams.lo.xyz;
    let d2 = sceneSdfParams.hi.xyz - pt;
    var d = min(min(min(d1.x, d1.y), d1.z), min(min(d2.x, d2.y), d2.z));
    if (sceneSdfParams.obs1.w > 0.5) {
        let cosA = sceneSdfParams.obs1.x;
        let sinA = sceneSdfParams.obs1.y;
        let cd = cos(sceneSdfParams.obs1.z * dt);
        let sd = sin(sceneSdfParams.obs1.z * dt);
        let c = cosA * cd + sinA * sd;
        let s = sinA * cd - cosA * sd;
        let rx = pt.x - sceneSdfParams.obs0.x;
        let rz = pt.z - sceneSdfParams.obs0.y;
        let lx = c * rx + s * rz;
        let lz = -s * rx + c * rz;
        // Paddle collision spans floor..mid-height, matching the half-height visible mesh
        // (a finite 3D box, not an infinite vertical slab). Bounds derive from the box
        // lo.y / hi.y already in the params, so no extra obstacle fields are needed.
        let pTop = (sceneSdfParams.lo.y + sceneSdfParams.hi.y) * 0.5;
        let pYc = (sceneSdfParams.lo.y + pTop) * 0.5;
        let pYh = (pTop - sceneSdfParams.lo.y) * 0.5;
        let q = vec3<f32>(abs(lx) - sceneSdfParams.obs0.w, abs(pt.y - pYc) - pYh, abs(lz) - sceneSdfParams.obs0.z);
        let outside = length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        d = min(d, outside);
    }
    // Floating rigid bodies (moving boundaries) — supplied by the reusable system's bodiesSdf, which
    // reads the sceneSdfGrid storage buffer (pose header + baked grids) and unions every active body.
    d = min(d, bodiesSdf(pt, dt));
    return d;
}`,
        buffer: ctx.sceneSdfBuffer,
        sdfGrid: bodySystem.sdfBuffer,
    };

    // Glass tank (transparent, blends, skips depth writes so the liquid stays
    // visible through it).
    const glass = createStandardMaterial();
    glass.diffuseColor = [0.5, 0.66, 0.82];
    glass.specularColor = [0.8, 0.85, 0.95];
    glass.alpha = 0.15;
    const boxMesh = createBox(engine, 1);
    boxMesh.material = glass;
    boxMesh.scaling.set(BOX_MAX[0] - BOX_MIN[0], BOX_MAX[1] - BOX_MIN[1], BOX_MAX[2] - BOX_MIN[2]);
    boxMesh.position.set((BOX_MIN[0] + BOX_MAX[0]) / 2, (BOX_MIN[1] + BOX_MAX[1]) / 2, (BOX_MIN[2] + BOX_MAX[2]) / 2);
    addToScene(ctx.scene, boxMesh);
    setMeshVisible(boxMesh, false);
    let containerVisible = true; // toggled by the "Show container mesh" UI checkbox

    const paddleMat = createStandardMaterial();
    paddleMat.diffuseColor = [0.86, 0.5, 0.2];
    paddleMat.specularColor = [0.35, 0.35, 0.35];
    const paddleMesh = createBox(engine, 1);
    paddleMesh.material = paddleMat;
    let obsHalfWidth = OBS_HALF_WIDTH_BASE; // box footprint half-reach (scales with the box)
    // The paddle MESH is a half-height blade anchored on the floor (spans BOX_MIN[1]
    // up to the box mid-height). The collision SDF is an infinite-height vertical
    // slab (no Y extent — see writeScenePaddle/sceneSdf), so shortening the mesh
    // leaves the fluid physics unchanged (the liquid pools at the bottom anyway).
    paddleMesh.scaling.set(2 * OBS_HALF_THICK, 0.5 * (BOX_MAX[1] - BOX_MIN[1]), 2 * obsHalfWidth);
    paddleMesh.position.set(OBS_CENTER[0], BOX_MIN[1] + 0.25 * (BOX_MAX[1] - BOX_MIN[1]), OBS_CENTER[1]);
    addToScene(ctx.scene, paddleMesh);
    setMeshVisible(paddleMesh, false);

    // Kinematic (ANIMATED) Havok body matching the rotating paddle blade, so floating ducks physically
    // bounce off it (the fluid already feels the paddle via the moving-boundary SDF). It is driven in
    // ACTION prestep mode: we move its NODE each frame and the world's pre-step turns the delta into a
    // surface VELOCITY (HP_Body_SetTargetQTransform) — that velocity is what actually shoves the ducks.
    // A plain teleport (setPhysicsBodyTransform) snaps with zero velocity and does NOT push them.
    // Parked far below (via a zero-velocity TELEPORT) when the paddle is off or the demo is inactive.
    const PADDLE_PARK_Y = -1000;
    let paddleCollider: PhysicsCollider | null = null;
    let paddlePhysActive = false; // true while the blade is velocity-driven (ACTION prestep) and colliding
    const paddleBodyY = (): number => BOX_MIN[1] + 0.25 * (BOX_MAX[1] - BOX_MIN[1]);
    const rebuildPaddleCollider = (): void => {
        if (paddleCollider) {
            removePhysicsBody(physicsWorld, paddleCollider.body);
            releasePhysicsShape(physicsWorld, paddleCollider.shape);
            paddleCollider = null;
        }
        const node = createTransformNode("box-paddle-physics", 0, PADDLE_PARK_Y, 0);
        const shape = createPhysicsShape(physicsWorld, {
            type: PhysicsShapeType.BOX,
            parameters: { extents: { x: 2 * OBS_HALF_THICK, y: 0.5 * (BOX_MAX[1] - BOX_MIN[1]), z: 2 * obsHalfWidth } },
        });
        const body = createPhysicsBody(physicsWorld, node, PhysicsMotionType.ANIMATED);
        setPhysicsBodyShape(physicsWorld, body, shape);
        setPhysicsShapeMaterial(physicsWorld, shape, 0.4, 0.2);
        paddleCollider = { body, shape };
        paddlePhysActive = false; // fresh body is parked with the default TELEPORT prestep
    };
    const updatePaddleBody = (): void => {
        if (!paddleCollider) {
            return;
        }
        const body = paddleCollider.body;
        const wantActive = obstacleOn && bodyOn && demoActive;
        if (wantActive) {
            const py = paddleBodyY();
            const h = obstacleAngle * 0.5; // paddleMesh.rotation.y = obstacleAngle → yaw quaternion (0,sin,0,cos)
            if (!paddlePhysActive) {
                // Snap to the current pose with ZERO velocity first, then switch to velocity-driven
                // keyframing so re-activation doesn't fling the ducks with a huge catch-up velocity.
                setPhysicsBodyTransform(physicsWorld, body, { x: OBS_CENTER[0], y: py, z: OBS_CENTER[1] }, { x: 0, y: Math.sin(h), z: 0, w: Math.cos(h) });
                setPhysicsBodyPrestepType(body, PhysicsPrestepType.ACTION);
                paddlePhysActive = true;
            }
            // Drive the node target each frame; the pre-step derives ω from the per-frame angle delta.
            body.node.position.set(OBS_CENTER[0], py, OBS_CENTER[1]);
            body.node.rotationQuaternion.set(0, Math.sin(h), 0, Math.cos(h));
        } else if (paddlePhysActive) {
            // Deactivate: stop keyframing and park below with a zero-velocity teleport (no fling).
            setPhysicsBodyPrestepType(body, PhysicsPrestepType.TELEPORT);
            setPhysicsBodyTransform(physicsWorld, body, { x: 0, y: PADDLE_PARK_Y, z: 0 }, { x: 0, y: 0, z: 0, w: 1 });
            paddlePhysActive = false;
        }
    };
    rebuildPaddleCollider();

    // The floating body's display IS the loaded duck root — the system poses it (TRS) each frame; the
    // demo only toggles the duck meshes' visibility. No fallback mesh: the duck registers once loaded.
    const setBodyVisible = (v: boolean): void => {
        for (const m of duckMeshes) setMeshVisible(m, v && duckReady);
    };

    // ── Load the duck GLB, bake its SDF once, register N instances at staggered start positions ──
    // The bake grid is shared across every instance (deduped by object identity → one GPU upload); each
    // duck gets its OWN display root (the GLB is re-loaded per instance, served from the browser cache).
    const DUCK_SPAWNS: [number, number, number][] = [
        [0, 6.0, 0],
        [-2.6, 6.6, -2.4],
        [2.6, 7.2, 2.4],
        [2.6, 7.8, -2.6],
    ];
    const loadDuckRoot = async (): Promise<{ root: SceneNode; meshes: Mesh[] } | null> => {
        const asset = await loadGltf(engine, DUCK_URL);
        asset.animationGroups = undefined;
        addToScene(ctx.scene, asset);
        const root = asset.entities[0] as SceneNode;
        root.position.set(0, 0, 0);
        root.rotationQuaternion.set(0, 0, 0, 1);
        // KEEP the glTF loader's RH→LH mirror scale (__root__ is created with scale (-1,1,1); see
        // loader-gltf/load-gltf.ts). Resetting it to (1,1,1) renders the duck INSIDE-OUT. The bake below
        // reads the leaf worldMatrices in THIS mirrored frame, and the body is registered with a matching
        // displayScale, so the SDF and the visible mesh stay consistent.
        root.scaling.set(DUCK_MIRROR[0], DUCK_MIRROR[1], DUCK_MIRROR[2]);
        const meshes: Mesh[] = [];
        collectMeshes(root, meshes);
        return meshes.length ? { root, meshes } : null;
    };
    void (async (): Promise<void> => {
        try {
            const first = await loadDuckRoot();
            if (!first) return;
            const { root, meshes } = first;
            let totalV = 0, totalI = 0;
            for (const m of meshes) {
                const cm = m as CpuMeshNode;
                if (cm._cpuPositions && cm._cpuIndices) { totalV += cm._cpuPositions.length; totalI += cm._cpuIndices.length; }
            }
            if (totalV === 0 || totalI === 0) return;
            // Merge asset-frame WORLD positions (local × worldMatrix; root now identity) + vertex-offset
            // indices, tracking the AABB → centre + scale so the duck fits the bake cube.
            const q = new Float32Array(totalV);
            const idx = new Uint32Array(totalI);
            let pOff = 0, iOff = 0;
            let minx = Infinity, miny = Infinity, minz = Infinity, maxx = -Infinity, maxy = -Infinity, maxz = -Infinity;
            for (const m of meshes) {
                const cm = m as CpuMeshNode;
                const src = cm._cpuPositions, ix = cm._cpuIndices;
                if (!src || !ix) continue;
                const w = m.worldMatrix;
                const m0 = w[0]!, m1 = w[1]!, m2 = w[2]!, m4 = w[4]!, m5 = w[5]!, m6 = w[6]!, m8 = w[8]!, m9 = w[9]!, m10 = w[10]!, m12 = w[12]!, m13 = w[13]!, m14 = w[14]!;
                const base = pOff / 3;
                for (let i = 0; i < src.length; i += 3) {
                    const lx = src[i]!, ly = src[i + 1]!, lz = src[i + 2]!;
                    const x = m0 * lx + m4 * ly + m8 * lz + m12;
                    const y = m1 * lx + m5 * ly + m9 * lz + m13;
                    const z = m2 * lx + m6 * ly + m10 * lz + m14;
                    q[pOff++] = x; q[pOff++] = y; q[pOff++] = z;
                    minx = Math.min(minx, x); miny = Math.min(miny, y); minz = Math.min(minz, z);
                    maxx = Math.max(maxx, x); maxy = Math.max(maxy, y); maxz = Math.max(maxz, z);
                }
                for (let i = 0; i < ix.length; i++) idx[iOff++] = ix[i]! + base;
            }
            const cqx = (minx + maxx) / 2, cqy = (miny + maxy) / 2, cqz = (minz + maxz) / 2;
            const maxDim = Math.max(maxx - minx, maxy - miny, maxz - minz) || 1;
            const S = DUCK_TARGET / maxDim;
            const baked = new Float32Array(totalV);
            for (let i = 0; i < totalV; i += 3) {
                baked[i] = S * (q[i]! - cqx); baked[i + 1] = S * (q[i + 1]! - cqy); baked[i + 2] = S * (q[i + 2]! - cqz);
            }
            const g = generateMeshSdf(baked, idx, BAKE);
            if (g.data.length !== phGrid.data.length) {
                // eslint-disable-next-line no-console
                console.warn("[box] duck grid dims differ from placeholder — skipping");
                return;
            }
            const half: [number, number, number] = [(S * (maxx - minx)) / 2, (S * (maxy - miny)) / 2, (S * (maxz - minz)) / 2];
            const mass = BODY_REL_DENSITY * WATER_DENSITY * (volFromGrid(g) || 1);
            const inertia = boxInertia(mass, half);
            const centre: [number, number, number] = [cqx, cqy, cqz];
            const allMeshes: Mesh[] = [];
            // First instance reuses the already-loaded display root; the rest load their own.
            const register = (r: SceneNode, ms: Mesh[], pos: [number, number, number]): void => {
                const index = bodySystem.addBody({ grid: g, mass, inertia, half, position: pos, display: r, scale: S, displayScale: DUCK_MIRROR, centre, externalPose: true });
                duckBodies.push(createDuckPhysics(index, pos, mass, inertia, half));
                allMeshes.push(...ms);
                duckMeshes = allMeshes;
                duckReady = true;
                bodySystem.setEnabled(bodyOn);
                setBodyVisible(bodyOn);
            };
            register(root, meshes, DUCK_SPAWNS[0]!);
            for (let i = 1; i < DUCK_SPAWNS.length; i++) {
                const inst = await loadDuckRoot();
                if (inst) register(inst.root, inst.meshes, DUCK_SPAWNS[i]!);
            }
            // eslint-disable-next-line no-console
            console.warn(`[box] ${bodySystem.bodyCount} duck(s) loaded, scale ${S.toFixed(3)}, grid ${g.dims.join("×")}`);
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[box] duck load failed", err);
        }
    })();

    // Per-frame hybrid update: the reusable system measures submersion from the current Havok pose,
    // then this demo converts that measurement into Havok forces. Havok steps after the core's
    // onBeforeRender callback; onPhysicsAfterStep above copies the resulting pose back to the SDF.
    const updateBody = (dt: number): void => {
        bodySystem.update(dt, ctx.getActiveSim());
        applyDuckForces(dt);
        if (++raftReadoutFrames >= 10) {
            raftReadoutFrames = 0;
            const st = bodySystem.bodyState(0);
            const m = bodySystem.bodyMeasure(0);
            raftReadout.textContent = st && m ? `${bodySystem.bodyCount} duck(s) · #0 y ${st.position[1].toFixed(2)} · sub ${m.submergedCount} · Havok` : "ducks: off";
        }
    };

    // Paddle state: `obstacleAngle` accumulates from `obstacleSpeed` (rad/s).
    let boxScale = 1; // current box-size multiplier (mirrors the box-size slider; snapshotted in pair state)
    let obstacleOn = false;
    let obstacleSpeed = 1.2;
    let obstacleAngle = 0;

    const updatePaddleGeometry = (): void => {
        paddleMesh.scaling.set(2 * OBS_HALF_THICK, 0.5 * (BOX_MAX[1] - BOX_MIN[1]), 2 * obsHalfWidth);
        paddleMesh.position.set(OBS_CENTER[0], BOX_MIN[1] + 0.25 * (BOX_MAX[1] - BOX_MIN[1]), OBS_CENTER[1]);
    };
    const updatePaddleVisibility = (): void => {
        setMeshVisible(paddleMesh, obstacleOn);
        updatePaddleGeometry();
    };

    // Refresh the rotating-paddle sub-block (obs0/obs1) of the box SDF.
    const writeScenePaddle = (): void => {
        engine._device.queue.writeBuffer(
            ctx.sceneSdfBuffer,
            32,
            new Float32Array([
                OBS_CENTER[0],
                OBS_CENTER[1],
                obsHalfWidth,
                OBS_HALF_THICK, // obs0
                Math.cos(obstacleAngle),
                -Math.sin(obstacleAngle),
                obstacleOn ? obstacleSpeed : 0,
                obstacleOn ? 1 : 0, // obs1
            ])
        );
    };
    const writeSdfParams = (): void => {
        engine._device.queue.writeBuffer(ctx.sceneSdfBuffer, 0, new Float32Array([BOX_MIN[0], BOX_MIN[1], BOX_MIN[2], 0, BOX_MAX[0], BOX_MAX[1], BOX_MAX[2], 0]));
        writeScenePaddle();
    };

    // Resize the box footprint live (X/Z only; height fixed). Mutates BOX_MIN/MAX
    // in place so every later scene-SDF write picks up the current size. No reset:
    // the solver's per-step clamp lets the fluid flow into the new shape and the
    // grid bounds are unchanged (no realloc).
    const applyBoxScale = (scale: number): void => {
        boxScale = scale;
        const half = Math.min(BOX_HALF_BASE * scale, BOX_HALF_LIMIT); // clamp inside the grid
        BOX_MIN[0] = -half;
        BOX_MIN[2] = -half;
        BOX_MAX[0] = half;
        BOX_MAX[2] = half;
        boxMesh.scaling.set(BOX_MAX[0] - BOX_MIN[0], BOX_MAX[1] - BOX_MIN[1], BOX_MAX[2] - BOX_MIN[2]);
        obsHalfWidth = OBS_HALF_WIDTH_BASE * scale;
        updatePaddleGeometry();
        rebuildBoxColliders();
        rebuildPaddleCollider();
        writeSdfParams();
    };

    // ── Extra panel controls (box size slider + paddle toggle/speed) ──────
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
    boxSizeInput.max = "4.2";
    boxSizeInput.step = "0.1";
    boxSizeInput.value = "1";
    boxSizeInput.style.cssText = "width:100%;";
    boxSizeInput.oninput = () => {
        const s = parseFloat(boxSizeInput.value);
        boxSizeVal.textContent = `${s.toFixed(1)}×`;
        applyBoxScale(s);
    };
    boxSizeRow.append(boxSizeHead, boxSizeInput);

    const obstacleTitle = document.createElement("div");
    obstacleTitle.textContent = "Obstacle";
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
    speedInput.max = "15";
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

    // ── Floating raft toggle ──
    const raftTitle = document.createElement("div");
    raftTitle.textContent = "Floating body";
    raftTitle.style.cssText = "font-weight:600;margin:8px 0 6px;";
    const raftRow = document.createElement("label");
    raftRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:6px;cursor:pointer;";
    const raftChk = document.createElement("input");
    raftChk.type = "checkbox";
    const raftChkText = document.createElement("span");
    raftChkText.textContent = "Floating ducks (any mesh)";
    raftRow.append(raftChk, raftChkText);
    raftChk.onchange = () => {
        bodyOn = raftChk.checked;
        if (bodyOn) {
            bodySystem.reset();
        }
        setDucksEnabled(bodyOn, bodyOn);
        setBodyVisible(bodyOn);
    };

    return {
        key: "box",
        label: "Box (closed)",
        envUrl: ENV_STUDIO_URL,
        sdf,
        writeSdfParams,
        spawn() {
            return { min: BOX_SPAWN_MIN, max: BOX_SPAWN_MAX };
        },
        emitters() {
            return null;
        },
        onEnter(): void {
            demoActive = true;
            setMeshVisible(boxMesh, containerVisible);
            updatePaddleVisibility();
            setDucksEnabled(bodyOn);
            setBodyVisible(bodyOn);
            setMeshVisible(ctx.ground, false); // the box floor replaces the ground
        },
        onLeave(): void {
            demoActive = false;
            setMeshVisible(boxMesh, false);
            setMeshVisible(paddleMesh, false);
            setDucksEnabled(false);
            setBodyVisible(false);
            updatePaddleBody(); // parks the kinematic paddle body below the world while inactive
            updatePaddleBody(); // parks the kinematic paddle body below the world while inactive
        },
        setContainerVisible(v: boolean): void {
            containerVisible = v;
            setMeshVisible(boxMesh, v);
        },
        containerMeshes(): Mesh[] {
            // Only the translucent glass tank is drawn as a post-fluid overlay (see
            // fluid.ts). The paddle is OPAQUE — it writes depth and must stay in the
            // scene-colour pass so the fluid surface is correctly occluded by it.
            return [boxMesh];
        },
        update(dt: number): void {
            if (obstacleOn) {
                obstacleAngle += obstacleSpeed * dt;
                paddleMesh.rotation.y = obstacleAngle;
            }
            writeScenePaddle();
            updatePaddleBody();
            updateBody(dt);
        },
        demoParams() {
            return [];
        },
        applyParam(): void {
            /* box uses extraControls, not demo params */
        },
        extraControls() {
            return [boxSizeRow, obstacleTitle, obstacleRow, speedRow, raftTitle, raftRow, raftReadout];
        },
        snapshotState(): Record<string, number | boolean> {
            return { boxSize: boxScale, paddleOn: obstacleOn, paddleSpeed: obstacleSpeed };
        },
        restoreState(state: Record<string, number | boolean>): void {
            if (typeof state.boxSize === "number") {
                applyBoxScale(state.boxSize);
                boxSizeInput.value = String(state.boxSize);
                boxSizeVal.textContent = `${state.boxSize.toFixed(1)}×`;
            }
            if (typeof state.paddleOn === "boolean") {
                obstacleOn = state.paddleOn;
                obstacleChk.checked = obstacleOn;
            }
            if (typeof state.paddleSpeed === "number") {
                obstacleSpeed = state.paddleSpeed;
                speedInput.value = String(obstacleSpeed);
                speedVal.textContent = String(obstacleSpeed);
            }
            updatePaddleVisibility();
        },

    };
}
