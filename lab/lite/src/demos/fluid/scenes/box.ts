// Box demo — a closed glass tank sitting on the ground (its floor replaces the
// ground plane). The X/Z footprint is resizable live; a rotating paddle can stir
// the fluid. Camera + interaction are core-owned: LMB orbits, RMB slides,
// Shift+RMB pushes the fluid, and the wheel zooms.

import { addToScene, createBox, createStandardMaterial, loadGltf, setMeshVisible } from "babylon-lite";
import type { Mesh, SceneNode } from "babylon-lite";
import type { SceneSdfSpec } from "babylon-lite/fluid/sim-common.js";
import { createFloatingBodySystem } from "babylon-lite/fluid/floating-body.js";
import { generateMeshSdf } from "babylon-lite/fluid/volume-sampling/index.js";
import type { FluidCtx, FluidDemo } from "../demo.js";
import { ENV_STUDIO_URL } from "../demo.js";

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

export function createBoxDemo(ctx: FluidCtx): FluidDemo {
    const { engine } = ctx;

    // ── Floating rigid bodies (generic mesh → SDF → two-way fluid coupling) ──────────
    // Delegated to the reusable createFloatingBodySystem (packages/.../fluid/floating-body.ts): ANY mesh
    // baked into a LOCAL-space signed-distance grid (generateMeshSdf) is unioned into the fluid's
    // sceneSdf as a MOVING boundary that pushes the water; a GPU reduction + 6-DOF integrator float it.
    // Here we float a rubber DUCK loaded async from glTF (registered once it loads).
    const BODY_HALF = 1.3; // cube half-extent of the bake grid (must contain the scaled duck)
    const BODY_CELL = 0.1; // SDF bake resolution
    const DUCK_TARGET = 2.2; // the duck is uniformly scaled so its LARGEST dimension = this (fits the cube)
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
    });

    // Duck display state (filled by the async load). The system poses each duck root; the demo only
    // toggles the meshes' visibility.
    let bodyOn = false;
    let duckReady = false;
    let duckMeshes: Mesh[] = [];
    // Tuning read-out (scraped by QA + handy for the user).
    const raftReadout = document.createElement("div");
    raftReadout.style.cssText = "color:#9fb4cc;font-size:11px;margin:0 0 6px;";
    raftReadout.textContent = "ducks: off";
    let raftReadoutFrames = 0;

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

    // The floating body's display IS the loaded duck root — the system poses it (TRS) each frame; the
    // demo only toggles the duck meshes' visibility. No fallback mesh: the duck registers once loaded.
    const setBodyVisible = (v: boolean): void => {
        for (const m of duckMeshes) setMeshVisible(m, v && duckReady);
    };

    // ── Load the duck GLB, bake its SDF once, register N instances at staggered start positions ──
    // The bake grid is shared across every instance (deduped by object identity → one GPU upload); each
    // duck gets its OWN display root (the GLB is re-loaded per instance, served from the browser cache).
    const DUCK_SPAWNS: [number, number, number][] = [
        [0, 13, 0],
        [-2.6, 15.5, -2.4],
        [2.6, 18, 2.4],
        [2.6, 20.5, -2.6],
    ];
    const loadDuckRoot = async (): Promise<{ root: SceneNode; meshes: Mesh[] } | null> => {
        const asset = await loadGltf(engine, DUCK_URL);
        asset.animationGroups = undefined;
        addToScene(ctx.scene, asset);
        const root = asset.entities[0] as SceneNode;
        root.position.set(0, 0, 0);
        root.rotationQuaternion.set(0, 0, 0, 1);
        root.scaling.set(1, 1, 1);
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
                bodySystem.addBody({ grid: g, mass, inertia, half, position: pos, display: r, scale: S, centre });
                allMeshes.push(...ms);
                duckMeshes = allMeshes;
                duckReady = true;
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

    // Per-frame rigid-body integration is owned by the reusable floating-body system: it runs the
    // buoyancy reduction, integrates every enabled body, poses the display roots, and writes its own
    // SDF buffer. Here we just drive it and mirror body 0 into the read-out.
    const updateBody = (dt: number): void => {
        bodySystem.update(dt, ctx.getActiveSim());
        if (++raftReadoutFrames >= 10) {
            raftReadoutFrames = 0;
            const st = bodySystem.bodyState(0);
            raftReadout.textContent = st ? `${bodySystem.bodyCount} duck(s) · #0 y ${st.position[1].toFixed(2)} · disp ${st.displaced}` : "ducks: off";
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
        if (bodyOn) bodySystem.reset();
        bodySystem.setEnabled(bodyOn);
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
            setMeshVisible(boxMesh, containerVisible);
            updatePaddleVisibility();
            setBodyVisible(bodyOn);
            setMeshVisible(ctx.ground, false); // the box floor replaces the ground
        },
        onLeave(): void {
            setMeshVisible(boxMesh, false);
            setMeshVisible(paddleMesh, false);
            setBodyVisible(false);
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
