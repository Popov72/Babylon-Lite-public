// Reusable N-body floating-rigid-body system for the fluid demos.
//
// Any mesh baked into a LOCAL-space signed-distance grid (see generateMeshSdf) can be floated in the
// fluid with two-way coupling:
//   • mesh → fluid: each body is a MOVING boundary the fluid's `sceneSdf` unions (min), so it pushes /
//     carries the water (the solver recovers the surface velocity from −∂sceneSdf/∂t).
//   • fluid → mesh: a GPU reduction counts the fluid in a thin shell around each submerged hull →
//     Archimedes buoyancy (∝ count, up, at the submerged centroid → self-righting) + a drag-carry that
//     lerps the body toward the local current; a per-body 6-DOF integrator (quaternion) then floats it.
//
// Everything for up to `maxBodies` bodies lives in ONE storage buffer, so the demo needs no extra sim
// bindings: it passes `system.sdfBuffer` as `SceneSdfSpec.sdfGrid`, prepends `system.sdfWgsl` to its
// scene `sdf`, and unions `bodiesSdf(pt, dt)`. Buffer layout (f32):
//   [0] = active-body count, [1..3] = pad,
//   then per body a 24-float (6×vec4) pose/metadata block,
//   then the concatenated per-body SDF grids at GRID_BASE.
// Bodies sharing an identical grid object reuse a single grid slice (deduped by object identity).

import type { SceneNode } from "../scene/scene-node.js";
import type { FluidSim } from "./sim-common.js";

/** A baked local-space signed-distance grid (matches generateMeshSdf's output shape; − inside). */
export interface BodySdfGrid {
    data: Float32Array;
    dims: readonly [number, number, number];
    origin: readonly [number, number, number];
    cellSize: number;
}

export interface FloatingBodyConfig {
    /** Baked SDF grid in the body's centred+scaled LOCAL frame (negative inside the solid). Bodies
     *  passing the SAME grid object share one uploaded slice. */
    grid: BodySdfGrid;
    /** Rigid mass (less than the water it displaces → floats). */
    mass: number;
    /** Diagonal moment of inertia (principal axes ≈ world axes for a plausible float). */
    inertia: readonly [number, number, number];
    /** Body half-extents (world units) — used for the floor clamp. */
    half: readonly [number, number, number];
    /** Initial world position (the body drops from here). */
    position: readonly [number, number, number];
    /** Optional display node, posed each frame as world = pose ∘ (scale, −scale·centre) so it lines up
     *  with the SDF (baked in that centred+scaled frame). The system sets its TRS only — VISIBILITY is
     *  left to the caller. */
    display?: SceneNode;
    /** Uniform display scale (defaults to 1). */
    scale?: number;
    /** Display centre offset in the display node's own frame (defaults to 0). */
    centre?: readonly [number, number, number];
}

export interface FloatingBodySystemOptions {
    /** Max simultaneous bodies (buffer sizing). Default 8. */
    maxBodies?: number;
    /** Max f32 elements in any one body's grid data — sizes the grid arena. REQUIRED so the storage
     *  buffer (and hence `sdfBuffer`) exists up-front for `SceneSdfSpec.sdfGrid`. */
    gridFloats: number;
    /** Shell thickness (world) counted for buoyancy. Default 0.3. */
    band?: number;
    /** Buoyancy per shell particle. Default 0.02. */
    buoyK?: number;
    /** Gravity. Default 9.8. */
    gravity?: number;
    /** Vertical velocity damping (1/s). Default 6. */
    linDrag?: number;
    /** Angular velocity damping (1/s). Default 3.5. */
    angDrag?: number;
    /** EMA rate (1/s) smoothing the read-back displaced count. Default 8. */
    buoyEma?: number;
    /** Current drag-carry per shell-particle·s. Default 0.004. */
    dragCarry?: number;
    /** Optional live world-AABB clamp: bodies are kept inside [min,max] (XZ inset by `wallMargin`, plus
     *  a floor at min.y + half.y). Return null to skip clamping this frame. */
    getBounds?: () => { min: readonly [number, number, number]; max: readonly [number, number, number] } | null;
    /** XZ inset from the bounds walls (world units). Default 0.5. */
    wallMargin?: number;
}

export interface FloatingBodySystem {
    /** Register a body (uploads its grid; identical grid objects share one slice). Returns its index. */
    addBody(cfg: FloatingBodyConfig): number;
    /** Number of registered bodies. */
    readonly bodyCount: number;
    /** Live pose + last read-back displaced-particle count for body `i` (for readouts/QA). */
    bodyState(i: number): { position: [number, number, number]; displaced: number } | null;
    /** Storage buffer to pass as `SceneSdfSpec.sdfGrid` (valid immediately). */
    readonly sdfBuffer: GPUBuffer;
    /** WGSL: quaternion helpers + `fn bodiesSdf(pt: vec3<f32>, dt: f32) -> f32` (reads `sceneSdfGrid`).
     *  Prepend to the scene `sdf` string, before the `sceneSdf` fn that calls `bodiesSdf`. */
    readonly sdfWgsl: string;
    /** Per-frame: reduction → integrate every enabled body → write poses + pose display nodes. */
    update(dt: number, sim: FluidSim): void;
    /** Enable/disable physics for ALL bodies (active flag in the buffer). Display visibility is the
     *  caller's responsibility. */
    setEnabled(on: boolean): void;
    /** Whether physics is currently enabled. */
    readonly enabled: boolean;
    /** Re-drop every body at its start pose with zero velocity. */
    reset(): void;
    /** Release GPU resources. */
    dispose(): void;
}

const HEADER_FLOATS = 4; // [0]=count, [1..3]=pad
const BODY_STRIDE = 24; // per-body pose/metadata block (6 × vec4)
const RED_WG = 256;
const RED_FP = 256; // fixed-point scale for summed positions/velocities
const RED_SLOTS = 7; // per body: count, sumPos.xyz, sumVel.xyz

interface Body {
    gridOffset: number; // float index into the grid-data region
    dims: [number, number, number];
    origin: [number, number, number];
    invCell: number;
    mass: number;
    inertia: [number, number, number];
    half: [number, number, number];
    start: [number, number, number];
    pos: [number, number, number];
    quat: [number, number, number, number];
    linVel: [number, number, number];
    angVel: [number, number, number];
    smCount: number; // EMA-smoothed displaced count
    display?: SceneNode;
    scale: number;
    centre: [number, number, number];
}

const quatRotate = (q: readonly [number, number, number, number], v: readonly [number, number, number]): [number, number, number] => {
    const [x, y, z, w] = q;
    const tx = 2 * (y * v[2] - z * v[1]),
        ty = 2 * (z * v[0] - x * v[2]),
        tz = 2 * (x * v[1] - y * v[0]);
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
};

const clampAbs = (v: number, m: number): number => Math.max(-m, Math.min(m, v));

export function createFloatingBodySystem(device: GPUDevice, opts: FloatingBodySystemOptions): FloatingBodySystem {
    const maxBodies = opts.maxBodies ?? 8;
    const BAND = opts.band ?? 0.3;
    const BUOY_K = opts.buoyK ?? 0.02;
    const GRAVITY = opts.gravity ?? 9.8;
    const LIN_DRAG = opts.linDrag ?? 6;
    const ANG_DRAG = opts.angDrag ?? 3.5;
    const BUOY_EMA = opts.buoyEma ?? 8;
    const DRAG_CARRY = opts.dragCarry ?? 0.004;
    const WALL_MARGIN = opts.wallMargin ?? 0.5;
    const getBounds = opts.getBounds;
    const GRID_BASE = HEADER_FLOATS + maxBodies * BODY_STRIDE;
    const gridArena = opts.gridFloats * maxBodies;

    const bodies: Body[] = [];
    const gridSlices = new Map<BodySdfGrid, number>(); // dedupe identical grids → shared offset
    let gridFill = 0; // floats of grid data written so far
    let enabled = false;

    const totalFloats = GRID_BASE + gridArena;
    const sdfBuffer = device.createBuffer({ label: "floating-bodies-sdf", size: totalFloats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const sdfHeader = new Float32Array(GRID_BASE); // CPU staging for the header, re-written each frame

    const addBody = (cfg: FloatingBodyConfig): number => {
        if (bodies.length >= maxBodies) {
            throw new Error("floating-body: maxBodies exceeded");
        }
        // Upload (or reuse) this grid's data slice.
        let offset = gridSlices.get(cfg.grid);
        if (offset === undefined) {
            const gridLen = cfg.grid.data.length;
            if (gridFill + gridLen > gridArena) {
                throw new Error("floating-body: grid arena exhausted (raise gridFloats/maxBodies)");
            }
            offset = gridFill;
            const gd = new Float32Array(gridLen);
            gd.set(cfg.grid.data);
            device.queue.writeBuffer(sdfBuffer, (GRID_BASE + offset) * 4, gd);
            gridSlices.set(cfg.grid, offset);
            gridFill += gridLen;
        }
        const b: Body = {
            gridOffset: offset,
            dims: [cfg.grid.dims[0], cfg.grid.dims[1], cfg.grid.dims[2]],
            origin: [cfg.grid.origin[0], cfg.grid.origin[1], cfg.grid.origin[2]],
            invCell: 1 / cfg.grid.cellSize,
            mass: cfg.mass,
            inertia: [cfg.inertia[0], cfg.inertia[1], cfg.inertia[2]],
            half: [cfg.half[0], cfg.half[1], cfg.half[2]],
            start: [cfg.position[0], cfg.position[1], cfg.position[2]],
            pos: [cfg.position[0], cfg.position[1], cfg.position[2]],
            quat: [0, 0, 0, 1],
            linVel: [0, 0, 0],
            angVel: [0, 0, 0],
            smCount: 0,
            display: cfg.display,
            scale: cfg.scale ?? 1,
            centre: cfg.centre ? [cfg.centre[0], cfg.centre[1], cfg.centre[2]] : [0, 0, 0],
        };
        bodies.push(b);
        syncDisplay(b);
        writeHeader();
        return bodies.length - 1;
    };

    // Write every body's live pose + grid metadata into the header region.
    const writeHeader = (): void => {
        sdfHeader[0] = bodies.length;
        for (let i = 0; i < bodies.length; i++) {
            const b = bodies[i]!;
            const o = HEADER_FLOATS + i * BODY_STRIDE;
            sdfHeader[o] = b.pos[0];
            sdfHeader[o + 1] = b.pos[1];
            sdfHeader[o + 2] = b.pos[2];
            sdfHeader[o + 3] = enabled ? 1 : 0;
            sdfHeader[o + 4] = b.quat[0];
            sdfHeader[o + 5] = b.quat[1];
            sdfHeader[o + 6] = b.quat[2];
            sdfHeader[o + 7] = b.quat[3];
            sdfHeader[o + 8] = b.linVel[0];
            sdfHeader[o + 9] = b.linVel[1];
            sdfHeader[o + 10] = b.linVel[2];
            sdfHeader[o + 11] = b.invCell;
            sdfHeader[o + 12] = b.angVel[0];
            sdfHeader[o + 13] = b.angVel[1];
            sdfHeader[o + 14] = b.angVel[2];
            sdfHeader[o + 15] = b.gridOffset;
            sdfHeader[o + 16] = b.origin[0];
            sdfHeader[o + 17] = b.origin[1];
            sdfHeader[o + 18] = b.origin[2];
            sdfHeader[o + 19] = 0;
            sdfHeader[o + 20] = b.dims[0];
            sdfHeader[o + 21] = b.dims[1];
            sdfHeader[o + 22] = b.dims[2];
            sdfHeader[o + 23] = 0;
        }
        device.queue.writeBuffer(sdfBuffer, 0, sdfHeader);
    };

    const syncDisplay = (b: Body): void => {
        const n = b.display;
        if (!n) {
            return;
        }
        n.scaling.set(b.scale, b.scale, b.scale);
        n.rotationQuaternion.set(b.quat[0], b.quat[1], b.quat[2], b.quat[3]);
        const r = quatRotate(b.quat, [b.scale * b.centre[0], b.scale * b.centre[1], b.scale * b.centre[2]]);
        n.position.set(b.pos[0] - r[0], b.pos[1] - r[1], b.pos[2] - r[2]);
    };

    // ── GPU buoyancy reduction (all bodies in one dispatch) ──
    const redWgsl = `
const HB = ${HEADER_FLOATS}u;
const STRIDE = ${BODY_STRIDE}u;
const GBASE = ${GRID_BASE}u;
const FP = ${RED_FP}.0;
const BAND = ${BAND};
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> sdf: array<f32>;
@group(0) @binding(3) var<storage, read_write> acc: array<atomic<i32>>;
fn qRot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> { let u = q.xyz; return v + 2.0 * cross(u, cross(u, v) + q.w * v); }
fn qConj(q: vec4<f32>) -> vec4<f32> { return vec4<f32>(-q.xyz, q.w); }
fn gLoad(off: u32, i: i32, j: i32, k: i32, d: vec3<i32>) -> f32 {
    let c = clamp(vec3<i32>(i,j,k), vec3<i32>(0), d - vec3<i32>(1));
    return sdf[GBASE + off + u32(c.x + d.x * (c.y + d.y * c.z))];
}
fn sampleG(off: u32, pt: vec3<f32>, origin: vec3<f32>, invCell: f32, d: vec3<i32>) -> f32 {
    let g = (pt - origin) * invCell; let b = floor(g); let f = g - b;
    let i = i32(b.x); let j = i32(b.y); let k = i32(b.z);
    let c000 = gLoad(off,i,j,k,d); let c100 = gLoad(off,i+1,j,k,d); let c010 = gLoad(off,i,j+1,k,d); let c110 = gLoad(off,i+1,j+1,k,d);
    let c001 = gLoad(off,i,j,k+1,d); let c101 = gLoad(off,i+1,j,k+1,d); let c011 = gLoad(off,i,j+1,k+1,d); let c111 = gLoad(off,i+1,j+1,k+1,d);
    let x00 = mix(c000,c100,f.x); let x10 = mix(c010,c110,f.x); let x01 = mix(c001,c101,f.x); let x11 = mix(c011,c111,f.x);
    return mix(mix(x00,x10,f.y), mix(x01,x11,f.y), f.z);
}
@compute @workgroup_size(${RED_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&positions)) { return; }
    let p = positions[i].xyz;
    let v = velocities[i].xyz;
    let n = u32(sdf[0]);
    for (var b = 0u; b < n; b = b + 1u) {
        let o = HB + b * STRIDE;
        if (sdf[o + 3u] < 0.5) { continue; }
        let pos = vec3<f32>(sdf[o], sdf[o + 1u], sdf[o + 2u]);
        let quat = vec4<f32>(sdf[o + 4u], sdf[o + 5u], sdf[o + 6u], sdf[o + 7u]);
        let local = qRot(qConj(quat), p - pos);
        let origin = vec3<f32>(sdf[o + 16u], sdf[o + 17u], sdf[o + 18u]);
        let invCell = sdf[o + 11u];
        let dims = vec3<i32>(i32(sdf[o + 20u]), i32(sdf[o + 21u]), i32(sdf[o + 22u]));
        // Skip particles outside the grid extent: sampleG clamps to the edge there and would count far
        // water into the shell (→ runaway buoyancy).
        let gmax = origin + vec3<f32>(dims) / invCell;
        if (any(local < origin) || any(local > gmax)) { continue; }
        let d = sampleG(u32(sdf[o + 15u]), local, origin, invCell, dims);
        if (d < BAND) {
            let s = b * ${RED_SLOTS}u;
            atomicAdd(&acc[s], 1);
            atomicAdd(&acc[s + 1u], i32(p.x * FP));
            atomicAdd(&acc[s + 2u], i32(p.y * FP));
            atomicAdd(&acc[s + 3u], i32(p.z * FP));
            atomicAdd(&acc[s + 4u], i32(v.x * FP));
            atomicAdd(&acc[s + 5u], i32(v.y * FP));
            atomicAdd(&acc[s + 6u], i32(v.z * FP));
        }
    }
}`;
    let redPipe: GPUComputePipeline | null = null;
    let accBuf: GPUBuffer | null = null;
    let redBG: GPUBindGroup | null = null;
    let boundPos: GPUBuffer | null = null;
    const accBytes = maxBodies * RED_SLOTS * 4;
    const staging: GPUBuffer[] = [];
    const busy: boolean[] = [];
    const disp = new Int32Array(maxBodies * RED_SLOTS); // latest decoded per-body accumulators

    const ensureRed = (): void => {
        if (!redPipe) {
            redPipe = device.createComputePipeline({
                label: "floating-bodies-buoyancy",
                layout: "auto",
                compute: { module: device.createShaderModule({ code: redWgsl }), entryPoint: "main" },
            });
        }
        if (!accBuf) {
            accBuf = device.createBuffer({ label: "floating-bodies-acc", size: accBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        }
        if (staging.length === 0) {
            for (let i = 0; i < 2; i++) {
                staging.push(device.createBuffer({ label: `floating-bodies-staging-${i}`, size: accBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
                busy.push(false);
            }
        }
    };

    const runReduction = (sim: FluidSim): void => {
        ensureRed();
        if (boundPos !== sim.positionBuffer || !redBG) {
            boundPos = sim.positionBuffer;
            redBG = device.createBindGroup({
                layout: redPipe!.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: sim.positionBuffer } },
                    { binding: 1, resource: { buffer: sim.velocityBuffer } },
                    { binding: 2, resource: { buffer: sdfBuffer } },
                    { binding: 3, resource: { buffer: accBuf! } },
                ],
            });
        }
        const slot = busy[0] ? (busy[1] ? -1 : 1) : 0;
        if (slot < 0) {
            return;
        } // both readbacks in flight — keep last value this frame
        const enc = device.createCommandEncoder({ label: "floating-bodies-buoyancy" });
        enc.clearBuffer(accBuf!, 0, accBytes);
        const pass = enc.beginComputePass();
        pass.setPipeline(redPipe!);
        pass.setBindGroup(0, redBG!);
        pass.dispatchWorkgroups(Math.ceil(sim.count / RED_WG));
        pass.end();
        const st = staging[slot]!;
        enc.copyBufferToBuffer(accBuf!, 0, st, 0, accBytes);
        device.queue.submit([enc.finish()]);
        busy[slot] = true;
        void st
            .mapAsync(GPUMapMode.READ)
            .then(() => {
                disp.set(new Int32Array(st.getMappedRange()));
                st.unmap();
                busy[slot] = false;
            })
            .catch(() => {
                busy[slot] = false;
            });
    };

    const update = (dt: number, sim: FluidSim): void => {
        if (enabled && bodies.length > 0) {
            runReduction(sim);
        }
        const bounds = getBounds ? getBounds() : null;
        for (let i = 0; i < bodies.length; i++) {
            const b = bodies[i]!;
            if (enabled) {
                const s = i * RED_SLOTS;
                const count = disp[s] ?? 0;
                const sx = (disp[s + 1] ?? 0) / RED_FP,
                    sz = (disp[s + 3] ?? 0) / RED_FP;
                const vx = (disp[s + 4] ?? 0) / RED_FP,
                    vz = (disp[s + 6] ?? 0) / RED_FP;
                b.smCount += (count - b.smCount) * Math.min(BUOY_EMA * dt, 1);
                const buoy = BUOY_K * b.smCount;
                const cx = count > 0 ? sx / count : b.pos[0];
                const cz = count > 0 ? sz / count : b.pos[2];
                // Vertical: gravity + buoyancy, heavily damped.
                b.linVel[1] += ((buoy - b.mass * GRAVITY) / b.mass) * dt;
                b.linVel[1] *= Math.max(0, 1 - LIN_DRAG * dt);
                // Horizontal: carried by the local current.
                if (count > 0) {
                    const a = Math.min(DRAG_CARRY * count * dt, 0.6);
                    b.linVel[0] += a * (vx / count - b.linVel[0]);
                    b.linVel[2] += a * (vz / count - b.linVel[2]);
                } else {
                    const ad = Math.max(0, 1 - 0.5 * dt);
                    b.linVel[0] *= ad;
                    b.linVel[2] *= ad;
                }
                for (let k = 0; k < 3; k++) {
                    b.linVel[k] = clampAbs(b.linVel[k]!, 12);
                }
                // Righting torque from the submerged centroid: tau = r × (0, buoy, 0).
                const rx = cx - b.pos[0],
                    rz = cz - b.pos[2];
                b.angVel[0] += ((-rz * buoy) / b.inertia[0]) * dt;
                b.angVel[2] += ((rx * buoy) / b.inertia[2]) * dt;
                const angDamp = Math.max(0, 1 - ANG_DRAG * dt);
                for (let k = 0; k < 3; k++) {
                    b.angVel[k] = clampAbs(b.angVel[k]! * angDamp, 6);
                }
                // Integrate position + orientation.
                b.pos[0] += b.linVel[0] * dt;
                b.pos[1] += b.linVel[1] * dt;
                b.pos[2] += b.linVel[2] * dt;
                const [qx, qy, qz, qw] = b.quat;
                const [wx, wy, wz] = b.angVel;
                let nqx = qx + 0.5 * (wx * qw + wy * qz - wz * qy) * dt;
                let nqy = qy + 0.5 * (wy * qw + wz * qx - wx * qz) * dt;
                let nqz = qz + 0.5 * (wz * qw + wx * qy - wy * qx) * dt;
                let nqw = qw + 0.5 * (-wx * qx - wy * qy - wz * qz) * dt;
                const nl = Math.hypot(nqx, nqy, nqz, nqw) || 1;
                nqx /= nl;
                nqy /= nl;
                nqz /= nl;
                nqw /= nl;
                b.quat[0] = nqx;
                b.quat[1] = nqy;
                b.quat[2] = nqz;
                b.quat[3] = nqw;
                // Wall + floor clamp.
                if (bounds) {
                    b.pos[0] = Math.max(bounds.min[0] + WALL_MARGIN, Math.min(bounds.max[0] - WALL_MARGIN, b.pos[0]));
                    b.pos[2] = Math.max(bounds.min[2] + WALL_MARGIN, Math.min(bounds.max[2] - WALL_MARGIN, b.pos[2]));
                    if (b.pos[1] < bounds.min[1] + b.half[1]) {
                        b.pos[1] = bounds.min[1] + b.half[1];
                        if (b.linVel[1] < 0) {
                            b.linVel[1] = 0;
                        }
                    }
                }
            }
            syncDisplay(b);
        }
        writeHeader();
    };

    const setEnabled = (on: boolean): void => {
        enabled = on;
        writeHeader();
    };

    const reset = (): void => {
        for (const b of bodies) {
            b.pos = [...b.start];
            b.quat = [0, 0, 0, 1];
            b.linVel = [0, 0, 0];
            b.angVel = [0, 0, 0];
            b.smCount = 0;
            syncDisplay(b);
        }
        disp.fill(0);
        writeHeader();
    };

    const dispose = (): void => {
        sdfBuffer.destroy();
        accBuf?.destroy();
        for (const st of staging) {
            st.destroy();
        }
    };

    // WGSL the demo prepends to its scene SDF (reads the `sceneSdfGrid` storage binding the sim injects).
    const sdfWgsl = `
fn fbQuatRotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> { let u = q.xyz; return v + 2.0 * cross(u, cross(u, v) + q.w * v); }
fn fbQuatConj(q: vec4<f32>) -> vec4<f32> { return vec4<f32>(-q.xyz, q.w); }
fn fbQuatMul(a: vec4<f32>, bb: vec4<f32>) -> vec4<f32> { return vec4<f32>(a.w * bb.xyz + bb.w * a.xyz + cross(a.xyz, bb.xyz), a.w * bb.w - dot(a.xyz, bb.xyz)); }
fn fbQuatIntegrate(q: vec4<f32>, w: vec3<f32>, dt: f32) -> vec4<f32> { return normalize(q + 0.5 * fbQuatMul(vec4<f32>(w * dt, 0.0), q)); }
fn fbGLoad(off: u32, i: i32, j: i32, k: i32, d: vec3<i32>) -> f32 {
    let c = clamp(vec3<i32>(i,j,k), vec3<i32>(0), d - vec3<i32>(1));
    return sceneSdfGrid[${GRID_BASE}u + off + u32(c.x + d.x * (c.y + d.y * c.z))];
}
fn fbSampleG(off: u32, pt: vec3<f32>, origin: vec3<f32>, invCell: f32, d: vec3<i32>) -> f32 {
    let g = (pt - origin) * invCell; let b = floor(g); let f = g - b;
    let i = i32(b.x); let j = i32(b.y); let k = i32(b.z);
    let c000 = fbGLoad(off,i,j,k,d); let c100 = fbGLoad(off,i+1,j,k,d); let c010 = fbGLoad(off,i,j+1,k,d); let c110 = fbGLoad(off,i+1,j+1,k,d);
    let c001 = fbGLoad(off,i,j,k+1,d); let c101 = fbGLoad(off,i+1,j,k+1,d); let c011 = fbGLoad(off,i,j+1,k+1,d); let c111 = fbGLoad(off,i+1,j+1,k+1,d);
    let x00 = mix(c000,c100,f.x); let x10 = mix(c010,c110,f.x); let x01 = mix(c001,c101,f.x); let x11 = mix(c011,c111,f.x);
    return mix(mix(x00,x10,f.y), mix(x01,x11,f.y), f.z);
}
fn bodiesSdf(pt: vec3<f32>, dt: f32) -> f32 {
    var dmin = 1.0e9;
    let n = u32(sceneSdfGrid[0]);
    for (var b = 0u; b < n; b = b + 1u) {
        let o = ${HEADER_FLOATS}u + b * ${BODY_STRIDE}u;
        if (sceneSdfGrid[o + 3u] < 0.5) { continue; }
        let pos = vec3<f32>(sceneSdfGrid[o], sceneSdfGrid[o + 1u], sceneSdfGrid[o + 2u]) + vec3<f32>(sceneSdfGrid[o + 8u], sceneSdfGrid[o + 9u], sceneSdfGrid[o + 10u]) * dt;
        let invCell = sceneSdfGrid[o + 11u];
        let dims = vec3<i32>(i32(sceneSdfGrid[o + 20u]), i32(sceneSdfGrid[o + 21u]), i32(sceneSdfGrid[o + 22u]));
        // Cheap world-space bounding-sphere reject FIRST (a single length(), no quaternion): the grid
        // box has half-diagonal 0.5·|dims/invCell|, so points beyond that from the body centre are far
        // outside and can be skipped before paying for the quaternion transform + trilinear sample. This
        // is what keeps N bodies cheap for many-eval solvers (PB-MPM evaluates sceneSdf per cell × its
        // iteration loop) — the vast majority of cells reject here on the length() alone.
        let boundR = 0.5 * length(vec3<f32>(dims) / invCell);
        if (dot(pt - pos, pt - pos) > boundR * boundR) { continue; }
        let quat = fbQuatIntegrate(vec4<f32>(sceneSdfGrid[o + 4u], sceneSdfGrid[o + 5u], sceneSdfGrid[o + 6u], sceneSdfGrid[o + 7u]), vec3<f32>(sceneSdfGrid[o + 12u], sceneSdfGrid[o + 13u], sceneSdfGrid[o + 14u]), dt);
        let local = fbQuatRotate(fbQuatConj(quat), pt - pos);
        let origin = vec3<f32>(sceneSdfGrid[o + 16u], sceneSdfGrid[o + 17u], sceneSdfGrid[o + 18u]);
        let gmax = origin + vec3<f32>(dims) / invCell;
        if (any(local < origin) || any(local > gmax)) { continue; }
        let bd = fbSampleG(u32(sceneSdfGrid[o + 15u]), local, origin, invCell, dims);
        dmin = min(dmin, bd);
    }
    return dmin;
}`;

    return {
        addBody,
        get bodyCount(): number {
            return bodies.length;
        },
        bodyState(i: number): { position: [number, number, number]; displaced: number } | null {
            const b = bodies[i];
            if (!b) {
                return null;
            }
            return { position: [b.pos[0], b.pos[1], b.pos[2]], displaced: disp[i * RED_SLOTS] ?? 0 };
        },
        sdfBuffer,
        sdfWgsl,
        update,
        setEnabled,
        get enabled(): boolean {
            return enabled;
        },
        reset,
        dispose,
    };
}
