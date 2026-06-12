// Alternative GPU fluid backend — MLS-MPM (Moving Least Squares Material Point
// Method), the algorithm behind matsuoka-601's "Splash". A grid-transfer method
// (no neighbour search), which scales to far more particles than the PBF solver.
//
// Demo-local, exposes the same surface as the PBF sim (positionBuffer +
// debugBuffer in WORLD units, step/addHole/reset/dispose) so the renderer and
// the demo can switch between the two backends at runtime.
//
// Differences from the reference Splash implementation:
//   • Runs in WORLD units with an explicit grid cell size `dx` (Splash uses
//     grid units, dx=1). The quadratic-B-spline inverse-inertia factor is
//     therefore 4/dx² (Splash's literal "4") and gravity is in world units.
//   • The box walls are replaced by the demo's capsule-tank SDF + ground + hole
//     boundary (ported from the PBF apply pass), applied per-particle in G2P.
//   • Shadows / density-grid raymarching are omitted (we keep impostor render).
//
// Per substep: clearGrid → p2g_1 (mass + APIC momentum) → p2g_2 (EOS pressure +
// viscous stress momentum) → updateGrid (v = p/m, gravity, domain walls) → g2p
// (gather v + affine C, advect, capsule/hole boundary). A copy pass then packs
// world positions + speed for the renderer.

import type { EngineContext } from "babylon-lite";
import type { FluidSim, FluidSimOptions } from "./pbf-sim.js";

const WORKGROUP_SIZE = 64;
const MAX_HOLES = 8;
const FIXED_POINT = 1e7; // float→i32 scale for atomic grid accumulation

// Params uniform (std140), 16-byte rows:
//   origin.xyz, dx
//   gridDim.xyz (f32), pad
//   capsuleA.xyz, capsuleRadius
//   capsuleB.xyz, groundY
//   dt, gravity, restDensity, stiffness
//   viscosity, pad, pad, pad
//   counts: numParticles(u32), capsuleMode(u32), holeCount(u32), pad
//   holes[MAX_HOLES]: centre.xyz + radius
const PARAMS_F32 = 7 * 4 + MAX_HOLES * 4; // 7 vec4 header + holes
const PARAMS_BYTES = PARAMS_F32 * 4;
const COUNTS_OFFSET_F32 = 24; // start of the counts vec4 (u32 view)
const HOLE_BASE_F32 = 28;

const COMMON_WGSL = /* wgsl */ `
const FIXED_POINT: f32 = ${FIXED_POINT};
const FIXED_POINT_INV: f32 = ${1 / FIXED_POINT};

struct Params {
    origin: vec4<f32>,      // xyz origin (world), w = dx
    dim: vec4<f32>,         // xyz grid dims (as f32), w unused
    capsuleA: vec4<f32>,    // xyz + radius
    capsuleB: vec4<f32>,    // xyz + groundY
    sim0: vec4<f32>,        // dt, gravity, restDensity, stiffness
    sim1: vec4<f32>,        // viscosity, _, _, _
    counts: vec4<u32>,      // numParticles, capsuleMode, holeCount, _
    holes: array<vec4<f32>, ${MAX_HOLES}>,
};

fn enc(x: f32) -> i32 { return i32(x * FIXED_POINT); }
fn dec(x: i32) -> f32 { return f32(x) * FIXED_POINT_INV; }

fn cellOf(worldPos: vec3<f32>, p: Params) -> vec3<i32> {
    return vec3<i32>(floor((worldPos - p.origin.xyz) / p.origin.w));
}
fn index1D(c: vec3<i32>, p: Params) -> i32 {
    return (c.x * i32(p.dim.y) + c.y) * i32(p.dim.z) + c.z;
}
fn inGrid(c: vec3<i32>, p: Params) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(p.dim.xyz));
}
`;

const PARTICLE_STRUCT = /* wgsl */ `
struct Particle {
    position: vec3<f32>,
    v: vec3<f32>,
    C: mat3x3<f32>,
};
`;

const CLEAR_WGSL = /* wgsl */ `
struct Cell { vx: atomic<i32>, vy: atomic<i32>, vz: atomic<i32>, mass: atomic<i32>, };
@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&cells)) { return; }
    atomicStore(&cells[i].vx, 0);
    atomicStore(&cells[i].vy, 0);
    atomicStore(&cells[i].vz, 0);
    atomicStore(&cells[i].mass, 0);
}`;

// Quadratic B-spline weights from the fractional cell position.
const WEIGHTS_WGSL = /* wgsl */ `
fn weightsOf(worldPos: vec3<f32>, p: Params) -> array<vec3<f32>, 3> {
    let fc = (worldPos - p.origin.xyz) / p.origin.w;
    let base = floor(fc);
    let d = fc - (base + 0.5);
    var w: array<vec3<f32>, 3>;
    w[0] = 0.5 * (0.5 - d) * (0.5 - d);
    w[1] = 0.75 - d * d;
    w[2] = 0.5 * (0.5 + d) * (0.5 + d);
    return w;
}`;

const P2G1_WGSL = /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${WEIGHTS_WGSL}
struct Cell { vx: atomic<i32>, vy: atomic<i32>, vz: atomic<i32>, mass: atomic<i32>, };
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.counts.x) { return; }
    let pos = particles[i].position;
    let v = particles[i].v;
    let C = particles[i].C;
    let base = cellOf(pos, p);
    let w = weightsOf(pos, p);
    let dx = p.origin.w;

    for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
    for (var gz = 0; gz < 3; gz++) {
        let weight = w[gx].x * w[gy].y * w[gz].z;
        let node = base + vec3<i32>(gx - 1, gy - 1, gz - 1);
        if (!inGrid(node, p)) { continue; }
        let nodeCenter = p.origin.xyz + (vec3<f32>(node) + 0.5) * dx;
        let cellDist = nodeCenter - pos;
        let Q = C * cellDist;
        let mass = weight;                  // particle mass = 1
        let vel = mass * (v + Q);
        let idx = index1D(node, p);
        atomicAdd(&cells[idx].mass, enc(mass));
        atomicAdd(&cells[idx].vx, enc(vel.x));
        atomicAdd(&cells[idx].vy, enc(vel.y));
        atomicAdd(&cells[idx].vz, enc(vel.z));
    }}}
}`;

const P2G2_WGSL = /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${WEIGHTS_WGSL}
struct Cell { vx: atomic<i32>, vy: atomic<i32>, vz: atomic<i32>, mass: i32, };
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.counts.x) { return; }
    let pos = particles[i].position;
    let base = cellOf(pos, p);
    let w = weightsOf(pos, p);
    let dx = p.origin.w;
    let dt = p.sim0.x;

    var density = 0.0;
    for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
    for (var gz = 0; gz < 3; gz++) {
        let weight = w[gx].x * w[gy].y * w[gz].z;
        let node = base + vec3<i32>(gx - 1, gy - 1, gz - 1);
        if (!inGrid(node, p)) { continue; }
        density += dec(cells[index1D(node, p)].mass) * weight;
    }}}
    if (density <= 0.0) { return; }

    let volume = 1.0 / density;
    let pressure = max(0.0, p.sim0.w * (density / p.sim0.z - 1.0));
    let dudv = particles[i].C;
    let strain = dudv + transpose(dudv);
    var stress = mat3x3<f32>(-pressure, 0.0, 0.0, 0.0, -pressure, 0.0, 0.0, 0.0, -pressure);
    stress += p.sim1.x * strain;
    let Dinv = 4.0 / (dx * dx);
    let term0 = -volume * Dinv * dt * stress;

    for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
    for (var gz = 0; gz < 3; gz++) {
        let weight = w[gx].x * w[gy].y * w[gz].z;
        let node = base + vec3<i32>(gx - 1, gy - 1, gz - 1);
        if (!inGrid(node, p)) { continue; }
        let nodeCenter = p.origin.xyz + (vec3<f32>(node) + 0.5) * dx;
        let cellDist = nodeCenter - pos;
        let momentum = (term0 * cellDist) * weight;
        let idx = index1D(node, p);
        atomicAdd(&cells[idx].vx, enc(momentum.x));
        atomicAdd(&cells[idx].vy, enc(momentum.y));
        atomicAdd(&cells[idx].vz, enc(momentum.z));
    }}}
}`;

const UPDATE_GRID_WGSL = /* wgsl */ `
${COMMON_WGSL}
struct Cell { vx: i32, vy: i32, vz: i32, mass: i32, };
@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> p: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&cells)) { return; }
    if (cells[i].mass <= 0) { return; }
    let invMass = 1.0 / dec(cells[i].mass);
    var v = vec3<f32>(dec(cells[i].vx), dec(cells[i].vy), dec(cells[i].vz)) * invMass;

    v.y -= p.sim0.y * p.sim0.x; // gravity * dt

    // Domain walls: zero the outward normal velocity in the 2-cell border.
    let dimZ = i32(p.dim.z);
    let dimYZ = i32(p.dim.y) * dimZ;
    let x = i32(i) / dimYZ;
    let y = (i32(i) / dimZ) % i32(p.dim.y);
    let z = i32(i) % dimZ;
    if (x < 2 || x > i32(p.dim.x) - 3) { v.x = 0.0; }
    if (y < 2 || y > i32(p.dim.y) - 3) { v.y = 0.0; }
    if (z < 2 || z > i32(p.dim.z) - 3) { v.z = 0.0; }

    cells[i].vx = enc(v.x);
    cells[i].vy = enc(v.y);
    cells[i].vz = enc(v.z);
}`;

const G2P_WGSL = /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${WEIGHTS_WGSL}
struct Cell { vx: i32, vy: i32, vz: i32, mass: i32, };
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read_write> escaped: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.counts.x) { return; }
    let pos = particles[i].position;
    let base = cellOf(pos, p);
    let w = weightsOf(pos, p);
    let dx = p.origin.w;

    var vel = vec3<f32>(0.0);
    var B = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
    for (var gz = 0; gz < 3; gz++) {
        let weight = w[gx].x * w[gy].y * w[gz].z;
        let node = base + vec3<i32>(gx - 1, gy - 1, gz - 1);
        if (!inGrid(node, p)) { continue; }
        let nodeCenter = p.origin.xyz + (vec3<f32>(node) + 0.5) * dx;
        let cellDist = nodeCenter - pos;
        let idx = index1D(node, p);
        let wv = vec3<f32>(dec(cells[idx].vx), dec(cells[idx].vy), dec(cells[idx].vz)) * weight;
        vel += wv;
        B += mat3x3<f32>(wv * cellDist.x, wv * cellDist.y, wv * cellDist.z);
    }}}

    let Dinv = 4.0 / (dx * dx);
    // Dissipation so the fluid actually comes to rest (pure APIC + EOS is nearly
    // energy-conserving and would slosh forever): damp the APIC affine field
    // toward PIC (sim1.z) and bleed bulk kinetic energy (sim1.y).
    var C = (B * Dinv) * p.sim1.z;
    vel *= p.sim1.y;
    let dt = p.sim0.x;
    var np = pos + vel * dt;
    var esc = escaped[i];

    // Capsule tank boundary (unless the particle has escaped through a hole).
    if (p.counts.y != 0u && esc == 0u) {
        let a = p.capsuleA.xyz;
        let b = p.capsuleB.xyz;
        let r = p.capsuleA.w;
        let ba = b - a;
        let hh = clamp(dot(np - a, ba) / dot(ba, ba), 0.0, 1.0);
        let axisPt = a + ba * hh;
        let radial = np - axisPt;
        let dist = length(radial);
        if (dist > r) {
            var inHole = false;
            for (var k = 0u; k < p.counts.z; k = k + 1u) {
                if (distance(np, p.holes[k].xyz) < p.holes[k].w) { inHole = true; break; }
            }
            if (inHole) {
                esc = 1u;
            } else {
                let n = radial / max(dist, 1e-6);
                np = axisPt + n * r;
                vel -= max(dot(vel, n), 0.0) * n; // remove the outward velocity (no bounce)
            }
        }
    }

    // Ground floor.
    let groundY = p.capsuleB.w;
    if (np.y < groundY) { np.y = groundY; vel.y = max(vel.y, 0.0); }

    // Keep inside the grid domain (2-cell margin).
    let lo = p.origin.xyz + dx * 2.0;
    let hi = p.origin.xyz + (p.dim.xyz - 3.0) * dx;
    np = clamp(np, lo, hi);

    particles[i].position = np;
    particles[i].v = vel;
    particles[i].C = C;
    escaped[i] = esc;
}`;

const COPY_WGSL = /* wgsl */ `
${PARTICLE_STRUCT}
struct Params2 { count: u32, debugScale: f32, _a: f32, _b: f32, };
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> renderPos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dbg: array<f32>;
@group(0) @binding(3) var<uniform> pc: Params2;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.count) { return; }
    renderPos[i] = vec4<f32>(particles[i].position, 1.0);
    dbg[i] = length(particles[i].v);
}`;

export interface MlsMpmOptions extends FluidSimOptions {
    /** Grid cell size in world units (smaller = finer fluid, more cells). Default 0.25. */
    dx?: number;
    /** Equation-of-state stiffness. Default 50. */
    stiffness?: number;
    /** Rest density in particles-per-cell. Default derived from spawn packing. */
    restDensity?: number;
    /** Dynamic (viscous) stress coefficient. Default 0.1. */
    viscosity?: number;
    /** Fixed simulation sub-steps per frame. Default 2. */
    substeps?: number;
    /** Sub-step time (seconds). Default 1/120. */
    subDt?: number;
    /** Per-substep velocity multiplier (<1 bleeds bulk kinetic energy so the
     *  fluid settles to rest). Default 0.98. */
    damping?: number;
    /** Per-substep APIC affine (C) multiplier (<1 blends toward dissipative PIC,
     *  killing residual swirl). Default 0.95. */
    affineDamping?: number;
}

export function createMlsMpmSim(engine: EngineContext, options: MlsMpmOptions = {}): FluidSim {
    const device = engine._device;
    const count = options.count ?? 60000;
    const particleRadius = options.particleRadius ?? 0.09;
    const spawnMin = options.spawnMin ?? [-2, 4, -2];
    const spawnMax = options.spawnMax ?? [2, 12, 2];
    const gravity = options.gravity ?? 9.8;
    const dx = options.dx ?? 0.25;
    const boundsMin = options.boundsMin ?? [-20, 0, -20];
    const boundsMax = options.boundsMax ?? [20, 15, 20];
    const groundY = options.groundY ?? boundsMin[1];
    const capsuleA = options.capsuleA ?? null;
    const capsuleB = options.capsuleB ?? null;
    const capsuleRadius = options.capsuleRadius ?? 0;
    const stiffness = options.stiffness ?? 50;
    const viscosity = options.viscosity ?? 0.1;
    const substeps = options.substeps ?? 2;
    const subDt = options.subDt ?? 1 / 120;
    const damping = options.damping ?? 0.98;
    const affineDamping = options.affineDamping ?? 0.95;

    const gridDim: [number, number, number] = [
        Math.max(4, Math.ceil((boundsMax[0] - boundsMin[0]) / dx)),
        Math.max(4, Math.ceil((boundsMax[1] - boundsMin[1]) / dx)),
        Math.max(4, Math.ceil((boundsMax[2] - boundsMin[2]) / dx)),
    ];
    const numCells = gridDim[0] * gridDim[1] * gridDim[2];

    // Rest density (particles per cell): from the spawn packing if not given.
    const spawnVolCells = Math.max(
        1,
        ((spawnMax[0] - spawnMin[0]) * (spawnMax[1] - spawnMin[1]) * (spawnMax[2] - spawnMin[2])) / (dx * dx * dx),
    );
    const restDensity = options.restDensity ?? Math.max(2, count / spawnVolCells);

    // ── Buffers ──────────────────────────────────────────────────────
    const PARTICLE_STRIDE = 80;
    const particleBuffer = device.createBuffer({ label: "mpm-particles", size: count * PARTICLE_STRIDE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const cellBuffer = device.createBuffer({ label: "mpm-cells", size: numCells * 16, usage: GPUBufferUsage.STORAGE });
    const escapedBuffer = device.createBuffer({ label: "mpm-escaped", size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const positionBuffer = device.createBuffer({ label: "mpm-render-pos", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const debugBuffer = device.createBuffer({ label: "mpm-debug", size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const paramsBuffer = device.createBuffer({ label: "mpm-params", size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const copyParamsBuffer = device.createBuffer({ label: "mpm-copy-params", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const paramsData = new ArrayBuffer(PARAMS_BYTES);
    const pf = new Float32Array(paramsData);
    const pu = new Uint32Array(paramsData);
    pf[0] = boundsMin[0];
    pf[1] = boundsMin[1];
    pf[2] = boundsMin[2];
    pf[3] = dx;
    pf[4] = gridDim[0];
    pf[5] = gridDim[1];
    pf[6] = gridDim[2];
    pf[8] = capsuleA ? capsuleA[0] : 0;
    pf[9] = capsuleA ? capsuleA[1] : 0;
    pf[10] = capsuleA ? capsuleA[2] : 0;
    pf[11] = capsuleRadius;
    pf[12] = capsuleB ? capsuleB[0] : 0;
    pf[13] = capsuleB ? capsuleB[1] : 0;
    pf[14] = capsuleB ? capsuleB[2] : 0;
    pf[15] = groundY;
    pf[16] = subDt;
    pf[17] = gravity;
    pf[18] = restDensity;
    pf[19] = stiffness;
    pf[20] = viscosity;
    pf[21] = damping;
    pf[22] = affineDamping;
    pu[COUNTS_OFFSET_F32] = count;
    pu[COUNTS_OFFSET_F32 + 1] = capsuleA && capsuleB ? 1 : 0;
    pu[COUNTS_OFFSET_F32 + 2] = 0; // holeCount

    {
        const cp = new ArrayBuffer(16);
        new Uint32Array(cp)[0] = count;
        new Float32Array(cp)[1] = 0; // debugScale unused
        device.queue.writeBuffer(copyParamsBuffer, 0, cp);
    }

    function seed(): void {
        const buf = new ArrayBuffer(count * PARTICLE_STRIDE);
        const f = new Float32Array(buf);
        const rp = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) {
            const o = (i * PARTICLE_STRIDE) / 4; // float offset into the particle struct
            const x = spawnMin[0] + Math.random() * (spawnMax[0] - spawnMin[0]);
            const y = spawnMin[1] + Math.random() * (spawnMax[1] - spawnMin[1]);
            const z = spawnMin[2] + Math.random() * (spawnMax[2] - spawnMin[2]);
            f[o] = x;
            f[o + 1] = y;
            f[o + 2] = z;
            // v (o+4..6) and C (o+8..19) left at 0
            rp[i * 4] = x;
            rp[i * 4 + 1] = y;
            rp[i * 4 + 2] = z;
            rp[i * 4 + 3] = 1;
        }
        device.queue.writeBuffer(particleBuffer, 0, buf);
        device.queue.writeBuffer(escapedBuffer, 0, new Uint32Array(count));
        device.queue.writeBuffer(debugBuffer, 0, new Float32Array(count));
        // Seed render positions so the first frame draws the spawn before any step.
        device.queue.writeBuffer(positionBuffer, 0, rp);
    }
    seed();

    function pipeline(label: string, code: string): GPUComputePipeline {
        return device.createComputePipeline({ label, layout: "auto", compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" } });
    }
    const clearPipe = pipeline("mpm-clear", CLEAR_WGSL);
    const p2g1Pipe = pipeline("mpm-p2g1", P2G1_WGSL);
    const p2g2Pipe = pipeline("mpm-p2g2", P2G2_WGSL);
    const updatePipe = pipeline("mpm-update", UPDATE_GRID_WGSL);
    const g2pPipe = pipeline("mpm-g2p", G2P_WGSL);
    const copyPipe = pipeline("mpm-copy", COPY_WGSL);

    const clearBG = device.createBindGroup({ layout: clearPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: cellBuffer } }] });
    const p2g1BG = device.createBindGroup({
        layout: p2g1Pipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: cellBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
        ],
    });
    const p2g2BG = device.createBindGroup({
        layout: p2g2Pipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: cellBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
        ],
    });
    const updateBG = device.createBindGroup({
        layout: updatePipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cellBuffer } },
            { binding: 1, resource: { buffer: paramsBuffer } },
        ],
    });
    const g2pBG = device.createBindGroup({
        layout: g2pPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: cellBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
            { binding: 3, resource: { buffer: escapedBuffer } },
        ],
    });
    const copyBG = device.createBindGroup({
        layout: copyPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: positionBuffer } },
            { binding: 2, resource: { buffer: debugBuffer } },
            { binding: 3, resource: { buffer: copyParamsBuffer } },
        ],
    });

    const particleGroups = Math.ceil(count / WORKGROUP_SIZE);
    const cellGroups = Math.ceil(numCells / WORKGROUP_SIZE);

    function dispatch(encoder: GPUCommandEncoder, label: string, pipe: GPUComputePipeline, bg: GPUBindGroup, groups: number): void {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(groups);
        pass.end();
    }

    let holeWriteSlot = 0;
    let holeActiveCount = 0;

    return {
        count,
        particleRadius,
        positionBuffer,
        debugBuffer,
        debugNorm: 1 / 6,
        step(encoder: GPUCommandEncoder, _dt: number): void {
            device.queue.writeBuffer(paramsBuffer, 0, paramsData);
            for (let s = 0; s < substeps; s++) {
                dispatch(encoder, "mpm-clear", clearPipe, clearBG, cellGroups);
                dispatch(encoder, "mpm-p2g1", p2g1Pipe, p2g1BG, particleGroups);
                dispatch(encoder, "mpm-p2g2", p2g2Pipe, p2g2BG, particleGroups);
                dispatch(encoder, "mpm-update", updatePipe, updateBG, cellGroups);
                dispatch(encoder, "mpm-g2p", g2pPipe, g2pBG, particleGroups);
            }
            dispatch(encoder, "mpm-copy", copyPipe, copyBG, particleGroups);
        },
        addHole(center: [number, number, number], radius: number): void {
            const o = HOLE_BASE_F32 + holeWriteSlot * 4;
            pf[o] = center[0];
            pf[o + 1] = center[1];
            pf[o + 2] = center[2];
            pf[o + 3] = radius;
            holeWriteSlot = (holeWriteSlot + 1) % MAX_HOLES;
            holeActiveCount = Math.min(holeActiveCount + 1, MAX_HOLES);
            pu[COUNTS_OFFSET_F32 + 2] = holeActiveCount;
        },
        reset(): void {
            holeWriteSlot = 0;
            holeActiveCount = 0;
            pu[COUNTS_OFFSET_F32 + 2] = 0;
            pf.fill(0, HOLE_BASE_F32, HOLE_BASE_F32 + MAX_HOLES * 4);
            seed();
        },
        dispose(): void {
            particleBuffer.destroy();
            cellBuffer.destroy();
            escapedBuffer.destroy();
            positionBuffer.destroy();
            debugBuffer.destroy();
            paramsBuffer.destroy();
            copyParamsBuffer.destroy();
        },
    };
}
