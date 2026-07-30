// GPU fluid simulation — Phases 1–3 (Position Based Fluids).
//
// References:
//   • PBF: Macklin & Müller 2013, "Position Based Fluids" —
//     https://mmacklin.com/pbf_sig_preprint.pdf
//   • Foam (spray/foam/bubbles): Ihmsen et al. 2012, "Unified spray, foam and air bubbles for
//     particle-based fluids" —
//     https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf
//
// Demo-local (not babylon-lite core). Builds WebGPU compute pipelines from
// `engine._device` and encodes the per-frame passes into the frame command
// encoder (see the demo's onBeforeRender → sim.step).
//
// Phase 1: gravity integration + ground bounce.
// Phase 2: spatial-hash neighbour grid (fixed-capacity uniform grid).
// Phase 3: PBF density-constraint solver (Macklin & Müller 2013) on top of the
//   grid — the fluid becomes incompressible, holds volume, pools and splashes.
//
// PBF per frame (encoded as a chain of compute passes):
//   1. predict      x* = x + (v + g·dt)·dt              (predicted positions)
//   2. sort         counting-sort particles by cell (histogram → prefix sum →
//                   scatter) building sortedIdx (slot → original index)
//   3. reorder-in   copy the live working set (predicted/pos/vel) into sorted order
//   4. × iterations (all dispatched over cell-SORTED slots):
//        a. lambda  ρ_i (poly6) → C_i = ρ_i/ρ0 − 1 → λ_i = −C_i / (Σ|∇C|² + ε)
//        b. delta   Δp_i = (1/ρ0) Σ_j (λ_i+λ_j+s_corr) ∇W_spiky(x*_i − x*_j)
//        c. apply   x* += Δp; confine to the sceneSdf + clamp (in place)
//   5. finalize     v = (x* − x0)/dt                     (x0 = frame-start position)
//   6. viscosity    XSPH smoothing v += c/ρ0 Σ_j (v_j−v_i) W (double-buffered via
//                   sortedPos0 to avoid a read/write race, then copied back to sortedVel)
//   7. scatter-back write the solved sorted pos/vel back to original order (+ speed)
//   8. foam?        normals → emit → update, also over sorted slots
//
// Neighbour grid — COUNTING-SORT uniform grid (Hoetzlein 2014, fast fixed-radius
//   nearest neighbours): cellSize equals the smoothing radius h so neighbours lie
//   in the 3×3×3 cell stencil. Each frame a counting sort by cell produces
//   cellCount[cell] (population), cellStart[cell] (exclusive prefix sum) and
//   sortedIdx[] (original indices in cell-sorted order). Instead of solving in
//   original order and reading neighbours through the sort, the WHOLE solve runs in
//   cell-SORTED order (Fluids v5.0 countingSortFull): reorder-in copies the working
//   set into the sorted buffers ONCE, every neighbour pass dispatches over slot k so
//   consecutive threads are spatially adjacent particles that reuse the same
//   neighbour cells in cache, and scatter-back writes results to original order ONCE.
//   The sorted working buffers are mutated in place across the iterations (no
//   per-iteration gathers). There is no per-cell capacity cap.

import type { EngineContext } from "../engine/engine.js";
import type { FluidSim, FluidSimBaseOptions, SceneSdfSpec, EmitterConfig, ForceFieldSpec, FoamConfig, DiffusePool, FluidProfiler } from "./sim-common.js";
import {
    EMITTERS_FLOATS,
    SPAWN_ACCEPT_TRIES,
    EMITTER_STRUCT_WGSL,
    EMITTER_SPAWN_WGSL,
    FOAM_BYTES,
    FOAM_COMMON_WGSL,
    packEmitters,
    SCENE_NORMAL_WGSL,
    SCENE_SDF_GRID_WGSL,
} from "./sim-common.js";

// Opt-in GPU timing hook (see FluidProfiler / lab gpu-profiler.ts). Module-scoped:
// null by default so `profiler?.pass(...)` is undefined and timing costs nothing.
// The lab installs a profiler via the returned task's setProfiler().
let profiler: FluidProfiler | null = null;

export interface PbfOptions extends FluidSimBaseOptions {
    /** Smoothing radius h = neighbour-grid cell size (world units). Default 0.4. */
    smoothingRadius?: number;
    /** Simulation box min corner — the neighbour-grid domain AABB. Default [-4, 0, -4]. */
    boundsMin?: [number, number, number];
    /** Simulation box max corner — the neighbour-grid domain AABB. Default [4, 20, 4]. */
    boundsMax?: [number, number, number];
    /** Capsule tank boundary: centre of the bottom hemisphere. Default null (box boundary). */
    capsuleA?: [number, number, number];
    /** Capsule tank boundary: centre of the top hemisphere. */
    capsuleB?: [number, number, number];
    /** Capsule tank radius. */
    capsuleRadius?: number;
    /** Ground plane height; particles are floored at y = groundY. Default boundsMin.y. */
    groundY?: number;
    /** Explicit PBF rest density. If omitted, derived from spawn number density × restDensityScale. */
    restDensity?: number;
    /** Multiplier applied to the derived rest density. Default 1.0. */
    restDensityScale?: number;
    /** PBF constraint solver iterations per frame. Default 3. */
    iterations?: number;
    /** Constraint-force relaxation ε (added to the λ denominator). Default 50. */
    relaxation?: number;
    /** XSPH viscosity coefficient (0 = none). Default 0.08. */
    viscosity?: number;
    /** Artificial-pressure (s_corr) strength — counters tensile instability and
     *  keeps a free surface from collapsing under its own cohesion. Default 0.02. */
    scorr?: number;
    /** Boundary density support: fraction of ρ₀ added to particles touching a
     *  solid wall, compensating the SPH density deficiency there (0 = off). Default 0. */
    boundaryDensity?: number;
    /** Max particles stored per grid cell. Default 64. */
    maxPerCell?: number;
    /** Explicit per-particle seed positions as flat world-space xyz triples
     *  (`[x0,y0,z0, x1,y1,z1, …]`). When present, `seed()`/`reset()` places each
     *  particle `i (< count)` at `initialPositions[3i..3i+2]` with zero velocity —
     *  instead of drawing a random point in the spawn box. Size `count` to
     *  `initialPositions.length / 3` so every particle is real. Absent (the default)
     *  restores the random-in-spawn-box behaviour. Used to fill a mesh with a
     *  volume-sampled particle set (see fluid/volume-sampling), mirroring the same
     *  hook on the MLS-MPM backend. */
    initialPositions?: Float32Array;
}

const WORKGROUP_SIZE = 64;
// WebGPU caps a dispatch at 65535 workgroups per dimension. The neighbour grid
// can need far more groups than that at small particle sizes (very fine cells),
// so cell-indexed dispatches spill the overflow into a second (y) dimension and
// the cell kernels rebuild the linear index from num_workgroups.x.
const MAX_WORKGROUPS = 65535;
// Counting-sort prefix-sum workgroup width (matches the MLS-MPM backend). The
// per-chunk exclusive scan processes SCAN_WG cells per workgroup.
const SCAN_WG = 256;

// Sim uniform. Per-frame mutable (dt); the rest are constant or set on demand.
//   [0] dt        [1] gravity   [2] restDensity [3] h
//   [4] h2        [5] poly6     [6] spikyGrad   [7] eps
//   [8] scorrK    [9] scorrInvWdq [10] scorrN   [11] viscosity
//   [12] count(u32) [13] containerMode(u32) [14] boundaryDensity [15] live(u32)
//   [16..19] boundsMin.xyz + pad
//   [20..23] boundsMax.xyz + pad
//   [24..27] capsuleA.xyz + capsuleRadius
//   [28..31] capsuleB.xyz + groundY
//   [32..35] boxMin.xyz+pad, [36..39] boxMax.xyz+pad (read by distToSolid's boundaryDensity term)
const BOX_BASE_F32 = 32;
const SIM_BYTES = (BOX_BASE_F32 + 8) * 4;

// Grid uniform — 32 bytes. originCell = (origin.xyz, cellSize); dim = (gridDim.xyz, maxPerCell).
const GRID_BYTES = 32;

const COMMON_WGSL = /* wgsl */ `
const PI = 3.14159265359;

struct Sim {
    dt: f32,
    gravity: f32,
    restDensity: f32,
    h: f32,
    h2: f32,
    poly6: f32,
    spikyGrad: f32,
    eps: f32,
    scorrK: f32,
    scorrInvWdq: f32,
    scorrN: f32,
    viscosity: f32,
    count: u32,
    containerMode: u32,
    boundaryDensity: f32,
    live: u32,
    boundsMin: vec4<f32>,
    boundsMax: vec4<f32>,
    capsuleA: vec4<f32>,
    capsuleB: vec4<f32>,
    boxMin: vec4<f32>,
    boxMax: vec4<f32>,
};

struct Grid {
    originCell: vec4<f32>,
    dim: vec4<u32>,
};

fn poly6(r2: f32, sim: Sim) -> f32 {
    if (r2 >= sim.h2) { return 0.0; }
    let t = sim.h2 - r2;
    return sim.poly6 * t * t * t;
}

// ∇W_spiky(r) with r = p_i - p_j (points along r). sim.spikyGrad already carries the sign.
fn spikyGradient(r: vec3<f32>, rlen: f32, sim: Sim) -> vec3<f32> {
    let c = sim.spikyGrad * (sim.h - rlen) * (sim.h - rlen) / rlen;
    return c * r;
}

fn cellCoordOf(p: vec3<f32>, grid: Grid) -> vec3<i32> {
    let rel = (p - grid.originCell.xyz) / grid.originCell.w;
    return clamp(vec3<i32>(floor(rel)), vec3<i32>(0), vec3<i32>(grid.dim.xyz) - vec3<i32>(1));
}
fn cellLinear(c: vec3<i32>, grid: Grid) -> u32 {
    return u32((c.z * i32(grid.dim.y) + c.y) * i32(grid.dim.x) + c.x);
}

// Distance from p to the nearest confining solid boundary (capsule wall and/or
// ground floor). Positive inside the fluid region. Used to add the missing SPH
// density near walls so the fluid doesn't climb them.
fn distToSolid(p: vec3<f32>, sim: Sim) -> f32 {
    var d = p.y - sim.capsuleB.w; // ground floor
    if (sim.containerMode == 1u) {
        let a = sim.capsuleA.xyz;
        let ba = sim.capsuleB.xyz - a;
        let hh = clamp(dot(p - a, ba) / dot(ba, ba), 0.0, 1.0);
        let dWall = sim.capsuleA.w - length(p - (a + ba * hh));
        d = min(d, dWall);
    } else if (sim.containerMode == 2u) {
        let dlo = p - sim.boxMin.xyz;
        let dhi = sim.boxMax.xyz - p;
        d = min(d, min(min(dlo.x, dlo.y), min(dlo.z, min(dhi.x, min(dhi.y, dhi.z)))));
    }
    return d;
}
`;

// Gravity, then advect to the predicted position. Force-free: any external force
// (setForceField) runs as its OWN dedicated compute pass (fluid-force) at the start
// of step(), so this shader is a single static pipeline that pays nothing for a
// feature that is not in use.
function buildPredictWgsl(): string {
    return /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> predicted: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> sim: Sim;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.count) { return; }
    if (i >= sim.live) { return; } // warm-up: dormant particles are frozen off-screen
    var v = vel[i].xyz;
    v.y -= sim.gravity * sim.dt;
    let p = pos[i].xyz + v * sim.dt;
    predicted[i] = vec4<f32>(p, 1.0);
}`;
}

// Dedicated external-force compute pass, built from an injected ForceFieldSpec.
// Mirrors the emit pass: dispatched at the start of step() ONLY while a force is
// active, so the main integration shaders stay force-free. Reads each particle's
// position + velocity, adds the app's velocity delta, and writes velocity back
// (preserving .w); predict then advects with the force-updated velocity.
function buildForceWgsl(force: ForceFieldSpec): string {
    return /* wgsl */ `
${COMMON_WGSL}
${force.struct}
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> sim: Sim;
@group(0) @binding(3) var<uniform> forceFieldParams: ForceFieldParams;
${force.wgsl}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.count) { return; }
    if (i >= sim.live) { return; } // warm-up: dormant particles feel no external force
    var v = vel[i].xyz;
    v += externalForce(pos[i].xyz, v, sim.dt);
    vel[i] = vec4<f32>(v, vel[i].w);
}`;
}

// Clear both counting-sort accumulators to 0: cellCount (histogram population) and
// cellCursor (scatter write cursor). cellStart is fully overwritten by the scan.
const CLEAR_GRID_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cellCursor: array<u32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let c = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (c >= arrayLength(&cellCount)) { return; }
    atomicStore(&cellCount[c], 0u);
    cellCursor[c] = 0u;
}`;

// ── Counting-sort grid (Hoetzlein 2014) ──────────────────────────────
// Sort the live particles by cell so each cell's members form a contiguous run in
// sortedIdx[cellStart[cell] .. +cellCount[cell]). S1 histogram → S2a/b/c exclusive
// prefix sum (multi-level, robust for arbitrary numCells) → S3 scatter. Cell-sorted
// payload mirrors (sortedPos/sortedLambda/sortedVel) are then gathered so the hot
// neighbour loops read contiguous memory. The bucket is cellLinear(cellCoordOf(x*)).

// S1 — histogram: each live particle bumps its cell's count.
const HISTOGRAM_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> predicted: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> grid: Grid;
@group(0) @binding(3) var<uniform> sim: Sim;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= sim.count) { return; }
    if (i >= sim.live) { return; } // warm-up: dormant particles stay out of the grid
    atomicAdd(&cellCount[cellLinear(cellCoordOf(predicted[i].xyz, grid), grid)], 1u);
}`;

// S2a — per-chunk exclusive scan of cellCount into cellStart, plus each chunk's total
// into partialSums. Hillis-Steele inclusive scan in shared memory, converted to exclusive.
const SCAN_LOCAL_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> cellCount: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(2) var<storage, read_write> partialSums: array<u32>;
var<workgroup> s: array<u32, ${SCAN_WG}>;
@compute @workgroup_size(${SCAN_WG})
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let n = arrayLength(&cellCount);
    let chunk = wid.x + wid.y * ng.x;
    let idx = chunk * ${SCAN_WG}u + lid;
    var v = 0u;
    if (idx < n) { v = cellCount[idx]; }
    s[lid] = v;
    workgroupBarrier();
    var offset = 1u;
    loop {
        if (offset >= ${SCAN_WG}u) { break; }
        var t = 0u;
        if (lid >= offset) { t = s[lid - offset]; }
        workgroupBarrier();
        if (lid >= offset) { s[lid] = s[lid] + t; }
        workgroupBarrier();
        offset = offset * 2u;
    }
    if (idx < n) { cellStart[idx] = s[lid] - v; } // inclusive - self = exclusive
    if (lid == ${SCAN_WG}u - 1u) { partialSums[chunk] = s[${SCAN_WG}u - 1u]; } // chunk total
}`;

// S2b — exclusive scan of the per-chunk totals (partialSums), in place, by a SINGLE
// workgroup that chains over the array in SCAN_WG-wide strides carrying a running offset.
const SCAN_PARTIALS_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> partialSums: array<u32>;
var<workgroup> s: array<u32, ${SCAN_WG}>;
var<workgroup> carry: u32;
@compute @workgroup_size(${SCAN_WG})
fn main(@builtin(local_invocation_index) lid: u32) {
    let n = arrayLength(&partialSums);
    if (lid == 0u) { carry = 0u; }
    workgroupBarrier();
    var base = 0u;
    loop {
        if (base >= n) { break; }
        let idx = base + lid;
        var v = 0u;
        if (idx < n) { v = partialSums[idx]; }
        s[lid] = v;
        workgroupBarrier();
        var offset = 1u;
        loop {
            if (offset >= ${SCAN_WG}u) { break; }
            var t = 0u;
            if (lid >= offset) { t = s[lid - offset]; }
            workgroupBarrier();
            if (lid >= offset) { s[lid] = s[lid] + t; }
            workgroupBarrier();
            offset = offset * 2u;
        }
        if (idx < n) { partialSums[idx] = carry + (s[lid] - v); }
        workgroupBarrier();
        if (lid == 0u) { carry = carry + s[${SCAN_WG}u - 1u]; }
        workgroupBarrier();
        base = base + ${SCAN_WG}u;
    }
}`;

// S2c — add each chunk's scanned offset back into cellStart -> global exclusive prefix sum.
const SCAN_ADD_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(1) var<storage, read> partialSums: array<u32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&cellStart)) { return; }
    cellStart[i] = cellStart[i] + partialSums[i / ${SCAN_WG}u];
}`;

// S3 — scatter each live particle index into its cell's contiguous run.
const SCATTER_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> predicted: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cellStart: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellCursor: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sortedIdx: array<u32>;
@group(0) @binding(4) var<uniform> grid: Grid;
@group(0) @binding(5) var<uniform> sim: Sim;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= sim.count) { return; }
    if (i >= sim.live) { return; }
    let cell = cellLinear(cellCoordOf(predicted[i].xyz, grid), grid);
    let slot = cellStart[cell] + atomicAdd(&cellCursor[cell], 1u);
    sortedIdx[slot] = i;
}`;

// Reorder the live working set into cell-sorted order ONCE per frame (Fluids v5.0
// countingSortFull): slot k reads its original index o = sortedIdx[k] and copies the
// predicted position / current position / velocity into the sorted working buffers.
// sortedPos0 keeps the pre-predict (frame-start) position for finalize's velocity
// v = (x* - x0)/dt. Every solver + foam pass then runs over slot k so consecutive
// threads are spatially adjacent particles that share neighbour cells (cross-thread
// cache reuse); scatter-back writes the results to original order at the very end.
const REORDER_IN_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(1) var<storage, read> predicted: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> vel: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> sortedPos: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> sortedPos0: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> sortedVel: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> sim: Sim;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let k = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (k >= sim.live) { return; }
    let o = sortedIdx[k];
    sortedPos[k] = predicted[o];
    sortedPos0[k] = pos[o];
    sortedVel[k] = vel[o];
}`;

// Scatter the solved sorted working set back to original order ONCE per frame: slot k
// writes its original particle o = sortedIdx[k]. Only live slots (k < sim.live) are
// written, so dormant (never-inserted) particles keep their parked pos/vel. debug is the
// speed used for colouring (was written by the old original-order viscosity pass).
const SCATTER_BACK_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(1) var<storage, read> sortedPos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> sortedVel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> debug: array<f32>;
@group(0) @binding(6) var<uniform> sim: Sim;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let k = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (k >= sim.live) { return; }
    let o = sortedIdx[k];
    pos[o] = vec4<f32>(sortedPos[k].xyz, 1.0);
    vel[o] = vec4<f32>(sortedVel[k].xyz, 0.0);
    debug[o] = length(sortedVel[k].xyz);
}`;

// λ_i = −C_i / (|∇_i C|² + Σ_j |∇_j C|² + ε)
// Sorted-order dispatch (Fluids v5.0): thread k is sorted slot k; both the centre and
// the neighbours read the contiguous sorted working buffer, so consecutive threads reuse
// the same neighbour cells in cache.
const LAMBDA_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> sortedPos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<uniform> grid: Grid;
@group(0) @binding(4) var<uniform> sim: Sim;
@group(0) @binding(5) var<storage, read_write> sortedLambda: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= sim.live) { return; }
    let pi = sortedPos[k].xyz;
    let base = cellCoordOf(pi, grid);
    let invRho = 1.0 / sim.restDensity;

    var rho = 0.0;
    var gradI = vec3<f32>(0.0);
    var sumGrad2 = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cc = base + vec3<i32>(dx, dy, dz);
        if (any(cc < vec3<i32>(0)) || any(cc >= vec3<i32>(grid.dim.xyz))) { continue; }
        let cell = cellLinear(cc, grid);
        let start = cellStart[cell];
        let cnt = atomicLoad(&cellCount[cell]);
        for (var s = 0u; s < cnt; s = s + 1u) {
            let r = pi - sortedPos[start + s].xyz;
            let r2 = dot(r, r);
            if (r2 < sim.h2) {
                rho += poly6(r2, sim);
                if (r2 > 1e-9) {
                    let g = spikyGradient(r, sqrt(r2), sim) * invRho;
                    gradI += g;
                    sumGrad2 += dot(g, g);
                }
            }
        }
    }}}

    // Boundary density support: near a solid wall the kernel sphere is partly
    // outside the fluid, so the neighbour sum under-counts ρ. Add the density the
    // missing (wall-side) fluid would contribute — without it, under-pressured
    // wall particles get shoved up the wall and pile into a raised rim.
    if (sim.boundaryDensity > 0.0) {
        let dWall = distToSolid(pi, sim);
        if (dWall < sim.h) {
            let t = clamp(dWall / sim.h, 0.0, 1.0);
            rho += sim.restDensity * sim.boundaryDensity * (1.0 - t) * (1.0 - t);
        }
    }

    // Full PBF constraint (negative pressure allowed): a free surface is
    // under-dense, and the resulting cohesion — balanced by the s_corr term in
    // the Δp pass — is what holds the surface flat without it collapsing. (A
    // compression-only clamp keeps the bulk stable but leaves wall particles
    // pressure-less, so the hydrostatic gradient squeezes them up the walls.)
    let ci = rho * invRho - 1.0;
    let denom = dot(gradI, gradI) + sumGrad2 + sim.eps;
    sortedLambda[k] = -ci / denom;
}`;

// Δp_i = (1/ρ0) Σ_j (λ_i + λ_j + s_corr) ∇W_spiky  (sorted-order dispatch over slot k)
const DELTA_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> sortedPos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> sortedLambda: array<f32>;
@group(0) @binding(4) var<uniform> grid: Grid;
@group(0) @binding(5) var<uniform> sim: Sim;
@group(0) @binding(6) var<storage, read_write> sortedDelta: array<vec4<f32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= sim.live) { return; }
    let pi = sortedPos[k].xyz;
    let li = sortedLambda[k];
    let base = cellCoordOf(pi, grid);

    var dp = vec3<f32>(0.0);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cc = base + vec3<i32>(dx, dy, dz);
        if (any(cc < vec3<i32>(0)) || any(cc >= vec3<i32>(grid.dim.xyz))) { continue; }
        let cell = cellLinear(cc, grid);
        let start = cellStart[cell];
        let cnt = atomicLoad(&cellCount[cell]);
        // The self entry has r2 == 0, excluded by the r2 > 1e-9 guard below (this is
        // why no explicit j == i skip is needed once neighbours come from the mirror).
        for (var s = 0u; s < cnt; s = s + 1u) {
            let k2 = start + s;
            let r = pi - sortedPos[k2].xyz;
            let r2 = dot(r, r);
            if (r2 < sim.h2 && r2 > 1e-9) {
                let w = poly6(r2, sim);
                let scorr = -sim.scorrK * pow(w * sim.scorrInvWdq, sim.scorrN);
                dp += (li + sortedLambda[k2] + scorr) * spikyGradient(r, sqrt(r2), sim);
            }
        }
    }}}

    sortedDelta[k] = vec4<f32>(dp / sim.restDensity, 0.0);
}`;

// Confinement using the per-demo sceneSdf (always injected by the demo before the
// first step): push penetrating particles back along the SDF gradient (position
// only — PBF derives velocity in finalize, so a moving boundary stirs for free).
// The unified SDF confines the interior cavity AND the exterior/ground alike, so
// drained fluid is caught by the SAME field — no per-particle "escaped" state.
const APPLY_CONFINE_SDF = /* wgsl */ `
    let d = sceneSdf(p, 0.0);
    if (d < 0.0) { p += sceneNormal(p, 0.0) * (-d); }
    // Global ground safety floor (the unified SDF also floors at groundY; cheap backstop).
    p.y = max(p.y, sim.capsuleB.w);`;

// The per-demo sceneSdf is always injected (setSceneSdf) before the first step,
// so the apply pass always uses the SDF confinement path.
function buildApplyWgsl(scene: SceneSdfSpec): string {
    // Baked SDF grid (optional): binding 4 is free here (0=sortedPos, 1=sortedDelta,
    // 2=sim, 3=sceneSdfParams). Injected BEFORE scene.sdf so sceneSdf can call
    // sampleSdfGrid.
    const gridInject = scene.sdfGrid ? `\n@group(0) @binding(4) var<storage, read> sceneSdfGrid: array<f32>;\n${SCENE_SDF_GRID_WGSL}` : "";
    const decls = `${scene.struct}\n@group(0) @binding(3) var<uniform> sceneSdfParams: SceneSdfParams;${gridInject}\n${scene.sdf}\n${SCENE_NORMAL_WGSL}`;
    const confine = APPLY_CONFINE_SDF;
    return /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> sortedPos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> sortedDelta: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> sim: Sim;
${decls}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= sim.live) { return; } // warm-up: dormant slots are not solved
    var p = sortedPos[k].xyz + sortedDelta[k].xyz;
${confine}
    // Safety: never leave the neighbour-grid domain.
    p = clamp(p, sim.boundsMin.xyz, sim.boundsMax.xyz);
    sortedPos[k] = vec4<f32>(p, 1.0);
}`;
}

// PBF velocity update: v = (x* - x0)/dt, where x0 is the pre-predict (frame-start)
// position captured by reorder-in into sortedPos0. Sorted-order dispatch over slot k;
// the original-order pos/vel are written later by scatter-back.
const FINALIZE_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> sortedPos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> sortedPos0: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> sortedVel: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> sim: Sim;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= sim.live) { return; } // warm-up: dormant slots are not solved
    let xnew = sortedPos[k].xyz;
    let v = (xnew - sortedPos0[k].xyz) / sim.dt;
    sortedVel[k] = vec4<f32>(v, 0.0);
}`;

// XSPH viscosity, sorted-order dispatch over slot k. Both the centre and the neighbours
// read the contiguous sorted working buffers (sortedPos is the finalised position;
// sortedVel is finalize's velocity). To avoid a storage read-after-write DATA RACE
// (reading neighbour sortedVel while writing our own), the smoothed velocity is written
// to sortedPos0 — DEAD after finalize consumed it as x0 — so sortedVel stays read-only
// this pass and sortedPos0 is write-only. A trivial copy-vel pass then restores
// sortedPos0 -> sortedVel before scatter-back / foam read the final velocity.
const VISCOSITY_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> sortedPos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> sortedVel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> cellStart: array<u32>;
@group(0) @binding(4) var<uniform> grid: Grid;
@group(0) @binding(5) var<uniform> sim: Sim;
@group(0) @binding(6) var<storage, read_write> sortedVelOut: array<vec4<f32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= sim.live) { return; }
    let pi = sortedPos[k].xyz;
    let vi = sortedVel[k].xyz;
    let base = cellCoordOf(pi, grid);

    var dv = vec3<f32>(0.0);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cc = base + vec3<i32>(dx, dy, dz);
        if (any(cc < vec3<i32>(0)) || any(cc >= vec3<i32>(grid.dim.xyz))) { continue; }
        let cell = cellLinear(cc, grid);
        let start = cellStart[cell];
        let cnt = atomicLoad(&cellCount[cell]);
        for (var s = 0u; s < cnt; s = s + 1u) {
            let k2 = start + s;
            let r = pi - sortedPos[k2].xyz;
            let r2 = dot(r, r);
            if (r2 < sim.h2) {
                dv += (sortedVel[k2].xyz - vi) * poly6(r2, sim);
            }
        }
    }}}

    let v = vi + (sim.viscosity / sim.restDensity) * dv;
    sortedVelOut[k] = vec4<f32>(v, 0.0);
}`;

// Trivial copy of the just-smoothed velocity (viscosity wrote it into sortedPos0 to dodge
// the read/write race) back into sortedVel, so scatter-back and the foam passes read the
// FINAL post-viscosity velocity from sortedVel exactly as before. Only live slots copy;
// no same-buffer read+write (src and dst are distinct buffers, disjoint index per thread).
const COPY_VEC4_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> sim: Sim;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let k = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (k >= sim.live) { return; }
    dst[k] = src[k];
}`;

// ── Generic particle emitter / recycling ─────────────────────────────
// Fixed particle pool: "emitting" recycles particles rather than adding them.
// A compute pass run first each step relaunches particles that sit inside the
// pump-intake box, probabilistically (rand < rate·dt, to throttle so jets are
// continuous streams), at a random emitter nozzle with its jet velocity.
// (EmitterConfig / MAX_EMITTERS / EMITTERS_FLOATS / packEmitters live in sim-common.)
const EMIT_WGSL = /* wgsl */ `
${EMITTER_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> em: Emitters;

fn hashU(x0: u32) -> u32 { var h = x0; h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u; return h; }
fn rnd(x: u32) -> f32 { return f32(hashU(x)) / 4294967296.0; }
${EMITTER_SPAWN_WGSL}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u32(em.head2.x)) { return; }
    let ec = u32(em.head.x);
    if (ec == 0u) { return; }
    let p = pos[i].xyz;
    let seed = u32(em.head.z) * 2654435761u + i;
    let fixedN = u32(em.head2.z);
    // Dedicated fixed-size stream (fixedN > 0): indices [0, fixedN) form a self-contained TIGHT
    // LOOP through the LAST emitter. The instant such a particle sinks below the drain height
    // (head2.w) it relaunches DETERMINISTICALLY — no rate throttle, no floor-intake test — so
    // ~fixedN particles are always in flight at a cadence set only by gravity + geometry, fully
    // INDEPENDENT of the total particle count or how deep the main pool is. These particles never
    // touch the shared pump-intake, so the main pool can't dilute or starve the jet.
    if (fixedN > 0u && i < fixedN) {
        if (p.y < em.head2.w) {
            let e = em.list[ec - 1u];
            let sj = (vec3<f32>(rnd(seed * 11u), rnd(seed * 13u), rnd(seed * 17u)) - 0.5) * (e.d.w * em.head.w);
            pos[i] = vec4<f32>(spawnPoint(e, seed), 1.0);
            vel[i] = vec4<f32>(e.d.xyz * e.d.w + sj, 0.0);
        }
        return;
    }
    // Everything else: the original probabilistic pump-intake recycle. When fixedN > 0 the last
    // emitter is reserved for the fixed stream, so the general pool recycles through the first
    // ec-1 emitters; when fixedN == 0 all emitters are shared (byte-identical original path).
    if (all(p >= em.intakeMin.xyz) && all(p <= em.intakeMax.xyz)) {
        if (rnd(seed) < em.head.y * em.head2.y) {
            var ei = hashU(seed) % ec;
            if (fixedN > 0u) { ei = hashU(seed) % max(ec - 1u, 1u); }
            let e = em.list[ei];
            let sj = (vec3<f32>(rnd(seed * 11u), rnd(seed * 13u), rnd(seed * 17u)) - 0.5) * (e.d.w * em.head.w);
            pos[i] = vec4<f32>(spawnPoint(e, seed), 1.0);
            vel[i] = vec4<f32>(e.d.xyz * e.d.w + sj, 0.0);
        }
    }
}`;

// ── Foam: diffuse particles (spray / foam / bubbles) ─────────────────
// Compute passes appended to the END of step() when foam is enabled, reusing the
// finalized fluid positions/velocities and the neighbour grid built earlier in the
// same step. Generation follows Ihmsen 2012: foam-normals → foam-emit (trapped-air +
// wave-crest + kinetic potentials from the neighbour grid), selective — fires at
// impacts, convergence and wave crests. The foam-update pass then classifies +
// advects the pool. Like the solver kernels, the neighbour loops here read the cell-
// sorted MIRRORS (sortedPos / sortedVel / sortedNormals) contiguously (coalesced)
// rather than sortedIdx-indirected original arrays: sortedVel is re-gathered from the
// post-viscosity velocities and sortedNormals is gathered from the per-particle normals
// written by foam-normals. The shared foam WGSL (the Foam UBO, the diffuse slot struct,
// the radial hat kernel W, the [0,1] clamp map Φ, and a hash PRNG) + the FoamParams UBO
// layout live in sim-common.ts (`FOAM_COMMON_WGSL` / `FOAM_BYTES`) so the MLS-MPM
// backend injects the identical struct. h / dt / gravity / bounds are read from the
// PBF Sim UBO here, so only the foam-specific knobs live in FoamParams.

// Pass 1 — SPH surface normal per fluid particle (for wave-crest detection).
// n_i = normalize(Σ_j ∇W_spiky(x_i - x_j)); that colour-field gradient points toward
// the denser interior, so the OUTWARD normal is its negation. .w stores the neighbour
// count as a surface indicator (surface particles have fewer neighbours). Sorted-order
// dispatch over slot k: centre and neighbours both read the sortedPos mirror (self
// excluded by the r2>1e-9 guard) and the normal is written directly into sortedNormals,
// so foam-emit/update read it contiguously without a gather.
const FOAM_NORMALS_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> sortedPos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<uniform> grid: Grid;
@group(0) @binding(4) var<uniform> sim: Sim;
@group(0) @binding(5) var<storage, read_write> sortedNormals: array<vec4<f32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= sim.live) { return; }
    let pi = sortedPos[k].xyz;
    let base = cellCoordOf(pi, grid);
    var grad = vec3<f32>(0.0);
    var cnt = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cc = base + vec3<i32>(dx, dy, dz);
        if (any(cc < vec3<i32>(0)) || any(cc >= vec3<i32>(grid.dim.xyz))) { continue; }
        let cell = cellLinear(cc, grid);
        let start = cellStart[cell];
        let ncell = atomicLoad(&cellCount[cell]);
        for (var s = 0u; s < ncell; s = s + 1u) {
            let r = pi - sortedPos[start + s].xyz;
            let r2 = dot(r, r);
            if (r2 < sim.h2 && r2 > 1e-9) {
                grad += spikyGradient(r, sqrt(r2), sim);
                cnt += 1.0;
            }
        }
    }}}
    let glen = length(grad);
    var nrm = vec3<f32>(0.0, 1.0, 0.0);
    if (glen > 1e-6) { nrm = -grad / glen; }
    sortedNormals[k] = vec4<f32>(nrm, cnt);
}`;

// Pass 2 — generation. Per fluid particle compute the trapped-air, wave-crest and
// kinetic potentials, derive a capped count n_d, and append n_d sampled diffuse
// particles into the ring (atomicAdd on the write-head, modulo capacity). Dispatched
// over the cell-SORTED order (slot k), so self position/velocity/normal come from the
// sortedPos/sortedVel/sortedNormals mirrors and neighbour reads are contiguous; this
// also keeps the pass within the 8-storage-buffer-per-stage device limit. The original
// index i = sortedIdx[k] is recovered only to seed the PRNG identically. Occupied sorted
// slots are [0, sim.live); the self entry is skipped by the existing rlen<1e-6 guard.
const FOAM_EMIT_WGSL = /* wgsl */ `
${COMMON_WGSL}
${FOAM_COMMON_WGSL}
@group(0) @binding(0) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> sortedPos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> sortedVel: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> sortedNormals: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> grid: Grid;
@group(0) @binding(7) var<uniform> sim: Sim;
@group(0) @binding(8) var<uniform> foam: Foam;
@group(0) @binding(9) var<storage, read_write> diffuse: array<Diffuse>;
@group(0) @binding(10) var<storage, read_write> head: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= sim.live) { return; }
    let i = sortedIdx[slot];
    let pi = sortedPos[slot].xyz;
    let vi = sortedVel[slot].xyz;
    let speed = length(vi);
    if (speed < 1e-4) { return; }
    let vhat = vi / speed;
    let ni = sortedNormals[slot].xyz;
    let base = cellCoordOf(pi, grid);

    var vdiff = 0.0;
    var kappa = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cc = base + vec3<i32>(dx, dy, dz);
        if (any(cc < vec3<i32>(0)) || any(cc >= vec3<i32>(grid.dim.xyz))) { continue; }
        let cell = cellLinear(cc, grid);
        let start = cellStart[cell];
        let ncell = atomicLoad(&cellCount[cell]);
        for (var s = 0u; s < ncell; s = s + 1u) {
            let k2 = start + s;
            let xij = pi - sortedPos[k2].xyz;
            let rlen = length(xij);
            if (rlen >= sim.h || rlen < 1e-6) { continue; }
            let w = wHat(rlen, sim.h);
            let vij = vi - sortedVel[k2].xyz;
            let vl = length(vij);
            if (vl > 1e-6) {
                vdiff += vl * (1.0 - dot(vij / vl, xij / rlen)) * w;
            }
            // wave crest: convex neighbours only (x̂_ji · n̂_i < 0).
            if (dot(-xij / rlen, ni) < 0.0) {
                kappa += (1.0 - dot(ni, sortedNormals[k2].xyz)) * w;
            }
        }
    }}}

    let dvn = select(0.0, 1.0, dot(vhat, ni) >= 0.6);
    let ita = phi(vdiff, foam.tauTaMin, foam.tauTaMax);
    let iwc = phi(kappa * dvn, foam.tauWcMin, foam.tauWcMax);
    let ek = 0.5 * speed * speed;
    let ik = phi(ek, foam.tauKMin, foam.tauKMax);
    let ndf = ik * (foam.kTa * ita + foam.kWc * iwc) * sim.dt;
    // Stochastic rounding: nd must have EXPECTED value ndf (Ihmsen 2012). Rounding to nearest
    // instead turned the generation rates into a step function — dt is clamped to 1/60, so every
    // particle shares it, and ndf crossed 0.5 for the whole population at the same rate value
    // (kTa = 30 at 60fps). The control jumped from "no foam at all" to "one per particle per
    // frame" in a single slider step. Carrying the fraction as a spawn PROBABILITY spreads that
    // crossing across the population, so the rate responds continuously.
    let ndWhole = floor(ndf);
    var nd = i32(ndWhole) + select(0, 1, fRnd((i * 2246822519u) ^ (foam.frameSeed * 22695477u) ^ 0x9e3779b9u) < (ndf - ndWhole));
    if (nd <= 0) { return; }
    nd = min(nd, 8);

    let potential = clamp(ita + iwc, 0.0, 1.0);
    let life = mix(foam.tMin, foam.tMax, potential);
    // Orthonormal basis perpendicular to v̂ (cylinder axis).
    var e1 = cross(vhat, vec3<f32>(0.0, 1.0, 0.0));
    if (!(dot(e1, e1) > 1e-6)) { e1 = cross(vhat, vec3<f32>(1.0, 0.0, 0.0)); }
    e1 = normalize(e1);
    let e2 = cross(vhat, e1);
    let cap = arrayLength(&diffuse);
    let dtv = length(sim.dt * vi);
    for (var k = 0; k < nd; k = k + 1) {
        let seed = (i * 2654435761u) ^ (foam.frameSeed * 40503u) ^ (u32(k) * 2246822519u);
        let xr = fRnd(seed);
        let xt = fRnd(seed * 3u + 1u);
        let xh = fRnd(seed * 7u + 5u);
        let rr = foam.rv * sqrt(xr);
        let th = 6.28318530718 * xt;
        let off = e1 * (rr * cos(th)) + e2 * (rr * sin(th));
        let xd = pi + off + vhat * (xh * dtv);
        let vd = off + vi;
        let idx = atomicAdd(&head[0], 1u) % cap;
        diffuse[idx].p = vec4<f32>(xd, life);
        diffuse[idx].v = vec4<f32>(vd, 0.0);
    }
}`;

// Pass 3 — classify + advect + dissolve, over the whole diffuse pool. Fluid-neighbour
// count n classifies each live particle (n<6 spray, n>20 bubble, else foam); each class
// advects differently; foam decays its lifetime; particles die on lifetime<=0 or on
// leaving the domain. kind is written into v.w for the renderer. The fluid-neighbour
// position/velocity come from the sortedPos/sortedVel mirrors (a diffuse particle has
// no self entry in the fluid set, so there is no self case to skip).
const FOAM_UPDATE_WGSL = /* wgsl */ `
${COMMON_WGSL}
${FOAM_COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> cellStart: array<u32>;
@group(0) @binding(2) var<storage, read> sortedPos: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> sortedVel: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> grid: Grid;
@group(0) @binding(5) var<uniform> sim: Sim;
@group(0) @binding(6) var<uniform> foam: Foam;
@group(0) @binding(7) var<storage, read_write> diffuse: array<Diffuse>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Pool slots can exceed one dispatch dimension, in which case dispatch() spills into y
    // with an x extent of exactly MAX_WORKGROUPS groups — fold that back into a flat index.
    // gid.y is 0 whenever the dispatch fits in x, so this is a no-op for small pools.
    let i = gid.x + gid.y * ${MAX_WORKGROUPS * WORKGROUP_SIZE}u;
    if (i >= arrayLength(&diffuse)) { return; }
    let p0 = diffuse[i].p;
    if (p0.w <= 0.0) { return; }
    let pp = p0.xyz;
    if (any(pp < sim.boundsMin.xyz) || any(pp > sim.boundsMax.xyz)) {
        diffuse[i].p = vec4<f32>(pp, 0.0);
        return;
    }
    var v = diffuse[i].v.xyz;
    let base = cellCoordOf(pp, grid);
    var nn = 0u;
    var vsum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cc = base + vec3<i32>(dx, dy, dz);
        if (any(cc < vec3<i32>(0)) || any(cc >= vec3<i32>(grid.dim.xyz))) { continue; }
        let cell = cellLinear(cc, grid);
        let start = cellStart[cell];
        let ncell = atomicLoad(&cellCount[cell]);
        for (var s = 0u; s < ncell; s = s + 1u) {
            let k2 = start + s;
            let rlen = length(pp - sortedPos[k2].xyz);
            if (rlen < sim.h) {
                nn = nn + 1u;
                let w = wHat(rlen, sim.h);
                vsum += sortedVel[k2].xyz * w;
                wsum += w;
            }
        }
    }}}
    var vf = vec3<f32>(0.0);
    if (wsum > 1e-6) { vf = vsum / wsum; }

    let g = sim.gravity;
    let dt = sim.dt;
    var kind = 1u;
    if (nn < 6u) { kind = 0u; } else if (nn > 20u) { kind = 2u; }

    var np = pp;
    var life = p0.w;
    if (kind == 0u) {
        // Spray: ballistic (Euler-Cromer).
        v.y -= g * dt;
        np = pp + dt * v;
    } else if (kind == 2u) {
        // Bubble: buoyancy up + drag toward the local flow.
        v.y += dt * foam.kb * g;
        v += foam.kd * (vf - v);
        np = pp + dt * v;
    } else {
        // Foam: rides the surface at the SPH-averaged fluid velocity; lifetime decays.
        v = vf;
        np = pp + dt * vf;
        life = p0.w - dt;
    }
    if (life <= 0.0) {
        diffuse[i].p = vec4<f32>(np, 0.0);
        return;
    }
    diffuse[i].p = vec4<f32>(np, life);
    diffuse[i].v = vec4<f32>(v, f32(kind));
}`;

export function createPbfSim(engine: EngineContext, options: PbfOptions = {}): FluidSim {
    const device = engine._device;
    const count = options.count ?? 30000;
    const particleRadius = options.particleRadius ?? 0.08;
    const spawnMin: [number, number, number] = options.spawnMin ?? [-2, 6, -2];
    const spawnMax: [number, number, number] = options.spawnMax ?? [2, 12, 2];
    let spawnAccept: ((x: number, y: number, z: number) => boolean) | null = null;
    // Start-of-sim WARM-UP ramp (mirrors the MLS-MPM backend): a dense seed released
    // all at once spikes the density solver into spray on an open shelf. Instead only
    // `liveCount` particles are live each frame (ramping over `warmupFrames`); dormant
    // ones are parked off-screen and skipped by the solver passes, then teleported to
    // their stored spawn position as they activate. 0 (default) = release immediately.
    let warmupFrames = Math.max(0, Math.floor(options.warmupFrames ?? 0));
    let warmupStep = warmupFrames > 0 ? Math.max(1, Math.ceil(count / warmupFrames)) : count;
    let liveCount = count;
    const seedPositions = new Float32Array(count * 4);
    const gravity = options.gravity ?? 9.8;
    const h = options.smoothingRadius ?? 0.4;
    const boundsMin = options.boundsMin ?? [-4, 0, -4];
    const boundsMax = options.boundsMax ?? [4, 20, 4];
    const capsuleA = options.capsuleA ?? null;
    const capsuleB = options.capsuleB ?? null;
    const capsuleRadius = options.capsuleRadius ?? 0;
    const groundY = options.groundY ?? boundsMin[1];
    const restDensityScale = options.restDensityScale ?? 1.0;
    const iterations = options.iterations ?? 3;
    let iterationsMut = iterations;
    const relaxation = options.relaxation ?? 50;
    const viscosity = options.viscosity ?? 0.08;
    const scorrK = options.scorr ?? 0.02;
    const boundaryDensity = options.boundaryDensity ?? 0;
    const maxPerCell = options.maxPerCell ?? 64;
    // Optional explicit per-particle seed (flat world-space xyz). When set, seed()
    // reads position i from here instead of the random spawn draw (see the interface).
    const initialPositions = options.initialPositions ?? null;

    // Rest density: spawn number density (particles / spawn volume) × scale.
    const spawnVol = Math.max(1e-6, (spawnMax[0] - spawnMin[0]) * (spawnMax[1] - spawnMin[1]) * (spawnMax[2] - spawnMin[2]));
    const restDensity = options.restDensity ?? (count / spawnVol) * restDensityScale;

    // Grid derived from the bounds so cells tightly cover the active region.
    const gridDim: [number, number, number] = [
        Math.max(1, Math.ceil((boundsMax[0] - boundsMin[0]) / h)),
        Math.max(1, Math.ceil((boundsMax[1] - boundsMin[1]) / h)),
        Math.max(1, Math.ceil((boundsMax[2] - boundsMin[2]) / h)),
    ];
    const numCells = gridDim[0] * gridDim[1] * gridDim[2];
    // Per-chunk totals for the multi-level counting-sort prefix scan over numCells.
    const scanChunks = Math.ceil(numCells / SCAN_WG);

    // Kernel coefficients (mass = 1).
    const h2 = h * h;
    const poly6Coef = 315 / (64 * Math.PI * Math.pow(h, 9));
    const spikyGradCoef = -45 / (Math.PI * Math.pow(h, 6));
    const dq = 0.2 * h;
    const wDq = poly6Coef * Math.pow(h2 - dq * dq, 3);
    const scorrInvWdq = 1 / wDq;
    const scorrN = 4;

    // ── Buffers ──────────────────────────────────────────────────────
    const positionBuffer = device.createBuffer({ label: "fluid-positions", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const velocityBuffer = device.createBuffer({ label: "fluid-velocities", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const predictedBuffer = device.createBuffer({ label: "fluid-predicted", size: count * 16, usage: GPUBufferUsage.STORAGE });
    const debugBuffer = device.createBuffer({ label: "fluid-debug", size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // Counting-sort grid buffers (Hoetzlein 2014) + sorted-order working set (Fluids v5.0
    // countingSortFull). All sized from numCells/count so a domainScale rebuild
    // (createPbfSim re-run) resizes them alongside every other buffer. cellCount
    // (histogram population), cellStart (exclusive prefix sum), cellCursor (scatter write
    // cursor), partialSums (per-chunk scan carries), sortedIdx (original indices in
    // cell-sorted order). The solver runs over sorted slots on the sorted working buffers:
    // sortedPos (mutable predicted position), sortedPos0 (frame-start position, for the
    // finalize velocity), sortedVel (velocity), sortedLambda (λ), sortedDelta (Δp).
    const cellCountBuffer = device.createBuffer({ label: "fluid-cell-count", size: numCells * 4, usage: GPUBufferUsage.STORAGE });
    const cellStartBuffer = device.createBuffer({ label: "fluid-cell-start", size: numCells * 4, usage: GPUBufferUsage.STORAGE });
    const cellCursorBuffer = device.createBuffer({ label: "fluid-cell-cursor", size: numCells * 4, usage: GPUBufferUsage.STORAGE });
    const partialSumsBuffer = device.createBuffer({ label: "fluid-partial-sums", size: scanChunks * 4, usage: GPUBufferUsage.STORAGE });
    const sortedIdxBuffer = device.createBuffer({ label: "fluid-sorted-idx", size: count * 4, usage: GPUBufferUsage.STORAGE });
    const sortedPosBuffer = device.createBuffer({ label: "fluid-sorted-pos", size: count * 16, usage: GPUBufferUsage.STORAGE });
    const sortedPos0Buffer = device.createBuffer({ label: "fluid-sorted-pos0", size: count * 16, usage: GPUBufferUsage.STORAGE });
    const sortedLambdaBuffer = device.createBuffer({ label: "fluid-sorted-lambda", size: count * 4, usage: GPUBufferUsage.STORAGE });
    const sortedDeltaBuffer = device.createBuffer({ label: "fluid-sorted-delta", size: count * 16, usage: GPUBufferUsage.STORAGE });
    const sortedVelBuffer = device.createBuffer({ label: "fluid-sorted-vel", size: count * 16, usage: GPUBufferUsage.STORAGE });
    const simBuffer = device.createBuffer({ label: "fluid-sim", size: SIM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const gridBuffer = device.createBuffer({ label: "fluid-grid", size: GRID_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const emittersBuffer = device.createBuffer({ label: "fluid-emitters", size: EMITTERS_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const emitData = new Float32Array(EMITTERS_FLOATS);
    emitData[4] = count; // head2.x = particle count
    let emitEnabled = false;
    let emitSeed = 0;

    const simData = new ArrayBuffer(SIM_BYTES);
    const simF32 = new Float32Array(simData);
    const simU32 = new Uint32Array(simData);
    simF32[1] = gravity;
    simF32[2] = restDensity;
    simF32[3] = h;
    simF32[4] = h2;
    simF32[5] = poly6Coef;
    simF32[6] = spikyGradCoef;
    simF32[7] = relaxation;
    simF32[8] = scorrK;
    simF32[9] = scorrInvWdq;
    simF32[10] = scorrN;
    simF32[11] = viscosity;
    simU32[12] = count;
    simU32[13] = capsuleA && capsuleB ? 1 : 0;
    simF32[14] = boundaryDensity;
    simU32[15] = count;
    simF32[16] = boundsMin[0];
    simF32[17] = boundsMin[1];
    simF32[18] = boundsMin[2];
    simF32[20] = boundsMax[0];
    simF32[21] = boundsMax[1];
    simF32[22] = boundsMax[2];
    simF32[24] = capsuleA ? capsuleA[0] : 0;
    simF32[25] = capsuleA ? capsuleA[1] : 0;
    simF32[26] = capsuleA ? capsuleA[2] : 0;
    simF32[27] = capsuleRadius;
    simF32[28] = capsuleB ? capsuleB[0] : 0;
    simF32[29] = capsuleB ? capsuleB[1] : 0;
    simF32[30] = capsuleB ? capsuleB[2] : 0;
    simF32[31] = groundY;

    // Grid params are static for the lifetime of the sim.
    {
        const gridData = new ArrayBuffer(GRID_BYTES);
        const gf32 = new Float32Array(gridData);
        const gu32 = new Uint32Array(gridData);
        gf32[0] = boundsMin[0];
        gf32[1] = boundsMin[1];
        gf32[2] = boundsMin[2];
        gf32[3] = h; // cellSize
        gu32[4] = gridDim[0];
        gu32[5] = gridDim[1];
        gu32[6] = gridDim[2];
        gu32[7] = maxPerCell;
        device.queue.writeBuffer(gridBuffer, 0, gridData);
    }

    function seed(): void {
        // Reset the warm-up ramp: start with just the first batch live (or everything
        // when disabled). step() grows liveCount, teleporting dormant particles from
        // off-screen to their stored spawn position as they activate.
        liveCount = warmupFrames > 0 ? Math.min(count, warmupStep) : count;
        const positions = new Float32Array(count * 4);
        // Last position that passed `spawnAccept`, reused when a particle exhausts its retries.
        let lastOkX = 0;
        let lastOkY = 0;
        let lastOkZ = 0;
        let haveLastOk = false;
        for (let i = 0; i < count; i++) {
            const o = i * 4;
            let x: number;
            let y: number;
            let z: number;
            if (initialPositions) {
                // Explicit per-particle seed (e.g. a volume-sampled mesh fill): read the
                // world position straight from the caller's flat xyz array. Velocity stays
                // zeroed (written below), exactly as in the random path.
                x = initialPositions[i * 3]!;
                y = initialPositions[i * 3 + 1]!;
                z = initialPositions[i * 3 + 2]!;
            } else {
                x = spawnMin[0] + Math.random() * (spawnMax[0] - spawnMin[0]);
                y = spawnMin[1] + Math.random() * (spawnMax[1] - spawnMin[1]);
                z = spawnMin[2] + Math.random() * (spawnMax[2] - spawnMin[2]);
                if (spawnAccept) {
                    // Reject-sample so particles fit a non-box container shape: redraw
                    // uniformly in the box until accepted. On exhaustion reuse the last
                    // ACCEPTED point rather than keeping a rejected one, which would place
                    // particles outside the container (see the note in mls-mpm-sim).
                    let ok = spawnAccept(x, y, z);
                    for (let tries = 0; !ok && tries < SPAWN_ACCEPT_TRIES; tries++) {
                        x = spawnMin[0] + Math.random() * (spawnMax[0] - spawnMin[0]);
                        y = spawnMin[1] + Math.random() * (spawnMax[1] - spawnMin[1]);
                        z = spawnMin[2] + Math.random() * (spawnMax[2] - spawnMin[2]);
                        ok = spawnAccept(x, y, z);
                    }
                    if (ok) {
                        lastOkX = x;
                        lastOkY = y;
                        lastOkZ = z;
                        haveLastOk = true;
                    } else if (haveLastOk) {
                        x = lastOkX;
                        y = lastOkY;
                        z = lastOkZ;
                    }
                }
            }
            // Keep the real spawn position for the warm-up teleport-on-activation.
            seedPositions[o] = x;
            seedPositions[o + 1] = y;
            seedPositions[o + 2] = z;
            seedPositions[o + 3] = 1;
            const live = i < liveCount;
            positions[o] = live ? x : 0;
            positions[o + 1] = live ? y : -1.0e5; // park dormant particles off-screen
            positions[o + 2] = live ? z : 0;
            positions[o + 3] = 1;
        }
        device.queue.writeBuffer(positionBuffer, 0, positions);
        device.queue.writeBuffer(velocityBuffer, 0, new Float32Array(count * 4));
        device.queue.writeBuffer(debugBuffer, 0, new Float32Array(count));
    }
    seed();

    // ── Pipelines ────────────────────────────────────────────────────
    function computePipeline(label: string, code: string): GPUComputePipeline {
        return device.createComputePipeline({ label, layout: "auto", compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" } });
    }
    const clearGridPipeline = computePipeline("fluid-clear-grid", CLEAR_GRID_WGSL);
    const histogramPipeline = computePipeline("fluid-histogram", HISTOGRAM_WGSL);
    const scanLocalPipeline = computePipeline("fluid-scan-local", SCAN_LOCAL_WGSL);
    const scanPartialsPipeline = computePipeline("fluid-scan-partials", SCAN_PARTIALS_WGSL);
    const scanAddPipeline = computePipeline("fluid-scan-add", SCAN_ADD_WGSL);
    const scatterPipeline = computePipeline("fluid-scatter", SCATTER_WGSL);
    const reorderInPipeline = computePipeline("fluid-reorder-in", REORDER_IN_WGSL);
    const scatterBackPipeline = computePipeline("fluid-scatter-back", SCATTER_BACK_WGSL);
    const copyVec4Pipeline = computePipeline("fluid-copy-vec4", COPY_VEC4_WGSL);
    const lambdaPipeline = computePipeline("fluid-lambda", LAMBDA_WGSL);
    const deltaPipeline = computePipeline("fluid-delta", DELTA_WGSL);
    // Apply-pass pipeline/bind-group are built lazily in setSceneSdf (always called
    // before the first step); compiled variants are cached by source so re-selecting
    // a demo is instant.
    const applyPipelineCache = new Map<string, GPUComputePipeline>();
    function getApplyPipeline(scene: SceneSdfSpec): GPUComputePipeline {
        const src = buildApplyWgsl(scene);
        let pipe = applyPipelineCache.get(src);
        if (!pipe) {
            pipe = computePipeline("fluid-apply", src);
            applyPipelineCache.set(src, pipe);
        }
        return pipe;
    }
    let applyPipeline: GPUComputePipeline | null = null;
    // Predict is a single STATIC pipeline again (force-free). Any external force
    // (setForceField) runs as its own dedicated compute pass, so predict never
    // needs recompiling.
    const predictPipeline = computePipeline("fluid-predict", buildPredictWgsl());
    const finalizePipeline = computePipeline("fluid-finalize", FINALIZE_WGSL);
    const viscosityPipeline = computePipeline("fluid-viscosity", VISCOSITY_WGSL);
    const emitPipeline = computePipeline("fluid-emit", EMIT_WGSL);

    const predictBG = device.createBindGroup({
        layout: predictPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: velocityBuffer } },
            { binding: 2, resource: { buffer: predictedBuffer } },
            { binding: 3, resource: { buffer: simBuffer } },
        ],
    });

    // Dedicated external-force pass (setForceField). Built LAZILY on the first
    // non-null injection and cached by the force WGSL source, so NOTHING force
    // related is compiled until a force is actually used. `forceSpec` is the enable
    // flag (null = disabled); `forceBuiltSpec` tracks what the cached pipeline +
    // bind-group were built for, so re-enabling the same spec is free.
    const forcePipelineCache = new Map<string, GPUComputePipeline>();
    let forceSpec: ForceFieldSpec | null = null;
    let forceBuiltSpec: ForceFieldSpec | null = null;
    let forcePipeline: GPUComputePipeline | null = null;
    let forceBG: GPUBindGroup | null = null;
    function buildForceBG(pipe: GPUComputePipeline, spec: ForceFieldSpec): GPUBindGroup {
        return device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: positionBuffer } },
                { binding: 1, resource: { buffer: velocityBuffer } },
                { binding: 2, resource: { buffer: simBuffer } },
                { binding: 3, resource: { buffer: spec.buffer } },
            ],
        });
    }
    const clearGridBG = device.createBindGroup({
        layout: clearGridPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cellCountBuffer } },
            { binding: 1, resource: { buffer: cellCursorBuffer } },
        ],
    });
    const histogramBG = device.createBindGroup({
        layout: histogramPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: predictedBuffer } },
            { binding: 1, resource: { buffer: cellCountBuffer } },
            { binding: 2, resource: { buffer: gridBuffer } },
            { binding: 3, resource: { buffer: simBuffer } },
        ],
    });
    const scanLocalBG = device.createBindGroup({
        layout: scanLocalPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cellCountBuffer } },
            { binding: 1, resource: { buffer: cellStartBuffer } },
            { binding: 2, resource: { buffer: partialSumsBuffer } },
        ],
    });
    const scanPartialsBG = device.createBindGroup({
        layout: scanPartialsPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: partialSumsBuffer } }],
    });
    const scanAddBG = device.createBindGroup({
        layout: scanAddPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cellStartBuffer } },
            { binding: 1, resource: { buffer: partialSumsBuffer } },
        ],
    });
    const scatterBG = device.createBindGroup({
        layout: scatterPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: predictedBuffer } },
            { binding: 1, resource: { buffer: cellStartBuffer } },
            { binding: 2, resource: { buffer: cellCursorBuffer } },
            { binding: 3, resource: { buffer: sortedIdxBuffer } },
            { binding: 4, resource: { buffer: gridBuffer } },
            { binding: 5, resource: { buffer: simBuffer } },
        ],
    });
    // Reorder the live working set into sorted order (predicted/pos/vel -> sortedPos/
    // sortedPos0/sortedVel); the whole solve then runs over sorted slots.
    const reorderInBG = device.createBindGroup({
        layout: reorderInPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: sortedIdxBuffer } },
            { binding: 1, resource: { buffer: predictedBuffer } },
            { binding: 2, resource: { buffer: positionBuffer } },
            { binding: 3, resource: { buffer: velocityBuffer } },
            { binding: 4, resource: { buffer: sortedPosBuffer } },
            { binding: 5, resource: { buffer: sortedPos0Buffer } },
            { binding: 6, resource: { buffer: sortedVelBuffer } },
            { binding: 7, resource: { buffer: simBuffer } },
        ],
    });
    // Scatter the solved sorted working set back to original order (sortedPos/sortedVel ->
    // pos/vel + debug speed). Only live slots write, so dormant particles stay parked.
    const scatterBackBG = device.createBindGroup({
        layout: scatterBackPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: sortedIdxBuffer } },
            { binding: 1, resource: { buffer: sortedPosBuffer } },
            { binding: 2, resource: { buffer: sortedVelBuffer } },
            { binding: 3, resource: { buffer: positionBuffer } },
            { binding: 4, resource: { buffer: velocityBuffer } },
            { binding: 5, resource: { buffer: debugBuffer } },
            { binding: 6, resource: { buffer: simBuffer } },
        ],
    });
    const lambdaBG = device.createBindGroup({
        layout: lambdaPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: sortedPosBuffer } },
            { binding: 1, resource: { buffer: cellCountBuffer } },
            { binding: 2, resource: { buffer: cellStartBuffer } },
            { binding: 3, resource: { buffer: gridBuffer } },
            { binding: 4, resource: { buffer: simBuffer } },
            { binding: 5, resource: { buffer: sortedLambdaBuffer } },
        ],
    });
    const deltaBG = device.createBindGroup({
        layout: deltaPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: sortedPosBuffer } },
            { binding: 1, resource: { buffer: cellCountBuffer } },
            { binding: 2, resource: { buffer: cellStartBuffer } },
            { binding: 3, resource: { buffer: sortedLambdaBuffer } },
            { binding: 4, resource: { buffer: gridBuffer } },
            { binding: 5, resource: { buffer: simBuffer } },
            { binding: 6, resource: { buffer: sortedDeltaBuffer } },
        ],
    });
    function buildApplyBG(pipeline: GPUComputePipeline, scene: SceneSdfSpec): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: sortedPosBuffer } },
            { binding: 1, resource: { buffer: sortedDeltaBuffer } },
            { binding: 2, resource: { buffer: simBuffer } },
            { binding: 3, resource: { buffer: scene.buffer } },
        ];
        // Baked SDF grid: matches the @binding(4) storage decl injected by buildApplyWgsl.
        if (scene.sdfGrid) {
            entries.push({ binding: 4, resource: { buffer: scene.sdfGrid } });
        }
        return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    }
    let applyBG: GPUBindGroup | null = null;
    const finalizeBG = device.createBindGroup({
        layout: finalizePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: sortedPosBuffer } },
            { binding: 1, resource: { buffer: sortedPos0Buffer } },
            { binding: 2, resource: { buffer: sortedVelBuffer } },
            { binding: 3, resource: { buffer: simBuffer } },
        ],
    });
    const viscosityBG = device.createBindGroup({
        layout: viscosityPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: sortedPosBuffer } },
            { binding: 1, resource: { buffer: sortedVelBuffer } },
            { binding: 2, resource: { buffer: cellCountBuffer } },
            { binding: 3, resource: { buffer: cellStartBuffer } },
            { binding: 4, resource: { buffer: gridBuffer } },
            { binding: 5, resource: { buffer: simBuffer } },
            { binding: 6, resource: { buffer: sortedPos0Buffer } },
        ],
    });
    // Copy the smoothed velocity (viscosity wrote it into sortedPos0) back into sortedVel.
    const copyVelBG = device.createBindGroup({
        layout: copyVec4Pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: sortedPos0Buffer } },
            { binding: 1, resource: { buffer: sortedVelBuffer } },
            { binding: 2, resource: { buffer: simBuffer } },
        ],
    });
    const emitBG = device.createBindGroup({
        layout: emitPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: velocityBuffer } },
            { binding: 2, resource: { buffer: emittersBuffer } },
        ],
    });

    const particleGroups = Math.ceil(count / WORKGROUP_SIZE);
    const cellGroups = Math.ceil(numCells / WORKGROUP_SIZE);

    function dispatch(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline, bg: GPUBindGroup, groups: number): void {
        // Opt-in GPU timing: labels containing "foam" are the diffuse-particle passes
        // ("Foam gen"), everything else is the core solver ("Simulation"). No profiler
        // set → undefined → timing off. See FluidProfiler.
        const pass = encoder.beginComputePass({ label, timestampWrites: profiler?.pass(label.includes("foam") ? "Foam gen" : "Simulation") });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        if (groups > MAX_WORKGROUPS) {
            pass.dispatchWorkgroups(MAX_WORKGROUPS, Math.ceil(groups / MAX_WORKGROUPS), 1);
        } else {
            pass.dispatchWorkgroups(groups);
        }
        pass.end();
    }

    // ── Foam (Ihmsen 2012 diffuse particles) — lazily allocated on first setFoam ──
    // The pool + the three compute passes are built the first time foam is enabled, so
    // a sim that never turns foam on pays nothing (no buffers, no compiled shaders).
    // The ring buffer is sized D = poolScale × count (capped) and reused: foam-emit
    // overwrites the oldest slots via an atomic write-head modulo D.
    // The pool is sized purely from poolScale × count — no arbitrary ceiling. The only
    // bounds left are the device's: the pool is ONE storage buffer bound to the compute and
    // render passes (so it cannot exceed maxStorageBufferBindingSize), and the update pass
    // dispatches over it in a 2D-spilled grid (MAX_WORKGROUPS² groups, effectively boundless).
    const FOAM_CAP_LIMIT = Math.min(Math.floor(device.limits.maxStorageBufferBindingSize / 32), MAX_WORKGROUPS * MAX_WORKGROUPS * WORKGROUP_SIZE);
    const foamData = new ArrayBuffer(FOAM_BYTES);
    const foamF32 = new Float32Array(foamData);
    const foamU32 = new Uint32Array(foamData);
    let foamEnabled = false;
    let foamSeed = 0;
    let foamCapacity = 0;
    let foamPoolGroups = 0;
    let diffuseBuffer: GPUBuffer | null = null;
    let diffuseHeadBuffer: GPUBuffer | null = null;
    let sortedNormalBuffer: GPUBuffer | null = null;
    let foamParamsBuffer: GPUBuffer | null = null;
    let foamNormalsPipeline: GPUComputePipeline | null = null;
    let foamEmitPipeline: GPUComputePipeline | null = null;
    let foamUpdatePipeline: GPUComputePipeline | null = null;
    let foamNormalsBG: GPUBindGroup | null = null;
    let foamEmitBG: GPUBindGroup | null = null;
    let foamUpdateBG: GPUBindGroup | null = null;
    let diffusePool: DiffusePool | undefined;

    function buildFoamBindGroups(): void {
        foamNormalsBG = device.createBindGroup({
            layout: foamNormalsPipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: sortedPosBuffer } },
                { binding: 1, resource: { buffer: cellCountBuffer } },
                { binding: 2, resource: { buffer: cellStartBuffer } },
                { binding: 3, resource: { buffer: gridBuffer } },
                { binding: 4, resource: { buffer: simBuffer } },
                { binding: 5, resource: { buffer: sortedNormalBuffer! } },
            ],
        });
        foamEmitBG = device.createBindGroup({
            layout: foamEmitPipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: sortedIdxBuffer } },
                { binding: 1, resource: { buffer: cellCountBuffer } },
                { binding: 2, resource: { buffer: cellStartBuffer } },
                { binding: 3, resource: { buffer: sortedPosBuffer } },
                { binding: 4, resource: { buffer: sortedVelBuffer } },
                { binding: 5, resource: { buffer: sortedNormalBuffer! } },
                { binding: 6, resource: { buffer: gridBuffer } },
                { binding: 7, resource: { buffer: simBuffer } },
                { binding: 8, resource: { buffer: foamParamsBuffer! } },
                { binding: 9, resource: { buffer: diffuseBuffer! } },
                { binding: 10, resource: { buffer: diffuseHeadBuffer! } },
            ],
        });
        foamUpdateBG = device.createBindGroup({
            layout: foamUpdatePipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: cellCountBuffer } },
                { binding: 1, resource: { buffer: cellStartBuffer } },
                { binding: 2, resource: { buffer: sortedPosBuffer } },
                { binding: 3, resource: { buffer: sortedVelBuffer } },
                { binding: 4, resource: { buffer: gridBuffer } },
                { binding: 5, resource: { buffer: simBuffer } },
                { binding: 6, resource: { buffer: foamParamsBuffer! } },
                { binding: 7, resource: { buffer: diffuseBuffer! } },
            ],
        });
    }

    function ensureFoam(cfg: FoamConfig): void {
        if (!foamNormalsPipeline) {
            foamNormalsPipeline = computePipeline("fluid-foam-normals", FOAM_NORMALS_WGSL);
            foamEmitPipeline = computePipeline("fluid-foam-emit", FOAM_EMIT_WGSL);
            foamUpdatePipeline = computePipeline("fluid-foam-update", FOAM_UPDATE_WGSL);
            sortedNormalBuffer = device.createBuffer({ label: "fluid-foam-sorted-normals", size: count * 16, usage: GPUBufferUsage.STORAGE });
            foamParamsBuffer = device.createBuffer({ label: "fluid-foam-params", size: FOAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            diffuseHeadBuffer = device.createBuffer({ label: "fluid-foam-head", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        }
        let cap = Math.round(count * (cfg.poolScale ?? 3));
        cap = Math.max(1024, Math.min(cap, cfg.poolCapMax ?? Infinity, FOAM_CAP_LIMIT));
        if (cap !== foamCapacity || !diffuseBuffer) {
            diffuseBuffer?.destroy();
            diffuseBuffer = device.createBuffer({ label: "fluid-foam-pool", size: cap * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            foamCapacity = cap;
            foamPoolGroups = Math.ceil(cap / WORKGROUP_SIZE);
            // Zero the pool (all slots dead) and reset the ring head via a one-off encoder
            // (cheaper than uploading a multi-MB zero array on a pool resize).
            const enc = device.createCommandEncoder({ label: "fluid-foam-clear" });
            enc.clearBuffer(diffuseBuffer);
            enc.clearBuffer(diffuseHeadBuffer!);
            device.queue.submit([enc.finish()]);
            diffusePool = { buffer: diffuseBuffer, headBuffer: diffuseHeadBuffer!, capacity: cap };
            buildFoamBindGroups();
        }
        // Foam-specific knobs (h / dt / gravity / bounds are read from the Sim UBO).
        foamF32[0] = 5; // tauTaMin
        foamF32[1] = 20; // tauTaMax
        foamF32[2] = 2; // tauWcMin
        foamF32[3] = 8; // tauWcMax
        foamF32[4] = 5; // tauKMin
        foamF32[5] = 50; // tauKMax
        foamF32[6] = cfg.kTa ?? 40;
        foamF32[7] = cfg.kWc ?? 40;
        foamF32[8] = cfg.kb ?? 0.8;
        foamF32[9] = cfg.kd ?? 0.5;
        foamF32[10] = cfg.rv ?? particleRadius;
        foamF32[12] = cfg.tMin ?? 0.3;
        foamF32[13] = cfg.tMax ?? 2.0;
        // foamU32[14] (frameSeed) is written per frame in step().
        device.queue.writeBuffer(foamParamsBuffer!, 0, foamData);
    }

    return {
        count,
        particleRadius,
        positionBuffer,
        velocityBuffer,
        debugBuffer,
        // Speed colour normalisation: typical lively splash speed ≈ 5 world units/s.
        debugNorm: 1 / 5,
        get gpuBytes(): number {
            // Sum every GPU buffer this backend owns; the lazily-allocated foam pool +
            // uniforms are added only once foam has been enabled.
            let b =
                positionBuffer.size +
                velocityBuffer.size +
                predictedBuffer.size +
                debugBuffer.size +
                cellCountBuffer.size +
                cellStartBuffer.size +
                cellCursorBuffer.size +
                partialSumsBuffer.size +
                sortedIdxBuffer.size +
                sortedPosBuffer.size +
                sortedPos0Buffer.size +
                sortedLambdaBuffer.size +
                sortedDeltaBuffer.size +
                sortedVelBuffer.size +
                simBuffer.size +
                gridBuffer.size +
                emittersBuffer.size;
            if (diffuseBuffer) {
                b += diffuseBuffer.size;
            }
            if (diffuseHeadBuffer) {
                b += diffuseHeadBuffer.size;
            }
            if (sortedNormalBuffer) {
                b += sortedNormalBuffer.size;
            }
            if (foamParamsBuffer) {
                b += foamParamsBuffer.size;
            }
            return b;
        },
        get diffuse(): DiffusePool | undefined {
            return diffusePool;
        },
        step(encoder: GPUCommandEncoder, dt: number): void {
            // finalize divides by dt; a zero/NaN dt (e.g. the very first frame)
            // would poison every position with NaN, so skip such frames.
            if (!(dt > 0)) {
                return;
            }
            simF32[0] = dt;
            // Warm-up ramp: grow the live count one batch per frame, teleporting the
            // newly-activated particles from off-screen to their stored spawn position.
            if (liveCount < count) {
                const prev = liveCount;
                liveCount = Math.min(count, liveCount + warmupStep);
                device.queue.writeBuffer(positionBuffer, prev * 16, seedPositions, prev * 4, (liveCount - prev) * 4);
            }
            simU32[15] = liveCount;
            device.queue.writeBuffer(simBuffer, 0, simData);

            // PIX / GPU-capture debug group: scopes this frame's PBF compute passes
            // (plus the nested solver + foam groups) into one collapsible event.
            // Balanced by the popDebugGroup at the end of step().
            encoder.pushDebugGroup("PBF sim step");
            if (emitEnabled) {
                emitData[2] = emitSeed++;
                emitData[5] = dt;
                device.queue.writeBuffer(emittersBuffer, 0, emitData);
                dispatch(encoder, "fluid-emit", emitPipeline, emitBG, particleGroups);
            }
            if (forceSpec && forcePipeline && forceBG) {
                dispatch(encoder, "fluid-force", forcePipeline, forceBG, particleGroups);
            }
            dispatch(encoder, "fluid-predict", predictPipeline, predictBG, particleGroups);
            dispatch(encoder, "fluid-clear-grid", clearGridPipeline, clearGridBG, cellGroups);
            // Counting-sort the live particles by cell (Hoetzlein 2014): histogram ->
            // exclusive prefix sum (multi-level) -> scatter builds sortedIdx (slot ->
            // original index). reorder-in then copies the working set (predicted/pos/vel)
            // into sorted order, and the ENTIRE solve runs over sorted slots so
            // consecutive threads share neighbour cells in cache (Fluids v5.0
            // countingSortFull). No per-iteration gathers: the sorted buffers are mutated
            // in place and scatter-back writes results to original order once at the end.
            dispatch(encoder, "fluid-histogram", histogramPipeline, histogramBG, particleGroups);
            dispatch(encoder, "fluid-scan-local", scanLocalPipeline, scanLocalBG, scanChunks);
            dispatch(encoder, "fluid-scan-partials", scanPartialsPipeline, scanPartialsBG, 1);
            dispatch(encoder, "fluid-scan-add", scanAddPipeline, scanAddBG, cellGroups);
            dispatch(encoder, "fluid-scatter", scatterPipeline, scatterBG, particleGroups);
            dispatch(encoder, "fluid-reorder-in", reorderInPipeline, reorderInBG, particleGroups);
            encoder.pushDebugGroup(`constraint solve (${iterationsMut} iters)`);
            for (let it = 0; it < iterationsMut; it++) {
                dispatch(encoder, "fluid-lambda", lambdaPipeline, lambdaBG, particleGroups);
                dispatch(encoder, "fluid-delta", deltaPipeline, deltaBG, particleGroups);
                if (applyPipeline && applyBG) {
                    dispatch(encoder, "fluid-apply", applyPipeline, applyBG, particleGroups);
                }
            }
            encoder.popDebugGroup();
            dispatch(encoder, "fluid-finalize", finalizePipeline, finalizeBG, particleGroups);
            dispatch(encoder, "fluid-viscosity", viscosityPipeline, viscosityBG, particleGroups);
            // Viscosity wrote the smoothed velocity into sortedPos0 (to avoid a read/write
            // race on sortedVel); copy it back so scatter-back + foam read the FINAL
            // post-viscosity velocity from sortedVel exactly as before.
            dispatch(encoder, "fluid-copy-vel", copyVec4Pipeline, copyVelBG, particleGroups);
            // Write the solved sorted working set back to original order (pos/vel + speed).
            dispatch(encoder, "fluid-scatter-back", scatterBackPipeline, scatterBackBG, particleGroups);
            // Foam passes: generate + advect diffuse particles on the finalized fluid
            // state, reusing the neighbour grid built above. All foam passes dispatch over
            // the SAME cell-sorted slots and read the sorted buffers directly (sortedVel is
            // already the final post-viscosity velocity), so no gathers are needed.
            // Ihmsen: normals → emit. The update pass then classifies + advects the pool.
            if (foamEnabled && foamUpdateBG) {
                foamU32[14] = foamSeed++;
                device.queue.writeBuffer(foamParamsBuffer!, 0, foamData);
                encoder.pushDebugGroup("foam");
                dispatch(encoder, "fluid-foam-normals", foamNormalsPipeline!, foamNormalsBG!, particleGroups);
                dispatch(encoder, "fluid-foam-emit", foamEmitPipeline!, foamEmitBG!, particleGroups);
                dispatch(encoder, "fluid-foam-update", foamUpdatePipeline!, foamUpdateBG, foamPoolGroups);
                encoder.popDebugGroup();
            }
            encoder.popDebugGroup();
        },
        reset(): void {
            seed();
        },
        setParam(key: string, value: number): void {
            switch (key) {
                case "gravity":
                    simF32[1] = value;
                    break;
                case "restDensity":
                    simF32[2] = value;
                    break;
                case "relaxation":
                    simF32[7] = value;
                    break;
                case "scorr":
                    simF32[8] = value;
                    break;
                case "viscosity":
                    simF32[11] = value;
                    break;
                case "boundaryDensity":
                    simF32[14] = value;
                    break;
                case "iterations":
                    iterationsMut = Math.max(1, Math.round(value));
                    break;
            }
        },
        setSceneSdf(spec: SceneSdfSpec | null): void {
            if (spec) {
                applyPipeline = getApplyPipeline(spec);
                applyBG = buildApplyBG(applyPipeline, spec);
            } else {
                applyPipeline = null;
                applyBG = null;
            }
        },
        setEmitters(cfg: EmitterConfig | null): void {
            packEmitters(emitData, cfg);
            emitEnabled = !!cfg && cfg.emitters.length > 0;
            device.queue.writeBuffer(emittersBuffer, 0, emitData);
        },
        setSpawn(min: [number, number, number], max: [number, number, number], accept?: ((x: number, y: number, z: number) => boolean) | null): void {
            spawnMin[0] = min[0];
            spawnMin[1] = min[1];
            spawnMin[2] = min[2];
            spawnMax[0] = max[0];
            spawnMax[1] = max[1];
            spawnMax[2] = max[2];
            spawnAccept = accept ?? null;
        },
        setWarmup(frames: number): void {
            // Frames over which reset()/seed() gradually releases particles (0 = all at
            // once). Takes effect on the next seed()/reset(). Mirrors the MLS backend.
            warmupFrames = Math.max(0, Math.floor(frames));
            warmupStep = warmupFrames > 0 ? Math.max(1, Math.ceil(count / warmupFrames)) : count;
        },
        setForceField(spec: ForceFieldSpec | null): void {
            forceSpec = spec;
            if (!spec || spec === forceBuiltSpec) {
                return;
            }
            let pipe = forcePipelineCache.get(spec.wgsl);
            if (!pipe) {
                pipe = computePipeline("fluid-force", buildForceWgsl(spec));
                forcePipelineCache.set(spec.wgsl, pipe);
            }
            forcePipeline = pipe;
            forceBG = buildForceBG(pipe, spec);
            forceBuiltSpec = spec;
        },
        setFoam(cfg: FoamConfig | null): void {
            if (!cfg) {
                foamEnabled = false;
                // Empty the pool so a later re-enable starts clean (no frozen ghosts).
                if (diffuseBuffer && diffuseHeadBuffer) {
                    const enc = device.createCommandEncoder({ label: "fluid-foam-off-clear" });
                    enc.clearBuffer(diffuseBuffer);
                    enc.clearBuffer(diffuseHeadBuffer);
                    device.queue.submit([enc.finish()]);
                }
                return;
            }
            ensureFoam(cfg);
            foamEnabled = true;
        },
        setProfiler(p: FluidProfiler | null): void {
            profiler = p;
        },
        dispose(): void {
            positionBuffer.destroy();
            velocityBuffer.destroy();
            predictedBuffer.destroy();
            debugBuffer.destroy();
            cellCountBuffer.destroy();
            cellStartBuffer.destroy();
            cellCursorBuffer.destroy();
            partialSumsBuffer.destroy();
            sortedIdxBuffer.destroy();
            sortedPosBuffer.destroy();
            sortedPos0Buffer.destroy();
            sortedLambdaBuffer.destroy();
            sortedDeltaBuffer.destroy();
            sortedVelBuffer.destroy();
            simBuffer.destroy();
            gridBuffer.destroy();
            diffuseBuffer?.destroy();
            diffuseHeadBuffer?.destroy();
            sortedNormalBuffer?.destroy();
            foamParamsBuffer?.destroy();
        },
    };
}
