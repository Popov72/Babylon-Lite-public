// Alternative GPU fluid backend — MLS-MPM (Moving Least Squares Material Point
// Method), the algorithm behind matsuoka-601's "Splash". A grid-transfer method
// (no neighbour search), which scales to far more particles than the PBF solver.
//
// References:
//   • MLS-MPM: Hu et al. 2018, "A Moving Least Squares Material Point Method with Displacement
//     Discontinuity and Two-Way Rigid Body Coupling" —
//     https://yuanming.taichi.graphics/publication/2018-mlsmpm/mls-mpm-cpic.pdf
//   • APIC transfer: Jiang et al. 2015, "The Affine Particle-In-Cell Method" —
//     https://disneyanimation.com/publications/the-affine-particle-in-cell-method/
//   • Foam (spray/foam/bubbles): Ihmsen et al. 2012, "Unified spray, foam and air bubbles for
//     particle-based fluids" —
//     https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf
//
// Demo-local, exposes the same surface as the PBF sim (positionBuffer +
// debugBuffer in WORLD units, step/reset/dispose) so the renderer and
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
// Per substep: clearGrid → block counting sort of the live particles (histogram → prefix
// sum → scatter into sortedIdx) → tiled p2g_mass → tiled p2g_vel → updateGrid (v = p/m,
// gravity, domain walls) → g2p (gather v + affine C, advect, capsule/hole boundary). The
// two P2G scatters run one workgroup per grid BLOCK: each block stages its particles'
// mass / momentum into a workgroup-shared apron tile and flushes it to the global grid
// with a single atomicAdd per touched node, trading heavily contended global atomics for
// far cheaper workgroup-shared ones. Because the grid accumulates in integer fixed-point,
// reordering the particles and staging through shared memory is bit-identical to the old
// per-particle global scatter (integer addition is order-independent). A copy pass then
// packs world positions + speed for the renderer.

import type { EngineContext } from "../engine/engine.js";
import type { FluidSim, FluidSimBaseOptions, SceneSdfSpec, EmitterConfig, FluidFlowConfig, ForceFieldSpec, FoamConfig, DiffusePool, FluidProfiler } from "./sim-common.js";
import {
    SPAWN_ACCEPT_TRIES,
    FLUID_FLOW_RUNTIME_WGSL,
    FLUID_FLOW_STRUCT_WGSL,
    FOAM_ACTIVE_FINISH_WGSL,
    FOAM_ACTIVE_PREPARE_WGSL,
    FOAM_BYTES,
    FOAM_COMMON_WGSL,
    createFluidFlowState,
    createFluidInitialParticles,
    disposeFluidFlowState,
    foamActiveListOffset,
    foamActiveStateBytes,
    legacyEmitterConfigToFluidFlow,
    prepareFluidFlowFrame,
    resetFluidFlowState,
    SCENE_NORMAL_WGSL,
    SCENE_SDF_GRID_WGSL,
    setFluidFlowConfig,
} from "./sim-common.js";

// Opt-in GPU timing hook (see FluidProfiler / lab gpu-profiler.ts). Module-scoped:
// null by default so `profiler?.pass(...)` is undefined and timing costs nothing.
let profiler: FluidProfiler | null = null;

const WORKGROUP_SIZE = 64;
// WebGPU caps a dispatch at 65535 workgroups per dimension. The MLS grid can need
// far more groups than that at small particle sizes (very fine cells), so
// cell-indexed dispatches spill the overflow into a second (y) dimension and the
// cell kernels rebuild the linear index from num_workgroups.x.
const MAX_WORKGROUPS = 65535;
const FIXED_POINT = 1e7; // float→i32 scale for atomic grid accumulation
// Block-tiled P2G: TILE grid cells per block per axis. A block's particles have their
// base cell inside a TILE^3 region and therefore touch a (TILE+2)^3 apron of grid nodes.
// That apron is staged in workgroup-shared memory and flushed once to global (one global
// atomicAdd per touched node) instead of one global atomic per (particle, node) — trading
// contended global atomics for far cheaper workgroup-shared atomics.
const TILE = 4;
const TILE_NODES = TILE + 2; // apron size per axis (base-1 .. base+TILE)
const TILE_NODES3 = TILE_NODES * TILE_NODES * TILE_NODES; // shared-tile slot count (6^3 = 216)
// Prefix-sum workgroup width for the per-substep block counting-sort scan.
const SCAN_WG = 256;

// Params uniform (std140), 16-byte rows:
//   origin.xyz, dx
//   gridDim.xyz (f32), pad
//   capsuleA.xyz, capsuleRadius
//   capsuleB.xyz, groundY
//   dt, gravity, restDensity, stiffness
//   viscosity, pad, pad, pad
//   counts: numParticles(u32), containerMode(u32), pad, pad
//   boxMin.xyz+pad, boxMax.xyz+pad (container box for containerMode 2)
//   obsA (cx,cz,halfWidth,halfThickness), obsB (cos,sin,omega,enabled) — rotating paddle
//   misc2 (restitution, _, _, _)
const PARAMS_F32 = 7 * 4 + 8 + 8 + 4; // header + box + obstacle + misc2
const PARAMS_BYTES = PARAMS_F32 * 4;
const COUNTS_OFFSET_F32 = 24; // start of the counts vec4 (u32 view)
const BOX_BASE_F32 = COUNTS_OFFSET_F32 + 4; // boxMin/boxMax follow the counts vec4
const OBS_BASE_F32 = BOX_BASE_F32 + 8;
const MISC2_BASE_F32 = OBS_BASE_F32 + 8;

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
    counts: vec4<u32>,      // numParticles, containerMode, _, _
    boxMin: vec4<f32>,
    boxMax: vec4<f32>,
    obsA: vec4<f32>,        // cx, cz, halfWidth, halfThickness (rotating paddle)
    obsB: vec4<f32>,        // cos, sin, omega, enabled
    misc2: vec4<f32>,       // x = restitution (0 = free-slip, 1 = elastic mirror)
};

// Reflect a velocity for a collision, given n = the penetration normal (a unit
// vector pointing FROM the fluid INTO the solid). The component of v along n is
// the part driving into the surface; e is the restitution: e = 0 removes it
// (free-slip, no bounce), e = 1 reverses it (elastic mirror), in between bounces
// partially. Tangential velocity is always preserved, so the fluid slips along
// the surface and spreads instead of clumping.
fn reflectVel(v: vec3<f32>, n: vec3<f32>, e: f32) -> vec3<f32> {
    let vn = dot(v, n);
    if (vn <= 0.0) { return v; }
    return v - (1.0 + e) * vn * n;
}

// World-space surface velocity of the rotating paddle at a point (rx,rz) given
// relative to the pivot. Consistent with the position rotation used in the slab
// test: v = d/dt(local→world) with d(angle)/dt = omega.
fn obstacleSurfaceVel(rx: f32, rz: f32, p: Params) -> vec3<f32> {
    let omega = p.obsB.z;
    return vec3<f32>(-omega * rz, 0.0, omega * rx);
}

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
@group(0) @binding(0) var<storage, read_write> cells: array<vec4<u32>>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&cells)) { return; }
    cells[i] = vec4<u32>(0u);
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

// Block math shared by the counting sort + the tiled P2G. A block is a TILE^3 tile of
// grid cells; blockDim = ceil(gridDim / TILE) and the block of a cell is floor(cell/TILE)
// linearised as (bx*blockDim.y + by)*blockDim.z + bz. Computed from p.dim (the grid dims)
// so it always matches the numBlocks the CPU dispatches. Requires COMMON_WGSL (Params).
const BLOCK_WGSL = /* wgsl */ `
const TILE_I: i32 = ${TILE};
const TILE_U: u32 = ${TILE}u;
const TN_U: u32 = ${TILE_NODES}u; // nodes per axis in a block apron (TILE + 2)
fn blockDimOf(p: Params) -> vec3<u32> {
    let g = vec3<u32>(u32(p.dim.x), u32(p.dim.y), u32(p.dim.z));
    return (g + vec3<u32>(TILE_U - 1u)) / vec3<u32>(TILE_U);
}
fn numBlocksOf(p: Params) -> u32 {
    let b = blockDimOf(p);
    return b.x * b.y * b.z;
}
// Block index of an in-grid (non-negative) cell.
fn blockIndexOfCell(c: vec3<i32>, p: Params) -> u32 {
    let bd = blockDimOf(p);
    let b = vec3<u32>(c) / vec3<u32>(TILE_U);
    return (b.x * bd.y + b.y) * bd.z + b.z;
}
// Block coordinate (bx,by,bz) of a linear block index.
fn blockCoordOf(bIdx: u32, p: Params) -> vec3<u32> {
    let bd = blockDimOf(p);
    return vec3<u32>(bIdx / (bd.y * bd.z), (bIdx / bd.z) % bd.y, bIdx % bd.z);
}`;

const PAGE_HELPERS_WGSL = /* wgsl */ `
fn pageLocalIndex(node: vec3<i32>) -> u32 {
    let local = vec3<u32>(node) % vec3<u32>(TILE_U);
    return (local.x * TILE_U + local.y) * TILE_U + local.z;
}
fn pageCellIndex(node: vec3<i32>, p: Params) -> u32 {
    return pageMap[blockIndexOfCell(node, p)] * ${TILE * TILE * TILE}u + pageLocalIndex(node);
}`;

// ── Per-substep block counting sort ──────────────────────────────────────────────
// The tiled P2G runs one workgroup per grid BLOCK and needs that block's particles as a
// contiguous run. A counting sort by block builds exactly that: histogram (blockCount),
// exclusive prefix sum (blockStart), then a scatter of each live particle index into
// sortedIdx[blockStart[block] + cursor]. blockCount/blockCursor are zeroed by the CPU
// (clearBuffer) before the histogram/scatter. Reordering the particles does not change
// any accumulated sum (the grid is integer fixed-point, so the transfer is bit-identical
// regardless of order), only the memory access pattern of the scatter.
//
// S1 — histogram: each live particle bumps its block's count.
function buildHistogramWgsl(fusedBlockDiscovery: boolean): string {
    const fusedDecls = fusedBlockDiscovery
        ? `
@group(0) @binding(3) var<storage, read_write> activeBlockList: array<u32>;
@group(0) @binding(4) var<storage, read_write> activeCount: array<atomic<u32>>;`
        : "";
    const increment = fusedBlockDiscovery
        ? `
    let b = blockIndexOfCell(c, p);
    if (atomicAdd(&blockCount[b], 1u) == 0u) {
        activeBlockList[atomicAdd(&activeCount[0], 1u)] = b;
    }`
        : `
    atomicAdd(&blockCount[blockIndexOfCell(c, p)], 1u);`;
    return /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${BLOCK_WGSL}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> blockCount: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> p: Params;
${fusedDecls}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x) { return; }
    if (i >= p.counts.z) { return; } // warm-up: dormant particles are not sorted
    let c = cellOf(particles[i].position, p);
    if (!inGrid(c, p)) { return; } // out-of-grid base cell deposits nothing (as in the old P2G)
${increment}
}`;
}

// S2a — per-chunk exclusive scan of blockCount into blockStart, plus each chunk's total
// into partialSums. Hillis-Steele inclusive scan in shared memory, converted to exclusive.
const SCAN_LOCAL_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> blockCount: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockStart: array<u32>;
@group(0) @binding(2) var<storage, read_write> partialSums: array<u32>;
var<workgroup> s: array<u32, ${SCAN_WG}>;
@compute @workgroup_size(${SCAN_WG})
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let n = arrayLength(&blockCount);
    let chunk = wid.x + wid.y * ng.x;
    let idx = chunk * ${SCAN_WG}u + lid;
    var v = 0u;
    if (idx < n) { v = blockCount[idx]; }
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
    if (idx < n) { blockStart[idx] = s[lid] - v; } // inclusive - self = exclusive
    if (lid == ${SCAN_WG}u - 1u) { partialSums[chunk] = s[${SCAN_WG}u - 1u]; } // chunk total
}`;

// S2b — exclusive scan of the per-chunk totals (partialSums), in place, by a SINGLE
// workgroup that chains over the array in SCAN_WG-wide strides carrying a running
// offset. Handles an arbitrary number of chunks (no single-workgroup size assumption).
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

// S2c — add each chunk's scanned offset back into blockStart, yielding the global
// exclusive prefix sum.
const SCAN_ADD_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> blockStart: array<u32>;
@group(0) @binding(1) var<storage, read> partialSums: array<u32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&blockStart)) { return; }
    blockStart[i] = blockStart[i] + partialSums[i / ${SCAN_WG}u];
}`;

// S3 — scatter each live particle index into its block's contiguous run.
const SCATTER_WGSL = /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${BLOCK_WGSL}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> blockStart: array<u32>;
@group(0) @binding(2) var<storage, read_write> blockCursor: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sortedIdx: array<u32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= p.counts.x) { return; }
    if (i >= p.counts.z) { return; }
    let c = cellOf(particles[i].position, p);
    if (!inGrid(c, p)) { return; }
    let b = blockIndexOfCell(c, p);
    let slot = blockStart[b] + atomicAdd(&blockCursor[b], 1u);
    sortedIdx[slot] = i;
}`;

// Sparse dispatch. Particle blocks with a non-zero histogram count are
// compacted into activeBlockList, then expanded to the unique 3x3x3 halo of grid-node
// blocks their P2G stencils can touch. The lists drive indirect P2G / clear / update
// dispatches while the legacy path continues to process the full dense grid.
const COMPACT_ACTIVE_BLOCKS_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> blockCount: array<u32>;
@group(0) @binding(1) var<storage, read_write> activeBlockList: array<u32>;
@group(0) @binding(2) var<storage, read_write> activeCount: array<atomic<u32>>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&blockCount) || blockCount[i] == 0u) { return; }
    activeBlockList[atomicAdd(&activeCount[0], 1u)] = i;
}`;

const FINALIZE_INDIRECT_WGSL = /* wgsl */ `
const MAX_GROUPS: u32 = ${MAX_WORKGROUPS}u;
@group(0) @binding(0) var<storage, read> count: array<u32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let n = count[0];
    args[0] = min(n, MAX_GROUPS);
    args[1] = select(0u, (n + MAX_GROUPS - 1u) / MAX_GROUPS, n > 0u);
    args[2] = 1u;
}`;

const MARK_ACTIVE_NODE_BLOCKS_WGSL = /* wgsl */ `
${COMMON_WGSL}
${BLOCK_WGSL}
@group(0) @binding(0) var<storage, read> activeBlockList: array<u32>;
@group(0) @binding(1) var<storage, read> activeCount: array<u32>;
@group(0) @binding(2) var<storage, read_write> nodeFlags: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> nodeBlockList: array<u32>;
@group(0) @binding(4) var<storage, read_write> nodeCount: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p: Params;
@compute @workgroup_size(32)
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>
) {
    let activeSlot = wid.x + wid.y * ng.x;
    if (activeSlot >= activeCount[0] || lid >= 27u) { return; }
    let b = vec3<i32>(blockCoordOf(activeBlockList[activeSlot], p));
    let d = vec3<i32>(i32(lid / 9u), i32((lid / 3u) % 3u), i32(lid % 3u)) - vec3<i32>(1);
    let nb = b + d;
    let bd = vec3<i32>(blockDimOf(p));
    if (any(nb < vec3<i32>(0)) || any(nb >= bd)) { return; }
    let nbu = vec3<u32>(nb);
    let idx = (nbu.x * u32(bd.y) + nbu.y) * u32(bd.z) + nbu.z;
    if (atomicExchange(&nodeFlags[idx], 1u) == 0u) {
        nodeBlockList[atomicAdd(&nodeCount[0], 1u)] = idx;
    }
}`;

function buildClearActiveBlocksWgsl(pagedGrid: boolean): string {
    const pageDecl = pagedGrid ? "\n@group(0) @binding(4) var<storage, read> pageMap: array<u32>;" : "";
    const clear = pagedGrid
        ? `
    if (inGrid(node, p)) {
        let page = pageMap[nodeBlockList[slot]];
        if (page != 0u) { cells[page * ${TILE * TILE * TILE}u + pageLocalIndex(node)] = vec4<u32>(0u); }
    }`
        : "\n    if (inGrid(node, p)) { cells[index1D(node, p)] = vec4<u32>(0u); }";
    return /* wgsl */ `
${COMMON_WGSL}
${BLOCK_WGSL}
@group(0) @binding(0) var<storage, read_write> cells: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> nodeBlockList: array<u32>;
@group(0) @binding(2) var<storage, read> nodeCount: array<u32>;
@group(0) @binding(3) var<uniform> p: Params;
${pageDecl}
${pagedGrid ? PAGE_HELPERS_WGSL : ""}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>
) {
    let slot = wid.x + wid.y * ng.x;
    if (slot >= nodeCount[0]) { return; }
    let b0 = vec3<i32>(blockCoordOf(nodeBlockList[slot], p)) * TILE_I;
    let lc = vec3<i32>(i32(lid / (TILE_U * TILE_U)), i32((lid / TILE_U) % TILE_U), i32(lid % TILE_U));
    let node = b0 + lc;
${clear}
}`;
}

const ASSIGN_GRID_PAGES_WGSL = /* wgsl */ `
${COMMON_WGSL}
${BLOCK_WGSL}
@group(0) @binding(0) var<storage, read> nodeBlockList: array<u32>;
@group(0) @binding(1) var<storage, read> nodeCount: array<u32>;
@group(0) @binding(2) var<storage, read_write> pageMap: array<u32>;
@group(0) @binding(3) var<storage, read_write> pageState: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> cells: array<vec4<u32>>;
@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let slot = wid.x + wid.y * ng.x;
    if (slot >= nodeCount[0]) { return; }
    let rawPage = atomicAdd(&pageState[1], 1u);
    let maxPages = arrayLength(&cells) / ${TILE * TILE * TILE}u - 1u;
    if (rawPage < maxPages) {
        pageMap[nodeBlockList[slot]] = rawPage + 1u;
    } else {
        atomicStore(&pageState[2], 1u);
    }
}`;

// Tiled particle-to-grid transfer. One workgroup per grid block scatters its sorted run
// of particles into a workgroup-shared apron tile, then flushes the tile to the global
// grid with one atomicAdd per touched node. Because the grid is integer fixed-point, the
// staged sums are bit-identical to the per-particle global scatter (integer addition is
// order-independent); only the number of contended global atomics changes.
//
// A block's base cells lie in [B0, B0+TILE); their 3x3x3 stencils touch nodes in
// [B0-1, B0+TILE], a TILE_NODES^3 apron. Local node index = (base - B0) + g (g in 0..2),
// always in [0, TILE_NODES) per axis. Uniform control flow around the barriers: the whole
// workgroup shares bIdx and blockCount, so the out-of-range and empty-block returns are
// uniform (all threads or none) and happen before any barrier.
//
// Pass 1 — mass only (staged, 1 shared atomic + at most 1 global atomic per node).
function buildP2gMassTiledWgsl(activeBlocks: boolean, pagedGrid: boolean): string {
    const activeDecls = activeBlocks
        ? `
@group(0) @binding(6) var<storage, read> activeBlockList: array<u32>;
@group(0) @binding(7) var<storage, read> activeCount: array<u32>;`
        : "";
    const blockLookup = activeBlocks
        ? `
    let activeSlot = wid.x + wid.y * ng.x;
    if (activeSlot >= activeCount[0]) { return; }
    let bIdx = activeBlockList[activeSlot];`
        : `
    let bIdx = wid.x + wid.y * ng.x;
    if (bIdx >= numBlocksOf(p)) { return; }`;
    const pageDecls = pagedGrid
        ? `
@group(0) @binding(8) var<storage, read> pageMap: array<u32>;
${PAGE_HELPERS_WGSL}`
        : "";
    const writeMass = pagedGrid
        ? `
            let page = pageMap[blockIndexOfCell(node, p)];
            if (page != 0u) { atomicAdd(&cells[page * ${TILE * TILE * TILE}u + pageLocalIndex(node)].mass, m); }`
        : " atomicAdd(&cells[index1D(node, p)].mass, m);";
    return /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${WEIGHTS_WGSL}
${BLOCK_WGSL}
struct Cell { vx: atomic<i32>, vy: atomic<i32>, vz: atomic<i32>, mass: atomic<i32>, };
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> blockStart: array<u32>;
@group(0) @binding(4) var<storage, read> blockCount: array<u32>;
@group(0) @binding(5) var<storage, read> sortedIdx: array<u32>;
${activeDecls}
${pageDecls}
var<workgroup> tileMass: array<atomic<i32>, ${TILE_NODES3}>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
${blockLookup}
    ${pagedGrid ? "if (activeCount[2] != 0u) { return; }" : ""}
    let cnt = blockCount[bIdx];
    if (cnt == 0u) { return; }
    let start = blockStart[bIdx];
    let B0 = vec3<i32>(blockCoordOf(bIdx, p)) * TILE_I;

    for (var si = tid; si < ${TILE_NODES3}u; si = si + ${WORKGROUP_SIZE}u) { atomicStore(&tileMass[si], 0); }
    workgroupBarrier();

    for (var k = tid; k < cnt; k = k + ${WORKGROUP_SIZE}u) {
        let pos = particles[sortedIdx[start + k]].position;
        let base = cellOf(pos, p);
        let w = weightsOf(pos, p);
        let lbase = base - B0; // per-axis in [0, TILE)
        for (var gx = 0; gx < 3; gx++) {
        for (var gy = 0; gy < 3; gy++) {
        for (var gz = 0; gz < 3; gz++) {
            let weight = w[gx].x * w[gy].y * w[gz].z;
            let ln = ((lbase.x + gx) * i32(TN_U) + (lbase.y + gy)) * i32(TN_U) + (lbase.z + gz);
            atomicAdd(&tileMass[ln], enc(weight)); // particle mass = 1
        }}}
    }
    workgroupBarrier();

    for (var si = tid; si < ${TILE_NODES3}u; si = si + ${WORKGROUP_SIZE}u) {
        let m = atomicLoad(&tileMass[si]);
        if (m != 0) {
            let lx = i32(si / (TN_U * TN_U));
            let ly = i32((si / TN_U) % TN_U);
            let lz = i32(si % TN_U);
            let node = B0 - vec3<i32>(1) + vec3<i32>(lx, ly, lz);
            if (inGrid(node, p)) {${writeMass}
            }
        }
    }
}`;
}

// Pass 2 — gather the current node density from GLOBAL mass, compute the EOS/viscous
// stress exactly as the old p2g-vel, then stage the APIC + stress momentum into a shared
// velocity tile and flush (3 global atomics per touched node).
function buildP2gVelTiledWgsl(activeBlocks: boolean, pagedGrid: boolean): string {
    const activeDecls = activeBlocks
        ? `
@group(0) @binding(6) var<storage, read> activeBlockList: array<u32>;
@group(0) @binding(7) var<storage, read> activeCount: array<u32>;`
        : "";
    const blockLookup = activeBlocks
        ? `
    let activeSlot = wid.x + wid.y * ng.x;
    if (activeSlot >= activeCount[0]) { return; }
    let bIdx = activeBlockList[activeSlot];`
        : `
    let bIdx = wid.x + wid.y * ng.x;
    if (bIdx >= numBlocksOf(p)) { return; }`;
    const pageDecls = pagedGrid
        ? `
@group(0) @binding(8) var<storage, read> pageMap: array<u32>;
${PAGE_HELPERS_WGSL}`
        : "";
    const massIndex = pagedGrid ? "pageCellIndex(node, p)" : "u32(index1D(node, p))";
    const writeVelocity = pagedGrid
        ? `
            let page = pageMap[blockIndexOfCell(node, p)];
            if (page != 0u) {
                let idx = page * ${TILE * TILE * TILE}u + pageLocalIndex(node);
                atomicAdd(&cells[idx].vx, vx);
                atomicAdd(&cells[idx].vy, vy);
                atomicAdd(&cells[idx].vz, vz);
            }`
        : `
            let idx = u32(index1D(node, p));
            atomicAdd(&cells[idx].vx, vx);
            atomicAdd(&cells[idx].vy, vy);
            atomicAdd(&cells[idx].vz, vz);`;
    return /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${WEIGHTS_WGSL}
${BLOCK_WGSL}
struct Cell { vx: atomic<i32>, vy: atomic<i32>, vz: atomic<i32>, mass: i32, };
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> blockStart: array<u32>;
@group(0) @binding(4) var<storage, read> blockCount: array<u32>;
@group(0) @binding(5) var<storage, read> sortedIdx: array<u32>;
${activeDecls}
${pageDecls}
var<workgroup> tileVx: array<atomic<i32>, ${TILE_NODES3}>;
var<workgroup> tileVy: array<atomic<i32>, ${TILE_NODES3}>;
var<workgroup> tileVz: array<atomic<i32>, ${TILE_NODES3}>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
${blockLookup}
    ${pagedGrid ? "if (activeCount[2] != 0u) { return; }" : ""}
    let cnt = blockCount[bIdx];
    if (cnt == 0u) { return; }
    let start = blockStart[bIdx];
    let B0 = vec3<i32>(blockCoordOf(bIdx, p)) * TILE_I;
    let dx = p.origin.w;
    let dt = p.sim0.x;

    for (var si = tid; si < ${TILE_NODES3}u; si = si + ${WORKGROUP_SIZE}u) {
        atomicStore(&tileVx[si], 0);
        atomicStore(&tileVy[si], 0);
        atomicStore(&tileVz[si], 0);
    }
    workgroupBarrier();

    for (var k = tid; k < cnt; k = k + ${WORKGROUP_SIZE}u) {
        let pi = sortedIdx[start + k];
        let pos = particles[pi].position;
        let v = particles[pi].v;
        let C = particles[pi].C;
        let base = cellOf(pos, p);
        let w = weightsOf(pos, p);
        let lbase = base - B0;

        var density = 0.0;
        for (var gx = 0; gx < 3; gx++) {
        for (var gy = 0; gy < 3; gy++) {
        for (var gz = 0; gz < 3; gz++) {
            let weight = w[gx].x * w[gy].y * w[gz].z;
            let node = base + vec3<i32>(gx - 1, gy - 1, gz - 1);
            if (!inGrid(node, p)) { continue; }
            density += dec(cells[${massIndex}].mass) * weight;
        }}}

        // Stress term (pressure + viscous). Isolated particle (zero density) deposits APIC
        // momentum only, matching the old p2g-vel zero-density skip.
        var term0 = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
        if (density > 0.0) {
            let volume = 1.0 / density;
            let pressure = max(0.0, p.sim0.w * (density / p.sim0.z - 1.0));
            let strain = C + transpose(C);
            var stress = mat3x3<f32>(-pressure, 0.0, 0.0, 0.0, -pressure, 0.0, 0.0, 0.0, -pressure);
            stress += p.sim1.x * strain;
            let Dinv = 4.0 / (dx * dx);
            term0 = -volume * Dinv * dt * stress;
        }

        for (var gx = 0; gx < 3; gx++) {
        for (var gy = 0; gy < 3; gy++) {
        for (var gz = 0; gz < 3; gz++) {
            let weight = w[gx].x * w[gy].y * w[gz].z;
            let node = base + vec3<i32>(gx - 1, gy - 1, gz - 1);
            let nodeCenter = p.origin.xyz + (vec3<f32>(node) + 0.5) * dx;
            let cellDist = nodeCenter - pos;
            let Q = C * cellDist;
            let vel = weight * (v + Q) + (term0 * cellDist) * weight; // APIC momentum + stress
            let ln = ((lbase.x + gx) * i32(TN_U) + (lbase.y + gy)) * i32(TN_U) + (lbase.z + gz);
            atomicAdd(&tileVx[ln], enc(vel.x));
            atomicAdd(&tileVy[ln], enc(vel.y));
            atomicAdd(&tileVz[ln], enc(vel.z));
        }}}
    }
    workgroupBarrier();

    for (var si = tid; si < ${TILE_NODES3}u; si = si + ${WORKGROUP_SIZE}u) {
        let vx = atomicLoad(&tileVx[si]);
        let vy = atomicLoad(&tileVy[si]);
        let vz = atomicLoad(&tileVz[si]);
        if (vx != 0 || vy != 0 || vz != 0) {
            let lx = i32(si / (TN_U * TN_U));
            let ly = i32((si / TN_U) % TN_U);
            let lz = i32(si % TN_U);
            let node = B0 - vec3<i32>(1) + vec3<i32>(lx, ly, lz);
            if (inGrid(node, p)) {
${writeVelocity}
            }
        }
    }
}`;
}

// The grid-velocity update. A CLOSED container (gridConfine !== false) confines the
// fluid at the grid with a generic SDF separating wall: at nodes just outside the
// domain the wall velocity BC — reflectVel with the restitution (free-slip at 0, bounce
// at >0) — packs the fluid against the wall over the kernel width, with no glued layer
// and no gap (a per-particle position snap causes one or the other). A per-particle
// container (gridConfine === false) skips the grid wall — the coarse grid reflection
// would miss its thin curved shell — and confines per-particle in G2P instead. Static
// walls only; a moving boundary (paddle: |−∂sdf/∂t| large) is left to the per-particle
// G2P moving-boundary resolve.
function buildUpdateGridWgsl(scene: SceneSdfSpec, activeBlocks: boolean, pagedGrid: boolean): string {
    const closed = scene.gridConfine !== false;
    // Baked SDF grid (optional): only the CLOSED path injects the scene SDF here, so the
    // storage grid + sampler are injected only then (else the binding would be unused and
    // stripped from the layout:"auto" layout, breaking the bind group). Binding 3 is free
    // (0=cells, 1=p, 2=sceneSdfParams).
    const gridInject = closed && scene.sdfGrid ? `\n@group(0) @binding(3) var<storage, read> sceneSdfGrid: array<f32>;\n${SCENE_SDF_GRID_WGSL}` : "";
    const decls = closed ? `${scene.struct}\n@group(0) @binding(2) var<uniform> sceneSdfParams: SceneSdfParams;${gridInject}\n${scene.sdf}\n${SCENE_NORMAL_WGSL}` : "";
    const wall = closed
        ? `
    let cw = p.origin.xyz + (vec3<f32>(f32(x), f32(y), f32(z)) + 0.5) * p.origin.w;
    if (sceneSdf(cw, 0.0) < 0.0) {
        let vN = -(sceneSdf(cw, 0.002) - sceneSdf(cw, 0.0)) / 0.002; // boundary normal speed
        if (abs(vN) < 0.05) { // static wall only — the moving paddle is handled in G2P
            let n = sceneNormal(cw, 0.0);
            v = reflectVel(v, -n, p.misc2.x);
        }
    }`
        : "";
    const activeDecls = activeBlocks
        ? `
@group(0) @binding(4) var<storage, read> nodeBlockList: array<u32>;
@group(0) @binding(5) var<storage, read> nodeCount: array<u32>;`
        : "";
    const pageDecls = pagedGrid
        ? `
@group(0) @binding(6) var<storage, read> pageMap: array<u32>;
@group(0) @binding(7) var<storage, read> pageState: array<u32>;
${PAGE_HELPERS_WGSL}`
        : "";
    const invocation = activeBlocks
        ? `
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>
) {
    let slot = wid.x + wid.y * ng.x;
    if (slot >= nodeCount[0]) { return; }
    let b0 = vec3<i32>(blockCoordOf(nodeBlockList[slot], p)) * TILE_I;
    let lc = vec3<i32>(i32(lid / (TILE_U * TILE_U)), i32((lid / TILE_U) % TILE_U), i32(lid % TILE_U));
    let node = b0 + lc;
    if (!inGrid(node, p)) { return; }
    let i = ${pagedGrid ? "i32(pageCellIndex(node, p))" : "index1D(node, p)"};
    let x = node.x;
    let y = node.y;
    let z = node.z;`
        : `
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (i >= arrayLength(&cells)) { return; }
    let dimZ = i32(p.dim.z);
    let dimYZ = i32(p.dim.y) * dimZ;
    let x = i32(i) / dimYZ;
    let y = (i32(i) / dimZ) % i32(p.dim.y);
    let z = i32(i) % dimZ;`;
    return /* wgsl */ `
${COMMON_WGSL}
${activeBlocks ? BLOCK_WGSL : ""}
${decls}
@group(0) @binding(0) var<storage, read_write> cells: array<vec4<i32>>;
@group(0) @binding(1) var<uniform> p: Params;
${activeDecls}
${pageDecls}

@compute @workgroup_size(${WORKGROUP_SIZE})
${invocation}
    ${pagedGrid ? "if (pageState[2] != 0u) { return; }" : ""}
    let c = cells[i];
    if (c.w <= 0) { return; }
    let invMass = 1.0 / dec(c.w);
    var v = vec3<f32>(dec(c.x), dec(c.y), dec(c.z)) * invMass;

    v.y -= p.sim0.y * p.sim0.x; // gravity * dt

    // Domain walls: one-sided FREE-SLIP in the 2-cell border — remove only the
    // into-wall (outward) velocity, keeping the pressure-driven push-BACK, so fluid
    // pressed against the grid bounds decompresses instead of gluing into a stuck sheet.
    // (A plain v = 0 kills the push-back too, which is the flat-wall gluing artefact.)
    if (x < 2) { v.x = max(v.x, 0.0); } else if (x > i32(p.dim.x) - 3) { v.x = min(v.x, 0.0); }
    if (y < 2) { v.y = max(v.y, 0.0); } else if (y > i32(p.dim.y) - 3) { v.y = min(v.y, 0.0); }
    if (z < 2) { v.z = max(v.z, 0.0); } else if (z > i32(p.dim.z) - 3) { v.z = min(v.z, 0.0); }
${wall}
    cells[i] = vec4<i32>(enc(v.x), enc(v.y), enc(v.z), c.w);
}`;
}

// SDF confinement resolve (position). Penetrating particles (sceneSdf < 0) are pushed
// out along the SDF gradient: a moving boundary (nonzero -∂sceneSdf/∂t, e.g. the paddle)
// hard-pushes and reflects relative to its own motion so the sweep stirs the fluid. A
// static wall's velocity: for a CLOSED container it was already applied at the grid
// separating wall, so G2P only nudges position (half depth, avoids gluing on flat walls);
// a per-particle container has no grid wall, so G2P does the full push + restitution
// reflect per-particle. The unified SDF confines interior + exterior/ground alike, so
// drained fluid is caught by the SAME field.
function g2pConfineSdf(scene: SceneSdfSpec): string {
    const closed = scene.gridConfine !== false;
    const staticResolve = closed ? "np += gn * (-d) * 0.5;" : "np += gn * (-d);\n                    vel = reflectVel(vel, -gn, p.misc2.x);";
    return /* wgsl */ `
    let d = sceneSdf(np, 0.0);
    if (d < 0.0) {
        let gn = sceneNormal(np, 0.0);
        let vN = -(sceneSdf(np, 0.002) - sceneSdf(np, 0.0)) / 0.002; // boundary normal speed (along gn)
        if (abs(vN) > 0.05) {
            np += gn * (-d);
            let bvel = vN * gn;
            vel = bvel + reflectVel(vel - bvel, -gn, p.misc2.x);
        } else {
            ${staticResolve}
        }
    }`;
}

// The per-demo sceneSdf is always injected (setSceneSdf) before the first step, so
// the G2P pass always uses the SDF confinement path. External forces run as their
// OWN dedicated compute pass (mpm-force) before the p2g transfer, so G2P stays
// force-free (a single pipeline cache-keyed on the scene only).
function buildG2pWgsl(scene: SceneSdfSpec, pagedGrid: boolean): string {
    // Baked SDF grid (optional): binding 4 is free here (0=particles, 1=cells, 2=p,
    // 3=sceneSdfParams). Injected BEFORE scene.sdf so `sceneSdf` can call sampleSdfGrid.
    const gridInject = scene.sdfGrid ? `\n@group(0) @binding(4) var<storage, read> sceneSdfGrid: array<f32>;\n${SCENE_SDF_GRID_WGSL}` : "";
    const decls = `${scene.struct}\n@group(0) @binding(3) var<uniform> sceneSdfParams: SceneSdfParams;${gridInject}\n${scene.sdf}\n${SCENE_NORMAL_WGSL}`;
    const pageDecls = pagedGrid
        ? `
@group(0) @binding(5) var<storage, read> pageMap: array<u32>;
@group(0) @binding(6) var<storage, read> pageState: array<u32>;
${BLOCK_WGSL}
${PAGE_HELPERS_WGSL}`
        : "";
    const confine = g2pConfineSdf(scene);
    return /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${WEIGHTS_WGSL}
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<vec4<i32>>;
@group(0) @binding(2) var<uniform> p: Params;
${decls}
${pageDecls}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.counts.x) { return; }
    if (i >= p.counts.z) { return; } // warm-up: skip dormant particles
    ${pagedGrid ? "if (pageState[2] != 0u) { return; }" : ""}
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
        let idx = ${pagedGrid ? "i32(pageCellIndex(node, p))" : "index1D(node, p)"};
        let cv = cells[idx];   // single 16-byte load (vx, vy, vz, mass)
        let wv = vec3<f32>(dec(cv.x), dec(cv.y), dec(cv.z)) * weight;
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
    let groundY = p.capsuleB.w;
${confine}

    // Keep inside the grid domain (2-cell margin).
    let lo = p.origin.xyz + dx * 2.0;
    let hi = p.origin.xyz + (p.dim.xyz - 3.0) * dx;
    np = clamp(np, lo, hi);

    // Near-ground anti-ripple: damp only the VERTICAL velocity within a thin
    // layer above the floor — this bleeds the up/down oscillation that would
    // otherwise launch floor ripples, while leaving the horizontal flow (and the
    // affine field) intact so the fluid keeps slipping and spreading. (Damping
    // the affine field C here as well used to keep the bottom layer sluggish and
    // made landing fluid mound up.) sim1.w = strength, dim.w = layer height.
    let gt = clamp((np.y - groundY) / max(p.dim.w, 1e-3), 0.0, 1.0);
    let gd = mix(p.sim1.w, 1.0, gt);
    vel.y *= gd;

    particles[i].position = np;
    particles[i].v = vel;
    particles[i].C = C;
}`;
}

// Dedicated external-force compute pass, built from an injected ForceFieldSpec.
// Mirrors the emit pass: dispatched once at the start of step() ONLY while a force
// is active, BEFORE the p2g transfer, so the app-supplied velocity delta flows
// through the grid this step and the G2P pass stays force-free. Reads the particle
// position + velocity, adds the delta, and writes only the velocity field back
// (position + affine field C untouched). count + sub-step dt come from Params.
function buildForceWgsl(force: ForceFieldSpec): string {
    return /* wgsl */ `
${PARTICLE_STRUCT}
struct Params {
    origin: vec4<f32>,
    dim: vec4<f32>,
    capsuleA: vec4<f32>,
    capsuleB: vec4<f32>,
    sim0: vec4<f32>,
    sim1: vec4<f32>,
    counts: vec4<u32>,
};
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> p: Params;
${force.struct}
@group(0) @binding(2) var<uniform> forceFieldParams: ForceFieldParams;
${force.wgsl}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.counts.x) { return; }
    if (i >= p.counts.z) { return; } // warm-up: dormant particles feel no external force
    var v = particles[i].v;
    v += externalForce(particles[i].position, v, p.sim0.x);
    particles[i].v = v;
}`;
}

const COPY_WGSL = /* wgsl */ `
${PARTICLE_STRUCT}
struct Params2 { count: u32, debugScale: f32, live: u32, _b: f32, };
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> renderPos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dbg: array<f32>;
@group(0) @binding(3) var<uniform> pc: Params2;
@group(0) @binding(4) var<storage, read_write> renderVel: array<vec4<f32>>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.count) { return; }
    if (i >= pc.live) {
        // Dormant (not-yet-released) particle during the start warm-up: park it far
        // off-screen so it stays invisible until it is activated (see warmupFrames).
        renderPos[i] = vec4<f32>(0.0, -1.0e5, 0.0, 1.0);
        dbg[i] = 0.0;
        renderVel[i] = vec4<f32>(0.0);
        return;
    }
    renderPos[i] = vec4<f32>(particles[i].position, 1.0);
    dbg[i] = length(particles[i].v);
    renderVel[i] = vec4<f32>(particles[i].v, 0.0);
}`;

const FLOW_WGSL = /* wgsl */ `
${PARTICLE_STRUCT}
${FLUID_FLOW_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> flow: FluidFlowData;
@group(0) @binding(2) var<storage, read_write> flowCounters: array<atomic<u32>>;
${FLUID_FLOW_RUNTIME_WGSL}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= flow.header.z) { return; }
    let launch = fluidTryRelaunch(particles[i].position, i);
    if (launch.launched != 0u) {
        particles[i].position = launch.position;
        particles[i].v = launch.velocity;
        particles[i].C = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    }
}`;

// ── Foam: diffuse particles (spray / foam / bubbles), grid-derived ──────────────
// Compute passes appended ONCE per frame at the end of step() (after the substep loop,
// before the copy) when foam is enabled. Generation follows Ihmsen 2012 (`FOAM_EMIT_WGSL`):
// the three generation potentials, derived from the MLS GRID quantities the last substep
// produced (see the tuning notes below). The pool, classification and advection
// (mpm-foam-update) then run method-agnostically.
// There is no neighbour list on this backend, so the Ihmsen potentials are derived from:
//   • the grid mass (a local density ρ) + velocity (v = momentum / mass), and
//   • the particle v and the APIC affine C = ∇v (the local velocity gradient).
// The shared Foam UBO + Diffuse slot struct + Φ / hash helpers come from
// `FOAM_COMMON_WGSL`; gravity / restDensity / dx / bounds are read from the Params
// UBO, and the FRAME dt (not the substep dt) is passed through Params.misc2.y.
//
// Tuning notes (Ihmsen path — deviations from the neighbour-based PBF generator):
//   • Trapped air uses ‖strain-rate‖ = ‖0.5(C + Cᵀ)‖_F (the symmetric part of the
//     velocity gradient) as the turbulence proxy — it excludes pure rotation and fires
//     on the shear/compression the paddle entrains, staying ~0 in the calm interior.
//   • Wave crest has no neighbour-normal curvature; it is approximated by a surface
//     gate (low ρ), an outward-motion gate (v̂·n ≥ 0.6 with n = −∇ρ/‖∇ρ‖) and the
//     outward speed as the crest strength.
//   • Classification is by grid density ρ (fraction of restDensity) instead of a
//     neighbour count: low ρ → spray, high ρ → bubble, mid → foam.
// K_STRAIN / WC_SCALE and the RHO_* fractions below are the tuned live constants.
function buildFoamEmitWgsl(activeParticles: boolean, pagedGrid: boolean): string {
    const headDecl = activeParticles
        ? "@group(0) @binding(5) var<storage, read_write> activeState: array<atomic<u32>>;"
        : "@group(0) @binding(5) var<storage, read_write> head: array<atomic<u32>>;";
    const activeDecl = activeParticles
        ? `
fn activeStride(cap: u32) -> u32 { return ((cap + 63u) / 64u) * 64u; }
fn activeListBase(side: u32, cap: u32) -> u32 { return 64u + side * activeStride(cap); }
fn activeFlagBase(cap: u32) -> u32 { return 64u + 2u * activeStride(cap); }`
        : "";
    const activate = activeParticles
        ? `
        if (atomicExchange(&activeState[activeFlagBase(cap) + idx], 1u) == 0u) {
            let side = atomicLoad(&activeState[3]);
            let dst = atomicAdd(&activeState[1u + side], 1u);
            atomicStore(&activeState[activeListBase(side, cap) + dst], idx);
        }`
        : "";
    const pageDecls = pagedGrid
        ? `
@group(0) @binding(6) var<storage, read> pageMap: array<u32>;
@group(0) @binding(7) var<storage, read> pageState: array<u32>;
${BLOCK_WGSL}
${PAGE_HELPERS_WGSL}`
        : "";
    const cellIndex = pagedGrid ? "pageCellIndex(node, p)" : "u32(index1D(node, p))";
    return /* wgsl */ `
${COMMON_WGSL}
${PARTICLE_STRUCT}
${WEIGHTS_WGSL}
${FOAM_COMMON_WGSL}
struct Cell { vx: i32, vy: i32, vz: i32, mass: i32, };
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<uniform> foam: Foam;
@group(0) @binding(4) var<storage, read_write> diffuse: array<Diffuse>;
${headDecl}
${activeDecl}
${pageDecls}

// Scales ‖strain-rate‖ (1/s) into the trapped-air potential; tuned so the paddle wake
// reaches the τ_ta band while the calm interior stays below it.
const K_STRAIN: f32 = 1.6;
// Scales the wave-crest argument (surfaceness · outward-speed) into the τ_wc band.
const WC_SCALE: f32 = 1.2;
// Surfaceness ramp: 1 below RHO_SURF_LO·restDensity (free surface), 0 above RHO_SURF_HI·restDensity (interior).
const RHO_SURF_LO: f32 = 0.15;
const RHO_SURF_HI: f32 = 0.85;

fn frob(m: mat3x3<f32>) -> f32 { return sqrt(dot(m[0], m[0]) + dot(m[1], m[1]) + dot(m[2], m[2])); }

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.counts.x) { return; }
    ${pagedGrid ? "if (pageState[2] != 0u) { return; }" : ""}
    let pos = particles[i].position;
    let vi = particles[i].v;
    let speed = length(vi);
    if (speed < 1e-4) { return; }
    let vhat = vi / speed;
    let C = particles[i].C;
    let dx = p.origin.w;
    let Dinv = 4.0 / (dx * dx);
    let restD = max(p.sim0.z, 1e-3);
    let frameDt = p.misc2.y;

    // Gather the grid mass field around the particle → local density ρ and its gradient
    // ∇ρ (the MLS/APIC least-squares reconstruction: ∇ρ = Dinv·Σ w·(x_node − x)·ρ_node).
    let base = cellOf(pos, p);
    let w = weightsOf(pos, p);
    var rho = 0.0;
    var gradRho = vec3<f32>(0.0);
    for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
    for (var gz = 0; gz < 3; gz++) {
        let weight = w[gx].x * w[gy].y * w[gz].z;
        let node = base + vec3<i32>(gx - 1, gy - 1, gz - 1);
        if (!inGrid(node, p)) { continue; }
        let m = dec(cells[${cellIndex}].mass);
        let cellDist = (p.origin.xyz + (vec3<f32>(node) + 0.5) * dx) - pos;
        rho += m * weight;
        gradRho += (m * weight) * cellDist;
    }}}
    gradRho *= Dinv;

    // Trapped air — strain-rate Frobenius norm (symmetric part of the velocity gradient
    // C = ∇v; excludes pure rotation, captures shear + compression entraining air).
    let strain = 0.5 * (C + transpose(C));
    let ita = phi(K_STRAIN * frob(strain), foam.tauTaMin, foam.tauTaMax);
    // Kinetic energy modulator.
    let ek = 0.5 * speed * speed;
    let ik = phi(ek, foam.tauKMin, foam.tauKMax);
    // Wave crest — outward normal n = −∇ρ/‖∇ρ‖; surface gate + moving-outward gate ×
    // outward speed as the crest strength (no neighbour-normal curvature on this backend).
    var n = vec3<f32>(0.0, 1.0, 0.0);
    let gl = length(gradRho);
    if (gl > 1e-6) { n = -gradRho / gl; }
    let surfaceness = 1.0 - smoothstep(RHO_SURF_LO * restD, RHO_SURF_HI * restD, rho);
    let vn = dot(vi, n);
    let dvn = select(0.0, 1.0, (vn / speed) >= 0.6);
    let crest = max(0.0, vn);
    let iwc = phi(WC_SCALE * surfaceness * dvn * crest, foam.tauWcMin, foam.tauWcMax);

    let ndf = ik * (foam.kTa * ita + foam.kWc * iwc) * frameDt;
    // Stochastic rounding: nd must have EXPECTED value ndf (Ihmsen 2012). Rounding to nearest
    // instead turned the generation rates into a step function — frameDt is clamped to 1/60, so
    // every particle shares it, and ndf crossed 0.5 for the whole population at the same rate
    // value (kTa = 30 at 60fps). The control jumped from "no foam at all" to "one per particle
    // per frame" in a single slider step. Carrying the fraction as a spawn PROBABILITY spreads
    // that crossing across the population, so the rate responds continuously.
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
    let dtv = length(frameDt * vi);
    for (var k = 0; k < nd; k = k + 1) {
        let seed = (i * 2654435761u) ^ (foam.frameSeed * 40503u) ^ (u32(k) * 2246822519u);
        let xr = fRnd(seed);
        let xt = fRnd(seed * 3u + 1u);
        let xh = fRnd(seed * 7u + 5u);
        let rr = foam.rv * sqrt(xr);
        let th = 6.28318530718 * xt;
        let off = e1 * (rr * cos(th)) + e2 * (rr * sin(th));
        let xd = pos + off + vhat * (xh * dtv);
        let vd = off + vi;
        let idx = atomicAdd(&${activeParticles ? "activeState" : "head"}[0], 1u) % cap;
        diffuse[idx].p = vec4<f32>(xd, life);
        diffuse[idx].v = vec4<f32>(vd, 0.0);
${activate}
    }
}`;
}

// Pass 2 — classify + advect + dissolve over the whole diffuse pool. The local fluid
// velocity ṽ_f is the grid velocity gathered at the diffuse position (trivial in MLS:
// a weighted 3×3×3 gather of the post-update cell velocity), and the local density ρ is
// the grid mass gather. ρ (as a fraction of restDensity) classifies each live particle
// — low ρ → spray, high ρ → bubble, else foam — then each class advects like PBF.
function buildFoamUpdateWgsl(activeParticles: boolean, pagedGrid: boolean): string {
    const activeDecl = activeParticles
        ? `
@group(0) @binding(4) var<storage, read_write> activeState: array<atomic<u32>>;
fn activeStride(cap: u32) -> u32 { return ((cap + 63u) / 64u) * 64u; }
fn activeListBase(side: u32, cap: u32) -> u32 { return 64u + side * activeStride(cap); }
fn activeFlagBase(cap: u32) -> u32 { return 64u + 2u * activeStride(cap); }`
        : "";
    const slotLookup = activeParticles
        ? `
    let cap = arrayLength(&diffuse);
    let side = atomicLoad(&activeState[3]);
    let slot = gid.x + gid.y * ${MAX_WORKGROUPS * WORKGROUP_SIZE}u;
    if (slot >= atomicLoad(&activeState[1u + side])) { return; }
    let i = atomicLoad(&activeState[activeListBase(side, cap) + slot]);`
        : `
    let i = gid.x + gid.y * ${MAX_WORKGROUPS * WORKGROUP_SIZE}u;
    if (i >= arrayLength(&diffuse)) { return; }`;
    const deactivate = activeParticles ? "atomicStore(&activeState[activeFlagBase(cap) + i], 0u);" : "";
    const keepActive = activeParticles
        ? `
    let nextSide = 1u - side;
    let dst = atomicAdd(&activeState[1u + nextSide], 1u);
    atomicStore(&activeState[activeListBase(nextSide, cap) + dst], i);`
        : "";
    const pageDecls = pagedGrid
        ? `
@group(0) @binding(6) var<storage, read> pageMap: array<u32>;
@group(0) @binding(7) var<storage, read> pageState: array<u32>;
${BLOCK_WGSL}
${PAGE_HELPERS_WGSL}`
        : "";
    const cellIndex = pagedGrid ? "pageCellIndex(node, p)" : "u32(index1D(node, p))";
    return /* wgsl */ `
${COMMON_WGSL}
${WEIGHTS_WGSL}
${FOAM_COMMON_WGSL}
struct Cell { vx: i32, vy: i32, vz: i32, mass: i32, };
@group(0) @binding(0) var<storage, read> cells: array<Cell>;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<uniform> foam: Foam;
@group(0) @binding(3) var<storage, read_write> diffuse: array<Diffuse>;
${activeDecl}
${pageDecls}

// Classification thresholds as a fraction of restDensity (the interior packs at ~restD).
const RHO_SPRAY: f32 = 0.35;  // ρ below this → spray (near-empty, flying droplet)
const RHO_BUBBLE: f32 = 0.9;  // ρ above this → bubble (submerged, rises)

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Pool slots can exceed one dispatch dimension, in which case dispatch() spills into y
    // with an x extent of exactly MAX_WORKGROUPS groups — fold that back into a flat index.
    // gid.y is 0 whenever the dispatch fits in x, so this is a no-op for small pools.
${slotLookup}
    let p0 = diffuse[i].p;
    if (p0.w <= 0.0) { ${deactivate} return; }
    ${pagedGrid ? `if (pageState[2] != 0u) { ${keepActive} return; }` : ""}
    let pp = p0.xyz;
    let dx = p.origin.w;
    let lo = p.origin.xyz;
    let hi = p.origin.xyz + p.dim.xyz * dx;
    if (any(pp < lo) || any(pp > hi)) {
        diffuse[i].p = vec4<f32>(pp, 0.0);
        ${deactivate}
        return;
    }
    var v = diffuse[i].v.xyz;

    // Gather the post-update grid at the diffuse position → local fluid velocity ṽ_f
    // (cells hold v = momentum/mass after updateGrid) + local density ρ (grid mass).
    let base = cellOf(pp, p);
    let w = weightsOf(pp, p);
    var vf = vec3<f32>(0.0);
    var rho = 0.0;
    for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
    for (var gz = 0; gz < 3; gz++) {
        let weight = w[gx].x * w[gy].y * w[gz].z;
        let node = base + vec3<i32>(gx - 1, gy - 1, gz - 1);
        if (!inGrid(node, p)) { continue; }
        let idx = ${cellIndex};
        vf += vec3<f32>(dec(cells[idx].vx), dec(cells[idx].vy), dec(cells[idx].vz)) * weight;
        rho += dec(cells[idx].mass) * weight;
    }}}

    let g = p.sim0.y;
    let dt = p.misc2.y; // frame dt
    let restD = max(p.sim0.z, 1e-3);
    var kind = 1u;
    if (rho < RHO_SPRAY * restD) { kind = 0u; } else if (rho > RHO_BUBBLE * restD) { kind = 2u; }

    var np = pp;
    var life = p0.w;
    if (kind == 0u) {
        // Spray: ballistic (Euler-Cromer).
        v.y -= g * dt;
        np = pp + dt * v;
    } else if (kind == 2u) {
        // Bubble: buoyancy up + drag toward the local grid flow.
        v.y += dt * foam.kb * g;
        v += foam.kd * (vf - v);
        np = pp + dt * v;
    } else {
        // Foam: rides the surface at the grid fluid velocity; lifetime decays.
        v = vf;
        np = pp + dt * vf;
        life = p0.w - dt;
    }
    if (life <= 0.0) {
        diffuse[i].p = vec4<f32>(np, 0.0);
        ${deactivate}
        return;
    }
    diffuse[i].p = vec4<f32>(np, life);
    diffuse[i].v = vec4<f32>(v, f32(kind));
${keepActive}
}`;
}

export interface MlsMpmOptions extends FluidSimBaseOptions {
    /** Simulation box min corner — the MLS grid domain AABB. Default [-20, 0, -20]. */
    boundsMin?: [number, number, number];
    /** Simulation box max corner — the MLS grid domain AABB. Default [20, 15, 20]. */
    boundsMax?: [number, number, number];
    /** Exact grid cell count along X/Y/Z. Derived from bounds when omitted. */
    gridDim?: [number, number, number];
    /** Ground plane height; particles are floored at y = groundY. Default boundsMin.y. */
    groundY?: number;
    /** Capsule tank boundary: centre of the bottom hemisphere. Default null. */
    capsuleA?: [number, number, number];
    /** Capsule tank boundary: centre of the top hemisphere. Default null. */
    capsuleB?: [number, number, number];
    /** Capsule tank radius. Default 0. */
    capsuleRadius?: number;
    /** Grid cell size in world units (smaller = finer fluid, more cells). Default 0.25. */
    dx?: number;
    /** Equation-of-state stiffness. Default 50. */
    stiffness?: number;
    /** Rest density in particles-per-cell. Default derived from spawn packing. */
    restDensity?: number;
    /** Dynamic (viscous) stress coefficient. Default 0.1. */
    viscosity?: number;
    /** Sub-steps per frame: the frame's dt is split into this many MLS-MPM steps.
     *  More substeps = more stable (smaller dt per step) but more compute. Splash
     *  uses 1 (its grid-unit scaling keeps a single big step stable); this
     *  world-unit port needs a few. Default 3. */
    substeps?: number;
    /** Safety cap on a single sub-step's dt (seconds); the per-frame dt/substeps
     *  is clamped to this so a hitch can't blow the integration up. Default 1/120. */
    maxSubDt?: number;
    /** Per-substep velocity multiplier (below 1 bleeds bulk kinetic energy so the
     *  fluid settles to rest). Default 0.98. */
    damping?: number;
    /** Per-substep APIC affine (C) multiplier (below 1 blends toward dissipative PIC,
     *  killing residual swirl). Default 0.95. */
    affineDamping?: number;
    /** Extra per-substep velocity/affine damping applied within `groundDampHeight`
     *  of the ground, so the thin floor pool settles without rippling. Default 0.9. */
    groundDamp?: number;
    /** Height (world units) of the near-ground damping layer. Default 1.2. */
    groundDampHeight?: number;
    /** Collision restitution for the capsule wall + ground (0 = free-slip / no
     *  bounce, 1 = elastic mirror). Default 0.3. */
    restitution?: number;
    /** Sparse execution mode. Compacts occupied particle blocks and
     *  dispatches P2G plus grid clear/update only over their touched node-block halo.
     *  The dense cell buffer is retained, so this reduces GPU time rather than memory.
     *  Default false. */
    activeBlocks?: boolean;
    /** Append a particle block to the active list on its histogram count's 0→1
     *  transition, eliminating the separate block-compaction pass. Requires
     *  `activeBlocks`; ignored otherwise. Default false. */
    fusedBlockDiscovery?: boolean;
    /** Store grid nodes in a bounded pool of 4³-cell pages instead of the dense
     *  domain-sized grid. Implies `activeBlocks`. Default false. */
    pagedGrid?: boolean;
    /** Maximum number of live grid pages. Required when `pagedGrid` is enabled.
     *  If the active node-block halo exceeds this cap, the solver freezes before
     *  integrating particles and reports the required page count. */
    pagedGridMaxPages?: number;
    /** Called asynchronously after a paged-grid overflow is read back. */
    onPagedGridOverflow?: (requiredPages: number, capacity: number) => void;
    /** Explicit per-particle seed positions as flat world-space xyz triples
     *  (`[x0,y0,z0, x1,y1,z1, …]`). When present, `seed()`/`reset()` places each
     *  particle `i (< count)` at `initialPositions[3i..3i+2]` with zero velocity and
     *  a zero affine field — instead of drawing a random point in the spawn box. Size
     *  `count` to `initialPositions.length / 3` so every particle is real. Absent (the
     *  default) restores the random-in-spawn-box behaviour. Used to fill a mesh with a
     *  volume-sampled particle set (see fluid/volume-sampling). */
    initialPositions?: Float32Array;
}

export function createMlsMpmSim(engine: EngineContext, options: MlsMpmOptions = {}): FluidSim {
    const device = engine._device;
    const count = options.count ?? 60000;
    const particleRadius = options.particleRadius ?? 0.09;
    const spawnMin: [number, number, number] = options.spawnMin ?? [-2, 4, -2];
    const spawnMax: [number, number, number] = options.spawnMax ?? [2, 12, 2];
    let spawnAccept: ((x: number, y: number, z: number) => boolean) | null = null;
    // Start-of-sim WARM-UP ramp: releasing all particles at once spikes the density
    // (on an open shelf the stiff MLS pressure then flings them into spray). Instead,
    // only `liveCount` particles are live each frame, ramping up over `warmupFrames`
    // frames; dormant particles deposit no mass (skipped in force/P2G/G2P) and are
    // parked off-screen by the copy pass, so the body fills in gradually and stays
    // grouped. 0 (default) = release everything immediately (original behaviour).
    let warmupFrames = Math.max(0, Math.floor(options.warmupFrames ?? 0));
    let warmupStep = warmupFrames > 0 ? Math.max(1, Math.ceil(count / warmupFrames)) : count;
    let liveCount = count;
    let initialTargetCount = count;
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
    const substeps = options.substeps ?? 3;
    let substepsMut = substeps;
    // Mutable so the host can trade stability against cost at runtime (setParam "maxSubDtMs"):
    // it is the cap that decides whether `step()` has to run MORE sub-steps than requested.
    let maxSubDt = options.maxSubDt ?? 1 / 120;
    const damping = options.damping ?? 0.98;
    const affineDamping = options.affineDamping ?? 0.95;
    const groundDamp = options.groundDamp ?? 0.9;
    const groundDampHeight = options.groundDampHeight ?? 1.2;
    const restitution = options.restitution ?? 0.3;
    const pagedGrid = options.pagedGrid ?? false;
    const activeBlocks = pagedGrid || (options.activeBlocks ?? false);
    const fusedBlockDiscovery = activeBlocks && (options.fusedBlockDiscovery ?? false);
    // Optional explicit per-particle seed (flat world-space xyz). When set, seed()
    // reads position i from here instead of the random spawn draw (see the interface).
    const initialPositions = options.initialPositions ?? null;

    const gridDim: [number, number, number] = options.gridDim
        ? (options.gridDim.map((value) => Math.max(4, Math.round(value))) as [number, number, number])
        : [
              Math.max(4, Math.ceil((boundsMax[0] - boundsMin[0]) / dx)),
              Math.max(4, Math.ceil((boundsMax[1] - boundsMin[1]) / dx)),
              Math.max(4, Math.ceil((boundsMax[2] - boundsMin[2]) / dx)),
          ];
    const numCells = gridDim[0] * gridDim[1] * gridDim[2];

    // Block grid for the tiled P2G counting sort: blockDim = ceil(gridDim / TILE).
    const blockDim: [number, number, number] = [Math.ceil(gridDim[0] / TILE), Math.ceil(gridDim[1] / TILE), Math.ceil(gridDim[2] / TILE)];
    const numBlocks = blockDim[0] * blockDim[1] * blockDim[2];
    if (pagedGrid && (!Number.isFinite(options.pagedGridMaxPages) || (options.pagedGridMaxPages ?? 0) <= 0)) {
        throw new Error("[MLS-MPM] pagedGrid requires a positive pagedGridMaxPages capacity.");
    }
    const pagedGridMaxPages = pagedGrid ? Math.min(numBlocks, Math.floor(options.pagedGridMaxPages!)) : 0;
    const maxPagedGridPages = Math.floor(device.limits.maxStorageBufferBindingSize / (TILE * TILE * TILE * 16)) - 1;
    if (pagedGrid && pagedGridMaxPages > maxPagedGridPages) {
        throw new Error(`[MLS-MPM] pagedGridMaxPages ${pagedGridMaxPages} exceeds this device's ${maxPagedGridPages}-page storage-buffer limit.`);
    }
    const scanChunks = Math.ceil(numBlocks / SCAN_WG); // per-chunk totals for the multi-level scan

    // Rest density (particles per cell): from the spawn packing if not given.
    const spawnVolCells = Math.max(1, ((spawnMax[0] - spawnMin[0]) * (spawnMax[1] - spawnMin[1]) * (spawnMax[2] - spawnMin[2])) / (dx * dx * dx));
    const restDensity = options.restDensity ?? Math.max(2, count / spawnVolCells);

    // ── Buffers ──────────────────────────────────────────────────────
    const PARTICLE_STRIDE = 80;
    const particleBuffer = device.createBuffer({ label: "mpm-particles", size: count * PARTICLE_STRIDE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const cellBuffer = device.createBuffer({
        label: pagedGrid ? "mpm-paged-cells" : "mpm-cells",
        size: pagedGrid ? (pagedGridMaxPages + 1) * TILE * TILE * TILE * 16 : numCells * 16,
        usage: GPUBufferUsage.STORAGE | (pagedGrid ? GPUBufferUsage.COPY_DST : 0),
    });
    const positionBuffer = device.createBuffer({ label: "mpm-render-pos", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const velocityBuffer = device.createBuffer({ label: "mpm-render-vel", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const debugBuffer = device.createBuffer({ label: "mpm-debug", size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const paramsBuffer = device.createBuffer({ label: "mpm-params", size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const copyParamsBuffer = device.createBuffer({ label: "mpm-copy-params", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const flowState = createFluidFlowState(device, count, particleRadius);
    let flowSeedsInitialParticles = true;

    // Block counting-sort buffers (per-substep). blockCount/blockCursor are zeroed each
    // substep via clearBuffer (hence COPY_DST); blockStart/partialSums/sortedIdx are fully
    // overwritten each substep. Sized from numBlocks/count so createSims rebuilds them on a
    // domainScale change alongside every other buffer.
    const blockCountBuffer = device.createBuffer({ label: "mpm-block-count", size: numBlocks * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const blockStartBuffer = device.createBuffer({ label: "mpm-block-start", size: numBlocks * 4, usage: GPUBufferUsage.STORAGE });
    const blockCursorBuffer = device.createBuffer({ label: "mpm-block-cursor", size: numBlocks * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const partialSumsBuffer = device.createBuffer({ label: "mpm-partial-sums", size: scanChunks * 4, usage: GPUBufferUsage.STORAGE });
    const sortedIdxBuffer = device.createBuffer({ label: "mpm-sorted-idx", size: count * 4, usage: GPUBufferUsage.STORAGE });
    const activeBlockListBuffer = activeBlocks ? device.createBuffer({ label: "mpm-active-block-list", size: numBlocks * 4, usage: GPUBufferUsage.STORAGE }) : null;
    const activeBlockCountBuffer = activeBlocks
        ? device.createBuffer({
              label: "mpm-active-block-state",
              size: pagedGrid ? 12 : 4,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | (pagedGrid ? GPUBufferUsage.COPY_SRC : 0),
          })
        : null;
    const activeBlockIndirectBuffer = activeBlocks
        ? device.createBuffer({ label: "mpm-active-block-indirect", size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT })
        : null;
    const nodeBlockFlagsBuffer = activeBlocks
        ? device.createBuffer({ label: "mpm-node-block-flags", size: numBlocks * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
        : null;
    const nodeBlockListBuffer = activeBlocks ? device.createBuffer({ label: "mpm-node-block-list", size: numBlocks * 4, usage: GPUBufferUsage.STORAGE }) : null;
    const nodeBlockCountBuffer = activeBlocks ? device.createBuffer({ label: "mpm-node-block-count", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }) : null;
    const nodeBlockIndirectBuffer = activeBlocks
        ? device.createBuffer({ label: "mpm-node-block-indirect", size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT })
        : null;
    const pageMapBuffer = pagedGrid ? device.createBuffer({ label: "mpm-page-map", size: numBlocks * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }) : null;
    const pageStatusStagingBuffers = pagedGrid
        ? [0, 1].map((i) =>
              device.createBuffer({
                  label: `mpm-page-status-readback-${i}`,
                  size: 8,
                  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
              })
          )
        : [];

    const paramsData = new ArrayBuffer(PARAMS_BYTES);
    const pf = new Float32Array(paramsData);
    const pu = new Uint32Array(paramsData);
    // Copy-pass params (Params2): [0]=count, [1]=debugScale(unused), [2]=active
    // (warm-up live count, gates the render copy), [3]=pad.
    const copyData = new ArrayBuffer(16);
    const copyU32 = new Uint32Array(copyData);
    copyU32[0] = count;
    copyU32[2] = count;
    pf[0] = boundsMin[0];
    pf[1] = boundsMin[1];
    pf[2] = boundsMin[2];
    pf[3] = dx;
    pf[4] = gridDim[0];
    pf[5] = gridDim[1];
    pf[6] = gridDim[2];
    pf[7] = groundDampHeight; // dim.w
    pf[8] = capsuleA ? capsuleA[0] : 0;
    pf[9] = capsuleA ? capsuleA[1] : 0;
    pf[10] = capsuleA ? capsuleA[2] : 0;
    pf[11] = capsuleRadius;
    pf[12] = capsuleB ? capsuleB[0] : 0;
    pf[13] = capsuleB ? capsuleB[1] : 0;
    pf[14] = capsuleB ? capsuleB[2] : 0;
    pf[15] = groundY;
    pf[16] = maxSubDt; // updated per-frame in step() to frameDt / substeps (capped)
    pf[17] = gravity;
    pf[18] = restDensity;
    pf[19] = stiffness;
    pf[20] = viscosity;
    pf[21] = damping;
    pf[22] = affineDamping;
    pf[23] = groundDamp; // sim1.w
    pf[MISC2_BASE_F32] = restitution;
    pu[COUNTS_OFFSET_F32] = count;
    pu[COUNTS_OFFSET_F32 + 1] = capsuleA && capsuleB ? 1 : 0;

    device.queue.writeBuffer(copyParamsBuffer, 0, copyData);

    function seed(): void {
        resetFluidFlowState(flowState);
        const flowParticles = initialPositions || !flowSeedsInitialParticles ? null : createFluidInitialParticles(count, flowState.config, flowState.particleVolume);
        initialTargetCount = initialPositions ? count : (flowParticles?.activeCount ?? count);
        warmupStep = warmupFrames > 0 ? Math.max(1, Math.ceil(initialTargetCount / warmupFrames)) : initialTargetCount;
        // Reset the warm-up ramp: start with just the first initial batch live (or the
        // whole initial prefix when disabled). Inflow capacity remains dormant.
        liveCount = warmupFrames > 0 ? Math.min(initialTargetCount, warmupStep) : initialTargetCount;
        const buf = new ArrayBuffer(count * PARTICLE_STRIDE);
        const f = new Float32Array(buf);
        const rp = new Float32Array(count * 4);
        const flowPositions = flowParticles?.positions;
        const flowVelocities = flowParticles?.velocities;
        const renderVelocities = new Float32Array(count * 4);
        // Last position that passed `spawnAccept`, reused when a particle exhausts its retries.
        let lastOkX = 0;
        let lastOkY = 0;
        let lastOkZ = 0;
        let haveLastOk = false;
        for (let i = 0; i < count; i++) {
            const o = (i * PARTICLE_STRIDE) / 4; // float offset into the particle struct
            let x: number;
            let y: number;
            let z: number;
            if (initialPositions) {
                // Explicit per-particle seed (e.g. a volume-sampled mesh fill): read the
                // world position straight from the caller's flat xyz array. Velocity (o+4..6)
                // and the affine field C (o+8..19) stay zeroed, exactly as in the random path.
                x = initialPositions[i * 3]!;
                y = initialPositions[i * 3 + 1]!;
                z = initialPositions[i * 3 + 2]!;
            } else if (flowPositions && i < initialTargetCount) {
                x = flowPositions[i * 3]!;
                y = flowPositions[i * 3 + 1]!;
                z = flowPositions[i * 3 + 2]!;
            } else if (flowParticles) {
                x = 0;
                y = 0;
                z = 0;
            } else {
                x = spawnMin[0] + Math.random() * (spawnMax[0] - spawnMin[0]);
                y = spawnMin[1] + Math.random() * (spawnMax[1] - spawnMin[1]);
                z = spawnMin[2] + Math.random() * (spawnMax[2] - spawnMin[2]);
                if (spawnAccept) {
                    // Reject-sample so particles fit a non-box container shape: redraw
                    // uniformly in the box until accepted.
                    let ok = spawnAccept(x, y, z);
                    for (let tries = 0; !ok && tries < SPAWN_ACCEPT_TRIES; tries++) {
                        x = spawnMin[0] + Math.random() * (spawnMax[0] - spawnMin[0]);
                        y = spawnMin[1] + Math.random() * (spawnMax[1] - spawnMin[1]);
                        z = spawnMin[2] + Math.random() * (spawnMax[2] - spawnMin[2]);
                        ok = spawnAccept(x, y, z);
                    }
                    // On exhaustion reuse the last ACCEPTED point instead of keeping a rejected
                    // one. Keeping the rejection put particles anywhere in the bounding box —
                    // for a sparse container (two prisms on a rock summit) that is mostly open
                    // air and rock interior, and the collision SDF then shoves them out as a
                    // burst of spray. Duplicating a good point is invisible by comparison.
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
            f[o] = x;
            f[o + 1] = y;
            f[o + 2] = z;
            if (flowVelocities && i < initialTargetCount) {
                f[o + 4] = flowVelocities[i * 3]!;
                f[o + 5] = flowVelocities[i * 3 + 1]!;
                f[o + 6] = flowVelocities[i * 3 + 2]!;
                renderVelocities[i * 4] = f[o + 4]!;
                renderVelocities[i * 4 + 1] = f[o + 5]!;
                renderVelocities[i * 4 + 2] = f[o + 6]!;
            }
            // C (o+8..19) stays zeroed.
            const live = i < liveCount;
            rp[i * 4] = live ? x : 0;
            rp[i * 4 + 1] = live ? y : -1.0e5; // park dormant (warm-up) particles off-screen
            rp[i * 4 + 2] = live ? z : 0;
            rp[i * 4 + 3] = 1;
        }
        device.queue.writeBuffer(particleBuffer, 0, buf);
        device.queue.writeBuffer(debugBuffer, 0, new Float32Array(count));
        // Seed render positions so the first frame draws the spawn before any step.
        device.queue.writeBuffer(positionBuffer, 0, rp);
        device.queue.writeBuffer(velocityBuffer, 0, renderVelocities);
    }

    function activateInflowParticles(start: number, particles: NonNullable<ReturnType<typeof prepareFluidFlowFrame>["particles"]>): void {
        const data = new ArrayBuffer(particles.activeCount * PARTICLE_STRIDE);
        const values = new Float32Array(data);
        for (let i = 0; i < particles.activeCount; i++) {
            const source = i * 3;
            const target = (i * PARTICLE_STRIDE) / 4;
            values[target] = particles.positions[source]!;
            values[target + 1] = particles.positions[source + 1]!;
            values[target + 2] = particles.positions[source + 2]!;
            values[target + 4] = particles.velocities[source]!;
            values[target + 5] = particles.velocities[source + 1]!;
            values[target + 6] = particles.velocities[source + 2]!;
        }
        device.queue.writeBuffer(particleBuffer, start * PARTICLE_STRIDE, data);
    }
    seed();

    function pipeline(label: string, code: string): GPUComputePipeline {
        return device.createComputePipeline({ label, layout: "auto", compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" } });
    }
    const clearPipe = pipeline("mpm-clear", activeBlocks ? buildClearActiveBlocksWgsl(pagedGrid) : CLEAR_WGSL);
    // Block counting-sort + tiled P2G pipelines (replace the old global-atomic p2g-mass /
    // p2g-vel scatter with a shared-memory-staged transfer over sorted blocks).
    const histogramPipe = pipeline("mpm-histogram", buildHistogramWgsl(fusedBlockDiscovery));
    const scanLocalPipe = pipeline("mpm-scan-local", SCAN_LOCAL_WGSL);
    const scanPartialsPipe = pipeline("mpm-scan-partials", SCAN_PARTIALS_WGSL);
    const scanAddPipe = pipeline("mpm-scan-add", SCAN_ADD_WGSL);
    const scatterPipe = pipeline("mpm-scatter", SCATTER_WGSL);
    const p2gMassTiledPipe = pipeline("mpm-p2g-mass-tiled", buildP2gMassTiledWgsl(activeBlocks, pagedGrid));
    const p2gVelTiledPipe = pipeline("mpm-p2g-vel-tiled", buildP2gVelTiledWgsl(activeBlocks, pagedGrid));
    const compactActiveBlocksPipe = activeBlocks && !fusedBlockDiscovery ? pipeline("mpm-compact-active-blocks", COMPACT_ACTIVE_BLOCKS_WGSL) : null;
    const finalizeIndirectPipe = activeBlocks ? pipeline("mpm-finalize-indirect", FINALIZE_INDIRECT_WGSL) : null;
    const markActiveNodeBlocksPipe = activeBlocks ? pipeline("mpm-mark-active-node-blocks", MARK_ACTIVE_NODE_BLOCKS_WGSL) : null;
    const assignGridPagesPipe = pagedGrid ? pipeline("mpm-assign-grid-pages", ASSIGN_GRID_PAGES_WGSL) : null;
    // update-grid + g2p pipelines/bind-groups are built lazily in setSceneSdf (always
    // called before the first step); compiled variants are cached by source so
    // re-selecting a demo is instant.
    const updatePipeCache = new Map<string, GPUComputePipeline>();
    function getUpdatePipe(scene: SceneSdfSpec): GPUComputePipeline {
        const src = buildUpdateGridWgsl(scene, activeBlocks, pagedGrid);
        let pipe = updatePipeCache.get(src);
        if (!pipe) {
            pipe = pipeline("mpm-update", src);
            updatePipeCache.set(src, pipe);
        }
        return pipe;
    }
    let updatePipe: GPUComputePipeline | null = null;
    // g2p pipeline/bind-group are built lazily in setSceneSdf (scene is always
    // injected before the first step); compiled variants are cached by the scene
    // WGSL source so re-selecting a demo is instant. Force-free: any external force
    // runs as its own dedicated pass, so G2P is keyed on the scene alone.
    const g2pPipeCache = new Map<string, GPUComputePipeline>();
    function getG2pPipe(scene: SceneSdfSpec): GPUComputePipeline {
        const src = buildG2pWgsl(scene, pagedGrid);
        let pipe = g2pPipeCache.get(src);
        if (!pipe) {
            pipe = pipeline("mpm-g2p", src);
            g2pPipeCache.set(src, pipe);
        }
        return pipe;
    }
    let g2pPipe: GPUComputePipeline | null = null;
    const copyPipe = pipeline("mpm-copy", COPY_WGSL);
    const flowPipe = pipeline("mpm-flow", FLOW_WGSL);
    let pagedGridOverflowed = false;
    let pageStatusGeneration = 0;
    let disposed = false;
    const pageStatusStates: Array<"idle" | "copied" | "mapping"> = pageStatusStagingBuffers.map(() => "idle");
    function pollPagedGridStatus(): void {
        if (!pagedGrid) {
            return;
        }
        for (let i = 0; i < pageStatusStates.length; i++) {
            if (pageStatusStates[i] !== "copied") {
                continue;
            }
            pageStatusStates[i] = "mapping";
            const generation = pageStatusGeneration;
            const staging = pageStatusStagingBuffers[i]!;
            staging
                .mapAsync(GPUMapMode.READ)
                .then(() => {
                    const status = new Uint32Array(staging.getMappedRange());
                    const requiredPages = status[0]!;
                    const overflow = status[1]!;
                    staging.unmap();
                    pageStatusStates[i] = "idle";
                    if (!disposed && generation === pageStatusGeneration && overflow !== 0 && !pagedGridOverflowed) {
                        pagedGridOverflowed = true;
                        options.onPagedGridOverflow?.(requiredPages, pagedGridMaxPages);
                    }
                })
                .catch(() => {
                    if (!disposed) {
                        pageStatusStates[i] = "idle";
                    }
                });
        }
    }
    const flowBG = device.createBindGroup({
        layout: flowPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: flowState.uniformBuffer } },
            { binding: 2, resource: { buffer: flowState.counterBuffer } },
        ],
    });

    const clearBG = device.createBindGroup({
        layout: clearPipe.getBindGroupLayout(0),
        entries: activeBlocks
            ? [
                  { binding: 0, resource: { buffer: cellBuffer } },
                  { binding: 1, resource: { buffer: nodeBlockListBuffer! } },
                  { binding: 2, resource: { buffer: nodeBlockCountBuffer! } },
                  { binding: 3, resource: { buffer: paramsBuffer } },
                  ...(pagedGrid ? [{ binding: 4, resource: { buffer: pageMapBuffer! } }] : []),
              ]
            : [{ binding: 0, resource: { buffer: cellBuffer } }],
    });
    const histogramEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: particleBuffer } },
        { binding: 1, resource: { buffer: blockCountBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
    ];
    if (fusedBlockDiscovery) {
        histogramEntries.push({ binding: 3, resource: { buffer: activeBlockListBuffer! } }, { binding: 4, resource: { buffer: activeBlockCountBuffer! } });
    }
    const histogramBG = device.createBindGroup({
        layout: histogramPipe.getBindGroupLayout(0),
        entries: histogramEntries,
    });
    const scanLocalBG = device.createBindGroup({
        layout: scanLocalPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: blockCountBuffer } },
            { binding: 1, resource: { buffer: blockStartBuffer } },
            { binding: 2, resource: { buffer: partialSumsBuffer } },
        ],
    });
    const scanPartialsBG = device.createBindGroup({
        layout: scanPartialsPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: partialSumsBuffer } }],
    });
    const scanAddBG = device.createBindGroup({
        layout: scanAddPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: blockStartBuffer } },
            { binding: 1, resource: { buffer: partialSumsBuffer } },
        ],
    });
    const scatterBG = device.createBindGroup({
        layout: scatterPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: blockStartBuffer } },
            { binding: 2, resource: { buffer: blockCursorBuffer } },
            { binding: 3, resource: { buffer: sortedIdxBuffer } },
            { binding: 4, resource: { buffer: paramsBuffer } },
        ],
    });
    const p2gEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: particleBuffer } },
        { binding: 1, resource: { buffer: cellBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
        { binding: 3, resource: { buffer: blockStartBuffer } },
        { binding: 4, resource: { buffer: blockCountBuffer } },
        { binding: 5, resource: { buffer: sortedIdxBuffer } },
    ];
    if (activeBlocks) {
        p2gEntries.push({ binding: 6, resource: { buffer: activeBlockListBuffer! } }, { binding: 7, resource: { buffer: activeBlockCountBuffer! } });
    }
    if (pagedGrid) {
        p2gEntries.push({ binding: 8, resource: { buffer: pageMapBuffer! } });
    }
    const p2gMassTiledBG = device.createBindGroup({
        layout: p2gMassTiledPipe.getBindGroupLayout(0),
        entries: p2gEntries,
    });
    const p2gVelTiledBG = device.createBindGroup({
        layout: p2gVelTiledPipe.getBindGroupLayout(0),
        entries: p2gEntries,
    });
    const compactActiveBlocksBG =
        activeBlocks && !fusedBlockDiscovery
            ? device.createBindGroup({
                  layout: compactActiveBlocksPipe!.getBindGroupLayout(0),
                  entries: [
                      { binding: 0, resource: { buffer: blockCountBuffer } },
                      { binding: 1, resource: { buffer: activeBlockListBuffer! } },
                      { binding: 2, resource: { buffer: activeBlockCountBuffer! } },
                  ],
              })
            : null;
    const finalizeActiveBlocksBG = activeBlocks
        ? device.createBindGroup({
              layout: finalizeIndirectPipe!.getBindGroupLayout(0),
              entries: [
                  { binding: 0, resource: { buffer: activeBlockCountBuffer! } },
                  { binding: 1, resource: { buffer: activeBlockIndirectBuffer! } },
              ],
          })
        : null;
    const markActiveNodeBlocksBG = activeBlocks
        ? device.createBindGroup({
              layout: markActiveNodeBlocksPipe!.getBindGroupLayout(0),
              entries: [
                  { binding: 0, resource: { buffer: activeBlockListBuffer! } },
                  { binding: 1, resource: { buffer: activeBlockCountBuffer! } },
                  { binding: 2, resource: { buffer: nodeBlockFlagsBuffer! } },
                  { binding: 3, resource: { buffer: nodeBlockListBuffer! } },
                  { binding: 4, resource: { buffer: nodeBlockCountBuffer! } },
                  { binding: 5, resource: { buffer: paramsBuffer } },
              ],
          })
        : null;
    const finalizeNodeBlocksBG = activeBlocks
        ? device.createBindGroup({
              layout: finalizeIndirectPipe!.getBindGroupLayout(0),
              entries: [
                  { binding: 0, resource: { buffer: nodeBlockCountBuffer! } },
                  { binding: 1, resource: { buffer: nodeBlockIndirectBuffer! } },
              ],
          })
        : null;
    const assignGridPagesBG = pagedGrid
        ? device.createBindGroup({
              layout: assignGridPagesPipe!.getBindGroupLayout(0),
              entries: [
                  { binding: 0, resource: { buffer: nodeBlockListBuffer! } },
                  { binding: 1, resource: { buffer: nodeBlockCountBuffer! } },
                  { binding: 2, resource: { buffer: pageMapBuffer! } },
                  { binding: 3, resource: { buffer: activeBlockCountBuffer! } },
                  { binding: 4, resource: { buffer: cellBuffer } },
              ],
          })
        : null;
    function buildUpdateBG(pipe: GPUComputePipeline, scene: SceneSdfSpec): GPUBindGroup {
        // Only a CLOSED container's update-grid pass reads the scene SDF (for its grid
        // separating wall); a per-particle container has no grid wall, so binding 2 is absent.
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: cellBuffer } },
            { binding: 1, resource: { buffer: paramsBuffer } },
        ];
        if (scene.gridConfine !== false) {
            entries.push({ binding: 2, resource: { buffer: scene.buffer } });
            // Baked SDF grid: matches the @binding(3) storage decl injected by buildUpdateGridWgsl.
            if (scene.sdfGrid) {
                entries.push({ binding: 3, resource: { buffer: scene.sdfGrid } });
            }
        }
        if (activeBlocks) {
            entries.push({ binding: 4, resource: { buffer: nodeBlockListBuffer! } }, { binding: 5, resource: { buffer: nodeBlockCountBuffer! } });
        }
        if (pagedGrid) {
            entries.push({ binding: 6, resource: { buffer: pageMapBuffer! } }, { binding: 7, resource: { buffer: activeBlockCountBuffer! } });
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }
    let updateBG: GPUBindGroup | null = null;
    function buildG2pBG(pipe: GPUComputePipeline, scene: SceneSdfSpec): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: cellBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
            { binding: 3, resource: { buffer: scene.buffer } },
        ];
        // Baked SDF grid: matches the @binding(4) storage decl injected by buildG2pWgsl.
        if (scene.sdfGrid) {
            entries.push({ binding: 4, resource: { buffer: scene.sdfGrid } });
        }
        if (pagedGrid) {
            entries.push({ binding: 5, resource: { buffer: pageMapBuffer! } }, { binding: 6, resource: { buffer: activeBlockCountBuffer! } });
        }
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    }
    let g2pBG: GPUBindGroup | null = null;
    // Current scene SDF; setSceneSdf rebuilds the G2P + update-grid passes from it.
    let currentScene: SceneSdfSpec | null = null;
    function rebuildScenePipes(): void {
        if (currentScene) {
            g2pPipe = getG2pPipe(currentScene);
            g2pBG = buildG2pBG(g2pPipe, currentScene);
            updatePipe = getUpdatePipe(currentScene);
            updateBG = buildUpdateBG(updatePipe, currentScene);
        } else {
            g2pPipe = null;
            g2pBG = null;
            updatePipe = null;
            updateBG = null;
        }
    }
    const copyBG = device.createBindGroup({
        layout: copyPipe.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: positionBuffer } },
            { binding: 2, resource: { buffer: debugBuffer } },
            { binding: 3, resource: { buffer: copyParamsBuffer } },
            { binding: 4, resource: { buffer: velocityBuffer } },
        ],
    });

    // Dedicated external-force pass (setForceField). Built LAZILY on the first
    // non-null injection and cached by the force WGSL source, so NOTHING force
    // related is compiled until a force is actually used. `forceSpec` is the enable
    // flag (null = disabled); `forceBuiltSpec` tracks what the cached pipeline +
    // bind-group were built for, so re-enabling the same spec is free.
    const forcePipeCache = new Map<string, GPUComputePipeline>();
    let forceSpec: ForceFieldSpec | null = null;
    let forceBuiltSpec: ForceFieldSpec | null = null;
    let forcePipe: GPUComputePipeline | null = null;
    let forceBG: GPUBindGroup | null = null;
    function buildForceBG(pipe: GPUComputePipeline, spec: ForceFieldSpec): GPUBindGroup {
        return device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer } },
                { binding: 1, resource: { buffer: paramsBuffer } },
                { binding: 2, resource: { buffer: spec.buffer } },
            ],
        });
    }
    const particleGroups = Math.ceil(count / WORKGROUP_SIZE);
    const cellGroups = Math.ceil(numCells / WORKGROUP_SIZE);
    // Block counting-sort dispatch sizes: one workgroup per block for the tiled P2G, and
    // block-/chunk-indexed groups for the scan (all spill past MAX_WORKGROUPS via dispatch()).
    const blockDispatch = numBlocks; // one workgroup per block (tiled P2G)
    const blockGroups = Math.ceil(numBlocks / WORKGROUP_SIZE); // block-indexed particle-style passes

    function dispatch(encoder: GPUCommandEncoder, label: string, pipe: GPUComputePipeline, bg: GPUBindGroup, groups: number): void {
        // Opt-in GPU timing: "foam" labels → "Foam gen", everything else → "Simulation".
        const pass = encoder.beginComputePass({ label, timestampWrites: profiler?.pass(label.includes("foam") ? "Foam gen" : "Simulation") });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bg);
        if (groups > MAX_WORKGROUPS) {
            pass.dispatchWorkgroups(MAX_WORKGROUPS, Math.ceil(groups / MAX_WORKGROUPS), 1);
        } else {
            pass.dispatchWorkgroups(groups);
        }
        pass.end();
    }

    function dispatchIndirect(encoder: GPUCommandEncoder, label: string, pipe: GPUComputePipeline, bg: GPUBindGroup, args: GPUBuffer): void {
        const pass = encoder.beginComputePass({ label, timestampWrites: profiler?.pass(label.includes("foam") ? "Foam gen" : "Simulation") });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroupsIndirect(args, 0);
        pass.end();
    }

    // ── Foam (Ihmsen 2012 diffuse particles) — lazily allocated on first setFoam ──
    // Mirrors the PBF backend: the ring pool + the two grid-based compute passes are
    // built the first time foam is enabled, so a sim that never turns foam on pays
    // nothing. The pool is sized D = poolScale × count (capped) and reused; the emit
    // pass overwrites the oldest slots via an atomic write-head modulo D. The shared
    // FoamParams UBO carries the tuning knobs; frame dt / gravity / restDensity / dx /
    // bounds are read from the Params UBO (frame dt via misc2.y, written each frame).
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
    let foamActiveParticles = false;
    let foamActiveSide = 0;
    let diffuseBuffer: GPUBuffer | null = null;
    let diffuseHeadBuffer: GPUBuffer | null = null;
    let foamActiveStateBuffer: GPUBuffer | null = null;
    let foamActiveDispatchBuffer: GPUBuffer | null = null;
    let foamDrawIndirectBuffer: GPUBuffer | null = null;
    let foamParamsBuffer: GPUBuffer | null = null;
    let foamEmitPipe: GPUComputePipeline | null = null;
    let foamDenseEmitPipe: GPUComputePipeline | null = null;
    let foamActiveEmitPipe: GPUComputePipeline | null = null;
    let foamUpdatePipe: GPUComputePipeline | null = null;
    let foamDenseUpdatePipe: GPUComputePipeline | null = null;
    let foamActiveUpdatePipe: GPUComputePipeline | null = null;
    let foamActivePreparePipe: GPUComputePipeline | null = null;
    let foamActiveFinishPipe: GPUComputePipeline | null = null;
    let foamEmitBG: GPUBindGroup | null = null;
    let foamUpdateBG: GPUBindGroup | null = null;
    let foamActivePrepareBG: GPUBindGroup | null = null;
    let foamActiveFinishBG: GPUBindGroup | null = null;
    let diffusePool: DiffusePool | undefined;

    function buildFoamBindGroups(): void {
        foamEmitBG = device.createBindGroup({
            layout: foamEmitPipe!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer } },
                { binding: 1, resource: { buffer: cellBuffer } },
                { binding: 2, resource: { buffer: paramsBuffer } },
                { binding: 3, resource: { buffer: foamParamsBuffer! } },
                { binding: 4, resource: { buffer: diffuseBuffer! } },
                { binding: 5, resource: { buffer: foamActiveParticles ? foamActiveStateBuffer! : diffuseHeadBuffer! } },
                ...(pagedGrid
                    ? [
                          { binding: 6, resource: { buffer: pageMapBuffer! } },
                          { binding: 7, resource: { buffer: activeBlockCountBuffer! } },
                      ]
                    : []),
            ],
        });
        const updateEntries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: cellBuffer } },
            { binding: 1, resource: { buffer: paramsBuffer } },
            { binding: 2, resource: { buffer: foamParamsBuffer! } },
            { binding: 3, resource: { buffer: diffuseBuffer! } },
        ];
        if (foamActiveParticles) {
            updateEntries.push({ binding: 4, resource: { buffer: foamActiveStateBuffer! } });
        }
        if (pagedGrid) {
            updateEntries.push({ binding: 6, resource: { buffer: pageMapBuffer! } }, { binding: 7, resource: { buffer: activeBlockCountBuffer! } });
        }
        foamUpdateBG = device.createBindGroup({
            layout: foamUpdatePipe!.getBindGroupLayout(0),
            entries: updateEntries,
        });
        foamActivePrepareBG = foamActiveParticles
            ? device.createBindGroup({
                  layout: foamActivePreparePipe!.getBindGroupLayout(0),
                  entries: [
                      { binding: 0, resource: { buffer: foamActiveStateBuffer! } },
                      { binding: 1, resource: { buffer: foamActiveDispatchBuffer! } },
                  ],
              })
            : null;
        foamActiveFinishBG = foamActiveParticles
            ? device.createBindGroup({
                  layout: foamActiveFinishPipe!.getBindGroupLayout(0),
                  entries: [
                      { binding: 0, resource: { buffer: foamActiveStateBuffer! } },
                      { binding: 1, resource: { buffer: foamDrawIndirectBuffer! } },
                  ],
              })
            : null;
    }

    function ensureFoam(cfg: FoamConfig): void {
        if (!foamEmitPipe) {
            foamDenseEmitPipe = pipeline("mpm-foam-emit", buildFoamEmitWgsl(false, pagedGrid));
            foamDenseUpdatePipe = pipeline("mpm-foam-update", buildFoamUpdateWgsl(false, pagedGrid));
            foamParamsBuffer = device.createBuffer({ label: "mpm-foam-params", size: FOAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            diffuseHeadBuffer = device.createBuffer({ label: "mpm-foam-head", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        }
        const nextActiveParticles = cfg.activeParticles ?? false;
        const activeModeChanged = nextActiveParticles !== foamActiveParticles;
        if (nextActiveParticles && !foamActiveUpdatePipe) {
            foamActiveEmitPipe = pipeline("mpm-foam-emit-active", buildFoamEmitWgsl(true, pagedGrid));
            foamActiveUpdatePipe = pipeline("mpm-foam-update-active", buildFoamUpdateWgsl(true, pagedGrid));
            foamActivePreparePipe = pipeline("mpm-foam-active-prepare", FOAM_ACTIVE_PREPARE_WGSL);
            foamActiveFinishPipe = pipeline("mpm-foam-active-finish", FOAM_ACTIVE_FINISH_WGSL);
        }
        foamActiveParticles = nextActiveParticles;
        foamEmitPipe = foamActiveParticles ? foamActiveEmitPipe! : foamDenseEmitPipe!;
        foamUpdatePipe = foamActiveParticles ? foamActiveUpdatePipe! : foamDenseUpdatePipe!;
        let cap = Math.round(count * (cfg.poolScale ?? 3));
        cap = Math.max(1024, Math.min(cap, cfg.poolCapMax ?? Infinity, FOAM_CAP_LIMIT));
        if (foamActiveParticles) {
            cap = Math.min(cap, Math.max(1024, Math.floor((device.limits.maxStorageBufferBindingSize - 256) / 12)));
            while (foamActiveStateBytes(cap) > device.limits.maxStorageBufferBindingSize) {
                cap--;
            }
        }
        const resizePool = cap !== foamCapacity || !diffuseBuffer;
        const rebuildActiveResources = foamActiveParticles && (resizePool || !foamActiveStateBuffer || !foamActiveDispatchBuffer || !foamDrawIndirectBuffer);
        if (resizePool) {
            diffuseBuffer?.destroy();
            diffuseBuffer = device.createBuffer({ label: "mpm-foam-pool", size: cap * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            foamCapacity = cap;
            foamPoolGroups = Math.ceil(cap / WORKGROUP_SIZE);
            // Zero the pool (all slots dead) + reset the ring head on (re)allocation.
            const enc = device.createCommandEncoder({ label: "mpm-foam-clear" });
            enc.clearBuffer(diffuseBuffer);
            enc.clearBuffer(diffuseHeadBuffer!);
            device.queue.submit([enc.finish()]);
        }
        if (rebuildActiveResources) {
            foamActiveStateBuffer?.destroy();
            foamActiveDispatchBuffer?.destroy();
            foamDrawIndirectBuffer?.destroy();
            foamActiveStateBuffer = device.createBuffer({
                label: "mpm-foam-active-state",
                size: foamActiveStateBytes(cap),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            foamActiveDispatchBuffer = device.createBuffer({
                label: "mpm-foam-active-dispatch",
                size: 12,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
            });
            foamDrawIndirectBuffer = device.createBuffer({
                label: "mpm-foam-draw-indirect",
                size: 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
            });
            foamActiveSide = 0;
        } else if (!foamActiveParticles && foamActiveStateBuffer) {
            foamActiveStateBuffer.destroy();
            foamActiveDispatchBuffer!.destroy();
            foamDrawIndirectBuffer!.destroy();
            foamActiveStateBuffer = null;
            foamActiveDispatchBuffer = null;
            foamDrawIndirectBuffer = null;
        }
        if (foamActiveParticles && (rebuildActiveResources || activeModeChanged)) {
            const enc = device.createCommandEncoder({ label: "mpm-foam-active-reset" });
            enc.clearBuffer(diffuseBuffer!);
            enc.clearBuffer(diffuseHeadBuffer!);
            enc.clearBuffer(foamActiveStateBuffer!);
            device.queue.submit([enc.finish()]);
            device.queue.writeBuffer(foamActiveStateBuffer!, 16, new Uint32Array([cap]));
            foamActiveSide = 0;
        } else if (!foamActiveParticles && activeModeChanged) {
            const enc = device.createCommandEncoder({ label: "mpm-foam-dense-head-reset" });
            enc.clearBuffer(diffuseHeadBuffer!);
            device.queue.submit([enc.finish()]);
        }
        diffusePool = {
            buffer: diffuseBuffer!,
            headBuffer: foamActiveParticles ? foamActiveStateBuffer! : diffuseHeadBuffer!,
            capacity: cap,
            ...(foamActiveParticles
                ? {
                      activeIndices: foamActiveStateBuffer!,
                      activeIndicesOffset: foamActiveListOffset(cap, foamActiveSide),
                      drawIndirect: foamDrawIndirectBuffer!,
                  }
                : {}),
        };
        buildFoamBindGroups();
        // Foam-specific knobs (dt / gravity / restDensity / dx / bounds come from Params).
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

    function clearFoamPool(label: string): void {
        if (!diffuseBuffer || !diffuseHeadBuffer) {
            return;
        }
        const enc = device.createCommandEncoder({ label });
        enc.clearBuffer(diffuseBuffer);
        enc.clearBuffer(diffuseHeadBuffer);
        if (foamActiveStateBuffer) {
            enc.clearBuffer(foamActiveStateBuffer);
            enc.clearBuffer(foamDrawIndirectBuffer!);
        }
        device.queue.submit([enc.finish()]);
        if (foamActiveStateBuffer) {
            device.queue.writeBuffer(foamActiveStateBuffer, 16, new Uint32Array([foamCapacity]));
            foamActiveSide = 0;
            if (diffusePool) {
                diffusePool = { ...diffusePool, activeIndicesOffset: foamActiveListOffset(foamCapacity, foamActiveSide) };
            }
        }
    }

    return {
        count,
        get activeCount(): number {
            return liveCount;
        },
        particleRadius,
        // MLS-MPM particles settle on a near-regular lattice spaced wider than the
        // PBF fluid packs, so without bigger impostors the surface shows the
        // individual spheres. Enlarge them (and the blur) to match SPH smoothness.
        surfaceSizeScale: 1.5,
        positionBuffer,
        velocityBuffer,
        debugBuffer,
        debugNorm: 1 / 6,
        get gpuBytes(): number {
            // Sum every GPU buffer this backend owns; the lazily-allocated foam pool +
            // uniforms are added only once foam has been enabled.
            let b =
                particleBuffer.size +
                cellBuffer.size +
                positionBuffer.size +
                velocityBuffer.size +
                debugBuffer.size +
                paramsBuffer.size +
                copyParamsBuffer.size +
                flowState.uniformBuffer.size +
                flowState.counterBuffer.size;
            // Block counting-sort buffers (always allocated).
            b += blockCountBuffer.size + blockStartBuffer.size + blockCursorBuffer.size + partialSumsBuffer.size + sortedIdxBuffer.size;
            if (activeBlockListBuffer) {
                b +=
                    activeBlockListBuffer.size +
                    activeBlockCountBuffer!.size +
                    activeBlockIndirectBuffer!.size +
                    nodeBlockFlagsBuffer!.size +
                    nodeBlockListBuffer!.size +
                    nodeBlockCountBuffer!.size +
                    nodeBlockIndirectBuffer!.size;
            }
            if (pageMapBuffer) {
                b += pageMapBuffer.size;
                for (const buffer of pageStatusStagingBuffers) {
                    b += buffer.size;
                }
            }
            if (diffuseBuffer) {
                b += diffuseBuffer.size;
            }
            if (diffuseHeadBuffer) {
                b += diffuseHeadBuffer.size;
            }
            if (foamActiveStateBuffer) {
                b += foamActiveStateBuffer.size + foamActiveDispatchBuffer!.size + foamDrawIndirectBuffer!.size;
            }
            if (foamParamsBuffer) {
                b += foamParamsBuffer.size;
            }
            return b;
        },
        step(encoder: GPUCommandEncoder, dt: number): void {
            pollPagedGridStatus();
            if (pagedGridOverflowed) {
                return;
            }
            // Split the (real-time) frame dt into `substeps` MLS-MPM steps, so the
            // substeps slider trades stability vs cost without changing playback
            // speed.
            //
            // `maxSubDt` is honoured by ADDING sub-steps, never by shortening the frame:
            // clamping the sub-step dt (the old `min(frameDt / substeps, maxSubDt)`) silently
            // dropped simulated time whenever `frameDt / substeps` exceeded the cap, so playback
            // speed tracked the frame rate instead of the clock — at substeps = 1 and 60 fps a
            // frame advanced 1/120 s, i.e. HALF speed, and a machine rendering faster ran the
            // fluid faster. The frame now always advances exactly `frameDt`.
            const frameDt = dt > 0 ? dt : 1 / 60;
            // The epsilon absorbs float error so an exact multiple (1/60 over a 1/120 cap) stays
            // at 2 sub-steps instead of rounding up to 3.
            const stepCount = Math.max(substepsMut, Math.ceil(frameDt / maxSubDt - 1e-9));
            pf[16] = frameDt / stepCount;
            pf[MISC2_BASE_F32 + 1] = frameDt; // misc2.y — frame dt for the foam emit count
            // Warm-up ramp: grow the live-particle count by one batch per frame. counts.z
            // gates the mass/integration passes; copyU32[2] gates the render copy pass.
            if (liveCount < initialTargetCount) {
                liveCount = Math.min(initialTargetCount, liveCount + warmupStep);
            }
            const flowFrame = prepareFluidFlowFrame(flowState, frameDt, liveCount, liveCount >= initialTargetCount ? count - liveCount : 0);
            if (flowFrame.particles) {
                activateInflowParticles(liveCount, flowFrame.particles);
                liveCount += flowFrame.particles.activeCount;
            }
            pu[COUNTS_OFFSET_F32 + 2] = liveCount;
            if (copyU32[2] !== liveCount) {
                copyU32[2] = liveCount;
                device.queue.writeBuffer(copyParamsBuffer, 0, copyData);
            }
            device.queue.writeBuffer(paramsBuffer, 0, paramsData);
            // PIX / GPU-capture debug group: scopes this frame's MLS-MPM compute
            // passes (plus the nested substep + foam groups). Balanced by the
            // popDebugGroup at the end of step().
            encoder.pushDebugGroup("MLS-MPM sim step");
            if (flowFrame.recycleActive) {
                dispatch(encoder, "mpm-flow", flowPipe, flowBG, particleGroups);
            }
            encoder.pushDebugGroup(`substeps (${stepCount})`);
            for (let s = 0; s < stepCount; s++) {
                // Optional interactive force (setForceField) runs as its own pass at
                // the start of each substep, so the per-frame velocity impulse is
                // accel·substepDt × substeps = accel·frameDt — independent of the
                // substeps count (matches the old in-G2P application). Only dispatched
                // while a force is active; nothing force-related runs (or compiles) idle.
                if (forceSpec && forcePipe && forceBG) {
                    dispatch(encoder, "mpm-force", forcePipe, forceBG, particleGroups);
                }
                if (activeBlocks) {
                    // The node list is intentionally retained between substeps: it identifies
                    // exactly which grid blocks the previous P2G populated, so this indirect
                    // clear removes stale values before the list is rebuilt for the new state.
                    dispatchIndirect(encoder, "mpm-clear-active", clearPipe, clearBG, nodeBlockIndirectBuffer!);
                    if (pagedGrid) {
                        encoder.clearBuffer(pageMapBuffer!);
                    }
                } else {
                    dispatch(encoder, "mpm-clear", clearPipe, clearBG, cellGroups);
                }
                // Block counting sort of the live particles (histogram -> prefix sum ->
                // scatter), then the shared-memory-tiled P2G. blockCount/blockCursor are
                // zeroed here (clearBuffer) before the histogram/scatter accumulate into them.
                encoder.clearBuffer(blockCountBuffer);
                encoder.clearBuffer(blockCursorBuffer);
                if (activeBlocks) {
                    encoder.clearBuffer(activeBlockCountBuffer!);
                    encoder.clearBuffer(nodeBlockCountBuffer!);
                    encoder.clearBuffer(nodeBlockFlagsBuffer!);
                }
                dispatch(encoder, "mpm-histogram", histogramPipe, histogramBG, particleGroups);
                if (activeBlocks) {
                    if (!fusedBlockDiscovery) {
                        dispatch(encoder, "mpm-compact-active-blocks", compactActiveBlocksPipe!, compactActiveBlocksBG!, blockGroups);
                    }
                    dispatch(encoder, "mpm-finalize-active-blocks", finalizeIndirectPipe!, finalizeActiveBlocksBG!, 1);
                    dispatchIndirect(encoder, "mpm-mark-active-node-blocks", markActiveNodeBlocksPipe!, markActiveNodeBlocksBG!, activeBlockIndirectBuffer!);
                    dispatch(encoder, "mpm-finalize-node-blocks", finalizeIndirectPipe!, finalizeNodeBlocksBG!, 1);
                    if (pagedGrid) {
                        dispatchIndirect(encoder, "mpm-assign-grid-pages", assignGridPagesPipe!, assignGridPagesBG!, nodeBlockIndirectBuffer!);
                    }
                }
                dispatch(encoder, "mpm-scan-local", scanLocalPipe, scanLocalBG, scanChunks);
                dispatch(encoder, "mpm-scan-partials", scanPartialsPipe, scanPartialsBG, 1);
                dispatch(encoder, "mpm-scan-add", scanAddPipe, scanAddBG, blockGroups);
                dispatch(encoder, "mpm-scatter", scatterPipe, scatterBG, particleGroups);
                if (activeBlocks) {
                    dispatchIndirect(encoder, "mpm-p2g-mass", p2gMassTiledPipe, p2gMassTiledBG, activeBlockIndirectBuffer!);
                    dispatchIndirect(encoder, "mpm-p2g-vel", p2gVelTiledPipe, p2gVelTiledBG, activeBlockIndirectBuffer!);
                } else {
                    dispatch(encoder, "mpm-p2g-mass", p2gMassTiledPipe, p2gMassTiledBG, blockDispatch);
                    dispatch(encoder, "mpm-p2g-vel", p2gVelTiledPipe, p2gVelTiledBG, blockDispatch);
                }
                if (updatePipe && updateBG) {
                    if (activeBlocks) {
                        dispatchIndirect(encoder, "mpm-update", updatePipe, updateBG, nodeBlockIndirectBuffer!);
                    } else {
                        dispatch(encoder, "mpm-update", updatePipe, updateBG, cellGroups);
                    }
                }
                if (g2pPipe && g2pBG) {
                    dispatch(encoder, "mpm-g2p", g2pPipe, g2pBG, particleGroups);
                }
            }
            encoder.popDebugGroup();
            // Foam: generate + advect diffuse particles ONCE per frame, AFTER the substep
            // loop and BEFORE the copy — the grid then holds the last substep's mass +
            // velocity and particles[i].v/.C hold the last G2P output. Skipped when off.
            if (foamEnabled && foamUpdateBG) {
                foamU32[14] = foamSeed++;
                device.queue.writeBuffer(foamParamsBuffer!, 0, foamData);
                encoder.pushDebugGroup("foam");
                dispatch(encoder, "mpm-foam-emit", foamEmitPipe!, foamEmitBG!, particleGroups);
                if (foamActiveParticles) {
                    dispatch(encoder, "mpm-foam-active-prepare", foamActivePreparePipe!, foamActivePrepareBG!, 1);
                    dispatchIndirect(encoder, "mpm-foam-update", foamUpdatePipe!, foamUpdateBG, foamActiveDispatchBuffer!);
                    dispatch(encoder, "mpm-foam-active-finish", foamActiveFinishPipe!, foamActiveFinishBG!, 1);
                    foamActiveSide = 1 - foamActiveSide;
                    diffusePool = {
                        buffer: diffuseBuffer!,
                        headBuffer: foamActiveStateBuffer!,
                        capacity: foamCapacity,
                        activeIndices: foamActiveStateBuffer!,
                        activeIndicesOffset: foamActiveListOffset(foamCapacity, foamActiveSide),
                        drawIndirect: foamDrawIndirectBuffer!,
                    };
                } else {
                    dispatch(encoder, "mpm-foam-update", foamUpdatePipe!, foamUpdateBG, foamPoolGroups);
                }
                encoder.popDebugGroup();
            }
            dispatch(encoder, "mpm-copy", copyPipe, copyBG, particleGroups);
            if (pagedGrid) {
                const stagingIndex = pageStatusStates.indexOf("idle");
                if (stagingIndex !== -1) {
                    encoder.copyBufferToBuffer(activeBlockCountBuffer!, 4, pageStatusStagingBuffers[stagingIndex]!, 0, 8);
                    pageStatusStates[stagingIndex] = "copied";
                }
            }
            encoder.popDebugGroup();
        },
        get diffuse(): DiffusePool | undefined {
            return diffusePool;
        },
        reset(): void {
            seed();
            clearFoamPool("mpm-foam-reset");
            if (pagedGrid) {
                pagedGridOverflowed = false;
                pageStatusGeneration++;
                const enc = device.createCommandEncoder({ label: "mpm-paged-grid-reset" });
                enc.clearBuffer(cellBuffer);
                enc.clearBuffer(pageMapBuffer!);
                enc.clearBuffer(activeBlockCountBuffer!);
                device.queue.submit([enc.finish()]);
            }
        },
        setParam(key: string, value: number): void {
            switch (key) {
                case "gravity":
                    pf[17] = value;
                    break;
                case "restDensity":
                    pf[18] = value;
                    break;
                case "stiffness":
                    pf[19] = value;
                    break;
                case "viscosity":
                    pf[20] = value;
                    break;
                case "damping":
                    pf[21] = value;
                    break;
                case "affineDamping":
                    pf[22] = value;
                    break;
                case "groundDamp":
                    pf[23] = value;
                    break;
                case "groundDampHeight":
                    pf[7] = value;
                    break;
                case "restitution":
                    pf[MISC2_BASE_F32] = value;
                    break;
                case "substeps":
                    substepsMut = Math.max(1, Math.round(value));
                    break;
                case "maxSubDtMs":
                    // Largest dt a single sub-step may integrate, in MILLISECONDS. Raising it lets a
                    // frame be covered by fewer sub-steps (cheaper, less stable); lowering it forces
                    // more (costlier, more stable). It never changes how much time a frame advances.
                    maxSubDt = Math.max(1e-4, value / 1000);
                    break;
            }
        },
        setSceneSdf(spec: SceneSdfSpec | null): void {
            currentScene = spec;
            rebuildScenePipes();
        },
        setEmitters(config: EmitterConfig | null): void {
            flowSeedsInitialParticles = false;
            setFluidFlowConfig(flowState, legacyEmitterConfigToFluidFlow(config, count));
        },
        setFlow(config: FluidFlowConfig | null): void {
            flowSeedsInitialParticles = true;
            setFluidFlowConfig(flowState, config);
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
            // Number of frames over which reset()/seed() gradually releases initial
            // particles (0 = all at once). Takes effect on the next seed()/reset().
            warmupFrames = Math.max(0, Math.floor(frames));
            warmupStep = warmupFrames > 0 ? Math.max(1, Math.ceil(count / warmupFrames)) : count;
        },
        setForceField(spec: ForceFieldSpec | null): void {
            forceSpec = spec;
            if (!spec || spec === forceBuiltSpec) {
                return;
            }
            let pipe = forcePipeCache.get(spec.wgsl);
            if (!pipe) {
                pipe = pipeline("mpm-force", buildForceWgsl(spec));
                forcePipeCache.set(spec.wgsl, pipe);
            }
            forcePipe = pipe;
            forceBG = buildForceBG(pipe, spec);
            forceBuiltSpec = spec;
        },
        setFoam(cfg: FoamConfig | null): void {
            if (!cfg) {
                foamEnabled = false;
                // Empty the pool so a later re-enable starts clean (no frozen ghosts).
                clearFoamPool("mpm-foam-off-clear");
                return;
            }
            ensureFoam(cfg);
            foamEnabled = true;
        },
        setProfiler(p: FluidProfiler | null): void {
            profiler = p;
        },
        dispose(): void {
            disposed = true;
            particleBuffer.destroy();
            cellBuffer.destroy();
            positionBuffer.destroy();
            velocityBuffer.destroy();
            debugBuffer.destroy();
            paramsBuffer.destroy();
            copyParamsBuffer.destroy();
            disposeFluidFlowState(flowState);
            blockCountBuffer.destroy();
            blockStartBuffer.destroy();
            blockCursorBuffer.destroy();
            partialSumsBuffer.destroy();
            sortedIdxBuffer.destroy();
            activeBlockListBuffer?.destroy();
            activeBlockCountBuffer?.destroy();
            activeBlockIndirectBuffer?.destroy();
            nodeBlockFlagsBuffer?.destroy();
            nodeBlockListBuffer?.destroy();
            nodeBlockCountBuffer?.destroy();
            nodeBlockIndirectBuffer?.destroy();
            pageMapBuffer?.destroy();
            for (const buffer of pageStatusStagingBuffers) {
                buffer.destroy();
            }
            diffuseBuffer?.destroy();
            diffuseHeadBuffer?.destroy();
            foamActiveStateBuffer?.destroy();
            foamActiveDispatchBuffer?.destroy();
            foamDrawIndirectBuffer?.destroy();
            foamParamsBuffer?.destroy();
        },
    };
}
