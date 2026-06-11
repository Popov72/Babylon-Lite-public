// GPU fluid simulation — Phases 1–2.
//
// Demo-local (not babylon-lite core). Builds WebGPU compute pipelines from
// `engine._device` and encodes the per-frame passes into the frame command
// encoder (see the demo's onBeforeRender → sim.step).
//
// Phase 1: gravity integration + ground bounce.
// Phase 2: spatial-hash neighbour grid (the backbone for the PBF density
//   solver in Phase 3) + a neighbour-count debug pass used to colour particles
//   so the grid can be verified visually.
//
// Neighbour grid design — FIXED-CAPACITY uniform grid (no prefix-sum scan):
//   • The simulation domain is a box [origin, origin + dim*cellSize]; cellSize
//     equals the smoothing radius h, so a particle's neighbours lie in its own
//     cell + the 26 surrounding cells (3×3×3).
//   • cellCount[cell]    — atomic per-cell population (clamped to maxPerCell on read).
//     cellParticles[cell*maxPerCell + slot] — the particle indices in each cell.
//   • Build is two passes: clear cellCount, then each particle atomicAdd's its
//     cell slot and writes its index. This trades memory (numCells*maxPerCell
//     u32) for avoiding a GPU prefix-sum + scatter (counting sort). Fine for a
//     demo; switch to counting-sort if particle counts/domain grow large.
//   • Particles outside the domain clamp to edge cells (escaped/falling liquid
//     just stops contributing to neighbour search — acceptable).

import type { EngineContext } from "babylon-lite";

export interface FluidSimOptions {
    /** Particle count. Default 30000. */
    count?: number;
    /** Render/visual particle radius in world units. Default 0.08. */
    particleRadius?: number;
    /** Axis-aligned spawn box min the particles are seeded into. */
    spawnMin?: [number, number, number];
    /** Axis-aligned spawn box max the particles are seeded into. */
    spawnMax?: [number, number, number];
    /** Gravity acceleration (m/s²). Default 9.8. */
    gravity?: number;
    /** Ground plane height; particles bounce off `y = groundY`. Default 0. */
    groundY?: number;
    /** Restitution of the ground bounce (0 = stick, 1 = perfectly elastic). Default 0.3. */
    restitution?: number;
    /** Smoothing radius h = neighbour-grid cell size (world units). Default 0.4. */
    smoothingRadius?: number;
    /** Neighbour-grid domain min corner. Default [-10, 0, -10]. */
    gridOrigin?: [number, number, number];
    /** Neighbour-grid dimensions in cells per axis. Default [50, 50, 50]. */
    gridDim?: [number, number, number];
    /** Max particles stored per grid cell. Default 48. */
    maxPerCell?: number;
}

export interface FluidSim {
    readonly count: number;
    readonly particleRadius: number;
    /** vec4<f32>-per-particle position buffer (STORAGE). Read by the renderer. */
    readonly positionBuffer: GPUBuffer;
    /** f32-per-particle neighbour count (Phase 2 debug). Read by the renderer to tint by density. */
    readonly debugBuffer: GPUBuffer;
    /** Normalisation reciprocal for `debugBuffer` (≈ 1 / typical max neighbour count). */
    readonly debugNorm: number;
    /** Encode one simulation step into `encoder`. `dt` is seconds. */
    step(encoder: GPUCommandEncoder, dt: number): void;
    /** Re-seed all particles into the spawn box with zero velocity. */
    reset(): void;
    dispose(): void;
}

const WORKGROUP_SIZE = 64;

const INTEGRATE_WGSL = /* wgsl */ `
struct Params {
    dt: f32,
    gravity: f32,
    groundY: f32,
    restitution: f32,
    count: u32,
    _p0: u32, _p1: u32, _p2: u32,
};
@group(0) @binding(0) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    var p = pos[i].xyz;
    var v = vel[i].xyz;
    v.y -= params.gravity * params.dt;
    p += v * params.dt;
    if (p.y < params.groundY) {
        p.y = params.groundY;
        v.y = -v.y * params.restitution;
        v.x *= 0.92;
        v.z *= 0.92;
    }
    pos[i] = vec4<f32>(p, 1.0);
    vel[i] = vec4<f32>(v, 0.0);
}`;

// Shared grid-uniform declaration + cell math. originCell = (origin.xyz, cellSize);
// dim = (gridDim.xyz, maxPerCell).
const GRID_HEADER_WGSL = /* wgsl */ `
struct GridParams {
    originCell: vec4<f32>,
    dim: vec4<u32>,
    count: u32,
    _g0: u32, _g1: u32, _g2: u32,
};
fn cellCoordOf(p: vec3<f32>, grid: GridParams) -> vec3<i32> {
    let rel = (p - grid.originCell.xyz) / grid.originCell.w;
    return clamp(vec3<i32>(floor(rel)), vec3<i32>(0), vec3<i32>(grid.dim.xyz) - vec3<i32>(1));
}
fn cellLinear(c: vec3<i32>, grid: GridParams) -> u32 {
    return u32((c.z * i32(grid.dim.y) + c.y) * i32(grid.dim.x) + c.x);
}`;

const CLEAR_GRID_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> cellCount: array<atomic<u32>>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= arrayLength(&cellCount)) { return; }
    atomicStore(&cellCount[c], 0u);
}`;

const BUILD_GRID_WGSL = /* wgsl */ `
${GRID_HEADER_WGSL}
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> cellParticles: array<u32>;
@group(0) @binding(3) var<uniform> grid: GridParams;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= grid.count) { return; }
    let cell = cellLinear(cellCoordOf(pos[i].xyz, grid), grid);
    let slot = atomicAdd(&cellCount[cell], 1u);
    if (slot < grid.dim.w) {
        cellParticles[cell * grid.dim.w + slot] = i;
    }
}`;

const NEIGHBOR_COUNT_WGSL = /* wgsl */ `
${GRID_HEADER_WGSL}
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> cellParticles: array<u32>;
@group(0) @binding(3) var<uniform> grid: GridParams;
@group(0) @binding(4) var<storage, read_write> dbg: array<f32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= grid.count) { return; }
    let pi = pos[i].xyz;
    let h = grid.originCell.w;
    let h2 = h * h;
    let base = cellCoordOf(pi, grid);
    var cnt = 0u;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let cc = base + vec3<i32>(dx, dy, dz);
                if (any(cc < vec3<i32>(0)) || any(cc >= vec3<i32>(grid.dim.xyz))) { continue; }
                let cell = cellLinear(cc, grid);
                let n = min(atomicLoad(&cellCount[cell]), grid.dim.w);
                for (var s = 0u; s < n; s = s + 1u) {
                    let j = cellParticles[cell * grid.dim.w + s];
                    let d = pi - pos[j].xyz;
                    if (dot(d, d) < h2) { cnt = cnt + 1u; }
                }
            }
        }
    }
    dbg[i] = f32(cnt);
}`;

export function createFluidSim(engine: EngineContext, options: FluidSimOptions = {}): FluidSim {
    const device = engine._device;
    const count = options.count ?? 30000;
    const particleRadius = options.particleRadius ?? 0.08;
    const spawnMin = options.spawnMin ?? [-2, 6, -2];
    const spawnMax = options.spawnMax ?? [2, 12, 2];
    const gravity = options.gravity ?? 9.8;
    const groundY = options.groundY ?? 0;
    const restitution = options.restitution ?? 0.3;
    const h = options.smoothingRadius ?? 0.4;
    const gridOrigin = options.gridOrigin ?? [-10, 0, -10];
    const gridDim = options.gridDim ?? [50, 50, 50];
    const maxPerCell = options.maxPerCell ?? 48;
    const numCells = gridDim[0] * gridDim[1] * gridDim[2];

    // ── Buffers ──────────────────────────────────────────────────────
    const positionBuffer = device.createBuffer({ label: "fluid-positions", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const velocityBuffer = device.createBuffer({ label: "fluid-velocities", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const debugBuffer = device.createBuffer({ label: "fluid-debug", size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const cellCountBuffer = device.createBuffer({ label: "fluid-cell-count", size: numCells * 4, usage: GPUBufferUsage.STORAGE });
    const cellParticlesBuffer = device.createBuffer({ label: "fluid-cell-particles", size: numCells * maxPerCell * 4, usage: GPUBufferUsage.STORAGE });
    const paramsBuffer = device.createBuffer({ label: "fluid-params", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const gridParamsBuffer = device.createBuffer({ label: "fluid-grid-params", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const paramsData = new ArrayBuffer(32);
    const paramsF32 = new Float32Array(paramsData);
    const paramsU32 = new Uint32Array(paramsData);

    // Grid params are static for the lifetime of the sim.
    {
        const gridData = new ArrayBuffer(48);
        const gf32 = new Float32Array(gridData);
        const gu32 = new Uint32Array(gridData);
        gf32[0] = gridOrigin[0];
        gf32[1] = gridOrigin[1];
        gf32[2] = gridOrigin[2];
        gf32[3] = h; // cellSize
        gu32[4] = gridDim[0];
        gu32[5] = gridDim[1];
        gu32[6] = gridDim[2];
        gu32[7] = maxPerCell;
        gu32[8] = count;
        device.queue.writeBuffer(gridParamsBuffer, 0, gridData);
    }

    function seed(): void {
        const positions = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) {
            const o = i * 4;
            positions[o] = spawnMin[0] + Math.random() * (spawnMax[0] - spawnMin[0]);
            positions[o + 1] = spawnMin[1] + Math.random() * (spawnMax[1] - spawnMin[1]);
            positions[o + 2] = spawnMin[2] + Math.random() * (spawnMax[2] - spawnMin[2]);
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
    const integratePipeline = computePipeline("fluid-integrate", INTEGRATE_WGSL);
    const clearGridPipeline = computePipeline("fluid-clear-grid", CLEAR_GRID_WGSL);
    const buildGridPipeline = computePipeline("fluid-build-grid", BUILD_GRID_WGSL);
    const neighborPipeline = computePipeline("fluid-neighbor-count", NEIGHBOR_COUNT_WGSL);

    const integrateBG = device.createBindGroup({
        layout: integratePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: velocityBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
        ],
    });
    const clearGridBG = device.createBindGroup({
        layout: clearGridPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: cellCountBuffer } }],
    });
    const buildGridBG = device.createBindGroup({
        layout: buildGridPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: cellCountBuffer } },
            { binding: 2, resource: { buffer: cellParticlesBuffer } },
            { binding: 3, resource: { buffer: gridParamsBuffer } },
        ],
    });
    const neighborBG = device.createBindGroup({
        layout: neighborPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: cellCountBuffer } },
            { binding: 2, resource: { buffer: cellParticlesBuffer } },
            { binding: 3, resource: { buffer: gridParamsBuffer } },
            { binding: 4, resource: { buffer: debugBuffer } },
        ],
    });

    const particleGroups = Math.ceil(count / WORKGROUP_SIZE);
    const cellGroups = Math.ceil(numCells / WORKGROUP_SIZE);

    function dispatch(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline, bg: GPUBindGroup, groups: number): void {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(groups);
        pass.end();
    }

    return {
        count,
        particleRadius,
        positionBuffer,
        debugBuffer,
        // Typical max neighbour count within h ≈ a full 3×3×3 stencil; normalise to that for colour.
        debugNorm: 1 / 60,
        step(encoder: GPUCommandEncoder, dt: number): void {
            paramsF32[0] = dt;
            paramsF32[1] = gravity;
            paramsF32[2] = groundY;
            paramsF32[3] = restitution;
            paramsU32[4] = count;
            device.queue.writeBuffer(paramsBuffer, 0, paramsData);

            dispatch(encoder, "fluid-integrate", integratePipeline, integrateBG, particleGroups);
            dispatch(encoder, "fluid-clear-grid", clearGridPipeline, clearGridBG, cellGroups);
            dispatch(encoder, "fluid-build-grid", buildGridPipeline, buildGridBG, particleGroups);
            dispatch(encoder, "fluid-neighbor-count", neighborPipeline, neighborBG, particleGroups);
        },
        reset(): void {
            seed();
        },
        dispose(): void {
            positionBuffer.destroy();
            velocityBuffer.destroy();
            debugBuffer.destroy();
            cellCountBuffer.destroy();
            cellParticlesBuffer.destroy();
            paramsBuffer.destroy();
            gridParamsBuffer.destroy();
        },
    };
}
