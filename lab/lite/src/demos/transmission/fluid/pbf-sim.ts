// GPU fluid simulation — Phases 1–3 (Position Based Fluids).
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
//   2. clearGrid    cellCount = 0
//   3. buildGrid    bucket each particle by x* into its cell
//   4. × iterations:
//        a. lambda  ρ_i (poly6) → C_i = ρ_i/ρ0 − 1 → λ_i = −C_i / (Σ|∇C|² + ε)
//        b. delta   Δp_i = (1/ρ0) Σ_j (λ_i+λ_j+s_corr) ∇W_spiky(x*_i − x*_j)
//        c. apply   x* += Δp; clamp to the box boundary
//   5. finalize     v = (x* − x)/dt ; x = x*               (write rendered pos)
//   6. viscosity    XSPH smoothing v += c/ρ0 Σ_j (v_j−v_i) W ; debug = speed
//
// Neighbour grid — FIXED-CAPACITY uniform grid (no prefix-sum scan): cellSize
//   equals the smoothing radius h so neighbours lie in the 3×3×3 cell stencil.
//   cellCount[cell] is an atomic population; cellParticles[cell*maxPerCell+slot]
//   holds the indices. The grid is built once per frame from the predicted
//   positions and reused across all solver iterations (standard PBF).

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
}

export interface FluidSim {
    readonly count: number;
    readonly particleRadius: number;
    /** Optional multiplier on the screen-space surface impostor size (and the
     *  bilateral-blur kernel derived from it). Backends whose particles settle at
     *  wider spacing (e.g. MLS-MPM) need bigger, more-overlapping impostors to
     *  render a smooth surface instead of visible individual spheres. Default 1. */
    readonly surfaceSizeScale?: number;
    /** vec4<f32>-per-particle position buffer (STORAGE). Read by the renderer. */
    readonly positionBuffer: GPUBuffer;
    /** f32-per-particle speed (Phase 3 debug). Read by the renderer to tint by motion. */
    readonly debugBuffer: GPUBuffer;
    /** Normalisation reciprocal for `debugBuffer` (≈ 1 / typical max speed). */
    readonly debugNorm: number;
    /** Encode one simulation step into `encoder`. `dt` is seconds. */
    step(encoder: GPUCommandEncoder, dt: number): void;
    /** Open a new spherical hole in the capsule wall (up to MAX_HOLES; the oldest
     *  is replaced past the cap). Liquid reaching any hole escapes and drains. */
    addHole(center: [number, number, number], radius: number): void;
    /** Re-seed all particles into the spawn box with zero velocity and seal the tank. */
    reset(): void;
    /** Live-update a named simulation parameter (for the demo's tuning UI). */
    setParam(key: string, value: number): void;
    /** Switch the confining container: 0 = none, 1 = capsule (creation params),
     *  2 = axis-aligned box [min, max]. */
    setContainer(mode: number, min: [number, number, number], max: [number, number, number]): void;
    /** Set an interactive push force: particles within `radius` of the ray
     *  (origin, dir) are accelerated along `push` by `accel` (0 = disabled). */
    setForce(origin: [number, number, number], dir: [number, number, number], push: [number, number, number], radius: number, accel: number): void;
    /** Configure a rotating vertical paddle obstacle (box mode only). `active`
     *  toggles it; `center` is the XZ pivot, `halfWidth` the horizontal reach
     *  from the pivot, `halfThickness` the slab half-thickness, `angle` the
     *  current rotation (radians about +Y) and `omega` its angular velocity. */
    setObstacle(active: boolean, center: [number, number], halfWidth: number, halfThickness: number, angle: number, omega: number): void;
    dispose(): void;
}

const WORKGROUP_SIZE = 64;
// WebGPU caps a dispatch at 65535 workgroups per dimension. The neighbour grid
// can need far more groups than that at small particle sizes (very fine cells),
// so cell-indexed dispatches spill the overflow into a second (y) dimension and
// the cell kernels rebuild the linear index from num_workgroups.x.
const MAX_WORKGROUPS = 65535;

// Up to this many simultaneous holes (ring buffer; a new press past the cap
// replaces the oldest). Kept small so the holes array stays a tiny uniform.
const MAX_HOLES = 8;

// Sim uniform. Per-frame mutable (dt); the rest are constant or set on demand.
//   [0] dt        [1] gravity   [2] restDensity [3] h
//   [4] h2        [5] poly6     [6] spikyGrad   [7] eps
//   [8] scorrK    [9] scorrInvWdq [10] scorrN   [11] viscosity
//   [12] count(u32) [13] containerMode(u32) [14] boundaryDensity [15] holeCount(u32)
//   [16..19] boundsMin.xyz + pad
//   [20..23] boundsMax.xyz + pad
//   [24..27] capsuleA.xyz + capsuleRadius
//   [28..31] capsuleB.xyz + groundY
//   [32..]   holes[MAX_HOLES] — each vec4 (centre.xyz + radius; radius 0 = unused)
//   then     boxMin.xyz+pad, boxMax.xyz+pad (container box for containerMode 2)
//   then     forceO.xyz+radius, forceD.xyz+accel, forceP.xyz+pad (mouse force)
//   then     obsA (cx,cz,halfWidth,halfThickness), obsB (cos,sin,omega,enabled) — rotating paddle
const HOLE_BASE_F32 = 32;
const BOX_BASE_F32 = HOLE_BASE_F32 + MAX_HOLES * 4;
const FORCE_BASE_F32 = BOX_BASE_F32 + 8;
const OBS_BASE_F32 = FORCE_BASE_F32 + 12;
const SIM_BYTES = (OBS_BASE_F32 + 8) * 4;

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
    holeCount: u32,
    boundsMin: vec4<f32>,
    boundsMax: vec4<f32>,
    capsuleA: vec4<f32>,
    capsuleB: vec4<f32>,
    holes: array<vec4<f32>, ${MAX_HOLES}>,
    boxMin: vec4<f32>,
    boxMax: vec4<f32>,
    forceO: vec4<f32>,
    forceD: vec4<f32>,
    forceP: vec4<f32>,
    obsA: vec4<f32>,
    obsB: vec4<f32>,
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

// Interactive mouse force: an acceleration pushing particles near the cursor ray
// (origin forceO.xyz, dir forceD.xyz) along forceP.xyz, with linear falloff over
// radius forceO.w. forceD.w (accel) = 0 disables it.
fn mouseForce(p: vec3<f32>, sim: Sim) -> vec3<f32> {
    let accel = sim.forceD.w;
    if (accel <= 0.0) { return vec3<f32>(0.0); }
    let o = sim.forceO.xyz;
    let dir = sim.forceD.xyz;
    let t = dot(p - o, dir);
    if (t <= 0.0) { return vec3<f32>(0.0); }
    let dist = length(p - (o + t * dir));
    if (dist >= sim.forceO.w) { return vec3<f32>(0.0); }
    return sim.forceP.xyz * (accel * (1.0 - dist / sim.forceO.w));
}

// Rotating vertical paddle obstacle (box mode). If p is inside the thin slab,
// push it out along the slab normal to the nearer face. PBF derives velocity
// from the position change, so the sweeping paddle imparts the stir for free.
// obsA = (cx, cz, halfWidth, halfThickness); obsB = (cos, sin, omega, enabled).
fn obstacleResolve(p: vec3<f32>, sim: Sim) -> vec3<f32> {
    if (sim.obsB.w < 0.5) { return p; }
    let c = sim.obsB.x;
    let s = sim.obsB.y;
    let rx = p.x - sim.obsA.x;
    let rz = p.z - sim.obsA.y;
    let lx =  c * rx + s * rz; // local slab-normal axis
    let lz = -s * rx + c * rz; // local slab-width axis
    if (abs(lx) < sim.obsA.w && abs(lz) < sim.obsA.z) {
        let nl = select(-sim.obsA.w, sim.obsA.w, lx >= 0.0); // nearer face
        let nrx = c * nl - s * lz;
        let nrz = s * nl + c * lz;
        return vec3<f32>(sim.obsA.x + nrx, p.y, sim.obsA.y + nrz);
    }
    return p;
}
`;

const PREDICT_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> predicted: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> sim: Sim;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.count) { return; }
    var v = vel[i].xyz;
    v.y -= sim.gravity * sim.dt;
    v += mouseForce(pos[i].xyz, sim) * sim.dt;
    let p = pos[i].xyz + v * sim.dt;
    predicted[i] = vec4<f32>(p, 1.0);
}`;

const CLEAR_GRID_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> cellCount: array<atomic<u32>>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let c = gid.x + gid.y * ng.x * ${WORKGROUP_SIZE}u;
    if (c >= arrayLength(&cellCount)) { return; }
    atomicStore(&cellCount[c], 0u);
}`;

const BUILD_GRID_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> predicted: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> cellParticles: array<u32>;
@group(0) @binding(3) var<uniform> grid: Grid;
@group(0) @binding(4) var<uniform> sim: Sim;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.count) { return; }
    let cell = cellLinear(cellCoordOf(predicted[i].xyz, grid), grid);
    let slot = atomicAdd(&cellCount[cell], 1u);
    if (slot < grid.dim.w) {
        cellParticles[cell * grid.dim.w + slot] = i;
    }
}`;

// λ_i = −C_i / (|∇_i C|² + Σ_j |∇_j C|² + ε)
const LAMBDA_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> predicted: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> cellParticles: array<u32>;
@group(0) @binding(3) var<uniform> grid: Grid;
@group(0) @binding(4) var<uniform> sim: Sim;
@group(0) @binding(5) var<storage, read_write> lambda: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.count) { return; }
    let pi = predicted[i].xyz;
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
        let n = min(atomicLoad(&cellCount[cell]), grid.dim.w);
        for (var s = 0u; s < n; s = s + 1u) {
            let j = cellParticles[cell * grid.dim.w + s];
            let r = pi - predicted[j].xyz;
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
    lambda[i] = -ci / denom;
}`;

// Δp_i = (1/ρ0) Σ_j (λ_i + λ_j + s_corr) ∇W_spiky
const DELTA_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> predicted: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> cellParticles: array<u32>;
@group(0) @binding(3) var<uniform> grid: Grid;
@group(0) @binding(4) var<uniform> sim: Sim;
@group(0) @binding(5) var<storage, read> lambda: array<f32>;
@group(0) @binding(6) var<storage, read_write> delta: array<vec4<f32>>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.count) { return; }
    let pi = predicted[i].xyz;
    let li = lambda[i];
    let base = cellCoordOf(pi, grid);

    var dp = vec3<f32>(0.0);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cc = base + vec3<i32>(dx, dy, dz);
        if (any(cc < vec3<i32>(0)) || any(cc >= vec3<i32>(grid.dim.xyz))) { continue; }
        let cell = cellLinear(cc, grid);
        let n = min(atomicLoad(&cellCount[cell]), grid.dim.w);
        for (var s = 0u; s < n; s = s + 1u) {
            let j = cellParticles[cell * grid.dim.w + s];
            if (j == i) { continue; }
            let r = pi - predicted[j].xyz;
            let r2 = dot(r, r);
            if (r2 < sim.h2 && r2 > 1e-9) {
                let w = poly6(r2, sim);
                let scorr = -sim.scorrK * pow(w * sim.scorrInvWdq, sim.scorrN);
                dp += (li + lambda[j] + scorr) * spikyGradient(r, sqrt(r2), sim);
            }
        }
    }}}

    delta[i] = vec4<f32>(dp / sim.restDensity, 0.0);
}`;

const APPLY_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> predicted: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> delta: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> sim: Sim;
@group(0) @binding(3) var<storage, read_write> escaped: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.count) { return; }
    var p = predicted[i].xyz + delta[i].xyz;
    var esc = escaped[i];

    // Particles that have left through the hole are free; only those still
    // inside are confined by the capsule wall.
    if (sim.containerMode == 1u && esc == 0u) {
        // Project onto the capsule's inner surface along the radial direction
        // from the nearest point on its core segment. The rounded boundary
        // leaves no flat faces or sharp edges for particles to align against.
        let a = sim.capsuleA.xyz;
        let b = sim.capsuleB.xyz;
        let r = sim.capsuleA.w;
        let ba = b - a;
        let hh = clamp(dot(p - a, ba) / dot(ba, ba), 0.0, 1.0);
        let axisPt = a + ba * hh;
        let radial = p - axisPt;
        let dist = length(radial);
        if (dist > r) {
            // Outside the wall: escape through any hole, otherwise bounce back in.
            var inHole = false;
            for (var k = 0u; k < sim.holeCount; k = k + 1u) {
                if (distance(p, sim.holes[k].xyz) < sim.holes[k].w) {
                    inHole = true;
                    break;
                }
            }
            if (inHole) {
                esc = 1u;
            } else {
                p = axisPt + radial * (r / max(dist, 1e-6));
            }
        }
    } else if (sim.containerMode == 2u) {
        // Closed axis-aligned box container (no holes).
        p = clamp(p, sim.boxMin.xyz, sim.boxMax.xyz);
        p = obstacleResolve(p, sim);
    }

    // Ground floor (escaped liquid lands here once the tank is breached).
    p.y = max(p.y, sim.capsuleB.w);
    // Safety: never leave the neighbour-grid domain.
    p = clamp(p, sim.boundsMin.xyz, sim.boundsMax.xyz);
    predicted[i] = vec4<f32>(p, 1.0);
    escaped[i] = esc;
}`;

const FINALIZE_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> predicted: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> sim: Sim;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.count) { return; }
    let xnew = predicted[i].xyz;
    let v = (xnew - pos[i].xyz) / sim.dt;
    pos[i] = vec4<f32>(xnew, 1.0);
    vel[i] = vec4<f32>(v, 0.0);
}`;

// XSPH viscosity + speed readout for colouring. Reuses the per-frame grid
// (positions now equal the finalised predicted positions).
const VISCOSITY_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> cellParticles: array<u32>;
@group(0) @binding(4) var<uniform> grid: Grid;
@group(0) @binding(5) var<uniform> sim: Sim;
@group(0) @binding(6) var<storage, read_write> dbg: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sim.count) { return; }
    let pi = pos[i].xyz;
    let vi = vel[i].xyz;
    let base = cellCoordOf(pi, grid);

    var dv = vec3<f32>(0.0);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cc = base + vec3<i32>(dx, dy, dz);
        if (any(cc < vec3<i32>(0)) || any(cc >= vec3<i32>(grid.dim.xyz))) { continue; }
        let cell = cellLinear(cc, grid);
        let n = min(atomicLoad(&cellCount[cell]), grid.dim.w);
        for (var s = 0u; s < n; s = s + 1u) {
            let j = cellParticles[cell * grid.dim.w + s];
            let r = pi - pos[j].xyz;
            let r2 = dot(r, r);
            if (r2 < sim.h2) {
                dv += (vel[j].xyz - vi) * poly6(r2, sim);
            }
        }
    }}}

    let v = vi + (sim.viscosity / sim.restDensity) * dv;
    vel[i] = vec4<f32>(v, 0.0);
    dbg[i] = length(v);
}`;

export function createFluidSim(engine: EngineContext, options: FluidSimOptions = {}): FluidSim {
    const device = engine._device;
    const count = options.count ?? 30000;
    const particleRadius = options.particleRadius ?? 0.08;
    const spawnMin = options.spawnMin ?? [-2, 6, -2];
    const spawnMax = options.spawnMax ?? [2, 12, 2];
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

    // Rest density: spawn number density (particles / spawn volume) × scale.
    const spawnVol = Math.max(
        1e-6,
        (spawnMax[0] - spawnMin[0]) * (spawnMax[1] - spawnMin[1]) * (spawnMax[2] - spawnMin[2]),
    );
    const restDensity = options.restDensity ?? (count / spawnVol) * restDensityScale;

    // Grid derived from the bounds so cells tightly cover the active region.
    const gridDim: [number, number, number] = [
        Math.max(1, Math.ceil((boundsMax[0] - boundsMin[0]) / h)),
        Math.max(1, Math.ceil((boundsMax[1] - boundsMin[1]) / h)),
        Math.max(1, Math.ceil((boundsMax[2] - boundsMin[2]) / h)),
    ];
    const numCells = gridDim[0] * gridDim[1] * gridDim[2];

    // Kernel coefficients (mass = 1).
    const h2 = h * h;
    const poly6Coef = 315 / (64 * Math.PI * Math.pow(h, 9));
    const spikyGradCoef = -45 / (Math.PI * Math.pow(h, 6));
    const dq = 0.2 * h;
    const wDq = poly6Coef * Math.pow(h2 - dq * dq, 3);
    const scorrInvWdq = 1 / wDq;
    const scorrN = 4;

    // ── Buffers ──────────────────────────────────────────────────────
    const positionBuffer = device.createBuffer({ label: "fluid-positions", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const velocityBuffer = device.createBuffer({ label: "fluid-velocities", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const predictedBuffer = device.createBuffer({ label: "fluid-predicted", size: count * 16, usage: GPUBufferUsage.STORAGE });
    const lambdaBuffer = device.createBuffer({ label: "fluid-lambda", size: count * 4, usage: GPUBufferUsage.STORAGE });
    const deltaBuffer = device.createBuffer({ label: "fluid-delta", size: count * 16, usage: GPUBufferUsage.STORAGE });
    const debugBuffer = device.createBuffer({ label: "fluid-debug", size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const escapedBuffer = device.createBuffer({ label: "fluid-escaped", size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const cellCountBuffer = device.createBuffer({ label: "fluid-cell-count", size: numCells * 4, usage: GPUBufferUsage.STORAGE });
    const cellParticlesBuffer = device.createBuffer({ label: "fluid-cell-particles", size: numCells * maxPerCell * 4, usage: GPUBufferUsage.STORAGE });
    const simBuffer = device.createBuffer({ label: "fluid-sim", size: SIM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const gridBuffer = device.createBuffer({ label: "fluid-grid", size: GRID_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

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
        device.queue.writeBuffer(escapedBuffer, 0, new Uint32Array(count));
    }
    seed();

    // ── Pipelines ────────────────────────────────────────────────────
    function computePipeline(label: string, code: string): GPUComputePipeline {
        return device.createComputePipeline({ label, layout: "auto", compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" } });
    }
    const predictPipeline = computePipeline("fluid-predict", PREDICT_WGSL);
    const clearGridPipeline = computePipeline("fluid-clear-grid", CLEAR_GRID_WGSL);
    const buildGridPipeline = computePipeline("fluid-build-grid", BUILD_GRID_WGSL);
    const lambdaPipeline = computePipeline("fluid-lambda", LAMBDA_WGSL);
    const deltaPipeline = computePipeline("fluid-delta", DELTA_WGSL);
    const applyPipeline = computePipeline("fluid-apply", APPLY_WGSL);
    const finalizePipeline = computePipeline("fluid-finalize", FINALIZE_WGSL);
    const viscosityPipeline = computePipeline("fluid-viscosity", VISCOSITY_WGSL);

    const predictBG = device.createBindGroup({
        layout: predictPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: velocityBuffer } },
            { binding: 2, resource: { buffer: predictedBuffer } },
            { binding: 3, resource: { buffer: simBuffer } },
        ],
    });
    const clearGridBG = device.createBindGroup({
        layout: clearGridPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: cellCountBuffer } }],
    });
    const buildGridBG = device.createBindGroup({
        layout: buildGridPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: predictedBuffer } },
            { binding: 1, resource: { buffer: cellCountBuffer } },
            { binding: 2, resource: { buffer: cellParticlesBuffer } },
            { binding: 3, resource: { buffer: gridBuffer } },
            { binding: 4, resource: { buffer: simBuffer } },
        ],
    });
    const lambdaBG = device.createBindGroup({
        layout: lambdaPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: predictedBuffer } },
            { binding: 1, resource: { buffer: cellCountBuffer } },
            { binding: 2, resource: { buffer: cellParticlesBuffer } },
            { binding: 3, resource: { buffer: gridBuffer } },
            { binding: 4, resource: { buffer: simBuffer } },
            { binding: 5, resource: { buffer: lambdaBuffer } },
        ],
    });
    const deltaBG = device.createBindGroup({
        layout: deltaPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: predictedBuffer } },
            { binding: 1, resource: { buffer: cellCountBuffer } },
            { binding: 2, resource: { buffer: cellParticlesBuffer } },
            { binding: 3, resource: { buffer: gridBuffer } },
            { binding: 4, resource: { buffer: simBuffer } },
            { binding: 5, resource: { buffer: lambdaBuffer } },
            { binding: 6, resource: { buffer: deltaBuffer } },
        ],
    });
    const applyBG = device.createBindGroup({
        layout: applyPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: predictedBuffer } },
            { binding: 1, resource: { buffer: deltaBuffer } },
            { binding: 2, resource: { buffer: simBuffer } },
            { binding: 3, resource: { buffer: escapedBuffer } },
        ],
    });
    const finalizeBG = device.createBindGroup({
        layout: finalizePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: velocityBuffer } },
            { binding: 2, resource: { buffer: predictedBuffer } },
            { binding: 3, resource: { buffer: simBuffer } },
        ],
    });
    const viscosityBG = device.createBindGroup({
        layout: viscosityPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: velocityBuffer } },
            { binding: 2, resource: { buffer: cellCountBuffer } },
            { binding: 3, resource: { buffer: cellParticlesBuffer } },
            { binding: 4, resource: { buffer: gridBuffer } },
            { binding: 5, resource: { buffer: simBuffer } },
            { binding: 6, resource: { buffer: debugBuffer } },
        ],
    });

    const particleGroups = Math.ceil(count / WORKGROUP_SIZE);
    const cellGroups = Math.ceil(numCells / WORKGROUP_SIZE);

    function dispatch(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline, bg: GPUBindGroup, groups: number): void {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        if (groups > MAX_WORKGROUPS) {
            pass.dispatchWorkgroups(MAX_WORKGROUPS, Math.ceil(groups / MAX_WORKGROUPS), 1);
        } else {
            pass.dispatchWorkgroups(groups);
        }
        pass.end();
    }

    // Hole ring-buffer state (CPU side; written into the holes uniform array).
    let holeWriteSlot = 0;
    let holeActiveCount = 0;

    return {
        count,
        particleRadius,
        positionBuffer,
        debugBuffer,
        // Speed colour normalisation: typical lively splash speed ≈ 5 world units/s.
        debugNorm: 1 / 5,
        step(encoder: GPUCommandEncoder, dt: number): void {
            // finalize divides by dt; a zero/NaN dt (e.g. the very first frame)
            // would poison every position with NaN, so skip such frames.
            if (!(dt > 0)) {
                return;
            }
            simF32[0] = dt;
            device.queue.writeBuffer(simBuffer, 0, simData);

            dispatch(encoder, "fluid-predict", predictPipeline, predictBG, particleGroups);
            dispatch(encoder, "fluid-clear-grid", clearGridPipeline, clearGridBG, cellGroups);
            dispatch(encoder, "fluid-build-grid", buildGridPipeline, buildGridBG, particleGroups);
            for (let it = 0; it < iterationsMut; it++) {
                dispatch(encoder, "fluid-lambda", lambdaPipeline, lambdaBG, particleGroups);
                dispatch(encoder, "fluid-delta", deltaPipeline, deltaBG, particleGroups);
                dispatch(encoder, "fluid-apply", applyPipeline, applyBG, particleGroups);
            }
            dispatch(encoder, "fluid-finalize", finalizePipeline, finalizeBG, particleGroups);
            dispatch(encoder, "fluid-viscosity", viscosityPipeline, viscosityBG, particleGroups);
        },
        reset(): void {
            holeWriteSlot = 0;
            holeActiveCount = 0;
            simU32[15] = 0;
            simF32.fill(0, HOLE_BASE_F32, HOLE_BASE_F32 + MAX_HOLES * 4);
            seed();
        },
        setParam(key: string, value: number): void {
            switch (key) {
                case "gravity": simF32[1] = value; break;
                case "restDensity": simF32[2] = value; break;
                case "relaxation": simF32[7] = value; break;
                case "scorr": simF32[8] = value; break;
                case "viscosity": simF32[11] = value; break;
                case "boundaryDensity": simF32[14] = value; break;
                case "iterations": iterationsMut = Math.max(1, Math.round(value)); break;
            }
        },
        setContainer(mode: number, min: [number, number, number], max: [number, number, number]): void {
            simU32[13] = mode;
            simF32[BOX_BASE_F32] = min[0];
            simF32[BOX_BASE_F32 + 1] = min[1];
            simF32[BOX_BASE_F32 + 2] = min[2];
            simF32[BOX_BASE_F32 + 4] = max[0];
            simF32[BOX_BASE_F32 + 5] = max[1];
            simF32[BOX_BASE_F32 + 6] = max[2];
        },
        setForce(origin: [number, number, number], dir: [number, number, number], push: [number, number, number], radius: number, accel: number): void {
            simF32[FORCE_BASE_F32] = origin[0];
            simF32[FORCE_BASE_F32 + 1] = origin[1];
            simF32[FORCE_BASE_F32 + 2] = origin[2];
            simF32[FORCE_BASE_F32 + 3] = radius;
            simF32[FORCE_BASE_F32 + 4] = dir[0];
            simF32[FORCE_BASE_F32 + 5] = dir[1];
            simF32[FORCE_BASE_F32 + 6] = dir[2];
            simF32[FORCE_BASE_F32 + 7] = accel;
            simF32[FORCE_BASE_F32 + 8] = push[0];
            simF32[FORCE_BASE_F32 + 9] = push[1];
            simF32[FORCE_BASE_F32 + 10] = push[2];
        },
        setObstacle(active: boolean, center: [number, number], halfWidth: number, halfThickness: number, angle: number, omega: number): void {
            simF32[OBS_BASE_F32] = center[0];
            simF32[OBS_BASE_F32 + 1] = center[1];
            simF32[OBS_BASE_F32 + 2] = halfWidth;
            simF32[OBS_BASE_F32 + 3] = halfThickness;
            // Babylon's rotation.y about +Y maps local +Z → world (sinθ,0,cosθ)
            // (left-handed), the mirror of the textbook +sinθ used by the slab
            // maths. Negate sin so the collision slab aligns with the visible
            // mesh, and negate omega so the paddle's surface velocity pushes in
            // the direction the mesh actually spins.
            simF32[OBS_BASE_F32 + 4] = Math.cos(angle);
            simF32[OBS_BASE_F32 + 5] = -Math.sin(angle);
            simF32[OBS_BASE_F32 + 6] = -omega;
            simF32[OBS_BASE_F32 + 7] = active ? 1 : 0;
        },
        addHole(center: [number, number, number], radius: number): void {
            const o = HOLE_BASE_F32 + holeWriteSlot * 4;
            simF32[o] = center[0];
            simF32[o + 1] = center[1];
            simF32[o + 2] = center[2];
            simF32[o + 3] = radius;
            holeWriteSlot = (holeWriteSlot + 1) % MAX_HOLES;
            holeActiveCount = Math.min(holeActiveCount + 1, MAX_HOLES);
            simU32[15] = holeActiveCount;
        },
        dispose(): void {
            positionBuffer.destroy();
            velocityBuffer.destroy();
            predictedBuffer.destroy();
            lambdaBuffer.destroy();
            deltaBuffer.destroy();
            debugBuffer.destroy();
            escapedBuffer.destroy();
            cellCountBuffer.destroy();
            cellParticlesBuffer.destroy();
            simBuffer.destroy();
            gridBuffer.destroy();
        },
    };
}
