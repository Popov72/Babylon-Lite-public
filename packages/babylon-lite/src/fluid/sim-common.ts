// Shared contract + helpers for the demo-local GPU fluid backends.
//
// Both `createPbfSim` (Position Based Fluids) and `createMlsMpmSim` (MLS-MPM)
// implement the same `FluidSim` interface so the renderer and the demo can swap
// backends at runtime. This module owns everything the two solvers have in
// common: the interface, the base options, the per-demo scene-SDF injection
// contract, the recirculating-emitter config + packing, and the generic SDF
// normal WGSL snippet both solvers inject.
//
// Foam reference (the shared diffuse-particle model, `FOAM_COMMON_WGSL`): Ihmsen et al. 2012,
// "Unified spray, foam and air bubbles for particle-based fluids" —
// https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf

/** Opt-in GPU timing hook. The concrete implementation lives in the lab (see
 *  lab/lite/src/demos/fluid/gpu-profiler.ts); the package only references this
 *  type, so no profiler code is bundled unless an app opts in. */
export interface FluidProfiler {
    /** timestampWrites for a pass tagged `stage` (allocates a begin/end query
     *  pair), or undefined to disable timing for this pass. The returned object is
     *  structurally valid for BOTH GPUComputePassDescriptor.timestampWrites and
     *  GPURenderPassDescriptor.timestampWrites. */
    pass(stage: string): { querySet: GPUQuerySet; beginningOfPassWriteIndex: number; endOfPassWriteIndex: number } | undefined;
}

/** vec4<f32>-per-particle position + a per-particle speed the renderer reads.
 *  Both backends expose their state through these buffers in WORLD units. */
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
    /** vec4<f32>-per-particle WORLD velocity buffer (STORAGE), same indexing as
     *  `positionBuffer`. Exposed for coupling (e.g. drag on a floating body). */
    readonly velocityBuffer: GPUBuffer;
    /** f32-per-particle speed (Phase 3 debug). Read by the renderer to tint by motion. */
    readonly debugBuffer: GPUBuffer;
    /** Normalisation reciprocal for `debugBuffer` (≈ 1 / typical max speed). */
    readonly debugNorm: number;
    /** Estimated total bytes of the GPU buffers this backend owns (particle state,
     *  neighbour grid, render positions, foam pool + uniforms). Re-read live: the
     *  foam pool is allocated lazily and re-sized, so the value grows once foam is on. */
    readonly gpuBytes: number;
    /** Encode one simulation step into `encoder`. `dt` is seconds. */
    step(encoder: GPUCommandEncoder, dt: number): void;
    /** Re-seed all particles into the spawn box with zero velocity. */
    reset(): void;
    /** Live-update a named simulation parameter (for the demo's tuning UI). */
    setParam(key: string, value: number): void;
    /** Optional PB-MPM material selector: 0 liquid, 1 elastic, 2 sand, 3 viscoelastic. */
    setMaterial?(material: number): void;
    /** Inject the per-demo scene SDF used for collision (or null to disable it).
     *  Rebuilds the confinement pipeline; compiled variants are cached by source. */
    setSceneSdf(spec: SceneSdfSpec | null): void;
    /** Configure recirculating jet emitters (or null to disable). Particles inside
     *  the intake box are probabilistically relaunched from a nozzle each step. */
    setEmitters(cfg: EmitterConfig | null): void;
    /** Set the spawn box used by `reset()` to re-seed particles. `accept`, when
     *  provided, restricts seeding to positions where it returns true (CPU
     *  reject-sampling), so particles fit a non-box container shape. */
    setSpawn(min: [number, number, number], max: [number, number, number], accept?: ((x: number, y: number, z: number) => boolean) | null): void;
    /** Set the start-of-sim warm-up length (frames) used by `reset()`/`seed()` to
     *  gradually release particles instead of all at once. Optional: only MLS-MPM
     *  implements it (a dense open-shelf seed otherwise spikes and sprays); PBF omits
     *  it and callers no-op via `?.`. */
    setWarmup?(frames: number): void;
    /** Inject a generic external force field (or null to disable it). Rebuilds the
     *  integration pass; compiled variants are cached by source. When null the force
     *  path costs nothing: no force buffer is bound and the injected default force is
     *  a no-op the compiler folds away. Mirrors `setSceneSdf`. */
    setForceField(spec: ForceFieldSpec | null): void;
    /** Enable/disable the diffuse-particle (spray/foam/bubbles) system. Optional: both the
     *  PBF and MLS-MPM backends implement it. Pass a config to enable (the pool + the compute
     *  passes are allocated/compiled lazily on the first call), or null to disable (the foam
     *  passes are skipped, nothing renders). Generation follows the Ihmsen 2012 potentials. */
    setFoam?(cfg: FoamConfig | null): void;
    /** Opt-in GPU timing hook (optional, like setFoam). Pass a profiler to tag this
     *  backend's compute passes with timestampWrites, or null to turn timing off. The
     *  profiler machinery lives entirely in the app (lab); the package only references
     *  the {@link FluidProfiler} type, so no profiler code is bundled unless opted in. */
    setProfiler?(p: FluidProfiler | null): void;
    /** Diffuse-particle pool exposed to the foam renderer. Present only once foam has
     *  been enabled; holds the ring buffer, the atomic write-head, and the slot count. */
    readonly diffuse?: DiffusePool;
    dispose(): void;
}

/** GPU-resident diffuse-particle pool (spray/foam/bubbles) produced by a fluid backend
 *  and consumed by the foam renderer. */
export interface DiffusePool {
    /** Storage buffer of `capacity` slots, 32 bytes each: two vec4 packed as
     *  p = (pos.xyz, lifetime) and v = (vel.xyz, kind). A slot is dead when lifetime is
     *  at or below zero. */
    readonly buffer: GPUBuffer;
    /** Small storage buffer holding the atomic ring write-head at u32 index 0. */
    readonly headBuffer: GPUBuffer;
    /** Number of slots in the ring buffer. */
    readonly capacity: number;
}

/** Diffuse-particle (foam) tuning knobs. All optional, with paper defaults. Drives the
 *  Ihmsen 2012 generation potentials (kTa/kWc) plus the shared pool/advection knobs; the
 *  classification, advection and rendering are all method-agnostic. */
export interface FoamConfig {
    /** Trapped-air generation rate (max samples per second per fluid particle). Default 40. */
    kTa?: number;
    /** Wave-crest generation rate (max samples per second per fluid particle). Default 40. */
    kWc?: number;
    /** Bubble buoyancy coefficient (fraction of gravity applied upward). Default 0.8. */
    kb?: number;
    /** Bubble drag coefficient toward the local fluid velocity, in the range 0 to 1. Default 0.5. */
    kd?: number;
    /** Emission cylinder radius rV (world units). Default: the fluid particle radius. */
    rv?: number;
    /** Minimum foam lifetime in seconds. Default 0.3. */
    tMin?: number;
    /** Maximum foam lifetime in seconds. Default 2.0. */
    tMax?: number;
    /** Pool capacity as a multiple of the fluid particle count. Default 3. */
    poolScale?: number;
    /** Hard ceiling on the pool capacity, in slots. Default 1_500_000. */
    poolCapMax?: number;
}

// Shared foam (diffuse-particle) WGSL, injected by BOTH backends. Defines the per-frame
// `Foam` UBO (tuning knobs), the 32-byte `Diffuse` slot struct the foam renderer reads,
// the radial hat kernel W (used by the PBF neighbour gathers), the [0,1] clamp map Φ, and
// a hash PRNG. The `Diffuse` slot layout is FIXED — the foam renderer reads the pool, so
// it must not be reordered. Both backends pack the `Foam` UBO identically (indices below),
// so its layout is shared too; extend only at the END.
//   FoamParams UBO layout (16 floats / 64 bytes):
//     [0..3]  tauTaMin, tauTaMax, tauWcMin, tauWcMax
//     [4..7]  tauKMin, tauKMax, kTa, kWc
//     [8..11] kb, kd, rv, _pad0
//     [12..15] tMin, tMax, frameSeed(u32), _pad1
export const FOAM_BYTES = 64;

export const FOAM_COMMON_WGSL = /* wgsl */ `
struct Foam {
    tauTaMin: f32, tauTaMax: f32, tauWcMin: f32, tauWcMax: f32,
    tauKMin: f32, tauKMax: f32, kTa: f32, kWc: f32,
    kb: f32, kd: f32, rv: f32, _pad0: f32,
    tMin: f32, tMax: f32, frameSeed: u32, _pad1: u32,
};
struct Diffuse { p: vec4<f32>, v: vec4<f32> };
// Radial hat weight W(r,h) = 1 - r/h for r<=h (better near a free surface than poly6).
fn wHat(rlen: f32, h: f32) -> f32 { return max(0.0, 1.0 - rlen / h); }
// Clamp/normalise Φ(I, lo, hi) = (min(I,hi) - min(I,lo)) / (hi - lo) in [0,1].
fn phi(x: f32, lo: f32, hi: f32) -> f32 { return clamp((min(x, hi) - min(x, lo)) / (hi - lo), 0.0, 1.0); }
fn fHashU(x0: u32) -> u32 { var h = x0; h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u; return h; }
fn fRnd(x: u32) -> f32 { return f32(fHashU(x)) / 4294967296.0; }
`;

/** Fields common to BOTH backends. Method-specific tuning lives in the per-solver
 *  `PbfOptions` / `MlsMpmOptions`, which each extend this. */
export interface FluidSimBaseOptions {
    /** Particle count. */
    count?: number;
    /** Render/visual particle radius in world units. */
    particleRadius?: number;
    /** Axis-aligned spawn box min the particles are seeded into. */
    spawnMin?: [number, number, number];
    /** Axis-aligned spawn box max the particles are seeded into. */
    spawnMax?: [number, number, number];
    /** Gravity acceleration (m/s²). Default 9.8. */
    gravity?: number;
    /** Number of frames over which `reset()`/`seed()` gradually releases particles
     *  (a start-of-sim warm-up so a dense seed fills in without a pressure spike).
     *  0 (default) releases everything immediately. MLS-MPM only; PBF ignores it. */
    warmupFrames?: number;
}

// Per-demo scene SDF injection. The demo supplies a `struct SceneSdfParams {…}`
// declaration and an `fn sceneSdf(p) -> f32` (positive inside the fluid domain,
// negative when penetrating), both reading the injected `sceneSdfParams` uniform.
// The generic `sceneNormal` (central differences) works for any injected SDF.
export interface SceneSdfSpec {
    /** WGSL `struct SceneSdfParams { … };` matching the CPU packing of `buffer`. */
    struct: string;
    /** WGSL `fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 { … }` reading `sceneSdfParams`.
     *  `dt` is a small time perturbation used to derive moving-boundary velocity
     *  from -∂sceneSdf/∂t; static scenes ignore it. */
    sdf: string;
    /** Uniform buffer holding the packed `SceneSdfParams`, owned by the demo. */
    buffer: GPUBuffer;
    /** Optional baked SDF grid sampled by the scene's `sceneSdf` WGSL: a STORAGE buffer holding a
     *  flat Float32Array of signed distances (NEGATIVE inside the solid, POSITIVE outside), x-fastest:
     *  idx = i + dims.x*(j + dims.y*k). When present the sim binds it as `sceneSdfGrid` and injects the
     *  shared `SCENE_SDF_GRID_WGSL` sampler; the scene packs the grid's origin/cellSize/dims into `buffer`
     *  (the UBO) and calls `sampleSdfGrid(pt, origin, invCell, dims)` from its `sceneSdf`. Absent → the
     *  analytic-only path (unchanged). */
    sdfGrid?: GPUBuffer;
    /** Confinement style — **MLS-MPM only** (the PBF/SPH backend ignores this and
     *  always confines per-particle against the SDF). For MLS-MPM: when true (default)
     *  the container is CLOSED — confined at the sim grid with a separating wall + a
     *  G2P half-nudge (avoids flat-wall gluing on box walls). When false, per-particle
     *  full push-out + restitution reflect against the SDF (for thin curved shells like
     *  the capsule that the coarse grid wall would miss). */
    gridConfine?: boolean;
}

export const SCENE_NORMAL_WGSL = /* wgsl */ `
fn sceneNormal(pt: vec3<f32>, dt: f32) -> vec3<f32> {
    let e = vec2<f32>(0.01, 0.0);
    return normalize(vec3<f32>(
        sceneSdf(pt + e.xyy, dt) - sceneSdf(pt - e.xyy, dt),
        sceneSdf(pt + e.yxy, dt) - sceneSdf(pt - e.yxy, dt),
        sceneSdf(pt + e.yyx, dt) - sceneSdf(pt - e.yyx, dt)));
}`;

// Trilinear sampler for a baked SDF grid (see `SceneSdfSpec.sdfGrid`). Injected by the
// solvers — BEFORE the scene's own `sceneSdf` — only when `sdfGrid` is set, alongside the
// `@group(0) @binding(N) var<storage, read> sceneSdfGrid: array<f32>;` declaration. The
// scene's `sceneSdf` calls `sampleSdfGrid(pt, origin, invCell, dims)` to read a signed
// distance (NEGATIVE inside the solid). Clamp-to-edge outside the grid is intentional: the
// grid is padded and the scene intersects it with a domain box, so edge nodes read as
// far-outside positive.
export const SCENE_SDF_GRID_WGSL = /* wgsl */ `
fn sdfGridLoad(i: i32, j: i32, k: i32, d: vec3<i32>) -> f32 {
    let c = clamp(vec3<i32>(i,j,k), vec3<i32>(0), d - vec3<i32>(1));
    return sceneSdfGrid[c.x + d.x * (c.y + d.y * c.z)];
}
fn sampleSdfGrid(pt: vec3<f32>, origin: vec3<f32>, invCell: f32, dims: vec3<i32>) -> f32 {
    let g = (pt - origin) * invCell;
    let b = floor(g);
    let f = g - b;
    let i = i32(b.x); let j = i32(b.y); let k = i32(b.z);
    let c000 = sdfGridLoad(i,   j,   k,   dims); let c100 = sdfGridLoad(i+1, j,   k,   dims);
    let c010 = sdfGridLoad(i,   j+1, k,   dims); let c110 = sdfGridLoad(i+1, j+1, k,   dims);
    let c001 = sdfGridLoad(i,   j,   k+1, dims); let c101 = sdfGridLoad(i+1, j,   k+1, dims);
    let c011 = sdfGridLoad(i,   j+1, k+1, dims); let c111 = sdfGridLoad(i+1, j+1, k+1, dims);
    let x00 = mix(c000, c100, f.x); let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x); let x11 = mix(c011, c111, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}`;

// Generic external force-field injection. The caller supplies a `struct
// ForceFieldParams {…}` declaration and an `fn externalForce(pos, vel, dt) -> vec3`
// that reads the injected `forceFieldParams` uniform and returns a velocity DELTA
// (already dt-scaled) to add to the particle this step. Mirrors `SceneSdfSpec`.
// When no spec is injected the solver uses `DEFAULT_FORCE_FIELD_WGSL` (a no-op)
// and binds no force buffer, so the force path costs nothing.
export interface ForceFieldSpec {
    /** WGSL `struct ForceFieldParams { … };` matching the CPU packing of `buffer`. */
    struct: string;
    /** WGSL `fn externalForce(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> vec3<f32> { … }`
     *  reading `forceFieldParams`. Returns a velocity DELTA to add this step (already dt-scaled). */
    wgsl: string;
    /** Uniform buffer holding the packed `ForceFieldParams`, owned by the caller. */
    buffer: GPUBuffer;
}

export const DEFAULT_FORCE_FIELD_WGSL = "fn externalForce(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> vec3<f32> { return vec3<f32>(0.0); }";

// ── Generic particle emitter / recycling ─────────────────────────────
// Fixed particle pool: "emitting" recycles particles rather than adding them.
// A compute pass run first each step relaunches particles that sit inside the
// pump-intake box, probabilistically (rand < rate·dt, to throttle so jets are
// continuous streams), at a random emitter nozzle with its jet velocity.
export const MAX_EMITTERS = 16;

export interface EmitterConfig {
    /** Jet nozzles. Each relaunches recycled particles at `dir`·`speed` from
     *  `pos` (± a `radius` position jitter). */
    emitters: { pos: [number, number, number]; dir: [number, number, number]; speed: number; radius: number }[];
    /** Axis-aligned pump-intake box min: particles inside are eligible to recycle. */
    intakeMin: [number, number, number];
    /** Axis-aligned pump-intake box max. */
    intakeMax: [number, number, number];
    /** Per-second probability an eligible particle relaunches (throttles the jets). */
    rate: number;
    /** Random velocity spread added at launch (world units/s). Default 0. */
    spread?: number;
    /** When positive, the LAST emitter becomes a DEDICATED stream fed only by particle indices
     *  [0, fixedStreamCount); the other emitters serve indices [fixedStreamCount, count).
     *  This gives that nozzle a roughly fixed-size stream INDEPENDENT of the total particle
     *  count (e.g. a wheel-driving jet that looks the same at 40k and 200k). Default 0 (off:
     *  all emitters share every recycled particle, the original behaviour). */
    fixedStreamCount?: number;
    /** Drain height for the fixed stream (only meaningful with fixedStreamCount positive). The
     *  fixed-stream particles form a self-contained TIGHT LOOP: the instant one sinks below this
     *  world-Y it relaunches DETERMINISTICALLY at the last emitter, never touching the shared
     *  pump-intake or the main pool. So ~fixedStreamCount particles are always in flight over the
     *  target, at a cadence set only by gravity + geometry — fully independent of the total count
     *  or how deep the main pool is. Default 0 (fixed particles fall through to the shared intake). */
    fixedStreamDrainY?: number;
}

// Emitters-UBO float layout: head, head2, intakeMin, intakeMax, then
// MAX_EMITTERS × (pos+radius, dir+speed). packEmitters writes everything except
// the sim-owned fields: head2.x (particle count) and head.z / head2.y (seed, dt).
// head2.z = fixedStreamCount (dedicated last-emitter stream, 0 = off).
// head2.w = fixedStreamDrainY (fixed-stream tight-loop drain height).
export const EMITTERS_FLOATS = 16 + MAX_EMITTERS * 8;

export function packEmitters(data: Float32Array, cfg: EmitterConfig | null): void {
    // Preserve head2.x (particle count) written by the sim; clear the rest.
    const count = data[4];
    data.fill(0);
    data[4] = count!;
    if (!cfg || cfg.emitters.length === 0) {
        return;
    }
    const n = Math.min(cfg.emitters.length, MAX_EMITTERS);
    data[0] = n;
    data[1] = cfg.rate;
    // data[2] = seed and data[5] = dt are written per frame by the sim.
    data[3] = cfg.spread ?? 0;
    data[6] = cfg.fixedStreamCount ?? 0; // head2.z — dedicated-stream particle count (0 = off)
    data[7] = cfg.fixedStreamDrainY ?? 0; // head2.w — fixed-stream tight-loop drain height
    data[8] = cfg.intakeMin[0];
    data[9] = cfg.intakeMin[1];
    data[10] = cfg.intakeMin[2];
    data[12] = cfg.intakeMax[0];
    data[13] = cfg.intakeMax[1];
    data[14] = cfg.intakeMax[2];
    for (let k = 0; k < n; k++) {
        const o = 16 + k * 8;
        const e = cfg.emitters[k]!;
        data[o] = e.pos[0];
        data[o + 1] = e.pos[1];
        data[o + 2] = e.pos[2];
        data[o + 3] = e.radius;
        data[o + 4] = e.dir[0];
        data[o + 5] = e.dir[1];
        data[o + 6] = e.dir[2];
        data[o + 7] = e.speed;
    }
}
