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
    /** Compact list of live diffuse-particle slot indices. Present when active-particle
     *  processing is enabled. */
    readonly activeIndices?: GPUBuffer;
    /** Byte offset of the current compact list inside `activeIndices`. */
    readonly activeIndicesOffset?: number;
    /** Indirect draw arguments `[6, liveCount, 0, 0]` for the foam renderer. Present
     *  together with `activeIndices`. */
    readonly drawIndirect?: GPUBuffer;
}

/** Diffuse-particle (foam) tuning knobs. All optional, with paper defaults. Drives the
 *  Ihmsen 2012 generation potentials (kTa/kWc) plus the shared pool/advection knobs; the
 *  classification, advection and rendering are all method-agnostic. */
export interface FoamConfig {
    /** Track live diffuse-particle slots persistently so update and render passes
     *  visit only active particles. Default false. */
    activeParticles?: boolean;
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
    /** Optional hard ceiling on the pool capacity, in slots. Default: none — the pool is
     *  sized purely from `poolScale` × particle count, bounded only by the device's own
     *  storage-buffer and dispatch limits. */
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

export const FOAM_ACTIVE_HEADER_U32 = 64;

export function foamActiveListStride(capacity: number): number {
    return Math.ceil(capacity / 64) * 64;
}

export function foamActiveListOffset(capacity: number, side: number): number {
    return (FOAM_ACTIVE_HEADER_U32 + side * foamActiveListStride(capacity)) * 4;
}

export function foamActiveStateBytes(capacity: number): number {
    return (FOAM_ACTIVE_HEADER_U32 + foamActiveListStride(capacity) * 2 + capacity) * 4;
}

/** Builds update-dispatch arguments from the current persistent live list. */
export const FOAM_ACTIVE_PREPARE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> computeArgs: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let side = atomicLoad(&state[3]);
    let count = atomicLoad(&state[1u + side]);
    let groups = (count + 63u) / 64u;
    computeArgs[0] = min(groups, 65535u);
    computeArgs[1] = (groups + 65534u) / 65535u;
    computeArgs[2] = 1u;
}`;

/** Publishes the survivor list for rendering and the next frame, then clears the
 *  consumed list count so it can be reused as the following output list. */
export const FOAM_ACTIVE_FINISH_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> drawArgs: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let oldSide = atomicLoad(&state[3]);
    let newSide = 1u - oldSide;
    let count = atomicLoad(&state[1u + newSide]);
    drawArgs[0] = 6u;
    drawArgs[1] = count;
    drawArgs[2] = 0u;
    drawArgs[3] = 0u;
    atomicStore(&state[3], newSide);
    atomicStore(&state[1u + oldSide], 0u);
}`;

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
/** Triangles available to polygon emitters, shared across every emitter in the config. */
export const MAX_EMITTER_TRIS = 64;

/** Rejection-sampling attempts per particle when `setSpawn` carries an `accept` predicate.
 *  Shapes that fill little of their bounding box (two small prisms on a summit fill only a few
 *  percent of it) reject often, so this needs headroom — the seeders additionally fall back to
 *  the last ACCEPTED point rather than a rejected one, so exhausting it can never place a
 *  particle outside the container. */
export const SPAWN_ACCEPT_TRIES = 64;

export interface EmitterConfig {
    /** Jet nozzles. Each relaunches recycled particles at `dir`·`speed` from a random point in
     *  its spawn volume. That volume is a BOX: `halfExtents` when given, otherwise a cube of
     *  half-extent `radius` (so a nozzle stays a point-ish jet unless it opts in). A box emitter
     *  lets a source cover a real surface — e.g. the flat shelves on top of a waterfall — instead
     *  of pretending to be a point. `polygon` generalises that to an arbitrary outline. */
    emitters: {
        pos: [number, number, number];
        dir: [number, number, number];
        speed: number;
        radius: number;
        /** Per-axis half-extents (width/2, height/2, depth/2) of the spawn box, world units.
         *  Defaults to `(radius, radius, radius)`. */
        halfExtents?: [number, number, number];
        /** Spawn area as a simple closed polygon in the world XZ plane — the vertices only, with
         *  no repeated closing vertex; either winding works. When set it REPLACES the box's X/Z
         *  extents, so the source can match a real surface (e.g. a terrace on a rock) instead of
         *  the bounding box around it; `pos[0]`/`pos[2]` are then ignored. Height still comes from
         *  `halfExtents[1]` about `pos[1]`, which is what gives the outline its "small height".
         *
         *  Triangulated here on the CPU (ear clipping) and uploaded as an area-weighted triangle
         *  list, so the shader samples it uniformly in O(triangles) with no rejection sampling —
         *  rejection would both waste relaunches and bias density when the outline fills little of
         *  its bounding box. Polygons are shared out of a {@link MAX_EMITTER_TRIS} budget. */
        polygon?: [number, number][];
    }[];
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
// MAX_EMITTERS × (pos+radius, dir+speed, halfExtents+pad, triStart+triCount+pad2), then a
// shared triangle table of MAX_EMITTER_TRIS × (ax,az,bx,bz | cx,cz,cumArea,pad) for polygon
// emitters. packEmitters writes everything except the sim-owned fields: head2.x (particle
// count) and head.z / head2.y (seed, dt).
// head2.z = fixedStreamCount (dedicated last-emitter stream, 0 = off).
// head2.w = fixedStreamDrainY (fixed-stream tight-loop drain height).
export const EMITTERS_FLOATS = 16 + MAX_EMITTERS * 16 + MAX_EMITTER_TRIS * 8;

/** Twice the signed area of a closed polygon (positive when counter-clockwise in XZ). */
function polyArea2(poly: readonly [number, number][]): number {
    let a = 0;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        a += poly[j]![0] * poly[i]![1] - poly[i]![0] * poly[j]![1];
    }
    return a;
}

function inTriangle(px: number, py: number, ax: number, ay: number, bx: number, by: number, cx: number, cy: number): boolean {
    const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
    const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
    const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
    const neg = d1 < 0 || d2 < 0 || d3 < 0;
    const pos = d1 > 0 || d2 > 0 || d3 > 0;
    return !(neg && pos);
}

/** Ear-clipping triangulation of a SIMPLE polygon (no holes, no self-intersections), returning
 *  flat index triples into `poly`. Ear clipping rather than a triangle fan because a fan is only
 *  correct for convex outlines — a hand-drawn terrace is routinely concave, and a fan would then
 *  spawn particles outside the shape. Bails out returning what it has if the input turns out not
 *  to be simple, so bad authoring degrades instead of hanging. */
function triangulatePolygon(poly: readonly [number, number][]): number[] {
    const n = poly.length;
    if (n < 3) {
        return [];
    }
    // Work counter-clockwise so the convexity test has one consistent sign.
    const idx: number[] = [];
    for (let i = 0; i < n; i++) {
        idx.push(i);
    }
    if (polyArea2(poly) < 0) {
        idx.reverse();
    }
    const out: number[] = [];
    let guard = n * n + 8;
    while (idx.length > 3 && guard-- > 0) {
        let clipped = false;
        for (let k = 0; k < idx.length; k++) {
            const i0 = idx[(k + idx.length - 1) % idx.length]!;
            const i1 = idx[k]!;
            const i2 = idx[(k + 1) % idx.length]!;
            const [ax, ay] = poly[i0]!;
            const [bx, by] = poly[i1]!;
            const [cx, cy] = poly[i2]!;
            // Reflex corner (or collinear) — not an ear.
            if ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay) <= 0) {
                continue;
            }
            let contains = false;
            for (const m of idx) {
                if (m !== i0 && m !== i1 && m !== i2 && inTriangle(poly[m]![0], poly[m]![1], ax, ay, bx, by, cx, cy)) {
                    contains = true;
                    break;
                }
            }
            if (contains) {
                continue;
            }
            out.push(i0, i1, i2);
            idx.splice(k, 1);
            clipped = true;
            break;
        }
        if (!clipped) {
            break;
        }
    }
    if (idx.length === 3) {
        out.push(idx[0]!, idx[1]!, idx[2]!);
    }
    return out;
}

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
    const triBase = 16 + MAX_EMITTERS * 16;
    let triCursor = 0;
    for (let k = 0; k < n; k++) {
        const o = 16 + k * 16;
        const e = cfg.emitters[k]!;
        data[o] = e.pos[0];
        data[o + 1] = e.pos[1];
        data[o + 2] = e.pos[2];
        data[o + 3] = e.radius;
        data[o + 4] = e.dir[0];
        data[o + 5] = e.dir[1];
        data[o + 6] = e.dir[2];
        data[o + 7] = e.speed;
        // Spawn-box half-extents. A plain nozzle keeps the historical CUBE of half-extent
        // `radius`, so omitting halfExtents reproduces the original jitter exactly.
        const h = e.halfExtents;
        data[o + 8] = h ? h[0] : e.radius;
        data[o + 9] = h ? h[1] : e.radius;
        data[o + 10] = h ? h[2] : e.radius;
        // Polygon spawn area: triangulate, then store each triangle with the RUNNING FRACTION of
        // the outline's area it completes. The shader picks with a single uniform random against
        // those fractions, so big triangles are chosen proportionally more often and the outline
        // fills evenly — a uniform pick would crowd particles into the slivers.
        const poly = e.polygon;
        if (!poly || poly.length < 3) {
            continue;
        }
        const tris = triangulatePolygon(poly);
        const budget = Math.min(tris.length / 3, MAX_EMITTER_TRIS - triCursor);
        let total = 0;
        for (let t = 0; t < budget; t++) {
            const a = poly[tris[t * 3]!]!;
            const b = poly[tris[t * 3 + 1]!]!;
            const c = poly[tris[t * 3 + 2]!]!;
            total += Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2;
        }
        if (budget < 1 || total <= 0) {
            continue; // degenerate outline — fall back to the box extents above
        }
        const start = triCursor;
        let acc = 0;
        for (let t = 0; t < budget; t++) {
            const a = poly[tris[t * 3]!]!;
            const b = poly[tris[t * 3 + 1]!]!;
            const c = poly[tris[t * 3 + 2]!]!;
            acc += Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2;
            const to = triBase + (start + t) * 8;
            data[to] = a[0];
            data[to + 1] = a[1];
            data[to + 2] = b[0];
            data[to + 3] = b[1];
            data[to + 4] = c[0];
            data[to + 5] = c[1];
            // Last triangle pinned to exactly 1 so a random of 1.0 can never fall through.
            data[to + 6] = t === budget - 1 ? 1 : acc / total;
        }
        triCursor += budget;
        data[o + 12] = start;
        data[o + 13] = budget;
    }
}

/** Emitter UBO declarations for the backends' emit shaders. Shared so the three sims can never
 *  drift from each other or from {@link packEmitters} — the layout is written in one place only.
 *  On PB-MPM `intakeMin.w` additionally carries subDt. */
export const EMITTER_STRUCT_WGSL = /* wgsl */ `
struct Emitter {
    p: vec4<f32>,           // pos.xyz, radius
    d: vec4<f32>,           // dir.xyz, speed
    e: vec4<f32>,           // spawn-box half-extents.xyz
    q: vec4<f32>,           // triStart, triCount (polygon spawn area; triCount 0 = plain box)
};
struct Emitters {
    head: vec4<f32>,        // emitterCount, rate, seed, spread
    head2: vec4<f32>,       // particleCount, dt, fixedStreamCount, fixedStreamDrainY
    intakeMin: vec4<f32>,
    intakeMax: vec4<f32>,
    list: array<Emitter, ${MAX_EMITTERS}>,
    // Polygon triangles: pairs of vec4 = (ax, az, bx, bz) then (cx, cz, cumAreaFraction, pad).
    tris: array<vec4<f32>, ${MAX_EMITTER_TRIS * 2}>,
};`;

/** Spawn-point sampling shared by the three emit shaders. Requires `em` and `rnd` in scope.
 *  Returns an ABSOLUTE world position: polygon emitters take X/Z from the outline (so `p.xz` is
 *  unused), everything else keeps the historical box jitter about `p.xyz` — and consumes the very
 *  same random stream in that case, so non-polygon emitters are bit-for-bit unchanged. */
export const EMITTER_SPAWN_WGSL = /* wgsl */ `
fn spawnPoint(e: Emitter, seed: u32) -> vec3<f32> {
    let tc = u32(e.q.y);
    if (tc > 0u) {
        // Area-weighted triangle pick: cumulative fractions ascend to exactly 1 on the last one.
        let t0 = u32(e.q.x);
        let r = rnd(seed * 23u);
        var pick = tc - 1u;
        for (var k = 0u; k < tc; k = k + 1u) {
            if (r <= em.tris[(t0 + k) * 2u + 1u].z) { pick = k; break; }
        }
        let ab = em.tris[(t0 + pick) * 2u];
        let cc = em.tris[(t0 + pick) * 2u + 1u];
        // Uniform barycentric sample; folding u+v>1 back mirrors the far half of the
        // parallelogram into the triangle, which keeps the distribution even.
        var u = rnd(seed * 3u);
        var v = rnd(seed * 7u);
        if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
        let x = ab.x + u * (ab.z - ab.x) + v * (cc.x - ab.x);
        let z = ab.y + u * (ab.w - ab.y) + v * (cc.y - ab.y);
        return vec3<f32>(x, e.p.y + (rnd(seed * 5u) - 0.5) * (2.0 * e.e.y), z);
    }
    let jit = (vec3<f32>(rnd(seed * 3u), rnd(seed * 5u), rnd(seed * 7u)) - 0.5) * (2.0 * e.e.xyz);
    return e.p.xyz + jit;
}`;
