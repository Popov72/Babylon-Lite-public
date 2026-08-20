// Shared contract + helpers for the demo-local GPU fluid backends.
//
// PBF, MLS-MPM and PB-MPM implement the same `FluidSim` interface so renderers
// can swap backends. This module owns their common contracts, generic fluid-flow
// sampling/routing/packing, shared WGSL, scene-SDF helpers and foam state.
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
    /** Number of particles currently participating in simulation and rendering. */
    readonly activeCount?: number;
    /** Exact reset-time marker allocation accepted for each Initial emitter. */
    readonly initialEmitterParticleCounts?: ReadonlyMap<string, number>;
    /** Contiguous active prefix safe for direct instanced rendering. Omitted when active
     *  slots may contain holes, in which case renderers must draw the full capacity. */
    readonly renderCount?: number;
    readonly particleRadius: number;
    /** Optional multiplier on the screen-space surface impostor size (and the
     *  bilateral-blur kernel derived from it). Backends whose particles settle at
     *  wider spacing (e.g. MLS-MPM) need bigger, more-overlapping impostors to
     *  render a smooth surface instead of visible individual spheres. Default 1. */
    readonly surfaceSizeScale?: number;
    /** Optional multiplier for one marker's thickness contribution. Default 1. */
    readonly surfaceThicknessScale?: number;
    /** Reject sparse front markers before screen-space surface reconstruction. */
    readonly surfaceRejectSparseMarkers?: boolean;
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
    /** Re-seed enabled initial volumes, reserve dormant capacity for inflows, or use the
     *  legacy spawn box when no enabled emitter exists. */
    reset(): void;
    /** Live-update a named simulation parameter (for the demo's tuning UI). */
    setParam(key: string, value: number): void;
    /** Optional PB-MPM material selector: 0 liquid, 1 elastic, 2 sand, 3 viscoelastic. */
    setMaterial?(material: number): void;
    /** Inject the per-demo scene SDF used for collision (or null to disable it).
     *  Rebuilds the confinement pipeline; compiled variants are cached by source. */
    setSceneSdf(spec: SceneSdfSpec | null): void;
    /** Configure the legacy recirculating-emitter model. */
    setEmitters(config: EmitterConfig | null): void;
    /** Configure solver-independent initial volumes, inflows and recycling sinks. */
    setFlow(config: FluidFlowConfig | null): void;
    /** Update one installed emitter's transform and initial velocity without resetting flow budgets. */
    updateFlowEmitter(emitter: FluidEmitter): void;
    /** Set the spawn box used by `reset()` to re-seed particles. `accept`, when
     *  provided, restricts seeding to positions where it returns true (CPU
     *  reject-sampling), so particles fit a non-box container shape. */
    setSpawn(min: [number, number, number], max: [number, number, number], accept?: ((x: number, y: number, z: number) => boolean) | null): void;
    /** Set the start-of-sim warm-up length used by `reset()`/`seed()` to gradually
     *  release particles instead of all at once. */
    setWarmup?(frames: number): void;
    /** Inject a generic external force field (or null to disable it). Rebuilds the
     *  integration pass; compiled variants are cached by source. When null the force
     *  path costs nothing: no force buffer is bound and the injected default force is
     *  a no-op the compiler folds away. Mirrors `setSceneSdf`. */
    setForceField(spec: ForceFieldSpec | null): void;
    /** Enable/disable the diffuse-particle (spray/foam/bubbles) system. Optional: supported
     *  fluid backends implement it. Pass a config to enable (the pool + the compute
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

// Shared foam (diffuse-particle) WGSL, injected by the supporting backends. Defines the per-frame
// `Foam` UBO (tuning knobs), the 32-byte `Diffuse` slot struct the foam renderer reads,
// the radial hat kernel W (used by the PBF neighbour gathers), the [0,1] clamp map Φ, and
// a hash PRNG. The `Diffuse` slot layout is FIXED — the foam renderer reads the pool, so
// it must not be reordered. Both backends pack the `Foam` UBO identically (indices below),
// so its layout is shared too; extend only at the END.
//   FoamParams UBO layout (16 floats / 64 bytes):
//     [0..3]  tauTaMin, tauTaMax, tauWcMin, tauWcMax
//     [4..7]  tauKMin, tauKMax, kTa, kWc
//     [8..11] kb, kd, rv, frameDt
//     [12..15] tMin, tMax, frameSeed(u32), _pad1
export const FOAM_BYTES = 64;

export const FOAM_COMMON_WGSL = /* wgsl */ `
struct Foam {
    tauTaMin: f32, tauTaMax: f32, tauWcMin: f32, tauWcMax: f32,
    tauKMin: f32, tauKMax: f32, kTa: f32, kWc: f32,
    kb: f32, kd: f32, rv: f32, frameDt: f32,
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

/** Fields common to all fluid backends. Method-specific tuning lives in the per-solver
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

// ── Generic particle flow ────────────────────────────────────────────
export type FluidVec3 = [number, number, number];

export interface FluidTransform {
    position: FluidVec3;
    rotation: [number, number, number, number];
    scale: FluidVec3;
}

export type FluidShape =
    | { type: "box"; size: FluidVec3 }
    | { type: "sphere"; radius: number }
    | { type: "cylinder"; radius: number; height: number; innerRadius?: number }
    | { type: "cone"; bottomRadius: number; topRadius: number; height: number }
    | { type: "capsule"; radius: number; height: number }
    | { type: "polygonPrism"; points: [number, number][]; thickness: number };

export interface FluidEmitter {
    id: string;
    name: string;
    enabled: boolean;
    behavior: "initial" | "inflow";
    transform: FluidTransform;
    shape: FluidShape;
    sampling: "volume" | "surface";
    velocity: FluidVec3;
    velocitySpace: "local" | "world";
    /** Optional imported scene-node name whose animated world transform drives this analytical emitter. */
    sourceNode?: string;
    /** Optional world-space source-object velocity. Combined with velocity before GPU upload. */
    sourceVelocity?: FluidVec3;
    /** Multiplier applied only to sourceVelocity. Defaults to 1. */
    sourceVelocityFactor?: number;
    /** Optional speed along the analytical shape's outward normal. */
    normalVelocity?: number;
    spread: number;
    /** Optional maximum inflow replenishment in world-volume/second. Omitted means uncapped. Ignored by initial emitters. */
    volumeRate?: number;
    /** Simulation-time seconds before an Inflow starts. Ignored by initial emitters. Defaults to 0. */
    delayBeforeStart?: number;
}

export interface FluidSink {
    id: string;
    name: string;
    enabled: boolean;
    transform: FluidTransform;
    shape: FluidShape;
    /** Delete captured particles by default. Explicit recycle mode preserves closed-loop pump behavior. */
    mode?: "delete" | "recycle";
    targets: string[];
    /** Omitted means every captured particle is handled. Otherwise world-volume/second. */
    volumeRate?: number;
    /** Legacy field name: per-particle capture attempts/second. Mutually exclusive with volumeRate. */
    perParticleRecycleRate?: number;
}

export interface FluidFlowConfig {
    emitters: FluidEmitter[];
    sinks: FluidSink[];
    /** Fill the complete particle pool from initial emitters even when enabled inflows exist. */
    initialEmittersFillCapacity?: boolean;
    /** @internal Exact compatibility metadata produced by legacyEmitterConfigToFluidFlow(). */
    _legacyEmitter?: LegacyEmitterFlowCompatibility;
}

export interface LegacyEmitterFlowCompatibility {
    readonly rate: number;
    readonly fixedStreamCount: number;
    readonly fixedStreamDrainY: number;
    readonly emitterSpeeds: readonly number[];
}

export const MAX_FLUID_EMITTERS = 16;
export const MAX_FLUID_SINKS = 16;
export const MAX_FLUID_POLYGON_TRIANGLES = 128;
export const MAX_FLUID_POLYGON_POINTS = 256;

/** Legacy fixed-pool recirculating-emitter configuration retained for existing fluid demos. */
export interface EmitterConfig {
    emitters: {
        pos: FluidVec3;
        dir: FluidVec3;
        speed: number;
        radius: number;
        halfExtents?: FluidVec3;
        polygon?: [number, number][];
    }[];
    intakeMin: FluidVec3;
    intakeMax: FluidVec3;
    rate: number;
    spread?: number;
    fixedStreamCount?: number;
    fixedStreamDrainY?: number;
}

/** Legacy per-particle relaunch probability for one frame. */
export function legacyEmitterRelaunchProbability(rate: number, dt: number): number {
    return fluidPerParticleRecycleProbability(rate, dt);
}

/** Per-particle recycle probability for one frame. */
export function fluidPerParticleRecycleProbability(rate: number, dt: number): number {
    return Math.min(1, Math.max(0, rate * dt));
}

/** Adapts the legacy fixed-pool emitter model to a flow graph with exact compatibility metadata. */
export function legacyEmitterConfigToFluidFlow(config: EmitterConfig | null, particleCount: number): FluidFlowConfig | null {
    if (!config || config.emitters.length === 0) {
        return null;
    }
    const sourceEmitters = config.emitters.slice(0, MAX_FLUID_EMITTERS);
    const emitters: FluidEmitter[] = sourceEmitters.map((emitter, index) => {
        const halfExtents = emitter.halfExtents ?? [emitter.radius, emitter.radius, emitter.radius];
        const polygon = emitter.polygon;
        const shape: FluidShape = polygon
            ? { type: "polygonPrism", points: polygon.map(([x, z]): [number, number] => [x, z]), thickness: halfExtents[1] * 2 }
            : { type: "box", size: [halfExtents[0] * 2, halfExtents[1] * 2, halfExtents[2] * 2] };
        return {
            id: `legacy-emitter-${index}`,
            name: `Legacy emitter ${index + 1}`,
            enabled: true,
            behavior: "inflow",
            transform: {
                position: polygon ? [0, emitter.pos[1], 0] : [...emitter.pos],
                rotation: [0, 0, 0, 1],
                scale: [1, 1, 1],
            },
            shape,
            sampling: "volume",
            velocity: [emitter.dir[0] * emitter.speed, emitter.dir[1] * emitter.speed, emitter.dir[2] * emitter.speed],
            velocitySpace: "world",
            spread: config.spread ?? 0,
        };
    });
    const intakeSize: FluidVec3 = [
        Math.max(0, config.intakeMax[0] - config.intakeMin[0]),
        Math.max(0, config.intakeMax[1] - config.intakeMin[1]),
        Math.max(0, config.intakeMax[2] - config.intakeMin[2]),
    ];
    return {
        emitters,
        sinks: [
            {
                id: "legacy-intake",
                name: "Legacy intake",
                enabled: true,
                transform: {
                    position: [
                        (config.intakeMin[0] + config.intakeMax[0]) * 0.5,
                        (config.intakeMin[1] + config.intakeMax[1]) * 0.5,
                        (config.intakeMin[2] + config.intakeMax[2]) * 0.5,
                    ],
                    rotation: [0, 0, 0, 1],
                    scale: [1, 1, 1],
                },
                shape: { type: "box", size: intakeSize },
                mode: "recycle",
                targets: emitters.map((emitter) => emitter.id),
            },
        ],
        _legacyEmitter: {
            rate: config.rate,
            fixedStreamCount: Math.min(Math.max(0, Math.trunc(config.fixedStreamCount ?? 0)), Math.max(0, Math.trunc(particleCount))),
            fixedStreamDrainY: config.fixedStreamDrainY ?? 0,
            emitterSpeeds: sourceEmitters.map((emitter) => emitter.speed),
        },
    };
}

/** Rejection attempts for the legacy spawn-box acceptance fallback. */
export const SPAWN_ACCEPT_TRIES = 64;

function polyArea2(poly: readonly [number, number][]): number {
    let area = 0;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        area += poly[j]![0] * poly[i]![1] - poly[i]![0] * poly[j]![1];
    }
    return area;
}

function inTriangle(px: number, py: number, ax: number, ay: number, bx: number, by: number, cx: number, cy: number): boolean {
    const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
    const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
    const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
    return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
}

/** Ear-clips a simple polygon for area-weighted CPU and GPU sampling. */
export function triangulateFluidPolygon(poly: readonly [number, number][]): number[] {
    if (poly.length < 3) {
        return [];
    }
    const indices = Array.from({ length: poly.length }, (_, i) => i);
    if (polyArea2(poly) < 0) {
        indices.reverse();
    }
    const result: number[] = [];
    let guard = poly.length * poly.length + 8;
    while (indices.length > 3 && guard-- > 0) {
        let clipped = false;
        for (let k = 0; k < indices.length; k++) {
            const i0 = indices[(k + indices.length - 1) % indices.length]!;
            const i1 = indices[k]!;
            const i2 = indices[(k + 1) % indices.length]!;
            const [ax, ay] = poly[i0]!;
            const [bx, by] = poly[i1]!;
            const [cx, cy] = poly[i2]!;
            if ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay) <= 0) {
                continue;
            }
            let contains = false;
            for (const m of indices) {
                if (m !== i0 && m !== i1 && m !== i2 && inTriangle(poly[m]![0], poly[m]![1], ax, ay, bx, by, cx, cy)) {
                    contains = true;
                    break;
                }
            }
            if (!contains) {
                result.push(i0, i1, i2);
                indices.splice(k, 1);
                clipped = true;
                break;
            }
        }
        if (!clipped) {
            break;
        }
    }
    if (indices.length === 3) {
        result.push(indices[0]!, indices[1]!, indices[2]!);
    }
    return result;
}

interface PreparedPolygon {
    area: number;
    perimeter: number;
    triangles: { a: [number, number]; b: [number, number]; c: [number, number]; cumulative: number }[];
    edges: { a: [number, number]; b: [number, number]; cumulative: number }[];
}

function preparePolygon(points: readonly [number, number][]): PreparedPolygon {
    const indices = triangulateFluidPolygon(points);
    const triangles: PreparedPolygon["triangles"] = [];
    let area = 0;
    for (let i = 0; i + 2 < indices.length; i += 3) {
        const a = points[indices[i]!]!;
        const b = points[indices[i + 1]!]!;
        const c = points[indices[i + 2]!]!;
        area += Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) * 0.5;
        triangles.push({ a, b, c, cumulative: area });
    }
    const edges: PreparedPolygon["edges"] = [];
    let perimeter = 0;
    for (let i = 0; i < points.length; i++) {
        const a = points[i]!;
        const b = points[(i + 1) % points.length]!;
        perimeter += Math.hypot(b[0] - a[0], b[1] - a[1]);
        edges.push({ a, b, cumulative: perimeter });
    }
    return { area, perimeter, triangles, edges };
}

function absFinite(value: number): number {
    return Number.isFinite(value) ? Math.abs(value) : 0;
}

function localShapeVolume(shape: FluidShape): number {
    switch (shape.type) {
        case "box":
            return absFinite(shape.size[0] * shape.size[1] * shape.size[2]);
        case "sphere": {
            const r = absFinite(shape.radius);
            return (4 / 3) * Math.PI * r ** 3;
        }
        case "cylinder": {
            const r = absFinite(shape.radius);
            const inner = Math.min(r, absFinite(shape.innerRadius ?? 0));
            return Math.PI * (r * r - inner * inner) * absFinite(shape.height);
        }
        case "cone": {
            const r0 = absFinite(shape.bottomRadius);
            const r1 = absFinite(shape.topRadius);
            return (Math.PI * absFinite(shape.height) * (r0 * r0 + r0 * r1 + r1 * r1)) / 3;
        }
        case "capsule": {
            const r = absFinite(shape.radius);
            const segment = Math.max(0, absFinite(shape.height) - 2 * r);
            return Math.PI * r * r * segment + (4 / 3) * Math.PI * r ** 3;
        }
        case "polygonPrism":
            return Math.abs(polyArea2(shape.points)) * 0.5 * absFinite(shape.thickness);
    }
}

export function fluidShapeVolume(shape: FluidShape, transform: FluidTransform): number {
    return localShapeVolume(shape) * absFinite(transform.scale[0] * transform.scale[1] * transform.scale[2]);
}

export function fluidParticleVolume(particleRadius: number): number {
    const r = absFinite(particleRadius);
    return (4 / 3) * Math.PI * r ** 3;
}

function randomUnitVector(): FluidVec3 {
    const y = Math.random() * 2 - 1;
    const angle = Math.random() * Math.PI * 2;
    const horizontal = Math.sqrt(Math.max(0, 1 - y * y));
    return [Math.cos(angle) * horizontal, y, Math.sin(angle) * horizontal];
}

function sampleDisk(radius: number): [number, number] {
    const r = radius * Math.sqrt(Math.random());
    const angle = Math.random() * Math.PI * 2;
    return [Math.cos(angle) * r, Math.sin(angle) * r];
}

function sampleAnnulus(inner: number, outer: number): [number, number] {
    const r = Math.sqrt(inner * inner + Math.random() * (outer * outer - inner * inner));
    const angle = Math.random() * Math.PI * 2;
    return [Math.cos(angle) * r, Math.sin(angle) * r];
}

function sampleTriangle(poly: PreparedPolygon): [number, number] {
    if (!(poly.area > 0) || poly.triangles.length === 0) {
        return [0, 0];
    }
    const pick = Math.random() * poly.area;
    const tri = poly.triangles.find((value) => pick <= value.cumulative) ?? poly.triangles[poly.triangles.length - 1]!;
    let u = Math.random();
    let v = Math.random();
    if (u + v > 1) {
        u = 1 - u;
        v = 1 - v;
    }
    return [tri.a[0] + u * (tri.b[0] - tri.a[0]) + v * (tri.c[0] - tri.a[0]), tri.a[1] + u * (tri.b[1] - tri.a[1]) + v * (tri.c[1] - tri.a[1])];
}

function sampleEdge(poly: PreparedPolygon): [number, number] {
    if (!(poly.perimeter > 0) || poly.edges.length === 0) {
        return [0, 0];
    }
    const pick = Math.random() * poly.perimeter;
    const edge = poly.edges.find((value) => pick <= value.cumulative) ?? poly.edges[poly.edges.length - 1]!;
    const t = Math.random();
    return [edge.a[0] + (edge.b[0] - edge.a[0]) * t, edge.a[1] + (edge.b[1] - edge.a[1]) * t];
}

function sampleLocalShape(shape: FluidShape, sampling: "volume" | "surface"): FluidVec3 {
    switch (shape.type) {
        case "box": {
            const size = shape.size.map(absFinite) as FluidVec3;
            const p: FluidVec3 = [(Math.random() - 0.5) * size[0], (Math.random() - 0.5) * size[1], (Math.random() - 0.5) * size[2]];
            if (sampling === "volume") {
                return p;
            }
            const areas = [size[1] * size[2], size[0] * size[2], size[0] * size[1]];
            const pick = Math.random() * 2 * (areas[0]! + areas[1]! + areas[2]!);
            if (pick < 2 * areas[0]!) {
                p[0] = pick < areas[0]! ? -size[0] * 0.5 : size[0] * 0.5;
            } else if (pick < 2 * areas[0]! + 2 * areas[1]!) {
                p[1] = pick < 2 * areas[0]! + areas[1]! ? -size[1] * 0.5 : size[1] * 0.5;
            } else {
                p[2] = pick < 2 * areas[0]! + 2 * areas[1]! + areas[2]! ? -size[2] * 0.5 : size[2] * 0.5;
            }
            return p;
        }
        case "sphere": {
            const direction = randomUnitVector();
            const radius = absFinite(shape.radius) * (sampling === "surface" ? 1 : Math.cbrt(Math.random()));
            return [direction[0] * radius, direction[1] * radius, direction[2] * radius];
        }
        case "cylinder": {
            const outer = absFinite(shape.radius);
            const inner = Math.min(outer, absFinite(shape.innerRadius ?? 0));
            const height = absFinite(shape.height);
            if (sampling === "volume") {
                const [x, z] = sampleAnnulus(inner, outer);
                return [x, (Math.random() - 0.5) * height, z];
            }
            const outerArea = 2 * Math.PI * outer * height;
            const innerArea = 2 * Math.PI * inner * height;
            const capArea = Math.PI * (outer * outer - inner * inner);
            const pick = Math.random() * (outerArea + innerArea + 2 * capArea);
            if (pick < outerArea) {
                const angle = Math.random() * Math.PI * 2;
                return [Math.cos(angle) * outer, (Math.random() - 0.5) * height, Math.sin(angle) * outer];
            }
            if (pick < outerArea + innerArea) {
                const angle = Math.random() * Math.PI * 2;
                return [Math.cos(angle) * inner, (Math.random() - 0.5) * height, Math.sin(angle) * inner];
            }
            const [x, z] = sampleAnnulus(inner, outer);
            return [x, pick < outerArea + innerArea + capArea ? -height * 0.5 : height * 0.5, z];
        }
        case "cone": {
            const r0 = absFinite(shape.bottomRadius);
            const r1 = absFinite(shape.topRadius);
            const height = absFinite(shape.height);
            const delta = r1 - r0;
            if (sampling === "volume") {
                const radius = Math.abs(delta) > 1e-8 ? Math.cbrt(r0 ** 3 + Math.random() * (r1 ** 3 - r0 ** 3)) : r0;
                const t = Math.abs(delta) > 1e-8 ? (radius - r0) / delta : Math.random();
                const [x, z] = sampleDisk(radius);
                return [x, (t - 0.5) * height, z];
            }
            const lateral = Math.PI * (r0 + r1) * Math.hypot(delta, height);
            const bottom = Math.PI * r0 * r0;
            const top = Math.PI * r1 * r1;
            const pick = Math.random() * (lateral + bottom + top);
            if (pick < lateral) {
                const target = Math.random() * (r0 + delta * 0.5);
                const t = Math.abs(delta) > 1e-8 ? (-r0 + Math.sqrt(Math.max(0, r0 * r0 + 2 * delta * target))) / delta : Math.random();
                const radius = r0 + delta * t;
                const angle = Math.random() * Math.PI * 2;
                return [Math.cos(angle) * radius, (t - 0.5) * height, Math.sin(angle) * radius];
            }
            const isTop = pick >= lateral + bottom;
            const [x, z] = sampleDisk(isTop ? r1 : r0);
            return [x, isTop ? height * 0.5 : -height * 0.5, z];
        }
        case "capsule": {
            const radius = absFinite(shape.radius);
            const segment = Math.max(0, absFinite(shape.height) - 2 * radius);
            const cylinderMeasure = sampling === "volume" ? Math.PI * radius * radius * segment : 2 * Math.PI * radius * segment;
            const capMeasure = sampling === "volume" ? (4 / 3) * Math.PI * radius ** 3 : 4 * Math.PI * radius * radius;
            if (Math.random() * (cylinderMeasure + capMeasure) < cylinderMeasure) {
                const angle = Math.random() * Math.PI * 2;
                const r = sampling === "volume" ? radius * Math.sqrt(Math.random()) : radius;
                return [Math.cos(angle) * r, (Math.random() - 0.5) * segment, Math.sin(angle) * r];
            }
            const sign = Math.random() < 0.5 ? -1 : 1;
            const y = Math.random();
            const angle = Math.random() * Math.PI * 2;
            const h = Math.sqrt(Math.max(0, 1 - y * y));
            const r = sampling === "volume" ? radius * Math.cbrt(Math.random()) : radius;
            return [Math.cos(angle) * h * r, sign * (segment * 0.5 + y * r), Math.sin(angle) * h * r];
        }
        case "polygonPrism": {
            const polygon = preparePolygon(shape.points);
            const thickness = absFinite(shape.thickness);
            if (sampling === "volume") {
                const [x, z] = sampleTriangle(polygon);
                return [x, (Math.random() - 0.5) * thickness, z];
            }
            if (Math.random() * (2 * polygon.area + polygon.perimeter * thickness) < 2 * polygon.area) {
                const [x, z] = sampleTriangle(polygon);
                return [x, Math.random() < 0.5 ? -thickness * 0.5 : thickness * 0.5, z];
            }
            const [x, z] = sampleEdge(polygon);
            return [x, (Math.random() - 0.5) * thickness, z];
        }
    }
}

function normaliseVector(value: FluidVec3, fallback: FluidVec3 = [0, 1, 0]): FluidVec3 {
    const length = Math.hypot(value[0], value[1], value[2]);
    return length > 1e-8 ? [value[0] / length, value[1] / length, value[2] / length] : fallback;
}

function localPolygonPrismNormal(shape: Extract<FluidShape, { type: "polygonPrism" }>, point: FluidVec3): FluidVec3 {
    const halfThickness = absFinite(shape.thickness) * 0.5;
    const capDistance = halfThickness - Math.abs(point[1]);
    let nearestDistance2 = Number.POSITIVE_INFINITY;
    let nearest: [number, number] = [0, 1];
    const winding = polyArea2(shape.points) < 0 ? -1 : 1;
    for (let index = 0; index < shape.points.length; index++) {
        const a = shape.points[index]!;
        const b = shape.points[(index + 1) % shape.points.length]!;
        const dx = b[0] - a[0];
        const dz = b[1] - a[1];
        const length2 = dx * dx + dz * dz;
        const t = length2 > 1e-12 ? Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[2] - a[1]) * dz) / length2)) : 0;
        const offsetX = point[0] - (a[0] + dx * t);
        const offsetZ = point[2] - (a[1] + dz * t);
        const distance2 = offsetX * offsetX + offsetZ * offsetZ;
        if (distance2 < nearestDistance2) {
            nearestDistance2 = distance2;
            const inverseLength = length2 > 1e-12 ? 1 / Math.sqrt(length2) : 0;
            nearest = [winding * dz * inverseLength, -winding * dx * inverseLength];
        }
    }
    return capDistance <= Math.sqrt(nearestDistance2) ? [0, point[1] < 0 ? -1 : 1, 0] : [nearest[0], 0, nearest[1]];
}

function localShapeNormal(shape: FluidShape, point: FluidVec3): FluidVec3 {
    switch (shape.type) {
        case "box": {
            const half: FluidVec3 = [absFinite(shape.size[0]) * 0.5, absFinite(shape.size[1]) * 0.5, absFinite(shape.size[2]) * 0.5];
            const distances: FluidVec3 = [half[0] - Math.abs(point[0]), half[1] - Math.abs(point[1]), half[2] - Math.abs(point[2])];
            const axis = distances[0] <= distances[1] && distances[0] <= distances[2] ? 0 : distances[1] <= distances[2] ? 1 : 2;
            const result: FluidVec3 = [0, 0, 0];
            result[axis] = point[axis] < 0 ? -1 : 1;
            return result;
        }
        case "sphere":
            return normaliseVector(point);
        case "cylinder": {
            const radius = absFinite(shape.radius);
            const innerRadius = Math.min(radius, absFinite(shape.innerRadius ?? 0));
            const radial = Math.hypot(point[0], point[2]);
            const outerDistance = radius - radial;
            const innerDistance = innerRadius > 0 ? radial - innerRadius : Number.POSITIVE_INFINITY;
            const capDistance = absFinite(shape.height) * 0.5 - Math.abs(point[1]);
            if (capDistance <= outerDistance && capDistance <= innerDistance) {
                return [0, point[1] < 0 ? -1 : 1, 0];
            }
            const direction: FluidVec3 = radial > 1e-8 ? [point[0] / radial, 0, point[2] / radial] : [1, 0, 0];
            return innerDistance < outerDistance ? [-direction[0], 0, -direction[2]] : direction;
        }
        case "cone": {
            const height = absFinite(shape.height);
            const bottomRadius = absFinite(shape.bottomRadius);
            const topRadius = absFinite(shape.topRadius);
            const delta = topRadius - bottomRadius;
            const t = height > 1e-8 ? Math.max(0, Math.min(1, point[1] / height + 0.5)) : 0.5;
            const radius = bottomRadius + delta * t;
            const radial = Math.hypot(point[0], point[2]);
            const slope = height > 1e-8 ? delta / height : 0;
            const lateralDistance = Math.abs(radius - radial) / Math.hypot(1, slope);
            const bottomDistance = point[1] + height * 0.5;
            const topDistance = height * 0.5 - point[1];
            if (bottomDistance <= lateralDistance && bottomDistance <= topDistance) {
                return [0, -1, 0];
            }
            if (topDistance <= lateralDistance) {
                return [0, 1, 0];
            }
            return normaliseVector([radial > 1e-8 ? point[0] / radial : 1, -slope, radial > 1e-8 ? point[2] / radial : 0]);
        }
        case "capsule": {
            const radius = absFinite(shape.radius);
            const segmentHalf = Math.max(0, absFinite(shape.height) - 2 * radius) * 0.5;
            const closestY = Math.max(-segmentHalf, Math.min(segmentHalf, point[1]));
            return normaliseVector([point[0], point[1] - closestY, point[2]], [1, 0, 0]);
        }
        case "polygonPrism":
            return localPolygonPrismNormal(shape, point);
    }
}

function normalisedRotation(rotation: FluidTransform["rotation"]): FluidTransform["rotation"] {
    const length = Math.hypot(rotation[0], rotation[1], rotation[2], rotation[3]);
    return length > 1e-8 ? [rotation[0] / length, rotation[1] / length, rotation[2] / length, rotation[3] / length] : [0, 0, 0, 1];
}

function rotateVector(rotation: FluidTransform["rotation"], value: FluidVec3): FluidVec3 {
    const [qx, qy, qz, qw] = normalisedRotation(rotation);
    const tx = 2 * (qy * value[2] - qz * value[1]);
    const ty = 2 * (qz * value[0] - qx * value[2]);
    const tz = 2 * (qx * value[1] - qy * value[0]);
    return [value[0] + qw * tx + qy * tz - qz * ty, value[1] + qw * ty + qz * tx - qx * tz, value[2] + qw * tz + qx * ty - qy * tx];
}

function transformFluidPoint(transform: FluidTransform, local: FluidVec3): FluidVec3 {
    const scaled: FluidVec3 = [local[0] * transform.scale[0], local[1] * transform.scale[1], local[2] * transform.scale[2]];
    const rotated = rotateVector(transform.rotation, scaled);
    return [rotated[0] + transform.position[0], rotated[1] + transform.position[1], rotated[2] + transform.position[2]];
}

export function sampleFluidEmitterPosition(emitter: FluidEmitter): FluidVec3 {
    return transformFluidPoint(emitter.transform, sampleLocalShape(emitter.shape, emitter.sampling));
}

function emitterBaseVelocity(emitter: FluidEmitter): FluidVec3 {
    const authored = emitter.velocitySpace === "world" ? emitter.velocity : rotateVector(emitter.transform.rotation, emitter.velocity);
    const source = emitter.sourceVelocity ?? [0, 0, 0];
    const factor = emitter.sourceVelocityFactor ?? 1;
    return [authored[0] + source[0] * factor, authored[1] + source[1] * factor, authored[2] + source[2] * factor];
}

function worldShapeNormal(emitter: FluidEmitter, localPoint: FluidVec3): FluidVec3 {
    const local = localShapeNormal(emitter.shape, localPoint);
    const safeScale = emitter.transform.scale.map((value) => (Math.abs(value) > 1e-8 ? value : 1e-8)) as FluidVec3;
    const inverseScaled: FluidVec3 = [local[0] / safeScale[0], local[1] / safeScale[1], local[2] / safeScale[2]];
    return normaliseVector(rotateVector(emitter.transform.rotation, inverseScaled));
}

function fluidEmitterLaunchAtLocal(emitter: FluidEmitter, local: FluidVec3): { position: FluidVec3; velocity: FluidVec3 } {
    const base = emitterBaseVelocity(emitter);
    const normalSpeed = emitter.normalVelocity ?? 0;
    if (normalSpeed !== 0) {
        const normal = worldShapeNormal(emitter, local);
        base[0] += normal[0] * normalSpeed;
        base[1] += normal[1] * normalSpeed;
        base[2] += normal[2] * normalSpeed;
    }
    const spread = Math.hypot(base[0], base[1], base[2]) * emitter.spread;
    return {
        position: transformFluidPoint(emitter.transform, local),
        velocity: [base[0] + (Math.random() - 0.5) * spread, base[1] + (Math.random() - 0.5) * spread, base[2] + (Math.random() - 0.5) * spread],
    };
}

function sampleFluidEmitterLaunch(emitter: FluidEmitter): { position: FluidVec3; velocity: FluidVec3 } {
    return fluidEmitterLaunchAtLocal(emitter, sampleLocalShape(emitter.shape, emitter.sampling));
}

function localShapeBounds(shape: FluidShape): { min: FluidVec3; max: FluidVec3 } {
    switch (shape.type) {
        case "box": {
            const half = shape.size.map((value) => absFinite(value) * 0.5) as FluidVec3;
            return { min: [-half[0], -half[1], -half[2]], max: half };
        }
        case "sphere": {
            const radius = absFinite(shape.radius);
            return { min: [-radius, -radius, -radius], max: [radius, radius, radius] };
        }
        case "cylinder": {
            const radius = absFinite(shape.radius);
            const halfHeight = absFinite(shape.height) * 0.5;
            return { min: [-radius, -halfHeight, -radius], max: [radius, halfHeight, radius] };
        }
        case "cone": {
            const radius = Math.max(absFinite(shape.bottomRadius), absFinite(shape.topRadius));
            const halfHeight = absFinite(shape.height) * 0.5;
            return { min: [-radius, -halfHeight, -radius], max: [radius, halfHeight, radius] };
        }
        case "capsule": {
            const radius = absFinite(shape.radius);
            const halfHeight = absFinite(shape.height) * 0.5;
            return { min: [-radius, -halfHeight, -radius], max: [radius, halfHeight, radius] };
        }
        case "polygonPrism": {
            const halfThickness = absFinite(shape.thickness) * 0.5;
            if (shape.points.length === 0) {
                return { min: [0, -halfThickness, 0], max: [0, halfThickness, 0] };
            }
            let minX = Number.POSITIVE_INFINITY;
            let maxX = Number.NEGATIVE_INFINITY;
            let minZ = Number.POSITIVE_INFINITY;
            let maxZ = Number.NEGATIVE_INFINITY;
            for (const [x, z] of shape.points) {
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minZ = Math.min(minZ, z);
                maxZ = Math.max(maxZ, z);
            }
            return { min: [minX, -halfThickness, minZ], max: [maxX, halfThickness, maxZ] };
        }
    }
}

function localShapeContains(shape: FluidShape, point: FluidVec3, polygon?: PreparedPolygon): boolean {
    const epsilon = 1e-7;
    switch (shape.type) {
        case "box":
            return (
                Math.abs(point[0]) <= absFinite(shape.size[0]) * 0.5 + epsilon &&
                Math.abs(point[1]) <= absFinite(shape.size[1]) * 0.5 + epsilon &&
                Math.abs(point[2]) <= absFinite(shape.size[2]) * 0.5 + epsilon
            );
        case "sphere":
            return point[0] * point[0] + point[1] * point[1] + point[2] * point[2] <= absFinite(shape.radius) ** 2 + epsilon;
        case "cylinder": {
            const radius = absFinite(shape.radius);
            const radial2 = point[0] * point[0] + point[2] * point[2];
            const innerRadius = Math.min(radius, absFinite(shape.innerRadius ?? 0));
            return Math.abs(point[1]) <= absFinite(shape.height) * 0.5 + epsilon && radial2 <= radius * radius + epsilon && radial2 + epsilon >= innerRadius * innerRadius;
        }
        case "cone": {
            const height = absFinite(shape.height);
            if (Math.abs(point[1]) > height * 0.5 + epsilon) {
                return false;
            }
            const t = height > 1e-8 ? Math.max(0, Math.min(1, point[1] / height + 0.5)) : 0.5;
            const radius = absFinite(shape.bottomRadius) + (absFinite(shape.topRadius) - absFinite(shape.bottomRadius)) * t;
            return point[0] * point[0] + point[2] * point[2] <= radius * radius + epsilon;
        }
        case "capsule": {
            const radius = absFinite(shape.radius);
            const segmentHalf = Math.max(0, absFinite(shape.height) - 2 * radius) * 0.5;
            const closestY = Math.max(-segmentHalf, Math.min(segmentHalf, point[1]));
            const dy = point[1] - closestY;
            return point[0] * point[0] + dy * dy + point[2] * point[2] <= radius * radius + epsilon;
        }
        case "polygonPrism":
            return (
                Math.abs(point[1]) <= absFinite(shape.thickness) * 0.5 + epsilon &&
                (polygon ?? preparePolygon(shape.points)).triangles.some((triangle) =>
                    inTriangle(point[0], point[2], triangle.a[0], triangle.a[1], triangle.b[0], triangle.b[1], triangle.c[0], triangle.c[1])
                )
            );
    }
}

export interface FluidInitialBounds {
    min: FluidVec3;
    max: FluidVec3;
}

function fluidPointInsideBounds(point: FluidVec3, bounds: FluidInitialBounds): boolean {
    return (
        point[0] >= bounds.min[0] && point[0] <= bounds.max[0] && point[1] >= bounds.min[1] && point[1] <= bounds.max[1] && point[2] >= bounds.min[2] && point[2] <= bounds.max[2]
    );
}

function createFluidInitialLattice(emitter: FluidEmitter, count: number, worldVolume: number, worldBounds?: FluidInitialBounds): FluidVec3[] {
    if (count <= 0 || !(worldVolume > 0)) {
        return [];
    }
    const scale = emitter.transform.scale.map(absFinite) as FluidVec3;
    if (scale.some((value) => value === 0)) {
        return Array.from({ length: count }, () => [0, 0, 0] as FluidVec3);
    }
    const bounds = localShapeBounds(emitter.shape);
    const extents: FluidVec3 = [bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]];
    const center: FluidVec3 = [(bounds.min[0] + bounds.max[0]) * 0.5, (bounds.min[1] + bounds.max[1]) * 0.5, (bounds.min[2] + bounds.max[2]) * 0.5];
    const polygon = emitter.shape.type === "polygonPrism" ? preparePolygon(emitter.shape.points) : undefined;
    let worldSpacing = Math.cbrt(worldVolume / count);
    let candidates: FluidVec3[] = [];
    for (let attempt = 0; attempt < 20; attempt++) {
        const spacing: FluidVec3 = [worldSpacing / scale[0], worldSpacing / scale[1], worldSpacing / scale[2]];
        const dimensions: FluidVec3 = [
            Math.max(1, Math.ceil(extents[0] / spacing[0] - 1e-8)),
            Math.max(1, Math.ceil(extents[1] / spacing[1] - 1e-8)),
            Math.max(1, Math.ceil(extents[2] / spacing[2] - 1e-8)),
        ];
        const start: FluidVec3 = [
            center[0] - ((dimensions[0] - 1) * spacing[0]) / 2,
            center[1] - ((dimensions[1] - 1) * spacing[1]) / 2,
            center[2] - ((dimensions[2] - 1) * spacing[2]) / 2,
        ];
        candidates = [];
        let unclippedCount = 0;
        for (let y = 0; y < dimensions[1]; y++) {
            for (let z = 0; z < dimensions[2]; z++) {
                for (let x = 0; x < dimensions[0]; x++) {
                    const point: FluidVec3 = [start[0] + x * spacing[0], start[1] + y * spacing[1], start[2] + z * spacing[2]];
                    if (localShapeContains(emitter.shape, point, polygon)) {
                        unclippedCount++;
                        if (!worldBounds || fluidPointInsideBounds(transformFluidPoint(emitter.transform, point), worldBounds)) {
                            candidates.push(point);
                        }
                    }
                }
            }
        }
        if (unclippedCount >= count) {
            break;
        }
        worldSpacing *= 0.96;
    }
    if (worldBounds && candidates.length < count) {
        return candidates;
    }
    if (candidates.length < count) {
        const fallback = candidates.length > 0 ? candidates : [center];
        return Array.from({ length: count }, (_, index) => fallback[index % fallback.length]!);
    }
    if (candidates.length === count) {
        return candidates;
    }
    return Array.from({ length: count }, (_, index) => candidates[Math.floor(((index + 0.5) * candidates.length) / count)]!);
}

function countFluidInitialLattice(emitter: FluidEmitter, count: number, worldVolume: number, worldBounds?: FluidInitialBounds): number {
    if (count <= 0 || !(worldVolume > 0)) {
        return 0;
    }
    const scale = emitter.transform.scale.map(absFinite) as FluidVec3;
    if (scale.some((value) => value === 0)) {
        return count;
    }
    const bounds = localShapeBounds(emitter.shape);
    const extents: FluidVec3 = [bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]];
    const center: FluidVec3 = [(bounds.min[0] + bounds.max[0]) * 0.5, (bounds.min[1] + bounds.max[1]) * 0.5, (bounds.min[2] + bounds.max[2]) * 0.5];
    const polygon = emitter.shape.type === "polygonPrism" ? preparePolygon(emitter.shape.points) : undefined;
    let worldSpacing = Math.cbrt(worldVolume / count);
    let acceptedCount = 0;
    for (let attempt = 0; attempt < 20; attempt++) {
        const spacing: FluidVec3 = [worldSpacing / scale[0], worldSpacing / scale[1], worldSpacing / scale[2]];
        const dimensions: FluidVec3 = [
            Math.max(1, Math.ceil(extents[0] / spacing[0] - 1e-8)),
            Math.max(1, Math.ceil(extents[1] / spacing[1] - 1e-8)),
            Math.max(1, Math.ceil(extents[2] / spacing[2] - 1e-8)),
        ];
        const start: FluidVec3 = [
            center[0] - ((dimensions[0] - 1) * spacing[0]) / 2,
            center[1] - ((dimensions[1] - 1) * spacing[1]) / 2,
            center[2] - ((dimensions[2] - 1) * spacing[2]) / 2,
        ];
        acceptedCount = 0;
        let unclippedCount = 0;
        for (let y = 0; y < dimensions[1]; y++) {
            for (let z = 0; z < dimensions[2]; z++) {
                for (let x = 0; x < dimensions[0]; x++) {
                    const point: FluidVec3 = [start[0] + x * spacing[0], start[1] + y * spacing[1], start[2] + z * spacing[2]];
                    if (!localShapeContains(emitter.shape, point, polygon)) {
                        continue;
                    }
                    unclippedCount++;
                    if (!worldBounds || fluidPointInsideBounds(transformFluidPoint(emitter.transform, point), worldBounds)) {
                        acceptedCount++;
                    }
                }
            }
        }
        if (unclippedCount >= count) {
            break;
        }
        worldSpacing *= 0.96;
    }
    return worldBounds ? Math.min(count, acceptedCount) : count;
}

export interface FluidInitialParticleCounts {
    activeCount: number;
    emitterCounts: ReadonlyMap<string, number>;
}

/** Counts reset-time Initial markers without allocating position or velocity arrays. */
export function countFluidInitialParticles(
    count: number,
    config: FluidFlowConfig | null,
    particleVolume = 1,
    bounds?: FluidInitialBounds,
    deriveInitialCount = false
): FluidInitialParticleCounts | null {
    if (!config) {
        return null;
    }
    const enabled = config.emitters.slice(0, MAX_FLUID_EMITTERS).filter((emitter) => emitter.enabled);
    const initial = enabled.filter((emitter) => emitter.behavior === "initial");
    const emitterCounts = new Map(initial.map((emitter) => [emitter.id, 0]));
    if (initial.length === 0) {
        return { activeCount: 0, emitterCounts };
    }
    const volumes = initial.map((emitter) => fluidShapeVolume(emitter.shape, emitter.transform));
    const total = volumes.reduce((sum, value) => sum + value, 0);
    if (!(total > 0)) {
        return { activeCount: 0, emitterCounts };
    }
    const hasInflow = enabled.some((emitter) => emitter.behavior === "inflow");
    const deriveCount = (hasInflow || deriveInitialCount) && !config.initialEmittersFillCapacity;
    const requestedActiveCount = deriveCount ? Math.min(count, Math.max(0, Math.ceil(total / Math.max(particleVolume, 1e-12)))) : count;
    const allocations = volumes.map((volume, index) => {
        const exact = (requestedActiveCount * volume) / total;
        return { index, count: Math.floor(exact), remainder: exact - Math.floor(exact) };
    });
    const left = requestedActiveCount - allocations.reduce((sum, allocation) => sum + allocation.count, 0);
    const ranked = [...allocations].sort((a, b) => {
        const remainder = b.remainder - a.remainder;
        if (remainder !== 0) {
            return remainder;
        }
        const aid = initial[a.index]!.id;
        const bid = initial[b.index]!.id;
        return aid < bid ? -1 : aid > bid ? 1 : a.index - b.index;
    });
    for (let index = 0; index < left; index++) {
        ranked[index]!.count++;
    }
    let activeCount = 0;
    for (const allocation of allocations) {
        const emitter = initial[allocation.index]!;
        const accepted = emitter.sampling === "volume" ? countFluidInitialLattice(emitter, allocation.count, volumes[allocation.index]!, bounds) : allocation.count;
        emitterCounts.set(emitter.id, accepted);
        activeCount += accepted;
    }
    return { activeCount, emitterCounts };
}

export interface FluidInitialParticles {
    positions: Float32Array;
    velocities: Float32Array;
    activeCount: number;
    emitterCounts: ReadonlyMap<string, number>;
}

/** Returns reset-time initial particles, or null to preserve legacy spawn-box seeding.
 *  By default, initial-only graphs activate the complete selected pool. Graphs with enabled
 *  inflows, and callers that request volume-derived seeding, activate only the initial volumes'
 *  demand and reserve the remaining slots for inflow.
 *  `initialEmittersFillCapacity` explicitly fills the complete pool instead.
 *  An inflow-only graph deliberately returns an empty active prefix so subsequent frames can
 *  activate dormant slots at the authored inflow rate. */
export function createFluidInitialParticles(
    count: number,
    config: FluidFlowConfig | null,
    particleVolume = 1,
    bounds?: FluidInitialBounds,
    deriveInitialCount = false
): FluidInitialParticles | null {
    if (!config) {
        return null;
    }
    const enabled = (config?.emitters.slice(0, MAX_FLUID_EMITTERS) ?? []).filter((emitter) => emitter.enabled);
    const initial = enabled.filter((emitter) => emitter.behavior === "initial");
    const hasInflow = enabled.some((emitter) => emitter.behavior === "inflow");
    const emitterCounts = new Map(initial.map((emitter) => [emitter.id, 0]));
    if (initial.length === 0) {
        return { positions: new Float32Array(0), velocities: new Float32Array(0), activeCount: 0, emitterCounts };
    }
    const volumes = initial.map((emitter) => fluidShapeVolume(emitter.shape, emitter.transform));
    const total = volumes.reduce((sum, value) => sum + value, 0);
    if (!(total > 0)) {
        return { positions: new Float32Array(0), velocities: new Float32Array(0), activeCount: 0, emitterCounts };
    }
    const deriveCount = (hasInflow || deriveInitialCount) && !config?.initialEmittersFillCapacity;
    const activeCount = deriveCount ? Math.min(count, Math.max(0, Math.ceil(total / Math.max(particleVolume, 1e-12)))) : count;
    const allocations = volumes.map((volume, index) => {
        const exact = (activeCount * volume) / total;
        return { index, count: Math.floor(exact), remainder: exact - Math.floor(exact) };
    });
    const left = activeCount - allocations.reduce((sum, allocation) => sum + allocation.count, 0);
    const ranked = [...allocations].sort((a, b) => {
        const remainder = b.remainder - a.remainder;
        if (remainder !== 0) {
            return remainder;
        }
        const aid = initial[a.index]!.id;
        const bid = initial[b.index]!.id;
        return aid < bid ? -1 : aid > bid ? 1 : a.index - b.index;
    });
    for (let i = 0; i < left; i++) {
        ranked[i]!.count++;
    }
    const positions = new Float32Array(activeCount * 3);
    const velocities = new Float32Array(activeCount * 3);
    let cursor = 0;
    for (const allocation of allocations) {
        const emitter = initial[allocation.index]!;
        const emitterStart = cursor;
        const localPoints = emitter.sampling === "volume" ? createFluidInitialLattice(emitter, allocation.count, volumes[allocation.index]!, bounds) : undefined;
        const launchCount = localPoints?.length ?? allocation.count;
        for (let i = 0; i < launchCount; i++) {
            let launch = localPoints ? fluidEmitterLaunchAtLocal(emitter, localPoints[i]!) : sampleFluidEmitterLaunch(emitter);
            if (!localPoints && bounds && !fluidPointInsideBounds(launch.position, bounds)) {
                let accepted = false;
                for (let attempt = 0; attempt < 31; attempt++) {
                    launch = sampleFluidEmitterLaunch(emitter);
                    if (fluidPointInsideBounds(launch.position, bounds)) {
                        accepted = true;
                        break;
                    }
                }
                if (!accepted) {
                    continue;
                }
            }
            const { position: point, velocity } = launch;
            positions[cursor] = point[0];
            velocities[cursor++] = velocity[0];
            positions[cursor] = point[1];
            velocities[cursor++] = velocity[1];
            positions[cursor] = point[2];
            velocities[cursor++] = velocity[2];
        }
        emitterCounts.set(emitter.id, (cursor - emitterStart) / 3);
    }
    return {
        positions: cursor === positions.length ? positions : positions.slice(0, cursor),
        velocities: cursor === velocities.length ? velocities : velocities.slice(0, cursor),
        activeCount: cursor / 3,
        emitterCounts,
    };
}

const FLOW_HEADER_FLOATS = 8;
const FLOW_ENTITY_FLOATS = 32;
const FLOW_EMITTER_BASE = FLOW_HEADER_FLOATS;
const FLOW_SINK_BASE = FLOW_EMITTER_BASE + MAX_FLUID_EMITTERS * FLOW_ENTITY_FLOATS;
const FLOW_TRI_BASE = FLOW_SINK_BASE + MAX_FLUID_SINKS * FLOW_ENTITY_FLOATS;
const FLOW_POINT_BASE = FLOW_TRI_BASE + MAX_FLUID_POLYGON_TRIANGLES * 8;
export const FLUID_FLOW_FLOATS = FLOW_POINT_BASE + MAX_FLUID_POLYGON_POINTS * 4;
export const FLUID_FLOW_BYTES = FLUID_FLOW_FLOATS * 4;
const FLOW_EMITTER_COUNTER_U32 = MAX_FLUID_EMITTERS * 2;
export const FLUID_FLOW_COUNTER_BYTES = (FLOW_EMITTER_COUNTER_U32 + MAX_FLUID_SINKS) * 4;
const UNLIMITED_FLOW_BUDGET = 0xffffffff;
const FLOW_SINK_OPERATION_DELETE = 0;
const FLOW_SINK_OPERATION_RECYCLE = 1;
const FLOW_SINK_OPERATION_LEGACY = 2;
const FLOW_SINK_RATE_VOLUME = 0;
const FLOW_SINK_RATE_PER_PARTICLE = 1;
const FLUID_LIFECYCLE_HEADER_U32 = 4;
const FLUID_PARTICLE_FREE = 0;
const FLUID_PARTICLE_ACTIVE = 1;
const FLUID_PARTICLE_RESERVED = 2;

export interface FluidFlowState {
    readonly device: GPUDevice;
    readonly uniformBuffer: GPUBuffer;
    readonly counterBuffer: GPUBuffer;
    readonly data: ArrayBuffer;
    readonly f32: Float32Array;
    readonly u32: Uint32Array;
    readonly counterData: Uint32Array;
    readonly particleVolume: number;
    readonly capacity: number;
    readonly lifecycleBuffer: GPUBuffer;
    config: FluidFlowConfig | null;
    legacyEmitter: LegacyEmitterFlowCompatibility | null;
    active: boolean;
    activeCount: number;
    frameSeed: number;
    emitterCursor: number;
    emitterIds: string[];
    emitterRates: (number | undefined)[];
    emitterActive: boolean[];
    emitterDelays: number[];
    emitterCarries: Float64Array;
    elapsedSeconds: number;
    sinkIds: string[];
    sinkRates: (number | undefined)[];
    sinkCarries: Float64Array;
    activeCountReadback: FluidActiveCountReadback | null;
}

export interface FluidFlowFrame {
    readonly flowActive: boolean;
    readonly deleteActive: boolean;
    readonly emitActive: boolean;
    /** Number of finite-rate particles requested this frame, capped to capacity. */
    readonly emitCount: number;
    /** True when at least one started inflow has no finite volume-rate limit. */
    readonly emitUnlimited: boolean;
}

interface FluidActiveCountReadback {
    readonly buffers: GPUBuffer[];
    readonly states: Array<"idle" | "copied" | "mapping">;
    readonly generations: number[];
    generation: number;
    error: unknown;
    next: number;
}

export interface FluidVolumeBudget {
    readonly count: number;
    readonly carry: number;
}

/** Converts a world-volume rate to a whole-particle frame budget with fractional carry. */
export function fluidVolumeBudget(volumeRate: number, dt: number, particleVolume: number, carry: number): FluidVolumeBudget {
    const exact = carry + (Math.max(0, Number.isFinite(volumeRate) ? volumeRate : 0) * Math.max(0, dt)) / Math.max(particleVolume, 1e-12);
    const count = Math.min(Math.floor(exact), UNLIMITED_FLOW_BUDGET - 1);
    return { count, carry: exact - Math.floor(exact) };
}

/** Allocates finite inflow budgets first, then divides remaining capacity among unlimited inflows. */
export function allocateFluidInflowCapacity(budgets: ArrayLike<number>, unlimited: ArrayLike<boolean>, inactiveCapacity: number, cursor = 0): Uint32Array {
    const count = Math.max(budgets.length, unlimited.length);
    const allocations = new Uint32Array(count);
    let remaining = Math.max(0, Math.floor(inactiveCapacity));
    for (let offset = 0; offset < count && remaining > 0; offset++) {
        const index = (cursor + offset) % Math.max(1, count);
        if (unlimited[index] || !(budgets[index]! > 0)) {
            continue;
        }
        const allocated = Math.min(remaining, Math.floor(budgets[index]!));
        allocations[index] = allocated;
        remaining -= allocated;
    }
    const unlimitedIndices: number[] = [];
    for (let i = 0; i < count; i++) {
        if (unlimited[i]) {
            unlimitedIndices.push(i);
        }
    }
    if (remaining > 0 && unlimitedIndices.length > 0) {
        const perEmitter = Math.floor(remaining / unlimitedIndices.length);
        let extra = remaining % unlimitedIndices.length;
        for (const index of unlimitedIndices) {
            allocations[index] = perEmitter + (extra-- > 0 ? 1 : 0);
        }
    }
    return allocations;
}

interface PolygonPackCursor {
    triangle: number;
    point: number;
}

function shapeKind(shape: FluidShape): number {
    return shape.type === "box" ? 0 : shape.type === "sphere" ? 1 : shape.type === "cylinder" ? 2 : shape.type === "cone" ? 3 : shape.type === "capsule" ? 4 : 5;
}

function packFluidShape(f32: Float32Array, u32: Uint32Array, offset: number, transform: FluidTransform, shape: FluidShape, cursor: PolygonPackCursor): boolean {
    f32.set(transform.position, offset);
    f32.set(normalisedRotation(transform.rotation), offset + 4);
    f32.set(transform.scale, offset + 8);
    f32[offset + 11] = shapeKind(shape);
    switch (shape.type) {
        case "box":
            f32.set(shape.size, offset + 12);
            return true;
        case "sphere":
            f32[offset + 12] = shape.radius;
            return true;
        case "cylinder":
            f32.set([shape.radius, shape.height, shape.innerRadius ?? 0], offset + 12);
            return true;
        case "cone":
            f32.set([shape.bottomRadius, shape.topRadius, shape.height], offset + 12);
            return true;
        case "capsule":
            f32.set([shape.radius, shape.height], offset + 12);
            return true;
        case "polygonPrism": {
            const polygon = preparePolygon(shape.points);
            f32.set([shape.thickness, polygon.area, polygon.perimeter], offset + 12);
            if (
                polygon.triangles.length === 0 ||
                polygon.triangles.length > MAX_FLUID_POLYGON_TRIANGLES - cursor.triangle ||
                polygon.edges.length > MAX_FLUID_POLYGON_POINTS - cursor.point
            ) {
                return false;
            }
            u32.set([cursor.triangle, polygon.triangles.length, cursor.point, polygon.edges.length], offset + 20);
            for (const tri of polygon.triangles) {
                const to = FLOW_TRI_BASE + cursor.triangle++ * 8;
                f32.set([tri.a[0], tri.a[1], tri.b[0], tri.b[1], tri.c[0], tri.c[1], tri.cumulative / polygon.area], to);
            }
            for (const edge of polygon.edges) {
                const po = FLOW_POINT_BASE + cursor.point++ * 4;
                f32.set([edge.a[0], edge.a[1], edge.cumulative / polygon.perimeter], po);
            }
            return true;
        }
    }
}

export function createFluidFlowState(device: GPUDevice, particleCount: number, particleRadius: number, particleVolume?: number): FluidFlowState {
    const data = new ArrayBuffer(FLUID_FLOW_BYTES);
    const capacity = Math.max(1, Math.floor(particleCount));
    const state: FluidFlowState = {
        device,
        uniformBuffer: device.createBuffer({ label: "fluid-flow", size: FLUID_FLOW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        counterBuffer: device.createBuffer({ label: "fluid-flow-counters", size: FLUID_FLOW_COUNTER_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
        lifecycleBuffer: device.createBuffer({
            label: "fluid-particle-lifecycle",
            size: (FLUID_LIFECYCLE_HEADER_U32 + capacity) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        }),
        data,
        f32: new Float32Array(data),
        u32: new Uint32Array(data),
        counterData: new Uint32Array(FLOW_EMITTER_COUNTER_U32 + MAX_FLUID_SINKS),
        particleVolume: Math.max(particleVolume ?? fluidParticleVolume(particleRadius), 1e-12),
        capacity,
        config: null,
        legacyEmitter: null,
        active: false,
        activeCount: capacity,
        frameSeed: 0,
        emitterCursor: 0,
        emitterIds: [],
        emitterRates: [],
        emitterActive: [],
        emitterDelays: [],
        emitterCarries: new Float64Array(MAX_FLUID_EMITTERS),
        elapsedSeconds: 0,
        sinkIds: [],
        sinkRates: [],
        sinkCarries: new Float64Array(MAX_FLUID_SINKS),
        activeCountReadback: null,
    };
    state.u32[2] = capacity;
    resetFluidParticleLifecycle(state, capacity, capacity);
    device.queue.writeBuffer(state.uniformBuffer, 0, data);
    return state;
}

export function resetFluidParticleLifecycle(state: FluidFlowState, activeCount: number, reservedEnd: number): void {
    const active = Math.min(state.capacity, Math.max(0, Math.floor(activeCount)));
    const reserved = Math.min(state.capacity, Math.max(active, Math.floor(reservedEnd)));
    const data = new Uint32Array(FLUID_LIFECYCLE_HEADER_U32 + state.capacity);
    data[0] = active;
    data[1] = state.capacity;
    for (let i = 0; i < state.capacity; i++) {
        data[FLUID_LIFECYCLE_HEADER_U32 + i] = i < active ? FLUID_PARTICLE_ACTIVE : i < reserved ? FLUID_PARTICLE_RESERVED : FLUID_PARTICLE_FREE;
    }
    state.activeCount = active;
    if (state.activeCountReadback) {
        state.activeCountReadback.generation++;
    }
    state.device.queue.writeBuffer(state.lifecycleBuffer, 0, data);
}

export function enableFluidActiveCountReadback(state: FluidFlowState): void {
    if (state.activeCountReadback) {
        return;
    }
    state.activeCountReadback = {
        buffers: [0, 1].map((index) =>
            state.device.createBuffer({
                label: `fluid-active-count-readback-${index}`,
                size: 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            })
        ),
        states: ["idle", "idle"],
        generations: [0, 0],
        generation: 0,
        error: null,
        next: 0,
    };
}

export function pollFluidActiveCount(state: FluidFlowState): void {
    const readback = state.activeCountReadback;
    if (!readback) {
        return;
    }
    if (readback.error) {
        const error = readback.error;
        readback.error = null;
        throw error;
    }
    for (let i = 0; i < readback.buffers.length; i++) {
        if (readback.states[i] !== "copied") {
            continue;
        }
        readback.states[i] = "mapping";
        const buffer = readback.buffers[i]!;
        const generation = readback.generations[i]!;
        void buffer
            .mapAsync(GPUMapMode.READ)
            .then(() => {
                if (generation === readback.generation) {
                    state.activeCount = new Uint32Array(buffer.getMappedRange())[0] ?? state.activeCount;
                }
                buffer.unmap();
                readback.states[i] = "idle";
            })
            .catch((error: unknown) => {
                readback.error = error;
                readback.states[i] = "idle";
            });
    }
}

export function encodeFluidActiveCountReadback(state: FluidFlowState, encoder: GPUCommandEncoder): void {
    const readback = state.activeCountReadback;
    if (!readback) {
        return;
    }
    for (let offset = 0; offset < readback.buffers.length; offset++) {
        const index = (readback.next + offset) % readback.buffers.length;
        if (readback.states[index] !== "idle") {
            continue;
        }
        encoder.copyBufferToBuffer(state.lifecycleBuffer, 0, readback.buffers[index]!, 0, 4);
        readback.states[index] = "copied";
        readback.generations[index] = readback.generation;
        readback.next = (index + 1) % readback.buffers.length;
        return;
    }
}

export function setFluidFlowConfig(state: FluidFlowState, config: FluidFlowConfig | null): void {
    if ((config?.emitters.length ?? 0) > MAX_FLUID_EMITTERS) {
        throw new RangeError(`Fluid flow supports at most ${MAX_FLUID_EMITTERS} emitters.`);
    }
    if ((config?.sinks.length ?? 0) > MAX_FLUID_SINKS) {
        throw new RangeError(`Fluid flow supports at most ${MAX_FLUID_SINKS} sinks.`);
    }
    const ids = new Set<string>();
    for (const item of [...(config?.emitters ?? []), ...(config?.sinks ?? [])]) {
        if (!item.id || ids.has(item.id)) {
            throw new Error(`Fluid flow object IDs must be non-empty and unique: "${item.id}".`);
        }
        ids.add(item.id);
    }
    for (const sink of config?.sinks ?? []) {
        if (sink.volumeRate !== undefined && sink.perParticleRecycleRate !== undefined) {
            throw new Error(`Fluid sink "${sink.id}" cannot define both volumeRate and perParticleRecycleRate.`);
        }
        if (sink.perParticleRecycleRate !== undefined && (!Number.isFinite(sink.perParticleRecycleRate) || sink.perParticleRecycleRate < 0)) {
            throw new RangeError(`Fluid sink "${sink.id}" perParticleRecycleRate must be a finite non-negative number.`);
        }
    }
    for (const emitter of config?.emitters ?? []) {
        if (emitter.sourceVelocityFactor !== undefined && !Number.isFinite(emitter.sourceVelocityFactor)) {
            throw new RangeError(`Fluid emitter "${emitter.id}" sourceVelocityFactor must be finite.`);
        }
        if (emitter.normalVelocity !== undefined && !Number.isFinite(emitter.normalVelocity)) {
            throw new RangeError(`Fluid emitter "${emitter.id}" normalVelocity must be finite.`);
        }
        if (emitter.delayBeforeStart !== undefined && (!Number.isFinite(emitter.delayBeforeStart) || emitter.delayBeforeStart < 0)) {
            throw new RangeError(`Fluid emitter "${emitter.id}" delayBeforeStart must be a finite non-negative number.`);
        }
    }
    const previousEmitterCarries = new Map(state.emitterIds.map((id, index) => [id, state.emitterCarries[index]!]));
    const previousSinkCarries = new Map(state.sinkIds.map((id, index) => [id, state.sinkCarries[index]!]));
    state.config = config;
    state.legacyEmitter = config?._legacyEmitter ?? null;
    state.active = false;
    state.emitterIds = [];
    state.emitterRates = [];
    state.emitterActive = [];
    state.emitterDelays = [];
    state.emitterCarries.fill(0);
    state.sinkIds = [];
    state.sinkRates = [];
    state.sinkCarries.fill(0);
    const particleCount = state.u32[2];
    state.f32.fill(0);
    state.u32[2] = particleCount!;
    const emitters = config?.emitters ?? [];
    const sinks = config?.sinks ?? [];
    state.u32[0] = emitters.length;
    state.u32[1] = sinks.length;
    const cursor: PolygonPackCursor = { triangle: 0, point: 0 };
    const emitterPacked: boolean[] = [];
    for (let i = 0; i < emitters.length; i++) {
        const emitter = emitters[i]!;
        const offset = FLOW_EMITTER_BASE + i * FLOW_ENTITY_FLOATS;
        const packed = packFluidShape(state.f32, state.u32, offset, emitter.transform, emitter.shape, cursor);
        if (!packed) {
            throw new RangeError(`Fluid emitter "${emitter.id}" exceeds the polygon flow-buffer capacity or has an invalid polygon.`);
        }
        emitterPacked.push(packed);
        state.f32[offset + 15] = state.legacyEmitter ? (state.legacyEmitter.emitterSpeeds[i] ?? 0) : (emitter.normalVelocity ?? 0);
        state.f32.set(emitterBaseVelocity(emitter), offset + 24);
        state.f32[offset + 27] = emitter.spread;
        state.u32.set(
            [emitter.enabled && packed ? 1 : 0, emitter.behavior === "inflow" ? 1 : 0, emitter.sampling === "surface" ? 1 : 0, emitter.velocitySpace === "world" ? 1 : 0],
            offset + 28
        );
        const active = emitter.enabled && packed && emitter.behavior === "inflow";
        state.emitterIds.push(emitter.id);
        state.emitterRates.push(emitter.volumeRate);
        state.emitterActive.push(active);
        state.emitterDelays.push(emitter.delayBeforeStart ?? 0);
        state.emitterCarries[i] = previousEmitterCarries.get(emitter.id) ?? 0;
    }
    for (let i = 0; i < sinks.length; i++) {
        const sink = sinks[i]!;
        const offset = FLOW_SINK_BASE + i * FLOW_ENTITY_FLOATS;
        const sinkPacked = packFluidShape(state.f32, state.u32, offset, sink.transform, sink.shape, cursor);
        if (!sinkPacked) {
            throw new RangeError(`Fluid sink "${sink.id}" exceeds the polygon flow-buffer capacity or has an invalid polygon.`);
        }
        const operation = state.legacyEmitter && i === 0 ? FLOW_SINK_OPERATION_LEGACY : sink.mode === "recycle" ? FLOW_SINK_OPERATION_RECYCLE : FLOW_SINK_OPERATION_DELETE;
        const targetIds = new Set(sink.targets);
        let targetMask = 0;
        if (operation !== FLOW_SINK_OPERATION_DELETE) {
            for (let emitterIndex = 0; emitterIndex < emitters.length; emitterIndex++) {
                const emitter = emitters[emitterIndex]!;
                if (emitter.enabled && emitterPacked[emitterIndex] && emitter.behavior === "inflow" && targetIds.has(emitter.id)) {
                    targetMask |= 1 << emitterIndex;
                }
            }
        }
        state.u32.set([sink.enabled && sinkPacked ? 1 : 0, targetMask >>> 0, sink.volumeRate === undefined ? UNLIMITED_FLOW_BUDGET : 0, i], offset + 24);
        state.u32[offset + 28] = operation;
        state.u32[offset + 29] = sink.perParticleRecycleRate === undefined ? FLOW_SINK_RATE_VOLUME : FLOW_SINK_RATE_PER_PARTICLE;
        if (sink.perParticleRecycleRate !== undefined) {
            state.f32[offset + 30] = sink.perParticleRecycleRate;
        }
        if (i === 0 && state.legacyEmitter) {
            state.u32[offset + 29] = state.legacyEmitter.fixedStreamCount;
            state.f32[offset + 30] = state.legacyEmitter.rate;
            state.f32[offset + 31] = state.legacyEmitter.fixedStreamDrainY;
        }
        state.sinkIds.push(sink.id);
        state.sinkRates.push(sink.volumeRate);
        state.sinkCarries[i] = previousSinkCarries.get(sink.id) ?? 0;
        state.active ||= sink.enabled && sinkPacked && (operation === FLOW_SINK_OPERATION_DELETE || targetMask !== 0);
    }
    state.device.queue.writeBuffer(state.uniformBuffer, 0, state.data);
}

/** Update one already-installed emitter's dynamic transform and velocity fields without disturbing emission carries. */
export function updateFluidFlowEmitter(state: FluidFlowState, emitter: FluidEmitter): void {
    const index = state.emitterIds.indexOf(emitter.id);
    if (index < 0) {
        return;
    }
    const offset = FLOW_EMITTER_BASE + index * FLOW_ENTITY_FLOATS;
    state.f32.set(emitter.transform.position, offset);
    state.f32.set(normalisedRotation(emitter.transform.rotation), offset + 4);
    state.f32.set(emitter.transform.scale, offset + 8);
    state.f32[offset + 15] = emitter.normalVelocity ?? 0;
    state.f32.set(emitterBaseVelocity(emitter), offset + 24);
    state.device.queue.writeBuffer(state.uniformBuffer, offset * 4, state.data, offset * 4, FLOW_ENTITY_FLOATS * 4);
}

export function resetFluidFlowState(state: FluidFlowState): void {
    state.frameSeed = 0;
    state.emitterCursor = 0;
    state.emitterCarries.fill(0);
    state.sinkCarries.fill(0);
    state.elapsedSeconds = 0;
}

/** Updates exact emitter/sink budgets and initializes GPU atomic counters.
 *  Unmet whole-particle capacity is discarded each frame rather than accumulated. */
export function prepareFluidFlowFrame(state: FluidFlowState, dt: number, velocityScale = 1): FluidFlowFrame {
    if (!state.active && !state.emitterActive.some(Boolean)) {
        return { flowActive: false, deleteActive: false, emitActive: false, emitCount: 0, emitUnlimited: false };
    }
    if (state.legacyEmitter) {
        state.u32[2] = state.capacity;
        state.u32[3] = state.frameSeed++;
        state.f32[4] = dt;
        state.f32[5] = velocityScale;
        state.counterData.fill(0);
        state.device.queue.writeBuffer(state.uniformBuffer, 0, state.data);
        state.device.queue.writeBuffer(state.counterBuffer, 0, state.counterData);
        return { flowActive: state.active, deleteActive: state.active, emitActive: false, emitCount: 0, emitUnlimited: false };
    }
    const emitterBudgets = new Uint32Array(MAX_FLUID_EMITTERS);
    const emitterStarted = new Array<boolean>(state.emitterRates.length).fill(false);
    const frameStart = state.elapsedSeconds;
    const frameDuration = Math.max(0, dt);
    const frameEnd = frameStart + frameDuration;
    for (let i = 0; i < state.emitterRates.length; i++) {
        if (!state.emitterActive[i]) {
            continue;
        }
        const delay = state.emitterDelays[i] ?? 0;
        const activeDt = Math.max(0, frameEnd - Math.max(frameStart, delay));
        const started = delay <= frameStart || activeDt > 0;
        emitterStarted[i] = started;
        state.u32[FLOW_EMITTER_BASE + i * FLOW_ENTITY_FLOATS + 28] = started ? 1 : 0;
        if (!started) {
            continue;
        }
        const rate = state.emitterRates[i];
        if (rate === undefined) {
            emitterBudgets[i] = UNLIMITED_FLOW_BUDGET;
        } else {
            const budget = fluidVolumeBudget(rate, activeDt, state.particleVolume, state.emitterCarries[i]!);
            state.emitterCarries[i] = budget.carry;
            emitterBudgets[i] = budget.count;
        }
    }
    state.elapsedSeconds = frameEnd;
    if (state.emitterRates.length > 0) {
        state.emitterCursor = (state.emitterCursor + 1) % state.emitterRates.length;
    }
    state.u32[2] = state.capacity;
    state.u32[3] = state.frameSeed++;
    state.f32[4] = dt;
    state.f32[5] = velocityScale;
    state.counterData.fill(0);
    for (let i = 0; i < MAX_FLUID_EMITTERS; i++) {
        state.counterData[i * 2] = 0;
        state.counterData[i * 2 + 1] = emitterBudgets[i]!;
    }
    for (let i = 0; i < state.sinkRates.length; i++) {
        const rate = state.sinkRates[i];
        const budgetOffset = FLOW_SINK_BASE + i * FLOW_ENTITY_FLOATS + 26;
        if (rate === undefined) {
            state.u32[budgetOffset] = UNLIMITED_FLOW_BUDGET;
            continue;
        }
        const budget = fluidVolumeBudget(rate, dt, state.particleVolume, state.sinkCarries[i]!);
        state.sinkCarries[i] = budget.carry;
        state.u32[budgetOffset] = budget.count;
    }
    state.device.queue.writeBuffer(state.uniformBuffer, 0, state.data);
    state.device.queue.writeBuffer(state.counterBuffer, 0, state.counterData);
    let emitCount = 0;
    let emitUnlimited = false;
    for (let i = 0; i < emitterBudgets.length; i++) {
        if (!emitterStarted[i]) {
            continue;
        }
        const budget = emitterBudgets[i]!;
        if (budget === UNLIMITED_FLOW_BUDGET) {
            emitUnlimited = true;
        } else {
            emitCount = Math.min(state.capacity, emitCount + budget);
        }
    }
    if (emitUnlimited) {
        emitCount = state.capacity;
    }
    const emitActive = emitCount > 0;
    return { flowActive: state.active || emitActive, deleteActive: state.active, emitActive, emitCount, emitUnlimited };
}

export function disposeFluidFlowState(state: FluidFlowState): void {
    state.uniformBuffer.destroy();
    state.counterBuffer.destroy();
    state.lifecycleBuffer.destroy();
    for (const buffer of state.activeCountReadback?.buffers ?? []) {
        buffer.destroy();
    }
}

export function fluidFlowGpuBytes(state: FluidFlowState): number {
    return (
        state.uniformBuffer.size +
        state.counterBuffer.size +
        state.lifecycleBuffer.size +
        (state.activeCountReadback?.buffers.reduce((total, buffer) => total + buffer.size, 0) ?? 0)
    );
}

export function estimateFluidFlowGpuBytes(capacity: number, activeCountReadback = true): number {
    const particleCapacity = Math.max(1, Math.floor(capacity));
    return FLUID_FLOW_BYTES + FLUID_FLOW_COUNTER_BYTES + (FLUID_LIFECYCLE_HEADER_U32 + particleCapacity) * 4 + (activeCountReadback ? 8 : 0);
}

export interface FluidWarmupState {
    readonly pipeline: GPUComputePipeline;
    readonly bindGroup: GPUBindGroup;
    readonly paramsBuffer: GPUBuffer;
}

export const FLUID_LIFECYCLE_STRUCT_WGSL = /* wgsl */ `
struct FluidLifecycle {
    activeCount: atomic<u32>,
    capacity: u32,
    reserved0: u32,
    reserved1: u32,
    states: array<atomic<u32>>,
};`;

export const FLUID_LIFECYCLE_RUNTIME_WGSL = /* wgsl */ `
const FLUID_PARTICLE_FREE: u32 = ${FLUID_PARTICLE_FREE}u;
const FLUID_PARTICLE_ACTIVE: u32 = ${FLUID_PARTICLE_ACTIVE}u;
const FLUID_PARTICLE_RESERVED: u32 = ${FLUID_PARTICLE_RESERVED}u;
fn fluidParticleIsActive(index:u32)->bool{return atomicLoad(&lifecycle.states[index])==FLUID_PARTICLE_ACTIVE;}
fn fluidParticleIsFree(index:u32)->bool{return atomicLoad(&lifecycle.states[index])==FLUID_PARTICLE_FREE;}
fn fluidDeleteParticle(index:u32)->bool{
    loop{
        let current=atomicLoad(&lifecycle.states[index]);if(current!=FLUID_PARTICLE_ACTIVE){return false;}
        if(atomicCompareExchangeWeak(&lifecycle.states[index],current,FLUID_PARTICLE_FREE).exchanged){atomicSub(&lifecycle.activeCount,1u);return true;}
    }
}
fn fluidActivateParticle(index:u32)->bool{
    loop{
        let current=atomicLoad(&lifecycle.states[index]);if(current!=FLUID_PARTICLE_FREE){return false;}
        if(atomicCompareExchangeWeak(&lifecycle.states[index],current,FLUID_PARTICLE_ACTIVE).exchanged){atomicAdd(&lifecycle.activeCount,1u);return true;}
    }
}`;

const FLUID_WARMUP_WGSL = /* wgsl */ `
${FLUID_LIFECYCLE_STRUCT_WGSL}
struct WarmupParams { start:u32,end:u32,reserved0:u32,reserved1:u32, };
@group(0) @binding(0) var<storage,read_write> lifecycle:FluidLifecycle;
@group(0) @binding(1) var<uniform> p:WarmupParams;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
    let index=p.start+gid.x;if(index>=p.end||index>=lifecycle.capacity){return;}
    loop{
        let current=atomicLoad(&lifecycle.states[index]);if(current!=${FLUID_PARTICLE_RESERVED}u){return;}
        if(atomicCompareExchangeWeak(&lifecycle.states[index],current,${FLUID_PARTICLE_ACTIVE}u).exchanged){atomicAdd(&lifecycle.activeCount,1u);return;}
    }
}`;

export function createFluidWarmupState(state: FluidFlowState): FluidWarmupState {
    const pipeline = state.device.createComputePipeline({
        label: "fluid-warmup",
        layout: "auto",
        compute: { module: state.device.createShaderModule({ label: "fluid-warmup", code: FLUID_WARMUP_WGSL }), entryPoint: "main" },
    });
    const paramsBuffer = state.device.createBuffer({ label: "fluid-warmup-params", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    return {
        pipeline,
        paramsBuffer,
        bindGroup: state.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: state.lifecycleBuffer } },
                { binding: 1, resource: { buffer: paramsBuffer } },
            ],
        }),
    };
}

export function encodeFluidWarmup(state: FluidFlowState, warmup: FluidWarmupState, encoder: GPUCommandEncoder, start: number, end: number): void {
    const first = Math.min(state.capacity, Math.max(0, Math.floor(start)));
    const last = Math.min(state.capacity, Math.max(first, Math.floor(end)));
    if (last <= first) {
        return;
    }
    state.device.queue.writeBuffer(warmup.paramsBuffer, 0, new Uint32Array([first, last, 0, 0]));
    const pass = encoder.beginComputePass({ label: "fluid-warmup" });
    pass.setPipeline(warmup.pipeline);
    pass.setBindGroup(0, warmup.bindGroup);
    pass.dispatchWorkgroups(Math.ceil((last - first) / 64));
    pass.end();
}

export const FLUID_FLOW_STRUCT_WGSL = /* wgsl */ `
struct FluidShapeData { position: vec4<f32>, rotation: vec4<f32>, scaleKind: vec4<f32>, params0: vec4<f32>, params1: vec4<f32>, polygon: vec4<u32>, };
struct FluidEmitterData { shape: FluidShapeData, velocitySpread: vec4<f32>, flags: vec4<u32>, };
struct FluidSinkData { shape: FluidShapeData, route: vec4<u32>, compat: vec4<u32>, };
struct FluidFlowData {
    header: vec4<u32>, frame: vec4<f32>,
    emitters: array<FluidEmitterData, ${MAX_FLUID_EMITTERS}>, sinks: array<FluidSinkData, ${MAX_FLUID_SINKS}>,
    triangles: array<vec4<f32>, ${MAX_FLUID_POLYGON_TRIANGLES * 2}>, points: array<vec4<f32>, ${MAX_FLUID_POLYGON_POINTS}>,
};
struct FluidLaunch { position: vec3<f32>, velocity: vec3<f32>, launched: u32, };`;

/** Requires global `flow` uniform and `flowCounters` atomic-storage declarations. */
export const FLUID_FLOW_RUNTIME_WGSL = /* wgsl */ `
fn fluidHash(x0: u32) -> u32 { var h=x0; h^=h>>16u; h*=0x7feb352du; h^=h>>15u; h*=0x846ca68bu; h^=h>>16u; return h; }
fn fluidRnd(x: u32) -> f32 { return f32(fluidHash(x))/4294967296.0; }
fn fluidQuatRotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> { return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v); }
fn fluidToLocal(s: FluidShapeData, p: vec3<f32>) -> vec3<f32> {
    let q=vec4<f32>(-s.rotation.xyz,s.rotation.w); let z=s.scaleKind.xyz;
    let safe=select(vec3<f32>(1e-6),z,abs(z)>vec3<f32>(1e-6));
    return fluidQuatRotate(q,p-s.position.xyz)/safe;
}
fn fluidToWorld(s: FluidShapeData, p: vec3<f32>) -> vec3<f32> { return s.position.xyz+fluidQuatRotate(s.rotation,p*s.scaleKind.xyz); }
fn fluidInsidePolygon(s: FluidShapeData, p: vec2<f32>) -> bool {
    let start=s.polygon.z; let count=s.polygon.w; if(count<3u){return false;} var inside=false; var j=count-1u;
    for(var i=0u;i<count;i=i+1u){
        let a=flow.points[start+i].xy; let b=flow.points[start+j].xy;
        if((a.y>p.y)!=(b.y>p.y)&&p.x<(b.x-a.x)*(p.y-a.y)/(b.y-a.y)+a.x){inside=!inside;} j=i;
    }
    return inside;
}
fn fluidInsideShape(s: FluidShapeData, world: vec3<f32>) -> bool {
    let p=fluidToLocal(s,world); let k=u32(s.scaleKind.w);
    if(k==0u){return all(abs(p)<=abs(s.params0.xyz)*0.5);}
    if(k==1u){return dot(p,p)<=s.params0.x*s.params0.x;}
    if(k==2u){
        let r2=dot(p.xz,p.xz); let ro=abs(s.params0.x); let ri=min(ro,abs(s.params0.z));
        return abs(p.y)<=abs(s.params0.y)*0.5&&r2<=ro*ro&&r2>=ri*ri;
    }
    if(k==3u){
        let h=max(abs(s.params0.z),1e-6); let t=clamp(p.y/h+0.5,0.0,1.0); let r=mix(abs(s.params0.x),abs(s.params0.y),t);
        return abs(p.y)<=h*0.5&&dot(p.xz,p.xz)<=r*r;
    }
    if(k==4u){
        let r=abs(s.params0.x); let l=max(0.0,abs(s.params0.y)-2.0*r);
        let q=vec3<f32>(p.x,p.y-clamp(p.y,-l*0.5,l*0.5),p.z); return dot(q,q)<=r*r;
    }
    return abs(p.y)<=abs(s.params0.x)*0.5&&fluidInsidePolygon(s,p.xz);
}
fn fluidUnit(seed:u32)->vec3<f32>{
    let y=fluidRnd(seed)*2.0-1.0; let a=fluidRnd(seed*3u+1u)*6.28318530718; let h=sqrt(max(0.0,1.0-y*y));
    return vec3<f32>(cos(a)*h,y,sin(a)*h);
}
fn fluidDisk(r:f32,seed:u32)->vec2<f32>{
    let q=r*sqrt(fluidRnd(seed)); let a=fluidRnd(seed*3u+1u)*6.28318530718; return vec2<f32>(cos(a),sin(a))*q;
}
fn fluidAnnulus(ri:f32,ro:f32,seed:u32)->vec2<f32>{
    let q=sqrt(ri*ri+fluidRnd(seed)*(ro*ro-ri*ri)); let a=fluidRnd(seed*3u+1u)*6.28318530718; return vec2<f32>(cos(a),sin(a))*q;
}
fn fluidTriangle(s:FluidShapeData,seed:u32)->vec2<f32>{
    let st=s.polygon.x; let n=s.polygon.y; if(n==0u){return vec2<f32>(0.0);}
    let r=fluidRnd(seed); var pick=n-1u;
    for(var i=0u;i<n;i=i+1u){if(r<=flow.triangles[(st+i)*2u+1u].z){pick=i;break;}}
    let ab=flow.triangles[(st+pick)*2u]; let c=flow.triangles[(st+pick)*2u+1u];
    var u=fluidRnd(seed*3u+1u); var v=fluidRnd(seed*5u+2u); if(u+v>1.0){u=1.0-u;v=1.0-v;}
    return ab.xy+u*(ab.zw-ab.xy)+v*(c.xy-ab.xy);
}
fn fluidEdge(s:FluidShapeData,seed:u32)->vec2<f32>{
    let st=s.polygon.z; let n=s.polygon.w; if(n==0u){return vec2<f32>(0.0);}
    let r=fluidRnd(seed); var pick=n-1u;
    for(var i=0u;i<n;i=i+1u){if(r<=flow.points[st+i].z){pick=i;break;}}
    return mix(flow.points[st+pick].xy,flow.points[st+(pick+1u)%n].xy,fluidRnd(seed*3u+1u));
}
fn fluidSampleLocal(s:FluidShapeData,surface:bool,seed:u32)->vec3<f32>{
    let k=u32(s.scaleKind.w);
    if(k==0u){
        let z=abs(s.params0.xyz); var p=(vec3<f32>(fluidRnd(seed),fluidRnd(seed*3u+1u),fluidRnd(seed*5u+2u))-0.5)*z;
        if(!surface){return p;} let ax=z.y*z.z; let ay=z.x*z.z; let az=z.x*z.y; let q=fluidRnd(seed*7u+3u)*2.0*(ax+ay+az);
        if(q<2.0*ax){p.x=select(z.x*0.5,-z.x*0.5,q<ax);}
        else if(q<2.0*ax+2.0*ay){p.y=select(z.y*0.5,-z.y*0.5,q<2.0*ax+ay);}
        else{p.z=select(z.z*0.5,-z.z*0.5,q<2.0*ax+2.0*ay+az);} return p;
    }
    if(k==1u){return fluidUnit(seed)*abs(s.params0.x)*select(pow(fluidRnd(seed*7u+3u),1.0/3.0),1.0,surface);}
    if(k==2u){
        let ro=abs(s.params0.x); let h=abs(s.params0.y); let ri=min(ro,abs(s.params0.z));
        if(!surface){let xz=fluidAnnulus(ri,ro,seed);return vec3<f32>(xz.x,(fluidRnd(seed*5u+2u)-0.5)*h,xz.y);}
        let ao=6.28318530718*ro*h; let ai=6.28318530718*ri*h; let ac=3.14159265359*(ro*ro-ri*ri);
        let q=fluidRnd(seed*7u+3u)*(ao+ai+2.0*ac);
        if(q<ao){let a=fluidRnd(seed)*6.28318530718;return vec3<f32>(cos(a)*ro,(fluidRnd(seed*3u+1u)-0.5)*h,sin(a)*ro);}
        if(q<ao+ai){let a=fluidRnd(seed)*6.28318530718;return vec3<f32>(cos(a)*ri,(fluidRnd(seed*3u+1u)-0.5)*h,sin(a)*ri);}
        let xz=fluidAnnulus(ri,ro,seed);return vec3<f32>(xz.x,select(h*0.5,-h*0.5,q<ao+ai+ac),xz.y);
    }
    if(k==3u){
        let r0=abs(s.params0.x); let r1=abs(s.params0.y); let h=abs(s.params0.z); let d=r1-r0;
        if(!surface){
            let r=select(pow(max(0.0,r0*r0*r0+fluidRnd(seed)*(r1*r1*r1-r0*r0*r0)),1.0/3.0),r0,abs(d)<1e-6);
            let t=select((r-r0)/d,fluidRnd(seed*3u+1u),abs(d)<1e-6); let xz=fluidDisk(r,seed*5u+2u);
            return vec3<f32>(xz.x,(t-0.5)*h,xz.y);
        }
        let al=3.14159265359*(r0+r1)*length(vec2<f32>(d,h)); let ab=3.14159265359*r0*r0; let at=3.14159265359*r1*r1;
        let q=fluidRnd(seed)*(al+ab+at);
        if(q<al){
            let radialPick=fluidRnd(seed*3u+1u)*(r0+0.5*d);
            let t=select((-r0+sqrt(max(0.0,r0*r0+2.0*d*radialPick)))/d,fluidRnd(seed*5u+2u),abs(d)<1e-6);
            let r=r0+d*t; let a=fluidRnd(seed*7u+3u)*6.28318530718; return vec3<f32>(cos(a)*r,(t-0.5)*h,sin(a)*r);
        }
        let top=q>=al+ab; let xz=fluidDisk(select(r0,r1,top),seed*3u+1u); return vec3<f32>(xz.x,select(-h*0.5,h*0.5,top),xz.y);
    }
    if(k==4u){
        let r=abs(s.params0.x); let l=max(0.0,abs(s.params0.y)-2.0*r);
        let cm=select(3.14159265359*r*r*l,6.28318530718*r*l,surface); let sm=select(4.18879020479*r*r*r,12.56637061436*r*r,surface);
        if(fluidRnd(seed)*(cm+sm)<cm){
            let rr=r*select(sqrt(fluidRnd(seed*3u+1u)),1.0,surface); let a=fluidRnd(seed*5u+2u)*6.28318530718;
            return vec3<f32>(cos(a)*rr,(fluidRnd(seed*7u+3u)-0.5)*l,sin(a)*rr);
        }
        let sg=select(-1.0,1.0,fluidRnd(seed*3u+1u)>=0.5); let y=fluidRnd(seed*5u+2u); let a=fluidRnd(seed*7u+3u)*6.28318530718;
        let rr=r*select(pow(fluidRnd(seed*11u+5u),1.0/3.0),1.0,surface); let q=sqrt(max(0.0,1.0-y*y));
        return vec3<f32>(cos(a)*q*rr,sg*(l*0.5+y*rr),sin(a)*q*rr);
    }
    let h=abs(s.params0.x); let area=s.params0.y; let per=s.params0.z;
    if(!surface){let xz=fluidTriangle(s,seed);return vec3<f32>(xz.x,(fluidRnd(seed*7u+3u)-0.5)*h,xz.y);}
    if(fluidRnd(seed)*(2.0*area+per*h)<2.0*area){
        let xz=fluidTriangle(s,seed*3u+1u);return vec3<f32>(xz.x,select(-h*0.5,h*0.5,fluidRnd(seed*5u+2u)>=0.5),xz.y);
    }
    let xz=fluidEdge(s,seed*3u+1u);return vec3<f32>(xz.x,(fluidRnd(seed*5u+2u)-0.5)*h,xz.y);
}
fn fluidSafeNormal(v:vec3<f32>,fallback:vec3<f32>)->vec3<f32>{let l=length(v);return select(fallback,v/l,l>1e-6);}
fn fluidLocalNormal(s:FluidShapeData,p:vec3<f32>)->vec3<f32>{
    let k=u32(s.scaleKind.w);
    if(k==0u){
        let h=abs(s.params0.xyz)*0.5;let d=h-abs(p);
        if(d.x<=d.y&&d.x<=d.z){return vec3<f32>(select(-1.0,1.0,p.x>=0.0),0.0,0.0);}
        if(d.y<=d.z){return vec3<f32>(0.0,select(-1.0,1.0,p.y>=0.0),0.0);}
        return vec3<f32>(0.0,0.0,select(-1.0,1.0,p.z>=0.0));
    }
    if(k==1u){return fluidSafeNormal(p,vec3<f32>(0.0,1.0,0.0));}
    if(k==2u){
        let ro=abs(s.params0.x);let ri=min(ro,abs(s.params0.z));let radial=length(p.xz);
        let od=ro-radial;let id=select(1.0e30,radial-ri,ri>0.0);let cd=abs(s.params0.y)*0.5-abs(p.y);
        if(cd<=od&&cd<=id){return vec3<f32>(0.0,select(-1.0,1.0,p.y>=0.0),0.0);}
        let n=fluidSafeNormal(vec3<f32>(p.x,0.0,p.z),vec3<f32>(1.0,0.0,0.0));return select(n,-n,id<od);
    }
    if(k==3u){
        let r0=abs(s.params0.x);let r1=abs(s.params0.y);let h=abs(s.params0.z);let slope=(r1-r0)/max(h,1.0e-6);
        let t=clamp(p.y/max(h,1.0e-6)+0.5,0.0,1.0);let r=mix(r0,r1,t);let radial=length(p.xz);
        let ld=abs(r-radial)/sqrt(1.0+slope*slope);let bd=p.y+h*0.5;let td=h*0.5-p.y;
        if(bd<=ld&&bd<=td){return vec3<f32>(0.0,-1.0,0.0);}if(td<=ld){return vec3<f32>(0.0,1.0,0.0);}
        let q=select(vec2<f32>(1.0,0.0),p.xz/radial,radial>1.0e-6);return normalize(vec3<f32>(q.x,-slope,q.y));
    }
    if(k==4u){
        let r=abs(s.params0.x);let sh=max(0.0,abs(s.params0.y)-2.0*r)*0.5;let y=clamp(p.y,-sh,sh);
        return fluidSafeNormal(vec3<f32>(p.x,p.y-y,p.z),vec3<f32>(1.0,0.0,0.0));
    }
    let h=abs(s.params0.x)*0.5;let st=s.polygon.z;let n=s.polygon.w;var area2=0.0;var nearest=1.0e30;var edgeN=vec2<f32>(0.0,1.0);
    for(var i=0u;i<n;i=i+1u){
        let a=flow.points[st+i].xy;let b=flow.points[st+(i+1u)%n].xy;let e=b-a;area2=area2+a.x*b.y-b.x*a.y;
        let t=clamp(dot(p.xz-a,e)/max(dot(e,e),1.0e-12),0.0,1.0);let d=p.xz-(a+t*e);let d2=dot(d,d);
        if(d2<nearest){nearest=d2;edgeN=fluidSafeNormal(vec3<f32>(e.y,0.0,-e.x),vec3<f32>(1.0,0.0,0.0)).xz;}
    }
    edgeN*=select(-1.0,1.0,area2>=0.0);if(h-abs(p.y)<=sqrt(nearest)){return vec3<f32>(0.0,select(-1.0,1.0,p.y>=0.0),0.0);}
    return vec3<f32>(edgeN.x,0.0,edgeN.y);
}
fn fluidNormalToWorld(s:FluidShapeData,n:vec3<f32>)->vec3<f32>{
    let scale=select(vec3<f32>(1.0e-6),s.scaleKind.xyz,abs(s.scaleKind.xyz)>vec3<f32>(1.0e-6));
    return fluidSafeNormal(fluidQuatRotate(s.rotation,n/scale),vec3<f32>(0.0,1.0,0.0));
}
fn fluidClaimEmitter(index:u32)->bool{
    let counterIndex=index*2u;let budget=atomicLoad(&flowCounters[counterIndex+1u]);
    if(budget==0xffffffffu){atomicAdd(&flowCounters[counterIndex],1u);return true;}
    loop{let old=atomicLoad(&flowCounters[counterIndex]);if(old>=budget){return false;}if(atomicCompareExchangeWeak(&flowCounters[counterIndex],old,old+1u).exchanged){return true;}}
}
fn fluidChooseEmitter(mask:u32,seed:u32)->u32{
    var n=0u;for(var i=0u;i<${MAX_FLUID_EMITTERS}u;i=i+1u){if((mask&(1u<<i))!=0u&&flow.emitters[i].flags.x!=0u){n=n+1u;}}
    if(n==0u){return ${MAX_FLUID_EMITTERS}u;}let wanted=fluidHash(seed)%n;
    for(var attempt=0u;attempt<n;attempt=attempt+1u){
        let ordinal=(wanted+attempt)%n;var seen=0u;
        for(var i=0u;i<${MAX_FLUID_EMITTERS}u;i=i+1u){
            if((mask&(1u<<i))!=0u&&flow.emitters[i].flags.x!=0u){if(seen==ordinal&&fluidClaimEmitter(i)){return i;}seen=seen+1u;}
        }
    }
    return ${MAX_FLUID_EMITTERS}u;
}
fn fluidClaimSink(index:u32,budget:u32)->bool{
    let counterIndex=${FLOW_EMITTER_COUNTER_U32}u+index;
    if(budget==0xffffffffu){atomicAdd(&flowCounters[counterIndex],1u);return true;}
    loop{let old=atomicLoad(&flowCounters[counterIndex]);if(old>=budget){return false;}if(atomicCompareExchangeWeak(&flowCounters[counterIndex],old,old+1u).exchanged){return true;}}
}
fn fluidLegacySamplePosition(e:FluidEmitterData,seed:u32)->vec3<f32>{
    let s=e.shape;
    if(u32(s.scaleKind.w)==5u&&s.polygon.y>0u){
        let tc=s.polygon.y;let r=fluidRnd(seed*23u);var pick=tc-1u;
        for(var k=0u;k<tc;k=k+1u){if(r<=flow.triangles[(s.polygon.x+k)*2u+1u].z){pick=k;break;}}
        let ab=flow.triangles[(s.polygon.x+pick)*2u];let c=flow.triangles[(s.polygon.x+pick)*2u+1u];
        var u=fluidRnd(seed*3u);var v=fluidRnd(seed*7u);if(u+v>1.0){u=1.0-u;v=1.0-v;}
        let p=vec3<f32>(ab.x+u*(ab.z-ab.x)+v*(c.x-ab.x),(fluidRnd(seed*5u)-0.5)*abs(s.params0.x),ab.y+u*(ab.w-ab.y)+v*(c.y-ab.y));
        return fluidToWorld(s,p);
    }
    let p=(vec3<f32>(fluidRnd(seed*3u),fluidRnd(seed*5u),fluidRnd(seed*7u))-0.5)*abs(s.params0.xyz);
    return fluidToWorld(s,p);
}
fn fluidLegacyLaunch(e:FluidEmitterData,seed:u32)->FluidLaunch{
    let spread=(vec3<f32>(fluidRnd(seed*11u),fluidRnd(seed*13u),fluidRnd(seed*17u))-0.5)*(e.shape.params0.w*e.velocitySpread.w);
    return FluidLaunch(fluidLegacySamplePosition(e,seed),e.velocitySpread.xyz+spread,1u);
}
fn fluidPerParticleLaunch(e:FluidEmitterData,seed:u32)->FluidLaunch{
    let s=e.shape;let kind=u32(s.scaleKind.w);let normalSpeed=s.params0.w;var position=vec3<f32>(0.0);var local=vec3<f32>(0.0);
    if(normalSpeed==0.0&&e.flags.z==0u&&(kind==0u||kind==5u)){position=fluidLegacySamplePosition(e,seed);}
    else{local=fluidSampleLocal(s,e.flags.z!=0u,seed);position=fluidToWorld(s,local);}
    var velocity=e.velocitySpread.xyz;if(e.flags.w==0u){velocity=fluidQuatRotate(s.rotation,velocity);}
    if(normalSpeed!=0.0){velocity+=fluidNormalToWorld(s,fluidLocalNormal(s,local))*normalSpeed;}
    let spread=(vec3<f32>(fluidRnd(seed*11u),fluidRnd(seed*13u),fluidRnd(seed*17u))-0.5)*(length(velocity)*e.velocitySpread.w);
    return FluidLaunch(position,velocity+spread,1u);
}
fn fluidTryEmit(seed:u32)->FluidLaunch{
    var mask=0u;
    for(var i=0u;i<flow.header.x;i=i+1u){
        let e=flow.emitters[i];if(e.flags.x!=0u&&e.flags.y!=0u){mask=mask|(1u<<i);}
    }
    let ei=fluidChooseEmitter(mask,seed);if(ei>=flow.header.x){return FluidLaunch(vec3<f32>(0.0),vec3<f32>(0.0),0u);}
    return fluidPerParticleLaunch(flow.emitters[ei],seed);
}
fn fluidSinkCaptures(sink:FluidSinkData,world:vec3<f32>,particleIndex:u32,si:u32)->bool{
    if(sink.route.x==0u||!fluidInsideShape(sink.shape,world)){return false;}
    let perParticle=sink.compat.y==${FLOW_SINK_RATE_PER_PARTICLE}u;
    let seed=(flow.header.w*2654435761u)^(particleIndex*2246822519u)^(si*3266489917u);
    if(perParticle&&!(fluidRnd(seed)<clamp(bitcast<f32>(sink.compat.z)*flow.frame.x,0.0,1.0))){return false;}
    return fluidClaimSink(sink.route.w,sink.route.z);
}
fn fluidTryLegacyRelaunch(world:vec3<f32>,particleIndex:u32)->FluidLaunch{
    if(flow.header.x==0u){return FluidLaunch(world,vec3<f32>(0.0),0u);}
    let sink=flow.sinks[0];let fixedN=sink.compat.y;let seed=flow.header.w*2654435761u+particleIndex;
    if(fixedN>0u&&particleIndex<fixedN){
        if(world.y<bitcast<f32>(sink.compat.w)){return fluidLegacyLaunch(flow.emitters[flow.header.x-1u],seed);}
        return FluidLaunch(world,vec3<f32>(0.0),0u);
    }
    if(sink.route.x==0u||!fluidInsideShape(sink.shape,world)){return FluidLaunch(world,vec3<f32>(0.0),0u);}
    let probability=clamp(bitcast<f32>(sink.compat.z)*flow.frame.x,0.0,1.0);
    if(!(fluidRnd(seed)<probability)){return FluidLaunch(world,vec3<f32>(0.0),0u);}
    var ei=fluidHash(seed)%flow.header.x;if(fixedN>0u){ei=fluidHash(seed)%max(flow.header.x-1u,1u);}
    return fluidLegacyLaunch(flow.emitters[ei],seed);
}
fn fluidTryDelete(world:vec3<f32>,particleIndex:u32)->bool{
    for(var si=0u;si<flow.header.y;si=si+1u){
        let sink=flow.sinks[si];
        if(sink.compat.x==${FLOW_SINK_OPERATION_DELETE}u&&fluidSinkCaptures(sink,world,particleIndex,si)){return true;}
    }
    return false;
}
fn fluidTryRelaunch(world:vec3<f32>,particleIndex:u32)->FluidLaunch{
    if(flow.header.y>0u&&flow.sinks[0].compat.x==${FLOW_SINK_OPERATION_LEGACY}u){return fluidTryLegacyRelaunch(world,particleIndex);}
    for(var si=0u;si<flow.header.y;si=si+1u){
        let sink=flow.sinks[si];
        if(sink.compat.x!=${FLOW_SINK_OPERATION_RECYCLE}u||sink.route.y==0u||!fluidSinkCaptures(sink,world,particleIndex,si)){continue;}
        let seed=(flow.header.w*2654435761u)^(particleIndex*2246822519u)^(si*3266489917u);
        let ei=fluidChooseEmitter(sink.route.y,seed);if(ei>=flow.header.x){continue;}let e=flow.emitters[ei];
        return fluidPerParticleLaunch(e,seed);
    }
    return FluidLaunch(world,vec3<f32>(0.0),0u);
}`;
