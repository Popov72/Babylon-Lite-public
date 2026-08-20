# Module: Fluid Particle Lifecycle

> Package path: `packages/babylon-lite/src/fluid/`

## Purpose

Provide one solver-independent lifecycle for PBF, FLIP, MLS-MPM, and PB-MPM particles:

- `count` is immutable GPU capacity.
- `activeCount` is the number of particles currently participating in simulation and rendering.
- Initial emitters activate particles once.
- Continuous emitters independently activate free particles according to their emission budgets.
- Delete sinks deactivate captured particles and free their slots.
- Explicit recycle sinks may immediately relaunch captured particles for legacy and artistic closed-loop effects.

Sink deletion and emitter creation are separate observable operations. Reusing the same GPU slot is an implementation detail.

## Public API Surface

```ts
interface FluidEmitter {
    id: string;
    name: string;
    enabled: boolean;
    behavior: "initial" | "inflow";
    transform: FluidTransform;
    shape: FluidShape;
    sampling: "volume" | "surface";
    /** Authored directional velocity. */
    velocity: FluidVec3;
    velocitySpace: "local" | "world";
    /** Optional world-space source-object velocity, combined once when flow data is packed. */
    sourceVelocity?: FluidVec3;
    /** Multiplier applied only to sourceVelocity. */
    sourceVelocityFactor?: number;
    /** Optional speed along the analytical emitter shape's outward normal. */
    normalVelocity?: number;
    /** Legacy relative random jitter; independent from Blender Initial Velocity. */
    spread: number;
    volumeRate?: number;
}

interface FluidSink {
    id: string;
    name: string;
    enabled: boolean;
    transform: FluidTransform;
    shape: FluidShape;
    mode?: "delete" | "recycle";
    targets: string[];
    volumeRate?: number;
    perParticleRecycleRate?: number;
}
```

`mode` defaults to `delete` for newly-authored runtime graphs. Format-6 and older JSON imports infer `recycle` to preserve their historical targeted-relaunch behavior. Format-7 and newer exports write the mode explicitly. `targets` is used only by recycle sinks.

`volumeRate` limits captured world volume per second. `perParticleRecycleRate` is retained for file compatibility; it controls the per-particle capture probability for either mode and remains mutually exclusive with `volumeRate`.

Advanced initial velocity is opt-in per emitter. `velocity` and `sourceVelocity * sourceVelocityFactor` are combined on the CPU before upload. `normalVelocity` reuses the otherwise-unused fourth shape-parameter float, so the GPU emitter record remains 128 bytes and the flow buffer does not grow. The launch shader computes an analytical shape normal only when `normalVelocity` is nonzero. Emitters that omit the optional fields retain the existing launch path and memory footprint.

The fluid controls panel is user-resizable in both axes within the viewport. Its shared General section owns method selection and the PB-MPM Material selector, so every host using the shared panel exposes the same Liquid, Elastic, Sand, and Viscoelastic choices when PB-MPM is active. Compact emitter/sink numeric editors use locale-independent decimal text entry and commit on change, accepting either `.` or `,` while retaining their previous value when parsing or range validation fails.

## GPU Lifecycle State

Every solver owns one lifecycle storage buffer:

```text
u32 0       atomic activeCount
u32 1       capacity
u32 2..3    reserved
u32 4+i     atomic state for particle slot i
```

Particle states:

```text
0 = free
1 = active
2 = reserved for initial-emitter warm-up
```

The buffer is initialized on reset:

- the released prefix is active;
- the unreleased part of the initial-emitter allocation is reserved;
- remaining capacity is free for continuous emitters.
- an explicitly installed empty flow graph (`{ emitters: [], sinks: [] }`) starts with zero active particles;
- a `null` flow graph retains the legacy solver spawn-box initialization.

Warm-up changes reserved slots to active through an atomic GPU pass. It never exposes those slots to continuous emitters prematurely.

## Per-Frame Lifecycle

Before solver integration:

1. Release the next initial-emitter warm-up range.
2. Compute independent emitter and sink budgets from `dt`, particle volume, and fractional carry.
3. Run the sink pass over capacity:
    - inactive slots are ignored;
    - recycle sinks relaunch only when a target emitter has remaining budget;
    - delete sinks atomically change state `active -> free` and decrement `activeCount`.
4. Run the emission pass over capacity:
    - only free slots participate;
    - an enabled inflow with remaining budget is selected fairly;
    - the slot is initialized for the active solver;
    - state changes `free -> active` and increments `activeCount`.
5. Run the solver. Particle-indexed passes skip non-active slots. Spatial sorting includes active slots only.

Emission never occurs merely because `activeCount < count`; it requires an enabled emitter and available emitter budget. Unlimited inflows may consume all currently free slots.

## Active Count Readback

Simulation and emission remain fully GPU-driven. A small double-buffered asynchronous readback copies only the four-byte `activeCount`; it updates `FluidSim.activeCount` without stalling frame submission. Rendering continues to draw capacity instances, with inactive render positions parked off-screen.

## Solver Integration

### PBF

- Original-order predict, force, histogram, and scatter passes test the lifecycle state.
- The histogram/prefix/scatter sequence produces a dense active sorted range.
- The GPU active count is copied into the existing simulation-count uniform before sorted constraint passes.
- Deleted slots are parked off-screen immediately.

### FLIP

- P2G, force, and G2P passes test lifecycle state.
- Delete sinks park freed slots off-screen; emitters initialize particle position and velocity directly.
- Warm-up restores retained seed position and velocity before changing a reserved slot to active.
- Cell marking and MAC-grid transfer include active slots only.

### MLS-MPM

- Particle histogram, force, G2P, and render-copy passes test lifecycle state.
- P2G block lists contain active particles only.
- Render copy writes inactive instances off-screen.

### PB-MPM

- Constraint, force, P2G, G2P, integration, and render-copy passes test lifecycle state.
- Render copy writes inactive instances off-screen.

## Legacy Compatibility

`legacyEmitterConfigToFluidFlow()` writes `mode: "recycle"` and retains its fixed-stream metadata. Legacy presets therefore keep their exact pump behavior. New Blender exports write format 7 delete sinks and independent inflows.

## Validation

- Unit tests cover lifecycle initialization, warm-up reservation, independent budgets, default delete mode, legacy recycle mode, and format migration.
- Focused browser coverage imports a current-format Blender JSON, verifies active count falls when inflow is disabled, then verifies an enabled inflow refills free capacity at its configured rate.
- PBF, MLS-MPM, and PB-MPM must report the same lifecycle behavior.

## File Manifest

- `packages/babylon-lite/src/fluid/sim-common.ts`
- `packages/babylon-lite/src/fluid/pbf-sim.ts`
- `packages/babylon-lite/src/fluid/flip-sim.ts`
- `packages/babylon-lite/src/fluid/mls-mpm-sim.ts`
- `packages/babylon-lite/src/fluid/pbmpm-sim.ts`
- `lab/lite/src/demos/fluid/blender-fluid-json.ts`
- `lab/lite/src/demos/fluid/preset-io.ts`
- `lab/lite/src/demos/fluid.ts`
- `config/fluid-flow.schema.json`
- `scripts/blender-fluid-addon.py`
