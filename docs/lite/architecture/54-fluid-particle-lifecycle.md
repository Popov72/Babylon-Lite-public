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
type FluidSimulationSamplingType = "fluid" | "mesh";

interface FluidSimulationDiscretization {
    /** Non-FLIP particle-size multiplier authored by the shared fluid controls. */
    physicsParticleSize: number;
    /** Fluid graphs derive radius from physicsParticleSize; mesh sampling uses particleRadius. */
    samplingType: FluidSimulationSamplingType;
    /** Mesh-sampling particle radius in world units. */
    particleRadius?: number;
}

function fluidParticleRadiusForPhysicsScale(physicsScale: number): number;
function fluidCellSizeForParticleRadius(method: string, particleRadius: number): number;
function fluidSimulationParticleRadius(config: FluidSimulationDiscretization): number;
function fluidSimulationCellSize(method: string, config: FluidSimulationDiscretization): number;
function fluidSimulationParticleCapacity(method: string, requestedCount: number, flow: FluidFlowConfig, particleVolume: number): number;

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

All hosts use these package-level functions rather than owning solver discretization rules. Fluid
sampling maps `physicsParticleSize` to radius as `0.08 * physicsParticleSize`; mesh sampling uses
the positive finite authored `particleRadius`, falling back to that same mapping. PBF cell size is
`max(radius * 4, 0.3)`. MLS-MPM and PB-MPM cell size is `max(radius * 2.4, 0.18)`. FLIP owns its
separate `gridResolution` discretization and does not use `fluidSimulationCellSize`.

`fluidSimulationParticleCapacity` honors a positive authored capacity. A missing/non-positive
capacity derives a default from enabled initial/inflow emitters. FLIP additionally guarantees that
capacity can contain all initial-emitter particles. This is the single capacity rule for every
host.

PBF, MLS-MPM, and PB-MPM receive the same radius-derived particle volume used by shared capacity
planning. Their reset-time emitter density and emitter/sink volume-rate budgets therefore agree
with the controls preview. PB-MPM additionally uses that value for its liquid rest-volume
projection. Physics particle size consequently changes represented liquid volume consistently
across the non-FLIP backends. When an exact volume emitter requires more particles than the selected
capacity, reset keeps the requested lattice spacing and takes complete lower layers instead of
thinning particles uniformly across the entire emitter.

`mode` defaults to `delete` for newly-authored runtime graphs. Format-6 and older JSON imports infer `recycle` to preserve their historical targeted-relaunch behavior. Format-7 and newer exports write the mode explicitly. `targets` is used only by recycle sinks.

`volumeRate` limits captured world volume per second. `perParticleRecycleRate` is retained for file compatibility; it controls the per-particle capture probability for either mode and remains mutually exclusive with `volumeRate`.

Emitters and sinks may independently specify `delayBeforeStart` in simulation-time seconds. A delayed sink is excluded from delete and recycle matching until its delay is reached. The crossing frame uses only its post-delay duration for finite world-volume budgets and per-particle capture probability.

Advanced initial velocity is opt-in per emitter. `velocity` and `sourceVelocity * sourceVelocityFactor` are combined on the CPU before upload. `normalVelocity` reuses the otherwise-unused fourth shape-parameter float, so the GPU emitter record remains 128 bytes and the flow buffer does not grow. The launch shader computes an analytical shape normal only when `normalVelocity` is nonzero. Emitters that omit the optional fields retain the existing launch path and memory footprint.

The fluid controls panel is user-resizable in both axes within the viewport. Its shared General section owns method selection and the PB-MPM Material selector, so every host using the shared panel exposes the same Liquid, Elastic, Sand, and Viscoelastic choices when PB-MPM is active. Compact emitter/sink numeric editors use locale-independent decimal text entry and commit on change, accepting either `.` or `,` while retaining their previous value when parsing or range validation fails.

The Fluid and AquanovaFluidSim demos share the same lab-side emitter/sink editor. Its per-object wireframe and transform-gizmo controls are part of the default shared UI contract; hosts must provide the visualization adapter, and may omit the rows only through the explicit `hideVisualControls` option. AquanovaFluidSim exposes two explicit modes: Mesh liquefaction samples a clicked Aquanova ship mesh into a dedicated solver, while Fluid starts a solver from the authored flow graph. The mode selector hides controls that do not apply to the active workflow. Both modes use the domain configured by Grid position and Grid size in the shared Physics section. Fluid mode treats Grid position as an absolute world-space center; Mesh liquefaction treats it as an offset from each liquefied mesh's world-space AABB center. Emitter and sink positions remain grid-local. The optional bounds wireframe and position/scale gizmos edit that same domain. Both modes apply the current solver, render, foam, and analytical collision settings. Each start selects authored ship collision primitives intersecting a configurable world-space neighborhood sphere; mesh starts exclude the liquefied placement itself. In Fluid mode the Demo action and `R` share a stop/start state machine: the first action stops a running solver, and the next starts a new solver from the current controls. Construction-time method, grid, capacity, paging, material, physics, and foam edits prepare every live replacement through the shared facade before committing any of them. Allocation or configuration failure cancels the full candidate batch, preserving the running simulations, render bindings, and committed control snapshot; successful commits keep stable simulation handles and retire old GPU resources after submission. The shared particle-usage rows show current and projected particle and GPU-memory usage, while an unallocated solver is reported as Not running rather than Calculating. Shift+RMB drag injects the shared cursor-ray force without rotating the camera; its mouse-speed multiplier and world-space influence radius are authored in the Demo section and persisted in `demoParams`.

AquanovaFluidSim reads environment grading and material reflection settings from `/aquanova/ship_manifest.json`. Its simulation selector reads the manifest's `fluidSim` list and loads presets from `/aquanova/fluidSim`. Selecting or importing a different preset retains a shared `FluidPresetSession`, prepares live solver replacements before committing the new state, and preserves recursive forward fields and lifecycle values on export. Production Aquanova imports the same session state and delegates solver normalization to the same facade options path. Presets store the demo's FreeCamera pose in a dedicated `freeCamera` object containing `position` and `target`; the legacy `camera` object remains reserved for ArcRotate alpha/beta/radius framing. During local lab development, the Vite authoring endpoint creates, updates, and deletes those preset files while keeping the manifest list synchronized; static deployments expose loading and export only, and report that persistence requires the development server.

Production combines live simulations with `FluidSimulationCollection` render layers for surfaces and foam. Its gameplay/electricity stream uses manual collection refresh: the facade records current source aggregation into the frame encoder before spatial-count and electricity queries, so query offsets and particle data always describe the same frame without exposing WebGPU handles to host policy.

The Demo section's Aquanova rendering checkbox reloads the authoring demo into the ship's production lighting path. The mode builds the authored clustered and chunk-scoped lights before scene registration and assigns each PBR ship mesh one immutable intersecting box-projected local cubemap. Per-fragment cubemap blending and portal visibility rendering are intentionally absent: the camera-selected dominant probe only supplies the fluid surface reflection, while the ordinary scene task submits the entire visible ship every frame so fluid performance is measured against a stable real-world rendering workload. In either render mode, Shift+LMB on a ship mesh toggles a combined world-translation/local-rotation gizmo for that exact primitive; ordinary LMB remains reserved for liquefaction in Mesh liquefaction mode.

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

- `packages/babylon-lite/src/fluid/core/sim-common.ts`
- `packages/babylon-lite/src/fluid/solvers/pbf-sim.ts`
- `packages/babylon-lite/src/fluid/solvers/flip-sim.ts`
- `packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts`
- `packages/babylon-lite/src/fluid/solvers/pbmpm-sim.ts`
- `packages/babylon-lite/src/fluid/authoring/blender-fluid-json.ts`
- `packages/babylon-lite/src/fluid/controls/flow-editor.ts`
- `packages/babylon-lite/src/fluid/authoring/preset-io.ts`
- `lab/lite/src/demos/fluid.ts`
- `lab/lite/src/demos/aquanova-fluid-sim.ts`
- `lab/lite/src/demos/aquanova/collision-field.ts`
- `lab/lite/src/demos/aquanova/collision-shapes.ts`
- `config/fluid-flow.schema.json`
- `scripts/blender-fluid-addon.py`
