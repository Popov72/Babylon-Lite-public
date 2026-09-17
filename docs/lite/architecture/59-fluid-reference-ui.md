# FLIP Reference UI integration

## Scope

Expose the reference implementation as an explicitly selected FLIP backend, preserving the
intrinsic FLIP discretization and existing production defaults. The first integration supports
particle spheres, the existing screen-space water renderer, and opt-in foam/spray/bubbles.
Polygon reconstruction remains unsupported.
Unsupported physics/flow features are visible limitations, never silent no-ops.

Select **FLIP** in the existing method dropdown, then choose **FLIP Reference (experimental)**
under **FLIP implementation**. The implementation selector occupies the shared panel's
`implementationSlot`; it is not another intrinsic fluid method.

## Backend extension

`FluidSimulationOptions.backend` is an optional pure-state provider with identity, intrinsic
method, stepping mode, and supported rendering/physics metadata. Its package-internal factory
creates the existing `FluidSim` contract. The reference provider is loaded explicitly through a
root-exported async loader; ordinary fluid simulations do not import the reference implementation.

`loadFlipReferenceBackend(): Promise<FluidSimulationBackend>` loads the provider. The provider's
internal allocation-plan hook uses the same core buffer-size functions as real allocations, plus
its stable publication buffers and collision-sampler uniform. Exact imported seeds produce exact
byte projections; ordinary initial-volume previews conservatively use particle capacity.

The interactive provider uses `steppingMode: "frame"` and records its entire simulation frame
into the engine encoder. The explicitly awaited `submitFluidSimulationStep` remains available
for offline capture, but interactive execution never waits for a mapping or queue fence.
The original awaited core solver remains a numerical comparison oracle, not the UI scheduler.

## GPU-resident execution

`gpu-runtime.ts` owns per-instance control storage, dispatch arguments, component scratch,
frame parameters, and a bounded asynchronous telemetry ring. `recordFlipReferenceFrame` records
work without submitting or waiting. Its frame parameters are `(frameDt, maxSubDt, minSubsteps,
cfl)`. The shared solver WGSL is generated for CPU-oracle or GPU-runtime count/timestep access;
the transfer, pressure matrix, projection, extrapolation, advection, and removal math is shared.

Runtime storage is 32 atomic u32 words (f32 values use bit representations):

| Word  | Meaning                                                                                     |
| ----- | ------------------------------------------------------------------------------------------- |
| 0     | Live working particle count                                                                 |
| 1     | Sticky failure bits                                                                         |
| 2-3   | Current substep enabled flag and completed substep count                                    |
| 4-7   | Substep dt, frame dt, remaining frame time, maximum surviving speed                         |
| 8-12  | Active pressure cells, conditioned components/cells, sealed components, maximum sealed flux |
| 13-15 | Completed frames and cumulative contact/extreme removal counts                              |
| 16    | Successfully published simulation time                                                      |
| 17-19 | Cumulative skipped time, last-frame skipped time, last-frame consumed time                  |
| 20    | Previous successfully solved physical pressure substep dt                                   |

The scheduler chooses `min(remaining, remaining / requiredMinimumSteps, maxSubDt,
cfl * dx / maxSpeed)` on the GPU, with the CFL bound omitted when disabled or speed is zero.
It recomputes this after every substep using the surviving markers' maximum speed. `maxSubsteps`
is a real recording budget. On exhaustion, publish the successfully completed stable substeps,
report the remaining time as skipped, and continue on the next frame. Do not stretch the last
substep beyond its CFL or hard-dt limit, exceed the recording budget, or retain a catch-up debt.
The scene/requested-frame clock continues normally; fluid time under-advances when the budget
is insufficient, and this timing loss is explicitly shown in the panel.
The adapter rejects substep budgets above 256 at configuration boundaries rather than allocating
an unbounded command stream.

Each substep records transfer and matrix assembly, GPU connected-component conditioning and
gauges, a convergence-controlled pressure solve, projection/extrapolation, advection, stable
compaction, and count/timestep finalization. Dispatch arguments are in a separate storage/indirect
buffer; no dispatch simultaneously binds its indirect input as writable storage. Inactive
substeps dispatch zero workgroups. The pressure solver must loop to the existing true-residual
tolerance on the GPU, not emit a maximum-iterations command skeleton: zero-work indirect commands
still have significant GPU command-processing cost.

The component graph uses monotone atomic root linking, with dispatch boundaries between linking,
aggregation, conditioning, and gauge finalization; it has no cross-workgroup spin barrier.
Conditioning uses the float32 `1e-6` aperture threshold and skips singletons. Gauge connectivity
uses every positive aperture. Sealed-component compatibility retains the existing flux test and
reports failure instead of subtracting an arbitrary RHS mean.

`gpu-components.ts` adds binding 12 with exactly 32 bytes per dense cell: conditioning parent,
gauge parent, conditioning count, gauge count, conditioning air, gauge air, flux bits, and
squared-RHS sum bits, all atomic u32 fields. Roots decrease monotonically through weak-CAS
retry linking; aggregation compresses paths only after the linking dispatch has finished.
Air-connected components do not accumulate floating-point flux. Sealed components sum finite
RHS and RHS-squared values using CAS, reject
`abs(flux) > 4 * sqrt(cellCount) * max(absTolerance, relTolerance * sqrt(rhsSquared))`, and
set the minimum-index root's `positive.w` gauge flag. The matrix operator excludes gauge
neighbors symmetrically.

Component entry points are `initializeGpuComponents`, `linkGpuComponents`,
`aggregateGpuComponents`, `markGpuClosedPockets`, `sumGpuSealedFlux`, and
`fixGpuPressureGauges`, each using 128-thread workgroups. Initialization reuses the no-longer-needed
particle linked-list heads for compact liquid indices. All subsequent component kernels traverse
this active list; its length lives in runtime word 8. The status tail is never overwritten.

The pressure system and UI thresholds remain unchanged: Jacobi-PCG, relative L2 `1e-5`, absolute
L2 `1e-8`, at most 400 iterations, with recomputed true residual and restart when recursive
convergence is insufficient. An unconverged or nonfinite result cannot reach publication.

`gpu-pressure.ts` implements `solveGpuPressure` as one 256-invocation workgroup after the
full-grid `prepareGpuPressure` and `initializeGpuPressure` dispatches (128 invocations each).
Each solver lane traverses compact liquid indices with stride 256.
Matrix products, pressure/residual updates, and direction updates are separated by storage
barriers; scalar products use a workgroup binary-tree reduction and `workgroupUniformLoad`.
Only workgroup-local barriers are used, so scheduling never relies on simultaneous residency
of independent workgroups. It requires 4,104 bytes of workgroup storage plus the cached
operator buffer described below. The existing PCG controls retain residual/rhs squared
norms and iteration count.

Recursive convergence always triggers a recomputed `b - A*x` residual. Failure to meet the same
tolerance restarts the direction from the preconditioned true residual; the total iteration
budget is never reset. Pressure codes are `1` non-positive direction product, `2` nonfinite
state, `3` stalled residual replacement, `4` bounded nonconvergence, and `5` invalid runtime
configuration. Failure latches runtime bit 16 before any projection or publication.

This deliberately trades GPU parallelism for eliminating hundreds of synchronized pressure
dispatches. A large connected liquid region can take more GPU execution time than the
multi-workgroup oracle even while its submission-to-completion latency is substantially lower.
The algorithm, tolerance, and successful-state publication contract are not weakened to hide
that tradeoff.

### Pressure and command-cost optimizations

Pressure preprocessing caches the six neighbor indices, their gauge-adjusted coefficients,
the diagonal, inverse diagonal, and RHS in binding 19, an instance-owned GPU buffer with
exactly 64 bytes per dense cell. Four aligned vec3/scalar rows contain positive coefficients
and diagonal, negative coefficients and inverse diagonal, positive neighbor IDs and RHS,
and negative neighbor IDs and flags. Flag masks are 1 for liquid, 2 for an unusable warm
guess, and 4 for an invalid matrix row. Rebuild the cache after matrix assembly and
gauge finalization, not once per CG iteration. The true-residual
acceptance criterion remains based on the current system's RHS norm, never the residual of
an approximate initial guess. A pressure warm start scales the previous pressure impulse by
the ratio of current to previous physical substep dt; reset and the first solve start cold.
Discard a nonfinite or worse-than-zero initial guess rather than weakening convergence.
Pressure history belongs to successful solver substeps, not requested frame duration or CPU
telemetry. Runtime word 20 is reserved for the previous pressure substep dt; words 0-19 retain
their existing scheduler, publication, and budget-diagnostic meanings.

Preparation rewrites guesses and clears the other PCG components. Residual initialization
reads neighboring guesses but writes only the local residual/direction components, never x;
even rewriting x with the same value would introduce a storage read/write race. The solver
checks the active list and matrix cache once before iteration. Iterative products use cached
rows, while acceptance recomputes the true residual through the original uncached operator.
An invalid matrix remains fatal; discarding a bad warm guess only chooses the valid cold solve.

`clearGpuParticleStage` clears particle-cell lists together with removal counters, using the
same helper bodies as the CPU oracle's separate entry points. Liquid level-set construction
and particle-to-grid transfer remain separate: fusing them did not provide a reliable GPU gain.

`conditionGpuMatrix` combines marked-face solid-velocity clearing with an RHS-only update,
reusing the open volumes cached by the first matrix assembly. It does not recompute geometry,
diagonals, or coefficients. The RHS reads individual face area/velocity fields and reads solid
velocity only on unmarked faces, which no invocation writes; marked faces use a virtual zero.
This avoids a cross-invocation read/write race. A zero conditioned-component count skips the
body, and gauge finalization still follows this dispatch. The original CPU conditioning and
full matrix-rebuild entry points remain an independent numerical oracle.

These changes must not reduce `maxSubsteps`, extrapolation layers, particle counts, resolution,
or pressure accuracy to obtain their speedup.

Failure bits are `1` invalid particles/compaction, `2` unresolved collision, `4` advection budget,
`8` unsupported velocity-band sampling, `16` pressure/matrix failure, `32` incompatible sealed
flux, and `64` an unrepresentable/nonprogressing timestep. Ordinary substep-budget exhaustion
is not a failure bit. The adapter requires eight compute storage
bindings and 256-invocation workgroups; it does not silently fall back to the readback-driven oracle on smaller devices.

## Publication and lifecycle

The adapter owns stable rendering position, velocity, and speed buffers separately from the
reference solver's working buffers. Only completed simulation state is copied into the rendering
buffers. Capacity stays fixed. A separate 16-byte draw-indirect buffer holds
`(6, publishedCount, 0, 0)` and is authoritative for sphere and surface draws. Anisotropy
copies its live count from this buffer into uniform byte offset 32 before compute; its dispatch
range may cover capacity but its shaders guard against the GPU live count. CPU active/render
counts are delayed diagnostics, never inputs to GPU work.

Simulation, publication, and rendering are ordered in the same GPU command stream. Publication
requires valid, converged completed substeps, but not consumption of the entire requested frame.
An insufficient substep budget is a nonfatal warning; invalid pressure, particles, or collisions
still leave the previous publication intact. Reset replaces working state and immediately
publishes the seed; stale telemetry is ignored by generation. Mapping diagnostic copies is
nonblocking and bounded, and no callback submits additional physics work.

The adapter exposes the existing `FluidTimestepDiagnostics` shape: `deferredSeconds` is zero,
`droppedSeconds` is cumulative skipped time since reset, and `saturated` records whether any
budget was exhausted. The panel displays this independently from fatal errors and does not pause
playback. Reset clears this warning along with the GPU clock/counters. R also releases an
error-induced pause after a successful reset, while preserving an intentional user pause.
Offline capture likewise continues and its CLI progress distinguishes requested, simulated,
and skipped time, so a budget-limited recording is not mislabeled as full physical advancement.
Imported animations and collision motion history rewind together with particles on plain R;
the existing Shift+R behavior that preserves animation is unchanged.
For analytical initial volumes, source-node motion preserves the authored shape-to-source
transform (see module 53); resetting must not substitute the GLB node's origin/axes for the
exported prism or box transform. Regression probes compare an analytical fresh start with
both reset-time particles and later replay, not only an exact baked seed.

The scene owner advances animation once to the requested frame-end pose. Before each physical
substep, the GPU samples SceneSDF at temporal offset `consumed + substepDt - frameDt`;
moving normal velocity uses centered temporal differences around that offset. This uses the
shared SceneSDF temporal model (including its existing inverse-transform interpolation), not a
second CPU animation clock. The current shared binding remains the owner of its buffers. Initial emitters reuse the
shared initial-particle sampling code; exact imported positions and velocities take precedence.
An explicitly empty flow stays empty; only the legacy null-flow case uses spawn-box fallback.

## Persistence and transitions

`PairState.backendId` and `FluidExportJson.backendId` optionally store `"flip-reference"`; omission
means production FLIP. Provider objects/functions are never serialized. Unknown IDs and Reference
IDs on non-FLIP methods are rejected. Reference exports retain the six supported physical/timestep
keys and preserve shared whitewater settings; they cannot re-enable polygons or continuous flow.

Implementation switching preserves the authored grid, initial emitters, capacity, camera, and scene
parameters, while keeping implementation-specific physics/render state separate. Imported animations
rewind together with the reset liquid. Changing exact imported seed data forces reconfiguration even
when all numerical settings and particle counts remain identical.

Both implementations use the normal engine loop. `reference-pump.ts` remains the transition
barrier for asynchronous imports and implementation changes, not an awaited interactive physics loop.
The loop continues rendering while paused.
`reference-timing.ts` maps declared physical FPS to glTF timeline FPS without applying native time scale
twice. Ordinary demos keep a 1/60-second base step and a shared animation clock.

Screen-space anisotropy remains supported: its renderer computes the needed data from the same published
particle buffers. The UI deliberately does not feed Reference working buffers directly to rendering.

## Initial limitations

The reference backend is dense and initial-only. Polygon reconstruction, active inflows/sinks,
viscosity, and surface tension remain unsupported. Shared
render settings such as water color, absorption, refraction, and screen-space filtering continue to
control the existing renderer.

## Foam, spray, and bubbles

Whitewater is an opt-in, one-way secondary simulation: it samples Reference particles, the
projected MAC field, liquid phi, and collision nodes but never writes any liquid buffer or control
word. It uses the shared `FoamConfig`, `FOAM_COMMON_WGSL`, 32-byte `Diffuse` slot layout, and
existing foam renderer, not a second liquid solver. This is shared-style whitewater over Reference
fields, not a claim of exact native FLIP Fluids whitewater parity.

`whitewater.ts` owns all optional pipelines, buffers, configuration, publication, and telemetry.
The adapter calls it once after `recordFlipReferenceFrame`, outside the liquid Simulation timing
span, with its compute passes tagged **Foam gen**. It reads runtime word 19 (successfully consumed
frame time), not requested time, for generation, motion, and lifetime. A core error or zero consumed
time disables its GPU work and publication; budget-limited but valid frames still advance it by the
actual consumed time. No CPU readback gates either simulation.

The field-preparation pass derives surface normals and dimensionless curvature from cell phi
(which is stored in cell units), and optional curl/strain turbulence from the MAC velocities.
A 32-byte per-cell whitewater field caches `(normal.xyz, curvature)` and
`(phi, turbulence, interfaceStrength, openVolume)`. Sampling outside the grid returns air/no
fluid velocity. With no live liquid, stale core grid data must not keep foam or bubbles submerged.
Whitewater velocity sampling never calls a liquid sampler that writes liquid error/status flags.

Generation follows the existing trapped-air, wave-crest, optional turbulence, speed, upward-surface,
radius, and lifetime controls. Spawn counts are stochastically rounded with a per-frame GPU seed.
The expected count is capped at eight per liquid marker before float-to-integer conversion, so
large generation rates cannot overflow the birth counter. Threshold intervals must remain strictly
ordered after conversion to f32; rejecting collapsed intervals avoids zero-denominator shader ramps.
Existing whitewater updates before new emission: spray is ballistic with optional aerodynamic
drag, bubbles receive buoyancy and drag toward the sampled liquid velocity, and surface foam is
advected by that velocity with the existing surface/lifetime behavior. Surface-depth and normal
hysteresis keep foam from flickering between classifications. Kind IDs remain spray 0, foam 1,
bubble 2. Each generation toggle also removes existing particles that classify into a disabled kind.
Collision handling samples the existing Reference nodal SDF without affecting the liquid.

The working pool has a GPU-owned free-slot stack and two compact active lists. It reuses the
shared 64-u32 header/aligned-list layout; the final capacity-sized region stores free indices
instead of flags. Update returns dead slots to the stack before emission pops them, with a dispatch
boundary between the two operations. Unique CAS pops ensure no two invocations write the same
new slot. A full pool rejects new births rather than wrapping and racing writes to live slots.
The active-particle mode uses indirect live-count dispatch; the dense mode visits capacity with
dead-slot guards and retains the same particle semantics.

Publication uses separate stable pool/list/draw buffers so failed liquid frames cannot reveal
in-flight whitewater against old water. The published live prefix is compact, with a fixed side-0
identity list after the standard header, compatible with all existing `DiffusePool` consumers.
GPU counts and draw arguments are authoritative. Delayed total/spray/foam/bubble counters are
telemetry only; failed mappings or invalid capacity/type totals clear the diagnostic sample
and log a warning, never pause liquid stepping. No particle arrays are downloaded for simulation
or rendering.

Enable allocates only the configured pool and optional field resources; live numerical settings
reuse them, while a pool resize clears the secondary particles, matching the existing pool-size
behavior. Disable releases the optional resources. Reset recreates/rebinds them against the new
liquid generation and clears whitewater, seeds, counts, and stale telemetry. Force input continues
to affect whitewater indirectly through the liquid. Memory previews and runtime allocation use
the same capacity/resource planner, including the publication copy and telemetry buffers.

### Whitewater API and allocation

Internal entry points in `whitewater.ts`:

```typescript
planFlipReferenceWhitewater(particleCapacity: number, cells: number, config: FoamConfig,
    limits?: FluidDeviceLimitsSnapshot): FlipReferenceWhitewaterPlan;
createFlipReferenceWhitewater(core: FlipReferenceSimulation, gpu: FlipReferenceGpuRuntime,
    particleCapacity: number, config: FoamConfig): FlipReferenceWhitewater;
configureFlipReferenceWhitewater(ww: FlipReferenceWhitewater, config: FoamConfig): void;
recordFlipReferenceWhitewater(ww: FlipReferenceWhitewater, encoder: GPUCommandEncoder,
    profiler?: FluidProfiler | null): void;
collectFlipReferenceWhitewaterStatus(ww: FlipReferenceWhitewater): Promise<void>;
disposeFlipReferenceWhitewater(ww: FlipReferenceWhitewater): void;
```

The plan returns `capacity`, `resources`, and explicit `errors`. The state exposes `pool`, `bytes`,
`ready`, and a latched initialization `error`, and owns its pipelines, bind groups, optional buffers,
two telemetry slots, configuration, and last recorded encoder. No shared liquid resource is owned
or destroyed by this module.

Capacity is at least 1024, starting from `round(particleCapacity * poolScale)`, capped by
`poolCapMax`, buffer/storage limits, aligned-list size, and 2D dispatch capacity. Devices unable to
support the minimum fail planning. With `C` slots, `N` cells, and
`S = (64 + 2 * ceil(C / 64) * 64 + C) * 4` bytes:

| Resource | Bytes | Usage beyond COPY_DST |
|---|---:|---|
| Shared Foam parameters | 128 | UNIFORM |
| Cached surface field | `32 * N` | STORAGE |
| Working particle pool | `32 * C` | STORAGE |
| Working header, active A/B, free stack | `S` | STORAGE, COPY_SRC |
| Field/update/emission dispatch arguments | 36 | STORAGE, INDIRECT |
| Published particle pool | `32 * C` | STORAGE, COPY_SRC |
| Published header and fixed identity list | `S` | STORAGE, COPY_SRC |
| Published draw arguments | 16 | STORAGE, COPY_SRC, INDIRECT |
| Publication dispatch arguments | 12 | STORAGE, INDIRECT |
| Two telemetry buffers | 64 total | MAP_READ |

Working header words 0–7 hold the frame seed, A/B counts, current side, capacity, free count,
publication sequence, and rejected-birth/overflow count. Words 12/13 hold active/dense mode and
the device's maximum workgroups per dimension. Submit the initialization clear/list-fill encoder
**before** writing those settings: an earlier queue write would be erased by the clear.
Active lists start at word 64 and are individually padded to 64 entries; the free stack follows
both lists. Published state uses side 0, capacity in word 4, and total/spray/foam/bubble counts in
words 8–11. Draw arguments are `[6, liveCount, 0, 0]`.

### Whitewater compute contract

One compute pass records prepare, cached field, update, emission, finish, then publication.
Prepare/finish use one invocation; the other kernels use 64. Prepare builds 2D indirect arguments
from GPU counts, with update choosing the live list or full capacity. Publication always covers
capacity to clear the inactive tail. Failed liquid frames zero dispatches and skip finish without
touching the previous publication. Each publication workgroup accumulates the four counters
locally before adding to the published header.

Bindings 0/1/2/4/5/7/11 reuse liquid parameters/positions/velocities/faces/cells/solids/runtime.
Whitewater bindings 12–20 are parameters, cached fields, working pool, working state, working
dispatches, published pool, published state, published draw, and published dispatch, respectively.
Each entry binds only its statically used resources, within the portable storage-binding limit.

The phi gradient uses centered cell-unit differences. Within `abs(phi) <= 1.5`, cache the inward
normal `-gradient / length(gradient)` and interface strength `length(gradient)`; elsewhere they
are zero. Curvature is `-0.5 * divergence(inwardNormal)`. Open volume comes from the nodal SDF,
and volume at most `1e-5` suppresses interface strength. Optional turbulence is
`dx * sqrt(curl² + 2 * strainFrobenius²)`, using centered physical-space derivatives of
face-averaged MAC velocity. Cache sampling is trilinear. MAC interpolation renormalizes over valid
faces and returns zero when none contribute, without setting liquid errors.

Generation requires speed at least `1e-4` and interface strength at least `0.05`.
With outward normal `n`, its upward weight is `smoothstep(0.15, 0.65, n.y)`.
Trapped-air potential ramps `max(0, -dot(v,n))` over `[0.5, 4]`; crest potential ramps
`max(0, curvature) * max(0, dot(v,n))` over the configured curvature interval. Turbulence and
speed use their configured intervals. Multiply the weighted sum of generation rates by speed
potential, upward weight, and consumed dt, then cap and stochastically round. Lifetime interpolates
`tMin`/`tMax` using the largest generation potential. Sample a uniform disk perpendicular to
velocity with radius `rv`, and a uniform offset along `speed * consumedDt`; reject out-of-domain
or disabled-kind candidates before taking a free slot.

Classification enters foam at interface strength `0.05` and upward normal `0.45`, retaining
previous foam down to `0.025`/`0.2`. Optional layer depth checks up to four cells above the marker
with a half-cell smooth falloff. Otherwise nonnegative phi or closed volume means spray, and
negative phi means bubble. Spray uses gravity and `exp(-sprayDrag * dt)`. Bubbles apply
`-kb * gravity * dt` and `kd * (liquidVelocity - velocity)`. Foam moves with liquid velocity and
alone consumes lifetime. Post-advection collision projection maintains a `0.02 * dx` margin,
removing inward relative normal velocity; unrecoverable gradients or domain exits kill the slot.

After publication, copy the four counters, sequence/rejection totals, liquid error word, and
consumed dt into a free 32-byte telemetry slot. Start mappings only after their encoder submission,
never await them in interactive stepping, and discard obsolete sequences or disposed generations.

Required focused cases include generation and update of all three kinds; kind toggles; zero
liquid; pool saturation; active/dense modes; pause/reset/resize/disable; imported collisions;
nonfatal budget-limited dt; error-gated publication; and equivalent liquid trajectories with
whitewater enabled and disabled. Constructor regressions cover post-clear settings ordering,
INDIRECT usage, publication isolation, exact memory totals, and disposal.

## Custom and manual forces

Reference supports the existing `FluidForceField`/internal `ForceFieldSpec` protocol, including
the shared Shift+right-drag ray force. `externalForce(position, velocity, dt)` returns a velocity
delta, already scaled by dt. Each enabled GPU substep applies this delta to the live particle
prefix before particle-to-grid transfer, matching the production FLIP force ordering.
The subsequent snapshot therefore includes the particle impulse, while pressure projection
and FLIP/PIC transfer do not apply that impulse a second time.

`force.ts` generates `applyFlipReferenceForce`, using 128-invocation workgroups and explicit
bindings: force uniform 0, read-only positions 1, writable velocities 2, GPU control 3.
An explicit layout permits valid force functions that do not read their uniform. Count and
dt come from GPU runtime words 0 and 4, never from CPU count telemetry or requested frame dt.
Nonfinite force results latch the existing particle-error bit and disable the substep before
P2G; the last valid publication remains intact.

The adapter builds the force pipeline only when source changes, and rebuilds its bind group
when the force uniform or solver generation changes. Reset retains an attached force but
retargets it to fresh working buffers. Setting the force to null records no force dispatch;
reusing the same field after mouse release reuses its pipeline. Force uniforms remain owned
by the caller, and no additional simulation buffer or readback is allocated.

The host uses the same gesture, temporary sample retention, expiry, pointer-cancel handling,
and mirrored-camera ray/push mapping for Reference and production. Backend reconfiguration
preserves supported force bindings. Paused rendering does not apply impulses; offline stepping
updates the scene-owned force after the scene callback and before recording the requested frame.
Tests cover impulse direction/magnitude under different substep counts, disabled/cleared forces,
buffer replacement, reset, nonfinite output, and manual interaction with the imported camera.

The UI sampler derives moving normal velocity from the existing scene-SDF protocol. It is not the
offline comparison harness's full affine velocity field. Native-style closed-pocket conditioning
is therefore applied before pressure assembly to prevent an incompatible moving-solid flux in sealed
multi-cell liquid pockets. No arbitrary RHS-mean correction or velocity damping is substituted.

The UI enables contact removal and the existing native-style extreme-speed cleanup. The latter
prevents sparse trapped-marker velocity outliers from feeding repeated FLIP transfers until CFL
requests an unrepresentably small timestep. It deletes outliers, never clamps surviving velocities
or loosens pressure accuracy. Reference pressure diagnostics use L2 norms and are not mislabeled
as the production maximum-norm diagnostics.

The cleanup histogram uses `CFL * dx / requestedFrameDt` as its speed interval, using the full
requested frame duration rather than the current adaptive substep or unconsumed remainder.
Its live bin count is Maximum substeps. Allocate 256 bins (the adapter's configuration limit)
once, so changing Maximum substeps requires no allocation; the allocation preview includes
the same storage. With CFL disabled (`0`), disable both adaptive CFL and its speed-based cleanup.
The numerical comparison core retains its explicit opt-in cleanup option.

Cleanup runs after advection and before stable compaction, timestep finalization, or publication.
It reuses the native histogram allowance, high-bin `max` policy, and sparse-maximum/near-maximum
rule already specified in module 58. Maximum speed for the next CFL decision is reduced over
survivors only. GPU runtime word 15 records the cumulative extreme-removal count separately from
contact-removal word 14. Reset restores the seed, both totals, and all pressure history.
An actually invalid/nonprogressing timestep still fails explicitly; it is not reclassified as
ordinary budget exhaustion or allowed to spin indefinitely.

## GPU timing

The interactive Simulation timestamp span encloses recorded physics and publication inside the
render-frame envelope. It is not added to Total a second time. Query collection is nonblocking,
using the normal profiler ring; neither simulation nor the next frame awaits timing. Paused
rendering does not retain stale Simulation time. The original offline oracle retains per-command
timestamp spans, never one interval spanning CPU readback waits.

Query exhaustion marks a sample incomplete instead of presenting partial work as a full GPU time.
Devices without timestamp-query support retain their normal unavailable state. Production profiler
capacity and frame submission behavior remain unchanged.
