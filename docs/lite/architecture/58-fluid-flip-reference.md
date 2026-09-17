# FLIP Reference experiment

## Purpose

Isolate liquid-solver experiments from production FLIP. The experiment compares particle state
against baked FLIP Fluids data with foam and water shading excluded. The reference algorithm is
implemented independently from mathematical descriptions; upstream code is read for understanding,
not copied.

## Numerical contract

The first implementation uses a dense staggered MAC grid and explicit XYZ particle positions and
velocities. Grid origin, dimensions, spacing, seed count, solid boundary geometry, gravity, PIC
fraction, and physical timestep must be declared. No UI defaults or automatic resolution changes
are inherited.

Each physical substep uses particle-to-grid transfer, a particle-derived liquid level set, gravity,
a cut-cell free-surface pressure system, velocity projection/extrapolation, FLIP/PIC velocity
transfer, and particle advection with solid collision handling. The pressure solve reports its
residual and iteration count; a real-time catch-up budget must never silently drop experimental
simulation time.

Solid distances are supplied on grid vertices in world units, negative inside obstacles. Solid
velocity is supplied at those same vertices as XYZ world units per physical second. Domain walls
are closed. Coordinate conversion is applied once before inputs reach the solver.

## Comparison contract

Blender timeline time, physical simulation time, and video playback time are distinct. In the
beach fixture, one timeline frame is 1/25 second but advances physics by 1/50 second. Therefore the
moving obstacle must advance one complete timeline frame during each 1/50-second physical step.
Changing only the video encoding rate or physics timestep does not correct obstacle motion.

The harness reads exact first-frame marker positions/velocities, verifies cache frame phase, and
compares both solvers at matching physical times and obstacle poses. Metrics include marker count,
center of mass, kinetic energy, height/run-up, and normalized spatial occupancy. Particle IDs are
not assumed to align across frame files. A shared sphere debug view avoids interpreting shader
differences as fluid-volume differences.

## Scope

The experimental backend does not replace the production FLIP method or add demo controls. Foam,
surface reconstruction, inflows, reseeding, paging, and optical rendering are outside the first
milestone. Unsupported features must be explicit rather than silently approximated.

## Experimental API

These exports live in `fluid/experimental/flip-reference/solver.ts`; they are not a replacement
for the root package's production `createFluidSimulation` method or its controls.

```typescript
export declare function createFlipReferenceSimulation(engine: EngineContext, options: FlipReferenceOptions): FlipReferenceSimulation;
export declare function updateFlipReferenceSolids(sim: FlipReferenceSimulation, distances: Float32Array, velocities: Float32Array): void;
export declare function stepFlipReferenceSimulation(sim: FlipReferenceSimulation, dtSeconds: number): Promise<FlipReferenceDiagnostics>;
export declare function readFlipReferenceParticles(sim: FlipReferenceSimulation): Promise<{ positions: Float32Array; velocities: Float32Array }>;
export declare function disposeFlipReferenceSimulation(sim: FlipReferenceSimulation): void;
```

`FlipReferenceOptions` requires `gridOrigin`, `gridDimensions`, `cellSize`, `initialPositions`,
and `initialVelocities`. Positions and velocities are matching XYZ arrays, including the valid
empty state. Inputs are checked for finite representable float32 values, half-open grid containment,
resource sizes, and device dispatch limits. The simulation is plain state; behavior is standalone.

| Optional setting                                 | Baseline default          | Reference preset / comparison use                              |
| ------------------------------------------------ | ------------------------- | -------------------------------------------------------------- |
| `referenceNumerics`                              | `false`                   | Harness passes `true`                                          |
| `gravity`, `picFraction`                         | `[0,-9.81,0]`, `0.05`     | Explicit fixture values                                        |
| `kernel`, `transferRadius`                       | `"trilinear"`, `h`        | `"wyvill"`, `sqrt(3)h/2`                                       |
| `liquidRadius`, `particleRadius`                 | `sqrt(3)h/2`, `0.1h`      | Liquid radius unchanged; debug size is independent             |
| `advection`, `extrapolationLayers`               | `"rk2"`, `8`              | `"rk3"`, `12`                                                  |
| `ghostFluidThetaMin`                             | `0.01`                    | `1/26`                                                         |
| `geometryFractions`                              | `"center"`                | `"reference"`                                                  |
| `constrainSnapshot`                              | `false`                   | `true`                                                         |
| `solidVolumeCorrection`                          | `true`                    | `true`; explicit `false` is a legacy ablation                  |
| `collisionMode`, `collisionRadius`               | `"project"`, `0.01h`      | `"sweep"`, `0.2h`                                              |
| `domainInset`, `domainCollisionRadius`           | `0`, collision radius     | `1.5h+0.00005`, `0.1h`                                         |
| `maxAdvectionSubsteps`                           | `256`                     | Exhaustion fails explicitly; no time is discarded              |
| `pressureTolerance`, `pressureAbsoluteTolerance` | `1e-6`, `1e-8`            | Harness explicitly requests `1e-5`, `1e-8`                     |
| `maxPressureIterations`                          | `400`                     | Bounded float32 Jacobi-PCG, not native MIC(0)                  |
| `solidsIncludeDomain`                            | `false`                   | `true` when the supplied nodal union includes the native shell |
| `solidDistances`, `solidVelocities`              | No obstacles / stationary | Nodal float32 fields; may be updated before stepping           |
| `removeInsideSolids`                             | `false`                   | Harness passes `true`                                          |
| `extremeVelocityRemoval`                         | Disabled                  | Fixture opt-in `{frameDtSeconds, cfl, maxFrameSubsteps}`       |

The `"radial"` transfer option retains the initial compact poly6 experiment. Explicit options
override preset defaults, allowing individual numerical stages to be compared without changing
production FLIP.

`sim.count` is the read-only live prefix length, while `sim.capacity` and GPU buffer identities
remain fixed. Zero survivors are valid. Creation uploads initial state; recreating the simulation
is the reset operation. Concurrent steps/readbacks or updates during an active operation fail.
Shader/allocation errors surface on the first async operation. A failed step poisons further use;
disposal remains idempotent and releases all owned resources.

## Storage and execution

The 176-byte uniform contains eleven vec4 slots: grid dimensions/live count; origin/cell width;
gravity/dt; PIC/theta/liquid radius/collision radius; cell/face/U/V counts; kernel/advection/sweep/
reduction settings; squared pressure tolerances; transfer radius/domain clearances/weight threshold;
numerical mode bits; capacity/scan groups/histogram/removal bits; and histogram speed interval.

Positions and velocities are vec4 float32 arrays; speed is one float32 per capacity slot. MAC faces
are flattened U, then V, then W, with dimensions `(nx+1,ny,nz)`, `(nx,ny+1,nz)`, `(nx,ny,nz+1)`.
Each face occupies 32 bytes: velocity, old velocity, open area, solid component, validity and padding.
Two face arrays provide synchronous six-neighbor extrapolation ping-pong storage.

Each 64-byte cell stores liquid phi, diagonal, RHS, pressure-cell flag; three positive coefficients
and pressure-gauge flag; three negative coefficients and air-connection flag; and open volume.
Particle cell lists use atomic heads/links with `0xffffffff` as the sentinel. Nodal solids are
`vec4(distance, velocityXYZ)`. Pressure storage contains per-cell impulse/residual/direction/matvec,
128-thread reduction partials, and controls. CPU row/component scratch and one mapping buffer
are allocated with the solver; pressure connected components fix Neumann gauges before PCG.

Per substep:

1. Build particle cell lists, sphere-union phi, and face-centric weighted particle velocity.
2. Extrapolate, save the old velocity field, apply gravity, and assemble the pressure system.
3. Condition native closed-pocket solid velocities, rebuild affected fluxes, fix pressure gauges, and solve.
4. Rebuild projected-face validity, measure finite-volume divergence, extrapolate, and constrain snapshots.
5. Apply FLIP/PIC transfer, advect with the projected grid, and resolve swept contacts.
6. Optionally select removals and stably compact survivors; publish live count and diagnostics.

Pressure iterations run in batches of at most eight with GPU reductions. Recursive convergence is
followed by a recomputed true residual; residual replacement can restart CG within the same total
iteration budget. Diagnostics report actual convergence, iterations, L2 residual in velocity units,
finite-volume divergence, collisions/fallbacks, live/capacity counts, and removal statistics.

## Numerical stage definitions

With `R = sqrt(3)h/2` and `q = distance(face,marker)/R`, reference P2G uses
`w = (1-q*q)^2 * (1-4*q*q/9)` for `q<1`, otherwise zero. A face is valid when total weight exceeds
`1e-6`. Liquid phi is the local union of radius-R marker spheres, bounded by `+3h`. Near solid
interiors, phi below `h/2` becomes `-h/2`; magnitudes below `0.005h` retain sign with a nonzero floor.

Open face fractions use linear edge crossings, one-/three-corner triangles, adjacent-corner
trapezoids, and the mean-sign rule for diagonal ambiguity. Open cell volume averages the two
five-tetrahedron cube partitions, weighting each central tetrahedron twice each corner tetrahedron.
There is no positive aperture floor.

Pressure is stored as impulse `q = dt*p/(rho*h)`, with density one. For outward face sign `s`,
open area `a`, open volume `c`, solid velocity `us`, and forced fluid velocity `u`, the RHS is
`sum(s * ((a-c)*us - a*u))`. Fluid-fluid off-diagonals are `-a`; a free-surface diagonal contributes
`a/theta`, where `theta = max(phiLiquid/(phiLiquid-phiAir), 1/26)`. The velocity correction uses
the same theta convention and does not multiply its pressure gradient by face area again.

Projection validity is rebuilt from open faces bordering liquid. Twelve synchronous layers average
known six-connected neighbor values. Fully blocked faces receive solid velocity after extrapolation,
on both the new field and the enabled old FLIP snapshot. The particle velocity is
`PIC_fraction*I(new) + (1-PIC_fraction)*(oldParticle + I(new)-I(oldGrid))`.

RK3 samples the projected grid at `x`, `x+dt*k1/2`, and `x+3dt*k2/4`, then advances with weights
`2/9, 1/3, 4/9`. Swept contact samples are at most `0.1h` apart. Negative obstacle distance triggers
projection toward `phi=0.2h`; invalid projection falls back to the last safe sample. Contact correction
does not rescale the stored FLIP velocity. Remaining solid-interior markers are explicitly removed
only when the removal stage is enabled; otherwise unresolved contacts fail the step.

## Native-style particle removal

The harness always enables contact removal and restores the exported native extreme-removal flag.
`--no-extreme-removal` isolates contact-only behavior. Both modes keep pre-removal maximum speed
in diagnostics, so pruning cannot conceal a numerical acceleration spike.

For full physical frame duration `T`, CFL `C`, and native maximum substeps `M`, the histogram interval
is `S=C*h/T`. Marker speed bins are `min(floor(speed/S), M-1)`, using all markers before pruning.
Starting at threshold `M*S`, inspect descending bins with a population allowance
`min(floor(0.0005*N),35)`. Each accepted bin `i` sets the threshold to `max(i+4,M)*S` (the maximum is
intentional). If both the `[0.9*vmax,0.99999*vmax)` and top `>=0.99999*vmax` groups contain at most
six markers, reduce the threshold to `min(threshold,0.99999*vmax)`. Removal compares squared speed
strictly against squared threshold, without changing survivor velocity.

An optional particle-state buffer contains reasons, per-marker exclusive prefixes, workgroup
prefixes, eight controls, and the histogram. A 128-thread scan plus group scan preserves survivor
order. Survivors are copied through two-vec4 scratch records, committed back to the original buffers,
and inactive tails are cleared. CPU count publication occurs only after compaction completes.
Per-step and cumulative inside/extreme removal counts are reported separately.

## Executable comparison harness

Run `pnpm exec tsx scripts\compare-fluid-flip.ts --help` for the current arguments. A comparison
requires an exact-state bundle, its corresponding native cache, and a new output directory:

```powershell
pnpm exec tsx scripts\compare-fluid-flip.ts --input <exact.json> --cache <cache-directory> --output <new-directory> --frames 250 --sample-every 1 --video
```

`--backend production` runs the unchanged production algorithm with the same exact seed,
voxel-snapped grid, externally driven physical substeps, and sampled moving solids. This aligned
baseline disables reseeding and whitewater; it is not a replay of the old demo's clock mismatch.
`--prepared <case.json>` reuses prepared geometry/cache data for an algorithm A/B run.
`--analytic` uses a small closed-domain dam break without claiming a native reference.

The runner builds only its own unminified Vite viewer, uses a loopback-only temporary server and
Chrome/WebGPU, and closes both on completion. It leaves a `results.json`, packed particle
snapshots, and optional JPG/MP4 debug views. Incomplete runs retain partial results and a failure
message. Existing result directories cannot be overwritten.

The harness defaults to a `1e-5` relative L2 pressure target for its float32 pressure system.
The r80 fixture can hit a true-residual roundoff floor above `1e-6`; requesting an unattainable
tighter target produces an explicit failed run, not a silent tolerance increase. Requested solver
settings and actual residuals are stored in `results.json`. Native f64 MIC(0) infinity-norm
convergence is a different numerical contract and must not be equated with this value.

### Input preparation and coordinate convention

`prepare.ts` uses the existing fluid bundle parser and glTF accessor/keyframe evaluation helpers.
It checks the first FFP3 positions and velocities against the exported BLFI values exactly.
Both FFP3 streams must have equal category counts, ID-selection tables, and payload lengths.
The cache's `.bbox` is authoritative for spacing and voxel-snapped extents, rather than the
authored object's unsnapped bounding box.

The normalized coordinate basis is Blender `(x,y,z) -> (x,z,-y)`. Because the general glTF loader
adds an X-reflected right-handed-to-left-handed root, preparation removes that root before using
its world matrices with the already-normalized particle/SDF data. The same normalized geometry
is used for both debug views. Camera mirroring is presentation-only and never changes winding
or simulation coordinates.

Native visible particle caches are written after the first physical substep of a frame, after
initial-fluid insertion. Therefore the first cache frame contains unadvanced seeds; continuing
from it executes that frame's remaining substeps before the next frame's first substep. With two
substeps, continuation from frame 150 samples obstacle timeline times 0.02 then 0.04 seconds
before comparison with cache frame 151. Autosaves instead represent the end of all substeps and
cannot be substituted for the visible cache without a phase change.

For timeline frame `F` and substep-start fraction `a`, the obstacle transform is the affine
vertex interpolation `M = (1-a) M_F + a M_(F+1)`, not quaternion interpolation at fractional
frames. Its velocity transform is
`D = ((1-a) (M_F-M_(F-1)) + a (M_(F+1)-M_F)) / physicalFrameSeconds`.
Missing neighbor frames use the current frame, giving zero initial backward displacement.
World velocity at a current world point `p` is `D * inverse(M) * p`. The signed-distance union
chooses the corresponding obstacle velocity; static obstacles have zero velocity. This avoids
differentiating quantized SDF values to discover an obstacle's speed in the reference backend.
Nodal interpolation remains distinct from native closest-triangle barycentric face velocities.

Native fixtures include the solid domain shell inset `1.5h + 0.00005` on each side in the supplied
nodal solid distances. Closed faces at the outer allocation boundary are not a replacement for
this physical wall inset. The native particle collision AABB has an additional `0.1h` inset.

### Measurements

Snapshots contain an eight-byte header (little-endian u32 marker count, u32 reserved), followed
by tightly packed XYZ f32 positions and XYZ f32 velocities. Neither IDs nor pressure/grid history
are inferred from this format.
An empty state has an eight-byte snapshot and null centroid/percentile/mean-energy values; it is
not replaced with artificial zero-position particles. Two empty occupancy sets compare identically;
one empty set against a nonempty set reports disjoint occupancy.

Metrics include marker count, mean position, min/max and 1st/99th coordinate percentiles, mean
speed, kinetic energy per unit marker mass, occupied cell count, escaped marker count, and 99th
percentile height per Z slab. Normalized occupancy difference is
`0.5 * sum_cells(abs(actualCount / actualTotal - nativeCount / nativeTotal))`; zero is identical
spatial mass distribution and one is disjoint. Occupancy IoU uses nonempty cells and ignores marker
ordering. Marker count is reported separately rather than treated as proof of equal liquid volume.

`auditReferenceGeometry(caseFile, frameIndex)` checks native markers against the exported signed
distance fields before attributing discrepancies to the solver. The exported static/local fields
remain an approximation to native mesh level sets: for the r80 beach at frame 250, the audit found
218 of 29,156 native markers inside the exported solids, including 30 deeper than one quarter cell.
Consequently an improved pressure residual alone cannot establish native-equivalent boundaries.

## Remaining reference differences

The pressure solver is float32 Jacobi-PCG with L2 stopping, not native double-precision MIC(0)
with infinity-norm stopping. After native-style solid-velocity pocket conditioning, closed-volume
handling fixes gauges and rejects any remaining incompatible flux. The extrapolation border treatment
does not yet reproduce native outer-border `DONE` semantics.

Solid velocities are interpolated from the supplied nodal union, not assembled from per-object
solid-area-weighted closest-triangle face velocities. Exported SDF discretization, those boundary
differences, and float32 arithmetic can change trajectories and which sparse extrema are removed.
Matching particle counts is therefore neither required for a valid comparison nor proof of equivalence.
Native per-cell overcrowding removal is not implemented in this initial-only experiment.

The numerical specification was derived from official FLIP Fluids
[v1.8.8 / commit 00534b1](https://github.com/rlguy/Blender-FLIP-Fluids/tree/00534b1d1c098e5bb664c1a3cad9f4dab8e4c927),
especially `fluidsimulation.cpp`, `velocityadvector.cpp`, `particlelevelset.cpp`,
`pressuresolver.cpp`, `levelsetutils.cpp`, and the add-on cache writer. The GPU implementation is
independent; the source model and numerical adaptations should remain explicit in future comparisons.

Closed-pocket conditioning uses a separate connectivity threshold of `1e-6` open area. Liquid
components of more than one cell without an air connection at that threshold zero every surrounding
solid-face velocity before RHS assembly and final snapshot constraints. Matrix coefficients still
retain all strictly positive openings. Single-cell components are not conditioned. Diagnostics count
conditioned components/cells separately from Neumann pressure gauges. This is especially important
for the UI's distance-derived moving-boundary velocity field; subtracting an arbitrary RHS mean would
not reproduce the native rule.

## Initial beach comparison

The completed 250-frame runs use identical cached seeds, physical clocks, and prepared boundaries.
At the final frame, with source-enabled native-style pruning:

| Resolution | Production occupancy difference | Reference occupancy difference | Production centroid error | Reference centroid error | Reference/native live markers |
| ---------- | ------------------------------- | ------------------------------ | ------------------------- | ------------------------ | ----------------------------- |
| 40         | 0.4434                          | 0.2030                         | 0.9676                    | 0.1918                   | 2235 / 2220                   |
| 80         | 0.3289                          | 0.2162                         | 0.3098                    | 0.1124                   | 29055 / 29090                 |

Centroid errors are in world units. These are measured trajectories, not acceptance ceilings or
proof of equivalence. The r80 final mean kinetic energy was 0.07546 versus native 0.07748 and
production 0.16650. Local crest coverage and backwash still differ; a better global metric does not
mean every rock is covered at exactly the native time. Sparse-extreme selection can differ across
float32 trajectories, so exact survivor counts are not a reproducibility requirement.

## File manifest

- `packages\babylon-lite\src\fluid\experimental\flip-reference\types.ts`: pure-state experimental contracts.
- `packages\babylon-lite\src\fluid\experimental\flip-reference\solver.ts`: allocation, submission, pressure control, readback, disposal.
- `packages\babylon-lite\src\fluid\experimental\flip-reference\shaders.ts`: MAC stages, pressure kernels, collision and compaction.
- `packages\babylon-lite\src\fluid\experimental\flip-reference\closed-pockets.ts`: thresholded native pocket connectivity.
- `packages\babylon-lite\src\fluid\experimental\flip-reference\allocation.ts`: shared buffer sizes for allocation and projection.
- `scripts\compare-fluid-flip.ts`: isolated preparation/build/capture/encoding command.
- `scripts\fluid-reference\prepare.ts`: FFP3 decoding, exact-seed validation, native grid and glTF poses.
- `scripts\fluid-reference\animation.ts`: affine obstacle poses and physical-frame velocity sampling.
- `scripts\fluid-reference\types.ts`: portable case/snapshot/metric data.
- `scripts\fluid-reference\solids.ts`: sampled nodal solid geometry and physical boundary velocity.
- `scripts\fluid-reference\production.ts`: explicitly configured unchanged production baseline.
- `scripts\fluid-reference\metrics.ts`: order-independent quantitative state comparisons.
- `scripts\fluid-reference\audit.ts`: native-marker versus exported-solid geometry audit.
- `scripts\fluid-reference\browser.ts`: synchronized browser-side execution and readback.
- `scripts\fluid-reference\viewer.ts`: shared existing particle-sphere renderer and neutral obstacles.
- `tests\lite\unit\fluid\fluid-reference-comparison.test.ts`: snapshot, metrics, SDF, and clock contracts.
