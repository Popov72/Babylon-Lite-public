# Module: FLIP Fluid Simulation

> Package paths: `packages/babylon-lite/src/fluid/flip-sim.ts`, `lab/lite/src/demos/fluid.ts`

## Purpose

Add a GPU FLIP backend to the shared `FluidSim` contract. The backend simulates an incompressible liquid with marker particles and a staggered MAC velocity grid, preserving the existing fluid renderer, emitters, sinks, lifecycle state, scene SDF collisions, Whiteboard method switching, imported scene bundles, force fields, and profiling hooks.

The implementation adopts the mathematical model described by Bridson, Zhu and Bridson, and GridFluidSim3D, but is written independently in TypeScript and WGSL. It does not copy source from those projects.

Primary references:

- Robert Bridson, _Fluid Simulation for Computer Graphics_: <https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf>
- Zhu and Bridson, _Animating Sand as a Fluid_: <https://www.cs.ubc.ca/~rbridson/docs/zhu-siggraph05-sandfluid.pdf>
- GridFluidSim3D: <https://github.com/rlguy/GridFluidSim3D>
- Jiang et al., _The Affine Particle-In-Cell Method_: <https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf>
- Braun et al., _Spatiotemporal FLIP_: <https://doi.org/10.1145/3811289>

## Public API Surface

```ts
export interface FlipOptions extends FluidSimBaseOptions {
    boundsMin?: [number, number, number];
    boundsMax?: [number, number, number];
    gridDim?: [number, number, number];
    groundY?: number;
    dx?: number;
    markersPerCell?: number;
    minSubsteps?: number;
    maxSubsteps?: number;
    cflNumber?: number;
    maxSubDt?: number;
    pressureIterations?: number;
    pressureRelaxation?: number;
    flipRatio?: number;
    velocityDamping?: number;
    kinematicViscosity?: number;
    viscosityIterations?: number;
    surfaceTension?: number;
    restitution?: number;
    initialPositions?: Float32Array;
}

export function createFlipSim(engine: EngineContext, options?: FlipOptions): FluidSim;
```

The returned object implements the existing `FluidSim` interface. FLIP adds no raw GPU handle to the public package API. The solver module remains a tree-shakable opt-in deep import, consistent with the existing fluid backends.

Physics controls:

| Key                   | Range | Default | Meaning                                                               |
| --------------------- | ----: | ------: | --------------------------------------------------------------------- |
| `gravity`             | 0-200 |     9.8 | Downward acceleration in world units per second squared.              |
| `flipRatio`           |   0-1 |    0.95 | FLIP share in the FLIP/PIC velocity blend.                            |
| `kinematicViscosity`  |   0-5 |       0 | Implicit MAC-grid velocity diffusion in world units squared/second.   |
| `surfaceTension`      |   0-5 |       0 | Continuum surface force coefficient; zero skips all interface passes. |
| `minSubsteps`         |  1-16 |       1 | Minimum solver substeps per rendered frame.                           |
| `maxSubsteps`         |  1-32 |       8 | Maximum adaptive solver substeps per rendered frame.                  |
| `cflNumber`           |  0-10 |       2 | Maximum cells travelled per substep; zero disables adaptive CFL.      |
| `restitution`         |   0-1 |       0 | Normal bounce retained after particle collisions.                     |
| `velocityDamping`     |  0-10 |       0 | Non-physical exponential particle drag per second.                    |
| `pressureIterations`  | 1-100 |      40 | Weighted-Jacobi pressure iterations per substep.                      |
| `pressureRelaxation`  | 0.1-1 |     0.8 | Weighted-Jacobi pressure relaxation factor.                           |
| `viscosityIterations` |  1-40 |      12 | Jacobi iterations when physical viscosity is nonzero.                 |
| `maxSubDtMs`          |  1-20 |     8.4 | Absolute maximum time represented by one substep.                     |

The fluid demo estimates marker density as:

```text
active markers / (enabled initial-emitter world volume / dx^3)
```

For an initial-only flow, a red warning appears above 16 markers per authored MAC cell. At that density, additional markers do not increase grid resolution; they primarily increase GPU cost and screen-space Beer-Lambert thickness. The estimate is suppressed for active inflows because their accumulated world volume is not available synchronously.

The FLIP-facing controls use grid resolution and marker sampling density:

```text
cellSize = longestDomainSide / resolutionDivisions
initialMarkers = ceil(initialFluidVolume / cellSize^3 * markersPerCell)
```

`markersPerCell` defaults to `8`, the standard 2 x 2 x 2 sub-cell layout. The generic Physics particle size and Particle count controls are not presented as FLIP quality controls. The former remains only as a serialized compatibility scale derived from cell size. FLIP instead exposes Particle capacity: the number of marker slots preallocated in its GPU buffers.

Each FLIP marker is one simulation particle. Initial active markers are derived from emitter volume, cell size, and marker density. Inflows append markers into the preallocated capacity, while sinks recycle slots. The Physics simulation section shows active markers, allocated capacity, and solver-buffer GPU memory, plus the values expected after restart. Editing an emitter, Particle capacity, Grid size, Grid position, or Resolution divisions updates this preview without interrupting the running simulation. Reset simulation reallocates only when Particle capacity or grid storage changes; emitter-behavior changes reuse the existing particle buffers and bind groups.

Particle capacity is bounded by the active WebGPU device's `maxStorageBufferBindingSize` and `maxBufferSize`, and is allocated up front. If derived initial demand exceeds the selected capacity, or the MAC grid exceeds device limits, reset reduces Resolution divisions to the highest value that fits. If the minimum resolution still cannot fit, reset fails explicitly with the required initial markers and selected capacity.

## Internal Architecture

### Particle State

The shared renderer buffers are also the authoritative FLIP particle arrays:

```text
positionBuffer  vec4<f32>[capacity]  xyz world position, w = 1
velocityBuffer  vec4<f32>[capacity]  xyz world velocity, w unused
debugBuffer     f32[capacity]        world speed
```

Inactive positions are parked at `y = -100000`. Particle activity is authoritative in the shared lifecycle buffer from `sim-common.ts`:

```text
u32 0       atomic activeCount
u32 1       capacity
u32 2..3    reserved
u32 4+i     atomic slot state: free, active, or reserved
```

Reset-time initial emitters use `createFluidInitialParticles()`. FLIP inflows are occupancy sources, matching FLIP Fluids semantics: once per rendered frame, a lightweight pass counts active markers in each preallocated MAC cell, and enabled inflows add markers only where the sampled source cell is below `markersPerCell`. The emitter velocity initializes the new marker velocity but does not control whether liquid is created, so a zero-velocity source still replenishes space vacated by gravity or other motion. `volumeRate`, when present, is a maximum world-volume-per-second replenishment budget; omitting it means fill empty source space without a rate cap.

Without active sinks, accepted inflow markers append atomically to the contiguous active prefix. Finite-rate sources retain a budget-sized candidate dispatch, while uncapped sources dispatch only the marker capacity derived from their transformed source volumes rather than the complete global particle capacity. The previous FLIP substep's cell counts are reused until a sink or warm-up changes particle activity, avoiding a redundant full-particle occupancy pass. Once asynchronous lifecycle readback reports a full particle pool, enabled inflows encode no occupancy or emission passes. Sink/recycle graphs retain the general lifecycle scan. Warm-up retains CPU seed position and velocity arrays so a reserved slot can be restored immediately before it becomes active.

### MAC Grid

For `Nx x Ny x Nz` pressure cells with world cell width `dx`:

```text
u faces: (Nx + 1) x Ny x Nz, at (i, j + 1/2, k + 1/2)
v faces: Nx x (Ny + 1) x Nz, at (i + 1/2, j, k + 1/2)
w faces: Nx x Ny x (Nz + 1), at (i + 1/2, j + 1/2, k)
```

Buffers:

```text
uAccum/vAccum/wAccum  array<FaceAccum>
FaceAccum             atomic<i32> momentum, atomic<i32> weight

uOld/vOld/wOld        f32 per face
uA/vA/wA              vec2<f32> per face: value, valid flag
uB/vB/wB              vec2<f32> per face: extrapolation ping-pong
viscosityRhs           f32 per face: pre-diffusion right-hand side
duA/dvA/dwA           vec2<f32> per face: projected minus pre-force value, valid
duB/dvB/dwB           vec2<f32> per face: extrapolation ping-pong

cellMarks             atomic<u32>[Nx*Ny*Nz]
cellType              u32[Nx*Ny*Nz]: 0 air, 1 fluid, 2 solid
divergence            f32[Nx*Ny*Nz]
pressureA/pressureB   f32[Nx*Ny*Nz], storing pressure impulse q = subDt * pressure
surfaceNormal         vec4<f32>[Nx*Ny*Nz]: interface normal and gradient magnitude
surfaceCurvature      f32[Nx*Ny*Nz]
maxSpeed              atomic<u32>[1], positive-float bits for asynchronous CFL reduction
```

P2G uses integer fixed-point atomics because baseline WebGPU has no portable floating-point atomic addition. Momentum and interpolation weights use the same scale, so normalization divides their decoded values. Particle velocity is CFL-clamped before encoding to prevent integer overflow.

All 3D arrays flatten X-fastest:

```text
index = x + dimX * (y + dimY * z)
```

### Uniform Layout

`FlipParams`, 128 bytes:

```text
vec4<f32> originDx       xyz = boundsMin, w = dx
vec4<f32> dimGround      xyz = grid dimensions as floats, w = groundY
vec4<f32> boundsMin
vec4<f32> boundsMax
vec4<f32> sim            x = subDt, y = gravity, z = flipRatio, w = velocityDamping
vec4<f32> solve          x = pressureRelaxation, y = restitution,
                         z = particleRadius, w = CFL velocity cap
vec4<f32> material       x = kinematicViscosity, y = surfaceTension,
                         z = viscosityIterations, w = CFL number
vec4<u32> counts         x = particle capacity, y = markers per cell
```

## Pipeline Configuration

Every stage is a compute pipeline with `layout: "auto"` and workgroup size 64. Cell- and face-indexed dispatches spill into a second workgroup dimension when the first dimension would exceed WebGPU's 65535-workgroup limit.

Per rendered frame:

1. Poll asynchronous active-count and previous-frame maximum-speed readbacks.
2. Release one warm-up range.
3. Prepare exact emitter and sink budgets.
4. Run delete/recycle and emit passes once.
5. Select the substep count without stalling the GPU:

    ```text
    advectiveSteps = ceil(frameDt * previousMaxSpeed / (cflNumber * dx))
    hardDtSteps = ceil(frameDt / maxSubDt)
    capillaryDt = 0.5 * sqrt(dx^3 / surfaceTension)
    capillarySteps = ceil(frameDt / capillaryDt)
    steps = clamp(max(minSubsteps, advectiveSteps, hardDtSteps, capillarySteps),
                  minSubsteps, maxSubsteps)
    ```

    Advective CFL is omitted when `cflNumber = 0`; the capillary restriction is omitted when surface tension is zero.

6. For each substep:
    1. apply optional particle force field;
    2. clear face accumulators and cell marks;
    3. particle-to-grid transfer and cell marking;
    4. classify cells and solid geometry;
    5. normalize face velocities, save pre-force values, and add gravity;
    6. when enabled, solve implicit MAC-grid viscosity;
    7. when enabled, compute interface normals, curvature, and surface force;
    8. compute fluid-cell divergence;
    9. run warm-started weighted-Jacobi pressure iterations from the previous substep's solution;
    10. project face velocities and create FLIP delta fields;
    11. extrapolate projected velocity and delta into two air-cell layers;
    12. grid-to-particle transfer, gated marker redistribution, RK2 advection, and collision.
7. Run one particle-speed pass that writes debug speed and atomically reduces maximum speed.
8. Encode active-count and maximum-speed readbacks into double-buffered staging buffers.

## Shader Logic

### Particle-to-Grid Transfer

Each active particle scatters each velocity component to the eight surrounding faces of that component's staggered grid using the trilinear hat kernel:

```text
N(r) = max(0, 1 - abs(r))
w(face, particle) = N(dx) * N(dy) * N(dz)
```

For each face:

```text
atomic momentum += encode(w * particleVelocityComponent)
atomic weight   += encode(w)
```

The containing pressure cell receives `atomicAdd(cellMarks[cell], 1)`.

### Cell Classification and Boundaries

A cell is solid when an installed scene SDF evaluates to at most `-0.5 * cellSize` at the cell center. Cells crossed by the exact surface remain cut fluid cells; particle-level SDF collision enforces the exact boundary. This avoids zeroing tangential grid velocities when a wall happens to cross a cell center. Samples outside the pressure-cell domain are treated as solid, so every outer MAC face is a closed numerical boundary.

Otherwise it is fluid when `cellMarks > 0`, and air when it has no marker particle.

The outer grid is a FLIP-specific closed numerical boundary. It is not the MLS-MPM Whiteboard inset SDF. Scene SDFs remain independent and are sampled in addition to the grid boundary.

A face touching a solid cell has zero velocity for the initial implementation. Static and animated SDF geometry therefore prevents penetration. Exact moving-solid normal velocities are a later extension.

### Grid Forces

After normalizing weighted P2G values:

```text
oldFaceVelocity = momentum / weight
uStar = oldFaceVelocity
vStar = oldFaceVelocity - gravity * subDt
wStar = oldFaceVelocity
```

Unknown faces retain `valid = 0`.

### Physical Viscosity

When `kinematicViscosity > 0`, the solver applies backward-Euler diffusion independently to each staggered velocity component:

```text
(I - viscosity * subDt * laplacian) velocityNew = velocityStar
```

Weighted Jacobi uses the saved pre-diffusion velocity as a fixed right-hand side and ping-pongs through the existing face-velocity buffers. Solid faces remain zero and invalid free-surface neighbours use a zero-gradient boundary. The entire RHS copy and solve are skipped when viscosity is zero.

### Surface Tension

When `surfaceTension > 0`, marker count per cell defines a clamped liquid indicator. Three compute passes:

1. calculate the indicator gradient and normalized interface normal;
2. estimate curvature as `-div(normal)`;
3. apply the continuum surface force `surfaceTension * curvature * grad(indicator)` to staggered faces.

Solid-neighbour samples reuse the local value so the indicator is not differentiated through collision geometry. All normal, curvature, and force dispatches are skipped at zero surface tension.

### Divergence

For a fluid cell:

```text
div = (u[x+1] - u[x] + v[y+1] - v[y] + w[z+1] - w[z]) / dx
```

Solid-adjacent faces have already been set to their boundary velocity.

### Pressure Projection

The solver stores pressure impulse `q = subDt * pressure`, making the warm-started field independent of frame/substep duration. Its uniform-density Poisson equation is:

```text
laplacian(q) = divergence
```

For each fluid cell, a weighted-Jacobi iteration computes:

```text
diag = number of non-solid six-neighbours
sum = pressure impulse of fluid neighbours; air pressure impulse is zero
candidate = (sum - divergence * dx * dx) / diag
qNext = mix(qCurrent, candidate, pressureRelaxation)
```

Solid neighbours implement a zero-normal-gradient pressure boundary by not contributing to `diag`. Air neighbours contribute to `diag` with pressure zero, implementing the free-surface Dirichlet boundary.

Pressure ping-pong buffers persist between substeps and rendered frames. Each solve starts from the previous projected state, while Jacobi writes zero into cells that are no longer fluid. This warm start is required for hydrostatic pressure to converge through deep liquid volumes with a bounded iteration count; clearing pressure every substep makes tall authored volumes numerically compress into a shallow marker layer.

The divergence target also includes a bounded marker-density correction for compressed cells. It is disabled beside stationary solids: collision projection naturally concentrates markers in that cut-cell layer, and interpreting that boundary concentration as lost volume would push the neighbouring liquid inward while leaving a detached marker sheet pinned to the wall. The correction remains enabled beside moving boundaries, where it is needed to preserve occupied volume while an obstacle actively compresses and stirs the liquid.

Projection at a face between cells L and R:

```text
uNew = uStar - (qR - qL) / dx
```

Air pressure is zero. Faces touching a solid cell remain at the solid boundary velocity. The FLIP delta stored for each valid face is:

```text
delta = projectedVelocity - oldFaceVelocity
```

### Velocity Extrapolation

Two ping-pong passes extend both projected velocity and FLIP delta into nearby air faces. An invalid non-solid face averages its valid six-neighbours. A valid face copies through unchanged. Solid faces remain invalid and zero.

### Grid-to-Particle Transfer

Each active particle samples the staggered velocity and delta fields with the same trilinear kernels used for P2G. Invalid stencil entries are omitted and the remaining weights are renormalized.

```text
pic = sample(projectedVelocity, position)
flip = oldParticleVelocity + sample(deltaVelocity, position)
newVelocity = mix(pic, flip, flipRatio)
newVelocity *= exp(-velocityDamping * subDt)
```

Velocity magnitude is clamped to `0.9 * dx / subDt` as a final safety fallback. After all substeps, a separate pass writes per-particle debug speed and atomically records the positive floating-point maximum-speed bits. Keeping this reduction out of G2P leaves room for a storage-backed scene SDF while remaining within WebGPU's portable eight-storage-buffer compute limit. The CPU consumes the latest completed double-buffered readback on a later frame, so adaptive CFL never waits synchronously for the GPU.

Particle advection uses midpoint RK2:

```text
mid = position + 0.5 * subDt * sampleVelocity(position)
next = position + subDt * sampleVelocity(mid)
```

After P2G has counted markers per cell, G2P checks only the source cell count for each particle. Cells at or below `1.25 * markersPerCell` pay one atomic load and return immediately. In an overcrowded cell, a bounded fraction of markers is deterministically redistributed among lower-density six-neighbours:

- ordinary redistribution targets only already-occupied liquid cells;
- cells above `2 * markersPerCell` may refill empty horizontal or upward cells;
- target cell centres must remain inside the scene SDF by at least the particle radius;
- moved markers receive deterministic sub-cell jitter;
- no marker is created or deleted.

This prevents stationary-wall and corner clumps from collapsing many markers into a small occupied volume while keeping the normal-case particle cost low.

### Collision

Particles are clamped to the domain by their render radius. The lower Y clamp is `max(boundsMin.y, groundY) + radius`.

For an installed scene SDF, positive distance denotes fluid space. When:

```text
distance < particleRadius
```

the particle is pushed along the normalized SDF gradient by `particleRadius - distance`. If its velocity has an inward normal component, that component is reflected with the configured restitution:

```text
vn = dot(velocity, normal)
if vn < 0:
    velocity -= (1 + restitution) * vn * normal
```

## State Machine and Lifecycle

- `setFlow()` installs solver-independent emitters and sinks without resetting budgets.
- `updateFlowEmitter()` updates animated imported source transforms and velocities in place.
- `reset()` resets flow budgets, lifecycle states, particles, pressure, and debug values.
- Warm-up transitions reserved initial slots to active and uploads their retained seed state.
- Delete sinks atomically transition active slots to free and park them off-screen.
- Emitters atomically transition free slots to active and initialize position and velocity.
- Screen-space surface rendering can use a provisional depth pass plus near-front thickness support. When depth and thickness targets have matching resolutions, a second FLIP-only depth pass rejects unsupported one- or two-marker fronts so the connected liquid behind them becomes visible instead of reconstructing dark sphere disks. The rejection pass is skipped for downscaled thickness because coarse support texels cannot safely classify full-resolution silhouettes and wall-adjacent surfaces.
- Method switching recreates only solver-specific GPU state. Whiteboard's shared authored state continues through `carryMethodIndependentState()`.
- FLIP has no PB-MPM material state and no MLS-MPM sparse-grid state.
- Whitewater is opt-in through the shared `setFoam()` contract. The diffuse pool, uniforms, pipelines, active-list buffers, generation passes, and update passes are created only after foam is enabled. Reset and disable clear the pool.

### Whitewater

FLIP reuses the solver-independent 32-byte diffuse-particle pool and foam renderer:

```text
p = position.xyz, lifetime
v = velocity.xyz, kind
kind 0 = spray, 1 = foam, 2 = bubble
```

Generation runs once per rendered frame after the final G2P, not once per adaptive substep. The final marker occupancy produces an interface normal and curvature even when physical surface tension is disabled. Each marker trilinearly samples both fields and receives a smooth upward-facing emission weight; this avoids binary cell-height terraces while suppressing vertical/front-face emission. Each active surface marker combines:

- inward normal motion and under-filled surface occupancy for trapped-air potential;
- positive curvature and outward normal motion for wave-crest potential;
- marker kinetic energy as the common emission gate.

Expected generation is multiplied by frame duration and stochastically rounded, so changing adaptive substeps does not change the authored rate. New diffuse particles are sampled in a short velocity-aligned cylinder.

The interface field is nonzero only in fluid cells with a six-connected air neighbour. Marker-density gradients inside the liquid or beside solid/domain walls are not treated as free surface. Diffuse particles trilinearly sample this cell-centred field to avoid cell-aligned type bands. Stable foam uses continuous upward-normal enter/retain thresholds, with hysteresis as marker occupancy changes; it does not depend on a binary “air cell directly above” test. The update pass samples the finalized staggered MAC velocity and classifies particles from local marker occupancy and that smoothed free-surface field:

- spray is ballistic under gravity in low-occupancy cells;
- foam follows the MAC velocity only on upward-facing top-surface regions and consumes lifetime;
- every occupied non-interface particle is a bubble, following the liquid with drag plus upward buoyancy.

Foam retention is evaluated before the low-occupancy spray transition, so a surface-attached particle does not alternate between foam and spray as it crosses a cell boundary. The renderer likewise fades spray in only after its eye-space depth is clearly detached from the reconstructed liquid surface.

Dense mode scans the fixed pool. Optional active-particle mode maintains two compact survivor lists, indirect compute dispatch, and indirect draw arguments shared with the existing foam renderer. The FLIP emitter binds exactly eight storage buffers, preserving the portable WebGPU per-stage limit.

The screen-space renderer samples the liquid depth at each foam particle centre, estimates the local depth tangent plane, and softly attenuates billboard fragments by their deviation from that plane. A reconstructed surface-normal mask retains upward-facing foam while fading steep/front-facing liquid, and diffuse gain fades during the final 0.3 seconds of lifetime. Unlike spray, surface foam cannot project an entire constant-depth billboard over a farther front-facing liquid surface, large splats do not form hard moving intersection contours, and short-lived particles do not disappear at full opacity.

## FLIP Concept Map

| FLIP concept               | Babylon Lite FLIP                                                         |
| -------------------------- | ------------------------------------------------------------------------- |
| Marker particles           | `positionBuffer`, `velocityBuffer`, lifecycle states                      |
| MAC velocity grid          | staggered U/V/W face buffers                                              |
| PIC/FLIP velocity transfer | `flipRatio` blend                                                         |
| Pressure solver            | fixed-iteration weighted Jacobi                                           |
| Solid boundary mesh/SDF    | existing `SceneSdfSpec`                                                   |
| Domain                     | editable Grid position and size                                           |
| Resolution                 | longest-side `gridResolution`, deriving `dx`                              |
| Particles per cell         | `markersPerCell`, default 8                                               |
| CFL/adaptive timestep      | previous-frame GPU max speed, min/max substeps, hard and capillary limits |
| Physical viscosity         | optional implicit Jacobi diffusion on MAC faces                           |
| Surface tension            | optional marker-indicator CSF normal/curvature/force passes               |
| Inflow/outflow             | shared flow emitter and delete-sink passes                                |
| Initial fluid objects      | deterministic shared initial-emitter lattice                              |
| Particle capacity          | user-selected fixed GPU allocation; active prefix is derived/dynamic      |
| Particle reseeding         | bounded in-place redistribution from overcrowded cells                    |
| Whitewater                 | opt-in trapped-air/wave-crest diffuse spray, foam, and bubbles            |

The first implementation intentionally uses weighted Jacobi instead of a production PCG solve. It is slower to converge but maps cleanly to fixed WebGPU passes without global reductions or CPU synchronization. PCG or multigrid is the primary performance/quality extension after the baseline is validated.

## Dependencies

- `engine/engine.ts`
- `fluid/sim-common.ts`
- `fluid/controls-panel.ts`
- existing particle and surface renderers
- Whiteboard method-independent state and preset infrastructure

No module-level cache or registration side effect is added.

## Test Specification

### Unit

- FLIP uses its own `0.25` world-unit base cell size and MPM scale limits.
- FLIP physics keys pass strict self-contained fluid JSON validation; unknown FLIP keys fail.
- Method-independent Whiteboard merging preserves FLIP-specific physics while carrying gravity and authored state.
- Format-11 export/import round-trips `meta.method = "FLIP"` and all FLIP physics values.
- FLIP exports use the current velocity-damping, physical-material, and adaptive-timestep fields directly.

### Browser

- The Whiteboard method selector exposes FLIP.
- Switching PBF -> FLIP preserves emitter, sink, grid position, grid size, particle size, gravity, bounds visibility, gizmo visibility, camera, environment, and imported bundle state.
- FLIP displays its own controls and hides PBF/MLS-MPM/PB-MPM-only controls.
- An enabled initial emitter activates particles; an explicit empty graph remains at zero active particles.
- Delete sinks reduce active count and an enabled inflow refills free slots.
- Different authored initial-emitter heights retain measurably different settled particle heights at equal marker density.
- FLIP marker-density estimation reports particles per authored MAC cell and warns above the high-density threshold.
- Format-11 export/import round-trips FLIP resolution divisions, markers per cell, adaptive timestep controls, viscosity, and surface tension.
- A WebGPU regression dispatches nonzero viscosity and surface-tension passes and verifies finite particle velocity output.
- A WebGPU regression verifies that FLIP foam is absent while disabled, generates diffuse particles under energetic surface motion, validates portable bindings, and clears on reset/disable.
- The 651,086-marker box preset retains liquid height and occupied cells for 20 simulated seconds without growing corner columns.

### Focused Validation

- Run only FLIP-related unit tests and the Whiteboard/browser FLIP regression.
- Run package, lab, and test typechecks plus ESLint/Prettier.
- Run only the filtered fluid demo bundle and its bundle-size check.
- Do not update golden images, MAD thresholds, or bundle-size ceilings.

## Quality Extensions

1. PCG with a Jacobi or MIC(0) preconditioner.
2. Deterministic cell-sorted gather P2G to remove fixed-point atomics.
3. Particle-derived liquid level set and ghost-fluid free-surface pressure.
4. Moving-solid normal velocity.
5. APIC affine transfers.
6. Multigrid pressure projection.
7. ST-FLIP large-timestep reconstruction.
8. Obstacle-driven dust generation as a separate diffuse-particle type.

## File Manifest

- `docs/lite/architecture/56-fluid-flip.md`
- `packages/babylon-lite/src/fluid/flip-sim.ts`
- `packages/babylon-lite/src/fluid/controls-panel.ts`
- `lab/lite/src/demos/fluid.ts`
- `lab/lite/src/demos/fluid/grid-settings.ts`
- `lab/lite/src/demos/fluid/preset-io.ts`
- `lab/lite/src/demos/fluid/blender-fluid-json.ts`
- `lab/lite/src/demos/fluid/quality-presets.ts`
- `tests/lite/unit/fluid-grid-settings.test.ts`
- `tests/lite/unit/fluid-blender-json.test.ts`
- `tests/lite/unit/fluid-method-independent-state.test.ts`
- `tests/lite/parity/regressions/fluid-whiteboard.spec.ts`
