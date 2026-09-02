# Module: Fluid Whiteboard Demo

> Package paths: `lab/lite/src/demos/fluid/scenes/whiteboard.ts`, `lab/lite/src/demos/fluid.ts`

## Purpose

Provide an empty fluid-authoring workspace with no demo geometry, authored emitters, sinks, or per-method quality presets. The Whiteboard keeps solver-independent authoring state unchanged when the simulation method changes; only the method-specific physics controls and backend are replaced.

## Demo Contract

`FluidDemo` exposes three optional capabilities:

```ts
interface FluidDemo {
    readonly methodIndependentAuthoring?: boolean;
    readonly usesQualityPresets?: boolean;
    readonly useGridFloor?: boolean;
}
```

Whiteboard sets all three as follows:

```ts
{
    methodIndependentAuthoring: true,
    usesQualityPresets: false,
    useGridFloor: true
}
```

Its base scene SDF is empty space (`sceneSdf > 0` everywhere), its flow graph contains no emitters or sinks, and it creates no meshes. The shared presentation ground is hidden while Whiteboard is active. The solver safety floor follows the editable grid's lower Y bound.

MLS-MPM additionally receives an invisible analytical box container. Its six inner faces sit half a cell inside the solver's mandatory two-cell/upper-three-cell stencil clamp, so the SDF collision response occurs before the hard position clamp. The box tracks Grid position, Grid size, and Physics particle size. Imported Blender collision SDFs are intersected with the same container free space; PBF and PB-MPM retain their existing boundaries.

The shared General controls expose the standard PB-MPM Material selector for Liquid, Elastic, Sand, and Viscoelastic whenever PB-MPM is selected. Every fluid demo therefore uses the same selector and state path. Each Whiteboard material retains its own PB-MPM physics and presentation state while sharing Whiteboard's solver-independent authoring state. A new Sand state uses the Box demo's sand presentation defaults: `#c2b280` and sphere rendering. Later color/render edits remain local to that material.

## Method Switching

The core keeps one method-specific state slot per backend so each backend retains its own specialized physics parameters. It additionally stores one latest shared Whiteboard state. When switching methods:

1. Snapshot the live Whiteboard state.
2. Store the complete snapshot as the latest shared state.
3. Load the target method's prior state, or that method's built-in defaults on first use.
4. Overlay the shared state while retaining target-method-only fields.
5. Copy `gravity` from the shared physics schema into the target schema.
6. Rebuild the target solver and refresh its visible physics controls.

Shared state includes emitters, sinks, initial-fill behavior, particle capacity, Physics particle size, grid position and size, grid-bounds visibility, grid-gizmo visibility, simulation timing, rendering, foam, camera, environment, and MSAA settings. The grid gizmo is already core-global and is never rewritten by pair loading.

Target-method-only fields are:

- physics schema entries other than `gravity`;
- FLIP pressure-projection, FLIP/PIC blend, damping, and timestep controls;
- PB-MPM material;
- MLS-MPM active-block, paged-grid, page-capacity, and fused-discovery controls.

Changing methods necessarily resets simulated particle data because the GPU representation differs between solvers; it does not reset authored settings.

If Whiteboard currently owns an imported self-contained Blender scene, a method change also preserves that scene's GLB entities, animation playback, collision SDF, source-node bindings, camera framing, retained Blender metadata, and selected environment cubemap background. Editing Grid position translates the complete imported bundle: presentation nodes, collision-SDF origin, emitters, and sinks retain their authored alignment. The translation is excluded from animated source-velocity derivation. The imported scene is cleared only when the user changes to another demo or explicitly clears/replaces the import.

## Presets and Quality

Whiteboard never queries or applies quality preset files. Its pair key excludes quality, and the quality selector is hidden while it is active. Export and import still use the normal fluid JSON format.

## Empty Lifecycle

An explicit flow graph with zero emitters activates zero particles at reset. This differs from a `null` flow graph, which preserves the legacy spawn-box behavior. Therefore:

```ts
setFlow({ emitters: [], sinks: [] }); // empty simulation
setFlow(null); // legacy spawn-box seeding
```

## Validation

- Unit coverage verifies explicit empty-flow initialization and method-independent state merging.
- Focused browser coverage creates Whiteboard authoring state, changes methods, and verifies that flow, grid, shared physics, bounds visibility, and gizmo state remain unchanged while the method changes.

## File Manifest

- `docs/lite/architecture/55-fluid-whiteboard.md`
- `lab/lite/src/demos/fluid/scenes/whiteboard.ts`
- `packages/babylon-lite/src/fluid/authoring/method-independent-state.ts`
- `lab/lite/src/demos/fluid/demo.ts`
- `lab/lite/src/demos/fluid.ts`
- `packages/babylon-lite/src/fluid/core/sim-common.ts`
- `tests/lite/unit/fluid/fluid-method-independent-state.test.ts`
- `tests/lite/unit/fluid/fluid-flow-initial-particles.test.ts`
- `tests/lite/parity/regressions/fluid-whiteboard.spec.ts`
