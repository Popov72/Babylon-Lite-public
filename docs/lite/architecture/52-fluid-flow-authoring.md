# Module: Fluid Flow Authoring

> Package path: `packages/babylon-lite/src/fluid/`

## Purpose

Define solver-independent fluid sources and recycling sinks that can be authored in the lab, serialized to JSON, exported from Blender, and replayed by PBF, MLS-MPM, and PB-MPM without demo-specific emitter code.

## Public API Surface

The package root exports these pure-state types:

```ts
type FluidVec3 = [number, number, number];

interface FluidTransform {
    position: FluidVec3;
    rotation: [number, number, number, number];
    scale: FluidVec3;
}

type FluidShape =
    | { type: "box"; size: FluidVec3 }
    | { type: "sphere"; radius: number }
    | { type: "cylinder"; radius: number; height: number; innerRadius?: number }
    | { type: "cone"; bottomRadius: number; topRadius: number; height: number }
    | { type: "capsule"; radius: number; height: number }
    | { type: "polygonPrism"; points: [number, number][]; thickness: number };

interface FluidEmitter {
    id: string;
    name: string;
    enabled: boolean;
    behavior: "initial" | "inflow";
    transform: FluidTransform;
    shape: FluidShape;
    sampling: "volume" | "surface";
    velocity: FluidVec3;
    velocitySpace: "local" | "world";
    spread: number;
    volumeRate?: number;
}

interface FluidSink {
    id: string;
    name: string;
    enabled: boolean;
    transform: FluidTransform;
    shape: FluidShape;
    targets: string[];
    volumeRate?: number;
}

interface FluidFlowConfig {
    emitters: FluidEmitter[];
    sinks: FluidSink[];
}
```

`FluidSim.setFlow(config)` installs the graph. Call `reset()` after changing initial emitters so their positions and velocities are reseeded.

## Shape Conventions

- Box dimensions are full local extents.
- Sphere radius is local before transform scale.
- Cylinder, annulus, cone, and capsule use local Y as their axis.
- Capsule `height` is total end-to-end height.
- Polygon prisms use local XZ points and local Y thickness.
- Rotation is a normalized XYZW quaternion.
- Negative transform scale mirrors sampling; volume uses the absolute scale determinant.

## Runtime Architecture

When a graph contains initial emitters but no enabled inflow, reset activates the complete selected global particle pool and distributes it among initial emitters in proportion to transformed shape volume. This preserves the established meaning of the particle-count control: initial-only simulations receive the selected number of particles.

When enabled inflows are present, initial emitters instead consume only the slots required by their transformed world volume divided by the solver particle volume, capped by the pool capacity. The remaining slots stay dormant for subsequent inflow activation. Initial emitters never own separate particle pools. Explicit solver `initialPositions` remain higher priority and activate the complete supplied pool. If an authored graph has enabled inflows but no enabled initial emitter, reset starts with zero active particles; the inflows activate dormant slots during subsequent frames. The legacy spawn box is used only when no enabled emitter exists.

Every solver stores live particles in a contiguous prefix and reserves the remaining pool slots as dormant capacity. An inflow activates the next dormant slots, samples their position and velocity from its authored shape, and never grows the fixed GPU allocation. Finite inflows use `volumeRate` as exact world volume per second; omission means unlimited and consumes all remaining dormant capacity in the next simulation frame. Multiple finite inflows consume their frame budgets before unlimited inflows divide the remainder. Capacity exhaustion stops direct activation without reallocating or replacing active particles.

The lab labels the dropdown **Particle capacity** and reports **Active particles** directly below it. The values are identical for initial-only graphs; inflow graphs may begin below capacity and rise as emitters activate dormant slots.

Inflow emitters are also launch destinations. A sink captures active particles inside its shape and relaunches them through one of its enabled inflow targets. Sink `volumeRate` is the maximum captured world volume per second; omission means unlimited recycling. The target inflow's rate budget applies to both newly activated dormant slots and sink recycling. Direct activation consumes that emitter's budget first while dormant capacity remains; after the pool is full, the same budget is available to recycling. The runtime converts volume rates to exact integer particle budgets using spherical particle volume and carries fractional budgets between frames. Unmet whole-particle budget is discarded rather than accumulated into a later burst.

Start-up warm-up applies only to reset-time initial particles. Rate-limited inflows begin after that initial prefix has finished activating, which preserves the contiguous live-prefix invariant.

The shared runtime supports at most 16 emitters, 16 sinks, 128 packed polygon triangles, and 256 packed polygon points. Object IDs must be non-empty and unique. Invalid capacity is reported as an exception rather than silently truncating the graph.

## Simulation Lifetime

The fluid demo exposes **Simulation duration** and **Alpha decay** in the General section. Duration is measured in advancing simulation seconds; zero keeps the simulation alive indefinitely. Manual pause time does not consume the duration.

After a positive duration elapses, the solvers continue advancing while a global opacity falls linearly from one to zero over `alphaDecay` seconds. The opacity applies consistently to the screen-space fluid surface, sphere rendering, and diffuse foam particles. At zero opacity, solver updates and all fluid-particle render passes stop; the scene, camera, and non-fluid meshes continue rendering. Reset, method changes, demo changes, and particle-buffer rebuilds restart the lifetime at full opacity. Setting duration back to zero also resumes an expired simulation.

Complete presets store optional top-level `simulationDuration` and `alphaDecay` values. Older presets default to `0` and `2` seconds respectively.

## GPU Data Layout

One 12,320-byte uniform buffer contains:

1. An 8-float header.
2. Sixteen 128-byte emitter records.
3. Sixteen 128-byte sink records.
4. Packed polygon triangles.
5. Packed polygon points and edge CDFs.

A 192-byte atomic storage buffer contains two words per emitter (`used`, `budget`) followed by one counter per sink. CPU-side dormant-slot activation initializes each emitter's used count before the command buffer runs; sink recycling atomically claims the remaining emitter budget and its sink budget. The same WGSL shape containment, sampling, routing, and budgeting functions are embedded in all three solver adapters.

PB-MPM converts launch velocity to its per-substep displacement representation. PBF and MLS-MPM store launch velocity directly.

## JSON Format

Fluid presets use flat top-level arrays:

```json
{
    "formatVersion": 5,
    "gridPosition": [0, 9.5, 0],
    "gridSize": [40, 21, 40],
    "emitters": [],
    "sinks": []
}
```

Complete lab presets keep their established physics, render, foam, demo, and camera fields alongside these arrays. Presets before format 4 stored emitter and sink positions in world space; import subtracts the migrated grid position so the stored transforms become grid-local without moving the installed objects. Format-4 presets stored exact cell counts; import converts those counts to the equivalent world-space size using the preset's cell size.

The machine-readable schema is `config/fluid-flow.schema.json`.

## Simulation Grid Authoring

Complete lab presets store a per-method grid center, exact integer cell dimensions, and `showGridBounds` alongside the existing physics settings:

```json
"gridPosition": [0, 9.5, 0],
"gridSize": [40, 21, 40]
```

`gridPosition` is the world-space center of the axis-aligned simulation grid. `gridSize` is its exact world-space extent along X/Y/Z. Cell size is derived solely from the method and explicit Physics particle size, while the GPU allocation rounds each axis up to whole cubic cells:

```text
cellSize = methodBaseCellSize * physicsParticleSize
gridCells = ceil(gridSize / cellSize)
boundsMin = gridPosition - gridSize / 2
boundsMax = gridPosition + gridSize / 2
```

The UI exposes editable **Grid position** and **Grid size** XYZ values and displays **Cell size** as a derived, read-only value. Positions must be finite and sizes must be positive finite world-space dimensions. The internal derived cell allocation must fit the method/device limits. Invalid values remain visible with an explicit error and are not installed. Position, size, or Physics particle size changes rebuild the solver resources and restart the simulation because grid buffers, bounds, particle spacing, and solver constants are creation-time state.

PBF uses the derived cell size as both its neighbour-grid cell size and smoothing radius. MLS-MPM and PB-MPM use it as `dx`. Physics particle size supports `0.1×` through `8×` in `0.01×` increments. Increasing it enlarges cells and particles while preserving the authored world-space domain, so fewer cells are allocated.

The base simulation-particle radius is `0.09` world units (`0.18` diameter) at physics scale `1`. The installed radius is always `0.09 × physicsScale`; no demo- or mesh-scale multiplier is applied implicitly. A demo mesh-scale change may update grid position, grid size, and explicit Physics particle size by the same ratio, retaining the derived cell allocation. The separate render particle-size control only changes the visual footprint.

Legacy `domain`/`gridResolution` presets migrate by deriving their center and exact world-space size from the domain bounds. Format-4 `gridCells` presets preserve their previous effective extent as `gridSize = gridCells × cellSize`. Presets without either field use the historical bounds: PBF `[-20, 0, -20]..[20, 20, 20]`, and MLS-MPM/PB-MPM `[-20, -1, -20]..[20, 20, 20]`, transformed by the demo's mesh scale. Missing `showGridBounds` defaults to false.

Waterfall presets encode the former hidden `3×` scale explicitly as grid position `[0, 30, 0]`, grid size `[120, 60, 120]`, and a Physics particle size three times the legacy preset value. This preserves both the authored world-space domain and the previous derived cell allocation. Their flow graph remains demo-owned rather than preset-owned so terrain-derived terrace polygons and pond state are always rebuilt from the current scene.

Emitter and sink transform positions are grid-local world-unit offsets:

```text
installedWorldPosition = gridPosition + authoredLocalPosition
```

Moving `gridPosition` therefore moves the solver domain, emitters, and sinks together. It does not move collision SDFs, scene meshes, lights, cameras, or any other scene-owned object. Rotation, scale, shape dimensions, and velocity are unaffected by grid translation. The authoring UI shows local positions; wireframes and gizmos are displayed in installed world space and subtract the grid position when writing a dragged position back.

**Show grid bounds** displays a live, visualization-only wireframe of the derived AABB. Scaling a demo's world scales grid position and explicit Physics particle size while retaining cell counts. The overlay does not draw every cell and has no simulation cost.

A **Gizmo** checkbox attaches the world-space position and axis/uniform scale gizmos together to the grid bounds wireframe. The wireframe is shown while the gizmos are active. Dragging previews the transform; releasing commits the grid position or size and rebuilds the simulation. Scaled sizes are rounded to the nearest `0.1` world unit.

## Blender Mapping

`scripts/export-blender-fluid.py` and the installable `scripts/blender-fluid-addon.py` map:

- Geometry flow objects to `behavior: "initial"`.
- Inflow objects to `behavior: "inflow"`.
- Outflow objects to sinks.
- Blender Z-up transforms to Babylon Lite Y-up transforms.

Custom Blender properties select standard shapes, annular cylinder radius, polygon points, target emitters, spread, emitter/sink volume rate, and static collision. `blite_volume_rate` applies to Blender Inflow and Outflow objects; omission or `0` exports an unlimited flow. `blite_collision` includes a visible mesh in the collision SDF without requiring a Blender Fluid Effector modifier. The add-on exposes these properties directly in its Scene-properties authoring panel while retaining compatibility with existing raw custom properties.

The standalone Blender exporter intentionally writes legacy format 2 flow-only JSON with world-space positions because Blender flow objects do not define the lab's simulation grid. Import localizes those positions against the currently active grid. Export from the lab afterward to produce a complete format 5 preset with explicit grid position, world-space size, and grid-local flow transforms.

The Blender add-on instead finds the active or first liquid domain and exports one complete `.blitefluid` bundle. Install `scripts/blender-fluid-addon.py` through Blender's **Edit > Preferences > Add-ons > Install from Disk**, enable **Babylon Lite Fluid Bundle**, then use **Scene Properties > Babylon Lite Fluid**. The add-on exports a format-5 live-simulation preset, visible static meshes as a self-contained GLB, and liquid effectors (or meshes marked with `blite_collision = true`) as a baked 3D collision SDF. Flow positions are written grid-local, while scene meshes and collision remain fixed in world space when the lab grid is moved. See [Blender Fluid Bundle](53-blender-fluid-bundle.md) for the binary contract and limits.

Blender fluid objects may use arbitrary mesh geometry, but Babylon Lite currently reproduces only the analytical shapes listed above and extruded 2D polygons. The exporter approximates an unsupported/default mesh with a box derived from its local bounding box unless a supported `blite_shape` property is provided. Exact arbitrary-mesh flow regions would require a future volumetric representation such as a baked SDF or voxel field.

## Lab Authoring

The fluid panel has separate **Emitters** and **Sinks** sections. Each section selects one existing object and can create, duplicate, delete, enable, rename, and edit it. The emitter editor exposes behavior, shape, sampling, transform, velocity, velocity space, spread, and an inflow-only unlimited/volume-per-second rate. The sink editor exposes shape, transform, inflow targets, and recycling rate. The editor enforces the runtime object-count limits and only presents inflow emitters as valid sink targets.

Each section has an independent **Wireframe** checkbox for the selected object. Emitters use a cyan overlay and sinks use an orange overlay; both follow the authored position, quaternion, and scale and are visualization-only.

Every flow parameter has a native hover tooltip. A **Gizmo** checkbox attaches position, rotation, and scale controls together to the selected emitter or sink; only one flow object owns the gizmos at a time. Transform gizmo drags repack the selected installed object continuously without resetting particle state.

The panel can be resized from its left edge. Numeric fields accept decimal points or decimal commas, and each flow object remembers its in-session parameter draft for every shape type while switching shapes. The running simulation owns a cloned installed graph so editing one object cannot leak unrelated staged changes from another object.

When the edited object already exists in the installed graph, runtime-safe fields repack that object immediately without resetting:

- Sinks: enabled state, transform, analytical shape, target routing, and rate.
- Emitters: enabled state, behavior, transform, analytical shape, sampling, velocity, velocity space, spread, and rate.

These edits affect subsequent emission, containment, and recycling only; they never teleport particles that are already active. Initial-emitter edits therefore become visible when the simulation is next reset, while inflow-emitter and sink edits affect the running simulation immediately. Add, duplicate, and delete remain staged because they change graph topology; press **R** (or **Reset simulation**) to install those changes and reseed.

Intrinsic dimensions and transform scale intentionally coexist. Shape values define reusable local geometry and preserve direct Blender/JSON semantics; transform scale instances that geometry, supports mirroring, and provides non-uniform scaling without rewriting shape-specific parameters.

## Test Specification

1. Type-check the package and lab.
2. Compile the production fluid bundle.
3. Compile all three composed flow shaders on a WebGPU device.
4. Run box, capsule, fountain, waterfall, and marble-tower flows across PBF, MLS-MPM, and PB-MPM.
5. Verify editor add/duplicate/delete and pair-state restoration.
6. Verify format-5 export/import round trips, format-4 cell-count migration, and legacy world-space flow migration.
7. Verify initial-volume active counts, fractional inflow budgets, capacity exhaustion, unlimited allocation, and shared sink/emitter rate limits.
