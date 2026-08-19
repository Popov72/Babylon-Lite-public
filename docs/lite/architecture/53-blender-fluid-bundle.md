# Module: Blender Fluid JSON

> Package paths: `scripts/blender-fluid-addon.py`, `lab/lite/src/demos/fluid/preset-io.ts`, `lab/lite/src/demos/fluid/blender-fluid-json.ts`

## Purpose

Export a native Blender Mantaflow liquid setup as one self-contained Babylon Lite fluid preset. The add-on does not expose a second simulation UI: Blender's domain, flow, effector, timeline, mesh, and secondary-particle controls are the authoring source, and the Scene-properties panel contains one **Export Babylon Lite Fluid JSON** button.

The export is a live simulation description, not a baked animation. Babylon Lite recreates the liquid with its PBF solver.

## Format

Format 9 extends the existing `FluidExportJson` used by the demo and quality presets. It retains format 6's embedded scene payload, format 7's explicit sink lifecycle behavior, and format 8's optional Blender Initial Velocity fields, then adds a stable emitter-to-GLB-node binding:

```ts
export interface FluidExportJson {
    formatVersion?: number;
    meta: { demo: string; method: string };
    source?: {
        application: string;
        version?: string;
        settings?: Record<string, unknown>;
    };
    simulationTimeScale?: number;
    emitters?: Array<{
        velocity: [number, number, number];
        sourceNode?: string;
        // Retained for format-8/static authoring compatibility. Format 9 derives
        // this value each frame from sourceNode animation.
        sourceVelocity?: [number, number, number];
        sourceVelocityFactor?: number;
        normalVelocity?: number;
        delayBeforeStart?: number;
        // Existing emitter fields...
    }>;
    sinks?: Array<{
        mode: "delete" | "recycle";
        targets: string[];
        // Existing sink fields...
    }>;
    // Existing physics, flow, grid, render, foam, and lifecycle fields...
    scene?: {
        encoding: "base64";
        glb: string;
        collision: string;
        anchorPosition?: [number, number, number];
    };
}
```

`scene.glb` is a base64 self-contained GLB containing every visible Blender presentation mesh and supported punctual light, including flow objects and effectors but excluding the liquid Domain control volume. `scene.collision` is the base64 binary SDF payload. `scene.anchorPosition` records the grid position at which those immutable payload coordinates were authored, allowing a moved imported bundle to be exported and imported again without losing alignment. Files without it use their top-level `gridPosition`. Parameter-only presets omit `scene`. Self-contained Blender exports use format 6 through 9; format 6 sinks are migrated to recycle mode.

The embedded collision payload is little-endian:

```text
offset  type      meaning
0       u32       magic "BLSF" (0x46534c42)
4       u32       version = 1
8       u32       dim X
12      u32       dim Y
16      u32       dim Z
20      u32       reserved
24      f32       origin X
28      f32       origin Y
32      f32       origin Z
36      f32       uniform cell size
40..63  bytes     reserved
64..    f32[]     signed distances, X-fastest
```

## Native Blender Authoring

Use Blender's **Physics Properties > Fluid** controls:

- A liquid **Domain** defines the simulation grid and solver settings.
- `GEOMETRY` liquid Flow modifiers become finite initial emitters.
- `INFLOW` liquid Flow modifiers become continuous sources whose supply is effectively unlimited while enabled.
- `OUTFLOW` liquid Flow modifiers become `mode: "delete"` sinks. They free particle slots independently of any emitter.
- Every visible presentation mesh and glTF-compatible punctual light (`POINT`, `SUN`, or `SPOT`), including Flow, Outflow, collision Effectors, and ordinary scene geometry, is exported to reconstruct the Blender scene. The liquid Domain control volume is excluded because Babylon Lite reconstructs the fluid itself. Blender area lights have no `KHR_lights_punctual` representation and produce an explicit export warning instead of being silently approximated.
- Enabled collision Effectors additionally contribute to the baked collision SDF, even when **Disable in Renders** is set. Render-hidden effectors are collision-only and are omitted from `scene.glb`.
- Emitter records retain a `sourceNode` link to their Blender flow object's GLB node. Emission remains analytical; source mesh triangles are not sampled.

An Outflow must overlap the liquid particle layer, not merely the domain boundary. At coarse Mantaflow resolutions, particle centers can remain one or more cells above the domain floor. Prefer a thin floor-level mesh with enough native **Surface Emission** distance to reach that layer rather than raising the physical Outflow mesh. The exporter expands the corresponding Babylon Lite sink by the same number of Mantaflow cells.

Mantaflow's **System Maximum** is a cap on particles alive simultaneously, not a lifetime emission budget. An Outflow deletes particles and frees slots under that cap; an enabled continuous Inflow can then create new particles. Babylon Lite models the same independent deletion and emission behavior with sparse GPU particle slots. Explicit recycle sinks remain available for authored closed-loop pumps and legacy files.

The exporter validates before writing. Missing or degenerate domains, flow objects wholly outside the domain, object-count overflow, missing presentation geometry, and oversized grids fail explicitly. Colliders outside the domain and non-manifold collision meshes produce warnings.

## Mantaflow Mapping

Babylon Lite uses PBF because Mantaflow's FLIP/APIC liquid is an incompressible particle liquid and does not map honestly to elastic/sand PB-MPM materials.

| Blender setting                                        | Babylon Lite field                                    |
| ------------------------------------------------------ | ----------------------------------------------------- |
| Domain cage bounds                                     | `gridPosition`, `gridSize`                            |
| `resolution_max`                                       | physics particle size and derived simulation capacity |
| Add-on Collision SDF resolution                        | embedded collision-grid resolution                    |
| `sys_particle_maximum`, or cell count x `particle_max` | `particleCount`                                       |
| Downward Z gravity                                     | PBF `gravity`                                         |
| CFL condition                                          | PBF `relaxation`                                      |
| Maximum adaptive timesteps                             | PBF `iterations`                                      |
| Diffusion surface tension                              | PBF `scorr`                                           |
| Diffusion viscosity base/exponent                      | PBF XSPH `viscosity`                                  |
| `time_scale`                                           | `simulationTimeScale`                                 |
| Flow transform/bounds                                  | grid-local box emitter/sink                           |
| Enabled Initial Velocity X/Y/Z                         | emitter `velocity` in world space                     |
| Flow object/node link                                  | emitter `sourceNode`                                  |
| Enabled Initial Velocity Source                        | animated node velocity x `sourceVelocityFactor`       |
| Enabled Initial Velocity Normal                        | per-particle analytical `normalVelocity`              |
| Continuous liquid Inflow                               | calibrated `volumeRate = 50`                          |
| `use_mesh`, mesh particle radius/smoothing             | surface render settings                               |
| Foam, Spray, or Bubbles enabled                        | `foam.enableFoam`                                     |
| Bubbles enabled                                        | `foam.subsurfaceBubbleStrength = 0.2`                 |

For an unparented Domain without constraints, cage bounds use the authored `matrix_basis`. Blender normally keeps it identical to `matrix_world`, but saved Mantaflow scenes can retain the intended location and scale in `matrix_basis` while exposing a stale identity `matrix_world`. Parented or constrained Domains continue to use `matrix_world` so inherited and evaluated transforms remain authoritative.

Values without a direct runtime equivalent are retained exactly under `source.settings`, grouped into timeline, domain, flows, and effectors. This prevents information loss and gives future mappings a stable source without adding Babylon-specific controls to Blender.

The uncalibrated PBF particle-size conversion is the Mantaflow voxel size divided by Babylon Lite's base PBF cell size:

```text
mantaflowVoxelSize = max(gridSize) / resolution_max
uncalibratedPhysicsParticleSize = mantaflowVoxelSize / 0.4
physicsParticleSize = clamp(uncalibratedPhysicsParticleSize * 2.1333333333333333, 0.7, 8)
```

For the `12 x 6 x 12`, resolution-32 smoke domain, the original conversion was `12 / 32 / 0.4 = 0.9375`. Visual comparison with Blender requires the calibrated multiplier `2.1333333333333333`, producing `2.0`.

Mantaflow does not expose a physical volume-per-second value that maps directly to Babylon Lite. Continuous Inflows therefore export with the visually calibrated fixed rate `20`. Flow dimensions, FPS, and subframes remain available under `source.settings` but do not multiply the exported rate.

When Blender **Diffusion** is enabled, viscosity Base and Exponent are first combined as `base * 10^-exponent`, then mapped logarithmically into PBF's dimensionless XSPH viscosity range and clamped to `1.9`. Surface Tension is mapped linearly into PBF `scorr`. Both are calibrated visual approximations rather than unit-equivalent solver parameters: XSPH is velocity smoothing, while `scorr` is PBF's artificial-pressure surface-stability term. Blender's separate **High Viscosity** solver toggle remains metadata because Babylon Lite has no equivalent numerical solver mode.

On JSON import, gravity and PBF artificial pressure (`scorr`) are truncated toward zero to three decimal places so Blender floating-point serialization noise does not leak into the controls or live solver state. Other physics values retain their authored precision.

Every emitter exports its Blender flow-object name as `sourceNode`. Blender fluid bundles already convert presentation, analytical flow, grid, and collision data into one shared bundle basis, so Whiteboard cancels the glTF loader's synthetic X mirror on the imported asset root before adding it to the scene. Babylon Lite then resolves each unique glTF transform node, ignoring the attached render-mesh child even when Blender gave the glTF node and mesh the same name. Animated position/rotation/scale and derived Source velocity follow that rendered bundle-basis transform directly. When a bundle references a source node absent from its GLB, import emits a warning and retains the emitter's authored static analytical transform instead of rejecting the complete scene. Initial Velocity fields are emitted only when Blender's **Initial Velocity** option is enabled: the authored Source multiplier applies to that derived velocity, while Normal velocity is evaluated per launched particle from the analytical emitter shape. The source mesh remains visual/transform data only; its triangles are not used for emission.

Reset-time volume-sampled Initial emitters use a deterministic, evenly spaced lattice rather than independent random points. The lattice spacing is derived from the emitter's allocated world volume per particle, transformed through the emitter's scale and rotation, and tightened only when curved boundaries leave too few valid sites. Candidate sites outside the simulation grid are discarded before activation rather than being clamped together by the first solver step; this matches Mantaflow's clipping of Initial geometry to its Domain. These rules remove random overlaps, density clumps, and boundary-compression spikes, so Initial liquid is visible immediately and begins close to the solver's rest spacing. Gravity, collision geometry, and a free surface can still cause a small physical adjustment; exact hydrostatic equilibrium cannot be inferred from emitter geometry alone.

`delayBeforeStart` remains an optional non-negative simulation-time delay for independently authored Inflows. During the delay, the Inflow receives no volume budget and cannot be selected as a recycle target. Blender exports no automatic Initial-settling delay: Initial lattice particles and Inflows both begin at simulation time zero.

The emitter editor shows the linked **Source mesh** name and retains **Delay before start** for independently delayed Inflows plus optional **Source factor** and **Normal velocity** controls behind **Source + normal**. Source velocity itself is read-only derived state and is not authored in the UI. **Velocity (XYZ)** remains independent because it is the constant launch velocity Blender authors explicitly; the derived Source velocity is the linked object's motion.

Blender exports use the Whiteboard PBF render profile: surface rendering, particle size `0.6`, depth blur `17`, thickness blur `2`, half-resolution rendering, thickness downscale `8`, and the narrow-range filter with delta `2` and mu `1`, together with the profile's remaining color, refraction, reflection, and anisotropy values.

When Blender enables **Foam**, **Spray**, or **Bubbles**, the exported preset enables Babylon Lite foam and applies the approved smoke-scene profile: trapped-air rate `51`, wave-crest rate `48`, lifetime `1.0416666666666667`, minimum lifetime `0.4166666666666667`, buoyancy `4.2`, drag `0.45`, pool size `3.5`, softness `0`, density `8.25`, blur radius `1`, ambient `1`, size `0.15`, and subsurface color `#5380ea`. Subsurface bubble strength is `0.2` only while Blender **Bubbles** is enabled; otherwise it is `0`, including Foam-only and Spray-only exports.

Blender 5.2 retains a hidden `velocity_random` RNA property, but its liquid-flow UI row is disabled and Mantaflow does not apply the value. It remains in `source.settings.flows` for lossless diagnostics and is not mapped to Babylon Lite `spread`.

The simulation duration is indefinite and alpha decay is disabled because Mantaflow does not fade liquid at the cache end. Blender cache settings are retained as source metadata but do not limit the live Babylon simulation.

Foam debug mode uses `"off"` for normal rendering. Imports normalize the older `"none"` value and the empty value produced by affected round trips to `"off"`.

When a self-contained Blender JSON is active in any demo, including Whiteboard, **Export parameters** emits another self-contained JSON. Live simulation/UI values replace the preset values, while the active GLB, collision SDF, and Blender `source` metadata are preserved.

## Collision

The add-on exposes one export setting, **Collision SDF resolution**, independent from Mantaflow's **Resolution Divisions**. It sets the number of grid points on the longest SDF axis. Other axes use the same cell size:

```text
cellSize = max(gridSize) / (resolution - 1)
dims = ceil(gridSize / cellSize) + 1
origin = gridPosition - gridSize / 2
textureBytes = dims.x * dims.y * dims.z * 4
```

The add-on displays the resulting dimensions and `R32Float` GPU texture size in MiB below the resolution control. The JSON collision payload adds a 64-byte header and base64 encoding increases its textual size by roughly one third.

Blender evaluates modifiers, combines enabled liquid collision effectors into one BVH regardless of render visibility, and stores negative distances inside solids. Render-hidden effectors provide collision without adding presentation geometry to `scene.glb`. Collision meshes should be closed and consistently oriented.

## Runtime Import

1. Parse and validate the format-6 through format-9 preset, inferring recycle sinks for format 6.
2. Decode and validate the embedded GLB and collision payload.
3. Load the GLB from its `ArrayBuffer`, register all animation groups, and play every clip on a continuous loop.
4. Suspend the dashboard demo's presentation meshes, ground, and direct lights. Whiteboard keeps its selected environment cubemap as the background and leaves the single-sample/MSAA scene passes in skybox-covered no-clear mode; imports hosted by other demos hide the skybox and switch those passes to per-frame clear mode. Both paths redraw every pixel after camera motion, so stale color cannot remain.
5. Cancel the imported GLB root's synthetic X mirror, resolve each format-9 emitter's `sourceNode`, then copy its rendered bundle-basis transform to the analytical emitter after animation advances each frame. Derived Source velocity updates without resetting emission budgets.
6. Install the collision storage/uniform buffers as the active `SceneSdfSpec`. In Whiteboard's MLS-MPM backend, intersect its free space with the invisible grid container so imported obstacles and the inset numerical-boundary walls are both enforced.
7. Set each solver's safety floor to the imported grid's lower bound. Treat the imported grid position as the bundle's translation anchor: editing it translates the GLB presentation root, baked collision-SDF origin, analytical emitters, and sinks together without changing their grid-local layout or generating synthetic source velocity.
8. Restore grid, PBF physics, time scale, emitters, sinks, render, and foam settings.
9. For Whiteboard imports only, use the Blender front view and frame the camera around the union of the simulation grid and every emitter/sink's transformed analytical shape. Presentation-only meshes are excluded so oversized grounds, walls, and backgrounds cannot pull the camera away from the fluid or place the camera behind a backdrop.
10. Reset and begin visible playback immediately. Volume-sampled Initial emitters are seeded on their deterministic rest-spacing lattice before the first rendered simulation frame.

Importing another authored scene removes the previous GLB animation groups and retires its GPU resources. Switching to a built-in demo removes the imported scene override and restores that demo's authored default camera instead of retaining the imported framing or a previously manipulated view. Format-5 JSON presets remain parameter-only.

Before allocating simulation buffers, imports requesting more than `100,000` particles show the exact requested count and ask whether to use the safer `40,000` count instead. Accepting replaces only the live imported capacity; cancelling preserves the JSON value.

Resetting the fluid with `R` or **Reset simulation** also rewinds imported GLB animations so source meshes and the simulation restart together. Holding Shift while pressing `R` or clicking the button resets only the fluid and preserves the current mesh-animation time.

## Validation

- Unit tests cover format-6 migration, format-7 sink modes, format-8 Initial Velocity fields, format-9 source-node bindings, dynamic emitter-buffer updates, base64 scene parsing, collision headers, preset bounds, flow references, and lifecycle/time-scale bounds.
- The focused Chrome/WebGPU workflow imports format-9 JSON, verifies scene/SDF/preset restoration, plays an animated GLB, follows a linked source node, then confirms cleanup on demo switch.
- Blender background smoke export verifies native domain/flow/effector discovery and produces a self-contained `.json`.
