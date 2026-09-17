# Module: Blender Fluid JSON

> Package paths: `scripts/blender-fluid-addon.py`, `packages/babylon-lite/src/fluid/authoring/preset-io.ts`, `packages/babylon-lite/src/fluid/authoring/blender-fluid-json.ts`

## Purpose

Export either a native liquid setup or a third-party FLIP add-on setup as a Babylon Lite fluid preset. The scene GLB and collision SDF can be embedded in one JSON or written as separately referenced files. Both domain types map to Babylon Lite's FLIP backend; the exporter reads the source-specific settings and normalizes them into one shared FLIP schema.

The export is a live simulation description, not a baked animation. Babylon Lite recreates the liquid with its FLIP solver.

## Format

Format 16 adds an optional exact baked FLIP initial state. It retains format 15's rigid animated mesh collisions, format 14's explicit solver-value semantics, format 10's FLIP-native resolution and marker-density fields, format 6's embedded scene payload, format 7's explicit sink lifecycle behavior, format 8's optional Blender Initial Velocity fields, format 9's stable emitter-to-GLB-node binding, and the later FLIP/foam controls:

```ts
export interface FluidExportJson {
    formatVersion?: number;
    simulationSemantics?: {
        version: 1;
        profile: "normalized-v1" | "legacy-fluid" | "legacy-aquanova";
        pbfPhysics: "scale-adjusted" | "literal";
    };
    meta: { demo: string; method: string };
    source?: {
        application: string;
        version?: string;
        settings?: Record<string, unknown>;
    };
    simulationTimeScale?: number;
    gridResolution?: number;
    markersPerCell?: number;
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
        delayBeforeStart?: number;
        mode: "delete" | "recycle";
        targets: string[];
        // Existing sink fields...
    }>;
    // Existing physics, flow, grid, render, foam, and lifecycle fields...
    scene?: {
        encoding: "base64" | "external";
        glb: string;
        collision: string;
        sdfCompression?: "zlib";
        collisionEnabled?: boolean;
        collisionTrilinear?: boolean;
        collisionByteLength?: number;
        anchorPosition?: [number, number, number];
        initialState?: {
            data: string;
            byteOffset?: number;
            byteLength?: number;
            count: number;
            frame: number;
            space: "world";
        };
        animatedCollisions?: Array<{
            id: string;
            node: string;
            sdf: string;
            byteOffset?: number;
            byteLength?: number;
            space: "node-local";
            resolution: number;
            bakeFrame: number;
            presentation: boolean;
            enabled?: boolean;
            trilinear?: boolean;
        }>;
    };
}
```

With `encoding: "base64"`, every GLB/SDF/initial-state string contains embedded data. With `encoding: "external"`, the export remains exactly three files: JSON, GLB, and one shared SDF container. New exports set `sdfCompression: "zlib"`. External mode compresses the complete concatenated container; embedded mode compresses each static/animated BLSF or BLFI payload before base64 encoding. Byte offsets and lengths always address decompressed bytes. Legacy uncompressed and separate-SDF format-15 manifests remain readable. `scene.collision` remains the static world-space union. Each `animatedCollisions` entry links one object-local SDF to a unique glTF transform node. The optional `initialState` is a world-space BLFI payload containing aligned float32 XYZ position and velocity arrays from the first effective bake frame. `presentation: false` keeps a hidden collision-only node and its animation in the GLB while suppressing its render mesh after import. The GLB otherwise contains visible Blender presentation meshes and, when enabled, supported punctual lights, excluding the liquid Domain control volume and FLIP Fluids' generated surface/whitewater cache objects. Babylon Lite reconstructs those outputs live. Optional Blender Decimate modifiers target a user-authored total triangle count across exported presentation meshes and are removed immediately after GLB generation; selected animated collision meshes are protected from decimation. `scene.anchorPosition` records the grid position at which the immutable static payload coordinates were authored, allowing a moved imported bundle to be exported and imported again without losing alignment. Files without it use their top-level `gridPosition`. Parameter-only presets omit `scene`. Blender exports accept formats 6 through 16; format 6 sinks are migrated to recycle mode. Formats 14 through 16 require explicit semantics. FLIP exports use `gridResolution` as divisions along the longest domain side and default `markersPerCell` to `8`.

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

## Blender Authoring

Use Blender's **Physics Properties > Fluid** controls:

- A liquid **Domain** defines the simulation grid and solver settings.
- **Target initial-fluid particles** optionally replaces the source domain resolution for export. `0` preserves the authored Resolution Divisions; a positive value chooses the longest-axis resolution whose domain-clipped deterministic marker lattice is closest to the target. Rotated or oversized Initial objects therefore count only sites the runtime can actually seed inside the FLIP domain.
- **Target inflow particles** adds dormant capacity for all Inflow emitters on top of the initial marker count. It does not affect Resolution Divisions. The exported capacity is `resolved initial markers + target inflow particles`, preventing a large Initial volume from consuming every available slot before an Inflow starts.
- Each target input is disabled when the scene has no emitter of the corresponding class.
- The panel displays the resulting **Resolution Divisions** and the informational breakdown `initial + inflow = total particles`. This preview uses the same clipped-lattice derivation as export and caches unchanged scene inputs so regular panel redraws remain inexpensive.
- Planar Initial objects are included in the initial-particle target. They use the same one-cell-thick analytical shape exported to the runtime, matching its surface-emitter count semantics.
- `GEOMETRY` liquid Flow modifiers become finite initial emitters.
- `INFLOW` liquid Flow modifiers become continuous sources whose supply is effectively unlimited while enabled.
- `OUTFLOW` liquid Flow modifiers become `mode: "delete"` sinks. They free particle slots independently of any emitter.
- Every visible presentation mesh and glTF-compatible punctual light (`POINT`, `SUN`, or `SPOT`), including Flow, Outflow, collision Effectors, and ordinary scene geometry, is exported to reconstruct the Blender scene. A flow mesh linked through `sourceNode` is also exported when render-hidden so its transform remains resolvable; `sourcePresentation: false` suppresses its attached render mesh after import. The liquid Domain control volume is excluded because Babylon Lite reconstructs the fluid itself. Blender area lights have no `KHR_lights_punctual` representation and produce an explicit export warning instead of being silently approximated.
- The active perspective render camera is exported as an ArcRotate-compatible eye/target orbit plus the effective render-frame vertical field of view. Its target lies on the authored camera ray near the fluid-domain center, preserving the Blender eye position and view direction while providing a useful Whiteboard orbit pivot. The camera also requests a horizontal presentation mirror: the imported GLB basis cancels the loader's synthetic X reflection for fluid-coordinate alignment, so Whiteboard flips only its final fullscreen composite. Scene and polygon passes retain their original winding/front-face semantics. Horizontal orbit and pan input are reversed, and custom-force gestures mirror both cursor-ray X and drag X, so all visible screen-space interaction remains natural. Orthographic cameras remain unsupported and fall back to automatic Whiteboard framing.
- Mirrored camera input belongs entirely to the Fluid app's `camera-controls.ts` adapter, not the core controls API. Its pointer-move listeners bracket the stock control listener with paired sign conversions of horizontal orbit/pan inertia. Existing momentum is restored, newly accumulated horizontal input is reversed, and a gizmo abort still clears inertia. Vertical input, wheel/pinch zoom, original pointer coordinates, and external-interaction guards remain unchanged. The live mirror flag applies on each move. The adapter's disposer removes its listeners and the stock controls together, and the host registers it for scene disposal.
- A directly connected Blender **Checker Texture → Base Color** is converted temporarily to an embedded image texture for glTF export and restored afterward. This preserves checkerboard floor planes that glTF would otherwise flatten to the material's gray fallback because procedural shader nodes are not part of core glTF.
- Enabled collision Effectors additionally contribute to the baked collision SDF, even when **Disable in Renders** is set. Render-hidden effectors are collision-only and are omitted from `scene.glb`.
- Every animated mesh is listed under **Animated mesh collisions** with an independent `0..2048` local-SDF resolution. `0` means the mesh has no fluid collision, even when it is a native Effector or FLIP obstacle. A positive value exports a separate local-space SDF and includes the animated transform node in the GLB, including render-hidden objects such as `WaveMaker`.
- **Export baked initial positions/velocities** is a FLIP Fluids-only opt-in. Enabling it also enables full fluid-particle position and velocity cache output for the next bake. Export requires both `.ffp3` files at the effective first bake frame, converts Blender XYZ into Babylon world coordinates, and stores the exact markers in BLFI format. Babylon activates that exact marker count on reset while retaining `particleCount` as spare capacity for reseeding.
- The export interval is the effective fluid bake range: FLIP Fluids' Timeline/Custom range (including its resumable-cache start) or Mantaflow's cache frame range. The add-on displays this interval in the export panel, samples collision geometry only inside it, bakes static and local SDFs at its first frame, and temporarily constrains glTF animation export to the same bounds. Exported animation timestamps are shifted so the first bake frame is time zero. Keyframes, modifier changes, and transforms entirely before or after this interval do not affect collision classification or the exported clip.
- Source diagnostics store the effective simulation FPS separately from Blender's render FPS. Offline rendering uses the simulation FPS for solver `fixed-dt` and the render FPS for MP4 playback by default, preserving scenes that intentionally simulate at a different cadence than their timeline.
- FLIP `maxSubDtMs` is derived from one Blender simulation frame divided by Minimum substeps. This lets an unsaturated Babylon frame begin with the same minimum substep cadence as FLIP Fluids; adaptive CFL may still increase the count, and the two engines' timestep schedulers are not otherwise identical.
- Animated collision baking currently supports rigid object/parent transforms within the effective fluid interval. An object with animation data but no in-range change is folded into the static world-space collision union; an in-range rigid transform uses one object-local SDF plus its glTF node animation. In-range deformation or topology changes are rejected explicitly; armature, shape-key, and time-varying modifier collisions require a future deforming-SDF format.
- Emitter records retain a `sourceNode` link to their Blender flow object's GLB node. Emission remains analytical; source mesh triangles are not sampled.
- Planar flow objects are expanded to one simulation-cell thickness along zero-width axes. This preserves Blender's planar inflow authoring while satisfying the shared analytical box-shape contract.
- Axis-aligned extruded flow meshes are exported as analytical `polygonPrism` shapes. Other meshes use their local bounding box centered on the actual local bounds, not the Blender object origin. This preserves offset wedges and prisms such as ramp-following initial-fluid volumes without embedding arbitrary flow triangles in the runtime buffer.
- **Export separate files** always writes exactly `.json`, `.glb`, and `.sdf`. The SDF file concatenates the static BLSF payload and every enabled animated BLSF payload; JSON byte ranges identify each grid. Browser import therefore keeps the original three-file selection workflow.
- **Export lights** controls whether visible point, sun, and spot lights are included.
- **Decimate exported meshes** adds temporary Decimate modifiers with one global ratio derived from the requested total scene triangle count. Meshes with at most 256 triangles remain exact so low-poly structural surfaces, ramps, containers, and flow guides cannot be topologically damaged for negligible savings. No exporter ceiling is applied to the target.

An Outflow must overlap the liquid particle layer, not merely the domain boundary. At coarse Mantaflow resolutions, particle centers can remain one or more cells above the domain floor. Prefer a thin floor-level mesh with enough native **Surface Emission** distance to reach that layer rather than raising the physical Outflow mesh. The exporter expands the corresponding Babylon Lite sink by the same number of Mantaflow cells.

Mantaflow's **System Maximum** is a cap on particles alive simultaneously, not a lifetime emission budget. An Outflow deletes particles and frees slots under that cap; an enabled continuous Inflow can then create new particles. Babylon Lite models the same independent deletion and emission behavior with sparse GPU particle slots. Explicit recycle sinks remain available for authored closed-loop pumps and legacy files.

The exporter validates before writing. Missing or degenerate domains, flow objects wholly outside the domain, object-count overflow, missing presentation geometry, and oversized grids fail explicitly. Colliders outside the domain and non-manifold collision meshes produce warnings.

## Mantaflow Mapping

Native Mantaflow liquid uses FLIP or APIC particles on a MAC grid, so it maps to Babylon Lite's FLIP backend rather than the unrelated SPH/PBF solver.

| Blender setting                            | Babylon Lite field                               |
| ------------------------------------------ | ------------------------------------------------ |
| Domain cage bounds                         | `gridPosition`, `gridSize`                       |
| `resolution_max`                           | `gridResolution`                                 |
| `particle_number³`                         | `markersPerCell`, clamped to `1..64`             |
| `sys_particle_maximum`                     | `particleCount` capacity hint                    |
| Downward Z gravity                         | FLIP `gravity`                                   |
| `flip_ratio`                               | `flipRatio`                                      |
| Minimum/maximum adaptive timesteps         | `minSubsteps`, `maxSubsteps`                     |
| CFL condition                              | `cflNumber`                                      |
| Diffusion viscosity base/exponent          | `kinematicViscosity`                             |
| High-viscosity value                       | lower-bound contribution to `kinematicViscosity` |
| Diffusion surface tension                  | `surfaceTension`                                 |
| Minimum/maximum particles per cell         | FLIP reseeding bounds                            |
| Fractional obstacles                       | `fractionalSolids`                               |
| Add-on Collision SDF resolution            | embedded collision-grid resolution               |
| `time_scale`                               | `simulationTimeScale`                            |
| Flow transform/bounds                      | grid-local box emitter/sink                      |
| Enabled Initial Velocity X/Y/Z             | emitter `velocity` in world space                |
| Flow object/node link                      | emitter `sourceNode`                             |
| Enabled Initial Velocity Source            | animated node velocity x `sourceVelocityFactor`  |
| Enabled Initial Velocity Normal            | per-particle analytical `normalVelocity`         |
| Continuous liquid Inflow                   | calibrated `volumeRate = 20`                     |
| `use_mesh`, mesh particle radius/smoothing | surface render settings                          |
| Foam, Spray, or Bubbles enabled            | `foam.enableFoam`                                |
| Bubbles enabled                            | `foam.subsurfaceBubbleStrength = 0.2`            |

## Add-on FLIP Mapping

The exporter recognizes objects whose `flip_fluid.object_type` is `TYPE_DOMAIN`, `TYPE_FLUID`, `TYPE_INFLOW`, `TYPE_OUTFLOW`, or `TYPE_OBSTACLE`.

| Add-on setting or concept          | Babylon Lite field                                    |
| ---------------------------------- | ----------------------------------------------------- |
| Domain bounds                      | `gridPosition`, `gridSize`                            |
| Simulation `resolution`            | `gridResolution`                                      |
| Standard 2 x 2 x 2 marker layout   | `markersPerCell = 8`                                  |
| PIC/FLIP ratio                     | `flipRatio = 1 - PICFLIP_ratio`                       |
| Pressure solver maximum iterations | `pressureIterations`, clamped to the WebGPU range     |
| Minimum/maximum frame substeps     | `minSubsteps`, `maxSubsteps`                          |
| CFL condition number               | `cflNumber`                                           |
| Fluid object                       | Initial volume emitter                                |
| Inflow object                      | Continuous inflow emitter                             |
| Outflow object                     | Delete sink; gradual rate maps to world volume/second |
| Obstacle object                    | Collision SDF contributor                             |
| Initial/inflow velocity            | Emitter launch velocity                               |
| Add object velocity                | Animated `sourceNode` velocity and influence          |

For FLIP presets, active initial markers are derived from authored initial-fluid volume, cell size, and markers per cell. `particleCount` remains GPU capacity, primarily providing dormant slots for inflows. The exporter does not silently lower authored marker capacity or subdivision quality; users opt into mesh decimation, while runtime import retains its explicit high-particle warning and device-fit controls.

For an unparented Domain without constraints, cage bounds use the authored `matrix_basis`. Blender normally keeps it identical to `matrix_world`, but saved Mantaflow scenes can retain the intended location and scale in `matrix_basis` while exposing a stale identity `matrix_world`. Parented or constrained Domains continue to use `matrix_world` so inherited and evaluated transforms remain authoritative.

The same authored-transform rule applies to unparented Flow and static collider objects. This keeps analytical emitters and collision baking aligned with glTF, which exports Blender's authored location even when a saved Mantaflow modifier exposes a stale identity `matrix_world`.

Values without a direct runtime equivalent are retained exactly under `source.settings`, grouped into timeline, domain, flows, and effectors. This prevents information loss and gives future mappings a stable source without adding Babylon-specific controls to Blender.

Native `particle_number` is the number of marker samples per cell axis, so the exported marker density is its cube. The default value `2` therefore becomes `markersPerCell = 8`. The authored FLIP ratio and adaptive timestep bounds carry over directly.

Mantaflow does not expose a physical volume-per-second value that maps directly to Babylon Lite. Continuous Inflows therefore export with the visually calibrated fixed rate `20`. Flow dimensions, FPS, and subframes remain available under `source.settings` but do not multiply the exported rate.

When Blender **Diffusion** is enabled, viscosity Base and Exponent combine as `base * 10^-exponent` and map directly to FLIP kinematic viscosity. An enabled High Viscosity value provides a lower bound for that coefficient. Surface Tension maps directly to the FLIP surface-force coefficient; both values are clamped to the shared control ranges.

Every emitter exports its Blender flow-object name as `sourceNode`. Source nodes are included in the GLB even when **Disable in Renders** is set, and `sourcePresentation` records whether their mesh should be shown. Blender fluid bundles already convert presentation, analytical flow, grid, and collision data into one shared bundle basis, so Whiteboard cancels the glTF loader's synthetic X mirror on the imported asset root before adding it to the scene. Babylon Lite then resolves each unique glTF transform node, ignoring or hiding the attached render-mesh child as requested. Animated position/rotation/scale follow that rendered bundle-basis transform while retaining the analytical shape's initial offset and axis basis; derived Source velocity tracks the resolved shape center. Legacy bundles that reference a source node absent from their GLB emit a warning and retain the emitter's authored static analytical transform instead of rejecting the complete scene. Initial Velocity fields are emitted only when Blender's **Initial Velocity** option is enabled: the authored Source multiplier applies to that derived velocity, while Normal velocity is evaluated per launched particle from the analytical emitter shape. The source mesh remains transform data only for analytical emission; its triangles are not sampled.

Reset-time volume-sampled Initial emitters use a deterministic, evenly spaced lattice rather than independent random points. The lattice spacing is derived from the emitter's allocated world volume per particle, transformed through the emitter's scale and rotation, and tightened only when curved boundaries leave too few valid sites. Candidate sites outside the simulation grid are discarded before activation rather than being clamped together by the first solver step; this matches Mantaflow's clipping of Initial geometry to its Domain. These rules remove random overlaps, density clumps, and boundary-compression spikes, so Initial liquid is visible immediately and begins close to the solver's rest spacing. Gravity, collision geometry, and a free surface can still cause a small physical adjustment; exact hydrostatic equilibrium cannot be inferred from emitter geometry alone.

`delayBeforeStart` is an optional non-negative simulation-time delay for independently authored Inflows and Sinks. During an Inflow delay, it receives no volume budget and cannot be selected as a recycle target. During a Sink delay, it neither deletes nor recycles particles; a finite capture rate budgets only the active fraction of the frame that crosses the delay. Blender exports no automatic Initial-settling delay: Initial lattice particles, Inflows, and Outflows all begin at simulation time zero unless edited after import.

The emitter editor shows the linked **Source mesh** name and retains **Delay before start** for independently delayed Inflows plus optional **Source factor** and **Normal velocity** controls behind **Source + normal**. The sink editor exposes the same **Delay before start** control for delete and recycle sinks. Source velocity itself is read-only derived state and is not authored in the UI. **Velocity (XYZ)** remains independent because it is the constant launch velocity Blender authors explicitly; the derived Source velocity is the linked object's motion.

Blender exports use the shared Whiteboard render profile: surface rendering, particle size `0.6`, depth blur `18`, thickness blur `6`, half-resolution rendering, thickness downscale `8`, and the narrow-range filter with delta `10` and mu `1`. Polygon-surface reconstruction is disabled by default. The fluid color is read from the active surface shader's unlinked Base Color on the native Domain or FLIP generated surface and converted from Blender linear RGB to sRGB; scenes without a readable material retain the blue fallback.

When Blender enables **Foam**, **Spray**, or **Bubbles**, the exported preset enables Babylon Lite foam and applies the approved smoke-scene profile: trapped-air rate `51`, wave-crest rate `48`, lifetime `1.0416666666666667`, minimum lifetime `0.4166666666666667`, buoyancy `4.2`, drag `0.45`, pool size `3.5`, softness `0`, density `8.25`, blur radius `1`, ambient `1`, size `0.15`, and subsurface color `#5380ea`. Subsurface bubble strength is `0.2` only while Blender **Bubbles** is enabled; otherwise it is `0`, including Foam-only and Spray-only exports.

Blender 5.2 retains a hidden `velocity_random` RNA property, but its liquid-flow UI row is disabled and Mantaflow does not apply the value. It remains in `source.settings.flows` for lossless diagnostics and is not mapped to Babylon Lite `spread`.

The simulation duration is indefinite and alpha decay is disabled because Mantaflow does not fade liquid at the cache end. Blender cache settings are retained as source metadata but do not limit the live Babylon simulation.

Foam debug mode uses `"off"` for normal rendering. Imports normalize the older `"none"` value and the empty value produced by affected round trips to `"off"`.

When a Blender JSON is active in any demo, including Whiteboard, **Export parameters** preserves the live simulation/UI values and Blender `source` metadata. An externally sourced scene exposes **Embed external GLB/SDF in JSON**: unchecked preserves every original external GLB/SDF filename and changes only JSON-authored state; checked embeds the already-resolved static and animated SDF bytes into a self-contained JSON. Embedded imports remain embedded.

## Collision

The add-on exposes **Static collision SDF resolution**, independent from Mantaflow's **Resolution Divisions**, with a control range of 8 through 2048. It sets the number of grid points on the longest domain-space SDF axis. Other axes use the same cell size:

```text
cellSize = max(gridSize) / (resolution - 1)
dims = ceil(gridSize / cellSize) + 1
origin = gridPosition - gridSize / 2
textureBytes = dims.x * dims.y * dims.z * 4
```

The add-on displays the resulting dimensions and uncompressed f32 payload size in MiB below the static resolution control. Each animated mesh has a separate longest-axis resolution and up to two positive padding cells around its evaluated local bounds. Every collision payload uses the same 64-byte BLSF header. Zlib compression is lossless and typically reduces smooth SDF fields to roughly one third of their raw size; embedded base64 then adds its usual one-third textual overhead to the compressed bytes. Decompression is bounded to 512 MiB per resource. The 16-million-voxel guard applies to the combined static and animated atlas, and at most 16 animated collision grids can be installed.

Blender evaluates modifiers and builds one BVH per enabled static liquid collision effector regardless of render visibility. Animated meshes are always excluded from this static union, so resolution `0` cannot leave a frozen ghost collider. For each collision-grid sample, the exporter finds the nearest unsigned surface distance and classifies the point against each candidate object's closed volume with ray parity. The signed-distance union is negative when any collider contains the point. Per-object classification avoids nearest-triangle sign ambiguity at corners and edges, which previously created phantom vertical solids and localized FLIP marker traps. Animated SDFs use the same distance/sign calculation in object-local bundle coordinates. Collision meshes should be closed and consistently oriented.

The Fluid panel's **Collision** section appears for imported SDF bundles. It lists the static grid and every animated local grid with authored resolution and actual dimensions. Each entry can be disabled independently or switched between trilinear and nearest-neighbour sampling. These settings update uniform metadata only: atlas buffers, bind groups, and solver pipelines remain unchanged. The flags round-trip through `collisionEnabled`, `collisionTrilinear`, and each animated entry's `enabled`/`trilinear` fields.

The on-disk BLSF payload remains f32 for authoring fidelity and compatibility. During import, every distance is divided by its grid cell size, converted to IEEE float16, and packed two values per `u32` in the shared GPU atlas. WGSL decodes with `unpack2x16float` and restores world/local distance by multiplying by cell size after nearest or trilinear sampling. Cell-relative quantization keeps precision concentrated around the zero surface while halving atlas storage. Legacy custom and floating-body SDF buffers retain their original f32 layouts.

Nearest mode reconstructs a local linear distance from the nearest voxel and its central-difference gradient instead of returning a piecewise-constant raw cell value. For animated local grids it also disables sub-frame matrix extrapolation: the obstacle follows its current visual pose, but contributes no derived moving-boundary velocity. This avoids voxel-boundary jumps becoming extreme false wall speeds in FLIP; trilinear mode remains the higher-quality choice for continuously velocity-coupled animated obstacles.

The Foam panel hides its FLIP-only whitewater controls—turbulence generation, energy/curvature/turbulence ranges, layer depth, and spray drag—whenever another solver method is active. Shared diffuse controls remain visible for methods that support them.

## Runtime Import

1. Parse and validate the format-6 through format-15 preset, inferring recycle sinks for format 6.
2. Decode embedded payloads, or resolve and validate the external GLB/static-SDF/animated-SDF resources supplied alongside the JSON.
3. Load the GLB from its `ArrayBuffer`, register all animation groups, and play every clip on a continuous loop.
4. Suspend the dashboard demo's presentation meshes, ground, and direct lights. Whiteboard keeps its selected environment cubemap as the background and leaves the single-sample/MSAA scene passes in skybox-covered no-clear mode; imports hosted by other demos hide the skybox and switch those passes to per-frame clear mode. Both paths redraw every pixel after camera motion, so stale color cannot remain.
5. Cancel the imported GLB root's synthetic X mirror, resolve each format-9 emitter's `sourceNode`, and capture its source-to-analytical-shape mapping. Apply source motion through this mapping after animation advances each frame, preserving the authored shape center and axes. Derived Source velocity updates without resetting emission budgets.
6. Pack the static grid and up to 16 animated local grids into one storage atlas plus one fixed-layout uniform buffer. Per-grid uniform flags select enabled/disabled and trilinear/nearest sampling. After glTF animation advances, update each linked node's current and previous world-to-local matrix without replacing buffers or recompiling solver pipelines. Every solver continues consuming one `SceneSdfSpec`; FLIP derives moving-solid normal velocity through its existing time-offset SDF samples. In Whiteboard's MLS-MPM backend, intersect free space with the invisible grid container so imported obstacles and inset numerical-boundary walls are both enforced.
7. Set each solver's safety floor to the imported grid's lower bound. Treat the imported grid position as the bundle's translation anchor: editing it translates the GLB presentation root, baked collision-SDF origin, analytical emitters, and sinks together without changing their grid-local layout or generating synthetic source velocity.
8. Restore grid, PBF physics, time scale, emitters, sinks, render, and foam settings.
9. For Whiteboard imports, restore the exported active Blender camera orbit, target, and vertical field of view. Presets without an authored camera retain the legacy Blender front view framed around the union of the simulation grid and every emitter/sink's transformed analytical shape; presentation-only meshes remain excluded from that fallback.
10. Reset and begin visible playback immediately. Volume-sampled Initial emitters are seeded on their deterministic rest-spacing lattice before the first rendered simulation frame.

Importing another authored scene removes the previous GLB animation groups and retires its GPU resources. Switching to a built-in demo removes the imported scene override and restores that demo's authored default camera instead of retaining the imported framing or a previously manipulated view. Format-5 JSON presets remain parameter-only.

Imports preserve authored particle capacities and Resolution Divisions without fixed safety substitutions. Before creating solver resources, the Fluid host validates each required allocation against the active WebGPU device and reports the requested and supported per-buffer memory when the preset does not fit.

Imported glTF animation, emitter-source transforms, animated collision transforms, and fluid stepping share the same simulation timestep in realtime, deterministic capture, and offline rendering. A slow interactive frame therefore advances the imported scene by the same capped simulation delta as FLIP instead of allowing presentation or collision animation to run ahead according to wall-clock time.

Resetting the fluid with `R` or **Reset simulation** also rewinds imported GLB animations so source meshes and the simulation restart together. Holding Shift while pressing `R` or clicking the button resets only the fluid and preserves the current mesh-animation time.

### Analytical emitter transforms and linked source nodes

An exported analytical shape does not necessarily use the source mesh's origin or axes.
`flow_shape_transform_for` recenters boxes/prisms, rotates prism axes, and reorders scale.
Consequently, `sourceNode` identifies motion, not a replacement for the emitter transform.
At binding creation, capture the source's world matrix `S0` and the authored shape's world
matrix `E0`, then store `localShape = inverse(S0) * E0`. Live source motion resolves to
`E(t) = S(t) * localShape`. This applies to both initial and inflow emitters and preserves
the exporter-authored shape center, rotation, and nonuniform scale during resets.

The Fluid host's `emitter-source-transform.ts` helper owns the binding state and math.
Creation rejects a singular source matrix rather than creating a misplaced emitter.
When the source returns exactly to `S0`, reuse a copy of the authored TRS rather than
round-tripping it through float32 matrix decomposition, so an unchanged analytical seed
retains its deterministic lattice. Source-velocity sampling uses the resolved shape center.
The mapping follows parent/grid placement through `S(t)` and does not alter rendering or
collision transforms. Exact baked seeds remain independent of analytical shape sampling.

Regression coverage includes a non-baked off-origin triangular prism with a different
source-node axis/scale basis, R and button resets, subsequent replay, moving-source offsets,
and mirrored source transforms. A seed-only check using baked particle arrays is insufficient.

## Validation

- Unit coverage includes format-15 animated collision parsing, unique node links, embedded/external re-export, collision headers, preset bounds, flow references, and lifecycle/time-scale bounds.
- The focused Chrome/WebGPU workflow imports a selected JSON/GLB/static-SDF/animated-SDF set, verifies the packed collision binding, plays the linked glTF animation, advances the collision transform after animation, and confirms cleanup on demo switch.
- Blender background export verifies animated-mesh listing, `0` collision override, rigid local SDF baking, hidden-node animation export, the 2048 control ceilings, optional lights, separate resource files, and decimation protection.
