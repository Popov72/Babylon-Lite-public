# Module: Blender Fluid Bundle

> Package paths: `scripts/blender-fluid-addon.py`, `lab/lite/src/demos/fluid/blitefluid-bundle.ts`

## Purpose

Export an authored Blender liquid setup as one portable `.blitefluid` file and import it into the fluid demo as a live Babylon Lite simulation. The bundle carries the simulation domain, solver/presentation preset, emitters, sinks, static visual meshes, and a baked three-dimensional signed-distance field for collision.

The bundle is not a baked animation. Babylon Lite recreates and advances the simulation with its selected PBF, MLS-MPM, or PB-MPM solver.

## Public API Surface

```ts
export interface BliteFluidBundle {
    manifest: BliteFluidManifest;
    sceneGlb: ArrayBuffer;
    collision: BliteFluidCollision;
}

export interface BliteFluidManifest {
    bundleVersion: 1;
    preset: FluidExportJson;
    scene: {
        glb: "scene.glb";
        collision: "collision.blsdf";
    };
}

export interface BliteFluidCollision {
    dims: [number, number, number];
    origin: [number, number, number];
    cellSize: number;
    distances: Float32Array;
}

export function parseBliteFluidBundle(data: ArrayBuffer): BliteFluidBundle;
```

The Blender add-on registers:

- `BLITEFLUID_PT_export`: Scene-properties panel.
- `BLITEFLUID_OT_export`: file-browser export operator for `.blitefluid`.
- `BLITEFLUID_OT_validate`: setup validation without writing a bundle.
- `BLITEFLUID_OT_set_role`: selected-mesh role assignment for liquid domain, initial volume, inflow, sink, or collider.

The export operator is available both in the panel and under **File > Export > Babylon Lite Fluid (.blitefluid)**.

## Internal Architecture

### Container

`.blitefluid` is an uncompressed ZIP archive (`ZIP_STORED`) with exactly:

```text
manifest.json
scene.glb
collision.blsdf
```

The browser parser reads local ZIP file records directly. Compressed or encrypted entries are rejected explicitly; no third-party decompressor is bundled.

### Manifest

`manifest.json` wraps a complete format-5 `FluidExportJson`. Emitter and sink transforms are grid-local. `gridPosition` and `gridSize` are authoritative world-space values. Older parameter JSON files remain supported by the existing importer.

### Scene

`scene.glb` contains visible Blender mesh objects except:

- the fluid domain object;
- fluid flow objects;
- hidden or disabled objects.

Fluid effectors and objects with `blite_collision = true` are included as visible scene meshes and in the collision bake. Additional visible meshes are exported for presentation but do not affect collision unless marked.

### Blender Authoring

The Scene-properties panel is the add-on's complete authoring surface:

- It reports the detected liquid domain's world-space center, size, collision-grid dimensions, voxel count, and uncompressed SDF size.
- Role buttons configure the selected mesh with the appropriate Blender Fluid modifier and liquid behavior.
- The selected PBF, MLS-MPM, or PB-MPM method exposes the same physics defaults and ranges as the fluid demo; each method keeps independent Blender scene properties and exports a complete physics block.
- Flow controls expose the Babylon Lite analytical shape, volume rate, spread, annular/top radii, polygon points, and sink targets.
- Any visible mesh can opt into static collision without requiring a Mantaflow effector by enabling **Babylon Lite collider**.

The properties use the existing `blite_*` custom-property names, so scenes authored for the standalone exporter remain compatible and can be edited through the add-on UI.

Validation runs before export and is also available independently. It rejects missing/degenerate domains, unsupported shape data, object-count overflow, and missing presentation geometry. It warns about missing emitters/colliders, flows outside the domain, unresolved sink target names, and non-manifold collision geometry. Warnings do not block export because open meshes can be intentional, but the SDF sign is reliable only for closed consistently oriented surfaces.

### Collision Grid

`collision.blsdf` is little-endian:

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

Distances are negative inside collision solids and positive outside. Blender evaluates modifiers before baking, combines all collision triangles into one BVH, and signs the nearest-surface distance from the nearest outward normal. Collision meshes should therefore be closed and consistently oriented.

The longest grid axis uses the add-on's requested SDF resolution. Other axes use the same cell size:

```text
cellSize = max(gridSize) / (resolution - 1)
dims = ceil(gridSize / cellSize) + 1
origin = gridPosition - gridSize / 2
```

The simulation domain is derived from the authored domain cage mesh, not the modifier-evaluated object bounds: Blender evaluates a liquid domain as its current liquid surface, which would otherwise collapse the exported grid around the inflow.

## Pipeline Configuration

The imported GLB is loaded from its raw `ArrayBuffer` and added to the already-running scene. Importing another bundle removes the previous asset container.

The collision data creates:

- one storage buffer containing `distances`;
- one 32-byte uniform buffer containing `(origin.xyz, 1 / cellSize)` and `(dims.xyz, 0)`;
- one `SceneSdfSpec` using the shared `sampleSdfGrid` WGSL helper.

The imported SDF overrides the active demo's scene SDF. Demo-owned meshes are hidden, demo updates/force fields are suspended, and the solver continues using the imported preset's grid, emitters, and sinks. Switching to another built-in demo clears the imported scene override.

GPU buffers are retired through the engine's deferred retirement queue after the frame that can reference them has submitted.

## Shader Logic

```wgsl
struct SceneSdfParams {
    grid: vec4<f32>, // origin.xyz, inverse cell size
    dims: vec4<f32>, // dimensions.xyz
};

fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    return sampleSdfGrid(
        pt,
        sceneSdfParams.grid.xyz,
        sceneSdfParams.grid.w,
        vec3<i32>(sceneSdfParams.dims.xyz)
    );
}
```

The SDF is static; `dt` is ignored.

## State Machine / Lifecycle

1. User chooses **Import preset / Blender bundle**.
2. Existing imported asset and collision resources are detached.
3. ZIP records and binary headers are validated.
4. GLB bytes are loaded and added to the scene.
5. The manifest method/preset is restored.
6. The imported SDF is installed on all three solvers.
7. The simulation resets at full opacity and runs normally.
8. Importing another bundle repeats steps 2–7.
9. Switching built-in demos clears the imported override.

Failures are reported to the console and do not install a partially parsed bundle.

## Babylon.js Equivalence Map

- Blender Mantaflow domain → Babylon Lite world-space grid.
- Blender liquid `FLOW` modifiers → `FluidEmitter`/`FluidSink`.
- Blender liquid `EFFECTOR` modifiers → baked collision SDF.
- Blender glTF export → Babylon Lite `loadGltf(ArrayBuffer)`.
- Blender cache playback has no equivalent here because the target remains a live simulation.

## Dependencies

- Blender Python: `bpy`, `bmesh`, `mathutils.bvhtree`, `zipfile`.
- Babylon Lite: `loadGltf`, `addToScene`, `removeFromScene`, `SceneSdfSpec`, deferred GPU retirement.
- Existing fluid preset import and flow schema.

## Test Specification

1. Parse a valid stored ZIP and reject compressed, encrypted, missing, truncated, or duplicate entries.
2. Parse collision headers and reject invalid magic, version, dimensions, lengths, or cell sizes.
3. Round-trip a synthetic manifest/GLB/collision bundle.
4. Import in Chrome/WebGPU and verify:
    - the GLB appears;
    - the manifest grid/method/flow values restore;
    - particles collide with the baked SDF;
    - re-import and demo switching remove the prior asset;
    - no destroyed-buffer validation errors occur.
5. Run Blender background export on a minimal domain/flow/effector scene when Blender is available.

## File Manifest

- `scripts/blender-fluid-addon.py` — installable Blender add-on and exporter.
- `lab/lite/src/demos/fluid/blitefluid-bundle.ts` — ZIP and collision parser.
- `lab/lite/src/demos/fluid.ts` — import UI and runtime installation.
- `config/fluid-flow.schema.json` — bundle metadata fields remain additional format-5 properties.
- `tests/lite/unit/fluid-blitefluid-bundle.test.ts` — binary parser tests.
