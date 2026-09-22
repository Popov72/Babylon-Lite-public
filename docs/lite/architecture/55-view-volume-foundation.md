# Module: View-Volume Foundation

> Package path: `packages/babylon-lite/src/render/volume/`

## Purpose

Provide the shared, tree-shakable infrastructure required by camera-aligned volumetric effects without coupling their physical models or render passes.

The first consumers are:

- Wronski/Frostbite-style volumetric fog, which stores continuous participating-media properties, injected lighting, integrated in-scattering, and transmittance in a stable frustum-aligned grid.
- Adaptive Voxel-Based Order-Independent Transparency (AVBOIT), which captures discrete transparent-surface optical-depth events into atomic buffers and adaptively remaps virtual depth slices into a bounded physical grid.

The features share:

- camera-volume coordinate and depth-slice math;
- Beer-Lambert and optical-transfer helpers;
- 3D sampled/storage texture allocation;
- column-dispatch and debugging conventions;
- linear-HDR scene-color ownership;
- clustered-light and shadow resource access.

They do not share:

- one physical grid or depth mapping instance;
- extinction buffers;
- temporal histories;
- capture, injection, integration, or resolve pipelines;
- final composition semantics.

Fog needs stable world-space slice thickness for temporal reprojection and physically based medium integration. AVBOIT deliberately changes its depth mapping according to current transparent-surface occupancy. Combining their resources would make fog history unstable and AVBOIT less adaptive.

## Public API Surface

The foundation has no package-root public API in its first phase. Feature modules import it internally. No raw WebGPU handle is exposed through `@babylonjs/lite`.

### View-volume grid

```ts
export type ViewVolumeDepthMapping = { readonly kind: "linear" } | { readonly kind: "log" } | { readonly kind: "power"; readonly exponent: number };

export interface ViewVolumeGridOptions {
    readonly targetWidth: number;
    readonly targetHeight: number;
    readonly tileSize: number;
    readonly depthSlices: number;
    readonly nearDepth: number;
    readonly farDepth: number;
    readonly depthMapping: ViewVolumeDepthMapping;
}

export interface ViewVolumeGrid {
    readonly targetWidth: number;
    readonly targetHeight: number;
    readonly tileSize: number;
    readonly width: number;
    readonly height: number;
    readonly depth: number;
    readonly nearDepth: number;
    readonly farDepth: number;
    readonly depthMapping: ViewVolumeDepthMapping;
}

export interface ViewVolumeSliceBounds {
    readonly nearDepth: number;
    readonly farDepth: number;
    readonly centerDepth: number;
    readonly thickness: number;
}

export function createViewVolumeGrid(options: ViewVolumeGridOptions): ViewVolumeGrid;
export function viewDepthToVolumeSlice(grid: ViewVolumeGrid, viewDepth: number): number;
export function volumeSliceToViewDepth(grid: ViewVolumeGrid, sliceCoordinate: number): number;
export function getViewVolumeSliceBounds(grid: ViewVolumeGrid, sliceIndex: number): ViewVolumeSliceBounds;
export function viewVolumeTextureBytes(grid: ViewVolumeGrid, bytesPerTexel: number): number;
```

`viewDepthToVolumeSlice` returns a continuous coordinate in `[0, grid.depth]`. Integer slice `i` covers `[i, i + 1]`. Texture sampling uses the center coordinate `(i + 0.5) / grid.depth`.
`viewVolumeSliceThickness` returns a difference in **view-space axial Z**, not distance traveled along a perspective view ray. The WGSL functions require `nearDepth > 0`, `farDepth > nearDepth`, and `sliceCount > 0`, as established by `createViewVolumeGrid`; they must not substitute epsilon denominators that change valid narrow ranges.

### WGSL depth mapping

```ts
export function buildViewVolumeDepthWgsl(depthMapping: ViewVolumeDepthMapping): WgslSource;
```

The emitted source defines:

```wgsl
fn viewVolumeDepthToSlice(
    viewDepth: f32,
    nearDepth: f32,
    farDepth: f32,
    sliceCount: f32
) -> f32;

fn viewVolumeSliceToDepth(
    sliceCoordinate: f32,
    nearDepth: f32,
    farDepth: f32,
    sliceCount: f32
) -> f32;

fn viewVolumeSliceThickness(
    sliceIndex: u32,
    nearDepth: f32,
    farDepth: f32,
    sliceCount: u32
) -> f32;
```

The mapping is specialized when the shader string is composed. There is no runtime mapping-mode branch in froxel hot paths.

### Optical transfer

```ts
export type OpticalRgb = readonly [number, number, number];

export interface OpticalTransfer {
    readonly radiance: OpticalRgb;
    readonly transmittance: OpticalRgb;
}

export function alphaToOpticalDepth(alpha: number): number;
export function transmittanceFromOpticalDepth(opticalDepth: number): number;
export function composeOpticalTransfer(front: OpticalTransfer, back: OpticalTransfer): OpticalTransfer;
export function integrateHomogeneousMedium(source: OpticalRgb, extinction: OpticalRgb, distance: number): OpticalTransfer;
```

`OPTICAL_TRANSFER_WGSL` defines equivalent GPU functions and an `OpticalTransfer` struct.

### Internal 3D volume texture

```ts
export type VolumeTextureFormat = "rgba8unorm" | "rgba16float";

export interface VolumeTexture3DOptions {
    readonly label?: string;
    readonly width: number;
    readonly height: number;
    readonly depth: number;
    readonly format: VolumeTextureFormat;
    readonly filter?: GPUFilterMode;
}

export interface VolumeTexture3D {
    texture: GPUTexture | null;
    readonly view: GPUTextureView;
    readonly sampler: GPUSampler;
    readonly width: number;
    readonly height: number;
    readonly depth: number;
    readonly format: VolumeTextureFormat;
}

export function createVolumeTexture3D(engine: EngineContext, options: VolumeTexture3DOptions): VolumeTexture3D;
export function disposeVolumeTexture3D(volume: VolumeTexture3D): void;
```

This API remains internal because it exposes raw GPU handles. Public fog and transparency APIs will expose managed feature state or `Texture2D` facades instead.

## Internal Architecture

### Grid dimensions

For a target of `targetWidth x targetHeight` and square screen tile size `tileSize`:

```text
width  = ceil(targetWidth  / tileSize)
height = ceil(targetHeight / tileSize)
depth  = depthSlices
```

Dimensions, tile size, and depth count are positive integers. `nearDepth` must be finite and greater than zero. `farDepth` must be finite and greater than `nearDepth`.

### Depth mappings

#### Linear

```text
u = clamp((z - near) / (far - near), 0, 1)
slice = u * sliceCount
z = near + (far - near) * clamp(slice / sliceCount, 0, 1)
```

#### Logarithmic

```text
u = log(clamp(z, near, far) / near) / log(far / near)
slice = u * sliceCount
z = near * pow(far / near, clamp(slice / sliceCount, 0, 1))
```

This is equivalent to Babylon Lite's clustered-light Z convention while exposing the inverse mapping and physical slice thickness needed by volumetrics.

#### Power

The exponent describes the slice-boundary curve:

```text
z = near + (far - near) * pow(u, exponent)
u = pow(clamp((z - near) / (far - near), 0, 1), 1 / exponent)
slice = u * sliceCount
```

An exponent greater than one allocates thinner slices near the camera. The exponent must be finite and greater than zero.

AVBOIT's adaptive mapping is not part of this first module. Its feature module will wrap a fixed base mapping with a GPU-generated warp LUT while retaining the same forward/inverse contract.

### Optical transfer

For front transfer `A` and back transfer `B`:

```text
L = A.L + A.T * B.L
T = A.T * B.T
```

Composition is associative. It is the shared mathematical basis for front-to-back scans.

#### Surface alpha

AVBOIT converts surface opacity into a dimensionless optical-depth event:

```text
tau = -log(max(1 - alpha, epsilon))
```

It does not multiply this value by slice thickness.

#### Homogeneous participating medium

For extinction coefficient `sigmaT` (per unit ray length), source coefficient `S` (per unit ray length), and **view-ray** segment length `d`:

```text
T = exp(-sigmaT * d)
L = S * (1 - T) / sigmaT
```

For a fixed distance, the zero-extinction limit is `L = S * d`. The CPU implementation uses `-expm1(-sigmaT * d)` to preserve small positive optical depths. The WGSL implementation uses a cubic series for optical depths below `0.05` and evaluates the exponential form otherwise. Neither selects the zero-extinction limit from `sigmaT` alone: a low extinction coefficient over a long distance can still cause significant attenuation.

`getViewVolumeSliceBounds` and `viewVolumeSliceThickness` measure **axial** depth. A perspective fog integrator must convert each slice's axial thickness `deltaZ` to traveled ray length `d = deltaZ / abs(normalizedViewRay.z)`, where `normalizedViewRay` is the view-space direction through the froxel's XY center (positive Z in front of the camera). For an orthographic camera, use the forward view ray, giving `d = deltaZ`.

The homogeneous-medium formula must not be applied to AVBOIT surface events.

### Resource ownership

`createVolumeTexture3D` allocates:

- dimension: `"3d"`;
- usage: `TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC | COPY_DST`;
- view dimension: `"3d"`;
- clamp-to-edge sampling on all axes.

The helper validates dimensions against `device.limits.maxTextureDimension3D`. Allocation is unpublished until texture, view, and sampler creation have all succeeded. Disposal is idempotent.

No cache or GPU object is allocated at module initialization.

### Future render-phase coordinator

Volumetric fog and AVBOIT both need work between opaque rendering and ordinary transparency. They must not install nested one-off `RenderTask.execute` wrappers.

The planned internal execution phases are:

```text
prepare
opaque
afterOpaque
volumeCapture
volumeIntegration
opaqueVolumeComposite
transparentShading
transparentResolve
fallbackTransparency
overlay
```

Transmission, fog, and AVBOIT will register feature-owned phase callbacks through one task-owned coordinator. The coordinator remains absent from ordinary render tasks, preserving the existing hot path and bundle output for scenes that import none of the features.

The coordinator is intentionally not implemented until the first feature consumes it. The initial foundation introduces no branch into `render-task-base.ts`.

### Linear HDR ownership

Fog, AVBOIT, and transmission require one shared linear `rgba16float` scene color before image processing. The existing transmission-only retargeting logic will be extracted when the first second consumer lands.

Requirements:

- idempotent acquisition by multiple features;
- one offscreen HDR target;
- one trailing image-processing task;
- feature release must not destroy a target still requested by another feature;
- no mutation of the surface swapchain target descriptor;
- MSAA ownership and resolve behavior remain explicit.

## Pipeline Configuration

The foundation creates no render or compute pipeline.

Future consumers specialize the shared WGSL and create their own pipelines:

- fog media injection: compute, floating-point storage texture output;
- fog lighting injection: compute, clustered lights and shadow sampling;
- fog integration: one invocation per XY column, loop over Z;
- AVBOIT capture: raster fragment atomics into storage buffers;
- AVBOIT integration: compute, sparse surface optical-depth scan;
- AVBOIT shading: full-resolution material-owned MRT pipelines.

## Shader Logic

### Fog column scan

Reconstruct `normalizedViewRay` for the current XY column once from the active camera projection before scanning Z. For perspective cameras, reuse its reciprocal axial component in every slice; for orthographic cameras, the view ray is camera-forward and `invRayZ = 1`.

```wgsl
let invRayZ = 1.0 / abs(normalizedViewRay.z);
var accumulated = OpticalTransfer(vec3f(0.0), vec3f(1.0));
for (var z = 0u; z < grid.depth; z++) {
    let axialThickness = viewVolumeSliceThickness(z, nearDepth, farDepth, grid.depth);
    let distance = axialThickness * invRayZ;
    let segment = integrateHomogeneousMedium(source[z], extinction[z], distance);
    accumulated = composeOpticalTransfer(accumulated, segment);
    integrated[z] = vec4f(accumulated.radiance, accumulated.transmittance.x);
}
```

Fog may use RGB transmittance in a later extension. The first implementation stores scalar extinction/transmittance in alpha.

### AVBOIT column scan

```wgsl
var transmittance = vec3f(1.0);
for (var z = firstOccupied; z <= lastOccupied; z++) {
    transmittance *= exp(-surfaceOpticalDepth[z]);
    integrated[z] = transmittance;
    if (max(transmittance.r, max(transmittance.g, transmittance.b)) <= zeroThreshold) {
        recordZeroSlice(z);
        break;
    }
}
```

AVBOIT does not use fog's source-radiance integration.

## State Machine / Lifecycle

### Grid

```text
create options
  -> validate
  -> derive immutable grid dimensions
  -> use in CPU setup and shader composition
```

A target resize creates a new immutable grid value. Feature state compares dimensions and reallocates only when they changed.

### Volume texture

```text
create
  -> validate dimensions/format
  -> create GPUTexture
  -> create 3D view
  -> resolve sampler
  -> publish VolumeTexture3D

dispose
  -> destroy current texture once
  -> set texture = null
```

Device-loss recovery is owned by the eventual feature task. The low-level allocation helper does not register global recovery state.

## Babylon.js Equivalence Map

Babylon.js has no AVBOIT implementation. Its existing OIT path uses dual depth peeling and is a comparison oracle, not an implementation source.

Babylon.js's ordinary fog is analytic per-fragment fog and corresponds to Babylon Lite's current `scene.fog`. The volumetric-fog feature is a separate WebGPU-only renderer based on the Wronski/Frostbite model.

| Concept                            | Babylon Lite foundation                  |
| ---------------------------------- | ---------------------------------------- |
| Clustered logarithmic depth slices | `ViewVolumeDepthMapping { kind: "log" }` |
| Analytic material fog              | Existing `shader/wgsl-fog.ts`, unchanged |
| Wronski/Frostbite froxel grid      | Future volumetric-fog consumer           |
| Babylon.js dual depth peeling      | AVBOIT comparison/reference path         |

## Dependencies

- `engine/engine.ts` for device access in the internal volume allocator.
- `engine/gpu-flags.ts` for minifiable WebGPU usage aliases.
- `resource/texture-sampler-pool.ts` for sampler deduplication.
- `shader/wgsl.ts` for marked WGSL source.

The pure grid and optical-math modules have no engine, scene, camera, material, or GPU dependencies.

## Test Specification

### Grid math

- Invalid dimensions, depth range, tile size, slice count, or power exponent throw.
- Derived XY dimensions use ceiling division.
- Linear, logarithmic, and power mappings round-trip at near, far, boundaries, and interior depths.
- CPU and generated WGSL depth mappings clamp depths outside the configured near/far range, including logarithmic mapping; validated narrow ranges must preserve exact endpoints.
- Slice bounds are monotonic, positive-thickness, and cover the complete configured range.
- Power exponent greater than one produces a thinner first slice than linear mapping.
- CPU and generated WGSL use equivalent formulas.
- Texture-byte calculation detects unsafe integer overflow.

### Numerical WGSL checks

`tests/lite/plumbing/view-volume-wgsl.spec.ts` compiles the generated WGSL helpers into a WebGPU compute shader and reads back actual GPU results. Compare all three depth mappings against CPU results for near/far endpoints, values outside the grid, narrow near/far ranges, inverse mapping and slice thickness. Compare RGB optical radiance and transmittance against the CPU integration for long low-extinction segments, tiny positive optical depths, zero extinction, and samples on either side of the `0.05` WGSL approximation threshold. Use explicit float32 error bounds; a missing WebGPU adapter is a test failure rather than a silent skip. The PR `UnitTests` job runs this spec in its focused Playwright plumbing step.

### Optical math

- Alpha zero produces zero optical depth.
- Alpha 0.5 produces `ln(2)`.
- Optical-depth/transmittance round-trip is stable.
- Transfer composition is associative within floating-point tolerance.
- Zero-extinction homogeneous media use the `source * distance` limit.
- Subdividing a long, low-extinction medium does not change its integrated radiance; small positive optical depth remains numerically stable on CPU and WGSL.
- Positive extinction produces bounded transmittance and finite radiance.

### Volume texture

- Rejects non-positive or over-limit dimensions.
- Creates a 3D texture with sampled, storage, and copy usage.
- Creates a 3D view and clamp-to-edge sampler.
- Disposal destroys exactly once.
- A failed view creation destroys the unpublished texture.

### Feature integration

Added with the first consumer:

- volumetric fog samples the same CPU/WGSL depth boundaries;
- an off-axis homogeneous fog column at 60 degrees from the view axis uses twice the axial slice thickness and therefore has `exp(-2 * sigmaT * deltaZ)` transmittance;
- AVBOIT's adaptive warp preserves the base grid's near/far endpoints;
- scenes that import neither feature remain byte-identical.

## File Manifest

```text
docs/lite/architecture/55-view-volume-foundation.md
packages/babylon-lite/src/render/volume/view-volume-grid.ts
packages/babylon-lite/src/render/volume/view-volume-grid-wgsl.ts
packages/babylon-lite/src/render/volume/optical-transfer.ts
packages/babylon-lite/src/render/volume/optical-transfer-wgsl.ts
packages/babylon-lite/src/render/volume/volume-texture.ts
tests/lite/unit/view-volume-foundation.test.ts
tests/lite/plumbing/view-volume-wgsl.spec.ts
```
