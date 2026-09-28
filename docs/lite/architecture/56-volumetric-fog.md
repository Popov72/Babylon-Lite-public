# Module: Volumetric Fog

> Package paths: `packages/babylon-lite/src/volumetric-fog/`, `packages/babylon-lite/src/frame-graph/linear-scene-color.ts`
>
> References: [Wronski, SIGGRAPH 2014](https://bartwronski.com/wp-content/uploads/2014/08/bwronski_volumetric_fog_siggraph2014.pdf), [Hillaire, SIGGRAPH 2015](https://advances.realtimerendering.com/s2015/index.html), [view-volume foundation](55-view-volume-foundation.md).

## Purpose

Add opt-in, WebGPU-only participating-media fog. Unlike Babylon Lite's existing analytic `scene.fog`, density varies over three-dimensional space and incoming light is scattered toward the viewer. The core follows [Wronski's slide 24](https://bartwronski.com/wp-content/uploads/2014/08/bwronski_volumetric_fog_siggraph2014.pdf#page=24): evaluate density and lighting at frustum-aligned froxels, integrate depth columns, then apply on screen. V1 evaluates density and lighting in the same compute shader, writing one injection volume before raymarching it into a second; the slide allows density and lighting in the same or separate passes. The core supplies homogeneous/height density and ambient plus one unshadowed directional light; Scene 316 alone supplies its bounded, animated cloud density model through the generic density-source seam. Later increments add multiple local volumes, volumetric shadows, jitter/history and transparent surfaces.

Fog and AVBOIT have separate grids and integration passes. They share the view-volume math, optical transfer, linear scene color and render-phase ownership. In particular, AVBOIT must not change the stable Z mapping of fog's grid or re-use its history.

The numbered visual fixture is Scene 316 (`lab/lite/scene316.html`). It shows a bright, billowing fog bank above a dark floor, with an irregular upper silhouette and fine-scale internal density changes; the cubes remain legible above the layer. Viewers can left-drag to orbit, right-drag to pan, scroll/pinch to zoom and click **Reset view**; these pointer controls install no keyboard bindings. **Base extinction remains constant** while temporally continuous world-anchored noise changes the bank's upper silhouette and internal density in place: the lab shader has no wind advection, so the cloud cannot slide sideways. `?seekTime=N` freezes the noise at N seconds for deterministic captures without disabling manual camera movement; the animation button reads **Animation frozen** in this mode. `?density=0` still yields the unfogged image. The debug-only `?transmission` and `?recovery` branches load their exported enablers from separate lazy lab helpers: default viewing never fetches those unrelated runtimes. It is Lite-native, so it skips Babylon.js screenshot parity; the WebGPU plumbing spec checks the actual shaders, image and device recovery instead.

Scene 316 also exposes **Cloud shape: On/Off** and **3D texture size** controls. The shape button freezes the lab-owned cloud model's `time` at its current value; density is unchanged. Resuming continues from the frozen noise phase without catching up. `?seekTime=N` takes precedence and disables the animation button. The pause state uses `?animate=off` (updated without a reload) so selecting a texture size retains it; existing `?scroll=off` URLs still pause the morph but are rewritten to `?animate=off` when the button is used. The 3D noise itself is procedural and has **no** noise texture to resize: the size selector configures both `rgba16float` 3D froxel textures, each displayed as `ceil(canvasWidth/tileSize) × ceil(canvasHeight/tileSize) × depthSlices`. Presets are low (16-pixel tile, 64 slices), standard (8, 80; default), depth detail (8, 160) and XY detail (4, 80). Changing presets sets `?volumeSize=low|standard|depth|fine` and reloads the current scene URL, preserving other parameters. Invalid names fail explicitly; a canvas resize updates the displayed dimensions and rebuilds the grid through the scene's existing resize path. The no-fog `?baseline=1` variant hides both fog-specific controls.

## Public API Surface

```ts
export interface VolumetricFogOptions {
    /** Fog extinction per world unit at `height`. Strictly nonnegative. */
    density: number;
    /** World-space height at which density equals `density`. */
    height?: number; // default 0
    /** Density multiplier per unit of height: exp(-heightFalloff * (worldY - height)). */
    heightFalloff?: number; // default 0; nonnegative
    /** Upper bound for local density (avoids numerical overflow below the height plane). */
    maxDensity?: number; // default 1; positive
    /** Per-channel single-scattering albedo, in [0,1]. */
    albedo?: readonly [number, number, number]; // default [1,1,1]
    /** Constant incoming ambient radiance in linear units. */
    ambientRadiance?: readonly [number, number, number]; // default [0.1,0.1,0.1]
    /** Directional light for homogeneous anisotropic inscattering. */
    sun?: DirectionalLight | null; // default null
    /** Henyey-Greenstein asymmetry; in [-0.95, 0.95]. */
    anisotropy?: number; // default 0
    /** Pixel width/height per froxel; default 16, positive integer. */
    tileSize?: number;
    /** Z slice count, default 64, positive integer. */
    depthSlices?: number;
    /** View distance at the last slice; default min(camera.farPlane, 100). */
    maxDistance?: number;
    /** Optional spatial density model; without it, use homogeneous/height fog. */
    densitySource?: VolumetricFogDensitySource;
}

export interface VolumetricFogDensitySource {
    /** WGSL defines fn fogDensityAt(worldPosition: vec3f) -> f32. */
    readonly shader: string;
    /** Optional binding 2 uniform size in floats, divisible by four. */
    readonly uniformFloatCount?: number;
    /** Validate caller-owned mutable inputs before an enabled frame. */
    validate?(): void;
    /** Fill every float of the reusable binding 2 uniform before an enabled frame. */
    writeUniforms?(data: Float32Array): void;
}

export interface VolumetricFog {
    /** Toggle integration and compositing without reallocating the grid. */
    enabled: boolean;
    density: number;
    height: number;
    heightFalloff: number;
    maxDensity: number;
    albedo: readonly [number, number, number];
    ambientRadiance: readonly [number, number, number];
    sun: DirectionalLight | null;
    anisotropy: number;
}

export function enableVolumetricFog(scene: SceneContext, engine: EngineContext, options: VolumetricFogOptions): VolumetricFog;
export function disposeVolumetricFog(fog: VolumetricFog): void;
```

The initial API is deliberately restricted to full-viewport perspective cameras on an unregistered scene's single default render task. Floating origin, analytic `scene.fog`, an enabled nonstandard tone mapper, and shadowed directional lights are not supported and fail explicitly. **Only opaque PBR renderables with the linear-output material variant are supported for the entire lifetime of a fog-owned HDR target**, including after fog is disabled or disposed. Standard/Node/Shader materials, hosted billboards/text and transparent PBR surfaces are rejected rather than silently changing their appearance. The retained final image-processing pass runs after blending; admitting transparent PBR while fog is off would change its ordinary alpha-blended output even though no volumetric pass runs. The initial reference/demo therefore uses only PBR meshes and a plain clear color. Initial material families are preflighted before retargeting; transparent bindings and later material changes fail after binding refresh but before issuing a GPU draw.

Fog-owned HDR presentation cannot compose two SceneContexts registered on the same surface. Enabling fog with an already-registered scene fails before changing targets; if a fog-enabled scene was prepared first, the later scene's lazy overlay-configuration path rejects the registration instead of silently overwriting or disabling an overlay. A transmission-only HDR base followed by an ordinary single-sample overlay remains supported: its final present writes the swapchain before the overlay loads it. Existing MSAA transmission-overlay limitations are unchanged.

The initial output has no temporal history. Camera changes or animated fog fields take effect in the next frame. `density`, `height`, `heightFalloff`, `maxDensity`, `albedo`, `ambientRadiance`, `sun`, `anisotropy`, and caller-owned density-source parameters may be changed directly. The density source's WGSL and uniform size are fixed at installation; `tileSize`, `depthSlices`, and `maxDistance` are likewise fixed. A custom shader can call the core's `fogBaseExtinction()` and `fogBaseDensity(worldPosition)` (homogeneous/height fog), and must define `fogDensityAt(worldPosition)`; the core clamps the resulting extinction to `[0,maxDensity]` and multiplies it by `albedo` for per-channel scattering. If a source supplies uniforms, its WGSL owns the `@group(0) @binding(2)` declaration, `uniformFloatCount` is a nonnegative multiple of four, and `writeUniforms` fills every float each enabled frame. `registerScene()` awaits WGSL compilation and a pipeline whose declared binding-2 minimum size matches that uniform; malformed shaders or incompatible layouts reject registration before GPU fog state is published. Invalid non-finite, negative, incomplete or device-unsupported values also fail explicitly. Disposing fog frees its GPU state but does not make the same scene eligible for a second installation: the render task keeps its installed lifecycle wrappers and linear output path. The supported sun source has no volumetric shadowing/light shafts yet.

## Internal Architecture

### Shared linear scene color

`frame-graph/linear-scene-color.ts` owns the existing transmission-only HDR retargeting operation and makes it idempotent for fog and transmission:

1. Find the scene's default surface render task. Reject an unrecognized/custom task rather than retargeting a caller-owned RTT.
2. Replace a surface-targeted color buffer with an owned `rgba16float` offscreen color RT sized to the original render target, retaining the existing depth format, depth comparison and sample count. Keep borrowed external depth separate; the scene render task still owns its internal depth.
3. Disable in-material image processing for participating PBR variants via an opt-in material fragment, while preserving the scene's configured image processing for the trailing pass. The PBR template supplies dedicated `IP_BEGIN`/`IP_END` anchors **after** `NI` and analytic-fog composition and **around only** tonemap/exposure, gamma, clamping and contrast. Without a linear-output fragment these generic anchors compose to empty WGSL; no registration-time global state or cross-chunk template dependency is required. No-IBL unlit/lightmap/etc. slots run before this bracket; later `BC` extensions run after it. The HDR sentinel gates only image processing, not material color assignments.
4. Reject an active nonstandard tone mapper before entering the render task, including while fog is toggled off or disposed: the shared final present implements only standard/disabled tone mapping. Then run execution under the temporary linear-image-processing state and inverse-processed clear color, restoring the scene state in `finally`.
5. Append **one** image-processing task after all consumers that modify the linear target, targeting the scene's bound surface (which can differ from the engine's primary surface). Enabling transmission and fog in either order must neither allocate a second linear target nor apply image processing twice.

The enabler marks existing PBR materials and, after HDR ownership commits, installs an opt-in scene material-change hook to mark meshes added or re-materialed later, before their PBR renderables are built. Failed transactional transmission setup must not leave this hook installed. It remains installed after fog disposal because the linear target and final present remain active until scene disposal. The hook bumps a scene-local material revision; the fog-owned HDR task checks the transparent-binding count and caches compatible PBR renderables by renderable-array identity, published source version, material epoch and that revision. Both opt-in split-pass paths validate this invariant **after automatic binding refresh but before beginning a GPU pass**, regardless of fog.enabled. Active fog additionally validates its directly mutable medium/camera/light state each frame. There is no O(mesh count) scan in steady frames with unchanged renderables/materials. This extracts shared ownership but leaves refraction snapshot and mip generation in the transmission feature. Features not enabled load neither fog nor shared linear helpers.

### Render phases

The scene pass keeps the opaque bundle and direct draw semantics. A task-owned opt-in `afterOpaque` hook runs after opaque/direct bindings and before transparent bindings:

```text
shadow tasks -> opaque/direct draw -> end pass
    -> fog inject/integrate -> fog composite (color load + blend; depth sampled while unbound)
    -> image processing / present
```

Fog-owned HDR rejects all transparent bindings even when fog is disabled or disposed. Active fog therefore does **not** resume an empty scene pass after compositing, whether the fog-only or transmission-plus-fog executor owns the split. When fog is disabled/disposed, the installed split executor still validates HDR-compatible opaque PBR materials and draws them in a single scene pass. Transmission's split-pass path invokes the same hook at the same point. Its HDR target and image-processing pass remain shared in either enable order, but V1 fog deliberately rejects actual transmissive surfaces: a later material fog-sampling variant is required before such a draw can be correct. Each new frame restores the task's initial depth-load policy (owned depth clears unless `depthClear` is false; eager external depth loads), even after a previous pass resumed with depth load. The ordinary single-pass path for tasks without either opt-in feature remains unchanged.

With MSAA the composite samples `texture_depth_multisampled_2d` at `sample_index` and renders into the multisampled color target at the same sample count. For single-sample depth it samples `texture_depth_2d`. The composite has **no depth attachment**, so sampling the depth texture does not create a WebGPU attachment feedback loop.

### Grid

Use `createViewVolumeGrid` with `tileSize=16`, `depthSlices=64`, `nearDepth=camera.nearPlane`, `farDepth=min(camera.farPlane,maxDistance)` and `depthMapping={kind:"power",exponent:2}`. The grid is frustum-aligned and stable with respect to translucent object occupancy.

Per froxel `(x,y,z)`, evaluate at the screen-tile center and view-Z slice midpoint. For perspective projection, recover a normalized camera ray from camera projection, transform it by camera world orientation, and obtain the world-space center at `viewZ / rayView.z`. For every XY column, true traveled slice length is `deltaViewZ / abs(rayView.z)`. Never use axial thickness alone off-axis.

GPU volume data:

| Resource   | Format                          | Content                                                      |
| ---------- | ------------------------------- | ------------------------------------------------------------ |
| Injected   | `rgba16float` 3D storage/sample | Scattered-light source `sigmaS * Li` in RGB, extinction in A |
| Integrated | `rgba16float` 3D storage/sample | Accumulated in-scattering RGB, scalar transmittance in A     |
| Parameters | Persistent uniform buffer       | Camera/grid, medium/lighting, dispatch dimensions            |

At 1080p, 16x16 screen tiles and 64 slices require approximately 4 MiB per `rgba16float` volume; two volumes total about 8 MiB. Validate all three dimensions against `maxTextureDimension3D` and the combined 16 bytes/froxel against the safe-integer range before allocation. A custom density source can have a separate, fixed-size uniform buffer; the core's 48-float uniform layout never changes.

### Fused density and lighting injection

The density function of the per-froxel injection shader evaluates a height-varying medium at each froxel center:

```text
sigmaT = min(maxDensity, density * exp(clamp(-heightFalloff * (worldY - height), -80, 80)))
sigmaS = sigmaT * albedo
```

The shader retains separate `fogDensityAt(worldPosition)` and `fogIncidentLighting(worldRay)` functions. A caller-provided WGSL density model may replace the former while sharing the core's world-position reconstruction, camera-aligned grid, uniform base-density helper and clamping. The core allocates one extra binding-2 uniform only if that model requests it, reuses its staging array, and validates all written floats before the GPU submission. Shader text and buffer size are fixed at installation; caller-owned `validate()` protects mutable model inputs. Asynchronous, device-scoped shader/pipeline validation runs before any fog-volume allocation, not per frame; a failed preload can be retried on the same device and recovery prepares again on the replacement device. No cloud-shaped density law or animation code exists in the library.

For nonempty froxels, `fogIncidentLighting` evaluates ambient plus directional radiance:

```text
cosTheta = dot(sunLightTravelDirection, directionFromFroxelToCamera)
phaseHG = (1 - g*g) / (4*pi * (1 + g*g - 2*g*cosTheta)^(3/2))
Li = ambientRadiance + sun.diffuse * sun.intensity * phaseHG
injected = (sigmaS * Li, sigmaT)
```

Compute uses the sun's world-transformed direction. No sun means ambient only. Zero-extinction froxels store zero without running lighting. Shadow-map visibility will attenuate the directional term **inside this lighting function** when shadow injection is added; V1 explicitly rejects a shadowed sun rather than pretending unshadowed fog has light shafts. Density and lighting remain separate shader responsibilities, but the injection pass shares the reconstructed ray and writes only one volume.

### Scene 316 density model (lab only)

`lab/lite/src/shared/scene316-cloud-density.ts` supplies cloud-bank WGSL to the generic density-source seam. The fixed bank dimensions, footprint, noise scale and temporal speed are lab shader constants, **not library API or uniforms**. Only animation time varies: an independent binding-2 `vec4f` uniform holds it in X (four floats total), and the lab source rejects non-finite time before installation and every enabled frame. This keeps the fixture small while preserving its UI, URLs and frozen images; any future tunable density model can instead supply its own uniforms through the same core seam. Zero base extinction remains exactly zero even with noise.

Use deterministic continuous 3D value noise: hash the eight integer lattice corners with a u32 avalanche hash, and trilinearly interpolate them using cubic Hermite weights `f*f*(3-2*f)`. Compute a low-frequency sample on `(worldX, top, worldZ)` so the upper edge moves coherently in XZ, and a higher-frequency 3D sample at the actual froxel position to break up internal density and the lower edge. Sample at world-space positions without advection; the footprint stays fixed, so a bank never moves outside its declared bounds.

At fixed `phase = time * 0.5`, add a fourth **temporal lattice coordinate** to the corner hash. Evaluate the same spatial lattice twice with integer time seeds `floor(phase)` and `floor(phase)+1`, and interpolate with the cubic Hermite weight of `fract(phase)`. Time zero reproduces the original 3D hash exactly. This produces C1-continuous morphing at fixed spatial coordinates, **not** coordinate translation. No noise texture, per-frame pipeline rebuild or screen-space temporal accumulation is needed. Skip the noise work entirely for froxels outside the widest possible vertical extent **or** XZ ellipse. The NodeMaterial `CloudBlock`'s six-octave emitter is not used in this per-froxel compute pass.

```text
temporalNoise3(p, phase) = mix(noise3(p, floor(phase)),
                                noise3(p, floor(phase) + 1),
                                smoothstep(0, 1, fract(phase)))
coarse = temporalNoise3((worldX, top, worldZ) * noiseScale, time*0.5)
fine = temporalNoise3(worldPosition * noiseScale * 2.19 + fixedOffset, time*0.5)
localTop = top + edgeVariation * (2*coarse - 1)
localBottom = bottom + 0.2*edgeVariation * (2*fine - 1)
ellipse = length((worldPosition.xz - footprint.center) / footprint.radius)
rim = 1 - smoothstep(1 - footprint.feather, 1, ellipse)
upper = 1 - smoothstep(localTop - feather, localTop, worldY)
mask = smoothstep(localBottom, localBottom + feather, worldY) * upper * upper * rim
variation = mix(1, 2*smoothstep(0.35, 0.72, 0.65*coarse + 0.35*fine), densityVariation)
sigmaT = min(maxDensity, density * exp(clamp(-heightFalloff*(worldY-height), -80, 80)) * mask * variation)
```

Both height and footprint masks are in `[0,1]` and `variation` is nonnegative. Squaring `upper` preserves tenuous grey wisps ahead of the opaque body; the thresholded coarse/fine noise makes gaps and clumps, not just a small brightness modulation. Even if strongly displaced edges locally cross, the product cannot create density outside the bank's union or above `maxDensity`. The finite footprint stops the camera from sitting inside an infinite white sheet: foreground ground rays encounter no fog until they enter the bank. The default core density model has none of these cloud-specific branches.

### Front-to-back integration

One compute invocation scans all Z slices of an XY column. For each slice, read injected `sourceRGB = sigmaS * Li` and extinction `sigmaT`; compute true view-ray length and call the shared `integrateHomogeneousMedium` WGSL helper. Compose the segment with the accumulated optical transfer and store `(L.rgb, T)` at the **far boundary** of each Z slice.

Samples at fractional slice `k+f` interpolate integrated values at the adjacent _boundaries_. For the first slice, the previous boundary is the identity `(L=0,T=1)`, not the already-integrated first voxel. This prevents spurious near-plane fog. At depth beyond the fog volume, use its final slice. A clear reverse-Z depth pixel samples the farthest fog slice so the background has the configured finite-distance fog.

### Post-opaque composite

At every covered output sample, reconstruct view-space Z from the opaque depth and lookup `(L,T)` in the integrated volume. Draw a fullscreen triangle into the existing scene color target with additive/attenuating blending:

```text
sourceRGB = L
sourceAlpha = T
blendRGB = sourceRGB + existingRGB * sourceAlpha
```

The corresponding `GPUBlendState` is `color: {srcFactor:"one",dstFactor:"src-alpha"}`; alpha preserves the existing destination alpha (`srcFactor:"zero",dstFactor:"one"`). The computed output is linear; image processing follows only once in a trailing task.

### Transparent surfaces

This initial increment **fails explicitly** for unsupported transparent renderables throughout the retained fog-owned HDR lifetime. A later material-owned extension will sample the integrated volume at each transparent fragment's own depth and apply `C_fogged(z)=L(z)+T(z)*C_surface` before ordinary back-to-front blending. Supporting transparent rendering consistently while fog is off also requires accounting for blending before the shared nonlinear image-processing present. This is valid for correctly sorted alpha transparency, but cannot simply be reused by AVBOIT's weighted resolve: fog in-scattering behind surface extinction needs a dedicated combined resolve. Transmission may coexist only once its transparent PBR surface acquires this sampler variant.

### Lifecycle

- `enable`: validate camera, scene fog mode, task topology and options; install shared HDR ownership, the after-opaque phase and one final present task. Must be called before `registerScene`.
- `preload`: on registration and device recovery, await compilation diagnostics and asynchronous creation of the injection pipeline using the current device and the density-source uniform's minimum binding size. Reject invalid caller WGSL/layouts before frame-graph build; cache only a successfully validated pipeline for that device.
- `record`: synchronously allocate/rescale the two 3D textures and any caller-owned density uniform for the task target; build the integration/composite pipelines and bind groups against the current device, depth texture, sample count and prepared injection pipeline. A failed generation must not replace working state; destroy unpublished textures/buffers on error.
- `execute`: validate the scene's active tone mapper before temporarily disabling per-material image processing. After bindings synchronize, assert task-level HDR material compatibility and opaque bindings even if fog is off; when fog is on, read/validate camera, finite sun direction components, medium state and caller density-source inputs. Draw opaque geometry; upload reusable uniforms, inject density and lighting together (skipping lighting for empty cells), scan each depth column into the integrated volume, and composite using the unbound opaque depth. Do not resume an empty scene render pass in V1.
- `resize`: rebuild grid and volumes in frame-graph task order, rebind inputs; the following frame uses the new target sizes without stale views.
- `disable`: skip injection/integration/composite, keeping the last textures for reuse; linear PBR output, the task-level HDR guard and the final image-processing task remain active, so supported opaque materials stay visually equivalent without a GPU-resource rebind.
- `dispose`: detach the after-opaque hook and free fog-owned GPU textures and buffers once. The scene-owned linear color target, PBR shader variant, HDR guard and final present stay active until scene disposal; their removal would require a material/target rebuild, and transmission may still use them.
- `device loss`: scene recovery awaits task preloads on the replacement device before rebuilding its frame graph. Pipeline preparation errors fail recovery rather than publishing stale/invalid fog; a successful rebuild recreates volumes and bind groups on the new device. No old-device view is sampled.

## Pipeline Configuration

- **Injection:** WGSL compute with an 8x8x1 dispatch (or another validated workgroup shape); evaluate the density model, skip lighting if extinction is zero, otherwise multiply scattering by incident radiance. Store `(sourceRGB,sigmaT)` in one `texture_storage_3d<rgba16float,write>`. Optional density-model uniforms are bound separately; camera/grid and lighting uniforms remain shared.
- **Integration:** WGSL compute, one invocation per XY column; sample the injected source/extinction volume and integrate true ray length into storage 3D `rgba16float`. Reuse `opticalTransferWgsl()` and `buildViewVolumeDepthWgsl({kind:"power",exponent:2})`.
- **Composite:** Fullscreen triangle with no depth attachment; a depth texture bound as read-only plus filterable integrated 3D texture and sampler. Separate modules for `texture_depth_2d` and `texture_depth_multisampled_2d` (sample index for the latter). Color attachment is the linear `rgba16float` scene buffer at sampleCount 1 or 4, with color blend `(one,src-alpha)` and alpha blend `(zero,one)`.

## Shader Logic

See the [view-volume foundation](55-view-volume-foundation.md#shader-logic) for the optical transfer and grid mapping formulas. `sun.direction` is the **travel direction** of incoming sunlight; `directionFromFroxelToCamera` is the negative outgoing view ray, so their dot product is the Henyey-Greenstein scattering cosine. All colors entering the fog shader are linear RGB. No analytic `calcFogFactor` is applied simultaneously.

## State Machine / Lifecycle

Feature installation, records and disposal follow the steps above; resources are per scene, per render task, per device, not process-global. WGSL is composed lazily at record time, with no module-initialization shader-building calls. Binding and pipeline state is recreated if the device or target identity changes; no per-frame texture allocation. Fog and transmission may share HDR ownership, but rendering transmissive surfaces while fog is enabled fails explicitly in V1.

## Babylon.js Equivalence Map

Babylon Lite `setFog(scene, FogConfig)` remains the analytic fog path and has no volume, lighting injection or compute. This feature implements Wronski's camera-aligned participating-medium integration, not Babylon.js analytic fog. AVBOIT later shares only mathematical and scheduling substrates.

## Dependencies

- `render/volume/` for validated grid mapping, optical transfer and 3D volume allocation.
- `camera/camera.ts` for active-camera projection, world transform and change detection.
- `light/directional-light.ts` for a source direction and linear radiance.
- `frame-graph/` for ordered compute, opaque render, fog composite and image processing.
- `material/pbr/fragments/` for opt-in linear HDR output.
- `engine/render-target.ts` and `resource/texture-sampler-pool.ts` for target and sampler ownership.

## Test Specification

- CPU unit: invalid values; spatial density decreases with height and saturates at maxDensity; HG phase normalization and forward/backward ordering; true off-axis distance at 60 degrees is twice axial slice thickness.
- CPU unit: shared linear HDR retarget is idempotent under fog and transmission enable order, preserves external depth, compare and sample count, and never mutates `surface.scRT`; only one final image-processing task.
- CPU unit: render phase calls fog after opaque/direct and before sorted transparent; absent phase preserves the original opaque bundle, draw count and pass count.
- GPU numeric: injected `(sigmaS * Li,sigmaT)` and integrated-volume readbacks for a homogeneous medium match the stated coefficients and `integrateHomogeneousMedium`; include complete directional-light source, extinction and integrated radiance/transmittance, zero extinction, and off-axis ray thickness. Near-plane result is the identity, and motion of a height-fog camera changes view-ray lighting consistently.
- PBR GPU: no-IBL opaque unlit material preserves its `baseColor * unlitColor` assignment in HDR fog, and matches the unfogged baseline for a foreground cube; shader composition places both the unlit assignment and other `NI` fragments outside the image-processing gate.
- GPU cloud fixture: compile the core default and lab-only density source with the same injection/integration passes on a real adapter. Measure exact-zero source/extinction outside the fixed bank, positive extinction within it, in-place temporal variation at a fixed froxel with constant base density, continuity across temporal seed boundaries, and zero density regardless of noise. The lab source rejects non-finite animation time; the core rejects invalid source layout or callback failures before changing the render task.
- WebGPU plumbing: real opaque scene, toggled fog, zero-density identity, **fog-enabled frozen-image camera and resize comparisons**, optional 4x MSAA, scene disposal and device recovery emit no validation errors. Test that analytic fog and unsupported transparency fail explicitly.
- Lifetime/unsupported cases: malformed density WGSL, wrong binding type or too-small caller uniform reject scene registration without publishing GPU state; per-device preparation failures reject recovery before frame-graph build. A Standard material or transparent PBR added after fog disable **or disposal** fails before rendering; unchanged opaque material generations skip the scan. Non-finite directional light components fail on installation and while fog is active. A custom tone mapper is rejected only when enabled. Same-surface two-scene composition fails explicitly whether the fog scene would be the first or second registered scene; a transmission-only HDR base still allows an ordinary single-sample overlay.
- Scene 316: `?seekTime=0`, `1` and `2` freeze distinct shapes with the **same** base extinction (0.28), camera and footprint; compare both the upper contour and interior. Verify the first cube, black sky and dark foreground stay clear, and the bright bank does not disappear as if extinction were pulsing. Pointer orbit, right-drag pan and wheel zoom change the camera without changing frozen noise; the visible reset control restores the initial image. Without `seekTime`, the noise evolves each frame but base density remains constant; pausing Cloud shape freezes the entire bank and resuming morphs it again. Old `?scroll=off` links pause the new animation. Every preset reload preserves URL options, updates both textures' displayed dimensions and remains responsive after viewport resize.
- Bundle: a non-fog scene does not grow relative to public master and fetches neither fog shader code nor the phase feature. Byte counts alone do not imply identical chunk contents.

## File Manifest

```text
docs/lite/architecture/56-volumetric-fog.md
packages/babylon-lite/src/frame-graph/linear-scene-color.ts
packages/babylon-lite/src/frame-graph/render-task-phase-pass.ts
packages/babylon-lite/src/frame-graph/render-task-base.ts
packages/babylon-lite/src/frame-graph/image-processing-task.ts
packages/babylon-lite/src/frame-graph/transmission.ts
packages/babylon-lite/src/material/pbr/fragments/linear-image-processing-fragment.ts
packages/babylon-lite/src/material/pbr/fragments/refraction-rtt-fragment.ts
packages/babylon-lite/src/material/pbr/pbr-material.ts
packages/babylon-lite/src/material/pbr/pbr-template.ts
packages/babylon-lite/src/shader/fragment-types.ts
packages/babylon-lite/src/scene/swapchain-overlay.ts
packages/babylon-lite/src/render/volume/optical-transfer-wgsl.ts
packages/babylon-lite/src/volumetric-fog/volumetric-fog.ts
packages/babylon-lite/src/volumetric-fog/volumetric-fog-wgsl.ts
packages/babylon-lite/src/index.ts
tests/lite/unit/volumetric-fog.test.ts
tests/lite/unit/scene316-cloud-density.test.ts
tests/lite/plumbing/volumetric-fog.spec.ts
lab/lite/scene316.html
lab/lite/bundle-scene316.html
lab/lite/src/lite/scene316.ts
lab/lite/src/shared/scene316-recovery.ts
lab/lite/src/shared/scene316-transmission.ts
lab/lite/src/shared/scene316-unlit.ts
lab/lite/src/shared/scene316-cloud-density.ts
lab/public/thumbnails/scene316.jpg
scene-config.json
azure-pipelines.yml
```
