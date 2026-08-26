import { createBloomPostProcessTask, createRenderTarget, getProjectionMatrix, getViewMatrix, getViewProjectionMatrix } from "babylon-lite";
import type { BloomPostProcessTask, Camera, EngineContext, RenderTarget, SceneContext, Task } from "babylon-lite";
import { buildRenderTarget, disposeRenderTarget } from "babylon-lite/engine/render-target.js";
import { ELECTRICITY_PARTICLE_MASK_RADIUS_SCALE, electricityPropagationRadius, type FluidElectricityFrameDomain } from "./fluid-runtime.js";

const MAX_ELECTRIFIED_DOMAINS = 256;
const DOMAIN_UNIFORM_BYTES = 256;
const LOW_QUALITY_MASK_SCALE = 0.5;

const MASK_WGSL = `
struct Domain {
    viewProjection: mat4x4<f32>,
    view: mat4x4<f32>,
    cameraRight: vec4<f32>,
    cameraUp: vec4<f32>,
    originRadius: vec4<f32>,
    settings: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> particleAlpha: array<f32>;
@group(0) @binding(2) var<uniform> domain: Domain;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) worldPosition: vec3<f32>,
    @location(2) alpha: f32,
}

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) particleIndex: u32) -> VertexOut {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );
    let corner = corners[vertexIndex];
    let center = positions[particleIndex].xyz;
    let radius = domain.settings.x;
    let worldPosition = center + (domain.cameraRight.xyz * corner.x + domain.cameraUp.xyz * corner.y) * radius;
    var out: VertexOut;
    out.position = domain.viewProjection * vec4<f32>(worldPosition, 1.0);
    out.corner = corner;
    out.worldPosition = worldPosition;
    out.alpha = particleAlpha[particleIndex];
    return out;
}

struct FragmentOut {
    @location(0) positionAlpha: vec4<f32>,
    @location(1) elapsedSeconds: f32,
}

@fragment
fn fs(input: VertexOut) -> FragmentOut {
    if (input.alpha <= 0.001) {
        discard;
    }
    let center = input.worldPosition -
        (domain.cameraRight.xyz * input.corner.x + domain.cameraUp.xyz * input.corner.y) * domain.settings.x;
    if (distance(center, domain.originRadius.xyz) > domain.originRadius.w) {
        discard;
    }
    var out: FragmentOut;
    out.positionAlpha = vec4<f32>(input.worldPosition, input.alpha);
    out.elapsedSeconds = domain.settings.y;
    return out;
}`;

const OVERLAY_WGSL = `
struct Overlay {
    view: mat4x4<f32>,
    cameraWorld: mat4x4<f32>,
    values: vec4<f32>,
    output: vec4<f32>,
}
@group(0) @binding(0) var maskTexture: texture_2d<f32>;
@group(0) @binding(1) var maskTimeTexture: texture_2d<f32>;
@group(0) @binding(2) var maskSampler: sampler;
@group(0) @binding(3) var sceneDepth: texture_depth_2d;
@group(0) @binding(4) var<uniform> overlay: Overlay;
@group(0) @binding(5) var fluidSurfaceDepth: texture_2d<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    return out;
}

fn hash1(value: vec3<f32>) -> f32 {
    return fract(sin(dot(value, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn boltLayer(
    position: vec2<f32>,
    direction: vec2<f32>,
    spacing: f32,
    segmentLength: f32,
    epoch: f32,
    transition: f32,
    seed: f32
) -> vec2<f32> {
    let normal = vec2<f32>(-direction.y, direction.x);
    let along = dot(position, direction);
    let across = dot(position, normal);
    let track = floor(across / spacing + 0.5);
    let segment = floor(along / segmentLength);
    let segmentPhase = fract(along / segmentLength);

    let jitter0 = mix(
        hash1(vec3<f32>(segment, track + seed * 19.0, epoch + seed)),
        hash1(vec3<f32>(segment + 1.0, track + seed * 19.0, epoch + seed)),
        segmentPhase
    );
    let jitter1 = mix(
        hash1(vec3<f32>(segment, track + seed * 19.0, epoch + 1.0 + seed)),
        hash1(vec3<f32>(segment + 1.0, track + seed * 19.0, epoch + 1.0 + seed)),
        segmentPhase
    );
    let centre = track * spacing + (mix(jitter0, jitter1, transition) - 0.5) * spacing * 0.55;
    let lineDistance = abs(across - centre);

    let blockLength = segmentLength * 18.0;
    let blockPosition = (along + seed * 1.37) / blockLength;
    let block = floor(blockPosition);
    let blockPhase = fract(blockPosition);
    let active0 = smoothstep(0.58, 0.78, hash1(vec3<f32>(track + seed * 5.0, block, epoch + seed * 3.0)));
    let active1 = smoothstep(0.58, 0.78, hash1(vec3<f32>(track + seed * 5.0, block, epoch + 1.0 + seed * 3.0)));
    let activation = mix(active0, active1, transition) * smoothstep(0.0, 0.08, blockPhase) * smoothstep(0.0, 0.08, 1.0 - blockPhase);

    let core = (1.0 - smoothstep(0.012, 0.038, lineDistance)) * activation;
    let glow = (1.0 - smoothstep(0.035, 0.14, lineDistance)) * activation;
    return vec2<f32>(core, glow);
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = position.xy * overlay.output.xy;
    let mask = textureSampleLevel(maskTexture, maskSampler, uv, 0.0);
    let elapsedSeconds = textureSampleLevel(maskTimeTexture, maskSampler, uv, 0.0).r;
    if (mask.a <= 0.001) {
        discard;
    }
    var p = mask.xyz;
    if (overlay.values.x > 0.5) {
        let surfaceDimensions = vec2<i32>(textureDimensions(fluidSurfaceDepth));
        let surfacePixel = min(vec2<i32>(uv * vec2<f32>(surfaceDimensions)), surfaceDimensions - vec2<i32>(1));
        let surfaceEyeDepth = textureLoad(fluidSurfaceDepth, surfacePixel, 0).r;
        if (surfaceEyeDepth <= 0.0 || surfaceEyeDepth >= 1.0e5) {
            discard;
        }
        let viewPosition = vec3<f32>(
            (uv.x * 2.0 - 1.0) * surfaceEyeDepth * overlay.values.w * overlay.output.z,
            (1.0 - uv.y * 2.0) * surfaceEyeDepth * overlay.values.w,
            surfaceEyeDepth
        );
        p = (overlay.cameraWorld * vec4<f32>(viewPosition, 1.0)).xyz;
    }
    let eyeDepth = (overlay.view * vec4<f32>(p, 1.0)).z;
    let depthPixel = min(vec2<i32>(position.xy), vec2<i32>(textureDimensions(sceneDepth)) - vec2<i32>(1));
    let sceneNdc = textureLoad(sceneDepth, depthPixel, 0);
    let sceneEyeDepth = overlay.values.z / (sceneNdc - overlay.values.y);
    if (sceneNdc > 0.0 && eyeDepth > sceneEyeDepth + 0.02) {
        discard;
    }
    let tick = elapsedSeconds * 7.0;
    let epoch = floor(tick);
    let transition = smoothstep(0.62, 1.0, fract(tick));
    let surface = p.xz + p.y * vec2<f32>(0.21, 0.37);
    let side = p.xy + p.z * vec2<f32>(0.19, 0.31);
    let density = max(overlay.output.w, 0.25);
    let primary = boltLayer(surface, vec2<f32>(0.8944, 0.4472), 1.8 / density, 0.34, epoch, transition, 2.3);
    let crossing = boltLayer(surface, vec2<f32>(-0.5299, 0.8480), 2.5 / density, 0.28, epoch, transition, 7.1);
    let branch0 = boltLayer(surface, vec2<f32>(0.2334, 0.9724), 2.2 / density, 0.19, epoch, transition, 13.9);
    let branch1 = boltLayer(surface, vec2<f32>(-0.9231, 0.3846), 2.4 / density, 0.21, epoch, transition, 17.3);
    let vertical = boltLayer(side, vec2<f32>(0.7682, 0.6402), 2.6 / density, 0.30, epoch, transition, 11.7);
    let core = max(primary.x, max(crossing.x * 0.85, max(branch0.x * 0.65, max(branch1.x * 0.6, vertical.x * 0.7))));
    let glow = max(primary.y, max(crossing.y * 0.7, max(branch0.y * 0.5, max(branch1.y * 0.45, vertical.y * 0.55))));
    let flicker = 0.72 + 0.28 * sin(elapsedSeconds * 38.0 + dot(p, vec3<f32>(4.1, 5.3, 3.7)));
    let strength = mask.a * flicker;
    let flashClock = elapsedSeconds * 8.0;
    let flashCell = floor(p * 1.35);
    let flashRandom = hash1(flashCell + vec3<f32>(floor(flashClock) * 0.73, floor(flashClock) * 1.17, floor(flashClock) * 1.91));
    let flash = smoothstep(0.82, 0.97, flashRandom) * (1.0 - smoothstep(0.08, 0.72, fract(flashClock)));
    let color =
        vec3<f32>(0.02, 0.20, 0.55) * glow * strength * 0.45 +
        vec3<f32>(0.52, 0.84, 1.0) * core * strength * 0.85 +
        vec3<f32>(1.0) * (core + glow * 0.22) * mask.a * flash * 1.35;
    if (max(color.r, max(color.g, color.b)) < 0.015) {
        discard;
    }
    return vec4<f32>(color, 0.0);
}`;

const BLOOM_PRESENT_WGSL = `
@group(0) @binding(0) var bloomedTexture: texture_2d<f32>;
@group(0) @binding(1) var effectTexture: texture_2d<f32>;
@group(0) @binding(2) var effectSampler: sampler;
@group(0) @binding(3) var<uniform> presentSettings: vec4<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    return out;
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(effectTexture));
    let uv = position.xy / dimensions;
    let comparison = presentSettings.x > 0.5;
    let sampleUv = select(uv, vec2<f32>(fract(uv.x * 2.0), uv.y), comparison);
    let center = textureSampleLevel(effectTexture, effectSampler, sampleUv, 0.0).rgb;
    let bloomed = textureSampleLevel(bloomedTexture, effectSampler, sampleUv, 0.0).rgb;
    var color = bloomed;
    if (comparison) {
        color = select(center, bloomed, uv.x >= 0.5);
        if (abs(position.x - dimensions.x * 0.5) < 1.0) {
            color = vec3<f32>(1.0);
        }
    }
    return vec4<f32>(color, 0.0);
}`;

export interface FluidElectricityRendererOptions {
    readonly target: RenderTarget;
    readonly sceneDepth: RenderTarget;
    readonly camera: Camera;
    readonly positionBuffer: GPUBuffer;
    readonly alphaBuffer: GPUBuffer;
    readonly surfaceDepthView: () => GPUTextureView | null;
}

export interface FluidElectricityRenderer extends Task {
    setDomains(domains: readonly FluidElectricityFrameDomain[]): void;
    setImproved(improved: boolean): void;
    setAnimationEnabled(enabled: boolean): void;
    setArcDensity(density: number): void;
    setBloom(threshold: number, strength: number, radius: number): void;
    setBloomDebug(enabled: boolean): void;
}

export function createFluidElectricityRenderer(engine: EngineContext, scene: SceneContext, options: FluidElectricityRendererOptions): FluidElectricityRenderer {
    const device = engine._device;
    let domains: readonly FluidElectricityFrameDomain[] = [];
    let improved = false;
    let animationEnabled = true;
    let arcDensity = 1;
    let frozenElapsedByDomain = new WeakMap<FluidElectricityFrameDomain["domain"], number>();
    let maskTexture: GPUTexture | null = null;
    let maskView: GPUTextureView | null = null;
    let maskTimeTexture: GPUTexture | null = null;
    let maskTimeView: GPUTextureView | null = null;
    let maskDepthTexture: GPUTexture | null = null;
    let maskDepthView: GPUTextureView | null = null;
    let maskWidth = 0;
    let maskHeight = 0;
    let domainUniform: GPUBuffer | null = null;
    let overlayUniform: GPUBuffer | null = null;
    let bloomPresentUniform: GPUBuffer | null = null;
    let bloomThreshold = 0.62;
    let bloomStrength = 0.9;
    let bloomRadius = 16;
    let bloomDebug = false;
    let bloomRecorded = false;
    let maskPipeline: GPURenderPipeline | null = null;
    let overlayDirectPipeline: GPURenderPipeline | null = null;
    let overlayEffectPipeline: GPURenderPipeline | null = null;
    let bloomPresentPipeline: GPURenderPipeline | null = null;
    let overlayLayout: GPUBindGroupLayout | null = null;
    let overlayModule: GPUShaderModule | null = null;
    let maskBindGroup: GPUBindGroup | null = null;
    let overlayBindGroup: GPUBindGroup | null = null;
    let bloomPresentBindGroup: GPUBindGroup | null = null;
    let overlayDepthView: GPUTextureView | null = null;
    let overlaySurfaceDepthView: GPUTextureView | null = null;
    let sampler: GPUSampler | null = null;
    const effectTarget = createRenderTarget({ lbl: "aq-fluid-electricity-effect", format: "rgba16float", samples: 1, size: engine });
    const bloomOutputTarget = createRenderTarget({ lbl: "aq-fluid-electricity-bloom-output", format: "rgba16float", samples: 1, size: engine });
    const bloomTask: BloomPostProcessTask = createBloomPostProcessTask(
        {
            name: "aq-fluid-electricity-bloom",
            sourceTexture: effectTarget,
            targetTexture: bloomOutputTarget,
            sourceSamplingMode: "linear",
            threshold: bloomThreshold,
            weight: bloomStrength,
            kernel: bloomRadius * 2 + 1,
            bloomScale: 0.5,
        },
        engine,
        scene
    );

    const disposeTargets = (): void => {
        maskTexture?.destroy();
        maskTimeTexture?.destroy();
        maskDepthTexture?.destroy();
        maskTexture = null;
        maskView = null;
        maskTimeTexture = null;
        maskTimeView = null;
        maskDepthTexture = null;
        maskDepthView = null;
        overlayBindGroup = null;
        overlayDepthView = null;
        overlaySurfaceDepthView = null;
    };

    const ensureTargets = (): void => {
        const scale = improved ? 1 : LOW_QUALITY_MASK_SCALE;
        const width = Math.max(1, Math.ceil(engine.canvas.width * scale));
        const height = Math.max(1, Math.ceil(engine.canvas.height * scale));
        if (maskTexture && width === maskWidth && height === maskHeight) {
            return;
        }
        disposeTargets();
        maskWidth = width;
        maskHeight = height;
        maskTexture = device.createTexture({
            label: "aq-fluid-electricity-mask",
            size: [width, height],
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        maskView = maskTexture.createView();
        maskTimeTexture = device.createTexture({
            label: "aq-fluid-electricity-mask-time",
            size: [width, height],
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        maskTimeView = maskTimeTexture.createView();
        maskDepthTexture = device.createTexture({
            label: "aq-fluid-electricity-mask-depth",
            size: [width, height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        maskDepthView = maskDepthTexture.createView();
    };

    const ensureResources = (): void => {
        if (maskPipeline) {
            ensureTargets();
            return;
        }
        const maskLayout = device.createBindGroupLayout({
            label: "aq-fluid-electricity-mask",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform", hasDynamicOffset: true } },
            ],
        });
        domainUniform = device.createBuffer({
            label: "aq-fluid-electricity-domains",
            size: MAX_ELECTRIFIED_DOMAINS * DOMAIN_UNIFORM_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        overlayUniform = device.createBuffer({
            label: "aq-fluid-electricity-overlay",
            size: 160,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const maskModule = device.createShaderModule({ label: "aq-fluid-electricity-mask", code: MASK_WGSL });
        maskPipeline = device.createRenderPipeline({
            label: "aq-fluid-electricity-mask",
            layout: device.createPipelineLayout({ bindGroupLayouts: [maskLayout] }),
            vertex: { module: maskModule, entryPoint: "vs" },
            fragment: { module: maskModule, entryPoint: "fs", targets: [{ format: "rgba16float" }, { format: "r16float" }] },
            primitive: { topology: "triangle-list" },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: true,
                depthCompare: "greater",
            },
        });
        maskBindGroup = device.createBindGroup({
            label: "aq-fluid-electricity-mask",
            layout: maskLayout,
            entries: [
                { binding: 0, resource: { buffer: options.positionBuffer } },
                { binding: 1, resource: { buffer: options.alphaBuffer } },
                { binding: 2, resource: { buffer: domainUniform, size: 192 } },
            ],
        });
        overlayLayout = device.createBindGroupLayout({
            label: "aq-fluid-electricity-overlay",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
            ],
        });
        overlayModule = device.createShaderModule({ label: "aq-fluid-electricity-overlay", code: OVERLAY_WGSL });
        overlayDirectPipeline = device.createRenderPipeline({
            label: "aq-fluid-electricity-overlay",
            layout: device.createPipelineLayout({ bindGroupLayouts: [overlayLayout] }),
            vertex: { module: overlayModule, entryPoint: "vs" },
            fragment: {
                module: overlayModule,
                entryPoint: "fs",
                targets: [
                    {
                        format: engine.format,
                        blend: {
                            color: { srcFactor: "one", dstFactor: "one", operation: "add" },
                            alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
                        },
                    },
                ],
            },
            primitive: { topology: "triangle-list" },
        });
        sampler = device.createSampler({
            label: "aq-fluid-electricity-mask",
            minFilter: "linear",
            magFilter: "linear",
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge",
        });
        ensureTargets();
    };

    const ensureImprovedResources = (): void => {
        if (overlayEffectPipeline && bloomPresentPipeline) {
            ensureTargets();
            return;
        }
        if (!overlayLayout || !overlayModule) {
            throw new Error("[aquanova] electrical overlay resources must exist before improved resources");
        }
        overlayEffectPipeline = device.createRenderPipeline({
            label: "aq-fluid-electricity-effect",
            layout: device.createPipelineLayout({ bindGroupLayouts: [overlayLayout] }),
            vertex: { module: overlayModule, entryPoint: "vs" },
            fragment: { module: overlayModule, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
            primitive: { topology: "triangle-list" },
        });
        const bloomPresentLayout = device.createBindGroupLayout({
            label: "aq-fluid-electricity-bloom-present",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ],
        });
        bloomPresentUniform = device.createBuffer({
            label: "aq-fluid-electricity-bloom-present",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bloomPresentUniform, 0, new Float32Array([bloomDebug ? 1 : 0, 0, 0, 0]));
        const bloomPresentModule = device.createShaderModule({ label: "aq-fluid-electricity-bloom-present", code: BLOOM_PRESENT_WGSL });
        bloomPresentPipeline = device.createRenderPipeline({
            label: "aq-fluid-electricity-bloom-present",
            layout: device.createPipelineLayout({ bindGroupLayouts: [bloomPresentLayout] }),
            vertex: { module: bloomPresentModule, entryPoint: "vs" },
            fragment: {
                module: bloomPresentModule,
                entryPoint: "fs",
                targets: [
                    {
                        format: engine.format,
                        blend: {
                            color: { srcFactor: "one", dstFactor: "one", operation: "add" },
                            alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
                        },
                    },
                ],
            },
            primitive: { topology: "triangle-list" },
        });
        ensureTargets();
    };

    const buildOverlayBindGroup = (): GPUBindGroup | null => {
        const sceneDepthView = options.sceneDepth._depthView;
        const surfaceDepthView = options.surfaceDepthView();
        if (!maskView || !maskTimeView || !sceneDepthView || !surfaceDepthView || !sampler || !overlayUniform || !overlayLayout) {
            return null;
        }
        if (!overlayBindGroup || overlayDepthView !== sceneDepthView || overlaySurfaceDepthView !== surfaceDepthView) {
            overlayBindGroup = device.createBindGroup({
                label: "aq-fluid-electricity-overlay",
                layout: overlayLayout,
                entries: [
                    { binding: 0, resource: maskView },
                    { binding: 1, resource: maskTimeView },
                    { binding: 2, resource: sampler },
                    { binding: 3, resource: sceneDepthView },
                    { binding: 4, resource: { buffer: overlayUniform } },
                    { binding: 5, resource: surfaceDepthView },
                ],
            });
            overlayDepthView = sceneDepthView;
            overlaySurfaceDepthView = surfaceDepthView;
        }
        return overlayBindGroup;
    };

    const buildBloomPresentBindGroup = (): GPUBindGroup | null => {
        if (!effectTarget._colorView || !bloomOutputTarget._colorView || !sampler || !bloomPresentPipeline || !bloomPresentUniform) {
            return null;
        }
        bloomPresentBindGroup ??= device.createBindGroup({
            label: "aq-fluid-electricity-bloom-present",
            layout: bloomPresentPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: bloomOutputTarget._colorView },
                { binding: 1, resource: effectTarget._colorView },
                { binding: 2, resource: sampler },
                { binding: 3, resource: { buffer: bloomPresentUniform } },
            ],
        });
        return bloomPresentBindGroup;
    };

    const recordBloomResources = (): void => {
        buildRenderTarget(effectTarget, engine);
        bloomTask.record();
        bloomTask.updateUniforms();
        bloomRecorded = true;
        bloomPresentBindGroup = null;
    };

    return {
        name: "aquanova-fluid-electricity",
        engine,
        scene,
        _passes: [],
        setDomains(frameDomains): void {
            domains = frameDomains.filter((frame) => frame.domain.electricity !== null);
            if (!animationEnabled) {
                for (const frame of domains) {
                    if (!frozenElapsedByDomain.has(frame.domain)) {
                        frozenElapsedByDomain.set(frame.domain, frame.elapsedSeconds);
                    }
                }
            }
        },
        setImproved(on): void {
            if (improved === on) {
                return;
            }
            improved = on;
            disposeTargets();
        },
        setAnimationEnabled(on): void {
            if (animationEnabled === on) {
                return;
            }
            animationEnabled = on;
            frozenElapsedByDomain = new WeakMap();
            if (!on) {
                for (const frame of domains) {
                    frozenElapsedByDomain.set(frame.domain, frame.elapsedSeconds);
                }
            }
        },
        setArcDensity(density): void {
            if (!Number.isFinite(density) || density <= 0) {
                throw new RangeError("[aquanova] electrical arc density must be finite and positive");
            }
            arcDensity = density;
        },
        setBloom(threshold, strength, radius): void {
            if (!Number.isFinite(threshold) || threshold < 0 || !Number.isFinite(strength) || strength < 0 || !Number.isFinite(radius) || radius <= 0) {
                throw new RangeError("[aquanova] electrical bloom threshold, strength, and radius must be finite, with non-negative threshold/strength and positive radius");
            }
            bloomThreshold = threshold;
            bloomStrength = strength;
            bloomRadius = radius;
            bloomTask.threshold = threshold;
            bloomTask.weight = strength;
            bloomTask.kernel = radius * 2 + 1;
            if (bloomRecorded) {
                bloomTask.updateUniforms();
            }
        },
        setBloomDebug(on): void {
            bloomDebug = on;
            if (bloomPresentUniform) {
                device.queue.writeBuffer(bloomPresentUniform, 0, new Float32Array([on ? 1 : 0, 0, 0, 0]));
            }
        },
        record(): void {
            recordBloomResources();
        },
        execute(): number {
            if (domains.length === 0 || !options.target._colorView) {
                return 0;
            }
            if (domains.length > MAX_ELECTRIFIED_DOMAINS) {
                throw new RangeError(`[aquanova] ${domains.length} electrified fluid domains exceed the ${MAX_ELECTRIFIED_DOMAINS} renderer capacity`);
            }
            ensureResources();
            if (improved) {
                ensureImprovedResources();
            }
            const overlayGroup = buildOverlayBindGroup();
            const overlayPipeline = improved ? overlayEffectPipeline : overlayDirectPipeline;
            const overlayTarget = improved ? effectTarget._colorView : options.target._colorView;
            if (
                !maskPipeline ||
                !maskBindGroup ||
                !maskView ||
                !maskTimeView ||
                !maskDepthView ||
                !domainUniform ||
                !overlayPipeline ||
                !overlayTarget ||
                !overlayGroup ||
                !overlayUniform
            ) {
                return 0;
            }
            const aspect = engine.canvas.width / Math.max(1, engine.canvas.height);
            const viewProjection = getViewProjectionMatrix(options.camera, aspect);
            const view = getViewMatrix(options.camera);
            const projection = getProjectionMatrix(options.camera, aspect);
            const world = options.camera.worldMatrix;
            const data = new Float32Array((domains.length * DOMAIN_UNIFORM_BYTES) / 4);
            for (let index = 0; index < domains.length; index++) {
                const frame = domains[index]!;
                const electricity = frame.domain.electricity!;
                const visualElapsedSeconds = animationEnabled ? frame.elapsedSeconds : (frozenElapsedByDomain.get(frame.domain) ?? frame.elapsedSeconds);
                const offset = (index * DOMAIN_UNIFORM_BYTES) / 4;
                data.set(viewProjection, offset);
                data.set(view, offset + 16);
                data.set([world[0]!, world[1]!, world[2]!, 0], offset + 32);
                data.set([world[4]!, world[5]!, world[6]!, 0], offset + 36);
                data.set([electricity.origin[0], electricity.origin[1], electricity.origin[2], electricityPropagationRadius(electricity, frame.elapsedSeconds)], offset + 40);
                data.set([frame.particleRadius * ELECTRICITY_PARTICLE_MASK_RADIUS_SCALE, visualElapsedSeconds % 32, 1, 0], offset + 44);
            }
            device.queue.writeBuffer(domainUniform, 0, data);

            const encoder = engine._currentEncoder;
            const maskPass = encoder.beginRenderPass({
                label: "aq-fluid-electricity-mask",
                colorAttachments: [
                    { view: maskView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } },
                    { view: maskTimeView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } },
                ],
                depthStencilAttachment: { view: maskDepthView, depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 0 },
            });
            maskPass.setPipeline(maskPipeline);
            for (let index = 0; index < domains.length; index++) {
                const frame = domains[index]!;
                maskPass.setBindGroup(0, maskBindGroup, [index * DOMAIN_UNIFORM_BYTES]);
                maskPass.draw(6, frame.count, 0, frame.offset);
            }
            maskPass.end();

            const overlayData = new Float32Array(40);
            overlayData.set(view, 0);
            overlayData.set(world, 16);
            overlayData.set([improved ? 1 : 0, projection[10]!, projection[14]!, Math.tan(options.camera.fov * 0.5)], 32);
            overlayData.set([1 / Math.max(1, engine.canvas.width), 1 / Math.max(1, engine.canvas.height), aspect, arcDensity], 36);
            device.queue.writeBuffer(overlayUniform, 0, overlayData);
            const overlayPass = encoder.beginRenderPass({
                label: "aq-fluid-electricity-overlay",
                colorAttachments: [{ view: overlayTarget, loadOp: improved ? "clear" : "load", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
            });
            overlayPass.setPipeline(overlayPipeline);
            overlayPass.setBindGroup(0, overlayGroup);
            overlayPass.draw(3);
            overlayPass.end();
            if (improved) {
                const bloomPresentGroup = buildBloomPresentBindGroup();
                if (!bloomPresentPipeline || !bloomPresentGroup) {
                    return domains.length + 1;
                }
                const bloomPasses = bloomTask.execute?.() ?? 0;
                const bloomPresentPass = encoder.beginRenderPass({
                    label: "aq-fluid-electricity-bloom-present",
                    colorAttachments: [
                        {
                            view: options.target._colorView,
                            loadOp: bloomDebug ? "clear" : "load",
                            storeOp: "store",
                            clearValue: { r: 0, g: 0, b: 0, a: 1 },
                        },
                    ],
                });
                bloomPresentPass.setPipeline(bloomPresentPipeline);
                bloomPresentPass.setBindGroup(0, bloomPresentGroup);
                bloomPresentPass.draw(3);
                bloomPresentPass.end();
                return domains.length + bloomPasses + 2;
            }
            return domains.length + 1;
        },
        dispose(): void {
            disposeTargets();
            bloomTask.dispose();
            disposeRenderTarget(effectTarget);
            disposeRenderTarget(bloomOutputTarget);
            domainUniform?.destroy();
            overlayUniform?.destroy();
            bloomPresentUniform?.destroy();
            domainUniform = null;
            overlayUniform = null;
            bloomPresentUniform = null;
            maskPipeline = null;
            overlayDirectPipeline = null;
            overlayEffectPipeline = null;
            bloomPresentPipeline = null;
            overlayLayout = null;
            overlayModule = null;
            maskBindGroup = null;
            overlayBindGroup = null;
            bloomPresentBindGroup = null;
            overlayDepthView = null;
            overlaySurfaceDepthView = null;
            sampler = null;
        },
    };
}
