import type { EngineContext } from "../engine/engine.js";
import type { RenderTarget } from "../engine/render-target.js";
import { buildRenderTarget, createRenderTarget, disposeRenderTarget } from "../engine/render-target.js";
import type { Task } from "../frame-graph/task.js";
import type { SceneContext } from "../scene/scene-core.js";

/** One independently reconstructed fluid layer consumed by the compositor. */
export interface FluidRenderLayer {
    readonly colorTarget: RenderTarget;
    readonly surfaceDepthView: () => GPUTextureView | null;
}

/** Depth-aware compositor for independently reconstructed fluid surface layers. */
export interface FluidRenderCompositor extends Task {
    /** Set the active independent layers and whether the base color contains a shared-path fluid surface. */
    setLayers(layers: readonly FluidRenderLayer[], baseDepthActive: boolean): void;
    /** Final nearest-fluid eye-space depth, or the shared-path depth when no independent layer is active. */
    surfaceDepthView(): GPUTextureView | null;
}

const SHADER = /* wgsl */ `
struct Params {
    currentDepthActive: u32,
    _pad0: vec3<u32>,
};
@group(0) @binding(0) var currentColor: texture_2d<f32>;
@group(0) @binding(1) var currentDepth: texture_2d<f32>;
@group(0) @binding(2) var candidateColor: texture_2d<f32>;
@group(0) @binding(3) var candidateDepth: texture_2d<f32>;
@group(0) @binding(4) var<uniform> params: Params;

struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> Varyings {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    let uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0)
    );
    var out: Varyings;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    out.uv = uvs[index];
    return out;
}

struct Output {
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
};

fn textureCoord(uv: vec2<f32>, dimensions: vec2<u32>) -> vec2<i32> {
    let pixel = min(vec2<u32>(uv * vec2<f32>(dimensions)), dimensions - vec2<u32>(1));
    return vec2<i32>(pixel);
}

@fragment
fn fs(input: Varyings) -> Output {
    let currentColorDimensions = textureDimensions(currentColor);
    let candidateColorDimensions = textureDimensions(candidateColor);
    let currentDepthDimensions = textureDimensions(currentDepth);
    let candidateDepthDimensions = textureDimensions(candidateDepth);
    let oldColor = textureLoad(currentColor, textureCoord(input.uv, currentColorDimensions), 0);
    let newColor = textureLoad(candidateColor, textureCoord(input.uv, candidateColorDimensions), 0);
    let oldDepth = textureLoad(currentDepth, textureCoord(input.uv, currentDepthDimensions), 0).r;
    let newDepth = textureLoad(candidateDepth, textureCoord(input.uv, candidateDepthDimensions), 0).r;
    let newDepthValid = newDepth < 500000.0;
    let useNew = newDepthValid && (params.currentDepthActive == 0u || newDepth < oldDepth);
    var out: Output;
    out.color = select(oldColor, newColor, useNew);
    out.depth = select(select(1000000.0, oldDepth, params.currentDepthActive != 0u), newDepth, useNew);
    return out;
}`;

interface CachedBinding {
    readonly currentColor: GPUTextureView;
    readonly currentDepth: GPUTextureView;
    readonly candidateColor: GPUTextureView;
    readonly candidateDepth: GPUTextureView;
    readonly currentDepthActive: boolean;
    readonly params: GPUBuffer;
    readonly bindGroup: GPUBindGroup;
}

/** Create a compositor that merges independently reconstructed fluid layers into `baseColorTarget`. */
export function createFluidRenderCompositor(
    engine: EngineContext,
    scene: SceneContext,
    baseColorTarget: RenderTarget,
    baseDepthView: () => GPUTextureView | null
): FluidRenderCompositor {
    const device = engine._device;
    const colorFormat = baseColorTarget._descriptor.format;
    if (!colorFormat) {
        throw new Error("Fluid render compositor requires a color target.");
    }
    const baseSnapshot = createRenderTarget({ lbl: "fluid-composite-base", format: colorFormat, samples: 1, size: engine });
    const colorTargets = [
        createRenderTarget({ lbl: "fluid-composite-color-a", format: colorFormat, samples: 1, size: engine }),
        createRenderTarget({ lbl: "fluid-composite-color-b", format: colorFormat, samples: 1, size: engine }),
    ] as const;
    const depthTargets = [
        createRenderTarget({ lbl: "fluid-composite-depth-a", format: "r32float", samples: 1, size: engine }),
        createRenderTarget({ lbl: "fluid-composite-depth-b", format: "r32float", samples: 1, size: engine }),
    ] as const;
    let pipeline: GPURenderPipeline | null = null;
    let layers: readonly FluidRenderLayer[] = [];
    let baseActive = false;
    let lastDepthView: GPUTextureView | null = null;
    const cachedBindings: CachedBinding[] = [];

    const clearBindingCache = (): void => {
        for (const cached of cachedBindings) {
            cached.params.destroy();
        }
        cachedBindings.length = 0;
    };

    const bindGroupFor = (
        currentColor: GPUTextureView,
        currentDepth: GPUTextureView,
        candidateColor: GPUTextureView,
        candidateDepth: GPUTextureView,
        currentDepthActive: boolean
    ): GPUBindGroup => {
        const cached = cachedBindings.find(
            (entry) =>
                entry.currentColor === currentColor &&
                entry.currentDepth === currentDepth &&
                entry.candidateColor === candidateColor &&
                entry.candidateDepth === candidateDepth &&
                entry.currentDepthActive === currentDepthActive
        );
        if (cached) {
            return cached.bindGroup;
        }
        const params = device.createBuffer({
            label: "fluid-composite-params",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(params, 0, new Uint32Array([currentDepthActive ? 1 : 0, 0, 0, 0]));
        const bindGroup = device.createBindGroup({
            label: "fluid-composite-bind-group",
            layout: pipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: currentColor },
                { binding: 1, resource: currentDepth },
                { binding: 2, resource: candidateColor },
                { binding: 3, resource: candidateDepth },
                { binding: 4, resource: { buffer: params } },
            ],
        });
        cachedBindings.push({ currentColor, currentDepth, candidateColor, candidateDepth, currentDepthActive, params, bindGroup });
        return bindGroup;
    };

    return {
        name: "fluid-render-compositor",
        engine,
        scene,
        _passes: [],
        setLayers(nextLayers, baseDepthActive): void {
            layers = nextLayers;
            baseActive = baseDepthActive;
            if (layers.length === 0) {
                lastDepthView = null;
            }
        },
        surfaceDepthView(): GPUTextureView | null {
            return lastDepthView ?? baseDepthView();
        },
        record(): void {
            buildRenderTarget(baseSnapshot, engine);
            for (const target of colorTargets) {
                buildRenderTarget(target, engine);
            }
            for (const target of depthTargets) {
                buildRenderTarget(target, engine);
            }
            clearBindingCache();
            if (!pipeline) {
                const module = device.createShaderModule({ label: "fluid-render-compositor", code: SHADER });
                const layout = device.createBindGroupLayout({
                    label: "fluid-render-compositor",
                    entries: [
                        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
                        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                    ],
                });
                pipeline = device.createRenderPipeline({
                    label: "fluid-render-compositor",
                    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
                    vertex: { module, entryPoint: "vs" },
                    fragment: { module, entryPoint: "fs", targets: [{ format: colorFormat }, { format: "r32float" }] },
                    primitive: { topology: "triangle-list" },
                });
            }
        },
        execute(): number {
            if (layers.length === 0 || !pipeline || !baseColorTarget._colorTexture || !baseSnapshot._colorTexture) {
                lastDepthView = null;
                return 0;
            }
            const readyLayers = layers.flatMap((layer) => {
                const color = layer.colorTarget._colorView;
                const depth = layer.surfaceDepthView();
                return color && depth ? [{ color, depth }] : [];
            });
            if (readyLayers.length === 0) {
                lastDepthView = null;
                return 0;
            }
            const encoder = engine._currentEncoder;
            encoder.copyTextureToTexture(
                { texture: baseColorTarget._colorTexture },
                { texture: baseSnapshot._colorTexture },
                { width: baseSnapshot._width, height: baseSnapshot._height, depthOrArrayLayers: 1 }
            );
            let currentColor = baseSnapshot._colorView!;
            const initialBaseDepth = baseDepthView();
            let currentDepth = initialBaseDepth ?? readyLayers[0]!.depth;
            let currentDepthActive = baseActive && initialBaseDepth !== null;
            let outputIndex = 0;
            for (const layer of readyLayers) {
                const colorTarget = colorTargets[outputIndex]!;
                const depthTarget = depthTargets[outputIndex]!;
                const pass = encoder.beginRenderPass({
                    label: "fluid-render-composite-layer",
                    colorAttachments: [
                        { view: colorTarget._colorView!, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } },
                        { view: depthTarget._colorView!, loadOp: "clear", storeOp: "store", clearValue: { r: 1e6, g: 0, b: 0, a: 0 } },
                    ],
                });
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindGroupFor(currentColor, currentDepth, layer.color, layer.depth, currentDepthActive));
                pass.draw(3);
                pass.end();
                currentColor = colorTarget._colorView!;
                currentDepth = depthTarget._colorView!;
                currentDepthActive = true;
                outputIndex = 1 - outputIndex;
            }
            const finalIndex = 1 - outputIndex;
            encoder.copyTextureToTexture(
                { texture: colorTargets[finalIndex]!._colorTexture! },
                { texture: baseColorTarget._colorTexture },
                { width: baseSnapshot._width, height: baseSnapshot._height, depthOrArrayLayers: 1 }
            );
            lastDepthView = depthTargets[finalIndex]!._colorView;
            return readyLayers.length;
        },
        dispose(): void {
            clearBindingCache();
            disposeRenderTarget(baseSnapshot);
            for (const target of colorTargets) {
                disposeRenderTarget(target);
            }
            for (const target of depthTargets) {
                disposeRenderTarget(target);
            }
        },
    };
}
