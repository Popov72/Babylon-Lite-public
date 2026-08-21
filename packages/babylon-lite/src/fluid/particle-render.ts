// Particle renderer for the GPU fluid sim.
//
// Draws the GPU-resident particles as camera-facing billboarded sphere
// impostors, reading the simulation's position buffer directly in the vertex
// shader (no CPU round-trip, no per-frame vertex upload). Implemented as a
// custom frame-graph Task that draws into the scene's colour target while
// sharing the scene depth buffer, so particles depth-test against the ground
// (and, later, the tank) and occlude correctly.

import { getEffectiveAspectRatio, getViewProjectionMatrix } from "../camera/camera.js";
import type { Camera } from "../camera/camera.js";
import type { EngineContext } from "../engine/engine.js";
import type { RenderTarget } from "../engine/render-target.js";
import type { SceneContext } from "../scene/scene-core.js";
import type { Task } from "../frame-graph/task.js";
import type { FluidSim, FluidProfiler } from "./sim-common.js";

// Opt-in GPU timing hook (see FluidProfiler / lab gpu-profiler.ts). Module-scoped:
// null by default so `profiler?.pass(...)` is undefined and timing costs nothing.
let profiler: FluidProfiler | null = null;

export interface ParticleRenderOptions {
    /** Colour target the particles draw into (typically `engine.scRT`). */
    colorRT: RenderTarget;
    /** Depth target shared with the scene render task (load + test + write). */
    depthRT: RenderTarget;
    camera: Camera;
    sim: FluidSim;
}

/**
 * Opt-in replacement for the complete particle billboard WGSL module.
 *
 * Bind group 0 is stable: binding 0 is the 128-byte camera uniform used by the
 * built-in shader, binding 1 is the simulation `array<vec4<f32>>` position
 * buffer, binding 2 is the simulation `array<f32>` debug buffer, and binding 3
 * is present only when `customUniforms` is supplied.
 */
export interface ParticleRenderShaderOptions {
    /** Complete WGSL shader-module source. */
    code: string;
    /** Vertex entry point. Defaults to `vs`. */
    vertexEntryPoint?: string;
    /** Fragment entry point. Defaults to `fs`. */
    fragmentEntryPoint?: string;
    /**
     * Initial bytes for an optional caller-defined payload exposed as a uniform
     * buffer at group 0, binding 3. Its byte length must be a multiple of four.
     */
    customUniforms?: ArrayBufferView;
    /** Target blend state. Defaults to the built-in alpha blend; `null` disables blending. */
    blend?: GPUBlendState | null;
    /** Whether particle fragments write depth. Defaults to `true`. */
    depthWriteEnabled?: boolean;
    /** Depth comparison function. Defaults to reverse-Z `greater-equal`. */
    depthCompare?: GPUCompareFunction;
}

/** Public controls returned by {@link createParticleRenderTask}. */
export interface ParticleRenderTask extends Task {
    setSim(s: FluidSim): void;
    setEnabled(on: boolean): void;
    setOpacity(v: number): void;
    setSizeScale(s: number): void;
    setTint(rgb: [number, number, number]): void;
    setVelocityBrighten(v: number): void;
    setProfiler(p: FluidProfiler | null): void;
    /**
     * Select a complete custom WGSL module, or restore the built-in module with
     * `null`. Pipeline recreation is deferred until the task next records or
     * executes.
     */
    setShader(shader: ParticleRenderShaderOptions | null): void;
    /**
     * Replace the binding-3 payload configured by the active custom shader.
     * The replacement must have the same byte length as `customUniforms`.
     */
    setCustomUniforms(data: ArrayBufferView): void;
}

const RENDER_WGSL = /* wgsl */ `
struct Cam {
    vp: mat4x4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    misc: vec4<f32>,   // x = particle radius, y = debug normalisation reciprocal, z = opacity
    tint: vec4<f32>,   // rgb = base particle colour (settable, e.g. blue liquid / tan sand)
};
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> dbg: array<f32>;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    let center = positions[ii].xyz;
    let r = cam.misc.x;
    let world = center + cam.right.xyz * (c.x * r) + cam.up.xyz * (c.y * r);
    var o: VOut;
    o.clip = cam.vp * vec4<f32>(world, 1.0);
    o.uv = c;
    // Colour by speed from the settable base tint: settled particles use the base colour, fast-moving
    // splashes brighten toward white by cam.tint.w (0 = no brightening, e.g. sand; 1 = full, e.g. water).
    let t = clamp(dbg[ii] * cam.misc.y, 0.0, 1.0);
    let base = cam.tint.rgb;
    let hot = mix(base, min(base + vec3<f32>(0.5), vec3<f32>(1.0)), cam.tint.w);
    o.color = mix(base, hot, t);
    return o;
}

@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
    let r2 = dot(i.uv, i.uv);
    if (r2 > 1.0) { discard; }
    let nz = sqrt(1.0 - r2);
    let n = normalize(vec3<f32>(i.uv, nz));
    let L = normalize(vec3<f32>(0.4, 0.7, 0.6));
    let V = vec3<f32>(0.0, 0.0, 1.0);
    let H = normalize(L + V);
    let diff = max(dot(n, L), 0.0);
    let spec = pow(max(dot(n, H), 0.0), 48.0);   // wet Blinn-Phong highlight
    let fres = pow(1.0 - nz, 3.0);               // fresnel rim toward the silhouette
    var col = i.color * (0.32 + 0.68 * diff);
    col += vec3<f32>(0.9, 0.97, 1.0) * (spec * 0.6);
    col += vec3<f32>(0.45, 0.65, 0.95) * (fres * 0.25);
    return vec4<f32>(col, cam.misc.z);
}`;

const DEFAULT_BLEND: GPUBlendState = {
    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
};

interface CustomShaderState {
    code: string;
    vertexEntryPoint: string;
    fragmentEntryPoint: string;
    customUniforms: Uint8Array | null;
    blend: GPUBlendState | null;
    depthWriteEnabled: boolean;
    depthCompare: GPUCompareFunction;
}

function copyCustomUniforms(data: ArrayBufferView): Uint8Array {
    if (data.byteLength === 0 || data.byteLength % 4 !== 0) {
        throw new Error("Particle renderer custom uniforms must have a non-zero byte length divisible by 4.");
    }
    return new Uint8Array(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
}

export function createParticleRenderTask(engine: EngineContext, scene: SceneContext, opts: ParticleRenderOptions): ParticleRenderTask {
    const device = engine._device;
    const { colorRT, depthRT, camera } = opts;
    let currentSim = opts.sim;
    let enabled = true;
    let opacity = 1;
    let sizeScale = 1; // user-controlled visual particle-size multiplier
    const tint: [number, number, number] = [0.1, 0.35, 0.85]; // base particle colour (deep-blue liquid default)
    let velocityBrighten = 1; // how much fast particles brighten toward white (0 = uniform, e.g. sand)

    const camData = new Float32Array(32); // mat4 (16) + right (4) + up (4) + misc (4) + tint (4)
    const camBuffer = device.createBuffer({
        label: "fluid-particle-cam",
        size: camData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let pipeline: GPURenderPipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;
    let customShader: CustomShaderState | null = null;
    let customUniformBuffer: GPUBuffer | null = null;

    function buildBindGroup(): void {
        if (!pipeline) {
            return;
        }
        bindGroup = device.createBindGroup({
            label: "fluid-particles",
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: camBuffer } },
                { binding: 1, resource: { buffer: currentSim.positionBuffer } },
                { binding: 2, resource: { buffer: currentSim.debugBuffer } },
                ...(customUniformBuffer ? [{ binding: 3, resource: { buffer: customUniformBuffer } }] : []),
            ],
        });
    }

    function build(): void {
        if (pipeline) {
            return;
        }
        const shader = customShader;
        const module = device.createShaderModule({ label: "fluid-particles", code: shader?.code ?? RENDER_WGSL });
        const blend = shader ? shader.blend : DEFAULT_BLEND;
        const target: GPUColorTargetState = { format: engine.format };
        if (blend) {
            target.blend = blend;
        }
        let layout: GPUPipelineLayout | "auto" = "auto";
        if (shader) {
            const visibility = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
            const entries: GPUBindGroupLayoutEntry[] = [
                { binding: 0, visibility, buffer: { type: "uniform" } },
                { binding: 1, visibility, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility, buffer: { type: "read-only-storage" } },
            ];
            if (shader.customUniforms) {
                entries.push({ binding: 3, visibility, buffer: { type: "uniform" } });
            }
            const bindGroupLayout = device.createBindGroupLayout({
                label: "fluid-particle-custom-bindings",
                entries,
            });
            layout = device.createPipelineLayout({
                label: "fluid-particle-custom-pipeline-layout",
                bindGroupLayouts: [bindGroupLayout],
            });
        }
        pipeline = device.createRenderPipeline({
            label: "fluid-particles",
            layout,
            vertex: { module, entryPoint: shader?.vertexEntryPoint ?? "vs" },
            fragment: {
                module,
                entryPoint: shader?.fragmentEntryPoint ?? "fs",
                targets: [target],
            },
            primitive: { topology: "triangle-list", cullMode: "none" },
            depthStencil: {
                format: depthRT._descriptor.dFormat!,
                depthWriteEnabled: shader?.depthWriteEnabled ?? true,
                depthCompare: shader?.depthCompare ?? "greater-equal", // reverse-Z, matching the scene render task
            },
            multisample: { count: depthRT._descriptor.samples },
        });
        if (shader?.customUniforms) {
            customUniformBuffer = device.createBuffer({
                label: "fluid-particle-custom-uniforms",
                size: shader.customUniforms.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(customUniformBuffer, 0, shader.customUniforms);
        }
        buildBindGroup();
    }

    function updateCamera(): void {
        const aspect = getEffectiveAspectRatio(camera, engine.canvas.width, engine.canvas.height);
        const vp = getViewProjectionMatrix(camera, aspect);
        for (let k = 0; k < 16; k++) {
            camData[k] = vp[k]!;
        }
        const wm = camera.worldMatrix;
        // Camera world-space basis vectors for billboarding.
        camData[16] = wm[0]!;
        camData[17] = wm[1]!;
        camData[18] = wm[2]!;
        camData[19] = 0;
        camData[20] = wm[4]!;
        camData[21] = wm[5]!;
        camData[22] = wm[6]!;
        camData[23] = 0;
        camData[24] = currentSim.particleRadius * sizeScale;
        camData[25] = currentSim.debugNorm;
        camData[26] = opacity;
        camData[27] = 0;
        camData[28] = tint[0];
        camData[29] = tint[1];
        camData[30] = tint[2];
        camData[31] = velocityBrighten;
        device.queue.writeBuffer(camBuffer, 0, camData);
    }

    function executeBuilt(): number {
        if (!enabled || opacity <= 0) {
            return 0;
        }
        const colorView = colorRT._colorView;
        const depthView = depthRT._depthView;
        if (!pipeline || !bindGroup || !colorView || !depthView) {
            return 0;
        }
        updateCamera();
        engine._currentEncoder.pushDebugGroup("Fluid particles (spheres)");
        const pass = engine._currentEncoder.beginRenderPass({
            label: "fluid-particles",
            colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
            depthStencilAttachment: { view: depthView, depthLoadOp: "load", depthStoreOp: "store" },
            timestampWrites: profiler?.pass("Particles"),
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(6, currentSim.renderCount ?? currentSim.count);
        pass.end();
        engine._currentEncoder.popDebugGroup();
        return 1;
    }

    function executeNeedsBuild(): number {
        build();
        task.execute = executeBuilt;
        return executeBuilt();
    }

    const task: ParticleRenderTask = {
        name: "fluid-particles",
        engine,
        scene,
        _passes: [],
        /** Switch the rendered simulation backend (rebinds to its buffers). */
        setSim(s: FluidSim): void {
            currentSim = s;
            buildBindGroup();
        },
        /** Enable/disable this renderer (so the demo can swap to surface mode). */
        setEnabled(on: boolean): void {
            enabled = on;
        },
        /** Fade every rendered sphere without changing simulation state. */
        setOpacity(v: number): void {
            opacity = Math.max(0, Math.min(1, v));
        },
        /** Visual particle-size multiplier (does not affect the physics). */
        setSizeScale(s: number): void {
            sizeScale = s;
        },
        /** Base particle colour (rgb 0..1). Fast particles brighten toward white from this. */
        setTint(rgb: [number, number, number]): void {
            tint[0] = rgb[0];
            tint[1] = rgb[1];
            tint[2] = rgb[2];
        },
        /** How much fast particles brighten toward white (0 = uniform colour, e.g. sand; 1 = water). */
        setVelocityBrighten(v: number): void {
            velocityBrighten = v;
        },
        setProfiler(p: FluidProfiler | null): void {
            profiler = p;
        },
        setShader(shader: ParticleRenderShaderOptions | null): void {
            customShader = shader
                ? {
                      code: shader.code,
                      vertexEntryPoint: shader.vertexEntryPoint ?? "vs",
                      fragmentEntryPoint: shader.fragmentEntryPoint ?? "fs",
                      customUniforms: shader.customUniforms ? copyCustomUniforms(shader.customUniforms) : null,
                      blend: shader.blend === undefined ? DEFAULT_BLEND : shader.blend,
                      depthWriteEnabled: shader.depthWriteEnabled ?? true,
                      depthCompare: shader.depthCompare ?? "greater-equal",
                  }
                : null;
            customUniformBuffer?.destroy();
            customUniformBuffer = null;
            pipeline = null;
            bindGroup = null;
            task.execute = executeNeedsBuild;
        },
        setCustomUniforms(data: ArrayBufferView): void {
            if (!customShader?.customUniforms) {
                throw new Error("The active particle shader does not define a custom uniform payload.");
            }
            if (data.byteLength !== customShader.customUniforms.byteLength) {
                throw new Error(`Particle renderer custom uniform update is ${data.byteLength} bytes; expected ${customShader.customUniforms.byteLength}.`);
            }
            customShader.customUniforms.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
            if (customUniformBuffer) {
                device.queue.writeBuffer(customUniformBuffer, 0, customShader.customUniforms);
            }
        },
        record(): void {
            build();
            task.execute = executeBuilt;
        },
        execute: executeBuilt,
        dispose(): void {
            customUniformBuffer?.destroy();
            camBuffer.destroy();
        },
    };
    return task;
}
