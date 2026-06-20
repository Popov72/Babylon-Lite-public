// Particle renderer for the GPU fluid sim.
//
// Draws the GPU-resident particles as camera-facing billboarded sphere
// impostors, reading the simulation's position buffer directly in the vertex
// shader (no CPU round-trip, no per-frame vertex upload). Implemented as a
// custom frame-graph Task that draws into the scene's colour target while
// sharing the scene depth buffer, so particles depth-test against the ground
// (and, later, the tank) and occlude correctly.

import { getEffectiveAspectRatio, getViewProjectionMatrix } from "babylon-lite";
import type { Camera, EngineContext, RenderTarget, SceneContext, Task } from "babylon-lite";
import type { FluidSim } from "./pbf-sim.js";

export interface ParticleRenderOptions {
    /** Colour target the particles draw into (typically `engine.scRT`). */
    colorRT: RenderTarget;
    /** Depth target shared with the scene render task (load + test + write). */
    depthRT: RenderTarget;
    camera: Camera;
    sim: FluidSim;
}

const RENDER_WGSL = /* wgsl */ `
struct Cam {
    vp: mat4x4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    misc: vec4<f32>,   // x = particle radius, y = debug normalisation reciprocal
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
    // Colour by speed (Phase 3): slow/settled liquid is deep blue, fast-moving
    // splashes/foam tend toward bright cyan-white.
    let t = clamp(dbg[ii] * cam.misc.y, 0.0, 1.0);
    o.color = mix(vec3<f32>(0.10, 0.35, 0.85), vec3<f32>(0.85, 0.95, 1.0), t);
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
    return vec4<f32>(col, 1.0);
}`;

export function createParticleRenderTask(engine: EngineContext, scene: SceneContext, opts: ParticleRenderOptions): Task & { setSim(s: FluidSim): void; setEnabled(on: boolean): void } {
    const device = engine._device;
    const { colorRT, depthRT, camera } = opts;
    let currentSim = opts.sim;
    let enabled = true;

    const camData = new Float32Array(28); // mat4 (16) + right (4) + up (4) + misc (4)
    const camBuffer = device.createBuffer({
        label: "fluid-particle-cam",
        size: camData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let pipeline: GPURenderPipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;

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
            ],
        });
    }

    function build(): void {
        if (pipeline) {
            return;
        }
        const module = device.createShaderModule({ label: "fluid-particles", code: RENDER_WGSL });
        pipeline = device.createRenderPipeline({
            label: "fluid-particles",
            layout: "auto",
            vertex: { module, entryPoint: "vs" },
            fragment: { module, entryPoint: "fs", targets: [{ format: engine.format }] },
            primitive: { topology: "triangle-list", cullMode: "none" },
            depthStencil: {
                format: depthRT._descriptor.dFormat!,
                depthWriteEnabled: true,
                depthCompare: "greater-equal", // reverse-Z, matching the scene render task
            },
            multisample: { count: depthRT._descriptor.samples },
        });
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
        camData[24] = currentSim.particleRadius;
        camData[25] = currentSim.debugNorm;
        camData[26] = 0;
        camData[27] = 0;
        device.queue.writeBuffer(camBuffer, 0, camData);
    }

    return {
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
        record(): void {
            build();
        },
        execute(): number {
            if (!enabled) {
                return 0;
            }
            const colorView = colorRT._colorView;
            const depthView = depthRT._depthView;
            if (!pipeline || !bindGroup || !colorView || !depthView) {
                return 0;
            }
            updateCamera();
            const pass = engine._currentEncoder.beginRenderPass({
                label: "fluid-particles",
                colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
                depthStencilAttachment: { view: depthView, depthLoadOp: "load", depthStoreOp: "store" },
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.draw(6, currentSim.count);
            pass.end();
            return 1;
        },
        dispose(): void {
            camBuffer.destroy();
        },
    };
}
