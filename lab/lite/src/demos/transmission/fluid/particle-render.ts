// Particle renderer for the GPU fluid sim — Phase 1.
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
    // Colour by local neighbour density (Phase 2 grid validation): cool (sparse)
    // → warm (dense). A working grid shows the packed block hot and the spray cool.
    let t = clamp(dbg[ii] * cam.misc.y, 0.0, 1.0);
    o.color = mix(vec3<f32>(0.15, 0.45, 0.95), vec3<f32>(1.0, 0.85, 0.25), t);
    return o;
}

@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
    let r2 = dot(i.uv, i.uv);
    if (r2 > 1.0) { discard; }
    let nz = sqrt(1.0 - r2);
    let n = normalize(vec3<f32>(i.uv, nz));
    let l = max(dot(n, normalize(vec3<f32>(0.4, 0.7, 0.6))), 0.0);
    return vec4<f32>(i.color * (0.3 + 0.7 * l), 1.0);
}`;

export function createParticleRenderTask(engine: EngineContext, scene: SceneContext, opts: ParticleRenderOptions): Task {
    const device = engine._device;
    const { colorRT, depthRT, camera, sim } = opts;

    const camData = new Float32Array(28); // mat4 (16) + right (4) + up (4) + misc (4)
    const camBuffer = device.createBuffer({
        label: "fluid-particle-cam",
        size: camData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let pipeline: GPURenderPipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;

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
        bindGroup = device.createBindGroup({
            label: "fluid-particles",
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: camBuffer } },
                { binding: 1, resource: { buffer: sim.positionBuffer } },
                { binding: 2, resource: { buffer: sim.debugBuffer } },
            ],
        });
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
        camData[24] = sim.particleRadius;
        camData[25] = sim.debugNorm;
        camData[26] = 0;
        camData[27] = 0;
        device.queue.writeBuffer(camBuffer, 0, camData);
    }

    return {
        name: "fluid-particles",
        engine,
        scene,
        _passes: [],
        record(): void {
            build();
        },
        execute(): number {
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
            pass.draw(6, sim.count);
            pass.end();
            return 1;
        },
        dispose(): void {
            camBuffer.destroy();
        },
    };
}
