// Procedural gradient skybox for the fluid demo.
//
// A self-contained background (no external cubemap assets): a fullscreen pass
// that reconstructs the per-pixel world-space view ray from the camera basis and
// shades a sky dome (zenith → horizon → ground gradient + a sun disc/halo). It
// renders FIRST into the offscreen scene-colour target, so the scene geometry —
// and the refracting fluid surface, which samples that target — sit against the
// sky. The scene render task must use `clr: false` so it preserves this.

import type { Camera, EngineContext, RenderTarget, SceneContext, Task } from "babylon-lite";

export interface SkyOptions {
    /** Target to render the sky into (the offscreen scene colour). */
    targetRT: RenderTarget;
    camera: Camera;
}

// World-space sun direction (points toward the sun).
const SUN_DIR: [number, number, number] = [0.45, 0.42, -0.78];

const SKY_WGSL = /* wgsl */ `
struct Sky {
    right: vec4<f32>,  // camera world basis
    up: vec4<f32>,
    fwd: vec4<f32>,
    params: vec4<f32>, // tanHalfFov, aspect, 1/width, 1/height
    sun: vec4<f32>,    // xyz sun dir
};
@group(0) @binding(0) var<uniform> s: Sky;

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ndcX = (pos.x + 0.5) * s.params.z * 2.0 - 1.0;
    let ndcY = 1.0 - (pos.y + 0.5) * s.params.w * 2.0;
    let dir = normalize(s.fwd.xyz
        + ndcX * s.params.x * s.params.y * s.right.xyz
        + ndcY * s.params.x * s.up.xyz);

    let zenith = vec3<f32>(0.12, 0.32, 0.72);
    let horizon = vec3<f32>(0.58, 0.73, 0.88);
    let ground = vec3<f32>(0.26, 0.29, 0.34);
    let h = dir.y;
    var col: vec3<f32>;
    if (h > 0.0) {
        col = mix(horizon, zenith, pow(clamp(h, 0.0, 1.0), 0.42));
    } else {
        col = mix(horizon, ground, clamp(-h * 3.0, 0.0, 1.0));
    }

    let sd = normalize(s.sun.xyz);
    let sun = max(dot(dir, sd), 0.0);
    col = col + vec3<f32>(1.0, 0.96, 0.86) * pow(sun, 400.0) * 2.0;  // disc
    col = col + vec3<f32>(1.0, 0.9, 0.72) * pow(sun, 12.0) * 0.18;   // halo
    return vec4<f32>(col, 1.0);
}`;

export function createSkyTask(engine: EngineContext, scene: SceneContext, opts: SkyOptions): Task {
    const device = engine._device;
    const { targetRT, camera } = opts;

    const data = new Float32Array(20);
    const buffer = device.createBuffer({ label: "sky-uniform", size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    let pipeline: GPURenderPipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;

    function build(): void {
        if (pipeline) {
            return;
        }
        const module = device.createShaderModule({ label: "sky", code: SKY_WGSL });
        pipeline = device.createRenderPipeline({
            label: "sky",
            layout: "auto",
            vertex: { module, entryPoint: "vs" },
            fragment: { module, entryPoint: "fs", targets: [{ format: engine.format }] },
            primitive: { topology: "triangle-list" },
        });
        bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
    }

    function updateUniform(): void {
        const wm = camera.worldMatrix;
        data[0] = wm[0]!; data[1] = wm[1]!; data[2] = wm[2]!; data[3] = 0;   // right
        data[4] = wm[4]!; data[5] = wm[5]!; data[6] = wm[6]!; data[7] = 0;   // up
        data[8] = wm[8]!; data[9] = wm[9]!; data[10] = wm[10]!; data[11] = 0; // forward
        const aspect = engine.canvas.width / Math.max(1, engine.canvas.height);
        data[12] = Math.tan(camera.fov * 0.5);
        data[13] = aspect;
        data[14] = 1 / engine.canvas.width;
        data[15] = 1 / engine.canvas.height;
        data[16] = SUN_DIR[0]; data[17] = SUN_DIR[1]; data[18] = SUN_DIR[2]; data[19] = 0;
        device.queue.writeBuffer(buffer, 0, data);
    }

    return {
        name: "sky",
        engine,
        scene,
        _passes: [],
        record(): void {
            build();
        },
        execute(): number {
            const view = targetRT._colorView;
            if (!pipeline || !bindGroup || !view) {
                return 0;
            }
            updateUniform();
            const pass = engine._currentEncoder.beginRenderPass({
                label: "sky",
                colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.draw(3);
            pass.end();
            return 1;
        },
        dispose(): void {
            buffer.destroy();
        },
    };
}
