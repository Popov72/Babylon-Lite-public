// Cubemap skybox for the fluid demo.
//
// Renders an environment cube map as the background: a fullscreen pass
// reconstructs the per-pixel world-space view ray from the camera basis and
// samples the cube. It draws FIRST into the offscreen scene-colour target, so
// the scene geometry — and the refracting fluid surface, which samples that
// target — sit against the real sky. The same cube is sampled by the fluid
// surface pass for reflections (see fluid-surface-render.ts).
//
// The cube texture is loaded asynchronously (loadCubeTexture); until it arrives
// a 1×1 sky-blue placeholder is sampled so the layout/pipeline never change.

import type { Camera, EngineContext, RenderTarget, SceneContext, Task } from "babylon-lite";

export interface SkyOptions {
    /** Target to render the sky into (the offscreen scene colour). */
    targetRT: RenderTarget;
    camera: Camera;
}

export interface EnvMap {
    view: GPUTextureView;
    sampler: GPUSampler;
}

const SKY_WGSL = /* wgsl */ `
struct Sky {
    right: vec4<f32>,
    up: vec4<f32>,
    fwd: vec4<f32>,
    params: vec4<f32>, // tanHalfFov, aspect, 1/width, 1/height
};
@group(0) @binding(0) var<uniform> s: Sky;
@group(0) @binding(1) var envTex: texture_cube<f32>;
@group(0) @binding(2) var envSamp: sampler;

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
    // Babylon authors cube faces for a left-handed frame; flip Z to match.
    return vec4<f32>(textureSampleLevel(envTex, envSamp, vec3<f32>(dir.x, dir.y, -dir.z), 0.0).rgb, 1.0);
}`;

/** A 1×1 sky-blue cube used until the real environment finishes loading. */
export function createPlaceholderCube(device: GPUDevice): GPUTextureView {
    const tex = device.createTexture({ label: "env-placeholder", size: [1, 1, 6], format: "rgba8unorm", dimension: "2d", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const px = new Uint8Array([110, 150, 210, 255]);
    for (let i = 0; i < 6; i++) {
        device.queue.writeTexture({ texture: tex, origin: [0, 0, i] }, px, { bytesPerRow: 4 }, [1, 1, 1]);
    }
    return tex.createView({ dimension: "cube" });
}

/** Load a 6-face cube map (`<base>_px<ext>` … `_nz<ext>`) into a GPU cube texture.
 *  Level-0 only (the demo samples it with an explicit LOD of 0, so no mips). */
export async function loadEnvCube(engine: EngineContext, baseUrl: string, ext = ".jpg"): Promise<EnvMap> {
    const device = engine._device;
    const faces = ["_px", "_nx", "_py", "_ny", "_pz", "_nz"];
    const bitmaps = await Promise.all(
        faces.map(async (s) => {
            const r = await fetch(`${baseUrl}${s}${ext}`);
            if (!r.ok) {
                throw new Error(`cube face load failed: ${baseUrl}${s}${ext}`);
            }
            return createImageBitmap(await r.blob(), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
        }),
    );
    const sz = bitmaps[0]!.width;
    const tex = device.createTexture({ label: "env-cube", size: [sz, sz, 6], format: "rgba8unorm", dimension: "2d", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    for (let i = 0; i < 6; i++) {
        device.queue.copyExternalImageToTexture({ source: bitmaps[i]! }, { texture: tex, origin: [0, 0, i] }, [sz, sz, 1]);
        bitmaps[i]!.close();
    }
    const sampler = device.createSampler({ label: "env-samp", magFilter: "linear", minFilter: "linear" });
    return { view: tex.createView({ dimension: "cube" }), sampler };
}

export function createSkyTask(engine: EngineContext, scene: SceneContext, opts: SkyOptions): Task & { setEnvMap(e: EnvMap): void } {
    const device = engine._device;
    const { targetRT, camera } = opts;

    const data = new Float32Array(16);
    const buffer = device.createBuffer({ label: "sky-uniform", size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const fallbackSampler = device.createSampler({ label: "sky-samp", magFilter: "linear", minFilter: "linear", mipmapFilter: "linear" });
    let envView: GPUTextureView = createPlaceholderCube(device);
    let envSampler: GPUSampler = fallbackSampler;

    let pipeline: GPURenderPipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;

    function buildBindGroup(): void {
        if (!pipeline) {
            return;
        }
        bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer } },
                { binding: 1, resource: envView },
                { binding: 2, resource: envSampler },
            ],
        });
    }

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
        buildBindGroup();
    }

    function updateUniform(): void {
        const wm = camera.worldMatrix;
        data[0] = wm[0]!; data[1] = wm[1]!; data[2] = wm[2]!; data[3] = 0;
        data[4] = wm[4]!; data[5] = wm[5]!; data[6] = wm[6]!; data[7] = 0;
        data[8] = wm[8]!; data[9] = wm[9]!; data[10] = wm[10]!; data[11] = 0;
        data[12] = Math.tan(camera.fov * 0.5);
        data[13] = engine.canvas.width / Math.max(1, engine.canvas.height);
        data[14] = 1 / engine.canvas.width;
        data[15] = 1 / engine.canvas.height;
        device.queue.writeBuffer(buffer, 0, data);
    }

    return {
        name: "sky",
        engine,
        scene,
        _passes: [],
        setEnvMap(e: EnvMap): void {
            envView = e.view;
            envSampler = e.sampler;
            buildBindGroup();
        },
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
