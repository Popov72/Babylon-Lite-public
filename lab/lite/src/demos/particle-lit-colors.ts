// Per-particle colours taken from the mesh as it is CURRENTLY RENDERED.
//
// The albedo path (see particle-uvs.ts) samples the base-colour texture, which is unlit: a dark
// metallic panel in a dim interior has an albedo several times brighter than the pixels it actually
// draws, so the water it melts into reads as a pale slab. The scene colour target already holds the
// mesh fully lit, exposed and tone-mapped, so the honest colour is simply "the pixel you can see".
//
// Particles that project onto the mesh's visible front surface take that pixel directly. Interior and
// back-facing particles have no visible pixel, so they keep their albedo scaled by the RGB lighting
// factor measured from the visible ones — texture detail everywhere, lit brightness everywhere.

const SCALE = 1024; // fixed-point factor for the atomic sums (u32-safe up to ~4M particles)

const WGSL = /* wgsl */ `
struct P {
    viewProj: mat4x4<f32>,
    view: mat4x4<f32>,
    misc: vec4<f32>, // proj[3].z, proj[2].z, count, depth tolerance (metres)
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> albedo: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outCol: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> sums: array<atomic<u32>>; // litRGB, albRGB
@group(0) @binding(5) var<storage, read_write> factor: array<f32>;
@group(0) @binding(6) var sceneColor: texture_2d<f32>;
@group(0) @binding(7) var sceneDepth: texture_depth_2d;

// Pass 1 — every particle that lands on the mesh's visible front surface takes that pixel.
@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u32(p.misc.z)) { return; }
    outCol[i] = vec4<f32>(0.0);
    let world = vec4<f32>(pos[i].xyz, 1.0);
    let clip = p.viewProj * world;
    if (clip.w <= 0.0) { return; }
    let ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) { return; }
    let dims = vec2<f32>(textureDimensions(sceneDepth));
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let coord = vec2<i32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * dims - vec2<f32>(0.5));
    let sceneNdc = textureLoad(sceneDepth, clamp(coord, vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1)), 0);
    if (sceneNdc <= 0.0) { return; }                       // reverse-Z: 0 = far, nothing drawn here
    let sceneEye = p.misc.x / (sceneNdc - p.misc.y);       // depth buffer -> eye-space Z
    let particleEye = (p.view * world).z;
    // Only the visible surface counts: a particle must BE (within tolerance) the nearest surface at
    // that pixel. One-sided rejection would let particles floating in front of the mesh sample whatever
    // is behind them.
    if (abs(particleEye - sceneEye) > p.misc.w) { return; }
    let cdims = vec2<f32>(textureDimensions(sceneColor));
    let ccoord = vec2<i32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * cdims - vec2<f32>(0.5));
    let lit = textureLoad(sceneColor, clamp(ccoord, vec2<i32>(0), vec2<i32>(cdims) - vec2<i32>(1)), 0).rgb;
    outCol[i] = vec4<f32>(lit, 1.0);
    let a = albedo[i].rgb;
    atomicAdd(&sums[0], u32(clamp(lit.r, 0.0, 4.0) * ${SCALE}.0));
    atomicAdd(&sums[1], u32(clamp(lit.g, 0.0, 4.0) * ${SCALE}.0));
    atomicAdd(&sums[2], u32(clamp(lit.b, 0.0, 4.0) * ${SCALE}.0));
    atomicAdd(&sums[3], u32(clamp(a.r, 0.0, 4.0) * ${SCALE}.0));
    atomicAdd(&sums[4], u32(clamp(a.g, 0.0, 4.0) * ${SCALE}.0));
    atomicAdd(&sums[5], u32(clamp(a.b, 0.0, 4.0) * ${SCALE}.0));
}

// Pass 2 — mean(lit) / mean(albedo) over the visible particles. Falls back to 1 when nothing was
// visible (mesh fully occluded / off-screen), which leaves the albedo path untouched.
@compute @workgroup_size(1)
fn reduce() {
    for (var c = 0u; c < 3u; c = c + 1u) {
        let lit = f32(atomicLoad(&sums[c]));
        let alb = f32(atomicLoad(&sums[c + 3u]));
        factor[c] = select(1.0, clamp(lit / alb, 0.0, 4.0), alb > 0.5);
    }
}

// Pass 3 — hidden particles keep their own texture colour, scaled into the lit range.
@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u32(p.misc.z)) { return; }
    if (outCol[i].a > 0.5) { return; }
    let f = vec3<f32>(factor[0], factor[1], factor[2]);
    outCol[i] = vec4<f32>(clamp(albedo[i].rgb * f, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}`;

/** Lazily-built pipelines + layout, cached per device. */
interface LitColorKit {
    device: GPUDevice;
    bgl: GPUBindGroupLayout;
    classify: GPUComputePipeline;
    reduce: GPUComputePipeline;
    fill: GPUComputePipeline;
}
let _kit: LitColorKit | null = null;

function getKit(device: GPUDevice): LitColorKit {
    if (_kit && _kit.device === device) return _kit;
    const bgl = device.createBindGroupLayout({
        label: "lit-color-bgl",
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "depth" } },
        ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const module = device.createShaderModule({ label: "lit-color", code: WGSL });
    _kit = {
        device,
        bgl,
        classify: device.createComputePipeline({ label: "lit-color-classify", layout, compute: { module, entryPoint: "classify" } }),
        reduce: device.createComputePipeline({ label: "lit-color-reduce", layout, compute: { module, entryPoint: "reduce" } }),
        fill: device.createComputePipeline({ label: "lit-color-fill", layout, compute: { module, entryPoint: "fill" } }),
    };
    return _kit;
}

/** Scene targets the liquefied mesh was last drawn into. */
export interface LitColorScene {
    colorView: GPUTextureView;
    depthView: GPUTextureView;
    /** Column-major view and view-projection matrices used for that draw. */
    view: ArrayLike<number>;
    viewProj: ArrayLike<number>;
    /** Projection matrix entries [3][2] and [2][2] — reverse-Z depth → eye-space Z. */
    projZW: number;
    projZZ: number;
}

/**
 * Replace per-particle albedo with the mesh's CURRENTLY RENDERED colours.
 *
 * @param device GPU device.
 * @param count particle count.
 * @param positions xyz per particle, world space (same space the scene was drawn in).
 * @param albedoBuf vec4 per particle — the gamma-encoded albedo from the UV-sampling path.
 * @param scene the colour/depth targets the mesh was drawn into, plus that draw's matrices.
 * @returns a new vec4-per-particle colour buffer. The caller owns it; `albedoBuf` is left untouched.
 */
export function buildLitParticleColors(device: GPUDevice, count: number, positions: Float32Array, albedoBuf: GPUBuffer, scene: LitColorScene, depthTolerance = 0.08): GPUBuffer {
    const kit = getKit(device);
    const out = device.createBuffer({ label: "lit-color-out", size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const pos4 = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
        pos4[i * 4] = positions[i * 3]!;
        pos4[i * 4 + 1] = positions[i * 3 + 1]!;
        pos4[i * 4 + 2] = positions[i * 3 + 2]!;
        pos4[i * 4 + 3] = 1;
    }
    const posBuf = device.createBuffer({ label: "lit-color-pos", size: pos4.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(posBuf, 0, pos4);

    const params = new Float32Array(36);
    params.set(scene.viewProj as ArrayLike<number> as never, 0);
    params.set(scene.view as ArrayLike<number> as never, 16);
    params[32] = scene.projZW;
    params[33] = scene.projZZ;
    params[34] = count;
    params[35] = depthTolerance;
    const paramBuf = device.createBuffer({ label: "lit-color-params", size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(paramBuf, 0, params);

    const sumsBuf = device.createBuffer({ label: "lit-color-sums", size: 24, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(sumsBuf, 0, new Uint32Array(6));
    const factorBuf = device.createBuffer({ label: "lit-color-factor", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(factorBuf, 0, new Float32Array([1, 1, 1, 0]));

    const bg = device.createBindGroup({
        layout: kit.bgl,
        entries: [
            { binding: 0, resource: { buffer: paramBuf } },
            { binding: 1, resource: { buffer: posBuf } },
            { binding: 2, resource: { buffer: albedoBuf } },
            { binding: 3, resource: { buffer: out } },
            { binding: 4, resource: { buffer: sumsBuf } },
            { binding: 5, resource: { buffer: factorBuf } },
            { binding: 6, resource: scene.colorView },
            { binding: 7, resource: scene.depthView },
        ],
    });

    const enc = device.createCommandEncoder({ label: "lit-color" });
    const groups = Math.ceil(count / 64);
    const pass = enc.beginComputePass({ label: "lit-color" });
    pass.setBindGroup(0, bg);
    pass.setPipeline(kit.classify);
    pass.dispatchWorkgroups(groups);
    pass.setPipeline(kit.reduce);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(kit.fill);
    pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([enc.finish()]);

    posBuf.destroy();
    paramBuf.destroy();
    sumsBuf.destroy();
    factorBuf.destroy();
    return out;
}
