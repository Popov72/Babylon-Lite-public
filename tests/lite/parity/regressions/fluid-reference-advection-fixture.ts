export async function runReferenceAdvectionProbe(advection: string) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
        throw new Error("WebGPU is required for the Reference advection regression.");
    }
    const device = await adapter.requestDevice();
    const cases = [
        { name: "ordinary-rk3", p: [5, 5, 5, 0.1], v: [1, 0.5, -0.25, 1], field: [0, 0, 0, 0], mode: [0, 0, 256, 1], expected: [5.1, 5.05, 4.975], error: -1 },
        { name: "ordinary-rk2", p: [5, 5, 5, 0.1], v: [1, 0.5, -0.25, 1], field: [0, 0, 0, 0], mode: [0, 0, 256, 0], expected: [5.1, 5.05, 4.975], error: -1 },
        { name: "ordinary-nonlinear-rk3", p: [5, 5, 5, 0.1], v: [0, 0, 0, 1], field: [0, 0, 20, 6], mode: [1, 0, 256, 1], expected: [6 + 1 / 3, 5, 5], error: -1 },
        { name: "ordinary-nonlinear-rk2", p: [5, 5, 5, 0.1], v: [0, 0, 0, 1], field: [0, 0, 20, 6], mode: [1, 0, 256, 0], expected: [5, 5, 5], error: -1 },
        { name: "near-wall-overlong", p: [5, 5, 5, 0.01], v: [300, 0, 0, 0.1], field: [5.005, 0, 0, 0], mode: [0, 1, 256, 1], expected: [4.985, 5, 5], error: -1 },
        { name: "far-wall-within-budget", p: [5, 5, 5, 0.01], v: [300, 0, 0, 0.1], field: [7, 0, 0, 0], mode: [0, 1, 256, 1], expected: [6.98, 5, 5], error: -1 },
        { name: "thin-slab-overlong", p: [5, 5, 5, 0.01], v: [300, 0, 0, 0.1], field: [5.04, 0.012, 0, 0], mode: [0, 2, 256, 1], expected: [5.008, 5, 5], error: -1 },
        { name: "nonlinear-refinement", p: [5, 5, 5, 0.01], v: [0, 0, 0, 0.1], field: [0, 0, 400, 6], mode: [1, 0, 256, 1], expected: [6 - Math.exp(-4), 5, 5], error: -1 },
        { name: "discarded-unsupported-trial", p: [5, 5, 5, 0.01], v: [30, 0, 0, 0.01], field: [5.005, 5.02, 0, 0], mode: [2, 1, 256, 1], expected: [5.003, 5, 5], error: -1 },
        { name: "unsupported-origin", p: [5, 5, 5, 0.1], v: [1, 0, 0, 1], field: [0, 0, 0, 0], mode: [4, 0, 256, 1], expected: [5, 5, 5], error: 6 },
        { name: "nonfinite-velocity", p: [5, 5, 5, 0.1], v: [NaN, 0, 0, 1], field: [0, 0, 0, 0], mode: [3, 0, 256, 1], expected: [5, 5, 5], error: 0 },
        { name: "free-flight-budget", p: [5, 5, 5, 0.01], v: [300, 0, 0, 0.1], field: [0, 0, 0, 0], mode: [0, 0, 256, 1], expected: [5, 5, 5], error: 5 },
        { name: "exact-budget", p: [0, 5, 5, 0.1], v: [8, 0, 0, 1], field: [0, 0, 0, 0], mode: [0, 0, 8, 1], expected: [0.8, 5, 5], error: -1 },
        { name: "tiny-budget-early-collision", p: [5, 5, 5, 0.1], v: [1000, 0, 0, 1], field: [5.05, 0, 0, 0], mode: [0, 1, 1, 1], expected: [4.85, 5, 5], error: -1 },
        { name: "flat-gradient-contact", p: [5, 5, 5, 0.1], v: [1, 0, 0, 1], field: [0, 0, 0, 0], mode: [0, 3, 256, 1], expected: [5, 5, 5], error: 1 },
        { name: "nonprogressing-refinement", p: [5, 5, 5, 1], v: [0, 0, 0, 1], field: [0, 0, 1e10, 6], mode: [1, 0, 256, 1], expected: [5, 5, 5], error: 5 },
    ];
    const header = `
struct Params { grid: vec4<u32>, origin: vec4<f32>, geometry: vec4<f32>, settings: vec4<f32>, switches: vec4<u32> }
struct Case { p: vec4<f32>, v: vec4<f32>, field: vec4<f32>, mode: vec4<u32> }
struct Correction { position: vec3<f32>, velocity: vec3<f32>, contacted: bool }
struct SolidSample { distance: f32, gradient: vec3<f32>, velocity: vec3<f32> }
@group(0) @binding(0) var<storage, read> inputs: array<Case>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> lists: array<atomic<u32>>;
var<private> params: Params;
var<private> input: Case;
var<private> caseIndex: u32;
fn statusIndex(index: u32) -> u32 { return caseIndex * 16u + index; }
fn sampleVelocityWithSupport(point: vec3<f32>, previous: bool) -> vec4<f32> {
    atomicAdd(&lists[statusIndex(9u)], 1u);
    if (input.mode.x == 3u) { return vec4<f32>(input.v.xyz, 1.0); }
    if (input.mode.x == 4u || (input.mode.x == 2u && point.x > input.field.y)) { return vec4<f32>(0.0); }
    if (input.mode.x == 1u) { return vec4<f32>(input.field.z * (input.field.w - point.x), 0.0, 0.0, 1.0); }
    return vec4<f32>(input.v.xyz, 1.0);
}
fn sampleObstacle(point: vec3<f32>) -> SolidSample {
    atomicAdd(&lists[statusIndex(8u)], 1u);
    if (input.mode.y == 1u) { return SolidSample(input.field.x - point.x, vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(0.0)); }
    if (input.mode.y == 2u) { return SolidSample(abs(point.x - input.field.x) - input.field.y, vec3<f32>(sign(point.x - input.field.x), 0.0, 0.0), vec3<f32>(0.0)); }
    if (input.mode.y == 3u) { return SolidSample(-1.0, vec3<f32>(0.0), vec3<f32>(0.0)); }
    return SolidSample(1000.0, vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0));
}
fn reportCollisionFailure(point: vec3<f32>) { atomicOr(&lists[statusIndex(1u)], 1u); }
`;
    const entry = `
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    caseIndex = gid.x;
    input = inputs[gid.x];
    params = Params(vec4<u32>(1024u), vec4<f32>(0.0, 0.0, 0.0, input.v.w), vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 0.2 * input.v.w), vec4<u32>(0u, input.mode.w, input.mode.z, 0u));
    let first = sampleVelocityWithSupport(input.p.xyz, false);
    let result = advectSweptParticle(input.p.xyz, input.p.w, first, vec3<f32>(7.0, 8.0, 9.0));
    output[gid.x * 2u] = vec4<f32>(result.position, f32(result.contacted));
    output[gid.x * 2u + 1u] = vec4<f32>(result.velocity, 0.0);
}`;
    const owned: GPUBuffer[] = [];
    try {
        device.pushErrorScope("validation");
        const module = device.createShaderModule({ code: header + advection + entry });
        const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
        const inputBuffer = device.createBuffer({ size: cases.length * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const outputBuffer = device.createBuffer({ size: cases.length * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const flags = device.createBuffer({ size: cases.length * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const readback = device.createBuffer({ size: outputBuffer.size + flags.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        owned.push(inputBuffer, outputBuffer, flags, readback);
        const data = new Float32Array(cases.length * 16);
        const integers = new Uint32Array(data.buffer);
        for (let i = 0; i < cases.length; i++) {
            data.set(cases[i]!.p, i * 16);
            data.set(cases[i]!.v, i * 16 + 4);
            data.set(cases[i]!.field, i * 16 + 8);
            integers.set(cases[i]!.mode, i * 16 + 12);
        }
        device.queue.writeBuffer(inputBuffer, 0, data);
        const bindings = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [inputBuffer, outputBuffer, flags].map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindings);
        pass.dispatchWorkgroups(cases.length);
        pass.end();
        encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, outputBuffer.size);
        encoder.copyBufferToBuffer(flags, 0, readback, outputBuffer.size, flags.size);
        device.queue.submit([encoder.finish()]);
        const validation = await device.popErrorScope();
        if (validation) {
            throw new Error(validation.message);
        }
        await readback.mapAsync(GPUMapMode.READ);
        const mapped = readback.getMappedRange();
        const values = new Float32Array(mapped, 0, cases.length * 8);
        const counters = new Uint32Array(mapped, outputBuffer.size, cases.length * 16);
        const result = cases.map((item, i) => {
            const position = Array.from(values.slice(i * 8, i * 8 + 3));
            const velocity = Array.from(values.slice(i * 8 + 4, i * 8 + 7));
            const errors = Array.from(counters.slice(i * 16, i * 16 + 8));
            const collisionQueries = counters[i * 16 + 8]!;
            const velocityQueries = counters[i * 16 + 9]!;
            for (const field of [0, 1, 5, 6]) {
                if ((errors[field] !== 0) !== (item.error === field)) {
                    throw new Error(`${item.name}: wrong failure flags ${errors.join(",")}`);
                }
            }
            const tolerance = item.name === "nonlinear-refinement" ? 0.002 : 1e-5;
            if (position.some((value, axis) => !Number.isFinite(value) || Math.abs(value - item.expected[axis]!) > tolerance)) {
                throw new Error(`${item.name}: incorrect position ${position.join(",")}`);
            }
            if (velocity.some((value, axis) => value !== [7, 8, 9][axis])) {
                throw new Error(`${item.name}: advection changed the G2P velocity.`);
            }
            if (collisionQueries > item.mode[2]! + 4 || velocityQueries > 3 * item.mode[2]! + 1) {
                throw new Error(`${item.name}: unbounded collision or trial work.`);
            }
            return { name: item.name, position, errors, collisionQueries, velocityQueries };
        });
        readback.unmap();
        return result;
    } finally {
        for (const buffer of owned) {
            buffer.destroy();
        }
        device.destroy();
    }
}
