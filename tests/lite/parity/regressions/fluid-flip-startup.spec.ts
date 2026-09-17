import { expect, test } from "@playwright/test";
import { FLIP_FACE_APERTURE_WGSL } from "../../../../packages/babylon-lite/src/fluid/solvers/flip-face-aperture";
import { FLIP_DENSITY_CORRECTION_WGSL } from "../../../../packages/babylon-lite/src/fluid/solvers/flip-density-correction";

test("production FLIP preserves solid-face topology and bounded startup density feedback", async ({ page }) => {
    await page.goto("/");
    const cases = [
        { phi: [-0.49, -0.9, -1, -2], dx: 1, expected: 0 },
        { phi: [-1e-8, -1e-8, -1e-8, -1e-8], dx: 1, expected: 0 },
        { phi: [0, 0, 0, 0], dx: 1, expected: 0 },
        { phi: [-0.01, 0, 0, 0], dx: 1, expected: 0 },
        { phi: [1e-8, 1e-8, 1e-8, 1e-8], dx: 1, expected: 1 },
        { phi: [0, 0, 0, 0.01], dx: 1, expected: 1 },
        { phi: [-0.25, 0.25, -0.25, 0.25], dx: 1, expected: 0.5 },
        { phi: [-1, -1, -1, 1], dx: 1, expected: 0.25 },
        { phi: [1, 1, 1, -1], dx: 1, expected: 0.75 },
        { phi: [-0.025, 0.025, -0.025, 0.025], dx: 0.1, expected: 0.5 },
    ];
    const densityCases = [
        { compression: 1, subDt: 1 / 120, expected: 12 },
        { compression: 1, subDt: 1 / 60, expected: 6 },
        { compression: 1, subDt: 0.000005, expected: 12 },
        { compression: 5, subDt: 0.000005, expected: 60 },
        { compression: 10, subDt: 0.000005, expected: 60 },
        { compression: 0, subDt: 0.000005, expected: 0 },
        { compression: 0.25, subDt: 0.001, expected: 3 },
    ];
    const values = await page.evaluate(
        async ({ wgsl, cases, densityCases }) => {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                throw new Error("WebGPU adapter unavailable for the production FLIP face-aperture regression.");
            }
            const device = await adapter.requestDevice();
            const buffers: GPUBuffer[] = [];
            try {
                device.pushErrorScope("validation");
                const module = device.createShaderModule({
                    code:
                        wgsl +
                        `
struct Input { phi: vec4<f32>, dimensions: vec4<f32> }
@group(0) @binding(0) var<storage, read> inputs: array<Input>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let input = inputs[id.x];
    output[id.x] = vec2<f32>(flipFaceAperture(input.phi, input.dimensions.x), flipDensityExpansion(input.dimensions.y, input.dimensions.z));
}`,
                });
                const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
                const input = device.createBuffer({ size: cases.length * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                const output = device.createBuffer({ size: cases.length * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                const readback = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
                buffers.push(input, output, readback);
                const data = new Float32Array(cases.length * 8);
                for (let index = 0; index < cases.length; index++) {
                    data.set(cases[index]!.phi, index * 8);
                    data[index * 8 + 4] = cases[index]!.dx;
                    const density = densityCases[index % densityCases.length]!;
                    data[index * 8 + 5] = density.compression;
                    data[index * 8 + 6] = density.subDt;
                }
                device.queue.writeBuffer(input, 0, data);
                const bindings = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: input } },
                        { binding: 1, resource: { buffer: output } },
                    ],
                });
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindings);
                pass.dispatchWorkgroups(cases.length);
                pass.end();
                encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
                device.queue.submit([encoder.finish()]);
                const validation = await device.popErrorScope();
                if (validation) {
                    throw new Error(validation.message);
                }
                await readback.mapAsync(GPUMapMode.READ);
                const result = Array.from(new Float32Array(readback.getMappedRange()));
                readback.unmap();
                return result;
            } finally {
                for (const buffer of buffers) {
                    buffer.destroy();
                }
                device.destroy();
            }
        },
        { wgsl: FLIP_FACE_APERTURE_WGSL + FLIP_DENSITY_CORRECTION_WGSL, cases, densityCases }
    );
    for (let index = 0; index < cases.length; index++) {
        expect(values[index * 2]).toBeCloseTo(cases[index]!.expected, 7);
        expect(values[index * 2 + 1]).toBeCloseTo(densityCases[index % densityCases.length]!.expected, 4);
    }
});
