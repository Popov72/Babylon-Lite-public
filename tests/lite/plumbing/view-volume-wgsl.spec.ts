import type { Page } from "@playwright/test";
import { test, expect } from "../parity/parity-fixtures";
import {
    createViewVolumeGrid,
    getViewVolumeSliceBounds,
    viewDepthToVolumeSlice,
    volumeSliceToViewDepth,
    type ViewVolumeDepthMapping,
    type ViewVolumeGrid,
} from "../../../packages/babylon-lite/src/render/volume/view-volume-grid";
import { buildViewVolumeDepthWgsl } from "../../../packages/babylon-lite/src/render/volume/view-volume-grid-wgsl";
import { integrateHomogeneousMedium, type OpticalRgb } from "../../../packages/babylon-lite/src/render/volume/optical-transfer";
import { OPTICAL_TRANSFER_WGSL } from "../../../packages/babylon-lite/src/render/volume/optical-transfer-wgsl";

type Vec4 = readonly [number, number, number, number];
type ShaderSample = readonly [Vec4, Vec4];

const DEPTH_BODY = `
let i=id.x*2u;
let p=inputs[i];
let q=inputs[i+1u];
outputs[i]=vec4f(
    viewVolumeDepthToSlice(p.x,p.y,p.z,p.w),
    viewVolumeSliceToDepth(q.x,p.y,p.z,p.w),
    viewVolumeSliceThickness(u32(q.y),p.y,p.z,u32(p.w)),
    0.0);
`;

const OPTICAL_BODY = `
let i=id.x*2u;
let p=inputs[i];
let q=inputs[i+1u];
let integrated=integrateHomogeneousMedium(p.xyz,q.xyz,p.w);
outputs[i]=vec4f(integrated.radiance,0.0);
outputs[i+1u]=vec4f(integrated.transmittance,0.0);
`;

test.beforeEach(async ({ page }) => {
    await page.goto("/", { waitUntil: "domcontentloaded" });
});

for (const mapping of [{ kind: "linear" }, { kind: "log" }, { kind: "power", exponent: 2 }] as const satisfies readonly ViewVolumeDepthMapping[]) {
    test(`${mapping.kind} WGSL maps and clamps view depths like the CPU`, async ({ page }) => {
        const standard = createViewVolumeGrid({
            targetWidth: 1,
            targetHeight: 1,
            tileSize: 1,
            depthSlices: 64,
            nearDepth: 0.1,
            farDepth: 1000,
            depthMapping: mapping,
        });
        const narrow = createViewVolumeGrid({
            targetWidth: 1,
            targetHeight: 1,
            tileSize: 1,
            depthSlices: 64,
            nearDepth: 1e-6,
            farDepth: 1.5e-6,
            depthMapping: mapping,
        });
        const cases: { grid: ViewVolumeGrid; depth: number; slice: number; thicknessIndex: number }[] = [
            { grid: standard, depth: 0.05, slice: -1, thicknessIndex: 0 },
            { grid: standard, depth: 0.1, slice: 0, thicknessIndex: 0 },
            { grid: standard, depth: 10, slice: 12.5, thicknessIndex: 12 },
            { grid: standard, depth: 1000, slice: 64, thicknessIndex: 63 },
            { grid: standard, depth: 2000, slice: 65, thicknessIndex: 63 },
            { grid: narrow, depth: 0, slice: -1, thicknessIndex: 0 },
            { grid: narrow, depth: 1e-6, slice: 0, thicknessIndex: 0 },
            { grid: narrow, depth: 1.5e-6, slice: 64, thicknessIndex: 63 },
            { grid: narrow, depth: 2e-6, slice: 65, thicknessIndex: 63 },
        ];
        const samples: ShaderSample[] = cases.map(({ grid, depth, slice, thicknessIndex }) => [
            [depth, grid.nearDepth, grid.farDepth, grid.depth],
            [slice, thicknessIndex, 0, 0],
        ]);
        const result = await runVolumeShader(page, buildViewVolumeDepthWgsl(mapping), DEPTH_BODY, samples);
        for (let i = 0; i < cases.length; i++) {
            const { grid, depth, slice, thicknessIndex } = cases[i]!;
            const offset = i * 8;
            expectF32Near(result[offset]!, viewDepthToVolumeSlice(grid, depth));
            expectF32Near(result[offset + 1]!, volumeSliceToViewDepth(grid, slice));
            expectF32Near(result[offset + 2]!, getViewVolumeSliceBounds(grid, thicknessIndex).thickness);
        }
    });
}

test("optical WGSL matches the CPU through thin, long, and threshold-crossing media", async ({ page }) => {
    const cases: { source: OpticalRgb; extinction: OpticalRgb; distance: number }[] = [
        { source: [1, 1, 1], extinction: [1e-6, 1e-6, 1e-6], distance: 1e6 },
        { source: [1, 1, 1], extinction: [1e-6, 1e-6, 1e-6], distance: 5e5 },
        { source: [1, 1, 1], extinction: [1e-3, 1e-3, 1e-3], distance: 1e-7 },
        { source: [2, 3, 4], extinction: [0, 0, 0], distance: 5 },
        { source: [1, 2, 3], extinction: [0.0499, 0.05, 0.0501], distance: 1 },
        { source: [1, 1, 1], extinction: [0.049999, 0.05, 0.050001], distance: 1 },
        { source: [1, 2, 3], extinction: [1e-4, 0.005, 0.5], distance: 10 },
    ];
    const samples: ShaderSample[] = cases.map(({ source, extinction, distance }) => [
        [source[0], source[1], source[2], distance],
        [extinction[0], extinction[1], extinction[2], 0],
    ]);
    const result = await runVolumeShader(page, OPTICAL_TRANSFER_WGSL, OPTICAL_BODY, samples);
    for (let i = 0; i < cases.length; i++) {
        const { source, extinction, distance } = cases[i]!;
        const expected = integrateHomogeneousMedium(source, extinction, distance);
        for (let channel = 0; channel < 3; channel++) {
            expectF32Near(result[i * 8 + channel]!, expected.radiance[channel]!);
            expectF32Near(result[i * 8 + 4 + channel]!, expected.transmittance[channel]!);
        }
    }
    const whole = result[0]!;
    const half = result[8]!;
    const halfTransmittance = result[12]!;
    expectF32Near(whole, half * (1 + halfTransmittance));
});

function expectF32Near(actual: number, expected: number): void {
    expect(Number.isFinite(actual)).toBe(true);
    expect(Math.abs(actual - expected)).toBeLessThanOrEqual(Math.max(1e-11, Math.abs(expected) * 1e-4));
}

async function runVolumeShader(page: Page, functions: string, body: string, samples: readonly ShaderSample[]): Promise<number[]> {
    const input = new Float32Array(samples.length * 8);
    for (let i = 0; i < samples.length; i++) {
        input.set(samples[i]![0], i * 8);
        input.set(samples[i]![1], i * 8 + 4);
    }
    return page.evaluate(
        async ({ functions, body, input }) => {
            const adapter = await navigator.gpu?.requestAdapter();
            if (!adapter) {
                throw new Error("View-volume WGSL tests require a WebGPU adapter.");
            }
            const device = await adapter.requestDevice();
            let source: GPUBuffer | null = null;
            let output: GPUBuffer | null = null;
            let readback: GPUBuffer | null = null;
            try {
                const shader = device.createShaderModule({
                    code: `${functions}
@group(0) @binding(0) var<storage,read> inputs:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> outputs:array<vec4f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u) {
${body}
}`,
                });
                const compilation = await shader.getCompilationInfo();
                const errors = compilation.messages.filter((message) => message.type === "error");
                if (errors.length > 0) {
                    throw new Error(errors.map((message) => message.message).join("\n"));
                }
                const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
                const bytes = input.length * 4;
                source = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                output = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
                device.queue.writeBuffer(source, 0, new Float32Array(input));
                const group = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: source } },
                        { binding: 1, resource: { buffer: output } },
                    ],
                });
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, group);
                pass.dispatchWorkgroups(input.length / 8);
                pass.end();
                encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
                device.queue.submit([encoder.finish()]);
                await readback.mapAsync(GPUMapMode.READ);
                return Array.from(new Float32Array(readback.getMappedRange()));
            } finally {
                if (readback?.mapState === "mapped") {
                    readback.unmap();
                }
                readback?.destroy();
                output?.destroy();
                source?.destroy();
                device.destroy();
            }
        },
        { functions, body, input: Array.from(input) }
    );
}
