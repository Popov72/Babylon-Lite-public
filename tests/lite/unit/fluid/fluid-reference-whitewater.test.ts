import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { BU } from "../../../../packages/babylon-lite/src/engine/gpu-flags";
import { foamActiveStateBytes } from "../../../../packages/babylon-lite/src/fluid/core/sim-common";
import type { FlipReferenceGpuRuntime } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/gpu-runtime";
import type { FlipReferenceSimulation } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/types";
import { FLIP_REFERENCE_WHITEWATER_BINDINGS } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/whitewater-shaders";
import {
    createFlipReferenceWhitewater,
    disposeFlipReferenceWhitewater,
    planFlipReferenceWhitewater,
} from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/whitewater";

describe("FLIP Reference whitewater", () => {
    it("plans the exact working, publication, field, and bounded telemetry resources", () => {
        const plan = planFlipReferenceWhitewater(400, 8000, { poolScale: 3 });
        expect(plan.capacity).toBe(1200);
        expect(plan.errors).toEqual([]);
        expect(plan.resources).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ name: "Reference whitewater fields", bytes: 8000 * 32 }),
                expect.objectContaining({ name: "Reference whitewater working pool", bytes: 1200 * 32 }),
                expect.objectContaining({ name: "Reference whitewater working state", bytes: foamActiveStateBytes(1200) }),
                expect.objectContaining({ name: "Reference whitewater published pool", bytes: 1200 * 32 }),
                expect.objectContaining({ name: "Reference whitewater published state", bytes: foamActiveStateBytes(1200) }),
            ])
        );
        expect(plan.resources.filter((resource) => resource.name.startsWith("Reference whitewater telemetry"))).toHaveLength(2);
    });

    it("reports invalid settings and impossible device limits instead of silently substituting them", () => {
        const plan = planFlipReferenceWhitewater(
            400,
            8000,
            { energySpeedMin: 2, energySpeedMax: 1, kd: 2, poolCapMax: 100 },
            { maxBufferSize: 512, maxStorageBufferBindingSize: 512 }
        );
        expect(plan.errors.join(" ")).toContain("energySpeedMax");
        expect(plan.errors.join(" ")).toContain("foam.kd");
        expect(plan.errors.join(" ")).toContain("foam.poolCapMax");
        expect(plan.errors.join(" ")).toContain("minimum 1024-slot pool");
    });

    it("keeps every compute entry below the eight-storage-binding portability floor", () => {
        for (const bindings of Object.values(FLIP_REFERENCE_WHITEWATER_BINDINGS)) {
            expect(bindings.length).toBeLessThanOrEqual(8);
        }
    });

    it("rejects threshold intervals that collapse at shader precision", () => {
        const plan = planFlipReferenceWhitewater(400, 8000, {
            energySpeedMin: 1,
            energySpeedMax: 1 + Number.EPSILON,
            curvatureMin: 1,
            curvatureMax: 1 + Number.EPSILON,
            turbulenceMin: 1,
            turbulenceMax: 1 + Number.EPSILON,
        });
        expect(plan.errors).toEqual([
            "foam.energySpeedMax must be greater than foam.energySpeedMin.",
            "foam.curvatureMax must be greater than foam.curvatureMin.",
            "foam.turbulenceMax must be greater than foam.turbulenceMin.",
        ]);
    });

    it("initializes dispatch settings after clearing state and publishes an indirect-renderable pool", async () => {
        const writeBuffer = vi.fn<GPUQueue["writeBuffer"]>();
        const submit = vi.fn<GPUQueue["submit"]>();
        const createBuffer = vi.fn((descriptor: GPUBufferDescriptor) => ({
            label: descriptor.label ?? "",
            size: descriptor.size,
            usage: descriptor.usage,
            destroy: vi.fn(),
        }));
        const device = {
            limits: {
                maxStorageBuffersPerShaderStage: 8,
                maxComputeInvocationsPerWorkgroup: 64,
                maxComputeWorkgroupSizeX: 64,
                maxComputeWorkgroupsPerDimension: 8,
                maxBufferSize: 1048576,
                maxStorageBufferBindingSize: 1048576,
            } as GPUSupportedLimits,
            queue: { writeBuffer, submit } as unknown as GPUQueue,
            createBuffer: (descriptor: GPUBufferDescriptor) => createBuffer(descriptor) as unknown as GPUBuffer,
            pushErrorScope: vi.fn(),
            popErrorScope: () => Promise.resolve(null),
            createShaderModule: () => ({ getCompilationInfo: () => Promise.resolve({ messages: [] }) }) as unknown as GPUShaderModule,
            createComputePipeline: () => ({ getBindGroupLayout: () => ({}) as GPUBindGroupLayout }) as unknown as GPUComputePipeline,
            createBindGroup: () => ({}) as GPUBindGroup,
            createCommandEncoder: () =>
                ({
                    clearBuffer: vi.fn(),
                    beginComputePass: () =>
                        ({
                            setPipeline: vi.fn(),
                            setBindGroup: vi.fn(),
                            dispatchWorkgroups: vi.fn(),
                            end: vi.fn(),
                        }) as unknown as GPUComputePassEncoder,
                    finish: () => ({}) as GPUCommandBuffer,
                }) as unknown as GPUCommandEncoder,
        } as unknown as GPUDevice;
        const coreBuffer = device.createBuffer({ size: 256, usage: BU.STORAGE | BU.COPY_DST });
        const core = {
            _device: device,
            _disposed: false,
            _cells: 8,
            _ready: Promise.resolve(),
            _error: null,
            _uniformBuffer: coreBuffer,
            _positionBuffer: coreBuffer,
            _velocityBuffer: coreBuffer,
            _faceBuffer: coreBuffer,
            _cellBuffer: coreBuffer,
            _solidBuffer: coreBuffer,
            particleRadius: 0.05,
        } as FlipReferenceSimulation;
        const gpu = { core, disposed: false, stateBuffer: coreBuffer, ready: Promise.resolve(), error: null } as FlipReferenceGpuRuntime;
        const ww = createFlipReferenceWhitewater(core, gpu, 400, {});
        try {
            await ww.ready;
            const settingsIndex = writeBuffer.mock.calls.findIndex(([buffer, offset]) => buffer === ww._workingState && offset === 48);
            expect(settingsIndex).toBeGreaterThanOrEqual(0);
            expect(writeBuffer.mock.calls[settingsIndex]![2]).toEqual(new Uint32Array([1, 8]));
            expect(writeBuffer.mock.invocationCallOrder[settingsIndex]).toBeGreaterThan(submit.mock.invocationCallOrder[0]!);
            expect(ww.pool.drawIndirect!.usage & BU.INDIRECT).toBe(BU.INDIRECT);
            expect(ww.pool.activeIndices).not.toBe(ww._workingState);
            expect(ww.pool.activeIndicesOffset).toBe(256);
            expect(ww.bytes).toBe(ww._buffers.reduce((sum, buffer) => sum + buffer.size, 0));
        } finally {
            disposeFlipReferenceWhitewater(ww);
        }
        for (const buffer of createBuffer.mock.results.slice(1)) {
            expect(buffer.value.destroy).toHaveBeenCalledOnce();
        }
    });

    it("uses consumed runtime time, CAS free-slot allocation, and separate publication", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/experimental/flip-reference/whitewater-shaders.ts"), "utf8");
        const runtime = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/experimental/flip-reference/whitewater.ts"), "utf8");
        expect(source).toContain("atomicLoad(&runtime[19])");
        expect(source).toContain("atomicCompareExchangeWeak(&workingState[5]");
        expect(source).toContain("updateFlipReferenceWhitewater");
        expect(source).toContain("emitFlipReferenceWhitewater");
        expect(source).toContain("publishFlipReferenceWhitewater");
        expect(runtime.indexOf('indirect(ww, pass, "updateFlipReferenceWhitewater"')).toBeLessThan(runtime.indexOf('indirect(ww, pass, "emitFlipReferenceWhitewater"'));
    });
});
