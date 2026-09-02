import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { EngineContext } from "../../../../packages/babylon-lite/src/engine/engine.js";
import { createFlipSim, estimateFlipGpuBytes, resolveFlipPageLayout } from "../../../../packages/babylon-lite/src/fluid/solvers/flip-sim";
import { FLUID_FLOW_SHAPE_WGSL } from "../../../../packages/babylon-lite/src/fluid/core/sim-common";

const gpuGlobals = globalThis as Omit<typeof globalThis, "GPUBufferUsage" | "GPUTextureUsage"> & {
    GPUBufferUsage?: {
        STORAGE: number;
        COPY_DST: number;
        COPY_SRC: number;
        UNIFORM: number;
        MAP_READ: number;
        VERTEX: number;
        INDEX: number;
        INDIRECT: number;
    };
    GPUTextureUsage?: {
        TEXTURE_BINDING: number;
        STORAGE_BINDING: number;
    };
};
gpuGlobals.GPUBufferUsage ??= {
    STORAGE: 1,
    COPY_DST: 2,
    COPY_SRC: 4,
    UNIFORM: 8,
    MAP_READ: 16,
    VERTEX: 32,
    INDEX: 64,
    INDIRECT: 128,
} as unknown as GPUBufferUsage;
gpuGlobals.GPUTextureUsage ??= {
    TEXTURE_BINDING: 1,
    STORAGE_BINDING: 2,
} as unknown as GPUTextureUsage;

type MockLimits = Pick<GPUSupportedLimits, "maxBufferSize" | "maxStorageBufferBindingSize" | "maxTextureDimension2D">;

function makeBuffer(size: number): GPUBuffer {
    return {
        size,
        destroy: vi.fn(),
        mapAsync: vi.fn(async () => undefined),
        getMappedRange: vi.fn(() => new ArrayBuffer(size)),
        unmap: vi.fn(),
    } as unknown as GPUBuffer;
}

function makeEngine(limits: Partial<MockLimits> = {}): {
    engine: EngineContext;
    createBuffer: ReturnType<typeof vi.fn>;
    rejectPageStatusMaps: () => void;
    deferPageStatusMaps: () => void;
    resolvePendingPageStatusMaps: () => void;
    resolvePendingPageStatusMap: (index: number) => void;
    rejectPendingPageStatusMaps: () => void;
    setPageStatus: (completedFrame: number, required: number, overflowFrame: number) => void;
    setPageStatusSlot: (index: number, completedFrame: number, active: number, overflowFrame: number, required: number) => void;
    deferPressureMaps: () => void;
    resolvePendingPressureMap: (index: number) => void;
    setPressureStatus: (index: number, residual: number, rhs: number, divergence: number, cells: number) => void;
    pageStatusMapCount: () => number;
    polygonTriangleMapCount: () => number;
    computeDispatchCount: () => number;
} {
    let rejectPageStatus = false;
    let deferPageStatus = false;
    let deferPressure = false;
    const pendingPageStatusMaps: Array<{ index: number; resolve: () => void; reject: (error: unknown) => void }> = [];
    const pendingPressureMaps: Array<{ index: number; resolve: () => void }> = [];
    const pageStatusRanges: ArrayBuffer[] = [];
    const pressureRanges: ArrayBuffer[] = [];
    const pageStatusMapAsyncs: Array<ReturnType<typeof vi.fn>> = [];
    const polygonTriangleMapAsyncs: Array<ReturnType<typeof vi.fn>> = [];
    const dispatchWorkgroups = vi.fn();
    const createBuffer = vi.fn((descriptor: GPUBufferDescriptor) => {
        const buffer = makeBuffer(Number(descriptor.size));
        if (String(descriptor.label).startsWith("flip-page-status-readback-")) {
            const readbackIndex = Number(String(descriptor.label).at(-1));
            const range = new ArrayBuffer(Number(descriptor.size));
            pageStatusRanges.push(range);
            buffer.getMappedRange = vi.fn(() => range);
            buffer.mapAsync = vi.fn((): Promise<undefined> => {
                if (deferPageStatus) {
                    return new Promise<undefined>((resolve, reject) => {
                        pendingPageStatusMaps.push({ index: readbackIndex, resolve: () => resolve(undefined), reject });
                    });
                }
                if (rejectPageStatus) {
                    return Promise.reject(new Error("page status map failed"));
                }
                return Promise.resolve(undefined);
            });
            pageStatusMapAsyncs.push(buffer.mapAsync as ReturnType<typeof vi.fn>);
        } else if (String(descriptor.label).startsWith("flip-pressure-diagnostics-readback-")) {
            const readbackIndex = Number(String(descriptor.label).at(-1));
            const range = new ArrayBuffer(Number(descriptor.size));
            pressureRanges.push(range);
            buffer.getMappedRange = vi.fn(() => range);
            buffer.mapAsync = vi.fn((): Promise<undefined> => {
                if (deferPressure) {
                    return new Promise<undefined>((resolve) => {
                        pendingPressureMaps.push({ index: readbackIndex, resolve: () => resolve(undefined) });
                    });
                }
                return Promise.resolve(undefined);
            });
        } else if (String(descriptor.label).startsWith("flip-polygon-surface-triangle-count-")) {
            polygonTriangleMapAsyncs.push(buffer.mapAsync as ReturnType<typeof vi.fn>);
        }
        return buffer;
    });
    const device = {
        limits: {
            maxBufferSize: 1_000_000_000,
            maxStorageBufferBindingSize: 1_000_000_000,
            maxTextureDimension2D: 8192,
            ...limits,
        },
        createBuffer,
        createBindGroup: vi.fn((descriptor: GPUBindGroupDescriptor) => descriptor as unknown as GPUBindGroup),
        createCommandEncoder: vi.fn(() => ({
            clearBuffer: vi.fn(),
            copyBufferToBuffer: vi.fn(),
            pushDebugGroup: vi.fn(),
            popDebugGroup: vi.fn(),
            beginComputePass: vi.fn(() => ({
                setPipeline: vi.fn(),
                setBindGroup: vi.fn(),
                dispatchWorkgroups,
                dispatchWorkgroupsIndirect: vi.fn(),
                end: vi.fn(),
            })),
            finish: vi.fn(() => ({})),
        })),
        createComputePipeline: vi.fn(() => ({
            getBindGroupLayout: vi.fn((index: number) => ({ index }) as unknown as GPUBindGroupLayout),
        })),
        createShaderModule: vi.fn((descriptor: GPUShaderModuleDescriptor) => descriptor as unknown as GPUShaderModule),
        createTexture: vi.fn(
            (_descriptor: GPUTextureDescriptor) =>
                ({
                    createView: vi.fn(() => ({}) as GPUTextureView),
                    destroy: vi.fn(),
                }) as unknown as GPUTexture
        ),
        queue: {
            submit: vi.fn(),
            writeBuffer: vi.fn(),
        },
    } as unknown as GPUDevice;
    return {
        engine: { _device: device } as unknown as EngineContext,
        createBuffer,
        rejectPageStatusMaps: () => {
            rejectPageStatus = true;
        },
        deferPageStatusMaps: () => {
            deferPageStatus = true;
        },
        resolvePendingPageStatusMaps: () => {
            deferPageStatus = false;
            for (const pending of pendingPageStatusMaps.splice(0)) {
                pending.resolve();
            }
        },
        resolvePendingPageStatusMap: (index) => {
            const matches = pendingPageStatusMaps.filter((pending) => pending.index === index);
            for (const pending of matches) {
                pendingPageStatusMaps.splice(pendingPageStatusMaps.indexOf(pending), 1);
                pending.resolve();
            }
        },
        rejectPendingPageStatusMaps: () => {
            deferPageStatus = false;
            for (const pending of pendingPageStatusMaps.splice(0)) {
                pending.reject(new Error("stale page status map failed"));
            }
        },
        setPageStatus: (completedFrame, required, overflowFrame) => {
            for (const range of pageStatusRanges) {
                new Uint32Array(range).set([completedFrame, required, overflowFrame, required]);
            }
        },
        setPageStatusSlot: (index, completedFrame, active, overflowFrame, required) => {
            new Uint32Array(pageStatusRanges[index]!).set([completedFrame, active, overflowFrame, required]);
        },
        deferPressureMaps: () => {
            deferPressure = true;
        },
        resolvePendingPressureMap: (index) => {
            const matches = pendingPressureMaps.filter((pending) => pending.index === index);
            for (const pending of matches) {
                pendingPressureMaps.splice(pendingPressureMaps.indexOf(pending), 1);
                pending.resolve();
            }
        },
        setPressureStatus: (index, residual, rhs, divergence, cells) => {
            new Float32Array(pressureRanges[index]!).set([residual, rhs, divergence]);
            new Uint32Array(pressureRanges[index]!)[3] = cells;
        },
        pageStatusMapCount: () => pageStatusMapAsyncs.reduce((count, mapAsync) => count + mapAsync.mock.calls.length, 0),
        polygonTriangleMapCount: () => polygonTriangleMapAsyncs.reduce((count, mapAsync) => count + mapAsync.mock.calls.length, 0),
        computeDispatchCount: () => dispatchWorkgroups.mock.calls.length,
    };
}

describe("paged FLIP layout", () => {
    it("clamps requested capacity to the virtual block count for estimates and live allocation", () => {
        const layout = resolveFlipPageLayout([8, 8, 8], 64, 64);

        expect(layout).toMatchObject({
            numBlocks: 1,
            maxPages: 1,
            allocatedCells: 513,
            allocatedFaces: 1537,
            lookupWords: 4,
            lookupHeight: 1,
        });
        expect(estimateFlipGpuBytes(16, [8, 8, 8], "jacobi", { pagedGrid: true, pagedGridMaxPages: 64 })).toBe(
            estimateFlipGpuBytes(16, [8, 8, 8], "jacobi", { pagedGrid: true, pagedGridMaxPages: 1 })
        );

        const { engine } = makeEngine({
            maxBufferSize: 20_000,
            maxStorageBufferBindingSize: 20_000,
            maxTextureDimension2D: 64,
        });
        const sim = createFlipSim(engine, {
            count: 16,
            initialPositions: new Float32Array(0),
            boundsMin: [0, 0, 0],
            boundsMax: [1, 1, 1],
            gridDim: [8, 8, 8],
            dx: 0.125,
            particleRadius: 0.04,
            pagedGrid: true,
            pagedGridMaxPages: 64,
            minSubsteps: 1,
            maxSubsteps: 1,
            maxSubDt: 1,
        });

        expect(sim.gpuBytes).toBeGreaterThan(0);
        sim.dispose();
    });

    it("throws before allocating oversized dense multigrid coarse buffers", () => {
        const limit = 8_000_000;
        const { engine, createBuffer } = makeEngine({
            maxBufferSize: limit,
            maxStorageBufferBindingSize: limit,
            maxTextureDimension2D: 8192,
        });

        expect(() =>
            createFlipSim(engine, {
                count: 32,
                boundsMin: [0, 0, 0],
                boundsMax: [25.6, 25.6, 25.6],
                gridDim: [256, 256, 256],
                dx: 0.1,
                particleRadius: 0.04,
                pagedGrid: true,
                pagedGridMaxPages: 64,
                pressureSolver: "multigrid",
                minSubsteps: 1,
                maxSubsteps: 1,
                maxSubDt: 1,
            })
        ).toThrow(/Multigrid dense level 1 .*coarse-grid buffer/);
        expect(createBuffer.mock.calls.some(([descriptor]) => Number((descriptor as GPUBufferDescriptor).size) > limit)).toBe(false);
    });

    it("surfaces a rejected page-status map on the next step", async () => {
        const { engine, rejectPageStatusMaps } = makeEngine();
        const sim = createFlipSim(engine, {
            count: 1,
            initialPositions: new Float32Array([0.25, 0.25, 0.25]),
            boundsMin: [0, 0, 0],
            boundsMax: [1, 1, 1],
            gridDim: [8, 8, 8],
            dx: 0.125,
            pagedGrid: true,
            pagedGridMaxPages: 1,
            minSubsteps: 1,
            maxSubsteps: 1,
            maxSubDt: 1,
        });
        const device = engine._device;
        const first = device.createCommandEncoder();
        sim.step(first, 1 / 60);
        rejectPageStatusMaps();
        const second = device.createCommandEncoder();
        sim.step(second, 1 / 60);
        await Promise.resolve();
        await Promise.resolve();

        expect(() => sim.step(device.createCommandEncoder(), 1 / 60)).toThrow("page status map failed");
        sim.dispose();
    });

    it("retires copied page status when reset occurs before mapping", async () => {
        const onPages = vi.fn();
        const onOverflow = vi.fn();
        const { engine, pageStatusMapCount, setPageStatus } = makeEngine();
        const sim = createFlipSim(engine, {
            count: 1,
            initialPositions: new Float32Array([0.25, 0.25, 0.25]),
            boundsMin: [0, 0, 0],
            boundsMax: [1, 1, 1],
            gridDim: [8, 8, 8],
            dx: 0.125,
            pagedGrid: true,
            pagedGridMaxPages: 1,
            onPagedGridPages: onPages,
            onPagedGridOverflow: onOverflow,
            minSubsteps: 1,
            maxSubsteps: 1,
            maxSubDt: 1,
        });
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        setPageStatus(1, 8, 1);

        sim.reset();
        sim.step(engine._device.createCommandEncoder(), 0);
        await Promise.resolve();

        expect(pageStatusMapCount()).toBe(0);
        expect(onPages).not.toHaveBeenCalled();
        expect(onOverflow).not.toHaveBeenCalled();
        sim.dispose();
    });

    it("ignores a stale page-status fulfillment after reset", async () => {
        const onPages = vi.fn();
        const onOverflow = vi.fn();
        const { engine, deferPageStatusMaps, resolvePendingPageStatusMaps, setPageStatus } = makeEngine();
        const sim = createFlipSim(engine, {
            count: 1,
            initialPositions: new Float32Array([0.25, 0.25, 0.25]),
            boundsMin: [0, 0, 0],
            boundsMax: [1, 1, 1],
            gridDim: [8, 8, 8],
            dx: 0.125,
            pagedGrid: true,
            pagedGridMaxPages: 1,
            onPagedGridPages: onPages,
            onPagedGridOverflow: onOverflow,
            minSubsteps: 1,
            maxSubsteps: 1,
            maxSubDt: 1,
        });
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        setPageStatus(1, 8, 1);
        deferPageStatusMaps();
        sim.step(engine._device.createCommandEncoder(), 1 / 60);

        sim.reset();
        resolvePendingPageStatusMaps();
        await Promise.resolve();
        await Promise.resolve();

        expect(onPages).not.toHaveBeenCalled();
        expect(onOverflow).not.toHaveBeenCalled();
        expect(() => sim.step(engine._device.createCommandEncoder(), 0)).not.toThrow();
        sim.dispose();
    });

    it("ignores a stale page-status rejection after reset", async () => {
        const { engine, deferPageStatusMaps, rejectPendingPageStatusMaps } = makeEngine();
        const sim = createFlipSim(engine, {
            count: 1,
            initialPositions: new Float32Array([0.25, 0.25, 0.25]),
            boundsMin: [0, 0, 0],
            boundsMax: [1, 1, 1],
            gridDim: [8, 8, 8],
            dx: 0.125,
            pagedGrid: true,
            pagedGridMaxPages: 1,
            minSubsteps: 1,
            maxSubsteps: 1,
            maxSubDt: 1,
        });
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        deferPageStatusMaps();
        sim.step(engine._device.createCommandEncoder(), 1 / 60);

        sim.reset();
        rejectPendingPageStatusMaps();
        await Promise.resolve();
        await Promise.resolve();

        expect(() => sim.step(engine._device.createCommandEncoder(), 0)).not.toThrow();
        sim.dispose();
    });

    it("keeps submitting while status maps are delayed and observes sticky overflow", async () => {
        const onOverflow = vi.fn();
        const { engine, computeDispatchCount, deferPageStatusMaps, resolvePendingPageStatusMaps, setPageStatus } = makeEngine();
        const sim = createFlipSim(engine, {
            count: 1,
            initialPositions: new Float32Array([0.25, 0.25, 0.25]),
            boundsMin: [0, 0, 0],
            boundsMax: [1, 1, 1],
            gridDim: [8, 8, 8],
            dx: 0.125,
            pagedGrid: true,
            pagedGridMaxPages: 1,
            onPagedGridOverflow: onOverflow,
            minSubsteps: 1,
            maxSubsteps: 1,
            maxSubDt: 1,
        });
        deferPageStatusMaps();

        const before = computeDispatchCount();
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        const afterFirst = computeDispatchCount();
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        const afterSecond = computeDispatchCount();
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        const afterThird = computeDispatchCount();

        expect(afterFirst).toBeGreaterThan(before);
        expect(afterSecond).toBeGreaterThan(afterFirst);
        expect(afterThird).toBeGreaterThan(afterSecond);

        setPageStatus(2, 2, 0);
        resolvePendingPageStatusMaps();
        await Promise.resolve();
        await Promise.resolve();

        expect(onOverflow).not.toHaveBeenCalled();
        const beforeStickyReadback = computeDispatchCount();
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        expect(computeDispatchCount()).toBeGreaterThan(beforeStickyReadback);
        setPageStatus(4, 4, 3);
        sim.step(engine._device.createCommandEncoder(), 0);
        await Promise.resolve();
        await Promise.resolve();

        expect(onOverflow).toHaveBeenCalledOnce();
        const beforeBlocked = computeDispatchCount();
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        expect(computeDispatchCount()).toBe(beforeBlocked);
        sim.dispose();
    });

    it("publishes only the newest page sample and separates active from high-water demand", async () => {
        const onPages = vi.fn();
        const onOverflow = vi.fn();
        const { engine, deferPageStatusMaps, resolvePendingPageStatusMap, setPageStatusSlot } = makeEngine();
        const sim = createFlipSim(engine, {
            count: 1,
            initialPositions: new Float32Array([0.25, 0.25, 0.25]),
            boundsMin: [0, 0, 0],
            boundsMax: [1, 1, 1],
            gridDim: [8, 8, 8],
            dx: 0.125,
            pagedGrid: true,
            pagedGridMaxPages: 1,
            onPagedGridPages: onPages,
            onPagedGridOverflow: onOverflow,
            minSubsteps: 1,
            maxSubsteps: 1,
            maxSubDt: 1,
        });
        deferPageStatusMaps();
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        setPageStatusSlot(0, 1, 1, 0, 4);
        setPageStatusSlot(1, 2, 1, 2, 7);

        resolvePendingPageStatusMap(1);
        await Promise.resolve();
        await Promise.resolve();
        expect(onPages).toHaveBeenLastCalledWith(1, 1);
        expect(onOverflow).toHaveBeenLastCalledWith(7, 1);

        resolvePendingPageStatusMap(0);
        await Promise.resolve();
        await Promise.resolve();
        expect(onPages).toHaveBeenCalledTimes(1);
        expect(onOverflow).toHaveBeenCalledTimes(1);
        sim.dispose();
    });

    it("bounds speculative page frames until status progress resumes", async () => {
        const { engine, computeDispatchCount, deferPageStatusMaps, resolvePendingPageStatusMaps, setPageStatus } = makeEngine();
        const sim = createFlipSim(engine, {
            count: 1,
            initialPositions: new Float32Array([0.25, 0.25, 0.25]),
            boundsMin: [0, 0, 0],
            boundsMax: [1, 1, 1],
            gridDim: [8, 8, 8],
            dx: 0.125,
            pagedGrid: true,
            pagedGridMaxPages: 1,
            minSubsteps: 1,
            maxSubsteps: 1,
            maxSubDt: 1,
        });
        deferPageStatusMaps();
        const dispatchCounts: number[] = [];
        for (let frame = 0; frame < 12; frame++) {
            sim.step(engine._device.createCommandEncoder(), 1 / 60);
            dispatchCounts.push(computeDispatchCount());
        }

        expect(dispatchCounts[7]).toBeGreaterThan(dispatchCounts[6]!);
        expect(dispatchCounts.slice(8)).toEqual(new Array(4).fill(dispatchCounts[7]));

        setPageStatus(8, 1, 0);
        resolvePendingPageStatusMaps();
        await Promise.resolve();
        await Promise.resolve();
        const beforeResume = computeDispatchCount();
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        expect(computeDispatchCount()).toBeGreaterThan(beforeResume);
        sim.dispose();
    });

    it("ignores out-of-order and pre-reset pressure completions", async () => {
        const { engine, deferPressureMaps, resolvePendingPressureMap, setPressureStatus } = makeEngine();
        const sim = createFlipSim(engine, {
            count: 1,
            initialPositions: new Float32Array([0.25, 0.25, 0.25]),
            boundsMin: [0, 0, 0],
            boundsMax: [1, 1, 1],
            gridDim: [8, 8, 8],
            dx: 0.125,
            pressureDiagnostics: true,
            pressureIterations: 3,
            minSubsteps: 1,
            maxSubsteps: 1,
            maxSubDt: 1,
        });
        deferPressureMaps();
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        setPressureStatus(0, 2, 4, 0.5, 8);
        setPressureStatus(1, 9, 3, 1.5, 12);

        resolvePendingPressureMap(1);
        await Promise.resolve();
        await Promise.resolve();
        expect(sim.pressureDiagnostics).toEqual({
            maxResidual: 9,
            maxRhs: 3,
            relativeResidual: 3,
            maxDivergence: 1.5,
            fluidCellCount: 12,
            pressureIterations: 3,
        });

        resolvePendingPressureMap(0);
        await Promise.resolve();
        await Promise.resolve();
        expect(sim.pressureDiagnostics?.maxResidual).toBe(9);

        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        sim.step(engine._device.createCommandEncoder(), 1 / 60);
        sim.reset();
        resolvePendingPressureMap(0);
        resolvePendingPressureMap(1);
        await Promise.resolve();
        await Promise.resolve();
        expect(sim.pressureDiagnostics).toBeUndefined();
        sim.dispose();
    });

    it("gates every paged polygon mutation and defers count sampling until page acceptance", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/solvers/flip-sim.ts"), "utf8");
        const polygon = source.slice(source.indexOf("function encodePolygonSurface("), source.indexOf("function ensureMultigrid("));

        expect(polygon).not.toContain("dispatch(encoder");
        expect(polygon).toContain("PAGE_DISPATCH_POLYGON_CELLS_OFFSET");
        expect(polygon).toContain("PAGE_DISPATCH_POLYGON_CUBES_OFFSET");
        expect(polygon).toContain("PAGE_DISPATCH_SINGLE_OFFSET");
        expect(polygon).toContain("if (!pageLayout)");
        expect(source).toContain("function encodeConfirmedPolygonTriangleCountReadback(");
        expect(source).toContain("latestAcceptedPageStatusFrame <= polygonTriangleReadbackPageFrame");
    });

    it("does not map a paged polygon count copy before its command encoder is submitted", async () => {
        const { engine, polygonTriangleMapCount, setPageStatus } = makeEngine();
        const sim = createFlipSim(engine, {
            count: 1,
            initialPositions: new Float32Array([0.25, 0.25, 0.25]),
            boundsMin: [0, 0, 0],
            boundsMax: [1, 1, 1],
            gridDim: [8, 8, 8],
            dx: 0.125,
            pagedGrid: true,
            pagedGridMaxPages: 1,
            polygonSurface: true,
            minSubsteps: 1,
            maxSubsteps: 1,
            maxSubDt: 1,
        });
        const device = engine._device;
        sim.step(device.createCommandEncoder(), 1 / 60);
        setPageStatus(1, 1, 0);
        sim.step(device.createCommandEncoder(), 1 / 60);
        await Promise.resolve();
        await Promise.resolve();

        sim.step(device.createCommandEncoder(), 1 / 60);
        expect(polygonTriangleMapCount()).toBe(0);

        sim.step(device.createCommandEncoder(), 0);
        expect(polygonTriangleMapCount()).toBe(1);
        sim.dispose();
    });

    it("discovers relaunch destinations and retries weak page locks until terminal", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/solvers/flip-sim.ts"), "utf8");
        const allocation = source.slice(source.indexOf("function buildPagedFlipAllocationWgsl"), source.indexOf("const FLIP_PAGE_DISPATCH_CONFIG_WGSL"));

        expect(allocation).toContain("loop {");
        expect(allocation).not.toContain("attempt < 4u");
        expect(allocation).not.toContain("if (!acquired)");
        expect(source).toContain("flowFrame.emitActive || relaunchActive ? flowState.emitterIds.length : 0");
        expect(source).toContain("flowState.legacyEmitter !== null");
    });

    it("orders page and pressure samples by reset generation and bounds speculative frames", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/solvers/flip-sim.ts"), "utf8");

        expect(source).toContain("atomicStore(&statusData[1], requiredPages)");
        expect(source).toContain("atomicMax(&statusData[3], requiredPages)");
        expect(source).toContain("options.onPagedGridPages?.(activePages, pageLayout.maxPages)");
        expect(source).toContain("pageStatusSampleIds[pageStatusIndex] = ++pageStatusSampleId");
        expect(source).toContain("sampleId > publishedPageStatusSampleId");
        expect(source).toContain("pageStatusSnapshots.size >= MAX_PAGED_SPECULATIVE_FRAMES");
        expect(source).toContain("resources.readbackSampleIds[index] = ++pressureDiagnosticSampleId");
        expect(source).toContain("generation === pressureDiagnosticGeneration");
        expect(source).toContain("sampleId > publishedPressureDiagnosticSampleId");
        expect(source).toContain("pressureIterations: sampledIterations");
        expect(source).toContain("retirePressureDiagnostics()");
    });

    it("shares conservative page intersections with authoritative shape semantics", () => {
        expect(FLUID_FLOW_SHAPE_WGSL).toContain("let h=max(r,0.5*abs(s.params0.y))");
        expect(FLUID_FLOW_SHAPE_WGSL).toContain("fluidShapeIntersectsWorldAabb");
        expect(FLUID_FLOW_SHAPE_WGSL).toContain("fluidPolygonIntersectsRect");
        expect(FLUID_FLOW_SHAPE_WGSL).toContain("let inner=min(radius,abs(s.params0.z))");
    });
});
