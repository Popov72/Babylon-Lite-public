import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { AquanovaFluidRuntime } from "../../../../lab/lite/src/demos/aquanova/fluid-runtime.js";
import { GpuReadbackPool } from "../../../../packages/babylon-lite/src/fluid/core/gpu-readback.js";

// A minimal, controllable WebGPU staging-buffer stand-in. `mapAsync` returns a promise the test
// resolves or rejects on demand, and `destroy()` rejects any in-flight map exactly like the real
// API — the precise condition (buffer destroyed / device lost mid-map) that finding 16 is about.
interface PendingMap {
    readonly buffer: FakeBuffer;
    resolve(): void;
    reject(error: unknown): void;
}

class FakeBuffer {
    public mapState: "unmapped" | "pending" | "mapped" = "unmapped";
    public destroyed = false;
    public unmapCount = 0;
    public readonly bytes: ArrayBuffer;
    private pending: PendingMap | null = null;

    constructor(
        public readonly label: string,
        size: number,
        private readonly gpu: FakeGpu
    ) {
        this.bytes = new ArrayBuffer(Math.max(4, size));
    }

    setUint32(index: number, value: number): void {
        new Uint32Array(this.bytes)[index] = value;
    }

    mapAsync(_mode: number, _offset = 0, _size?: number): Promise<void> {
        if (this.destroyed) {
            return Promise.reject(new Error(`${this.label} destroyed`));
        }
        this.mapState = "pending";
        return new Promise<void>((resolve, reject) => {
            const entry: PendingMap = {
                buffer: this,
                resolve: () => {
                    this.mapState = "mapped";
                    resolve();
                },
                reject: (error) => {
                    this.mapState = "unmapped";
                    reject(error);
                },
            };
            this.pending = entry;
            this.gpu.pendingMaps.push(entry);
        });
    }

    getMappedRange(offset = 0, size?: number): ArrayBuffer {
        if (this.mapState !== "mapped") {
            throw new Error(`${this.label} getMappedRange while ${this.mapState}`);
        }
        return this.bytes.slice(offset, size != null ? offset + size : undefined);
    }

    unmap(): void {
        this.unmapCount++;
        this.mapState = "unmapped";
    }

    destroy(): void {
        this.destroyed = true;
        const pending = this.pending;
        if (pending) {
            this.pending = null;
            this.gpu.removePending(pending);
            pending.reject(new Error(`${this.label} destroyed during map`));
        }
    }

    clearPending(): void {
        this.pending = null;
    }
}

class FakeGpu {
    public readonly pendingMaps: PendingMap[] = [];
    public readonly buffers: FakeBuffer[] = [];

    createBuffer(desc: { label?: string; size: number }): FakeBuffer {
        const buffer = new FakeBuffer(desc.label ?? "buffer", desc.size, this);
        this.buffers.push(buffer);
        return buffer;
    }

    // Minimal stubs so AquanovaFluidRuntime.installParticleCounter can build its (never-executed)
    // compute plumbing. The test only exercises the async readback path, not real GPU work.
    createShaderModule(): object {
        return {};
    }

    createComputePipeline(): { getBindGroupLayout(index: number): object } {
        return { getBindGroupLayout: () => ({}) };
    }

    createBindGroup(): object {
        return {};
    }

    get limits(): Record<string, number> {
        return { maxStorageBufferBindingSize: 1 << 27 };
    }

    readonly queue = {
        writeBuffer: (): void => {},
    };

    removePending(entry: PendingMap): void {
        const index = this.pendingMaps.indexOf(entry);
        if (index >= 0) {
            this.pendingMaps.splice(index, 1);
        }
    }

    resolveAll(): void {
        const entries = this.pendingMaps.splice(0);
        for (const entry of entries) {
            entry.buffer.clearPending();
            entry.resolve();
        }
    }

    rejectAll(error: unknown): void {
        const entries = this.pendingMaps.splice(0);
        for (const entry of entries) {
            entry.buffer.clearPending();
            entry.reject(error);
        }
    }
}

// A recording-only command encoder. `clearBuffer` is intentionally a no-op so a test-injected
// counter value survives into the readback copy; `copyBufferToBuffer` moves real bytes so the
// mapped result reflects what the (simulated) compute pass produced.
function fakeEncoder(): GPUCommandEncoder {
    return {
        clearBuffer: (): void => {},
        beginComputePass: () => ({
            setPipeline: (): void => {},
            setBindGroup: (): void => {},
            dispatchWorkgroups: (): void => {},
            end: (): void => {},
        }),
        copyBufferToBuffer: (src: FakeBuffer, srcOffset: number, dst: FakeBuffer, dstOffset: number, size: number): void => {
            new Uint8Array(dst.bytes).set(new Uint8Array(src.bytes.slice(srcOffset, srcOffset + size)), dstOffset);
        },
    } as unknown as GPUCommandEncoder;
}

const flush = async (): Promise<void> => {
    // Let every queued `.then/.catch/.finally` in the pool's map chain settle.
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
};

describe("Aquanova GPU readback pool", () => {
    let unhandled: unknown[];
    const trackUnhandled = (reason: unknown): void => {
        unhandled.push(reason);
    };

    beforeEach(() => {
        unhandled = [];
        process.on("unhandledRejection", trackUnhandled);
    });

    afterEach(async () => {
        await flush();
        process.off("unhandledRejection", trackUnhandled);
        expect(unhandled).toEqual([]);
    });

    it("delivers a completed readback and recycles the slot", async () => {
        const gpu = new FakeGpu();
        const onComplete = vi.fn();
        const pool = new GpuReadbackPool<undefined>({
            device: gpu as unknown as GPUDevice,
            label: "test",
            slotCount: 2,
            byteLength: 4,
            defaultPayload: undefined,
            onComplete,
            onError: vi.fn(),
        });

        const slot = pool.acquire();
        expect(slot).not.toBeNull();
        (slot!.buffer as unknown as FakeBuffer).setUint32(0, 42);
        pool.submit(slot!, 1, undefined);
        pool.pump();
        gpu.resolveAll();
        await flush();

        expect(onComplete).toHaveBeenCalledTimes(1);
        const [view] = onComplete.mock.calls[0]!;
        expect(new Uint32Array(view as ArrayBuffer)[0]).toBe(42);
        // Slot recycled: both slots are idle again.
        expect(pool.acquire()).not.toBeNull();
        expect((slot!.buffer as unknown as FakeBuffer).unmapCount).toBe(1);
        pool.dispose();
    });

    it("surfaces a map rejection outside disposal without an unhandled rejection", async () => {
        const gpu = new FakeGpu();
        const onComplete = vi.fn();
        const onError = vi.fn();
        const pool = new GpuReadbackPool<undefined>({
            device: gpu as unknown as GPUDevice,
            label: "test",
            slotCount: 1,
            byteLength: 4,
            defaultPayload: undefined,
            onComplete,
            onError,
        });

        const slot = pool.acquire()!;
        pool.submit(slot, 7, undefined);
        pool.pump();
        gpu.rejectAll(new Error("device lost"));
        await flush();

        expect(onComplete).not.toHaveBeenCalled();
        expect(onError).toHaveBeenCalledTimes(1);
        expect(onError.mock.calls[0]![1]).toBe(slot); // carries the failed slot (with its generation)
        // Slot recovers to idle for reuse.
        expect(pool.acquire()).toBe(slot);
        pool.dispose();
    });

    it("swallows rejections caused by disposal and never calls onError", async () => {
        const gpu = new FakeGpu();
        const onComplete = vi.fn();
        const onError = vi.fn();
        const pool = new GpuReadbackPool<undefined>({
            device: gpu as unknown as GPUDevice,
            label: "test",
            slotCount: 2,
            byteLength: 4,
            defaultPayload: undefined,
            onComplete,
            onError,
        });

        const slot = pool.acquire()!;
        pool.submit(slot, 3, undefined);
        pool.pump();
        // Disposing destroys buffers, rejecting the in-flight map.
        pool.dispose();
        await flush();

        expect(onComplete).not.toHaveBeenCalled();
        expect(onError).not.toHaveBeenCalled();
        expect(pool.disposed).toBe(true);
        expect(pool.acquire()).toBeNull();
    });
});

describe("AquanovaFluidRuntime particle-count readback", () => {
    let unhandled: unknown[];
    const trackUnhandled = (reason: unknown): void => {
        unhandled.push(reason);
    };

    beforeEach(() => {
        unhandled = [];
        process.on("unhandledRejection", trackUnhandled);
    });

    describe("Aquanova production aggregate ordering", () => {
        it("refreshes the current collection before recording gameplay queries", () => {
            const source = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/aquanova/main.ts"), "utf8");
            const frame = source.slice(
                source.lastIndexOf("const fluidFrame = syncFluidCollections()"),
                source.indexOf("canvas.dataset.particleCount", source.lastIndexOf("const fluidFrame"))
            );
            expect(source).toContain('createFluidSimulationCollectionParticleStream(allFluidCollection, scene, MAX_TOTAL, { update: "manual" })');
            expect(frame.indexOf("refreshFluidSimulationCollectionParticleStream(allFluidStream)")).toBeGreaterThan(-1);
            expect(frame.indexOf("recordParticleCount(")).toBeGreaterThan(frame.indexOf("refreshFluidSimulationCollectionParticleStream(allFluidStream)"));
            expect(frame.indexOf("recordElectricity(")).toBeGreaterThan(frame.indexOf("refreshFluidSimulationCollectionParticleStream(allFluidStream)"));
        });
    });

    afterEach(async () => {
        await flush();
        process.off("unhandledRejection", trackUnhandled);
        expect(unhandled).toEqual([]);
    });

    function makeRuntime(): { gpu: FakeGpu; runtime: AquanovaFluidRuntime; counter: FakeBuffer } {
        const gpu = new FakeGpu();
        const runtime = new AquanovaFluidRuntime();
        const positionBuffer = gpu.createBuffer({ label: "pos", size: 64 });
        const alphaBuffer = gpu.createBuffer({ label: "alpha", size: 16 });
        const stream = {
            count: 4,
            capacity: 4,
            _resolve: () => ({
                positionBuffer: positionBuffer as unknown as GPUBuffer,
                alphaBuffer: alphaBuffer as unknown as GPUBuffer,
                count: 4,
                capacity: 4,
            }),
        };
        runtime.installParticleCounter({
            engine: {
                _device: gpu as unknown as GPUDevice,
                _currentEncoder: fakeEncoder(),
            } as never,
            stream,
        });
        const counter = gpu.buffers.find((buffer) => buffer.label === "fluid-spatial-results")!;
        return { gpu, runtime, counter };
    }

    const aabb = { min: [0, 0, 0] as const, max: [1, 1, 1] as const };

    // Drive frames until `countParticlesInAabb` reflects a settled value. A submit is throttled to
    // once every few frames, and a readback resolves one frame after it is submitted, so several
    // iterations are required before the asynchronous count catches up.
    async function runFrames(gpu: FakeGpu, runtime: AquanovaFluidRuntime, counter: FakeBuffer, count: number, frames: number, settle: "resolve" | "reject"): Promise<void> {
        for (let frame = 0; frame < frames; frame++) {
            runtime.countParticlesInAabb(aabb);
            counter.setUint32(0, count);
            runtime.recordParticleCount(50);
            if (settle === "reject") {
                gpu.rejectAll(new Error("device lost"));
            } else {
                gpu.resolveAll();
            }
            await flush();
        }
    }

    it("reports the asynchronously completed count", async () => {
        const { gpu, runtime, counter } = makeRuntime();
        await runFrames(gpu, runtime, counter, 100, 12, "resolve");
        expect(runtime.countParticlesInAabb(aabb)).toBe(100);
        runtime.dispose();
    });

    it("invalidates the count on device loss instead of preserving a stale success", async () => {
        const { gpu, runtime, counter } = makeRuntime();
        await runFrames(gpu, runtime, counter, 100, 12, "resolve");
        expect(runtime.countParticlesInAabb(aabb)).toBe(100);

        // Every subsequent readback map rejects (device lost); the last good count must not persist.
        await runFrames(gpu, runtime, counter, 100, 12, "reject");
        expect(runtime.countParticlesInAabb(aabb)).toBe(0);
        runtime.dispose();
    });

    it("disposes cleanly while a readback map is still pending", async () => {
        const { gpu, runtime, counter } = makeRuntime();
        // Submit a readback, then map it (pending) on the next frame without resolving.
        runtime.countParticlesInAabb(aabb);
        counter.setUint32(0, 77);
        runtime.recordParticleCount(50);
        runtime.countParticlesInAabb(aabb);
        runtime.recordParticleCount(50); // pumps the prior submission into a pending map
        expect(gpu.pendingMaps.length).toBeGreaterThan(0);

        // Disposal destroys the staging buffers, rejecting the pending map. It must be swallowed.
        runtime.dispose();
        await flush();
        expect(runtime.countParticlesInAabb(aabb)).toBe(0);
    });
});
