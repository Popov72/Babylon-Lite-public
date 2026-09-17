import { afterEach, describe, expect, it, vi } from "vitest";
import { createFluidProfiler } from "../../../../packages/babylon-lite/src/fluid/core/gpu-profiler";

afterEach(() => vi.unstubAllGlobals());

function fixture() {
    vi.stubGlobal("GPUBufferUsage", { QUERY_RESOLVE: 1, COPY_SRC: 2, MAP_READ: 4, COPY_DST: 8 });
    vi.stubGlobal("GPUMapMode", { READ: 1 });
    let timestamps = new BigInt64Array();
    let clock = 0n;
    type BufferStub = {
        data: ArrayBuffer;
        mapState: string;
        mapAsync: () => Promise<void>;
        getMappedRange: () => ArrayBuffer;
        unmap: () => void;
        destroy: () => void;
    };
    const buffers: BufferStub[] = [];
    function buffer(size: number): BufferStub {
        const value: BufferStub = {
            data: new ArrayBuffer(size),
            mapState: "unmapped",
            mapAsync: async () => {
                value.mapState = "mapped";
            },
            getMappedRange: () => value.data,
            unmap: () => {
                value.mapState = "unmapped";
            },
            destroy: vi.fn(() => {
                value.mapState = "unmapped";
            }),
        };
        buffers.push(value);
        return value;
    }
    const device = {
        features: new Set(["timestamp-query"]),
        createQuerySet: ({ count }: { count: number }) => {
            timestamps = new BigInt64Array(count);
            return { destroy: vi.fn() };
        },
        createBuffer: ({ size }: { size: number }) => buffer(size),
    };
    const encoder = {
        beginComputePass: ({ timestampWrites }: { timestampWrites: GPUComputePassTimestampWrites }) => {
            if (timestampWrites.beginningOfPassWriteIndex !== undefined) timestamps[timestampWrites.beginningOfPassWriteIndex] = clock;
            if (timestampWrites.endOfPassWriteIndex !== undefined) timestamps[timestampWrites.endOfPassWriteIndex] = clock;
            return { end: () => {} };
        },
        resolveQuerySet: (_set: unknown, first: number, count: number, target: ReturnType<typeof buffer>, offset: number) => {
            new Uint8Array(target.data, offset, count * 8).set(new Uint8Array(timestamps.buffer, first * 8, count * 8));
        },
        copyBufferToBuffer: (source: ReturnType<typeof buffer>, sourceOffset: number, target: ReturnType<typeof buffer>, targetOffset: number, size: number) => {
            new Uint8Array(target.data, targetOffset, size).set(new Uint8Array(source.data, sourceOffset, size));
        },
    };
    return {
        device,
        encoder,
        buffers,
        time: (ms: number) => {
            clock = BigInt(ms) * 1_000_000n;
        },
        interval: (begin: number, end: number, from: number, to: number) => {
            timestamps[begin] = BigInt(from) * 1_000_000n;
            timestamps[end] = BigInt(to) * 1_000_000n;
        },
    };
}

describe("asynchronous fluid GPU profiling", () => {
    it("does not add GPU-resident simulation spans to the frame envelope twice", async () => {
        const f = fixture();
        const profiler = createFluidProfiler(f.device as never, 16);
        profiler.beginFrame();
        profiler.frameStart(f.encoder as never);
        const simulation = profiler.stageSpan!("Simulation")!;
        f.interval(simulation.begin.beginningOfPassWriteIndex, simulation.end.endOfPassWriteIndex, 1, 5);
        const surface = profiler.pass("Surface")!;
        f.interval(surface.beginningOfPassWriteIndex, surface.endOfPassWriteIndex, 6, 8);
        f.time(9);
        profiler.frameStop(f.encoder as never);
        profiler.resolveInto(f.encoder as never);
        await profiler.collectSubmitted();
        expect(profiler.results()).toEqual({ stages: { Simulation: 4, Surface: 2 }, total: 6, frameTotal: 9 });
        profiler.dispose();
    });

    it("adds submitted GPU spans to the render envelope without counting CPU wait gaps", async () => {
        const f = fixture();
        const profiler = createFluidProfiler(f.device as never, 32);
        profiler.beginFrame();
        profiler.frameStart(f.encoder as never);
        const surface = profiler.pass("Surface")!;
        f.interval(surface.beginningOfPassWriteIndex, surface.endOfPassWriteIndex, 1, 3);
        f.time(5);
        profiler.frameStop(f.encoder as never);
        f.time(100);
        const first = profiler.commandSpan!(f.encoder as never, "Simulation")!;
        f.time(104);
        first();
        f.time(500);
        const second = profiler.commandSpan!(f.encoder as never, "Simulation")!;
        f.time(502);
        second();
        profiler.resolveInto(f.encoder as never);
        await profiler.collectSubmitted();
        expect(profiler.results()).toEqual({ stages: { Surface: 2, Simulation: 6 }, total: 8, frameTotal: 11 });
        profiler.dispose();
    });

    it("marks an exhausted query budget instead of presenting a partial sample as complete", async () => {
        const f = fixture();
        const profiler = createFluidProfiler(f.device as never, 8);
        profiler.beginFrame();
        for (let i = 0; i < 4; i++) {
            const end = profiler.commandSpan!(f.encoder as never, "Simulation")!;
            f.time(i + 1);
            end();
        }
        expect(profiler.commandSpan!(f.encoder as never, "Simulation")).toBeUndefined();
        profiler.resolveInto(f.encoder as never);
        await profiler.collectSubmitted();
        expect(profiler.results()?.overflowed).toBe(true);
        profiler.dispose();
    });

    it("clears Simulation time on render-only frames and destroys buffers even during collection", async () => {
        const f = fixture();
        const profiler = createFluidProfiler(f.device as never, 16);
        profiler.beginFrame();
        profiler.frameStart(f.encoder as never);
        f.time(2);
        profiler.frameStop(f.encoder as never);
        profiler.resolveInto(f.encoder as never);
        await profiler.collectSubmitted();
        expect(profiler.results()).toEqual({ stages: {}, total: 0, frameTotal: 2 });
        profiler.beginFrame();
        profiler.frameStart(f.encoder as never);
        profiler.frameStop(f.encoder as never);
        profiler.resolveInto(f.encoder as never);
        const pending = profiler.collectSubmitted();
        profiler.dispose();
        await pending;
        for (const buffer of f.buffers) expect(buffer.destroy).toHaveBeenCalledOnce();
    });
});
