// Opt-in GPU profiler for fluid backends (the demo's "Timing" section).
//
// This is the CONCRETE implementation of the type-only `FluidProfiler` hook in
// ./sim-common.ts. It is a side-effect-free module, so the bundler pays zero bytes for
// the timestamp-query machinery unless an app actually imports `createFluidProfiler`:
// the sims / render tasks only reference the `FluidProfiler` TYPE (erased at compile
// time) and add `timestampWrites: profiler?.pass(stage)` to their pass descriptors,
// which is a no-op while no profiler is installed. It lives in the reusable core (shared
// by every fluid host) but is intentionally NOT root-exported, because its interface
// exposes raw WebGPU handles (GPUDevice/GPUCommandEncoder) that must never reach the
// public package API; hosts import it via `babylon-lite/fluid/core/gpu-profiler.js`.
//
// Mechanism — portable WebGPU timestamp queries via `timestampWrites` (NOT the
// non-portable encoder.writeTimestamp):
//   • A single `GPUQuerySet` of type "timestamp" holds a begin/end pair per timed
//     pass. `pass(stage)` hands out the next free pair as a `timestampWrites`
//     descriptor and remembers which stage it belongs to.
//   • Once per frame the demo's resolve task calls `resolveInto(encoder)` which
//     resolves the written queries into a buffer, then copies them into a mappable
//     readback buffer. The readback is mapped ONE FRAME LATER (guaranteed submitted
//     by then), read as nanosecond BigInt timestamps, and reduced to per-stage ms.
//   • `results()` exposes the latest per-stage + total figures for the UI.
//
// Note: Chrome quantizes timestamp-query resolution (typically to ~100 µs) for
// privacy, so the reported ms values are coarse — fine for a relative profiler.

import type { FluidProfiler } from "./sim-common.js";

/** The lab-side profiler: the package-facing `FluidProfiler` plus the frame-driver
 *  methods the demo calls (beginFrame / resolveInto / results / dispose). */
/** @internal */
export interface FluidProfilerImpl extends FluidProfiler {
    /** Reset the per-frame query cursor + stage records. Call at the very start of
     *  the frame, before any timed pass is encoded. */
    beginFrame(): void;
    /** Encode an empty envelope pass capturing the frame-START GPU timestamp. Call
     *  FIRST in the frame's encoder, before any real pass. Paired with frameStop to
     *  measure the WHOLE frame's GPU time (results().frameTotal). */
    frameStart(encoder: GPUCommandEncoder): void;
    /** Encode an empty envelope pass capturing the frame-END GPU timestamp. Call LAST,
     *  immediately before resolveInto. */
    frameStop(encoder: GPUCommandEncoder): void;
    /** Resolve this frame's written queries + copy them to a readback buffer. Must be
     *  recorded LAST in the frame's encoder (after every timed pass), before submit. */
    resolveInto(encoder: GPUCommandEncoder): void;
    /** Collect queries whose resolve/copy command has already been submitted. */
    collectSubmitted(): Promise<void>;
    /** Latest computed GPU times (ms), or null before the first readback has completed.
     *  `total` is the SUM of the individually timed passes; `frameTotal` is the whole
     *  frame's GPU time (frameStart→frameStop envelope), so `frameTotal - total` is the
     *  untimed remainder ("Other"). frameTotal is 0 until the envelope has been read. */
    results(): { stages: Record<string, number>; total: number; frameTotal: number; overflowed?: boolean } | null;
    /** Free the query set + all buffers. */
    dispose(): void;
}

// Default query capacity. Async backends can request a larger window for their
// independently submitted command spans; exhausted windows are reported incomplete.
const CAPACITY = 256;
// Mappable readback buffers cycled so a buffer is never re-used while its map is in
// flight (guards against overlapping mapAsync on the same buffer).
const READBACK_POOL = 3;

interface StageRec {
    stage: string;
    begin: number;
    end: number;
    outsideFrame?: boolean;
}
interface Inflight {
    buf: GPUBuffer;
    count: number;
    records: StageRec[];
    /** Query indices of the whole-frame envelope (frameStart begin / frameStop end),
     *  or -1 when the envelope was not encoded this frame. */
    frameBegin: number;
    frameEnd: number;
    overflowed: boolean;
    sequence: number;
}

/** Create the fluid GPU profiler. Throws if the device lacks the "timestamp-query"
 *  feature — the lab catches this and shows a UI fallback. */
/** @internal */
export function createFluidProfiler(device: GPUDevice, queryCapacity = CAPACITY): FluidProfilerImpl {
    if (!device.features.has("timestamp-query")) {
        throw new Error("timestamp-query feature not available");
    }
    if (!Number.isInteger(queryCapacity) || queryCapacity < 8 || queryCapacity > 8192) {
        throw new RangeError("GPU profiler query capacity must be an integer from 8 to 8192.");
    }

    const querySet = device.createQuerySet({ label: "fluid-timing", type: "timestamp", count: queryCapacity });
    const resolveBuf = device.createBuffer({
        label: "fluid-timing-resolve",
        size: queryCapacity * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const free: GPUBuffer[] = [];
    const buffers: GPUBuffer[] = [];
    for (let i = 0; i < READBACK_POOL; i++) {
        const buffer = device.createBuffer({
            label: `fluid-timing-readback-${i}`,
            size: queryCapacity * 8,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        free.push(buffer);
        buffers.push(buffer);
    }

    let cursor = 0;
    let frameRecords: StageRec[] = [];
    // Whole-frame envelope query indices for the current frame (set by frameStart /
    // frameStop, -1 when not encoded). frameBeginIdx = the begin query of the early
    // empty pass; frameEndIdx = the end query of the late empty pass. The stop pair is
    // RESERVED up front by frameStart so a busy frame (near CAPACITY) can never starve
    // the frame-end timestamp.
    let frameBeginIdx = -1;
    let frameEndIdx = -1;
    let frameStopReserved: { begin: number; end: number } | null = null;
    // Copies recorded in a PREVIOUS frame's resolveInto — submitted by now, so safe to
    // map. Reaped at the top of the next resolveInto.
    const inflight: Inflight[] = [];
    let latest: { stages: Record<string, number>; total: number; frameTotal: number; overflowed?: boolean } | null = null;
    let disposed = false;
    let overflowed = false;
    let sequence = 0;
    let publishedSequence = 0;

    // Allocate the next free begin/end query pair, or null when the set is full.
    function allocPair(): { begin: number; end: number } | null {
        if (cursor + 2 > queryCapacity) {
            overflowed = true;
            return null;
        }
        const begin = cursor;
        const end = cursor + 1;
        cursor += 2;
        return { begin, end };
    }

    // Map the buffers whose copy has already been submitted (previous frames), read
    // the timestamps, reduce to per-stage ms, then return the buffer to the pool.
    async function reap(): Promise<void> {
        if (inflight.length === 0) {
            return;
        }
        const ready = inflight.splice(0, inflight.length);
        await Promise.all(
            ready.map((e) =>
                e.buf
                    .mapAsync(GPUMapMode.READ)
                    .then(() => {
                        if (disposed) {
                            return;
                        }
                        const ts = new BigInt64Array(e.buf.getMappedRange());
                        const stages: Record<string, number> = {};
                        let total = 0;
                        let outsideFrame = 0;
                        for (const r of e.records) {
                            // GPU nanoseconds; clamp to avoid tiny negatives from quantization.
                            const ms = Math.max(0, Number(ts[r.end]! - ts[r.begin]!) / 1e6);
                            stages[r.stage] = (stages[r.stage] ?? 0) + ms;
                            total += ms;
                            if (r.outsideFrame) {
                                outsideFrame += ms;
                            }
                        }
                        // Whole-frame GPU time from the envelope (frameStart → frameStop).
                        let frameTotal = 0;
                        if (e.frameBegin >= 0 && e.frameEnd >= 0) {
                            frameTotal = Math.max(0, Number(ts[e.frameEnd]! - ts[e.frameBegin]!) / 1e6);
                        }
                        frameTotal += outsideFrame;
                        if (e.sequence >= publishedSequence) {
                            publishedSequence = e.sequence;
                            latest = { stages, total, frameTotal, ...(e.overflowed ? { overflowed: true } : {}) };
                        }
                    })
                    .catch((error: unknown) => {
                        if (!disposed) {
                            if (e.sequence >= publishedSequence) {
                                publishedSequence = e.sequence;
                                latest = null;
                            }
                            console.warn("[fluid] GPU timestamp readback failed", error);
                        }
                    })
                    .finally(() => {
                        if (e.buf.mapState === "mapped") {
                            e.buf.unmap();
                        }
                        if (!disposed) {
                            free.push(e.buf);
                        }
                    })
            )
        );
    }

    return {
        commandSpan(encoder: GPUCommandEncoder, stage: string) {
            const p = allocPair();
            if (!p) {
                return undefined;
            }
            frameRecords.push({ stage, begin: p.begin, end: p.end, outsideFrame: true });
            encoder.beginComputePass({ label: "gpu-command-start", timestampWrites: { querySet, beginningOfPassWriteIndex: p.begin } }).end();
            return () => {
                encoder.beginComputePass({ label: "gpu-command-end", timestampWrites: { querySet, endOfPassWriteIndex: p.end } }).end();
            };
        },
        stageSpan(stage: string) {
            const p = allocPair();
            if (!p) {
                return undefined;
            }
            frameRecords.push({ stage, begin: p.begin, end: p.end });
            return {
                begin: { querySet, beginningOfPassWriteIndex: p.begin },
                end: { querySet, endOfPassWriteIndex: p.end },
            };
        },
        pass(stage: string): { querySet: GPUQuerySet; beginningOfPassWriteIndex: number; endOfPassWriteIndex: number } | undefined {
            const p = allocPair();
            if (!p) {
                return undefined;
            }
            frameRecords.push({ stage, begin: p.begin, end: p.end });
            return { querySet, beginningOfPassWriteIndex: p.begin, endOfPassWriteIndex: p.end };
        },
        beginFrame(): void {
            cursor = 0;
            frameRecords = [];
            frameBeginIdx = -1;
            frameEndIdx = -1;
            frameStopReserved = null;
            overflowed = false;
        },
        frameStart(encoder: GPUCommandEncoder): void {
            // Reserve BOTH envelope pairs first (start now, stop for later) so the frame
            // end can always be recorded even if the frame fills the query set.
            const startPair = allocPair();
            const stopPair = allocPair();
            if (!startPair || !stopPair) {
                return;
            }
            // Empty compute pass: its begin timestamp marks the frame's GPU start. A
            // pass with zero dispatches is valid and carries timestampWrites.
            encoder.beginComputePass({ label: "frame-start", timestampWrites: { querySet, beginningOfPassWriteIndex: startPair.begin, endOfPassWriteIndex: startPair.end } }).end();
            frameBeginIdx = startPair.begin;
            frameStopReserved = stopPair;
        },
        frameStop(encoder: GPUCommandEncoder): void {
            const p = frameStopReserved;
            if (!p) {
                return;
            }
            encoder.beginComputePass({ label: "frame-stop", timestampWrites: { querySet, beginningOfPassWriteIndex: p.begin, endOfPassWriteIndex: p.end } }).end();
            frameEndIdx = p.end;
        },
        resolveInto(encoder: GPUCommandEncoder): void {
            // Map previous frames' readbacks (their copies have been submitted).
            void reap();
            if (cursor === 0) {
                return;
            }
            const buf = free.pop();
            if (!buf) {
                // Every readback buffer is still in flight — skip this frame's readback
                // rather than risk an overlapping map. Timing simply pauses a frame.
                return;
            }
            const count = cursor;
            encoder.resolveQuerySet(querySet, 0, count, resolveBuf, 0);
            encoder.copyBufferToBuffer(resolveBuf, 0, buf, 0, count * 8);
            inflight.push({ buf, count, records: frameRecords.slice(), frameBegin: frameBeginIdx, frameEnd: frameEndIdx, overflowed, sequence: ++sequence });
        },
        collectSubmitted: reap,
        results(): { stages: Record<string, number>; total: number; frameTotal: number; overflowed?: boolean } | null {
            return latest;
        },
        dispose(): void {
            disposed = true;
            querySet.destroy();
            resolveBuf.destroy();
            for (const b of buffers) {
                b.destroy();
            }
            buffers.length = 0;
            free.length = 0;
            inflight.length = 0;
        },
    };
}
