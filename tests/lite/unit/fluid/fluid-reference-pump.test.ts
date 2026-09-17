import { afterEach, describe, expect, it, vi } from "vitest";
import { createFluidReferencePump, queueFluidReferenceChange, startFluidReferencePump, stopFluidReferencePump } from "../../../../lab/lite/src/demos/fluid/reference-pump";
import { fluidReferenceTiming } from "../../../../lab/lite/src/demos/fluid/reference-timing";

afterEach(() => vi.unstubAllGlobals());

function frameQueue() {
    let next = 1;
    const callbacks = new Map<number, FrameRequestCallback>();
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
        const id = next++;
        callbacks.set(id, callback);
        return id;
    });
    vi.stubGlobal("cancelAnimationFrame", (id: number) => callbacks.delete(id));
    return {
        get size() {
            return callbacks.size;
        },
        run(time: number) {
            const entry = callbacks.entries().next().value;
            if (!entry) {
                throw new Error("No frame was scheduled.");
            }
            callbacks.delete(entry[0]);
            entry[1](time);
        },
    };
}

describe("serialized reference UI pump", () => {
    it("renders completed states and never renders or overlaps another step while work is pending", async () => {
        const frames = frameQueue();
        let finish!: () => void;
        const work = new Promise<void>((resolve) => {
            finish = resolve;
        });

        describe("reference UI animation timing", () => {
            it("maps physical 50-Hz steps to a 25-FPS authored animation", () => {
                expect(
                    fluidReferenceTiming({
                        source: { application: "FLIP add-on", settings: { timeline: { fps: 25, fpsBase: 1, simulationFps: 50 } } },
                        simulationTimeScale: 1,
                    })
                ).toEqual({ frameDelta: 0.02, animationRate: 2 });
            });

            it("does not apply the native time scale twice", () => {
                expect(
                    fluidReferenceTiming({
                        source: {
                            application: "FLIP add-on",
                            settings: {
                                timeline: { fps: 25 },
                                domain: { simulation: { frame_rate_mode: "FRAME_RATE_MODE_CUSTOM", frame_rate_custom: 50, time_scale: 2 } },
                            },
                        },
                        simulationTimeScale: 2,
                    })
                ).toEqual({ frameDelta: 0.02, animationRate: 1 });
            });

            it("keeps ordinary demos on their existing shared clock and rejects malformed authored rates", () => {
                expect(fluidReferenceTiming()).toEqual({ frameDelta: 1 / 60, animationRate: 1 });
                expect(() => fluidReferenceTiming({ source: { application: "FLIP add-on", settings: { timeline: { fps: 0 } } } })).toThrow("timeline FPS");
            });
        });
        const render = vi.fn();
        const step = vi.fn(() => work);
        const onError = vi.fn();
        const pump = createFluidReferencePump({ render, step, shouldStep: () => true, onError, suspend: vi.fn(), resume: vi.fn() });
        startFluidReferencePump(pump);
        frames.run(10);
        await Promise.resolve();
        expect(render).toHaveBeenCalledTimes(1);
        expect(step).toHaveBeenCalledTimes(1);
        expect(frames.size).toBe(0);
        startFluidReferencePump(pump);
        expect(frames.size).toBe(0);
        finish();
        await pump.pending;
        expect(frames.size).toBe(1);
        expect(onError).not.toHaveBeenCalled();
        stopFluidReferencePump(pump);
    });

    it("drains an in-flight step before reset or replacement and resumes only after all queued changes", async () => {
        const frames = frameQueue();
        let finish!: () => void;
        const work = new Promise<void>((resolve) => {
            finish = resolve;
        });
        const changes: number[] = [];
        const resume = vi.fn();
        const pump = createFluidReferencePump({
            render: vi.fn(),
            step: () => work,
            shouldStep: () => true,
            onError: vi.fn(),
            suspend: vi.fn(),
            resume,
        });
        startFluidReferencePump(pump);
        frames.run(10);
        const first = queueFluidReferenceChange(pump, () => {
            changes.push(1);
        });
        const second = queueFluidReferenceChange(pump, () => {
            changes.push(2);
        });
        await Promise.resolve();
        expect(changes).toEqual([]);
        finish();
        await first;
        await second;
        expect(changes).toEqual([1, 2]);
        expect(resume).toHaveBeenCalledTimes(1);
        expect(frames.size).toBe(0);
    });

    it("keeps paused rendering active without submitting simulation work", () => {
        const frames = frameQueue();
        const render = vi.fn();
        const step = vi.fn(async () => {});
        const pump = createFluidReferencePump({ render, step, shouldStep: () => false, onError: vi.fn(), suspend: vi.fn(), resume: vi.fn() });
        startFluidReferencePump(pump);
        frames.run(10);
        frames.run(26);
        expect(render).toHaveBeenCalledTimes(2);
        expect(step).not.toHaveBeenCalled();
        stopFluidReferencePump(pump);
    });
});
