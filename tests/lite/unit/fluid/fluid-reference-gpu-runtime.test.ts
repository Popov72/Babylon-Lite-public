import { afterEach, describe, expect, it, vi } from "vitest";
import { collectFlipReferenceGpuStatus, recordFlipReferenceFrame } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/gpu-runtime";

afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
});

function recordingFixture() {
    const pass = { setPipeline: vi.fn(), setBindGroup: vi.fn(), dispatchWorkgroups: vi.fn(), dispatchWorkgroupsIndirect: vi.fn(), end: vi.fn() };
    const encoder = { beginComputePass: vi.fn(() => pass), copyBufferToBuffer: vi.fn() };
    const queue = { writeBuffer: vi.fn(), submit: vi.fn(), onSubmittedWorkDone: vi.fn() };
    const readbackData = new ArrayBuffer(208);
    const readback = {
        mapState: "unmapped",
        mapAsync: vi.fn(async () => {
            readback.mapState = "mapped";
        }),
        getMappedRange: () => readbackData,
        unmap: vi.fn(() => {
            readback.mapState = "unmapped";
        }),
    };
    const gpu = {
        core: {
            _device: { queue },
            _uniformBuffer: {},
            _params: new Float32Array(44),
            _cells: 8,
            _reductionGroups: 1,
            _extrapolationLayers: 12,
            _maxPressureIterations: 400,
            capacity: 128,
            _removalEnabled: true,
            referenceNumerics: true,
            _extremeRemoval: false,
            _controlReadback: new Float32Array(8),
            _statusReadback: new Uint32Array(8),
            elapsedSeconds: 0,
        },
        frameData: new Float32Array(4),
        pressureWorkgroupSize: 1024,
        frameBuffer: {},
        argumentBuffer: {},
        pipelines: new Proxy<Record<string, string>>({}, { get: (_target, entry) => entry }),
        bindGroups: {},
        error: null,
        telemetryError: null,
        timestepDiagnostics: { deferredSeconds: 0, droppedSeconds: 0, saturated: false },
        disposed: false,
        slots: [{ buffer: readback, pending: false, mapping: null, sequence: 0 }],
        sequence: 0,
        publishedSequence: 0,
        publishedCount: 0,
        lastEncoder: null,
    };
    return { gpu, encoder, queue, readback, readbackData, pass };
}

describe("GPU-resident Reference frame recording", () => {
    it("bounds parallel pressure to the minimum substeps and preserves the search direction on resume", () => {
        const f = recordingFixture();
        f.gpu.core.capacity = 65536;
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.02, 2, 8, 0.01, 5, null);
        const entries = f.pass.setPipeline.mock.calls.map(([entry]) => entry);
        expect(entries.filter((entry) => entry === "initializeParallelGpuPressure")).toHaveLength(2);
        expect(entries.filter((entry) => entry === "applyParallelPressure")).toHaveLength(256);
        expect(entries.filter((entry) => entry === "finishParallelPressureIteration")).toHaveLength(256);
        expect(entries).not.toContain("alphaParallelPressure");
        expect(entries.filter((entry) => entry === "solveGpuPressure")).toHaveLength(6);
        const finish = entries.indexOf("finishParallelPressure");
        expect(entries.slice(finish, finish + 4)).toEqual(["finishParallelPressure", "continueParallelPressure", "resumeGpuPressure", "updateGpuDispatch"]);
        expect(f.queue.submit).not.toHaveBeenCalled();
        expect(f.readback.mapAsync).not.toHaveBeenCalled();
    });

    it.each([
        { capacity: 262144, workgroup: 1024 },
        { capacity: 65536, workgroup: 256 },
    ])("keeps vector updates parallel for $capacity particles with $workgroup lanes", ({ capacity, workgroup }) => {
        const f = recordingFixture();
        f.gpu.core.capacity = capacity;
        f.gpu.pressureWorkgroupSize = workgroup;
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.01, 1, 1, 0.01, 5, null);
        const entries = f.pass.setPipeline.mock.calls.map(([entry]) => entry);
        expect(entries.filter((entry) => entry === "alphaParallelPressure")).toHaveLength(128);
        expect(entries.filter((entry) => entry === "updateParallelPressure")).toHaveLength(128);
        expect(entries).not.toContain("finishParallelPressureIteration");
    });

    it("never records more parallel iterations than the existing pressure budget", () => {
        const f = recordingFixture();
        f.gpu.core.capacity = 65536;
        f.gpu.core._maxPressureIterations = 3;
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.01, 1, 1, 0.01, 5, null);
        expect(f.pass.setPipeline.mock.calls.filter(([entry]) => entry === "applyParallelPressure")).toHaveLength(3);
    });

    it("does not record a parallel prefix when compaction scratch was not allocated", () => {
        const f = recordingFixture();
        f.gpu.core.capacity = 65536;
        f.gpu.core._removalEnabled = false;
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.01, 1, 1, 0.01, 5, null);
        const entries = f.pass.setPipeline.mock.calls.map(([entry]) => entry);
        expect(entries).not.toContain("initializeParallelGpuPressure");
        expect(entries).toContain("solveGpuPressure");
    });

    it("applies an installed force before P2G and gates invalid force output before transfer", () => {
        const f = recordingFixture();
        const force = { pipeline: "manual-force" as never, bindGroup: {} as GPUBindGroup };
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.02, 1, 1, 0.02, 5, null, null, false, force);
        const entries = f.pass.setPipeline.mock.calls.map(([entry]) => entry);
        const index = entries.indexOf("manual-force");
        expect(index).toBeGreaterThan(entries.indexOf("beginGpuSubstep"));
        expect(entries[index + 1]).toBe("updateGpuDispatch");
        expect(index).toBeLessThan(entries.indexOf("particleToGrid"));
        expect(index).toBeLessThan(entries.indexOf("snapshotAndForce"));
        expect(f.pass.dispatchWorkgroupsIndirect).toHaveBeenCalledWith(f.gpu.argumentBuffer, 36);
        expect(f.queue.submit).not.toHaveBeenCalled();
    });

    it("cleans velocity outliers before compaction, CFL finalization and publication", () => {
        const f = recordingFixture();
        f.gpu.core._extremeRemoval = true;
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.02, 1, 1, 0.02, 5, null);
        const entries = f.pass.setPipeline.mock.calls.map(([entry]) => entry);
        expect(entries.indexOf("countExtremeGroups")).toBeGreaterThan(entries.indexOf("gridToParticles"));
        expect(entries.indexOf("chooseExtremeThreshold")).toBeGreaterThan(entries.indexOf("countExtremeGroups"));
        expect(entries.indexOf("scanSurvivors")).toBeGreaterThan(entries.indexOf("chooseExtremeThreshold"));
        expect(entries.indexOf("finishGpuSubstep")).toBeGreaterThan(entries.indexOf("commitSurvivors"));
        expect(entries.indexOf("publishGpuParticles")).toBeGreaterThan(entries.indexOf("finishGpuSubstep"));
    });

    it("records fused clearing and conditioning before preparing the pressure cache", () => {
        const f = recordingFixture();
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.01, 1, 1, 0.01, 5, null);
        const entries = f.pass.setPipeline.mock.calls.map(([entry]) => entry);
        expect(entries).not.toContain("clearLists");
        expect(entries).not.toContain("clearRemoval");
        expect(entries).not.toContain("conditionSolidVelocities");
        expect(entries.filter((entry) => entry === "buildMatrix")).toHaveLength(1);
        expect(entries.indexOf("clearGpuParticleStage")).toBeLessThan(entries.indexOf("linkParticles"));
        expect(entries.indexOf("conditionGpuMatrix")).toBeGreaterThan(entries.indexOf("markGpuClosedPockets"));
        expect(entries.indexOf("prepareGpuPressure")).toBeGreaterThan(entries.indexOf("fixGpuPressureGauges"));
        expect(entries.indexOf("initializeGpuPressure")).toBeGreaterThan(entries.indexOf("prepareGpuPressure"));
        expect(entries.indexOf("solveGpuPressure")).toBeGreaterThan(entries.indexOf("initializeGpuPressure"));
        expect(entries.indexOf("project")).toBeGreaterThan(entries.indexOf("solveGpuPressure"));
    });

    it("exposes skipped-time warnings without converting a valid partial frame into a fatal error", async () => {
        vi.stubGlobal("GPUMapMode", { READ: 1 });
        const f = recordingFixture();
        const words = new Uint32Array(f.readbackData);
        const floats = new Float32Array(f.readbackData);
        words[0] = 32;
        words[49] = 32;
        floats[16] = 0.002;
        floats[17] = 0.018;
        floats[18] = 0.018;
        floats[19] = 0.002;
        f.gpu.slots[0]!.pending = true;
        await collectFlipReferenceGpuStatus(f.gpu as never);
        expect(f.gpu.error).toBeNull();
        expect(f.gpu.timestepDiagnostics.saturated).toBe(true);
        expect(f.gpu.timestepDiagnostics.droppedSeconds).toBeCloseTo(0.018);
        expect(f.gpu.timestepDiagnostics.deferredSeconds).toBe(0);
        expect(f.gpu.core.elapsedSeconds).toBeCloseTo(0.002);
        expect(f.gpu.publishedCount).toBe(32);
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.02, 2, 2, 0.001, 5, null);
        expect(f.pass.dispatchWorkgroupsIndirect).toHaveBeenCalled();
    });

    it("reports diagnostic mapping failures without stopping GPU-controlled physics", async () => {
        vi.stubGlobal("GPUMapMode", { READ: 1 });
        const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
        const f = recordingFixture();
        f.gpu.slots[0]!.pending = true;
        f.readback.mapAsync.mockRejectedValue(new Error("Readback unavailable"));
        await collectFlipReferenceGpuStatus(f.gpu as never);
        expect(f.gpu.error).toBeNull();
        expect(f.gpu.telemetryError).toBeInstanceOf(Error);
        expect(warning).toHaveBeenCalledOnce();
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.02, 2, 8, 0.01, 5, null);
        expect(f.pass.dispatchWorkgroupsIndirect).toHaveBeenCalled();
    });

    it("records without submitting, mapping the current frame, or waiting for GPU completion", () => {
        const f = recordingFixture();
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.02, 2, 8, 0.01, 5, null);
        expect(f.queue.submit).not.toHaveBeenCalled();
        expect(f.queue.onSubmittedWorkDone).not.toHaveBeenCalled();
        expect(f.readback.mapAsync).not.toHaveBeenCalled();
        expect(f.pass.dispatchWorkgroupsIndirect).toHaveBeenCalled();
        expect(f.encoder.copyBufferToBuffer).toHaveBeenCalledTimes(4);
        expect(f.gpu.frameData).toEqual(new Float32Array([0.02, 0.01, 2, 5]));
    });

    it("does not use stale CPU zero counts to suppress GPU-owned simulation or publication", () => {
        const f = recordingFixture();
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.02, 2, 8, 0.01, 5, null);
        expect(f.gpu.publishedCount).toBe(0);
        expect(f.pass.dispatchWorkgroups).toHaveBeenLastCalledWith(1);
        expect(f.pass.dispatchWorkgroupsIndirect).toHaveBeenCalledWith(f.gpu.argumentBuffer, 36);
    });

    it("prevents two frames in one encoder from overwriting their shared frame uniforms", () => {
        const f = recordingFixture();
        recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.02, 2, 8, 0.01, 5, null);
        expect(() => recordFlipReferenceFrame(f.gpu as never, f.encoder as never, 0.01, 2, 8, 0.01, 5, null)).toThrow("one simulation frame per command encoder");
        expect(f.queue.writeBuffer).toHaveBeenCalledTimes(2);
    });
});
