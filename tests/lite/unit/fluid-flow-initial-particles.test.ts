import { afterEach, describe, expect, it, vi } from "vitest";

import {
    allocateFluidInflowCapacity,
    createFluidFlowState,
    createFluidInitialParticles,
    fluidVolumeBudget,
    legacyEmitterConfigToFluidFlow,
    legacyEmitterRelaunchProbability,
    prepareFluidFlowFrame,
    setFluidFlowConfig,
    type EmitterConfig,
    type FluidFlowConfig,
} from "../../../packages/babylon-lite/src/fluid/sim-common";

const transform = (position: [number, number, number]) => ({
    position,
    rotation: [0, 0, 0, 1] as [number, number, number, number],
    scale: [1, 1, 1] as [number, number, number],
});

const emitter = (id: string, behavior: "initial" | "inflow", position: [number, number, number]) => ({
    id,
    name: id,
    enabled: true,
    behavior,
    transform: transform(position),
    shape: { type: "box" as const, size: [2, 2, 2] as [number, number, number] },
    sampling: "volume" as const,
    velocity: [0, 0, 0] as [number, number, number],
    velocitySpace: "world" as const,
    spread: 0,
});

function expectInsideBox(positions: Float32Array, center: [number, number, number]): void {
    for (let i = 0; i < positions.length; i += 3) {
        expect(positions[i]).toBeGreaterThanOrEqual(center[0] - 1);
        expect(positions[i]).toBeLessThanOrEqual(center[0] + 1);
        expect(positions[i + 1]).toBeGreaterThanOrEqual(center[1] - 1);
        expect(positions[i + 1]).toBeLessThanOrEqual(center[1] + 1);
        expect(positions[i + 2]).toBeGreaterThanOrEqual(center[2] - 1);
        expect(positions[i + 2]).toBeLessThanOrEqual(center[2] + 1);
    }
}

describe("fluid flow reset seeding", () => {
    it("reserves the complete pool for an inflow-only graph", () => {
        const config: FluidFlowConfig = { emitters: [emitter("source", "inflow", [8, 9, 10])], sinks: [] };
        const particles = createFluidInitialParticles(128, config, 1);

        expect(particles).not.toBeNull();
        expect(particles!.activeCount).toBe(0);
        expect(particles!.positions).toHaveLength(0);
    });

    it("activates initial emitters according to world volume", () => {
        const config: FluidFlowConfig = {
            emitters: [emitter("initial", "initial", [2, 3, 4]), emitter("inflow", "inflow", [20, 30, 40])],
            sinks: [],
        };
        const particles = createFluidInitialParticles(128, config, 1);

        expect(particles).not.toBeNull();
        expect(particles!.activeCount).toBe(8);
        expectInsideBox(particles!.positions, [2, 3, 4]);
    });

    it("activates the complete selected pool for an initial-only graph", () => {
        const config: FluidFlowConfig = { emitters: [emitter("initial", "initial", [2, 3, 4])], sinks: [] };
        const particles = createFluidInitialParticles(128, config, 100);

        expect(particles!.activeCount).toBe(128);
        expect(particles!.positions).toHaveLength(128 * 3);
    });

    it("preserves legacy spawn-box seeding when no enabled emitter exists", () => {
        expect(createFluidInitialParticles(128, { emitters: [], sinks: [] })).toBeNull();
    });
});

describe("fluid volume budgets", () => {
    it("carries fractional particles without carrying unmet whole-particle budget", () => {
        const first = fluidVolumeBudget(1, 0.25, 0.1, 0);
        const second = fluidVolumeBudget(1, 0.25, 0.1, first.carry);

        expect(first.count).toBe(2);
        expect(first.carry).toBeCloseTo(0.5);
        expect(second.count).toBe(3);
        expect(second.carry).toBeCloseTo(0);
    });

    it("prioritizes finite rates and divides remaining capacity among unlimited inflows", () => {
        expect([...allocateFluidInflowCapacity([3, 0, 0], [false, true, true], 8)]).toEqual([3, 3, 2]);
        expect([...allocateFluidInflowCapacity([6, 6], [false, false], 8, 1)]).toEqual([2, 6]);
    });
});

describe("legacy emitter compatibility", () => {
    afterEach(() => {
        vi.unstubAllGlobals();
    });

    const legacyConfig: EmitterConfig = {
        emitters: [
            { pos: [1, 2, 3], dir: [0, 1, 0], speed: 4, radius: 0.1 },
            { pos: [5, 6, 7], dir: [1, 0, 0], speed: 8, radius: 0.2 },
        ],
        intakeMin: [-10, -0.01, -10],
        intakeMax: [10, 0.01, 10],
        rate: 3,
        spread: 0.25,
        fixedStreamCount: 24,
        fixedStreamDrainY: -2,
    };

    it("preserves fixed-stream and per-particle rate metadata without a volume budget", () => {
        const flow = legacyEmitterConfigToFluidFlow(legacyConfig, 20);

        expect(flow?._legacyEmitter).toEqual({
            rate: 3,
            fixedStreamCount: 20,
            fixedStreamDrainY: -2,
            emitterSpeeds: [4, 8],
        });
        expect(flow?.sinks[0]?.volumeRate).toBeUndefined();
        expect(flow?.sinks[0]?.targets).toEqual(["legacy-emitter-0", "legacy-emitter-1"]);
        expect(flow?.emitters.at(-1)?.transform.position).toEqual([5, 6, 7]);
    });

    it("clamps the legacy per-particle frame probability", () => {
        expect(legacyEmitterRelaunchProbability(3, 0.1)).toBeCloseTo(0.3);
        expect(legacyEmitterRelaunchProbability(30, 0.1)).toBe(1);
        expect(legacyEmitterRelaunchProbability(-3, 0.1)).toBe(0);
    });

    it("packs and clears the explicit legacy state without changing the flow UBO layout", () => {
        vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 });
        const device = {
            createBuffer: ({ size }: GPUBufferDescriptor) => ({ size, destroy: vi.fn() }),
            queue: { writeBuffer: vi.fn() },
        } as unknown as GPUDevice;
        const state = createFluidFlowState(device, 20, 0.1);
        const flow = legacyEmitterConfigToFluidFlow(legacyConfig, 20)!;

        setFluidFlowConfig(state, flow);

        const firstSinkOffset = 8 + 16 * 32;
        expect(state.legacyEmitter).toBe(flow._legacyEmitter);
        expect(state.u32[firstSinkOffset + 28]).toBe(1);
        expect(state.u32[firstSinkOffset + 29]).toBe(20);
        expect(state.f32[firstSinkOffset + 30]).toBe(3);
        expect(state.f32[firstSinkOffset + 31]).toBe(-2);
        expect(state.f32[8 + 15]).toBe(4);
        expect(state.f32[8 + 32 + 15]).toBe(8);

        const frame = prepareFluidFlowFrame(state, 0.1, 10, 10);
        expect(frame).toEqual({ particles: null, recycleActive: true });
        expect(state.u32[2]).toBe(10);
        expect(state.f32[4]).toBeCloseTo(0.1);

        setFluidFlowConfig(state, { emitters: [], sinks: [] });

        expect(state.legacyEmitter).toBeNull();
        expect(state.u32[firstSinkOffset + 28]).toBe(0);
    });
});
