import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { presetFromExportJson, type FluidExportJson } from "../../../lab/lite/src/demos/fluid/preset-io";
import {
    allocateFluidInflowCapacity,
    createFluidFlowState,
    createFluidInitialParticles,
    fluidPerParticleRecycleProbability,
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

    it("fills the complete pool from initial emitters when explicitly enabled alongside inflows", () => {
        const config: FluidFlowConfig = {
            emitters: [emitter("initial", "initial", [2, 3, 4]), emitter("inflow", "inflow", [20, 30, 40])],
            sinks: [],
            initialEmittersFillCapacity: true,
        };
        const particles = createFluidInitialParticles(128, config, 1);

        expect(particles).not.toBeNull();
        expect(particles!.activeCount).toBe(128);
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

describe("fluid flow preset compatibility", () => {
    const waterfallPreset = (): FluidExportJson =>
        JSON.parse(readFileSync(resolve(process.cwd(), "lab/public/fluid-presets/waterfall.pbmpm.liquid.high.json"), "utf8")) as FluidExportJson;

    it("marks the gridless Waterfall preset for demo-owned flow reconstruction", () => {
        const preset = presetFromExportJson(waterfallPreset());

        expect(preset.legacyFlow).toBe(true);
        expect(preset.emitters).toBeUndefined();
        expect(preset.sinks).toBeUndefined();
        expect(preset.grid).toBeUndefined();
    });

    it("migrates pre-format-4 emitter and sink positions to grid-local coordinates", () => {
        const json = waterfallPreset();
        json.formatVersion = 3;
        json.gridPosition = [10, 20, 30];
        json.gridSize = [40, 40, 40];
        json.emitters = [emitter("source", "inflow", [12, 23, 34])];
        json.sinks = [
            {
                id: "sink",
                name: "sink",
                enabled: true,
                transform: transform([8, 19, 35]),
                shape: { type: "box", size: [2, 2, 2] },
                targets: ["source"],
            },
        ];

        const preset = presetFromExportJson(json);

        expect(preset.legacyFlow).toBe(false);
        expect(preset.emitters?.[0]?.transform.position).toEqual([2, 3, 4]);
        expect(preset.sinks?.[0]?.transform.position).toEqual([-2, -1, 5]);
    });

    it("imports full-capacity initial allocation and per-particle sink semantics", () => {
        const json = waterfallPreset();
        json.formatVersion = 5;
        json.gridPosition = [10, 20, 30];
        json.gridSize = [40, 40, 40];
        json.initialEmittersFillCapacity = true;
        json.emitters = [emitter("source", "inflow", [2, 3, 4])];
        json.sinks = [
            {
                id: "sink",
                name: "sink",
                enabled: true,
                transform: transform([-2, -1, 5]),
                shape: { type: "box", size: [2, 2, 2] },
                targets: ["source"],
                perParticleRecycleRate: 0.7,
            },
        ];

        const preset = presetFromExportJson(json);

        expect(preset.initialEmittersFillCapacity).toBe(true);
        expect(preset.emitters).toEqual(json.emitters);
        expect(preset.sinks).toEqual(json.sinks);
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

    it("clamps the public per-particle recycle probability", () => {
        expect(fluidPerParticleRecycleProbability(0.7, 0.1)).toBeCloseTo(0.07);
        expect(fluidPerParticleRecycleProbability(30, 0.1)).toBe(1);
        expect(fluidPerParticleRecycleProbability(-3, 0.1)).toBe(0);
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

describe("per-particle fluid sinks", () => {
    afterEach(() => {
        vi.unstubAllGlobals();
    });

    const createState = () => {
        vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 });
        const device = {
            createBuffer: ({ size }: GPUBufferDescriptor) => ({ size, destroy: vi.fn() }),
            queue: { writeBuffer: vi.fn() },
        } as unknown as GPUDevice;
        return createFluidFlowState(device, 20, 0.1);
    };

    const flow = (sink: { volumeRate?: number; perParticleRecycleRate?: number }): FluidFlowConfig => ({
        emitters: [emitter("source", "inflow", [0, 1, 0])],
        sinks: [
            {
                id: "sink",
                name: "sink",
                enabled: true,
                transform: transform([0, 0, 0]),
                shape: { type: "box", size: [2, 2, 2] },
                targets: ["source"],
                ...sink,
            },
        ],
    });

    it("packs the per-particle mode and rate without a volume budget", () => {
        const state = createState();
        setFluidFlowConfig(state, flow({ perParticleRecycleRate: 0.7 }));

        const firstSinkOffset = 8 + 16 * 32;
        expect(state.u32[firstSinkOffset + 26]).toBe(0xffffffff);
        expect(state.u32[firstSinkOffset + 28]).toBe(2);
        expect(state.f32[firstSinkOffset + 30]).toBeCloseTo(0.7);
    });

    it("rejects ambiguous or invalid per-particle sink rates", () => {
        const state = createState();

        expect(() => setFluidFlowConfig(state, flow({ volumeRate: 1, perParticleRecycleRate: 0.7 }))).toThrow(/cannot define both/);
        expect(() => setFluidFlowConfig(state, flow({ perParticleRecycleRate: -1 }))).toThrow(/finite non-negative/);
    });
});
