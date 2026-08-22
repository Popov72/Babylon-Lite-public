import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { presetFromExportJson, type FluidExportJson } from "../../../../lab/lite/src/demos/fluid/preset-io";
import {
    allocateFluidInflowCapacity,
    countFluidInitialParticles,
    createFluidFlowState,
    createFluidInitialParticles,
    fluidPerParticleRecycleProbability,
    fluidVolumeBudget,
    legacyEmitterConfigToFluidFlow,
    legacyEmitterRelaunchProbability,
    prepareFluidFlowFrame,
    resetFluidParticleLifecycle,
    resetFluidFlowState,
    setFluidFlowConfig,
    updateFluidFlowEmitter,
    type EmitterConfig,
    type FluidEmitter,
    type FluidFlowConfig,
    type FluidShape,
} from "../../../../packages/babylon-lite/src/fluid/sim-common";

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

    it("can reserve dormant capacity while deriving an initial-only active prefix", () => {
        const config: FluidFlowConfig = { emitters: [emitter("initial", "initial", [2, 3, 4])], sinks: [] };
        const particles = createFluidInitialParticles(128, config, 1, undefined, true);

        expect(particles!.activeCount).toBe(8);
        expect(particles!.positions).toHaveLength(8 * 3);
    });

    it("combines authored, source, and analytical normal velocity only when configured", () => {
        const source: FluidEmitter = {
            ...emitter("initial", "initial", [0, 0, 0]),
            velocity: [1, 2, 3] as [number, number, number],
            sourceVelocity: [2, 0, -1] as [number, number, number],
            sourceVelocityFactor: 0.5,
            normalVelocity: 4,
        };

        const particles = createFluidInitialParticles(1, { emitters: [source], sinks: [] });

        expect([...particles!.velocities]).toEqual([6, 2, 2.5]);
    });

    it("seeds volume Initial emitters deterministically at rest-volume spacing", () => {
        const config: FluidFlowConfig = {
            emitters: [emitter("initial", "initial", [0, 0, 0]), emitter("inflow", "inflow", [10, 0, 0])],
            sinks: [],
        };

        const first = createFluidInitialParticles(128, config, 1)!;
        const second = createFluidInitialParticles(128, config, 1)!;

        expect([...second.positions]).toEqual([...first.positions]);
        expect(first.activeCount).toBe(8);
        let minimumDistance = Number.POSITIVE_INFINITY;
        for (let a = 0; a < first.activeCount; a++) {
            for (let b = a + 1; b < first.activeCount; b++) {
                const ai = a * 3;
                const bi = b * 3;
                minimumDistance = Math.min(
                    minimumDistance,
                    Math.hypot(
                        first.positions[ai]! - first.positions[bi]!,
                        first.positions[ai + 1]! - first.positions[bi + 1]!,
                        first.positions[ai + 2]! - first.positions[bi + 2]!
                    )
                );
            }
        }
        expect(minimumDistance).toBeCloseTo(1);
    });

    it("keeps lattice particles inside rotated non-uniformly scaled emitters", () => {
        const source: FluidEmitter = {
            ...emitter("initial", "initial", [3, 4, 5]),
            transform: {
                position: [3, 4, 5],
                rotation: [0, Math.SQRT1_2, 0, Math.SQRT1_2],
                scale: [2, 1, 0.5],
            },
        };
        const particles = createFluidInitialParticles(24, { emitters: [source], sinks: [], initialEmittersFillCapacity: true }, 1)!;

        for (let index = 0; index < particles.positions.length; index += 3) {
            const dx = particles.positions[index]! - 3;
            const dy = particles.positions[index + 1]! - 4;
            const dz = particles.positions[index + 2]! - 5;
            const localX = -dz / 2;
            const localY = dy;
            const localZ = dx / 0.5;
            expect(Math.abs(localX)).toBeLessThanOrEqual(1.000001);
            expect(Math.abs(localY)).toBeLessThanOrEqual(1.000001);
            expect(Math.abs(localZ)).toBeLessThanOrEqual(1.000001);
        }
    });

    it("discards Initial lattice sites outside the simulation domain without repacking them", () => {
        const source = emitter("initial", "initial", [0, 0, 0]);
        source.shape.size = [4, 4, 4];

        const particles = createFluidInitialParticles(64, { emitters: [source], sinks: [], initialEmittersFillCapacity: true }, 1, { min: [-1, -1, -1], max: [1, 1, 1] })!;
        const counts = countFluidInitialParticles(64, { emitters: [source], sinks: [], initialEmittersFillCapacity: true }, 1, {
            min: [-1, -1, -1],
            max: [1, 1, 1],
        })!;

        expect(particles.activeCount).toBe(8);
        expect(counts.activeCount).toBe(particles.activeCount);
        expect(counts.emitterCounts.get("initial")).toBe(8);
        expect(particles.emitterCounts.get("initial")).toBe(8);
        expectInsideBox(particles.positions, [0, 0, 0]);
        for (const coordinate of particles.positions) {
            expect(Math.abs(coordinate)).toBe(0.5);
        }
    });

    it("clips lattice sites to every supported volume shape", () => {
        const shapes: { shape: FluidShape; contains: (point: [number, number, number]) => boolean }[] = [
            { shape: { type: "box", size: [2, 2, 2] }, contains: ([x, y, z]) => Math.max(Math.abs(x), Math.abs(y), Math.abs(z)) <= 1.000001 },
            { shape: { type: "sphere", radius: 1 }, contains: ([x, y, z]) => x * x + y * y + z * z <= 1.000001 },
            {
                shape: { type: "cylinder", radius: 1, height: 2, innerRadius: 0.35 },
                contains: ([x, y, z]) => Math.abs(y) <= 1.000001 && x * x + z * z <= 1.000001 && x * x + z * z >= 0.35 ** 2 - 0.000001,
            },
            {
                shape: { type: "cone", bottomRadius: 1, topRadius: 0.25, height: 2 },
                contains: ([x, y, z]) => {
                    const radius = 1 + (0.25 - 1) * (y / 2 + 0.5);
                    return Math.abs(y) <= 1.000001 && x * x + z * z <= radius * radius + 0.000001;
                },
            },
            {
                shape: { type: "capsule", radius: 0.5, height: 2 },
                contains: ([x, y, z]) => {
                    const dy = y - Math.max(-0.5, Math.min(0.5, y));
                    return x * x + dy * dy + z * z <= 0.250001;
                },
            },
            {
                shape: {
                    type: "polygonPrism",
                    points: [
                        [-1, -0.5],
                        [1, -0.5],
                        [0.5, 1],
                        [-0.5, 1],
                    ],
                    thickness: 1,
                },
                contains: ([x, y, z]) =>
                    Math.abs(y) <= 0.500001 &&
                    z >= -0.500001 &&
                    z <= 1.000001 &&
                    x >= (z <= -0.5 ? -1 : -1 + (z + 0.5) / 3) - 0.000001 &&
                    x <= (z <= -0.5 ? 1 : 1 - (z + 0.5) / 3) + 0.000001,
            },
        ];

        for (const { shape, contains } of shapes) {
            const source: FluidEmitter = { ...emitter(shape.type, "initial", [0, 0, 0]), shape };
            const particles = createFluidInitialParticles(32, { emitters: [source], sinks: [], initialEmittersFillCapacity: true }, 1)!;
            for (let index = 0; index < particles.positions.length; index += 3) {
                expect(contains([particles.positions[index]!, particles.positions[index + 1]!, particles.positions[index + 2]!]), shape.type).toBe(true);
            }
        }
    });

    it("allocates deterministic lattice sites proportionally across Initial emitter volumes", () => {
        const small = emitter("small", "initial", [-10, 0, 0]);
        small.shape.size = [1, 1, 1];
        const large = emitter("large", "initial", [10, 0, 0]);
        large.shape.size = [2, 1, 1];
        const config: FluidFlowConfig = {
            emitters: [small, large, emitter("inflow", "inflow", [0, 10, 0])],
            sinks: [],
        };

        const particles = createFluidInitialParticles(100, config, 0.25)!;
        const smallCount = Array.from({ length: particles.activeCount }, (_, index) => particles.positions[index * 3]!).filter((x) => x < 0).length;

        expect(particles.activeCount).toBe(12);
        expect(smallCount).toBe(4);
    });

    it("updates an installed emitter transform and velocity without resetting its emission carry", () => {
        vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 });
        const writeBuffer = vi.fn();
        const device = {
            createBuffer: ({ size }: GPUBufferDescriptor) => ({ size, destroy: vi.fn() }),
            queue: { writeBuffer },
        } as unknown as GPUDevice;
        const state = createFluidFlowState(device, 20, 0.1);
        const source: FluidEmitter = {
            ...emitter("source", "inflow", [0, 1, 0]),
            sourceVelocity: [0, 0, 0] as [number, number, number],
            sourceVelocityFactor: 2,
        };
        setFluidFlowConfig(state, { emitters: [source], sinks: [] });
        state.emitterCarries[0] = 0.75;
        writeBuffer.mockClear();
        source.transform.position = [3, 4, 5];
        source.sourceVelocity = [1, -2, 0.5];
        source.normalVelocity = -1;

        updateFluidFlowEmitter(state, source);

        expect([...state.f32.slice(8, 11)]).toEqual([3, 4, 5]);
        expect(state.f32[23]).toBe(-1);
        expect([...state.f32.slice(32, 35)]).toEqual([2, -4, 1]);
        expect(state.emitterCarries[0]).toBe(0.75);
        expect(writeBuffer).toHaveBeenCalledOnce();
    });

    it("keeps an explicit empty flow graph empty", () => {
        const particles = createFluidInitialParticles(128, { emitters: [], sinks: [] });

        expect(particles).not.toBeNull();
        expect(particles!.activeCount).toBe(0);
        expect(particles!.positions).toHaveLength(0);
    });

    it("preserves legacy spawn-box seeding when no flow graph is installed", () => {
        expect(createFluidInitialParticles(128, null)).toBeNull();
    });
});

describe("fluid flow preset compatibility", () => {
    const waterfallPreset = (): FluidExportJson =>
        JSON.parse(readFileSync(resolve(process.cwd(), "lab/public/fluid-presets/waterfall.pbmpm.liquid.high.json"), "utf8")) as FluidExportJson;

    it("marks the Waterfall preset for demo-owned flow reconstruction while retaining its grid", () => {
        const preset = presetFromExportJson(waterfallPreset());

        expect(preset.legacyFlow).toBe(true);
        expect(preset.emitters).toBeUndefined();
        expect(preset.sinks).toBeUndefined();
        expect(preset.grid).toEqual({ position: [0, 30, 0], size: [120, 60, 120] });
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
        expect(preset.sinks).toEqual(json.sinks?.map((sink) => ({ ...sink, mode: "recycle" })));
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

    describe("fluid particle lifecycle initialization", () => {
        afterEach(() => {
            vi.unstubAllGlobals();
        });

        it("separates active, reserved warm-up, and free slots", () => {
            vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 });
            const writeBuffer = vi.fn();
            const device = {
                createBuffer: ({ size, label }: GPUBufferDescriptor) => ({ size, label, destroy: vi.fn() }),
                queue: { writeBuffer },
            } as unknown as GPUDevice;
            const state = createFluidFlowState(device, 6, 0.1);
            writeBuffer.mockClear();

            resetFluidParticleLifecycle(state, 2, 4);

            expect(state.activeCount).toBe(2);
            const data = writeBuffer.mock.calls[0]![2] as Uint32Array;
            expect([...data]).toEqual([2, 6, 0, 0, 1, 1, 2, 2, 0, 0]);
        });
    });

    it("prioritizes finite rates and divides remaining capacity among unlimited inflows", () => {
        expect([...allocateFluidInflowCapacity([3, 0, 0], [false, true, true], 8)]).toEqual([3, 3, 2]);
        expect([...allocateFluidInflowCapacity([6, 6], [false, false], 8, 1)]).toEqual([2, 6]);
    });

    it("waits for an inflow delay and budgets only the active part of the crossing frame", () => {
        vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 });
        const device = {
            createBuffer: ({ size }: GPUBufferDescriptor) => ({ size, destroy: vi.fn() }),
            queue: { writeBuffer: vi.fn() },
        } as unknown as GPUDevice;
        const state = createFluidFlowState(device, 20, 0.1);
        const source: FluidEmitter = {
            ...emitter("source", "inflow", [0, 1, 0]),
            delayBeforeStart: 1.5,
            volumeRate: state.particleVolume * 10,
        };
        setFluidFlowConfig(state, { emitters: [source], sinks: [] });

        expect(prepareFluidFlowFrame(state, 1)).toMatchObject({ emitActive: false, emitCount: 0, emitUnlimited: false });
        expect(state.u32[36]).toBe(0);
        expect(state.counterData[1]).toBe(0);
        expect(prepareFluidFlowFrame(state, 1)).toMatchObject({ emitActive: true, emitCount: 5, emitUnlimited: false });
        expect(state.u32[36]).toBe(1);
        expect(state.counterData[1]).toBe(5);

        resetFluidFlowState(state);
        expect(prepareFluidFlowFrame(state, 0.5)).toMatchObject({ emitActive: false, emitCount: 0, emitUnlimited: false });
        expect(state.elapsedSeconds).toBe(0.5);
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
        expect(state.u32[firstSinkOffset + 28]).toBe(2);
        expect(state.u32[firstSinkOffset + 29]).toBe(20);
        expect(state.f32[firstSinkOffset + 30]).toBe(3);
        expect(state.f32[firstSinkOffset + 31]).toBe(-2);
        expect(state.f32[8 + 15]).toBe(4);
        expect(state.f32[8 + 32 + 15]).toBe(8);

        const frame = prepareFluidFlowFrame(state, 0.1);
        expect(frame).toEqual({ flowActive: true, deleteActive: true, emitActive: false, emitCount: 0, emitUnlimited: false });
        expect(state.u32[2]).toBe(20);
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
                mode: "recycle",
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
        expect(state.u32[firstSinkOffset + 28]).toBe(1);
        expect(state.u32[firstSinkOffset + 29]).toBe(1);
        expect(state.f32[firstSinkOffset + 30]).toBeCloseTo(0.7);
    });

    it("packs delete sinks without requiring recycle targets", () => {
        const state = createState();
        const config = flow({ volumeRate: 1 });
        config.sinks[0]!.mode = "delete";
        config.sinks[0]!.targets = [];

        setFluidFlowConfig(state, config);

        const firstSinkOffset = 8 + 16 * 32;
        expect(state.u32[firstSinkOffset + 25]).toBe(0);
        expect(state.u32[firstSinkOffset + 28]).toBe(0);
    });

    it("rejects ambiguous or invalid per-particle sink rates", () => {
        const state = createState();

        expect(() => setFluidFlowConfig(state, flow({ volumeRate: 1, perParticleRecycleRate: 0.7 }))).toThrow(/cannot define both/);
        expect(() => setFluidFlowConfig(state, flow({ perParticleRecycleRate: -1 }))).toThrow(/finite non-negative/);
    });
});
