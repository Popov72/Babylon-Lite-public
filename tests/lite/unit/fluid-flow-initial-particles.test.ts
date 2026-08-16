import { describe, expect, it } from "vitest";

import { allocateFluidInflowCapacity, createFluidInitialParticles, fluidVolumeBudget, type FluidFlowConfig } from "../../../packages/babylon-lite/src/fluid/sim-common";

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
