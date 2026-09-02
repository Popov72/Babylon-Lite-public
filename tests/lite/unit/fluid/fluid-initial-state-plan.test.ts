import { describe, expect, it } from "vitest";

import {
    createFluidInitialStatePlanCache,
    fluidInitialEmitterVolume,
    planFluidInitialState,
    resolveFluidReconfigurationPlan,
} from "../../../../packages/babylon-lite/src/fluid/core/initial-state-plan";
import { createFluidInitialParticles, type FluidEmitter, type FluidFlowConfig, type FluidShape } from "../../../../packages/babylon-lite/src/fluid/core/sim-common";

const transform = (position: [number, number, number]) => ({
    position,
    rotation: [0, 0, 0, 1] as [number, number, number, number],
    scale: [1, 1, 1] as [number, number, number],
});

function initialEmitter(id: string, shape: FluidShape): FluidEmitter {
    return {
        id,
        name: id,
        enabled: true,
        behavior: "initial",
        transform: transform([0.4, 0, 0]),
        shape,
        sampling: "volume",
        velocity: [0, 0, 0],
        velocitySpace: "world",
        spread: 0,
    };
}

function flowFor(shape: FluidShape): FluidFlowConfig {
    return {
        emitters: [
            initialEmitter(shape.type, shape),
            {
                ...initialEmitter("inflow", { type: "box", size: [1, 1, 1] }),
                behavior: "inflow",
                transform: transform([5, 0, 0]),
            },
        ],
        sinks: [],
    };
}

describe("fluid initial-state planning", () => {
    const bounds = {
        min: [-0.5, -0.75, -0.75] as [number, number, number],
        max: [0.5, 0.75, 0.75] as [number, number, number],
    };
    const shapes: FluidShape[] = [
        { type: "box", size: [2, 2, 2] },
        { type: "sphere", radius: 1 },
        { type: "cylinder", radius: 1, height: 2, innerRadius: 0.25 },
        { type: "cone", bottomRadius: 1, topRadius: 0.25, height: 2 },
        { type: "capsule", radius: 0.5, height: 2 },
        {
            type: "polygonPrism",
            points: [
                [-1, -0.5],
                [1, -0.5],
                [0.5, 1],
                [-0.5, 1],
            ],
            thickness: 1,
        },
    ];

    it.each(shapes.map((shape) => [shape.type, shape] as const))("matches generated clipped lattice sites for %s emitters", (_type, shape) => {
        const flow = flowFor(shape);
        const particleCapacity = 64;
        const particleVolume = 0.125;
        const plan = planFluidInitialState({
            particleCapacity,
            particleVolume,
            flow,
            bounds,
            deriveInitialCount: true,
        });
        const generated = createFluidInitialParticles(particleCapacity, flow, particleVolume, bounds, true)!;
        const requiredGenerated = createFluidInitialParticles(
            plan.diagnostics.authoredRequiredCount,
            { ...flow, initialEmittersFillCapacity: false },
            particleVolume,
            bounds,
            true
        )!;

        expect(plan.activeCount).toBe(generated.activeCount);
        expect([...plan.emitterCounts]).toEqual([...generated.emitterCounts]);
        expect(plan.requiredCount).toBe(requiredGenerated.activeCount);
        expect(plan.emitters[0]).toMatchObject({
            id: shape.type,
            shape: shape.type,
            sampling: "volume",
            clippedRequiredCount: requiredGenerated.emitterCounts.get(shape.type),
            activeCount: generated.emitterCounts.get(shape.type),
        });
        expect(plan.clippedVolume).toBe(plan.requiredCount * particleVolume);
        expect(plan.diagnostics.exactClippedSites).toBe(true);
    });

    it("reports authored and bounds-clipped box demand separately", () => {
        const flow = flowFor({ type: "box", size: [2, 2, 2] });
        const plan = planFluidInitialState({
            particleCapacity: 128,
            particleVolume: 1,
            flow,
            bounds,
            deriveInitialCount: true,
        });

        expect(plan.authoredVolume).toBe(8);
        expect(plan.diagnostics.authoredRequiredCount).toBe(8);
        expect(plan.requiredCount).toBeLessThan(8);
        expect(plan.emitters[0]!.authoredVolume).toBe(8);
        expect(plan.emitters[0]!.clippedVolume).toBe(plan.emitters[0]!.clippedRequiredCount);
    });

    it("normalizes invalid scalar inputs without producing non-finite fitting state", () => {
        const plan = planFluidInitialState({
            particleCapacity: Number.NaN,
            particleVolume: Number.POSITIVE_INFINITY,
            flow: flowFor({ type: "box", size: [2, 2, 2] }),
            bounds,
            deriveInitialCount: true,
        });

        expect(plan.resolutionFitting.particleCapacity).toBe(0);
        expect(plan.resolutionFitting.particleVolume).toBe(1);
        expect(plan.diagnostics).toMatchObject({
            particleCapacityNormalized: true,
            particleVolumeNormalized: true,
        });
        expect(Object.values(plan.resolutionFitting).every(Number.isFinite)).toBe(true);
    });

    it("combines initial demand with planner-backed contextual capacity", () => {
        const flow = flowFor({ type: "box", size: [2, 2, 2] });
        const plan = resolveFluidReconfigurationPlan({
            allocation: {
                method: "PBF",
                particleCount: 100,
                gridDim: [4, 4, 4],
                limits: {
                    maxStorageBufferBindingSize: 100_000,
                    maxBufferSize: 100_000,
                },
            },
            particleVolume: 1,
            flow,
            bounds,
            deriveInitialCount: true,
        });

        expect(plan.particleCapacity.capacity).toBe(6_250);
        expect(plan.allocation.dimensions.particleCount).toBe(100);
        expect(plan.resolutionFitting).toMatchObject({
            particleCapacity: 100,
            maximumParticleCapacity: 6_250,
            fitsDeviceLimits: true,
            fitsInitialDemand: true,
        });
    });

    it("changes initial-state identity only when reset-time placement inputs change", () => {
        const base = {
            particleCapacity: 128,
            particleVolume: 0.125,
            flow: flowFor({ type: "box", size: [2, 2, 2] }),
            bounds,
            deriveInitialCount: true,
        };
        const original = planFluidInitialState(base).initialStateKey;
        const runtimeOnlyFlow = structuredClone(base.flow);
        runtimeOnlyFlow.emitters[1]!.velocity = [3, 4, 5];
        expect(planFluidInitialState({ ...base, flow: runtimeOnlyFlow }).initialStateKey).toBe(original);

        const movedInitialFlow = structuredClone(base.flow);
        movedInitialFlow.emitters[0]!.transform.position[0] += 0.25;
        expect(planFluidInitialState({ ...base, flow: movedInitialFlow }).initialStateKey).not.toBe(original);
        expect(planFluidInitialState({ ...base, particleCapacity: 129 }).initialStateKey).not.toBe(original);
    });

    it("reuses exact plans by reset-time identity and exposes authored initial volume", () => {
        const flow = flowFor({ type: "box", size: [2, 2, 2] });
        const input = {
            particleCapacity: 128,
            particleVolume: 0.125,
            flow,
            bounds,
            deriveInitialCount: true,
        };
        const cache = createFluidInitialStatePlanCache(2);
        const first = cache.resolve(input);

        expect(cache.resolve(structuredClone(input))).toBe(first);
        expect(fluidInitialEmitterVolume(flow)).toBe(first.authoredVolume);

        cache.clear();
        expect(cache.resolve(input)).not.toBe(first);
    });

});
