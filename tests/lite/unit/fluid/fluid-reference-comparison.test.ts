import { describe, expect, it } from "vitest";
import { compareParticles, measureParticles } from "../../../../scripts/fluid-reference/metrics";
import { loadComparisonSolids, sampleSdf, updateComparisonSolids } from "../../../../scripts/fluid-reference/solids";
import { sampleReferenceObstacleFrames } from "../../../../scripts/fluid-reference/animation";
import { mat4Translation } from "../../../../packages/babylon-lite/src/math/mat4-translation";
import { decodeParticleState } from "../../../../scripts/fluid-reference/types";
import type { ParticleState, ReferenceCase, ReferenceGrid } from "../../../../scripts/fluid-reference/types";

const grid: ReferenceGrid = { origin: [0, 0, 0], dimensions: [4, 4, 4], cellSize: 1 };
const particles = (positions: number[]): ParticleState => ({
    positions: Float32Array.from(positions),
    velocities: new Float32Array(positions.length),
});

describe("FLIP Reference comparison measurements", () => {
    it("does not require marker ordering to match", () => {
        const a = particles([0.2, 0.4, 0.6, 2.2, 2.4, 2.6]);
        const b = particles([2.2, 2.4, 2.6, 0.2, 0.4, 0.6]);
        expect(compareParticles(a, b, grid)).toEqual({
            centerDistance: 0,
            occupancyIntersectionOverUnion: 1,
            normalizedOccupancyDifference: 0,
            countDifference: 0,
        });
    });

    it("separates marker count from normalized liquid occupancy", () => {
        const a = particles([0.2, 0.4, 0.6]);
        const b = particles([0.2, 0.4, 0.6, 0.2, 0.4, 0.6]);
        expect(compareParticles(a, b, grid).normalizedOccupancyDifference).toBe(0);
        expect(compareParticles(a, b, grid).countDifference).toBe(-1);
    });

    it("reports disjoint occupancy and escaped markers instead of clamping them into the domain", () => {
        const a = particles([0.2, 0.4, 0.6]);
        const b = particles([-1.2, 0.4, 0.6]);
        expect(compareParticles(a, b, grid).normalizedOccupancyDifference).toBe(1);
        expect(compareParticles(a, b, grid).occupancyIntersectionOverUnion).toBe(0);
        expect(measureParticles(b, grid).outsideParticles).toBe(1);
    });

    it("rejects nonfinite states rather than emitting success-shaped metrics", () => {
        expect(() => measureParticles(particles([0, NaN, 0]), grid)).toThrow("Nonfinite");
    });

    it("represents a fully pruned state without inventing a centroid or losing its count difference", () => {
        const empty = particles([]);
        expect(measureParticles(empty, grid)).toMatchObject({ count: 0, center: null, meanKineticEnergy: null });
        expect(compareParticles(empty, particles([1, 1, 1]), grid)).toMatchObject({ centerDistance: null, normalizedOccupancyDifference: 1, countDifference: -1 });
        expect(compareParticles(empty, empty, grid).normalizedOccupancyDifference).toBe(0);
        expect(decodeParticleState(new ArrayBuffer(8)).positions).toHaveLength(0);
    });

    it("checks snapshot count and byte length", () => {
        const buffer = new ArrayBuffer(32);
        new DataView(buffer).setUint32(0, 1, true);
        new Float32Array(buffer, 8, 6).set([1, 2, 3, 4, 5, 6]);
        expect(Array.from(decodeParticleState(buffer).velocities)).toEqual([4, 5, 6]);
        new DataView(buffer).setUint32(0, 2, true);
        expect(() => decodeParticleState(buffer)).toThrow("byte length");
    });
});

describe("FLIP Reference collision inputs", () => {
    const local = {
        origin: [-1, -1, -1] as [number, number, number],
        dimensions: [5, 5, 5] as [number, number, number],
        cellSize: 0.5,
        file: "plane.f32",
        distances: Float32Array.from({ length: 125 }, (_, i) => -1 + (i % 5) * 0.5),
    };

    it("interpolates nodal signed distance without changing the cell-center convention", () => {
        expect(sampleSdf(local, 0.25, 0.3, -0.1)).toBeCloseTo(0.25);
        expect(sampleSdf(local, -0.25, 0.3, -0.1)).toBeCloseTo(-0.25);
    });

    it("includes the native domain shell even without an authored static SDF", async () => {
        const input: ReferenceCase = {
            version: 1,
            label: "closed domain",
            grid: { origin: [0, 0, 0], dimensions: [2, 2, 2], cellSize: 0.5 },
            domainInset: 0.1,
            startFrame: 0,
            frames: 1,
            simulationFps: 50,
            timelineFps: 25,
            substeps: 2,
            gravity: [0, -9.81, 0],
            picFraction: 0.05,
            initialState: "initial.bin",
            obstacles: [],
            meshes: [],
            camera: { alpha: 0, beta: 1, radius: 2, target: [0, 0, 0], fov: 0.8, mirrorX: false },
            provenance: {},
        };
        const state = await loadComparisonSolids(input);
        expect(state.distances).toHaveLength(27);
        expect(state.distances[0]).toBeCloseTo(-0.1);
        expect(state.distances[13]).toBeCloseTo(0.4);
        expect(state.distances[26]).toBeCloseTo(-0.1);
    });

    it("uses native backward/forward differences and zero startup backward velocity", () => {
        const frames = [mat4Translation(0, 0, 0), mat4Translation(0.02, 0, 0), mat4Translation(0.08, 0, 0)];
        const fast = sampleReferenceObstacleFrames(frames, 2, 50);
        const slow = sampleReferenceObstacleFrames(frames, 2, 25);
        expect(fast.transforms).toEqual(slow.transforms);
        expect(fast.transforms[1]![12]).toBeCloseTo(0.01);
        expect(fast.transforms[3]![12]).toBeCloseTo(0.05);
        expect(fast.velocityTransforms[0]![12]).toBe(0);
        expect(fast.velocityTransforms[1]![12]).toBeCloseTo(0.5);
        expect(fast.velocityTransforms[2]![12]).toBeCloseTo(1);
        expect(fast.velocityTransforms[3]![12]).toBeCloseTo(2);
        expect(slow.velocityTransforms[3]![12]).toBeCloseTo(1);
    });

    it("applies the supplied physical velocity transform at the sampled local point", () => {
        const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const moved = [...identity];
        moved[12] = 0.1;
        const inverse = [...identity];
        inverse[12] = -0.1;
        const velocity = new Array<number>(16).fill(0);
        velocity[12] = 10;
        const input: ReferenceCase = {
            version: 1,
            label: "moving plane",
            grid: { origin: [0, 0, 0], dimensions: [1, 1, 1], cellSize: 0.25 },
            domainInset: 0,
            startFrame: 150,
            frames: 2,
            simulationFps: 50,
            timelineFps: 25,
            substeps: 2,
            gravity: [0, -9.81, 0],
            picFraction: 0.05,
            initialState: "initial.bin",
            obstacles: [
                {
                    ...local,
                    name: "wall",
                    transforms: [identity, moved],
                    inverseTransforms: [identity, inverse],
                    velocityTransforms: [new Array<number>(16).fill(0), velocity],
                },
            ],
            meshes: [],
            camera: { alpha: 0, beta: 1, radius: 2, target: [0, 0, 0], fov: 0.8, mirrorX: false },
            provenance: {},
        };
        const state = {
            distances: new Float32Array(8),
            velocities: new Float32Array(24),
            staticDistances: new Float32Array(8).fill(100),
            local: [local],
        };
        updateComparisonSolids(input, state, 1);
        expect(state.distances[0]).toBeCloseTo(-0.1);
        expect(state.velocities[0]).toBeCloseTo(10);
        expect(state.velocities[1]).toBe(0);
        expect(state.velocities[2]).toBe(0);
    });
});
