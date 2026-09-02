import { describe, expect, it } from "vitest";

import { FLUID_MESH_MAX_OPENNESS, FluidMeshSamplingError, sampleFluidMeshParticles, transferFluidMeshParticleUvs } from "../../../../packages/babylon-lite/src/index";

interface TestMesh {
    positions: Float32Array;
    indices: Uint32Array;
}

function makeBox(x: number, y: number, z: number): TestMesh {
    return {
        positions: new Float32Array([0, 0, 0, x, 0, 0, x, y, 0, 0, y, 0, 0, 0, z, x, 0, z, x, y, z, 0, y, z]),
        indices: new Uint32Array([1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3, 3, 7, 6, 3, 6, 2, 0, 1, 5, 0, 5, 4, 4, 5, 6, 4, 6, 7, 0, 3, 2, 0, 2, 1]),
    };
}

function expectResultBuffersToMatchCount(result: ReturnType<typeof sampleFluidMeshParticles>): void {
    expect(result.positions.length).toBe(result.count * 3);
    expect(result.uvs === null || result.uvs.length === result.count * 2).toBe(true);
    expect(result.texIndices === null || result.texIndices.length === result.count).toBe(true);
    expect(result.radius).toBeGreaterThan(0);
    for (let i = 0; i < result.positions.length; i += 3) {
        expect(result.positions[i]).toBeGreaterThanOrEqual(result.bounds.min[0] - 1e-6);
        expect(result.positions[i]).toBeLessThanOrEqual(result.bounds.max[0] + 1e-6);
        expect(result.positions[i + 1]).toBeGreaterThanOrEqual(result.bounds.min[1] - 1e-6);
        expect(result.positions[i + 1]).toBeLessThanOrEqual(result.bounds.max[1] + 1e-6);
        expect(result.positions[i + 2]).toBeGreaterThanOrEqual(result.bounds.min[2] - 1e-6);
        expect(result.positions[i + 2]).toBeLessThanOrEqual(result.bounds.max[2] + 1e-6);
    }
}

describe("fluid mesh particle sampling policy", () => {
    it("volume-samples a watertight mesh with count, radius, bounds, and metrics kept consistent", () => {
        const mesh = makeBox(1, 1, 1);
        const automatic = sampleFluidMeshParticles({ ...mesh, radius: 0.1, mode: "dense" });
        const forced = sampleFluidMeshParticles({ ...mesh, radius: 0.1, mode: "dense", strategy: "volume" });

        expect(automatic.strategy).toBe("volume");
        expect(automatic.shell).toBe(false);
        expect(automatic.selectionReason).toBe("volume-sampled");
        expect(automatic.metrics.openness).toBeCloseTo(0);
        expect(automatic.metrics.thickness).toBeGreaterThanOrEqual(1);
        expect(automatic.warnings).toEqual([]);
        expect(automatic.count).toBe(forced.count);
        expect(automatic.radius).toBe(forced.radius);
        expect(automatic.radius).toBeCloseTo(0.1);
        expect(forced.selectionReason).toBe("forced-volume");
        expectResultBuffersToMatchCount(automatic);
    });

    it("surface-samples an open mesh using vector-area openness and barycentric UVs", () => {
        const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
        const indices = new Uint32Array([0, 1, 2]);
        const uvs = new Float32Array([0, 0, 1, 0, 0, 1]);
        const result = sampleFluidMeshParticles({ positions, indices, uvs, radius: 0.1, mode: "dense" });

        expect(result.strategy).toBe("surface");
        expect(result.shell).toBe(true);
        expect(result.selectionReason).toBe("open-mesh");
        expect(result.metrics.openness).toBeGreaterThan(FLUID_MESH_MAX_OPENNESS);
        expect(result.metrics.area).toBeCloseTo(0.5);
        expect(result.count).toBe(64);
        expect(result.radius).toBeCloseTo(Math.sqrt(result.metrics.area / result.count));
        expect(result.uvs).not.toBeNull();
        for (let i = 0; i < result.count; i++) {
            expect(result.uvs![i * 2]).toBeCloseTo(result.positions[i * 3]!, 6);
            expect(result.uvs![i * 2 + 1]).toBeCloseTo(result.positions[i * 3 + 1]!, 6);
        }
        expectResultBuffersToMatchCount(result);
    });

    it("uses the unchanged thickness formula to select a shell for a closed thin mesh", () => {
        const mesh = makeBox(1, 0.15, 1);
        const result = sampleFluidMeshParticles({ ...mesh, radius: 0.1, mode: "dense" });

        expect(result.metrics.openness).toBeCloseTo(0);
        expect(result.metrics.thickness).not.toBeNull();
        expect(result.metrics.thickness).toBeCloseTo((8 * 27 * 0.1 * 0.1) / 2.6, 5);
        expect(result.metrics.thickness).toBeLessThan(1);
        expect(result.strategy).toBe("surface");
        expect(result.selectionReason).toBe("thin-mesh");
        expect(result.warnings).toEqual([]);
    });

    it("honours forced surface and automatic surface-only strategies", () => {
        const mesh = makeBox(1, 1, 1);
        const forced = sampleFluidMeshParticles({ ...mesh, radius: 0.1, mode: "dense", strategy: "surface" });
        const hinted = sampleFluidMeshParticles({ ...mesh, radius: 0.1, mode: "dense", surfaceOnly: true });

        expect(forced.strategy).toBe("surface");
        expect(forced.selectionReason).toBe("forced-surface");
        expect(forced.radius).toBeCloseTo(Math.sqrt(forced.metrics.area / forced.count));
        expect(hinted.strategy).toBe("surface");
        expect(hinted.selectionReason).toBe("surface-only");
    });

    it("documents automatic fallback when the volume sampler produces no particles", () => {
        const mesh = makeBox(1, 0.05, 1);
        const result = sampleFluidMeshParticles({ ...mesh, radius: 0.1, mode: "dense" });

        expect(result.strategy).toBe("surface");
        expect(result.selectionReason).toBe("volume-fallback");
        expect(result.warnings).toEqual([
            expect.objectContaining({
                code: "AUTO_VOLUME_EMPTY",
            }),
        ]);
    });

    it("does not convert an explicitly forced empty volume sample to a shell", () => {
        const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
        const indices = new Uint32Array([0, 1, 2]);

        expect(() => sampleFluidMeshParticles({ positions, indices, radius: 0.1, mode: "dense", strategy: "volume" })).toThrowError(
            expect.objectContaining<Partial<FluidMeshSamplingError>>({
                code: "VOLUME_SAMPLER_EMPTY",
                requestedStrategy: "volume",
            })
        );
    });

    it("reports degenerate and malformed meshes as structured errors before sampling", () => {
        const degenerate = {
            positions: new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]),
            indices: new Uint32Array([0, 1, 2]),
        };
        const invalidIndices = {
            positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
            indices: new Uint32Array([0, 1, 9]),
        };

        expect(() => sampleFluidMeshParticles({ ...degenerate, radius: 0.1, mode: "dense" })).toThrowError(
            expect.objectContaining<Partial<FluidMeshSamplingError>>({ code: "DEGENERATE_MESH" })
        );
        expect(() => sampleFluidMeshParticles({ ...invalidIndices, radius: 0.1, mode: "dense", strategy: "volume" })).toThrowError(
            expect.objectContaining<Partial<FluidMeshSamplingError>>({
                code: "INVALID_INDICES",
                requestedStrategy: "volume",
            })
        );
    });
});

describe("closest-surface fluid particle UV transfer", () => {
    it("interpolates the closest triangle barycentrically and carries its texture index", () => {
        const result = transferFluidMeshParticleUvs({
            particles: new Float32Array([0.25, 0.25, 1, 2, 0, 0]),
            count: 2,
            positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
            indices: new Uint32Array([0, 1, 2]),
            uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
            texIndices: new Uint32Array([7, 7, 7]),
        });

        expect([...result.uvs]).toEqual([0.25, 0.25, 1, 0]);
        expect([...result.texIndices]).toEqual([7, 7]);
    });
});
