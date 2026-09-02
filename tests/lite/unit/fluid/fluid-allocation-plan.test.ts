import { describe, expect, it } from "vitest";

import {
    AQUANOVA_COMBINED_PARTICLE_CAPACITY,
    aquanovaCombinedParticleCapacity,
    DEFAULT_PARTICLE_BYTES_PER_SLOT,
    FLIP_PARTICLE_BYTES_PER_SLOT,
    MLS_MPM_PARTICLE_BYTES_PER_SLOT,
    PBF_PARTICLE_BYTES_PER_SLOT,
    PB_MPM_PARTICLE_BYTES_PER_SLOT,
    fluidAllocationCapabilities,
    fluidDeviceParticleCapacity,
    fluidFlipMacBufferBytes,
    fluidGridCellCountLimit,
    fluidMaximumPageCapacity,
    fluidParticleBufferLimitBytes,
    fluidParticleBytesPerSlot,
    mlsMpmDefaultPageCapacity,
    resolveFluidAllocationPlan,
    resolveFluidParticleCapacity,
} from "../../../../packages/babylon-lite/src/fluid/core/allocation-plan";
import { estimateFlipGpuBytes, flipMacFaceBufferBytes, pagedFlipStorageCounts } from "../../../../packages/babylon-lite/src/fluid/solvers/flip-sim";

describe("fluid allocation plan", () => {
    it.each([
        ["PBF", "fluid-positions"],
        ["FLIP", "flip-particle-positions"],
        ["MLS-MPM", "mpm-particles"],
        ["PB-MPM", "pbmpm-particles"],
    ] as const)("enumerates actual %s buffers", (method, representativeResource) => {
        const plan = resolveFluidAllocationPlan({ method, particleCount: 1_000, gridDim: [16, 12, 8] });

        expect(plan.steadyBytes).toBeGreaterThan(0);
        expect(plan.resources.some((resource) => resource.name === representativeResource)).toBe(true);
        expect(plan.resources.every((resource) => resource.bytes >= 0 && Number.isFinite(resource.bytes))).toBe(true);
        expect(plan.rebuildPeakBytes).toBe(plan.steadyBytes);
    });

    it("normalizes invalid finite-number inputs instead of producing invalid allocation sizes", () => {
        const plan = resolveFluidAllocationPlan({
            method: "FLIP",
            particleCount: Number.NaN,
            gridDim: [Number.POSITIVE_INFINITY, -3, 9.8],
            previousSteadyBytes: Number.NEGATIVE_INFINITY,
            maxParticleBudget: Number.NaN,
            pagedGrid: true,
            pagedGridMaxPages: Number.POSITIVE_INFINITY,
            flipWarmup: {
                initialLiveCount: Number.NaN,
                initialTargetCount: Number.POSITIVE_INFINITY,
            },
            quality: { polygonSurface: true, polygonReconstructionMultiplier: Number.NaN },
            foam: { enabled: true, poolScale: Number.NEGATIVE_INFINITY },
            limits: {
                maxBufferSize: Number.NaN,
                maxStorageBufferBindingSize: Number.POSITIVE_INFINITY,
                maxTextureDimension2D: -1,
            },
        });

        expect(plan.dimensions).toMatchObject({ particleCount: 1, gridDim: [4, 4, 9] });
        expect(plan.rebuildPeakBytes).toBe(plan.steadyBytes);
        expect(plan.resources.every((resource) => Number.isFinite(resource.bytes))).toBe(true);
        expect(plan.errors.some((error) => error.includes("must be finite"))).toBe(true);
        expect(plan.errors.some((error) => error.includes("normalized"))).toBe(true);
    });

    it("validates every buffer against its applicable per-buffer limits", () => {
        const storageTight = resolveFluidAllocationPlan({
            method: "PBF",
            particleCount: 100,
            gridDim: [4, 4, 4],
            limits: { maxStorageBufferBindingSize: 1_500, maxBufferSize: 5_000 },
        });
        expect(storageTight.errors.some((error) => error.includes("fluid-positions") && error.includes("maxStorageBufferBindingSize"))).toBe(true);

        const bufferTight = resolveFluidAllocationPlan({
            method: "PBF",
            particleCount: 100,
            gridDim: [4, 4, 4],
            limits: { maxStorageBufferBindingSize: 5_000, maxBufferSize: 1_500 },
        });
        expect(bufferTight.errors.some((error) => error.includes("fluid-positions") && error.includes("maxBufferSize"))).toBe(true);
    });

    it("does not compare aggregate steady bytes against maxBufferSize", () => {
        const plan = resolveFluidAllocationPlan({
            method: "PB-MPM",
            particleCount: 10,
            gridDim: [10, 10, 10],
            limits: { maxStorageBufferBindingSize: 20_000, maxBufferSize: 20_000 },
        });

        expect(plan.steadyBytes).toBeGreaterThan(20_000);
        expect(plan.resources.every((resource) => resource.bytes <= 20_000)).toBe(true);
        expect(plan.errors).toEqual([]);
    });

    it("adds the previous allocation into the transient rebuild peak", () => {
        const plan = resolveFluidAllocationPlan({
            method: "FLIP",
            particleCount: 50_000,
            gridDim: [64, 64, 64],
            previousSteadyBytes: 1_000_000,
        });
        expect(plan.rebuildPeakBytes).toBe(plan.steadyBytes + 1_000_000);
    });

    it("accounts for padded FLIP lookup rows, dispatch args, and authoritative page capacity", () => {
        const limits = { maxStorageBufferBindingSize: 1 << 26, maxBufferSize: 1 << 26, maxTextureDimension2D: 16 };
        const plan = resolveFluidAllocationPlan({
            method: "FLIP",
            particleCount: 100,
            gridDim: [33, 17, 9],
            pagedGrid: true,
            pagedGridMaxPages: 10,
            limits,
        });

        expect(plan.pages).toBeDefined();
        expect(plan.pages!.maximumPageCapacity).toBe(fluidMaximumPageCapacity("FLIP", [33, 17, 9], limits));
        expect(plan.pages).toMatchObject({
            lookupWidth: 16,
            lookupHeight: 3,
            lookupWords: 42,
            lookupPaddedWords: 48,
            pageDispatchBytes: 96,
        });
        expect(plan.pages!.pageDispatchBytes).toBe(96);
        expect(plan.resources.find((resource) => resource.name === "flip-page-lookup")!.bytes).toBe(plan.pages!.lookupPaddedWords * 4);
        expect(plan.resources.find((resource) => resource.name === "flip-page-data")!.bytes).toBe(plan.pages!.lookupWords * 4);
        expect(plan.resources.find((resource) => resource.name === "flip-page-dispatch")!.bytes).toBe(96);
        expect(plan.resources.find((resource) => resource.name === "flip-page-dispatch-config")!.bytes).toBe(32);
        expect(plan.resources.find((resource) => resource.name === "flip-page-status")!.bytes).toBe(16);
        expect(plan.resources.find((resource) => resource.name === "flip-page-status-readback-0")!.bytes).toBe(16);
        expect(plan.resources.find((resource) => resource.name === "flip-page-status-readback-1")!.bytes).toBe(16);
        expect(plan.resources.some((resource) => resource.name.startsWith("flip-warmup-seed-"))).toBe(false);
        expect(plan.steadyBytes).toBe(estimateFlipGpuBytes(100, [33, 17, 9], "jacobi", { pagedGrid: true, pagedGridMaxPages: 10 }, false, 16));
    });

    it("matches live FLIP bytes with warm-up seed buffers inactive and active", () => {
        const input = {
            method: "FLIP" as const,
            particleCount: 100,
            gridDim: [4, 5, 6] as const,
        };
        const inactive = resolveFluidAllocationPlan({
            ...input,
            flipWarmup: { initialLiveCount: 80, initialTargetCount: 80 },
        });
        const active = resolveFluidAllocationPlan({
            ...input,
            flipWarmup: { initialLiveCount: 20, initialTargetCount: 80 },
        });

        expect(inactive.flipWarmup).toEqual({
            initialLiveCount: 80,
            initialTargetCount: 80,
            seedBuffersAllocated: false,
        });
        expect(inactive.resources.some((resource) => resource.name.startsWith("flip-warmup-seed-"))).toBe(false);
        expect(inactive.steadyBytes).toBe(estimateFlipGpuBytes(100, [4, 5, 6]));

        expect(active.flipWarmup).toEqual({
            initialLiveCount: 20,
            initialTargetCount: 80,
            seedBuffersAllocated: true,
        });
        expect(active.resources.find((resource) => resource.name === "flip-warmup-seed-positions")!.bytes).toBe(100 * 16);
        expect(active.resources.find((resource) => resource.name === "flip-warmup-seed-velocities")!.bytes).toBe(100 * 16);
        expect(active.steadyBytes).toBe(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", {}, true));
    });

    it("clamps FLIP and MLS-MPM pages to one authoritative maximumPageCapacity", () => {
        const limits = { maxStorageBufferBindingSize: 128 * 1024, maxBufferSize: 128 * 1024, maxTextureDimension2D: 64 };
        for (const method of ["FLIP", "MLS-MPM"] as const) {
            const plan = resolveFluidAllocationPlan({
                method,
                particleCount: 100,
                gridDim: [64, 64, 64],
                pagedGrid: true,
                pagedGridMaxPages: 1_000_000,
                limits,
            });

            const maximum = fluidMaximumPageCapacity(method, [64, 64, 64], limits);
            expect(plan.pages!.maximumPageCapacity).toBe(maximum);
            expect(plan.pages!.maxPages).toBe(maximum);
            expect(plan.errors.some((error) => error.includes("maximumPageCapacity"))).toBe(true);
        }
    });

    it("includes MLS-MPM transactional working buffers in steady bytes", () => {
        const plan = resolveFluidAllocationPlan({ method: "MLS-MPM", particleCount: 250, gridDim: [8, 8, 8] });
        const bytes = (name: string): number => plan.resources.find((resource) => resource.name === name)!.bytes;

        expect(bytes("mpm-particles")).toBe(250 * 80);
        expect(bytes("mpm-particles-working")).toBe(bytes("mpm-particles"));
        expect(bytes("fluid-particle-lifecycle")).toBe((250 + 4) * 4);
        expect(bytes("mpm-lifecycle-working")).toBe(bytes("fluid-particle-lifecycle"));
        expect(bytes("mpm-flow-counters-working")).toBe(bytes("fluid-flow-counters"));
        expect(bytes("fluid-warmup-params")).toBe(16);
        expect(bytes("mpm-params")).toBe(256);
        expect(bytes("mpm-cell-count")).toBe(8 * 8 * 8 * 4);
        expect(bytes("mpm-accumulation-state")).toBe(16);
        expect(plan.steadyBytes).toBe(plan.resources.reduce((total, resource) => total + resource.bytes, 0));
    });

    it("validates each MLS-MPM working buffer against both per-buffer limits", () => {
        const storageTight = resolveFluidAllocationPlan({
            method: "MLS-MPM",
            particleCount: 250,
            gridDim: [8, 8, 8],
            limits: { maxStorageBufferBindingSize: 19_999, maxBufferSize: 100_000 },
        });
        expect(storageTight.errors).toContain("mpm-particles-working requires 20000 bytes in one storage binding, exceeding maxStorageBufferBindingSize 19999.");

        const bufferTight = resolveFluidAllocationPlan({
            method: "MLS-MPM",
            particleCount: 250,
            gridDim: [8, 8, 8],
            limits: { maxStorageBufferBindingSize: 100_000, maxBufferSize: 19_999 },
        });
        expect(bufferTight.errors).toContain("mpm-particles-working requires 20000 bytes, exceeding maxBufferSize 19999.");

        const allWorkingStorageTight = resolveFluidAllocationPlan({
            method: "MLS-MPM",
            particleCount: 250,
            gridDim: [8, 8, 8],
            limits: { maxStorageBufferBindingSize: 15, maxBufferSize: 100_000 },
        });
        for (const name of ["mpm-particles-working", "mpm-lifecycle-working", "mpm-flow-counters-working"]) {
            expect(allWorkingStorageTight.errors.some((error) => error.startsWith(`${name} requires`) && error.includes("maxStorageBufferBindingSize"))).toBe(true);
        }

        const allWorkingBufferTight = resolveFluidAllocationPlan({
            method: "MLS-MPM",
            particleCount: 250,
            gridDim: [8, 8, 8],
            limits: { maxStorageBufferBindingSize: 100_000, maxBufferSize: 15 },
        });
        for (const name of ["mpm-particles-working", "mpm-lifecycle-working", "mpm-flow-counters-working", "fluid-warmup-params"]) {
            expect(allWorkingBufferTight.errors.some((error) => error.startsWith(`${name} requires`) && error.includes("maxBufferSize"))).toBe(true);
        }
    });

    it("uses the live MLS-MPM cell-buffer label in dense and paged plans", () => {
        const dense = resolveFluidAllocationPlan({ method: "MLS-MPM", particleCount: 100, gridDim: [8, 8, 8] });
        const paged = resolveFluidAllocationPlan({
            method: "MLS-MPM",
            particleCount: 100,
            gridDim: [8, 8, 8],
            pagedGrid: true,
            pagedGridMaxPages: 4,
        });

        expect(dense.resources.some((resource) => resource.name === "mpm-cells")).toBe(true);
        expect(paged.resources.some((resource) => resource.name === "mpm-paged-cells")).toBe(true);
        expect(paged.resources.some((resource) => resource.name === "mpm-cells")).toBe(false);
    });

    it("derives active-foam capacity from the pool and active-state binding limits", () => {
        const limit = 64 * 1024;
        const plan = resolveFluidAllocationPlan({
            method: "PBF",
            particleCount: 10_000,
            gridDim: [8, 8, 8],
            foam: { enabled: true, activeParticles: true, poolScale: 10 },
            limits: { maxStorageBufferBindingSize: limit, maxBufferSize: limit },
        });

        expect(plan.foamCapacity).toBeGreaterThan(0);
        expect(plan.resources.find((resource) => resource.name === "pbf-foam-pool")!.bytes).toBe(plan.foamCapacity * 32);
        expect(plan.resources.find((resource) => resource.name === "pbf-foam-active-state")!.bytes).toBeLessThanOrEqual(limit);
        expect(plan.resources.some((resource) => resource.name === "fluid-foam-sorted-normals")).toBe(true);
    });

    it("reports the exact even polygon triangle capacity and related buffers", () => {
        const plan = resolveFluidAllocationPlan({
            method: "FLIP",
            particleCount: 10,
            gridDim: [4, 4, 4],
            quality: { polygonSurface: true },
            surfaceMaxTriangles: 99,
            limits: { maxStorageBufferBindingSize: 10_000, maxBufferSize: 10_000 },
        });

        expect(plan.polygonTriangleCapacity).toBe(98);
        expect(plan.resources.find((resource) => resource.name === "flip-polygon-surface-indices")!.bytes).toBe(98 * 12);
        expect(plan.resources.find((resource) => resource.name === "flip-polygon-surface-wireframe-indices")!.bytes).toBe(98 * 16);
    });

    it("flags particle budgets and describes every backend's allocation capabilities", () => {
        const plan = resolveFluidAllocationPlan({ method: "FLIP", particleCount: 100_000, gridDim: [64, 64, 64], maxParticleBudget: 50_000 });
        expect(plan.errors.some((error) => error.includes("exceeds budget"))).toBe(true);
        expect(fluidAllocationCapabilities("FLIP")).toMatchObject({
            supportsPagedGrid: true,
            supportsStaticByteEstimate: true,
            supportsPolygonSurface: true,
        });
        expect(fluidAllocationCapabilities("PBF")).toMatchObject({ supportsPagedGrid: false, supportsStaticByteEstimate: true });
        expect(fluidAllocationCapabilities("MLS-MPM")).toMatchObject({ supportsPagedGrid: true, supportsStaticByteEstimate: true });
        expect(fluidAllocationCapabilities("PB-MPM")).toMatchObject({ supportsPagedGrid: false, supportsStaticByteEstimate: true });
    });

    it("centralizes particle, grid, MAC, and MLS default sizing", () => {
        expect(fluidParticleBytesPerSlot("FLIP")).toBe(FLIP_PARTICLE_BYTES_PER_SLOT);
        expect(fluidParticleBytesPerSlot("PBF")).toBe(PBF_PARTICLE_BYTES_PER_SLOT);
        expect(fluidParticleBytesPerSlot("MLS-MPM")).toBe(MLS_MPM_PARTICLE_BYTES_PER_SLOT);
        expect(fluidParticleBytesPerSlot("PB-MPM")).toBe(PB_MPM_PARTICLE_BYTES_PER_SLOT);
        expect(fluidParticleBytesPerSlot("unknown")).toBe(DEFAULT_PARTICLE_BYTES_PER_SLOT);
        const limits = { maxStorageBufferBindingSize: 134_217_728, maxBufferSize: 268_435_456 };
        expect(fluidParticleBufferLimitBytes(limits)).toBe(134_217_728);
        expect(fluidDeviceParticleCapacity(limits, "FLIP")).toBe(Math.floor(134_217_728 / 16));
        expect(fluidGridCellCountLimit(limits)).toBe(Math.floor(134_217_728 / 16));

        const cells: [number, number, number] = [96, 96, 96];
        expect(fluidFlipMacBufferBytes(cells)).toBe(flipMacFaceBufferBytes(cells));
        expect(fluidFlipMacBufferBytes(cells, { enabled: true, maxPages: 4_000 })).toBe(pagedFlipStorageCounts(4_000).faces * 8);
        expect(mlsMpmDefaultPageCapacity(80_000)).toBe(51_000);
        expect(mlsMpmDefaultPageCapacity(100)).toBe(1_000);
    });

    it("uses the largest mandatory per-particle binding rather than aggregate bytes", () => {
        const limits = { maxStorageBufferBindingSize: 160, maxBufferSize: 160 };

        expect(fluidDeviceParticleCapacity(limits, "PBF")).toBe(10);
        expect(fluidDeviceParticleCapacity(limits, "FLIP")).toBe(10);
        expect(fluidDeviceParticleCapacity(limits, "MLS-MPM")).toBe(2);
        expect(fluidDeviceParticleCapacity(limits, "PB-MPM")).toBe(1);
        expect(fluidDeviceParticleCapacity({ maxStorageBufferBindingSize: 160, maxBufferSize: 159 }, "PBF")).toBe(9);
    });

    it("uses the complete contextual plan for optional and static resource limits", () => {
        const limits = { maxStorageBufferBindingSize: 100_000, maxBufferSize: 100_000, maxTextureDimension2D: 1024 };
        const pbf = resolveFluidParticleCapacity({
            method: "PBF",
            gridDim: [4, 4, 4],
            limits,
        });

        expect(pbf.capacity).toBe(6_250);
        expect(pbf.bytesPerParticleResource).toBe(16);
        expect(pbf.plan.resources.find((resource) => resource.name === "fluid-positions")?.bytes).toBe(100_000);
        expect(pbf.plan.steadyBytes).toBeGreaterThan(limits.maxBufferSize);
        expect(pbf.plan.resources.every((resource) => resource.bytes <= limits.maxBufferSize)).toBe(true);

        const polygonLimited = resolveFluidParticleCapacity({
            method: "FLIP",
            gridDim: [10, 10, 10],
            quality: {
                polygonSurface: true,
                polygonReconstructionMultiplier: 2,
            },
            limits,
        });
        expect(fluidDeviceParticleCapacity(limits, "FLIP")).toBe(6_250);
        expect(polygonLimited.capacity).toBe(0);
        expect(polygonLimited.plan.resources.some((resource) => resource.name === "flip-polygon-surface-vertices" && resource.bytes > limits.maxBufferSize)).toBe(true);
    });
});

describe("Aquanova combined render-stream capacity", () => {
    const ampleLimits = { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 31 };

    it("uses one 600k authoring/production authority and clamps only to device limits", () => {
        expect(AQUANOVA_COMBINED_PARTICLE_CAPACITY).toBe(600_000);
        expect(aquanovaCombinedParticleCapacity(ampleLimits)).toBe(600_000);
        expect(aquanovaCombinedParticleCapacity({ maxStorageBufferBindingSize: 480_000 * 16, maxBufferSize: 2 ** 31 })).toBe(480_000);
    });
});
