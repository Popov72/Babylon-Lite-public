import { describe, expect, it, vi } from "vitest";
import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import {
    createViewVolumeGrid,
    getViewVolumeSliceBounds,
    viewDepthToVolumeSlice,
    volumeSliceToViewDepth,
    viewVolumeTextureBytes,
    type ViewVolumeDepthMapping,
} from "../../../packages/babylon-lite/src/render/volume/view-volume-grid";
import { buildViewVolumeDepthWgsl } from "../../../packages/babylon-lite/src/render/volume/view-volume-grid-wgsl";
import {
    alphaToOpticalDepth,
    composeOpticalTransfer,
    integrateHomogeneousMedium,
    transmittanceFromOpticalDepth,
    type OpticalTransfer,
} from "../../../packages/babylon-lite/src/render/volume/optical-transfer";
import { OPTICAL_TRANSFER_WGSL } from "../../../packages/babylon-lite/src/render/volume/optical-transfer-wgsl";
import { createVolumeTexture3D, disposeVolumeTexture3D } from "../../../packages/babylon-lite/src/render/volume/volume-texture";

const mappings: readonly ViewVolumeDepthMapping[] = [{ kind: "linear" }, { kind: "log" }, { kind: "power", exponent: 2 }];

describe("view-volume grid", () => {
    it("uses ceiling division for screen tiles", () => {
        const grid = createGrid({ kind: "linear" });
        expect(grid.width).toBe(241);
        expect(grid.height).toBe(136);
        expect(viewVolumeTextureBytes(grid, 8)).toBe(241 * 136 * 64 * 8);
    });

    it.each(mappings)("round-trips $kind depth mapping", (mapping) => {
        const grid = createGrid(mapping);
        for (const depth of [grid.nearDepth, 1, 10, 100, grid.farDepth]) {
            const slice = viewDepthToVolumeSlice(grid, depth);
            expect(volumeSliceToViewDepth(grid, slice)).toBeCloseTo(depth, 10);
        }
    });

    it.each(mappings)("clamps $kind view depths outside the grid", (mapping) => {
        const grid = createGrid(mapping);
        expect(viewDepthToVolumeSlice(grid, grid.nearDepth / 2)).toBe(0);
        expect(viewDepthToVolumeSlice(grid, grid.farDepth * 2)).toBe(grid.depth);
        expect(volumeSliceToViewDepth(grid, -1)).toBe(grid.nearDepth);
        expect(volumeSliceToViewDepth(grid, grid.depth + 1)).toBe(grid.farDepth);
    });

    it.each(mappings)("maps both endpoints of a narrow $kind range without an epsilon floor", (mapping) => {
        const grid = createViewVolumeGrid({ ...gridOptions(), nearDepth: 1e-6, farDepth: 1.5e-6, depthMapping: mapping });
        expect(viewDepthToVolumeSlice(grid, 1e-6)).toBe(0);
        expect(viewDepthToVolumeSlice(grid, 1.5e-6)).toBe(grid.depth);
        expect(volumeSliceToViewDepth(grid, grid.depth)).toBe(grid.farDepth);
    });

    it("uses thinner near slices for a power exponent above one", () => {
        const linear = getViewVolumeSliceBounds(createGrid({ kind: "linear" }), 0);
        const power = getViewVolumeSliceBounds(createGrid({ kind: "power", exponent: 2 }), 0);
        expect(power.thickness).toBeLessThan(linear.thickness);
    });

    it("emits specialized WGSL without a runtime mapping branch", () => {
        const log = buildViewVolumeDepthWgsl({ kind: "log" });
        const power = buildViewVolumeDepthWgsl({ kind: "power", exponent: 2 });
        expect(log).toContain("clamp(log(clamp(viewDepth,nearDepth,farDepth)/nearDepth)/log(farDepth/nearDepth),0.0,1.0)*sliceCount");
        expect(log).not.toContain("max(log(farDepth/nearDepth),1e-6)");
        expect(power).toContain("(viewDepth-nearDepth)/(farDepth-nearDepth)");
        expect(power).toContain("pow(normalizedDepth,0.5)");
        expect(power).not.toContain("mappingMode");
    });

    it("rejects invalid grid inputs", () => {
        expect(() => createViewVolumeGrid({ ...gridOptions(), nearDepth: 0 })).toThrow("nearDepth");
        expect(() => createViewVolumeGrid({ ...gridOptions(), farDepth: 0.05 })).toThrow("farDepth");
        expect(() => createViewVolumeGrid({ ...gridOptions(), depthMapping: { kind: "power", exponent: 0 } })).toThrow("depthMapping.exponent");
    });
});

describe("optical transfer", () => {
    it("converts alpha to optical depth and back", () => {
        const opticalDepth = alphaToOpticalDepth(0.5);
        expect(opticalDepth).toBeCloseTo(Math.log(2), 12);
        expect(transmittanceFromOpticalDepth(opticalDepth)).toBeCloseTo(0.5, 12);
    });

    it("composes transfers associatively", () => {
        const a: OpticalTransfer = { radiance: [1, 2, 3], transmittance: [0.8, 0.7, 0.6] };
        const b: OpticalTransfer = { radiance: [2, 1, 0.5], transmittance: [0.5, 0.4, 0.3] };
        const c: OpticalTransfer = { radiance: [0.25, 0.5, 1], transmittance: [0.9, 0.8, 0.7] };
        const left = composeOpticalTransfer(composeOpticalTransfer(a, b), c);
        const right = composeOpticalTransfer(a, composeOpticalTransfer(b, c));
        for (let i = 0; i < 3; i++) {
            expect(left.radiance[i]).toBeCloseTo(right.radiance[i]!, 12);
            expect(left.transmittance[i]).toBeCloseTo(right.transmittance[i]!, 12);
        }
    });

    it("uses the zero-extinction homogeneous-medium limit", () => {
        const transfer = integrateHomogeneousMedium([2, 3, 4], [0, 0, 0], 5);
        expect(transfer.radiance).toEqual([10, 15, 20]);
        expect(transfer.transmittance).toEqual([1, 1, 1]);
    });

    it("integrates low extinction over a long distance independently of subdivision", () => {
        const source: [number, number, number] = [1, 1, 1];
        const extinction: [number, number, number] = [1e-6, 1e-6, 1e-6];
        const whole = integrateHomogeneousMedium(source, extinction, 1e6);
        const half = integrateHomogeneousMedium(source, extinction, 5e5);
        const subdivided = composeOpticalTransfer(half, half);
        const expected = -Math.expm1(-1) / 1e-6;
        expect(whole.transmittance[0]).toBeCloseTo(Math.exp(-1), 12);
        expect(whole.radiance[0]).toBeCloseTo(expected, 6);
        expect(subdivided.radiance[0]).toBeCloseTo(expected, 6);
    });

    it("avoids cancellation for small positive optical depth", () => {
        const sigma = 1e-3;
        const distance = 1e-7;
        const expectedFactor = -Math.expm1(-sigma * distance) / (sigma * distance);
        const medium = integrateHomogeneousMedium([1, 1, 1], [sigma, sigma, sigma], distance);
        expect(medium.radiance[0] / distance).toBeCloseTo(expectedFactor, 12);
        expect(OPTICAL_TRANSFER_WGSL).toContain("let opticalDepth=sigma*travel;");
        expect(OPTICAL_TRANSFER_WGSL).toContain("let integralFactor=select(ratio,series,opticalDepth<vec3f(0.05));");
    });
});

describe("volume texture", () => {
    it("creates sampled and writable 3D storage", () => {
        const destroy = vi.fn();
        const createView = vi.fn(() => ({}) as GPUTextureView);
        const createTexture = vi.fn(
            () =>
                ({
                    createView,
                    destroy,
                }) as unknown as GPUTexture
        );
        const createSampler = vi.fn(() => ({}) as GPUSampler);
        const device = {
            limits: { maxTextureDimension3D: 256 },
            createTexture,
            createSampler,
        } as unknown as GPUDevice;
        const engine = { _device: device } as unknown as EngineContext;

        const volume = createVolumeTexture3D(engine, { label: "volume", width: 32, height: 16, depth: 64, format: "rgba16float" });
        expect(createTexture).toHaveBeenCalledWith(
            expect.objectContaining({
                dimension: "3d",
                format: "rgba16float",
                size: { width: 32, height: 16, depthOrArrayLayers: 64 },
            })
        );
        expect(createView).toHaveBeenCalledWith({ dimension: "3d" });
        disposeVolumeTexture3D(volume);
        disposeVolumeTexture3D(volume);
        expect(destroy).toHaveBeenCalledOnce();
    });
});

function gridOptions() {
    return {
        targetWidth: 1921,
        targetHeight: 1081,
        tileSize: 8,
        depthSlices: 64,
        nearDepth: 0.1,
        farDepth: 1000,
        depthMapping: { kind: "linear" } as ViewVolumeDepthMapping,
    };
}

function createGrid(mapping: ViewVolumeDepthMapping) {
    return createViewVolumeGrid({ ...gridOptions(), depthMapping: mapping });
}
