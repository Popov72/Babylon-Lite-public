import { describe, expect, it } from "vitest";

import { resolveFluidControlsCapabilities } from "../../../../packages/babylon-lite/src/fluid/controls/controls-capabilities";
import { DEFAULT_FLUID_SCHEMAS } from "../../../../packages/babylon-lite/src/fluid/controls/controls-panel";

describe("fluid controls capabilities contract", () => {
    it("gates FLIP-only control groups to FLIP", () => {
        const caps = resolveFluidControlsCapabilities({ method: "FLIP" });
        expect(caps).toMatchObject({ flipTuning: true, polygonSurface: true, pressureDiagnostics: true, pagedGrid: true, material: false, activeBlocks: false });
    });

    it("gates active blocks + paging to MLS-MPM", () => {
        const caps = resolveFluidControlsCapabilities({ method: "MLS-MPM" });
        expect(caps).toMatchObject({ activeBlocks: true, pagedGrid: true, flipTuning: false, polygonSurface: false, material: false, pressureDiagnostics: false });
    });

    it("gates the material selector to PB-MPM", () => {
        expect(resolveFluidControlsCapabilities({ method: "PB-MPM" })).toMatchObject({ material: true, pagedGrid: false, activeBlocks: false, flipTuning: false });
    });

    it("exposes no method-specific groups for PBF", () => {
        expect(resolveFluidControlsCapabilities({ method: "PBF" })).toMatchObject({
            material: false,
            activeBlocks: false,
            flipTuning: false,
            pagedGrid: false,
            polygonSurface: false,
            pressureDiagnostics: false,
        });
    });

    it("keeps method-independent groups on for every backend", () => {
        for (const method of ["PBF", "FLIP", "MLS-MPM", "PB-MPM"]) {
            expect(resolveFluidControlsCapabilities({ method })).toMatchObject({ independentRendering: true, gridVisuals: true, authoringSurfaces: true });
        }
    });

    it("allows PB-MPM liquid viscosity up to 1.0", () => {
        expect(DEFAULT_FLUID_SCHEMAS["PB-MPM"]!.find((entry) => entry.key === "liquidViscosity")?.max).toBe(1);
    });

    it("gates timing on device timestamp-query support with an optimistic default", () => {
        expect(resolveFluidControlsCapabilities({ method: "FLIP" }).timing).toBe(true);
        expect(resolveFluidControlsCapabilities({ method: "FLIP", timestampQuerySupported: false }).timing).toBe(false);
        expect(resolveFluidControlsCapabilities({ method: "PBF", timestampQuerySupported: true }).timing).toBe(true);
    });

    it("intersects backend support with explicit host capabilities", () => {
        expect(
            resolveFluidControlsCapabilities({
                method: "FLIP",
                hostCapabilities: {
                    independentRendering: false,
                    polygonSurface: false,
                    gridVisuals: false,
                },
            })
        ).toMatchObject({
            independentRendering: false,
            polygonSurface: false,
            gridVisuals: false,
            flipTuning: true,
            pagedGrid: true,
        });
    });

    it("cannot use host capabilities to enable an unsupported backend feature", () => {
        expect(
            resolveFluidControlsCapabilities({
                method: "PBF",
                hostCapabilities: {
                    polygonSurface: true,
                    pressureDiagnostics: true,
                    activeBlocks: true,
                },
            })
        ).toMatchObject({
            polygonSurface: false,
            pressureDiagnostics: false,
            activeBlocks: false,
        });
    });
});
