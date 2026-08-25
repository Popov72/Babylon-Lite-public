import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

describe("fluid thickness debug view", () => {
    it("compresses additive HDR thickness instead of clipping values above one", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/fluid-surface-render.ts"), "utf8");
        expect(source).toContain("return value / (1.0 + value);");
        expect(source.match(/thicknessViz\(/g)).toHaveLength(3);
    });

    it("normalizes marker thickness by physical radius and sampling density", () => {
        const surface = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/fluid-surface-render.ts"), "utf8");
        const flip = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(surface).toContain("PARTICLE_THICKNESS_ALPHA * (radius / 0.09) * (currentSim.surfaceThicknessScale ?? 1)");
        expect(flip).toContain("surfaceThicknessScale: 8 / markersPerCell");
    });

    it("widens thickness splats while preserving their integrated contribution", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/fluid-surface-render.ts"), "utf8");
        expect(source).toContain("const PARTICLE_THICKNESS_SPLAT_SCALE = 1.5;");
        expect(source).toContain("cam.misc.x * thicknessSplatScale");
        expect(source).toContain("cam.misc.w * thickness / (thicknessSplatScale * thicknessSplatScale)");
        expect(source.match(/constants: \{ thicknessSplatScale: PARTICLE_THICKNESS_SPLAT_SCALE \}/g)).toHaveLength(2);
    });

    it("does not reconstruct a lone marker as a complete reflective surface disk", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/fluid-surface-render.ts"), "utf8");
        expect(source).toContain("let support = select(0.0, contribution, nearFrontSurface(realViewPos.z, i.ndc));");
        expect(source).toContain("let support = select(0.0, contribution, nearFrontSurface(hit.z, i.ndc));");
        expect(source).toContain('entryPoint: "fsDepthFiltered"');
        expect(source).toContain('entryPoint: "fsDepthFilteredAniso"');
        expect(source).toContain("surfaceSupport(i.ndc) <= oneMarker * 2.1");
        expect(source).toContain("const rejectSparseSurface = currentSim.surfaceRejectSparseMarkers === true && thickW === depthW && thickH === depthH;");
        expect(source).toContain("pass.setBindGroup(2, getSurfaceSupportBindGroup(views.thick!));");
    });
});
