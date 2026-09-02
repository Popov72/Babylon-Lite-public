import { describe, expect, it } from "vitest";

import type { PairState } from "../../../../packages/babylon-lite/src/fluid/authoring/authoring-state";
import { exportJsonFromPairState, presetFromExportJson, type FluidExportJson } from "../../../../packages/babylon-lite/src/fluid/authoring/preset-io";

const basePairState = (): PairState => ({
    schema: { gravity: 9.8 },
    demoParams: {},
    color: "#16a3c3",
    half: false,
    thicknessDownscale: 1,
    absorption: 1,
    size: 1,
    physScale: 1,
    count: 15_000,
});

describe("lossless fluid preset round-trip", () => {
    it("preserves unknown/forward-compatible top-level fields through import and re-export", () => {
        const json = exportJsonFromPairState("box", "PBF", basePairState());
        // Simulate a newer format revision this build does not model.
        const injected = json as unknown as Record<string, unknown>;
        injected.futureKnob = 42;
        injected.experimentalSection = { alpha: 1, beta: [1, 2, 3] };

        const preset = presetFromExportJson(json);
        expect(preset.forwardCompatibleFields).toEqual({ futureKnob: 42, experimentalSection: { alpha: 1, beta: [1, 2, 3] } });

        const reexported = exportJsonFromPairState("box", "PBF", { ...basePairState(), forwardCompatibleFields: preset.forwardCompatibleFields }) as unknown as Record<
            string,
            unknown
        >;
        expect(reexported.futureKnob).toBe(42);
        expect(reexported.experimentalSection).toEqual({ alpha: 1, beta: [1, 2, 3] });
        // Known fields must still win over any forward-compatible collision.
        expect(reexported.formatVersion).toBe(14);
        expect(reexported.particleCount).toBe(15_000);
    });

    it("preserves unknown fields recursively inside known sections", () => {
        const json = exportJsonFromPairState("box", "PBF", basePairState());
        const record = json as unknown as Record<string, unknown>;
        (record.meta as Record<string, unknown>).pipelineRevision = "next";
        (record.render as Record<string, unknown>).futureSurface = {
            mode: "spectral",
            tuning: { wavelength: 512, weights: [0.2, 0.8] },
        };
        (record.foam as Record<string, unknown>).futureClassifier = { threshold: 0.75 };
        record.camera = { alpha: 1, beta: 2, radius: 3, futureLens: { aperture: 1.4 } };

        const preset = presetFromExportJson(json);
        expect(preset.forwardCompatibleFields).toEqual({
            meta: { pipelineRevision: "next" },
            render: { futureSurface: { mode: "spectral", tuning: { wavelength: 512, weights: [0.2, 0.8] } } },
            foam: { futureClassifier: { threshold: 0.75 } },
            camera: { futureLens: { aperture: 1.4 } },
        });

        const reexported = exportJsonFromPairState("box", "PBF", {
            ...basePairState(),
            ...preset,
            schema: preset.schema!,
            demoParams: preset.demoParams!,
        }) as unknown as Record<string, unknown>;
        expect((reexported.meta as Record<string, unknown>).pipelineRevision).toBe("next");
        expect((reexported.render as Record<string, unknown>).futureSurface).toEqual({
            mode: "spectral",
            tuning: { wavelength: 512, weights: [0.2, 0.8] },
        });
        expect((reexported.foam as Record<string, unknown>).futureClassifier).toEqual({ threshold: 0.75 });
        expect((reexported.camera as Record<string, unknown>).futureLens).toEqual({ aperture: 1.4 });
    });

    it("round-trips an explicitly disabled active-particle foam path", () => {
        const state = basePairState();
        state.foam = {
            enabled: true,
            activeParticles: false,
            kTa: 0,
            kWc: 0,
            kb: 0,
            kd: 0,
            tMin: 0,
            tMax: 1,
            poolScale: 1,
            blurRadius: 0,
            lightIntensity: 0,
            ambient: 0,
            aoStrength: 0,
            normalStrength: 0,
            debugTexture: "off",
        };

        const exported = exportJsonFromPairState("fluid", "PBF", state);
        const imported = presetFromExportJson(exported);

        expect(exported.foam.activeParticles).toBe(false);
        expect(imported.foam?.activeParticles).toBe(false);
    });

    it("supports a lossless Fluid to Aquanova to export round trip for shared host-owned sections", () => {
        const fluidJson = exportJsonFromPairState("fluid", "PBF", basePairState()) as unknown as Record<string, unknown>;
        (fluidJson.foam as Record<string, unknown>).activeParticles = false;
        fluidJson.source = {
            application: "future-fluid-author",
            version: "20",
            settings: { nested: { extension: true } },
        };
        fluidJson.impulse = { intensity: 8, direction: [0, 1, 0], futureFalloff: { curve: "smooth" } };
        fluidJson.scene = {
            encoding: "base64",
            glb: "AAAA",
            collision: "BBBB",
            futureCollision: { compression: "zstd" },
        };

        const imported = presetFromExportJson(fluidJson as unknown as FluidExportJson);
        const aquanovaState: PairState = {
            ...basePairState(),
            ...imported,
            schema: imported.schema!,
            demoParams: imported.demoParams!,
        };
        const exported = exportJsonFromPairState("aquanova", "PBF", aquanovaState) as unknown as Record<string, unknown>;

        expect(exported.source).toEqual(fluidJson.source);
        expect(exported.impulse).toEqual(fluidJson.impulse);
        expect(exported.scene).toEqual(fluidJson.scene);
        expect(exported.meta).toEqual({ demo: "aquanova", method: "PBF" });
        expect((exported.foam as Record<string, unknown>).activeParticles).toBe(false);
    });

    it("does not fabricate a forward-compatible bag for a fully-understood file", () => {
        const json = exportJsonFromPairState("box", "PBF", basePairState());
        const preset = presetFromExportJson(json);
        expect(preset.forwardCompatibleFields).toBeUndefined();
    });

    it("never captures known schema fields as forward-compatible", () => {
        const json = exportJsonFromPairState("box", "FLIP", basePairState());
        const preset = presetFromExportJson(json as FluidExportJson);
        expect(preset.forwardCompatibleFields).toBeUndefined();
    });
});
