import { describe, expect, it } from "vitest";

import type { PairState } from "../../../../packages/babylon-lite/src/fluid/authoring/authoring-state";
import { exportJsonFromPairState, type FluidExportJson } from "../../../../packages/babylon-lite/src/fluid/authoring/preset-io";
import { editFluidPresetSession, exportFluidPresetSession, importFluidPresetSession } from "../../../../packages/babylon-lite/src/fluid/authoring/preset-session";

function pairState(): PairState {
    return {
        schema: { gravity: 9.8 },
        demoParams: {},
        simulationDuration: 12,
        alphaDecay: 3.5,
        simulationTimeScale: 0.75,
        color: "#16a3c3",
        half: false,
        thicknessDownscale: 1,
        absorption: 1,
        size: 1,
        physScale: 1,
        count: 15_000,
        independentRendering: true,
        envIntensity: 0.6,
        msaa: true,
        foam: {
            enabled: true,
            activeParticles: false,
            kTa: 1,
            kWc: 2,
            kb: 3,
            kd: 0.5,
            tMin: 0.5,
            tMax: 4,
            poolScale: 2,
            size: 1,
            blurRadius: 2,
            lightIntensity: 1,
            ambient: 0.2,
            aoStrength: 0.3,
            normalStrength: 4,
            debugTexture: "off",
        },
    };
}

describe("fluid preset session", () => {
    it("retains imported state and recursively merges cross-host application edits", () => {
        const json = exportJsonFromPairState("fluid", "PBF", pairState()) as FluidExportJson & Record<string, unknown>;
        json.futureRoot = { nested: { retained: true } };
        (json.render as unknown as Record<string, unknown>).futureSurface = { spectral: { wavelength: 512 } };
        (json.foam as unknown as Record<string, unknown>).futureClassifier = { threshold: 0.7 };
        json.source = {
            application: "fluid-author",
            version: "20",
            settings: { pipeline: { retained: "yes", futureFlag: true } },
        };
        json.impulse = {
            intensity: 4,
            direction: [0, 1, 0],
            radius: 2,
        };
        (json.impulse as unknown as Record<string, unknown>).response = { curve: "smooth", nested: { retained: true } };
        json.grid = { x: 20, y: 10, z: 30, position: [1, 2, 3] };
        (json.grid as unknown as Record<string, unknown>).futureBounds = { padding: [1, 2, 3] };
        json.scene = {
            encoding: "base64",
            glb: "AAAA",
            collision: "BBBB",
            anchorPosition: [1, 2, 3],
        };
        (json.scene as unknown as Record<string, unknown>).futureCollision = {
            codec: "zstd",
            options: { level: 7 },
        };

        const defaults = pairState();
        defaults.simulationDuration = 0;
        defaults.alphaDecay = 2;
        defaults.simulationTimeScale = 1;
        defaults.independentRendering = false;
        defaults.envIntensity = 1;
        defaults.msaa = false;

        const imported = importFluidPresetSession(json, defaults);
        const edited = editFluidPresetSession(imported, {
            state: {
                color: "#0044aa",
                simulationDuration: 18,
                simulationTimeScale: undefined,
                foam: { size: 1.5 },
            },
            application: {
                source: { settings: { pipeline: { editedBy: "aquanova" } } },
                impulse: { intensity: 9 },
                grid: { x: 24 },
                scene: { anchorPosition: [4, 5, 6] },
            },
        });
        const exported = exportFluidPresetSession(edited, {
            demo: "aquanova",
            method: "PBF",
            state: { absorption: 1.25 },
        }) as FluidExportJson & Record<string, unknown>;

        expect(imported.state.simulationDuration).toBe(12);
        expect((imported.application.impulse as Record<string, unknown>).intensity).toBe(4);

        expect(exported.meta).toEqual({ demo: "aquanova", method: "PBF" });
        expect(exported.simulationDuration).toBe(18);
        expect(exported.alphaDecay).toBe(3.5);
        expect(exported.simulationTimeScale).toBe(0.75);
        expect(exported.render.independentRendering).toBe(true);
        expect(exported.envIntensity).toBe(0.6);
        expect(exported.msaa).toBe(true);
        expect(exported.render.waterColor).toBe("#0044aa");
        expect(exported.render.absorption).toBe(1.25);
        expect(exported.foam.activeParticles).toBe(false);
        expect(exported.foam.foamSize).toBe(1.5);

        expect(exported.futureRoot).toEqual({ nested: { retained: true } });
        expect((exported.render as unknown as Record<string, unknown>).futureSurface).toEqual({ spectral: { wavelength: 512 } });
        expect((exported.foam as unknown as Record<string, unknown>).futureClassifier).toEqual({ threshold: 0.7 });
        expect(exported.source).toEqual({
            application: "fluid-author",
            version: "20",
            settings: {
                pipeline: {
                    retained: "yes",
                    futureFlag: true,
                    editedBy: "aquanova",
                },
            },
        });
        expect(exported.impulse).toEqual({
            intensity: 9,
            direction: [0, 1, 0],
            radius: 2,
            response: { curve: "smooth", nested: { retained: true } },
        });
        expect(exported.grid).toEqual({
            x: 24,
            y: 10,
            z: 30,
            position: [1, 2, 3],
            futureBounds: { padding: [1, 2, 3] },
        });
        expect(exported.scene).toEqual({
            encoding: "base64",
            glb: "AAAA",
            collision: "BBBB",
            anchorPosition: [4, 5, 6],
            futureCollision: { codec: "zstd", options: { level: 7 } },
        });
    });
});
