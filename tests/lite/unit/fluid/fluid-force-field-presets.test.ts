import { describe, expect, it } from "vitest";
import type { PairState } from "../../../../packages/babylon-lite/src/fluid/authoring/authoring-state";
import { exportJsonFromPairState, presetFromExportJson } from "../../../../packages/babylon-lite/src/fluid/authoring/preset-io";
import { importFluidPresetSession } from "../../../../packages/babylon-lite/src/fluid/authoring/preset-session";
import { carryMethodIndependentState } from "../../../../packages/babylon-lite/src/fluid/authoring/method-independent-state";
import { createDefaultFluidForceField } from "../../../../packages/babylon-lite/src/fluid/forces/force-field-config";

const state = (): PairState => ({
    schema: { gravity: 9.81 },
    demoParams: {},
    color: "#3399ff",
    half: true,
    thicknessDownscale: 1,
    absorption: 0.5,
    size: 1,
    physScale: 1,
    count: 128,
    grid: { position: [100, 100, 100], size: [4, 4, 4] },
});

describe("force-field preset persistence", () => {
    it("round-trips world-space fields, signed strengths and disabled definitions", () => {
        const fields = [
            { ...createDefaultFluidForceField("point", "point", [2, 3, 4]), strength: -12, future: { version: 2 } },
            { ...createDefaultFluidForceField("guide", "guide", [5, 6, 7]), enabled: false, flowStrength: -3, spinStrength: 5, endCaps: false },
        ];
        const original = { ...state(), forceFields: fields };
        const exported = exportJsonFromPairState("whiteboard", "PBF", original);
        const imported = presetFromExportJson(exported);
        expect(imported.forceFields).toEqual(fields);
        expect(imported.forceFields).not.toBe(fields);
        expect(carryMethodIndependentState(state(), original).forceFields).toEqual(fields);
    });

    it("clears old force fields when importing a legacy preset without the section", () => {
        const previous = { ...state(), forceFields: [createDefaultFluidForceField("point", "old")] };
        const legacy = exportJsonFromPairState("whiteboard", "PBF", state());
        expect(legacy.forceFields).toBeUndefined();
        expect(importFluidPresetSession(legacy, previous).state.forceFields).toEqual([]);
    });

    it("rejects unsupported field types and degenerate guides at the import boundary", () => {
        const preset = exportJsonFromPairState("whiteboard", "PBF", state());
        Object.assign(preset, { forceFields: [{ ...createDefaultFluidForceField("point", "bad"), type: "volume" }] });
        expect(() => presetFromExportJson(preset)).toThrow("unsupported force-field type");
        preset.forceFields = [{ ...createDefaultFluidForceField("guide", "bad"), start: [0, 0, 0], end: [0, 0, 0] }];
        expect(() => presetFromExportJson(preset)).toThrow("endpoints");
    });

    it("excludes enabled authored fields from Reference but preserves disabled configuration", () => {
        const reference: PairState = {
            ...state(),
            backendId: "flip-reference",
            schema: { gravity: 9.81, flipRatio: 0.95, minSubsteps: 2, maxSubsteps: 8, maxSubDtMs: 10, cflNumber: 5 },
            forceFields: [createDefaultFluidForceField("point", "point")],
        };
        expect(() => exportJsonFromPairState("whiteboard", "FLIP", reference)).toThrow("configured force fields");
        reference.forceFields = reference.forceFields!.map((field) => ({ ...field, enabled: false }));
        const preset = exportJsonFromPairState("whiteboard", "FLIP", reference);
        expect(presetFromExportJson(preset).forceFields).toEqual(reference.forceFields);
    });
});
