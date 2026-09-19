import { describe, expect, it, vi } from "vitest";
import { createDefaultFluidForceField, evaluateFluidForceFields, validateFluidForceFields } from "../../../../packages/babylon-lite/src/fluid/forces/force-field-config";
import { createFluidForcePasses, updateFluidForcePasses, type FluidForcePasses } from "../../../../packages/babylon-lite/src/fluid/core/force-field-passes";
import type { ForceFieldSpec } from "../../../../packages/babylon-lite/src/fluid/core/sim-common";

describe("authored fluid force fields", () => {
    const point = () => ({ ...createDefaultFluidForceField("point", "point", [0, 0, 0]), strength: 10, useMinDistance: false, useMaxDistance: false });

    it("uses signed radial acceleration with native-style falloff and force limits", () => {
        const fields = validateFluidForceFields([{ ...point(), falloffPower: 2 }]);
        expect(evaluateFluidForceFields(fields, [2, 0, 0])).toEqual([2.5, 0, 0]);
        expect(evaluateFluidForceFields(fields, [0.1, 0, 0])[0]).toBeCloseTo(30);
        expect(evaluateFluidForceFields(validateFluidForceFields([{ ...point(), strength: -10, falloffPower: 0 }]), [2, 0, 0])).toEqual([-10, 0, 0]);
    });

    it("preserves minimum-distance plateaus, maximum-distance cutoffs and center symmetry", () => {
        const fields = validateFluidForceFields([{ ...point(), useMinDistance: true, minDistance: 2, useMaxDistance: true, maxDistance: 4, falloffPower: 1 }]);
        expect(evaluateFluidForceFields(fields, [1, 0, 0])).toEqual([5, 0, 0]);
        expect(evaluateFluidForceFields(fields, [4, 0, 0])).toEqual([2.5, 0, 0]);
        expect(evaluateFluidForceFields(fields, [4.01, 0, 0])).toEqual([0, 0, 0]);
        expect(evaluateFluidForceFields(fields, [0, 0, 0])).toEqual([0, 0, 0]);
    });

    it("combines guide attraction, flow and right-hand-rule spin", () => {
        const guide = { ...createDefaultFluidForceField("guide", "guide", [0, 0, 0]), strength: -2, flowStrength: 4, spinStrength: 6, falloffPower: 0 };
        const fields = validateFluidForceFields([guide]);
        expect(evaluateFluidForceFields(fields, [1, 0, 0])).toEqual([-2, 4, -6]);
        expect(evaluateFluidForceFields(validateFluidForceFields([{ ...guide, endCaps: false }]), [1, 2, 0])).toEqual([0, 0, 0]);
        expect(evaluateFluidForceFields(fields, [1, 2, 0])[2]).toBeCloseTo(-6 / Math.sqrt(2));
    });

    it("sums multiple fields and makes disabled fields inert", () => {
        const fields = validateFluidForceFields([
            { ...point(), falloffPower: 0 },
            { ...point(), id: "other", strength: -3, falloffPower: 0 },
        ]);
        expect(evaluateFluidForceFields(fields, [1, 0, 0])).toEqual([7, 0, 0]);
        expect(evaluateFluidForceFields(validateFluidForceFields([{ ...point(), enabled: false }]), [1, 0, 0])).toEqual([0, 0, 0]);
    });

    it("rejects invalid settings instead of silently changing them", () => {
        expect(() => validateFluidForceFields([point(), point()])).toThrow("duplicate");
        expect(() => validateFluidForceFields([{ ...point(), strength: Infinity }])).toThrow("strength");
        expect(() => validateFluidForceFields([{ ...point(), falloffPower: -1 }])).toThrow("falloffPower");
        expect(() => validateFluidForceFields([{ ...point(), useMinDistance: true, minDistance: 4, useMaxDistance: true, maxDistance: 2 }])).toThrow("minDistance");
        expect(() => validateFluidForceFields([{ ...createDefaultFluidForceField("guide", "guide"), start: [0, 0, 0], end: [0, 0, 0] }])).toThrow("endpoints");
        expect(() => validateFluidForceFields([{ ...point(), type: "surface" }])).toThrow("unsupported");
    });

    it("clones definitions while preserving forward-compatible metadata", () => {
        const input = [{ ...point(), future: { revision: 3 } }];
        const output = validateFluidForceFields(input);
        expect(output).toEqual(input);
        expect(output[0]).not.toBe(input[0]);
    });
});

describe("ordered fluid force passes", () => {
    const spec = (name: string): ForceFieldSpec => ({ struct: name, wgsl: "force", buffer: {} as GPUBuffer });
    it("caches complete shader sources and preserves base-before-authored order", () => {
        const pipeline = vi.fn<FluidForcePasses["createPipeline"]>(() => ({}) as GPUComputePipeline);
        const bindings = vi.fn<FluidForcePasses["createBindGroup"]>(() => ({}) as GPUBindGroup);
        const state = createFluidForcePasses((entry) => entry.struct + entry.wgsl, pipeline, bindings);
        const base = spec("base"),
            authored = spec("authored");
        const combined = { ...base, additional: [authored] };
        updateFluidForcePasses(state, combined);
        expect(pipeline.mock.calls).toHaveLength(2);
        expect(bindings.mock.calls.map((call) => call[1])).toEqual([combined, authored]);
        updateFluidForcePasses(state, null);
        expect(state.commands).toEqual([]);
        updateFluidForcePasses(state, combined);
        expect(pipeline).toHaveBeenCalledTimes(2);
        expect(bindings).toHaveBeenCalledTimes(2);
    });

    it("rejects cycles without changing the current command list", () => {
        const state = createFluidForcePasses(
            (entry) => entry.wgsl,
            () => ({}) as GPUComputePipeline,
            () => ({}) as GPUBindGroup
        );
        const children: ForceFieldSpec[] = [];
        const field = { ...spec("cycle"), additional: children };
        children.push(field);
        expect(() => updateFluidForcePasses(state, field)).toThrow("cyclic");
        expect(state.commands).toEqual([]);
        expect(state.spec).toBeNull();
    });
});
