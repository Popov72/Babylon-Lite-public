import { describe, expect, it } from "vitest";
import { flipReferenceForceWgsl } from "../../../../packages/babylon-lite/src/fluid/experimental/flip-reference/force";

describe("Reference force integration shader", () => {
    it("uses the GPU live prefix and physical substep delta without rescaling the supplied impulse", () => {
        const source = flipReferenceForceWgsl({
            struct: "struct ForceFieldParams { force: vec4<f32> }",
            wgsl: "fn externalForce(pos: vec3<f32>, vel: vec3<f32>, dt: f32) -> vec3<f32> { return forceFieldParams.force.xyz * dt; }",
            buffer: {} as GPUBuffer,
        });
        expect(source).toContain("id.x >= atomicLoad(&forceRuntime[0])");
        expect(source).toContain("bitcast<f32>(atomicLoad(&forceRuntime[4]))");
        expect(source).toContain("previous + externalForce(forcePositions[id.x].xyz, previous, dt)");
        expect(source).toContain("atomicOr(&forceRuntime[1], 1u)");
        expect(source).toContain("atomicStore(&forceRuntime[2], 0u)");
        expect(source).not.toContain("clampLength");
    });
});
