import { describe, expect, it } from "vitest";
import { mat4Compose, mat4Multiply, mat4Translation } from "../../../../packages/babylon-lite/src/index";
import type { FluidEmitter, FluidTransform, Mat4 } from "../../../../packages/babylon-lite/src/index";
import { createFluidInitialParticles } from "../../../../packages/babylon-lite/src/fluid/core/sim-common";
import { createEmitterSourceTransform, resolveEmitterSourceTransform } from "../../../../lab/lite/src/demos/fluid/emitter-source-transform";

const authored: FluidTransform = {
    position: [0.26668, 0.368409, -2.167276],
    rotation: [0.7071067094802856, 0.7071067094802856, 0, 0],
    scale: [1, 1.41396963596344, 1],
};

function compose(transform: FluidTransform): Mat4 {
    const [x, y, z] = transform.position;
    const [qx, qy, qz, qw] = transform.rotation;
    const length = Math.hypot(qx, qy, qz, qw);
    return mat4Compose(x, y, z, qx / length, qy / length, qz / length, qw / length, ...transform.scale);
}

function expectMatrix(actual: Mat4, expected: Mat4): void {
    for (let index = 0; index < 16; index++) {
        expect(actual[index]).toBeCloseTo(expected[index]!, 5);
    }
}

describe("imported emitter shape-to-source transforms", () => {
    it("preserves a recentered prism's original axes and scale at the source rest pose", () => {
        const source = mat4Compose(0.26668, 0, 0, 0, 0, 0, 1, 1.41396963596344, 1, 1);
        const binding = createEmitterSourceTransform(source, authored);
        const reset = resolveEmitterSourceTransform(binding, source);
        expect(reset).toEqual(authored);
        reset.position[0] = 999;
        expect(resolveEmitterSourceTransform(binding, source)).toEqual(authored);
    });

    it("produces the same non-baked initial particle lattice before and after source motion and rewind", () => {
        const source = mat4Compose(0.26668, 0, 0, 0, 0, 0, 1, 1.41396963596344, 1, 1);
        const binding = createEmitterSourceTransform(source, authored);
        const emitter: FluidEmitter = {
            id: "water",
            name: "Water",
            enabled: true,
            behavior: "initial",
            transform: structuredClone(authored),
            shape: {
                type: "polygonPrism",
                points: [
                    [-0.418, 1.833],
                    [0.418, 1.833],
                    [-0.052, -1.833],
                ],
                thickness: 2,
            },
            sampling: "volume",
            velocity: [0, 0, 0],
            velocitySpace: "world",
            sourceNode: "Water",
            spread: 0,
        };
        const flow = { emitters: [emitter], sinks: [] };
        const bounds = { min: [-1.1, 0.05, -4] as [number, number, number], max: [1.7, 1.5, 4] as [number, number, number] };
        const initial = createFluidInitialParticles(512, flow, 0.01, bounds, true)!;
        expect(initial.activeCount).toBeGreaterThan(0);
        emitter.transform = resolveEmitterSourceTransform(binding, mat4Multiply(mat4Translation(0, 0, 2), source));
        emitter.transform = resolveEmitterSourceTransform(binding, source);
        const reset = createFluidInitialParticles(512, flow, 0.01, bounds, true)!;
        expect(reset.activeCount).toBe(initial.activeCount);
        expect(reset.positions).toEqual(initial.positions);
        expect(reset.velocities).toEqual(initial.velocities);
    });

    it("applies source rotation and translation to the entire offset shape", () => {
        const source = mat4Compose(0.26668, 0, 0, 0, 0, 0, 1, 1.41396963596344, 1, 1);
        const binding = createEmitterSourceTransform(source, authored);
        const motion = mat4Compose(3, 2, 1, 0, Math.sin(0.3), 0, Math.cos(0.3), 1, 1, 1);
        const actual = resolveEmitterSourceTransform(binding, mat4Multiply(motion, source));
        expectMatrix(compose(actual), mat4Multiply(motion, compose(authored)));
        expect(resolveEmitterSourceTransform(binding, source)).toEqual(authored);
    });

    it("retains the analytical transform under mirrored sources and parent placement", () => {
        const source = mat4Compose(1, 2, 3, 0, 0, 0, 1, -2, 3, 4);
        const binding = createEmitterSourceTransform(source, authored);
        const shift = mat4Translation(10, -5, 2);
        expectMatrix(compose(resolveEmitterSourceTransform(binding, mat4Multiply(shift, source))), mat4Multiply(shift, compose(authored)));
    });

    it("rejects a singular source matrix rather than seeding at a fallback origin", () => {
        const source = mat4Compose(0, 0, 0, 0, 0, 0, 1, 0, 1, 1);
        expect(() => createEmitterSourceTransform(source, authored)).toThrow("singular source-node transform");
    });
});
