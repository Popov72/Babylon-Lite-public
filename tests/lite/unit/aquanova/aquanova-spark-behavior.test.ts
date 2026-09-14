import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import type { Mesh } from "../../../../packages/babylon-lite/src";
import { SparkBehavior } from "../../../../lab/lite/src/demos/aquanova/behaviors/spark";
import { sparkEmitterDisplayed } from "../../../../lab/lite/src/demos/aquanova/behaviors/spark-runtime";

function testMesh(name = "sparking-panel"): Mesh {
    return { name } as Mesh;
}

describe("Aquanova spark behavior", () => {
    it("registers the owner mesh with the default spark configuration", () => {
        const dispose = vi.fn();
        const sparks = { register: vi.fn(() => ({ dispose })) };
        const mesh = testMesh();
        const behavior = new SparkBehavior("sparking-panel", [mesh], {}, { sparks });

        behavior.start();

        expect(sparks.register).toHaveBeenCalledWith(mesh, {
            rate: 30,
            speed: 2.5,
            lifetime: 0.45,
            size: 0.035,
            spread: 0.08,
            gravity: 9.81,
        });
        behavior.dispose();
        expect(dispose).toHaveBeenCalledOnce();
    });

    it("accepts custom values and rejects unsupported fields", () => {
        const sparks = { register: vi.fn(() => ({ dispose: vi.fn() })) };
        const behavior = new SparkBehavior("sparking-panel", [testMesh()], { rate: 12, speed: 4, lifetime: 0.8, size: 0.06, spread: 0, gravity: 3 }, { sparks });

        expect(behavior.options).toEqual({ rate: 12, speed: 4, lifetime: 0.8, size: 0.06, spread: 0, gravity: 3 });
        expect(() => new SparkBehavior("sparking-panel", [testMesh()], { unsupported: true } as never, { sparks })).toThrow("[aquanova] spark.unsupported is not supported");
    });

    it("exposes matching editor defaults", () => {
        const metadata = JSON.parse(readFileSync("lab/public/aquanova/behaviors.json", "utf8")) as {
            behaviors: Record<string, { properties: Record<string, { default?: unknown }> }>;
        };

        expect(metadata.behaviors.spark?.properties).toMatchObject({
            rate: { default: 30 },
            speed: { default: 2.5 },
            lifetime: { default: 0.45 },
            size: { default: 0.035 },
            spread: { default: 0.08 },
            gravity: { default: 9.81 },
        });
    });

    it("ignores mesh visibility and suppresses only portal-culled emitters", () => {
        const mesh = { visible: false } as Mesh;
        expect(sparkEmitterDisplayed(mesh, () => false)).toBe(true);
        expect(sparkEmitterDisplayed(mesh, () => true)).toBe(false);
    });
});
