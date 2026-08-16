import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("fluid foam lifecycle opacity", () => {
    const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/foam-render.ts"), "utf8");

    it("applies global opacity to every debug composite branch", () => {
        expect(source).toContain("fn debugComposite(rgb: vec3<f32>) -> vec4<f32>");
        expect(source).toContain("return vec4<f32>(rgb, u.sub.w);");
        expect(source.match(/return debugComposite\(/g)).toHaveLength(7);
        expect(source).toContain("return vec4<f32>(c, min(a, 1.0) * 0.95 * u.sub.w);");
    });

    it("keeps coverage discard independent from lifecycle opacity", () => {
        expect(source).toContain("if (baseA <= 0.002) { discard; }");
        expect(source).toContain("return vec4<f32>(outRGB, min(baseA, 1.0) * u.sub.w);");
        expect(source).toContain("if (!enabled || opacity <= 0)");
    });
});
