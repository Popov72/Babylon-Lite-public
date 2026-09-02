import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("fluid foam lifecycle opacity", () => {
    const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/rendering/foam-render.ts"), "utf8");

    it("applies global opacity to every debug composite branch", () => {
        expect(source).toContain("fn debugComposite(rgb: vec3<f32>) -> vec4<f32>");
        expect(source).toContain("return vec4<f32>(rgb, u.sub.w);");
        expect(source.match(/return debugComposite\(/g)).toHaveLength(7);
        expect(source).toContain("return vec4<f32>(c, min(a, 1.0) * 0.95 * u.sub.w);");
    });

    it("fades diffuse particles before their lifetime reaches zero", () => {
        expect(source).toContain("gain *= smoothstep(0.0, 0.3, d.p.w);");
    });

    it("fades spray in only after it separates from the water surface", () => {
        expect(source).toContain("let separation = surfZ - i.eyeZ;");
        expect(source).toContain("outc.b = w * smoothstep(u.texel.z, 2.0 * u.texel.z, separation);");
    });

    describe("fluid foam surface depth", () => {
        it("follows the local liquid-depth tangent plane with a soft error mask", () => {
            const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/rendering/foam-render.ts"), "utf8");
            expect(source).toContain("@location(4) @interpolate(flat) centreUv");
            expect(source).toContain("let expectedZ = centreZ + dot(gradient, fragXY - centrePixel);");
            expect(source).toContain("let patchWeight = 1.0 - smoothstep");
            expect(source).toContain("orientationWeight = smoothstep");
            expect(source).toContain("outc.r = w * centreWeight * patchWeight * orientationWeight;");
            expect(source).toContain("splatData[46] = r * 1.5");
        });
    });

    it("keeps coverage discard independent from lifecycle opacity", () => {
        expect(source).toContain("if (baseA <= 0.002) { discard; }");
        expect(source).toContain("return vec4<f32>(outRGB, min(baseA, 1.0) * u.sub.w);");
        expect(source).toContain("if (!enabled || opacity <= 0)");
    });
});
