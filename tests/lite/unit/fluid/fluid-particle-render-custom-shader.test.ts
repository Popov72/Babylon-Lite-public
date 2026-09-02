import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

describe("fluid particle custom shader contract", () => {
    const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/rendering/particle-render.ts"), "utf8");

    it("documents stable built-in and custom uniform bindings", () => {
        expect(source).toContain("binding 0 is the 128-byte camera uniform");
        expect(source).toContain("binding 1 is the simulation `array<vec4<f32>>` position");
        expect(source).toContain("binding 2 is the simulation `array<f32>` debug buffer");
        expect(source).toContain("binding 3");
        expect(source).toContain('{ binding: 0, visibility, buffer: { type: "uniform" } }');
        expect(source).toContain('{ binding: 1, visibility, buffer: { type: "read-only-storage" } }');
        expect(source).toContain('{ binding: 2, visibility, buffer: { type: "read-only-storage" } }');
        expect(source).toContain("...(customUniformBuffer ? [{ binding: 3");
    });

    it("keeps built-in shader, blend, and reverse-Z depth defaults", () => {
        expect(source).toContain("code: shader?.code ?? RENDER_WGSL");
        expect(source).toContain("const blend = shader ? shader.blend : DEFAULT_BLEND");
        expect(source).toContain('depthCompare: shader?.depthCompare ?? "greater-equal"');
        expect(source).toContain("depthWriteEnabled: shader?.depthWriteEnabled ?? true");
    });

    it("preserves the authored sphere tint while applying motion and rim lighting", () => {
        expect(source).toContain("min(base * 1.5, vec3<f32>(1.0))");
        expect(source).toContain("col += i.color * (fres * 0.25);");
        expect(source).not.toContain("base + vec3<f32>(0.5)");
        expect(source).not.toContain("vec3<f32>(0.45, 0.65, 0.95) * (fres * 0.25)");
    });

    it("supports lazy runtime shader replacement and restoration", () => {
        expect(source).toContain("setShader(shader: ParticleRenderShaderOptions | null): void");
        expect(source).toContain("pipeline = null;");
        expect(source).toContain("task.execute = executeNeedsBuild;");
        expect(source).toContain("task.execute = executeBuilt;");
    });

    it("updates and destroys the owned custom uniform buffer", () => {
        expect(source).toContain("setCustomUniforms(data: ArrayBufferView): void");
        expect(source).toContain("device.queue.writeBuffer(customUniformBuffer, 0, customShader.customUniforms)");
        expect(source).toContain("customUniformBuffer?.destroy();");
    });

    it("preserves simulation-buffer rebinding", () => {
        expect(source).toMatch(/setSim\(s: FluidSim\): void \{\s+currentSim = s;\s+buildBindGroup\(\);/);
    });
});
