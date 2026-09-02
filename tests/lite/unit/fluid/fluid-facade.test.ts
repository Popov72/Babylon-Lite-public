import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const SOURCE = readFileSync(resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/core/fluid-facade.ts"), "utf-8");
const POLYGON_SOURCE = readFileSync(resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/rendering/polygon-surface-render.ts"), "utf-8");
const FLUID_HOST = readFileSync(resolve(__dirname, "../../../../lab/lite/src/demos/fluid.ts"), "utf-8");
const AQUANOVA_FLUID_HOST = readFileSync(resolve(__dirname, "../../../../lab/lite/src/demos/aquanova-fluid-sim.ts"), "utf-8");
const GPU_TYPES = ["GPUBuffer", "GPUTexture", "GPUTextureView", "GPUSampler", "GPUDevice", "GPUCommandEncoder", "GPUQuerySet"] as const;

describe("fluid simulation pure-state runtime", () => {
    it("publishes standalone lifecycle and render-layer functions", () => {
        for (const name of [
            "prepareFluidReconfiguration",
            "prepareFluidReconfigurationUpdate",
            "commitFluidReconfiguration",
            "cancelFluidReconfiguration",
            "disposePreparedFluidReconfiguration",
            "createFluidSimulation",
            "reconfigureFluidSimulation",
            "stepFluidSimulation",
            "setFluidSimulationSceneSdf",
            "setFluidSimulationForceField",
            "setFluidSimulationProfiler",
            "readFluidSimulationPressureDiagnostics",
            "resetFluidSimulation",
            "setFluidSimulationFlow",
            "setFluidSimulationFoam",
            "setFluidSimulationParameter",
            "setFluidSimulationMaterial",
            "refreshFluidSimulationPolygonSurface",
            "writeFluidSimulationPositions",
            "readFluidSimulationPositions",
            "createFluidSimulationCollection",
            "setFluidSimulationCollectionSources",
            "createFluidSimulationCollectionParticleStream",
            "refreshFluidSimulationCollectionParticleStream",
            "attachFluidSimulationCollectionRenderLayer",
            "createFluidParticleChannel",
            "readFluidParticleChannel",
            "createFluidParticleSpatialQuery",
            "sampleFluidParticleSpatialQuery",
            "createFluidRenderEnvironment",
            "getFluidSimulationDiagnostics",
            "getFluidSimulationCollectionDiagnostics",
            "attachFluidSimulationRenderLayer",
            "configureFluidSimulationRenderLayer",
            "detachFluidSimulationRenderLayer",
            "createFluidSimulationRenderCompositor",
            "configureFluidSimulationRenderCompositor",
            "disposeFluidSimulationRenderCompositor",
            "disposeFluidSimulation",
        ]) {
            expect(SOURCE).toContain(`export function ${name}`);
        }
    });

    it("keeps behavior off the public state interfaces", () => {
        const simulation = SOURCE.match(/export interface FluidSimulation \{[\s\S]*?\n\}/)?.[0];
        const renderLayer = SOURCE.match(/export interface FluidSimulationRenderLayer \{[\s\S]*?\n\}/)?.[0];
        const prepared = SOURCE.match(/export interface PreparedFluidReconfiguration \{[\s\S]*?\n\}/)?.[0];
        const collection = SOURCE.match(/export interface FluidSimulationCollection \{[\s\S]*?\n\}/)?.[0];
        const channel = SOURCE.match(/export interface FluidParticleChannel \{[\s\S]*?\n\}/)?.[0];
        expect(simulation).toBeDefined();
        expect(renderLayer).toBeDefined();
        expect(prepared).toBeDefined();
        expect(collection).toBeDefined();
        expect(channel).toBeDefined();
        expect(simulation).not.toMatch(/^\s+\w+\([^)]*\):/m);
        expect(renderLayer).not.toMatch(/^\s+\w+\([^)]*\):/m);
        expect(prepared).not.toMatch(/^\s+\w+\([^)]*\):/m);
        expect(collection).not.toMatch(/^\s+\w+\([^)]*\):/m);
        expect(channel).not.toMatch(/^\s+\w+\([^)]*\):/m);
    });

    it("references no raw WebGPU type in its public state interfaces", () => {
        for (const name of [
            "FluidSimulation",
            "FluidSimulationCollection",
            "FluidSimulationCollectionParticleStream",
            "FluidParticleStream",
            "FluidParticleChannel",
            "FluidParticleSpatialQuery",
            "FluidSimulationRenderLayer",
        ]) {
            const declaration = SOURCE.match(new RegExp(`export interface ${name} \\{[\\s\\S]*?\\n\\}`))?.[0];
            expect(declaration, name).toBeDefined();
            for (const gpu of GPU_TYPES) {
                expect(declaration, `${name} must not reference ${gpu}`).not.toContain(gpu);
            }
        }
    });

    it("resolves declarative options through the shared simulation config", () => {
        expect(SOURCE).toContain("resolveFluidSimulationConfig(options.compatibilityProfile");
        expect(SOURCE).toContain("physics: options.physics");
        expect(SOURCE).toContain("semantics: options.semantics");
    });

    it("uses the shared deferred-retirement path for replaced backends and tasks", () => {
        expect(SOURCE).toContain("retireGpuResources(engineOf(entry.simulation), () => entry.previous.dispose())");
        expect(SOURCE).toContain("retireGpuResources(engineOf(simulation), () => {");
        expect(SOURCE).toContain("task.dispose();");
        expect(SOURCE).not.toContain("entry.previous.dispose();\n");
    });

    it("selects live polygon depth for polygon foam without weakening polygon occlusion", () => {
        expect(SOURCE).toContain("readonly polygonSurfaceLayer?: FluidSimulationRenderLayer");
        expect(SOURCE).toContain("foamPolygonSurfaceDepth && options.polygonSurfaceLayer");
        expect(SOURCE).toContain("layerDepthView(options.polygonSurfaceLayer)");
        expect(FLUID_HOST).toContain("polygonSurfaceLayer: polygonSurfaceTask");
        expect(AQUANOVA_FLUID_HOST).toContain("polygonSurfaceLayer: polygonSurfaceTask");
        expect(POLYGON_SOURCE).toContain("if (input.eyeDepth > sceneEyeDepth + 0.02) {\n            discard;");
        expect(POLYGON_SOURCE).toContain('targets: [{ format: engine.format }, { format: "rg32float" }]');
    });
});
