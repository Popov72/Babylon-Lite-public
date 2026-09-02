import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const demoSource = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/fluid.ts"), "utf8");

describe("fluid demo single active facade simulation", () => {
    it("constructs one opaque facade simulation", () => {
        expect(demoSource).toContain("function createSim(count: number, scale: number): FluidSimulation");
        expect(demoSource).toContain("createFluidSimulation(engine, simulationOptions(count, scale))");
        expect(demoSource).not.toMatch(/\bcreate(?:Pbf|Flip|MlsMpm|PbMpm)Sim\b/);
        expect(demoSource).not.toMatch(/\bFluidSim\b/);
        expect(demoSource).not.toContain("._sim");
    });

    it("reconfigures the stable handle transactionally", () => {
        expect(demoSource).toContain("prepareFluidReconfigurationUpdate(");
        expect(demoSource).toContain("commitFluidReconfiguration(prepared)");
        expect(demoSource).toContain("cancelFluidReconfiguration(prepared)");
        expect(demoSource).toContain("particleCount = previousParticleCount");
        expect(demoSource).toContain("physicsScale = previousPhysicsScale");
        expect(demoSource).toContain("flipParticleCapacityRequest = previousCapacityRequest");
    });

    it("routes host operations through standalone facade functions", () => {
        for (const operation of [
            "setFluidSimulationFlow(activeSim",
            "setFluidSimulationSceneSdf(activeSim",
            "updateFluidSimulationEmitter(activeSim",
            "setFluidSimulationProfiler(activeSim",
            "setFluidSimulationMaterial(activeSim",
            "resetFluidSimulation(activeSim",
            "stepFluidSimulation(activeSim",
        ]) {
            expect(demoSource).toContain(operation);
        }
    });

    it("keeps authoritative active and transition diagnostics", () => {
        expect(demoSource).toContain("function trackSimBuild(method: string)");
        expect(demoSource).toContain("canvas.dataset.simulationActiveBackend = activeSimMethod");
        expect(demoSource).toContain("canvas.dataset.backendBuildsFlip");
        expect(demoSource).toContain("canvas.dataset.backendBuildsPbf");
        expect(demoSource).toContain("canvas.dataset.simulationGpuBytes = String(activeSim.gpuBytes)");
        expect(demoSource).toContain("lastTransitionPeakBytes = prepared.transitionPeakBytes");
    });

    it("keeps foam rendering and diagnostics available on every foam-capable backend", () => {
        const visibleStart = demoSource.indexOf("function foamRenderVisible()");
        const visibleEnd = demoSource.indexOf("\n    function currentFoamConfig", visibleStart);
        const countsStart = demoSource.indexOf("function refreshFoamParticleCounts()");
        const countsEnd = demoSource.indexOf("\n    function refreshParticleUsageStatus", countsStart);
        const visible = demoSource.slice(visibleStart, visibleEnd);
        const counts = demoSource.slice(countsStart, countsEnd);

        expect(visible).toContain("controls.getValues().foam.enabled");
        expect(visible).not.toContain('activeSim.method === "FLIP"');
        expect(counts).not.toContain('activeSim.method === "FLIP"');
    });
});
