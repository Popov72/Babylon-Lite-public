import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("Fluid lab host normalized controls integration", () => {
    const source = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/fluid.ts"), "utf8");
    const onApplyStart = source.indexOf("onApply:");
    const onApplyEnd = source.indexOf("\n        on: {", onApplyStart);
    const onApplyBody = source.slice(onApplyStart, onApplyEnd);
    const onStart = onApplyEnd;
    const onEnd = source.indexOf("\n        },\n    });", onStart);
    const legacyCallbacks = source.slice(onStart, onEnd);

    it("routes normalized control snapshots through onApply", () => {
        expect(onApplyStart).toBeGreaterThan(-1);
        expect(onApplyEnd).toBeGreaterThan(onApplyStart);
        for (const key of ["method", "schema", "pagedGrid", "foam"] as const) {
            expect(onApplyBody).toContain(`changed("${key}")`);
        }
    });

    it("keeps only immediate actions on the legacy callback surface", () => {
        expect(legacyCallbacks).toContain("onGridSettings:");
        expect(legacyCallbacks).toContain("onGridGizmo:");
        expect(legacyCallbacks).toContain("onReset:");
        expect(legacyCallbacks).not.toContain("onColor:");
        expect(legacyCallbacks).not.toContain("onPhysicsParam:");
        expect(legacyCallbacks).not.toContain("onFoamEnable:");
    });

    it("applies one complete foam snapshot with one simulation push", () => {
        const foamStart = onApplyBody.indexOf('if (changed("foam"))');
        const foamBody = onApplyBody.slice(foamStart);
        expect(foamStart).toBeGreaterThan(-1);
        expect(foamBody.match(/\bpushFoam\(\)/g)).toHaveLength(1);
        expect(foamBody).toContain("configureFluidSimulationRenderLayer(foamTask");
        expect(foamBody).toContain("surfaceFiltering:");
        expect(foamBody).toContain("debugTexture:");
    });

    it("applies one complete pair snapshot before resolving the replacement target", () => {
        const loadStart = source.indexOf("function loadPairState(");
        const loadEnd = source.indexOf("\n    // Switch to a (demo, method) pair", loadStart);
        const loadPairState = source.slice(loadStart, loadEnd);

        expect(loadPairState).toContain("loadingPairState = true");
        expect(loadPairState).toMatch(/loadingPairState = true;[\s\S]*?controls\.runTransaction\(/);
        expect(loadPairState).toContain("controls.setParticleCount(st.count)");
        expect(loadPairState).toContain("controls.setFlipParticleCapacity(st.count)");
        expect(loadPairState).toContain("controls.setPhysScale(nextPhysicsScale)");
        expect(loadPairState).toContain("controls.setGridSettings([...targetGrid.position], [...targetGrid.size], targetCellSize)");
        expect(loadPairState).toMatch(/finally \{\s*loadingPairState = false;/);
        expect(onApplyBody).toContain("controlsBinding && (applyingPairState || !pairChange)");
        expect(onApplyBody).toContain("applyingPairState ? undefined : changedKeys");
        expect(onApplyBody).toContain("applyingPairState ? bindingPlan.changedKeys : changedKeys");
        expect(onApplyBody).toContain('if (!applyingPairState && (changed("method") || changed("material")))');
        expect(source).toContain("preserveState: !loadingPairState");
    });

    it("keeps the committed simulation and controls state until replacement allocation succeeds", () => {
        const rebuildStart = source.indexOf("function rebuildSims(");
        const rebuildEnd = source.indexOf("\n    // ── Per-(demo, simulation)", rebuildStart);
        const rebuild = source.slice(rebuildStart, rebuildEnd);
        expect(rebuild).toContain("prepareFluidReconfigurationUpdate(");
        expect(rebuild).toContain("commitFluidReconfiguration(prepared)");
        expect(rebuild).toMatch(/commitFluidReconfiguration\(prepared\)[\s\S]*?catch \(error\)[\s\S]*?particleCount = previousParticleCount/);
        expect(rebuild).toContain("flipParticleCapacityRequest = previousCapacityRequest");
        expect(rebuild).toContain("lastTransitionPeakBytes = prepared.transitionPeakBytes");
    });

    it("initializes the shared controls binding only after the startup demo is active", () => {
        const bindingDefinition = source.indexOf("function initializeFluidControlsBinding()");
        const startupSwitch = source.indexOf('switchPair(boxDemo, "MLS-MPM", quality)');
        const bindingCall = source.indexOf("initializeFluidControlsBinding();", startupSwitch);

        expect(bindingDefinition).toBeGreaterThan(-1);
        expect(startupSwitch).toBeGreaterThan(bindingDefinition);
        expect(bindingCall).toBeGreaterThan(startupSwitch);
        expect(source).toContain('throw new Error("[fluid] cannot resolve the scene SDF before the initial demo is active.")');
    });
});
