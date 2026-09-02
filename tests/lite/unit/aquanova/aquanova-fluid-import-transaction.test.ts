import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

// The transaction contract itself is covered in fluid-controls-transaction.test.ts. These source
// assertions guard the host integration, whose render/GPU dependencies make it impractical to
// instantiate in a unit environment.
describe("Aquanova preset import applies render/foam setters atomically", () => {
    const source = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/aquanova-fluid-sim.ts"), "utf8");
    const start = source.indexOf("function applyImportedPreset(");
    const end = source.indexOf("const buildCurrentPreset", start);
    const body = source.slice(start, end);
    const onApplyStart = source.indexOf("onApply:");
    const onApplyEnd = source.indexOf("\n        on: {", onApplyStart);
    const onApplyBody = source.slice(onApplyStart, onApplyEnd);

    it("locates the applyImportedPreset body", () => {
        expect(start).toBeGreaterThan(-1);
        expect(end).toBeGreaterThan(start);
    });

    it("batches the render + foam setters inside controls.runTransaction", () => {
        const txIndex = body.indexOf("controls.runTransaction(");
        expect(txIndex).toBeGreaterThan(-1);
        // Representative render setters and the foam setter are applied at/after the transaction open.
        for (const setter of ["controls.setColor(", "controls.setAbsorption(", "controls.setSurfaceFilter(", "controls.setFoam("]) {
            const index = body.indexOf(setter);
            expect(index).toBeGreaterThan(txIndex);
        }
    });

    it("prepares live replacements before committing the imported session", () => {
        const prepare = body.lastIndexOf("reconfigureRunningSimulations()");
        const commitSession = body.indexOf("presetSession = nextSession");
        expect(body).not.toContain("restart()");
        expect(prepare).toBeGreaterThan(-1);
        expect(commitSession).toBeGreaterThan(prepare);
        expect(source).toContain("prepareFluidCollectionReconfiguration(");
        expect(source).toContain("cancelFluidCollectionReconfiguration(prepared)");
        expect(body).toContain("applyImportedPreset(previousPreset, false, false)");
    });

    it("rebuilds solver foam only through the committed facade transaction", () => {
        expect(onApplyStart).toBeGreaterThan(-1);
        expect(onApplyEnd).toBeGreaterThan(onApplyStart);
        expect(onApplyBody).toContain('if (changed("foam"))');
        expect(onApplyBody).toContain("reconfigureRunningSimulations()");
        expect(onApplyBody).not.toContain("setFluidSimulationFoam(");
        expect(body).not.toContain("setFluidSimulationFoam(");
    });
});
