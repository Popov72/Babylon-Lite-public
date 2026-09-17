import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("FLIP Reference shared controls", () => {
    it("does not force whitewater off during Reference selection or reconfiguration", () => {
        const provider = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/experimental/flip-reference-backend.ts"), "utf8");
        const host = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/fluid.ts"), "utf8");
        expect(provider).toContain("supportsFoam: true");
        expect(host).toContain("foam: referenceBackend!.supportsFoam");
        expect(host).not.toContain("foamEnabled: !referenceSelected()");
        expect(host).not.toContain("foam: state.foam ? { ...state.foam, enabled: false }");
        expect(host).toContain("foam: base.backend?.supportsFoam !== false && foamValues.enabled");
    });

    it("provides a General-section implementation slot and mutable capability refresh", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/controls/controls-panel.ts"), "utf8");

        expect(source).toContain('implementationSlot.dataset.fluidImplementationSlot = "true"');
        expect(source).toContain("generalItems.push(methodTitle, methodSel, implementationSlot)");
        expect(source).toContain("setCapabilities(capabilities: Partial<FluidControlsCapabilities>)");
        expect(source).toContain("hostCapabilities = capabilities");
        expect(source).toContain("const backendPhysics = methodCaps().physicsParameters");
        expect(source).toContain('const foamSection = sections.get("Foam")');
    });

    it("keeps flow objects visible while enforcing initial-only creation and activation", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/controls/flow-editor.ts"), "utf8");

        expect(source).toContain("readonly setInitialOnly: (enabled: boolean) => void");
        expect(source).toContain('behavior: initialOnly ? "initial" : "inflow"');
        expect(source).toContain("inflowOption.disabled = true");
        expect(source).toContain('initialOnly && kind === "sink"');
        expect(source).toContain("enabled.disabled = unsupportedActivation && !object.enabled");
        expect(source).toContain("Existing entries remain visible so they can be disabled or deleted.");
        expect(source).not.toContain("flow.sinks = []");
    });
});
