import { describe, expect, it } from "vitest";
import { aquanovaFluidSimNames, renameAquanovaFluidSimReferences } from "../../../../lab/aquanova-fluid-sim-authoring";

describe("Aquanova fluid simulation authoring", () => {
    it("renames catalogue, preset, and entity fluidSim references without changing unrelated values", () => {
        const source = `{
  "fluidSim": [
    "water",
    "smoke"
  ],
  "behaviorPresets": {
    "liquid": {
      "base": "liquefaction",
      "fluidSim": [
        "water"
      ]
    }
  },
  "entities": {
    "tank": {
      "behaviors": [
        {
          "name": "fluidSimulation",
          "fluidSim": "water"
        }
      ]
    }
  },
  "label": "water"
}`;

        const renamed = renameAquanovaFluidSimReferences(source, "water", "clear-water");

        expect(JSON.parse(renamed)).toEqual({
            fluidSim: ["clear-water", "smoke"],
            behaviorPresets: {
                liquid: {
                    base: "liquefaction",
                    fluidSim: ["clear-water"],
                },
            },
            entities: {
                tank: {
                    behaviors: [
                        {
                            name: "fluidSimulation",
                            fluidSim: "clear-water",
                        },
                    ],
                },
            },
            label: "water",
        });
        expect(renamed).toContain('"label": "water"');
    });

    it("normalizes and deduplicates catalogue names", () => {
        expect(aquanovaFluidSimNames({ fluidSim: ["Water.JSON", "water", "liquid-slow"] })).toEqual(["water", "liquid-slow"]);
    });
});
