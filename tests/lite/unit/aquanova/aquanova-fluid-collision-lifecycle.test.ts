import { describe, expect, it } from "vitest";
import { forEachFluidCollisionSet } from "../../../../lab/lite/src/demos/aquanova/fluid-collision-lifecycle";

describe("Aquanova fluid collision lifecycle", () => {
    it("visits collision sets owned by liquefaction and behavior simulations", () => {
        const liquefaction = [{ collision: "liquefaction-a" }, { collision: "liquefaction-b" }];
        const behavior = [{ collision: "behavior-a" }, { collision: null }];
        const visited: string[] = [];

        forEachFluidCollisionSet(liquefaction, behavior, (collision) => visited.push(collision));

        expect(visited).toEqual(["liquefaction-a", "liquefaction-b", "behavior-a"]);
    });
});
