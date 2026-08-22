import { describe, expect, it } from "vitest";
import { collisionShapesForModule, type ShipCollisionShape } from "../../../../lab/lite/src/demos/aquanova/collision-shapes";

const shape: ShipCollisionShape = {
    kind: "box",
    centre: [0, 0, 0],
    halfExtents: [1, 1, 1],
};

describe("Aquanova collision module lookup", () => {
    it("matches kit-relative GLB module ids to full manifest catalogue ids", () => {
        const collision = {
            "Modular SciFi MegaKit/Platforms/Platform_Metal2": shape,
        };

        expect(collisionShapesForModule(collision, "Platforms/Platform_Metal2")).toBe(shape);
        expect(collisionShapesForModule(collision, "Modular SciFi MegaKit/Platforms/Platform_Metal2")).toBe(shape);
    });

    it("also matches a full GLB module id to a relative manifest key", () => {
        const collision = {
            "Props/Prop_Crate2": shape,
        };

        expect(collisionShapesForModule(collision, "Modular SciFi MegaKit/Props/Prop_Crate2")).toBe(shape);
        expect(collisionShapesForModule(collision, "Props\\Prop_Crate2")).toBe(shape);
        expect(collisionShapesForModule(collision, "Props/Missing")).toBeUndefined();
    });
});
