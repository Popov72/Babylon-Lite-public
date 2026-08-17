import { describe, expect, it } from "vitest";
import { playerWeaponSwayMultiplier } from "../../../lab/lite/src/demos/aquanova/behaviors/player";

describe("Aquanova player weapon sway", () => {
    it("uses idle, walking, and running multipliers from movement input", () => {
        expect(playerWeaponSwayMultiplier(new Set())).toBe(1);
        expect(playerWeaponSwayMultiplier(new Set(["ShiftLeft"]))).toBe(1);
        expect(playerWeaponSwayMultiplier(new Set(["KeyW"]))).toBe(2);
        expect(playerWeaponSwayMultiplier(new Set(["ArrowRight", "ShiftRight"]))).toBe(4);
        expect(playerWeaponSwayMultiplier(new Set(["KeyW", "ShiftLeft"]), true)).toBe(1);
    });
});
