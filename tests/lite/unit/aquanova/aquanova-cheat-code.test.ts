import { describe, expect, it, vi } from "vitest";

import { AquanovaBehaviorManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/aquanova-behavior-manager";
import { createCheatCodeMatcher } from "../../../../lab/lite/src/demos/aquanova/cheat-code";
import { createSceneNode } from "../../../../packages/babylon-lite/src/scene/scene-node";
import type { Mesh } from "../../../../packages/babylon-lite/src";

function mesh(name: string): Mesh {
    return createSceneNode(name) as Mesh;
}

describe("Aquanova cheat codes", () => {
    it("recognizes idkfa case-insensitively and ignores non-character keys", () => {
        const activate = vi.fn();
        const enter = createCheatCodeMatcher("idkfa", activate);

        for (const key of ["i", "d", "Shift", "K", "f", "a"]) enter(key);
        expect(activate).toHaveBeenCalledOnce();

        for (const key of ["x", "i", "d", "k", "f", "a"]) enter(key);
        expect(activate).toHaveBeenCalledTimes(2);
    });

    it("enables every weapon entity through the normal acquisition event", () => {
        const liquefactor = mesh("liquefactor");
        const antiGravityGun = mesh("antiGravityGun");
        const manager = new AquanovaBehaviorManager({
            library: {
                weaponLiquefactor: {},
                weaponAntiGravityGun: {},
            },
            entities: {
                itemLiquefactor: { behaviors: [{ name: "weaponLiquefactor" }] },
                itemAntiGravityGun: { behaviors: [{ name: "weaponAntiGravityGun" }] },
            },
            meshesByEntityName: new Map([
                ["itemLiquefactor", [liquefactor]],
                ["itemAntiGravityGun", [antiGravityGun]],
            ]),
            entityNameOf: (target) => target.name,
        });
        const emit = vi.spyOn(manager.events, "emit");

        manager.acquireAllWeapons();

        expect(emit.mock.calls).toEqual([
            ["entityEvent", { name: "itemLiquefactor", event: "enable" }],
            ["entityEvent", { name: "itemAntiGravityGun", event: "enable" }],
        ]);
    });
});
