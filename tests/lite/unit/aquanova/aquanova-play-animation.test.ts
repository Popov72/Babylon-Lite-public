import { describe, expect, it } from "vitest";
import type { AnimationGroup, Mesh } from "../../../../packages/babylon-lite/src";
import { createSceneNode } from "../../../../packages/babylon-lite/src/scene/scene-node";
import { AquanovaBehaviorManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/aquanova-behavior-manager";
import { pauseAnimationsTargetingEntities, PlayAnimationBehavior, resumeAnimations } from "../../../../lab/lite/src/demos/aquanova/behaviors/play-animation";
import type { AquanovaGameContext } from "../../../../lab/lite/src/demos/aquanova/behaviors/game-context";

function mesh(name: string): Mesh {
    return createSceneNode(name) as Mesh;
}

function animationGroup(name: string, targetName: string): AnimationGroup {
    return {
        name,
        duration: 1,
        isPlaying: false,
        currentTime: 0.75,
        targetedAnimations: [{ targetName, path: "rotation" }],
        speedRatio: 1,
        loopAnimation: false,
        weight: 1,
        _stopped: true,
    };
}

describe("Aquanova playAnimation behavior", () => {
    it("plays the named animation from the beginning and honours loop false", () => {
        const idle = animationGroup("Idle", "other");
        const fan = animationGroup("Fan", "fan_blades");
        const behavior = new PlayAnimationBehavior("fan", [mesh("fan")], { animation: "Fan", loop: false }, { animationGroups: [idle, fan] });

        behavior.start();

        expect(idle.isPlaying).toBe(false);
        expect(fan.isPlaying).toBe(true);
        expect(fan.currentTime).toBe(0);
        expect(fan.loopAnimation).toBe(false);

        behavior.dispose();
        expect(fan.isPlaying).toBe(false);
        expect(fan.currentTime).toBe(0);
        expect(fan._stopped).toBe(true);
    });

    it("plays the first animation and loops by default", () => {
        const unrelated = animationGroup("Unrelated", "other");
        const first = animationGroup("First", "animated");
        const second = animationGroup("Second", "animated_blades");
        const behavior = new PlayAnimationBehavior("animated", [mesh("animated")], {}, { animationGroups: [unrelated, first, second] });

        behavior.start();

        expect(first.isPlaying).toBe(true);
        expect(first.loopAnimation).toBe(true);
        expect(second.isPlaying).toBe(false);
        expect(unrelated.isPlaying).toBe(false);
    });

    it("keeps identically named clips on separate entities independent", () => {
        const firstFan = animationGroup("Fan_Idle", "fan1_propeller");
        const secondFan = animationGroup("Fan_Idle", "fan2_propeller");
        const behavior = new PlayAnimationBehavior("fan2", [mesh("fan2")], { animation: "Fan_Idle" }, { animationGroups: [firstFan, secondFan] });

        behavior.start();

        expect(firstFan.isPlaying).toBe(false);
        expect(secondFan.isPlaying).toBe(true);
    });

    it("pauses and resumes only playing animations that target liquefied entities", () => {
        const firstFan = animationGroup("Fan", "fan1_propeller");
        const secondFan = animationGroup("Fan", "fan2_propeller");
        const alreadyPaused = animationGroup("Idle", "fan1");
        firstFan.isPlaying = true;
        secondFan.isPlaying = true;

        const paused = pauseAnimationsTargetingEntities([firstFan, secondFan, alreadyPaused], new Set(["fan1"]));

        expect(paused).toEqual([firstFan]);
        expect(firstFan.isPlaying).toBe(false);
        expect(secondFan.isPlaying).toBe(true);
        expect(alreadyPaused.isPlaying).toBe(false);

        resumeAnimations(paused);
        expect(firstFan.isPlaying).toBe(true);
        expect(alreadyPaused.isPlaying).toBe(false);
    });

    it("does nothing when no animation is available and none was requested", () => {
        const behavior = new PlayAnimationBehavior("static", [mesh("static")], {}, { animationGroups: [] });
        expect(() => behavior.start()).not.toThrow();
        behavior.dispose();
    });

    it("rejects invalid parameters and missing named animations", () => {
        expect(() => new PlayAnimationBehavior("invalid", [mesh("invalid")], { animation: "" }, { animationGroups: [] })).toThrow("animation must be a non-empty animation name");
        expect(() => new PlayAnimationBehavior("invalid", [mesh("invalid")], { loop: "yes" } as unknown as { loop: boolean }, { animationGroups: [] })).toThrow(
            "loop must be a boolean"
        );
        expect(() => new PlayAnimationBehavior("invalid", [mesh("invalid")], { animation: "Missing" }, { animationGroups: [] }).start()).toThrow(
            'animation "Missing" was not found'
        );
    });

    it("creates one runtime instance per entity rather than per mesh primitive", async () => {
        const group = animationGroup("Fan", "fan_blades");
        const firstPrimitive = mesh("fan-primitive-0");
        const secondPrimitive = mesh("fan-primitive-1");
        const manager = new AquanovaBehaviorManager({
            library: { playAnimation: { animation: "Fan" } },
            entities: { fan: { behaviors: [{ name: "playAnimation", loop: false }] } },
            meshesByEntityName: new Map([["fan", [firstPrimitive, secondPrimitive]]]),
            entityNameOf: () => "fan",
        });

        await manager.start({ animationGroups: [group] } as unknown as Omit<AquanovaGameContext, "events" | "weaponInventory">);

        expect(manager.describeInstances()).toEqual([{ name: "playAnimation", mesh: "fan" }]);
        expect(group.isPlaying).toBe(true);
        expect(group.loopAnimation).toBe(false);

        manager.dispose();
        expect(group.isPlaying).toBe(false);
    });
});
