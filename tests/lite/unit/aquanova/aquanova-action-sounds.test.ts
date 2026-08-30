import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import type { Mesh } from "../../../../packages/babylon-lite/src";
import { AquanovaEventManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/aquanova-event-manager";
import { ExplodeBehavior } from "../../../../lab/lite/src/demos/aquanova/behaviors/explode";
import { WeaponPistolBehavior } from "../../../../lab/lite/src/demos/aquanova/behaviors/weapon-pistol";

function testMesh(name: string): Mesh {
    return { name } as Mesh;
}

function managedSound(source: string) {
    return { label: source, source, sound: {} as never };
}

describe("Aquanova action sounds", () => {
    it("plays pistolShot when the pistol launches a shot", async () => {
        const events = new AquanovaEventManager();
        const shotSound = managedSound("/aquanova/sounds/pistolShot.mp3");
        const sounds = {
            load: vi.fn(async () => shotSound),
            play: vi.fn(),
        };
        const weaponInventory = { acquire: vi.fn() };
        const weaponPistol = {
            setEnabled: vi.fn(),
            isReady: vi.fn(() => true),
            fire: vi.fn(),
            update: vi.fn(() => []),
            clear: vi.fn(),
        };
        const behavior = new WeaponPistolBehavior("pistol", [testMesh("pistol")], {}, { events, sounds, weaponInventory, weaponPistol } as never);

        await behavior.init();
        behavior.start();
        events.emit("entityEvent", { name: "pistol", event: "enable" });
        events.emit("weaponEquippedChanged", { slot: 3 });
        events.emit("weaponTriggerPressed", { held: false });
        events.emit("weaponAimUpdated", { mesh: null, point: null, distance: null });

        expect(sounds.load).toHaveBeenCalledWith("weaponPistol:pistolShot", "/aquanova/sounds/pistolShot.mp3?v=20260813-1", { preloadCount: 1 });
        expect(weaponPistol.fire).toHaveBeenCalledWith(null, null, null, 100, 80);
        expect(sounds.play).toHaveBeenCalledWith(shotSound);
        behavior.dispose();
    });

    it("plays bigExplosion once when the explosion activates", async () => {
        const events = new AquanovaEventManager();
        const explosionSound = managedSound("/aquanova/sounds/bigExplosion.mp3");
        const sounds = {
            load: vi.fn(async () => explosionSound),
            play: vi.fn(),
        };
        const explosions = { explode: vi.fn() };
        const mesh = testMesh("barrel");
        const behavior = new ExplodeBehavior("barrel", [mesh], {}, { events, explosions, sounds } as never);

        await behavior.init();
        behavior.start();
        events.emit("entityEvent", { name: "barrel", event: "explode" });
        events.emit("entityEvent", { name: "barrel", event: "explode" });

        expect(sounds.load).toHaveBeenCalledWith("explode:bigExplosion", "/aquanova/sounds/bigExplosion.mp3?v=20260813-1", { preloadCount: 1 });
        expect(sounds.play).toHaveBeenCalledOnce();
        expect(sounds.play).toHaveBeenCalledWith(explosionSound);
        expect(explosions.explode).toHaveBeenCalledOnce();
        expect(explosions.explode).toHaveBeenCalledWith("barrel", [mesh], {
            radius: 10,
            fragmentCount: 8,
            strength: 12,
            debrisLifetime: 15,
            fadeDuration: 2,
        });
        behavior.dispose();
    });

    it("accepts custom sound names and exposes both defaults in the editor", async () => {
        const events = new AquanovaEventManager();
        const sounds = {
            load: vi.fn(async (_label: string, source: string) => managedSound(source)),
            play: vi.fn(),
        };
        const pistol = new WeaponPistolBehavior("pistol", [testMesh("pistol")], { sound: "customShot" }, {
            events,
            sounds,
            weaponInventory: { acquire: vi.fn() },
            weaponPistol: { setEnabled: vi.fn(), isReady: vi.fn(), fire: vi.fn(), update: vi.fn(() => []), clear: vi.fn() },
        } as never);
        const explosion = new ExplodeBehavior("barrel", [testMesh("barrel")], { sound: "customExplosion" }, { events, explosions: { explode: vi.fn() }, sounds } as never);

        await Promise.all([pistol.init(), explosion.init()]);

        expect(sounds.load).toHaveBeenCalledWith("weaponPistol:customShot", "/aquanova/sounds/customShot.mp3?v=20260813-1", { preloadCount: 1 });
        expect(sounds.load).toHaveBeenCalledWith("explode:customExplosion", "/aquanova/sounds/customExplosion.mp3?v=20260813-1", { preloadCount: 1 });

        const metadata = JSON.parse(readFileSync("lab/lite/src/demos/aquanova/editor/tool/public/data/behavior-definitions.json", "utf8")) as {
            behaviors: Record<
                string,
                { properties: Record<string, { default?: unknown; minimum?: number; exclusiveMinimum?: boolean; optionsSource?: string }> }
            >;
        };
        expect(metadata.behaviors.weaponPistol?.properties.sound).toMatchObject({ default: "pistolShot", optionsSource: "sounds" });
        expect(metadata.behaviors.weaponPistol?.properties.bulletHoleSize).toMatchObject({ default: 1, minimum: 0, exclusiveMinimum: true });
        expect(metadata.behaviors.explode?.properties.sound).toMatchObject({ default: "bigExplosion", optionsSource: "sounds" });
    });

    it.each([
        ["the default", {}, 1],
        ["a custom factor", { bulletHoleSize: 0.5 }, 0.5],
    ])("includes %s bullet-hole size in pistol impacts", (_description, config, expectedBulletHoleSize) => {
        const events = new AquanovaEventManager();
        const impact = {
            mesh: testMesh("glass"),
            point: [1, 2, 3] as const,
            distance: 4,
            direction: [0, 0, 1] as const,
        };
        const weaponPistol = {
            setEnabled: vi.fn(),
            isReady: vi.fn(),
            fire: vi.fn(),
            update: vi.fn(() => [impact]),
            clear: vi.fn(),
        };
        const behavior = new WeaponPistolBehavior("pistol", [testMesh("pistol")], config, {
            events,
            sounds: {},
            weaponInventory: { acquire: vi.fn() },
            weaponPistol,
        } as never);
        const onHit = vi.fn();
        events.on("hitWithPistol", onHit);

        behavior.start();
        events.emit("entityEvent", { name: "pistol", event: "enable" });
        events.emit("frameEnd", { deltaMs: 16 });

        expect(onHit).toHaveBeenCalledWith({ ...impact, impulse: 10, bulletHoleSize: expectedBulletHoleSize });
        behavior.dispose();
    });

    it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY])("rejects the invalid bullet-hole size %s", (bulletHoleSize) => {
        expect(() => new WeaponPistolBehavior("pistol", [], { bulletHoleSize }, {} as never)).toThrow(
            "weaponPistol.bulletHoleSize must be a finite positive number"
        );
    });
});
