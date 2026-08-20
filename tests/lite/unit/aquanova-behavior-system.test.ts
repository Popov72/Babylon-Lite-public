import { describe, expect, it, vi } from "vitest";
import type { Mesh } from "../../../packages/babylon-lite/src";
import { createSceneNode } from "../../../packages/babylon-lite/src/scene/scene-node";
import type { Behavior } from "../../../lab/lite/src/demos/aquanova/behavior-system/behavior";
import { BehaviorManager } from "../../../lab/lite/src/demos/aquanova/behavior-system/behavior-manager";
import * as aquanovaBehaviorConstructors from "../../../lab/lite/src/demos/aquanova/behaviors/behavior-constructors";
import { AquanovaEventManager } from "../../../lab/lite/src/demos/aquanova/behaviors/aquanova-event-manager";
import type { AquanovaGameContext } from "../../../lab/lite/src/demos/aquanova/behaviors/game-context";
import { AquanovaBehaviorManager } from "../../../lab/lite/src/demos/aquanova/behaviors/aquanova-behavior-manager";

function mesh(name: string): Mesh {
    return createSceneNode(name) as Mesh;
}

describe("generic behavior system", () => {
    it("constructs every behavior before initializing and starting them", async () => {
        const steps: string[] = [];
        const meshes = [mesh("entity")];
        let resolveFirstInitialization = (): void => {};

        class FirstBehavior implements Behavior<"first"> {
            public readonly name = "first";
            public readonly mesh = meshes[0]!;

            public constructor() {
                steps.push("construct:first");
            }

            public init(): Promise<void> {
                steps.push("init:first");
                return new Promise((resolve) => {
                    resolveFirstInitialization = resolve;
                });
            }

            public start(): void {
                steps.push("start:first");
            }

            public dispose(): void {
                steps.push("dispose:first");
            }
        }

        class SecondBehavior implements Behavior<"second"> {
            public readonly name = "second";
            public readonly mesh = meshes[0]!;

            public constructor() {
                steps.push("construct:second");
            }

            public init(): void {
                steps.push("init:second");
            }

            public start(): void {
                steps.push("start:second");
            }

            public dispose(): void {
                steps.push("dispose:second");
            }
        }

        const manager = new BehaviorManager({
            library: { first: {}, second: {} },
            entities: { entity: { behaviors: [{ name: "first" }, { name: "second" }] } },
            meshesByEntityName: new Map([["entity", meshes]]),
            constructors: { FirstBehavior, SecondBehavior },
        });

        const starting = manager.start({});
        await vi.waitFor(() => expect(steps).toContain("init:second"));

        expect(steps).toEqual(["construct:first", "construct:second", "init:first", "init:second"]);
        resolveFirstInitialization();
        await starting;
        expect(steps.slice(-2)).toEqual(["start:first", "start:second"]);

        manager.dispose();
        expect(steps.slice(-2)).toEqual(["dispose:second", "dispose:first"]);
    });

    it("reports the class name required by an unknown behavior", async () => {
        const manager = new BehaviorManager({
            library: { customAction: {} },
            entities: { entity: { behaviors: [{ name: "customAction" }] } },
            meshesByEntityName: new Map([["entity", [mesh("entity")]]]),
            constructors: {},
        });

        await expect(manager.start({})).rejects.toThrow('Behavior "customAction" requires exported class "CustomActionBehavior"');
    });

    it("constructs explicitly meshless behavior owners", async () => {
        const steps: string[] = [];

        class DoorBehavior implements Behavior<"door"> {
            public readonly name = "door";
            public readonly mesh = null;

            public constructor(entityName: string, meshes: readonly Mesh[]) {
                expect(entityName).toBe("Door_D06");
                expect(meshes).toEqual([]);
                steps.push("construct");
            }

            public init(): void {
                steps.push("init");
            }

            public start(): void {
                steps.push("start");
            }

            public dispose(): void {
                steps.push("dispose");
            }
        }

        const manager = new BehaviorManager({
            library: { door: {} },
            meshlessOwners: { Door_D06: { behaviors: [{ name: "door" }] } },
            meshesByEntityName: new Map(),
            constructors: { DoorBehavior },
        });

        await manager.start({});

        expect(steps).toEqual(["construct", "init", "start"]);
        expect(manager.describeInstances(() => "unexpected")).toEqual([{ name: "door", mesh: "Door_D06" }]);
        manager.dispose();
        expect(steps.at(-1)).toBe("dispose");
    });

    it("skips and reports behavior owners without runtime meshes", async () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const manager = new BehaviorManager({
            library: { dynamic: {} },
            entities: { missing: { behaviors: [{ name: "dynamic" }] } },
            meshesByEntityName: new Map(),
            constructors: {},
        });

        await manager.start({});

        expect(manager.instances).toEqual([]);
        expect(warn).toHaveBeenCalledWith('[behavior] skipping owner "missing": no runtime meshes were found for behaviors "dynamic"');
        manager.dispose();
        warn.mockRestore();
    });

    it("constructs door behaviors declared in the entities map", async () => {
        const manager = new AquanovaBehaviorManager({
            library: { enableEntity: {} },
            entities: {
                Door_D00: {
                    behaviors: [
                        {
                            name: "enableEntity",
                            events: [{ name: "startLiquefaction", source: "storageDoorLF" }],
                        },
                    ],
                },
            },
            doors: [{ id: "Door_D00" }],
            meshesByEntityName: new Map(),
            entityNameOf: (value) => value.name,
        });

        await manager.start({} as Omit<AquanovaGameContext, "events" | "weaponInventory">);
        const entityEvents: Array<{ name: string; event: string }> = [];
        manager.events.on("entityEvent", (event) => entityEvents.push(event));
        manager.events.emit("entityEvent", { name: "storageDoorLF", event: "startLiquefaction" });

        expect(manager.instances.map(({ name }) => name)).toEqual(["enableEntity"]);
        expect(entityEvents).toContainEqual({ name: "Door_D00", event: "enable" });
        manager.dispose();
    });

    it("resolves Aquanova liquefaction aliases and handles every mesh in the entity", async () => {
        const first = mesh("first");
        const second = mesh("second");
        const events = new AquanovaEventManager();
        const liquefy = vi.fn();
        const manager = new BehaviorManager<AquanovaGameContext>({
            library: { stdLiquefaction: { liquefiable: true } },
            entities: { crate: { behaviors: [{ name: "stdLiquefaction" }] } },
            meshesByEntityName: new Map([["crate", [first, second]]]),
            constructors: aquanovaBehaviorConstructors,
        });

        await manager.start({
            events,
            isLiquefiable: () => true,
            getLiquefiableConfig: () => manager.assignmentsOf("crate")[0] as never,
            liquefy,
        } as unknown as AquanovaGameContext);
        events.emit("hitWithWeapon", { mesh: second, point: [1, 2, 3], distance: 4 });

        expect(manager.instances).toHaveLength(1);
        expect(manager.instances[0]?.name).toBe("stdLiquefaction");
        expect(liquefy).toHaveBeenCalledWith(second, [1, 2, 3], expect.objectContaining({ name: "stdLiquefaction", liquefiable: true }));

        manager.dispose();
    });

    it("copies liquefaction behavior to linked entities", async () => {
        const primary = mesh("primary");
        const linked = mesh("linked");
        const liquefy = vi.fn();
        const manager = new AquanovaBehaviorManager({
            library: { stdLiquefaction: { liquefiable: true } },
            entities: { primary: { behaviors: [{ name: "stdLiquefaction", linked: ["linked"] }] } },
            meshesByEntityName: new Map([
                ["primary", [primary]],
                ["linked", [linked]],
            ]),
            entityNameOf: (value) => value.name,
        });
        manager.classifyMeshes([primary, linked], {
            isDisabled: () => false,
            instanceIdOf: () => undefined,
        });
        const config = manager.getLiquefiableConfig(primary);

        expect(config).toBeDefined();
        expect(manager.getLiquefiableConfig(linked)).toBe(config);
        expect(manager.isLiquefiable(linked)).toBe(true);
        expect(manager.getLinkedEntityNames(primary)).toEqual(["linked"]);
        expect(manager.getLinkedEntityNames(linked)).toEqual(["primary"]);

        await manager.start({
            isLiquefiable: (value: Mesh) => manager.isLiquefiable(value),
            getLiquefiableConfig: (value: Mesh) => manager.getLiquefiableConfig(value),
            liquefy,
        } as unknown as Omit<AquanovaGameContext, "events" | "weaponInventory">);
        manager.events.emit("hitWithWeapon", { mesh: linked, point: [1, 2, 3], distance: 4 });

        expect(liquefy).toHaveBeenCalledWith(linked, [1, 2, 3], config);
        manager.dispose();
    });

    it("inherits linked liquefaction from an owner that has a runtime mesh", async () => {
        const active = mesh("active");
        const linked = mesh("linked");
        const liquefy = vi.fn();
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const manager = new AquanovaBehaviorManager({
            library: { stdLiquefaction: { liquefiable: true } },
            entities: {
                stale: { behaviors: [{ name: "stdLiquefaction", linked: ["linked"] }] },
                active: { behaviors: [{ name: "stdLiquefaction", linked: ["linked"] }] },
            },
            meshesByEntityName: new Map([
                ["active", [active]],
                ["linked", [linked]],
            ]),
            entityNameOf: (value) => value.name,
        });
        manager.classifyMeshes([active, linked], {
            isDisabled: () => false,
            instanceIdOf: () => undefined,
        });
        const activeConfig = manager.getLiquefiableConfig(active);

        expect(activeConfig).toBeDefined();
        expect(manager.getLiquefiableConfig(linked)).toBe(activeConfig);

        await manager.start({
            isLiquefiable: (value: Mesh) => manager.isLiquefiable(value),
            getLiquefiableConfig: (value: Mesh) => manager.getLiquefiableConfig(value),
            liquefy,
        } as unknown as Omit<AquanovaGameContext, "events" | "weaponInventory">);
        manager.events.emit("hitWithWeapon", { mesh: linked, point: [1, 2, 3], distance: 4 });

        expect(liquefy).toHaveBeenCalledWith(linked, [1, 2, 3], activeConfig);
        expect(warn).toHaveBeenCalledWith('[behavior] skipping owner "stale": no runtime meshes were found for behaviors "stdLiquefaction"');
        manager.dispose();
        warn.mockRestore();
    });

    it("keeps an entity's own liquefaction behavior when it is linked by another entity", () => {
        const primary = mesh("primary");
        const linked = mesh("linked");
        const manager = new AquanovaBehaviorManager({
            library: {
                quickLiquefaction: { liquefiable: true, sound: "quickSplash" },
                longLiquefaction: { liquefiable: true, sound: "longSplash" },
            },
            entities: {
                primary: { behaviors: [{ name: "quickLiquefaction", linked: ["linked"] }] },
                linked: { behaviors: [{ name: "longLiquefaction" }] },
            },
            meshesByEntityName: new Map([
                ["primary", [primary]],
                ["linked", [linked]],
            ]),
            entityNameOf: (value) => value.name,
        });

        manager.classifyMeshes([primary, linked], {
            isDisabled: () => false,
            instanceIdOf: () => undefined,
        });

        expect(manager.getLiquefiableConfig(primary)?.sound).toBe("quickSplash");
        expect(manager.getLiquefiableConfig(linked)?.sound).toBe("longSplash");
        manager.dispose();
    });

    it("rejects conflicting inherited liquefaction behaviors", () => {
        const first = mesh("first");
        const second = mesh("second");
        const shared = mesh("shared");
        const manager = new AquanovaBehaviorManager({
            library: {
                quickLiquefaction: { liquefiable: true, sound: "quickSplash" },
                longLiquefaction: { liquefiable: true, sound: "longSplash" },
            },
            entities: {
                first: { behaviors: [{ name: "quickLiquefaction", linked: ["shared"] }] },
                second: { behaviors: [{ name: "longLiquefaction", linked: ["shared"] }] },
            },
            meshesByEntityName: new Map([
                ["first", [first]],
                ["second", [second]],
                ["shared", [shared]],
            ]),
            entityNameOf: (value) => value.name,
        });

        expect(() =>
            manager.classifyMeshes([first, second, shared], {
                isDisabled: () => false,
                instanceIdOf: () => undefined,
            })
        ).toThrow('[aquanova] linked entity "shared" inherits conflicting liquefaction behaviors from "first", "second"');
        manager.dispose();
    });
});
