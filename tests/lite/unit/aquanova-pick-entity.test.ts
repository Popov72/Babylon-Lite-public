import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Mesh, SceneNode } from "../../../packages/babylon-lite/src";
import { EventManager } from "../../../lab/lite/src/demos/aquanova/behaviors/event-manager";
import { PickEntityBehavior } from "../../../lab/lite/src/demos/aquanova/behaviors/pick-entity";

const runtime = vi.hoisted(() => ({
    createAudioEngineAsync: vi.fn(),
    createStreamingSoundAsync: vi.fn(),
    disposeAudioEngine: vi.fn(),
    getMeshTriangles: vi.fn(),
    playStreamingSound: vi.fn(),
    preloadStreamingInstanceAsync: vi.fn(),
    setMeshVisible: vi.fn(),
}));

vi.mock("../../../packages/babylon-lite/src/index.ts", () => runtime);

const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function mesh(name: string, x = 0): Mesh {
    const worldMatrix = IDENTITY.slice();
    worldMatrix[12] = x;
    return {
        name,
        parent: null,
        rotation: {
            x: 0,
            y: 0,
            z: 0,
            set(xValue: number, yValue: number, zValue: number) {
                this.x = xValue;
                this.y = yValue;
                this.z = zValue;
            },
        },
        worldMatrix,
    } as unknown as Mesh;
}

function entityNode(name: string): SceneNode {
    return mesh(name) as unknown as SceneNode;
}

function createHarness(position = { x: 4, y: 0, z: 0 }) {
    const events = new EventManager();
    const character = {
        getPosition: vi.fn(() => position),
        shapeOptions: {
            capsuleHeight: 1.8,
            capsuleRadius: 0.4,
        },
    };
    const meshes = [mesh("pickup-a"), mesh("pickup-b")];
    const behavior = new PickEntityBehavior(
        meshes,
        {
            raiseEvent: { name: "itemLiquefactor", event: "enable" },
            sound: "click",
        },
        { character, events } as never
    );
    return { behavior, character, events, meshes, position };
}

function minimalContext(): never {
    return {
        events: new EventManager(),
        character: {
            getPosition: () => ({ x: 0, y: 0, z: 0 }),
            shapeOptions: {
                capsuleHeight: 1.8,
                capsuleRadius: 0.4,
            },
        },
    } as never;
}

describe("Aquanova pickEntity behavior", () => {
    beforeEach(() => {
        PickEntityBehavior.dispose();
        runtime.createAudioEngineAsync.mockReset().mockResolvedValue({ id: "audio-engine" });
        runtime.createStreamingSoundAsync.mockReset().mockImplementation(async (_engine: unknown, source: string) => source);
        runtime.disposeAudioEngine.mockReset();
        runtime.getMeshTriangles.mockReset().mockReturnValue({
            positions: new Float32Array([-1, -1, -1, 1, 1, 1]),
            indices: new Uint16Array([0, 1, 1]),
        });
        runtime.playStreamingSound.mockReset();
        runtime.preloadStreamingInstanceAsync.mockReset().mockResolvedValue(undefined);
        runtime.setMeshVisible.mockReset();
    });

    afterEach(() => {
        PickEntityBehavior.dispose();
        vi.restoreAllMocks();
    });

    it("preloads each configured sound once in static init", async () => {
        await PickEntityBehavior.init([{ sound: "click" }, { sound: "click" }, {}]);
        await PickEntityBehavior.init([{ sound: "ignored-after-initialization" }]);

        expect(runtime.createAudioEngineAsync).toHaveBeenCalledOnce();
        expect(runtime.createStreamingSoundAsync.mock.calls).toEqual([
            [{ id: "audio-engine" }, "/aquanova/sounds/click.mp3?v=20260813-1", { preloadCount: 1 }],
            [{ id: "audio-engine" }, "/aquanova/sounds/pickItem.mp3?v=20260813-1", { preloadCount: 1 }],
        ]);
    });

    it("collects once when the player capsule intersects the entity bounds", async () => {
        await PickEntityBehavior.init([{ sound: "click" }]);
        const harness = createHarness();
        const raised = vi.fn();
        harness.events.on("entityEvent", raised);
        harness.behavior.start();

        harness.events.emit("physicsStep", { deltaSeconds: 1 / 60 });
        expect(runtime.setMeshVisible).not.toHaveBeenCalled();
        expect(raised).not.toHaveBeenCalled();

        harness.position.x = 1.4;
        harness.events.emit("physicsStep", { deltaSeconds: 1 / 60 });
        harness.events.emit("physicsStep", { deltaSeconds: 1 / 60 });

        expect(runtime.setMeshVisible.mock.calls).toEqual([
            [harness.meshes[0], false],
            [harness.meshes[1], false],
        ]);
        expect(runtime.playStreamingSound).toHaveBeenCalledWith("/aquanova/sounds/click.mp3?v=20260813-1");
        expect(runtime.preloadStreamingInstanceAsync).toHaveBeenCalledWith("/aquanova/sounds/click.mp3?v=20260813-1");
        expect(raised).toHaveBeenCalledOnce();
        expect(raised).toHaveBeenCalledWith({ name: "itemLiquefactor", event: "enable" });
    });

    it("scales the entity bounds around their centre before testing intersection", async () => {
        await PickEntityBehavior.init([{}]);
        const events = new EventManager();
        const position = { x: 12.4, y: 0, z: 0 };
        const target = mesh("pickup", 10);
        const behavior = new PickEntityBehavior([target], { boundingBoxScale: [2, 1, 1] }, {
            events,
            character: {
                getPosition: () => position,
                shapeOptions: {
                    capsuleHeight: 1.8,
                    capsuleRadius: 0.4,
                },
            },
        } as never);
        behavior.start();

        events.emit("physicsStep", { deltaSeconds: 1 / 60 });

        expect(runtime.setMeshVisible).toHaveBeenCalledWith(target, false);
    });

    it("rotates the authored entity node around Y once every three seconds by default", async () => {
        await PickEntityBehavior.init([{}]);
        const events = new EventManager();
        const owner = entityNode("itemLiquefactor");
        const meshes = [mesh("pickup-a"), mesh("pickup-b")];
        meshes[0]!.parent = owner;
        meshes[1]!.parent = owner;
        const behavior = new PickEntityBehavior(
            meshes,
            {},
            {
                events,
                character: {
                    getPosition: () => ({ x: 10, y: 0, z: 0 }),
                    shapeOptions: {
                        capsuleHeight: 1.8,
                        capsuleRadius: 0.4,
                    },
                },
            } as never,
            "itemLiquefactor"
        );
        behavior.start();

        events.emit("physicsStep", { deltaSeconds: 1 });
        expect(owner.rotation.y).toBeCloseTo((Math.PI * 2) / 3);
        expect(meshes[0]!.rotation.y).toBe(0);
        expect(meshes[1]!.rotation.y).toBe(0);

        events.emit("physicsStep", { deltaSeconds: 2 });
        expect(owner.rotation.y).toBeCloseTo(Math.PI * 2);
    });

    it("scales the rotation rate with speed", async () => {
        await PickEntityBehavior.init([{ speed: 2 }]);
        const events = new EventManager();
        const target = mesh("pickup");
        const behavior = new PickEntityBehavior([target], { speed: 2 }, {
            events,
            character: {
                getPosition: () => ({ x: 10, y: 0, z: 0 }),
                shapeOptions: {
                    capsuleHeight: 1.8,
                    capsuleRadius: 0.4,
                },
            },
        } as never);
        behavior.start();

        events.emit("physicsStep", { deltaSeconds: 0.75 });

        expect(target.rotation.y).toBeCloseTo(Math.PI);
    });

    it("uses the default pickItem sound when sound is omitted", async () => {
        await PickEntityBehavior.init([{}]);
        const events = new EventManager();
        const target = mesh("pickup");
        const behavior = new PickEntityBehavior([target], {}, {
            events,
            character: {
                getPosition: () => ({ x: 0, y: 0, z: 0 }),
                shapeOptions: {
                    capsuleHeight: 1.8,
                    capsuleRadius: 0.4,
                },
            },
        } as never);
        behavior.start();

        events.emit("physicsStep", { deltaSeconds: 1 / 60 });

        expect(runtime.createAudioEngineAsync).toHaveBeenCalledOnce();
        expect(runtime.createStreamingSoundAsync).toHaveBeenCalledWith({ id: "audio-engine" }, "/aquanova/sounds/pickItem.mp3?v=20260813-1", { preloadCount: 1 });
        expect(runtime.setMeshVisible).toHaveBeenCalledWith(target, false);
        expect(runtime.playStreamingSound).toHaveBeenCalledWith("/aquanova/sounds/pickItem.mp3?v=20260813-1");
    });

    it("still collects the entity without playing audio when sounds are disabled", async () => {
        await PickEntityBehavior.init([{ sound: "click" }]);
        PickEntityBehavior.setSoundEnabled(false);
        const events = new EventManager();
        const target = mesh("pickup");
        const raised = vi.fn();
        events.on("entityEvent", raised);
        const behavior = new PickEntityBehavior(
            [target],
            { sound: "click", raiseEvent: { name: "itemLiquefactor", event: "enable" } },
            {
                events,
                character: {
                    getPosition: () => ({ x: 0, y: 0, z: 0 }),
                    shapeOptions: {
                        capsuleHeight: 1.8,
                        capsuleRadius: 0.4,
                    },
                },
            } as never
        );
        behavior.start();

        events.emit("physicsStep", { deltaSeconds: 1 / 60 });

        expect(runtime.setMeshVisible).toHaveBeenCalledWith(target, false);
        expect(runtime.playStreamingSound).not.toHaveBeenCalled();
        expect(raised).toHaveBeenCalledWith({ name: "itemLiquefactor", event: "enable" });
    });

    it("rejects incomplete event configuration and invalid sound paths", async () => {
        expect(() => new PickEntityBehavior([mesh("pickup")], { raiseEvent: { name: "", event: "enable" } }, minimalContext())).toThrow(
            "pickEntity.raiseEvent requires non-empty name and event values"
        );
        expect(() => new PickEntityBehavior([mesh("pickup")], { boundingBoxScale: [1, -1, 1] }, minimalContext())).toThrow(
            "pickEntity.boundingBoxScale must contain three finite non-negative values"
        );
        expect(() => new PickEntityBehavior([mesh("pickup")], { speed: 0 }, minimalContext())).toThrow("pickEntity.speed must be a finite positive value");
        await expect(PickEntityBehavior.init([{ sound: "folder/click" }])).rejects.toThrow('pickEntity sound "folder/click" must be an MP3 file name without its extension');
    });
});
