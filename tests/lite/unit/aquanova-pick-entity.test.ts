import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Mesh, SceneNode } from "../../../packages/babylon-lite/src";
import { AquanovaEventManager } from "../../../lab/lite/src/demos/aquanova/behaviors/aquanova-event-manager";
import { PickEntityBehavior } from "../../../lab/lite/src/demos/aquanova/behaviors/pick-entity";
import { SoundManager } from "../../../lab/lite/src/demos/aquanova/behaviors/sound-manager";
import { PLAYER_CAPSULE_HEIGHT, PLAYER_CAPSULE_RADIUS } from "../../../lab/lite/src/demos/aquanova/constants";

const runtime = vi.hoisted(() => ({
    createAudioEngineAsync: vi.fn(),
    createStreamingSoundAsync: vi.fn(),
    disposeAudioEngine: vi.fn(),
    getMeshTriangles: vi.fn(),
    playStreamingSound: vi.fn(),
    preloadStreamingInstanceAsync: vi.fn(),
    setMasterVolume: vi.fn(),
    setMeshVisible: vi.fn(),
    stopStreamingSound: vi.fn(),
}));

vi.mock("../../../packages/babylon-lite/src/index.ts", () => runtime);

const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const soundManagers: SoundManager[] = [];

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
    const events = new AquanovaEventManager();
    const sounds = createSoundManager();
    const character = {
        getPosition: vi.fn(() => position),
        shapeOptions: {
            capsuleHeight: PLAYER_CAPSULE_HEIGHT,
            capsuleRadius: PLAYER_CAPSULE_RADIUS,
        },
    };
    const meshes = [mesh("pickup-a"), mesh("pickup-b")];
    const behavior = new PickEntityBehavior(
        "pickup",
        meshes,
        {
            raiseEvent: { target: "itemLiquefactor", event: "enable" },
            sound: "click",
        },
        { character, events, sounds } as never
    );
    return { behavior, character, events, meshes, position, sounds };
}

function minimalContext(): never {
    return {
        events: new AquanovaEventManager(),
        sounds: createSoundManager(),
        character: {
            getPosition: () => ({ x: 0, y: 0, z: 0 }),
            shapeOptions: {
                capsuleHeight: PLAYER_CAPSULE_HEIGHT,
                capsuleRadius: PLAYER_CAPSULE_RADIUS,
            },
        },
    } as never;
}

function createSoundManager(): SoundManager {
    const manager = new SoundManager();
    soundManagers.push(manager);
    return manager;
}

describe("Aquanova pickEntity behavior", () => {
    beforeEach(() => {
        soundManagers.length = 0;
        runtime.createAudioEngineAsync.mockReset().mockResolvedValue({ id: "audio-engine" });
        runtime.createStreamingSoundAsync.mockReset().mockImplementation(async (_engine: unknown, source: string) => source);
        runtime.disposeAudioEngine.mockReset();
        runtime.getMeshTriangles.mockReset().mockReturnValue({
            positions: new Float32Array([-1, -1, -1, 1, 1, 1]),
            indices: new Uint16Array([0, 1, 1]),
        });
        runtime.playStreamingSound.mockReset();
        runtime.preloadStreamingInstanceAsync.mockReset().mockResolvedValue(undefined);
        runtime.setMasterVolume.mockReset();
        runtime.setMeshVisible.mockReset();
        runtime.stopStreamingSound.mockReset();
    });

    afterEach(() => {
        for (const manager of soundManagers.splice(0)) {
            manager.dispose();
        }
        vi.restoreAllMocks();
    });

    it("preloads each configured sound once as behaviors initialize", async () => {
        const context = minimalContext();
        const behaviors = [
            new PickEntityBehavior("first", [mesh("first")], { sound: "click" }, context),
            new PickEntityBehavior("second", [mesh("second")], { sound: "click" }, context),
            new PickEntityBehavior("third", [mesh("third")], {}, context),
            new PickEntityBehavior("fourth", [mesh("fourth")], { sound: "ignored-after-initialization" }, context),
        ];

        await Promise.all(behaviors.map((behavior) => behavior.init()));

        expect(runtime.createAudioEngineAsync).toHaveBeenCalledOnce();
        expect(runtime.setMasterVolume).toHaveBeenCalledWith({ id: "audio-engine" }, 1);
        expect(runtime.createStreamingSoundAsync.mock.calls).toEqual([
            [{ id: "audio-engine" }, "/aquanova/sounds/click.mp3?v=20260813-1", { preloadCount: 1 }],
            [{ id: "audio-engine" }, "/aquanova/sounds/pickItem.mp3?v=20260813-1", { preloadCount: 1 }],
            [{ id: "audio-engine" }, "/aquanova/sounds/ignored-after-initialization.mp3?v=20260813-1", { preloadCount: 1 }],
        ]);
    });

    it("applies the global sound volume before and after audio initialization", async () => {
        const sounds = createSoundManager();
        sounds.setVolume(0.35);
        await sounds.load("click", "/aquanova/sounds/click.mp3?v=20260813-1");
        expect(runtime.setMasterVolume).toHaveBeenLastCalledWith({ id: "audio-engine" }, 0.35);

        sounds.setVolume(0.7);
        expect(runtime.setMasterVolume).toHaveBeenLastCalledWith({ id: "audio-engine" }, 0.7);
    });

    it("collects once when the player capsule intersects the entity bounds", async () => {
        const harness = createHarness();
        await harness.behavior.init();
        const raised = vi.fn();
        harness.events.on("entityEvent", raised);
        harness.behavior.start();

        harness.events.emit("physicsStep", { deltaSeconds: 1 / 60 });
        expect(runtime.setMeshVisible).not.toHaveBeenCalled();
        expect(raised).not.toHaveBeenCalled();

        harness.position.x = 1.3;
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

    it("raises events on the owning entity when no target override is provided", async () => {
        const events = new AquanovaEventManager();
        const sounds = createSoundManager();
        const raised = vi.fn();
        events.on("entityEvent", raised);
        const behavior = new PickEntityBehavior("pickupOwner", [mesh("pickup")], { sound: "click", raiseEvent: { event: "enable" } }, {
            events,
            character: {
                getPosition: () => ({ x: 0, y: 0, z: 0 }),
                shapeOptions: {
                    capsuleHeight: PLAYER_CAPSULE_HEIGHT,
                    capsuleRadius: PLAYER_CAPSULE_RADIUS,
                },
            },
            sounds,
        } as never);
        await behavior.init();
        behavior.start();

        events.emit("physicsStep", { deltaSeconds: 1 / 60 });

        expect(raised).toHaveBeenCalledWith({ name: "pickupOwner", event: "enable" });
    });

    it("scales the entity bounds around their centre before testing intersection", async () => {
        const events = new AquanovaEventManager();
        const sounds = createSoundManager();
        const position = { x: 12.3, y: 0, z: 0 };
        const target = mesh("pickup", 10);
        const behavior = new PickEntityBehavior("pickup", [target], { boundingBoxScale: [2, 1, 1] }, {
            events,
            character: {
                getPosition: () => position,
                shapeOptions: {
                    capsuleHeight: PLAYER_CAPSULE_HEIGHT,
                    capsuleRadius: PLAYER_CAPSULE_RADIUS,
                },
            },
            sounds,
        } as never);
        await behavior.init();
        behavior.start();

        events.emit("physicsStep", { deltaSeconds: 1 / 60 });

        expect(runtime.setMeshVisible).toHaveBeenCalledWith(target, false);
    });

    it("rotates the authored entity node around Y once every three seconds by default", async () => {
        const events = new AquanovaEventManager();
        const sounds = createSoundManager();
        const owner = entityNode("itemLiquefactor");
        const meshes = [mesh("pickup-a"), mesh("pickup-b")];
        meshes[0]!.parent = owner;
        meshes[1]!.parent = owner;
        const behavior = new PickEntityBehavior("itemLiquefactor", meshes, {}, {
            events,
            character: {
                getPosition: () => ({ x: 10, y: 0, z: 0 }),
                shapeOptions: {
                    capsuleHeight: PLAYER_CAPSULE_HEIGHT,
                    capsuleRadius: PLAYER_CAPSULE_RADIUS,
                },
            },
            sounds,
        } as never);
        await behavior.init();
        behavior.start();

        events.emit("physicsStep", { deltaSeconds: 1 });
        expect(owner.rotation.y).toBeCloseTo((Math.PI * 2) / 3);
        expect(meshes[0]!.rotation.y).toBe(0);
        expect(meshes[1]!.rotation.y).toBe(0);

        events.emit("physicsStep", { deltaSeconds: 2 });
        expect(owner.rotation.y).toBeCloseTo(Math.PI * 2);
    });

    it("scales the rotation rate with speed", async () => {
        const events = new AquanovaEventManager();
        const sounds = createSoundManager();
        const target = mesh("pickup");
        const behavior = new PickEntityBehavior("pickup", [target], { speed: 2 }, {
            events,
            character: {
                getPosition: () => ({ x: 10, y: 0, z: 0 }),
                shapeOptions: {
                    capsuleHeight: PLAYER_CAPSULE_HEIGHT,
                    capsuleRadius: PLAYER_CAPSULE_RADIUS,
                },
            },
            sounds,
        } as never);
        await behavior.init();
        behavior.start();

        events.emit("physicsStep", { deltaSeconds: 0.75 });

        expect(target.rotation.y).toBeCloseTo(Math.PI);
    });

    it("uses the default pickItem sound when sound is omitted", async () => {
        const events = new AquanovaEventManager();
        const sounds = createSoundManager();
        const target = mesh("pickup");
        const behavior = new PickEntityBehavior("pickup", [target], {}, {
            events,
            character: {
                getPosition: () => ({ x: 0, y: 0, z: 0 }),
                shapeOptions: {
                    capsuleHeight: PLAYER_CAPSULE_HEIGHT,
                    capsuleRadius: PLAYER_CAPSULE_RADIUS,
                },
            },
            sounds,
        } as never);
        await behavior.init();
        behavior.start();

        events.emit("physicsStep", { deltaSeconds: 1 / 60 });

        expect(runtime.createAudioEngineAsync).toHaveBeenCalledOnce();
        expect(runtime.createStreamingSoundAsync).toHaveBeenCalledWith({ id: "audio-engine" }, "/aquanova/sounds/pickItem.mp3?v=20260813-1", { preloadCount: 1 });
        expect(runtime.setMeshVisible).toHaveBeenCalledWith(target, false);
        expect(runtime.playStreamingSound).toHaveBeenCalledWith("/aquanova/sounds/pickItem.mp3?v=20260813-1");
    });
    it("still collects the entity without playing audio when sounds are disabled", async () => {
        const events = new AquanovaEventManager();
        const sounds = createSoundManager();
        const target = mesh("pickup");
        const raised = vi.fn();
        events.on("entityEvent", raised);
        const behavior = new PickEntityBehavior("pickup", [target], { sound: "click", raiseEvent: { target: "itemLiquefactor", event: "enable" } }, {
            events,
            character: {
                getPosition: () => ({ x: 0, y: 0, z: 0 }),
                shapeOptions: {
                    capsuleHeight: PLAYER_CAPSULE_HEIGHT,
                    capsuleRadius: PLAYER_CAPSULE_RADIUS,
                },
            },
            sounds,
        } as never);
        await behavior.init();
        sounds.setEnabled(false);
        behavior.start();

        events.emit("physicsStep", { deltaSeconds: 1 / 60 });

        expect(runtime.setMeshVisible).toHaveBeenCalledWith(target, false);
        expect(runtime.playStreamingSound).not.toHaveBeenCalled();
        expect(raised).toHaveBeenCalledWith({ name: "itemLiquefactor", event: "enable" });
    });

    it("rejects incomplete event configuration and invalid sound paths", () => {
        expect(() => new PickEntityBehavior("pickup", [mesh("pickup")], { raiseEvent: { target: "", event: "enable" } }, minimalContext())).toThrow(
            "pickEntity.raiseEvent.target must be a non-empty entity or door name when provided"
        );
        expect(() => new PickEntityBehavior("pickup", [mesh("pickup")], { raiseEvent: { unexpected: "target", event: "enable" } } as never, minimalContext())).toThrow(
            "pickEntity.raiseEvent.unexpected is not supported"
        );
        expect(() => new PickEntityBehavior("pickup", [mesh("pickup")], { boundingBoxScale: [1, -1, 1] }, minimalContext())).toThrow(
            "pickEntity.boundingBoxScale must contain three finite non-negative values"
        );
        expect(() => new PickEntityBehavior("pickup", [mesh("pickup")], { speed: 0 }, minimalContext())).toThrow("pickEntity.speed must be a finite positive value");
        expect(() => new PickEntityBehavior("pickup", [mesh("pickup")], { sound: "folder/click" }, minimalContext())).toThrow(
            'pickEntity sound "folder/click" must be an MP3 file name without its extension'
        );
    });
});
