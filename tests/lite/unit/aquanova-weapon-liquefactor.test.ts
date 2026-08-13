import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Mesh } from "../../../packages/babylon-lite/src";
import { createSceneNode } from "../../../packages/babylon-lite/src/scene/scene-node";
import { EventManager } from "../../../lab/lite/src/demos/aquanova/behaviors/event-manager";
import type { BehaviorContext, WeaponLiquefactorBehaviorConfig, WeaponLiquefactorRuntime } from "../../../lab/lite/src/demos/aquanova/behaviors/types";
import { WeaponLiquefactorBehavior } from "../../../lab/lite/src/demos/aquanova/behaviors/weapon-liquefactor";

const audio = vi.hoisted(() => ({
    createAudioEngineAsync: vi.fn(),
    createStreamingSoundAsync: vi.fn(),
    disposeAudioEngine: vi.fn(),
    playStreamingSound: vi.fn(),
    preloadStreamingInstanceAsync: vi.fn(),
    stopStreamingSound: vi.fn(),
}));

vi.mock("../../../packages/babylon-lite/src/index.ts", () => audio);

type WeaponContext = Pick<
    BehaviorContext,
    "events" | "weaponLiquefactor" | "requestFusionResume" | "resolveFusionResume" | "resolveFusionTarget" | "fusionTargetLost" | "reverseFusion"
>;

function mesh(name: string): Mesh {
    return createSceneNode(name) as Mesh;
}

function createHarness(
    options: {
        resumeToken?: number | null;
        resumeResult?: "resumed" | "start-new" | "await-target" | "continue";
        config?: WeaponLiquefactorBehaviorConfig;
    } = {}
) {
    const events = new EventManager();
    let reachedTarget = false;
    const runtime: WeaponLiquefactorRuntime = {
        setTargetDistance: vi.fn(),
        stop: vi.fn(),
        update: vi.fn(() => reachedTarget),
    };
    const context: WeaponContext = {
        events,
        weaponLiquefactor: runtime,
        requestFusionResume: vi.fn(() => options.resumeToken ?? null),
        resolveFusionResume: vi.fn(() => options.resumeResult ?? "start-new"),
        resolveFusionTarget: vi.fn((mesh) => mesh),
        fusionTargetLost: vi.fn(() => false),
        reverseFusion: vi.fn(),
    };
    const behavior = new WeaponLiquefactorBehavior(mesh("weapon"), options.config ?? {}, context);
    behavior.start();
    return {
        behavior,
        context,
        events,
        runtime,
        setReachedTarget: (reached: boolean): void => {
            reachedTarget = reached;
        },
    };
}

describe("Aquanova Liquefactor weapon behavior", () => {
    beforeEach(() => {
        WeaponLiquefactorBehavior.dispose();
        audio.createAudioEngineAsync.mockReset().mockResolvedValue({ id: "audio-engine" });
        audio.createStreamingSoundAsync.mockReset().mockImplementation(async (_engine: unknown, source: string) => source);
        audio.disposeAudioEngine.mockReset();
        audio.playStreamingSound.mockReset();
        audio.preloadStreamingInstanceAsync.mockReset().mockResolvedValue(undefined);
        audio.stopStreamingSound.mockReset();
    });

    afterEach(() => {
        WeaponLiquefactorBehavior.dispose();
        vi.restoreAllMocks();
    });

    it("statically preloads shared and manifest sounds once, then randomly plays the requested splash category", async () => {
        const config = {
            sounds: {
                quickSplash: ["quick-a", "quick-b"],
                bigSplash: ["big"],
            },
        };
        await WeaponLiquefactorBehavior.init(config);
        await WeaponLiquefactorBehavior.init(config);
        const harness = createHarness({
            config,
        });
        vi.spyOn(Math, "random").mockReturnValue(0.75);

        harness.events.emit("liquefactionCompleted", { sound: "quickSplash" });

        expect(audio.createAudioEngineAsync).toHaveBeenCalledOnce();
        expect(audio.createStreamingSoundAsync.mock.calls.map(([, source]) => source).sort()).toEqual([
            "/aquanova/sounds/big.mp3?v=20260813-1",
            "/aquanova/sounds/liquefactorLiquefy.mp3?v=20260813-1",
            "/aquanova/sounds/liquefactorStartShot.mp3?v=20260813-1",
            "/aquanova/sounds/quick-a.mp3?v=20260813-1",
            "/aquanova/sounds/quick-b.mp3?v=20260813-1",
        ]);
        expect(audio.playStreamingSound).toHaveBeenCalledWith("/aquanova/sounds/quick-b.mp3?v=20260813-1");
        expect(audio.preloadStreamingInstanceAsync).toHaveBeenCalledWith("/aquanova/sounds/quick-b.mp3?v=20260813-1");

        harness.behavior.dispose();
        expect(audio.disposeAudioEngine).not.toHaveBeenCalled();
        WeaponLiquefactorBehavior.dispose();
        expect(audio.disposeAudioEngine).toHaveBeenCalledWith({ id: "audio-engine" });
    });

    it("stops the shot sound when liquefaction starts and stops the liquefaction loop before the splash", async () => {
        await WeaponLiquefactorBehavior.init({
            sounds: {
                quickSplash: ["quick"],
            },
        });
        const harness = createHarness();
        audio.playStreamingSound.mockClear();
        audio.stopStreamingSound.mockClear();
        audio.preloadStreamingInstanceAsync.mockClear();

        harness.events.emit("weaponTriggerPressed", { held: true });
        expect(audio.playStreamingSound).toHaveBeenLastCalledWith("/aquanova/sounds/liquefactorStartShot.mp3?v=20260813-1");

        audio.playStreamingSound.mockClear();
        audio.stopStreamingSound.mockClear();
        harness.events.emit("liquefactionStarted", {});
        expect(audio.stopStreamingSound).toHaveBeenCalledWith("/aquanova/sounds/liquefactorStartShot.mp3?v=20260813-1");
        expect(audio.playStreamingSound).toHaveBeenLastCalledWith("/aquanova/sounds/liquefactorLiquefy.mp3?v=20260813-1", { loop: true });

        audio.playStreamingSound.mockClear();
        audio.stopStreamingSound.mockClear();
        harness.events.emit("liquefactionCompleted", { sound: "quickSplash" });
        expect(audio.stopStreamingSound).toHaveBeenCalledWith("/aquanova/sounds/liquefactorLiquefy.mp3?v=20260813-1");
        expect(audio.playStreamingSound).toHaveBeenLastCalledWith("/aquanova/sounds/quick.mp3?v=20260813-1");
    });

    it("identifies the exact static sound file when preloading fails", async () => {
        const decodeError = new DOMException("Unable to decode audio data", "EncodingError");
        audio.createStreamingSoundAsync.mockImplementation(async (_engine: unknown, source: string) => {
            if (source.includes("broken-splash")) throw decodeError;
            return source;
        });

        await expect(
            WeaponLiquefactorBehavior.init({
                sounds: {
                    quickSplash: ["broken-splash"],
                },
            })
        ).rejects.toThrow(
            '[aquanova] failed to preload Liquefactor sound "broken-splash" from "/aquanova/sounds/broken-splash.mp3?v=20260813-1"'
        );
        expect(audio.disposeAudioEngine).toHaveBeenCalledWith({ id: "audio-engine" });
    });

    it("delivers one hit only after the laser reaches its target", () => {
        const harness = createHarness();
        const target = mesh("target");
        const hits: Mesh[] = [];
        harness.events.on("hitWithWeapon", ({ mesh: hitMesh }) => hits.push(hitMesh));

        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: target, point: [1, 2, 3], distance: 10 });
        harness.events.emit("frameEnd", { deltaMs: 16 });
        expect(hits).toEqual([]);

        harness.setReachedTarget(true);
        harness.events.emit("frameEnd", { deltaMs: 16 });
        harness.events.emit("frameEnd", { deltaMs: 16 });
        expect(hits).toEqual([target]);
    });

    it("uses the logical active-fusion target when clipping exposes geometry behind it", () => {
        const behind = mesh("behind");
        const activeTarget = mesh("active");
        const harness = createHarness();
        const hits: Mesh[] = [];
        vi.mocked(harness.context.resolveFusionTarget).mockReturnValue(activeTarget);
        harness.events.on("hitWithWeapon", ({ mesh: hitMesh }) => hits.push(hitMesh));

        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: behind, point: [1, 2, 3], distance: 5 });
        harness.setReachedTarget(true);
        harness.events.emit("frameEnd", { deltaMs: 16 });

        expect(hits).toEqual([activeTarget]);
    });

    it("cancels a pending hit when the trigger is released", () => {
        const harness = createHarness();
        const hits = vi.fn();
        harness.events.on("hitWithWeapon", hits);

        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: mesh("target"), point: null, distance: 5 });
        harness.events.emit("weaponTriggerReleased", {});
        harness.setReachedTarget(true);
        harness.events.emit("frameEnd", { deltaMs: 16 });

        expect(hits).not.toHaveBeenCalled();
        expect(harness.context.reverseFusion).toHaveBeenCalledOnce();
        expect(harness.runtime.stop).toHaveBeenCalled();
    });

    it("cancels a one-shot trigger when no target is acquired", () => {
        const harness = createHarness({ resumeToken: 7 });

        harness.events.emit("weaponTriggerPressed", { held: false });
        harness.events.emit("weaponAimUpdated", { mesh: null, point: null, distance: null });

        expect(harness.context.reverseFusion).toHaveBeenCalledOnce();
        expect(harness.runtime.stop).toHaveBeenCalled();
    });

    it("keeps a held laser alive at maximum range when no geometry is hit", () => {
        const harness = createHarness();
        harness.events.emit("weaponTriggerPressed", { held: true });
        vi.mocked(harness.runtime.stop).mockClear();

        harness.events.emit("weaponAimUpdated", { mesh: null, point: null, distance: null });

        expect(harness.runtime.setTargetDistance).toHaveBeenLastCalledWith(100, true);
        expect(harness.runtime.stop).not.toHaveBeenCalled();
    });

    it("resumes reversing fusion on arrival without starting a second hit", () => {
        const harness = createHarness({ resumeToken: 7, resumeResult: "resumed" });
        const target = mesh("target");
        const hits = vi.fn();
        harness.events.on("hitWithWeapon", hits);

        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: target, point: null, distance: 4 });
        harness.setReachedTarget(true);
        harness.events.emit("frameEnd", { deltaMs: 16 });

        expect(harness.context.resolveFusionResume).toHaveBeenCalledWith(7, target);
        expect(hits).not.toHaveBeenCalled();
    });

    it("starts a fresh fusion when the reversing group is gone before re-entry", () => {
        const target = mesh("target");
        const harness = createHarness({ resumeToken: 7, resumeResult: "continue" });
        const hits: Mesh[] = [];
        harness.events.on("hitWithWeapon", ({ mesh: hitMesh }) => hits.push(hitMesh));

        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: target, point: null, distance: 4 });
        harness.setReachedTarget(true);
        harness.events.emit("frameEnd", { deltaMs: 16 });

        expect(hits).toEqual([target]);
    });

    it("keeps firing and delivers a new hit when held aim moves onto another mesh", () => {
        const outside = mesh("outside");
        const liquefiable = mesh("liquefiable");
        const harness = createHarness();
        const hits: Mesh[] = [];
        harness.events.on("hitWithWeapon", ({ mesh: hitMesh }) => hits.push(hitMesh));
        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: outside, point: null, distance: 3 });
        harness.setReachedTarget(true);
        harness.events.emit("frameEnd", { deltaMs: 16 });
        vi.mocked(harness.runtime.stop).mockClear();
        harness.events.emit("weaponAimUpdated", { mesh: liquefiable, point: [1, 2, 3], distance: 4 });
        harness.events.emit("frameEnd", { deltaMs: 16 });

        expect(hits).toEqual([outside, liquefiable]);
        expect(harness.context.reverseFusion).not.toHaveBeenCalled();
        expect(harness.runtime.stop).not.toHaveBeenCalled();
    });

    it("reverses fusion without stopping when held aim leaves a liquefiable mesh", () => {
        const liquefiable = mesh("liquefiable");
        const outside = mesh("outside");
        const harness = createHarness();
        vi.mocked(harness.context.requestFusionResume).mockReturnValueOnce(null).mockReturnValueOnce(7);
        vi.mocked(harness.context.resolveFusionResume).mockReturnValueOnce("await-target").mockReturnValueOnce("resumed");
        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: liquefiable, point: null, distance: 3 });
        harness.setReachedTarget(true);
        harness.events.emit("frameEnd", { deltaMs: 16 });
        vi.mocked(harness.runtime.stop).mockClear();
        vi.mocked(harness.context.reverseFusion).mockClear();
        vi.mocked(harness.context.fusionTargetLost).mockReturnValueOnce(true);

        harness.events.emit("weaponAimUpdated", { mesh: outside, point: null, distance: 4 });
        harness.events.emit("frameEnd", { deltaMs: 16 });

        expect(harness.context.reverseFusion).toHaveBeenCalledOnce();
        expect(harness.runtime.stop).not.toHaveBeenCalled();

        harness.events.emit("weaponAimUpdated", { mesh: liquefiable, point: null, distance: 3 });
        harness.events.emit("frameEnd", { deltaMs: 16 });
        expect(harness.context.resolveFusionResume).toHaveBeenCalledWith(7, liquefiable);
    });

    it("retains a pending fusion-resume token while the beam is outside liquefiable geometry", () => {
        const outside = mesh("outside");
        const liquefiable = mesh("liquefiable");
        const harness = createHarness({ resumeToken: 7 });
        vi.mocked(harness.context.resolveFusionResume).mockReturnValueOnce("await-target").mockReturnValueOnce("resumed");
        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: outside, point: null, distance: 3 });
        harness.setReachedTarget(true);
        harness.events.emit("frameEnd", { deltaMs: 16 });
        expect(harness.context.resolveFusionResume).toHaveBeenCalledWith(7, outside);

        harness.events.emit("weaponAimUpdated", { mesh: liquefiable, point: null, distance: 4 });
        harness.events.emit("frameEnd", { deltaMs: 16 });

        expect(harness.context.resolveFusionResume).toHaveBeenCalledWith(7, liquefiable);
    });

    it("keeps the laser alive after liquefaction completes while the trigger is held", () => {
        const harness = createHarness();
        harness.events.emit("weaponTriggerPressed", { held: true });
        harness.events.emit("weaponAimUpdated", { mesh: mesh("target"), point: null, distance: 3 });
        vi.mocked(harness.runtime.stop).mockClear();

        harness.events.emit("liquefactionCompleted", { sound: "quickSplash" });

        expect(harness.runtime.stop).not.toHaveBeenCalled();

        harness.events.emit("weaponAimUpdated", { mesh: null, point: null, distance: null });
        expect(harness.runtime.setTargetDistance).toHaveBeenLastCalledWith(3, false);

        harness.events.emit("weaponTriggerReleased", {});
        expect(harness.runtime.stop).toHaveBeenCalledOnce();
    });
});
