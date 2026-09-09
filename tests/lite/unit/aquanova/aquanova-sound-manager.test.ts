import { beforeEach, describe, expect, it, vi } from "vitest";
import { SoundManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/sound-manager";
import { mat4Identity } from "../../../../packages/babylon-lite/src/math/mat4-identity";

const audio = vi.hoisted(() => ({
    attachSpatialTarget: vi.fn(),
    createAudioEngineAsync: vi.fn(),
    createStreamingSoundAsync: vi.fn(),
    detachSpatialTarget: vi.fn(),
    disposeAudioEngine: vi.fn(),
    enableSpatial: vi.fn(),
    playStreamingSound: vi.fn(),
    preloadStreamingInstanceAsync: vi.fn(),
    setMasterVolume: vi.fn(),
    setStreamingSoundVolume: vi.fn(),
    stopStreamingSound: vi.fn(),
    updateSpatialAudio: vi.fn(),
}));

vi.mock("../../../../packages/babylon-lite/src/index.ts", () => audio);

describe("Aquanova sound manager", () => {
    beforeEach(() => {
        audio.attachSpatialTarget.mockReset();
        audio.createAudioEngineAsync.mockReset().mockResolvedValue({ id: "audio-engine" });
        audio.createStreamingSoundAsync.mockReset().mockImplementation(async (_engine: unknown, source: string) => source);
        audio.detachSpatialTarget.mockReset();
        audio.disposeAudioEngine.mockReset();
        audio.enableSpatial.mockReset();
        audio.playStreamingSound.mockReset();
        audio.preloadStreamingInstanceAsync.mockReset().mockResolvedValue(undefined);
        audio.setMasterVolume.mockReset();
        audio.setStreamingSoundVolume.mockReset();
        audio.stopStreamingSound.mockReset();
        audio.updateSpatialAudio.mockReset();
    });

    it("owns one lazy engine, deduplicates loads, and applies global controls", async () => {
        const manager = new SoundManager();
        manager.setVolume(0.4);

        const [first, second] = await Promise.all([
            manager.load("first", "/sounds/shared.mp3", { preloadCount: 1 }),
            manager.load("second", "/sounds/shared.mp3", { preloadCount: 1 }),
        ]);

        expect(first).toBe(second);
        expect(audio.createAudioEngineAsync).toHaveBeenCalledOnce();
        expect(audio.createStreamingSoundAsync).toHaveBeenCalledOnce();
        expect(audio.setMasterVolume).toHaveBeenCalledWith({ id: "audio-engine" }, 0.4);

        manager.play(first, { loop: true });
        manager.setEnabled(false);
        expect(audio.stopStreamingSound).toHaveBeenCalledWith("/sounds/shared.mp3");

        audio.playStreamingSound.mockClear();
        manager.play(first);
        expect(audio.playStreamingSound).not.toHaveBeenCalled();

        manager.setEnabled(true);
        manager.play(first);
        expect(audio.playStreamingSound).toHaveBeenCalledWith("/sounds/shared.mp3");
        expect(audio.preloadStreamingInstanceAsync).toHaveBeenCalledWith("/sounds/shared.mp3");

        manager.dispose();
        expect(audio.disposeAudioEngine).toHaveBeenCalledWith({ id: "audio-engine" });
    });

    it("fades playback in and delays stopping until a fade-out completes", async () => {
        vi.useFakeTimers();
        try {
            const manager = new SoundManager();
            const sound = await manager.load("ambient", "/sounds/ambient.mp3");

            manager.play(sound, { fade: 2 });
            expect(audio.setStreamingSoundVolume.mock.calls.slice(-2)).toEqual([
                ["/sounds/ambient.mp3", 0, { shape: "none" }],
                ["/sounds/ambient.mp3", 1, { duration: 2, shape: "linear" }],
            ]);

            manager.stop(sound, 3);
            expect(audio.setStreamingSoundVolume).toHaveBeenLastCalledWith("/sounds/ambient.mp3", 0, { duration: 3, shape: "linear" });
            expect(audio.stopStreamingSound).not.toHaveBeenCalled();

            vi.advanceTimersByTime(2999);
            expect(audio.stopStreamingSound).not.toHaveBeenCalled();
            vi.advanceTimersByTime(1);
            expect(audio.stopStreamingSound).toHaveBeenCalledWith("/sounds/ambient.mp3");
            expect(audio.setStreamingSoundVolume).toHaveBeenLastCalledWith("/sounds/ambient.mp3", 1, { shape: "none" });
            manager.dispose();
        } finally {
            vi.useRealTimers();
        }
    });

    it("keeps playback IDs independent even when they use the same source", async () => {
        const manager = new SoundManager();
        manager.registerPlayback("first-loop", "/sounds/shared.mp3", { preloadCount: 1 });
        manager.registerPlayback("second-loop", "/sounds/shared.mp3", { preloadCount: 1 });

        const [first, second] = await Promise.all([manager.resolvePlayback("first-loop"), manager.resolvePlayback("second-loop")]);

        expect(first).not.toBe(second);
        expect(audio.createStreamingSoundAsync).toHaveBeenCalledTimes(2);
        expect(() => manager.registerPlayback("first-loop", "/sounds/other.mp3")).toThrow('sound playback ID "first-loop" is defined more than once');
        expect(() => manager.resolvePlayback("missing")).toThrow('sound playback ID "missing" is not defined');
        manager.dispose();
    });

    it("configures linear distance attenuation independently from playback volume", () => {
        const manager = new SoundManager();
        const engine = { id: "audio-engine" };
        const runtimeSound = { _engine: engine };
        const sound = { label: "alarm", source: "/sounds/alarm.mp3", sound: runtimeSound } as never;
        const source = { worldMatrix: mat4Identity() };
        const listener = { worldMatrix: mat4Identity() };

        manager.enableDistanceAttenuation(sound, source, listener, 12);

        expect(audio.enableSpatial).toHaveBeenCalledWith(runtimeSound, {
            attachedTo: source,
            attachmentType: "position",
            panningEnabled: false,
            distanceModel: "linear",
            minDistance: 1e-6,
            maxDistance: 12,
            rolloffFactor: 1,
        });
        expect(audio.attachSpatialTarget).toHaveBeenCalledWith(engine, listener, "position");
        manager.updateSpatial();
        expect(audio.updateSpatialAudio).toHaveBeenCalledWith(engine);

        manager.disableDistanceAttenuation(sound);
        expect(audio.detachSpatialTarget).toHaveBeenNthCalledWith(1, runtimeSound);
        expect(audio.detachSpatialTarget).toHaveBeenNthCalledWith(2, engine);
    });
});
