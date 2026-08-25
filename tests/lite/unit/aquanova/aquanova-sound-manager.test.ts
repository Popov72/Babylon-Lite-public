import { beforeEach, describe, expect, it, vi } from "vitest";
import { SoundManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/sound-manager";

const audio = vi.hoisted(() => ({
    createAudioEngineAsync: vi.fn(),
    createStreamingSoundAsync: vi.fn(),
    disposeAudioEngine: vi.fn(),
    playStreamingSound: vi.fn(),
    preloadStreamingInstanceAsync: vi.fn(),
    setMasterVolume: vi.fn(),
    setStreamingSoundVolume: vi.fn(),
    stopStreamingSound: vi.fn(),
}));

vi.mock("../../../../packages/babylon-lite/src/index.ts", () => audio);

describe("Aquanova sound manager", () => {
    beforeEach(() => {
        audio.createAudioEngineAsync.mockReset().mockResolvedValue({ id: "audio-engine" });
        audio.createStreamingSoundAsync.mockReset().mockImplementation(async (_engine: unknown, source: string) => source);
        audio.disposeAudioEngine.mockReset();
        audio.playStreamingSound.mockReset();
        audio.preloadStreamingInstanceAsync.mockReset().mockResolvedValue(undefined);
        audio.setMasterVolume.mockReset();
        audio.setStreamingSoundVolume.mockReset();
        audio.stopStreamingSound.mockReset();
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
});
