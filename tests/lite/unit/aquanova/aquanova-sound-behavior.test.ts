import { afterEach, describe, expect, it, vi } from "vitest";
import { AquanovaEventManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/aquanova-event-manager";
import { SoundBehavior } from "../../../../lab/lite/src/demos/aquanova/behaviors/sound";

function harness(config: ConstructorParameters<typeof SoundBehavior>[2]) {
    const events = new AquanovaEventManager();
    const soundById = {
        alarmLoop: { label: "alarm", source: "/aquanova/sounds/alarm.mp3", sound: "runtime-alarm" },
        chimeOnce: { label: "chime", source: "/aquanova/sounds/chime.mp3", sound: "runtime-chime" },
        externalAlarm: { label: "external-alarm", source: "/aquanova/sounds/alarm.mp3", sound: "runtime-external-alarm" },
    };
    const sounds = {
        registerPlayback: vi.fn(),
        resolvePlayback: vi.fn(async (id: string) => soundById[id as keyof typeof soundById]),
        play: vi.fn(),
        stop: vi.fn(),
    };
    const behavior = new SoundBehavior("speaker", [], config, { events, sounds } as never);
    return { behavior, events, soundById, sounds };
}

describe("Aquanova sound behavior", () => {
    afterEach(() => {
        vi.useRealTimers();
    });

    it("registers independent playback IDs and runs cues without subscriptions at startup", async () => {
        const { behavior, soundById, sounds } = harness({
            cues: [
                { action: "play", id: "alarmLoop", sound: "alarm", fade: 0.75, volume: 0.4, loop: true },
                { action: "stop", soundId: "alarmLoop", events: [{ source: "panel", name: "deactivated" }] },
                { action: "play", id: "chimeOnce", sound: "chime" },
            ],
        });

        await behavior.init();
        behavior.start();

        expect(sounds.registerPlayback).toHaveBeenNthCalledWith(1, "alarmLoop", "/aquanova/sounds/alarm.mp3?v=20260813-1", { preloadCount: 1 });
        expect(sounds.registerPlayback).toHaveBeenNthCalledWith(2, "chimeOnce", "/aquanova/sounds/chime.mp3?v=20260813-1", { preloadCount: 1 });
        expect(sounds.resolvePlayback).toHaveBeenCalledTimes(2);
        expect(sounds.play).toHaveBeenNthCalledWith(1, soundById.alarmLoop, { fade: 0.75, loop: true, volume: 0.4 });
        expect(sounds.play).toHaveBeenNthCalledWith(2, soundById.chimeOnce, { fade: 0, loop: false, volume: 1 });
        behavior.dispose();
        expect(sounds.stop).toHaveBeenCalledWith(soundById.alarmLoop);
        expect(sounds.stop).toHaveBeenCalledWith(soundById.chimeOnce);
    });

    it("runs every matching cue independently and cancels pending actions on disposal", async () => {
        vi.useFakeTimers();
        const { behavior, events, soundById, sounds } = harness({
            cues: [
                {
                    action: "stop",
                    soundId: "externalAlarm",
                    events: [{ source: ["panel-a", "panel-b"], name: "deactivated" }],
                    delay: 1.5,
                    fade: 2,
                },
                {
                    action: "play",
                    id: "chimeOnce",
                    sound: "chime",
                    events: [{ source: "panel-b", name: "deactivated" }],
                    fade: 0.25,
                },
            ],
        });

        await behavior.init();
        behavior.start();
        events.emit("entityEvent", { name: "other", event: "deactivated" });
        expect(sounds.stop).not.toHaveBeenCalled();

        events.emit("entityEvent", { name: "panel-b", event: "deactivated" });
        expect(sounds.play).toHaveBeenCalledWith(soundById.chimeOnce, { fade: 0.25, loop: false, volume: 1 });
        vi.advanceTimersByTime(1499);
        expect(sounds.stop).not.toHaveBeenCalled();
        vi.advanceTimersByTime(1);
        expect(sounds.stop).toHaveBeenCalledWith(soundById.externalAlarm, 2);

        events.emit("entityEvent", { name: "panel-a", event: "deactivated" });
        sounds.stop.mockClear();
        behavior.dispose();
        expect(sounds.stop).toHaveBeenCalledWith(soundById.chimeOnce);
        expect(sounds.stop).not.toHaveBeenCalledWith(soundById.externalAlarm);
        sounds.stop.mockClear();
        vi.runAllTimers();
        expect(sounds.stop).not.toHaveBeenCalled();
    });

    it("rejects missing cues and invalid cue values", () => {
        expect(() => harness({ cues: [] })).toThrow("sound.cues must contain at least one cue");
        expect(() => harness({ cues: [{ action: "pause" } as never] })).toThrow('sound.cues[].action must be "play" or "stop"');
        expect(() => harness({ cues: [{ action: "play", id: "", sound: "alarm" }] })).toThrow("sound.cues[].id must be a non-empty sound playback ID");
        expect(() => harness({ cues: [{ action: "stop", soundId: "" }] })).toThrow("sound.cues[].soundId must be a non-empty sound playback ID");
        expect(() => harness({ cues: [{ action: "play", id: "alarmLoop", sound: "alarm.mp3" }] })).toThrow("must be an MP3 file name without its extension");
        expect(() => harness({ cues: [{ action: "play", id: "alarmLoop", sound: "alarm", volume: 1.1 }] })).toThrow("sound.cues[].volume must be finite and between 0 and 1");
        expect(() => harness({ cues: [{ action: "play", id: "alarmLoop", sound: "alarm", loop: 1 as never }] })).toThrow("sound.cues[].loop must be true or false");
        expect(() => harness({ cues: [{ action: "stop", soundId: "alarmLoop", sound: "alarm" } as never] })).toThrow("sound.cues[].sound is not supported");
        expect(() => harness({ cues: [{ action: "play", id: "alarmLoop", sound: "alarm", delay: -1 }] })).toThrow("sound.cues[].delay must be finite and non-negative");
        expect(() => harness({ cues: [{ action: "stop", soundId: "alarmLoop", fade: Number.NaN }] })).toThrow("sound.cues[].fade must be finite and non-negative");
        expect(() => harness({ cues: [{ action: "play", id: "alarmLoop", sound: "alarm", events: [] }] })).toThrow("sound.cues[].events must contain at least one event");
    });
});
