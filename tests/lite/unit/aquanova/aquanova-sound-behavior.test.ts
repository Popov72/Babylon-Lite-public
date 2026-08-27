import { afterEach, describe, expect, it, vi } from "vitest";
import { AquanovaEventManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/aquanova-event-manager";
import { SoundBehavior } from "../../../../lab/lite/src/demos/aquanova/behaviors/sound";
import { PlaySoundBehavior, StopSoundBehavior } from "../../../../lab/lite/src/demos/aquanova/behaviors/sound-control";

function harness(config: ConstructorParameters<typeof SoundBehavior>[2]) {
    const events = new AquanovaEventManager();
    const soundByName = {
        alarm: { label: "alarm", source: "/aquanova/sounds/alarm.mp3", sound: "runtime-alarm" },
        chime: { label: "chime", source: "/aquanova/sounds/chime.mp3", sound: "runtime-chime" },
    };
    const sounds = {
        load: vi.fn(async (label: string) => soundByName[label.slice("sound:".length) as keyof typeof soundByName]),
        play: vi.fn(),
        stop: vi.fn(),
    };
    const behavior = new SoundBehavior("speaker", [], config, { events, sounds } as never);
    return { behavior, events, soundByName, sounds };
}

describe("Aquanova sound behavior", () => {
    afterEach(() => {
        vi.useRealTimers();
    });

    function controlHarness() {
        const events = new AquanovaEventManager();
        const sound = { label: "electric", source: "/aquanova/sounds/electricField.mp3", sound: "runtime-electric" };
        const sounds = {
            registerPlayback: vi.fn(),
            resolvePlayback: vi.fn(async () => sound),
            play: vi.fn(),
            stop: vi.fn(),
        };
        return { events, sound, sounds };
    }

    describe("Aquanova sound control behaviors", () => {
        it("defines and plays an independent sound ID", async () => {
            const { events, sound, sounds } = controlHarness();
            const behavior = new PlaySoundBehavior("electrifier", [], { id: "electricLoop", sound: "electricField", fadeInDelay: 1.5, volume: 0.4, loop: true }, {
                events,
                sounds,
            } as never);

            await behavior.init();
            behavior.start();

            expect(sounds.registerPlayback).toHaveBeenCalledWith("electricLoop", "/aquanova/sounds/electricField.mp3?v=20260813-1", { preloadCount: 1 });
            expect(sounds.resolvePlayback).toHaveBeenCalledWith("electricLoop");
            expect(sounds.play).toHaveBeenCalledWith(sound, { fade: 1.5, loop: true, volume: 0.4 });
            behavior.dispose();
            expect(sounds.stop).toHaveBeenCalledWith(sound);
        });

        it("stops an existing sound ID after a matching event", async () => {
            const { events, sound, sounds } = controlHarness();
            const behavior = new StopSoundBehavior("fluid", [], { soundId: "electricLoop", source: "fluid", event: "endSimulation", fadeOutDelay: 2 }, { events, sounds } as never);

            await behavior.init();
            behavior.start();
            events.emit("entityEvent", { name: "other", event: "endSimulation" });
            expect(sounds.stop).not.toHaveBeenCalled();
            events.emit("entityEvent", { name: "fluid", event: "endSimulation" });
            expect(sounds.stop).toHaveBeenCalledWith(sound, 2);
        });

        it("rejects missing IDs and invalid playback options", () => {
            const { events, sounds } = controlHarness();
            expect(() => new PlaySoundBehavior("fluid", [], { id: "", sound: "electricField" }, { events, sounds } as never)).toThrow(
                "playSound.id must be a non-empty sound playback ID"
            );
            expect(() => new StopSoundBehavior("fluid", [], { soundId: "" }, { events, sounds } as never)).toThrow("stopSound.soundId must be a non-empty sound playback ID");
            expect(() => new PlaySoundBehavior("fluid", [], { id: "electric", sound: "electricField", volume: 1.1 }, { events, sounds } as never)).toThrow(
                "playSound.volume must be finite and between 0 and 1"
            );
            expect(() => new PlaySoundBehavior("fluid", [], { id: "electric", sound: "electricField", source: "electrifier" }, { events, sounds } as never)).toThrow(
                "playSound.source and playSound.event must be provided together"
            );
            expect(() => new StopSoundBehavior("fluid", [], { soundId: "electricLoop", event: "endSimulation" }, { events, sounds } as never)).toThrow(
                "stopSound.source and stopSound.event must be provided together"
            );
            expect(() => new StopSoundBehavior("fluid", [], { soundId: "electricLoop", fadeOutDelay: -1 }, { events, sounds } as never)).toThrow(
                "stopSound.fadeOutDelay must be finite and non-negative"
            );
        });
    });

    it("preloads each distinct sound once and runs cues without subscriptions at startup", async () => {
        const { behavior, soundByName, sounds } = harness({
            cues: [
                { sound: "alarm", action: "play", fade: 0.75 },
                { sound: "alarm", action: "stop", events: [{ source: "panel", name: "deactivated" }] },
                { sound: "chime", action: "play" },
            ],
        });

        await behavior.init();
        behavior.start();

        expect(sounds.load).toHaveBeenCalledWith("sound:alarm", "/aquanova/sounds/alarm.mp3?v=20260813-1", { preloadCount: 1 });
        expect(sounds.load).toHaveBeenCalledWith("sound:chime", "/aquanova/sounds/chime.mp3?v=20260813-1", { preloadCount: 1 });
        expect(sounds.load).toHaveBeenCalledTimes(2);
        expect(sounds.play).toHaveBeenNthCalledWith(1, soundByName.alarm, { fade: 0.75 });
        expect(sounds.play).toHaveBeenNthCalledWith(2, soundByName.chime, { fade: 0 });
        behavior.dispose();
    });

    it("runs every matching cue independently and cancels pending actions on disposal", async () => {
        vi.useFakeTimers();
        const { behavior, events, soundByName, sounds } = harness({
            cues: [
                {
                    sound: "alarm",
                    action: "stop",
                    events: [{ source: ["panel-a", "panel-b"], name: "deactivated" }],
                    delay: 1.5,
                    fade: 2,
                },
                {
                    sound: "chime",
                    action: "play",
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
        expect(sounds.play).toHaveBeenCalledWith(soundByName.chime, { fade: 0.25 });
        vi.advanceTimersByTime(1499);
        expect(sounds.stop).not.toHaveBeenCalled();
        vi.advanceTimersByTime(1);
        expect(sounds.stop).toHaveBeenCalledWith(soundByName.alarm, 2);

        events.emit("entityEvent", { name: "panel-a", event: "deactivated" });
        behavior.dispose();
        sounds.stop.mockClear();
        vi.runAllTimers();
        expect(sounds.stop).not.toHaveBeenCalled();
    });

    it("rejects missing cues and invalid cue values", () => {
        expect(() => harness({ cues: [] })).toThrow("sound.cues must contain at least one cue");
        expect(() => harness({ cues: [{ sound: "alarm", action: "pause" as "play" }] })).toThrow('sound.cues[].action must be "play" or "stop"');
        expect(() => harness({ cues: [{ sound: "alarm.mp3", action: "play" }] })).toThrow("must be an MP3 file name without its extension");
        expect(() => harness({ cues: [{ sound: "alarm", action: "play", delay: -1 }] })).toThrow("sound.cues[].delay must be finite and non-negative");
        expect(() => harness({ cues: [{ sound: "alarm", action: "play", fade: Number.NaN }] })).toThrow("sound.cues[].fade must be finite and non-negative");
        expect(() => harness({ cues: [{ sound: "alarm", action: "play", events: [] }] })).toThrow("sound.cues[].events must contain at least one event");
    });
});
