import { afterEach, describe, expect, it, vi } from "vitest";
import type { Mesh } from "../../../../packages/babylon-lite/src";
import { createSceneNode } from "../../../../packages/babylon-lite/src/scene/scene-node";
import { AquanovaEventManager } from "../../../../lab/lite/src/demos/aquanova/behaviors/aquanova-event-manager";
import { FluidSimulationBehavior } from "../../../../lab/lite/src/demos/aquanova/behaviors/fluid-simulation";
import {
    fluidEmissionCompletionTarget,
    fluidSimulationShutdownLifecycle,
    fluidSimulationShutdownStepDelta,
    FluidSimulationRuntime,
    type FluidSimulationRegistration,
    type FluidSimulationState,
} from "../../../../lab/lite/src/demos/aquanova/behaviors/fluid-simulation-runtime";
import { flipParticleCountForVolume } from "../../../../lab/lite/src/demos/fluid/grid-settings";

const setting = {
    formatVersion: 12,
    meta: { demo: "aquanova-fluid-sim", method: "MLS-MPM" },
    demoParams: {},
    physics: {},
    physicsParticleSize: 1,
    gridPosition: [10, 20, 30],
    gridSize: [4, 5, 6],
    particleCount: 10,
    emitters: [{ id: "inlet", name: "Main inlet", enabled: false }],
    sinks: [{ id: "drain", name: "Floor drain", enabled: false }],
    render: {},
    foam: {},
};

afterEach(() => {
    vi.unstubAllGlobals();
});

function mesh(): Mesh {
    return createSceneNode("fluid-anchor") as Mesh;
}

describe("Aquanova fluidSimulation behavior", () => {
    it("runs the shutdown duration, fades, and supports zero-length phases", () => {
        expect(fluidSimulationShutdownLifecycle(9, 10, 2)).toEqual({ opacity: 1, stopped: false });
        expect(fluidSimulationShutdownLifecycle(11, 10, 2)).toEqual({ opacity: 0.5, stopped: false });
        expect(fluidSimulationShutdownLifecycle(12, 10, 2)).toEqual({ opacity: 0, stopped: true });
        expect(fluidSimulationShutdownStepDelta(11.9, 1, 10, 2)).toBeCloseTo(0.1);
        expect(fluidSimulationShutdownLifecycle(0, 0, 0)).toEqual({ opacity: 0, stopped: true });
        expect(fluidSimulationShutdownLifecycle(1, 0, 2)).toEqual({ opacity: 0.5, stopped: false });
    });

    it("uses the FLIP-derived reset count unless an inflow can fill the capacity", () => {
        const derivedFlipCount = flipParticleCountForVolume(96, 0.25, 8);

        expect(derivedFlipCount).toBe(49_152);
        expect(fluidEmissionCompletionTarget(100_000, derivedFlipCount, false)).toBe(49_152);
        expect(fluidEmissionCompletionTarget(100_000, derivedFlipCount, true)).toBe(100_000);
    });

    it("maps external events to lifecycle and named flow-object actions", async () => {
        const loadedSetting = structuredClone(setting);
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue({
                ok: true,
                status: 200,
                json: async () => loadedSetting,
            })
        );
        const events = new AquanovaEventManager();
        const fluidSimulations = new FluidSimulationRuntime();
        const owner = createSceneNode("waterfall");
        owner.position.set(3, 4, 5);
        const primitive = mesh();
        primitive.parent = owner;
        const behavior = new FluidSimulationBehavior(
            "waterfall",
            [primitive],
            {
                fluidSim: "flip-test",
                shutdownDuration: 7,
                shutdownAlphaDecay: 3,
                eventActions: [
                    { source: "control-panel", event: "activated", action: "enableSimulation" },
                    { source: "control-panel", event: "deactivated", action: "disableSimulation" },
                    { source: "pause-panel", event: "activated", action: "pauseSimulation" },
                    { source: "pause-panel", event: "deactivated", action: "unpauseSimulation" },
                    { source: "collision-panel", event: "activated", action: "enablePlayerCollision" },
                    { source: "collision-panel", event: "deactivated", action: "disablePlayerCollision" },
                    { source: ["valve-a", "valve-b"], event: "opened", action: "enableEmitter", emitter: "Main inlet" },
                    { source: "valve-a", event: "closed", action: "disableEmitter", emitter: "Main inlet" },
                    { source: "drain-panel", event: "opened", action: "enableSink", sink: "Floor drain" },
                    { source: "drain-panel", event: "closed", action: "disableSink", sink: "Floor drain" },
                    { source: "shutdown-panel", event: "activated", action: "shutdownSimulation" },
                ],
            },
            { events, fluidSimulations }
        );

        await behavior.init();
        behavior.start();
        events.emit("entityEvent", { name: "valve-b", event: "opened" });
        events.emit("entityEvent", { name: "pause-panel", event: "activated" });
        events.emit("entityEvent", { name: "pause-panel", event: "deactivated" });
        events.emit("entityEvent", { name: "control-panel", event: "activated" });
        events.emit("entityEvent", { name: "collision-panel", event: "deactivated" });

        const registered: FluidSimulationRegistration[] = [];
        const updates: FluidSimulationState[] = [];
        const playerCollisionUpdates: boolean[] = [];
        const flowUpdates: Array<[string, string, boolean]> = [];
        const unregistered: FluidSimulationRegistration[] = [];
        fluidSimulations.installBackend({
            register: (registration) => registered.push(registration),
            update: (_registration, state) => updates.push(state),
            updatePlayerCollision: (_registration, enabled) => playerCollisionUpdates.push(enabled),
            updateFlowObject: (_registration, kind, name, enabled) => flowUpdates.push([kind, name, enabled]),
            unregister: (registration) => unregistered.push(registration),
        });

        expect(fetch).toHaveBeenCalledWith("/aquanova/fluidSim/flip-test.json");
        expect(registered).toHaveLength(1);
        expect(registered[0]).toMatchObject({
            entityName: "waterfall",
            settingName: "flip-test",
            setting: loadedSetting,
            shutdownDuration: 7,
            shutdownAlphaDecay: 3,
        });
        expect(registered[0]?.anchor).toBe(owner);
        expect([registered[0]?.anchor.worldMatrix[12], registered[0]?.anchor.worldMatrix[13], registered[0]?.anchor.worldMatrix[14]]).toEqual([3, 4, 5]);
        expect(loadedSetting.emitters[0]?.enabled).toBe(true);
        expect(updates).toEqual(["running"]);
        expect(playerCollisionUpdates).toEqual([false]);
        expect(flowUpdates).toEqual([]);
        const completionEvent = vi.fn();
        events.on("entityEvent", completionEvent);
        registered[0]?.onEmissionComplete?.();
        expect(completionEvent).toHaveBeenCalledWith({ name: "waterfall", event: "emissionComplete" });

        events.emit("entityEvent", { name: "waterfall", event: "disable" });
        events.emit("entityEvent", { name: "control-panel", event: "deactivated" });
        events.emit("entityEvent", { name: "control-panel", event: "activated" });
        events.emit("entityEvent", { name: "pause-panel", event: "activated" });
        events.emit("entityEvent", { name: "pause-panel", event: "deactivated" });
        events.emit("entityEvent", { name: "collision-panel", event: "activated" });
        events.emit("entityEvent", { name: "collision-panel", event: "deactivated" });
        events.emit("entityEvent", { name: "valve-a", event: "closed" });
        events.emit("entityEvent", { name: "drain-panel", event: "opened" });
        events.emit("entityEvent", { name: "drain-panel", event: "closed" });
        events.emit("entityEvent", { name: "shutdown-panel", event: "activated" });
        events.emit("entityEvent", { name: "control-panel", event: "deactivated" });
        events.emit("entityEvent", { name: "collision-panel", event: "activated" });
        events.emit("entityEvent", { name: "valve-a", event: "opened" });
        expect(updates).toEqual(["running", "paused", "running", "paused", "running", "shutdown"]);
        expect(playerCollisionUpdates).toEqual([false, true, false, true]);
        expect(flowUpdates).toEqual([
            ["emitter", "Main inlet", false],
            ["sink", "Floor drain", true],
            ["sink", "Floor drain", false],
        ]);
        expect(loadedSetting.emitters[0]?.enabled).toBe(false);
        expect(loadedSetting.sinks[0]?.enabled).toBe(false);

        behavior.dispose();
        expect(unregistered).toEqual(registered);
        events.dispose();
        fluidSimulations.dispose();
    });

    it("uses the documented shutdown defaults", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue({
                ok: true,
                status: 200,
                json: async () => structuredClone(setting),
            })
        );
        const events = new AquanovaEventManager();
        const fluidSimulations = new FluidSimulationRuntime();
        const behavior = new FluidSimulationBehavior(
            "waterfall",
            [mesh()],
            { fluidSim: "liquid-crate", eventActions: [{ source: "panel", event: "activated", action: "enableSimulation" }] },
            { events, fluidSimulations }
        );
        let registration: FluidSimulationRegistration | undefined;
        fluidSimulations.installBackend({
            register: (value) => {
                registration = value;
            },
            update: () => {},
            updatePlayerCollision: () => {},
            updateFlowObject: () => {},
            unregister: () => {},
        });

        await behavior.init();
        behavior.start();

        expect(registration).toMatchObject({ shutdownDuration: 10, shutdownAlphaDecay: 2 });
        behavior.dispose();
        events.dispose();
        fluidSimulations.dispose();
    });

    it("rejects malformed configuration and failed setting loads", async () => {
        const events = new AquanovaEventManager();
        const fluidSimulations = new FluidSimulationRuntime();

        expect(
            () =>
                new FluidSimulationBehavior(
                    "waterfall",
                    [mesh()],
                    { fluidSim: "bad.json", eventActions: [{ source: "panel", event: "activated", action: "enableSimulation" }] },
                    { events, fluidSimulations }
                )
        ).toThrow("must be a non-empty file name without the .json extension");
        expect(() => new FluidSimulationBehavior("waterfall", [mesh()], { fluidSim: "bad", eventActions: [] }, { events, fluidSimulations })).toThrow(
            "eventActions must contain at least one action"
        );
        expect(
            () =>
                new FluidSimulationBehavior(
                    "waterfall",
                    [mesh()],
                    {
                        fluidSim: "bad",
                        eventActions: [{ source: "panel", event: "activated", action: "enableSimulation" }],
                        shutdownDuration: -1,
                    },
                    {
                        events,
                        fluidSimulations,
                    }
                )
        ).toThrow("shutdownDuration must be finite and non-negative");

        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404 }));
        const behavior = new FluidSimulationBehavior(
            "waterfall",
            [mesh()],
            { fluidSim: "missing", eventActions: [{ source: "panel", event: "activated", action: "enableSimulation" }] },
            { events, fluidSimulations }
        );
        await expect(behavior.init()).rejects.toThrow('fluidSimulation "missing" could not be loaded: HTTP 404');

        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue({
                ok: true,
                status: 200,
                json: async () => structuredClone(setting),
            })
        );
        const unknownEmitter = new FluidSimulationBehavior(
            "waterfall",
            [mesh()],
            { fluidSim: "flip-test", eventActions: [{ source: "valve", event: "opened", action: "enableEmitter", emitter: "Missing inlet" }] },
            { events, fluidSimulations }
        );
        await expect(unknownEmitter.init()).rejects.toThrow('has no emitter named "Missing inlet"');
        events.dispose();
        fluidSimulations.dispose();
    });
});
