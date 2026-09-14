import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import type { Mesh } from "../../../../packages/babylon-lite/src";
import { ElectricalDetonatorBehavior } from "../../../../lab/lite/src/demos/aquanova/behaviors/electrical-detonator";
import { FluidElectrifierBehavior } from "../../../../lab/lite/src/demos/aquanova/behaviors/fluid-electrifier";
import type { AquanovaGameContext } from "../../../../lab/lite/src/demos/aquanova/behaviors/game-context";
import { LiquefiableBehavior } from "../../../../lab/lite/src/demos/aquanova/behaviors/liquefiable";
import { electricityPropagationRadius } from "../../../../lab/lite/src/demos/aquanova/fluid-runtime";
import { meshGroupAabbProvider } from "../../../../lab/lite/src/demos/aquanova/mesh-bounds";

function testMesh(): Mesh {
    return {
        _cpuPositions: new Float32Array([-1, -2, -3, 1, 2, 3, 0, 0, 0]),
        _cpuIndices: new Uint32Array([0, 1, 2]),
        worldMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1]),
    } as unknown as Mesh;
}

describe("Aquanova fluid electricity", () => {
    it("transforms cached local corners into the live owner AABB", () => {
        const bounds = meshGroupAabbProvider([testMesh()])();

        expect(bounds).toEqual({
            min: [4, 4, 4],
            max: [6, 8, 10],
        });
    });

    it("uses a non-negative expanding propagation front", () => {
        const electricity = {
            origin: [1, 2, 3] as const,
            startedAtSeconds: 10,
            propagationSpeed: 8,
        };

        expect(electricityPropagationRadius(electricity, 9)).toBe(0);
        expect(electricityPropagationRadius(electricity, 10)).toBe(0);
        expect(electricityPropagationRadius(electricity, 10.5)).toBe(4);
    });

    it("rejects a non-boolean liquefaction electrifiable value", () => {
        expect(() => new LiquefiableBehavior("water", [testMesh()], { name: "liquefaction", electrifiable: "false" as unknown as boolean }, {} as AquanovaGameContext)).toThrow(
            "liquefaction.electrifiable must be true or false"
        );
    });

    it("registers a default electrifier and raises its declared event", () => {
        const emit = vi.fn();
        let registered: Parameters<NonNullable<AquanovaGameContext["fluidSimulations"]["registerElectrifier"]>>[0] | undefined;
        const dispose = vi.fn();
        const context = {
            events: { emit },
            fluidSimulations: {
                registerElectrifier: vi.fn((registration) => {
                    registered = registration;
                    return { dispose };
                }),
            },
        } as unknown as AquanovaGameContext;
        const behavior = new FluidElectrifierBehavior("wire", [testMesh()], {}, context);

        behavior.start();
        expect(behavior.particleThreshold).toBe(24);
        expect(behavior.propagationSpeed).toBe(8);
        expect(registered?.aabb()).toEqual({ min: [4, 4, 4], max: [6, 8, 10] });
        registered?.onElectrified?.({ id: 1, label: "water", electricity: null });
        expect(emit).toHaveBeenCalledWith("entityEvent", { name: "wire", event: "fluidElectrified" });

        behavior.dispose();
        expect(dispose).toHaveBeenCalledOnce();
    });

    it("raises explode exactly once after the detonator threshold", () => {
        const emit = vi.fn();
        let onCount: ((count: number) => void) | undefined;
        const dispose = vi.fn();
        const context = {
            events: { emit },
            fluidSimulations: {
                registerElectricityReceiver: vi.fn((registration) => {
                    onCount = registration.onCount;
                    return { dispose };
                }),
            },
        } as unknown as AquanovaGameContext;
        const behavior = new ElectricalDetonatorBehavior("barrel", [testMesh()], {}, context);

        behavior.start();
        onCount?.(3);
        expect(emit).not.toHaveBeenCalled();
        onCount?.(4);
        onCount?.(20);
        expect(emit).toHaveBeenCalledTimes(1);
        expect(emit).toHaveBeenCalledWith("entityEvent", { name: "barrel", event: "explode" });
        expect(dispose).toHaveBeenCalledOnce();
    });

    it("declares opt-in domains and both behavior event surfaces in metadata", () => {
        const metadata = JSON.parse(readFileSync("lab/public/aquanova/behaviors.json", "utf8")) as {
            behaviors: Record<string, { eventsRaised: Array<{ name: string }>; properties: Record<string, { default?: unknown; integer?: boolean }> }>;
        };

        expect(metadata.behaviors.liquefaction?.properties.electrifiable?.default).toBe(false);
        expect(metadata.behaviors.fluidSimulation?.properties.electrifiable?.default).toBe(false);
        expect(metadata.behaviors.player?.properties.electrifiedParticleCount?.integer).toBe(true);
        expect(metadata.behaviors.fluidElectrifier?.properties.particleThreshold?.integer).toBe(true);
        expect(metadata.behaviors.electricalDetonator?.properties.particleThreshold?.integer).toBe(true);
        expect(metadata.behaviors.fluidElectrifier?.eventsRaised).toContainEqual({ name: "fluidElectrified" });
        expect(metadata.behaviors.electricalDetonator?.eventsRaised).toContainEqual({ name: "explode" });
        expect(metadata.behaviors.player?.eventsRaised).toEqual([{ name: "electricalContact" }, { name: "electricalContactEnded" }]);
    });
});
