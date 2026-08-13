import type { Mesh } from "babylon-lite";

/** All gameplay and system events understood by Aquanova behaviors. */
export interface EventMap {
    frameStart: { deltaMs: number };
    physicsStep: { deltaSeconds: number };
    frameEnd: { deltaMs: number };
    weaponTriggerPressed: { held: boolean };
    hitWithWeapon: {
        mesh: Mesh;
        point: readonly [number, number, number] | null;
        distance: number;
    };
    weaponAimUpdated: {
        mesh: Mesh | null;
        point: readonly [number, number, number] | null;
        distance: number | null;
    };
    weaponTriggerReleased: Record<string, never>;
    liquefactionStarted: Record<string, never>;
    liquefactionReversed: Record<string, never>;
    liquefactionCompleted: { sound: string };
}
