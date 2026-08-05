import type { Mesh } from "babylon-lite";

/** All gameplay and system events understood by Aquanova behaviors. */
export interface EventMap {
    frameStart: { deltaMs: number };
    physicsStep: { deltaSeconds: number };
    frameEnd: { deltaMs: number };
    hitWithWeapon: {
        mesh: Mesh;
        point: readonly [number, number, number] | null;
    };
}
