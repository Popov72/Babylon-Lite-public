import type { Mesh } from "babylon-lite";

/** All gameplay and system events understood by Aquanova behaviors. */
export interface AquanovaEventMap {
    entityEvent: { name: string; event: string };
    weaponSlotSelected: { slot: number };
    weaponCycleRequested: { direction: -1 | 1 };
    weaponEquippedChanged: { slot: number | null };
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
    liquefactionStarted: { meshes: readonly Mesh[] };
    liquefactionReversed: Record<string, never>;
    liquefactionCancelled: { meshes: readonly Mesh[] };
    liquefactionCompleted: { meshes: readonly Mesh[]; sound: string };
}
