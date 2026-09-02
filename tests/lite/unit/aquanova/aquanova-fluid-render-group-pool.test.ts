import { describe, expect, it, vi } from "vitest";

import { createReferenceCountedPool } from "../../../../packages/babylon-lite/src/fluid/rendering/render-group-pool.js";

// A stand-in for a production render group: owns GPU buffers and a frame-graph task whose disposal
// tears both down. The test asserts the pool destroys exactly these when the last holder releases.
interface FakeGroup {
    readonly key: string;
    destroyed: boolean;
    taskRemoved: boolean;
}

function makeGroup(key: string): FakeGroup {
    return { key, destroyed: false, taskRemoved: false };
}

describe("Aquanova independent render-group pool", () => {
    it("shares one group across holders and evicts it only on the last release", () => {
        const disposed: FakeGroup[] = [];
        const pool = createReferenceCountedPool<string, FakeGroup>((group) => {
            group.destroyed = true;
            group.taskRemoved = true;
            disposed.push(group);
        });
        const create = vi.fn((key: string) => makeGroup(key));

        const first = pool.acquire("water", () => create("water"));
        const second = pool.acquire("water", () => create("water"));
        expect(second).toBe(first); // same profile → shared group
        expect(create).toHaveBeenCalledTimes(1);
        expect(pool.size).toBe(1);
        expect(pool.refCount("water")).toBe(2);

        // First holder leaves: the group stays alive for the second holder.
        expect(pool.release("water")).toBe(false);
        expect(first.destroyed).toBe(false);
        expect(pool.size).toBe(1);

        // Last holder leaves: buffers destroyed, frame-graph task removed, map shrinks.
        expect(pool.release("water")).toBe(true);
        expect(first.destroyed).toBe(true);
        expect(first.taskRemoved).toBe(true);
        expect(pool.size).toBe(0);
        expect(pool.has("water")).toBe(false);
        expect(disposed).toEqual([first]);
    });

    it("keeps distinct profiles separate and returns to a stable, empty map", () => {
        const disposed: string[] = [];
        const pool = createReferenceCountedPool<string, FakeGroup>((group) => disposed.push(group.key));

        const keys = ["water", "lava", "acid"];
        const groups = keys.map((key) => pool.acquire(key, () => makeGroup(key)));
        expect(pool.size).toBe(3);
        expect(new Set(groups).size).toBe(3); // three distinct groups

        // Re-acquire one profile, then release every reference in a different order.
        pool.acquire("lava", () => makeGroup("lava"));
        expect(pool.refCount("lava")).toBe(2);

        expect(pool.release("water")).toBe(true);
        expect(pool.release("lava")).toBe(false); // one reference remains
        expect(pool.release("lava")).toBe(true);
        expect(pool.release("acid")).toBe(true);

        expect(pool.size).toBe(0);
        expect(disposed.sort()).toEqual(["acid", "lava", "water"]);
    });

    it("ignores releases for unknown keys and disposes everything on teardown", () => {
        const disposed: string[] = [];
        const pool = createReferenceCountedPool<string, FakeGroup>((group) => disposed.push(group.key));

        expect(pool.release("missing")).toBe(false);

        pool.acquire("a", () => makeGroup("a"));
        pool.acquire("b", () => makeGroup("b"));
        pool.disposeAll();

        expect(pool.size).toBe(0);
        expect(disposed.sort()).toEqual(["a", "b"]);
    });
});
