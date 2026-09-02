import { describe, expect, it } from "vitest";

import {
    createFluidControlsTransaction,
    initializeFluidControlsTransaction,
    markFluidControlsChanged,
    runFluidControlsTransaction,
    wrapFluidControlsCallbacks,
} from "../../../../packages/babylon-lite/src/fluid/controls/controls-transaction";

describe("fluid controls transaction", () => {
    it("applies an idle change as one normalized snapshot", () => {
        const state = { color: "#000", foam: 0 };
        const applied: { values: typeof state; keys: readonly (keyof typeof state)[] }[] = [];
        const tx = createFluidControlsTransaction({
            read: () => ({ ...state }),
            onApply: (values, keys) => applied.push({ values, keys }),
        });

        state.color = "#fff";
        markFluidControlsChanged(tx, "color");

        expect(applied).toEqual([{ values: { color: "#fff", foam: 0 }, keys: ["color"] }]);
    });

    it("coalesces duplicate changes and applies only at the outermost successful boundary", () => {
        const state = { color: "#000", foam: 0 };
        const applied: { values: typeof state; keys: readonly (keyof typeof state)[] }[] = [];
        const tx = createFluidControlsTransaction({
            read: () => ({ ...state }),
            onApply: (values, keys) => applied.push({ values, keys }),
        });

        runFluidControlsTransaction(tx, () => {
            state.foam = 1;
            markFluidControlsChanged(tx, "foam");
            markFluidControlsChanged(tx, "foam");
            runFluidControlsTransaction(tx, () => {
                state.color = "#fff";
                markFluidControlsChanged(tx, "color");
            });
            expect(applied).toEqual([]);
        });

        expect(applied).toEqual([{ values: { color: "#fff", foam: 1 }, keys: ["foam", "color"] }]);
    });

    it("rolls back queued changes when the transaction body throws", () => {
        const state = { color: "#000", foam: 0 };
        const applied: string[][] = [];
        const tx = createFluidControlsTransaction({
            read: () => ({ ...state }),
            onApply: (_values, keys) => applied.push([...keys]),
        });

        expect(() =>
            runFluidControlsTransaction(tx, () => {
                state.foam = 1;
                markFluidControlsChanged(tx, "foam");
                throw new Error("boom");
            })
        ).toThrow(/boom/);

        expect(applied).toEqual([]);
        expect(tx.active).toBe(false);
        state.color = "#fff";
        markFluidControlsChanged(tx, "color");
        expect(applied).toEqual([["color"]]);
    });

    it("rolls back only a failing nested body when its parent catches the error", () => {
        const state = { outer: 0, inner: 0, after: 0 };
        const applied: string[][] = [];
        const tx = createFluidControlsTransaction({
            read: () => ({ ...state }),
            onApply: (_values, keys) => applied.push([...keys]),
        });

        runFluidControlsTransaction(tx, () => {
            state.outer = 1;
            markFluidControlsChanged(tx, "outer");
            try {
                runFluidControlsTransaction(tx, () => {
                    state.inner = 1;
                    markFluidControlsChanged(tx, "inner");
                    throw new Error("nested");
                });
            } catch {
                // The external state is illustrative; only the queued effect is transaction-owned.
            }
            state.after = 1;
            markFluidControlsChanged(tx, "after");
        });

        expect(applied).toEqual([["outer", "after"]]);
    });

    it("clears the commit before invoking a failing callback so later changes stay independent", () => {
        const state = { first: 0, second: 0 };
        const applied: string[][] = [];
        let fail = true;
        const tx = createFluidControlsTransaction({
            read: () => ({ ...state }),
            onApply: (_values, keys) => {
                applied.push([...keys]);
                if (fail) {
                    fail = false;
                    throw new Error("host failed");
                }
            },
        });

        expect(() =>
            runFluidControlsTransaction(tx, () => {
                state.first = 1;
                markFluidControlsChanged(tx, "first");
            })
        ).toThrow(/host failed/);
        expect(tx.active).toBe(false);

        state.second = 1;
        markFluidControlsChanged(tx, "second");
        expect(applied).toEqual([["first"], ["second"]]);
    });

    it("restores the last committed snapshot when snapshot application fails", () => {
        let state = { pages: 10, paged: false };
        let fail = false;
        const tx = createFluidControlsTransaction({
            read: () => ({ ...state }),
            restore: (snapshot) => {
                state = snapshot;
            },
            onApply: () => {
                if (fail) {
                    throw new Error("allocation failed");
                }
            },
        });
        initializeFluidControlsTransaction(tx);
        state = { pages: 20, paged: true };
        fail = true;

        expect(() => markFluidControlsChanged(tx, "pages", "paged")).toThrow("allocation failed");
        expect(state).toEqual({ pages: 10, paged: false });
    });

    it("lets an external consumer implement only onApply while immediate validators keep returns", () => {
        type Values = { color: string; foam: number };
        type Callbacks = {
            onColor?(color: string): void;
            onFoam?(value: number): void;
            onValidate?(value: number): string | undefined;
        };
        const state: Values = { color: "#000", foam: 0 };
        const applied: { values: Values; keys: readonly (keyof Values)[] }[] = [];
        const tx = createFluidControlsTransaction({
            read: () => ({ ...state }),
            onApply: (values, keys) => applied.push({ values, keys }),
        });
        const callbacks = wrapFluidControlsCallbacks<Callbacks, keyof Values>({ onValidate: (value) => (value > 0 ? undefined : "must be positive") }, tx, {
            immediate: ["onValidate"],
            changedKeys: { onColor: "color", onFoam: "foam" },
        });

        expect(callbacks.onValidate?.(-1)).toBe("must be positive");
        runFluidControlsTransaction(tx, () => {
            state.foam = 1;
            for (let i = 0; i < 15; i++) {
                callbacks.onFoam?.(i);
            }
            state.color = "#fff";
            callbacks.onColor?.("#fff");
        });

        expect(applied).toEqual([{ values: { color: "#fff", foam: 1 }, keys: ["foam", "color"] }]);
    });

    it("keeps legacy callbacks immediate outside transactions and rejects unsafe replay", () => {
        type Callbacks = { onColor?(color: string): void };
        const calls: string[] = [];
        const tx = createFluidControlsTransaction<{ color: string }, "color">({
            read: () => ({ color: "#fff" }),
        });
        const callbacks = wrapFluidControlsCallbacks<Callbacks, "color">({ onColor: (color) => calls.push(color) }, tx, {
            changedKeys: { onColor: "color" },
        });

        callbacks.onColor?.("#aaa");
        expect(() => runFluidControlsTransaction(tx, () => callbacks.onColor?.("#bbb"))).toThrow(/require an onApply snapshot consumer/);

        expect(calls).toEqual(["#aaa"]);
    });
});
