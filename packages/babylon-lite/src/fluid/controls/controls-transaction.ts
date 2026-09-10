// Atomic snapshot application for shared fluid controls.
//
// A transaction never replays individual callbacks. It coalesces changed keys and, after the
// outermost body succeeds, invokes one host effect with the normalized current snapshot. If a body
// throws, keys added by that body are rolled back. JavaScript cannot roll back mutations already
// performed inside a synchronous host callback, so the callback is the explicit atomic boundary:
// it must validate before mutating. If it throws, no later effect is attempted and the transaction
// is reset before the error propagates.

/** Coalesces control changes into one normalized snapshot application. */
export interface FluidControlsTransaction<TKey extends PropertyKey = string> {
    /** Whether a transaction body is currently open. */
    active: boolean;
    /** Whether this transaction has a snapshot consumer. */
    readonly appliesSnapshots: boolean;
    /** @internal */
    _depth: number;
    /** @internal */
    _applying: boolean;
    /** @internal */
    _changed: Set<TKey>;
    /** @internal */
    _committed: object | null;
    /** @internal */
    _read: () => object;
    /** @internal */
    _onApply?: (values: object, changedKeys: readonly TKey[]) => void;
    /** @internal */
    _restore?: (values: object) => void;
}

/** Create a fresh snapshot transaction. */
export function createFluidControlsTransaction<TValues extends object, TKey extends keyof TValues = keyof TValues>(options: {
    read: () => TValues;
    onApply?: (values: TValues, changedKeys: readonly TKey[]) => void;
    restore?: (values: TValues) => void;
}): FluidControlsTransaction<TKey> {
    return {
        active: false,
        appliesSnapshots: options.onApply !== undefined,
        _depth: 0,
        _applying: false,
        _changed: new Set<TKey>(),
        _committed: null,
        _read: options.read,
        ...((options.onApply ? { _onApply: options.onApply as (values: object, changedKeys: readonly TKey[]) => void } : {}) satisfies Pick<
            FluidControlsTransaction<TKey>,
            "_onApply"
        >),
        ...((options.restore ? { _restore: options.restore as (values: object) => void } : {}) satisfies Pick<FluidControlsTransaction<TKey>, "_restore">),
    };
}

function flushFluidControlsTransaction<TKey extends PropertyKey>(transaction: FluidControlsTransaction<TKey>): void {
    if (transaction._changed.size === 0) {
        return;
    }
    const changedKeys = [...transaction._changed];
    transaction._changed.clear();
    if (!transaction._onApply) {
        throw new Error("Fluid controls transactions with changed values require an onApply snapshot consumer.");
    }
    transaction._applying = true;
    try {
        const snapshot = transaction._read();
        transaction._onApply(snapshot, changedKeys);
        transaction._committed = structuredClone(snapshot);
    } catch (error) {
        if (transaction._committed && transaction._restore) {
            transaction._applying = false;
            try {
                transaction._restore(structuredClone(transaction._committed));
            } catch (restoreError) {
                throw new AggregateError([error, restoreError], "Fluid controls application failed and its committed snapshot could not be restored.", { cause: restoreError });
            } finally {
                transaction._applying = true;
            }
        }
        throw error;
    } finally {
        transaction._applying = false;
    }
}

/** Record normalized keys changed by one logical operation. Applies immediately when idle. */
export function markFluidControlsChanged<TKey extends PropertyKey>(transaction: FluidControlsTransaction<TKey>, ...keys: readonly TKey[]): void {
    if (transaction._applying) {
        throw new Error("Fluid controls cannot be changed reentrantly from onApply.");
    }
    for (const key of keys) {
        transaction._changed.add(key);
    }
    if (transaction._depth === 0) {
        flushFluidControlsTransaction(transaction);
    }
}

/** Run a possibly nested body. A throwing body contributes no keys to its parent transaction. */
export function runFluidControlsTransaction<TKey extends PropertyKey, T>(transaction: FluidControlsTransaction<TKey>, apply: () => T): T {
    if (transaction._applying) {
        throw new Error("Fluid controls transactions cannot start reentrantly from onApply.");
    }
    const checkpoint = new Set(transaction._changed);
    transaction._depth++;
    transaction.active = true;
    let result: T;
    try {
        result = apply();
    } catch (error) {
        transaction._depth--;
        transaction.active = transaction._depth > 0;
        transaction._changed = checkpoint;
        if (transaction._depth === 0) {
            transaction._changed.clear();
        }
        throw error;
    }
    transaction._depth--;
    transaction.active = transaction._depth > 0;
    if (transaction._depth === 0) {
        flushFluidControlsTransaction(transaction);
    }
    return result;
}

/** Capture the initial committed snapshot after the control surface has been constructed. */
export function initializeFluidControlsTransaction<TKey extends PropertyKey>(transaction: FluidControlsTransaction<TKey>): void {
    if (transaction._depth > 0 || transaction._applying) {
        throw new Error("Fluid controls cannot initialize a committed snapshot during a transaction.");
    }
    transaction._committed = structuredClone(transaction._read());
}

/** Options for adapting an existing callback surface to snapshot application. */
export interface WrapFluidControlsCallbacksOptions<T extends object, TKey extends PropertyKey> {
    /** Callback names that must remain synchronous because callers consume their return value. */
    immediate?: readonly (keyof T & string)[];
    /** Normalized snapshot keys affected by each callback. */
    changedKeys: Partial<Record<keyof T & string, TKey | readonly TKey[]>>;
}

/**
 * Adapt callback-shaped control code to a snapshot transaction.
 *
 * With an `onApply` consumer, mapped callbacks only mark normalized keys; missing mapped callbacks
 * are synthesized so an external consumer needs to implement just `onApply`. Without `onApply`,
 * callbacks retain immediate legacy behavior while idle. A transaction containing legacy changes
 * without an `onApply` consumer fails explicitly rather than replaying or silently dropping them.
 * Immediate validators/actions always call the supplied callback directly.
 */
export function wrapFluidControlsCallbacks<T extends object, TKey extends PropertyKey>(
    callbacks: T,
    transaction: FluidControlsTransaction<TKey>,
    options: WrapFluidControlsCallbacksOptions<T, TKey>
): T {
    const immediate = new Set<string>(options.immediate ?? []);
    const names = new Set<string>(Object.keys(callbacks));
    if (transaction.appliesSnapshots) {
        for (const name of Object.keys(options.changedKeys)) {
            names.add(name);
        }
    }

    const wrapped: Record<string, unknown> = {};
    for (const name of names) {
        const value = (callbacks as Record<string, unknown>)[name];
        if (typeof value !== "function") {
            if (value !== undefined) {
                wrapped[name] = value;
            } else if (transaction.appliesSnapshots && options.changedKeys[name as keyof T & string] !== undefined) {
                wrapped[name] = (..._args: unknown[]): void => {
                    const configured = options.changedKeys[name as keyof T & string]!;
                    const keys = Array.isArray(configured) ? configured : [configured];
                    markFluidControlsChanged(transaction, ...(keys as readonly TKey[]));
                };
            }
            continue;
        }

        const call = value as (...args: unknown[]) => unknown;
        if (immediate.has(name)) {
            wrapped[name] = (...args: unknown[]): unknown => call(...args);
            continue;
        }
        const configured = options.changedKeys[name as keyof T & string];
        wrapped[name] = (...args: unknown[]): unknown => {
            if (configured !== undefined && (transaction.active || transaction.appliesSnapshots)) {
                const keys = Array.isArray(configured) ? configured : [configured];
                markFluidControlsChanged(transaction, ...(keys as readonly TKey[]));
                return undefined;
            }
            if (!transaction.active) {
                return call(...args);
            }
            return undefined;
        };
    }
    return wrapped as T;
}
