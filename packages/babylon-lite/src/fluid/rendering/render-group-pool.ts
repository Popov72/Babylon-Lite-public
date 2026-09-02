// Reference-counted keyed resource pool.
//
// A fluid host that renders simulations through shared render groups (buffers, output target and
// surface task keyed by render profile) needs those groups to live exactly as long as some
// simulation uses them. Without a reference count they accumulate for the life of the page: a group
// keeps its buffers plus a full-resolution target even after the last simulation using it is gone.
//
// This pool tracks how many live simulations hold each key. `acquire` reuses an existing value or
// builds one; `release` drops a reference and disposes the value when the count reaches zero. It is
// deliberately generic and host-agnostic: the disposal policy (which, for a render group, removes
// the host's frame-graph task and destroys buffers/targets) is supplied by the owner so this
// primitive carries no rendering or GPU coupling.

/** @internal */
export interface ReferenceCountedPool<K, V> {
    /** Reuse the value for `key` (incrementing its reference count) or build and register a new one. */
    acquire(key: K, create: () => V): V;
    /** Drop one reference to `key`. Disposes and evicts the value when the last reference is released. */
    release(key: K): boolean;
    has(key: K): boolean;
    /** Live reference count for `key` (0 when absent). */
    refCount(key: K): number;
    /** Number of distinct live values. */
    readonly size: number;
    /** Snapshot of live values, safe to iterate while acquiring/releasing. */
    values(): V[];
    /** Dispose and evict every value. Used on host teardown. */
    disposeAll(): void;
}

interface PoolEntry<V> {
    readonly value: V;
    refCount: number;
}

/** @internal */
export function createReferenceCountedPool<K, V>(dispose: (value: V) => void): ReferenceCountedPool<K, V> {
    const entries = new Map<K, PoolEntry<V>>();
    return {
        acquire(key, create) {
            const existing = entries.get(key);
            if (existing) {
                existing.refCount++;
                return existing.value;
            }
            const value = create();
            entries.set(key, { value, refCount: 1 });
            return value;
        },
        release(key) {
            const entry = entries.get(key);
            if (!entry) {
                return false;
            }
            entry.refCount--;
            if (entry.refCount > 0) {
                return false;
            }
            entries.delete(key);
            dispose(entry.value);
            return true;
        },
        has(key) {
            return entries.has(key);
        },
        refCount(key) {
            return entries.get(key)?.refCount ?? 0;
        },
        get size() {
            return entries.size;
        },
        values() {
            return [...entries.values()].map((entry) => entry.value);
        },
        disposeAll() {
            const live = [...entries.values()];
            entries.clear();
            for (const entry of live) {
                dispose(entry.value);
            }
        },
    };
}
