// Uniform spatial-hash neighbor search for the SPH relaxers.
//
// Mirrors the role of CompactNSearch in the SPlisHSPlasH tool: bins particles into a hash grid
// with cell size = support radius (4*particleRadius) and enumerates, for each particle, every
// OTHER particle within the support radius (searching the 27 surrounding cells). Self is excluded
// (density adds the W(0) self term separately). Rebuilt each relaxation step because positions
// mutate in place; all backing arrays are reused across rebuilds (only the bins are recomputed).
//
// Hash collisions are handled by storing each particle's integer cell and, while walking a bucket
// for a target cell, only accepting particles whose stored cell equals that target. This makes
// each real neighbor visited exactly once even when two of the 27 cells collide to the same bucket.
//
// Pure CPU — no GPU handles, zero module-level side effects.

/** A rebuildable neighbor query over a fixed particle set. */
export interface Neighborhood {
    /** Re-bin all particles from their current positions (call once per step before querying). */
    rebuild(): void;
    /** Invoke `cb` for every particle `j != i` within the support radius. `dx=xi-xj`, `r2=|xi-xj|^2`. */
    forEachNeighbor(i: number, cb: (j: number, dx: number, dy: number, dz: number, r2: number) => void): void;
}

/**
 * Build a spatial-hash neighborhood over `count` particles stored in `positions` (flat xyz).
 *
 * @param positions - live flat xyz particle positions (read on every `rebuild`).
 * @param count - particle count.
 * @param supportRadius - neighbor cutoff and cell size.
 * @returns a {@link Neighborhood}.
 */
export function buildNeighborhood(positions: Float32Array, count: number, supportRadius: number): Neighborhood {
    const cellSize = supportRadius;
    const invCell = 1.0 / cellSize;
    const r2Max = supportRadius * supportRadius;

    // Bucket table sized to a power of two >= 2*count (load factor ~0.5) with a small floor.
    let nb = 16;
    while (nb < count * 2) {
        nb <<= 1;
    }
    const mask = nb - 1;

    const head = new Int32Array(nb); // bucket -> first particle index (or -1)
    const next = new Int32Array(Math.max(1, count)); // particle -> next particle in its bucket
    const cellX = new Int32Array(Math.max(1, count));
    const cellY = new Int32Array(Math.max(1, count));
    const cellZ = new Int32Array(Math.max(1, count));

    const hash = (ix: number, iy: number, iz: number): number => (Math.imul(ix, 73856093) ^ Math.imul(iy, 19349663) ^ Math.imul(iz, 83492791)) & mask;

    const rebuild = (): void => {
        head.fill(-1);
        for (let i = 0; i < count; i++) {
            const ix = Math.floor(positions[3 * i]! * invCell);
            const iy = Math.floor(positions[3 * i + 1]! * invCell);
            const iz = Math.floor(positions[3 * i + 2]! * invCell);
            cellX[i] = ix;
            cellY[i] = iy;
            cellZ[i] = iz;
            const b = hash(ix, iy, iz);
            next[i] = head[b]!;
            head[b] = i;
        }
    };

    const forEachNeighbor = (i: number, cb: (j: number, dx: number, dy: number, dz: number, r2: number) => void): void => {
        const xi = positions[3 * i]!;
        const yi = positions[3 * i + 1]!;
        const zi = positions[3 * i + 2]!;
        const bx = cellX[i]!;
        const by = cellY[i]!;
        const bz = cellZ[i]!;
        for (let oz = -1; oz <= 1; oz++) {
            const tz = bz + oz;
            for (let oy = -1; oy <= 1; oy++) {
                const ty = by + oy;
                for (let ox = -1; ox <= 1; ox++) {
                    const tx = bx + ox;
                    let j = head[hash(tx, ty, tz)]!;
                    while (j !== -1) {
                        // Guard against hash collisions: only accept particles actually in this cell.
                        if (j !== i && cellX[j] === tx && cellY[j] === ty && cellZ[j] === tz) {
                            const dx = xi - positions[3 * j]!;
                            const dy = yi - positions[3 * j + 1]!;
                            const dz = zi - positions[3 * j + 2]!;
                            const r2 = dx * dx + dy * dy + dz * dz;
                            if (r2 <= r2Max) {
                                cb(j, dx, dy, dz, r2);
                            }
                        }
                        j = next[j]!;
                    }
                }
            }
        }
    };

    return { rebuild, forEachNeighbor };
}
