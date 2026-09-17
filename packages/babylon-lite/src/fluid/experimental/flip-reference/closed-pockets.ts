/** @internal Marks multi-cell liquid pockets without an air connection through an aperture of at least 1e-6. */
export function markFlipReferenceClosedPockets(
    rows: Float32Array,
    dimensions: readonly [number, number, number],
    visited: Uint8Array,
    queue: Int32Array
): { components: number; cells: number } {
    const count = dimensions[0] * dimensions[1] * dimensions[2];
    const strides = [1, dimensions[0], dimensions[0] * dimensions[1]];
    const threshold = Math.fround(1e-6);
    const result = { components: 0, cells: 0 };
    visited.fill(0);
    for (let i = 0; i < count; i++) {
        rows[i * 16 + 14] = 0;
    }
    for (let root = 0; root < count; root++) {
        if (rows[root * 16 + 3] === 0 || visited[root]) {
            continue;
        }
        let head = 0;
        let tail = 1;
        let hasAir = false;
        queue[0] = root;
        visited[root] = 1;
        while (head < tail) {
            const index = queue[head++]!;
            const offset = index * 16;
            hasAir ||= rows[offset + 13] !== 0;
            for (let axis = 0; axis < 3; axis++) {
                for (let sign = -1; sign <= 1; sign += 2) {
                    if (rows[offset + (sign > 0 ? 4 : 8) + axis]! >= threshold) {
                        const neighbor = index + sign * strides[axis]!;
                        if (!visited[neighbor]) {
                            visited[neighbor] = 1;
                            queue[tail++] = neighbor;
                        }
                    }
                }
            }
        }
        if (tail > 1 && !hasAir) {
            result.components++;
            result.cells += tail;
            for (let i = 0; i < tail; i++) {
                rows[queue[i]! * 16 + 14] = 1;
            }
        }
    }
    return result;
}
