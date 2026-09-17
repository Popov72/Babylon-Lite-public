import type { ParticleComparison, ParticleMetrics, ParticleState, ReferenceGrid, Triple } from "./types";

function histogram(state: ParticleState, grid: ReferenceGrid): Map<string, number> {
    const result = new Map<string, number>();
    for (let i = 0; i < state.positions.length; i += 3) {
        const x = Math.floor((state.positions[i]! - grid.origin[0]) / grid.cellSize);
        const y = Math.floor((state.positions[i + 1]! - grid.origin[1]) / grid.cellSize);
        const z = Math.floor((state.positions[i + 2]! - grid.origin[2]) / grid.cellSize);
        const key = `${x},${y},${z}`;
        result.set(key, (result.get(key) ?? 0) + 1);
    }
    return result;
}

export function measureParticles(state: ParticleState, grid: ReferenceGrid): ParticleMetrics {
    if (state.positions.length % 3 !== 0 || state.positions.length !== state.velocities.length) {
        throw new Error("Particle metrics require aligned XYZ positions and velocities.");
    }
    const count = state.positions.length / 3;
    if (count === 0) {
        return {
            count,
            center: null,
            lower: null,
            upper: null,
            p01: null,
            p99: null,
            meanSpeed: null,
            meanKineticEnergy: null,
            occupiedCells: 0,
            outsideParticles: 0,
            heightByZ: new Array<number | null>(grid.dimensions[2]).fill(null),
        };
    }
    const axes = [new Float32Array(count), new Float32Array(count), new Float32Array(count)];
    const center: Triple = [0, 0, 0];
    const lower: Triple = [Infinity, Infinity, Infinity];
    const upper: Triple = [-Infinity, -Infinity, -Infinity];
    const heights = Array.from({ length: grid.dimensions[2] }, () => [] as number[]);
    let meanSpeed = 0;
    let meanKineticEnergy = 0;
    let outsideParticles = 0;
    for (let p = 0; p < count; p++) {
        let speedSquared = 0;
        let outside = false;
        for (let a = 0; a < 3; a++) {
            const x = state.positions[3 * p + a]!;
            const v = state.velocities[3 * p + a]!;
            if (!Number.isFinite(x) || !Number.isFinite(v)) {
                throw new Error(`Nonfinite particle state at marker ${p}, axis ${a}.`);
            }
            center[a]! += x / count;
            lower[a] = Math.min(lower[a]!, x);
            upper[a] = Math.max(upper[a]!, x);
            axes[a]![p] = x;
            speedSquared += v * v;
            outside ||= x < grid.origin[a]! || x > grid.origin[a]! + grid.dimensions[a]! * grid.cellSize;
        }
        outsideParticles += Number(outside);
        meanSpeed += Math.sqrt(speedSquared) / count;
        meanKineticEnergy += (0.5 * speedSquared) / count;
        const z = Math.floor((state.positions[3 * p + 2]! - grid.origin[2]) / grid.cellSize);
        heights[z]?.push(state.positions[3 * p + 1]!);
    }
    for (const axis of axes) {
        axis.sort();
    }
    const p01 = axes.map((a) => a[Math.floor(0.01 * (count - 1))]!) as Triple;
    const p99 = axes.map((a) => a[Math.floor(0.99 * (count - 1))]!) as Triple;
    return {
        count,
        center,
        lower,
        upper,
        p01,
        p99,
        meanSpeed,
        meanKineticEnergy,
        occupiedCells: histogram(state, grid).size,
        outsideParticles,
        heightByZ: heights.map((h) => (h.length ? h.sort((a, b) => a - b)[Math.floor(0.99 * (h.length - 1))]! : null)),
    };
}

export function compareParticles(actual: ParticleState, reference: ParticleState, grid: ReferenceGrid): ParticleComparison {
    const a = histogram(actual, grid);
    const b = histogram(reference, grid);
    const na = actual.positions.length / 3;
    const nb = reference.positions.length / 3;
    if (na === 0 || nb === 0) {
        return {
            centerDistance: null,
            occupancyIntersectionOverUnion: na === nb ? 1 : 0,
            normalizedOccupancyDifference: na === nb ? 0 : 1,
            countDifference: na - nb,
        };
    }
    let intersection = 0;
    let difference = 0;
    for (const [key, count] of a) {
        const other = b.get(key) ?? 0;
        intersection += Number(other > 0);
        difference += Math.abs(count / na - other / nb);
    }
    for (const [key, count] of b) {
        if (!a.has(key)) {
            difference += count / nb;
        }
    }
    const center = (s: ParticleState, axis: number): number => {
        let total = 0;
        for (let i = axis; i < s.positions.length; i += 3) {
            total += s.positions[i]!;
        }
        return total / (s.positions.length / 3);
    };
    return {
        centerDistance: Math.hypot(...[0, 1, 2].map((axis) => center(actual, axis) - center(reference, axis))),
        occupancyIntersectionOverUnion: intersection / (a.size + b.size - intersection),
        normalizedOccupancyDifference: difference * 0.5,
        countDifference: na - nb,
    };
}
