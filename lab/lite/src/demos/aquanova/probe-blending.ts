export type ProbeBlendVec3 = readonly [number, number, number];

export interface BoxProbeInfluence {
    readonly id: string;
    readonly centre: ProbeBlendVec3;
    /** Half extents where this probe still has full influence. */
    readonly innerHalfSize: ProbeBlendVec3;
    /** Half extents where this probe reaches zero influence. */
    readonly outerHalfSize: ProbeBlendVec3;
}

export interface BoxProbeRegion {
    readonly id: string;
    readonly projectionCentre: ProbeBlendVec3;
    readonly projectionHalfSize: ProbeBlendVec3;
}

export interface ProbeBlendWeight {
    readonly id: string;
    readonly weight: number;
    readonly ndf: number;
}

const EPSILON = 1e-6;

/**
 * Normalized distance field from Sébastien Lagarde's POI cubemap blending method:
 * <= 0 inside the inner box, 1 at the outer box, and > 1 outside it.
 */
export function boxProbeNdf(probe: BoxProbeInfluence, point: ProbeBlendVec3): number {
    let ndf = Number.NEGATIVE_INFINITY;
    for (let axis = 0; axis < 3; axis++) {
        const inner = probe.innerHalfSize[axis]!;
        const outer = probe.outerHalfSize[axis]!;
        const span = outer - inner;
        const distance = Math.abs(point[axis]! - probe.centre[axis]!);
        const axisNdf = span > EPSILON ? (distance - inner) / span : distance <= outer ? 0 : Number.POSITIVE_INFINITY;
        ndf = Math.max(ndf, axisNdf);
    }
    return ndf;
}

function volume(probe: BoxProbeInfluence): number {
    return probe.outerHalfSize[0] * probe.outerHalfSize[1] * probe.outerHalfSize[2] * 8;
}

/**
 * Resolve the probe whose authored parallax box contains the point.
 *
 * Influence volumes intentionally overlap to make blending smooth, but the non-blended fallback
 * must follow the physical room box instead of whichever influence weight happens to be larger.
 * If authored boxes overlap, the tighter box wins deterministically.
 */
export function selectContainingBoxProbe<T extends BoxProbeRegion>(probes: readonly T[], point: ProbeBlendVec3): T | undefined {
    let selected: T | undefined;
    let selectedVolume = Number.POSITIVE_INFINITY;
    for (const probe of probes) {
        let contains = true;
        for (let axis = 0; axis < 3; axis++) {
            if (Math.abs(point[axis]! - probe.projectionCentre[axis]!) > probe.projectionHalfSize[axis]! + EPSILON) {
                contains = false;
                break;
            }
        }
        if (!contains) continue;
        const boxVolume = probe.projectionHalfSize[0] * probe.projectionHalfSize[1] * probe.projectionHalfSize[2] * 8;
        if (!selected || boxVolume < selectedVolume || (boxVolume === selectedVolume && probe.id.localeCompare(selected.id) < 0)) {
            selected = probe;
            selectedVolume = boxVolume;
        }
    }
    return selected;
}

/**
 * Select and weight the probes affecting one point of interest.
 *
 * The returned order is stable by id so crossing the midpoint between two probes changes only the
 * weights, not the bound texture order. This module intentionally has no Babylon-Lite dependencies
 * so the editor can port the same data and math directly to Babylon.js.
 */
export function selectPoiProbeBlend(probes: readonly BoxProbeInfluence[], point: ProbeBlendVec3, maxProbes = 2): ProbeBlendWeight[] {
    if (maxProbes < 1 || probes.length === 0) return [];

    const ranked = probes
        .map((probe) => ({ probe, ndf: boxProbeNdf(probe, point) }))
        .sort((a, b) => a.ndf - b.ndf || volume(a.probe) - volume(b.probe) || a.probe.id.localeCompare(b.probe.id));

    const inner = ranked.find((candidate) => candidate.ndf <= 0);
    if (inner) return [{ id: inner.probe.id, weight: 1, ndf: inner.ndf }];

    let selected = ranked.filter((candidate) => candidate.ndf < 1).slice(0, maxProbes);
    if (selected.length === 0) selected = ranked.slice(0, 1);
    if (selected.length === 1) return [{ id: selected[0]!.probe.id, weight: 1, ndf: selected[0]!.ndf }];

    const clamped = selected.map((candidate) => Math.min(Math.max(candidate.ndf, 0), 1));
    const sumNdf = clamped.reduce((sum, value) => sum + value, 0);
    const sumInverseNdf = clamped.reduce((sum, value) => sum + 1 - value, 0);
    const count = selected.length;
    const weights = selected.map((_candidate, index) => {
        const ndf = clamped[index]!;
        const boundaryWeight = sumNdf > EPSILON ? (1 - ndf / sumNdf) / (count - 1) : 1 / count;
        const centreWeight = sumInverseNdf > EPSILON ? (1 - ndf) / sumInverseNdf : 1 / count;
        return Math.max(0, boundaryWeight * centreWeight);
    });
    const sumWeights = weights.reduce((sum, value) => sum + value, 0) || 1;

    return selected
        .map((candidate, index) => ({
            id: candidate.probe.id,
            weight: weights[index]! / sumWeights,
            ndf: candidate.ndf,
        }))
        .sort((a, b) => a.id.localeCompare(b.id));
}
