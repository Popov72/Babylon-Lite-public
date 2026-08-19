export type ProbeBlendVec3 = readonly [number, number, number];

export interface BoxProbeInfluence {
    readonly id: string;
    readonly centre: ProbeBlendVec3;
    /** Half extents where this probe still has full influence. */
    readonly innerHalfSize: ProbeBlendVec3;
    /** Half extents where this probe reaches zero influence. */
    readonly outerHalfSize: ProbeBlendVec3;
    /** Yaw in radians. Defaults to zero. */
    readonly angleRadians?: number;
}

export interface BoxProbeRegion {
    readonly id: string;
    readonly projectionCentre: ProbeBlendVec3;
    readonly projectionHalfSize: ProbeBlendVec3;
    /** Yaw in radians. Defaults to zero. */
    readonly angleRadians?: number;
}

export interface ProbeBlendWeight {
    readonly id: string;
    readonly weight: number;
    readonly ndf: number;
}

export interface ProbeWorldBounds {
    readonly centre: ProbeBlendVec3;
    readonly halfSize: ProbeBlendVec3;
}

const EPSILON = 1e-6;

function probeLocalOffset(point: ProbeBlendVec3, centre: ProbeBlendVec3, angleRadians = 0): ProbeBlendVec3 {
    const x = point[0] - centre[0];
    const y = point[1] - centre[1];
    const z = point[2] - centre[2];
    const cosine = Math.cos(angleRadians);
    const sine = Math.sin(angleRadians);
    return [cosine * x - sine * z, y, sine * x + cosine * z];
}

/**
 * Normalized distance field from Sébastien Lagarde's POI cubemap blending method:
 * <= 0 inside the inner box, 1 at the outer box, and > 1 outside it.
 */
export function boxProbeNdf(probe: BoxProbeInfluence, point: ProbeBlendVec3): number {
    const local = probeLocalOffset(point, probe.centre, probe.angleRadians);
    let ndf = Number.NEGATIVE_INFINITY;
    for (let axis = 0; axis < 3; axis++) {
        const inner = probe.innerHalfSize[axis]!;
        const outer = probe.outerHalfSize[axis]!;
        const span = outer - inner;
        const distance = Math.abs(local[axis]!);
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
        const local = probeLocalOffset(point, probe.projectionCentre, probe.angleRadians);
        let contains = true;
        for (let axis = 0; axis < 3; axis++) {
            if (Math.abs(local[axis]!) > probe.projectionHalfSize[axis]! + EPSILON) {
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

function projectionVolume(probe: BoxProbeRegion): number {
    return probe.projectionHalfSize[0] * probe.projectionHalfSize[1] * probe.projectionHalfSize[2] * 8;
}

function projectionDistanceSquared(probe: BoxProbeRegion, point: ProbeBlendVec3): number {
    const local = probeLocalOffset(point, probe.projectionCentre, probe.angleRadians);
    let distanceSquared = 0;
    for (let axis = 0; axis < 3; axis++) {
        const outside = Math.max(0, Math.abs(local[axis]!) - probe.projectionHalfSize[axis]!);
        distanceSquared += outside * outside;
    }
    return distanceSquared;
}

/** Exact intersection between a world AABB and a probe box with yaw-only orientation. */
export function intersectsProbeProjectionBox(probe: BoxProbeRegion, bounds: ProbeWorldBounds): boolean {
    const dx = bounds.centre[0] - probe.projectionCentre[0];
    const dy = bounds.centre[1] - probe.projectionCentre[1];
    const dz = bounds.centre[2] - probe.projectionCentre[2];
    const probeHalf = probe.projectionHalfSize;
    if (Math.abs(dy) > bounds.halfSize[1] + probeHalf[1] + EPSILON) {
        return false;
    }

    const cosine = Math.cos(probe.angleRadians ?? 0);
    const sine = Math.sin(probe.angleRadians ?? 0);
    const absCosine = Math.abs(cosine);
    const absSine = Math.abs(sine);
    if (Math.abs(dx) > bounds.halfSize[0] + absCosine * probeHalf[0] + absSine * probeHalf[2] + EPSILON) {
        return false;
    }
    if (Math.abs(dz) > bounds.halfSize[2] + absSine * probeHalf[0] + absCosine * probeHalf[2] + EPSILON) {
        return false;
    }

    const localX = cosine * dx - sine * dz;
    const localZ = sine * dx + cosine * dz;
    if (Math.abs(localX) > probeHalf[0] + absCosine * bounds.halfSize[0] + absSine * bounds.halfSize[2] + EPSILON) {
        return false;
    }
    return Math.abs(localZ) <= probeHalf[2] + absSine * bounds.halfSize[0] + absCosine * bounds.halfSize[2] + EPSILON;
}

/**
 * Resolve one immutable probe assignment for a mesh.
 *
 * A containing box wins first. Otherwise the nearest intersecting box wins. Geometry outside every
 * authored box falls back to the closest box so every PBR mesh still receives deterministic IBL.
 */
export function selectStaticBoxProbe<T extends BoxProbeRegion>(probes: readonly T[], bounds: ProbeWorldBounds): T | undefined {
    const containing = selectContainingBoxProbe(probes, bounds.centre);
    if (containing) {
        return containing;
    }
    return probes
        .map((probe) => ({
            probe,
            intersects: intersectsProbeProjectionBox(probe, bounds),
            distanceSquared: projectionDistanceSquared(probe, bounds.centre),
        }))
        .sort(
            (a, b) =>
                Number(b.intersects) - Number(a.intersects) ||
                a.distanceSquared - b.distanceSquared ||
                projectionVolume(a.probe) - projectionVolume(b.probe) ||
                a.probe.id.localeCompare(b.probe.id)
        )[0]?.probe;
}

/** Rank a conservative shader candidate set by POI distance without calculating shader weights. */
export function selectPoiProbeCandidates(probes: readonly BoxProbeInfluence[], point: ProbeBlendVec3, maxProbes: number): number[] {
    if (maxProbes < 1) return [];
    return probes
        .map((probe, index) => ({ index, probe, ndf: boxProbeNdf(probe, point) }))
        .sort((a, b) => a.ndf - b.ndf || volume(a.probe) - volume(b.probe) || a.probe.id.localeCompare(b.probe.id))
        .slice(0, maxProbes)
        .map((candidate) => candidate.index);
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
