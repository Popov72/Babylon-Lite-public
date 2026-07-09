// Deterministic lattice sampling — VolumeSampling modes 0/1/2.
//
// Port of the SPlisHSPlasH `SamplingBase::sampleObject` (Tools/VolumeSampling). Iterates a
// regular / hexagonal / hexagonal-close-packed lattice over the mesh bounding box and keeps
// each candidate whose signed-distance grid value is > 0 (i.e. inside the mesh). The lattice
// is walked z-major (step = diameter), y-mid (step = yshift), x-inner (step = xshift), with
// `counterX` reset each y-row and `counterY` reset each z-layer.
//
// Pure CPU — no GPU handles, zero module-level side effects.

import type { SignedDistanceGrid } from "./mesh-sdf.js";

/**
 * Sample a deterministic particle lattice inside the mesh.
 *
 * @param sdf - inverted signed-distance grid (inside is `> 0`).
 * @param bbMin - lattice bounding-box minimum corner.
 * @param bbMax - lattice bounding-box maximum corner.
 * @param radius - particle radius (diameter = 2*radius).
 * @param mode - 0 regular / 1 almost-dense / 2 dense HCP.
 * @returns a flat array of accepted particle positions as xyz triples.
 */
export function sampleLattice(sdf: SignedDistanceGrid, bbMin: [number, number, number], bbMax: [number, number, number], radius: number, mode: number): number[] {
    const diameter = 2 * radius;

    let xshift = diameter;
    let yshift = diameter;
    if (mode === 1) {
        yshift = Math.sqrt(3) * radius;
    } else if (mode === 2) {
        xshift = Math.sqrt(3) * radius;
        yshift = (Math.sqrt(6) * diameter) / 3;
    }

    const out: number[] = [];
    for (let z = bbMin[2]; z <= bbMax[2]; z += diameter) {
        let counterY = 0;
        for (let y = bbMin[1]; y <= bbMax[1]; y += yshift) {
            let counterX = 0;
            for (let x = bbMin[0]; x <= bbMax[0]; x += xshift) {
                let px: number;
                let py: number;
                let pz: number;
                if (mode === 0) {
                    px = x + radius;
                    py = y + radius;
                    pz = z + radius;
                } else if (mode === 1) {
                    if (counterY % 2 === 0) {
                        px = x;
                        py = y + radius;
                        pz = z + radius;
                    } else {
                        px = x + radius;
                        py = y + radius;
                        pz = z;
                    }
                } else {
                    // mode 2 — dense HCP: base position + counter-driven shift vector.
                    px = x;
                    py = y + radius;
                    pz = z + radius;
                    let sz = 0;
                    if (counterX % 2 === 1) {
                        sz += diameter / (2 * (counterY % 2 === 1 ? -1 : 1));
                    }
                    if (counterY % 2 === 1) {
                        px += xshift / 2;
                        sz += diameter / 2;
                    }
                    pz += sz;
                }

                // Accept iff inside the mesh. `interpolate` returns +Infinity outside the domain;
                // guard against it so the (finite) `> 0` inside test matches the C++
                // `dist != Infinity && dist > 0`.
                const d = sdf.interpolate(px, py, pz);
                if (Number.isFinite(d) && d > 0) {
                    out.push(px, py, pz);
                }
                counterX++;
            }
            counterY++;
        }
    }
    return out;
}
