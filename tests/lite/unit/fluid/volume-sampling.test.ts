import { describe, expect, it } from "vitest";

import { buildMeshDistance, buildSignedDistanceGrid, createVolumeSampler, sampleMeshVolume } from "../../../../packages/babylon-lite/src/fluid/volume-sampling/index";
import { createCubicKernel } from "../../../../packages/babylon-lite/src/fluid/volume-sampling/sph-kernels";
import { buildNeighborhood } from "../../../../packages/babylon-lite/src/fluid/volume-sampling/neighborhood";

// ------------------------------------------------------------------
// Analytic test meshes (built in-code so the tests are self-contained).
// ------------------------------------------------------------------

interface Mesh {
    positions: Float32Array;
    indices: Uint32Array;
}

// Axis-aligned box (8 verts, 12 tris). Winding is fixed to outward below.
function makeBox(min: [number, number, number], max: [number, number, number]): Mesh {
    const [x0, y0, z0] = min;
    const [x1, y1, z1] = max;
    const positions = new Float32Array([
        x0,
        y0,
        z0, // 0
        x1,
        y0,
        z0, // 1
        x1,
        y1,
        z0, // 2
        x0,
        y1,
        z0, // 3
        x0,
        y0,
        z1, // 4
        x1,
        y0,
        z1, // 5
        x1,
        y1,
        z1, // 6
        x0,
        y1,
        z1, // 7
    ]);
    // Two triangles per face.
    const indices = new Uint32Array([
        1,
        2,
        6,
        1,
        6,
        5, // +X
        0,
        4,
        7,
        0,
        7,
        3, // -X
        3,
        7,
        6,
        3,
        6,
        2, // +Y
        0,
        1,
        5,
        0,
        5,
        4, // -Y
        4,
        5,
        6,
        4,
        6,
        7, // +Z
        0,
        3,
        2,
        0,
        2,
        1, // -Z
    ]);
    const center: [number, number, number] = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2];
    fixOutwardWinding(positions, indices, center);
    return { positions, indices };
}

// UV sphere centered at the origin (convex -> outward winding fixed below).
function makeUvSphere(radius: number, stacks: number, slices: number): Mesh {
    const verts: number[] = [];
    for (let i = 0; i <= stacks; i++) {
        const theta = (i / stacks) * Math.PI;
        const st = Math.sin(theta);
        const ct = Math.cos(theta);
        for (let j = 0; j <= slices; j++) {
            const phi = (j / slices) * 2 * Math.PI;
            verts.push(radius * st * Math.cos(phi), radius * ct, radius * st * Math.sin(phi));
        }
    }
    const idx: number[] = [];
    const row = slices + 1;
    for (let i = 0; i < stacks; i++) {
        for (let j = 0; j < slices; j++) {
            const a = i * row + j;
            const b = a + row;
            idx.push(a, b, b + 1, a, b + 1, a + 1);
        }
    }
    const positions = new Float32Array(verts);
    const indices = new Uint32Array(idx);
    fixOutwardWinding(positions, indices, [0, 0, 0]);
    return { positions, indices };
}

// For a convex mesh, flip any triangle whose normal points toward `center`.
function fixOutwardWinding(positions: Float32Array, indices: Uint32Array, center: [number, number, number]): void {
    for (let t = 0; t < indices.length / 3; t++) {
        const i0 = indices[3 * t]!;
        const i1 = indices[3 * t + 1]!;
        const i2 = indices[3 * t + 2]!;
        const ax = positions[3 * i0]!;
        const ay = positions[3 * i0 + 1]!;
        const az = positions[3 * i0 + 2]!;
        const bx = positions[3 * i1]!;
        const by = positions[3 * i1 + 1]!;
        const bz = positions[3 * i1 + 2]!;
        const cx = positions[3 * i2]!;
        const cy = positions[3 * i2 + 1]!;
        const cz = positions[3 * i2 + 2]!;
        const nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
        const ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
        const nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        // Outward direction ~ centroid - center.
        const ox = (ax + bx + cx) / 3 - center[0];
        const oy = (ay + by + cy) / 3 - center[1];
        const oz = (az + bz + cz) / 3 - center[2];
        if (nx * ox + ny * oy + nz * oz < 0) {
            indices[3 * t + 1] = i2;
            indices[3 * t + 2] = i1;
        }
    }
}

// Small deterministic PRNG for reproducible random query points.
function makeRng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
    };
}

// A box with ONE face (the +Y "top", triangles 4 & 5) removed -> an OPEN, non-manifold shell.
// Pseudonormals give the wrong inside/outside sign in the exterior above the opening; the winding
// number does not. This is the regression fixture for the "spurious exterior fill" bug.
function makeOpenTopBox(min: [number, number, number], max: [number, number, number]): Mesh {
    const box = makeBox(min, max);
    const keep: number[] = [];
    for (let t = 0; t < box.indices.length / 3; t++) {
        if (t === 4 || t === 5) {
            continue; // +Y face is the 3rd face -> triangles 4 and 5.
        }
        keep.push(box.indices[3 * t]!, box.indices[3 * t + 1]!, box.indices[3 * t + 2]!);
    }
    return { positions: box.positions, indices: new Uint32Array(keep) };
}

describe("volume-sampling: mesh signed distance", () => {
    it("box: correct interior/exterior signs and origin inside", () => {
        const { positions, indices } = makeBox([-1, -1, -1], [1, 1, 1]);
        const md = buildMeshDistance(positions, indices);

        // Origin is inside -> negative, distance ~= 1 to the nearest face.
        const o = md.signedDistance(0, 0, 0);
        expect(o.distance).toBeLessThan(0);
        expect(Math.abs(o.distance)).toBeCloseTo(1, 5);

        // A few interior points -> negative.
        for (const p of [
            [0.5, 0.5, 0.5],
            [-0.9, 0.2, -0.3],
            [0, 0.99, 0],
        ] as const) {
            expect(md.signedDistance(p[0], p[1], p[2]).distance).toBeLessThan(0);
        }

        // Far / exterior points -> positive.
        expect(md.signedDistance(5, 5, 5).distance).toBeGreaterThan(0);
        expect(md.signedDistance(1.5, 0, 0).distance).toBeGreaterThan(0);
    });

    it("sphere: correct interior/exterior signs", () => {
        const { positions, indices } = makeUvSphere(1, 16, 24);
        const md = buildMeshDistance(positions, indices);

        expect(md.signedDistance(0, 0, 0).distance).toBeLessThan(0);
        expect(md.signedDistance(0, 0.5, 0).distance).toBeLessThan(0);
        expect(md.signedDistance(0, 2, 0).distance).toBeGreaterThan(0);
        expect(md.signedDistance(3, 0, 0).distance).toBeGreaterThan(0);
    });

    it("BVH matches brute force on random query points (both meshes)", () => {
        const meshes = [makeBox([-1, -1, -1], [1, 1, 1]), makeUvSphere(1, 12, 18)];
        const rng = makeRng(12345);
        for (const { positions, indices } of meshes) {
            const md = buildMeshDistance(positions, indices);
            for (let n = 0; n < 200; n++) {
                const px = (rng() - 0.5) * 4;
                const py = (rng() - 0.5) * 4;
                const pz = (rng() - 0.5) * 4;
                const a = md.signedDistance(px, py, pz);
                const b = md.signedDistanceBrute(px, py, pz);
                expect(a.distance).toBeCloseTo(b.distance, 6);
                expect(a.cx).toBeCloseTo(b.cx, 6);
                expect(a.cy).toBeCloseTo(b.cy, 6);
                expect(a.cz).toBeCloseTo(b.cz, 6);
            }
        }
    });
});

describe("volume-sampling: winding number", () => {
    // A closed manifold has generalized winding number ~±1 inside and ~0 outside; the BVH-accelerated
    // fast path (dipole far-field + exact solid-angle leaves) must agree with the brute exact sum.
    for (const [name, mesh] of [
        ["box", makeBox([-1, -1, -1], [1, 1, 1])],
        ["sphere", makeUvSphere(1, 16, 24)],
    ] as const) {
        it(`${name}: |w|~1 inside, ~0 far outside, fast matches brute`, () => {
            const md = buildMeshDistance(mesh.positions, mesh.indices);

            // Interior points: |w| ~ 1.
            for (const p of [
                [0, 0, 0],
                [0.3, -0.2, 0.1],
                [-0.1, 0.4, -0.25],
            ] as const) {
                expect(Math.abs(md.windingNumber(p[0], p[1], p[2]))).toBeGreaterThan(0.9);
                expect(Math.abs(md.windingNumber(p[0], p[1], p[2]))).toBeLessThan(1.1);
            }

            // Far exterior: |w| ~ 0.
            expect(Math.abs(md.windingNumber(6, 6, 6))).toBeLessThan(0.02);
            expect(Math.abs(md.windingNumber(-5, 3, 4))).toBeLessThan(0.02);

            // Fast vs brute at random points a bit away from the surface (gate on |distance| > 0.25).
            const rng = makeRng(24680);
            let checked = 0;
            for (let n = 0; n < 400; n++) {
                const px = (rng() - 0.5) * 6;
                const py = (rng() - 0.5) * 6;
                const pz = (rng() - 0.5) * 6;
                if (Math.abs(md.signedDistance(px, py, pz).distance) <= 0.25) {
                    continue; // skip near-surface, where the winding number is noisy.
                }
                expect(Math.abs(md.windingNumber(px, py, pz) - md.windingNumberBrute(px, py, pz))).toBeLessThan(0.02);
                checked++;
            }
            expect(checked).toBeGreaterThan(50);
        });
    }
});

describe("volume-sampling: signed-distance grid", () => {
    it("box grid: inside is positive, far outside is <= 0 or Infinity", () => {
        const { positions, indices } = makeBox([-1, -1, -1], [1, 1, 1]);
        const md = buildMeshDistance(positions, indices);
        const r = 0.1;
        const pad = r * 4;
        const grid = buildSignedDistanceGrid(md, [-1 - pad, -1 - pad, -1 - pad], [1 + pad, 1 + pad, 1 + pad], [30, 30, 30], false);

        expect(grid.interpolate(0, 0, 0)).toBeGreaterThan(0);

        // Inside the domain but outside the mesh -> non-positive.
        const insideDomainOutsideMesh = grid.interpolate(1.25, 0, 0);
        expect(insideDomainOutsideMesh).toBeLessThanOrEqual(0);

        // Well outside the domain -> Infinity.
        const farOutside = grid.interpolate(10, 10, 10);
        expect(farOutside === Infinity || farOutside <= 0).toBe(true);
    });

    it("grid exposes raw fields and gradient/normal helpers", () => {
        const { positions, indices } = makeBox([-1, -1, -1], [1, 1, 1]);
        const md = buildMeshDistance(positions, indices);
        const grid = buildSignedDistanceGrid(md, [-1.4, -1.4, -1.4], [1.4, 1.4, 1.4], [30, 30, 30], false);

        expect(grid.res).toEqual([30, 30, 30]);
        expect(grid.data.length).toBe(31 * 31 * 31);
        expect(grid.cellSize[0]).toBeCloseTo(2.8 / 30, 6);

        const g = grid.interpolateWithGradient(0.9, 0, 0);
        expect(Number.isFinite(g.value)).toBe(true);
        // Gradient of the inverted (inside-positive) field points inward near the +X face.
        expect(g.gx).toBeLessThan(0);

        const dn = grid.distanceWithNormal(0.9, 0, 0, 0);
        expect(Number.isFinite(dn.dist)).toBe(true);
        expect(Math.hypot(dn.nx, dn.ny, dn.nz)).toBeCloseTo(1, 5);
    });
});

describe("volume-sampling: lattice sampling", () => {
    for (const mode of ["regular", "dense"] as const) {
        it(`box mode "${mode}": every particle inside, within AABB, sane count`, () => {
            const { positions, indices } = makeBox([-1, -1, -1], [1, 1, 1]);
            const radius = 0.1;
            const res = sampleMeshVolume({ positions, indices, radius, mode });

            expect(res.count).toBeGreaterThan(0);
            expect(res.positions.length).toBe(res.count * 3);

            const tol = 2 * radius;
            for (let i = 0; i < res.positions.length; i += 3) {
                const px = res.positions[i]!;
                const py = res.positions[i + 1]!;
                const pz = res.positions[i + 2]!;
                // Inside the mesh.
                expect(res.sdf.interpolate(px, py, pz)).toBeGreaterThan(0);
                // Within the AABB (allow a particle-radius margin for HCP shifts).
                expect(px).toBeGreaterThanOrEqual(res.bounds.min[0] - tol);
                expect(px).toBeLessThanOrEqual(res.bounds.max[0] + tol);
                expect(py).toBeGreaterThanOrEqual(res.bounds.min[1] - tol);
                expect(py).toBeLessThanOrEqual(res.bounds.max[1] + tol);
                expect(pz).toBeGreaterThanOrEqual(res.bounds.min[2] - tol);
                expect(pz).toBeLessThanOrEqual(res.bounds.max[2] + tol);
            }

            // Order-of-magnitude count check: volume / (2r)^3.
            const volume = 2 * 2 * 2;
            const expected = volume / Math.pow(2 * radius, 3);
            expect(res.count).toBeGreaterThan(expected * 0.3);
            expect(res.count).toBeLessThan(expected * 2.0);
        });
    }

    it('sphere mode "dense": every particle inside, sane count', () => {
        const { positions, indices } = makeUvSphere(1, 20, 30);
        const radius = 0.1;
        const res = sampleMeshVolume({ positions, indices, radius, mode: "dense" });

        expect(res.count).toBeGreaterThan(0);
        for (let i = 0; i < res.positions.length; i += 3) {
            expect(res.sdf.interpolate(res.positions[i]!, res.positions[i + 1]!, res.positions[i + 2]!)).toBeGreaterThan(0);
        }

        const volume = (4 / 3) * Math.PI;
        const expected = volume / Math.pow(2 * radius, 3);
        expect(res.count).toBeGreaterThan(expected * 0.3);
        expect(res.count).toBeLessThan(expected * 2.0);
    });

    it("is deterministic: two runs give identical counts and positions", () => {
        const { positions, indices } = makeBox([-1, -1, -1], [1, 1, 1]);
        const a = sampleMeshVolume({ positions, indices, radius: 0.12, mode: "dense" });
        const b = sampleMeshVolume({ positions, indices, radius: 0.12, mode: "dense" });
        expect(a.count).toBe(b.count);
        expect(a.positions.length).toBe(b.positions.length);
        for (let i = 0; i < a.positions.length; i++) {
            expect(a.positions[i]).toBe(b.positions[i]);
        }
    });

    it("is winding-robust: reversed triangle winding fills the interior, not the exterior", () => {
        // The SDF sign now comes from the generalized winding number: |w| > 0.5 == inside, which is
        // independent of triangle orientation. A mesh wound the "other way" (procedural factory
        // meshes and glTF can be either) must still fill the INTERIOR (|w| ~ 1 there regardless of
        // sign) — this replaces the old known-outside domain-corner auto-orient hack.
        const outward = makeBox([-1, -1, -1], [1, 1, 1]);
        const reversed = { positions: outward.positions, indices: outward.indices.slice() };
        for (let t = 0; t < reversed.indices.length / 3; t++) {
            const tmp = reversed.indices[3 * t + 1]!;
            reversed.indices[3 * t + 1] = reversed.indices[3 * t + 2]!;
            reversed.indices[3 * t + 2] = tmp;
        }
        const a = sampleMeshVolume({ positions: outward.positions, indices: outward.indices, radius: 0.12, mode: "dense" });
        const b = sampleMeshVolume({ positions: reversed.positions, indices: reversed.indices, radius: 0.12, mode: "dense" });
        // Same interior fill regardless of winding (a full [-1,1]^3 box, ~identical counts).
        expect(b.count).toBe(a.count);
        expect(b.count).toBeGreaterThan(0);
        for (let i = 0; i < b.positions.length; i += 3) {
            expect(b.sdf.interpolate(b.positions[i]!, b.positions[i + 1]!, b.positions[i + 2]!)).toBeGreaterThan(0);
        }
    });

    it("is open-mesh robust: an OPEN shell does NOT fill the empty exterior above the opening", () => {
        // Regression guard for the reported bug: sampling an OPEN / non-manifold mesh generated
        // spurious particles in empty exterior space (mostly ABOVE the model), because the SDF sign
        // came from angle-weighted pseudonormals (only valid for closed manifolds). The winding
        // number fixes the exterior sign, so the exterior stays empty.
        const { positions, indices } = makeOpenTopBox([-1, -1, -1], [1, 1, 1]);
        const radius = 0.1;
        const res = sampleMeshVolume({ positions, indices, radius, mode: "dense" });
        expect(res.count).toBeGreaterThan(0);

        // A deep-interior point (far below the open top) is inside.
        expect(res.sdf.interpolate(0, -0.7, 0)).toBeGreaterThan(0);

        // Points ABOVE the opening (still inside the padded domain) must read as OUTSIDE (<= 0),
        // NOT inside — this is exactly where the old pseudonormal sign filled the exterior.
        expect(res.sdf.interpolate(0, 1.2, 0)).toBeLessThanOrEqual(0);
        expect(res.sdf.interpolate(0.6, 1.2, 0.6)).toBeLessThanOrEqual(0);

        // Direct guard: no particle sits above the model's top face (y = 1); the exterior is empty.
        for (let i = 0; i < res.count; i++) {
            expect(res.positions[3 * i + 1]!).toBeLessThan(1 + radius);
        }
    });
});

describe("volume-sampling: SPH relaxation modes", () => {
    // Cubic-kernel SPH density estimate (mirrors mode 4's computeDensities) for test assertions.
    function cubicDensityStats(positions: Float32Array, count: number, radius: number, density0 = 1000): { meanAbsError: number; mean: number } {
        const support = 4 * radius;
        const mass = density0 * Math.pow(2 * radius, 3);
        const k = createCubicKernel(support);
        const nb = buildNeighborhood(positions, count, support);
        nb.rebuild();
        let sumErr = 0;
        let sumD = 0;
        for (let i = 0; i < count; i++) {
            let d = k.wZero;
            nb.forEachNeighbor(i, (_j, _dx, _dy, _dz, r2) => {
                d += k.W(Math.sqrt(r2));
            });
            d *= mass;
            sumErr += Math.abs(d - density0);
            sumD += d;
        }
        return { meanAbsError: sumErr / count, mean: sumD / count };
    }

    for (const mode of ["kugelstadt2021", "jiang2015"] as const) {
        it(`mode "${mode}": relaxes without throwing, preserves count, stays finite & inside`, () => {
            const { positions, indices } = makeBox([-1, -1, -1], [1, 1, 1]);
            const radius = 0.2;
            const steps = 20;

            const sampler = createVolumeSampler({ positions, indices, radius, mode, steps });
            const seedCount = sampler.count;
            expect(seedCount).toBeGreaterThan(0);
            expect(sampler.done).toBe(false);
            expect(sampler.totalSteps).toBe(steps);

            // Run to completion via the convenience path (drives step() internally).
            const res = sampleMeshVolume({ positions, indices, radius, mode, steps });

            // Relaxation moves particles; it never adds or removes them.
            expect(res.count).toBe(seedCount);
            expect(res.positions.length).toBe(seedCount * 3);

            const tol = 2 * radius;
            for (let i = 0; i < res.count; i++) {
                const px = res.positions[3 * i]!;
                const py = res.positions[3 * i + 1]!;
                const pz = res.positions[3 * i + 2]!;
                // No NaN/Infinity anywhere.
                expect(Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(pz)).toBe(true);
                // Particles stay essentially inside (small negative tolerance for grid slack /
                // mode 3's velocity-only boundary handling, which lacks a hard projection).
                expect(res.sdf.interpolate(px, py, pz)).toBeGreaterThan(-0.05);
                // Bounded region — no explosion.
                expect(px).toBeGreaterThanOrEqual(res.bounds.min[0] - tol);
                expect(px).toBeLessThanOrEqual(res.bounds.max[0] + tol);
                expect(py).toBeGreaterThanOrEqual(res.bounds.min[1] - tol);
                expect(py).toBeLessThanOrEqual(res.bounds.max[1] + tol);
                expect(pz).toBeGreaterThanOrEqual(res.bounds.min[2] - tol);
                expect(pz).toBeLessThanOrEqual(res.bounds.max[2] + tol);
            }
        });
    }

    it('mode "kugelstadt2021": relaxation drives density toward rest density', () => {
        const { positions, indices } = makeBox([-1, -1, -1], [1, 1, 1]);
        const radius = 0.2;
        const steps = 20;
        const sampler = createVolumeSampler({ positions, indices, radius, mode: "kugelstadt2021", steps });
        const count = sampler.count;

        // The seed is overseeded at 0.95*radius -> too dense -> mean density error above rest.
        const before = cubicDensityStats(sampler.positions, count, radius);
        while (!sampler.done) {
            sampler.step();
        }
        const after = cubicDensityStats(sampler.positions, count, radius);

        expect(Number.isFinite(after.meanAbsError)).toBe(true);
        // DFSPH position correction reduces the density error and keeps mean density sane.
        expect(after.meanAbsError).toBeLessThan(before.meanAbsError);
        expect(after.mean).toBeGreaterThan(500);
        expect(after.mean).toBeLessThan(1500);
    });

    for (const mode of ["kugelstadt2021", "jiang2015"] as const) {
        it(`mode "${mode}": deterministic — two full runs give identical positions`, () => {
            const { positions, indices } = makeBox([-1, -1, -1], [1, 1, 1]);
            const a = sampleMeshVolume({ positions, indices, radius: 0.2, mode, steps: 15 });
            const b = sampleMeshVolume({ positions, indices, radius: 0.2, mode, steps: 15 });
            expect(a.count).toBe(b.count);
            expect(a.positions.length).toBe(b.positions.length);
            for (let i = 0; i < a.positions.length; i++) {
                expect(a.positions[i]).toBe(b.positions[i]);
            }
        });

        it(`mode "${mode}": resumable step() path equals run-to-completion`, () => {
            const { positions, indices } = makeBox([-1, -1, -1], [1, 1, 1]);
            const steps = 18;
            const opts = { positions, indices, radius: 0.2, mode, steps } as const;

            const sampler = createVolumeSampler(opts);
            for (let s = 0; s < steps; s++) {
                sampler.step();
            }
            // Exactly totalSteps steps -> done, counter parked at totalSteps.
            expect(sampler.done).toBe(true);
            expect(sampler.stepIndex).toBe(steps);
            // Extra steps are a no-op once done.
            sampler.step();
            expect(sampler.stepIndex).toBe(steps);

            const res = sampleMeshVolume(opts);
            expect(res.count).toBe(sampler.count);
            for (let i = 0; i < res.positions.length; i++) {
                expect(res.positions[i]).toBe(sampler.positions[i]);
            }
        });
    }
});
