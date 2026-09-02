import { describe, expect, it } from "vitest";

import type { MeshSdfGrid } from "../../../../packages/babylon-lite/src/fluid/sampling/volume/sdf-gen";
import { generateMeshSdf } from "../../../../packages/babylon-lite/src/fluid/sampling/volume/sdf-gen";

// ------------------------------------------------------------------
// Analytic test meshes (built in-code so the tests are self-contained).
// Mirrors the helpers used by volume-sampling.test.ts.
// ------------------------------------------------------------------

interface Mesh {
    positions: Float32Array;
    indices: Uint32Array;
}

// Axis-aligned box (8 verts, 12 tris), winding fixed outward.
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

// UV sphere centered at `center` (convex -> outward winding fixed below).
function makeUvSphere(radius: number, stacks: number, slices: number, center: [number, number, number] = [0, 0, 0]): Mesh {
    const verts: number[] = [];
    for (let i = 0; i <= stacks; i++) {
        const theta = (i / stacks) * Math.PI;
        const st = Math.sin(theta);
        const ct = Math.cos(theta);
        for (let j = 0; j <= slices; j++) {
            const phi = (j / slices) * 2 * Math.PI;
            verts.push(center[0] + radius * st * Math.cos(phi), center[1] + radius * ct, center[2] + radius * st * Math.sin(phi));
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
    fixOutwardWinding(positions, indices, center);
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
        const ox = (ax + bx + cx) / 3 - center[0];
        const oy = (ay + by + cy) / 3 - center[1];
        const oz = (az + bz + cz) / 3 - center[2];
        if (nx * ox + ny * oy + nz * oz < 0) {
            indices[3 * t + 1] = i2;
            indices[3 * t + 2] = i1;
        }
    }
}

// Mesh AABB.
function meshAabb(positions: Float32Array): { min: [number, number, number]; max: [number, number, number] } {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < positions.length; i += 3) {
        minX = Math.min(minX, positions[i]!);
        minY = Math.min(minY, positions[i + 1]!);
        minZ = Math.min(minZ, positions[i + 2]!);
        maxX = Math.max(maxX, positions[i]!);
        maxY = Math.max(maxY, positions[i + 1]!);
        maxZ = Math.max(maxZ, positions[i + 2]!);
    }
    return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

// Sample the baked grid at an arbitrary world point via trilinear interpolation of the node values.
function sampleGrid(grid: MeshSdfGrid, x: number, y: number, z: number): number {
    const [ni, nj, nk] = grid.dims;
    const gx = (x - grid.origin[0]) / grid.cellSize;
    const gy = (y - grid.origin[1]) / grid.cellSize;
    const gz = (z - grid.origin[2]) / grid.cellSize;
    const i0 = Math.max(0, Math.min(ni - 2, Math.floor(gx)));
    const j0 = Math.max(0, Math.min(nj - 2, Math.floor(gy)));
    const k0 = Math.max(0, Math.min(nk - 2, Math.floor(gz)));
    const fx = gx - i0;
    const fy = gy - j0;
    const fz = gz - k0;
    const at = (i: number, j: number, k: number): number => grid.data[i + ni * (j + nj * k)]!;
    const c000 = at(i0, j0, k0);
    const c100 = at(i0 + 1, j0, k0);
    const c010 = at(i0, j0 + 1, k0);
    const c110 = at(i0 + 1, j0 + 1, k0);
    const c001 = at(i0, j0, k0 + 1);
    const c101 = at(i0 + 1, j0, k0 + 1);
    const c011 = at(i0, j0 + 1, k0 + 1);
    const c111 = at(i0 + 1, j0 + 1, k0 + 1);
    const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
    const x00 = lerp(c000, c100, fx);
    const x10 = lerp(c010, c110, fx);
    const x01 = lerp(c001, c101, fx);
    const x11 = lerp(c011, c111, fx);
    const y0 = lerp(x00, x10, fy);
    const y1 = lerp(x01, x11, fy);
    return lerp(y0, y1, fz);
}

// Analytic signed distance to an axis-aligned box (negative inside).
function boxSdf(px: number, py: number, pz: number, min: [number, number, number], max: [number, number, number]): number {
    const cx = (min[0] + max[0]) / 2;
    const cy = (min[1] + max[1]) / 2;
    const cz = (min[2] + max[2]) / 2;
    const hx = (max[0] - min[0]) / 2;
    const hy = (max[1] - min[1]) / 2;
    const hz = (max[2] - min[2]) / 2;
    const qx = Math.abs(px - cx) - hx;
    const qy = Math.abs(py - cy) - hy;
    const qz = Math.abs(pz - cz) - hz;
    const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0));
    const inside = Math.min(Math.max(qx, Math.max(qy, qz)), 0);
    return outside + inside;
}

describe("sdf-gen: signed distance field baker", () => {
    it("sphere: matches |p - center| - R (negative inside, positive outside)", () => {
        const R = 1.0;
        const center: [number, number, number] = [0.3, -0.2, 0.15];
        const { positions, indices } = makeUvSphere(R, 40, 60, center);
        const { min, max } = meshAabb(positions);
        const cellSize = (2 * R) / 40; // ~40 cells across the diameter
        const grid = generateMeshSdf(positions, indices, { min, max, cellSize });

        const tol = 2 * cellSize; // within ~2 cells (UV-sphere is a faceted approximation)
        // Sample a spread of interior + exterior points inside the padded domain.
        const dirs: Array<[number, number, number]> = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [-1, 1, -1],
            [0.5, -1, 0.3],
            [-0.7, -0.4, 0.9],
        ];
        for (const [dx, dy, dz] of dirs) {
            const len = Math.hypot(dx, dy, dz);
            for (const rFrac of [0.25, 0.6, 0.95, 1.15, 1.4]) {
                const r = rFrac * R;
                const px = center[0] + (dx / len) * r;
                const py = center[1] + (dy / len) * r;
                const pz = center[2] + (dz / len) * r;
                const expected = r - R; // analytic sphere SDF, negative inside
                const got = sampleGrid(grid, px, py, pz);
                expect(Math.abs(got - expected)).toBeLessThan(tol);
                // Sign must be correct comfortably away from the surface.
                if (Math.abs(expected) > 1.5 * cellSize) {
                    expect(Math.sign(got)).toBe(Math.sign(expected));
                }
            }
        }
    });

    it("box: correct sign vs analytic box SDF (interior negative, exterior positive)", () => {
        const min: [number, number, number] = [-0.8, -0.5, -0.6];
        const max: [number, number, number] = [0.8, 0.5, 0.6];
        const { positions, indices } = makeBox(min, max);
        const cellSize = 0.08;
        const grid = generateMeshSdf(positions, indices, { min, max, cellSize });

        const tol = 2 * cellSize;
        const rng = makeRng(1234);
        let checked = 0;
        for (let s = 0; s < 400; s++) {
            // Points spread over the padded domain.
            const px = min[0] - 0.3 + rng() * (max[0] - min[0] + 0.6);
            const py = min[1] - 0.3 + rng() * (max[1] - min[1] + 0.6);
            const pz = min[2] - 0.3 + rng() * (max[2] - min[2] + 0.6);
            const expected = boxSdf(px, py, pz, min, max);
            const got = sampleGrid(grid, px, py, pz);
            expect(Math.abs(got - expected)).toBeLessThan(tol);
            if (Math.abs(expected) > 1.5 * cellSize) {
                expect(Math.sign(got)).toBe(Math.sign(expected));
                checked++;
            }
        }
        expect(checked).toBeGreaterThan(50);
    });

    it("sign correctness: deep interior negative, far exterior positive", () => {
        const R = 1.0;
        const { positions, indices } = makeUvSphere(R, 32, 48);
        const { min, max } = meshAabb(positions);
        const grid = generateMeshSdf(positions, indices, { min, max, cellSize: 0.06 });

        // Center is deep inside -> negative, magnitude ~ R.
        const centerVal = sampleGrid(grid, 0, 0, 0);
        expect(centerVal).toBeLessThan(0);
        expect(centerVal).toBeCloseTo(-R, 1);

        // A node in the padding shell is outside -> positive.
        const cornerVal = grid.data[0]!; // node (0,0,0) sits in the padded shell
        expect(cornerVal).toBeGreaterThan(0);

        // Every deep-interior sample negative; every far-exterior sample positive.
        const rng = makeRng(99);
        for (let s = 0; s < 200; s++) {
            const dx = rng() * 2 - 1;
            const dy = rng() * 2 - 1;
            const dz = rng() * 2 - 1;
            const len = Math.hypot(dx, dy, dz) || 1;
            const inside = sampleGrid(grid, (dx / len) * 0.4, (dy / len) * 0.4, (dz / len) * 0.4);
            expect(inside).toBeLessThan(0);
        }
    });

    it("grid indexing: length matches dims and node values are self-consistent", () => {
        const { positions, indices } = makeBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
        const grid = generateMeshSdf(positions, indices, { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5], cellSize: 0.1 });
        const [ni, nj, nk] = grid.dims;
        expect(grid.data.length).toBe(ni * nj * nk);

        // Sampling the grid exactly at a node returns that node's stored value.
        const i = 3;
        const j = 4;
        const k = 5;
        const nodeVal = grid.data[i + ni * (j + nj * k)]!;
        const px = grid.origin[0] + i * grid.cellSize;
        const py = grid.origin[1] + j * grid.cellSize;
        const pz = grid.origin[2] + k * grid.cellSize;
        expect(sampleGrid(grid, px, py, pz)).toBeCloseTo(nodeVal, 5);
    });

    it("determinism: two bakes of the same mesh are bit-identical", () => {
        const { positions, indices } = makeUvSphere(1.0, 24, 36);
        const { min, max } = meshAabb(positions);
        const a = generateMeshSdf(positions, indices, { min, max, cellSize: 0.08 });
        const b = generateMeshSdf(positions, indices, { min, max, cellSize: 0.08 });
        expect(a.dims).toEqual(b.dims);
        expect(a.origin).toEqual(b.origin);
        expect(a.data.length).toBe(b.data.length);
        for (let idx = 0; idx < a.data.length; idx++) {
            expect(a.data[idx]).toBe(b.data[idx]);
        }
    });

    it("accepts Uint16 indices and the explicit dims form", () => {
        const box = makeBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
        const indices16 = Uint16Array.from(box.indices);
        // Explicit dims form: origin = min, node counts verbatim. Pad the box into a 20^3 grid.
        const cellSize = 0.1;
        const dims: [number, number, number] = [20, 20, 20];
        const originMin: [number, number, number] = [-1.0, -1.0, -1.0];
        const grid = generateMeshSdf(box.positions, indices16, { min: originMin, dims, cellSize });
        expect(grid.dims).toEqual(dims);
        expect(grid.origin).toEqual(originMin);
        expect(grid.data.length).toBe(20 * 20 * 20);
        // Center inside the box -> negative; a far corner in the padding -> positive.
        expect(sampleGrid(grid, 0, 0, 0)).toBeLessThan(0);
        expect(grid.data[0]!).toBeGreaterThan(0);
    });
});

// Small deterministic PRNG for reproducible random query points.
function makeRng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
    };
}
