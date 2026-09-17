import type { ReferenceCase, ReferenceSdf } from "./types";

interface LoadedSdf extends ReferenceSdf {
    distances: Float32Array;
}

export interface ComparisonSolids {
    distances: Float32Array;
    velocities: Float32Array;
    staticDistances: Float32Array;
    local: LoadedSdf[];
}

function addDomainShell(input: ReferenceCase, distances: Float32Array): void {
    const [nx, ny, nz] = input.grid.dimensions;
    const h = input.grid.cellSize;
    let i = 0;
    for (let z = 0; z <= nz; z++) {
        for (let y = 0; y <= ny; y++) {
            for (let x = 0; x <= nx; x++, i++) {
                const wall = Math.min(x, y, z, nx - x, ny - y, nz - z) * h - input.domainInset;
                distances[i] = Math.min(distances[i]!, wall);
            }
        }
    }
}

export function sampleSdf(grid: LoadedSdf, x: number, y: number, z: number): number {
    const [nx, ny, nz] = grid.dimensions;
    const gx = (x - grid.origin[0]) / grid.cellSize;
    const gy = (y - grid.origin[1]) / grid.cellSize;
    const gz = (z - grid.origin[2]) / grid.cellSize;
    const cx = Math.max(0, Math.min(nx - 1, gx));
    const cy = Math.max(0, Math.min(ny - 1, gy));
    const cz = Math.max(0, Math.min(nz - 1, gz));
    const ix = Math.min(nx - 2, Math.floor(cx));
    const iy = Math.min(ny - 2, Math.floor(cy));
    const iz = Math.min(nz - 2, Math.floor(cz));
    const fx = cx - ix;
    const fy = cy - iy;
    const fz = cz - iz;
    let d = 0;
    for (let dz = 0; dz < 2; dz++) {
        for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
                d += grid.distances[ix + dx + nx * (iy + dy + ny * (iz + dz))]! * (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
            }
        }
    }
    const outside = Math.hypot(gx - cx, gy - cy, gz - cz) * grid.cellSize;
    return outside > 0 ? Math.max(d, 0) + outside : d;
}

export async function loadComparisonSolids(input: ReferenceCase): Promise<ComparisonSolids> {
    const load = async (sdf: ReferenceSdf): Promise<LoadedSdf> => {
        const response = await fetch(sdf.file);
        if (!response.ok) {
            throw new Error(`Cannot load collision field ${sdf.file}: ${response.status}.`);
        }
        const distances = new Float32Array(await response.arrayBuffer());
        if (distances.length !== sdf.dimensions[0] * sdf.dimensions[1] * sdf.dimensions[2] || distances.some((d) => !Number.isFinite(d))) {
            throw new Error(`Invalid collision data in ${sdf.file}.`);
        }
        return { ...sdf, distances };
    };
    const count = input.grid.dimensions.reduce((n, size) => n * (size + 1), 1);
    const staticDistances = new Float32Array(count).fill(1e6);
    if (input.staticSdf) {
        const grid = await load(input.staticSdf);
        const [nx, ny, nz] = input.grid.dimensions;
        let i = 0;
        for (let z = 0; z <= nz; z++) {
            for (let y = 0; y <= ny; y++) {
                for (let x = 0; x <= nx; x++) {
                    staticDistances[i++] = sampleSdf(
                        grid,
                        input.grid.origin[0] + x * input.grid.cellSize,
                        input.grid.origin[1] + y * input.grid.cellSize,
                        input.grid.origin[2] + z * input.grid.cellSize
                    );
                }
            }
        }
    }
    if (input.domainInset > 0) {
        addDomainShell(input, staticDistances);
    }
    const local: LoadedSdf[] = [];
    for (const obstacle of input.obstacles) {
        local.push(await load(obstacle));
    }
    return { distances: staticDistances.slice(), velocities: new Float32Array(count * 3), staticDistances, local };
}

export function updateComparisonSolids(input: ReferenceCase, state: ComparisonSolids, step: number): void {
    state.distances.set(state.staticDistances);
    state.velocities.fill(0);
    const [nx, ny, nz] = input.grid.dimensions;
    for (let obstacle = 0; obstacle < input.obstacles.length; obstacle++) {
        const definition = input.obstacles[obstacle]!;
        const current = definition.transforms[step];
        const velocity = definition.velocityTransforms[step];
        if (!current || !velocity || current.length !== 16 || velocity.length !== 16) {
            throw new Error(`Missing ${definition.name} transform for physical substep ${step}.`);
        }
        const inverse = definition.inverseTransforms[step];
        if (!inverse || inverse.length !== 16) {
            throw new Error(`Missing inverse obstacle transform: ${definition.name}.`);
        }
        const scale =
            1 / Math.max(Math.hypot(inverse[0]!, inverse[4]!, inverse[8]!), Math.hypot(inverse[1]!, inverse[5]!, inverse[9]!), Math.hypot(inverse[2]!, inverse[6]!, inverse[10]!));
        let i = 0;
        for (let z = 0; z <= nz; z++) {
            for (let y = 0; y <= ny; y++) {
                for (let x = 0; x <= nx; x++, i++) {
                    const wx = input.grid.origin[0] + x * input.grid.cellSize;
                    const wy = input.grid.origin[1] + y * input.grid.cellSize;
                    const wz = input.grid.origin[2] + z * input.grid.cellSize;
                    const lx = inverse[0]! * wx + inverse[4]! * wy + inverse[8]! * wz + inverse[12]!;
                    const ly = inverse[1]! * wx + inverse[5]! * wy + inverse[9]! * wz + inverse[13]!;
                    const lz = inverse[2]! * wx + inverse[6]! * wy + inverse[10]! * wz + inverse[14]!;
                    const distance = sampleSdf(state.local[obstacle]!, lx, ly, lz) * scale;
                    if (distance >= state.distances[i]!) {
                        continue;
                    }
                    state.distances[i] = distance;
                    state.velocities[i * 3] = velocity[0]! * lx + velocity[4]! * ly + velocity[8]! * lz + velocity[12]!;
                    state.velocities[i * 3 + 1] = velocity[1]! * lx + velocity[5]! * ly + velocity[9]! * lz + velocity[13]!;
                    state.velocities[i * 3 + 2] = velocity[2]! * lx + velocity[6]! * ly + velocity[10]! * lz + velocity[14]!;
                }
            }
        }
    }
}
