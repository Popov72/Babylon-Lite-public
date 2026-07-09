// Resumable SPH blue-noise relaxers — the SPH-relaxed VolumeSampling modes.
//
//   • createKugelstadtRelaxer  = mode 4 (Kugelstadt et al. 2021, the tool DEFAULT)
//         SPHVolumeSampling.cpp — DFSPH position-correction relaxation with CubicKernel.
//   • createJiang2015Relaxer   = mode 3 (Jiang et al. 2015)
//         SPHVolumeSampling_Jiang2015.cpp — weakly-compressible EOS relaxation with velocity state,
//         Poly6 density, Spiky gradient, Akinci cohesion.
//
// Both are ported 1:1 from the C++ (`step()`, plus the shared SPHSamplingBase helpers
// computeDensities / computeDFSPHFactor / pressureSolveIteration and the `distance(x,tol)` wrapper,
// which is `SignedDistanceGrid.distanceWithNormal`). Each holds its own state (densities, kernels,
// neighborhood, and — for mode 3 — velocity/accel/normal/pressure) and mutates the shared
// `positions` Float32Array IN PLACE on every `step()`. The caller (index.ts) owns the step counter
// and termination; a relaxer only knows how to advance one step.
//
// Pure CPU — no GPU handles, zero module-level side effects.

import type { SignedDistanceGrid } from "./mesh-sdf.js";
import type { KernelGrad } from "./sph-kernels.js";
import { createCohesionKernel, createCubicKernel, createPoly6Kernel, createSpikyKernel } from "./sph-kernels.js";
import { buildNeighborhood } from "./neighborhood.js";

/** A single relaxation stepper. `step()` advances the shared positions by one iteration. */
export interface SphRelaxer {
    step(): void;
}

/** Resolved parameters for the mode-4 (Kugelstadt 2021) relaxer. */
export interface KugelstadtParams {
    density0: number;
    cohesion: number;
    adhesion: number;
    cflFactor: number;
}

/** Resolved parameters for the mode-3 (Jiang 2015) relaxer. */
export interface Jiang2015Params {
    density0: number;
    cohesion: number;
    stiffness: number;
    dt: number;
}

// SPHSamplingBase.h: m_eps = 1e-5 (used by the DFSPH factor + pressure-solve thresholds).
const EPS = 1.0e-5;

/**
 * Mode 4 — Kugelstadt et al. 2021 (DEFAULT). DFSPH position-correction relaxation.
 *
 * @param positions - live flat xyz seed positions, mutated in place each step.
 * @param count - particle count.
 * @param radius - particle radius (full radius; diameter = 2*radius, supportRadius = 4*radius).
 * @param sdf - inverted signed-distance grid (inside is positive) for boundary handling.
 * @param params - resolved density0 / cohesion / adhesion / cflFactor.
 * @returns an {@link SphRelaxer}.
 */
export function createKugelstadtRelaxer(positions: Float32Array, count: number, radius: number, sdf: SignedDistanceGrid, params: KugelstadtParams): SphRelaxer {
    const { density0, cohesion, adhesion, cflFactor } = params;
    const diameter = 2.0 * radius;
    const supportRadius = 4.0 * radius;
    const volume = diameter * diameter * diameter;
    const mass = volume * density0;

    // Mode 4 uses only the cubic kernel — cohesion/adhesion weight with CubicKernel::W (m_kernelFct),
    // and the DFSPH factor/pressure gradient uses CubicKernel::gradW (m_gradKernelFct).
    const cubic = createCubicKernel(supportRadius);
    const wZero = cubic.wZero;

    const densities = new Float64Array(count);
    const factors = new Float64Array(count);
    const corr = new Float64Array(3 * count);
    const grad: KernelGrad = { x: 0, y: 0, z: 0 };
    const neighborhood = buildNeighborhood(positions, count, supportRadius);

    const computeDensities = (): void => {
        for (let i = 0; i < count; i++) {
            let d = wZero;
            neighborhood.forEachNeighbor(i, (_j, _dx, _dy, _dz, r2) => {
                d += cubic.W(Math.sqrt(r2));
            });
            densities[i] = d * mass;
        }
    };

    const computeDFSPHFactor = (): void => {
        for (let i = 0; i < count; i++) {
            let sumGradPk = 0.0;
            let gpix = 0.0;
            let gpiy = 0.0;
            let gpiz = 0.0;
            neighborhood.forEachNeighbor(i, (_j, dx, dy, dz) => {
                cubic.gradW(dx, dy, dz, grad);
                const gpjx = -volume * grad.x;
                const gpjy = -volume * grad.y;
                const gpjz = -volume * grad.z;
                sumGradPk += gpjx * gpjx + gpjy * gpjy + gpjz * gpjz;
                gpix -= gpjx;
                gpiy -= gpjy;
                gpiz -= gpjz;
            });
            sumGradPk += gpix * gpix + gpiy * gpiy + gpiz * gpiz;
            factors[i] = sumGradPk > EPS ? -1.0 / sumGradPk : 0.0;
        }
    };

    const pressureSolveIteration = (): void => {
        for (let i = 0; i < count; i++) {
            const bi = densities[i]! / density0 - 1.0;
            const ki = bi * factors[i]!;
            let cx = corr[3 * i]!;
            let cy = corr[3 * i + 1]!;
            let cz = corr[3 * i + 2]!;
            neighborhood.forEachNeighbor(i, (j, dx, dy, dz) => {
                const bj = densities[j]! / density0 - 1.0;
                const kj = bj * factors[j]!;
                const kSum = ki + kj;
                if (Math.abs(kSum) > EPS) {
                    cubic.gradW(dx, dy, dz, grad);
                    const gpjx = -volume * grad.x;
                    const gpjy = -volume * grad.y;
                    const gpjz = -volume * grad.z;
                    cx -= kSum * gpjx;
                    cy -= kSum * gpjy;
                    cz -= kSum * gpjz;
                }
            });
            corr[3 * i] = cx;
            corr[3 * i + 1] = cy;
            corr[3 * i + 2] = cz;
        }
    };

    const step = (): void => {
        neighborhood.rebuild();
        computeDensities();

        // Cohesion (Becker 2007) + optional adhesion, accumulated into corr (reset per particle).
        for (let i = 0; i < count; i++) {
            const xi = positions[3 * i]!;
            const yi = positions[3 * i + 1]!;
            const zi = positions[3 * i + 2]!;
            let cx = 0.0;
            let cy = 0.0;
            let cz = 0.0;
            neighborhood.forEachNeighbor(i, (j, dx, dy, dz, r2) => {
                if (r2 > 1e-6) {
                    const rl = Math.sqrt(r2);
                    const C = rl - diameter;
                    // corr -= cohesion * mass/density[j] * C * (dx,dy,dz)/rl * W(rl)
                    const f = ((cohesion * mass) / densities[j]!) * C * (cubic.W(rl) / rl);
                    cx -= f * dx;
                    cy -= f * dy;
                    cz -= f * dz;
                }
            });

            if (adhesion !== 0.0) {
                const dn = sdf.distanceWithNormal(xi, yi, zi, 0.0);
                if (dn.dist < supportRadius) {
                    // (xi - cp) = dist*normal ; weighted by CubicKernel::W(|xi - cp|).
                    const ex = xi - dn.cx;
                    const ey = yi - dn.cy;
                    const ez = zi - dn.cz;
                    const f = ((adhesion * mass) / densities[i]!) * cubic.W(Math.sqrt(ex * ex + ey * ey + ez * ez));
                    cx -= f * ex;
                    cy -= f * ey;
                    cz -= f * ez;
                }
            }
            corr[3 * i] = cx;
            corr[3 * i + 1] = cy;
            corr[3 * i + 2] = cz;
        }

        computeDFSPHFactor();
        pressureSolveIteration();

        // CFL from the largest correction magnitude (maxCorr is stored squared).
        let maxCorr = 0.0;
        for (let i = 0; i < count; i++) {
            const cx = corr[3 * i]!;
            const cy = corr[3 * i + 1]!;
            const cz = corr[3 * i + 2]!;
            const m = cx * cx + cy * cy + cz * cz;
            if (m > maxCorr) {
                maxCorr = m;
            }
        }
        let cfl = cflFactor * 0.4 * (diameter / Math.sqrt(maxCorr));
        cfl = Math.min(cfl, 1.0);

        // Apply corrections, then project stragglers back to 1 radius inside the surface.
        for (let i = 0; i < count; i++) {
            let px = positions[3 * i]! + cfl * corr[3 * i]!;
            let py = positions[3 * i + 1]! + cfl * corr[3 * i + 1]!;
            let pz = positions[3 * i + 2]! + cfl * corr[3 * i + 2]!;
            const dn = sdf.distanceWithNormal(px, py, pz, 0.0);
            if (dn.dist < radius) {
                px = dn.cx + radius * dn.nx;
                py = dn.cy + radius * dn.ny;
                pz = dn.cz + radius * dn.nz;
            }
            positions[3 * i] = px;
            positions[3 * i + 1] = py;
            positions[3 * i + 2] = pz;
        }
    };

    return { step };
}

/**
 * Mode 3 — Jiang et al. 2015. Weakly-compressible SPH (EOS) relaxation with explicit velocity
 * integration, Poly6 density, Spiky gradient, and Akinci cohesion.
 *
 * @param positions - live flat xyz seed positions, mutated in place each step.
 * @param count - particle count.
 * @param radius - particle radius (full radius; diameter = 2*radius, supportRadius = 4*radius).
 * @param sdf - inverted signed-distance grid (inside is positive) for boundary handling.
 * @param params - resolved density0 / cohesion / stiffness / dt.
 * @returns an {@link SphRelaxer}.
 */
export function createJiang2015Relaxer(positions: Float32Array, count: number, radius: number, sdf: SignedDistanceGrid, params: Jiang2015Params): SphRelaxer {
    const { density0, cohesion, stiffness, dt } = params;
    const diameter = 2.0 * radius;
    const supportRadius = 4.0 * radius;
    const mass = diameter * diameter * diameter * density0;

    const poly6 = createPoly6Kernel(supportRadius); // density
    const spiky = createSpikyKernel(supportRadius); // gradient (normals + pressure)
    const cohesionK = createCohesionKernel(supportRadius); // cohesion weight
    const wZero = poly6.wZero;

    const densities = new Float64Array(count);
    const v = new Float64Array(3 * count); // persistent velocity state
    const a = new Float64Array(3 * count);
    const n = new Float64Array(3 * count); // SPH surface normals
    const p = new Float64Array(count);
    const grad: KernelGrad = { x: 0, y: 0, z: 0 };
    const neighborhood = buildNeighborhood(positions, count, supportRadius);

    const computeDensities = (): void => {
        for (let i = 0; i < count; i++) {
            let d = wZero;
            neighborhood.forEachNeighbor(i, (_j, _dx, _dy, _dz, r2) => {
                d += poly6.W(Math.sqrt(r2));
            });
            densities[i] = d * mass;
        }
    };

    const computeNormals = (): void => {
        for (let i = 0; i < count; i++) {
            let nx = 0.0;
            let ny = 0.0;
            let nz = 0.0;
            neighborhood.forEachNeighbor(i, (j, dx, dy, dz) => {
                spiky.gradW(dx, dy, dz, grad);
                const f = mass / densities[j]!;
                nx -= f * grad.x;
                ny -= f * grad.y;
                nz -= f * grad.z;
            });
            const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 0.0) {
                nx /= len;
                ny /= len;
                nz /= len;
            }
            n[3 * i] = nx;
            n[3 * i + 1] = ny;
            n[3 * i + 2] = nz;
        }
    };

    const computePressure = (): void => {
        // EOS pressure for every particle first (the accel loop reads neighbor pressures).
        for (let i = 0; i < count; i++) {
            p[i] = Math.max(stiffness * (densities[i]! - density0), 0.0);
        }
        for (let i = 0; i < count; i++) {
            const di = densities[i]!;
            const piTerm = p[i]! / (di * di);
            let ax = a[3 * i]!;
            let ay = a[3 * i + 1]!;
            let az = a[3 * i + 2]!;
            neighborhood.forEachNeighbor(i, (j, dx, dy, dz) => {
                spiky.gradW(dx, dy, dz, grad);
                const dj = densities[j]!;
                const coef = mass * (piTerm + p[j]! / (dj * dj));
                ax -= coef * grad.x;
                ay -= coef * grad.y;
                az -= coef * grad.z;
            });
            // Outside the mesh (dist <= 0): remove the normal component of the pressure accel.
            const dn = sdf.distanceWithNormal(positions[3 * i]!, positions[3 * i + 1]!, positions[3 * i + 2]!, 0.0);
            if (dn.dist <= 0.0) {
                const nx = n[3 * i]!;
                const ny = n[3 * i + 1]!;
                const nz = n[3 * i + 2]!;
                const d = ax * nx + ay * ny + az * nz;
                ax -= d * nx;
                ay -= d * ny;
                az -= d * nz;
            }
            a[3 * i] = ax;
            a[3 * i + 1] = ay;
            a[3 * i + 2] = az;
        }
    };

    const computeCohesion = (): void => {
        for (let i = 0; i < count; i++) {
            const di = densities[i]!;
            let ax = a[3 * i]!;
            let ay = a[3 * i + 1]!;
            let az = a[3 * i + 2]!;
            neighborhood.forEachNeighbor(i, (j, dx, dy, dz, r2) => {
                if (r2 > 0.0) {
                    const Kij = (2.0 * density0) / (di + densities[j]!);
                    const rl = Math.sqrt(r2);
                    // a -= cohesion * mass * Kij * W(rl) * (dx,dy,dz)/rl
                    const coef = (cohesion * mass * Kij * cohesionK.W(rl)) / rl;
                    ax -= coef * dx;
                    ay -= coef * dy;
                    az -= coef * dz;
                }
            });
            a[3 * i] = ax;
            a[3 * i + 1] = ay;
            a[3 * i + 2] = az;
        }
    };

    const step = (): void => {
        neighborhood.rebuild();
        a.fill(0.0);
        computeDensities();
        computeNormals();
        computePressure();
        computeCohesion();

        // Explicit integration with velocity damping and velocity-based boundary handling.
        for (let i = 0; i < count; i++) {
            let vx = (v[3 * i]! + dt * a[3 * i]!) * 0.9;
            let vy = (v[3 * i + 1]! + dt * a[3 * i + 1]!) * 0.9;
            let vz = (v[3 * i + 2]! + dt * a[3 * i + 2]!) * 0.9;

            const px = positions[3 * i]!;
            const py = positions[3 * i + 1]!;
            const pz = positions[3 * i + 2]!;
            const dn = sdf.distanceWithNormal(px, py, pz, 0.0);
            if (dn.dist < 0.0) {
                // Particle outside the mesh: kill the outward (normal) velocity component.
                const nx = n[3 * i]!;
                const ny = n[3 * i + 1]!;
                const nz = n[3 * i + 2]!;
                const d = vx * nx + vy * ny + vz * nz;
                vx -= d * nx;
                vy -= d * ny;
                vz -= d * nz;
            } else {
                positions[3 * i] = px + dt * vx;
                positions[3 * i + 1] = py + dt * vy;
                positions[3 * i + 2] = pz + dt * vz;
            }
            v[3 * i] = vx;
            v[3 * i + 1] = vy;
            v[3 * i + 2] = vz;
        }
    };

    return { step };
}
