// SPH smoothing kernels — exact ports of SPlisHSPlasH `SPlisHSPlasH/SPHKernels.h`.
//
// Each factory takes the support radius `h` (= 4 * particleRadius in this tool) and returns a
// plain object with the precomputed normalization constants baked in. All formulas and branch
// cutoffs match the C++ 1:1 (CubicKernel / Poly6Kernel / SpikyKernel / CohesionKernel). The
// kernels are pure and allocation-free in their hot paths: `W(r)` takes the scalar distance and
// `gradW(dx,dy,dz, out)` writes the gradient of W(xi-xj) into a caller-provided scratch vector
// (called O(N*neighbors) per relaxation step).
//
// Pure CPU — no GPU handles, zero module-level side effects (constants live inside the factories).

/** Mutable 3-component vector used as scratch output for {@link SphKernel.gradW}. */
export interface KernelGrad {
    x: number;
    y: number;
    z: number;
}

/** A scalar SPH kernel: self weight `wZero = W(0)` and `W(r)` for a scalar distance `r`. */
export interface SphScalarKernel {
    readonly wZero: number;
    W(r: number): number;
}

/** A scalar kernel that also provides the gradient of `W(xi-xj)` w.r.t. `xi`. */
export interface SphKernel extends SphScalarKernel {
    /** Writes gradW(dx,dy,dz) (dx,dy,dz = xi-xj) into `out`. Zero outside the support / at r=0. */
    gradW(dx: number, dy: number, dz: number, out: KernelGrad): void;
}

/**
 * Cubic spline kernel (SPHKernels.h:CubicKernel). Used by mode 4 for both density (`W`) and the
 * DFSPH pressure/factor gradient (`gradW`), and for the mode-4 cohesion/adhesion weighting.
 *
 * @param h - support radius.
 * @returns a {@link SphKernel}.
 */
export function createCubicKernel(h: number): SphKernel {
    const h3 = h * h * h;
    const k = 8.0 / (Math.PI * h3);
    const l = 48.0 / (Math.PI * h3);
    const wZero = k; // W(0) = k*(6*0^3 - 6*0^2 + 1) = k

    const W = (r: number): number => {
        const q = r / h;
        if (q <= 1.0) {
            if (q <= 0.5) {
                const q2 = q * q;
                const q3 = q2 * q;
                return k * (6.0 * q3 - 6.0 * q2 + 1.0);
            }
            const t = 1.0 - q;
            return k * (2.0 * t * t * t);
        }
        return 0.0;
    };

    const gradW = (dx: number, dy: number, dz: number, out: KernelGrad): void => {
        const rl = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const q = rl / h;
        if (rl > 1.0e-9 && q <= 1.0) {
            // gradq = (r / rl) / h ; res = factor * gradq  =>  scale = factor / (rl*h)
            const factor = q <= 0.5 ? l * q * (3.0 * q - 2.0) : -(l * (1.0 - q) * (1.0 - q));
            const s = factor / (rl * h);
            out.x = dx * s;
            out.y = dy * s;
            out.z = dz * s;
        } else {
            out.x = 0.0;
            out.y = 0.0;
            out.z = 0.0;
        }
    };

    return { wZero, W, gradW };
}

/**
 * Poly6 kernel (SPHKernels.h:Poly6Kernel). Used by mode 3 for density.
 *
 * @param h - support radius.
 * @returns a {@link SphKernel}.
 */
export function createPoly6Kernel(h: number): SphKernel {
    const h2 = h * h;
    const h9 = Math.pow(h, 9.0);
    const k = 315.0 / (64.0 * Math.PI * h9);
    const l = -945.0 / (32.0 * Math.PI * h9);
    const wZero = k * h2 * h2 * h2; // W(0) = k*(h^2)^3 = k*h^6

    const W = (r: number): number => {
        const r2 = r * r;
        if (r2 <= h2) {
            const t = h2 - r2;
            return t * t * t * k;
        }
        return 0.0;
    };

    const gradW = (dx: number, dy: number, dz: number, out: KernelGrad): void => {
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= h2) {
            const t = h2 - r2;
            const s = l * t * t;
            out.x = dx * s;
            out.y = dy * s;
            out.z = dz * s;
        } else {
            out.x = 0.0;
            out.y = 0.0;
            out.z = 0.0;
        }
    };

    return { wZero, W, gradW };
}

/**
 * Spiky kernel (SPHKernels.h:SpikyKernel). Used by mode 3 as the gradient kernel for normals and
 * the pressure force.
 *
 * @param h - support radius.
 * @returns a {@link SphKernel}.
 */
export function createSpikyKernel(h: number): SphKernel {
    const h2 = h * h;
    const radius6 = Math.pow(h, 6.0);
    const k = 15.0 / (Math.PI * radius6);
    const l = -45.0 / (Math.PI * radius6);
    const wZero = k * h * h * h; // W(0) = k*(h-0)^3 = k*h^3

    const W = (r: number): number => {
        if (r * r <= h2) {
            const hr = h - r;
            return k * hr * hr * hr;
        }
        return 0.0;
    };

    const gradW = (dx: number, dy: number, dz: number, out: KernelGrad): void => {
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= h2 && r2 > 0.0) {
            const rl = Math.sqrt(r2);
            const hr = h - rl;
            const s = (l * hr * hr) / rl;
            out.x = dx * s;
            out.y = dy * s;
            out.z = dz * s;
        } else {
            out.x = 0.0;
            out.y = 0.0;
            out.z = 0.0;
        }
    };

    return { wZero, W, gradW };
}

/**
 * Cohesion kernel (Akinci 2013; SPHKernels.h:CohesionKernel). Used by mode 3's cohesion force.
 * Note `W(0) = -h^6/64` in the source (not 0), but r=0 never occurs (self is excluded).
 *
 * @param h - support radius.
 * @returns a {@link SphScalarKernel} (no gradient variant exists in the source).
 */
export function createCohesionKernel(h: number): SphScalarKernel {
    const h2 = h * h;
    const k = 32.0 / (Math.PI * Math.pow(h, 9.0));
    const c = Math.pow(h, 6.0) / 64.0;
    const halfH = 0.5 * h;
    const wZero = -c; // W(0): r1=0 => 2*k*(h)^3*0 - c = -c

    const W = (r: number): number => {
        if (r * r <= h2) {
            const r3 = r * r * r;
            const hr = h - r;
            const hr3 = hr * hr * hr;
            if (r > halfH) {
                return k * hr3 * r3;
            }
            return k * 2.0 * hr3 * r3 - c;
        }
        return 0.0;
    };

    return { wZero, W };
}
