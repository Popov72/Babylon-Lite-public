export type OpticalRgb = readonly [number, number, number];

export interface OpticalTransfer {
    readonly radiance: OpticalRgb;
    readonly transmittance: OpticalRgb;
}

const OPTICAL_EPSILON = 1e-6;

export function alphaToOpticalDepth(alpha: number): number {
    if (!Number.isFinite(alpha)) {
        throw new RangeError(`Optical transfer: alpha must be finite (got ${alpha}).`);
    }
    return -Math.log(Math.max(1 - clamp(alpha, 0, 1), OPTICAL_EPSILON));
}

export function transmittanceFromOpticalDepth(opticalDepth: number): number {
    if (!Number.isFinite(opticalDepth)) {
        throw new RangeError(`Optical transfer: opticalDepth must be finite (got ${opticalDepth}).`);
    }
    return Math.exp(-Math.max(opticalDepth, 0));
}

export function composeOpticalTransfer(front: OpticalTransfer, back: OpticalTransfer): OpticalTransfer {
    return {
        radiance: [
            front.radiance[0] + front.transmittance[0] * back.radiance[0],
            front.radiance[1] + front.transmittance[1] * back.radiance[1],
            front.radiance[2] + front.transmittance[2] * back.radiance[2],
        ],
        transmittance: [front.transmittance[0] * back.transmittance[0], front.transmittance[1] * back.transmittance[1], front.transmittance[2] * back.transmittance[2]],
    };
}

export function integrateHomogeneousMedium(source: OpticalRgb, extinction: OpticalRgb, distance: number): OpticalTransfer {
    if (!Number.isFinite(distance) || distance < 0) {
        throw new RangeError(`Optical transfer: distance must be finite and non-negative (got ${distance}).`);
    }
    const radiance = [0, 0, 0] as [number, number, number];
    const transmittance = [0, 0, 0] as [number, number, number];
    for (let i = 0; i < 3; i++) {
        const sigma = extinction[i]!;
        if (!Number.isFinite(sigma) || sigma < 0) {
            throw new RangeError(`Optical transfer: extinction[${i}] must be finite and non-negative (got ${sigma}).`);
        }
        const value = source[i]!;
        if (!Number.isFinite(value)) {
            throw new RangeError(`Optical transfer: source[${i}] must be finite (got ${value}).`);
        }
        const opticalDepth = sigma * distance;
        const channelTransmittance = Math.exp(-opticalDepth);
        transmittance[i] = channelTransmittance;
        const integratedDistance = opticalDepth === 0 ? distance : Number.isFinite(opticalDepth) ? (distance * -Math.expm1(-opticalDepth)) / opticalDepth : 1 / sigma;
        radiance[i] = value * integratedDistance;
    }
    return { radiance, transmittance };
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}
