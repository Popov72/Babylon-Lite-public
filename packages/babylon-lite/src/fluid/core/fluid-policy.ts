import type { FluidFlowConfig, FluidTransform, FluidVec3 } from "./sim-common.js";

export interface FluidFlowTransform {
    readonly translation?: readonly [number, number, number];
    readonly rotation?: readonly [number, number, number, number];
    readonly scale?: readonly [number, number, number];
}

export interface FluidGridResolutionFit {
    readonly requestedResolution: number;
    readonly fittedResolution: number;
}

export interface FluidFlipDiscretizationLimits {
    readonly minimumResolution?: number;
    readonly maximumResolution?: number;
    readonly minimumMarkersPerCell?: number;
    readonly maximumMarkersPerCell?: number;
}

export interface FluidFlipDiscretization {
    readonly gridResolution: number;
    readonly markersPerCell: number;
}

export interface FluidRenderModeInput {
    readonly method: "PBF" | "FLIP" | "MLS-MPM" | "PB-MPM";
    readonly renderSpheres: boolean;
    readonly anisotropicSurface: boolean;
    readonly polygonSurface: boolean;
    readonly foamEnabled: boolean;
    readonly surfaceDebugActive?: boolean;
}

export interface FluidRenderMode {
    readonly particleEnabled: boolean;
    readonly surfaceMode: "surface" | "ellipsoidDebug" | "blit";
    readonly polygonEnabled: boolean;
    readonly foamEnabled: boolean;
    readonly foamPolygonSurfaceDepth: boolean;
    readonly diagnosticMode: "surface" | "spheres" | "ellipsoids" | "polygon";
}

function rotateVector(value: readonly [number, number, number], rotation: readonly [number, number, number, number]): FluidVec3 {
    const [x, y, z] = value;
    const [qx, qy, qz, qw] = rotation;
    const tx = 2 * (qy * z - qz * y);
    const ty = 2 * (qz * x - qx * z);
    const tz = 2 * (qx * y - qy * x);
    return [x + qw * tx + qy * tz - qz * ty, y + qw * ty + qz * tx - qx * tz, z + qw * tz + qx * ty - qy * tx];
}

function multiplyQuaternion(left: readonly [number, number, number, number], right: readonly [number, number, number, number]): [number, number, number, number] {
    return [
        left[3] * right[0] + left[0] * right[3] + left[1] * right[2] - left[2] * right[1],
        left[3] * right[1] - left[0] * right[2] + left[1] * right[3] + left[2] * right[0],
        left[3] * right[2] + left[0] * right[1] - left[1] * right[0] + left[2] * right[3],
        left[3] * right[3] - left[0] * right[0] - left[1] * right[1] - left[2] * right[2],
    ];
}

function transformObjectTransform(value: FluidTransform, transform: FluidFlowTransform): FluidTransform {
    const translation = transform.translation ?? [0, 0, 0];
    const rotation = transform.rotation ?? [0, 0, 0, 1];
    const scale = transform.scale ?? [1, 1, 1];
    const scaled: FluidVec3 = [value.position[0] * scale[0], value.position[1] * scale[1], value.position[2] * scale[2]];
    const rotated = rotateVector(scaled, rotation);
    return {
        position: [rotated[0] + translation[0], rotated[1] + translation[1], rotated[2] + translation[2]],
        rotation: multiplyQuaternion(rotation, value.rotation),
        scale: [value.scale[0] * scale[0], value.scale[1] * scale[1], value.scale[2] * scale[2]],
    };
}

/** Apply a scene transform to declarative flow objects without mutating the source flow. */
export function transformFluidFlow(flow: FluidFlowConfig, transform: FluidFlowTransform): FluidFlowConfig {
    const rotation = transform.rotation ?? [0, 0, 0, 1];
    return {
        emitters: flow.emitters.map((emitter) => ({
            ...structuredClone(emitter),
            transform: transformObjectTransform(emitter.transform, transform),
            ...(emitter.velocitySpace === "world" ? { velocity: rotateVector(emitter.velocity, rotation) } : {}),
            ...(emitter.sourceVelocity ? { sourceVelocity: rotateVector(emitter.sourceVelocity, rotation) } : {}),
        })),
        sinks: flow.sinks.map((sink) => ({
            ...structuredClone(sink),
            transform: transformObjectTransform(sink.transform, transform),
        })),
        ...(flow.initialEmittersFillCapacity !== undefined ? { initialEmittersFillCapacity: flow.initialEmittersFillCapacity } : {}),
        ...(flow._legacyEmitter !== undefined ? { _legacyEmitter: structuredClone(flow._legacyEmitter) } : {}),
    };
}

/** Select the highest resolution no greater than the request that satisfies authoritative planning. */
export function fitFluidGridResolution(requestedResolution: number, minimumResolution: number, fits: (resolution: number) => boolean): FluidGridResolutionFit | null {
    if (!Number.isFinite(requestedResolution) || !Number.isFinite(minimumResolution)) {
        throw new TypeError("[fluid] grid-resolution fitting requires finite bounds.");
    }

    const requested = Math.max(1, Math.floor(requestedResolution));
    const minimum = Math.max(1, Math.floor(minimumResolution));
    if (requested < minimum) {
        throw new RangeError("[fluid] requested grid resolution is below the minimum.");
    }
    for (let resolution = requested; resolution >= minimum; resolution--) {
        if (fits(resolution)) {
            return { requestedResolution: requested, fittedResolution: resolution };
        }
    }
    return null;
}

/** Normalize authored FLIP discretization through one shared import/control policy. */
export function normalizeFluidFlipDiscretization(gridResolution: number, markersPerCell: number, limits: FluidFlipDiscretizationLimits = {}): FluidFlipDiscretization {
    const minimumResolution = Math.max(1, Math.floor(limits.minimumResolution ?? 16));
    const maximumResolution = Math.max(minimumResolution, Math.floor(limits.maximumResolution ?? Number.MAX_SAFE_INTEGER));
    const minimumMarkers = Math.max(1, Math.floor(limits.minimumMarkersPerCell ?? 1));
    const maximumMarkers = Math.max(minimumMarkers, Math.floor(limits.maximumMarkersPerCell ?? 64));
    if (![gridResolution, markersPerCell, minimumResolution, maximumResolution, minimumMarkers, maximumMarkers].every(Number.isFinite)) {
        throw new TypeError("[fluid] FLIP discretization values and limits must be finite.");
    }
    return {
        gridResolution: Math.max(minimumResolution, Math.min(maximumResolution, Math.round(gridResolution))),
        markersPerCell: Math.max(minimumMarkers, Math.min(maximumMarkers, Math.round(markersPerCell))),
    };
}

/** Resolve mutually dependent sphere, surface, polygon, and foam rendering state. */
export function resolveFluidRenderMode(input: FluidRenderModeInput): FluidRenderMode {
    const polygonEnabled = !input.renderSpheres && input.method === "FLIP" && input.polygonSurface;
    const ellipsoids = input.renderSpheres && input.anisotropicSurface;
    const particles = input.renderSpheres && !input.anisotropicSurface;
    return {
        particleEnabled: particles,
        surfaceMode: ellipsoids ? "ellipsoidDebug" : particles || polygonEnabled ? "blit" : "surface",
        polygonEnabled,
        foamEnabled: input.foamEnabled && !input.surfaceDebugActive,
        foamPolygonSurfaceDepth: polygonEnabled,
        diagnosticMode: ellipsoids ? "ellipsoids" : particles ? "spheres" : polygonEnabled ? "polygon" : "surface",
    };
}

export function formatFluidPageDiagnostics(requiredPages: number, capacity: number): string {
    return `${Math.max(0, Math.floor(requiredPages)).toLocaleString()}\u00a0/\u00a0${Math.max(0, Math.floor(capacity)).toLocaleString()}\u00a0pages`;
}
