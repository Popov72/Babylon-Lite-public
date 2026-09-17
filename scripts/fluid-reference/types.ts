export type Triple = [number, number, number];

export interface ReferenceGrid {
    origin: Triple;
    dimensions: Triple;
    cellSize: number;
}

export interface ReferenceSdf {
    origin: Triple;
    dimensions: Triple;
    cellSize: number;
    file: string;
}

export interface ReferenceObstacle extends ReferenceSdf {
    name: string;
    /** Column-major local-to-world transforms at each physical substep boundary. */
    transforms: number[][];
    inverseTransforms: number[][];
    /** Native interpolated backward/forward vertex displacement per physical second. */
    velocityTransforms: number[][];
}

export interface ReferenceMesh {
    name: string;
    positions: string;
    indices: string;
    color: Triple;
    transform: number[];
    obstacle?: number;
}

export interface ReferenceCase {
    version: 1;
    label: string;
    grid: ReferenceGrid;
    /** Native solid domain shell thickness; zero for analytic full-domain probes. */
    domainInset: number;
    startFrame: number;
    frames: number;
    simulationFps: number;
    timelineFps: number;
    substeps: number;
    gravity: Triple;
    picFraction: number;
    extremeVelocityRemoval?: { cfl: number; maxFrameSubsteps: number };
    initialState: string;
    staticSdf?: ReferenceSdf;
    obstacles: ReferenceObstacle[];
    meshes: ReferenceMesh[];
    camera: { alpha: number; beta: number; radius: number; target: Triple; fov: number; mirrorX: boolean };
    referencePattern?: string;
    provenance: Record<string, unknown>;
}

export interface ParticleState {
    positions: Float32Array;
    velocities: Float32Array;
}

export interface ParticleMetrics {
    count: number;
    center: Triple | null;
    lower: Triple | null;
    upper: Triple | null;
    p01: Triple | null;
    p99: Triple | null;
    meanSpeed: number | null;
    meanKineticEnergy: number | null;
    occupiedCells: number;
    outsideParticles: number;
    heightByZ: Array<number | null>;
}

export interface ParticleComparison {
    centerDistance: number | null;
    occupancyIntersectionOverUnion: number;
    normalizedOccupancyDifference: number;
    countDifference: number;
}

export function decodeParticleState(buffer: ArrayBuffer): ParticleState {
    if (buffer.byteLength < 8) {
        throw new Error("Particle snapshot is missing its header.");
    }
    const count = new DataView(buffer).getUint32(0, true);
    if (buffer.byteLength !== 8 + count * 24) {
        throw new Error("Particle snapshot count does not match its byte length.");
    }
    return {
        positions: new Float32Array(buffer, 8, count * 3),
        velocities: new Float32Array(buffer, 8 + count * 12, count * 3),
    };
}
