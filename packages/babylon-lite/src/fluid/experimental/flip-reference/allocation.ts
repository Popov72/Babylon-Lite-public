export const FLIP_REFERENCE_PARAMETER_FLOATS = 44;
export const FLIP_REFERENCE_MAX_FRAME_SUBSTEPS = 256;

/** @internal Shared by allocation and UI projection; order matches the solver's storage bindings. */
export function flipReferenceStorageSizes(cells: number, faces: number, vertices: number, particles: number, removal: boolean, histogramBins: number): number[] {
    return [
        Math.max(16, particles * 16),
        Math.max(16, particles * 16),
        Math.max(4, particles * 4),
        faces * 64,
        cells * 64,
        (cells + particles + 8) * 4,
        vertices * 16,
        (cells + Math.ceil(cells / 128) + 2) * 16,
        removal ? Math.max(16, particles * 32) : 16,
        removal ? (particles * 2 + Math.max(1, Math.ceil(particles / 128)) + 8 + histogramBins) * 4 : 4,
    ];
}

/** @internal */
export function flipReferenceReadbackBytes(cells: number, particles: number): number {
    return Math.max(cells * 64 + 32, particles * 32, 32);
}
