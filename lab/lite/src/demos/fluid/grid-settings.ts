export const PHYS_MIN_SCALE = 0.1;
export const PHYS_MAX_SCALE = 8;
export const PBF_MIN_SCALE = 0.1;
export const PBF_MAX_SCALE = 8;
export const MPM_MIN_SCALE = 0.1;
export const MPM_MAX_SCALE = 8;
export const PBMPM_MIN_SCALE = 0.1;
export const PBMPM_MAX_SCALE = 8;

export const GRID_DOMAIN_LONGEST = 40;
export const GRID_RESOLUTION_MIN = 16;
export const GRID_RESOLUTION_MAX = 2048;

const PBF_BASE_CELL_SIZE = 0.4;
const MPM_BASE_CELL_SIZE = 0.22;

const baseCellSizeForMethod = (method: string): number => (method === "PBF" ? PBF_BASE_CELL_SIZE : MPM_BASE_CELL_SIZE);
export const cellSizeForPhysicsScale = (method: string, scale: number): number => baseCellSizeForMethod(method) * scale;
export const physicsScaleForCellSize = (method: string, cellSize: number): number => cellSize / baseCellSizeForMethod(method);
export const gridWorldSize = (cells: readonly [number, number, number], cellSize: number): [number, number, number] => [
    cells[0] * cellSize,
    cells[1] * cellSize,
    cells[2] * cellSize,
];
export const gridBounds = (
    position: readonly [number, number, number],
    size: readonly [number, number, number]
): { min: [number, number, number]; max: [number, number, number] } => ({
    min: [position[0] - size[0] * 0.5, position[1] - size[1] * 0.5, position[2] - size[2] * 0.5],
    max: [position[0] + size[0] * 0.5, position[1] + size[1] * 0.5, position[2] + size[2] * 0.5],
});
export const gridPositionForBounds = (bounds: { min: readonly [number, number, number]; max: readonly [number, number, number] }): [number, number, number] => [
    (bounds.min[0] + bounds.max[0]) * 0.5,
    (bounds.min[1] + bounds.max[1]) * 0.5,
    (bounds.min[2] + bounds.max[2]) * 0.5,
];
export const gridSizeForBounds = (bounds: { min: readonly [number, number, number]; max: readonly [number, number, number] }): [number, number, number] => [
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
];
export const gridCellsForSize = (size: readonly [number, number, number], cellSize: number): [number, number, number] =>
    size.map((extent) => {
        const exactCells = extent / cellSize;
        const nearestInteger = Math.round(exactCells);
        const tolerance = Number.EPSILON * 16 * Math.max(1, Math.abs(exactCells));
        const cells = Math.abs(exactCells - nearestInteger) <= tolerance ? nearestInteger : Math.ceil(exactCells);
        return Math.max(4, cells);
    }) as [number, number, number];
export const gridCellsForBounds = (bounds: { min: readonly [number, number, number]; max: readonly [number, number, number] }, cellSize: number): [number, number, number] =>
    gridCellsForSize(gridSizeForBounds(bounds), cellSize);
const baseGridResolutionForMethod = (method: string, domainLongest = GRID_DOMAIN_LONGEST): number => domainLongest / baseCellSizeForMethod(method);
export const scaleLimitsForMethod = (method: string): [number, number] =>
    method === "PBF" ? [PBF_MIN_SCALE, PBF_MAX_SCALE] : method === "PB-MPM" ? [PBMPM_MIN_SCALE, PBMPM_MAX_SCALE] : [MPM_MIN_SCALE, MPM_MAX_SCALE];

export function gridResolutionLimitsForMethod(method: string, domainLongest = GRID_DOMAIN_LONGEST): [number, number] {
    const [minScale, maxScale] = scaleLimitsForMethod(method);
    const baseResolution = baseGridResolutionForMethod(method, domainLongest);
    return [Math.max(GRID_RESOLUTION_MIN, Math.ceil(baseResolution / maxScale)), Math.min(GRID_RESOLUTION_MAX, Math.floor(baseResolution / minScale))];
}

export function clampGridResolution(method: string, resolution: number, domainLongest = GRID_DOMAIN_LONGEST): number {
    const [minResolution, maxResolution] = gridResolutionLimitsForMethod(method, domainLongest);
    return Math.min(maxResolution, Math.max(minResolution, Math.round(resolution)));
}

export function rawScaleForGridResolution(method: string, resolution: number, domainLongest = GRID_DOMAIN_LONGEST): number {
    return baseGridResolutionForMethod(method, domainLongest) / Math.max(1, resolution);
}

export function scaleForGridResolution(method: string, resolution: number, domainLongest = GRID_DOMAIN_LONGEST): number {
    const [minScale, maxScale] = scaleLimitsForMethod(method);
    return Math.min(maxScale, Math.max(minScale, rawScaleForGridResolution(method, clampGridResolution(method, resolution, domainLongest), domainLongest)));
}

export function gridResolutionForScale(method: string, scale: number, domainLongest = GRID_DOMAIN_LONGEST): number {
    return clampGridResolution(method, baseGridResolutionForMethod(method, domainLongest) / scale, domainLongest);
}

export function cellSizeForGridResolution(resolution: number, domainLongest = GRID_DOMAIN_LONGEST): number {
    return domainLongest / resolution;
}
