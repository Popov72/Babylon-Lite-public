export type ViewVolumeDepthMapping = { readonly kind: "linear" } | { readonly kind: "log" } | { readonly kind: "power"; readonly exponent: number };

export interface ViewVolumeGridOptions {
    readonly targetWidth: number;
    readonly targetHeight: number;
    readonly tileSize: number;
    readonly depthSlices: number;
    readonly nearDepth: number;
    readonly farDepth: number;
    readonly depthMapping: ViewVolumeDepthMapping;
}

export interface ViewVolumeGrid {
    readonly targetWidth: number;
    readonly targetHeight: number;
    readonly tileSize: number;
    readonly width: number;
    readonly height: number;
    readonly depth: number;
    readonly nearDepth: number;
    readonly farDepth: number;
    readonly depthMapping: ViewVolumeDepthMapping;
}

export interface ViewVolumeSliceBounds {
    readonly nearDepth: number;
    readonly farDepth: number;
    readonly centerDepth: number;
    readonly thickness: number;
}

export function createViewVolumeGrid(options: ViewVolumeGridOptions): ViewVolumeGrid {
    const targetWidth = positiveInteger(options.targetWidth, "targetWidth");
    const targetHeight = positiveInteger(options.targetHeight, "targetHeight");
    const tileSize = positiveInteger(options.tileSize, "tileSize");
    const depth = positiveInteger(options.depthSlices, "depthSlices");
    const nearDepth = positiveFinite(options.nearDepth, "nearDepth");
    const farDepth = positiveFinite(options.farDepth, "farDepth");
    if (farDepth <= nearDepth) {
        throw new RangeError(`ViewVolumeGrid: farDepth ${farDepth} must be greater than nearDepth ${nearDepth}.`);
    }
    const depthMapping = validateDepthMapping(options.depthMapping);
    return {
        targetWidth,
        targetHeight,
        tileSize,
        width: Math.ceil(targetWidth / tileSize),
        height: Math.ceil(targetHeight / tileSize),
        depth,
        nearDepth,
        farDepth,
        depthMapping,
    };
}

export function viewDepthToVolumeSlice(grid: ViewVolumeGrid, viewDepth: number): number {
    const normalized = normalizeViewDepth(grid, viewDepth);
    switch (grid.depthMapping.kind) {
        case "linear":
            return normalized * grid.depth;
        case "log":
            return (Math.log(clamp(viewDepth, grid.nearDepth, grid.farDepth) / grid.nearDepth) / Math.log(grid.farDepth / grid.nearDepth)) * grid.depth;
        case "power":
            return Math.pow(normalized, 1 / grid.depthMapping.exponent) * grid.depth;
    }
}

export function volumeSliceToViewDepth(grid: ViewVolumeGrid, sliceCoordinate: number): number {
    const normalized = clamp(sliceCoordinate / grid.depth, 0, 1);
    switch (grid.depthMapping.kind) {
        case "linear":
            return grid.nearDepth + (grid.farDepth - grid.nearDepth) * normalized;
        case "log":
            return grid.nearDepth * Math.pow(grid.farDepth / grid.nearDepth, normalized);
        case "power":
            return grid.nearDepth + (grid.farDepth - grid.nearDepth) * Math.pow(normalized, grid.depthMapping.exponent);
    }
}

export function getViewVolumeSliceBounds(grid: ViewVolumeGrid, sliceIndex: number): ViewVolumeSliceBounds {
    if (!Number.isInteger(sliceIndex) || sliceIndex < 0 || sliceIndex >= grid.depth) {
        throw new RangeError(`ViewVolumeGrid: sliceIndex ${sliceIndex} is outside [0, ${grid.depth - 1}].`);
    }
    const nearDepth = volumeSliceToViewDepth(grid, sliceIndex);
    const farDepth = volumeSliceToViewDepth(grid, sliceIndex + 1);
    return {
        nearDepth,
        farDepth,
        centerDepth: volumeSliceToViewDepth(grid, sliceIndex + 0.5),
        thickness: farDepth - nearDepth,
    };
}

export function viewVolumeTextureBytes(grid: ViewVolumeGrid, bytesPerTexel: number): number {
    const bytes = positiveInteger(bytesPerTexel, "bytesPerTexel");
    const total = grid.width * grid.height * grid.depth * bytes;
    if (!Number.isSafeInteger(total)) {
        throw new RangeError("ViewVolumeGrid: texture byte size exceeds JavaScript's safe integer range.");
    }
    return total;
}

function normalizeViewDepth(grid: ViewVolumeGrid, viewDepth: number): number {
    if (!Number.isFinite(viewDepth)) {
        throw new RangeError(`ViewVolumeGrid: viewDepth must be finite (got ${viewDepth}).`);
    }
    return clamp((viewDepth - grid.nearDepth) / (grid.farDepth - grid.nearDepth), 0, 1);
}

function validateDepthMapping(mapping: ViewVolumeDepthMapping): ViewVolumeDepthMapping {
    if (mapping.kind !== "power") {
        return mapping;
    }
    const exponent = positiveFinite(mapping.exponent, "depthMapping.exponent");
    return { kind: "power", exponent };
}

function positiveFinite(value: number, name: string): number {
    if (!Number.isFinite(value) || value <= 0) {
        throw new RangeError(`ViewVolumeGrid: ${name} must be finite and greater than zero (got ${value}).`);
    }
    return value;
}

function positiveInteger(value: number, name: string): number {
    if (!Number.isSafeInteger(value) || value <= 0) {
        throw new RangeError(`ViewVolumeGrid: ${name} must be a positive safe integer (got ${value}).`);
    }
    return value;
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}
