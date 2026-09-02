import type { PairState } from "./authoring-state.js";

export interface CarryMethodIndependentStateOptions {
    /** Retain the target pair's material-specific color and render mode. */
    retainTargetPresentation?: boolean;
}

/** Overlay solver-independent authoring state while retaining target-method-only controls. */
export function carryMethodIndependentState(target: PairState, shared: PairState, options: CarryMethodIndependentStateOptions = {}): PairState {
    const gravity = shared.schema.gravity;
    return {
        ...structuredClone(shared),
        schema: {
            ...target.schema,
            ...(gravity !== undefined ? { gravity } : {}),
        },
        demoParams: { ...target.demoParams },
        demoState: target.demoState ? structuredClone(target.demoState) : undefined,
        ...(options.retainTargetPresentation ? { color: target.color, renderMode: target.renderMode } : {}),
        material: target.material,
        gridResolution: target.gridResolution,
        markersPerCell: target.markersPerCell,
        activeBlocks: target.activeBlocks,
        pagedGrid: target.pagedGrid,
        pagedGridMaxPages: target.pagedGridMaxPages,
        fusedBlockDiscovery: target.fusedBlockDiscovery,
    };
}
