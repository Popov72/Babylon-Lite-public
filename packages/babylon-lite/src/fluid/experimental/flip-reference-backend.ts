import type { FluidSimulationBackend } from "../core/fluid-facade.js";

/** Load the accuracy-oriented, GPU-resident FLIP implementation explicitly. */
export async function loadFlipReferenceBackend(): Promise<FluidSimulationBackend> {
    const { createFlipReferenceFluidSim, planFlipReferenceFluidAllocation } = await import("./flip-reference/facade-adapter.js");
    return {
        id: "flip-reference",
        name: "FLIP Reference",
        method: "FLIP",
        steppingMode: "frame",
        renderModes: ["spheres", "surface"],
        supportsFoam: true,
        supportsForces: true,
        supportsContinuousFlow: false,
        physicsParameters: ["gravity", "flipRatio", "minSubsteps", "maxSubsteps", "maxSubDtMs", "cflNumber"],
        description:
            "Experimental, accuracy-oriented FLIP. Initial fluid, manual/custom forces, screen-space water or spheres, foam/spray/bubbles. Native-style velocity-outlier cleanup follows CFL. No polygons, inflows, sinks, viscosity or surface tension.",
        _create: createFlipReferenceFluidSim,
        _allocationPlan: planFlipReferenceFluidAllocation,
    };
}
