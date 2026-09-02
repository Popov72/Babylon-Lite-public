// Production Aquanova fluid capability gates.

import { fluidRuntimeCapabilityRejection } from "babylon-lite";
import type { FluidRuntimeCapabilities } from "babylon-lite";

/** Production uses the shared collection surface, polygon, and foam renderers. */
export const AQUANOVA_PRODUCTION_FLUID_CAPABILITIES: FluidRuntimeCapabilities = {
    polygonSurface: true,
    independentRendering: true,
};

/**
 * Reason a production fluid preset cannot run, or null when it is renderable.
 */
export function productionFluidCapabilityRejection(method: string, schema: Record<string, number> | undefined): string | null {
    return fluidRuntimeCapabilityRejection(AQUANOVA_PRODUCTION_FLUID_CAPABILITIES, { method, physics: schema });
}
