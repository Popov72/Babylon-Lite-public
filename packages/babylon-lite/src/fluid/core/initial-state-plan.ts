import {
    resolveFluidAllocationPlan,
    resolveFluidParticleCapacity,
    type FluidAllocationPlan,
    type FluidAllocationPlanInput,
    type FluidParticleCapacityPlan,
} from "./allocation-plan.js";
import {
    MAX_FLUID_EMITTERS,
    countFluidInitialParticles,
    fluidShapeVolume,
    type FluidEmitter,
    type FluidFlowConfig,
    type FluidInitialBounds,
    type FluidShape,
} from "./sim-common.js";

export interface FluidInitialStatePlanInput {
    particleCapacity: number;
    particleVolume: number;
    flow: FluidFlowConfig | null;
    bounds?: FluidInitialBounds;
    deriveInitialCount?: boolean;
}

export interface FluidInitialEmitterPlan {
    readonly id: string;
    readonly shape: FluidShape["type"];
    readonly sampling: "volume" | "surface";
    readonly authoredVolume: number;
    /** Density-derived sites accepted by the same clipped lattice policy as reset seeding. */
    readonly clippedRequiredCount: number;
    readonly clippedVolume: number;
    /** Sites assigned to this emitter at the requested particle capacity. */
    readonly activeCount: number;
}

export interface FluidInitialStateDiagnostics {
    readonly mode: "legacy-spawn" | "empty-flow" | "initial-emitters";
    readonly exactClippedSites: boolean;
    readonly particleCapacityNormalized: boolean;
    readonly particleVolumeNormalized: boolean;
    readonly authoredRequiredCount: number;
    readonly capacityShortfall: number;
}

export interface FluidResolutionFittingInputs {
    readonly particleCapacity: number;
    readonly particleVolume: number;
    readonly requiredCount: number;
    readonly authoredVolume: number;
    readonly clippedVolume: number;
}

export interface FluidInitialStatePlan {
    /** Stable identity for every input that can change reset-time particle placement. */
    readonly initialStateKey: string;
    /** Bounds-clipped density demand used when fitting capacity or resolution. */
    readonly requiredCount: number;
    /** Reset-time active prefix at the requested capacity and flow semantics. */
    readonly activeCount: number;
    readonly emitterCounts: ReadonlyMap<string, number>;
    readonly emitters: readonly FluidInitialEmitterPlan[];
    readonly authoredVolume: number;
    /** Quantized volume represented by accepted lattice sites. */
    readonly clippedVolume: number;
    readonly resolutionFitting: FluidResolutionFittingInputs;
    readonly diagnostics: FluidInitialStateDiagnostics;
}

export interface FluidInitialStatePlanCache {
    readonly resolve: (input: FluidInitialStatePlanInput) => FluidInitialStatePlan;
    readonly clear: () => void;
}

export interface FluidReconfigurationPlanInput extends Omit<FluidInitialStatePlanInput, "particleCapacity"> {
    allocation: FluidAllocationPlanInput;
    maximumParticleCount?: number;
}

export interface FluidReconfigurationPlan {
    readonly allocation: FluidAllocationPlan;
    readonly particleCapacity: FluidParticleCapacityPlan;
    readonly initialState: FluidInitialStatePlan;
    readonly resolutionFitting: FluidResolutionFittingInputs & {
        readonly maximumParticleCapacity: number;
        readonly fitsDeviceLimits: boolean;
        readonly fitsInitialDemand: boolean;
    };
}

function normalizeParticleCapacity(value: number): number {
    return Number.isFinite(value) ? Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, Math.floor(value))) : 0;
}

function normalizeParticleVolume(value: number): number {
    return Number.isFinite(value) && value > 0 ? value : 1;
}

function requiredParticleCount(volume: number, particleVolume: number): number {
    return Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, Math.ceil(volume / particleVolume)));
}

function fluidInitialStateKey(input: FluidInitialStatePlanInput, particleCapacity: number, particleVolume: number): string {
    return JSON.stringify({
        particleCapacity,
        particleVolume,
        bounds: input.bounds ?? null,
        deriveInitialCount: input.deriveInitialCount ?? false,
        initialEmittersFillCapacity: input.flow?.initialEmittersFillCapacity ?? false,
        emitters: input.flow?.emitters.slice(0, MAX_FLUID_EMITTERS).filter((emitter) => emitter.enabled && emitter.behavior === "initial") ?? null,
    });
}

function initialEmitters(flow: FluidFlowConfig | null): readonly FluidEmitter[] {
    return flow?.emitters.slice(0, MAX_FLUID_EMITTERS).filter((emitter) => emitter.enabled && emitter.behavior === "initial") ?? [];
}

/** Returns the authored world-space volume of enabled reset-time emitters. */
export function fluidInitialEmitterVolume(flow: FluidFlowConfig | null): number {
    return initialEmitters(flow).reduce((sum, emitter) => sum + fluidShapeVolume(emitter.shape, emitter.transform), 0);
}

/**
 * Plans reset-time particles without allocating particle arrays. Volume emitters delegate to the
 * same deterministic clipped-lattice counter used by `createFluidInitialParticles`.
 */
export function planFluidInitialState(input: FluidInitialStatePlanInput): FluidInitialStatePlan {
    const particleCapacity = normalizeParticleCapacity(input.particleCapacity);
    const particleVolume = normalizeParticleVolume(input.particleVolume);
    const initialStateKey = fluidInitialStateKey(input, particleCapacity, particleVolume);
    if (!input.flow) {
        const resolutionFitting = {
            particleCapacity,
            particleVolume,
            requiredCount: particleCapacity,
            authoredVolume: 0,
            clippedVolume: 0,
        };
        return {
            initialStateKey,
            requiredCount: particleCapacity,
            activeCount: particleCapacity,
            emitterCounts: new Map(),
            emitters: [],
            authoredVolume: 0,
            clippedVolume: 0,
            resolutionFitting,
            diagnostics: {
                mode: "legacy-spawn",
                exactClippedSites: false,
                particleCapacityNormalized: particleCapacity !== input.particleCapacity,
                particleVolumeNormalized: particleVolume !== input.particleVolume,
                authoredRequiredCount: particleCapacity,
                capacityShortfall: 0,
            },
        };
    }

    const emitters = initialEmitters(input.flow);
    const authoredVolumes = emitters.map((emitter) => fluidShapeVolume(emitter.shape, emitter.transform));
    const authoredVolume = authoredVolumes.reduce((sum, volume) => sum + volume, 0);
    const authoredRequiredCount = requiredParticleCount(authoredVolume, particleVolume);
    const densityFlow: FluidFlowConfig = {
        ...input.flow,
        initialEmittersFillCapacity: false,
    };
    const clippedCounts = countFluidInitialParticles(authoredRequiredCount, densityFlow, particleVolume, input.bounds, true);
    const activeCounts = countFluidInitialParticles(particleCapacity, input.flow, particleVolume, input.bounds, input.deriveInitialCount ?? false);
    const requiredCount = clippedCounts?.activeCount ?? 0;
    const activeCount = activeCounts?.activeCount ?? 0;
    const emitterCounts = activeCounts?.emitterCounts ?? new Map<string, number>();
    const emitterPlans = emitters.map((emitter, index): FluidInitialEmitterPlan => {
        const clippedRequiredCount = clippedCounts?.emitterCounts.get(emitter.id) ?? 0;
        return {
            id: emitter.id,
            shape: emitter.shape.type,
            sampling: emitter.sampling,
            authoredVolume: authoredVolumes[index]!,
            clippedRequiredCount,
            clippedVolume: clippedRequiredCount * particleVolume,
            activeCount: emitterCounts.get(emitter.id) ?? 0,
        };
    });
    const clippedVolume = requiredCount * particleVolume;
    const resolutionFitting = {
        particleCapacity,
        particleVolume,
        requiredCount,
        authoredVolume,
        clippedVolume,
    };
    return {
        initialStateKey,
        requiredCount,
        activeCount,
        emitterCounts,
        emitters: emitterPlans,
        authoredVolume,
        clippedVolume,
        resolutionFitting,
        diagnostics: {
            mode: emitters.length === 0 ? "empty-flow" : "initial-emitters",
            exactClippedSites: emitters.every((emitter) => emitter.sampling === "volume"),
            particleCapacityNormalized: particleCapacity !== input.particleCapacity,
            particleVolumeNormalized: particleVolume !== input.particleVolume,
            authoredRequiredCount,
            capacityShortfall: Math.max(0, requiredCount - particleCapacity),
        },
    };
}

/** Creates a bounded per-consumer cache for exact initial-state plans. */
export function createFluidInitialStatePlanCache(maxEntries = 8): FluidInitialStatePlanCache {
    const capacity = Number.isFinite(maxEntries) ? Math.max(1, Math.floor(maxEntries)) : 8;
    const plans = new Map<string, FluidInitialStatePlan>();
    return {
        resolve(input) {
            const particleCapacity = normalizeParticleCapacity(input.particleCapacity);
            const particleVolume = normalizeParticleVolume(input.particleVolume);
            const key = fluidInitialStateKey(input, particleCapacity, particleVolume);
            const cached = plans.get(key);
            if (cached) {
                plans.delete(key);
                plans.set(key, cached);
                return cached;
            }
            const plan = planFluidInitialState(input);
            if (plans.size >= capacity) {
                const oldest = plans.keys().next().value;
                if (oldest !== undefined) {
                    plans.delete(oldest);
                }
            }
            plans.set(plan.initialStateKey, plan);
            return plan;
        },
        clear() {
            plans.clear();
        },
    };
}

/** Combines exact allocation limits and initial-state demand for a future host rebuild decision. */
export function resolveFluidReconfigurationPlan(input: FluidReconfigurationPlanInput): FluidReconfigurationPlan {
    const allocation = resolveFluidAllocationPlan(input.allocation);
    const { particleCount: _particleCount, previousSteadyBytes: _previousSteadyBytes, maxParticleBudget: _maxParticleBudget, ...capacityContext } = input.allocation;
    const particleCapacity = resolveFluidParticleCapacity({
        ...capacityContext,
        maximumParticleCount: input.maximumParticleCount,
    });
    const initialState = planFluidInitialState({
        particleCapacity: allocation.dimensions.particleCount,
        particleVolume: input.particleVolume,
        flow: input.flow,
        bounds: input.bounds,
        deriveInitialCount: input.deriveInitialCount,
    });
    return {
        allocation,
        particleCapacity,
        initialState,
        resolutionFitting: {
            ...initialState.resolutionFitting,
            maximumParticleCapacity: particleCapacity.capacity,
            fitsDeviceLimits: allocation.dimensions.particleCount <= particleCapacity.capacity,
            fitsInitialDemand: initialState.requiredCount <= allocation.dimensions.particleCount,
        },
    };
}
