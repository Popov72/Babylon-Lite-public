import { fluidShapeVolume } from "./sim-common.js";
import type { FluidFlowConfig } from "./sim-common.js";

const FLUID_BASE_PARTICLE_RADIUS = 0.08;
const MPM_MIN_CELL_SIZE = 0.18;
const MPM_CELL_RADIUS_RATIO = 2.4;
const PBF_MIN_CELL_SIZE = 0.3;
const PBF_CELL_RADIUS_RATIO = 4;

export type FluidSimulationSamplingType = "fluid" | "mesh";

export interface FluidSimulationDiscretization {
    physicsParticleSize: number;
    samplingType: FluidSimulationSamplingType;
    particleRadius?: number;
}

export const fluidParticleRadiusForPhysicsScale = (physicsScale: number): number => FLUID_BASE_PARTICLE_RADIUS * physicsScale;

export const fluidCellSizeForParticleRadius = (method: string, particleRadius: number): number =>
    method === "PBF" ? Math.max(particleRadius * PBF_CELL_RADIUS_RATIO, PBF_MIN_CELL_SIZE) : Math.max(particleRadius * MPM_CELL_RADIUS_RATIO, MPM_MIN_CELL_SIZE);

export function fluidSimulationParticleRadius(config: FluidSimulationDiscretization): number {
    const authoredRadius = config.particleRadius;
    return config.samplingType === "fluid" || typeof authoredRadius !== "number" || !Number.isFinite(authoredRadius) || authoredRadius <= 0
        ? fluidParticleRadiusForPhysicsScale(config.physicsParticleSize)
        : authoredRadius;
}

export const fluidSimulationCellSize = (method: string, config: FluidSimulationDiscretization): number =>
    fluidCellSizeForParticleRadius(method, fluidSimulationParticleRadius(config));

export function fluidSimulationParticleCapacity(method: string, requestedCount: number, flow: FluidFlowConfig, particleVolume: number): number {
    const emitters = flow.emitters.filter((emitter) => emitter.enabled);
    const effectiveParticleVolume = Math.max(particleVolume, 1e-6);
    const initialDemand = emitters
        .filter((emitter) => emitter.behavior === "initial")
        .reduce((sum, emitter) => sum + Math.ceil(fluidShapeVolume(emitter.shape, emitter.transform) / effectiveParticleVolume), 0);
    const inflowDemand = emitters.some((emitter) => emitter.behavior === "inflow") ? Math.max(initialDemand * 2, 20000) : 0;
    const automatic = Math.max(initialDemand + inflowDemand, initialDemand || 20000);
    const requested = requestedCount > 0 ? Math.round(requestedCount) : automatic;
    return Math.max(1, method === "FLIP" ? Math.max(initialDemand, requested) : requested);
}
