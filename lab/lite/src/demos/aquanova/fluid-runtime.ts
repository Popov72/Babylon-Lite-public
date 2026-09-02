import {
    createFluidParticleSpatialQuery,
    disposeFluidParticleSpatialQuery,
    FLUID_SPATIAL_QUERY_SAMPLE_INTERVAL_FRAMES,
    readFluidParticleSpatialQuery,
    sampleFluidParticleSpatialQuery,
} from "babylon-lite";
import type { EngineContext, FluidParticleSpatialQuery, FluidParticleSpatialQueryRequest, FluidParticleStream } from "babylon-lite";

import { FluidSimulationRuntime } from "./behaviors/fluid-simulation-runtime.js";

export interface FluidParticleAabb {
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
}

export interface FluidParticleCounterSource {
    readonly engine: EngineContext;
    readonly stream: FluidParticleStream;
}

export interface FluidElectricityState {
    readonly origin: readonly [number, number, number];
    readonly startedAtSeconds: number;
    readonly propagationSpeed: number;
}

export function electricityPropagationRadius(state: FluidElectricityState, elapsedSeconds: number): number {
    return Math.max(0, elapsedSeconds - state.startedAtSeconds) * state.propagationSpeed;
}

export interface FluidElectricityDomain {
    readonly id: number;
    readonly label: string;
    readonly electricity: FluidElectricityState | null;
}

export interface FluidElectricityFrameDomain {
    readonly domain: FluidElectricityDomain;
    readonly offset: number;
    readonly count: number;
    readonly particleRadius: number;
    readonly gridAabb: FluidParticleAabb;
    readonly elapsedSeconds: number;
}

export interface FluidElectrifierRegistration {
    readonly entityName: string;
    readonly particleThreshold: number;
    readonly propagationSpeed: number;
    readonly aabb: () => FluidParticleAabb | null;
    readonly onElectrified?: (domain: FluidElectricityDomain) => void;
}

export interface ElectrifiedFluidReceiverRegistration {
    readonly entityName: string;
    readonly particleThreshold: number;
    readonly aabb: () => FluidParticleAabb | null;
    readonly includeParticleRadius?: boolean;
    readonly onCount: (particleCount: number) => void;
}

export interface FluidElectricityRegistration {
    dispose(): void;
}

interface FluidElectricityDomainInternal extends FluidElectricityDomain {
    electricity: FluidElectricityState | null;
}

type ElectricityResult =
    | {
          readonly kind: "electrifier";
          readonly registration: FluidElectrifierRegistration;
          readonly domain: FluidElectricityDomainInternal;
          readonly origin: readonly [number, number, number];
          readonly elapsedSeconds: number;
      }
    | {
          readonly kind: "receiver";
          readonly registration: ElectrifiedFluidReceiverRegistration;
          readonly domain: FluidElectricityDomainInternal;
      };

export const ELECTRICITY_PARTICLE_MASK_RADIUS_SCALE = 1.35;
const ELECTRICITY_MAX_QUERY_PAIRS = 1024;

/** Aquanova gameplay policy over the shared opaque fluid particle-query runtime. */
export class AquanovaFluidRuntime extends FluidSimulationRuntime {
    private stream: FluidParticleStream | null = null;
    private countQuery: FluidParticleSpatialQuery | null = null;
    private electricityQuery: FluidParticleSpatialQuery | null = null;
    private requestedAabb: FluidParticleAabb | null = null;
    private requestedVersion = 0;
    private completedVersion = 0;
    private completedCount = 0;
    private sampleCooldown = 0;
    private electricityGeneration = 0;
    private electricitySampleCooldown = 0;
    private readonly pendingElectricityResults = new Map<string, ElectricityResult>();
    private nextElectricityDomainId = 1;
    private readonly electricityDomains = new Set<FluidElectricityDomainInternal>();
    private readonly electrifiers = new Set<FluidElectrifierRegistration>();
    private readonly electricityReceivers = new Set<ElectrifiedFluidReceiverRegistration>();
    private electricityPairCount = 0;
    private disposed = false;

    public installParticleCounter(source: FluidParticleCounterSource): void {
        if (this.stream) throw new Error("[aquanova] fluid particle counter is already installed");
        if (this.disposed) throw new Error("[aquanova] cannot install a fluid particle counter after disposal");
        this.stream = source.stream;
        this.countQuery = createFluidParticleSpatialQuery(source.engine, { maximumQueries: 1 });
        this.electricityQuery = createFluidParticleSpatialQuery(source.engine, { maximumQueries: ELECTRICITY_MAX_QUERY_PAIRS });
    }

    public createElectricityDomain(label: string): FluidElectricityDomain {
        if (this.disposed) throw new Error("[aquanova] cannot create a fluid electricity domain after disposal");
        const domain: FluidElectricityDomainInternal = { id: this.nextElectricityDomainId++, label, electricity: null };
        this.electricityDomains.add(domain);
        return domain;
    }

    public disposeElectricityDomain(domain: FluidElectricityDomain): void {
        this.electricityDomains.delete(domain as FluidElectricityDomainInternal);
    }

    public registerElectrifier(registration: FluidElectrifierRegistration): FluidElectricityRegistration {
        validatePositiveInteger(registration.particleThreshold, `${registration.entityName}.particleThreshold`);
        validatePositiveFinite(registration.propagationSpeed, `${registration.entityName}.propagationSpeed`);
        this.electrifiers.add(registration);
        return { dispose: () => this.electrifiers.delete(registration) };
    }

    public registerElectricityReceiver(registration: ElectrifiedFluidReceiverRegistration): FluidElectricityRegistration {
        validatePositiveInteger(registration.particleThreshold, `${registration.entityName}.particleThreshold`);
        this.electricityReceivers.add(registration);
        return { dispose: () => this.electricityReceivers.delete(registration) };
    }

    public recordElectricity(frameDomains: readonly FluidElectricityFrameDomain[]): void {
        const query = this.electricityQuery;
        const stream = this.stream;
        if (!query || !stream) return;
        const completedResults = readFluidParticleSpatialQuery(query);
        if (completedResults.some(({ key }) => this.pendingElectricityResults.has(String(key)))) {
            this.applyElectricityResults(completedResults);
        }
        if (query.status === "failed") {
            for (const registration of this.electricityReceivers) registration.onCount(0);
        }
        if (this.electricitySampleCooldown > 0) {
            this.electricitySampleCooldown--;
            return;
        }
        const domains = frameDomains.filter(
            (frame): frame is FluidElectricityFrameDomain & { domain: FluidElectricityDomainInternal } =>
                frame.count > 0 && this.electricityDomains.has(frame.domain as FluidElectricityDomainInternal)
        );
        const requests: FluidParticleSpatialQueryRequest[] = [];
        const receivers = [...this.electricityReceivers];
        const generation = this.electricityGeneration++;
        const push = (
            aabb: FluidParticleAabb,
            frame: FluidElectricityFrameDomain & { domain: FluidElectricityDomainInternal },
            result: ElectricityResult,
            propagation: FluidElectricityState | null
        ): void => {
            const key = `electricity:${generation}:${requests.length}`;
            this.pendingElectricityResults.set(key, result);
            requests.push({
                key,
                offset: frame.offset,
                count: frame.count,
                bounds: aabb,
                ...(propagation
                    ? {
                          sphere: {
                              origin: propagation.origin,
                              radius: electricityPropagationRadius(propagation, frame.elapsedSeconds),
                          },
                      }
                    : {}),
            });
        };
        for (const registration of this.electrifiers) {
            const aabb = registration.aabb();
            if (!aabb) continue;
            validateAabb(aabb);
            for (const frame of domains) {
                if (frame.domain.electricity || !aabbIntersects(aabb, frame.gridAabb)) continue;
                push(
                    aabb,
                    frame,
                    {
                        kind: "electrifier",
                        registration,
                        domain: frame.domain,
                        origin: aabbCenter(aabb),
                        elapsedSeconds: frame.elapsedSeconds,
                    },
                    null
                );
            }
        }
        for (const registration of receivers) {
            const aabb = registration.aabb();
            if (!aabb) continue;
            validateAabb(aabb);
            for (const frame of domains) {
                const propagation = frame.domain.electricity;
                const bounds = registration.includeParticleRadius ? expandAabb(aabb, frame.particleRadius * ELECTRICITY_PARTICLE_MASK_RADIUS_SCALE) : aabb;
                if (!propagation || !aabbIntersects(bounds, frame.gridAabb)) continue;
                push(bounds, frame, { kind: "receiver", registration, domain: frame.domain }, propagation);
            }
        }
        this.electricityPairCount = requests.length;
        if (requests.length === 0) {
            for (const registration of receivers) if (this.electricityReceivers.has(registration)) registration.onCount(0);
        } else {
            sampleFluidParticleSpatialQuery(query, stream, requests);
        }
        this.electricitySampleCooldown = FLUID_SPATIAL_QUERY_SAMPLE_INTERVAL_FRAMES - 1;
    }

    public electricityStats(): { domains: number; electrified: number; electrifiers: number; receivers: number; queryPairs: number } {
        return {
            domains: this.electricityDomains.size,
            electrified: [...this.electricityDomains].filter((domain) => domain.electricity !== null).length,
            electrifiers: this.electrifiers.size,
            receivers: this.electricityReceivers.size,
            queryPairs: this.electricityPairCount,
        };
    }

    public countParticlesInAabb(aabb: FluidParticleAabb): number {
        validateAabb(aabb);
        if (this.countQuery?.status === "failed") {
            this.completedCount = 0;
            this.completedVersion = this.requestedVersion;
        }
        this.requestedAabb = { min: [...aabb.min], max: [...aabb.max] };
        this.requestedVersion++;
        return this.completedCount;
    }

    public recordParticleCount(particleCount: number): void {
        const query = this.countQuery;
        const stream = this.stream;
        const aabb = this.requestedAabb;
        if (!query || !stream || !aabb) return;
        for (const result of readFluidParticleSpatialQuery(query)) {
            const version = Number(result.key);
            if (version >= this.completedVersion) {
                this.completedVersion = version;
                this.completedCount = result.count;
            }
            if (query.status === "failed") {
                this.completedCount = 0;
                this.completedVersion = this.requestedVersion;
            }
        }
        const count = Math.min(stream.count, Math.max(0, Math.floor(particleCount)));
        if (count === 0) {
            this.completedCount = 0;
            this.completedVersion = this.requestedVersion;
            this.sampleCooldown = 0;
            return;
        }
        if (this.sampleCooldown > 0) {
            this.sampleCooldown--;
            return;
        }
        sampleFluidParticleSpatialQuery(query, stream, [{ key: this.requestedVersion, count, bounds: aabb }]);
        this.sampleCooldown = FLUID_SPATIAL_QUERY_SAMPLE_INTERVAL_FRAMES - 1;
    }

    public override dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        super.dispose();
        if (this.countQuery) disposeFluidParticleSpatialQuery(this.countQuery);
        if (this.electricityQuery) disposeFluidParticleSpatialQuery(this.electricityQuery);
        this.countQuery = null;
        this.electricityQuery = null;
        this.stream = null;
        this.pendingElectricityResults.clear();
        this.electricityDomains.clear();
        this.electrifiers.clear();
        this.electricityReceivers.clear();
        this.completedCount = 0;
    }

    private applyElectricityResults(results: readonly { readonly key: string | number; readonly count: number }[]): void {
        const receiverCounts = new Map([...this.electricityReceivers].map((registration) => [registration, 0]));
        for (const { key, count } of results) {
            const result = this.pendingElectricityResults.get(String(key));
            if (!result) continue;
            this.pendingElectricityResults.delete(String(key));
            if (result.kind === "receiver") {
                if (this.electricityReceivers.has(result.registration) && this.electricityDomains.has(result.domain)) {
                    receiverCounts.set(result.registration, (receiverCounts.get(result.registration) ?? 0) + count);
                }
            } else if (
                count >= result.registration.particleThreshold &&
                this.electrifiers.has(result.registration) &&
                this.electricityDomains.has(result.domain) &&
                !result.domain.electricity
            ) {
                result.domain.electricity = {
                    origin: [...result.origin],
                    startedAtSeconds: result.elapsedSeconds,
                    propagationSpeed: result.registration.propagationSpeed,
                };
                result.registration.onElectrified?.(result.domain);
            }
        }
        for (const [registration, count] of receiverCounts) {
            if (this.electricityReceivers.has(registration)) registration.onCount(count);
        }
    }
}

function validateAabb(aabb: FluidParticleAabb): void {
    for (let axis = 0; axis < 3; axis++) {
        const min = aabb.min[axis]!;
        const max = aabb.max[axis]!;
        if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) {
            throw new Error(`[aquanova] fluid particle AABB axis ${axis} must have finite min <= max`);
        }
    }
}

function validatePositiveInteger(value: number, label: string): void {
    if (!Number.isInteger(value) || value <= 0) throw new Error(`[aquanova] ${label} must be a positive integer`);
}

function validatePositiveFinite(value: number, label: string): void {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`[aquanova] ${label} must be finite and positive`);
}

function aabbIntersects(a: FluidParticleAabb, b: FluidParticleAabb): boolean {
    return a.min[0] <= b.max[0] && a.max[0] >= b.min[0] && a.min[1] <= b.max[1] && a.max[1] >= b.min[1] && a.min[2] <= b.max[2] && a.max[2] >= b.min[2];
}

function expandAabb(aabb: FluidParticleAabb, padding: number): FluidParticleAabb {
    return {
        min: [aabb.min[0] - padding, aabb.min[1] - padding, aabb.min[2] - padding],
        max: [aabb.max[0] + padding, aabb.max[1] + padding, aabb.max[2] + padding],
    };
}

function aabbCenter(aabb: FluidParticleAabb): [number, number, number] {
    return [(aabb.min[0] + aabb.max[0]) * 0.5, (aabb.min[1] + aabb.max[1]) * 0.5, (aabb.min[2] + aabb.max[2]) * 0.5];
}
