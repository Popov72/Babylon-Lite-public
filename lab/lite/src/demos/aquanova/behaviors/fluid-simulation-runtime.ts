import type { IWorldMatrixProvider, Mesh } from "babylon-lite";

export type FluidSimulationState = "registered" | "running" | "paused" | "shutdown" | "disposed";
export type FluidSimulationFlowObjectKind = "emitter" | "sink";

export interface FluidSimulationRegistration {
    readonly entityName: string;
    readonly mesh: Mesh;
    /** Owning glTF node whose world-space translation anchors the simulation grid. */
    readonly anchor: IWorldMatrixProvider;
    readonly settingName: string;
    readonly setting: unknown;
    readonly electrifiable: boolean;
    readonly shutdownDuration: number;
    readonly shutdownAlphaDecay: number;
    readonly onEmissionComplete?: () => void;
    readonly onStarted?: () => void;
    readonly onShutdownComplete?: () => void;
}

export interface FluidSimulationBackend {
    register(registration: FluidSimulationRegistration): void;
    update(registration: FluidSimulationRegistration, state: FluidSimulationState): void;
    updatePlayerCollision(registration: FluidSimulationRegistration, enabled: boolean): void;
    updateFlowObject(registration: FluidSimulationRegistration, kind: FluidSimulationFlowObjectKind, name: string, enabled: boolean): void;
    unregister(registration: FluidSimulationRegistration): void;
}

export interface FluidSimulationShutdownLifecycle {
    opacity: number;
    stopped: boolean;
}

export function fluidSimulationShutdownLifecycle(elapsed: number, duration: number, alphaDecay: number): FluidSimulationShutdownLifecycle {
    const safeElapsed = Math.max(0, elapsed);
    const safeDuration = Math.max(0, duration);
    const safeDecay = Math.max(0, alphaDecay);
    if (safeElapsed < safeDuration) return { opacity: 1, stopped: false };
    if (safeDecay === 0 || safeElapsed >= safeDuration + safeDecay) return { opacity: 0, stopped: true };
    return { opacity: 1 - (safeElapsed - safeDuration) / safeDecay, stopped: false };
}

export function fluidSimulationShutdownStepDelta(elapsed: number, requestedDelta: number, duration: number, alphaDecay: number): number {
    const stopTime = Math.max(0, duration) + Math.max(0, alphaDecay);
    return Math.min(Math.max(0, requestedDelta), Math.max(0, stopTime - Math.max(0, elapsed)));
}

export function fluidEmissionCompletionTarget(capacity: number, resetActiveCount: number, hasEnabledInflow: boolean): number {
    const safeCapacity = Math.max(0, Math.floor(capacity));
    return hasEnabledInflow ? safeCapacity : Math.min(safeCapacity, Math.max(0, Math.floor(resetActiveCount)));
}

interface RuntimeEntry {
    readonly registration: FluidSimulationRegistration;
    state: FluidSimulationState;
    playerCollisionEnabled: boolean;
}

export class FluidSimulationRuntime {
    private readonly entries = new Map<FluidSimulationRegistration, RuntimeEntry>();
    private backend: FluidSimulationBackend | null = null;

    public register(registration: FluidSimulationRegistration): void {
        if (this.entries.has(registration)) {
            throw new Error(`[aquanova] fluidSimulation "${registration.entityName}" is already registered`);
        }
        this.entries.set(registration, { registration, state: "registered", playerCollisionEnabled: true });
        this.backend?.register(registration);
    }

    public update(registration: FluidSimulationRegistration, state: FluidSimulationState): void {
        const entry = this.entries.get(registration);
        if (!entry || entry.state === "disposed") return;
        if (entry.state === "shutdown" && state !== "disposed") return;
        entry.state = state;
        this.backend?.update(registration, state);
    }

    public updateFlowObject(registration: FluidSimulationRegistration, kind: FluidSimulationFlowObjectKind, name: string, enabled: boolean): void {
        const entry = this.entries.get(registration);
        if (!entry || entry.state === "disposed" || entry.state === "shutdown") return;
        this.backend?.updateFlowObject(registration, kind, name, enabled);
    }

    public updatePlayerCollision(registration: FluidSimulationRegistration, enabled: boolean): void {
        const entry = this.entries.get(registration);
        if (!entry || entry.state === "disposed") return;
        entry.playerCollisionEnabled = enabled;
        this.backend?.updatePlayerCollision(registration, enabled);
    }

    public unregister(registration: FluidSimulationRegistration): void {
        if (!this.entries.delete(registration)) return;
        this.backend?.unregister(registration);
    }

    public installBackend(backend: FluidSimulationBackend): void {
        if (this.backend) {
            throw new Error("[aquanova] fluidSimulation backend is already installed");
        }
        this.backend = backend;
        for (const entry of this.entries.values()) {
            backend.register(entry.registration);
            if (!entry.playerCollisionEnabled) backend.updatePlayerCollision(entry.registration, false);
            if (entry.state !== "registered") backend.update(entry.registration, entry.state);
        }
    }

    public dispose(): void {
        if (this.backend) {
            for (const entry of this.entries.values()) this.backend.unregister(entry.registration);
        }
        this.entries.clear();
        this.backend = null;
    }
}
