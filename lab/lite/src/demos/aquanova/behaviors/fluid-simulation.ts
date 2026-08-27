import type { IWorldMatrixProvider, Mesh } from "babylon-lite";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import type { AquanovaGameContext } from "./game-context.js";
import type { Behavior, FluidSimulationBehaviorConfig, FluidSimulationEventAction } from "./types.js";
import type { FluidSimulationFlowObjectKind, FluidSimulationRegistration, FluidSimulationRuntime, FluidSimulationState } from "./fluid-simulation-runtime.js";

type FluidSimulationContext = Pick<AquanovaGameContext, "events"> & { fluidSimulations: FluidSimulationRuntime };

const DEFAULT_SHUTDOWN_DURATION = 10;
const DEFAULT_SHUTDOWN_ALPHA_DECAY = 2;
const SIMULATION_ACTIONS = new Set(["enableSimulation", "disableSimulation", "pauseSimulation", "unpauseSimulation", "shutdownSimulation"]);
const PLAYER_COLLISION_ACTIONS = new Set(["enablePlayerCollision", "disablePlayerCollision"]);
const EMITTER_ACTIONS = new Set(["enableEmitter", "disableEmitter"]);
const SINK_ACTIONS = new Set(["enableSink", "disableSink"]);

interface SettingFlowObject {
    name?: unknown;
    enabled?: unknown;
}

interface FluidSimulationSetting {
    emitters?: unknown;
    sinks?: unknown;
    readonly [key: string]: unknown;
}

export class FluidSimulationBehavior implements Behavior<"fluidSimulation"> {
    public readonly name = "fluidSimulation";
    public readonly mesh: Mesh;
    public readonly config: FluidSimulationBehaviorConfig;
    private readonly entityName: string;
    private readonly context: FluidSimulationContext;
    private registration: FluidSimulationRegistration | null = null;
    private stopEntityEvent: (() => void) | null = null;
    private state: FluidSimulationState = "registered";
    private hasStarted = false;

    public constructor(entityName: string, meshes: readonly Mesh[], config: FluidSimulationBehaviorConfig, context: FluidSimulationContext) {
        const mesh = meshes[0];
        if (!mesh) throw new Error("[aquanova] fluidSimulation requires at least one mesh");
        assertBehaviorConfigKeys(config, "fluidSimulation", ["fluidSim", "electrifiable", "eventActions", "shutdownDuration", "shutdownAlphaDecay"]);
        if (typeof config.fluidSim !== "string" || !config.fluidSim || /\.json$/i.test(config.fluidSim)) {
            throw new Error("[aquanova] fluidSimulation.fluidSim must be a non-empty file name without the .json extension");
        }
        if (config.electrifiable !== undefined && typeof config.electrifiable !== "boolean") {
            throw new Error("[aquanova] fluidSimulation.electrifiable must be true or false");
        }
        validateEventActions(config.eventActions);
        validateDuration("shutdownDuration", config.shutdownDuration);
        validateDuration("shutdownAlphaDecay", config.shutdownAlphaDecay);
        this.entityName = entityName;
        this.mesh = mesh;
        this.config = config;
        this.context = context;
    }

    public async init(): Promise<void> {
        const response = await fetch(`/aquanova/fluidSim/${this.config.fluidSim}.json`);
        if (!response.ok) {
            throw new Error(`[aquanova] fluidSimulation "${this.config.fluidSim}" could not be loaded: HTTP ${response.status}`);
        }
        const setting: unknown = await response.json();
        if (!setting || typeof setting !== "object" || Array.isArray(setting)) {
            throw new Error(`[aquanova] fluidSimulation "${this.config.fluidSim}" must contain a JSON object`);
        }
        validateFlowObjectReferences(this.config.eventActions, this.config.fluidSim, setting as FluidSimulationSetting);
        this.registration = {
            entityName: this.entityName,
            mesh: this.mesh,
            anchor: ownerAnchor(this.mesh, this.entityName),
            settingName: this.config.fluidSim,
            setting,
            electrifiable: this.config.electrifiable ?? false,
            shutdownDuration: this.config.shutdownDuration ?? DEFAULT_SHUTDOWN_DURATION,
            shutdownAlphaDecay: this.config.shutdownAlphaDecay ?? DEFAULT_SHUTDOWN_ALPHA_DECAY,
            onEmissionComplete: (): void => {
                this.context.events.emit("entityEvent", { name: this.entityName, event: "emissionComplete" });
            },
            onStarted: (): void => {
                this.context.events.emit("entityEvent", { name: this.entityName, event: "startSimulation" });
            },
            onShutdownComplete: (): void => {
                this.context.events.emit("entityEvent", { name: this.entityName, event: "endSimulation" });
            },
        };
    }

    public start(): void {
        if (!this.registration) throw new Error(`[aquanova] fluidSimulation "${this.entityName}" started before initialization`);
        this.context.fluidSimulations.register(this.registration);
        this.stopEntityEvent = this.context.events.on("entityEvent", ({ name, event }) => {
            for (const action of this.config.eventActions) {
                if (action.event === event && sourceMatches(action.source, name)) {
                    this.applyAction(action);
                }
            }
        });
    }

    public dispose(): void {
        this.stopEntityEvent?.();
        this.stopEntityEvent = null;
        if (this.registration) this.context.fluidSimulations.unregister(this.registration);
        this.registration = null;
        this.state = "disposed";
    }

    private transition(state: FluidSimulationState): void {
        if (!this.registration || this.state === "disposed" || this.state === "shutdown") return;
        this.state = state;
        this.context.fluidSimulations.update(this.registration, state);
    }

    private applyAction(action: FluidSimulationEventAction): void {
        if (!this.registration || this.state === "disposed") return;
        if (action.action === "enablePlayerCollision" || action.action === "disablePlayerCollision") {
            this.context.fluidSimulations.updatePlayerCollision(this.registration, action.action === "enablePlayerCollision");
            return;
        }
        if (this.state === "shutdown") return;
        if (action.action === "enableSimulation") {
            this.hasStarted = true;
            this.transition("running");
        } else if (action.action === "disableSimulation") {
            this.transition("paused");
        } else if (action.action === "pauseSimulation") {
            if (this.hasStarted) this.transition("paused");
        } else if (action.action === "unpauseSimulation") {
            if (this.hasStarted) this.transition("running");
        } else if (action.action === "shutdownSimulation") {
            this.transition("shutdown");
        } else if (action.action === "enableEmitter" || action.action === "disableEmitter") {
            this.setFlowObjectEnabled("emitter", action.emitter, action.action === "enableEmitter");
        } else if (action.action === "enableSink" || action.action === "disableSink") {
            this.setFlowObjectEnabled("sink", action.sink, action.action === "enableSink");
        }
    }

    private setFlowObjectEnabled(kind: FluidSimulationFlowObjectKind, name: string, enabled: boolean): void {
        if (!this.registration) return;
        const setting = this.registration.setting as FluidSimulationSetting;
        const objects = flowObjects(setting, kind);
        const object = objects.find((candidate) => candidate.name === name);
        if (!object) {
            throw new Error(`[aquanova] fluidSimulation.${kind} "${name}" is no longer present in "${this.config.fluidSim}"`);
        }
        object.enabled = enabled;
        this.context.fluidSimulations.updateFlowObject(this.registration, kind, name, enabled);
    }
}

function ownerAnchor(mesh: Mesh, entityName: string): IWorldMatrixProvider {
    let current: IWorldMatrixProvider | null = mesh;
    while (current) {
        if ("name" in current && current.name === entityName) return current;
        const parent: unknown = "parent" in current ? current.parent : null;
        current = parent && typeof parent === "object" && "worldMatrix" in parent ? (parent as IWorldMatrixProvider) : null;
    }
    return mesh;
}

function validateDuration(name: string, value: number | undefined): void {
    if (value !== undefined && (!Number.isFinite(value) || value < 0)) {
        throw new Error(`[aquanova] fluidSimulation.${name} must be finite and non-negative`);
    }
}

function validateEventActions(actions: FluidSimulationEventAction[] | undefined): void {
    if (!Array.isArray(actions) || actions.length === 0) {
        throw new Error("[aquanova] fluidSimulation.eventActions must contain at least one action");
    }
    for (const action of actions) {
        if (!action || typeof action !== "object" || Array.isArray(action)) {
            throw new Error("[aquanova] fluidSimulation.eventActions[] must be an object");
        }
        assertBehaviorConfigKeys(action, "fluidSimulation.eventActions[]", ["source", "event", "action", "emitter", "sink"]);
        if (typeof action.event !== "string" || !action.event.trim()) {
            throw new Error("[aquanova] fluidSimulation.eventActions[].event must be a non-empty event name");
        }
        validateSource(action.source);
        if (
            typeof action.action !== "string" ||
            (!SIMULATION_ACTIONS.has(action.action) && !PLAYER_COLLISION_ACTIONS.has(action.action) && !EMITTER_ACTIONS.has(action.action) && !SINK_ACTIONS.has(action.action))
        ) {
            throw new Error(`[aquanova] fluidSimulation.eventActions[].action "${String(action.action)}" is not supported`);
        }
        const emitter = "emitter" in action ? action.emitter : undefined;
        const sink = "sink" in action ? action.sink : undefined;
        if (EMITTER_ACTIONS.has(action.action)) {
            if (typeof emitter !== "string" || !emitter.trim()) {
                throw new Error("[aquanova] fluidSimulation.eventActions[].emitter must be a non-empty emitter name");
            }
            if (sink !== undefined) throw new Error("[aquanova] fluidSimulation.eventActions[].sink is only valid for sink actions");
        } else if (SINK_ACTIONS.has(action.action)) {
            if (typeof sink !== "string" || !sink.trim()) {
                throw new Error("[aquanova] fluidSimulation.eventActions[].sink must be a non-empty sink name");
            }
            if (emitter !== undefined) throw new Error("[aquanova] fluidSimulation.eventActions[].emitter is only valid for emitter actions");
        } else if (emitter !== undefined || sink !== undefined) {
            throw new Error("[aquanova] fluidSimulation.eventActions[] simulation and player-collision actions cannot target an emitter or sink");
        }
    }
}

function validateSource(source: string | string[]): void {
    if (typeof source === "string") {
        if (!source.trim()) throw new Error("[aquanova] fluidSimulation.eventActions[].source must be a non-empty entity or door name");
        return;
    }
    if (!Array.isArray(source) || source.length === 0 || source.some((name) => typeof name !== "string" || !name.trim())) {
        throw new Error("[aquanova] fluidSimulation.eventActions[].source must contain at least one non-empty entity or door name");
    }
}

function validateFlowObjectReferences(actions: readonly FluidSimulationEventAction[], settingName: string, setting: FluidSimulationSetting): void {
    const referenced = {
        emitter: new Set(actions.flatMap((action) => ("emitter" in action ? [action.emitter] : []))),
        sink: new Set(actions.flatMap((action) => ("sink" in action ? [action.sink] : []))),
    };
    for (const kind of ["emitter", "sink"] as const) {
        const counts = new Map<string, number>();
        for (const object of flowObjects(setting, kind)) {
            if (typeof object.name === "string" && object.name.trim()) {
                counts.set(object.name, (counts.get(object.name) ?? 0) + 1);
            }
        }
        for (const name of referenced[kind]) {
            const count = counts.get(name) ?? 0;
            if (count === 0) {
                throw new Error(`[aquanova] fluidSimulation "${settingName}" has no ${kind} named "${name}"`);
            }
            if (count > 1) {
                throw new Error(`[aquanova] fluidSimulation ${kind} name "${name}" is ambiguous`);
            }
        }
    }
}

function flowObjects(setting: FluidSimulationSetting, kind: FluidSimulationFlowObjectKind): SettingFlowObject[] {
    const value = kind === "emitter" ? setting.emitters : setting.sinks;
    return Array.isArray(value) ? value.filter((object): object is SettingFlowObject => !!object && typeof object === "object" && !Array.isArray(object)) : [];
}

function sourceMatches(sources: string | string[], source: string): boolean {
    return typeof sources === "string" ? sources === source : sources.includes(source);
}
