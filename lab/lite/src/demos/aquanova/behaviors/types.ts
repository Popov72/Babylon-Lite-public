export interface DynamicBehaviorConfig {
    dynamic?: boolean;
    /** Rigid-body mass in kilograms. Defaults to 10. */
    mass?: number;
}

export interface LiquefiableBehaviorConfig {
    liquefiable?: true;
    fluidSim?: string[];
    linked?: string[];
    /** Splash sound category played when this mesh enters the fluid phase. */
    sound?: string;
}

export interface PlayerBehaviorConfig {
    direction?: number[];
    /** Maximum force the character controller applies while pushing dynamic bodies. */
    characterStrength?: number;
}

export interface PickEntityRaiseEventConfig {
    /** Entity or door that receives the event. Defaults to the behavior owner. */
    target?: string;
    event: string;
}

export interface BehaviorEventSubscription {
    /** Event emitted by the source owner. */
    name: string;
    /** Entity or door whose event is observed, or several equivalent sources. */
    source: string | string[];
}

export interface EntityToggleBehaviorConfig {
    /** Events that cause this behavior to act on its owning entity or door. */
    events?: BehaviorEventSubscription[];
}

export interface SoundCueConfig {
    /** MP3 file name without extension under `/aquanova/sounds/`. */
    sound: string;
    action: "play" | "stop";
    /** Events that trigger the action. The action runs at startup when omitted. */
    events?: BehaviorEventSubscription[];
    /** Seconds between the trigger and the action. Defaults to 0. */
    delay?: number;
    /** Fade-in duration for play or fade-out duration for stop. Defaults to 0. */
    fade?: number;
}

export interface SoundBehaviorConfig {
    /** Independently triggered sound actions, evaluated in declaration order. */
    cues: SoundCueConfig[];
}

export interface FluidSimHollowCylinderShape {
    type: "hollowCylinder";
    /** Base-center in the behavior mesh's local space. The axis extends from here along local +Y. */
    start: readonly [number, number, number];
    height: number;
    innerRadius: number;
    outerRadius: number;
}

export type FluidSimShape = FluidSimHollowCylinderShape;

export type FluidSimulationEventAction =
    | {
          source: string | string[];
          event: string;
          action: "enableSimulation" | "disableSimulation" | "pauseSimulation" | "unpauseSimulation" | "shutdownSimulation" | "enablePlayerCollision" | "disablePlayerCollision";
      }
    | {
          source: string | string[];
          event: string;
          action: "enableEmitter" | "disableEmitter";
          emitter: string;
      }
    | {
          source: string | string[];
          event: string;
          action: "enableSink" | "disableSink";
          sink: string;
      };

export interface SetCollisionShapeBehaviorConfig {
    type?: "mesh";
    /** Optional collision shape used only when this entity is injected into a fluid simulation. */
    fluidSimShape?: FluidSimShape;
}

export interface FluidSimulationBehaviorConfig {
    /** Fluid setting file name without `.json`, resolved under `/aquanova/fluidSim/`. */
    fluidSim: string;
    /** External entity events and the simulation or flow-object actions they trigger. */
    eventActions: FluidSimulationEventAction[];
    /** Fully visible simulated seconds after a shutdown action. Defaults to 10. */
    shutdownDuration?: number;
    /** Simulated seconds spent fading after shutdownDuration. Defaults to 2. */
    shutdownAlphaDecay?: number;
}

export interface TriggerBehaviorConfig {
    onIntersection: {
        enterEvent?: string;
        exitEvent?: string;
        playerOnly?: boolean;
    };
}

export interface PickEntityBehaviorConfig {
    /** Per-axis scale applied to the pickup's world-space bounding box. Defaults to `[1, 1, 1]`. */
    boundingBoxScale?: number[];
    raiseEvent?: PickEntityRaiseEventConfig;
    /** MP3 file name without extension under `/aquanova/sounds/`. Defaults to `pickItem`. */
    sound?: string;
    /** Multiplier for the default one-revolution-per-3-seconds Y rotation. Defaults to `1`. */
    speed?: number;
}

export interface PlayAnimationBehaviorConfig {
    /** Exact glTF animation name. Defaults to the first animation in file order. */
    animation?: string;
    /** Whether playback loops. Defaults to true. */
    loop?: boolean;
}

export interface WeaponLiquefactorBehaviorConfig {
    /** Maximum beam range when the crosshair does not hit geometry. */
    range?: number;
    /** Sound category names mapped to MP3 file names without extensions. */
    sounds?: Record<string, string[]>;
}

export interface WeaponAntiGravityGunBehaviorConfig {
    /** Maximum centre-screen hit distance that may be grabbed. Defaults to 6 metres. */
    maxGrabDistance?: number;
    /** Maximum dynamic-body mass that may be grabbed. Defaults to 100 kilograms. */
    maxMass?: number;
}

/** All parameters that a manifest behavior definition or entity override may provide. */
export interface BehaviorConfig {
    readonly [key: string]: unknown;
    dynamic?: boolean;
    mass?: number;
    liquefiable?: boolean;
    fluidSim?: string | string[];
    linked?: string[];
    direction?: number[];
    characterStrength?: number;
    range?: number;
    sound?: string;
    sounds?: Record<string, string[]>;
    cues?: SoundCueConfig[];
    speed?: number;
    boundingBoxScale?: number[];
    events?: BehaviorEventSubscription[];
    raiseEvent?: PickEntityRaiseEventConfig;
    reflectionProbe?: "exclude";
    animation?: string;
    loop?: boolean;
    maxGrabDistance?: number;
    maxMass?: number;
    type?: "mesh";
    fluidSimShape?: FluidSimShape;
    shutdownDuration?: number;
    shutdownAlphaDecay?: number;
    eventActions?: FluidSimulationEventAction[];
    onIntersection?: TriggerBehaviorConfig["onIntersection"];
}

export interface BehaviorReference extends BehaviorConfig {
    name: string;
}

export type BehaviorAssignment = BehaviorReference;
export interface BehaviorPreset extends BehaviorConfig {
    base: string;
}
export type BehaviorPresets = Record<string, BehaviorPreset>;
export type Entities = Record<string, { behaviors?: BehaviorReference[] }>;
export type { Behavior } from "../behavior-system/behavior.js";
