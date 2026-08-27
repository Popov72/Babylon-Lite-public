export type DynamicRotationAxis = "x" | "y" | "z";

export interface DynamicBehaviorConfig {
    dynamic?: boolean;
    /** Rigid-body mass in kilograms. Defaults to 10. */
    mass?: number;
    /** Principal body axes around which angular motion is locked. */
    lockedRotationAxes?: DynamicRotationAxis[];
}

export interface LiquefiableBehaviorConfig {
    liquefiable?: true;
    /** Whether this liquefaction domain may retain electrical charge. Defaults to false. */
    electrifiable?: boolean;
    fluidSim?: string[];
    linked?: string[];
    /** Splash sound category played when this mesh enters the fluid phase. */
    sound?: string;
}

export interface PlayerBehaviorConfig {
    direction?: number[];
    /** Maximum force the character controller applies while pushing dynamic bodies. */
    characterStrength?: number;
    /** Maximum centre-screen hit distance that the anti-gravity gun may grab. Defaults to 8 metres. */
    maxGrabDistance?: number;
    /** Maximum player-to-object distance before an anti-gravity-held body is released. Defaults to 8 metres. */
    maxHeldObjectDistance?: number;
    /** Visible particles above the player's head required to enter the submerged state. */
    submergedParticleCount?: number;
    /** Electrified particle centres inside the live capsule required to enter electrical contact. */
    electrifiedParticleCount?: number;
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

export interface SoundEventTriggerConfig {
    /** Entity or door whose event is observed, or several equivalent sources. Requires `event`. */
    source?: string | string[];
    /** Event emitted by `source`. Requires `source`. */
    event?: string;
}

export interface PlaySoundBehaviorConfig extends SoundEventTriggerConfig {
    /** Unique playback channel ID referenced by stopSound. */
    id: string;
    /** MP3 file name without extension under `/aquanova/sounds/`. */
    sound: string;
    /** Fade-in duration in seconds. Defaults to 0. */
    fadeInDelay?: number;
    /** Per-play volume from 0 to 1. Defaults to 1. */
    volume?: number;
    /** Whether playback loops. Defaults to false. */
    loop?: boolean;
}

export interface StopSoundBehaviorConfig extends SoundEventTriggerConfig {
    /** Playback channel defined by a playSound behavior. */
    soundId: string;
    /** Fade-out duration in seconds. Defaults to 0. */
    fadeOutDelay?: number;
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
    /** Whether this authored simulation may retain electrical charge. Defaults to false. */
    electrifiable?: boolean;
    /** External entity events and the simulation or flow-object actions they trigger. */
    eventActions: FluidSimulationEventAction[];
    /** Fully visible simulated seconds after a shutdown action. Defaults to 10. */
    shutdownDuration?: number;
    /** Simulated seconds spent fading after shutdownDuration. Defaults to 2. */
    shutdownAlphaDecay?: number;
}

export interface FluidElectrifierBehaviorConfig {
    /** Visible particle centres required inside the owner AABB. Defaults to 24. */
    particleThreshold?: number;
    /** Metres per second travelled by the charge propagation front. Defaults to 8. */
    propagationSpeed?: number;
}

export interface ElectricalDetonatorBehaviorConfig {
    /** Propagated electrified particle centres required inside the owner AABB. Defaults to 4. */
    particleThreshold?: number;
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
    /** Local axis around which the pickup rotates. Defaults to `y`. */
    rotationAxis?: "x" | "y" | "z";
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
    /** @deprecated Configure `player.maxGrabDistance` instead. Retained as a manifest compatibility override. */
    maxGrabDistance?: number;
    /** Maximum dynamic-body mass that may be grabbed. Defaults to 100 kilograms. */
    maxMass?: number;
}

export interface WeaponPistolBehaviorConfig {
    /** Maximum bullet travel distance in metres. Defaults to 100. */
    range?: number;
    /** Visible projectile speed in metres per second. Defaults to 80. */
    bulletSpeed?: number;
    /** Point impulse applied to dynamic entities in kg m/s. Defaults to 10. */
    impactImpulse?: number;
}

/** All parameters that a manifest behavior definition or entity override may provide. */
export interface BehaviorConfig {
    readonly [key: string]: unknown;
    dynamic?: boolean;
    mass?: number;
    liquefiable?: boolean;
    fluidSim?: string | string[];
    electrifiable?: boolean;
    linked?: string[];
    direction?: number[];
    characterStrength?: number;
    maxHeldObjectDistance?: number;
    range?: number;
    sound?: string;
    sounds?: Record<string, string[]>;
    cues?: SoundCueConfig[];
    source?: string | string[];
    event?: string;
    fadeInDelay?: number;
    fadeOutDelay?: number;
    id?: string;
    soundId?: string;
    volume?: number;
    speed?: number;
    boundingBoxScale?: number[];
    events?: BehaviorEventSubscription[];
    raiseEvent?: PickEntityRaiseEventConfig;
    reflectionProbe?: "exclude";
    animation?: string;
    loop?: boolean;
    maxGrabDistance?: number;
    propagationSpeed?: number;
    maxMass?: number;
    bulletSpeed?: number;
    impactImpulse?: number;
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
