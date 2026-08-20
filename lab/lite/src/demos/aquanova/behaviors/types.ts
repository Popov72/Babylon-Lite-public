export interface DynamicBehaviorConfig {
    dynamic?: boolean;
    /** Rigid-body mass in kilograms. Defaults to 10. */
    mass?: number;
}

export interface LiquefiableBehaviorConfig {
    liquefiable: true;
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
    /** @deprecated Use `target`; retained while existing manifests are migrated. */
    entity?: string;
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
    /** @deprecated Use `events`; retained while existing manifests are migrated. */
    onEvent?: string;
    /** @deprecated Legacy action target paired with `onEvent`. */
    entity?: string;
}

export interface SetCollisionShapeBehaviorConfig {
    type?: "mesh";
    /** Compatibility with manifests authored before the parameter was named `type`. */
    shape?: "mesh";
}

export interface TriggerBehaviorConfig {
    onIntersection: {
        enterEvent?: string;
        exitEvent?: string;
        playerOnly?: boolean;
        /** @deprecated Use `enterEvent`; retained while existing manifests are migrated. */
        raiseEvent?: string;
        /** @deprecated New trigger events are always raised by the behavior owner. */
        entity?: string;
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
    direction?: number[];
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
    fluidSim?: string[];
    linked?: string[];
    direction?: number[];
    characterStrength?: number;
    range?: number;
    sound?: string;
    sounds?: Record<string, string[]>;
    speed?: number;
    boundingBoxScale?: number[];
    entity?: string;
    onEvent?: string;
    events?: BehaviorEventSubscription[];
    raiseEvent?: PickEntityRaiseEventConfig;
    reflectionProbe?: "exclude";
    animation?: string;
    loop?: boolean;
    maxGrabDistance?: number;
    maxMass?: number;
    type?: "mesh";
    shape?: "mesh";
    onIntersection?: TriggerBehaviorConfig["onIntersection"];
}

export interface BehaviorReference extends BehaviorConfig {
    name: string;
}

export type BehaviorAssignment = BehaviorReference;
export type BehaviorLibrary = Record<string, BehaviorConfig>;
export type Entities = Record<string, { behaviors?: BehaviorReference[] }>;
export type { Behavior } from "../behavior-system/behavior.js";
