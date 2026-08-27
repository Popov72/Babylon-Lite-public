import type { Mesh } from "babylon-lite";

export interface Behavior<Name extends string = string> {
    readonly name: Name;
    readonly mesh: Mesh | null;
    /** Keep this logical behavior active after its physical owner is permanently removed. */
    readonly retainOnEntityRetire?: boolean;
    init(): void | Promise<void>;
    start(): void;
    dispose(): void;
}

export interface BehaviorAssignment {
    readonly name: string;
    readonly [key: string]: unknown;
}

export interface BehaviorPreset extends Readonly<Record<string, unknown>> {
    readonly base: string;
}
export type BehaviorPresets = Readonly<Record<string, BehaviorPreset>>;
export type BehaviorEntities = Readonly<Record<string, { readonly behaviors?: readonly BehaviorAssignment[] }>>;

export type BehaviorConstructor<Context> = new (entityName: string, meshes: readonly Mesh[], assignment: BehaviorAssignment, context: Context) => Behavior;

export type BehaviorConstructorNamespace = Readonly<Record<string, unknown>>;
