import type { Mesh } from "babylon-lite";

export interface Behavior<Name extends string = string> {
    readonly name: Name;
    readonly mesh: Mesh | null;
    init(): void | Promise<void>;
    start(): void;
    dispose(): void;
}

export interface BehaviorAssignment {
    readonly name: string;
    readonly [key: string]: unknown;
}

export type BehaviorDefinition = Readonly<Record<string, unknown>>;
export type BehaviorLibrary = Readonly<Record<string, BehaviorDefinition>>;
export type BehaviorEntities = Readonly<Record<string, { readonly behaviors?: readonly BehaviorAssignment[] }>>;

export type BehaviorConstructor<Context> = new (entityName: string, meshes: readonly Mesh[], assignment: BehaviorAssignment, context: Context) => Behavior;

export type BehaviorConstructorNamespace = Readonly<Record<string, unknown>>;
