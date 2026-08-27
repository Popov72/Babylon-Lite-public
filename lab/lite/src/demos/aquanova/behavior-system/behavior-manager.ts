import type { Mesh } from "babylon-lite";
import type { Behavior, BehaviorAssignment, BehaviorConstructor, BehaviorConstructorNamespace, BehaviorEntities, BehaviorPresets } from "./behavior.js";

function behaviorClassName(name: string): string {
    return `${name[0]?.toUpperCase() ?? ""}${name.slice(1)}Behavior`;
}

function constructorFor(constructors: BehaviorConstructorNamespace, name: string): unknown {
    return constructors[behaviorClassName(name)];
}

export interface BehaviorManagerOptions {
    readonly presets?: BehaviorPresets;
    readonly entities?: BehaviorEntities;
    readonly meshlessOwners?: BehaviorEntities;
    readonly meshesByEntityName: ReadonlyMap<string, readonly Mesh[]>;
    readonly constructors: BehaviorConstructorNamespace;
    readonly shouldInstantiate?: (assignment: BehaviorAssignment) => boolean;
}

export interface BehaviorEntityMatch {
    readonly entityName: string;
    readonly assignment: BehaviorAssignment;
    readonly meshes: readonly Mesh[];
}

export class BehaviorManager<Context> {
    public readonly instances: Behavior[] = [];
    private readonly presets: BehaviorPresets | undefined;
    private readonly entities: BehaviorEntities | undefined;
    private readonly meshlessOwners: BehaviorEntities | undefined;
    private readonly meshesByEntityName: ReadonlyMap<string, readonly Mesh[]>;
    private readonly constructors: BehaviorConstructorNamespace;
    private readonly shouldInstantiate: (assignment: BehaviorAssignment) => boolean;
    private readonly assignmentCache = new Map<string, BehaviorAssignment[]>();
    private readonly entityNameByInstance = new Map<Behavior, string>();
    private started = false;

    public constructor(options: BehaviorManagerOptions) {
        this.presets = options.presets;
        this.entities = options.entities;
        this.meshlessOwners = options.meshlessOwners;
        this.meshesByEntityName = options.meshesByEntityName;
        this.constructors = options.constructors;
        this.shouldInstantiate = options.shouldInstantiate ?? (() => true);
        for (const name of Object.keys(this.presets ?? {})) {
            if (constructorFor(this.constructors, name)) {
                throw new Error(`Behavior preset "${name}" conflicts with a base behavior of the same name`);
            }
        }
    }

    public assignmentsOf(entityName: string): readonly BehaviorAssignment[] {
        let assignments = this.assignmentCache.get(entityName);
        if (assignments) {
            return assignments;
        }
        const references = this.entities?.[entityName]?.behaviors ?? this.meshlessOwners?.[entityName]?.behaviors ?? [];
        assignments = references.map((reference) => {
            const preset = this.presets?.[reference.name];
            const behaviorName = preset?.base ?? reference.name;
            const assignment = { ...(preset ?? {}), ...reference, name: behaviorName };
            delete (assignment as { base?: unknown }).base;
            return assignment;
        });
        this.assignmentCache.set(entityName, assignments);
        return assignments;
    }

    public entityNames(): readonly string[] {
        return [...new Set([...Object.keys(this.entities ?? {}), ...Object.keys(this.meshlessOwners ?? {})])];
    }

    public findEntityWithBehavior(behaviorName: string): BehaviorEntityMatch | undefined {
        for (const entityName of this.entityNames()) {
            const assignment = this.assignmentsOf(entityName).find((candidate) => candidate.name === behaviorName);
            if (assignment) {
                return { entityName, assignment, meshes: this.meshesByEntityName.get(entityName) ?? [] };
            }
        }
        return undefined;
    }

    public async start(context: Context): Promise<void> {
        if (this.started) {
            throw new Error("Behaviors are already started");
        }
        this.started = true;
        try {
            for (const entityName of this.entityNames()) {
                const meshes = this.meshesByEntityName.get(entityName) ?? [];
                if (meshes.length === 0 && !this.meshlessOwners?.[entityName]) {
                    const behaviorNames = this.entities?.[entityName]?.behaviors?.map(({ name }) => name) ?? [];
                    if (behaviorNames.length > 0) {
                        // eslint-disable-next-line no-console
                        console.warn(
                            `[behavior] skipping owner "${entityName}": no runtime meshes were found for behaviors ${behaviorNames.map((name) => `"${name}"`).join(", ")}`
                        );
                    }
                    continue;
                }
                for (const assignment of this.assignmentsOf(entityName)) {
                    if (!this.shouldInstantiate(assignment)) {
                        continue;
                    }
                    const className = behaviorClassName(assignment.name);
                    const Constructor = constructorFor(this.constructors, assignment.name);
                    if (typeof Constructor !== "function") {
                        throw new Error(`Behavior "${assignment.name}" requires exported class "${className}"`);
                    }

                    const behavior = new (Constructor as BehaviorConstructor<Context>)(entityName, meshes, assignment, context);
                    this.instances.push(behavior);
                    this.entityNameByInstance.set(behavior, entityName);
                }
            }
            await Promise.all(this.instances.map((behavior) => behavior.init()));
            for (const behavior of this.instances) {
                behavior.start();
            }
        } catch (error) {
            this.disposeInstances();
            this.started = false;
            throw error;
        }
    }

    public describeInstances(entityNameOf: (mesh: Mesh) => string): Array<{ name: string; mesh: string }> {
        return this.instances.map((behavior) => ({
            name: behavior.name,
            mesh: behavior.mesh ? entityNameOf(behavior.mesh) : (this.entityNameByInstance.get(behavior) ?? ""),
        }));
    }

    public retireEntityInstances(entityName: string, preserve: (behavior: Behavior) => boolean = () => false): void {
        for (let index = this.instances.length - 1; index >= 0; index--) {
            const behavior = this.instances[index]!;
            if (this.entityNameByInstance.get(behavior) !== entityName || preserve(behavior)) {
                continue;
            }
            behavior.dispose();
            this.instances.splice(index, 1);
            this.entityNameByInstance.delete(behavior);
        }
    }

    public dispose(): void {
        this.disposeInstances();
        this.started = false;
    }

    private disposeInstances(): void {
        for (let index = this.instances.length - 1; index >= 0; index--) {
            this.instances[index]!.dispose();
        }
        this.instances.length = 0;
        this.entityNameByInstance.clear();
    }
}
