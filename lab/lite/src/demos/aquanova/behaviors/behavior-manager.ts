import type { Mesh, PhysicsWorld, SceneContext } from "babylon-lite";
import { DynamicBehavior } from "./dynamic.js";
import { EventManager } from "./event-manager.js";
import { LiquefiableBehavior } from "./liquefiable.js";
import { PlayerBehavior } from "./player.js";
import { isLiquefiableBehaviorConfig } from "./types.js";
import type { Behavior, BehaviorAssignment, BehaviorContext, BehaviorLibrary, Entities, LiquefiableBehaviorConfig } from "./types.js";
import { WeaponBehavior } from "./weapon.js";

export interface BehaviorManagerOptions {
    readonly library: BehaviorLibrary | undefined;
    readonly entities: Entities | undefined;
    readonly meshesByEntityName: ReadonlyMap<string, readonly Mesh[]>;
    readonly entityNameOf: (mesh: Mesh) => string;
}

export interface BehaviorMeshClassification {
    readonly isDisabled: (mesh: Mesh) => boolean;
    readonly instanceIdOf: (mesh: Mesh) => string | undefined;
}

export interface BehaviorEntityMatch {
    readonly entityName: string;
    readonly assignment: BehaviorAssignment;
    readonly meshes: readonly Mesh[];
}

export interface MeshBehaviorAvailability {
    readonly liquefiable: boolean;
    readonly dissolvable: boolean;
}

/** Owns manifest behavior resolution, mesh classification, instances, and gameplay queries. */
export class BehaviorManager {
    public readonly events = new EventManager();
    public readonly instances: Behavior[] = [];
    public readonly dynamicMeshes = new Set<Mesh>();
    public readonly movableMeshes = new Set<Mesh>();
    public readonly liquefiableMeshes = new Set<Mesh>();
    public readonly dissolvableMeshes = new Set<Mesh>();
    public readonly dissolvableInstanceIds = new Set<string>();
    private readonly library: BehaviorLibrary | undefined;
    private readonly entities: Entities | undefined;
    private readonly meshesByEntityName: ReadonlyMap<string, readonly Mesh[]>;
    private readonly entityNameOf: (mesh: Mesh) => string;
    private readonly assignmentCache = new Map<string, BehaviorAssignment[]>();
    private readonly liquefiableConfigByMesh = new Map<Mesh, LiquefiableBehaviorConfig>();
    private started = false;

    public constructor(options: BehaviorManagerOptions) {
        this.library = options.library;
        this.entities = options.entities;
        this.meshesByEntityName = options.meshesByEntityName;
        this.entityNameOf = options.entityNameOf;
    }

    public findEntityWithBehavior(behaviorName: string): BehaviorEntityMatch | undefined {
        for (const entityName of Object.keys(this.entities ?? {})) {
            const assignment = this.assignmentsOf(entityName).find((candidate) => candidate.name === behaviorName);
            if (assignment) {
                return {
                    entityName,
                    assignment,
                    meshes: this.meshesByEntityName.get(entityName) ?? [],
                };
            }
        }
        return undefined;
    }

    public classifyMeshes(meshes: readonly Mesh[], classification: BehaviorMeshClassification): void {
        this.clearClassification();
        const dissolvableEntityNames = this.resolveDissolvableEntityNames();
        for (const mesh of meshes) {
            if (classification.isDisabled(mesh)) continue;
            const entityName = this.entityNameOf(mesh);
            const assignments = this.assignmentsOf(entityName);
            const liquefiableConfig = this.liquefiableConfigOf(entityName);
            if (liquefiableConfig) this.liquefiableConfigByMesh.set(mesh, liquefiableConfig);
            const explicitlyDynamic = assignments.some((assignment) => assignment.name === "dynamic" && assignment.dynamic === true);
            const dissolvable = dissolvableEntityNames.has(entityName);
            if (dissolvable) {
                this.dissolvableMeshes.add(mesh);
                const instanceId = classification.instanceIdOf(mesh);
                if (instanceId) this.dissolvableInstanceIds.add(instanceId);
            }
            if (dissolvable || explicitlyDynamic) this.dynamicMeshes.add(mesh);
            if (explicitlyDynamic) this.movableMeshes.add(mesh);
            if (liquefiableConfig) this.liquefiableMeshes.add(mesh);
        }
    }

    public start(context: Omit<BehaviorContext, "events">): void {
        if (this.started) throw new Error("[aquanova] behaviors are already started");
        this.started = true;
        const behaviorContext: BehaviorContext = { ...context, events: this.events };
        for (const entityName of Object.keys(this.entities ?? {})) {
            const meshes = this.meshesByEntityName.get(entityName) ?? [];
            for (const assignment of this.assignmentsOf(entityName)) {
                const targets = assignment.name === "player" || assignment.name === "weapon" ? meshes.slice(0, 1) : meshes;
                for (const mesh of targets) this.instances.push(createBehavior(assignment, mesh, behaviorContext));
            }
        }
        for (const behavior of this.instances) behavior.start();
    }

    public bindSystemEvents(scene: SceneContext, world: PhysicsWorld): void {
        this.events.bindSystemEvents(scene, world);
    }

    public get player(): PlayerBehavior | null {
        return this.instances.find((behavior): behavior is PlayerBehavior => behavior instanceof PlayerBehavior) ?? null;
    }

    public isLiquefiable(mesh: Mesh): boolean {
        return this.liquefiableMeshes.has(mesh);
    }

    public isDissolvable(mesh: Mesh): boolean {
        return this.dissolvableMeshes.has(mesh);
    }

    public getLiquefiableConfig(mesh: Mesh): LiquefiableBehaviorConfig | undefined {
        return this.liquefiableConfigByMesh.get(mesh);
    }

    public getLinkedEntityNames(mesh: Mesh): readonly string[] {
        return nonEmptyNames(this.liquefiableConfigByMesh.get(mesh)?.linked);
    }

    public getExcludedEntityNames(mesh: Mesh): readonly string[] {
        return nonEmptyNames(this.liquefiableConfigByMesh.get(mesh)?.excludeSDF);
    }

    public retireMesh(mesh: Mesh): MeshBehaviorAvailability {
        return {
            liquefiable: this.liquefiableMeshes.delete(mesh),
            dissolvable: this.dissolvableMeshes.delete(mesh),
        };
    }

    public restoreMesh(mesh: Mesh, availability: MeshBehaviorAvailability): void {
        if (availability.liquefiable) this.liquefiableMeshes.add(mesh);
        if (availability.dissolvable) this.dissolvableMeshes.add(mesh);
    }

    public describeInstances(): Array<{ name: string; mesh: string }> {
        return this.instances.map((behavior) => ({
            name: behavior.name,
            mesh: this.entityNameOf(behavior.mesh),
        }));
    }

    public dispose(): void {
        for (let index = this.instances.length - 1; index >= 0; index--) this.instances[index]!.dispose();
        this.instances.length = 0;
        this.events.dispose();
        this.started = false;
    }

    private assignmentsOf(entityName: string): BehaviorAssignment[] {
        let assignments = this.assignmentCache.get(entityName);
        if (assignments) return assignments;
        const references = this.entities?.[entityName]?.behaviors ?? [];
        assignments = references.map((reference) => {
            const base = this.library?.[reference.name];
            if (!base) throw new Error(`[aquanova] entity "${entityName}" references undefined behavior "${reference.name}"`);
            return { ...base, ...reference };
        });
        this.assignmentCache.set(entityName, assignments);
        return assignments;
    }

    private liquefiableConfigOf(entityName: string): LiquefiableBehaviorConfig | undefined {
        return this.assignmentsOf(entityName).find(isLiquefiableBehaviorConfig);
    }

    private resolveDissolvableEntityNames(): Set<string> {
        const dissolvableNames = new Set<string>();
        const pending = Object.keys(this.entities ?? {}).filter((name) => this.liquefiableConfigOf(name) !== undefined);
        while (pending.length) {
            const entityName = pending.pop()!;
            if (dissolvableNames.has(entityName)) continue;
            dissolvableNames.add(entityName);
            for (const linked of nonEmptyNames(this.liquefiableConfigOf(entityName)?.linked)) pending.push(linked);
        }
        return dissolvableNames;
    }

    private clearClassification(): void {
        this.dynamicMeshes.clear();
        this.movableMeshes.clear();
        this.liquefiableMeshes.clear();
        this.dissolvableMeshes.clear();
        this.dissolvableInstanceIds.clear();
        this.liquefiableConfigByMesh.clear();
    }
}

function createBehavior(assignment: BehaviorAssignment, mesh: Mesh, context: BehaviorContext): Behavior {
    switch (assignment.name) {
        case "dynamic":
            return new DynamicBehavior(mesh, assignment);
        case "player":
            return new PlayerBehavior(mesh, assignment, context);
        case "weapon":
            return new WeaponBehavior(mesh, assignment);
    }
    if (isLiquefiableBehaviorConfig(assignment)) return new LiquefiableBehavior(assignment.name, mesh, assignment, context);
    throw new Error(`[aquanova] behavior "${assignment.name}" has no runtime implementation`);
}

function nonEmptyNames(names: readonly string[] | undefined): string[] {
    return (names ?? []).filter((name) => name.length > 0);
}
