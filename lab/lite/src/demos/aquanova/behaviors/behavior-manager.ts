import type { Mesh, PhysicsWorld, SceneContext } from "babylon-lite";
import { DynamicBehavior, resolveDynamicMass } from "./dynamic.js";
import { DisableEntityBehavior, EnableEntityBehavior } from "./entity-toggle.js";
import { EventManager } from "./event-manager.js";
import { LiquefiableBehavior } from "./liquefiable.js";
import { PickEntityBehavior } from "./pick-entity.js";
import { PlayAnimationBehavior } from "./play-animation.js";
import { PlayerBehavior } from "./player.js";
import { isEntityToggleBehaviorConfig, isLiquefiableBehaviorConfig, isPickEntityBehaviorConfig } from "./types.js";
import type { Behavior, BehaviorAssignment, BehaviorContext, BehaviorLibrary, Entities, LiquefiableBehaviorConfig } from "./types.js";
import { WeaponLiquefactorBehavior } from "./weapon-liquefactor.js";
import { WeaponAntiGravityGunBehavior } from "./weapon-anti-gravity-gun.js";
import { WeaponInventory } from "./weapon-inventory.js";

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
    private readonly dynamicMassByMesh = new Map<Mesh, number>();
    private readonly weaponInventory = new WeaponInventory();
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
            const explicitlyDynamic = assignments.some((assignment) => assignment.name === "dynamic");
            const dynamicConfig = assignments.find((assignment) => assignment.name === "dynamic");
            const dissolvable = dissolvableEntityNames.has(entityName);
            if (dissolvable) {
                this.dissolvableMeshes.add(mesh);
                const instanceId = classification.instanceIdOf(mesh);
                if (instanceId) this.dissolvableInstanceIds.add(instanceId);
            }
            if (dissolvable || explicitlyDynamic) this.dynamicMeshes.add(mesh);
            if (explicitlyDynamic) {
                this.movableMeshes.add(mesh);
                this.dynamicMassByMesh.set(mesh, resolveDynamicMass(dynamicConfig!));
            }
            if (liquefiableConfig) this.liquefiableMeshes.add(mesh);
        }
    }

    public async start(context: Omit<BehaviorContext, "events" | "weaponInventory">): Promise<void> {
        if (this.started) throw new Error("[aquanova] behaviors are already started");
        this.started = true;
        this.weaponInventory.start(this.events);
        const behaviorContext: BehaviorContext = { ...context, events: this.events, weaponInventory: this.weaponInventory };
        try {
            const weapon = this.findEntityWithBehavior("weaponLiquefactor");
            if (weapon?.meshes.length) await WeaponLiquefactorBehavior.init(weapon.assignment);
            const pickEntityConfigs = Object.keys(this.entities ?? {}).flatMap((entityName) => {
                if (!(this.meshesByEntityName.get(entityName)?.length ?? 0)) {
                    return [];
                }
                return this.assignmentsOf(entityName).filter(isPickEntityBehaviorConfig);
            });
            await PickEntityBehavior.init(pickEntityConfigs);
            for (const entityName of Object.keys(this.entities ?? {})) {
                const meshes = this.meshesByEntityName.get(entityName) ?? [];
                for (const assignment of this.assignmentsOf(entityName)) {
                    if (assignment.reflectionProbe) {
                        continue;
                    }
                    if (isPickEntityBehaviorConfig(assignment)) {
                        if (meshes.length) {
                            this.instances.push(new PickEntityBehavior(meshes, assignment, behaviorContext, entityName));
                        }
                        continue;
                    }
                    const targets =
                        assignment.name === "player" ||
                        assignment.name === "weaponLiquefactor" ||
                        assignment.name === "weaponAntiGravityGun" ||
                        assignment.name === "playAnimation" ||
                        isEntityToggleBehaviorConfig(assignment)
                            ? meshes.slice(0, 1)
                            : meshes;
                    for (const mesh of targets) this.instances.push(createBehavior(assignment, mesh, behaviorContext, entityName));
                }
            }
            for (const behavior of this.instances) behavior.start();
        } catch (error) {
            for (let index = this.instances.length - 1; index >= 0; index--) this.instances[index]!.dispose();
            this.instances.length = 0;
            PickEntityBehavior.dispose();
            WeaponLiquefactorBehavior.dispose();
            this.weaponInventory.dispose();
            this.started = false;
            throw error;
        }
    }

    public setSoundsEnabled(enabled: boolean): void {
        PickEntityBehavior.setSoundEnabled(enabled);
        WeaponLiquefactorBehavior.setSoundEnabled(enabled);
    }

    public setSoundVolume(volume: number): void {
        PickEntityBehavior.setSoundVolume(volume);
        WeaponLiquefactorBehavior.setSoundVolume(volume);
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

    public getDynamicMass(mesh: Mesh): number | null {
        return this.dynamicMassByMesh.get(mesh) ?? null;
    }

    public getLinkedEntityNames(mesh: Mesh): readonly string[] {
        return nonEmptyNames(this.liquefiableConfigByMesh.get(mesh)?.linked);
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
        PickEntityBehavior.dispose();
        WeaponLiquefactorBehavior.dispose();
        this.weaponInventory.dispose();
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
        this.dynamicMassByMesh.clear();
    }
}

function createBehavior(assignment: BehaviorAssignment, mesh: Mesh, context: BehaviorContext, entityName: string): Behavior {
    if (isEntityToggleBehaviorConfig(assignment)) {
        return assignment.name === "disableEntity"
            ? new DisableEntityBehavior(entityName, mesh, assignment, context)
            : new EnableEntityBehavior(entityName, mesh, assignment, context);
    }
    switch (assignment.name) {
        case "dynamic":
            return new DynamicBehavior(mesh, assignment);
        case "player":
            return new PlayerBehavior(mesh, assignment, context);
        case "playAnimation":
            return new PlayAnimationBehavior(entityName, mesh, assignment, context.animationGroups);
        case "weaponLiquefactor":
            return new WeaponLiquefactorBehavior(entityName, mesh, assignment, context);
        case "weaponAntiGravityGun":
            return new WeaponAntiGravityGunBehavior(entityName, mesh, assignment, context);
    }
    if (isLiquefiableBehaviorConfig(assignment)) return new LiquefiableBehavior(assignment.name, mesh, assignment, context);
    throw new Error(`[aquanova] behavior "${assignment.name}" has no runtime implementation`);
}

function nonEmptyNames(names: readonly string[] | undefined): string[] {
    return (names ?? []).filter((name) => name.length > 0);
}
