import type { Mesh, PhysicsWorld, SceneContext } from "babylon-lite";
import { BehaviorManager as GenericBehaviorManager } from "../behavior-system/behavior-manager.js";
import * as behaviorConstructors from "./behavior-constructors.js";
import { resolveDynamicMass } from "./dynamic.js";
import { AquanovaEventManager } from "./aquanova-event-manager.js";
import type { AquanovaGameContext } from "./game-context.js";
import { AquanovaFluidRuntime } from "../fluid-runtime.js";
import { PlayerBehavior } from "./player.js";
import type { Behavior, BehaviorAssignment, BehaviorPresets, Entities, LiquefiableBehaviorConfig } from "./types.js";
import { WeaponInventory } from "./weapon-inventory.js";

export interface AquanovaBehaviorManagerOptions {
    readonly presets?: BehaviorPresets;
    readonly entities: Entities | undefined;
    readonly doors?: readonly { readonly id: string; readonly behaviors?: readonly BehaviorAssignment[] }[];
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

/** Aquanova-owned behavior services and gameplay queries built over the generic behavior runtime. */
export class AquanovaBehaviorManager {
    public readonly events = new AquanovaEventManager();
    public readonly dynamicMeshes = new Set<Mesh>();
    public readonly movableMeshes = new Set<Mesh>();
    public readonly liquefiableMeshes = new Set<Mesh>();
    public readonly dissolvableMeshes = new Set<Mesh>();
    public readonly dissolvableInstanceIds = new Set<string>();
    public readonly fluidSimulations = new AquanovaFluidRuntime();
    private readonly core: GenericBehaviorManager<AquanovaGameContext>;
    private readonly entityNameOf: (mesh: Mesh) => string;
    private readonly meshesByEntityName: ReadonlyMap<string, readonly Mesh[]>;
    private readonly liquefiableConfigByMesh = new Map<Mesh, LiquefiableBehaviorConfig>();
    private readonly linkedEntityNamesByEntityName = new Map<string, readonly string[]>();
    private readonly dynamicMassByMesh = new Map<Mesh, number>();
    private readonly weaponInventory = new WeaponInventory();
    private started = false;

    public constructor(options: AquanovaBehaviorManagerOptions) {
        this.entityNameOf = options.entityNameOf;
        this.meshesByEntityName = options.meshesByEntityName;
        this.core = new GenericBehaviorManager({
            presets: options.presets,
            entities: options.entities,
            meshlessOwners: Object.fromEntries((options.doors ?? []).map((door) => [door.id, { behaviors: door.behaviors }])),
            meshesByEntityName: options.meshesByEntityName,
            constructors: behaviorConstructors,
            shouldInstantiate: (assignment) => assignment.reflectionProbe === undefined,
        });
    }

    public get instances(): readonly Behavior[] {
        return this.core.instances;
    }

    public findEntityWithBehavior(behaviorName: string): BehaviorEntityMatch | undefined {
        return this.core.findEntityWithBehavior(behaviorName) as BehaviorEntityMatch | undefined;
    }

    public acquireAllWeapons(): void {
        for (const behaviorName of ["weaponLiquefactor", "weaponAntiGravityGun"]) {
            const weapon = this.findEntityWithBehavior(behaviorName);
            if (weapon?.meshes.length) {
                this.events.emit("entityEvent", { name: weapon.entityName, event: "enable" });
            }
        }
    }

    public classifyMeshes(meshes: readonly Mesh[], classification: BehaviorMeshClassification): void {
        this.clearClassification();
        const liquefactionGraph = this.resolveLiquefactionGraph();
        for (const [entityName, linkedNames] of liquefactionGraph.linkedNamesByEntityName) {
            this.linkedEntityNamesByEntityName.set(entityName, linkedNames);
        }
        for (const mesh of meshes) {
            if (classification.isDisabled(mesh)) {
                continue;
            }
            const entityName = this.entityNameOf(mesh);
            const assignments = this.assignmentsOf(entityName);
            const liquefiableConfig = liquefactionGraph.configByEntityName.get(entityName);
            if (liquefiableConfig) {
                this.liquefiableConfigByMesh.set(mesh, liquefiableConfig);
            }
            const dynamicConfig = assignments.find((assignment) => assignment.name === "dynamic");
            const explicitlyDynamic = dynamicConfig !== undefined;
            const dissolvable = liquefactionGraph.linkedNamesByEntityName.has(entityName);
            if (dissolvable) {
                this.dissolvableMeshes.add(mesh);
                const instanceId = classification.instanceIdOf(mesh);
                if (instanceId) {
                    this.dissolvableInstanceIds.add(instanceId);
                }
            }
            if (dissolvable || explicitlyDynamic) {
                this.dynamicMeshes.add(mesh);
            }
            if (dynamicConfig) {
                this.movableMeshes.add(mesh);
                this.dynamicMassByMesh.set(mesh, resolveDynamicMass(dynamicConfig));
            }
            if (liquefiableConfig) {
                this.liquefiableMeshes.add(mesh);
            }
        }
    }

    public async start(context: Omit<AquanovaGameContext, "events" | "fluidSimulations" | "weaponInventory">): Promise<void> {
        if (this.started) {
            throw new Error("[aquanova] behaviors are already started");
        }
        this.started = true;
        this.weaponInventory.start(this.events);
        try {
            await this.core.start({ ...context, events: this.events, fluidSimulations: this.fluidSimulations, weaponInventory: this.weaponInventory });
        } catch (error) {
            this.weaponInventory.dispose();
            this.started = false;
            throw error;
        }
    }

    public bindSystemEvents(scene: SceneContext, world: PhysicsWorld): void {
        this.events.bindSystemEvents(scene, world);
    }

    public get player(): PlayerBehavior | null {
        return this.core.instances.find((behavior): behavior is PlayerBehavior => behavior instanceof PlayerBehavior) ?? null;
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
        return this.linkedEntityNamesByEntityName.get(this.entityNameOf(mesh)) ?? [];
    }

    public retireMesh(mesh: Mesh): MeshBehaviorAvailability {
        return {
            liquefiable: this.liquefiableMeshes.delete(mesh),
            dissolvable: this.dissolvableMeshes.delete(mesh),
        };
    }

    public restoreMesh(mesh: Mesh, availability: MeshBehaviorAvailability): void {
        if (availability.liquefiable) {
            this.liquefiableMeshes.add(mesh);
        }
        if (availability.dissolvable) {
            this.dissolvableMeshes.add(mesh);
        }
    }

    public describeInstances(): Array<{ name: string; mesh: string }> {
        return this.core.describeInstances(this.entityNameOf);
    }

    public dispose(): void {
        this.core.dispose();
        this.fluidSimulations.dispose();
        this.weaponInventory.dispose();
        this.events.dispose();
        this.started = false;
    }

    public assignmentsOf(entityName: string): readonly BehaviorAssignment[] {
        return this.core.assignmentsOf(entityName) as readonly BehaviorAssignment[];
    }

    private liquefiableConfigOf(entityName: string): LiquefiableBehaviorConfig | undefined {
        const assignment = this.assignmentsOf(entityName).find((candidate) => candidate.name === "liquefaction");
        return assignment as LiquefiableBehaviorConfig | undefined;
    }

    private resolveLiquefactionGraph(): {
        configByEntityName: Map<string, LiquefiableBehaviorConfig>;
        linkedNamesByEntityName: Map<string, readonly string[]>;
    } {
        const ownConfigs = new Map(
            this.entityNames()
                .filter((entityName) => (this.meshesByEntityName.get(entityName)?.length ?? 0) > 0)
                .map((entityName) => [entityName, this.liquefiableConfigOf(entityName)] as const)
                .filter((entry): entry is readonly [string, LiquefiableBehaviorConfig] => entry[1] !== undefined)
        );
        const adjacency = new Map<string, Set<string>>();
        const inheritedCandidates = new Map<string, Array<{ source: string; config: LiquefiableBehaviorConfig }>>();
        for (const [source, config] of ownConfigs) {
            ensureLinkedEntity(adjacency, source);
            for (const target of nonEmptyNames(config.linked)) {
                addLinkedEdge(adjacency, source, target);
                if (!ownConfigs.has(target)) {
                    let candidates = inheritedCandidates.get(target);
                    if (!candidates) {
                        candidates = [];
                        inheritedCandidates.set(target, candidates);
                    }
                    candidates.push({ source, config });
                }
            }
        }
        const configByEntityName = new Map(ownConfigs);
        for (const [entityName, candidates] of inheritedCandidates) {
            const profiles = new Map(candidates.map((candidate) => [liquefactionProfileKey(candidate.config), candidate]));
            if (profiles.size > 1) {
                throw new Error(
                    `[aquanova] linked entity "${entityName}" inherits conflicting liquefaction behaviors from ${candidates.map(({ source }) => `"${source}"`).join(", ")}`
                );
            }
            configByEntityName.set(entityName, candidates[0]!.config);
        }
        const linkedNamesByEntityName = new Map<string, readonly string[]>();
        const visited = new Set<string>();
        for (const entityName of adjacency.keys()) {
            if (visited.has(entityName)) continue;
            const component: string[] = [];
            const pending = [entityName];
            while (pending.length) {
                const next = pending.pop()!;
                if (visited.has(next)) continue;
                visited.add(next);
                component.push(next);
                for (const linked of adjacency.get(next) ?? []) pending.push(linked);
            }
            for (const member of component) {
                linkedNamesByEntityName.set(
                    member,
                    component.filter((candidate) => candidate !== member)
                );
            }
        }
        return { configByEntityName, linkedNamesByEntityName };
    }

    private entityNames(): string[] {
        return [...this.core.entityNames()];
    }

    private clearClassification(): void {
        this.dynamicMeshes.clear();
        this.movableMeshes.clear();
        this.liquefiableMeshes.clear();
        this.dissolvableMeshes.clear();
        this.dissolvableInstanceIds.clear();
        this.liquefiableConfigByMesh.clear();
        this.linkedEntityNamesByEntityName.clear();
        this.dynamicMassByMesh.clear();
    }
}

function ensureLinkedEntity(adjacency: Map<string, Set<string>>, entityName: string): Set<string> {
    let links = adjacency.get(entityName);
    if (!links) {
        links = new Set();
        adjacency.set(entityName, links);
    }
    return links;
}

function addLinkedEdge(adjacency: Map<string, Set<string>>, first: string, second: string): void {
    ensureLinkedEntity(adjacency, first).add(second);
    ensureLinkedEntity(adjacency, second).add(first);
}

function liquefactionProfileKey(config: LiquefiableBehaviorConfig): string {
    return JSON.stringify({
        name: (config as LiquefiableBehaviorConfig & { name?: string }).name,
        fluidSim: config.fluidSim ?? [],
        sound: config.sound,
    });
}

function nonEmptyNames(names: readonly string[] | undefined): string[] {
    return (names ?? []).filter((name) => name.length > 0);
}
