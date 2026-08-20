import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
import type { Behavior, BehaviorEventSubscription, EntityToggleBehaviorConfig } from "./types.js";

type EntityToggleContext = Pick<AquanovaGameContext, "events">;
type EntityToggleBehaviorName = "disableCollision" | "disableEntity" | "enableCollision" | "enableEntity" | "hideEntity" | "removeEntity" | "showEntity";
type EntityToggleEventName = "disable" | "disableCollision" | "enable" | "enableCollision" | "hide" | "remove" | "show";

class EntityToggleBehavior<Name extends EntityToggleBehaviorName> implements Behavior<Name> {
    public readonly name: Name;
    public readonly mesh: Mesh | null;
    public readonly config: EntityToggleBehaviorConfig;
    private readonly entityName: string;
    private readonly outputEvent: EntityToggleEventName;
    private readonly context: EntityToggleContext;
    private stopEntityEvent: (() => void) | null = null;

    public constructor(
        name: Name,
        outputEvent: EntityToggleEventName,
        entityName: string,
        meshes: readonly Mesh[],
        config: EntityToggleBehaviorConfig,
        context: EntityToggleContext
    ) {
        validateEventConfiguration(name, config);
        this.name = name;
        this.mesh = meshes[0] ?? null;
        this.config = config;
        this.entityName = entityName;
        this.outputEvent = outputEvent;
        this.context = context;
    }

    public init(): void {}

    public start(): void {
        if (!this.config.events && !this.config.onEvent && !this.config.entity) {
            this.context.events.emit("entityEvent", {
                name: this.entityName,
                event: this.outputEvent,
            });
            return;
        }
        this.stopEntityEvent = this.context.events.on("entityEvent", ({ name, event }) => {
            if (this.config.events?.some((subscription) => subscriptionMatches(subscription, name, event))) {
                this.context.events.emit("entityEvent", {
                    name: this.entityName,
                    event: this.outputEvent,
                });
            } else if (this.config.onEvent && this.config.entity && name === this.entityName && event === this.config.onEvent) {
                this.context.events.emit("entityEvent", {
                    name: this.config.entity,
                    event: this.outputEvent,
                });
            }
        });
    }

    public dispose(): void {
        this.stopEntityEvent?.();
        this.stopEntityEvent = null;
    }
}

function validateEventConfiguration(name: EntityToggleBehaviorName, config: EntityToggleBehaviorConfig): void {
    if (config.events !== undefined) {
        if (config.onEvent !== undefined || config.entity !== undefined) {
            throw new Error(`[aquanova] ${name} cannot combine events with legacy onEvent/entity`);
        }
        if (config.events.length === 0) {
            throw new Error(`[aquanova] ${name}.events must contain at least one event`);
        }
        for (const event of config.events) {
            if (!event.name) {
                throw new Error(`[aquanova] ${name}.events[].name must be a non-empty event name`);
            }
            if (typeof event.source === "string" && !event.source) {
                throw new Error(`[aquanova] ${name}.events[].source must be a non-empty entity or door name`);
            }
            if (Array.isArray(event.source) && event.source.length === 0) {
                throw new Error(`[aquanova] ${name}.events[].source must contain at least one entity or door name`);
            }
            if (Array.isArray(event.source) && event.source.some((source) => !source)) {
                throw new Error(`[aquanova] ${name}.events[].source entries must be non-empty entity or door names`);
            }
        }

        return;
    }
    if ((config.onEvent === undefined) !== (config.entity === undefined)) {
        throw new Error(`[aquanova] ${name} legacy onEvent and entity must be provided together`);
    }
    if (config.onEvent !== undefined && !config.onEvent) {
        throw new Error(`[aquanova] ${name}.onEvent must be a non-empty event name`);
    }
    if (config.entity !== undefined && !config.entity) {
        throw new Error(`[aquanova] ${name}.entity must be a non-empty entity or door name`);
    }
}

function subscriptionMatches(subscription: BehaviorEventSubscription, source: string, event: string): boolean {
    if (subscription.name !== event) {
        return false;
    }
    return typeof subscription.source === "string" ? subscription.source === source : subscription.source.includes(source);
}

export class DisableEntityBehavior extends EntityToggleBehavior<"disableEntity"> {
    public constructor(entityName: string, meshes: readonly Mesh[], config: EntityToggleBehaviorConfig, context: EntityToggleContext) {
        super("disableEntity", "disable", entityName, meshes, config, context);
    }
}

export class DisableCollisionBehavior extends EntityToggleBehavior<"disableCollision"> {
    public constructor(entityName: string, meshes: readonly Mesh[], config: EntityToggleBehaviorConfig, context: EntityToggleContext) {
        super("disableCollision", "disableCollision", entityName, meshes, config, context);
    }
}

export class EnableCollisionBehavior extends EntityToggleBehavior<"enableCollision"> {
    public constructor(entityName: string, meshes: readonly Mesh[], config: EntityToggleBehaviorConfig, context: EntityToggleContext) {
        super("enableCollision", "enableCollision", entityName, meshes, config, context);
    }
}

export class EnableEntityBehavior extends EntityToggleBehavior<"enableEntity"> {
    public constructor(entityName: string, meshes: readonly Mesh[], config: EntityToggleBehaviorConfig, context: EntityToggleContext) {
        super("enableEntity", "enable", entityName, meshes, config, context);
    }
}

export class HideEntityBehavior extends EntityToggleBehavior<"hideEntity"> {
    public constructor(entityName: string, meshes: readonly Mesh[], config: EntityToggleBehaviorConfig, context: EntityToggleContext) {
        super("hideEntity", "hide", entityName, meshes, config, context);
    }
}

export class RemoveEntityBehavior extends EntityToggleBehavior<"removeEntity"> {
    public constructor(entityName: string, meshes: readonly Mesh[], config: EntityToggleBehaviorConfig, context: EntityToggleContext) {
        super("removeEntity", "remove", entityName, meshes, config, context);
    }
}

export class ShowEntityBehavior extends EntityToggleBehavior<"showEntity"> {
    public constructor(entityName: string, meshes: readonly Mesh[], config: EntityToggleBehaviorConfig, context: EntityToggleContext) {
        super("showEntity", "show", entityName, meshes, config, context);
    }
}
