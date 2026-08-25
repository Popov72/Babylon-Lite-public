import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext } from "./game-context.js";
import type { Behavior, EntityToggleBehaviorConfig } from "./types.js";
import { assertBehaviorConfigKeys } from "./behavior-config-validation.js";
import { eventSubscriptionMatches, validateEventSubscriptions } from "./event-subscription.js";

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
        assertBehaviorConfigKeys(config, name, ["events"]);
        validateEventSubscriptions(name, config.events);
        this.name = name;
        this.mesh = meshes[0] ?? null;
        this.config = config;
        this.entityName = entityName;
        this.outputEvent = outputEvent;
        this.context = context;
    }

    public init(): void {}

    public start(): void {
        if (!this.config.events) {
            this.context.events.emit("entityEvent", {
                name: this.entityName,
                event: this.outputEvent,
            });
            return;
        }
        this.stopEntityEvent = this.context.events.on("entityEvent", ({ name, event }) => {
            if (this.config.events?.some((subscription) => eventSubscriptionMatches(subscription, name, event))) {
                this.context.events.emit("entityEvent", {
                    name: this.entityName,
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
