import type { Mesh } from "babylon-lite";
import type { Behavior, BehaviorContext, EntityToggleBehaviorConfig } from "./types.js";

type EntityToggleContext = Pick<BehaviorContext, "events">;
type EntityToggleBehaviorName = "disableEntity" | "enableEntity";
type EntityToggleEventName = "disable" | "enable";

class EntityToggleBehavior<Name extends EntityToggleBehaviorName> implements Behavior<Name> {
    public readonly name: Name;
    public readonly mesh: Mesh;
    public readonly config: EntityToggleBehaviorConfig;
    private readonly entityName: string;
    private readonly outputEvent: EntityToggleEventName;
    private readonly context: EntityToggleContext;
    private stopEntityEvent: (() => void) | null = null;

    public constructor(name: Name, outputEvent: EntityToggleEventName, entityName: string, mesh: Mesh, config: EntityToggleBehaviorConfig, context: EntityToggleContext) {
        if (!config.onEvent) {
            throw new Error(`[aquanova] ${name}.onEvent must be a non-empty event name`);
        }
        if (!config.entity) {
            throw new Error(`[aquanova] ${name}.entity must be a non-empty entity or door name`);
        }
        this.name = name;
        this.mesh = mesh;
        this.config = config;
        this.entityName = entityName;
        this.outputEvent = outputEvent;
        this.context = context;
    }

    public start(): void {
        this.stopEntityEvent = this.context.events.on("entityEvent", ({ name, event }) => {
            if (name === this.entityName && event === this.config.onEvent) {
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

export class DisableEntityBehavior extends EntityToggleBehavior<"disableEntity"> {
    public constructor(entityName: string, mesh: Mesh, config: EntityToggleBehaviorConfig, context: EntityToggleContext) {
        super("disableEntity", "disable", entityName, mesh, config, context);
    }
}

export class EnableEntityBehavior extends EntityToggleBehavior<"enableEntity"> {
    public constructor(entityName: string, mesh: Mesh, config: EntityToggleBehaviorConfig, context: EntityToggleContext) {
        super("enableEntity", "enable", entityName, mesh, config, context);
    }
}
