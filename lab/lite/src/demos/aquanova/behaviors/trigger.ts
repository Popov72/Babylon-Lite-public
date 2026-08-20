import type { Mesh } from "babylon-lite";
import type { AquanovaGameContext, IntersectionTriggerRegistration } from "./game-context.js";
import type { Behavior, TriggerBehaviorConfig } from "./types.js";

type TriggerContext = Pick<AquanovaGameContext, "events" | "registerIntersectionTrigger">;

export class TriggerBehavior implements Behavior<"trigger"> {
    public readonly name = "trigger";
    public readonly mesh: Mesh;
    public readonly config: TriggerBehaviorConfig;
    private readonly entityName: string;
    private readonly context: TriggerContext;
    private registration: IntersectionTriggerRegistration | null = null;
    private stopEntityEvent: (() => void) | null = null;
    private enabled = true;

    public constructor(entityName: string, meshes: readonly Mesh[], config: TriggerBehaviorConfig, context: TriggerContext) {
        const mesh = meshes[0];
        if (!mesh) {
            throw new Error("[aquanova] trigger requires at least one mesh");
        }
        const intersection = config.onIntersection;
        if (!intersection) {
            throw new Error("[aquanova] trigger.onIntersection must be provided");
        }
        if (intersection.enterEvent !== undefined && !intersection.enterEvent) {
            throw new Error("[aquanova] trigger.onIntersection.enterEvent must be a non-empty event name when provided");
        }
        if (intersection.exitEvent !== undefined && !intersection.exitEvent) {
            throw new Error("[aquanova] trigger.onIntersection.exitEvent must be a non-empty event name when provided");
        }
        if (intersection.raiseEvent !== undefined && !intersection.raiseEvent) {
            throw new Error("[aquanova] trigger.onIntersection.raiseEvent must be a non-empty event name when provided");
        }
        if ((intersection.enterEvent !== undefined || intersection.exitEvent !== undefined) && intersection.raiseEvent !== undefined) {
            throw new Error("[aquanova] trigger.onIntersection cannot combine enterEvent/exitEvent with legacy raiseEvent");
        }
        if (intersection.entity !== undefined && !intersection.entity) {
            throw new Error("[aquanova] trigger.onIntersection.entity must be a non-empty entity name when provided");
        }
        if (intersection.entity !== undefined && intersection.raiseEvent === undefined) {
            throw new Error("[aquanova] trigger.onIntersection.entity is only supported with legacy raiseEvent");
        }
        this.entityName = entityName;
        this.mesh = mesh;
        this.config = config;
        this.context = context;
    }

    public init(): void {}

    public start(): void {
        const intersection = this.config.onIntersection;
        this.registration = this.context.registerIntersectionTrigger(this.entityName, intersection.playerOnly ?? false, {
            onEntered: () => {
                this.raise(intersection.enterEvent ?? intersection.raiseEvent, intersection.entity);
            },
            onExited: () => {
                this.raise(intersection.exitEvent);
            },
        });
        this.stopEntityEvent = this.context.events.on("entityEvent", ({ name, event }) => {
            if (name !== this.entityName) {
                return;
            }
            if (event === "enable") {
                this.enabled = true;
                this.registration?.setEnabled(true);
            } else if (event === "disable") {
                this.enabled = false;
                this.registration?.setEnabled(false);
            }
        });
    }

    public dispose(): void {
        this.stopEntityEvent?.();
        this.stopEntityEvent = null;
        this.registration?.dispose();
        this.registration = null;
    }

    private raise(event: string | undefined, legacyTarget?: string): void {
        if (!this.enabled || !event) {
            return;
        }
        this.context.events.emit("entityEvent", {
            name: legacyTarget ?? this.entityName,
            event,
        });
    }
}
