import { onPhysicsTriggerBodies, setPhysicsShapeFilterCollideMask, setPhysicsShapeIsTrigger } from "babylon-lite";
import type { PhysicsBody, PhysicsTriggerBodyInfo, PhysicsWorld } from "babylon-lite";
import type { IntersectionTriggerCallbacks, IntersectionTriggerRegistration } from "./behaviors/game-context.js";

interface RegisteredTrigger {
    readonly bodies: ReadonlySet<PhysicsBody>;
    readonly playerOnly: boolean;
    readonly callbacks: IntersectionTriggerCallbacks;
    readonly overlaps: Map<PhysicsBody, number>;
    enabled: boolean;
}

export interface IntersectionTriggerRegistry {
    register(entityName: string, playerOnly: boolean, callbacks: IntersectionTriggerCallbacks): IntersectionTriggerRegistration;
    dispose(): void;
}

export function createIntersectionTriggerRegistry(
    world: PhysicsWorld,
    playerBody: PhysicsBody,
    bodiesOfEntity: (entityName: string) => readonly PhysicsBody[]
): IntersectionTriggerRegistry {
    const triggers = new Set<RegisteredTrigger>();
    const handleTrigger = ({ type, bodyA, bodyB }: PhysicsTriggerBodyInfo): void => {
        if (!bodyA || !bodyB) return;
        for (const trigger of triggers) {
            if (!trigger.enabled) continue;
            const aIsTrigger = trigger.bodies.has(bodyA);
            const bIsTrigger = trigger.bodies.has(bodyB);
            if (aIsTrigger === bIsTrigger) continue;
            const other = aIsTrigger ? bodyB : bodyA;
            if (trigger.playerOnly && other !== playerBody) continue;
            const previous = trigger.overlaps.get(other) ?? 0;
            if (type === "ENTERED") {
                trigger.overlaps.set(other, previous + 1);
                if (previous === 0 && trigger.overlaps.size === 1) trigger.callbacks.onEntered();
            } else if (previous <= 1) {
                trigger.overlaps.delete(other);
                if (previous > 0 && trigger.overlaps.size === 0) trigger.callbacks.onExited();
            } else {
                trigger.overlaps.set(other, previous - 1);
            }
        }
    };
    let stopTriggerEvents: (() => void) | null = null;

    return {
        register: (entityName, playerOnly, callbacks) => {
            const bodies = new Set(bodiesOfEntity(entityName).filter((body) => body._shape));
            if (bodies.size === 0) throw new Error(`[aquanova] trigger entity "${entityName}" has no collision shape`);
            for (const body of bodies) setPhysicsShapeIsTrigger(world, body._shape!, true);
            const trigger: RegisteredTrigger = { bodies, playerOnly, callbacks, overlaps: new Map(), enabled: true };
            triggers.add(trigger);
            stopTriggerEvents ??= onPhysicsTriggerBodies(world, handleTrigger);
            return {
                setEnabled: (enabled) => {
                    if (trigger.enabled === enabled) return;
                    trigger.enabled = enabled;
                    trigger.overlaps.clear();
                    for (const body of bodies) setPhysicsShapeFilterCollideMask(world, body._shape!, enabled ? 0xffffffff : 0);
                },
                dispose: () => {
                    if (!triggers.delete(trigger)) return;
                    trigger.overlaps.clear();
                    for (const body of bodies) {
                        setPhysicsShapeFilterCollideMask(world, body._shape!, 0xffffffff);
                        setPhysicsShapeIsTrigger(world, body._shape!, false);
                    }
                    if (triggers.size === 0) {
                        stopTriggerEvents?.();
                        stopTriggerEvents = null;
                    }
                },
            };
        },
        dispose: () => {
            stopTriggerEvents?.();
            stopTriggerEvents = null;
            for (const trigger of triggers) {
                for (const body of trigger.bodies) {
                    if (body._shape) {
                        setPhysicsShapeFilterCollideMask(world, body._shape, 0xffffffff);
                        setPhysicsShapeIsTrigger(world, body._shape, false);
                    }
                }
            }
            triggers.clear();
        },
    };
}
