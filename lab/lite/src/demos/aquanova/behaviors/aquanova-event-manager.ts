import type { PhysicsWorld, SceneContext } from "babylon-lite";
import { onPhysicsAfterStep } from "babylon-lite";
import { EventManager as GenericEventManager } from "../behavior-system/event-manager.js";
import type { AquanovaEventMap } from "./events.js";
import type { SystemEventMap } from "./system-events.js";

export type EventMap = AquanovaEventMap & SystemEventMap;

/** Owns typed gameplay dispatch and the engine hooks that publish system events. */
export class AquanovaEventManager extends GenericEventManager<EventMap> {
    private scene: SceneContext | null = null;
    private world: PhysicsWorld | null = null;

    /**
     * Bind system events after all demo frame callbacks have been registered.
     * The scene executes `_beforeRender` in array order, so the manager brackets the existing
     * callbacks with frame-start and frame-end publishers.
     */
    public bindSystemEvents(scene: SceneContext, world: PhysicsWorld): void {
        if (this.scene || this.world) {
            throw new Error("[aquanova] system events are already bound");
        }
        this.scene = scene;
        this.world = world;
        scene._beforeRender.unshift(this.frameStart);
        scene._beforeRender.push(this.frameEnd);
        onPhysicsAfterStep(world, this.physicsStep);
    }

    public dispose(): void {
        if (this.scene) {
            removeCallback(this.scene._beforeRender, this.frameStart);
            removeCallback(this.scene._beforeRender, this.frameEnd);
        }
        if (this.world?._afterStep) {
            removeCallback(this.world._afterStep, this.physicsStep);
        }
        this.scene = null;
        this.world = null;
        super.dispose();
    }

    private readonly frameStart = (deltaMs: number): void => {
        this.emit("frameStart", { deltaMs });
    };

    private readonly physicsStep = (deltaSeconds: number): void => {
        this.emit("physicsStep", { deltaSeconds });
    };

    private readonly frameEnd = (deltaMs: number): void => {
        this.emit("frameEnd", { deltaMs });
    };
}

function removeCallback<T>(callbacks: T[], callback: T): void {
    const index = callbacks.indexOf(callback);
    if (index >= 0) callbacks.splice(index, 1);
}
