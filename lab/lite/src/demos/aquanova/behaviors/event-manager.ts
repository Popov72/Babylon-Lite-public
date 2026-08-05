import type { PhysicsWorld, SceneContext } from "babylon-lite";
import { onPhysicsAfterStep } from "babylon-lite";
import { TypedEventBus } from "./event-bus.js";
import type { EventMap } from "./events.js";

/** Owns typed gameplay dispatch and the engine hooks that publish system events. */
export class EventManager {
    private readonly bus = new TypedEventBus<EventMap>();
    private scene: SceneContext | null = null;
    private world: PhysicsWorld | null = null;

    public on<Name extends keyof EventMap>(name: Name, handler: (event: EventMap[Name]) => void): () => void {
        return this.bus.on(name, handler);
    }

    public emit<Name extends keyof EventMap>(name: Name, event: EventMap[Name]): void {
        this.bus.emit(name, event);
    }

    /**
     * Bind system events after all demo frame callbacks have been registered.
     * The scene executes `_beforeRender` in array order, so the manager brackets the existing
     * callbacks with frame-start and frame-end publishers.
     */
    public bindSystemEvents(scene: SceneContext, world: PhysicsWorld): void {
        if (this.scene || this.world) throw new Error("[aquanova] system events are already bound");
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
        if (this.world?._afterStep) removeCallback(this.world._afterStep, this.physicsStep);
        this.scene = null;
        this.world = null;
        this.bus.clear();
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
