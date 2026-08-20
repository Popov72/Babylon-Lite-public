import { TypedEventBus, type EventMap } from "./event-bus.js";

export class EventManager<Events extends EventMap> {
    private readonly bus = new TypedEventBus<Events>();

    public on<Name extends keyof Events>(name: Name, handler: (event: Events[Name]) => void): () => void {
        return this.bus.on(name, handler);
    }

    public emit<Name extends keyof Events>(name: Name, event: Events[Name]): void {
        this.bus.emit(name, event);
    }

    public dispose(): void {
        this.bus.clear();
    }
}
