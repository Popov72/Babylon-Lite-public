export type EventMap = object;
export type EventHandler<Events extends EventMap, Name extends keyof Events> = (event: Events[Name]) => void;

export class TypedEventBus<Events extends EventMap> {
    private readonly handlers = new Map<keyof Events, Set<(event: Events[keyof Events]) => void>>();

    public on<Name extends keyof Events>(name: Name, handler: EventHandler<Events, Name>): () => void {
        let handlers = this.handlers.get(name);
        if (!handlers) {
            handlers = new Set();
            this.handlers.set(name, handlers);
        }
        handlers.add(handler as (event: Events[keyof Events]) => void);
        return () => {
            handlers?.delete(handler as (event: Events[keyof Events]) => void);
            if (handlers?.size === 0 && this.handlers.get(name) === handlers) {
                this.handlers.delete(name);
            }
        };
    }

    public emit<Name extends keyof Events>(name: Name, event: Events[Name]): void {
        const handlers = this.handlers.get(name);
        if (!handlers) {
            return;
        }
        for (const handler of [...handlers]) {
            handler(event);
        }
    }

    public clear(): void {
        this.handlers.clear();
    }
}
