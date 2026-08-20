import type { AquanovaEventManager } from "./behaviors/aquanova-event-manager.js";

export function registerEntityCollisionEventHandlers(events: AquanovaEventManager, setCollisionActive: (entityName: string, active: boolean) => void): () => void {
    return events.on("entityEvent", ({ name, event }) => {
        if (event === "disableCollision" || event === "remove") {
            setCollisionActive(name, false);
        } else if (event === "enableCollision") {
            setCollisionActive(name, true);
        }
    });
}
