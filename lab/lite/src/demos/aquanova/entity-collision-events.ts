import type { AquanovaEventManager } from "./behaviors/aquanova-event-manager.js";

export type EntityCollisionScope = "all" | "fluidSimulation";

export function registerEntityCollisionEventHandlers(
    events: AquanovaEventManager,
    setCollisionActive: (entityName: string, active: boolean, scope: EntityCollisionScope) => void
): () => void {
    return events.on("entityEvent", ({ name, event }) => {
        if (event === "disableCollision" || event === "remove") {
            setCollisionActive(name, false, "all");
        } else if (event === "disableFluidSimulationCollision") {
            setCollisionActive(name, false, "fluidSimulation");
        } else if (event === "enableCollision") {
            setCollisionActive(name, true, "all");
        }
    });
}
