import { removeFromScene, setMeshVisible, type Mesh, type SceneContext } from "babylon-lite";
import type { AquanovaEventManager } from "./behaviors/aquanova-event-manager.js";

export interface MeshEntityEventOperations {
    readonly remove: (scene: SceneContext, mesh: Mesh) => void;
    readonly setVisible: (mesh: Mesh, visible: boolean) => void;
}

const DEFAULT_OPERATIONS: MeshEntityEventOperations = {
    remove: removeFromScene,
    setVisible: setMeshVisible,
};

export function createPersistentMeshEntityEventOperations(hiddenMeshes: Set<Mesh>, operations: MeshEntityEventOperations = DEFAULT_OPERATIONS): MeshEntityEventOperations {
    return {
        remove(scene, mesh) {
            hiddenMeshes.add(mesh);
            operations.remove(scene, mesh);
        },
        setVisible(mesh, visible) {
            if (visible) hiddenMeshes.delete(mesh);
            else hiddenMeshes.add(mesh);
            operations.setVisible(mesh, visible);
        },
    };
}

export function registerMeshEntityEventHandlers(
    events: AquanovaEventManager,
    scene: SceneContext,
    meshesByEntityName: ReadonlyMap<string, readonly Mesh[]>,
    operations: MeshEntityEventOperations = DEFAULT_OPERATIONS
): () => void {
    return events.on("entityEvent", ({ name, event }) => {
        if (event !== "hide" && event !== "remove" && event !== "show") {
            return;
        }
        for (const mesh of meshesByEntityName.get(name) ?? []) {
            if (event === "remove") {
                operations.remove(scene, mesh);
            } else {
                operations.setVisible(mesh, event === "show");
            }
        }
    });
}
