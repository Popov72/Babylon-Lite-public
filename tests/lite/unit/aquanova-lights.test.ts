import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Mesh, SceneContext, SceneNode } from "../../../packages/babylon-lite/src";
import { buildRuntimeLights } from "../../../lab/lite/src/demos/aquanova/lights";

const runtime = vi.hoisted(() => ({
    addClusteredLightContainer: vi.fn(),
    addToScene: vi.fn(),
    createClusteredLightContainer: vi.fn(),
    createClusteredPointLight: vi.fn(),
    createClusteredSpotLight: vi.fn(),
    createDirectionalLight: vi.fn(),
    createPointLight: vi.fn(),
    createSpotLight: vi.fn(),
    markClusteredLightContainerDirty: vi.fn(),
    setMaxLights: vi.fn(),
    MAX_LIGHTS: 16,
}));

vi.mock("../../../packages/babylon-lite/src/index.ts", () => runtime);

const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function mesh(name: string): Mesh {
    return { name, material: {}, children: [] } as unknown as Mesh;
}

function scene(meshes: Mesh[]): SceneContext {
    return { meshes } as unknown as SceneContext;
}

function shipRoot(): SceneNode {
    const light = {
        children: [],
        worldMatrix: IDENTITY,
        metadata: {
            gltf: {
                extras: {
                    id: "storage-light",
                    kind: "light",
                    owner: "storage",
                    chunk: "CH00_Storage",
                    runtime: {
                        type: "point",
                        clustered: true,
                        color: [1, 0.8, 0.6],
                        intensity: 3,
                        range: 5,
                        angle: 90,
                    },
                },
            },
        },
    };
    return { children: [light] } as unknown as SceneNode;
}

describe("Aquanova runtime lights", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        runtime.createClusteredLightContainer.mockReturnValue({
            kind: "clusteredLightContainer",
            pointLights: [],
            spotLights: [],
            horizontalTiles: 64,
            verticalTiles: 64,
            zSlices: 16,
            _version: 0,
        });
        runtime.createClusteredPointLight.mockImplementation((container: { pointLights: unknown[] }, options: unknown) => {
            container.pointLights.push(options);
            return options;
        });
    });

    it("attaches one shared clustered container to the ship and weapon scenes", () => {
        const shipMesh = mesh("ship");
        const shipScene = scene([shipMesh]);
        const stats = buildRuntimeLights(shipScene, shipRoot(), [shipMesh], new Map([[shipMesh, "CH00_Storage"]]));
        const weaponScene = scene([mesh("weapon")]);

        stats.attachClusteredScene(weaponScene);
        stats.attachClusteredScene(weaponScene);

        const container = runtime.createClusteredLightContainer.mock.results[0]!.value;
        expect(runtime.addClusteredLightContainer.mock.calls).toEqual([
            [shipScene, container],
            [weaponScene, container],
        ]);
    });
});
