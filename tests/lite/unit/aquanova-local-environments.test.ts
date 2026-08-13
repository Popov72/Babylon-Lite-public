import { beforeEach, describe, expect, it, vi } from "vitest";

const lite = vi.hoisted(() => ({
    enablePbrLocalCubemap: vi.fn(async () => {}),
    isPbrMaterial: vi.fn((material: { kind?: string }) => material.kind === "pbr"),
    loadEnvironment: vi.fn(),
    markMaterialUboDirty: vi.fn(),
}));

vi.mock("babylon-lite", () => lite);
vi.mock("../../../packages/babylon-lite/src/index", () => lite);

import { applyLocalEnvironmentProbes } from "../../../lab/lite/src/demos/aquanova/local-environments";

const worldAt = (x: number, y: number, z: number): number[] => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];

describe("Aquanova local environment probes", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        vi.stubGlobal("location", new URL("http://localhost/lite/demo-aquanova.html"));
    });

    it("assigns static elements spatially and chooses the smallest containing probe", async () => {
        const globalEnvironment = { name: "global" };
        const scene = {
            _envTextures: globalEnvironment,
            imageProcessing: { exposure: 0.8, contrast: 1.2 },
        };
        interface TestMaterial {
            kind: string;
            roughness?: number;
            _renderFeatures?: number;
            environmentIntensity?: number;
        }
        const material: TestMaterial = { kind: "pbr", roughness: 0.4, _renderFeatures: 17 };
        const largeStatic = { material, boundMin: [-0.5, -0.5, -0.5], boundMax: [0.5, 0.5, 0.5], worldMatrix: worldAt(-4, 3, 4) };
        const smallStatic = { material, boundMin: [-0.5, -0.5, -0.5], boundMax: [0.5, 0.5, 0.5], worldMatrix: worldAt(-2, 3, 4) };
        const dynamic = { material, boundMin: [-0.5, -0.5, -0.5], boundMax: [0.5, 0.5, 0.5], worldMatrix: worldAt(5, 2, 1) };
        const standard = {
            material: { kind: "standard" },
            boundMin: [-0.5, -0.5, -0.5],
            boundMax: [0.5, 0.5, 0.5],
            worldMatrix: worldAt(-2, 3, 4),
        };
        const meshes = [largeStatic, smallStatic, dynamic, standard];
        vi.stubGlobal(
            "fetch",
            vi.fn(async () => ({
                ok: true,
                json: async () => ({
                    probes: {
                        ENV_LARGE: {
                            url: "environments/ENV_LARGE.env",
                            position: [2, 3, 4],
                            boxPosition: [2, 3, 4],
                            boxSize: [8, 5, 6],
                            resolution: 256,
                            bytes: 1200,
                        },
                        ENV_SMALL: {
                            url: "environments/ENV_SMALL.env",
                            position: [2, 3, 4],
                            boxPosition: [2, 3, 4],
                            boxSize: [2, 2, 2],
                            resolution: 128,
                            bytes: 800,
                        },
                    },
                }),
            }))
        );
        lite.loadEnvironment.mockImplementation(async (loadedScene, url) => {
            loadedScene._envTextures = { name: "temporary" };
            loadedScene.imageProcessing.exposure = 99;
            return { url };
        });

        const controller = await applyLocalEnvironmentProbes(scene as never, meshes as never, [[largeStatic], [smallStatic], [standard]] as never);

        expect(lite.enablePbrLocalCubemap).toHaveBeenCalledOnce();
        expect(lite.loadEnvironment).toHaveBeenCalledTimes(2);
        expect(lite.loadEnvironment.mock.calls[0]![1]).toBe("http://localhost/aquanova/environments/ENV_LARGE.env");
        expect(scene._envTextures).toBe(globalEnvironment);
        expect(scene.imageProcessing).toEqual({ exposure: 0.8, contrast: 1.2 });
        expect(largeStatic.material).not.toBe(material);
        expect(smallStatic.material).not.toBe(material);
        expect(smallStatic.material).not.toBe(largeStatic.material);
        expect(dynamic.material).toBe(material);
        expect(largeStatic.material).toMatchObject({
            roughness: 0.4,
            localEnvironment: {
                boundingBoxPosition: [-2, 3, 4],
                boundingBoxSize: [8, 5, 6],
            },
        });
        expect(smallStatic.material).toMatchObject({
            localEnvironment: {
                boundingBoxPosition: [-2, 3, 4],
                boundingBoxSize: [2, 2, 2],
            },
        });
        expect(largeStatic.material).not.toHaveProperty("_renderFeatures");
        expect(controller).toMatchObject({ loaded: 2, assigned: 2, bytes: 2000, missing: [] });
        expect(controller?.probeAt([-2, 3, 4])).toBe("ENV_SMALL");
        expect(controller?.probeAt([-4, 3, 4])).toBe("ENV_LARGE");
        expect(controller?.probeAt([20, 3, 4])).toBeUndefined();
        expect(controller?.environment("ENV_SMALL")).toBe((smallStatic.material as TestMaterial & { localEnvironment?: unknown }).localEnvironment);
        expect(controller?.environment(undefined)).toBeUndefined();

        expect(controller?.update([dynamic] as never, controller.probeAt([-2, 3, 4]))).toBe(1);
        expect(dynamic.material).toBe(smallStatic.material);
        dynamic.material.environmentIntensity = 1.75;
        expect(controller?.update([dynamic] as never, "ENV_LARGE")).toBe(1);
        expect(dynamic.material.environmentIntensity).toBe(1.75);
        expect(lite.markMaterialUboDirty).toHaveBeenCalled();
        expect(controller?.update([dynamic] as never, undefined)).toBe(1);
        expect(dynamic.material).toBe(material);
    });

    it("resolves one probe per element from its unioned bounds, by intersection", async () => {
        const scene = { _envTextures: null, imageProcessing: {} };
        const material = { kind: "pbr" };
        // Two adjacent probes meeting at x = 0, as the ship's rooms and corridors do.
        //   ENV_ROOM x ∈ [-4, 0]   ENV_HALL x ∈ [0, 6]   both y ∈ [0.5, 5.5], z ∈ [2, 6]
        // The element is a wall face well inside the room plus a zero-thickness trim band that pokes
        // 20 mm past the seam. Resolved per primitive the band would be handed to the hall, and under
        // the old containment test the element would have got no probe at all.
        const face = { material, boundMin: [-0.5, -0.5, -0.5], boundMax: [0.5, 0.5, 0.5], worldMatrix: worldAt(-1, 3, 4) };
        const trim = { material, boundMin: [1.02, 0, 0], boundMax: [1.02, 0, 0], worldMatrix: worldAt(-1, 3, 4) };
        const outside = { material, boundMin: [0, 0, 0], boundMax: [0, 0, 0], worldMatrix: worldAt(-1, 3, 40) };
        const meshes = [face, trim, outside];
        vi.stubGlobal(
            "fetch",
            vi.fn(async () => ({
                ok: true,
                json: async () => ({
                    probes: {
                        ENV_ROOM: { url: "environments/ENV_ROOM.env", position: [2, 3, 4], boxPosition: [2, 3, 4], boxSize: [4, 5, 4], resolution: 256, bytes: 10 },
                        ENV_HALL: { url: "environments/ENV_HALL.env", position: [-3, 3, 4], boxPosition: [-3, 3, 4], boxSize: [6, 5, 4], resolution: 256, bytes: 10 },
                    },
                }),
            }))
        );
        lite.loadEnvironment.mockImplementation(async (_scene: unknown, url: string) => ({ url }));

        const controller = await applyLocalEnvironmentProbes(scene as never, meshes as never, [[face, trim], [outside]] as never);

        const probeOf = (mesh: { material: unknown }): unknown => (mesh.material as { localEnvironment?: { url?: string } }).localEnvironment?.url;
        expect(probeOf(face)).toBe("http://localhost/aquanova/environments/ENV_ROOM.env");
        expect(probeOf(trim)).toBe(probeOf(face));
        expect(outside.material).toBe(material);
        expect(controller).toMatchObject({ assigned: 2 });

        // Straddling the seam evenly is a tie on share, so the tighter box wins it.
        expect(controller?.probeForBounds([-0.6, 2.5, 3.5], [0.6, 3.5, 4.5])).toBe("ENV_ROOM");
        expect(controller?.probeForBounds([1, 2.5, 3.5], [2, 3.5, 4.5])).toBe("ENV_HALL");
        expect(controller?.probeForBounds([-0.6, 2.5, 20], [0.6, 3.5, 21])).toBeUndefined();
    });

    it("accepts a legacy chunk-keyed runtime index", async () => {
        const globalEnvironment = { name: "global" };
        const scene = { _envTextures: globalEnvironment, imageProcessing: {} };
        vi.stubGlobal(
            "fetch",
            vi.fn(async () => ({
                ok: true,
                json: async () => ({
                    chunks: {
                        CH00: {
                            url: "environments/CH00.env",
                            position: [0, 0, 0],
                            boxPosition: [0, 0, 0],
                            boxSize: [8, 5, 6],
                            resolution: 256,
                            bytes: 1200,
                        },
                    },
                }),
            }))
        );
        lite.loadEnvironment.mockResolvedValue({});

        const controller = await applyLocalEnvironmentProbes(scene as never, [], []);

        expect(controller?.probeAt([0, 0, 0])).toBe("CH00");
        expect(controller).toMatchObject({ loaded: 1, assigned: 0, bytes: 1200, missing: [] });
    });

    it("treats an unpublished generated index as non-fatal", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(async () => ({ ok: false, status: 404 }))
        );
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

        const result = await applyLocalEnvironmentProbes({ imageProcessing: {}, _envTextures: {} } as never, [], []);

        expect(result).toBeNull();
        expect(lite.enablePbrLocalCubemap).not.toHaveBeenCalled();
        expect(warn).toHaveBeenCalledOnce();
        warn.mockRestore();
    });

    it("keeps other ship rendering available when one generated environment is missing", async () => {
        const globalEnvironment = { name: "global" };
        const scene = {
            _envTextures: globalEnvironment,
            imageProcessing: { exposure: 0.8 },
        };
        vi.stubGlobal(
            "fetch",
            vi.fn(async () => ({
                ok: true,
                json: async () => ({
                    probes: {
                        ENV00: {
                            url: "environments/ENV00.env",
                            position: [0, 0, 0],
                            boxPosition: [0, 0, 0],
                            boxSize: [8, 5, 6],
                            resolution: 256,
                            bytes: 1200,
                        },
                    },
                }),
            }))
        );
        lite.loadEnvironment.mockImplementation(async (loadedScene) => {
            loadedScene._envTextures = { name: "partial" };
            throw new Error("missing");
        });
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

        const controller = await applyLocalEnvironmentProbes(scene as never, [], []);

        expect(scene._envTextures).toBe(globalEnvironment);
        expect(controller).toMatchObject({ loaded: 0, assigned: 0, bytes: 1200, missing: ["ENV00"] });
        expect(warn).toHaveBeenCalledOnce();
        warn.mockRestore();
    });
});
