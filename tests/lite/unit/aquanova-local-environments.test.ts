import { beforeEach, describe, expect, it, vi } from "vitest";

const lite = vi.hoisted(() => ({
    createPbrLocalEnvironmentBlend: vi.fn((_scene, options) => ({
        ...options,
        weight: options.weight ?? 0,
        parallaxCorrection: options.parallaxCorrection !== false,
        _bindVersion: 0,
    })),
    enablePbrLocalCubemap: vi.fn(async () => {}),
    isPbrMaterial: vi.fn((material: { kind?: string }) => material.kind === "pbr"),
    loadEnvironment: vi.fn(),
    markMaterialBindingsDirty: vi.fn((material) => {
        material._bindVersion = (material._bindVersion ?? 0) + 1;
    }),
    updatePbrLocalEnvironmentBlend: vi.fn((blend, update) => {
        const bindingsChanged = (update.primary !== undefined && update.primary !== blend.primary) || (update.secondary !== undefined && update.secondary !== blend.secondary);
        Object.assign(blend, update);
        if (bindingsChanged) blend._bindVersion++;
        return bindingsChanged;
    }),
}));

vi.mock("babylon-lite", () => lite);
vi.mock("../../../packages/babylon-lite/src/index", () => lite);

import { applyLocalEnvironmentProbes } from "../../../lab/lite/src/demos/aquanova/local-environments";

function twoProbeIndex(): Record<string, unknown> {
    return {
        probes: {
            A: {
                url: "environments/A.env",
                position: [2, 0, 0],
                boxPosition: [2, 0, 0],
                boxSize: [6, 6, 6],
                influenceBoxPosition: [2, 0, 0],
                influenceBoxSize: [8, 8, 8],
                influenceInnerBoxSize: [2, 2, 2],
                resolution: 256,
                bytes: 1200,
            },
            B: {
                url: "environments/B.env",
                position: [-2, 0, 0],
                boxPosition: [-2, 0, 0],
                boxSize: [10, 10, 10],
                influenceBoxPosition: [-2, 0, 0],
                influenceBoxSize: [8, 8, 8],
                influenceInnerBoxSize: [2, 2, 2],
                resolution: 128,
                bytes: 800,
            },
        },
    };
}

describe("Aquanova local environment probes", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        vi.stubGlobal("location", new URL("http://localhost/lite/demo-aquanova.html"));
        vi.stubGlobal(
            "fetch",
            vi.fn(async () => ({
                ok: true,
                json: async () => twoProbeIndex(),
            }))
        );
        lite.loadEnvironment.mockImplementation(async (loadedScene, url) => {
            loadedScene._envTextures = { name: "temporary" };
            loadedScene.imageProcessing.exposure = 99;
            return { url };
        });
    });

    it("loads probes, restores scene state, and assigns one shared blend to every PBR mesh", async () => {
        const globalEnvironment = { name: "global" };
        const scene = {
            _envTextures: globalEnvironment,
            imageProcessing: { exposure: 0.8, contrast: 1.2 },
        };
        const shared = { kind: "pbr", roughness: 0.4, _renderFeatures: 17 };
        const other = { kind: "pbr", metallic: 0.7 };
        const first = { material: shared };
        const second = { material: shared };
        const third = { material: other };
        const standard = { material: { kind: "standard" } };

        const controller = await applyLocalEnvironmentProbes(scene as never, [first, second, third, standard] as never, {
            blendingEnabled: false,
        });

        expect(lite.enablePbrLocalCubemap).toHaveBeenCalledOnce();
        expect(lite.loadEnvironment).toHaveBeenCalledTimes(2);
        expect(lite.loadEnvironment.mock.calls[0]![1]).toBe("http://localhost/aquanova/environments/A.env");
        expect(scene._envTextures).toBe(globalEnvironment);
        expect(scene.imageProcessing).toEqual({ exposure: 0.8, contrast: 1.2 });
        expect(first.material).toBe(second.material);
        expect(first.material).not.toBe(shared);
        expect(third.material).not.toBe(other);
        expect(standard.material).toEqual({ kind: "standard" });
        expect(first.material).not.toHaveProperty("_renderFeatures");
        expect(first.material).toMatchObject({
            roughness: 0.4,
            localEnvironmentBlend: {
                primary: {
                    boundingBoxPosition: [-2, 0, 0],
                    boundingBoxSize: [6, 6, 6],
                },
                secondary: {
                    boundingBoxPosition: [-2, 0, 0],
                    boundingBoxSize: [6, 6, 6],
                },
                weight: 0,
                parallaxCorrection: true,
            },
        });
        expect(controller).toMatchObject({ loaded: 2, assigned: 3, bytes: 2000, missing: [] });
        expect(controller?.environment("B")).toMatchObject({ boundingBoxPosition: [2, 0, 0], boundingBoxSize: [10, 10, 10] });
        expect(controller?.probeVolumes()).toEqual([
            {
                id: "A",
                capturePosition: [-2, 0, 0],
                projectionCentre: [-2, 0, 0],
                projectionHalfSize: [3, 3, 3],
                centre: [-2, 0, 0],
                innerHalfSize: [1, 1, 1],
                outerHalfSize: [4, 4, 4],
                environment: expect.anything(),
            },
            {
                id: "B",
                capturePosition: [2, 0, 0],
                projectionCentre: [2, 0, 0],
                projectionHalfSize: [5, 5, 5],
                centre: [2, 0, 0],
                innerHalfSize: [1, 1, 1],
                outerHalfSize: [4, 4, 4],
                environment: expect.anything(),
            },
        ]);
        expect(controller?.environment(undefined)).toBeUndefined();
    });

    it("updates a stable two-probe pair, rebuilds only for texture changes, and exposes the dominant probe", async () => {
        const scene = { _envTextures: null, imageProcessing: {} };
        interface TestMaterial {
            kind: string;
            localEnvironmentBlend?: { weight: number; parallaxCorrection: boolean; _bindVersion: number };
        }
        const material: TestMaterial = { kind: "pbr" };
        const mesh = { material };
        const controller = await applyLocalEnvironmentProbes(scene as never, [mesh] as never);

        const centre = controller?.updatePoi([0, 0, 0]);
        expect(centre?.probes.map((probe) => probe.id)).toEqual(["A", "B"]);
        expect(centre?.probes[0]?.weight).toBeCloseTo(0.5);
        expect(centre?.probes[1]?.weight).toBeCloseTo(0.5);
        expect(mesh.material.localEnvironmentBlend?.weight).toBeCloseTo(0.5);
        expect(lite.updatePbrLocalEnvironmentBlend).toHaveBeenCalledTimes(1);
        expect(mesh.material.localEnvironmentBlend?._bindVersion).toBe(1);

        const nearerB = controller?.updatePoi([0.5, 0, 0]);
        expect(nearerB?.probes.map((probe) => probe.id)).toEqual(["A", "B"]);
        expect(nearerB?.dominantProbeId).toBe("B");
        expect(lite.updatePbrLocalEnvironmentBlend).toHaveBeenCalledTimes(2);
        expect(mesh.material.localEnvironmentBlend?._bindVersion).toBe(1);
        expect(controller?.dominantEnvironment()).toBe(controller?.environment("B"));

        const outside = controller?.updatePoi([40, 0, 0]);
        expect(outside?.probes).toHaveLength(1);
        expect(outside?.dominantProbeId).toBe("B");
        expect(mesh.material.localEnvironmentBlend?.weight).toBe(0);
        expect(lite.updatePbrLocalEnvironmentBlend).toHaveBeenCalledTimes(3);
        expect(mesh.material.localEnvironmentBlend?._bindVersion).toBe(2);
    });

    it("switches to immutable per-mesh box-projected probes in fallback mode", async () => {
        const scene = { _envTextures: null, imageProcessing: {} };
        const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const shared = { kind: "pbr" };
        const meshA: {
            material: {
                kind: string;
                localEnvironmentBlend?: {
                    primary: { url: string };
                    secondary: { url: string };
                    weight: number;
                    parallaxCorrection: boolean;
                    _bindVersion: number;
                };
            };
            boundMin: number[];
            boundMax: number[];
            worldMatrix: number[];
        } = {
            material: shared,
            boundMin: [-0.5, -0.5, -0.5],
            boundMax: [0.5, 0.5, 0.5],
            worldMatrix: [...identity.slice(0, 12), -2, 0, 0, 1],
        };
        const meshB: typeof meshA = {
            material: shared,
            boundMin: [-0.5, -0.5, -0.5],
            boundMax: [0.5, 0.5, 0.5],
            worldMatrix: [...identity.slice(0, 12), 2, 0, 0, 1],
        };
        const weapon: typeof meshA = {
            material: { kind: "pbr" },
            boundMin: [-0.5, -0.5, -0.5],
            boundMax: [0.5, 0.5, 0.5],
            worldMatrix: identity,
        };
        const controller = await applyLocalEnvironmentProbes(scene as never, [meshA, meshB, weapon] as never, {
            staticElements: [[meshA], [meshB]] as never,
            poiMeshes: [weapon] as never,
        });

        expect(controller?.updatePoi([0.5, 0, 0]).dominantProbeId).toBe("B");
        vi.clearAllMocks();

        expect(controller?.blendingEnabled()).toBe(true);
        meshA.material = { ...meshA.material };
        controller?.setBlendingEnabled(false);
        expect(controller?.blendingEnabled()).toBe(false);
        expect(controller?.blendInfo().probes).toEqual([{ id: "A", weight: 1, ndf: expect.any(Number) }]);
        expect(meshA.material.localEnvironmentBlend).toMatchObject({
            primary: { url: "http://localhost/aquanova/environments/A.env" },
            secondary: { url: "http://localhost/aquanova/environments/A.env" },
            weight: 0,
            parallaxCorrection: true,
        });
        expect(meshB.material.localEnvironmentBlend).toMatchObject({
            primary: { url: "http://localhost/aquanova/environments/B.env" },
            secondary: { url: "http://localhost/aquanova/environments/B.env" },
            weight: 0,
            parallaxCorrection: true,
        });
        expect(meshA.material.localEnvironmentBlend).not.toBe(meshB.material.localEnvironmentBlend);
        expect(lite.updatePbrLocalEnvironmentBlend).not.toHaveBeenCalled();
        expect(weapon.material.localEnvironmentBlend).toBe(meshA.material.localEnvironmentBlend);
        expect(lite.markMaterialBindingsDirty).toHaveBeenCalledTimes(3);

        controller?.setBlendingEnabled(false);
        expect(lite.markMaterialBindingsDirty).toHaveBeenCalledTimes(3);

        controller?.updatePoi([5.5, 0, 0]);
        expect(controller?.blendInfo().probes).toEqual([{ id: "B", weight: 1, ndf: expect.any(Number) }]);
        expect(meshA.material.localEnvironmentBlend?.primary.url).toBe("http://localhost/aquanova/environments/A.env");
        expect(meshB.material.localEnvironmentBlend?.primary.url).toBe("http://localhost/aquanova/environments/B.env");
        expect(weapon.material.localEnvironmentBlend?.primary.url).toBe("http://localhost/aquanova/environments/B.env");
        expect(lite.updatePbrLocalEnvironmentBlend).not.toHaveBeenCalled();

        controller?.setBlendingEnabled(true);
        expect(meshA.material.localEnvironmentBlend).toBe(meshB.material.localEnvironmentBlend);
        expect(meshA.material.localEnvironmentBlend?.primary.url).toBe("http://localhost/aquanova/environments/B.env");
    });

    it("accepts a legacy chunk-keyed runtime index", async () => {
        vi.mocked(fetch).mockResolvedValueOnce({
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
        } as Response);
        const scene = { _envTextures: null, imageProcessing: {} };

        const controller = await applyLocalEnvironmentProbes(scene as never, []);

        expect(controller?.updatePoi([0, 0, 0]).dominantProbeId).toBe("CH00");
        expect(controller?.probeVolumes()[0]).toMatchObject({
            innerHalfSize: [2.5, 1, 1.5],
            outerHalfSize: [5.5, 4, 4.5],
        });
        expect(controller?.probeVolumes()[0]?.centre[0]).toBeCloseTo(0);
        expect(controller?.probeVolumes()[0]?.centre[1]).toBeCloseTo(0);
        expect(controller?.probeVolumes()[0]?.centre[2]).toBeCloseTo(0);
        expect(controller).toMatchObject({ loaded: 1, assigned: 0, bytes: 1200, missing: [] });
    });

    it("treats an unpublished generated index as non-fatal", async () => {
        vi.mocked(fetch).mockResolvedValueOnce({ ok: false, status: 404 } as Response);
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

        const result = await applyLocalEnvironmentProbes({ imageProcessing: {}, _envTextures: {} } as never, []);

        expect(result).toBeNull();
        expect(lite.enablePbrLocalCubemap).not.toHaveBeenCalled();
        expect(warn).toHaveBeenCalledOnce();
        warn.mockRestore();
    });

    it("keeps other ship rendering available when every generated environment is missing", async () => {
        const globalEnvironment = { name: "global" };
        const scene = {
            _envTextures: globalEnvironment,
            imageProcessing: { exposure: 0.8 },
        };
        lite.loadEnvironment.mockImplementation(async (loadedScene) => {
            loadedScene._envTextures = { name: "partial" };
            throw new Error("missing");
        });
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

        const controller = await applyLocalEnvironmentProbes(scene as never, []);

        expect(scene._envTextures).toBe(globalEnvironment);
        expect(controller).toMatchObject({ loaded: 0, assigned: 0, bytes: 2000, missing: ["A", "B"] });
        expect(controller?.updatePoi([0, 0, 0]).probes).toEqual([]);
        expect(warn).toHaveBeenCalledTimes(2);
        warn.mockRestore();
    });
});
