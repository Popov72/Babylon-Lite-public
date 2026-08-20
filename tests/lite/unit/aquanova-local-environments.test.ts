import { beforeEach, describe, expect, it, vi } from "vitest";

const lite = vi.hoisted(() => ({
    MAX_PBR_LOCAL_ENVIRONMENT_CANDIDATES: 4,
    createPbrLocalEnvironmentProbeSet: vi.fn((_scene, options) => ({ ...options })),
    enablePbrLocalCubemap: vi.fn(async () => {}),
    getPbrLocalEnvironmentProbeGridCell: vi.fn((set: { probes: readonly unknown[] }, position: readonly number[]) => ({
        coordinates: [0, 0, 0],
        probeIndices: position[0]! > 3 && set.probes.length > 1 ? [1] : set.probes.map((_probe, index) => index),
        outside: false,
    })),
    isPbrMaterial: vi.fn((material: { kind?: string }) => material.kind === "pbr"),
    loadEnvironment: vi.fn(),
    setPbrLocalEnvironment: vi.fn((material, environment, options) => {
        material._testLocalEnvironment = { environment, options };
        delete material._testLocalEnvironmentProbeSet;
    }),
    setPbrLocalEnvironmentProbeDebug: vi.fn(),
    setPbrLocalEnvironmentProbeSet: vi.fn((material, set) => {
        material._testLocalEnvironmentProbeSet = set;
        delete material._testLocalEnvironment;
    }),
}));

vi.mock("babylon-lite", () => lite);
vi.mock("../../../packages/babylon-lite/src/index", () => lite);

import { applyLocalEnvironmentProbes } from "../../../lab/lite/src/demos/aquanova/local-environments";

function worldMatrixAt(x = 0, y = 0, z = 0): number[] {
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
}

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
                angle: 30,
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

    it("loads all probes into one shared array set, restores scene state, and assigns PBR materials", async () => {
        interface TestMaterial {
            kind: string;
            roughness?: number;
            metallic?: number;
            _renderFeatures?: number;
            _testLocalEnvironmentProbeSet?: unknown;
        }
        const globalEnvironment = { name: "global" };
        const scene = {
            _envTextures: globalEnvironment,
            imageProcessing: { exposure: 0.8, contrast: 1.2 },
        };
        const shared: TestMaterial = { kind: "pbr", roughness: 0.4, _renderFeatures: 17 };
        const other: TestMaterial = { kind: "pbr", metallic: 0.7 };
        const first = { material: shared, worldMatrix: worldMatrixAt() };
        const second = { material: shared, worldMatrix: worldMatrixAt() };
        const third = { material: other, worldMatrix: worldMatrixAt() };
        const standard = { material: { kind: "standard" }, worldMatrix: worldMatrixAt() };

        const controller = await applyLocalEnvironmentProbes(scene as never, [first, second, third, standard] as never);

        expect(lite.enablePbrLocalCubemap).toHaveBeenCalledWith();
        expect(lite.loadEnvironment).toHaveBeenCalledTimes(2);
        expect(lite.loadEnvironment.mock.calls[0]![1]).toBe("http://localhost/aquanova/environments/A.env");
        expect(scene._envTextures).toBe(globalEnvironment);
        expect(scene.imageProcessing).toEqual({ exposure: 0.8, contrast: 1.2 });
        expect(first.material).toBe(second.material);
        expect(first.material).not.toBe(shared);
        expect(third.material).not.toBe(other);
        expect(standard.material).toEqual({ kind: "standard" });
        expect(first.material).not.toHaveProperty("_renderFeatures");
        expect(first.material).toHaveProperty("_testLocalEnvironmentProbeSet");
        expect(first.material._testLocalEnvironmentProbeSet).toBe(third.material._testLocalEnvironmentProbeSet);
        expect(lite.createPbrLocalEnvironmentProbeSet).toHaveBeenCalledWith(
            scene,
            expect.objectContaining({
                probes: [
                    expect.objectContaining({
                        capturePosition: [-2, 0, 0],
                        projectionPosition: [-2, 0, 0],
                        projectionSize: [6, 6, 6],
                        influencePosition: [-2, 0, 0],
                        influenceInnerSize: [2, 2, 2],
                        influenceOuterSize: [8, 8, 8],
                        angleRadians: -Math.PI / 6,
                        debugColor: expect.any(Array),
                    }),
                    expect.objectContaining({
                        capturePosition: [2, 0, 0],
                        angleRadians: 0,
                    }),
                ],
                voxelGrid: {
                    minimum: [-10, -6, -8],
                    maximum: [8, 6, 8],
                    cellSize: 2,
                },
            })
        );
        expect(controller).toMatchObject({ loaded: 2, assigned: 3, bytes: 2000, missing: [] });
        expect(controller?.probeVolumes()[0]).toMatchObject({
            id: "A",
            capturePosition: [-2, 0, 0],
            projectionCentre: [-2, 0, 0],
            projectionHalfSize: [3, 3, 3],
            centre: [-2, 0, 0],
            innerHalfSize: [1, 1, 1],
            outerHalfSize: [4, 4, 4],
            angleRadians: -Math.PI / 6,
        });
    });

    it("reports the camera voxel probe set while the POI dominant probe changes", async () => {
        const scene = { _envTextures: null, imageProcessing: {} };
        const mesh: { material: { kind: string; _testLocalEnvironmentProbeSet?: unknown }; worldMatrix: number[] } = {
            material: { kind: "pbr" },
            worldMatrix: worldMatrixAt(),
        };
        const controller = await applyLocalEnvironmentProbes(scene as never, [mesh] as never);
        const probeSet = lite.createPbrLocalEnvironmentProbeSet.mock.results[0]!.value;

        const centre = controller?.updatePoi([0, 0, 0]);
        expect(centre?.probes.map((probe) => probe.id)).toEqual(["A", "B"]);
        expect(centre?.dominantProbeId).toBe("A");
        expect(centre?.cameraVoxelProbeIds).toEqual(["A", "B"]);

        controller?.updatePoi([0, 0, 0]);

        const insideB = controller?.updatePoi([5.5, 0, 0]);
        expect(insideB?.dominantProbeId).toBe("B");
        expect(insideB?.cameraVoxelProbeIds).toEqual(["B"]);
        expect(controller?.dominantEnvironment()).toBe(controller?.environment("B"));
        expect(mesh.material._testLocalEnvironmentProbeSet).toBe(probeSet);
    });

    it("uses immutable intersecting single-probe materials while fragment blending is disabled", async () => {
        const scene = { _envTextures: null, imageProcessing: {} };
        interface TestMaterial {
            kind: string;
            _testLocalEnvironment?: { environment: unknown; options: unknown };
            _testLocalEnvironmentProbeSet?: unknown;
            plugins?: unknown[];
        }
        const source: TestMaterial = { kind: "pbr" };
        const meshA = { material: source, worldMatrix: worldMatrixAt(-2), boundMin: [-0.25, -0.25, -0.25], boundMax: [0.25, 0.25, 0.25] };
        const meshB = { material: source, worldMatrix: worldMatrixAt(5.5), boundMin: [-0.25, -0.25, -0.25], boundMax: [0.25, 0.25, 0.25] };
        const controller = await applyLocalEnvironmentProbes(scene as never, [meshA, meshB] as never, { blendingEnabled: false });
        const probeSet = lite.createPbrLocalEnvironmentProbeSet.mock.results[0]!.value;

        expect(controller?.blendingEnabled()).toBe(false);
        expect(meshA.material).not.toBe(meshB.material);
        expect(meshA.material._testLocalEnvironment).toMatchObject({
            environment: controller?.environment("A"),
            options: { projectionPosition: [-2, 0, 0], projectionSize: [6, 6, 6] },
        });
        expect(meshB.material._testLocalEnvironment).toMatchObject({
            environment: controller?.environment("B"),
            options: { projectionPosition: [2, 0, 0], projectionSize: [10, 10, 10] },
        });
        expect(controller?.updatePoi([5.5, 0, 0])).toMatchObject({
            dominantProbeId: "B",
            probes: [{ id: "B", weight: 1 }],
        });

        const plugins = [{ name: "liquefy" }];
        meshA.material = { ...meshA.material, plugins };
        controller?.setBlendingEnabled(true);
        expect(controller?.blendingEnabled()).toBe(true);
        expect(meshA.material._testLocalEnvironmentProbeSet).toBe(probeSet);
        expect(meshB.material._testLocalEnvironmentProbeSet).toBe(probeSet);
        expect(meshA.material._testLocalEnvironment).toBeUndefined();
        expect(meshB.material._testLocalEnvironment).toBeUndefined();
        expect(meshA.material.plugins).toBe(plugins);
    });

    it("enables blended probe colors only while blending is active", async () => {
        const scene = { _envTextures: null, imageProcessing: {} };
        const mesh = { material: { kind: "pbr" }, worldMatrix: worldMatrixAt() };
        const controller = await applyLocalEnvironmentProbes(scene as never, [mesh] as never, { blendingEnabled: false });
        const probeSet = lite.createPbrLocalEnvironmentProbeSet.mock.results[0]!.value;

        controller?.setDebugEnabled(true);
        expect(controller?.debugEnabled()).toBe(true);
        expect(lite.setPbrLocalEnvironmentProbeDebug).toHaveBeenLastCalledWith(probeSet, false);

        controller?.setBlendingEnabled(true);
        expect(lite.setPbrLocalEnvironmentProbeDebug).toHaveBeenLastCalledWith(probeSet, true);

        controller?.setBlendingEnabled(false);
        expect(lite.setPbrLocalEnvironmentProbeDebug).toHaveBeenLastCalledWith(probeSet, false);
    });

    it("accepts a legacy chunk-keyed runtime index and defaults missing angles to zero", async () => {
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
            angleRadians: 0,
        });
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
        expect(lite.enablePbrLocalCubemap).not.toHaveBeenCalled();
        expect(warn).toHaveBeenCalledTimes(2);
        warn.mockRestore();
    });
});
