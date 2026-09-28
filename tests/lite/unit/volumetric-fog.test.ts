import { describe, expect, it, vi } from "vitest";
import { createFreeCamera } from "../../../packages/babylon-lite/src/camera/free-camera";
import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import { createRenderTarget } from "../../../packages/babylon-lite/src/engine/render-target";
import { rebuildRegisteredScenes } from "../../../packages/babylon-lite/src/engine/recovery-rebuild";
import type { SurfaceContext } from "../../../packages/babylon-lite/src/engine/surface";
import type { DirectionalLight } from "../../../packages/babylon-lite/src/light/directional-light";
import type { Material } from "../../../packages/babylon-lite/src/material/material";
import { makeLinearImageProcessingExt } from "../../../packages/babylon-lite/src/material/pbr/fragments/linear-image-processing-fragment";
import { createUnlitFragment } from "../../../packages/babylon-lite/src/material/pbr/fragments/unlit-fragment";
import { createPbrTemplate, type PbrTemplateConfig } from "../../../packages/babylon-lite/src/material/pbr/pbr-template";
import type { Mesh } from "../../../packages/babylon-lite/src/mesh/mesh";
import type { DrawBinding, Renderable } from "../../../packages/babylon-lite/src/render/renderable";
import { composeShader } from "../../../packages/babylon-lite/src/shader/shader-composer";
import { addToScene, createSceneContext } from "../../../packages/babylon-lite/src/scene/scene-core";
import { _t, enableSceneTransmission, executePassWithTransmission, type RenderTaskTransmissionState } from "../../../packages/babylon-lite/src/frame-graph/transmission";
import type { RenderTaskBase } from "../../../packages/babylon-lite/src/frame-graph/render-task-base";
import { beginTaskPass } from "../../../packages/babylon-lite/src/frame-graph/render-task-phase-pass";
import { disposeVolumetricFog, enableVolumetricFog } from "../../../packages/babylon-lite/src/volumetric-fog/volumetric-fog";
import type { VolumetricFogDensitySource } from "../../../packages/babylon-lite/src/volumetric-fog/volumetric-fog";

function createMockEngine(sampleCount: 1 | 4): EngineContext {
    const device = { createBuffer: vi.fn(() => ({ destroy: vi.fn() }) as unknown as GPUBuffer) } as unknown as GPUDevice;
    const scRT = createRenderTarget({ lbl: "swapchain", format: "bgra8unorm", samples: 1, size: { width: 0, height: 0 } });
    scRT._eager = true;
    const engine = {
        _device: device,
        canvas: { width: 640, height: 360 },
        format: "bgra8unorm",
        msaaSamples: sampleCount,
        scRT,
        _renderingContexts: [],
        useFloatingOrigin: false,
    } as unknown as EngineContext;
    Object.assign(engine, { engine });
    return engine;
}

function createFogScene(sampleCount: 1 | 4 = 4) {
    const engine = createMockEngine(sampleCount);
    const scene = createSceneContext(engine);
    scene.camera = createFreeCamera({ x: 0, y: 2, z: -8 }, { x: 0, y: 0, z: 0 });
    const task = scene._frameGraph._tasks[0] as RenderTaskBase;
    return { engine, scene, task };
}

function createPreparationDevice() {
    const shader = { getCompilationInfo: vi.fn(async () => ({ messages: [] })) } as unknown as GPUShaderModule;
    return {
        limits: { maxUniformBufferBindingSize: 65536 },
        pushErrorScope: vi.fn(),
        popErrorScope: vi.fn(async () => null),
        createBindGroupLayout: vi.fn(() => ({}) as GPUBindGroupLayout),
        createShaderModule: vi.fn(() => shader),
        createPipelineLayout: vi.fn(() => ({}) as GPUPipelineLayout),
        createComputePipelineAsync: vi.fn(async () => ({}) as GPUComputePipeline),
    };
}

describe("linear PBR image processing", () => {
    it("keeps an opaque unlit assignment outside the no-IBL image-processing bypass", () => {
        const linear = makeLinearImageProcessingExt();
        const features2 = linear.detect!({ _linearImageProcessing: true }).f2;
        const fragment = linear.frag!({
            _features: 0,
            _features2: features2,
            _meshFeatures: 0,
            _hasIbl: false,
            _hasAnyNormal: false,
            _hasSpecularAA: false,
        })!;
        const config: PbrTemplateConfig = {
            _normalMode: "none",
            _hasEmissiveTexture: false,
            _hasSpecGloss: false,
            _hasDoubleSided: false,
            _hasTonemap: false,
            _hasAlphaBlend: false,
            _hasSpecularAA: false,
            _hasGammaAlbedo: false,
            _hasMorph: false,
            _hasOcclusion: false,
            _hasEmissiveColor: false,
            _hasReflectanceExt: false,
            _hasIbl: false,
        };
        const source = composeShader(createPbrTemplate(config), [fragment, createUnlitFragment(false)])._fragmentWGSL;
        const unlit = source.indexOf("color = baseColor * material.unlitColor;");
        const imageStart = source.indexOf("if(scene.vImageInfos.w>=0.0){");
        const exposure = source.indexOf("color*=scene.vImageInfos.x;");
        const colorClamp = source.indexOf("color=max(color,", exposure);
        const imageEnd = source.indexOf("}", colorClamp);

        expect(unlit).toBeGreaterThanOrEqual(0);
        expect(imageStart).toBeGreaterThan(unlit);
        expect(exposure).toBeGreaterThan(imageStart);
        expect(colorClamp).toBeGreaterThan(exposure);
        expect(imageEnd).toBeGreaterThan(colorClamp);
    });
});

describe("volumetric fog installation", () => {
    it.each([1, 4] as const)("retargets the default %ix render task and preserves depth ownership", (samples) => {
        const { engine, scene, task } = createFogScene(samples);
        const original = task._config.rt;
        const originalDepth = task._config.depth;
        const fog = enableVolumetricFog(scene, engine, { density: 0.1 });

        expect(task._config.rt).not.toBe(original);
        expect(task._config.rt._descriptor.format).toBe("rgba16float");
        expect(task._config.rt._descriptor.samples).toBe(samples);
        expect(task._config.rt._descriptor.size).toBe(engine);
        expect(task._config.depth).toBe(originalDepth);
        expect(task._config.rst).toBeUndefined();
        expect(task._afterOpaque).toBe(fog._phase);
        expect(fog._densitySource).toBeNull();
        expect(scene._frameGraph._tasks).toHaveLength(2);
        expect(scene._frameGraph._tasks[1]!.name).toBe("volumetric-fog-image-processing");

        disposeVolumetricFog(fog);
        expect(task._afterOpaque).toBeUndefined();
        expect(task._executeWithTransmission).toBeDefined();
        expect(task._config.rt._descriptor.format).toBe("rgba16float");
        expect(scene._frameGraph._tasks).toHaveLength(2);
        disposeVolumetricFog(fog);
        expect(() => enableVolumetricFog(scene, engine, { density: 0.1, depthSlices: 32 })).toThrow("already installed");
    });

    it("sizes linear color from the bound auxiliary surface, not the engine's primary swapchain", () => {
        const engine = createMockEngine(1);
        const swap = createRenderTarget({ lbl: "auxiliary-swapchain", format: "bgra8unorm", samples: 1, size: { width: 0, height: 0 } });
        swap._eager = true;
        const surface = { engine, scRT: swap, msaaSamples: 1, format: "bgra8unorm", canvas: { width: 320, height: 200 }, _renderingContexts: [] } as unknown as SurfaceContext;
        const scene = createSceneContext(surface);
        scene.camera = createFreeCamera({ x: 0, y: 2, z: -8 }, { x: 0, y: 0, z: 0 });
        const task = scene._frameGraph._tasks[0] as RenderTaskBase;
        const fog = enableVolumetricFog(scene, engine, { density: 0.1 });

        expect(task._config.rt._descriptor.size).toBe(surface);
        expect(task._config.rt._descriptor.format).toBe("rgba16float");
        expect(swap._descriptor.size).toEqual({ width: 0, height: 0 });
        disposeVolumetricFog(fog);
    });

    it("marks meshes added or re-materialed after fog installation for linear PBR output", () => {
        const { engine, scene } = createFogScene();
        const fog = enableVolumetricFog(scene, engine, { density: 0.1 });
        const original = { _buildGroup: { _materialFamily: "pbr" }, _renderFeatures: { features: 0 } } as unknown as Material;
        const mesh = { _gpu: {}, material: original } as unknown as Mesh;
        addToScene(scene, mesh);
        expect((original as Material & { _linearImageProcessing?: boolean })._linearImageProcessing).toBe(true);
        expect(original._renderFeatures).toBeUndefined();

        const swapped = { _buildGroup: { _materialFamily: "pbr" }, _renderFeatures: { features: 0 } } as unknown as Material;
        mesh.material = swapped;
        expect((swapped as Material & { _linearImageProcessing?: boolean })._linearImageProcessing).toBe(true);
        expect(swapped._renderFeatures).toBeUndefined();

        disposeVolumetricFog(fog);
        const addedAfterDispose = { _buildGroup: { _materialFamily: "pbr" } } as unknown as Material;
        addToScene(scene, { _gpu: {}, material: addedAfterDispose } as unknown as Mesh);
        expect((addedAfterDispose as Material & { _linearImageProcessing?: boolean })._linearImageProcessing).toBe(true);
    });

    it("does not install a material-change hook when a transmission rebuild rolls back", () => {
        const { engine, scene } = createFogScene();
        const [, rollback] = _t(scene, engine);
        expect(scene._meshMaterialChange).toBeUndefined();
        rollback();
        expect(scene._meshMaterialChange).toBeUndefined();
    });

    it.each(["before", "after"] as const)("shares the linear target and one present when transmission is enabled %s fog", (order) => {
        const { engine, scene, task } = createFogScene();
        if (order === "before") {
            enableSceneTransmission(scene, engine);
        }
        const fog = enableVolumetricFog(scene, engine, { density: 0.1 });
        const linear = task._config.rt;
        const present = scene._frameGraph._tasks[1];
        if (order === "after") {
            enableSceneTransmission(scene, engine);
        }
        expect(task._transmissionInstalled).toBe(true);
        expect(task._config.rt).toBe(linear);
        expect(task._afterOpaque).toBe(fog._phase);
        expect(scene._frameGraph._tasks).toHaveLength(2);
        expect(scene._frameGraph._tasks[1]).toBe(present);
        disposeVolumetricFog(fog);
        expect(task._executeWithTransmission).toBeDefined();
    });

    it("rejects incompatible materials before changing the render target", () => {
        const { engine, scene, task } = createFogScene();
        const oldTarget = task._config.rt;
        const material = { _buildGroup: { _materialFamily: "standard" } } as unknown as Material;
        const mesh = { material } as Mesh;
        scene._renderables.push({ order: 100, isTransparent: false, mesh, bind: vi.fn() } as unknown as Renderable);
        expect(() => enableVolumetricFog(scene, engine, { density: 0.1 })).toThrow("opaque PBR");
        expect(task._config.rt).toBe(oldTarget);
        expect(scene._frameGraph._tasks).toHaveLength(1);
    });

    it("retains a cached linear material guard when fog is disabled or disposed", () => {
        const { engine, scene, task } = createFogScene(1);
        const builder = { _materialFamily: "pbr" };
        const group = vi.fn(() => builder);
        const material = { _buildGroup: builder } as unknown as Material;
        Object.defineProperty(material, "_buildGroup", { get: group });
        const mesh = { _gpu: {}, material } as unknown as Mesh;
        addToScene(scene, mesh);
        const fog = enableVolumetricFog(scene, engine, { density: 0.1 });
        task._renderables = [{ mesh } as Renderable];
        task._lastVersion = 1;
        group.mockClear();
        task._assertLinearReady?.();
        const initialLookups = group.mock.calls.length;
        task._assertLinearReady?.();
        expect(group).toHaveBeenCalledTimes(initialLookups);

        fog.enabled = false;
        mesh.material = { _buildGroup: { _materialFamily: "standard" } } as unknown as Material;
        expect(() => task._assertLinearReady?.()).toThrow("Linear scene color requires PBR");
        disposeVolumetricFog(fog);
        expect(task._executeWithTransmission).toBeDefined();
        expect(() => task._executeWithTransmission?.(1)).toThrow("Linear scene color requires PBR");
    });

    it("rejects transparent PBR even after fog is disabled or disposed", () => {
        const { engine, scene, task } = createFogScene(1);
        const fog = enableVolumetricFog(scene, engine, { density: 0.1 });
        const material = { _buildGroup: { _materialFamily: "pbr" }, _linearImageProcessing: true } as unknown as Material;
        const renderable = { mesh: { material } as Mesh, isTransparent: true } as Renderable;
        task._renderables = [renderable];
        task._lastVersion = 1;
        task._assertLinearReady?.();
        task._transparentBindings.push({ renderable } as DrawBinding);

        fog.enabled = false;
        expect(() => task._assertLinearReady?.()).toThrow("transparent renderables");
        disposeVolumetricFog(fog);
        expect(() => task._executeWithTransmission?.(1)).toThrow("transparent renderables");
    });

    it("rejects another registered scene on the same surface before installing HDR", () => {
        const { engine, scene, task } = createFogScene(1);
        const oldTarget = task._config.rt;
        engine._renderingContexts.push(createSceneContext(engine));
        expect(() => enableVolumetricFog(scene, engine, { density: 0.1 })).toThrow("same surface");
        expect(task._config.rt).toBe(oldTarget);
        expect(scene._frameGraph._tasks).toHaveLength(1);
    });

    it("rejects shadowed sun and incompatible image processing before installation", () => {
        const { engine, scene, task } = createFogScene();
        const oldTarget = task._config.rt;
        const sun = { lightType: "directional", intensity: 1, diffuse: [1, 1, 1], direction: { x: 0, y: -1, z: 0 }, shadowGenerator: {} } as DirectionalLight;
        expect(() => enableVolumetricFog(scene, engine, { density: 0.1, sun })).toThrow("shadowed sun");
        scene.imageProcessing.toneMapping = { id: "custom", helpersWGSL: "", callWGSL: "" };
        scene.imageProcessing.toneMappingEnabled = true;
        expect(() => enableVolumetricFog(scene, engine, { density: 0.1 })).toThrow("standard tone mapping");
        expect(task._config.rt).toBe(oldTarget);
    });

    it("rejects non-finite sun directions both at installation and after mutation", () => {
        const { engine, scene } = createFogScene();
        const sun = { lightType: "directional", intensity: 1, diffuse: [1, 1, 1], direction: { x: NaN, y: -1, z: 0 } } as DirectionalLight;
        expect(() => enableVolumetricFog(scene, engine, { density: 0.1, sun })).toThrow("sun.direction components must be finite");
        sun.direction.x = Infinity;
        expect(() => enableVolumetricFog(scene, engine, { density: 0.1, sun })).toThrow("sun.direction components must be finite");
        sun.direction.x = 0;
        const fog = enableVolumetricFog(scene, engine, { density: 0.1, sun });
        sun.direction.z = NaN;
        expect(() => fog._phase.validate?.()).toThrow("sun.direction components must be finite");
        disposeVolumetricFog(fog);
    });
    it("allows an inactive custom tone mapper but rejects it when enabled", () => {
        const { engine, scene } = createFogScene();
        scene.imageProcessing.toneMapping = { id: "custom", helpersWGSL: "", callWGSL: "" };
        const fog = enableVolumetricFog(scene, engine, { density: 0.1 });
        scene.imageProcessing.toneMappingEnabled = true;
        expect(() => fog._task.execute?.()).toThrow("standard tone mapping");
        fog.enabled = false;
        expect(() => fog._task.execute?.()).toThrow("standard tone mapping");
        disposeVolumetricFog(fog);
        expect(() => fog._task.execute?.()).toThrow("standard tone mapping");
    });

    it("rejects unsupported linear image processing before installing transmission", () => {
        const { engine, scene, task } = createFogScene(1);
        scene.imageProcessing.toneMapping = { id: "custom", helpersWGSL: "", callWGSL: "" };
        scene.imageProcessing.toneMappingEnabled = true;
        expect(() => enableSceneTransmission(scene, engine)).toThrow("standard tone mapping");
        expect(task._config.rt).toBe(engine.scRT);
        expect(scene._meshMaterialChange).toBeUndefined();
    });

    it("restores the task's depth-load policy after a split pass", () => {
        const { engine, task } = createFogScene(1);
        engine._currentEncoder = { beginRenderPass: vi.fn(() => ({})) } as unknown as GPUCommandEncoder;
        const depthAttachment = {
            view: {} as GPUTextureView,
            depthLoadOp: "clear" as GPULoadOp,
            depthStoreOp: "store" as GPUStoreOp,
        };
        task._renderPassDescriptor.depthStencilAttachment = depthAttachment;
        task._config.clr = false;
        beginTaskPass(task, null, 1, false);
        expect(depthAttachment.depthLoadOp).toBe("clear");
        beginTaskPass(task, null, 1, true);
        expect(depthAttachment.depthLoadOp).toBe("load");
        beginTaskPass(task, null, 1, false);
        expect(depthAttachment.depthLoadOp).toBe("clear");

        task._config.depth!._eager = true;
        beginTaskPass(task, null, 1, false);
        expect(depthAttachment.depthLoadOp).toBe("load");
        task._config.depth = undefined;
        task._config.depthClear = false;
        beginTaskPass(task, null, 1, false);
        expect(depthAttachment.depthLoadOp).toBe("load");
    });

    for (const order of ["before", "after"] as const) {
        it.each([1, 4] as const)(`does not resume an empty ${order}-transmission fog pass at %ix MSAA`, (samples) => {
            const { engine, scene, task } = createFogScene(samples);
            if (order === "before") {
                enableSceneTransmission(scene, engine);
            }
            const fog = enableVolumetricFog(scene, engine, { density: 0.1 });
            if (order === "after") {
                enableSceneTransmission(scene, engine);
            }
            const pass = { setBindGroup: vi.fn(), executeBundles: vi.fn(), end: vi.fn() };
            const beginRenderPass = vi.fn(() => pass);
            engine._currentEncoder = { beginRenderPass } as unknown as GPUCommandEncoder;
            Object.assign(engine._device, {
                createRenderBundleEncoder: vi.fn(() => ({ setBindGroup: vi.fn(), finish: vi.fn(() => ({})) })),
            });
            const composite = vi.spyOn(fog._phase, "execute").mockReturnValue(1);
            const state = { _copies: -1 } as RenderTaskTransmissionState;

            expect(executePassWithTransmission(task, engine, state, samples)).toBe(1);
            expect(composite).toHaveBeenCalledOnce();
            expect(beginRenderPass).toHaveBeenCalledOnce();
            expect(pass.end).toHaveBeenCalledOnce();
            expect(state._copies).toBe(0);
        });
    }

    it("rejects invalid medium values without installing the feature", () => {
        const { engine, scene, task } = createFogScene();
        const originalTarget = task._config.rt;
        expect(() => enableVolumetricFog(scene, engine, { density: -1 })).toThrow("density");
        expect(() => enableVolumetricFog(scene, engine, { density: 0.1, maxDistance: 0.05 })).toThrow("maxDistance");
        expect(() => enableVolumetricFog(scene, engine, { density: 0.1, albedo: [1, 2, 1] })).toThrow("albedo");
        expect(task._config.rt).toBe(originalTarget);
        expect(task._afterOpaque).toBeUndefined();
    });

    it("validates a caller-owned density source before changing the render task", () => {
        const { engine, scene, task } = createFogScene();
        const originalTarget = task._config.rt;
        const shader = "fn fogDensityAt(worldPosition:vec3f)->f32{return fogBaseDensity(worldPosition);}";
        const reject = (densitySource: VolumetricFogDensitySource, message: string): void => {
            expect(() => enableVolumetricFog(scene, engine, { density: 0.2, densitySource })).toThrow(message);
            expect(task._config.rt).toBe(originalTarget);
            expect(task._afterOpaque).toBeUndefined();
        };
        reject({ shader: "" }, "densitySource.shader");
        reject({ shader, uniformFloatCount: -4 }, "densitySource.uniformFloatCount");
        reject({ shader, uniformFloatCount: 2 }, "densitySource.uniformFloatCount");
        reject({ shader, uniformFloatCount: Infinity }, "densitySource.uniformFloatCount");
        reject({ shader, uniformFloatCount: 4 }, "densitySource.writeUniforms");
        reject(
            {
                shader,
                validate: () => {
                    throw new RangeError("invalid density inputs");
                },
            },
            "invalid density inputs"
        );
    });

    it("validates mutable density inputs but keeps the installed shader and uniform layout fixed", () => {
        const { engine, scene } = createFogScene();
        const shader = "fn fogDensityAt(worldPosition:vec3f)->f32{return fogBaseDensity(worldPosition);}";
        const source = { shader, uniformFloatCount: 4, validate: vi.fn(), writeUniforms: (data: Float32Array) => data.fill(0) };
        const fog = enableVolumetricFog(scene, engine, { density: 0.2, densitySource: source });
        expect(fog._densitySource).toBe(source);
        source.validate.mockClear();
        fog._phase.validate?.();
        expect(source.validate).toHaveBeenCalledOnce();
        source.validate.mockImplementation(() => {
            throw new RangeError("invalid density inputs");
        });
        expect(() => fog._phase.validate?.()).toThrow("invalid density inputs");
        source.validate.mockReset();
        source.shader = `${shader}\n`;
        expect(() => fog._phase.validate?.()).toThrow("shader and uniform size cannot change");
        source.shader = shader;
        source.uniformFloatCount = 8;
        expect(() => fog._phase.validate?.()).toThrow("shader and uniform size cannot change");
        disposeVolumetricFog(fog);
    });

    it("prepares injection once per device and fails recovery before recording when replacement preparation rejects", async () => {
        const { engine, scene, task } = createFogScene(1);
        const densitySource: VolumetricFogDensitySource = {
            shader: "struct Params { values:vec4f }; @group(0) @binding(2) var<uniform> params:Params; fn fogDensityAt(p:vec3f)->f32 { return params.values.x; }",
            uniformFloatCount: 4,
            writeUniforms: (data) => data.fill(0.1),
        };
        const fog = enableVolumetricFog(scene, engine, { density: 0.1, densitySource });
        const initialDevice = createPreparationDevice();
        engine._device = initialDevice as unknown as GPUDevice;

        await task._preload?.();
        await task._preload?.();
        expect(initialDevice.createComputePipelineAsync).toHaveBeenCalledOnce();
        expect(initialDevice.createBindGroupLayout).toHaveBeenCalledWith({
            entries: expect.arrayContaining([{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", minBindingSize: 16 } }]),
        });
        expect(fog._preparedInjection?.device).toBe(initialDevice);

        Object.assign(engine, { surfaces: [engine] });
        engine._renderingContexts.push(scene);
        const replacement = createPreparationDevice();
        replacement.createComputePipelineAsync.mockRejectedValueOnce(new Error("replacement pipeline rejected"));
        engine._device = replacement as unknown as GPUDevice;
        const build = vi.spyOn(scene._frameGraph, "build");

        await expect(rebuildRegisteredScenes(engine)).rejects.toThrow("replacement pipeline rejected");
        expect(replacement.createComputePipelineAsync).toHaveBeenCalledOnce();
        expect(replacement.popErrorScope).toHaveBeenCalledOnce();
        expect(build).not.toHaveBeenCalled();
        expect(fog._preparedInjection).toBeNull();
        expect(fog._gpu).toBeNull();

        await task._preload?.();
        expect(replacement.createComputePipelineAsync).toHaveBeenCalledTimes(2);
        expect(fog._preparedInjection?.device).toBe(replacement);
        disposeVolumetricFog(fog);
        expect(fog._preparedInjection).toBeNull();
    });
});
