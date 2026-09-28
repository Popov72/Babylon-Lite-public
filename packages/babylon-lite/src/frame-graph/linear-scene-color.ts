import type { EngineContext } from "../engine/engine.js";
import { createRenderTarget, type RenderTarget } from "../engine/render-target.js";
import { isPbrMaterial } from "../material/material-guards.js";
import { _registerPbrExt } from "../material/pbr/pbr-flags.js";
import { makeLinearImageProcessingExt } from "../material/pbr/fragments/linear-image-processing-fragment.js";
import type { Mesh } from "../mesh/mesh.js";
import type { SceneContext } from "../scene/scene-core.js";
import { createImageProcessingTask } from "./image-processing-task.js";
import type { RenderTaskBase } from "./render-task-base.js";
import type { Task } from "./task.js";

let preparedScenes: WeakSet<SceneContext> | null = null;

function markMaterialLinear(material: Mesh["material"] | undefined): void {
    if (material && isPbrMaterial(material)) {
        material._linearImageProcessing = true;
        material._renderFeatures = undefined;
    }
}

/** @internal Use the same PBR output variant for transmission and volumetric effects. */
export function prepareSceneLinearPbr(scene: SceneContext): void {
    _registerPbrExt(makeLinearImageProcessingExt());
    for (const mesh of scene.meshes) {
        markMaterialLinear(mesh.material);
    }
}

/** @internal Track later material additions only after the linear scene target is committed. */
export function trackSceneLinearPbr(scene: SceneContext): void {
    if ((preparedScenes ??= new WeakSet()).has(scene)) {
        return;
    }
    preparedScenes.add(scene);
    const previous = scene._meshMaterialChange;
    const changed: NonNullable<SceneContext["_meshMaterialChange"]> = (mesh, material) => {
        scene._linearMaterialRevision = (scene._linearMaterialRevision ?? 0) + 1;
        previous?.(mesh, material);
        markMaterialLinear(material);
    };
    scene._meshMaterialChange = changed;
    scene._disposables.push(() => {
        preparedScenes?.delete(scene);
        if (scene._meshMaterialChange === changed) {
            scene._meshMaterialChange = previous;
        }
    });
}

/** @internal Retain the PBR linear-output contract for the lifetime of a fog-owned HDR target. */
export function installLinearMaterialGuard(task: RenderTaskBase): void {
    let checkedRenderables: RenderTaskBase["_renderables"] | null = null;
    let checkedVersion: number | undefined;
    let checkedEpoch: number | undefined;
    let checkedRevision: number | undefined;
    task._assertLinearReady = () => {
        if (task._transparentBindings.length > 0) {
            throw new Error("Volumetric fog does not support transparent renderables while its HDR target is installed.");
        }
        const scene = task.scene;
        const renderables = task._renderables;
        if (
            checkedRenderables === renderables &&
            checkedVersion === task._lastVersion &&
            checkedEpoch === scene._materialEpoch &&
            checkedRevision === scene._linearMaterialRevision
        ) {
            return;
        }
        for (const renderable of renderables) {
            const material = renderable.mesh?.material;
            if (!material || !isPbrMaterial(material) || material._linearImageProcessing !== true) {
                throw new Error("Linear scene color requires PBR renderables with linear image processing.");
            }
        }
        checkedRenderables = renderables;
        checkedVersion = task._lastVersion;
        checkedEpoch = scene._materialEpoch;
        checkedRevision = scene._linearMaterialRevision;
    };
}

/** @internal The shared final present implements only standard or disabled tone mapping. */
export function assertLinearImageProcessing(scene: SceneContext): void {
    const imageProcessing = scene.imageProcessing;
    if (imageProcessing.toneMappingEnabled === true && imageProcessing.toneMapping && imageProcessing.toneMapping.id !== "standard") {
        throw new Error("Linear scene color requires standard tone mapping (or tone mapping disabled).");
    }
}

/** @internal Retarget a scene render task only once, regardless of which feature enabled linear output first. */
export function ensureLinearRenderTask(task: RenderTaskBase, engine: EngineContext): RenderTarget {
    if (task.engine !== engine) {
        throw new Error("Linear scene color: render task belongs to a different engine.");
    }
    if (task._linearSceneColor) {
        return task._config.rt;
    }
    const cfg = task._config;
    if (cfg.sharedRt) {
        throw new Error("Linear scene color requires a render task that owns its color target.");
    }
    const oldDesc = cfg.rt._descriptor;
    const sampleCount = oldDesc.samples ?? 1;
    const newRt = createRenderTarget({
        lbl: "scene-linear",
        format: "rgba16float",
        dFormat: cfg.depth ? undefined : (oldDesc.dFormat ?? "depth24plus-stencil8"),
        _depthClearValue: oldDesc._depthClearValue,
        _depthCompare: oldDesc._depthCompare,
        samples: sampleCount,
        size: cfg.rt === task.scene.surface.scRT ? task.scene.surface : oldDesc.size,
    });
    cfg.rt = newRt;
    cfg.rst = undefined;
    const signature = task._targetSignature as {
        _colorFormat?: GPUTextureFormat;
        _depthStencilFormat?: GPUTextureFormat;
        _depthCompare?: GPUCompareFunction;
        _sampleCount: number;
    };
    signature._colorFormat = "rgba16float";
    signature._depthStencilFormat = cfg.depth?._descriptor.dFormat ?? newRt._descriptor.dFormat;
    signature._depthCompare = newRt._descriptor._depthCompare;
    signature._sampleCount = sampleCount;
    task._ob.length = 0;
    task._lastVersion = -1;

    const execute = task.execute?.bind(task);
    if (execute) {
        task.execute = () => executeRenderTaskLinear(task.scene, execute);
    }
    task._linearSceneColor = true;
    return newRt;
}

/** @internal One scene-color present after all features have composited into the shared HDR target. */
export function ensureLinearScenePresent(task: RenderTaskBase, engine: EngineContext, name: string): Task {
    if (task._linearPresentTask) {
        return task._linearPresentTask;
    }
    const present = createImageProcessingTask({ name, source: task._config.rt, target: task.scene.surface.scRT }, engine, task.scene);
    task.scene._frameGraph._tasks.push(present);
    task._linearPresentTask = present;
    return present;
}

function executeRenderTaskLinear(scene: SceneContext, execute: () => number): number {
    assertLinearImageProcessing(scene);
    const imageProcessing = scene.imageProcessing as { exposure: number; contrast: number; toneMappingEnabled: boolean | number };
    const toneMappingEnabled = imageProcessing.toneMappingEnabled;
    const clearColor = scene.clearColor;
    scene.clearColor = {
        r: inverseImageProcessedChannel(clearColor.r, imageProcessing.exposure, imageProcessing.contrast, toneMappingEnabled === true),
        g: inverseImageProcessedChannel(clearColor.g, imageProcessing.exposure, imageProcessing.contrast, toneMappingEnabled === true),
        b: inverseImageProcessedChannel(clearColor.b, imageProcessing.exposure, imageProcessing.contrast, toneMappingEnabled === true),
        a: clearColor.a,
    };
    imageProcessing.toneMappingEnabled = -1;
    try {
        return execute();
    } finally {
        scene.clearColor = clearColor;
        imageProcessing.toneMappingEnabled = toneMappingEnabled;
    }
}

function inverseImageProcessedChannel(value: number, exposure: number, contrast: number, toneMapping: boolean): number {
    let color = Math.min(Math.max(value, 0), 1);
    if (contrast < 1) {
        color = contrast > 0 ? Math.min(Math.max((color - 0.5 * (1 - contrast)) / contrast, 0), 1) : 0.5;
    } else if (contrast > 1) {
        const mixAmount = contrast - 1;
        let lo = 0;
        let hi = 1;
        for (let i = 0; i < 16; i++) {
            const mid = (lo + hi) * 0.5;
            const high = mid * mid * (3 - 2 * mid);
            const out = mid + (high - mid) * mixAmount;
            if (out < color) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        color = (lo + hi) * 0.5;
    }
    color **= 2.2;
    if (toneMapping) {
        color = -Math.log2(Math.max(1 - color, 1e-6)) / 1.5905790328979492;
    }
    return exposure > 0 ? color / exposure : color;
}
