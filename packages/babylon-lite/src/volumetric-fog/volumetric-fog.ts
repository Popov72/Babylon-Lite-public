import type { Camera } from "../camera/camera.js";
import { getEffectiveAspectRatio, getProjectionMatrix } from "../camera/camera.js";
import type { EngineContext } from "../engine/engine.js";
import { BU, SS } from "../engine/gpu-flags.js";
import type { DirectionalLight } from "../light/directional-light.js";
import { isPbrMaterial } from "../material/material-guards.js";
import { writeWorldLightDirection } from "../light/light-base.js";
import { packMat4IntoF32 } from "../math/pack-mat4-into-f32.js";
import type { OpticalRgb } from "../render/volume/optical-transfer.js";
import { createViewVolumeGrid, viewVolumeTextureBytes, type ViewVolumeGrid } from "../render/volume/view-volume-grid.js";
import { createVolumeTexture3D, disposeVolumeTexture3D, type VolumeTexture3D } from "../render/volume/volume-texture.js";
import type { SceneContext } from "../scene/scene-core.js";
import { drawList, type RenderTaskBase } from "../frame-graph/render-task-base.js";
import { beginTaskPass, drawTaskOpaque, setTaskPassState } from "../frame-graph/render-task-phase-pass.js";
import {
    assertLinearImageProcessing,
    ensureLinearRenderTask,
    ensureLinearScenePresent,
    installLinearMaterialGuard,
    prepareSceneLinearPbr,
    trackSceneLinearPbr,
} from "../frame-graph/linear-scene-color.js";
import { fogInjectWgsl, fogIntegrateWgsl, fogCompositeWgsl } from "./volumetric-fog-wgsl.js";

/** An optional spatial density model. Its WGSL defines `fogDensityAt(worldPosition:vec3f)->f32`
 *  and may call the core's `fogBaseDensity(worldPosition)` or `fogBaseExtinction()`. A model with uniforms declares
 *  its own `@group(0) @binding(2) var<uniform>` in the WGSL source. */
export interface VolumetricFogDensitySource {
    readonly shader: string;
    /** Number of floats in the optional binding 2 uniform; a multiple of four. */
    readonly uniformFloatCount?: number;
    /** Validate mutable inputs before each enabled frame. */
    validate?(): void;
    /** Fill the entire reusable binding 2 uniform before each enabled frame. */
    writeUniforms?(data: Float32Array): void;
}

/** Configuration for Wronski-style frustum-aligned participating-media fog. */
export interface VolumetricFogOptions {
    density: number;
    height?: number;
    heightFalloff?: number;
    maxDensity?: number;
    albedo?: readonly [number, number, number];
    ambientRadiance?: readonly [number, number, number];
    sun?: DirectionalLight | null;
    anisotropy?: number;
    tileSize?: number;
    depthSlices?: number;
    maxDistance?: number;
    /** Optional caller-owned density model; omitted for homogeneous/height fog. */
    densitySource?: VolumetricFogDensitySource;
}

/** Editable fog medium and lighting; grid dimensions are fixed at installation. */
export interface VolumetricFog {
    enabled: boolean;
    density: number;
    height: number;
    heightFalloff: number;
    maxDensity: number;
    albedo: readonly [number, number, number];
    ambientRadiance: readonly [number, number, number];
    sun: DirectionalLight | null;
    anisotropy: number;
    /** @internal */
    _scene: SceneContext;
    /** @internal */
    _task: RenderTaskBase;
    /** @internal */
    _tileSize: number;
    /** @internal */
    _depthSlices: number;
    /** @internal */
    _maxDistance: number;
    /** @internal Density shader and uniform layout are fixed at installation. */
    _densitySource: VolumetricFogDensitySource | null;
    /** @internal */
    _densityShader: string | null;
    /** @internal */
    _densityUniformFloatCount: number;
    /** @internal Validated injection pipeline for the current device. */
    _preparedInjection: PreparedFogInjection | null;
    /** @internal */
    _gpu: FogGpuState | null;
    /** @internal */
    _disposed: boolean;
    /** @internal */
    _phase: NonNullable<RenderTaskBase["_afterOpaque"]>;
    /** @internal */
    _executeFog?: NonNullable<RenderTaskBase["_executeWithTransmission"]>;
}

interface PreparedFogInjection {
    readonly device: GPUDevice;
    readonly layout: GPUBindGroupLayout;
    readonly pipeline: GPUComputePipeline;
}

interface FogGpuState {
    grid: ViewVolumeGrid;
    readonly injected: VolumeTexture3D;
    readonly integrated: VolumeTexture3D;
    readonly paramsBuffer: GPUBuffer;
    readonly paramsData: Float32Array;
    readonly densityParamsBuffer: GPUBuffer | null;
    readonly densityParamsData: Float32Array | null;
    readonly injectPipeline: GPUComputePipeline;
    readonly integratePipeline: GPUComputePipeline;
    readonly compositePipeline: GPURenderPipeline;
    readonly injectBindings: GPUBindGroup;
    readonly integrateBindings: GPUBindGroup;
    readonly compositeBindings: GPUBindGroup;
    readonly compositeDescriptor: GPURenderPassDescriptor;
    readonly depthTexture: GPUTexture;
    readonly colorTexture: GPUTexture;
}

function finiteNonnegative(value: number, name: string): number {
    if (!Number.isFinite(value) || value < 0) {
        throw new RangeError(`Volumetric fog: ${name} must be finite and nonnegative (got ${value}).`);
    }
    return value;
}

function finitePositive(value: number, name: string): number {
    if (!Number.isFinite(value) || value <= 0) {
        throw new RangeError(`Volumetric fog: ${name} must be finite and positive (got ${value}).`);
    }
    return value;
}

function validateRgb(value: OpticalRgb, name: string, unitRange = false): void {
    for (let i = 0; i < 3; i++) {
        finiteNonnegative(value[i]!, `${name}[${i}]`);
        if (unitRange && value[i]! > 1) {
            throw new RangeError(`Volumetric fog: ${name}[${i}] must not exceed 1 (got ${value[i]}).`);
        }
    }
}

function fogCamera(fog: VolumetricFog): Camera {
    const camera = fog._task._config.cam ?? fog._scene.camera;
    if (!camera || camera.ortho || (camera.viewport && (camera.viewport.x !== 0 || camera.viewport.y !== 0 || camera.viewport.width !== 1 || camera.viewport.height !== 1))) {
        throw new Error("Volumetric fog requires a full-viewport perspective camera.");
    }
    if (!Number.isFinite(camera.nearPlane) || !Number.isFinite(camera.farPlane) || camera.nearPlane <= 0 || camera.farPlane <= camera.nearPlane) {
        throw new RangeError("Volumetric fog requires finite camera near/far planes with 0 < near < far.");
    }
    return camera;
}

function validateFogState(fog: VolumetricFog): void {
    finiteNonnegative(fog.density, "density");
    if (!Number.isFinite(fog.height)) {
        throw new RangeError("Volumetric fog: height must be finite.");
    }
    finiteNonnegative(fog.heightFalloff, "heightFalloff");
    finitePositive(fog.maxDensity, "maxDensity");
    validateRgb(fog.albedo, "albedo", true);
    validateRgb(fog.ambientRadiance, "ambientRadiance");
    if (!Number.isFinite(fog.anisotropy) || Math.abs(fog.anisotropy) > 0.95) {
        throw new RangeError("Volumetric fog: anisotropy must be finite and within [-0.95, 0.95].");
    }
    const source = fog._densitySource;
    if (source) {
        if (source.shader !== fog._densityShader || (source.uniformFloatCount ?? 0) !== fog._densityUniformFloatCount) {
            throw new Error("Volumetric fog: density source shader and uniform size cannot change after installation.");
        }
        source.validate?.();
    }
    if (fog.sun) {
        if (fog.sun.lightType !== "directional") {
            throw new Error("Volumetric fog: sun must be a directional light.");
        }
        finiteNonnegative(fog.sun.intensity, "sun.intensity");
        validateRgb(fog.sun.diffuse, "sun.diffuse");
        if (fog.sun.shadowGenerator) {
            throw new Error("Volumetric fog: shadowed sun injection is not available yet; supply an unshadowed directional light.");
        }
        const direction = fog.sun.direction;
        if (!Number.isFinite(direction.x) || !Number.isFinite(direction.y) || !Number.isFinite(direction.z)) {
            throw new RangeError("Volumetric fog: sun.direction components must be finite.");
        }
        if (Math.hypot(direction.x, direction.y, direction.z) <= 1e-8) {
            throw new RangeError("Volumetric fog: sun.direction must not be zero.");
        }
    }
}

/** Enable volume fog for a scene's default render task before registering the scene. */
export function enableVolumetricFog(scene: SceneContext, engine: EngineContext, options: VolumetricFogOptions): VolumetricFog {
    if (scene.surface.engine !== engine || scene._built || scene.fog) {
        throw new Error("Volumetric fog requires an unregistered scene on the supplied engine with analytic scene.fog disabled.");
    }
    if (engine.useFloatingOrigin) {
        throw new Error("Volumetric fog does not yet support floating-origin rendering.");
    }
    if (scene.surface._renderingContexts.some((context) => context._kind === "scene" && context !== scene)) {
        throw new Error("Volumetric fog does not support another scene on the same surface.");
    }
    const renderTasks = scene._frameGraph._tasks.filter((entry): entry is RenderTaskBase => "_renderables" in entry);
    if (renderTasks.length !== 1 || renderTasks[0]!.name !== "scene") {
        throw new Error("Volumetric fog requires the scene's single default RenderTask.");
    }
    const task = renderTasks[0]!;
    if (task._volumetricFogInstalled) {
        throw new Error("Volumetric fog was already installed on this render task; create a new scene to change its grid.");
    }
    if (task._afterOpaque || task._disposed) {
        throw new Error("Volumetric fog: the render task is already disposed or owns an after-opaque feature.");
    }
    const tileSize = options.tileSize ?? 16;
    const depthSlices = options.depthSlices ?? 64;
    if (!Number.isSafeInteger(tileSize) || tileSize <= 0 || !Number.isSafeInteger(depthSlices) || depthSlices <= 0) {
        throw new RangeError("Volumetric fog: tileSize and depthSlices must be positive safe integers.");
    }
    if (options.densitySource !== undefined && (!options.densitySource || typeof options.densitySource !== "object")) {
        throw new TypeError("Volumetric fog: densitySource must be an object when supplied.");
    }
    const densitySource = options.densitySource ?? null;
    const densityUniformFloatCount = densitySource?.uniformFloatCount ?? 0;
    if (densitySource) {
        if (typeof densitySource.shader !== "string" || densitySource.shader.trim().length === 0) {
            throw new TypeError("Volumetric fog: densitySource.shader must contain WGSL defining fogDensityAt.");
        }
        if (!Number.isSafeInteger(densityUniformFloatCount) || densityUniformFloatCount < 0 || densityUniformFloatCount % 4 !== 0) {
            throw new RangeError("Volumetric fog: densitySource.uniformFloatCount must be a nonnegative multiple of four.");
        }
        if (densitySource.validate !== undefined && typeof densitySource.validate !== "function") {
            throw new TypeError("Volumetric fog: densitySource.validate must be a function.");
        }
        if (densitySource.writeUniforms !== undefined && typeof densitySource.writeUniforms !== "function") {
            throw new TypeError("Volumetric fog: densitySource.writeUniforms must be a function.");
        }
        if (densityUniformFloatCount > 0 && !densitySource.writeUniforms) {
            throw new TypeError("Volumetric fog: densitySource.writeUniforms is required for nonempty uniforms.");
        }
    }
    const maxDistance = finitePositive(options.maxDistance ?? Math.min(scene.camera?.farPlane ?? 100, 100), "maxDistance");
    const fog: VolumetricFog = {
        enabled: true,
        density: options.density,
        height: options.height ?? 0,
        heightFalloff: options.heightFalloff ?? 0,
        maxDensity: options.maxDensity ?? 1,
        albedo: options.albedo ?? [1, 1, 1],
        ambientRadiance: options.ambientRadiance ?? [0.1, 0.1, 0.1],
        sun: options.sun ?? null,
        anisotropy: options.anisotropy ?? 0,
        _scene: scene,
        _task: task,
        _tileSize: tileSize,
        _depthSlices: depthSlices,
        _maxDistance: maxDistance,
        _densitySource: densitySource,
        _densityShader: densitySource?.shader ?? null,
        _densityUniformFloatCount: densityUniformFloatCount,
        _preparedInjection: null,
        _gpu: null,
        _disposed: false,
        _phase: {
            enabled: () => fog.enabled && !fog._disposed,
            validate: () => assertFogReady(fog),
            execute: () => executeFogPhase(fog),
        },
    };
    const camera = fogCamera(fog);
    assertFogReady(fog);
    for (const mesh of scene.meshes) {
        if (!mesh.material || !isPbrMaterial(mesh.material)) {
            throw new Error("Volumetric fog initially supports opaque PBR renderables with linear image processing only.");
        }
    }
    for (const renderable of scene._renderables) {
        if (!renderable.mesh?.material || !isPbrMaterial(renderable.mesh.material)) {
            throw new Error("Volumetric fog initially supports opaque PBR renderables with linear image processing only.");
        }
    }
    assertLinearImageProcessing(scene);
    if (Math.min(camera.farPlane, maxDistance) <= camera.nearPlane) {
        throw new RangeError("Volumetric fog: maxDistance must exceed the camera near plane.");
    }

    prepareSceneLinearPbr(scene);
    ensureLinearRenderTask(task, engine);
    installLinearMaterialGuard(task);
    const originalPreload = task._preload?.bind(task);
    let pendingInjection: { device: GPUDevice; promise: Promise<PreparedFogInjection> } | null = null;
    task._preload = async () => {
        if (originalPreload) {
            await originalPreload();
        }
        if (fog._disposed) {
            return;
        }
        const device = engine._device;
        if (fog._preparedInjection?.device === device) {
            return;
        }
        fog._preparedInjection = null;
        if (pendingInjection?.device !== device) {
            pendingInjection = { device, promise: prepareFogInjection(fog, device) };
        }
        const pending = pendingInjection;
        try {
            const prepared = await pending.promise;
            if (engine._device !== device || fog._disposed) {
                throw new Error("Volumetric fog: GPU device changed or fog was disposed during injection preparation.");
            }
            fog._preparedInjection = prepared;
        } finally {
            if (pendingInjection === pending) {
                pendingInjection = null;
            }
        }
    };
    const originalRecord = task.record.bind(task);
    const originalDispose = task.dispose.bind(task);
    task.record = () => {
        originalRecord();
        task._assertLinearReady?.();
        if (fog._disposed) {
            return;
        }
        if (fog.enabled) {
            assertFogReady(fog);
        }
        const next = createFogGpuState(fog, engine);
        const previous = fog._gpu;
        fog._gpu = next;
        disposeFogGpuState(previous);
    };
    task.dispose = () => {
        disposeVolumetricFog(fog);
        originalDispose();
    };
    task._afterOpaque = fog._phase;
    if (!task._transmissionInstalled) {
        const execute = (samples: number): number => executeFogSplitPass(fog, samples);
        fog._executeFog = execute;
        task._executeWithTransmission = execute;
    }
    ensureLinearScenePresent(task, engine, "volumetric-fog-image-processing");
    trackSceneLinearPbr(scene);
    task._volumetricFogInstalled = true;
    return fog;
}

/** Release fog-owned GPU state and detach the post-opaque phase. Idempotent. */
export function disposeVolumetricFog(fog: VolumetricFog): void {
    if (fog._disposed) {
        return;
    }
    fog._disposed = true;
    const task = fog._task;
    if (task._afterOpaque === fog._phase) {
        task._afterOpaque = undefined;
    }
    fog._preparedInjection = null;
    disposeFogGpuState(fog._gpu);
    fog._gpu = null;
}

function executeFogSplitPass(fog: VolumetricFog, sampleCount: number): number {
    const task = fog._task;
    task._assertLinearReady?.();
    const afterOpaque = task._afterOpaque;
    const runAfterOpaque = afterOpaque?.enabled() ?? false;
    if (runAfterOpaque) {
        afterOpaque!.validate?.();
    }
    const resolve = sampleCount > 1 ? (task._config.rst?._colorView ?? null) : null;
    let pass = beginTaskPass(task, resolve, sampleCount, false);
    let draws = drawTaskOpaque(task, pass);
    if (runAfterOpaque) {
        pass.end();
        draws += afterOpaque!.execute(task.engine, sampleCount);
        if (task._transparentBindings.length === 0) {
            return draws;
        }
        pass = beginTaskPass(task, resolve, sampleCount, true);
        setTaskPassState(task, pass);
    }
    draws += drawList(pass, task._transparentBindings, task.engine);
    pass.end();
    return draws;
}

function assertFogReady(fog: VolumetricFog): void {
    const task = fog._task;
    const camera = fogCamera(fog);
    validateFogState(fog);
    if (task._transparentBindings.length > 0) {
        throw new Error("Volumetric fog does not yet support transparent renderables.");
    }
    if (fog._scene.fog) {
        throw new Error("Volumetric fog cannot run together with analytic scene.fog.");
    }
    const farDepth = Math.min(camera.farPlane, fog._maxDistance);
    if (farDepth <= camera.nearPlane) {
        throw new RangeError("Volumetric fog: camera near plane must remain closer than fog maxDistance.");
    }
}

function executeFogPhase(fog: VolumetricFog): number {
    const task = fog._task;
    const state = fog._gpu;
    if (!state) {
        throw new Error("Volumetric fog GPU state has not been recorded.");
    }
    const depthTexture = task._config.depth?._depthTexture ?? task._config.rt._depthTexture;
    if (depthTexture !== state.depthTexture || task._config.rt._colorTexture !== state.colorTexture) {
        throw new Error("Volumetric fog render targets changed without a frame-graph rebuild.");
    }
    const camera = fogCamera(fog);
    const farDepth = Math.min(camera.farPlane, fog._maxDistance);
    if (state.grid.nearDepth !== camera.nearPlane || state.grid.farDepth !== farDepth) {
        state.grid = createViewVolumeGrid({
            targetWidth: state.grid.targetWidth,
            targetHeight: state.grid.targetHeight,
            tileSize: state.grid.tileSize,
            depthSlices: state.grid.depth,
            nearDepth: camera.nearPlane,
            farDepth,
            depthMapping: state.grid.depthMapping,
        });
    }
    writeFogParams(fog, state, camera);
    const encoder = task.engine._currentEncoder;
    const injection = encoder.beginComputePass({ label: "volumetric-fog-inject" });
    injection.setPipeline(state.injectPipeline);
    injection.setBindGroup(0, state.injectBindings);
    injection.dispatchWorkgroups(Math.ceil(state.grid.width / 8), Math.ceil(state.grid.height / 8), state.grid.depth);
    injection.end();
    const integration = encoder.beginComputePass({ label: "volumetric-fog-integrate" });
    integration.setPipeline(state.integratePipeline);
    integration.setBindGroup(0, state.integrateBindings);
    integration.dispatchWorkgroups(Math.ceil(state.grid.width / 8), Math.ceil(state.grid.height / 8));
    integration.end();
    const composite = encoder.beginRenderPass(state.compositeDescriptor);
    composite.setPipeline(state.compositePipeline);
    composite.setBindGroup(0, state.compositeBindings);
    composite.draw(3);
    composite.end();
    return 1;
}

async function prepareFogInjection(fog: VolumetricFog, device: GPUDevice): Promise<PreparedFogInjection> {
    const uniformBytes = fog._densityUniformFloatCount * 4;
    if (uniformBytes > device.limits.maxUniformBufferBindingSize) {
        throw new RangeError("Volumetric fog: density source uniforms exceed maxUniformBufferBindingSize.");
    }
    device.pushErrorScope("validation");
    let prepared: PreparedFogInjection | null = null;
    let preparationError: unknown;
    try {
        const entries: GPUBindGroupLayoutEntry[] = [
            { binding: 0, visibility: SS.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: SS.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
        ];
        if (uniformBytes > 0) {
            entries.push({ binding: 2, visibility: SS.COMPUTE, buffer: { type: "uniform", minBindingSize: uniformBytes } });
        }
        const layout = device.createBindGroupLayout({ entries });
        const shader = device.createShaderModule({ label: "fog-inject", code: fogInjectWgsl(fog._densityShader) });
        const compilation = await shader.getCompilationInfo();
        const issue = compilation.messages.find((message) => message.type === "error");
        if (issue) {
            throw new Error(`density shader WGSL at ${issue.lineNum}:${issue.linePos}: ${issue.message}`);
        }
        const pipeline = await device.createComputePipelineAsync({
            layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module: shader, entryPoint: "main" },
        });
        prepared = { device, layout, pipeline };
    } catch (error) {
        preparationError = error;
    }
    const validationError = await device.popErrorScope();
    if (preparationError) {
        const detail = preparationError instanceof Error ? preparationError.message : String(preparationError);
        throw new Error(`Volumetric fog: injection pipeline preparation failed: ${detail}`, { cause: preparationError });
    }
    if (validationError) {
        throw new Error(`Volumetric fog: injection pipeline validation failed: ${validationError.message}`, { cause: validationError });
    }
    if (!prepared) {
        throw new Error("Volumetric fog: injection pipeline preparation produced no pipeline.");
    }
    return prepared;
}

function createFogGpuState(fog: VolumetricFog, engine: EngineContext): FogGpuState {
    const task = fog._task;
    const camera = fogCamera(fog);
    const rt = task._config.rt;
    const depthTexture = task._config.depth?._depthTexture ?? rt._depthTexture;
    if (!depthTexture || !rt._colorTexture || !rt._colorView || rt._width <= 0 || rt._height <= 0) {
        throw new Error("Volumetric fog needs a recorded scene color and depth target.");
    }
    const grid = createViewVolumeGrid({
        targetWidth: rt._width,
        targetHeight: rt._height,
        tileSize: fog._tileSize,
        depthSlices: fog._depthSlices,
        nearDepth: camera.nearPlane,
        farDepth: Math.min(camera.farPlane, fog._maxDistance),
        depthMapping: { kind: "power", exponent: 2 },
    });
    viewVolumeTextureBytes(grid, 16);
    const maxWorkgroups = engine._device.limits.maxComputeWorkgroupsPerDimension;
    if (Math.ceil(grid.width / 8) > maxWorkgroups || Math.ceil(grid.height / 8) > maxWorkgroups || grid.depth > maxWorkgroups) {
        throw new RangeError("Volumetric fog: grid dispatch exceeds maxComputeWorkgroupsPerDimension.");
    }
    const prepared = fog._preparedInjection;
    if (!prepared || prepared.device !== engine._device) {
        throw new Error("Volumetric fog: injection pipeline must be prepared for this device before recording the frame graph.");
    }
    let injected: VolumeTexture3D | null = null;
    let integrated: VolumeTexture3D | null = null;
    let paramsBuffer: GPUBuffer | null = null;
    let densityParamsBuffer: GPUBuffer | null = null;
    try {
        injected = createVolumeTexture3D(engine, { label: "fog-injection", width: grid.width, height: grid.height, depth: grid.depth, format: "rgba16float" });
        integrated = createVolumeTexture3D(engine, { label: "fog-integrated", width: grid.width, height: grid.height, depth: grid.depth, format: "rgba16float" });
        const device = engine._device;
        const paramsData = new Float32Array(48);
        paramsBuffer = device.createBuffer({ label: "fog-params", size: paramsData.byteLength, usage: BU.UNIFORM | BU.COPY_DST });
        let densityParamsData: Float32Array | null = null;
        if (fog._densityUniformFloatCount > 0) {
            const size = fog._densityUniformFloatCount * 4;
            densityParamsData = new Float32Array(fog._densityUniformFloatCount);
            densityParamsBuffer = device.createBuffer({ label: "fog-density-params", size, usage: BU.UNIFORM | BU.COPY_DST });
        }
        const uniform = { binding: 0, visibility: SS.COMPUTE, buffer: { type: "uniform" } } satisfies GPUBindGroupLayoutEntry;
        const integrationLayout = device.createBindGroupLayout({
            entries: [
                uniform,
                { binding: 1, visibility: SS.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
                { binding: 2, visibility: SS.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
            ],
        });
        const samples = task._targetSignature._sampleCount;
        const compositeLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: SS.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: SS.FRAGMENT, texture: { sampleType: "depth", multisampled: samples > 1 } },
                { binding: 2, visibility: SS.FRAGMENT, texture: { sampleType: "float", viewDimension: "3d" } },
                { binding: 3, visibility: SS.FRAGMENT, sampler: { type: "filtering" } },
            ],
        });
        const integrateModule = device.createShaderModule({ label: "fog-integrate", code: fogIntegrateWgsl() });
        const compositeModule = device.createShaderModule({ label: "fog-composite", code: fogCompositeWgsl(samples > 1) });
        const integratePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [integrationLayout] }),
            compute: { module: integrateModule, entryPoint: "main" },
        });
        const compositePipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [compositeLayout] }),
            vertex: { module: compositeModule, entryPoint: "vs" },
            fragment: {
                module: compositeModule,
                entryPoint: "fs",
                targets: [
                    {
                        format: "rgba16float",
                        blend: {
                            color: { srcFactor: "one", dstFactor: "src-alpha", operation: "add" },
                            alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
                        },
                    },
                ],
            },
            primitive: { topology: "triangle-list" },
            multisample: { count: samples },
        });
        const injectEntries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: injected.view },
        ];
        if (densityParamsBuffer) {
            injectEntries.push({ binding: 2, resource: { buffer: densityParamsBuffer } });
        }
        const injectBindings = device.createBindGroup({ layout: prepared.layout, entries: injectEntries });
        const integrateBindings = device.createBindGroup({
            layout: integrationLayout,
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: injected.view },
                { binding: 2, resource: integrated.view },
            ],
        });
        const compositeBindings = device.createBindGroup({
            layout: compositeLayout,
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: depthTexture.createView({ aspect: "depth-only" }) },
                { binding: 2, resource: integrated.view },
                { binding: 3, resource: integrated.sampler },
            ],
        });
        const colorAttachment: GPURenderPassColorAttachment = { view: rt._colorView, loadOp: "load", storeOp: "store" };
        return {
            grid,
            injected,
            integrated,
            paramsBuffer,
            paramsData,
            densityParamsBuffer,
            densityParamsData,
            injectPipeline: prepared.pipeline,
            integratePipeline,
            compositePipeline,
            injectBindings,
            integrateBindings,
            compositeBindings,
            compositeDescriptor: { label: "volumetric-fog-composite", colorAttachments: [colorAttachment] },
            depthTexture,
            colorTexture: rt._colorTexture,
        };
    } catch (error) {
        densityParamsBuffer?.destroy();
        paramsBuffer?.destroy();
        if (integrated) {
            disposeVolumeTexture3D(integrated);
        }
        if (injected) {
            disposeVolumeTexture3D(injected);
        }
        throw error;
    }
}

function writeFogParams(fog: VolumetricFog, state: FogGpuState, camera: Camera): void {
    const data = state.paramsData;
    const grid = state.grid;
    const aspect = getEffectiveAspectRatio(camera, grid.targetWidth, grid.targetHeight);
    const projection = getProjectionMatrix(camera, aspect);
    packMat4IntoF32(data, camera.worldMatrix);
    data[16] = grid.width;
    data[17] = grid.height;
    data[18] = grid.depth;
    data[19] = grid.tileSize;
    data[20] = grid.nearDepth;
    data[21] = grid.farDepth;
    data[22] = projection[0]!;
    data[23] = projection[5]!;
    data[24] = fog.density;
    data[25] = fog.height;
    data[26] = fog.heightFalloff;
    data[27] = fog.maxDensity;
    data[28] = fog.albedo[0];
    data[29] = fog.albedo[1];
    data[30] = fog.albedo[2];
    data[31] = fog.anisotropy;
    data[32] = fog.ambientRadiance[0];
    data[33] = fog.ambientRadiance[1];
    data[34] = fog.ambientRadiance[2];
    data[35] = 0;
    const sun = fog.sun;
    if (sun) {
        writeWorldLightDirection(data, 36, sun.worldMatrix, sun.direction);
        data[39] = sun.intensity;
        data[40] = sun.diffuse[0];
        data[41] = sun.diffuse[1];
        data[42] = sun.diffuse[2];
    } else {
        data[36] = 0;
        data[37] = -1;
        data[38] = 0;
        data[39] = data[40] = data[41] = data[42] = 0;
    }
    data[43] = 0;
    data[44] = grid.targetWidth;
    data[45] = grid.targetHeight;
    data[46] = camera.farPlane;
    data[47] = 0;
    fog._task.engine._device.queue.writeBuffer(state.paramsBuffer, 0, data);
    if (state.densityParamsData && state.densityParamsBuffer) {
        const densityData = state.densityParamsData;
        densityData.fill(NaN);
        fog._densitySource!.writeUniforms!(densityData);
        for (let i = 0; i < densityData.length; i++) {
            if (!Number.isFinite(densityData[i])) {
                throw new RangeError(`Volumetric fog: densitySource uniform[${i}] must be finite.`);
            }
        }
        fog._task.engine._device.queue.writeBuffer(state.densityParamsBuffer, 0, densityData);
    }
}

function disposeFogGpuState(state: FogGpuState | null): void {
    if (!state) {
        return;
    }
    state.paramsBuffer.destroy();
    state.densityParamsBuffer?.destroy();
    disposeVolumeTexture3D(state.injected);
    disposeVolumeTexture3D(state.integrated);
}
