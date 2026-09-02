import type { EngineContext } from "../../engine/engine.js";
import type { ForceFieldSpec, SceneSdfSpec } from "./sim-common.js";

interface SceneSdfBindingOptions {
    readonly struct: string;
    readonly sdf: string;
    readonly params: Float32Array;
    readonly sdfGrid?: Float32Array;
    readonly gridConfine?: boolean;
}

interface ForceFieldBindingOptions {
    readonly struct: string;
    readonly wgsl: string;
    readonly params: Float32Array;
}

export interface SceneSdfRuntimeBinding {
    readonly spec: SceneSdfSpec;
    updateParams(params: Float32Array): void;
    updateSdfGrid(data: Float32Array): void;
    dispose(): void;
}

export interface ForceFieldRuntimeBinding {
    readonly spec: ForceFieldSpec;
    updateParams(params: Float32Array): void;
    dispose(): void;
}

function alignedUniformSize(byteLength: number): number {
    return Math.max(16, Math.ceil(byteLength / 16) * 16);
}

export function createSceneSdfRuntimeBinding(engine: EngineContext, options: SceneSdfBindingOptions): SceneSdfRuntimeBinding {
    const device = engine._device;
    const params = device.createBuffer({
        label: "fluid-scene-sdf-params",
        size: alignedUniformSize(options.params.byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    let sdfGrid: GPUBuffer | undefined;
    try {
        device.queue.writeBuffer(params, 0, options.params);
        if (options.sdfGrid) {
            sdfGrid = device.createBuffer({
                label: "fluid-scene-sdf-grid",
                size: Math.max(4, options.sdfGrid.byteLength),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(sdfGrid, 0, options.sdfGrid);
        }
    } catch (error) {
        params.destroy();
        sdfGrid?.destroy();
        throw error;
    }

    const spec: SceneSdfSpec = {
        struct: options.struct,
        sdf: options.sdf,
        buffer: params,
        ...(sdfGrid ? { sdfGrid } : {}),
        ...(options.gridConfine !== undefined ? { gridConfine: options.gridConfine } : {}),
    };
    return {
        spec,
        updateParams(data): void {
            if (data.byteLength > params.size) {
                throw new RangeError("[fluid] scene SDF parameters exceed the allocated binding size.");
            }
            device.queue.writeBuffer(params, 0, data);
        },
        updateSdfGrid(data): void {
            if (!sdfGrid) {
                throw new Error("[fluid] this scene SDF has no grid binding.");
            }
            if (data.byteLength > sdfGrid.size) {
                throw new RangeError("[fluid] scene SDF grid data exceeds the allocated binding size.");
            }
            device.queue.writeBuffer(sdfGrid, 0, data);
        },
        dispose(): void {
            params.destroy();
            sdfGrid?.destroy();
        },
    };
}

export function createForceFieldRuntimeBinding(engine: EngineContext, options: ForceFieldBindingOptions): ForceFieldRuntimeBinding {
    const device = engine._device;
    const params = device.createBuffer({
        label: "fluid-force-field-params",
        size: alignedUniformSize(options.params.byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    try {
        device.queue.writeBuffer(params, 0, options.params);
    } catch (error) {
        params.destroy();
        throw error;
    }
    return {
        spec: {
            struct: options.struct,
            wgsl: options.wgsl,
            buffer: params,
        },
        updateParams(data): void {
            if (data.byteLength > params.size) {
                throw new RangeError("[fluid] force-field parameters exceed the allocated binding size.");
            }
            device.queue.writeBuffer(params, 0, data);
        },
        dispose(): void {
            params.destroy();
        },
    };
}
