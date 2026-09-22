import type { EngineContext } from "../../engine/engine.js";
import { TU } from "../../engine/gpu-flags.js";
import { getOrCreateSampler } from "../../resource/texture-sampler-pool.js";

export type VolumeTextureFormat = "rgba8unorm" | "rgba16float";

export interface VolumeTexture3DOptions {
    readonly label?: string;
    readonly width: number;
    readonly height: number;
    readonly depth: number;
    readonly format: VolumeTextureFormat;
    readonly filter?: GPUFilterMode;
}

export interface VolumeTexture3D {
    texture: GPUTexture | null;
    readonly view: GPUTextureView;
    readonly sampler: GPUSampler;
    readonly width: number;
    readonly height: number;
    readonly depth: number;
    readonly format: VolumeTextureFormat;
}

export function createVolumeTexture3D(engine: EngineContext, options: VolumeTexture3DOptions): VolumeTexture3D {
    const width = positiveInteger(options.width, "width");
    const height = positiveInteger(options.height, "height");
    const depth = positiveInteger(options.depth, "depth");
    const limit = engine._device.limits.maxTextureDimension3D;
    if (width > limit || height > limit || depth > limit) {
        throw new RangeError(`VolumeTexture3D: ${width}x${height}x${depth} exceeds maxTextureDimension3D ${limit}.`);
    }
    const texture = engine._device.createTexture({
        label: options.label,
        size: { width, height, depthOrArrayLayers: depth },
        dimension: "3d",
        format: options.format,
        usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_SRC | TU.COPY_DST,
    });
    try {
        const view = texture.createView({ dimension: "3d" });
        const sampler = getOrCreateSampler(engine, {
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge",
            addressModeW: "clamp-to-edge",
            minFilter: options.filter ?? "linear",
            magFilter: options.filter ?? "linear",
        });
        return { texture, view, sampler, width, height, depth, format: options.format };
    } catch (error) {
        texture.destroy();
        throw error;
    }
}

export function disposeVolumeTexture3D(volume: VolumeTexture3D): void {
    volume.texture?.destroy();
    volume.texture = null;
}

function positiveInteger(value: number, name: string): number {
    if (!Number.isSafeInteger(value) || value <= 0) {
        throw new RangeError(`VolumeTexture3D: ${name} must be a positive safe integer (got ${value}).`);
    }
    return value;
}
