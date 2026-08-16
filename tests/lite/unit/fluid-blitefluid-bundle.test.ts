import { describe, expect, it } from "vitest";

import { parseBliteFluidBundle, parseBliteFluidCollision } from "../../../lab/lite/src/demos/fluid/blitefluid-bundle";
import type { FluidExportJson } from "../../../lab/lite/src/demos/fluid/preset-io";

function crc32(bytes: Uint8Array): number {
    let crc = 0xffffffff;
    for (const byte of bytes) {
        crc ^= byte;
        for (let bit = 0; bit < 8; bit++) {
            crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
        }
    }
    return (crc ^ 0xffffffff) >>> 0;
}

function collisionBytes(): Uint8Array {
    const bytes = new Uint8Array(64 + 8 * 4);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x46534c42, true);
    view.setUint32(4, 1, true);
    view.setUint32(8, 2, true);
    view.setUint32(12, 2, true);
    view.setUint32(16, 2, true);
    view.setFloat32(24, -1, true);
    view.setFloat32(28, -2, true);
    view.setFloat32(32, -3, true);
    view.setFloat32(36, 0.5, true);
    for (let index = 0; index < 8; index++) {
        view.setFloat32(64 + index * 4, index - 4, true);
    }
    return bytes;
}

function glbBytes(): Uint8Array {
    const bytes = new Uint8Array(12);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x46546c67, true);
    view.setUint32(4, 2, true);
    view.setUint32(8, 12, true);
    return bytes;
}

function storedZip(entries: Record<string, Uint8Array>): ArrayBuffer {
    const encoder = new TextEncoder();
    const parts: Uint8Array[] = [];
    let size = 0;
    for (const [name, payload] of Object.entries(entries)) {
        const nameBytes = encoder.encode(name);
        const header = new Uint8Array(30 + nameBytes.byteLength);
        const view = new DataView(header.buffer);
        view.setUint32(0, 0x04034b50, true);
        view.setUint16(4, 20, true);
        view.setUint32(14, crc32(payload), true);
        view.setUint32(18, payload.byteLength, true);
        view.setUint32(22, payload.byteLength, true);
        view.setUint16(26, nameBytes.byteLength, true);
        header.set(nameBytes, 30);
        parts.push(header, payload);
        size += header.byteLength + payload.byteLength;
    }
    const output = new Uint8Array(size);
    let offset = 0;
    for (const part of parts) {
        output.set(part, offset);
        offset += part.byteLength;
    }
    return output.buffer;
}

function validPreset(): FluidExportJson {
    return {
        formatVersion: 5,
        meta: { demo: "blender", method: "PBF" },
        physics: { gravity: 9.8, viscosity: 0.08, relaxation: 50, scorr: 0.02, iterations: 3, restDensity: 341, boundaryDensity: 0 },
        demoParams: {},
        demoState: {},
        emitters: [
            {
                id: "source",
                name: "Source",
                enabled: true,
                behavior: "inflow",
                transform: { position: [0, 2, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                shape: { type: "box", size: [2, 1, 2] },
                sampling: "volume",
                velocity: [0, -1, 0],
                velocitySpace: "world",
                spread: 0,
                volumeRate: 10,
            },
        ],
        sinks: [
            {
                id: "drain",
                name: "Drain",
                enabled: true,
                transform: { position: [0, -1, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
                shape: { type: "sphere", radius: 1 },
                targets: ["source"],
                volumeRate: 10,
            },
        ],
        showContainer: false,
        envIntensity: 1,
        msaa: true,
        activeBlocks: false,
        pagedGrid: false,
        fusedBlockDiscovery: false,
        physicsParticleSize: 1,
        gridPosition: [0, 10, 0],
        gridSize: [40, 20, 40],
        showGridBounds: false,
        particleCount: 80_000,
        material: 0,
        render: {
            renderAsSpheres: false,
            waterColor: "#16a3c3",
            absorption: 1,
            particleSize: 0.7,
            refractionStrength: 0.1,
            specularPower: 250,
            reflectionExposure: 2,
            reflectionContrast: 0.6,
            waterReflectivity: 0.02,
            surfaceDepthBlur: 3,
            depthBlurEdgeThreshold: 0.05,
            surfaceThicknessBlur: 1,
            halfRendering: false,
            thicknessDownscale: 1,
            surfaceFilter: "bilateral",
            narrowRangeDelta: 1,
            narrowRangeMu: 1,
            anisotropicSurface: false,
            anisoRadiusDamping: 0.2,
        },
        foam: {
            enableFoam: false,
            activeParticles: false,
            trappedAirRate: 40,
            waveCrestRate: 40,
            foamLifetime: 2,
            foamLifetimeMin: 0.3,
            bubbleBuoyancy: 0.8,
            bubbleDrag: 0.5,
            poolSize: 3,
            foamSoftness: 1,
            foamDensity: 1,
            subsurfaceBubbleStrength: 0,
            subsurfaceBubbleColor: "#ffffff",
            foamBlurRadius: 2,
            foamLightIntensity: 1,
            foamAmbient: 0.2,
            foamAO: 0,
            foamNormalStrength: 1,
            foamDebug: "none",
            foamSize: 1,
        },
    };
}

function bundleWithPreset(preset: unknown): ArrayBuffer {
    const manifest = new TextEncoder().encode(
        JSON.stringify({
            bundleVersion: 1,
            preset,
            scene: { glb: "scene.glb", collision: "collision.blsdf" },
        })
    );
    return storedZip({ "manifest.json": manifest, "scene.glb": glbBytes(), "collision.blsdf": collisionBytes() });
}

describe(".blitefluid bundle", () => {
    it("parses a stored ZIP with preset, GLB, and signed-distance data", () => {
        const bundle = parseBliteFluidBundle(bundleWithPreset(validPreset()));

        expect(bundle.manifest.preset.meta.method).toBe("PBF");
        expect(bundle.collision.dims).toEqual([2, 2, 2]);
        expect(bundle.collision.origin).toEqual([-1, -2, -3]);
        expect([...bundle.collision.distances]).toEqual([-4, -3, -2, -1, 0, 1, 2, 3]);
    });

    it("rejects truncated collision payloads", () => {
        expect(() => parseBliteFluidCollision(collisionBytes().subarray(0, -4))).toThrow("payload length");
    });

    it("rejects corrupted ZIP entries", () => {
        const archive = new Uint8Array(storedZip({ "scene.glb": glbBytes() }));
        archive[archive.length - 1] = archive[archive.length - 1]! ^ 1;
        expect(() => parseBliteFluidBundle(archive.buffer)).toThrow("CRC mismatch");
    });

    it.each([
        ["unknown bundle demo", (preset: FluidExportJson) => (preset.meta.demo = "waterfall"), 'meta.demo must be "blender"'],
        ["unknown solver method", (preset: FluidExportJson) => (preset.meta.method = "SPH"), "meta.method"],
        ["unbounded particle count", (preset: FluidExportJson) => (preset.particleCount = 2_000_001), "particleCount"],
        ["unbounded physics particle size", (preset: FluidExportJson) => (preset.physicsParticleSize = 9), "physicsParticleSize"],
        ["negative grid extent", (preset: FluidExportJson) => (preset.gridSize = [40, -1, 40]), "gridSize[1]"],
        ["unbounded grid allocation", (preset: FluidExportJson) => (preset.gridSize = [10_000, 10_000, 10_000]), "allocation limits"],
        [
            "too many emitters",
            (preset: FluidExportJson) => {
                preset.emitters = Array.from({ length: 17 }, (_, index) => ({ ...structuredClone(preset.emitters![0]!), id: `source-${index}` }));
            },
            "at most 16",
        ],
        [
            "malformed shape vector",
            (preset: FluidExportJson) => {
                preset.emitters![0]!.shape = { type: "box", size: [1, 0, 1] };
            },
            "shape.size[1]",
        ],
        [
            "unbounded emitter velocity",
            (preset: FluidExportJson) => {
                preset.emitters![0]!.velocity = [0, 100_001, 0];
            },
            "velocity[1]",
        ],
        [
            "invalid sink target",
            (preset: FluidExportJson) => {
                preset.sinks![0]!.targets = ["missing"];
            },
            "unknown or non-inflow emitter",
        ],
        [
            "conflicting sink rates",
            (preset: FluidExportJson) => {
                preset.sinks![0]!.perParticleRecycleRate = 1;
            },
            "cannot define both",
        ],
        [
            "malformed nested render object",
            (preset: FluidExportJson) => {
                preset.render = [] as unknown as FluidExportJson["render"];
            },
            "render must be an object",
        ],
        [
            "self-intersecting polygon shape",
            (preset: FluidExportJson) => {
                preset.emitters![0]!.shape = {
                    type: "polygonPrism",
                    points: [
                        [-1, -1],
                        [1, 1],
                        [-1, 1],
                        [1, -1],
                    ],
                    thickness: 1,
                };
            },
            "non-zero area",
        ],
    ])("rejects %s before installation", (_name, mutate, message) => {
        const preset = validPreset();
        mutate(preset);
        expect(() => parseBliteFluidBundle(bundleWithPreset(preset))).toThrow(message);
    });
});
