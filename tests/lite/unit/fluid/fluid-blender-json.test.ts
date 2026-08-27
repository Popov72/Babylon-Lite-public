import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { parseBlenderFluidCollision, parseBlenderFluidJson, scenePayloadFromBlenderFluidJson } from "../../../../lab/lite/src/demos/fluid/blender-fluid-json";
import type { FluidExportJson } from "../../../../lab/lite/src/demos/fluid/preset-io";

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

describe("Blender FLIP Fluids exporter", () => {
    it("exports the format-13 quality-comparison settings", () => {
        const source = readFileSync(resolve(process.cwd(), "scripts/blender-fluid-addon.py"), "utf8");
        expect(source).toContain('"version": (3, 2, 0)');
        expect(source).toContain('"formatVersion": 13');
        for (const key of [
            "pressureSolver",
            "multigridCycles",
            "pressureTolerance",
            "pressureDiagnostics",
            "liquidSdf",
            "ghostFluid",
            "fractionalSolids",
            "movingSolidBoundaries",
            "reseedParticles",
            "reseedMinParticles",
            "reseedTargetParticles",
            "reseedMaxParticles",
            "reseedInterval",
            "particleSheeting",
            "sheetingStrength",
            "sheetingInterval",
            "polygonSurface",
        ]) {
            expect(source).toContain(`"${key}":`);
        }
    });

    it("accepts format-13 self-contained exports", () => {
        const preset = validPreset();
        const json = JSON.parse(selfContainedJson(preset)) as FluidExportJson;
        json.formatVersion = 13;

        expect(parseBlenderFluidJson(JSON.stringify(json)).preset.formatVersion).toBe(13);
    });
});

function validPreset(): FluidExportJson {
    return {
        formatVersion: 5,
        meta: { demo: "blender", method: "PBF" },
        physics: { gravity: 9.8, viscosity: 0.08, relaxation: 50, scorr: 0.02, iterations: 3, restDensity: 341, boundaryDensity: 0 },
        demoParams: {},
        demoState: {},
        simulationDuration: 12,
        alphaDecay: 1.5,
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
        showGridBoundsSolid: false,
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
            foamDebug: "off",
            foamSize: 1,
        },
    };
}

function selfContainedJson(preset: FluidExportJson): string {
    preset.formatVersion = 11;
    for (const sink of preset.sinks ?? []) {
        sink.mode ??= "recycle";
    }
    preset.scene = {
        encoding: "base64",
        glb: Buffer.from(glbBytes()).toString("base64"),
        collision: Buffer.from(collisionBytes()).toString("base64"),
        anchorPosition: [1, 2, 3],
    };
    return JSON.stringify(preset);
}

describe("Blender fluid JSON", () => {
    it("parses the self-contained format-6 Blender JSON", () => {
        const preset = validPreset();
        preset.formatVersion = 6;
        preset.simulationTimeScale = 0.5;
        preset.scene = {
            encoding: "base64",
            glb: Buffer.from(glbBytes()).toString("base64"),
            collision: Buffer.from(collisionBytes()).toString("base64"),
        };

        const bundle = parseBlenderFluidJson(JSON.stringify(preset));

        expect(bundle.preset.simulationTimeScale).toBe(0.5);
        expect(bundle.preset.sinks?.[0]?.mode).toBe("recycle");
        expect(bundle.sceneGlb.byteLength).toBe(glbBytes().byteLength);
        expect(bundle.collision.dims).toEqual([2, 2, 2]);
    });

    it("parses format-7 delete sinks and requires an explicit mode", () => {
        const preset = validPreset();
        preset.formatVersion = 7;
        preset.sinks![0]!.mode = "delete";
        preset.sinks![0]!.targets = [];
        preset.scene = {
            encoding: "base64",
            glb: Buffer.from(glbBytes()).toString("base64"),
            collision: Buffer.from(collisionBytes()).toString("base64"),
        };

        expect(parseBlenderFluidJson(JSON.stringify(preset)).preset.sinks?.[0]?.mode).toBe("delete");
        delete preset.sinks![0]!.mode;
        expect(() => parseBlenderFluidJson(JSON.stringify(preset))).toThrow(/mode must be/);
    });

    it("parses format-8 optional Initial Velocity fields", () => {
        const preset = validPreset();
        preset.formatVersion = 8;
        preset.emitters![0]!.sourceVelocity = [1, 2, 3];
        preset.emitters![0]!.sourceVelocityFactor = 0.75;
        preset.emitters![0]!.normalVelocity = -2;
        preset.sinks![0]!.mode = "delete";
        preset.sinks![0]!.targets = [];
        preset.scene = {
            encoding: "base64",
            glb: Buffer.from(glbBytes()).toString("base64"),
            collision: Buffer.from(collisionBytes()).toString("base64"),
        };

        const emitter = parseBlenderFluidJson(JSON.stringify(preset)).preset.emitters![0]!;
        expect(emitter.sourceVelocity).toEqual([1, 2, 3]);
        expect(emitter.sourceVelocityFactor).toBe(0.75);
        expect(emitter.normalVelocity).toBe(-2);
    });

    it("parses format-9 emitter source-node bindings", () => {
        const preset = validPreset();
        preset.formatVersion = 9;
        preset.emitters![0]!.sourceNode = "Animated Inflow";
        preset.emitters![0]!.sourceVelocityFactor = 0.75;
        preset.emitters![0]!.delayBeforeStart = 5;
        preset.sinks![0]!.mode = "recycle";
        preset.scene = {
            encoding: "base64",
            glb: Buffer.from(glbBytes()).toString("base64"),
            collision: Buffer.from(collisionBytes()).toString("base64"),
        };

        const emitter = parseBlenderFluidJson(JSON.stringify(preset)).preset.emitters![0]!;
        expect(emitter.sourceNode).toBe("Animated Inflow");
        expect(emitter.sourceVelocityFactor).toBe(0.75);
        expect(emitter.delayBeforeStart).toBe(5);
    });

    it("parses FLIP physics and preserves the selected method", () => {
        const preset = validPreset();
        preset.formatVersion = 11;
        preset.meta.method = "FLIP";
        preset.gridResolution = 160;
        preset.markersPerCell = 8;
        preset.physics = {
            gravity: 9.8,
            flipRatio: 0.95,
            kinematicViscosity: 0,
            surfaceTension: 0,
            minSubsteps: 2,
            maxSubsteps: 8,
            cflNumber: 2,
            restitution: 0,
            velocityDamping: 0,
            pressureSolver: 1,
            pressureIterations: 40,
            pressureRelaxation: 0.8,
            multigridCycles: 2,
            pressureTolerance: 0.001,
            pressureDiagnostics: 1,
            liquidSdf: 1,
            ghostFluid: 1,
            fractionalSolids: 1,
            movingSolidBoundaries: 1,
            reseedParticles: 1,
            reseedMinParticles: 4,
            reseedTargetParticles: 8,
            reseedMaxParticles: 12,
            reseedInterval: 5,
            particleSheeting: 1,
            sheetingStrength: 0.5,
            sheetingInterval: 5,
            polygonSurface: 1,
            viscosityIterations: 12,
            maxSubDtMs: 8.4,
        };

        const parsed = parseBlenderFluidJson(selfContainedJson(preset)).preset;

        expect(parsed.meta.method).toBe("FLIP");
        expect(parsed.physics).toEqual(preset.physics);
        expect(parsed.gridResolution).toBe(160);
        expect(parsed.markersPerCell).toBe(8);
    });

    it("parses advanced FLIP whitewater controls", () => {
        const preset = validPreset();
        Object.assign(preset.foam!, {
            generateSpray: false,
            generateFoam: true,
            generateBubbles: false,
            turbulenceRate: 25,
            energySpeedMin: 0.8,
            energySpeedMax: 7,
            curvatureMin: 0.1,
            curvatureMax: 2,
            turbulenceMin: 0.2,
            turbulenceMax: 3,
            foamLayerDepth: 1.5,
            sprayDrag: 0.6,
        });

        expect(parseBlenderFluidJson(selfContainedJson(preset)).preset.foam).toMatchObject({
            generateSpray: false,
            generateFoam: true,
            generateBubbles: false,
            turbulenceRate: 25,
            energySpeedMin: 0.8,
            energySpeedMax: 7,
            curvatureMin: 0.1,
            curvatureMax: 2,
            turbulenceMin: 0.2,
            turbulenceMax: 3,
            foamLayerDepth: 1.5,
            sprayDrag: 0.6,
        });
    });

    it("accepts particle counts above the former fixed ceiling", () => {
        const preset = validPreset();
        preset.particleCount = 2_500_000;

        expect(parseBlenderFluidJson(selfContainedJson(preset)).preset.particleCount).toBe(2_500_000);
    });

    it.each(["", "none"])('normalizes legacy foamDebug "%s" to "off"', (foamDebug) => {
        const preset = validPreset();
        preset.foam!.foamDebug = foamDebug;

        expect(parseBlenderFluidJson(selfContainedJson(preset)).preset.foam?.foamDebug).toBe("off");
    });

    it("preserves embedded scene payloads when re-exporting", () => {
        const scene = scenePayloadFromBlenderFluidJson(parseBlenderFluidJson(selfContainedJson(validPreset())));
        const glb = Buffer.from(scene.glb, "base64");
        const collision = parseBlenderFluidCollision(Buffer.from(scene.collision, "base64"));

        expect([...glb]).toEqual([...glbBytes()]);
        expect(collision.dims).toEqual([2, 2, 2]);
        expect([...collision.distances]).toEqual([-4, -3, -2, -1, 0, 1, 2, 3]);
        expect(scene.anchorPosition).toEqual([1, 2, 3]);
    });

    it("rejects truncated collision payloads", () => {
        expect(() => parseBlenderFluidCollision(collisionBytes().subarray(0, -4))).toThrow("payload length");
    });

    it.each([
        ["unknown bundle demo", (preset: FluidExportJson) => (preset.meta.demo = "waterfall"), 'meta.demo must be "blender"'],
        ["unknown solver method", (preset: FluidExportJson) => (preset.meta.method = "SPH"), "meta.method"],
        ["unsafe particle count", (preset: FluidExportJson) => (preset.particleCount = Number.MAX_SAFE_INTEGER + 1), "particleCount"],
        ["unbounded physics particle size", (preset: FluidExportJson) => (preset.physicsParticleSize = 9), "physicsParticleSize"],
        [
            "unbounded FLIP grid resolution",
            (preset: FluidExportJson) => {
                preset.meta.method = "FLIP";
                preset.physics = {
                    gravity: 9.8,
                    flipRatio: 0.95,
                    kinematicViscosity: 0,
                    surfaceTension: 0,
                    minSubsteps: 2,
                    maxSubsteps: 8,
                    cflNumber: 2,
                    restitution: 0,
                    velocityDamping: 0,
                    pressureIterations: 40,
                    pressureRelaxation: 0.8,
                    viscosityIterations: 12,
                    maxSubDtMs: 8.4,
                };
                preset.gridResolution = 2049;
            },
            "gridResolution",
        ],
        [
            "unbounded markers per cell",
            (preset: FluidExportJson) => {
                preset.meta.method = "FLIP";
                preset.physics = {
                    gravity: 9.8,
                    flipRatio: 0.95,
                    kinematicViscosity: 0,
                    surfaceTension: 0,
                    minSubsteps: 2,
                    maxSubsteps: 8,
                    cflNumber: 2,
                    restitution: 0,
                    velocityDamping: 0,
                    pressureIterations: 40,
                    pressureRelaxation: 0.8,
                    viscosityIterations: 12,
                    maxSubDtMs: 8.4,
                };
                preset.markersPerCell = 65;
            },
            "markersPerCell",
        ],
        ["negative simulation duration", (preset: FluidExportJson) => (preset.simulationDuration = -1), "simulationDuration"],
        ["unbounded alpha decay", (preset: FluidExportJson) => (preset.alphaDecay = 11), "alphaDecay"],
        ["unbounded simulation time scale", (preset: FluidExportJson) => (preset.simulationTimeScale = 101), "simulationTimeScale"],
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
        expect(() => parseBlenderFluidJson(selfContainedJson(preset))).toThrow(message);
    });
});
