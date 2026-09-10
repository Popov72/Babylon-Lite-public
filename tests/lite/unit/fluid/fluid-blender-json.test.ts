import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";
import { unzlibSync, zlibSync } from "fflate";

import { parseBlenderFluidCollision, parseBlenderFluidJson, scenePayloadFromBlenderFluidJson } from "../../../../packages/babylon-lite/src/fluid/authoring/blender-fluid-json";
import type { FluidExportJson } from "../../../../packages/babylon-lite/src/fluid/authoring/preset-io";

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

function concatenateBytes(...entries: Uint8Array[]): Uint8Array {
    const bytes = new Uint8Array(entries.reduce((sum, entry) => sum + entry.byteLength, 0));
    let offset = 0;
    for (const entry of entries) {
        bytes.set(entry, offset);
        offset += entry.byteLength;
    }
    return bytes;
}

describe("Blender FLIP Fluids exporter", () => {
    it("does not allocate mutable Sets at module import time", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/authoring/blender-fluid-json.ts"), "utf8");
        expect(source.slice(0, source.indexOf("function "))).not.toContain("new Set(");
    });

    it("exports the format-15 quality-comparison settings", () => {
        const source = readFileSync(resolve(process.cwd(), "scripts/blender-fluid-addon.py"), "utf8");
        expect(source).toContain('"version": (3, 9, 0)');
        expect(source).toContain('"formatVersion": 15');
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

    it("limits imported FLIP Fluids resolution to a runtime-compatible grid", () => {
        const source = readFileSync(resolve(process.cwd(), "scripts/blender-fluid-addon.py"), "utf8");
        expect(source).toContain("MAX_GRID_CELL_COUNT = 8 * 1024 * 1024");
        expect(source).toContain("resolution = compatible_flip_resolution(grid_size, raw_resolution)");
        expect(source).toContain("to fit Babylon Lite grid allocation limits");
    });

    it("exports Blender lights in Babylon Lite's unitless lighting mode", () => {
        const source = readFileSync(resolve(process.cwd(), "scripts/blender-fluid-addon.py"), "utf8");
        expect(source).toContain('export_import_convert_lighting_mode="COMPAT"');
    });

    it("maps native Mantaflow liquid domains to FLIP discretization", () => {
        const source = readFileSync(resolve(process.cwd(), "scripts/blender-fluid-addon.py"), "utf8");
        expect(source).toContain("marker_axis = max(1, int(settings.particle_number))");
        expect(source).toContain("markers_per_cell = int(clamp(marker_axis**3, 1, 64))");
        expect(source).toContain('"flipRatio": flip_ratio');
        expect(source).not.toContain('"method": "PBF"');
    });

    it("offers explicit external-resource, light, decimation, and 2048-SDF options", () => {
        const source = readFileSync(resolve(process.cwd(), "scripts/blender-fluid-addon.py"), "utf8");
        expect(source).toContain("max=2048");
        expect(source).toContain('"encoding": "external"');
        expect(source).toContain("export_lights=export_lights");
        expect(source).toContain('obj.modifiers.new("Babylon Lite export decimation", "DECIMATE")');
        expect(source).toContain("target_triangles - fixed_triangles");
        expect(source).toContain("MIN_DECIMATABLE_TRIANGLES = 256");
        expect(source).not.toContain("MAX_EXPORT_FLIP_PARTICLES");
        expect(source).not.toContain("MAX_EXPORT_SUBDIVISION_LEVEL");
        expect(source).toContain("FLIP_FLUIDS_GENERATED_OBJECTS");
        expect(source).toContain("blitefluid_animated_sdf_resolution");
        expect(source).toContain("def bake_local_collision(");
        expect(source).toContain('"animatedCollisions"');
        expect(source).toContain('"collisionByteLength": len(collision)');
        expect(source).toContain('"sourcePresentation": bool(not obj.hide_render and obj.visible_get())');
        expect(source).toContain("def is_emitter_source_object(obj):");
        expect(source).toContain("blitefluid_target_initial_particles");
        expect(source).toContain("blitefluid_target_inflow_particles");
        expect(source).toContain("def flow_emitter_counts(scene):");
        expect(source).toContain("def target_flip_resolution(");
        expect(source).toContain("def clipped_initial_particle_count(");
        expect(source).toContain("def particle_target_preview(");
        expect(source).toContain("def authored_object_transform(obj):");
        expect(source).toContain('zlib.compress(sdf_container, level=6)');
        expect(source).toContain('"sdfCompression": "zlib"');
        expect(source).toContain("particle_count = max(1, initial_markers + target_inflow_particles)");
        expect(source).toContain("initial_row.enabled = initial_count > 0");
        expect(source).toContain("inflow_row.enabled = inflow_count > 0");
        expect(source).toContain("Resolution Divisions:");
        expect(source).toContain("total_particles:,} total");
        expect(source).toContain("def add_export_checker_textures(objects):");
        expect(source).toContain('"waterColor": fluid_render_color(scene, domain)');
        expect(source).toContain('"polygonSurface": 0');
        expect(source).toContain('"surfaceDepthBlur": 18');
        expect(source).toContain('"surfaceThicknessBlur": 6');
        expect(source).toContain('"narrowRangeDelta": 10');
    });

    it("builds collision unions without nearest-triangle sign ambiguity", () => {
        const source = readFileSync(resolve(process.cwd(), "scripts/blender-fluid-addon.py"), "utf8");
        expect(source).toContain("def build_collision_bvhs(objects):");
        expect(source).toContain("def point_inside_bvh(point, bvh, minimum, maximum, epsilon):");
        expect(source).toContain("return -nearest_distance if inside else nearest_distance");
    });

    it("exposes per-grid collision enablement and filtering in the Fluid host", () => {
        const runtime = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/core/fluid-runtime-bindings.ts"), "utf8");
        const demo = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/fluid.ts"), "utf8");
        expect(runtime).toContain("updateGridSettings");
        expect(runtime).toContain("entry.options.y > 0.5");
        expect(runtime).toContain("fn sampleNearestSdfGrid");
        expect(runtime).toContain("entry.options.y > 0.5 && elapsed > 1.0e-6");
        expect(runtime).toContain("new Uint32Array(Math.ceil(totalVoxels / 2))");
        expect(runtime).toContain("unpack2x16float");
        expect(runtime).toContain('sdfGridFormat: "packed-f16"');
        expect(demo).toContain('controls.makeSection("Collision"');
        expect(demo).toContain('checkbox("Trilinear filtering"');
        expect(demo).toContain('checkbox("Enabled"');
    });

    it("gives planar Blender flow objects one simulation-cell thickness", () => {
        const source = readFileSync(resolve(process.cwd(), "scripts/blender-fluid-addon.py"), "utf8");
        expect(source).toContain("def ensure_flow_shape_thickness(shape, transform, cell_size):");
        expect(source).toContain('ensure_flow_shape_thickness(shape, transform, derived["cell_size"])');
    });

    it("preserves offset prism emitter geometry instead of using an origin-centered box", () => {
        const source = readFileSync(resolve(process.cwd(), "scripts/blender-fluid-addon.py"), "utf8");
        expect(source).toContain('return {"type": "polygonPrism", "points": points, "thickness": float(size[axis])}');
        expect(source).toContain("position = obj.matrix_world @ (local_center or Vector((0, 0, 0)))");
        expect(source).toContain("shape, transform = flow_shape_transform_for(obj, grid_position)");
    });

    it("accepts format-13 self-contained exports", () => {
        const preset = validPreset();
        const json = JSON.parse(selfContainedJson(preset)) as FluidExportJson;
        json.formatVersion = 13;

        expect(parseBlenderFluidJson(JSON.stringify(json)).preset.formatVersion).toBe(13);
    });

    it("preserves hidden emitter source-node presentation metadata", () => {
        const preset = validPreset();
        preset.emitters![0]!.sourceNode = "Water";
        preset.emitters![0]!.sourcePresentation = false;

        expect(parseBlenderFluidJson(selfContainedJson(preset)).preset.emitters?.[0]?.sourcePresentation).toBe(false);
    });

    it("accepts explicit format-14 semantics and rejects a missing semantics record", () => {
        const json = JSON.parse(selfContainedJson(validPreset())) as FluidExportJson;
        json.formatVersion = 14;
        json.simulationSemantics = { version: 1, profile: "normalized-v1", pbfPhysics: "scale-adjusted" };
        expect(parseBlenderFluidJson(JSON.stringify(json)).preset.simulationSemantics).toEqual(json.simulationSemantics);

        delete json.simulationSemantics;
        expect(() => parseBlenderFluidJson(JSON.stringify(json))).toThrow(/simulationSemantics/);
    });

    it("accepts PB-MPM liquid viscosity up to 1.0", () => {
        const preset = validPreset();
        preset.meta.method = "PB-MPM";
        preset.physics = {
            gravity: 9.8,
            iterations: 5,
            liquidRelaxation: 1.5,
            liquidViscosity: 1,
            elasticityRatio: 0.3,
            elasticRelaxation: 0.3,
            frictionAngle: 35,
            plasticity: 0.8,
            restitution: 0,
            substeps: 3,
            maxSubDtMs: 8.4,
        };

        expect(parseBlenderFluidJson(selfContainedJson(preset)).preset.physics.liquidViscosity).toBe(1);
        preset.physics.liquidViscosity = 1.001;
        expect(() => parseBlenderFluidJson(selfContainedJson(preset))).toThrow(/liquidViscosity/);
    });

    it("accepts zero MLS-MPM ground damping", () => {
        const preset = validPreset();
        preset.meta.method = "MLS-MPM";
        preset.physics = {
            gravity: 9.8,
            stiffness: 60,
            viscosity: 0.01,
            restDensity: 10,
            damping: 0.998,
            affineDamping: 1,
            groundDamp: 0,
            groundDampHeight: 0,
            restitution: 0,
            substeps: 2,
            maxSubDtMs: 8.4,
        };

        expect(parseBlenderFluidJson(selfContainedJson(preset)).preset.physics.groundDamp).toBe(0);
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

    it("loads external GLB and SDF resources and re-embeds them for UI export", () => {
        const preset = validPreset();
        preset.formatVersion = 13;
        preset.scene = {
            encoding: "external",
            glb: "external-scene.glb",
            collision: "external-scene.sdf",
            anchorPosition: [1, 2, 3],
        };
        const resources = new Map<string, ArrayBufferLike>([
            ["external-scene.glb", glbBytes().buffer],
            ["external-scene.sdf", collisionBytes().buffer],
        ]);

        const parsed = parseBlenderFluidJson(JSON.stringify(preset), resources);
        const embedded = scenePayloadFromBlenderFluidJson(parsed);
        const external = scenePayloadFromBlenderFluidJson(parsed, { preserveExternal: true });

        expect(parsed.sceneGlb.byteLength).toBe(glbBytes().byteLength);
        expect(parsed.collision.dims).toEqual([2, 2, 2]);
        expect(external).toEqual({ ...preset.scene, collisionEnabled: true, collisionTrilinear: true });
        expect(embedded.encoding).toBe("base64");
        expect(embedded.sdfCompression).toBe("zlib");
        expect(Buffer.from(embedded.glb, "base64")).toEqual(Buffer.from(glbBytes()));
        expect(Buffer.from(unzlibSync(Buffer.from(embedded.collision, "base64")))).toEqual(Buffer.from(collisionBytes()));
    });

    it("loads and re-exports format-15 animated collision SDF resources", () => {
        const preset = validPreset();
        preset.formatVersion = 15;
        preset.simulationSemantics = { version: 1, profile: "normalized-v1", pbfPhysics: "scale-adjusted" };
        const staticSdf = collisionBytes();
        const animatedSdf = collisionBytes();
        const sdfContainer = concatenateBytes(staticSdf, animatedSdf);
        preset.scene = {
            encoding: "external",
            glb: "animated.glb",
            collision: "animated.sdf",
            sdfCompression: "zlib",
            collisionEnabled: false,
            collisionTrilinear: false,
            collisionByteLength: staticSdf.byteLength,
            animatedCollisions: [
                {
                    id: "animated-collision-0001",
                    node: "WaveMaker",
                    sdf: "animated.sdf",
                    byteOffset: staticSdf.byteLength,
                    byteLength: animatedSdf.byteLength,
                    space: "node-local",
                    resolution: 64,
                    bakeFrame: 150,
                    presentation: false,
                    enabled: false,
                    trilinear: false,
                },
            ],
        };
        const resources = new Map<string, ArrayBufferLike>([
            ["animated.glb", glbBytes().buffer],
            ["animated.sdf", zlibSync(sdfContainer).buffer],
        ]);

        const parsed = parseBlenderFluidJson(JSON.stringify(preset), resources);
        const embedded = scenePayloadFromBlenderFluidJson(parsed);
        const external = scenePayloadFromBlenderFluidJson(parsed, { preserveExternal: true });

        expect(parsed.animatedCollisions).toHaveLength(1);
        expect(parsed.animatedCollisions[0]?.node).toBe("WaveMaker");
        expect(parsed.collisionEnabled).toBe(false);
        expect(parsed.collisionTrilinear).toBe(false);
        expect(parsed.animatedCollisions[0]?.enabled).toBe(false);
        expect(parsed.animatedCollisions[0]?.trilinear).toBe(false);
        expect(parsed.animatedCollisions[0]?.collision.dims).toEqual([2, 2, 2]);
        expect(external).toEqual(preset.scene);
        expect(Buffer.from(unzlibSync(Buffer.from(embedded.animatedCollisions?.[0]?.sdf ?? "", "base64")))).toEqual(Buffer.from(collisionBytes()));
        expect(embedded.collisionByteLength).toBeUndefined();
        expect(embedded.animatedCollisions?.[0]?.byteOffset).toBeUndefined();
        expect(embedded.animatedCollisions?.[0]?.byteLength).toBeUndefined();
    });

    it("rejects duplicate animated collision nodes", () => {
        const preset = validPreset();
        preset.formatVersion = 15;
        preset.simulationSemantics = { version: 1, profile: "normalized-v1", pbfPhysics: "scale-adjusted" };
        const sdf = Buffer.from(collisionBytes()).toString("base64");
        preset.scene = {
            encoding: "base64",
            glb: Buffer.from(glbBytes()).toString("base64"),
            collision: sdf,
            animatedCollisions: [
                { id: "first", node: "WaveMaker", sdf, space: "node-local", resolution: 64, bakeFrame: 1, presentation: true },
                { id: "second", node: "WaveMaker", sdf, space: "node-local", resolution: 64, bakeFrame: 1, presentation: true },
            ],
        };

        expect(() => parseBlenderFluidJson(JSON.stringify(preset))).toThrow(/node must be unique/);
    });

    it("reports missing external scene resources", () => {
        const preset = validPreset();
        preset.formatVersion = 13;
        preset.scene = { encoding: "external", glb: "scene.glb", collision: "scene.sdf" };

        expect(() => parseBlenderFluidJson(JSON.stringify(preset), new Map())).toThrow('external resource "scene.glb" was not provided');
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
        preset.sinks![0]!.delayBeforeStart = 3;
        preset.scene = {
            encoding: "base64",
            glb: Buffer.from(glbBytes()).toString("base64"),
            collision: Buffer.from(collisionBytes()).toString("base64"),
        };

        const parsed = parseBlenderFluidJson(JSON.stringify(preset)).preset;
        const emitter = parsed.emitters![0]!;
        expect(emitter.sourceNode).toBe("Animated Inflow");
        expect(emitter.sourceVelocityFactor).toBe(0.75);
        expect(emitter.delayBeforeStart).toBe(5);
        expect(parsed.sinks![0]!.delayBeforeStart).toBe(3);
    });

    it("parses FLIP physics and preserves the selected method", () => {
        const preset = validPreset();
        preset.formatVersion = 11;
        preset.meta.method = "FLIP";
        preset.gridResolution = 160;
        preset.markersPerCell = 8;
        delete preset.physicsParticleSize;
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
            polygonReconstructionMultiplier: 1.75,
            viscosityIterations: 12,
            maxSubDtMs: 8.4,
        };

        const parsed = parseBlenderFluidJson(selfContainedJson(preset)).preset;

        expect(parsed.meta.method).toBe("FLIP");
        expect(parsed.physics).toEqual(preset.physics);
        expect(parsed.gridResolution).toBe(160);
        expect(parsed.markersPerCell).toBe(8);
        expect(parsed.physicsParticleSize).toBeUndefined();
    });

    it("rejects an out-of-range FLIP polygon reconstruction multiplier", () => {
        const preset = validPreset();
        preset.meta.method = "FLIP";
        preset.gridResolution = 160;
        preset.markersPerCell = 8;
        delete preset.physicsParticleSize;
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
            polygonReconstructionMultiplier: 2.25,
            viscosityIterations: 12,
            maxSubDtMs: 8.4,
        };

        expect(() => parseBlenderFluidJson(selfContainedJson(preset))).toThrow("polygonReconstructionMultiplier");
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
