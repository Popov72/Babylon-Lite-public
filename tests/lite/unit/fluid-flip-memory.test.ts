import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { estimateFlipGpuBytes } from "../../../packages/babylon-lite/src/fluid/flip-sim";

describe("FLIP GPU memory estimate", () => {
    it("matches the solver buffer layout", () => {
        expect(estimateFlipGpuBytes(100, [4, 5, 6])).toBe(42_340);
    });

    it("adds 40 bytes for each particle slot", () => {
        expect(estimateFlipGpuBytes(101, [4, 5, 6]) - estimateFlipGpuBytes(100, [4, 5, 6])).toBe(40);
    });

    it("includes the selected multigrid hierarchy", () => {
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "multigrid")).toBe(43_348);
        expect(estimateFlipGpuBytes(100, [16, 16, 16], "multigrid")).toBeGreaterThan(estimateFlipGpuBytes(100, [16, 16, 16]));
    });

    it("adds optional subcell buffers only when selected", () => {
        const base = estimateFlipGpuBytes(100, [4, 5, 6]);
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", { liquidSdf: true }) - base).toBe(960);
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", { fractionalSolids: true }) - base).toBe(3_472);
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", { liquidSdf: true, fractionalSolids: true }) - base).toBe(4_432);
    });
});

describe("FLIP particle dispatch", () => {
    it("flattens two-dimensional dispatches in every particle shader", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).not.toContain("let i = gid.x;");
    });

    it("reuses the force bind group while only uniform contents change", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("forceBuffer !== spec.buffer");
    });

    it("appends finite inflow candidates with a budget-sized dispatch instead of scanning capacity", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("const FLOW_EMIT_APPEND_WGSL");
        expect(source).toContain('dispatch(encoder, "flip-flow-emit-append"');
        expect(source).toContain("Math.ceil(appendCount / WORKGROUP_SIZE)");
        expect(source).toContain("liveCount = Math.min(count, liveCount + appendCount)");
    });

    it("refills only under-occupied inflow cells independently of emission velocity", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("const FLOW_MARK_OCCUPANCY_WGSL");
        expect(source).toContain("fn fluidTryFillEmptySpace");
        expect(source).toContain("let markerTarget = max(1u, p.counts.y);");
        expect(source).toContain("if (current >= markerTarget)");
        expect(source).toContain("let launch = fluidPerParticleLaunch");
        expect(source.indexOf('dispatch(encoder, "flip-flow-mark-occupancy"')).toBeLessThan(source.indexOf('dispatch(encoder, "flip-flow-emit-append"'));
    });

    it("bounds uncapped inflow work by source capacity and skips full pools", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("fluidShapeVolume(emitter.shape, emitter.transform) / flowState.particleVolume");
        expect(source).toContain("const canRefillCapacity = flowFrame.deleteActive || flowState.activeCount < count;");
        expect(source).toContain("if (!flowOccupancyValid || flowFrame.deleteActive || releasedWarmupParticles)");
    });

    it("uses asynchronous previous-frame speed readback for adaptive CFL", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("atomicMax(&maxSpeedBits[0]");
        expect(source).toContain("void buffer");
        expect(source).toContain(".mapAsync(GPUMapMode.READ)");
        expect(source).toContain("const cflSteps =");
    });

    it("uploads warm-up seed ranges using typed-array element offsets", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("seedPositions, previous * 4, (liveCount - previous) * 4");
        expect(source).toContain("seedVelocities, previous * 4, (liveCount - previous) * 4");
    });

    it("keeps G2P within the portable compute storage-buffer limit", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        const g2p = source.slice(source.indexOf("function buildG2pWgsl"), source.indexOf("const SPEED_REDUCE_WGSL"));
        expect(g2p).not.toContain("debugSpeed");
        expect(g2p).not.toContain("maxSpeedBits");
        expect(source).toContain('dispatch(encoder, "flip-speed-reduce"');
    });

    it("skips physical material passes when their coefficients are zero", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("if (kinematicViscosity > 0 && viscosityIterations > 0)");
        expect(source).toContain("if (surfaceTension > 0)");
    });

    it("provides a lazy geometric multigrid pressure path alongside Jacobi", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain('export type FlipPressureSolver = "jacobi" | "multigrid"');
        expect(source).toContain("const MULTIGRID_RESTRICT_WGSL");
        expect(source).toContain("const MULTIGRID_PROLONGATE_WGSL");
        expect(source).toContain("function ensureMultigrid()");
        expect(source).toContain('if (pressureSolver === "multigrid")');
        expect(source).toContain("encodeMultigridPressure(encoder)");
        expect(source).toContain("multigridResources?.gpuBytes ?? 0");
        const controls = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/controls-panel.ts"), "utf8");
        expect(controls).toContain('key: "pressureSolver"');
        expect(controls).toContain('{ label: "Weighted Jacobi", value: 0 }');
        expect(controls).toContain('{ label: "Multigrid", value: 1 }');
        expect(controls).toContain('key: "multigridCycles"');
        expect(controls).toContain('visibleWhen: { key: "pressureSolver", equals: 0 }');
        expect(controls).toContain('visibleWhen: { key: "pressureSolver", equals: 1 }');
    });

    it("keeps subcell liquid and solid geometry independently opt-in", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("const LIQUID_SDF_SEED_WGSL");
        expect(source).toContain("function buildSolidFaceGeometryWgsl");
        expect(source).toContain("if (liquidSdfEnabled)");
        expect(source).toContain("if (fractionalSolidsEnabled)");
        expect(source).toContain("-gradient / magnitude");
        expect(source).toContain("magnitude / p.originDx.w");
        expect(source).toContain("function buildPressureResidualWgsl");
        expect(source).toContain("levelIndex === 0 ? multigridFineResidualPipeline");
        expect(source).toContain("if (leftType != CELL_FLUID && rightType != CELL_FLUID)");
        expect(source).toContain("velocity[i] = vec2<f32>(geometry.y, 0.0)");
        expect(source).toContain("liquidSdfResources?.gpuBytes ?? 0");
        expect(source).toContain("solidFaceResources?.gpuBytes ?? 0");
        const controls = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/controls-panel.ts"), "utf8");
        expect(controls).toContain('key: "liquidSdf"');
        expect(controls).toContain('key: "ghostFluid"');
        expect(controls).toContain('key: "fractionalSolids"');
        expect(controls).toContain('key: "movingSolidBoundaries"');
        expect(controls).toContain('{ label: "Off (fast)", value: 0 }');
    });

    it("keeps FLIP whitewater lazy and runs it once after the final substep", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("function ensureFoam(config: FoamConfig)");
        expect(source).toContain("if (surfaceTension > 0 || (foamEnabled && step === stepCount - 1))");
        expect(source.indexOf('dispatch(encoder, "flip-g2p"')).toBeLessThan(source.indexOf('dispatch(encoder, "flip-foam-emit"'));
        expect(source).toContain("foamF32[11] = frameDt");
        expect(source).toContain("setFoam(config: FoamConfig | null)");
        expect(source).toContain("get diffuse(): DiffusePool | undefined");
    });

    it("samples diffuse-particle usage asynchronously with per-kind workgroup reduction", () => {
        const common = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/sim-common.ts"), "utf8");
        expect(common).toContain("export interface DiffuseParticleCounts");
        expect(common).toContain("var<workgroup> localCounts: array<atomic<u32>, 4>");
        expect(common).toContain("frame % 30 !== 0");
        expect(common).toContain(".mapAsync(GPUMapMode.READ)");
        expect(common).toContain("pass.dispatchWorkgroupsIndirect(activeDispatch, 0)");
        expect(common).toContain("@group(0) @binding(2) var<storage, read_write> computeArgs");
        const controls = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/controls-panel.ts"), "utf8");
        expect(controls).not.toContain("Active foam particles");
        expect(controls).toContain("\\u00a0/\\u00a0");
        expect(controls.indexOf('"Generate foam"')).toBeLessThan(controls.indexOf('"Generate spray"'));
        expect(controls.indexOf('"Generate spray"')).toBeLessThan(controls.indexOf('"Generate bubbles"'));
        expect(controls).toContain("setFoamParticleCounts");
        const demo = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/fluid.ts"), "utf8");
        expect(demo).toContain("canvas.dataset.diffuseParticleCount");
        expect(demo).toContain("canvas.dataset.sprayParticleCount");
        expect(demo).toContain("canvas.dataset.foamParticleCount");
        expect(demo).toContain("canvas.dataset.bubbleParticleCount");
        for (const file of ["flip-sim.ts", "pbf-sim.ts", "mls-mpm-sim.ts", "pbmpm-sim.ts"]) {
            const source = readFileSync(resolve(process.cwd(), `packages/babylon-lite/src/fluid/${file}`), "utf8");
            expect(source).toContain("createDiffuseCountTracker");
            expect(source).toContain("foamCountTracker?.encode");
            expect(source).toContain("foamCountTracker?.gpuBytes");
            expect(source).toContain("foamCountTracker?.dispose()");
            expect(source).toMatch(/activeParticles \?\? true/);
        }
    });

    it("classifies foam only on fluid cells that touch air", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        const surface = source.slice(source.indexOf("const SURFACE_NORMAL_WGSL"), source.indexOf("const SURFACE_CURVATURE_WGSL"));
        const update = source.slice(source.indexOf("function buildFoamUpdateWgsl"), source.indexOf("function buildForceWgsl"));
        expect(surface).toContain("fn touchesAir");
        expect(surface).toContain("cellTypes[i] != CELL_FLUID || !touchesAir(c)");
        expect(update).toContain("fn sampleSurface");
        expect(update).toContain("let enterFoam =");
        expect(update).toContain("let keepFoam =");
        expect(update).toContain("var kind = 2u;");
        expect(update.indexOf("if (enterFoam || keepFoam)")).toBeLessThan(update.indexOf("else if (occupancy < 0.2)"));
    });

    it("keeps the FLIP foam emitter within eight storage buffers", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        const emitter = source.slice(source.indexOf("function buildFoamEmitWgsl"), source.indexOf("function buildFoamUpdateWgsl"));
        expect(emitter).toContain("@group(0) @binding(2) var<storage, read> faceVelocity");
        expect(emitter).toContain("@group(0) @binding(9) var<storage, read_write> lifecycle");
        expect(emitter).not.toContain("@binding(10)");
        expect(emitter).toContain("fn sampleCurvature");
        expect(emitter).toContain("fn sampleTurbulence");
        expect(emitter).toContain("let curl =");
        expect(emitter).toContain("let strainSq =");
        expect(emitter).toContain("if (foam.kTurb > 0.0)");
        expect(emitter).toContain("let topWeight = smoothstep");
        expect(emitter).toContain("let trappedAir =");
        expect(emitter).toContain("let waveCrest =");
        expect(emitter).toContain("foam.kTurb * iturb");
    });

    it("supports FLIP foam layers and aerodynamic spray drag", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        const update = source.slice(source.indexOf("function buildFoamUpdateWgsl"), source.indexOf("function buildForceWgsl"));
        expect(update).toContain("fn sampleFoamLayer");
        expect(update).toContain("foam.foamLayerDepth");
        expect(update).toContain("if (foam.sprayDrag > 0.0)");
        expect(update).toContain("velocity *= exp(-foam.sprayDrag * dt)");
        expect(source).toContain("foamF32[16] = config.kTurb ?? 0");
        expect(source).toContain("foamF32[24] = Math.max(0, config.foamLayerDepth ?? 0)");
        expect(source).toContain("foamU32[28] = config.generateSpray === false ? 0 : 1");
        expect(source).toContain("foamU32[29] = config.generateFoam === false ? 0 : 1");
        expect(source).toContain("foamU32[30] = config.generateBubbles === false ? 0 : 1");
        expect(update).toContain("if (!foamKindEnabled(kind))");
    });
});

describe("FLIP density drift correction", () => {
    it("does not apply compression expansion beside stationary solid cells", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("let nearSolid =");
        expect(source).toContain("if (nearSolid) {");
        expect(source).toContain("boundaryMoves = abs(sceneSdf(center, 0.002) - sceneSdf(center, 0.0)) > 1.0e-5;");
        expect(source).toContain("let suppressBoundaryCorrection = nearSolid && !boundaryMoves;");
        expect(source).toContain("let expansion = select(min(compression * 0.1");
        expect(source).toContain("0.0, suppressBoundaryCorrection);");
    });
});

describe("FLIP marker redistribution", () => {
    it("reseeds only overcrowded cells toward valid lower-density neighbours", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("fn markerRedistribution");
        expect(source).toContain("let crowded = u32(ceil(1.25 * f32(p.counts.y)));");
        expect(source).toContain("let moveProbability = min(0.25");
        expect(source).toContain("sceneSdf(neighbourCenter, 0.0) <= p.solve.z");
        expect(source).toContain("let refillEmpty = current > 2u * p.counts.y");
    });
});

describe("fluid surface bind groups", () => {
    it("does not allocate bind groups directly in the per-frame execute path", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/fluid-surface-render.ts"), "utf8");
        const execute = source.slice(source.indexOf("execute(): number {"), source.indexOf("dispose(): void", source.indexOf("execute(): number {")));
        expect(execute).not.toContain("device.createBindGroup");
        expect(source).toContain("const blurBindGroups = new Map<string, BlurBindGroupCacheEntry>()");
    });
});
