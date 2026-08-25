import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import {
    estimateFlipGpuBytes,
    FLIP_DEFAULT_PAGE_CAPACITY,
    FLIP_PAGE_CELLS,
    FLIP_PAGE_SIZE,
    flipMacFaceBufferBytes,
    pagedFlipStorageCounts,
} from "../../../../packages/babylon-lite/src/fluid/flip-sim";

describe("FLIP GPU memory estimate", () => {
    it("matches the solver buffer layout", () => {
        expect(estimateFlipGpuBytes(100, [4, 5, 6])).toBe(42_372);
    });

    it("adds 40 bytes for each particle slot", () => {
        expect(estimateFlipGpuBytes(101, [4, 5, 6]) - estimateFlipGpuBytes(100, [4, 5, 6])).toBe(40);
    });

    it("identifies the waterfall preset's per-binding resolution boundary", () => {
        const twoGiBWebGpuLimit = 2_147_483_644;
        expect(flipMacFaceBufferBytes([562, 281, 562])).toBe(2_135_105_440);
        expect(flipMacFaceBufferBytes([562, 281, 562])).toBeLessThan(twoGiBWebGpuLimit);
        expect(flipMacFaceBufferBytes([563, 282, 563])).toBe(2_150_322_200);
        expect(flipMacFaceBufferBytes([563, 282, 563])).toBeGreaterThan(twoGiBWebGpuLimit);
    });

    it("sizes bounded 8-cubed page pools with dedicated sentinel slots", () => {
        expect(FLIP_PAGE_SIZE).toBe(8);
        expect(FLIP_PAGE_CELLS).toBe(512);
        expect(FLIP_DEFAULT_PAGE_CAPACITY).toBe(8_000);
        expect(pagedFlipStorageCounts(8_000)).toEqual({ cells: 4_096_001, faces: 12_288_001 });
    });

    it("keeps a 600-division waterfall grid below one GiB with paging", () => {
        const paged = estimateFlipGpuBytes(300_000, [600, 288, 600], "jacobi", {
            pagedGrid: true,
            pagedGridMaxPages: 8_000,
        });
        expect(paged).toBe(767_360_860);
        expect(paged).toBeLessThan(1024 ** 3);
        expect(estimateFlipGpuBytes(300_000, [600, 288, 600])).toBeGreaterThan(17 * 1024 ** 3);
    });

    it("keeps dense storage as the default estimate", () => {
        expect(estimateFlipGpuBytes(100, [16, 16, 16])).toBe(estimateFlipGpuBytes(100, [16, 16, 16], "jacobi", { pagedGrid: false }));
    });

    it("includes the selected multigrid hierarchy", () => {
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "multigrid")).toBe(43_860);
        expect(estimateFlipGpuBytes(100, [16, 16, 16], "multigrid")).toBeGreaterThan(estimateFlipGpuBytes(100, [16, 16, 16]));
    });

    it("adds optional subcell buffers only when selected", () => {
        const base = estimateFlipGpuBytes(100, [4, 5, 6]);
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", { pressureDiagnostics: true }) - base).toBe(552);
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", { liquidSdf: true }) - base).toBe(960);
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", { fractionalSolids: true }) - base).toBe(3_472);
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", { liquidSdf: true, fractionalSolids: true }) - base).toBe(4_432);
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", { reseedParticles: true }) - base).toBe(832);
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", { particleSheeting: true }) - base).toBe(1_376);
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", { polygonSurface: true }) - base).toBe(13_648);
        expect(estimateFlipGpuBytes(100, [4, 5, 6], "jacobi", { polygonSurface: true, polygonReconstructionMultiplier: 2 }) - base).toBe(147_448);
    });
});

describe("FLIP particle dispatch", () => {
    it("flattens two-dimensional dispatches in every particle shader", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).not.toContain("let i = gid.x;");
    });

    it("dispatches paged cell and face passes over discovered pages", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("flip-page-dispatch");
        expect(source).toContain("dispatchWorkgroupsIndirect(args, offset)");
        expect(source).toContain("writeDispatch(0u, pages * ${FLIP_PAGE_CELLS}u)");
        expect(source).toContain("writeDispatch(3u, pages * ${FLIP_PAGE_CELLS * 3}u)");
        expect(source).toContain("clearPagedGrid(encoder)");
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

    it("restricts residuals over the complete coarse-cell volume", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        const coarseOperators = source.slice(source.indexOf("const MULTIGRID_SMOOTH_WGSL"), source.indexOf("const MULTIGRID_RESTRICT_WGSL"));
        const restriction = source.slice(source.indexOf("const MULTIGRID_RESTRICT_WGSL"), source.indexOf("const MULTIGRID_PROLONGATE_WGSL"));
        expect(coarseOperators).toContain("*diagonal += 1.0;");
        expect(coarseOperators).not.toContain("fluidFraction");
        expect(restriction).toContain("coarseFraction[i] = fractionSum / max(1.0, f32(childCount));");
        expect(restriction).toContain("coarseRhs[i] = 4.0 * residualSum / max(1.0, f32(childCount));");
        expect(restriction).toContain("coarseTypes[i] = select(CELL_AIR, CELL_SOLID, solidCount > 0u);");
        expect(restriction).not.toContain("coarseRhs[i] = 4.0 * residualSum / fractionSum;");
    });

    it("bounds projected-grid velocity and RK2 advection by the CFL speed", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        const project = source.slice(source.indexOf("function buildProjectWgsl"), source.indexOf("const EXTRAPOLATE_WGSL"));
        const g2p = source.slice(source.indexOf("function buildG2pWgsl"), source.indexOf("const SPEED_REDUCE_WGSL"));
        expect(project).toContain("let finiteProjected = select(0.0, rawProjected, rawProjected == rawProjected);");
        expect(project).toContain("let projected = clamp(finiteProjected, -p.solve.w, p.solve.w);");
        expect(g2p).toContain("let advectionVelocity = clampLength(pic, p.solve.w);");
        expect(g2p).toContain("let midpointVelocity = clampLength(sampleVector(midpoint, false), p.solve.w);");
        expect(g2p).toContain("var next = world + midpointVelocity * p.sim.x;");
    });

    it("keeps subcell liquid and solid geometry independently opt-in", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("const LIQUID_SDF_SCATTER_WGSL");
        expect(source).toContain("atomicMin(&orderedSdf");
        expect(source).toContain("let reconstructionRadius = max(p.solve.z, 0.75 * p.originDx.w)");
        expect(source).toContain("let stepDistance = length(vec3<f32>(offset)) * p.originDx.w");
        expect(source).toContain("function buildSolidFaceGeometryWgsl");
        expect(source).toContain("if (usesLiquidSdf())");
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
        expect(controls).toContain('control: "checkbox"');
        expect(controls).toContain("details.dataset.fluidPhysicsGroup = p.group");
        expect(controls).toContain("!dependencyApplies &&");
        expect(controls).toContain("input.checked = false;");
        expect(controls).toContain("on.onPhysicsParam?.(key, 0);");
    });

    it("samples pressure quality asynchronously and adapts multigrid within a cycle cap", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("const PRESSURE_DIAGNOSTIC_REDUCE_WGSL");
        expect(source).toContain("const POST_DIVERGENCE_DIAGNOSTIC_WGSL");
        expect(source).toContain("relativeResidual > pressureTolerance");
        expect(source).toContain("adaptiveMultigridCycles = Math.min(multigridCycles");
        expect(source).toContain(".mapAsync(GPUMapMode.READ)");
        expect(source).toContain("function ensurePressureDiagnosticResources()");
        expect(source).toContain("destroyPressureDiagnosticResources()");
        expect(source).toContain("pressureDiagnosticResources?.gpuBytes ?? 0");
        const controls = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/controls-panel.ts"), "utf8");
        expect(controls).toContain('key: "pressureTolerance"');
        expect(controls).toContain('key: "pressureDiagnostics"');
        expect(controls).toContain("Pressure residual:");
        expect(controls).toContain("Post-project divergence:");
    });

    it("keeps min-target-max marker reseeding lazy and recyclable", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("function buildReseedDeleteWgsl");
        expect(source).toContain("const RESEED_BUILD_WGSL");
        expect(source).toContain("function buildReseedEmitWgsl");
        expect(source).toContain("function ensureReseedResources()");
        expect(source).toContain("fluidDeleteParticle(i)");
        expect(source).toContain("fluidActivateParticle(donor)");
        expect(source).toContain("fn touchesFreeSurface");
        expect(source).toContain("let available = min(min(atomicLoad(&state[0]), atomicLoad(&state[4]))");
        expect(source).toContain("listCapacity + ticket], i");
        expect(source).toContain("if (ticket == 0xffffffffu)");
        expect(source).toContain("fluidActivateParticle(donor);");
        expect(source).toContain('dispatchCells(encoder, "flip-reseed-build"');
        expect(source.indexOf('dispatchCells(encoder, "flip-reseed-build"')).toBeLessThan(source.indexOf('dispatch(encoder, "flip-reseed-delete-overfull"'));
        expect(source.indexOf('dispatch(encoder, "flip-reseed-delete-overfull"')).toBeLessThan(source.indexOf('dispatch(encoder, "flip-reseed-delete-surplus"'));
        expect(source).toContain("if (sceneSdf(position, 0.0) < radius)");
        expect(source).toContain("reseedResources?.gpuBytes ?? 0");
        expect(source).toContain("reseedResources.state.destroy()");
        const controls = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/controls-panel.ts"), "utf8");
        for (const key of ["reseedParticles", "reseedMinParticles", "reseedTargetParticles", "reseedMaxParticles", "reseedInterval"]) {
            expect(controls).toContain(`key: "${key}"`);
        }
    });

    it("adds bounded GPU sheeting and indexed surface reconstruction", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        expect(source).toContain("const SHEETING_BUILD_WGSL");
        expect(source).toContain("fn isThinSheet");
        expect(source).toContain("fluidActivateParticle(i)");
        expect(source).toContain('dispatch(encoder, "flip-sheeting-emit"');
        expect(source).toContain("const SURFACE_NET_VERTEX_WGSL");
        expect(source).toContain("const SURFACE_NET_STABILIZE_WGSL");
        expect(source).toContain("const SURFACE_NET_INDEX_WGSL");
        expect(source).toContain("drawIndirect");
        expect(source).toContain("wireframeDrawIndirect");
        expect(source).toContain('dispatch(encoder, "flip-polygon-surface-stabilize"');
        expect(source).toContain("usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX");
        expect(source).toContain("usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX");
        expect(source).toContain("liquidSdfBuffer: stabilizedSdf");
        expect(source).toContain("get triangleCount(): number | undefined");
        expect(source).toContain("encodePolygonTriangleCountReadback");
        expect(source).toContain("copyBufferToBuffer(resources.surface.drawIndirect, 0");
        expect(source).toContain("gridOrigin: boundsMin");
        expect(source).toContain("gridDimensions: dimensions");
        expect(source).toContain("const POLYGON_SDF_UPSAMPLE_WGSL");
        expect(source).toContain("polygonReconstructionMultiplier");
        expect(source).toContain("refreshPolygonSurface(encoder: GPUCommandEncoder)");
        expect(source).toContain("polygonSurfaceRefreshPending = true;");
        expect(source).toContain("const dimensions = gridDim.map");
        expect(source).toContain('dispatch(encoder, "flip-polygon-sdf-upsample"');
        expect(source).toContain("fineSdf[i] = mix(z0, z1, weight.z);");
        expect(source).toContain("if (c.x == 0 && !lowXNegative && highXNegative)");
        expect(source).toContain("if (c.z == 0 && !lowZNegative && highZNegative)");
        expect(source).toContain("return mix(fine, coarse, 0.65);");
        expect(source).toContain("return liquidSdfEnabled || particleSheetingEnabled;");
        expect(source).toContain("return usesLiquidSdf() || polygonSurfaceEnabled;");
        expect(source).toContain('encodeLiquidSdf(encoder, "Surface")');
        expect(source).toMatch(/flip-polygon-surface-finalize[\s\S]*?resources\.finalizeBindGroup[\s\S]*?"Surface"/);
        const controls = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/controls-panel.ts"), "utf8");
        for (const key of ["particleSheeting", "sheetingStrength", "sheetingInterval", "polygonSurface"]) {
            expect(controls).toContain(`key: "${key}"`);
        }
        expect(controls).toContain('key: "polygonReconstructionMultiplier"');
        expect(controls).toContain('visibleWhen: { key: "polygonSurface", equals: 1 }');
        const renderer = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/polygon-surface-render.ts"), "utf8");
        expect(renderer).toContain("drawIndexedIndirect");
        expect(renderer).toContain("setSims(sims: readonly FluidSim[]): void");
        expect(renderer).toContain("for (const currentSim of sims)");
        expect(renderer).toContain("surfaceDepthView");
        expect(renderer).toContain("let geometricNormal = cross(dpdx(input.worldPosition), dpdy(input.worldPosition));");
        expect(renderer).toContain("if (dot(normal, viewDirection) > 0.0)");
        expect(renderer).toContain("vec3<f32>(reflectionDirection.x, reflectionDirection.y, -reflectionDirection.z)");
        expect(renderer).toContain("fn sampleLiquidSdf(worldPosition: vec3<f32>)");
        expect(renderer).toContain("fn traceLiquid(originView: vec3<f32>, directionView: vec3<f32>)");
        expect(renderer).toContain("fn traceOpaqueScene(");
        expect(renderer).toContain("fn sceneHitConfidence(");
        expect(renderer).not.toContain("fn stableSceneDepthAt(");
        expect(renderer).toContain("fn sampleHiZLinear(");
        expect(renderer).toContain("if (enteredLiquid && hitConfidence > 0.5)");
        expect(renderer).toContain("let boundedDistance = min(liquidHit.waterDistance, 4.0 * grid.originDx.w);");
        expect(renderer).toContain("return LiquidHit(entryUv, originView + directionView * distance");
        expect(renderer).not.toContain("physicalExitDirection");
        expect(renderer).not.toContain("let afterExit = traceOpaqueScene(");
        expect(renderer).toContain("textureSampleLevel(background, linearSampler, refractedUv, 0.0)");
        expect(renderer).toContain("let grazingReflectance = max(1.0 - roughness, f0);");
        expect(renderer).toContain("let tracedLiquidHit = traceLiquid(traceViewPosition, refractionView);");
        expect(renderer).toContain("var enteredLiquid = sampleLiquidSdf(entryWorld)");
        expect(renderer).toContain("if (!enteredLiquid && distance > 0.75 * grid.originDx.w)");
        expect(renderer).toContain("let transportNormal = normal;");
        expect(renderer).toContain("let entryBias = clamp(");
        expect(renderer).toContain("physicalRefraction = refract(viewDirection, transportNormal");
        expect(renderer).toContain("let minimumSheetThickness = clamp(");
        expect(renderer).toContain("let thickness = max(liquidHit.waterDistance, minimumSheetThickness);");
        expect(renderer).toContain("let transmittance = exp(");
        expect(renderer).toContain("var physicalRefraction = refract(");
        expect(renderer).toContain("let scatterAmount = vec3<f32>(1.0) - exp(");
        expect(renderer).toContain("u.colorAbsorption.rgb * scatterAmount * 0.28");
        expect(renderer).toContain("let reflectionHit = traceOpaqueScene(");
        expect(renderer).toContain("depthPyramid = createDepthPyramid");
        expect(renderer).toContain('depthPyramid!.build(depthSource, engine._currentEncoder, () => profiler?.pass("Surface"))');
        expect(renderer).toContain("{ binding: 7, resource: { buffer: surface.liquidSdfBuffer } }");
        const depthPyramid = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/frame-graph/depth-pyramid.ts"), "utf8");
        expect(depthPyramid).toContain("let extraX=(sd.x&1)==1");
        expect(depthPyramid).toContain("let extraY=(sd.y&1)==1");
        expect(renderer).toContain("let distribution = alpha2 /");
        expect(renderer).toContain("let normalVariance = min(");
        expect(renderer).toContain("sqrt(baseRoughness * baseRoughness + normalVariance)");
        expect(renderer).toContain("let color = clamp(");
        expect(renderer).toContain("@vertex fn wireVs(");
        expect(renderer).toContain("out.clip.z = min(out.clip.z + 1.0e-4 * out.clip.w, out.clip.w);");
        expect(renderer).toContain('entryPoint: "wireVs"');
        expect(renderer).toContain('primitive: { topology: "line-list" }');
        expect(renderer).toContain("surface.wireframeDrawIndirect");
        expect(controls).toContain('{ value: "polygonWireframe", label: "Polygon wireframe" }');
        expect(renderer).toContain("surface.gridSpacing");
        expect(renderer).toContain('export type FluidPolygonShading = "physical" | "ocean"');
        expect(renderer).toContain("if (u.render.z > 0.5)");
        expect(renderer).toContain("let referenceRoughness = 0.311;");
        expect(renderer).toContain("let distanceGloss = mix(");
        expect(renderer).toContain("let subsurfaceColor = vec3<f32>(0.1541919, 0.8857628, 0.990566);");
        expect(renderer).toContain("let bodyColor = u.colorAbsorption.rgb * 0.12;");
        expect(renderer).toContain("let oceanOpticalPath = mix(1.25, 4.0, 1.0 - facing);");
        expect(renderer).toContain("-max(u.colorAbsorption.w, 0.0) *");
        expect(renderer).not.toContain("max(u.colorAbsorption.w, 0.0) - 1.0");
        expect(renderer).toContain("waterColor = waterColor * oceanTransmittance;");
        expect(renderer).toContain("backgroundColor * oceanTransmittance +");
        expect(renderer).toContain("mix(transmittedWater, reflection, fresnel)");
        expect(renderer).toContain("setShadingMode(mode: FluidPolygonShading)");
        expect(renderer).toContain('cullMode: "back"');
        expect(renderer).not.toContain('cullMode: "front"');
        expect(renderer.match(/frontFace: "cw"/g)).toHaveLength(1);
        expect(controls).toContain("const polygonSurfaceBytes = polygonSurfaceEnabled ? px * 18 : 0;");
        expect(controls).toContain("onPolygonSurface?(enabled: boolean): void");
        expect(controls).toContain('key !== "polygonSurface" || on.onPolygonSurface !== undefined');
        expect(controls).toContain("on.onPolygonSurface?.(checkbox.checked)");
        expect(controls).toContain("setPolygonTriangleCount(count: number | undefined, visible: boolean)");
        expect(controls).toContain('polygonShader: "physical" | "ocean"');
        expect(controls).toContain('polygonShaderLabel.textContent = "Surface shader"');
        expect(controls).toContain('polygonShaderSelect.dataset.fluidSurfaceShader = "true"');
        expect(controls).toContain('polygonShaderSelect.dataset.fluidPolygonShader = "true"');
        expect(controls).toContain('{ value: "ocean", label: "Ocean PBR" }');
        expect(controls).toContain("Triangles:\\u00a0Calculating...");
        const foamRenderer = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/foam-render.ts"), "utf8");
        expect(foamRenderer).toContain("let polygonDepth = u.gains.w > 1.5;");
        expect(foamRenderer).toContain("orientationWeight = 1.0;");
        expect(foamRenderer).toContain("r * 1.5 * (polygonSurfaceDepth ? 2 : 1)");
        expect(foamRenderer).toContain("setPolygonSurfaceDepth(on: boolean)");
        const demo = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/fluid.ts"), "utf8");
        expect(demo).toContain('stage === "Surface" ? "Surface render" : stage');
        expect(demo).toContain('"Surface render", "Foam render"');
        expect(demo).toContain("foamTask.setPolygonSurfaceDepth(true)");
        expect(demo).toContain("polygonSurfaceTask.setEnvRotationY(rad)");
        expect(demo).toContain("polygonSurfaceTask.setEnvReflection(exposure, contrast)");
        expect(demo).toContain("polygonSurfaceTask.setFresnelF0(v)");
        expect(demo).toContain('polygonSurfaceTask.setWireframe(mode === "polygonWireframe")');
        expect(demo).toContain("onPolygonSurface: (enabled)");
        expect(demo).toContain("controls.setPolygonTriangleCount");
        expect(demo).toContain("activeSim.refreshPolygonSurface?.(engine._currentEncoder)");
        const aquanova = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/aquanova-fluid-sim.ts"), "utf8");
        expect(aquanova).toContain("createFluidPolygonSurfaceTask");
        expect(aquanova).toContain("polygonSurfaceTask.setSims(runningSims())");
        expect(aquanova).toContain("onPolygonSurface: () => syncPolygonSurfaceRendering()");
        expect(aquanova).toContain("polygonSurface: (phys.polygonSurface ?? 0) >= 0.5");
        expect(aquanova).toContain("particleSheeting: (phys.particleSheeting ?? 0) >= 0.5");
        expect(aquanova).toContain("reseedParticles: (phys.reseedParticles ?? 0) >= 0.5");
        expect(aquanova).toContain("pressureDiagnostics: (phys.pressureDiagnostics ?? 0) >= 0.5");
        expect(aquanova).toContain("polygonSurfaceTask.setEnvMap");
        expect(aquanova).toContain('polygonSurfaceTask.setWireframe(mode === "polygonWireframe")');
        expect(aquanova).toContain("polygonSurfaces.reduce");
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

    it("profiles iterative FLIP stages with a bounded query count", () => {
        const common = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/sim-common.ts"), "utf8");
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/flip-sim.ts"), "utf8");
        const profiler = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/fluid/gpu-profiler.ts"), "utf8");
        const demo = readFileSync(resolve(process.cwd(), "lab/lite/src/demos/fluid.ts"), "utf8");
        expect(common).toContain("stageSpan?(stage: string)");
        expect(source).toContain("const activeProfileSpans = new Set<string>();");
        expect(source).toContain('beginProfileSpan(encoder, "Simulation")');
        expect(source).toContain('beginProfileSpan(encoder, "Surface")');
        expect(source).toContain('beginProfileSpan(encoder, "Foam gen")');
        expect(source).toContain("activeProfileSpans.has(stage) ? undefined");
        expect(profiler).toContain("stageSpan(stage: string)");
        expect(profiler).toContain("begin: { querySet, beginningOfPassWriteIndex: p.begin }");
        expect(profiler).toContain("end: { querySet, endOfPassWriteIndex: p.end }");
        expect(demo).toContain("profiler!.stageSpan?.");
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
    it("supports the shared Ocean PBR shading mode", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/fluid-surface-render.ts"), "utf8");
        expect(source).toContain('export type FluidSurfaceShading = "physical" | "ocean"');
        expect(source).toContain("setShadingMode(mode: FluidSurfaceShading): void");
        expect(source).toContain('shadingMode === "ocean" ? 1 : 0');
        expect(source).toContain("let oceanMode = u.env.w > 0.5;");
        expect(source).toContain("let oceanOpticalPath = max(thickness, u.extra.w);");
        expect(source).toContain("let bodyScale = select(0.12, 0.35, meshColored);");
        expect(source).toContain("-max(density, 0.0) *");
        expect(source).not.toContain("max(density, 0.0) - 1.0");
        expect(source).toContain("var oceanRefractionDir = refract(rayDir, normal, ETA);");
        expect(source).toContain("oceanBackground * oceanTransmittance +");
        expect(source).toContain("mix(transmittedWater, reflectionColor, fresnel)");
        expect(source).toContain("fn reconstructViewNormal(");
        expect(source).toContain("normal * 4.0 + normalXp + normalXn + normalYp + normalYn");
        expect(source).toContain("(0.5 * outputTexel.x / depthTexel.x)");
        expect(source).toContain("0.5 * (dot(normalDx, normalDx) + dot(normalDy, normalDy))");
        expect(source).toContain("let reflViewDir = reflect(rayDir, reflectionNormal);");
        expect(source).toContain("let oceanRoughness = clamp(");
        expect(source).toContain("let oceanSpecular = min(");
    });

    it("does not allocate bind groups directly in the per-frame execute path", () => {
        const source = readFileSync(resolve(process.cwd(), "packages/babylon-lite/src/fluid/fluid-surface-render.ts"), "utf8");
        const execute = source.slice(source.indexOf("execute(): number {"), source.indexOf("dispose(): void", source.indexOf("execute(): number {")));
        expect(execute).not.toContain("device.createBindGroup");
        expect(source).toContain("const blurBindGroups = new Map<string, BlurBindGroupCacheEntry>()");
    });
});
