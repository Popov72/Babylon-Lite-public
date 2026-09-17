import { spawnSync } from "child_process";
import { existsSync, readdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import { resolve } from "path";
import { pathToFileURL } from "url";
import * as ts from "typescript";
import { beforeAll, describe, expect, it } from "vitest";

const ROOT = resolve(__dirname, "../../..");
const PACKAGE_DIR = resolve(ROOT, "packages/babylon-lite");
const BUILD_DIR = resolve(PACKAGE_DIR, "build");
const DTS_PATH = resolve(BUILD_DIR, "index.d.ts");
const SOURCE_PACKAGE_JSON_PATH = resolve(PACKAGE_DIR, "package.json");
const PACKAGE_JSON_PATH = resolve(BUILD_DIR, "package.json");

function typescriptFilesUnder(directory: string): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
        const path = resolve(directory, entry.name);
        return entry.isDirectory() ? typescriptFilesUnder(path) : entry.isFile() && entry.name.endsWith(".ts") ? [path] : [];
    });
}

// Invoke binaries directly via their JS entry points and the current node
// executable, so the test does not depend on PATH (which may not contain
// pnpm/npx when launched from the VS Code Vitest extension).
const NODE = process.execPath;
const VITE_JS = resolve(PACKAGE_DIR, "node_modules/vite/bin/vite.js");
const TSC_JS = resolve(ROOT, "node_modules/typescript/bin/tsc");

// Build babylon-lite once for all build/* assertions in this file. The package
// build is two Vite passes: `--mode dist` emits the prebundled CDN tree and the
// shared rolled-up `index.d.ts`; `--mode lib` emits the module-granular tree and the
// publish-ready `package.json`. Both are required for the assertions below.
beforeAll(() => {
    rmSync(BUILD_DIR, { recursive: true, force: true });
    for (const mode of ["dist", "lib"]) {
        const build = spawnSync(NODE, [VITE_JS, "build", "--mode", mode], {
            cwd: PACKAGE_DIR,
            encoding: "utf-8",
        });
        if (build.status !== 0) {
            throw new Error(`babylon-lite build (--mode ${mode}) failed:\n${build.stdout ?? ""}${build.stderr ?? ""}`);
        }
    }
}, 300_000);

describe("build/index.d.ts", () => {
    it("does not export raw fluid backends or low-level GPU helpers at runtime", async () => {
        const api = (await import(`${pathToFileURL(resolve(BUILD_DIR, "lib/index.js")).href}?raw-fluid-export-check`)) as Record<string, unknown>;
        for (const name of [
            "createPbfSim",
            "createFlipSim",
            "createMlsMpmSim",
            "createPbMpmSim",
            "transferFlipSimState",
            "createParticleRenderTask",
            "createFluidSurfaceTask",
            "createFluidPolygonSurfaceTask",
            "createFluidRenderCompositor",
            "createFoamRenderTask",
            "createFluidProfiler",
            "GpuReadbackPool",
            "fluidSimulationBackendForSceneIntegration",
        ]) {
            expect(api, `${name} must not be a root runtime export`).not.toHaveProperty(name);
        }
    });

    it("requires at least one source for separate-file KTX2 arrays", () => {
        const probePath = resolve(BUILD_DIR, "ktx2-array-sources.probe.ts");
        try {
            writeFileSync(
                probePath,
                `import {
    loadKtx2Texture2DArrayFromUrls,
    uploadKtx2Texture2DArrayFromBuffers,
    type EngineContext,
} from "./index.js";
declare const engine: EngineContext;
declare const buffer: ArrayBuffer;
uploadKtx2Texture2DArrayFromBuffers(engine, [buffer]);
loadKtx2Texture2DArrayFromUrls(engine, ["layer.ktx2"]);
// @ts-expect-error Separate-file KTX2 arrays require at least one buffer.
uploadKtx2Texture2DArrayFromBuffers(engine, []);
// @ts-expect-error Separate-file KTX2 arrays require at least one URL.
loadKtx2Texture2DArrayFromUrls(engine, []);
`
            );
            const result = spawnSync(
                NODE,
                [
                    TSC_JS,
                    "--ignoreConfig",
                    "--noEmit",
                    "--strict",
                    "--target",
                    "es2022",
                    "--module",
                    "esnext",
                    "--moduleResolution",
                    "bundler",
                    "--lib",
                    "es2022,dom,dom.iterable",
                    "--types",
                    "webxr",
                    probePath,
                ],
                { cwd: PACKAGE_DIR, encoding: "utf-8" }
            );
            expect(result.status, `${result.stdout ?? ""}${result.stderr ?? ""}`).toBe(0);
        } finally {
            rmSync(probePath, { force: true });
        }
    });

    it("exposes standalone task population and opt-in RTT factories", () => {
        const probePath = resolve(BUILD_DIR, "render-task-opt-in.probe.ts");
        try {
            writeFileSync(
                probePath,
                `import {
    createSceneContext, createRenderTask, addMeshToTask, createRenderTargetTexture,
    createSurfaceRenderTargetTexture, onRenderTargetTextureResize, withSampledDepthTexture,
    type EngineContext, type Mesh,
} from "./index.js";
declare const engine: EngineContext;
declare const mesh: Mesh;
const fixed = createRenderTargetTexture(engine, {
    format: "rgba8unorm", samples: 1, size: { width: 32, height: 32 },
});
const fixedDepth = createRenderTargetTexture(engine, {
    dFormat: "depth32float", samples: 1, size: { width: 32, height: 32 },
}, withSampledDepthTexture);
const surface = createSurfaceRenderTargetTexture(engine, {
    format: "rgba8unorm", dFormat: "depth32float", samples: 1, size: engine,
}, withSampledDepthTexture);
const surfaceDepth = createSurfaceRenderTargetTexture(engine, {
    dFormat: "depth32float", samples: 1, size: engine,
}, withSampledDepthTexture);
const task = createRenderTask({ name: "explicit", rt: fixed.rt }, engine, createSceneContext(engine));
addMeshToTask(task, mesh);
// @ts-expect-error Task mesh population is a tree-shakable standalone API.
task.addMesh(mesh);
onRenderTargetTextureResize(surface, () => {})();
onRenderTargetTextureResize(surfaceDepth, () => {})();
fixedDepth.texture satisfies typeof fixedDepth.depthTexture;
`
            );
            const result = spawnSync(
                NODE,
                [
                    TSC_JS,
                    "--ignoreConfig",
                    "--noEmit",
                    "--strict",
                    "--target",
                    "es2022",
                    "--module",
                    "esnext",
                    "--moduleResolution",
                    "bundler",
                    "--lib",
                    "es2022,dom,dom.iterable",
                    "--types",
                    "webxr",
                    probePath,
                ],
                { cwd: PACKAGE_DIR, encoding: "utf-8" }
            );
            expect(result.status, `${result.stdout ?? ""}${result.stderr ?? ""}`).toBe(0);
        } finally {
            rmSync(probePath, { force: true });
        }
    });

    it("type-checks cleanly with no references to internal-only types", () => {
        expect(existsSync(DTS_PATH)).toBe(true);

        // Type-check the generated declaration file in isolation, without
        // skipLibCheck, so that any unresolved (e.g. internal-only) types
        // leaking into the public API surface are caught.
        //
        // `--ignoreConfig` is required under TypeScript 6: passing a file on the
        // command line while a tsconfig.json exists in cwd is now an error
        // (TS5112) unless config loading is explicitly skipped.
        //
        // WebGPU types come from TypeScript 6's built-in `dom` lib (which now
        // bundles them). The `@webgpu/types` package is intentionally NOT loaded
        // here: doing so duplicates those declarations and, without skipLibCheck,
        // trips TS6200/TS2717 conflicts between the package and the native lib.
        //
        // WebXR types are NOT in the `dom` lib, so `@types/webxr` (a declared
        // optional peer) is loaded via `--types webxr` to stand in for the
        // consumer's own compile path. The public WebXR API references ambient
        // WebXR globals (`XRSession`, `XRFrame`, `XRView`, `XRReferenceSpace`,
        // `XRProjectionLayer`, `XRSubImage`, ...) that the rollup treats as
        // consumer-provided, exactly like `@webgpu/types` — see the design note in
        // src/xr/xr-webgpu-binding.ts.
        const result = spawnSync(
            NODE,
            [
                TSC_JS,
                "--ignoreConfig",
                "--noEmit",
                "--strict",
                "--target",
                "es2022",
                "--module",
                "esnext",
                "--moduleResolution",
                "bundler",
                "--lib",
                "es2022,dom,dom.iterable",
                "--types",
                "webxr",
                DTS_PATH,
            ],
            {
                cwd: PACKAGE_DIR,
                encoding: "utf-8",
            }
        );

        const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
        if (result.status !== 0) {
            // Rewrite tsc's relative paths (e.g. "dist/index.d.ts(619,52):")
            // into absolute paths so they're clickable in the VS Code terminal
            // / test output panel.
            const clickable = output.replace(/(^|\s)(build[\\/][^\s(]+)\((\d+),(\d+)\)/g, (_m, lead: string, rel: string, line: string, col: string) => {
                const abs = resolve(PACKAGE_DIR, rel).replace(/\\/g, "/");
                return `${lead}${abs}:${line}:${col}`;
            });
            throw new Error(`build/index.d.ts has TypeScript errors (likely internal-only types leaking into the public API):\n${clickable}`);
        }
        expect(result.status).toBe(0);
    }, 300_000);

    it("supports public enum values with verbatimModuleSyntax", () => {
        const probePath = resolve(BUILD_DIR, "public-enums-verbatim.probe.ts");
        try {
            writeFileSync(
                probePath,
                `import {
    CharacterSupportedState,
    FgAnimationValueType,
    FgBlockType,
    FgEventType,
    FgType,
    PhysicsConstraintAxis,
    PhysicsConstraintType,
    PhysicsMotionType,
    PhysicsPrestepType,
    PhysicsShapeType,
} from "./index.js";

const publicEnumValues = [
    CharacterSupportedState.SUPPORTED,
    FgAnimationValueType.Quaternion,
    FgBlockType.NoOp,
    FgEventType.Start,
    FgType.Number,
    PhysicsConstraintAxis.LINEAR_X,
    PhysicsConstraintType.HINGE,
    PhysicsMotionType.DYNAMIC,
    PhysicsPrestepType.TELEPORT,
    PhysicsShapeType.SPHERE,
] as const;
void publicEnumValues;
`
            );

            const result = spawnSync(
                NODE,
                [
                    TSC_JS,
                    "--ignoreConfig",
                    "--noEmit",
                    "--strict",
                    "--target",
                    "es2022",
                    "--module",
                    "esnext",
                    "--moduleResolution",
                    "bundler",
                    "--verbatimModuleSyntax",
                    "true",
                    "--lib",
                    "es2022,dom,dom.iterable",
                    "--types",
                    "webxr",
                    probePath,
                ],
                {
                    cwd: PACKAGE_DIR,
                    encoding: "utf-8",
                }
            );

            const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
            expect(result.status, output).toBe(0);
        } finally {
            rmSync(probePath, { force: true });
        }
    });

    it("accepts public and mutable matrix representations", () => {
        const probePath = resolve(BUILD_DIR, "public-matrix-types.probe.ts");
        try {
            writeFileSync(
                probePath,
                `import { createIdentityMat4, setMat4Translation } from "./index.js";

const liteMatrix = createIdentityMat4();
const float32Matrix = new Float32Array(16);
const float64Matrix = new Float64Array(16);

const liteResult = setMat4Translation(liteMatrix, 1, 2, 3);
const float32Result = setMat4Translation(float32Matrix, 1, 2, 3);
const float64Result = setMat4Translation(float64Matrix, 1, 2, 3);

liteResult satisfies typeof liteMatrix;
float32Result satisfies Float32Array;
float64Result satisfies Float64Array;
`,
                "utf-8"
            );

            const result = spawnSync(
                NODE,
                [
                    TSC_JS,
                    "--ignoreConfig",
                    "--noEmit",
                    "--strict",
                    "--target",
                    "es2022",
                    "--module",
                    "esnext",
                    "--moduleResolution",
                    "bundler",
                    "--lib",
                    "es2022,dom,dom.iterable",
                    "--types",
                    "webxr",
                    probePath,
                ],
                {
                    cwd: PACKAGE_DIR,
                    encoding: "utf-8",
                }
            );

            const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
            expect(result.status, output).toBe(0);
        } finally {
            rmSync(probePath, { force: true });
        }
    });

    it("does not reference any external (npm) modules", () => {
        expect(existsSync(DTS_PATH)).toBe(true);

        const dts = readFileSync(DTS_PATH, "utf-8");

        // Collect every module specifier the .d.ts file refers to via:
        //   - top-level `import ... from "X"` declarations
        //   - top-level `export ... from "X"` re-exports
        //   - inline `import("X").Y` type expressions
        //   - triple-slash `<reference types="X" />` directives
        const specifiers = new Set<string>();
        for (const m of dts.matchAll(/(?:^|\n)\s*(?:import|export)[^;\n]*?\sfrom\s+["']([^"']+)["']/g)) {
            specifiers.add(m[1]!);
        }
        for (const m of dts.matchAll(/\bimport\s*\(\s*["']([^"']+)["']\s*\)/g)) {
            specifiers.add(m[1]!);
        }
        for (const m of dts.matchAll(/\/\/\/\s*<reference\s+types\s*=\s*["']([^"']+)["']/g)) {
            specifiers.add(m[1]!);
        }

        // Any specifier that is not a relative path is a leaked external type:
        // the rolled-up d.ts is supposed to be fully self-contained so that
        // consumers never need to install any of our build-time dependencies.
        const external = [...specifiers].filter((s) => !s.startsWith("./") && !s.startsWith("../"));
        expect(external, `build/index.d.ts leaks types from external modules: ${external.join(", ")}`).toEqual([]);
    });

    it("strips the shader-source brand so consumers can pass plain strings", () => {
        const dts = readFileSync(DTS_PATH, "utf-8");

        expect(dts).not.toContain("WgslSource");
        expect(dts).not.toContain("wgslSourceBrand");
        expect(dts).toContain("readonly vertexSource: string;");
        expect(dts).toContain("readonly fragmentSource: string;");

        const probePath = resolve(BUILD_DIR, "wgsl-source-types.probe.ts");
        try {
            writeFileSync(
                probePath,
                `import type { ShaderMaterialOptions } from "./index.js";
const options: ShaderMaterialOptions = {
    vertexSource: "plain consumer vertex WGSL",
    fragmentSource: "plain consumer fragment WGSL",
    attributes: [],
};
void options;
`,
                "utf-8"
            );
            const result = spawnSync(
                NODE,
                [
                    TSC_JS,
                    "--ignoreConfig",
                    "--noEmit",
                    "--strict",
                    "--target",
                    "es2022",
                    "--module",
                    "esnext",
                    "--moduleResolution",
                    "bundler",
                    "--lib",
                    "es2022,dom,dom.iterable",
                    "--types",
                    "webxr",
                    probePath,
                ],
                {
                    cwd: PACKAGE_DIR,
                    encoding: "utf-8",
                }
            );
            const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
            expect(result.status, output).toBe(0);
        } finally {
            rmSync(probePath, { force: true });
        }
    });

    it("exposes only the build-time moving-emitter provider API", () => {
        const dts = readFileSync(DTS_PATH, "utf-8");

        expect(dts).toContain("withNodeParticleEmitterProvider");
        expect(dts).toContain("withNodeParticleEmitterProvider<T extends object = BuildNodeParticleOptions>");
        expect(dts).toContain("options?: T & BuildNodeParticleOptions): T & BuildNodeParticleOptions;");
        expect(dts).toContain("buildNodeParticleSetWithEmitterProvider");
        expect(dts).not.toContain("enableNodeParticleEmitterProvider");
        expect(dts).not.toMatch(/\b_(?:capture|setup)Emitter\b|\b_emitterProvider\b|\bParticleEmitterState\b/);
    });

    it("exposes the graph normalizer without its internal runtime or marker", () => {
        const dts = readFileSync(DTS_PATH, "utf-8");

        expect(dts).toContain("normalizeNodeParticleGraph");
        expect(dts).toMatch(/normalizeNodeParticleGraph\(graph: ParticleGraph\): Promise<ParticleGraph>/);
        expect(dts).not.toContain("normalizeNodeParticleGraphRuntime");
        expect(dts).not.toContain("_isGraphPlumbingNormalized");
        expect(dts).not.toContain("_localVariableLoopEpoch");
    });

    it("exposes rigid-body rotation axis locks", () => {
        const dts = readFileSync(DTS_PATH, "utf-8");

        expect(dts).toContain('type PhysicsRotationAxis = "x" | "y" | "z"');
        expect(dts).toMatch(/lockPhysicsBodyRotationAxes\(world: PhysicsWorld, body: PhysicsBody, axes: readonly PhysicsRotationAxis\[\]\): void/);
        expect(dts).toMatch(/unlockPhysicsBodyRotationAxes\(world: PhysicsWorld, body: PhysicsBody, axes: readonly PhysicsRotationAxis\[\]\): void/);
    });

    it("exposes only GPU-safe public fluid declarations and hides low-level solvers", () => {
        const dts = readFileSync(DTS_PATH, "utf-8");
        expect(dts).toContain("createFluidSimulation");
        expect(dts).toContain("interface FluidSimulation");
        expect(dts).toContain("stepFluidSimulation");
        expect(dts).toContain("sampleMeshVolume");
        expect(dts).toContain("generateMeshSdf");

        const source = ts.createSourceFile(DTS_PATH, dts, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
        const declarations = source.statements.filter((statement) => {
            if (
                ts.isInterfaceDeclaration(statement) ||
                ts.isTypeAliasDeclaration(statement) ||
                ts.isFunctionDeclaration(statement) ||
                ts.isClassDeclaration(statement) ||
                ts.isEnumDeclaration(statement)
            ) {
                return !!statement.name && /fluid/i.test(statement.name.text);
            }
            if (ts.isVariableStatement(statement)) {
                return statement.declarationList.declarations.some((declaration) => ts.isIdentifier(declaration.name) && /fluid/i.test(declaration.name.text));
            }
            return false;
        });

        expect(declarations.length).toBeGreaterThan(0);
        for (const declaration of declarations) {
            const text = declaration.getText(source);
            for (const gpu of ["GPUBuffer", "GPUTexture", "GPUTextureView", "GPUSampler", "GPUDevice", "GPUCommandEncoder", "GPUQuerySet"]) {
                expect(text, `public fluid declaration leaks ${gpu}:\n${text}`).not.toContain(gpu);
            }
        }
        for (const internalName of [
            "FluidSim",
            "FluidSimBaseOptions",
            "FluidPolygonSurface",
            "FluidProfiler",
            "ForceFieldSpec",
            "SceneSdfSpec",
            "createPbfSim",
            "createFlipSim",
            "createMlsMpmSim",
            "createPbMpmSim",
            "createFloatingBodySystem",
            "createFluidSurfaceTask",
            "createFluidPolygonSurfaceTask",
            "createParticleRenderTask",
            "GpuReadbackPool",
            "GpuReadbackPoolOptions",
            "GpuReadbackSlot",
            "GpuReadbackState",
            "adoptFluidParticleChannel",
            "fluidParticleStreamForSceneIntegration",
            "fluidSimulationBackendForSceneIntegration",
            "fluidSimulationProfilerForSceneIntegration",
            "fluidSimulationRenderLayerDepthForSceneIntegration",
            "stepFluidSimulationForSceneIntegration",
        ]) {
            expect(dts, `${internalName} must be internal`).not.toMatch(new RegExp(`\\b(?:class|interface|function|type) ${internalName}\\b`));
        }
    });

    it("keeps public fluid runtime handles as pure state", () => {
        const dts = readFileSync(DTS_PATH, "utf-8");
        const source = ts.createSourceFile(DTS_PATH, dts, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
        const names = new Set(["FluidTimestepScheduler", "FluidControlsTransaction", "FluidFlowEditor", "FluidGpuHandle", "FluidControlsHandle", "FluidControlsBinding"]);
        const interfaces = source.statements.filter((statement): statement is ts.InterfaceDeclaration => ts.isInterfaceDeclaration(statement) && names.has(statement.name.text));

        expect(interfaces.map((declaration) => declaration.name.text).sort()).toEqual([...names].sort());
        for (const declaration of interfaces) {
            for (const member of declaration.members) {
                expect(ts.isMethodSignature(member), `${declaration.name.text} exposes a method`).toBe(false);
                if (ts.isPropertySignature(member) && member.type) {
                    expect(ts.isFunctionTypeNode(member.type), `${declaration.name.text}.${member.name.getText(source)} exposes attached behavior`).toBe(false);
                }
            }
        }
    });

    it("keeps lab demo imports on the single public package entry", () => {
        const labSource = resolve(ROOT, "lab/lite/src");
        const demoSource = resolve(labSource, "demos");
        expect(existsSync(resolve(labSource, "demos/aquanova/gpu-readback.ts"))).toBe(false);
        const aquanovaRuntime = readFileSync(resolve(labSource, "demos/aquanova/fluid-runtime.ts"), "utf-8");
        expect(aquanovaRuntime).toContain("createFluidParticleSpatialQuery");
        expect(aquanovaRuntime).not.toContain("GpuReadbackPool");
        expect(aquanovaRuntime).not.toMatch(/\bGPU(?:Buffer|Device|CommandEncoder|QuerySet)\b/);
        for (const file of typescriptFilesUnder(demoSource)) {
            const source = readFileSync(file, "utf-8");
            expect(source, file).not.toMatch(/(?:from\s+|import\()["']babylon-lite\//);
            expect(source, file).not.toMatch(/(?:from\s+|import\()\s*["'][^"']*packages[\\/]babylon-lite[\\/]src(?:[\\/]|["'])/);
        }
        for (const file of typescriptFilesUnder(labSource)) {
            const source = readFileSync(file, "utf-8");
            expect(source, `${file} duplicates the shared readback pool`).not.toMatch(/\bclass\s+GpuReadbackPool\b/);
        }
    });

    it("compiles a minimal root-import-only fluid runtime consumer", () => {
        const probePath = resolve(BUILD_DIR, "public-fluid-runtime.probe.ts");
        try {
            writeFileSync(
                probePath,
                `import {
    applyFluidControls,
    attachFluidSimulationRenderLayer,
    beginFluidSimulationProfilerFrame,
    bindFluidControls,
    commitFluidReconfiguration,
    configureFluidSimulationRenderLayer,
    configureFluidSimulationRenderCompositor,
    createFluidForceField,
    createFluidSceneSdf,
    createFluidSimulation,
    createFluidSimulationCollection,
    createFluidSimulationCollectionParticleStream,
    createFluidSimulationProfiler,
    createFluidSimulationRenderCompositor,
    disposeFluidSimulation,
    disposeFluidSimulationRenderCompositor,
    endFluidSimulationProfilerFrame,
    prepareFluidReconfiguration,
    prepareFluidReconfigurationUpdate,
    refreshFluidSimulationCollectionParticleStream,
    setFluidSimulationCollectionSources,
    readFluidSimulationPressureDiagnostics,
    readFluidSimulationProfiler,
    stepFluidSimulation,
    type Camera,
    type EngineContext,
    type FluidControlsBinding,
    type FluidControlsHandle,
    type FluidControlValues,
    type RenderTarget,
    type SceneContext,
} from "./index.js";

declare const engine: EngineContext;
declare const scene: SceneContext;
declare const camera: Camera;
declare const depth: RenderTarget;
declare const output: RenderTarget;
declare const controlsHandle: FluidControlsHandle;
declare const controlValues: FluidControlValues;

const options = {
    method: "PBF" as const,
    particleCount: 1_000,
    bounds: { min: [-1, 0, -1] as const, max: [1, 2, 1] as const },
    physicsScale: 2,
    physics: { restDensity: 341, relaxation: 50 },
};
const sceneSdf = createFluidSceneSdf(engine, {
    struct: "struct SceneSdfParams { bounds: vec4<f32>, };",
    sdf: "fn sceneSdf(p: vec3<f32>, dt: f32) -> f32 { return p.y + dt; }",
    params: new Float32Array(4),
});
const forceField = createFluidForceField(engine, {
    struct: "struct ForceFieldParams { force: vec4<f32>, };",
    wgsl: "fn externalForce(p: vec3<f32>, v: vec3<f32>, dt: f32) -> vec3<f32> { return v * dt; }",
    params: new Float32Array(4),
});
const profiler = createFluidSimulationProfiler(engine);
const simulation = createFluidSimulation(engine, { ...options, sceneSdf, forceField, profiler });
stepFluidSimulation(simulation, 1 / 60);
const prepared = prepareFluidReconfiguration(simulation, { ...options, particleCount: 2_000, sceneSdf, forceField, profiler });
commitFluidReconfiguration(prepared);
const updated = prepareFluidReconfigurationUpdate(simulation, { particleCount: 1_750 }, true);
commitFluidReconfiguration(updated);
const collection = createFluidSimulationCollection(engine);
setFluidSimulationCollectionSources(collection, [{ simulation }]);
const collectionStream = createFluidSimulationCollectionParticleStream(collection, scene, 2_000, { update: "manual" });
refreshFluidSimulationCollectionParticleStream(collectionStream);
const controls: FluidControlsBinding = bindFluidControls({
    controls: controlsHandle,
    target: simulation,
    deviceLimits: {
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
        maxBufferSize: 512 * 1024 * 1024,
        maxTextureDimension2D: 8192,
    },
    resolveTarget: (target) => ({
        simulation: target,
        options: { ...options, sceneSdf, forceField, profiler },
    }),
});
applyFluidControls(controls, controlValues);
const layer = attachFluidSimulationRenderLayer(simulation, {
    scene,
    camera,
    mode: "polygon",
    depthTarget: depth,
    backgroundTarget: output,
    outputTarget: output,
    profile: { polygonShader: "ocean", absorption: 1.5 },
});
configureFluidSimulationRenderLayer(layer, { enabled: true, opacity: 0.75, profile: { waterColor: "#4488aa" } });
const compositor = createFluidSimulationRenderCompositor(engine, { scene, baseColorTarget: output });
configureFluidSimulationRenderCompositor(compositor, [layer]);
beginFluidSimulationProfilerFrame(profiler);
endFluidSimulationProfilerFrame(profiler);
readFluidSimulationProfiler(profiler);
readFluidSimulationPressureDiagnostics(simulation);
disposeFluidSimulationRenderCompositor(compositor);
disposeFluidSimulation(simulation);
// @ts-expect-error raw solver state is intentionally unavailable
simulation.positionBuffer;
`
            );
            const result = spawnSync(
                NODE,
                [
                    TSC_JS,
                    "--ignoreConfig",
                    "--noEmit",
                    "--strict",
                    "--target",
                    "es2022",
                    "--module",
                    "esnext",
                    "--moduleResolution",
                    "bundler",
                    "--lib",
                    "es2022,dom,dom.iterable",
                    "--types",
                    "webxr",
                    probePath,
                ],
                { cwd: PACKAGE_DIR, encoding: "utf-8" }
            );
            expect(result.status, `${result.stdout ?? ""}${result.stderr ?? ""}`).toBe(0);
        } finally {
            rmSync(probePath, { force: true });
        }
    });

    it("expresses all three fluid host contracts without backend or raw GPU types", () => {
        const probePath = resolve(BUILD_DIR, "public-fluid-host-contract.probe.ts");
        const probe = `import {
    attachFluidSimulationCollectionRenderLayer,
    commitFluidReconfiguration,
    configureFluidSimulationRenderLayer,
    configureFluidSimulationRenderCompositor,
    createFluidParticleChannel,
    createFluidParticleSpatialQuery,
    createFluidRenderEnvironment,
    createFluidSimulationCollection,
    createFluidSimulationRenderCompositor,
    disposeFluidSimulation,
    fillFluidParticleChannel,
    getFluidSimulationCollectionDiagnostics,
    getFluidSimulationDiagnostics,
    importFluidPresetSession,
    planFluidInitialState,
    prepareFluidReconfiguration,
    readFluidParticleSpatialQuery,
    readFluidParticleChannel,
    readFluidSimulationPositions,
    refreshFluidSimulationCollectionPolygonSurfaces,
    resetFluidSimulation,
    sampleFluidParticleSpatialQuery,
    setFluidSimulationCollectionFlow,
    setFluidSimulationCollectionFoam,
    setFluidSimulationCollectionForceField,
    setFluidSimulationCollectionMaterial,
    setFluidSimulationCollectionParameter,
    setFluidSimulationCollectionProfiler,
    setFluidSimulationCollectionSceneSdf,
    setFluidSimulationCollectionSources,
    setFluidSimulationFlow,
    setFluidSimulationFoam,
    setFluidSimulationMaterial,
    setFluidSimulationParameter,
    stepFluidSimulation,
    stepFluidSimulationCollection,
    updateFluidSimulationEmitter,
    writeFluidSimulationPositions,
    type Camera,
    type EngineContext,
    type FluidExportJson,
    type FluidForceField,
    type FluidFlowConfig,
    type FluidRenderEnvironmentSource,
    type FluidPresetSession,
    type FluidSceneSdf,
    type FluidSimulation,
    type FluidSimulationOptions,
    type FluidSimulationProfiler,
    type PairState,
    type RenderTarget,
    type SceneContext,
} from "./index.js";

declare const engine: EngineContext;
declare const scene: SceneContext;
declare const camera: Camera;
declare const depth: RenderTarget;
declare const color: RenderTarget;
declare const simulations: FluidSimulation[];
declare const flow: FluidFlowConfig;
declare const sdf: FluidSceneSdf;
declare const force: FluidForceField;
declare const profiler: FluidSimulationProfiler;
declare const nextOptions: FluidSimulationOptions;
declare const preset: FluidExportJson;
declare const defaults: PairState;
declare const localProbeEnvironment: FluidRenderEnvironmentSource;

function driveWhiteboard(simulation: FluidSimulation): void {
    setFluidSimulationFlow(simulation, flow);
    updateFluidSimulationEmitter(simulation, flow.emitters[0]!);
    setFluidSimulationFoam(simulation, {});
    setFluidSimulationParameter(simulation, "gravity", -9.81);
    setFluidSimulationMaterial(simulation, 0);
    resetFluidSimulation(simulation);
    stepFluidSimulation(simulation, 1 / 60);
    refreshFluidSimulationCollectionPolygonSurfaces(createFluidSimulationCollection(engine, [simulation]));
    const prepared = prepareFluidReconfiguration(simulation, nextOptions, true);
    commitFluidReconfiguration(prepared);
    getFluidSimulationDiagnostics(simulation);
}

function driveLiquefactor(): void {
    const collection = createFluidSimulationCollection(engine, simulations);
    const alpha = createFluidParticleChannel(engine, { capacity: 600_000, components: 1 });
    const rgba = createFluidParticleChannel(engine, { capacity: 600_000, components: 4 });
    fillFluidParticleChannel(alpha, 1);
    void readFluidParticleChannel(rgba, { particleCount: 4 });
    setFluidSimulationCollectionSources(
        collection,
        simulations.map((simulation) => ({ simulation, alpha, color: rgba, opacity: 1 }))
    );
    const surface = attachFluidSimulationCollectionRenderLayer(collection, {
        scene,
        camera,
        mode: "surface",
        depthTarget: depth,
        backgroundTarget: color,
        outputTarget: color,
        particleCapacity: 600_000,
    });
    const polygon = attachFluidSimulationCollectionRenderLayer(collection, {
        scene,
        camera,
        mode: "polygon",
        depthTarget: depth,
        backgroundTarget: color,
        outputTarget: color,
    });
    attachFluidSimulationCollectionRenderLayer(collection, {
        scene,
        camera,
        mode: "foam",
        depthTarget: depth,
        colorTarget: color,
        surfaceLayer: polygon,
    });
    const compositor = createFluidSimulationRenderCompositor(engine, { scene, baseColorTarget: color, baseLayer: surface });
    configureFluidSimulationRenderCompositor(compositor, [polygon], surface);
    setFluidSimulationCollectionFoam(collection, {});
    setFluidSimulationCollectionParameter(collection, "gravity", -9.81);
    setFluidSimulationCollectionMaterial(collection, 0);
    setFluidSimulationCollectionSceneSdf(collection, sdf);
    setFluidSimulationCollectionForceField(collection, force);
    setFluidSimulationCollectionProfiler(collection, profiler);
    stepFluidSimulationCollection(collection, 1 / 60);
    getFluidSimulationCollectionDiagnostics(collection);
}

function driveAquanova(): void {
    const collection = createFluidSimulationCollection(engine, simulations);
    setFluidSimulationCollectionFlow(collection, flow, true);
    const layer = attachFluidSimulationCollectionRenderLayer(collection, {
        scene,
        camera,
        mode: "surface",
        depthTarget: depth,
        backgroundTarget: color,
        outputTarget: color,
        particleCapacity: 600_000,
    });
    const stream = layer.particleStream!;
    configureFluidSimulationRenderLayer(layer, { environment: createFluidRenderEnvironment(localProbeEnvironment) });
    configureFluidSimulationRenderLayer(layer, {
        environmentRotationY: Math.PI,
        surfaceMode: "ellipsoidDebug",
        particleVelocityBrighten: 0,
    });
    const query = createFluidParticleSpatialQuery(engine, { maximumQueries: 1024 });
    sampleFluidParticleSpatialQuery(query, stream, [{
        key: "electricity-receiver",
        offset: 0,
        count: stream.count,
        bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
        sphere: { origin: [0, 0, 0], radius: 4 },
    }]);
    readFluidParticleSpatialQuery(query);
    writeFluidSimulationPositions(simulations[0]!, new Float32Array(4));
    void readFluidSimulationPositions(simulations[0]!, { particleCount: 1 });
    disposeFluidSimulation(simulations[0]!);
}

const initial = planFluidInitialState({ particleCapacity: 1000, particleVolume: 0.001, flow });
const session: FluidPresetSession = importFluidPresetSession(preset, defaults);
void [driveWhiteboard, driveLiquefactor, driveAquanova, initial, session];
`;
        try {
            expect(probe).not.toMatch(/\bFluidSim\b|\bGPU(?:Buffer|Texture|TextureView|Sampler|Device|CommandEncoder|QuerySet)\b/);
            writeFileSync(probePath, probe);
            const result = spawnSync(
                NODE,
                [
                    TSC_JS,
                    "--ignoreConfig",
                    "--noEmit",
                    "--strict",
                    "--target",
                    "es2022",
                    "--module",
                    "esnext",
                    "--moduleResolution",
                    "bundler",
                    "--lib",
                    "es2022,dom,dom.iterable",
                    "--types",
                    "webxr",
                    probePath,
                ],
                { cwd: PACKAGE_DIR, encoding: "utf-8" }
            );
            expect(result.status, `${result.stdout ?? ""}${result.stderr ?? ""}`).toBe(0);
        } finally {
            rmSync(probePath, { force: true });
        }
    });

    it("exposes readonly rendering-context introspection without internal registries", () => {
        const dts = readFileSync(DTS_PATH, "utf-8");

        expect(dts).toMatch(/getRenderingContextKind\(context: RenderingContext(?:_\d+)?\): string/);
        expect(dts).toMatch(/getRenderingContexts\(surface: SurfaceContext\): readonly RenderingContext(?:_\d+)?\[\]/);
        expect(dts).not.toMatch(/\btype RenderingContextKind\b/);
        expect(dts).not.toMatch(/declare interface RenderingContext(?:_\d+)? \{[^}]*\bkind:/s);
        expect(dts).not.toContain("_renderingContextKind");
        expect(dts).not.toMatch(/^\s*_renderingContexts:/m);
        expect(dts).toMatch(/interface SceneContext extends RenderingContext(?:_\d+)? \{[^}]*name\?: string;/s);
    });

    it("exposes readonly material texture introspection without internal slots", () => {
        const dts = readFileSync(DTS_PATH, "utf-8");

        expect(dts).toMatch(/getMaterialTextures\(material: Material(?:_\d+)?\): readonly Texture2D(?:_\d+)?\[\]/);
        expect(dts).not.toMatch(/^\s*_textureSlots:/m);
    });

    it("rejects invalid emitter fields while preserving extended provider options", () => {
        const probePath = resolve(BUILD_DIR, "public-api-types.probe.ts");
        try {
            writeFileSync(
                probePath,
                `import { withNodeParticleEmitterProvider, type NodeParticleEmitterProvider } from "./index.js";
declare const provider: NodeParticleEmitterProvider;
// @ts-expect-error emitter remains Vec3-only when generic extension fields are accepted
withNodeParticleEmitterProvider(provider, { emitter: "not-a-vec3" });
const extended = withNodeParticleEmitterProvider(provider, { snippetServer: "https://example.invalid" });
const snippetServer: string = extended.snippetServer;
void snippetServer;
`
            );

            const result = spawnSync(
                NODE,
                [
                    TSC_JS,
                    "--ignoreConfig",
                    "--noEmit",
                    "--strict",
                    "--target",
                    "es2022",
                    "--module",
                    "esnext",
                    "--moduleResolution",
                    "bundler",
                    "--lib",
                    "es2022,dom,dom.iterable",
                    "--types",
                    "webxr",
                    probePath,
                ],
                {
                    cwd: PACKAGE_DIR,
                    encoding: "utf-8",
                }
            );

            const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
            expect(result.status, output).toBe(0);
        } finally {
            rmSync(probePath, { force: true });
        }
    });

    it("is the only declaration file in the published package", () => {
        const declarationFiles = readdirSync(BUILD_DIR, { recursive: true, encoding: "utf-8" })
            .filter((file) => file.endsWith(".d.ts"))
            .sort();
        expect(declarationFiles).toEqual(["index.d.ts"]);
    });
});

describe("build/package.json", () => {
    it("exposes only the root entry in source and published package manifests", () => {
        expect(existsSync(SOURCE_PACKAGE_JSON_PATH)).toBe(true);
        expect(existsSync(PACKAGE_JSON_PATH)).toBe(true);

        const sourcePkg = JSON.parse(readFileSync(SOURCE_PACKAGE_JSON_PATH, "utf-8")) as {
            exports?: Record<string, { import?: string; types?: string }>;
        };
        const publishedPkg = JSON.parse(readFileSync(PACKAGE_JSON_PATH, "utf-8")) as {
            exports?: Record<string, { import?: string; types?: string }>;
        };

        expect(sourcePkg.exports).toEqual({
            ".": {
                import: "./src/index.ts",
                types: "./src/index.ts",
            },
        });
        expect(publishedPkg.exports).toEqual({
            ".": {
                types: "./index.d.ts",
                import: "./lib/index.js",
            },
        });
    });

    it("declares no runtime dependencies and only strictly-optional allowlisted peers", () => {
        expect(existsSync(PACKAGE_JSON_PATH)).toBe(true);

        const pkg = JSON.parse(readFileSync(PACKAGE_JSON_PATH, "utf-8")) as Record<string, unknown>;

        // The published package must bundle every transitive *runtime* dep as an
        // opaque implementation detail, so `dependencies` is always empty and a
        // plain `npm i @babylonjs/lite` (or CDN usage) pulls in nothing else.
        expect(pkg.dependencies ?? {}).toEqual({});

        // A small, curated allowlist of OPTIONAL peer dependencies is permitted.
        // These are never bundled and — being optional — are never auto-installed
        // or warned about by npm/pnpm/yarn when the corresponding feature is unused:
        //   - @babylonjs/havok: injected by the caller into `createHavokWorld()`;
        //     Lite never imports it. The peer entry only advertises the supported range.
        //   - @webgpu/types: ambient/global types referenced by the public .d.ts;
        //     TypeScript consumers need them at compile time.
        //   - @types/webxr: ambient/global WebXR types referenced by the public
        //     .d.ts (the WebXR API); TypeScript consumers need them at compile time.
        // Every allowlisted peer MUST be marked optional. Keep this allowlist in sync
        // with `emitPackageJson()` in packages/babylon-lite/vite.config.ts.
        const ALLOWED_OPTIONAL_PEERS = ["@babylonjs/havok", "@webgpu/types", "@types/webxr"];
        const peers = (pkg.peerDependencies ?? {}) as Record<string, string>;
        const peerMeta = (pkg.peerDependenciesMeta ?? {}) as Record<string, { optional?: boolean }>;

        // The declared peers must be EXACTLY the allowlist: no unexpected peer may
        // leak in, and — just as importantly — the whole `peerDependencies` block
        // must not be accidentally dropped from `emitPackageJson()`, which would
        // silently regress the feature while still passing a subset check.
        expect(Object.keys(peers).sort()).toEqual([...ALLOWED_OPTIONAL_PEERS].sort());

        // ...and every one of them must be strictly optional so no package manager
        // errors or auto-installs when the corresponding feature is unused.
        for (const name of ALLOWED_OPTIONAL_PEERS) {
            expect(peerMeta[name]?.optional, `peer dependency '${name}' must be marked optional in peerDependenciesMeta`).toBe(true);
        }
    });
});
