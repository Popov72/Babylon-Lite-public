import { readFileSync } from "node:fs";
import path from "node:path";

import { expect, test } from "../parity-fixtures";
import { waitForCanvasReady } from "../compare-utils";

function collisionBytes(dims: [number, number, number], origin: [number, number, number], cellSize: number): Uint8Array {
    const count = dims[0] * dims[1] * dims[2];
    const bytes = new Uint8Array(64 + count * 4);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x46534c42, true);
    view.setUint32(4, 1, true);
    view.setUint32(8, dims[0], true);
    view.setUint32(12, dims[1], true);
    view.setUint32(16, dims[2], true);
    view.setFloat32(24, origin[0], true);
    view.setFloat32(28, origin[1], true);
    view.setFloat32(32, origin[2], true);
    view.setFloat32(36, cellSize, true);
    for (let index = 0; index < count; index++) {
        view.setFloat32(64 + index * 4, 100, true);
    }
    return bytes;
}

function preset(gridSize: [number, number, number], particleCount: number, emitterCount: number, gridPosition: [number, number, number] = [1, 5, -2]): object {
    return {
        formatVersion: 5,
        meta: { demo: "blender", method: "PBF" },
        physics: { gravity: 9.8, viscosity: 0.08, relaxation: 50, scorr: 0.02, iterations: 3, restDensity: 341, boundaryDensity: 0 },
        demoParams: {},
        demoState: {},
        emitters: Array.from({ length: emitterCount }, (_, index) => ({
            id: `source-${index}`,
            name: `Source ${index}`,
            enabled: true,
            behavior: index === 0 ? "initial" : "inflow",
            transform: { position: [index * 2, 1, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "box", size: index === 0 ? [4, 4, 4] : [1, 1, 1] },
            sampling: "volume",
            velocity: [0, -1, 0],
            velocitySpace: "world",
            spread: 0,
            ...(index === 0 ? {} : { volumeRate: 2 }),
        })),
        sinks: [],
        showContainer: false,
        envIntensity: 1,
        msaa: true,
        activeBlocks: false,
        pagedGrid: false,
        fusedBlockDiscovery: false,
        physicsParticleSize: 1,
        gridPosition,
        gridSize,
        showGridBounds: false,
        particleCount,
        material: 0,
        render: {
            renderAsSpheres: true,
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

function positiveXAnimationGlb(): Buffer {
    const binary = Buffer.alloc(32);
    binary.writeFloatLE(0, 0);
    binary.writeFloatLE(4, 4);
    binary.writeFloatLE(0, 8);
    binary.writeFloatLE(0, 12);
    binary.writeFloatLE(0, 16);
    binary.writeFloatLE(3, 20);
    binary.writeFloatLE(0, 24);
    binary.writeFloatLE(0, 28);
    const gltf = {
        asset: { version: "2.0" },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ name: "MovingSource" }],
        buffers: [{ byteLength: binary.byteLength }],
        bufferViews: [
            { buffer: 0, byteOffset: 0, byteLength: 8 },
            { buffer: 0, byteOffset: 8, byteLength: 24 },
        ],
        accessors: [
            { bufferView: 0, componentType: 5126, count: 2, type: "SCALAR", min: [0], max: [4] },
            { bufferView: 1, componentType: 5126, count: 2, type: "VEC3", min: [0, 0, 0], max: [3, 0, 0] },
        ],
        animations: [
            {
                samplers: [{ input: 0, output: 1, interpolation: "LINEAR" }],
                channels: [{ sampler: 0, target: { node: 0, path: "translation" } }],
            },
        ],
    };
    const json = Buffer.from(JSON.stringify(gltf));
    const paddedJsonLength = (json.byteLength + 3) & ~3;
    const totalLength = 12 + 8 + paddedJsonLength + 8 + binary.byteLength;
    const glb = Buffer.alloc(totalLength, 0x20);
    glb.writeUInt32LE(0x46546c67, 0);
    glb.writeUInt32LE(2, 4);
    glb.writeUInt32LE(totalLength, 8);
    glb.writeUInt32LE(paddedJsonLength, 12);
    glb.writeUInt32LE(0x4e4f534a, 16);
    json.copy(glb, 20);
    const binaryHeader = 20 + paddedJsonLength;
    glb.writeUInt32LE(binary.byteLength, binaryHeader);
    glb.writeUInt32LE(0x004e4942, binaryHeader + 4);
    binary.copy(glb, binaryHeader + 8);
    return glb;
}

function jsonExport(
    gridSize: [number, number, number],
    particleCount: number,
    emitterCount: number,
    dims: [number, number, number],
    gridPosition?: [number, number, number],
    foamDebug = "off",
    glbSource: string | Buffer = "lab/public/gltf-assets/NegativeScaleTest/NegativeScaleTest.glb",
    sourceNode?: string
): Buffer {
    const data = preset(gridSize, particleCount, emitterCount, gridPosition) as Record<string, unknown>;
    data.formatVersion = 11;
    data.simulationTimeScale = 0.75;
    (data.foam as Record<string, unknown>).foamDebug = foamDebug;
    if (sourceNode) {
        const emitter = (data.emitters as Array<Record<string, unknown>>)[0]!;
        emitter.sourceNode = sourceNode;
        emitter.sourceVelocityFactor = 1;
        const missingSourceEmitter = (data.emitters as Array<Record<string, unknown>>)[1];
        if (missingSourceEmitter) {
            missingSourceEmitter.sourceNode = "Missing Source";
        }
    }
    data.scene = {
        encoding: "base64",
        glb: (typeof glbSource === "string" ? readFileSync(path.resolve(process.cwd(), glbSource)) : glbSource).toString("base64"),
        collision: Buffer.from(collisionBytes(dims, [-5, -4, -3], 0.5)).toString("base64"),
    };
    return Buffer.from(JSON.stringify(data));
}

test("Whiteboard restores the camera orbit and target from a fluid preset", async ({ page }) => {
    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });
    await page.locator('select:has(option[value="whiteboard"])').selectOption("whiteboard");
    const data = preset([10, 8, 12], 5_000, 0, [0, 4, 0]) as Record<string, unknown>;
    data.formatVersion = 11;
    data.meta = { demo: "whiteboard", method: "PBF" };
    data.camera = { alpha: 0.35, beta: 1.2, radius: 22, target: [3, 4, 5] };

    await page.locator('input[type="file"][accept*=".json"]').setInputFiles({
        name: "whiteboard-camera.json",
        mimeType: "application/json",
        buffer: Buffer.from(JSON.stringify(data)),
    });

    const canvas = page.locator("canvas");
    await expect(canvas).toHaveAttribute("data-camera-alpha", "0.35");
    await expect(canvas).toHaveAttribute("data-camera-beta", "1.2");
    await expect(canvas).toHaveAttribute("data-camera-radius", "22");
    await expect(canvas).toHaveAttribute("data-camera-target", "3,4,5");
});

test("FLIP initial-emitter information uses the domain-clipped reset allocation", async ({ page }) => {
    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });
    await page.locator('select:has(option[value="whiteboard"])').selectOption("whiteboard");
    await page.locator('input[type="file"][accept*=".json"]').setInputFiles({
        name: "clipped-initial.json",
        mimeType: "application/json",
        buffer: flipClippedInitialPreset(),
    });

    const canvas = page.locator("canvas");
    await expect(canvas).toHaveAttribute("data-method", "FLIP");
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count"))).toBe(32_768);
    await expect(page.locator("[data-fluid-initial-emitter-particle-count]")).toHaveText("32,768");
    await expect(canvas).toHaveAttribute("data-selected-initial-emitter-particle-count", "32768");
});

function lifecycleFlow(kind: "delete" | "inflow"): Buffer {
    const transform = { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] };
    return Buffer.from(
        JSON.stringify({
            formatVersion: 9,
            emitters:
                kind === "inflow"
                    ? [
                          {
                              id: "source",
                              name: "Source",
                              enabled: true,
                              behavior: "inflow",
                              transform,
                              shape: { type: "box", size: [2, 2, 2] },
                              sampling: "volume",
                              velocity: [0, 1, 0],
                              velocitySpace: "world",
                              sourceVelocity: [0.5, 0, 0],
                              sourceVelocityFactor: 2,
                              normalVelocity: 0.5,
                              spread: 0,
                              volumeRate: 100,
                          },
                      ]
                    : [],
            sinks:
                kind === "delete"
                    ? [
                          {
                              id: "drain",
                              name: "Drain",
                              enabled: true,
                              mode: "delete",
                              transform,
                              shape: { type: "box", size: [1_000, 1_000, 1_000] },
                              targets: [],
                          },
                      ]
                    : [],
        })
    );
}

function lifecyclePreset(method: "PBF" | "FLIP" | "MLS-MPM" | "PB-MPM", kind: "delete" | "inflow"): Buffer {
    const data = preset([12, 10, 14], 5_000, 0) as Record<string, unknown>;
    data.formatVersion = 9;
    data.meta = { demo: "blender", method };
    data.physics =
        method === "PBF"
            ? { gravity: 9.8, viscosity: 0.08, relaxation: 50, scorr: 0.02, iterations: 3, restDensity: 341, boundaryDensity: 0 }
            : method === "FLIP"
              ? {
                    gravity: 9.8,
                    flipRatio: 0.95,
                    kinematicViscosity: 0,
                    surfaceTension: 0,
                    minSubsteps: 1,
                    maxSubsteps: 8,
                    cflNumber: 2,
                    restitution: 0.1,
                    velocityDamping: 0,
                    pressureIterations: 20,
                    pressureRelaxation: 0.8,
                    viscosityIterations: 12,
                    maxSubDtMs: 16.7,
                }
              : method === "MLS-MPM"
                ? {
                      gravity: 9.8,
                      stiffness: 80,
                      viscosity: 0.01,
                      restDensity: 4,
                      damping: 0.999,
                      affineDamping: 0.99,
                      groundDamp: 0.9,
                      groundDampHeight: 0.5,
                      restitution: 0.1,
                      substeps: 1,
                      maxSubDtMs: 16.7,
                  }
                : {
                      gravity: 9.8,
                      iterations: 1,
                      liquidRelaxation: 0.1,
                      liquidViscosity: 0,
                      elasticityRatio: 1,
                      elasticRelaxation: 1,
                      frictionAngle: 60,
                      plasticity: 0.8,
                      restitution: 0.1,
                      substeps: 1,
                      maxSubDtMs: 16.7,
                  };
    const flow = JSON.parse(lifecycleFlow(kind).toString()) as { emitters: unknown[]; sinks: unknown[] };
    data.emitters = flow.emitters;
    data.sinks = flow.sinks;
    return Buffer.from(JSON.stringify(data));
}

function flipScenePreset(): Buffer {
    const data = JSON.parse(jsonExport([12, 10, 14], 5_000, 1, [3, 4, 5]).toString()) as Record<string, unknown>;
    data.formatVersion = 11;
    data.meta = { demo: "blender", method: "FLIP" };
    data.gridResolution = 56;
    data.markersPerCell = 8;
    data.physics = {
        gravity: 9.8,
        flipRatio: 0.95,
        kinematicViscosity: 0,
        surfaceTension: 0,
        minSubsteps: 1,
        maxSubsteps: 8,
        cflNumber: 2,
        restitution: 0.1,
        velocityDamping: 0,
        pressureIterations: 20,
        pressureRelaxation: 0.8,
        viscosityIterations: 12,
        maxSubDtMs: 16.7,
    };
    return Buffer.from(JSON.stringify(data));
}

function flipHighDensityPreset(): Buffer {
    const data = JSON.parse(flipScenePreset().toString()) as Record<string, unknown>;
    delete data.scene;
    data.markersPerCell = 32;
    return Buffer.from(JSON.stringify(data));
}

function flipLowDensityPreset(): Buffer {
    const data = JSON.parse(flipHighDensityPreset().toString()) as Record<string, unknown>;
    data.markersPerCell = 8;
    return Buffer.from(JSON.stringify(data));
}

function flipClippedInitialPreset(): Buffer {
    const data = JSON.parse(flipScenePreset().toString()) as Record<string, unknown>;
    delete data.scene;
    data.meta = { demo: "whiteboard", method: "FLIP" };
    data.particleCount = 64;
    data.gridPosition = [0, 0, 0];
    data.gridSize = [2, 2, 2];
    data.gridResolution = 16;
    data.emitters = [
        {
            id: "initial",
            name: "Initial",
            enabled: true,
            behavior: "initial",
            transform: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "box", size: [4, 4, 4] },
            sampling: "volume",
            velocity: [0, 0, 0],
            velocitySpace: "world",
            spread: 0,
        },
    ];
    data.sinks = [];
    return Buffer.from(JSON.stringify(data));
}

function flipClippedHighCountPreset(): Buffer {
    const data = JSON.parse(flipClippedInitialPreset().toString()) as Record<string, unknown>;
    data.particleCount = 600_000;
    data.gridSize = [4, 4, 4];
    data.gridResolution = 40;
    const emitter = (data.emitters as Array<Record<string, unknown>>)[0]!;
    emitter.shape = { type: "box", size: [6, 6, 6] };
    return Buffer.from(JSON.stringify(data));
}

function flipInflowOnlyPreset(): Buffer {
    const data = JSON.parse(flipScenePreset().toString()) as Record<string, unknown>;
    data.particleCount = 80_000;
    data.gridPosition = [0, 3, 0];
    data.gridSize = [10.5, 6, 10.5];
    data.gridResolution = 82;
    data.emitters = [
        {
            id: "inflow",
            name: "Inflow",
            enabled: true,
            behavior: "inflow",
            transform: { position: [0.64, 1.5, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "box", size: [2, 1, 2] },
            sampling: "volume",
            velocity: [3, 0, 0],
            velocitySpace: "world",
            spread: 0,
            volumeRate: 8,
        },
    ];
    data.sinks = [
        {
            id: "sink",
            name: "Sink",
            enabled: true,
            mode: "delete",
            transform: { position: [0, -2.7, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
            shape: { type: "box", size: [11.75, 1.25, 11.75] },
            targets: [],
        },
    ];
    return Buffer.from(JSON.stringify(data));
}

test("Whiteboard imports, re-exports, replaces, and clears Blender fluid JSON", async ({ page }) => {
    test.setTimeout(180_000);
    const errors: string[] = [];
    page.on("console", (message) => {
        if (message.type() === "error") {
            errors.push(message.text());
        }
    });
    page.on("pageerror", (error) => errors.push(error.message));

    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });
    await page.locator('select:has(option[value="whiteboard"])').selectOption("whiteboard");
    await expect(page.locator("canvas")).toHaveAttribute("data-demo", "whiteboard");

    const input = page.locator('input[type="file"][accept*=".json"]');
    await input.setInputFiles({
        name: "first.json",
        mimeType: "application/json",
        buffer: jsonExport([12, 10, 14], 5_000, 1, [3, 4, 5], [1, -1, -2], "", "lab/public/gltf-assets/NegativeScaleTest/NegativeScaleTest.glb", "NotShinyMinus1"),
    });
    await expect.poll(async () => (await page.locator("canvas").getAttribute("data-imported-bundle")) ?? errors.join("\n"), { timeout: 15_000 }).toBe("true");
    await expect(page.locator("canvas")).toHaveAttribute("data-method", "PBF");
    await expect(page.locator("canvas")).toHaveAttribute("data-grid-position", "1,-1,-2");
    await expect(page.locator("canvas")).toHaveAttribute("data-grid-size", "12,10,14");
    await expect(page.locator("canvas")).toHaveAttribute("data-simulation-ground-y", "-6");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-host-ground-hidden", "true");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-host-scene-cleared", "true");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-host-lights-disabled", "true");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-host-sky-hidden", "false");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-frame-clearing", "false");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-light-count", "0");
    await expect(page.locator("canvas")).toHaveAttribute("data-particle-count", "5000");
    const importedParticleCount = page.locator('select:has(option[value="5000"])');
    await expect(importedParticleCount).toHaveValue("5000");
    await expect(importedParticleCount.locator('option[value="5000"]')).toHaveText("5k");
    await expect(page.locator("canvas")).toHaveAttribute("data-emitter-count", "1");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-bound-emitter-count", "1");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-missing-emitter-source-count", "0");
    await expect.poll(async () => Number((await page.locator("canvas").getAttribute("data-imported-emitter-position"))?.split(",")[0])).toBeCloseTo(3);
    await expect.poll(async () => Number(await page.locator("canvas").getAttribute("data-active-particle-count"))).toBeGreaterThan(0);
    await expect.poll(async () => Number(await page.locator("canvas").getAttribute("data-simulation-opacity"))).toBeGreaterThan(0);
    await expect(page.locator("canvas")).toHaveAttribute("data-simulation-time-scale", "0.75");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-collision-dims", "3,4,5");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-collision-origin", "-5,-4,-3");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-scene-offset", "0,0,0");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-camera-framed", "true");
    await expect.poll(async () => Number(await page.locator("canvas").getAttribute("data-imported-camera-alpha"))).toBeCloseTo(Math.PI / 2);
    await expect.poll(async () => Number(await page.locator("canvas").getAttribute("data-imported-camera-radius"))).toBeGreaterThan(0);
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-camera-target", /.+/);
    expect(Number(await page.locator("canvas").getAttribute("data-imported-mesh-count"))).toBeGreaterThan(0);
    const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Export parameters" }).click()]);
    const downloadPath = await download.path();
    expect(downloadPath).not.toBeNull();
    const exported = JSON.parse(readFileSync(downloadPath!, "utf8")) as {
        meta: { demo: string };
        foam: { foamDebug: string };
        scene?: { encoding: string; glb: string; collision: string; anchorPosition?: [number, number, number] };
    };
    expect(exported.meta.demo).toBe("blender");
    expect(exported.foam.foamDebug).toBe("off");
    expect(exported.scene?.encoding).toBe("base64");
    expect(exported.scene?.anchorPosition).toEqual([1, -1, -2]);
    expect(Buffer.from(exported.scene!.glb, "base64").byteLength).toBeGreaterThan(0);
    const exportedCollision = Buffer.from(exported.scene!.collision, "base64");
    expect(exportedCollision.readUInt32LE(0)).toBe(0x46534c42);
    expect([exportedCollision.readUInt32LE(8), exportedCollision.readUInt32LE(12), exportedCollision.readUInt32LE(16)]).toEqual([3, 4, 5]);
    await page.locator("label").filter({ hasText: "Render as spheres" }).locator('input[type="checkbox"]').uncheck();
    const anisotropic = page.locator("label").filter({ hasText: "Anisotropic surface" }).locator('input[type="checkbox"]');
    await expect(anisotropic).toBeVisible();
    await anisotropic.check();
    await page.waitForTimeout(1_000);
    expect(errors.filter((message) => /WGSL|palpha|pipeline/i.test(message))).toEqual([]);
    const activeParticleCounter = page.locator('[data-fluid-particle-usage="true"] > div').first();
    await expect(activeParticleCounter).toBeVisible();
    await page.keyboard.press("p");
    await expect(page.locator("canvas")).toHaveAttribute("data-paused", "true");
    await expect(page.locator("canvas")).toHaveAttribute("data-active-particle-count", /^\d+$/);
    await expect
        .poll(async () => {
            const count = Number(await page.locator("canvas").getAttribute("data-active-particle-count"));
            return (await activeParticleCounter.textContent())?.replace(/\s/g, "").startsWith(`Particles:${count.toLocaleString("en-US")}active/`) === true;
        })
        .toBe(true);

    await input.setInputFiles({
        name: "second.json",
        mimeType: "application/json",
        buffer: jsonExport([16, 12, 18], 6_000, 2, [4, 3, 2], undefined, "off", positiveXAnimationGlb(), "MovingSource"),
    });
    await expect(page.locator("canvas")).toHaveAttribute("data-grid-size", "16,12,18");
    await expect(page.locator("canvas")).toHaveAttribute("data-particle-count", "6000");
    await expect(page.locator("canvas")).toHaveAttribute("data-emitter-count", "2");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-collision-dims", "4,3,2");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-animation-count", /^[1-9]\d*$/);
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-playing-animation-count", /^[1-9]\d*$/);
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-bound-emitter-count", "1");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-missing-emitter-source-count", "1");
    await expect
        .poll(async () => {
            const positionX = Number((await page.locator("canvas").getAttribute("data-imported-emitter-position"))?.split(",")[0]);
            const velocityX = Number((await page.locator("canvas").getAttribute("data-imported-emitter-source-velocity"))?.split(",")[0]);
            return positionX > 0.05 && velocityX > 0.05;
        })
        .toBe(true);
    const animationTime = async (): Promise<number> => Number(await page.locator("canvas").getAttribute("data-imported-animation-time"));
    await expect.poll(animationTime).toBeGreaterThan(1);
    await page.keyboard.press("r");
    await page.waitForTimeout(100);
    expect(await animationTime()).toBeLessThan(0.5);

    await expect.poll(animationTime).toBeGreaterThan(1);
    const beforeShiftKeyReset = await animationTime();
    await page.keyboard.press("Shift+R");
    await page.waitForTimeout(100);
    expect(await animationTime()).toBeGreaterThan(beforeShiftKeyReset);

    const resetButton = page.getByRole("button", { name: "Reset simulation" });
    await resetButton.click();
    await page.waitForTimeout(100);
    expect(await animationTime()).toBeLessThan(0.5);

    await expect.poll(animationTime).toBeGreaterThan(1);
    const beforeShiftClickReset = await animationTime();
    await resetButton.click({ modifiers: ["Shift"] });
    await page.waitForTimeout(100);
    expect(await animationTime()).toBeGreaterThan(beforeShiftClickReset);

    await page.locator('select:has(option[value="MLS-MPM"])').first().selectOption("MLS-MPM");
    await expect(page.locator("canvas")).toHaveAttribute("data-method", "MLS-MPM");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-bundle", "true");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-mesh-count", "0");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-collision-dims", "4,3,2");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-animation-count", /^[1-9]\d*$/);
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-bound-emitter-count", "1");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-missing-emitter-source-count", "1");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-camera-framed", "true");
    await page.evaluate(() => {
        const label = [...document.querySelectorAll("span")].find((candidate) => candidate.firstChild?.textContent === "Grid position");
        const input = label?.parentElement?.querySelectorAll<HTMLInputElement>('input[type="number"]')[0];
        if (!input) {
            throw new Error("Grid position X input was not found.");
        }
        input.value = "10.5";
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await expect(page.locator("canvas")).toHaveAttribute("data-grid-position", "10.5,5,-2");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-collision-origin", "4.5,-4,-3");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-scene-offset", "9.5,0,0");
    await expect
        .poll(async () => {
            const positionX = Number((await page.locator("canvas").getAttribute("data-imported-emitter-position"))?.split(",")[0]);
            const velocityX = Number((await page.locator("canvas").getAttribute("data-imported-emitter-source-velocity"))?.split(",")[0]);
            return positionX > 9.55 && Math.abs(velocityX) < 5;
        })
        .toBe(true);
    const [movedDownload] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Export parameters" }).click()]);
    const movedDownloadPath = await movedDownload.path();
    expect(movedDownloadPath).not.toBeNull();
    const movedExport = JSON.parse(readFileSync(movedDownloadPath!, "utf8")) as {
        gridPosition: [number, number, number];
        scene: { anchorPosition?: [number, number, number] };
    };
    expect(movedExport.gridPosition).toEqual([10.5, 5, -2]);
    expect(movedExport.scene.anchorPosition).toEqual([1, 5, -2]);
    await input.setInputFiles({
        name: "moved-reexport.json",
        mimeType: "application/json",
        buffer: readFileSync(movedDownloadPath!),
    });
    await expect(page.locator("canvas")).toHaveAttribute("data-grid-position", "10.5,5,-2");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-collision-origin", "4.5,-4,-3");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-scene-offset", "9.5,0,0");

    await page.locator('select:has(option[value="capsule"])').first().selectOption("capsule");
    await expect(page.locator("canvas")).toHaveAttribute("data-demo", "capsule");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-bundle", "false");
    await expect.poll(async () => Number(await page.locator("canvas").getAttribute("data-camera-alpha"))).toBeCloseTo(-Math.PI / 2);
    await expect.poll(async () => Number(await page.locator("canvas").getAttribute("data-camera-beta"))).toBeCloseTo(1.1);
    await expect(page.locator("canvas")).toHaveAttribute("data-camera-target", "0,6,0");
    await expect(page.locator("canvas")).toHaveAttribute("data-imported-mesh-count", "0");
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-collision-dims", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-collision-origin", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-scene-offset", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-animation-count", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-animation-time", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-playing-animation-count", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-bound-emitter-count", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-emitter-position", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-emitter-source-velocity", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-missing-emitter-source-count", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-host-ground-hidden", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-host-scene-cleared", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-host-lights-disabled", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-host-sky-hidden", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-light-count", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-frame-clearing", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-camera-framed", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-camera-alpha", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-camera-radius", /.+/);
    await expect(page.locator("canvas")).not.toHaveAttribute("data-imported-camera-target", /.+/);
    await expect.poll(async () => Number(await page.locator("canvas").getAttribute("data-camera-alpha"))).toBeCloseTo(-Math.PI / 2);
    await expect.poll(async () => Number(await page.locator("canvas").getAttribute("data-camera-beta"))).toBeCloseTo(1.1);
    await expect(page.locator("canvas")).toHaveAttribute("data-camera-target", "0,6,0");

    expect(errors.filter((message) => /destroyed|validation|out of memory|failed to import|WGSL|palpha|pipeline/i.test(message))).toEqual([]);
});

test("FLIP runs with an imported SDF collision grid", async ({ page }) => {
    test.setTimeout(120_000);
    const errors: string[] = [];
    page.on("console", (message) => {
        if (message.type() === "error" || message.type() === "warning") {
            errors.push(message.text());
        }
    });
    page.on("pageerror", (error) => errors.push(error.message));

    await page.goto("/demo-fluid.html?demo=box&method=PBF&quality=low");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });
    const canvas = page.locator("canvas");
    await page.locator('input[type="file"][accept*=".json"]').setInputFiles({
        name: "flip-scene.json",
        mimeType: "application/json",
        buffer: flipScenePreset(),
    });

    await expect
        .poll(
            async () =>
                errors.filter((message) => /destroyed|validation|out of memory|failed to import|shader|device|WGSL|pipeline/i.test(message)).join("\n") ||
                ((await canvas.getAttribute("data-imported-bundle")) === "true" ? "imported" : "waiting"),
            { timeout: 30_000 }
        )
        .toBe("imported");
    await expect(canvas).toHaveAttribute("data-method", "FLIP");
    const particleBufferLimit = Number(await canvas.getAttribute("data-device-particle-buffer-limit-bytes"));
    await expect(canvas).toHaveAttribute("data-device-particle-bytes-per-slot", "16");
    await expect(canvas).toHaveAttribute("data-device-particle-capacity", String(Math.floor(particleBufferLimit / 16)));
    await expect(canvas).toHaveAttribute("data-imported-collision-dims", "3,4,5");
    await expect(canvas).toHaveAttribute("data-grid-resolution", "56");
    await expect(canvas).toHaveAttribute("data-flip-markers-per-cell", "8");
    await expect(canvas).toHaveAttribute("data-particle-count", "32768");
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count")), { timeout: 15_000 }).toBe(32_768);
    await expect(page.getByText(/^Resolution divisions/)).toBeVisible();
    await expect(page.getByText(/^Markers per cell/)).toBeVisible();
    await expect(page.getByText(/^Particle capacity/)).toBeVisible();
    await expect(page.getByText(/^Physics particle size/)).toBeHidden();
    await expect(page.getByText(/^Particle usage/)).toBeVisible();
    await expect(page.getByText(/^Particles:\s*32,768\s*active\s*\/\s*32,768\s*capacity$/)).toBeVisible();
    await expect(page.getByText(/^Simulation GPU memory:\s*\d/)).toBeVisible();
    await expect(page.locator('[data-fluid-cell-size="true"] + [data-fluid-particle-usage="true"]')).toBeVisible();
    const currentParticleText = page.locator('[data-fluid-particle-usage="true"] > div').nth(0);
    const currentMemoryText = page.getByText(/^Simulation GPU memory:\s*\d/).first();
    expect(await currentParticleText.evaluate((element) => element.textContent)).toBe("Particles:\u00a032,768\u00a0active\u00a0/\u00a032,768\u00a0capacity");
    expect(await currentMemoryText.evaluate((element) => element.textContent)).toMatch(/^Simulation GPU memory:\u00a0\d+\.\d MiB$/);

    const currentGpuBytes = Number(await canvas.getAttribute("data-simulation-gpu-bytes"));
    const gridSizeZ = page.locator('[data-fluid-grid-vector="Grid size"] input').nth(2);
    await gridSizeZ.fill("12");
    await gridSizeZ.press("Enter");
    await expect(canvas).toHaveAttribute("data-grid-restart-pending", "true");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "32768");
    await expect(canvas).toHaveAttribute("data-restart-particle-required", "52035");
    await expect(canvas).toHaveAttribute("data-particle-count", "32768");
    await expect(canvas).toHaveAttribute("data-simulation-gpu-bytes", String(currentGpuBytes));
    await expect(page.getByText(/^After restart:\s*32,768\s*active\s*\/\s*32,768\s*capacity$/)).toBeVisible();

    await gridSizeZ.fill("14");
    await gridSizeZ.press("Enter");
    await expect(canvas).toHaveAttribute("data-grid-restart-pending", "false");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "32768");
    await expect(page.getByText(/^After restart:/)).toBeHidden();

    const resolutionInput = page.locator('[data-fluid-grid-resolution="true"] input[type="range"]');
    await resolutionInput.evaluate((input) => {
        (input as HTMLInputElement).value = "64";
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await expect(canvas).toHaveAttribute("data-grid-restart-pending", "true");
    await expect(canvas).toHaveAttribute("data-grid-resolution", "64");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "32768");
    await expect(canvas).toHaveAttribute("data-restart-particle-required", "48914");
    await expect(canvas).toHaveAttribute("data-particle-count", "32768");
    await expect(canvas).toHaveAttribute("data-simulation-gpu-bytes", String(currentGpuBytes));

    await resolutionInput.evaluate((input) => {
        (input as HTMLInputElement).value = "256";
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await expect(canvas).toHaveAttribute("data-grid-restart-pending", "true");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "32768");
    await expect(canvas).toHaveAttribute("data-restart-particle-required", "3130443");
    await expect(canvas).toHaveAttribute("data-restart-particle-capacity", "32768");
    await expect(canvas).toHaveAttribute("data-particle-count", "32768");

    await resolutionInput.evaluate((input) => {
        (input as HTMLInputElement).value = "400";
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await expect(canvas).toHaveAttribute("data-grid-resolution", "400");
    await expect(canvas).toHaveAttribute("data-grid-restart-pending", "true");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "32768");
    await expect(canvas).toHaveAttribute("data-restart-particle-required", "11941691");
    await expect(canvas).toHaveAttribute("data-particle-count", "32768");
    await expect(page.getByText(/When the simulation is restarted, Resolution divisions will be adjusted from\s*400\s*to\s*\d+/)).toBeVisible();

    await resolutionInput.evaluate((input) => {
        (input as HTMLInputElement).value = "56";
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await expect(canvas).toHaveAttribute("data-grid-restart-pending", "false");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "32768");
    await expect(page.getByText(/When the simulation is restarted, Resolution divisions will be adjusted/)).toBeHidden();

    const particleCapacityInput = page.locator('[data-fluid-flip-particle-capacity="true"] input');
    await particleCapacityInput.fill("65536");
    await particleCapacityInput.press("Enter");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "32768");
    await expect(canvas).toHaveAttribute("data-restart-particle-capacity", "65536");
    await expect(page.getByText(/^After restart:\s*32,768\s*active\s*\/\s*65,536\s*capacity$/)).toBeVisible();
    await expect.poll(async () => Number(await canvas.getAttribute("data-restart-simulation-gpu-bytes"))).toBeGreaterThan(currentGpuBytes);
    await expect(canvas).toHaveAttribute("data-particle-count", "32768");

    await page.getByRole("button", { name: "Reset simulation" }).click();
    await expect(canvas).toHaveAttribute("data-particle-count", "65536");
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count")), { timeout: 15_000 }).toBe(32_768);
    await expect(page.getByText(/^Particles:\s*32,768\s*active\s*\/\s*65,536\s*capacity$/)).toBeVisible();
    await expect(page.getByText(/^After restart:/)).toBeHidden();

    const emitterSizeX = page.locator('[data-flow-field-label="Size"] input').first();
    const capacityGpuBytes = await canvas.getAttribute("data-simulation-gpu-bytes");
    await emitterSizeX.fill("8");
    await emitterSizeX.press("Enter");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "65536");
    await expect(canvas).toHaveAttribute("data-restart-particle-capacity", "65536");
    await expect(page.getByText(/^After restart:\s*65,536\s*active\s*\/\s*65,536\s*capacity$/)).toBeVisible();

    await page.getByRole("button", { name: "Reset simulation" }).click();
    await expect(canvas).toHaveAttribute("data-particle-count", "65536");
    await expect(canvas).toHaveAttribute("data-simulation-gpu-bytes", capacityGpuBytes!);
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count")), { timeout: 15_000 }).toBe(65_536);
    await expect(page.getByText(/^Particles:\s*65,536\s*active\s*\/\s*65,536\s*capacity$/)).toBeVisible();
    await expect(page.getByText(/^After restart:/)).toBeHidden();

    const markersPerCellInput = page
        .getByText(/^Markers per cell/)
        .locator("..")
        .locator('input[type="number"]');
    const postEmitterGpuBytes = await canvas.getAttribute("data-simulation-gpu-bytes");
    await markersPerCellInput.fill("16");
    await markersPerCellInput.press("Enter");
    await expect(canvas).toHaveAttribute("data-grid-restart-pending", "true");
    await expect(canvas).toHaveAttribute("data-flip-markers-per-cell", "16");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "65536");
    await expect(canvas).toHaveAttribute("data-restart-particle-required", "131072");
    await expect(canvas).toHaveAttribute("data-particle-count", "65536");
    await expect(canvas).toHaveAttribute("data-simulation-gpu-bytes", postEmitterGpuBytes!);
    await expect(page.getByText(/initial fluid requires 131,072 particles, but Particle capacity is 65,536/)).toBeVisible();

    await particleCapacityInput.fill("131072");
    await particleCapacityInput.press("Enter");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "131072");
    await expect(canvas).toHaveAttribute("data-restart-particle-capacity", "131072");
    await expect(page.getByText(/^After restart:\s*131,072\s*active\s*\/\s*131,072\s*capacity$/)).toBeVisible();

    await page.getByRole("button", { name: "Reset simulation" }).click();
    await expect(canvas).toHaveAttribute("data-grid-restart-pending", "false");
    await expect(canvas).toHaveAttribute("data-particle-count", "131072");
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count")), { timeout: 15_000 }).toBe(131_072);
    await expect(page.getByText(/^Particles:\s*131,072\s*active\s*\/\s*131,072\s*capacity$/)).toBeVisible();
    await expect(page.getByText(/^After restart:/)).toBeHidden();

    const gridPositionX = page.locator('[data-fluid-grid-vector="Grid position"] input').first();
    await gridPositionX.fill("2.5");
    await gridPositionX.press("Enter");
    await expect(canvas).toHaveAttribute("data-grid-position", "2.5,5,-2");
    await expect(canvas).toHaveAttribute("data-grid-restart-pending", "true");
    await expect(canvas).toHaveAttribute("data-imported-collision-origin", "-5,-4,-3");
    await expect(canvas).toHaveAttribute("data-imported-scene-offset", "0,0,0");
    await expect(canvas).toHaveAttribute("data-particle-count", "131072");

    await page.getByRole("button", { name: "Reset simulation" }).click();
    await expect(canvas).toHaveAttribute("data-grid-restart-pending", "false");
    await expect(canvas).toHaveAttribute("data-imported-collision-origin", "-3.5,-4,-3");
    await expect(canvas).toHaveAttribute("data-imported-scene-offset", "1.5,0,0");
    await expect(canvas).toHaveAttribute("data-particle-count", "131072");

    const bounds = await canvas.boundingBox();
    expect(bounds).not.toBeNull();
    await page.keyboard.down("Shift");
    await page.mouse.move(bounds!.x + bounds!.width / 2, bounds!.y + bounds!.height / 2);
    await page.mouse.down({ button: "right" });
    await page.waitForTimeout(25);
    await page.mouse.move(bounds!.x + bounds!.width / 2 + 50, bounds!.y + bounds!.height / 2, { steps: 4 });
    await page.waitForTimeout(250);
    await page.mouse.up({ button: "right" });
    await page.keyboard.up("Shift");
    await page.waitForTimeout(250);
    expect(errors.filter((message) => /destroyed|validation|out of memory|failed to import|shader|device|WGSL|pipeline/i.test(message))).toEqual([]);
});

test("FLIP inflow-only imports report emitted particles instead of legacy capacity", async ({ page }) => {
    await page.goto("/demo-fluid.html?demo=whiteboard&method=PBF&quality=low");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });
    const canvas = page.locator("canvas");
    await page.locator('input[type="file"][accept*=".json"]').setInputFiles({
        name: "fluid-whiteboard-FLIP.json",
        mimeType: "application/json",
        buffer: flipInflowOnlyPreset(),
    });
    await expect(canvas).toHaveAttribute("data-method", "FLIP");
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count"))).toBeGreaterThan(0);
    const activeCount = Number(await canvas.getAttribute("data-active-particle-count"));
    expect(activeCount).toBeLessThan(80_000);
    await expect(page.getByText(/^Particles:\s*[\d,]+\s*active\s*\/\s*80,000\s*capacity$/)).toBeVisible();
    await expect(page.getByText(/^After restart:/)).toBeHidden();
    await expect(page.locator('[data-fluid-flip-particle-capacity="true"] input')).toHaveValue("80000");

    const behavior = () => page.locator('[data-flow-field-label="Behavior"] select').first();
    const gpuBytes = await canvas.getAttribute("data-simulation-gpu-bytes");
    await behavior().selectOption("initial");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "15242");
    await expect(canvas).toHaveAttribute("data-restart-particle-capacity", "80000");
    await expect(page.getByText(/^After restart:\s*15,242\s*active\s*\/\s*80,000\s*capacity$/)).toBeVisible();

    await page.getByRole("button", { name: "Reset simulation" }).click();
    await expect(canvas).toHaveAttribute("data-particle-count", "80000");
    await expect(canvas).toHaveAttribute("data-flip-particle-capacity-request", "80000");
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count"))).toBe(15_242);
    await expect(canvas).toHaveAttribute("data-simulation-gpu-bytes", gpuBytes!);

    await behavior().selectOption("inflow");
    await expect(canvas).toHaveAttribute("data-restart-particle-count", "0");
    await expect(canvas).toHaveAttribute("data-restart-particle-capacity", "80000");
    await expect(page.getByText(/^After restart:\s*0\s*active\s*\/\s*80,000\s*capacity$/)).toBeVisible();

    await page.getByRole("button", { name: "Reset simulation" }).click();
    await expect(canvas).toHaveAttribute("data-particle-count", "80000");
    await expect(canvas).toHaveAttribute("data-flip-particle-capacity-request", "80000");
    await expect(canvas).toHaveAttribute("data-simulation-gpu-bytes", gpuBytes!);
    await expect(page.getByText(/^After restart:/)).toBeHidden();
});

test("FLIP warns when marker density exceeds useful MAC-grid sampling", async ({ page }) => {
    await page.goto("/demo-fluid.html?demo=box&method=PBF&quality=low");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });
    const canvas = page.locator("canvas");
    const input = page.locator('input[type="file"][accept*=".json"]');
    const highCountDialogPromise = page.waitForEvent("dialog");
    const highDensityImport = input.setInputFiles({
        name: "flip-high-density.json",
        mimeType: "application/json",
        buffer: flipHighDensityPreset(),
    });
    const highCountDialog = await highCountDialogPromise;
    expect(highCountDialog.message()).toContain("131,072");
    expect(highCountDialog.message()).toContain("Resolution divisions from 56 to 37");
    await highCountDialog.accept();
    await highDensityImport;

    await expect(canvas).toHaveAttribute("data-method", "FLIP");
    await expect(canvas).toHaveAttribute("data-particle-count", "40000");
    await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count"))).toBe(37_806);
    await expect(canvas).toHaveAttribute("data-flip-marker-density-warning", "true");
    const warning = page.getByText(/^High FLIP marker density:/);
    await expect(warning).toBeVisible();
    await expect(warning).toHaveCSS("color", "rgb(255, 95, 86)");
    expect(await warning.textContent()).toContain("density:\u00a0about\u00a032.0\u00a0markers");
    expect(await warning.textContent()).toContain("cell.\u00a0Above\u00a016,\u00a0extra");
    expect(await warning.textContent()).toContain("Reduce Markers per cell.");

    await input.setInputFiles({
        name: "flip-low-density.json",
        mimeType: "application/json",
        buffer: flipLowDensityPreset(),
    });
    await expect(canvas).toHaveAttribute("data-flip-marker-density-warning", "false");
    await expect(warning).toBeHidden();
});

test("deletes and independently emits particles in every solver", async ({ page }) => {
    test.setTimeout(180_000);
    const errors: string[] = [];
    page.on("console", (message) => {
        if (message.type() === "error" || message.type() === "warning") {
            errors.push(message.text());
        }
    });
    page.on("pageerror", (error) => errors.push(error.message));

    await page.goto("/demo-fluid.html?demo=box&method=PBF&quality=low");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });
    const canvas = page.locator("canvas");
    const input = page.locator('input[type="file"][accept*=".json"]');

    for (const method of ["PB-MPM", "PBF", "FLIP", "MLS-MPM"] as const) {
        await test.step(method, async () => {
            await input.setInputFiles({ name: `${method}-delete.json`, mimeType: "application/json", buffer: lifecyclePreset(method, "delete") });
            await expect(canvas).toHaveAttribute("data-method", method);
            await expect(canvas).toHaveAttribute("data-particle-count", "5000");
            await expect(canvas).toHaveAttribute("data-sink-count", "1");
            await expect.poll(async () => Number(await canvas.getAttribute("data-active-particle-count")), { timeout: 15_000 }).toBe(0);

            await input.setInputFiles({ name: `${method}-inflow.json`, mimeType: "application/json", buffer: lifecyclePreset(method, "inflow") });
            await expect(canvas).toHaveAttribute("data-emitter-count", "1");
            await expect
                .poll(
                    async () =>
                        errors.filter((message) => /destroyed|validation|out of memory|failed to import|shader|device/i.test(message)).join("\n") ||
                        (Number(await canvas.getAttribute("data-active-particle-count")) > 0 ? "active" : "waiting"),
                    { timeout: 15_000 }
                )
                .toBe("active");
        });
    }

    expect(errors.filter((message) => /destroyed|validation|out of memory|failed to import|shader/i.test(message))).toEqual([]);
});

test("warns before importing a high particle count", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });
    await page.locator('select:has(option[value="whiteboard"])').selectOption("whiteboard");

    const input = page.locator('input[type="file"][accept*=".json"]');
    const safeDialogPromise = page.waitForEvent("dialog");
    const safeImport = input.setInputFiles({
        name: "high-safe.json",
        mimeType: "application/json",
        buffer: jsonExport([12, 10, 14], 100_001, 1, [3, 4, 5]),
    });
    const safeDialog = await safeDialogPromise;
    expect(safeDialog.type()).toBe("confirm");
    expect(safeDialog.message()).toContain("100,001");
    expect(safeDialog.message()).toContain("40,000");
    await safeDialog.accept();
    await safeImport;
    await expect(page.locator("canvas")).toHaveAttribute("data-particle-count", "40000");

    const keepDialogPromise = page.waitForEvent("dialog");
    const keepImport = input.setInputFiles({
        name: "high-keep.json",
        mimeType: "application/json",
        buffer: jsonExport([12, 10, 14], 100_002, 1, [3, 4, 5]),
    });
    const keepDialog = await keepDialogPromise;
    expect(keepDialog.message()).toContain("100,002");
    await keepDialog.dismiss();
    await keepImport;
    await expect(page.locator("canvas")).toHaveAttribute("data-particle-count", "100002");
});

test("FLIP import warning reports the domain-clipped initial count", async ({ page }) => {
    await page.goto("/demo-fluid.html");
    await waitForCanvasReady(page, { timeout: 60_000, label: "Fluid demo" });
    await page.locator('select:has(option[value="whiteboard"])').selectOption("whiteboard");

    const dialogPromise = page.waitForEvent("dialog");
    const importPromise = page.locator('input[type="file"][accept*=".json"]').setInputFiles({
        name: "flip-clipped-high.json",
        mimeType: "application/json",
        buffer: flipClippedHighCountPreset(),
    });
    const dialog = await dialogPromise;
    expect(dialog.message()).toContain("512,000");
    expect(dialog.message()).not.toContain("1,728,000");
    await dialog.accept();
    await importPromise;
});
