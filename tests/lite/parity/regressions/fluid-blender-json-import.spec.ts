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

function formatParticleCount(count: number): string {
    if (count >= 1_000_000) {
        return `${(count / 1_000_000).toFixed(count % 1_000_000 === 0 ? 0 : 2)}M`;
    }
    if (count >= 1_000) {
        return `${(count / 1_000).toFixed(count % 1_000 === 0 ? 0 : 1)}k`;
    }
    return String(count);
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
    data.formatVersion = 9;
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

function lifecyclePreset(method: "PBF" | "MLS-MPM" | "PB-MPM", kind: "delete" | "inflow"): Buffer {
    const data = preset([12, 10, 14], 5_000, 0) as Record<string, unknown>;
    data.formatVersion = 9;
    data.meta = { demo: "blender", method };
    data.physics =
        method === "PBF"
            ? { gravity: 9.8, viscosity: 0.08, relaxation: 50, scorr: 0.02, iterations: 3, restDensity: 341, boundaryDensity: 0 }
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
    const activeParticleCounter = page.getByText(/^Active particles:/).first();
    await expect(activeParticleCounter).toBeVisible();
    await page.keyboard.press("p");
    await expect(page.locator("canvas")).toHaveAttribute("data-paused", "true");
    await expect(page.locator("canvas")).toHaveAttribute("data-active-particle-count", /^\d+$/);
    await expect
        .poll(async () => {
            const count = Number(await page.locator("canvas").getAttribute("data-active-particle-count"));
            return (await activeParticleCounter.textContent())?.replace(/\s/g, "") === `Activeparticles:${formatParticleCount(count)}`;
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

    for (const method of ["PB-MPM", "PBF", "MLS-MPM"] as const) {
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
