import pngjs from "pngjs";
import type { Page } from "@playwright/test";
import { resolve } from "path";
import { test, expect } from "../parity/parity-fixtures";
import { integrateHomogeneousMedium } from "../../../packages/babylon-lite/src/render/volume/optical-transfer";
import { fogInjectWgsl, fogIntegrateWgsl } from "../../../packages/babylon-lite/src/volumetric-fog/volumetric-fog-wgsl";
import { createScene316CloudDensitySource } from "../../../lab/lite/src/shared/scene316-cloud-density";

const { PNG } = pngjs;
const TEST_PAGE = "/lite/scene316.html";
const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;

interface PixelRegion {
    left: number;
    top: number;
    right: number;
    bottom: number;
}

function meanLuminance(image: Buffer): number {
    const { width, height, data } = PNG.sync.read(image);
    let total = 0;
    let pixels = 0;
    for (let y = Math.floor(height * 0.25); y < Math.floor(height * 0.85); y += 8) {
        for (let x = Math.floor(width * 0.25); x < Math.floor(width * 0.85); x += 8) {
            const index = (y * width + x) * 4;
            total += data[index]! * 0.2126 + data[index + 1]! * 0.7152 + data[index + 2]! * 0.0722;
            pixels++;
        }
    }
    return total / pixels;
}

function meanRgbDifference(actual: Buffer, expected: Buffer, region?: PixelRegion): number {
    const actualImage = PNG.sync.read(actual);
    const expectedImage = PNG.sync.read(expected);
    if (actualImage.width !== expectedImage.width || actualImage.height !== expectedImage.height) {
        throw new Error("Fog comparison screenshots must have the same dimensions.");
    }
    const { left = 0, top = 80, right = actualImage.width, bottom = actualImage.height } = region ?? {};
    if (left < 0 || top < 0 || right > actualImage.width || bottom > actualImage.height || left >= right || top >= bottom) {
        throw new RangeError("Fog comparison region must be inside the screenshot.");
    }
    let difference = 0;
    let channels = 0;
    for (let y = top; y < bottom; y++) {
        for (let x = left; x < right; x++) {
            const pixel = (y * actualImage.width + x) * 4;
            for (let channel = 0; channel < 3; channel++) {
                difference += Math.abs(actualImage.data[pixel + channel]! - expectedImage.data[pixel + channel]!);
                channels++;
            }
        }
    }
    return difference / channels;
}

function cloudEdgeVariation(image: Buffer): number {
    const { width, height, data } = PNG.sync.read(image);
    const heights: number[] = [];
    for (const [start, end] of [
        [0.04, 0.2],
        [0.72, 0.94],
    ] as const) {
        for (let x = Math.floor(width * start); x < width * end; x += 16) {
            for (let y = Math.floor(height * 0.15); y < height * 0.55; y++) {
                const i = (y * width + x) * 4;
                const luminance = data[i]! * 0.2126 + data[i + 1]! * 0.7152 + data[i + 2]! * 0.0722;
                if (luminance > 30) {
                    heights.push(y);
                    break;
                }
            }
        }
    }
    if (heights.length < 10) {
        throw new Error("Cloud bank has no visible upper edge.");
    }
    return Math.max(...heights) - Math.min(...heights);
}

async function waitForReady(page: Page): Promise<void> {
    await page.waitForFunction(() => {
        const data = document.querySelector("canvas")?.dataset;
        return data?.ready === "true" || !!data?.error;
    });
    expect(await page.locator("canvas").getAttribute("data-error")).toBeNull();
    expect(await page.locator("canvas").getAttribute("data-ready")).toBe("true");
}

async function captureBaseline(page: Page): Promise<Buffer> {
    const reference = await page.context().newPage();
    try {
        await reference.setViewportSize(page.viewportSize() ?? { width: 720, height: 405 });
        await reference.goto(`${TEST_PAGE}?baseline=1`);
        await waitForReady(reference);
        return await reference.locator("canvas").screenshot();
    } finally {
        await reference.close();
    }
}

test.describe("volumetric fog integration", () => {
    for (const msaa of [1, 4]) {
        test(`renders, toggles and resizes at ${msaa}x MSAA`, async ({ page }) => {
            const errors: string[] = [];
            page.on("pageerror", (error) => errors.push(error.message));
            await page.setViewportSize({ width: 720, height: 405 });
            await page.goto(`${TEST_PAGE}?msaa=${msaa}&seekTime=0`);
            await waitForReady(page);
            const canvas = page.locator("canvas");
            const fogged = meanLuminance(await canvas.screenshot());
            const previousFrame = Number(await canvas.getAttribute("data-frames"));
            await page.locator("#fogToggle").click();
            await page.waitForFunction((frame) => Number(document.querySelector("canvas")?.dataset.frames) >= frame + 4, previousFrame);
            const clear = meanLuminance(await canvas.screenshot());
            expect(Math.abs(fogged - clear)).toBeGreaterThan(10);
            expect(await canvas.getAttribute("data-error")).toBeNull();

            await page.setViewportSize({ width: 864, height: 486 });
            await page.waitForFunction(() => document.querySelector("canvas")?.dataset.width === "864");
            const beforeCameraMove = await canvas.screenshot();
            await page.mouse.move(340, 280);
            await page.mouse.down();
            await page.mouse.move(510, 315, { steps: 8 });
            await page.mouse.up();
            const rotatedFrame = Number(await canvas.getAttribute("data-frames"));
            await page.waitForFunction((frame) => Number(document.querySelector("canvas")?.dataset.frames) > frame + 4, rotatedFrame);
            expect(meanRgbDifference(beforeCameraMove, await canvas.screenshot())).toBeGreaterThan(5);

            await page.mouse.move(390, 280);
            await page.mouse.down({ button: "right" });
            await page.mouse.move(480, 310, { steps: 8 });
            await page.mouse.up({ button: "right" });
            await page.mouse.wheel(0, 250);
            const movedFrame = Number(await canvas.getAttribute("data-frames"));
            await page.waitForFunction((frame) => Number(document.querySelector("canvas")?.dataset.frames) > frame + 4, movedFrame);
            expect(meanRgbDifference(beforeCameraMove, await canvas.screenshot())).toBeGreaterThan(5);
            await page.locator("#cameraReset").click();
            const resetFrame = Number(await canvas.getAttribute("data-frames"));
            await page.waitForFunction((frame) => Number(document.querySelector("canvas")?.dataset.frames) > frame + 4, resetFrame);
            expect(meanRgbDifference(beforeCameraMove, await canvas.screenshot())).toBeLessThan(2);

            const reenabledFrame = Number(await canvas.getAttribute("data-frames"));
            await page.locator("#fogToggle").click();
            await page.waitForFunction((frame) => {
                const data = document.querySelector("canvas")?.dataset;
                return data?.fogEnabled === "true" && Number(data.frames) >= frame + 4;
            }, reenabledFrame);
            const foggedAfterResize = await canvas.screenshot();
            const reference = await page.context().newPage();
            try {
                await reference.setViewportSize({ width: 864, height: 486 });
                await reference.goto(`${TEST_PAGE}?msaa=${msaa}&seekTime=0`);
                await waitForReady(reference);
                expect(meanRgbDifference(foggedAfterResize, await reference.locator("canvas").screenshot())).toBeLessThan(2);
            } finally {
                await reference.close();
            }
            await page.mouse.move(340, 280);
            await page.mouse.down();
            await page.mouse.move(510, 315, { steps: 8 });
            await page.mouse.up();
            const foggedCameraFrame = Number(await canvas.getAttribute("data-frames"));
            await page.waitForFunction((frame) => Number(document.querySelector("canvas")?.dataset.frames) > frame + 4, foggedCameraFrame);
            expect(meanRgbDifference(foggedAfterResize, await canvas.screenshot())).toBeGreaterThan(5);
            await page.locator("#cameraReset").click();
            const restoredFrame = Number(await canvas.getAttribute("data-frames"));
            await page.waitForFunction((frame) => Number(document.querySelector("canvas")?.dataset.frames) > frame + 4, restoredFrame);
            expect(meanRgbDifference(foggedAfterResize, await canvas.screenshot())).toBeLessThan(2);
            expect(await canvas.getAttribute("data-error")).toBeNull();
            expect(errors).toEqual([]);
        });
    }

    for (const order of ["before", "after"]) {
        test(`shares the HDR scene target when transmission is enabled ${order} fog`, async ({ page }) => {
            const errors: string[] = [];
            page.on("pageerror", (error) => errors.push(error.message));
            await page.goto(`${TEST_PAGE}?transmission=${order}`);
            await waitForReady(page);
            expect(await page.locator("canvas").getAttribute("data-error")).toBeNull();
            expect(errors).toEqual([]);
        });
    }

    test("renders PBR meshes added after fog installation", async ({ page }) => {
        await page.setViewportSize({ width: 720, height: 405 });
        await page.goto(`${TEST_PAGE}?early=1&seekTime=0`);
        await waitForReady(page);
        const reference = await page.context().newPage();
        try {
            await reference.setViewportSize({ width: 720, height: 405 });
            await reference.goto(`${TEST_PAGE}?seekTime=0`);
            await waitForReady(reference);
            expect(meanRgbDifference(await page.locator("canvas").screenshot(), await reference.locator("canvas").screenshot())).toBeLessThan(2);
        } finally {
            await reference.close();
        }
    });

    test("keeps an opaque no-IBL unlit PBR mesh color when fog owns linear HDR", async ({ page }) => {
        await page.setViewportSize({ width: 720, height: 405 });
        await page.goto(`${TEST_PAGE}?unlit=1&baseline=1&seekTime=0`);
        await waitForReady(page);
        const unlitBaseline = await page.locator("canvas").screenshot();
        await page.goto(`${TEST_PAGE}?unlit=1&seekTime=0`);
        await waitForReady(page);
        const unlitWithFog = await page.locator("canvas").screenshot();
        const foregroundCube = { left: 172, top: 155, right: 239, bottom: 197 };

        expect(meanRgbDifference(unlitWithFog, unlitBaseline, foregroundCube)).toBeLessThan(2);
        expect(meanRgbDifference(unlitBaseline, await captureBaseline(page), foregroundCube)).toBeGreaterThan(5);
    });

    test("morphs a billowing bank in place at constant density", async ({ page }) => {
        await page.setViewportSize({ width: 1280, height: 720 });
        await page.goto(`${TEST_PAGE}?seekTime=0`);
        await waitForReady(page);
        const canvas = page.locator("canvas");
        const startingDensity = Number(await canvas.getAttribute("data-fog-density"));
        const initial = await canvas.screenshot();

        await page.goto(`${TEST_PAGE}?seekTime=1`);
        await waitForReady(page);
        const intermediateDensity = Number(await canvas.getAttribute("data-fog-density"));
        const intermediate = await canvas.screenshot();
        await page.goto(`${TEST_PAGE}?seekTime=2`);
        await waitForReady(page);
        const finalDensity = Number(await canvas.getAttribute("data-fog-density"));
        const final = await canvas.screenshot();
        const baseline = await captureBaseline(page);
        const bank = { left: 900, top: 300, right: 1200, bottom: 370 };

        expect(startingDensity).toBeCloseTo(0.28, 3);
        expect(intermediateDensity).toBeCloseTo(startingDensity, 3);
        expect(finalDensity).toBeCloseTo(startingDensity, 3);
        expect(meanRgbDifference(initial, baseline, bank)).toBeGreaterThan(100);
        expect(meanRgbDifference(initial, intermediate, { left: 35, top: 185, right: 245, bottom: 310 })).toBeGreaterThan(10);
        expect(meanRgbDifference(initial, intermediate, { left: 920, top: 180, right: 1200, bottom: 305 })).toBeGreaterThan(10);
        expect(meanRgbDifference(initial, final, { left: 920, top: 180, right: 1200, bottom: 305 })).toBeGreaterThan(10);
        expect(meanRgbDifference(initial, intermediate, bank)).toBeLessThan(20);
        expect(cloudEdgeVariation(initial)).toBeGreaterThan(25);
        expect(meanRgbDifference(initial, baseline, { left: 120, top: 540, right: 1150, bottom: 660 })).toBeLessThan(1);
        expect(meanRgbDifference(initial, baseline, { left: 305, top: 275, right: 425, bottom: 350 })).toBeLessThan(2);
        expect(meanRgbDifference(initial, baseline, { left: 900, top: 70, right: 1200, bottom: 150 })).toBeLessThan(1);

        await page.setViewportSize({ width: 640, height: 360 });
        await page.goto(`${TEST_PAGE}?msaa=1`);
        await waitForReady(page);
        const initialTime = Number(await canvas.getAttribute("data-cloud-time"));
        await page.waitForFunction((time) => Number(document.querySelector("canvas")?.dataset.cloudTime) > time + 0.5, initialTime, { timeout: 15_000 });
        expect(Number(await canvas.getAttribute("data-fog-density"))).toBeCloseTo(startingDensity, 3);
    });

    test("pauses and resumes cloud shape without changing extinction", async ({ page }) => {
        await page.setViewportSize({ width: 720, height: 405 });
        await page.goto(TEST_PAGE);
        await waitForReady(page);
        const canvas = page.locator("canvas");
        const shapeButton = page.locator("#shapeToggle");
        await shapeButton.click();
        const frozenTime = await canvas.getAttribute("data-cloud-time");
        const density = await canvas.getAttribute("data-fog-density");
        const frozenImage = await canvas.screenshot();
        const frozenFrame = Number(await canvas.getAttribute("data-frames"));
        await page.waitForFunction((frame) => Number(document.querySelector("canvas")?.dataset.frames) > frame + 30, frozenFrame, { timeout: 15_000 });
        expect(meanRgbDifference(frozenImage, await canvas.screenshot(), { left: 500, top: 140, right: 690, bottom: 220 })).toBeLessThan(1);
        expect(await canvas.getAttribute("data-cloud-time")).toBe(frozenTime);
        expect(await canvas.getAttribute("data-fog-density")).toBe(density);
        expect(await canvas.getAttribute("data-animating")).toBe("false");
        expect(new URL(page.url()).searchParams.get("animate")).toBe("off");

        await shapeButton.click();
        await page.waitForFunction((frozen) => Number(document.querySelector("canvas")?.dataset.cloudTime) > Number(frozen) + 1, frozenTime, { timeout: 15_000 });
        expect(meanRgbDifference(frozenImage, await canvas.screenshot(), { left: 500, top: 140, right: 690, bottom: 220 })).toBeGreaterThan(5);
        expect(await canvas.getAttribute("data-fog-density")).toBe(density);
        expect(await canvas.getAttribute("data-animating")).toBe("true");
        expect(new URL(page.url()).searchParams.has("animate")).toBe(false);
        expect(new URL(page.url()).searchParams.has("scroll")).toBe(false);

        await page.goto(`${TEST_PAGE}?scroll=off`);
        await waitForReady(page);
        expect(await page.locator("#shapeToggle").textContent()).toBe("Cloud shape: Off");
        await page.locator("#shapeToggle").click();
        expect(new URL(page.url()).searchParams.has("scroll")).toBe(false);
        expect(await canvas.getAttribute("data-animating")).toBe("true");

        await page.goto(`${TEST_PAGE}?seekTime=0`);
        await waitForReady(page);
        expect(await page.locator("#shapeToggle").isDisabled()).toBe(true);
        expect(await page.locator("#shapeToggle").textContent()).toBe("Animation frozen");
        expect(await canvas.getAttribute("data-animating")).toBe("false");
        expect(await canvas.getAttribute("data-cloud-time")).toBe("0.000");
    });

    test("reloads both 3D fog textures at the selected size while preserving URL options", async ({ page }) => {
        await page.setViewportSize({ width: 720, height: 405 });
        await page.goto(`${TEST_PAGE}?msaa=1&density=0.2&animate=off&volumeSize=standard`);
        await waitForReady(page);
        const canvas = page.locator("canvas");
        expect(await canvas.getAttribute("data-volume-size")).toBe("90×51×80");
        expect(await page.locator("#volumeDims").textContent()).toBe("90×51×80");

        await page.locator("#volumeSize").selectOption("depth");
        await page.waitForURL((url) => url.searchParams.get("volumeSize") === "depth");
        await waitForReady(page);
        const url = new URL(page.url());
        expect(url.searchParams.get("msaa")).toBe("1");
        expect(url.searchParams.get("density")).toBe("0.2");
        expect(url.searchParams.get("animate")).toBe("off");
        expect(await canvas.getAttribute("data-animating")).toBe("false");
        expect(await canvas.getAttribute("data-volume-size")).toBe("90×51×160");

        await page.setViewportSize({ width: 864, height: 486 });
        await page.waitForFunction(() => document.querySelector("canvas")?.dataset.volumeSize === "108×61×160");
        expect(await page.locator("#volumeDims").textContent()).toBe("108×61×160");
        await page.locator("#volumeSize").selectOption("low");
        await page.waitForURL((next) => next.searchParams.get("volumeSize") === "low");
        await waitForReady(page);
        expect(await canvas.getAttribute("data-volume-size")).toBe("54×31×64");
    });

    test("zero-density enabled fog preserves the unfogged image", async ({ page }) => {
        await page.setViewportSize({ width: 720, height: 405 });
        await page.goto(`${TEST_PAGE}?density=0`);
        await waitForReady(page);
        const zeroDensity = await page.locator("canvas").screenshot();
        expect(meanRgbDifference(zeroDensity, await captureBaseline(page))).toBeLessThan(1);
    });

    test("a disabled fog path retains the unfogged scene's output", async ({ page }) => {
        await page.setViewportSize({ width: 720, height: 405 });
        await page.goto(TEST_PAGE);
        await waitForReady(page);
        const canvas = page.locator("canvas");
        await page.locator("#fogToggle").click();
        await page.waitForFunction(() => Number(document.querySelector("canvas")?.dataset.frames) > 10);
        expect(meanRgbDifference(await canvas.screenshot(), await captureBaseline(page))).toBeLessThan(2);
    });

    test("restores fog GPU resources after device loss", async ({ page }) => {
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        await page.setViewportSize({ width: 720, height: 405 });
        await page.goto(`${TEST_PAGE}?recovery=1&seekTime=0`);
        await waitForReady(page);
        const canvas = page.locator("canvas");
        const before = await canvas.screenshot();
        const previousFrame = Number(await canvas.getAttribute("data-frames"));
        await page.locator("#deviceLoss").click();
        await page.waitForFunction((frame) => {
            const data = document.querySelector("canvas")?.dataset;
            return !!data?.error || (data?.recovered === "true" && Number(data.frames) > frame + 3);
        }, previousFrame);
        expect(await canvas.getAttribute("data-error")).toBeNull();
        expect(meanRgbDifference(before, await canvas.screenshot())).toBeLessThan(2);
        expect(errors).toEqual([]);
    });
});

test("invalid caller density shaders reject scene registration before publishing fog GPU state", async ({ page }) => {
    await page.goto("/");
    const results = await page.evaluate(async (entry) => {
        const lite = await import(entry);
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = 128;
        document.body.append(canvas);
        const engine = await lite.createEngine(canvas, { msaaSamples: 1, maxDevicePixelRatio: 1 });
        const gpuErrors: string[] = [];
        engine._device.addEventListener("uncapturederror", (event: GPUUncapturedErrorEvent) => gpuErrors.push(event.error.message));
        const cases = [
            { name: "unresolved WGSL value", shader: "fn fogDensityAt(p:vec3f)->f32 { return missingDensityValue; }", uniformFloatCount: 0 },
            {
                name: "storage instead of uniform binding",
                shader: "@group(0) @binding(2) var<storage,read> bad:array<f32>; fn fogDensityAt(p:vec3f)->f32 { return bad[0]; }",
                uniformFloatCount: 4,
            },
            {
                name: "shader uniform exceeds declared size",
                shader: "struct Density { one:vec4f, two:vec4f }; @group(0) @binding(2) var<uniform> p:Density; fn fogDensityAt(worldPosition:vec3f)->f32 { return p.two.x; }",
                uniformFloatCount: 4,
            },
        ];
        try {
            const failures = [];
            for (const { name, shader, uniformFloatCount } of cases) {
                const scene = lite.createSceneContext(engine);
                scene.camera = lite.createFreeCamera({ x: 0, y: 2, z: -8 }, { x: 0, y: 0, z: 0 });
                const fog = lite.enableVolumetricFog(scene, engine, {
                    density: 0.1,
                    densitySource: { shader, uniformFloatCount, writeUniforms: (data: Float32Array) => data.fill(0) },
                });
                let frameGraphBuilt = false;
                const build = scene._frameGraph.build;
                scene._frameGraph.build = () => {
                    frameGraphBuilt = true;
                    build();
                };
                let registrationError: string | null = null;
                try {
                    await lite.registerScene(scene);
                } catch (error) {
                    registrationError = error instanceof Error ? error.message : String(error);
                }
                failures.push({
                    name,
                    registrationError,
                    registered: engine._renderingContexts.includes(scene),
                    frameGraphBuilt,
                    gpuStatePublished: fog._gpu !== null,
                    pipelinePublished: fog._preparedInjection !== null,
                });
                lite.disposeScene(scene);
            }
            await engine._device.queue.onSubmittedWorkDone();
            return { failures, gpuErrors };
        } finally {
            lite.disposeEngine(engine);
            canvas.remove();
        }
    }, LITE_ENTRY);

    for (const failure of results.failures) {
        expect(failure.registrationError, failure.name).toContain("Volumetric fog: injection pipeline preparation failed");
        expect(failure.registered, failure.name).toBe(false);
        expect(failure.frameGraphBuilt, failure.name).toBe(false);
        expect(failure.gpuStatePublished, failure.name).toBe(false);
        expect(failure.pipelinePublished, failure.name).toBe(false);
    }
    expect(results.failures[0]!.registrationError).toContain("missingDensityValue");
    expect(results.failures[1]!.registrationError).toMatch(/binding|layout/i);
    expect(results.failures[2]!.registrationError).toMatch(/binding|size|layout/i);
    expect(results.gpuErrors).toEqual([]);
});

test("injected scattering and ray integration match homogeneous media and evolve bounded cloud extinction", async ({ page }) => {
    await page.goto("/");
    const results = await page.evaluate(
        async ({ injectionShader, integrationShader, cloudShader }) => {
            const adapter = await navigator.gpu?.requestAdapter();
            if (!adapter) {
                throw new Error("Volumetric fog needs a WebGPU adapter.");
            }
            const device = await adapter.requestDevice();
            const buffers: GPUBuffer[] = [];
            const textures: GPUTexture[] = [];
            try {
                device.pushErrorScope("validation");
                const injectionModule = device.createShaderModule({ code: injectionShader });
                const cloudModule = device.createShaderModule({ code: cloudShader });
                const integrationModule = device.createShaderModule({ code: integrationShader });
                const readModule = device.createShaderModule({
                    code: `@group(0) @binding(0) var injected:texture_3d<f32>;
@group(0) @binding(1) var integrated:texture_3d<f32>;
@group(0) @binding(2) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(1) fn main(){
result[0]=textureLoad(injected,vec3i(0,0,3),0);
result[1]=textureLoad(integrated,vec3i(0,0,3),0);
}`,
                });
                const injection = await device.createComputePipelineAsync({ layout: "auto", compute: { module: injectionModule, entryPoint: "main" } });
                const cloudInjection = await device.createComputePipelineAsync({ layout: "auto", compute: { module: cloudModule, entryPoint: "main" } });
                const integration = await device.createComputePipelineAsync({ layout: "auto", compute: { module: integrationModule, entryPoint: "main" } });
                const read = await device.createComputePipelineAsync({ layout: "auto", compute: { module: readModule, entryPoint: "main" } });
                const sample = async (
                    screenWidth: number,
                    projectionX: number,
                    density: number,
                    cloud?: { eyeX: number; eyeY: number; time: number },
                    sunIntensity = 0
                ): Promise<{ injected: number[]; integrated: number[] }> => {
                    const parameters = new Float32Array(48);
                    parameters[0] = parameters[5] = parameters[10] = parameters[15] = 1;
                    const cloudParameters = cloud ? new Float32Array(4) : null;
                    if (cloud) {
                        parameters[12] = cloud.eyeX;
                        parameters[13] = cloud.eyeY;
                        cloudParameters![0] = cloud.time;
                    }
                    parameters.set([1, 1, 4, 16], 16);
                    parameters.set([0.1, 10, projectionX, 1], 20);
                    parameters.set([density, 0, 0, 1], 24);
                    parameters.set([0.5, 0.8, 1, 0], 28);
                    parameters.set([0.3, 0.4, 0.5, 0], 32);
                    if (sunIntensity > 0) {
                        parameters[31] = 0.35;
                        parameters.set([0, 0, -1, sunIntensity], 36);
                        parameters.set([0.9, 0.6, 0.3, 0], 40);
                    } else {
                        parameters.set([0, -1, 0, 0], 36);
                    }
                    parameters.set([screenWidth, 16, 10, 0], 44);
                    const params = device.createBuffer({ size: parameters.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                    const cloudParams = cloudParameters && device.createBuffer({ size: cloudParameters.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                    const createVolume = (): GPUTexture =>
                        device.createTexture({
                            size: { width: 1, height: 1, depthOrArrayLayers: 4 },
                            dimension: "3d",
                            format: "rgba16float",
                            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
                        });
                    const injected = createVolume();
                    const integrated = createVolume();
                    const output = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                    const readback = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
                    buffers.push(params, output, readback);
                    if (cloudParams) {
                        buffers.push(cloudParams);
                        device.queue.writeBuffer(cloudParams, 0, cloudParameters!);
                    }
                    textures.push(injected, integrated);
                    device.queue.writeBuffer(params, 0, parameters);
                    const activeInjection = cloud ? cloudInjection : injection;
                    const injectGroup = device.createBindGroup({
                        layout: activeInjection.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: params } },
                            { binding: 1, resource: injected.createView() },
                            ...(cloudParams ? [{ binding: 2, resource: { buffer: cloudParams } }] : []),
                        ],
                    });
                    const integrateGroup = device.createBindGroup({
                        layout: integration.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: params } },
                            { binding: 1, resource: injected.createView() },
                            { binding: 2, resource: integrated.createView() },
                        ],
                    });
                    const readGroup = device.createBindGroup({
                        layout: read.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: injected.createView() },
                            { binding: 1, resource: integrated.createView() },
                            { binding: 2, resource: { buffer: output } },
                        ],
                    });
                    const encoder = device.createCommandEncoder();
                    const injectPass = encoder.beginComputePass();
                    injectPass.setPipeline(activeInjection);
                    injectPass.setBindGroup(0, injectGroup);
                    injectPass.dispatchWorkgroups(1, 1, 4);
                    injectPass.end();
                    const integratePass = encoder.beginComputePass();
                    integratePass.setPipeline(integration);
                    integratePass.setBindGroup(0, integrateGroup);
                    integratePass.dispatchWorkgroups(1, 1);
                    integratePass.end();
                    const readPass = encoder.beginComputePass();
                    readPass.setPipeline(read);
                    readPass.setBindGroup(0, readGroup);
                    readPass.dispatchWorkgroups(1);
                    readPass.end();
                    encoder.copyBufferToBuffer(output, 0, readback, 0, 32);
                    device.queue.submit([encoder.finish()]);
                    await readback.mapAsync(GPUMapMode.READ);
                    const values = Array.from(new Float32Array(readback.getMappedRange()));
                    readback.unmap();
                    return { injected: values.slice(0, 4), integrated: values.slice(4, 8) };
                };
                const center = await sample(16, 1, 0.2);
                const offAxis = await sample(32, 0.5 / Math.sqrt(3), 0.2);
                const sunlit = await sample(16, 1, 0.2, undefined, 2);
                const clear = await sample(16, 1, 0);
                const insideCloud = await sample(16, 1, 0.2, { eyeX: 0, eyeY: 0.8, time: 0 });
                const aboveCloud = await sample(16, 1, 0.2, { eyeX: 0, eyeY: 4, time: 0 });
                const outsideCloud = await sample(16, 1, 0.2, { eyeX: 120, eyeY: 0.8, time: 0 });
                const clearCloud = await sample(16, 1, 0, { eyeX: 0, eyeY: 0.8, time: 0 });
                const evolvingCloudAtStart = await sample(16, 1, 0.2, { eyeX: 8, eyeY: 0.8, time: 0 });
                const evolvingCloudAtHalf = await sample(16, 1, 0.2, { eyeX: 8, eyeY: 0.8, time: 1 });
                const evolvingCloudAtNext = await sample(16, 1, 0.2, { eyeX: 8, eyeY: 0.8, time: 2 });
                const beforeTransition = await sample(16, 1, 0.2, { eyeX: 8, eyeY: 0.8, time: 1.99 });
                const afterTransition = await sample(16, 1, 0.2, { eyeX: 8, eyeY: 0.8, time: 2.01 });
                const validation = await device.popErrorScope();
                if (validation) {
                    throw new Error(validation.message);
                }
                return {
                    center,
                    offAxis,
                    sunlit,
                    clear,
                    insideCloud,
                    aboveCloud,
                    outsideCloud,
                    clearCloud,
                    evolvingCloudAtStart,
                    evolvingCloudAtHalf,
                    evolvingCloudAtNext,
                    beforeTransition,
                    afterTransition,
                };
            } finally {
                for (const buffer of buffers) {
                    buffer.destroy();
                }
                for (const texture of textures) {
                    texture.destroy();
                }
                device.destroy();
            }
        },
        {
            injectionShader: fogInjectWgsl(),
            integrationShader: fogIntegrateWgsl(),
            cloudShader: fogInjectWgsl(createScene316CloudDensitySource({ time: 0 }).shader),
        }
    );

    const extinction: [number, number, number] = [0.2, 0.2, 0.2];
    const source: [number, number, number] = [0.03, 0.064, 0.1];
    const centerExpected = integrateHomogeneousMedium(source, extinction, 9.9);
    const offAxisExpected = integrateHomogeneousMedium(source, extinction, 19.8);
    const g = 0.35;
    const sunPhase = (1 - g * g) / (4 * Math.PI * Math.pow(1 + g * g - 2 * g, 1.5));
    const albedo = [0.5, 0.8, 1];
    const ambient = [0.3, 0.4, 0.5];
    const sunColor = [0.9, 0.6, 0.3];
    const sunSource: [number, number, number] = [
        0.2 * albedo[0]! * (ambient[0]! + 2 * sunColor[0]! * sunPhase),
        0.2 * albedo[1]! * (ambient[1]! + 2 * sunColor[1]! * sunPhase),
        0.2 * albedo[2]! * (ambient[2]! + 2 * sunColor[2]! * sunPhase),
    ];
    const sunlitExpected = integrateHomogeneousMedium(sunSource, extinction, 9.9);
    for (let channel = 0; channel < 3; channel++) {
        expect(Math.abs(results.center.injected[channel]! - source[channel]!)).toBeLessThan(0.001);
        expect(Math.abs(results.sunlit.injected[channel]! - sunSource[channel]!)).toBeLessThan(0.001);
        expect(Math.abs(results.sunlit.integrated[channel]! - sunlitExpected.radiance[channel]!)).toBeLessThan(0.003);
        expect(Math.abs(results.center.integrated[channel]! - centerExpected.radiance[channel]!)).toBeLessThan(0.003);
        expect(Math.abs(results.offAxis.integrated[channel]! - offAxisExpected.radiance[channel]!)).toBeLessThan(0.003);
        expect(results.clear.injected[channel]).toBe(0);
        expect(results.clear.integrated[channel]).toBe(0);
    }
    expect(Math.abs(results.center.injected[3]! - 0.2)).toBeLessThan(0.001);
    expect(Math.abs(results.sunlit.injected[3]! - 0.2)).toBeLessThan(0.001);
    expect(Math.abs(results.sunlit.integrated[3]! - sunlitExpected.transmittance[0])).toBeLessThan(0.003);
    expect(Math.abs(results.center.integrated[3]! - centerExpected.transmittance[0])).toBeLessThan(0.003);
    expect(Math.abs(results.offAxis.integrated[3]! - offAxisExpected.transmittance[0])).toBeLessThan(0.003);
    expect(results.clear.integrated[3]).toBe(1);
    expect(results.insideCloud.integrated[3]).toBeLessThan(0.99);
    expect(Math.abs(results.evolvingCloudAtStart.integrated[3]! - results.evolvingCloudAtHalf.integrated[3]!)).toBeGreaterThan(0.001);
    expect(Math.abs(results.evolvingCloudAtHalf.integrated[3]! - results.evolvingCloudAtNext.integrated[3]!)).toBeGreaterThan(0.001);
    expect(Math.abs(results.beforeTransition.integrated[3]! - results.afterTransition.integrated[3]!)).toBeLessThan(0.005);
    for (const empty of [results.aboveCloud, results.outsideCloud, results.clearCloud]) {
        expect(empty.injected).toEqual([0, 0, 0, 0]);
        expect(empty.integrated.slice(0, 3)).toEqual([0, 0, 0]);
        expect(empty.integrated[3]).toBe(1);
    }
});
