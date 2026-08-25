import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const RENDER_TARGET_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/engine/render-target.ts").replace(/\\/g, "/")}`;

test("particle renderer switches custom WGSL and uniforms at runtime", async ({ page }) => {
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="64" height="64"></canvas>
<script type="module">
import {
    createArcRotateCamera,
    createEngine,
    createParticleRenderTask,
    createSceneContext,
} from "${LITE_ENTRY}";
import { buildRenderTarget, createRenderTarget, disposeRenderTarget } from "${RENDER_TARGET_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

const CUSTOM_WGSL = \`
struct Cam {
    vp: mat4x4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    misc: vec4<f32>,
    tint: vec4<f32>,
};
struct Custom {
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> debugSpeed: array<f32>;
@group(0) @binding(3) var<uniform> custom: Custom;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex fn customVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let corner = corners[vertexIndex];
    let center = positions[instanceIndex].xyz;
    let world = center + cam.right.xyz * corner.x * cam.misc.x + cam.up.xyz * corner.y * cam.misc.x;
    var output: VertexOutput;
    output.clip = cam.vp * vec4<f32>(world, 1.0);
    output.color = custom.color + vec4<f32>(debugSpeed[instanceIndex] * 0.0);
    return output;
}

@fragment fn customFragment(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
\`;

async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    const scene = createSceneContext(engine);
    const camera = createArcRotateCamera(0, Math.PI / 2, 3, { x: 0, y: 0, z: 0 });
    scene.camera = camera;
    const colorRT = createRenderTarget({ lbl: "particle-test-color", format: engine.format, samples: 1, size: { width: 64, height: 64 } });
    const depthRT = createRenderTarget({ lbl: "particle-test-depth", dFormat: "depth24plus", samples: 1, size: { width: 64, height: 64 } });
    buildRenderTarget(colorRT, engine);
    buildRenderTarget(depthRT, engine);
    const positionBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const debugBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(positionBuffer, 0, new Float32Array([0, 0, 0, 1]));
    device.queue.writeBuffer(debugBuffer, 0, new Float32Array([0]));
    const sim = { count: 1, renderCount: 1, particleRadius: 0.25, debugNorm: 1, positionBuffer, debugBuffer };
    const task = createParticleRenderTask(engine, scene, { colorRT, depthRT, camera, sim });

    device.pushErrorScope("validation");
    task.setShader({
        code: CUSTOM_WGSL,
        vertexEntryPoint: "customVertex",
        fragmentEntryPoint: "customFragment",
        customUniforms: new Float32Array([1, 0, 0, 1]),
    });
    task.setCustomUniforms(new Float32Array([0, 1, 0, 1]));
    task.record();
    engine._currentEncoder = device.createCommandEncoder();
    task.execute();
    device.queue.submit([engine._currentEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const customError = await device.popErrorScope();

    device.pushErrorScope("validation");
    task.setShader(null);
    task.record();
    engine._currentEncoder = device.createCommandEncoder();
    task.execute();
    device.queue.submit([engine._currentEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const builtInError = await device.popErrorScope();

    device.pushErrorScope("validation");
    task.setShader({ code: "not valid WGSL" });
    task.record();
    const invalidShaderError = await device.popErrorScope();

    canvas.dataset.result = JSON.stringify({
        custom: customError?.message ?? null,
        builtIn: builtInError?.message ?? null,
        invalidSurfaced: invalidShaderError !== null,
    });
    task.dispose();
    positionBuffer.destroy();
    debugBuffer.destroy();
    disposeRenderTarget(colorRT);
    disposeRenderTarget(depthRT);
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await page.waitForFunction(
        () => {
            const element = document.querySelector("#renderCanvas");
            return element?.hasAttribute("data-result") || element?.hasAttribute("data-error");
        },
        undefined,
        { timeout: 90_000 }
    );
    await expect(canvas).toHaveAttribute("data-result", /./);
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    expect(JSON.parse((await canvas.getAttribute("data-result"))!)).toEqual({
        custom: null,
        builtIn: null,
        invalidSurfaced: true,
    });
});

test("polygon surface ray-marches liquid SDF and submerged scene geometry", async ({ page }) => {
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="64" height="64"></canvas>
<script type="module">
import {
    createArcRotateCamera,
    createEngine,
    createFluidPolygonSurfaceTask,
    getProjectionMatrix,
    createSceneContext,
} from "${LITE_ENTRY}";
import { buildRenderTarget, createRenderTarget, disposeRenderTarget } from "${RENDER_TARGET_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    const scene = createSceneContext(engine);
    const camera = createArcRotateCamera(Math.PI / 2, Math.PI / 2, 3, { x: 0, y: 0, z: 0 });
    scene.camera = camera;
    const backgroundRT = createRenderTarget({ lbl: "polygon-test-background", format: engine.format, samples: 1, size: { width: 64, height: 64 } });
    const outputRT = createRenderTarget({ lbl: "polygon-test-output", format: engine.format, samples: 1, size: { width: 64, height: 64 } });
    const depthRT = createRenderTarget({ lbl: "polygon-test-depth", dFormat: "depth24plus", samples: 1, size: { width: 64, height: 64 } });
    buildRenderTarget(backgroundRT, engine);
    buildRenderTarget(outputRT, engine);
    buildRenderTarget(depthRT, engine);

    const vertexData = new Float32Array([
        -0.9, -0.6, 0, 1, 0, 0, 1, 1,
        -0.1, -0.6, 0, 1, 0, 0, 1, 1,
        -0.1,  0.6, 0, 1, 0, 0, 1, 1,
        -0.9,  0.6, 0, 1, 0, 0, 1, 1,
         0.1, -0.6, 0, 1, 0, 0, -1, 1,
         0.9, -0.6, 0, 1, 0, 0, -1, 1,
         0.9,  0.6, 0, 1, 0, 0, -1, 1,
         0.1,  0.6, 0, 1, 0, 0, -1, 1,
        -3, -0.2, -0.4, 1, 0, 0, -1, 1,
         0, -0.2, -0.4, 1, 0, 0, -1, 1,
         0,  0.2, -0.4, 1, 0, 0, -1, 1,
        -3,  0.2, -0.4, 1, 0, 0, -1, 1,
         0, -0.2, -0.8, 1, 0, 0, -1, 1,
         3, -0.2, -0.8, 1, 0, 0, -1, 1,
         3,  0.2, -0.8, 1, 0, 0, -1, 1,
         0,  0.2, -0.8, 1, 0, 0, -1, 1,
    ]);
    const indexData = new Uint32Array([
        0, 1, 2, 0, 2, 3,
        4, 5, 6, 4, 6, 7,
    ]);
    const wireframeIndexData = new Uint32Array([
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
    ]);
    const vertexBuffer = device.createBuffer({ size: vertexData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    const indexBuffer = device.createBuffer({ size: indexData.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    const wireframeIndexBuffer = device.createBuffer({ size: wireframeIndexData.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    const drawIndirect = device.createBuffer({ size: 20, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    const wireframeDrawIndirect = device.createBuffer({ size: 20, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    const gridOrigin = [-1.25, -1.25, -1.5];
    const gridDimensions = [10, 10, 8];
    const gridSpacing = 0.25;
    const sdfData = new Float32Array(gridDimensions[0] * gridDimensions[1] * gridDimensions[2]);
    for (let z = 0; z < gridDimensions[2]; z++) {
        const worldZ = gridOrigin[2] + (z + 0.5) * gridSpacing;
        for (let y = 0; y < gridDimensions[1]; y++) {
            for (let x = 0; x < gridDimensions[0]; x++) {
                const liquidDepth = 1.0;
                sdfData[x + gridDimensions[0] * (y + gridDimensions[1] * z)] = Math.max(worldZ, -liquidDepth - worldZ);
            }
        }
    }
    const liquidSdfBuffer = device.createBuffer({ size: sdfData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(vertexBuffer, 0, vertexData);
    device.queue.writeBuffer(indexBuffer, 0, indexData);
    device.queue.writeBuffer(wireframeIndexBuffer, 0, wireframeIndexData);
    device.queue.writeBuffer(drawIndirect, 0, new Uint32Array([indexData.length, 1, 0, 0, 0]));
    device.queue.writeBuffer(wireframeDrawIndirect, 0, new Uint32Array([wireframeIndexData.length, 1, 0, 0, 0]));
    device.queue.writeBuffer(liquidSdfBuffer, 0, sdfData);
    const sim = {
        polygonSurface: {
            vertexBuffer,
            indexBuffer,
            drawIndirect,
            wireframeIndexBuffer,
            wireframeDrawIndirect,
            indexFormat: "uint32",
            vertexStride: 32,
            triangleCapacity: 4,
            liquidSdfBuffer,
            gridOrigin,
            gridDimensions,
            gridSpacing,
        },
    };
    device.pushErrorScope("validation");
    const task = createFluidPolygonSurfaceTask(engine, scene, {
        bgRT: backgroundRT,
        outRT: outputRT,
        depthRT,
        camera,
        sim,
    });
    task.setEnabled(true);
    task.setWireframe(true);
    task.setRefractionStrength(1);
    task.setDirLight([0, 0, -1]);
    const blackEnvironment = device.createTexture({
        size: [1, 1, 6],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    task.setEnvMap({
        view: blackEnvironment.createView({ dimension: "cube" }),
        sampler: device.createSampler({ magFilter: "linear", minFilter: "linear" }),
    });
    const projection = getProjectionMatrix(camera, 1);
    const opaqueEyeDepth = 3.4;
    const opaqueNdcDepth =
        (projection[10] * opaqueEyeDepth + projection[14]) /
        (projection[11] * opaqueEyeDepth + projection[15]);
    const opaqueVertices = new Float32Array([
        0.02, -1, opaqueNdcDepth,
        1, -1, opaqueNdcDepth,
        1, 1, opaqueNdcDepth,
        0.02, -1, opaqueNdcDepth,
        1, 1, opaqueNdcDepth,
        0.02, 1, opaqueNdcDepth,
    ]);
    const opaqueVertexBuffer = device.createBuffer({
        size: opaqueVertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(opaqueVertexBuffer, 0, opaqueVertices);
    const opaqueShader = device.createShaderModule({ code: \`
        struct VertexOut { @builtin(position) position: vec4<f32> };
        @vertex fn vs(@location(0) position: vec3<f32>) -> VertexOut {
            var out: VertexOut;
            out.position = vec4<f32>(position, 1.0);
            return out;
        }
        @fragment fn fs() -> @location(0) vec4<f32> {
            return vec4<f32>(0.95, 0.08, 0.03, 1.0);
        }
    \` });
    const opaquePipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: opaqueShader,
            entryPoint: "vs",
            buffers: [{
                arrayStride: 12,
                attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
            }],
        },
        fragment: { module: opaqueShader, entryPoint: "fs", targets: [{ format: engine.format }] },
        primitive: { topology: "triangle-list" },
        depthStencil: {
            format: "depth24plus",
            depthWriteEnabled: true,
            depthCompare: "greater",
        },
    });

    engine._currentEncoder = device.createCommandEncoder();
    const backgroundPass = engine._currentEncoder.beginRenderPass({
        colorAttachments: [{ view: backgroundRT._colorView, loadOp: "clear", storeOp: "store", clearValue: { r: 0.8, g: 0.8, b: 0.8, a: 1 } }],
        depthStencilAttachment: { view: depthRT._depthView, depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 0 },
    });
    backgroundPass.setPipeline(opaquePipeline);
    backgroundPass.setVertexBuffer(0, opaqueVertexBuffer);
    backgroundPass.draw(6);
    backgroundPass.end();
    const outputPass = engine._currentEncoder.beginRenderPass({
        colorAttachments: [{ view: outputRT._colorView, loadOp: "clear", storeOp: "store", clearValue: { r: 0.8, g: 0.8, b: 0.8, a: 1 } }],
    });
    outputPass.end();
    task.execute();
    const readback = device.createBuffer({ size: 64 * 64 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const backgroundReadback = device.createBuffer({ size: 64 * 64 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    engine._currentEncoder.copyTextureToBuffer(
        { texture: outputRT._colorTexture },
        { buffer: readback, bytesPerRow: 256 },
        [64, 64, 1],
    );
    engine._currentEncoder.copyTextureToBuffer(
        { texture: backgroundRT._colorTexture },
        { buffer: backgroundReadback, bytesPerRow: 256 },
        [64, 64, 1],
    );
    device.queue.submit([engine._currentEncoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    await backgroundReadback.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(readback.getMappedRange());
    const backgroundPixels = new Uint8Array(backgroundReadback.getMappedRange());
    function averageLuminance(x0, x1) {
        let sum = 0;
        let count = 0;
        for (let y = 24; y < 40; y++) {
            for (let x = x0; x < x1; x++) {
                const offset = (y * 64 + x) * 4;
                sum += (pixels[offset] + pixels[offset + 1] + pixels[offset + 2]) / 3;
                count++;
            }
        }
        return sum / count;
    }
    function sampleLuminance(x, y) {
        const offset = (y * 64 + x) * 4;
        return (pixels[offset] + pixels[offset + 1] + pixels[offset + 2]) / 3;
    }
    function sampleBgra(x, y) {
        const offset = (y * 64 + x) * 4;
        return [pixels[offset], pixels[offset + 1], pixels[offset + 2]];
    }
    const left = averageLuminance(12, 28);
    const right = averageLuminance(36, 52);
    const leftCenter = sampleLuminance(20, 32);
    const rightCenter = sampleLuminance(44, 32);
    const leftOpen = sampleLuminance(20, 25);
    const rightOpen = sampleLuminance(44, 25);
    const rightBgra = sampleBgra(44, 32);
    const backgroundOffset = (32 * 64 + 44) * 4;
    const rightBackgroundBgra = [
        backgroundPixels[backgroundOffset],
        backgroundPixels[backgroundOffset + 1],
        backgroundPixels[backgroundOffset + 2],
    ];
    let wireframePixels = 0;
    for (let offset = 0; offset < pixels.length; offset += 4) {
        if (pixels[offset + 1] > 200 && pixels[offset + 2] < 80) {
            wireframePixels++;
        }
    }
    readback.unmap();
    readback.destroy();
    backgroundReadback.unmap();
    backgroundReadback.destroy();
    task.setWireframe(false);
    task.setFluidColor([1, 1, 1]);
    task.setAbsorption(0);
    task.setShadingMode("ocean");
    engine._currentEncoder = device.createCommandEncoder();
    const oceanOutputPass = engine._currentEncoder.beginRenderPass({
        colorAttachments: [{ view: outputRT._colorView, loadOp: "clear", storeOp: "store", clearValue: { r: 0.8, g: 0.8, b: 0.8, a: 1 } }],
    });
    oceanOutputPass.end();
    task.execute();
    const oceanReadback = device.createBuffer({ size: 64 * 64 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    engine._currentEncoder.copyTextureToBuffer(
        { texture: outputRT._colorTexture },
        { buffer: oceanReadback, bytesPerRow: 256 },
        [64, 64, 1],
    );
    device.queue.submit([engine._currentEncoder.finish()]);
    await oceanReadback.mapAsync(GPUMapMode.READ);
    const oceanPixels = new Uint8Array(oceanReadback.getMappedRange());
    const oceanOffset = (32 * 64 + 44) * 4;
    const oceanRightBgra = [
        oceanPixels[oceanOffset],
        oceanPixels[oceanOffset + 1],
        oceanPixels[oceanOffset + 2],
    ];
    oceanReadback.unmap();
    oceanReadback.destroy();
    task.setAbsorption(8);
    engine._currentEncoder = device.createCommandEncoder();
    const absorbedOceanOutputPass = engine._currentEncoder.beginRenderPass({
        colorAttachments: [{ view: outputRT._colorView, loadOp: "clear", storeOp: "store", clearValue: { r: 0.8, g: 0.8, b: 0.8, a: 1 } }],
    });
    absorbedOceanOutputPass.end();
    task.execute();
    const absorbedOceanReadback = device.createBuffer({ size: 64 * 64 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    engine._currentEncoder.copyTextureToBuffer(
        { texture: outputRT._colorTexture },
        { buffer: absorbedOceanReadback, bytesPerRow: 256 },
        [64, 64, 1],
    );
    device.queue.submit([engine._currentEncoder.finish()]);
    await absorbedOceanReadback.mapAsync(GPUMapMode.READ);
    const absorbedOceanPixels = new Uint8Array(absorbedOceanReadback.getMappedRange());
    const absorbedOceanRightBgra = [
        absorbedOceanPixels[oceanOffset],
        absorbedOceanPixels[oceanOffset + 1],
        absorbedOceanPixels[oceanOffset + 2],
    ];
    absorbedOceanReadback.unmap();
    absorbedOceanReadback.destroy();
    const validationError = await device.popErrorScope();
    canvas.dataset.result = JSON.stringify({
        validationError: validationError?.message ?? null,
        left,
        right,
        leftCenter,
        rightCenter,
        leftOpen,
        rightOpen,
        rightBgra,
        rightBackgroundBgra,
        oceanRightBgra,
        absorbedOceanRightBgra,
        wireframePixels,
    });

    task.dispose();
    vertexBuffer.destroy();
    indexBuffer.destroy();
    wireframeIndexBuffer.destroy();
    drawIndirect.destroy();
    wireframeDrawIndirect.destroy();
    liquidSdfBuffer.destroy();
    opaqueVertexBuffer.destroy();
    blackEnvironment.destroy();
    disposeRenderTarget(backgroundRT);
    disposeRenderTarget(outputRT);
    disposeRenderTarget(depthRT);
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./);
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        validationError: string | null;
        left: number;
        right: number;
        leftCenter: number;
        rightCenter: number;
        leftOpen: number;
        rightOpen: number;
        rightBgra: [number, number, number];
        rightBackgroundBgra: [number, number, number];
        oceanRightBgra: [number, number, number];
        absorbedOceanRightBgra: [number, number, number];
        wireframePixels: number;
    };
    expect(result.validationError).toBeNull();
    expect(result.left).toBeGreaterThan(40);
    expect(result.right).toBeGreaterThan(40);
    expect(Math.abs(result.rightCenter - result.leftCenter), JSON.stringify(result)).toBeGreaterThan(8);
    // The canvas target is BGRA8 on the browser backend, so red is byte 2.
    expect(result.rightBackgroundBgra[2] - result.rightBackgroundBgra[1], JSON.stringify(result)).toBeGreaterThan(100);
    expect(result.rightBgra[2] - result.rightBgra[1], JSON.stringify(result)).toBeGreaterThan(20);
    expect(result.oceanRightBgra[2] - result.oceanRightBgra[1], JSON.stringify(result)).toBeGreaterThan(100);
    expect(Math.abs(result.oceanRightBgra[2] - result.rightBackgroundBgra[2]), JSON.stringify(result)).toBeLessThan(35);
    expect(
        result.absorbedOceanRightBgra.reduce((difference, channel, index) => difference + Math.abs(channel - result.rightBackgroundBgra[index]!), 0),
        JSON.stringify(result)
    ).toBeGreaterThan(result.oceanRightBgra.reduce((difference, channel, index) => difference + Math.abs(channel - result.rightBackgroundBgra[index]!), 0) + 10);
    expect(result.wireframePixels, JSON.stringify(result)).toBeGreaterThan(20);
    expect(result.leftOpen, JSON.stringify(result)).toBeGreaterThan(40);
    expect(result.rightOpen, JSON.stringify(result)).toBeGreaterThan(40);
});
