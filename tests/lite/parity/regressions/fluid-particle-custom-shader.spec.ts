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
    await expect(canvas).toHaveAttribute("data-result", /./);
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    expect(JSON.parse((await canvas.getAttribute("data-result"))!)).toEqual({
        custom: null,
        builtIn: null,
        invalidSurfaced: true,
    });
});
