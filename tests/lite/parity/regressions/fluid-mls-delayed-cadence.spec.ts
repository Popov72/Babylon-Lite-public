import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const MLS_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/solvers/mls-mpm-sim.ts").replace(/\\/g, "/")}`;

test("MLS-MPM preserves elapsed force and gravity while status maps are delayed", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createMlsMpmSim } from "${MLS_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

function delayedStatusEngine(device, gates) {
    return {
        _device: new Proxy(device, {
            get(target, prop) {
                if (prop === "createBuffer") {
                    return (descriptor) => {
                        const readable = descriptor.label === "mpm-render-vel";
                        const buffer = target.createBuffer(readable ? { ...descriptor, usage: descriptor.usage | GPUBufferUsage.COPY_SRC } : descriptor);
                        if (descriptor.label?.includes("mpm-accumulation-status-readback")) {
                            const nativeMap = buffer.mapAsync.bind(buffer);
                            Object.defineProperty(buffer, "mapAsync", {
                                configurable: true,
                                value: (...args) => new Promise((resolve, reject) => {
                                    gates.push(() => nativeMap(...args).then(resolve, reject));
                                }),
                            });
                        }
                        return buffer;
                    };
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
            },
        }),
    };
}

async function submitStep(device, sim, dt) {
    const encoder = device.createCommandEncoder();
    sim.step(encoder, dt);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
}

async function main() {
    const engine = await createEngine(canvas);
    const device = engine._device;
    const gates = [];
    const count = 64;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        positions[i * 3] = 1.25 + (i % 4) * 0.5;
        positions[i * 3 + 1] = 2.25 + (Math.floor(i / 4) % 4) * 0.5;
        positions[i * 3 + 2] = 1.25 + Math.floor(i / 16) * 0.5;
    }
    const sim = createMlsMpmSim(delayedStatusEngine(device, gates), {
        count,
        initialPositions: positions,
        boundsMin: [0, 0, 0],
        boundsMax: [6, 6, 6],
        gridDim: [12, 12, 12],
        dx: 0.5,
        gravity: 1,
        restDensity: 3,
        stiffness: 0,
        viscosity: 0,
        damping: 1,
        affineDamping: 1,
        groundDamp: 1,
        substeps: 2,
        maxSubDt: 0.01,
    });
    const forceBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(forceBuffer, 0, new Float32Array([1, 0, 0, 0]));
    sim.setForceField({
        struct: "struct ForceFieldParams { acceleration: vec4<f32>, };",
        wgsl: "fn externalForce(position: vec3<f32>, velocity: vec3<f32>, dt: f32) -> vec3<f32> { return forceFieldParams.acceleration.xyz * dt; }",
        buffer: forceBuffer,
    });

    await submitStep(device, sim, 0.01);
    await submitStep(device, sim, 0.01);
    await submitStep(device, sim, 0.01);
    await submitStep(device, sim, 0.02);
    await submitStep(device, sim, 0);
    await gates[0]();
    await new Promise((resolve) => setTimeout(resolve, 0));
    await submitStep(device, sim, 0.01);

    const readback = device.createBuffer({ size: count * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const copy = device.createCommandEncoder();
    copy.copyBufferToBuffer(sim.velocityBuffer, 0, readback, 0, count * 16);
    device.queue.submit([copy.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const velocities = new Float32Array(readback.getMappedRange());
    let x = 0;
    let y = 0;
    for (let i = 0; i < count; i++) {
        x += velocities[i * 4];
        y += velocities[i * 4 + 1];
    }
    readback.unmap();
    readback.destroy();
    sim.dispose();
    forceBuffer.destroy();
    return { gateCount: gates.length, x: x / count, y: y / count };
}

main()
    .then((result) => {
        canvas.dataset.result = JSON.stringify(result);
    })
    .catch((error) => {
        canvas.dataset.error = error?.message ?? String(error);
    });
</script>`);

    await page.waitForFunction(() => {
        const canvas = document.querySelector("canvas");
        return Boolean(canvas?.dataset.result || canvas?.dataset.error);
    });
    const state = await page.locator("canvas").evaluate((node) => ({ error: node.dataset.error, result: node.dataset.result }));
    expect(state.error).toBeUndefined();
    const result = JSON.parse(state.result!);
    expect(result.gateCount).toBe(2);
    expect(result.x).toBeGreaterThan(0.05);
    expect(result.x).toBeLessThan(0.07);
    expect(result.y).toBeLessThan(-0.05);
    expect(result.y).toBeGreaterThan(-0.07);
});
