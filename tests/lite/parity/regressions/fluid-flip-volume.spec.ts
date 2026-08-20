import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const LITE_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/index.ts").replace(/\\/g, "/")}`;
const FLIP_ENTRY = `/@fs/${resolve(__dirname, "../../../../packages/babylon-lite/src/fluid/flip-sim.ts").replace(/\\/g, "/")}`;

test("FLIP preserves different authored liquid volumes", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto("/");
    await page.setContent(`
<canvas id="renderCanvas" width="16" height="16"></canvas>
<script type="module">
import { createEngine } from "${LITE_ENTRY}";
import { createFlipSim } from "${FLIP_ENTRY}";

const canvas = document.getElementById("renderCanvas");
window.addEventListener("error", (event) => { canvas.dataset.error = event.message; });
window.addEventListener("unhandledrejection", (event) => { canvas.dataset.error = event.reason?.message ?? String(event.reason); });

function lattice(sizeY) {
    const spacing = 0.125;
    const positions = [];
    for (let y = 0.125; y < sizeY; y += spacing) {
        for (let z = -1.4375; z <= 1.4375; z += spacing) {
            for (let x = -1.4375; x <= 1.4375; x += spacing) {
                positions.push(x, y, z);
            }
        }
    }
    return new Float32Array(positions);
}

async function settledHeight(engine, sizeY) {
    const initialPositions = lattice(sizeY);
    const sim = createFlipSim(engine, {
        count: initialPositions.length / 3,
        initialPositions,
        boundsMin: [-2, 0, -2],
        boundsMax: [2, 8, 2],
        groundY: 0,
        dx: 0.25,
        particleRadius: 0.09,
        gravity: 9.8,
        flipRatio: 0.95,
        pressureIterations: 40,
        pressureRelaxation: 0.8,
        minSubsteps: 1,
        maxSubDt: 1 / 60,
    });
    for (let frame = 0; frame < 180; frame++) {
        const encoder = engine._device.createCommandEncoder();
        sim.step(encoder, 1 / 60);
        engine._device.queue.submit([encoder.finish()]);
        if (frame % 15 === 14) {
            await engine._device.queue.onSubmittedWorkDone();
        }
    }
    await engine._device.queue.onSubmittedWorkDone();
    const bytes = sim.count * 16;
    const readback = engine._device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = engine._device.createCommandEncoder();
    encoder.copyBufferToBuffer(sim.positionBuffer, 0, readback, 0, bytes);
    engine._device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const positions = new Float32Array(readback.getMappedRange().slice(0));
    const ys = [];
    for (let index = 0; index < sim.count; index++) {
        const y = positions[index * 4 + 1];
        if (Number.isFinite(y) && y > -1000) {
            ys.push(y);
        }
    }
    ys.sort((a, b) => a - b);
    const percentile = (p) => ys[Math.min(ys.length - 1, Math.floor(ys.length * p))];
    const result = { count: ys.length, height: percentile(0.99) - percentile(0.01), top: percentile(0.99) };
    readback.unmap();
    readback.destroy();
    sim.dispose();
    return result;
}

async function main() {
    const engine = await createEngine(canvas);
    const small = await settledHeight(engine, 2);
    const large = await settledHeight(engine, 6);
    canvas.dataset.result = JSON.stringify({ small, large });
}

main().catch((error) => { canvas.dataset.error = error?.message ?? String(error); });
</script>`);

    const canvas = page.locator("#renderCanvas");
    await expect(canvas).toHaveAttribute("data-result", /./, { timeout: 150_000 });
    await expect(canvas).not.toHaveAttribute("data-error", /./);
    const result = JSON.parse((await canvas.getAttribute("data-result"))!) as {
        small: { count: number; height: number; top: number };
        large: { count: number; height: number; top: number };
    };
    expect(result.large.count).toBeGreaterThan(result.small.count * 2.9);
    expect(result.large.height).toBeGreaterThan(result.small.height * 2);
});
