import { setMeshVisible } from "babylon-lite";
import type { SceneSdfSpec } from "babylon-lite";
import type { FluidCtx, FluidDemo } from "../demo.js";
import { ENV_STUDIO_URL } from "../demo.js";

export function createWhiteboardDemo(ctx: FluidCtx): FluidDemo {
    let meshScale = 1;
    let pendingScale = meshScale;
    let scaleFrame = 0;
    const scaleRow = document.createElement("div");
    scaleRow.style.cssText = "margin:6px 0;";
    const scaleHead = document.createElement("div");
    scaleHead.style.cssText = "display:flex;justify-content:space-between;";
    const scaleLabel = document.createElement("span");
    scaleLabel.textContent = "Global mesh scale";
    const scaleValue = document.createElement("span");
    scaleValue.style.cssText = "color:#9fb4cc;";
    scaleValue.textContent = "1×";
    scaleHead.append(scaleLabel, scaleValue);
    const scaleInput = document.createElement("input");
    scaleInput.type = "range";
    scaleInput.min = "-2";
    scaleInput.max = "2";
    scaleInput.step = "0.01";
    scaleInput.value = "0";
    scaleInput.style.cssText = "width:100%;";
    scaleInput.title = "Logarithmic scale from 10^-2 to 10^2";
    scaleInput.oninput = () => {
        pendingScale = 10 ** Number.parseFloat(scaleInput.value);
        meshScale = pendingScale;
        scaleValue.textContent = `${pendingScale.toFixed(pendingScale < 1 ? 2 : 1)}×`;
        if (scaleFrame !== 0) {
            return;
        }
        scaleFrame = requestAnimationFrame(() => {
            scaleFrame = 0;
            ctx.setTransientSceneMeshScale(pendingScale);
        });
    };
    scaleRow.append(scaleHead, scaleInput);

    const sdf: SceneSdfSpec = {
        struct: "struct SceneSdfParams { unused: vec4<f32>, };",
        sdf: `fn sceneSdf(pt: vec3<f32>, dt: f32) -> f32 {
    return 1.0e6 + sceneSdfParams.unused.x;
}`,
        buffer: ctx.sceneSdfBuffer,
    };

    return {
        key: "whiteboard",
        label: "Whiteboard",
        interactiveForceScale: 0.1,
        envUrl: ENV_STUDIO_URL,
        methodIndependentAuthoring: true,
        usesQualityPresets: false,
        useGridFloor: true,
        sdf,
        writeSdfParams(): void {
            ctx.engine._device.queue.writeBuffer(ctx.sceneSdfBuffer, 0, new Float32Array(4));
        },
        flow: () => ({ emitters: [], sinks: [] }),
        onEnter(): void {
            setMeshVisible(ctx.ground, false);
            ctx.setTransientSceneMeshScale(meshScale);
        },
        onLeave(): void {
            if (scaleFrame !== 0) {
                cancelAnimationFrame(scaleFrame);
                scaleFrame = 0;
            }
            ctx.setTransientSceneMeshScale(1);
            setMeshVisible(ctx.ground, true);
        },
        update(): void {
            /* Empty authoring workspace. */
        },
        demoParams: () => [],
        applyParam(): void {
            /* No demo-specific parameters. */
        },
        extraControls: () => [scaleRow],
    };
}
