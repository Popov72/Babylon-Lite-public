import { setMeshVisible } from "babylon-lite";
import type { SceneSdfSpec } from "babylon-lite";
import type { FluidCtx, FluidDemo } from "../demo.js";
import { ENV_STUDIO_URL } from "../demo.js";

export function createWhiteboardDemo(ctx: FluidCtx): FluidDemo {
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
        },
        onLeave(): void {
            setMeshVisible(ctx.ground, true);
        },
        update(): void {
            /* Empty authoring workspace. */
        },
        demoParams: () => [],
        applyParam(): void {
            /* No demo-specific parameters. */
        },
        extraControls: () => [],
    };
}
