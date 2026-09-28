import { _vis } from "../engine/engine.js";
import { drawList, type RenderTaskBase } from "./render-task-base.js";

/** @internal Begin or resume the owning render task, preserving MSAA resolve and depth loads. */
export function beginTaskPass(task: RenderTaskBase, resolveTarget: GPUTextureView | null, sampleCount: number, load: boolean): GPURenderPassEncoder {
    const att = task._colorAttachment;
    const depthLoadOp = load || (task._config.depth ? task._config.depth._eager : task._config.depthClear === false) ? "load" : "clear";
    if (load) {
        att.loadOp = "load";
    }
    const depthAttachment = task._renderPassDescriptor.depthStencilAttachment;
    if (depthAttachment) {
        depthAttachment.depthLoadOp = depthLoadOp;
        if (depthAttachment.stencilLoadOp) {
            depthAttachment.stencilLoadOp = depthLoadOp;
        }
    }
    att.resolveTarget = sampleCount > 1 ? (resolveTarget ?? undefined) : undefined;
    return task.engine._currentEncoder.beginRenderPass(task._renderPassDescriptor);
}

/** @internal Restore the scene bind group and active viewport after a split pass. */
export function setTaskPassState(task: RenderTaskBase, pass: GPURenderPassEncoder): void {
    const cfg = task._config;
    const rt = cfg.rt;
    const camera = cfg.cam ?? task.scene.camera;
    const v = camera?.viewport;
    if (v) {
        const rw = rt._width;
        const rh = rt._height;
        const x = Math.floor(v.x * rw);
        const y = Math.floor((1 - v.y - v.height) * rh);
        const w = Math.ceil((v.x + v.width) * rw) - x;
        const h = Math.ceil((1 - v.y) * rh) - y;
        pass.setViewport(x, y, w, h, 0, 1);
        pass.setScissorRect(x, y, w, h);
    }
    pass.setBindGroup(0, task._sceneBG!);
}

/** @internal Replay cached opaque bundles and direct bindings without drawing transparencies. */
export function drawTaskOpaque(task: RenderTaskBase, pass: GPURenderPassEncoder): number {
    const eng = task.engine;
    const rt = task._config.rt;
    const opaqueBindings = task._opaqueBindings;
    const opaqueBundles = task._ob;

    setTaskPassState(task, pass);
    if (task._lastVis !== _vis || opaqueBundles.length === 0) {
        const desc = rt._descriptor;
        const be = eng._device.createRenderBundleEncoder({
            colorFormats: desc.format ? [desc.format] : [],
            depthStencilFormat: task._targetSignature._depthStencilFormat,
            sampleCount: desc.samples ?? 1,
        });
        be.setBindGroup(0, task._sceneBG!);
        drawList(be, opaqueBindings, eng);
        opaqueBundles[0] = be.finish();
        task._lastVis = _vis;
    }
    let draws = opaqueBindings.length;
    pass.executeBundles(opaqueBundles);
    pass.setBindGroup(0, task._sceneBG!);
    draws += drawList(pass, task._directBindings, eng);
    return draws;
}
