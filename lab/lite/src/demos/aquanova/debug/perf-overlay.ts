// The `P` overlay: frame rate and per-task GPU cost.
//
// Deliberately independent of the `I` inspect overlay: this is the panel you want open WHILE doing
// something else — walking, liquefying, flipping between AA modes — whereas inspect frees the mouse
// to hover-pick and is a stop-and-look tool. They can both be open at once, so this sits top-right.
//
// Both numbers are averaged over a window rather than shown per frame: an instantaneous frame time
// is unreadable, and rewriting the panel every frame would cost more than it measures.
//
// GPU timing is switched on only while the panel is open — it wraps every pass in timestamp writes,
// so it is not something to leave running.

import { getRenderTaskGpuTimings, setRenderTaskGpuTimingEnabled, type EngineContext } from "babylon-lite";

export interface PerfOverlayOptions {
    engine: EngineContext;
    /** Current portal-culling workload, sampled from the frame that is about to render. */
    portalWorkload?: () => { currentChunk: string; chunks: number; exteriorChunks: number; meshes: number; totalMeshes: number };
    /** Exterior-only chunk IDs, shown to make sky-frustum decisions directly inspectable. */
    exteriorChunks?: () => readonly string[];
    /** Camera view used to reproduce a portal-culling frame. */
    viewpoint?: () => {
        position: { x: number; y: number; z: number };
        target: { x: number; y: number; z: number };
    };
    /** Current fluid workload displayed independently of GPU timestamp availability. */
    fluidWorkload?: () => { simulations: number; pausedSimulations: number; particles: number };
    /**
     * Latest fluid stage times in ms, or null when unavailable.
     *
     * The fluid sim is stepped straight onto the frame's encoder rather than as a frame-graph task,
     * so the engine's per-task timer cannot see it — it needs the fluid's own profiler. Supplying
     * this makes the panel account for the whole frame instead of silently omitting the sim.
     */
    fluidStages?: () => { stages: Record<string, number>; total: number; frameTotal: number } | null;
    /** Called when the panel opens/closes, so the caller can attach or detach the fluid profiler. */
    onToggle?: (on: boolean) => void;
}

export interface PerfOverlay {
    toggle(): void;
    isOn(): boolean;
    /** Per-frame tick. No-op while off. */
    onFrame(deltaMs: number): void;
}

const WINDOW_MS = 500;
const MAX_TASK_LINES = 10;
/** The only fluid stage encoded outside the frame graph: the solver is stepped straight onto the
 *  frame encoder in onBeforeRender, so the frame-graph task timer never sees it. Every other stage
 *  ("Surface") happens inside a task that the task timer already reports. */
const OUTSIDE_GRAPH_STAGE = "Simulation";
/** Below this, a positive envelope−parts gap is indistinguishable from timestamp quantization
 *  (Chrome rounds timestamps to ~100 µs for privacy) and is not worth a line in the panel. */
const UNMEASURED_EPS_MS = 0.02;
/** Name column width. Longer task names are truncated so a row can never widen the panel. */
const MAX_NAME_CH = 20;
/** Panel width. The widest line is the "measured parts" row:
 *  2 indent + 20 name + 8 value + " ms  " + "999%" (percentage is capped and right-aligned to 3 ch)
 *  + " (overlap)" = 49 ch. 52 leaves slack so no state can reflow the panel. */
const PANEL_CH = 52;

/**
 * Create the performance overlay.
 *
 * @param opts - Engine to time, plus optional fluid-profiler wiring.
 * @returns The overlay handle; call `onFrame` every frame and `toggle` from the key handler.
 */
export function createPerfOverlay(opts: PerfOverlayOptions): PerfOverlay {
    const { engine, portalWorkload, exteriorChunks, viewpoint, fluidWorkload, fluidStages, onToggle } = opts;
    let on = false;
    let winFrames = 0;
    let winElapsed = 0;
    let fps = 0;
    let cpuMs = 0;

    const panel = document.createElement("div");
    panel.id = "aq-perf";
    // Fixed width, in ch of the monospace font. The panel is anchored to the right edge, so any
    // change in its longest line would move every column sideways — and the "(overlap)" suffix comes
    // and goes from frame to frame, which made the whole readout jitter. PANEL_CH is sized to the
    // widest line the panel can ever emit (see MAX_NAME_CH and the header formats below).
    panel.style.cssText =
        `position:fixed;right:12px;top:12px;z-index:20;display:none;width:${PANEL_CH}ch;` +
        "font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;color:#d6eefc;background:rgba(0,0,0,.6);" +
        "padding:8px 11px;border-radius:6px;pointer-events:none;white-space:pre;overflow:hidden;";
    document.body.appendChild(panel);

    // Everything shown is averaged over the window, per source.
    //
    // This is not just for readability. The two instruments are INDEPENDENT and publish from
    // different frames — the task snapshot lags "a frame or two", the fluid readback maps one frame
    // later — so differencing their instantaneous values is meaningless during a transient, and did
    // in fact produce a negative `unaccounted` (accounted > envelope, impossible within one frame).
    // Averaging each source over the same wall-clock window makes them comparable again.
    const rowAcc = new Map<string, number>();
    let taskSumAcc = 0;
    let taskSamples = 0;
    let envAcc = 0;
    let outsideAcc = 0;
    let fluidSamples = 0;
    let lastStatus = "";

    const resetAcc = (): void => {
        rowAcc.clear();
        taskSumAcc = 0;
        taskSamples = 0;
        envAcc = 0;
        outsideAcc = 0;
        fluidSamples = 0;
    };

    /** Sample both timers. Called every frame; each returns a cached snapshot, so this is cheap. */
    const sample = (): void => {
        const t = getRenderTaskGpuTimings(engine);
        lastStatus = t.supported ? t.status : "unsupported";
        if (t.status === "available" && t.tasks.length > 0) {
            let sum = 0;
            for (const task of t.tasks) {
                rowAcc.set(task.name, (rowAcc.get(task.name) ?? 0) + task.durationMs);
                sum += task.durationMs;
            }
            taskSumAcc += sum;
            taskSamples++;
        }
        const fluid = fluidStages?.();
        if (fluid && fluid.frameTotal > 0) {
            const outside = fluid.stages[OUTSIDE_GRAPH_STAGE] ?? 0;
            rowAcc.set(`${OUTSIDE_GRAPH_STAGE} *`, (rowAcc.get(`${OUTSIDE_GRAPH_STAGE} *`) ?? 0) + outside);
            outsideAcc += outside;
            envAcc += fluid.frameTotal;
            fluidSamples++;
        }
    };

    /**
     * Whole-frame GPU time, measured END−START rather than by summing parts.
     *
     * This distinction matters: the frame-graph task timer only sees tasks, so anything encoded
     * outside the graph — notably the fluid solver, which is stepped straight onto the frame encoder
     * — is invisible to it. A "total" built by adding up the parts would silently under-report by
     * exactly the amount you forgot to instrument. The envelope cannot: it brackets everything in
     * the frame's command buffer, so `unaccounted` below is the honest size of the blind spot.
     */
    const refresh = (): void => {
        const head = fps > 0 ? `FPS ${fps.toFixed(1)}   cpu ${cpuMs.toFixed(2)} ms/frame` : "FPS —";
        const portals = portalWorkload?.() ?? { currentChunk: "—", chunks: 0, exteriorChunks: 0, meshes: 0, totalMeshes: 0 };
        const chunkHead = `Current chunk ${portals.currentChunk}`;
        const portalHead = `Chunks ${portals.chunks} drawn   Meshes ${portals.meshes} / ${portals.totalMeshes}`;
        const exteriorHead = `Exterior chunks ${portals.exteriorChunks} drawn`;
        const exteriorIds = exteriorChunks?.() ?? [];
        const exteriorIdHead = exteriorIds.map((id) => `  ${id}`).join("\n");
        const view = viewpoint?.();
        const fmtVec = (value: { x: number; y: number; z: number }): string => `${value.x.toFixed(3)}, ${value.y.toFixed(3)}, ${value.z.toFixed(3)}`;
        const viewpointHead = view ? `Position ${fmtVec(view.position)}\nTarget   ${fmtVec(view.target)}` : "";
        const sceneHead = [chunkHead, portalHead, exteriorHead, exteriorIdHead, viewpointHead].filter(Boolean).join("\n");
        const workload = fluidWorkload?.() ?? { simulations: 0, pausedSimulations: 0, particles: 0 };
        const fluidHead = `Fluid ${workload.simulations} sim(s)   ${workload.pausedSimulations} paused   ${workload.particles.toLocaleString("en-US")} particles`;
        if (lastStatus === "unsupported") {
            panel.textContent = `PERF (P)\n${head}\n${sceneHead}\n${fluidHead}\nGPU: timestamp-query unsupported on this device`;
            return;
        }
        if (taskSamples === 0) {
            // "pending" is normal for the first frames: the readback lands a frame or two behind.
            panel.textContent = `PERF (P)\n${head}\n${sceneHead}\n${fluidHead}\nGPU: ${lastStatus || "pending"}`;
            return;
        }
        const taskSum = taskSumAcc / taskSamples;
        const outsideGraph = fluidSamples > 0 ? outsideAcc / fluidSamples : 0;
        const envelope = fluidSamples > 0 ? envAcc / fluidSamples : 0;

        const rows = [...rowAcc.entries()].map(([name, sum]) => ({ name, ms: sum / (name.endsWith(" *") ? Math.max(1, fluidSamples) : taskSamples) })).sort((a, b) => b.ms - a.ms);
        const shown = rows.slice(0, MAX_TASK_LINES);
        let rest = 0;
        for (const r of rows.slice(MAX_TASK_LINES)) {
            rest += r.ms;
        }
        const fmt = (name: string, ms: number): string => `  ${name.slice(0, MAX_NAME_CH).padEnd(MAX_NAME_CH)}${ms.toFixed(3).padStart(8)} ms`;
        const lines = shown.map((r) => fmt(r.name, r.ms));
        if (rest > 0) {
            lines.push(fmt(`+${rows.length - MAX_TASK_LINES} more`, rest));
        }

        const accounted = taskSum + outsideGraph;
        let header: string;
        if (envelope > 0) {
            // The envelope is the authoritative frame cost. The per-pass sum is NOT a partition of
            // it: the GPU pipelines passes, so their individual durations overlap in wall-clock time
            // and the sum routinely EXCEEDS the envelope — measured here at ~2% idle, rising with
            // load (the more work in flight, the more overlap). Reporting envelope − sum as
            // "unaccounted" therefore produced a meaningless negative number.
            //
            // Coverage is the useful signal instead: it should sit near (or just above) 100%.
            // Well under 100% means real work is going unmeasured — which is exactly what a
            // forgotten instrumentation step looks like, and what the envelope exists to catch.
            // Round ONCE and use that for both the number and the label, or a value like 100.4
            // prints "100%" while still being flagged "(overlap)", which reads as a contradiction.
            // Capped at 999 so a pathologically small envelope cannot widen the line past PANEL_CH.
            const pct = Math.min(999, Math.round((accounted / envelope) * 100));
            const gap = envelope - accounted;
            header =
                `GPU ${envelope.toFixed(3)} ms/frame  (envelope: start→end)\n` +
                `  ${"measured parts".padEnd(MAX_NAME_CH)}${accounted.toFixed(3).padStart(8)} ms  ${String(pct).padStart(3)}%${pct > 100 ? " (overlap)" : ""}`;
            if (gap > UNMEASURED_EPS_MS) {
                header += `\n  ${"unmeasured".padEnd(MAX_NAME_CH)}${gap.toFixed(3).padStart(8)} ms`;
            }
        } else {
            // No envelope available (profiler off or unsupported): say so rather than presenting a
            // sum of parts as if it were the frame total. Kept short enough not to widen the panel.
            header = `GPU ${accounted.toFixed(3)} ms  (sum of parts, no envelope)`;
        }
        const foot = outsideGraph > 0 ? "\n  * encoded outside the frame graph" : "";
        panel.textContent = `PERF (P)\n${head}\n${sceneHead}\n${fluidHead}\n${header}\n${lines.join("\n")}${foot}`;
    };

    return {
        isOn: () => on,
        onFrame(deltaMs: number): void {
            if (!on) {
                return;
            }
            sample();
            winFrames++;
            winElapsed += deltaMs;
            if (winElapsed < WINDOW_MS) {
                return;
            }
            fps = (winFrames * 1000) / winElapsed;
            cpuMs = winElapsed / winFrames;
            winFrames = 0;
            winElapsed = 0;
            refresh();
            resetAcc();
        },
        toggle(): void {
            on = !on;
            onToggle?.(on);
            void setRenderTaskGpuTimingEnabled(engine, on).catch(() => {
                // Unsupported device or a readback failure — refresh() reports the status instead.
            });
            winFrames = 0;
            winElapsed = 0;
            fps = 0;
            resetAcc();
            if (on) {
                refresh();
                panel.style.display = "block";
            } else {
                panel.style.display = "none";
            }
        },
    };
}
