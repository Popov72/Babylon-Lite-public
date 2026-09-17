# Module: Fluid Offline Rendering

> Package paths: `packages/babylon-lite/src/fluid/core/fluid-facade.ts`, `lab/lite/src/demos/fluid.ts`, `scripts/render-fluid-offline.ts`

## Purpose

Run an exported fluid preset at a fixed simulation timestep without real-time pacing. A dedicated headless Chrome process owns the WebGPU device, advances the solver as quickly as the GPU permits, renders only requested output frames, and writes numbered PNG files.

Offline rendering does not make an individual solver step cheaper. It removes the interactive frame deadline and skips scene/surface rendering between output frames.

Realtime, deterministic capture, and offline rendering drive imported glTF animation, emitter-source transforms, animated collision transforms, and fluid stepping from the same simulation delta. Realtime wall-clock stalls are capped rather than advancing imported animation ahead of the fluid.

## Public API Surface

```ts
export async function submitFluidSimulationStep(simulation: FluidSimulation, deltaSeconds: number): Promise<void>;
```

The function:

1. rejects disposed simulations and invalid timesteps;
2. rejects calls made while the engine is recording a live frame;
3. creates one command encoder;
4. records exactly one backend frame through the simulation facade;
5. submits it;
6. resolves after all preceding GPU work completes.

Raw WebGPU handles remain internal.

## Fluid Demo Offline Protocol

The Fluid demo enters offline mode with:

```text
?offline=1&fixedDt=<seconds>&demo=<key>&method=<method>&quality=<quality>
```

After scene preparation, the live render loop stops and the canvas publishes:

```text
data-offline-ready="true"
data-offline-status="idle"
data-offline-step="<completed simulation steps>"
data-offline-simulation-time="<seconds>"
```

The CLI dispatches a `fluid-offline-run` `CustomEvent` on `#renderCanvas`:

```ts
interface FluidOfflineRunDetail {
    token: string;
    steps: number;
    render: boolean;
}
```

The demo serializes requests. For each requested simulation step it:

1. advances imported glTF animation or the active built-in demo;
2. refreshes animated emitter and collision transforms;
3. installs the active force field;
4. submits one fixed fluid step and waits for the GPU;
5. advances the simulation lifecycle.

When `render` is true, one normal frame-graph frame is recorded with zero render delta and no additional simulation step. Completion is published through:

```text
data-offline-completed-token="<token>"
data-offline-status="idle"
```

Errors are surfaced through:

```text
data-offline-error-token="<token>"
data-offline-error="<message>"
data-offline-status="error"
```

## CLI

```text
pnpm render:fluid-offline -- \
  --input "D:\path\fluid.json" \
  --output "D:\path\frames" \
  --frames 300 \
  --fixed-dt 0.0166666667 \
  --steps-per-frame 1 \
  --output-fps 60 \
  --width 1280 \
  --height 720
```

The CLI:

- serves `lab/` from an ephemeral local port unless `--url` is supplied;
- launches installed Chrome through Playwright with WebGPU enabled;
- reads the JSON manifest and automatically selects referenced external GLB/SDF siblings;
- defaults `fixed-dt` to the exported Blender simulation FPS and video FPS to the exported render FPS when those values are available; `--fixed-dt` and `--output-fps` override them independently;
- waits for preset import and offline preparation;
- prints the resolved live simulation method, Resolution Divisions, grid dimensions and cell size, active and total-capacity particle counts, simulation GPU allocation, device per-buffer limit, paging, pressure solver, render mode, foam capacity, and requested output cadence;
- advances `steps-per-frame` fixed simulation steps per image;
- waits for GPU completion before each capture;
- saves `frame-000001.png`, `frame-000002.png`, and so on;
- closes the browser after capture, invokes `ffmpeg` from `PATH`, and writes `fluid-offline.mp4` inside the output directory;
- limits encoding to the frames captured by the current run, so stale higher-numbered PNGs in a reused directory cannot extend the video;
- prints progress, simulation time, elapsed wall time, and the final video path.

Pass `--no-video` to retain only the lossless PNG sequence. Missing or failed ffmpeg encoding is reported explicitly and leaves the captured PNG files intact.

The CLI resolves ffmpeg from `--ffmpeg <executable>`, then `FFMPEG_PATH`, then the standard Windows installation `C:\Program Files\ffmpeg\bin\ffmpeg.exe`, and finally `PATH`.

The Fluid demo bundle must already exist:

```text
pnpm build:bundle-demo fluid --debug
```

Preset import has no fixed grid-cell or particle-count safety substitution. The Fluid host preserves authored settings, validates the resulting resources against the active WebGPU device, and reports the requested and supported per-buffer memory when an allocation does not fit.

## State and Limitations

- The implementation generates an image sequence and an H.264 MP4, not a restartable solver cache.
- The simulation remains in GPU memory for the duration of the process.
- A dedicated browser process isolates JavaScript and rendering from the interactive lab, but both processes may still compete for the same physical GPU.
- Each simulation step is submitted separately and synchronized for correctness with animated transforms, runtime uniform writes, paged-grid status, and asynchronous solver bookkeeping.
- PNG output is lossless. MP4 output uses ffmpeg/libx264 with `yuv420p`; `--no-video` skips this step.

## Test Specification

- Import an embedded preset and an external JSON/GLB/SDF set.
- Advance multiple fixed steps with the live render loop stopped.
- Confirm simulation time and active state advance only by explicit requests.
- Capture at least two non-empty 1280x720 PNG frames.
- Confirm invalid input, missing sibling resources, import errors, and browser-side offline errors fail the CLI explicitly.
