export interface FluidReferencePumpOptions {
    render: (deltaMs: number) => void;
    shouldStep: () => boolean;
    step: () => Promise<void>;
    onError: (error: unknown) => void;
    suspend: () => void;
    resume: () => void;
}

export interface FluidReferencePump {
    pending: Promise<void> | null;
    transitions: number;
    error: unknown;
    _running: boolean;
    _frame: number;
    _previousTime: number | null;
    _tail: Promise<void>;
    _options: FluidReferencePumpOptions;
    _tick: (time: number) => void;
}

function schedule(pump: FluidReferencePump): void {
    if (pump._running && !pump._frame && !pump.pending && pump.transitions === 0) {
        pump._frame = requestAnimationFrame(pump._tick);
    }
}

function tick(pump: FluidReferencePump, time: number): void {
    pump._frame = 0;
    if (!pump._running || pump.pending || pump.transitions > 0) {
        return;
    }
    const delta = pump._previousTime === null ? 0 : Math.max(0, time - pump._previousTime);
    pump._previousTime = time;
    try {
        pump._options.render(delta);
    } catch (error) {
        pump.error = error;
        stopFluidReferencePump(pump);
        pump._options.onError(error);
        return;
    }
    if (!pump._running || pump.transitions > 0) {
        return;
    }
    if (!pump._options.shouldStep()) {
        schedule(pump);
        return;
    }
    // Defer until renderFrame has cleared its encoder, then keep the canvas at that complete state.
    pump.pending = Promise.resolve()
        .then(pump._options.step)
        .catch((error: unknown) => {
            pump.error = error;
            pump._options.onError(error);
        })
        .finally(() => {
            pump.pending = null;
            schedule(pump);
        });
}

export function createFluidReferencePump(options: FluidReferencePumpOptions): FluidReferencePump {
    const pump: FluidReferencePump = {
        pending: null,
        transitions: 0,
        error: null,
        _running: false,
        _frame: 0,
        _previousTime: null,
        _tail: Promise.resolve(),
        _options: options,
        _tick: (time) => tick(pump, time),
    };
    return pump;
}

export function startFluidReferencePump(pump: FluidReferencePump): void {
    if (!pump._running) {
        pump._running = true;
        pump._previousTime = null;
    }
    schedule(pump);
}

export function stopFluidReferencePump(pump: FluidReferencePump): void {
    pump._running = false;
    if (pump._frame) {
        cancelAnimationFrame(pump._frame);
        pump._frame = 0;
    }
}

/** Apply scene/solver mutations only after the current physical operation has released its resources. */
export function queueFluidReferenceChange(pump: FluidReferencePump, action: () => void | Promise<void>): Promise<void> {
    stopFluidReferencePump(pump);
    pump._options.suspend();
    pump.transitions++;
    const pending = pump.pending;
    const operation = pump._tail.then(async () => {
        await pending;
        await action();
    });
    pump._tail = operation
        .catch((error: unknown) => {
            pump.error = error;
            pump._options.onError(error);
        })
        .finally(() => {
            pump.transitions--;
            if (pump.transitions === 0) {
                pump._options.resume();
            }
        });
    return pump._tail;
}
