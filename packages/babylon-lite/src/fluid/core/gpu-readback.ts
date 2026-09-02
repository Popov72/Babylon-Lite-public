/** @internal */
export type GpuReadbackState = "idle" | "copied" | "mapping";

/** @internal */
export interface GpuReadbackSlot<P> {
    readonly buffer: GPUBuffer;
    state: GpuReadbackState;
    generation: number;
    byteLength: number;
    payload: P;
}

/** @internal */
export interface GpuReadbackPoolOptions<P> {
    readonly device: GPUDevice;
    readonly label: string;
    readonly slotCount: number;
    readonly byteLength: number;
    readonly defaultPayload: P;
    onComplete(view: ArrayBuffer, slot: Readonly<GpuReadbackSlot<P>>): void;
    onError(error: unknown, slot: Readonly<GpuReadbackSlot<P>>): void;
}

/** @internal Shared generation-aware asynchronous GPU readback ring. */
export class GpuReadbackPool<P> {
    private readonly slots: GpuReadbackSlot<P>[] = [];
    private _disposed = false;

    constructor(private readonly options: GpuReadbackPoolOptions<P>) {
        const { device, label, slotCount, byteLength, defaultPayload } = options;
        for (let index = 0; index < slotCount; index++) {
            this.slots.push({
                buffer: device.createBuffer({
                    label: `${label}-${index}`,
                    size: byteLength,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                }),
                state: "idle",
                generation: 0,
                byteLength,
                payload: defaultPayload,
            });
        }
    }

    get disposed(): boolean {
        return this._disposed;
    }

    acquire(): GpuReadbackSlot<P> | null {
        if (this._disposed) {
            return null;
        }
        return this.slots.find((slot) => slot.state === "idle") ?? null;
    }

    submit(slot: GpuReadbackSlot<P>, generation: number, payload: P, byteLength: number = this.options.byteLength): void {
        slot.generation = generation;
        slot.payload = payload;
        slot.byteLength = Math.max(0, Math.min(this.options.byteLength, byteLength));
        slot.state = "copied";
    }

    pump(): void {
        if (this._disposed) {
            return;
        }
        for (const slot of this.slots) {
            if (slot.state !== "copied") {
                continue;
            }
            slot.state = "mapping";
            void slot.buffer
                .mapAsync(GPUMapMode.READ, 0, slot.byteLength)
                .then(() => {
                    if (this._disposed) {
                        return;
                    }
                    try {
                        this.options.onComplete(slot.buffer.getMappedRange(0, slot.byteLength), slot);
                    } finally {
                        if (slot.buffer.mapState === "mapped") {
                            slot.buffer.unmap();
                        }
                    }
                })
                .catch((error: unknown) => {
                    if (!this._disposed) {
                        this.options.onError(error, slot);
                    }
                })
                .finally(() => {
                    if (!this._disposed) {
                        slot.state = "idle";
                        slot.payload = this.options.defaultPayload;
                    }
                });
        }
    }

    dispose(): void {
        if (this._disposed) {
            return;
        }
        this._disposed = true;
        for (const slot of this.slots) {
            slot.buffer.destroy();
        }
        this.slots.length = 0;
    }
}
