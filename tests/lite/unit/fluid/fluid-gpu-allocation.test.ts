import { describe, expect, it, vi } from "vitest";

import { createDiffuseCountTracker, createGpuBuffersAtomically } from "../../../../packages/babylon-lite/src/fluid/core/sim-common";

function mockBuffer(label: string): GPUBuffer {
    return {
        label,
        destroy: vi.fn(),
    } as unknown as GPUBuffer;
}

describe("fluid GPU allocation", () => {
    it("destroys partial replacements when a related allocation fails", () => {
        const created = [mockBuffer("first"), mockBuffer("second")];
        const device = {
            createBuffer: vi
                .fn()
                .mockReturnValueOnce(created[0])
                .mockReturnValueOnce(created[1])
                .mockImplementationOnce(() => {
                    throw new Error("allocation failed");
                }),
        } as unknown as GPUDevice;

        expect(() =>
            createGpuBuffersAtomically(device, [
                { label: "first", size: 4, usage: GPUBufferUsage.STORAGE },
                { label: "second", size: 8, usage: GPUBufferUsage.STORAGE },
                { label: "third", size: 12, usage: GPUBufferUsage.STORAGE },
            ])
        ).toThrow("allocation failed");
        expect(created[0]!.destroy).toHaveBeenCalledOnce();
        expect(created[1]!.destroy).toHaveBeenCalledOnce();
    });

    it("returns a complete replacement set without retiring it", () => {
        const created = [mockBuffer("first"), mockBuffer("second")];
        const device = {
            createBuffer: vi.fn().mockReturnValueOnce(created[0]).mockReturnValueOnce(created[1]),
        } as unknown as GPUDevice;

        expect(
            createGpuBuffersAtomically(device, [
                { label: "first", size: 4, usage: GPUBufferUsage.STORAGE },
                { label: "second", size: 8, usage: GPUBufferUsage.STORAGE },
            ])
        ).toEqual(created);
        expect(created[0]!.destroy).not.toHaveBeenCalled();
        expect(created[1]!.destroy).not.toHaveBeenCalled();
    });

    it("cleans up count-tracker buffers when pipeline creation fails", () => {
        const created = [mockBuffer("counter"), mockBuffer("readback-0"), mockBuffer("readback-1")];
        const device = {
            createBuffer: vi.fn().mockReturnValueOnce(created[0]).mockReturnValueOnce(created[1]).mockReturnValueOnce(created[2]),
            createShaderModule: vi.fn().mockReturnValue({}),
            createComputePipeline: vi
                .fn()
                .mockReturnValueOnce({})
                .mockImplementationOnce(() => {
                    throw new Error("pipeline failed");
                }),
        } as unknown as GPUDevice;

        expect(() => createDiffuseCountTracker(device, "test")).toThrow("pipeline failed");
        for (const buffer of created) {
            expect(buffer.destroy).toHaveBeenCalledOnce();
        }
    });

    it("can retry count-tracker configuration after bind-group creation fails", () => {
        const trackerBuffers = [mockBuffer("counter"), mockBuffer("readback-0"), mockBuffer("readback-1")];
        const pipeline = { getBindGroupLayout: vi.fn().mockReturnValue({}) };
        const createBindGroup = vi
            .fn()
            .mockReturnValueOnce({})
            .mockImplementationOnce(() => {
                throw new Error("bind group failed");
            })
            .mockReturnValueOnce({})
            .mockReturnValueOnce({});
        const device = {
            createBuffer: vi.fn().mockReturnValueOnce(trackerBuffers[0]).mockReturnValueOnce(trackerBuffers[1]).mockReturnValueOnce(trackerBuffers[2]),
            createShaderModule: vi.fn().mockReturnValue({}),
            createComputePipeline: vi.fn().mockReturnValue(pipeline),
            createBindGroup,
            queue: { writeBuffer: vi.fn() },
        } as unknown as GPUDevice;
        const tracker = createDiffuseCountTracker(device, "test");
        const diffuse = mockBuffer("diffuse");
        const active = mockBuffer("active");

        expect(() => tracker.configure(diffuse, 64, active)).toThrow("bind group failed");
        expect(() => tracker.configure(diffuse, 64, active)).not.toThrow();
        expect(createBindGroup).toHaveBeenCalledTimes(4);
        tracker.dispose();
    });
});
