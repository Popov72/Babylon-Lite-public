import type { ForceFieldSpec } from "./sim-common.js";

export interface FluidForcePass {
    readonly pipeline: GPUComputePipeline;
    readonly bindGroup: GPUBindGroup;
}

export interface FluidForcePasses {
    spec: ForceFieldSpec | null;
    commands: FluidForcePass[];
    readonly pipelines: Map<string, GPUComputePipeline>;
    readonly bindings: WeakMap<ForceFieldSpec, { source: string; buffer: GPUBuffer; command: FluidForcePass }>;
    readonly buildSource: (spec: ForceFieldSpec) => string;
    readonly createPipeline: (source: string) => GPUComputePipeline;
    readonly createBindGroup: (pipeline: GPUComputePipeline, spec: ForceFieldSpec) => GPUBindGroup;
}

export function createFluidForcePasses(
    buildSource: FluidForcePasses["buildSource"],
    createPipeline: FluidForcePasses["createPipeline"],
    createBindGroup: FluidForcePasses["createBindGroup"]
): FluidForcePasses {
    return { spec: null, commands: [], pipelines: new Map(), bindings: new WeakMap(), buildSource, createPipeline, createBindGroup };
}

export function updateFluidForcePasses(state: FluidForcePasses, spec: ForceFieldSpec | null): void {
    if (state.spec === spec) {
        return;
    }
    const specs: ForceFieldSpec[] = [];
    const visiting = new Set<ForceFieldSpec>();
    const visit = (entry: ForceFieldSpec): void => {
        if (visiting.has(entry)) {
            throw new Error("[fluid] cyclic force-field composition.");
        }
        visiting.add(entry);
        specs.push(entry);
        for (const child of entry.additional ?? []) {
            visit(child);
        }
        visiting.delete(entry);
    };
    if (spec) {
        visit(spec);
    }
    const commands = specs.map((entry) => {
        const source = state.buildSource(entry);
        const cached = state.bindings.get(entry);
        if (cached?.source === source && cached.buffer === entry.buffer) {
            return cached.command;
        }
        let pipeline = state.pipelines.get(source);
        if (!pipeline) {
            pipeline = state.createPipeline(source);
            state.pipelines.set(source, pipeline);
        }
        const command = { pipeline, bindGroup: state.createBindGroup(pipeline, entry) };
        state.bindings.set(entry, { source, buffer: entry.buffer, command });
        return command;
    });
    state.spec = spec;
    state.commands = commands;
}
