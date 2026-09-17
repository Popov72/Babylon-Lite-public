import type { FluidEmitter, FluidFlowConfig, FluidShape, FluidSink, FluidTransform } from "../core/sim-common.js";
import { MAX_FLUID_EMITTERS, MAX_FLUID_SINKS } from "../core/sim-common.js";

export type FluidFlowObjectKind = "emitter" | "sink";

export type FluidFlowEditorChange =
    | { type: "flow"; rebuildEditor: boolean }
    | { type: "object"; kind: FluidFlowObjectKind; object: FluidEmitter | FluidSink; rebuildEditor: boolean }
    | { type: "metadata"; kind: FluidFlowObjectKind; object: FluidEmitter | FluidSink; rebuildEditor: true };

export interface FluidFlowEditorVisuals {
    readonly getWireframeVisible: (kind: FluidFlowObjectKind) => boolean;
    readonly setWireframeVisible: (kind: FluidFlowObjectKind, visible: boolean) => void;
    readonly getGizmoVisible: (kind: FluidFlowObjectKind) => boolean;
    readonly setGizmoVisible: (kind: FluidFlowObjectKind, visible: boolean) => void;
}

export interface FluidFlowEditorOptions {
    emittersHost: HTMLElement;
    sinksHost: HTMLElement;
    flow: FluidFlowConfig;
    onChange: (flow: FluidFlowConfig, change: FluidFlowEditorChange) => void;
    onRefresh?(): void;
    /** Required so shared visual controls cannot silently disappear in one host. */
    visuals: FluidFlowEditorVisuals;
    /** Explicitly suppresses the Wireframe and Gizmo rows for hosts that cannot render them. */
    hideVisualControls?: boolean;
    getEmitterRateMode?: () => "occupancy-refill" | "unlimited-toggle";
    getInitialEmitterParticleCount?: (emitter: FluidEmitter) => number | undefined;
    onInitialEmitterParticleCountDisplayed?: (count: number | undefined) => void;
}

export interface FluidFlowEditor {
    readonly emittersHost: HTMLElement;
    readonly sinksHost: HTMLElement;
    /** @internal */
    readonly setFlow: (flow: FluidFlowConfig) => void;
    /** Restrict authoring to reset-time Initial emitters. Existing unsupported entries remain
     *  visible so the user can disable, convert, or delete them; none are silently removed. */
    readonly setInitialOnly: (enabled: boolean) => void;
    /** @internal */
    readonly clearSelection: (refreshEditor?: boolean) => void;
    /** @internal */
    readonly refresh: () => void;
    /** @internal */
    readonly refreshComputedValues: () => void;
    /** @internal */
    readonly getSelected: (kind: FluidFlowObjectKind) => FluidEmitter | FluidSink | undefined;
}

export function setFluidFlowEditorFlow(editor: FluidFlowEditor, flow: FluidFlowConfig): void {
    editor.setFlow(flow);
}

export function clearFluidFlowEditorSelection(editor: FluidFlowEditor, refreshEditor?: boolean): void {
    editor.clearSelection(refreshEditor);
}

export function refreshFluidFlowEditor(editor: FluidFlowEditor): void {
    editor.refresh();
}

export function refreshFluidFlowEditorComputedValues(editor: FluidFlowEditor): void {
    editor.refreshComputedValues();
}

export function getFluidFlowEditorSelection(editor: FluidFlowEditor, kind: FluidFlowObjectKind): FluidEmitter | FluidSink | undefined {
    return editor.getSelected(kind);
}

const identityFlowTransform = (): FluidTransform => ({ position: [0, 1, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] });

const defaultFlowShape = (type: FluidShape["type"]): FluidShape => {
    if (type === "sphere") {
        return { type, radius: 0.25 };
    }
    if (type === "cylinder") {
        return { type, radius: 0.5, height: 0.5 };
    }
    if (type === "cone") {
        return { type, bottomRadius: 0.5, topRadius: 0, height: 1 };
    }
    if (type === "capsule") {
        return { type, radius: 0.5, height: 1 };
    }
    if (type === "polygonPrism") {
        return {
            type,
            points: [
                [-0.5, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5],
            ],
            thickness: 0.25,
        };
    }
    return { type: "box", size: [1, 1, 1] };
};

const flowButton = (label: string, onClick: () => void): HTMLButtonElement => {
    const button = document.createElement("button");
    button.textContent = label;
    button.style.cssText = "padding:3px 7px;background:#25354a;color:#e8eef5;border:1px solid #40536d;border-radius:3px;cursor:pointer;";
    button.onclick = onClick;
    return button;
};

const flowField = (label: string, control: HTMLElement, info?: string): HTMLElement => {
    const row = document.createElement("label");
    row.style.cssText = "display:grid;grid-template-columns:105px 1fr;align-items:center;gap:6px;margin:4px 0;";
    row.dataset.flowFieldLabel = label;
    const text = document.createElement("span");
    text.textContent = label;
    if (info) {
        row.title = info;
        const icon = document.createElement("span");
        icon.textContent = " ⓘ";
        icon.style.cssText = "color:#6d7f95;cursor:help;";
        text.appendChild(icon);
    }
    row.append(text, control);
    return row;
};

const numberInput = (value: number, onChange: (value: number) => void, step = 0.1, min?: number): HTMLInputElement => {
    let committed = value;
    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "decimal";
    input.dataset.step = String(step);
    if (min !== undefined) {
        input.dataset.min = String(min);
    }
    input.value = String(value);
    input.style.cssText = "width:100%;min-width:0;box-sizing:border-box;background:#182233;color:#e8eef5;border:1px solid #40536d;border-radius:3px;padding:3px 5px;";
    input.onchange = () => {
        const next = Number(input.value.trim().replace(",", "."));
        if (Number.isFinite(next) && (min === undefined || next >= min)) {
            committed = next;
            input.value = String(next);
            onChange(next);
        } else {
            input.value = String(committed);
        }
    };
    input.onkeydown = (event) => {
        if (event.key === "Enter") {
            input.blur();
            return;
        }
        if (event.key === "ArrowUp" || event.key === "ArrowDown") {
            event.preventDefault();
            const parsed = Number(input.value.trim().replace(",", "."));
            const base = Number.isFinite(parsed) ? parsed : committed;
            const next = Math.max(min ?? Number.NEGATIVE_INFINITY, base + (event.key === "ArrowUp" ? step : -step));
            committed = next;
            input.value = String(next);
            onChange(next);
        }
    };
    return input;
};

const textInput = (value: string, onChange: (value: string) => void): HTMLInputElement => {
    const input = document.createElement("input");
    input.type = "text";
    input.value = value;
    input.style.cssText = "width:100%;box-sizing:border-box;background:#182233;color:#e8eef5;border:1px solid #40536d;border-radius:3px;padding:2px 4px;";
    input.onchange = () => onChange(input.value.trim() || value);
    return input;
};

const selectInput = <T extends string>(value: T, values: readonly T[], onChange: (value: T) => void): HTMLSelectElement => {
    const select = document.createElement("select");
    select.style.cssText = "width:100%;background:#182233;color:#e8eef5;border:1px solid #40536d;border-radius:3px;padding:2px;";
    for (const item of values) {
        const option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        select.appendChild(option);
    }
    select.value = value;
    select.onchange = () => onChange(select.value as T);
    return select;
};

const vec3Editor = (value: [number, number, number], onChange: (value: [number, number, number]) => void, step = 0.1): HTMLElement => {
    const current: [number, number, number] = [...value];
    const host = document.createElement("div");
    host.style.cssText = "display:grid;grid-template-columns:repeat(3,1fr);gap:3px;";
    for (let axis = 0; axis < 3; axis++) {
        host.appendChild(
            numberInput(
                value[axis]!,
                (next) => {
                    current[axis] = next;
                    onChange([...current]);
                },
                step
            )
        );
    }
    return host;
};

const quatFromEulerDegrees = (value: [number, number, number]): [number, number, number, number] => {
    const x = (value[0] * Math.PI) / 360;
    const y = (value[1] * Math.PI) / 360;
    const z = (value[2] * Math.PI) / 360;
    const sx = Math.sin(x);
    const cx = Math.cos(x);
    const sy = Math.sin(y);
    const cy = Math.cos(y);
    const sz = Math.sin(z);
    const cz = Math.cos(z);
    return [sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz];
};

const eulerDegreesFromQuat = (q: [number, number, number, number]): [number, number, number] => {
    const [x, y, z, w] = q;
    const rx = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
    const sy = Math.max(-1, Math.min(1, 2 * (w * y - z * x)));
    const ry = Math.asin(sy);
    const rz = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    return [(rx * 180) / Math.PI, (ry * 180) / Math.PI, (rz * 180) / Math.PI];
};

export function createFluidFlowEditor(options: FluidFlowEditorOptions): FluidFlowEditor {
    let flow = options.flow;
    let initialOnly = false;
    let selectedEmitterId: string | null = null;
    let selectedSinkId: string | null = null;
    let initialEmitterParticleCountValue: HTMLElement | null = null;
    let initialEmitterParticleCountEmitter: FluidEmitter | null = null;

    const getSelected = (kind: FluidFlowObjectKind): FluidEmitter | FluidSink | undefined => {
        const selectedId = kind === "emitter" ? selectedEmitterId : selectedSinkId;
        return kind === "emitter" ? flow.emitters.find((emitter) => emitter.id === selectedId) : flow.sinks.find((sink) => sink.id === selectedId);
    };

    const uniqueFlowId = (prefix: string): string => {
        const used = new Set([...flow.emitters.map((emitter) => emitter.id), ...flow.sinks.map((sink) => sink.id)]);
        let index = 1;
        while (used.has(`${prefix}-${index}`)) {
            index++;
        }
        return `${prefix}-${index}`;
    };

    const finishChange = (change: FluidFlowEditorChange): void => {
        options.onChange(flow, change);
        if (change.rebuildEditor) {
            refresh();
        } else if (change.type === "object") {
            options.onRefresh?.();
        }
    };

    const updateObject = (kind: FluidFlowObjectKind, object: FluidEmitter | FluidSink, rebuildEditor = true): void => {
        finishChange({ type: "object", kind, object, rebuildEditor });
    };

    const updateMetadata = (kind: FluidFlowObjectKind, object: FluidEmitter | FluidSink): void => {
        finishChange({ type: "metadata", kind, object, rebuildEditor: true });
    };

    const appendTransformEditor = (host: HTMLElement, kind: FluidFlowObjectKind, target: FluidEmitter | FluidSink): void => {
        const transform = target.transform;
        host.append(
            flowField(
                "Position",
                vec3Editor(transform.position, (value) => {
                    transform.position = value;
                    updateObject(kind, target, false);
                }),
                "Grid-local offset from Grid position."
            ),
            flowField(
                "Rotation °",
                vec3Editor(
                    eulerDegreesFromQuat(transform.rotation),
                    (value) => {
                        transform.rotation = quatFromEulerDegrees(value);
                        updateObject(kind, target, false);
                    },
                    1
                )
            ),
            flowField(
                "Scale",
                vec3Editor(
                    transform.scale,
                    (value) => {
                        transform.scale = value;
                        updateObject(kind, target, false);
                    },
                    0.05
                )
            )
        );
    };

    const appendShapeEditor = (host: HTMLElement, kind: FluidFlowObjectKind, target: FluidEmitter | FluidSink): void => {
        const shapeTypes: FluidShape["type"][] = ["box", "sphere", "cylinder", "cone", "capsule", "polygonPrism"];
        host.appendChild(
            flowField(
                "Shape",
                selectInput(target.shape.type, shapeTypes, (type) => {
                    target.shape = defaultFlowShape(type);
                    updateObject(kind, target);
                })
            )
        );
        const shape = target.shape;
        if (shape.type === "box") {
            host.appendChild(
                flowField(
                    "Size",
                    vec3Editor(shape.size, (value) => {
                        shape.size = value;
                        updateObject(kind, target, false);
                    })
                )
            );
        } else if (shape.type === "sphere") {
            host.appendChild(
                flowField(
                    "Radius",
                    numberInput(
                        shape.radius,
                        (value) => {
                            shape.radius = value;
                            updateObject(kind, target, false);
                        },
                        0.05,
                        0
                    )
                )
            );
        } else if (shape.type === "cylinder") {
            host.append(
                flowField(
                    "Radius",
                    numberInput(
                        shape.radius,
                        (value) => {
                            shape.radius = value;
                            updateObject(kind, target, false);
                        },
                        0.05,
                        0
                    )
                ),
                flowField(
                    "Inner radius",
                    numberInput(
                        shape.innerRadius ?? 0,
                        (value) => {
                            shape.innerRadius = value;
                            updateObject(kind, target, false);
                        },
                        0.05,
                        0
                    )
                ),
                flowField(
                    "Height",
                    numberInput(
                        shape.height,
                        (value) => {
                            shape.height = value;
                            updateObject(kind, target, false);
                        },
                        0.05,
                        0
                    )
                )
            );
        } else if (shape.type === "cone") {
            host.append(
                flowField(
                    "Bottom radius",
                    numberInput(
                        shape.bottomRadius,
                        (value) => {
                            shape.bottomRadius = value;
                            updateObject(kind, target, false);
                        },
                        0.05,
                        0
                    )
                ),
                flowField(
                    "Top radius",
                    numberInput(
                        shape.topRadius,
                        (value) => {
                            shape.topRadius = value;
                            updateObject(kind, target, false);
                        },
                        0.05,
                        0
                    )
                ),
                flowField(
                    "Height",
                    numberInput(
                        shape.height,
                        (value) => {
                            shape.height = value;
                            updateObject(kind, target, false);
                        },
                        0.05,
                        0
                    )
                )
            );
        } else if (shape.type === "capsule") {
            host.append(
                flowField(
                    "Radius",
                    numberInput(
                        shape.radius,
                        (value) => {
                            shape.radius = value;
                            updateObject(kind, target, false);
                        },
                        0.05,
                        0
                    )
                ),
                flowField(
                    "Total height",
                    numberInput(
                        shape.height,
                        (value) => {
                            shape.height = value;
                            updateObject(kind, target, false);
                        },
                        0.05,
                        0
                    )
                )
            );
        } else {
            const points = document.createElement("textarea");
            points.value = shape.points.map((point) => `${point[0]},${point[1]}`).join("; ");
            points.rows = 3;
            points.style.cssText = "width:100%;box-sizing:border-box;background:#182233;color:#e8eef5;border:1px solid #40536d;border-radius:3px;";
            points.onchange = () => {
                const parsed = points.value
                    .split(";")
                    .map((entry) => entry.split(",").map(Number))
                    .filter((entry) => entry.length === 2 && entry.every(Number.isFinite))
                    .map((entry) => [entry[0]!, entry[1]!] as [number, number]);
                if (parsed.length >= 3) {
                    shape.points = parsed;
                    updateObject(kind, target, false);
                }
            };
            host.append(
                flowField("Points x,z", points),
                flowField(
                    "Thickness",
                    numberInput(
                        shape.thickness,
                        (value) => {
                            shape.thickness = value;
                            updateObject(kind, target, false);
                        },
                        0.05,
                        0
                    )
                )
            );
        }
    };

    const flowWireframeCheckbox = (kind: FluidFlowObjectKind): HTMLInputElement => {
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = options.visuals.getWireframeVisible(kind);
        checkbox.onchange = () => {
            options.visuals.setWireframeVisible(kind, checkbox.checked);
            options.onRefresh?.();
        };
        return checkbox;
    };

    const flowGizmoCheckbox = (kind: FluidFlowObjectKind): HTMLInputElement => {
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = options.visuals.getGizmoVisible(kind);
        checkbox.onchange = () => {
            options.visuals.setGizmoVisible(kind, checkbox.checked);
            refresh();
        };
        return checkbox;
    };

    const flowList = (kind: FluidFlowObjectKind): HTMLSelectElement => {
        const objects = kind === "emitter" ? flow.emitters : flow.sinks;
        const selectedId = kind === "emitter" ? selectedEmitterId : selectedSinkId;
        const list = document.createElement("select");
        list.size = Math.min(8, Math.max(3, objects.length));
        list.style.cssText = "width:100%;background:#182233;color:#e8eef5;border:1px solid #40536d;border-radius:3px;";
        for (const object of objects) {
            const option = document.createElement("option");
            option.value = object.id;
            option.textContent = object.name;
            list.appendChild(option);
        }
        if (selectedId) {
            list.value = selectedId;
        }
        list.onchange = () => {
            if (kind === "emitter") {
                selectedEmitterId = list.value || null;
            } else {
                selectedSinkId = list.value || null;
            }
            refresh();
        };
        return list;
    };

    const flowButtons = (kind: FluidFlowObjectKind): HTMLElement => {
        const host = document.createElement("div");
        host.style.cssText = "display:flex;gap:4px;flex-wrap:wrap;margin:5px 0;";
        const selected = getSelected(kind);
        const objects = kind === "emitter" ? flow.emitters : flow.sinks;
        const limit = kind === "emitter" ? MAX_FLUID_EMITTERS : MAX_FLUID_SINKS;
        const add = flowButton(kind === "emitter" ? "+ Emitter" : "+ Sink", () => {
            const id = uniqueFlowId(kind);
            if (kind === "emitter") {
                flow.emitters.push({
                    id,
                    name: "New emitter",
                    enabled: true,
                    behavior: initialOnly ? "initial" : "inflow",
                    transform: identityFlowTransform(),
                    shape: defaultFlowShape("box"),
                    sampling: "volume",
                    velocity: [0, 1, 0],
                    velocitySpace: "local",
                    spread: 0,
                });
                selectedEmitterId = id;
            } else {
                flow.sinks.push({
                    id,
                    name: "New sink",
                    enabled: true,
                    mode: "delete",
                    transform: identityFlowTransform(),
                    shape: defaultFlowShape("box"),
                    targets: [],
                    volumeRate: 1,
                });
                selectedSinkId = id;
            }
            finishChange({ type: "flow", rebuildEditor: true });
        });
        add.disabled = objects.length >= limit || (initialOnly && kind === "sink");
        if (initialOnly && kind === "sink") {
            add.title = "This implementation is initial-only and does not support sinks.";
        }
        const duplicate = flowButton("Duplicate", () => {
            if (!selected) {
                return;
            }
            const copy = structuredClone(selected);
            copy.id = uniqueFlowId(kind);
            copy.name += " copy";
            if (kind === "emitter") {
                flow.emitters.push(copy as FluidEmitter);
                selectedEmitterId = copy.id;
            } else {
                flow.sinks.push(copy as FluidSink);
                selectedSinkId = copy.id;
            }
            finishChange({ type: "flow", rebuildEditor: true });
        });
        duplicate.disabled = !selected || objects.length >= limit || (initialOnly && (kind === "sink" || (kind === "emitter" && (selected as FluidEmitter).behavior === "inflow")));
        if (initialOnly && kind === "sink") {
            duplicate.title = "This implementation is initial-only and does not support sinks.";
        } else if (initialOnly && kind === "emitter" && (selected as FluidEmitter | undefined)?.behavior === "inflow") {
            duplicate.title = "Convert this emitter to Initial before duplicating it for an initial-only implementation.";
        }
        const remove = flowButton("Delete", () => {
            if (!selected) {
                return;
            }
            if (kind === "emitter") {
                flow.emitters = flow.emitters.filter((emitter) => emitter.id !== selected.id);
                for (const sink of flow.sinks) {
                    sink.targets = sink.targets.filter((id) => id !== selected.id);
                }
                selectedEmitterId = null;
            } else {
                flow.sinks = flow.sinks.filter((sink) => sink.id !== selected.id);
                selectedSinkId = null;
            }
            finishChange({ type: "flow", rebuildEditor: true });
        });
        remove.disabled = !selected;
        host.append(add, duplicate, remove);
        return host;
    };

    const commonFlowEditor = (kind: FluidFlowObjectKind, object: FluidEmitter | FluidSink): HTMLElement => {
        const editor = document.createElement("div");
        editor.style.cssText = "border-top:1px solid #33445b;margin-top:6px;padding-top:5px;";
        const enabled = document.createElement("input");
        enabled.type = "checkbox";
        enabled.checked = object.enabled;
        const unsupportedActivation = initialOnly && (kind === "sink" || (object as FluidEmitter).behavior === "inflow");
        enabled.disabled = unsupportedActivation && !object.enabled;
        if (unsupportedActivation) {
            enabled.title = kind === "sink" ? "Sinks are unavailable in initial-only mode." : "Inflow emitters are unavailable in initial-only mode.";
        }
        enabled.onchange = () => {
            object.enabled = enabled.checked;
            updateObject(kind, object);
        };
        if (!options.hideVisualControls) {
            editor.append(flowField("Wireframe", flowWireframeCheckbox(kind)), flowField("Gizmo", flowGizmoCheckbox(kind)));
        }
        editor.append(
            flowField("Enabled", enabled),
            flowField(
                "Name",
                textInput(object.name, (value) => {
                    object.name = value;
                    updateMetadata(kind, object);
                })
            )
        );
        appendTransformEditor(editor, kind, object);
        appendShapeEditor(editor, kind, object);
        return editor;
    };

    const refreshEmitterUI = (): void => {
        initialEmitterParticleCountValue = null;
        initialEmitterParticleCountEmitter = null;
        options.onInitialEmitterParticleCountDisplayed?.(undefined);
        const fillCapacity = document.createElement("input");
        fillCapacity.type = "checkbox";
        fillCapacity.checked = flow.initialEmittersFillCapacity === true;
        fillCapacity.onchange = () => {
            flow.initialEmittersFillCapacity = fillCapacity.checked;
            finishChange({ type: "flow", rebuildEditor: false });
        };
        if (!flow.emitters.some((emitter) => emitter.id === selectedEmitterId)) {
            selectedEmitterId = flow.emitters[0]?.id ?? null;
        }
        const emitter = flow.emitters.find((item) => item.id === selectedEmitterId);
        const editor = emitter ? commonFlowEditor("emitter", emitter) : document.createElement("div");
        if (emitter) {
            const sourceAndNormalEnabled =
                emitter.sourceVelocityFactor !== undefined || emitter.normalVelocity !== undefined || (emitter.sourceNode === undefined && emitter.sourceVelocity !== undefined);
            const sourceAndNormal = document.createElement("input");
            sourceAndNormal.type = "checkbox";
            sourceAndNormal.checked = sourceAndNormalEnabled;
            sourceAndNormal.onchange = () => {
                if (sourceAndNormal.checked) {
                    emitter.sourceVelocityFactor ??= 1;
                    emitter.normalVelocity ??= 0;
                } else {
                    delete emitter.sourceVelocity;
                    delete emitter.sourceVelocityFactor;
                    delete emitter.normalVelocity;
                }
                updateObject("emitter", emitter);
            };
            if (emitter.sourceNode) {
                const sourceNode = document.createElement("code");
                sourceNode.textContent = emitter.sourceNode;
                sourceNode.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
                sourceNode.title = emitter.sourceNode;
                editor.appendChild(flowField("Source mesh", sourceNode, "Imported GLB node whose animated world transform drives this analytical emitter."));
            }
            const usesOccupancyRefill = options.getEmitterRateMode?.() !== "unlimited-toggle";
            const rateToggle = document.createElement("input");
            rateToggle.type = "checkbox";
            rateToggle.checked = usesOccupancyRefill ? emitter.volumeRate !== undefined : emitter.volumeRate === undefined;
            rateToggle.onchange = () => {
                const rateLimited = usesOccupancyRefill ? rateToggle.checked : !rateToggle.checked;
                emitter.volumeRate = rateLimited ? (emitter.volumeRate ?? 1) : undefined;
                updateObject("emitter", emitter);
            };
            const behavior = selectInput(emitter.behavior, ["initial", "inflow"] as const, (value) => {
                emitter.behavior = value;
                updateObject("emitter", emitter);
            });
            if (initialOnly) {
                const inflowOption = Array.from(behavior.options).find((option) => option.value === "inflow");
                if (inflowOption) {
                    inflowOption.disabled = true;
                }
                behavior.title = "This implementation supports reset-time Initial emitters only.";
            }
            editor.prepend(
                flowField(
                    "Behavior",
                    behavior,
                    initialOnly
                        ? "This implementation supports reset-time Initial emitters only. Existing Inflow entries remain visible so they can be converted or disabled."
                        : undefined
                ),
                flowField(
                    "Sampling",
                    selectInput(emitter.sampling, ["volume", "surface"] as const, (value) => {
                        emitter.sampling = value;
                        updateObject("emitter", emitter);
                    })
                )
            );
            const initialParticleCount = options.getInitialEmitterParticleCount?.(emitter);
            if (initialParticleCount !== undefined) {
                const particleCount = document.createElement("output");
                particleCount.style.cssText = "font-variant-numeric:tabular-nums;";
                particleCount.dataset.fluidInitialEmitterParticleCount = emitter.id;
                initialEmitterParticleCountValue = particleCount;
                initialEmitterParticleCountEmitter = emitter;
                editor.prepend(
                    flowField(
                        "FLIP particles",
                        particleCount,
                        "Read-only marker allocation accepted for this Initial emitter after Reset simulation, including clipping to the FLIP domain, grid cell size, Markers per cell, other Initial emitters, and Particle capacity."
                    )
                );
                refreshComputedValues();
            }
            if (emitter.behavior === "inflow") {
                editor.appendChild(
                    flowField(
                        "Delay before start",
                        numberInput(
                            emitter.delayBeforeStart ?? 0,
                            (value) => {
                                emitter.delayBeforeStart = value;
                                updateObject("emitter", emitter, false);
                            },
                            0.1,
                            0
                        ),
                        "Simulation-time seconds to wait after reset before this Inflow emits or accepts recycled particles."
                    )
                );
                editor.appendChild(
                    flowField(
                        usesOccupancyRefill ? "Limit volume rate" : "Unlimited",
                        rateToggle,
                        usesOccupancyRefill
                            ? "When disabled, FLIP refills empty marker space inside the emitter shape. Enable this to cap the replenished liquid volume per simulation second; emission velocity remains independent."
                            : undefined
                    )
                );
                if (emitter.volumeRate !== undefined) {
                    editor.appendChild(
                        flowField(
                            "Volume / second",
                            numberInput(
                                emitter.volumeRate,
                                (value) => {
                                    emitter.volumeRate = value;
                                    updateObject("emitter", emitter, false);
                                },
                                0.1,
                                0
                            ),
                            usesOccupancyRefill
                                ? "Maximum world-space liquid volume replenished per simulation second. The actual amount may be lower when the emitter region is already full."
                                : undefined
                        )
                    );
                }
            }
            editor.append(
                flowField(
                    "Velocity (XYZ)",
                    vec3Editor(emitter.velocity, (value) => {
                        emitter.velocity = value;
                        updateObject("emitter", emitter, false);
                    }),
                    "Authored launch velocity added to every emitted particle. This value is independent of the source mesh's motion."
                ),
                flowField(
                    "Velocity space",
                    selectInput(emitter.velocitySpace, ["local", "world"] as const, (value) => {
                        emitter.velocitySpace = value;
                        updateObject("emitter", emitter);
                    })
                ),
                flowField(
                    "Spread",
                    numberInput(
                        emitter.spread,
                        (value) => {
                            emitter.spread = value;
                            updateObject("emitter", emitter, false);
                        },
                        0.05,
                        0
                    )
                ),
                flowField("Source + normal", sourceAndNormal, "Opt-in inherited velocity from the linked Source mesh plus analytical shape-normal velocity.")
            );
            if (sourceAndNormalEnabled) {
                editor.append(
                    flowField(
                        "Source factor",
                        numberInput(emitter.sourceVelocityFactor ?? 1, (value) => {
                            emitter.sourceVelocityFactor = value;
                            updateObject("emitter", emitter, false);
                        }),
                        "Multiplier applied to velocity derived each frame from the linked Source mesh's animation."
                    ),
                    flowField(
                        "Normal velocity",
                        numberInput(emitter.normalVelocity ?? 0, (value) => {
                            emitter.normalVelocity = value;
                            updateObject("emitter", emitter, false);
                        }),
                        "Speed along the emitter shape's outward analytical normal. Negative values point inward."
                    )
                );
            }
        }
        const notice = document.createElement("div");
        notice.dataset.fluidInitialOnlyNotice = "emitters";
        notice.style.cssText = "display:none;margin:0 0 8px;color:#a8bed3;font-size:11px;line-height:1.4;";
        notice.textContent = "Initial-only implementation: new emitters are created as Initial, and Inflow cannot be selected.";
        notice.style.display = initialOnly ? "block" : "none";
        options.emittersHost.replaceChildren(notice, flowField("Initial emitters fill capacity", fillCapacity), flowList("emitter"), flowButtons("emitter"), editor);
    };

    const refreshSinkUI = (): void => {
        if (!flow.sinks.some((sink) => sink.id === selectedSinkId)) {
            selectedSinkId = flow.sinks[0]?.id ?? null;
        }
        const sink = flow.sinks.find((item) => item.id === selectedSinkId);
        const editor = sink ? commonFlowEditor("sink", sink) : document.createElement("div");
        if (sink) {
            const operation = selectInput(sink.mode ?? "delete", ["delete", "recycle"] as const, (value) => {
                sink.mode = value;
                if (value === "recycle" && sink.targets.length === 0) {
                    const firstInflow = flow.emitters.find((emitter) => emitter.behavior === "inflow");
                    sink.targets = firstInflow ? [firstInflow.id] : [];
                }
                updateObject("sink", sink);
            });
            const targets = document.createElement("div");
            targets.style.cssText = "display:grid;gap:2px;";
            for (const emitter of flow.emitters.filter((item) => item.behavior === "inflow")) {
                const label = document.createElement("label");
                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.checked = sink.targets.includes(emitter.id);
                checkbox.onchange = () => {
                    sink.targets = checkbox.checked ? [...sink.targets, emitter.id] : sink.targets.filter((id) => id !== emitter.id);
                    updateObject("sink", sink);
                };
                label.append(checkbox, ` ${emitter.name}`);
                targets.appendChild(label);
            }
            const rateMode = sink.perParticleRecycleRate !== undefined ? "perParticle" : sink.volumeRate !== undefined ? "volume" : "all";
            const rate = selectInput(rateMode, ["all", "volume", "perParticle"] as const, (value) => {
                if (value === "all") {
                    sink.volumeRate = undefined;
                    sink.perParticleRecycleRate = undefined;
                } else if (value === "volume") {
                    sink.volumeRate ??= 1;
                    sink.perParticleRecycleRate = undefined;
                } else {
                    sink.volumeRate = undefined;
                    sink.perParticleRecycleRate ??= 1;
                }
                updateObject("sink", sink);
            });
            editor.append(flowField("Behavior", operation));
            editor.appendChild(
                flowField(
                    "Delay before start",
                    numberInput(
                        sink.delayBeforeStart ?? 0,
                        (value) => {
                            sink.delayBeforeStart = value;
                            updateObject("sink", sink, false);
                        },
                        0.1,
                        0
                    ),
                    "Simulation-time seconds to wait after reset before this Sink starts deleting or recycling particles."
                )
            );
            if ((sink.mode ?? "delete") === "recycle") {
                editor.append(flowField("Targets", targets));
            }
            editor.append(flowField("Capture limit", rate));
            if (sink.volumeRate !== undefined) {
                editor.appendChild(
                    flowField(
                        "Volume / second",
                        numberInput(
                            sink.volumeRate,
                            (value) => {
                                sink.volumeRate = value;
                                updateObject("sink", sink, false);
                            },
                            0.1,
                            0
                        )
                    )
                );
            } else if (sink.perParticleRecycleRate !== undefined) {
                editor.appendChild(
                    flowField(
                        "Per-particle / second",
                        numberInput(
                            sink.perParticleRecycleRate,
                            (value) => {
                                sink.perParticleRecycleRate = value;
                                updateObject("sink", sink, false);
                            },
                            0.05,
                            0
                        )
                    )
                );
            }
        }
        const notice = document.createElement("div");
        notice.dataset.fluidInitialOnlyNotice = "sinks";
        notice.style.cssText = "display:none;margin:0 0 8px;color:#a8bed3;font-size:11px;line-height:1.4;";
        notice.textContent = "Sinks are unavailable in this initial-only implementation. Existing entries remain visible so they can be disabled or deleted.";
        notice.style.display = initialOnly ? "block" : "none";
        options.sinksHost.replaceChildren(notice, flowList("sink"), flowButtons("sink"), editor);
    };

    function refreshComputedValues(): void {
        if (!initialEmitterParticleCountValue || !initialEmitterParticleCountEmitter) {
            options.onInitialEmitterParticleCountDisplayed?.(undefined);
            return;
        }
        const count = options.getInitialEmitterParticleCount?.(initialEmitterParticleCountEmitter);
        if (count === undefined) {
            options.onInitialEmitterParticleCountDisplayed?.(undefined);
            return;
        }
        initialEmitterParticleCountValue.textContent = count.toLocaleString();
        options.onInitialEmitterParticleCountDisplayed?.(count);
    }

    function refresh(): void {
        refreshEmitterUI();
        refreshSinkUI();
        options.onRefresh?.();
    }

    const editor: FluidFlowEditor = {
        emittersHost: options.emittersHost,
        sinksHost: options.sinksHost,
        setFlow(nextFlow) {
            flow = nextFlow;
            refresh();
        },
        setInitialOnly(enabled) {
            initialOnly = enabled;
            refresh();
        },
        clearSelection(refreshEditor = true) {
            selectedEmitterId = null;
            selectedSinkId = null;
            if (refreshEditor) {
                refresh();
            }
        },
        refresh,
        refreshComputedValues,
        getSelected,
    };
    refresh();
    return editor;
}
