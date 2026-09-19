import { createDefaultFluidForceField, validateFluidForceFields, type FluidForceFieldDefinition, type FluidForceFieldVector } from "../forces/force-field-config.js";

export interface FluidForceFieldEditorOptions {
    readonly fields?: readonly FluidForceFieldDefinition[];
    readonly capacity: number;
    readonly onChange: (fields: FluidForceFieldDefinition[]) => void;
    readonly getDefaultPosition?: () => FluidForceFieldVector;
}

export interface FluidForceFieldEditor {
    readonly root: HTMLElement;
    /** @internal */
    _fields: FluidForceFieldDefinition[];
    /** @internal */
    _selected: string | null;
    /** @internal */
    _enabled: boolean;
    /** @internal */
    readonly _capacity: number;
    /** @internal */
    _draw: () => void;
}

export function getFluidForceFieldEditorFields(editor: FluidForceFieldEditor): FluidForceFieldDefinition[] {
    return structuredClone(editor._fields);
}

export function setFluidForceFieldEditorFields(editor: FluidForceFieldEditor, fields: readonly FluidForceFieldDefinition[]): void {
    const validated = validateFluidForceFields(fields);
    if (validated.length > editor._capacity) {
        throw new RangeError(`[fluid] this device supports at most ${editor._capacity} force fields.`);
    }
    editor._fields = validated;
    if (!editor._fields.some((field) => field.id === editor._selected)) {
        editor._selected = editor._fields[0]?.id ?? null;
    }
    editor._draw();
}

export function setFluidForceFieldEditorEnabled(editor: FluidForceFieldEditor, enabled: boolean): void {
    editor._enabled = enabled;
    editor._draw();
}

export function createFluidForceFieldEditor(options: FluidForceFieldEditorOptions): FluidForceFieldEditor {
    if (!Number.isSafeInteger(options.capacity) || options.capacity < 1) {
        throw new RangeError("[fluid] force-field editor capacity must be a positive integer.");
    }
    const root = document.createElement("div");
    root.dataset.fluidForceFields = "true";
    const fields = validateFluidForceFields(options.fields ?? []);
    if (fields.length > options.capacity) {
        throw new RangeError(`[fluid] this device supports at most ${options.capacity} force fields.`);
    }
    const editor: FluidForceFieldEditor = { root, _fields: fields, _selected: fields[0]?.id ?? null, _enabled: true, _capacity: options.capacity, _draw: draw };
    let message = "";
    const inputStyle = "width:100%;min-width:0;box-sizing:border-box;background:#182233;color:#e8eef5;border:1px solid #40536d;border-radius:3px;padding:3px 5px;";

    function commit(next: unknown, selectedId = editor._selected): void {
        try {
            const validated = validateFluidForceFields(next);
            if (validated.length > options.capacity) {
                throw new RangeError(`This device supports at most ${options.capacity} force fields.`);
            }
            options.onChange(structuredClone(validated));
            editor._fields = validated;
            editor._selected = selectedId;
            if (!validated.some((field) => field.id === editor._selected)) {
                editor._selected = validated[0]?.id ?? null;
            }
            message = "";
        } catch (error) {
            message = error instanceof Error ? error.message : String(error);
        }
        draw();
    }

    function update(patch: Record<string, unknown>): void {
        commit(editor._fields.map((field) => (field.id === editor._selected ? { ...field, ...patch } : field)));
    }

    function field(label: string, control: HTMLElement, info?: string): HTMLElement {
        const row = document.createElement("label");
        row.style.cssText = "display:grid;grid-template-columns:115px 1fr;align-items:center;gap:6px;margin:5px 0;";
        row.dataset.forceFieldLabel = label;
        const text = document.createElement("span");
        text.textContent = label;
        row.append(text, control);
        if (info) {
            row.title = info;
        }
        return row;
    }

    function numeric(label: string, value: number, key: string, minimum?: number, disabled = false): HTMLElement {
        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.value = String(value);
        input.style.cssText = inputStyle;
        input.disabled = disabled;
        if (minimum !== undefined) {
            input.min = String(minimum);
        }
        input.onchange = () => {
            if (input.value.trim() === "") {
                message = `${label} requires a number.`;
                draw();
                return;
            }
            update({ [key]: Number(input.value) });
        };
        return field(label, input);
    }

    function checkbox(label: string, checked: boolean, key: string, info?: string): HTMLElement {
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = checked;
        input.onchange = () => update({ [key]: input.checked });
        return field(label, input, info);
    }

    function coordinates(label: string, value: FluidForceFieldVector, key: string): HTMLElement {
        const host = document.createElement("div");
        host.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:3px;";
        for (let axis = 0; axis < 3; axis++) {
            const input = document.createElement("input");
            input.type = "number";
            input.step = "any";
            input.value = String(value[axis]);
            input.style.cssText = inputStyle;
            input.setAttribute("aria-label", `${label} ${["X", "Y", "Z"][axis]}`);
            input.onchange = () => {
                const next = [...value];
                next[axis] = input.value.trim() === "" ? NaN : Number(input.value);
                update({ [key]: next });
            };
            host.append(input);
        }
        return field(label, host, "World-space X, Y and Z coordinates.");
    }

    function button(label: string, action: () => void): HTMLButtonElement {
        const control = document.createElement("button");
        control.type = "button";
        control.textContent = label;
        control.style.cssText = "padding:4px 7px;background:#25354a;color:#e8eef5;border:1px solid #40536d;border-radius:3px;cursor:pointer;";
        control.onclick = action;
        return control;
    }

    function draw(): void {
        root.replaceChildren();
        const list = document.createElement("select");
        list.style.cssText = inputStyle;
        list.setAttribute("aria-label", "Force field");
        for (const entry of editor._fields) {
            const option = document.createElement("option");
            option.value = entry.id;
            option.textContent = `${entry.name}${entry.enabled ? "" : " (disabled)"}`;
            list.append(option);
        }
        list.value = editor._selected ?? "";
        list.disabled = editor._fields.length === 0;
        list.onchange = () => {
            editor._selected = list.value;
            message = "";
            draw();
        };
        root.append(field("Force field", list));
        const actions = document.createElement("div");
        actions.style.cssText = "display:flex;gap:5px;margin:6px 0;";
        for (const type of ["point", "guide"] as const) {
            const add = button(type === "point" ? "Add point" : "Add guide", () => {
                let suffix = 1;
                while (editor._fields.some((entry) => entry.id === `force-${suffix}`)) {
                    suffix++;
                }
                const next = createDefaultFluidForceField(type, `force-${suffix}`, options.getDefaultPosition?.());
                commit([...editor._fields, next], next.id);
            });
            add.disabled = editor._fields.length >= options.capacity;
            actions.append(add);
        }
        const remove = button("Remove force", () => commit(editor._fields.filter((entry) => entry.id !== editor._selected)));
        remove.disabled = editor._selected === null;
        actions.append(remove);
        root.append(actions);
        const selected = editor._fields.find((entry) => entry.id === editor._selected);
        if (selected) {
            const name = document.createElement("input");
            name.value = selected.name;
            name.style.cssText = inputStyle;
            name.onchange = () => update({ name: name.value });
            root.append(field("Name", name), checkbox("Enabled", selected.enabled, "enabled"));
            if (selected.type === "point") {
                root.append(coordinates("Position", selected.position, "position"));
            } else {
                root.append(coordinates("Start", selected.start, "start"), coordinates("End", selected.end, "end"));
            }
            root.append(
                numeric("Strength", selected.strength, "strength"),
                numeric("Falloff power", selected.falloffPower, "falloffPower", 0),
                checkbox("Use min distance", selected.useMinDistance, "useMinDistance"),
                numeric("Min distance", selected.minDistance, "minDistance", 0, !selected.useMinDistance),
                checkbox("Use max distance", selected.useMaxDistance, "useMaxDistance"),
                numeric("Max distance", selected.maxDistance, "maxDistance", 0, !selected.useMaxDistance),
                numeric("Force limit factor", selected.maxForceLimitFactor, "maxForceLimitFactor", 0)
            );
            if (selected.type === "guide") {
                root.append(
                    numeric("Flow strength", selected.flowStrength, "flowStrength"),
                    numeric("Spin strength", selected.spinStrength, "spinStrength"),
                    checkbox("End caps", selected.endCaps, "endCaps", "Allow attraction or repulsion beyond the guide endpoints.")
                );
            }
            const help = document.createElement("div");
            help.style.cssText = "font-size:11px;color:#9fb4cc;margin-top:7px;";
            help.textContent = "Negative strength attracts; positive repels. Positive flow follows Start to End; positive spin follows the right-hand rule.";
            root.append(help);
        }
        if (!editor._enabled) {
            const notice = document.createElement("div");
            notice.textContent = "Configured force fields are not available for this implementation.";
            notice.style.cssText = "color:#e4bd71;margin-top:6px;";
            root.append(notice);
            root.querySelectorAll<HTMLInputElement | HTMLButtonElement | HTMLSelectElement>("input,button,select").forEach((control) => {
                control.disabled = true;
            });
        }
        if (message) {
            const error = document.createElement("div");
            error.setAttribute("role", "alert");
            error.style.cssText = "color:#ff8c8c;margin-top:6px;";
            error.textContent = message;
            root.append(error);
        }
    }
    draw();
    return editor;
}
