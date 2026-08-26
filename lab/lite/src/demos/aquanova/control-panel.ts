import { LAB_DEBUG } from "./debug-flag.js";

type MaybePromise = void | Promise<void>;

interface ToggleOption {
    readonly label: string;
    readonly get: () => boolean;
    readonly set: (on: boolean) => MaybePromise;
}

interface ActionOption {
    readonly label: string;
    readonly run: () => MaybePromise;
    readonly status?: () => string;
}

interface VectorEditOption {
    readonly set: (value: readonly [number, number, number]) => void;
}

export interface WeaponTransformValues {
    readonly position: readonly [number, number, number];
    readonly rotationDegrees: readonly [number, number, number];
    readonly scale: readonly [number, number, number];
    readonly localGuidePosition: readonly [number, number, number];
    readonly localGuideRotationDegrees: readonly [number, number, number];
}

export interface CameraTransformValues {
    readonly position: readonly [number, number, number];
    readonly target: readonly [number, number, number];
}

export interface AquanovaControlPanelOptions {
    readonly canvas: HTMLCanvasElement;
    readonly antiAliasing: {
        readonly msaa: ToggleOption;
        readonly smaa: ToggleOption;
        readonly taa: ToggleOption;
        readonly specularAA: ToggleOption;
        readonly ssaa: ToggleOption;
    };
    readonly effects: {
        readonly improvedElectricity: ToggleOption;
        readonly animateElectricity: ToggleOption;
        readonly electricityArcDensity: {
            readonly get: () => number;
            readonly set: (value: number) => void;
        };
        readonly electricityBloomDebug?: ToggleOption;
        readonly electricityBloomThreshold: {
            readonly get: () => number;
            readonly set: (value: number) => void;
        };
        readonly electricityBloomStrength: {
            readonly get: () => number;
            readonly set: (value: number) => void;
        };
        readonly electricityBloomRadius: {
            readonly get: () => number;
            readonly set: (value: number) => void;
        };
    };
    readonly weapon: {
        readonly model: {
            readonly get: () => number;
            readonly set: (index: number) => MaybePromise;
            readonly options: readonly string[];
        };
        readonly sway: ToggleOption;
        readonly debug?: {
            readonly positionGizmo: ToggleOption;
            readonly rotationGizmo: ToggleOption;
            readonly scaleGizmo: ToggleOption;
            readonly localGuideGizmo: ToggleOption;
            readonly localGuideYaw: {
                readonly get: () => number;
                readonly set: (degrees: number) => void;
            };
        };
    };
    readonly audio: {
        readonly sounds: ToggleOption;
        readonly volume: {
            readonly get: () => number;
            readonly set: (value: number) => void;
        };
    };
    readonly environment: {
        readonly localCubemapBlending: ToggleOption;
        readonly envIntensity: {
            readonly get: () => number;
            readonly set: (value: number) => void;
        };
        readonly exposure: {
            readonly get: () => number;
            readonly set: (value: number) => void;
        };
        readonly toneMapping: {
            readonly get: () => number;
            readonly set: (index: number) => MaybePromise;
            readonly options: readonly string[];
        };
    };
    readonly debug: {
        readonly cameraPosition: VectorEditOption;
        readonly cameraTarget: VectorEditOption;
        readonly toggles: readonly ToggleOption[];
        readonly actions: readonly ActionOption[];
    };
    readonly onVisibilityChange?: (visible: boolean) => void;
}

export interface AquanovaControlPanel {
    toggle(): void;
    refresh(): void;
    isVisible(): boolean;
    updateWeaponTransform(values: WeaponTransformValues): void;
    updateCameraTransform(values: CameraTransformValues): void;
}

export function createAquanovaControlPanel(options: AquanovaControlPanelOptions): AquanovaControlPanel {
    const panel = document.getElementById("aquanovaControls");
    if (!(panel instanceof HTMLElement)) {
        throw new Error("Missing #aquanovaControls");
    }
    const refreshers: Array<() => void> = [];
    let visible = true;

    const addSection = (title: string): HTMLFieldSetElement => {
        const section = document.createElement("fieldset");
        section.className = "control-section";
        const legend = document.createElement("legend");
        legend.textContent = title;
        section.append(legend);
        panel.append(section);
        return section;
    };

    const runAndRefresh = (run: () => MaybePromise, control: HTMLInputElement | HTMLSelectElement | HTMLButtonElement): void => {
        control.disabled = true;
        Promise.resolve(run())
            .catch((err: unknown) => {
                console.warn("[aquanova] control update failed", err);
            })
            .finally(() => {
                control.disabled = false;
                refresh();
            });
    };

    const addToggle = (section: HTMLElement, option: ToggleOption): void => {
        const label = document.createElement("label");
        label.className = "control-toggle";
        const input = document.createElement("input");
        input.type = "checkbox";
        const text = document.createElement("span");
        text.textContent = option.label;
        label.append(input, text);
        section.append(label);
        input.addEventListener("change", () => runAndRefresh(() => option.set(input.checked), input));
        refreshers.push(() => {
            input.checked = option.get();
        });
    };

    const addSelect = (section: HTMLElement, labelText: string, labels: readonly string[], get: () => number, set: (index: number) => MaybePromise): void => {
        const label = document.createElement("label");
        label.className = "control-row";
        const text = document.createElement("span");
        text.textContent = labelText;
        const select = document.createElement("select");
        for (const optionLabel of labels) {
            const option = document.createElement("option");
            option.textContent = optionLabel;
            select.append(option);
        }
        label.append(text, select);
        section.append(label);
        select.addEventListener("change", () => runAndRefresh(() => set(select.selectedIndex), select));
        refreshers.push(() => {
            select.selectedIndex = get();
        });
    };

    const addSlider = (section: HTMLElement, labelText: string, min: number, max: number, step: number, get: () => number, set: (value: number) => void): void => {
        const label = document.createElement("label");
        label.className = "control-slider";
        const header = document.createElement("span");
        header.className = "control-slider-label";
        const text = document.createElement("span");
        text.textContent = labelText;
        const value = document.createElement("output");
        header.append(text, value);
        const input = document.createElement("input");
        input.type = "range";
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        label.append(header, input);
        section.append(label);
        input.addEventListener("input", () => {
            set(Number(input.value));
            refresh();
        });
        refreshers.push(() => {
            const current = get();
            input.value = String(current);
            value.textContent = current.toFixed(2);
        });
    };

    const addAction = (section: HTMLElement, option: ActionOption): void => {
        const row = document.createElement("div");
        row.className = "control-action";
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = option.label;
        const status = document.createElement("span");
        row.append(button, status);
        section.append(row);
        button.addEventListener("click", () => runAndRefresh(option.run, button));
        refreshers.push(() => {
            status.textContent = option.status?.() ?? "";
        });
    };

    const addVectorReadout = (section: HTMLElement, labelText: string): HTMLOutputElement => {
        const row = document.createElement("div");
        row.className = "control-vector";
        const label = document.createElement("span");
        label.textContent = labelText;
        const output = document.createElement("output");
        row.append(label, output);
        section.append(row);
        return output;
    };

    const addVectorInput = (section: HTMLElement, labelText: string, option: VectorEditOption): HTMLInputElement => {
        const label = document.createElement("label");
        label.className = "control-vector";
        const text = document.createElement("span");
        text.textContent = labelText;
        const input = document.createElement("input");
        input.type = "text";
        input.inputMode = "decimal";
        input.spellcheck = false;
        input.autocomplete = "off";
        label.append(text, input);
        section.append(label);
        input.addEventListener("change", () => {
            const parts = input.value
                .trim()
                .split(/[\s,]+/)
                .filter(Boolean)
                .map(Number);
            if (parts.length !== 3 || parts.some((value) => !Number.isFinite(value))) {
                input.setCustomValidity("Enter three finite numbers separated by commas or spaces.");
                input.reportValidity();
                return;
            }
            input.setCustomValidity("");
            option.set([parts[0]!, parts[1]!, parts[2]!]);
        });
        return input;
    };

    const aa = addSection("Anti-aliasing");
    addToggle(aa, options.antiAliasing.msaa);
    addToggle(aa, options.antiAliasing.smaa);
    addToggle(aa, options.antiAliasing.taa);
    addToggle(aa, options.antiAliasing.specularAA);
    addToggle(aa, options.antiAliasing.ssaa);

    const effects = addSection("Effects");
    addToggle(effects, options.effects.improvedElectricity);
    addToggle(effects, options.effects.animateElectricity);
    addSlider(effects, "Electrical arc density", 0.25, 3, 0.05, options.effects.electricityArcDensity.get, options.effects.electricityArcDensity.set);
    addSlider(effects, "Bloom threshold", 0, 2, 0.01, options.effects.electricityBloomThreshold.get, options.effects.electricityBloomThreshold.set);
    addSlider(effects, "Bloom strength", 0, 10, 0.05, options.effects.electricityBloomStrength.get, options.effects.electricityBloomStrength.set);
    addSlider(effects, "Bloom blur radius", 1, 32, 1, options.effects.electricityBloomRadius.get, options.effects.electricityBloomRadius.set);
    if (LAB_DEBUG && options.effects.electricityBloomDebug) {
        addToggle(effects, options.effects.electricityBloomDebug);
    }

    const weapon = addSection("Weapon");
    addSelect(weapon, "Model detail", options.weapon.model.options, options.weapon.model.get, options.weapon.model.set);
    addToggle(weapon, options.weapon.sway);
    const weaponDebug = LAB_DEBUG ? options.weapon.debug : undefined;
    let weaponPosition: HTMLOutputElement | null = null;
    let weaponRotation: HTMLOutputElement | null = null;
    let weaponScale: HTMLOutputElement | null = null;
    let localGuidePosition: HTMLOutputElement | null = null;
    let localGuideRotation: HTMLOutputElement | null = null;
    if (weaponDebug) {
        addToggle(weapon, weaponDebug.positionGizmo);
        addToggle(weapon, weaponDebug.rotationGizmo);
        addToggle(weapon, weaponDebug.scaleGizmo);
        addToggle(weapon, weaponDebug.localGuideGizmo);
        addSlider(weapon, "Aim Y rotation", -180, 180, 0.1, weaponDebug.localGuideYaw.get, weaponDebug.localGuideYaw.set);
        weaponPosition = addVectorReadout(weapon, "Position");
        weaponRotation = addVectorReadout(weapon, "Rotation (deg)");
        weaponScale = addVectorReadout(weapon, "Scale");
        localGuidePosition = addVectorReadout(weapon, "Aim origin position");
        localGuideRotation = addVectorReadout(weapon, "Aim rotation (deg)");
    }

    const audio = addSection("Audio");
    addToggle(audio, options.audio.sounds);
    addSlider(audio, "Volume", 0, 1, 0.05, options.audio.volume.get, options.audio.volume.set);

    const environment = addSection("Environment");
    addToggle(environment, options.environment.localCubemapBlending);
    addSlider(environment, "Env intensity", 0, 4, 0.05, options.environment.envIntensity.get, options.environment.envIntensity.set);
    addSlider(environment, "Exposure", 0, 4, 0.05, options.environment.exposure.get, options.environment.exposure.set);
    addSelect(environment, "Tone mapping", options.environment.toneMapping.options, options.environment.toneMapping.get, options.environment.toneMapping.set);

    const debug = addSection("Debug");
    const cameraPosition = addVectorInput(debug, "Camera position", options.debug.cameraPosition);
    const cameraTarget = addVectorInput(debug, "Camera target", options.debug.cameraTarget);
    for (const toggle of options.debug.toggles) addToggle(debug, toggle);
    for (const action of options.debug.actions) addAction(debug, action);

    const refresh = (): void => {
        for (const update of refreshers) update();
    };
    const setVisible = (on: boolean): void => {
        visible = on;
        panel.hidden = !on;
        document.body.classList.toggle("aquanova-controls-open", on);
        if (on && document.pointerLockElement === options.canvas) {
            document.exitPointerLock();
        }
        options.onVisibilityChange?.(on);
        refresh();
    };

    setVisible(true);
    const setVector = (output: HTMLOutputElement, values: readonly [number, number, number], digits: number): void => {
        const text = values.map((value) => value.toFixed(digits)).join(", ");
        if (output.value !== text) output.value = text;
    };
    const setVectorInput = (input: HTMLInputElement, values: readonly [number, number, number]): void => {
        if (document.activeElement === input) return;
        const text = values.map((value) => value.toFixed(4)).join(", ");
        if (input.value !== text) input.value = text;
        input.setCustomValidity("");
    };
    return {
        toggle: () => setVisible(!visible),
        refresh,
        isVisible: () => visible,
        updateWeaponTransform: (values) => {
            if (weaponPosition) setVector(weaponPosition, values.position, 4);
            if (weaponRotation) setVector(weaponRotation, values.rotationDegrees, 2);
            if (weaponScale) setVector(weaponScale, values.scale, 4);
            if (localGuidePosition) setVector(localGuidePosition, values.localGuidePosition, 4);
            if (localGuideRotation) setVector(localGuideRotation, values.localGuideRotationDegrees, 2);
        },
        updateCameraTransform: (values) => {
            setVectorInput(cameraPosition, values.position);
            setVectorInput(cameraTarget, values.target);
        },
    };
}
