// Pure import/edit/export state for hosts that author fluid presets over multiple UI operations.

import type { PairState } from "./authoring-state.js";
import { exportJsonFromPairState, mergeFluidPresetData, presetFromExportJson, type FluidExportJson } from "./preset-io.js";

type DeepPartial<T> = T extends readonly unknown[] ? T : T extends object ? { [K in keyof T]?: DeepPartial<T[K]> } : T;

/** Sections whose known values belong to an embedding application rather than PairState. */
export type FluidPresetApplicationSections = DeepPartial<Pick<FluidExportJson, "impulse" | "grid" | "scene" | "source">>;

/** Immutable state retained between preset import, UI edits, and a later export. */
export interface FluidPresetSession {
    readonly state: PairState;
    readonly application: FluidPresetApplicationSections;
}

export interface FluidPresetSessionEdit {
    /** Known authoring-state changes. Omitted and undefined fields retain their imported values. */
    state?: DeepPartial<PairState>;
    /** Application-owned section changes, recursively merged so unmodeled children survive. */
    application?: FluidPresetApplicationSections;
}

export interface FluidPresetSessionExport extends FluidPresetSessionEdit {
    demo: string;
    method: string;
}

function applicationSectionsFrom(json: FluidExportJson): FluidPresetApplicationSections {
    return {
        ...(json.impulse !== undefined ? { impulse: structuredClone(json.impulse) } : {}),
        ...(json.grid !== undefined ? { grid: structuredClone(json.grid) } : {}),
        ...(json.scene !== undefined ? { scene: structuredClone(json.scene) } : {}),
        ...(json.source !== undefined ? { source: structuredClone(json.source) } : {}),
    };
}

/** Import a preset over host defaults while retaining all mapped and forward-compatible state. */
export function importFluidPresetSession(json: FluidExportJson, defaults: PairState): FluidPresetSession {
    return {
        state: mergeFluidPresetData(defaults, presetFromExportJson(json)) as PairState,
        application: applicationSectionsFrom(json),
    };
}

/** Return a new session with edits applied; the input session and its nested values are unchanged. */
export function editFluidPresetSession(session: FluidPresetSession, edit: FluidPresetSessionEdit): FluidPresetSession {
    return {
        state: mergeFluidPresetData(session.state, edit.state ?? {}) as PairState,
        application: mergeFluidPresetData(session.application, edit.application ?? {}),
    };
}

/** Export the retained session to another host, optionally applying its latest state in one step. */
export function exportFluidPresetSession(session: FluidPresetSession, options: FluidPresetSessionExport): FluidExportJson {
    const edited = editFluidPresetSession(session, options);
    const mapped = exportJsonFromPairState(options.demo, options.method, edited.state);
    const exported = mergeFluidPresetData(mapped, edited.application) as FluidExportJson;
    if (exported.scene?.animatedCollisions?.length) {
        exported.formatVersion = Math.max(exported.formatVersion ?? 0, 15);
    }
    return exported;
}
