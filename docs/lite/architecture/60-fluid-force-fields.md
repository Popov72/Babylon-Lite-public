# Module 60: Authored fluid force fields

## Purpose and first-version scope

Provide editable Point and straight-line Guide forces for production PBF, FLIP, MLS-MPM and
PB-MPM through the Fluid panel. Reference is explicitly unsupported for this authoring feature;
its existing manual/custom-force path is unchanged. Surface/Volume mesh fields, arbitrary curves,
gravity scaling, Blender export and animation binding are outside this first version.

The model follows the FLIP Fluids force-field controls and documented equations:

- [Object settings](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Force-Field-Object-Settings)
- [Force magnitude and limits](https://github.com/rlguy/Blender-FLIP-Fluids/blob/master/src/engine/forcefield.cpp)
- [Point field](https://github.com/rlguy/Blender-FLIP-Fluids/blob/master/src/engine/forcefieldpoint.cpp)
- [Curve guide](https://github.com/rlguy/Blender-FLIP-Fluids/blob/master/src/engine/forcefieldcurve.cpp)

This is an independent analytic implementation, evaluated at particle positions through the
existing force hook. FLIP Fluids rasterizes a separate MAC force grid; identical force equations
do not imply identical grid-sampling results.

## Data and public API

`FluidForceFieldDefinition` is a discriminated union with common `id`, `name`, `enabled`,
`strength`, `falloffPower`, `useMinDistance`, `minDistance`, `useMaxDistance`, `maxDistance`,
and `maxForceLimitFactor`. Point fields contain `type: "point"` and `position`. Guides contain
`type: "guide"`, `start`, `end`, `flowStrength`, `spinStrength`, and `endCaps`.
All positions and distances are in simulation world units, not normalized grid coordinates.

Pure helpers create defaults, validate/clone definition arrays and evaluate acceleration for
unit coverage. Validation rejects unknown types, duplicate/empty IDs, nonfinite or unrepresentable
float32 values, negative falloff/distance/limit values, inconsistent enabled distance limits,
degenerate guides, and unsafe aggregate force magnitudes. Zero strength/limit is a valid no-op.

```typescript
type FluidForceFieldKind = "point" | "guide";
type FluidForceFieldVector = readonly [number, number, number];
createDefaultFluidForceField(type: "point", id: string, position?: FluidForceFieldVector): FluidPointForceField;
createDefaultFluidForceField(type: "guide", id: string, position?: FluidForceFieldVector): FluidGuideForceField;
createDefaultFluidForceField(type: FluidForceFieldKind, id: string, position?: FluidForceFieldVector): FluidForceFieldDefinition;
validateFluidForceFields(value: unknown): FluidForceFieldDefinition[];
evaluateFluidForceFields(fields: readonly FluidForceFieldDefinition[], point: FluidForceFieldVector): [number, number, number];
fluidForceFieldCapacity(engine: EngineContext): number;
createFluidConfiguredForceField(engine: EngineContext, fields: readonly FluidForceFieldDefinition[],
    baseForce?: FluidForceField | null): FluidForceField;
updateFluidConfiguredForceField(force: FluidForceField, fields: readonly FluidForceFieldDefinition[]): void;
```

Defaults are enabled, strength `-9.81`, power `1`, minimum distance enabled at `0.5`,
maximum distance enabled at `5`, and limit factor `3`. The default center is `[0, 1, 0]`.
Guides extend one unit either side of the center on Y, with flow `5`, spin `0`, and end caps enabled.
Power is restricted to `[0, 16]`. Disabled definitions are validated too, but excluded from the
aggregate force-limit check and packed GPU prefix.

The returned opaque force is managed with the existing `setFluidSimulationForceField` and
`disposeFluidForceField` APIs. Its uniform capacity is bounded by the device uniform-buffer
limit, not a hidden per-frame allocation. Live edits upload parameters without recompiling shaders.
An optional base force is applied first, retaining the existing temporary manual-force override
of demo-local forces; authored fields add their acceleration afterward.

## Force equations

For a Point field, `delta = particle - position`. For a Guide, compute the nearest point on the
finite segment `start -> end`. End caps clamp the segment parameter; disabling them makes positions
beyond either endpoint unaffected. The segment must remain nondegenerate at float32 precision.

Let `d = length(delta)` and `r = max(d, enabledMinDistance)`. Points within `1e-6` of the force
origin/guide have no contribution, matching the native singularity guard. An enabled maximum
distance cuts off the field outside its range. The signed radial acceleration is
`strength / r^falloffPower` in direction `delta / d`, capped to
`abs(strength) * maxForceLimitFactor`. Negative strength attracts and positive strength repels.
Power zero gives constant magnitude.

Guide flow uses the normalized start-to-end tangent. Guide spin uses
`cross(tangent, radialDirection)`, with positive spin following the right-hand rule.
Radial, flow and spin contributions receive the same distance falloff and independent magnitude
limits based on their respective strengths, then are summed. Stable bounded evaluation avoids
overflow near the singularity. The force kernel returns acceleration multiplied by its supplied
physical substep dt exactly once; PB-MPM's existing backend converts that velocity delta into
displacement. No new keyboard gesture or simulation-time convention is introduced.

## Runtime composition and ownership

`ForceFieldSpec.additional` represents ordered additional force passes. Production backends use
one shared lazy pass-cache helper to flatten/validate the specification, compile each distinct
complete shader source and bind its own uniform buffer. No shader-text rewriting, parameter-layout
parsing, GPU readback or uniform-buffer copying is used for composition. The hot path iterates an
already-built command list, retaining each backend's existing active-particle dispatch and substep
ordering. With no force, no force pipeline is built or dispatched.

The configured-force enabler owns its analytic uniform and retains any base force through the
existing ownership mechanism. It never destroys a caller-owned base. An optional implementation
validator on the internal force binding rejects implementation overrides before installation.
The general facade only passes method/backend scalar identities through that seam; it does not
contain authored-force equations or Reference-specific logic.

The uniform is `16 + capacity * 64` bytes, with capacity
`floor((min(maxUniformBufferBindingSize, maxBufferSize) - 16) / 64)`.
Its header is a `vec4u` whose X is the enabled-field count. Each 64-byte field contains, in order:
`originStrength: vec4f` (point/guide-start XYZ, radial strength),
`axisLength: vec4f` (guide unit tangent, segment length),
`strengths: vec4f` (flow, spin, power, limit factor),
`range: vec2f` (min/max distances), `flags: u32` (min=1, max=2, end-caps=4),
and `kind: u32` (point=0, guide=1). Only the used prefix is uploaded on edits.
Disposing a wrapper releases base ownership immediately; its own GPU buffer retires behind
the existing GPU fence. Detach wrappers from simulations before disposal.

## UI and persistence

The reusable force-field editor is installed in a collapsible **Force fields** panel section.
It supports adding/removing/selecting named fields, enabled toggles, point position, guide endpoints,
radial/flow/spin strengths, falloff, distance-limit toggles and values, end caps, and the force-limit
factor. Invalid edits retain the last valid definition and display an error rather than silently
clamping or ignoring input. No shortcuts are added.

```typescript
interface FluidForceFieldEditorOptions {
    readonly fields?: readonly FluidForceFieldDefinition[];
    readonly capacity: number;
    readonly onChange: (fields: FluidForceFieldDefinition[]) => void;
    readonly getDefaultPosition?: () => FluidForceFieldVector;
}
interface FluidForceFieldEditor { readonly root: HTMLElement; /* internal state omitted */ }
createFluidForceFieldEditor(options: FluidForceFieldEditorOptions): FluidForceFieldEditor;
getFluidForceFieldEditorFields(editor: FluidForceFieldEditor): FluidForceFieldDefinition[];
setFluidForceFieldEditorFields(editor: FluidForceFieldEditor, fields: readonly FluidForceFieldDefinition[]): void;
setFluidForceFieldEditorEnabled(editor: FluidForceFieldEditor, enabled: boolean): void;
```

Input `change` events validate and invoke `onChange` before committing editor state. Getters/setters
clone definitions. Programmatic setters do not emit `onChange`. The host positions new fields at
the current simulation-domain center. There are no viewport gizmos in this version.

`PairState.forceFields` and `FluidExportJson.forceFields` store definitions. Legacy presets without
this field restore an empty list; unknown additional metadata is retained. The authoring state
round-trips through export/import, reset, quality selection and method-independent scene changes.
World-space field locations do not follow pending grid edits implicitly.

The Fluid host caches configured bindings by the current base-force identity and updates them only
when authored fields change. The enabled-state flag is computed on edits, not scanned every frame.
Empty/all-disabled authoring uses the original base force directly.
Reconfiguration retains the selected force binding; old owned bindings are detached before disposal.
Reference selection rejects enabled authored forces with an explicit message, and its editor is
disabled. Disabled definitions can be retained without silently executing unsupported fields.

## Validation

Cover attraction/repulsion, constant/inverse-distance/inverse-square falloff, distance boundaries,
force caps, guide flow/spin orientation, end-cap behavior, singularities and invalid inputs.
Verify ordered composition, live uniform updates, no disabled dispatch, ownership/disposal,
Reference rejection and preset round-trips. Focused GPU probes must exercise all four production
methods and preserve the manual push path. UI checks cover editing and reset/switch/import behavior.

## File manifest

- `forces/force-field-config.ts`: public definitions, defaults, validation and CPU reference math.
- `forces/configured-force-field.ts`: GPU packing, analytic WGSL, capacity, composition and ownership.
- `core/force-field-passes.ts`: lazy ordered pass compilation and binding caches, shared by production solvers.
- `controls/force-field-editor.ts`: standalone panel editor; the Fluid demo owns its runtime binding.
- `authoring/authoring-state.ts`, `preset-io.ts`, `blender-fluid-json.ts`: preset state and import boundaries.
- `tests/lite/unit/fluid/fluid-force-fields.test.ts`, `fluid-force-field-presets.test.ts`,
  `fluid-facade-transaction.test.ts`: equations, validation, caching, presets and transactional ownership.
- `tests/lite/parity/force-field-probe.ts`, `regressions/fluid-force-fields.spec.ts`:
  numeric GPU and authoring regressions, without screenshot comparisons.
