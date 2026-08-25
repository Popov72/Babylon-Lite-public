# Aquanova Gameplay Behaviors

> Demo path: `lab/lite/src/demos/aquanova/behaviors/`

> **Status: Implemented.**

Aquanova gameplay is composed from manifest-assigned behaviors instead of being
hard-wired to mesh-name checks in `main.ts`. A manifest entity may assign
several behaviors to the same mesh, and every assignment is resolved and
instantiated independently.

## Local environment probe modes

Aquanova loads every authored reflection probe once and assigns one shared
`rgba16float` cube-texture-array probe set to PBR materials. A dense 2 m world-space voxel grid
stores every probe whose oriented outer influence box intersects each cell.
Fragments address that grid from `worldPos`, then calculate exact oriented-box
weights only for the probes in their cell. Empty cells use their deterministic
nearest probe, and any cell exceeding the initialization-time capacity fails
explicitly instead of silently dropping a probe.

When blending is disabled, probe selection is static and mesh-owned rather than
player-owned. Each mesh's setup-time world AABB is tested against the authored
oriented projection boxes. The containing box wins; otherwise the nearest
intersecting box wins, with the closest probe as a deterministic fallback.
Materials shared by meshes assigned to different probes are cloned per
source/probe pair and use the lightweight `localEnvironment` single-cubemap
path. Runtime mode switches clone the mesh's current material state so later
plugins and gameplay material changes are preserved.

The `V` probe debug mode is active only while blending is enabled. It shows a
colored sphere at every probe in the camera's current voxel and enables the
core probe-set color diagnostic. The shader uses the same per-fragment weights
as production rendering but blends each probe's assigned color instead of
sampling/calculating the final PBR color. Its panel lists only that voxel set
with each probe's centre and full inner/outer box sizes; POI-derived diagnostic
weights are intentionally omitted because final influence is fragment-dependent.

## Typed behavior model

`behavior-definitions.json` is the catalog of executable base behaviors and
their typed parameters. Entities may attach a base behavior directly. The
manifest's optional `behaviorPresets` object defines named parameter profiles
derived from one base behavior:

```json
{
    "behaviorPresets": {
        "stdLiquefaction": {
            "base": "liquefaction",
            "fluidSim": ["liquid-slow"]
        }
    }
}
```

Preset names must be distinct from base behavior names, and presets cannot
derive from other presets. Preset values are merged with entity overrides, then
the resolved base behavior is instantiated. The current base behaviors are:

| Behavior               | Responsibility                                          |
| ---------------------- | ------------------------------------------------------- |
| `dynamic`              | Makes a prop physically simulated and player-pushable   |
| `liquefaction`         | Liquefies with an optional authored fluid setting       |
| `player`               | Owns first-person controls and disables its marker      |
| `weaponLiquefactor`    | Gates and drives the Liquefactor weapon                 |
| `weaponAntiGravityGun` | Grabs and throws dynamic rigid bodies                   |
| `pickEntity`           | Collects an intersected entity and emits an event       |
| `enableEntity`         | Enables its owner after a configured source event       |
| `disableEntity`        | Disables its owner after a configured source event      |
| `sound`                | Plays or stops an authored sound                        |
| `fluidSimulation`      | Runs an event-controlled authored fluid simulation      |
| `setCollisionShape`    | Replaces an entity collider from its visible geometry   |
| `trigger`              | Raises owner events when its collider is entered/exited |
| `playAnimation`        | Starts one animation clip from the ship glTF            |

The ship manifest uses `stdLiquefaction` and `explosiveLiquefaction` presets of
the `liquefaction` base. A direct `liquefaction` assignment uses the behavior's
defaults. At runtime all three resolve to `LiquefactionBehavior`; preset values
and per-entity overrides remain data rather than separate constructor aliases.

Each behavior receives:

- the owning entity or door name;
- every mesh primitive owned by that entity;
- its resolved manifest assignment;
- one Aquanova-specific `AquanovaGameContext` containing the camera, canvas,
  character controller, picker access, event manager, and gameplay services.

## Events

The generic behavior system provides `TypedEventBus` and `EventManager` without
declaring Aquanova events. `SystemEventMap` separately declares frame and
physics events, while `AquanovaEventMap` declares gameplay events. Their
intersection is the typed event map used by `AquanovaEventManager`.

`AquanovaEventManager` owns the engine integration. It brackets the
scene's registered frame callbacks with `frameStart` / `frameEnd` and translates
Havok's after-step callback into `physicsStep`. Gameplay code subscribes through
the manager instead of registering engine callbacks directly.

The system currently emits:

- `frameStart` before Aquanova's per-frame render work;
- `physicsStep` after Havok has stepped and synchronized nodes;
- `frameEnd` after Aquanova's registered per-frame update callbacks.

`entityEvent` carries an entity or door name and event name. It is the generic
manifest-driven link between otherwise independent behaviors. Subscriptions
treat that owner name as their `source`. `pickEntity` may define
`raiseEvent: { target, event }`; after collection it emits that payload exactly
once. `target` is optional and defaults to the entity carrying `pickEntity`.
Consumers ignore events addressed to other entities.

Portal visibility also publishes chunk transitions through `entityEvent`.
A chunk raises `visible` when it enters the portal traversal's displayed chunk
set and `notVisible` when it leaves. Exterior-only shells seen through sky
portals do not count as displayed chunks. Chunk event sources keep their raw
chunk ID in the manifest and appear as `<ID> chunk` in editor source pickers.

`setCollisionShape` builds a Havok box from each entity mesh group's world-space
AABB by default. Explicit `{ "type": "mesh" }` instead builds a static
triangle-mesh shape from every primitive owned by the entity. Mesh collision
shapes are rejected on physically dynamic entities.

`fluidSimulation` loads the extensionless `fluidSim` setting name from
`public/aquanova/fluidSim/`. Its axis-aligned grid keeps the authored size but
uses the behavior owner's world position instead of the position stored in the
JSON; emitter and sink positions remain grid-local and are translated with that
new center. The simulation is absent and does not render until the owner
receives an event mapped to `enableSimulation`. `eventActions` maps an external
entity or door's event to one of:

- `enableSimulation`, `disableSimulation`, `pauseSimulation`,
  `unpauseSimulation`, or `shutdownSimulation`;
- `enablePlayerCollision` or `disablePlayerCollision`;
- `enableEmitter` or `disableEmitter`, with a named `emitter`;
- `enableSink` or `disableSink`, with a named `sink`.

For example:

```json
{
    "name": "fluidSimulation",
    "fluidSim": "capsule-filling",
    "eventActions": [
        {
            "source": ["valveA", "valveB"],
            "event": "opened",
            "action": "enableEmitter",
            "emitter": "Main inlet"
        },
        {
            "source": "controlPanel",
            "event": "activated",
            "action": "enableSimulation"
        }
    ]
}
```

Behavior-owned simulations prepare only CPU-side descriptors during demo
loading. A one-particle, small-grid throwaway solver for each unique authored
setting exercises the exact method, parameter, flow, and collision-shader
pipeline variants, relying on the browser's WGSL-source cache; its temporary
buffers are disposed before the engine starts. The real grid, particle, and
collision GPU buffers are allocated only on the simulation's first
`enableSimulation` or `shutdownSimulation`, so unused behavior simulations
consume no persistent fluid GPU memory while first-use shader compilation stays
off the triggering frame.

Flow-object actions received before the first `enableSimulation` update the
prepared CPU descriptor's flow configuration without allocating a solver.
Later actions update the live configuration without resetting its particles.
Initial emitters still seed only when the solver is reset; live replenishment
uses inflow emitters.

`pauseSimulation` and `unpauseSimulation` affect only a simulation that has
already been started by `enableSimulation`. Before that first
start both actions are no-ops. After startup they pause and resume stepping
respectively; like enable/disable and flow-object actions, both are ignored
after shutdown begins.

`enablePlayerCollision` and `disablePlayerCollision` control whether the
behavior-owned simulation's fluid collides with the live player capsule. The
setting is retained before the solver is allocated and updates the reserved
player collision slot immediately when the simulation is already running. It
remains mutable while the simulation is in its active shutdown phase, allowing
an event to enable collision and start shutdown together. It does not affect
liquefaction simulations.

The fixed `emissionComplete` event is raised from the behavior owner exactly
once when the aggregate active particle count reaches its emission target. For
FLIP flows containing only Initial emitters, that target is the exact
reset-time count computed by the solver from emitter volume, grid bounds,
resolution-derived cell size, markers per cell, clipping, and configured
capacity—the same inputs shown by the fluid UI. When any Inflow emitter exists,
the target is the solver capacity because that emitter may be enabled later and
continue filling dormant slots. Enabled sinks can prevent that target from ever
being reached.

`AquanovaFluidRuntime` is the shared behavior-facing service for authored fluid
simulations and liquefaction water. In addition to the behavior simulation
lifecycle, it owns a delayed GPU query over the combined visible particle
stream. `countParticlesInAabb()` requests a world-space AABB sample and returns
the latest completed count. A compute reduction tests the actual aggregated
position and alpha buffers, copies only one four-byte counter to a three-buffer
readback ring, and maps it on a later frame. Sampling is limited to once every
six rendered frames; callers continue receiving the last completed count
between samples. Invisible dissolve-front particles,
fully faded water, inactive capacity, and particles omitted by aggregate
capacity are therefore not counted. Temporal interpretation of delayed counts
belongs to the caller.

`shutdownSimulation` is irreversible. It resumes a paused or not-yet-enabled
simulation, runs it at full opacity for `shutdownDuration` simulated seconds
(default `10`), fades it over `shutdownAlphaDecay` additional simulated seconds
(default `2`), then disposes its GPU and collision resources. Every later event
action is ignored once shutdown begins.

`sound` requires a non-empty `cues` array. Each cue contains an extensionless
MP3 `sound` name, a `play` or `stop` `action`, optional `events`
subscriptions, and optional `delay` and `fade` durations. A cue without
subscriptions runs once at behavior startup. `delay` postpones that cue's
action, while `fade` is a fade-in for `play` and a fade-out before `stop`; both
are finite non-negative seconds and default to zero. Distinct MP3s are
preloaded once per behavior, matching cues run in declaration order, and all
subscriptions and delayed actions are cancelled together when the behavior is
disposed.

Behavior-owned simulations and liquefaction blobs share Aquanova's single
particle-surface renderer. The most recently enabled fluid setting therefore
controls the shared water look when several simulations overlap; particle
positions, stepping, pause state, collision fields, opacity, and disposal remain
independent per simulation.

`trigger` accepts
`{ "onIntersection": { "enterEvent": "activated",
"exitEvent": "deactivated", "playerOnly": true } }`. `enterEvent` and
`exitEvent` are independently optional, and `playerOnly` defaults to `false`.
The collider becomes a Havok trigger volume, so overlaps produce no physical
response. The enter event is raised when the first qualifying body enters and
the exit event when the final qualifying body leaves. Both events are raised by
the entity carrying the behavior. Entity `disable` and `enable` events disable
and re-enable overlap reporting respectively.

Entity action behaviors (`enableEntity`, `disableEntity`, `removeEntity`,
`hideEntity`, `showEntity`, `enableCollision`, and `disableCollision`) always
act on their owner. Their optional `events` array identifies event sources:

```json
{
    "name": "showEntity",
    "events": [
        { "name": "activated", "source": ["trapTrigger", "backupTrigger"] },
        { "name": "opened", "source": "Door_D06" }
    ]
}
```

The behavior runs when any entry matches. `source` is an entity or door id, or
an array of ids with equivalent OR semantics; sources do not need to own
meshes. With no `events`, the action runs immediately during behavior startup.
Unsupported behavior keys are rejected at load time so stale manifest syntax
cannot silently change behavior.

`fluidSim` entries are extensionless setting names such as `liquid-slow`.
The runtime appends `.json` only when fetching the corresponding file from
`/aquanova/fluidSim/`.

The player emits `hitWithWeapon` after a center-screen pick. The player does not
know whether the picked mesh is liquefiable. Each liquefiable behavior listens
for that event, accepts hits addressed to its own mesh while it remains an
active target, and invokes the shared liquefaction service.

While the canvas owns pointer lock, each mouse-wheel notch emits a typed
`weaponCycleRequested` event and prevents page scrolling. A shared weapon
inventory owns slots, ownership, and the currently equipped slot. Slot 1 is the
Liquefactor, slot 2 is the anti-gravity gun, and the wheel cycles through every
owned slot plus the hidden state in the requested direction. Selecting an
unowned numbered slot holsters the current weapon; selecting the equipped slot
again also holsters it.

The Havok character controller uses a 45-degree maximum walkable slope
(`maxSlopeCosine = cos(45°)`). Shallower surfaces remain fully supported instead
of entering the sliding state. While supported, horizontal intent is projected
onto the support plane and the contact force is applied along the surface normal
rather than world-down. An idle player therefore has no downhill tangent to
slide along, while deliberate uphill/downhill movement remains responsive.
Steeper surfaces may slide.

The `player` definition may set `characterStrength`, the maximum force applied
to contacted dynamic bodies. It defaults to `10000`; setting it to `0`
preserves collision while disabling player pushes.

The entity carrying `player` is a placement marker rather than scenery. Its
mesh is hidden and non-pickable during scene setup, and the behavior disables
collision for the complete owning entity when it starts. The marker is authored
as the standing `1.8 m` capsule volume, with its bottom on the spawn floor; its
world-space AABB centre is therefore the character controller's initial centre.

Door behaviors enable or disable their associated visibility portal. Recursive
portal traversal treats the rectangle as two-sided: its corners define the
clipping plane, and the current camera position selects the source side. Chunk
AABB centres do not orient that plane because connected chunk bounds may overlap
or contain one another.

`C` toggles crouching. Over `0.2 s`, the character controller keeps the
capsule's foot position fixed while smoothly changing its total height from
`1.8 m` to `0.7 m`. Its radius remains `0.3 m` in both positions. Camera height
and the 25% crouched movement-speed reduction
interpolate with the same progress. The transition is reversible and works on
the ground or during a jump. A jump request or held run key requests standing
first; expansion is accepted only while the taller capsule has overhead
clearance. Jump input is buffered through the standing transition. While a
forward jump is active, clearance rays across the standing and crouched capsule
profiles detect low apertures: if the standing profile is blocked and the
crouched profile is clear, crouch engages automatically. Once fully crouched,
the controller adds up to `0.2 m` of collision-resolved forward movement to carry
the player past the aperture edge. The airborne automatic transition preserves
the capsule centre rather than its foot position, tucking the feet upward instead
of dropping the player onto the lower sill. Entry progress is measured from the
controller's resolved displacement, so a blocked frame does not consume the
assist, and touching the sill does not cancel it. The clearance test evaluates
candidate centre offsets at `0`, `5`, and `10 cm` on either side; the closest
clear candidate is applied as a lateral correction, allowing a small jamb
overlap to be compensated without reducing the collision capsule.

## Runtime lifecycle

The generic `behavior-system/BehaviorManager` owns manifest assignment
resolution, dynamic constructor lookup, lifecycle ordering, and disposal. A
behavior named `disableCollision` resolves to the externally supplied export
`DisableCollisionBehavior`: uppercase the first character and append
`Behavior`. The generic manager has no imports or branches for Aquanova
implementations.

`AquanovaBehaviorManager` composes that generic runtime with mesh
classification, linked-entity closure, weapon inventory, system-event binding,
and gameplay queries. Its constructor catalog lives outside the generic
package.

Exactly one behavior instance is constructed per owner assignment. Owners may
be meshed entities or meshless doors. Geometry-dependent behaviors still reject
owners without meshes; event-only entity action behaviors support both. Every
non-door owner with configured behaviors but no runtime mesh is skipped with a
browser-console warning.
instance is constructed first, then every `init()` is invoked and the resulting
promises are awaited together with `Promise.all`, then every `start()` runs.
Behaviors that operate on geometry retain the complete entity mesh group and
handle the relevant primitive internally.

`pickEntity` is also one instance per manifest entity, but it owns every mesh
primitive under that entity. At each physics step it intersects the live player
capsule's world AABB with the entity's initial world AABB. Optional
`boundingBoxScale: [x, y, z]` scales that box's half-extents around its centre
before intersection testing and defaults to `[1, 1, 1]`. Until pickup, every
authored node represented by the manifest entity rotates around its local Y axis.
The default angular speed is one full revolution every 3 seconds. Optional
`speed` must be finite and positive and multiplies that angular speed, so the
revolution duration is `3 / speed` seconds. The first intersection hides all
owned primitives, optionally plays its preloaded MP3, optionally emits
`entityEvent`, and unregisters both the intersection check and rotation.
`sound` is an MP3 file name without extension under `/aquanova/sounds/` and
defaults to `pickItem`. Each instance's `init()` lazily creates the shared audio
engine through `SoundManager` and preloads that pickup's sound before any
behavior starts; repeated sound URLs are deduplicated.

`playAnimation` is one instance per manifest entity. Its optional `animation`
parameter selects an animation group targeting that entity, or one of its
exported child nodes, by its exact glTF animation name. This target filtering
keeps identically named clips on separate placed modules independent. When
`animation` is absent, it selects the entity's first animation group in file
order; if the entity has no animations, startup is a no-op. A configured name
that does not exist for the entity is an authoring error and stops behavior
startup with an explicit message instead of silently selecting another clip.
The optional `loop` parameter controls `AnimationGroup.loopAnimation` and
defaults to `true`.
Aquanova registers the complete ship asset container so its animation groups
are ticked by the scene, but stops every group before behavior startup; only
clips selected by `playAnimation` begin playback. Disposing the behavior stops
and rewinds its selected group. The editor excludes `playAnimation` entities
from reflection-probe captures because their exported transform is not static.

`weaponLiquefactor` starts unowned and hidden. The behavior listens for
`entityEvent` addressed to the manifest entity on which it was instantiated.
The `enable` event grants ownership and equips the weapon by animating it from a
lowered, muzzle-down pose to its horizontal firing pose. `Digit1` selects or
holsters it. Holstering reverses the same presentation animation. Trigger
events are ignored while the weapon is unowned, holstered, or still moving into
position.

`weaponAntiGravityGun` uses slot 2 and the same pickup, presentation, model
detail, sway, and crosshair lifecycle. It has no audio. Its optional
`maxGrabDistance` and `maxMass` parameters must be finite positive numbers and
default to `6 m` and `100 kg`. The first trigger press may grab only a mesh with
an explicit `dynamic` behavior whose configured mass is within that limit and
whose centre-screen hit distance is within range.

A grabbed body switches from dynamic to kinematic motion, has its velocities
cleared, and is attracted toward a point `2.5 m` along the camera aim ray while
retaining its orientation. It remains held after the first trigger is released.
The next trigger press starts charging a throw. Releasing within `150 ms` drops
the body with zero velocity; longer holds scale linearly to a mass-independent
launch speed of `15 m/s`, capped after `2 s`. Holstering, disposal, target
removal, or liquefaction drops the held body without throwing it.

The `dynamic` behavior may define `mass` in kilograms. It must be finite and
positive and defaults to `10 kg`. That value is applied to the Havok rigid body
after its authored collision shape is attached, so shape-derived inertia is
preserved while anti-gravity mass filtering and physical response use the same
authoritative mass.

The held weapon can apply a subtle procedural balancing motion made from
layered low-frequency translation and rotation. This motion is cosmetic: the
aim guide, laser, and crosshair remain stable. The `Weapon sway` control-panel
checkbox persists the preference and blends the motion in or out smoothly.
Walking scales both its amplitude and pace to 2× the idle motion; running with
Shift scales them to 4×. Holding the weapon trigger suppresses sway immediately
so the rendered muzzle remains aligned with the laser origin, then sway blends
back after the trigger is released.

Weapon position/rotation/scale gizmos, the aim-origin gizmo and yaw slider, and
their transform readouts are debug-build tooling. Release demo bundles omit
both these controls and the corresponding gizmo construction; model detail and
weapon sway remain available in every build.

`SoundManager` owns Aquanova's single lazily created audio engine, streaming
sound loading, URL deduplication, playback replenishment, active-loop tracking,
master volume, and disposal. Behaviors receive it through
`AquanovaGameContext`; neither behavior manager imports audio functions or
knows which behaviors play sounds. The player preloads `stepMetallic.mp3` and
plays it from grounded, collision-resolved movement, with a longer stride while
running; airborne, blocked, frozen, stationary, and noclip movement stay silent.

The `Sounds` control-panel checkbox persists a global gameplay-audio
preference. Disabling it immediately stops active Liquefactor loops and
suppresses subsequent pickup, firing, liquefaction, and splash sounds. The
persisted `Volume` slider controls the shared engine's master gain from `0` to
`1`; changing it also affects active weapon loops.

When liquefaction starts it raises `startLiquefaction` once for every unique
entity in the target's linked liquefaction group. If reversal restores the
group completely, it raises `cancelLiquefaction` for those same entities.
Liquefaction behavior is inherited by linked entities that do not define one,
so any member can be targeted directly. Linked relations form one symmetric,
transitive group for melting; an entity's own liquefaction behavior takes
precedence over an inherited one.

## Behavior boundary

The behavior layer owns gameplay policy and event flow. The existing GPU fluid
simulation, collision construction, render graph, and debug tooling remain
services in `main.ts`; they are invoked through the typed behavior context so
those subsystems can move behind narrower services independently.

Fluid collision primitives retain stable buffer slots with an `active` flag.
When a placement stops being collidable, Aquanova updates only that flag in
every running liquefaction and behavior-owned simulation; the collision shader
skips inactive slots. A prop's collision is deactivated at the same point its
Havok body is removed: immediately before the fluid simulation's first step.
The retired state also excludes it from behavior-owned simulations enabled
later.

Primitive inclusion is tested against the exact world-space AABB passed to the
fluid solver as its simulation grid. The liquefied mesh's sampled AABB is used
to position and automatically size that grid, but never directly filters the
collision shapes.

The third `B` collider-debug mode shows the deduplicated union of primitives
packed into all running simulations as opaque magenta wireframes. Solid
translucent volumes are intentionally avoided: the selected set commonly
contains complete floor, wall, and ceiling primitives whose faces cover much of
the camera view and would tint unrelated visible geometry. Deduplication avoids
misleading line-brightness accumulation where independently liquefied objects
use the same collision primitive. The player capsule remains packed into every
simulation but is omitted from this overlay because it is always present and
would obscure too much of the scene.

A hit and every mesh reached through its linked-entity closure produce one
shared fluid simulation. Their asynchronously sampled particles are concatenated
into one solver buffer, while per-mesh dissolve state, wriggle transform,
particle range, and colour buffer remain separate. The solver stays frozen until
every linked member has dissolved, then all members enter the fluid phase
together.

Fusion advances only while the left mouse button remains pressed. Releasing it
reverses the dissolve at twice the forward speed. Pressing again performs a new
center-screen pick but resumes the same shot, origin, sampled particles, and
linked group only when that pick hits one of the group's meshes; a miss or a hit
on a non-liquefiable object leaves the reversal running. Hitting a different
available liquefiable mesh detaches the old reversing group and starts a new
forward fusion for the newly picked mesh. The same direction state and target
validation apply while particle sampling is still running: a paused sample
keeps its original shot pose and can resume, while a completed reversal restores
target availability without creating a solver.

The `P` performance panel reports the current number of fluid simulations and
their total particle count alongside CPU and GPU timing.

The `L` runtime-light overlay shows the lights authored for the player's current
chunk using the exact records consumed by the renderer. A small light-coloured
sphere marks each source; three wire rings show a point-light range, while a
wire base ring and ribs show a spot-light cone and a directional light uses an
arrow. The panel reports colour, intensity,
clustered/scoped mode, range, direction, and actual/requested shadow state.
Transforms come from the baked glTF `LIGHT_*` nodes, while matching
`ship_manifest.json` records override runtime parameters by light id.
Bake-only values such as lamp watts still require rebaking the lightmaps.
Runtime lamps are material-gated to meshes carrying the literal `dynamic`
behavior. Liquefiable-only meshes do not receive them.

Every non-ground-only fluid collision set also reserves one capsule slot for
the player, even when the player starts outside that simulation's domain. The
slot is refreshed from the character controller's position and velocity before
fluid stepping, and is disabled while the player is in noclip mode. The fluid
capsule uses twice the physical character radius (0.6 m instead of 0.3 m)
to make displacement around the player more visible. Its lower endpoint also
extends downward by half the physical capsule radius so the boundary remains
partially submerged at floor level.
