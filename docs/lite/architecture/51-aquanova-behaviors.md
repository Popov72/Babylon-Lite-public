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

The manifest's `behaviors` object is the source of truth for behavior names and
parameters. The current definitions are:

| Behavior                | Responsibility                                        |
| ----------------------- | ----------------------------------------------------- |
| `dynamic`               | Makes a prop physically simulated and player-pushable |
| `anyLiquefaction`       | Liquefies with the manifest's default fluid setting   |
| `stdLiquefaction`       | Liquefies with the standard fluid setting             |
| `explosiveLiquefaction` | Liquefies with the explosive fluid setting            |
| `player`                | Owns first-person input, movement, and weapon firing  |
| `weaponLiquefactor`     | Gates and drives the Liquefactor weapon               |
| `weaponAntiGravityGun`  | Grabs and throws dynamic rigid bodies                 |
| `pickEntity`            | Collects an intersected entity and emits an event     |
| `enableEntity`          | Forwards a configured entity event as `enable`        |
| `disableEntity`         | Forwards a configured entity event as `disable`       |
| `playAnimation`         | Starts one animation clip from the ship glTF          |

Definition parameters are merged with per-entity overrides while retaining the
behavior identity; assignments are never flattened into one anonymous parameter
bag.

Liquefaction behavior names are data-driven. Any new definition with
`liquefiable: true` is instantiated as `LiquefiableBehavior`; its `fluidSim`
parameter selects the simulation. Adding another liquefaction preset therefore
requires only a manifest definition and assignment, not a TypeScript class.

Each behavior receives:

- the mesh it is attached to;
- its behavior-specific typed configuration;
- one shared `BehaviorContext` containing the camera, canvas, character
  controller, picker access, event bus, and gameplay services.

## Events

`EventMap` centrally declares every event and payload. `TypedEventBus`
only accepts names from that map and enforces the corresponding payload type for
both publishers and handlers.

`EventManager` owns that bus and the engine integration. It brackets the
scene's registered frame callbacks with `frameStart` / `frameEnd` and translates
Havok's after-step callback into `physicsStep`. Gameplay code subscribes through
the manager instead of registering engine callbacks directly.

The system currently emits:

- `frameStart` before Aquanova's per-frame render work;
- `physicsStep` after Havok has stepped and synchronized nodes;
- `frameEnd` after Aquanova's registered per-frame update callbacks.

`entityEvent` carries a target entity name and event name. It is the generic
manifest-driven link between otherwise independent behaviors. `pickEntity` may
define `raiseEvent: { name, event }`; after collection it emits that payload
exactly once. Consumers ignore events addressed to other entities.

`enableEntity` and `disableEntity` listen for `onEvent` addressed to the entity
that owns the behavior, then forward `enable` or `disable` to the configured
`entity` target. Door ids are valid targets. Runtime door handlers update both
the door's `enabled` property and its portal-traversal state.

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
to contacted dynamic bodies. It defaults to `100`; setting it to `0` preserves
collision while disabling player pushes.

`C` toggles crouching. Over `0.2 s`, the character controller keeps the
capsule's foot position fixed while smoothly changing its total height from
`1.8 m` to `0.7 m`. Its crouched radius is `0.35 m`, keeping the capsule valid
while reducing its standing `0.4 m` radius. Camera height and the 25% crouched movement-speed reduction
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

`BehaviorManager` owns manifest assignment resolution, linked-entity
closure, mesh classification, target availability, behavior construction, and
disposal. `main.ts` consumes its classified mesh sets and gameplay queries but
does not parse or merge behavior definitions.

Named implementations remain explicit for structurally different behaviors
such as `player`, `weaponLiquefactor`, `weaponAntiGravityGun`, `pickEntity`, and
`dynamic`; liquefiable definitions use the shared implementation.

Player and weapon are singleton entity behaviors. Geometry behaviors are
instantiated for every matching mesh primitive so a picked primitive can receive
the event directly.

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
defaults to `pickItem`. The resolved sound for every pickup is loaded once by
`PickEntityBehavior.init()` before behavior instances start.

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

The `Sounds` control-panel checkbox persists a global gameplay-audio
preference. Disabling it immediately stops active Liquefactor loops and
suppresses subsequent pickup, firing, liquefaction, and splash sounds.
The persisted `Volume` slider controls the master gain of both Aquanova audio
engines from `0` to `1`; changing it also affects active weapon loops.

When liquefaction starts it raises `startLiquefaction` once for every unique
entity in the target's linked liquefaction group. If reversal restores the
group completely, it raises `cancelLiquefaction` for those same entities.

## Migration boundary

The behavior layer owns gameplay policy and event flow. The existing GPU fluid
simulation, collision construction, render graph, and debug tooling remain
services in `main.ts`; they are invoked through the typed behavior context. This
keeps the migration behavior-preserving while allowing those subsystems to move
behind narrower services later.

Fluid collision primitives retain stable buffer slots with an `active` flag.
When a placement stops being collidable, Aquanova updates only that flag in
every running simulation; the collision shader skips inactive slots. A prop's
collision is deactivated at the same point its Havok body is removed:
immediately before the fluid simulation's first step.

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
capsule uses twice the physical character radius (0.8 m instead of 0.4 m)
to make displacement around the player more visible. Its lower endpoint also
extends downward by half the physical capsule radius so the boundary remains
partially submerged at floor level.
