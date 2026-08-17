# Aquanova Gameplay Behaviors

> Demo path: `lab/lite/src/demos/aquanova/behaviors/`

> **Status: Implemented.**

Aquanova gameplay is composed from manifest-assigned behaviors instead of being
hard-wired to mesh-name checks in `main.ts`. A manifest entity may assign
several behaviors to the same mesh, and every assignment is resolved and
instantiated independently.

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
| `pickEntity`            | Collects an intersected entity and emits an event     |
| `enableEntity`          | Forwards a configured entity event as `enable`        |
| `disableEntity`         | Forwards a configured entity event as `disable`       |

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

The `player` definition may set `characterStrength`, the maximum force applied
to contacted dynamic bodies. It defaults to `100`; setting it to `0` preserves
collision while disabling player pushes.

`C` toggles crouching. Over `0.2 s`, the character controller keeps the
capsule's foot position fixed while smoothly changing its total height from
`1.8 m` to `0.8 m`; camera height and the 25% crouched movement-speed reduction
interpolate with the same progress. The transition is reversible and works on
the ground or during a jump. A jump request or held run key requests standing
first; expansion is accepted only while the taller capsule has overhead
clearance. Jump input is buffered through the standing transition.

## Runtime lifecycle

`BehaviorManager` owns manifest assignment resolution, linked-entity
closure, mesh classification, target availability, behavior construction, and
disposal. `main.ts` consumes its classified mesh sets and gameplay queries but
does not parse or merge behavior definitions.

Named implementations remain explicit for structurally different behaviors
such as `player`, `weaponLiquefactor`, `pickEntity`, and `dynamic`; liquefiable definitions use the
shared implementation.

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

`weaponLiquefactor` starts unowned and hidden. The behavior listens for
`entityEvent` addressed to the manifest entity on which it was instantiated.
The `enable` event grants ownership and equips the weapon by animating it from a
lowered, muzzle-down pose to its horizontal firing pose. `Digit1` then toggles
the owned Liquefactor between equipped and holstered; `Digit2` holsters it and
leaves the player unarmed until the pistol behavior is implemented. Holstering
reverses the same presentation animation. Trigger events are ignored while the
weapon is unowned, holstered, or still moving into position.

The held weapon can apply a subtle procedural balancing motion made from
layered low-frequency translation and rotation. This motion is cosmetic: the
aim guide, laser, and crosshair remain stable. The `Weapon sway` control-panel
checkbox persists the preference and blends the motion in or out smoothly.
Walking scales both its amplitude and pace to 2× the idle motion; running with
Shift scales them to 4×. Holding the weapon trigger suppresses sway immediately
so the rendered muzzle remains aligned with the laser origin, then sway blends
back after the trigger is released.

The `Sounds` control-panel checkbox persists a global gameplay-audio
preference. Disabling it immediately stops active Liquefactor loops and
suppresses subsequent pickup, firing, liquefaction, and splash sounds.

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
packed into all running simulations, avoiding misleading colour accumulation
where independently liquefied objects use the same collision primitive.

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
