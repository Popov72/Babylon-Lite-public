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

| Behavior                | Responsibility                                       |
| ----------------------- | ---------------------------------------------------- |
| `dynamic`               | Marks a prop as physically movable                   |
| `anyLiquefaction`       | Liquefies with the manifest's default fluid setting  |
| `stdLiquefaction`       | Liquefies with the standard fluid setting            |
| `explosiveLiquefaction` | Liquefies with the explosive fluid setting           |
| `player`                | Owns first-person input, movement, and weapon firing |
| `weapon`                | Represents the weapon entity                         |

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

The player emits `hitWithWeapon` after a center-screen pick. The player does not
know whether the picked mesh is liquefiable. Each liquefiable behavior listens
for that event, accepts hits addressed to its own mesh while it remains an
active target, and invokes the shared liquefaction service.

## Runtime lifecycle

`BehaviorManager` owns manifest assignment resolution, linked-entity
closure, mesh classification, target availability, behavior construction, and
disposal. `main.ts` consumes its classified mesh sets and gameplay queries but
does not parse or merge behavior definitions.

Named implementations remain explicit for structurally different behaviors
such as `player`, `weapon`, and `dynamic`; liquefiable definitions use the
shared implementation.

Player and weapon are singleton entity behaviors. Geometry behaviors are
instantiated for every matching mesh primitive so a picked primitive can receive
the event directly.

## Migration boundary

The behavior layer owns gameplay policy and event flow. The existing GPU fluid
simulation, collision construction, render graph, and debug tooling remain
services in `main.ts`; they are invoked through the typed behavior context. This
keeps the migration behavior-preserving while allowing those subsystems to move
behind narrower services later.
