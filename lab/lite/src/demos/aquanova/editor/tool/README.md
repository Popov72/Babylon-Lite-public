# Ship layout tool

A small standalone editor for assembling the **Quaternius sci-fi kits** into a
chunked, portal-ready ship. It does one job: pick a module, place it on a grid,
move/rotate/scale it, and say which chunk it belongs to. No asset authoring, no
lighting, no gameplay.

```
cd tool
npm start          # -> http://localhost:5180
```

Node 22+, no dependencies. The kits live in the **BabylonAssets** repository
next to this one and are read either from that local checkout or from
`https://assets.babylonjs.com` — see [Configuration](#configuration). Everything
else comes from the Babylon.js CDN.

---

## Why this exists instead of an off-the-shelf tool

Godot 4 + the _Simple Asset Placer_ plugin, the kit's own Unity project and
Crocotile 3D all handle "place modules on a grid" perfectly well. None of them
emit the thing this project actually needs: a **chunk-tagged layout** with
portal rectangles between compartments. Kenney's Asset Forge is the nicest UX of
the lot but exports a single merged mesh, which destroys per-instance transforms
and chunk boundaries outright.

So the tool is deliberately thin, and the interesting part is the manifest.

---

## Data model

`export/ship_manifest.json` is the **source of truth**. It stores each placement
as a module id plus a transform, so a layout reloads exactly:

```jsonc
{
  "instances": [
    { "id": "P0001", "module": "Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight", "node": "weapon locker",
      "chunk": "CH00_Storage", "position": [0,0,0], "rotation": [0,90,0], "scale": [1,1,1] }
  ],
  "markers":  [ /* doors, see below */ ],
  "lights":   [ /* authored lights, riding a placement — see below */ ],
  "chunks":   [ { "id": "CH00_Storage", "node": "CHUNK_CH00_Storage", "aabb": {...} } ],
  "environmentProbes": [
    { "id": "ENV0001", "boxPosition": [-8,2.5,0], "boxSize": [16,5,8],
      "capturePosition": [-8,2.5,0],
      // where the runtime blends this probe, as opposed to what it projects
      "influenceBoxPosition": [-8,2.5,0], "influenceBoxSize": [19,8,11],
      "influenceInnerBoxSize": [13,2,5], "resolution": 512 }
  ],
  "portals":  [ { "id", "chunkA", "chunkB", "door", "centre", "normal", "corners" } ],
  "doors":    [ { "id", "chunkA", "chunkB", "position", "triggerRadius", "sealed", "leaves" } ],
  // chunkB may be "__SKYBOX__": a window through the hull, with space behind it
  // rather than a room. Such a door is always sealed, and is not in "chunks".
  "adjacency": { "CH00_Storage": [ { "to": "CH01_CorridorA", "portal": "Portal_Door_D00" } ] },
  "view":     { "position": [...], "rotation": [...], "target": [...] },
  "environment": { "strength", "toneMapping", "exposure", "specularAA", "reflectionRoughness" },
  "editorEnvironment": { "strength", "toneMapping", "exposure" },
  "editorPrefs": { "veilAlpha", "bigPalette", "strayChunkCheck", "probes" },
  "fluidSim": [ "viscosity-inplace", "liquid-slow.json" ],
  "behaviors": { "door_liquefiable": { "liquefiable": true } },
  "entities":  { "storageDoorL": { "behaviors": [ { "name": "door_liquefiable", "linked": ["storageDoorR"] } ] } }
}
```

### Names, ids and the .glb

An element's **name** is what its **parent node** is called in `ship.glb`; its
primitives are numbered off it as `<name>_primitive0`, `_primitive1`… The editor
never deals in primitives, and neither does the manifest — everything the
runtime resolves goes through the node name, and it walks down from there.

The name is deliberately **not unique**: naming six crates `crate` gives six
nodes called `crate`, which is exactly how one behaviour entry comes to govern
all six. An element with no name falls back to its **id** (`P0007`), which is
the editor's own handle and what the tool reloads from.

`instances[].node` records the resolved name for each element, so nothing has to
re-derive it, and `doors[].leaves[].node` uses the same.

glTF _meshes_ stay unnamed, and that is not an oversight: the exporter shares
one glTF mesh between every instance of a module, so a mesh entry belongs to the
module, not to any one element. The **node** is the only per-element name slot,
and it is the name Babylon gives the node when the .glb is loaded back.

### Every exported node carries its identity

Because the name is deliberately not unique, it cannot be an identity. Each
placement's glTF node therefore also carries an `extras` block:

```json
{ "id": "P0192", "module": "Modular SciFi MegaKit/Props/Prop_Crate4", "chunk": "CH00_Storage" }
```

Written by putting `metadata.gltf.extras` on the node for the duration of the
export — Babylon's serializer reads exactly that address by default
(`metadataSelector: (m) => m?.gltf?.extras`), and it is put back afterwards so
the editor's own `metadata.placement` is untouched.

**Both loaders hand it back at the same address**, `mesh.metadata.gltf.extras`:
Babylon.js through its `ExtrasAsMetadata` loader extension, and **Babylon-Lite**
through `gltf-feature-extras.ts`, whose detector explicitly includes
`json.nodes.some(n => n.extras !== undefined)` — so node extras are enough to
pull the feature module in, and it assigns to the node _and_ the mesh.

> **This is what makes a mesh removable.** Liquefying a prop takes its mesh out
> of the scene, and its Havok body has to go with it — which needs the runtime
> to know _which_ body that is. Matching on the node name cannot do it: this
> ship has two placements called `crate4` (`P0192` and `P0194`, stacked at 11,3),
> so the .glb has two nodes with that name and a name lookup is a coin toss. The
> id is the manifest's own key and unique by construction.
>
> `module` and `chunk` ride along so a mesh reaches its hull in
> `moduleCollision[module]` in one hop, with no join back through `instances`.
> The e2e suite **fails the run** if a tagged node's id is missing, duplicated,
> or names an instance the manifest does not have.

### Behaviours

Two halves, matching the manifest:

```jsonc
"fluidSim": [ "viscosity-inplace", "liquid-slow.json" ],   // the global list
"behaviors": {                       // a LIBRARY of named definitions
  "door_liquefiable": { "liquefiable": true, "fluidSim": ["viscosity-inplace"] },
  "dynamic":          { "dynamic": true }
},
"entities": {                        // which node names carry which
  "storageDoorL": { "behaviors": [ { "name": "door_liquefiable", "linked": ["storageDoorR"] } ] },
  "storageDoorR": { "behaviors": [ { "name": "door_liquefiable", "linked": ["storageDoorL"] } ] }
}
```

The global `fluidSim` list is seeded from the tool's `config.json`, re-read on
every catalogue request — so adding a sim needs a page refresh, not a server
restart. **A loaded ship's own list wins over it**: `config.json` is what a
_new_ ship starts from, but a saved one carries the list it was authored
against, and letting the two drift apart would silently repoint its behaviours.
An undo snapshot carries no list, so undoing never disturbs the one in force.

**Definition bodies are free-form JSON, written through untouched.** The runtime
owns which flags exist; a tool that normalised the ones it happened to know
about today would quietly drop the rest, and would need editing every time one
was added. So the window is a name and a textarea, and the only thing checked is
that the text parses to a JSON _object_ — an array or a bare number there would
be silently ignored by the runtime rather than rejected.

**Edit behaviours…** in the inspector opens the library: pick from the list,
edit, `New`, `Save`, `Delete`. The list is **alphabetical**, sorted for display
only — definitions accumulate as the ship is built, so the order they were
written in is the history of the project rather than anything you could look a
name up by, and the manifest keeps saying what it always said. Two operations
keep the file consistent by themselves:

- **Renaming** a definition rewrites every entity that referenced it. A rename
  that left them pointing at the old name would silently drop their behaviour.
- **Deleting** one strips it from every entity that carried it, rather than
  leaving entries the runtime would ignore.

**Attaching** happens on the selected element, keyed by its **node name** — so
the panel shows how many elements that name governs (`"crate" — 4 elements`).
Attaching one can be affecting one crate or thirty, and there is no other way to
tell. A name is required first: a behaviour with nothing to key on could never
be matched to an element.

**The same behaviour may be attached more than once.** An entity's behaviours
are a _list_ of assignments, not a set keyed by name: the runtime walks the list
and builds one instance per entry, so two entries of one behaviour carrying
different parameters are two different things happening to one prop. Which means
a name is **not** an identity here — the panel keys its rows, and the model keys
its edits, by **position** in the list. Keying by name is exactly the bug this
replaced: the second row's `Remove` took the first one away, and both parameter
boxes wrote to the same assignment. Repeated names are numbered on screen
(`#1`, `#2`), since otherwise nothing would tell the rows apart.

**`linked`** appears only for definitions with `liquefiable: true`, because
linking is for pieces that melt as one — a pair of door halves. The candidates
are the other named nodes **in the same room**: linked pieces are neighbours in
practice, and a ship-wide list would be hundreds of entries long. The list is
de-duplicated, drops the node itself, and is omitted from the manifest when
empty — the absence is what "this one stands alone" means.

**Parameters** ride the applied entry beside `name`, and are edited as **raw
JSON**, one box per attached behaviour:

```jsonc
{ "name": "player_startpos", "direction": [-1, 0, 0] }
```

Raw JSON for the same reason the definition body is raw JSON: the **runtime**
owns which parameters a behaviour understands. The panel used to offer three
number fields for `direction` and nothing else — one runtime parameter promoted
to a widget, with no way to author a second, and no way to see one that some
other tool had written. Every key is now carried through untouched, in and out,
by one symmetric pair of helpers (`readBehaviorExtras` / `writeBehaviorExtras`)
shared by the manifest writer and the undo snapshot, so the two can never
disagree about what an assignment may carry.

The box holds the entry **minus its `name`** — the name is the identity of the
assignment, not a parameter, and a `name` typed inside is ignored. An empty box
means "no parameters"; its placeholder shows what the definition suggests, which
is the hint the old fields gave by pre-filling. Invalid JSON is reported under
the box and **changes nothing**, and the text is left as typed rather than
replaced with the stored value — that would throw away the edit in progress.
Committed on blur, not on every keystroke: JSON is invalid for most of the time
it takes to type one.

Three keys are not opaque:

- **`name`** — ignored, as above.
- **`linked`** — cleaned exactly as the picker cleans it, and shown in the JSON
  as well so neither view can silently contradict the other.
- **`direction`** — a vector, so it is mirrored between editor and glTF space.
  Stored **as typed, not normalised**: normalising on every commit would fight
  you as you type. An all-zero vector is dropped rather than written, since it
  names no direction at all.

> A hand-edit that splits a parameter into a sibling entry of its own —
> `[ { "name": "player_startpos" }, { "direction": [...] } ]` — is **folded back
> into the entry above it** on load, with a console warning. An entry with no
> `name` means nothing to the runtime, so that is the only reading under which
> it means anything, and dropping it would silently lose the edit.

> A manifest written before `entities` existed keyed `behaviors` by _node_ name,
> so each entry meant "this node has these flags". Loading one keeps the bodies
> as definitions **and applies each to the node it was named after** — keeping
> them without the application would silently un-liquefy the ship. Detected by
> the `entities` key being absent rather than empty, since `serialize()` always
> writes both.

### Lights

A light **rides a placement**: its node is a child of that element's node, so
moving or turning the module carries its light, and the authored `offset` /
`rotation` are read in the module's own local space. That is the whole reason
per-module defaults are possible — "the panel's lamp sits 5 cm under its face"
is true of every copy of the panel, wherever it ends up.

```jsonc
"lights": [
  {
    "id": "L0001", "owner": "P0192",
    "offset": [0, -0.05, 0], "rotation": [0, 0, 0],
    "runtime": { "type": "point", "clustered": true, "color": [1,1,1],
                 "intensity": 1, "range": 8, "angle": 90, "castsShadows": false }
  }
]
```

`runtime` is the Babylon-Lite light the game creates, and it is the **whole** of
the ship's direct lighting: there is no second, pre-computed half, so what is
authored here is exactly what is rendered — in the editor's **Runtime** view and
in the game alike.

`type` may be **`"none"`**, which switches a lamp off without deleting it. Its
position, colour and settings survive, which is what makes trying a room with one
fewer light a two-click experiment rather than an edit you have to undo.

#### Each kind of lamp starts at its own numbers

`intensity`, `range` and `angle` do not mean the same thing from kind to kind. A
point light fills a small room from the inside, where `1` is already bright; a
spot is aimed at a surface metres away through a cone, and needs two orders of
magnitude more before the wall it points at looks lit at all. Carrying one
number across a change of kind is how a spot ends up looking broken, and how you
end up assuming spot lights "do not work".

So `DEFAULT_LIGHT` is the shared starting point and `LIGHT_TYPE_DEFAULTS` holds
only what a kind disagrees about — today, just the spot:

| kind        | intensity | range | cone |
| ----------- | --------- | ----- | ---- |
| point       | 1         | 8 m   | —    |
| spot        | **80**    | **6 m** | **120°** |
| directional | 1         | —     | —    |

`defaultsFor(type)` is the one place those are combined, and all three paths go
through it: creating a light, seeding one from `kit_lights.json`, and filling a
gap in a loaded record.

**Changing the kind brings the new kind's numbers with it, for the fields you
never chose.** A field still sitting at the *outgoing* kind's default was never
an opinion, so it follows the lamp; a number you typed is kept, whichever kind
you typed it for. That is what makes "add a light, choose spot" land on a usable
spot, while a spot tuned to `50` stays at `50` across a switch to `none` and
back. It also means a kind change is never destructive: nothing you entered by
hand is overwritten.

**A light emits along its own local −Y**, so the default rotation `[0,0,0]` is a
ceiling panel shining at the floor. Down is where almost every lamp in the kit
points, and −Y is the only axis that needs no rotation to get there — which is
why it was chosen over the more obvious −Z.

Three combinations are **settled on the way in** rather than trusted to the
inspector, a loaded manifest and `kit_lights.json` each getting them right on
their own — the same treatment `sealed` gets on a skybox door:

| Rule                                   | Why                                                 |
| -------------------------------------- | --------------------------------------------------- |
| a point light never casts a shadow     | Babylon-Lite has no cube shadow generator           |
| a clustered light never casts a shadow | the cluster is a data texture with no shadow map    |
| a directional light is never clustered | it has no position to bin and no falloff to cluster |

A light is part of what an element **is**, so it is copied with `Ctrl+D` and
deleted with its owner, exactly like that element's collision shapes — and what
is copied is the lamp as it stands, tuned intensity and all, not the kit's
default for that module. A light whose owner is missing on load is dropped
rather than stranded at the origin.

#### Modules arrive lit — `public/data/kit_lights.json`

Placing `Prop_Light_Wide` and then hunting for where its lamp should go, every
time, for every panel in the ship, is the kind of work the editor exists to
remove. `kit_lights.json` keys a list of light partials by module id, and
`placeAt` applies them:

```jsonc
"modules": {
  "Modular SciFi MegaKit/Props/Prop_Light_Wide": [
    { "offset": [0.58, -0.1, 0], "rotation": [0, 0, 0],
      "runtime": { "type": "point", "clustered": true, "range": 8 } }
  ]
}
```

The values were **measured off each module's `M_Light` primitive** — the
emissive strip _is_ the lamp, so the lamp is seeded on its face and pointed the
way it faces. Anything left out falls back to the defaults **for the kind the
seed asks for**, so a seed that says `"type": "spot"` and nothing else gets a
spot's 80 / 6 m / 120°, and the whole record goes through `normalizeLight()`, so
the file cannot author an impossible light.

A **list**, because one strip is not always one lamp: `Prop_Light_Corner` is a
quarter-circle arc that one lamp cannot light evenly, so it is served by three
spaced along it, turned to the tangent at their midpoints. `Prop_Light_Floor` is the one
that does not point down — its face is tilted 18° off vertical, so it is rolled
161.6° about X to throw the light up the wall rather than at the floor it is
standing on.

Offsets are in **editor space**, which is not the space the `.gltf` is written
in: Babylon's loader turns the file 180° about Y and mirrors Z, and the two net
out to `x_editor = −x_gltf`. Reading a strip's centre out of the file and using
it unchanged puts the lamp on the wrong side of the module.

These defaults are a **seed, not a derivation**. They fire once, when the module
is first placed; from then on the lights belong to that element, so an edit
survives a save and a copy carries the edit rather than the default. Editing
`kit_lights.json` therefore never touches a ship that is already laid out — and
the two paths that bring their own lights, `Ctrl+D` and loading, place with
`noLights` so the seed cannot fire a second time beside them.

#### Authoring one — the panel and the plate

**Add light** in the inspector attaches one to the selected element; with a
light already selected it attaches a second to the same owner, which is the
obvious next click. Selecting it swaps the inspector into the light form.
Position is relabelled **Offset** — a light's node hangs off its owner, so those
three numbers are read in that element's own space — and Scale and Size go away,
because a lamp has neither.

Rows the engine has no meaning for are **disabled rather than hidden**, with a
line underneath saying why: a greyed-out Cone row still tells you a spot light is
the thing that has one. The rules are exactly `normalizeLight()`'s, so the panel
can never author a light the model would quietly rewrite behind it.

**Range works on every lamp that has a position**, and that is a recent thing
worth spelling out, because for a while it did not. Babylon's PBR materials
default to *physical* falloff — `1 / d²`, computed in
`computeDistanceLightFalloff_Physical`, with no cut-off anywhere in it — so a
light's `range` reached the shader and was never read. Worse, it was not merely
ignored on a clustered lamp: `ClusteredLightContainer` still **sizes and culls
each light proxy by `range`** (`clusteredLightContainer.pure.ts` writes it into
the cluster data and uses it to pick the depth slices a light occupies), so a
physical-falloff clustered lamp was still lit at full `1 / d²` strength right up
to the edge of its proxy and then stopped dead. That straight-edged block is
what Babylon's own [clustered lighting
page](https://doc.babylonjs.com/features/featuresDeepDive/lights/clusteredLighting/#lights-with-a-falloff-other-than-falloff_default-are-not-supported)
warns about, and its remedy is the one taken here.

The page offers two: `usePhysicalLightFalloff = false`, which gives Babylon's
linear *standard* ramp, and `useGLTFLightFalloff = true`, which gives the glTF
window `saturate(1 − (d²/range²)²)² / d²`. The preview takes the second, in
`materialFor()` in `runtime.js`, for one reason: it is the curve Babylon-Lite's
clustered shader **hardcodes** (`clustered-light-wgsl.ts`), so a clustered lamp
— which is nearly every lamp on this ship — now fades on screen exactly as it
will in the game. Its cone matches too: Babylon's glTF cone falloff is shaped
from the angle alone, the same `saturate((cos θ − cos ½α) / (1 − cos ½α))²`
ramp the clustered shader uses, which is why the `exponent` argument the preview
passes to `SpotLight` is `0` and means nothing.

An **unclustered** lamp is the near miss, and the panel says so under the field.
Babylon-Lite's analytic path (`multilight-wgsl.ts`, `singlelight-point-wgsl.ts`,
`singlelight-spot-wgsl.ts`) has no glTF branch at all — its `lightFalloffMode`
selects between physical `1 / d²` and the linear ramp `max(0, 1 − d/range)` and
nothing else. The game therefore builds its ship materials with
`usePhysicalLightFalloff: false` (in `lights.ts`, on the same material clones
that carry the clustered state), which is the only range-respecting curve on
offer there. So the two agree on *where* an unclustered lamp ends and disagree
on its shape between here and there, and an unclustered spot has a harder rim in
the game than in this preview — `pow(cos θ, exponent)` steps to zero at the cone
edge whatever the exponent, so there is no value that would round it off. No
lamp on the ship is unclustered today; the note under the field is there for the
first one that is.

A **directional** light is the one lamp Range is still disabled on. It has no
position, so no distance, so nothing to fall off over.

**None of those four numbers has to be typed.** While the selection is lights
and nothing else, the wheel over the viewport tunes the lamp instead of driving
the camera and the transform tools — intensity bare, range on `Ctrl`, a spot's
cone on `Alt`, and `Shift` to turn it 0.5° about the current `R` axis in the
lamp's own space. Range and cone move by one press of the matching arrow in the
panel; intensity moves by a whole unit on a spot and by `0.05` on a point light,
since the same number means very different things on the two. The panel follows
along as it moves, and the whole gesture undoes in one step. It
is the one place the bare wheel is not the camera dolly — see [Carrying versus
dragging](#carrying-versus-dragging) for why that trade is worth making, and a
line under the panel says so on screen.

In the viewport a light draws as a fixed **25 cm plate** with a short line out of
its face showing which way it emits — amber while the lamp is on, a cold
blue-grey once `type` is `"none"`. The plate is pickable and draggable like
anything else, and dragging it moves the light within its owner rather than
through the world.

Lights are the **only** thing in this editor parented to another element, so the
plate carries `metadata.gizmo` and every path that walks an element's meshes
skips it: the ghosting veil, the Size readout and chunk volumes, the outline,
and — the one that would have shipped a lamp-shaped hole into the game — the
glTF export.

#### Leaving in the .glb

Each light reaches `ship.glb` as a **bare node named `LIGHT_<id>`**, still a
child of the element it rides, carrying its whole record in the glTF node's
`extras`:

```jsonc
{ "name": "LIGHT_L0001",
  "translation": [0, 2.5, 0],
  "extras": { "id": "L0001", "kind": "light", "owner": "P0192",
              "chunk": "CH00_Storage",
              "runtime": { "type": "point", "clustered": true, … } } }
```

There is nothing to draw, so the node has **no mesh and no children** — the
gizmo is left out of the export, and Babylon's serializer is happy to write a
node that is only a name, a transform and `extras`. The offset rides in that
transform rather than in `extras`, because the node is parented exactly as it
was authored: the hierarchy carries the offset across for free, in the module's
own space.

`kind: "light"` is what tells these apart from a placement's extras, which carry
a `module` instead — anything reading the file can pick out the lamps in one
pass. `chunk` rides along so a consumer can work one room at a time without
walking back up the hierarchy. The whole record goes out rather than a summary of
it: the runtime cannot see the editor, so anything it would have to re-derive is
something that will eventually drift.

The manifest carries the **same** records, by id, and the runtime lets it win:
the glb node supplies the transform, because it is parented under the placement
that owns the lamp and its world matrix already carries both that transform and
the loader's glTF→Lite mirror. Lighting can therefore be re-tuned and saved
without re-exporting 38 MB of geometry.

### Environment

```jsonc
"environment": {                // what the DEMOS read
  "strength": 1.7,              // scene.environmentIntensity
  "toneMapping": "Khronos PBR Neutral",
  "exposure": 0.55,             // the linear multiplier, exactly as the slider shows it
  "specularAA": true,           // authored default for the player's graphics toggle
  "reflectionRoughness": 1      // multiplier over every ship material's authored roughness
},
"editorEnvironment": {          // this tool only — the demos must not read it
  "strength": 1.5,
  "toneMapping": "Khronos PBR Neutral",
  "exposure": 0.55
},
"editorPrefs": {                // this tool only, and not lighting at all
  "veilAlpha": 0.5,             // how see-through Shift+H makes an element
  "bigPalette": true,           // double-width palette with double-size tiles
  "strayChunkCheck": true,      // warn about elements that look mis-chunked
  "probes": {                   // which probe boxes the window puts back up
    "CH01_Corridor1": {
      "alwaysVisible": true, "envFaces": true,
      // which of the probe's three volumes are drawn — its section eyes
      "visibleParts": { "box": true, "influence": true, "inner": false }
    }
  }
}
```

**Two rigs, because there are two pictures.** The editor's authoring rig adds
four analytic lights the game does not have, so one pair of Env/Exposure values
cannot serve both: what reads well while building is nothing like what the game
needs. The two are separate controls, not one control that changes meaning —
**Settings ▸ Editor** holds the authoring rig and **Settings ▸ Runtime** holds
the ship's, so both are visible at once and neither can be edited by accident.

`environment` is the **Runtime** rig — the picture the
demos render — and it is what `aquanova` and `liquefactor` read.
`editorEnvironment` is the other one, filed separately because it describes this
tool and not the ship.

**`environment` also carries two material settings**, because they change the
same picture and the game applies them the same way. `specularAA` is the
authored default for PBR specular anti-aliasing; the game seeds its graphics
setting from it, and a player who has picked their own value in the control
panel keeps theirs — the manifest moves the default, it does not overrule
anyone. `reflectionRoughness` multiplies every **ship** material's
`roughnessFactor` on load, scaling the ORM texture's green channel rather than
replacing it, so the kit's per-texel variation survives; above 1 it broadens
metallic reflections, which is the cheapest cure for specular shimmer. The
weapon viewmodel is deliberately left out: it is not in the editor's preview, so
a dial tuned against corridor panels must not restyle the gun in your hands. The
exported `.glb` keeps the authored roughness either way — the multiplier exists
only here, which is what makes the preview and the game agree.

**`editorPrefs` is not lighting**, which is why it is a block of its own rather
than more fields on `editorEnvironment`. It is saved with the ship all the same:
reopening a ship with the palette half the size you left it, or ghosts at
someone else's opacity, is the tool having forgotten how you were working on
*this* ship. Both keep a `localStorage` copy as the fallback for a manifest that
predates the block — the manifest wins whenever it carries a value.

There is **one** `strength` per rig, not two. It used to be split — a second
`dynamicStrength` for the meshes a lightmap could not cover — and with the
lightmaps gone there is no longer a class of mesh that needs its own number.

**`exposure` is the plain linear multiplier**, used identically at both ends:
what the slider shows is what `scene.imageProcessingConfiguration.exposure` gets
here and what `scene.imageProcessing.exposure` gets there. It used to be stored
in _stops_ and raised to a power by the runtime, which made a
plausible-looking `0.3` mean `2^0.3 = 1.23` — more than twice what it appeared
to say. Nobody could hold both readings in mind, so the conversion is gone from
both sides.

> A **negative** exposure can only have come from that old format, since the
> slider has never gone below `0.15`. Those are converted with `2 ** v` on load
> rather than clamped up to the floor, with a console warning.

`toneMapping` is **applied**, not assumed: the editor resolves the name the same
way the demos do (`neutral` / `aces` / `standard`, and `none`/`off`/`linear` to
disable it). Hard-coding it here was the one way the two could still disagree
after the exposure was made literal.

The HDRI itself is **not** in the manifest. It is fixed for this ship
(`ENV_HDRI` in `editor.js`, served from the env folder), and a field the runtime
does not read is a field that goes stale.

### Why the editor does not look like the game

Because the editor has four analytic lights and the game has none. `aquanova.ts`
lights the ship from the HDRI and the kit's emissive fixtures alone; the
editor's `hemi`, `hemiUp`, `key` and `fill` exist purely so every module stays
legible from any angle while you build.

Measured on the real ship, they are the whole story: **mean 165.7 with the rig,
35.0 without**, so about four fifths of what you see in the editor is lighting
that will never ship. The `environment` numbers, by contrast, pass through
untouched — the demos apply `strength` and `exposure` exactly as this does.

So the **Runtime** view silences the rig and drops the global HDRI, leaving the
authored lamps and each room's own environment probe — which is exactly the set
of lights the game has. That is the only honest preview, and the only sound way
to set the `environment` rig, which _is_ what the demos read. Both rigs are held
at once, so switching views restores each exactly.

This used to be a **Runtime light** checkbox of its own. Silencing the rig was
only ever meaningful together with putting the runtime's own lighting up in its
place, and as a switch of its own it was one more thing to forget — so the view
mode owns it.

`export/ship.glb` is a **derived artefact** and is never read back — one file
with every chunk as a named `CHUNK_<id>` parent node. Reloading from a .glb
would be lossy, because an exported mesh no longer knows which kit module it
came from; **Load** re-instantiates from the kit folder using the manifest
alone.

**Save writes both.** There used to be an `Export glb` button beside `Save`, and
nothing good came of the pair: the manifest and the .glb describe the same ship,
both demos read them together, and a saved manifest sitting beside a .glb from
two edits ago is a ship that renders as neither of them. Since the
glb is derived, there is no state in which you want one and not the other — so
`Ctrl+S` writes the manifest first, then the glb, and a glb that fails to write
never lets the run report the manifest as unsaved.

> The price is honest: the real ship's glb is ~38 MB, so `Ctrl+S` now takes as
> long as an export did. Making it conditional on the geometry having changed
> would mean tracking that reliably, and the failure mode of getting it wrong is
> exactly the stale pairing this removed.

### Handedness: the manifest speaks glTF, the editor does not

The editor is a **left-handed** Babylon scene; glTF is **right-handed**, and the
exporter mirrors X on the way out. Measured: an element the editor holds at
`[7, 3, 5]` lands in the .glb at `[-7, 3, 5]`. The runtime's loader mirrors it
back, so the geometry round-trips — but the _manifest_ sits beside the .glb and
describes it, and the runtime reads it as glTF space, negating X in a dozen
places: colliders, portal openings, room membership, the player's facing.

So the split is by **who reads the field**:

| written mirrored (glTF space)                    | left in editor space    |
| ------------------------------------------------ | ----------------------- |
| `chunks[].aabb` — min/max swap with the flip     | `instances[]`           |
| `portals[].centre` / `normal` / `corners`        | `markers[]`             |
| `doors[].position` / `direction`                 | `colliders[]`           |
| `collision[chunk]`                               | `moduleShapes[]`        |
| `moduleCollision[module]` — **local**, see below | `stageLayout[]`, `view` |

The right-hand column is the tool's own reload data: it exists to rebuild the
editor and never leaves it. The left-hand column is everything the game
consumes; getting it wrong mirrors the ship against its own geometry, which is
how a player start authored to face the door came out facing away from it.

**The manifest says all of this itself**, under a `space` key naming every
top-level field as `gltf`, `editor` or `none`, with the conversion rules beside
it. A source comment is no use to a runtime, and this split has already caught
one out: composing a glTF-space module hull onto an editor-space `instances[]`
entry puts the collider on the **wrong side of the prop**, where it still looks
entirely plausible. An e2e check fails the run if a top-level key is not
declared, so a new block cannot ship without someone saying what space it is in.

> Two traps worth naming, both silent:
>
> - **Take a placement's transform from the loaded glTF node, not from
>   `instances[]`.** The node is in the same space as the collision blocks;
>   `instances` is not. The node's `extras` carries `id`, `module` and `chunk`
>   for exactly this.
> - **`moduleCollision` is a _local_ transform in glTF space.** Converting it
>   back needs the **rotation as well as the centre** — `[-x,y,z]` for the
>   point, `[-x,y,z,-w]` for the quaternion, because mirroring flips the
>   handedness of the turn too. Centre-only conversion leaves a turned box
>   mirrored, which looks correct on anything symmetrical.
>
> Mirroring `instances[]` as well would make the runtime-facing half uniform,
> but it would not make the _file_ uniform: `colliders`, `moduleShapes`,
> `stageLayout` and `view` are authoring data and would still be editor space,
> and `instances` is what the editor reloads from, so it would need un-mirroring
> on load and a schema bump to keep old manifests readable. The line has to fall
> somewhere; saying where is worth more than moving it.

`direction` is flipped on the way in as well, so the inspector and the `X` axis
gizmo always agree with each other. The flip is its own inverse and one helper
serves both directions — the undo snapshot uses the serialised form too, and a
one-way conversion there would mirror the facing on every undo.

### Names in the .glb

An element's **name** is what its parent node is called, with its primitives
numbered off it; an unnamed one falls back to its id. See the data-model
section above — the point is that _everything_ the runtime resolves goes
through the node name.

The renames last only for the duration of the export; inside the editor
elements keep their `P0007` node names.

**Scene skeletons are hidden for the duration too**, and for the same reason as
the veil stand-ins: they would be written into a file that has no use for them.
Several kits — the pirate characters above all — ship rigged meshes, so loading
one puts a `Skeleton` in `scene.skeletons`, and Babylon's glTF serialiser walks
that array directly rather than going through `shouldExportNode`. Every bone
whose transform node is not in the export's node map then logs *"Exporting a
bone without a linked transform node is currently unsupported"* — one line per
bone, hundreds of them on a save — and the skin is dropped from the file
regardless.

Dropping it is the right outcome, which is why the fix is to stop trying rather
than to make it work. glTF ignores a skinned node's own transform: every
instance of a skinned module shares one skin, so a file that carried them would
stack every copy of a character at the same place. The ship is exported as
static geometry and the runtime rigs what it needs. `exportGlbInner` therefore
`splice`s `scene.skeletons` empty before the write and pushes them back in its
`finally` — safe because `Scene.render` builds its active skeleton list from
`mesh.skeleton` during `_evaluateActiveMeshes`, not from that array, so the
viewport behind the overlay keeps drawing normally.

Every save moves the previous manifest aside as
`ship_manifest.<YYYYMMDD-HHMMSS>.json` in the same folder. The manifest is the
one thing in the project that cannot be regenerated, and the files are a few KB
each, so the history is kept in full rather than rolled.

### Loading locks the editor

A load rebuilds the scene one module at a time, so there is a long window in
which the ship is half there. An edit landing in that window acts on a scene
that does not exist yet — it can select an id that is about to be recreated,
drag a wall a frame before it is disposed, or push an undo snapshot of a
half-restored ship. None of that is recoverable, so input is shut off outright
rather than defended against case by case.

Two layers, because either alone leaks:

- **An overlay** covering the window, with the message and a spinner. It says
  what is happening and it swallows every pointer event.
- **`inert` on the four panels and on every floating tool window**, which takes
  them out of hit-testing _and_ out of the focus order in one attribute — an
  overlay alone would not stop a `Tab` into a toolbar select, or a keyboard
  shortcut. The keydown handler bails too, but it lets the **browser's own**
  keys through: `F5` and `Ctrl+F5` still reload, because a wedged editor is
  exactly when you need them.

The lock is a depth counter, not a flag, because loads nest: the boot autoload
runs inside the boot itself, and the inner one finishing must not reopen the
door. `whileBusy()` always unlocks, including when the load throws. The overlay
is in the markup **unhidden**, so there is never a frame in which a half-built
editor looks ready to use.

---

## Controls

The workflow is modal by state, not by buttons: there are **no gizmos and no
move/rotate/scale modes**.

Placing uses a **ghost** — the real textured module, drawn translucent —
following the cursor snapped to the build plane; clicking drops it, and while it
follows, the mouse cursor is hidden because the module _is_ the cursor.

Moving something already placed is a plain **drag**, which moves the real
geometry rather than a ghost so that it works on a whole selection at once.

The **current element** is whatever the ghost holds, or — when nothing is being
placed — **the selection**. Hover used to outrank the selection, on the
reasoning that pointing at something is a more immediate statement of intent.
In practice it made every edit conditional on where the mouse happened to be
resting: turn a wall, drift the cursor a pixel onto its neighbour, press `R`
again, and the neighbour turns. You end up parking the pointer over empty space
before touching the keyboard, which is no way to work. Selecting is one click,
and it stays put.

**`Del` is the exception**, and keeps its own hover rule: "get rid of that one"
is a complete thought on its own, and needs no selection to survive afterwards.
So is numpad `.`, which raises the build plane to the top of whatever you are
pointing at — it reads an element rather than changing one, so it has none of
the "the wrong thing moved" problem.

**Every transform reads the same way.** The bare letter picks the _axis_, `Shift`
walks that setting's _value_ and `Ctrl` walks it back; the _edit_ is a gesture,
not a letter.

|        | axis                        | value     | value back | the edit itself                                 |
| ------ | --------------------------- | --------- | ---------- | ----------------------------------------------- |
| move   | `V` (and `Y` for the space) | `Shift+V` | `Ctrl+V`   | drag, `M`, arrow keys                           |
| turn   | `R`                         | `Shift+R` | `Ctrl+R`   | `Shift`+wheel (`Alt` too: about a shared pivot) |
| scale  | `F`                         | `Shift+F` | `Ctrl+F`   | `Ctrl`+wheel                                    |
| mirror | _uses the scale axis_       | —         | —          | `Alt+F`                                         |

It was not always so: the letters used to carry the actions (`R` turned, `F`
mirrored) with the settings behind the modifiers, which left `V` reading one way
and `R` and `F` another for no reason anyone could give. Moving the two edits
onto the wheel — where the third already was — freed all three letters to mean
the same thing.

|                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Place            | click a palette tile to arm it, then click in the viewport. The module stays armed for repeat placement — except on the collision bench, where it is a one-shot.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Move             | **drag** an element (elements stay solid, button held), or **`M`** to pick the selection up and carry it hands-free as a translucent ghost — click to drop, `Esc` to put it back. Dragging one that is already selected moves the **whole selection**; dragging an unselected one selects just it first. **`V`, or the `Drag` combo,** cycles the drag axis: `X/Z (floor)` → `Y (up/down)` → `X only` → `Z only` — safe to change mid-drag. **`Y`, or the combo beside it,** says whose axis that is: `World` or `Local` (the element's own — so a wall turned 90° still slides along its length, and `R` turns it about its own axis). `Esc` or right-click mid-drag puts everything back. |
| Frame            | **double-click** an element                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Bring            | **`B`** — moves whatever is in hand to a grid spot just in front of the camera, resting on the deck under your feet, and **takes the build plane with it**. Works on the armed ghost and on a placed selection alike                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Axes             | **`X`** — show one element's **world** X/Y/Z arrows · **`Shift+X`** — its own **local** axes, which is what scaling acts on. Showing them **also puts moving and turning in that space**, since asking to see an axis is nearly always asking to work along it; `Y` overrides afterwards. With several selected, the one **nearest the cursor** gets them; an armed ghost counts too. They follow a single click to the next element, **keeping their flavour**. The same key again hides them (without touching the space), the other key re-aims them, and pressing either with nothing selected or hovered hides them                                                                    |
| Axis modes       | see the table above — one letter per transform, the same three modifiers on each. All three `Ctrl` pairs are claimed from the browser: reload, the find bar and paste                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Mirror           | **`Alt` + `F`** — mirrors on the current Scale axis (`all` is treated as X)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Turn as a group  | **`Alt` + `Shift` + wheel** — the selection swings about a shared pivot, snapped to the move grid so it lands back on-grid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Resize           | **`Ctrl` + wheel** — steps by the Scale snap on the current scale axis                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Aim a lamp       | while the selection is **lights and nothing else**, the wheel drives the light instead of the tools: bare wheel **intensity** (a whole unit on a spot, `0.05` on a point light, never below 0), `Ctrl` **range**, `Alt` **cone** (spot lights only), `Shift` **turns 0.5° about the current `R` axis, always in the lamp's own space** whatever `Y` says. Range and cone move by one press of the matching arrow in the panel, and the whole gesture is one undo step. `Esc` clears the selection and gives the bare wheel back to the camera                                                                                                                                                                                                        |
| What gets edited | the ghost if one is being placed, otherwise **the selection**. `Del` is the exception and takes the hovered element first                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Rotation axis    | `R` cycles Y → X → Z. Y first: it is the only one a modular kit usually needs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Scale axis       | `F` cycles all → X → Y → Z                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Build plane      | numpad `+` / `-` (or main-row `+` / `-`) by the Move step; numpad `.` jumps it to the top of the hovered element                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Select           | **quick** left-click · `Ctrl`- or `Shift`-click adds to the selection · click empty space clears it · **drag from empty space to rubber-band**, or press **Rect select** to start the rectangle on top of a module. `Ctrl` or `Shift` while banding adds. To reach something behind a door portal, `Shift+H` the door — a ghosted element is click-through                                                                                                                                                                                                                                                                                                                                  |
| Chunks           | the **Chunk** button toggles isolation — pressed (orange), every chunk but the one in the dropdown is hidden · `+` adds a chunk · **Rename** renames the active one everywhere it is used · `Assign` moves the selection into the active one                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Hide             | `Shift+H` cycles the selection **50% → hidden → 50%** — half alpha (and click-through) to see past something, then gone · `H` returns everything to fully opaque · the **Ghost** slider sets how see-through that first state is. Undoable, but not saved — a reload starts with everything visible                                                                                                                                                                                                                                                                                                                                                                                         |
| Id               | inspector `Id` row — read-only. The tool's handle for the element and its node name in `ship.glb` when no `Name` is set; doors, portals and behaviours all reference it, so it is not editable. In a multi-selection it names the element whose transform the fields below show                                                                                                                                                                                                                                                                                                                                                                                                             |
| Name             | inspector `Name` field — the element's **node** name in `ship.glb` (primitives are numbered off it), shared on purpose: elements with the same name share one behaviour entry. Shown in the corner overlay instead of the module id                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Behaviour        | inspector panel — attach library behaviours to the element's node name (the same one may be attached more than once; repeats are numbered), edit each one's parameters as **raw JSON**, and pick the `linked` nodes a liquefiable one melts with · **Edit behaviours…** opens the library (name + free-form JSON body)                                                                                                                                                                                                                                                                                                                                                                      |
| Eyedropper       | `Alt`-click a placed element to arm its module                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Compound         | a placed [compound](#compound-objects) selects as one: click any member and the whole group comes · **`Ctrl+Alt+click`** drills in to the single member under the cursor · `Ctrl+D` mints a new instance · inspector **Break apart** dissolves the group and leaves the pieces where they are                                                                                                                                                                                                                                                                                                                                                                                                |
| Nudge            | arrow keys move the selection on X/Z, `PageUp`/`PageDown` on Y — in whichever space `Y` has chosen                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Steps            | toolbar dropdowns — Move defaults to **1 m**, and **`Shift+V`** cycles it (`Ctrl+V` backwards). Move can be **off** (free positioning while dragging). Rot and Scale are keyboard _step sizes_, so instead of "off" they carry **`free`** — a fine step, `±0.5°` and `0.01`. Rot runs `-90°` to `90°`, the sign being which way `R` turns                                                                                                                                                                                                                                                                                                                                                   |
| Camera           | `WASD` flies, `Space`/`C` rise and descend · **right-drag looks** · **right button + wheel sets the fly speed** · `Shift` for 2× · wheel dollies · `F` frames the selection. The left button never moves the camera                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Lighting         | **Settings ▸ Editor** and **Settings ▸ Runtime** each carry their own **Env** slider — strength of the image-based lighting, which is where metals get nearly all their brightness — plus **Exposure** and **Tone**. The runtime rig is saved in the manifest as `environment`, alongside **Specular AA** and **Reflection roughness**, and the game reads all five; the editor's follows it as `editorEnvironment`, which the game must not read, and is also seeded from `localStorage`                                                                                                                                                                                                                        |
| View mode        | toolbar combo — **Editor** (the authoring rig) · **Editor unlit** (raw albedo, no lighting) · **Runtime** (the authoring rig off, the authored lamps rebuilt as real lights and each room reflecting its own environment probe — the lights the game actually has). See [The three view modes](#the-three-view-modes)                                                                                                                                                                                                                                                                                                                                                                       |
| Walk             | toolbar checkbox — walk at the player's eye height (1.8 m) instead of flying. `WASD` moves horizontally at the usual speed, the height follows whatever floor is underfoot, and `Space`/`C` are off                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Undo             | `Ctrl+Z` / `Ctrl+Shift+Z` (or `Ctrl+Y`) — whole-layout snapshots, capped by _memory_ rather than a fixed count (1000 steps on this ship, fewer as it grows), so _anything_ that pushes an entry is undoable: placing, deleting, dragging, turning, scaling, flipping, nudging, hiding, the **Runtime** lighting sliders, every inspector field and every behaviour edit. The **Editor** lighting sliders are deliberately not on the stack — see [Lighting is an edit](#lighting-is-an-edit-the-editors-own-view-is-not)                                                                                                                                                                     |
| Edit             | **`Ctrl+D` puts a copy of the current element — or of the whole selection — on the cursor** as a ghost, keeping every rotation and mirroring, bringing the source's lights, name and compound with it, and setting the drag axis back to `X/Z` so the copy arms where you can see it · **`Del`, or the middle mouse button, deletes the hovered element, or the selection if nothing is hovered** (deleting a hovered element leaves the rest of the selection intact)                                                                                                                                                                                                                                                                               |
| Grid             | `G` · **Editor unlit** shows raw albedo with no lighting · **Settings ▸ Editor ▸ Exposure** — lower keeps pale panels off the tone-mapping shoulder, where their detail flattens out                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Palette          | hover a tile to spin the module through a full 360° turn · **drag the grip** between the palette and the viewport to resize it, double-click the grip to restore the default width                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Save             | `Ctrl+S` — writes `ship_manifest.json` **and** `ship.glb`, and stores the camera position, so reloading puts you back where you were · **Load asks first if you have unsaved changes**, since it discards the whole scene in one click — and so does closing or reloading the tab                                                                                                                                                                                                                                                                                                                                                                                                            |

**Every toolbar control names its shortcut in its tooltip**, or says outright
that it has none. The keys are the whole point of the tool — the combos are a
readout of modal state you are meant to drive from the keyboard, not reach for
with the mouse — so a control that does not mention its key is a key nobody
finds. The `Move` combo was the case that prompted it: the single most used
setting, and nothing on screen said it had a shortcut at all. A test walks
every button, select, slider and checkbox in the toolbar and fails if any lacks
a tooltip, or if a tooltip mentions neither a key nor "no shortcut" — so a new
control cannot ship undocumented.

**A button flashes orange when you press it.** `Save` and `Load`
both do their work somewhere else — a file on disk, a line in the status bar — so
the button itself gave no sign it had been hit, and a press that missed looked
exactly like one that worked. One delegated listener on the document adds a
class for 260 ms, so a button added later is covered without anyone remembering
to, and no handler can forget. It runs on the **capture** phase, so a handler
that stops propagation, or throws, still gets its flash.

> **No transition on it**, and that is not an oversight. A fade _in_ is exactly
> wrong for a flash: with one, the colour was still climbing out of the idle
> grey when the timer took the class off again, and the button never actually
> went orange — measured at 50 ms into a 180 ms fade, sitting at `rgb(98,68,54)`.
> A press reads as instant, so it has to be instant.
>
> Toggle buttons opt out. They already latch solid orange and _stay_ there,
> which says the same thing for longer; flashing a slightly different orange
> first only muddies it. The test measures the colour in the same task as the
> click — a transition cannot hide behind that — and it does so on a **clone**
> of each button, swapped in for the duration, so checking that `Save` lights up
> does not save over the test server's ship.

The inspector shows the selection's **world** bounding box in metres — the world
box, not the local one, so it changes as you rotate. That is the number you want
when checking whether a piece still fits its 4 m tile. With several elements
selected it reports the combined box.

`Del` is the **only** edit that still takes the hovered element first, falling
back to the selection when the cursor is over nothing. "Get rid of that one" is
a complete thought on its own, and needs no selection to survive afterwards —
which is exactly why deleting a hovered element leaves an unrelated selection
intact: pointing at one thing is no reason to forget the others. Nothing
mid-gesture is deletable — `Esc` is the way out of those.

**`Shift+H` parks the selection out of sight**, so you can reach into a room
without flying round the wall in front of it. It cycles **50% → hidden → 50%**:
half alpha to see past something while still knowing it is there, then gone
entirely. There is no third step back to fully opaque, because `H` already does
that for everything at once, and a third state would only make the useful two
harder to reach.

**A ghosted element is click-through.** Half alpha on its own would be no use
for what hiding is _for_: every click would still land on the wall you can now
see through. So `isPickable` goes off with the alpha, and the two states read as
a progression — _see it but reach past it_, then _gone_. The **Ghost** slider in
the toolbar sets how see-through that first state is.

**The selection is kept**, unlike the plain-hide behaviour this replaced. It has
to be: the cycle is driven by pressing the same key again, and since a ghosted
element cannot be clicked, the selection is the only handle left on it.

Hiding **is** undoable, and rides the same snapshot as everything else — not
because it is an edit to the ship, but because a restore begins with
`clearAll()`, which empties the veil. Leaving it out of the snapshot did not
make it immune to undo; it made it _half_ undoable: every unrelated undo
revealed everything, and no redo could put it back. Symmetric is the only honest
option. It stays out of the manifest, so a reload starts with everything
visible, and veiled elements still export — the file on disk is the ship, not
the view of it.

**Hiding is the shifted one, not unhiding.** They are not symmetric: an
accidental hide on a large selection is expensive to notice and to undo, while
an accidental unhide costs a keystroke. The cheap direction gets the bare key.

Two details that would otherwise bite:

- **Half alpha cannot be per-instance the obvious way, and the shortcut is a
  trap.** Placements are hardware instances sharing one source mesh — and one
  _material_ — per module. Babylon refuses per-instance `visibility` outright
  (_"Setting visibility on an instanced mesh has no effect"_), and the next idea,
  forcing that shared material to `ALPHABLEND` with a per-instance colour
  buffer, **breaks the depth buffer**: it moves every mesh drawn with that
  material into the transparent pass, which does not write depth. The kit shares
  materials across modules, so ghosting one wall stopped most of the ship
  occluding and geometry behind walls started showing through. Cloning is the
  only way to make one element translucent without changing what everything else
  is drawn with — so a ghosted element's meshes are swapped for clones carrying
  a cached translucent copy of the material (`ghostMaterialFor`, shared with the
  placement ghost, sharing geometry _and_ textures so the cost is a draw call,
  not a 2–4 MB atlas).
- **The stand-ins are children of the element**, so they follow every drag,
  turn and scale for free. They must therefore be kept out of the export:
  `exportGlb` wraps its _whole_ body in `withVeilSuspended()`, not just the
  write, because the allowlist and the `_primitiveN` renaming are both built
  from `getChildMeshes()` — a stand-in still in the tree at that moment would
  take a primitive index and be written into the file.
- **One function decides what is on screen.** Isolation and hiding both work by
  disabling nodes, so neither can own `setEnabled` alone — isolating a chunk
  would reveal everything you had hidden, and unhiding would reveal the chunks
  you had isolated away. `applyVisibility()` derives it from both, every time.

**The `Ghost` slider** sets how see-through the 50% state actually is (5–95%).
It repaints the cached veil materials live, so stand-ins already on screen
follow it. Like the editor's exposure it is saved with the ship — in
`editorPrefs`, apart from the ship's own data, because it says nothing about the
ship and everything about how you like to look at it — with a `localStorage`
copy as the fallback for a manifest that has no such block.

The counts sit on the status bar (`… · 2 at 50% · 1 hidden`) whenever anything
is veiled, quoting the slider's current value. A hide you have forgotten about
looks exactly like an element you deleted by mistake, and there is no way back
except `H`.

**`Ctrl+D` hands you the copy** rather than dropping one beside the original: a
duplicate that lands next to its source has to be dragged where you actually
wanted it anyway. It arms a ghost exactly as clicking a palette tile does —
brush and all, so the copy stays armed for repeat placement — seeded with the
source's rotation and scale, so a turned or **mirrored** piece copies as it
looks. It **keeps the source's own height without moving the build plane**: the
ghost carries a private height for exactly this, so a copy of something on an
upper deck reappears up there rather than down at ground level, and the plane
you were building on is left where you put it. Moving the plane was the earlier
answer — one source of truth, visibly followed by the grid — but it meant a
duplicate silently changed where _everything placed afterwards_ would land.
**A multi-selection is carried too**: the ghost holds a
list of items, each with its own offset, turn and mirroring, so `Ctrl+D` on
twelve walls hands you twelve walls.

**The ghost also remembers _which element_ it came from**, as an `originId` on
each of its items, and the drop reads that element to finish the copy: its
[lights](#lights) come across as they are tuned right now, along with its name
and its [compound](#compound-objects) — a fresh group id per source group, so
the copy is its own instance. Without it a duplicate was a copy of the mesh
rather than of the element: a lit panel came back wearing the **kit's default
lamp** instead of the one you had dimmed, and a lamp you had added by hand to
something the kit does not light came back as nothing at all. The
multi-selection path always did this; the single-element one went through the
palette brush, which had no way to carry it. A compound _tile_ has no source
element in the scene, so it brings its lamps in its recipe instead.

**It also sets the drag axis back to `X/Z` first**, and says so. In `Y` mode the
cursor drives the _build plane_ rather than the ghost's own height, and it clears
`baseY` to take that over — which is the very height the copy was just given so
it would appear beside its source. So the copy jumped to the plane instead:
measured 9.5 m below its source, and from a camera down at deck level that is
off the top of the screen. Nothing about "put a copy over there" wants the
vertical axis, so the axis moves rather than the copy being armed somewhere you
cannot see. The status line and the combo both report it, because a mode that
changes itself quietly is worse than one you change by hand.

> **Every way of arming a ghost does this**, not only `Ctrl+D` — a palette tile,
> the eyedropper, a collision shape button. Even without a `baseY` to lose, `Y`
> mode makes the build plane follow however far the cursor has travelled
> vertically, and the ghost with it, so a freshly armed module leaves the screen
> before you have chosen where to put it.
>
> It lives in `interact.js`, next to the three functions that assign the ghost,
> rather than in the click handlers. The first attempt put it in the handlers
> and missed the palette — which is the route the bug was reported against,
> because a tile calls `setBrush` → `armGhost` directly rather than going
> through the `pickmodule` event. `setDragAxis` emits `modes`, so the combo
> follows on its own.
>
> An `M` carry is deliberately exempt: it moves what is already there, and
> raising something is a perfectly good reason to be in `Y` mode. `Ctrl+D` on a
> multi-selection goes through the _same_ function, so it is told apart by
> `opts.copy` rather than by which key was pressed.

### Carrying versus dragging

There are two ways to move something, deliberately, and they differ in what
your hand is doing:

- **Drag** — press, move, release. The elements stay **solid** and follow
  directly. The button is held throughout.
- **Ctrl+D** — a copy of the current element comes up on the cursor. It keeps
  the source's height **without moving the build plane**: moving the plane was
  the old behaviour, and it meant a duplicate silently changed where everything
  placed afterwards would land.
- **Carry** (`M`) — the selection lifts onto the cursor as a **translucent**
  ghost. No button held: move the mouse, turn with `R`, mirror with `F`, fly the
  camera, then click to drop. `Esc` puts everything back where it was.

**A carry is relative.** The elements stay exactly where they are when you press
`M`, and then move by however far the cursor travels, snapped to the move step —
the way Blender's `G` works. It used to teleport them onto the cursor the moment
the key went down, which threw a piece halfway across the room and re-snapped
anything deliberately placed off the grid. Relative motion also means an element
keeps whatever sub-grid offset it was placed with. A palette ghost has no "where
it already was" and simply sits on the cursor, as before.

Collapsing the two into one was tempting — the drag path duplicates rotation,
scale, snapping and axis constraints that the ghost already has. What stopped it
is the _release_: every 3D tool drops on mouse-up, and a press-drag-release that
instead left a wall stuck to the cursor would fight a reflex nobody wants to
retrain. So dragging keeps its contract, and carrying is the hands-free option
next to it. The transparency is what tells them apart at a glance.

While carried, the originals are **hidden rather than moved**: the ghost is the
preview, and leaving solid copies behind would read as "these have been
duplicated". Hidden, not deleted, so `Esc` can put them back untouched.

**Markers are skipped** by a carry — a door has no kit prototype to clone from —
so a selection of nothing but doors simply does not lift, and the drag handles
it as before.

> **The drop composes, it never decomposes.** Each item's landing transform is
> built as _group rotation × item rotation_ and a component-wise scale product,
> not read back out of the world matrix. A mirrored element has a negative
> determinant, and such a matrix has no unique rotation/scale split: measured,
> `Matrix.decompose()` moved a `[-1,1,1]` scale onto **Y** with a compensating
> turn. It looks identical on screen and is a different ship in the manifest.

**The bare wheel belongs to the camera.** It dollies, full stop — that is what a
wheel does in a 3D view, and every attempt to give it a _third_ job fought that
expectation. The two edits it does carry sit on its modifiers: `Shift` + wheel
turns the current element and `Ctrl` + wheel resizes it. That is what freed the
letters to carry those actions' _settings_ instead, so `R` and `F` read the same
way `V` always has. And **holding the right
button turns the wheel into a fly-speed control** — the right button already
means "I am driving the camera", so adjusting how fast reads naturally and
cannot collide with editing. Steps are multiplicative, so the control feels the
same at 3 m/s and 100 m/s, and the speed shows in the status line.

**A lamp selection is the one exception**, and a deliberate one. A lamp is
_aimed_ rather than built: the loop is nudge, look at what the room does, nudge
again, twenty times over, and the number under the hand nearly every time is the
intensity. So while the selection is lights and nothing else — mixed with
anything, or with a ghost armed, and the ordinary bindings are back — the wheel
tunes the lamp: bare is intensity, `Ctrl` is range, `Alt` is the cone of a spot
light, and `Shift` turns it. Putting the most-used of the four behind a modifier
would have turned that whole loop into a chord for nothing, and the dolly is one
`Esc` away, still on the right button, and still there the moment the selection
is anything else.

> The turn is **local, always**, whatever `Y` says, and **0.5°, always**,
> whatever the Rot snap says. Both follow from the same thing: a lamp emits
> along its own axis. "Tilt it a couple of degrees" can only mean about that
> axis — a world axis would swing the beam somewhere nobody asked for on any
> lamp already angled — and the Rot snap exists to lay walls out on a grid,
> where 90° steps are the point and are useless for aiming. The step is applied
> on the **right** of the node's quaternion, which composes it in the node's own
> frame; the shared `spinNode` used by `Shift` + wheel elsewhere multiplies a
> _world_ axis on the left, which for a lamp parented to the element it rides
> would be read through the owner's rotation.
>
> Range and cone step by the same amount as one press of the matching arrow in
> the light panel (0.5 m, 5°), so the two controls stay in agreement. Intensity
> cannot: the number is read in wildly different units by the two lamps that use
> it. A spot is aimed at a surface metres away and needs whole units before the
> room looks any different, while a point light fills a small room from the
> inside, where the panel's own 0.1 was already a jump — so a notch is **1 on a
> spot, 0.05 on a point light**, and each lamp in a mixed selection takes its
> own. Every step is clamped where `normalizeLight` would clamp it anyway —
> intensity at 0,
> range just above it, the cone inside 1–179°. A gesture is one undo entry, the
> same 400 ms rule the other wheel edits use. Aiming a lamp whose type has no
> such setting says so in the status line rather than doing nothing: a
> directional light has no range, and only a spot has a cone.

> Why not `Ctrl` + `WASD` for a slow mode? `Ctrl`+`W` **closes the browser tab**
> — it is reserved by Chrome and a page cannot cancel it — while `Ctrl`+`S` and
> `Ctrl`+`D` are already Save and Duplicate here. Only `Ctrl`+`A` was actually
> free.
>
> The same trap decided the axis-mode keys. `Ctrl`+`T` would have been the tidy
> pair for the move step, and it is **unusable**: `Ctrl`+`T`, `Ctrl`+`N`,
> `Ctrl`+`W` and their `Shift` variants are handled by the browser _before_ the
> page sees the key, so `preventDefault()` has no effect.
>
> `Ctrl`+`R` is **not** on that list, despite what much of the internet says.
> Confirmed by hand in Edge: the editor claims it and the page does not reload.
> Worth knowing, because automated tests cannot answer this — Playwright injects
> keys through CDP, straight into the renderer, so a reserved shortcut looks
> claimed to a test and still fires for a real user. The only reliable check is
> a finger on the key.

A wheel gesture on the right button also has to mark that button as _used_, or
releasing it would fire the cancel gesture (a right-click that never moved) and
throw away the selection you were about to fly over to.

`Alt` + `R` swings the whole selection about a shared, grid-snapped pivot —
that was `Alt` + wheel before the wheel gave up rotation. It must
`preventDefault()`: any `Alt` combination can open Chrome's hamburger menu, which
then swallows every following keystroke, the same trap as bare `Alt`.

**`V` cycles the drag axis** through `X/Z (floor)` → `Y (up/down)` → `X only` →
`Z only`, and the **Drag** combo in the toolbar shows which is active. It
applies to a **ghost you are placing** as well as to a drag.

The floor plane is the default, which is what a modular kit laid out on a grid
wants; stacking decks or hanging a ceiling part needs `Y`; and the two
single-axis modes are for sliding a piece along one line without the other
coordinate drifting off-grid. The single-axis modes still track the cursor on
the floor plane and simply drop the other component — for the ghost, the locked
coordinate stays wherever you left it, so it slides along one line from there.

The vertical plane is built with its normal along the **view direction flattened
onto the horizontal**, so it always faces the camera — a fixed world plane would
go edge-on and stop responding the moment you orbited round. `V` is safe to
press mid-drag: the drag re-anchors on the new axis so the element does not
jump.

### `Y` chooses whose axes those are

`V` says _which_ axis a move runs on; **`Y`, or the combo beside it, says whose**
— the world's, or the element's own. A modular kit turns every second wall 90°,
so "slide it along its length" is world Z on one and world X on the next; in
local space it is `X only` on both.

**It governs turning as well.** `R` used to turn about a world axis whatever the
element was doing, so on a wall already yawed 90° "turn about X" tumbled it
about the _room_ rather than about its own length. In local space it turns about
the element's own axis — which is the one you mean when you say "tilt this panel
back a bit". The state is called `axisSpace`, not `moveSpace`, for exactly this
reason: a name that covered only half of what it governs is the kind that goes
wrong later.

> The rotation axis is taken **per element**, matching the origin. Each element
> already turns about its own origin — "which is what a row of props wants" —
> so each turns about its own axis too, and a row of identically-placed props
> tilts together instead of fanning out. `Alt+Shift+wheel` is the exception: one axis has
> to serve a group swinging about a shared pivot, so it comes from the element
> the gesture is aimed at, the same rule the drag uses.

Both spaces run one expression for moving. The travel is projected onto each
live axis, the distance snapped, and the axis added back scaled by it — with the
world's own axes, that is exactly the old "drop the other component", so world
space is not a special case but the same code with an identity basis.

> **The frame is the element's world matrix, taken once.** Its normalised rows,
> not its rotation quaternion, so they are the axes `Shift+X` draws — mirroring
> included. Taken once at the start of a gesture, so turning a piece mid-drag
> cannot make its own axes run away from under the gesture. And `computeWorldMatrix(true)`,
> not the cached one: a turn earlier in the same frame has not been through a
> render yet, and reading the cache left every axis one turn behind. That was a
> real bug, caught by a test that turned an element and nudged it in the same
> breath.

Whose element is always the one being _acted on_: the piece under the cursor for
a drag, the anchor for an `M` carry, the first of the selection for an arrow
nudge — which is the one `X` puts its gizmo on. With several selected they all
move by the one delta measured in that element's frame, the way a set of objects
moves in Blender.

Two consequences worth knowing. **A constrained drag can legitimately do
nothing**: drag across the screen with `X only` on a wall whose own X points into
it, and there is no travel along that axis to speak of — the same as dragging
across a `Z only` constraint in world space. And **local snapping is along the
local axis**, so a piece lands on multiples of the step measured from where it
started, not on the world grid. Both are what Blender does, and both are what
"the axis belongs to the element" has to mean.

**Placing from the palette is always world.** A module that has not been dropped
has no place of its own for a local axis to be measured from, and the build
plane it snaps to is a world plane. The same reason the vertical drag keeps
driving the build plane rather than the element's own up.

**Which axes are live is shown on the gizmo**, in colour rather than only in
words: `X` red, `Y` green, `Z` blue — the standard convention — with the locked
arms dimmed. The toolbar combo names the mode in words a few pixels away; the
gizmo is where you are actually looking while you build.

### The overlay in the corner

Two rows, and only two:

|                 |                                                                    |
| --------------- | ------------------------------------------------------------------ |
| **Build plane** | the height new modules land on — `state.gridY`                     |
| **Current**     | what the next key will act on: the ghost `◆`, or the selection `■` |

Everything else it used to carry — rotation axis, scale axis, fly speed, drag
axis, lighting mode — is already on screen in the **toolbar**, whose combos _are_
those readouts. Repeating them a few hundred pixels away taught nothing and cost
a glance to decide which copy to trust.

These two are different. The build plane has no control of its own: it is
changed only by the numpad, and it is invisible until something lands at the
wrong height. Its old label was **Elevation**, which read as a property of the
selected element — it is not, which is why it sits at `0.00` for a ship built on
the floor. `Build plane` says what it is. The current element is derived from
three sources at once and has no control anywhere.

> Removing the speed row left the right-button+wheel fly-speed control with no
> feedback at all, so it now reports to the status line instead. A control you
> cannot see the effect of is one you will wonder about the next time the camera
> feels wrong.

### `X` — the axis gizmo

`X` draws one element's **world** X/Y/Z arrows; **`Shift+X`** draws its **own**.

Both are needed because the two things you do with an axis disagree about which
space they live in. A **drag** moves along the world axes by default — `node.position`
_is_ world position, since placements have no parent in the editor — so a world
gizmo is the honest answer for moving, unless the `World`/`Local` combo says
otherwise. **Scaling is always local**, so on anything that has been turned
(most of a ship built from a modular kit) a world gizmo cannot tell you which
way `X` will grow.

**The flavour survives clicking another element.** The gizmo already followed a
single pick — having asked to see it, you almost never want it left behind on
the piece you have moved away from — but re-showing it took the _default_, so a
local gizmo reverted to world on the next click and `Shift+X` had to be pressed
again for every element. It now carries whichever flavour it was in; `X` and
`Shift+X` still name one outright.

The local arms are aimed **individually**, from the world matrix's basis rows,
rather than by rotating the gizmo as a whole. A mirrored element — negative
scale — has no rotation that expresses it: its local `+X` genuinely points the
other way, and only a per-arm aim can show that. The rows are normalised first,
because the gizmo is a direction indicator and a 3× scaled element must not get
3× arrows.

> The basis is built by hand rather than with `Quaternion.FromLookDirectionLH`,
> which stores **`-forward`** in its Z row — a camera "back" convention that
> aims every arm 180° the wrong way. Measured, not assumed: two of the three
> arms happened to look plausible under a 90° Y turn, and only the `Y` arm gave
> it away.

With several elements selected, the one **nearest the cursor** gets them.
Screen distance, not world distance: "the one closer to the mouse" is a question
about what you are looking at, and two elements equally close on screen can be
far apart in the ship. Pressing the same key on the same element hides the
gizmo; the _other_ key re-aims it in the other space rather than hiding it. It
also moves when you press `X` on a different element, and when you
**single-click** another one while it is up, since having asked to see the axes
you almost never want them left behind on the piece you moved away from.

**Nothing selected or hovered hides them.** Leaving the last element's gizmo up
would leave it hanging off something you are no longer working on, with no key
that clears it.

**Showing them sets the space you work in.** `X` switches `World`/`Local` to
`World` and `Shift+X` to `Local`, because asking to see an axis is nearly always
asking to work along it — you press `Shift+X` to find which way the element's own
X grows _because_ the next thing you do is slide it that way. `Y` still overrides
it afterwards, so the coupling costs a keypress in the rare case and saves one in
the common case.

> Only on the way _up_. Hiding a gizmo says nothing about which space you want,
> and a toggle that quietly changed the drag axis on the way out would be a
> genuinely surprising way to lose a placement.

**The armed ghost counts as an element.** You set a module's rotation and
mirroring _before_ dropping it, which is exactly when the axes are worth seeing
— and it was the one case `X` did not cover. The ghost lives in `interact.js`,
which imports `editor.js`, so it registers its node through `hooks` rather than
being imported back and closing a cycle; the gizmo then follows it around the
cursor for free, and goes when the ghost is cancelled.

It carries all three modal axis settings at once, so "what will the next key do
to this element" is one glance rather than three readouts:

| on the gizmo                     | means                                          | set by    |
| -------------------------------- | ---------------------------------------------- | --------- |
| bright arrow, dim = locked       | the axes a drag moves along                    | `V`       |
| curved arrow encircling it       | the axis a turn goes about                     | `Shift+R` |
| cube on the tip                  | the axis a scale acts on (all three for `all`) | `F`       |
| the chip at the origin           | the move step, in metres, or `free`            | `Shift+V` |
| the chip inside the curved arrow | the turn angle, in degrees, signed             | `Ctrl+R`  |
| a chip on each lit cube          | the scale step                                 | `Ctrl+F`  |

Each modal setting has exactly one marker, and each marker means exactly one
thing — `V` never touches the ring, `Shift+R` never touches the brightness. The
curved arrow sweeps three quarters of a turn rather than closing into a full
ring, so it reads as a direction of travel and not as a collar.

**The turn angle is a magnitude.** It was briefly _signed_ — `-90°` through
`-5°` sat alongside the positives — because a key only ever turned one way, so
turning back meant four presses of 90 or switching the axis and reasoning about
which sign that gave. The wheel carries the direction now: one way turns, the
other turns back. So the sign went back out of the list, and the arrow the gizmo
draws lost the mirroring it had grown to keep up with it.

**`free` is a fine step, not no step.** Both lists carry one — `0.5°` on `Rot`,
`0.01` on `Scale` — for dialling in a value with the wheel. It is deliberately
_not_ zero: `Move` can be switched off because a drag is a continuous gesture
that then goes unsnapped, but a wheel notch is discrete, so a step of zero would
simply do nothing. That is exactly why a literal "off" was taken off these two
lists earlier, and a test still fails if `0` reappears in either.

> The scale floor came down with it. `scaleCurrent` clamped the magnitude at
> 5 cm so the wheel could never take something down to nothing — harmless when
> the smallest step was 0.05, and a wall the moment `free` arrived at exactly a
> fifth of it. 5 cm is also thicker than a good deal of what the kit is made of:
> a collision shell fitted to its 7.5 mm walls could not be nudged at all, and a
> decal sits well under it. The floor is **1 cm** now, or one step if that is
> somehow finer. It exists only to stop a notch reaching zero; anything above
> that is the tool second-guessing you.

The markers keep full brightness even on a dimmed arm: the rotation and scale
axes are often not among the drag axes, and a dim marker would read as "off".

The chips are **DOM overlays**, not scene geometry, for the same reason the
marquee is one: text has to be crisp at any distance, and these are readouts
rather than parts of the ship. Each hides itself when its anchor projects behind
the camera — which would otherwise park it on the opposite side of the screen
from the thing it belongs to — **and when it falls outside the canvas**. The
scale chips hang off the arrow _tips_, which swing out of view at close range,
and `#viewport` does not clip: one was caught sitting on a palette tile.

Putting each _value_ on the marker that governs it is the point: the gizmo then
answers **how far**, not just **which way**. All three settings decide what the
next keystroke does, and all three were otherwise only legible in a toolbar combo
at the far edge of the screen.

The scale step gets **one chip per lit cube** rather than a single shared one:
with the axis set to `all` the three cubes are the statement that all three axes
grow, and a value on only one of them would read as "just this one".

Three implementation notes worth keeping:

- **The curved arrow's node sits at the arc's centre**, with the geometry built
  around the origin — not at the arm's root with the arc pushed out along `Z`.
  That is what makes `getAbsolutePosition()` the ring's own position, which the
  angle chip needs. Built the other way round, the chip landed on the gizmo
  origin and sat on top of the move-step chip (measured: 3 px apart, versus
  100 px now).
- The gizmo's **position** is re-read from the element every frame rather than
  parented to it. Parenting would inherit the element's scale — and a 3× element
  must not get 3× arrows — while re-deriving it each frame also covers drags,
  undo, the ghost following the cursor, and deletion with no event plumbing: if
  the element goes, the gizmo goes.
- Every part is marked **`alwaysSelectAsActiveMesh`**, and — the point —
  **nothing sets `doNotSyncBoundingInfo`**. That flag was on every part as a
  micro-optimisation, and it is exactly the one that stops a bounding box
  following its mesh's world matrix. Since the gizmo moves by being
  _re-positioned_ rather than re-parented, the boxes stayed wherever it was
  built: select an element 50 m out and the boxes trail 50 m behind the arrows
  (measured — the drift was exactly 50), and the frustum test then culls arms,
  arrowheads and turn arcs on their old position. It reads as the arrows being
  **clipped**, worse the closer you fly, and clicking away and back cures it
  because re-showing rebuilds the meshes. Fifteen tiny meshes are not worth
  culling at all, so they are never culled and their bounds are left honest.
- Its materials are `StandardMaterial`, not PBR, precisely because
  `applyViewportMode()` only touches materials that have an `unlit` property.
  A gizmo built from these can never pick up the editor's unlit mode or its
  emissive lift.

For the ghost, Y mode drives the **build plane** rather than the ghost's own
height. Everything already hangs off that one value — the grid draws there, the
ghost sits there, numpad `+`/`-` moves it — so raising the plane keeps all of it
in step, and flipping back to X/Z simply resumes at the new height instead of
dropping the ghost back to the floor.

> A build plane level with the camera makes X/Z tracking degenerate: the picking
> ray is parallel to the plane, so there is no intersection to follow. That is
> inherent to plane tracking, and the same reason a horizontal drag falls back
> to a screen-space mapping at eye level.

> **Why `V` and not `Q`.** Movement is matched on `e.code` (the _physical_ key)
> so `WASD` stays under the same fingers on any layout — on AZERTY that is
> `ZQSD`, which is exactly what a French keyboard expects. The letter shortcuts
> match on `e.key` (the _label_). Those two collide on precisely one key: the
> AZERTY key labelled `Q` sits where QWERTY has `A`, so it arrives as
> `code: "KeyA"`, is claimed as strafe-left, and never reaches the shortcut
> switch. `Q` is therefore unusable as a shortcut for AZERTY users. `V` occupies
> the same position and carries the same label on both layouts — and reads as
> "vertical". The test suite pins this down by firing both spellings.

**The drag anchors on the point you clicked, not the element's origin.** Kit
origins sit at the _base_, so a column or door frame grabbed near the top, from
a camera near the floor, needed a horizontal reference plane that was **behind
the camera** — `cursorOnPlane()` returns null for `t <= 0`, so there was no
reference and the piece silently refused to move.

That alone was not enough, because **a horizontal drag needs two different
mappings**:

- **plane** — intersect the cursor ray with the horizontal plane through the
  grabbed point. The grabbed point stays exactly under the cursor, which is what
  you want looking down at the floor.
- **screen** — move along the camera's own right/forward axes by the pixel
  delta, scaled to world units at the grabbed point's distance.

The plane mapping **inverts** the moment that plane is above the camera: looking
up at a column, higher on screen is _nearer_, so pushing the mouse forward pulls
the piece towards you. It also runs to infinity as the ray approaches parallel.
Neither is fixable by picking a different height — at eye level the horizontal
plane is simply edge-on to the view. So `anchorDrag()` chooses: plane when the
grabbed point is below the camera _and_ the ray is more than ~15° off the plane,
screen otherwise. The choice is made **once**, at the start of the gesture, so
it can never switch mid-drag and jump.

`updateDrag()` also re-acquires its reference on the first frame it becomes
usable, rather than giving up for the rest of the gesture.

**The camera is saved with the layout** — position and rotation, under `view` in
the manifest, restored by `loadLayout()`. It lives in `buildManifest()` and
deliberately **not** in `serialize()`: that feeds the undo stack, and undoing an
edit must not also throw the view somewhere else. Manifests written before this
simply have no `view` and are left alone.

**Selection and hover are mutually exclusive.** A selected element is never also
reported as hovered, even with the cursor on it — one element, one state, or the
outline colour and "what will this key act on?" end up disagreeing. The subtlety
is that a _selection change_ has to re-evaluate the hover in **both**
directions: selecting the element under the cursor drops the hover, and
deselecting has to bring it back straight away. Only clearing it leaves `Esc`
followed by a keypress doing nothing until you jiggle the mouse.

**Rectangle select** is a rubber band drawn with the left button. A drag that
starts on **empty space** is always a rectangle — there is nothing else it could
mean, since the left button no longer moves the camera. Starting _on a module_
is the ambiguous case, and that is all the **Rect select** toggle is for:
with it on, a drag bands instead of picking the module up.

Hit-testing projects the corners of each element's **oriented** box — each mesh's
own box, transformed — and takes the convex hull of those points. Anything whose
**outline** meets the rectangle is selected, occluded or not, because "what is
inside this rectangle" is a screen question, not a visibility one. Corners behind
the camera are dropped: they project to a mirrored point that would otherwise
stretch the hull across the whole viewport.

> **Neither approximation could stay.** It used to take the screen-space
> _extent_ of the corners — an axis-aligned rectangle — and test that. For a
> slab lying diagonally across the view that rectangle covers the viewport
> corner to corner and is nearly all empty air, so a small band dropped in a gap
> selected everything around it: one drawn in clear air beside a corner hull
> picked up all of its boxes.
>
> And the corners came from the **world AABB**, so a collision box turned to
> follow a curve reported the upright box containing it. `vectorsWorld` is the
> box's own eight corners, which for a box collider is the shape exactly.
> Together they took the tested area down to **34% of what it was** on a turned
> hull, and a band drawn in clear air now catches nothing.
>
> Overlap, not containment: a band that merely clips something still takes it.
> On a bench full of hulls that do overlap on screen it means a tight band round
> one box may still catch its neighbours — but a band is a coarse tool, and
> having to _enclose_ a long wall to catch it is the worse trade.

**Markers band like anything else.** The player spawn and the doors are elements
you select, drag and delete exactly like a wall, so a rectangle drawn round one
has to catch it — they just live in a different map. Anything currently not on
screen is skipped, whether it was isolated away or parked with `H`: a rubber
band selects what you can see.

**Reaching the panels behind a portal.** A portal is a flat quad sitting exactly
where the door panels are, and it turns to face the camera, so unlike ordinary
occlusion you cannot get round it by flying — which makes selecting the panels
to register them as leaves needlessly fiddly.

`Shift` used to drop the portals from the pick, which cost it its usual job of
adding to the selection. **`Shift+H` does it now**: a ghosted element is
unpickable, so ghosting the door lets hover, click, drag and the rectangle all
fall straight through to what is behind it. One mechanism instead of two, and it
works on a wall in the way just as well as on a door — which the portal-only
version never did. `Shift` has its old job back, and adds to the selection
alongside `Ctrl`.

The band itself is a DOM overlay rather than scene geometry — it has to be crisp
at exactly one pixel and must never be pickable, and a `div` is both for free.

**The cursor is hidden while the right button is held**, the same as while
placing: you are steering the camera, and a pointer sitting in the middle of the
view is just noise. The ghost still owns the cursor if one is armed, so
releasing the button does not reveal it mid-placement.

**Hovering is suppressed while the right button is held.** Looking around sweeps
the cursor across the whole scene, so the outline would strobe over everything
it passed and the "current element" would keep changing under you. The pick is
skipped entirely for the duration — it was wasted work on every frame of a look
— and re-evaluated on release, again so it comes back without needing a further
mouse move.

**No context menu anywhere in the editor.** The right button is a camera
control, so the menu its release raises is never wanted. Guarding the canvas was
never enough: the measured order is `pointerdown@canvas`, `pointerup@canvas` —
the canvas holds pointer capture, so it gets the release wherever the cursor is
— and then `contextmenu` aimed at whatever is _actually_ under the cursor, which
capture does **not** redirect. A look that drifted onto the palette therefore
ended on a thumbnail's "Save image as". Because the menu arrives after the
release, a "right button still down" flag cannot catch it either.

So it is suppressed outright, in a capture-phase handler on `document`. It is
registered at module scope rather than with the scene because the **loading
overlay is up before there is a scene at all** — and that was the one place a
menu could still be raised. `Ctrl+C`/`Ctrl+V` still work in the text fields;
only the menu is gone.

The two buttons are split cleanly, and that is the whole point: **left is
editing, right is the camera**, with no overlap. Left-drag can only ever select,
place or move a module; right-drag can only ever move the view. There is no
gesture that could plausibly do the wrong one of the two.

The right button does double duty within that half: **dragged** it looks around,
and **released on the spot** it cancels. Babylon's mouse input claims all three
buttons for looking by default, so it is restricted to `[2]`.

Crucially it is _not_ a modifier. Turning (right-drag) and moving (`WASD`) are
disjoint bindings, so both run at the same time — turn the view with the mouse
while walking with the keys, exactly as in a game. An earlier version made the
right button turn `WASD` into look controls; that overlapped with right-drag
doing the same thing and has been removed.

### Walk mode

**Walk** puts the camera at the player's eye height (`EYE_HEIGHT = 1.8 m`) and
pins it to whatever floor is underfoot. A free camera flatters a level: corridor
heights, sight lines through a door and how much of a room is actually visible
from the floor are all things you cannot judge from above. Speed is unchanged —
this is a viewpoint, not a speed limit — and `Space`/`C` are ignored, since they
would only fight the grounding.

The ground ray needs **two different searches**, and getting this wrong made the
toggle look like it did nothing on a real ship:

- **While walking**, the ray starts at the feet plus `STEP_UP` (0.6 m) and goes
  down. Starting above the eye instead would find the **ceiling** of the room
  you are standing in and stand you on top of it, and the 0.6 m margin is what
  lets you step onto a low platform while a 2 m ledge stays a wall. A miss here
  means you have stepped over a hole, so the height is simply kept — a ship
  under construction is full of them and falling out of the world every time
  would be useless.
- **On entering**, that same search fails whenever the camera happens to be
  parked over a gap or below a floor, and the toggle then silently did nothing.
  Most of a half-built ship's bounding box is empty space, so this was the
  normal case, not an edge case. Entering therefore searches from above the
  whole world and falls back to the build plane, so it always lands you at eye
  height somewhere sensible.

The forward vector is **flattened** while walking, or looking at your feet and
pressing `W` would drive the camera into the floor and the grounding would fight
it every frame. Vertical inertia is zeroed each frame for the same reason.
Grounding also runs while standing still, since the floor can be deleted or
dragged out from under you.

Panning was removed rather than rebound. It only ever existed on right-drag, it
was the gesture most often fired by accident, and `WASD` + `Space`/`C` already
translate the camera in every direction.

**Drop to plane** rests the selection on the build plane, wherever that
currently is — so raise the plane and it becomes "rest this on the ceiling".

An edit applies to _every_ selected element, each turning about its own
origin rather than a shared pivot, which is what a row of props usually wants.

**Rotation composes quaternions; it does not accumulate Euler angles.** Reading
the Euler triple, adding a step and writing it back looks equivalent and is not.
`toEulerAngles()` decomposes **YXZ** with X clamped to ±90°, so the moment an
element is yawed, stepping the X component walks into gimbal lock — from a 90°
yaw, four 90° X steps gave `(90,90,0) → (0,-90,180) → (90,-270,0) → (0,-90,180)`,
oscillating instead of coming back round. Y and Z happened to survive it, which
is exactly why the bug hid for so long. `spinNode()` composes
`Quaternion.RotationAxis(worldAxis, rad).multiply(current)` instead. **Order
matters**: Babylon's `a.multiply(b)` applies `b` first, so the increment goes on
the _left_ to act in world space — the other order gives a local-axis turn.
Euler is now produced only at the boundary, when a placement is written to the
manifest.

**`Rot` and `Scale` have no "off".** They are keyboard _step sizes_, and a step
of nothing is meaningless — the old `off` option silently fell back to a hidden
default (15° / 0.05) rather than doing anything. Only `Move` has a real "off",
which means free positioning while dragging.

**A restore is not itself an edit.** `deserialize()` sets a `restoring` flag and
`pushUndo()` no-ops while it is set. Without that, a single stray `pushUndo()`
anywhere in the restore path does two invisible kinds of damage: it **clears the
redo stack**, and it pushes a _half-restored_ snapshot onto the undo stack.

**The runtime lighting is on the stack; the camera and the editor's own view are
not.** All of it was off the stack at first, on the same reasoning — "not an
edit to the ship". Only part of that held up. The **Runtime** rig's `Env`,
`Dynamic Env`, `Exposure` and `Tone` are _authored_ values: the manifest carries
them and the runtime reads them, so a lighting change you cannot take back is a
real edit lost. Where the camera happens to be standing is genuinely not, and
neither is how bright you like the ship while you build it.

<a id="lighting-is-an-edit-the-editors-own-view-is-not"></a>

So the snapshot carries `lightSets.runtime` and nothing else. The **Editor**
rig is a per-browser view preference, mirrored into `localStorage` rather than
the ship: dragging its slider pushes no entry, so restoring it on undo would
take back a change no entry ever recorded — you would move a wall, undo, and
watch the room's brightness jump for no reason you could name.

The runtime rig travels whether or not the viewport is rendering it. The view
mode only decides which rig is on screen, so restoring the visible one alone
would leave the other behind, to surface later as a value nothing ever put back.

**One entry per slider gesture.** A range fires `input` continuously while it is
dragged, so one sweep of `Env` would otherwise bury the stack in near-identical
snapshots. The push is armed on `pointerdown` (and on the first key or wheel)
and spent on the first change — the same shape as the inspector fields, but
without a focus event to hang it on, because a range keeps focus between drags.

**Load asks before discarding unsaved work.** It throws away the whole scene and
sits one button away from Save; nothing else in the tool destroys that much in a
single click. **Closing or reloading the tab gets the same guard**, via
`beforeunload` — the browser owns the wording there (custom text has been
ignored since 2016), so all the tool chooses is _whether_ to ask. Chrome also
requires the page to have been interacted with first, which is the behaviour we
want anyway: a tab you only looked at closes silently.

"Unsaved" is decided by comparing against a snapshot taken at the last save,
load or boot — not by a dirty _flag_, so undoing back to the saved state
correctly counts as clean again. `hidden` is stripped from that comparison: it
never reaches the manifest, so it can never be saved, and leaving it in would
make hiding one wall enough to prompt for the rest of the session.

That is not hypothetical — restoring a spawn marker did exactly this, because
`deserializeMarkers()` passed `silent: true` for doors but `setSpawn()` had no
such option and always pushed. Since `deserialize()` runs on every undo _and_
redo, the symptoms were: `Ctrl+Y` did nothing after an undo, and a second
`Ctrl+Z` lost the player start — but only on layouts that actually had a spawn,
which is why it survived the tests for so long. That path now takes `silent`
like the others, and the flag makes the whole class of bug unreachable.

**One undo entry per visit to an inspector field, not per keystroke.** The
number fields fire `input` on every character, so typing `12.5` pushed four
whole-layout snapshots and needed four `Ctrl+Z` to take back — and a stack that
short empties fast at that rate, quietly dropping real edits off the bottom. A
flag armed on `focus` and spent on the first edit reduces a whole editing
gesture to one entry. Blur/refocus starts a new one, which is the right grain:
that is exactly when you have decided the value is settled.

**History is capped by memory, not by a step count.** It was 80 snapshots,
which sounds cautious until you measure one: 13.5 KB on the real ship, 145 bytes
an instance, so the whole stack held **1.1 MB** — beside the 11 MB of glb
already in the page, that is not a limit worth having, and it was low enough
that the per-keystroke bug above could push real edits out of reach. But a count
generous enough for this ship is reckless for one ten times the size, so the
budget is what is capped (32 M chars ≈ 64 MB, since JS strings are UTF-16) with
a 1000-entry rail against a near-empty layout keeping half a million of them. In
practice: this ship gets all 1000 steps, a 1000-instance ship ~230, a
5000-instance ship ~45 — deep where it is cheap, bounded where it is not. One
snapshot is always kept even if it alone busts the budget, or undo would
silently do nothing on exactly the ships that need it most.

**The inspector never rewrites the field you are typing in.** Editing a field
applies the value and re-selects, which refreshes the whole inspector — and
rewriting the focused field with the parsed number destroys whatever is
half-typed. `"1."` parses as `1` and comes straight back as `"1"`, so the
decimal point vanished as fast as you typed it; `"0.05"` lost its trailing zero
the same way. `setField()` skips `document.activeElement`. This is the same trap
the uniform-scale mirroring already dodged for a leading `-`, generalised.

**Negative scale mirrors an element** — useful for turning a left-hand cornerpiece into a right-hand one. `F` does it on the current Scale axis, applying to
the ghost you are placing, the elements you are dragging, or the selection. The
wheel then resizes the _magnitude_ and leaves the sign alone. A mirrored node
has a negative transform determinant, which glTF explicitly allows: the winding
order flips with it, and both Babylon and the exporter honour that.

Scale axis `all` would make `F` a point inversion rather than a mirror, so it is
treated as X — the horizontal flip a modular kit almost always wants.

### Building a ceiling

Set the Move step to the height you want, then numpad `+` until the overlay
reads the right build-plane height, and build normally — the ghost sits on the plane, not on
the floor. Faster still: hover a wall you already placed and press numpad `.`,
which puts the plane exactly on top of it.

Grabbing an element also moves the build plane to that element's height, so
picking up a ceiling piece keeps you working at ceiling level instead of
dropping the next one to the floor.

### `B` fetches what you cannot find

Arming a module does not choose a position. The ghost is simply wherever the
cursor ray crosses the build plane, and from inside a finished room that is
routinely nowhere useful: with the plane still down on the deck you started
from, looking level or up misses it entirely — `cursorOnPlane` returns `null`
for anything at or above the eye, so the ghost stays wherever it last was —
while looking barely down puts it hundreds of metres away. Neither can be
dragged back into view, because it is not in view to drag. `B` fetches it.

Two rays decide where it lands. **Forward** first, so the spot is never pushed
through the wall you are facing — the preferred stand-back distance would
otherwise drop the piece in the next room every time you built against
something. Then **straight down** from there, so it rests on the deck you are
standing on rather than floating at eye height. With neither — out in the open,
or over a gap — the raw point in front of the camera is still somewhere you can
see, which is all that was asked for. How far in front scales with the piece's
own size, clamped to 3–8 m: a hull section needs room, a handrail wants to be
within reach.

> **It moves the build plane too, and that is the half that matters.** A
> one-shot teleport would be undone by the very next mouse move, which
> re-derives the ghost's position from the plane — the piece would spring
> straight back to the far side of the map and the shortcut would look broken.
> The same applies to a placed selection: leaving the plane behind means a
> following `M` grab drops it back to the old height.

It is deliberately not a _frame_ — the camera does not move. Framing answers
"where is it", which double-click already does; `B` answers "bring it here",
which is what you want when the answer to "where is it" is 300 m away in a
direction you were never going to fly.

## Chunks, doors and portals

Chunks are what the portal renderer streams and culls. Pick the active chunk in
the toolbar; new placements join it. **Assign** moves the current selection to
it, **Chunks…** opens the pane that owns everything else about them, and the
**Chunk** button itself toggles isolation — pressed, everything outside the
active chunk is hidden, and it turns orange so the view being partial is never a
mystery. Switching the dropdown while isolated follows the new chunk.

### The Chunks pane

`+` and `Rename` used to sit in the toolbar as two `prompt()` boxes. They are
now one **Chunks…** pane, because a chunk is not just a name: it is the unit the
portal renderer streams, the thing placements carry and the thing door markers
join, and a list you can see is the only way to keep those in step. The pane
lists every chunk, says what each one holds, and offers:

- **New** and **Apply** — add and rename. Renaming rewrites every reference,
  not just the list entry.
- **Delete** — which **refuses while anything still refers to the chunk**, and
  says what. The alternatives were to drag the contents into some other room,
  which silently rewrites the ship's layout to service a button press, or to
  delete them with it, which turns one keystroke into unbounded loss. Emptying
  the room first is a deliberate act, and **Assign** already exists to do it.
  The last remaining chunk cannot go either: `activeChunk` is what new
  placements join, so there has to be one.

> Rooms used to carry per-room render settings here — an atlas size and a sample
> count for the offline lightmap bake. Lighting is now captured live from
> environment probes, which are **boxes of their own** rather than a property of
> a chunk (see [Environment probes](#environment-probes--the-ship-reflects-its-own-rooms)), so a chunk is back to
> being a name and a membership list.

**Renaming a chunk rewrites every reference**, not just the list entry. A chunk
id is not a label: placements carry it, door markers name the two chunks they
join, and the active chunk is one of them. Renaming the entry alone would orphan
every placement and quietly break the portals. Renaming onto an existing id, or
to nothing, is refused rather than half-applied.

**Elements can be named.** The `Name` field in the inspector is a label of your
own — "weapon locker", "airlock door left" — and it shows in the corner overlay in
place of the module id. Ids stay machine-generated and stable, because the manifest,
doors and portals all key off them; the name is stored alongside in the manifest
and is purely for finding things again.

Doors carry the portal:

- **Door** arms a marker you drop on the grid.
- **Door from sel** is the usual route — select the door geometry you already
  placed and it creates a marker centred on it, sized to its opening, with those
  placements registered as the animated leaves.
- Chunk A/B default to `(auto)`, which resolves to the two nearest chunk
  volumes at save time. Set them explicitly when that guess is wrong.
- **Chunk B also offers `Skybox (outer space)`** — a window through the hull
  rather than a doorway between rooms. See below.
- **Enabled** controls the portal visibility link. Disabling it keeps the door
  geometry and collision in place, but portal traversal no longer sees through
  it. It is written to both `doors[].enabled` and `portals[].enabled`, and
  defaults to `true` for older manifests.
- **Sealed** marks a portal you can see through but not walk through — a window
  onto space rather than a doorway. The renderer still draws the far chunk;
  collision generation keeps the opening solid. It is written on the door
  record only (`doors[].sealed`), not on the portal, and defaults to `false` so
  every manifest written before it reads back as an ordinary doorway.
- **Doors resize two ways.** The inspector's `W`/`H` fields set the authored
  opening; `Ctrl+wheel` and the inspector's `Scale` fields scale the node like
  any other element. Both were once blocked for markers — `scaleCurrent()`
  filtered them out and `applyInspector()` guarded the write — which just made
  doors feel broken. The exported `width`/`height` fold the node scale in
  (`width × |scale.x|`), because `portalOf()` reads the **world matrix** and so
  the portal grows with the node regardless; leaving the raw width in the
  manifest would have made a door disagree with its own portal.

Portals and the adjacency graph are derived from doors, so there is nothing
extra to keep in sync.

**A door onto space is a reserved chunk id, not a fourth boolean.** Chunk B
offers `Skybox (outer space)`, which writes `__SKYBOX__`. A window through the
hull is still a portal — the renderer needs the opening and its shape — but
there is no room behind it to draw. Expressing that as a _side_ means
everything that already reasons about a door's two sides keeps working
untouched: isolation still shows the door in the room it belongs to, validation
still sees both sides resolved, `portalOf()` still produces a portal record. A
`skybox: true` flag beside `sealed` would have needed every one of those places
taught about it.

The id buys that at the cost of two obligations, both enforced:

- **It cannot collide with a real room.** `addChunk` and `renameChunk` both
  refuse `__SKYBOX__`, so no chunk can ever be given that name by any route.
- **"Opens onto space" and "not sealed" cannot both be true.** There is nothing
  out there to walk into. Picking Skybox ticks **Sealed** and disables the box;
  `normalizeDoorSides()` settles the pair wherever a door is made or loaded, and
  `buildManifest` restates it (`sealed: !!m.sealed || b === SKYBOX_CHUNK`) so a
  hand-edited manifest cannot describe a window you could step out of.

Two smaller consequences fall out of it. Space is **not** listed in `chunks` —
it has no aabb and no contents — so the adjacency edge is deliberately one-way:
`CH_x → __SKYBOX__` is written, because a renderer standing in the room has to
know this opening leads outside, but nothing leads back. And the "no leaves
assigned" warning is suppressed for these doors: a hole in the hull has nothing
to slide, so the warning would be permanent noise.

**Isolation shows a door in both the rooms it joins.** A door is not _in_ a
chunk, so markers were exempt from isolation outright - which left a door
hanging in the middle of a room it has nothing to do with. Sides left on
`(auto)` are resolved by nearest chunk volume, the same rule the manifest uses,
so what isolation shows you is what will be written.

**`buildManifest()` forces world matrices first.** Babylon only refreshes them
at render time, so an export fired straight after a scale — a wheel notch, an
inspector keystroke — measured the ship as it was one frame ago. That was
invisible while every derived figure came from the same stale matrix, and became
a real disagreement the moment door width started coming from `scaling`
directly.

**Start positions are not markers.** They used to be — a `Player` and a `Weapon`
button dropping a `spawns` block — but they moved into behaviours: place a dummy
element, name it, and give it a `player_startingpos` (or `weapon_startingpos`)
behaviour. One mechanism for "this node means something to the runtime" is
better than two, and the behaviour body can carry whatever the runtime grows to
need. Loading a manifest that still has a `spawns` block **drops it and says
so** in the status bar rather than keeping it and quietly re-saving it.

Markers are editor-only: they are written to the manifest but excluded from the
exported .glb.

## Collision

Collision geometry is authored beside the ship, not derived from it at runtime.
It is stored in `state.colliders`, drawn in green, kept out of `ship.glb`, and
written to the manifest grouped by chunk in **Havok's own parameters** — so the
runtime hands each record straight to a shape constructor with no
interpretation.

| shape    | what the manifest carries                        | why                                                                                                        |
| -------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| box      | `centre`, `rotation` (quaternion), `halfExtents` | `PhysicsShapeBox` takes an explicit turn                                                                   |
| sphere   | `centre`, `radius`                               | no orientation exists                                                                                      |
| capsule  | `pointA`, `pointB`, `radius`                     | `PhysicsShapeCapsule`'s turn is implicit in the segment; the segment is a diameter shorter than the height |
| cylinder | `pointA`, `pointB`, `radius`                     | likewise, but the segment _is_ the height                                                                  |

**The scale is constrained per kind, on every write path.** Havok's sphere is a
single radius and its capsule a radius plus two endpoints, so an ellipsoid has
no representation at all: the runtime would have to silently resize it, and the
ship you built would not be the ship you play. `constrainScale()` therefore runs
on the inspector, the wheel, a carried duplicate and a loaded manifest alike —
a sphere is forced round, a capsule and cylinder locked to one radius in X/Z,
and only the box takes an arbitrary scale. A capsule additionally cannot be
shorter than it is wide: at `h == d` its two caps meet and it _is_ a sphere, and
Havok agrees — its capsule is a segment plus a radius, and there is no segment
of negative length.

### A capsule is drawn as a capsule

A box, a sphere and a cylinder are each a single unit mesh under a scale. A
capsule is not, and this is the one place the "unit shape sized by scaling" rule
does not hold: a capsule's caps are hemispheres of the _tube's_ radius, so
stretching one unit mesh in Y stretches the caps with it into an ellipsoid.

It was worse than that. The unit mesh was `CreateCapsule({height: 1, radius:
0.5})`, and Babylon's capsule `height` **includes** the caps —
`heightMinusCaps = height − radiusTop − radiusBottom` — so at radius 0.5 there
was no tube left at all. Every capsule in the editor was a sphere pulled into a
lozenge, 141 mm off a true capsule's surface at 1 × 3 m, while Havok collided
with a proper pill.

The trick that keeps it to a single mesh: build the geometry at the right
**ratio** — radius 0.5, total height `h/d` — and counter-scale the mesh's own Y
by `d/h`. The two cancel, so the mesh ends up wearing a _uniform_ world scale of
`d`: the caps come out as true hemispheres, the normals stay correct without an
inverse-transpose, and the geometry only has to be rebuilt when the proportions
change rather than every time the shape is resized.

> **Watching the result again.** A collider's scale changes from the gizmo, the
> inspector, the wheel, a carried duplicate, a fit, an undo and a load. One
> `onBeforeRenderObservable` comparing `h/d` against what each capsule was last
> built at catches all seven — and the ghost, whose size lives on the ghost root
> rather than on the shape's own parent, which is why the ratio is read off the
> parent's **world matrix** and not its local scaling.

**The size always goes on a holder, never on the shape mesh.** That is what
makes the counter-scale possible, and the ship's inherited-hull preview was the
one place that did not do it: it wrote the composed transform straight onto the
mesh, which looked equivalent to what a real collider does and flattened the
counter-scale, so every capsule out on the ship was drawn as a **sphere** while
the same hull on the bench was a proper pill. The preview builds a
`TransformNode` per shape now, exactly as `addCollider` does.

> The preview also runs its composed scale through `constrainScale`. A placement
> may be scaled unevenly — this ship has doors at `[0.65, 0.85, 1]` — and a
> capsule or a sphere composed with that is an ellipsoid, which Havok has no
> shape for at all. Drawing the ellipsoid would promise something the runtime
> cannot deliver, so the preview shows the shape Havok actually gets: one
> radius, averaged from X and Z, the same rule the authoring path applies.
> `capsuleRatio` averages the two the same way, so nothing downstream can be
> handed a width that depends on which way you look at it.

**A capsule's height is the whole pill, caps included** — the same reading as a
box's side or a sphere's diameter, and what the editor draws and the inspector
says. Havok's capsule is a segment _grown by the radius in every direction_, so
`shapeRecord()` writes a segment one diameter shorter than the height. A
cylinder has flat ends and keeps its full height. Getting that wrong makes every
capsule a diameter taller in play than it looks in the editor — invisible in the
tool and baffling in the game.

Because a capsule at scale 1 is one metre wide and one metre tall — which is a
sphere — the Collision pane arms it at 1 × 2 m, so the brush looks like the
thing it places.

### Primitives land corner first

A kit module is modelled _from_ its origin: a wall's geometry starts at the
origin and runs 4 m to the side, so dropping it with the origin on the build
plane leaves it resting on the floor and filling whole grid cells. A collision
primitive is a **unit shape centred on its origin**, because Havok wants a
centre and not a corner — so the identical drop buried half of it under the
plane and put its faces through the middle of a cell.

The ghost therefore shifts a lone primitive by its own half size, putting the
_corner_ where the origin was snapped to. The lift is derived from the live
transforms rather than a constant, so it survives the wheel scaling the ghost
and `R` turning it, and it uses the world-axis extents (`|m₀|+|m₄|+|m₈|` and so
on) so a turned box reports the box that actually contains it.

The build plane consequently means a primitive's **base**, not its origin. That
is not cosmetic: referencing the centre made every `M` grab set the plane to the
centre and then drop the shape half a height higher, so a box climbed off the
floor a little more each time it was moved.

Only a lone primitive gets the lift. For a set, the offsets are measured from an
anchor whose own half size is not the group's, and the two references would
fight.

### What the primitives are for

The four buttons in the Collision pane do two different jobs, depending on
where you are.

**On the staging area** they are how you author a hull that is not a single
box: a capsule for a barrel, a cylinder for a pipe, several boxes leaving a
doorway open. **Fit a box** is only a starting point — it fits the module's
bounding volume, which is exactly the shape authoring is meant to improve on.
**Fit a hull** goes further and reads the actual triangles; see below.
Whatever you drop is claimed by the staged element it sits on and becomes that
module's collision.

> **The authored convention is a face, not a box round the mesh.** A hull's
> _surface_ is put flush with the visible surface the player meets, and the body
> extends **away** from the play space. Floors show it most plainly: every
> `Platforms/*` hull is `position.y = -0.25`, `scale.y = 0.5` — a half-metre
> slab hanging entirely _below_ the walking surface, so you stand exactly on the
> plate. Walls do the same horizontally: `WallWideBand_Straight` has a 108 mm
> panel and a **1 m** hull whose inner face is on the grid line and whose body
> runs outward into the wall's thickness.
>
> So a hull that sits ~0.45 m off its mesh centre, or juts 0.9 m past its mesh
> AABB, is not stale — it is that convention, and 30 of the 49 authored modules
> show it. Measured on the storage room, whose four `WallWideBand_*` walls give
> an interior of X 4→12, Z −4→4 — 8 × 8 m, exactly as built: **all seven hulls
> lie entirely outside that interior**, so the player is stopped flush with the
> visible wall and the overhang is in the void behind it. A thick backstop is
> also what stops a fast body tunnelling through a 7.5 mm panel.

**On the ship** they create _room_ colliders: world-space one-offs belonging to
a chunk, for things with no module behind them — an invisible barrier, a
blocker over a gap. They are written to `collision[chunk]` as authored.

There used to be a third path: **Generate for active chunk**, which fitted a
box to every solid module in a room, skipped decals, and cut unsealed doorways
out of the walls they crossed. It is gone, along with the ~150 lines behind it.
Every module's collision is now authored once and inherited by every placement,
which is both better geometry and less of it — and a door frame's opening is
simply left out when its hull is authored, rather than subtracted afterwards
from a slab that should never have been solid. `sealed` keeps its runtime
meaning on the door record.

For a fluid or a player, **over-approximating is safe and gaps leak** —
overlapping shapes are harmless, a hole is not. Err on the generous side.

### The collision staging area

A room's primitives are world space and belong to that room. Collision on a
**kit module** is authored once, in the module's own local space, and every
placement of it inherits it — the only sane answer for props, whose bounding box
is a poor fit and which are placed many times over.

**Edit collision** opens a staging area. It is a _mode_, not a property of the
selection: it starts empty, and you stage whatever modules you want to work on.
Clicking a palette tile arms a ghost you place yourself; clicking one already
there focuses it rather than adding a second, because a second instance would
give the association rule below two equally good answers.

Staged elements are real placements carrying `stage: true`. That is what makes
every existing tool work on them unchanged — selection, `X`, `H`, `Del`,
`Ctrl+D`, dragging, the marquee, the inspector, undo. They are filtered out at
the two places that walk _every_ placement (the manifest's instance list and the
.glb exporter, both via `shipPlacements()`), and their chunk id is never in
`state.chunks`, so nothing else can reach them either.

**Which element a shape belongs to is decided by where it sits**, not by a tag:
a shape is owned by the staged element whose bounding box, grown by
`ASSOCIATION_MARGIN` (0.5 m), it overlaps most. That is what makes copying work
— `Ctrl+D` a hull off one barrel, drag it onto a similar one, and it simply
becomes that one's. It is only unambiguous because the stage lays elements out
with **more than twice that margin** between them, so no two grown boxes can
ever meet; the largest-overlap rule is the belt to that braces.

A shape that overlaps nothing belongs to nothing. Rather than being dropped
quietly it is **counted in the banner**, which says how many will be lost.

**Removing a staged element keeps what was fitted to it.** `Del` goes through
`unstageModule()`, which reads the area back into the per-module record first —
so staging that module again brings its shapes straight back. Shapes are stored
_relative to the element_, so it does not matter where on the area it lands next
time.

**A module dropped on the bench is a one-shot.** On the ship a palette tile
stays armed, so a row of panels is just repeated clicks. Here it is the
opposite: a module comes to the bench once, to have a hull fitted to it, and
staging the same one twice is refused anyway — so staying armed only ever left
a second stand-in on the cursor to be dismissed. The tile goes out and the hint
clears with it, which is the visible difference between the two modes.

> **Collision primitives still repeat.** A hull genuinely is a run of boxes
> along a wall, so the box, sphere, capsule and cylinder ghosts stay armed. The
> rule is about _modules_, not about the bench.

**`Ctrl+D` there copies shapes, never the module** — for the same reason: a
module may only be on the bench once. The rule lives in `grabSelection()` rather
than only at the key, which used to filter a list it then did not pass on, so a
module _selected alongside a shape_ was copied anyway. `M` still picks a staged
module up to move it; it is copying that makes no sense, not carrying.

**Fit a box** acts on the selection: exactly one element, and not a collision
shape. Anything else says so plainly rather than guessing — "fit the current
module" stopped meaning anything the moment the area could hold more than one.
The **Hull offset** setting below applies to it too: the collision shell can
leave its box a good deal thicker than the art, and the offset decides which
side of the art that extra thickness goes.

**Fit a hull** is the same rules with a better answer: it reads the element's
own triangles instead of its bounding box, and lays out as many boxes as the
shape asks for. Three candidates are fitted and the best-scoring one wins —

| candidate | what it is                                                   | what it suits                  |
| --------- | ------------------------------------------------------------ | ------------------------------ |
| `box`     | the whole module in one box                                  | a crate, a floor plate, a door |
| `slabs`   | a slab per surface patch, laid on it and extending backwards | a wall with a lip              |
| `split`   | cut in half and recurse until each piece is worth boxing     | a corner, a door frame         |

The score is **coverage × solidity, less a small penalty per box**, where
solidity is the share of the hull's volume that sits near real surface.
Coverage alone is a bad judge: one enormous box covers every vertex and fills
the room the player is meant to walk in. Letting the score choose means no
single heuristic has to be right about which kind of module it is looking at.

Three settings on the Settings pane govern it.

**Hull tolerance** is how far the hull may stray from the art before that
counts as wrong, in metres, and it is deliberately the _only_ dial for how
finely a shape is approximated. It is the resolution the hull is judged at, so
tightening it fails the coarse candidates and a finer one has to take over — a
rounded platform is worth one box at 25 cm and eight at 4 cm, while a crate
stays a single box whatever you set. Fine values cost noticeably more time
(tens of milliseconds at 25 cm, a second or so at 2.5 cm on a big module).

A separate "how many boxes" setting would have to be kept in step with it, and
the two would disagree. The choice is made in two questions, in order: is the
hull good enough — does it contain the art, and is it within the tolerance of
it nearly everywhere — and of the ones that are, which is the smallest? That
ordering is what stops a tighter tolerance ever handing back a _bulkier_ hull
than the setting before it.

> **Volume is the honest measure of a collision hull.** Every cubic metre of
> it that is not art is somewhere the player cannot stand. Ranking on how
> snugly a hull fits its art instead looks reasonable and is not: it happily
> picks twenty overlapping boxes over three good ones.

**Hull thickness** is the depth a hull is given along its thinnest axis. The
kit's walls are millimetres thick and its floors are single planes with no
depth at all; a collider that thin is something a fast-moving body goes
straight through. It is a _minimum_ — a crate is already thicker and is left
alone.

**Hull offset** decides which side of the art that depth goes.

|            | what it does                                                              |
| ---------- | ------------------------------------------------------------------------- |
| `centered` | splits the depth either side of the art                                   |
| `negative` | tucks the hull behind the visible surface — the convention this ship uses |
| `positive` | stands it in front                                                        |

The axis is oriented to the surface normal, so it points out of the solid and
the two names mean the same thing on every module rather than depending on how
the box happened to be fitted. Either way **the art stays inside its hull**:
the face is placed a millimetre clear of the visible surface, and clamped so
that when the hull is no thicker than the art — where there is no slack to
give — that millimetre is not taken out of the other side.

> **The offset names where a face goes; it does not nudge the box.** A slab is
> fitted lying against its own surface already, so pushing it "by the padding"
> pushes it a second time and the art comes out of the back. Naming the face is
> the same answer whatever the box started as, and asking twice changes nothing.

The `slabs` pass is the one that reads the kit's own convention: triangle
normals point _out_ of the solid, so laying a slab on a patch and extending it
backwards puts the body away from the play space for free — without needing to
know which side the room is on. Its depth is **capped**; using the patch's own
extent instead let a curved patch, whose points wrap right round an arc, grow a
slab that swallowed everything the arc enclosed (`WallAstra_Corner_Round_Outer`
came out at 87 m³ against a hand-drawn 12). Its bin angle is _derived from the
tolerance_ rather than fixed: leaving it constant was what stopped a tighter
tolerance from ever improving a rounded corner, because the slab pass kept
winning with the same coarse three.

The `split` pass cuts **axis-aligned** and offers _every depth_ as a candidate.
Both of those are deliberate. Cutting square to the world means the pieces nest
inside their parent, so an extra level can only shrink the hull; and stopping on
a threshold meant the threshold was sometimes met one cut too early, so handing
the whole ladder to the score lets it pick.

The _box drawn round_ a piece may still be turned, though, and each level takes
whichever of the two is smaller **by total volume**. Turned boxes are far
tighter on a curve — the whole reason a rounded corner wants a hull that follows
it — but neighbouring pieces then overlap, and summed volume counts an overlap
twice, so a turned set is charged for exactly what it wastes and only wins when
it really is the smaller hull.

> Choosing per _level_ rather than per piece is what makes that safe. Choosing
> per piece minimises each piece and lets the overlap between them run free,
> which is how an earlier version made a hull _bulkier_ the more finely it was
> cut.

This is what fixed the rounded corners with a lot of surface detail.
`TopCables_Corner_Round_Outer` carries 3804 triangles of cable, which drowns the
slab pass in tiny normal bins, so the split was the only candidate left — and
while it was square-only that meant an axis-aligned box round an arc:

| module                               | before                        | after                      |
| ------------------------------------ | ----------------------------- | -------------------------- |
| `TopAstra_Corner_Round_Inner`        | 16 square, 13.1 m³, 58% solid | **4 turned, 4.8 m³, 100%** |
| `TopCables_Corner_Round_Outer`       | 16 square, 10.4 m³, 63% solid | **4 turned, 7.3 m³, 100%** |
| `TopCables_Corner_Round_Inner`       | 16 square, 10.6 m³, 63% solid | **4 turned, 7.6 m³, 100%** |
| `ShortWall_WhitePlate2_Corner_Inner` | 24 square, 10.8 m³            | **2 turned, 4.3 m³**       |

> **Solidity judges the shape, volume judges the price.** Solidity is measured
> _before_ the thickness is added. Measuring after it has a thin floor plate,
> padded to a walkable depth, fail its own test — most of that hull is
> deliberately not near the art — and the fitter chops the plate up trying to
> fix it.

> **A hollow shell is solid inside.** Whether a sample is in the art is decided
> by ray parity, not by proximity to a triangle: a crate is a hollow mesh, so
> every point in the middle of it is far from any surface, and judging on
> proximity alone marks a hull that fills the crate as mostly empty air. The
> crossings along each ray are _paired_, and an odd one left over is dropped —
> the kit is full of open shapes, and plain parity would mark everything beyond
> a single-plane floor as solid.

> **It declines rather than guess.** A hull that covers ≥ 95%, sits ≥ 85% on
> surface at the tolerance and needed ≤ 4 boxes is reported plainly; anything
> else is fitted but flagged _"worth checking by eye"_. In practice that is the
> curved corners, which are genuinely better drawn by hand. Saying so is more
> use than a hull that looks plausible and leaks.

Measured against the 53 hand-authored hulls (`fit-boxes.mjs`, geometry dumped by
`fit-dump.mjs`, sweep by `fit-sweep.mjs`) at the default 10 cm tolerance and
35 cm thickness:

|               | coverage  | boxes | volume | on surface |
| ------------- | --------- | ----- | ------ | ---------- |
| fitted        | **99.7%** | 3.8   | 7.5 m³ | **89.8%**  |
| hand-authored | 76.8%     | 1.5   | 6.4 m³ | 68.3%      |

It matches or beats the authored hull on 60 of 61 placed modules, and is
confident about 44 of them — those at 100% coverage and 97% solidity. Loosening
the tolerance to 15 cm trades fidelity for tidiness (3.2 boxes, 6.8 m³, 49
confident); tightening to 6 cm does the reverse (4.5 boxes, 39 confident).

Running it also turned up two real faults in the ship: `P0209` is a
`Door_Frame_Square_Blocked` with **no hull at all** (the player walks through a
blocked door), and the `Platform_Round1` / `Platform_Round2` hulls cover only
**6.7% / 9.1%** — a square box on a round plate.

> **Rotations cross the boundary as basis vectors, not Euler angles.** The
> fitter is plain geometry with no Babylon in it, and converting by hand into
> Babylon's YXZ convention is easy to get subtly wrong; `fitHullToSelection`
> composes the frame into a matrix and lets `decompose`/`toEulerAngles` do it.

> The fitter lives in `public/js/hullfit.js`, imported by _both_ the editor and
> the offline harness, so the thresholds baked into it are the ones the harness
> measured.

**A neural network was considered and rejected.** 53 examples — 42 of them a
single box — is far too few to train on, and the geometry fully determines a
good answer, so the problem is searched rather than learned.

The area is read back into the record on every change that matters — leaving it,
saving, or removing an element — so there is no separate commit step and nothing
to forget.

`Esc` closes the area once there is nothing in your hand. **The right button
does not**: it cancels an armed shape exactly like `Esc`, but never closes the
bench. RMB is also the camera button, and tearing down what you were working on
is far too much to hang off a button you press to look around.

**The bench keeps what you left on it.** Closing the area records which modules
were on it and where, and re-opening puts them back. Coming back to a blank
stage after stepping out to look at the ship was the wrong default: this is a
workbench, not a dialog. The roster rides in the collision file, so it survives
a reload as well as a trip back to the ship.

**Each side keeps its own viewpoint.** Coming back to the ship pointing at a
barrel, or to the bench pointing across the ship, meant finding your bearings
again on every switch. The bench's viewpoint rides in the collision
file as well, so a reload puts you back where you were working.

> Both sides store the camera as the camera itself holds it — **position and
> rotation**, through the same `serializeView` / `applyView` used by the layout
> — not a position and a target. `getTarget()` on a `FreeCamera` reports the
> last point something _explicitly_ aimed it at, and free look never updates
> that. The first version captured that stale target and then aimed at it, so
> the camera returned to the right place looking the wrong way, which reads as
> "the view was not restored" — because it wasn't. The test missed it for the
> same reason: it drove the camera with `setTarget`, the one gesture that keeps
> the target honest. It now aims by rotation, the way a user does, and checks
> the rotation comes back too. `stageView` in the collision file therefore holds
> `{position, rotation}`; an older `{position, target}` entry is dropped on load
> rather than misapplied.

**A save made from the bench belongs to both sides.** There is one camera
serving two rooms, and `view: serializeView()` read whichever room was on
screen — so saving without closing the bench first wrote the _bench's_
viewpoint into the ship's `view`, moving where the ship reopens to wherever the
bench happened to be. Two records had the mirror-image fault at the same time:
`stageView` and `stageLayout` are only written when the bench _closes_, so the
same save wrote last session's bench viewpoint and last session's roster —
quietly losing everything staged since.

> Three records, one mistake: **reading the live thing when the live thing is
> the other side's.** `shipViewpoint()`, `stageViewpoint()` and
> `stageLayoutNow()` each ask which side the live camera and scene currently
> _are_, and hand back the stash for the other. The stash cannot be stale:
> nothing can drive a camera that is not on screen, or move a stand-in that is
> not in the scene.
>
> `stageLayoutNow()` also feeds `serialize()`, which is what the dirty check
> hashes — so reading `state.stageLayout` directly had left the check blind to
> anything staged since the bench opened. The e2e suite now **fails the run** if
> a save made from the bench misplaces either viewpoint or the roster, for the
> same reason the backup-rotation guard does: this is the class of bug that
> loses work silently.

**Clicking a palette tile arms a ghost**, the same as placing anything else, so
you choose where a stand-in goes. Clicking one already on the bench focuses it
instead of adding a second — a second instance would give the association rule
two equally good answers.

`Ctrl+D` follows the same rule: it copies a **shape** — a hull is often several
boxes — but is refused on a stand-in, and says so. The shape ghost is drawn in
the collision green with its edges, like the shapes themselves; it came out in
the scene's default grey until the ghost was given a material to clone, since
`ghostMaterialFor()` returns nothing for nothing.

**The bench has its own undo history.** Its contents are deliberately not in
`serialize()` - they must never reach the ship - so a _ship_ snapshot restores
as "no bench at all", which is precisely how `Ctrl+Z` used to wipe it. A
separate stack also means undoing one box does not rebuild a hundred placements,
and the ship's own history is left untouched while you work.

**Moving or turning a stand-in carries its shapes with it**, and changes nothing
about the hull: the hull is authored in the module's own frame, so shifting the
stand-in is a _view_ operation. Left to itself the element slid out from under
its shapes, which then belonged to nothing — or worse, to whichever neighbour
they had drifted into — and deleting it afterwards could not find them either,
so they were stranded on the bench.

That is done by **watching the result, not by hooking the causes**. There are at
least six ways an element's transform changes — a drag, an `M` carry, the
inspector, an arrow-key nudge, `R`, `F` — and one watcher on the render loop
catches all of them, including any added later. Each shape remembers which
element claimed it, so a move knows whose shapes to carry without re-deciding
ownership half way through.

### Collision travels in its own file

Saving writes `ship_collision.json` beside `ship_manifest.json`, and loading
reads it back **in preference to whatever the ship carries**. That is the whole
point of it living apart: the collision is a property of the _kit_, not of any
one ship, so once these hulls are fitted the file can be shipped and the next
ship built from the same kit starts fully fitted.

**A failure writing it does not fail the save.** The ship is already on disk by
then, and throwing would leave the editor believing it had unsaved work — which
is exactly what happened against a server too old to know the route: every save
appeared to fail, and every reload warned about losing changes that were in fact
safely written. The status line names the problem instead.

The manifest carries three collision blocks, and they are **not** three copies
of the same thing:

- `collision[chunk]` — a room's **own** one-off shapes, world space, in Havok's
  parameters. Grouped by chunk because collision is streamed per room.
- `moduleCollision[moduleId]` — what a kit module carries, in the module's own
  local space, in Havok's parameters. **Written once per module**, for the
  runtime to instance onto every placement of it and to share one Havok shape
  between them.
- `moduleShapes[moduleId]` — the same hulls in the **authoring** form: editor
  coordinates, position/rotation/scale. The source the tool reloads from, and
  what `moduleCollision` is derived from — exactly the way `colliders` relates
  to `collision`.
- `colliders` — the editor's record of the _room's_ shapes.

**A module's hull is deliberately not expanded per placement.** It used to be,
and that block was both the largest of the three and the only one that grew with
the ship: placements × shapes, against modules × shapes. The runtime does not
need it — `instances` already gives every placement's module, chunk and
transform, which is everything required to place the hull. On a 114-instance
ship dropping the expansion took the manifest from 87 KB to 69 KB, and the gap
widens with every room.

`moduleCollision` and `moduleShapes` were once the same key, and a reload
silently produced empty shape lists: the reader expects editor coordinates and
filtered out every runtime-shaped record. One name, one meaning. A manifest
written before the split is converted on load rather than lost.

### Seeing collision the ship inherits

A module's shapes are stored once and instanced onto every placement at export
time, so the ship carried collision that was drawn nowhere but the staging area.
On a ship whose collision is all inherited, **Collision only** showed an empty
room — which reads exactly like a broken switch.

They are now drawn in place as well, from the record, as a **preview**: not in
`state.colliders`, not pickable, not exported, and rebuilt whenever anything it
depends on moves. There is nothing to keep in sync and nothing to edit by
accident — to change them you open the staging area, which is the one place
they are editable.

It is built at the end of `applyVisibility()`, so it follows chunk isolation,
hiding and the layer switch like everything else.

**It keeps up with the ship, too.** It is drawn from each placement's world
matrix, and `applyVisibility()` is not called by a move, a turn or a delete - so
the hulls used to sit where the elements had been, and outlive the elements
entirely. A watcher on the render loop notices those matrices changing, which
catches every route including the ones that only report themselves at the end of
a drag. Only placements whose module actually carries collision are looked at. That runs on every collider
added, so the rebuild is coalesced into one pass per burst — fitting a room of
eighty boxes would otherwise rebuild the whole ship's preview eighty times.

**The palette says which modules are done.** A tile whose module carries
collision gets a green dot — the same green the shapes are drawn in — and its
tooltip says how many shapes. Fitting a kit is a job you do a few modules at a
time and come back to, and there are 277 of them; without a mark the only way to
tell which were done was to stage each one and look. The dot is live: it lights
as soon as a shape is fitted and goes out when the last one is removed.

**A module that carries its own collision is covered everywhere, always.** The
manifest instances its shapes onto every placement of it, so there is nothing
per-room to keep in step and nothing to re-run after an edit: change the hull
once and every room that uses that module changes with it.

### What the viewport shows

The **Ship + collision / Ship only / Collision only** switch in the toolbar
composes with everything else rather than fighting it: it can only ever take
things _off_ screen, so chunk isolation and the `Shift+H` veil keep the last
word. It runs through `applyVisibility()`, the one place that decides what is
enabled, for exactly that reason. Hiding a layer drops any selection it hides,
so the gizmo and the inspector never act on something nobody can see. New
editor sessions start in **Ship only**.

**Runtime** is not a fourth world the way the collision staging area is: it
lights the ship you are building rather than swapping it for something else, so
elements stay pickable and an edit is re-lit on the next frame. See
[The three view modes](#the-three-view-modes).

**The view-mode combo is derived, never stored.** The viewport's state is two
independent flags — `runtime` and `unlit` — and `VIEW_MODES` maps each named
mode onto a pair of them. `viewModeFlags()` reads the table forwards to apply a
mode; `viewMode()` reads it backwards to name the flags. A `state.viewMode`
field beside them would be two truths about one thing, and the flags move on
their own — `setRuntimePreview()` moves `runtime` at both edges of a load — so
the stored name would be the one that drifted, and the combo would sit there
claiming a mode the viewport was not in.

These used to be checkboxes, which could spell states nothing rendered:
"diffuse only" with no lighting up, say. Named modes cannot.

### The shell thickness

The kit models its floors and ceilings as **single planes with no depth at
all**: `Platforms/Platform_3Plates` measures `[4, 0, 4]`. A zero extent has no
shape, and the guard against it used to read `Math.abs(v) || 1` — and `0 || 1`
is **one metre**. A two-plate room therefore came out with metre-thick slabs top
and bottom while its walls were 7.5 mm.

A module with no depth on some axis is now given at least the **collision shell
thickness**, and so is one that is merely _thinner_ than it. The shell is a
**minimum**, not a filler for zero-depth axes: it began as the latter, which
made it look broken — a barrel has real depth on all three axes, so nothing you
typed ever changed anything. A minimum is both what the name implies and what is
actually useful, since the kit's walls measure 7.5 mm and a collider that thin is
something a fast body tunnels straight through. The guard against a zero extent
clamps to a hair rather than a metre, which is a floor on nonsense, not a source
of it.

### What the inspector measures

For a primitive the `Size` row shows **Havok's own parameters**, not a bounding
box: a box gives its three sides, a sphere one radius, a capsule or cylinder a
radius and a height. A world AABB would show a sphere as three identical sides
and a _turned_ capsule as something with no relation to the radius it is
actually built from — and the whole point of constraining the scale is that
those numbers are the truth.

## Compound objects

A **compound** is several modules saved as one reusable object: a wall with its
lamp, a doorway with its frame, a bank of crates you keep rebuilding. It exists
because the ship is mostly repetition, and the repetition is not one module —
placing a lit wall is a wall, a light prop and an authored lamp, three times per
corridor.

**A compound is a recipe, not a mesh.** Saving one records what its members are
and where they sit relative to its origin; placing one **expands** that into
ordinary placements, one per member, sharing a `group` id. Nothing is baked, and
there is no compound object at runtime — the manifest and the `.glb` see exactly
the elements they would have seen had you placed each piece by hand.

That is a deliberate choice over baking the members into a single `.glb`:

- **A lamp is not geometry.** It is a `state.lights` entry riding a placement,
  and no mesh file could carry one. _Wall plus lamp_ is the case the feature
  exists for, so a model-file compound would have missed it entirely.
- **Collision is authored per module**, in the module's own local space, and
  every placement inherits it. Members keep theirs; a baked mesh would have been
  a new module with no hull, needing a fresh trip through the staging area.
- **Deleting one piece has to work.** A crate stack you place and then thin out
  is the ordinary case, not an edge case — and with a recipe it is just `Del`
  on one element.

### The bench

**Edit compounds** in the Compounds pane opens the **compound bench**. Like the
collision staging area it is a _mode_, not a property of the selection: the ship
is hidden, the bench is shown, and what is on it is not ship data. Members are
real placements carrying `stage: true` in the `__compound_bench` chunk, which is
what makes every tool work on them unchanged — the palette, `X`, `H`, `Del`,
`Ctrl+D`, dragging, the marquee, the inspector, **Add light**. They are filtered
out of the manifest and the `.glb` by `shipPlacements()`, and their chunk id is
never in `state.chunks`, so nothing can reach them from the ship side.

The two benches are separate chunks rather than one, because each has to be able
to tell its own contents apart from the other's — and they disable each other,
so only one is ever open.

The inspector's **Chunk** row is hidden on either bench. A bench member carries
a private pseudo-chunk that is never written to the ship, so the row could only
ever show a lie or take an edit that goes nowhere. The **Owner** row on a lamp,
by contrast, lists the _bench's_ members while a bench is open: fitting a lamp
to a bench piece and being offered every element in the ship except the right
one is no help at all.

**The bench keeps its own undo stack**, and its own viewpoint. A `Ctrl+Z` on the
bench undoes a bench edit; it can never reach past the moment the bench opened
and start undoing the ship behind it. The stack is emptied on the way in _and_
on the way out, so re-opening never offers to undo edits to pieces that are no
longer there.

**Closing keeps the bench.** What is on it is written to `localStorage` and
restored next time, the same way the collision area remembers its layout — a
compound is usually built over more than one sitting.

### Saving one

**Save as…** files what is on the bench under a name, in a kit and category of
your choosing. Filing it into a kit is what makes it findable: it appears as an
ordinary palette tile, in that kit's list, under that category, and is searched
and thumbnailed like any other. A compound tile carries a **blue dot** so you
can tell it is a recipe, and its thumbnail and turntable are rendered from the
members rather than loaded from a file — it has no `url`, because it has no
model of its own.

**Save** re-saves under the name the bench already has, with no dialog. After
the first save the name, kit and category are settled, and asking all three
again on every tweak turns _adjust the lamp, save, look_ into a five-click loop
whose one expensive misclick — the name field — silently forks the compound in
two. The bench banner says which compound is on it, so what **Save** will
overwrite is always on screen. It is greyed out until the bench has a name.

**Its origin is its first member's origin** — the piece you started with. That
piece is what you are really placing: a wall with a lamp on it is a wall, and it
should land where a wall lands. This began as the bottom centre of the bench's
bounding box, on the reasoning that a compound should meet the build plane the
way a module does, and that was wrong. A module's origin is on the grid because
the artist put it there; a bounding-box centre lands wherever the arithmetic of
two differently sized pieces puts it. A 4 m wall beside a 12 m platform centres
at 5, so every member came out half a metre off and a compound dropped with
`Move` at 1 m produced positions like `-0.5` and `3.5`. Anchoring on a real
module's origin gives integer offsets in, integer positions out — and gives a
group turn a pivot that is a real thing rather than an average. Recipes saved
under the old rule are shifted onto the new one as they are read, so they place
and turn like the rest.

**Clicking a compound tile while the bench is open loads it** rather than
placing it. It is the way back to a saved recipe — and placing a compound inside
a compound would fold a whole recipe into the next save as loose members, which
is what **Break apart** is for.

Recipes live in `ship_compounds.json` in the export directory — **not** inside a
kit folder, which holds bought art that the editor has no business writing to.
The server folds them into the catalogue as it builds it, so a compound is in
the palette from the next catalogue read onwards.

**Saving over a name replaces the recipe, and by default leaves the ship alone.**
Copies already placed are ordinary elements; they were expanded the moment they
landed and have no link back. The dialog says so as you type — it names what
already exists and how many copies are in the ship — as a warning rather than a
refusal, because saving over a compound is how one is edited. The same is true
of the **red ×** on a compound tile, which forgets the recipe: nothing already
in the ship changes. That is what being a macro rather than an instance means,
and it is the trade — you cannot edit every lit wall at once, and in exchange
you can edit any one of them.

**Update copies in the ship**, in the save dialog, is the way to break that
trade when you want to. Tick it and every copy of the compound is rebuilt from
what is on the bench: fit a lamp to a wall, place forty, find the lamp a foot
too low, fix it once. Each copy is re-laid **around its own anchor** — its first
member keeps its exact world transform and the rest are rebuilt around it — so a
fleet of walls does not shift by a millimetre when the lamp above them moves,
and each copy keeps its group id and its chunk. It is one undo step for the
whole push, and it is undoable **in the ship**, not on the bench whose history
is thrown away on the way out. What it costs is anything done to a copy _as a
copy_: a deleted component comes back, a renamed member goes back to its recipe
name, and a behaviour hung on one of its pieces goes with the piece. The choice
is remembered between saves, so **Save** pushes updates too once you have asked
for them.

### Placing, selecting, breaking apart

Placing a compound arms a ghost showing all its members and, on the drop, mints
**one fresh group id** and creates the elements under it — with their lamps.
Like any other tile the brush **stays armed** for a run of them.

**Selection is the only place a group is expanded.** Click any member and the
whole compound is selected; once every member is in `state.selection`, every
transform the editor already has — drag, `X`, rotate, scale, `Ctrl+D`, `Del` —
moves it as one, with no compound-aware code in any of them. The marquee, the
double-click and the hover outline expand the same way, so hovering a member
outlines the whole thing.

**`Ctrl+Alt+click` drills in** to the single member under the cursor, for when
you want to delete one crate off the stack or nudge a lamp. It is _not_ `Alt` on
its own: that is the eyedropper. Drilling in is simply not expanding the group,
so there is no second selection model to keep in step.

**`Ctrl+D` on a compound mints a new group.** Sharing the original's id would
have made selecting either select both, and moving one move the other.

**A compound turns as one piece, about its anchor.** The ordinary rule is that
every selected element spins about its own origin, which is right for a row of
props each facing its own way and wrong for a compound: the lamp is _on_ the
wall, and spinning both in place leaves it hanging in the air where the wall
used to be. So a selection that is exactly one whole compound always turns
rigidly about its first member — with `Shift+wheel`, and in the inspector's
rotation fields, which likewise carry the rest of the group along instead of
editing the first element alone. Scale is not carried: the wheel resizes every
selected element about its own origin, and the inspector matches it.

**Break apart** — in the inspector, shown only when the selection is grouped —
dissolves the group and leaves the pieces exactly where they are. From then on
each is selected, moved and deleted on its own. It is one undo step, and it is
the escape hatch: a compound that is _nearly_ right is placed, broken and
edited, rather than being a reason to go back to the bench.

## Settings

Ship-wide constants live in `state.config`, are edited in the **Settings** pane,
and go in the layout. They are authoring decisions, not preferences: a ship
fitted with an 8 mm shell and reloaded on another machine has to come back with
the same shell, or its collision silently changes. So they are **saved with the
layout**, go through the **undo stack** like any other edit, and are read back
through `{ ...CONFIG_DEFAULTS, ...saved }` so a layout written before a setting
existed returns that setting's default rather than `undefined`.

The pane also holds the two lighting sections, the **Ghost** transparency slider
and the **Big icons** palette toggle. Every row on it is now **saved with the
ship**, in one of three places: `environment` for the **Runtime** section, which
is authored ship data the game reads; `editorEnvironment` for the **Editor**
section, and `editorPrefs` for Ghost and Big icons, neither of which the demos
may read. The editor-side rows keep a `localStorage` copy as the fallback for a
ship whose manifest predates the block — a value in the manifest always wins.
Nothing on this pane is a setting you have to make again next session.

**The lighting is split by whose picture it describes, not by mode.** The
**Editor** section is how you happen to be looking at the ship in this browser;
the **Runtime** section is what the demos will render. Both are on screen at
once, because the previous arrangement — a single `Env`/`Exposure` pair on the
toolbar whose meaning changed under you the moment a runtime view was up — gave no way
to tell which of the two you had just tuned, and no way to see the other. It
also wrote whichever rig was live into the editor's `localStorage` keys, so an
afternoon of tuning the runtime look quietly redefined the editor's.

| setting         | default    | what it does                                                                        |
| --------------- | ---------- | ----------------------------------------------------------------------------------- |
| Collision shell | `0.008` m  | the _minimum_ thickness any collision box is given on any axis                      |
| Auto-save every | `2` min    | how often a recovery copy is written; `0` turns it off                              |
| Hull tolerance  | `0.1` m    | how far a fitted hull may stray from the art — the fidelity dial for **Fit a hull** |
| Hull thickness  | `0.35` m   | the depth a fitted hull gets along its thinnest axis                                |
| Hull offset     | `centered` | which side of the art that depth goes: `centered`, `negative`, `positive`           |

`CONFIG_RANGE` checks a numeric setting against `min`/`max` and a worded one
against its `choices`. Hull offset is the only worded one so far, which is why
`setConfig` grew a second branch rather than coercing everything through
`Number()` — `Number("centered")` is `NaN`, and the whole row would have been
silently unsettable.

The default matches what the kit's walls read as in the inspector. They in fact
measure 0.0075 m; the field rounds. Because the shell is a minimum, leaving it
at the default brings those walls up to 8 mm rather than leaving them at 7.5 —
set it to 0.0075 if you want them exactly as modelled, or higher (0.03 is a
reasonable choice) if you would rather nothing thin enough to tunnel through.

The rest of the pane is not `state.config` — those rows have their own state and
their own place in the manifest:

| setting              | section | default                | saved in            | what it does                                                                    |
| -------------------- | ------- | ---------------------- | ------------------- | ------------------------------------------------------------------------------- |
| Env                  | Editor  | `1.5`                  | `editorEnvironment` | IBL strength while authoring — where metals get nearly all their brightness      |
| Exposure             | Editor  | `0.55`                 | `editorEnvironment` | linear exposure of the editor views                                              |
| Tone                 | Editor  | `Khronos PBR Neutral`  | `editorEnvironment` | view transform of the editor views                                               |
| Ghost                | Editor  | `0.5`                  | `editorPrefs`       | how see-through `Shift+H` makes an element                                       |
| Big icons            | Editor  | on                     | `editorPrefs`       | double-width palette with double-size tiles                                      |
| Stray-chunk check    | Editor  | on                     | `editorPrefs`       | report elements that look like they were left in the wrong chunk                 |
| Env                  | Runtime | `1.5`                  | `environment`       | IBL strength in the game                                                         |
| Exposure             | Runtime | `0.55`                 | `environment`       | linear exposure in the game                                                      |
| Tone                 | Runtime | `Khronos PBR Neutral`  | `environment`       | view transform in the game                                                       |
| Specular AA          | Runtime | on                     | `environment`       | authored default for the player's specular-AA toggle                             |
| Reflection roughness | Runtime | `1`×                   | `environment`       | multiplier over every ship material's authored roughness, `0.5`–`2`              |

The **Runtime** rows are edits to the ship, so they push undo entries — the
sliders once per gesture, the checkbox once per click. The **Editor** rows do
not: they are how you are looking at the ship, and taking one back would spend
an entry on a change no edit made. **Reset to defaults** puts every row above
back, the ship-side ones under a single undo entry and only when at least one of
them has actually moved.

### Folding the panes

**Collision** and **Settings** are `<details>`, so their headers fold them. A
`<details>` rather than a hand-rolled toggle: the open state is then a real
attribute the browser keeps, keyboard and screen readers get it for free, and
Ctrl+F still reaches a folded pane's contents. Which panes you keep rolled up
is in `localStorage`, not the layout — whether _you_ fold Settings says nothing
about the ship, and putting it in the layout would make folding a pane count as
unsaved work.

> **The brush label used to fall off the bottom.** Both panes are sized to their
> content and could not shrink, so once they were taller than the room left in
> the palette they pushed the label — the only thing that tells you which module
> you are holding — out of the palette and under the status bar. Below about
> 615 px of window height there was no label at all.
>
> They are now one block that shrinks and scrolls, capped at 60% of the palette,
> and the module list yields first: `flex-shrink: 100` against the block's `1`.
> Sharing the shrinking evenly is no good, because it goes in proportion to
> content and with 277 tiles the list is so much the bigger that the panes still
> lost a third of themselves while the list kept hundreds of spare pixels.

### Resizing the palette and the inspector

A 5 px grip sits either side of the viewport. Drag it to resize the panel,
double-click it to restore the default, or focus it and use the arrow keys —
the grips are real `tabindex` elements with `role="separator"`, so the
keyboard reaches what the mouse can. The viewport takes the difference, being
the only flexible column.

Widths live in `localStorage` and are clamped to `180 px … 45%` of the window
on every drag **and** on every window resize, so a layout dragged wide on a big
monitor cannot come back on a laptop with no viewport left in the middle.

> **Two widths are remembered for the palette**, `paletteWidth` and
> `paletteWidth.big`. With one, dragging the palette once would have pinned it
> and made **Big icons** — which needs a much wider column to be worth having —
> do nothing ever after.
>
> **The width is set on `document.body`, not on `:root`.** `body.big-palette`
> already declares `--palette-w` as a class rule on `body`; a variable set on
> the html element is shadowed by that for every descendant, so the inline
> style has to sit on the same element to win.

### The floating tool windows

**Chunks…**, **Probes…** and **Edit behaviours…** open windows rather than
modal dialogs. Each one is a place you edit the ship _while looking at it_ — a
probe box you are dragging, a room you are renaming, the JSON of a behaviour
you are watching take effect — so a scrim over the viewport would hide the only
thing that says whether the edit was right. They are not modal in the keyboard
either: shortcuts are claimed only while the focus is _inside_ a window, where
`Escape` closes it, and the ship stays live behind.

Which means each of them owes you the two affordances a window has to have:

- **Drag** it by the titlebar. The position is clamped into the viewport on
  every move, on open, and on every browser resize — a window can be pushed at
  the edge but not over it, and shrinking the browser cannot strand one outside.
- **Resize** it from the bottom-right grip. That grip is the browser's own
  `resize: both`: the platform already draws it, tracks the pointer and honours
  `min-*`/`max-*`, and a hand-rolled one would be a worse copy. It is why the
  window element is a bare positioned box and the **panel inside it** carries
  the size — `resize` needs a clipping box, and clipping the window would take
  the panel's drop shadow with it.

In the behaviours window the height a bigger window buys goes to the JSON box,
which is the field that needed it; the other two are rows of controls and stay
content-tall.

## Live checks

The inspector re-runs these on every change (they replace `build_ship.py`'s
batch validators):

- overlapping chunk volumes — ambiguous portal membership
- objects below `y = 0`, or off the current move-snap grid
- doors whose two sides resolve to the same chunk, or that have no leaves
- chunks no door reaches
- elements that look like they were left in the wrong chunk (below)

### Elements left in the wrong chunk

The easiest mistake to make and the hardest to see: you build on into the next
room and forget to move the active chunk on with you. Nothing looks wrong — the
piece is where you put it — but at runtime it belongs to a room the player is
not in, so it vanishes through the portal or hangs in the air beyond it.

The obvious test does not work. **A chunk has no authored volume**: its box is
the union of whatever is assigned to it, computed after the fact by
`buildManifest`. Ask "is this element inside its chunk?" and the answer is
always yes, because the element is one of the things that decided where the
chunk is.

What a chunk does have is a shape. The pieces that make up a room are stuck to
one another — tile against tile, trim against wall — so the question worth
asking is which of a chunk's members hang together and which hang off on their
own. `strayChunkMembers()` groups each chunk's members by what touches what and
keeps the largest group; anything outside it is reported. When some _other_
chunk's group does reach the piece, that chunk is named, because that is the one
it was meant for:

```
2 element(s) assigned to CH00_Storage sit in CH03_StorageC2: P0040, P0041
1 element(s) assigned to CH02_StorageCTL touch nothing else in it: P0210
```

> **Grouping, not measuring each piece against the rest of its chunk in turn.**
> That is the obvious implementation and it is wrong: two pieces left behind in
> the same chunk hide each other, because the "rest" that each is measured
> against contains the other and stretches over all the ground between them.
> Group them instead and each is on its own.

Touching is generous — half a metre (`STRAY_CHUNK_SLACK`). Trim, decals and
door frames are all mounted a few centimetres proud of the surface they belong
to, and on this ship the widest such gap measured 25 cm, so anything tighter
reports honest work as a mistake. A piece genuinely in the wrong room is a kit
tile away at the very least, and the kit's grid is 4 m. Half a metre clears
every mounting offset and is still eight times inside the smallest real
mistake. Measured on the finished ship: 135 placements, four rooms, zero
reports — and moving one real placement to the wrong chunk is caught, with the
right chunk named.

It is a **warning, never an error**, and it never blocks a save. Ships are
built outwards, and a chunk being filled in right now legitimately holds a piece
or two that reach nothing yet. For the same reason a chunk with one member is
left alone — there is nothing for it to be outside of — as is one that splits
evenly, where there is no larger group to call the odd one out from.

Besides the Live checks panel it is repeated in the status bar on every save,
appended to the usual message, because the panel is easy to build past:

```
saved · 135 objects · 5 chunks — check chunks: 1 element(s) assigned to CH00_Storage sit in CH03_StorageC2: P0040
```

Turn it off with **Stray-chunk check** in Settings ▸ Editor if you are working
in a way it cannot follow — it lives on `editorPrefs`, so the choice is
remembered.

---

## Implementation notes

**Shared materials are not optional.** Every module .gltf in the kit references
the same ~20 root-level textures and names its materials identically
(`MI_Trim_01`, `MI_Trim_02`, `M_Light`…). Loading them naively would give you
one copy of a 2–4 MB atlas per module. `kit.js` keeps the first material seen
under each key and disposes the duplicates _with their textures_; `thumbs.js`
does the same on its own engine, using the same key function. Placing 300
modules costs 8 materials and 9 textures.

> **The key is the name _and_ its transparency**, not the name alone. The kit
> authors some of those shared names two different ways: `M_Glass` is `BLEND`
> with an alpha of `0` in two files and `OPAQUE` in twelve, and `M_Decal_White`
> is `MASK` in thirty-one and `OPAQUE` in twenty-six. Keyed by name alone,
> whichever module happened to load first decided how glass looked _everywhere_
> — and since the palette keeps its own cache, filled in a different order, a
> window could be see-through on its tile and solid in the ship at the same
> time. That is how this was found.
>
> Transparency is the only thing those pairs differ in and the one thing that
> cannot be shared, so it goes in the key. It costs two extra materials across
> the whole catalogue — one, now that glass has an authored answer.

**`kit_materials.json` carries the transparency the .gltf lost.** Quaternius'
own shaders make the glass see-through — Godot's `M_Glass_Base.tres` is
`blend_mix` with `ALPHA = mix(0.05, 0.5, perlin)` scrolling over a pale green —
and glTF has no way to say "alpha driven by noise", so the export wrote it
`OPAQUE` and every window in the kit came out solid. The file already carried
the emissive values the export flattened; it now carries alpha the same way:

```json
"transparency": {
  "M_Glass": { "alpha": 0.5, "tint": [0.0, 0.427451, 0.043137], "roughness": 0.5, "ior": 1 }
}
```

The kit's own pale green `(120, 198, 152)` at the middle of that alpha range is
technically faithful and reads as almost nothing: at 28% opacity the pane is
three-quarters whatever is behind it, and the tint led red by ten values out of 255. These are the values judged in the Sandbox against the real ship instead —
a deep green `(0, 109, 11)` at half alpha, which is a window you can see is
glass.

`ior` is why it reads as a tint rather than a shine. A dielectric's reflectance
is `((n−1)/(n+1))²`, so an index of refraction of 1 makes it zero: the pane
stops catching a white highlight and the colour is left to be read on its own.

> **There is no `KHR_materials_transmission` in the export, and there must not
> be.** Babylon's serializer only emits it when `subSurface.isRefractionEnabled`
> and the refraction intensity is non-zero; nothing here touches either, and
> setting `indexOfRefraction` does _not_ turn them on — measured: after
> `mat.indexOfRefraction = 1`, `isRefractionEnabled` is still `false` and the
> material exports with `KHR_materials_ior` alone. `e2e.mjs` asserts the
> absence.
>
> Worth knowing anyway, because the two get confused: **Babylon-Lite's loader
> starts its refraction sub-feature from `KHR_materials_ior` on its own.**
> `gltf-ext-dielectric.ts` writes `subsurface.refraction = { indexOfRefraction }`
> for any material carrying the ior extension, and `RefractionProps` is
> presence-enabled — no transmission factor needed. That path retargets the
> scene to an offscreen HDR buffer, which hides the fluid surface compositing
> after it, so `aquanova/main.ts` (~line 348) walks the ship meshes and clears
> `subsurface.refraction` at load. So an exported `ior` is harmless in the file
> and handled at the far end; it is not the transmission extension.

All of it is applied to the ship and the palette alike, and all of it **is
exported**: the glb comes out `alphaMode: "BLEND"` with the tint and alpha in
`baseColorFactor` and the index of refraction in `KHR_materials_ior`, so the
runtime gets the same glass. Like the emissive values, it is authored state, not
a viewport aid — unlike `backFaceCulling`, which is put back the way the kit had
it on the way out.

> Applied **before** the dedupe key is taken, deliberately. Once the override
> has settled what `M_Glass` is, the kit's two spellings of it are the same
> material again and share one copy, instead of being kept apart over a
> difference that no longer exists.

**Two materials are the same material when they read the same textures.** The
dedupe key is `name | transparency | the textures the material reads` — the
name alone is not an identity, because the packs reuse names over genuinely
different materials:

| Where | What the name hides |
| --- | --- |
| Pirate Kit | *every* model calls its material `Atlas` and **embeds its own 32×32 slice of the palette**. The props' slice is white from row 7 down; a character reads its skin and clothes from row 9. |
| MegaKit vs Essentials Kit | `MI_Trim_01`, `MI_Trim_02`, `MI_Trim_03`, `MI_Trim_03_Dark` and `M_Black` exist in both, over different atlases — `T_Trim_01_BaseColor.png` in one, `T_Trim_01_BaseColor_Red.png` in the other. |
| Essentials Kit alone | `MI_Trim_02` maps to two different atlases across its own files. |

Keyed by name, whichever module loaded first decided what everything with that
name looked like — and it showed: **the Pirate characters rendered grey except
for the prop in their hand**, which is the one part of them reading from a row
the props' atlas also fills in. The palette keeps a second cache of its own,
filled in a different order, so a module could be right on its tile and wrong in
the ship at the same time.

`Texture.url` is the identity for both kinds of kit: a file the loader fetched
is its resolved URL — already through the kit-root redirect, so two modules
naming the same atlas agree — and an image embedded in a .gltf is given
`data:<the .gltf's url>#image0`, which names the file it came out of. Sharing
then happens exactly when it is free: the MegaKit's modules still collapse to
one material each, while a kit that embeds a texture per model gets one material
per model, which for a 32×32 palette costs nothing.

**A cached thumbnail is versioned.** The stills and turntables live on the
server's disk and outlive any reload, so a change to how a thumbnail is
_rendered_ would otherwise be invisible until someone deleted the folder by
hand — which, for a bug whose whole symptom was a tile disagreeing with the
ship, is exactly the wrong failure mode. `THUMB_VERSION` is stamped into every
cache key, and a `PUT` sweeps the superseded copies of that module, unversioned
ones included: bumping it costs one re-render each, not an orphaned folder.

**Geometry is instanced.** Each module is loaded once as a disabled prototype;
placements are `createInstance()` children of a `TransformNode` named after the
placement id. Picking walks `mesh.metadata.placementRoot` back to that node.

**Ghosts are textured clones.** Instances cannot carry their own material, so
the ghost clones the prototype's meshes (`Mesh.clone()` shares geometry — zero
new geometries) and gives each one a cached translucent copy of its _real_
material, so you can tell which face of a module you are looking at.
`PBRMaterial.clone()` re-creates every texture, which would duplicate the kit's
2–4 MB atlases per material, so the copies are disposed and the slots
re-pointed at the originals — the scene texture count does not move.

**Outlines use per-instance edges rendering.** This is the only one of the three
options that follows a single instance: `HighlightLayer.addMesh()` throws on an
`InstancedMesh`, and `renderOutline` is read off the _source_ mesh, so it would
outline every copy of that module at once (both verified). Edges rendering also
reads the instance world matrix every frame, so nothing has to be re-synced
after a rotate or scale. Hover is orange, selection is blue, and the two can
never land on the same element.

The edges go on whatever is actually **drawn**, which in the Runtime view is not
the mesh that was picked. There the authored instance is `isVisible = false` and
a stand-in draws in its place, and Babylon dispatches an edges renderer from
`_processActiveMeshes` — an invisible mesh is never an active mesh, so an
outline hung on it is built, updated and never drawn. `edgeMeshes()` therefore
maps each authored mesh through `renderListFor()`, which hands back the stand-in
in a runtime mode and the mesh itself in an editor one. Picking is unaffected
either way: it walks `metadata.placementRoot`, and the authored instance is
deliberately left enabled and pickable.

> **`edgesWidth` is not a pixel width.** The line shader offsets the vertex in
> _clip_ space, before the perspective divide, and the renderer hands it
> `edgesWidth / 50` — so what you see is
> `edgesWidth * renderHeight / (100 * viewDepth)`, which doubles every time you
> halve your distance. A fixed 5 read as a fine line across a room and as a
> 16 px slab of colour with the camera against a crate, hiding the very thing
> it was drawn to point at.
>
> Turning that around gives the width to ask for, so it is set from the view
> depth of each outlined mesh on every frame — the camera moves without
> emitting anything, so this rides the render loop. It is the _view_ depth, not
> the distance: that is what the shader divides by, and the plain distance
> would fatten the outline on anything off to the side of the screen. Measured:
> 4 px at 12 m and 3 px at 2.5 m, against 5 → 16 px before.

**The ghost compensates for the module's origin.** Kit modules are not modelled
around their own origin — `ShortWall_Band2_Straight`'s body sits 2 m away from
it — so snapping the origin straight to the grid drops the piece a whole cell
away from the pointer. The ghost offsets by the module's local body centre
before snapping, which bounds the error at half a cell (measured: 0.40 m
compensated vs 3.60 m without).

**The ghost tracks the cursor on a plane through the module's _body_.** Kit
modules are not modelled around their origin: a wall sits 2 m to the side of it
and `TopCables_Corner_*` a full 4 m above it. Compensating only in X/Z left tall
modules floating far up-screen from the pointer — far enough that you ran out of
window before you could place one near the bottom of the viewport. The offset is
now applied in all three axes and the tracking plane raised to match, so the
body lands on the pointer (measured 0 px with snapping off, versus 134 px for
the bare origin).

**Keyboard shortcuts die while a toolbar control holds focus** — a `<select>`
that keeps focus after being used swallows the numpad keys entirely. Toolbar
controls are blurred on change, and any pointer-down in the viewport releases
focus back to the app. `+`/`-` on the main row work as aliases for the numpad
pair.

**A number field swallowed letters, which made `R` look like it needed several
presses.** Standing aside for a focused control is right — that is what typing
is — but `Position X` and `Intensity` are `type="number"`, and a number field
cannot spell a letter: the browser drops it silently. So `R` after typing a
value did nothing, again did nothing, and only came back once something was
clicked. A control now keeps only the keys it can actually use: a `textarea`, a
`select` (a letter jumps it to the matching option) and a text field keep
everything, while a **blind** input — number, range, colour, checkbox, radio —
hands a bare or `Shift`ed letter back to the editor, which costs it nothing
because it was discarding it. `e` is left with the field, since a number reads
it as an exponent and no shortcut is spelled with it; `Ctrl`/`Cmd` chords stay
with the field too, so `Ctrl`+`Z` in a half-typed number is still the browser's
own text undo. `Enter` now blurs a focused control — "done with this field" —
which hands the keyboard back after a search as well, where the letters really
were the field's; `Escape` already did.

**Alt is claimed from the browser.** On Windows, Chromium hands `Alt` to the
menu bar — which then swallows the following keystrokes, so WASD silently stops
working after any `Alt` press. Both the keydown _and_ the keyup are
`preventDefault()`ed (menu activation happens on release). Worth knowing when
debugging: this does **not** reproduce under Playwright, whose injected key
events bypass the browser's own accelerator handling entirely.

**Dragging moves the real elements, not a ghost.** A ghost is per-module and
single; a drag has to move a whole selection. The delta is measured on a
horizontal plane at the _selection's own height_ rather than the build plane, so
the motion tracks the cursor instead of being skewed by perspective — and the
**delta** is snapped rather than each element, which keeps a group's internal
spacing exactly as authored even when it was placed off-grid.

**A drag has to out-rank the camera.** Left-drag normally looks around, so
`setupPointer` registers its observer _before_ `camera.attachControl` and sets
`eventState.skipNextObservers` on a pointer-down that starts a drag. The camera
input then never records a start position, so the whole gesture is swallowed —
no need to detach and re-attach anything mid-drag.

**Double-click is detected by hand, not via `POINTERDOUBLETAP`.** Babylon's
double-tap requires the single tap to wait out the timeout, and a placement tool
cannot afford 300 ms of latency on every click. Instead the first click always
acts immediately and a second one within 320 ms and 6 px _also_ emits
`dblclick`, which frames the element. While a module is armed the double-click
handler bows out, which keeps rapid clicking in the same spot placing two
modules.

**There is no pick-plane mesh.** The build plane is solved analytically by
ray/plane intersection (`cursorOnGrid()`). An actual mesh would sit in front of
the geometry the moment the plane is raised above it, making every element
unselectable — and at `y = 0` it tied with floor tiles resting exactly on it.
Consequence worth knowing: if the plane is above the camera _and_ you are
looking down, there is legitimately no intersection and the ghost stops moving.

**Everything is double-sided in the editor.** Kit modules are single-sided in
places, and a wall turned away from the camera simply vanishes — correct in
game, useless while building. `scene.onNewMaterialAddedObservable` forces
`backFaceCulling = false` on every material, backed by a re-sweep whenever the
material count changes: the observable can fire from the base `Material`
constructor, _before_ a subclass has applied its own culling default. The
authored value is remembered in a `WeakMap` and `withAuthoredCulling()` puts it
back for the duration of the glTF export, so the runtime gets exactly what the
kit shipped rather than the editor's convenience setting.

**One free-flying `UniversalCamera`, no arc-rotate.** An arc-rotate camera is
always tethered to a pivot: its speed and reach are tied to the orbit radius, so
it slows to a crawl as you close in and cannot simply walk down a corridor.
Movement is fed through `cameraDirection` / `cameraRotation` rather than by
writing `position` directly, so camera inertia still smooths it — with a
`(1 - inertia)` factor on each impulse so the steady-state speed is the one
asked for regardless of the damping. The camera therefore _coasts_ to a stop
instead of halting dead: measured 0.68 m of coast, then nothing.

**Zoom is a dolly, not a radius.** A free camera has no pivot to collide with,
so the wheel moves it along its own view direction by a fixed step at any
distance — the same amount 200 m out as at 40 cm from a wall.

**`scene.doNotHandleCursors = true`.** Babylon rewrites `canvas.style.cursor` on
every pointer move, which would immediately undo hiding the cursor during a
placement.

**The wheel is intercepted on `#viewport` in the capture phase**
(`passive: false`) and both `preventDefault()`ed and `stopPropagation()`ed. A
listener on the canvas itself is not enough: capture and bubble listeners on the
same target fire in registration order, so the camera would zoom a second time
on top of the dolly. `preventDefault()` also stops `Ctrl` + wheel from
page-zooming the browser.

**The kit's .gltf files reference textures by bare filename** but the PNGs live
once at the glTF root rather than beside each module, so the server falls back
to the root before returning 404.

**Thumbnails** render lazily on a second engine as tiles scroll into view, then
POST to `cache/thumbs/` so later sessions are instant. Shaders only compile
while rendering, so the readiness wait has to call `scene.render()` on every
tick — waiting without rendering spins forever and yields a blank tile.

The framing is re-applied _after_ `setTarget()`. `ArcRotateCamera.setTarget()`
re-derives alpha and beta from wherever the camera currently is, so a module
modelled high above its origin (`TopCables_Corner_*` sits at y = 4) pushed beta
past 90° and got rendered from underneath, while alpha drifted per module and no
two tiles matched. Note that _negating_ beta does not fix this — `cos` is even,
so the camera keeps its height and only mirrors in azimuth. Delete
`cache/thumbs/` (and `cache/turntable/`) to force a re-render after changing the
framing or the lighting.

**`grid-auto-rows: max-content`** on the palette is load-bearing. Without it the
implicit grid rows collapse to zero height and every thumbnail is invisible even
though the images loaded fine.

**Selection needs a _quick_ click.** A left press only selects if it both stayed
within 4 px and was released inside `CLICK_MS` (300 ms); anything held longer is
a camera gesture and selects nothing. Both halves are needed, and the movement
test alone was not enough:

- `onClick` already ignored gestures that moved, so a look-drag never selected.
- But `beginDragCandidate()` used to call `select()` on **pointer-down**, so
  merely holding the button over an element picked it up regardless. Selection
  is now deferred — a drag selects what it moves at the moment it activates,
  and a press that stays put is only a selection if it was quick enough.

That ordering matters: a deliberately slow drag must still select the thing it
is moving, which is why the drag path selects on activation rather than relying
on the click.

**Viewport lighting is fixed and not authorable.** Most kit materials are fully
metallic and render black without an IBL, so the ship's own HDRI is loaded for
reflections and pushed well above its render value, with a hemi + key + fill rig
on top. Legibility beats fidelity here; the real look lives in the render
pipeline. `public/data/kit_materials.json` carries Quaternius' authoritative
emissive values (from the Unity URP project) and damps them for the viewport.

**There are two hemispheres, aimed at each other.** Babylon's
`HemisphericLight.direction` points at its _sky_, so a single one pointing up
gives every downward-facing surface nothing but its (dark) `groundColor` — and
that is every ceiling panel, pipe run and platform underside in the kit. A
second hemisphere pointing **down** lights those. Its own `groundColor` is black
so it does not light the floors a second time.

Worth knowing what this can and cannot do: a hemispheric light contributes only
a **diffuse** term, so it transforms the pale dielectric panels (`MI_Trim_03`,
metalness 0 — near-black underside to fully readable) and barely touches the
fully metallic ones, which have no diffuse term at all and are lit by the IBL
instead. That is physically correct, not a bug; raise `environmentIntensity` if
metal undersides ever need more.

> Measuring this needs `page.screenshot()`, **not** `getImageData()` on the main
> canvas: the editor's engine is created without `preserveDrawingBuffer`, so
> reading its backbuffer returns stale pixels and every setting appears to give
> an identical result. The thumbnail engine does set that flag, which is why the
> same technique works there.

### Exposure, and why pale panels used to look blank

The single most damaging legibility bug in this tool was **exposure**, and it
took a while to find because it looks like a texture problem.

`MI_Trim_03` — the material on `Platform_Simple*` and `Door_Simple` — is a pale
**dielectric** (its ORM blue channel is 0 everywhere; it is not metallic at all,
despite what the flat white render suggests). At exposure 1.0 it renders at
about **240/255**: deep in the KHR PBR Neutral _highlight shoulder_, where the
tone curve is almost flat. Everything the artist painted — dirt mottling, panel
seams, the bolt strips — gets compressed into the top ~6% of the output range
and simply disappears. Measured contrast over such a panel:

| exposure | mean | std-dev |
| -------- | ---- | ------- |
| 1.0      | 240  | 6.5     |
| 0.7      | 234  | 10.7    |
| 0.55     | 229  | 12.2    |
| 0.45     | 208  | 13.1    |

The giveaway was that the **same module looked sharper while being dragged**
than once it was placed. There is no material difference at all: forcing the
ghost's material opaque makes it render pixel-identical to a placed one
(mean 240.8, sd 0.33 for both). The ghost's `GHOST_ALPHA = 0.7` was quietly
acting as an exposure cut, blending the surface down the curve and out of the
shoulder.

So the default is `EXPOSURE_DEFAULT = 0.55`, with a slider in **Settings ▸
Editor** because the trade is real: lower exposure buys detail on pale panels
and costs brightness on genuinely dark props. The editor's own value is
remembered in `localStorage`, as `editorExposure`.

The thumbnail scene had exactly the same problem, worse — it ran at exposure 1.4
with `environmentIntensity` 2.4, which is why `Platform_Simple_*` and
`Door_Simple` tiles looked like flat blocks of colour. It now runs at 0.5 / 1.6.

**Beware `Number(v) || DEFAULT` when clamping.** A literal `0` is falsy, so that
idiom silently jumps to the default instead of clamping to the floor;
`setExposure` uses `Number.isFinite`.

### What is _not_ wrong

**No ORM colour-space fix is needed**, and the loader is not mis-tagging
anything. It is easy to talk yourself into the opposite, because the flags read
backwards:

| texture    | `gammaSpace` | `useSRGBBuffer` |
| ---------- | ------------ | --------------- |
| Base Color | `false`      | **`true`**      |
| ORM        | `true`       | `false`         |
| Normal     | `true`       | `false`         |

Base Color is decoded by the **GPU** through an sRGB internal format, so
`gammaSpace` is correctly `false` — the shader must not decode it a second time.
ORM and normal maps never consult `gammaSpace` in the PBR shader at all, so the
`true` there is an inert leftover default. Verified by loading a kit `.gltf`
into a bare Babylon scene with none of this tool's code: identical flags.

### Unlit mode

**Unlit mode** (`PBRMaterial.unlit`) drops lighting entirely and shows raw
albedo. It stays available as the **Editor unlit** view mode for reading a very
dark prop, but it is
**not** used for thumbnails any more — it throws away far too much (all shading,
all form, every specular cue) and the flat-tile problem it was introduced to
solve turned out to be the exposure bug above.

Raw albedo leaves the darkest props very nearly black, so unlit mode adds a flat
**emissive lift** of `UNLIT_LIFT = 0.16`. Emissive is _additive_ and is honoured
in unlit mode — measured on a 0.08 albedo, the rendered pixel goes 81 → 199 with
a 0.5 emissive — which is exactly what a near-black texture needs; scaling the
albedo instead would leave black black. Two things make this fiddly:

- Materials carrying an **`emissiveTexture`** are the light strips, and their
  `emissiveColor` _multiplies_ that texture. Overwriting it would break them
  rather than lift anything, so `applyViewportMode()` skips them.
- The kit's own emissive values are written by `applyKitValues()` _after_ the
  material is constructed, so the snapshot taken by
  `scene.onNewMaterialAddedObservable` is a stale black. `applyKitValues()`
  re-takes it with `noteAuthoredEmissive(mat, true)` once its values are in
  place — otherwise toggling unlit off would darken every kit light.

Like `backFaceCulling`, neither `unlit` nor the lift may reach the export —
`unlit` serialises as `KHR_materials_unlit` and the lift as a grey glow on every
surface — so `withAuthoredMaterials()` clears all three for the duration of the
glTF write. It sweeps **every** material, not only the ones the editor touched
for culling, so a material that skipped that sweep cannot smuggle its lift into
the file.

### Palette tiles

Thumbnails are rendered over a **checkerboard backdrop** (a `DynamicTexture` on
a background `Layer`). Plenty of pieces have alpha-cut or genuinely transparent
parts — grilles, glass, cables — and against a flat fill those read as solid
background; a checker makes every see-through region obvious at a glance.

**Hovering a tile spins the module through a full turn.** A still cannot tell
you which way a corner turns or which side a cable run is on. Each turntable is
`TURN_FRAMES = 12` renders packed into one horizontal **sprite sheet**
(2304 × 192) rather than an animated GIF: GIF's 256-colour palette would band
exactly the subtle greys this is meant to reveal. One sheet is also a single
request, so a preview cannot tear halfway through a turn. Sheets are rendered on
first hover only — 12 renders is far too expensive to do up front for 277
modules — and cached under `cache/turntable/` via `/api/turn/<key>`.

The animation is stepped in JS rather than with a CSS `steps()` animation.
Percentage `background-position` is measured against _(box − image)_, so with a
12-frame sheet the step between frames is `1/11` of the range, not `1/12`;
driving it explicitly avoids that off-by-one entirely.

**All thumbnail-scene work is serialised through one promise chain.** Stills and
turntables share a single scene and a single camera, but they are triggered
independently — stills by scrolling, turntables by hovering — so they _will_
overlap in normal use. A turntable holds its module in the scene across 12
renders, and any still rendered in that window came out with **two different
modules in the picture**. `exclusive()` queues every use of the scene, and
`loadFrameAndRun()` additionally sweeps any leftover mesh before loading, so a
missed disposal cannot photobomb the next tile either.

**A tile whose picture does not exist yet says so.** The first visit to a kit
has nothing in `cache/thumbs/`, so a screenful of tiles would otherwise sit
blank for a few seconds and read as a broken palette or as modules that failed
to load. When `request()` misses the cache the tile takes a `thumb-pending`
class, whose `::before` covers the thumbnail square with **"Generating
preview…"**, and drops it when the render lands — success or failure, so a
module that cannot be loaded ends up an empty tile rather than a permanent
promise. The message is deliberately on the tile rather than in a corner
progress line: it is the *tile* that is waiting, and several are waiting
independently.

---

## Environment probes — the ship reflects its own rooms

Babylon-Lite has no global illumination, and this ship is a corridor crawler:
almost every surface is a metal panel a metre or two from another metal panel.
Lit by analytic lamps alone, those panels reflect nothing — or, worse, reflect a
single ship-wide skybox, so a sealed storage room mirrors the stars.

An **environment probe** is the answer. It is a small cubemap of one room,
prefiltered for roughness, that every material in that room reflects. It is not
a lightmap: it carries no baked diffuse, nothing is unwrapped, and no second UV
set exists anywhere in the pipeline. It is a picture of a room, taken from
inside it, used as that room's reflection and its diffuse irradiance.

### What a probe is

Probes are authored in **Probes…** in the toolbar. Each one is:

| | |
| --- | --- |
| **box** | position and size of the volume the probe covers, in editor space |
| **capture point** | where the six faces are rendered from — usually eye height, not the box centre |
| **influence centre / size / inner size** | the volume the *runtime* blends this probe over — see [The influence volumes](#the-influence-volumes) |
| **resolution** | face size; 256 by default, which is the kit's own texel density |

An element belongs to a probe when its bounding box **intersects** that probe's
box, and the element's own bounds are the **union of all its primitives** — a
wall cannot have its trim band lit by the corridor and its face by the room.
Where boxes overlap, the probe holding the larger share of the element wins, and
an exact tie goes to the tighter box, so a cupboard nested inside a corridor's
probe still wins its own geometry. The share is a per-axis fraction rather than
an overlap volume, because ship trim is frequently a zero-thickness sliver whose
volume is exactly zero against every probe.

An element in no probe box at all reflects **nothing**, which is exactly what
the runtime does with it.

### Reading the probe pane

Three near-identical triples of numbers stacked in one list read as a wall of
digits, and the boxes they describe are nested inside one another in the
viewport, so it is easy to drag the wrong one. The pane is therefore split by
volume, each with its own heading:

| section | rows |
| --- | --- |
| **Probe box** | Centre, Size, Camera |
| **Influence box** | Centre, Size |
| **Inner box** | Size |

**ID**, **Always visible**, **Env faces** and **Texture size** sit above all
three, because they belong to the probe rather than to any one of its boxes.
Texture size follows Env faces directly: both are about the cubemap, and
separating them put a dozen coordinates between two rows that are read together.

There is **no Apply**. A probe is edited the way an element is: type into a
field and the viewport follows, keystroke by keystroke. One visit to a field is
one **undo entry**, however many characters it took — the rule the inspector
already follows — so `Ctrl`+`Z` takes back the value you typed, not the last
digit of it. Typing is deliberately quiet about refusals, because `0.5` and `-3`
are both unusable on their way in; a value that never became a probe is called
out when you **leave** the field, and the field snaps back to what the record
still holds. The **ID** is the one exception: it commits when the field is left
rather than as it is typed, because renaming per character would leave a trail
of probes called `E`, `EN`, `ENV`…

**Always visible** keeps this probe's boxes on screen while another one is being
edited, and it is **per probe** — it is how you place a room's fade region
against its neighbour's rather than from memory. The probe being edited is drawn
in the **bright** palette; every probe held beside it is drawn in a **dim** one,
about a third as saturated and half as opaque, so the one your keystrokes reach
is never in doubt. Dim boxes stay pickable: clicking one selects that probe,
which then turns bright and takes the pane with it. **Env faces** is per probe
too, and follows the same rule — a probe shows its captured cubemap when it is
selected, or when it is being held on screen.

Both flags last as long as the **window** does: closing **Probes…** takes every
gizmo down whatever they say, because "always" means "while I am working on the
probes", not "for ever". They are saved beside the ship in `editorPrefs`, never
inside `environmentProbes[]` — the runtime has no business reading two booleans
about the editor's viewport — and they survive an undo, so `Ctrl`+`Z` on a typo
does not also put the probes you had on screen back down.

The probe **list** takes whatever height the window is given and gives it back,
so a tall window is a long list rather than a short list with grey space under
it.

Each heading carries an **eye** that shows and hides that volume on its own, so
you can pull the influence boxes out of the way while placing the capture box,
or hide the capture box to see the fade region against the room unobstructed.
Hiding a volume also **drops it from the selection** — a gizmo that is not drawn
cannot be dragged, framed or outlined, and leaving it selected would leave the
arrow keys moving something invisible. The eyes are **per probe**, like the two
flags above them: which box is in your way depends on the room you are working
in, and pulling one probe's influence box aside has no business taking its
neighbour's down at the same time — least of all when the neighbour is on screen
precisely so the two can be read against each other. So the choice follows the
probe the pane moves to, and lands in `editorPrefs` beside Always visible and
Env faces rather than in `environmentProbes[]`.

Selecting a box in the viewport marks its heading in the pane — accent colour
and a **selected** badge — which is the other half of the same problem: with
three boxes drawn one inside another, the pane otherwise gives no clue which of
them `Ctrl`+wheel is about to resize.

### The influence volumes

The box above answers "what does this cubemap *show*, and onto what does it get
projected". It does **not** answer "when the player is standing here, whose
cubemap is this room". That is a separate pair of boxes, because the two
genuinely differ: the projection box has to be the room's walls or the parallax
correction is wrong, while the hand-over to the next room wants to start before
the doorway and finish after it.

The runtime blends two probes with Lagarde's normalized distance field: **full
strength inside the inner box, fading to nothing at the faces of the outer
one**. Both boxes share one centre — a probe that faded out asymmetrically would
have to be two probes — so the pane has one **Influence centre** row and two
sizes. Sizes are **full extents**, not half-extents, like every other size in
the tool. Violet in the viewport is the outer box; the red box inside it is
the inner one.

Both are editable directly, as parts of the probe: click either to select it,
drag the violet one to move the pair, and `Ctrl`+wheel to resize whichever is
selected. The red box does not move — it has no centre of its own — so it has
no **Position** row in the inspector and ignores a drag and the arrow keys.
Neither box rotates: a blend region is axis-aligned by construction, as the
capture box is.

An inner size may be **zero** on an axis — a corridor narrower than the fade
simply has no full-strength core — but never larger than the influence size on
that axis, because the runtime divides by `outer − inner` per axis and a
negative width is a gradient pointing the wrong way. A typed pair like that is
**refused** — the pane says so when you leave the field and puts the good value
back — while direct manipulation **clamps** instead, shrinking the inner box as
the outer one closes in on it. The difference is deliberate: a typed number is a
claim to be checked, while a drag that silently stopped halfway would be a tool
fighting the hand.

A probe that has never had them typed derives them from its box: **1.5 m out on
every face, and 1.5 m in**, which is exactly what the runtime falls back to on
its own. A ship authored before these fields existed therefore blends
identically the day it is re-saved. Dragging or scaling the box carries them
rigidly — a move shifts the centre, a resize grows both sizes by the same
**absolute** amount, so the margins the author set survive a room getting 2 m
longer.

#### What a probe deliberately does not see

A probe is a photograph of a room's **fixed** geometry, taken once and worn by
every material in it for the life of the level. Anything that will not still be
standing exactly there is left out of the render list, by behaviour:

| behaviour | why it is left out |
| --- | --- |
| `reflectionProbe: "exclude"` | the authored opt-out, for something fixed that still must not be photographed |
| `dynamic: true` | a rigid body — its authored pose is a starting position, not a fact about the room |
| `liquefiable: true` | it is going to melt; what the probe would record is its shape before the game starts |
| the weapon behaviour | a first-person viewmodel rides the camera, so it is never in the room at all |

The weapon is matched by **name** (`weaponLiquefactor`) rather than by a flag,
because that is the runtime's own contract — `behavior-manager.ts` switches on
`assignment.name` — and a definition body invented here to mirror it would be a
second source of truth that nothing enforces.

The filter lives in `meshesInProbeBox`, which is deliberately the **one** list
both the render list and the digest are built from. Excluding a crate in the
render alone would leave it in the digest, and nudging it would then mark every
probe in its room stale for a capture that could not possibly look different.

### Taking the capture

**Capture all**, in the **Probes…** window, renders every probe whose room has
changed. It runs **in the editor, in the browser**, not in a build step —
Babylon's GGX prefilter and `.env` serialiser are right there, so the file that
comes out is produced by exactly the code the runtime parses it with, and a
probe can be re-taken the moment a room is rebuilt without leaving the tool.

**Capture**, beside it, takes the **selected probe alone** and leaves every
other file in the folder as it is. Naming a probe is an instruction rather than
a question about staleness, so that one is re-rendered whether or not its stamp
still matches — it is what you reach for after nudging a lamp in one room, and
it costs one room's six renders instead of the whole ship's.

It is a **deliberate** act, and that is a change: entering the Runtime view used
to bring every stale probe up to date on the way in. That made the one thing you
reach for to take a quick look at the game's lighting cost a minute of rendering
nobody asked for. What the Runtime view shows now is whatever `.env` files are
on disk — which is exactly what the game would load if it started right now.

Per probe, in `local-environments.js`:

1. a square render target is created at the probe's resolution, with the room's
   meshes as its render list — geometry beyond the box is another room's
   business and would only be visible through a doorway, which is what the
   portal renderer is for;
2. the **six faces are rendered one at a time**, each from its own camera, and
   read back — see below for why one cube target will not do;
3. the six buffers are uploaded as a `RawCubeTexture`, and `HDRFiltering`
   prefilters it for roughness at `FILTER_QUALITY = 1024`. Babylon's OFFLINE
   quality (4096) is meant for a one-off conversion of a sky; a ship has a probe
   per room and they are re-taken on every edit;
4. `EnvironmentTextureTools.CreateEnvTextureAsync` serialises it, with
   `disableIrradianceTexture` — the runtime reads diffuse irradiance from the
   spherical harmonics the serialiser computes anyway, so the irradiance texture
   would be a second copy of the same information at several times the size;
5. `PUT /api/local-environment/<id>?hash=` writes it to
   `export/environments/local_environment_<id>.env`, with the digest beside it.

#### The capture is rendered explicitly

`target.render()`, not `scene.render()`, and the difference is not cosmetic.
**A scene frame does not draw a probe.** `scene.reflectionProbes` is a registry
and nothing walks it; a probe's render target reaches `scene._renderTargets`
only as a side effect of some material in the scene referencing its
`cubeTexture`. A capture references it from nowhere — the whole point is to
write it to a file — so a scene frame allocated the cube, rendered nothing into
it, and serialised six faces of pure zero. Every `.env` in the folder was black,
and every check around it passed: right size, right place, right digest beside
it. A black 1024 probe is still 200 KB.

#### Six separate renders, because clustered lighting is screen-space

The obvious shape for this is one cube render target, or Babylon's
`ReflectionProbe`, which is a wrapper around exactly that. Both draw six faces
from one `initRender` — the pass is set up once and the faces are looped inside
it. **That is the one thing this ship cannot do**, and it is why the first
captures came out black except for the emissive panels.

Aquanova lights the ship with *clustered* lamps, and clustered lighting is a
screen-space algorithm. `ClusteredLightingSceneComponent` registers a tile-mask
render target through the *active camera's* gather stage and tiles that camera's
frustum — the tile size is derived from `engine.getRenderWidth()`. A probe's six
90° frustums are neither the screen nor that camera, so every lookup a face
shader makes lands in a mask belonging to a different view: no lamp is found,
and only the emissive term survives. A cube target sets its lighting up once and
then draws six different views through it, so five of the six faces are wrong by
construction and the sixth only by accident.

Turning clustering *off* for the capture is not an option either, and it is
worth recording why so nobody tries it again. Un-clustered lamps are one uniform
block each, on top of the scene, mesh and material blocks, and WebGL2 guarantees
only `GL_MAX_VERTEX_UNIFORM_BUFFERS = 12` — about **nine analytic lights**. The
ship has sixteen. The shaders do not merely run slowly, they fail to link, which
makes `isReady()` false for ever and hangs the capture on its readiness wait.
Culling lamps by range only got a probe down to twelve.

So the capture drives one **2D** target six times, and each render is a real
frame from the tile mask's point of view:

```js
const target = new BABYLON.RenderTargetTexture(name, size, scene, {
  type, generateMipMaps: false, samplingMode: BILINEAR,
  enableClusteredLights: true,          // tile the face, not the screen
});
target.activeCamera = camera;           // the face camera, per face
...
scene.incrementRenderId();              // not optional, see below
target.render();
```

Three details make it exact rather than nearly right:

| | |
| --- | --- |
| **`enableClusteredLights`** | the `ObjectRenderer` builds the tile mask for *its* camera, inside `onInitRenderingObservable` — before the framebuffer is bound, the only moment at which it is safe |
| **`scene.incrementRenderId()`** | `_updateLightData()` early-returns while `_lightDataRenderId === scene.getRenderId()`, and a 2D render does not bump the id; without it, faces two to six reuse face one's lamp data |
| **a frozen projection** | `getProjectionMatrix()` asks `engine.getAspectRatio()`, which reads the **canvas** at tiling time because the target is not bound yet — a 16:9 frustum against a square face. The camera is given `Matrix.PerspectiveFovLH(π/2, 1, …)` outright |

The result was checked against the truth rather than assumed: the same view
rendered through `scene.render()` and through this path agree to an RMS of
`1e-5`.

#### Which way is up

Cube faces have a convention, and nothing in it is guessable. The capture states
it, in `CUBE_FACES`, as six explicit `{ forward, up }` bases in `.env` face
order — `+X, −X, +Y, −Y, +Z, −Z` — with the poles rolled so `+Y` looks down `−Z`
and `−Y` looks down `+Z`. Babylon's own `ReflectionProbe` gets the poles
inverted here, through a private `_invertYAxis` that has no setter; taking the
cameras by hand is what removes that dependency.

Two smaller traps sit under that:

- `TargetCamera.setTarget` **cannot express straight up or down**. It recovers
  pitch through `Math.atan` and forces roll to zero, which is degenerate on the
  poles. `FaceCamera` overrides `_getViewMatrix()` with a `Matrix.LookAtLHToRef`
  built from the face's own basis, and `_isSynchronizedViewMatrix()` to `false`
  so the override is never cached away.
- `readPixels` returns rows **bottom first** — it is a framebuffer read, and a
  framebuffer's origin is its bottom left — while a cube face wants the camera's
  up on row zero. The `RawCubeTexture` upload is therefore given `invertY =
  true`.

That second one is worth a warning, because it survived a first round of
eyeballing. A vertical mirror leaves the centre of a face fixed, so every
axis-aligned test direction matched perfectly and the poles landed in the right
slots; only sampling at 45° off-axis exposed it, cleanly swapping each reading
with its Y-mirror. Looking at the images did not help either — the corridor's
orange truss is on the *ceiling*, and an upside-down corridor looks entirely
plausible.

It has one more tooth. A texel row flip is an **up/down** mirror only on the
four side faces, whose vertical axis is the world's; on `+Y` and `−Y` it mirrors
world **Z**. So a test that tilts more than 45° off the horizon lands on a pole
face, where a broken build reads exactly the same as a correct one along that
pair — the first version of the check in `e2e.mjs` sampled at 60° and passed a
deliberately mirrored capture. The check now compares each horizontal bearing
tilted 30° up against the same bearing 30° down, which keeps the sample on a
side face, and it is proved by mutation: flip `invertY` back to `false` and the
suite fails.

#### Nothing that is still compiling is allowed through

Babylon **skips** a mesh whose effect is not ready rather than queueing it, so a
face rendered a moment too early has a permanent hole in it — and a hole in a
cubemap is not a visible glitch, it is a slightly wrong reflection for ever.
`waitForRenderList` polls `mesh.isReady(true)` for every mesh in the render list
before each face, and throws rather than proceeding if they are not all ready
inside two minutes. This is also why the render loop keeps running during a
capture: Babylon only compiles shaders while it renders.

The poll waits on a frame **or** a 250 ms timer, whichever comes first. That is
not belt and braces: `requestAnimationFrame` does not fire in a background tab,
and the deadline above is only reached by going round the loop — so on
`requestAnimationFrame` alone, switching away from the editor mid-capture parked
it there for as long as the tab stayed hidden, holding the busy lock with it.

#### Nothing waits for ever

Every long await in a capture is a promise Babylon settles from inside the
render loop, and each one can wait on something that will never happen: a
material whose shader fails to link never reports ready, a read-back on a lost
context never returns, a server that stopped answering never replies. **None of
them throw.**

That matters far beyond the probe, because the capture holds the busy lock. A
stall does not cost you a cubemap, it costs you the editor: the overlay stays
up, every panel stays `inert`, and the only way out is a reload. So each step
runs under a deadline — `withDeadline` for the ones that cannot be cancelled,
an `AbortController` for the two that can — and a stall surfaces as an error
naming the step, which `whileBusy` then unwinds in its `finally`:

| step | how it is bounded |
| --- | --- |
| waiting for the render list | its own poll deadline, with the message saying how many meshes never came ready |
| reading a face back | `withDeadline` |
| `HDRFiltering.prefilter` | `withDeadline` |
| `CreateEnvTextureAsync` | `withDeadline` |
| declaring the probe list, uploading an `.env` | `AbortController` — these can genuinely be cancelled, and an abandoned upload should stop pushing a megabyte at nobody |
| the scene becoming drawable again afterwards | `withDeadline`, logged rather than thrown |

That last one was the one that actually bit. Restoring `applyByPostProcess` at
the end of `withRuntimeCapture` changes a shader define on every material in the
scene, and the `whenReadyAsync` that waits for the recompile is the **last**
thing a capture does — so a single material that would not link showed up as the
final probe hanging on "capturing environment probe … (4/4)" for ever, with the
editor locked behind its own overlay. Two minutes is far longer than the slowest
step measured on the real ship, so none of these can fire on merely slow work.

#### Nothing of the editor gets into the file

A cubemap is a record of **radiance**, not a picture. Four things are held off
for the duration of a capture, and each one is a way the file could otherwise
come out wrong:

| | |
| --- | --- |
| **the lighting** | the runtime's, not the editor's four-light rig — the mode is entered if it is not already up |
| **the ship's own reflections** | nulled, or a capture photographs the cubemaps the last one left and no two runs agree |
| **exposure and tone mapping** | out of the materials, so the runtime is free to expose the result however it likes |
| **the camera** | the capture's own, moved to each capture point |

**Yes — the materials have no environment map at all while a face is rendered.**
`withRuntimeCapture` sets `reflectionTexture = null` on every dressed material
and puts each one back in its `finally`. That is a decision, not an omission,
and it makes a probe a strictly **single-bounce** record: direct light from the
authored lamps, plus whatever the emissive surfaces contribute, and nothing
else. The alternative is a feedback loop. Leave the previous generation's
cubemaps on, and capture two photographs the reflections in capture one, capture
three photographs those, and the ship's rooms brighten a little on every pass
with no fixed point to converge on and no way to tell a re-capture from a
change. Nulling them makes the output a function of the ship alone, which is
what lets the digest downstream mean anything at all.

The cost is honest and small: a mirror-finish panel does not show the room
reflecting itself back. In a corridor of brushed metal at roughness 0.4 that is
not a difference anyone can point at, and the single bounce is what the runtime
does anyway — it hangs one probe on a material, and that probe is this file.

The target is created half-float, and `gammaSpace` is set to `false` on both it
and the cube. `gammaSpace` marks the result as linear; the float format follows
from it — a room lit by a bright lamp carries values well above 1, and an 8-bit
target would clip them to white, which the `.env` serialiser then refuses
outright. The clear colour is **black**, not the editor's backdrop: a hole in a
spaceship shows space, not the tool's own grey.

**Marking the target linear is not enough on its own to keep image processing
out**, which is worth stating plainly because Babylon's own documentation reads
as though it is. `ReflectionProbe` flips `applyByPostProcess` when the target is
*bound* — by which point
every material in the scene has already compiled with exposure and tone mapping
baked into its shader, and the flag changes a shader define, not a uniform.
Measured on the real ship: flipped at bind time, the six faces peak at exactly
`1.0`, tone-mapped and clamped; raised before the shaders are asked for, they
peak at `14.6`. So `withRuntimeCapture` raises it for the whole capture session,
ahead of the readiness wait, and puts it back afterwards.

The camera matters for the same class of reason. The editor's camera drives
level-of-detail selection and culling, so leaving it in place would let *where
the user happens to be parked* decide which meshes a room's cubemap recorded. A
fresh `FreeCamera`, made active by `withRuntimeCapture` and moved to each
capture point by `setCaptureViewpoint`, is what makes two runs from the same
ship agree. It is not the camera the faces are rendered from — each face has its
own — but it is the one the scene's own frames use while a capture is running.

#### The editor is modal for the duration

All four of those are scene-wide state, so a frame drawn in the middle of a
capture is a picture of no state the editor is meant to have: the ship from the
capture point, untone-mapped, with its reflections stripped.

The render loop cannot simply be stopped, though: Babylon compiles shaders only
while it renders, and every face waits on exactly that. So the capture takes the
**busy lock** instead — the same overlay a load uses, which makes the toolbar,
palette, viewport, inspector and every floating tool window `inert` and swallows
the keyboard — and lets the viewport keep drawing whatever the capture's own
state makes of it, behind the overlay. The overlay names the room being taken
and counts through them, via `setBusyMessage`.

The tool windows are in that list because the **Probes…** window is where a
capture is started from, so it is guaranteed to be open for the longest lock the
editor ever takes, with a New and a Delete button on it that would otherwise
still answer the keyboard.

What the lock does **not** swallow is the browser's own keys. `F5` and `Ctrl+F5`
reach the browser whatever the editor is doing, because the state this guard
describes — the tool wedged behind an overlay — is precisely the state in which
you want to be able to reload the page. Claiming every key while busy meant that
a capture that hung took the reload with it, and the only way back was to give
focus to the URL bar and press Enter.

`HDRFiltering.prefilter()` swaps the cube's internal texture and destroys the
original, so the `RawCubeTexture` a capture builds is **single use** whatever
happens next; it is disposed with the target and the face camera in the
capture's `finally`.

### What makes a probe stale

Every capture declares the **whole** authored probe list to the server first,
each with a digest, and gets back the ones that still owe a render. The digest
covers everything the six renders can see and nothing else:

- the probe's own box, capture point and resolution;
- the name, **authored** material and world matrix of every mesh inside the box,
  sorted — Babylon's mesh order follows creation, so an undo that rebuilds the
  same ship in a different order would otherwise read as a change. "Inside the
  box" is `meshesInProbeBox`, so the four excluded behaviours above are absent
  from the digest as well as from the render;
- every authored lamp: type, clustered flag, colour, intensity, range, angle,
  position and emission axis, sorted.

All lamps, not only the room's. A clustered lamp is scene-global by agreement,
so the room next door genuinely can light this one, and deciding which lamps
reach a box is exactly the question the renderer answers — guessing at it here
would be a second, disagreeing implementation. The camera position is
deliberately absent: the capture forces its own lamp scoping, so where you
happen to be standing cannot change the result.

The **authored** material, emphatically. The digest runs inside the Runtime
view, where every mesh is drawn by a stand-in whose material is named after the
probe it resolved to — and that probe only exists once a capture has been taken.
Hashing what is on screen would make the first capture change the state the
second is compared against, and every pass would find every probe stale for ever.

The server compares each declared digest against the `.stamp` file beside the
`.env` and reports the mismatches. Declaring is also what **prunes**: an id that
is not declared has its `.env` and stamp deleted, and nothing else would ever
remove a deleted probe's file from the publish.

**Shift-click Capture all** forces every probe. It works by deleting the stamps
rather than by special-casing the comparison, so there is still exactly one rule
for what is pending — the stamp beside a file has to match the digest declared
for it. A forced probe simply has no stamp, which is also true after a crash
mid-capture, and it means an interrupted force leaves the remaining probes
pending rather than looking done. Use it when something the digest cannot see
has changed: a kit asset replaced on disk, or a fix to the capture itself.

A single-probe **Capture** declares the whole authored list too — that is what
prunes deleted probes and records the digests — but it declares it **unforced**,
and then captures its one probe regardless of what came back as pending. Forcing
the declaration would delete every stamp, which would leave the rooms it did not
render looking stale for a capture nobody asked for.

## The three view modes

| mode | what it shows |
| --- | --- |
| **Editor** | the authoring rig: four lights, a global HDRI, and the Editor exposure/tone pair |
| **Editor unlit** | raw albedo, for reading a very dark prop |
| **Runtime** | the game's lighting: authored lamps only, each room reflecting its own probe |

**Runtime** is not a second file. It dresses the **live authored ship**, which
is what lets you pick, drag, rotate, duplicate, delete and undo while it is up —
there is no "look at it" mode to leave, because the whole point of looking at
the runtime lighting is to find things to fix, and a view you have to leave
before fixing them turns every fix into a round trip.

Entering it drops the editor's rig and its global HDRI. Both are things the game
does not have, and both would sit on top of the authored lighting and hide
exactly what is being inspected. **Env**, **Exposure** and **Tone** each have
their own value for this mode, kept in **Settings ▸ Runtime** and separate from
the **Editor** set beside them.

### Why the ship is drawn twice, and by whom

Every kit mesh in the ship is an **`InstancedMesh`**: a module is loaded once as
a hidden prototype and every placement is a `createInstance` of it (see
`kit.js`). Babylon's `InstancedMesh.material` is a getter onto the source mesh,
and its **setter is a no-op that logs a warning** — an instance cannot carry a
material of its own, because the whole point of instancing is that one draw call
serves them all.

So a probe cannot be hung on an instance. It has to be hung on a source mesh,
and there has to be one source mesh per probe the module appears in. That is
what `dressMeshes()` builds:

- one **material clone** per (authored material, probe), carrying that probe's
  cubemap as its `reflectionTexture`. Kit materials are deduplicated across the
  whole catalogue, so one `MI_Trim_01` serves every room in the ship and there
  is no way to hang two cubemaps on one material;
- one **prototype clone** per (kit prototype, probe), wearing that material,
  disabled and unpickable. `Mesh.clone` shares geometry — Babylon refcounts it —
  so a prototype costs a draw-call bucket and no memory;
- one **stand-in instance** per authored mesh, off the right prototype,
  **parented to the authored instance at identity**. It therefore inherits the
  authored world matrix, the authored enabled state, chunk isolation, hiding and
  the veil, all for free and with no per-frame sync.

The authored instance is then made **invisible, not disabled**: disabling it
would take the stand-in hanging off it down too, and would take the element out
of the editor's own picking predicate. The authored ship stays enabled, stays
pickable and stays exactly where it was — only the pixels come from somewhere
else.

A stand-in is marked `metadata.runtimePreview`, and `isRuntimeStandIn()` is what
everything walking an element's meshes filters on — the exporter's primitive
list above all, exactly as `isVeilClone()` and `isGizmoMesh()` are filtered
beside it. A stand-in that reached `artMeshes()` would take a primitive index
and be written into the ship.

A ship mesh that is **not** an instance is dressed the old way, by swapping its
material. Nothing in the kit produces one today, but the two paths are one
`sourceMesh` check apart and the alternative is a mesh that silently renders
with the editor's HDRI.

### Following the ship

`syncElementEnvironments()` re-resolves every element's probe once per frame, off
`onBeforeRender` rather than off an edit event: dragging a crate through a
doorway has to move it onto the next room's cubemap the moment its bounds cross.
The comparison is against the probe the element already has, so a still ship
costs one bounds union per element and no churn at all. A changed probe means a
changed **source mesh**, not just a changed material, so that element's
stand-ins are thrown away and re-made off the right prototype — a handful of
instances of already-loaded geometry, on the one frame it actually crosses.

Adding, deleting or re-kitting an element brings in meshes the preview has never
seen, so `redressPreview()` runs off the `placements` event and rebuilds the
whole dressing. Rebuilding rather than patching is deliberate: the prototypes and
clones are keyed by probe and source, so an unchanged ship rebuilds exactly the
handful it already had, and a patch would have to reimplement every rule in
`dressMeshes()` to decide what to keep.

`undressMeshes()` is the exact inverse, and its order matters: the authored
meshes are visible again and back on their own materials **before** anything the
preview made is freed, so nothing is left pointing at a disposed material or
standing invisible with nothing drawn in its place, even for a frame.

> **`Material.clone` deep-copies every texture slot.** Babylon's `CopySource`
> runs `sourceProperty.clone()` on each one, so every material clone owns a
> private wrapper for the ship's base colour, normal and ORM maps that nobody
> else can see. Left behind they leak a full set of the ship's textures on every
> mode switch. `dressMeshes()` diffs `getActiveTextures()` against the source to
> find them: a texture the clone *shares* rather than owns — `CopySource` passes
> render targets straight through — is in both sets and is left alone. The e2e
> counts `scene.textures` either side of a switch.

### The lamps

The preview builds the runtime's lamps from `state.lights`, the editor's **live**
record, so an inspector edit is on screen on the next frame and nothing here
reads a file. This is `demos/aquanova/lights.ts` rebuilt on Babylon.js, and the
two are kept deliberately parallel — same −Y emission axis, same
clustered-versus-scoped rule.

The scoping rule is the runtime's: a **clustered** lamp lights the whole ship, a
scoped one only the meshes of the chunk nearest the camera. Materials are
compiled with `maxSimultaneousLights = MAX_PREVIEW_LIGHTS = 4`, and the
clustered container counts as **one** slot however many lamps it holds — so four
is one cluster plus three scoped lamps, which is what a room of this ship
actually lights with. Babylon silently drops the rest, so this is a ceiling
worth knowing about.

Edits are **poked into the existing light**, and only a change of *shape*
rebuilds it. A slider emits on every tick, and disposing a light dirties every
material it touched, which is a shader recompile per frame for a number.
`lightSignature()` is the line between the two: type, `clustered` and the mesh
scope pick the constructor, which of the two lighting paths draws the lamp, and
`includedOnlyMeshes` — none of which can be changed after the fact — and
everything else is a scalar. A lamp is a
child of the element it rides, so a drag carries it; a lamp left behind would
light the place the fitting used to be.

A lamp with `type: "none"` creates nothing, which is what a fitting that is
purely decoration wants.

## Publishing

`lab/public/aquanova/scripts/sync-ship.ts` copies what the editor wrote into the
demo:

```
pnpm tsx lab/public/aquanova/scripts/sync-ship.ts
```

It takes `export/ship.glb`, `export/ship_manifest.json`,
`export/ship_collision.json` and the whole of `export/environments/`, and writes
them under `lab/public/aquanova/`. The `.stamp` beside each `.env` is what lets
it refuse a half-captured folder: a probe whose stamp does not match the digest
in `local-environments.json` has not finished being taken, and publishing it
would ship a cubemap of a room that no longer exists.

`export/environments/local-environments.json` is **generated, never authored**,
which is why it is git-ignored. Every capture rewrites it from scratch out of
the probe list the browser declares, so a value hand-typed into it survives
exactly until the next capture. It records what was *taken* — the `.env` file,
its digest, its byte count, and the box that was rendered — and it is in
**editor space**, because that is what the browser sent.

The authored truth is `export/ship_manifest.json` → `environmentProbes[]`, in
glTF space. `sync-ship.ts` therefore reads the geometry out of the manifest and
only the file facts out of the generated index, so the published
`lab/public/aquanova/local-environments.json` is glTF space throughout, which is
what the runtime's `toLite` conversion expects.

## Testing

```
npm test           # spins up a private server and runs every suite
```

The runner starts its own server instance on port 5199 pointed at a **throwaway
export directory**, runs the three suites against it, then deletes it. A test
run therefore cannot touch a real ship — earlier the suites saved and exported
straight into `export/`, which would have overwritten whatever you were working
on. `SHIP_EXPORT_DIR` and `SHIP_PORT` override `config.json` if you want to
point a server anywhere else. `SHIP_TEST_KEEP=1` leaves the throwaway directory
behind instead of deleting it, which is the only way to look at the `.glb`, the
manifest and the captured `.env` cubemaps a failing run actually produced.

Individual suites can still be run by hand, but they will not guess a server:

```
TOOL_URL=http://localhost:5199/ node test/smoke.mjs      # catalogue, materials, markers, manifest, export
TOOL_URL=http://localhost:5199/ node test/interact.mjs   # ghost, drag, hover, wheel, camera, keyboard
TOOL_URL=http://localhost:5199/ node test/e2e.mjs        # clean-state build, save, probe capture, Runtime view
```

They have to run in that order when run by hand: `e2e.mjs` builds a ship, saves
it and then captures its probes from what it just built, so it needs the export
directory the two before it left behind.

> **`TOOL_URL` has no default, and 5180 is refused outright.** It used to
> default to 5180 — the port the editor runs on for real work — so running a
> suite by hand pointed it straight at the ship being built. The suites are
> destructive: they call `clearAll()`, place and delete elements, change
> settings and let the auto-save tick fire. Doing it once wrote a one-instance
> manifest over `ship_autosave.json` and an empty hull set over
> `ship_collision.json` (73 modules and 163 shapes, gone), recoverable only
> because the server keeps timestamped backups of every write.
>
> `test/target.mjs` now refuses to start without `TOOL_URL`, and refuses 5180
> even when named unless `TOOL_URL_I_MEAN_IT=yes`. The convenience of a default
> was worth nothing against that.

Drives installed Edge through the Playwright already present in the Babylon.js
checkout, so nothing extra is downloaded. Between them they cover boot, palette
population, material/texture de-duplication, placement, the ghost workflow
(arm, snap, rotate, scale, place), dragging (single, multi-selection, snapped
delta, cancel), the hover outline, camera navigation, grid elevation and axis
cycling, door markers, portal and adjacency derivation, the collision staging
area, the compound bench (save, place, group selection, drill-in, break apart,
delete), manifest round-trip, save rotation and .glb export.

> `page.mouse.move()` in a single jump can be coalesced away and never reach the
> hover path — the tests pass `{ steps: 4 }`, which is also what a real mouse
> does.

## Nothing deletes your work

The server has no `unlink`/`rm` call anywhere: it only ever writes, copies,
renames and creates directories. There is no bundler and no `node_modules` —
`npm start` is literally `node server.mjs` — so there is no "clean the dist
folder" step of the kind Vite or webpack would bring. Files in `export/` are
removed only if you remove them.

### Auto-save

Every couple of minutes — the interval is in the **Settings** pane, and `0`
turns it off — the editor writes `ship_autosave.json` beside the ship, but
**only when something has changed** since the last save or auto-save.

Three deliberate choices:

- **It is not the manifest.** A background write must never overwrite the ship
  you last chose to save. Recovery is a copy you reach for, not a thing that
  happens to your work.
- **Every one is kept.** Each write rotates the previous copy to a timestamped
  name, so a long session leaves hundreds of small files. That is the point: the
  value of an auto-save is having the state from _before_ whatever went wrong,
  and you cannot know in advance which one that is. They are a few kilobytes
  each, and deleting them is one command.
- **It keeps its own baseline.** An auto-save does _not_ clear "you have unsaved
  work", because the manifest still does not have those changes. Sharing one
  baseline would mean a background write quietly disarmed the guard that stops
  you closing the tab on an hour of work — the exact opposite of what an
  auto-save is for.

To recover, copy `ship_autosave.json` (or any of its timestamped predecessors)
over `ship_manifest.json` and reload. The file names itself
`"generator": "SciFiShip layout tool (auto-save)"` and carries `"autoSaved":
true`, so it is never mistaken for a deliberate save.

### Why every write is rotated

`ship_manifest.json`, `ship_collision.json` and `ship_autosave.json` all keep a
timestamped copy of what they replaced.

The collision file was briefly exempted, on the reasoning that every manifest
carries the same hulls in `moduleShapes`, so a lost copy could always be
rebuilt. **That reasoning is only as good as its source.** It was overwritten
with an empty file once, and the manifest had been emptied in the same breath —
so the derived-from argument was worth nothing, and the timestamped copies were
the only thing that got the work back.

Storage is cheap and these files are kilobytes. Recovering an afternoon is not.

## Configuration

`config.json`:

```jsonc
{
    "kits": {
        "source": "local", // "local" (BabylonAssets checkout) or "online" (the CDN)
        "localDir": "../../../../../../../../BabylonAssets", // the checkout
        "onlineBase": "https://assets.babylonjs.com/",
        "prefix": "kits", // where the kits sit inside either one
        "folders": ["Modular SciFi MegaKit", "Sci-Fi Essentials Kit"], // palette order
    },
    "exportDir": "../export", // ship.glb + ship_manifest.json
    "envDir": "../env", // HDRI, served at /env
    "port": 5180,
}
```

**Every path is resolved against `config.json` itself, not the working
directory**, so the relative defaults keep pointing at this copy's own folders
wherever the editor is cloned or moved to. An absolute path still wins, for
anyone holding BabylonAssets somewhere else.

### Where the kits come from

The kits are **not** bundled with the editor. They live in the BabylonAssets
repository, whose root maps 1:1 onto `https://assets.babylonjs.com`, and
`kits.source` picks which of the two the browser loads modules from:

- **`local`** — the checkout named by `localDir`, mounted read-only at
  `/assets/`. The app is then same-origin with its own assets: no second web
  server to start, no CORS.
- **`online`** — `https://assets.babylonjs.com/kits/...` directly.

The two use the *same paths*; only the origin differs. `SHIP_KITS_SOURCE` and
`SHIP_ASSETS_DIR` override both, so a run can be flipped without editing the
config.

`localDir` has to be present **even in online mode**, because the catalogue is
built by listing the folder — a static CDN cannot be asked what files it holds —
and because the thumbnailer renders from it.

`folders` is palette order, not a whitelist: a kit dropped into `kits/` shows up
without a config edit, after the ones listed here.

### The palette shows one kit at a time

Any number of kits can be installed, but the palette displays **exactly one**,
chosen in the combo box above the search field. There is no "All kits" entry,
and that is a decision rather than an omission: the category strips of unrelated
packs do not merge into anything meaningful — `Walls, Platforms, Columns, Props,
Decals, Aliens, Guns, Enemies, Rocks, PineTrees, Potions, Axes` names nothing you
would look for, and "Props" would then mean sci-fi crates *and* pirate barrels in
the same tab. Nothing is lost: picking a kit is one click, and search already
spans the modules of the kit you are in.

The tabs are therefore **that kit's own categories**, rebuilt when the kit
changes. If the new kit has no tab matching the active one, the selection falls
back to **All** instead of showing an empty list.

The choice is kept in `localStorage` under `paletteKit` — it is a view
preference, like the folded panes, not ship data, so it must not touch the
manifest. A stored kit that is no longer installed is discarded on load and the
first kit is used.

### The three layouts, and the one to aim for

- **Category subfolders** — `Walls/`, `Platforms/`, … The folder *is* the
  category. **Every kit in BabylonAssets is arranged this way**, because it is
  the only layout where the palette tabs, and therefore the module ids, are a
  decision someone made rather than a by-product of how the models happened to
  be named.
- **A flat root** — the grouping is in the filename (`Gun_Pistol`,
  `Enemy_Raptor`, whose prefixes become the `Guns` and `Enemies` categories).
- **A format wrapper** — one folder named after the file format, `glTF/` or
  `FBX/`. That is packaging, not a category: an `FBX` tab holding the entire
  kit would say nothing. Those names are listed in `FORMAT_DIRS` and read as a
  flat root one level down. The list is wider than what the editor can load —
  `Blend/`, `OBJ/`, `Source/` — so a pack shipping several formats is read
  once, through the best one it offers, rather than appearing to be foldered.

The last two are how packs arrive from Quaternius, and the reason they are read
at all is that **a pack should be usable the moment it is dropped into
`kits/`** — before anyone has looked at it. It is not where a pack stays:
copying a pack in means sorting it. The Pirate Kit came as `glTF/` and is now
`Characters/ Enemies/ Environments/ Props/ UI/ Weapons/`; the Nature and RPG
packs came as `FBX/` and are now a single `Nature/` and `Items/`, because
neither has any grouping worth a tab strip; the Essentials Kit came flat and is
now `Enemies/ Guns/ Props/`, which is exactly what its filenames were already
saying, so not one module id changed.

A derived category has to earn its tab: a filename prefix becomes one only when
**at least three** modules share it (`MIN_DERIVED_CATEGORY`), otherwise the
module lands in **Other**. In several packs the underscore marks a *variant*
rather than a group — `Potion1_Empty`, `Potion1_Filled` — and taking every
prefix at face value produced a strip of one-tile tabs named `Potion1`,
`Potion2`, `Potion3`. Names are pluralised on the way in (`Axe` → `Axes`,
`Bush` → `Bushes`, `Enemy` → `Enemies`, `Characters` left alone). None of this
applies to a foldered kit: there the folder name is the tab name, verbatim,
which is why the Pirate kit's tab reads `UI` and not `UIs`.

A module id is **kit-qualified** —
`Modular SciFi MegaKit/Walls/ShortWall_Band2_Straight` — so two kits may each
hold a `Props/Crate` without one shadowing the other. That id is what the
manifest stores, so it is a permanent contract rather than a display string:
sort a kit's folders *before* building with it, because moving a model between
categories afterwards renames every instance of it in the ship. Sorting a flat
kit into the folders its filenames already named is the one free case, and it
is why the Essentials Kit could be foldered after the fact.

### `.fbx` kits, and the unit they are modelled in

Several Quaternius packs ship `.fbx` and no glTF. Babylon's loaders bundle
registers an FBX plugin itself, so nothing had to be added to load them — but
**it ignores the file's `UnitScaleFactor`**. The loader parses the value into
its `GlobalSettings` and then never reads it again, and exposes it on nothing:
container, scene and root metadata all come back `null`.

That is not a rounding matter. FBX's `UnitScaleFactor` is *centimetres per
unit*, and Quaternius exports at the Blender default of `1.0`, so a tree that
should be 2.48 m tall arrives **248 units** across and swallows the ship.

`fbxMetresPerUnit()` reads the value out of the bytes — the length-prefixed name
token in a binary file (matched with its prefix so `OriginalUnitScaleFactor`
cannot be picked up instead), a regex in an ASCII one — and defaults to `1`, the
FBX default, when the file says nothing. `loadModuleContainer()` fetches the
file once, hands the same buffer to the loader as a `File`, keeping `rootUrl`
intact for relative textures, and scales the container's root nodes by
`unit / 100`. Both the palette prototype and the thumbnailer go through it, so a
tile and a placed module can never disagree about scale.

### Packs exported flat, and `kits.json`

Those same `.fbx` packs are exported with **every face flat**. It is not the
loader: `CommonTree_1`'s own `LayerElementNormal` holds one normal per corner
and no two at a vertex agree — all 1450 of its vertices are split, by up to
159° — and the mesh Babylon builds matches the file exactly. Quaternius' own
renders are smooth-shaded, and a faceted trunk beside a smooth-shaded ship reads
as broken rather than as stylised.

`smoothMeshNormals(mesh, maxAngleDeg)` puts it right, and it is **Blender's
auto-smooth**, edge-based, for two reasons that both have a wrong answer:

- averaging every normal that meets at a point would round off the corners of a
  crate;
- clustering by angle to the first normal seen would leave an eight-sided trunk
  faceted, because its far side is 180° from where the cluster started.

So an *edge* is smooth when the two faces sharing it are less than the threshold
apart; corners are joined across smooth edges and each group takes its faces'
area-weighted average. Smoothness chains, so all eight sides of the trunk end up
in one group and shade as a cylinder while the cap stays a cap. Only the normal
buffer is rewritten — vertex count, indices, UVs and skinning are untouched, so
instancing and the UVs cannot tell the difference. Vertices
are welded by position, so a seam split only to carry a second UV does not show
as a shading crease.

It is **opt-in per kit**, in `public/data/kits.json`:

```json
"Ultimate Nature Pack": { "smoothNormalsBelowDeg": 60 }
```

A kit not named there is loaded exactly as it ships, which is every glTF kit —
smoothing the sci-fi kits would change every ship already built from them.
60° is the threshold because these packs' facets sit at 45°, which has to
smooth, while a sword's bevels and a rock's corners sit at 75° and over, which
has to stay hard. `CommonTree_1` goes from 1450 split vertices to 92.

> It runs inside `loadModuleContainer()`, beside the unit fix and for the same
> reason: the ship and the thumbnail scene must never disagree about how a
> module looks. Thumbnails cached before a threshold changes are stale — drop
> the kit's files from `cache/thumbs` and `cache/turntable`, or bump
> `THUMB_VERSION` if the change touches every kit.

### The values an export dropped, per kit

`kits.json` also carries **material values a pack lost on the way out of
Blender**. In the RPG pack, `Glass` and the five `Liquid_*` materials are the
only ones that reach Babylon with no colour at all — 53 of them, against 214
that carry one — and they are exactly the see-through ones. Blender's FBX
exporter writes Phong properties out of a Principled BSDF and has nothing to
write for a transparent shader, so it wrote no property block, and Babylon fell
back on its default 0.8 grey.

The symptom was that **a filled potion looked identical to an empty one**: the
liquid, the glass and the air between them were all the same grey, which reads
as an empty bottle.

```json
"Liquid_Red": { "tint": [0.640, 0.067, 0.045] },
"Glass":      { "tint": [0.640, 0.680, 0.720], "alpha": 0.3 }
```

The colours are the pack's own palette — the opaque material of the matching
hue, times the `0.8` `DiffuseFactor` the loader applies to those — so a red
potion is the same red as a red gem. The liquids stay opaque and the glass
blends, which also gets the draw order right for free: the liquid goes out in
the opaque pass, the bottle over it in the transparent one.

Authored **per kit**, not by bare material name, for the reason the dedupe key
learned the hard way: `Glass`, `M_Glass` and `MI_Trim_01` all mean different
things in different packs. Applied on the way out of `loadModuleContainer()`,
so the material is in its final state before either cache takes its key.

### Where a foldered kit keeps its textures

The MegaKit ships its modules in `Walls/`, `Platforms/`… but every one of them
names its textures with a **bare filename** (`T_Trim_01_ORM.png`), and those
textures sit one level up, at the **kit root**. So the loader, resolving the URI
next to the .gltf, asks for `Walls/T_Trim_01_ORM.png`, which is not there.

The obvious repair — rewriting the URI to `../T_Trim_01_ORM.png` — is not
available. glTF forbids a URI from leaving its own directory and Babylon
enforces it (`GLTFLoader._ValidateUri` rejects any `..`), so a file rewritten
that way fails to load outright:

```
RuntimeError: Unable to load … /images/0/uri: '../T_Trim_01_Normal.png' is invalid
```

Copying the atlas set into each of the six folders would work, and would put
99 MB on the CDN to say the same 27 MB six times.

So the textures are left **exactly where Quaternius puts them** — which also
keeps refreshing a kit a straight copy — and the *loader* is told the
convention instead. `scanKit` lists the image files at the kit root and the
catalogue carries them per kit; `kit.js` installs a `preprocessUrlAsync` on the
glTF loader that moves a request for one of *those* filenames from *that* kit's
model folders up to the kit root. Nothing else can be caught by it, so the
`.bin` beside each .gltf keeps resolving normally, and a kit whose models sit
at its root (textures already beside the .gltf) gets no rule at all.

This is why sorting a flat kit into folders costs nothing at load time: the
Essentials Kit's atlases stayed at the root when its models moved into
`Enemies/ Guns/ Props/`, and the same rule that serves the MegaKit now serves
it. The Pirate kit works the same way — six category folders,
`Atlas_Pirate.png` alone at the root.

The editor's own server used to paper over this by falling back to the kit root
on a 404. That works for a folder it is serving and not at all for a CDN, which
is the whole reason the rule now lives in the loader.

#### If every kit suddenly 404s its textures, restart the server

The server reads `public/` from disk on every request, so edits to the client
JS are live — but `server.mjs` itself is only read at startup. A dev server left
running for days therefore answers a **new page** with an **old catalogue**.

That skew is invisible except here. The redirect is keyed on the per-kit
`modelDirs` / `rootTextures` the catalogue carries, and a kit missing them is
skipped rather than guessed at — so a catalogue built before those fields
existed installs *zero* rules and every kit loses its atlases at once. The
symptom (`GET …/Props/T_Props_Batch1_Normal.png 404`) reads as a broken kit, not
as a stale process.

`assertCatalogueShape()` in `kit.js` runs on every `loadCatalogue()` and turns
that into a message naming the missing field and the cure. Both it and a
turntable render of a foldered module — the path the failure was first seen on —
are covered in the interact suite.

> The kits are CC0, but the folders in BabylonAssets also contain models that
> come with Quaternius' PRO subscription. **Get the kits from
> [quaternius.com](https://quaternius.com/) rather than copying them out of
> BabylonAssets** — see the `license.txt` in each kit folder.

`SHIP_EXPORT_DIR` and `SHIP_PORT` override `exportDir` and `port` — that is how
the test runner keeps itself away from real data.
