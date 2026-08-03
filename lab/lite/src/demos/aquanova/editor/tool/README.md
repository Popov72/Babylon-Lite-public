# Ship layout tool

A small standalone editor for assembling the **Quaternius Modular SciFi MegaKit**
into a chunked, portal-ready ship. It does one job: pick a module, place it on a
grid, move/rotate/scale it, and say which chunk it belongs to. No asset
authoring, no lighting, no gameplay.

```
cd tool
npm start          # -> http://localhost:5180
```

Node 22+, no dependencies, nothing to download: the kit ships with the editor
(`../kit`, 35 MB, CC0 — see `../license.txt`). Everything else comes from the
Babylon.js CDN.

---

## Why this exists instead of an off-the-shelf tool

Godot 4 + the *Simple Asset Placer* plugin, the kit's own Unity project and
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
    { "id": "P0001", "module": "Walls/ShortWall_Band2_Straight", "node": "weapon locker",
      "chunk": "CH00_Storage", "position": [0,0,0], "rotation": [0,90,0], "scale": [1,1,1] }
  ],
  "markers":  [ /* doors, see below */ ],
  "chunks":   [ { "id": "CH00_Storage", "node": "CHUNK_CH00_Storage", "aabb": {...} } ],
  "portals":  [ { "id", "chunkA", "chunkB", "door", "centre", "normal", "corners" } ],
  "doors":    [ { "id", "chunkA", "chunkB", "position", "triggerRadius", "leaves" } ],
  "adjacency": { "CH00_Storage": [ { "to": "CH01_CorridorA", "portal": "Portal_Door_D00" } ] },
  "view":     { "position": [...], "rotation": [...], "target": [...] },
  "environment": { "strength", "toneMapping", "exposure" },
  "editorEnvironment": { "strength", "exposure" },
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

glTF *meshes* stay unnamed, and that is not an oversight: the exporter shares
one glTF mesh between every instance of a module, so a mesh entry belongs to the
module, not to any one element. The **node** is the only per-element name slot,
and it is the name Babylon gives the node when the .glb is loaded back.

### Every exported node carries its identity

Because the name is deliberately not unique, it cannot be an identity. Each
placement's glTF node therefore also carries an `extras` block:

```json
{ "id": "P0192", "module": "Props/Prop_Crate4", "chunk": "CH00_Storage" }
```

Written by putting `metadata.gltf.extras` on the node for the duration of the
export — Babylon's serializer reads exactly that address by default
(`metadataSelector: (m) => m?.gltf?.extras`), and it is put back afterwards so
the editor's own `metadata.placement` is untouched.

**Both loaders hand it back at the same address**, `mesh.metadata.gltf.extras`:
Babylon.js through its `ExtrasAsMetadata` loader extension, and **Babylon-Lite**
through `gltf-feature-extras.ts`, whose detector explicitly includes
`json.nodes.some(n => n.extras !== undefined)` — so node extras are enough to
pull the feature module in, and it assigns to the node *and* the mesh.

> **This is what makes a mesh removable.** Liquefying a prop takes its mesh out
> of the scene, and its Havok body has to go with it — which needs the runtime
> to know *which* body that is. Matching on the node name cannot do it: this
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
*new* ship starts from, but a saved one carries the list it was authored
against, and letting the two drift apart would silently repoint its behaviours.
An undo snapshot carries no list, so undoing never disturbs the one in force.

**Definition bodies are free-form JSON, written through untouched.** The runtime
owns which flags exist; a tool that normalised the ones it happened to know
about today would quietly drop the rest, and would need editing every time one
was added. So the dialog is a name and a textarea, and the only thing checked is
that the text parses to a JSON *object* — an array or a bare number there would
be silently ignored by the runtime rather than rejected.

**Edit behaviours…** in the inspector opens the library: pick from the list,
edit, `New`, `Save`, `Delete`. Two operations keep the file consistent by
themselves:

* **Renaming** a definition rewrites every entity that referenced it. A rename
  that left them pointing at the old name would silently drop their behaviour.
* **Deleting** one strips it from every entity that carried it, rather than
  leaving entries the runtime would ignore.

**Attaching** happens on the selected element, keyed by its **node name** — so
the panel shows how many elements that name governs (`"crate" — 4 elements`).
Attaching one can be affecting one crate or thirty, and there is no other way to
tell. A name is required first: a behaviour with nothing to key on could never
be matched to an element.

**`linked`** appears only for definitions with `liquefiable: true`, because
linking is for pieces that melt as one — a pair of door halves. The candidates
are the other named nodes **in the same room**: linked pieces are neighbours in
practice, and a ship-wide list would be hundreds of entries long. The list is
de-duplicated, drops the node itself, and is omitted from the manifest when
empty — the absence is what "this one stands alone" means.

**`excludeSDF`** sits beside it, on the same liquefiable-only gate, and names
the nodes whose baked SDF the fluid simulation must drop while that shot's water
is alive. A door names its frame: the water is seeded inside the door's own
volume, which sits *within* the frame, so leaving the frame in the SDF union
would eject the particles and dam the opening the door just left behind.

Its candidates are the **dynamic** nodes in the room, not every named one — a
node with no rigid body has no SDF in the simulation to drop, so offering it
would only be a way to get it wrong. "Dynamic" uses the runtime's own rule:
`dynamic: true`, or `liquefiable: true`, which implies it.

**`direction`** is the way an entity faces — what a start position needs — and
rides the applied entry beside `name`:

```jsonc
{ "name": "player_startpos", "direction": [-1, 0, 0] }
```

It is **optional on every applied behaviour**, so the three fields always
appear. Gating them on the definition declaring a `direction` key was the first
design and it was wrong: it made an optional parameter invisible until you knew
to declare it, which is exactly the thing the person editing does not know. A
definition that *does* name a `direction` still supplies the starting value, but
nothing is written until the entity says so.

Stored **as typed, not normalised**: normalising on every commit fights you as
you fill the three fields in — typing `1` into Y after X would turn both into
`0.707` before you reached Z. An all-zero vector is dropped rather than written,
since it names no direction at all.

> A hand-edit that splits the parameter into a sibling entry of its own —
> `[ { "name": "player_startpos" }, { "direction": [...] } ]` — is **folded back
> into the entry above it** on load, with a console warning. An entry with no
> `name` means nothing to the runtime, so that is the only reading under which
> it means anything, and dropping it would silently lose the edit.

> A manifest written before `entities` existed keyed `behaviors` by *node* name,
> so each entry meant "this node has these flags". Loading one keeps the bodies
> as definitions **and applies each to the node it was named after** — keeping
> them without the application would silently un-liquefy the ship. Detected by
> the `entities` key being absent rather than empty, since `serialize()` always
> writes both.

### Environment

```jsonc
"environment": {                // what the DEMOS read
  "strength": 1.7,              // scene.environmentIntensity
  "toneMapping": "Khronos PBR Neutral",
  "exposure": 0.55              // the linear multiplier, exactly as the slider shows it
},
"editorEnvironment": {          // this tool only — the demos must not read it
  "strength": 1.5,
  "exposure": 0.55
}
```

**Two pairs, because there are two pictures.** The editor's authoring rig adds
four analytic lights the game does not have, so one pair of Env/Exposure values
cannot serve both: what reads well while building is nothing like what the game
needs. The **Runtime light** checkbox switches which pair the sliders edit, and
each keeps its own values, so flipping between them never costs you a setting.

`environment` is the pair tuned with **Runtime light on** — the picture the
demos render — and it is what `aquanova` and `liquefactor` read.
`editorEnvironment` is the other one, filed separately because it describes this
tool and not the ship.

**`exposure` is the plain linear multiplier**, used identically at both ends:
what the slider shows is what `scene.imageProcessingConfiguration.exposure` gets
here and what `scene.imageProcessing.exposure` gets there. It used to be stored
in Blender *stops* and raised to a power by the runtime, which made a
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

So the **Runtime light** checkbox silences the rig, leaving the HDRI alone, and
the toolbar checkbox says which mode you are in — "is this what the game shows?"
should never be a guess. It is the only honest preview, and the only sound way to set the
`environment` pair, which *is* what the demos read. The authored intensities are
remembered, so switching back restores them exactly, along with the editor's own
`Env`/`Exposure`.

`export/ship.glb` is a **derived artefact** and is never read back — one file
with every chunk as a named `CHUNK_<id>` parent node. Reloading from a .glb
would be lossy, because a baked mesh no longer knows which kit module it came
from; **Load** re-instantiates from the kit folder using the manifest alone.

### Handedness: the manifest speaks glTF, the editor does not

The editor is a **left-handed** Babylon scene; glTF is **right-handed**, and the
exporter mirrors X on the way out. Measured: an element the editor holds at
`[7, 3, 5]` lands in the .glb at `[-7, 3, 5]`. The runtime's loader mirrors it
back, so the geometry round-trips — but the *manifest* sits beside the .glb and
describes it, and the runtime reads it as glTF space, negating X in a dozen
places: colliders, portal openings, room membership, the player's facing.

So the split is by **who reads the field**:

| written mirrored (glTF space) | left in editor space |
|---|---|
| `chunks[].aabb` — min/max swap with the flip | `instances[]` |
| `portals[].centre` / `normal` / `corners` | `markers[]` |
| `doors[].position` / `direction` | `colliders[]` |
| `collision[chunk]` | `moduleShapes[]` |
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
> - **`moduleCollision` is a *local* transform in glTF space.** Converting it
>   back needs the **rotation as well as the centre** — `[-x,y,z]` for the
>   point, `[-x,y,z,-w]` for the quaternion, because mirroring flips the
>   handedness of the turn too. Centre-only conversion leaves a turned box
>   mirrored, which looks correct on anything symmetrical.
>
> Mirroring `instances[]` as well would make the runtime-facing half uniform,
> but it would not make the *file* uniform: `colliders`, `moduleShapes`,
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
section above — the point is that *everything* the runtime resolves goes
through the node name.

The renames last only for the duration of the export; inside the editor
elements keep their `P0007` node names.

Every save moves the previous manifest aside as
`ship_manifest.<YYYYMMDD-HHMMSS>.json` in the same folder. The manifest is the
one thing in the project that cannot be regenerated, and the files are a few KB
each, so the history is kept in full rather than rolled. The server also keeps a
one-time `*.blender.bak.*` copy of any manifest or .glb it did not write itself,
so switching over from the Blender pipeline is safe.

### Loading locks the editor

A load rebuilds the scene one module at a time, so there is a long window in
which the ship is half there. An edit landing in that window acts on a scene
that does not exist yet — it can select an id that is about to be recreated,
drag a wall a frame before it is disposed, or push an undo snapshot of a
half-restored ship. None of that is recoverable, so input is shut off outright
rather than defended against case by case.

Two layers, because either alone leaks:

* **An overlay** covering the window, with the message and a spinner. It says
  what is happening and it swallows every pointer event.
* **`inert` on the four panels**, which takes them out of hit-testing *and* out
  of the focus order in one attribute — an overlay alone would not stop a `Tab`
  into a toolbar select, or a keyboard shortcut. The keydown handler bails too.

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
follows, the mouse cursor is hidden because the module *is* the cursor.

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

**Every transform reads the same way.** The bare letter picks the *axis*, `Shift`
walks that setting's *value* and `Ctrl` walks it back; the *edit* is a gesture,
not a letter.

| | axis | value | value back | the edit itself |
| --- | --- | --- | --- | --- |
| move | `V` (and `Y` for the space) | `Shift+V` | `Ctrl+V` | drag, `M`, arrow keys |
| turn | `R` | `Shift+R` | `Ctrl+R` | `Shift`+wheel (`Alt` too: about a shared pivot) |
| scale | `F` | `Shift+F` | `Ctrl+F` | `Ctrl`+wheel |
| mirror | *uses the scale axis* | — | — | `Alt+F` |

It was not always so: the letters used to carry the actions (`R` turned, `F`
mirrored) with the settings behind the modifiers, which left `V` reading one way
and `R` and `F` another for no reason anyone could give. Moving the two edits
onto the wheel — where the third already was — freed all three letters to mean
the same thing.

| | |
|---|---|
| Place | click a palette tile to arm it, then click in the viewport. The module stays armed for repeat placement — except on the collision bench, where it is a one-shot. |
| Move | **drag** an element (elements stay solid, button held), or **`M`** to pick the selection up and carry it hands-free as a translucent ghost — click to drop, `Esc` to put it back. Dragging one that is already selected moves the **whole selection**; dragging an unselected one selects just it first. **`V`, or the `Drag` combo,** cycles the drag axis: `X/Z (floor)` → `Y (up/down)` → `X only` → `Z only` — safe to change mid-drag. **`Y`, or the combo beside it,** says whose axis that is: `World` or `Local` (the element's own — so a wall turned 90° still slides along its length, and `R` turns it about its own axis). `Esc` or right-click mid-drag puts everything back. |
| Frame | **double-click** an element |
| Axes | **`X`** — show one element's **world** X/Y/Z arrows · **`Shift+X`** — its own **local** axes, which is what scaling acts on. Showing them **also puts moving and turning in that space**, since asking to see an axis is nearly always asking to work along it; `Y` overrides afterwards. With several selected, the one **nearest the cursor** gets them; an armed ghost counts too. They follow a single click to the next element, **keeping their flavour**. The same key again hides them (without touching the space), the other key re-aims them, and pressing either with nothing selected or hovered hides them |
| Axis modes | see the table above — one letter per transform, the same three modifiers on each. All three `Ctrl` pairs are claimed from the browser: reload, the find bar and paste |
| Mirror | **`Alt` + `F`** — mirrors on the current Scale axis (`all` is treated as X) |
| Turn as a group | **`Alt` + `Shift` + wheel** — the selection swings about a shared pivot, snapped to the move grid so it lands back on-grid |
| Resize | **`Ctrl` + wheel** — steps by the Scale snap on the current scale axis |
| What gets edited | the ghost if one is being placed, otherwise **the selection**. `Del` is the exception and takes the hovered element first |
| Rotation axis | `R` cycles Y → X → Z. Y first: it is the only one a modular kit usually needs. |
| Scale axis | `F` cycles all → X → Y → Z |
| Build plane | numpad `+` / `-` (or main-row `+` / `-`) by the Move step; numpad `.` jumps it to the top of the hovered element |
| Select | **quick** left-click · `Ctrl`- or `Shift`-click adds to the selection · click empty space clears it · **drag from empty space to rubber-band**, or press **Rect select** to start the rectangle on top of a module. `Ctrl` or `Shift` while banding adds. To reach something behind a door portal, `Shift+H` the door — a ghosted element is click-through |
| Chunks | the **Chunk** button toggles isolation — pressed (orange), every chunk but the one in the dropdown is hidden · `+` adds a chunk · **Rename** renames the active one everywhere it is used · `Assign` moves the selection into the active one |
| Hide | `Shift+H` cycles the selection **50% → hidden → 50%** — half alpha (and click-through) to see past something, then gone · `H` returns everything to fully opaque · the **Ghost** slider sets how see-through that first state is. Undoable, but not saved — a reload starts with everything visible |
| Id | inspector `Id` row — read-only. The tool's handle for the element and its node name in `ship.glb` when no `Name` is set; doors, portals and behaviours all reference it, so it is not editable. In a multi-selection it names the element whose transform the fields below show |
| Name | inspector `Name` field — the element's **node** name in `ship.glb` (primitives are numbered off it), shared on purpose: elements with the same name share one behaviour entry. Shown in the corner overlay instead of the module id |
| Behaviour | inspector panel — attach library behaviours to the element's node name, and pick the `linked` nodes a liquefiable one melts with · **Edit behaviours…** opens the library (name + free-form JSON body) |
| Eyedropper | `Alt`-click a placed element to arm its module |
| Nudge | arrow keys move the selection on X/Z, `PageUp`/`PageDown` on Y — in whichever space `Y` has chosen |
| Steps | toolbar dropdowns — Move defaults to **1 m**, and **`Shift+V`** cycles it (`Ctrl+V` backwards). Move can be **off** (free positioning while dragging). Rot and Scale are keyboard *step sizes*, so instead of "off" they carry **`free`** — a fine step, `±0.5°` and `0.01`. Rot runs `-90°` to `90°`, the sign being which way `R` turns |
| Camera | `WASD` flies, `Space`/`C` rise and descend · **right-drag looks** · **right button + wheel sets the fly speed** · `Shift` for 2× · wheel dollies · `F` frames the selection. The left button never moves the camera |
| Lighting | **Env** slider — strength of the image-based lighting, which is where metals get nearly all their brightness · **Exposure** slider. Both are saved in the manifest and restored on Load · **Runtime light** drops the editor's own lights, leaving the HDRI the game actually uses |
| Walk | toolbar checkbox — walk at the player's eye height (1.8 m) instead of flying. `WASD` moves horizontally at the usual speed, the height follows whatever floor is underfoot, and `Space`/`C` are off |
| Undo | `Ctrl+Z` / `Ctrl+Shift+Z` (or `Ctrl+Y`) — whole-layout snapshots, capped by *memory* rather than a fixed count (1000 steps on this ship, fewer as it grows), so *anything* that pushes an entry is undoable: placing, deleting, dragging, turning, scaling, flipping, nudging, hiding, the Env and Exposure sliders, every inspector field and every behaviour edit |
| Edit | **`Ctrl+D` puts a copy of the current element — or of the whole selection — on the cursor** as a ghost, keeping every rotation and mirroring, and setting the drag axis back to `X/Z` so the copy arms where you can see it · **`Del`, or the middle mouse button, deletes the hovered element, or the selection if nothing is hovered** (deleting a hovered element leaves the rest of the selection intact) |
| Grid | `G` · **Big icons** doubles the palette width and tile size (on by default) · **Unlit** shows raw albedo with no lighting · **Exposure** slider — lower keeps pale panels off the tone-mapping shoulder, where their detail flattens out |
| Palette | hover a tile to spin the module through a full 360° turn |
| Save | `Ctrl+S` — also stores the camera position, so reloading puts you back where you were · **Load asks first if you have unsaved changes**, since it discards the whole scene in one click — and so does closing or reloading the tab |

**Every toolbar control names its shortcut in its tooltip**, or says outright
that it has none. The keys are the whole point of the tool — the combos are a
readout of modal state you are meant to drive from the keyboard, not reach for
with the mouse — so a control that does not mention its key is a key nobody
finds. The `Move` combo was the case that prompted it: the single most used
setting, and nothing on screen said it had a shortcut at all. A test walks
every button, select, slider and checkbox in the toolbar and fails if any lacks
a tooltip, or if a tooltip mentions neither a key nor "no shortcut" — so a new
control cannot ship undocumented.

**A button flashes orange when you press it.** `Save`, `Load` and `Export glb`
all do their work somewhere else — a file on disk, a line in the status bar — so
the button itself gave no sign it had been hit, and a press that missed looked
exactly like one that worked. One delegated listener on the document adds a
class for 260 ms, so a button added later is covered without anyone remembering
to, and no handler can forget. It runs on the **capture** phase, so a handler
that stops propagation, or throws, still gets its flash.

> **No transition on it**, and that is not an oversight. A fade *in* is exactly
> wrong for a flash: with one, the colour was still climbing out of the idle
> grey when the timer took the class off again, and the button never actually
> went orange — measured at 50 ms into a 180 ms fade, sitting at `rgb(98,68,54)`.
> A press reads as instant, so it has to be instant.
>
> Toggle buttons opt out. They already latch solid orange and *stay* there,
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
for what hiding is *for*: every click would still land on the wall you can now
see through. So `isPickable` goes off with the alpha, and the two states read as
a progression — *see it but reach past it*, then *gone*. The **Ghost** slider in
the toolbar sets how see-through that first state is.

**The selection is kept**, unlike the plain-hide behaviour this replaced. It has
to be: the cycle is driven by pressing the same key again, and since a ghosted
element cannot be clicked, the selection is the only handle left on it.

Hiding **is** undoable, and rides the same snapshot as everything else — not
because it is an edit to the ship, but because a restore begins with
`clearAll()`, which empties the veil. Leaving it out of the snapshot did not
make it immune to undo; it made it *half* undoable: every unrelated undo
revealed everything, and no redo could put it back. Symmetric is the only honest
option. It stays out of the manifest, so a reload starts with everything
visible, and veiled elements still export — the file on disk is the ship, not
the view of it.

**Hiding is the shifted one, not unhiding.** They are not symmetric: an
accidental hide on a large selection is expensive to notice and to undo, while
an accidental unhide costs a keystroke. The cheap direction gets the bare key.

Two details that would otherwise bite:

* **Half alpha cannot be per-instance the obvious way, and the shortcut is a
  trap.** Placements are hardware instances sharing one source mesh — and one
  *material* — per module. Babylon refuses per-instance `visibility` outright
  (*"Setting visibility on an instanced mesh has no effect"*), and the next idea,
  forcing that shared material to `ALPHABLEND` with a per-instance colour
  buffer, **breaks the depth buffer**: it moves every mesh drawn with that
  material into the transparent pass, which does not write depth. The kit shares
  materials across modules, so ghosting one wall stopped most of the ship
  occluding and geometry behind walls started showing through. Cloning is the
  only way to make one element translucent without changing what everything else
  is drawn with — so a ghosted element's meshes are swapped for clones carrying
  a cached translucent copy of the material (`ghostMaterialFor`, shared with the
  placement ghost, sharing geometry *and* textures so the cost is a draw call,
  not a 2–4 MB atlas).
* **The stand-ins are children of the element**, so they follow every drag,
  turn and scale for free. They must therefore be kept out of the export:
  `exportGlb` wraps its *whole* body in `withVeilSuspended()`, not just the
  write, because the allowlist and the `_primitiveN` renaming are both built
  from `getChildMeshes()` — a stand-in still in the tree at that moment would
  take a primitive index and be written into the file.
* **One function decides what is on screen.** Isolation and hiding both work by
  disabling nodes, so neither can own `setEnabled` alone — isolating a chunk
  would reveal everything you had hidden, and unhiding would reveal the chunks
  you had isolated away. `applyVisibility()` derives it from both, every time.

**The `Ghost` slider** sets how see-through the 50% state actually is (5–95%).
It repaints the cached veil materials live, so stand-ins already on screen
follow it. Like the exposure it lives in `localStorage`, not the manifest: it
says nothing about the ship and everything about how you like to look at it.

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
duplicate silently changed where *everything placed afterwards* would land.
**A multi-selection is carried too**: the ghost holds a
list of items, each with its own offset, turn and mirroring, so `Ctrl+D` on
twelve walls hands you twelve walls.

**It also sets the drag axis back to `X/Z` first**, and says so. In `Y` mode the
cursor drives the *build plane* rather than the ghost's own height, and it clears
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
> multi-selection goes through the *same* function, so it is told apart by
> `opts.copy` rather than by which key was pressed.

### Carrying versus dragging

There are two ways to move something, deliberately, and they differ in what
your hand is doing:

* **Drag** — press, move, release. The elements stay **solid** and follow
  directly. The button is held throughout.
* **Ctrl+D** — a copy of the current element comes up on the cursor. It keeps
  the source's height **without moving the build plane**: moving the plane was
  the old behaviour, and it meant a duplicate silently changed where everything
  placed afterwards would land.
* **Carry** (`M`) — the selection lifts onto the cursor as a **translucent**
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
is the *release*: every 3D tool drops on mouse-up, and a press-drag-release that
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
> built as *group rotation × item rotation* and a component-wise scale product,
> not read back out of the world matrix. A mirrored element has a negative
> determinant, and such a matrix has no unique rotation/scale split: measured,
> `Matrix.decompose()` moved a `[-1,1,1]` scale onto **Y** with a compensating
> turn. It looks identical on screen and is a different ship in the manifest.

**The bare wheel belongs to the camera.** It dollies, full stop — that is what a
wheel does in a 3D view, and every attempt to give it a *third* job fought that
expectation. The two edits it does carry sit on its modifiers: `Shift` + wheel
turns the current element and `Ctrl` + wheel resizes it. That is what freed the
letters to carry those actions' *settings* instead, so `R` and `F` read the same
way `V` always has. And **holding the right
button turns the wheel into a fly-speed control** — the right button already
means "I am driving the camera", so adjusting how fast reads naturally and
cannot collide with editing. Steps are multiplicative, so the control feels the
same at 3 m/s and 100 m/s, and the speed shows in the status line.

> Why not `Ctrl` + `WASD` for a slow mode? `Ctrl`+`W` **closes the browser tab**
> — it is reserved by Chrome and a page cannot cancel it — while `Ctrl`+`S` and
> `Ctrl`+`D` are already Save and Duplicate here. Only `Ctrl`+`A` was actually
> free.
>
> The same trap decided the axis-mode keys. `Ctrl`+`T` would have been the tidy
> pair for the move step, and it is **unusable**: `Ctrl`+`T`, `Ctrl`+`N`,
> `Ctrl`+`W` and their `Shift` variants are handled by the browser *before* the
> page sees the key, so `preventDefault()` has no effect.
>
> `Ctrl`+`R` is **not** on that list, despite what much of the internet says.
> Confirmed by hand in Edge: the editor claims it and the page does not reload.
> Worth knowing, because automated tests cannot answer this — Playwright injects
> keys through CDP, straight into the renderer, so a reserved shortcut looks
> claimed to a test and still fires for a real user. The only reliable check is
> a finger on the key.

A wheel gesture on the right button also has to mark that button as *used*, or
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

`V` says *which* axis a move runs on; **`Y`, or the combo beside it, says whose**
— the world's, or the element's own. A modular kit turns every second wall 90°,
so "slide it along its length" is world Z on one and world X on the next; in
local space it is `X only` on both.

**It governs turning as well.** `R` used to turn about a world axis whatever the
element was doing, so on a wall already yawed 90° "turn about X" tumbled it
about the *room* rather than about its own length. In local space it turns about
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

Whose element is always the one being *acted on*: the piece under the cursor for
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

| | |
|---|---|
| **Build plane** | the height new modules land on — `state.gridY` |
| **Current** | what the next key will act on: the ghost `◆`, or the selection `■` |

Everything else it used to carry — rotation axis, scale axis, fly speed, drag
axis, lighting mode — is already on screen in the **toolbar**, whose combos *are*
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
*is* world position, since placements have no parent in the editor — so a world
gizmo is the honest answer for moving, unless the `World`/`Local` combo says
otherwise. **Scaling is always local**, so on anything that has been turned
(most of a ship built from a modular kit) a world gizmo cannot tell you which
way `X` will grow.

**The flavour survives clicking another element.** The gizmo already followed a
single pick — having asked to see it, you almost never want it left behind on
the piece you have moved away from — but re-showing it took the *default*, so a
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
gizmo; the *other* key re-aims it in the other space rather than hiding it. It
also moves when you press `X` on a different element, and when you
**single-click** another one while it is up, since having asked to see the axes
you almost never want them left behind on the piece you moved away from.

**Nothing selected or hovered hides them.** Leaving the last element's gizmo up
would leave it hanging off something you are no longer working on, with no key
that clears it.

**Showing them sets the space you work in.** `X` switches `World`/`Local` to
`World` and `Shift+X` to `Local`, because asking to see an axis is nearly always
asking to work along it — you press `Shift+X` to find which way the element's own
X grows *because* the next thing you do is slide it that way. `Y` still overrides
it afterwards, so the coupling costs a keypress in the rare case and saves one in
the common case.

> Only on the way *up*. Hiding a gizmo says nothing about which space you want,
> and a toggle that quietly changed the drag axis on the way out would be a
> genuinely surprising way to lose a placement.

**The armed ghost counts as an element.** You set a module's rotation and
mirroring *before* dropping it, which is exactly when the axes are worth seeing
— and it was the one case `X` did not cover. The ghost lives in `interact.js`,
which imports `editor.js`, so it registers its node through `hooks` rather than
being imported back and closing a cycle; the gizmo then follows it around the
cursor for free, and goes when the ghost is cancelled.

It carries all three modal axis settings at once, so "what will the next key do
to this element" is one glance rather than three readouts:

| on the gizmo | means | set by |
|---|---|---|
| bright arrow, dim = locked | the axes a drag moves along | `V` |
| curved arrow encircling it | the axis a turn goes about | `Shift+R` |
| cube on the tip | the axis a scale acts on (all three for `all`) | `F` |
| the chip at the origin | the move step, in metres, or `free` | `Shift+V` |
| the chip inside the curved arrow | the turn angle, in degrees, signed | `Ctrl+R` |
| a chip on each lit cube | the scale step | `Ctrl+F` |

Each modal setting has exactly one marker, and each marker means exactly one
thing — `V` never touches the ring, `Shift+R` never touches the brightness. The
curved arrow sweeps three quarters of a turn rather than closing into a full
ring, so it reads as a direction of travel and not as a collar.

**The turn angle is a magnitude.** It was briefly *signed* — `-90°` through
`-5°` sat alongside the positives — because a key only ever turned one way, so
turning back meant four presses of 90 or switching the axis and reasoning about
which sign that gave. The wheel carries the direction now: one way turns, the
other turns back. So the sign went back out of the list, and the arrow the gizmo
draws lost the mirroring it had grown to keep up with it.

**`free` is a fine step, not no step.** Both lists carry one — `0.5°` on `Rot`,
`0.01` on `Scale` — for dialling in a value with the wheel. It is deliberately
*not* zero: `Move` can be switched off because a drag is a continuous gesture
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
scale chips hang off the arrow *tips*, which swing out of view at close range,
and `#viewport` does not clip: one was caught sitting on a palette tile.

Putting each *value* on the marker that governs it is the point: the gizmo then
answers **how far**, not just **which way**. All three settings decide what the
next keystroke does, and all three were otherwise only legible in a toolbar combo
at the far edge of the screen.

The scale step gets **one chip per lit cube** rather than a single shared one:
with the axis set to `all` the three cubes are the statement that all three axes
grow, and a value on only one of them would read as "just this one".

Three implementation notes worth keeping:

* **The curved arrow's node sits at the arc's centre**, with the geometry built
  around the origin — not at the arm's root with the arc pushed out along `Z`.
  That is what makes `getAbsolutePosition()` the ring's own position, which the
  angle chip needs. Built the other way round, the chip landed on the gizmo
  origin and sat on top of the move-step chip (measured: 3 px apart, versus
  100 px now).
* The gizmo's **position** is re-read from the element every frame rather than
  parented to it. Parenting would inherit the element's scale — and a 3× element
  must not get 3× arrows — while re-deriving it each frame also covers drags,
  undo, the ghost following the cursor, and deletion with no event plumbing: if
  the element goes, the gizmo goes.
* Every part is marked **`alwaysSelectAsActiveMesh`**, and — the point —
  **nothing sets `doNotSyncBoundingInfo`**. That flag was on every part as a
  micro-optimisation, and it is exactly the one that stops a bounding box
  following its mesh's world matrix. Since the gizmo moves by being
  *re-positioned* rather than re-parented, the boxes stayed wherever it was
  built: select an element 50 m out and the boxes trail 50 m behind the arrows
  (measured — the drift was exactly 50), and the frustum test then culls arms,
  arrowheads and turn arcs on their old position. It reads as the arrows being
  **clipped**, worse the closer you fly, and clicking away and back cures it
  because re-showing rebuilds the meshes. Fifteen tiny meshes are not worth
  culling at all, so they are never culled and their bounds are left honest.
* Its materials are `StandardMaterial`, not PBR, precisely because
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

> **Why `V` and not `Q`.** Movement is matched on `e.code` (the *physical* key)
> so `WASD` stays under the same fingers on any layout — on AZERTY that is
> `ZQSD`, which is exactly what a French keyboard expects. The letter shortcuts
> match on `e.key` (the *label*). Those two collide on precisely one key: the
> AZERTY key labelled `Q` sits where QWERTY has `A`, so it arrives as
> `code: "KeyA"`, is claimed as strafe-left, and never reaches the shortcut
> switch. `Q` is therefore unusable as a shortcut for AZERTY users. `V` occupies
> the same position and carries the same label on both layouts — and reads as
> "vertical". The test suite pins this down by firing both spellings.

**The drag anchors on the point you clicked, not the element's origin.** Kit
origins sit at the *base*, so a column or door frame grabbed near the top, from
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
up at a column, higher on screen is *nearer*, so pushing the mouse forward pulls
the piece towards you. It also runs to infinity as the ray approaches parallel.
Neither is fixable by picking a different height — at eye level the horizontal
plane is simply edge-on to the view. So `anchorDrag()` chooses: plane when the
grabbed point is below the camera *and* the ray is more than ~15° off the plane,
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
is that a *selection change* has to re-evaluate the hover in **both**
directions: selecting the element under the cursor drops the hover, and
deselecting has to bring it back straight away. Only clearing it leaves `Esc`
followed by a keypress doing nothing until you jiggle the mouse.

**Rectangle select** is a rubber band drawn with the left button. A drag that
starts on **empty space** is always a rectangle — there is nothing else it could
mean, since the left button no longer moves the camera. Starting *on a module*
is the ambiguous case, and that is all the **Rect select** toggle is for:
with it on, a drag bands instead of picking the module up.

Hit-testing projects the corners of each element's **oriented** box — each mesh's
own box, transformed — and takes the convex hull of those points. Anything whose
**outline** meets the rectangle is selected, occluded or not, because "what is
inside this rectangle" is a screen question, not a visibility one. Corners behind
the camera are dropped: they project to a mirrored point that would otherwise
stretch the hull across the whole viewport.

> **Neither approximation could stay.** It used to take the screen-space
> *extent* of the corners — an axis-aligned rectangle — and test that. For a
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
> having to *enclose* a long wall to catch it is the worse trade.

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
— and then `contextmenu` aimed at whatever is *actually* under the cursor, which
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

Crucially it is *not* a modifier. Turning (right-drag) and moving (`WASD`) are
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

An edit applies to *every* selected element, each turning about its own
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
the *left* to act in world space — the other order gives a local-axis turn.
Euler is now produced only at the boundary, when a placement is written to the
manifest.

**`Rot` and `Scale` have no "off".** They are keyboard *step sizes*, and a step
of nothing is meaningless — the old `off` option silently fell back to a hidden
default (15° / 0.05) rather than doing anything. Only `Move` has a real "off",
which means free positioning while dragging.

**A restore is not itself an edit.** `deserialize()` sets a `restoring` flag and
`pushUndo()` no-ops while it is set. Without that, a single stray `pushUndo()`
anywhere in the restore path does two invisible kinds of damage: it **clears the
redo stack**, and it pushes a *half-restored* snapshot onto the undo stack.

**The lighting is on the stack; the camera is not.** Both were off it at first,
on the same reasoning — "not an edit to the ship". Only half of that held up.
`Env` and `Exposure` are *authored* values: the manifest carries them and the
runtime reads them, so a lighting change you cannot take back is a real edit
lost. Where the camera happens to be standing is genuinely not.

Both light sets travel together, not just the one on screen. The **Runtime
light** toggle only decides which pair the sliders edit, so restoring the
visible pair alone would leave the hidden one behind, to surface later as a
value nothing ever put back.

**One entry per slider gesture.** A range fires `input` continuously while it is
dragged, so one sweep of `Env` would otherwise bury the stack in near-identical
snapshots. The push is armed on `pointerdown` (and on the first key or wheel)
and spent on the first change — the same shape as the inspector fields, but
without a focus event to hang it on, because a range keeps focus between drags.

**Load asks before discarding unsaved work.** It throws away the whole scene and
sits one button away from Save; nothing else in the tool destroys that much in a
single click. **Closing or reloading the tab gets the same guard**, via
`beforeunload` — the browser owns the wording there (custom text has been
ignored since 2016), so all the tool chooses is *whether* to ask. Chrome also
requires the page to have been interacted with first, which is the behaviour we
want anyway: a tab you only looked at closes silently.

"Unsaved" is decided by comparing against a snapshot taken at the last save,
load or boot — not by a dirty *flag*, so undoing back to the saved state
correctly counts as clean again. `hidden` is stripped from that comparison: it
never reaches the manifest, so it can never be saved, and leaving it in would
make hiding one wall enough to prompt for the rest of the session.

That is not hypothetical — restoring a spawn marker did exactly this, because
`deserializeMarkers()` passed `silent: true` for doors but `setSpawn()` had no
such option and always pushed. Since `deserialize()` runs on every undo *and*
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
wheel then resizes the *magnitude* and leaves the sign alone. A mirrored node
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

## Chunks, doors and portals

Chunks are what the portal renderer streams and culls. Pick the active chunk in
the toolbar; new placements join it. **Assign** moves the current selection to
it, **`+`** adds one and **Rename** renames it, and the **Chunk** button itself
toggles isolation — pressed, everything outside the active chunk is hidden, and
it turns orange so the view being partial is never a mystery. Switching the
dropdown while isolated follows the new chunk.

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

* **Door** arms a marker you drop on the grid.
* **Door from sel** is the usual route — select the door geometry you already
  placed and it creates a marker centred on it, sized to its opening, with those
  placements registered as the animated leaves.
* Chunk A/B default to `(auto)`, which resolves to the two nearest chunk
  volumes at save time. Set them explicitly when that guess is wrong.
* **Sealed** marks a portal you can see through but not walk through — a window
  onto space rather than a doorway. The renderer still draws the far chunk;
  collision generation keeps the opening solid. It is written on the door
  record only (`doors[].sealed`), not on the portal, and defaults to `false` so
  every manifest written before it reads back as an ordinary doorway.
* **Doors resize two ways.** The inspector's `W`/`H` fields set the authored
  opening; `Ctrl+wheel` and the inspector's `Scale` fields scale the node like
  any other element. Both were once blocked for markers — `scaleCurrent()`
  filtered them out and `applyInspector()` guarded the write — which just made
  doors feel broken. The exported `width`/`height` fold the node scale in
  (`width × |scale.x|`), because `portalOf()` reads the **world matrix** and so
  the portal grows with the node regardless; leaving the raw width in the
  manifest would have made a door disagree with its own portal.

Portals and the adjacency graph are derived from doors, so there is nothing
extra to keep in sync.

**Isolation shows a door in both the rooms it joins.** A door is not *in* a
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

| shape | what the manifest carries | why |
| --- | --- | --- |
| box | `centre`, `rotation` (quaternion), `halfExtents` | `PhysicsShapeBox` takes an explicit turn |
| sphere | `centre`, `radius` | no orientation exists |
| capsule | `pointA`, `pointB`, `radius` | `PhysicsShapeCapsule`'s turn is implicit in the segment; the segment is a diameter shorter than the height |
| cylinder | `pointA`, `pointB`, `radius` | likewise, but the segment *is* the height |

**The scale is constrained per kind, on every write path.** Havok's sphere is a
single radius and its capsule a radius plus two endpoints, so an ellipsoid has
no representation at all: the runtime would have to silently resize it, and the
ship you built would not be the ship you play. `constrainScale()` therefore runs
on the inspector, the wheel, a carried duplicate and a loaded manifest alike —
a sphere is forced round, a capsule and cylinder locked to one radius in X/Z,
and only the box takes an arbitrary scale. A capsule additionally cannot be
shorter than it is wide: at `h == d` its two caps meet and it *is* a sphere, and
Havok agrees — its capsule is a segment plus a radius, and there is no segment
of negative length.

### A capsule is drawn as a capsule

A box, a sphere and a cylinder are each a single unit mesh under a scale. A
capsule is not, and this is the one place the "unit shape sized by scaling" rule
does not hold: a capsule's caps are hemispheres of the *tube's* radius, so
stretching one unit mesh in Y stretches the caps with it into an ellipsoid.

It was worse than that. The unit mesh was `CreateCapsule({height: 1, radius:
0.5})`, and Babylon's capsule `height` **includes** the caps —
`heightMinusCaps = height − radiusTop − radiusBottom` — so at radius 0.5 there
was no tube left at all. Every capsule in the editor was a sphere pulled into a
lozenge, 141 mm off a true capsule's surface at 1 × 3 m, while Havok collided
with a proper pill.

The trick that keeps it to a single mesh: build the geometry at the right
**ratio** — radius 0.5, total height `h/d` — and counter-scale the mesh's own Y
by `d/h`. The two cancel, so the mesh ends up wearing a *uniform* world scale of
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
says. Havok's capsule is a segment *grown by the radius in every direction*, so
`shapeRecord()` writes a segment one diameter shorter than the height. A
cylinder has flat ends and keeps its full height. Getting that wrong makes every
capsule a diameter taller in play than it looks in the editor — invisible in the
tool and baffling in the game.

Because a capsule at scale 1 is one metre wide and one metre tall — which is a
sphere — the Collision pane arms it at 1 × 2 m, so the brush looks like the
thing it places.

### Primitives land corner first

A kit module is modelled *from* its origin: a wall's geometry starts at the
origin and runs 4 m to the side, so dropping it with the origin on the build
plane leaves it resting on the floor and filling whole grid cells. A collision
primitive is a **unit shape centred on its origin**, because Havok wants a
centre and not a corner — so the identical drop buried half of it under the
plane and put its faces through the middle of a cell.

The ghost therefore shifts a lone primitive by its own half size, putting the
*corner* where the origin was snapped to. The lift is derived from the live
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
> *surface* is put flush with the visible surface the player meets, and the body
> extends **away** from the play space. Floors show it most plainly: every
> `Platforms/*` hull is `position.y = -0.25`, `scale.y = 0.5` — a half-metre
> slab hanging entirely *below* the walking surface, so you stand exactly on the
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

**On the ship** they create *room* colliders: world-space one-offs belonging to
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

**Edit collision** opens a staging area. It is a *mode*, not a property of the
selection: it starts empty, and you stage whatever modules you want to work on.
Clicking a palette tile arms a ghost you place yourself; clicking one already
there focuses it rather than adding a second, because a second instance would
give the association rule below two equally good answers.

Staged elements are real placements carrying `stage: true`. That is what makes
every existing tool work on them unchanged — selection, `X`, `H`, `Del`,
`Ctrl+D`, dragging, the marquee, the inspector, undo. They are filtered out at
the two places that walk *every* placement (the manifest's instance list and the
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
*relative to the element*, so it does not matter where on the area it lands next
time.

**A module dropped on the bench is a one-shot.** On the ship a palette tile
stays armed, so a row of panels is just repeated clicks. Here it is the
opposite: a module comes to the bench once, to have a hull fitted to it, and
staging the same one twice is refused anyway — so staying armed only ever left
a second stand-in on the cursor to be dismissed. The tile goes out and the hint
clears with it, which is the visible difference between the two modes.

> **Collision primitives still repeat.** A hull genuinely is a run of boxes
> along a wall, so the box, sphere, capsule and cylinder ghosts stay armed. The
> rule is about *modules*, not about the bench.

**`Ctrl+D` there copies shapes, never the module** — for the same reason: a
module may only be on the bench once. The rule lives in `grabSelection()` rather
than only at the key, which used to filter a list it then did not pass on, so a
module *selected alongside a shape* was copied anyway. `M` still picks a staged
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

| candidate | what it is | what it suits |
| --- | --- | --- |
| `box` | the whole module in one box | a crate, a floor plate, a door |
| `slabs` | a slab per surface patch, laid on it and extending backwards | a wall with a lip |
| `split` | cut in half and recurse until each piece is worth boxing | a corner, a door frame |

The score is **coverage × solidity, less a small penalty per box**, where
solidity is the share of the hull's volume that sits near real surface.
Coverage alone is a bad judge: one enormous box covers every vertex and fills
the room the player is meant to walk in. Letting the score choose means no
single heuristic has to be right about which kind of module it is looking at.

Three settings on the Settings pane govern it.

**Hull tolerance** is how far the hull may stray from the art before that
counts as wrong, in metres, and it is deliberately the *only* dial for how
finely a shape is approximated. It is the resolution the hull is judged at, so
tightening it fails the coarse candidates and a finer one has to take over — a
rounded platform is worth one box at 25 cm and eight at 4 cm, while a crate
stays a single box whatever you set. Fine values cost noticeably more time
(tens of milliseconds at 25 cm, a second or so at 2.5 cm on a big module).

A separate "how many boxes" setting would have to be kept in step with it, and
the two would disagree. The choice is made in two questions, in order: is the
hull good enough — does it contain the art, and is it within the tolerance of
it nearly everywhere — and of the ones that are, which is the smallest? That
ordering is what stops a tighter tolerance ever handing back a *bulkier* hull
than the setting before it.

> **Volume is the honest measure of a collision hull.** Every cubic metre of
> it that is not art is somewhere the player cannot stand. Ranking on how
> snugly a hull fits its art instead looks reasonable and is not: it happily
> picks twenty overlapping boxes over three good ones.

**Hull thickness** is the depth a hull is given along its thinnest axis. The
kit's walls are millimetres thick and its floors are single planes with no
depth at all; a collider that thin is something a fast-moving body goes
straight through. It is a *minimum* — a crate is already thicker and is left
alone.

**Hull offset** decides which side of the art that depth goes.

| | what it does |
| --- | --- |
| `centered` | splits the depth either side of the art |
| `negative` | tucks the hull behind the visible surface — the convention this ship uses |
| `positive` | stands it in front |

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
normals point *out* of the solid, so laying a slab on a patch and extending it
backwards puts the body away from the play space for free — without needing to
know which side the room is on. Its depth is **capped**; using the patch's own
extent instead let a curved patch, whose points wrap right round an arc, grow a
slab that swallowed everything the arc enclosed (`WallAstra_Corner_Round_Outer`
came out at 87 m³ against a hand-drawn 12). Its bin angle is *derived from the
tolerance* rather than fixed: leaving it constant was what stopped a tighter
tolerance from ever improving a rounded corner, because the slab pass kept
winning with the same coarse three.

The `split` pass cuts **axis-aligned** and offers *every depth* as a candidate.
Both of those are deliberate. Cutting square to the world means the pieces nest
inside their parent, so an extra level can only shrink the hull; and stopping on
a threshold meant the threshold was sometimes met one cut too early, so handing
the whole ladder to the score lets it pick.

The *box drawn round* a piece may still be turned, though, and each level takes
whichever of the two is smaller **by total volume**. Turned boxes are far
tighter on a curve — the whole reason a rounded corner wants a hull that follows
it — but neighbouring pieces then overlap, and summed volume counts an overlap
twice, so a turned set is charged for exactly what it wastes and only wins when
it really is the smaller hull.

> Choosing per *level* rather than per piece is what makes that safe. Choosing
> per piece minimises each piece and lets the overlap between them run free,
> which is how an earlier version made a hull *bulkier* the more finely it was
> cut.

This is what fixed the rounded corners with a lot of surface detail.
`TopCables_Corner_Round_Outer` carries 3804 triangles of cable, which drowns the
slab pass in tiny normal bins, so the split was the only candidate left — and
while it was square-only that meant an axis-aligned box round an arc:

| module | before | after |
| --- | --- | --- |
| `TopAstra_Corner_Round_Inner` | 16 square, 13.1 m³, 58% solid | **4 turned, 4.8 m³, 100%** |
| `TopCables_Corner_Round_Outer` | 16 square, 10.4 m³, 63% solid | **4 turned, 7.3 m³, 100%** |
| `TopCables_Corner_Round_Inner` | 16 square, 10.6 m³, 63% solid | **4 turned, 7.6 m³, 100%** |
| `ShortWall_WhitePlate2_Corner_Inner` | 24 square, 10.8 m³ | **2 turned, 4.3 m³** |

> **Solidity judges the shape, volume judges the price.** Solidity is measured
> *before* the thickness is added. Measuring after it has a thin floor plate,
> padded to a walkable depth, fail its own test — most of that hull is
> deliberately not near the art — and the fitter chops the plate up trying to
> fix it.

> **A hollow shell is solid inside.** Whether a sample is in the art is decided
> by ray parity, not by proximity to a triangle: a crate is a hollow mesh, so
> every point in the middle of it is far from any surface, and judging on
> proximity alone marks a hull that fills the crate as mostly empty air. The
> crossings along each ray are *paired*, and an odd one left over is dropped —
> the kit is full of open shapes, and plain parity would mark everything beyond
> a single-plane floor as solid.

> **It declines rather than guess.** A hull that covers ≥ 95%, sits ≥ 85% on
> surface at the tolerance and needed ≤ 4 boxes is reported plainly; anything
> else is fitted but flagged *"worth checking by eye"*. In practice that is the
> curved corners, which are genuinely better drawn by hand. Saying so is more
> use than a hull that looks plausible and leaks.

Measured against the 53 hand-authored hulls (`fit-boxes.mjs`, geometry dumped by
`fit-dump.mjs`, sweep by `fit-sweep.mjs`) at the default 10 cm tolerance and
35 cm thickness:

| | coverage | boxes | volume | on surface |
| --- | --- | --- | --- | --- |
| fitted | **99.7%** | 3.8 | 7.5 m³ | **89.8%** |
| hand-authored | 76.8% | 1.5 | 6.4 m³ | 68.3% |

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

> The fitter lives in `public/js/hullfit.js`, imported by *both* the editor and
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
> last point something *explicitly* aimed it at, and free look never updates
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
screen — so saving without closing the bench first wrote the *bench's*
viewpoint into the ship's `view`, moving where the ship reopens to wherever the
bench happened to be. Two records had the mirror-image fault at the same time:
`stageView` and `stageLayout` are only written when the bench *closes*, so the
same save wrote last session's bench viewpoint and last session's roster —
quietly losing everything staged since.

> Three records, one mistake: **reading the live thing when the live thing is
> the other side's.** `shipViewpoint()`, `stageViewpoint()` and
> `stageLayoutNow()` each ask which side the live camera and scene currently
> *are*, and hand back the stash for the other. The stash cannot be stale:
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
`serialize()` - they must never reach the ship - so a *ship* snapshot restores
as "no bench at all", which is precisely how `Ctrl+Z` used to wipe it. A
separate stack also means undoing one box does not rebuild a hundred placements,
and the ship's own history is left untouched while you work.

**Moving or turning a stand-in carries its shapes with it**, and changes nothing
about the hull: the hull is authored in the module's own frame, so shifting the
stand-in is a *view* operation. Left to itself the element slid out from under
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
point of it living apart: the collision is a property of the *kit*, not of any
one ship, so once these hulls are fitted the file can be shipped and the next
ship built from the same kit starts fully fitted.

**A failure writing it does not fail the save.** The ship is already on disk by
then, and throwing would leave the editor believing it had unsaved work — which
is exactly what happened against a server too old to know the route: every save
appeared to fail, and every reload warned about losing changes that were in fact
safely written. The status line names the problem instead.

The manifest carries three collision blocks, and they are **not** three copies
of the same thing:

* `collision[chunk]` — a room's **own** one-off shapes, world space, in Havok's
  parameters. Grouped by chunk because collision is streamed per room.
* `moduleCollision[moduleId]` — what a kit module carries, in the module's own
  local space, in Havok's parameters. **Written once per module**, for the
  runtime to instance onto every placement of it and to share one Havok shape
  between them.
* `moduleShapes[moduleId]` — the same hulls in the **authoring** form: editor
  coordinates, position/rotation/scale. The source the tool reloads from, and
  what `moduleCollision` is derived from — exactly the way `colliders` relates
  to `collision`.
* `colliders` — the editor's record of the *room's* shapes.

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
things *off* screen, so chunk isolation and the `Shift+H` veil keep the last
word. It runs through `applyVisibility()`, the one place that decides what is
enabled, for exactly that reason. Hiding a layer drops any selection it hides,
so the gizmo and the inspector never act on something nobody can see.

### The shell thickness

The kit models its floors and ceilings as **single planes with no depth at
all**: `Platforms/Platform_3Plates` measures `[4, 0, 4]`. A zero extent has no
shape, and the guard against it used to read `Math.abs(v) || 1` — and `0 || 1`
is **one metre**. A two-plate room therefore came out with metre-thick slabs top
and bottom while its walls were 7.5 mm.

A module with no depth on some axis is now given at least the **collision shell
thickness**, and so is one that is merely *thinner* than it. The shell is a
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
and a *turned* capsule as something with no relation to the radius it is
actually built from — and the whole point of constraining the scale is that
those numbers are the truth.

## Settings

Ship-wide constants live in `state.config`, are edited in the **Settings** pane,
and go in the layout. They are authoring decisions, not preferences: a ship
fitted with an 8 mm shell and reloaded on another machine has to come back with
the same shell, or its collision silently changes. So they are **saved with the
layout**, go through the **undo stack** like any other edit, and are read back
through `{ ...CONFIG_DEFAULTS, ...saved }` so a layout written before a setting
existed returns that setting's default rather than `undefined`.

| setting | default | what it does |
| --- | --- | --- |
| Collision shell | `0.008` m | the *minimum* thickness any collision box is given on any axis |
| Auto-save every | `2` min | how often a recovery copy is written; `0` turns it off |
| Hull tolerance | `0.1` m | how far a fitted hull may stray from the art — the fidelity dial for **Fit a hull** |
| Hull thickness | `0.35` m | the depth a fitted hull gets along its thinnest axis |
| Hull offset | `centered` | which side of the art that depth goes: `centered`, `negative`, `positive` |

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

### Folding the panes

**Collision** and **Settings** are `<details>`, so their headers fold them. A
`<details>` rather than a hand-rolled toggle: the open state is then a real
attribute the browser keeps, keyboard and screen readers get it for free, and
Ctrl+F still reaches a folded pane's contents. Which panes you keep rolled up
is in `localStorage`, not the layout — whether *you* fold Settings says nothing
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

## Live checks

The inspector re-runs these on every change (they replace `build_ship.py`'s
batch validators):

* overlapping chunk volumes — ambiguous portal membership
* objects below `y = 0`, or off the current move-snap grid
* doors whose two sides resolve to the same chunk, or that have no leaves
* chunks no door reaches

---

## Implementation notes

**Shared materials are not optional.** Every module .gltf in the kit references
the same ~20 root-level textures and names its materials identically
(`MI_Trim_01`, `MI_Trim_02`, `M_Light`…). Loading them naively would give you
one copy of a 2–4 MB atlas per module. `kit.js` keeps the first material seen
under each key and disposes the duplicates *with their textures*; `thumbs.js`
does the same on its own engine, using the same key function. Placing 300
modules costs 8 materials and 9 textures.

> **The key is the name *and* its transparency**, not the name alone. The kit
> authors some of those shared names two different ways: `M_Glass` is `BLEND`
> with an alpha of `0` in two files and `OPAQUE` in twelve, and `M_Decal_White`
> is `MASK` in thirty-one and `OPAQUE` in twenty-six. Keyed by name alone,
> whichever module happened to load first decided how glass looked *everywhere*
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
three-quarters whatever is behind it, and the tint led red by ten values out of
255. These are the values judged in the Sandbox against the real ship instead —
a deep green `(0, 109, 11)` at half alpha, which is a window you can see is
glass.

`ior` is why it reads as a tint rather than a shine. A dielectric's reflectance
is `((n−1)/(n+1))²`, so an index of refraction of 1 makes it zero: the pane
stops catching a white highlight and the colour is left to be read on its own.

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

**A cached thumbnail is versioned.** The stills and turntables live on the
server's disk and outlive any reload, so a change to how a thumbnail is
*rendered* would otherwise be invisible until someone deleted the folder by
hand — which, for a bug whose whole symptom was a tile disagreeing with the
ship, is exactly the wrong failure mode. `THUMB_VERSION` is stamped into every
cache key, and a `PUT` sweeps the superseded copies of that module, unversioned
ones included: bumping it costs one re-render each, not an orphaned folder.

**Geometry is instanced.** Each module is loaded once as a disabled prototype;
placements are `createInstance()` children of a `TransformNode` named after the
placement id. Picking walks `mesh.metadata.placementRoot` back to that node.

**Ghosts are textured clones.** Instances cannot carry their own material, so
the ghost clones the prototype's meshes (`Mesh.clone()` shares geometry — zero
new geometries) and gives each one a cached translucent copy of its *real*
material, so you can tell which face of a module you are looking at.
`PBRMaterial.clone()` re-creates every texture, which would duplicate the kit's
2–4 MB atlases per material, so the copies are disposed and the slots
re-pointed at the originals — the scene texture count does not move.

**Outlines use per-instance edges rendering.** This is the only one of the three
options that follows a single instance: `HighlightLayer.addMesh()` throws on an
`InstancedMesh`, and `renderOutline` is read off the *source* mesh, so it would
outline every copy of that module at once (both verified). Edges rendering also
reads the instance world matrix every frame, so nothing has to be re-synced
after a rotate or scale. Hover is orange, selection is blue, and the two can
never land on the same element.

> **`edgesWidth` is not a pixel width.** The line shader offsets the vertex in
> *clip* space, before the perspective divide, and the renderer hands it
> `edgesWidth / 50` — so what you see is
> `edgesWidth * renderHeight / (100 * viewDepth)`, which doubles every time you
> halve your distance. A fixed 5 read as a fine line across a room and as a
> 16 px slab of colour with the camera against a crate, hiding the very thing
> it was drawn to point at.
>
> Turning that around gives the width to ask for, so it is set from the view
> depth of each outlined mesh on every frame — the camera moves without
> emitting anything, so this rides the render loop. It is the *view* depth, not
> the distance: that is what the shader divides by, and the plain distance
> would fatten the outline on anything off to the side of the screen. Measured:
> 4 px at 12 m and 3 px at 2.5 m, against 5 → 16 px before.

**The ghost compensates for the module's origin.** Kit modules are not modelled
around their own origin — `ShortWall_Band2_Straight`'s body sits 2 m away from
it — so snapping the origin straight to the grid drops the piece a whole cell
away from the pointer. The ghost offsets by the module's local body centre
before snapping, which bounds the error at half a cell (measured: 0.40 m
compensated vs 3.60 m without).

**The ghost tracks the cursor on a plane through the module's *body*.** Kit
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

**Alt is claimed from the browser.** On Windows, Chromium hands `Alt` to the
menu bar — which then swallows the following keystrokes, so WASD silently stops
working after any `Alt` press. Both the keydown *and* the keyup are
`preventDefault()`ed (menu activation happens on release). Worth knowing when
debugging: this does **not** reproduce under Playwright, whose injected key
events bypass the browser's own accelerator handling entirely.

**Dragging moves the real elements, not a ghost.** A ghost is per-module and
single; a drag has to move a whole selection. The delta is measured on a
horizontal plane at the *selection's own height* rather than the build plane, so
the motion tracks the cursor instead of being skewed by perspective — and the
**delta** is snapped rather than each element, which keeps a group's internal
spacing exactly as authored even when it was placed off-grid.

**A drag has to out-rank the camera.** Left-drag normally looks around, so
`setupPointer` registers its observer *before* `camera.attachControl` and sets
`eventState.skipNextObservers` on a pointer-down that starts a drag. The camera
input then never records a start position, so the whole gesture is swallowed —
no need to detach and re-attach anything mid-drag.

**Double-click is detected by hand, not via `POINTERDOUBLETAP`.** Babylon's
double-tap requires the single tap to wait out the timeout, and a placement tool
cannot afford 300 ms of latency on every click. Instead the first click always
acts immediately and a second one within 320 ms and 6 px *also* emits
`dblclick`, which frames the element. While a module is armed the double-click
handler bows out, which keeps rapid clicking in the same spot placing two
modules.

**There is no pick-plane mesh.** The build plane is solved analytically by
ray/plane intersection (`cursorOnGrid()`). An actual mesh would sit in front of
the geometry the moment the plane is raised above it, making every element
unselectable — and at `y = 0` it tied with floor tiles resting exactly on it.
Consequence worth knowing: if the plane is above the camera *and* you are
looking down, there is legitimately no intersection and the ghost stops moving.

**Everything is double-sided in the editor.** Kit modules are single-sided in
places, and a wall turned away from the camera simply vanishes — correct in
game, useless while building. `scene.onNewMaterialAddedObservable` forces
`backFaceCulling = false` on every material, backed by a re-sweep whenever the
material count changes: the observable can fire from the base `Material`
constructor, *before* a subclass has applied its own culling default. The
authored value is remembered in a `WeakMap` and `withAuthoredCulling()` puts it
back for the duration of the glTF export, so the runtime gets exactly what the
kit shipped rather than the editor's convenience setting.

**One free-flying `UniversalCamera`, no arc-rotate.** An arc-rotate camera is
always tethered to a pivot: its speed and reach are tied to the orbit radius, so
it slows to a crawl as you close in and cannot simply walk down a corridor.
Movement is fed through `cameraDirection` / `cameraRotation` rather than by
writing `position` directly, so camera inertia still smooths it — with a
`(1 - inertia)` factor on each impulse so the steady-state speed is the one
asked for regardless of the damping. The camera therefore *coasts* to a stop
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

The framing is re-applied *after* `setTarget()`. `ArcRotateCamera.setTarget()`
re-derives alpha and beta from wherever the camera currently is, so a module
modelled high above its origin (`TopCables_Corner_*` sits at y = 4) pushed beta
past 90° and got rendered from underneath, while alpha drifted per module and no
two tiles matched. Note that *negating* beta does not fix this — `cos` is even,
so the camera keeps its height and only mirrors in azimuth. Delete
`cache/thumbs/` (and `cache/turntable/`) to force a re-render after changing the
framing or the lighting.

**`grid-auto-rows: max-content`** on the palette is load-bearing. Without it the
implicit grid rows collapse to zero height and every thumbnail is invisible even
though the images loaded fine.

**Selection needs a *quick* click.** A left press only selects if it both stayed
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
`HemisphericLight.direction` points at its *sky*, so a single one pointing up
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
about **240/255**: deep in the KHR PBR Neutral *highlight shoulder*, where the
tone curve is almost flat. Everything the artist painted — dirt mottling, panel
seams, the bolt strips — gets compressed into the top ~6% of the output range
and simply disappears. Measured contrast over such a panel:

| exposure | mean | std-dev |
|---|---|---|
| 1.0 | 240 | 6.5 |
| 0.7 | 234 | 10.7 |
| 0.55 | 229 | 12.2 |
| 0.45 | 208 | 13.1 |

The giveaway was that the **same module looked sharper while being dragged**
than once it was placed. There is no material difference at all: forcing the
ghost's material opaque makes it render pixel-identical to a placed one
(mean 240.8, sd 0.33 for both). The ghost's `GHOST_ALPHA = 0.7` was quietly
acting as an exposure cut, blending the surface down the curve and out of the
shoulder.

So the default is `EXPOSURE_DEFAULT = 0.55`, with a toolbar slider because the
trade is real: lower exposure buys detail on pale panels and costs brightness on
genuinely dark props. The setting is remembered in `localStorage`.

The thumbnail scene had exactly the same problem, worse — it ran at exposure 1.4
with `environmentIntensity` 2.4, which is why `Platform_Simple_*` and
`Door_Simple` tiles looked like flat blocks of colour. It now runs at 0.5 / 1.6.

**Beware `Number(v) || DEFAULT` when clamping.** A literal `0` is falsy, so that
idiom silently jumps to the default instead of clamping to the floor;
`setExposure` uses `Number.isFinite`.

### What is *not* wrong

**No ORM colour-space fix is needed**, and the loader is not mis-tagging
anything. It is easy to talk yourself into the opposite, because the flags read
backwards:

| texture | `gammaSpace` | `useSRGBBuffer` |
|---|---|---|
| Base Color | `false` | **`true`** |
| ORM | `true` | `false` |
| Normal | `true` | `false` |

Base Color is decoded by the **GPU** through an sRGB internal format, so
`gammaSpace` is correctly `false` — the shader must not decode it a second time.
ORM and normal maps never consult `gammaSpace` in the PBR shader at all, so the
`true` there is an inert leftover default. Verified by loading a kit `.gltf`
into a bare Babylon scene with none of this tool's code: identical flags.

### Unlit mode

**Unlit mode** (`PBRMaterial.unlit`) drops lighting entirely and shows raw
albedo. It stays available as a toggle for reading a very dark prop, but it is
**not** used for thumbnails any more — it throws away far too much (all shading,
all form, every specular cue) and the flat-tile problem it was introduced to
solve turned out to be the exposure bug above.

Raw albedo leaves the darkest props very nearly black, so unlit mode adds a flat
**emissive lift** of `UNLIT_LIFT = 0.16`. Emissive is *additive* and is honoured
in unlit mode — measured on a 0.08 albedo, the rendered pixel goes 81 → 199 with
a 0.5 emissive — which is exactly what a near-black texture needs; scaling the
albedo instead would leave black black. Two things make this fiddly:

- Materials carrying an **`emissiveTexture`** are the light strips, and their
  `emissiveColor` *multiplies* that texture. Overwriting it would break them
  rather than lift anything, so `applyViewportMode()` skips them.
- The kit's own emissive values are written by `applyKitValues()` *after* the
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
Percentage `background-position` is measured against *(box − image)*, so with a
12-frame sheet the step between frames is `1/11` of the range, not `1/12`;
driving it explicitly avoids that off-by-one entirely.

**All thumbnail-scene work is serialised through one promise chain.** Stills and
turntables share a single scene and a single camera, but they are triggered
independently — stills by scrolling, turntables by hovering — so they *will*
overlap in normal use. A turntable holds its module in the scene across 12
renders, and any still rendered in that window came out with **two different
modules in the picture**. `exclusive()` queues every use of the scene, and
`loadFrameAndRun()` additionally sweeps any leftover mesh before loading, so a
missed disposal cannot photobomb the next tile either.

---

## Relationship to the Blender pipeline

`build_ship.py` is superseded as the authoring path. What carried over is data,
not code:

| From `build_ship.py` | Now |
|---|---|
| kit material values | `public/data/kit_materials.json` |
| grid constants (4 m tile, door slide 1.51, trigger 3.5) | snap config + `DEFAULT_DOOR` |
| `validate()` / `check_door_clearance()` | live checks in the inspector |
| manifest schema | `manifest.js` |
| EEVEE render / compositor / probes | still Blender's job, if you want stills |

## Testing

```
npm test           # spins up a private server and runs every suite
```

The runner starts its own server instance on port 5199 pointed at a **throwaway
export directory**, runs the three suites against it, then deletes it. A test
run therefore cannot touch a real ship — earlier the suites saved and exported
straight into `export/`, which would have overwritten whatever you were working
on. `SHIP_EXPORT_DIR` and `SHIP_PORT` override `config.json` if you want to
point a server anywhere else.

Individual suites can still be run by hand, but they will not guess a server:

```
TOOL_URL=http://localhost:5199/ node test/smoke.mjs      # catalogue, materials, markers, manifest, export
TOOL_URL=http://localhost:5199/ node test/interact.mjs   # ghost, drag, hover, wheel, camera, keyboard
TOOL_URL=http://localhost:5199/ node test/e2e.mjs        # clean-state build, save, artefact preservation
```

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
cycling, door markers, portal and adjacency derivation, manifest
round-trip, save rotation and .glb export.

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

* **It is not the manifest.** A background write must never overwrite the ship
  you last chose to save. Recovery is a copy you reach for, not a thing that
  happens to your work.
* **Every one is kept.** Each write rotates the previous copy to a timestamped
  name, so a long session leaves hundreds of small files. That is the point: the
  value of an auto-save is having the state from *before* whatever went wrong,
  and you cannot know in advance which one that is. They are a few kilobytes
  each, and deleting them is one command.
* **It keeps its own baseline.** An auto-save does *not* clear "you have unsaved
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
  "kitDir":    "../kit",     // the bundled modules, mounted read-only at /kit
  "exportDir": "../export",  // ship.glb + ship_manifest.json
  "envDir":    "../env",     // HDRI, served at /env
  "port": 5180
}
```

**Every path is resolved against `config.json` itself, not the working
directory**, so the relative defaults keep pointing at this copy's own folders
wherever the editor is cloned or moved to. An absolute path still wins, for
anyone pointing the tool at a kit held somewhere else.

The kit ships **inside** the editor (`kit/`, 35 MB, CC0 — see `license.txt`),
so there is nothing to download or buy before the first run. It used to be an
absolute path to wherever the pack happened to be unzipped, which meant the
editor only ran on the machine that authored it.

`SHIP_EXPORT_DIR` and `SHIP_PORT` override `exportDir` and `port` — that is how
the test runner keeps itself away from real data.
