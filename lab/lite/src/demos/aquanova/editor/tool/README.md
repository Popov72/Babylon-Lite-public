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
| `doors[].position` | `view` |
| `entities[].behaviors[].direction` | |

The right-hand column is the tool's own reload data: it exists to rebuild the
editor and never leaves it. The left-hand column is everything the game
consumes; getting it wrong mirrors the ship against its own geometry, which is
how a player start authored to face the door came out facing away from it.

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
placed — **whatever the cursor hovers**, and only then the selection. Hover
outranks the selection deliberately: pointing at something is a more immediate
statement of intent than a selection made earlier, and it saves clearing the
selection before acting on a different element.

| | |
|---|---|
| Place | click a palette tile to arm it, then click in the viewport. The module stays armed for repeat placement. |
| Select | **click** an element · `Ctrl`- or `Shift`-click to add or remove · click empty space to clear |
| Move | **drag** an element (elements stay solid, button held), or **`M`** to pick the selection up and carry it hands-free as a translucent ghost — click to drop, `Esc` to put it back. Dragging one that is already selected moves the **whole selection**; dragging an unselected one selects just it first. **`V`, or the `Drag` combo,** cycles the drag axis: `X/Z (floor)` → `Y (up/down)` → `X only` → `Z only` — safe to change mid-drag. `Esc` or right-click mid-drag puts everything back. |
| Frame | **double-click** an element |
| Turn | **`R`** — about the element's own origin |
| Axes | **`X`** — show one element's **world** X/Y/Z arrows · **`Shift+X`** — its own **local** axes, which is what scaling acts on. With several selected, the one **nearest the cursor** gets them; an armed ghost counts too. The same key again hides them, the other key re-aims them, and pressing either with nothing selected or hovered hides them |
| Axis modes | one letter per action, its settings behind the modifiers. **`V`** cycles the drag axis, **`Shift+V`** the move step (**`Ctrl+V`** back) · **`Shift+R`** the rotation axis, **`Ctrl+R`** the rotation angle · **`Shift+F`** the scale axis, **`Ctrl+F`** the scale step. `Ctrl+Shift` reverses the two step cycles. All three `Ctrl` pairs are claimed from the browser — reload, the find bar and paste |
| Mirror | **`F`** — mirrors on the current Scale axis (`all` is treated as X) |
| Turn as a group | **`Alt` + `R`** — the selection swings about a shared pivot, snapped to the move grid so it lands back on-grid |
| Resize | **`Shift` + wheel** — steps by the Scale snap on the current scale axis |
| What gets edited | the ghost if one is being placed, otherwise **whatever is hovered**, otherwise the selection |
| Rotation axis | `Shift+R` cycles Y → X → Z. Y first: it is the only one a modular kit usually needs. |
| Scale axis | `Shift+F` cycles all → X → Y → Z |
| Build plane | numpad `+` / `-` (or main-row `+` / `-`) by the Move step; numpad `.` jumps it to the top of the hovered element |
| Select | **quick** left-click · `Ctrl`- or `Shift`-click adds to the selection · click empty space clears it · **drag from empty space to rubber-band**, or press **Rect select** to start the rectangle on top of a module. `Ctrl` or `Shift` while banding adds. To reach something behind a door portal, `Shift+H` the door — a ghosted element is click-through |
| Chunks | the **Chunk** button toggles isolation — pressed (orange), every chunk but the one in the dropdown is hidden · `+` adds a chunk · **Rename** renames the active one everywhere it is used · `Assign` moves the selection into the active one |
| Hide | `Shift+H` cycles the selection **50% → hidden → 50%** — half alpha (and click-through) to see past something, then gone · `H` returns everything to fully opaque · the **Ghost** slider sets how see-through that first state is. Undoable, but not saved — a reload starts with everything visible |
| Id | inspector `Id` row — read-only. The tool's handle for the element and its node name in `ship.glb` when no `Name` is set; doors, portals and behaviours all reference it, so it is not editable. In a multi-selection it names the element whose transform the fields below show |
| Name | inspector `Name` field — the element's **node** name in `ship.glb` (primitives are numbered off it), shared on purpose: elements with the same name share one behaviour entry. Shown in the corner overlay instead of the module id |
| Behaviour | inspector panel — attach library behaviours to the element's node name, and pick the `linked` nodes a liquefiable one melts with · **Edit behaviours…** opens the library (name + free-form JSON body) |
| Eyedropper | `Alt`-click a placed element to arm its module |
| Nudge | arrow keys move the selection on X/Z, `PageUp`/`PageDown` on Y |
| Steps | toolbar dropdowns — Move defaults to **1 m**, and **`Shift+V`** cycles it (`Ctrl+V` backwards). Move can be **off** (free positioning while dragging); Rot and Scale are keyboard *step sizes*, so they have no "off" |
| Camera | `WASD` flies, `Space`/`C` rise and descend · **right-drag looks** · **right button + wheel sets the fly speed** · `Shift` for 2× · wheel dollies · `F` frames the selection. The left button never moves the camera |
| Lighting | **Env** slider — strength of the image-based lighting, which is where metals get nearly all their brightness · **Exposure** slider. Both are saved in the manifest and restored on Load · **Runtime light** drops the editor's own lights, leaving the HDRI the game actually uses |
| Walk | toolbar checkbox — walk at the player's eye height (1.8 m) instead of flying. `WASD` moves horizontally at the usual speed, the height follows whatever floor is underfoot, and `Space`/`C` are off |
| Undo | `Ctrl+Z` / `Ctrl+Shift+Z` (or `Ctrl+Y`) — whole-layout snapshots, capped by *memory* rather than a fixed count (1000 steps on this ship, fewer as it grows), so *anything* that pushes an entry is undoable: placing, deleting, dragging, turning, scaling, flipping, nudging, hiding, the Env and Exposure sliders, every inspector field and every behaviour edit |
| Edit | **`Ctrl+D` puts a copy of the current element — or of the whole selection — on the cursor** as a ghost, keeping every rotation and mirroring · **`Del`, or the middle mouse button, deletes the hovered element, or the selection if nothing is hovered** (deleting a hovered element leaves the rest of the selection intact) |
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

The inspector shows the selection's **world** bounding box in metres — the world
box, not the local one, so it changes as you rotate. That is the number you want
when checking whether a piece still fits its 4 m tile. With several elements
selected it reports the combined box.

`Del` follows the same precedence as everything else: whatever the cursor is
over first, then the selection. Deleting a hovered element leaves an unrelated
selection intact — pointing at one thing is no reason to forget the others.
Nothing mid-gesture is deletable — `Esc` is the way out of those.

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
looks. It also **raises the build plane to the source's own height**: the ghost
rides that plane, so without it a copy of something on an upper deck would
reappear down at ground level. Moving the plane rather than giving the ghost a
private height keeps one source of truth, and the grid visibly follows so it is
obvious what happened. **A multi-selection is carried too**: the ghost holds a
list of items, each with its own offset, turn and mirroring, so `Ctrl+D` on
twelve walls hands you twelve walls.

### Carrying versus dragging

There are two ways to move something, deliberately, and they differ in what
your hand is doing:

* **Drag** — press, move, release. The elements stay **solid** and follow
  directly. The button is held throughout.
* **Carry** (`M`) — the selection lifts onto the cursor as a **translucent**
  ghost. No button held: move the mouse, turn with `R`, mirror with `F`, fly the
  camera, then click to drop. `Esc` puts everything back where it was.

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

**The wheel belongs to the camera.** It dollies, full stop — that is what a
wheel does in a 3D view, and every attempt to give it a second job fought that
expectation. It no longer rotates anything; `R` does, and it acts on the
hovered element without needing a selection first. Two exceptions, both
deliberate: `Shift` + wheel resizes the current element, and **holding the right
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

**Which axes are live is shown on the gizmo**, in colour rather than only in
words: `X` red, `Y` green, `Z` blue — the standard convention — with the locked
arms dimmed. The toolbar combo names the mode in words a few pixels away; the
gizmo is where you are actually looking while you build.

### The overlay in the corner

Two rows, and only two:

| | |
|---|---|
| **Build plane** | the height new modules land on — `state.gridY` |
| **Current** | what the next key will act on: the ghost `◆`, the hover `▸`, or the selection `■` |

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
space they live in. A **drag** moves along the world axes — `node.position` *is*
world position, since placements have no parent in the editor — so a world gizmo
is the honest answer for moving. **Scaling is local**, so on anything that has
been turned (most of a ship built from a modular kit) a world gizmo cannot tell
you which way `X` will grow.

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
| cube on the tip | the axis a scale acts on (all three for `all`) | `Shift+F` |
| the chip at the origin | the move step, in metres, or `free` | `Shift+V` |
| the chip inside the curved arrow | the turn angle, in degrees | `Ctrl+R` |
| a chip on each lit cube | the scale step | `Ctrl+F` |

Each modal setting has exactly one marker, and each marker means exactly one
thing — `V` never touches the ring, `Shift+R` never touches the brightness. The
curved arrow sweeps three quarters of a turn rather than closing into a full
ring, so it reads as a direction of travel and not as a collar.

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

Hit-testing projects **all eight corners** of each element's world box and takes
the screen-space extent. Two corners would badly under-report a rotated piece —
the projection of a world AABB is not the AABB of the projection. Anything whose
screen box overlaps the rectangle is selected, occluded or not, because "what is
inside this rectangle" is a screen question, not a visibility one. Corners
behind the camera are dropped: they project to a mirrored point that would
otherwise stretch the box across the whole viewport.

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
* **Doors resize two ways.** The inspector's `W`/`H` fields set the authored
  opening; `Shift+wheel` and the inspector's `Scale` fields scale the node like
  any other element. Both were once blocked for markers — `scaleCurrent()`
  filtered them out and `applyInspector()` guarded the write — which just made
  doors feel broken. The exported `width`/`height` fold the node scale in
  (`width × |scale.x|`), because `portalOf()` reads the **world matrix** and so
  the portal grows with the node regardless; leaving the raw width in the
  manifest would have made a door disagree with its own portal.

Portals and the adjacency graph are derived from doors, so there is nothing
extra to keep in sync.

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
under each name and disposes the duplicates *with their textures*; `thumbs.js`
does the same on its own engine. Placing 300 modules costs 8 materials and 9
textures.

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

Individual suites can still be run by hand against an already-running server:

```
node test/smoke.mjs        # catalogue, materials, markers, manifest, export
node test/interact.mjs     # ghost, drag, hover, wheel, camera, keyboard
node test/e2e.mjs          # clean-state build, save, artefact preservation
```

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
