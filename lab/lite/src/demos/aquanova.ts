// Aquanova demo — first-person sci-fi shooter tech demo built on Lite.
//
// The player traverses a confined spaceship wielding the "Liquefactor": melt flying alien foes and
// ship props into GPU fluid. The ship is a chunked modular interior authored in Blender from the CC0
// Quaternius "Modular SciFi MegaKit" and exported to a single glTF (ship.glb) with a companion
// ship_manifest.json describing chunks, portals, doors, behaviours and spawns (see
// lab/public/aquanova/ASSET-LICENSES.md and SciFiShip/README.md).
//
// This file is only the BUNDLE ENTRY — the bundler resolves each demo from
// `lab/lite/src/demos/<slug>.ts` — so the implementation lives in `./aquanova/`:
//
//   aquanova/main.ts          scene build, player, liquefaction, fluid, debug overlays
//   aquanova/constants.ts     asset URLs, room heights, glTF -> Lite conversion
//   aquanova/manifest.ts      ship_manifest.json types + loading
//   aquanova/fluid-setting.ts fluidSim/*.json parsing (shared shape with the Liquefactor export)
//   aquanova/colliders.ts     Havok box shell built from the manifest chunks
//   aquanova/sdf-bake.ts      world-space SDF bake used for the fluid's collision
//   aquanova/sdf-overlay.ts   F / Shift+F voxel view of what the fluid actually collides with
//
// `main` is imported and called explicitly rather than relying on a bare side-effect import: the
// package is marked `sideEffects: false`, so an import with no used binding is legal to delete and
// the whole demo would silently vanish from the bundle.

import { main } from "./aquanova/main.js";

main().catch((err) => {
    // eslint-disable-next-line no-console
    console.error("[aquanova] fatal", err);
});
