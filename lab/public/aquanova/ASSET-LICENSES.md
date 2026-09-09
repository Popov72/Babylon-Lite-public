# Aquanova demo — asset attribution & licensing

This document records the origin and license of every third-party asset used by
the **Aquanova** demo, so anyone building on this repository knows exactly what
they may reuse.

## CC0 (Public Domain) — free for anyone to use

The composed ship export `ship.glb` (with `ship_manifest.json`) comes from the
**Modular SciFi MegaKit** by **Quaternius** (https://quaternius.com,
https://www.patreon.com/quaternius). `ship.glb` is a Blender-authored assembly of
the CC0 kit modules (built in ../../../SciFiShip, which is outside this
repository); the demo loads that single file directly.

- **License:** CC0 1.0 Universal (Public Domain Dedication)
  https://creativecommons.org/publicdomain/zero/1.0/
- **What this means:** these assets are in the public domain. Anyone may copy,
  modify, and redistribute them, for any purpose, without attribution. (Credit
  to Quaternius is appreciated but not required.)
- **What is included here:** `ship.glb` ONLY — a single self-contained binary
  glTF holding the assembled geometry together with the kit's trim/wall texture
  set (20 images, 2048×2048, embedded as PNG; nothing is referenced by URI), plus
  the `ship_manifest.json` that describes the assembly.
- **What is NOT included here:** the individual kit modules the ship was built
  from. The per-module `models/*.gltf` + `*.bin` templates and the loose
  `T_*.png` trim textures used to sit alongside this file and have been removed —
  everything they contributed now lives inside `ship.glb`. Re-download the kit
  from Quaternius if you need the modules themselves. The kit's `.blend` source
  files and engine project files were never included.

The image-based lighting environment `bank_vault_2k.hdr` is the **"bank_vault"**
HDRI from **Poly Haven** (https://polyhaven.com/a/bank_vault), also **CC0 1.0**.
It is retained only by the ship editor for the non-baked authoring view. Baked
preview and the Aquanova runtime use the generated bounded local probes instead.

The skybox cube faces in `skybox/` (`sky_px.png` … `sky_nz.png`) were generated
with **Space 3D** by **wwwtyro** (https://tools.wwwtyro.net/space-3d/), a
procedural space-scene generator whose source
(https://github.com/wwwtyro/space-3d) is released under **the Unlicense**, a
public-domain dedication. The generated output may be used for any purpose,
commercial or not, with no attribution required.

- **What this is used for:** backdrop only — what you see through the ship's
  openings. It is _not_ an IBL source.
- **Why it is not committed:** purely size, as with the ship and the HDR. It is
  freely redistributable, but at ~4.5 MB it would only pay off if a fresh clone
  could run the demo — and it cannot, because `ship.glb` is gitignored too.
  Re-generate or drop the six faces into `lab/public/aquanova/skybox/` to see it;
  the demo logs a warning and runs against the clear colour without them.

## Non-free / restricted assets

The first-person Liquefactor weapon models under `weapons/` were supplied for
this project. No separate redistribution license has been documented for them,
so treat those files as project-only assets rather than reusable CC0 content.

The `sounds/spark.mp3` effect was likewise supplied for the Aquanova project.
No separate redistribution license has been documented for it, so treat it as
a project-only asset rather than reusable CC0 content.

Any asset added here that is **not** CC0 (or otherwise freely redistributable)
MUST be listed in this section with its source and license terms, so downstream
users know it cannot be reused freely. Do not commit non-redistributable assets.
