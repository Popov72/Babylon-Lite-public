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
The demo uses it purely as an IBL source (ambient + specular); the ship carries no
punctual lights — its ceiling fixtures are emissive geometry.

## Non-free / restricted assets

_None yet._

Any asset added here that is **not** CC0 (or otherwise freely redistributable)
MUST be listed in this section with its source and license terms, so downstream
users know it cannot be reused freely. Do not commit non-redistributable assets.
