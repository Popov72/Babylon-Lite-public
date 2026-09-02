# Fluid module layout

Reusable fluid functionality is grouped by responsibility:

| Folder | Responsibility |
| --- | --- |
| `core/` | Public runtime facade, shared contracts, configuration, allocation, scheduling, GPU readback, and runtime bindings |
| `solvers/` | PBF, FLIP, MLS-MPM, PB-MPM, solver layouts, math, and shared WGSL |
| `rendering/` | Particle, surface, polygon, foam, compositor, render-profile, and render-resource code |
| `controls/` | Shared controls panel, capabilities, transactions, bindings, flow editor, and grid visualization |
| `authoring/` | Presets, Blender JSON, authored grid state, migrations, and method-independent state |
| `sampling/` | Mesh/volume particle sampling and per-particle UV transfer |

Applications should continue importing the supported API from the root `babylon-lite` entry point. Scene-specific hosts provide assets and orchestration; reusable fluid calculations and policies remain in this directory.
