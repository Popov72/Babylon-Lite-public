// Compile-time switch for the demo's debug tooling (inspect overlay, collider overlay, SDF overlay,
// QA hooks).
//
// `__LAB_DEBUG__` is injected by the bundle build (scripts/bundle-demos-core.ts) as a literal, so the
// bundler folds every `if (LAB_DEBUG)` branch and drops the `./debug/` modules entirely. The default
// is OFF: a shipped demo carries none of it.
//
// The dev server injects nothing, and `typeof` on an undeclared identifier is legal, so a plain
// `vite dev` run leaves debug ON — which is what you want while working on the demo. Both forms fold
// to a constant, so nothing is paid at runtime either way.

declare const __LAB_DEBUG__: boolean | undefined;

export const LAB_DEBUG: boolean = typeof __LAB_DEBUG__ === "undefined" ? true : __LAB_DEBUG__;
