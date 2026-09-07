// Test runner.
//
// Spins up a private server instance pointed at a throwaway export directory,
// runs every suite against it, then tears the whole thing down. Nothing here
// can touch the real export folder, so running the tests can never overwrite a
// ship you are actually working on.

import { spawn } from "node:child_process";
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, "..");
const PORT = Number(process.env.SHIP_TEST_PORT || 5199);
const URL = `http://localhost:${PORT}/`;

const scratch = await fsp.mkdtemp(path.join(os.tmpdir(), "scifiship-test-"));
const collisionScratch = path.join(scratch, "kits");

// e2e checks that a manifest and .glb the tool did not write are preserved on
// first save, so seed a pair that look like something else's output.
await fsp.writeFile(path.join(scratch, "ship_manifest.json"),
  JSON.stringify({ generator: "other", units: "metres", chunks: [] }, null, 2));
await fsp.writeFile(path.join(scratch, "ship.glb"), Buffer.alloc(2048, 7));

const env = {
  ...process.env,
  SHIP_EXPORT_DIR: scratch,
  SHIP_COLLISION_KITS_DIR: collisionScratch,
  SHIP_PORT: String(PORT),
};
const server = spawn(process.execPath, [path.join(ROOT, "server.mjs")],
  { cwd: ROOT, env, stdio: ["ignore", "pipe", "pipe"] });
server.stdout.on("data", (d) => process.stdout.write(`  [server] ${d}`));
server.stderr.on("data", (d) => process.stderr.write(`  [server] ${d}`));

async function waitForServer(timeoutMs = 15000) {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    try {
      const r = await fetch(URL + "api/modules");
      if (r.ok) return true;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("test server did not start");
}

function run(script) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [path.join(HERE, script)], {
      cwd: ROOT,
      env: { ...env, TOOL_URL: URL },
      stdio: "inherit",
    });
    child.on("exit", (code) => resolve(code ?? 1));
  });
}

let failures = 0;
try {
  await waitForServer();
  console.log(`\nscratch export dir: ${scratch}\n`);
  for (const script of ["behavior-editor.mjs", "smoke.mjs", "animation-export.mjs", "interact.mjs", "e2e.mjs"]) {
    console.log(`\n=== ${script} ${"=".repeat(60 - script.length)}`);
    const code = await run(script);
    if (code !== 0) failures++;
    console.log(`=== ${script}: ${code === 0 ? "ok" : `FAILED (exit ${code})`}`);
  }
} finally {
  server.kill();
  // SHIP_TEST_KEEP leaves the scratch export directory behind, for looking at
  // what a suite actually wrote when one of them disagrees with you.
  if (process.env.SHIP_TEST_KEEP) console.log(`kept scratch: ${scratch}`);
  else await fsp.rm(scratch, { recursive: true, force: true }).catch(() => {});
}

console.log(`\n${failures ? `${failures} suite(s) failed` : "all suites passed"}`);
process.exit(failures ? 1 : 0);
