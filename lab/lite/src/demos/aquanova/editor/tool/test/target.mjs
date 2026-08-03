/**
 * Which server a suite is allowed to talk to.
 *
 * The suites are destructive: they call clearAll(), place and delete elements,
 * change settings and let the auto-save tick fire. Run one against the editor
 * you are actually building the ship in and it will write a one-instance
 * manifest over your auto-save and an empty hull set over your collision file.
 * That is not hypothetical - it happened, and the recovery was only possible
 * because the server keeps timestamped backups.
 *
 * So there is no default. `npm test` starts an isolated server on its own port
 * with its own export directory and passes TOOL_URL; anything else has to say
 * out loud where it is pointing.
 */
export function toolUrl() {
  const url = process.env.TOOL_URL;
  if (!url) {
    console.error([
      "",
      "  This suite has no server to talk to, and will not guess one.",
      "",
      "  It is destructive - it clears the scene, places and deletes elements,",
      "  changes settings and lets auto-save fire - so pointing it at the editor",
      "  you build the ship in would overwrite your work.",
      "",
      "  Run `npm test`, which starts an isolated server with its own export",
      "  directory, or set TOOL_URL yourself if you really mean to.",
      "",
    ].join("\n"));
    process.exit(2);
  }
  if (/:5180(\/|$)/.test(url) && process.env.TOOL_URL_I_MEAN_IT !== "yes") {
    console.error([
      "",
      `  Refusing to run against ${url}.`,
      "",
      "  5180 is the port the editor runs on for real work. Set",
      "  TOOL_URL_I_MEAN_IT=yes if this really is a throwaway instance.",
      "",
    ].join("\n"));
    process.exit(2);
  }
  return url;
}
