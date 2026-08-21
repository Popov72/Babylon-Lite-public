import { createRequire } from "node:module";
import { toolUrl } from "./target.mjs";

const require = createRequire("D:/alexis/TombRaider/Popov72/Babylon.js/package.json");
const { chromium } = require("playwright");

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
const errors = [];
page.on("console", (message) => {
  if (message.type() === "error") errors.push(message.text());
});
page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));

try {
  await page.goto(toolUrl(), { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelectorAll("#palette-list .item").length > 0,
    null, { timeout: 60000 });

  const result = await page.evaluate(async () => {
    const ui = await import("/js/behavior-metadata.js");
    const catalog = await ui.loadBehaviorMetadata();

    const makeHost = () => {
      const host = document.createElement("div");
      document.body.append(host);
      return host;
    };
    const change = (control, value) => {
      if (control.type === "checkbox") control.checked = value;
      else control.value = value;
      control.dispatchEvent(new Event("change", { bubbles: true }));
    };
    // Every list the UI offers, read the way a person sees it: the placeholder
    // entry is not a choice, so it never counts as one.
    const offered = (select) => [...select.options].map((o) => o.value).filter(Boolean);
    const overrideFor = (host, text) => [...host.querySelectorAll(".behavior-override")]
      .find((label) => label.textContent.includes(text))?.querySelector("input");
    // The form rebuilds itself whenever a choice changes the shape of what
    // follows, so every handle has to be taken again after each click.
    const addRow = (host, selector = ".behavior-array-add") => host.querySelector(selector).click();
    const sourceSelects = (host) =>
      [...host.querySelectorAll(".behavior-array-row .behavior-array-row select")];
    const eventSelect = (host) =>
      [...host.querySelectorAll(".behavior-array-row > .behavior-object > .behavior-nested")]
        .at(-1).querySelector("select");

    const raisedBy = {
      Door_D00: ["activated", "opened"],
      S1: ["activated", "hummed"],
      S2: ["activated", "opened", "splashed"],
    };
    const intersect = (sources) => {
      const list = (Array.isArray(sources) ? sources : sources ? [sources] : []).filter(Boolean);
      if (!list.length) return [];
      return list.map((s) => raisedBy[s] ?? [])
        .reduce((common, raised) => common.filter((name) => raised.includes(name)));
    };
    const shipOptions = {
      // The MP3s come from the definition file, exactly as the editor merges
      // them in: a sound is chosen from what the game actually ships.
      ...ui.behaviorFileOptions(catalog),
      nearbyEntities: [], entities: ["Door_D00", "S1", "S2"], fluidSim: [],
      events: ["activated", "hummed", "opened", "splashed"],
      eventsOfSources: intersect,
    };

    // ---- an entity list, never a typed name -------------------------------
    const linkedHost = makeHost();
    const linkedForm = ui.createBehaviorForm(linkedHost, {
      metadata: ui.behaviorMetadata(catalog, "stdLiquefaction"),
      value: { legacyFlag: { kept: true } },
      inherited: { liquefiable: true, fluidSim: ["liquid-slow"] },
      scope: "assignment",
      options: { ...shipOptions, nearbyEntities: ["S1", "S2"] },
    });
    change(overrideFor(linkedHost, "Linked entities"), true);
    addRow(linkedHost);
    addRow(linkedHost);
    const linkedTyped = linkedHost.querySelectorAll(".behavior-array-row input[type=text]").length;
    const linkedOffers = offered(linkedHost.querySelector(".behavior-array-row select"));
    change([...linkedHost.querySelectorAll(".behavior-array-row select")][1], "S2");

    // ---- sources first, then the events they all raise --------------------
    const actionHost = makeHost();
    const actionForm = ui.createBehaviorForm(actionHost, {
      metadata: ui.behaviorMetadata(catalog, "enableEntity"),
      value: {},
      inherited: {},
      scope: "assignment",
      options: shipOptions,
    });
    change(actionHost.querySelector(".behavior-override input"), true);
    addRow(actionHost);
    const subscriptionLabels = [...actionHost.querySelectorAll(".behavior-array-row .behavior-label")]
      .map((label) => label.textContent);
    const emptySourceEvents = offered(eventSelect(actionHost));
    addRow(actionHost, ".behavior-array-row .behavior-array-add");
    const singleSource = actionForm.read().events[0].source;
    addRow(actionHost, ".behavior-array-row .behavior-array-add");
    change(sourceSelects(actionHost)[1], "S2");
    const eventOffers = offered(eventSelect(actionHost));
    change(eventSelect(actionHost), "opened");
    const eventTyped = actionHost.querySelectorAll(".behavior-object input[type=text]").length;

    // ---- the room filter --------------------------------------------------
    const scopedHost = makeHost();
    ui.createBehaviorForm(scopedHost, {
      metadata: ui.behaviorMetadata(catalog, "enableEntity"),
      value: {},
      inherited: {},
      scope: "assignment",
      options: {
        ...shipOptions,
        chunks: ["CH00", "CH01"],
        currentChunks: ["CH00"],
        entitiesByChunk: { CH00: ["S1"], CH01: ["Door_D00", "S2"] },
      },
      entityScope: {},
    });
    change(scopedHost.querySelector(".behavior-override input"), true);
    addRow(scopedHost);
    addRow(scopedHost, ".behavior-array-row .behavior-array-add");
    const scopeModes = offered(scopedHost.querySelector(".behavior-scope select"));
    const inRoom = offered(sourceSelects(scopedHost)[0]);
    change(scopedHost.querySelector(".behavior-scope select"), "all");
    const shipWide = offered(sourceSelects(scopedHost)[0]);

    // ---- a new event may still be named where one originates --------------
    const triggerHost = makeHost();
    const triggerForm = ui.createBehaviorForm(triggerHost, {
      metadata: ui.behaviorMetadata(catalog, "trigger"),
      value: {},
      inherited: {},
      scope: "assignment",
      options: shipOptions,
    });
    const beforeTrigger = triggerForm.validate();
    change(triggerHost.querySelector(".behavior-override input"), true);
    const enter = triggerHost.querySelector(".behavior-object select");
    change(enter, "activated");
    const newValue = [...enter.options].map((o) => o.value).find((v) => v.startsWith("\u0000"));
    change(triggerHost.querySelector(".behavior-object select"), newValue);
    change(triggerHost.querySelector(".behavior-picker-new input"), "chargeStarted");

    const pickHost = makeHost();
    const pickForm = ui.createBehaviorForm(pickHost, {
      metadata: ui.behaviorMetadata(catalog, "pickEntity"),
      value: {},
      inherited: {},
      scope: "assignment",
      options: shipOptions,
    });
    change(overrideFor(pickHost, "Event to raise"), true);
    const pickSelects = [...pickHost.querySelectorAll('[data-behavior-key="raiseEvent"] select')];
    change(pickSelects[0], "Door_D00");
    change(pickSelects[1], "activated");
    change(overrideFor(pickHost, "Sound"), true);
    const soundOffers = offered(pickHost.querySelector('[data-behavior-key="sound"] select'));

    // ---- inherited values read as prose, never as JSON --------------------
    const soundsHost = makeHost();
    ui.createBehaviorForm(soundsHost, {
      metadata: ui.behaviorMetadata(catalog, "weaponLiquefactor"),
      value: {},
      inherited: { sounds: { quickSplash: ["waterQuickSplash"], bigSplash: ["waterBigSplash"] } },
      scope: "assignment",
      options: shipOptions,
    });
    const inheritedSounds = [...soundsHost.querySelectorAll(".behavior-inherited")]
      .map((node) => node.textContent).join(" | ");

    const dynamicHost = makeHost();
    const dynamicForm = ui.createBehaviorForm(dynamicHost, {
      metadata: ui.behaviorMetadata(catalog, "dynamic"),
      value: {},
      inherited: {},
      scope: "assignment",
      options: {},
    });
    change(dynamicHost.querySelector('[data-behavior-key="mass"] .behavior-override input'), true);
    const defaultMass = dynamicForm.read().mass;
    change(dynamicHost.querySelector('[data-behavior-key="mass"] input[type=number]'), "0");
    const badMass = dynamicForm.validate();

    const collisionHost = makeHost();
    const collisionForm = ui.createBehaviorForm(collisionHost, {
      metadata: ui.behaviorMetadata(catalog, "setCollisionShape"),
      value: {},
      scope: "definition",
      options: {},
    });
    change(collisionHost.querySelector('[data-behavior-key="type"] select'), "mesh");

    const definitions = new Map([
      ["stdLiquefaction", { liquefiable: true }],
      ["trigger", {}],
    ]);
    const entityMap = new Map([
      ["triggerOwner", [{ name: "trigger", onIntersection: { enterEvent: "activated" } }]],
      ["quietOwner", [{ name: "trigger", onIntersection: { exitEvent: "left" } }]],
    ]);
    const raised = ui.collectRaisedEventNames(catalog, definitions, entityMap);
    const raisedByOne = ui.eventsRaisedBy(catalog, definitions, entityMap, "triggerOwner");
    const raisedByAll = ui.eventsRaisedByAll(catalog, definitions, entityMap,
      ["triggerOwner", "quietOwner"]);

    const editor = await import("/js/editor.js");
    document.querySelector("#btn-bhv-library").click();
    document.querySelector("#btn-bhv-new").click();
    change(document.querySelector("#bhv-name"), "dynamic");
    document.querySelector("#btn-bhv-save").click();
    const uiCreatedDefinition = editor.getBehaviorDef("dynamic");
    editor.setBehaviorDef("stdLiquefaction", {
      liquefiable: true, fluidSim: ["liquid-slow"], legacyFlag: { kept: true },
    });
    editor.state.entities.set("source", [{
      name: "stdLiquefaction", linked: ["S1", "S2"], legacyAssignment: 7,
    }]);
    document.querySelector("#btn-bhv-library").click();
    change(document.querySelector("#bhv-list"), "stdLiquefaction");
    const libraryFields = document.querySelector("#bhv-fields").textContent;
    const manifest = await import("/js/manifest.js");
    const serialized = manifest.buildManifest();
    document.querySelector("#btn-bhv-close").click();

    // ---- a rename carries every reference with it -------------------------
    editor.clearAll();
    editor.setBehaviorDef("stdLiquefaction", { liquefiable: true });
    editor.setBehaviorDef("enableEntity", {});
    const kit = await import("/js/kit.js");
    const module = [...kit.getCatalogue().byId.keys()]
      .find((id) => id.startsWith("Modular SciFi MegaKit/Props/"));
    const crateId = (await editor.placeAt(module, [0, 0, 0])).id;
    const watcherId = (await editor.placeAt(module, [4, 0, 0])).id;
    editor.renamePlacement(crateId, "Crate");
    editor.renamePlacement(watcherId, "Watcher");
    editor.state.entities.set("Watcher", [
      { name: "stdLiquefaction", linked: ["Crate", "Other"] },
      { name: "enableEntity", events: [{ source: ["Crate", "Other"], name: "activated" }] },
      { name: "pickEntity", raiseEvent: { target: "Crate", event: "activated" } },
    ]);
    editor.setBehaviorDef("enableEntity", { events: [{ source: "Crate", name: "activated" }] });
    editor.renamePlacement(crateId, "Barrel");
    const renamed = {
      assignments: editor.state.entities.get("Watcher"),
      definition: editor.getBehaviorDef("enableEntity"),
      staleKey: editor.state.entities.has("Crate"),
    };
    // The renamed element's own behaviours travel with it, and a group must not
    // end up pointing at its owner.
    editor.state.entities.set("Barrel", [{ name: "stdLiquefaction", linked: ["Watcher"] }]);
    editor.renamePlacement(watcherId, "Barrel");
    const selfLink = editor.state.entities.get("Barrel");
    const byChunk = manifest.liveEntitiesByChunk();

    return {
      metadataCount: ui.behaviorMetadataNames(catalog).length,
      linked: linkedForm.read(),
      linkedTyped,
      linkedOffers,
      subscriptionLabels,
      emptySourceEvents,
      action: actionForm.read(),
      singleSource,
      eventOffers,
      eventTyped,
      scopeModes,
      inRoom,
      shipWide,
      triggerBefore: beforeTrigger,
      trigger: triggerForm.read(),
      pick: pickForm.read(),
      soundOffers,
      inheritedSounds,
      defaultMass,
      badMass,
      collision: collisionForm.read(),
      uiCreatedDefinition,
      raised,
      raisedByOne,
      raisedByAll,
      serializedDefinition: serialized.behaviors.stdLiquefaction,
      serializedAssignment: serialized.entities.source,
      libraryFields,
      renamed,
      selfLink,
      byChunkRooms: Object.keys(byChunk).length,
      rawJsonEditorPresent: !!document.querySelector("#bhv-json"),
    };
  });

  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const fail = (message) => { throw new Error(message); };

  if (result.metadataCount !== 19) fail(`expected 19 behavior descriptions, got ${result.metadataCount}`);

  // Lists, not typing.
  if (result.linkedTyped !== 0) fail("linked entities are still typed into a text box");
  if (!same(result.linkedOffers, ["S1", "S2"])) {
    fail(`linked entities were not offered from the room: ${JSON.stringify(result.linkedOffers)}`);
  }
  if (!same(result.linked.linked, ["S1", "S2"])) {
    fail(`linked entity array did not round-trip: ${JSON.stringify(result.linked)}`);
  }
  if (!result.linked.legacyFlag?.kept) fail("unknown assignment field was not preserved");

  // Sources before events, and only the events they share.
  const labels = (result.subscriptionLabels ?? []).map((text) => text.replace(" *", ""));
  if (labels[0] !== "Sources" || labels[1] !== "Event") {
    fail(`sources must be asked for before the event: ${JSON.stringify(result.subscriptionLabels)}`);
  }
  if (result.emptySourceEvents.length) {
    fail(`events were offered before a source was chosen: ${JSON.stringify(result.emptySourceEvents)}`);
  }
  if (result.singleSource !== "Door_D00") {
    fail(`one event source was not serialized as a string: ${JSON.stringify(result.singleSource)}`);
  }
  if (!same(result.eventOffers, ["activated", "opened"])) {
    fail(`event list was not the intersection of its sources: ${JSON.stringify(result.eventOffers)}`);
  }
  if (result.eventTyped !== 0) fail("an event can still be typed into a text box");
  if (result.action.events?.[0]?.name !== "opened"
      || !same(result.action.events[0].source, ["Door_D00", "S2"])) {
    fail(`event source/name UI failed: ${JSON.stringify(result.action)}`);
  }

  // The room filter.
  if (!same(result.scopeModes, ["chunk", "chosen", "all"])) {
    fail(`entity scope choices were wrong: ${JSON.stringify(result.scopeModes)}`);
  }
  if (!same(result.inRoom, ["S1"])) fail(`room filter did not narrow: ${JSON.stringify(result.inRoom)}`);
  if (!same(result.shipWide, ["Door_D00", "S1", "S2"])) {
    fail(`ship-wide scope did not widen: ${JSON.stringify(result.shipWide)}`);
  }

  // Naming a genuinely new event is still possible where one originates.
  if (!result.triggerBefore.some((message) => message.includes("onIntersection is required"))) {
    fail(`required trigger object was not validated: ${JSON.stringify(result.triggerBefore)}`);
  }
  if (result.trigger.onIntersection?.enterEvent !== "chargeStarted") {
    fail(`a new event could not be named: ${JSON.stringify(result.trigger)}`);
  }
  if (result.pick.raiseEvent?.target !== "Door_D00" || result.pick.raiseEvent?.event !== "activated") {
    fail(`nested pick event was not edited: ${JSON.stringify(result.pick)}`);
  }
  if (!result.soundOffers.includes("pickItem") || !result.soundOffers.includes("waterBigSplash")) {
    fail(`sounds were not offered from the definition file: ${JSON.stringify(result.soundOffers)}`);
  }

  // No JSON anywhere a person reads.
  if (!result.inheritedSounds.includes("quickSplash: waterQuickSplash")
      || /[{}"[]/.test(result.inheritedSounds)) {
    fail(`inherited value was not written as prose: ${result.inheritedSounds}`);
  }
  if (/[{}]|":/.test(result.libraryFields)) fail(`behaviour library shows JSON: ${result.libraryFields}`);

  if (result.defaultMass !== 10 || !result.badMass.some((m) => m.includes("greater than 0"))) {
    fail(`number default/validation failed: ${result.defaultMass}, ${JSON.stringify(result.badMass)}`);
  }
  if (result.collision.type !== "mesh") fail(`enum field failed: ${JSON.stringify(result.collision)}`);
  if (result.uiCreatedDefinition?.dynamic !== true) {
    fail(`library UI did not create a typed definition: ${JSON.stringify(result.uiCreatedDefinition)}`);
  }
  for (const event of ["activated", "startLiquefaction", "cancelLiquefaction", "endLiquefaction"]) {
    if (!result.raised.includes(event)) fail(`raised event "${event}" was not discovered`);
  }
  if (!same(result.raisedByOne, ["activated"])) {
    fail(`events of one source were wrong: ${JSON.stringify(result.raisedByOne)}`);
  }
  if (result.raisedByAll.length) {
    fail(`sources sharing no event must offer none: ${JSON.stringify(result.raisedByAll)}`);
  }
  if (!result.serializedDefinition.legacyFlag?.kept
      || result.serializedAssignment.behaviors[0].legacyAssignment !== 7) {
    fail("manifest serialization deleted unknown legacy behavior data");
  }
  if (!result.libraryFields.includes("Fluid simulations")
      || !result.libraryFields.includes("Splash sound category")) {
    fail(`behavior library did not render typed fields: ${result.libraryFields}`);
  }

  // A rename must reach every reference, in assignments and definitions alike.
  const [group, subscription, raise] = result.renamed.assignments ?? [];
  if (!same(group?.linked, ["Barrel", "Other"])) {
    fail(`rename did not follow a linked group: ${JSON.stringify(group)}`);
  }
  if (!same(subscription?.events?.[0]?.source, ["Barrel", "Other"])) {
    fail(`rename did not follow an event source: ${JSON.stringify(subscription)}`);
  }
  if (raise?.raiseEvent?.target !== "Barrel") {
    fail(`rename did not follow a raise target: ${JSON.stringify(raise)}`);
  }
  if (result.renamed.definition?.events?.[0]?.source !== "Barrel") {
    fail(`rename did not follow into a definition: ${JSON.stringify(result.renamed.definition)}`);
  }
  if (result.renamed.staleKey) fail("the old entity key survived the rename");
  if (result.selfLink?.[0]?.linked?.length) {
    fail(`a rename linked an element to itself: ${JSON.stringify(result.selfLink)}`);
  }
  if (!result.byChunkRooms) fail("entities were not filed by room");

  if (result.rawJsonEditorPresent) fail("raw behavior JSON editor is still present");
  if (errors.length) fail(`browser errors:\n${[...new Set(errors)].join("\n")}`);

  console.log("behavior editor:", JSON.stringify(result));
} finally {
  await browser.close();
}
