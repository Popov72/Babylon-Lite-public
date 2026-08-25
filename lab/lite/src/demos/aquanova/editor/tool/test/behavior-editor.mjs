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
    await page.waitForFunction(() => document.querySelectorAll("#palette-list .item").length > 0, null, { timeout: 60000 });

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
    // The form rebuilds itself whenever a choice changes the shape of what
    // follows, so every handle has to be taken again after each click.
    const addRow = (host, selector = ".behavior-array-add") => host.querySelector(selector).click();
        const sourceSelects = (host) => [...host.querySelectorAll(".behavior-array-row .behavior-array-row select")];
        const eventSelect = (host) => [...host.querySelectorAll(".behavior-array-row > .behavior-object > .behavior-nested")].at(-1).querySelector("select");
        const nestedField = (host, label) =>
          [...host.querySelectorAll(".behavior-nested")].find(
            (row) => row.querySelector(":scope > .behavior-label")?.textContent.replace(" *", "") === label
          );

    const raisedBy = {
      CH01: ["notVisible", "visible"],
      Door_D00: ["activated", "opened"],
      S1: ["activated", "hummed"],
      S2: ["activated", "opened", "splashed"],
    };
    const intersect = (sources) => {
      const list = (Array.isArray(sources) ? sources : sources ? [sources] : []).filter(Boolean);
      if (!list.length) return [];
            return list.map((s) => raisedBy[s] ?? []).reduce((common, raised) => common.filter((name) => raised.includes(name)));
    };
    // The MP3s are listed off the game's own sound folder by the editor
    // server and reach a form through the catalogue, so the harness asks for
    // them exactly where the editor does rather than restating them.
    const liveSounds = (await import("/js/kit.js")).getCatalogue().sounds ?? [];
    const shipOptions = {
      sounds: liveSounds,
            nearbyEntities: [],
            eventEntities: ["Door_D00", "S1", "S2"],
            fluidSim: ["capsule-filling", "flip-test"],
            fluidEmitters: (fluidSim) => ({ "capsule-filling": ["Capsule inlet"], "flip-test": ["Manual source"] })[fluidSim] ?? [],
            fluidSinks: (fluidSim) => ({ "flip-test": ["New sink"] })[fluidSim] ?? [],
      events: ["activated", "hummed", "opened", "splashed"],
      eventsOfSources: intersect,
    };

    // ---- an entity list, never a typed name -------------------------------
    const linkedHost = makeHost();
    const linkedForm = ui.createBehaviorForm(linkedHost, {
            metadata: ui.behaviorMetadata(catalog, "liquefaction"),
      value: {},
      inherited: { fluidSim: ["liquid-slow"] },
      scope: "assignment",
      options: { ...shipOptions, nearbyEntities: ["S1", "S2"] },
    });
    // Every other field of the behaviour now renders its own editor too, so
    // reach for this one by name rather than by being first.
    const linkedField = linkedHost.querySelector('[data-behavior-key="linked"]');
    addRow(linkedField);
    addRow(linkedHost.querySelector('[data-behavior-key="linked"]'));
        const linkedRows = () => linkedHost.querySelector('[data-behavior-key="linked"]');
    const linkedTyped = linkedRows().querySelectorAll(".behavior-array-row input[type=text]").length;
    const linkedOffers = offered(linkedRows().querySelector(".behavior-array-row select"));
    change([...linkedRows().querySelectorAll(".behavior-array-row select")][1], "S2");

    // ---- sources first, then the events they all raise --------------------
    const actionHost = makeHost();
    const actionForm = ui.createBehaviorForm(actionHost, {
      metadata: ui.behaviorMetadata(catalog, "enableEntity"),
      value: {},
      inherited: {},
      scope: "assignment",
      options: shipOptions,
    });
    addRow(actionHost);
        const subscriptionLabels = [...actionHost.querySelectorAll(".behavior-array-row .behavior-label")].map((label) => label.textContent);
    const emptySourceEvents = offered(eventSelect(actionHost));
    addRow(actionHost, ".behavior-array-row .behavior-array-add");
    const singleSource = actionForm.read().events[0].source;
    addRow(actionHost, ".behavior-array-row .behavior-array-add");
    change(sourceSelects(actionHost)[1], "S2");
    const eventOffers = offered(eventSelect(actionHost));
    change(eventSelect(actionHost), "opened");
    const eventTyped = actionHost.querySelectorAll(".behavior-object input[type=text]").length;

    // ---- fluid actions expose only the target their action can use --------
    const fluidHost = makeHost();
    const fluidForm = ui.createBehaviorForm(fluidHost, {
      metadata: ui.behaviorMetadata(catalog, "fluidSimulation"),
      value: { fluidSim: "capsule-filling" },
      inherited: {},
      scope: "assignment",
      options: shipOptions,
    });
    const fluidField = () => fluidHost.querySelector('[data-behavior-key="eventActions"]');
    const fluidNested = (label) =>
      [...fluidField().querySelectorAll(".behavior-nested")].find((row) => row.querySelector(":scope > .behavior-label")?.textContent.replace(" *", "") === label);
    addRow(fluidField());
    addRow(fluidNested("Sources"));
    change(fluidNested("Sources").querySelector("select"), "S2");
    change(fluidNested("Event").querySelector("select"), "opened");
    const fluidActionOffers = offered(fluidNested("Action").querySelector("select"));
    change(fluidNested("Action").querySelector("select"), "enableEmitter");
    const capsuleEmitters = offered(fluidNested("Emitter").querySelector("select"));
    const emitterOnly = {
      emitter: !!fluidNested("Emitter"),
      sink: !!fluidNested("Sink"),
    };
    change(fluidHost.querySelector('[data-behavior-key="fluidSim"] select'), "flip-test");
    const flipEmitters = offered(fluidNested("Emitter").querySelector("select"));
    change(fluidNested("Emitter").querySelector("select"), "Manual source");
    change(fluidNested("Action").querySelector("select"), "enableSink");
    const sinkOnly = {
      emitter: !!fluidNested("Emitter"),
      sink: !!fluidNested("Sink"),
      offers: offered(fluidNested("Sink").querySelector("select")),
    };
    change(fluidNested("Sink").querySelector("select"), "New sink");

    const chunkHost = makeHost();
    const chunkForm = ui.createBehaviorForm(chunkHost, {
      metadata: ui.behaviorMetadata(catalog, "enableEntity"),
      value: { events: [{ source: "CH01", name: "visible" }] },
      inherited: {},
      scope: "assignment",
      options: {
        ...shipOptions,
        eventEntities: ["CH01", ...shipOptions.eventEntities],
        optionLabels: { eventEntities: { CH01: "CH01 chunk" } },
        events: [...shipOptions.events, "notVisible", "visible"],
      },
    });
    const chunkSourceSelect = sourceSelects(chunkHost)[0];
    const chunkSource = {
      value: chunkForm.read().events[0].source,
      label: [...chunkSourceSelect.options].find((option) => option.value === "CH01")?.textContent,
      events: offered(eventSelect(chunkHost)),
    };

    // ---- sound actions expose catalogue sounds and timing -----------------
    const soundHost = makeHost();
    const soundForm = ui.createBehaviorForm(soundHost, {
      metadata: ui.behaviorMetadata(catalog, "sound"),
      value: {
        cues: [
          {
            events: [{ source: "S1", name: "hummed" }],
            sound: "stepMetallic",
            action: "stop",
            delay: 1.5,
            fade: 2,
          },
          { sound: "pickItem", action: "play" },
        ],
      },
      inherited: {},
      scope: "assignment",
      options: {
        ...shipOptions,
        chunks: ["CH00"],
        currentChunks: ["CH00"],
        entitiesByChunk: { CH00: ["S1", "S2"] },
      },
      entityScope: {},
    });
    const behaviorSoundOffers = offered(nestedField(soundHost, "Sound").querySelector("select"));
    const soundScopeCount = soundHost.querySelectorAll(".behavior-scope").length;
    const soundCueControls = {
      add: !!soundHost.querySelector('[data-behavior-key="cues"] button[title="Add Sound actions"]'),
      remove: !!soundHost.querySelector('[data-behavior-key="cues"] button[title="Remove"]'),
    };

    // ---- an empty vocabulary is still a list ------------------------------
    // A field that declares an optionsSource must never fall back to a text
    // box when its list happens to be empty: that is precisely when a typed
    // name is a guess, and the hint explaining the emptiness would never be
    // reached. An editor server too old to list the game's MP3s is one way to
    // arrive here; a weapon that has not been given its categories is another.
    const mutedHost = makeHost();
    ui.createBehaviorForm(mutedHost, {
      metadata: ui.behaviorMetadata(catalog, "sound"),
      value: { cues: [{}] },
      inherited: {},
      scope: "assignment",
      options: { ...shipOptions, sounds: [] },
    });
    const mutedField = nestedField(mutedHost, "Sound");
    const mutedSound = {
      typed: mutedField.querySelectorAll("input[type=text]").length,
      disabled: !!mutedField.querySelector("select")?.disabled,
    };

    // ...and one that may be named by hand says so, and stays usable: the
    // clips of a behaviour in the library cannot be known, since it has no
    // owner to read them off yet.
    const clipHost = makeHost();
    ui.createBehaviorForm(clipHost, {
      metadata: ui.behaviorMetadata(catalog, "playAnimation"),
      value: {},
      inherited: {},
      scope: "definition",
      options: { ...shipOptions, animations: [] },
    });
    const clipField = clipHost.querySelector('[data-behavior-key="animation"]');
    const clipSelect = clipField.querySelector("select");
    const emptyClips = {
      typed: clipField.querySelectorAll("input[type=text]").length,
      disabled: !!clipSelect?.disabled,
      offersNew: [...(clipSelect?.options ?? [])].some((o) => o.textContent.includes("name a new one")),
      hint: !!clipField.querySelector(".behavior-help"),
    };

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
    const pickSelects = [...pickHost.querySelectorAll('[data-behavior-key="raiseEvent"] select')];
    change(pickSelects[0], "Door_D00");
    change(pickSelects[1], "activated");
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
    // An inherited value is shown by the editors themselves now, so it has to be
    // legible there: the categories the weapon defines, in the fields that edit
    // them, and no JSON anywhere on the way.
    const inheritedSounds = {
            hint: [...soundsHost.querySelectorAll(".behavior-inherited")].map((node) => node.textContent).join(" | "),
      shown: [...soundsHost.querySelectorAll("input, select")].map((node) => node.value),
      text: soundsHost.textContent,
      greyed: !!soundsHost.querySelector(".behavior-inheriting"),
    };

    const dynamicHost = makeHost();
    const dynamicForm = ui.createBehaviorForm(dynamicHost, {
      metadata: ui.behaviorMetadata(catalog, "dynamic"),
      value: {},
      inherited: {},
      scope: "assignment",
      options: {},
    });
    // Looking at a field is not editing it: an untouched row writes nothing and
    // names its default rather than quietly adopting it.
    const massInput = dynamicHost.querySelector('[data-behavior-key="mass"] input[type=number]');
    const untouchedMass = { placeholder: massInput.placeholder, written: dynamicForm.read().mass ?? null };
    change(massInput, "0");
    const badMass = dynamicForm.validate();

    // ---- taking a value over, and handing it back -------------------------
    const revertHost = makeHost();
    const revertForm = ui.createBehaviorForm(revertHost, {
      metadata: ui.behaviorMetadata(catalog, "dynamic"),
      value: {},
      inherited: { mass: 4 },
      scope: "assignment",
      options: {},
    });
    const massOf = () => revertHost.querySelector('[data-behavior-key="mass"] input[type=number]');
    const revert = {
      opensOnInherited: massOf().value,
      greyedBefore: !!revertHost.querySelector(".behavior-inheriting"),
      offeredBefore: !!revertHost.querySelector(".behavior-revert"),
      readBefore: revertForm.read().mass ?? null,
    };
    change(massOf(), "9");
    revert.readAfter = revertForm.read().mass ?? null;
    revert.greyedAfter = !!revertHost.querySelector(".behavior-inheriting");
    revert.offeredAfter = !!revertHost.querySelector(".behavior-revert");
    revertHost.querySelector(".behavior-revert").click();
    revert.readReverted = revertForm.read().mass ?? null;
    revert.greyedReverted = !!revertHost.querySelector(".behavior-inheriting");
    revert.showsInheritedAgain = massOf().value;

    const collisionHost = makeHost();
    const collisionForm = ui.createBehaviorForm(collisionHost, {
      metadata: ui.behaviorMetadata(catalog, "setCollisionShape"),
      value: {},
      scope: "definition",
      options: {},
    });
    change(collisionHost.querySelector('[data-behavior-key="type"] select'), "mesh");
        const fluidShapeInputs = collisionHost.querySelectorAll('[data-behavior-key="fluidSimShape"] input[type=number]');
        ["1", "2", "3", "4", "0.75", "1"].forEach((value, index) => change(fluidShapeInputs[index], value));

        const definitions = new Map([["stdLiquefaction", { base: "liquefaction" }]]);
    const entityMap = new Map([
      ["triggerOwner", [{ name: "trigger", onIntersection: { enterEvent: "activated" } }]],
      ["quietOwner", [{ name: "trigger", onIntersection: { exitEvent: "left" } }]],
      ["fluidOwner", [{ name: "fluidSimulation", fluidSim: "capsule-filling", eventActions: [] }]],
    ]);
    const raised = ui.collectRaisedEventNames(catalog, definitions, entityMap);
    const raisedByOne = ui.eventsRaisedBy(catalog, definitions, entityMap, "triggerOwner");
        const raisedByAll = ui.eventsRaisedByAll(catalog, definitions, entityMap, ["triggerOwner", "quietOwner"]);
    const fluidSourceEvents = ui.eventsRaisedByAll(catalog, definitions, entityMap, ["fluidOwner"]);
    const externalChunkEvents = (source) => (source === "CH01" ? ["visible", "notVisible"] : []);
    const chunkSourceEvents = ui.eventsRaisedByAll(catalog, definitions, entityMap, ["CH01"], externalChunkEvents);
    const mixedChunkEvents = ui.eventsRaisedByAll(catalog, definitions, entityMap, ["CH01", "fluidOwner"], externalChunkEvents);

    const editor = await import("/js/editor.js");
    document.querySelector("#btn-bhv-library").click();
    document.querySelector("#btn-bhv-new").click();
        change(document.querySelector("#bhv-name"), "heavyDynamic");
        change(document.querySelector("#bhv-base"), "dynamic");
    document.querySelector("#btn-bhv-save").click();
        const uiCreatedDefinition = editor.getBehaviorPreset("heavyDynamic");
    editor.setBehaviorDef("stdLiquefaction", {
            base: "liquefaction",
            fluidSim: ["liquid-slow"],
    });
        editor.state.entities.set("source", [
            {
                name: "stdLiquefaction",
                linked: ["S1", "S2"],
            },
        ]);
    document.querySelector("#btn-bhv-library").click();
    change(document.querySelector("#bhv-list"), "stdLiquefaction");
    const libraryFields = document.querySelector("#bhv-fields").textContent;
    const manifest = await import("/js/manifest.js");
    const serialized = manifest.buildManifest();
    document.querySelector("#btn-bhv-close").click();
        const catalogCollision = editor.setBehaviorDef("dynamic", { base: "dynamic" });

    // ---- a rename carries every reference with it -------------------------
    editor.clearAll();
        editor.setBehaviorDef("stdLiquefaction", { base: "liquefaction" });
        editor.setBehaviorDef("doorEnabler", { base: "enableEntity" });
    const kit = await import("/js/kit.js");
        const module = [...kit.getCatalogue().byId.keys()].find((id) => id.startsWith("Modular SciFi MegaKit/Props/"));
    const crateId = (await editor.placeAt(module, [0, 0, 0])).id;
    const watcherId = (await editor.placeAt(module, [4, 0, 0])).id;
    editor.renamePlacement(crateId, "Crate");
    editor.renamePlacement(watcherId, "Watcher");
    editor.state.entities.set("Watcher", [
      { name: "stdLiquefaction", linked: ["Crate", "Other"] },
            { name: "doorEnabler", events: [{ source: ["Crate", "Other"], name: "activated" }] },
      { name: "pickEntity", raiseEvent: { target: "Crate", event: "activated" } },
    ]);
        editor.setBehaviorDef("doorEnabler", {
            base: "enableEntity",
            events: [{ source: "Crate", name: "activated" }],
        });
    editor.renamePlacement(crateId, "Barrel");
    const renamed = {
      assignments: editor.state.entities.get("Watcher"),
            definition: editor.getBehaviorDef("doorEnabler"),
      staleKey: editor.state.entities.has("Crate"),
    };
    // The renamed element's own behaviours travel with it, and a group must not
    // end up pointing at its owner.
    editor.state.entities.set("Barrel", [{ name: "stdLiquefaction", linked: ["Watcher"] }]);
    editor.renamePlacement(watcherId, "Barrel");
    const selfLink = editor.state.entities.get("Barrel");
    const byChunk = manifest.liveEntitiesByChunk();

    // ---- what an entity picker is allowed to offer ------------------------
    // Only what takes part in the event traffic: something carrying at least
    // one behaviour, or a door. A module's exported pieces - a
    // `_primitive<i>` per submesh - carry nothing, so none of them can appear,
    // while the manifest still knows every one of them by name.
    editor.clearAll();
    const quietId = (await editor.placeAt(module, [0, 0, 0])).id;
    const loudId = (await editor.placeAt(module, [4, 0, 0])).id;
    editor.renamePlacement(quietId, "Quiet");
    editor.renamePlacement(loudId, "Loud");
    editor.state.entities.set("Loud", [{ name: "stdLiquefaction" }]);
    editor.state.chunks.push("CH_EVENT");
    const markers = await import("/js/markers.js");
    const door = markers.addDoor([2, 0, 0], { silent: true });
    const participants = {
      offered: manifest.eventEntityNames(),
      door: door.id,
      exported: [...manifest.liveEntityNames()].filter((name) => name.startsWith("Loud")),
    };

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
      fluid: fluidForm.read(),
      fluidActionOffers,
      chunkSource,
      liveFluidFlow: kit.getCatalogue().fluidSimFlow,
      capsuleEmitters,
      flipEmitters,
      emitterOnly,
      sinkOnly,
      sound: soundForm.read(),
      behaviorSoundOffers,
      soundScopeCount,
      soundCueControls,
      liveSounds,
      mutedSound,
      emptyClips,
      scopeModes,
      inRoom,
      shipWide,
      triggerBefore: beforeTrigger,
      trigger: triggerForm.read(),
      pick: pickForm.read(),
      soundOffers,
      inheritedSounds,
      untouchedMass,
      revert,
      badMass,
      collision: collisionForm.read(),
            badFluidShape: ui.validateBehaviorConfig(catalog, "setCollisionShape", {
                fluidSimShape: {
                    type: "hollowCylinder",
                    start: [0, 0, 0],
                    height: 1,
                    innerRadius: 1,
                    outerRadius: 0.5,
                },
            }),
      uiCreatedDefinition,
            catalogCollision,
            serializedPresets: serialized.behaviorPresets,
            serializedHasLegacyLibrary: Object.hasOwn(serialized, "behaviors"),
      raised,
      raisedByOne,
      raisedByAll,
      fluidSourceEvents,
      chunkSourceEvents,
      mixedChunkEvents,
      libraryFields,
      renamed,
      selfLink,
      participants,
      byChunkRooms: Object.keys(byChunk).length,
      overrideBoxes: document.querySelectorAll(".behavior-override").length,
      rawJsonEditorPresent: !!document.querySelector("#bhv-json"),
    };
  });

  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const fail = (message) => {
        throw new Error(message);
    };

    if (result.metadataCount !== 19) fail(`expected 19 behavior descriptions, got ${result.metadataCount}`);

  // Lists, not typing.
  if (result.linkedTyped !== 0) fail("linked entities are still typed into a text box");
  if (!same(result.linkedOffers, ["S1", "S2"])) {
    fail(`linked entities were not offered from the room: ${JSON.stringify(result.linkedOffers)}`);
  }
  if (!same(result.linked.linked, ["S1", "S2"])) {
    fail(`linked entity array did not round-trip: ${JSON.stringify(result.linked)}`);
  }
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
    if (result.action.events?.[0]?.name !== "opened" || !same(result.action.events[0].source, ["Door_D00", "S2"])) {
    fail(`event source/name UI failed: ${JSON.stringify(result.action)}`);
  }
  if (!same(result.capsuleEmitters, ["Capsule inlet"]) || !same(result.flipEmitters, ["Manual source"])) {
    fail(`fluid emitter options did not follow fluidSim: ${JSON.stringify([result.capsuleEmitters, result.flipEmitters])}`);
  }
  if (!result.emitterOnly.emitter || result.emitterOnly.sink || result.sinkOnly.emitter || !result.sinkOnly.sink) {
    fail(`fluid action target fields were not conditional: ${JSON.stringify([result.emitterOnly, result.sinkOnly])}`);
  }
  if (!same(result.sinkOnly.offers, ["New sink"])) {
    fail(`fluid sink options were wrong: ${JSON.stringify(result.sinkOnly.offers)}`);
  }
  if (!result.fluidActionOffers.includes("pauseSimulation") || !result.fluidActionOffers.includes("unpauseSimulation")) {
    fail(`fluid pause actions were not offered: ${JSON.stringify(result.fluidActionOffers)}`);
  }
  if (
    result.chunkSource.value !== "CH01" ||
    result.chunkSource.label !== "CH01 chunk" ||
    !same(result.chunkSource.events, ["notVisible", "visible"])
  ) {
    fail(`chunk event source failed: ${JSON.stringify(result.chunkSource)}`);
  }
  if (!result.liveFluidFlow?.["flip-test"]?.emitters?.includes("Manual source") || !result.liveFluidFlow?.["flip-test"]?.sinks?.includes("New sink")) {
    fail(`server fluid-flow catalogue was incomplete: ${JSON.stringify(result.liveFluidFlow?.["flip-test"])}`);
  }
  const [stopCue, startupCue] = result.sound.cues ?? [];
  if (
    stopCue?.events?.[0]?.source !== "S1" ||
    stopCue?.events?.[0]?.name !== "hummed" ||
    stopCue?.sound !== "stepMetallic" ||
    stopCue?.action !== "stop" ||
    stopCue?.delay !== 1.5 ||
    stopCue?.fade !== 2 ||
    startupCue?.sound !== "pickItem" ||
    startupCue?.action !== "play" ||
    result.soundScopeCount !== 1 ||
    !result.soundCueControls.add ||
    !result.soundCueControls.remove ||
    !result.behaviorSoundOffers.includes("stepMetallic")
  ) {
    fail(`sound behavior form failed: ${JSON.stringify([result.sound, result.behaviorSoundOffers])}`);
  }
  // The picker offers the game's sound folder, whole: the list used to be typed
  // into behavior-definitions.json and had already drifted from what is on disk.
  if (!result.liveSounds.length || !same(result.behaviorSoundOffers, result.liveSounds)) {
    fail(`sound list did not come from the game's sound folder: ${JSON.stringify([result.liveSounds, result.behaviorSoundOffers])}`);
  }
  // A field that declares a list stays a list even when the list is empty:
  // degrading to a text box turns "the editor has nothing to offer you" into
  // "type whatever you like", and hides the sentence that says why.
  if (result.mutedSound.typed || !result.mutedSound.disabled) {
    fail(`an empty sound list did not stay a disabled picker: ${JSON.stringify(result.mutedSound)}`);
  }
  if (result.emptyClips.typed || result.emptyClips.disabled || !result.emptyClips.offersNew || !result.emptyClips.hint) {
    fail(`an empty animation list was not a nameable picker with a hint: ${JSON.stringify(result.emptyClips)}`);
  }
  const fluidAction = result.fluid.eventActions?.[0];
  if (
    result.fluid.fluidSim !== "flip-test" ||
    fluidAction?.source !== "S2" ||
    fluidAction?.event !== "opened" ||
    fluidAction?.action !== "enableSink" ||
    fluidAction?.sink !== "New sink" ||
    "emitter" in fluidAction
  ) {
    fail(`fluid event action did not round-trip: ${JSON.stringify(result.fluid)}`);
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

  // No JSON anywhere a person reads, and an inherited value legible in the
  // fields that edit it rather than described beside them.
  for (const shown of ["quickSplash", "waterQuickSplash", "bigSplash", "waterBigSplash"]) {
    if (!result.inheritedSounds.shown.includes(shown)) {
      fail(`inherited value was not shown in its editors: ${JSON.stringify(result.inheritedSounds)}`);
    }
  }
  if (!result.inheritedSounds.greyed) {
    fail("an inherited value was not marked as the behaviour's");
  }
  if (/[{}"[]/.test(result.inheritedSounds.text)) {
    fail(`inherited value was written as JSON: ${result.inheritedSounds.text}`);
  }
    if (/[{}]|":/.test(result.libraryFields)) fail(`behavior preset editor shows JSON: ${result.libraryFields}`);

  // Nothing is overridden by ticking a box: the field is simply edited, and
  // handed back with the one control that remains.
  if (result.overrideBoxes) fail(`${result.overrideBoxes} override checkboxes are still rendered`);
    if (result.untouchedMass.written !== null || !result.untouchedMass.placeholder.includes("10")) {
    fail(`an untouched field must stay unset and name its default: ${JSON.stringify(result.untouchedMass)}`);
  }
  if (!result.badMass.some((m) => m.includes("greater than 0"))) {
    fail(`number validation failed: ${JSON.stringify(result.badMass)}`);
  }
  const r = result.revert;
  if (r.opensOnInherited !== "4" || !r.greyedBefore || r.offeredBefore || r.readBefore !== null) {
    fail(`a field must open on the behaviour's value without taking it: ${JSON.stringify(r)}`);
  }
  if (r.readAfter !== 9 || r.greyedAfter || !r.offeredAfter) {
    fail(`editing a field did not take the value over: ${JSON.stringify(r)}`);
  }
  if (r.readReverted !== null || !r.greyedReverted || r.showsInheritedAgain !== "4") {
    fail(`the value could not be handed back to the behaviour: ${JSON.stringify(r)}`);
  }
    if (
        result.collision.type !== "mesh" ||
        JSON.stringify(result.collision.fluidSimShape) !==
            JSON.stringify({
                type: "hollowCylinder",
                start: [1, 2, 3],
                height: 4,
                innerRadius: 0.75,
                outerRadius: 1,
            })
    ) {
        fail(`collision shape fields failed: ${JSON.stringify(result.collision)}`);
    }
    if (!result.badFluidShape.includes("fluidSimShape.outerRadius must be greater than fluidSimShape.innerRadius.")) {
        fail(`invalid hollow-cylinder radii were accepted: ${JSON.stringify(result.badFluidShape)}`);
    }
    if (result.uiCreatedDefinition?.base !== "dynamic" || result.uiCreatedDefinition?.dynamic !== true) {
        fail(`preset UI did not create a typed preset: ${JSON.stringify(result.uiCreatedDefinition)}`);
    }
    if (result.catalogCollision !== false) {
        fail(`a preset reused a base behavior name: ${JSON.stringify(result.catalogCollision)}`);
    }
  if (
      result.serializedPresets?.stdLiquefaction?.base !== "liquefaction" ||
      "liquefiable" in result.serializedPresets.stdLiquefaction ||
      result.serializedHasLegacyLibrary
  ) {
      fail(`manifest did not serialize behavior presets: ${JSON.stringify(result.serializedPresets)}`);
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
  if (!same(result.fluidSourceEvents, ["emissionComplete"])) {
    fail(`fluidSimulation source did not offer emissionComplete: ${JSON.stringify(result.fluidSourceEvents)}`);
  }
  if (!same(result.chunkSourceEvents, ["notVisible", "visible"]) || result.mixedChunkEvents.length) {
    fail(`external chunk events were not resolved generically: ${JSON.stringify([result.chunkSourceEvents, result.mixedChunkEvents])}`);
  }
    if (!result.libraryFields.includes("Fluid simulations") || !result.libraryFields.includes("Splash sound category")) {
        fail(`behavior preset editor did not render typed fields: ${result.libraryFields}`);
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

  // An entity picker offers what takes part in events, and nothing else.
  const parts = result.participants.exported.filter((name) => /_primitive\d+$/.test(name));
  if (!parts.length) fail("the fixture exported no parts, so the rule below proves nothing");
  if (!result.participants.offered.includes("Loud")) {
    fail(`an element carrying a behaviour was not offered: ${JSON.stringify(result.participants)}`);
  }
  if (!result.participants.offered.includes(result.participants.door)) {
    fail(`a door was not offered: ${JSON.stringify(result.participants)}`);
  }
  if (!result.participants.offered.includes("CH_EVENT")) {
    fail(`a chunk was not offered as an event source: ${JSON.stringify(result.participants)}`);
  }
  if (result.participants.offered.includes("Quiet")) {
    fail(`an element carrying no behaviour was offered: ${JSON.stringify(result.participants)}`);
  }
  if (result.participants.offered.some((name) => parts.includes(name))) {
    fail(`the pieces of an element were offered: ${JSON.stringify(result.participants)}`);
  }

  if (result.rawJsonEditorPresent) fail("raw behavior JSON editor is still present");
  if (errors.length) fail(`browser errors:\n${[...new Set(errors)].join("\n")}`);

  console.log("behavior editor:", JSON.stringify(result));
} finally {
  await browser.close();
}
