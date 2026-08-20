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

    const linkedHost = makeHost();
    const linkedForm = ui.createBehaviorForm(linkedHost, {
      metadata: ui.behaviorMetadata(catalog, "stdLiquefaction"),
      value: { legacyFlag: { kept: true } },
      inherited: { liquefiable: true, fluidSim: ["liquid-slow"] },
      scope: "assignment",
      options: { nearbyEntities: ["S1", "S2"], entities: ["S1", "S2"], events: [], fluidSim: [] },
    });
    const linkedOverride = [...linkedHost.querySelectorAll(".behavior-override")]
      .find((label) => label.textContent.includes("Linked entities"))?.querySelector("input");
    change(linkedOverride, true);
    linkedHost.querySelector(".behavior-array-add").click();
    linkedHost.querySelector(".behavior-array-add").click();
    const linkedInputs = [...linkedHost.querySelectorAll(".behavior-array-row input")];
    change(linkedInputs[1], "S2");

    const actionHost = makeHost();
    const actionForm = ui.createBehaviorForm(actionHost, {
      metadata: ui.behaviorMetadata(catalog, "enableEntity"),
      value: {},
      inherited: {},
      scope: "assignment",
      options: {
        nearbyEntities: [], entities: ["Door_D00", "S1", "S2"],
        events: ["activated", "startLiquefaction"], fluidSim: [],
      },
    });
    change(actionHost.querySelector(".behavior-override input"), true);
    actionHost.querySelector(".behavior-array-add").click();
    const eventInputs = [...actionHost.querySelectorAll(".behavior-array-row input")];
    change(eventInputs[0], "activated");
    const sourceAdd = actionHost.querySelector(".behavior-array-row .behavior-array-add");
    sourceAdd.click();
    const singleSource = actionForm.read().events[0].source;
    actionHost.querySelector(".behavior-array-row .behavior-array-add").click();
    const sourceInputs = [...actionHost.querySelectorAll(".behavior-array-row .behavior-array-row input")];
    change(sourceInputs[1], "S2");

    const triggerHost = makeHost();
    const triggerForm = ui.createBehaviorForm(triggerHost, {
      metadata: ui.behaviorMetadata(catalog, "trigger"),
      value: {},
      inherited: {},
      scope: "assignment",
      options: { nearbyEntities: [], entities: [], events: ["activated"], fluidSim: [] },
    });
    const beforeTrigger = triggerForm.validate();
    change(triggerHost.querySelector(".behavior-override input"), true);
    const triggerInputs = [...triggerHost.querySelectorAll(".behavior-object input[type=text]")];
    change(triggerInputs[0], "activated");

    const pickHost = makeHost();
    const pickForm = ui.createBehaviorForm(pickHost, {
      metadata: ui.behaviorMetadata(catalog, "pickEntity"),
      value: {},
      inherited: {},
      scope: "assignment",
      options: { nearbyEntities: [], entities: ["Door_D00"], events: ["activated"], fluidSim: [] },
    });
    const raiseOverride = [...pickHost.querySelectorAll(".behavior-override")]
      .find((label) => label.textContent.includes("Event to raise"))?.querySelector("input");
    change(raiseOverride, true);
    const pickInputs = [...pickHost.querySelectorAll('[data-behavior-key="raiseEvent"] input[type=text]')];
    change(pickInputs[0], "Door_D00");
    change(pickInputs[1], "activated");

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
    const massInput = dynamicHost.querySelector('[data-behavior-key="mass"] input[type=number]');
    change(massInput, "0");
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
    const entities = new Map([
      ["triggerOwner", [{ name: "trigger", onIntersection: { enterEvent: "activated" } }]],
    ]);
    const raised = ui.collectRaisedEventNames(catalog, definitions, entities);

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

    return {
      metadataCount: ui.behaviorMetadataNames(catalog).length,
      linked: linkedForm.read(),
      action: actionForm.read(),
      singleSource,
      triggerBefore: beforeTrigger,
      trigger: triggerForm.read(),
      pick: pickForm.read(),
      defaultMass,
      badMass,
      collision: collisionForm.read(),
      uiCreatedDefinition,
      raised,
      serializedDefinition: serialized.behaviors.stdLiquefaction,
      serializedAssignment: serialized.entities.source,
      libraryFields,
      rawJsonEditorPresent: !!document.querySelector("#bhv-json"),
    };
  });

  const expected = {
    linked: ["S1", "S2"],
    source: ["Door_D00", "S2"],
  };
  if (result.metadataCount !== 19) throw new Error(`expected 19 behavior descriptions, got ${result.metadataCount}`);
  if (JSON.stringify(result.linked.linked) !== JSON.stringify(expected.linked)) {
    throw new Error(`linked entity array did not round-trip: ${JSON.stringify(result.linked)}`);
  }
  if (!result.linked.legacyFlag?.kept) throw new Error("unknown assignment field was not preserved");
  if (result.action.events?.[0]?.name !== "activated"
      || JSON.stringify(result.action.events[0].source) !== JSON.stringify(expected.source)) {
    throw new Error(`event source string/array UI failed: ${JSON.stringify(result.action)}`);
  }
  if (result.singleSource !== "Door_D00") {
    throw new Error(`one event source was not serialized as a string: ${JSON.stringify(result.singleSource)}`);
  }
  if (!result.triggerBefore.some((message) => message.includes("onIntersection is required"))) {
    throw new Error(`required trigger object was not validated: ${JSON.stringify(result.triggerBefore)}`);
  }
  if (result.trigger.onIntersection?.enterEvent !== "activated") {
    throw new Error(`optional trigger event was not edited: ${JSON.stringify(result.trigger)}`);
  }
  if (result.pick.raiseEvent?.target !== "Door_D00"
      || result.pick.raiseEvent?.event !== "activated") {
    throw new Error(`nested pick event was not edited: ${JSON.stringify(result.pick)}`);
  }
  if (result.defaultMass !== 10
      || !result.badMass.some((message) => message.includes("greater than 0"))) {
    throw new Error(`number default/validation failed: ${result.defaultMass}, ${JSON.stringify(result.badMass)}`);
  }
  if (result.collision.type !== "mesh") {
    throw new Error(`enum field failed: ${JSON.stringify(result.collision)}`);
  }
  if (result.uiCreatedDefinition?.dynamic !== true) {
    throw new Error(`library UI did not create a typed definition: ${JSON.stringify(result.uiCreatedDefinition)}`);
  }
  for (const event of ["activated", "startLiquefaction", "cancelLiquefaction", "endLiquefaction"]) {
    if (!result.raised.includes(event)) throw new Error(`raised event "${event}" was not discovered`);
  }
  if (!result.serializedDefinition.legacyFlag?.kept
      || result.serializedAssignment.behaviors[0].legacyAssignment !== 7) {
    throw new Error("manifest serialization deleted unknown legacy behavior data");
  }
  if (!result.libraryFields.includes("Fluid simulations")
      || !result.libraryFields.includes("Splash sound category")) {
    throw new Error(`behavior library did not render typed fields: ${result.libraryFields}`);
  }
  if (result.rawJsonEditorPresent) throw new Error("raw behavior JSON editor is still present");
  if (errors.length) throw new Error(`browser errors:\n${[...new Set(errors)].join("\n")}`);

  console.log("behavior editor:", JSON.stringify(result));
} finally {
  await browser.close();
}
