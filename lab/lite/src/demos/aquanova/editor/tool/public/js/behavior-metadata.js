let catalogPromise = null;

const clone = (value) => value === undefined ? undefined : JSON.parse(JSON.stringify(value));
const own = (object, key) => Object.prototype.hasOwnProperty.call(object ?? {}, key);

export async function loadBehaviorMetadata(url = "/data/behavior-definitions.json") {
  catalogPromise ??= fetch(url, { cache: "no-store" }).then(async (response) => {
    if (!response.ok) throw new Error(`failed to load behavior metadata (${response.status})`);
    const raw = await response.json();
    if (raw?.version !== 1 || !raw.behaviors || typeof raw.behaviors !== "object") {
      throw new Error("behavior metadata must contain version 1 and a behaviors object");
    }
    const resolved = {};
    const resolving = new Set();
    const resolve = (name) => {
      if (resolved[name]) return resolved[name];
      const source = raw.behaviors[name];
      if (!source) throw new Error(`behavior metadata references unknown behavior "${name}"`);
      if (resolving.has(name)) throw new Error(`behavior metadata inheritance cycle at "${name}"`);
      resolving.add(name);
      const parent = source.extends
        ? (raw.behaviors[source.extends] ? resolve(source.extends) : raw.templates?.[source.extends])
        : null;
      if (source.extends && !parent) {
        throw new Error(`behavior metadata references unknown template "${source.extends}"`);
      }
      resolved[name] = {
        ...(parent ? clone(parent) : {}),
        ...clone(source),
        properties: {
          ...(parent?.properties ? clone(parent.properties) : {}),
          ...(source.properties ? clone(source.properties) : {}),
        },
        eventsRaised: clone(source.eventsRaised ?? parent?.eventsRaised ?? []),
      };
      delete resolved[name].extends;
      resolving.delete(name);
      return resolved[name];
    };
    for (const name of Object.keys(raw.behaviors)) resolve(name);
    return { version: raw.version, behaviors: resolved, options: clone(raw.options ?? {}) };
  });
  return catalogPromise;
}

export function behaviorMetadata(catalog, name) {
  return catalog?.behaviors?.[name] ?? null;
}

export function behaviorMetadataNames(catalog) {
  return Object.keys(catalog?.behaviors ?? {}).sort((a, b) => a.localeCompare(b));
}

export function validateBehaviorConfig(catalog, name, value, { partial = false } = {}) {
  const metadata = behaviorMetadata(catalog, name);
  if (!metadata) return [`No metadata describes behavior "${name}".`];
  if (!isPlainObject(value)) return [`Behavior "${name}" must be an object.`];
  return validateObject(metadata.properties ?? {}, value, "", partial);
}

/**
 * The option lists the metadata file ships with, keyed by `optionsSource`.
 *
 * Vocabularies that belong to the game rather than to the ship - the MP3s under
 * `/aquanova/sounds/`, say - have no home in the editor's state, and typing one
 * of them out is exactly the kind of guesswork a picker exists to remove. They
 * live beside the schema that refers to them so the two are edited together.
 */
export function behaviorFileOptions(catalog) {
  return catalog?.options ?? {};
}

export function defaultBehaviorDefinition(metadata) {
  const out = {};
  for (const [key, schema] of Object.entries(metadata?.properties ?? {})) {
    if (schema.definition === false) continue;
    if (schema.type === "constant") out[key] = clone(schema.value);
  }
  return out;
}

export function collectRaisedEventNames(catalog, definitions, entities) {
  const names = new Set();
  for (const [name, definition] of definitions) addRaisedEvents(catalog, name, definition, names);
  for (const list of entities.values()) {
    for (const assignment of list) {
      addRaisedEvents(catalog, assignment.name, effectiveConfig(definitions, assignment), names);
    }
  }
  return sorted(names);
}

/**
 * What one entity raises, rather than what the whole ship raises.
 *
 * A subscription names both the event and whose event it is, and those two are
 * not independent: `startLiquefaction` from a crate that cannot be liquefied is
 * a subscription that will never fire, and nothing in the editor would have
 * said so. Asking the source first and offering only what it raises makes that
 * impossible to author rather than merely discouraged.
 */
export function eventsRaisedBy(catalog, definitions, entities, entityName) {
  const names = new Set();
  for (const assignment of entities.get(String(entityName || "").trim()) ?? []) {
    addRaisedEvents(catalog, assignment.name, effectiveConfig(definitions, assignment), names);
  }
  return sorted(names);
}

/**
 * The events *every* one of these sources raises.
 *
 * Several sources on one subscription mean "any of these, they are equivalent"
 * - the two halves of a fan, the five panels of a door. The behaviour reacts
 * once whichever of them speaks, so an event only one of them raises would make
 * the group behave differently depending on which member fired. The offer is
 * therefore the intersection, not the union.
 */
export function eventsRaisedByAll(catalog, definitions, entities, sources) {
  const list = (Array.isArray(sources) ? sources : sources ? [sources] : [])
    .map((item) => String(item ?? "").trim()).filter(Boolean);
  if (!list.length) return [];
  let common = null;
  for (const source of list) {
    const raised = new Set(eventsRaisedBy(catalog, definitions, entities, source));
    common = common === null ? raised : new Set([...common].filter((name) => raised.has(name)));
    if (!common.size) break;
  }
  return sorted(common ?? new Set());
}

function addRaisedEvents(catalog, behaviorName, config, into) {
  const metadata = behaviorMetadata(catalog, behaviorName);
  for (const event of metadata?.eventsRaised ?? []) {
    if (event.name) into.add(event.name);
    const value = event.property ? valueAt(config, event.property.split(".")) : undefined;
    if (typeof value === "string" && value.trim()) into.add(value.trim());
  }
}

const effectiveConfig = (definitions, assignment) =>
  ({ ...(definitions.get(assignment.name) ?? {}), ...assignment });

const sorted = (names) => [...names].sort((a, b) => a.localeCompare(b));

/** Vocabularies the ship defines: always chosen from, never typed. */
const PICKED_TYPES = new Set(["enum", "entity", "event"]);
/** The one option source the room filter applies to; `nearbyEntities` is already a room. */
const SCOPED_ENTITIES = "eventEntities";
/** Sentinel select value that opens the "name a new one" input. */
const NEW_VALUE = "\u0000new";

/**
 * Point every entity reference at a name that has just changed.
 *
 * A behaviour names the things it watches and acts on, and those names are the
 * *old* ones the moment an element is renamed - a subscription to a source that
 * no longer exists is a behaviour that silently never fires, and nothing in the
 * editor would have said so. The schema already knows which fields hold an
 * entity, so the rewrite follows it rather than a hand-kept list of key names
 * that would fall behind the next behaviour added to the file.
 *
 * Mutates `config` in place and returns how many references it moved.
 */
export function renameEntityReferences(catalog, behaviorName, config, from, to) {
  const before = String(from ?? "").trim();
  const after = String(to ?? "").trim();
  if (!before || !after || before === after || !isPlainObject(config)) return 0;
  return rewrite(behaviorMetadata(catalog, behaviorName)?.properties ?? {}, config);

  function rewrite(properties, value) {
    let moved = 0;
    for (const [key, schema] of Object.entries(properties)) {
      if (!own(value, key)) continue;
      const [next, count] = rewriteValue(schema, value[key]);
      value[key] = next;
      moved += count;
    }
    return moved;
  }

  function rewriteValue(schema, item) {
    if (item === undefined || item === null) return [item, 0];
    if (schema.type === "entity" || schema.type === "entitySource") {
      if (Array.isArray(item)) {
        let moved = 0;
        const next = item.map((name) => {
          if (name !== before) return name;
          moved++;
          return after;
        });
        return [next, moved];
      }
      return item === before ? [after, 1] : [item, 0];
    }
    if (schema.type === "object" && isPlainObject(item)) {
      return [item, rewrite(schema.properties ?? {}, item)];
    }
    if (schema.type === "record" && isPlainObject(item)) {
      let moved = 0;
      for (const [key, child] of Object.entries(item)) {
        const [next, count] = rewriteValue(schema.values ?? {}, child);
        item[key] = next;
        moved += count;
      }
      return [item, moved];
    }
    if (schema.type === "array" && Array.isArray(item)) {
      let moved = 0;
      const next = item.map((child) => {
        const [value, count] = rewriteValue(schema.items ?? {}, child);
        moved += count;
        return value;
      });
      return [next, moved];
    }
    return [item, 0];
  }
}

export function createBehaviorForm(host, {
  metadata,
  value = {},
  inherited = {},
  scope = "definition",
  options = {},
  entityScope = null,
  onChange = null,
}) {
  let draft = clone(value) ?? {};
  let validationHost = null;
  // Owned by the caller so it survives the panel rebuilding this form after
  // every edit; a fresh object per form is fine for one-off uses.
  const rooms = entityScope ?? {};

  const api = {
    read: () => clone(draft),
    validate: (effective = scope === "assignment" ? { ...inherited, ...draft } : draft) =>
      validateObject(metadata?.properties ?? {}, effective, "", scope === "definition"),
    render,
  };

  function changed(structural = false) {
    const errors = api.validate();
    if (validationHost) {
      validationHost.textContent = errors.join(" ");
      validationHost.hidden = errors.length === 0;
    }
    onChange?.(api.read(), errors);
    if (structural) render();
  }

  /**
   * Which rooms the entity pickers offer, and what they therefore contain.
   *
   * A name typed from memory is a name that can be wrong, so every entity is
   * chosen from a list - but "every entity on the ship" is hundreds of names to
   * scroll for the one crate in the room you are looking at. The room the
   * element is in is the useful default, and widening it is one click away for
   * the cases that genuinely reach across the ship, like a door watching the
   * panels of the room next door.
   */
  function scopedEntities(all) {
    if (rooms.mode === "all") return all;
    const here = roomNames();
    return all.filter((name) => here.has(name));
  }

  /** The names the chosen rooms hold, whether or not anything offers them. */
  function roomNames() {
    const chosen = rooms.mode === "chosen" ? (rooms.chunks ?? []) : (options.currentChunks ?? []);
    const byChunk = options.entitiesByChunk ?? {};
    const keep = new Set();
    for (const room of chosen) for (const name of byChunk[room] ?? []) keep.add(name);
    return keep;
  }

  function choicesFor(schema, siblings) {
    const all = optionValues(schema, options, siblings);
    return schema?.optionsSource === SCOPED_ENTITIES ? scopedEntities(all) : all;
  }

  /**
   * Open on the room, unless that would hide something already chosen.
   *
   * An entry authored across the ship must not come back reading "not in the
   * list" merely because the panel opened on the narrowest view of it. The test
   * is the room and not the offer, because widening is the only thing the
   * filter can put right: a name that left the offer - whose behaviours were
   * taken away, say - is missing from every room equally, and opening on the
   * whole ship to find it would only be a longer list that still lacks it.
   */
  function pickInitialScope() {
    if (rooms.mode) return;
    rooms.mode = "all";
    if (!options.entitiesByChunk || !(options.currentChunks ?? []).length) return;
    rooms.mode = "chunk";
    const here = roomNames();
    const missing = entityValues(metadata?.properties ?? {}, draft)
      .some((name) => !here.has(name));
    if (missing) rooms.mode = "all";
  }

  function renderScopeRow() {
    const box = element("div", "behavior-scope");
    const here = (options.currentChunks ?? []).join(", ");
    const mode = document.createElement("select");
    if (here) addOption(mode, "chunk", `this room (${here})`);
    addOption(mode, "chosen", "chosen rooms…");
    addOption(mode, "all", "anywhere on the ship");
    mode.value = rooms.mode;
    mode.addEventListener("change", () => {
      rooms.mode = mode.value;
      if (rooms.mode === "chosen" && !rooms.chunks?.length) {
        rooms.chunks = [...(options.currentChunks ?? [])];
      }
      render();
    });
    box.append(element("span", "behavior-scope-label", "Entities from"), mode);
    if (rooms.mode === "chosen") {
      const picked = document.createElement("select");
      picked.multiple = true;
      picked.size = Math.min(6, Math.max(3, (options.chunks ?? []).length));
      for (const chunk of options.chunks ?? []) {
        addOption(picked, chunk, chunk);
        picked.lastChild.selected = (rooms.chunks ?? []).includes(chunk);
      }
      picked.addEventListener("change", () => {
        rooms.chunks = [...picked.selectedOptions].map((option) => option.value);
        render();
      });
      box.append(picked);
    }
    return box;
  }

  function render() {
    host.replaceChildren();
    host.className = "behavior-form";
    if (!metadata) {
      host.append(element("div", "behavior-validation", "No metadata describes this behavior."));
      return;
    }
    // What the behaviour is for is written where its name is - the applied
    // list's row, the library's picker - rather than above its fields: it is
    // the same sentence every time, and the fields are what the panel is for.
    pickInitialScope();
    if (options.entitiesByChunk && anySchema(metadata.properties ?? {},
      (schema) => schema.optionsSource === SCOPED_ENTITIES)) {
      host.append(renderScopeRow());
    }
    for (const [key, schema] of Object.entries(metadata.properties ?? {})) {
      if (scope === "definition" && schema.definition === false) continue;
      if (scope === "assignment" && schema.assignment === false) continue;
      if (scope === "assignment" && schema.type === "constant") continue;
      host.append(renderTopProperty(key, schema));
    }
    const errors = api.validate();
    validationHost = element("div", "behavior-validation", errors.join(" "));
    validationHost.hidden = errors.length === 0;
    host.append(validationHost);
  }

  /**
   * One top-level field, and where its value comes from.
   *
   * An assignment either sets a value or takes the behaviour's, and that used
   * to be a checkbox you ticked before the field would appear: a click spent
   * saying "yes, I do want to edit the thing I just clicked on", and a row
   * reading `Override Splash sound category` directly above one reading
   * `Splash sound category`. The field is simply editable instead. It opens
   * showing what the behaviour says, greyed to say the value is not yours yet,
   * and the first change makes it yours. Handing it back is the only part that
   * still needs asking for, so that is the only control left - one ↺ beside the
   * label, and only once there is something to hand back.
   *
   * Editing writes to a shadow rather than to the draft, so looking is free:
   * opening a picker, reading what the behaviour set, changing your mind and
   * closing it again must not leave a copy of the inherited value behind, which
   * would quietly detach the element from a definition it is still following.
   */
  function renderTopProperty(key, schema) {
    const wrapper = element("div", "behavior-property");
    wrapper.dataset.behaviorKey = key;
    const assigned = own(draft, key);
    const inheritedValue = scope === "assignment" ? inherited?.[key] : undefined;
    // What may be left unset: anything on an assignment, and a definition's
    // optional sections - the rest of a definition is the behaviour itself.
    const optional = schema.type !== "constant" && (scope === "assignment" || schema.type === "object");
    const label = element("div", "behavior-label",
      `${schema.label ?? key}${schema.required ? " *" : ""}`);
    if (schema.description) label.title = schema.description;
    wrapper.append(label);

    if (assigned || !optional) {
      if (assigned && optional) label.append(revertControl(key, inheritedValue));
      wrapper.append(renderValue(schema, draft, key, [key]));
      if (schema.description) wrapper.append(element("div", "behavior-help", schema.description));
      return wrapper;
    }

    wrapper.classList.add("behavior-inheriting");
    const shadow = {};
    if (inheritedValue !== undefined) shadow[key] = clone(inheritedValue);
    const hint = element("div", "behavior-inherited", inheritedValue !== undefined
      ? "From the behaviour."
      : scope === "assignment" ? "Not set, here or on the behaviour." : "Not set.");
    const settle = () => {
      wrapper.classList.remove("behavior-inheriting");
      hint.remove();
      label.append(revertControl(key, inheritedValue));
    };
    // Taking the value over is a change of state as well as of value, but the
    // row can say so itself. A full redraw here would throw away the focus of
    // whoever is still working their way along the form.
    const promote = (structural = false) => {
      if (!own(shadow, key)) {
        delete draft[key];
        changed(true);
        return;
      }
      const first = !own(draft, key);
      draft[key] = shadow[key];
      if (first && !structural) settle();
      changed(structural);
    };
    wrapper.append(renderValue(schema, shadow, key, [key], promote), hint);
    if (schema.description) wrapper.append(element("div", "behavior-help", schema.description));
    return wrapper;
  }

  function revertControl(key, inheritedValue) {
    const button = smallButton("↺", inheritedValue !== undefined
      ? "Use the behaviour's value" : "Leave this unset", () => {
      delete draft[key];
      changed(true);
    });
    button.classList.add("behavior-revert");
    return button;
  }

  function renderValue(schema, parent, key, path, localChanged = changed) {
    if (schema.type === "constant") {
      if (!own(parent, key)) parent[key] = clone(schema.value);
      return element("div", "behavior-constant", describe(schema, schema.value));
    }
    if (schema.type === "object") return renderObject(schema, parent, key, path, localChanged);
    if (schema.type === "array" || schema.type === "entitySource") {
      return renderArray(schema, parent, key, path, localChanged);
    }
    if (schema.type === "record") return renderRecord(schema, parent, key, path, localChanged);
    if (schema.type === "vector3") return renderVector(schema, parent, key, localChanged);
    return renderScalar(schema, parent, key, localChanged);
  }

  function renderObject(schema, parent, key, path, localChanged) {
    const box = element("div", "behavior-object");
    if (!isPlainObject(parent[key])) parent[key] = {};
    // A field another field's list is drawn from has to redraw that list when it
    // changes, or the event picker would keep offering whatever the *previous*
    // source raised. Derived from the schema so the two cannot drift apart.
    const controls = new Set(Object.values(schema.properties ?? {})
      .map((child) => child.optionsFrom).filter(Boolean));
    for (const [childKey, childSchema] of Object.entries(schema.properties ?? {})) {
      const row = element("div", "behavior-nested");
      const childChanged = controls.has(childKey) ? () => localChanged(true) : localChanged;
      row.append(
        element("div", "behavior-label", `${childSchema.label ?? childKey}${childSchema.required ? " *" : ""}`),
        renderValue(childSchema, parent[key], childKey, [...path, childKey], childChanged)
      );
      if (childSchema.description) row.append(element("div", "behavior-help", childSchema.description));
      box.append(row);
    }
    return box;
  }

  /**
   * A list wherever a list is knowable, and a text box only where it is not.
   *
   * Entity names and event names are never typed: both are vocabularies the
   * ship already defines, and a typo in either is a behaviour that silently
   * does nothing at runtime. Where the vocabulary is empty the control stays a
   * disabled picker saying why, rather than quietly turning back into a field
   * that invites the guess this exists to prevent.
   */
  function renderScalar(schema, parent, key, localChanged) {
    if (schema.type === "boolean") {
      const select = document.createElement("select");
      addOption(select, "", schema.default === undefined ? "(not set)" : `(default: ${describe(schema, schema.default)})`);
      addOption(select, "true", "Yes");
      addOption(select, "false", "No");
      select.value = own(parent, key) ? String(parent[key]) : "";
      select.addEventListener("change", () => {
        if (!select.value) delete parent[key];
        else parent[key] = select.value === "true";
        localChanged();
      });
      return select;
    }
    const listed = choicesFor(schema, parent);
    if (PICKED_TYPES.has(schema.type) || (schema.type === "string" && listed.length)) {
      return renderPicker(schema, parent, key, localChanged, listed);
    }
    const input = document.createElement("input");
    input.type = schema.type === "number" ? "number" : "text";
    if (schema.type === "number") {
      input.step = "any";
      if (schema.minimum !== undefined) input.min = String(schema.minimum);
      if (schema.maximum !== undefined) input.max = String(schema.maximum);
    }
    input.value = own(parent, key) ? String(parent[key]) : "";
    if (!input.value && schema.default !== undefined) input.placeholder = `default: ${describe(schema, schema.default)}`;
    input.addEventListener("change", () => {
      const text = input.value.trim();
      if (!text) delete parent[key];
      else parent[key] = schema.type === "number" ? Number(text) : text;
      localChanged();
    });
    return input;
  }

  function renderPicker(schema, parent, key, localChanged, listed) {
    const box = element("div", "behavior-picker");
    const current = own(parent, key) ? String(parent[key]) : "";
    const select = document.createElement("select");
    if (!schema.required || !current) {
      addOption(select, "", schema.type === "enum" ? (schema.defaultLabel ?? "(not set)")
        : schema.default !== undefined ? `(default: ${describe(schema, schema.default)})`
          : "(not set)");
    }
    for (const value of listed) addOption(select, String(value), String(value));
    // Never drop a value just because the current list cannot account for it -
    // a narrowed room filter, a source that stopped raising it, a name the game
    // knows and the ship does not. Shown, flagged, and kept until it is changed.
    if (current && !listed.some((value) => String(value) === current)) {
      addOption(select, current, `${current} — not in this list`);
    }
    if (schema.allowNew) addOption(select, NEW_VALUE, "＋ name a new one…");
    select.value = current;
    select.disabled = !listed.length && !current && !schema.allowNew;
    select.addEventListener("change", () => {
      if (select.value === NEW_VALUE) { askForNewName(); return; }
      if (!select.value) delete parent[key];
      else parent[key] = select.value;
      localChanged();
    });
    box.append(select);
    if (select.disabled && schema.emptyHint) box.append(element("div", "behavior-help", schema.emptyHint));
    return box;

    function askForNewName() {
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = `new ${schema.label ?? schema.type}`;
      const commit = () => {
        const text = input.value.trim();
        if (text) parent[key] = text;
        localChanged(true);
      };
      input.addEventListener("change", commit);
      const row = element("div", "behavior-picker-new");
      row.append(input, smallButton("×", "Keep what was there", () => localChanged(true)));
      box.replaceChildren(row);
      input.focus();
    }
  }

  function renderVector(schema, parent, key, localChanged) {
    const box = element("div", "behavior-vector");
    const current = Array.isArray(parent[key]) ? parent[key] : [];
    const inputs = [0, 1, 2].map((axis) => {
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.placeholder = ["X", "Y", "Z"][axis];
      input.value = Number.isFinite(current[axis]) ? String(current[axis]) : "";
      return input;
    });
    // A vector is written whole but edited an axis at a time, and an axis left
    // blank is the zero it looks like: typing -1 into X of an empty field means
    // (-1, 0, 0), not "still nothing". Emptying all three unsets it again.
    const commit = () => {
      const values = inputs.map((input) => input.value.trim());
      if (values.every((item) => !item)) delete parent[key];
      else if (values.every((item) => item === "" || Number.isFinite(Number(item)))) {
        parent[key] = values.map((item) => (item === "" ? 0 : Number(item)));
      }
      localChanged();
    };
    inputs.forEach((input) => input.addEventListener("change", commit));
    box.append(...inputs);
    return box;
  }

  function renderArray(schema, parent, key, path, localChanged) {
    const box = element("div", "behavior-array");
    const source = schema.type === "entitySource"
      ? (Array.isArray(parent[key]) ? parent[key] : parent[key] ? [parent[key]] : [])
      : (Array.isArray(parent[key]) ? parent[key] : []);
    const values = clone(source);
    const itemSchema = schema.type === "entitySource"
      ? { type: "entity", optionsSource: schema.optionsSource }
      : schema.items;
    values.forEach((item, index) => {
      const row = element("div", "behavior-array-row");
      const holder = { value: item };
      const sync = () => {
        values[index] = holder.value;
        parent[key] = schema.type === "entitySource"
          ? (values.length === 1 ? values[0] : values)
          : values;
      };
      const itemChanged = (structural = false) => {
        sync();
        localChanged(structural);
      };
      row.append(renderValue(itemSchema, holder, "value", [...path, String(index)], itemChanged));
      row.append(
        smallButton("↑", "Move up", () => move(values, index, index - 1, parent, key, schema)),
        smallButton("↓", "Move down", () => move(values, index, index + 1, parent, key, schema)),
        smallButton("×", "Remove", () => {
          values.splice(index, 1);
          setArrayValue(parent, key, schema, values);
          localChanged(true);
        })
      );
      box.append(row);
    });
    const add = smallButton("+ Add", `Add ${schema.label ?? "item"}`, () => {
      values.push(initialItemValue(itemSchema, choicesFor(itemSchema, parent)));
      setArrayValue(parent, key, schema, values);
      localChanged(true);
    });
    add.classList.add("behavior-array-add");
    box.append(add);
    return box;

    function move(items, from, to, target, targetKey, targetSchema) {
      if (to < 0 || to >= items.length) return;
      [items[from], items[to]] = [items[to], items[from]];
      setArrayValue(target, targetKey, targetSchema, items);
      localChanged(true);
    }
  }

  function renderRecord(schema, parent, key, path, localChanged) {
    const box = element("div", "behavior-record");
    const record = isPlainObject(parent[key]) ? parent[key] : {};
    for (const [recordKey, recordValue] of Object.entries(record)) {
      const row = element("div", "behavior-record-row");
      const name = document.createElement("input");
      name.type = "text";
      name.value = recordKey;
      name.placeholder = schema.keyLabel ?? "Category";
      const holder = { value: recordValue };
      const valueChanged = (structural = false) => {
        record[recordKey] = holder.value;
        parent[key] = record;
        localChanged(structural);
      };
      row.append(name, renderValue(schema.values, holder, "value", [...path, recordKey], valueChanged));
      name.addEventListener("change", () => {
        const next = name.value.trim();
        if (!next || next === recordKey || own(record, next)) return;
        delete record[recordKey];
        record[next] = holder.value;
        parent[key] = record;
        localChanged(true);
      });
      row.append(smallButton("×", "Remove category", () => {
        delete record[recordKey];
        if (Object.keys(record).length) parent[key] = record;
        else delete parent[key];
        localChanged(true);
      }));
      box.append(row);
    }
    const add = smallButton("+ Add category", "Add category", () => {
      let index = 1;
      while (own(record, `category${index}`)) index++;
      record[`category${index}`] = initialValue(schema.values);
      parent[key] = record;
      localChanged(true);
    });
    add.classList.add("behavior-array-add");
    box.append(add);
    return box;
  }

  render();
  return api;
}

function setArrayValue(parent, key, schema, values) {
  if (!values.length) delete parent[key];
  else parent[key] = schema.type === "entitySource" && values.length === 1 ? values[0] : values;
}

function initialValue(schema, inherited) {
  if (inherited !== undefined) return clone(inherited);
  if (schema?.type === "constant") return clone(schema.value);
  if (schema?.default !== undefined) return clone(schema.default);
  if (schema?.type === "object") {
    const out = {};
    for (const [key, child] of Object.entries(schema.properties ?? {})) {
      if (child.type === "constant" || child.required && child.default !== undefined) {
        out[key] = initialValue(child);
      }
    }
    return out;
  }
  if (schema?.type === "array" || schema?.type === "entitySource") return [];
  if (schema?.type === "record") return {};
  if (schema?.type === "boolean") return false;
  if (schema?.type === "number") return schema.minimum ?? 0;
  if (schema?.type === "vector3") return [0, 0, 0];
  return "";
}

function initialItemValue(schema, choices) {
  return choices?.length ? clone(choices[0]) : initialValue(schema);
}

function validateObject(properties, value, prefix, partial) {
  const errors = [];
  for (const key of Object.keys(value)) {
    if (properties[key]) continue;
    const path = prefix ? `${prefix}.${key}` : key;
    errors.push(`${path} is not supported.`);
  }
  for (const [key, schema] of Object.entries(properties)) {
    const path = prefix ? `${prefix}.${key}` : key;
    const present = own(value, key);
    if (!present) {
      if (schema.required && !partial) errors.push(`${path} is required.`);
      continue;
    }
    errors.push(...validateValue(schema, value[key], path, partial));
  }
  return errors;
}

function validateValue(schema, value, path, partial) {
  if (schema.type === "constant") {
    return JSON.stringify(value) === JSON.stringify(schema.value) ? [] : [`${path} must be ${describe(schema, schema.value)}.`];
  }
  if (schema.type === "number") {
    if (!Number.isFinite(value)) return [`${path} must be a number.`];
    if (schema.minimum !== undefined && (schema.exclusiveMinimum ? value <= schema.minimum : value < schema.minimum)) {
      return [`${path} must be ${schema.exclusiveMinimum ? "greater than" : "at least"} ${schema.minimum}.`];
    }
    if (schema.maximum !== undefined && value > schema.maximum) return [`${path} must be at most ${schema.maximum}.`];
    return [];
  }
  if (schema.type === "boolean") return typeof value === "boolean" ? [] : [`${path} must be true or false.`];
  if (schema.type === "enum") return schema.values?.includes(value) ? [] : [`${path} has an unsupported value.`];
  if (["string", "entity", "event"].includes(schema.type)) {
    return typeof value === "string" && value.trim() ? [] : [`${path} must not be empty.`];
  }
  if (schema.type === "vector3") {
    if (!Array.isArray(value) || value.length !== 3 || value.some((item) => !Number.isFinite(item))) {
      return [`${path} must contain three numbers.`];
    }
    if (schema.minimum !== undefined && value.some((item) => item < schema.minimum)) {
      return [`${path} values must be at least ${schema.minimum}.`];
    }
    return [];
  }
  if (schema.type === "object") {
    return isPlainObject(value)
      ? validateObject(schema.properties ?? {}, value, path, false)
      : [`${path} must be an object.`];
  }
  if (schema.type === "record") {
    if (!isPlainObject(value)) return [`${path} must be a map.`];
    return Object.entries(value).flatMap(([key, item]) =>
      key.trim() ? validateValue(schema.values, item, `${path}.${key}`, partial) : [`${path} contains an empty key.`]);
  }
  if (schema.type === "entitySource") {
    const values = Array.isArray(value) ? value : [value];
    if (!values.length || values.some((item) => typeof item !== "string" || !item.trim())) {
      return [`${path} must contain at least one entity.`];
    }
    return [];
  }
  if (schema.type === "array") {
    if (!Array.isArray(value)) return [`${path} must be an array.`];
    if (schema.minItems !== undefined && value.length < schema.minItems) {
      return [`${path} must contain at least ${schema.minItems} item${schema.minItems === 1 ? "" : "s"}.`];
    }
    return value.flatMap((item, index) => validateValue(schema.items, item, `${path}[${index}]`, partial));
  }
  return [];
}

/**
 * What a picker may offer, given what its siblings currently say.
 *
 * A plain array is a fixed vocabulary. A function is one that depends on the
 * rest of the entry - `optionsFrom` names the sibling property it is computed
 * from, which is how an event list narrows to what its chosen sources raise.
 */
function optionValues(schema, options, siblings) {
  const source = schema?.optionsSource ? options?.[schema.optionsSource] : undefined;
  const resolved = typeof source === "function"
    ? source(schema.optionsFrom ? siblings?.[schema.optionsFrom] : undefined)
    : source;
  const list = resolved ?? schema?.values ?? [];
  return Array.isArray(list) ? list : [];
}

/** Walk a schema tree - object properties, array items, record values alike. */
function anySchema(properties, predicate) {
  for (const schema of Object.values(properties ?? {})) {
    if (predicate(schema)) return true;
    const children = schema.type === "object" ? schema.properties
      : schema.type === "record" ? { value: schema.values }
        : schema.items ? { item: schema.items } : null;
    if (children && anySchema(children, predicate)) return true;
  }
  return false;
}

/** Every entity name the value currently holds, wherever the schema puts one. */
function entityValues(properties, value, out = []) {
  for (const [key, schema] of Object.entries(properties ?? {})) {
    if (!own(value, key)) continue;
    collect(schema, value[key]);
  }
  return out;

  function collect(schema, item) {
    if (item === undefined || item === null) return;
    if (schema.type === "entity" || schema.type === "entitySource") {
      for (const name of Array.isArray(item) ? item : [item]) {
        if (typeof name === "string" && name.trim()) out.push(name.trim());
      }
      return;
    }
    if (schema.type === "object") entityValues(schema.properties ?? {}, item, out);
    else if (schema.type === "record" && isPlainObject(item)) {
      for (const child of Object.values(item)) collect(schema.values, child);
    } else if (schema.type === "array" && Array.isArray(item)) {
      for (const child of item) collect(schema.items ?? {}, child);
    }
  }
}

/**
 * Say a value the way a person would read it, never the way a file stores it.
 *
 * Inherited values, constants and defaults are all shown rather than edited,
 * and `JSON.stringify` of a sound map is a line of punctuation nobody should
 * have to parse to find out which MP3 a splash plays.
 */
function describe(schema, value) {
  if (value === undefined || value === null) return "not set";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    if (schema?.type === "vector3") return `X ${value[0]} · Y ${value[1]} · Z ${value[2]}`;
    if (!value.length) return "empty";
    return value.map((item) => describe(schema?.items, item)).join(", ");
  }
  if (isPlainObject(value)) {
    const entries = Object.entries(value);
    if (!entries.length) return "empty";
    return entries
      .map(([key, item]) => schema?.type === "record"
        ? `${key}: ${describe(schema.values, item)}`
        : `${schema?.properties?.[key]?.label ?? key}: ${describe(schema?.properties?.[key], item)}`)
      .join(" · ");
  }
  return String(value);
}

function addOption(select, value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.append(option);
}

function smallButton(text, title, action) {
  const button = element("button", "behavior-small", text);
  button.type = "button";
  button.title = title;
  button.addEventListener("click", action);
  return button;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function isPlainObject(value) {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function valueAt(value, path) {
  let current = value;
  for (const key of path) current = current?.[key];
  return current;
}
