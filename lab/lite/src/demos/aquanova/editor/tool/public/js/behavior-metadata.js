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
    return { version: raw.version, behaviors: resolved };
  });
  return catalogPromise;
}

export function behaviorMetadata(catalog, name) {
  return catalog?.behaviors?.[name] ?? null;
}

export function behaviorMetadataNames(catalog) {
  return Object.keys(catalog?.behaviors ?? {}).sort((a, b) => a.localeCompare(b));
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
  const inspect = (behaviorName, config) => {
    const metadata = behaviorMetadata(catalog, behaviorName);
    for (const event of metadata?.eventsRaised ?? []) {
      if (event.name) names.add(event.name);
      const value = event.property ? valueAt(config, event.property.split(".")) : undefined;
      if (typeof value === "string" && value.trim()) names.add(value.trim());
    }
  };
  for (const [name, definition] of definitions) inspect(name, definition);
  for (const list of entities.values()) {
    for (const assignment of list) {
      inspect(assignment.name, { ...(definitions.get(assignment.name) ?? {}), ...assignment });
    }
  }
  return [...names].sort((a, b) => a.localeCompare(b));
}

export function createBehaviorForm(host, {
  metadata,
  value = {},
  inherited = {},
  scope = "definition",
  options = {},
  onChange = null,
}) {
  let draft = clone(value) ?? {};
  let validationHost = null;

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

  function render() {
    host.replaceChildren();
    host.className = "behavior-form";
    if (!metadata) {
      const warning = element("div", "behavior-legacy",
        "No metadata describes this behavior. Its existing data will be preserved.");
      host.append(warning);
      const paths = unknownPaths({}, draft);
      if (paths.length) host.append(element("div", "behavior-unknown", `Preserved fields: ${paths.join(", ")}`));
      return;
    }
    if (metadata.description) host.append(element("p", "behavior-description", metadata.description));
    const unknown = unknownPaths(metadata.properties ?? {}, draft);
    if (unknown.length) {
      host.append(element("div", "behavior-unknown",
        `Legacy fields are preserved but not editable: ${unknown.join(", ")}`));
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

  function renderTopProperty(key, schema) {
    const wrapper = element("div", "behavior-property");
    wrapper.dataset.behaviorKey = key;
    const assigned = own(draft, key);
    if (scope === "assignment" && schema.type !== "constant") {
      const override = element("label", "behavior-override");
      const check = document.createElement("input");
      check.type = "checkbox";
      check.checked = assigned;
      const inheritedValue = inherited?.[key];
      override.append(check, document.createTextNode(
        ` Override ${schema.label ?? key}${schema.required ? " *" : ""}`));
      if (!assigned && inheritedValue !== undefined) {
        override.title = `Inherited: ${compact(inheritedValue)}`;
      }
      check.addEventListener("change", () => {
        if (check.checked) draft[key] = initialValue(schema, inheritedValue);
        else delete draft[key];
        changed(true);
      });
      wrapper.append(override);
      if (!assigned) {
        wrapper.append(element("div", "behavior-inherited",
          inheritedValue === undefined ? "Not set" : `Inherited: ${compact(inheritedValue)}`));
        return wrapper;
      }
    }
    if (scope === "definition" && schema.type === "object" && !assigned) {
      const configure = element("label", "behavior-override");
      const check = document.createElement("input");
      check.type = "checkbox";
      configure.append(check, document.createTextNode(
        ` Configure ${schema.label ?? key}${schema.required ? " *" : ""}`));
      check.addEventListener("change", () => {
        if (check.checked) draft[key] = initialValue(schema);
        else delete draft[key];
        changed(true);
      });
      wrapper.append(configure);
      return wrapper;
    }
    const label = element("div", "behavior-label", `${schema.label ?? key}${schema.required ? " *" : ""}`);
    if (schema.description) label.title = schema.description;
    wrapper.append(label, renderValue(schema, draft, key, [key]));
    if (schema.description) wrapper.append(element("div", "behavior-help", schema.description));
    return wrapper;
  }

  function renderValue(schema, parent, key, path, localChanged = changed) {
    if (schema.type === "constant") {
      if (!own(parent, key)) parent[key] = clone(schema.value);
      return element("div", "behavior-constant", compact(schema.value));
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
    for (const [childKey, childSchema] of Object.entries(schema.properties ?? {})) {
      const row = element("div", "behavior-nested");
      row.append(
        element("div", "behavior-label", `${childSchema.label ?? childKey}${childSchema.required ? " *" : ""}`),
        renderValue(childSchema, parent[key], childKey, [...path, childKey], localChanged)
      );
      if (childSchema.description) row.append(element("div", "behavior-help", childSchema.description));
      box.append(row);
    }
    return box;
  }

  function renderScalar(schema, parent, key, localChanged) {
    if (schema.type === "boolean") {
      const select = document.createElement("select");
      addOption(select, "", schema.default === undefined ? "(not set)" : `(default: ${schema.default})`);
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
    if (schema.type === "enum") {
      const select = document.createElement("select");
      addOption(select, "", schema.defaultLabel ?? "(not set)");
      for (const value of schema.values ?? []) addOption(select, String(value), String(value));
      select.value = own(parent, key) ? String(parent[key]) : "";
      select.addEventListener("change", () => {
        if (!select.value) delete parent[key];
        else parent[key] = select.value;
        localChanged();
      });
      return select;
    }
    const input = document.createElement("input");
    input.type = schema.type === "number" ? "number" : "text";
    if (schema.type === "number") {
      input.step = "any";
      if (schema.minimum !== undefined) input.min = String(schema.minimum);
      if (schema.maximum !== undefined) input.max = String(schema.maximum);
    } else {
      attachDatalist(input, optionValues(schema, options), `bhv-${schema.type}`);
    }
    input.value = own(parent, key) ? String(parent[key]) : "";
    if (!input.value && schema.default !== undefined) input.placeholder = `default: ${compact(schema.default)}`;
    input.addEventListener("change", () => {
      const text = input.value.trim();
      if (!text) delete parent[key];
      else parent[key] = schema.type === "number" ? Number(text) : text;
      localChanged();
    });
    return input;
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
    const commit = () => {
      const values = inputs.map((input) => input.value.trim());
      if (values.every((item) => !item)) delete parent[key];
      else if (values.every((item) => item !== "" && Number.isFinite(Number(item)))) {
        parent[key] = values.map(Number);
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
      values.push(initialItemValue(itemSchema, options));
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
      name.placeholder = "Category";
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

function initialItemValue(schema, options) {
  const choices = optionValues(schema, options);
  return choices.length ? clone(choices[0]) : initialValue(schema);
}

function validateObject(properties, value, prefix, partial) {
  const errors = [];
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
    return JSON.stringify(value) === JSON.stringify(schema.value) ? [] : [`${path} must be ${compact(schema.value)}.`];
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

function unknownPaths(properties, value, prefix = "") {
  if (!isPlainObject(value)) return [];
  const out = [];
  for (const [key, item] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${key}` : key;
    const schema = properties[key];
    if (!schema) {
      out.push(path);
    } else if (schema.type === "object" && isPlainObject(item)) {
      out.push(...unknownPaths(schema.properties ?? {}, item, path));
    }
  }
  return out;
}

function optionValues(schema, options) {
  return options[schema.optionsSource] ?? schema.values ?? [];
}

let datalistId = 0;
const datalists = new Map();
function attachDatalist(input, values, prefix) {
  if (!values?.length) return;
  const cacheKey = `${prefix}:${JSON.stringify(values)}`;
  let id = datalists.get(cacheKey);
  if (id) {
    input.setAttribute("list", id);
    return;
  }
  const list = document.createElement("datalist");
  id = `${prefix}-${++datalistId}`;
  list.id = id;
  for (const value of values) addOption(list, String(value), String(value));
  input.setAttribute("list", list.id);
  document.body.append(list);
  datalists.set(cacheKey, id);
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

function compact(value) {
  return typeof value === "string" ? value : JSON.stringify(value);
}
