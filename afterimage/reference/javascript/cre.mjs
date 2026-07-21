#!/usr/bin/env node
// Independent JavaScript reference implementation of CRE 0.1.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const I64_MIN = -(1n << 63n);
const I64_MAX = (1n << 63n) - 1n;
const ID_PREFIX = "sha256:";
const DEFAULT_LIMITS = {
  max_base_events: 100000n,
  max_derived_events: 1000000n,
  max_bindings_tested: 10000000n,
  max_value_bytes: 1048576n,
  max_projection_records: 1000000n,
};

class CREError extends Error {
  constructor(code, message, context = {}) {
    super(message);
    this.code = code;
    this.context = context;
  }
  asValue() {
    return { code: this.code, message: this.message, context: this.context };
  }
}

function fail(code, message, context = {}) {
  throw new CREError(code, message, context);
}

function mapValue(entries = []) {
  const result = Object.create(null);
  for (const [key, value] of entries) result[key] = value;
  return result;
}

function isMap(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value) && !(value instanceof Uint8Array);
}

function checkedI64(value, where = "integer") {
  if (typeof value !== "bigint") fail("type_error", `${where} must be Int`);
  if (value < I64_MIN || value > I64_MAX) fail("integer_overflow", `${where} is outside signed 64-bit range`);
  return value;
}

function normalizeText(value) {
  if (typeof value !== "string") fail("type_error", "text value must be Text");
  for (const char of value) {
    const point = char.codePointAt(0);
    if (point >= 0xd800 && point <= 0xdfff) fail("invalid_text", "Text must contain Unicode scalar values");
  }
  return value.normalize("NFC");
}

function decodeBytes(encoded) {
  if (typeof encoded !== "string" || encoded.includes("=") || !/^[A-Za-z0-9_-]*$/.test(encoded)) {
    fail("invalid_bytes", "base64url bytes must omit padding");
  }
  const raw = new Uint8Array(Buffer.from(encoded, "base64url"));
  if (Buffer.from(raw).toString("base64url") !== encoded) fail("invalid_bytes", "non-canonical base64url bytes");
  return raw;
}

function normalizeValue(value) {
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "bigint") return checkedI64(value);
  if (typeof value === "number") fail("type_error", "floating-point values are forbidden");
  if (typeof value === "string") return normalizeText(value);
  if (value instanceof Uint8Array) return new Uint8Array(value);
  if (Array.isArray(value)) return value.map(normalizeValue);
  if (isMap(value)) {
    const keys = Object.keys(value);
    if (keys.includes("$bytes")) {
      if (keys.length !== 1) fail("invalid_bytes", "$bytes is reserved for a sole text field");
      return decodeBytes(value.$bytes);
    }
    const result = mapValue();
    const originals = mapValue();
    for (const rawKey of keys) {
      const key = normalizeText(rawKey);
      if (Object.hasOwn(result, key)) {
        fail("duplicate_key", "map keys collide after NFC normalization", { first: originals[key], second: rawKey });
      }
      originals[key] = rawKey;
      result[key] = normalizeValue(value[rawKey]);
    }
    return result;
  }
  fail("type_error", `unsupported value type: ${typeof value}`);
}

function cloneValue(value) {
  if (value instanceof Uint8Array) return new Uint8Array(value);
  if (Array.isArray(value)) return value.map(cloneValue);
  if (isMap(value)) return mapValue(Object.keys(value).map((key) => [key, cloneValue(value[key])]));
  return value;
}

function escapeText(value) {
  let output = '"';
  const short = new Map([[8, "\\b"], [9, "\\t"], [10, "\\n"], [12, "\\f"], [13, "\\r"]]);
  for (const char of normalizeText(value)) {
    const point = char.codePointAt(0);
    if (char === '"') output += '\\"';
    else if (char === "\\") output += "\\\\";
    else if (short.has(point)) output += short.get(point);
    else if (point < 0x20) output += `\\u${point.toString(16).padStart(4, "0")}`;
    else output += char;
  }
  return output + '"';
}

function canonicalText(raw) {
  const value = normalizeValue(raw);
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "string") return escapeText(value);
  if (value instanceof Uint8Array) return `{"$bytes":${escapeText(Buffer.from(value).toString("base64url"))}}`;
  if (Array.isArray(value)) return `[${value.map(canonicalText).join(",")}]`;
  if (isMap(value)) {
    const keys = Object.keys(value).sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)));
    return `{${keys.map((key) => `${escapeText(key)}:${canonicalText(value[key])}`).join(",")}}`;
  }
  fail("type_error", "unreachable canonical value type");
}

const canonicalBytes = (value) => Buffer.from(canonicalText(value), "utf8");

function uleb128(value) {
  let n = BigInt(value);
  if (n < 0n) fail("invalid_length", "ULEB128 value must be non-negative");
  const output = [];
  do {
    let byte = Number(n & 0x7fn);
    n >>= 7n;
    if (n) byte |= 0x80;
    output.push(byte);
  } while (n);
  return Buffer.from(output);
}

function digestRaw(domain, ...parts) {
  const state = createHash("sha256");
  state.update(Buffer.from(normalizeText(domain), "utf8"));
  state.update(Buffer.from([0]));
  for (const partValue of parts) {
    const part = Buffer.from(partValue);
    state.update(uleb128(part.length));
    state.update(part);
  }
  return new Uint8Array(state.digest());
}

function digestId(domain, ...parts) {
  return ID_PREFIX + Buffer.from(digestRaw(domain, ...parts)).toString("hex");
}

function parseId(value, where = "ID") {
  if (typeof value !== "string" || !/^sha256:[0-9a-f]{64}$/.test(value)) {
    fail("invalid_id", `${where} must be sha256:<64 lowercase hex>`);
  }
  return Buffer.from(value.slice(ID_PREFIX.length), "hex");
}

const sameValue = (left, right) => Buffer.compare(canonicalBytes(left), canonicalBytes(right)) === 0;

function compareValues(left, right) {
  if (typeof left === "bigint" && typeof right === "bigint") return left < right ? -1 : left > right ? 1 : 0;
  if (typeof left === "string" && typeof right === "string") return Buffer.compare(Buffer.from(left), Buffer.from(right));
  if (left instanceof Uint8Array && right instanceof Uint8Array) return Buffer.compare(Buffer.from(left), Buffer.from(right));
  fail("type_error", "ordered values must have matching Int, Text, or Bytes type");
}

function pointerTokens(pointer) {
  if (typeof pointer !== "string") fail("invalid_pointer", "JSON Pointer must be Text");
  if (pointer === "") return [];
  if (!pointer.startsWith("/")) fail("invalid_pointer", "JSON Pointer must be empty or begin with /");
  return pointer.slice(1).split("/").map((raw) => {
    let value = "";
    for (let index = 0; index < raw.length; index++) {
      if (raw[index] !== "~") value += raw[index];
      else {
        const next = raw[++index];
        if (next !== "0" && next !== "1") fail("invalid_pointer", "invalid JSON Pointer escape");
        value += next === "0" ? "~" : "/";
      }
    }
    return value;
  });
}

function pointerGet(value, pointer) {
  let current = value;
  for (const token of pointerTokens(pointer)) {
    if (isMap(current)) {
      if (!Object.hasOwn(current, token)) fail("missing_path", "JSON Pointer map key is missing", { token });
      current = current[token];
    } else if (Array.isArray(current)) {
      if (!/^(0|[1-9][0-9]*)$/.test(token)) fail("invalid_pointer", "list index must be canonical decimal");
      const index = Number(token);
      if (index >= current.length) fail("missing_path", "JSON Pointer list index is out of range", { index: BigInt(index) });
      current = current[index];
    } else fail("missing_path", "JSON Pointer traverses a scalar", { token });
  }
  return current;
}

function pointerReplace(value, pointer, replacement) {
  const tokens = pointerTokens(pointer);
  if (!tokens.length) return normalizeValue(replacement);
  const result = cloneValue(value);
  let current = result;
  for (const token of tokens.slice(0, -1)) {
    if (isMap(current) && Object.hasOwn(current, token)) current = current[token];
    else if (Array.isArray(current) && /^(0|[1-9][0-9]*)$/.test(token) && Number(token) < current.length) current = current[Number(token)];
    else fail("missing_path", "replace pointer path is missing", { token });
  }
  const final = tokens.at(-1);
  if (isMap(current) && Object.hasOwn(current, final)) current[final] = normalizeValue(replacement);
  else if (Array.isArray(current) && /^(0|[1-9][0-9]*)$/.test(final) && Number(final) < current.length) current[Number(final)] = normalizeValue(replacement);
  else fail("missing_path", "replace pointer target is missing", { token: final });
  return result;
}

const eventView = (event) => mapValue(["id", "topic", "at", "payload", "parents", "origin"].map((key) => [key, event[key]]));
const eventId = (body) => digestId("afterimage/event/1", canonicalBytes(body));

function validateOrigin(raw) {
  const origin = normalizeValue(raw);
  if (!isMap(origin) || typeof origin.kind !== "string") fail("invalid_origin", "origin must contain a kind");
  const keys = Object.keys(origin).sort().join(",");
  if (origin.kind === "base") {
    if (keys !== "kind,sequence,source" || typeof origin.source !== "string") fail("invalid_origin", "base origin requires kind, source, and sequence");
    if (checkedI64(origin.sequence, "origin.sequence") < 0n) fail("invalid_origin", "origin.sequence must be non-negative");
  } else if (origin.kind === "derived") {
    if (keys !== "binding,kind,ordinal,rule") fail("invalid_origin", "derived origin has wrong fields");
    if (typeof origin.rule !== "string" || !(origin.binding instanceof Uint8Array)) fail("invalid_origin", "derived rule and binding have wrong type");
    if (checkedI64(origin.ordinal, "origin.ordinal") < 0n) fail("invalid_origin", "origin.ordinal must be non-negative");
  } else if (origin.kind === "player") {
    if (keys !== "branch_parent,kind,ordinal,supersedes") fail("invalid_origin", "player origin has wrong fields");
    parseId(origin.branch_parent, "origin.branch_parent");
    if (checkedI64(origin.ordinal, "origin.ordinal") < 0n) fail("invalid_origin", "origin.ordinal must be non-negative");
    if (origin.supersedes !== null) parseId(origin.supersedes, "origin.supersedes");
  } else fail("invalid_origin", `unknown origin kind: ${origin.kind}`);
  return origin;
}

function makeEvent(raw, claimedId = null) {
  const body = normalizeValue(raw);
  if (!isMap(body) || Object.keys(body).sort().join(",") !== "at,origin,parents,payload,topic") fail("invalid_event", "event body has wrong fields");
  if (typeof body.topic !== "string" || Buffer.byteLength(body.topic) < 1 || Buffer.byteLength(body.topic) > 128) fail("invalid_event", "event topic must be 1..128 UTF-8 bytes");
  checkedI64(body.at, "event.at");
  if (!Array.isArray(body.parents)) fail("invalid_event", "event.parents must be a list");
  const rawParents = body.parents.map((parent) => parseId(parent, "event parent"));
  const sorted = [...rawParents].sort(Buffer.compare);
  if (rawParents.some((parent, index) => Buffer.compare(parent, sorted[index])) || new Set(body.parents).size !== body.parents.length) fail("invalid_event", "event parents must be unique and raw-digest sorted");
  body.origin = validateOrigin(body.origin);
  const identifier = eventId(body);
  if (claimedId !== null && claimedId !== identifier) fail("event_id_mismatch", "claimed event ID does not match body", { claimed: claimedId, actual: identifier });
  return mapValue([["id", identifier], ...Object.keys(body).map((key) => [key, body[key]])]);
}

function loadEvent(raw) {
  const value = normalizeValue(raw);
  if (isMap(value) && Object.hasOwn(value, "id")) {
    return makeEvent(mapValue(Object.keys(value).filter((key) => key !== "id").map((key) => [key, value[key]])), value.id);
  }
  return makeEvent(value);
}

class Counters {
  constructor(supplied = null) {
    this.limits = { ...DEFAULT_LIMITS };
    for (const [key, value] of Object.entries(supplied || {})) {
      if (!Object.hasOwn(this.limits, key) || typeof value !== "bigint" || value < 0n) fail("invalid_limit", `invalid resource limit: ${key}`);
      this.limits[key] = value;
    }
    this.base_events = 0n;
    this.derived_events = 0n;
    this.bindings_tested = 0n;
    this.projection_records = 0n;
  }
  charge(name, amount = 1n) {
    this[name] += BigInt(amount);
    const limit = `max_${name}`;
    if (this[name] > this.limits[limit]) fail("resource_exhausted", `crossed ${limit}`, { counter: name, value: this[name] });
  }
  asValue() {
    return mapValue(["base_events", "derived_events", "bindings_tested", "projection_records"].map((key) => [key, this[key]]));
  }
}

function validateBaseEvents(values, counters) {
  counters.charge("base_events", values.length);
  const events = new Map(values.map((event) => [event.id, event]));
  if (events.size !== values.length) fail("duplicate_event", "duplicate base event ID");
  for (const event of values) {
    if (event.origin.kind === "derived") fail("invalid_base", "derived event supplied as base");
    if (BigInt(canonicalBytes(event.payload).length) > counters.limits.max_value_bytes) fail("resource_exhausted", "event payload exceeds max_value_bytes");
    for (const parent of event.parents) {
      if (!events.has(parent)) fail("missing_parent", "base event parent is inactive", { event: event.id, parent });
      if (event.origin.kind === "base" && events.get(parent).origin.kind !== "base") fail("invalid_parent", "base event may cite only a base parent");
      if (event.at < events.get(parent).at) fail("invalid_time", "event time precedes parent", { event: event.id, parent });
    }
  }
  const indegree = new Map(values.map((event) => [event.id, event.parents.length]));
  const children = new Map(values.map((event) => [event.id, []]));
  for (const event of values) for (const parent of event.parents) children.get(parent).push(event.id);
  const orderKey = (identifier) => [events.get(identifier).at, parseId(identifier)];
  const sortReady = (a, b) => orderKey(a)[0] < orderKey(b)[0] ? -1 : orderKey(a)[0] > orderKey(b)[0] ? 1 : Buffer.compare(orderKey(a)[1], orderKey(b)[1]);
  const ready = [...indegree].filter(([, degree]) => degree === 0).map(([id]) => id).sort(sortReady);
  const ordered = [];
  while (ready.length) {
    const identifier = ready.shift();
    ordered.push(events.get(identifier));
    for (const child of children.get(identifier)) {
      indegree.set(child, indegree.get(child) - 1);
      if (indegree.get(child) === 0) { ready.push(child); ready.sort(sortReady); }
    }
  }
  if (ordered.length !== values.length) fail("causal_cycle", "base event parent graph contains a cycle");
  return ordered;
}

function requireBool(value, where) {
  if (typeof value !== "boolean") fail("type_error", `${where} must evaluate to Bool`);
  return value;
}

function evalExpr(expr, env, aggregates) {
  if (!Array.isArray(expr) || !expr.length || typeof expr[0] !== "string") fail("invalid_expression", "expression must be a non-empty opcode list");
  const [op, ...args] = expr;
  const arity = (count) => { if (args.length !== count) fail("invalid_expression", `${op} expects ${count} arguments`); };
  if (op === "const") { arity(1); return normalizeValue(args[0]); }
  if (op === "get") { arity(2); if (typeof args[0] !== "string" || !Object.hasOwn(env, args[0])) fail("unknown_alias", `unknown event alias: ${args[0]}`); return pointerGet(eventView(env[args[0]]), args[1]); }
  if (op === "list") return args.map((arg) => evalExpr(arg, env, aggregates));
  if (op === "map") {
    if (args.length % 2) fail("invalid_expression", "map expects key/expression pairs");
    const result = mapValue();
    for (let index = 0; index < args.length; index += 2) {
      if (typeof args[index] !== "string") fail("invalid_expression", "map key must be Text");
      const key = normalizeText(args[index]);
      if (Object.hasOwn(result, key)) fail("duplicate_key", "duplicate map key in expression", { key });
      result[key] = evalExpr(args[index + 1], env, aggregates);
    }
    return result;
  }
  if (op === "eq" || op === "ne") { arity(2); const equal = sameValue(evalExpr(args[0], env, aggregates), evalExpr(args[1], env, aggregates)); return op === "eq" ? equal : !equal; }
  if (["lt", "le", "gt", "ge"].includes(op)) {
    arity(2); const c = compareValues(evalExpr(args[0], env, aggregates), evalExpr(args[1], env, aggregates));
    return op === "lt" ? c < 0 : op === "le" ? c <= 0 : op === "gt" ? c > 0 : c >= 0;
  }
  if (op === "and") { for (const arg of args) if (!requireBool(evalExpr(arg, env, aggregates), "and operand")) return false; return true; }
  if (op === "or") { for (const arg of args) if (requireBool(evalExpr(arg, env, aggregates), "or operand")) return true; return false; }
  if (op === "not") { arity(1); return !requireBool(evalExpr(args[0], env, aggregates), "not operand"); }
  if (["add", "sub", "mul", "div", "mod"].includes(op)) {
    arity(2); const left = checkedI64(evalExpr(args[0], env, aggregates), `${op} left operand`); const right = checkedI64(evalExpr(args[1], env, aggregates), `${op} right operand`);
    let result;
    if (op === "add") result = left + right;
    else if (op === "sub") result = left - right;
    else if (op === "mul") result = left * right;
    else { if (right === 0n) fail("division_by_zero", `${op} by zero`); result = op === "div" ? left / right : left % right; }
    return checkedI64(result, `${op} result`);
  }
  if (op === "min" || op === "max") {
    if (!args.length) fail("invalid_expression", `${op} expects at least one argument`);
    let best = evalExpr(args[0], env, aggregates);
    for (const arg of args.slice(1)) { const value = evalExpr(arg, env, aggregates); const c = compareValues(value, best); if ((op === "min" && c < 0) || (op === "max" && c > 0)) best = value; }
    return best;
  }
  if (op === "concat") {
    const values = args.map((arg) => evalExpr(arg, env, aggregates));
    if (values.every((value) => typeof value === "string")) return values.join("");
    if (values.every((value) => value instanceof Uint8Array)) return new Uint8Array(Buffer.concat(values.map(Buffer.from)));
    fail("type_error", "concat operands must be all Text or all Bytes");
  }
  if (op === "length") { arity(1); const value = evalExpr(args[0], env, aggregates); if (typeof value === "string") return BigInt([...value].length); if (value instanceof Uint8Array || Array.isArray(value) || isMap(value)) return BigInt(value instanceof Uint8Array || Array.isArray(value) ? value.length : Object.keys(value).length); fail("type_error", "length operand has no length"); }
  if (op === "contains") {
    arity(2); const container = evalExpr(args[0], env, aggregates); const needle = evalExpr(args[1], env, aggregates);
    if (isMap(container)) return typeof needle === "string" && Object.hasOwn(container, needle);
    if (typeof container === "string" && typeof needle === "string") return container.includes(needle);
    if (container instanceof Uint8Array && needle instanceof Uint8Array) return Buffer.from(container).includes(Buffer.from(needle));
    if (Array.isArray(container)) return container.some((item) => sameValue(item, needle));
    fail("type_error", "contains operands have incompatible types");
  }
  if (op === "if") { arity(3); return evalExpr(requireBool(evalExpr(args[0], env, aggregates), "if condition") ? args[1] : args[2], env, aggregates); }
  if (op === "hash") { arity(2); if (typeof args[0] !== "string") fail("type_error", "hash domain must be literal Text"); return digestRaw(args[0], canonicalBytes(evalExpr(args[1], env, aggregates))); }
  if (op === "aggregate") { arity(1); if (typeof args[0] !== "string" || !Object.hasOwn(aggregates, args[0])) fail("unknown_aggregate", `unknown aggregate: ${args[0]}`); return aggregates[args[0]]; }
  fail("unknown_opcode", `unknown expression opcode: ${op}`);
}

const sortedEvents = (events) => [...events].sort((a, b) => Buffer.compare(parseId(a.id), parseId(b.id)));

function *enumerateBindings(positive, pool, distinct) {
  if (!positive.length) fail("invalid_rule", "at least one positive clause is required");
  const aliases = new Set();
  for (const clause of positive) {
    if (!isMap(clause) || Object.keys(clause).sort().join(",") !== "alias,topic,where") fail("invalid_rule", "positive clause has wrong fields");
    if (typeof clause.alias !== "string" || aliases.has(clause.alias)) fail("invalid_rule", "positive alias must be unique Text");
    if (typeof clause.topic !== "string" || !clause.topic) fail("invalid_rule", "positive topic must be non-empty Text");
    aliases.add(clause.alias);
  }
  for (const group of distinct) if (!Array.isArray(group) || group.length < 2 || new Set(group).size !== group.length || group.some((alias) => !aliases.has(alias))) fail("invalid_rule", "distinct group must name at least two positive aliases");
  const candidates = mapValue(positive.map((clause) => [clause.alias, sortedEvents(pool).filter((event) => event.topic === clause.topic)]));
  function *walk(index, env) {
    if (index === positive.length) {
      for (const group of distinct) if (new Set(group.map((alias) => env[alias].id)).size !== group.length) return;
      yield { ...env }; return;
    }
    const clause = positive[index];
    for (const event of candidates[clause.alias]) {
      env[clause.alias] = event;
      if (requireBool(evalExpr(clause.where, env, mapValue()), "positive where")) yield *walk(index + 1, env);
      delete env[clause.alias];
    }
  }
  yield *walk(0, mapValue());
}

function queryEvents(clause, env, pool) {
  const required = ["alias", "topic", "where"];
  const allowed = new Set([...required, "name", "op", "value"]);
  if (!isMap(clause) || required.some((key) => !Object.hasOwn(clause, key)) || Object.keys(clause).some((key) => !allowed.has(key))) fail("invalid_rule", "query clause has wrong fields");
  if (typeof clause.alias !== "string" || Object.hasOwn(env, clause.alias) || typeof clause.topic !== "string" || !clause.topic) fail("invalid_rule", "query alias/topic is invalid");
  const matches = [];
  for (const event of sortedEvents(pool)) {
    if (event.topic !== clause.topic) continue;
    const local = { ...env, [clause.alias]: event };
    if (requireBool(evalExpr(clause.where, local, mapValue()), "query where")) matches.push([event, local]);
  }
  return matches;
}

function evaluateSideClauses(negative, aggregate, env, pool) {
  for (const clause of negative) {
    if (!isMap(clause) || Object.keys(clause).sort().join(",") !== "alias,topic,where") fail("invalid_rule", "negative clause has wrong fields");
    if (queryEvents(clause, env, pool).length) return null;
  }
  const results = mapValue();
  for (const clause of aggregate) {
    if (!isMap(clause) || Object.keys(clause).some((key) => !["name", "op", "alias", "topic", "where", "value"].includes(key))) fail("invalid_rule", "aggregate clause has wrong fields");
    if (typeof clause.name !== "string" || Object.hasOwn(results, clause.name)) fail("invalid_rule", "aggregate name must be unique Text");
    if (!["count", "sum", "min", "max"].includes(clause.op)) fail("invalid_rule", "unknown aggregate operation");
    if (clause.op === "count" && Object.hasOwn(clause, "value")) fail("invalid_rule", "count aggregate must omit value");
    if (clause.op !== "count" && !Object.hasOwn(clause, "value")) fail("invalid_rule", `${clause.op} aggregate requires value`);
    const matches = queryEvents(clause, env, pool);
    if (clause.op === "count") { results[clause.name] = checkedI64(BigInt(matches.length), "count aggregate"); continue; }
    const values = matches.map(([, local]) => evalExpr(clause.value, local, results));
    if (clause.op === "sum") {
      let total = 0n;
      for (const value of values) total = checkedI64(total + checkedI64(value, "sum value"), "sum aggregate");
      results[clause.name] = total;
    } else {
      if (!values.length) fail("empty_aggregate", `${clause.op} over empty set`);
      let best = values[0];
      for (const value of values.slice(1)) { const c = compareValues(value, best); if ((clause.op === "min" && c < 0) || (clause.op === "max" && c > 0)) best = value; }
      results[clause.name] = best;
    }
  }
  return results;
}

function bindingBytes(positive, env) {
  return digestRaw("afterimage/binding/1", canonicalBytes(positive.map((clause) => [clause.alias, env[clause.alias].id])));
}

function evaluateWorld(programRaw, baseValues, limitsRaw = null) {
  const program = normalizeValue(programRaw);
  if (!isMap(program) || Object.keys(program).sort().join(",") !== "semantics,strata" || program.semantics !== "cre/0.1") fail("invalid_program", "program must use cre/0.1 and contain strata");
  if (!Array.isArray(program.strata)) fail("invalid_program", "program.strata must be a list");
  const counters = new Counters(limitsRaw);
  const baseOrder = validateBaseEvents(baseValues.map(loadEvent), counters);
  const events = new Map();
  const trace = [];
  for (const event of baseOrder) { events.set(event.id, { ...event, _stratum: -1n }); trace.push({ stratum: -1n, rule: null, binding: null, ordinal: null, event: event.id }); }
  let expectedIndex = 0n;
  const ruleIds = new Set();
  for (const stratum of program.strata) {
    if (!isMap(stratum) || Object.keys(stratum).sort().join(",") !== "index,rules" || stratum.index !== expectedIndex) fail("invalid_program", "stratum indices must be contiguous from zero");
    if (!Array.isArray(stratum.rules)) fail("invalid_program", "stratum.rules must be a list");
    for (const rule of stratum.rules) {
      const required = ["id", "positive", "negative", "aggregate", "distinct", "guard", "emit"];
      if (!isMap(rule) || Object.keys(rule).sort().join(",") !== [...required].sort().join(",") || typeof rule.id !== "string" || !rule.id) fail("invalid_rule", "rule has wrong fields or ID");
      if (ruleIds.has(rule.id)) fail("invalid_rule", "rule ID must be globally unique", { rule: rule.id });
      ruleIds.add(rule.id);
      if (["positive", "negative", "aggregate", "distinct", "emit"].some((key) => !Array.isArray(rule[key]))) fail("invalid_rule", "rule clause fields must be lists");
    }
    const rules = [...stratum.rules].sort((a, b) => Buffer.compare(Buffer.from(a.id), Buffer.from(b.id)));
    while (true) {
      const additions = new Map();
      const lowerPool = [...events.values()].filter((event) => event._stratum < expectedIndex);
      const activePool = [...events.values()];
      for (const rule of rules) for (const env of enumerateBindings(rule.positive, activePool, rule.distinct)) {
        counters.charge("bindings_tested");
        const aggregates = evaluateSideClauses(rule.negative, rule.aggregate, env, lowerPool);
        if (aggregates === null || !requireBool(evalExpr(rule.guard, env, aggregates), "rule guard")) continue;
        const binding = bindingBytes(rule.positive, env);
        for (let ordinal = 0; ordinal < rule.emit.length; ordinal++) {
          const emission = rule.emit[ordinal];
          if (!isMap(emission) || Object.keys(emission).sort().join(",") !== "at,parents,payload,topic") fail("invalid_rule", "emission has wrong fields", { rule: rule.id });
          const parents = new Set(Object.values(env).map((event) => event.id));
          if (!Array.isArray(emission.parents)) fail("invalid_rule", "emission.parents must be a list");
          for (const expression of emission.parents) { const parent = evalExpr(expression, env, aggregates); parseId(parent, "emission parent"); parents.add(parent); }
          const body = mapValue([
            ["topic", evalExpr(emission.topic, env, aggregates)],
            ["at", checkedI64(evalExpr(emission.at, env, aggregates), "emission.at")],
            ["payload", evalExpr(emission.payload, env, aggregates)],
            ["parents", [...parents].sort((a, b) => Buffer.compare(parseId(a), parseId(b)))],
            ["origin", mapValue([["kind", "derived"], ["rule", rule.id], ["binding", binding], ["ordinal", BigInt(ordinal)]])],
          ]);
          const event = makeEvent(body);
          const combined = new Map([...events, ...[...additions].map(([id, pair]) => [id, pair[0]])]);
          for (const parent of event.parents) {
            if (!combined.has(parent)) fail("missing_parent", "derived event parent is inactive", { parent });
            if (event.at < combined.get(parent).at) fail("invalid_time", "derived event precedes parent", { parent });
          }
          if (BigInt(canonicalBytes(event.payload).length) > counters.limits.max_value_bytes) fail("resource_exhausted", "derived payload exceeds max_value_bytes");
          const item = { stratum: expectedIndex, rule: rule.id, binding: ID_PREFIX + Buffer.from(binding).toString("hex"), ordinal: BigInt(ordinal), event: event.id };
          if (!events.has(event.id) && !additions.has(event.id)) additions.set(event.id, [{ ...event, _stratum: expectedIndex }, item]);
        }
      }
      if (!additions.size) break;
      for (const [identifier, [event, item]] of additions) { counters.charge("derived_events"); events.set(identifier, event); trace.push(item); }
    }
    expectedIndex += 1n;
  }
  return [events, trace, counters];
}

function evaluateProjection(raw, events, counters) {
  const projection = normalizeValue(raw);
  if (!isMap(projection) || Object.keys(projection).sort().join(",") !== "id,rows" || !Array.isArray(projection.rows)) fail("invalid_projection", "projection requires id and rows");
  const pool = [...events.values()];
  const records = [];
  let sequence = 0;
  for (const row of projection.rows) {
    const required = ["positive", "negative", "aggregate", "distinct", "guard", "value", "sort"];
    if (!isMap(row) || Object.keys(row).sort().join(",") !== [...required].sort().join(",") || ["positive", "negative", "aggregate", "distinct", "sort"].some((key) => !Array.isArray(row[key]))) fail("invalid_projection", "projection row has wrong fields");
    for (const env of enumerateBindings(row.positive, pool, row.distinct)) {
      counters.charge("bindings_tested");
      const aggregates = evaluateSideClauses(row.negative, row.aggregate, env, pool);
      if (aggregates === null || !requireBool(evalExpr(row.guard, env, aggregates), "projection guard")) continue;
      const value = evalExpr(row.value, env, aggregates);
      let keys = row.sort.map((expr) => canonicalBytes(evalExpr(expr, env, aggregates)));
      if (!keys.length) keys = [canonicalBytes(value)];
      records.push([keys, sequence++, value]); counters.charge("projection_records");
    }
  }
  records.sort((left, right) => {
    for (let i = 0; i < Math.min(left[0].length, right[0].length); i++) { const c = Buffer.compare(left[0][i], right[0][i]); if (c) return c; }
    return left[0].length - right[0].length || left[1] - right[1];
  });
  const output = records.map((record) => record[2]);
  return [output, digestId("afterimage/projection/1", canonicalBytes(output))];
}

function canonicalOperations(raw) {
  const operations = normalizeValue(raw);
  if (!Array.isArray(operations)) fail("invalid_operation", "operations must be a list");
  const ranks = { suppress: 0, replace: 1, retime: 2, inject: 3 };
  const targets = new Set();
  const normalized = [];
  for (const operation of operations) {
    if (!isMap(operation) || !Object.hasOwn(ranks, operation.kind)) fail("invalid_operation", "operation has unknown kind");
    const fields = { suppress: ["kind", "event"], replace: ["kind", "event", "pointer", "value"], retime: ["kind", "event", "at"], inject: ["kind", "topic", "at", "payload", "parents"] }[operation.kind];
    if (Object.keys(operation).sort().join(",") !== fields.sort().join(",")) fail("invalid_operation", `${operation.kind} operation has wrong fields`);
    let secondary;
    if (operation.kind !== "inject") {
      secondary = parseId(operation.event, "operation target");
      if (targets.has(operation.event)) fail("operation_conflict", "target appears in multiple operations", { event: operation.event });
      targets.add(operation.event);
    } else {
      if (typeof operation.topic !== "string" || !Array.isArray(operation.parents)) fail("invalid_operation", "inject topic/parents have wrong type");
      checkedI64(operation.at, "inject.at"); operation.parents.forEach((parent) => parseId(parent, "inject parent"));
      const body = mapValue(Object.keys(operation).filter((key) => key !== "kind").map((key) => [key, operation[key]]));
      secondary = digestRaw("afterimage/inject-operation/1", canonicalBytes(body));
    }
    normalized.push([[ranks[operation.kind], Buffer.from(secondary)], operation]);
  }
  normalized.sort((a, b) => a[0][0] - b[0][0] || Buffer.compare(a[0][1], b[0][1]));
  return normalized.map((entry) => entry[1]);
}

function rootBranchId(bundleDigest) { parseId(bundleDigest, "bundle digest"); return digestId("afterimage/root-branch/1", canonicalBytes({ bundle: bundleDigest })); }

function applyBranch(baseValues, bundleDigest, operationsRaw, parentBranch = null) {
  const base = baseValues.map(loadEvent);
  const root = rootBranchId(bundleDigest);
  const parent = parentBranch || root; parseId(parent, "parent branch");
  if (!operationsRaw || !operationsRaw.length) return [base, parent];
  const operations = canonicalOperations(operationsRaw);
  const branch = digestId("afterimage/branch/1", canonicalBytes({ parent, operations }));
  const original = new Map(base.map((event) => [event.id, event]));
  const suppressed = new Set(); const injections = [];
  for (let ordinal = 0; ordinal < operations.length; ordinal++) {
    const operation = operations[ordinal];
    if (operation.kind !== "inject") {
      if (!original.has(operation.event) || original.get(operation.event).origin.kind === "derived") fail("invalid_operation", "operation target is not an active base/player event", { event: operation.event });
      suppressed.add(operation.event);
    }
    if (operation.kind === "suppress") continue;
    let body;
    if (operation.kind === "inject") body = { topic: operation.topic, at: operation.at, payload: operation.payload, parents: [...operation.parents].sort((a, b) => Buffer.compare(parseId(a), parseId(b))), origin: { kind: "player", branch_parent: parent, ordinal: BigInt(ordinal), supersedes: null } };
    else {
      const old = original.get(operation.event);
      body = mapValue(["topic", "at", "payload", "parents", "origin"].map((key) => [key, cloneValue(old[key])]));
      body.origin = { kind: "player", branch_parent: parent, ordinal: BigInt(ordinal), supersedes: old.id };
      if (operation.kind === "retime") body.at = checkedI64(operation.at, "retime.at");
      else { if (typeof operation.pointer !== "string" || !operation.pointer.startsWith("/payload")) fail("invalid_operation", "replace pointer must target /payload in 0.1"); body = pointerReplace(body, operation.pointer, operation.value); }
    }
    injections.push(makeEvent(body));
  }
  let changed = true;
  while (changed) { changed = false; for (const event of base) if (!suppressed.has(event.id) && event.parents.some((parent) => suppressed.has(parent))) { suppressed.add(event.id); changed = true; } }
  return [[...base.filter((event) => !suppressed.has(event.id)), ...injections], branch];
}

const traceDigest = (trace) => digestId("afterimage/trace/1", canonicalBytes(trace));

function resolveFixture(caseValue) {
  const labels = new Map(); const bodies = []; let pending = [...caseValue.base_events];
  while (pending.length) {
    let progress = false; const remaining = [];
    for (const entry of pending) {
      const wrapped = isMap(entry) && Object.hasOwn(entry, "body");
      const label = wrapped ? entry.label : null; const body = cloneValue(wrapped ? entry.body : entry);
      const parents = body.parents || [];
      if (parents.some((parent) => typeof parent === "string" && parent.startsWith("@") && !labels.has(parent.slice(1)))) { remaining.push(entry); continue; }
      body.parents = parents.map((parent) => typeof parent === "string" && parent.startsWith("@") ? labels.get(parent.slice(1)) : parent);
      const event = loadEvent(body); bodies.push(eventView(event));
      if (label !== null && label !== undefined) { if (typeof label !== "string" || labels.has(label)) fail("invalid_fixture", "fixture labels must be unique Text"); labels.set(label, event.id); }
      progress = true;
    }
    if (!progress) fail("invalid_fixture", "fixture parent labels contain a cycle or missing name");
    pending = remaining;
  }
  const operations = caseValue.operations ? cloneValue(caseValue.operations) : null;
  for (const operation of operations || []) {
    if (typeof operation.event === "string" && operation.event.startsWith("@")) { const name = operation.event.slice(1); if (!labels.has(name)) fail("invalid_fixture", "unknown operation label", { label: name }); operation.event = labels.get(name); }
    if (operation.parents) operation.parents = operation.parents.map((parent) => typeof parent === "string" && parent.startsWith("@") ? labels.get(parent.slice(1)) : parent);
  }
  return [bodies, operations];
}

function runCase(raw, oracleLimit = null) {
  const caseValue = normalizeValue(raw);
  if (!isMap(caseValue) || ["name", "bundle_digest", "program", "base_events"].some((key) => !Object.hasOwn(caseValue, key))) fail("invalid_fixture", "case requires name, bundle_digest, program, and base_events");
  const [base, operations] = resolveFixture(caseValue);
  const limits = { ...(caseValue.limits || {}) };
  if (oracleLimit !== null) limits.max_derived_events = limits.max_derived_events === undefined ? oracleLimit : limits.max_derived_events < oracleLimit ? limits.max_derived_events : oracleLimit;
  const [branched, branch] = applyBranch(base, caseValue.bundle_digest, operations, caseValue.parent_branch);
  const [events, trace, counters] = evaluateWorld(caseValue.program, branched, limits);
  let projection = null; let projectionDigest = null;
  if (Object.hasOwn(caseValue, "projection")) [projection, projectionDigest] = evaluateProjection(caseValue.projection, events, counters);
  return { name: caseValue.name, branch, event_ids: [...events.keys()].sort((a, b) => Buffer.compare(parseId(a), parseId(b))), projection, projection_digest: projectionDigest, trace_digest: traceDigest(trace), counters: counters.asValue() };
}

function runSuite(raw, oracleLimit = null) {
  const suite = normalizeValue(raw);
  if (!isMap(suite) || Object.keys(suite).sort().join(",") !== "canonical_vectors,cases,format") fail("invalid_fixture", "suite has wrong fields");
  const vectors = suite.canonical_vectors.map((vector) => { const encoded = canonicalBytes(vector.value); return { name: vector.name, canonical: encoded.toString("utf8"), digest: digestId(vector.domain || "afterimage/test-vector/1", encoded) }; });
  const cases = suite.cases.map((caseValue) => { try { return { ok: runCase(caseValue, oracleLimit) }; } catch (error) { if (!(error instanceof CREError)) throw error; return { error: error.asValue(), name: caseValue.name }; } });
  return { format: "afterimage-conformance-result/0.1", canonical_vectors: vectors, cases };
}

function protocolRecords(raw) {
  const caseValue = normalizeValue(raw);
  const result = runCase(caseValue);
  const records = [{ type: "ready", semantics: "cre/0.1", bundle: caseValue.bundle_digest }];
  for (let index = 0; index < (result.projection || []).length; index++) {
    records.push({ type: "projection", index: BigInt(index), value: result.projection[index] });
  }
  records.push({ type: "done", branch: result.branch, projection: result.projection_digest, trace: result.trace_digest, counters: result.counters });
  return records;
}

const protocolError = (error) => ({ type: "error", ...error.asValue() });

// JSON parser that preserves every signed 64-bit integer exactly.
function parseJson(text) {
  let index = 0;
  const whitespace = () => { while ([" ", "\t", "\r", "\n"].includes(text[index])) index++; };
  const value = () => {
    whitespace(); const char = text[index];
    if (char === '"') return string();
    if (char === "[") return array();
    if (char === "{") return object();
    if (text.startsWith("true", index)) { index += 4; return true; }
    if (text.startsWith("false", index)) { index += 5; return false; }
    if (text.startsWith("null", index)) { index += 4; return null; }
    const match = text.slice(index).match(/^-?(?:0|[1-9][0-9]*)/);
    if (!match) fail("invalid_json", `unexpected token at byte ${Buffer.byteLength(text.slice(0, index))}`);
    index += match[0].length;
    if (/[.eE]/.test(text[index] || "")) fail("type_error", "floating-point values are forbidden");
    return checkedI64(BigInt(match[0]));
  };
  const string = () => {
    const start = index++;
    while (index < text.length) {
      const char = text[index++];
      if (char === '"') {
        try {
          const decoded = JSON.parse(text.slice(start, index));
          for (const scalar of decoded) {
            const point = scalar.codePointAt(0);
            if (point >= 0xd800 && point <= 0xdfff) fail("invalid_text", "Text must contain Unicode scalar values");
          }
          return decoded;
        }
        catch (error) {
          if (error instanceof CREError) throw error;
          fail("invalid_json", "invalid JSON string");
        }
      }
      if (char === "\\") { if (index >= text.length) fail("invalid_json", "unterminated JSON escape"); if (text[index++] === "u") index += 4; }
      else if (char.charCodeAt(0) < 0x20) fail("invalid_json", "unescaped control in JSON string");
    }
    fail("invalid_json", "unterminated JSON string");
  };
  const array = () => {
    index++; whitespace(); const result = [];
    if (text[index] === "]") { index++; return result; }
    while (true) { result.push(value()); whitespace(); if (text[index] === "]") { index++; return result; } if (text[index++] !== ",") fail("invalid_json", "expected comma in array"); }
  };
  const object = () => {
    index++; whitespace(); const result = mapValue();
    if (text[index] === "}") { index++; return result; }
    while (true) {
      whitespace(); if (text[index] !== '"') fail("invalid_json", "object key must be a string"); const key = string();
      if (Object.hasOwn(result, key)) fail("duplicate_key", "duplicate JSON object key", { key });
      whitespace(); if (text[index++] !== ":") fail("invalid_json", "expected colon in object"); result[key] = value(); whitespace();
      if (text[index] === "}") { index++; return result; } if (text[index++] !== ",") fail("invalid_json", "expected comma in object");
    }
  };
  const result = value(); whitespace(); if (index !== text.length) fail("invalid_json", "trailing JSON input"); return normalizeValue(result);
}

function prettyValue(value) {
  if (typeof value === "bigint") return value >= Number.MIN_SAFE_INTEGER && value <= Number.MAX_SAFE_INTEGER ? Number(value) : value.toString();
  if (value instanceof Uint8Array) return { $bytes: Buffer.from(value).toString("base64url") };
  if (Array.isArray(value)) return value.map(prettyValue);
  if (isMap(value)) return Object.fromEntries(Object.keys(value).map((key) => [key, prettyValue(value[key])]));
  return value;
}

function main(argv) {
  let pretty = false; let oracleLimit = null; const positional = [];
  for (let index = 0; index < argv.length; index++) {
    if (argv[index] === "--pretty") pretty = true;
    else if (argv[index] === "--oracle-limit") oracleLimit = checkedI64(BigInt(argv[++index]), "oracle limit");
    else positional.push(argv[index]);
  }
  if (positional.length !== 2 || !["case", "suite", "protocol"].includes(positional[0])) {
    console.error("usage: cre.mjs {case|suite|protocol} INPUT [--oracle-limit N] [--pretty]"); return 2;
  }
  let input;
  try {
    input = parseJson(readFileSync(positional[1], "utf8"));
  } catch (error) {
    if (error instanceof CREError) {
      const record = positional[0] === "protocol" ? protocolError(error) : { error: error.asValue() };
      console.log(canonicalText(record));
    }
    else console.error(String(error));
    return 2;
  }
  try {
    if (positional[0] === "protocol") {
      for (const record of protocolRecords(input)) console.log(canonicalText(record));
    } else {
      const result = positional[0] === "case" ? runCase(input, oracleLimit) : runSuite(input, oracleLimit);
      console.log(pretty ? JSON.stringify(prettyValue(result), null, 2) : canonicalText(result));
    }
    return 0;
  } catch (error) {
    if (error instanceof CREError) {
      const record = positional[0] === "protocol" ? protocolError(error) : { error: error.asValue() };
      console.log(canonicalText(record));
      return error.code === "resource_exhausted" ? 4 : 3;
    }
    console.error(error.stack || String(error)); return 5;
  }
}

process.exitCode = main(process.argv.slice(2));
