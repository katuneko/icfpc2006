#!/usr/bin/env python3
"""Independent Python reference implementation of CRE 0.1.

The implementation favors direct correspondence with the normative spec over
performance. It has no third-party dependencies and is suitable for golden
conformance generation and the bounded trace oracle.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
ID_PREFIX = "sha256:"
DEFAULT_LIMITS = {
    "max_base_events": 100_000,
    "max_derived_events": 1_000_000,
    "max_bindings_tested": 10_000_000,
    "max_value_bytes": 1_048_576,
    "max_projection_records": 1_000_000,
}


class CREError(Exception):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}

    def as_value(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


def fail(code: str, message: str, **context: Any) -> None:
    raise CREError(code, message, context)


def checked_i64(value: Any, where: str = "integer") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail("type_error", f"{where} must be Int")
    if not I64_MIN <= value <= I64_MAX:
        fail("integer_overflow", f"{where} is outside signed 64-bit range")
    return value


def normalize_text(value: str) -> str:
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        fail("invalid_text", "Text must contain Unicode scalar values")
    return unicodedata.normalize("NFC", value)


def normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return checked_i64(value)
    if isinstance(value, float):
        fail("type_error", "floating-point values are forbidden")
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, bytes):
        return value
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        if "$bytes" in value:
            if set(value) != {"$bytes"} or not isinstance(value["$bytes"], str):
                fail("invalid_bytes", "$bytes is reserved for a sole text field")
            encoded = value["$bytes"]
            if "=" in encoded:
                fail("invalid_bytes", "base64url bytes must omit padding")
            try:
                raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            except Exception as exc:  # binascii.Error varies by Python version
                raise CREError("invalid_bytes", "invalid base64url bytes") from exc
            canonical = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
            if canonical != encoded:
                fail("invalid_bytes", "non-canonical base64url bytes")
            return raw
        result: dict[str, Any] = {}
        raw_keys: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                fail("type_error", "map key must be Text")
            normalized = normalize_text(key)
            if normalized in result:
                fail(
                    "duplicate_key",
                    "map keys collide after NFC normalization",
                    first=raw_keys[normalized],
                    second=key,
                )
            raw_keys[normalized] = key
            result[normalized] = normalize_value(item)
        return result
    fail("type_error", f"unsupported value type: {type(value).__name__}")


def json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"$bytes": base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    return value


def escape_text(value: str) -> str:
    pieces = ['"']
    short = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    for char in normalize_text(value):
        code = ord(char)
        if char == '"':
            pieces.append('\\"')
        elif char == "\\":
            pieces.append("\\\\")
        elif char in short:
            pieces.append(short[char])
        elif code < 0x20:
            pieces.append(f"\\u{code:04x}")
        else:
            pieces.append(char)
    pieces.append('"')
    return "".join(pieces)


def canonical_text(value: Any) -> str:
    value = normalize_value(value)
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return escape_text(value)
    if isinstance(value, bytes):
        encoded = base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
        return '{"$bytes":' + escape_text(encoded) + "}"
    if isinstance(value, list):
        return "[" + ",".join(canonical_text(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: key.encode("utf-8"))
        return "{" + ",".join(escape_text(key) + ":" + canonical_text(value[key]) for key in keys) + "}"
    fail("type_error", "unreachable canonical value type")


def canonical_bytes(value: Any) -> bytes:
    return canonical_text(value).encode("utf-8")


def uleb128(value: int) -> bytes:
    if value < 0:
        fail("invalid_length", "ULEB128 value must be non-negative")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            output.append(byte | 0x80)
        else:
            output.append(byte)
            return bytes(output)


def digest_raw(domain: str, *parts: bytes) -> bytes:
    domain = normalize_text(domain)
    state = hashlib.sha256()
    state.update(domain.encode("utf-8"))
    state.update(b"\0")
    for part in parts:
        state.update(uleb128(len(part)))
        state.update(part)
    return state.digest()


def digest_id(domain: str, *parts: bytes) -> str:
    return ID_PREFIX + digest_raw(domain, *parts).hex()


def parse_id(value: Any, where: str = "ID") -> bytes:
    if not isinstance(value, str) or not value.startswith(ID_PREFIX) or len(value) != 71:
        fail("invalid_id", f"{where} must be sha256:<64 lowercase hex>")
    text = value[len(ID_PREFIX) :]
    if any(char not in "0123456789abcdef" for char in text):
        fail("invalid_id", f"{where} must use lowercase hexadecimal")
    return bytes.fromhex(text)


def same_value(left: Any, right: Any) -> bool:
    return canonical_bytes(left) == canonical_bytes(right)


def compare_values(left: Any, right: Any) -> int:
    if isinstance(left, bool) or isinstance(right, bool):
        fail("type_error", "Bool is not ordered")
    if isinstance(left, int) and isinstance(right, int):
        return (left > right) - (left < right)
    if isinstance(left, str) and isinstance(right, str):
        lkey, rkey = left.encode("utf-8"), right.encode("utf-8")
        return (lkey > rkey) - (lkey < rkey)
    if isinstance(left, bytes) and isinstance(right, bytes):
        return (left > right) - (left < right)
    fail("type_error", "ordered values must have matching Int, Text, or Bytes type")


def pointer_tokens(pointer: Any) -> list[str]:
    if not isinstance(pointer, str):
        fail("invalid_pointer", "JSON Pointer must be Text")
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        fail("invalid_pointer", "JSON Pointer must be empty or begin with /")
    result = []
    for raw in pointer[1:].split("/"):
        index = 0
        decoded = []
        while index < len(raw):
            if raw[index] == "~":
                if index + 1 >= len(raw) or raw[index + 1] not in "01":
                    fail("invalid_pointer", "invalid JSON Pointer escape")
                decoded.append("~" if raw[index + 1] == "0" else "/")
                index += 2
            else:
                decoded.append(raw[index])
                index += 1
        result.append("".join(decoded))
    return result


def pointer_get(value: Any, pointer: Any) -> Any:
    current = value
    for token in pointer_tokens(pointer):
        if isinstance(current, dict):
            if token not in current:
                fail("missing_path", "JSON Pointer map key is missing", token=token)
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                fail("invalid_pointer", "list index must be canonical decimal")
            index = int(token)
            if index >= len(current):
                fail("missing_path", "JSON Pointer list index is out of range", index=index)
            current = current[index]
        else:
            fail("missing_path", "JSON Pointer traverses a scalar", token=token)
    return current


def pointer_replace(value: Any, pointer: Any, replacement: Any) -> Any:
    tokens = pointer_tokens(pointer)
    if not tokens:
        return normalize_value(replacement)
    result = copy.deepcopy(value)
    current = result
    for token in tokens[:-1]:
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            fail("missing_path", "replace pointer path is missing", token=token)
    final = tokens[-1]
    if isinstance(current, dict) and final in current:
        current[final] = normalize_value(replacement)
    elif isinstance(current, list) and final.isdigit() and int(final) < len(current):
        current[int(final)] = normalize_value(replacement)
    else:
        fail("missing_path", "replace pointer target is missing", token=final)
    return result


def event_view(event: dict[str, Any]) -> dict[str, Any]:
    return {key: event[key] for key in ("id", "topic", "at", "payload", "parents", "origin")}


def event_id(body: dict[str, Any]) -> str:
    return digest_id("afterimage/event/1", canonical_bytes(body))


def validate_origin(origin: Any) -> dict[str, Any]:
    if not isinstance(origin, dict) or not isinstance(origin.get("kind"), str):
        fail("invalid_origin", "origin must contain a kind")
    kind = origin["kind"]
    if kind == "base":
        if set(origin) != {"kind", "source", "sequence"} or not isinstance(origin["source"], str):
            fail("invalid_origin", "base origin requires kind, source, and sequence")
        checked_i64(origin["sequence"], "origin.sequence")
        if origin["sequence"] < 0:
            fail("invalid_origin", "origin.sequence must be non-negative")
    elif kind == "derived":
        if set(origin) != {"kind", "rule", "binding", "ordinal"}:
            fail("invalid_origin", "derived origin has wrong fields")
        if not isinstance(origin["rule"], str) or not isinstance(origin["binding"], bytes):
            fail("invalid_origin", "derived rule and binding have wrong type")
        checked_i64(origin["ordinal"], "origin.ordinal")
        if origin["ordinal"] < 0:
            fail("invalid_origin", "origin.ordinal must be non-negative")
    elif kind == "player":
        if set(origin) != {"kind", "branch_parent", "ordinal", "supersedes"}:
            fail("invalid_origin", "player origin has wrong fields")
        parse_id(origin["branch_parent"], "origin.branch_parent")
        checked_i64(origin["ordinal"], "origin.ordinal")
        if origin["ordinal"] < 0:
            fail("invalid_origin", "origin.ordinal must be non-negative")
        if origin["supersedes"] is not None:
            parse_id(origin["supersedes"], "origin.supersedes")
    else:
        fail("invalid_origin", f"unknown origin kind: {kind}")
    return normalize_value(origin)


def make_event(body_value: Any, claimed_id: str | None = None) -> dict[str, Any]:
    body = normalize_value(body_value)
    if not isinstance(body, dict) or set(body) != {"topic", "at", "payload", "parents", "origin"}:
        fail("invalid_event", "event body has wrong fields")
    if not isinstance(body["topic"], str) or not 0 < len(body["topic"].encode("utf-8")) <= 128:
        fail("invalid_event", "event topic must be 1..128 UTF-8 bytes")
    checked_i64(body["at"], "event.at")
    if not isinstance(body["parents"], list):
        fail("invalid_event", "event.parents must be a list")
    parent_raw = [parse_id(parent, "event parent") for parent in body["parents"]]
    if parent_raw != sorted(parent_raw) or len(set(parent_raw)) != len(parent_raw):
        fail("invalid_event", "event parents must be unique and raw-digest sorted")
    body["origin"] = validate_origin(body["origin"])
    identifier = event_id(body)
    if claimed_id is not None and claimed_id != identifier:
        fail("event_id_mismatch", "claimed event ID does not match body", claimed=claimed_id, actual=identifier)
    return {"id": identifier, **body}


def load_event(value: Any) -> dict[str, Any]:
    value = normalize_value(value)
    if isinstance(value, dict) and "id" in value:
        claimed = value["id"]
        return make_event({key: value[key] for key in value if key != "id"}, claimed)
    return make_event(value)


def validate_base_events(values: list[dict[str, Any]], limits: "Counters") -> list[dict[str, Any]]:
    limits.charge("base_events", len(values))
    events = {event["id"]: event for event in values}
    if len(events) != len(values):
        fail("duplicate_event", "duplicate base event ID")
    for event in values:
        if event["origin"]["kind"] == "derived":
            fail("invalid_base", "derived event supplied as base")
        if len(canonical_bytes(event["payload"])) > limits.limits["max_value_bytes"]:
            fail("resource_exhausted", "event payload exceeds max_value_bytes")
        for parent in event["parents"]:
            if parent not in events:
                fail("missing_parent", "base event parent is inactive", event=event["id"], parent=parent)
            if event["origin"]["kind"] == "base" and events[parent]["origin"]["kind"] != "base":
                fail("invalid_parent", "base event may cite only a base parent")
            if event["at"] < events[parent]["at"]:
                fail("invalid_time", "event time precedes parent", event=event["id"], parent=parent)

    indegree = {identifier: 0 for identifier in events}
    children: dict[str, list[str]] = {identifier: [] for identifier in events}
    for event in values:
        indegree[event["id"]] = len(event["parents"])
        for parent in event["parents"]:
            children[parent].append(event["id"])
    ready = sorted(
        (identifier for identifier, degree in indegree.items() if degree == 0),
        key=lambda identifier: (events[identifier]["at"], parse_id(identifier)),
    )
    ordered: list[dict[str, Any]] = []
    while ready:
        identifier = ready.pop(0)
        ordered.append(events[identifier])
        for child in children[identifier]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=lambda item: (events[item]["at"], parse_id(item)))
    if len(ordered) != len(values):
        fail("causal_cycle", "base event parent graph contains a cycle")
    return ordered


@dataclass
class Counters:
    limits: dict[str, int]
    base_events: int = 0
    derived_events: int = 0
    bindings_tested: int = 0
    projection_records: int = 0

    @classmethod
    def create(cls, supplied: dict[str, Any] | None) -> "Counters":
        limits = dict(DEFAULT_LIMITS)
        if supplied:
            for key, value in supplied.items():
                if key not in limits or isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    fail("invalid_limit", f"invalid resource limit: {key}")
                limits[key] = value
        return cls(limits=limits)

    def charge(self, name: str, amount: int = 1) -> None:
        field = name
        setattr(self, field, getattr(self, field) + amount)
        limit_key = "max_" + name
        if getattr(self, field) > self.limits[limit_key]:
            fail("resource_exhausted", f"crossed {limit_key}", counter=name, value=getattr(self, field))

    def as_value(self) -> dict[str, int]:
        return {
            "base_events": self.base_events,
            "derived_events": self.derived_events,
            "bindings_tested": self.bindings_tested,
            "projection_records": self.projection_records,
        }


def require_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        fail("type_error", f"{where} must evaluate to Bool")
    return value


def eval_expr(expr: Any, env: dict[str, dict[str, Any]], aggregates: dict[str, Any]) -> Any:
    if not isinstance(expr, list) or not expr or not isinstance(expr[0], str):
        fail("invalid_expression", "expression must be a non-empty opcode list")
    op, args = expr[0], expr[1:]

    def arity(count: int) -> None:
        if len(args) != count:
            fail("invalid_expression", f"{op} expects {count} arguments")

    if op == "const":
        arity(1)
        return normalize_value(args[0])
    if op == "get":
        arity(2)
        alias, pointer = args
        if not isinstance(alias, str) or alias not in env:
            fail("unknown_alias", f"unknown event alias: {alias}")
        return pointer_get(event_view(env[alias]), pointer)
    if op == "list":
        return [eval_expr(arg, env, aggregates) for arg in args]
    if op == "map":
        if len(args) % 2:
            fail("invalid_expression", "map expects key/expression pairs")
        result: dict[str, Any] = {}
        for index in range(0, len(args), 2):
            key = args[index]
            if not isinstance(key, str):
                fail("invalid_expression", "map key must be Text")
            key = normalize_text(key)
            if key in result:
                fail("duplicate_key", "duplicate map key in expression", key=key)
            result[key] = eval_expr(args[index + 1], env, aggregates)
        return result
    if op in {"eq", "ne"}:
        arity(2)
        equal = same_value(eval_expr(args[0], env, aggregates), eval_expr(args[1], env, aggregates))
        return equal if op == "eq" else not equal
    if op in {"lt", "le", "gt", "ge"}:
        arity(2)
        comparison = compare_values(eval_expr(args[0], env, aggregates), eval_expr(args[1], env, aggregates))
        return {"lt": comparison < 0, "le": comparison <= 0, "gt": comparison > 0, "ge": comparison >= 0}[op]
    if op == "and":
        for arg in args:
            if not require_bool(eval_expr(arg, env, aggregates), "and operand"):
                return False
        return True
    if op == "or":
        for arg in args:
            if require_bool(eval_expr(arg, env, aggregates), "or operand"):
                return True
        return False
    if op == "not":
        arity(1)
        return not require_bool(eval_expr(args[0], env, aggregates), "not operand")
    if op in {"add", "sub", "mul", "div", "mod"}:
        arity(2)
        left = checked_i64(eval_expr(args[0], env, aggregates), f"{op} left operand")
        right = checked_i64(eval_expr(args[1], env, aggregates), f"{op} right operand")
        if op == "add":
            result = left + right
        elif op == "sub":
            result = left - right
        elif op == "mul":
            result = left * right
        else:
            if right == 0:
                fail("division_by_zero", f"{op} by zero")
            quotient = abs(left) // abs(right)
            if (left < 0) != (right < 0):
                quotient = -quotient
            result = quotient if op == "div" else left - quotient * right
        return checked_i64(result, f"{op} result")
    if op in {"min", "max"}:
        if not args:
            fail("invalid_expression", f"{op} expects at least one argument")
        values = [eval_expr(arg, env, aggregates) for arg in args]
        best = values[0]
        for candidate in values[1:]:
            comparison = compare_values(candidate, best)
            if (op == "min" and comparison < 0) or (op == "max" and comparison > 0):
                best = candidate
        return best
    if op == "concat":
        values = [eval_expr(arg, env, aggregates) for arg in args]
        if all(isinstance(value, str) for value in values):
            return "".join(values)
        if all(isinstance(value, bytes) for value in values):
            return b"".join(values)
        fail("type_error", "concat operands must be all Text or all Bytes")
    if op == "length":
        arity(1)
        value = eval_expr(args[0], env, aggregates)
        if isinstance(value, (str, bytes, list, dict)):
            return checked_i64(len(value), "length result")
        fail("type_error", "length operand has no length")
    if op == "contains":
        arity(2)
        container = eval_expr(args[0], env, aggregates)
        needle = eval_expr(args[1], env, aggregates)
        if isinstance(container, dict):
            return isinstance(needle, str) and needle in container
        if isinstance(container, (str, bytes)) and isinstance(needle, type(container)):
            return needle in container
        if isinstance(container, list):
            return any(same_value(needle, item) for item in container)
        fail("type_error", "contains operands have incompatible types")
    if op == "if":
        arity(3)
        branch = args[1] if require_bool(eval_expr(args[0], env, aggregates), "if condition") else args[2]
        return eval_expr(branch, env, aggregates)
    if op == "hash":
        arity(2)
        if not isinstance(args[0], str):
            fail("type_error", "hash domain must be literal Text")
        value = eval_expr(args[1], env, aggregates)
        return digest_raw(args[0], canonical_bytes(value))
    if op == "aggregate":
        arity(1)
        name = args[0]
        if not isinstance(name, str) or name not in aggregates:
            fail("unknown_aggregate", f"unknown aggregate: {name}")
        return aggregates[name]
    fail("unknown_opcode", f"unknown expression opcode: {op}")


def sorted_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(events, key=lambda event: parse_id(event["id"]))


def enumerate_bindings(
    positive: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    distinct: list[list[str]],
) -> Iterator[dict[str, dict[str, Any]]]:
    if not positive:
        fail("invalid_rule", "at least one positive clause is required")
    aliases: set[str] = set()
    for clause in positive:
        if set(clause) != {"alias", "topic", "where"}:
            fail("invalid_rule", "positive clause has wrong fields")
        if not isinstance(clause["alias"], str) or clause["alias"] in aliases:
            fail("invalid_rule", "positive alias must be unique Text")
        if not isinstance(clause["topic"], str) or not clause["topic"]:
            fail("invalid_rule", "positive topic must be non-empty Text")
        aliases.add(clause["alias"])
    for group in distinct:
        if (
            not isinstance(group, list)
            or len(group) < 2
            or len(set(group)) != len(group)
            or any(alias not in aliases for alias in group)
        ):
            fail("invalid_rule", "distinct group must name at least two positive aliases")

    candidates = {
        clause["alias"]: [event for event in sorted_events(pool) if event["topic"] == clause["topic"]]
        for clause in positive
    }

    def walk(index: int, env: dict[str, dict[str, Any]]) -> Iterator[dict[str, dict[str, Any]]]:
        if index == len(positive):
            for group in distinct:
                identifiers = [env[alias]["id"] for alias in group]
                if len(set(identifiers)) != len(identifiers):
                    return
            yield dict(env)
            return
        clause = positive[index]
        alias = clause["alias"]
        for event in candidates[alias]:
            env[alias] = event
            if require_bool(eval_expr(clause["where"], env, {}), "positive where"):
                yield from walk(index + 1, env)
            del env[alias]

    yield from walk(0, {})


def query_events(clause: dict[str, Any], env: dict[str, dict[str, Any]], pool: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, dict[str, Any]]]]:
    required = {"alias", "topic", "where"}
    if not isinstance(clause, dict) or not required.issubset(clause) or set(clause) - (required | {"name", "op", "value"}):
        fail("invalid_rule", "query clause has wrong fields")
    alias, topic = clause["alias"], clause["topic"]
    if not isinstance(alias, str) or alias in env or not isinstance(topic, str) or not topic:
        fail("invalid_rule", "query alias/topic is invalid")
    matches = []
    for event in sorted_events(pool):
        if event["topic"] != topic:
            continue
        local = dict(env)
        local[alias] = event
        if require_bool(eval_expr(clause["where"], local, {}), "query where"):
            matches.append((event, local))
    return matches


def evaluate_side_clauses(
    negative: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    env: dict[str, dict[str, Any]],
    pool: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for clause in negative:
        if set(clause) != {"alias", "topic", "where"}:
            fail("invalid_rule", "negative clause has wrong fields")
        if query_events(clause, env, pool):
            return None
    results: dict[str, Any] = {}
    for clause in aggregate:
        allowed = {"name", "op", "alias", "topic", "where", "value"}
        if not isinstance(clause, dict) or set(clause) - allowed:
            fail("invalid_rule", "aggregate clause has wrong fields")
        name, op = clause.get("name"), clause.get("op")
        if not isinstance(name, str) or name in results:
            fail("invalid_rule", "aggregate name must be unique Text")
        if op not in {"count", "sum", "min", "max"}:
            fail("invalid_rule", "unknown aggregate operation")
        if op == "count" and "value" in clause:
            fail("invalid_rule", "count aggregate must omit value")
        if op != "count" and "value" not in clause:
            fail("invalid_rule", f"{op} aggregate requires value")
        matches = query_events(clause, env, pool)
        if op == "count":
            results[name] = checked_i64(len(matches), "count aggregate")
            continue
        values = [eval_expr(clause["value"], local, results) for _, local in matches]
        if op == "sum":
            total = 0
            for value in values:
                total = checked_i64(total + checked_i64(value, "sum value"), "sum aggregate")
            results[name] = total
        else:
            if not values:
                fail("empty_aggregate", f"{op} over empty set")
            best = values[0]
            for value in values[1:]:
                comparison = compare_values(value, best)
                if (op == "min" and comparison < 0) or (op == "max" and comparison > 0):
                    best = value
            results[name] = best
    return results


def binding_bytes(positive: list[dict[str, Any]], env: dict[str, dict[str, Any]]) -> bytes:
    pairs = [[clause["alias"], env[clause["alias"]]["id"]] for clause in positive]
    return digest_raw("afterimage/binding/1", canonical_bytes(pairs))


def evaluate_world(
    program_value: Any,
    base_values: list[Any],
    limits_value: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], Counters]:
    program = normalize_value(program_value)
    if not isinstance(program, dict) or set(program) != {"semantics", "strata"} or program["semantics"] != "cre/0.1":
        fail("invalid_program", "program must use cre/0.1 and contain strata")
    if not isinstance(program["strata"], list):
        fail("invalid_program", "program.strata must be a list")
    counters = Counters.create(limits_value)
    base = [load_event(value) for value in base_values]
    base_order = validate_base_events(base, counters)
    events: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    for event in base_order:
        internal = {**event, "_stratum": -1}
        events[event["id"]] = internal
        trace.append({"stratum": -1, "rule": None, "binding": None, "ordinal": None, "event": event["id"]})

    expected_index = 0
    rule_ids: set[str] = set()
    for stratum in program["strata"]:
        if not isinstance(stratum, dict) or set(stratum) != {"index", "rules"} or stratum["index"] != expected_index:
            fail("invalid_program", "stratum indices must be contiguous from zero")
        if not isinstance(stratum["rules"], list):
            fail("invalid_program", "stratum.rules must be a list")
        for rule in stratum["rules"]:
            required = {"id", "positive", "negative", "aggregate", "distinct", "guard", "emit"}
            if not isinstance(rule, dict) or set(rule) != required or not isinstance(rule["id"], str) or not rule["id"]:
                fail("invalid_rule", "rule has wrong fields or ID")
            if rule["id"] in rule_ids:
                fail("invalid_rule", "rule ID must be globally unique", rule=rule["id"])
            rule_ids.add(rule["id"])
            if not all(isinstance(rule[key], list) for key in ("positive", "negative", "aggregate", "distinct", "emit")):
                fail("invalid_rule", "rule clause fields must be lists")
        rules = sorted(stratum["rules"], key=lambda rule: rule["id"].encode("utf-8"))

        while True:
            additions: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
            lower_pool = [event for event in events.values() if event["_stratum"] < expected_index]
            active_pool = list(events.values())
            for rule in rules:
                for env in enumerate_bindings(rule["positive"], active_pool, rule["distinct"]):
                    counters.charge("bindings_tested")
                    aggregates = evaluate_side_clauses(rule["negative"], rule["aggregate"], env, lower_pool)
                    if aggregates is None or not require_bool(eval_expr(rule["guard"], env, aggregates), "rule guard"):
                        continue
                    binding = binding_bytes(rule["positive"], env)
                    for ordinal, emission in enumerate(rule["emit"]):
                        if not isinstance(emission, dict) or set(emission) != {"topic", "at", "payload", "parents"}:
                            fail("invalid_rule", "emission has wrong fields", rule=rule["id"])
                        topic = eval_expr(emission["topic"], env, aggregates)
                        at = checked_i64(eval_expr(emission["at"], env, aggregates), "emission.at")
                        payload = eval_expr(emission["payload"], env, aggregates)
                        parents = {event["id"] for event in env.values()}
                        if not isinstance(emission["parents"], list):
                            fail("invalid_rule", "emission.parents must be a list")
                        for expression in emission["parents"]:
                            parent = eval_expr(expression, env, aggregates)
                            parse_id(parent, "emission parent")
                            parents.add(parent)
                        parent_list = sorted(parents, key=parse_id)
                        body = {
                            "topic": topic,
                            "at": at,
                            "payload": payload,
                            "parents": parent_list,
                            "origin": {"kind": "derived", "rule": rule["id"], "binding": binding, "ordinal": ordinal},
                        }
                        event = make_event(body)
                        combined = {**events, **{identifier: pair[0] for identifier, pair in additions.items()}}
                        for parent in event["parents"]:
                            if parent not in combined:
                                fail("missing_parent", "derived event parent is inactive", parent=parent)
                            if event["at"] < combined[parent]["at"]:
                                fail("invalid_time", "derived event precedes parent", parent=parent)
                        if len(canonical_bytes(event["payload"])) > counters.limits["max_value_bytes"]:
                            fail("resource_exhausted", "derived payload exceeds max_value_bytes")
                        item = {
                            "stratum": expected_index,
                            "rule": rule["id"],
                            "binding": ID_PREFIX + binding.hex(),
                            "ordinal": ordinal,
                            "event": event["id"],
                        }
                        if event["id"] in additions and not same_value(event_view(additions[event["id"]][0]), event_view(event)):
                            fail("digest_collision", "two event bodies claim one ID")
                        if event["id"] not in events and event["id"] not in additions:
                            additions[event["id"]] = ({**event, "_stratum": expected_index}, item)
            if not additions:
                break
            for identifier, (event, item) in additions.items():
                counters.charge("derived_events")
                events[identifier] = event
                trace.append(item)
        expected_index += 1
    return events, trace, counters


def evaluate_projection(
    projection_value: Any,
    events: dict[str, dict[str, Any]],
    counters: Counters,
) -> tuple[list[Any], str]:
    projection = normalize_value(projection_value)
    if not isinstance(projection, dict) or set(projection) != {"id", "rows"} or not isinstance(projection["rows"], list):
        fail("invalid_projection", "projection requires id and rows")
    pool = list(events.values())
    records: list[tuple[tuple[bytes, ...], int, Any]] = []
    sequence = 0
    for row in projection["rows"]:
        required = {"positive", "negative", "aggregate", "distinct", "guard", "value", "sort"}
        if not isinstance(row, dict) or set(row) != required or not all(
            isinstance(row[key], list) for key in ("positive", "negative", "aggregate", "distinct", "sort")
        ):
            fail("invalid_projection", "projection row has wrong fields")
        for env in enumerate_bindings(row["positive"], pool, row["distinct"]):
            counters.charge("bindings_tested")
            aggregates = evaluate_side_clauses(row["negative"], row["aggregate"], env, pool)
            if aggregates is None or not require_bool(eval_expr(row["guard"], env, aggregates), "projection guard"):
                continue
            value = eval_expr(row["value"], env, aggregates)
            keys = tuple(canonical_bytes(eval_expr(expr, env, aggregates)) for expr in row["sort"])
            if not keys:
                keys = (canonical_bytes(value),)
            records.append((keys, sequence, value))
            sequence += 1
            counters.charge("projection_records")
    records.sort(key=lambda item: (item[0], item[1]))
    output = [value for _, _, value in records]
    return output, digest_id("afterimage/projection/1", canonical_bytes(output))


def canonical_operations(operations_value: Any) -> list[dict[str, Any]]:
    operations = normalize_value(operations_value)
    if not isinstance(operations, list):
        fail("invalid_operation", "operations must be a list")
    ranks = {"suppress": 0, "replace": 1, "retime": 2, "inject": 3}
    normalized: list[tuple[tuple[int, bytes], dict[str, Any]]] = []
    targets: set[str] = set()
    for operation in operations:
        if not isinstance(operation, dict) or operation.get("kind") not in ranks:
            fail("invalid_operation", "operation has unknown kind")
        kind = operation["kind"]
        expected = {
            "suppress": {"kind", "event"},
            "replace": {"kind", "event", "pointer", "value"},
            "retime": {"kind", "event", "at"},
            "inject": {"kind", "topic", "at", "payload", "parents"},
        }[kind]
        if set(operation) != expected:
            fail("invalid_operation", f"{kind} operation has wrong fields")
        if kind != "inject":
            target = operation["event"]
            raw = parse_id(target, "operation target")
            if target in targets:
                fail("operation_conflict", "target appears in multiple operations", event=target)
            targets.add(target)
            secondary = raw
        else:
            if not isinstance(operation["topic"], str) or not isinstance(operation["parents"], list):
                fail("invalid_operation", "inject topic/parents have wrong type")
            checked_i64(operation["at"], "inject.at")
            for parent in operation["parents"]:
                parse_id(parent, "inject parent")
            body = {key: value for key, value in operation.items() if key != "kind"}
            secondary = digest_raw("afterimage/inject-operation/1", canonical_bytes(body))
        normalized.append(((ranks[kind], secondary), operation))
    normalized.sort(key=lambda item: item[0])
    return [operation for _, operation in normalized]


def root_branch_id(bundle_digest: str) -> str:
    parse_id(bundle_digest, "bundle digest")
    return digest_id("afterimage/root-branch/1", canonical_bytes({"bundle": bundle_digest}))


def apply_branch(
    base_values: list[Any],
    bundle_digest: str,
    operations_value: list[Any] | None,
    parent_branch: str | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    base = [load_event(value) for value in base_values]
    labels: dict[str, str] = {}
    root = root_branch_id(bundle_digest)
    parent = parent_branch or root
    parse_id(parent, "parent branch")
    if not operations_value:
        return base, parent, labels
    operations = canonical_operations(operations_value)
    branch = digest_id(
        "afterimage/branch/1",
        canonical_bytes({"parent": parent, "operations": operations}),
    )
    original = {event["id"]: event for event in base}
    suppressed: set[str] = set()
    injections: list[dict[str, Any]] = []
    for ordinal, operation in enumerate(operations):
        kind = operation["kind"]
        if kind != "inject":
            target = operation["event"]
            if target not in original or original[target]["origin"]["kind"] == "derived":
                fail("invalid_operation", "operation target is not an active base/player event", event=target)
            suppressed.add(target)
        if kind == "suppress":
            continue
        if kind == "inject":
            body = {
                "topic": operation["topic"],
                "at": operation["at"],
                "payload": operation["payload"],
                "parents": sorted(operation["parents"], key=parse_id),
                "origin": {"kind": "player", "branch_parent": parent, "ordinal": ordinal, "supersedes": None},
            }
        else:
            old = original[operation["event"]]
            body = {key: copy.deepcopy(old[key]) for key in ("topic", "at", "payload", "parents", "origin")}
            body["origin"] = {
                "kind": "player",
                "branch_parent": parent,
                "ordinal": ordinal,
                "supersedes": old["id"],
            }
            if kind == "retime":
                body["at"] = checked_i64(operation["at"], "retime.at")
            else:
                if not isinstance(operation["pointer"], str) or not operation["pointer"].startswith("/payload"):
                    fail("invalid_operation", "replace pointer must target /payload in 0.1")
                body = pointer_replace(body, operation["pointer"], operation["value"])
        injections.append(make_event(body))

    changed = True
    while changed:
        changed = False
        for event in base:
            if event["id"] not in suppressed and any(parent_id in suppressed for parent_id in event["parents"]):
                suppressed.add(event["id"])
                changed = True
    active = [event for event in base if event["id"] not in suppressed] + injections
    return active, branch, labels


def trace_digest(trace: list[dict[str, Any]]) -> str:
    return digest_id("afterimage/trace/1", canonical_bytes(trace))


def resolve_fixture(case: dict[str, Any]) -> tuple[list[Any], list[Any] | None]:
    labels: dict[str, str] = {}
    bodies: list[Any] = []
    pending = list(case["base_events"])
    while pending:
        progress = False
        remaining = []
        for entry in pending:
            if isinstance(entry, dict) and "body" in entry:
                label = entry.get("label")
                body = copy.deepcopy(entry["body"])
            else:
                label = None
                body = copy.deepcopy(entry)
            parent_values = body.get("parents", [])
            if any(isinstance(parent, str) and parent.startswith("@") and parent[1:] not in labels for parent in parent_values):
                remaining.append(entry)
                continue
            body["parents"] = [labels[parent[1:]] if isinstance(parent, str) and parent.startswith("@") else parent for parent in parent_values]
            event = load_event(body)
            bodies.append(event_view(event))
            if label is not None:
                if not isinstance(label, str) or label in labels:
                    fail("invalid_fixture", "fixture labels must be unique Text")
                labels[label] = event["id"]
            progress = True
        if not progress:
            fail("invalid_fixture", "fixture parent labels contain a cycle or missing name")
        pending = remaining
    operations = copy.deepcopy(case.get("operations"))
    if operations:
        for operation in operations:
            if isinstance(operation.get("event"), str) and operation["event"].startswith("@"):
                name = operation["event"][1:]
                if name not in labels:
                    fail("invalid_fixture", "unknown operation label", label=name)
                operation["event"] = labels[name]
            if "parents" in operation:
                operation["parents"] = [
                    labels[parent[1:]] if isinstance(parent, str) and parent.startswith("@") else parent
                    for parent in operation["parents"]
                ]
    return bodies, operations


def run_case(case_value: Any, oracle_limit: int | None = None) -> dict[str, Any]:
    case = normalize_value(case_value)
    required = {"name", "bundle_digest", "program", "base_events"}
    if not isinstance(case, dict) or not required.issubset(case):
        fail("invalid_fixture", "case requires name, bundle_digest, program, and base_events")
    base, operations = resolve_fixture(case)
    limits = dict(case.get("limits", {}))
    if oracle_limit is not None:
        current = limits.get("max_derived_events", DEFAULT_LIMITS["max_derived_events"])
        limits["max_derived_events"] = min(current, oracle_limit)
    branched, branch, _ = apply_branch(base, case["bundle_digest"], operations, case.get("parent_branch"))
    events, trace, counters = evaluate_world(case["program"], branched, limits)
    projection_output: list[Any] | None = None
    projection_id: str | None = None
    if "projection" in case:
        projection_output, projection_id = evaluate_projection(case["projection"], events, counters)
    return {
        "name": case["name"],
        "branch": branch,
        "event_ids": sorted(events, key=parse_id),
        "projection": projection_output,
        "projection_digest": projection_id,
        "trace_digest": trace_digest(trace),
        "counters": counters.as_value(),
    }


def run_suite(suite_value: Any, oracle_limit: int | None = None) -> dict[str, Any]:
    suite = normalize_value(suite_value)
    if not isinstance(suite, dict) or set(suite) != {"format", "canonical_vectors", "cases"}:
        fail("invalid_fixture", "suite has wrong fields")
    vectors = []
    for vector in suite["canonical_vectors"]:
        value = vector["value"]
        domain = vector.get("domain", "afterimage/test-vector/1")
        encoded = canonical_bytes(value)
        vectors.append(
            {
                "name": vector["name"],
                "canonical": encoded.decode("utf-8"),
                "digest": digest_id(domain, encoded),
            }
        )
    cases = []
    for case in suite["cases"]:
        try:
            cases.append({"ok": run_case(case, oracle_limit)})
        except CREError as exc:
            cases.append({"error": exc.as_value(), "name": case.get("name")})
    return {"format": "afterimage-conformance-result/0.1", "canonical_vectors": vectors, "cases": cases}


def protocol_records(case_value: Any) -> list[dict[str, Any]]:
    case = normalize_value(case_value)
    result = run_case(case)
    records: list[dict[str, Any]] = [
        {
            "type": "ready",
            "semantics": "cre/0.1",
            "bundle": case["bundle_digest"],
        }
    ]
    for index, value in enumerate(result["projection"] or []):
        records.append({"type": "projection", "index": index, "value": value})
    records.append(
        {
            "type": "done",
            "branch": result["branch"],
            "projection": result["projection_digest"],
            "trace": result["trace_digest"],
            "counters": result["counters"],
        }
    )
    return records


def protocol_error(error: CREError) -> dict[str, Any]:
    return {"type": "error", **error.as_value()}


def object_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CREError("duplicate_key", "duplicate JSON object key", {"key": key})
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs_no_duplicates)
    except CREError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CREError("invalid_json", f"cannot load {path}: {exc}") from exc
    return normalize_value(raw)


def output_json(value: Any, pretty: bool = False) -> str:
    tagged = json_value(normalize_value(value))
    if pretty:
        return json.dumps(tagged, ensure_ascii=False, indent=2, sort_keys=True)
    return canonical_text(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CRE 0.1 Python reference evaluator")
    parser.add_argument("command", choices=("case", "suite", "protocol"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--oracle-limit", type=int)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        value = load_json(args.input)
    except CREError as exc:
        record = protocol_error(exc) if args.command == "protocol" else {"error": exc.as_value()}
        print(output_json(record, args.pretty if args.command != "protocol" else False))
        return 2
    try:
        if args.command == "protocol":
            for record in protocol_records(value):
                print(output_json(record, False))
        else:
            result = run_case(value, args.oracle_limit) if args.command == "case" else run_suite(value, args.oracle_limit)
            print(output_json(result, args.pretty))
        return 0
    except CREError as exc:
        record = protocol_error(exc) if args.command == "protocol" else {"error": exc.as_value()}
        print(output_json(record, args.pretty if args.command != "protocol" else False))
        return 3 if exc.code != "resource_exhausted" else 4


if __name__ == "__main__":
    raise SystemExit(main())
