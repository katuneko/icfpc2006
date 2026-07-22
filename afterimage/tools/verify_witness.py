#!/usr/bin/env python3
"""Authoritative offline witness verifier for Afterimage 0.1."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import cre  # noqa: E402
import pulse as pulse_runtime  # noqa: E402
import mosaic as mosaic_runtime  # noqa: E402
import lens as lens_runtime  # noqa: E402
import covenant as covenant_runtime  # noqa: E402
import paradox as paradox_runtime  # noqa: E402


MAX_WITNESS_BYTES = 1024 * 1024
CASE_KEYS = {
    "id",
    "family",
    "title",
    "points",
    "requires",
    "input_branch",
    "world",
    "projection",
    "answer_schema",
    "validator",
    "intervention_policy",
    "score",
    "limits",
}
WITNESS_REQUIRED = {
    "format",
    "semantics",
    "bundle",
    "case",
    "parent_branch",
    "intervention",
    "answer",
}
WITNESS_OPTIONAL = {"claimed", "meta", "history"}
DIAGNOSTIC_KEYS = {"code", "message", "context"}
MAX_HISTORY_STEPS = 32


class VerificationError(Exception):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}

    def diagnostic(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


def fail(code: str, message: str, **context: Any) -> None:
    raise VerificationError(code, message, context)


def require_map(value: Any, keys: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        fail("invalid_schema", "object has wrong fields", path=location, expected=sorted(keys))
    return value


def checked_nonnegative(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail("invalid_schema", "value must be a non-negative Int", path=location)
    return value


def validate_case_descriptor(value: Any, world: kit.ValidatedBundle) -> dict[str, Any]:
    case = require_map(value, CASE_KEYS, "cases/index.json#/case")
    for key in ("id", "family", "title", "world", "projection", "answer_schema", "validator", "intervention_policy", "score"):
        if not isinstance(case[key], str) or not case[key]:
            fail("invalid_case", "case Text field must be non-empty", field=key)
    if case["family"] not in {"ORIENT", "CASCADE", "MERGE", "PULSE", "MOSAIC", "LENS", "COVENANT", "PARADOX"}:
        fail("invalid_case", "unknown case family", family=case["family"])
    if isinstance(case["points"], bool) or not isinstance(case["points"], int) or case["points"] <= 0:
        fail("invalid_case", "case points must be positive")
    if case["input_branch"] != "root":
        if not case["input_branch"].startswith("history:") or not case["input_branch"][len("history:"):]:
            fail("invalid_case", "input_branch must be root or history:<case-id>")
    if not isinstance(case["requires"], dict) or not isinstance(case["limits"], dict):
        fail("invalid_case", "case requires and limits must be maps")
    try:
        logical = kit.resolve_logical_world(world, case["world"])
    except kit.KitError as exc:
        raise VerificationError("invalid_case", exc.message, {"field": "world", **exc.context}) from exc
    projection_ids = {entry["id"] for entry in logical.projection_index["projections"]}
    if case["projection"] not in projection_ids:
        fail("invalid_case", "case references an unknown projection", projection=case["projection"])
    for key in ("answer_schema", "validator", "intervention_policy", "score"):
        path = kit.canonical_path(case[key])
        if path not in world.json_values:
            fail("invalid_case", "case references a missing canonical JSON file", field=key, path=path)
    allowed_limits = {"max_witness_bytes", "replay", "validator"}
    if set(case["limits"]) - allowed_limits:
        fail("invalid_case", "case limits contain unknown keys")
    if "max_witness_bytes" in case["limits"]:
        limit = checked_nonnegative(case["limits"]["max_witness_bytes"], "case.limits.max_witness_bytes")
        if limit <= 0 or limit > MAX_WITNESS_BYTES:
            fail("invalid_case", "max_witness_bytes is outside verifier bounds")
    for key in ("replay", "validator"):
        if key in case["limits"]:
            try:
                cre.Counters.create(case["limits"][key])
            except cre.CREError as exc:
                raise VerificationError("invalid_case", exc.message, {"field": f"limits.{key}", **exc.context}) from exc
    return case


def requirements_hold(expr: Any, facts: set[str]) -> bool:
    if not isinstance(expr, dict) or not expr or set(expr) - {"all", "any", "at_least"}:
        fail("invalid_case", "unlock expression is invalid")
    if "all" in expr:
        if not isinstance(expr["all"], list) or not all(isinstance(item, str) for item in expr["all"]):
            fail("invalid_case", "unlock all must be a Text list")
        if not all(item in facts for item in expr["all"]):
            return False
    if "any" in expr:
        if not isinstance(expr["any"], list) or not expr["any"] or not all(isinstance(item, str) for item in expr["any"]):
            fail("invalid_case", "unlock any must be a non-empty Text list")
        if not any(item in facts for item in expr["any"]):
            return False
    if "at_least" in expr:
        spec = require_map(expr["at_least"], {"count", "of"}, "case.requires.at_least")
        count = checked_nonnegative(spec["count"], "case.requires.at_least.count")
        if count <= 0 or not isinstance(spec["of"], list) or len(set(spec["of"])) != len(spec["of"]):
            fail("invalid_case", "unlock at_least is invalid")
        if not all(isinstance(item, str) for item in spec["of"]) or count > len(spec["of"]):
            fail("invalid_case", "unlock at_least options are invalid")
        if sum(item in facts for item in spec["of"]) < count:
            return False
    return True


def schema_error(path: str, message: str, **context: Any) -> VerificationError:
    return VerificationError("answer_schema", message, {"path": path, **context})


def validate_schema_node(schema_value: Any, value: Any, path: str = "") -> None:
    if not isinstance(schema_value, dict) or not isinstance(schema_value.get("type"), str):
        fail("invalid_answer_schema", "schema node requires a type", path=path)
    schema = schema_value
    kind = schema["type"]
    common = {"type", "const", "enum"}
    allowed_by_type = {
        "null": common,
        "bool": common,
        "int": common | {"minimum", "maximum"},
        "text": common | {"min_length", "max_length"},
        "bytes": common | {"min_length", "max_length"},
        "id": common,
        "list": common | {"items", "min_items", "max_items"},
        "map": common | {"required", "properties", "additional", "min_properties", "max_properties"},
    }
    if kind not in allowed_by_type or set(schema) - allowed_by_type[kind]:
        fail("invalid_answer_schema", "schema node has unknown type or fields", path=path, type=kind)
    if "const" in schema and not cre.same_value(value, schema["const"]):
        raise schema_error(path, "answer does not equal required constant")
    if "enum" in schema:
        if not isinstance(schema["enum"], list) or not schema["enum"]:
            fail("invalid_answer_schema", "enum must be a non-empty list", path=path)
        if not any(cre.same_value(value, choice) for choice in schema["enum"]):
            raise schema_error(path, "answer value is outside enum")

    type_ok = {
        "null": value is None,
        "bool": isinstance(value, bool),
        "int": isinstance(value, int) and not isinstance(value, bool),
        "text": isinstance(value, str),
        "bytes": isinstance(value, bytes),
        "id": isinstance(value, str),
        "list": isinstance(value, list),
        "map": isinstance(value, dict),
    }[kind]
    if not type_ok:
        raise schema_error(path, f"answer value must have type {kind}")
    if kind == "id":
        try:
            cre.parse_id(value, "answer ID")
        except cre.CREError as exc:
            raise schema_error(path, exc.message) from exc
    if kind == "int":
        for key, relation in (("minimum", lambda a, b: a >= b), ("maximum", lambda a, b: a <= b)):
            if key in schema:
                bound = schema[key]
                if isinstance(bound, bool) or not isinstance(bound, int):
                    fail("invalid_answer_schema", f"{key} must be Int", path=path)
                if not relation(value, bound):
                    raise schema_error(path, f"answer integer violates {key}", bound=bound)
    if kind in {"text", "bytes"}:
        length = len(value)
        for key, relation in (("min_length", lambda a, b: a >= b), ("max_length", lambda a, b: a <= b)):
            if key in schema:
                bound = checked_nonnegative(schema[key], f"schema{path}.{key}")
                if not relation(length, bound):
                    raise schema_error(path, f"answer length violates {key}", bound=bound)
    if kind == "list":
        if "items" not in schema:
            fail("invalid_answer_schema", "list schema requires items", path=path)
        for key, relation in (("min_items", lambda a, b: a >= b), ("max_items", lambda a, b: a <= b)):
            if key in schema:
                bound = checked_nonnegative(schema[key], f"schema{path}.{key}")
                if not relation(len(value), bound):
                    raise schema_error(path, f"answer list violates {key}", bound=bound)
        for index, item in enumerate(value):
            validate_schema_node(schema["items"], item, f"{path}/{index}")
    if kind == "map":
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        additional = schema.get("additional", False)
        if not isinstance(required, list) or len(set(required)) != len(required) or not all(isinstance(item, str) for item in required):
            fail("invalid_answer_schema", "map required must be a unique Text list", path=path)
        if not isinstance(properties, dict) or not isinstance(additional, bool):
            fail("invalid_answer_schema", "map properties/additional are invalid", path=path)
        missing = [key for key in required if key not in value]
        if missing:
            raise schema_error(path, "answer map lacks required properties", missing=missing)
        unknown = sorted(set(value) - set(properties))
        if unknown and not additional:
            raise schema_error(path, "answer map has additional properties", additional=unknown)
        for key, item in value.items():
            if key in properties:
                escaped = key.replace("~", "~0").replace("/", "~1")
                validate_schema_node(properties[key], item, f"{path}/{escaped}")
        for key, relation in (("min_properties", lambda a, b: a >= b), ("max_properties", lambda a, b: a <= b)):
            if key in schema:
                bound = checked_nonnegative(schema[key], f"schema{path}.{key}")
                if not relation(len(value), bound):
                    raise schema_error(path, f"answer map violates {key}", bound=bound)


def validate_answer_schema(document: Any, answer: Any) -> dict[str, Any] | None:
    if not isinstance(document, dict) or set(document) not in ({"format", "schema"}, {"format", "schema", "embedded_witness"}):
        fail("invalid_answer_schema", "answer schema document has wrong fields")
    schema_document = document
    if schema_document["format"] != "afterimage-answer-schema/0.1":
        fail("invalid_answer_schema", "answer schema format is unsupported")
    validate_schema_node(schema_document["schema"], answer)
    embedded = schema_document.get("embedded_witness")
    if embedded is None:
        return None
    embedded = require_map(embedded, {"pointer", "allowed_cases", "require_fact"}, "answer schema embedded_witness")
    if not isinstance(embedded["pointer"], str) or not embedded["pointer"].startswith("/"):
        fail("invalid_answer_schema", "embedded witness pointer must be a non-root JSON Pointer")
    if (
        not isinstance(embedded["allowed_cases"], list)
        or not embedded["allowed_cases"]
        or len(set(embedded["allowed_cases"])) != len(embedded["allowed_cases"])
        or not all(isinstance(item, str) and item for item in embedded["allowed_cases"])
        or not isinstance(embedded["require_fact"], bool)
    ):
        fail("invalid_answer_schema", "embedded witness policy is invalid")
    return embedded


@dataclass
class PolicyResult:
    operations: list[dict[str, Any]]
    weight: int


def validate_intervention(
    policy_value: Any,
    intervention: Any,
    *,
    witness_bundle: str,
    witness_case: str,
    parent_branch: str,
    base_events: list[dict[str, Any]],
    known_events: dict[str, dict[str, Any]] | None = None,
) -> PolicyResult:
    policy = require_map(
        policy_value,
        {"format", "required", "allowed_kinds", "max_operations", "weights", "topics", "pointers", "retime"},
        "intervention policy",
    )
    if policy["format"] != "afterimage-intervention-policy/0.1":
        fail("invalid_policy", "intervention policy format is unsupported")
    if not isinstance(policy["required"], bool):
        fail("invalid_policy", "policy required must be Bool")
    if not isinstance(policy["allowed_kinds"], list) or len(set(policy["allowed_kinds"])) != len(policy["allowed_kinds"]):
        fail("invalid_policy", "allowed_kinds must be a unique list")
    known_kinds = {"suppress", "replace", "retime", "inject"}
    if any(kind not in known_kinds for kind in policy["allowed_kinds"]):
        fail("invalid_policy", "allowed_kinds contains an unknown operation")
    max_operations = checked_nonnegative(policy["max_operations"], "policy.max_operations")
    if not isinstance(policy["weights"], dict) or set(policy["weights"]) != set(policy["allowed_kinds"]):
        fail("invalid_policy", "weights must exactly cover allowed_kinds")
    weights = {kind: checked_nonnegative(value, f"policy.weights.{kind}") for kind, value in policy["weights"].items()}
    if not isinstance(policy["topics"], list) or len(set(policy["topics"])) != len(policy["topics"]) or not all(isinstance(item, str) for item in policy["topics"]):
        fail("invalid_policy", "policy topics must be a unique Text list")
    if not isinstance(policy["pointers"], list) or len(set(policy["pointers"])) != len(policy["pointers"]) or not all(isinstance(item, str) for item in policy["pointers"]):
        fail("invalid_policy", "policy pointers must be a unique Text list")
    retime = require_map(policy["retime"], {"minimum", "maximum"}, "policy.retime")
    minimum = cre.checked_i64(retime["minimum"], "policy.retime.minimum")
    maximum = cre.checked_i64(retime["maximum"], "policy.retime.maximum")
    if minimum > maximum:
        fail("invalid_policy", "retime minimum exceeds maximum")

    if intervention is None:
        if policy["required"]:
            raise VerificationError("intervention_required", "case requires an intervention envelope")
        operations: list[dict[str, Any]] = []
    else:
        if not policy["required"] and max_operations == 0:
            raise VerificationError("unexpected_intervention", "non-branch case requires intervention to be null")
        envelope = require_map(
            intervention,
            {"format", "bundle", "parent_branch", "case", "operations"},
            "witness.intervention",
        )
        if envelope["format"] != "afterimage-intervention/0.1":
            raise VerificationError("invalid_intervention", "intervention format is unsupported")
        if envelope["bundle"] != witness_bundle or envelope["case"] != witness_case or envelope["parent_branch"] != parent_branch:
            raise VerificationError("invalid_intervention", "intervention envelope does not match witness")
        if not isinstance(envelope["operations"], list):
            raise VerificationError("invalid_intervention", "intervention operations must be a list")
        try:
            operations = cre.canonical_operations(envelope["operations"])
        except cre.CREError as exc:
            raise VerificationError(exc.code, exc.message, exc.context) from exc
    if len(operations) > max_operations:
        raise VerificationError("policy_violation", "operation count exceeds policy")
    if policy["required"] and not operations:
        raise VerificationError("intervention_required", "case requires at least one intervention operation")
    topics = set(policy["topics"])
    pointers = set(policy["pointers"])
    base_by_id = {event["id"]: event for event in base_events}
    total_weight = 0
    for operation in operations:
        kind = operation["kind"]
        if kind not in policy["allowed_kinds"]:
            raise VerificationError("policy_violation", "operation kind is forbidden", {"kind": kind})
        total_weight += weights[kind]
        if kind == "inject":
            if operation["topic"] not in topics:
                raise VerificationError("policy_violation", "injected topic is forbidden", {"topic": operation["topic"]})
        else:
            target = base_by_id.get(operation["event"])
            if target is None:
                known = (known_events or {}).get(operation["event"])
                if known is not None and known["origin"]["kind"] == "derived":
                    base_ids = set(base_by_id)
                    ancestors: set[str] = set()
                    pending = list(known["parents"])
                    visited: set[str] = set()
                    while pending:
                        event_id = pending.pop()
                        if event_id in visited:
                            continue
                        visited.add(event_id)
                        if event_id in base_ids:
                            ancestors.add(event_id)
                            continue
                        ancestor = (known_events or {}).get(event_id)
                        if ancestor is not None:
                            pending.extend(ancestor["parents"])
                    raise VerificationError(
                        "derived_event_not_intervenable",
                        "derived events must be changed through active base ancestors",
                        {
                            "event": operation["event"],
                            "base_ancestors": sorted(ancestors, key=cre.parse_id),
                        },
                    )
                raise VerificationError("policy_violation", "operation target is not a base event")
            if target["topic"] not in topics:
                raise VerificationError("policy_violation", "target topic is forbidden", {"topic": target["topic"]})
        if kind == "replace" and operation["pointer"] not in pointers:
            raise VerificationError("policy_violation", "replacement pointer is forbidden", {"pointer": operation["pointer"]})
        if kind == "retime" and not minimum <= operation["at"] <= maximum:
            raise VerificationError("policy_violation", "retime value is outside policy bounds")
    return PolicyResult(operations=operations, weight=total_weight)


@dataclass
class ReplayState:
    branch: str
    projection: str
    trace: str
    records: list[Any]
    counters: dict[str, int]
    events: dict[str, dict[str, Any]]
    trace_items: list[dict[str, Any]]


@dataclass
class ReplayResult:
    branch: str
    projection: str
    trace: str
    records: list[Any]
    counters: dict[str, int]
    events: dict[str, dict[str, Any]]
    trace_items: list[dict[str, Any]]
    baseline: ReplayState
    changed_event_ids: list[str]


def evaluate_case_state(
    world: kit.ValidatedBundle,
    case: dict[str, Any],
    base_events: list[dict[str, Any]],
    branch: str,
) -> ReplayState:
    logical = kit.resolve_logical_world(world, case["world"])
    projection_entry = next(
        entry for entry in logical.projection_index["projections"]
        if entry["id"] == case["projection"]
    )
    try:
        events, trace, counters = cre.evaluate_world(
            logical.program,
            base_events,
            case["limits"].get("replay"),
        )
        records, projection_digest = cre.evaluate_projection(
            world.json_values[projection_entry["path"]],
            events,
            counters,
        )
    except cre.CREError as exc:
        raise VerificationError(exc.code, exc.message, exc.context) from exc
    return ReplayState(
        branch=branch,
        projection=projection_digest,
        trace=cre.trace_digest(trace),
        records=records,
        counters=counters.as_value(),
        events=events,
        trace_items=trace,
    )


def replay_root_case(world: kit.ValidatedBundle, case: dict[str, Any]) -> ReplayState:
    logical = kit.resolve_logical_world(world, case["world"])
    root = cre.root_branch_id(world.bundle)
    try:
        base_events, branch, _ = cre.apply_branch(logical.base_events, world.bundle, [], root)
    except cre.CREError as exc:
        raise VerificationError(exc.code, exc.message, exc.context) from exc
    return evaluate_case_state(world, case, base_events, branch)


def replay_case(
    world: kit.ValidatedBundle,
    case: dict[str, Any],
    operations: list[dict[str, Any]],
    parent_branch: str,
    baseline: ReplayState | None = None,
    base_events: list[dict[str, Any]] | None = None,
) -> ReplayResult:
    logical = kit.resolve_logical_world(world, case["world"])
    replay_base = base_events if base_events is not None else [cre.load_event(value) for value in logical.base_events]
    baseline = baseline or evaluate_case_state(world, case, replay_base, parent_branch)
    try:
        branched, branch, _ = cre.apply_branch(
            replay_base,
            world.bundle,
            operations,
            parent_branch,
        )
    except cre.CREError as exc:
        raise VerificationError(exc.code, exc.message, exc.context) from exc
    if not operations and branch == baseline.branch:
        candidate = baseline
    else:
        candidate = evaluate_case_state(world, case, branched, branch)
    changed = sorted(set(baseline.events) ^ set(candidate.events), key=cre.parse_id)
    return ReplayResult(
        branch=candidate.branch,
        projection=candidate.projection,
        trace=candidate.trace,
        records=candidate.records,
        counters=candidate.counters,
        events=candidate.events,
        trace_items=candidate.trace_items,
        baseline=baseline,
        changed_event_ids=changed,
    )


@dataclass
class BranchHistoryResult:
    base_events: list[dict[str, Any]]
    branch: str
    steps: list[dict[str, Any]]


def resolve_input_history(
    world: kit.ValidatedBundle,
    case: dict[str, Any],
    history_value: Any,
    facts: set[str],
) -> BranchHistoryResult:
    logical = kit.resolve_logical_world(world, case["world"])
    base_events = [cre.load_event(value) for value in logical.base_events]
    root = cre.root_branch_id(world.bundle)
    if case["input_branch"] == "root":
        if history_value is not None:
            raise VerificationError("unexpected_history", "root-input case forbids a branch history")
        return BranchHistoryResult(base_events=base_events, branch=root, steps=[])

    if history_value is None:
        raise VerificationError("history_required", "non-root case requires a complete branch history")
    history = require_map(history_value, {"format", "bundle", "world", "steps"}, "witness.history")
    if history["format"] != "afterimage-branch-history/0.1":
        raise VerificationError("invalid_history", "branch history format is unsupported")
    if history["bundle"] != world.bundle or history["world"] != case["world"]:
        raise VerificationError("history_mismatch", "branch history does not match the witness bundle and world")
    steps_value = history["steps"]
    if not isinstance(steps_value, list) or not steps_value or len(steps_value) > MAX_HISTORY_STEPS:
        raise VerificationError("invalid_history", "branch history must contain 1 through 32 steps")

    case_values = world.json_values["cases/index.json"]["cases"]
    case_map = {item.get("id"): item for item in case_values if isinstance(item, dict)}
    expected_origin = case["input_branch"][len("history:"):]
    parent = root
    previous_case: str | None = None
    steps: list[dict[str, Any]] = []
    for index, step_value in enumerate(steps_value):
        step = require_map(step_value, {"case", "operations"}, f"witness.history.steps[{index}]")
        origin_id = step["case"]
        if not isinstance(origin_id, str) or origin_id not in case_map:
            raise VerificationError("invalid_history", "history step names an unknown case", {"index": index})
        if f"case:{origin_id}" not in facts:
            raise VerificationError("history_case_locked", "history step case has not been unlocked", {"case": origin_id})
        origin = validate_case_descriptor(case_map[origin_id], world)
        if origin["world"] != case["world"]:
            raise VerificationError("history_world_mismatch", "all history steps must use the target case world", {"case": origin_id})
        required_input = "root" if previous_case is None else f"history:{previous_case}"
        if origin["input_branch"] != required_input:
            raise VerificationError(
                "history_chain_mismatch",
                "history cases do not form the declared input-branch chain",
                {"case": origin_id, "index": index},
            )
        if not isinstance(step["operations"], list) or not step["operations"]:
            raise VerificationError("invalid_history", "history step operations must be a non-empty list", {"index": index})

        baseline = evaluate_case_state(world, origin, base_events, parent)
        policy_document = world.json_values[origin["intervention_policy"]]
        intervention = {
            "format": "afterimage-intervention/0.1",
            "bundle": world.bundle,
            "parent_branch": parent,
            "case": origin_id,
            "operations": step["operations"],
        }
        policy = validate_intervention(
            policy_document,
            intervention,
            witness_bundle=world.bundle,
            witness_case=origin_id,
            parent_branch=parent,
            base_events=base_events,
            known_events=baseline.events,
        )
        try:
            base_events, parent, _ = cre.apply_branch(base_events, world.bundle, policy.operations, parent)
        except cre.CREError as exc:
            raise VerificationError(exc.code, exc.message, exc.context) from exc
        steps.append({"case": origin_id, "operations": policy.operations})
        previous_case = origin_id

    if previous_case != expected_origin:
        raise VerificationError(
            "history_origin_mismatch",
            "branch history ends at the wrong case",
            {"expected_case": expected_origin},
        )
    return BranchHistoryResult(base_events=base_events, branch=parent, steps=steps)


def validation_event(topic: str, payload: Any, sequence: int) -> dict[str, Any]:
    return cre.make_event(
        {
            "topic": topic,
            "at": 0,
            "payload": payload,
            "parents": [],
            "origin": {"kind": "base", "source": "afterimage-verifier/0.1", "sequence": sequence},
        }
    )


def run_validator(
    validator_value: Any,
    *,
    answer: Any,
    intervention: Any,
    replay: ReplayResult,
    limits: dict[str, Any] | None,
    embedded_receipt: dict[str, Any] | None = None,
    family_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validator = require_map(validator_value, {"format", "program", "decision_projection"}, "validator")
    if validator["format"] != "afterimage-validator/0.1":
        fail("invalid_validator", "validator format is unsupported")
    replay_payload = {
        "branch": replay.branch,
        "projection": replay.projection,
        "trace": replay.trace,
        "records": replay.records,
        "active_event_count": len(replay.events),
        "active_event_ids": sorted(replay.events, key=cre.parse_id),
        "counters": replay.counters,
        "trace_items": replay.trace_items,
        "trace_event_ids": [item["event"] for item in replay.trace_items],
        "baseline_branch": replay.baseline.branch,
        "baseline_projection": replay.baseline.projection,
        "baseline_trace": replay.baseline.trace,
        "baseline_records": replay.baseline.records,
        "baseline_active_event_ids": sorted(replay.baseline.events, key=cre.parse_id),
        "changed_event_ids": replay.changed_event_ids,
    }
    validation_base = [
        validation_event("verify.answer", answer, 0),
        validation_event("verify.replay", replay_payload, 1),
        validation_event("verify.intervention", intervention, 2),
    ]
    for sequence, event_id in enumerate(sorted(replay.events, key=cre.parse_id), 3):
        validation_base.append(validation_event("verify.active", cre.event_view(replay.events[event_id]), sequence))
    offset = 3 + len(replay.events)
    for sequence, event_id in enumerate(sorted(replay.baseline.events, key=cre.parse_id), offset):
        validation_base.append(validation_event("verify.baseline-active", cre.event_view(replay.baseline.events[event_id]), sequence))
    if embedded_receipt is not None:
        validation_base.append(validation_event("verify.embedded", embedded_receipt, offset + len(replay.baseline.events)))
    if family_result is not None:
        validation_base.append(validation_event("verify.family", family_result, offset + len(replay.baseline.events) + 1))
    try:
        events, _trace, counters = cre.evaluate_world(validator["program"], validation_base, limits)
        decisions, _digest = cre.evaluate_projection(validator["decision_projection"], events, counters)
    except cre.CREError as exc:
        raise VerificationError("invalid_validator", exc.message, exc.context) from exc
    if len(decisions) != 1 or not isinstance(decisions[0], dict) or set(decisions[0]) != {"valid", "diagnostics", "metrics"}:
        fail("invalid_validator", "decision projection must produce exactly one decision map")
    decision = decisions[0]
    if not isinstance(decision["valid"], bool) or not isinstance(decision["diagnostics"], list) or not isinstance(decision["metrics"], dict):
        fail("invalid_validator", "decision fields have invalid types")
    diagnostics = []
    for item in decision["diagnostics"]:
        diagnostic = require_map(item, DIAGNOSTIC_KEYS, "validator diagnostic")
        if not isinstance(diagnostic["code"], str) or not isinstance(diagnostic["message"], str) or not isinstance(diagnostic["context"], dict):
            fail("invalid_validator", "validator diagnostic fields have invalid types")
        diagnostics.append(diagnostic)
    if decision["valid"] and diagnostics:
        fail("invalid_validator", "valid decision must not contain diagnostics")
    if not decision["valid"] and not diagnostics:
        fail("invalid_validator", "invalid decision must contain a diagnostic")
    return decision


def validate_claimed(claimed: Any, replay: ReplayResult) -> None:
    if claimed is None:
        return
    if not isinstance(claimed, dict) or set(claimed) - {"branch", "projection", "trace"}:
        raise VerificationError("invalid_claimed", "claimed must contain only branch/projection/trace")
    actual = {"branch": replay.branch, "projection": replay.projection, "trace": replay.trace}
    for key, value in claimed.items():
        try:
            cre.parse_id(value, f"claimed.{key}")
        except cre.CREError as exc:
            raise VerificationError("invalid_claimed", exc.message, {"field": key}) from exc
        if value != actual[key]:
            raise VerificationError("claimed_mismatch", "claimed digest does not match replay", {"field": key})


def orient_score(case: dict[str, Any], score_value: Any, answer: Any, validator_metrics: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    if not isinstance(score_value, dict) or set(score_value) not in (
        {"format", "family", "reference_scale", "metric_bounds"},
        {"format", "family", "reference_scale", "metric_bounds", "grants"},
    ):
        fail("invalid_score", "ORIENT score descriptor has wrong fields")
    score = score_value
    if score["format"] != "afterimage-score/0.1" or score["family"] != "ORIENT" or case["family"] != "ORIENT":
        fail("invalid_score", "ORIENT score descriptor is invalid")
    scale = score["reference_scale"]
    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        fail("invalid_score", "reference_scale must be positive")
    bounds = require_map(score["metric_bounds"], {"witness_units"}, "score.metric_bounds")
    grants = score.get("grants", [])
    if (
        not isinstance(grants, list)
        or len(set(grants)) != len(grants)
        or not all(isinstance(item, str) and item.startswith("cap:") for item in grants)
    ):
        fail("invalid_score", "score grants must be unique capability facts")
    max_units = checked_nonnegative(bounds["witness_units"], "score.metric_bounds.witness_units")
    if set(validator_metrics) != {"wrong_or_redundant_claims"} or validator_metrics["wrong_or_redundant_claims"] != 0:
        fail("invalid_validator", "valid ORIENT decision must report zero wrong claims")
    units = (len(cre.canonical_bytes(answer)) + 31) // 32
    if units > max_units:
        raise VerificationError("metric_limit", "answer exceeds witness_units bound", {"value": units, "maximum": max_units})
    effective = units + 1
    points = case["points"]
    completion = (65 * points + 99) // 100
    pool = points - completion
    quality_ppm = 1_000_000 * scale // (scale + effective)
    optimization = pool * quality_ppm // 1_000_000
    metrics = {
        "wrong_or_redundant_claims": 0,
        "witness_units": units,
        "effective_cost": effective,
    }
    score_result = {
        "completion": completion,
        "optimization": optimization,
        "total": completion + optimization,
        "nominal_max": points,
    }
    return metrics, score_result


def cascade_score(
    case: dict[str, Any],
    score_value: Any,
    answer: Any,
    validator_metrics: dict[str, Any],
    policy: PolicyResult,
    replay: ReplayResult,
) -> tuple[dict[str, int], dict[str, int]]:
    if not isinstance(score_value, dict) or set(score_value) not in (
        {"format", "family", "reference_scale", "metric_bounds", "diagnostic_topics"},
        {"format", "family", "reference_scale", "metric_bounds", "diagnostic_topics", "proof_contract"},
        {"format", "family", "reference_scale", "metric_bounds", "diagnostic_topics", "projection_contract"},
        {"format", "family", "reference_scale", "metric_bounds", "diagnostic_topics", "minimal_contract"},
    ):
        fail("invalid_score", "CASCADE score descriptor fields are invalid")
    score = score_value
    if score["format"] != "afterimage-score/0.1" or score["family"] != "CASCADE" or case["family"] != "CASCADE":
        fail("invalid_score", "CASCADE score descriptor is invalid")
    scale = score["reference_scale"]
    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        fail("invalid_score", "reference_scale must be positive")
    bounds = require_map(score["metric_bounds"], {"causal_footprint", "witness_units"}, "score.metric_bounds")
    max_footprint = checked_nonnegative(bounds["causal_footprint"], "score.metric_bounds.causal_footprint")
    max_units = checked_nonnegative(bounds["witness_units"], "score.metric_bounds.witness_units")
    topics = score["diagnostic_topics"]
    if not isinstance(topics, list) or len(set(topics)) != len(topics) or not all(isinstance(item, str) for item in topics):
        fail("invalid_score", "diagnostic_topics must be a unique Text list")
    expected_validator_metric = "proof_failures" if "proof_contract" in score else "projection_failures" if "projection_contract" in score else "minimality_failures" if "minimal_contract" in score else "contract_violations"
    if set(validator_metrics) != {expected_validator_metric} or validator_metrics[expected_validator_metric] != 0:
        fail("invalid_validator", f"valid CASCADE decision must report zero {expected_validator_metric.replace('_', ' ')}")
    all_events = {**replay.baseline.events, **replay.events}
    footprint = sum(all_events[event_id]["topic"] not in set(topics) for event_id in replay.changed_event_ids)
    units = (len(cre.canonical_bytes(answer)) + 63) // 64
    if footprint > max_footprint or units > max_units:
        raise VerificationError("metric_limit", "CASCADE metric exceeds public bound", {"causal_footprint": footprint, "witness_units": units})
    effective = ((policy.weight * (max_footprint + 1) + footprint) * (max_units + 1) + units) + 1
    points = case["points"]
    completion = (65 * points + 99) // 100
    pool = points - completion
    quality_ppm = 1_000_000 * scale // (scale + effective)
    optimization = pool * quality_ppm // 1_000_000
    metrics = {
        "intervention_weight": policy.weight,
        "causal_footprint": footprint,
        "witness_units": units,
        "effective_cost": effective,
    }
    return metrics, {
        "completion": completion,
        "optimization": optimization,
        "total": completion + optimization,
        "nominal_max": points,
    }


def validate_cascade_proof(answer: Any, replay: ReplayResult, policy: PolicyResult, proof: Any) -> dict[str, int]:
    proof = require_map(proof, {"relay_topic", "ready_topic", "policy_label", "relay_id", "capability_scope", "reroute_id"}, "score.proof_contract")
    answer = require_map(answer, {"contract", "public_rows", "relay", "branch", "projection"}, "CASCADE proof answer")
    relay_claim = require_map(answer["relay"], {"event", "relay_id", "provenance_digest", "projection_difference", "policy_label"}, "answer.relay")
    relays = [event for event in replay.events.values() if event["topic"] == proof["relay_topic"]]
    ready = [event for event in replay.events.values() if event["topic"] == proof["ready_topic"]]
    if len(relays) != 1 or len(ready) != 1:
        raise VerificationError("cascade_proof", "active route proof is incomplete")
    relay = relays[0]
    expected_digest = cre.digest_id("afterimage/provenance/1", cre.canonical_bytes(relay["payload"]["provenance"]))
    expected_relay = {
        "event": relay["id"], "relay_id": proof["relay_id"], "provenance_digest": expected_digest,
        "projection_difference": "whole_event_suppressed", "policy_label": proof["policy_label"],
    }
    if not cre.same_value(relay_claim, expected_relay):
        raise VerificationError("cascade_proof", "relay provenance certificate is invalid")
    if not cre.same_value(answer["contract"], ready[0]["payload"]) or answer["contract"].get("safe") is not True:
        raise VerificationError("cascade_proof", "evacuation contract is not satisfied")
    if not cre.same_value(answer["public_rows"], replay.records) or answer["branch"] != replay.branch or answer["projection"] != replay.projection:
        raise VerificationError("cascade_proof", "proof does not bind the independent replay")
    unavailable = [row for row in replay.records if isinstance(row, dict) and row.get("edge") == proof["relay_id"]]
    if unavailable != [{"edge": proof["relay_id"], "status": "unavailable", "policy": proof["policy_label"]}]:
        raise VerificationError("cascade_proof", "public projection does not expose the declared suppression")
    for operation in policy.operations:
        if operation["kind"] != "inject":
            raise VerificationError("cascade_proof", "reveal proof permits only scoped injections")
        if operation["topic"] == "audit.capability":
            if operation["payload"] != {"capability": "audit.relay", "scope": proof["capability_scope"]}:
                raise VerificationError("cascade_proof", "audit capability is not scoped to this case")
        elif operation["topic"] == "route.reroute":
            if operation["payload"] != {"route": proof["reroute_id"]}:
                raise VerificationError("cascade_proof", "physical reroute payload is invalid")
    return {"proof_failures": 0}


def validate_cascade_projection(answer: Any, replay: ReplayResult, contract: Any) -> dict[str, int]:
    contract = require_map(contract, {"contract_topic", "require_unchanged"}, "score.projection_contract")
    if not isinstance(contract["contract_topic"], str) or not contract["contract_topic"] or contract["require_unchanged"] is not True:
        raise VerificationError("invalid_score", "projection-sensitive CASCADE contract is invalid")
    answer = require_map(answer, {"contract", "baseline_records", "public_rows", "branch", "projection"}, "CASCADE projection answer")
    summaries = [event for event in replay.events.values() if event["topic"] == contract["contract_topic"]]
    if len(summaries) != 1 or not isinstance(summaries[0].get("payload"), dict) or summaries[0]["payload"].get("safe") is not True:
        raise VerificationError("cascade_projection", "active candidate does not satisfy the hidden-state contract")
    if not cre.same_value(answer["contract"], summaries[0]["payload"]):
        raise VerificationError("cascade_projection", "answer does not bind the active hidden-state contract")
    if not cre.same_value(replay.baseline.records, replay.records) or not cre.same_value(answer["baseline_records"], replay.baseline.records) or not cre.same_value(answer["public_rows"], replay.records):
        raise VerificationError("cascade_projection", "repair does not prove byte-identical public projections")
    if answer["branch"] != replay.branch or answer["projection"] != replay.projection:
        raise VerificationError("cascade_projection", "answer does not bind the independent replay")
    return {"projection_failures": 0}


def validate_cascade_minimal_explanation(
    world: kit.ValidatedBundle,
    case: dict[str, Any],
    answer: Any,
    replay: ReplayResult,
    policy: PolicyResult,
    contract: Any,
) -> dict[str, int]:
    contract = require_map(contract, {"contract_topic", "operation_count"}, "score.minimal_contract")
    topic = contract["contract_topic"]
    count = contract["operation_count"]
    if not isinstance(topic, str) or not topic or isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise VerificationError("invalid_score", "minimal explanation contract is invalid")
    answer = require_map(answer, {"contracts", "branch", "projection"}, "CASCADE minimal explanation answer")
    if len(policy.operations) != count:
        raise VerificationError("cascade_minimality", "candidate does not contain the required explanation size")
    summaries = [event for event in replay.events.values() if event["topic"] == topic]
    safe = [event for event in summaries if isinstance(event.get("payload"), dict) and event["payload"].get("safe") is True]
    if len(safe) != 1:
        raise VerificationError("cascade_minimality", "complete explanation does not establish exactly one safe contract")
    if not cre.same_value(answer["contracts"], replay.records) or answer["branch"] != replay.branch or answer["projection"] != replay.projection:
        raise VerificationError("cascade_minimality", "answer does not bind the independently replayed explanation")
    for subset_size in range(count):
        for indexes in itertools.combinations(range(count), subset_size):
            subset = [policy.operations[index] for index in indexes]
            partial = replay_case(world, case, subset, replay.baseline.branch, replay.baseline)
            if any(
                event["topic"] == topic
                and isinstance(event.get("payload"), dict)
                and event["payload"].get("safe") is True
                for event in partial.events.values()
            ):
                raise VerificationError("cascade_minimality", "a proper subset already establishes safety", {"operation_indexes": list(indexes)})
    return {"minimality_failures": 0}


def validate_merge_answer(
    world: kit.ValidatedBundle,
    case: dict[str, Any],
    score_value: Any,
    answer: Any,
) -> dict[str, int]:
    answer = require_map(answer, {"accepted", "rejected", "certificate"}, "MERGE answer")
    contract = score_value.get("contract") if isinstance(score_value, dict) else None
    contract_task: str | None = None
    if contract is not None:
        if not isinstance(contract, dict) or not isinstance(contract.get("task"), str):
            fail("invalid_score", "MERGE contract is invalid")
        contract_task = contract["task"]
        if contract_task == "deduplicate":
            contract = require_map(contract, {"task", "identity_field", "equivalence_field"}, "score.contract")
            for field in ("identity_field", "equivalence_field"):
                if not isinstance(contract[field], str) or not contract[field]:
                    fail("invalid_score", "MERGE dedup contract field is invalid", field=field)
            if contract["identity_field"] == contract["equivalence_field"]:
                fail("invalid_score", "MERGE dedup contract fields must differ")
        elif contract_task == "split-brain":
            contract = require_map(contract, {"task", "component_field", "writer_field"}, "score.contract")
            for field in ("component_field", "writer_field"):
                if not isinstance(contract[field], str) or not contract[field]:
                    fail("invalid_score", "MERGE split-brain contract field is invalid", field=field)
            if contract["component_field"] == contract["writer_field"]:
                fail("invalid_score", "MERGE split-brain contract fields must differ")
        elif contract_task == "quorum-ledger":
            contract = require_map(contract, {"task", "operation_field", "claim_field", "replica_field", "quorum"}, "score.contract")
            fields = ("operation_field", "claim_field", "replica_field")
            for field in fields:
                if not isinstance(contract[field], str) or not contract[field]:
                    fail("invalid_score", "MERGE quorum contract field is invalid", field=field)
            if len({contract[field] for field in fields}) != len(fields):
                fail("invalid_score", "MERGE quorum contract fields must differ")
            if isinstance(contract["quorum"], bool) or not isinstance(contract["quorum"], int) or contract["quorum"] < 2:
                fail("invalid_score", "MERGE quorum must be at least two")
        elif contract_task == "offset-domains":
            contract = require_map(contract, {"task", "domain_field", "minimum_domains", "minimum_records_per_domain"}, "score.contract")
            if not isinstance(contract["domain_field"], str) or not contract["domain_field"]:
                fail("invalid_score", "MERGE offset-domain field is invalid")
            for field in ("minimum_domains", "minimum_records_per_domain"):
                if isinstance(contract[field], bool) or not isinstance(contract[field], int) or contract[field] < 2:
                    fail("invalid_score", "MERGE offset-domain minimum must be at least two", field=field)
        elif contract_task == "minimal-conflict":
            contract = require_map(contract, {"task", "minimum_size", "maximum_size"}, "score.contract")
            for field in ("minimum_size", "maximum_size"):
                if isinstance(contract[field], bool) or not isinstance(contract[field], int) or contract[field] < 2:
                    fail("invalid_score", "MERGE minimal-conflict bound must be at least two", field=field)
            if contract["minimum_size"] > contract["maximum_size"]:
                fail("invalid_score", "MERGE minimal-conflict bounds are reversed")
        elif contract_task == "weighted-cut":
            contract = require_map(contract, {"task", "conflicts", "maximum_records"}, "score.contract")
            if isinstance(contract["maximum_records"], bool) or not isinstance(contract["maximum_records"], int) or not 2 <= contract["maximum_records"] <= 20:
                fail("invalid_score", "MERGE weighted-cut record bound is invalid")
            if not isinstance(contract["conflicts"], list) or not contract["conflicts"]:
                fail("invalid_score", "MERGE weighted-cut conflicts are empty")
            encodings: set[bytes] = set()
            for index, conflict in enumerate(contract["conflicts"]):
                if not isinstance(conflict, list) or len(conflict) < 2 or len(set(conflict)) != len(conflict) or not all(isinstance(key, str) and key for key in conflict):
                    fail("invalid_score", "MERGE weighted-cut conflict is invalid", index=index)
                if conflict != sorted(conflict, key=lambda key: key.encode("utf-8")):
                    fail("invalid_score", "MERGE weighted-cut conflicts must be canonical", index=index)
                encoded = cre.canonical_bytes(conflict)
                if encoded in encodings:
                    fail("invalid_score", "MERGE weighted-cut conflict is duplicated", index=index)
                encodings.add(encoded)
        elif contract_task == "echo-chain":
            contract = require_map(contract, {"task", "update_field", "echo_field"}, "score.contract")
            for field in ("update_field", "echo_field"):
                if not isinstance(contract[field], str) or not contract[field]:
                    fail("invalid_score", "MERGE echo-chain field is invalid", field=field)
            if contract["update_field"] == contract["echo_field"]:
                fail("invalid_score", "MERGE echo-chain fields must differ")
        elif contract_task == "weighted-evidence":
            contract = require_map(contract, {"task", "operation_field", "claim_field", "source_field", "weight_field", "minimum_margin"}, "score.contract")
            fields = ("operation_field", "claim_field", "source_field", "weight_field")
            if not all(isinstance(contract[field], str) and contract[field] for field in fields) or len({contract[field] for field in fields}) != len(fields):
                fail("invalid_score", "MERGE weighted-evidence fields are invalid")
            if isinstance(contract["minimum_margin"], bool) or not isinstance(contract["minimum_margin"], int) or contract["minimum_margin"] < 1:
                fail("invalid_score", "MERGE weighted-evidence margin is invalid")
        elif contract_task == "causal-compression":
            contract = require_map(contract, {"task", "maximum_certificate_edges"}, "score.contract")
            if isinstance(contract["maximum_certificate_edges"], bool) or not isinstance(contract["maximum_certificate_edges"], int) or contract["maximum_certificate_edges"] < 1:
                fail("invalid_score", "MERGE causal-compression edge bound is invalid")
        elif contract_task == "partial-order-archive":
            contract = require_map(contract, {"task", "maximum_certificate_edges", "minimum_solutions", "maximum_domain", "selection"}, "score.contract")
            for field in ("maximum_certificate_edges", "minimum_solutions", "maximum_domain"):
                if isinstance(contract[field], bool) or not isinstance(contract[field], int) or contract[field] < 1:
                    fail("invalid_score", "MERGE partial-order archive bound is invalid", field=field)
            if contract["minimum_solutions"] < 2 or contract["maximum_domain"] < contract["minimum_solutions"] or contract["selection"] != "lexicographic":
                fail("invalid_score", "MERGE partial-order archive contract is invalid")
        elif contract_task == "equivocation":
            contract = require_map(contract, {"task", "operation_field", "source_field", "claim_field", "minimum_distinct_claims", "maximum_equivocating_sources"}, "score.contract")
            fields = ("operation_field", "source_field", "claim_field")
            if not all(isinstance(contract[field], str) and contract[field] for field in fields) or len({contract[field] for field in fields}) != len(fields):
                fail("invalid_score", "MERGE equivocation fields are invalid")
            for field in ("minimum_distinct_claims", "maximum_equivocating_sources"):
                if isinstance(contract[field], bool) or not isinstance(contract[field], int) or contract[field] < 1:
                    fail("invalid_score", "MERGE equivocation bound is invalid", field=field)
            if contract["minimum_distinct_claims"] < 2:
                fail("invalid_score", "MERGE equivocation needs at least two distinct claims")
        elif contract_task == "non-unique-archives":
            contract = require_map(contract, {"task", "minimum_solutions", "maximum_domain", "selection"}, "score.contract")
            if (
                isinstance(contract["minimum_solutions"], bool)
                or not isinstance(contract["minimum_solutions"], int)
                or contract["minimum_solutions"] < 2
                or isinstance(contract["maximum_domain"], bool)
                or not isinstance(contract["maximum_domain"], int)
                or contract["maximum_domain"] < contract["minimum_solutions"]
                or contract["selection"] != "lexicographic"
            ):
                fail("invalid_score", "MERGE non-unique archive contract is invalid")
        else:
            fail("invalid_score", "MERGE contract task is unsupported")
    logical = kit.resolve_logical_world(world, case["world"])
    records = {event["id"]: event for event in logical.base_events if event["topic"] == "merge.record"}
    if not records:
        fail("invalid_case", "MERGE case has no records")
    keys = [event["payload"].get("key") for event in records.values() if isinstance(event.get("payload"), dict)]
    if len(keys) != len(set(keys)):
        raise VerificationError("invalid_case", "MERGE record keys are duplicated")
    for event_id, event in records.items():
        payload = event.get("payload")
        required = {"key", "local_time", "offset_min", "offset_max", "source", "sequence", "weight"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise VerificationError("invalid_case", "MERGE record payload is incomplete", {"event": event_id})
        if not isinstance(payload["key"], str) or not payload["key"] or not isinstance(payload["source"], str) or not payload["source"]:
            raise VerificationError("invalid_case", "MERGE record identity is invalid", {"event": event_id})
        for field in ("local_time", "offset_min", "offset_max", "sequence", "weight"):
            if isinstance(payload[field], bool) or not isinstance(payload[field], int):
                raise VerificationError("invalid_case", "MERGE record integer field is invalid", {"event": event_id, "field": field})
        if payload["offset_min"] > payload["offset_max"] or payload["sequence"] < 0 or payload["weight"] < 0:
            raise VerificationError("invalid_case", "MERGE record bounds are invalid", {"event": event_id})
        if contract_task == "deduplicate":
            for field in (contract["identity_field"], contract["equivalence_field"]):
                if not isinstance(payload.get(field), str) or not payload[field]:
                    raise VerificationError("invalid_case", "MERGE dedup field is missing or invalid", {"event": event_id, "field": field})
        elif contract_task == "split-brain":
            for field in (contract["component_field"], contract["writer_field"]):
                if not isinstance(payload.get(field), str) or not payload[field]:
                    raise VerificationError("invalid_case", "MERGE split-brain field is missing or invalid", {"event": event_id, "field": field})
        elif contract_task == "quorum-ledger":
            for field in (contract["operation_field"], contract["claim_field"], contract["replica_field"]):
                if not isinstance(payload.get(field), str) or not payload[field]:
                    raise VerificationError("invalid_case", "MERGE quorum field is missing or invalid", {"event": event_id, "field": field})
        elif contract_task == "offset-domains":
            field = contract["domain_field"]
            if not isinstance(payload.get(field), str) or not payload[field]:
                raise VerificationError("invalid_case", "MERGE offset-domain field is missing or invalid", {"event": event_id, "field": field})
        elif contract_task == "echo-chain":
            update = payload.get(contract["update_field"])
            echo = payload.get(contract["echo_field"])
            if not isinstance(update, str) or not update or (echo is not None and (not isinstance(echo, str) or not echo)):
                raise VerificationError("invalid_case", "MERGE echo-chain fields are missing or invalid", {"event": event_id})
        elif contract_task == "weighted-evidence":
            for field in (contract["operation_field"], contract["claim_field"], contract["source_field"]):
                if not isinstance(payload.get(field), str) or not payload[field]:
                    raise VerificationError("invalid_case", "MERGE weighted-evidence text field is missing", {"event": event_id, "field": field})
            weight = payload.get(contract["weight_field"])
            if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                raise VerificationError("invalid_case", "MERGE evidence weight must be positive", {"event": event_id})
        elif contract_task == "equivocation":
            for field in (contract["operation_field"], contract["source_field"], contract["claim_field"]):
                if not isinstance(payload.get(field), str) or not payload[field]:
                    raise VerificationError("invalid_case", "MERGE equivocation field is missing", {"event": event_id, "field": field})
    accepted: dict[str, int] = {}
    for index, item in enumerate(answer["accepted"] if isinstance(answer["accepted"], list) else []):
        item = require_map(item, {"event", "at"}, f"answer.accepted/{index}")
        if not isinstance(item["event"], str) or item["event"] not in records or item["event"] in accepted or isinstance(item["at"], bool) or not isinstance(item["at"], int):
            raise VerificationError("merge_classification", "accepted record entry is invalid", {"index": index})
        accepted[item["event"]] = item["at"]
    rejected: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(answer["rejected"] if isinstance(answer["rejected"], list) else []):
        if not isinstance(item, dict) or not isinstance(item.get("reason"), str):
            raise VerificationError("merge_classification", "rejected record entry is invalid", {"index": index})
        fields = {"event", "reason", "duplicate_of"} if item["reason"] in {"duplicate", "echo"} else {"event", "reason"}
        item = require_map(item, fields, f"answer.rejected/{index}")
        if not isinstance(item["event"], str) or item["event"] not in records or item["event"] in rejected or not isinstance(item["reason"], str):
            raise VerificationError("merge_classification", "rejected record entry is invalid", {"index": index})
        rejected[item["event"]] = item
    if set(accepted) & set(rejected) or set(accepted) | set(rejected) != set(records):
        raise VerificationError("merge_classification", "every record must be classified exactly once")

    def interval(event: dict[str, Any]) -> tuple[int, int]:
        payload = event["payload"]
        try:
            lo = payload["local_time"] - payload["offset_max"]
            hi = payload["local_time"] - payload["offset_min"]
        except (KeyError, TypeError) as exc:
            raise VerificationError("invalid_case", "MERGE record interval payload is invalid") from exc
        return lo, hi

    for event_id, at in accepted.items():
        lo, hi = interval(records[event_id])
        if not lo <= at <= hi:
            raise VerificationError("merge_interval", "assigned time is outside record interval", {"event": event_id})
    required: set[tuple[str, str, int]] = set()
    for child_id in accepted:
        child = records[child_id]
        for parent_id in child["parents"]:
            if parent_id in accepted:
                required.add((parent_id, child_id, 1))
                if accepted[parent_id] + 1 > accepted[child_id]:
                    raise VerificationError("merge_order", "message parent does not precede child", {"before": parent_id, "after": child_id})
    by_source: dict[str, list[str]] = {}
    for event_id in accepted:
        by_source.setdefault(records[event_id]["payload"]["source"], []).append(event_id)
    for ids in by_source.values():
        ids.sort(key=lambda value: records[value]["payload"]["sequence"])
        for before, after in zip(ids, ids[1:]):
            required.add((before, after, 1))
            if accepted[before] + 1 > accepted[after]:
                raise VerificationError("merge_order", "source sequence does not increase", {"before": before, "after": after})
    if contract_task in {"non-unique-archives", "partial-order-archive"}:
        if set(accepted) != set(records) or rejected:
            raise VerificationError("merge_non_unique", "non-unique archive must retain every record")
        ordered_ids = sorted(records, key=lambda event_id: records[event_id]["payload"]["key"].encode("utf-8"))
        ranges = []
        domain_size = 1
        for event_id in ordered_ids:
            lo, hi = interval(records[event_id])
            choices = range(lo, hi + 1)
            domain_size *= len(choices)
            if domain_size > contract["maximum_domain"]:
                raise VerificationError("invalid_case", "MERGE archive solution domain exceeds its public bound")
            ranges.append(choices)
        solutions = []
        for values in itertools.product(*ranges):
            schedule = dict(zip(ordered_ids, values))
            if all(schedule[before] + gap <= schedule[after] for before, after, gap in required):
                solutions.append(values)
        if len(solutions) < contract["minimum_solutions"]:
            raise VerificationError("invalid_case", "MERGE archive is not sufficiently non-unique", {"solutions": len(solutions)})
        submitted = tuple(accepted[event_id] for event_id in ordered_ids)
        if submitted != min(solutions):
            raise VerificationError("merge_non_unique", "submitted archive is not the canonical member of the consistent solution set", {"solutions": len(solutions)})
    supplied: set[tuple[str, str, int]] = set()
    if not isinstance(answer["certificate"], list):
        raise VerificationError("merge_certificate", "certificate must be a list")
    for index, item in enumerate(answer["certificate"]):
        item = require_map(item, {"before", "after", "minimum_gap"}, f"answer.certificate/{index}")
        if (
            not isinstance(item["before"], str)
            or not isinstance(item["after"], str)
            or isinstance(item["minimum_gap"], bool)
            or not isinstance(item["minimum_gap"], int)
            or item["minimum_gap"] < 0
        ):
            raise VerificationError("merge_certificate", "certificate edge fields are invalid", {"index": index})
        edge = (item["before"], item["after"], item["minimum_gap"])
        if edge in supplied:
            raise VerificationError("merge_certificate", "certificate edge is duplicated")
        supplied.add(edge)
    if contract_task in {"causal-compression", "partial-order-archive"}:
        if set(accepted) != set(records) or rejected:
            raise VerificationError("merge_compression", "causal compression must retain every record")
        if len(supplied) > contract["maximum_certificate_edges"] or not supplied <= required:
            raise VerificationError("merge_compression", "compressed certificate uses an unsupported or excessive edge")

        def covers(edges: set[tuple[str, str, int]], target: tuple[str, str, int]) -> bool:
            before, after, minimum_gap = target
            best = {before: 0}
            for _ in range(len(records)):
                changed = False
                for left, right, gap in edges:
                    if left in best and best.get(right, -1) < best[left] + gap:
                        best[right] = best[left] + gap
                        changed = True
                if not changed:
                    break
            return best.get(after, -1) >= minimum_gap

        if any(not covers(supplied, edge) for edge in required):
            raise VerificationError("merge_compression", "compressed certificate does not imply every active order constraint")
        for edge in supplied:
            if all(covers(supplied - {edge}, required_edge) for required_edge in required):
                raise VerificationError("merge_compression", "compressed certificate still contains a redundant edge")
        if contract_task == "partial-order-archive":
            ordered = sorted(records, key=cre.parse_id)
            incomparable = any(
                not covers(required, (left, right, 1)) and not covers(required, (right, left, 1))
                for index, left in enumerate(ordered)
                for right in ordered[index + 1:]
            )
            if not incomparable:
                raise VerificationError("invalid_case", "MERGE partial-order archive has no incomparable records")
    elif supplied != required:
        raise VerificationError("merge_certificate", "certificate does not exactly cover active order constraints")

    if contract_task == "deduplicate":
        identity_field = contract["identity_field"]
        equivalence_field = contract["equivalence_field"]
        groups: dict[str, set[str]] = {}
        for event_id, event in records.items():
            groups.setdefault(event["payload"][identity_field], set()).add(event_id)
        for identity, members in groups.items():
            if len(members) < 2:
                continue
            equivalents = {records[event_id]["payload"][equivalence_field] for event_id in members}
            if len(equivalents) != 1:
                raise VerificationError("invalid_case", "MERGE duplicate identity has non-equivalent bodies", {"identity": identity})
            survivors = members & set(accepted)
            if len(survivors) != 1:
                raise VerificationError("merge_duplicate", "each duplicate identity needs exactly one accepted survivor", {"identity": identity})
            survivor = next(iter(survivors))
            for event_id in members - survivors:
                claim = rejected.get(event_id)
                if claim is None or claim["reason"] != "duplicate" or claim["duplicate_of"] != survivor:
                    raise VerificationError("merge_duplicate", "duplicate rejection does not name its accepted survivor", {"event": event_id})
    elif contract_task == "split-brain":
        component_field = contract["component_field"]
        writer_field = contract["writer_field"]
        components: dict[str, dict[str, set[str]]] = {}
        for event_id, event in records.items():
            payload = event["payload"]
            components.setdefault(payload[component_field], {}).setdefault(payload[writer_field], set()).add(event_id)
        for component, writers in components.items():
            if len(writers) < 2 or any(len(members) < 2 for members in writers.values()):
                raise VerificationError("invalid_case", "split-brain component needs at least two multi-record writers", {"component": component})
            selected = [writer for writer, members in writers.items() if members & set(accepted)]
            if len(selected) != 1:
                raise VerificationError("merge_split", "split-brain component needs exactly one selected writer", {"component": component})
            survivor = selected[0]
            if not writers[survivor] <= set(accepted):
                raise VerificationError("merge_split", "selected writer must be retained as one complete branch", {"component": component, "writer": survivor})
            for writer, members in writers.items():
                if writer == survivor:
                    continue
                for event_id in members:
                    claim = rejected.get(event_id)
                    if claim is None or claim["reason"] != "conflict_set":
                        raise VerificationError("merge_split", "losing writer must be rejected as one conflict component", {"component": component, "writer": writer})
    elif contract_task == "quorum-ledger":
        operation_field = contract["operation_field"]
        claim_field = contract["claim_field"]
        replica_field = contract["replica_field"]
        operations: dict[str, dict[str, dict[str, str]]] = {}
        for event_id, event in records.items():
            payload = event["payload"]
            operation = payload[operation_field]
            claim_value = payload[claim_field]
            replica = payload[replica_field]
            claims = operations.setdefault(operation, {})
            replicas = claims.setdefault(claim_value, {})
            if replica in replicas:
                raise VerificationError("invalid_case", "quorum replica votes more than once for one claim", {"operation": operation, "replica": replica})
            replicas[replica] = event_id
        for operation, claims in operations.items():
            all_replicas: set[str] = set()
            for replicas in claims.values():
                overlap = all_replicas & set(replicas)
                if overlap:
                    raise VerificationError("invalid_case", "quorum replica equivocates across claims", {"operation": operation, "replica": sorted(overlap)[0]})
                all_replicas.update(replicas)
            qualified = [claim_value for claim_value, replicas in claims.items() if len(replicas) >= contract["quorum"]]
            if len(qualified) != 1:
                raise VerificationError("invalid_case", "quorum ledger needs one uniquely qualified claim", {"operation": operation})
            winner = qualified[0]
            winner_events = set(claims[winner].values())
            if not winner_events <= set(accepted):
                raise VerificationError("merge_quorum", "every attestation for the quorum claim must be retained", {"operation": operation, "claim": winner})
            accepted_claims = {
                records[event_id]["payload"][claim_field]
                for event_id in accepted
                if records[event_id]["payload"][operation_field] == operation
            }
            if accepted_claims != {winner}:
                raise VerificationError("merge_quorum", "accepted records do not select the unique quorum claim", {"operation": operation})
            for claim_value, replicas in claims.items():
                if claim_value == winner:
                    continue
                for event_id in replicas.values():
                    claim = rejected.get(event_id)
                    if claim is None or claim["reason"] != "minority":
                        raise VerificationError("merge_quorum", "non-quorum attestations must be rejected as minority", {"operation": operation, "claim": claim_value})
    elif contract_task == "offset-domains":
        domain_field = contract["domain_field"]
        domains: dict[str, list[str]] = {}
        for event_id, event in records.items():
            domains.setdefault(event["payload"][domain_field], []).append(event_id)
        if len(domains) < contract["minimum_domains"] or any(len(members) < contract["minimum_records_per_domain"] for members in domains.values()):
            raise VerificationError("invalid_case", "MERGE offset-domain case does not meet its public domain bounds")
        if rejected:
            raise VerificationError("merge_domain", "offset-domain reconstruction must retain every record")
        for domain, members in domains.items():
            offsets = {records[event_id]["payload"]["local_time"] - accepted[event_id] for event_id in members}
            if len(offsets) != 1:
                raise VerificationError("merge_domain", "records in one clock domain do not share one offset", {"domain": domain})
    elif contract_task == "minimal-conflict":
        if not contract["minimum_size"] <= len(rejected) <= contract["maximum_size"]:
            raise VerificationError("merge_minimal", "conflict component size is outside public bounds")
        if any(claim["reason"] != "conflict_set" for claim in rejected.values()):
            raise VerificationError("merge_minimal", "minimal conflict records must use conflict_set rejection")
    elif contract_task == "weighted-cut":
        if len(records) > contract["maximum_records"]:
            raise VerificationError("invalid_case", "MERGE weighted-cut case exceeds its exact-search bound")
        by_key = {event["payload"]["key"]: event_id for event_id, event in records.items()}
        conflict_sets: list[set[str]] = []
        for index, conflict in enumerate(contract["conflicts"]):
            if any(key not in by_key for key in conflict):
                raise VerificationError("invalid_case", "MERGE weighted-cut conflict names an unknown record", {"index": index})
            conflict_sets.append({by_key[key] for key in conflict})
        universe = sorted(set().union(*conflict_sets), key=cre.parse_id)
        submitted = set(rejected)
        if not submitted <= set(universe) or any(claim["reason"] != "cut" for claim in rejected.values()):
            raise VerificationError("merge_cut", "weighted cut contains an unsupported record or reason")
        optimal_weight: int | None = None
        optimal_sets: set[frozenset[str]] = set()
        for size in range(len(universe) + 1):
            for combination in itertools.combinations(universe, size):
                candidate = frozenset(combination)
                if not all(candidate & conflict for conflict in conflict_sets):
                    continue
                weight = sum(records[event_id]["payload"]["weight"] for event_id in candidate)
                if optimal_weight is None or weight < optimal_weight:
                    optimal_weight, optimal_sets = weight, {candidate}
                elif weight == optimal_weight:
                    optimal_sets.add(candidate)
        if frozenset(submitted) not in optimal_sets:
            raise VerificationError("merge_cut", "submitted rejection is not a minimum-weight conflict cut", {"submitted_weight": sum(records[event_id]["payload"]["weight"] for event_id in submitted), "minimum_weight": optimal_weight})
    elif contract_task == "echo-chain":
        update_field = contract["update_field"]
        echo_field = contract["echo_field"]
        by_key = {event["payload"]["key"]: event_id for event_id, event in records.items()}
        groups: dict[str, set[str]] = {}
        for event_id, event in records.items():
            groups.setdefault(event["payload"][update_field], set()).add(event_id)
        if not any(len(members) >= 3 for members in groups.values()):
            raise VerificationError("invalid_case", "MERGE echo-chain case needs a transitive chain")
        for update, members in groups.items():
            roots = {event_id for event_id in members if records[event_id]["payload"][echo_field] is None}
            if len(roots) != 1:
                raise VerificationError("invalid_case", "MERGE echo chain needs exactly one root", {"update": update})
            root = next(iter(roots))
            if members & set(accepted) != {root}:
                raise VerificationError("merge_echo", "only the original update may survive an echo chain", {"update": update})
            for event_id in members - {root}:
                parent_key = records[event_id]["payload"][echo_field]
                parent = by_key.get(parent_key)
                if parent not in members:
                    raise VerificationError("invalid_case", "MERGE echo points outside its update chain", {"event": event_id})
                claim = rejected.get(event_id)
                if claim is None or claim["reason"] != "echo" or claim["duplicate_of"] != parent:
                    raise VerificationError("merge_echo", "echo rejection must name its direct predecessor", {"event": event_id})
                seen = {event_id}
                cursor = parent
                while cursor != root:
                    if cursor in seen:
                        raise VerificationError("invalid_case", "MERGE echo chain contains a cycle", {"update": update})
                    seen.add(cursor)
                    next_key = records[cursor]["payload"][echo_field]
                    cursor = by_key.get(next_key)
                    if cursor not in members:
                        raise VerificationError("invalid_case", "MERGE echo chain is disconnected", {"update": update})
    elif contract_task == "weighted-evidence":
        operation_field = contract["operation_field"]
        claim_field = contract["claim_field"]
        source_field = contract["source_field"]
        weight_field = contract["weight_field"]
        operations: dict[str, dict[str, dict[str, str]]] = {}
        for event_id, event in records.items():
            payload = event["payload"]
            claims = operations.setdefault(payload[operation_field], {})
            sources = claims.setdefault(payload[claim_field], {})
            if payload[source_field] in sources:
                raise VerificationError("invalid_case", "MERGE evidence source votes twice for one claim", {"operation": payload[operation_field], "source": payload[source_field]})
            sources[payload[source_field]] = event_id
        for operation, claims in operations.items():
            all_sources: set[str] = set()
            for sources in claims.values():
                overlap = all_sources & set(sources)
                if overlap:
                    raise VerificationError("invalid_case", "MERGE evidence source equivocates across claims", {"operation": operation, "source": sorted(overlap)[0]})
                all_sources.update(sources)
            totals = {claim: sum(records[event_id]["payload"][weight_field] for event_id in sources.values()) for claim, sources in claims.items()}
            ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0].encode("utf-8")))
            if len(ranked) < 2 or ranked[0][1] - ranked[1][1] < contract["minimum_margin"]:
                raise VerificationError("invalid_case", "MERGE weighted evidence has no unique winner with the public margin", {"operation": operation})
            winner = ranked[0][0]
            winner_events = set(claims[winner].values())
            if set(accepted) & set().union(*(set(sources.values()) for claim, sources in claims.items() if claim != winner)) or not winner_events <= set(accepted):
                raise VerificationError("merge_weight", "accepted records do not select every source for the weighted winner", {"operation": operation, "claim": winner})
            for claim, sources in claims.items():
                if claim == winner:
                    continue
                for event_id in sources.values():
                    rejection = rejected.get(event_id)
                    if rejection is None or rejection["reason"] != "outvoted":
                        raise VerificationError("merge_weight", "losing weighted evidence must be rejected as outvoted", {"event": event_id})
    elif contract_task in {"causal-compression", "partial-order-archive"}:
        pass
    elif contract_task == "equivocation":
        grouped: dict[str, dict[str, dict[str, set[str]]]] = {}
        for event_id, event in records.items():
            payload = event["payload"]
            grouped.setdefault(payload[contract["operation_field"]], {}).setdefault(payload[contract["source_field"]], {}).setdefault(payload[contract["claim_field"]], set()).add(event_id)
        equivocal: set[str] = set()
        for operation, sources in grouped.items():
            bad_sources = {source for source, claims in sources.items() if len(claims) >= contract["minimum_distinct_claims"]}
            if not bad_sources or len(bad_sources) > contract["maximum_equivocating_sources"]:
                raise VerificationError("invalid_case", "MERGE operation has no bounded equivocating source", {"operation": operation})
            for source in bad_sources:
                equivocal.update(event_id for members in sources[source].values() for event_id in members)
            honest_claims = {claim for source, claims in sources.items() if source not in bad_sources for claim in claims}
            if len(honest_claims) != 1:
                raise VerificationError("invalid_case", "MERGE honest sources do not agree on one claim", {"operation": operation})
        if set(rejected) != equivocal or set(accepted) != set(records) - equivocal:
            raise VerificationError("merge_equivocation", "classification does not isolate exactly the equivocating source records")
        for event_id in equivocal:
            if rejected[event_id]["reason"] != "equivocation":
                raise VerificationError("merge_equivocation", "equivocating records need the equivocation reason", {"event": event_id})

    for event_id, claim in rejected.items():
        reason = claim["reason"]
        if reason not in {"inconsistent", "conflict_set", "duplicate", "minority", "cut", "echo", "outvoted", "equivocation"}:
            raise VerificationError("merge_rejection", "unsupported rejection reason", {"event": event_id})
        if reason == "duplicate":
            if contract_task != "deduplicate":
                raise VerificationError("merge_rejection", "duplicate rejection requires a public dedup contract", {"event": event_id})
            target = claim["duplicate_of"]
            if not isinstance(target, str) or target not in accepted:
                raise VerificationError("merge_duplicate", "duplicate survivor is not accepted", {"event": event_id})
            left, right = records[event_id]["payload"], records[target]["payload"]
            if left[contract["identity_field"]] != right[contract["identity_field"]] or left[contract["equivalence_field"]] != right[contract["equivalence_field"]]:
                raise VerificationError("merge_duplicate", "duplicate rejection links different operations", {"event": event_id})
            continue
        if reason == "minority":
            if contract_task != "quorum-ledger":
                raise VerificationError("merge_rejection", "minority rejection requires a public quorum contract", {"event": event_id})
            continue
        if reason == "cut":
            if contract_task != "weighted-cut":
                raise VerificationError("merge_rejection", "cut rejection requires a public weighted-cut contract", {"event": event_id})
            continue
        if reason == "echo":
            if contract_task != "echo-chain":
                raise VerificationError("merge_rejection", "echo rejection requires a public echo-chain contract", {"event": event_id})
            continue
        if reason == "outvoted":
            if contract_task != "weighted-evidence":
                raise VerificationError("merge_rejection", "outvoted rejection requires a public weighted-evidence contract", {"event": event_id})
            continue
        if reason == "equivocation":
            if contract_task != "equivocation":
                raise VerificationError("merge_rejection", "equivocation rejection requires a public equivocation contract", {"event": event_id})
            continue
        if reason == "conflict_set":
            continue
        lo, hi = interval(records[event_id])
        event = records[event_id]
        for parent in event["parents"]:
            if parent in accepted:
                lo = max(lo, accepted[parent] + 1)
        for child_id, child in records.items():
            if child_id in accepted and event_id in child["parents"]:
                hi = min(hi, accepted[child_id] - 1)
        same = [(other["payload"]["sequence"], other_id) for other_id, other in records.items() if other_id in accepted and other["payload"]["source"] == event["payload"]["source"]]
        for sequence, other_id in same:
            if sequence < event["payload"]["sequence"]:
                lo = max(lo, accepted[other_id] + 1)
            elif sequence > event["payload"]["sequence"]:
                hi = min(hi, accepted[other_id] - 1)
        if lo <= hi:
            raise VerificationError("merge_rejection", "rejected record remains feasible under submitted schedule", {"event": event_id})
    def conflict_infeasible(conflicts: set[str]) -> bool:
        selected = set(accepted) | conflicts
        lower = {event_id: accepted[event_id] if event_id in accepted else interval(records[event_id])[0] for event_id in selected}
        upper = {event_id: accepted[event_id] if event_id in accepted else interval(records[event_id])[1] for event_id in selected}
        edges: set[tuple[str, str]] = set()
        for child_id in selected:
            edges.update((parent_id, child_id) for parent_id in records[child_id]["parents"] if parent_id in selected)
        selected_by_source: dict[str, list[str]] = {}
        for event_id in selected:
            selected_by_source.setdefault(records[event_id]["payload"]["source"], []).append(event_id)
        for ids in selected_by_source.values():
            ids.sort(key=lambda value: records[value]["payload"]["sequence"])
            edges.update(zip(ids, ids[1:]))
        infeasible = False
        for _iteration in range(len(selected)):
            changed = False
            for before, after in edges:
                candidate = lower[before] + 1
                if candidate > lower[after]:
                    lower[after] = candidate
                    changed = True
                    if candidate > upper[after]:
                        infeasible = True
                        break
            if infeasible or not changed:
                break
        else:
            infeasible = True
        return infeasible

    conflict_set = {event_id for event_id, claim in rejected.items() if claim["reason"] == "conflict_set"}
    if conflict_set:
        if len(conflict_set) < 2:
            raise VerificationError("merge_rejection", "conflict_set requires at least two records")
        if not conflict_infeasible(conflict_set):
            raise VerificationError("merge_rejection", "conflict_set records remain jointly feasible")
        if contract_task == "minimal-conflict":
            for removed in conflict_set:
                if conflict_infeasible(conflict_set - {removed}):
                    raise VerificationError("merge_minimal", "conflict component is not inclusion-minimal", {"redundant": removed})
    displacement = sum(abs(at - sum(interval(records[event_id])) // 2) for event_id, at in accepted.items())
    return {
        "rejected_weight": sum(records[event_id]["payload"]["weight"] for event_id in rejected),
        "temporal_displacement": displacement,
        "certificate_units": (len(cre.canonical_bytes(answer["certificate"])) + 63) // 64,
    }


def merge_score(case: dict[str, Any], score_value: Any, raw: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    score_fields = {"format", "family", "reference_scale", "metric_bounds"}
    if isinstance(score_value, dict) and "contract" in score_value:
        score_fields.add("contract")
    score = require_map(score_value, score_fields, "score")
    if score["format"] != "afterimage-score/0.1" or score["family"] != "MERGE" or case["family"] != "MERGE":
        fail("invalid_score", "MERGE score descriptor is invalid")
    bounds = require_map(score["metric_bounds"], {"temporal_displacement", "certificate_units"}, "score.metric_bounds")
    displacement_bound = checked_nonnegative(bounds["temporal_displacement"], "score.metric_bounds.temporal_displacement")
    certificate_bound = checked_nonnegative(bounds["certificate_units"], "score.metric_bounds.certificate_units")
    if raw["temporal_displacement"] > displacement_bound or raw["certificate_units"] > certificate_bound:
        raise VerificationError("metric_limit", "MERGE metric exceeds public bound")
    effective = ((raw["rejected_weight"] * (displacement_bound + 1) + raw["temporal_displacement"]) * (certificate_bound + 1) + raw["certificate_units"]) + 1
    metrics = {**raw, "effective_cost": effective}
    points, scale = case["points"], score["reference_scale"]
    completion = (65 * points + 99) // 100
    pool = points - completion
    optimization = pool * (1_000_000 * scale // (scale + effective)) // 1_000_000
    return metrics, {"completion": completion, "optimization": optimization, "total": completion + optimization, "nominal_max": points}


def validate_pulse_answer(score_value: Any, answer: Any) -> dict[str, int]:
    if not isinstance(answer, dict) or set(answer) not in ({"program"}, {"program", "invariant"}):
        raise VerificationError("invalid_schema", "PULSE answer fields are invalid")
    try:
        return pulse_runtime.verify_program(answer["program"], score_value.get("contract") if isinstance(score_value, dict) else None)
    except pulse_runtime.PulseError as exc:
        raise VerificationError(exc.code, exc.message, exc.context) from exc


def pulse_score(case: dict[str, Any], score_value: Any, raw: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    score = require_map(score_value, {"format", "family", "reference_scale", "metric_bounds", "contract"}, "score")
    if score["format"] != "afterimage-score/0.1" or score["family"] != "PULSE" or case["family"] != "PULSE":
        fail("invalid_score", "PULSE score descriptor is invalid")
    scale = checked_nonnegative(score["reference_scale"], "score.reference_scale")
    if scale <= 0:
        fail("invalid_score", "PULSE reference scale must be positive")
    bounds = require_map(score["metric_bounds"], {"worst_latency", "live_state_cells"}, "score.metric_bounds")
    latency_bound = checked_nonnegative(bounds["worst_latency"], "score.metric_bounds.worst_latency")
    state_bound = checked_nonnegative(bounds["live_state_cells"], "score.metric_bounds.live_state_cells")
    if raw["worst_latency"] > latency_bound or raw["live_state_cells"] > state_bound:
        raise VerificationError("metric_limit", "PULSE metric exceeds public bound")
    effective = ((raw["program_bytes"] * (latency_bound + 1) + raw["worst_latency"]) * (state_bound + 1) + raw["live_state_cells"]) + 1
    metrics = {key: raw[key] for key in ("program_bytes", "worst_latency", "live_state_cells")}
    metrics["effective_cost"] = effective
    points = case["points"]
    completion = (65 * points + 99) // 100
    pool = points - completion
    optimization = pool * (1_000_000 * scale // (scale + effective)) // 1_000_000
    return metrics, {"completion": completion, "optimization": optimization, "total": completion + optimization, "nominal_max": points}


def validate_mosaic_answer(world: kit.ValidatedBundle, case: dict[str, Any], score_value: Any, answer: Any) -> dict[str, int]:
    logical = kit.resolve_logical_world(world, case["world"])
    fragments = [event["payload"] for event in logical.base_events if event["topic"] == "mosaic.fragment"]
    try:
        return mosaic_runtime.validate_answer(fragments, answer, score_value.get("contract") if isinstance(score_value, dict) else None)
    except mosaic_runtime.MosaicError as exc:
        raise VerificationError(exc.code, exc.message, exc.context) from exc


def mosaic_score(case: dict[str, Any], score_value: Any, raw: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    score = require_map(score_value, {"format", "family", "reference_scale", "metric_bounds", "contract"}, "score")
    if score["format"] != "afterimage-score/0.1" or score["family"] != "MOSAIC" or case["family"] != "MOSAIC":
        fail("invalid_score", "MOSAIC score descriptor is invalid")
    scale = checked_nonnegative(score["reference_scale"], "score.reference_scale")
    if scale <= 0:
        fail("invalid_score", "MOSAIC reference scale must be positive")
    bounds = require_map(score["metric_bounds"], {"graph_size", "certificate_units"}, "score.metric_bounds")
    graph_bound = checked_nonnegative(bounds["graph_size"], "score.metric_bounds.graph_size")
    certificate_bound = checked_nonnegative(bounds["certificate_units"], "score.metric_bounds.certificate_units")
    if raw["graph_size"] > graph_bound or raw["certificate_units"] > certificate_bound:
        raise VerificationError("metric_limit", "MOSAIC metric exceeds public bound")
    effective = ((raw["unexplained_weight"] * (graph_bound + 1) + raw["graph_size"]) * (certificate_bound + 1) + raw["certificate_units"]) + 1
    metrics = {**raw, "effective_cost": effective}
    points = case["points"]
    completion = (65 * points + 99) // 100
    pool = points - completion
    optimization = pool * (1_000_000 * scale // (scale + effective)) // 1_000_000
    return metrics, {"completion": completion, "optimization": optimization, "total": completion + optimization, "nominal_max": points}


def validate_lens_answer(score_value: Any, answer: Any) -> dict[str, int]:
    answer = require_map(answer, {"program"}, "LENS answer")
    try:
        return lens_runtime.verify_program(answer["program"], score_value.get("contract") if isinstance(score_value, dict) else None)
    except lens_runtime.LensError as exc:
        raise VerificationError(exc.code, exc.message, exc.context) from exc


def lens_score(case: dict[str, Any], score_value: Any, raw: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    score = require_map(score_value, {"format", "family", "reference_scale", "metric_bounds", "contract"}, "score")
    if score["format"] != "afterimage-score/0.1" or score["family"] != "LENS" or case["family"] != "LENS":
        fail("invalid_score", "LENS score descriptor is invalid")
    scale = checked_nonnegative(score["reference_scale"], "score.reference_scale")
    if scale <= 0:
        fail("invalid_score", "LENS reference scale must be positive")
    bounds = require_map(score["metric_bounds"], {"auxiliary_schema_cells", "worst_reductions"}, "score.metric_bounds")
    auxiliary_bound = checked_nonnegative(bounds["auxiliary_schema_cells"], "score.metric_bounds.auxiliary_schema_cells")
    reduction_bound = checked_nonnegative(bounds["worst_reductions"], "score.metric_bounds.worst_reductions")
    if raw["auxiliary_schema_cells"] > auxiliary_bound or raw["worst_reductions"] > reduction_bound:
        raise VerificationError("metric_limit", "LENS metric exceeds public bound")
    effective = ((raw["program_nodes"] * (auxiliary_bound + 1) + raw["auxiliary_schema_cells"]) * (reduction_bound + 1) + raw["worst_reductions"]) + 1
    metrics = {key: raw[key] for key in ("program_nodes", "auxiliary_schema_cells", "worst_reductions")}
    metrics["effective_cost"] = effective
    points = case["points"]
    completion = (65 * points + 99) // 100
    pool = points - completion
    optimization = pool * (1_000_000 * scale // (scale + effective)) // 1_000_000
    return metrics, {"completion": completion, "optimization": optimization, "total": completion + optimization, "nominal_max": points}


def validate_covenant_answer(score_value: Any, answer: Any) -> dict[str, int]:
    answer = require_map(answer, {"policy", "claimed_response_bound"}, "COVENANT answer")
    claimed = checked_nonnegative(answer["claimed_response_bound"], "COVENANT answer.claimed_response_bound")
    if not isinstance(score_value, dict) or "contract" not in score_value:
        fail("invalid_score", "COVENANT score descriptor lacks a contract")
    try:
        metrics = covenant_runtime.verify_policy(score_value["contract"], answer["policy"])
    except covenant_runtime.CovenantError as exc:
        raise VerificationError(exc.code, exc.message, exc.context) from exc
    if claimed != metrics["worst_response_bound"]:
        raise VerificationError("covenant_claim", "claimed response bound differs from exhaustive model checking")
    return metrics


def covenant_score(case: dict[str, Any], score_value: Any, raw: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    score = require_map(score_value, {"format", "family", "reference_scale", "metric_bounds", "contract"}, "score")
    if score["format"] != "afterimage-score/0.1" or score["family"] != "COVENANT" or case["family"] != "COVENANT":
        fail("invalid_score", "COVENANT score descriptor is invalid")
    scale = checked_nonnegative(score["reference_scale"], "score.reference_scale")
    if scale <= 0:
        fail("invalid_score", "COVENANT reference scale must be positive")
    bounds = require_map(score["metric_bounds"], {"worst_response_bound", "reachable_states"}, "score.metric_bounds")
    response_bound = checked_nonnegative(bounds["worst_response_bound"], "score.metric_bounds.worst_response_bound")
    states_bound = checked_nonnegative(bounds["reachable_states"], "score.metric_bounds.reachable_states")
    if raw["worst_response_bound"] > response_bound or raw["reachable_states"] > states_bound:
        raise VerificationError("metric_limit", "COVENANT metric exceeds public bound")
    effective = ((raw["policy_nodes"] * (response_bound + 1) + raw["worst_response_bound"]) * (states_bound + 1) + raw["reachable_states"]) + 1
    metrics = {**raw, "effective_cost": effective}
    points = case["points"]
    completion = (65 * points + 99) // 100
    pool = points - completion
    optimization = pool * (1_000_000 * scale // (scale + effective)) // 1_000_000
    return metrics, {"completion": completion, "optimization": optimization, "total": completion + optimization, "nominal_max": points}


def validate_paradox_answer(
    world: kit.ValidatedBundle,
    case: dict[str, Any],
    score_value: Any,
    answer: Any,
    facts: set[str],
) -> dict[str, int]:
    answer_map = require_map(answer, {"left_history", "right_history", "equivalence", "safety_evidence", "latent_difference"}, "PARADOX answer")
    if not isinstance(score_value, dict) or "contract" not in score_value:
        fail("invalid_score", "PARADOX score descriptor lacks a contract")
    try:
        contract = paradox_runtime.validate_contract(score_value["contract"])
    except paradox_runtime.ParadoxError as exc:
        raise VerificationError(exc.code, exc.message, exc.context) from exc
    replays: dict[str, ReplayState] = {}
    for side in ("left", "right"):
        history = answer_map[f"{side}_history"]
        if not isinstance(history, dict) or not isinstance(history.get("steps"), list) or not history["steps"]:
            raise VerificationError("paradox_history", "paired branch history is missing or empty", {"side": side})
        final_step = history["steps"][-1]
        if not isinstance(final_step, dict) or not isinstance(final_step.get("case"), str):
            raise VerificationError("paradox_history", "paired branch history has an invalid final step", {"side": side})
        target_case = contract.get("history_case", final_step["case"])
        target = {**case, "input_branch": f"history:{target_case}"}
        history_result = resolve_input_history(world, target, history, facts)
        replays[side] = evaluate_case_state(world, case, history_result.base_events, history_result.branch)
    try:
        return paradox_runtime.validate_certificate(
            contract,
            answer_map,
            left_branch=replays["left"].branch,
            right_branch=replays["right"].branch,
            left_records=replays["left"].records,
            right_records=replays["right"].records,
            left_events=replays["left"].events,
            right_events=replays["right"].events,
        )
    except paradox_runtime.ParadoxError as exc:
        raise VerificationError(exc.code, exc.message, exc.context) from exc


def paradox_score(case: dict[str, Any], score_value: Any, raw: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    score = require_map(score_value, {"format", "family", "reference_scale", "metric_bounds", "contract"}, "score")
    if score["format"] != "afterimage-score/0.1" or score["family"] != "PARADOX" or case["family"] != "PARADOX":
        fail("invalid_score", "PARADOX score descriptor is invalid")
    scale = checked_nonnegative(score["reference_scale"], "score.reference_scale")
    if scale <= 0:
        fail("invalid_score", "PARADOX reference scale must be positive")
    bounds = require_map(score["metric_bounds"], {"latent_difference_weight", "proof_steps"}, "score.metric_bounds")
    difference_bound = checked_nonnegative(bounds["latent_difference_weight"], "score.metric_bounds.latent_difference_weight")
    proof_bound = checked_nonnegative(bounds["proof_steps"], "score.metric_bounds.proof_steps")
    if raw["latent_difference_weight"] <= 0:
        raise VerificationError("paradox_no_difference", "PARADOX latent difference must be non-zero")
    if raw["latent_difference_weight"] > difference_bound or raw["proof_steps"] > proof_bound:
        raise VerificationError("metric_limit", "PARADOX metric exceeds public bound")
    effective = ((raw["paired_witness_units"] * (difference_bound + 1) + raw["latent_difference_weight"]) * (proof_bound + 1) + raw["proof_steps"]) + 1
    metrics = {**raw, "effective_cost": effective}
    points = case["points"]
    completion = (65 * points + 99) // 100
    pool = points - completion
    optimization = pool * (1_000_000 * scale // (scale + effective)) // 1_000_000
    return metrics, {"completion": completion, "optimization": optimization, "total": completion + optimization, "nominal_max": points}


def invalid_receipt(
    world: kit.ValidatedBundle,
    case_id: str,
    witness_digest: str,
    diagnostic: dict[str, Any],
    replay: ReplayResult | None = None,
) -> dict[str, Any]:
    receipt = {
        "format": "afterimage-receipt/0.1",
        "valid": False,
        "bundle": world.bundle,
        "case": case_id,
        "witness": witness_digest,
        "diagnostics": [diagnostic],
    }
    if replay is not None:
        receipt.update({"branch": replay.branch, "projection": replay.projection, "trace": replay.trace})
    return receipt


def verify_witness_bytes(
    world: kit.ValidatedBundle,
    witness_bytes: bytes,
    source: str,
    facts: Iterable[str] = (),
    depth: int = 0,
) -> dict[str, Any]:
    fact_set = set(facts)
    if len(witness_bytes) > MAX_WITNESS_BYTES:
        fail("limit_exceeded", "witness exceeds verifier hard limit")
    try:
        witness = kit.decode_json(witness_bytes, source)
    except kit.KitError as exc:
        raise VerificationError(exc.code, exc.message, exc.context) from exc
    if not isinstance(witness, dict) or not WITNESS_REQUIRED.issubset(witness) or set(witness) - (WITNESS_REQUIRED | WITNESS_OPTIONAL):
        fail("invalid_witness", "witness outer fields are invalid")
    if witness["format"] != "afterimage-witness/0.1" or witness["semantics"] != "cre/0.1":
        fail("invalid_witness", "witness format or semantics is unsupported")
    if "meta" in witness:
        meta = witness["meta"]
        if not isinstance(meta, dict) or not meta or set(meta) - {"producer", "comment"}:
            fail("invalid_witness", "witness meta fields are invalid")
        if not all(isinstance(value, str) for value in meta.values()):
            fail("invalid_witness", "witness meta values must be Text")
    if witness["bundle"] != world.bundle:
        fail("bundle_mismatch", "witness names a different bundle")
    if not isinstance(witness["case"], str):
        fail("invalid_witness", "witness case must be Text")
    witness_digest = cre.digest_id("afterimage/witness/1", witness_bytes)

    descriptors = {item.get("id"): item for item in world.json_values["cases/index.json"]["cases"] if isinstance(item, dict)}
    if witness["case"] not in descriptors:
        fail("unknown_case", "witness case is absent from bundle", case=witness["case"])
    case = validate_case_descriptor(descriptors[witness["case"]], world)
    witness_limit = case["limits"].get("max_witness_bytes", MAX_WITNESS_BYTES)
    if len(witness_bytes) > witness_limit:
        return invalid_receipt(world, case["id"], witness_digest, VerificationError("limit_exceeded", "witness exceeds case limit").diagnostic())
    if not requirements_hold(case["requires"], fact_set):
        return invalid_receipt(world, case["id"], witness_digest, VerificationError("case_locked", "case prerequisites are not satisfied").diagnostic())
    try:
        input_history = resolve_input_history(world, case, witness.get("history"), fact_set)
    except (VerificationError, cre.CREError) as exc:
        error = exc if isinstance(exc, VerificationError) else VerificationError(exc.code, exc.message, exc.context)
        return invalid_receipt(world, case["id"], witness_digest, error.diagnostic())
    expected_parent = input_history.branch
    if witness["parent_branch"] != expected_parent:
        return invalid_receipt(world, case["id"], witness_digest, VerificationError("parent_branch_mismatch", "witness uses the wrong input branch").diagnostic())

    base_events = input_history.base_events
    try:
        embedded_policy = validate_answer_schema(world.json_values[case["answer_schema"]], witness["answer"])
        embedded_receipt = None
        if embedded_policy is not None:
            if depth >= 1:
                raise VerificationError("embedded_depth", "embedded witnesses may not contain another embedded witness")
            embedded_witness = cre.pointer_get(witness["answer"], embedded_policy["pointer"])
            if not isinstance(embedded_witness, dict) or not isinstance(embedded_witness.get("case"), str):
                raise VerificationError("embedded_witness_invalid", "embedded witness must be a witness map with a case")
            embedded_case = embedded_witness["case"]
            if embedded_case not in embedded_policy["allowed_cases"]:
                raise VerificationError("embedded_case_forbidden", "embedded witness case is outside the export policy", {"case": embedded_case})
            if embedded_policy["require_fact"] and f"case:{embedded_case}" not in fact_set:
                raise VerificationError("embedded_case_locked", "embedded witness case has not been unlocked", {"case": embedded_case})
            try:
                embedded_receipt = verify_witness_bytes(
                    world,
                    cre.canonical_bytes(embedded_witness),
                    f"{source}#{embedded_policy['pointer']}",
                    fact_set,
                    depth + 1,
                )
            except VerificationError as exc:
                raise VerificationError(
                    "embedded_witness_invalid",
                    "embedded witness failed outer-envelope validation",
                    {"inner_code": exc.code},
                ) from exc
            if not embedded_receipt["valid"]:
                inner = embedded_receipt["diagnostics"][0]
                raise VerificationError(
                    "embedded_witness_invalid",
                    "embedded witness failed independent verification",
                    {"case": embedded_case, "inner_code": inner["code"]},
                )
        score_document = world.json_values[case["score"]]
        family_result = None
        merge_metrics = None
        pulse_metrics = None
        mosaic_metrics = None
        lens_metrics = None
        covenant_metrics = None
        paradox_metrics = None
        if case["family"] == "MERGE":
            merge_metrics = validate_merge_answer(world, case, score_document, witness["answer"])
            family_result = {"valid": True, "metrics": merge_metrics}
        elif case["family"] == "PULSE":
            pulse_metrics = validate_pulse_answer(score_document, witness["answer"])
            family_result = {"valid": True, "metrics": pulse_metrics}
        elif case["family"] == "MOSAIC":
            mosaic_metrics = validate_mosaic_answer(world, case, score_document, witness["answer"])
            family_result = {"valid": True, "metrics": mosaic_metrics}
        elif case["family"] == "LENS":
            lens_metrics = validate_lens_answer(score_document, witness["answer"])
            family_result = {"valid": True, "metrics": lens_metrics}
        elif case["family"] == "COVENANT":
            covenant_metrics = validate_covenant_answer(score_document, witness["answer"])
            family_result = {"valid": True, "metrics": covenant_metrics}
        elif case["family"] == "PARADOX":
            paradox_metrics = validate_paradox_answer(world, case, score_document, witness["answer"], fact_set)
            family_result = {"valid": True, "metrics": paradox_metrics}
        baseline = evaluate_case_state(world, case, base_events, expected_parent)
        policy = validate_intervention(
            world.json_values[case["intervention_policy"]],
            witness["intervention"],
            witness_bundle=world.bundle,
            witness_case=case["id"],
            parent_branch=expected_parent,
            base_events=base_events,
            known_events=baseline.events,
        )
        replay = replay_case(world, case, policy.operations, expected_parent, baseline, base_events)
        if case["family"] == "CASCADE" and isinstance(score_document, dict) and "proof_contract" in score_document:
            family_result = {"valid": True, "metrics": validate_cascade_proof(witness["answer"], replay, policy, score_document["proof_contract"])}
        elif case["family"] == "CASCADE" and isinstance(score_document, dict) and "projection_contract" in score_document:
            family_result = {"valid": True, "metrics": validate_cascade_projection(witness["answer"], replay, score_document["projection_contract"])}
        elif case["family"] == "CASCADE" and isinstance(score_document, dict) and "minimal_contract" in score_document:
            family_result = {"valid": True, "metrics": validate_cascade_minimal_explanation(world, case, witness["answer"], replay, policy, score_document["minimal_contract"])}
        validate_claimed(witness.get("claimed"), replay)
        decision = run_validator(
            world.json_values[case["validator"]],
            answer=witness["answer"],
            intervention=witness["intervention"],
            replay=replay,
            limits=case["limits"].get("validator"),
            embedded_receipt=embedded_receipt,
            family_result=family_result,
        )
        if not decision["valid"]:
            return invalid_receipt(world, case["id"], witness_digest, decision["diagnostics"][0], replay)
        if case["family"] == "ORIENT":
            metrics, score = orient_score(case, score_document, witness["answer"], decision["metrics"])
        elif case["family"] == "CASCADE":
            metrics, score = cascade_score(case, score_document, witness["answer"], decision["metrics"], policy, replay)
        elif case["family"] == "MERGE":
            metrics, score = merge_score(case, score_document, merge_metrics or {})
        elif case["family"] == "PULSE":
            metrics, score = pulse_score(case, score_document, pulse_metrics or {})
        elif case["family"] == "MOSAIC":
            metrics, score = mosaic_score(case, score_document, mosaic_metrics or {})
        elif case["family"] == "LENS":
            metrics, score = lens_score(case, score_document, lens_metrics or {})
        elif case["family"] == "COVENANT":
            metrics, score = covenant_score(case, score_document, covenant_metrics or {})
        elif case["family"] == "PARADOX":
            metrics, score = paradox_score(case, score_document, paradox_metrics or {})
        else:
            fail("unsupported_family", "family scoring is not implemented yet", family=case["family"])
    except VerificationError as exc:
        return invalid_receipt(world, case["id"], witness_digest, exc.diagnostic(), locals().get("replay"))
    except cre.CREError as exc:
        return invalid_receipt(world, case["id"], witness_digest, VerificationError(exc.code, exc.message, exc.context).diagnostic(), locals().get("replay"))

    return {
        "format": "afterimage-receipt/0.1",
        "valid": True,
        "bundle": world.bundle,
        "case": case["id"],
        "witness": witness_digest,
        "branch": replay.branch,
        "projection": replay.projection,
        "trace": replay.trace,
        "metrics": metrics,
        "score": score,
        "unlocks": [f"case:{case['id']}", *score_document.get("grants", [])],
        "diagnostics": [],
    }


def verify_witness(world_path: Path, witness_path: Path, facts: Iterable[str] = ()) -> dict[str, Any]:
    world = kit.verify_world(world_path)
    try:
        witness_bytes = witness_path.read_bytes()
    except OSError as exc:
        raise VerificationError("input_error", f"cannot read witness: {exc}") from exc
    return verify_witness_bytes(world, witness_bytes, str(witness_path), facts)


def emit(value: Any, pretty: bool) -> None:
    if pretty:
        print(json.dumps(cre.json_value(cre.normalize_value(value)), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(cre.canonical_text(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("witness", type=Path)
    parser.add_argument("--fact", action="append", default=[])
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = verify_witness(args.world, args.witness, args.fact)
        emit(receipt, args.pretty)
        return 0 if receipt["valid"] else 1
    except (VerificationError, kit.KitError) as exc:
        error = exc if isinstance(exc, VerificationError) else VerificationError(exc.code, exc.message, exc.context)
        emit({"type": "error", **error.diagnostic()}, args.pretty)
        return 2
    except OSError as exc:
        emit({"type": "error", "code": "io_error", "message": str(exc), "context": {}}, args.pretty)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
