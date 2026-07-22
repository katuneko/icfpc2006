#!/usr/bin/env python3
"""Certificate checker for paired PARADOX branch replays."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "python"))

import cre  # noqa: E402

HARD_MAX_PUBLIC_RECORDS = 4096
HARD_MAX_SAFETY_REQUIREMENTS = 64
HARD_MAX_LATENT_TOPICS = 256


class ParadoxError(Exception):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}


def fail(code: str, message: str, **context: Any) -> None:
    raise ParadoxError(code, message, context)


def require_map(value: Any, keys: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        fail("paradox_schema", "object has wrong fields", path=location, expected=sorted(keys))
    return value


def validate_contract(value: Any) -> dict[str, Any]:
    field_sets = {
        frozenset({"format", "safety_requirements", "latent_topics", "max_public_records"}),
        frozenset({"format", "safety_requirements", "latent_topics", "latent_pointers", "max_public_records"}),
        frozenset({"format", "history_case", "safety_requirements", "latent_topics", "max_public_records"}),
        frozenset({"format", "history_case", "safety_requirements", "latent_topics", "latent_pointers", "max_public_records"}),
    }
    if not isinstance(value, dict) or frozenset(value) not in field_sets:
        fail("paradox_schema", "contract has wrong fields")
    contract = value
    if contract["format"] != "afterimage-paradox-contract/0.1":
        fail("paradox_schema", "contract format is unsupported")
    if "history_case" in contract and (not isinstance(contract["history_case"], str) or not contract["history_case"]):
        fail("paradox_schema", "history_case must be non-empty Text")
    if isinstance(contract["max_public_records"], bool) or not isinstance(contract["max_public_records"], int) or contract["max_public_records"] < 0:
        fail("paradox_schema", "max_public_records must be a non-negative Int")
    if contract["max_public_records"] > HARD_MAX_PUBLIC_RECORDS:
        fail("paradox_limit", "max_public_records exceeds the hard ceiling")
    if not isinstance(contract["safety_requirements"], list) or not contract["safety_requirements"]:
        fail("paradox_schema", "safety_requirements must be a non-empty list")
    if len(contract["safety_requirements"]) > HARD_MAX_SAFETY_REQUIREMENTS:
        fail("paradox_limit", "safety requirement count exceeds the hard ceiling")
    requirements = []
    identities = set()
    for index, raw in enumerate(contract["safety_requirements"]):
        item = require_map(raw, {"id", "topic", "pointer", "equals", "minimum"}, f"contract.safety_requirements[{index}]")
        if not all(isinstance(item[key], str) and item[key] for key in ("id", "topic", "pointer")) or not item["pointer"].startswith("/"):
            fail("paradox_schema", "safety requirement Text fields are invalid")
        if item["id"] in identities:
            fail("paradox_schema", "safety requirement IDs must be unique")
        identities.add(item["id"])
        if isinstance(item["minimum"], bool) or not isinstance(item["minimum"], int) or item["minimum"] <= 0:
            fail("paradox_schema", "safety minimum must be positive")
        requirements.append(item)
    topics = contract["latent_topics"]
    if not isinstance(topics, dict) or not topics or not all(isinstance(topic, str) and topic for topic in topics):
        fail("paradox_schema", "latent_topics must be a non-empty Text-keyed map")
    if len(topics) > HARD_MAX_LATENT_TOPICS:
        fail("paradox_limit", "latent topic count exceeds the hard ceiling")
    for topic, weight in topics.items():
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            fail("paradox_schema", "latent topic weights must be positive", topic=topic)
    if "latent_pointers" in contract:
        pointers = contract["latent_pointers"]
        if not isinstance(pointers, dict) or set(pointers) != set(topics):
            fail("paradox_schema", "latent pointers must cover every latent topic")
        if not all(isinstance(pointer, str) and pointer.startswith("/") for pointer in pointers.values()):
            fail("paradox_schema", "latent pointers must be JSON pointers")
    return {**contract, "safety_requirements": requirements}


def event_ids_for_requirement(events: dict[str, dict[str, Any]], requirement: dict[str, Any]) -> list[str]:
    matches = []
    for event_id in sorted(events, key=cre.parse_id):
        event = events[event_id]
        if event["topic"] != requirement["topic"]:
            continue
        try:
            actual = cre.pointer_get(event, requirement["pointer"])
        except cre.CREError as exc:
            raise ParadoxError("paradox_safety", "safety evidence lacks the required field", {"requirement": requirement["id"]}) from exc
        if not cre.same_value(actual, requirement["equals"]):
            fail("paradox_safety", "branch violates a published safety requirement", requirement=requirement["id"])
        matches.append(event_id)
    if len(matches) < requirement["minimum"]:
        fail("paradox_safety", "branch lacks required safety evidence", requirement=requirement["id"])
    return matches


def validate_certificate(
    contract_value: Any,
    answer_value: Any,
    *,
    left_branch: str,
    right_branch: str,
    left_records: list[Any],
    right_records: list[Any],
    left_events: dict[str, dict[str, Any]],
    right_events: dict[str, dict[str, Any]],
) -> dict[str, int]:
    contract = validate_contract(contract_value)
    answer = require_map(
        answer_value,
        {"left_history", "right_history", "equivalence", "safety_evidence", "latent_difference"},
        "PARADOX answer",
    )
    if left_branch == right_branch:
        fail("paradox_same_branch", "paired histories must identify different branches")
    if len(left_records) > contract["max_public_records"] or len(right_records) > contract["max_public_records"]:
        fail("paradox_limit", "public projection exceeds max_public_records")
    if not cre.same_value(left_records, right_records):
        fail("paradox_projection", "paired histories have different public projections")

    equivalence = answer["equivalence"]
    if not isinstance(equivalence, list) or len(equivalence) != len(left_records):
        fail("paradox_equivalence", "equivalence certificate must cover every public record")
    seen_left: set[int] = set()
    seen_right: set[int] = set()
    for index, raw in enumerate(equivalence):
        item = require_map(raw, {"left", "right", "digest"}, f"answer.equivalence[{index}]")
        left_index, right_index = item["left"], item["right"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (left_index, right_index)):
            fail("paradox_equivalence", "equivalence indices must be Int")
        if not 0 <= left_index < len(left_records) or not 0 <= right_index < len(right_records):
            fail("paradox_equivalence", "equivalence index is out of range")
        if left_index in seen_left or right_index in seen_right:
            fail("paradox_equivalence", "equivalence mapping must be bijective")
        seen_left.add(left_index)
        seen_right.add(right_index)
        if not cre.same_value(left_records[left_index], right_records[right_index]):
            fail("paradox_equivalence", "equivalence pair names unequal records")
        expected = cre.digest_id("afterimage/paradox-public-record/1", cre.canonical_bytes(left_records[left_index]))
        if item["digest"] != expected:
            fail("paradox_equivalence", "equivalence record digest is wrong", index=index)

    evidence = require_map(answer["safety_evidence"], {"left", "right"}, "answer.safety_evidence")
    expected_evidence: dict[str, list[str]] = {"left": [], "right": []}
    for side, events in (("left", left_events), ("right", right_events)):
        for requirement in contract["safety_requirements"]:
            expected_evidence[side].extend(event_ids_for_requirement(events, requirement))
        expected_evidence[side] = sorted(set(expected_evidence[side]), key=cre.parse_id)
        if evidence[side] != expected_evidence[side]:
            fail("paradox_safety", "submitted safety evidence is incomplete or non-canonical", side=side)

    latent_topics = contract["latent_topics"]
    material_topics = set(latent_topics)
    if "latent_pointers" in contract:
        material_topics = set()
        for topic, pointer in contract["latent_pointers"].items():
            values = []
            for events in (left_events, right_events):
                side_values = []
                for event in events.values():
                    if event["topic"] != topic:
                        continue
                    try:
                        side_values.append(cre.canonical_bytes(cre.pointer_get(event, pointer)))
                    except cre.CREError as exc:
                        raise ParadoxError("paradox_difference", "latent event lacks the contracted value", {"topic": topic, "pointer": pointer}) from exc
                values.append(sorted(side_values))
            if values[0] != values[1]:
                material_topics.add(topic)
    differing = sorted(
        {
            event_id
            for event_id in set(left_events) ^ set(right_events)
            if (left_events.get(event_id) or right_events[event_id])["topic"] in material_topics
        },
        key=cre.parse_id,
    )
    if not differing:
        fail("paradox_no_difference", "paired histories have no material latent difference")
    if answer["latent_difference"] != differing:
        fail("paradox_difference", "latent-difference certificate is incomplete or non-canonical")
    difference_weight = sum(latent_topics[(left_events.get(event_id) or right_events[event_id])["topic"]] for event_id in differing)
    history_units = (len(cre.canonical_bytes(answer["left_history"])) + len(cre.canonical_bytes(answer["right_history"])) + 63) // 64
    proof_steps = len(equivalence) + len(expected_evidence["left"]) + len(expected_evidence["right"]) + len(differing)
    return {
        "paired_witness_units": history_units,
        "latent_difference_weight": difference_weight,
        "proof_steps": proof_steps,
    }
