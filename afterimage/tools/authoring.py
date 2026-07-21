"""Small deterministic authoring compilers for Afterimage case sources."""

from __future__ import annotations

from typing import Any


TRUE = ["const", True]


def compile_orient_event_claim(event_topic: str) -> dict[str, Any]:
    """Compile the ORIENT event-identity answer contract into CRE."""

    accept = {
        "id": "verify.orient-event.01-accept",
        "positive": [
            {"alias": "a", "topic": "verify.answer", "where": TRUE},
            {"alias": "r", "topic": "verify.replay", "where": TRUE},
            {
                "alias": "e",
                "topic": "verify.active",
                "where": ["eq", ["get", "e", "/payload/topic"], ["const", event_topic]],
            },
        ],
        "negative": [],
        "aggregate": [],
        "distinct": [],
        "guard": [
            "and",
            ["eq", ["get", "a", "/payload/event_id"], ["get", "e", "/payload/id"]],
            ["eq", ["get", "a", "/payload/topic"], ["get", "e", "/payload/topic"]],
            ["eq", ["get", "a", "/payload/at"], ["get", "e", "/payload/at"]],
            ["eq", ["get", "a", "/payload/projection"], ["get", "r", "/payload/projection"]],
        ],
        "emit": [
            {
                "topic": ["const", "verify.accept"],
                "at": ["const", 0],
                "payload": ["const", None],
                "parents": [],
            }
        ],
    }
    valid = {
        "id": "verify.orient-event.02-valid",
        "positive": [{"alias": "ok", "topic": "verify.accept", "where": TRUE}],
        "negative": [],
        "aggregate": [],
        "distinct": [],
        "guard": TRUE,
        "emit": [
            {
                "topic": ["const", "verify.decision"],
                "at": ["const", 0],
                "payload": [
                    "map",
                    "valid", ["const", True],
                    "diagnostics", ["list"],
                    "metrics", ["map", "wrong_or_redundant_claims", ["const", 0]],
                ],
                "parents": [],
            }
        ],
    }
    invalid = {
        "id": "verify.orient-event.03-invalid",
        "positive": [{"alias": "a", "topic": "verify.answer", "where": TRUE}],
        "negative": [{"alias": "ok", "topic": "verify.accept", "where": TRUE}],
        "aggregate": [],
        "distinct": [],
        "guard": TRUE,
        "emit": [
            {
                "topic": ["const", "verify.decision"],
                "at": ["const", 0],
                "payload": [
                    "map",
                    "valid", ["const", False],
                    "diagnostics", [
                        "list",
                        [
                            "map",
                            "code", ["const", "claim_mismatch"],
                            "message", ["const", "submitted event fields do not identify one replayed alarm"],
                            "context", ["map", "topic", ["const", event_topic]],
                        ],
                    ],
                    "metrics", ["map"],
                ],
                "parents": [],
            }
        ],
    }
    projection = {
        "id": "verify.decision",
        "rows": [
            {
                "positive": [{"alias": "d", "topic": "verify.decision", "where": TRUE}],
                "negative": [],
                "aggregate": [],
                "distinct": [],
                "guard": TRUE,
                "value": ["get", "d", "/payload"],
                "sort": [],
            }
        ],
    }
    return {
        "format": "afterimage-validator/0.1",
        "program": {
            "semantics": "cre/0.1",
            "strata": [
                {"index": 0, "rules": [accept]},
                {"index": 1, "rules": [valid, invalid]},
            ],
        },
        "decision_projection": projection,
    }


def orient_event_answer_schema() -> dict[str, Any]:
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map",
            "required": ["event_id", "topic", "at", "projection"],
            "properties": {
                "event_id": {"type": "id"},
                "topic": {"type": "text", "min_length": 1, "max_length": 128},
                "at": {"type": "int"},
                "projection": {"type": "id"},
            },
            "additional": False,
        },
    }


def _decision_program(accept_rule: dict[str, Any], mismatch_message: str, metric_name: str = "wrong_or_redundant_claims") -> dict[str, Any]:
    valid = {
        "id": "verify.orient-replay.02-valid",
        "positive": [{"alias": "ok", "topic": "verify.accept", "where": TRUE}],
        "negative": [],
        "aggregate": [],
        "distinct": [],
        "guard": TRUE,
        "emit": [{
            "topic": ["const", "verify.decision"], "at": ["const", 0],
            "payload": ["map", "valid", ["const", True], "diagnostics", ["list"], "metrics", ["map", metric_name, ["const", 0]]],
            "parents": [],
        }],
    }
    invalid = {
        "id": "verify.orient-replay.03-invalid",
        "positive": [{"alias": "a", "topic": "verify.answer", "where": TRUE}],
        "negative": [{"alias": "ok", "topic": "verify.accept", "where": TRUE}],
        "aggregate": [], "distinct": [], "guard": TRUE,
        "emit": [{
            "topic": ["const", "verify.decision"], "at": ["const", 0],
            "payload": ["map", "valid", ["const", False], "diagnostics", ["list", ["map", "code", ["const", "claim_mismatch"], "message", ["const", mismatch_message], "context", ["map"]]], "metrics", ["map"]],
            "parents": [],
        }],
    }
    projection = {
        "id": "verify.decision",
        "rows": [{
            "positive": [{"alias": "d", "topic": "verify.decision", "where": TRUE}],
            "negative": [], "aggregate": [], "distinct": [], "guard": TRUE,
            "value": ["get", "d", "/payload"], "sort": [],
        }],
    }
    return {
        "format": "afterimage-validator/0.1",
        "program": {"semantics": "cre/0.1", "strata": [{"index": 0, "rules": [accept_rule]}, {"index": 1, "rules": [valid, invalid]}]},
        "decision_projection": projection,
    }


def compile_orient_replay_claim() -> dict[str, Any]:
    accept = {
        "id": "verify.orient-replay.01-accept",
        "positive": [
            {"alias": "a", "topic": "verify.answer", "where": TRUE},
            {"alias": "r", "topic": "verify.replay", "where": TRUE},
        ],
        "negative": [], "aggregate": [], "distinct": [],
        "guard": ["and",
            ["eq", ["get", "a", "/payload/trace_event_ids"], ["get", "r", "/payload/trace_event_ids"]],
            ["eq", ["get", "a", "/payload/records"], ["get", "r", "/payload/records"]],
            ["eq", ["get", "a", "/payload/projection"], ["get", "r", "/payload/projection"]],
        ],
        "emit": [{"topic": ["const", "verify.accept"], "at": ["const", 0], "payload": ["const", None], "parents": []}],
    }
    return _decision_program(accept, "submitted trace or replay diagnosis differs from independent replay")


def orient_replay_answer_schema() -> dict[str, Any]:
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map",
            "required": ["trace_event_ids", "records", "projection"],
            "properties": {
                "trace_event_ids": {"type": "list", "items": {"type": "id"}, "min_items": 1, "max_items": 256},
                "records": {"type": "list", "items": {"type": "map", "required": [], "properties": {}, "additional": True}, "max_items": 64},
                "projection": {"type": "id"},
            },
            "additional": False,
        },
    }


def compile_orient_observation_claim(hidden_event_topic: str) -> dict[str, Any]:
    """Compile a claim that distinguishes active state from its projection."""

    accept = {
        "id": "verify.orient-observation.01-accept",
        "positive": [
            {"alias": "a", "topic": "verify.answer", "where": TRUE},
            {"alias": "r", "topic": "verify.replay", "where": TRUE},
            {
                "alias": "h",
                "topic": "verify.active",
                "where": ["eq", ["get", "h", "/payload/topic"], ["const", hidden_event_topic]],
            },
        ],
        "negative": [],
        "aggregate": [],
        "distinct": [],
        "guard": [
            "and",
            ["eq", ["get", "a", "/payload/active_event_count"], ["get", "r", "/payload/active_event_count"]],
            ["eq", ["get", "a", "/payload/projected_records"], ["get", "r", "/payload/records"]],
            ["eq", ["get", "a", "/payload/projection"], ["get", "r", "/payload/projection"]],
            ["eq", ["get", "a", "/payload/hidden_event"], ["get", "h", "/payload/id"]],
        ],
        "emit": [{"topic": ["const", "verify.accept"], "at": ["const", 0], "payload": ["const", None], "parents": []}],
    }
    return _decision_program(accept, "submitted active-state or projection observation differs from independent replay")


def orient_observation_answer_schema() -> dict[str, Any]:
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map",
            "required": ["active_event_count", "projected_records", "projection", "hidden_event"],
            "properties": {
                "active_event_count": {"type": "int", "minimum": 1, "maximum": 4096},
                "projected_records": {
                    "type": "list",
                    "items": {"type": "map", "required": [], "properties": {}, "additional": True},
                    "max_items": 64,
                },
                "projection": {"type": "id"},
                "hidden_event": {"type": "id"},
            },
            "additional": False,
        },
    }


def compile_orient_branch_claim() -> dict[str, Any]:
    """Compile a root-versus-counterfactual replay claim contract."""

    accept = {
        "id": "verify.orient-branch.01-accept",
        "positive": [
            {"alias": "a", "topic": "verify.answer", "where": TRUE},
            {"alias": "r", "topic": "verify.replay", "where": TRUE},
            {"alias": "i", "topic": "verify.intervention", "where": TRUE},
        ],
        "negative": [],
        "aggregate": [],
        "distinct": [],
        "guard": [
            "and",
            ["eq", ["length", ["get", "i", "/payload/operations"]], ["const", 1]],
            ["eq", ["get", "a", "/payload/branch"], ["get", "r", "/payload/branch"]],
            ["eq", ["get", "a", "/payload/baseline_records"], ["get", "r", "/payload/baseline_records"]],
            ["eq", ["get", "a", "/payload/candidate_records"], ["get", "r", "/payload/records"]],
            ["eq", ["get", "a", "/payload/changed_event_ids"], ["get", "r", "/payload/changed_event_ids"]],
            ["eq", ["get", "a", "/payload/projection"], ["get", "r", "/payload/projection"]],
        ],
        "emit": [{"topic": ["const", "verify.accept"], "at": ["const", 0], "payload": ["const", None], "parents": []}],
    }
    return _decision_program(accept, "submitted branch or causal difference differs from independent root and candidate replay")


def orient_branch_answer_schema() -> dict[str, Any]:
    record_list = {
        "type": "list",
        "items": {"type": "map", "required": [], "properties": {}, "additional": True},
        "max_items": 64,
    }
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map",
            "required": ["branch", "baseline_records", "candidate_records", "changed_event_ids", "projection"],
            "properties": {
                "branch": {"type": "id"},
                "baseline_records": record_list,
                "candidate_records": record_list,
                "changed_event_ids": {"type": "list", "items": {"type": "id"}, "min_items": 1, "max_items": 256},
                "projection": {"type": "id"},
            },
            "additional": False,
        },
    }


def no_intervention_policy() -> dict[str, Any]:
    return {
        "format": "afterimage-intervention-policy/0.1",
        "required": False,
        "allowed_kinds": [],
        "max_operations": 0,
        "weights": {},
        "topics": [],
        "pointers": [],
        "retime": {"minimum": 0, "maximum": 0},
    }


def required_retime_policy(topic: str, at: int, weight: int = 1) -> dict[str, Any]:
    return {
        "format": "afterimage-intervention-policy/0.1",
        "required": True,
        "allowed_kinds": ["retime"],
        "max_operations": 1,
        "weights": {"retime": weight},
        "topics": [topic],
        "pointers": [],
        "retime": {"minimum": at, "maximum": at},
    }


def compile_orient_envelope_claim() -> dict[str, Any]:
    accept = {
        "id": "verify.orient-envelope.01-accept",
        "positive": [
            {"alias": "a", "topic": "verify.answer", "where": TRUE},
            {"alias": "e", "topic": "verify.embedded", "where": TRUE},
        ],
        "negative": [],
        "aggregate": [],
        "distinct": [],
        "guard": [
            "and",
            ["eq", ["get", "e", "/payload/valid"], ["const", True]],
            ["eq", ["get", "a", "/payload/export/case"], ["get", "e", "/payload/case"]],
            ["eq", ["get", "a", "/payload/export/bundle"], ["get", "e", "/payload/bundle"]],
            ["contains", ["get", "e", "/payload/unlocks"], ["concat", ["const", "case:"], ["get", "e", "/payload/case"]]],
        ],
        "emit": [{"topic": ["const", "verify.accept"], "at": ["const", 0], "payload": ["const", None], "parents": []}],
    }
    return _decision_program(accept, "embedded witness receipt does not bind the exported claim")


def orient_envelope_answer_schema(allowed_cases: list[str]) -> dict[str, Any]:
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map",
            "required": ["export"],
            "properties": {
                "export": {"type": "map", "required": [], "properties": {}, "additional": True},
            },
            "additional": False,
        },
        "embedded_witness": {
            "pointer": "/export",
            "allowed_cases": allowed_cases,
            "require_fact": True,
        },
    }


def compile_cascade_contract_claim() -> dict[str, Any]:
    accept = {
        "id": "verify.cascade.01-accept",
        "positive": [
            {"alias": "a", "topic": "verify.answer", "where": TRUE},
            {"alias": "r", "topic": "verify.replay", "where": TRUE},
        ],
        "negative": [], "aggregate": [], "distinct": [],
        "guard": ["and",
            ["eq", ["get", "a", "/payload/contracts"], ["get", "r", "/payload/records"]],
            ["eq", ["get", "a", "/payload/branch"], ["get", "r", "/payload/branch"]],
            ["eq", ["get", "a", "/payload/projection"], ["get", "r", "/payload/projection"]],
            ["eq", ["get", "a", "/payload/contracts/0/safe"], ["const", True]],
            ["le", ["get", "a", "/payload/contracts/0/ambulance_clear_at"], ["get", "a", "/payload/contracts/0/ambulance_deadline"]],
            ["le", ["get", "a", "/payload/contracts/0/pedestrian_wait"], ["get", "a", "/payload/contracts/0/pedestrian_max_wait"]],
        ],
        "emit": [{"topic": ["const", "verify.accept"], "at": ["const", 0], "payload": ["const", None], "parents": []}],
    }
    return _decision_program(accept, "submitted intervention does not satisfy every service contract", "contract_violations")


def compile_cascade_safe_claim() -> dict[str, Any]:
    accept = {
        "id": "verify.cascade.01-safe",
        "positive": [
            {"alias": "a", "topic": "verify.answer", "where": TRUE},
            {"alias": "r", "topic": "verify.replay", "where": TRUE},
        ],
        "negative": [], "aggregate": [], "distinct": [],
        "guard": ["and",
            ["eq", ["get", "a", "/payload/contracts"], ["get", "r", "/payload/records"]],
            ["and",
                ["eq", ["get", "a", "/payload/branch"], ["get", "r", "/payload/branch"]],
                ["and",
                    ["eq", ["get", "a", "/payload/projection"], ["get", "r", "/payload/projection"]],
                    ["eq", ["get", "a", "/payload/contracts/0/safe"], ["const", True]],
                ],
            ],
        ],
        "emit": [{"topic": ["const", "verify.accept"], "at": ["const", 0], "payload": ["const", None], "parents": []}],
    }
    return _decision_program(accept, "submitted intervention does not satisfy every service contract", "contract_violations")


def cascade_answer_schema() -> dict[str, Any]:
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map", "required": ["contracts", "branch", "projection"],
            "properties": {
                "contracts": {"type": "list", "items": {"type": "map", "required": [], "properties": {}, "additional": True}, "min_items": 1, "max_items": 1},
                "branch": {"type": "id"}, "projection": {"type": "id"},
            },
            "additional": False,
        },
    }


def cascade_projection_answer_schema() -> dict[str, Any]:
    open_map = {"type": "map", "required": [], "properties": {}, "additional": True}
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map",
            "required": ["contract", "baseline_records", "public_rows", "branch", "projection"],
            "properties": {
                "contract": open_map,
                "baseline_records": {"type": "list", "items": open_map, "max_items": 16},
                "public_rows": {"type": "list", "items": open_map, "max_items": 16},
                "branch": {"type": "id"},
                "projection": {"type": "id"},
            },
            "additional": False,
        },
    }


def cascade_proof_answer_schema() -> dict[str, Any]:
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map", "required": ["contract", "public_rows", "relay", "branch", "projection"],
            "properties": {
                "contract": {"type": "map", "required": [], "properties": {}, "additional": True},
                "public_rows": {"type": "list", "items": {"type": "map", "required": [], "properties": {}, "additional": True}},
                "relay": {"type": "map", "required": [], "properties": {}, "additional": True},
                "branch": {"type": "id"}, "projection": {"type": "id"},
            },
            "additional": False,
        },
    }


def compile_host_family_claim(metric_name: str) -> dict[str, Any]:
    accept = {
        "id": "verify.family.01-accept",
        "positive": [{"alias": "f", "topic": "verify.family", "where": ["eq", ["get", "f", "/payload/valid"], ["const", True]]}],
        "negative": [], "aggregate": [], "distinct": [], "guard": TRUE,
        "emit": [{"topic": ["const", "verify.accept"], "at": ["const", 0], "payload": ["const", None], "parents": []}],
    }
    return _decision_program(accept, "family certificate failed independent verification", metric_name)


def merge_answer_schema() -> dict[str, Any]:
    return {"format": "afterimage-answer-schema/0.1", "schema": {"type": "map", "required": ["accepted", "rejected", "certificate"], "properties": {
        "accepted": {"type": "list", "items": {"type": "map", "required": [], "properties": {}, "additional": True}},
        "rejected": {"type": "list", "items": {"type": "map", "required": [], "properties": {}, "additional": True}},
        "certificate": {"type": "list", "items": {"type": "map", "required": [], "properties": {}, "additional": True}},
    }, "additional": False}}


def pulse_answer_schema() -> dict[str, Any]:
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map",
            "required": ["program"],
            "properties": {
                "program": {"type": "map", "required": [], "properties": {}, "additional": True},
                "invariant": {"type": "text", "max_length": 4096},
            },
            "additional": False,
        },
    }


def mosaic_answer_schema() -> dict[str, Any]:
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map", "required": ["global", "used", "unused"],
            "properties": {
                "global": {"type": "map", "required": [], "properties": {}, "additional": True},
                "used": {"type": "list", "items": {"type": "map", "required": [], "properties": {}, "additional": True}},
                "unused": {"type": "list", "items": {"type": "map", "required": [], "properties": {}, "additional": True}},
                "missing": {"type": "map", "required": [], "properties": {}, "additional": True},
            },
            "additional": False,
        },
    }


def lens_answer_schema() -> dict[str, Any]:
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {"type": "map", "required": ["program"], "properties": {"program": {"type": "map", "required": [], "properties": {}, "additional": True}}, "additional": False},
    }


def covenant_answer_schema() -> dict[str, Any]:
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map",
            "required": ["policy", "claimed_response_bound"],
            "properties": {
                "policy": {"type": "map", "required": [], "properties": {}, "additional": True},
                "claimed_response_bound": {"type": "int", "minimum": 0},
            },
            "additional": False,
        },
    }


def paradox_answer_schema() -> dict[str, Any]:
    open_map = {"type": "map", "required": [], "properties": {}, "additional": True}
    return {
        "format": "afterimage-answer-schema/0.1",
        "schema": {
            "type": "map",
            "required": ["left_history", "right_history", "equivalence", "safety_evidence", "latent_difference"],
            "properties": {
                "left_history": open_map,
                "right_history": open_map,
                "equivalence": {"type": "list", "items": open_map},
                "safety_evidence": open_map,
                "latent_difference": {"type": "list", "items": {"type": "id"}},
            },
            "additional": False,
        },
    }


def orient_score(reference_scale: int, max_witness_units: int = 4096, grants: list[str] | None = None) -> dict[str, Any]:
    result = {
        "format": "afterimage-score/0.1",
        "family": "ORIENT",
        "reference_scale": reference_scale,
        "metric_bounds": {"witness_units": max_witness_units},
    }
    if grants:
        result["grants"] = grants
    return result


def cascade_score(
    reference_scale: int,
    max_causal_footprint: int = 4096,
    max_witness_units: int = 4096,
    diagnostic_topics: list[str] | None = None,
    proof_contract: dict[str, Any] | None = None,
    projection_contract: dict[str, Any] | None = None,
    minimal_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "format": "afterimage-score/0.1",
        "family": "CASCADE",
        "reference_scale": reference_scale,
        "metric_bounds": {
            "causal_footprint": max_causal_footprint,
            "witness_units": max_witness_units,
        },
        "diagnostic_topics": diagnostic_topics or [],
    }
    if proof_contract is not None:
        result["proof_contract"] = proof_contract
    if projection_contract is not None:
        result["projection_contract"] = projection_contract
    if minimal_contract is not None:
        result["minimal_contract"] = minimal_contract
    return result


def merge_score(
    reference_scale: int,
    max_temporal_displacement: int,
    max_certificate_units: int,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {"format": "afterimage-score/0.1", "family": "MERGE", "reference_scale": reference_scale, "metric_bounds": {"temporal_displacement": max_temporal_displacement, "certificate_units": max_certificate_units}}
    if contract is not None:
        result["contract"] = contract
    return result


def pulse_score(reference_scale: int, max_worst_latency: int, max_live_state_cells: int, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "afterimage-score/0.1",
        "family": "PULSE",
        "reference_scale": reference_scale,
        "metric_bounds": {"worst_latency": max_worst_latency, "live_state_cells": max_live_state_cells},
        "contract": contract,
    }


def mosaic_score(reference_scale: int, max_graph_size: int, max_certificate_units: int, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "afterimage-score/0.1",
        "family": "MOSAIC",
        "reference_scale": reference_scale,
        "metric_bounds": {"graph_size": max_graph_size, "certificate_units": max_certificate_units},
        "contract": contract,
    }


def lens_score(reference_scale: int, max_auxiliary_schema_cells: int, max_worst_reductions: int, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "afterimage-score/0.1",
        "family": "LENS",
        "reference_scale": reference_scale,
        "metric_bounds": {"auxiliary_schema_cells": max_auxiliary_schema_cells, "worst_reductions": max_worst_reductions},
        "contract": contract,
    }


def covenant_score(reference_scale: int, max_worst_response_bound: int, max_reachable_states: int, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "afterimage-score/0.1",
        "family": "COVENANT",
        "reference_scale": reference_scale,
        "metric_bounds": {
            "worst_response_bound": max_worst_response_bound,
            "reachable_states": max_reachable_states,
        },
        "contract": contract,
    }


def paradox_score(reference_scale: int, max_latent_difference_weight: int, max_proof_steps: int, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "afterimage-score/0.1",
        "family": "PARADOX",
        "reference_scale": reference_scale,
        "metric_bounds": {
            "latent_difference_weight": max_latent_difference_weight,
            "proof_steps": max_proof_steps,
        },
        "contract": contract,
    }
