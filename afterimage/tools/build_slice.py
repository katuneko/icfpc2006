#!/usr/bin/env python3
"""Compile authored case sources into a reproducible Afterimage slice bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import authoring  # noqa: E402
import cre  # noqa: E402
import paradox as paradox_runtime  # noqa: E402
import verify_witness as verifier  # noqa: E402


DEFAULT_MANIFEST = ROOT / "manifests" / "vertical_slice.json"
DEFAULT_SOURCE_ROOT = ROOT / "content" / "vertical_slice" / "cases"
SOURCE_KEYS = {"format", "case", "events", "rules", "projection", "worked_value", "story"}
CASE_SOURCE_COMMON_KEYS = {
    "id",
    "family",
    "title",
    "points",
    "reference_scale",
    "requires",
    "projection",
    "validator_template",
    "limits",
}
CASE_SOURCE_TEMPLATE_KEYS = {
    "orient-event-claim/0.1": {"answer_event_topic"},
    "orient-replay-claim/0.1": set(),
    "orient-observation-claim/0.1": {"hidden_event_topic"},
    "orient-branch-claim/0.1": {"retime_topic", "retime_at"},
    "orient-envelope-claim/0.1": {"export_cases", "author_export_case", "grant"},
    "cascade-contract-claim/0.1": {"intervention_policy", "author_operation", "diagnostic_topics", "metric_bounds"},
    "cascade-safe-claim/0.1": {"intervention_policy", "author_operation", "diagnostic_topics", "metric_bounds"},
    "cascade-multi-contract-claim/0.1": {"intervention_policy", "author_operations", "diagnostic_topics", "metric_bounds"},
    "cascade-proof-claim/0.1": {"intervention_policy", "author_relay_topic", "diagnostic_topics", "metric_bounds", "proof_contract"},
    "cascade-projection-safe-claim/0.1": {"intervention_policy", "author_operation", "diagnostic_topics", "metric_bounds", "projection_contract"},
    "cascade-minimal-explanation-claim/0.1": {"intervention_policy", "author_operations", "diagnostic_topics", "metric_bounds", "minimal_contract"},
    "merge-certificate-claim/0.1": {"author_answer", "metric_bounds"},
    "merge-dedup-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "merge-split-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "merge-quorum-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "merge-offset-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "merge-minimal-conflict-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "merge-weighted-cut-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "merge-echo-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "merge-weighted-evidence-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "merge-causal-compression-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "merge-partial-order-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "merge-equivocation-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "merge-non-unique-certificate-claim/0.1": {"author_answer", "merge_contract", "metric_bounds"},
    "pulse-program-claim/0.1": {"author_program", "author_invariant", "pulse_contract", "metric_bounds"},
    "mosaic-graph-claim/0.1": {"author_answer", "mosaic_contract", "metric_bounds"},
    "lens-law-claim/0.1": {"author_program", "lens_contract", "metric_bounds"},
    "covenant-policy-claim/0.1": {"author_policy", "author_response_bound", "covenant_contract", "metric_bounds"},
    "paradox-paired-history-claim/0.1": {"history_case", "author_histories", "paradox_contract", "metric_bounds"},
}


class BuildError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def write_json(root: Path, relative: str, value: Any) -> None:
    destination = root.joinpath(*relative.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(cre.canonical_bytes(value))


def write_text(root: Path, relative: str, value: str) -> None:
    destination = root.joinpath(*relative.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(value, encoding="utf-8", newline="\n")


def load_design_manifest(path: Path) -> dict[str, Any]:
    value = cre.load_json(path)
    if "include" not in value:
        return value
    include = value["include"]
    require(isinstance(include, str) and include, f"{path}: include must be a path")
    base = load_design_manifest((path.parent / include).resolve())
    require(isinstance(base.get("cases"), list) and isinstance(value.get("cases"), list), f"{path}: included manifests need case lists")
    merged = {**base, **value, "cases": [*base["cases"], *value["cases"]]}
    merged.pop("include", None)
    return merged


def load_sources(
    manifest_path: Path = DEFAULT_MANIFEST,
    source_roots: list[Path] | None = None,
) -> list[dict[str, Any]]:
    design_manifest = load_design_manifest(manifest_path)
    designed = {item["id"]: item for item in design_manifest["cases"]}
    sources = []
    configured_roots = source_roots
    if configured_roots is None and "source_roots" in design_manifest:
        require(isinstance(design_manifest["source_roots"], list), f"{manifest_path}: source_roots must be a list")
        configured_roots = [(manifest_path.parent / item).resolve() for item in design_manifest["source_roots"]]
    configured_roots = configured_roots or [DEFAULT_SOURCE_ROOT]
    paths = [path for root in configured_roots for path in root.glob("*.json")]
    for path in sorted(paths, key=lambda item: str(item).encode("utf-8")):
        try:
            value = cre.load_json(path)
        except cre.CREError as exc:
            raise BuildError(f"{path}: {exc.code}: {exc.message}") from exc
        require(isinstance(value, dict) and set(value) == SOURCE_KEYS, f"{path}: source fields are invalid")
        require(value["format"] == "afterimage-case-source/0.1", f"{path}: source format is invalid")
        case = value["case"]
        require(isinstance(case, dict), f"{path}: case source must be a map")
        template = case.get("validator_template")
        require(template in CASE_SOURCE_TEMPLATE_KEYS, f"{path}: unsupported validator template")
        expected_keys = CASE_SOURCE_COMMON_KEYS | CASE_SOURCE_TEMPLATE_KEYS[template]
        require(set(case) == expected_keys, f"{path}: case source fields are invalid")
        require(path.stem == case["id"], f"{path}: filename and case ID differ")
        require(case["id"] in designed, f"{path}: case is absent from vertical_slice.json")
        design_case = designed[case["id"]]
        for field in ("family", "title", "points", "reference_scale", "requires"):
            require(
                cre.same_value(case[field], design_case[field]),
                f"{path}: {field} differs from vertical_slice.json",
            )
        require(isinstance(value["events"], list) and isinstance(value["rules"], list), f"{path}: events/rules must be lists")
        require(isinstance(value["projection"], dict) and value["projection"].get("id") == case["projection"], f"{path}: projection ID differs")
        sources.append(value)
    require(bool(sources), "no authored case sources found")
    ids = [source["case"]["id"] for source in sources]
    require(len(ids) == len(set(ids)), "duplicate case source ID")
    require(set(ids) == set(designed), f"source/manifest case mismatch: missing={sorted(set(designed) - set(ids))}, extra={sorted(set(ids) - set(designed))}")
    return sources


def _event_references(value: Any) -> set[str]:
    if isinstance(value, str) and value.startswith("@"):
        return {value[1:]}
    if isinstance(value, list):
        return set().union(*(_event_references(item) for item in value), set())
    if isinstance(value, dict):
        return set().union(*(_event_references(item) for item in value.values()), set())
    return set()


def _replace_event_references(value: Any, identifiers: dict[str, str]) -> Any:
    if isinstance(value, str) and value.startswith("@"):
        return identifiers[value[1:]]
    if isinstance(value, list):
        return [_replace_event_references(item, identifiers) for item in value]
    if isinstance(value, dict):
        return {key: _replace_event_references(item, identifiers) for key, item in value.items()}
    return value


def compile_base_events(case_id: str, source_events: list[Any]) -> list[dict[str, Any]]:
    """Resolve stable author labels, then construct and validate base events."""

    pending: list[tuple[str | None, dict[str, Any]]] = []
    labels: set[str] = set()
    for index, item in enumerate(source_events):
        label = None
        body = item
        if isinstance(item, dict) and set(item) == {"label", "body"}:
            label, body = item["label"], item["body"]
            require(
                isinstance(label, str) and label and not label.startswith("@") and label not in labels,
                f"{case_id}: event {index} has a duplicate or invalid label",
            )
            labels.add(label)
        require(isinstance(body, dict), f"{case_id}: event {index} body must be a map")
        pending.append((label, body))

    identifiers: dict[str, str] = {}
    compiled: list[dict[str, Any]] = []
    while pending:
        progressed = False
        next_pending = []
        for label, body in pending:
            references = _event_references(body)
            unknown = references - labels
            if unknown:
                raise BuildError(f"{case_id}: event references unknown label @{sorted(unknown)[0]}")
            if not references.issubset(identifiers):
                next_pending.append((label, body))
                continue
            try:
                event = cre.make_event(_replace_event_references(body, identifiers))
            except cre.CREError as exc:
                raise BuildError(f"{case_id}: invalid base event: {exc.code}: {exc.message}") from exc
            compiled.append(event)
            if label is not None:
                identifiers[label] = event["id"]
            progressed = True
        require(progressed, f"{case_id}: event label references contain a cycle")
        pending = next_pending
    return compiled


def story_markdown(source: dict[str, Any]) -> str:
    case = source["case"]
    story = source["story"]
    require(
        isinstance(story, dict) and set(story) == {"premise", "submission", "diagnostics", "hints"},
        f"{case['id']}: story fields are invalid",
    )
    require(all(isinstance(story[key], str) and story[key] for key in ("premise", "submission", "diagnostics")), f"{case['id']}: story prose is missing")
    require(isinstance(story["hints"], list) and all(isinstance(item, str) and item for item in story["hints"]), f"{case['id']}: hints are invalid")
    hints = "\n".join(f"{index}. {hint}" for index, hint in enumerate(story["hints"], 1))
    return (
        f"# {case['id']} — {case['title']}\n\n"
        f"{story['premise']}\n\n"
        f"## Submission\n\n{story['submission']}\n\n"
        f"## Diagnostics\n\n{story['diagnostics']}\n\n"
        f"## Hints\n\n{hints}\n"
    )


def compile_logical(root: Path, sources: list[dict[str, Any]]) -> dict[str, Any]:
    descriptors = []
    rule_ids: set[str] = set()
    projection_ids: set[str] = set()
    total_events = 0
    world_paths: dict[str, str] = {}
    case_projections: dict[str, str] = {}
    case_projection_values: dict[str, dict[str, Any]] = {}

    for source in sources:
        case = source["case"]
        case_id = case["id"]
        require(isinstance(case_id, str) and isinstance(case["title"], str), f"{case_id}: identity fields invalid")
        require(isinstance(case["points"], int) and case["points"] > 0, f"{case_id}: points invalid")
        require(isinstance(case["reference_scale"], int) and case["reference_scale"] > 0, f"{case_id}: reference scale invalid")
        if case["validator_template"] == "orient-event-claim/0.1":
            require(isinstance(case["answer_event_topic"], str) and case["answer_event_topic"], f"{case_id}: answer event topic invalid")
        if case["validator_template"] == "orient-observation-claim/0.1":
            require(isinstance(case["hidden_event_topic"], str) and case["hidden_event_topic"], f"{case_id}: hidden event topic invalid")
        if case["validator_template"] == "orient-branch-claim/0.1":
            require(isinstance(case["retime_topic"], str) and case["retime_topic"], f"{case_id}: retime topic invalid")
            require(isinstance(case["retime_at"], int) and not isinstance(case["retime_at"], bool), f"{case_id}: retime target invalid")
        if case["validator_template"] == "orient-envelope-claim/0.1":
            require(
                isinstance(case["export_cases"], list)
                and case["export_cases"]
                and len(set(case["export_cases"])) == len(case["export_cases"])
                and all(isinstance(item, str) and item for item in case["export_cases"]),
                f"{case_id}: export cases invalid",
            )
            require(case["author_export_case"] in case["export_cases"], f"{case_id}: author export case invalid")
            require(isinstance(case["grant"], str) and case["grant"].startswith("cap:"), f"{case_id}: capability grant invalid")
        if case["validator_template"] in {"cascade-contract-claim/0.1", "cascade-safe-claim/0.1", "cascade-multi-contract-claim/0.1", "cascade-proof-claim/0.1", "cascade-projection-safe-claim/0.1", "cascade-minimal-explanation-claim/0.1"}:
            require(isinstance(case["intervention_policy"], dict), f"{case_id}: intervention policy invalid")
            require(isinstance(case["diagnostic_topics"], list), f"{case_id}: diagnostic topics invalid")
            require(isinstance(case["metric_bounds"], dict) and set(case["metric_bounds"]) == {"causal_footprint", "witness_units"}, f"{case_id}: metric bounds invalid")
        if case["validator_template"] in {"cascade-contract-claim/0.1", "cascade-safe-claim/0.1", "cascade-projection-safe-claim/0.1"}:
            require(isinstance(case["author_operation"], dict), f"{case_id}: author operation invalid")
        if case["validator_template"] in {"cascade-multi-contract-claim/0.1", "cascade-minimal-explanation-claim/0.1"}:
            require(isinstance(case["author_operations"], list) and case["author_operations"], f"{case_id}: author operations invalid")
        if case["validator_template"] == "cascade-proof-claim/0.1":
            require(isinstance(case["author_relay_topic"], str) and case["author_relay_topic"], f"{case_id}: author relay topic invalid")
            require(isinstance(case["proof_contract"], dict), f"{case_id}: proof contract invalid")
        if case["validator_template"] == "cascade-projection-safe-claim/0.1":
            require(isinstance(case["projection_contract"], dict), f"{case_id}: projection contract invalid")
        if case["validator_template"] == "cascade-minimal-explanation-claim/0.1":
            require(isinstance(case["minimal_contract"], dict), f"{case_id}: minimal explanation contract invalid")
        if case["validator_template"] in {"merge-certificate-claim/0.1", "merge-dedup-certificate-claim/0.1", "merge-split-certificate-claim/0.1", "merge-quorum-certificate-claim/0.1", "merge-offset-certificate-claim/0.1", "merge-minimal-conflict-certificate-claim/0.1", "merge-weighted-cut-certificate-claim/0.1", "merge-echo-certificate-claim/0.1", "merge-weighted-evidence-certificate-claim/0.1", "merge-causal-compression-certificate-claim/0.1", "merge-partial-order-certificate-claim/0.1", "merge-equivocation-certificate-claim/0.1", "merge-non-unique-certificate-claim/0.1"}:
            require(isinstance(case["author_answer"], dict), f"{case_id}: author answer invalid")
            require(isinstance(case["metric_bounds"], dict) and set(case["metric_bounds"]) == {"temporal_displacement", "certificate_units"}, f"{case_id}: metric bounds invalid")
        if case["validator_template"] in {"merge-dedup-certificate-claim/0.1", "merge-split-certificate-claim/0.1", "merge-quorum-certificate-claim/0.1", "merge-offset-certificate-claim/0.1", "merge-minimal-conflict-certificate-claim/0.1", "merge-weighted-cut-certificate-claim/0.1", "merge-echo-certificate-claim/0.1", "merge-weighted-evidence-certificate-claim/0.1", "merge-causal-compression-certificate-claim/0.1", "merge-partial-order-certificate-claim/0.1", "merge-equivocation-certificate-claim/0.1", "merge-non-unique-certificate-claim/0.1"}:
            require(isinstance(case["merge_contract"], dict), f"{case_id}: MERGE contract invalid")
        if case["validator_template"] == "pulse-program-claim/0.1":
            require(isinstance(case["author_program"], dict), f"{case_id}: author program invalid")
            require(isinstance(case["author_invariant"], str), f"{case_id}: author invariant invalid")
            require(isinstance(case["pulse_contract"], dict), f"{case_id}: PULSE contract invalid")
            require(isinstance(case["metric_bounds"], dict) and set(case["metric_bounds"]) == {"worst_latency", "live_state_cells"}, f"{case_id}: metric bounds invalid")
        if case["validator_template"] == "mosaic-graph-claim/0.1":
            require(isinstance(case["author_answer"], dict), f"{case_id}: author answer invalid")
            require(isinstance(case["mosaic_contract"], dict), f"{case_id}: MOSAIC contract invalid")
            require(isinstance(case["metric_bounds"], dict) and set(case["metric_bounds"]) == {"graph_size", "certificate_units"}, f"{case_id}: metric bounds invalid")
        if case["validator_template"] == "lens-law-claim/0.1":
            require(isinstance(case["author_program"], dict), f"{case_id}: author program invalid")
            require(isinstance(case["lens_contract"], dict), f"{case_id}: LENS contract invalid")
            require(isinstance(case["metric_bounds"], dict) and set(case["metric_bounds"]) == {"auxiliary_schema_cells", "worst_reductions"}, f"{case_id}: metric bounds invalid")
        if case["validator_template"] == "covenant-policy-claim/0.1":
            require(isinstance(case["author_policy"], dict), f"{case_id}: author policy invalid")
            require(isinstance(case["author_response_bound"], int) and not isinstance(case["author_response_bound"], bool) and case["author_response_bound"] >= 0, f"{case_id}: author response bound invalid")
            require(isinstance(case["covenant_contract"], dict), f"{case_id}: COVENANT contract invalid")
            require(isinstance(case["metric_bounds"], dict) and set(case["metric_bounds"]) == {"worst_response_bound", "reachable_states"}, f"{case_id}: metric bounds invalid")
        if case["validator_template"] == "paradox-paired-history-claim/0.1":
            require(isinstance(case["history_case"], str) and case["history_case"], f"{case_id}: history case invalid")
            require(isinstance(case["author_histories"], dict) and set(case["author_histories"]) == {"left", "right"}, f"{case_id}: author histories invalid")
            require(isinstance(case["paradox_contract"], dict), f"{case_id}: PARADOX contract invalid")
            require(case["paradox_contract"].get("history_case") == case["history_case"], f"{case_id}: public PARADOX history case differs from author source")
            require(isinstance(case["metric_bounds"], dict) and set(case["metric_bounds"]) == {"latent_difference_weight", "proof_steps"}, f"{case_id}: metric bounds invalid")
            require(not source["events"] and not source["rules"], f"{case_id}: shared-world PARADOX source must not duplicate events or rules")
        case_events = compile_base_events(case_id, source["events"])
        identifiers = [event["id"] for event in case_events]
        require(len(identifiers) == len(set(identifiers)), f"{case_id}: base events duplicate")
        try:
            ordered_events = cre.validate_base_events(case_events, cre.Counters.create(None))
        except cre.CREError as exc:
            raise BuildError(f"{case_id}: base events invalid: {exc.code}: {exc.message}") from exc

        case_rules: dict[int, list[dict[str, Any]]] = {}
        for item in source["rules"]:
            require(isinstance(item, dict) and set(item) == {"stratum", "rule"}, f"{case_id}: authored rule wrapper invalid")
            index = item["stratum"]
            rule = item["rule"]
            require(isinstance(index, int) and index >= 0 and isinstance(rule, dict), f"{case_id}: authored rule invalid")
            require(isinstance(rule.get("id"), str) and rule["id"] not in rule_ids, f"{case_id}: duplicate/invalid rule ID")
            rule_ids.add(rule["id"])
            case_rules.setdefault(index, []).append(rule)
        if case_rules:
            highest = max(case_rules)
            require(set(case_rules) == set(range(highest + 1)), f"{case_id}: rule strata must be contiguous")
            strata = [
                {"index": index, "rules": sorted(case_rules[index], key=lambda value: value["id"].encode("utf-8"))}
                for index in range(highest + 1)
            ]
        else:
            strata = []

        projection = source["projection"]
        if case["validator_template"] == "paradox-paired-history-claim/0.1":
            history_case = case["history_case"]
            require(history_case in world_paths, f"{case_id}: shared history case has not been compiled")
            require(projection["id"] == case_projections[history_case], f"{case_id}: projection differs from shared history case")
            require(cre.same_value(projection, case_projection_values[history_case]), f"{case_id}: shared projection body differs from history case")
            world_path = world_paths[history_case]
        else:
            require(projection["id"] not in projection_ids, f"{case_id}: duplicate projection ID")
            projection_ids.add(projection["id"])
            world_root = f"worlds/{case_id}"
            program_path = f"{world_root}/program.cre.json"
            events_path = f"{world_root}/base.ndjson"
            projection_index_path = f"{world_root}/projections/index.json"
            projection_path = f"{world_root}/projections/{projection['id']}.cre.json"
            world_path = f"{world_root}/world.json"
            write_json(root, program_path, {"semantics": "cre/0.1", "strata": strata})
            event_bytes = b"".join(cre.canonical_bytes(cre.event_view(event)) + b"\n" for event in ordered_events)
            event_destination = root.joinpath(*events_path.split("/"))
            event_destination.parent.mkdir(parents=True, exist_ok=True)
            event_destination.write_bytes(event_bytes)
            write_json(root, projection_path, projection)
            write_json(
                root,
                projection_index_path,
                {"format": "afterimage-projections/0.1", "projections": [{"id": projection["id"], "path": projection_path}]},
            )
            write_json(
                root,
                world_path,
                {
                    "format": "afterimage-case-world/0.1",
                    "program": program_path,
                    "events": events_path,
                    "projections": projection_index_path,
                },
            )
            world_paths[case_id] = world_path
            case_projections[case_id] = projection["id"]
            case_projection_values[case_id] = projection

        case_root = f"cases/{case_id}"
        answer_schema_path = f"{case_root}/answer.schema.json"
        validator_path = f"{case_root}/validator.cre.json"
        policy_path = f"{case_root}/interventions.json"
        score_path = f"{case_root}/score.json"
        if case["validator_template"] == "orient-event-claim/0.1":
            answer_schema = authoring.orient_event_answer_schema()
            validator_program = authoring.compile_orient_event_claim(case["answer_event_topic"])
        elif case["validator_template"] == "orient-replay-claim/0.1":
            answer_schema = authoring.orient_replay_answer_schema()
            validator_program = authoring.compile_orient_replay_claim()
        elif case["validator_template"] == "orient-observation-claim/0.1":
            answer_schema = authoring.orient_observation_answer_schema()
            validator_program = authoring.compile_orient_observation_claim(case["hidden_event_topic"])
        elif case["validator_template"] == "orient-branch-claim/0.1":
            answer_schema = authoring.orient_branch_answer_schema()
            validator_program = authoring.compile_orient_branch_claim()
        elif case["validator_template"] == "orient-envelope-claim/0.1":
            answer_schema = authoring.orient_envelope_answer_schema(case["export_cases"])
            validator_program = authoring.compile_orient_envelope_claim()
        elif case["validator_template"] == "cascade-contract-claim/0.1":
            answer_schema = authoring.cascade_answer_schema()
            validator_program = authoring.compile_cascade_contract_claim()
        elif case["validator_template"] == "cascade-safe-claim/0.1":
            answer_schema = authoring.cascade_answer_schema()
            validator_program = authoring.compile_cascade_safe_claim()
        elif case["validator_template"] == "cascade-multi-contract-claim/0.1":
            answer_schema = authoring.cascade_answer_schema()
            validator_program = authoring.compile_cascade_safe_claim()
        elif case["validator_template"] == "cascade-proof-claim/0.1":
            answer_schema = authoring.cascade_proof_answer_schema()
            validator_program = authoring.compile_host_family_claim("proof_failures")
        elif case["validator_template"] == "cascade-projection-safe-claim/0.1":
            answer_schema = authoring.cascade_projection_answer_schema()
            validator_program = authoring.compile_host_family_claim("projection_failures")
        elif case["validator_template"] == "cascade-minimal-explanation-claim/0.1":
            answer_schema = authoring.cascade_answer_schema()
            validator_program = authoring.compile_host_family_claim("minimality_failures")
        elif case["validator_template"] in {"merge-certificate-claim/0.1", "merge-dedup-certificate-claim/0.1", "merge-split-certificate-claim/0.1", "merge-quorum-certificate-claim/0.1", "merge-offset-certificate-claim/0.1", "merge-minimal-conflict-certificate-claim/0.1", "merge-weighted-cut-certificate-claim/0.1", "merge-echo-certificate-claim/0.1", "merge-weighted-evidence-certificate-claim/0.1", "merge-causal-compression-certificate-claim/0.1", "merge-partial-order-certificate-claim/0.1", "merge-equivocation-certificate-claim/0.1", "merge-non-unique-certificate-claim/0.1"}:
            answer_schema = authoring.merge_answer_schema()
            validator_program = authoring.compile_host_family_claim("certificate_failures")
        elif case["validator_template"] == "pulse-program-claim/0.1":
            answer_schema = authoring.pulse_answer_schema()
            validator_program = authoring.compile_host_family_claim("program_failures")
        elif case["validator_template"] == "mosaic-graph-claim/0.1":
            answer_schema = authoring.mosaic_answer_schema()
            validator_program = authoring.compile_host_family_claim("certificate_failures")
        elif case["validator_template"] == "lens-law-claim/0.1":
            answer_schema = authoring.lens_answer_schema()
            validator_program = authoring.compile_host_family_claim("law_failures")
        elif case["validator_template"] == "covenant-policy-claim/0.1":
            answer_schema = authoring.covenant_answer_schema()
            validator_program = authoring.compile_host_family_claim("policy_failures")
        elif case["validator_template"] == "paradox-paired-history-claim/0.1":
            answer_schema = authoring.paradox_answer_schema()
            validator_program = authoring.compile_host_family_claim("paradox_failures")
        else:  # Defensive: load_sources already rejects unknown templates.
            raise BuildError(f"{case_id}: unsupported validator template")
        write_json(root, answer_schema_path, answer_schema)
        write_json(root, validator_path, validator_program)
        if case["validator_template"] in {"cascade-contract-claim/0.1", "cascade-safe-claim/0.1", "cascade-multi-contract-claim/0.1", "cascade-proof-claim/0.1", "cascade-projection-safe-claim/0.1", "cascade-minimal-explanation-claim/0.1"}:
            policy = case["intervention_policy"]
        elif case["validator_template"] == "orient-branch-claim/0.1":
            policy = authoring.required_retime_policy(case["retime_topic"], case["retime_at"])
        else:
            policy = authoring.no_intervention_policy()
        write_json(root, policy_path, policy)
        if case["family"] == "CASCADE":
            score = authoring.cascade_score(
                case["reference_scale"],
                max_causal_footprint=case["metric_bounds"]["causal_footprint"],
                max_witness_units=case["metric_bounds"]["witness_units"],
                diagnostic_topics=case["diagnostic_topics"],
                proof_contract=case.get("proof_contract"),
                projection_contract=case.get("projection_contract"),
                minimal_contract=case.get("minimal_contract"),
            )
        elif case["family"] == "MERGE":
            score = authoring.merge_score(
                case["reference_scale"],
                case["metric_bounds"]["temporal_displacement"],
                case["metric_bounds"]["certificate_units"],
                case.get("merge_contract"),
            )
        elif case["family"] == "PULSE":
            score = authoring.pulse_score(case["reference_scale"], case["metric_bounds"]["worst_latency"], case["metric_bounds"]["live_state_cells"], case["pulse_contract"])
        elif case["family"] == "MOSAIC":
            score = authoring.mosaic_score(case["reference_scale"], case["metric_bounds"]["graph_size"], case["metric_bounds"]["certificate_units"], case["mosaic_contract"])
        elif case["family"] == "LENS":
            score = authoring.lens_score(case["reference_scale"], case["metric_bounds"]["auxiliary_schema_cells"], case["metric_bounds"]["worst_reductions"], case["lens_contract"])
        elif case["family"] == "COVENANT":
            score = authoring.covenant_score(case["reference_scale"], case["metric_bounds"]["worst_response_bound"], case["metric_bounds"]["reachable_states"], case["covenant_contract"])
        elif case["family"] == "PARADOX":
            score = authoring.paradox_score(case["reference_scale"], case["metric_bounds"]["latent_difference_weight"], case["metric_bounds"]["proof_steps"], case["paradox_contract"])
        else:
            grants = [case["grant"]] if case["validator_template"] == "orient-envelope-claim/0.1" else None
            score = authoring.orient_score(case["reference_scale"], grants=grants)
        write_json(root, score_path, score)
        write_json(root, f"fixtures/{case_id}/worked-value.json", source["worked_value"])
        write_text(root, f"story/{case_id}.md", story_markdown(source))
        descriptors.append(
            {
                "id": case_id,
                "family": case["family"],
                "title": case["title"],
                "points": case["points"],
                "requires": case["requires"],
                "input_branch": "root",
                "world": world_path,
                "projection": case["projection"],
                "answer_schema": answer_schema_path,
                "validator": validator_path,
                "intervention_policy": policy_path,
                "score": score_path,
                "limits": case["limits"],
            }
        )
        total_events += len(case_events)

    # The required global world is reserved for the Continuity Desk. Case
    # fixtures remain isolated under worlds/<CASE-ID>/.
    write_json(root, "program/continuity.cre.json", {"semantics": "cre/0.1", "strata": []})
    global_events = root / "events/base.ndjson"
    global_events.parent.mkdir(parents=True, exist_ok=True)
    global_events.write_bytes(b"")
    write_json(root, "projections/index.json", {"format": "afterimage-projections/0.1", "projections": []})
    descriptors.sort(key=lambda item: item["id"].encode("utf-8"))
    write_json(root, "cases/index.json", {"format": "afterimage-cases/0.1", "cases": descriptors})
    write_json(root, "fixtures/conformance/index.json", {"format": "afterimage-conformance-index/0.1", "cases": []})
    return {"cases": len(descriptors), "base_events": total_events, "rules": len(rule_ids), "projections": len(projection_ids)}


def build_author_baseline(
    bundle: kit.ValidatedBundle,
    destination: Path,
    sources: list[dict[str, Any]],
    designed_order: list[str],
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise BuildError(f"author output already exists: {destination}")
    destination.mkdir(parents=True)
    world_path = destination / "world"
    kit.extract_bundle(bundle, world_path)
    world = kit.verify_world(world_path)
    receipts = []
    facts: set[str] = set()
    descriptor_by_id = {item["id"]: item for item in world.json_values["cases/index.json"]["cases"]}
    sources_by_id = {item["case"]["id"]: item for item in sources}
    for case_id in designed_order:
        if case_id not in descriptor_by_id:
            continue
        descriptor_value = descriptor_by_id[case_id]
        case = verifier.validate_case_descriptor(descriptor_value, world)
        source = sources_by_id[case["id"]]
        template = source["case"]["validator_template"]
        parent = cre.root_branch_id(world.bundle)
        intervention = None
        operations: list[dict[str, Any]] = []
        if template == "orient-branch-claim/0.1":
            logical = kit.resolve_logical_world(world, case["world"])
            topic = source["case"]["retime_topic"]
            targets = [event for event in logical.base_events if event["topic"] == topic]
            require(len(targets) == 1, f"{case['id']}: author retime topic must identify one base event")
            operations = [{"kind": "retime", "event": targets[0]["id"], "at": source["case"]["retime_at"]}]
            intervention = {
                "format": "afterimage-intervention/0.1",
                "bundle": world.bundle,
                "parent_branch": parent,
                "case": case["id"],
                "operations": operations,
            }
        elif template in {"cascade-contract-claim/0.1", "cascade-safe-claim/0.1", "cascade-projection-safe-claim/0.1"}:
            logical = kit.resolve_logical_world(world, case["world"])
            authored = source["case"]["author_operation"]
            require(isinstance(authored, dict) and authored.get("kind") in {"retime", "replace", "inject"}, f"{case['id']}: unsupported author operation")
            targets = [event for event in logical.base_events if event["topic"] == authored.get("target_topic")]
            if authored["kind"] == "retime":
                require(len(targets) == 1, f"{case['id']}: author target topic must identify one base event")
                require(set(authored) == {"kind", "target_topic", "at"}, f"{case['id']}: unsupported author retime")
                operations = [{"kind": "retime", "event": targets[0]["id"], "at": authored["at"]}]
            elif authored["kind"] == "replace":
                require(len(targets) == 1, f"{case['id']}: author target topic must identify one base event")
                require(set(authored) == {"kind", "target_topic", "pointer", "value"}, f"{case['id']}: unsupported author replacement")
                operations = [{"kind": "replace", "event": targets[0]["id"], "pointer": authored["pointer"], "value": authored["value"]}]
            else:
                require(set(authored) == {"kind", "topic", "at", "payload", "parents"}, f"{case['id']}: unsupported author injection")
                operations = [authored]
            intervention = {"format": "afterimage-intervention/0.1", "bundle": world.bundle, "parent_branch": parent, "case": case["id"], "operations": operations}
        elif template in {"cascade-multi-contract-claim/0.1", "cascade-minimal-explanation-claim/0.1"}:
            logical = kit.resolve_logical_world(world, case["world"])
            for authored in source["case"]["author_operations"]:
                require(isinstance(authored, dict) and authored.get("kind") in {"retime", "replace", "inject"}, f"{case['id']}: unsupported author operation")
                if authored["kind"] == "inject":
                    require(set(authored) == {"kind", "topic", "at", "payload", "parents"}, f"{case['id']}: unsupported author injection")
                    operations.append(authored)
                    continue
                targets = [event for event in logical.base_events if event["topic"] == authored["target_topic"]]
                require(len(targets) == 1, f"{case['id']}: author target topic must identify one base event")
                if authored["kind"] == "retime":
                    require(set(authored) == {"kind", "target_topic", "at"}, f"{case['id']}: unsupported author retime")
                    operations.append({"kind": "retime", "event": targets[0]["id"], "at": authored["at"]})
                else:
                    require(set(authored) == {"kind", "target_topic", "pointer", "value"}, f"{case['id']}: unsupported author replacement")
                    operations.append({"kind": "replace", "event": targets[0]["id"], "pointer": authored["pointer"], "value": authored["value"]})
            intervention = {"format": "afterimage-intervention/0.1", "bundle": world.bundle, "parent_branch": parent, "case": case["id"], "operations": operations}
        replay = verifier.replay_case(world, case, operations, parent)
        if template == "orient-event-claim/0.1":
            topic = source["case"]["answer_event_topic"]
            matches = [event for event in replay.events.values() if event["topic"] == topic]
            require(len(matches) == 1, f"{case['id']}: baseline expected exactly one {topic} event")
            event = matches[0]
            answer = {"event_id": event["id"], "topic": event["topic"], "at": event["at"], "projection": replay.projection}
        elif template == "orient-replay-claim/0.1":
            answer = {
                "trace_event_ids": [item["event"] for item in replay.trace_items],
                "records": replay.records,
                "projection": replay.projection,
            }
        elif template == "orient-observation-claim/0.1":
            hidden_topic = source["case"]["hidden_event_topic"]
            hidden = [event for event in replay.events.values() if event["topic"] == hidden_topic]
            require(len(hidden) == 1, f"{case['id']}: baseline expected exactly one {hidden_topic} event")
            answer = {
                "active_event_count": len(replay.events),
                "projected_records": replay.records,
                "projection": replay.projection,
                "hidden_event": hidden[0]["id"],
            }
        elif template == "orient-branch-claim/0.1":
            answer = {
                "branch": replay.branch,
                "baseline_records": replay.baseline.records,
                "candidate_records": replay.records,
                "changed_event_ids": replay.changed_event_ids,
                "projection": replay.projection,
            }
        elif template == "orient-envelope-claim/0.1":
            export_case = source["case"]["author_export_case"]
            export_path = destination / f"{export_case}.witness.json"
            require(export_path.is_file(), f"{case['id']}: author export witness is unavailable")
            answer = {"export": cre.load_json(export_path)}
        elif template in {"cascade-contract-claim/0.1", "cascade-safe-claim/0.1", "cascade-multi-contract-claim/0.1", "cascade-minimal-explanation-claim/0.1"}:
            answer = {"contracts": replay.records, "branch": replay.branch, "projection": replay.projection}
        elif template == "cascade-proof-claim/0.1":
            relays = [event for event in replay.events.values() if event["topic"] == source["case"]["author_relay_topic"]]
            ready = [event for event in replay.events.values() if event["topic"] == source["case"]["proof_contract"]["ready_topic"]]
            require(len(relays) == 1 and len(ready) == 1, f"{case['id']}: author proof events are unavailable")
            relay = relays[0]
            answer = {
                "contract": ready[0]["payload"], "public_rows": replay.records,
                "relay": {
                    "event": relay["id"], "relay_id": relay["payload"]["relay_id"],
                    "provenance_digest": cre.digest_id("afterimage/provenance/1", cre.canonical_bytes(relay["payload"]["provenance"])),
                    "projection_difference": "whole_event_suppressed", "policy_label": source["case"]["proof_contract"]["policy_label"],
                },
                "branch": replay.branch, "projection": replay.projection,
            }
        elif template == "cascade-projection-safe-claim/0.1":
            contracts = [event for event in replay.events.values() if event["topic"] == source["case"]["projection_contract"]["contract_topic"]]
            require(len(contracts) == 1, f"{case['id']}: author projection-safe contract is unavailable")
            answer = {
                "contract": contracts[0]["payload"],
                "baseline_records": replay.baseline.records,
                "public_rows": replay.records,
                "branch": replay.branch,
                "projection": replay.projection,
            }
        elif template in {"merge-certificate-claim/0.1", "merge-dedup-certificate-claim/0.1", "merge-split-certificate-claim/0.1", "merge-quorum-certificate-claim/0.1", "merge-offset-certificate-claim/0.1", "merge-minimal-conflict-certificate-claim/0.1", "merge-weighted-cut-certificate-claim/0.1", "merge-echo-certificate-claim/0.1", "merge-weighted-evidence-certificate-claim/0.1", "merge-causal-compression-certificate-claim/0.1", "merge-partial-order-certificate-claim/0.1", "merge-equivocation-certificate-claim/0.1", "merge-non-unique-certificate-claim/0.1"}:
            logical = kit.resolve_logical_world(world, case["world"])
            by_key = {event["payload"]["key"]: event["id"] for event in logical.base_events if event["topic"] == "merge.record"}
            authored = source["case"]["author_answer"]
            rejected = []
            for item in authored["rejected"]:
                converted = {"event": by_key[item["key"]], "reason": item["reason"]}
                if "duplicate_of" in item:
                    converted["duplicate_of"] = by_key[item["duplicate_of"]]
                rejected.append(converted)
            answer = {
                "accepted": [{"event": by_key[item["key"]], "at": item["at"]} for item in authored["accepted"]],
                "rejected": rejected,
                "certificate": [{"before": by_key[item["before"]], "after": by_key[item["after"]], "minimum_gap": item["minimum_gap"]} for item in authored["certificate"]],
            }
        elif template == "pulse-program-claim/0.1":
            answer = {"program": source["case"]["author_program"], "invariant": source["case"]["author_invariant"]}
        elif template == "mosaic-graph-claim/0.1":
            answer = source["case"]["author_answer"]
        elif template == "lens-law-claim/0.1":
            answer = {"program": source["case"]["author_program"]}
        elif template == "covenant-policy-claim/0.1":
            answer = {"policy": source["case"]["author_policy"], "claimed_response_bound": source["case"]["author_response_bound"]}
        elif template == "paradox-paired-history-claim/0.1":
            logical = kit.resolve_logical_world(world, case["world"])
            histories: dict[str, dict[str, Any]] = {}
            for side in ("left", "right"):
                authored = source["case"]["author_histories"][side]
                require(isinstance(authored, dict) and set(authored) == {"case", "operation"}, f"{case['id']}: invalid {side} history")
                require(authored["case"] == source["case"]["history_case"], f"{case['id']}: {side} history ends at the wrong case")
                operation = authored["operation"]
                require(isinstance(operation, dict) and operation.get("kind") == "replace", f"{case['id']}: {side} history operation must be replace")
                require(set(operation) == {"kind", "target_topic", "pointer", "value"}, f"{case['id']}: invalid {side} history replacement")
                targets = [event for event in logical.base_events if event["topic"] == operation["target_topic"]]
                require(len(targets) == 1, f"{case['id']}: {side} history target topic must identify one base event")
                histories[side] = {
                    "format": "afterimage-branch-history/0.1",
                    "bundle": world.bundle,
                    "world": case["world"],
                    "steps": [{
                        "case": authored["case"],
                        "operations": [{"kind": "replace", "event": targets[0]["id"], "pointer": operation["pointer"], "value": operation["value"]}],
                    }],
                }
            states = {}
            target = {**case, "input_branch": f"history:{source['case']['history_case']}"}
            for side in ("left", "right"):
                resolved = verifier.resolve_input_history(world, target, histories[side], facts)
                states[side] = verifier.evaluate_case_state(world, case, resolved.base_events, resolved.branch)
            require(cre.same_value(states["left"].records, states["right"].records), f"{case['id']}: author histories do not share a public projection")
            equivalence = [
                {"left": index, "right": index, "digest": cre.digest_id("afterimage/paradox-public-record/1", cre.canonical_bytes(record))}
                for index, record in enumerate(states["left"].records)
            ]
            contract = paradox_runtime.validate_contract(source["case"]["paradox_contract"])
            evidence = {side: [] for side in ("left", "right")}
            for side in ("left", "right"):
                for requirement in contract["safety_requirements"]:
                    evidence[side].extend(paradox_runtime.event_ids_for_requirement(states[side].events, requirement))
                evidence[side] = sorted(set(evidence[side]), key=cre.parse_id)
            latent_topics = contract["latent_topics"]
            differing = sorted(
                {
                    event_id
                    for event_id in set(states["left"].events) ^ set(states["right"].events)
                    if (states["left"].events.get(event_id) or states["right"].events[event_id])["topic"] in latent_topics
                },
                key=cre.parse_id,
            )
            answer = {
                "left_history": histories["left"], "right_history": histories["right"],
                "equivalence": equivalence, "safety_evidence": evidence, "latent_difference": differing,
            }
        else:
            raise BuildError(f"{case['id']}: unsupported validator template")
        witness = {
            "format": "afterimage-witness/0.1",
            "semantics": "cre/0.1",
            "bundle": world.bundle,
            "case": case["id"],
            "parent_branch": parent,
            "intervention": intervention,
            "answer": answer,
            "claimed": {"branch": replay.branch, "projection": replay.projection, "trace": replay.trace},
            "meta": {"producer": "afterimage-authoring/0.1", "comment": "author baseline; do not distribute with player bundle"},
        }
        witness_path = destination / f"{case['id']}.witness.json"
        witness_path.write_bytes(cre.canonical_bytes(witness))
        receipt = verifier.verify_witness(world_path, witness_path, facts)
        require(receipt["valid"], f"{case['id']}: generated author witness is invalid")
        receipt_path = destination / f"{case['id']}.receipt.json"
        receipt_path.write_bytes(cre.canonical_bytes(receipt))
        receipts.append(receipt)
        facts.update(receipt["unlocks"])
    shutil.rmtree(world_path)
    return {"witnesses": len(receipts), "scores": {receipt["case"]: receipt["score"]["total"] for receipt in receipts}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--author-dir", type=Path)
    parser.add_argument("--title", default="Afterimage vertical slice")
    parser.add_argument("--revision", default="slice-dev-0.1.0")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, action="append")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = load_design_manifest(args.manifest)
        sources = load_sources(args.manifest, args.source_root)
        with tempfile.TemporaryDirectory(prefix="afterimage-slice-build-") as temporary:
            logical = Path(temporary) / "logical"
            logical.mkdir()
            counts = compile_logical(logical, sources)
            logical_files = sum(1 for path in logical.rglob("*") if path.is_file())
            limits = None
            if logical_files > kit.DEFAULT_LIMITS["max_files"]:
                limits = {**kit.DEFAULT_LIMITS, "max_files": 1024}
            bundle = kit.pack_bundle(logical, args.output, args.title, args.revision, limits)
        designed_order = [item["id"] for item in manifest["cases"]]
        author = build_author_baseline(bundle, args.author_dir, sources, designed_order) if args.author_dir else None
        result = {"type": "slice-built", "path": str(args.output), **bundle.summary(), **counts, "author": author}
        if args.pretty:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(cre.canonical_text(result))
        return 0
    except (BuildError, kit.KitError, verifier.VerificationError, cre.CREError) as exc:
        code = getattr(exc, "code", "build_error")
        message = getattr(exc, "message", str(exc))
        context = getattr(exc, "context", {})
        print(cre.canonical_text({"type": "error", "code": code, "message": message, "context": context}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
