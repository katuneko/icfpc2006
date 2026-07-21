#!/usr/bin/env python3
"""Finite law checker for the typed first-order LENS 0.1 slice language."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "python"))

import cre


class LensError(Exception):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise LensError(code, message, context)


def normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def validate_timetable_contract(value: dict[str, Any]) -> dict[str, Any]:
    require(set(value) == {"task", "schedules", "gates", "platforms", "calendars", "provenance", "invalid_targets", "limits"}, "invalid_lens_contract", "LENS timetable contract fields are invalid")
    require(isinstance(value["schedules"], list) and value["schedules"], "invalid_lens_contract", "LENS schedule table is empty")
    source_keys: set[tuple[str, int]] = set()
    service_keys: set[str] = set()
    for index, item in enumerate(value["schedules"]):
        require(isinstance(item, dict) and set(item) == {"service_key", "line", "local_departure", "route_id", "utc_departure"}, "invalid_lens_contract", "LENS schedule row is invalid", index=index)
        require(all(isinstance(item[field], str) and item[field] for field in ("service_key", "line", "route_id")), "invalid_lens_contract", "LENS schedule text is invalid", index=index)
        require(all(isinstance(item[field], int) and not isinstance(item[field], bool) for field in ("local_departure", "utc_departure")), "invalid_lens_contract", "LENS schedule time is invalid", index=index)
        source_key = (item["line"], item["local_departure"])
        require(source_key not in source_keys and item["service_key"] not in service_keys, "invalid_lens_contract", "LENS schedule source identity is duplicated", index=index)
        source_keys.add(source_key)
        service_keys.add(item["service_key"])
    require(isinstance(value["gates"], list) and all(isinstance(item, dict) and set(item) == {"source", "target"} and isinstance(item["source"], str) and isinstance(item["target"], str) for item in value["gates"]), "invalid_lens_contract", "LENS gate table is invalid")
    require(isinstance(value["platforms"], list) and value["platforms"] and all(isinstance(item, str) for item in value["platforms"]), "invalid_lens_contract", "LENS platform domain is invalid")
    for field in ("calendars", "provenance"):
        require(isinstance(value[field], list) and value[field] and all(isinstance(item, list) and all(isinstance(part, str) for part in item) for item in value[field]), "invalid_lens_contract", f"LENS {field} domain is invalid")
    require(isinstance(value["invalid_targets"], list), "invalid_lens_contract", "LENS invalid targets must be a list")
    require(isinstance(value["limits"], dict) and set(value["limits"]) == {"max_program_bytes", "max_nodes", "max_auxiliary_cells", "max_reductions"}, "invalid_lens_contract", "LENS limit fields are invalid")
    require(all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value["limits"].values()), "invalid_lens_contract", "LENS limits must be positive integers")
    return value


def validate_history_contract(value: dict[str, Any]) -> dict[str, Any]:
    require(set(value) == {"task", "histories", "channels", "private_deltas", "audit_chains", "invalid_targets", "limits"}, "invalid_lens_contract", "LENS history contract fields are invalid")
    require(isinstance(value["histories"], list) and value["histories"], "invalid_lens_contract", "LENS history table is empty")
    history_keys: set[str] = set()
    internal_tips: set[str] = set()
    public_pairs: set[tuple[str, str]] = set()
    for index, item in enumerate(value["histories"]):
        require(isinstance(item, dict) and set(item) == {"history_key", "internal_tip", "record_id", "public_tip"}, "invalid_lens_contract", "LENS history row is invalid", index=index)
        require(all(isinstance(item[field], str) and item[field] for field in item), "invalid_lens_contract", "LENS history row text is invalid", index=index)
        require(item["history_key"] not in history_keys and item["internal_tip"] not in internal_tips, "invalid_lens_contract", "LENS private history identity is duplicated", index=index)
        history_keys.add(item["history_key"])
        internal_tips.add(item["internal_tip"])
        public_pairs.add((item["record_id"], item["public_tip"]))
    require(len(public_pairs) < len(value["histories"]), "invalid_lens_contract", "LENS history case must contain a public collision")
    require(isinstance(value["channels"], list) and all(isinstance(item, dict) and set(item) == {"source", "target"} and isinstance(item["source"], str) and isinstance(item["target"], str) for item in value["channels"]), "invalid_lens_contract", "LENS history channel table is invalid")
    for field in ("private_deltas", "audit_chains"):
        require(isinstance(value[field], list) and value[field] and all(isinstance(item, list) and all(isinstance(part, str) for part in item) for item in value[field]), "invalid_lens_contract", f"LENS {field} domain is invalid")
    require(isinstance(value["invalid_targets"], list), "invalid_lens_contract", "LENS invalid history targets must be a list")
    require(isinstance(value["limits"], dict) and set(value["limits"]) == {"max_program_bytes", "max_nodes", "max_auxiliary_cells", "max_reductions"}, "invalid_lens_contract", "LENS limit fields are invalid")
    require(all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value["limits"].values()), "invalid_lens_contract", "LENS limits must be positive integers")
    return value


def validate_contract(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("task") == "timetable":
        return validate_timetable_contract(value)
    if isinstance(value, dict) and value.get("task") == "history":
        return validate_history_contract(value)
    require(isinstance(value, dict) and set(value) == {"addresses", "entrances", "units", "provenance", "invalid_targets", "limits"}, "invalid_lens_contract", "LENS contract fields are invalid")
    require(isinstance(value["addresses"], list) and value["addresses"], "invalid_lens_contract", "LENS address table is empty")
    seen_sources: set[tuple[str, int]] = set()
    for index, item in enumerate(value["addresses"]):
        require(isinstance(item, dict) and set(item) == {"street", "number", "segment_id", "offset"}, "invalid_lens_contract", "LENS address row is invalid", index=index)
        require(isinstance(item["street"], str) and isinstance(item["segment_id"], str) and isinstance(item["number"], int) and not isinstance(item["number"], bool) and isinstance(item["offset"], int) and not isinstance(item["offset"], bool), "invalid_lens_contract", "LENS address row types are invalid", index=index)
        key = (normalize(item["street"]), item["number"])
        require(key not in seen_sources, "invalid_lens_contract", "LENS civic address is duplicated", index=index)
        seen_sources.add(key)
    require(isinstance(value["entrances"], list) and all(isinstance(item, dict) and set(item) == {"source", "target"} and isinstance(item["source"], str) and isinstance(item["target"], str) for item in value["entrances"]), "invalid_lens_contract", "LENS entrance table is invalid")
    require(isinstance(value["units"], list) and all(item is None or isinstance(item, str) for item in value["units"]), "invalid_lens_contract", "LENS unit domain is invalid")
    require(isinstance(value["provenance"], list) and all(isinstance(item, list) and all(isinstance(part, str) for part in item) for item in value["provenance"]), "invalid_lens_contract", "LENS provenance domain is invalid")
    require(isinstance(value["invalid_targets"], list), "invalid_lens_contract", "LENS invalid targets must be a list")
    require(isinstance(value["limits"], dict) and set(value["limits"]) == {"max_program_bytes", "max_nodes", "max_auxiliary_cells", "max_reductions"}, "invalid_lens_contract", "LENS limit fields are invalid")
    require(all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value["limits"].values()), "invalid_lens_contract", "LENS limits must be positive integers")
    return value


def node_count(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(node_count(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(node_count(item) for item in value)
    return 1


def compile_program(value: Any, contract: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"format", "complement_schema", "get", "put"}, "invalid_lens_program", "LENS program fields are invalid")
    require(value["format"] == "afterimage-lens/0.1", "invalid_lens_program", "LENS program format is unsupported")
    cells = value["complement_schema"]
    require(isinstance(cells, list), "invalid_lens_program", "LENS complement schema must be a list")
    task = contract.get("task", "address")
    timetable = task == "timetable"
    history = task == "history"
    allowed_cells = (
        {"platform": "text", "calendar": "text-list", "schedule_provenance": "text-list", "service_key": "text"}
        if timetable else {"history_key": "text", "private_delta": "text-list", "audit_chain": "text-list"}
        if history else {"unit": "option-text", "provenance": "text-list", "boundary_street": "text"}
    )
    schema: dict[str, str] = {}
    for index, item in enumerate(cells):
        require(isinstance(item, dict) and set(item) == {"name", "type"} and item["name"] in allowed_cells and item["type"] == allowed_cells[item["name"]] and item["name"] not in schema, "invalid_lens_program", "LENS complement cell is invalid", index=index)
        schema[item["name"]] = item["type"]
    known_get = {"forward_schedule", "encode_gate"} if timetable else {"forward_history", "encode_channel"} if history else {"forward_address", "encode_entrance"}
    known_put = {"reverse_schedule", "decode_gate", "restore_schedule"} if timetable else {"reverse_history", "decode_channel", "restore_history"} if history else {"reverse_address", "decode_entrance", "restore"}
    require(isinstance(value["get"], list) and isinstance(value["put"], list), "invalid_lens_program", "LENS pipelines must be lists")
    get_ops: dict[str, dict[str, Any]] = {}
    for item in value["get"]:
        require(isinstance(item, dict) and item.get("op") in known_get and item["op"] not in get_ops, "invalid_lens_program", "LENS get operation is invalid")
        if timetable and item["op"] == "forward_schedule":
            require(set(item) == {"op", "table"} and item["table"] == "schedules", "invalid_lens_program", "forward_schedule fields are invalid")
        elif timetable:
            require(set(item) == {"op", "table"} and item["table"] == "gates", "invalid_lens_program", "encode_gate fields are invalid")
        elif history and item["op"] == "forward_history":
            require(set(item) == {"op", "table"} and item["table"] == "histories", "invalid_lens_program", "forward_history fields are invalid")
        elif history:
            require(set(item) == {"op", "table"} and item["table"] == "channels", "invalid_lens_program", "encode_channel fields are invalid")
        elif item["op"] == "forward_address":
            require(set(item) == {"op", "table"} and item["table"] == "addresses", "invalid_lens_program", "forward_address fields are invalid")
        else:
            require(set(item) == {"op", "table"} and item["table"] == "entrances", "invalid_lens_program", "encode_entrance fields are invalid")
        get_ops[item["op"]] = item
    put_ops: dict[str, dict[str, Any]] = {}
    for item in value["put"]:
        require(isinstance(item, dict) and item.get("op") in known_put and item["op"] not in put_ops, "invalid_lens_program", "LENS put operation is invalid")
        if timetable and item["op"] == "reverse_schedule":
            require(set(item) == {"op", "table", "disambiguation"} and item["table"] == "schedules" and item["disambiguation"] in {None, "service_key"}, "invalid_lens_program", "reverse_schedule fields are invalid")
        elif timetable and item["op"] == "decode_gate":
            require(set(item) == {"op", "table"} and item["table"] == "gates", "invalid_lens_program", "decode_gate fields are invalid")
        elif timetable:
            require(set(item) == {"op", "fields"} and isinstance(item["fields"], list) and len(set(item["fields"])) == len(item["fields"]) and set(item["fields"]) <= {"platform", "calendar", "schedule_provenance"}, "invalid_lens_program", "restore_schedule fields are invalid")
        elif history and item["op"] == "reverse_history":
            require(set(item) == {"op", "table", "disambiguation"} and item["table"] == "histories" and item["disambiguation"] in {None, "history_key"}, "invalid_lens_program", "reverse_history fields are invalid")
        elif history and item["op"] == "decode_channel":
            require(set(item) == {"op", "table"} and item["table"] == "channels", "invalid_lens_program", "decode_channel fields are invalid")
        elif history:
            require(set(item) == {"op", "fields"} and isinstance(item["fields"], list) and len(set(item["fields"])) == len(item["fields"]) and set(item["fields"]) <= {"private_delta", "audit_chain"}, "invalid_lens_program", "restore_history fields are invalid")
        elif item["op"] == "reverse_address":
            require(set(item) == {"op", "table", "disambiguation"} and item["table"] == "addresses" and item["disambiguation"] in {None, "boundary_street"}, "invalid_lens_program", "reverse_address fields are invalid")
        elif item["op"] == "decode_entrance":
            require(set(item) == {"op", "table"} and item["table"] == "entrances", "invalid_lens_program", "decode_entrance fields are invalid")
        else:
            require(set(item) == {"op", "fields"} and isinstance(item["fields"], list) and len(set(item["fields"])) == len(item["fields"]) and set(item["fields"]) <= {"unit", "provenance"}, "invalid_lens_program", "restore fields are invalid")
        put_ops[item["op"]] = item
    size = len(cre.canonical_bytes(value))
    nodes = node_count(value)
    limits = contract["limits"]
    require(size <= limits["max_program_bytes"] and nodes <= limits["max_nodes"] and len(schema) <= limits["max_auxiliary_cells"], "lens_limit", "LENS program exceeds a static resource limit")
    return {"value": value, "schema": schema, "get_ops": get_ops, "put_ops": put_ops, "bytes": size, "nodes": nodes, "task": task}


def complement(compiled: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    if compiled.get("task") == "timetable":
        result = {}
        for field in ("platform", "calendar", "schedule_provenance", "service_key"):
            if field in compiled["schema"]:
                result[field] = source[field]
        return result
    if compiled.get("task") == "history":
        result = {}
        for field in ("history_key", "private_delta", "audit_chain"):
            if field in compiled["schema"]:
                result[field] = source[field]
        return result
    result = {}
    if "unit" in compiled["schema"]:
        result["unit"] = source["unit"]
    if "provenance" in compiled["schema"]:
        result["provenance"] = source["provenance"]
    if "boundary_street" in compiled["schema"]:
        result["boundary_street"] = normalize(source["street_name"])
    return result


def get_view(compiled: dict[str, Any], contract: dict[str, Any], source: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
    reductions = 0
    if compiled.get("task") == "timetable":
        if set(compiled["get_ops"]) != {"forward_schedule", "encode_gate"}:
            return None, reductions
        row = None
        for candidate in contract["schedules"]:
            reductions += 1
            if candidate["line"] == source["line"] and candidate["local_departure"] == source["local_departure"]:
                row = candidate
                break
        if row is None:
            return None, reductions
        gate_code = None
        if source["gate"] is not None:
            for candidate in contract["gates"]:
                reductions += 1
                if candidate["source"] == source["gate"]:
                    gate_code = candidate["target"]
                    break
            if gate_code is None:
                return None, reductions
        return {"route_id": row["route_id"], "utc_departure": row["utc_departure"], "gate_code": gate_code}, reductions
    if compiled.get("task") == "history":
        if set(compiled["get_ops"]) != {"forward_history", "encode_channel"}:
            return None, reductions
        row = None
        for candidate in contract["histories"]:
            reductions += 1
            if candidate["history_key"] == source["history_key"] and candidate["internal_tip"] == source["internal_tip"]:
                row = candidate
                break
        if row is None:
            return None, reductions
        channel_code = None
        if source["channel"] is not None:
            for candidate in contract["channels"]:
                reductions += 1
                if candidate["source"] == source["channel"]:
                    channel_code = candidate["target"]
                    break
            if channel_code is None:
                return None, reductions
        return {"record_id": row["record_id"], "public_tip": row["public_tip"], "channel_code": channel_code}, reductions
    if set(compiled["get_ops"]) != {"forward_address", "encode_entrance"}:
        return None, reductions
    row = None
    for candidate in contract["addresses"]:
        reductions += 1
        if normalize(candidate["street"]) == normalize(source["street_name"]) and candidate["number"] == source["number"]:
            row = candidate
            break
    if row is None:
        return None, reductions
    entrance_code = None
    if source["entrance"] is not None:
        for candidate in contract["entrances"]:
            reductions += 1
            if candidate["source"] == source["entrance"]:
                entrance_code = candidate["target"]
                break
        if entrance_code is None:
            return None, reductions
    return {"segment_id": row["segment_id"], "offset": row["offset"], "entrance_code": entrance_code}, reductions


def put_view(compiled: dict[str, Any], contract: dict[str, Any], source: dict[str, Any], target: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
    reductions = 0
    if compiled.get("task") == "timetable":
        if set(compiled["put_ops"]) != {"reverse_schedule", "decode_gate", "restore_schedule"} or not isinstance(target, dict) or set(target) != {"route_id", "utc_departure", "gate_code"}:
            return None, reductions
        candidates = []
        for row in contract["schedules"]:
            reductions += 1
            if row["route_id"] == target["route_id"] and row["utc_departure"] == target["utc_departure"]:
                candidates.append(row)
        if not candidates:
            return None, reductions
        comp = complement(compiled, source)
        reverse = compiled["put_ops"]["reverse_schedule"]
        chosen = None
        if reverse["disambiguation"] == "service_key" and "service_key" in comp:
            chosen = next((row for row in candidates if row["service_key"] == comp["service_key"]), None)
        chosen = chosen or sorted(candidates, key=lambda row: row["service_key"].encode("utf-8"))[0]
        gate = None
        if target["gate_code"] is not None:
            for row in contract["gates"]:
                reductions += 1
                if row["target"] == target["gate_code"]:
                    gate = row["source"]
                    break
            if gate is None:
                return None, reductions
        restore = set(compiled["put_ops"]["restore_schedule"]["fields"])
        return {
            "service_key": chosen["service_key"], "line": chosen["line"], "local_departure": chosen["local_departure"], "gate": gate,
            "platform": comp.get("platform") if "platform" in restore else "",
            "calendar": comp.get("calendar", []) if "calendar" in restore else [],
            "schedule_provenance": comp.get("schedule_provenance", []) if "schedule_provenance" in restore else [],
        }, reductions
    if compiled.get("task") == "history":
        if set(compiled["put_ops"]) != {"reverse_history", "decode_channel", "restore_history"} or not isinstance(target, dict) or set(target) != {"record_id", "public_tip", "channel_code"}:
            return None, reductions
        candidates = []
        for row in contract["histories"]:
            reductions += 1
            if row["record_id"] == target["record_id"] and row["public_tip"] == target["public_tip"]:
                candidates.append(row)
        if not candidates:
            return None, reductions
        comp = complement(compiled, source)
        reverse = compiled["put_ops"]["reverse_history"]
        chosen = None
        if reverse["disambiguation"] == "history_key" and "history_key" in comp:
            chosen = next((row for row in candidates if row["history_key"] == comp["history_key"]), None)
        chosen = chosen or sorted(candidates, key=lambda row: row["history_key"].encode("utf-8"))[0]
        channel = None
        if target["channel_code"] is not None:
            for row in contract["channels"]:
                reductions += 1
                if row["target"] == target["channel_code"]:
                    channel = row["source"]
                    break
            if channel is None:
                return None, reductions
        restore = set(compiled["put_ops"]["restore_history"]["fields"])
        return {
            "history_key": chosen["history_key"], "internal_tip": chosen["internal_tip"], "channel": channel,
            "private_delta": comp.get("private_delta", []) if "private_delta" in restore else [],
            "audit_chain": comp.get("audit_chain", []) if "audit_chain" in restore else [],
        }, reductions
    if set(compiled["put_ops"]) != {"reverse_address", "decode_entrance", "restore"} or not isinstance(target, dict) or set(target) != {"segment_id", "offset", "entrance_code"}:
        return None, reductions
    candidates = []
    for row in contract["addresses"]:
        reductions += 1
        if row["segment_id"] == target["segment_id"] and row["offset"] == target["offset"]:
            candidates.append(row)
    if not candidates:
        return None, reductions
    comp = complement(compiled, source)
    reverse = compiled["put_ops"]["reverse_address"]
    chosen = None
    if reverse["disambiguation"] == "boundary_street" and "boundary_street" in comp:
        chosen = next((row for row in candidates if normalize(row["street"]) == comp["boundary_street"]), None)
    chosen = chosen or sorted(candidates, key=lambda row: (normalize(row["street"]), row["number"]))[0]
    entrance = None
    if target["entrance_code"] is not None:
        for row in contract["entrances"]:
            reductions += 1
            if row["target"] == target["entrance_code"]:
                entrance = row["source"]
                break
        if entrance is None:
            return None, reductions
    restore = set(compiled["put_ops"]["restore"]["fields"])
    result = {
        "number": chosen["number"], "street_name": chosen["street"], "entrance": entrance,
        "unit": comp.get("unit") if "unit" in restore else None,
        "provenance": comp.get("provenance", []) if "provenance" in restore else [],
    }
    return result, reductions


def source_domain(contract: dict[str, Any]) -> list[dict[str, Any]]:
    if contract.get("task") == "timetable":
        values = []
        gates = [None, *[item["source"] for item in contract["gates"]]]
        for row, platform, gate, calendar, provenance in itertools.product(contract["schedules"], contract["platforms"], gates, contract["calendars"], contract["provenance"]):
            values.append({"service_key": row["service_key"], "line": row["line"], "local_departure": row["local_departure"], "platform": platform, "gate": gate, "calendar": calendar, "schedule_provenance": provenance})
        return sorted(values, key=cre.canonical_bytes)
    if contract.get("task") == "history":
        values = []
        channels = [None, *[item["source"] for item in contract["channels"]]]
        for row, channel, private_delta, audit_chain in itertools.product(contract["histories"], channels, contract["private_deltas"], contract["audit_chains"]):
            values.append({"history_key": row["history_key"], "internal_tip": row["internal_tip"], "channel": channel, "private_delta": private_delta, "audit_chain": audit_chain})
        return sorted(values, key=cre.canonical_bytes)
    values = []
    entrances = [None, *[item["source"] for item in contract["entrances"]]]
    for row, unit, entrance, provenance in itertools.product(contract["addresses"], contract["units"], entrances, contract["provenance"]):
        values.append({"number": row["number"], "street_name": row["street"], "unit": unit, "entrance": entrance, "provenance": provenance})
    return sorted(values, key=cre.canonical_bytes)


def target_domain(contract: dict[str, Any]) -> list[dict[str, Any]]:
    if contract.get("task") == "timetable":
        routes = {(row["route_id"], row["utc_departure"]) for row in contract["schedules"]}
        gates = [None, *[item["target"] for item in contract["gates"]]]
        return sorted([{"route_id": route, "utc_departure": at, "gate_code": gate} for (route, at), gate in itertools.product(routes, gates)], key=cre.canonical_bytes)
    if contract.get("task") == "history":
        records = {(row["record_id"], row["public_tip"]) for row in contract["histories"]}
        channels = [None, *[item["target"] for item in contract["channels"]]]
        return sorted([{"record_id": record, "public_tip": tip, "channel_code": channel} for (record, tip), channel in itertools.product(records, channels)], key=cre.canonical_bytes)
    routes = {(row["segment_id"], row["offset"]) for row in contract["addresses"]}
    entrances = [None, *[item["target"] for item in contract["entrances"]]]
    return sorted([{"segment_id": segment, "offset": offset, "entrance_code": entrance} for (segment, offset), entrance in itertools.product(routes, entrances)], key=cre.canonical_bytes)


def counterexample(law: str, source: dict[str, Any], edit: Any = None, expected: Any = None, observed: Any = None) -> LensError:
    return LensError("lens_counterexample", "LENS law failed on bounded domain", {"law": law, "source": source, "edit": edit, "expected": expected, "observed": observed})


def verify_program(program: Any, contract_value: Any) -> dict[str, int]:
    contract = validate_contract(contract_value)
    compiled = compile_program(program, contract)
    worst = 0
    sources = source_domain(contract)
    targets = target_domain(contract)
    for source in sources:
        view, reductions = get_view(compiled, contract, source)
        worst = max(worst, reductions)
        if view is None:
            raise counterexample("GetTotal", source, observed=None)
        restored, reductions = put_view(compiled, contract, source, view)
        worst = max(worst, reductions)
        if not cre.same_value(restored, source):
            raise counterexample("GetPut", source, view, source, restored)
        for target in targets:
            updated, reductions = put_view(compiled, contract, source, target)
            worst = max(worst, reductions)
            if updated is None:
                raise counterexample("PutTotal", source, target, "success", None)
            observed, reductions = get_view(compiled, contract, updated)
            worst = max(worst, reductions)
            if not cre.same_value(observed, target):
                raise counterexample("PutGet", source, target, target, observed)
            stable, reductions = put_view(compiled, contract, updated, target)
            worst = max(worst, reductions)
            if not cre.same_value(stable, updated):
                raise counterexample("Stability", source, target, updated, stable)
            if contract.get("task") == "timetable":
                preserved = {field: source[field] for field in ("platform", "calendar", "schedule_provenance")}
                observed_preserved = {field: updated[field] for field in preserved}
            elif contract.get("task") == "history":
                preserved = {field: source[field] for field in ("private_delta", "audit_chain")}
                observed_preserved = {field: updated[field] for field in preserved}
            else:
                preserved = {"unit": source["unit"], "provenance": source["provenance"]}
                observed_preserved = {"unit": updated["unit"], "provenance": updated["provenance"]}
            if not cre.same_value(preserved, observed_preserved):
                raise counterexample("Provenance", source, target, preserved, observed_preserved)
        for target in contract["invalid_targets"]:
            before = cre.canonical_bytes(source)
            observed, reductions = put_view(compiled, contract, source, target)
            worst = max(worst, reductions)
            if observed is not None or before != cre.canonical_bytes(source):
                raise counterexample("InvalidAtomic", source, target, None, observed)
    require(worst <= contract["limits"]["max_reductions"], "lens_limit", "LENS program exceeds reduction limit", observed=worst)
    return {"program_nodes": compiled["nodes"], "auxiliary_schema_cells": len(compiled["schema"]), "worst_reductions": worst, "domain_sources": len(sources), "domain_targets": len(targets)}
