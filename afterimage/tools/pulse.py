#!/usr/bin/env python3
"""Bounded deterministic runtime and exhaustive checker for PULSE 0.1."""

from __future__ import annotations

import argparse
import heapq
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "python"))

import cre


class PulseError(Exception):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise PulseError(code, message, context)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if is_int(value):
        return "int"
    return "value"


def validate_contract(value: Any) -> dict[str, Any]:
    debounce_keys = {
        "input_topic", "timer_topic", "output_topic", "output_payload", "quiet_ticks",
        "horizon", "max_pulses", "named_fixtures", "limits",
    }
    deduplicate_keys = {
        "task", "input_topic", "timer_topic", "output_topic", "horizon",
        "max_pulses", "named_fixtures", "limits",
    }
    timeout_keys = {
        "task", "input_topics", "timer_topic", "output_topic", "output_payload",
        "timeout_ticks", "horizon", "max_commands", "named_fixtures", "limits",
    }
    bucket_keys = {
        "task", "input_topic", "timer_topic", "output_topic", "capacity",
        "initial_tokens", "refill_per_tick", "horizon", "max_requests",
        "named_fixtures", "limits",
    }
    quorum_keys = {
        "task", "input_topics", "timer_topic", "output_topic", "horizon",
        "max_votes", "named_fixtures", "limits",
    }
    barrier_keys = {
        "task", "input_topics", "timer_topic", "output_topic", "horizon",
        "max_signals", "named_fixtures", "limits",
    }
    backpressure_keys = {
        "task", "input_topics", "timer_topic", "output_topic", "capacity",
        "initial_depth", "horizon", "max_commands", "named_fixtures", "limits",
    }
    failover_keys = {
        "task", "input_topics", "timer_topic", "output_topics",
        "horizon", "max_commands", "named_fixtures", "limits",
    }
    exactly_once_keys = {
        "task", "input_topics", "timer_topic", "output_topics",
        "horizon", "max_commands", "named_fixtures", "limits",
    }
    shared_deadline_keys = {
        "task", "input_topics", "timer_topic", "output_topic", "output_payload",
        "offsets", "horizon", "max_signals", "named_fixtures", "limits",
    }
    sliding_window_keys = {
        "task", "input_topic", "timer_topic", "output_topic", "capacity",
        "window_ticks", "horizon", "max_requests", "named_fixtures", "limits",
    }
    reorder_keys = {
        "task", "input_topics", "timer_topic", "output_topic", "output_payload",
        "max_index", "horizon", "max_events", "named_fixtures", "limits",
    }
    circuit_keys = {
        "task", "input_topics", "timer_topic", "output_topics",
        "failure_threshold", "cooldown_ticks", "horizon", "max_commands",
        "named_fixtures", "limits",
    }
    require(isinstance(value, dict), "invalid_pulse_contract", "PULSE contract must be a map")
    task = value.get("task", "debounce")
    require(task in {"debounce", "deduplicate-ticks", "cancelable-timeout", "token-bucket", "two-of-three", "multi-topic-barrier", "city-clock", "backpressure", "warm-failover", "exactly-once", "shared-deadline", "sliding-window", "reorder-buffer", "circuit-breaker"}, "invalid_pulse_contract", "PULSE task is unsupported", task=task)
    expected_keys = (
        debounce_keys if task == "debounce"
        else timeout_keys if task == "cancelable-timeout"
        else bucket_keys if task == "token-bucket"
        else quorum_keys if task == "two-of-three"
        else barrier_keys if task in {"multi-topic-barrier", "city-clock"}
        else backpressure_keys if task == "backpressure"
        else failover_keys if task == "warm-failover"
        else exactly_once_keys if task == "exactly-once"
        else shared_deadline_keys if task == "shared-deadline"
        else sliding_window_keys if task == "sliding-window"
        else reorder_keys if task == "reorder-buffer"
        else circuit_keys if task == "circuit-breaker"
        else deduplicate_keys
    )
    require(set(value) == expected_keys, "invalid_pulse_contract", "PULSE contract fields are invalid", task=task)
    topic_names = ("timer_topic", "output_topic") if task == "reorder-buffer" else ("timer_topic",) if task in {"warm-failover", "exactly-once", "circuit-breaker"} else (("timer_topic", "output_topic") if task in {"cancelable-timeout", "two-of-three", "multi-topic-barrier", "city-clock", "backpressure", "shared-deadline"} else ("input_topic", "timer_topic", "output_topic"))
    for name in topic_names:
        require(isinstance(value[name], str) and value[name], "invalid_pulse_contract", "PULSE topic is invalid", field=name)
    if task == "cancelable-timeout":
        topics = value["input_topics"]
        require(isinstance(topics, dict) and set(topics) == {"start", "cancel"}, "invalid_pulse_contract", "PULSE timeout input topics are invalid")
        require(all(isinstance(topic, str) and topic for topic in topics.values()) and len(set(topics.values())) == 2, "invalid_pulse_contract", "PULSE timeout topics must be distinct non-empty text")
        require(len(set(topics.values()) | {value["timer_topic"], value["output_topic"]}) == 4, "invalid_pulse_contract", "PULSE timeout input, timer, and output topics must differ")
    if task == "two-of-three":
        topics = value["input_topics"]
        require(isinstance(topics, dict) and set(topics) == {"A", "B", "C"}, "invalid_pulse_contract", "PULSE quorum input topics are invalid")
        require(all(isinstance(topic, str) and topic for topic in topics.values()) and len(set(topics.values())) == 3, "invalid_pulse_contract", "PULSE quorum topics must be distinct non-empty text")
        require(len(set(topics.values()) | {value["timer_topic"], value["output_topic"]}) == 5, "invalid_pulse_contract", "PULSE quorum input, timer, and output topics must differ")
    if task == "multi-topic-barrier":
        topics = value["input_topics"]
        require(isinstance(topics, dict) and set(topics) == {"wind", "bridge", "hospital"}, "invalid_pulse_contract", "PULSE barrier input topics are invalid")
        require(all(isinstance(topic, str) and topic for topic in topics.values()) and len(set(topics.values())) == 3, "invalid_pulse_contract", "PULSE barrier topics must be distinct non-empty text")
        require(len(set(topics.values()) | {value["timer_topic"], value["output_topic"]}) == 5, "invalid_pulse_contract", "PULSE barrier input, timer, and output topics must differ")
    if task == "city-clock":
        topics = value["input_topics"]
        require(isinstance(topics, dict) and set(topics) == {"archive", "controller", "policy"}, "invalid_pulse_contract", "PULSE city-clock input topics are invalid")
        require(all(isinstance(topic, str) and topic for topic in topics.values()) and len(set(topics.values())) == 3, "invalid_pulse_contract", "PULSE city-clock topics must be distinct non-empty text")
        require(len(set(topics.values()) | {value["timer_topic"], value["output_topic"]}) == 5, "invalid_pulse_contract", "PULSE city-clock input, timer, and output topics must differ")
    if task == "backpressure":
        topics = value["input_topics"]
        require(isinstance(topics, dict) and set(topics) == {"request", "drain"}, "invalid_pulse_contract", "PULSE backpressure input topics are invalid")
        require(all(isinstance(topic, str) and topic for topic in topics.values()) and len(set(topics.values())) == 2, "invalid_pulse_contract", "PULSE backpressure topics must be distinct non-empty text")
        require(len(set(topics.values()) | {value["timer_topic"], value["output_topic"]}) == 4, "invalid_pulse_contract", "PULSE backpressure input, timer, and output topics must differ")
    if task == "warm-failover":
        topics = value["input_topics"]
        outputs = value["output_topics"]
        require(isinstance(topics, dict) and set(topics) == {"request", "warm", "fail", "recover"}, "invalid_pulse_contract", "PULSE warm-failover input topics are invalid")
        require(isinstance(outputs, dict) and set(outputs) == {"primary", "secondary"}, "invalid_pulse_contract", "PULSE warm-failover output topics are invalid")
        all_topics = list(topics.values()) + list(outputs.values()) + [value["timer_topic"]]
        require(all(isinstance(topic, str) and topic for topic in all_topics) and len(set(all_topics)) == len(all_topics), "invalid_pulse_contract", "PULSE warm-failover topics must be distinct non-empty text")
    if task == "exactly-once":
        topics = value["input_topics"]
        outputs = value["output_topics"]
        require(isinstance(topics, dict) and set(topics) == {"A", "B", "fail", "recover"}, "invalid_pulse_contract", "PULSE exactly-once input topics are invalid")
        require(isinstance(outputs, dict) and set(outputs) == {"primary", "secondary"}, "invalid_pulse_contract", "PULSE exactly-once output topics are invalid")
        all_topics = list(topics.values()) + list(outputs.values()) + [value["timer_topic"]]
        require(all(isinstance(topic, str) and topic for topic in all_topics) and len(set(all_topics)) == len(all_topics), "invalid_pulse_contract", "PULSE exactly-once topics must be distinct non-empty text")
    if task == "shared-deadline":
        topics = value["input_topics"]
        require(isinstance(topics, dict) and set(topics) == {"A", "B"}, "invalid_pulse_contract", "PULSE shared-deadline input topics are invalid")
        require(all(isinstance(topic, str) and topic for topic in topics.values()) and len(set(topics.values()) | {value["timer_topic"], value["output_topic"]}) == 4, "invalid_pulse_contract", "PULSE shared-deadline topics must be distinct non-empty text")
        require(isinstance(value["offsets"], dict) and set(value["offsets"]) == {"A", "B"}, "invalid_pulse_contract", "PULSE shared-deadline offsets are invalid")
        require(all(is_int(offset) and offset > 0 for offset in value["offsets"].values()), "invalid_pulse_contract", "PULSE shared-deadline offsets must be positive")
    if task == "reorder-buffer":
        require(is_int(value["max_index"]) and 1 <= value["max_index"] <= 7, "invalid_pulse_contract", "PULSE reorder index bound is invalid")
        topics = value["input_topics"]
        expected = {str(index) for index in range(value["max_index"] + 1)}
        require(isinstance(topics, dict) and set(topics) == expected and all(isinstance(topic, str) and topic for topic in topics.values()), "invalid_pulse_contract", "PULSE reorder topics are invalid")
        require(len(set(topics.values()) | {value["timer_topic"], value["output_topic"]}) == len(topics) + 2, "invalid_pulse_contract", "PULSE reorder topics must be distinct")
    if task == "circuit-breaker":
        topics = value["input_topics"]
        outputs = value["output_topics"]
        require(isinstance(topics, dict) and set(topics) == {"request", "failure", "success"}, "invalid_pulse_contract", "PULSE circuit-breaker input topics are invalid")
        require(isinstance(outputs, dict) and set(outputs) == {"admit", "probe"}, "invalid_pulse_contract", "PULSE circuit-breaker output topics are invalid")
        all_topics = list(topics.values()) + list(outputs.values()) + [value["timer_topic"]]
        require(all(isinstance(topic, str) and topic for topic in all_topics) and len(set(all_topics)) == len(all_topics), "invalid_pulse_contract", "PULSE circuit-breaker topics must be distinct")
        require(is_int(value["failure_threshold"]) and value["failure_threshold"] >= 2, "invalid_pulse_contract", "PULSE circuit-breaker threshold is invalid")
        require(is_int(value["cooldown_ticks"]) and value["cooldown_ticks"] > 0, "invalid_pulse_contract", "PULSE circuit-breaker cooldown is invalid")
    count_field = "max_commands" if task in {"cancelable-timeout", "backpressure", "warm-failover", "exactly-once", "circuit-breaker"} else "max_events" if task == "reorder-buffer" else "max_requests" if task in {"token-bucket", "sliding-window"} else "max_votes" if task == "two-of-three" else "max_signals" if task in {"multi-topic-barrier", "city-clock", "shared-deadline"} else "max_pulses"
    for name in ("horizon", count_field):
        require(is_int(value[name]) and value[name] >= 0, "invalid_pulse_contract", "PULSE domain integer is invalid", field=name)
    require(value["horizon"] > 0, "invalid_pulse_contract", "PULSE horizon must be positive")
    if task == "debounce":
        require(is_int(value["quiet_ticks"]) and value["quiet_ticks"] > 0, "invalid_pulse_contract", "PULSE quiet period must be positive")
    if task == "cancelable-timeout":
        require(is_int(value["timeout_ticks"]) and value["timeout_ticks"] > 0, "invalid_pulse_contract", "PULSE timeout must be positive")
    if task == "token-bucket":
        for name in ("capacity", "initial_tokens", "refill_per_tick"):
            require(is_int(value[name]), "invalid_pulse_contract", "PULSE token-bucket integer is invalid", field=name)
        require(value["capacity"] > 0, "invalid_pulse_contract", "PULSE token-bucket capacity must be positive")
        require(0 <= value["initial_tokens"] <= value["capacity"], "invalid_pulse_contract", "PULSE initial tokens exceed capacity")
        require(value["refill_per_tick"] > 0, "invalid_pulse_contract", "PULSE refill rate must be positive")
    if task == "backpressure":
        require(is_int(value["capacity"]) and value["capacity"] > 0, "invalid_pulse_contract", "PULSE queue capacity must be positive")
        require(is_int(value["initial_depth"]) and 0 <= value["initial_depth"] <= value["capacity"], "invalid_pulse_contract", "PULSE initial queue depth is invalid")
    if task == "sliding-window":
        require(is_int(value["capacity"]) and value["capacity"] > 0, "invalid_pulse_contract", "PULSE sliding-window capacity must be positive")
        require(is_int(value["window_ticks"]) and value["window_ticks"] > 0, "invalid_pulse_contract", "PULSE sliding-window duration must be positive")
    fixtures = value["named_fixtures"]
    require(isinstance(fixtures, list), "invalid_pulse_contract", "PULSE named fixtures must be a list")
    for index, fixture in enumerate(fixtures):
        if task == "reorder-buffer":
            require(isinstance(fixture, list), "invalid_pulse_contract", "PULSE reorder fixture must be a list", index=index)
            previous = -1
            seen: set[int] = set()
            for event_index, event in enumerate(fixture):
                require(isinstance(event, dict) and set(event) == {"at", "index"}, "invalid_pulse_contract", "PULSE reorder event is invalid", index=index, event=event_index)
                require(is_int(event["at"]) and event["at"] >= previous and is_int(event["index"]) and 0 <= event["index"] <= value["max_index"] and event["index"] not in seen, "invalid_pulse_contract", "PULSE reorder event fields are invalid", index=index, event=event_index)
                previous = event["at"]
                seen.add(event["index"])
        elif task in {"cancelable-timeout", "two-of-three", "multi-topic-barrier", "city-clock", "backpressure", "warm-failover", "exactly-once", "shared-deadline", "circuit-breaker"}:
            noun = "timeout" if task == "cancelable-timeout" else "quorum" if task == "two-of-three" else "barrier" if task == "multi-topic-barrier" else "city-clock" if task == "city-clock" else "backpressure" if task == "backpressure" else "warm-failover" if task == "warm-failover" else "exactly-once" if task == "exactly-once" else "shared-deadline"
            field = "kind" if task == "cancelable-timeout" else "source"
            allowed = {"start", "cancel"} if task == "cancelable-timeout" else {"A", "B", "C"} if task == "two-of-three" else {"wind", "bridge", "hospital"} if task == "multi-topic-barrier" else {"archive", "controller", "policy"} if task == "city-clock" else {"request", "drain"} if task == "backpressure" else {"request", "warm", "fail", "recover"} if task == "warm-failover" else {"A", "B", "fail", "recover"} if task == "exactly-once" else {"request", "failure", "success"} if task == "circuit-breaker" else {"A", "B"}
            require(isinstance(fixture, list), "invalid_pulse_contract", f"PULSE {noun} fixture must be a list", index=index)
            previous = -1
            for command_index, command in enumerate(fixture):
                require(isinstance(command, dict) and set(command) == {"at", field}, "invalid_pulse_contract", f"PULSE {noun} command is invalid", index=index, command=command_index)
                require(is_int(command["at"]) and command["at"] >= previous and command[field] in allowed, "invalid_pulse_contract", f"PULSE {noun} command fields are invalid", index=index, command=command_index)
                previous = command["at"]
        else:
            require(isinstance(fixture, list) and all(is_int(item) and item >= 0 for item in fixture), "invalid_pulse_contract", "PULSE fixture is invalid", index=index)
            require(fixture == sorted(fixture), "invalid_pulse_contract", "PULSE fixture times must be sorted", index=index)
    limits = value["limits"]
    limit_keys = {"max_program_bytes", "max_state_cells", "max_steps", "max_queue", "max_scheduled_events"}
    require(isinstance(limits, dict) and set(limits) == limit_keys, "invalid_pulse_contract", "PULSE limit fields are invalid")
    require(all(is_int(item) and item > 0 for item in limits.values()), "invalid_pulse_contract", "PULSE limits must be positive integers")
    if task in {"debounce", "cancelable-timeout", "shared-deadline", "reorder-buffer"}:
        try:
            cre.canonical_bytes(value["output_payload"])
        except cre.CREError as exc:
            raise PulseError("invalid_pulse_contract", "PULSE output payload is not a CRE value") from exc
    return value


def compile_program(value: Any, contract: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"format", "cells", "handlers"}, "invalid_pulse_program", "PULSE program fields are invalid")
    require(value["format"] == "afterimage-pulse/0.1", "invalid_pulse_program", "PULSE program format is unsupported")
    program_bytes = len(cre.canonical_bytes(value))
    require(program_bytes <= contract["limits"]["max_program_bytes"], "pulse_limit", "PULSE program exceeds byte limit", observed=program_bytes)
    require(isinstance(value["cells"], list), "invalid_pulse_program", "PULSE cells must be a list")
    require(len(value["cells"]) <= contract["limits"]["max_state_cells"], "pulse_limit", "PULSE program exceeds state-cell limit")
    cell_types: dict[str, str] = {}
    initial: dict[str, Any] = {}
    for index, cell in enumerate(value["cells"]):
        require(isinstance(cell, dict) and set(cell) == {"name", "type", "initial"}, "invalid_pulse_program", "PULSE cell fields are invalid", index=index)
        name, kind = cell["name"], cell["type"]
        require(isinstance(name, str) and name and name not in cell_types, "invalid_pulse_program", "PULSE cell name is invalid", index=index)
        require(kind in {"int", "bool"} and value_type(cell["initial"]) == kind, "invalid_pulse_program", "PULSE cell type or initial value is invalid", cell=name)
        cell_types[name] = kind
        initial[name] = cell["initial"]

    def expression_type(expr: Any, path: str) -> str:
        require(isinstance(expr, list) and expr and isinstance(expr[0], str), "invalid_pulse_program", "PULSE expression is invalid", path=path)
        op = expr[0]
        if op == "const":
            require(len(expr) == 2, "invalid_pulse_program", "const expression arity is invalid", path=path)
            try:
                cre.canonical_bytes(expr[1])
            except cre.CREError as exc:
                raise PulseError("invalid_pulse_program", "const expression is not a CRE value", {"path": path}) from exc
            return value_type(expr[1])
        if op == "cell":
            require(len(expr) == 2 and expr[1] in cell_types, "invalid_pulse_program", "cell expression names an unknown cell", path=path)
            return cell_types[expr[1]]
        if op == "event":
            require(len(expr) == 2 and expr[1] in {"at", "payload"}, "invalid_pulse_program", "event expression selector is invalid", path=path)
            return "int" if expr[1] == "at" else "value"
        binary = {"add": ("int", "int"), "sub": ("int", "int"), "mul": ("int", "int"), "min": ("int", "int"), "lt": ("int", "bool"), "le": ("int", "bool"), "and": ("bool", "bool"), "or": ("bool", "bool")}
        if op in binary:
            argument_type, result_type = binary[op]
            require(len(expr) == 3, "invalid_pulse_program", "binary expression arity is invalid", path=path)
            require(expression_type(expr[1], f"{path}/1") == argument_type and expression_type(expr[2], f"{path}/2") == argument_type, "invalid_pulse_program", "binary expression types are invalid", path=path)
            return result_type
        if op == "eq":
            require(len(expr) == 3, "invalid_pulse_program", "eq expression arity is invalid", path=path)
            left, right = expression_type(expr[1], f"{path}/1"), expression_type(expr[2], f"{path}/2")
            require(left == right or "value" in {left, right}, "invalid_pulse_program", "eq expression types are invalid", path=path)
            return "bool"
        if op == "not":
            require(len(expr) == 2 and expression_type(expr[1], f"{path}/1") == "bool", "invalid_pulse_program", "not expression is invalid", path=path)
            return "bool"
        raise PulseError("invalid_pulse_program", "PULSE expression operator is unsupported", {"path": path, "operator": op})

    def validate_actions(actions: Any, path: str, depth: int = 0) -> None:
        require(isinstance(actions, list), "invalid_pulse_program", "PULSE actions must be a list", path=path)
        require(depth <= 8, "invalid_pulse_program", "PULSE conditional nesting is too deep", path=path)
        for index, action in enumerate(actions):
            here = f"{path}/{index}"
            require(isinstance(action, dict) and isinstance(action.get("op"), str), "invalid_pulse_program", "PULSE action is invalid", path=here)
            op = action["op"]
            if op == "set":
                require(set(action) == {"op", "cell", "value"} and action["cell"] in cell_types, "invalid_pulse_program", "set action fields are invalid", path=here)
                require(expression_type(action["value"], f"{here}/value") == cell_types[action["cell"]], "invalid_pulse_program", "set action type is invalid", path=here)
            elif op == "schedule":
                require(set(action) == {"op", "key", "topic", "at", "payload"}, "invalid_pulse_program", "schedule action fields are invalid", path=here)
                require(isinstance(action["key"], str) and action["key"] and action["topic"] == contract["timer_topic"], "invalid_pulse_program", "schedule key or topic is invalid", path=here)
                require(expression_type(action["at"], f"{here}/at") == "int", "invalid_pulse_program", "schedule time must be Int", path=here)
                expression_type(action["payload"], f"{here}/payload")
            elif op == "cancel":
                require(set(action) == {"op", "key"} and isinstance(action["key"], str) and action["key"], "invalid_pulse_program", "cancel action fields are invalid", path=here)
            elif op == "emit":
                output_topics = set(contract["output_topics"].values()) if contract.get("task") in {"warm-failover", "exactly-once", "circuit-breaker"} else {contract["output_topic"]}
                require(set(action) == {"op", "topic", "payload"} and action["topic"] in output_topics, "invalid_pulse_program", "emit action fields are invalid", path=here)
                expression_type(action["payload"], f"{here}/payload")
            elif op == "when":
                require(set(action) == {"op", "condition", "actions"}, "invalid_pulse_program", "when action fields are invalid", path=here)
                require(expression_type(action["condition"], f"{here}/condition") == "bool", "invalid_pulse_program", "when condition must be Bool", path=here)
                validate_actions(action["actions"], f"{here}/actions", depth + 1)
            else:
                raise PulseError("invalid_pulse_program", "PULSE action operator is unsupported", {"path": here, "operator": op})

    handlers = value["handlers"]
    require(isinstance(handlers, list) and handlers, "invalid_pulse_program", "PULSE handlers must be a non-empty list")
    handler_ids: set[str] = set()
    allowed_topics = set(contract.get("input_topics", {}).values()) | {contract.get("input_topic"), contract["timer_topic"]}
    allowed_topics.discard(None)
    for index, handler in enumerate(handlers):
        require(isinstance(handler, dict) and set(handler) == {"id", "on", "actions"}, "invalid_pulse_program", "PULSE handler fields are invalid", index=index)
        require(isinstance(handler["id"], str) and handler["id"] and handler["id"] not in handler_ids, "invalid_pulse_program", "PULSE handler ID is invalid", index=index)
        require(handler["on"] in allowed_topics, "invalid_pulse_program", "PULSE handler topic is invalid", index=index)
        handler_ids.add(handler["id"])
        validate_actions(handler["actions"], f"/handlers/{index}/actions")
    return {"value": value, "initial": initial, "cell_types": cell_types, "program_bytes": program_bytes}


def evaluate(expr: list[Any], state: dict[str, Any], event: dict[str, Any]) -> Any:
    op = expr[0]
    if op == "const":
        return expr[1]
    if op == "cell":
        return state[expr[1]]
    if op == "event":
        return event[expr[1]]
    if op == "not":
        return not evaluate(expr[1], state, event)
    left = evaluate(expr[1], state, event)
    right = evaluate(expr[2], state, event)
    return {
        "add": lambda: left + right,
        "sub": lambda: left - right,
        "mul": lambda: left * right,
        "min": lambda: min(left, right),
        "eq": lambda: cre.same_value(left, right),
        "lt": lambda: left < right,
        "le": lambda: left <= right,
        "and": lambda: left and right,
        "or": lambda: left or right,
    }[op]()


@dataclass
class RunResult:
    outputs: list[dict[str, Any]]
    steps: int
    scheduled: int
    max_queue: int


def input_events(contract: dict[str, Any], inputs: Iterable[Any]) -> list[dict[str, Any]]:
    events = []
    for sequence, item in enumerate(inputs):
        if contract.get("task") == "cancelable-timeout":
            at, kind = item
            event_id = cre.digest_id("afterimage/pulse-input/1", cre.canonical_bytes([sequence, at, kind]))
            events.append({"id": event_id, "at": at, "topic": contract["input_topics"][kind], "payload": {"sequence": sequence, "kind": kind}})
        elif contract.get("task") == "reorder-buffer":
            at, index = item
            source = str(index)
            event_id = cre.digest_id("afterimage/pulse-input/1", cre.canonical_bytes([sequence, at, index]))
            events.append({"id": event_id, "at": at, "topic": contract["input_topics"][source], "payload": {"sequence": sequence, "index": index}})
        elif contract.get("task") in {"two-of-three", "multi-topic-barrier", "city-clock", "backpressure", "warm-failover", "exactly-once", "shared-deadline", "circuit-breaker"}:
            at, source = item
            event_id = cre.digest_id("afterimage/pulse-input/1", cre.canonical_bytes([sequence, at, source]))
            events.append({"id": event_id, "at": at, "topic": contract["input_topics"][source], "payload": {"sequence": sequence, "source": source}})
        else:
            at = item
            event_id = cre.digest_id("afterimage/pulse-input/1", cre.canonical_bytes([sequence, at]))
            events.append({"id": event_id, "at": at, "topic": contract["input_topic"], "payload": {"sequence": sequence}})
    return sorted(events, key=lambda event: (event["at"], cre.parse_id(event["id"])))


def run(compiled: dict[str, Any], contract: dict[str, Any], pulse_times: Iterable[Any]) -> RunResult:
    program = compiled["value"]
    state = dict(compiled["initial"])
    queue: list[tuple[int, int, bytes, int, dict[str, Any]]] = []
    serial = 0
    for event in input_events(contract, pulse_times):
        heapq.heappush(queue, (event["at"], 0, cre.parse_id(event["id"]), serial, event))
        serial += 1
    active_timers: dict[str, int] = {}
    timer_generation = 0
    scheduled = 0
    steps = 0
    max_queue = len(queue)
    outputs: list[dict[str, Any]] = []
    handlers = sorted(program["handlers"], key=lambda item: item["id"].encode("utf-8"))
    limits = contract["limits"]

    def execute(actions: list[dict[str, Any]], event: dict[str, Any]) -> None:
        nonlocal serial, timer_generation, scheduled, steps, max_queue
        for action in actions:
            steps += 1
            require(steps <= limits["max_steps"], "pulse_limit", "PULSE execution exceeds step limit")
            op = action["op"]
            if op == "set":
                state[action["cell"]] = evaluate(action["value"], state, event)
            elif op == "schedule":
                at = evaluate(action["at"], state, event)
                require(at >= event["at"], "pulse_runtime", "PULSE timer was scheduled in the past")
                payload = evaluate(action["payload"], state, event)
                timer_generation += 1
                scheduled += 1
                require(scheduled <= limits["max_scheduled_events"], "pulse_limit", "PULSE execution exceeds scheduled-event limit")
                key = action["key"]
                active_timers[key] = timer_generation
                timer_id = cre.digest_id("afterimage/pulse-timer/1", cre.canonical_bytes([key, at, payload, timer_generation, event["id"]]))
                timer = {"id": timer_id, "at": at, "topic": action["topic"], "payload": payload, "timer_key": key, "timer_generation": timer_generation}
                heapq.heappush(queue, (at, 1, cre.parse_id(timer_id), serial, timer))
                serial += 1
                max_queue = max(max_queue, len(queue))
                require(max_queue <= limits["max_queue"], "pulse_limit", "PULSE execution exceeds queue limit")
            elif op == "cancel":
                active_timers.pop(action["key"], None)
            elif op == "emit":
                outputs.append({"at": event["at"], "topic": action["topic"], "payload": evaluate(action["payload"], state, event)})
            elif op == "when" and evaluate(action["condition"], state, event):
                execute(action["actions"], event)

    while queue:
        _at, _phase, _identity, _serial, event = heapq.heappop(queue)
        if "timer_key" in event:
            if active_timers.get(event["timer_key"]) != event["timer_generation"]:
                continue
            del active_timers[event["timer_key"]]
        for handler in handlers:
            if handler["on"] == event["topic"]:
                execute(handler["actions"], event)
    return RunResult(outputs=outputs, steps=steps, scheduled=scheduled, max_queue=max_queue)


def domain(contract: dict[str, Any]) -> list[tuple[Any, ...]]:
    if contract.get("task") == "reorder-buffer":
        generated: set[tuple[tuple[int, int], ...]] = set()
        indexes = tuple(range(contract["max_index"] + 1))
        for length in range(min(contract["max_events"], len(indexes)) + 1):
            for times in itertools.combinations_with_replacement(range(contract["horizon"]), length):
                for order in itertools.permutations(indexes, length):
                    generated.add(tuple(zip(times, order)))
        named = {tuple((event["at"], event["index"]) for event in fixture) for fixture in contract["named_fixtures"]}
        return sorted(generated | named)
    if contract.get("task") in {"cancelable-timeout", "two-of-three", "multi-topic-barrier", "city-clock", "backpressure", "warm-failover", "exactly-once", "shared-deadline", "circuit-breaker"}:
        generated = set()
        kinds = tuple(sorted(contract["input_topics"]))
        count = contract["max_commands"] if contract.get("task") in {"cancelable-timeout", "backpressure", "warm-failover", "exactly-once", "circuit-breaker"} else contract["max_votes"] if contract.get("task") == "two-of-three" else contract["max_signals"]
        for length in range(count + 1):
            for times in itertools.combinations_with_replacement(range(contract["horizon"]), length):
                for command_kinds in itertools.product(kinds, repeat=length):
                    generated.add(tuple(zip(times, command_kinds)))
        named = {
            tuple((command["at"], command["kind" if contract.get("task") == "cancelable-timeout" else "source"]) for command in fixture)
            for fixture in contract["named_fixtures"]
        }
        return sorted(generated | named)
    combinations = (
        itertools.combinations_with_replacement
        if contract.get("task") in {"deduplicate-ticks", "token-bucket", "sliding-window"}
        else itertools.combinations
    )
    generated = itertools.chain.from_iterable(
        combinations(range(contract["horizon"]), length)
        for length in range(contract.get("max_requests", contract.get("max_pulses", 0)) + 1)
    )
    return sorted(set(generated) | {tuple(item) for item in contract["named_fixtures"]})


def expected_times(pulses: tuple[int, ...], quiet_ticks: int) -> list[int]:
    if not pulses:
        return []
    result: list[int] = []
    last = pulses[0]
    for at in pulses[1:]:
        if at - last > quiet_ticks:
            result.append(last + quiet_ticks)
        last = at
    result.append(last + quiet_ticks)
    return result


def expected_outputs(contract: dict[str, Any], pulses: tuple[Any, ...]) -> list[dict[str, Any]]:
    task = contract.get("task", "debounce")
    if task == "debounce":
        return [
            {"at": at, "topic": contract["output_topic"], "payload": contract["output_payload"]}
            for at in expected_times(pulses, contract["quiet_ticks"])
        ]
    if task == "cancelable-timeout":
        outputs = []
        deadline: int | None = None
        for event in input_events(contract, pulses):
            if deadline is not None and deadline < event["at"]:
                outputs.append({"at": deadline, "topic": contract["output_topic"], "payload": contract["output_payload"]})
                deadline = None
            if event["topic"] == contract["input_topics"]["start"]:
                deadline = event["at"] + contract["timeout_ticks"]
            else:
                deadline = None
        if deadline is not None:
            outputs.append({"at": deadline, "topic": contract["output_topic"], "payload": contract["output_payload"]})
        return outputs
    if task == "token-bucket":
        outputs = []
        tokens = contract["initial_tokens"]
        last_at = 0
        for event in input_events(contract, pulses):
            tokens = min(contract["capacity"], tokens + (event["at"] - last_at) * contract["refill_per_tick"])
            last_at = event["at"]
            if tokens > 0:
                outputs.append({"at": event["at"], "topic": contract["output_topic"], "payload": event["payload"]})
                tokens -= 1
        return outputs
    if task == "two-of-three":
        outputs = []
        seen: set[str] = set()
        fired = False
        for event in input_events(contract, pulses):
            seen.add(event["payload"]["source"])
            if not fired and len(seen) >= 2:
                outputs.append({"at": event["at"], "topic": contract["output_topic"], "payload": event["payload"]})
                fired = True
        return outputs
    if task == "multi-topic-barrier":
        outputs = []
        seen: set[str] = set()
        fired = False
        for event in input_events(contract, pulses):
            seen.add(event["payload"]["source"])
            if not fired and seen == set(contract["input_topics"]):
                outputs.append({"at": event["at"], "topic": contract["output_topic"], "payload": event["payload"]})
                fired = True
        return outputs
    if task == "city-clock":
        outputs = []
        seen: set[str] = set()
        for event in input_events(contract, pulses):
            seen.add(event["payload"]["source"])
            if seen == set(contract["input_topics"]):
                outputs.append({"at": event["at"], "topic": contract["output_topic"], "payload": event["payload"]})
                seen.clear()
        return outputs
    if task == "backpressure":
        outputs = []
        depth = contract["initial_depth"]
        for event in input_events(contract, pulses):
            if event["payload"]["source"] == "drain":
                depth = max(0, depth - 1)
            elif depth < contract["capacity"]:
                outputs.append({"at": event["at"], "topic": contract["output_topic"], "payload": event["payload"]})
                depth += 1
        return outputs
    if task == "warm-failover":
        outputs = []
        primary_up = True
        secondary_warm = False
        for event in input_events(contract, pulses):
            source = event["payload"]["source"]
            if source == "warm":
                secondary_warm = True
            elif source == "fail":
                primary_up = False
            elif source == "recover":
                primary_up = True
            elif primary_up:
                outputs.append({"at": event["at"], "topic": contract["output_topics"]["primary"], "payload": event["payload"]})
            elif secondary_warm:
                outputs.append({"at": event["at"], "topic": contract["output_topics"]["secondary"], "payload": event["payload"]})
        return outputs
    if task == "exactly-once":
        outputs = []
        primary_up = True
        seen: set[str] = set()
        for event in input_events(contract, pulses):
            source = event["payload"]["source"]
            if source == "fail":
                primary_up = False
            elif source == "recover":
                primary_up = True
            elif source not in seen:
                seen.add(source)
                route = "primary" if primary_up else "secondary"
                outputs.append({"at": event["at"], "topic": contract["output_topics"][route], "payload": event["payload"]})
        return outputs
    if task == "shared-deadline":
        outputs = []
        deadline: int | None = None
        for event in input_events(contract, pulses):
            if deadline is not None and deadline < event["at"]:
                outputs.append({"at": deadline, "topic": contract["output_topic"], "payload": contract["output_payload"]})
                deadline = None
            candidate = event["at"] + contract["offsets"][event["payload"]["source"]]
            deadline = candidate if deadline is None else min(deadline, candidate)
        if deadline is not None:
            outputs.append({"at": deadline, "topic": contract["output_topic"], "payload": contract["output_payload"]})
        return outputs
    if task == "sliding-window":
        outputs = []
        admitted: list[int] = []
        for event in input_events(contract, pulses):
            admitted = [at for at in admitted if at > event["at"] - contract["window_ticks"]]
            if len(admitted) < contract["capacity"]:
                outputs.append({"at": event["at"], "topic": contract["output_topic"], "payload": event["payload"]})
                admitted.append(event["at"])
        return outputs
    if task == "reorder-buffer":
        outputs = []
        buffered: set[int] = set()
        next_index = 0
        for event in input_events(contract, pulses):
            buffered.add(event["payload"]["index"])
            while next_index in buffered:
                buffered.remove(next_index)
                outputs.append({"at": event["at"], "topic": contract["output_topic"], "payload": contract["output_payload"]})
                next_index += 1
        return outputs
    if task == "circuit-breaker":
        outputs = []
        mode = 0  # closed=0, open=1, half-open=2
        failures = 0
        opened_at = -contract["cooldown_ticks"]
        for event in input_events(contract, pulses):
            source = event["payload"]["source"]
            if source == "request":
                if mode == 0:
                    outputs.append({"at": event["at"], "topic": contract["output_topics"]["admit"], "payload": event["payload"]})
                elif mode == 1 and event["at"] - opened_at >= contract["cooldown_ticks"]:
                    outputs.append({"at": event["at"], "topic": contract["output_topics"]["probe"], "payload": event["payload"]})
                    mode = 2
            elif source == "failure":
                if mode == 2:
                    mode = 1
                    opened_at = event["at"]
                elif mode == 0:
                    failures += 1
                    if failures >= contract["failure_threshold"]:
                        mode = 1
                        failures = 0
                        opened_at = event["at"]
            elif source == "success":
                if mode == 2:
                    mode = 0
                    failures = 0
                elif mode == 0:
                    failures = 0
        return outputs
    outputs = []
    previous: int | None = None
    for event in input_events(contract, pulses):
        if event["at"] != previous:
            outputs.append({"at": event["at"], "topic": contract["output_topic"], "payload": event["payload"]})
            previous = event["at"]
    return outputs


def verify_program(program: Any, contract_value: Any) -> dict[str, int]:
    contract = validate_contract(contract_value)
    compiled = compile_program(program, contract)
    for pulses in domain(contract):
        expected = expected_outputs(contract, pulses)
        try:
            observed = run(compiled, contract, pulses).outputs
        except PulseError as exc:
            raise PulseError("pulse_counterexample", "PULSE program failed on exhaustive domain", {"input": display_input(contract, pulses), "expected": expected, "observed": [], "inner_code": exc.code}) from exc
        if not cre.same_value(observed, expected):
            raise PulseError("pulse_counterexample", "PULSE program disagrees with exhaustive domain", {"input": display_input(contract, pulses), "expected": expected, "observed": observed})
    return {
        "program_bytes": compiled["program_bytes"],
        "worst_latency": max(contract["offsets"].values()) if contract.get("task") == "shared-deadline" else contract["horizon"] - 1 if contract.get("task") == "reorder-buffer" else contract["cooldown_ticks"] if contract.get("task") == "circuit-breaker" else contract.get("quiet_ticks", contract.get("timeout_ticks", 0)),
        "live_state_cells": len(compiled["initial"]),
        "domain_cases": len(domain(contract)),
    }


def display_input(contract: dict[str, Any], inputs: tuple[Any, ...]) -> list[Any]:
    if contract.get("task") == "cancelable-timeout":
        return [{"at": at, "kind": kind} for at, kind in inputs]
    if contract.get("task") == "reorder-buffer":
        return [{"at": at, "index": index} for at, index in inputs]
    if contract.get("task") in {"two-of-three", "multi-topic-barrier", "city-clock", "backpressure", "warm-failover", "exactly-once", "shared-deadline", "circuit-breaker"}:
        return [{"at": at, "source": source} for at, source in inputs]
    return list(inputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PULSE 0.1 runtime library. Player submissions are checked through "
            "player.py verify; this command documents the standalone helper."
        )
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
