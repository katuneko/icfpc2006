#!/usr/bin/env python3
"""Exhaustive finite-state checker for Afterimage COVENANT policies."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "python"))

import cre  # noqa: E402

HARD_MAX_FIELDS = 64
HARD_MAX_DOMAIN_VALUES = 256
HARD_MAX_INITIAL_STATES = 256
HARD_MAX_ACTORS = 32
HARD_MAX_ACTIONS = 256
HARD_MAX_PROPERTIES = 64
HARD_MAX_FAIRNESS_WINDOW = 64
HARD_MAX_REACHABLE_STATES = 50_000
HARD_MAX_TRANSITIONS = 500_000
HARD_MAX_EXPRESSION_NODES = 100_000
HARD_MAX_EXPRESSION_DEPTH = 128


class CovenantError(Exception):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}

    def value(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


def fail(code: str, message: str, **context: Any) -> None:
    raise CovenantError(code, message, context)


def require_map(value: Any, keys: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        fail("covenant_schema", "object has wrong fields", path=location, expected=sorted(keys))
    return value


def nonnegative(value: Any, location: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        fail("covenant_schema", "integer bound is invalid", path=location)
    return value


def unique_texts(value: Any, location: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value) or not all(isinstance(item, str) and item for item in value):
        fail("covenant_schema", "expected a list of non-empty Text values", path=location)
    if len(set(value)) != len(value):
        fail("covenant_schema", "Text list contains duplicates", path=location)
    return value


def expression_nodes(expr: Any, depth: int = 0) -> int:
    if depth > HARD_MAX_EXPRESSION_DEPTH:
        fail("covenant_limit", "expression nesting exceeds the hard depth limit")
    if not isinstance(expr, list) or not expr or not isinstance(expr[0], str):
        fail("covenant_expression", "expression must be a non-empty opcode list")
    return 1 + sum(expression_nodes(item, depth + 1) for item in expr[1:] if isinstance(item, list))


def referenced_fields(expr: Any) -> set[str]:
    expression_nodes(expr)
    if expr[0] == "get":
        if len(expr) != 2 or not isinstance(expr[1], str):
            fail("covenant_expression", "get requires one field name")
        return {expr[1]}
    fields: set[str] = set()
    for item in expr[1:]:
        if isinstance(item, list):
            fields.update(referenced_fields(item))
    return fields


def eval_expr(expr: Any, state: dict[str, Any]) -> Any:
    expression_nodes(expr)
    op, args = expr[0], expr[1:]
    if op == "const" and len(args) == 1:
        return cre.normalize_value(args[0])
    if op == "get" and len(args) == 1 and isinstance(args[0], str):
        if args[0] not in state:
            fail("covenant_expression", "get names an unknown state field", field=args[0])
        return state[args[0]]
    if op == "not" and len(args) == 1:
        value = eval_expr(args[0], state)
        if not isinstance(value, bool):
            fail("covenant_expression", "not requires Bool")
        return not value
    if op in {"and", "or"} and len(args) >= 2:
        values = [eval_expr(item, state) for item in args]
        if not all(isinstance(item, bool) for item in values):
            fail("covenant_expression", f"{op} requires Bool operands")
        return all(values) if op == "and" else any(values)
    if op in {"eq", "ne", "lt", "le", "gt", "ge"} and len(args) == 2:
        left, right = (eval_expr(item, state) for item in args)
        if op == "eq":
            return cre.same_value(left, right)
        if op == "ne":
            return not cre.same_value(left, right)
        comparison = cre.compare_values(left, right)
        return {"lt": comparison < 0, "le": comparison <= 0, "gt": comparison > 0, "ge": comparison >= 0}[op]
    if op in {"add", "sub"} and len(args) == 2:
        left, right = (eval_expr(item, state) for item in args)
        if isinstance(left, bool) or isinstance(right, bool) or not isinstance(left, int) or not isinstance(right, int):
            fail("covenant_expression", f"{op} requires Int operands")
        return cre.checked_i64(left + right if op == "add" else left - right, f"covenant {op}")
    if op == "if" and len(args) == 3:
        condition = eval_expr(args[0], state)
        if not isinstance(condition, bool):
            fail("covenant_expression", "if condition requires Bool")
        return eval_expr(args[1] if condition else args[2], state)
    fail("covenant_expression", "unknown opcode or arity", opcode=op)


def predicate(expr: Any, state: dict[str, Any], location: str) -> bool:
    value = eval_expr(expr, state)
    if not isinstance(value, bool):
        fail("covenant_expression", "predicate must evaluate to Bool", path=location)
    return value


@dataclass(frozen=True)
class Model:
    domains: dict[str, list[Any]]
    initial: list[dict[str, Any]]
    actors: list[dict[str, Any]]
    actions: dict[str, dict[str, Any]]
    safety: list[dict[str, Any]]
    liveness: list[dict[str, Any]]
    fairness_window: int
    max_states: int
    max_transitions: int
    max_expression_nodes: int


def validate_contract(value: Any) -> Model:
    contract = require_map(
        value,
        {"format", "domains", "initial", "actors", "actions", "scheduler", "safety", "liveness", "limits"},
        "contract",
    )
    if contract["format"] != "afterimage-covenant-contract/0.1":
        fail("covenant_schema", "contract format is unsupported")
    domains_value = contract["domains"]
    if not isinstance(domains_value, dict) or not domains_value or not all(isinstance(key, str) and key for key in domains_value):
        fail("covenant_schema", "domains must be a non-empty Text-keyed map")
    if len(domains_value) > HARD_MAX_FIELDS:
        fail("covenant_limit", "state field count exceeds the hard limit")
    domains: dict[str, list[Any]] = {}
    for field, values in domains_value.items():
        if not isinstance(values, list) or not values:
            fail("covenant_schema", "each state domain must be non-empty", field=field)
        if len(values) > HARD_MAX_DOMAIN_VALUES:
            fail("covenant_limit", "state domain exceeds the hard value limit", field=field)
        normalized = [cre.normalize_value(item) for item in values]
        encodings = [cre.canonical_bytes(item) for item in normalized]
        if len(set(encodings)) != len(encodings):
            fail("covenant_schema", "state domain contains duplicates", field=field)
        domains[field] = normalized

    def validate_state(raw: Any, location: str) -> dict[str, Any]:
        if not isinstance(raw, dict) or set(raw) != set(domains):
            fail("covenant_schema", "state fields differ from domains", path=location)
        state = {field: cre.normalize_value(raw[field]) for field in domains}
        for field, item in state.items():
            if not any(cre.same_value(item, choice) for choice in domains[field]):
                fail("covenant_schema", "state value is outside its domain", path=location, field=field)
        return state

    if not isinstance(contract["initial"], list) or not contract["initial"]:
        fail("covenant_schema", "initial must contain at least one state")
    if len(contract["initial"]) > HARD_MAX_INITIAL_STATES:
        fail("covenant_limit", "initial state count exceeds the hard limit")
    initial = [validate_state(item, f"contract.initial[{index}]") for index, item in enumerate(contract["initial"])]
    if len({cre.canonical_bytes(item) for item in initial}) != len(initial):
        fail("covenant_schema", "initial states must be unique")

    actors = []
    actor_ids: set[str] = set()
    if not isinstance(contract["actors"], list) or not contract["actors"]:
        fail("covenant_schema", "actors must be a non-empty list")
    if len(contract["actors"]) > HARD_MAX_ACTORS:
        fail("covenant_limit", "actor count exceeds the hard limit")
    for index, raw in enumerate(contract["actors"]):
        actor = require_map(raw, {"id", "kind", "observes", "actions"}, f"contract.actors[{index}]")
        if not isinstance(actor["id"], str) or not actor["id"] or actor["id"] in actor_ids or actor["kind"] not in {"agent", "environment"}:
            fail("covenant_schema", "actor identity or kind is invalid", path=f"contract.actors[{index}]")
        actor_ids.add(actor["id"])
        observes = unique_texts(actor["observes"], f"contract.actors[{index}].observes")
        if set(observes) - set(domains) or (actor["kind"] == "environment" and observes):
            fail("covenant_schema", "actor observes unknown fields or environment observes state")
        actions = unique_texts(actor["actions"], f"contract.actors[{index}].actions", nonempty=True)
        actors.append({"id": actor["id"], "kind": actor["kind"], "observes": observes, "actions": actions})
    if [actor["id"] for actor in actors] != sorted(actor_ids, key=lambda item: item.encode("utf-8")):
        fail("covenant_schema", "actors must be UTF-8 ID sorted")

    actions: dict[str, dict[str, Any]] = {}
    if not isinstance(contract["actions"], list) or not contract["actions"]:
        fail("covenant_schema", "actions must be a non-empty list")
    if len(contract["actions"]) > HARD_MAX_ACTIONS:
        fail("covenant_limit", "action count exceeds the hard limit")
    expression_total = 0
    for index, raw in enumerate(contract["actions"]):
        action = require_map(raw, {"id", "actor", "guard", "updates"}, f"contract.actions[{index}]")
        if not isinstance(action["id"], str) or not action["id"] or action["id"] in actions or action["actor"] not in actor_ids:
            fail("covenant_schema", "action identity or actor is invalid", path=f"contract.actions[{index}]")
        if not isinstance(action["updates"], dict) or set(action["updates"]) - set(domains):
            fail("covenant_schema", "action updates unknown fields", action=action["id"])
        expression_total += expression_nodes(action["guard"])
        for expr in action["updates"].values():
            expression_total += expression_nodes(expr)
        actions[action["id"]] = action
    if list(actions) != sorted(actions, key=lambda item: item.encode("utf-8")):
        fail("covenant_schema", "actions must be UTF-8 ID sorted")
    for actor in actors:
        if any(action_id not in actions or actions[action_id]["actor"] != actor["id"] for action_id in actor["actions"]):
            fail("covenant_schema", "actor action list is inconsistent", actor=actor["id"])
        declared = {action_id for action_id, action in actions.items() if action["actor"] == actor["id"]}
        if set(actor["actions"]) != declared:
            fail("covenant_schema", "actor action list must cover exactly its actions", actor=actor["id"])

    safety = []
    if not isinstance(contract["safety"], list) or not contract["safety"]:
        fail("covenant_schema", "safety must be a non-empty list")
    if len(contract["safety"]) > HARD_MAX_PROPERTIES:
        fail("covenant_limit", "safety property count exceeds the hard limit")
    for index, raw in enumerate(contract["safety"]):
        item = require_map(raw, {"id", "predicate"}, f"contract.safety[{index}]")
        if not isinstance(item["id"], str) or not item["id"]:
            fail("covenant_schema", "safety ID is invalid")
        expression_total += expression_nodes(item["predicate"])
        safety.append(item)
    liveness = []
    if not isinstance(contract["liveness"], list) or not contract["liveness"]:
        fail("covenant_schema", "liveness must be a non-empty list")
    if len(contract["liveness"]) > HARD_MAX_PROPERTIES:
        fail("covenant_limit", "liveness property count exceeds the hard limit")
    for index, raw in enumerate(contract["liveness"]):
        item = require_map(raw, {"id", "trigger", "goal", "bound"}, f"contract.liveness[{index}]")
        if not isinstance(item["id"], str) or not item["id"]:
            fail("covenant_schema", "liveness ID is invalid")
        nonnegative(item["bound"], f"contract.liveness[{index}].bound", positive=True)
        expression_total += expression_nodes(item["trigger"]) + expression_nodes(item["goal"])
        liveness.append(item)
    if len({item["id"] for item in safety}) != len(safety) or len({item["id"] for item in liveness}) != len(liveness):
        fail("covenant_schema", "contract property IDs must be unique")

    scheduler = require_map(contract["scheduler"], {"fairness_window"}, "contract.scheduler")
    limits = require_map(contract["limits"], {"max_reachable_states", "max_transitions", "max_expression_nodes"}, "contract.limits")
    fairness = nonnegative(scheduler["fairness_window"], "contract.scheduler.fairness_window", positive=True)
    max_states = nonnegative(limits["max_reachable_states"], "contract.limits.max_reachable_states", positive=True)
    max_transitions = nonnegative(limits["max_transitions"], "contract.limits.max_transitions", positive=True)
    max_expression_nodes = nonnegative(limits["max_expression_nodes"], "contract.limits.max_expression_nodes", positive=True)
    if fairness > HARD_MAX_FAIRNESS_WINDOW or max_states > HARD_MAX_REACHABLE_STATES or max_transitions > HARD_MAX_TRANSITIONS or max_expression_nodes > HARD_MAX_EXPRESSION_NODES:
        fail("covenant_limit", "contract limit exceeds an implementation hard ceiling")
    if expression_total > max_expression_nodes:
        fail("covenant_limit", "contract expressions exceed max_expression_nodes")
    return Model(domains, initial, actors, actions, safety, liveness, fairness, max_states, max_transitions, max_expression_nodes)


def validate_policy(value: Any, model: Model) -> tuple[dict[str, dict[str, Any]], int]:
    policy = require_map(value, {"format", "agents"}, "policy")
    if policy["format"] != "afterimage-covenant-policy/0.1" or not isinstance(policy["agents"], list):
        fail("covenant_policy", "policy format or agents are invalid")
    agent_models = {actor["id"]: actor for actor in model.actors if actor["kind"] == "agent"}
    result: dict[str, dict[str, Any]] = {}
    nodes = 1
    for index, raw in enumerate(policy["agents"]):
        agent = require_map(raw, {"agent", "rules", "default"}, f"policy.agents[{index}]")
        actor_id = agent["agent"]
        if actor_id not in agent_models or actor_id in result or not isinstance(agent["rules"], list):
            fail("covenant_policy", "policy agent identity or rules are invalid", path=f"policy.agents[{index}]")
        allowed_actions = set(agent_models[actor_id]["actions"])
        if agent["default"] not in allowed_actions:
            fail("covenant_policy", "policy default action is not allowed", agent=actor_id)
        observes = set(agent_models[actor_id]["observes"])
        rules = []
        nodes += 2
        for rule_index, raw_rule in enumerate(agent["rules"]):
            rule = require_map(raw_rule, {"when", "action"}, f"policy.agents[{index}].rules[{rule_index}]")
            if rule["action"] not in allowed_actions:
                fail("covenant_policy", "policy rule action is not allowed", agent=actor_id)
            hidden = referenced_fields(rule["when"]) - observes
            if hidden:
                fail("covenant_locality", "local policy observes hidden state", agent=actor_id, fields=sorted(hidden))
            nodes += 2 + expression_nodes(rule["when"])
            rules.append(rule)
        result[actor_id] = {"rules": rules, "default": agent["default"]}
    if set(result) != set(agent_models):
        fail("covenant_policy", "policy must define every agent exactly once")
    if list(result) != sorted(result, key=lambda item: item.encode("utf-8")):
        fail("covenant_policy", "policy agents must be UTF-8 ID sorted")
    if nodes > model.max_expression_nodes:
        fail("covenant_limit", "policy exceeds max_expression_nodes")
    return result, nodes


def choose_action(policy: dict[str, Any], state: dict[str, Any]) -> str:
    for rule in policy["rules"]:
        if predicate(rule["when"], state, "policy rule"):
            return rule["action"]
    return policy["default"]


def apply_action(model: Model, action: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    updated = dict(state)
    values = {field: eval_expr(expr, state) for field, expr in action["updates"].items()}
    for field, value in values.items():
        if not any(cre.same_value(value, choice) for choice in model.domains[field]):
            fail("covenant_transition", "action update leaves the declared finite domain", action=action["id"], field=field)
        updated[field] = value
    return updated


def verify_policy(contract_value: Any, policy_value: Any) -> dict[str, int]:
    model = validate_contract(contract_value)
    policies, policy_nodes = validate_policy(policy_value, model)
    actor_ids = [actor["id"] for actor in model.actors]
    actor_by_id = {actor["id"]: actor for actor in model.actors}
    worst_response = 0
    transitions = 0
    visited: set[bytes] = set()
    queue: deque[tuple[dict[str, Any], tuple[int, ...], tuple[int | None, ...], list[dict[str, Any]]]] = deque()
    for state in model.initial:
        queue.append((state, tuple(0 for _ in actor_ids), tuple(None for _ in model.liveness), []))

    while queue:
        state, fairness_ages, obligation_ages, trace = queue.popleft()
        normalized_obligations: list[int | None] = []
        for index, obligation in enumerate(model.liveness):
            age = obligation_ages[index]
            if age is not None and age > obligation["bound"]:
                fail("covenant_liveness", "bounded response obligation expired", obligation=obligation["id"], trace=trace)
            if predicate(obligation["goal"], state, f"liveness {obligation['id']} goal"):
                if age is not None:
                    worst_response = max(worst_response, age)
                normalized_obligations.append(None)
            elif age is None and predicate(obligation["trigger"], state, f"liveness {obligation['id']} trigger"):
                normalized_obligations.append(0)
            else:
                normalized_obligations.append(age)
        obligation_tuple = tuple(normalized_obligations)
        key = cre.canonical_bytes({"state": state, "fairness": list(fairness_ages), "obligations": list(obligation_tuple)})
        if key in visited:
            continue
        visited.add(key)
        if len(visited) > model.max_states:
            fail("covenant_limit", "reachable state count exceeds max_reachable_states")
        for invariant in model.safety:
            if not predicate(invariant["predicate"], state, f"safety {invariant['id']}"):
                fail("covenant_safety", "safety invariant failed", invariant=invariant["id"], trace=trace, state=state)

        enabled: dict[str, list[dict[str, Any]]] = {}
        for actor_id in actor_ids:
            actor = actor_by_id[actor_id]
            if actor["kind"] == "agent":
                action_id = choose_action(policies[actor_id], state)
                action = model.actions[action_id]
                if not predicate(action["guard"], state, f"action {action_id} guard"):
                    fail("covenant_policy", "local policy selected a disabled action", agent=actor_id, action=action_id, trace=trace)
                enabled[actor_id] = [action]
            else:
                choices = [model.actions[action_id] for action_id in actor["actions"] if predicate(model.actions[action_id]["guard"], state, f"action {action_id} guard")]
                if choices:
                    enabled[actor_id] = choices
        if not enabled:
            outstanding = [model.liveness[index]["id"] for index, age in enumerate(obligation_tuple) if age is not None]
            if outstanding:
                fail("covenant_liveness", "deadlock leaves bounded obligations outstanding", obligations=outstanding, trace=trace)
            continue
        overdue = [actor for actor in enabled if fairness_ages[actor_ids.index(actor)] >= model.fairness_window]
        selectable = overdue or list(enabled)
        for actor_id in sorted(selectable, key=lambda item: item.encode("utf-8")):
            for action in sorted(enabled[actor_id], key=lambda item: item["id"].encode("utf-8")):
                transitions += 1
                if transitions > model.max_transitions:
                    fail("covenant_limit", "transition exploration exceeds max_transitions")
                next_state = apply_action(model, action, state)
                next_fairness = []
                for index, candidate in enumerate(actor_ids):
                    if candidate not in enabled:
                        next_fairness.append(0)
                    elif candidate == actor_id:
                        next_fairness.append(0)
                    else:
                        next_fairness.append(min(model.fairness_window, fairness_ages[index] + 1))
                next_obligations = tuple(None if age is None else age + 1 for age in obligation_tuple)
                step = {"actor": actor_id, "action": action["id"], "state": next_state}
                queue.append((next_state, tuple(next_fairness), next_obligations, [*trace, step]))

    return {"policy_nodes": policy_nodes, "worst_response_bound": worst_response, "reachable_states": len(visited)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        result: dict[str, Any] = {"valid": True, "metrics": verify_policy(contract, policy), "diagnostics": []}
        code = 0
    except CovenantError as exc:
        result = {"valid": False, "diagnostics": [exc.value()]}
        code = 1
    except (OSError, json.JSONDecodeError) as exc:
        result = {"valid": False, "diagnostics": [{"code": "input_error", "message": str(exc), "context": {}}]}
        code = 2
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True, separators=None if args.pretty else (",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
