"""Deterministically extend the hand-authored CRE baseline into the release matrix."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


TRUE = ["const", True]
ZERO_BUNDLE = "sha256:" + "0" * 64
UNKNOWN_ID = "sha256:" + "0" * 64


def origin(sequence: int) -> dict[str, Any]:
    return {"kind": "base", "source": "matrix", "sequence": sequence}


def event(
    topic: str,
    payload: Any,
    sequence: int,
    *,
    at: int = 0,
    parents: list[str] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    body = {
        "topic": topic,
        "at": at,
        "payload": payload,
        "parents": parents or [],
        "origin": origin(sequence),
    }
    return {"label": label, "body": body} if label is not None else body


def empty_program() -> dict[str, Any]:
    return {"semantics": "cre/0.1", "strata": []}


def positive(alias: str, topic: str, where: Any = TRUE) -> dict[str, Any]:
    return {"alias": alias, "topic": topic, "where": deepcopy(where)}


def rule(
    identifier: str,
    positives: list[dict[str, Any]],
    emissions: list[dict[str, Any]],
    *,
    negative: list[dict[str, Any]] | None = None,
    aggregate: list[dict[str, Any]] | None = None,
    distinct: list[list[str]] | None = None,
    guard: Any = TRUE,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "positive": positives,
        "negative": negative or [],
        "aggregate": aggregate or [],
        "distinct": distinct or [],
        "guard": deepcopy(guard),
        "emit": emissions,
    }


def emission(topic: str, payload: Any, *, at: Any = None, parents: list[Any] | None = None) -> dict[str, Any]:
    return {
        "topic": ["const", topic],
        "at": ["const", 0] if at is None else at,
        "payload": payload,
        "parents": parents or [],
    }


def one_stratum(*rules: dict[str, Any]) -> dict[str, Any]:
    return {"semantics": "cre/0.1", "strata": [{"index": 0, "rules": list(rules)}]}


def projection(topic: str, *, value: Any = None, sort: list[Any] | None = None) -> dict[str, Any]:
    return {
        "id": f"matrix.{topic}",
        "rows": [
            {
                "positive": [positive("x", topic)],
                "negative": [],
                "aggregate": [],
                "distinct": [],
                "guard": deepcopy(TRUE),
                "value": ["get", "x", "/payload"] if value is None else value,
                "sort": sort or [],
            }
        ],
    }


def case(name: str, base_events: list[Any], program: dict[str, Any], **extra: Any) -> dict[str, Any]:
    result = {
        "name": name,
        "bundle_digest": ZERO_BUNDLE,
        "base_events": base_events,
        "program": program,
    }
    result.update(extra)
    return result


def semantic_cases() -> list[dict[str, Any]]:
    same_time = case(
        "dag-same-time-parent",
        [
            event("dag.parent", None, 0, at=5, label="parent"),
            event("dag.child", {"ok": True}, 1, at=5, parents=["@parent"]),
        ],
        empty_program(),
        projection=projection("dag.child"),
    )

    claimed = event("dag.claimed", None, 0)
    claimed["id"] = "sha256:" + "f" * 64
    claimed_mismatch = case("dag-claimed-id-mismatch", [claimed], empty_program())

    missing_parent = case(
        "dag-missing-parent",
        [event("dag.orphan", None, 0, parents=[UNKNOWN_ID])],
        empty_program(),
    )
    reverse_time = case(
        "dag-time-reversal",
        [
            event("dag.parent", None, 0, at=5, label="parent"),
            event("dag.child", None, 1, at=4, parents=["@parent"]),
        ],
        empty_program(),
    )

    block_rule = rule(
        "gate.01-block",
        [positive("t", "gate.trigger")],
        [
            emission(
                "gate.block",
                ["map", "name", ["get", "t", "/payload/name"]],
                at=["get", "t", "/at"],
            )
        ],
    )
    accept_rule = rule(
        "gate.02-accept",
        [positive("q", "gate.request")],
        [
            emission(
                "gate.accepted",
                ["map", "name", ["get", "q", "/payload/name"]],
                at=["get", "q", "/at"],
            )
        ],
        negative=[
            {
                "alias": "b",
                "topic": "gate.block",
                "where": ["eq", ["get", "b", "/payload/name"], ["get", "q", "/payload/name"]],
            }
        ],
    )
    sealed_negation = case(
        "sealed-lower-stratum-negation",
        [
            event("gate.request", {"name": "alpha"}, 0),
            event("gate.request", {"name": "beta"}, 1),
            event("gate.trigger", {"name": "alpha"}, 2),
        ],
        {
            "semantics": "cre/0.1",
            "strata": [
                {"index": 0, "rules": [block_rule]},
                {"index": 1, "rules": [accept_rule]},
            ],
        },
        projection=projection(
            "gate.accepted",
            value=["get", "x", "/payload/name"],
            sort=[["get", "x", "/payload/name"]],
        ),
    )

    get = lambda pointer: ["get", "s", pointer]
    expression_payload = [
        "map",
        "pointer_escape", get("/payload/a~1b/~0key"),
        "list_index", get("/payload/list/1"),
        "list", ["list", ["const", 1], ["const", "two"]],
        "arithmetic", ["list", ["add", ["const", 7], ["const", 5]], ["sub", ["const", 7], ["const", 5]], ["mul", ["const", 7], ["const", 5]], ["div", ["const", -7], ["const", 3]], ["mod", ["const", -7], ["const", 3]]],
        "extrema", ["list", ["min", ["const", 9], ["const", 2], ["const", 5]], ["max", ["const", "a"], ["const", "z"]]],
        "concat_text", ["concat", ["const", "ca"], ["const", "fé"]],
        "concat_bytes", ["concat", ["const", {"$bytes": "AQI"}], ["const", {"$bytes": "_w"}]],
        "lengths", ["list", ["length", ["const", "😀"]], ["length", ["const", {"$bytes": "AAEC"}]], ["length", ["const", [1, 2]]], ["length", ["const", {"k": 1}]]],
        "contains", ["list", ["contains", ["const", [1, 2]], ["const", 2]], ["contains", ["const", {"key": 1}], ["const", "key"]], ["contains", ["const", "afterimage"], ["const", "image"]], ["contains", ["const", {"$bytes": "AAEC"}], ["const", {"$bytes": "AQ"}]]],
        "conditional", ["if", ["const", True], ["const", "yes"], ["const", "no"]],
        "hash", ["hash", "afterimage/matrix-hash/1", ["const", {"x": 1}]],
    ]
    guard = [
        "and",
        ["eq", ["const", 1], ["const", 1]],
        ["ne", ["const", 1], ["const", 2]],
        ["lt", ["const", 1], ["const", 2]],
        ["le", ["const", 2], ["const", 2]],
        ["gt", ["const", 3], ["const", 2]],
        ["ge", ["const", 3], ["const", 3]],
        ["or", ["const", False], ["const", True]],
        ["not", ["const", False]],
    ]
    expression_case = case(
        "expression-opcode-matrix",
        [event("expr.start", {"a/b": {"~key": 7}, "list": [10, 20]}, 0)],
        one_stratum(
            rule(
                "expr.all",
                [positive("s", "expr.start")],
                [
                    emission(
                        "expr.result",
                        expression_payload,
                        parents=[["get", "s", "/id"]],
                    )
                ],
                guard=guard,
            )
        ),
        projection=projection("expr.result"),
    )

    missing_path = case(
        "pointer-missing-path",
        [event("pointer.start", {}, 0)],
        one_stratum(rule("pointer.missing", [positive("s", "pointer.start")], [emission("pointer.result", ["get", "s", "/payload/nope"])])),
    )
    invalid_pointer = case(
        "pointer-invalid-escape",
        [event("pointer.start", {"x": 1}, 0)],
        one_stratum(rule("pointer.invalid", [positive("s", "pointer.start")], [emission("pointer.result", ["get", "s", "/payload/~2"])])),
    )
    invalid_distinct = case(
        "distinct-duplicate-alias",
        [event("distinct.start", None, 0)],
        one_stratum(
            rule(
                "distinct.invalid",
                [positive("a", "distinct.start")],
                [],
                distinct=[["a", "a"]],
            )
        ),
    )

    invalid_replace = case(
        "branch-invalid-replace-pointer",
        [event("branch.base", {"x": 1}, 0, label="base")],
        empty_program(),
        operations=[{"kind": "replace", "event": "@base", "pointer": "/topic", "value": "changed"}],
    )
    conflict = case(
        "branch-operation-conflict",
        [event("branch.base", {"x": 1}, 0, label="base")],
        empty_program(),
        operations=[
            {"kind": "suppress", "event": "@base"},
            {"kind": "retime", "event": "@base", "at": 1},
        ],
    )
    bad_parent = case(
        "branch-inject-missing-parent",
        [event("branch.base", None, 0)],
        empty_program(),
        operations=[
            {"kind": "inject", "topic": "branch.injected", "at": 0, "payload": None, "parents": [UNKNOWN_ID]}
        ],
    )
    non_root_noop = case(
        "branch-non-root-noop",
        [event("branch.base", {"stable": True}, 0)],
        empty_program(),
        parent_branch="sha256:" + "1" * 64,
        operations=[],
    )

    return [
        same_time,
        claimed_mismatch,
        missing_parent,
        reverse_time,
        sealed_negation,
        expression_case,
        missing_path,
        invalid_pointer,
        invalid_distinct,
        invalid_replace,
        conflict,
        bad_parent,
        non_root_noop,
    ]


def resource_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    for label, count in (("below", 1), ("exact", 2), ("over", 3)):
        cases.append(
            case(
                f"resource-base-{label}",
                [event("resource.base", index, index) for index in range(count)],
                empty_program(),
                limits={"max_base_events": 2},
            )
        )

    for label, count in (("below", 1), ("exact", 2), ("over", 3)):
        emissions = [emission("resource.derived", ["const", index]) for index in range(count)]
        cases.append(
            case(
                f"resource-derived-{label}",
                [event("resource.trigger", None, 0)],
                one_stratum(rule("resource.derive", [positive("t", "resource.trigger")], emissions)),
                limits={"max_derived_events": 2},
            )
        )

    for label, count in (("below", 1), ("exact", 2), ("over", 3)):
        cases.append(
            case(
                f"resource-bindings-{label}",
                [event("resource.binding", index, index) for index in range(count)],
                one_stratum(rule("resource.bind", [positive("x", "resource.binding")], [])),
                limits={"max_bindings_tested": 2},
            )
        )

    value_inputs = (("below", 0), ("exact", "abc"), ("over", "abcd"))
    for index, (label, payload) in enumerate(value_inputs):
        cases.append(
            case(
                f"resource-value-bytes-{label}",
                [event("resource.value", payload, index)],
                empty_program(),
                limits={"max_value_bytes": 5},
            )
        )

    for label, count in (("below", 1), ("exact", 2), ("over", 3)):
        cases.append(
            case(
                f"resource-projection-{label}",
                [event("resource.project", index, index) for index in range(count)],
                empty_program(),
                limits={"max_projection_records": 2},
                projection=projection("resource.project"),
            )
        )
    return cases


def extend_suite(base: dict[str, Any]) -> dict[str, Any]:
    suite = deepcopy(base)
    additions = semantic_cases() + resource_cases()
    existing = {item["name"] for item in suite["cases"]}
    duplicate = existing.intersection(item["name"] for item in additions)
    if duplicate:
        raise ValueError(f"duplicate generated case names: {sorted(duplicate)}")
    suite["format"] = "afterimage-conformance/0.1-release"
    suite["cases"].extend(additions)
    return suite
