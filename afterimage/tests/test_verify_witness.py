#!/usr/bin/env python3
"""End-to-end and adversarial tests for the Afterimage witness verifier."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import authoring  # noqa: E402
import cre  # noqa: E402
import verify_witness as verifier  # noqa: E402
from afterimage.tests.test_afterimage_kit import make_source, write_json  # noqa: E402


def validator_document() -> dict[str, object]:
    true = ["const", True]
    accept = {
        "id": "verify.01-accept",
        "positive": [
            {"alias": "a", "topic": "verify.answer", "where": true},
            {"alias": "r", "topic": "verify.replay", "where": true},
            {
                "alias": "e",
                "topic": "verify.active",
                "where": ["eq", ["get", "e", "/payload/topic"], ["const", "sample.output"]],
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
            {"topic": ["const", "verify.accept"], "at": ["const", 0], "payload": ["const", None], "parents": []}
        ],
    }
    valid = {
        "id": "verify.02-valid",
        "positive": [{"alias": "ok", "topic": "verify.accept", "where": true}],
        "negative": [],
        "aggregate": [],
        "distinct": [],
        "guard": true,
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
        "id": "verify.03-invalid",
        "positive": [{"alias": "a", "topic": "verify.answer", "where": true}],
        "negative": [{"alias": "ok", "topic": "verify.accept", "where": true}],
        "aggregate": [],
        "distinct": [],
        "guard": true,
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
                            "message", ["const", "submitted event fields do not identify the replayed output"],
                            "context", ["map"],
                        ],
                    ],
                    "metrics", ["map"],
                ],
                "parents": [],
            }
        ],
    }
    decision_projection = {
        "id": "verify.decision",
        "rows": [
            {
                "positive": [{"alias": "d", "topic": "verify.decision", "where": true}],
                "negative": [],
                "aggregate": [],
                "distinct": [],
                "guard": true,
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
        "decision_projection": decision_projection,
    }


def case_descriptor() -> dict[str, object]:
    limits = {
        "max_witness_bytes": 65536,
        "replay": {
            "max_base_events": 100,
            "max_derived_events": 100,
            "max_bindings_tested": 1000,
            "max_value_bytes": 1048576,
            "max_projection_records": 100,
        },
        "validator": {
            "max_base_events": 200,
            "max_derived_events": 100,
            "max_bindings_tested": 10000,
            "max_value_bytes": 1048576,
            "max_projection_records": 10,
        },
    }
    return {
        "id": "ORIENT.001",
        "family": "ORIENT",
        "title": "The First Bell",
        "points": 40,
        "requires": {"all": []},
        "input_branch": "root",
        "world": "global",
        "projection": "sample.public",
        "answer_schema": "cases/ORIENT.001/answer.schema.json",
        "validator": "cases/ORIENT.001/validator.cre.json",
        "intervention_policy": "cases/ORIENT.001/interventions.json",
        "score": "cases/ORIENT.001/score.json",
        "limits": limits,
    }


def augment_source(source: Path) -> None:
    schema = {
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
    policy = {
        "format": "afterimage-intervention-policy/0.1",
        "required": False,
        "allowed_kinds": [],
        "max_operations": 0,
        "weights": {},
        "topics": [],
        "pointers": [],
        "retime": {"minimum": 0, "maximum": 0},
    }
    score = {
        "format": "afterimage-score/0.1",
        "family": "ORIENT",
        "reference_scale": 16,
        "metric_bounds": {"witness_units": 4096},
    }
    write_json(source / "cases/index.json", {"format": "afterimage-cases/0.1", "cases": [case_descriptor()]})
    write_json(source / "cases/ORIENT.001/answer.schema.json", schema)
    write_json(source / "cases/ORIENT.001/validator.cre.json", validator_document())
    write_json(source / "cases/ORIENT.001/interventions.json", policy)
    write_json(source / "cases/ORIENT.001/score.json", score)


class WitnessVerifierTests(unittest.TestCase):
    def test_cascade_score_uses_policy_weight_filtered_footprint_and_lexicographic_packing(self) -> None:
        normal = cre.make_event({
            "topic": "service.changed", "at": 1, "payload": None, "parents": [],
            "origin": {"kind": "base", "source": "score-test", "sequence": 0},
        })
        diagnostic = cre.make_event({
            "topic": "diagnostic.trace", "at": 1, "payload": None, "parents": [],
            "origin": {"kind": "base", "source": "score-test", "sequence": 1},
        })
        baseline = verifier.ReplayState(
            branch="sha256:" + "1" * 64, projection="sha256:" + "2" * 64,
            trace="sha256:" + "3" * 64, records=[], counters={},
            events={normal["id"]: normal, diagnostic["id"]: diagnostic}, trace_items=[],
        )
        replay = verifier.ReplayResult(
            branch="sha256:" + "4" * 64, projection="sha256:" + "5" * 64,
            trace="sha256:" + "6" * 64, records=[], counters={}, events={}, trace_items=[],
            baseline=baseline, changed_event_ids=sorted([normal["id"], diagnostic["id"]], key=cre.parse_id),
        )
        answer = {"contracts": "ok"}
        metrics, score = verifier.cascade_score(
            {"family": "CASCADE", "points": 80},
            authoring.cascade_score(1000, max_causal_footprint=10, max_witness_units=10, diagnostic_topics=["diagnostic.trace"]),
            answer,
            {"contract_violations": 0},
            verifier.PolicyResult(operations=[], weight=2),
            replay,
        )
        units = (len(cre.canonical_bytes(answer)) + 63) // 64
        self.assertEqual(metrics["causal_footprint"], 1)
        self.assertEqual(metrics["intervention_weight"], 2)
        self.assertEqual(metrics["effective_cost"], ((2 * 11 + 1) * 11 + units) + 1)
        self.assertGreaterEqual(score["total"], 52)
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="afterimage-verifier-test-")
        self.root = Path(self.temporary.name)
        source = make_source(self.root)
        augment_source(source)
        bundle_path = self.root / "world.afterimage"
        bundle = kit.pack_bundle(source, bundle_path, "Verifier World", "verify-1")
        self.world_path = self.root / "world"
        kit.extract_bundle(bundle, self.world_path)
        self.world = kit.verify_world(self.world_path)
        self.case = verifier.validate_case_descriptor(self.world.json_values["cases/index.json"]["cases"][0], self.world)
        parent = cre.root_branch_id(self.world.bundle)
        self.replay = verifier.replay_case(self.world, self.case, [], parent)
        output = [event for event in self.replay.events.values() if event["topic"] == "sample.output"]
        self.assertEqual(len(output), 1)
        self.answer = {
            "event_id": output[0]["id"],
            "topic": output[0]["topic"],
            "at": output[0]["at"],
            "projection": self.replay.projection,
        }
        self.witness = {
            "format": "afterimage-witness/0.1",
            "semantics": "cre/0.1",
            "bundle": self.world.bundle,
            "case": "ORIENT.001",
            "parent_branch": parent,
            "intervention": None,
            "answer": self.answer,
            "claimed": {
                "branch": self.replay.branch,
                "projection": self.replay.projection,
                "trace": self.replay.trace,
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_witness(self, value: object, name: str = "witness.json") -> Path:
        path = self.root / name
        path.write_bytes(cre.canonical_bytes(value))
        return path

    def test_valid_witness_receipt_score_and_cli(self) -> None:
        path = self.write_witness(self.witness)
        receipt = verifier.verify_witness(self.world_path, path)
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["unlocks"], ["case:ORIENT.001"])
        self.assertEqual(receipt["metrics"]["wrong_or_redundant_claims"], 0)
        self.assertGreaterEqual(receipt["score"]["total"], 26)
        self.assertLess(receipt["score"]["total"], 40)
        self.assertEqual(receipt["diagnostics"], [])

        command = [sys.executable, str(ROOT / "tools/verify_witness.py"), str(self.world_path), str(path)]
        completed = subprocess.run(command, check=True, capture_output=True)
        self.assertEqual(json.loads(completed.stdout), receipt)

    def test_wrong_event_claim_is_invalid_without_expected_id_leak(self) -> None:
        witness = copy.deepcopy(self.witness)
        witness["answer"]["topic"] = "sample.wrong"
        receipt = verifier.verify_witness(self.world_path, self.write_witness(witness, "wrong.json"))
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "claim_mismatch")
        self.assertNotIn(self.answer["event_id"], json.dumps(receipt))
        self.assertNotIn("score", receipt)
        self.assertNotIn("unlocks", receipt)

    def test_answer_schema_claimed_and_parent_failures_are_stable(self) -> None:
        missing = copy.deepcopy(self.witness)
        del missing["answer"]["at"]
        receipt = verifier.verify_witness(self.world_path, self.write_witness(missing, "missing.json"))
        self.assertEqual(receipt["diagnostics"][0]["code"], "answer_schema")

        claimed = copy.deepcopy(self.witness)
        claimed["claimed"]["trace"] = "sha256:" + "0" * 64
        receipt = verifier.verify_witness(self.world_path, self.write_witness(claimed, "claimed.json"))
        self.assertEqual(receipt["diagnostics"][0], {
            "code": "claimed_mismatch",
            "message": "claimed digest does not match replay",
            "context": {"field": "trace"},
        })

        parent = copy.deepcopy(self.witness)
        parent["parent_branch"] = "sha256:" + "0" * 64
        receipt = verifier.verify_witness(self.world_path, self.write_witness(parent, "parent.json"))
        self.assertEqual(receipt["diagnostics"][0]["code"], "parent_branch_mismatch")

    def test_intervention_policy_and_bundle_mismatch(self) -> None:
        intervention = copy.deepcopy(self.witness)
        intervention["intervention"] = {
            "format": "afterimage-intervention/0.1",
            "bundle": self.world.bundle,
            "parent_branch": self.witness["parent_branch"],
            "case": "ORIENT.001",
            "operations": [],
        }
        receipt = verifier.verify_witness(self.world_path, self.write_witness(intervention, "intervention.json"))
        self.assertEqual(receipt["diagnostics"][0]["code"], "unexpected_intervention")

        forbidden = copy.deepcopy(intervention)
        forbidden["intervention"]["operations"] = [
            {"kind": "inject", "topic": "forbidden", "at": 0, "payload": None, "parents": []}
        ]
        receipt = verifier.verify_witness(self.world_path, self.write_witness(forbidden, "forbidden.json"))
        self.assertEqual(receipt["diagnostics"][0]["code"], "unexpected_intervention")

        wrong_bundle = copy.deepcopy(self.witness)
        wrong_bundle["bundle"] = "sha256:" + "0" * 64
        with self.assertRaises(verifier.VerificationError) as captured:
            verifier.verify_witness(self.world_path, self.write_witness(wrong_bundle, "bundle.json"))
        self.assertEqual(captured.exception.code, "bundle_mismatch")

    def test_meta_changes_identity_but_not_score(self) -> None:
        first_path = self.write_witness(self.witness, "first.json")
        first = verifier.verify_witness(self.world_path, first_path)
        second_witness = copy.deepcopy(self.witness)
        second_witness["meta"] = {"producer": "test", "comment": "same answer"}
        second = verifier.verify_witness(self.world_path, self.write_witness(second_witness, "second.json"))
        self.assertTrue(second["valid"])
        self.assertNotEqual(first["witness"], second["witness"])
        self.assertEqual(first["score"], second["score"])
        self.assertEqual(first["metrics"], second["metrics"])

    def test_noncanonical_and_unknown_outer_fields_are_rejected(self) -> None:
        noncanonical = self.root / "noncanonical.json"
        noncanonical.write_text(json.dumps(self.witness, indent=2), encoding="utf-8")
        with self.assertRaises(verifier.VerificationError) as captured:
            verifier.verify_witness(self.world_path, noncanonical)
        self.assertEqual(captured.exception.code, "noncanonical_json")

        unknown = copy.deepcopy(self.witness)
        unknown["surprise"] = True
        with self.assertRaises(verifier.VerificationError) as captured:
            verifier.verify_witness(self.world_path, self.write_witness(unknown, "unknown.json"))
        self.assertEqual(captured.exception.code, "invalid_witness")

    def test_non_root_history_is_replayed_and_bound_without_bundle_self_reference(self) -> None:
        history_root = self.root / "history-fixture"
        source = make_source(history_root)
        augment_source(source)
        root_case = case_descriptor()
        child_case = copy.deepcopy(root_case)
        child_case.update({
            "id": "ORIENT.002",
            "title": "Inherited Branch",
            "requires": {"all": ["case:ORIENT.001"]},
            "input_branch": "history:ORIENT.001",
            "answer_schema": "cases/ORIENT.002/answer.schema.json",
            "validator": "cases/ORIENT.002/validator.cre.json",
            "intervention_policy": "cases/ORIENT.002/interventions.json",
            "score": "cases/ORIENT.002/score.json",
        })
        grandchild_case = copy.deepcopy(child_case)
        grandchild_case.update({
            "id": "ORIENT.003",
            "title": "Inherited Branch Again",
            "requires": {"all": ["case:ORIENT.001", "case:ORIENT.002"]},
            "input_branch": "history:ORIENT.002",
            "answer_schema": "cases/ORIENT.003/answer.schema.json",
            "validator": "cases/ORIENT.003/validator.cre.json",
            "intervention_policy": "cases/ORIENT.003/interventions.json",
            "score": "cases/ORIENT.003/score.json",
        })
        write_json(source / "cases/index.json", {"format": "afterimage-cases/0.1", "cases": [root_case, child_case, grandchild_case]})
        write_json(source / "cases/ORIENT.001/interventions.json", {
            "format": "afterimage-intervention-policy/0.1",
            "required": False,
            "allowed_kinds": ["retime"],
            "max_operations": 1,
            "weights": {"retime": 1},
            "topics": ["sample.input"],
            "pointers": [],
            "retime": {"minimum": 0, "maximum": 10},
        })
        for name in ("answer.schema.json", "validator.cre.json", "score.json"):
            (source / f"cases/ORIENT.002/{name}").parent.mkdir(parents=True, exist_ok=True)
            (source / f"cases/ORIENT.002/{name}").write_bytes((source / f"cases/ORIENT.001/{name}").read_bytes())
        write_json(source / "cases/ORIENT.002/interventions.json", {
            "format": "afterimage-intervention-policy/0.1",
            "required": False,
            "allowed_kinds": ["retime"],
            "max_operations": 1,
            "weights": {"retime": 1},
            "topics": ["sample.input"],
            "pointers": [],
            "retime": {"minimum": 0, "maximum": 10},
        })
        for name in ("answer.schema.json", "validator.cre.json", "interventions.json", "score.json"):
            (source / f"cases/ORIENT.003/{name}").parent.mkdir(parents=True, exist_ok=True)
            (source / f"cases/ORIENT.003/{name}").write_bytes((source / f"cases/ORIENT.002/{name}").read_bytes())
        bundle_path = history_root / "history.afterimage"
        bundle = kit.pack_bundle(source, bundle_path, "History World", "history-1")
        world_path = history_root / "world"
        kit.extract_bundle(bundle, world_path)
        world = kit.verify_world(world_path)
        child = verifier.validate_case_descriptor(world.json_values["cases/index.json"]["cases"][1], world)
        base = cre.make_event({
            "topic": "sample.input",
            "at": 4,
            "payload": {"value": 7},
            "parents": [],
            "origin": {"kind": "base", "source": "sample", "sequence": 0},
        })
        history = {
            "format": "afterimage-branch-history/0.1",
            "bundle": world.bundle,
            "world": "global",
            "steps": [{"case": "ORIENT.001", "operations": [{"kind": "retime", "event": base["id"], "at": 9}]}],
        }
        resolved = verifier.resolve_input_history(world, child, history, {"case:ORIENT.001"})
        self.assertNotEqual(resolved.branch, cre.root_branch_id(world.bundle))
        self.assertEqual(next(event for event in resolved.base_events if event["topic"] == "sample.input")["at"], 9)
        replay = verifier.replay_case(world, child, [], resolved.branch, base_events=resolved.base_events)
        output = next(event for event in replay.events.values() if event["topic"] == "sample.output")
        witness = {
            "format": "afterimage-witness/0.1",
            "semantics": "cre/0.1",
            "bundle": world.bundle,
            "case": "ORIENT.002",
            "parent_branch": resolved.branch,
            "history": history,
            "intervention": None,
            "answer": {"event_id": output["id"], "topic": output["topic"], "at": output["at"], "projection": replay.projection},
            "claimed": {"branch": replay.branch, "projection": replay.projection, "trace": replay.trace},
        }
        path = history_root / "history-witness.json"
        path.write_bytes(cre.canonical_bytes(witness))
        receipt = verifier.verify_witness(world_path, path, {"case:ORIENT.001"})
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["branch"], resolved.branch)

        grandchild = verifier.validate_case_descriptor(world.json_values["cases/index.json"]["cases"][2], world)
        chained_history = copy.deepcopy(history)
        inherited_input = next(event for event in resolved.base_events if event["topic"] == "sample.input")
        chained_history["steps"].append({
            "case": "ORIENT.002",
            "operations": [{"kind": "retime", "event": inherited_input["id"], "at": 8}],
        })
        chained = verifier.resolve_input_history(
            world,
            grandchild,
            chained_history,
            {"case:ORIENT.001", "case:ORIENT.002"},
        )
        self.assertNotEqual(chained.branch, resolved.branch)
        with self.assertRaises(verifier.VerificationError) as captured:
            verifier.resolve_input_history(world, grandchild, chained_history, {"case:ORIENT.001"})
        self.assertEqual(captured.exception.code, "history_case_locked")
        broken_chain = copy.deepcopy(chained_history)
        del broken_chain["steps"][0]
        with self.assertRaises(verifier.VerificationError) as captured:
            verifier.resolve_input_history(
                world,
                grandchild,
                broken_chain,
                {"case:ORIENT.001", "case:ORIENT.002"},
            )
        self.assertEqual(captured.exception.code, "history_chain_mismatch")

        root_replay = verifier.replay_root_case(world, verifier.validate_case_descriptor(world.json_values["cases/index.json"]["cases"][0], world))
        derived_id = next(event["id"] for event in root_replay.events.values() if event["topic"] == "sample.output")
        derived_target = copy.deepcopy(history)
        derived_target["steps"][0]["operations"][0]["event"] = derived_id
        with self.assertRaises(verifier.VerificationError) as captured:
            verifier.resolve_input_history(world, child, derived_target, {"case:ORIENT.001"})
        self.assertEqual(captured.exception.code, "derived_event_not_intervenable")

        missing = copy.deepcopy(witness)
        del missing["history"]
        path.write_bytes(cre.canonical_bytes(missing))
        self.assertEqual(verifier.verify_witness(world_path, path, {"case:ORIENT.001"})["diagnostics"][0]["code"], "history_required")

        locked = copy.deepcopy(witness)
        path.write_bytes(cre.canonical_bytes(locked))
        self.assertEqual(verifier.verify_witness(world_path, path)["diagnostics"][0]["code"], "case_locked")

        tampered = copy.deepcopy(witness)
        tampered["history"]["steps"][0]["operations"][0]["at"] = 8
        path.write_bytes(cre.canonical_bytes(tampered))
        self.assertEqual(verifier.verify_witness(world_path, path, {"case:ORIENT.001"})["diagnostics"][0]["code"], "parent_branch_mismatch")

        unexpected = copy.deepcopy(self.witness)
        unexpected["history"] = history
        receipt = verifier.verify_witness(self.world_path, self.write_witness(unexpected, "root-history.json"))
        self.assertEqual(receipt["diagnostics"][0]["code"], "unexpected_history")

        invalid_case = copy.deepcopy(root_case)
        invalid_case["input_branch"] = "sha256:" + "1" * 64
        with self.assertRaises(verifier.VerificationError) as captured:
            verifier.validate_case_descriptor(invalid_case, world)
        self.assertEqual(captured.exception.code, "invalid_case")

    def test_requirements_and_schema_primitives(self) -> None:
        self.assertTrue(verifier.requirements_hold({"all": ["a"], "any": ["b", "c"]}, {"a", "c"}))
        self.assertFalse(verifier.requirements_hold({"at_least": {"count": 2, "of": ["a", "b", "c"]}}, {"a"}))
        verifier.validate_schema_node(
            {
                "type": "map",
                "required": ["values"],
                "properties": {"values": {"type": "list", "items": {"type": "int", "minimum": 0}, "max_items": 2}},
                "additional": False,
            },
            {"values": [0, 2]},
        )
        with self.assertRaises(verifier.VerificationError) as captured:
            verifier.validate_schema_node({"type": "id"}, "not-an-id")
        self.assertEqual(captured.exception.code, "answer_schema")


if __name__ == "__main__":
    unittest.main(verbosity=2)
