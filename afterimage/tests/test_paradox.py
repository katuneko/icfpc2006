from __future__ import annotations

import copy
import tempfile
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import authoring  # noqa: E402
import cre  # noqa: E402
import paradox  # noqa: E402
import verify_witness as verifier  # noqa: E402
import afterimage_kit as kit  # noqa: E402
from afterimage.tests.test_afterimage_kit import make_source, write_bytes, write_json  # noqa: E402


def event(topic: str, payload: object, sequence: int) -> dict:
    return cre.make_event({
        "topic": topic,
        "at": sequence,
        "payload": payload,
        "parents": [],
        "origin": {"kind": "base", "source": "paradox-test", "sequence": sequence},
    })


class ParadoxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.safety = event("contract.safe", {"safe": True}, 0)
        self.left_latent = event("latent.choice", {"route": "north"}, 1)
        self.right_latent = event("latent.choice", {"route": "south"}, 2)
        self.left_events = {item["id"]: item for item in (self.safety, self.left_latent)}
        self.right_events = {item["id"]: item for item in (self.safety, self.right_latent)}
        self.records = [{"public": "same"}]
        self.contract = {
            "format": "afterimage-paradox-contract/0.1",
            "safety_requirements": [
                {"id": "published-safe", "topic": "contract.safe", "pointer": "/payload/safe", "equals": True, "minimum": 1},
            ],
            "latent_topics": {"latent.choice": 2},
            "max_public_records": 8,
        }
        digest = cre.digest_id("afterimage/paradox-public-record/1", cre.canonical_bytes(self.records[0]))
        self.answer = {
            "left_history": {"format": "afterimage-branch-history/0.1", "bundle": "sha256:" + "0" * 64, "world": "global", "steps": [{"case": "CASCADE.010", "operations": [{"kind": "inject"}]}]},
            "right_history": {"format": "afterimage-branch-history/0.1", "bundle": "sha256:" + "0" * 64, "world": "global", "steps": [{"case": "CASCADE.011", "operations": [{"kind": "inject"}]}]},
            "equivalence": [{"left": 0, "right": 0, "digest": digest}],
            "safety_evidence": {"left": [self.safety["id"]], "right": [self.safety["id"]]},
            "latent_difference": sorted([self.left_latent["id"], self.right_latent["id"]], key=cre.parse_id),
        }

    def validate(self, answer=None, **changes):
        arguments = {
            "contract_value": self.contract,
            "answer_value": answer or self.answer,
            "left_branch": "sha256:" + "1" * 64,
            "right_branch": "sha256:" + "2" * 64,
            "left_records": self.records,
            "right_records": self.records,
            "left_events": self.left_events,
            "right_events": self.right_events,
        }
        arguments.update(changes)
        return paradox.validate_certificate(**arguments)

    def test_complete_certificate_proves_equivalence_safety_and_difference(self) -> None:
        metrics = self.validate()
        self.assertEqual(metrics["latent_difference_weight"], 4)
        self.assertEqual(metrics["proof_steps"], 5)
        self.assertGreater(metrics["paired_witness_units"], 0)
        self.assertEqual(metrics, self.validate())

    def test_projection_safety_and_material_difference_fail_independently(self) -> None:
        with self.assertRaises(paradox.ParadoxError) as captured:
            self.validate(right_records=[{"public": "different"}])
        self.assertEqual(captured.exception.code, "paradox_projection")

        unsafe = event("contract.safe", {"safe": False}, 3)
        right_events = {unsafe["id"]: unsafe, self.right_latent["id"]: self.right_latent}
        with self.assertRaises(paradox.ParadoxError) as captured:
            self.validate(right_events=right_events)
        self.assertEqual(captured.exception.code, "paradox_safety")

        with self.assertRaises(paradox.ParadoxError) as captured:
            self.validate(right_events=self.left_events)
        self.assertEqual(captured.exception.code, "paradox_no_difference")

    def test_certificate_coverage_digest_and_evidence_are_exact(self) -> None:
        wrong_digest = copy.deepcopy(self.answer)
        wrong_digest["equivalence"][0]["digest"] = "sha256:" + "f" * 64
        with self.assertRaises(paradox.ParadoxError) as captured:
            self.validate(wrong_digest)
        self.assertEqual(captured.exception.code, "paradox_equivalence")

        missing_evidence = copy.deepcopy(self.answer)
        missing_evidence["safety_evidence"]["left"] = []
        with self.assertRaises(paradox.ParadoxError) as captured:
            self.validate(missing_evidence)
        self.assertEqual(captured.exception.code, "paradox_safety")

        missing_latent = copy.deepcopy(self.answer)
        missing_latent["latent_difference"] = missing_latent["latent_difference"][:1]
        with self.assertRaises(paradox.ParadoxError) as captured:
            self.validate(missing_latent)
        self.assertEqual(captured.exception.code, "paradox_difference")

        with self.assertRaises(paradox.ParadoxError) as captured:
            self.validate(left_branch="sha256:" + "1" * 64, right_branch="sha256:" + "1" * 64)
        self.assertEqual(captured.exception.code, "paradox_same_branch")

    def test_semantic_latent_pointer_rejects_provenance_only_churn(self) -> None:
        contract = {**self.contract, "latent_pointers": {"latent.choice": "/payload/route"}}
        same_value = event("latent.choice", {"route": "north"}, 9)
        right_events = {self.safety["id"]: self.safety, same_value["id"]: same_value}
        answer = copy.deepcopy(self.answer)
        answer["latent_difference"] = sorted([self.left_latent["id"], same_value["id"]], key=cre.parse_id)
        with self.assertRaises(paradox.ParadoxError) as captured:
            paradox.validate_certificate(
                contract, answer, left_branch="sha256:" + "1" * 64, right_branch="sha256:" + "2" * 64,
                left_records=self.records, right_records=self.records,
                left_events=self.left_events, right_events=right_events,
            )
        self.assertEqual(captured.exception.code, "paradox_no_difference")

    def test_host_score_uses_all_three_lexicographic_metrics(self) -> None:
        raw = self.validate()
        score_document = authoring.paradox_score(100_000, 10, 20, self.contract)
        metrics, score = verifier.paradox_score({"family": "PARADOX", "points": 300}, score_document, raw)
        self.assertEqual(score["completion"], 195)
        self.assertLessEqual(score["total"], 300)
        self.assertGreater(metrics["effective_cost"], 0)
        limited = copy.deepcopy(score_document)
        limited["metric_bounds"]["proof_steps"] = 4
        with self.assertRaises(verifier.VerificationError) as captured:
            verifier.paradox_score({"family": "PARADOX", "points": 300}, limited, raw)
        self.assertEqual(captured.exception.code, "metric_limit")

    def test_authoritative_verifier_replays_both_histories_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory(prefix="afterimage-paradox-host-") as temporary:
            root = Path(temporary)
            source = make_source(root)
            safety = event("contract.safe", {"safe": True}, 0)
            latent = event("latent.choice", {"route": "central"}, 1)
            write_json(source / "program/continuity.cre.json", {"semantics": "cre/0.1", "strata": []})
            write_bytes(
                source / "events/base.ndjson",
                b"".join(cre.canonical_bytes(cre.event_view(item)) + b"\n" for item in sorted((safety, latent), key=lambda item: cre.parse_id(item["id"]))),
            )
            projection = {
                "id": "sample.public",
                "rows": [{
                    "positive": [{"alias": "s", "topic": "contract.safe", "where": ["const", True]}],
                    "negative": [], "aggregate": [], "distinct": [], "guard": ["const", True],
                    "value": ["map", "safe", ["get", "s", "/payload/safe"]], "sort": [],
                }],
            }
            write_json(source / "projections/sample.public.cre.json", projection)
            limits = {
                "max_witness_bytes": 262144,
                "replay": {"max_base_events": 32, "max_derived_events": 32, "max_bindings_tested": 1024, "max_value_bytes": 1048576, "max_projection_records": 16},
                "validator": {"max_base_events": 128, "max_derived_events": 16, "max_bindings_tested": 1024, "max_value_bytes": 1048576, "max_projection_records": 8},
            }
            cascade_case = {
                "id": "CASCADE.010", "family": "CASCADE", "title": "Choose Route", "points": 100,
                "requires": {"all": []}, "input_branch": "root", "world": "global", "projection": "sample.public",
                "answer_schema": "cases/CASCADE.010/answer.schema.json", "validator": "cases/CASCADE.010/validator.cre.json",
                "intervention_policy": "cases/CASCADE.010/interventions.json", "score": "cases/CASCADE.010/score.json", "limits": limits,
            }
            paradox_case = {
                "id": "PARADOX.001", "family": "PARADOX", "title": "Two Safe Tomorrows", "points": 300,
                "requires": {"all": ["case:CASCADE.010"]}, "input_branch": "root", "world": "global", "projection": "sample.public",
                "answer_schema": "cases/PARADOX.001/answer.schema.json", "validator": "cases/PARADOX.001/validator.cre.json",
                "intervention_policy": "cases/PARADOX.001/interventions.json", "score": "cases/PARADOX.001/score.json", "limits": limits,
            }
            write_json(source / "cases/index.json", {"format": "afterimage-cases/0.1", "cases": [cascade_case, paradox_case]})
            open_schema = {"format": "afterimage-answer-schema/0.1", "schema": {"type": "map", "required": [], "properties": {}, "additional": True}}
            write_json(source / "cases/CASCADE.010/answer.schema.json", open_schema)
            write_json(source / "cases/PARADOX.001/answer.schema.json", authoring.paradox_answer_schema())
            validator_document = {
                "format": "afterimage-validator/0.1",
                "program": {"semantics": "cre/0.1", "strata": [{"index": 0, "rules": [{
                    "id": "verify.family-valid",
                    "positive": [{"alias": "f", "topic": "verify.family", "where": ["get", "f", "/payload/valid"]}],
                    "negative": [], "aggregate": [], "distinct": [], "guard": ["const", True],
                    "emit": [{"topic": ["const", "verify.decision"], "at": ["const", 0], "payload": ["map", "valid", ["const", True], "diagnostics", ["list"], "metrics", ["map"]], "parents": []}],
                }]}]},
                "decision_projection": {"id": "verify.decision", "rows": [{
                    "positive": [{"alias": "d", "topic": "verify.decision", "where": ["const", True]}],
                    "negative": [], "aggregate": [], "distinct": [], "guard": ["const", True],
                    "value": ["get", "d", "/payload"], "sort": [],
                }]},
            }
            write_json(source / "cases/CASCADE.010/validator.cre.json", validator_document)
            write_json(source / "cases/PARADOX.001/validator.cre.json", validator_document)
            write_json(source / "cases/CASCADE.010/interventions.json", {
                "format": "afterimage-intervention-policy/0.1", "required": True,
                "allowed_kinds": ["replace"], "max_operations": 1, "weights": {"replace": 1},
                "topics": ["latent.choice"], "pointers": ["/payload/route"], "retime": {"minimum": 0, "maximum": 0},
            })
            write_json(source / "cases/PARADOX.001/interventions.json", {
                "format": "afterimage-intervention-policy/0.1", "required": False,
                "allowed_kinds": [], "max_operations": 0, "weights": {}, "topics": [], "pointers": [], "retime": {"minimum": 0, "maximum": 0},
            })
            write_json(source / "cases/CASCADE.010/score.json", authoring.cascade_score(1000))
            contract_value = {
                "format": "afterimage-paradox-contract/0.1",
                "safety_requirements": [{"id": "safe", "topic": "contract.safe", "pointer": "/payload/safe", "equals": True, "minimum": 1}],
                "latent_topics": {"latent.choice": 2}, "max_public_records": 8,
            }
            write_json(source / "cases/PARADOX.001/score.json", authoring.paradox_score(100000, 10, 20, contract_value))
            bundle_path = root / "paradox.afterimage"
            bundle = kit.pack_bundle(source, bundle_path, "Paradox Host", "paradox-1")
            world_path = root / "world"
            kit.extract_bundle(bundle, world_path)
            world = kit.verify_world(world_path)
            case = verifier.validate_case_descriptor(world.json_values["cases/index.json"]["cases"][1], world)

            def history(route: str) -> dict:
                return {
                    "format": "afterimage-branch-history/0.1", "bundle": world.bundle, "world": "global",
                    "steps": [{"case": "CASCADE.010", "operations": [{"kind": "replace", "event": latent["id"], "pointer": "/payload/route", "value": route}]}],
                }

            histories = {"left": history("north"), "right": history("south")}
            states = {}
            for side in ("left", "right"):
                target = {**case, "input_branch": "history:CASCADE.010"}
                resolved = verifier.resolve_input_history(world, target, histories[side], {"case:CASCADE.010"})
                states[side] = verifier.evaluate_case_state(world, case, resolved.base_events, resolved.branch)
            record_digest = cre.digest_id("afterimage/paradox-public-record/1", cre.canonical_bytes(states["left"].records[0]))
            differing = sorted(
                {event_id for event_id in set(states["left"].events) ^ set(states["right"].events) if (states["left"].events.get(event_id) or states["right"].events[event_id])["topic"] == "latent.choice"},
                key=cre.parse_id,
            )
            answer = {
                "left_history": histories["left"], "right_history": histories["right"],
                "equivalence": [{"left": 0, "right": 0, "digest": record_digest}],
                "safety_evidence": {"left": [safety["id"]], "right": [safety["id"]]},
                "latent_difference": differing,
            }
            root_branch = cre.root_branch_id(world.bundle)
            witness = {
                "format": "afterimage-witness/0.1", "semantics": "cre/0.1", "bundle": world.bundle,
                "case": "PARADOX.001", "parent_branch": root_branch, "intervention": None, "answer": answer,
            }
            witness_path = root / "paradox.witness.json"
            witness_path.write_bytes(cre.canonical_bytes(witness))
            receipt = verifier.verify_witness(world_path, witness_path, {"case:CASCADE.010"})
            self.assertTrue(receipt["valid"])
            self.assertEqual(receipt["metrics"]["latent_difference_weight"], 4)

            tampered = copy.deepcopy(witness)
            tampered["answer"]["safety_evidence"]["right"] = []
            witness_path.write_bytes(cre.canonical_bytes(tampered))
            invalid = verifier.verify_witness(world_path, witness_path, {"case:CASCADE.010"})
            self.assertFalse(invalid["valid"])
            self.assertEqual(invalid["diagnostics"][0]["code"], "paradox_safety")


if __name__ == "__main__":
    unittest.main()
