from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import covenant  # noqa: E402
import authoring  # noqa: E402
import verify_witness as verifier  # noqa: E402
import afterimage_kit as kit  # noqa: E402
import cre  # noqa: E402
from afterimage.tests.test_afterimage_kit import make_source, write_json  # noqa: E402


TRUE = ["const", True]
DISPATCH_SOURCE = json.loads((ROOT / "content/production/cases/COVENANT.001.json").read_text(encoding="utf-8"))
DISPATCH_CONTRACT = DISPATCH_SOURCE["case"]["covenant_contract"]
DISPATCH_POLICY = DISPATCH_SOURCE["case"]["author_policy"]


def contract() -> dict:
    return {
        "format": "afterimage-covenant-contract/0.1",
        "domains": {
            "alarm": [False, True],
            "assigned": [False, True],
            "sent": [False, True],
        },
        "initial": [{"alarm": False, "assigned": False, "sent": False}],
        "actors": [
            {"id": "dispatch", "kind": "agent", "observes": ["alarm", "assigned"], "actions": ["dispatch.assign", "dispatch.wait"]},
            {"id": "environment", "kind": "environment", "observes": [], "actions": ["env.idle", "env.raise"]},
            {"id": "response", "kind": "agent", "observes": ["assigned", "sent"], "actions": ["response.send", "response.wait"]},
        ],
        "actions": [
            {
                "id": "dispatch.assign",
                "actor": "dispatch",
                "guard": ["and", ["get", "alarm"], ["not", ["get", "assigned"]]],
                "updates": {"assigned": ["const", True]},
            },
            {"id": "dispatch.wait", "actor": "dispatch", "guard": TRUE, "updates": {}},
            {"id": "env.idle", "actor": "environment", "guard": TRUE, "updates": {}},
            {
                "id": "env.raise",
                "actor": "environment",
                "guard": ["not", ["get", "alarm"]],
                "updates": {"alarm": ["const", True]},
            },
            {
                "id": "response.send",
                "actor": "response",
                "guard": ["and", ["get", "assigned"], ["not", ["get", "sent"]]],
                "updates": {"sent": ["const", True]},
            },
            {"id": "response.wait", "actor": "response", "guard": TRUE, "updates": {}},
        ],
        "scheduler": {"fairness_window": 2},
        "safety": [
            {"id": "sent-only-after-assignment", "predicate": ["or", ["not", ["get", "sent"]], ["get", "assigned"]]},
        ],
        "liveness": [
            {"id": "alarm-eventually-sent", "trigger": ["get", "alarm"], "goal": ["get", "sent"], "bound": 6},
        ],
        "limits": {"max_reachable_states": 1000, "max_transitions": 10000, "max_expression_nodes": 1000},
    }


def policy() -> dict:
    return {
        "format": "afterimage-covenant-policy/0.1",
        "agents": [
            {
                "agent": "dispatch",
                "rules": [{"when": ["and", ["get", "alarm"], ["not", ["get", "assigned"]]], "action": "dispatch.assign"}],
                "default": "dispatch.wait",
            },
            {
                "agent": "response",
                "rules": [{"when": ["and", ["get", "assigned"], ["not", ["get", "sent"]]], "action": "response.send"}],
                "default": "response.wait",
            },
        ],
    }


class CovenantTests(unittest.TestCase):
    def test_dispatch_covenant_exhausts_both_initial_availability_states(self) -> None:
        self.assertEqual(covenant.verify_policy(DISPATCH_CONTRACT, DISPATCH_POLICY), {
            "policy_nodes": 32, "worst_response_bound": 5, "reachable_states": 80,
        })

    def test_dispatch_covenant_rejects_omniscience_and_waiting(self) -> None:
        omniscient = copy.deepcopy(DISPATCH_POLICY)
        omniscient["agents"][0]["rules"][0]["when"] = ["get", "sent"]
        with self.assertRaises(covenant.CovenantError) as captured:
            covenant.verify_policy(DISPATCH_CONTRACT, omniscient)
        self.assertEqual(captured.exception.code, "covenant_locality")

        waiting = copy.deepcopy(DISPATCH_POLICY)
        waiting["agents"][0]["rules"] = []
        with self.assertRaises(covenant.CovenantError) as captured:
            covenant.verify_policy(DISPATCH_CONTRACT, waiting)
        self.assertEqual(captured.exception.code, "covenant_liveness")
        self.assertEqual(captured.exception.context["obligation"], "alarm-eventually-sent")

    def test_exhaustive_local_policy_satisfies_all_fair_schedules(self) -> None:
        first = covenant.verify_policy(contract(), policy())
        second = covenant.verify_policy(contract(), policy())
        self.assertEqual(first, second)
        self.assertGreater(first["policy_nodes"], 0)
        self.assertGreater(first["reachable_states"], 8)
        self.assertLessEqual(first["worst_response_bound"], 6)

    def test_waiting_policy_returns_stable_shortest_liveness_counterexample(self) -> None:
        broken = policy()
        broken["agents"][0]["rules"] = []
        with self.assertRaises(covenant.CovenantError) as captured:
            covenant.verify_policy(contract(), broken)
        self.assertEqual(captured.exception.code, "covenant_liveness")
        trace = captured.exception.context["trace"]
        self.assertTrue(trace)
        self.assertEqual(trace, covenant.CovenantError(
            captured.exception.code,
            captured.exception.message,
            captured.exception.context,
        ).context["trace"])
        with self.assertRaises(covenant.CovenantError) as repeated:
            covenant.verify_policy(contract(), broken)
        self.assertEqual(repeated.exception.context["trace"], trace)

    def test_hidden_observation_disabled_action_and_safety_failure_are_distinct(self) -> None:
        hidden = policy()
        hidden["agents"][0]["rules"][0]["when"] = ["get", "sent"]
        with self.assertRaises(covenant.CovenantError) as captured:
            covenant.verify_policy(contract(), hidden)
        self.assertEqual(captured.exception.code, "covenant_locality")

        disabled = policy()
        disabled["agents"][0]["rules"] = []
        disabled["agents"][0]["default"] = "dispatch.assign"
        with self.assertRaises(covenant.CovenantError) as captured:
            covenant.verify_policy(contract(), disabled)
        self.assertEqual(captured.exception.code, "covenant_policy")

        unsafe_contract = contract()
        send = next(item for item in unsafe_contract["actions"] if item["id"] == "response.send")
        send["guard"] = TRUE
        unsafe_policy = policy()
        unsafe_policy["agents"][1]["rules"] = [{"when": ["not", ["get", "assigned"]], "action": "response.send"}]
        with self.assertRaises(covenant.CovenantError) as captured:
            covenant.verify_policy(unsafe_contract, unsafe_policy)
        self.assertEqual(captured.exception.code, "covenant_safety")
        self.assertEqual(captured.exception.context["invariant"], "sent-only-after-assignment")

    def test_finite_domain_and_exploration_limits_are_enforced(self) -> None:
        outside = contract()
        action = next(item for item in outside["actions"] if item["id"] == "env.raise")
        action["updates"]["alarm"] = ["const", "outside"]
        with self.assertRaises(covenant.CovenantError) as captured:
            covenant.verify_policy(outside, policy())
        self.assertEqual(captured.exception.code, "covenant_transition")

        limited = contract()
        limited["limits"]["max_reachable_states"] = 1
        with self.assertRaises(covenant.CovenantError) as captured:
            covenant.verify_policy(limited, policy())
        self.assertEqual(captured.exception.code, "covenant_limit")

        excessive = contract()
        excessive["limits"]["max_transitions"] = covenant.HARD_MAX_TRANSITIONS + 1
        with self.assertRaises(covenant.CovenantError) as captured:
            covenant.verify_policy(excessive, policy())
        self.assertEqual(captured.exception.code, "covenant_limit")

    def test_host_claim_and_lexicographic_score_are_recomputed(self) -> None:
        score_document = authoring.covenant_score(100_000, 6, 1000, contract())
        answer = {"policy": policy(), "claimed_response_bound": 5}
        raw = verifier.validate_covenant_answer(score_document, answer)
        self.assertEqual(raw, {"policy_nodes": 17, "worst_response_bound": 5, "reachable_states": 32})
        metrics, score = verifier.covenant_score({"family": "COVENANT", "points": 500}, score_document, raw)
        self.assertEqual(score["completion"], 325)
        self.assertLessEqual(score["total"], 500)
        self.assertGreater(metrics["effective_cost"], 0)

        wrong = copy.deepcopy(answer)
        wrong["claimed_response_bound"] = 4
        with self.assertRaises(verifier.VerificationError) as captured:
            verifier.validate_covenant_answer(score_document, wrong)
        self.assertEqual(captured.exception.code, "covenant_claim")

        too_small = copy.deepcopy(score_document)
        too_small["metric_bounds"]["reachable_states"] = 31
        with self.assertRaises(verifier.VerificationError) as captured:
            verifier.covenant_score({"family": "COVENANT", "points": 500}, too_small, raw)
        self.assertEqual(captured.exception.code, "metric_limit")

    def test_authoritative_verifier_accepts_model_checked_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="afterimage-covenant-host-") as temporary:
            root = Path(temporary)
            source = make_source(root)
            limits = {
                "max_witness_bytes": 262144,
                "replay": {"max_base_events": 32, "max_derived_events": 32, "max_bindings_tested": 1024, "max_value_bytes": 1048576, "max_projection_records": 16},
                "validator": {"max_base_events": 128, "max_derived_events": 16, "max_bindings_tested": 1024, "max_value_bytes": 1048576, "max_projection_records": 8},
            }
            descriptor = {
                "id": "COVENANT.001", "family": "COVENANT", "title": "Dispatch Covenant", "points": 500,
                "requires": {"all": []}, "input_branch": "root", "world": "global", "projection": "sample.public",
                "answer_schema": "cases/COVENANT.001/answer.schema.json", "validator": "cases/COVENANT.001/validator.cre.json",
                "intervention_policy": "cases/COVENANT.001/interventions.json", "score": "cases/COVENANT.001/score.json", "limits": limits,
            }
            write_json(source / "cases/index.json", {"format": "afterimage-cases/0.1", "cases": [descriptor]})
            write_json(source / "cases/COVENANT.001/answer.schema.json", authoring.covenant_answer_schema())
            write_json(source / "cases/COVENANT.001/interventions.json", {
                "format": "afterimage-intervention-policy/0.1", "required": False,
                "allowed_kinds": [], "max_operations": 0, "weights": {}, "topics": [], "pointers": [], "retime": {"minimum": 0, "maximum": 0},
            })
            write_json(source / "cases/COVENANT.001/score.json", authoring.covenant_score(100000, 6, 1000, contract()))
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
            write_json(source / "cases/COVENANT.001/validator.cre.json", validator_document)
            bundle_path = root / "covenant.afterimage"
            bundle = kit.pack_bundle(source, bundle_path, "Covenant Host", "covenant-1")
            world_path = root / "world"
            kit.extract_bundle(bundle, world_path)
            world = kit.verify_world(world_path)
            parent = cre.root_branch_id(world.bundle)
            witness = {
                "format": "afterimage-witness/0.1", "semantics": "cre/0.1", "bundle": world.bundle,
                "case": "COVENANT.001", "parent_branch": parent, "intervention": None,
                "answer": {"policy": policy(), "claimed_response_bound": 5},
            }
            path = root / "covenant.witness.json"
            path.write_bytes(cre.canonical_bytes(witness))
            receipt = verifier.verify_witness(world_path, path)
            self.assertTrue(receipt["valid"])
            self.assertEqual(receipt["metrics"]["reachable_states"], 32)
            wrong = copy.deepcopy(witness)
            wrong["answer"]["claimed_response_bound"] = 4
            path.write_bytes(cre.canonical_bytes(wrong))
            invalid = verifier.verify_witness(world_path, path)
            self.assertFalse(invalid["valid"])
            self.assertEqual(invalid["diagnostics"][0]["code"], "covenant_claim")


    def test_city_covenant_relays_hidden_heat_locally(self) -> None:
        source = json.loads((ROOT / "content/production/cases/COVENANT.002.json").read_text(encoding="utf-8"))
        self.assertEqual(covenant.verify_policy(source["case"]["covenant_contract"], source["case"]["author_policy"]), {
            "policy_nodes": 42, "worst_response_bound": 4, "reachable_states": 38,
        })

    def test_city_covenant_rejects_omniscient_dispatch(self) -> None:
        source = json.loads((ROOT / "content/production/cases/COVENANT.002.json").read_text(encoding="utf-8"))
        policy = copy.deepcopy(source["case"]["author_policy"])
        policy["agents"][0]["rules"][0]["when"] = ["not", ["get", "heat"]]
        with self.assertRaises(covenant.CovenantError) as raised:
            covenant.verify_policy(source["case"]["covenant_contract"], policy)
        self.assertEqual(raised.exception.code, "covenant_locality")


if __name__ == "__main__":
    unittest.main()
