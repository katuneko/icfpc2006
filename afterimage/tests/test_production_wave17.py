#!/usr/bin/env python3
"""Integration and adversarial tests for production wave 17."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference/python"))

import afterimage_kit as kit  # noqa: E402
import covenant  # noqa: E402
import cre  # noqa: E402
import mosaic  # noqa: E402
import pulse  # noqa: E402
import verify_witness as verifier  # noqa: E402


class ProductionWave17Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="afterimage-wave17-")
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "production.afterimage"
        cls.author = cls.root / "author"
        subprocess.run([
            sys.executable, str(ROOT / "tools/build_slice.py"), str(cls.bundle),
            "--manifest", str(ROOT / "manifests/production_release.json"),
            "--author-dir", str(cls.author), "--title", "Afterimage production release 1.8",
            "--revision", "production-dev-1.8.0",
        ], check=True, capture_output=True, text=True)
        cls.world_path = cls.root / "world"
        kit.extract_bundle(kit.load_bundle(cls.bundle), cls.world_path)
        cls.world = kit.verify_world(cls.world_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def case(self, case_id: str) -> dict[str, object]:
        descriptor = next(item for item in self.world.json_values["cases/index.json"]["cases"] if item["id"] == case_id)
        return verifier.validate_case_descriptor(descriptor, self.world)

    def witness(self, case_id: str) -> dict[str, object]:
        return json.loads((self.author / f"{case_id}.witness.json").read_text(encoding="utf-8"))

    def verify_mutation(self, case_id: str, name: str, answer: dict[str, object], facts: set[str]) -> dict[str, object]:
        witness = self.witness(case_id)
        witness["answer"] = answer
        witness.pop("meta", None)
        path = self.root / f"{case_id}-{name}.json"
        path.write_bytes(cre.canonical_bytes(witness))
        return verifier.verify_witness(self.world_path, path, facts)

    def test_all_five_author_baselines_are_valid(self) -> None:
        expected = {"CASCADE.017": 109, "MERGE.012": 129, "MOSAIC.010": 149, "PULSE.011": 129, "COVENANT.002": 523}
        for case_id, score in expected.items():
            receipt = json.loads((self.author / f"{case_id}.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["valid"], case_id)
            self.assertEqual(receipt["score"]["total"], score, case_id)

    def test_policy_blind_spot_changes_active_safety_not_public_bytes(self) -> None:
        case = self.case("CASCADE.017")
        root = verifier.replay_root_case(self.world, case)
        candidate = self.witness("CASCADE.017")
        operation = candidate["intervention"]["operations"][0]
        repaired = verifier.replay_case(self.world, case, [operation], cre.root_branch_id(self.world.bundle))
        root_contract = next(event for event in root.events.values() if event["topic"] == "contract.summary")
        fixed_contract = next(event for event in repaired.events.values() if event["topic"] == "contract.summary")
        self.assertFalse(root_contract["payload"]["safe"])
        self.assertTrue(fixed_contract["payload"]["safe"])
        self.assertEqual(root.records, repaired.records)

    def test_causal_compression_requires_a_minimal_sufficient_skeleton(self) -> None:
        case = self.case("MERGE.012")
        logical = kit.resolve_logical_world(self.world, case["world"])
        by_key = {event["payload"]["key"]: event["id"] for event in logical.base_events}
        answer = copy.deepcopy(self.witness("MERGE.012")["answer"])
        answer["certificate"].append({"before": by_key["A"], "after": by_key["C"], "minimum_gap": 1})
        receipt = self.verify_mutation("MERGE.012", "redundant-edge", answer, {"case:MERGE.011", "case:CASCADE.017"})
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "merge_compression")
        missing = copy.deepcopy(self.witness("MERGE.012")["answer"])
        missing["certificate"] = missing["certificate"][1:]
        receipt = self.verify_mutation("MERGE.012", "missing-edge", missing, {"case:MERGE.011", "case:CASCADE.017"})
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "merge_compression")

    def test_underground_layer_requires_exact_portals_and_connected_layers(self) -> None:
        source = json.loads((ROOT / "content/production/cases/MOSAIC.010.json").read_text(encoding="utf-8"))
        fragments = [event["payload"] for event in source["events"]]
        self.assertEqual(mosaic.validate_answer(fragments, source["case"]["author_answer"], source["case"]["mosaic_contract"])["graph_size"], 21)
        missing_portal = copy.deepcopy(source["case"]["mosaic_contract"])
        missing_portal["layers"]["portals"].remove("v3")
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(fragments, source["case"]["author_answer"], missing_portal)
        self.assertEqual(raised.exception.code, "mosaic_layer")

    def test_burst_budget_exhausts_boundary_and_rejects_over_admission(self) -> None:
        source = json.loads((ROOT / "content/production/cases/PULSE.011.json").read_text(encoding="utf-8"))
        program, contract = source["case"]["author_program"], source["case"]["pulse_contract"]
        self.assertEqual(pulse.verify_program(program, contract), {"program_bytes": 478, "worst_latency": 0, "live_state_cells": 2, "domain_cases": 792})
        expected = pulse.expected_outputs(contract, (0, 0, 0, 3))
        self.assertEqual([item["payload"]["sequence"] for item in expected], [0, 2, 3])
        over_admits = copy.deepcopy(program)
        over_admits["handlers"][0]["actions"][0]["condition"] = ["const", True]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(over_admits, contract)
        self.assertEqual(raised.exception.code, "pulse_counterexample")

    def test_city_covenant_requires_local_information_flow(self) -> None:
        source = json.loads((ROOT / "content/production/cases/COVENANT.002.json").read_text(encoding="utf-8"))
        contract, policy = source["case"]["covenant_contract"], source["case"]["author_policy"]
        self.assertEqual(covenant.verify_policy(contract, policy), {"policy_nodes": 42, "worst_response_bound": 4, "reachable_states": 38})
        omniscient = copy.deepcopy(policy)
        omniscient["agents"][0]["rules"][0]["when"] = ["not", ["get", "heat"]]
        with self.assertRaises(covenant.CovenantError) as raised:
            covenant.verify_policy(contract, omniscient)
        self.assertEqual(raised.exception.code, "covenant_locality")


if __name__ == "__main__":
    unittest.main(verbosity=2)
