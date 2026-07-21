#!/usr/bin/env python3
"""Integration and adversarial tests for production wave 15."""

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
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import cre  # noqa: E402
import verify_witness as verifier  # noqa: E402


class ProductionWave15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="afterimage-wave15-")
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "production.afterimage"
        cls.author = cls.root / "author"
        subprocess.run(
            [
                sys.executable, str(ROOT / "tools" / "build_slice.py"), str(cls.bundle),
                "--manifest", str(ROOT / "manifests" / "production_release.json"),
                "--author-dir", str(cls.author),
                "--title", "Afterimage production release 1.6",
                "--revision", "production-dev-1.6.0",
            ],
            check=True, capture_output=True, text=True,
        )
        cls.world_path = cls.root / "world"
        kit.extract_bundle(kit.load_bundle(cls.bundle), cls.world_path)
        cls.world = kit.verify_world(cls.world_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def case(self, case_id: str) -> dict[str, object]:
        descriptor = next(item for item in self.world.json_values["cases/index.json"]["cases"] if item["id"] == case_id)
        return verifier.validate_case_descriptor(descriptor, self.world)

    def author_answer(self, case_id: str) -> dict[str, object]:
        return json.loads((self.author / f"{case_id}.witness.json").read_text(encoding="utf-8"))["answer"]

    def verify_mutation(self, case_id: str, name: str, answer: dict[str, object], facts: set[str]) -> dict[str, object]:
        witness = json.loads((self.author / f"{case_id}.witness.json").read_text(encoding="utf-8"))
        witness["answer"] = answer
        witness.pop("meta", None)
        path = self.root / f"{case_id}-{name}.json"
        path.write_bytes(cre.canonical_bytes(witness))
        return verifier.verify_witness(self.world_path, path, facts)

    def test_all_five_author_baselines_are_valid(self) -> None:
        expected_scores = {
            "CASCADE.014": 99, "MERGE.010": 119, "MOSAIC.008": 129,
            "PULSE.009": 117, "LENS.003": 616,
        }
        for case_id, score in expected_scores.items():
            receipt = json.loads((self.author / f"{case_id}.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["valid"], case_id)
            self.assertEqual(receipt["score"]["total"], score, case_id)

    def test_shadow_bus_has_one_robust_integer_allocation(self) -> None:
        case = self.case("CASCADE.014")
        logical = kit.resolve_logical_world(self.world, case["world"])
        bus = next(event for event in logical.base_events if event["topic"] == "evacuation.bus-commitment")
        train = next(event for event in logical.base_events if event["topic"] == "evacuation.train-commitment")
        parent = cre.root_branch_id(self.world.bundle)
        safe = []
        for bus_load in range(71):
            train_load = 70 - bus_load
            replay = verifier.replay_case(self.world, case, [
                {"kind": "replace", "event": bus["id"], "pointer": "/payload/commitment", "value": bus_load},
                {"kind": "replace", "event": train["id"], "pointer": "/payload/commitment", "value": train_load},
            ], parent)
            if replay.records[0]["safe"]:
                safe.append((bus_load, train_load))
        self.assertEqual(safe, [(18, 52)])

    def test_echo_chain_rejects_a_shortcut_certificate(self) -> None:
        case = self.case("MERGE.010")
        logical = kit.resolve_logical_world(self.world, case["world"])
        by_key = {event["payload"]["key"]: event["id"] for event in logical.base_events}
        altered = copy.deepcopy(self.author_answer("MERGE.010"))
        e3 = next(item for item in altered["rejected"] if item["event"] == by_key["E3"])
        e3["duplicate_of"] = by_key["U0"]
        receipt = self.verify_mutation("MERGE.010", "shortcut", altered, {"case:MERGE.009", "case:CASCADE.014"})
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "merge_echo")

    def test_broken_ring_rejects_a_gap_not_matching_coverage(self) -> None:
        altered = copy.deepcopy(self.author_answer("MOSAIC.008"))
        altered["missing"]["edges"] = [{"a": "v6", "b": "v7"}]
        receipt = self.verify_mutation("MOSAIC.008", "wrong-gap", altered, {"case:MOSAIC.007", "case:MERGE.010"})
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "mosaic_ring")

    def test_exactly_once_rejects_route_local_memory(self) -> None:
        altered = copy.deepcopy(self.author_answer("PULSE.009"))
        fail = next(handler for handler in altered["program"]["handlers"] if handler["id"] == "exactly.fail")
        fail["actions"].append({"op": "set", "cell": "seen_A", "value": ["const", False]})
        receipt = self.verify_mutation("PULSE.009", "route-local", altered, {"case:PULSE.008", "case:MERGE.010", "case:MOSAIC.008"})
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "pulse_counterexample")

    def test_two_histories_requires_identity_and_audit_complement(self) -> None:
        facts = {"case:LENS.002", "case:CASCADE.014", "case:MERGE.010", "case:MOSAIC.008", "case:PULSE.009"}
        no_identity = copy.deepcopy(self.author_answer("LENS.003"))
        no_identity["program"]["complement_schema"] = [cell for cell in no_identity["program"]["complement_schema"] if cell["name"] != "history_key"]
        receipt = self.verify_mutation("LENS.003", "no-identity", no_identity, facts)
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "lens_counterexample")

        no_audit = copy.deepcopy(self.author_answer("LENS.003"))
        no_audit["program"]["put"][2]["fields"].remove("audit_chain")
        receipt = self.verify_mutation("LENS.003", "no-audit", no_audit, facts)
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "lens_counterexample")


if __name__ == "__main__":
    unittest.main(verbosity=2)
