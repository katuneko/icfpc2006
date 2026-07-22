#!/usr/bin/env python3
"""Integration and adversarial tests for production wave 18."""

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
import cre  # noqa: E402
import mosaic  # noqa: E402
import pulse  # noqa: E402
import verify_witness as verifier  # noqa: E402


class ProductionWave18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="afterimage-wave18-")
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "production.afterimage"
        cls.author = cls.root / "author"
        subprocess.run([
            sys.executable, str(ROOT / "tools/build_slice.py"), str(cls.bundle),
            "--manifest", str(ROOT / "manifests/production_release.json"),
            "--author-dir", str(cls.author), "--title", "Afterimage production release 1.9",
            "--revision", "production-dev-1.9.0",
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

    def verify_candidate(self, case_id: str, name: str, witness: dict[str, object], facts: set[str]) -> dict[str, object]:
        witness.pop("meta", None)
        path = self.root / f"{case_id}-{name}.json"
        path.write_bytes(cre.canonical_bytes(witness))
        return verifier.verify_witness(self.world_path, path, facts)

    def test_all_five_author_baselines_are_valid(self) -> None:
        expected = {"MERGE.013": 129, "MOSAIC.011": 149, "PULSE.012": 136, "CASCADE.018": 109, "CASCADE.019": 89}
        for case_id, score in expected.items():
            receipt = json.loads((self.author / f"{case_id}.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["valid"], case_id)
            self.assertEqual(receipt["score"]["total"], score, case_id)

    def test_equivocation_taints_both_statements_from_the_source(self) -> None:
        candidate = self.witness("MERGE.013")
        answer = copy.deepcopy(candidate["answer"])
        agreeing = next(item for item in answer["rejected"] if item["reason"] == "equivocation")
        answer["rejected"].remove(agreeing)
        answer["accepted"].append({"event": agreeing["event"], "at": 12})
        candidate["answer"] = answer
        receipt = self.verify_candidate("MERGE.013", "partial-source", candidate, {"case:MERGE.012", "case:COVENANT.002"})
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "merge_equivocation")

    def test_whole_city_is_non_square_covered_and_layer_connected(self) -> None:
        source = json.loads((ROOT / "content/production/cases/MOSAIC.011.json").read_text(encoding="utf-8"))
        fragments = [event["payload"] for event in source["events"]]
        metrics = mosaic.validate_answer(fragments, source["case"]["author_answer"], source["case"]["mosaic_contract"])
        self.assertEqual(metrics, {"unexplained_weight": 0, "graph_size": 29, "certificate_units": 11})
        wrong = copy.deepcopy(source["case"]["mosaic_contract"])
        wrong["layers"]["portals"].remove("v4")
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(fragments, source["case"]["author_answer"], wrong)
        self.assertEqual(raised.exception.code, "mosaic_layer")

    def test_reorder_buffer_drains_the_complete_prefix(self) -> None:
        source = json.loads((ROOT / "content/production/cases/PULSE.012.json").read_text(encoding="utf-8"))
        program = source["case"]["author_program"]
        contract = source["case"]["pulse_contract"]
        self.assertEqual(pulse.verify_program(program, contract), {"program_bytes": 5625, "worst_latency": 3, "live_state_cells": 5, "domain_cases": 1457})
        expected = pulse.expected_outputs(contract, ((0, 2), (1, 0), (2, 1)))
        self.assertEqual([item["at"] for item in expected], [1, 2, 2])
        one_only = copy.deepcopy(program)
        for handler in one_only["handlers"]:
            handler["actions"] = handler["actions"][:2]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(one_only, contract)
        self.assertEqual(raised.exception.code, "pulse_counterexample")

    def test_missing_cause_rejects_each_proper_subset(self) -> None:
        candidate = self.witness("CASCADE.018")
        candidate["intervention"]["operations"] = candidate["intervention"]["operations"][:1]
        receipt = self.verify_candidate(
            "CASCADE.018", "one-event",
            candidate,
            {"case:CASCADE.017", "case:MERGE.013", "case:MOSAIC.011", "case:PULSE.012"},
        )
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "cascade_minimality")

    def test_correction_notice_preserves_the_prior_statement(self) -> None:
        case = self.case("CASCADE.019")
        witness = self.witness("CASCADE.019")
        operation = witness["intervention"]["operations"][0]
        replay = verifier.replay_case(self.world, case, [operation], cre.root_branch_id(self.world.bundle))
        prior = next(event for event in replay.events.values() if event["topic"] == "notice.prior")
        summary = next(event for event in replay.events.values() if event["topic"] == "contract.summary")
        self.assertEqual(prior["payload"]["route"], "underground")
        self.assertEqual(summary["payload"]["prior_route"], "underground")
        self.assertEqual(summary["payload"]["new_route"], "surface")
        self.assertTrue(summary["payload"]["safe"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

