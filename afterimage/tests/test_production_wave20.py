#!/usr/bin/env python3
"""Integration and adversarial tests for the final production wave."""

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
import pulse  # noqa: E402
import verify_witness as verifier  # noqa: E402


class ProductionWave20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="afterimage-wave20-")
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "production.afterimage"
        cls.author = cls.root / "author"
        subprocess.run(
            [
                sys.executable, str(ROOT / "tools/build_slice.py"), str(cls.bundle),
                "--manifest", str(ROOT / "manifests/production_release.json"),
                "--author-dir", str(cls.author), "--title", "Afterimage production release 2.1",
                "--revision", "production-dev-2.1.0",
            ],
            check=True, capture_output=True, text=True,
        )
        cls.world_path = cls.root / "world"
        kit.extract_bundle(kit.load_bundle(cls.bundle), cls.world_path)
        cls.world = kit.verify_world(cls.world_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def witness(self, case_id: str) -> dict:
        return json.loads((self.author / f"{case_id}.witness.json").read_text())

    def case(self, case_id: str) -> dict:
        descriptor = next(item for item in self.world.json_values["cases/index.json"]["cases"] if item["id"] == case_id)
        return verifier.validate_case_descriptor(descriptor, self.world)

    def verify(self, case_id: str, name: str, witness: dict, facts: set[str]) -> dict:
        witness.pop("meta", None)
        path = self.root / f"{case_id}-{name}.json"
        path.write_bytes(cre.canonical_bytes(witness))
        return verifier.verify_witness(self.world_path, path, facts)

    def test_all_five_author_baselines_are_valid(self) -> None:
        expected = {"MERGE.015": 139, "PULSE.014": 169, "CASCADE.023": 119, "CASCADE.024": 129, "PARADOX.001": 299}
        for case_id, score in expected.items():
            receipt = json.loads((self.author / f"{case_id}.receipt.json").read_text())
            self.assertTrue(receipt["valid"], case_id)
            self.assertEqual(receipt["score"]["total"], score, case_id)

    def test_reconstruction_preserves_incomparability_and_canonical_schedule(self) -> None:
        witness = self.witness("MERGE.015")
        accepted = {item["event"]: item for item in witness["answer"]["accepted"]}
        logical = kit.resolve_logical_world(self.world, self.case("MERGE.015")["world"])
        by_key = {event["payload"]["key"]: event["id"] for event in logical.base_events}
        accepted[by_key["B"]]["at"] = 12
        receipt = self.verify("MERGE.015", "noncanonical", witness, {"case:MERGE.014", "case:CASCADE.022"})
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "merge_non_unique")

        missing = self.witness("MERGE.015")
        missing["answer"]["certificate"] = missing["answer"]["certificate"][:-1]
        receipt = self.verify("MERGE.015", "missing-edge", missing, {"case:MERGE.014", "case:CASCADE.022"})
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "merge_compression")

    def test_city_clock_exhausts_two_rounds_and_duplicate_coalescing(self) -> None:
        source = json.loads((ROOT / "content/production/cases/PULSE.014.json").read_text())["case"]
        self.assertEqual(
            pulse.verify_program(source["author_program"], source["pulse_contract"]),
            {"program_bytes": 1564, "worst_latency": 0, "live_state_cells": 3, "domain_cases": 27064},
        )
        broken = copy.deepcopy(source["author_program"])
        for handler in broken["handlers"]:
            handler["actions"][1]["actions"] = handler["actions"][1]["actions"][:1]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(broken, source["pulse_contract"])
        self.assertEqual(raised.exception.code, "pulse_counterexample")

    def test_hearing_rejects_stale_epoch_then_accepts_one_binding_repair(self) -> None:
        case = self.case("CASCADE.023")
        root = verifier.replay_root_case(self.world, case)
        witness = self.witness("CASCADE.023")
        repaired = verifier.replay_case(
            self.world, case, witness["intervention"]["operations"], cre.root_branch_id(self.world.bundle)
        )
        self.assertFalse(root.records[0]["safe"])
        self.assertTrue(repaired.records[0]["safe"])
        self.assertEqual(repaired.records[0]["epoch"], 7)

    def test_both_chosen_tomorrows_are_safe_and_publicly_equal(self) -> None:
        case = self.case("CASCADE.024")
        author = self.witness("CASCADE.024")
        surface = verifier.replay_case(self.world, case, author["intervention"]["operations"], cre.root_branch_id(self.world.bundle))
        underground_operations = copy.deepcopy(author["intervention"]["operations"])
        underground_operations[0]["value"] = "underground"
        underground = verifier.replay_case(self.world, case, underground_operations, cre.root_branch_id(self.world.bundle))
        self.assertTrue(surface.records[0]["safe"])
        self.assertEqual(surface.records, underground.records)
        self.assertNotEqual(surface.branch, underground.branch)

    def test_paradox_uses_shared_world_fixed_origin_and_semantic_difference(self) -> None:
        cascade = self.case("CASCADE.024")
        paradox_case = self.case("PARADOX.001")
        self.assertEqual(cascade["world"], paradox_case["world"])
        self.assertEqual(cascade["projection"], paradox_case["projection"])
        self.assertFalse((self.world_path / "worlds/PARADOX.001").exists())

        same = self.witness("PARADOX.001")
        same["answer"]["right_history"] = copy.deepcopy(same["answer"]["left_history"])
        receipt = self.verify(
            "PARADOX.001", "same-history", same,
            {"case:COVENANT.002", "case:LENS.003", "case:CASCADE.024"},
        )
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "paradox_same_branch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
