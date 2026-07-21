#!/usr/bin/env python3
"""Integration and adversarial tests for production wave 14."""

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


class ProductionWave14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="afterimage-wave14-")
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "production.afterimage"
        cls.author = cls.root / "author"
        subprocess.run(
            [
                sys.executable, str(ROOT / "tools" / "build_slice.py"), str(cls.bundle),
                "--manifest", str(ROOT / "manifests" / "production_release.json"),
                "--author-dir", str(cls.author),
                "--title", "Afterimage production release 1.5",
                "--revision", "production-dev-1.5.0",
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
        descriptor = next(
            item for item in self.world.json_values["cases/index.json"]["cases"]
            if item["id"] == case_id
        )
        return verifier.validate_case_descriptor(descriptor, self.world)

    def verify_mutation(self, case_id: str, name: str, answer: dict[str, object], facts: set[str]) -> dict[str, object]:
        witness = json.loads((self.author / f"{case_id}.witness.json").read_text(encoding="utf-8"))
        witness["answer"] = answer
        witness.pop("meta", None)
        path = self.root / f"{case_id}-{name}.json"
        path.write_bytes(cre.canonical_bytes(witness))
        return verifier.verify_witness(self.world_path, path, facts)

    def test_all_five_author_baselines_are_valid(self) -> None:
        expected_scores = {
            "CASCADE.013": 89, "MERGE.009": 116, "MOSAIC.007": 119,
            "PULSE.008": 118, "COVENANT.001": 445,
        }
        for case_id, score in expected_scores.items():
            receipt = json.loads((self.author / f"{case_id}.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["valid"], case_id)
            self.assertEqual(receipt["score"]["total"], score, case_id)
            self.assertEqual(receipt["unlocks"], [f"case:{case_id}"], case_id)

    def test_silent_gauge_has_one_robust_reserve_value(self) -> None:
        case = self.case("CASCADE.013")
        logical = kit.resolve_logical_world(self.world, case["world"])
        setting = next(event for event in logical.base_events if event["topic"] == "flood.reserve-setting")
        parent = cre.root_branch_id(self.world.bundle)
        safe = []
        for reserve in range(7):
            replay = verifier.replay_case(
                self.world, case,
                [{"kind": "replace", "event": setting["id"], "pointer": "/payload/reserve", "value": reserve}],
                parent,
            )
            if replay.records[0]["safe"]:
                safe.append(reserve)
        self.assertEqual(safe, [4])

    def test_minimal_cut_rejects_heavier_and_uncovered_cuts(self) -> None:
        case = self.case("MERGE.009")
        logical = kit.resolve_logical_world(self.world, case["world"])
        by_key = {event["payload"]["key"]: event["id"] for event in logical.base_events}
        author_answer = json.loads((self.author / "MERGE.009.witness.json").read_text(encoding="utf-8"))["answer"]
        facts = {"case:MERGE.008", "case:CASCADE.013"}

        heavier = copy.deepcopy(author_answer)
        heavier["accepted"] = [item for item in heavier["accepted"] if item["event"] not in {by_key["G1"], by_key["G3"]}]
        heavier["accepted"].append({"event": by_key["G2"], "at": 8})
        heavier["rejected"] = [
            {"event": by_key["G1"], "reason": "cut"},
            {"event": by_key["G3"], "reason": "cut"},
        ]
        receipt = self.verify_mutation("MERGE.009", "heavier", heavier, facts)
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "merge_cut")

        uncovered = copy.deepcopy(author_answer)
        uncovered["accepted"].append({"event": by_key["G2"], "at": 8})
        uncovered["accepted"] = [item for item in uncovered["accepted"] if item["event"] != by_key["G1"]]
        uncovered["rejected"] = [{"event": by_key["G1"], "reason": "cut"}]
        receipt = self.verify_mutation("MERGE.009", "uncovered", uncovered, facts)
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "merge_cut")

    def test_dispatch_covenant_rejects_false_response_claim(self) -> None:
        author_answer = json.loads((self.author / "COVENANT.001.witness.json").read_text(encoding="utf-8"))["answer"]
        altered = copy.deepcopy(author_answer)
        altered["claimed_response_bound"] = 4
        receipt = self.verify_mutation(
            "COVENANT.001", "false-bound", altered,
            {"case:CASCADE.013", "case:MERGE.009", "case:PULSE.008"},
        )
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "covenant_claim")


if __name__ == "__main__":
    unittest.main(verbosity=2)
