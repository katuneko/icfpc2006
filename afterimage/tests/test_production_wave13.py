#!/usr/bin/env python3
"""Integration and adversarial tests for production wave 13."""

from __future__ import annotations

import copy
import itertools
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


class ProductionWave13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="afterimage-wave13-")
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "production.afterimage"
        cls.author = cls.root / "author"
        subprocess.run(
            [
                sys.executable, str(ROOT / "tools" / "build_slice.py"), str(cls.bundle),
                "--manifest", str(ROOT / "manifests" / "production_release.json"),
                "--author-dir", str(cls.author),
                "--title", "Afterimage production release 1.4",
                "--revision", "production-dev-1.4.0",
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

    def test_all_five_author_baselines_are_valid(self) -> None:
        expected_scores = {
            "CASCADE.012": 109, "MERGE.008": 106, "MOSAIC.006": 109,
            "PULSE.007": 109, "LENS.002": 496,
        }
        for case_id, score in expected_scores.items():
            receipt = json.loads((self.author / f"{case_id}.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["valid"], case_id)
            self.assertEqual(receipt["score"]["total"], score, case_id)
            self.assertEqual(receipt["unlocks"], [f"case:{case_id}"], case_id)

    def test_three_departments_has_one_safe_three_retime_schedule(self) -> None:
        case = self.case("CASCADE.012")
        logical = kit.resolve_logical_world(self.world, case["world"])
        by_topic = {event["topic"]: event for event in logical.base_events}
        parent = cre.root_branch_id(self.world.bundle)
        safe = []
        for command, traffic, hospital in itertools.product(range(6, 12), repeat=3):
            operations = [
                {"kind": "retime", "event": by_topic["agency.bridge-command"]["id"], "at": command},
                {"kind": "retime", "event": by_topic["agency.traffic-release"]["id"], "at": traffic},
                {"kind": "retime", "event": by_topic["agency.hospital-departure"]["id"], "at": hospital},
            ]
            replay = verifier.replay_case(self.world, case, operations, parent)
            if replay.records[0]["safe"]:
                safe.append((command, traffic, hospital))
        self.assertEqual(safe, [(8, 10, 11)])

    def test_conflict_component_rejects_overbroad_deletion(self) -> None:
        case = self.case("MERGE.008")
        logical = kit.resolve_logical_world(self.world, case["world"])
        by_key = {event["payload"]["key"]: event["id"] for event in logical.base_events}
        witness = json.loads((self.author / "MERGE.008.witness.json").read_text(encoding="utf-8"))
        altered = copy.deepcopy(witness)
        event_id = by_key["A3"]
        altered["answer"]["accepted"] = [
            item for item in altered["answer"]["accepted"] if item["event"] != event_id
        ]
        altered["answer"]["rejected"].append({"event": event_id, "reason": "conflict_set"})
        altered["answer"]["certificate"] = [
            edge for edge in altered["answer"]["certificate"]
            if event_id not in (edge["before"], edge["after"])
        ]
        altered.pop("meta", None)
        path = self.root / "merge-008-overbroad.json"
        path.write_bytes(cre.canonical_bytes(altered))
        receipt = verifier.verify_witness(
            self.world_path, path, {"case:MERGE.007", "case:CASCADE.012"},
        )
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "merge_minimal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
