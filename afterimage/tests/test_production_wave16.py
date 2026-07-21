#!/usr/bin/env python3
"""Integration and adversarial tests for production wave 16."""

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
import mosaic  # noqa: E402
import verify_witness as verifier  # noqa: E402


class ProductionWave16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="afterimage-wave16-")
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "production.afterimage"
        cls.author = cls.root / "author"
        subprocess.run([
            sys.executable, str(ROOT / "tools/build_slice.py"), str(cls.bundle),
            "--manifest", str(ROOT / "manifests/production_release.json"),
            "--author-dir", str(cls.author), "--title", "Afterimage production release 1.7",
            "--revision", "production-dev-1.7.0",
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

    def answer(self, case_id: str) -> dict[str, object]:
        return json.loads((self.author / f"{case_id}.witness.json").read_text(encoding="utf-8"))["answer"]

    def verify_mutation(self, case_id: str, name: str, answer: dict[str, object], facts: set[str]) -> dict[str, object]:
        witness = json.loads((self.author / f"{case_id}.witness.json").read_text(encoding="utf-8"))
        witness["answer"] = answer
        witness.pop("meta", None)
        path = self.root / f"{case_id}-{name}.json"
        path.write_bytes(cre.canonical_bytes(witness))
        return verifier.verify_witness(self.world_path, path, facts)

    def test_all_five_author_baselines_are_valid(self) -> None:
        expected = {"CASCADE.015": 99, "MERGE.011": 129, "MOSAIC.009": 139, "PULSE.010": 127, "CASCADE.016": 109}
        for case_id, score in expected.items():
            receipt = json.loads((self.author / f"{case_id}.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["valid"], case_id)
            self.assertEqual(receipt["score"]["total"], score, case_id)

    def test_redacted_feeder_binds_without_disclosing_restricted_members(self) -> None:
        answer = self.answer("CASCADE.015")
        encoded = json.dumps(answer, sort_keys=True)
        for secret in ("dispatch-77/U0", "SHADOW-7", "sealed-grid-19"):
            self.assertNotIn(secret, encoded)
        altered = copy.deepcopy(answer)
        altered["relay"]["provenance_digest"] = "sha256:" + "0" * 64
        receipt = self.verify_mutation("CASCADE.015", "forged-digest", altered, {"case:CASCADE.014", "case:LENS.003"})
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "cascade_proof")

    def test_weighted_evidence_rejects_record_count_tie_reasoning(self) -> None:
        case = self.case("MERGE.011")
        logical = kit.resolve_logical_world(self.world, case["world"])
        by_key = {event["payload"]["key"]: event["id"] for event in logical.base_events}
        altered = copy.deepcopy(self.answer("MERGE.011"))
        altered["accepted"] = [{"event": by_key["L1"], "at": 6}, {"event": by_key["L2"], "at": 8}]
        altered["rejected"] = [{"event": by_key["S1"], "reason": "outvoted"}, {"event": by_key["S2"], "reason": "outvoted"}]
        receipt = self.verify_mutation("MERGE.011", "count-tie", altered, {"case:MERGE.010", "case:CASCADE.015"})
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "merge_weight")

    def test_adversarial_survey_requires_the_public_support_margin(self) -> None:
        source = json.loads((ROOT / "content/production/cases/MOSAIC.009.json").read_text(encoding="utf-8"))
        fragments = [copy.deepcopy(event["payload"]) for event in source["events"]]
        next(fragment for fragment in fragments if fragment["id"] == "F9-NORTH-SIGNED")["weight"] = 4
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(fragments, source["case"]["author_answer"], source["case"]["mosaic_contract"])
        self.assertEqual(raised.exception.code, "mosaic_weight")

    def test_shared_deadline_rejects_channel_local_allowance(self) -> None:
        altered = copy.deepcopy(self.answer("PULSE.010"))
        handler = next(item for item in altered["program"]["handlers"] if item["id"] == "deadline.B")

        def replace_twos(value: object) -> None:
            if isinstance(value, list):
                if value == ["const", 2]:
                    value[1] = 4
                else:
                    for item in value:
                        replace_twos(item)
            elif isinstance(value, dict):
                for item in value.values():
                    replace_twos(item)

        replace_twos(handler)
        receipt = self.verify_mutation("PULSE.010", "local-allowance", altered, {"case:PULSE.009", "case:MERGE.011", "case:MOSAIC.009"})
        self.assertFalse(receipt["valid"])
        self.assertEqual(receipt["diagnostics"][0]["code"], "pulse_counterexample")

    def test_witness_gap_has_one_supported_latent_cause(self) -> None:
        case = self.case("CASCADE.016")
        parent = cre.root_branch_id(self.world.bundle)
        valid = []
        for cause in ("breaker-fault", "manual-open", "thermal-overload"):
            replay = verifier.replay_case(self.world, case, [{"kind": "inject", "topic": "feeder.trip-cause", "at": 10, "payload": {"feeder": "FEEDER-9", "cause": cause, "witness": "bounded-gap"}, "parents": []}], parent)
            if replay.records and replay.records[0].get("safe") is True:
                valid.append(cause)
        self.assertEqual(valid, ["thermal-overload"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
