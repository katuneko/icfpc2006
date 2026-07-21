#!/usr/bin/env python3
"""Golden build test for the first full-production Afterimage release."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference/python"))

import afterimage_kit as kit  # noqa: E402
import cre  # noqa: E402
import verify_witness as verifier  # noqa: E402


GOLDEN = json.loads((ROOT / "content/production/golden.json").read_text(encoding="utf-8"))


class ProductionContentTests(unittest.TestCase):
    def test_release_builds_to_golden_and_private_baseline_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="afterimage-production-test-") as temporary:
            root = Path(temporary)
            bundle = root / "production.afterimage"
            author = root / "author"
            completed = subprocess.run(
                [
                    sys.executable, str(ROOT / "tools/build_slice.py"), str(bundle),
                    "--manifest", str(ROOT / "manifests/production_release.json"),
                    "--author-dir", str(author),
                    "--title", "Afterimage production release 2.1",
                    "--revision", "production-dev-2.1.0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["archive_sha256"], GOLDEN["archive_sha256"])
            self.assertEqual(summary["bundle"], GOLDEN["bundle"])
            for key, expected in GOLDEN["counts"].items():
                self.assertEqual(summary[key], expected)
            self.assertEqual(summary["author"]["witnesses"], 75)
            self.assertEqual(summary["author"]["scores"]["PULSE.002"], 78)
            self.assertEqual(summary["author"]["scores"]["PULSE.003"], 86)
            self.assertEqual(summary["author"]["scores"]["MERGE.002"], 79)
            self.assertEqual(summary["author"]["scores"]["MOSAIC.002"], 79)
            self.assertEqual(summary["author"]["scores"]["CASCADE.004"], 69)
            self.assertEqual(summary["author"]["scores"]["CASCADE.005"], 69)
            self.assertEqual(summary["author"]["scores"]["CASCADE.006"], 79)
            self.assertEqual(summary["author"]["scores"]["CASCADE.007"], 79)
            self.assertEqual(summary["author"]["scores"]["MERGE.003"], 89)
            self.assertEqual(summary["author"]["scores"]["MOSAIC.003"], 89)
            self.assertEqual(summary["author"]["scores"]["PULSE.004"], 95)
            self.assertEqual(summary["author"]["scores"]["CASCADE.008"], 89)
            self.assertEqual(summary["author"]["scores"]["MERGE.004"], 89)
            self.assertEqual(summary["author"]["scores"]["CASCADE.009"], 89)
            self.assertEqual(summary["author"]["scores"]["MERGE.005"], 99)
            self.assertEqual(summary["author"]["scores"]["PULSE.005"], 96)
            self.assertEqual(summary["author"]["scores"]["CASCADE.010"], 99)
            self.assertEqual(summary["author"]["scores"]["MERGE.006"], 99)
            self.assertEqual(summary["author"]["scores"]["MOSAIC.004"], 99)
            self.assertEqual(summary["author"]["scores"]["CASCADE.011"], 109)
            self.assertEqual(summary["author"]["scores"]["MERGE.007"], 109)
            self.assertEqual(summary["author"]["scores"]["MOSAIC.005"], 99)
            self.assertEqual(summary["author"]["scores"]["PULSE.006"], 106)
            self.assertEqual(summary["author"]["scores"]["CASCADE.012"], 109)
            self.assertEqual(summary["author"]["scores"]["MERGE.008"], 106)
            self.assertEqual(summary["author"]["scores"]["MOSAIC.006"], 109)
            self.assertEqual(summary["author"]["scores"]["PULSE.007"], 109)
            self.assertEqual(summary["author"]["scores"]["LENS.002"], 496)
            self.assertEqual(summary["author"]["scores"]["CASCADE.013"], 89)
            self.assertEqual(summary["author"]["scores"]["MERGE.009"], 116)
            self.assertEqual(summary["author"]["scores"]["MOSAIC.007"], 119)
            self.assertEqual(summary["author"]["scores"]["PULSE.008"], 118)
            self.assertEqual(summary["author"]["scores"]["COVENANT.001"], 445)
            self.assertEqual(summary["author"]["scores"]["CASCADE.014"], 99)
            self.assertEqual(summary["author"]["scores"]["MERGE.010"], 119)
            self.assertEqual(summary["author"]["scores"]["MOSAIC.008"], 129)
            self.assertEqual(summary["author"]["scores"]["PULSE.009"], 117)
            self.assertEqual(summary["author"]["scores"]["LENS.003"], 616)
            self.assertEqual(summary["author"]["scores"]["CASCADE.015"], 99)
            self.assertEqual(summary["author"]["scores"]["MERGE.011"], 129)
            self.assertEqual(summary["author"]["scores"]["MOSAIC.009"], 139)
            self.assertEqual(summary["author"]["scores"]["PULSE.010"], 127)
            self.assertEqual(summary["author"]["scores"]["CASCADE.016"], 109)
            self.assertEqual(summary["author"]["scores"]["CASCADE.017"], 109)
            self.assertEqual(summary["author"]["scores"]["MERGE.012"], 129)
            self.assertEqual(summary["author"]["scores"]["MOSAIC.010"], 149)
            self.assertEqual(summary["author"]["scores"]["PULSE.011"], 129)
            self.assertEqual(summary["author"]["scores"]["COVENANT.002"], 523)
            self.assertEqual(summary["author"]["scores"]["MERGE.013"], 129)
            self.assertEqual(summary["author"]["scores"]["MOSAIC.011"], 149)
            self.assertEqual(summary["author"]["scores"]["PULSE.012"], 136)
            self.assertEqual(summary["author"]["scores"]["CASCADE.018"], 109)
            self.assertEqual(summary["author"]["scores"]["CASCADE.019"], 89)
            self.assertEqual(summary["author"]["scores"]["MERGE.014"], 129)
            self.assertEqual(summary["author"]["scores"]["PULSE.013"], 149)
            self.assertEqual(summary["author"]["scores"]["CASCADE.020"], 99)
            self.assertEqual(summary["author"]["scores"]["CASCADE.021"], 109)
            self.assertEqual(summary["author"]["scores"]["CASCADE.022"], 109)
            self.assertEqual(summary["author"]["scores"]["MERGE.015"], 139)
            self.assertEqual(summary["author"]["scores"]["PULSE.014"], 169)
            self.assertEqual(summary["author"]["scores"]["CASCADE.023"], 119)
            self.assertEqual(summary["author"]["scores"]["CASCADE.024"], 129)
            self.assertEqual(summary["author"]["scores"]["PARADOX.001"], 299)

            receipt = json.loads((author / "PULSE.002.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["valid"])
            self.assertEqual(receipt["metrics"], GOLDEN["PULSE.002"]["metrics"])
            self.assertEqual(receipt["score"], GOLDEN["PULSE.002"]["score"])
            self.assertEqual(receipt["unlocks"], ["case:PULSE.002"])

            timeout_receipt = json.loads((author / "PULSE.003.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(timeout_receipt["valid"])
            self.assertEqual(timeout_receipt["metrics"], GOLDEN["PULSE.003"]["metrics"])
            self.assertEqual(timeout_receipt["score"], GOLDEN["PULSE.003"]["score"])
            self.assertEqual(timeout_receipt["unlocks"], ["case:PULSE.003"])

            mosaic_receipt = json.loads((author / "MOSAIC.002.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(mosaic_receipt["valid"])
            self.assertEqual(mosaic_receipt["metrics"], GOLDEN["MOSAIC.002"]["metrics"])
            self.assertEqual(mosaic_receipt["score"], GOLDEN["MOSAIC.002"]["score"])
            self.assertEqual(mosaic_receipt["unlocks"], ["case:MOSAIC.002"])

            with zipfile.ZipFile(bundle, "r") as archive:
                names = set(archive.namelist())
                self.assertIn("story/PULSE.002.md", names)
                self.assertIn("story/PULSE.003.md", names)
                self.assertIn("story/MOSAIC.002.md", names)
                self.assertIn("story/CASCADE.004.md", names)
                self.assertIn("story/CASCADE.005.md", names)
                self.assertIn("story/CASCADE.006.md", names)
                self.assertIn("story/CASCADE.007.md", names)
                self.assertIn("story/MERGE.003.md", names)
                self.assertIn("story/MOSAIC.003.md", names)
                self.assertIn("story/PULSE.004.md", names)
                self.assertIn("story/CASCADE.008.md", names)
                self.assertIn("story/MERGE.004.md", names)
                self.assertIn("story/CASCADE.009.md", names)
                self.assertIn("story/MERGE.005.md", names)
                self.assertIn("story/PULSE.005.md", names)
                self.assertIn("story/CASCADE.010.md", names)
                self.assertIn("story/MERGE.006.md", names)
                self.assertIn("story/MOSAIC.004.md", names)
                self.assertIn("story/CASCADE.011.md", names)
                self.assertIn("story/MERGE.007.md", names)
                self.assertIn("story/MOSAIC.005.md", names)
                self.assertIn("story/PULSE.006.md", names)
                self.assertIn("story/CASCADE.012.md", names)
                self.assertIn("story/MERGE.008.md", names)
                self.assertIn("story/MOSAIC.006.md", names)
                self.assertIn("story/PULSE.007.md", names)
                self.assertIn("story/LENS.002.md", names)
                self.assertIn("story/CASCADE.013.md", names)
                self.assertIn("story/MERGE.009.md", names)
                self.assertIn("story/MOSAIC.007.md", names)
                self.assertIn("story/PULSE.008.md", names)
                self.assertIn("story/COVENANT.001.md", names)
                self.assertIn("story/CASCADE.014.md", names)
                self.assertIn("story/MERGE.010.md", names)
                self.assertIn("story/MOSAIC.008.md", names)
                self.assertIn("story/PULSE.009.md", names)
                self.assertIn("story/LENS.003.md", names)
                self.assertIn("story/CASCADE.015.md", names)
                self.assertIn("story/MERGE.011.md", names)
                self.assertIn("story/MOSAIC.009.md", names)
                self.assertIn("story/PULSE.010.md", names)
                self.assertIn("story/CASCADE.016.md", names)
                self.assertIn("story/CASCADE.017.md", names)
                self.assertIn("story/MERGE.012.md", names)
                self.assertIn("story/MOSAIC.010.md", names)
                self.assertIn("story/PULSE.011.md", names)
                self.assertIn("story/COVENANT.002.md", names)
                self.assertIn("story/MERGE.013.md", names)
                self.assertIn("story/MOSAIC.011.md", names)
                self.assertIn("story/PULSE.012.md", names)
                self.assertIn("story/CASCADE.018.md", names)
                self.assertIn("story/CASCADE.019.md", names)
                self.assertIn("story/MERGE.014.md", names)
                self.assertIn("story/PULSE.013.md", names)
                self.assertIn("story/CASCADE.020.md", names)
                self.assertIn("story/CASCADE.021.md", names)
                self.assertIn("story/CASCADE.022.md", names)
                self.assertIn("story/MERGE.015.md", names)
                self.assertIn("story/PULSE.014.md", names)
                self.assertIn("story/CASCADE.023.md", names)
                self.assertIn("story/CASCADE.024.md", names)
                self.assertIn("story/PARADOX.001.md", names)
                self.assertFalse(any("witness" in name or "receipt" in name for name in names))
                story = archive.read("story/PULSE.002.md").decode("utf-8")
                self.assertIn("canonical first arrival", story)
                self.assertNotIn("author_invariant", story)
                timeout_story = archive.read("story/PULSE.003.md").decode("utf-8")
                self.assertIn("exactly at the deadline", timeout_story)
                mosaic_story = archive.read("story/MOSAIC.002.md").decode("utf-8")
                self.assertIn("independently corroborated", mosaic_story)
                cascade_story = archive.read("story/CASCADE.004.md").decode("utf-8")
                self.assertIn("restricted-relay finding", cascade_story)
                self.assertIn("tick 9", cascade_story)
                queue_story = archive.read("story/CASCADE.005.md").decode("utf-8")
                self.assertIn("non-preemptive grant", queue_story)
                ice_story = archive.read("story/CASCADE.006.md").decode("utf-8")
                self.assertIn("two unrelated alerts", ice_story)
                coupling_story = archive.read("story/CASCADE.007.md").decode("utf-8")
                self.assertIn("grid-window-31", coupling_story)
                merge_story = archive.read("story/MERGE.003.md").decode("utf-8")
                self.assertIn("without inventing", merge_story)
                rotated_story = archive.read("story/MOSAIC.003.md").decode("utf-8")
                self.assertIn("collapse is legitimate", rotated_story)
                token_story = archive.read("story/PULSE.004.md").decode("utf-8")
                self.assertIn("rejected work must not consume", token_story)
                island_story = archive.read("story/CASCADE.008.md").decode("utf-8")
                self.assertIn("sufficient daily energy", island_story)
                compensation_story = archive.read("story/MERGE.004.md").decode("utf-8")
                self.assertIn("without rewriting the past", compensation_story)
                train_story = archive.read("story/CASCADE.009.md").decode("utf-8")
                self.assertIn("180 evacuees", train_story)
                split_story = archive.read("story/MERGE.005.md").decode("utf-8")
                self.assertIn("synthetic third history", split_story)
                quorum_story = archive.read("story/PULSE.005.md").decode("utf-8")
                self.assertIn("two distinct sources", quorum_story)
                cooling_story = archive.read("story/CASCADE.010.md").decode("utf-8")
                self.assertIn("saturates at five", cooling_story)
                ledger_story = archive.read("story/MERGE.006.md").decode("utf-8")
                self.assertIn("every timestamp can coexist", ledger_story)
                missing_story = archive.read("story/MOSAIC.004.md").decode("utf-8")
                self.assertIn("central manifold tile", missing_story)
                bridge_story = archive.read("story/CASCADE.011.md").decode("utf-8")
                self.assertIn("stranded behind drawbridge", bridge_story)
                clocks_story = archive.read("story/MERGE.007.md").decode("utf-8")
                self.assertIn("two: the north controller", clocks_story)
                duplicate_story = archive.read("story/MOSAIC.005.md").decode("utf-8")
                self.assertIn("cached half-turned copy", duplicate_story)
                barrier_story = archive.read("story/PULSE.006.md").decode("utf-8")
                self.assertIn("three facts must coexist", barrier_story)

            merge_receipt = json.loads((author / "MERGE.002.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(merge_receipt["valid"])
            self.assertEqual(merge_receipt["metrics"], GOLDEN["MERGE.002"]["metrics"])
            self.assertEqual(merge_receipt["score"], GOLDEN["MERGE.002"]["score"])

            cascade_receipt = json.loads((author / "CASCADE.004.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(cascade_receipt["valid"])
            self.assertEqual(cascade_receipt["metrics"], GOLDEN["CASCADE.004"]["metrics"])
            self.assertEqual(cascade_receipt["score"], GOLDEN["CASCADE.004"]["score"])
            self.assertEqual(cascade_receipt["unlocks"], ["case:CASCADE.004"])

            queue_receipt = json.loads((author / "CASCADE.005.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(queue_receipt["valid"])
            self.assertEqual(queue_receipt["metrics"], GOLDEN["CASCADE.005"]["metrics"])
            self.assertEqual(queue_receipt["score"], GOLDEN["CASCADE.005"]["score"])
            self.assertEqual(queue_receipt["unlocks"], ["case:CASCADE.005"])

            ice_receipt = json.loads((author / "CASCADE.006.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(ice_receipt["valid"])
            self.assertEqual(ice_receipt["metrics"], GOLDEN["CASCADE.006"]["metrics"])
            self.assertEqual(ice_receipt["score"], GOLDEN["CASCADE.006"]["score"])
            self.assertEqual(ice_receipt["unlocks"], ["case:CASCADE.006"])

            coupling_receipt = json.loads((author / "CASCADE.007.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(coupling_receipt["valid"])
            self.assertEqual(coupling_receipt["metrics"], GOLDEN["CASCADE.007"]["metrics"])
            self.assertEqual(coupling_receipt["score"], GOLDEN["CASCADE.007"]["score"])
            self.assertEqual(coupling_receipt["unlocks"], ["case:CASCADE.007"])

            missing_ack_receipt = json.loads((author / "MERGE.003.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(missing_ack_receipt["valid"])
            self.assertEqual(missing_ack_receipt["metrics"], GOLDEN["MERGE.003"]["metrics"])
            self.assertEqual(missing_ack_receipt["score"], GOLDEN["MERGE.003"]["score"])

            rotated_receipt = json.loads((author / "MOSAIC.003.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(rotated_receipt["valid"])
            self.assertEqual(rotated_receipt["metrics"], GOLDEN["MOSAIC.003"]["metrics"])
            self.assertEqual(rotated_receipt["score"], GOLDEN["MOSAIC.003"]["score"])

            token_receipt = json.loads((author / "PULSE.004.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(token_receipt["valid"])
            self.assertEqual(token_receipt["metrics"], GOLDEN["PULSE.004"]["metrics"])
            self.assertEqual(token_receipt["score"], GOLDEN["PULSE.004"]["score"])
            self.assertEqual(token_receipt["unlocks"], ["case:PULSE.004"])

            island_receipt = json.loads((author / "CASCADE.008.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(island_receipt["valid"])
            self.assertEqual(island_receipt["metrics"], GOLDEN["CASCADE.008"]["metrics"])
            self.assertEqual(island_receipt["score"], GOLDEN["CASCADE.008"]["score"])
            self.assertEqual(island_receipt["unlocks"], ["case:CASCADE.008"])

            compensation_receipt = json.loads((author / "MERGE.004.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(compensation_receipt["valid"])
            self.assertEqual(compensation_receipt["metrics"], GOLDEN["MERGE.004"]["metrics"])
            self.assertEqual(compensation_receipt["score"], GOLDEN["MERGE.004"]["score"])
            self.assertEqual(compensation_receipt["unlocks"], ["case:MERGE.004"])

            train_receipt = json.loads((author / "CASCADE.009.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(train_receipt["valid"])
            self.assertEqual(train_receipt["metrics"], GOLDEN["CASCADE.009"]["metrics"])
            self.assertEqual(train_receipt["score"], GOLDEN["CASCADE.009"]["score"])
            self.assertEqual(train_receipt["unlocks"], ["case:CASCADE.009"])

            split_receipt = json.loads((author / "MERGE.005.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(split_receipt["valid"])
            self.assertEqual(split_receipt["metrics"], GOLDEN["MERGE.005"]["metrics"])
            self.assertEqual(split_receipt["score"], GOLDEN["MERGE.005"]["score"])
            self.assertEqual(split_receipt["unlocks"], ["case:MERGE.005"])

            quorum_receipt = json.loads((author / "PULSE.005.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(quorum_receipt["valid"])
            self.assertEqual(quorum_receipt["metrics"], GOLDEN["PULSE.005"]["metrics"])
            self.assertEqual(quorum_receipt["score"], GOLDEN["PULSE.005"]["score"])
            self.assertEqual(quorum_receipt["unlocks"], ["case:PULSE.005"])

            cooling_receipt = json.loads((author / "CASCADE.010.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(cooling_receipt["valid"])
            self.assertEqual(cooling_receipt["metrics"], GOLDEN["CASCADE.010"]["metrics"])
            self.assertEqual(cooling_receipt["score"], GOLDEN["CASCADE.010"]["score"])
            self.assertEqual(cooling_receipt["unlocks"], ["case:CASCADE.010"])

            ledger_receipt = json.loads((author / "MERGE.006.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(ledger_receipt["valid"])
            self.assertEqual(ledger_receipt["metrics"], GOLDEN["MERGE.006"]["metrics"])
            self.assertEqual(ledger_receipt["score"], GOLDEN["MERGE.006"]["score"])
            self.assertEqual(ledger_receipt["unlocks"], ["case:MERGE.006"])

            missing_receipt = json.loads((author / "MOSAIC.004.receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(missing_receipt["valid"])
            self.assertEqual(missing_receipt["metrics"], GOLDEN["MOSAIC.004"]["metrics"])
            self.assertEqual(missing_receipt["score"], GOLDEN["MOSAIC.004"]["score"])
            self.assertEqual(missing_receipt["unlocks"], ["case:MOSAIC.004"])

            for case_id in ("CASCADE.011", "MERGE.007", "MOSAIC.005", "PULSE.006", "CASCADE.012", "MERGE.008", "MOSAIC.006", "PULSE.007", "LENS.002", "CASCADE.013", "MERGE.009", "MOSAIC.007", "PULSE.008", "COVENANT.001", "CASCADE.014", "MERGE.010", "MOSAIC.008", "PULSE.009", "LENS.003", "CASCADE.015", "MERGE.011", "MOSAIC.009", "PULSE.010", "CASCADE.016", "CASCADE.017", "MERGE.012", "MOSAIC.010", "PULSE.011", "COVENANT.002", "MERGE.013", "MOSAIC.011", "PULSE.012", "CASCADE.018", "CASCADE.019", "MERGE.014", "PULSE.013", "CASCADE.020", "CASCADE.021", "CASCADE.022", "MERGE.015", "PULSE.014", "CASCADE.023", "CASCADE.024", "PARADOX.001"):
                new_receipt = json.loads((author / f"{case_id}.receipt.json").read_text(encoding="utf-8"))
                self.assertTrue(new_receipt["valid"], case_id)
                self.assertEqual(new_receipt["metrics"], GOLDEN[case_id]["metrics"], case_id)
                self.assertEqual(new_receipt["score"], GOLDEN[case_id]["score"], case_id)
                self.assertEqual(new_receipt["unlocks"], [f"case:{case_id}"], case_id)

            world_path = root / "world"
            kit.extract_bundle(kit.load_bundle(bundle), world_path)
            world = kit.verify_world(world_path)

            cascade_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "CASCADE.004")
            cascade_case = verifier.validate_case_descriptor(cascade_descriptor, world)
            cascade_logical = kit.resolve_logical_world(world, cascade_case["world"])
            occupancy = next(event for event in cascade_logical.base_events if event["topic"] == "platform.occupancy-sample")
            cascade_root = verifier.replay_root_case(world, cascade_case)
            self.assertFalse(cascade_root.records[0]["safe"])
            self.assertEqual(cascade_root.records[0]["temperature"], 11)

            def verify_cascade_operation(name: str, operation: dict[str, object]) -> dict[str, object]:
                parent = cre.root_branch_id(world.bundle)
                replay = verifier.replay_case(world, cascade_case, [operation], parent)
                intervention = {
                    "format": "afterimage-intervention/0.1", "bundle": world.bundle,
                    "parent_branch": parent, "case": cascade_case["id"], "operations": [operation],
                }
                candidate = {
                    "format": "afterimage-witness/0.1", "semantics": "cre/0.1", "bundle": world.bundle,
                    "case": cascade_case["id"], "parent_branch": parent, "intervention": intervention,
                    "answer": {"contracts": replay.records, "branch": replay.branch, "projection": replay.projection},
                    "claimed": {"branch": replay.branch, "projection": replay.projection, "trace": replay.trace},
                }
                path = root / f"cascade-004-{name}.json"
                path.write_bytes(cre.canonical_bytes(candidate))
                return verifier.verify_witness(world_path, path, {"case:CASCADE.003"})

            one_tick_late = verify_cascade_operation(
                "one-tick-late", {"kind": "retime", "event": occupancy["id"], "at": 10},
            )
            self.assertFalse(one_tick_late["valid"])
            self.assertEqual(one_tick_late["diagnostics"][0]["code"], "claim_mismatch")

            emergency_heat = verify_cascade_operation(
                "emergency-heat",
                {"kind": "inject", "topic": "platform.emergency-heat", "at": 11,
                 "payload": {"platform": "P-7", "energy": 10}, "parents": []},
            )
            self.assertTrue(emergency_heat["valid"])
            self.assertEqual(emergency_heat["metrics"]["intervention_weight"], 18)
            self.assertLess(emergency_heat["score"]["total"], cascade_receipt["score"]["total"])

            def verify_cascade_candidate(
                target_case: dict[str, object], operations: list[dict[str, object]],
                facts: set[str], name: str,
            ) -> dict[str, object]:
                parent = cre.root_branch_id(world.bundle)
                replay = verifier.replay_case(world, target_case, operations, parent)
                intervention = {
                    "format": "afterimage-intervention/0.1", "bundle": world.bundle,
                    "parent_branch": parent, "case": target_case["id"], "operations": operations,
                }
                candidate = {
                    "format": "afterimage-witness/0.1", "semantics": "cre/0.1", "bundle": world.bundle,
                    "case": target_case["id"], "parent_branch": parent, "intervention": intervention,
                    "answer": {"contracts": replay.records, "branch": replay.branch, "projection": replay.projection},
                    "claimed": {"branch": replay.branch, "projection": replay.projection, "trace": replay.trace},
                }
                path = root / f"{target_case['id'].lower()}-{name}.json"
                path.write_bytes(cre.canonical_bytes(candidate))
                return verifier.verify_witness(world_path, path, facts)

            queue_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "CASCADE.005")
            queue_case = verifier.validate_case_descriptor(queue_descriptor, world)
            queue_logical = kit.resolve_logical_world(world, queue_case["world"])
            maintenance_grant = next(event for event in queue_logical.base_events if event["topic"] == "queue.maintenance-grant")
            queue_root = verifier.replay_root_case(world, queue_case)
            self.assertFalse(queue_root.records[0]["safe"])
            self.assertEqual(queue_root.records[0]["public_first"], "medical")
            self.assertEqual(queue_root.records[0]["runtime_first"], "maintenance")
            self.assertEqual(queue_root.records[0]["medical_complete"], 17)

            equal_tick_grant = verify_cascade_candidate(
                queue_case, [{"kind": "retime", "event": maintenance_grant["id"], "at": 8}],
                {"case:CASCADE.004"}, "equal-tick-grant",
            )
            self.assertFalse(equal_tick_grant["valid"])
            self.assertEqual(equal_tick_grant["diagnostics"][0]["code"], "claim_mismatch")

            priority_boost = verify_cascade_candidate(
                queue_case,
                [{"kind": "inject", "topic": "queue.priority-boost", "at": 7,
                  "payload": {"queue": "Q-4"}, "parents": []}],
                {"case:CASCADE.004"}, "priority-boost",
            )
            self.assertTrue(priority_boost["valid"])
            self.assertEqual(priority_boost["metrics"]["intervention_weight"], 16)
            self.assertLess(priority_boost["score"]["total"], queue_receipt["score"]["total"])

            ice_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "CASCADE.006")
            ice_case = verifier.validate_case_descriptor(ice_descriptor, world)
            ice_logical = kit.resolve_logical_world(world, ice_case["world"])
            treatment = next(event for event in ice_logical.base_events if event["topic"] == "road.treatment")
            closure = next(event for event in ice_logical.base_events if event["topic"] == "road.closure")
            ice_root = verifier.replay_root_case(world, ice_case)
            self.assertFalse(ice_root.records[0]["safe"])
            self.assertEqual(ice_root.records[0]["treatment_effective_at"], 18)
            self.assertEqual(ice_root.records[0]["closure_at"], 13)

            treatment_only = verify_cascade_candidate(
                ice_case, [{"kind": "retime", "event": treatment["id"], "at": 11}],
                {"case:CASCADE.005"}, "treatment-only",
            )
            self.assertFalse(treatment_only["valid"])
            closure_only = verify_cascade_candidate(
                ice_case, [{"kind": "retime", "event": closure["id"], "at": 15}],
                {"case:CASCADE.005"}, "closure-only",
            )
            self.assertFalse(closure_only["valid"])
            late_treatment = verify_cascade_candidate(
                ice_case,
                [{"kind": "retime", "event": treatment["id"], "at": 12},
                 {"kind": "retime", "event": closure["id"], "at": 15}],
                {"case:CASCADE.005"}, "late-treatment",
            )
            self.assertFalse(late_treatment["valid"])

            valid_retime_pairs = []
            for treatment_at in range(11, 16):
                for closure_at in range(11, 16):
                    candidate = verify_cascade_candidate(
                        ice_case,
                        [{"kind": "retime", "event": treatment["id"], "at": treatment_at},
                         {"kind": "retime", "event": closure["id"], "at": closure_at}],
                        {"case:CASCADE.005"}, f"retime-{treatment_at}-{closure_at}",
                    )
                    if candidate["valid"]:
                        valid_retime_pairs.append((treatment_at, closure_at))
            self.assertEqual(valid_retime_pairs, [(11, 15)])

            mobile_barrier = verify_cascade_candidate(
                ice_case,
                [{"kind": "inject", "topic": "road.mobile-barrier", "at": 15,
                  "payload": {"road": "R-6"}, "parents": []}],
                {"case:CASCADE.005"}, "mobile-barrier",
            )
            self.assertTrue(mobile_barrier["valid"])
            self.assertEqual(mobile_barrier["metrics"]["intervention_weight"], 30)
            self.assertLess(mobile_barrier["score"]["total"], ice_receipt["score"]["total"])

            early_barrier = verify_cascade_candidate(
                ice_case,
                [{"kind": "inject", "topic": "road.mobile-barrier", "at": 14,
                  "payload": {"road": "R-6"}, "parents": []}],
                {"case:CASCADE.005"}, "early-barrier",
            )
            self.assertFalse(early_barrier["valid"])

            valid_barrier_times = []
            for barrier_at in range(11, 16):
                candidate = verify_cascade_candidate(
                    ice_case,
                    [{"kind": "inject", "topic": "road.mobile-barrier", "at": barrier_at,
                      "payload": {"road": "R-6"}, "parents": []}],
                    {"case:CASCADE.005"}, f"barrier-{barrier_at}",
                )
                if candidate["valid"]:
                    valid_barrier_times.append(barrier_at)
            self.assertEqual(valid_barrier_times, [15])

            coupling_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "CASCADE.007")
            coupling_case = verifier.validate_case_descriptor(coupling_descriptor, world)
            coupling_logical = kit.resolve_logical_world(world, coupling_case["world"])
            road_load = next(event for event in coupling_logical.base_events if event["topic"] == "power.road-treatment-load")
            coupling_root = verifier.replay_root_case(world, coupling_case)
            self.assertFalse(coupling_root.records[0]["safe"])
            self.assertEqual(coupling_root.records[0]["pressure"], 35)
            self.assertEqual(coupling_root.records[0]["road_authorization"], "grid-window-31")
            self.assertEqual(coupling_root.records[0]["traffic_authorization"], "traffic-priority-19")

            valid_coupling_retimes = []
            for road_at in range(11, 15):
                candidate = verify_cascade_candidate(
                    coupling_case,
                    [{"kind": "retime", "event": road_load["id"], "at": road_at}],
                    {"case:CASCADE.006"}, f"coupling-retime-{road_at}",
                )
                if candidate["valid"]:
                    valid_coupling_retimes.append(road_at)
            self.assertEqual(valid_coupling_retimes, [11])

            weak_generator = verify_cascade_candidate(
                coupling_case,
                [{"kind": "inject", "topic": "power.backup-generator", "at": 15,
                  "payload": {"feeder": "F-3", "power": 2}, "parents": []}],
                {"case:CASCADE.006"}, "weak-generator",
            )
            self.assertFalse(weak_generator["valid"])
            backup_generator = verify_cascade_candidate(
                coupling_case,
                [{"kind": "inject", "topic": "power.backup-generator", "at": 15,
                  "payload": {"feeder": "F-3", "power": 3}, "parents": []}],
                {"case:CASCADE.006"}, "backup-generator",
            )
            self.assertTrue(backup_generator["valid"])
            self.assertEqual(backup_generator["metrics"]["intervention_weight"], 24)
            self.assertLess(backup_generator["score"]["total"], coupling_receipt["score"]["total"])

            island_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "CASCADE.008")
            island_case = verifier.validate_case_descriptor(island_descriptor, world)
            island_logical = kit.resolve_logical_world(world, island_case["world"])
            signal_charge = next(event for event in island_logical.base_events if event["topic"] == "traffic.signal-charge")
            island_root = verifier.replay_root_case(world, island_case)
            self.assertFalse(island_root.records[0]["safe"])
            self.assertEqual(island_root.records[0]["signal_ready_at"], 17)
            self.assertEqual(island_root.records[0]["clinic_power_received"], 2)
            self.assertEqual(island_root.records[0]["capacity"], 6)

            valid_signal_retimes = []
            for charge_at in range(11, 15):
                candidate = verify_cascade_candidate(
                    island_case,
                    [{"kind": "retime", "event": signal_charge["id"], "at": charge_at}],
                    {"case:CASCADE.007"}, f"island-retime-{charge_at}",
                )
                if candidate["valid"]:
                    valid_signal_retimes.append(charge_at)
            self.assertEqual(valid_signal_retimes, [11])

            mobile_signal = verify_cascade_candidate(
                island_case,
                [{"kind": "inject", "topic": "traffic.mobile-signal", "at": 14,
                  "payload": {"intersection": "I-12", "self_powered": True}, "parents": []}],
                {"case:CASCADE.007"}, "island-mobile-signal",
            )
            self.assertTrue(mobile_signal["valid"])
            self.assertEqual(mobile_signal["metrics"]["intervention_weight"], 22)
            self.assertLess(mobile_signal["score"]["total"], island_receipt["score"]["total"])

            late_mobile_signal = verify_cascade_candidate(
                island_case,
                [{"kind": "inject", "topic": "traffic.mobile-signal", "at": 15,
                  "payload": {"intersection": "I-12", "self_powered": True}, "parents": []}],
                {"case:CASCADE.007"}, "island-late-mobile-signal",
            )
            self.assertFalse(late_mobile_signal["valid"])
            self.assertEqual(late_mobile_signal["diagnostics"][0]["code"], "claim_mismatch")

            train_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "CASCADE.009")
            train_case = verifier.validate_case_descriptor(train_descriptor, world)
            train_logical = kit.resolve_logical_world(world, train_case["world"])
            ventilation = next(event for event in train_logical.base_events if event["topic"] == "tunnel.ventilation")
            last_train = next(event for event in train_logical.base_events if event["topic"] == "transit.last-train")
            train_root = verifier.replay_root_case(world, train_case)
            self.assertFalse(train_root.records[0]["safe"])
            self.assertEqual(train_root.records[0]["departure_at"], 19)
            self.assertEqual(train_root.records[0]["traction_expires_at"], 17)
            self.assertEqual(train_root.records[0]["arrival_at"], 21)
            self.assertEqual(train_root.records[0]["bridge_deadline"], 20)

            valid_train_pairs = []
            for ventilation_at in range(12, 20):
                for departure_at in range(12, 20):
                    candidate = verify_cascade_candidate(
                        train_case,
                        [
                            {"kind": "retime", "event": ventilation["id"], "at": ventilation_at},
                            {"kind": "retime", "event": last_train["id"], "at": departure_at},
                        ],
                        {"case:CASCADE.008"}, f"train-{ventilation_at}-{departure_at}",
                    )
                    if candidate["valid"]:
                        valid_train_pairs.append((ventilation_at, departure_at))
            self.assertEqual(valid_train_pairs, [(13, 17)])

            for single_event, at in ((ventilation, 13), (last_train, 17)):
                candidate = verify_cascade_candidate(
                    train_case, [{"kind": "retime", "event": single_event["id"], "at": at}],
                    {"case:CASCADE.008"}, f"train-single-{single_event['topic']}",
                )
                self.assertFalse(candidate["valid"])

            bus_bridge = verify_cascade_candidate(
                train_case,
                [{"kind": "inject", "topic": "transit.bus-bridge", "at": 17,
                  "payload": {"station": "Harbor-4", "self_powered": True}, "parents": []}],
                {"case:CASCADE.008"}, "train-bus-bridge",
            )
            self.assertTrue(bus_bridge["valid"])
            self.assertEqual(bus_bridge["metrics"]["intervention_weight"], 40)
            self.assertLess(bus_bridge["score"]["total"], train_receipt["score"]["total"])

            for bus_at in (16, 18):
                candidate = verify_cascade_candidate(
                    train_case,
                    [{"kind": "inject", "topic": "transit.bus-bridge", "at": bus_at,
                      "payload": {"station": "Harbor-4", "self_powered": True}, "parents": []}],
                    {"case:CASCADE.008"}, f"train-bus-{bus_at}",
                )
                self.assertFalse(candidate["valid"])

            cooling_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "CASCADE.010")
            cooling_case = verifier.validate_case_descriptor(cooling_descriptor, world)
            cooling_logical = kit.resolve_logical_world(world, cooling_case["world"])
            cooling_command = next(event for event in cooling_logical.base_events if event["topic"] == "hospital.cooling-command")
            cooling_root = verifier.replay_root_case(world, cooling_case)
            self.assertFalse(cooling_root.records[0]["safe"])
            self.assertEqual(cooling_root.records[0]["flow"], 3)
            self.assertEqual(cooling_root.records[0]["temperature"], 12)
            self.assertEqual(cooling_root.records[0]["active_load"], 7)

            valid_cooling_flows = []
            for flow in range(11):
                candidate = verify_cascade_candidate(
                    cooling_case,
                    [{"kind": "replace", "event": cooling_command["id"],
                      "pointer": "/payload/flow", "value": flow}],
                    {"case:CASCADE.009"}, f"cooling-flow-{flow}",
                )
                if candidate["valid"]:
                    valid_cooling_flows.append(flow)
            self.assertEqual(valid_cooling_flows, [5])

            portable_chiller = verify_cascade_candidate(
                cooling_case,
                [{"kind": "inject", "topic": "hospital.portable-chiller", "at": 18,
                  "payload": {"ward": "surgery", "self_powered": True}, "parents": []}],
                {"case:CASCADE.009"}, "cooling-portable",
            )
            self.assertTrue(portable_chiller["valid"])
            self.assertEqual(portable_chiller["metrics"]["intervention_weight"], 35)
            self.assertLess(portable_chiller["score"]["total"], cooling_receipt["score"]["total"])

            for portable_at in (17, 19):
                candidate = verify_cascade_candidate(
                    cooling_case,
                    [{"kind": "inject", "topic": "hospital.portable-chiller", "at": portable_at,
                      "payload": {"ward": "surgery", "self_powered": True}, "parents": []}],
                    {"case:CASCADE.009"}, f"cooling-portable-{portable_at}",
                )
                self.assertFalse(candidate["valid"])

            missing_ack_path = author / "MERGE.003.witness.json"
            missing_ack_witness = cre.load_json(missing_ack_path)
            merge3_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "MERGE.003")
            merge3_case = verifier.validate_case_descriptor(merge3_descriptor, world)
            merge3_logical = kit.resolve_logical_world(world, merge3_case["world"])
            merge3_by_key = {event["payload"]["key"]: event["id"] for event in merge3_logical.base_events}

            def verify_merge3(name: str, mutate: object) -> dict[str, object]:
                altered = copy.deepcopy(missing_ack_witness)
                mutate(altered["answer"])
                altered.pop("meta", None)
                path = root / f"merge-003-{name}.json"
                path.write_bytes(cre.canonical_bytes(altered))
                return verifier.verify_witness(world_path, path, {"case:MERGE.002"})

            def remove_bridge(answer: dict[str, object]) -> None:
                answer["certificate"] = [
                    edge for edge in answer["certificate"]
                    if not (edge["before"] == merge3_by_key["G1"] and edge["after"] == merge3_by_key["G3"])
                ]

            no_bridge = verify_merge3("no-bridge", remove_bridge)
            self.assertFalse(no_bridge["valid"])
            self.assertEqual(no_bridge["diagnostics"][0]["code"], "merge_certificate")

            def add_phantom_edge(answer: dict[str, object]) -> None:
                answer["certificate"].append({
                    "before": merge3_by_key["G1"], "after": merge3_by_key["W1"], "minimum_gap": 1,
                })

            phantom_edge = verify_merge3("phantom-edge", add_phantom_edge)
            self.assertFalse(phantom_edge["valid"])
            self.assertEqual(phantom_edge["diagnostics"][0]["code"], "merge_certificate")

            def shift_gateway(answer: dict[str, object]) -> None:
                next(item for item in answer["accepted"] if item["event"] == merge3_by_key["G3"])["at"] = 13

            displaced = verify_merge3("displaced", shift_gateway)
            self.assertTrue(displaced["valid"])
            self.assertEqual(displaced["metrics"]["temporal_displacement"], 1)
            self.assertGreater(displaced["metrics"]["effective_cost"], missing_ack_receipt["metrics"]["effective_cost"])
            self.assertEqual(displaced["score"]["total"], missing_ack_receipt["score"]["total"])

            compensation_path = author / "MERGE.004.witness.json"
            compensation_witness = cre.load_json(compensation_path)
            merge4_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "MERGE.004")
            merge4_case = verifier.validate_case_descriptor(merge4_descriptor, world)
            merge4_logical = kit.resolve_logical_world(world, merge4_case["world"])
            merge4_by_key = {event["payload"]["key"]: event["id"] for event in merge4_logical.base_events}

            def verify_merge4(name: str, mutate: object) -> dict[str, object]:
                altered = copy.deepcopy(compensation_witness)
                mutate(altered["answer"])
                altered.pop("meta", None)
                path = root / f"merge-004-{name}.json"
                path.write_bytes(cre.canonical_bytes(altered))
                return verifier.verify_witness(world_path, path, {"case:MERGE.003"})

            def accept_rollback(answer: dict[str, object]) -> None:
                answer["rejected"].clear()
                answer["accepted"].append({"event": merge4_by_key["B1"], "at": 9})
                answer["certificate"].append({
                    "before": merge4_by_key["R1"], "after": merge4_by_key["B1"], "minimum_gap": 1,
                })

            rollback_archive = verify_merge4("accept-rollback", accept_rollback)
            self.assertFalse(rollback_archive["valid"])
            self.assertEqual(rollback_archive["diagnostics"][0]["code"], "merge_order")

            def erase_record(answer: dict[str, object], key: str) -> None:
                event_id = merge4_by_key[key]
                answer["accepted"] = [item for item in answer["accepted"] if item["event"] != event_id]
                answer["rejected"].append({"event": event_id, "reason": "inconsistent"})
                answer["certificate"] = [
                    edge for edge in answer["certificate"]
                    if event_id not in (edge["before"], edge["after"])
                ]

            erased_transfer = verify_merge4("erase-transfer", lambda answer: erase_record(answer, "T1"))
            self.assertFalse(erased_transfer["valid"])
            self.assertEqual(erased_transfer["diagnostics"][0]["code"], "merge_rejection")

            erased_compensation = verify_merge4("erase-compensation", lambda answer: erase_record(answer, "C1"))
            self.assertFalse(erased_compensation["valid"])
            self.assertEqual(erased_compensation["diagnostics"][0]["code"], "merge_rejection")

            def shift_compensation(answer: dict[str, object]) -> None:
                next(item for item in answer["accepted"] if item["event"] == merge4_by_key["C1"])["at"] = 14

            displaced_compensation = verify_merge4("displaced-compensation", shift_compensation)
            self.assertTrue(displaced_compensation["valid"])
            self.assertEqual(displaced_compensation["metrics"]["temporal_displacement"], 1)
            self.assertGreater(displaced_compensation["metrics"]["effective_cost"], compensation_receipt["metrics"]["effective_cost"])

            split_path = author / "MERGE.005.witness.json"
            split_witness = cre.load_json(split_path)
            merge5_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "MERGE.005")
            merge5_case = verifier.validate_case_descriptor(merge5_descriptor, world)
            merge5_logical = kit.resolve_logical_world(world, merge5_case["world"])
            merge5_by_key = {event["payload"]["key"]: event["id"] for event in merge5_logical.base_events}

            def verify_merge5(name: str, mutate: object) -> dict[str, object]:
                altered = copy.deepcopy(split_witness)
                mutate(altered["answer"])
                altered.pop("meta", None)
                path = root / f"merge-005-{name}.json"
                path.write_bytes(cre.canonical_bytes(altered))
                return verifier.verify_witness(world_path, path, {"case:MERGE.004"})

            def select_west(answer: dict[str, object]) -> None:
                answer["accepted"] = [
                    {"event": merge5_by_key["W1"], "at": 8},
                    {"event": merge5_by_key["W2"], "at": 10},
                ]
                answer["rejected"] = [
                    {"event": merge5_by_key[key], "reason": "conflict_set"}
                    for key in ("E1", "E2", "E3")
                ]
                answer["certificate"] = [{
                    "before": merge5_by_key["W1"], "after": merge5_by_key["W2"], "minimum_gap": 1,
                }]

            west_archive = verify_merge5("west-archive", select_west)
            self.assertTrue(west_archive["valid"])
            self.assertEqual(west_archive["metrics"]["rejected_weight"], 25)
            self.assertLess(west_archive["score"]["total"], split_receipt["score"]["total"])

            def splice_writers(answer: dict[str, object]) -> None:
                answer["rejected"] = [
                    item for item in answer["rejected"] if item["event"] != merge5_by_key["W2"]
                ]
                answer["accepted"].append({"event": merge5_by_key["W2"], "at": 10})
                answer["certificate"].append({
                    "before": merge5_by_key["E2"], "after": merge5_by_key["W2"], "minimum_gap": 1,
                })

            spliced_archive = verify_merge5("spliced-writers", splice_writers)
            self.assertFalse(spliced_archive["valid"])
            self.assertEqual(spliced_archive["diagnostics"][0]["code"], "merge_split")

            def relabel_conflict(answer: dict[str, object]) -> None:
                for item in answer["rejected"]:
                    item["reason"] = "inconsistent"

            relabeled = verify_merge5("relabel-conflict", relabel_conflict)
            self.assertFalse(relabeled["valid"])
            self.assertEqual(relabeled["diagnostics"][0]["code"], "merge_split")

            def drop_branch_tail(answer: dict[str, object]) -> None:
                event_id = merge5_by_key["E3"]
                answer["accepted"] = [item for item in answer["accepted"] if item["event"] != event_id]
                answer["rejected"].append({"event": event_id, "reason": "conflict_set"})
                answer["certificate"] = [
                    edge for edge in answer["certificate"]
                    if event_id not in (edge["before"], edge["after"])
                ]

            partial_writer = verify_merge5("partial-writer", drop_branch_tail)
            self.assertFalse(partial_writer["valid"])
            self.assertEqual(partial_writer["diagnostics"][0]["code"], "merge_split")

            ledger_path = author / "MERGE.006.witness.json"
            ledger_witness = cre.load_json(ledger_path)
            merge6_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "MERGE.006")
            merge6_case = verifier.validate_case_descriptor(merge6_descriptor, world)
            merge6_logical = kit.resolve_logical_world(world, merge6_case["world"])
            merge6_by_key = {event["payload"]["key"]: event["id"] for event in merge6_logical.base_events}
            merge6_facts = {"case:MERGE.005", "case:PULSE.005", "case:CASCADE.010"}

            def verify_merge6(name: str, mutate: object) -> dict[str, object]:
                altered = copy.deepcopy(ledger_witness)
                mutate(altered["answer"])
                altered.pop("meta", None)
                path = root / f"merge-006-{name}.json"
                path.write_bytes(cre.canonical_bytes(altered))
                return verifier.verify_witness(world_path, path, merge6_facts)

            def relabel_minority(answer: dict[str, object]) -> None:
                for item in answer["rejected"]:
                    item["reason"] = "inconsistent"

            false_clock_fault = verify_merge6("false-clock-fault", relabel_minority)
            self.assertFalse(false_clock_fault["valid"])
            self.assertEqual(false_clock_fault["diagnostics"][0]["code"], "merge_quorum")

            def accept_every_claim(answer: dict[str, object]) -> None:
                answer["accepted"].extend([
                    {"event": merge6_by_key["N1"], "at": 6},
                    {"event": merge6_by_key["N2"], "at": 8},
                ])
                answer["rejected"].clear()

            mixed_claims = verify_merge6("mixed-claims", accept_every_claim)
            self.assertFalse(mixed_claims["valid"])
            self.assertEqual(mixed_claims["diagnostics"][0]["code"], "merge_quorum")

            def select_two_vote_minority(answer: dict[str, object]) -> None:
                answer["accepted"] = [
                    {"event": merge6_by_key["N1"], "at": 6},
                    {"event": merge6_by_key["N2"], "at": 8},
                ]
                answer["rejected"] = [
                    {"event": merge6_by_key[key], "reason": "minority"}
                    for key in ("Q1", "Q2", "Q3")
                ]
                answer["certificate"] = []

            two_vote_archive = verify_merge6("two-vote-archive", select_two_vote_minority)
            self.assertFalse(two_vote_archive["valid"])
            self.assertEqual(two_vote_archive["diagnostics"][0]["code"], "merge_quorum")

            def drop_quorum_vote(answer: dict[str, object]) -> None:
                event_id = merge6_by_key["Q3"]
                answer["accepted"] = [item for item in answer["accepted"] if item["event"] != event_id]
                answer["rejected"].append({"event": event_id, "reason": "minority"})
                answer["certificate"] = [
                    edge for edge in answer["certificate"]
                    if event_id not in (edge["before"], edge["after"])
                ]

            incomplete_quorum = verify_merge6("incomplete-quorum", drop_quorum_vote)
            self.assertFalse(incomplete_quorum["valid"])
            self.assertEqual(incomplete_quorum["diagnostics"][0]["code"], "merge_quorum")

            rotated_path = author / "MOSAIC.003.witness.json"
            rotated_witness = cre.load_json(rotated_path)
            coordinate_transforms = {
                "r90": lambda x, y: (2 - y, x),
                "r180": lambda x, y: (2 - x, 2 - y),
                "r270": lambda x, y: (y, 2 - x),
                "mx": lambda x, y: (2 - x, y),
                "mxr90": lambda x, y: (2 - y, 2 - x),
                "mxr180": lambda x, y: (x, 2 - y),
                "mxr270": lambda x, y: (y, x),
            }

            for transform_name, transform in coordinate_transforms.items():
                altered = copy.deepcopy(rotated_witness)
                transformed_edges = []
                for edge in altered["answer"]["global"]["edges"]:
                    a_index = int(edge["a"][1:])
                    b_index = int(edge["b"][1:])
                    ax, ay = transform(a_index % 3, a_index // 3)
                    bx, by = transform(b_index % 3, b_index // 3)
                    transformed_edges.append({
                        **edge, "a": f"v{ay * 3 + ax}", "b": f"v{by * 3 + bx}",
                    })
                altered["answer"]["global"]["edges"] = transformed_edges
                altered.pop("meta", None)
                path = root / f"mosaic-003-{transform_name}.json"
                path.write_bytes(cre.canonical_bytes(altered))
                rejected = verifier.verify_witness(world_path, path, {"case:MOSAIC.002"})
                self.assertFalse(rejected["valid"], transform_name)
                self.assertEqual(rejected["diagnostics"][0]["code"], "mosaic_noncanonical", transform_name)

            wrong_transform = copy.deepcopy(rotated_witness)
            wrong_transform["answer"]["used"][0]["transform"] = "r270"
            wrong_transform.pop("meta", None)
            wrong_transform_path = root / "mosaic-003-wrong-transform.json"
            wrong_transform_path.write_bytes(cre.canonical_bytes(wrong_transform))
            wrong_transform_receipt = verifier.verify_witness(world_path, wrong_transform_path, {"case:MOSAIC.002"})
            self.assertFalse(wrong_transform_receipt["valid"])
            self.assertEqual(wrong_transform_receipt["diagnostics"][0]["code"], "mosaic_mapping")

            missing_path = author / "MOSAIC.004.witness.json"
            missing_witness = cre.load_json(missing_path)
            missing_facts = {"case:MOSAIC.003", "case:CASCADE.010"}

            def verify_missing(name: str, mutate: object) -> dict[str, object]:
                altered = copy.deepcopy(missing_witness)
                mutate(altered["answer"])
                altered.pop("meta", None)
                path = root / f"mosaic-004-{name}.json"
                path.write_bytes(cre.canonical_bytes(altered))
                return verifier.verify_witness(world_path, path, missing_facts)

            def corrupt_inferred_edge(answer: dict[str, object]) -> None:
                edge = next(
                    item for item in answer["global"]["edges"]
                    if {item["a"], item["b"]} == {"v1", "v4"}
                )
                edge["material"] = "plausible-but-invented"

            invented_attribute = verify_missing("invented-attribute", corrupt_inferred_edge)
            self.assertFalse(invented_attribute["valid"])
            self.assertEqual(invented_attribute["diagnostics"][0]["code"], "mosaic_checksum")

            wrong_hole = verify_missing(
                "wrong-hole", lambda answer: answer["missing"].update({"vertices": ["v0"]}),
            )
            self.assertFalse(wrong_hole["valid"])
            self.assertEqual(wrong_hole["diagnostics"][0]["code"], "mosaic_omission")

            bridge_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "CASCADE.011")
            bridge_case = verifier.validate_case_descriptor(bridge_descriptor, world)
            bridge_logical = kit.resolve_logical_world(world, bridge_case["world"])
            bridge_command = next(event for event in bridge_logical.base_events if event["topic"] == "bridge.open-command")
            bridge_root = verifier.replay_root_case(world, bridge_case)
            self.assertFalse(bridge_root.records[0]["safe"])
            self.assertEqual(bridge_root.records[0]["command_at"], 6)
            valid_bridge_retimes = []
            for at in range(6, 10):
                candidate = verify_cascade_candidate(
                    bridge_case, [{"kind": "retime", "event": bridge_command["id"], "at": at}],
                    {"case:CASCADE.010"}, f"bridge-retime-{at}",
                )
                if candidate["valid"]:
                    valid_bridge_retimes.append(at)
            self.assertEqual(valid_bridge_retimes, [8])
            for at, expected in ((9, False), (10, True), (11, False)):
                candidate = verify_cascade_candidate(
                    bridge_case,
                    [{"kind": "inject", "topic": "bridge.emergency-ferry", "at": at,
                      "payload": {"bridge": "B-17"}, "parents": []}],
                    {"case:CASCADE.010"}, f"ferry-{at}",
                )
                self.assertEqual(candidate["valid"], expected, at)
                if expected:
                    self.assertEqual(candidate["metrics"]["intervention_weight"], 30)
                    self.assertLess(candidate["score"]["total"], GOLDEN["CASCADE.011"]["score"]["total"])

            clock_path = author / "MERGE.007.witness.json"
            clock_witness = cre.load_json(clock_path)
            merge7_descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "MERGE.007")
            merge7_case = verifier.validate_case_descriptor(merge7_descriptor, world)
            merge7_logical = kit.resolve_logical_world(world, merge7_case["world"])
            merge7_by_key = {event["payload"]["key"]: event["id"] for event in merge7_logical.base_events}
            merge7_facts = {"case:MERGE.006", "case:CASCADE.011"}

            def verify_merge7(name: str, mutate: object) -> dict[str, object]:
                altered = copy.deepcopy(clock_witness)
                mutate(altered["answer"])
                altered.pop("meta", None)
                path = root / f"merge-007-{name}.json"
                path.write_bytes(cre.canonical_bytes(altered))
                return verifier.verify_witness(world_path, path, merge7_facts)

            def drift_one_record(answer: dict[str, object]) -> None:
                next(item for item in answer["accepted"] if item["event"] == merge7_by_key["N2"])["at"] = 6

            drifting_clock = verify_merge7("one-record-drift", drift_one_record)
            self.assertFalse(drifting_clock["valid"])
            self.assertEqual(drifting_clock["diagnostics"][0]["code"], "merge_domain")

            def move_both_domains(answer: dict[str, object]) -> None:
                times = {"N1": 3, "S1": 4, "N2": 6, "S2": 7, "N3": 9, "S3": 10}
                by_event = {merge7_by_key[key]: at for key, at in times.items()}
                for item in answer["accepted"]:
                    item["at"] = by_event[item["event"]]

            alternate_domains = verify_merge7("alternate-domains", move_both_domains)
            self.assertTrue(alternate_domains["valid"])
            self.assertGreater(alternate_domains["metrics"]["temporal_displacement"], 0)
            self.assertLess(alternate_domains["score"]["total"], GOLDEN["MERGE.007"]["score"]["total"])

            def reject_domain_record(answer: dict[str, object]) -> None:
                event_id = merge7_by_key["N3"]
                answer["accepted"] = [item for item in answer["accepted"] if item["event"] != event_id]
                answer["rejected"].append({"event": event_id, "reason": "inconsistent"})
                answer["certificate"] = [edge for edge in answer["certificate"] if event_id not in (edge["before"], edge["after"])]

            incomplete_domain = verify_merge7("incomplete-domain", reject_domain_record)
            self.assertFalse(incomplete_domain["valid"])
            self.assertEqual(incomplete_domain["diagnostics"][0]["code"], "merge_domain")

            duplicate_path = author / "MOSAIC.005.witness.json"
            duplicate_witness = cre.load_json(duplicate_path)
            duplicate_facts = {"case:MOSAIC.004", "case:CASCADE.011", "case:MERGE.007"}

            def verify_duplicate(name: str, mutate: object) -> dict[str, object]:
                altered = copy.deepcopy(duplicate_witness)
                mutate(altered["answer"])
                altered.pop("meta", None)
                path = root / f"mosaic-005-{name}.json"
                path.write_bytes(cre.canonical_bytes(altered))
                return verifier.verify_witness(world_path, path, duplicate_facts)

            wrong_duplicate_link = verify_duplicate(
                "wrong-link", lambda answer: answer["unused"][0].update({"duplicate_of": "F-BRIDGE-SOUTH"}),
            )
            self.assertFalse(wrong_duplicate_link["valid"])
            self.assertEqual(wrong_duplicate_link["diagnostics"][0]["code"], "mosaic_duplicate")

            def keep_cached_copy(answer: dict[str, object]) -> None:
                answer["used"][0] = {
                    "fragment": "F-BRIDGE-NORTH-CACHE", "transform": "r180",
                    "mapping": [
                        {"local": "d00", "global": "v5"}, {"local": "d10", "global": "v4"},
                        {"local": "d20", "global": "v3"}, {"local": "d01", "global": "v2"},
                        {"local": "d11", "global": "v1"}, {"local": "d21", "global": "v0"},
                    ],
                }
                answer["unused"][0] = {"fragment": "F-BRIDGE-NORTH", "reason": "duplicate", "duplicate_of": "F-BRIDGE-NORTH-CACHE"}

            lower_evidence = verify_duplicate("keep-cache", keep_cached_copy)
            self.assertTrue(lower_evidence["valid"])
            self.assertEqual(lower_evidence["metrics"]["unexplained_weight"], 8)
            self.assertLess(lower_evidence["score"]["total"], GOLDEN["MOSAIC.005"]["score"]["total"])

            descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "MERGE.002")
            case = verifier.validate_case_descriptor(descriptor, world)
            logical = kit.resolve_logical_world(world, case["world"])
            by_key = {event["payload"]["key"]: event["id"] for event in logical.base_events}
            witness_path = author / "MERGE.002.witness.json"
            witness = cre.load_json(witness_path)
            self.assertEqual(witness["answer"]["rejected"], [{
                "event": by_key["D1R"], "reason": "duplicate", "duplicate_of": by_key["D1"],
            }])

            def verify_mutation(name: str, mutate: object) -> dict[str, object]:
                altered = copy.deepcopy(witness)
                mutate(altered["answer"])
                altered.pop("meta", None)
                path = root / f"merge-002-{name}.json"
                path.write_bytes(cre.canonical_bytes(altered))
                return verifier.verify_witness(world_path, path, {"case:MERGE.001"})

            wrong_link = verify_mutation(
                "wrong-link",
                lambda answer: answer["rejected"][0].update({"duplicate_of": by_key["A1"]}),
            )
            self.assertFalse(wrong_link["valid"])
            self.assertEqual(wrong_link["diagnostics"][0]["code"], "merge_duplicate")

            def accept_both(answer: dict[str, object]) -> None:
                answer["rejected"].clear()
                answer["accepted"].append({"event": by_key["D1R"], "at": 10})
                answer["certificate"].extend([
                    {"before": by_key["I1"], "after": by_key["D1R"], "minimum_gap": 1},
                    {"before": by_key["D1R"], "after": by_key["A1"], "minimum_gap": 1},
                ])

            double_survivor = verify_mutation("double-survivor", accept_both)
            self.assertFalse(double_survivor["valid"])
            self.assertEqual(double_survivor["diagnostics"][0]["code"], "merge_duplicate")

            def keep_retry(answer: dict[str, object]) -> None:
                for item in answer["accepted"]:
                    if item["event"] == by_key["D1"]:
                        item["event"] = by_key["D1R"]
                answer["rejected"][0] = {"event": by_key["D1"], "reason": "duplicate", "duplicate_of": by_key["D1R"]}
                for edge in answer["certificate"]:
                    if edge["before"] == by_key["D1"]:
                        edge["before"] = by_key["D1R"]
                    if edge["after"] == by_key["D1"]:
                        edge["after"] = by_key["D1R"]

            lower_quality = verify_mutation("keep-retry", keep_retry)
            self.assertTrue(lower_quality["valid"])
            self.assertEqual(lower_quality["metrics"]["rejected_weight"], 5)
            self.assertLess(lower_quality["score"]["total"], merge_receipt["score"]["total"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
