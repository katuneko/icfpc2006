#!/usr/bin/env python3
"""Golden production test for the authored vertical-slice content."""

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
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import build_slice  # noqa: E402
import cre  # noqa: E402
import verify_witness as verifier  # noqa: E402


GOLDEN = json.loads((ROOT / "content/vertical_slice/golden.json").read_text(encoding="utf-8"))


class SliceContentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="afterimage-slice-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self, stem: str) -> tuple[Path, Path, dict[str, object]]:
        bundle = self.root / f"{stem}.afterimage"
        author = self.root / f"{stem}-author"
        command = [
            sys.executable,
            str(ROOT / "tools/build_slice.py"),
            str(bundle),
            "--author-dir",
            str(author),
        ]
        completed = subprocess.run(command, check=True, capture_output=True)
        return bundle, author, json.loads(completed.stdout)

    def test_reproducible_bundle_matches_golden_and_excludes_author_secrets(self) -> None:
        first, first_author, summary = self.build("first")
        second, second_author, second_summary = self.build("second")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(summary["bundle"], second_summary["bundle"])
        self.assertEqual(summary["archive_sha256"], GOLDEN["archive_sha256"])
        self.assertEqual(summary["bundle"], GOLDEN["bundle"])
        for key, value in GOLDEN["counts"].items():
            self.assertEqual(summary[key], value)
        self.assertEqual(
            (first_author / "ORIENT.001.witness.json").read_bytes(),
            (second_author / "ORIENT.001.witness.json").read_bytes(),
        )
        self.assertEqual(
            (first_author / "ORIENT.002.witness.json").read_bytes(),
            (second_author / "ORIENT.002.witness.json").read_bytes(),
        )
        self.assertEqual(
            (first_author / "ORIENT.003.witness.json").read_bytes(),
            (second_author / "ORIENT.003.witness.json").read_bytes(),
        )
        self.assertEqual(
            (first_author / "ORIENT.004.witness.json").read_bytes(),
            (second_author / "ORIENT.004.witness.json").read_bytes(),
        )
        self.assertEqual(
            (first_author / "ORIENT.005.witness.json").read_bytes(),
            (second_author / "ORIENT.005.witness.json").read_bytes(),
        )
        self.assertEqual(
            (first_author / "CASCADE.001.witness.json").read_bytes(),
            (second_author / "CASCADE.001.witness.json").read_bytes(),
        )
        self.assertEqual(
            (first_author / "CASCADE.002.witness.json").read_bytes(),
            (second_author / "CASCADE.002.witness.json").read_bytes(),
        )
        self.assertEqual(
            (first_author / "CASCADE.003.witness.json").read_bytes(),
            (second_author / "CASCADE.003.witness.json").read_bytes(),
        )
        self.assertEqual(
            (first_author / "MERGE.001.witness.json").read_bytes(),
            (second_author / "MERGE.001.witness.json").read_bytes(),
        )
        self.assertEqual(
            (first_author / "PULSE.001.witness.json").read_bytes(),
            (second_author / "PULSE.001.witness.json").read_bytes(),
        )
        self.assertEqual(
            (first_author / "MOSAIC.001.witness.json").read_bytes(),
            (second_author / "MOSAIC.001.witness.json").read_bytes(),
        )
        self.assertEqual(
            (first_author / "LENS.001.witness.json").read_bytes(),
            (second_author / "LENS.001.witness.json").read_bytes(),
        )
        with zipfile.ZipFile(first, "r") as archive:
            names = set(archive.namelist())
            self.assertFalse(any("witness" in name or "receipt" in name for name in names))
            story = archive.read("story/ORIENT.001.md").decode("utf-8")
            self.assertNotIn(GOLDEN["ORIENT.001"]["answer"]["event_id"], story)
            branch_story = archive.read("story/ORIENT.004.md").decode("utf-8")
            self.assertNotIn(GOLDEN["ORIENT.004"]["answer"]["changed_event_ids"][0], branch_story)

    def test_author_witness_receipt_and_four_claim_fields(self) -> None:
        bundle_path, author, _summary = self.build("claims")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        witness_path = author / "ORIENT.001.witness.json"
        witness = cre.load_json(witness_path)
        receipt = verifier.verify_witness(world_path, witness_path)
        self.assertTrue(receipt["valid"])
        self.assertEqual(witness["answer"], GOLDEN["ORIENT.001"]["answer"])
        self.assertEqual(receipt["branch"], GOLDEN["ORIENT.001"]["branch"])
        self.assertEqual(receipt["trace"], GOLDEN["ORIENT.001"]["trace"])
        self.assertEqual(receipt["score"], GOLDEN["ORIENT.001"]["score"])

        mutations = {
            "event_id": "sha256:" + "0" * 64,
            "topic": "alarm.wrong",
            "at": 13,
            "projection": "sha256:" + "0" * 64,
        }
        for field, replacement in mutations.items():
            altered = copy.deepcopy(witness)
            altered["answer"][field] = replacement
            altered.pop("meta", None)
            path = self.root / f"wrong-{field}.json"
            path.write_bytes(cre.canonical_bytes(altered))
            invalid = verifier.verify_witness(world_path, path)
            with self.subTest(field=field):
                self.assertFalse(invalid["valid"])
                self.assertEqual(invalid["diagnostics"][0]["code"], "claim_mismatch")
                self.assertNotIn("score", invalid)
                self.assertNotIn("unlocks", invalid)

    def test_world_has_three_base_events_one_derived_alarm_and_four_trace_items(self) -> None:
        bundle_path, _author, _summary = self.build("world")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        world = kit.verify_world(world_path)
        descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "ORIENT.001")
        case = verifier.validate_case_descriptor(descriptor, world)
        logical = kit.resolve_logical_world(world, case["world"])
        replay = verifier.replay_case(world, case, [], cre.root_branch_id(world.bundle))
        alarms = [event for event in replay.events.values() if event["topic"] == "alarm.test"]
        self.assertEqual(len(logical.base_events), 3)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(len(replay.events), 4)
        self.assertEqual(alarms[0]["parents"], sorted(alarms[0]["parents"], key=cre.parse_id))
        self.assertEqual(len(alarms[0]["parents"]), 3)
        self.assertEqual(replay.records, [{
            "event": GOLDEN["ORIENT.001"]["answer"]["event_id"],
            "topic": "alarm.test",
            "at": 12,
            "signal": "bell-17",
        }])

    def test_orient_002_audits_parent_tie_break_and_fixed_point(self) -> None:
        bundle_path, author, _summary = self.build("replay-audit")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        world = kit.verify_world(world_path)
        descriptors = {item["id"]: item for item in world.json_values["cases/index.json"]["cases"]}
        case = verifier.validate_case_descriptor(descriptors["ORIENT.002"], world)
        witness_path = author / "ORIENT.002.witness.json"
        witness = cre.load_json(witness_path)

        locked = verifier.verify_witness(world_path, witness_path)
        self.assertFalse(locked["valid"])
        self.assertEqual(locked["diagnostics"][0]["code"], "case_locked")
        receipt = verifier.verify_witness(world_path, witness_path, {"case:ORIENT.001"})
        self.assertTrue(receipt["valid"])
        self.assertEqual(witness["answer"]["records"], GOLDEN["ORIENT.002"]["records"])
        self.assertEqual(witness["answer"]["projection"], GOLDEN["ORIENT.002"]["projection"])
        self.assertEqual(len(witness["answer"]["trace_event_ids"]), GOLDEN["ORIENT.002"]["trace_event_count"])
        self.assertEqual(receipt["trace"], GOLDEN["ORIENT.002"]["trace"])
        self.assertEqual(receipt["score"], GOLDEN["ORIENT.002"]["score"])

        logical = kit.resolve_logical_world(world, case["world"])
        platform_records = [event for event in logical.base_events if event["topic"] == "platform.record"]
        self.assertEqual(len(platform_records), 6)
        self.assertEqual({event["at"] for event in platform_records}, {100, 101, 102})
        replay = verifier.replay_case(world, case, [], cre.root_branch_id(world.bundle))
        self.assertEqual([item["event"] for item in replay.trace_items], witness["answer"]["trace_event_ids"])
        reachable = {
            event["payload"]["node"]
            for event in replay.events.values()
            if event["topic"] == "platform.reachable"
        }
        self.assertEqual(reachable, {"NORTH", "NORTH-2", "SOUTH", "SOUTH-2"})
        errors = [event for event in replay.events.values() if event["topic"] == "audit.replay-error"]
        self.assertEqual({event["payload"]["code"] for event in errors}, {
            "parent_not_active", "wrong_tie_break", "fixed_point_incomplete",
        })

        altered = copy.deepcopy(witness)
        altered["answer"]["records"][0]["index"] = 9
        altered.pop("meta", None)
        wrong_path = self.root / "wrong-replay-record.json"
        wrong_path.write_bytes(cre.canonical_bytes(altered))
        invalid = verifier.verify_witness(world_path, wrong_path, {"case:ORIENT.001"})
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["diagnostics"][0]["code"], "claim_mismatch")

    def test_author_event_labels_allow_forward_references_and_reject_bad_graphs(self) -> None:
        parent = {
            "topic": "label.parent", "at": 0, "payload": None, "parents": [],
            "origin": {"kind": "base", "source": "test", "sequence": 0},
        }
        child = {
            "topic": "label.child", "at": 1, "payload": None, "parents": ["@parent"],
            "origin": {"kind": "base", "source": "test", "sequence": 1},
        }
        events = build_slice.compile_base_events("TEST", [
            {"label": "child", "body": child},
            {"label": "parent", "body": parent},
        ])
        by_topic = {event["topic"]: event for event in events}
        self.assertEqual(by_topic["label.child"]["parents"], [by_topic["label.parent"]["id"]])

        unknown = copy.deepcopy(child)
        unknown["parents"] = ["@missing"]
        with self.assertRaisesRegex(build_slice.BuildError, "unknown label"):
            build_slice.compile_base_events("TEST", [{"label": "child", "body": unknown}])

        cyclic_parent = copy.deepcopy(parent)
        cyclic_parent["parents"] = ["@child"]
        with self.assertRaisesRegex(build_slice.BuildError, "contain a cycle"):
            build_slice.compile_base_events("TEST", [
                {"label": "parent", "body": cyclic_parent},
                {"label": "child", "body": child},
            ])

    def test_orient_003_separates_active_state_from_projection(self) -> None:
        bundle_path, author, _summary = self.build("camera-view")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        world = kit.verify_world(world_path)
        descriptors = {item["id"]: item for item in world.json_values["cases/index.json"]["cases"]}
        case = verifier.validate_case_descriptor(descriptors["ORIENT.003"], world)
        witness_path = author / "ORIENT.003.witness.json"
        witness = cre.load_json(witness_path)

        receipt = verifier.verify_witness(world_path, witness_path, {"case:ORIENT.001"})
        self.assertTrue(receipt["valid"])
        self.assertEqual(witness["answer"], GOLDEN["ORIENT.003"]["answer"])
        self.assertEqual(receipt["trace"], GOLDEN["ORIENT.003"]["trace"])
        self.assertEqual(receipt["score"], GOLDEN["ORIENT.003"]["score"])

        logical = kit.resolve_logical_world(world, case["world"])
        replay = verifier.replay_case(world, case, [], cre.root_branch_id(world.bundle))
        self.assertEqual(len(logical.base_events), 6)
        self.assertEqual(len(replay.events), 13)
        self.assertEqual(replay.records, GOLDEN["ORIENT.003"]["answer"]["projected_records"])
        alert = [event for event in replay.events.values() if event["topic"] == "camera.alert"]
        self.assertEqual(len(alert), 1)
        self.assertEqual(alert[0]["id"], witness["answer"]["hidden_event"])
        self.assertNotIn(alert[0]["id"], cre.canonical_text(replay.records))
        service_detections = [
            event for event in replay.events.values()
            if event["topic"] == "camera.detection" and event["payload"]["zone"] == "service"
        ]
        service_public = [
            event for event in replay.events.values()
            if event["topic"] == "camera.public-sighting" and event["payload"]["zone"] == "service"
        ]
        self.assertEqual(len(service_detections), 1)
        self.assertEqual(service_public, [])

        mutations = {
            "active_event_count": 12,
            "projected_records": [],
            "projection": "sha256:" + "0" * 64,
            "hidden_event": "sha256:" + "0" * 64,
        }
        for field, replacement in mutations.items():
            altered = copy.deepcopy(witness)
            altered["answer"][field] = replacement
            altered.pop("meta", None)
            path = self.root / f"wrong-observation-{field}.json"
            path.write_bytes(cre.canonical_bytes(altered))
            invalid = verifier.verify_witness(world_path, path, {"case:ORIENT.001"})
            with self.subTest(field=field):
                self.assertFalse(invalid["valid"])
                self.assertEqual(invalid["diagnostics"][0]["code"], "claim_mismatch")

    def test_orient_004_retimes_base_cause_and_recomputes_symmetric_difference(self) -> None:
        bundle_path, author, _summary = self.build("counterfactual")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        world = kit.verify_world(world_path)
        descriptors = {item["id"]: item for item in world.json_values["cases/index.json"]["cases"]}
        case = verifier.validate_case_descriptor(descriptors["ORIENT.004"], world)
        witness_path = author / "ORIENT.004.witness.json"
        witness = cre.load_json(witness_path)

        locked = verifier.verify_witness(world_path, witness_path)
        self.assertFalse(locked["valid"])
        self.assertEqual(locked["diagnostics"][0]["code"], "case_locked")
        receipt = verifier.verify_witness(world_path, witness_path, {"case:ORIENT.002"})
        self.assertTrue(receipt["valid"])
        self.assertEqual(witness["answer"], GOLDEN["ORIENT.004"]["answer"])
        self.assertEqual(receipt["trace"], GOLDEN["ORIENT.004"]["trace"])
        self.assertEqual(receipt["score"], GOLDEN["ORIENT.004"]["score"])

        logical = kit.resolve_logical_world(world, case["world"])
        operation = witness["intervention"]["operations"]
        self.assertEqual(len(operation), 1)
        self.assertEqual(operation[0]["kind"], "retime")
        self.assertEqual(operation[0]["at"], 12)
        replay = verifier.replay_case(world, case, operation, witness["parent_branch"])
        self.assertEqual(len(replay.baseline.events), 7)
        self.assertEqual(len(replay.events), 2)
        self.assertEqual(replay.baseline.records, GOLDEN["ORIENT.004"]["answer"]["baseline_records"])
        self.assertEqual(replay.records, [])
        self.assertEqual(replay.changed_event_ids, GOLDEN["ORIENT.004"]["answer"]["changed_event_ids"])
        self.assertEqual(
            set(replay.changed_event_ids),
            set(replay.baseline.events) ^ set(replay.events),
        )
        original = next(event for event in logical.base_events if event["topic"] == "rail.maintenance-notice")
        replacement = next(event for event in replay.events.values() if event["topic"] == "rail.maintenance-notice")
        self.assertEqual(replacement["origin"]["kind"], "player")
        self.assertEqual(replacement["origin"]["supersedes"], original["id"])
        self.assertEqual(replacement["at"], 12)

        empty = copy.deepcopy(witness)
        empty["intervention"]["operations"] = []
        empty.pop("meta", None)
        empty_path = self.root / "empty-intervention.json"
        empty_path.write_bytes(cre.canonical_bytes(empty))
        empty_receipt = verifier.verify_witness(world_path, empty_path, {"case:ORIENT.002"})
        self.assertFalse(empty_receipt["valid"])
        self.assertEqual(empty_receipt["diagnostics"][0]["code"], "intervention_required")

        closure = next(event for event in replay.baseline.events.values() if event["topic"] == "rail.maintenance-closure")
        direct = copy.deepcopy(witness)
        direct["intervention"]["operations"][0]["event"] = closure["id"]
        direct.pop("meta", None)
        direct_path = self.root / "derived-intervention.json"
        direct_path.write_bytes(cre.canonical_bytes(direct))
        direct_receipt = verifier.verify_witness(world_path, direct_path, {"case:ORIENT.002"})
        self.assertFalse(direct_receipt["valid"])
        diagnostic = direct_receipt["diagnostics"][0]
        self.assertEqual(diagnostic["code"], "derived_event_not_intervenable")
        self.assertEqual(set(diagnostic["context"]["base_ancestors"]), {event["id"] for event in logical.base_events})

        mutations = {
            "branch": "sha256:" + "0" * 64,
            "baseline_records": [],
            "candidate_records": [{"unexpected": True}],
            "changed_event_ids": witness["answer"]["changed_event_ids"][:-1],
            "projection": "sha256:" + "0" * 64,
        }
        for field, replacement_value in mutations.items():
            altered = copy.deepcopy(witness)
            altered["answer"][field] = replacement_value
            altered.pop("meta", None)
            path = self.root / f"wrong-branch-{field}.json"
            path.write_bytes(cre.canonical_bytes(altered))
            invalid = verifier.verify_witness(world_path, path, {"case:ORIENT.002"})
            with self.subTest(field=field):
                self.assertFalse(invalid["valid"])
                self.assertEqual(invalid["diagnostics"][0]["code"], "claim_mismatch")

    def test_orient_005_reverifies_embedded_witness_and_grants_export(self) -> None:
        bundle_path, author, _summary = self.build("export")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        witness_path = author / "ORIENT.005.witness.json"
        witness = cre.load_json(witness_path)
        exported = cre.load_json(author / "ORIENT.004.witness.json")
        self.assertEqual(witness["answer"]["export"], exported)

        facts = {"case:ORIENT.002", "case:ORIENT.004"}
        receipt = verifier.verify_witness(world_path, witness_path, facts)
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["unlocks"], GOLDEN["ORIENT.005"]["unlocks"])
        self.assertEqual(receipt["score"], GOLDEN["ORIENT.005"]["score"])

        locked = verifier.verify_witness(world_path, witness_path, {"case:ORIENT.002", "case:ORIENT.003"})
        self.assertFalse(locked["valid"])
        self.assertEqual(locked["diagnostics"][0]["code"], "embedded_case_locked")

        malformed = copy.deepcopy(witness)
        malformed["answer"]["export"]["claimed"]["trace"] = "sha256:" + "0" * 64
        malformed.pop("meta", None)
        path = self.root / "bad-export.json"
        path.write_bytes(cre.canonical_bytes(malformed))
        invalid = verifier.verify_witness(world_path, path, facts)
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["diagnostics"][0]["code"], "embedded_witness_invalid")
        self.assertEqual(invalid["diagnostics"][0]["context"]["inner_code"], "claimed_mismatch")

    def test_cascade_001_contracts_and_intervention_ordering(self) -> None:
        bundle_path, author, _summary = self.build("late-green")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        world = kit.verify_world(world_path)
        descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "CASCADE.001")
        case = verifier.validate_case_descriptor(descriptor, world)
        facts = {"case:ORIENT.001", "case:ORIENT.002"}
        author_path = author / "CASCADE.001.witness.json"
        author_witness = cre.load_json(author_path)
        receipt = verifier.verify_witness(world_path, author_path, facts)
        self.assertTrue(receipt["valid"])
        self.assertEqual(author_witness["answer"], GOLDEN["CASCADE.001"]["answer"])
        self.assertEqual(receipt["metrics"], GOLDEN["CASCADE.001"]["metrics"])
        self.assertEqual(receipt["score"], GOLDEN["CASCADE.001"]["score"])

        root = verifier.replay_root_case(world, case)
        self.assertFalse(root.records[0]["safe"])
        logical = kit.resolve_logical_world(world, case["world"])
        request = next(event for event in logical.base_events if event["topic"] == "traffic.pedestrian-request")
        parent = cre.root_branch_id(world.bundle)

        def verify_operation(operation: dict[str, object], name: str) -> dict[str, object]:
            replay = verifier.replay_case(world, case, [operation], parent)
            intervention = {"format": "afterimage-intervention/0.1", "bundle": world.bundle, "parent_branch": parent, "case": case["id"], "operations": [operation]}
            witness = {"format": "afterimage-witness/0.1", "semantics": "cre/0.1", "bundle": world.bundle, "case": case["id"], "parent_branch": parent, "intervention": intervention, "answer": {"contracts": replay.records, "branch": replay.branch, "projection": replay.projection}, "claimed": {"branch": replay.branch, "projection": replay.projection, "trace": replay.trace}}
            path = self.root / f"cascade-{name}.json"
            path.write_bytes(cre.canonical_bytes(witness))
            return verifier.verify_witness(world_path, path, facts)

        reroute = verify_operation({"kind": "inject", "topic": "traffic.reroute-request", "at": 16, "payload": {"intersection": "I-9"}, "parents": []}, "reroute")
        self.assertTrue(reroute["valid"])
        self.assertEqual(reroute["metrics"]["intervention_weight"], 30)
        self.assertLess(receipt["metrics"]["effective_cost"], reroute["metrics"]["effective_cost"])

        suppressed = verify_operation({"kind": "suppress", "event": request["id"]}, "suppress")
        self.assertFalse(suppressed["valid"])
        self.assertEqual(suppressed["diagnostics"][0]["code"], "answer_schema")

        violation = next(event for event in root.events.values() if event["topic"] == "contract.violation")
        direct = copy.deepcopy(author_witness)
        direct["intervention"]["operations"][0]["event"] = violation["id"]
        direct.pop("meta", None)
        direct_path = self.root / "cascade-derived.json"
        direct_path.write_bytes(cre.canonical_bytes(direct))
        direct_receipt = verifier.verify_witness(world_path, direct_path, facts)
        self.assertFalse(direct_receipt["valid"])
        self.assertEqual(direct_receipt["diagnostics"][0]["code"], "derived_event_not_intervenable")

    def test_merge_001_classification_schedule_and_exact_certificate(self) -> None:
        bundle_path, author, _summary = self.build("three-clocks")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        world = kit.verify_world(world_path)
        descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "MERGE.001")
        case = verifier.validate_case_descriptor(descriptor, world)
        logical = kit.resolve_logical_world(world, case["world"])
        records = {event["id"]: event for event in logical.base_events if event["topic"] == "merge.record"}
        by_key = {event["payload"]["key"]: event for event in records.values()}
        author_path = author / "MERGE.001.witness.json"
        author_witness = cre.load_json(author_path)
        facts = {"case:ORIENT.002"}
        receipt = verifier.verify_witness(world_path, author_path, facts)

        self.assertTrue(receipt["valid"])
        self.assertEqual(len(records), 9)
        self.assertEqual(len(author_witness["answer"]["accepted"]), 8)
        self.assertEqual(author_witness["answer"]["rejected"], [{"event": by_key["C2"]["id"], "reason": "inconsistent"}])
        self.assertEqual(len(author_witness["answer"]["certificate"]), 12)
        self.assertEqual(author_witness["answer"], GOLDEN["MERGE.001"]["answer"])
        self.assertEqual(receipt["metrics"], GOLDEN["MERGE.001"]["metrics"])
        self.assertEqual(receipt["score"], GOLDEN["MERGE.001"]["score"])
        self.assertEqual([row["key"] for row in verifier.replay_root_case(world, case).records], ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"])

        def exact_certificate(answer: dict[str, object]) -> list[dict[str, object]]:
            accepted = {item["event"] for item in answer["accepted"]}
            required: set[tuple[str, str, int]] = set()
            for event_id in accepted:
                for parent in records[event_id]["parents"]:
                    if parent in accepted:
                        required.add((parent, event_id, 1))
            by_source: dict[str, list[str]] = {}
            for event_id in accepted:
                by_source.setdefault(records[event_id]["payload"]["source"], []).append(event_id)
            for event_ids in by_source.values():
                event_ids.sort(key=lambda event_id: records[event_id]["payload"]["sequence"])
                required.update((before, after, 1) for before, after in zip(event_ids, event_ids[1:]))
            return [
                {"before": before, "after": after, "minimum_gap": gap}
                for before, after, gap in sorted(required, key=lambda edge: tuple(cre.parse_id(value) if isinstance(value, str) and value.startswith("sha256:") else value for value in edge))
            ]

        def verify_mutation(name: str, mutate: object) -> dict[str, object]:
            altered = copy.deepcopy(author_witness)
            mutate(altered["answer"])
            altered.pop("meta", None)
            path = self.root / f"merge-{name}.json"
            path.write_bytes(cre.canonical_bytes(altered))
            return verifier.verify_witness(world_path, path, facts)

        def accept_c2(answer: dict[str, object]) -> None:
            rejected = answer["rejected"].pop()
            answer["accepted"].append({"event": rejected["event"], "at": 33})

        inconsistent = verify_mutation("accept-inconsistent", accept_c2)
        self.assertFalse(inconsistent["valid"])
        self.assertEqual(inconsistent["diagnostics"][0]["code"], "merge_order")

        outside = verify_mutation("outside-interval", lambda answer: answer["accepted"][0].update({"at": 12}))
        self.assertFalse(outside["valid"])
        self.assertEqual(outside["diagnostics"][0]["code"], "merge_interval")

        duplicate = verify_mutation("duplicate-certificate", lambda answer: answer["certificate"].append(copy.deepcopy(answer["certificate"][0])))
        self.assertFalse(duplicate["valid"])
        self.assertEqual(duplicate["diagnostics"][0]["code"], "merge_certificate")

        malformed = verify_mutation("malformed-certificate", lambda answer: answer["certificate"][0].update({"before": []}))
        self.assertFalse(malformed["valid"])
        self.assertEqual(malformed["diagnostics"][0]["code"], "merge_certificate")

        def reject_conflict_pair(answer: dict[str, object]) -> None:
            c3 = by_key["C3"]["id"]
            answer["accepted"] = [item for item in answer["accepted"] if item["event"] != c3]
            answer["rejected"][0]["reason"] = "conflict_set"
            answer["rejected"].append({"event": c3, "reason": "conflict_set"})
            answer["certificate"] = exact_certificate(answer)

        baseline = verify_mutation("conflict-set-baseline", reject_conflict_pair)
        self.assertTrue(baseline["valid"])
        self.assertEqual(baseline["metrics"]["rejected_weight"], 6)
        self.assertLess(baseline["score"]["total"], receipt["score"]["total"])

        def reject_feasible_a3(answer: dict[str, object]) -> None:
            a3 = by_key["A3"]["id"]
            answer["accepted"] = [item for item in answer["accepted"] if item["event"] != a3]
            answer["rejected"].append({"event": a3, "reason": "inconsistent"})
            answer["certificate"] = exact_certificate(answer)

        unsupported = verify_mutation("reject-feasible", reject_feasible_a3)
        self.assertFalse(unsupported["valid"])
        self.assertEqual(unsupported["diagnostics"][0]["code"], "merge_rejection")

        def claim_duplicate_without_contract(answer: dict[str, object]) -> None:
            answer["rejected"][0] = {
                "event": by_key["C2"]["id"],
                "reason": "duplicate",
                "duplicate_of": by_key["A3"]["id"],
            }

        no_contract = verify_mutation("duplicate-without-contract", claim_duplicate_without_contract)
        self.assertFalse(no_contract["valid"])
        self.assertEqual(no_contract["diagnostics"][0]["code"], "merge_rejection")

    def test_cascade_002_repairs_time_model_and_preserves_water_contracts(self) -> None:
        bundle_path, author, _summary = self.build("dry-hydrant")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        world = kit.verify_world(world_path)
        descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "CASCADE.002")
        case = verifier.validate_case_descriptor(descriptor, world)
        logical = kit.resolve_logical_world(world, case["world"])
        by_topic = {event["topic"]: event for event in logical.base_events}
        facts = {"case:CASCADE.001", "case:MERGE.001"}
        author_path = author / "CASCADE.002.witness.json"
        author_witness = cre.load_json(author_path)
        receipt = verifier.verify_witness(world_path, author_path, facts)
        self.assertTrue(receipt["valid"])
        self.assertEqual(author_witness["answer"], GOLDEN["CASCADE.002"]["answer"])
        self.assertEqual(receipt["metrics"], GOLDEN["CASCADE.002"]["metrics"])
        self.assertEqual(receipt["score"], GOLDEN["CASCADE.002"]["score"])

        root = verifier.replay_root_case(world, case)
        self.assertFalse(root.records[0]["safe"])
        self.assertEqual(root.records[0]["pressure"], 35)
        violation = next(event for event in root.events.values() if event["topic"] == "contract.violation")
        self.assertEqual(violation["payload"]["code"], "pressure_below_minimum")
        self.assertEqual(violation["payload"]["path"], ["R-2", "P-4", "T-1", "B-7", "H-9"])
        self.assertEqual((violation["payload"]["source_at"], violation["payload"]["ingestion_at"]), (8, 15))

        parent = cre.root_branch_id(world.bundle)

        def verify_operations(name: str, operations: list[dict[str, object]]) -> dict[str, object]:
            replay = verifier.replay_case(world, case, operations, parent)
            intervention = {"format": "afterimage-intervention/0.1", "bundle": world.bundle, "parent_branch": parent, "case": case["id"], "operations": operations}
            witness = {"format": "afterimage-witness/0.1", "semantics": "cre/0.1", "bundle": world.bundle, "case": case["id"], "parent_branch": parent, "intervention": intervention, "answer": {"contracts": replay.records, "branch": replay.branch, "projection": replay.projection}, "claimed": {"branch": replay.branch, "projection": replay.projection, "trace": replay.trace}}
            path = self.root / f"cascade2-{name}.json"
            path.write_bytes(cre.canonical_bytes(witness))
            return verifier.verify_witness(world_path, path, facts)

        pump = verify_operations("pump", [{"kind": "inject", "topic": "water.pump-dispatch", "at": 16, "payload": {"boost": 20, "energy": 30}, "parents": []}])
        self.assertTrue(pump["valid"])
        self.assertEqual(pump["metrics"]["intervention_weight"], 20)
        self.assertLess(receipt["metrics"]["effective_cost"], pump["metrics"]["effective_cost"])

        sample_only = verify_operations("sample-only", [{"kind": "retime", "event": by_topic["water.pressure-sample"]["id"], "at": 10}])
        self.assertFalse(sample_only["valid"])
        self.assertEqual(sample_only["diagnostics"][0]["code"], "claim_mismatch")

        suppressed = verify_operations("suppress", [{"kind": "suppress", "event": by_topic["water.pressure-sample"]["id"]}])
        self.assertFalse(suppressed["valid"])

        closed = verify_operations("valve", [{"kind": "inject", "topic": "water.valve-close", "at": 12, "payload": {"valve": "V-3"}, "parents": []}])
        self.assertFalse(closed["valid"])

    def test_pulse_001_exhaustive_program_and_smallest_counterexample(self) -> None:
        bundle_path, author, _summary = self.build("one-bell")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        author_path = author / "PULSE.001.witness.json"
        witness = cre.load_json(author_path)
        facts = {"case:ORIENT.002"}
        receipt = verifier.verify_witness(world_path, author_path, facts)
        self.assertTrue(receipt["valid"])
        self.assertEqual(witness["answer"], GOLDEN["PULSE.001"]["answer"])
        self.assertEqual(receipt["metrics"], GOLDEN["PULSE.001"]["metrics"])
        self.assertEqual(receipt["score"], GOLDEN["PULSE.001"]["score"])

        early = copy.deepcopy(witness)
        early["answer"]["program"]["handlers"][0]["actions"][0]["value"][2][1] = 4
        early.pop("meta", None)
        early_path = self.root / "pulse-early.json"
        early_path.write_bytes(cre.canonical_bytes(early))
        invalid = verifier.verify_witness(world_path, early_path, facts)
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["diagnostics"][0]["code"], "pulse_counterexample")
        self.assertEqual(invalid["diagnostics"][0]["context"]["input"], [0])
        self.assertEqual([item["at"] for item in invalid["diagnostics"][0]["context"]["expected"]], [5])
        self.assertEqual([item["at"] for item in invalid["diagnostics"][0]["context"]["observed"]], [4])

        malformed = copy.deepcopy(witness)
        malformed["answer"]["program"]["cells"][0]["initial"] = False
        malformed.pop("meta", None)
        malformed_path = self.root / "pulse-malformed.json"
        malformed_path.write_bytes(cre.canonical_bytes(malformed))
        rejected = verifier.verify_witness(world_path, malformed_path, facts)
        self.assertFalse(rejected["valid"])
        self.assertEqual(rejected["diagnostics"][0]["code"], "invalid_pulse_program")

    def test_mosaic_001_d4_mappings_coverage_and_supported_decoy(self) -> None:
        bundle_path, author, _summary = self.build("four-corners")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        author_path = author / "MOSAIC.001.witness.json"
        witness = cre.load_json(author_path)
        facts = {"case:ORIENT.003", "case:MERGE.001"}
        receipt = verifier.verify_witness(world_path, author_path, facts)
        self.assertTrue(receipt["valid"])
        self.assertEqual(len(witness["answer"]["global"]["vertices"]), 9)
        self.assertEqual(len(witness["answer"]["global"]["edges"]), 12)
        self.assertEqual([item["fragment"] for item in witness["answer"]["unused"]], ["F-X"])
        self.assertEqual(receipt["metrics"], GOLDEN["MOSAIC.001"]["metrics"])
        self.assertEqual(receipt["score"], GOLDEN["MOSAIC.001"]["score"])

        def verify_mutation(name: str, mutate: object) -> dict[str, object]:
            altered = copy.deepcopy(witness)
            mutate(altered["answer"])
            altered.pop("meta", None)
            path = self.root / f"mosaic-{name}.json"
            path.write_bytes(cre.canonical_bytes(altered))
            return verifier.verify_witness(world_path, path, facts)

        wrong_mapping = verify_mutation("mapping", lambda answer: answer["used"][0]["mapping"][0].update({"global": "v8"}))
        self.assertFalse(wrong_mapping["valid"])
        self.assertEqual(wrong_mapping["diagnostics"][0]["code"], "mosaic_mapping")

        def discard_real(answer: dict[str, object]) -> None:
            removed = answer["used"].pop(0)
            answer["unused"].append({"fragment": removed["fragment"], "reason": "invariant_conflict"})

        false_decoy = verify_mutation("false-decoy", discard_real)
        self.assertFalse(false_decoy["valid"])
        self.assertEqual(false_decoy["diagnostics"][0]["code"], "mosaic_classification")

    def test_lens_001_exhaustive_round_trip_and_provenance_laws(self) -> None:
        bundle_path, author, _summary = self.build("two-addresses")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        author_path = author / "LENS.001.witness.json"
        witness = cre.load_json(author_path)
        facts = {"case:CASCADE.002", "case:ORIENT.003"}
        receipt = verifier.verify_witness(world_path, author_path, facts)
        self.assertTrue(receipt["valid"])
        self.assertEqual(witness["answer"], GOLDEN["LENS.001"]["answer"])
        self.assertEqual(receipt["metrics"], GOLDEN["LENS.001"]["metrics"])
        self.assertEqual(receipt["score"], GOLDEN["LENS.001"]["score"])

        def verify_program(name: str, program: dict[str, object]) -> dict[str, object]:
            altered = copy.deepcopy(witness)
            altered["answer"]["program"] = program
            altered.pop("meta", None)
            path = self.root / f"lens-{name}.json"
            path.write_bytes(cre.canonical_bytes(altered))
            return verifier.verify_witness(world_path, path, facts)

        no_boundary = copy.deepcopy(witness["answer"]["program"])
        no_boundary["complement_schema"] = [item for item in no_boundary["complement_schema"] if item["name"] != "boundary_street"]
        boundary_failure = verify_program("no-boundary", no_boundary)
        self.assertFalse(boundary_failure["valid"])
        self.assertEqual(boundary_failure["diagnostics"][0]["code"], "lens_counterexample")
        self.assertEqual(boundary_failure["diagnostics"][0]["context"]["law"], "GetPut")

        no_provenance = copy.deepcopy(witness["answer"]["program"])
        no_provenance["complement_schema"] = [item for item in no_provenance["complement_schema"] if item["name"] != "provenance"]
        provenance_failure = verify_program("no-provenance", no_provenance)
        self.assertFalse(provenance_failure["valid"])
        self.assertEqual(provenance_failure["diagnostics"][0]["code"], "lens_counterexample")

    def test_cascade_003_proves_hidden_relay_without_disclosing_private_payload(self) -> None:
        bundle_path, author, _summary = self.build("tomorrow-redacted")
        world_path = self.root / "world"
        kit.extract_bundle(kit.load_bundle(bundle_path), world_path)
        world = kit.verify_world(world_path)
        descriptor = next(item for item in world.json_values["cases/index.json"]["cases"] if item["id"] == "CASCADE.003")
        case = verifier.validate_case_descriptor(descriptor, world)
        facts = {"case:CASCADE.002", "case:ORIENT.003", "case:MOSAIC.001"}
        author_path = author / "CASCADE.003.witness.json"
        witness = cre.load_json(author_path)
        receipt = verifier.verify_witness(world_path, author_path, facts)
        self.assertTrue(receipt["valid"])
        self.assertIsNone(witness["intervention"])
        self.assertNotIn(b"private-204", cre.canonical_bytes(witness["answer"]))
        self.assertEqual(witness["answer"], GOLDEN["CASCADE.003"]["answer"])
        self.assertEqual(receipt["metrics"], GOLDEN["CASCADE.003"]["metrics"])
        self.assertEqual(receipt["score"], GOLDEN["CASCADE.003"]["score"])
        self.assertEqual(next(row for row in witness["answer"]["public_rows"] if row["edge"] == "v4-v7"), {"edge": "v4-v7", "status": "unavailable", "policy": "housing.restricted"})

        wrong_digest = copy.deepcopy(witness)
        wrong_digest["answer"]["relay"]["provenance_digest"] = "sha256:" + "0" * 64
        wrong_digest.pop("meta", None)
        wrong_path = self.root / "cascade3-wrong-digest.json"
        wrong_path.write_bytes(cre.canonical_bytes(wrong_digest))
        invalid = verifier.verify_witness(world_path, wrong_path, facts)
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["diagnostics"][0]["code"], "cascade_proof")

        parent = cre.root_branch_id(world.bundle)

        def capability_witness(scope: str, name: str) -> dict[str, object]:
            operation = {"kind": "inject", "topic": "audit.capability", "at": 18, "payload": {"capability": "audit.relay", "scope": scope}, "parents": []}
            replay = verifier.replay_case(world, case, [operation], parent)
            altered = copy.deepcopy(witness)
            altered["parent_branch"] = parent
            altered["intervention"] = {"format": "afterimage-intervention/0.1", "bundle": world.bundle, "parent_branch": parent, "case": case["id"], "operations": [operation]}
            altered["answer"]["branch"] = replay.branch
            altered["answer"]["projection"] = replay.projection
            altered["answer"]["public_rows"] = replay.records
            altered["claimed"] = {"branch": replay.branch, "projection": replay.projection, "trace": replay.trace}
            altered.pop("meta", None)
            path = self.root / f"cascade3-{name}.json"
            path.write_bytes(cre.canonical_bytes(altered))
            return verifier.verify_witness(world_path, path, facts)

        scoped = capability_witness("CASCADE.003", "scoped")
        self.assertTrue(scoped["valid"])
        self.assertEqual(scoped["metrics"]["intervention_weight"], 15)
        self.assertLess(receipt["metrics"]["effective_cost"], scoped["metrics"]["effective_cost"])

        unscoped = capability_witness("all-cases", "unscoped")
        self.assertFalse(unscoped["valid"])
        self.assertEqual(unscoped["diagnostics"][0]["code"], "cascade_proof")


if __name__ == "__main__":
    unittest.main(verbosity=2)
