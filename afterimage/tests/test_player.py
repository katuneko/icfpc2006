from __future__ import annotations

import copy
import contextlib
import io
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

import cre  # noqa: E402
import player  # noqa: E402
import prepare_playtest  # noqa: E402


ARCHIVE = ROOT / "dist" / "afterimage-slice-dev.afterimage"
AUTHORS = ROOT / "dist" / "author-baselines"


class PlayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "desk"
        player.init_workspace(ARCHIVE, self.workspace, telemetry=True)
        self.state, self.world = player.load_workspace(self.workspace)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def refresh(self) -> None:
        self.state, self.world = player.load_workspace(self.workspace)

    def test_progress_branch_score_and_byte_exact_replay(self) -> None:
        initial = player.status(self.workspace, self.state, self.world)
        self.assertEqual(initial["solved"], [])
        self.assertEqual(initial["visible"], ["ORIENT.001"])

        receipt, code = player.verify_submission(
            self.workspace,
            self.state,
            self.world,
            AUTHORS / "ORIENT.001.witness.json",
        )
        self.assertEqual(code, 0)
        self.assertTrue(receipt["valid"])
        self.refresh()
        progressed = player.status(self.workspace, self.state, self.world)
        self.assertIn("ORIENT.002", progressed["visible"])
        self.assertIn("ORIENT.003", progressed["visible"])

        score = player.score_workspace(self.workspace, self.state, self.world)
        self.assertEqual(score["total"], 35)
        self.assertEqual(score["nominal_solved"], 40)
        self.assertEqual(score["nominal_slice"], 1200)

        intervention = self.root / "null.json"
        intervention.write_bytes(cre.canonical_bytes(None))
        branch = player.branch_case(self.workspace, self.state, self.world, "ORIENT.001", intervention)
        self.assertEqual(branch["branch"], receipt["branch"])
        branch_with_trace = player.branch_case(
            self.workspace,
            self.state,
            self.world,
            "ORIENT.001",
            intervention,
            include_trace_items=True,
        )
        self.assertTrue(branch_with_trace["trace_items"])
        stored = player.load_branch(self.workspace, branch["branch"])
        self.assertNotIn("trace_items", stored)
        comparison = player.compare_branches(self.workspace, self.state, branch["branch"], branch["branch"])
        self.assertEqual(comparison["only_first"], [])
        self.assertEqual(comparison["only_second"], [])

        witness = json.loads((AUTHORS / "ORIENT.001.witness.json").read_text())
        event_id = witness["answer"]["event_id"]
        traced = player.trace_event(self.workspace, self.state, self.world, event_id, "parents")
        self.assertEqual(traced["matches"][0]["event"]["id"], event_id)
        replay = player.verifier.replay_root_case(
            self.world,
            next(case for case in self.world.json_values["cases/index.json"]["cases"] if case["id"] == "ORIENT.001"),
        )
        derived = next(event for event in replay.events.values() if event["origin"]["kind"] == "derived")
        derived_trace = player.trace_event(self.workspace, self.state, self.world, derived["id"], "parents")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            player.emit_text(derived_trace)
        self.assertIn('"$bytes"', output.getvalue())
        hint = player.hint(self.workspace, self.state, self.world, "ORIENT.001", 1)
        self.assertEqual(hint["level"], 1)

        before = {path.name: path.read_bytes() for path in player.receipt_files(self.workspace)}
        replayed = player.replay_workspace(self.workspace, self.state, self.world)
        self.assertEqual(replayed["receipts"], 1)
        after = {path.name: path.read_bytes() for path in player.receipt_files(self.workspace)}
        self.assertEqual(after, before)
        reset = player.reset_workspace(self.workspace, self.state, self.world, keep_witnesses=True)
        self.assertEqual(reset["receipts"], 1)
        self.assertEqual({path.name: path.read_bytes() for path in player.receipt_files(self.workspace)}, before)

    def test_telemetry_is_monotonic_and_excludes_submission_material(self) -> None:
        witness_path = AUTHORS / "ORIENT.001.witness.json"
        witness = json.loads(witness_path.read_text())
        receipt, code = player.verify_submission(self.workspace, self.state, self.world, witness_path)
        self.assertEqual(code, 0)
        self.assertTrue(receipt["valid"])
        self.refresh()
        player.score_workspace(self.workspace, self.state, self.world)
        player.hint(self.workspace, self.state, self.world, "ORIENT.001", 2)
        output = self.root / "telemetry.json"
        exported = player.export_telemetry(self.workspace, self.state, self.world, output)
        self.assertEqual(exported["events"], 3)
        raw = output.read_bytes()
        self.assertNotIn(cre.canonical_bytes(witness["answer"]), raw)
        self.assertNotIn(witness["answer"]["event_id"].encode(), raw)
        self.assertNotIn(b'"intervention"', raw)
        self.assertNotIn(b'"parent_branch"', raw)
        payload = json.loads(raw)
        ticks = [event["tick"] for event in payload["events"]]
        times = [event["unix_ms"] for event in payload["events"]]
        self.assertEqual(ticks, sorted(set(ticks)))
        self.assertEqual(times, sorted(times))
        self.assertEqual(payload["events"][0]["event"], "verify")

    def test_json_cli_and_destructive_reset(self) -> None:
        command = [
            sys.executable,
            str(ROOT / "tools" / "player.py"),
            "--json",
            "status",
            str(self.workspace),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(completed.stdout)["visible"], ["ORIENT.001"])
        receipt, code = player.verify_submission(
            self.workspace,
            self.state,
            self.world,
            AUTHORS / "ORIENT.001.witness.json",
        )
        self.assertEqual(code, 0)
        self.assertTrue(receipt["valid"])
        reset = player.reset_workspace(self.workspace, self.state, self.world, keep_witnesses=False)
        self.assertEqual(reset["receipts"], 0)
        self.assertEqual(list((self.workspace / "witnesses").iterdir()), [])
        self.assertEqual(list((self.workspace / "receipts").iterdir()), [])

    def test_canonicalize_accepts_editor_json_and_rejects_ambiguous_input(self) -> None:
        draft = self.root / "draft.json"
        canonical = self.root / "canonical.json"
        value = {"format": "fixture", "text": "e\u0301", "items": [True, None, 7]}
        draft.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result = player.canonicalize_file(draft, canonical)
        normalized = {"format": "fixture", "text": "é", "items": [True, None, 7]}
        self.assertEqual(canonical.read_bytes(), cre.canonical_bytes(normalized))
        self.assertEqual(result["bytes"], len(canonical.read_bytes()))

        with self.assertRaises(player.PlayerError) as captured:
            player.canonicalize_file(draft, canonical)
        self.assertEqual(captured.exception.code, "output_exists")

        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
        with self.assertRaises(player.PlayerError) as captured:
            player.canonicalize_file(duplicate, self.root / "duplicate-canonical.json")
        self.assertEqual(captured.exception.code, "duplicate_key")

        command = [
            sys.executable,
            str(ROOT / "tools" / "player.py"),
            "--json",
            "canonicalize",
            str(draft),
            str(self.root / "cli-canonical.json"),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(completed.stdout)["type"], "canonicalized")

    def test_equal_integer_score_prefers_lower_raw_effective_cost(self) -> None:
        first = json.loads((AUTHORS / "PULSE.001.receipt.json").read_text())
        second = copy.deepcopy(first)
        first["witness"] = "sha256:" + "1" * 64
        second["witness"] = "sha256:" + "2" * 64
        first["metrics"]["effective_cost"] = 200
        second["metrics"]["effective_cost"] = 100
        player.canonical_write(self.workspace / "receipts" / "first.json", first)
        player.canonical_write(self.workspace / "receipts" / "second.json", second)
        score = player.score_workspace(self.workspace, self.state, self.world)
        self.assertEqual(score["cases"][0]["case"], "PULSE.001")
        self.assertEqual(score["cases"][0]["metrics"]["effective_cost"], 100)

    def test_all_twelve_author_receipts_unlock_score_and_replay_offline(self) -> None:
        order = [
            "ORIENT.001",
            "ORIENT.002",
            "ORIENT.003",
            "ORIENT.004",
            "ORIENT.005",
            "MERGE.001",
            "PULSE.001",
            "CASCADE.001",
            "CASCADE.002",
            "MOSAIC.001",
            "LENS.001",
            "CASCADE.003",
        ]
        for case_id in order:
            receipt, code = player.verify_submission(
                self.workspace,
                self.state,
                self.world,
                AUTHORS / f"{case_id}.witness.json",
            )
            self.assertEqual(code, 0, case_id)
            self.assertTrue(receipt["valid"], case_id)
            self.refresh()
        final = player.status(self.workspace, self.state, self.world)
        self.assertEqual(final["solved"], sorted(order))
        self.assertEqual(final["visible"], sorted(order))
        score = player.score_workspace(self.workspace, self.state, self.world)
        self.assertEqual(score["total"], 1121)
        self.assertEqual(score["nominal_solved"], 1200)
        before = {path.name: path.read_bytes() for path in player.receipt_files(self.workspace)}
        replayed = player.replay_workspace(self.workspace, self.state, self.world)
        self.assertEqual(replayed["receipts"], 12)
        self.assertEqual({path.name: path.read_bytes() for path in player.receipt_files(self.workspace)}, before)

    def test_playtest_packages_are_separated_reproducible_and_offline(self) -> None:
        first = self.root / "release-a"
        second = self.root / "release-b"
        first_result = prepare_playtest.prepare(ARCHIVE, first)
        second_result = prepare_playtest.prepare(ARCHIVE, second)
        self.assertEqual(
            [(item["path"], item["sha256"]) for item in first_result["packages"]],
            [(item["path"], item["sha256"]) for item in second_result["packages"]],
        )
        with zipfile.ZipFile(first / "engine-session.zip") as archive:
            names = set(archive.namelist())
            self.assertIn("conformance/suite.json", names)
            self.assertIn("conformance/full-suite.json", names)
            self.assertIn("conformance/golden.json", names)
            self.assertIn("conformance/check.py", names)
            self.assertNotIn("reference/python/cre.py", names)
            self.assertFalse(any("author" in name or name.startswith("content/") for name in names))
            self.assertFalse(any("golden" in name and name != "conformance/golden.json" for name in names))
        game_root = self.root / "game"
        with zipfile.ZipFile(first / "game-session.zip") as archive:
            names = set(archive.namelist())
            self.assertIn("afterimage-slice.afterimage", names)
            self.assertIn("tools/player.py", names)
            self.assertIn("tools/localization.py", names)
            for locale_code in ("en", "ja", "zh-Hans", "de"):
                self.assertIn(f"locales/{locale_code}.json", names)
            self.assertFalse(any("author-baselines" in name or "golden.json" in name or name.startswith("content/") for name in names))
            archive.extractall(game_root)
        staged_workspace = self.root / "staged-desk"
        completed = subprocess.run(
            [
                sys.executable,
                str(game_root / "tools" / "player.py"),
                "--json",
                "init",
                str(game_root / "afterimage-slice.afterimage"),
                str(staged_workspace),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={"PATH": "", "PYTHONHASHSEED": "random"},
        )
        self.assertEqual(json.loads(completed.stdout)["cases"], 12)
        completed = subprocess.run(
            [
                sys.executable,
                str(game_root / "tools" / "player.py"),
                "--locale",
                "ja",
                "--json",
                "inspect",
                str(staged_workspace),
                "ORIENT.001",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={"PATH": "", "PYTHONHASHSEED": "random"},
        )
        self.assertEqual(json.loads(completed.stdout)["locale"], "ja")
        operator_root = self.root / "operator"
        with zipfile.ZipFile(first / "operator-session.zip") as archive:
            names = set(archive.namelist())
            self.assertIn("OBSERVATION_SCHEMA.md", names)
            self.assertIn("AI_PROXY_PROTOCOL.md", names)
            self.assertIn("tools/analyze_playtest.py", names)
            self.assertIn("tools/analyze_ai_proxy.py", names)
            self.assertIn("tools/check_engine_generalization.py", names)
            self.assertIn("tests/conformance/full-suite.json", names)
            self.assertIn("tools/afterimage_kit.py", names)
            self.assertIn("reference/python/cre.py", names)
            archive.extractall(operator_root)
        completed = subprocess.run(
            [sys.executable, str(operator_root / "tools" / "analyze_playtest.py"), "--help"],
            check=True,
            capture_output=True,
            text=True,
            env={"PATH": "", "PYTHONHASHSEED": "random"},
        )
        self.assertIn("--canonicalize", completed.stdout)
        draft = operator_root / "campaign.draft.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(operator_root / "tools" / "analyze_playtest.py"),
                "--new",
                str(draft),
                "--bundle",
                first_result["bundle"],
                "--teams",
                "6",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={"PATH": "", "PYTHONHASHSEED": "random"},
        )
        self.assertEqual(json.loads(completed.stdout)["teams"], 6)
        self.assertEqual(draft.stat().st_mode & 0o777, 0o600)
        canonical = operator_root / "campaign.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(operator_root / "tools" / "analyze_playtest.py"),
                str(draft),
                "--canonicalize",
                str(canonical),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "", "PYTHONHASHSEED": "random"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(json.loads(completed.stdout)["code"], "campaign_incomplete")
        self.assertFalse(canonical.exists())

    def test_workspace_support_symlinks_are_rejected_before_writes(self) -> None:
        receipts = self.workspace / "receipts"
        receipts.rmdir()
        escape = self.root / "escape"
        escape.mkdir()
        receipts.symlink_to(escape, target_is_directory=True)
        with self.assertRaises(player.PlayerError) as caught:
            player.load_workspace(self.workspace)
        self.assertEqual(caught.exception.code, "invalid_workspace")
        self.assertEqual(list(escape.iterdir()), [])

    def test_branch_cli_replays_parent_history_by_branch_id(self) -> None:
        for case_id in ("ORIENT.001", "ORIENT.002", "CASCADE.001"):
            receipt, code = player.verify_submission(
                self.workspace,
                self.state,
                self.world,
                AUTHORS / f"{case_id}.witness.json",
            )
            self.assertEqual(code, 0, case_id)
            self.assertTrue(receipt["valid"], case_id)
            self.refresh()
        author = json.loads((AUTHORS / "CASCADE.001.witness.json").read_text())
        intervention_path = self.root / "cascade-001.intervention.json"
        intervention_path.write_bytes(cre.canonical_bytes(author["intervention"]))
        parent = player.branch_case(
            self.workspace,
            self.state,
            self.world,
            "CASCADE.001",
            intervention_path,
        )
        self.assertEqual(parent["history"]["steps"][0]["case"], "CASCADE.001")
        self.assertTrue(parent["history"]["steps"][0]["operations"])

        case_index = self.world.json_values["cases/index.json"]["cases"]
        source = next(item for item in case_index if item["id"] == "CASCADE.001")
        later = next(item for item in case_index if item["id"] == "CASCADE.002")
        replacement = copy.deepcopy(source)
        replacement.update({
            "id": "CASCADE.002",
            "title": "History CLI Fixture",
            "requires": {"all": ["case:CASCADE.001"]},
            "input_branch": "history:CASCADE.001",
            "intervention_policy": "cases/ORIENT.001/interventions.json",
        })
        later.clear()
        later.update(replacement)
        null_path = self.root / "null-history.json"
        null_path.write_bytes(cre.canonical_bytes(None))
        inherited = player.branch_case(
            self.workspace,
            self.state,
            self.world,
            "CASCADE.002",
            null_path,
            parent["branch"],
        )
        self.assertEqual(inherited["branch"], parent["branch"])
        self.assertEqual(inherited["history"], parent["history"])


if __name__ == "__main__":
    unittest.main()
