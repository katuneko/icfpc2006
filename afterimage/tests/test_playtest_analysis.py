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

import analyze_playtest as analyzer  # noqa: E402
import cre  # noqa: E402


BUNDLE = "sha256:edf8f32b5c0836b5adebe587ad690712b81d0df8082cb2f1ac44bff5d241fd27"
FAMILIES = ["ORIENT", "CASCADE", "MERGE", "PULSE", "MOSAIC", "LENS"]
COHORTS = ["runtime-builder", "runtime-builder", "algorithmic-contestant", "algorithmic-contestant", "curious-programmer", "curious-programmer"]


def campaign() -> dict:
    teams = []
    for index in range(6):
        teams.append({
            "id": f"T-{index + 1:04d}",
            "cohort": COHORTS[index],
            "engine_language": ["python", "rust", "go", "javascript", "ocaml", "cpp"][index],
            "first_receipt_minutes": [20, 30, 35, 40, 44, 60][index],
            "desk_boot_minutes": 120 + index,
            "reached_cascade003": True,
            "max_hint_level": 2,
            "projection_explained": True,
            "intended_observations_understood": True,
            "independent_valid_families": [FAMILIES[index]],
            "improved_valid_score": index < 3,
            "unrelated_families": [],
            "dominant_case": False,
            "computed_reveal": True,
            "route_count": 2,
            "cre_minutes": 45,
            "canonicalization_dominated": False,
            "pulse_or_lens_required_second_language": False,
            "cascade_blind_search": False,
            "validity_score_confused": False,
            "genre_guessed_reveal": False,
            "irreversible_progress_loss": False,
        })
    return {
        "format": analyzer.FORMAT,
        "bundle": BUNDLE,
        "system": {
            "participant_isolation_verified": True,
            "kernel_receipt_agreement": True,
            "all_cases_reachable": True,
            "all_cases_precise_diagnostics": True,
            "acceptance_deterministic": True,
            "reset_replay_lossless": True,
            "offline_verified": True,
            "verifier_cheaper_than_search": True,
            "semantic_invalidated_cases": 0,
        },
        "teams": teams,
    }


class PlaytestAnalysisTests(unittest.TestCase):
    def test_private_draft_generation_balances_cohorts_and_reports_every_blank(self) -> None:
        with tempfile.TemporaryDirectory(prefix="afterimage-playtest-draft-") as temporary:
            root = Path(temporary)
            draft = root / "campaign.draft.json"
            command = [
                sys.executable,
                str(ROOT / "tools" / "analyze_playtest.py"),
                "--new",
                str(draft),
                "--bundle",
                BUNDLE,
                "--teams",
                "9",
            ]
            completed = subprocess.run(command, check=False, capture_output=True)
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(draft.stat().st_mode & 0o777, 0o600)
            value = json.loads(draft.read_bytes())
            self.assertEqual(len(value["teams"]), 9)
            self.assertEqual(
                {cohort: sum(team["cohort"] == cohort for team in value["teams"]) for cohort in analyzer.COHORTS},
                {cohort: 3 for cohort in analyzer.COHORTS},
            )
            ids = [team["id"] for team in value["teams"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(team_id.startswith("T-") and team_id[2:].isdigit() for team_id in ids))
            missing = analyzer.unrecorded_paths(value)
            self.assertGreater(len(missing), 100)
            with self.assertRaises(analyzer.CampaignError) as captured:
                analyzer.analyze(value)
            self.assertEqual(captured.exception.code, "campaign_incomplete")
            self.assertEqual(captured.exception.context["count"], len(missing))

            canonical = root / "campaign.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "analyze_playtest.py"),
                    str(draft),
                    "--canonicalize",
                    str(canonical),
                ],
                check=False,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 2)
            error = json.loads(completed.stdout)
            self.assertEqual(error["code"], "campaign_incomplete")
            self.assertEqual(error["context"]["count"], len(missing))
            self.assertFalse(canonical.exists())

            repeated = subprocess.run(command, check=False, capture_output=True)
            self.assertEqual(repeated.returncode, 2)
            self.assertEqual(json.loads(repeated.stdout)["code"], "campaign_output")

        for count in (5, 65, True):
            with self.subTest(invalid_team_count=count):
                with self.assertRaises(analyzer.CampaignError) as captured:
                    analyzer.new_campaign_draft(BUNDLE, count)
                self.assertEqual(captured.exception.code, "campaign_schema")
        with self.assertRaises(analyzer.CampaignError):
            analyzer.new_campaign_draft("sha256:not-a-bundle", 6)

    def test_complete_six_team_campaign_passes_with_exact_aggregates(self) -> None:
        result = analyzer.analyze(campaign())
        self.assertEqual(result["decision"], "pass")
        self.assertEqual(result["hard"]["first_receipt_median"]["value"], {"numerator": 75, "denominator": 2})
        self.assertEqual(result["hard"]["first_receipt_p90"]["value"], 60)
        self.assertEqual(result["hard"]["independent_family_solutions"]["value"], sorted(FAMILIES))
        self.assertFalse(any(item["triggered"] for item in result["stop_triggers"].values()))
        changed = campaign()
        changed["teams"][0]["first_receipt_minutes"] += 1
        self.assertNotEqual(result["campaign"], analyzer.analyze(changed)["campaign"])

    def test_failed_criterion_revises_and_larger_campaign_scales_ratios(self) -> None:
        value = campaign()
        value["teams"][-1]["first_receipt_minutes"] = None
        self.assertEqual(analyzer.analyze(value)["decision"], "revise")
        self.assertIsNone(analyzer.analyze(value)["hard"]["first_receipt_p90"]["value"])

        scaled = campaign()
        additions = copy.deepcopy(scaled["teams"])
        for index, team in enumerate(additions, 7):
            team["id"] = f"T-{index:04d}"
        scaled["teams"].extend(additions)
        for team in scaled["teams"][7:]:
            team["desk_boot_minutes"] = None
        result = analyzer.analyze(scaled)
        self.assertEqual(result["decision"], "revise")
        self.assertEqual(result["hard"]["desk_boot"]["threshold"], ">=8 teams within 300 minutes")
        self.assertEqual(result["hard"]["desk_boot"]["value"], 7)

    def test_stop_trigger_takes_precedence_over_other_results(self) -> None:
        value = campaign()
        value["teams"][0]["canonicalization_dominated"] = True
        result = analyzer.analyze(value)
        self.assertEqual(result["decision"], "stop")
        self.assertTrue(result["stop_triggers"]["canonicalization_dominated"]["triggered"])

    def test_every_stop_and_redesign_trigger_is_wired(self) -> None:
        mutations = {
            "cre_stall": lambda value: [team.update({"cre_minutes": 240, "desk_boot_minutes": None}) for team in value["teams"][:2]],
            "canonicalization_dominated": lambda value: value["teams"][0].update({"canonicalization_dominated": True}),
            "second_language_required": lambda value: value["teams"][0].update({"pulse_or_lens_required_second_language": True}),
            "cascade_blind_search": lambda value: value["teams"][0].update({"cascade_blind_search": True}),
            "validity_score_confused": lambda value: value["teams"][0].update({"validity_score_confused": True}),
            "genre_guessed_reveal": lambda value: value["teams"][0].update({"genre_guessed_reveal": True}),
            "verifier_too_expensive": lambda value: value["system"].update({"verifier_cheaper_than_search": False}),
            "semantic_blast_radius": lambda value: value["system"].update({"semantic_invalidated_cases": 3}),
        }
        for expected, mutate in mutations.items():
            with self.subTest(trigger=expected):
                value = campaign()
                mutate(value)
                result = analyzer.analyze(value)
                self.assertEqual(result["decision"], "stop")
                self.assertTrue(result["stop_triggers"][expected]["triggered"])

    def test_exact_schema_rejects_unknown_personal_and_malformed_fields(self) -> None:
        unknown = campaign()
        unknown["teams"][0]["notes"] = "private terminal output"
        with self.assertRaises(analyzer.CampaignError) as captured:
            analyzer.analyze(unknown)
        self.assertEqual(captured.exception.code, "campaign_schema")

        personal = campaign()
        personal["teams"][0]["id"] = "T-Alice"
        with self.assertRaises(analyzer.CampaignError):
            analyzer.analyze(personal)

        free_text = campaign()
        free_text["teams"][0]["engine_language"] = "my custom runtime by Alice"
        with self.assertRaises(analyzer.CampaignError):
            analyzer.analyze(free_text)

        boolean_integer = campaign()
        boolean_integer["teams"][0]["first_receipt_minutes"] = True
        with self.assertRaises(analyzer.CampaignError):
            analyzer.analyze(boolean_integer)

    def test_cli_requires_canonical_input_and_emits_canonical_decision(self) -> None:
        with tempfile.TemporaryDirectory(prefix="afterimage-playtest-analysis-") as temporary:
            root = Path(temporary)
            valid = root / "campaign.json"
            valid.write_bytes(cre.canonical_bytes(campaign()))
            command = [sys.executable, str(ROOT / "tools" / "analyze_playtest.py"), str(valid)]
            completed = subprocess.run(command, check=False, capture_output=True)
            self.assertEqual(completed.returncode, 0)
            result = json.loads(completed.stdout)
            self.assertEqual(result["decision"], "pass")
            self.assertEqual(completed.stdout, cre.canonical_bytes(result) + b"\n")

            invalid = root / "pretty.json"
            invalid.write_text(json.dumps(campaign(), indent=2), encoding="utf-8")
            completed = subprocess.run(command[:-1] + [str(invalid)], check=False, capture_output=True)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["code"], "noncanonical_json")

            canonicalized = root / "canonicalized.json"
            completed = subprocess.run(
                command[:-1] + [str(invalid), "--canonicalize", str(canonicalized)],
                check=False,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(canonicalized.read_bytes(), cre.canonical_bytes(campaign()))
            canonicalized_result = json.loads(completed.stdout)
            self.assertEqual(canonicalized_result["campaign"], result["campaign"])
            repeated = subprocess.run(
                command[:-1] + [str(invalid), "--canonicalize", str(canonicalized)],
                check=False,
                capture_output=True,
            )
            self.assertEqual(repeated.returncode, 2)
            self.assertEqual(json.loads(repeated.stdout)["code"], "campaign_output")


if __name__ == "__main__":
    unittest.main()
