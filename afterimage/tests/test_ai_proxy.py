from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import analyze_ai_proxy as proxy  # noqa: E402
import analyze_playtest as human  # noqa: E402


class AIProxyTests(unittest.TestCase):
    def campaign(self) -> dict:
        system = {key: True for key in human.SYSTEM_KEYS}
        system["semantic_invalidated_cases"] = 0
        conditions = []
        for index, (code, assignment) in enumerate(proxy.MATRIX.items()):
            track, effort, cohort, language = assignment
            observation = {
                key: False
                for key in proxy.OBSERVATION_KEYS
                if key not in {"max_hint_level", "route_count", "independent_valid_families", "unrelated_families"}
            }
            observation.update(
                {
                    "reached_cascade003": True,
                    "max_hint_level": 2,
                    "projection_explained": True,
                    "intended_observations_understood": True,
                    "independent_valid_families": sorted(human.SLICE_FAMILIES),
                    "improved_valid_score": True,
                    "unrelated_families": [],
                    "computed_reveal": True,
                    "route_count": 2,
                }
            )
            start = 1_000_000 + index * 1_000
            conditions.append(
                {
                    "code": code,
                    "track": track,
                    "effort": effort,
                    "cohort": cohort,
                    "engine_language": language,
                    "timing": {
                        "start": start,
                        "understood": start + 30,
                        "first_bounded_run": start + 60,
                        "session_a_stop": start + 120,
                        "desk_boot": start + 180,
                        "first_receipt": start + 240,
                        "cascade003": start + 300,
                        "stop": start + 360,
                    },
                    "engine": {
                        "conformance_pass": True,
                        "vectors_passed": 9,
                        "cases_passed": 37,
                        "protocol_success_exercised": True,
                        "ambiguities": [],
                        "unclassified_error": None,
                    },
                    "observation": observation,
                }
            )
        return {
            "format": proxy.FORMAT,
            "bundle": "sha256:" + "1" * 64,
            "method": copy.deepcopy(proxy.METHOD),
            "system": system,
            "conditions": conditions,
        }

    def test_proxy_pass_preserves_limitations_and_sensitivity(self) -> None:
        result = proxy.analyze(self.campaign())
        self.assertEqual(result["decision"], "proxy-pass")
        self.assertFalse(result["model_selector_available"])
        self.assertFalse(result["native_effort_available"])
        self.assertEqual(result["central"]["decision"], "pass")
        self.assertEqual(result["sensitivity"]["0.75x"]["decision"], "pass")
        self.assertEqual(result["sensitivity"]["1.5x"]["decision"], "revise")
        self.assertTrue(result["low_effort_safe"]["passed"])

    def test_low_effort_failure_cannot_be_hidden_by_aggregate_pass(self) -> None:
        campaign = self.campaign()
        campaign["conditions"][0]["observation"]["improved_valid_score"] = False
        result = proxy.analyze(campaign)
        self.assertEqual(result["central"]["decision"], "pass")
        self.assertEqual(result["decision"], "proxy-revise")
        self.assertEqual(result["low_effort_safe"]["failures"], ["PX-1001.improved_valid_score"])

    def test_matrix_and_method_are_frozen(self) -> None:
        campaign = self.campaign()
        campaign["method"]["native_effort_available"] = True
        with self.assertRaises(proxy.ProxyError):
            proxy.analyze(campaign)


if __name__ == "__main__":
    unittest.main()
