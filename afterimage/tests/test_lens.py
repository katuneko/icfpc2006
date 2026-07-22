#!/usr/bin/env python3
"""Focused finite-law tests for LENS 0.1."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import lens  # noqa: E402


SOURCE = json.loads((ROOT / "content/vertical_slice/cases/LENS.001.json").read_text(encoding="utf-8"))
CONTRACT = SOURCE["case"]["lens_contract"]
PROGRAM = SOURCE["case"]["author_program"]
TIMETABLE_SOURCE = json.loads((ROOT / "content/production/cases/LENS.002.json").read_text(encoding="utf-8"))
TIMETABLE_CONTRACT = TIMETABLE_SOURCE["case"]["lens_contract"]
TIMETABLE_PROGRAM = TIMETABLE_SOURCE["case"]["author_program"]
HISTORY_SOURCE = json.loads((ROOT / "content/production/cases/LENS.003.json").read_text(encoding="utf-8"))
HISTORY_CONTRACT = HISTORY_SOURCE["case"]["lens_contract"]
HISTORY_PROGRAM = HISTORY_SOURCE["case"]["author_program"]


class LensTests(unittest.TestCase):
    def test_author_program_satisfies_complete_bounded_laws(self) -> None:
        self.assertEqual(len(lens.source_domain(CONTRACT)), 72)
        self.assertEqual(len(lens.target_domain(CONTRACT)), 15)
        self.assertEqual(lens.verify_program(PROGRAM, CONTRACT), {"program_nodes": 32, "auxiliary_schema_cells": 3, "worst_reductions": 8, "domain_sources": 72, "domain_targets": 15})

    def test_boundary_disambiguation_is_required_by_getput(self) -> None:
        altered = copy.deepcopy(PROGRAM)
        altered["complement_schema"] = [item for item in altered["complement_schema"] if item["name"] != "boundary_street"]
        with self.assertRaises(lens.LensError) as raised:
            lens.verify_program(altered, CONTRACT)
        self.assertEqual(raised.exception.code, "lens_counterexample")
        self.assertEqual(raised.exception.context["law"], "GetPut")
        self.assertIn(raised.exception.context["source"]["street_name"], {"North Quay", "South Quay"})

    def test_unrepresented_unit_cannot_be_dropped(self) -> None:
        altered = copy.deepcopy(PROGRAM)
        altered["complement_schema"] = [item for item in altered["complement_schema"] if item["name"] != "unit"]
        with self.assertRaises(lens.LensError) as raised:
            lens.verify_program(altered, CONTRACT)
        self.assertEqual(raised.exception.code, "lens_counterexample")
        self.assertIn(raised.exception.context["law"], {"GetPut", "Provenance"})

    def test_timetable_program_satisfies_complete_bounded_laws(self) -> None:
        self.assertEqual(len(lens.source_domain(TIMETABLE_CONTRACT)), 96)
        self.assertEqual(len(lens.target_domain(TIMETABLE_CONTRACT)), 9)
        self.assertEqual(lens.verify_program(TIMETABLE_PROGRAM, TIMETABLE_CONTRACT), {
            "program_nodes": 36, "auxiliary_schema_cells": 4,
            "worst_reductions": 6, "domain_sources": 96, "domain_targets": 9,
        })

    def test_timetable_service_identity_is_required_by_getput(self) -> None:
        altered = copy.deepcopy(TIMETABLE_PROGRAM)
        altered["complement_schema"] = [item for item in altered["complement_schema"] if item["name"] != "service_key"]
        with self.assertRaises(lens.LensError) as raised:
            lens.verify_program(altered, TIMETABLE_CONTRACT)
        self.assertEqual(raised.exception.code, "lens_counterexample")
        self.assertEqual(raised.exception.context["law"], "GetPut")
        self.assertIn(raised.exception.context["source"]["service_key"], {"north-am", "south-am"})

    def test_timetable_hidden_operational_fields_cannot_be_dropped(self) -> None:
        for field in ("platform", "calendar", "schedule_provenance"):
            altered = copy.deepcopy(TIMETABLE_PROGRAM)
            altered["put"][2]["fields"].remove(field)
            with self.assertRaises(lens.LensError) as raised:
                lens.verify_program(altered, TIMETABLE_CONTRACT)
            self.assertEqual(raised.exception.code, "lens_counterexample", field)
            self.assertIn(raised.exception.context["law"], {"GetPut", "Provenance"}, field)

    def test_timetable_invalid_edit_is_atomic(self) -> None:
        compiled = lens.compile_program(TIMETABLE_PROGRAM, lens.validate_contract(TIMETABLE_CONTRACT))
        source = lens.source_domain(TIMETABLE_CONTRACT)[0]
        before = copy.deepcopy(source)
        for target in TIMETABLE_CONTRACT["invalid_targets"]:
            observed, _reductions = lens.put_view(compiled, TIMETABLE_CONTRACT, source, target)
            self.assertIsNone(observed)
            self.assertEqual(source, before)

    def test_history_program_satisfies_complete_bounded_laws(self) -> None:
        self.assertEqual(len(lens.source_domain(HISTORY_CONTRACT)), 48)
        self.assertEqual(len(lens.target_domain(HISTORY_CONTRACT)), 9)
        self.assertEqual(lens.verify_program(HISTORY_PROGRAM, HISTORY_CONTRACT), {
            "program_nodes": 32, "auxiliary_schema_cells": 3,
            "worst_reductions": 6, "domain_sources": 48, "domain_targets": 9,
        })

    def test_history_identity_and_private_evidence_are_independent(self) -> None:
        no_identity = copy.deepcopy(HISTORY_PROGRAM)
        no_identity["complement_schema"] = [cell for cell in no_identity["complement_schema"] if cell["name"] != "history_key"]
        with self.assertRaises(lens.LensError) as raised:
            lens.verify_program(no_identity, HISTORY_CONTRACT)
        self.assertEqual(raised.exception.context["law"], "GetPut")

        for field in ("private_delta", "audit_chain"):
            altered = copy.deepcopy(HISTORY_PROGRAM)
            altered["put"][2]["fields"].remove(field)
            with self.assertRaises(lens.LensError) as raised:
                lens.verify_program(altered, HISTORY_CONTRACT)
            self.assertIn(raised.exception.context["law"], {"GetPut", "Provenance"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
