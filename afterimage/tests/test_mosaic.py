#!/usr/bin/env python3
"""Focused tests for MOSAIC D4 graph-certificate validation."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import mosaic  # noqa: E402


SOURCE = json.loads((ROOT / "content/vertical_slice/cases/MOSAIC.001.json").read_text(encoding="utf-8"))
FRAGMENTS = [event["payload"] for event in SOURCE["events"]]
ANSWER = SOURCE["case"]["author_answer"]
CONTRACT = SOURCE["case"]["mosaic_contract"]
SHARED_SOURCE = json.loads((ROOT / "content/production/cases/MOSAIC.002.json").read_text(encoding="utf-8"))
SHARED_FRAGMENTS = [event["payload"] for event in SHARED_SOURCE["events"]]
SHARED_ANSWER = SHARED_SOURCE["case"]["author_answer"]
SHARED_CONTRACT = SHARED_SOURCE["case"]["mosaic_contract"]
MISSING_SOURCE = json.loads((ROOT / "content/production/cases/MOSAIC.004.json").read_text(encoding="utf-8"))
MISSING_FRAGMENTS = [event["payload"] for event in MISSING_SOURCE["events"]]
MISSING_ANSWER = MISSING_SOURCE["case"]["author_answer"]
MISSING_CONTRACT = MISSING_SOURCE["case"]["mosaic_contract"]
DUPLICATE_SOURCE = json.loads((ROOT / "content/production/cases/MOSAIC.005.json").read_text(encoding="utf-8"))
DUPLICATE_FRAGMENTS = [event["payload"] for event in DUPLICATE_SOURCE["events"]]
DUPLICATE_ANSWER = DUPLICATE_SOURCE["case"]["author_answer"]
DUPLICATE_CONTRACT = DUPLICATE_SOURCE["case"]["mosaic_contract"]
LANDMARK_SOURCE = json.loads((ROOT / "content/production/cases/MOSAIC.006.json").read_text(encoding="utf-8"))
LANDMARK_FRAGMENTS = [event["payload"] for event in LANDMARK_SOURCE["events"]]
LANDMARK_ANSWER = LANDMARK_SOURCE["case"]["author_answer"]
LANDMARK_CONTRACT = LANDMARK_SOURCE["case"]["mosaic_contract"]
TIMESTAMP_SOURCE = json.loads((ROOT / "content/production/cases/MOSAIC.007.json").read_text(encoding="utf-8"))
TIMESTAMP_FRAGMENTS = [event["payload"] for event in TIMESTAMP_SOURCE["events"]]
TIMESTAMP_ANSWER = TIMESTAMP_SOURCE["case"]["author_answer"]
TIMESTAMP_CONTRACT = TIMESTAMP_SOURCE["case"]["mosaic_contract"]
RING_SOURCE = json.loads((ROOT / "content/production/cases/MOSAIC.008.json").read_text(encoding="utf-8"))
RING_FRAGMENTS = [event["payload"] for event in RING_SOURCE["events"]]
RING_ANSWER = RING_SOURCE["case"]["author_answer"]
RING_CONTRACT = RING_SOURCE["case"]["mosaic_contract"]
NOISY_SOURCE = json.loads((ROOT / "content/production/cases/MOSAIC.009.json").read_text(encoding="utf-8"))
NOISY_FRAGMENTS = [event["payload"] for event in NOISY_SOURCE["events"]]
NOISY_ANSWER = NOISY_SOURCE["case"]["author_answer"]
NOISY_CONTRACT = NOISY_SOURCE["case"]["mosaic_contract"]


class MosaicTests(unittest.TestCase):
    def test_author_certificate_covers_grid_and_rejects_only_decoy(self) -> None:
        self.assertEqual(mosaic.validate_answer(FRAGMENTS, ANSWER, CONTRACT), {"unexplained_weight": 1, "graph_size": 21, "certificate_units": 11})

    def test_rotated_isomorph_is_rejected_before_mapping(self) -> None:
        altered = copy.deepcopy(ANSWER)
        old_to_new = {}
        for index in range(9):
            x, y = index % 3, index // 3
            nx, ny = 2 - y, x
            old_to_new[f"v{index}"] = f"v{ny * 3 + nx}"
        for edge in altered["global"]["edges"]:
            edge["a"], edge["b"] = old_to_new[edge["a"]], old_to_new[edge["b"]]
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(FRAGMENTS, altered, CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_noncanonical")

    def test_embeddable_fragment_cannot_be_called_a_decoy(self) -> None:
        altered = copy.deepcopy(ANSWER)
        removed = altered["used"].pop(0)
        altered["unused"].append({"fragment": removed["fragment"], "reason": "invariant_conflict"})
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(FRAGMENTS, altered, CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_classification")

    def test_shared_wall_requires_two_independently_supported_edges(self) -> None:
        metrics = mosaic.validate_answer(SHARED_FRAGMENTS, SHARED_ANSWER, SHARED_CONTRACT)
        self.assertEqual(metrics["unexplained_weight"], 0)
        self.assertEqual(metrics["graph_size"], 21)

        insufficient = copy.deepcopy(SHARED_FRAGMENTS)
        south = next(fragment for fragment in insufficient if fragment["id"] == "F-SOUTH")
        south["edges"] = [
            edge for edge in south["edges"]
            if {edge["a"], edge["b"]} != {"s11", "s21"}
        ]
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(insufficient, SHARED_ANSWER, SHARED_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_overlap")
        self.assertEqual(raised.exception.context, {"observed": 1, "required": 2})

    def test_shared_wall_policy_is_stricter_than_union_coverage(self) -> None:
        stricter = copy.deepcopy(SHARED_CONTRACT)
        stricter["coverage"]["minimum_shared_edges"] = 3
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(SHARED_FRAGMENTS, SHARED_ANSWER, stricter)
        self.assertEqual(raised.exception.code, "mosaic_overlap")

    def test_missing_tile_certificate_matches_the_exact_interior_hole(self) -> None:
        self.assertEqual(
            mosaic.validate_answer(MISSING_FRAGMENTS, MISSING_ANSWER, MISSING_CONTRACT),
            {"unexplained_weight": 0, "graph_size": 21, "certificate_units": 10},
        )
        self.assertEqual(MISSING_ANSWER["missing"], {
            "vertices": ["v4"],
            "edges": [
                {"a": "v1", "b": "v4"}, {"a": "v3", "b": "v4"},
                {"a": "v4", "b": "v5"}, {"a": "v4", "b": "v7"},
            ],
        })

    def test_missing_tile_checksum_fixes_inferred_edge_attributes(self) -> None:
        altered = copy.deepcopy(MISSING_ANSWER)
        inferred = next(edge for edge in altered["global"]["edges"] if {edge["a"], edge["b"]} == {"v1", "v4"})
        inferred["material"] = "plausible-but-invented"
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(MISSING_FRAGMENTS, altered, MISSING_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_checksum")
        self.assertEqual(raised.exception.context["expected"], ["v-01-04", "2"])

    def test_missing_tile_cannot_hide_an_observed_perimeter_gap(self) -> None:
        altered_fragments = copy.deepcopy(MISSING_FRAGMENTS)
        ring_a = next(fragment for fragment in altered_fragments if fragment["id"] == "F-RING-A")
        ring_a["edges"] = [
            edge for edge in ring_a["edges"] if {edge["a"], edge["b"]} != {"a0", "a3"}
        ]
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(altered_fragments, MISSING_ANSWER, MISSING_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_omission")

    def test_double_exposure_accepts_one_d4_duplicate_survivor(self) -> None:
        self.assertEqual(
            mosaic.validate_answer(DUPLICATE_FRAGMENTS, DUPLICATE_ANSWER, DUPLICATE_CONTRACT),
            {"unexplained_weight": 1, "graph_size": 21, "certificate_units": 8},
        )
        fragments = [mosaic.validate_fragment(value) for value in DUPLICATE_FRAGMENTS]
        self.assertEqual(mosaic.fragment_fingerprint(fragments[0]), mosaic.fragment_fingerprint(fragments[2]))

    def test_double_exposure_cannot_count_duplicate_as_corroboration(self) -> None:
        altered = copy.deepcopy(DUPLICATE_ANSWER)
        altered["used"].append({
            "fragment": "F-BRIDGE-NORTH-CACHE", "transform": "r180",
            "mapping": [
                {"local": "d00", "global": "v5"}, {"local": "d10", "global": "v4"},
                {"local": "d20", "global": "v3"}, {"local": "d01", "global": "v2"},
                {"local": "d11", "global": "v1"}, {"local": "d21", "global": "v0"},
            ],
        })
        altered["unused"].clear()
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(DUPLICATE_FRAGMENTS, altered, DUPLICATE_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_duplicate")

    def test_double_exposure_duplicate_link_must_match_attributed_fingerprint(self) -> None:
        wrong_link = copy.deepcopy(DUPLICATE_ANSWER)
        wrong_link["unused"][0]["duplicate_of"] = "F-BRIDGE-SOUTH"
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(DUPLICATE_FRAGMENTS, wrong_link, DUPLICATE_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_duplicate")

        changed_attribute = copy.deepcopy(DUPLICATE_FRAGMENTS)
        cache = next(fragment for fragment in changed_attribute if fragment["id"] == "F-BRIDGE-NORTH-CACHE")
        cache["edges"][0]["level"] = "invented"
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(changed_attribute, DUPLICATE_ANSWER, DUPLICATE_CONTRACT)
        self.assertEqual(raised.exception.code, "invalid_case")

        wrong_hole = copy.deepcopy(MISSING_ANSWER)
        wrong_hole["missing"]["vertices"] = ["v0"]
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(MISSING_FRAGMENTS, wrong_hole, MISSING_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_omission")

    def test_false_landmark_has_geometric_support_but_no_anchored_embedding(self) -> None:
        self.assertEqual(
            mosaic.validate_answer(LANDMARK_FRAGMENTS, LANDMARK_ANSWER, LANDMARK_CONTRACT),
            {"unexplained_weight": 1, "graph_size": 21, "certificate_units": 8},
        )
        corrected = copy.deepcopy(LANDMARK_CONTRACT)
        corrected["landmarks"]["anchors"][0]["global"] = "v0"
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(LANDMARK_FRAGMENTS, LANDMARK_ANSWER, corrected)
        self.assertEqual(raised.exception.code, "mosaic_landmark")

    def test_false_landmark_cannot_be_used_against_its_public_anchor(self) -> None:
        altered = copy.deepcopy(LANDMARK_ANSWER)
        altered["unused"].clear()
        altered["used"].append({
            "fragment": "F-FALSE-LANDMARK", "transform": "r0",
            "mapping": [
                {"local": "d00", "global": "v0"}, {"local": "d10", "global": "v1"},
                {"local": "d01", "global": "v3"}, {"local": "d11", "global": "v4"},
            ],
        })
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(LANDMARK_FRAGMENTS, altered, LANDMARK_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_landmark")

    def test_timestamp_gauge_resolves_geometrically_valid_translation(self) -> None:
        self.assertEqual(
            mosaic.validate_answer(TIMESTAMP_FRAGMENTS, TIMESTAMP_ANSWER, TIMESTAMP_CONTRACT),
            {"unexplained_weight": 0, "graph_size": 21, "certificate_units": 8},
        )
        self.assertEqual(
            mosaic.validate_answer(TIMESTAMP_FRAGMENTS, TIMESTAMP_ANSWER, {"width": 3, "height": 3}),
            {"unexplained_weight": 0, "graph_size": 21, "certificate_units": 8},
        )

        shifted = copy.deepcopy(TIMESTAMP_ANSWER)
        shifted["used"][0]["mapping"] = [
            {"local": "w00", "global": "v1"}, {"local": "w10", "global": "v2"},
            {"local": "w01", "global": "v4"}, {"local": "w11", "global": "v5"},
            {"local": "w02", "global": "v7"}, {"local": "w12", "global": "v8"},
        ]
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(TIMESTAMP_FRAGMENTS, shifted, TIMESTAMP_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_timestamp")

    def test_timestamp_gauge_rejects_one_forged_scan_tick(self) -> None:
        forged = copy.deepcopy(TIMESTAMP_FRAGMENTS)
        forged[0]["vertices"][3]["observed_at"] += 1
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(forged, TIMESTAMP_ANSWER, TIMESTAMP_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_timestamp")

    def test_broken_ring_recovers_exactly_the_observed_gap(self) -> None:
        self.assertEqual(
            mosaic.validate_answer(RING_FRAGMENTS, RING_ANSWER, RING_CONTRACT),
            {"unexplained_weight": 0, "graph_size": 21, "certificate_units": 8},
        )
        wrong = copy.deepcopy(RING_ANSWER)
        wrong["missing"]["edges"] = [{"a": "v6", "b": "v7"}]
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(RING_FRAGMENTS, wrong, RING_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_ring")

    def test_broken_ring_cannot_invent_a_non_ring_gap(self) -> None:
        contract = copy.deepcopy(RING_CONTRACT)
        contract["ring_recovery"]["cycle"] = ["v0", "v1", "v4", "v3"]
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(RING_FRAGMENTS, RING_ANSWER, contract)
        self.assertEqual(raised.exception.code, "mosaic_ring")

    def test_adversarial_survey_is_geometric_but_outvoted(self) -> None:
        self.assertEqual(mosaic.validate_answer(NOISY_FRAGMENTS, NOISY_ANSWER, NOISY_CONTRACT), {
            "unexplained_weight": 2, "graph_size": 21, "certificate_units": 8,
        })
        weakened = copy.deepcopy(NOISY_FRAGMENTS)
        next(fragment for fragment in weakened if fragment["id"] == "F9-NORTH-SIGNED")["weight"] = 4
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(weakened, NOISY_ANSWER, NOISY_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_weight")

    def test_adversarial_survey_needs_outvoted_not_invariant_conflict(self) -> None:
        altered = copy.deepcopy(NOISY_ANSWER)
        altered["unused"][0]["reason"] = "invariant_conflict"
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(NOISY_FRAGMENTS, altered, NOISY_CONTRACT)
        self.assertEqual(raised.exception.code, "mosaic_classification")


    def test_two_layer_topology_has_exact_connected_portals(self) -> None:
        source = json.loads((ROOT / "content/production/cases/MOSAIC.010.json").read_text(encoding="utf-8"))
        fragments = [event["payload"] for event in source["events"]]
        self.assertEqual(mosaic.validate_answer(fragments, source["case"]["author_answer"], source["case"]["mosaic_contract"]), {
            "unexplained_weight": 0, "graph_size": 21, "certificate_units": 8,
        })

    def test_two_layer_topology_rejects_false_portal_material(self) -> None:
        source = json.loads((ROOT / "content/production/cases/MOSAIC.010.json").read_text(encoding="utf-8"))
        fragments = [event["payload"] for event in source["events"]]
        contract = copy.deepcopy(source["case"]["mosaic_contract"])
        contract["layers"]["portal_material"] = "painted-symbol"
        with self.assertRaises(mosaic.MosaicError) as raised:
            mosaic.validate_answer(fragments, source["case"]["author_answer"], contract)
        self.assertEqual(raised.exception.code, "mosaic_layer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
