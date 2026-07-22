#!/usr/bin/env python3
"""Generate or check the complete intentional Afterimage production catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manifests" / "production_catalog.json"

PLANNED: dict[str, list[tuple[str, int, str, str]]] = {
    "CASCADE": [
        ("Cold Platform", 70, "local failures", "delayed occupancy evidence"),
        ("Priority Queue", 70, "local failures", "priority inversion intervention"),
        ("Black Ice", 80, "local failures", "treatment timing and road closure"),
        ("Borrowed Pressure", 80, "coupled infrastructure", "water and power coupling"),
        ("Island Signal", 90, "coupled infrastructure", "microgrid and traffic coupling"),
        ("Last Train Through", 90, "coupled infrastructure", "transit evacuation window"),
        ("Cooling Window", 100, "coupled infrastructure", "hospital cooling dependencies"),
        ("The Open Bridge", 110, "coupled infrastructure", "wind, bridge, and emergency access"),
        ("Three Departments", 110, "coupled infrastructure", "multi-agency contract repair"),
        ("Silent Gauge", 90, "missing observations", "safe repair under a hidden gauge"),
        ("Shadow Bus", 100, "missing observations", "latent transit capacity"),
        ("Redacted Feeder", 100, "missing observations", "suppressed grid provenance"),
        ("Witness Gap", 110, "missing observations", "bounded missing-cause proof"),
        ("Policy Blind Spot", 110, "missing observations", "projection-sensitive safety"),
        ("The Missing Cause", 110, "missing observations", "minimal latent explanation"),
        ("Correction Notice", 90, "public record", "publishable counterfactual correction"),
        ("Competing Amendments", 100, "public record", "lexicographic policy choice"),
        ("Audit Window", 110, "public record", "time-bounded public evidence"),
        ("Rule of Least Change", 110, "public record", "global minimal intervention"),
        ("Continuity Hearing", 120, "public record", "cross-family evidence envelope"),
        ("The Chosen Tomorrow", 130, "public record", "expose the policy-selected history"),
    ],
    "MERGE": [
        ("Duplicate Dispatch", 80, "record faults", "duplicate suppression certificate"),
        ("Lost Acknowledgment", 90, "record faults", "missing edge reconstruction"),
        ("Compensating Transfer", 90, "record faults", "compensation versus rollback"),
        ("Split Brain", 100, "conflicts", "two-writer conflict component"),
        ("Quorum Ledger", 100, "conflicts", "quorum-supported acceptance"),
        ("Clock Islands", 110, "conflicts", "multiple offset domains"),
        ("Conflict Component", 110, "certificates", "minimal inconsistent subset"),
        ("Minimal Cut", 120, "certificates", "weighted rejection cut"),
        ("Echoed Update", 120, "certificates", "transitive duplication evidence"),
        ("Evidence Weight", 130, "adversarial logs", "source-weighted reconstruction"),
        ("Causal Compression", 130, "adversarial logs", "small sufficient certificate"),
        ("Equivocation", 130, "adversarial logs", "one source, incompatible claims"),
        ("Two Consistent Archives", 130, "adversarial logs", "non-unique reconstruction"),
        ("The Reconstruction", 140, "adversarial logs", "citywide causal archive"),
    ],
    "PULSE": [
        ("First Copy", 80, "stream hygiene", "canonical first-event deduplication"),
        ("Silent Timeout", 90, "stream hygiene", "timeout with cancellation"),
        ("Token Window", 100, "flow control", "bounded token-bucket admission"),
        ("Two of Three", 100, "coordination", "quorum trigger"),
        ("Barrier at Dawn", 110, "coordination", "multi-topic barrier"),
        ("Backpressure", 110, "flow control", "bounded queue feedback"),
        ("Warm Failover", 120, "resilience", "primary-to-secondary handoff"),
        ("Exactly Once", 120, "resilience", "deduplication across failover"),
        ("Shared Deadline", 130, "coordination", "deadline aggregation"),
        ("Burst Budget", 130, "flow control", "sliding-window rate limit"),
        ("Reorder Buffer", 140, "resilience", "bounded out-of-order release"),
        ("Circuit Breaker", 150, "resilience", "failure-window state machine"),
        ("City Clock", 170, "synthesis", "composed deterministic controller"),
    ],
    "MOSAIC": [
        ("Shared Wall", 80, "clean fragments", "edge overlap assembly"),
        ("Rotated Block", 90, "clean fragments", "D4-normalized placement"),
        ("Missing Tile", 100, "omissions", "hole-bounded completion"),
        ("Double Exposure", 100, "duplicates", "duplicate fragment attribution"),
        ("False Landmark", 110, "decoys", "supported decoy rejection"),
        ("Timestamp Gauge", 120, "mixed evidence", "topology plus temporal labels"),
        ("Broken Ring", 130, "mixed evidence", "cycle recovery with one omission"),
        ("Adversarial Survey", 140, "adversarial", "weighted noisy embeddings"),
        ("Underground Layer", 150, "adversarial", "two-layer topology"),
        ("Whole City", 150, "synthesis", "global attributed reconstruction"),
    ],
    "LENS": [
        ("One Timetable", 500, "transit", "schedule complement and lawful updates"),
        ("Two Histories", 620, "history", "bidirectional divergent-history synchronization"),
    ],
    "COVENANT": [
        ("Dispatch Covenant", 450, "emergency dispatch", "local policy under fair schedules"),
        ("City Covenant", 550, "citywide", "hidden-observation policy synthesis"),
    ],
    "PARADOX": [
        ("Two Tomorrows", 300, "finale", "paired safe histories with one public record"),
    ],
}


def requirement(case_id: str) -> dict[str, Any]:
    family, number_text = case_id.split(".")
    number = int(number_text)
    special = {
        "PULSE.002": {"all": ["case:PULSE.001", "case:ORIENT.003"]},
        "MOSAIC.002": {"all": ["case:MOSAIC.001", "case:ORIENT.004"]},
        "MOSAIC.004": {"all": ["case:MOSAIC.003", "case:CASCADE.010"]},
        "MERGE.006": {"all": ["case:MERGE.005", "case:PULSE.005", "case:CASCADE.010"]},
        "MERGE.007": {"all": ["case:MERGE.006", "case:CASCADE.011"]},
        "MOSAIC.005": {"all": ["case:MOSAIC.004", "case:CASCADE.011", "case:MERGE.007"]},
        "PULSE.006": {"all": ["case:PULSE.005", "case:CASCADE.011", "case:MERGE.007", "case:MOSAIC.005"]},
        "CASCADE.012": {"all": ["case:CASCADE.011", "case:PULSE.006"]},
        "MERGE.008": {"all": ["case:MERGE.007", "case:CASCADE.012"]},
        "MOSAIC.006": {"all": ["case:MOSAIC.005", "case:MERGE.008"]},
        "PULSE.007": {"all": ["case:PULSE.006", "case:CASCADE.012", "case:MOSAIC.006"]},
        "LENS.002": {"all": ["case:LENS.001", "case:PULSE.005", "case:MOSAIC.005", "case:PULSE.007", "case:MOSAIC.006"]},
        "CASCADE.013": {"all": ["case:CASCADE.012", "case:LENS.002"]},
        "MERGE.009": {"all": ["case:MERGE.008", "case:CASCADE.013"]},
        "MOSAIC.007": {"all": ["case:MOSAIC.006", "case:MERGE.009"]},
        "PULSE.008": {"all": ["case:PULSE.007", "case:CASCADE.013", "case:MOSAIC.007"]},
        "COVENANT.001": {"all": ["case:CASCADE.013", "case:MERGE.009", "case:PULSE.008"]},
        "CASCADE.014": {"all": ["case:CASCADE.013", "case:COVENANT.001"]},
        "MERGE.010": {"all": ["case:MERGE.009", "case:CASCADE.014"]},
        "MOSAIC.008": {"all": ["case:MOSAIC.007", "case:MERGE.010"]},
        "PULSE.009": {"all": ["case:PULSE.008", "case:MERGE.010", "case:MOSAIC.008"]},
        "LENS.003": {"all": ["case:LENS.002", "case:CASCADE.014", "case:MERGE.010", "case:MOSAIC.008", "case:PULSE.009"]},
        "CASCADE.015": {"all": ["case:CASCADE.014", "case:LENS.003"]},
        "MERGE.011": {"all": ["case:MERGE.010", "case:CASCADE.015"]},
        "MOSAIC.009": {"all": ["case:MOSAIC.008", "case:MERGE.011"]},
        "PULSE.010": {"all": ["case:PULSE.009", "case:MERGE.011", "case:MOSAIC.009"]},
        "CASCADE.016": {"all": ["case:CASCADE.015", "case:MERGE.011", "case:MOSAIC.009", "case:PULSE.010"]},
        "MERGE.012": {"all": ["case:MERGE.011", "case:CASCADE.017"]},
        "MOSAIC.010": {"all": ["case:MOSAIC.009", "case:CASCADE.017"]},
        "PULSE.011": {"all": ["case:PULSE.010", "case:CASCADE.017"]},
        "COVENANT.002": {"all": ["case:COVENANT.001", "case:CASCADE.017", "case:MERGE.012", "case:MOSAIC.010", "case:PULSE.011"]},
        "MERGE.013": {"all": ["case:MERGE.012", "case:COVENANT.002"]},
        "MOSAIC.011": {"all": ["case:MOSAIC.010", "case:COVENANT.002"]},
        "PULSE.012": {"all": ["case:PULSE.011", "case:COVENANT.002"]},
        "CASCADE.018": {"all": ["case:CASCADE.017", "case:MERGE.013", "case:MOSAIC.011", "case:PULSE.012"]},
        "CASCADE.019": {"all": ["case:CASCADE.018"]},
        "MERGE.014": {"all": ["case:MERGE.013", "case:CASCADE.019"]},
        "PULSE.013": {"all": ["case:PULSE.012", "case:CASCADE.019"]},
        "CASCADE.020": {"all": ["case:CASCADE.019", "case:MERGE.014"]},
        "CASCADE.021": {"all": ["case:CASCADE.020", "case:PULSE.013"]},
        "CASCADE.022": {"all": ["case:CASCADE.021"]},
        "MERGE.015": {"all": ["case:MERGE.014", "case:CASCADE.022"]},
        "PULSE.014": {"all": ["case:PULSE.013", "case:CASCADE.022", "case:MERGE.015"]},
        "CASCADE.023": {"all": ["case:CASCADE.022", "case:MERGE.015", "case:PULSE.014", "case:COVENANT.002"]},
        "CASCADE.024": {"all": ["case:CASCADE.023"]},
        "PARADOX.001": {"all": ["case:COVENANT.002", "case:LENS.003", "case:CASCADE.024"]},
    }
    if case_id in special:
        return special[case_id]
    if number > 1:
        return {"all": [f"case:{family}.{number - 1:03d}"]}
    raise ValueError(f"no production prerequisite for {case_id}")


def build() -> dict[str, Any]:
    slice_manifest = json.loads((ROOT / "manifests" / "vertical_slice.json").read_text(encoding="utf-8"))
    cases = []
    ordinals: dict[str, int] = {}
    for item in slice_manifest["cases"]:
        family = item["family"]
        ordinals[family] = max(ordinals.get(family, 0), int(item["id"].split(".")[1]))
        cases.append({
            "id": item["id"], "family": family, "title": item["title"],
            "points": item["points"], "act": "vertical slice", "band": item["difficulty"],
            "mechanic": item["aha"], "requires": item["requires"],
            "wave": "slice", "status": "golden",
        })
    for family, planned in PLANNED.items():
        for title, points, act, mechanic in planned:
            ordinals[family] = ordinals.get(family, 0) + 1
            case_id = f"{family}.{ordinals[family]:03d}"
            cases.append({
                "id": case_id, "family": family, "title": title, "points": points,
                "act": act, "band": min(5, 1 + (ordinals[family] - 1) // 4),
                "mechanic": mechanic, "requires": requirement(case_id),
                "wave": (
                    "production-01" if case_id == "PULSE.002"
                    else "production-02" if case_id == "MERGE.002"
                    else "production-03" if case_id == "PULSE.003"
                    else "production-04" if case_id == "MOSAIC.002"
                    else "production-05" if case_id == "CASCADE.004"
                    else "production-06" if case_id in {"CASCADE.005", "CASCADE.006"}
                    else "production-07" if case_id in {"CASCADE.007", "MERGE.003", "MOSAIC.003"}
                    else "production-08" if case_id == "PULSE.004"
                    else "production-09" if case_id in {"CASCADE.008", "MERGE.004"}
                    else "production-10" if case_id in {"CASCADE.009", "MERGE.005", "PULSE.005"}
                    else "production-11" if case_id in {"CASCADE.010", "MERGE.006", "MOSAIC.004"}
                    else "production-12" if case_id in {"CASCADE.011", "MERGE.007", "MOSAIC.005", "PULSE.006"}
                    else "production-13" if case_id in {"CASCADE.012", "MERGE.008", "MOSAIC.006", "PULSE.007", "LENS.002"}
                    else "production-14" if case_id in {"CASCADE.013", "MERGE.009", "MOSAIC.007", "PULSE.008", "COVENANT.001"}
                    else "production-15" if case_id in {"CASCADE.014", "MERGE.010", "MOSAIC.008", "PULSE.009", "LENS.003"}
                    else "production-16" if case_id in {"CASCADE.015", "MERGE.011", "MOSAIC.009", "PULSE.010", "CASCADE.016"}
                    else "production-17" if case_id in {"CASCADE.017", "MERGE.012", "MOSAIC.010", "PULSE.011", "COVENANT.002"}
                    else "production-18" if case_id in {"MERGE.013", "MOSAIC.011", "PULSE.012", "CASCADE.018", "CASCADE.019"}
                    else "production-19" if case_id in {"MERGE.014", "PULSE.013", "CASCADE.020", "CASCADE.021", "CASCADE.022"}
                    else "production-20" if case_id in {"MERGE.015", "PULSE.014", "CASCADE.023", "CASCADE.024", "PARADOX.001"}
                    else "backlog"
                ),
                "status": "authored" if case_id in {"PULSE.002", "MERGE.002", "PULSE.003", "MOSAIC.002", "MOSAIC.003", "MOSAIC.004", "MOSAIC.005", "MOSAIC.006", "MOSAIC.007", "MOSAIC.008", "MOSAIC.009", "MOSAIC.010", "MOSAIC.011", "CASCADE.004", "CASCADE.005", "CASCADE.006", "CASCADE.007", "CASCADE.008", "CASCADE.009", "CASCADE.010", "CASCADE.011", "CASCADE.012", "CASCADE.013", "CASCADE.014", "CASCADE.015", "CASCADE.016", "CASCADE.017", "CASCADE.018", "CASCADE.019", "CASCADE.020", "CASCADE.021", "CASCADE.022", "CASCADE.023", "CASCADE.024", "MERGE.003", "MERGE.004", "MERGE.005", "MERGE.006", "MERGE.007", "MERGE.008", "MERGE.009", "MERGE.010", "MERGE.011", "MERGE.012", "MERGE.013", "MERGE.014", "MERGE.015", "PULSE.004", "PULSE.005", "PULSE.006", "PULSE.007", "PULSE.008", "PULSE.009", "PULSE.010", "PULSE.011", "PULSE.012", "PULSE.013", "PULSE.014", "LENS.002", "LENS.003", "COVENANT.001", "COVENANT.002", "PARADOX.001"} else "planned",
            })
    return {
        "format": "afterimage-production-catalog/0.1",
        "title": "Afterimage complete production catalog",
        "total_cases": 75,
        "total_nominal_points": 10000,
        "cases": sorted(cases, key=lambda item: (item["family"], item["id"])),
    }


def render(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render(build())
    if args.check:
        actual = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if actual != expected:
            print(f"production catalog is stale: run {Path(__file__).name}")
            return 1
        print("production catalog: current")
        return 0
    OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT} ({len(build()['cases'])} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
