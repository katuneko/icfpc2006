#!/usr/bin/env python3
"""Show a normative CRE trace for a small case, capped at 128 derivations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "python"))

import cre  # noqa: E402


MAX_DERIVED = 128


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one CRE case or a conformance suite")
    parser.add_argument("--name", help="case name when input is a suite; defaults to the first case")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        loaded = cre.load_json(args.input)
        if isinstance(loaded, dict) and "cases" in loaded:
            candidates = loaded["cases"]
            if args.name is None:
                case = candidates[0]
            else:
                case = next((item for item in candidates if item.get("name") == args.name), None)
                if case is None:
                    cre.fail("invalid_fixture", "named case is absent from suite", name=args.name)
        else:
            case = loaded
        base, operations = cre.resolve_fixture(case)
        branched, branch, _ = cre.apply_branch(
            base,
            case["bundle_digest"],
            operations,
            case.get("parent_branch"),
        )
        limits = dict(case.get("limits", {}))
        limits["max_derived_events"] = min(
            limits.get("max_derived_events", MAX_DERIVED),
            MAX_DERIVED,
        )
        events, trace, counters = cre.evaluate_world(case["program"], branched, limits)
        firings = [item for item in trace if item["stratum"] >= 0]
        projection = None
        projection_digest = None
        if "projection" in case:
            projection, projection_digest = cre.evaluate_projection(case["projection"], events, counters)
        enriched = []
        for item in trace:
            enriched.append({**item, "body": cre.event_view(events[item["event"]])})
        result = {
            "format": "afterimage-trace-oracle/0.1",
            "case": case["name"],
            "branch": branch,
            "derived_firings": len(firings),
            "maximum_derived_firings": MAX_DERIVED,
            "trace": enriched,
            "trace_digest": cre.trace_digest(trace),
            "projection": projection,
            "projection_digest": projection_digest,
            "counters": counters.as_value(),
        }
        print(cre.output_json(result, args.pretty))
        return 0
    except cre.CREError as exc:
        print(cre.output_json({"error": exc.as_value()}), file=sys.stderr)
        return 4 if exc.code == "resource_exhausted" else 3


if __name__ == "__main__":
    raise SystemExit(main())
