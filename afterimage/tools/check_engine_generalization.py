#!/usr/bin/env python3
"""Privately distinguish general CRE engines from frozen-oracle adapters."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "python"))

import cre  # noqa: E402


def oracle(value: Any) -> Any:
    if not isinstance(value, dict) or value.get("format") != "afterimage-conformance-result/0.1":
        raise ValueError("engine output is not a conformance result")
    cases = []
    for entry in value.get("cases", []):
        if set(entry) == {"ok"}:
            cases.append(entry)
        elif set(entry) == {"name", "error"} and isinstance(entry["error"], dict):
            error = entry["error"]
            cases.append({"name": entry["name"], "error": {"code": error["code"], "context": error["context"]}})
        else:
            raise ValueError("engine case result has an unsupported shape")
    return {"canonical_vectors": value.get("canonical_vectors"), "cases": cases}


def first_difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        return path
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return path + " keys"
        for key in sorted(expected):
            difference = first_difference(expected[key], actual[key], f"{path}/{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return path + " length"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            difference = first_difference(left, right, f"{path}/{index}")
            if difference is not None:
                return difference
        return None
    return None if expected == actual else path


def run(command: list[str], suite: Path) -> Any:
    completed = subprocess.run([*command, "suite", str(suite)], check=False, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} exited {completed.returncode}")
    temporary = suite.parent / ("output-" + str(abs(hash(tuple(command)))) + ".json")
    temporary.write_bytes(completed.stdout)
    try:
        return cre.load_json(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.engine[1:] if args.engine[:1] == ["--"] else args.engine
    if not command:
        parser.error("provide an engine command after --")
    source = cre.load_json(ROOT / "tests" / "conformance" / "full-suite.json")
    hidden = cre.normalize_value(source)
    for item in [*hidden["canonical_vectors"], *hidden["cases"]]:
        item["name"] = "operator-generalization-" + item["name"]
    with tempfile.TemporaryDirectory(prefix="afterimage-engine-generalization-") as directory:
        suite = Path(directory) / "suite.json"
        suite.write_bytes(cre.canonical_bytes(hidden))
        try:
            expected = oracle(run([sys.executable, str(ROOT / "reference" / "python" / "cre.py")], suite))
            actual = oracle(run(command, suite))
        except (OSError, RuntimeError, ValueError, cre.CREError) as exc:
            print(f"engine generalization: FAIL: {exc}", file=sys.stderr)
            return 1
    difference = first_difference(expected, actual)
    if difference is not None:
        print(f"engine generalization: FAIL: first mismatch at {difference}", file=sys.stderr)
        return 1
    print(f"engine generalization: PASS: vectors={len(expected['canonical_vectors'])} cases={len(expected['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
