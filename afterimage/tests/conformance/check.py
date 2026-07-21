#!/usr/bin/env python3
"""Run a participant CRE suite command and localize its first oracle mismatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FULL_SUITE = HERE / "full-suite.json"
GOLDEN = HERE / "golden.json"
EXPECTED = HERE / "expected.sha256"
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


class CheckError(Exception):
    pass


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CheckError(f"duplicate output key: {key}")
        value[key] = item
    return value


def normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not I64_MIN <= value <= I64_MAX:
            raise CheckError("output integer exceeds signed 64-bit range")
        return value
    if isinstance(value, float):
        raise CheckError("floating-point output is forbidden")
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise CheckError("output keys collide after NFC normalization")
            normalized[normalized_key] = normalize(item)
        return normalized
    raise CheckError(f"unsupported output type: {type(value).__name__}")


def load_value(data: bytes, location: str) -> Any:
    if data.startswith(b"\xef\xbb\xbf"):
        raise CheckError(f"{location}: UTF-8 BOM is forbidden")
    try:
        parsed = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=no_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckError(f"{location}: invalid JSON: {exc}") from exc
    return normalize(parsed)


def public_oracle(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict) or result.get("format") != "afterimage-conformance-result/0.1":
        raise CheckError("engine output must be afterimage-conformance-result/0.1")
    if set(result) != {"format", "canonical_vectors", "cases"} or not isinstance(result["cases"], list):
        raise CheckError("engine conformance result has wrong fields")
    cases = []
    for index, entry in enumerate(result["cases"]):
        if not isinstance(entry, dict):
            raise CheckError(f"cases[{index}] is not an object")
        if set(entry) == {"ok"}:
            cases.append(entry)
        elif set(entry) == {"name", "error"} and isinstance(entry["error"], dict):
            error = entry["error"]
            if set(error) not in ({"code", "context"}, {"code", "message", "context"}):
                raise CheckError(f"cases[{index}].error has wrong fields")
            cases.append(
                {
                    "name": entry["name"],
                    "error": {"code": error["code"], "context": error["context"]},
                }
            )
        else:
            raise CheckError(f"cases[{index}] must contain exactly ok, or name+error")
    return {
        "format": "afterimage-conformance-oracle/0.1",
        "canonical_vectors": result["canonical_vectors"],
        "cases": cases,
    }


def first_difference(expected: Any, actual: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    if type(expected) is not type(actual):
        return path, expected, actual
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return f"{path} keys", sorted(expected_keys), sorted(actual_keys)
        for key in sorted(expected, key=lambda item: item.encode("utf-8")):
            difference = first_difference(expected[key], actual[key], f"{path}/{key.replace('~', '~0').replace('/', '~1')}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path} length", len(expected), len(actual)
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            difference = first_difference(left, right, f"{path}/{index}")
            if difference is not None:
                return difference
        return None
    return None if expected == actual else (path, expected, actual)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", nargs=argparse.REMAINDER, help="engine command; checker appends: suite FULL_SUITE")
    args = parser.parse_args(argv)
    command = args.engine[1:] if args.engine[:1] == ["--"] else args.engine
    if not command:
        print("conformance check failed: pass an engine command", file=sys.stderr)
        return 2
    try:
        golden_bytes = GOLDEN.read_bytes()
        expected_digest = EXPECTED.read_text(encoding="ascii").strip()
        if hashlib.sha256(golden_bytes).hexdigest() != expected_digest:
            raise CheckError("shipped golden.json digest is inconsistent")
        expected = load_value(golden_bytes, "golden.json")
        completed = subprocess.run(
            [*command, "suite", str(FULL_SUITE)],
            check=False,
            capture_output=True,
            cwd=Path.cwd(),
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[:2000]
            raise CheckError(f"engine exited {completed.returncode}: {detail}")
        if completed.stderr:
            raise CheckError("engine wrote to stderr on a successful suite run")
        if len(completed.stdout) > MAX_OUTPUT_BYTES:
            raise CheckError("engine output exceeds 64 MiB")
        actual = public_oracle(load_value(completed.stdout, "engine stdout"))
        difference = first_difference(expected, actual)
        if difference is not None:
            path, wanted, observed = difference
            raise CheckError(
                f"first mismatch at {path}\n"
                f"expected {json.dumps(wanted, ensure_ascii=False, sort_keys=True)}\n"
                f"actual   {json.dumps(observed, ensure_ascii=False, sort_keys=True)}"
            )
    except (OSError, CheckError) as exc:
        print(f"conformance check failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"conformance check: PASS: {expected_digest} "
        f"vectors={len(expected['canonical_vectors'])} cases={len(expected['cases'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
