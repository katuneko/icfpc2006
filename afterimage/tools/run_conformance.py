#!/usr/bin/env python3
"""Cross-check both CRE 0.1 references and the public localized oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "python"))

import cre  # noqa: E402

SUITE = ROOT / "tests" / "conformance" / "suite.json"
FULL_SUITE = ROOT / "tests" / "conformance" / "full-suite.json"
GOLDEN = ROOT / "tests" / "conformance" / "golden.json"
EXPECTED = ROOT / "tests" / "conformance" / "expected.sha256"
ENGINES = {
    "python": [sys.executable, str(ROOT / "reference" / "python" / "cre.py")],
    "javascript": ["node", str(ROOT / "reference" / "javascript" / "cre.mjs")],
}


def load_release_suite() -> dict[str, object]:
    base = json.loads(SUITE.read_text(encoding="utf-8"))
    matrix_path = ROOT / "tests" / "conformance" / "matrix.py"
    spec = importlib.util.spec_from_file_location("afterimage_conformance_matrix", matrix_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {matrix_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extend_suite(base)


def run_engine(name: str, command: list[str], suite_path: Path) -> bytes:
    completed = subprocess.run(
        [*command, "suite", str(suite_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{name} exited {completed.returncode}: {detail}")
    return completed.stdout


def first_difference(left: bytes, right: bytes) -> str:
    limit = min(len(left), len(right))
    offset = next((index for index in range(limit) if left[index] != right[index]), limit)
    return f"first difference at byte {offset}; sizes {len(left)} and {len(right)}"


def public_oracle(result: dict[str, object]) -> dict[str, object]:
    """Remove informative prose while retaining every normative case value."""
    if result.get("format") != "afterimage-conformance-result/0.1":
        raise RuntimeError("reference emitted an unsupported conformance result")
    cases = []
    for entry in result["cases"]:
        if "ok" in entry:
            cases.append({"ok": entry["ok"]})
            continue
        error = entry.get("error")
        if not isinstance(error, dict) or set(error) != {"code", "message", "context"}:
            raise RuntimeError("reference error has an unsupported shape")
        cases.append(
            {
                "name": entry["name"],
                "error": {"code": error["code"], "context": error["context"]},
            }
        )
    return {
        "format": "afterimage-conformance-oracle/0.1",
        "canonical_vectors": result["canonical_vectors"],
        "cases": cases,
    }


def check_exit_codes(suite: dict[str, object]) -> None:
    cases = {case["name"]: case for case in suite["cases"]}
    with tempfile.TemporaryDirectory(prefix="afterimage-conformance-") as temporary:
        directory = Path(temporary)
        inputs = {
            "invalid": (directory / "invalid.json", 2),
            "semantic": (directory / "semantic.json", 3),
            "resource": (directory / "resource.json", 4),
        }
        inputs["invalid"][0].write_text("{", encoding="utf-8")
        inputs["semantic"][0].write_text(
            json.dumps(cases["arithmetic-overflow"], ensure_ascii=False),
            encoding="utf-8",
        )
        inputs["resource"][0].write_text(
            json.dumps(cases["derived-resource-limit"], ensure_ascii=False),
            encoding="utf-8",
        )
        for engine, command in ENGINES.items():
            for label, (path, expected) in inputs.items():
                completed = subprocess.run(
                    [*command, "case", str(path)],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                )
                if completed.returncode != expected:
                    raise RuntimeError(
                        f"{engine} {label} exit code {completed.returncode}, expected {expected}"
                    )


def check_hostile_inputs() -> int:
    hostile = {
        "duplicate-key": (b'{"x":1,"x":2}', "duplicate_key"),
        "nfc-key-collision": ('{"é":1,"e\\u0301":2}'.encode("utf-8"), "duplicate_key"),
        "floating-point": (b'1.5', "type_error"),
        "invalid-scalar": (b'"\\ud800"', "invalid_text"),
        "bytes-padding": (b'{"$bytes":"AQ=="}', "invalid_bytes"),
        "bytes-unused-bits": (b'{"$bytes":"AB"}', "invalid_bytes"),
        "bytes-reserved-map": (b'{"$bytes":"AQ","extra":true}', "invalid_bytes"),
    }
    with tempfile.TemporaryDirectory(prefix="afterimage-hostile-") as temporary:
        directory = Path(temporary)
        for label, (payload, expected_code) in hostile.items():
            path = directory / f"{label}.json"
            path.write_bytes(payload)
            records = {}
            for engine, command in ENGINES.items():
                completed = subprocess.run(
                    [*command, "suite", str(path)],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                )
                if completed.returncode != 2:
                    raise RuntimeError(
                        f"{engine} hostile {label} exit code {completed.returncode}, expected 2"
                    )
                output = completed.stdout.strip() or completed.stderr.strip()
                try:
                    record = json.loads(output)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{engine} hostile {label} emitted invalid JSON") from exc
                actual_code = record.get("error", {}).get("code")
                if actual_code != expected_code:
                    raise RuntimeError(
                        f"{engine} hostile {label} code {actual_code!r}, expected {expected_code!r}"
                    )
                records[engine] = record
            if records["python"] != records["javascript"]:
                raise RuntimeError(f"hostile {label} error records disagree")
    return len(hostile)


def check_protocol(suite: dict[str, object]) -> tuple[bytes, int]:
    cases = {case["name"]: case for case in suite["cases"]}
    with tempfile.TemporaryDirectory(prefix="afterimage-protocol-") as temporary:
        directory = Path(temporary)
        success_path = directory / "success.json"
        success_path.write_text(
            json.dumps(cases["positive-recursion"], ensure_ascii=False),
            encoding="utf-8",
        )
        outputs = {}
        for engine, command in ENGINES.items():
            completed = subprocess.run(
                [*command, "protocol", str(success_path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"{engine} protocol success exited {completed.returncode}")
            if completed.stderr:
                raise RuntimeError(f"{engine} protocol success wrote stderr")
            outputs[engine] = completed.stdout
        if outputs["python"] != outputs["javascript"]:
            raise RuntimeError("successful protocol streams disagree")
        records = [json.loads(line) for line in outputs["python"].splitlines()]
        types = [record.get("type") for record in records]
        if types != ["ready", "projection", "projection", "projection", "done"]:
            raise RuntimeError(f"unexpected successful protocol sequence: {types}")
        if [record["index"] for record in records[1:4]] != [0, 1, 2]:
            raise RuntimeError("protocol projection indices are not contiguous")

        failures = {
            "semantic": (cases["arithmetic-overflow"], 3),
            "resource": (cases["derived-resource-limit"], 4),
        }
        for label, (case_value, expected_exit) in failures.items():
            path = directory / f"{label}.json"
            path.write_text(json.dumps(case_value, ensure_ascii=False), encoding="utf-8")
            failure_records = {}
            for engine, command in ENGINES.items():
                completed = subprocess.run(
                    [*command, "protocol", str(path)],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                )
                if completed.returncode != expected_exit:
                    raise RuntimeError(
                        f"{engine} protocol {label} exited {completed.returncode}, expected {expected_exit}"
                    )
                lines = completed.stdout.splitlines()
                if len(lines) != 1:
                    raise RuntimeError(f"{engine} protocol {label} did not emit one terminal line")
                failure_records[engine] = json.loads(lines[0])
                if failure_records[engine].get("type") != "error":
                    raise RuntimeError(f"{engine} protocol {label} terminal record is not error")
            if failure_records["python"] != failure_records["javascript"]:
                raise RuntimeError(f"protocol {label} error records disagree")

        invalid_path = directory / "invalid.json"
        invalid_path.write_bytes(b"{")
        for engine, command in ENGINES.items():
            completed = subprocess.run(
                [*command, "protocol", str(invalid_path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            if completed.returncode != 2 or len(completed.stdout.splitlines()) != 1:
                raise RuntimeError(f"{engine} invalid protocol input contract failed")
            if json.loads(completed.stdout).get("type") != "error":
                raise RuntimeError(f"{engine} invalid protocol record is not error")
    return outputs["python"], len(records)


def check_semantic_expectations(result: dict[str, object]) -> int:
    cases = {
        (entry["ok"]["name"] if "ok" in entry else entry["name"]): entry
        for entry in result["cases"]
    }
    assertions = 0

    def require(condition: bool, message: str) -> None:
        nonlocal assertions
        assertions += 1
        if not condition:
            raise RuntimeError(message)

    require(cases["positive-recursion"]["ok"]["projection"] == ["A", "B", "C"], "recursion projection drifted")
    require(cases["sealed-lower-stratum-negation"]["ok"]["projection"] == ["beta"], "sealed negation admitted the wrong requests")
    require(cases["dag-same-time-parent"]["ok"]["projection"] == [{"ok": True}], "same-time parent case drifted")
    require(
        cases["branch-non-root-noop"]["ok"]["branch"] == "sha256:" + "1" * 64,
        "empty non-root branch did not preserve its parent identity",
    )

    expression = cases["expression-opcode-matrix"]["ok"]["projection"][0]
    require(expression["pointer_escape"] == 7 and expression["list_index"] == 20, "JSON Pointer matrix drifted")
    require(expression["arithmetic"] == [12, 2, 35, -2, -1], "arithmetic opcode matrix drifted")
    require(expression["extrema"] == [2, "z"], "min/max opcode matrix drifted")
    require(expression["concat_text"] == "café" and expression["concat_bytes"] == {"$bytes": "AQL_"}, "concat opcode matrix drifted")
    require(expression["lengths"] == [1, 3, 2, 1], "length opcode matrix drifted")
    require(expression["contains"] == [True, True, True, True], "contains opcode matrix drifted")
    require(expression["conditional"] == "yes", "conditional opcode matrix drifted")

    error_codes = {
        "dag-claimed-id-mismatch": "event_id_mismatch",
        "dag-missing-parent": "missing_parent",
        "dag-time-reversal": "invalid_time",
        "pointer-missing-path": "missing_path",
        "pointer-invalid-escape": "invalid_pointer",
        "distinct-duplicate-alias": "invalid_rule",
        "branch-invalid-replace-pointer": "invalid_operation",
        "branch-operation-conflict": "operation_conflict",
        "branch-inject-missing-parent": "missing_parent",
    }
    for name, code in error_codes.items():
        require(cases[name]["error"]["code"] == code, f"{name} error code drifted")

    counters = {
        "base": "base_events",
        "derived": "derived_events",
        "bindings": "bindings_tested",
        "projection": "projection_records",
    }
    for family, counter in counters.items():
        require(cases[f"resource-{family}-below"]["ok"]["counters"][counter] == 1, f"resource {family} below boundary drifted")
        require(cases[f"resource-{family}-exact"]["ok"]["counters"][counter] == 2, f"resource {family} exact boundary drifted")
        require(cases[f"resource-{family}-over"]["error"]["code"] == "resource_exhausted", f"resource {family} over boundary did not fail")
    require("ok" in cases["resource-value-bytes-below"], "value byte below boundary failed")
    require("ok" in cases["resource-value-bytes-exact"], "value byte exact boundary failed")
    require(cases["resource-value-bytes-over"]["error"]["code"] == "resource_exhausted", "value byte over boundary did not fail")
    return assertions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="replace public full-suite/oracle/digest only after both references agree",
    )
    args = parser.parse_args(argv)

    try:
        suite = load_release_suite()
        with tempfile.TemporaryDirectory(prefix="afterimage-release-suite-") as temporary:
            suite_path = Path(temporary) / "suite.json"
            suite_path.write_text(
                json.dumps(suite, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            outputs = {
                name: run_engine(name, command, suite_path)
                for name, command in ENGINES.items()
            }
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"conformance: FAIL: {exc}", file=sys.stderr)
        return 1

    python_output = outputs["python"]
    javascript_output = outputs["javascript"]
    if python_output != javascript_output:
        print(f"conformance: FAIL: references disagree; {first_difference(python_output, javascript_output)}", file=sys.stderr)
        return 1

    try:
        protocol_output, protocol_records = check_protocol(suite)
    except (OSError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"conformance: FAIL: protocol stream check: {exc}", file=sys.stderr)
        return 1
    result = json.loads(python_output)
    try:
        semantic_assertions = check_semantic_expectations(result)
        check_exit_codes(suite)
        hostile_count = check_hostile_inputs()
    except (OSError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"conformance: FAIL: protocol check: {exc}", file=sys.stderr)
        return 1
    oracle = public_oracle(result)
    suite_bytes = cre.canonical_bytes(suite)
    oracle_bytes = cre.canonical_bytes(oracle)
    actual = hashlib.sha256(oracle_bytes).hexdigest()
    if args.update:
        FULL_SUITE.write_bytes(suite_bytes)
        GOLDEN.write_bytes(oracle_bytes)
        EXPECTED.write_text(actual + "\n", encoding="ascii")
    try:
        if FULL_SUITE.read_bytes() != suite_bytes:
            raise RuntimeError("public full-suite.json drifted from suite.json + matrix.py")
        if GOLDEN.read_bytes() != oracle_bytes:
            raise RuntimeError("public golden.json drifted from the agreed normative oracle")
    except OSError as exc:
        print(f"conformance: FAIL: cannot read public oracle artifacts: {exc}", file=sys.stderr)
        return 1
    try:
        expected = EXPECTED.read_text(encoding="ascii").strip()
    except OSError as exc:
        print(f"conformance: FAIL: cannot read golden digest: {exc}", file=sys.stderr)
        return 1
    if actual != expected:
        print(
            f"conformance: FAIL: agreed output drifted\nexpected {expected}\nactual   {actual}",
            file=sys.stderr,
        )
        return 1
    successes = sum("ok" in case for case in result["cases"])
    errors = len(result["cases"]) - successes
    print(f"conformance: PASS: Python == JavaScript == public oracle {actual}")
    print(
        f"vectors={len(result['canonical_vectors'])} cases={len(result['cases'])} "
        f"successes={successes} expected-errors={errors}"
    )
    print("protocol exits: invalid=2 semantic=3 resource=4 (both references)")
    print(f"NDJSON protocol: {protocol_records} success records plus single-line terminal errors")
    print(f"hostile canonical inputs: {hostile_count} rejected identically by both references")
    print(f"semantic intent assertions: {semantic_assertions}")
    print("localized oracle: full-suite.json + golden.json (informative error prose excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
