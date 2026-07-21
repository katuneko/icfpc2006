from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECK_PATH = ROOT / "tests" / "conformance" / "check.py"
GOLDEN_PATH = ROOT / "tests" / "conformance" / "golden.json"

SPEC = importlib.util.spec_from_file_location("afterimage_public_conformance_check", CHECK_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load public conformance checker")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class PublicConformanceTests(unittest.TestCase):
    def test_both_reference_commands_pass_shipped_public_checker(self) -> None:
        commands = (
            [sys.executable, str(ROOT / "reference" / "python" / "cre.py")],
            ["node", str(ROOT / "reference" / "javascript" / "cre.mjs")],
        )
        for command in commands:
            with self.subTest(engine=command[0]):
                completed = subprocess.run(
                    [sys.executable, str(CHECK_PATH), "--", *command],
                    cwd=ROOT.parent,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("vectors=9 cases=37", completed.stdout)

    def test_error_prose_is_ignored_but_normative_values_are_exact(self) -> None:
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        result = {
            "format": "afterimage-conformance-result/0.1",
            "canonical_vectors": copy.deepcopy(golden["canonical_vectors"]),
            "cases": [],
        }
        for entry in golden["cases"]:
            if "ok" in entry:
                result["cases"].append(entry)
            else:
                result["cases"].append(
                    {
                        "name": entry["name"],
                        "error": {
                            "code": entry["error"]["code"],
                            "message": "localized human prose may differ",
                            "context": entry["error"]["context"],
                        },
                    }
                )
        self.assertEqual(checker.public_oracle(result), golden)

        result["canonical_vectors"][0]["digest"] = "sha256:" + "f" * 64
        difference = checker.first_difference(golden, checker.public_oracle(result))
        self.assertIsNotNone(difference)
        self.assertEqual(difference[0], "$/canonical_vectors/0/digest")

    def test_cli_reports_first_json_pointer_instead_of_only_a_digest(self) -> None:
        result = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        result["format"] = "afterimage-conformance-result/0.1"
        result["canonical_vectors"][0]["canonical"] = "false"
        with tempfile.TemporaryDirectory(prefix="afterimage-bad-public-engine-") as temporary:
            script = Path(temporary) / "engine.py"
            script.write_text(
                "import json\n"
                f"value = {result!r}\n"
                "print(json.dumps(value, ensure_ascii=False, separators=(',', ':')))\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, str(CHECK_PATH), "--", sys.executable, str(script)],
                cwd=ROOT.parent,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("first mismatch at $/canonical_vectors/0/canonical", completed.stderr)

    def test_private_generalization_rejects_a_frozen_oracle_adapter(self) -> None:
        checker_path = ROOT / "tools" / "check_engine_generalization.py"
        reference = subprocess.run(
            [sys.executable, str(checker_path), "--", sys.executable, str(ROOT / "reference" / "python" / "cre.py")],
            cwd=ROOT.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(reference.returncode, 0, reference.stderr)
        self.assertIn("generalization: PASS", reference.stdout)

        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        frozen = {
            "format": "afterimage-conformance-result/0.1",
            "canonical_vectors": golden["canonical_vectors"],
            "cases": golden["cases"],
        }
        with tempfile.TemporaryDirectory(prefix="afterimage-frozen-adapter-") as temporary:
            script = Path(temporary) / "adapter.py"
            script.write_text(
                "import json\n"
                f"print(json.dumps({frozen!r}, ensure_ascii=False, separators=(',', ':')))\n",
                encoding="utf-8",
            )
            rejected = subprocess.run(
                [sys.executable, str(checker_path), "--", sys.executable, str(script)],
                cwd=ROOT.parent,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(rejected.returncode, 1)
        self.assertIn("generalization: FAIL", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
