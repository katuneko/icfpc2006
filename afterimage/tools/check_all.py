#!/usr/bin/env python3
"""Run every dependency-free Afterimage design and substrate gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    ("catalog", [sys.executable, str(ROOT / "tools" / "build_catalog.py"), "--check"]),
    ("design", [sys.executable, str(ROOT / "tools" / "check_design.py")]),
    ("cre-conformance", [sys.executable, str(ROOT / "tools" / "run_conformance.py")]),
    (
        "public-conformance-oracle",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_conformance_public.py"],
    ),
    (
        "bundle-security",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_afterimage_kit.py"],
    ),
    ("locales", [sys.executable, str(ROOT / "tools" / "build_locales.py"), "--check"]),
    (
        "localization-invariance",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_localization.py"],
    ),
    (
        "witness-verifier",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_verify_witness.py"],
    ),
    (
        "pulse-runtime",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_pulse.py"],
    ),
    (
        "mosaic-verifier",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_mosaic.py"],
    ),
    (
        "lens-laws",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_lens.py"],
    ),
    (
        "covenant-model-checker",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_covenant.py"],
    ),
    (
        "paradox-certificate",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_paradox.py"],
    ),
    (
        "playtest-decision",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_playtest_analysis.py"],
    ),
    (
        "ai-proxy-decision",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_ai_proxy.py"],
    ),
    (
        "player-workspace",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_player.py"],
    ),
    (
        "public-release",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_public_release.py"],
    ),
    (
        "slice-content",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_slice_content.py"],
    ),
    (
        "production-content",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_production_content.py"],
    ),
    (
        "production-waves",
        [sys.executable, "-m", "unittest", "-v", "afterimage/tests/test_production_wave13.py", "afterimage/tests/test_production_wave14.py", "afterimage/tests/test_production_wave15.py", "afterimage/tests/test_production_wave16.py", "afterimage/tests/test_production_wave17.py", "afterimage/tests/test_production_wave18.py", "afterimage/tests/test_production_wave19.py", "afterimage/tests/test_production_wave20.py"],
    ),
]


def main() -> int:
    for name, command in COMMANDS:
        print(f"== {name} ==", flush=True)
        completed = subprocess.run(command, cwd=ROOT.parent, check=False)
        if completed.returncode != 0:
            print(f"gate failed: {name} (exit {completed.returncode})", file=sys.stderr)
            return completed.returncode or 1
    print("all Afterimage gates: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
