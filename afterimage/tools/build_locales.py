#!/usr/bin/env python3
"""Generate the canonical English locale and validate every translation pack."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import localization  # noqa: E402


BUNDLES = [
    "sha256:517038cdd97cb7d3687f53272e8964a11ffcc1cca82cc69a73668bf56aea0514",
    "sha256:6180c2ae6e0e2e6d024a5713fcf43e7c2f34ef40280dfb56515675cfecc4b11b",
]
EN_UI = {
    "answer_schema": "Answer schema:", "points": "{points} points",
    "solved": "Solved {solved}/{total}: {cases}", "visible": "Visible: {cases}", "none": "none",
    "hint": "{case} hint {level}: {hint}",
    "score": "Score {total} (nominal solved {nominal_solved}/{nominal_total})",
    "case_score": "  {case}: {score}/{nominal}  metrics={metrics}",
    "canonicalized": "Canonical JSON: {bytes} bytes, sha256={sha256}",
    "valid": "VALID {case}  score={score}/{nominal}",
    "branch": "branch={branch} projection={projection} trace={trace}",
    "invalid": "INVALID {case}  {code}: {message}",
    "submission_heading": "Submission", "diagnostics_heading": "Diagnostics", "hints_heading": "Hints",
}


def sources() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for root in (ROOT / "content" / "vertical_slice" / "cases", ROOT / "content" / "production" / "cases"):
        for path in sorted(root.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            result[value["case"]["id"]] = value
    return result


def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in strings(child)]
    return []


def protected_tokens(source: dict[str, Any]) -> list[str]:
    story = source["story"]
    corpus = "\n".join([source["case"]["title"], story["premise"], story["submission"], story["diagnostics"], *story["hints"]])
    semantic_case = {key: value for key, value in source["case"].items() if key != "title"}
    semantic = {item for item in strings({"case": semantic_case, "events": source["events"], "rules": source["rules"], "projection": source["projection"]}) if len(item) >= 2}
    code_like = {
        token for token in semantic
        if token in corpus and (
            re.search(r"[0-9._:/]", token)
            or "-" in token
            or any(char.isupper() for char in token)
        )
    }
    quoted = set(re.findall(r"`([^`\n]+)`", corpus))
    numeric = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", corpus))
    semantic_literals = {item for item in ("surface", "underground", "clear", "hot") if item in semantic and item in corpus}
    return sorted(code_like | quoted | numeric | semantic_literals, key=lambda item: item.encode("utf-8"))


def english_value() -> dict[str, Any]:
    cases = {}
    for case_id, source in sorted(sources().items()):
        story = source["story"]
        cases[case_id] = {
            "title": source["case"]["title"], "premise": story["premise"],
            "submission": story["submission"], "diagnostics": story["diagnostics"],
            "hints": story["hints"], "protected": protected_tokens(source),
        }
    return {"format": localization.FORMAT, "locale": "en", "bundles": BUNDLES, "ui": EN_UI, "cases": cases}


def render(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-en", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    english_path = ROOT / "locales" / "en.json"
    expected = english_value()
    if args.sync_en:
        english_path.parent.mkdir(parents=True, exist_ok=True)
        english_path.write_text(render(expected), encoding="utf-8", newline="\n")
    if not english_path.is_file() or english_path.read_text(encoding="utf-8") != render(expected):
        print("English locale is stale; run build_locales.py --sync-en", file=sys.stderr)
        return 1
    expected_cases = set(expected["cases"])
    try:
        packs = {
            locale: localization.load_pack(locale, expected_cases=expected_cases, exact_cases=True)
            for locale in localization.SUPPORTED
        }
    except localization.LocalizationError as exc:
        print(f"locale error: {exc}", file=sys.stderr)
        return 1
    for locale, pack in packs.items():
        if locale != "en":
            for name in localization.UI_FIELDS:
                if pack.ui[name] == expected["ui"][name]:
                    print(f"locale error: {locale}: UI text {name} is untranslated", file=sys.stderr)
                    return 1
        for case_id in expected_cases:
            if pack.case(case_id)["protected"] != expected["cases"][case_id]["protected"]:
                print(f"locale error: {locale}/{case_id}: protected tokens differ", file=sys.stderr)
                return 1
            if locale != "en":
                translated = pack.case(case_id)
                original = expected["cases"][case_id]
                for field in ("premise", "submission", "diagnostics"):
                    if translated[field] == original[field]:
                        print(f"locale error: {locale}/{case_id}: {field} is untranslated", file=sys.stderr)
                        return 1
                for index, hint in enumerate(translated["hints"]):
                    if hint == original["hints"][index]:
                        print(f"locale error: {locale}/{case_id}: hint {index + 1} is untranslated", file=sys.stderr)
                        return 1
                localized_strings = [translated["title"], translated["premise"], translated["submission"], translated["diagnostics"], *translated["hints"]]
                if any("ZXQTERM" in item or "ZXQGLOSS" in item for item in localized_strings):
                    print(f"locale error: {locale}/{case_id}: translation placeholder leaked", file=sys.stderr)
                    return 1
                if locale in {"ja", "zh-Hans"} and any(not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", item) for item in localized_strings):
                    print(f"locale error: {locale}/{case_id}: translated text lacks CJK script", file=sys.stderr)
                    return 1
    print(f"locales: PASS: {len(packs)} languages, {len(expected_cases)} cases, {sum(len(item['hints']) for item in expected['cases'].values())} hints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
