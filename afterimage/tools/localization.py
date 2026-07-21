#!/usr/bin/env python3
"""Strict external locale packs for Afterimage player presentation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCALES_ROOT = ROOT / "locales"
FORMAT = "afterimage-locale-pack/0.1"
SUPPORTED = ("en", "zh-Hans", "ja", "de")
CASE_FIELDS = {"title", "premise", "submission", "diagnostics", "hints", "protected"}
UI_FIELDS = {
    "answer_schema", "points", "solved", "visible", "none", "hint", "score",
    "case_score", "canonicalized", "valid", "branch", "invalid",
    "submission_heading", "diagnostics_heading", "hints_heading",
}


class LocalizationError(Exception):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalizationError(message)


def _load_json(path: Path) -> Any:
    _require(path.is_file() and not path.is_symlink(), f"locale pack is missing or unsafe: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalizationError(f"cannot read locale pack {path.name}: {exc}") from exc


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LocalizationError(f"duplicate locale key: {key}")
        result[key] = value
    return result


def _validate_template(name: str, value: str, fields: set[str]) -> None:
    found = set(re.findall(r"\{([a-z0-9_]+)\}", value))
    _require(found == fields, f"UI template {name} placeholders differ: {sorted(found)}")


@dataclass(frozen=True)
class LocalePack:
    locale: str
    bundles: tuple[str, ...]
    ui: dict[str, str]
    cases: dict[str, dict[str, Any]]

    def case(self, case_id: str) -> dict[str, Any]:
        try:
            return self.cases[case_id]
        except KeyError as exc:
            raise LocalizationError(f"locale {self.locale} lacks case {case_id}") from exc

    def story(self, case_id: str) -> str:
        case = self.case(case_id)
        hints = "\n".join(f"{index}. {hint}" for index, hint in enumerate(case["hints"], 1))
        return (
            f"# {case_id} — {case['title']}\n\n"
            f"{case['premise']}\n\n"
            f"## {self.ui['submission_heading']}\n\n{case['submission']}\n\n"
            f"## {self.ui['diagnostics_heading']}\n\n{case['diagnostics']}\n\n"
            f"## {self.ui['hints_heading']}\n\n{hints}\n"
        )


TEMPLATE_FIELDS = {
    "answer_schema": set(),
    "points": {"points"},
    "solved": {"solved", "total", "cases"},
    "visible": {"cases"},
    "none": set(),
    "hint": {"case", "level", "hint"},
    "score": {"total", "nominal_solved", "nominal_total"},
    "case_score": {"case", "score", "nominal", "metrics"},
    "canonicalized": {"bytes", "sha256"},
    "valid": {"case", "score", "nominal"},
    "branch": {"branch", "projection", "trace"},
    "invalid": {"case", "code", "message"},
    "submission_heading": set(),
    "diagnostics_heading": set(),
    "hints_heading": set(),
}


def load_pack(
    locale: str,
    bundle: str | None = None,
    *,
    expected_cases: set[str] | None = None,
    exact_cases: bool = False,
) -> LocalePack:
    _require(locale in SUPPORTED, f"unsupported locale: {locale}")
    value = _load_json(LOCALES_ROOT / f"{locale}.json")
    _require(isinstance(value, dict) and set(value) == {"format", "locale", "bundles", "ui", "cases"}, f"{locale}: pack fields are invalid")
    _require(value["format"] == FORMAT and value["locale"] == locale, f"{locale}: identity is invalid")
    bundles = value["bundles"]
    _require(isinstance(bundles, list) and bundles and len(set(bundles)) == len(bundles), f"{locale}: bundles are invalid")
    _require(all(isinstance(item, str) and item.startswith("sha256:") for item in bundles), f"{locale}: bundle ID is invalid")
    if bundle is not None:
        _require(bundle in bundles, f"locale {locale} does not support bundle {bundle}")
    ui = value["ui"]
    _require(isinstance(ui, dict) and set(ui) == UI_FIELDS, f"{locale}: UI fields are invalid")
    for name, fields in TEMPLATE_FIELDS.items():
        _require(isinstance(ui[name], str) and ui[name], f"{locale}: UI text {name} is empty")
        _validate_template(name, ui[name], fields)
    cases = value["cases"]
    _require(isinstance(cases, dict) and cases, f"{locale}: cases are invalid")
    if expected_cases is not None:
        actual_cases = set(cases)
        if exact_cases:
            _require(actual_cases == expected_cases, f"{locale}: case coverage differs")
        else:
            _require(expected_cases <= actual_cases, f"{locale}: required case coverage is missing")
    for case_id, case in cases.items():
        _require(isinstance(case_id, str) and isinstance(case, dict) and set(case) == CASE_FIELDS, f"{locale}: invalid case entry {case_id}")
        for field in ("title", "premise", "submission", "diagnostics"):
            _require(isinstance(case[field], str) and case[field].strip(), f"{locale}/{case_id}: {field} is empty")
        _require(isinstance(case["hints"], list) and len(case["hints"]) == 3 and all(isinstance(item, str) and item.strip() for item in case["hints"]), f"{locale}/{case_id}: hints are invalid")
        _require(isinstance(case["protected"], list) and len(set(case["protected"])) == len(case["protected"]) and all(isinstance(item, str) and item for item in case["protected"]), f"{locale}/{case_id}: protected tokens are invalid")
        corpus = "\n".join([case["title"], case["premise"], case["submission"], case["diagnostics"], *case["hints"]])
        for token in case["protected"]:
            _require(token in corpus, f"{locale}/{case_id}: protected token is missing from translated prose: {token}")
    return LocalePack(locale=locale, bundles=tuple(bundles), ui=ui, cases=cases)
