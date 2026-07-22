#!/usr/bin/env python3
"""Validate an AI proxy campaign and wrap the ordinary slice decision."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import analyze_playtest as human  # noqa: E402
import cre  # noqa: E402


FORMAT = "afterimage-ai-proxy-campaign/0.1"
DECISION_FORMAT = "afterimage-ai-proxy-decision/0.1"
MATRIX = {
    "PX-1001": ("terra-style-proxy", "low", "runtime-builder", "python"),
    "PX-2001": ("luna-style-proxy", "low", "curious-programmer", "javascript"),
    "PX-3001": ("terra-style-proxy", "medium", "algorithmic-contestant", "cpp"),
    "PX-1002": ("luna-style-proxy", "medium", "runtime-builder", "javascript"),
    "PX-2002": ("terra-style-proxy", "high", "curious-programmer", "python"),
    "PX-3002": ("luna-style-proxy", "high", "algorithmic-contestant", "javascript"),
}
TIMING = {
    "runtime-builder": (4, 10, 25, 30),
    "algorithmic-contestant": (5, 15, 35, 40),
    "curious-programmer": (6, 20, 45, 50),
}
METHOD = {
    "model_selector_available": False,
    "native_effort_available": False,
    "timing_conversion": "cohort-multiplier-v1",
}
TIME_KEYS = {"start", "understood", "first_bounded_run", "session_a_stop", "desk_boot", "first_receipt", "cascade003", "stop"}
ENGINE_KEYS = {"conformance_pass", "vectors_passed", "cases_passed", "protocol_success_exercised", "ambiguities", "unclassified_error"}
OBSERVATION_KEYS = human.TEAM_KEYS - {"id", "cohort", "engine_language", "first_receipt_minutes", "desk_boot_minutes", "cre_minutes"}


class ProxyError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProxyError(message)


def load(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > human.MAX_CAMPAIGN_BYTES:
        raise ProxyError("campaign must be a regular file no larger than 1 MiB")
    try:
        return kit.decode_json(path.read_bytes(), str(path))
    except kit.KitError as exc:
        raise ProxyError(f"invalid canonical campaign: {exc.code}: {exc.message}") from exc


def validate(value: Any) -> list[dict[str, Any]]:
    require(isinstance(value, dict) and set(value) == {"format", "bundle", "method", "system", "conditions"}, "proxy envelope is invalid")
    require(value["format"] == FORMAT, "proxy format is unsupported")
    cre.parse_id(value["bundle"], "proxy bundle")
    require(value["method"] == METHOD, "proxy method does not match the frozen limitations")
    require(isinstance(value["system"], dict) and set(value["system"]) == human.SYSTEM_KEYS, "system evidence fields are invalid")
    conditions = value["conditions"]
    require(isinstance(conditions, list) and len(conditions) == 6, "proxy campaign requires exactly six conditions")
    require({item.get("code") for item in conditions if isinstance(item, dict)} == set(MATRIX), "proxy condition matrix is incomplete")
    for condition in conditions:
        require(isinstance(condition, dict) and set(condition) == {"code", "track", "effort", "cohort", "engine_language", "timing", "engine", "observation"}, "condition fields are invalid")
        expected = MATRIX[condition["code"]]
        require(tuple(condition[key] for key in ("track", "effort", "cohort", "engine_language")) == expected, f"{condition['code']}: matrix assignment drifted")
        timing = condition["timing"]
        require(isinstance(timing, dict) and set(timing) == TIME_KEYS, f"{condition['code']}: timing fields are invalid")
        for key, stamp in timing.items():
            require(stamp is None or isinstance(stamp, int) and not isinstance(stamp, bool) and stamp >= 0, f"{condition['code']}: {key} is not an epoch Int or null")
        require(timing["start"] is not None and timing["session_a_stop"] is not None and timing["stop"] is not None, f"{condition['code']}: required timestamps are missing")
        ordered = [timing[key] for key in ("start", "understood", "first_bounded_run", "session_a_stop", "desk_boot", "first_receipt", "cascade003", "stop") if timing[key] is not None]
        require(ordered == sorted(ordered), f"{condition['code']}: timestamps are not monotonic")
        engine = condition["engine"]
        require(isinstance(engine, dict) and set(engine) == ENGINE_KEYS, f"{condition['code']}: engine fields are invalid")
        require(isinstance(engine["conformance_pass"], bool) and isinstance(engine["protocol_success_exercised"], bool), f"{condition['code']}: engine booleans are invalid")
        require(all(isinstance(engine[key], int) and not isinstance(engine[key], bool) and engine[key] >= 0 for key in ("vectors_passed", "cases_passed")), f"{condition['code']}: engine counts are invalid")
        require(isinstance(engine["ambiguities"], list) and all(isinstance(item, str) for item in engine["ambiguities"]), f"{condition['code']}: ambiguities are invalid")
        require(engine["unclassified_error"] is None or isinstance(engine["unclassified_error"], str), f"{condition['code']}: unclassified error is invalid")
        observation = condition["observation"]
        require(isinstance(observation, dict) and set(observation) == OBSERVATION_KEYS, f"{condition['code']}: observation fields are invalid")
    return conditions


def elapsed_minutes(start: int, milestone: int) -> int:
    return math.ceil(max(0, milestone - start) / 60)


def estimated(condition: dict[str, Any]) -> dict[str, Any]:
    timing = condition["timing"]
    factor, understanding_floor, desk_floor, receipt_floor = TIMING[condition["cohort"]]
    start = timing["start"]

    def milestone(name: str, floor: int) -> int | None:
        stamp = timing[name]
        return None if stamp is None else max(floor, elapsed_minutes(start, stamp) * factor)

    return {
        "understanding_minutes": milestone("understood", understanding_floor),
        "desk_boot_minutes": milestone("desk_boot", desk_floor),
        "first_receipt_minutes": milestone("first_receipt", receipt_floor),
        "cre_minutes": min(300, elapsed_minutes(start, timing["session_a_stop"]) * factor),
        "raw_ai_seconds": {key: None if timing[key] is None else timing[key] - start for key in TIME_KEYS - {"start"}},
        "multiplier": factor,
    }


def scaled_minutes(value: int | None, numerator: int, denominator: int) -> int | None:
    if value is None:
        return None
    result = (value * numerator + denominator - 1) // denominator
    return None if result > 300 else result


def human_campaign(value: dict[str, Any], conditions: list[dict[str, Any]], scale: tuple[int, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    estimates = []
    teams = []
    for condition in conditions:
        estimate = estimated(condition)
        estimates.append({"code": condition["code"], **estimate})
        numerator, denominator = scale
        team = {
            "id": "T-" + condition["code"].split("-", 1)[1],
            "cohort": condition["cohort"],
            "engine_language": condition["engine_language"],
            "first_receipt_minutes": scaled_minutes(estimate["first_receipt_minutes"], numerator, denominator),
            "desk_boot_minutes": scaled_minutes(estimate["desk_boot_minutes"], numerator, denominator),
            "cre_minutes": scaled_minutes(estimate["cre_minutes"], numerator, denominator) or 0,
            **condition["observation"],
        }
        teams.append(team)
    return {
        "format": human.FORMAT,
        "bundle": value["bundle"],
        "system": value["system"],
        "teams": teams,
    }, estimates


def low_effort_safe(conditions: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    failures = []
    stop_keys = {
        "canonicalization_dominated", "pulse_or_lens_required_second_language",
        "cascade_blind_search", "validity_score_confused", "genre_guessed_reveal",
        "irreversible_progress_loss", "dominant_case",
    }
    for condition in (item for item in conditions if item["effort"] == "low"):
        observation = condition["observation"]
        required_true = {
            "reached_cascade003", "projection_explained", "intended_observations_understood",
            "improved_valid_score", "computed_reveal",
        }
        for key in sorted(required_true):
            if not observation[key]:
                failures.append(f"{condition['code']}.{key}")
        if observation["max_hint_level"] >= 3:
            failures.append(f"{condition['code']}.max_hint_level")
        if observation["route_count"] < 2:
            failures.append(f"{condition['code']}.route_count")
        if observation["unrelated_families"]:
            failures.append(f"{condition['code']}.unrelated_families")
        for key in sorted(stop_keys):
            if observation[key]:
                failures.append(f"{condition['code']}.{key}")
    return not failures, failures


def analyze(value: dict[str, Any]) -> dict[str, Any]:
    conditions = validate(value)
    central_campaign, estimates = human_campaign(value, conditions, (1, 1))
    low_campaign, _ = human_campaign(value, conditions, (3, 4))
    high_campaign, _ = human_campaign(value, conditions, (3, 2))
    central = human.analyze(central_campaign)
    sensitivity_low = human.analyze(low_campaign)
    sensitivity_high = human.analyze(high_campaign)
    low_safe, low_failures = low_effort_safe(conditions)
    if central["decision"] == "stop":
        decision = "proxy-stop"
    elif central["decision"] == "pass" and low_safe:
        decision = "proxy-pass"
    else:
        decision = "proxy-revise"
    return {
        "format": DECISION_FORMAT,
        "proxy_campaign": cre.digest_id("afterimage/ai-proxy-campaign/1", cre.canonical_bytes(value)),
        "bundle": value["bundle"],
        "decision": decision,
        "model_selector_available": False,
        "native_effort_available": False,
        "timing_basis": "estimated-human-equivalent/cohort-multiplier-v1",
        "estimates": estimates,
        "central": central,
        "sensitivity": {"0.75x": sensitivity_low, "1.5x": sensitivity_high},
        "low_effort_safe": {"passed": low_safe, "failures": low_failures},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = analyze(load(args.campaign))
        code = 0 if result["decision"] == "proxy-pass" else 1
    except (ProxyError, cre.CREError, human.CampaignError) as exc:
        result = {"type": "error", "code": "ai_proxy_invalid", "message": str(exc), "context": {}}
        code = 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2 if args.pretty else None, separators=None if args.pretty else (",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
