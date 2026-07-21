#!/usr/bin/env python3
"""Validate anonymized Afterimage observations and decide the slice playtest gate."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import cre  # noqa: E402


FORMAT = "afterimage-playtest-campaign/0.1"
RECORD_REQUIRED = "RECORD_REQUIRED"
COHORTS = {"runtime-builder", "algorithmic-contestant", "curious-programmer"}
COHORT_ORDER = ("runtime-builder", "algorithmic-contestant", "curious-programmer")
SLICE_FAMILIES = {"ORIENT", "CASCADE", "MERGE", "PULSE", "MOSAIC", "LENS"}
ENGINE_LANGUAGES = {
    "c", "cpp", "csharp", "elixir", "erlang", "go", "haskell", "java",
    "javascript", "kotlin", "lua", "ocaml", "other", "php", "python",
    "ruby", "rust", "scala", "swift", "typescript", "zig",
}
MAX_CAMPAIGN_BYTES = 1024 * 1024
MAX_MINUTES = 7 * 24 * 60
TEAM_KEYS = {
    "id", "cohort", "engine_language", "first_receipt_minutes", "desk_boot_minutes",
    "reached_cascade003", "max_hint_level", "projection_explained",
    "intended_observations_understood", "independent_valid_families",
    "improved_valid_score", "unrelated_families", "dominant_case",
    "computed_reveal", "route_count", "cre_minutes",
    "canonicalization_dominated", "pulse_or_lens_required_second_language",
    "cascade_blind_search", "validity_score_confused", "genre_guessed_reveal",
    "irreversible_progress_loss",
}
SYSTEM_KEYS = {
    "participant_isolation_verified", "kernel_receipt_agreement",
    "all_cases_reachable", "all_cases_precise_diagnostics",
    "acceptance_deterministic", "reset_replay_lossless", "offline_verified",
    "verifier_cheaper_than_search", "semantic_invalidated_cases",
}


class CampaignError(Exception):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}


def fail(code: str, message: str, **context: Any) -> None:
    raise CampaignError(code, message, context)


def integer(value: Any, location: str, minimum: int = 0, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        fail(
            "campaign_schema",
            "field must be a bounded Int",
            path=location,
            minimum=minimum,
            maximum=maximum,
        )
    return value


def boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        fail("campaign_schema", "field must be Bool", path=location)
    return value


def text_list(value: Any, location: str, allowed: set[str]) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item in allowed for item in value) or len(set(value)) != len(value):
        fail("campaign_schema", "field must be a unique allowed Text list", path=location)
    return value


def unrecorded_paths(value: Any, location: str = "") -> list[str]:
    if value == RECORD_REQUIRED:
        return [location or "$"]
    if isinstance(value, list):
        return [
            path
            for index, item in enumerate(value)
            for path in unrecorded_paths(item, f"{location}[{index}]")
        ]
    if isinstance(value, dict):
        return [
            path
            for key, item in value.items()
            for path in unrecorded_paths(item, f"{location}.{key}" if location else key)
        ]
    return []


def anonymous_team_ids(count: int) -> list[str]:
    result: set[str] = set()
    while len(result) < count:
        result.add(f"T-{secrets.randbelow(100_000_000):08d}")
    return sorted(result)


def new_campaign_draft(bundle: Any, count: Any) -> dict[str, Any]:
    try:
        cre.parse_id(bundle, "campaign bundle")
    except cre.CREError as exc:
        raise CampaignError("campaign_schema", exc.message, exc.context) from exc
    integer(count, "teams", minimum=6, maximum=64)
    cohorts = [cohort for cohort in COHORT_ORDER for _ in range(2)]
    cohorts.extend(COHORT_ORDER[index % len(COHORT_ORDER)] for index in range(count - 6))
    ids = anonymous_team_ids(count)
    teams = []
    for team_id, cohort in zip(ids, cohorts, strict=True):
        team = {key: RECORD_REQUIRED for key in TEAM_KEYS}
        team["id"] = team_id
        team["cohort"] = cohort
        teams.append(team)
    return {
        "format": FORMAT,
        "bundle": bundle,
        "system": {key: RECORD_REQUIRED for key in SYSTEM_KEYS},
        "teams": teams,
    }


def validate_campaign(value: Any) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    missing = unrecorded_paths(value)
    if missing:
        fail(
            "campaign_incomplete",
            "campaign still contains unrecorded fields",
            count=len(missing),
            paths=missing,
        )
    if not isinstance(value, dict) or set(value) != {"format", "bundle", "system", "teams"} or value.get("format") != FORMAT:
        fail("campaign_schema", "campaign envelope is invalid")
    try:
        cre.parse_id(value["bundle"], "campaign bundle")
    except cre.CREError as exc:
        raise CampaignError("campaign_schema", exc.message, exc.context) from exc
    system = value["system"]
    if not isinstance(system, dict) or set(system) != SYSTEM_KEYS:
        fail("campaign_schema", "system evidence fields are invalid")
    for key in SYSTEM_KEYS - {"semantic_invalidated_cases"}:
        boolean(system[key], f"system.{key}")
    integer(system["semantic_invalidated_cases"], "system.semantic_invalidated_cases", maximum=75)
    teams = value["teams"]
    if not isinstance(teams, list) or len(teams) < 6 or len(teams) > 64:
        fail("campaign_schema", "campaign must contain 6 through 64 teams")
    seen: set[str] = set()
    normalized = []
    for index, team in enumerate(teams):
        location = f"teams[{index}]"
        if not isinstance(team, dict) or set(team) != TEAM_KEYS:
            fail("campaign_schema", "team observation fields are invalid", path=location)
        team_id = team["id"]
        if (
            not isinstance(team_id, str)
            or not team_id.startswith("T-")
            or not 4 <= len(team_id[2:]) <= 12
            or team_id in seen
            or not team_id[2:].isascii()
            or not team_id[2:].isdigit()
        ):
            fail("campaign_schema", "team id must be a unique anonymous T- plus 4-12 digit code", path=f"{location}.id")
        seen.add(team_id)
        if team["cohort"] not in COHORTS:
            fail("campaign_schema", "team cohort is invalid", path=f"{location}.cohort")
        if team["engine_language"] not in ENGINE_LANGUAGES:
            fail("campaign_schema", "engine_language is not in the controlled vocabulary", path=f"{location}.engine_language")
        if team["first_receipt_minutes"] is not None:
            integer(team["first_receipt_minutes"], f"{location}.first_receipt_minutes", maximum=300)
        if team["desk_boot_minutes"] is not None:
            integer(team["desk_boot_minutes"], f"{location}.desk_boot_minutes", maximum=300)
        integer(team["max_hint_level"], f"{location}.max_hint_level", maximum=3)
        integer(team["route_count"], f"{location}.route_count", maximum=1000)
        integer(team["cre_minutes"], f"{location}.cre_minutes", maximum=MAX_MINUTES)
        for key in TEAM_KEYS - {
            "id", "cohort", "engine_language", "first_receipt_minutes", "desk_boot_minutes",
            "max_hint_level", "route_count", "cre_minutes", "independent_valid_families", "unrelated_families",
        }:
            boolean(team[key], f"{location}.{key}")
        text_list(team["independent_valid_families"], f"{location}.independent_valid_families", SLICE_FAMILIES)
        text_list(team["unrelated_families"], f"{location}.unrelated_families", SLICE_FAMILIES)
        if team["computed_reveal"] and not team["reached_cascade003"]:
            fail("campaign_schema", "computed_reveal requires reached_cascade003", path=f"{location}.computed_reveal")
        normalized.append(team)
    return value["bundle"], system, normalized


def criterion(passed: bool, value: Any, threshold: str) -> dict[str, Any]:
    return {"passed": passed, "value": value, "threshold": threshold}


def stop_trigger(triggered: bool, value: Any, threshold: str) -> dict[str, Any]:
    return {"triggered": triggered, "value": value, "threshold": threshold}


def ratio_threshold(count: int, numerator: int, denominator: int) -> int:
    return (count * numerator + denominator - 1) // denominator


def exact_median(values: list[int | None]) -> tuple[Any, bool]:
    middle = len(values) // 2
    if len(values) % 2:
        value = values[middle]
        return value, value is not None and value <= 45
    lower, upper = values[middle - 1], values[middle]
    if lower is None or upper is None:
        return None, False
    numerator = lower + upper
    display: Any = numerator // 2 if numerator % 2 == 0 else {"numerator": numerator, "denominator": 2}
    return display, numerator <= 90


def read_campaign(path: Path, canonical: bool) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CAMPAIGN_BYTES:
        fail("campaign_input", "campaign must be a regular file no larger than 1 MiB")
    data = path.read_bytes()
    if canonical:
        return kit.decode_json(data, str(path))
    if data.startswith(b"\xef\xbb\xbf"):
        fail("campaign_input", "UTF-8 BOM is forbidden")
    try:
        parsed = json.loads(data.decode("utf-8", errors="strict"), object_pairs_hook=cre.object_pairs_no_duplicates)
        return cre.normalize_value(parsed)
    except cre.CREError as exc:
        raise CampaignError("campaign_input", exc.message, exc.context) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError("campaign_input", f"invalid JSON draft: {exc}") from exc


def write_new_file(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        fail("campaign_output", "canonical output already exists", path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def analyze(value: Any) -> dict[str, Any]:
    bundle, system, teams = validate_campaign(value)
    campaign_id = cre.digest_id("afterimage/playtest-campaign/1", cre.canonical_bytes(value))
    count = len(teams)
    first_receipts = sorted(
        (team["first_receipt_minutes"] for team in teams),
        key=lambda item: 301 if item is None else item,
    )
    median, median_pass = exact_median(first_receipts)
    p90 = first_receipts[(9 * count + 9) // 10 - 1]
    cohort_counts = {cohort: sum(team["cohort"] == cohort for team in teams) for cohort in sorted(COHORTS)}
    boots = sum(team["desk_boot_minutes"] is not None and team["desk_boot_minutes"] <= 300 for team in teams)
    reveal_without_l3 = sum(team["reached_cascade003"] and team["max_hint_level"] < 3 for team in teams)
    independent = sorted({family for team in teams for family in team["independent_valid_families"]})
    non_python_rust = sum(team["engine_language"] not in {"python", "rust"} for team in teams)
    two_thirds = ratio_threshold(count, 2, 3)
    five_sixths = ratio_threshold(count, 5, 6)
    one_half = ratio_threshold(count, 1, 2)

    hard = {
        "blind_isolation": criterion(system["participant_isolation_verified"], system["participant_isolation_verified"], "true"),
        "kernel_receipt_agreement": criterion(system["kernel_receipt_agreement"], system["kernel_receipt_agreement"], "true"),
        "slice_reachability": criterion(system["all_cases_reachable"], system["all_cases_reachable"], "true"),
        "cohort_balance": criterion(all(value >= 2 for value in cohort_counts.values()), cohort_counts, ">=2 per cohort"),
        "ecosystem_diversity": criterion(non_python_rust >= 1, non_python_rust, ">=1 non-Python/non-Rust engine"),
        "first_receipt_median": criterion(median_pass, median, "<=45 minutes"),
        "first_receipt_p90": criterion(p90 is not None and p90 <= 90, p90, "<=90 minutes"),
        "desk_boot": criterion(boots >= two_thirds, boots, f">={two_thirds} teams within 300 minutes"),
        "reveal_without_level3": criterion(reveal_without_l3 >= two_thirds, reveal_without_l3, f">={two_thirds} teams"),
        "lossless_progress": criterion(not any(team["irreversible_progress_loss"] for team in teams) and system["reset_replay_lossless"], sum(team["irreversible_progress_loss"] for team in teams), "0 losses and system replay verified"),
        "precise_diagnostics": criterion(system["all_cases_precise_diagnostics"], system["all_cases_precise_diagnostics"], "true"),
        "independent_family_solutions": criterion(set(independent) == SLICE_FAMILIES, independent, "all six slice families"),
        "deterministic_acceptance": criterion(system["acceptance_deterministic"], system["acceptance_deterministic"], "true"),
        "offline_play": criterion(system["offline_verified"], system["offline_verified"], "true"),
    }

    unrelated_counts = {family: sum(family in team["unrelated_families"] for team in teams) for family in sorted(SLICE_FAMILIES)}
    reached = [team for team in teams if team["reached_cascade003"]]
    quality = {
        "projection_explained": criterion(sum(team["projection_explained"] for team in teams) >= five_sixths, sum(team["projection_explained"] for team in teams), f">={five_sixths} teams"),
        "intended_observations": criterion(sum(team["intended_observations_understood"] for team in teams) >= two_thirds, sum(team["intended_observations_understood"] for team in teams), f">={two_thirds} teams"),
        "no_majority_filler": criterion(all(value <= count // 2 for value in unrelated_counts.values()), unrelated_counts, "no family named by a majority"),
        "score_improvement": criterion(sum(team["improved_valid_score"] for team in teams) >= one_half, sum(team["improved_valid_score"] for team in teams), f">={one_half} teams"),
        "no_dominant_case": criterion(sum(team["dominant_case"] for team in teams) <= count // 2, sum(team["dominant_case"] for team in teams), "not a majority"),
        "computed_reveal": criterion(bool(reached) and all(team["computed_reveal"] for team in reached), sum(team["computed_reveal"] for team in reached), "all teams reaching reveal"),
        "two_routes": criterion(sum(team["route_count"] >= 2 for team in teams) >= five_sixths, sum(team["route_count"] >= 2 for team in teams), f">={five_sixths} teams"),
    }

    stuck = sum(team["cre_minutes"] >= 240 and team["desk_boot_minutes"] is None for team in teams)
    stops = {
        "cre_stall": stop_trigger(stuck * 4 > count, stuck, ">25% teams"),
        "canonicalization_dominated": stop_trigger(any(team["canonicalization_dominated"] for team in teams), sum(team["canonicalization_dominated"] for team in teams), ">=1 team"),
        "second_language_required": stop_trigger(any(team["pulse_or_lens_required_second_language"] for team in teams), sum(team["pulse_or_lens_required_second_language"] for team in teams), ">=1 team"),
        "cascade_blind_search": stop_trigger(any(team["cascade_blind_search"] for team in teams), sum(team["cascade_blind_search"] for team in teams), ">=1 team"),
        "validity_score_confused": stop_trigger(any(team["validity_score_confused"] for team in teams), sum(team["validity_score_confused"] for team in teams), ">=1 team"),
        "genre_guessed_reveal": stop_trigger(any(team["genre_guessed_reveal"] for team in teams), sum(team["genre_guessed_reveal"] for team in teams), ">=1 team"),
        "verifier_too_expensive": stop_trigger(not system["verifier_cheaper_than_search"], system["verifier_cheaper_than_search"], "false"),
        "semantic_blast_radius": stop_trigger(system["semantic_invalidated_cases"] > 2, system["semantic_invalidated_cases"], ">2 cases"),
    }
    hard_pass = all(item["passed"] for item in hard.values())
    quality_pass = all(item["passed"] for item in quality.values())
    stopped = any(item["triggered"] for item in stops.values())
    decision = "pass" if hard_pass and quality_pass and not stopped else "stop" if stopped else "revise"
    return {
        "format": "afterimage-playtest-decision/0.1",
        "campaign": campaign_id,
        "bundle": bundle,
        "teams": count,
        "decision": decision,
        "hard": hard,
        "quality": quality,
        "stop_triggers": stops,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, nargs="?")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--canonicalize", metavar="OUTPUT", type=Path, help="validate a pretty JSON draft and write canonical JSON without deciding")
    parser.add_argument("--new", metavar="OUTPUT", type=Path, help="create a private six-or-more-team observation draft")
    parser.add_argument("--bundle", help="BundleId for --new")
    parser.add_argument("--teams", type=int, help="team count for --new (default: 6)")
    args = parser.parse_args(argv)
    try:
        if args.new is not None:
            if args.campaign is not None or args.canonicalize is not None or args.bundle is None:
                fail("campaign_usage", "--new requires --bundle and cannot be combined with campaign or --canonicalize")
            value = new_campaign_draft(args.bundle, 6 if args.teams is None else args.teams)
            draft = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            write_new_file(args.new, draft)
            result = {"type": "campaign-draft-created", "bundle": args.bundle, "teams": len(value["teams"])}
            code = 0
        elif args.campaign is None or args.bundle is not None or args.teams is not None:
            fail("campaign_usage", "analysis requires CAMPAIGN; --bundle and --teams are only valid with --new")
        else:
            value = read_campaign(args.campaign, canonical=args.canonicalize is None)
            if args.canonicalize is not None:
                _, _, teams = validate_campaign(value)
                canonical = cre.canonical_bytes(value)
                write_new_file(args.canonicalize, canonical)
                result = {
                    "type": "campaign-canonicalized",
                    "campaign": cre.digest_id("afterimage/playtest-campaign/1", canonical),
                    "teams": len(teams),
                }
                code = 0
            else:
                result = analyze(value)
                code = 0 if result["decision"] == "pass" else 1
    except CampaignError as exc:
        result = {"type": "error", "code": exc.code, "message": exc.message, "context": exc.context}
        code = 2
    except (kit.KitError, OSError) as exc:
        result = {"type": "error", "code": getattr(exc, "code", "input_error"), "message": getattr(exc, "message", str(exc)), "context": getattr(exc, "context", {})}
        code = 2
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True, separators=None if args.pretty else (",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
