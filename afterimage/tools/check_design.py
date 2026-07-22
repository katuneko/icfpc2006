#!/usr/bin/env python3
"""Check machine-readable Afterimage design invariants.

This script intentionally has no third-party dependencies. It checks only
design consistency; it is not a game verifier.
"""

from __future__ import annotations

import json
import itertools
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCUMENTS = [
    ROOT / "README.md",
    ROOT / "design_bible.md",
    ROOT / "vertical_slice.md",
    ROOT / "spec" / "causal_reduction_engine.md",
    ROOT / "spec" / "conformance_fixture.md",
    ROOT / "spec" / "pulse_language.md",
    ROOT / "spec" / "merge_certificate.md",
    ROOT / "spec" / "mosaic_certificate.md",
    ROOT / "spec" / "lens_language.md",
    ROOT / "spec" / "covenant_language.md",
    ROOT / "spec" / "paradox_certificate.md",
    ROOT / "spec" / "bundle_and_witness.md",
    ROOT / "spec" / "localization.md",
    ROOT / "spec" / "scoring.md",
    ROOT / "playtest" / "PARTICIPANT_QUICKSTART.md",
    ROOT / "playtest" / "ENGINE_TASK.md",
    ROOT / "playtest" / "OPERATOR_PROTOCOL.md",
    ROOT / "playtest" / "INTERVIEW.md",
    ROOT / "playtest" / "OBSERVATION_SCHEMA.md",
    ROOT / "playtest" / "AI_SIMULATION_2026-07-15.md",
    ROOT / "playtest" / "AI_PROXY_PROTOCOL.md",
    ROOT / "playtest" / "AI_PROXY_RESULT_2026-07-16.md",
    ROOT / "playtest" / "ai_proxy_campaign_2026-07-16.json",
    ROOT / "playtest" / "ai_proxy_decision_2026-07-16.json",
    ROOT / "manifests" / "production_catalog.json",
    ROOT / "public" / "README.md",
    ROOT / "public" / "brand" / "BRAND_GUIDE.md",
    ROOT / "public" / "press" / "FACT_SHEET.md",
    ROOT / "public" / "press" / "FAQ.md",
    ROOT / "public" / "press" / "SPOILER_GUIDE.md",
    ROOT / "public" / "press" / "PUBLISHING_CHECKLIST.md",
    ROOT / "public" / "release" / "PLAYER_QUICKSTART.md",
    ROOT / "public" / "release" / "ENGINE_QUICKSTART.md",
]

REQUIRED_KERNEL_FILES = [
    ROOT / "reference" / "python" / "cre.py",
    ROOT / "reference" / "javascript" / "cre.mjs",
    ROOT / "tests" / "conformance" / "suite.json",
    ROOT / "tests" / "conformance" / "matrix.py",
    ROOT / "tests" / "conformance" / "full-suite.json",
    ROOT / "tests" / "conformance" / "golden.json",
    ROOT / "tests" / "conformance" / "check.py",
    ROOT / "tests" / "test_conformance_public.py",
    ROOT / "tests" / "conformance" / "expected.sha256",
    ROOT / "tools" / "run_conformance.py",
    ROOT / "tools" / "trace_oracle.py",
    ROOT / "tools" / "afterimage_kit.py",
    ROOT / "tests" / "test_afterimage_kit.py",
    ROOT / "tools" / "check_all.py",
    ROOT / "tools" / "verify_witness.py",
    ROOT / "tests" / "test_verify_witness.py",
    ROOT / "tools" / "authoring.py",
    ROOT / "tools" / "build_slice.py",
    ROOT / "content" / "vertical_slice" / "cases" / "ORIENT.001.json",
    ROOT / "content" / "vertical_slice" / "cases" / "ORIENT.002.json",
    ROOT / "content" / "vertical_slice" / "cases" / "ORIENT.003.json",
    ROOT / "content" / "vertical_slice" / "cases" / "ORIENT.004.json",
    ROOT / "content" / "vertical_slice" / "cases" / "ORIENT.005.json",
    ROOT / "content" / "vertical_slice" / "cases" / "CASCADE.001.json",
    ROOT / "content" / "vertical_slice" / "cases" / "CASCADE.002.json",
    ROOT / "content" / "vertical_slice" / "cases" / "CASCADE.003.json",
    ROOT / "content" / "vertical_slice" / "cases" / "MERGE.001.json",
    ROOT / "content" / "vertical_slice" / "cases" / "PULSE.001.json",
    ROOT / "content" / "vertical_slice" / "cases" / "MOSAIC.001.json",
    ROOT / "content" / "vertical_slice" / "cases" / "LENS.001.json",
    ROOT / "tools" / "lens.py",
    ROOT / "tests" / "test_lens.py",
    ROOT / "tools" / "mosaic.py",
    ROOT / "tests" / "test_mosaic.py",
    ROOT / "tools" / "pulse.py",
    ROOT / "tests" / "test_pulse.py",
    ROOT / "content" / "vertical_slice" / "golden.json",
    ROOT / "tests" / "test_slice_content.py",
    ROOT / "tools" / "player.py",
    ROOT / "tools" / "localization.py",
    ROOT / "tools" / "build_locales.py",
    ROOT / "tests" / "test_localization.py",
    ROOT / "locales" / "en.json",
    ROOT / "locales" / "ja.json",
    ROOT / "locales" / "zh-Hans.json",
    ROOT / "locales" / "de.json",
    ROOT / "tools" / "prepare_playtest.py",
    ROOT / "tests" / "test_player.py",
    ROOT / "tools" / "covenant.py",
    ROOT / "tests" / "test_covenant.py",
    ROOT / "tools" / "paradox.py",
    ROOT / "tests" / "test_paradox.py",
    ROOT / "tools" / "analyze_playtest.py",
    ROOT / "tests" / "test_playtest_analysis.py",
    ROOT / "tools" / "analyze_ai_proxy.py",
    ROOT / "tests" / "test_ai_proxy.py",
    ROOT / "tools" / "check_engine_generalization.py",
    ROOT / "tools" / "build_catalog.py",
    ROOT / "content" / "production" / "cases" / "PULSE.002.json",
    ROOT / "content" / "production" / "cases" / "PULSE.003.json",
    ROOT / "content" / "production" / "cases" / "MERGE.002.json",
    ROOT / "content" / "production" / "cases" / "MOSAIC.002.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.004.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.005.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.006.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.007.json",
    ROOT / "content" / "production" / "cases" / "MERGE.003.json",
    ROOT / "content" / "production" / "cases" / "MOSAIC.003.json",
    ROOT / "content" / "production" / "cases" / "PULSE.004.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.008.json",
    ROOT / "content" / "production" / "cases" / "MERGE.004.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.009.json",
    ROOT / "content" / "production" / "cases" / "MERGE.005.json",
    ROOT / "content" / "production" / "cases" / "PULSE.005.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.010.json",
    ROOT / "content" / "production" / "cases" / "MERGE.006.json",
    ROOT / "content" / "production" / "cases" / "MOSAIC.004.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.011.json",
    ROOT / "content" / "production" / "cases" / "MERGE.007.json",
    ROOT / "content" / "production" / "cases" / "MOSAIC.005.json",
    ROOT / "content" / "production" / "cases" / "PULSE.006.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.012.json",
    ROOT / "content" / "production" / "cases" / "MERGE.008.json",
    ROOT / "content" / "production" / "cases" / "MOSAIC.006.json",
    ROOT / "content" / "production" / "cases" / "PULSE.007.json",
    ROOT / "content" / "production" / "cases" / "LENS.002.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.013.json",
    ROOT / "content" / "production" / "cases" / "MERGE.009.json",
    ROOT / "content" / "production" / "cases" / "MOSAIC.007.json",
    ROOT / "content" / "production" / "cases" / "PULSE.008.json",
    ROOT / "content" / "production" / "cases" / "COVENANT.001.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.014.json",
    ROOT / "content" / "production" / "cases" / "MERGE.010.json",
    ROOT / "content" / "production" / "cases" / "MOSAIC.008.json",
    ROOT / "content" / "production" / "cases" / "PULSE.009.json",
    ROOT / "content" / "production" / "cases" / "LENS.003.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.015.json",
    ROOT / "content" / "production" / "cases" / "MERGE.011.json",
    ROOT / "content" / "production" / "cases" / "MOSAIC.009.json",
    ROOT / "content" / "production" / "cases" / "PULSE.010.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.016.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.017.json",
    ROOT / "content" / "production" / "cases" / "MERGE.012.json",
    ROOT / "content" / "production" / "cases" / "MOSAIC.010.json",
    ROOT / "content" / "production" / "cases" / "PULSE.011.json",
    ROOT / "content" / "production" / "cases" / "COVENANT.002.json",
    ROOT / "content" / "production" / "cases" / "MERGE.013.json",
    ROOT / "content" / "production" / "cases" / "MOSAIC.011.json",
    ROOT / "content" / "production" / "cases" / "PULSE.012.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.018.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.019.json",
    ROOT / "content" / "production" / "cases" / "MERGE.014.json",
    ROOT / "content" / "production" / "cases" / "PULSE.013.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.020.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.021.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.022.json",
    ROOT / "content" / "production" / "cases" / "MERGE.015.json",
    ROOT / "content" / "production" / "cases" / "PULSE.014.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.023.json",
    ROOT / "content" / "production" / "cases" / "CASCADE.024.json",
    ROOT / "content" / "production" / "cases" / "PARADOX.001.json",
    ROOT / "manifests" / "production_release.json",
    ROOT / "content" / "production" / "golden.json",
    ROOT / "tests" / "test_production_content.py",
    ROOT / "tools" / "build_public_assets.sh",
    ROOT / "tools" / "build_public_manifest.py",
    ROOT / "tools" / "render_public_previews.py",
    ROOT / "tools" / "build_public_release.py",
    ROOT / "tools" / "check_public_assets.py",
    ROOT / "tests" / "test_public_release.py",
    ROOT / "public" / "copy" / "launch-copy.json",
    ROOT / "public" / "site" / "index.html",
    ROOT / "public" / "site" / "styles.css",
    ROOT / "public" / "site" / "app.js",
    ROOT / "public" / "assets" / "manifest.json",
    ROOT / "public" / "assets" / "brand" / "mark.svg",
    ROOT / "public" / "assets" / "source" / "key-art-master-final.png",
    ROOT / "public" / "assets" / "source" / "poster-art-master.png",
]


class DesignError(Exception):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DesignError(f"cannot load {path.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(value, dict):
        raise DesignError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def load_included_manifest(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    chain = set() if seen is None else set(seen)
    require(resolved not in chain, f"manifest include cycle at {path.relative_to(ROOT)}")
    chain.add(resolved)
    value = load_json(resolved)
    if "include" not in value:
        return value
    include = value["include"]
    require(isinstance(include, str) and include, f"{path.relative_to(ROOT)}: include must be a path")
    base = load_included_manifest(resolved.parent / include, chain)
    require(isinstance(base.get("cases"), list) and isinstance(value.get("cases"), list), f"{path.relative_to(ROOT)}: included manifests need case lists")
    merged = {**base, **value, "cases": [*base["cases"], *value["cases"]]}
    merged.pop("include", None)
    return merged


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DesignError(message)


def dependency_ids(expr: Any, location: str) -> Iterable[str]:
    require(isinstance(expr, dict), f"{location}: requires must be an object")
    allowed = {"all", "any", "at_least"}
    unknown = set(expr) - allowed
    require(not unknown, f"{location}: unknown unlock keys: {sorted(unknown)}")

    if "all" in expr:
        values = expr["all"]
        require(isinstance(values, list), f"{location}.all must be a list")
        for value in values:
            require(isinstance(value, str), f"{location}.all entries must be strings")
            yield value

    if "any" in expr:
        values = expr["any"]
        require(isinstance(values, list) and values, f"{location}.any must be a non-empty list")
        for value in values:
            require(isinstance(value, str), f"{location}.any entries must be strings")
            yield value

    if "at_least" in expr:
        value = expr["at_least"]
        require(isinstance(value, dict), f"{location}.at_least must be an object")
        require(set(value) == {"count", "of"}, f"{location}.at_least requires count and of")
        count = value["count"]
        options = value["of"]
        require(isinstance(count, int) and count > 0, f"{location}.at_least.count must be positive")
        require(isinstance(options, list) and options, f"{location}.at_least.of must be non-empty")
        require(count <= len(options), f"{location}.at_least.count exceeds option count")
        require(len(set(options)) == len(options), f"{location}.at_least.of contains duplicates")
        for option in options:
            require(isinstance(option, str), f"{location}.at_least.of entries must be strings")
            yield option


def case_fact(case_id: str) -> str:
    return f"case:{case_id}"


def reachable_cases(cases: list[dict[str, Any]]) -> set[str]:
    """Return cases unlockable under monotone accumulation of all valid facts."""

    unlocked: set[str] = set()
    facts: set[str] = set()

    def satisfied(expr: dict[str, Any], facts: set[str]) -> bool:
        if "all" in expr and not all(item in facts for item in expr["all"]):
            return False
        if "any" in expr and not any(item in facts for item in expr["any"]):
            return False
        if "at_least" in expr:
            spec = expr["at_least"]
            if sum(item in facts for item in spec["of"]) < spec["count"]:
                return False
        return True

    changed = True
    while changed:
        changed = False
        for case in cases:
            case_id = case["id"]
            if case_id not in unlocked and satisfied(case["requires"], facts):
                unlocked.add(case_id)
                facts.add(case_fact(case_id))
                changed = True
    return unlocked


def unlock_satisfied(expr: dict[str, Any], facts: set[str]) -> bool:
    if "all" in expr and not all(item in facts for item in expr["all"]):
        return False
    if "any" in expr and not any(item in facts for item in expr["any"]):
        return False
    if "at_least" in expr:
        spec = expr["at_least"]
        if sum(item in facts for item in spec["of"]) < spec["count"]:
            return False
    return True


def subset_is_solvable(subset: set[str], cases_by_id: dict[str, dict[str, Any]]) -> bool:
    solved: set[str] = set()
    facts: set[str] = set()
    changed = True
    while changed:
        changed = False
        for case_id in sorted(subset - solved):
            if unlock_satisfied(cases_by_id[case_id]["requires"], facts):
                solved.add(case_id)
                facts.add(case_fact(case_id))
                changed = True
    return solved == subset


def minimal_reveal_paths(cases: list[dict[str, Any]], reveal: str) -> list[set[str]]:
    cases_by_id = {case["id"]: case for case in cases}
    other_ids = sorted(set(cases_by_id) - {reveal})
    for extra_count in range(len(other_ids) + 1):
        paths: list[set[str]] = []
        for extras in itertools.combinations(other_ids, extra_count):
            subset = set(extras) | {reveal}
            if subset_is_solvable(subset, cases_by_id):
                paths.append(subset)
        if paths:
            return paths
    return []


def case_score(points: int, reference_scale: int, cost: int) -> tuple[int, int, int]:
    require(points > 0, "score points must be positive")
    require(reference_scale > 0, "score reference scale must be positive")
    require(cost > 0, "score cost must be positive")
    completion = (65 * points + 99) // 100
    pool = points - completion
    quality_ppm = 1_000_000 * reference_scale // (reference_scale + cost)
    optimization = pool * quality_ppm // 1_000_000
    return completion, optimization, completion + optimization


def check() -> list[str]:
    for document in REQUIRED_DOCUMENTS:
        require(document.is_file(), f"missing document: {document.relative_to(ROOT)}")
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            path_text = target.split("#", 1)[0]
            require(path_text != "", f"{document.relative_to(ROOT)}: empty link target")
            resolved = (document.parent / path_text).resolve()
            require(resolved.is_relative_to(ROOT), f"{document.relative_to(ROOT)}: link escapes root: {target}")
            require(resolved.exists(), f"{document.relative_to(ROOT)}: broken link: {target}")

    for kernel_file in REQUIRED_KERNEL_FILES:
        require(kernel_file.is_file(), f"missing kernel file: {kernel_file.relative_to(ROOT)}")

    full = load_json(ROOT / "manifests" / "full_scope.json")
    slice_manifest = load_json(ROOT / "manifests" / "vertical_slice.json")
    catalog = load_json(ROOT / "manifests" / "production_catalog.json")
    production_release = load_included_manifest(ROOT / "manifests" / "production_release.json")

    families = full.get("families")
    require(isinstance(families, list) and families, "full_scope: families must be non-empty")
    family_ids = [family.get("id") for family in families]
    require(all(isinstance(value, str) and value for value in family_ids), "family IDs must be non-empty strings")
    require(len(family_ids) == len(set(family_ids)), "family IDs must be unique")

    case_total = sum(family.get("cases", -1) for family in families)
    point_total = sum(family.get("nominal_points", -1) for family in families)
    require(case_total == full.get("total_cases") == 75, f"full case total mismatch: {case_total}")
    require(point_total == full.get("total_nominal_points") == 10000, f"full point total mismatch: {point_total}")
    require(full.get("completion_percent") == 65, "completion percent must remain 65 at design gate")

    catalog_cases = catalog.get("cases")
    require(catalog.get("format") == "afterimage-production-catalog/0.1", "production catalog format mismatch")
    require(isinstance(catalog_cases, list), "production catalog cases must be a list")
    catalog_ids = [case.get("id") for case in catalog_cases]
    require(len(catalog_ids) == len(set(catalog_ids)) == catalog.get("total_cases") == case_total, "production catalog case total or uniqueness mismatch")
    require(sum(case.get("points", -1) for case in catalog_cases) == catalog.get("total_nominal_points") == point_total, "production catalog point total mismatch")
    catalog_id_set = set(catalog_ids)
    catalog_facts = {case_fact(case_id) for case_id in catalog_ids}
    catalog_family_counts = {family_id: 0 for family_id in family_ids}
    catalog_family_points = {family_id: 0 for family_id in family_ids}
    expected_catalog_keys = {"id", "family", "title", "points", "act", "band", "mechanic", "requires", "wave", "status"}
    for index, case in enumerate(catalog_cases):
        location = f"production_catalog.cases[{index}]"
        require(set(case) == expected_catalog_keys, f"{location}: fields mismatch")
        case_id = case["id"]
        family = case["family"]
        require(family in catalog_family_counts and re.fullmatch(rf"{family}\.\d{{3}}", case_id) is not None, f"{location}: family or ID mismatch")
        require(isinstance(case["title"], str) and case["title"], f"{location}: title missing")
        require(isinstance(case["points"], int) and case["points"] > 0, f"{location}: points invalid")
        require(isinstance(case["act"], str) and case["act"], f"{location}: act missing")
        require(isinstance(case["mechanic"], str) and case["mechanic"], f"{location}: mechanic missing")
        require(isinstance(case["band"], int) and 1 <= case["band"] <= 5, f"{location}: band invalid")
        require(case["status"] in {"golden", "authored", "planned"}, f"{location}: status invalid")
        require(isinstance(case["wave"], str) and case["wave"], f"{location}: wave missing")
        deps = list(dependency_ids(case["requires"], f"{location}.requires"))
        require(case_fact(case_id) not in deps, f"{location}: self-dependency")
        require(all(dep in catalog_facts for dep in deps), f"{location}: dependency outside catalog")
        catalog_family_counts[family] += 1
        catalog_family_points[family] += case["points"]
    for family in families:
        family_id = family["id"]
        require(catalog_family_counts[family_id] == family["cases"], f"catalog {family_id} case count mismatch")
        require(catalog_family_points[family_id] == family["nominal_points"], f"catalog {family_id} point total mismatch")
        ordinals = sorted(int(case["id"].split(".")[1]) for case in catalog_cases if case["family"] == family_id)
        require(ordinals == list(range(1, family["cases"] + 1)), f"catalog {family_id} ordinals are not contiguous")
    catalog_reachable = reachable_cases(catalog_cases)
    require(catalog_reachable == catalog_id_set, f"unreachable production cases: {sorted(catalog_id_set - catalog_reachable)}")

    cases = slice_manifest.get("cases")
    require(isinstance(cases, list) and cases, "vertical_slice: cases must be non-empty")
    ids = [case.get("id") for case in cases]
    require(all(isinstance(value, str) and value for value in ids), "slice case IDs must be non-empty strings")
    require(len(ids) == len(set(ids)), "slice case IDs must be unique")
    require(len(cases) == slice_manifest.get("total_cases") == full["vertical_slice"]["cases"] == 12, "slice case total mismatch")

    points = sum(case.get("points", -1) for case in cases)
    require(points == slice_manifest.get("total_nominal_points") == full["vertical_slice"]["nominal_points"] == 1200, f"slice point total mismatch: {points}")

    known_ids = set(ids)
    family_set = set(family_ids)
    known_facts = {case_fact(case_id) for case_id in known_ids}
    previous_facts: set[str] = set()
    for index, case in enumerate(cases):
        location = f"vertical_slice.cases[{index}]"
        case_id = case["id"]
        family = case.get("family")
        require(family in family_set, f"{location}: unknown family {family!r}")
        require(case_id.startswith(f"{family}."), f"{location}: ID does not match family")
        require(isinstance(case.get("points"), int) and case["points"] > 0, f"{location}: points must be positive")
        require(isinstance(case.get("reference_scale"), int) and case["reference_scale"] > 0, f"{location}: reference_scale must be positive")
        require(isinstance(case.get("expected_minutes"), int) and case["expected_minutes"] > 0, f"{location}: expected_minutes must be positive")
        require(isinstance(case.get("aha"), str) and case["aha"].strip(), f"{location}: aha must be non-empty")
        require(isinstance(case.get("metrics"), list) and case["metrics"], f"{location}: metrics must be non-empty")
        deps = list(dependency_ids(case.get("requires"), f"{location}.requires"))
        require(all(dep in known_facts for dep in deps), f"{location}: dependency refers outside slice")
        require(case_fact(case_id) not in deps, f"{location}: case depends on itself")
        # File order should remain a readable topological order even though
        # unlock expressions can offer alternatives.
        require(all(dep in previous_facts for dep in deps), f"{location}: dependency appears after case")
        previous_facts.add(case_fact(case_id))

    unlocked = reachable_cases(cases)
    require(unlocked == known_ids, f"unreachable slice cases: {sorted(known_ids - unlocked)}")
    reveal = slice_manifest.get("narrative_reveal_case")
    require(reveal in known_ids, "narrative reveal case is not in slice")
    require(reveal.startswith("CASCADE."), "narrative reveal must be a CASCADE case")
    reveal_paths = minimal_reveal_paths(cases, reveal)
    require(len(reveal_paths) >= 2, "narrative reveal needs at least two minimal paths")
    path_families = [{case_id.split(".", 1)[0] for case_id in path} for path in reveal_paths]
    require(any("MERGE" in families for families in path_families), "no minimal reveal path through MERGE")
    require(any("PULSE" in families for families in path_families), "no minimal reveal path through PULSE")
    cases_by_id = {case["id"]: case for case in cases}
    shortest_expected = min(
        sum(cases_by_id[case_id]["expected_minutes"] for case_id in path)
        for path in reveal_paths
    )
    require(
        shortest_expected <= slice_manifest.get("critical_path_target_minutes", 0),
        f"expected critical path {shortest_expected}m exceeds target",
    )

    family_counts: dict[str, int] = {family_id: 0 for family_id in family_ids}
    for case in cases:
        family_counts[case["family"]] += 1
    expected_slice_counts = {
        "ORIENT": 5,
        "CASCADE": 3,
        "MERGE": 1,
        "PULSE": 1,
        "MOSAIC": 1,
        "LENS": 1,
        "COVENANT": 0,
        "PARADOX": 0,
    }
    require(family_counts == expected_slice_counts, f"slice family distribution mismatch: {family_counts}")

    catalog_by_id = {case["id"]: case for case in catalog_cases}
    for case in cases:
        catalog_case = catalog_by_id[case["id"]]
        for field in ("family", "title", "points", "requires"):
            require(catalog_case[field] == case[field], f"catalog/slice mismatch for {case['id']} {field}")
        require(catalog_case["status"] == "golden", f"slice case {case['id']} must be golden in catalog")
    release_cases = production_release.get("cases")
    require(isinstance(release_cases, list), "production release cases must be a list")
    release_ids = {case["id"] for case in release_cases}
    authored_ids = {case["id"] for case in catalog_cases if case["status"] in {"golden", "authored"}}
    require(release_ids == authored_ids, f"production release/catalog authored mismatch: {sorted(release_ids ^ authored_ids)}")
    require(len(release_cases) == production_release.get("total_cases") == 75, "production release case total mismatch")
    require(sum(case["points"] for case in release_cases) == production_release.get("total_nominal_points") == 10000, "production release point total mismatch")

    vertical_text = (ROOT / "vertical_slice.md").read_text(encoding="utf-8")
    for case_id in ids:
        require(vertical_text.count(case_id) >= 2, f"vertical_slice.md lacks a detailed section for {case_id}")

    # Normative examples from spec/scoring.md.
    require(case_score(100, 1000, 1000) == (65, 17, 82), "scoring example C=1000 drifted")
    require(case_score(100, 1000, 250) == (65, 28, 93), "scoring example C=250 drifted")
    require(case_score(100, 1000, 4000) == (65, 7, 72), "scoring example C=4000 drifted")
    for case in cases:
        scale = case["reference_scale"]
        low = case_score(case["points"], scale, max(1, scale // 4))[2]
        middle = case_score(case["points"], scale, scale)[2]
        high = case_score(case["points"], scale, scale * 4)[2]
        require(low >= middle >= high, f"{case['id']}: score is not monotone")
        require(high >= (65 * case["points"] + 99) // 100, f"{case['id']}: valid score below completion")

    return [
        f"full scope: {case_total} cases, {point_total} points",
        f"production catalog: {len(catalog_cases)} reachable cases; family budgets exact",
        f"production release: {len(release_cases)} authored cases, {sum(case['points'] for case in release_cases)} points",
        f"vertical slice: {len(cases)} cases, {points} points",
        f"slice distribution: {family_counts}",
        f"all {len(unlocked)} slice cases reachable; reveal={reveal}",
        f"{len(reveal_paths)} minimal reveal paths of {len(reveal_paths[0])} cases; shortest expected={shortest_expected}m",
        f"{len(REQUIRED_DOCUMENTS)} required documents and {len(REQUIRED_KERNEL_FILES)} kernel files present; scoring examples stable",
    ]


def main() -> int:
    try:
        lines = check()
    except DesignError as exc:
        print(f"design check failed: {exc}", file=sys.stderr)
        return 1
    for line in lines:
        print(f"ok: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
