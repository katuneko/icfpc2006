#!/usr/bin/env python3
"""Offline player workspace and playtest CLI for the Afterimage vertical slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import cre  # noqa: E402
import localization  # noqa: E402
import verify_witness as verifier  # noqa: E402


STATE_FORMAT = "afterimage-player-state/0.1"
TELEMETRY_FORMAT = "afterimage-playtest-telemetry/0.1"
STATE_NAME = ".afterimage-player.json"
SAFE_TELEMETRY_FIELDS = {
    "event",
    "case",
    "valid",
    "code",
    "score",
    "metrics",
    "unlocks",
    "branch_count",
    "keep_witnesses",
    "hint_level",
    "visible_cases",
    "solved_cases",
}
MAX_CANONICALIZE_BYTES = 16 * 1024 * 1024


class PlayerError(Exception):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}

    def value(self) -> dict[str, Any]:
        return {"type": "error", "code": self.code, "message": self.message, "context": self.context}


def fail(code: str, message: str, **context: Any) -> None:
    raise PlayerError(code, message, context)


def canonical_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = cre.canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_canonical(path: Path) -> Any:
    try:
        return kit.decode_json(path.read_bytes(), str(path))
    except OSError as exc:
        raise PlayerError("input_error", f"cannot read {path.name}: {exc}") from exc
    except kit.KitError as exc:
        raise PlayerError(exc.code, exc.message, exc.context) from exc


def canonicalize_file(source: Path, target: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        fail("input_error", "canonicalize input must be a regular file")
    if target.exists() or target.is_symlink():
        fail("output_exists", "canonicalize output already exists")
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise PlayerError("input_error", f"cannot read canonicalize input: {exc}") from exc
    if len(data) > MAX_CANONICALIZE_BYTES:
        fail("limit_exceeded", "canonicalize input exceeds 16 MiB")
    if data.startswith(b"\xef\xbb\xbf"):
        fail("invalid_json", "UTF-8 BOM is forbidden")
    try:
        parsed = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=cre.object_pairs_no_duplicates,
        )
        value = cre.normalize_value(parsed)
    except cre.CREError as exc:
        raise PlayerError(exc.code, exc.message, exc.context) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlayerError("invalid_json", f"cannot parse JSON draft: {exc}") from exc
    encoded = cre.canonical_bytes(value)
    canonical_write(target, value)
    return {
        "type": "canonicalized",
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def state_path(workspace: Path) -> Path:
    return workspace / STATE_NAME


def load_workspace(workspace: Path) -> tuple[dict[str, Any], kit.ValidatedBundle]:
    if workspace.is_symlink() or not workspace.is_dir():
        fail("invalid_workspace", "workspace must be a real directory")
    if state_path(workspace).is_symlink() or not state_path(workspace).is_file():
        fail("invalid_workspace", "workspace state must be a regular file")
    for name in ("world", "witnesses", "receipts", "branches"):
        path = workspace / name
        if path.is_symlink() or not path.is_dir():
            fail("invalid_workspace", "workspace support path must be a real directory", path=name)
    telemetry_path = workspace / "telemetry.ndjson"
    if telemetry_path.exists() and (telemetry_path.is_symlink() or not telemetry_path.is_file()):
        fail("invalid_workspace", "workspace telemetry must be a regular file")
    state = read_canonical(state_path(workspace))
    expected = {"format", "bundle", "archive_sha256", "telemetry", "sequence", "last_unix_ms"}
    if not isinstance(state, dict) or set(state) != expected or state.get("format") != STATE_FORMAT:
        fail("invalid_workspace", "workspace state is invalid")
    if not isinstance(state["telemetry"], bool):
        fail("invalid_workspace", "workspace telemetry flag is invalid")
    if any(isinstance(state[key], bool) or not isinstance(state[key], int) or state[key] < 0 for key in ("sequence", "last_unix_ms")):
        fail("invalid_workspace", "workspace telemetry counters are invalid")
    try:
        world = kit.verify_world(workspace / "world")
    except kit.KitError as exc:
        raise PlayerError(exc.code, exc.message, exc.context) from exc
    if world.bundle != state["bundle"] or world.archive_sha256 != state["archive_sha256"]:
        fail("invalid_workspace", "workspace state does not match its extracted world")
    return state, world


def init_workspace(archive: Path, workspace: Path, telemetry: bool) -> dict[str, Any]:
    if workspace.exists() or workspace.is_symlink():
        fail("output_exists", "workspace already exists", path=str(workspace))
    try:
        bundle = kit.load_bundle(archive)
    except kit.KitError as exc:
        raise PlayerError(exc.code, exc.message, exc.context) from exc
    workspace.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{workspace.name}.", dir=workspace.parent))
    try:
        kit.extract_bundle(bundle, temporary / "world")
        for name in ("witnesses", "receipts", "branches"):
            (temporary / name).mkdir()
        state = {
            "format": STATE_FORMAT,
            "bundle": bundle.bundle,
            "archive_sha256": bundle.archive_sha256,
            "telemetry": telemetry,
            "sequence": 0,
            "last_unix_ms": 0,
        }
        canonical_write(temporary / STATE_NAME, state)
        os.replace(temporary, workspace)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"type": "initialized", "workspace": str(workspace), "bundle": bundle.bundle, "cases": len(bundle.json_values["cases/index.json"]["cases"]), "telemetry": telemetry}


def record(workspace: Path, state: dict[str, Any], fields: dict[str, Any]) -> None:
    if not state["telemetry"]:
        return
    if not set(fields) <= SAFE_TELEMETRY_FIELDS or "event" not in fields:
        fail("internal_error", "telemetry event contains a forbidden field")
    now = max(state["last_unix_ms"], time.time_ns() // 1_000_000)
    state["sequence"] += 1
    state["last_unix_ms"] = now
    event = {"format": TELEMETRY_FORMAT, "tick": state["sequence"], "unix_ms": now, **fields}
    telemetry_path = workspace / "telemetry.ndjson"
    if telemetry_path.exists() and (telemetry_path.is_symlink() or not telemetry_path.is_file()):
        fail("invalid_workspace", "workspace telemetry must be a regular file")
    with telemetry_path.open("ab") as stream:
        stream.write(cre.canonical_bytes(event) + b"\n")
    canonical_write(state_path(workspace), state)


def descriptors(world: kit.ValidatedBundle) -> dict[str, dict[str, Any]]:
    return {item["id"]: verifier.validate_case_descriptor(item, world) for item in world.json_values["cases/index.json"]["cases"]}


def locale_pack(locale_code: str, world: kit.ValidatedBundle | None = None) -> localization.LocalePack:
    try:
        return localization.load_pack(
            locale_code,
            bundle=None if world is None else world.bundle,
            expected_cases=None if world is None else set(descriptors(world)),
        )
    except localization.LocalizationError as exc:
        raise PlayerError("invalid_locale", str(exc)) from exc


def receipt_files(workspace: Path) -> list[Path]:
    return sorted((workspace / "receipts").glob("*.json"), key=lambda path: path.name.encode("utf-8"))


def retained_receipts(workspace: Path) -> list[dict[str, Any]]:
    receipts = []
    for path in receipt_files(workspace):
        if path.is_symlink() or not path.is_file():
            fail("invalid_workspace", "retained receipt must be a regular file", file=path.name)
        value = read_canonical(path)
        if not isinstance(value, dict) or value.get("format") != "afterimage-receipt/0.1" or value.get("valid") is not True:
            fail("invalid_workspace", "retained receipt is invalid", file=path.name)
        receipts.append(value)
    return receipts


def facts_from_receipts(receipts: Iterable[dict[str, Any]]) -> set[str]:
    facts: set[str] = set()
    for receipt in receipts:
        unlocks = receipt.get("unlocks", [])
        if not isinstance(unlocks, list) or not all(isinstance(item, str) for item in unlocks):
            fail("invalid_workspace", "retained receipt unlocks are invalid")
        facts.update(unlocks)
    return facts


def visible_cases(world: kit.ValidatedBundle, facts: set[str]) -> list[dict[str, Any]]:
    return [case for case in descriptors(world).values() if verifier.requirements_hold(case["requires"], facts)]


def status(workspace: Path, state: dict[str, Any], world: kit.ValidatedBundle) -> dict[str, Any]:
    receipts = retained_receipts(workspace)
    facts = facts_from_receipts(receipts)
    solved = sorted({item["case"] for item in receipts}, key=lambda item: item.encode("utf-8"))
    visible = sorted((item["id"] for item in visible_cases(world, facts)), key=lambda item: item.encode("utf-8"))
    result = {"type": "status", "bundle": world.bundle, "total_cases": len(descriptors(world)), "solved": solved, "visible": visible, "facts": sorted(facts, key=lambda item: item.encode("utf-8"))}
    record(workspace, state, {"event": "status", "visible_cases": len(visible), "solved_cases": len(solved)})
    return result


def require_visible_case(world: kit.ValidatedBundle, case_id: str, facts: set[str]) -> dict[str, Any]:
    case = descriptors(world).get(case_id)
    if case is None:
        fail("unknown_case", "case is absent from the bundle", case=case_id)
    if not verifier.requirements_hold(case["requires"], facts):
        fail("case_locked", "case prerequisites are not satisfied", case=case_id)
    return case


def inspect_target(workspace: Path, state: dict[str, Any], world: kit.ValidatedBundle, target: str, locale_code: str = "en") -> dict[str, Any]:
    facts = facts_from_receipts(retained_receipts(workspace))
    case_map = descriptors(world)
    if target in case_map:
        case = require_visible_case(world, target, facts)
        pack = locale_pack(locale_code, world)
        localized = pack.case(target)
        result = {
            "type": "case",
            "id": target,
            "family": case["family"],
            "locale": pack.locale,
            "title": localized["title"],
            "points": case["points"],
            "projection": case["projection"],
            "answer_schema": world.json_values[case["answer_schema"]],
            "story": pack.story(target),
        }
        record(workspace, state, {"event": "inspect", "case": target})
        return result
    try:
        cre.parse_id(target, "event")
    except cre.CREError as exc:
        raise PlayerError("unknown_target", "target is neither a case nor an EventId") from exc
    matches = []
    for case in visible_cases(world, facts):
        replay = verifier.replay_root_case(world, case)
        if target in replay.events:
            matches.append({"case": case["id"], "event": cre.event_view(replay.events[target])})
    if not matches:
        fail("unknown_event", "event is absent from every visible case")
    record(workspace, state, {"event": "inspect"})
    return {"type": "event", "matches": matches}


def trace_event(workspace: Path, state: dict[str, Any], world: kit.ValidatedBundle, event_id: str, direction: str) -> dict[str, Any]:
    try:
        cre.parse_id(event_id, "event")
    except cre.CREError as exc:
        raise PlayerError(exc.code, exc.message, exc.context) from exc
    facts = facts_from_receipts(retained_receipts(workspace))
    matches = []
    for case in visible_cases(world, facts):
        replay = verifier.replay_root_case(world, case)
        event = replay.events.get(event_id)
        if event is None:
            continue
        if direction == "parents":
            related = [cre.event_view(replay.events[parent]) for parent in event["parents"] if parent in replay.events]
        elif direction == "children":
            related = [cre.event_view(item) for item in replay.events.values() if event_id in item["parents"]]
            related.sort(key=lambda item: cre.parse_id(item["id"]))
        else:
            related = [item for item in replay.trace_items if item.get("event") == event_id]
        matches.append({"case": case["id"], "event": cre.event_view(event), direction: related})
    if not matches:
        fail("unknown_event", "event is absent from every visible case")
    record(workspace, state, {"event": "trace"})
    return {"type": "trace", "direction": direction, "matches": matches}


def verify_submission(workspace: Path, state: dict[str, Any], world: kit.ValidatedBundle, witness_path: Path) -> tuple[dict[str, Any], int]:
    facts = facts_from_receipts(retained_receipts(workspace))
    try:
        witness_bytes = witness_path.read_bytes()
        receipt = verifier.verify_witness_bytes(world, witness_bytes, str(witness_path), facts)
    except OSError as exc:
        raise PlayerError("input_error", f"cannot read witness: {exc}") from exc
    except verifier.VerificationError as exc:
        raise PlayerError(exc.code, exc.message, exc.context) from exc
    case_id = receipt["case"]
    if receipt["valid"]:
        digest = receipt["witness"].removeprefix("sha256:")
        witness_target = workspace / "witnesses" / f"{case_id}-{digest}.json"
        receipt_target = workspace / "receipts" / f"{case_id}-{digest}.json"
        canonical_write(witness_target, kit.decode_json(witness_bytes, str(witness_path)))
        canonical_write(receipt_target, receipt)
        record(workspace, state, {"event": "verify", "case": case_id, "valid": True, "score": receipt["score"], "metrics": receipt["metrics"], "unlocks": receipt["unlocks"]})
        return receipt, 0
    code = receipt["diagnostics"][0]["code"]
    record(workspace, state, {"event": "verify", "case": case_id, "valid": False, "code": code})
    return receipt, 1


def branch_case(
    workspace: Path,
    state: dict[str, Any],
    world: kit.ValidatedBundle,
    case_id: str,
    intervention_path: Path,
    history_source: str | None = None,
    include_trace_items: bool = False,
) -> dict[str, Any]:
    facts = facts_from_receipts(retained_receipts(workspace))
    case = require_visible_case(world, case_id, facts)
    intervention = read_canonical(intervention_path)
    history_value = None
    if history_source is not None:
        if history_source.startswith("sha256:"):
            history_value = load_branch(workspace, history_source)["history"]
        else:
            loaded = read_canonical(Path(history_source))
            history_value = loaded.get("history") if isinstance(loaded, dict) and loaded.get("format") == "afterimage-player-branch/0.1" else loaded
    input_history = verifier.resolve_input_history(world, case, history_value, facts)
    base_events = input_history.base_events
    parent = input_history.branch
    baseline = verifier.evaluate_case_state(world, case, base_events, parent)
    try:
        policy = verifier.validate_intervention(
            world.json_values[case["intervention_policy"]],
            intervention,
            witness_bundle=world.bundle,
            witness_case=case_id,
            parent_branch=parent,
            base_events=base_events,
            known_events=baseline.events,
        )
        replay = verifier.replay_case(world, case, policy.operations, parent, baseline, base_events)
    except verifier.VerificationError as exc:
        raise PlayerError(exc.code, exc.message, exc.context) from exc
    snapshot = {
        "format": "afterimage-player-branch/0.1",
        "bundle": world.bundle,
        "case": case_id,
        "parent_branch": parent,
        "branch": replay.branch,
        "projection": replay.projection,
        "trace": replay.trace,
        "records": replay.records,
        "changed_event_ids": replay.changed_event_ids,
        "counters": replay.counters,
        "intervention_weight": policy.weight,
        "history": (
            {
                "format": "afterimage-branch-history/0.1",
                "bundle": world.bundle,
                "world": case["world"],
                "steps": [*input_history.steps, {"case": case_id, "operations": policy.operations}],
            }
            if policy.operations
            else history_value
        ),
    }
    target = workspace / "branches" / f"{replay.branch.removeprefix('sha256:')}.json"
    canonical_write(target, snapshot)
    count = len(list((workspace / "branches").glob("*.json")))
    record(workspace, state, {"event": "branch", "case": case_id, "branch_count": count})
    return {**snapshot, "trace_items": replay.trace_items} if include_trace_items else snapshot


def load_branch(workspace: Path, branch_id: str) -> dict[str, Any]:
    try:
        cre.parse_id(branch_id, "branch")
    except cre.CREError as exc:
        raise PlayerError(exc.code, exc.message, exc.context) from exc
    path = workspace / "branches" / f"{branch_id.removeprefix('sha256:')}.json"
    if not path.is_file() or path.is_symlink():
        fail("unknown_branch", "branch snapshot is absent from this workspace")
    value = read_canonical(path)
    expected = {"format", "bundle", "case", "parent_branch", "branch", "projection", "trace", "records", "changed_event_ids", "counters", "intervention_weight", "history"}
    if not isinstance(value, dict) or set(value) != expected or value.get("format") != "afterimage-player-branch/0.1" or value.get("branch") != branch_id:
        fail("invalid_workspace", "branch snapshot is invalid")
    return value


def compare_branches(workspace: Path, state: dict[str, Any], first_id: str, second_id: str) -> dict[str, Any]:
    first = load_branch(workspace, first_id)
    second = load_branch(workspace, second_id)
    if first["bundle"] != second["bundle"] or first["case"] != second["case"]:
        fail("incomparable_branches", "branches must belong to the same bundle and case")
    first_records = {cre.canonical_text(item): item for item in first["records"]}
    second_records = {cre.canonical_text(item): item for item in second["records"]}
    result = {
        "type": "comparison",
        "case": first["case"],
        "first": first_id,
        "second": second_id,
        "only_first": [first_records[key] for key in sorted(set(first_records) - set(second_records))],
        "only_second": [second_records[key] for key in sorted(set(second_records) - set(first_records))],
        "changed_event_ids": sorted(set(first["changed_event_ids"]) ^ set(second["changed_event_ids"]), key=cre.parse_id),
    }
    record(workspace, state, {"event": "compare", "case": first["case"]})
    return result


def score_workspace(workspace: Path, state: dict[str, Any], world: kit.ValidatedBundle) -> dict[str, Any]:
    best: dict[str, dict[str, Any]] = {}
    for receipt in retained_receipts(workspace):
        current = best.get(receipt["case"])
        rank = (
            receipt["score"]["total"],
            -receipt["metrics"]["effective_cost"],
            receipt["witness"],
        )
        current_rank = None if current is None else (
            current["score"]["total"],
            -current["metrics"]["effective_cost"],
            current["witness"],
        )
        if current_rank is None or rank > current_rank:
            best[receipt["case"]] = receipt
    rows = [{"case": case, "score": receipt["score"], "metrics": receipt["metrics"]} for case, receipt in sorted(best.items())]
    result = {
        "type": "score",
        "total": sum(item["score"]["total"] for item in best.values()),
        "nominal_solved": sum(item["score"]["nominal_max"] for item in best.values()),
        "nominal_slice": sum(item["points"] for item in descriptors(world).values()),
        "cases": rows,
    }
    record(workspace, state, {"event": "score", "score": {"total": result["total"], "nominal_solved": result["nominal_solved"], "nominal_slice": result["nominal_slice"]}, "solved_cases": len(rows)})
    return result


def replay_retained(workspace: Path, world: kit.ValidatedBundle, compare: dict[str, bytes] | None = None) -> dict[str, Any]:
    witness_paths = sorted((workspace / "witnesses").glob("*.json"), key=lambda path: path.name.encode("utf-8"))
    if any(path.is_symlink() or not path.is_file() for path in witness_paths):
        fail("invalid_workspace", "retained witnesses must be regular files")
    shutil.rmtree(workspace / "receipts")
    (workspace / "receipts").mkdir()
    remaining = list(witness_paths)
    facts: set[str] = set()
    replayed = 0
    while remaining:
        progress = False
        for path in list(remaining):
            try:
                receipt = verifier.verify_witness_bytes(world, path.read_bytes(), str(path), facts)
            except (OSError, verifier.VerificationError) as exc:
                code = exc.code if isinstance(exc, verifier.VerificationError) else "input_error"
                fail("replay_failed", "retained witness cannot be replayed", witness=path.name, code=code)
            if not receipt["valid"]:
                if receipt["diagnostics"][0]["code"] == "case_locked":
                    continue
                fail("replay_failed", "retained witness became invalid", witness=path.name, code=receipt["diagnostics"][0]["code"])
            target = workspace / "receipts" / path.name
            encoded = cre.canonical_bytes(receipt)
            if compare is not None and compare.get(path.name) != encoded:
                fail("replay_drift", "retained receipt did not reproduce byte-for-byte", receipt=path.name)
            canonical_write(target, receipt)
            facts.update(receipt["unlocks"])
            remaining.remove(path)
            replayed += 1
            progress = True
        if not progress:
            fail("replay_failed", "retained witness prerequisites cannot be satisfied", witnesses=[path.name for path in remaining])
    if compare is not None and set(compare) != {path.name for path in receipt_files(workspace)}:
        fail("replay_drift", "retained receipt set changed during replay")
    return {"type": "replayed", "receipts": replayed, "facts": sorted(facts, key=lambda item: item.encode("utf-8"))}


def reset_workspace(workspace: Path, state: dict[str, Any], world: kit.ValidatedBundle, keep_witnesses: bool) -> dict[str, Any]:
    previous = {path.name: path.read_bytes() for path in receipt_files(workspace)} if keep_witnesses else None
    shutil.rmtree(workspace / "branches")
    (workspace / "branches").mkdir()
    if keep_witnesses:
        result = replay_retained(workspace, world, previous)
        result["type"] = "reset"
        result["kept_witnesses"] = True
    else:
        shutil.rmtree(workspace / "receipts")
        shutil.rmtree(workspace / "witnesses")
        (workspace / "receipts").mkdir()
        (workspace / "witnesses").mkdir()
        result = {"type": "reset", "kept_witnesses": False, "receipts": 0, "facts": []}
    record(workspace, state, {"event": "reset", "keep_witnesses": keep_witnesses, "solved_cases": len({item["case"] for item in retained_receipts(workspace)})})
    return result


def replay_workspace(workspace: Path, state: dict[str, Any], world: kit.ValidatedBundle) -> dict[str, Any]:
    previous = {path.name: path.read_bytes() for path in receipt_files(workspace)}
    result = replay_retained(workspace, world, previous)
    record(workspace, state, {"event": "replay", "solved_cases": len({item["case"] for item in retained_receipts(workspace)})})
    return result


def hint(workspace: Path, state: dict[str, Any], world: kit.ValidatedBundle, case_id: str, level: int, locale_code: str = "en") -> dict[str, Any]:
    facts = facts_from_receipts(retained_receipts(workspace))
    require_visible_case(world, case_id, facts)
    if level < 1 or level > 3:
        fail("invalid_hint", "hint level must be 1, 2, or 3")
    pack = locale_pack(locale_code, world)
    hints = pack.case(case_id)["hints"]
    record(workspace, state, {"event": "hint", "case": case_id, "hint_level": level})
    return {"type": "hint", "locale": pack.locale, "case": case_id, "level": level, "hint": hints[level - 1]}


def export_telemetry(workspace: Path, state: dict[str, Any], world: kit.ValidatedBundle, output: Path) -> dict[str, Any]:
    if not state["telemetry"]:
        fail("telemetry_disabled", "telemetry was not enabled for this workspace")
    if output.exists() or output.is_symlink():
        fail("output_exists", "telemetry output already exists")
    events = []
    source = workspace / "telemetry.ndjson"
    if source.exists():
        if source.is_symlink() or not source.is_file():
            fail("invalid_workspace", "workspace telemetry must be a regular file")
        try:
            events = kit.decode_ndjson(source.read_bytes(), str(source), 1024 * 1024)
        except (OSError, kit.KitError) as exc:
            if isinstance(exc, kit.KitError):
                raise PlayerError(exc.code, exc.message, exc.context) from exc
            raise PlayerError("input_error", f"cannot read telemetry: {exc}") from exc
    allowed = SAFE_TELEMETRY_FIELDS | {"format", "tick", "unix_ms"}
    previous_tick = 0
    previous_time = 0
    for event in events:
        if not isinstance(event, dict) or set(event) - allowed or event.get("format") != TELEMETRY_FORMAT:
            fail("invalid_telemetry", "telemetry contains a forbidden field")
        if event["tick"] <= previous_tick or event["unix_ms"] < previous_time:
            fail("invalid_telemetry", "telemetry ordering is not monotonic")
        previous_tick, previous_time = event["tick"], event["unix_ms"]
    export = {"format": TELEMETRY_FORMAT, "bundle": world.bundle, "events": events}
    canonical_write(output, export)
    return {"type": "telemetry-exported", "output": str(output), "events": len(events), "bundle": world.bundle}


def emit_json(value: Any) -> None:
    print(cre.canonical_text(value))


def emit_text(value: Any, locale_code: str = "en") -> None:
    pack = locale_pack(locale_code)
    ui = pack.ui
    kind = value.get("type") if isinstance(value, dict) else None
    if kind == "status":
        solved = ", ".join(value["solved"]) or ui["none"]
        print(ui["solved"].format(solved=len(value["solved"]), total=value["total_cases"], cases=solved))
        print(ui["visible"].format(cases=", ".join(value["visible"])))
    elif kind == "case":
        print(f"{value['id']} — {value['title']} ({ui['points'].format(points=value['points'])})")
        print(value["story"].rstrip())
        print(f"\n{ui['answer_schema']}")
        print(json.dumps(cre.json_value(value["answer_schema"]), ensure_ascii=False, indent=2, sort_keys=True))
    elif kind == "score":
        print(ui["score"].format(total=value["total"], nominal_solved=value["nominal_solved"], nominal_total=value["nominal_slice"]))
        for row in value["cases"]:
            print(ui["case_score"].format(case=row["case"], score=row["score"]["total"], nominal=row["score"]["nominal_max"], metrics=json.dumps(cre.json_value(row["metrics"]), sort_keys=True)))
    elif kind == "hint":
        print(ui["hint"].format(case=value["case"], level=value["level"], hint=value["hint"]))
    elif kind == "canonicalized":
        print(ui["canonicalized"].format(bytes=value["bytes"], sha256=value["sha256"]))
    elif isinstance(value, dict) and value.get("valid") is not None:
        if value["valid"]:
            print(ui["valid"].format(case=value["case"], score=value["score"]["total"], nominal=value["score"]["nominal_max"]))
            print(ui["branch"].format(branch=value["branch"], projection=value["projection"], trace=value["trace"]))
        else:
            diagnostic = value["diagnostics"][0]
            print(ui["invalid"].format(case=value["case"], code=diagnostic["code"], message=diagnostic["message"]))
    else:
        print(json.dumps(cre.json_value(value), ensure_ascii=False, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--json", action="store_true", help="emit canonical JSON instead of human-readable text")
    result.add_argument("--locale", choices=localization.SUPPORTED, default=os.environ.get("AFTERIMAGE_LOCALE", "en"), help="presentation language (or set AFTERIMAGE_LOCALE)")
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="validate and create an isolated player workspace")
    init.add_argument("archive", type=Path)
    init.add_argument("workspace", type=Path)
    init.add_argument("--telemetry", action="store_true", help="record consented, payload-free playtest telemetry")
    for name in ("status", "score", "replay"):
        command = commands.add_parser(name)
        command.add_argument("workspace", type=Path)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("workspace", type=Path)
    inspect.add_argument("target")
    trace = commands.add_parser("trace")
    trace.add_argument("workspace", type=Path)
    trace.add_argument("event")
    direction = trace.add_mutually_exclusive_group()
    direction.add_argument("--parents", action="store_true")
    direction.add_argument("--children", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("workspace", type=Path)
    verify.add_argument("witness", type=Path)
    branch = commands.add_parser("branch")
    branch.add_argument("workspace", type=Path)
    branch.add_argument("case")
    branch.add_argument("--intervention", required=True, type=Path)
    branch.add_argument("--history", help="parent branch ID, history JSON, or player branch snapshot")
    branch.add_argument("--trace-items", action="store_true", help="include the complete ordered branch trace in command output")
    compare = commands.add_parser("compare")
    compare.add_argument("workspace", type=Path)
    compare.add_argument("first")
    compare.add_argument("second")
    reset = commands.add_parser("reset")
    reset.add_argument("workspace", type=Path)
    reset.add_argument("--keep-witnesses", action="store_true")
    hint_command = commands.add_parser("hint")
    hint_command.add_argument("workspace", type=Path)
    hint_command.add_argument("case")
    hint_command.add_argument("level", type=int)
    telemetry = commands.add_parser("telemetry-export")
    telemetry.add_argument("workspace", type=Path)
    telemetry.add_argument("output", type=Path)
    canonicalize = commands.add_parser("canonicalize", help="convert an ordinary JSON draft to canonical CRE JSON")
    canonicalize.add_argument("input", type=Path)
    canonicalize.add_argument("output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code = 0
    try:
        if args.command == "canonicalize":
            result = canonicalize_file(args.input, args.output)
        elif args.command == "init":
            result = init_workspace(args.archive, args.workspace, args.telemetry)
        else:
            state, world = load_workspace(args.workspace)
            if args.command == "status":
                result = status(args.workspace, state, world)
            elif args.command == "inspect":
                result = inspect_target(args.workspace, state, world, args.target, args.locale)
            elif args.command == "trace":
                direction = "parents" if args.parents else "children" if args.children else "trace_items"
                result = trace_event(args.workspace, state, world, args.event, direction)
            elif args.command == "verify":
                result, code = verify_submission(args.workspace, state, world, args.witness)
            elif args.command == "branch":
                result = branch_case(args.workspace, state, world, args.case, args.intervention, args.history, args.trace_items)
            elif args.command == "compare":
                result = compare_branches(args.workspace, state, args.first, args.second)
            elif args.command == "score":
                result = score_workspace(args.workspace, state, world)
            elif args.command == "reset":
                result = reset_workspace(args.workspace, state, world, args.keep_witnesses)
            elif args.command == "replay":
                result = replay_workspace(args.workspace, state, world)
            elif args.command == "hint":
                result = hint(args.workspace, state, world, args.case, args.level, args.locale)
            else:
                result = export_telemetry(args.workspace, state, world, args.output)
        emit_json(result) if args.json else emit_text(result, args.locale)
        return code
    except (PlayerError, verifier.VerificationError, kit.KitError, cre.CREError) as exc:
        if isinstance(exc, PlayerError):
            error = exc.value()
        else:
            error = {"type": "error", "code": exc.code, "message": exc.message, "context": exc.context}
        emit_json(error) if args.json else print(f"ERROR {error['code']}: {error['message']}", file=sys.stderr)
        return 2
    except OSError as exc:
        error = {"type": "error", "code": "io_error", "message": str(exc), "context": {}}
        emit_json(error) if args.json else print(f"ERROR io_error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
