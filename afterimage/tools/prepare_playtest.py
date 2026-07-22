#!/usr/bin/env python3
"""Build separated, reproducible blind-playtest kits without author secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import cre  # noqa: E402


ENGINE_FILES = {
    "ENGINE_TASK.md": ROOT / "playtest" / "ENGINE_TASK.md",
    "spec/causal_reduction_engine.md": ROOT / "spec" / "causal_reduction_engine.md",
    "spec/conformance_fixture.md": ROOT / "spec" / "conformance_fixture.md",
    "spec/bundle_and_witness.md": ROOT / "spec" / "bundle_and_witness.md",
    "spec/scoring.md": ROOT / "spec" / "scoring.md",
    "conformance/suite.json": ROOT / "tests" / "conformance" / "suite.json",
    "conformance/matrix.py": ROOT / "tests" / "conformance" / "matrix.py",
    "conformance/full-suite.json": ROOT / "tests" / "conformance" / "full-suite.json",
    "conformance/golden.json": ROOT / "tests" / "conformance" / "golden.json",
    "conformance/check.py": ROOT / "tests" / "conformance" / "check.py",
    "conformance/expected.sha256": ROOT / "tests" / "conformance" / "expected.sha256",
}

GAME_FILES = {
    "QUICKSTART.md": ROOT / "playtest" / "PARTICIPANT_QUICKSTART.md",
    "tools/player.py": ROOT / "tools" / "player.py",
    "tools/localization.py": ROOT / "tools" / "localization.py",
    "tools/afterimage_kit.py": ROOT / "tools" / "afterimage_kit.py",
    "tools/verify_witness.py": ROOT / "tools" / "verify_witness.py",
    "tools/pulse.py": ROOT / "tools" / "pulse.py",
    "tools/mosaic.py": ROOT / "tools" / "mosaic.py",
    "tools/lens.py": ROOT / "tools" / "lens.py",
    "tools/covenant.py": ROOT / "tools" / "covenant.py",
    "tools/paradox.py": ROOT / "tools" / "paradox.py",
    "reference/python/cre.py": ROOT / "reference" / "python" / "cre.py",
    **{f"locales/{path.name}": path for path in sorted((ROOT / "locales").glob("*.json"))},
    **{f"spec/{path.name}": path for path in sorted((ROOT / "spec").glob("*.md"))},
}

OPERATOR_FILES = {
    "OPERATOR_PROTOCOL.md": ROOT / "playtest" / "OPERATOR_PROTOCOL.md",
    "INTERVIEW.md": ROOT / "playtest" / "INTERVIEW.md",
    "OBSERVATION_SCHEMA.md": ROOT / "playtest" / "OBSERVATION_SCHEMA.md",
    "AI_PROXY_PROTOCOL.md": ROOT / "playtest" / "AI_PROXY_PROTOCOL.md",
    "vertical_slice.md": ROOT / "vertical_slice.md",
    "production_status.md": ROOT / "production_status.md",
    "tools/analyze_playtest.py": ROOT / "tools" / "analyze_playtest.py",
    "tools/analyze_ai_proxy.py": ROOT / "tools" / "analyze_ai_proxy.py",
    "tools/check_engine_generalization.py": ROOT / "tools" / "check_engine_generalization.py",
    "tests/conformance/full-suite.json": ROOT / "tests" / "conformance" / "full-suite.json",
    "tools/afterimage_kit.py": ROOT / "tools" / "afterimage_kit.py",
    "reference/python/cre.py": ROOT / "reference" / "python" / "cre.py",
}

FORBIDDEN_PARTICIPANT_NAMES = {
    "author-baselines",
    "authoring.py",
    "build_slice.py",
    "golden.json",
    "content",
}


class PrepareError(Exception):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cre.canonical_bytes(value))


def copy_files(destination: Path, mapping: dict[str, Path]) -> None:
    for relative, source in sorted(mapping.items(), key=lambda item: item[0].encode("utf-8")):
        if not source.is_file() or source.is_symlink():
            raise PrepareError(f"required source is missing or unsafe: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def package_manifest(root: Path, kind: str) -> dict[str, Any]:
    files = []
    for path in sorted((item for item in root.rglob("*") if item.is_file() and item.name != "package.json"), key=lambda item: item.relative_to(root).as_posix().encode("utf-8")):
        relative = path.relative_to(root).as_posix()
        public_engine_oracle = kind == "engine" and relative == "conformance/golden.json"
        if path.is_symlink() or (
            any(part in FORBIDDEN_PARTICIPANT_NAMES for part in Path(relative).parts)
            and not public_engine_oracle
        ):
            raise PrepareError(f"forbidden participant file: {relative}")
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"format": "afterimage-playtest-package/0.1", "kind": kind, "files": files}


def zip_reproducible(source: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted((item for item in source.rglob("*") if item.is_file()), key=lambda item: item.relative_to(source).as_posix().encode("utf-8")):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as stream:
                archive.writestr(info, stream.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def prepare(archive: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise PrepareError("output already exists")
    try:
        bundle = kit.load_bundle(archive)
    except kit.KitError as exc:
        raise PrepareError(f"bundle validation failed: {exc.code}: {exc.message}") from exc
    golden = json.loads((ROOT / "content" / "vertical_slice" / "golden.json").read_text())
    if bundle.archive_sha256 != golden["archive_sha256"] or bundle.bundle != golden["bundle"]:
        raise PrepareError("archive does not match the frozen vertical-slice golden")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        stages = temporary / "stages"
        engine = stages / "engine-session"
        game = stages / "game-session"
        operator = stages / "operator-session"
        copy_files(engine, ENGINE_FILES)
        copy_files(game, GAME_FILES)
        shutil.copyfile(archive, game / "afterimage-slice.afterimage")
        copy_files(operator, OPERATOR_FILES)
        for root, kind in ((engine, "engine"), (game, "game"), (operator, "operator")):
            write_json(root / "package.json", package_manifest(root, kind))

        packages = []
        for root in (engine, game, operator):
            name = f"{root.name}.zip"
            target = temporary / name
            zip_reproducible(root, target)
            packages.append({"path": name, "bytes": target.stat().st_size, "sha256": sha256(target)})
        write_json(
            temporary / "checksums.json",
            {
                "format": "afterimage-playtest-release/0.1",
                "bundle": bundle.bundle,
                "archive_sha256": bundle.archive_sha256,
                "packages": packages,
            },
        )
        shutil.rmtree(stages)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "type": "playtest-prepared",
        "output": str(output),
        "bundle": bundle.bundle,
        "archive_sha256": bundle.archive_sha256,
        "packages": packages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path, default=ROOT / "dist" / "afterimage-slice-dev.afterimage")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.archive, args.output)
        if args.pretty:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(cre.canonical_text(result))
        return 0
    except (PrepareError, OSError) as exc:
        error = {"type": "error", "code": "prepare_failed", "message": str(exc), "context": {}}
        print(json.dumps(error, indent=2, sort_keys=True) if args.pretty else cre.canonical_text(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
