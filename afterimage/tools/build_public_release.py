#!/usr/bin/env python3
"""Build reproducible public Afterimage player, engine, media, and web kits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
ARCHIVE = ROOT / "dist" / "afterimage-production-2.1-dev.afterimage"
EXPECTED_ARCHIVE_SHA256 = (
    "4d2015a522281bddeaa3ec9fedda28715677663926bea924a05494ee78ca57af"
)
EXPECTED_BUNDLE = (
    "sha256:517038cdd97cb7d3687f53272e8964a11ffcc1cca82cc69a73668bf56aea0514"
)
VERSION = "2.1"

sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import cre  # noqa: E402


PLAYER_FILES = {
    "README.md": PUBLIC / "release" / "PLAYER_QUICKSTART.md",
    "RELEASE_NOTES.md": PUBLIC / "release" / "RELEASE_NOTES.md",
    "afterimage-2.1.afterimage": ARCHIVE,
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
    **{
        f"locales/{path.name}": path
        for path in sorted((ROOT / "locales").glob("*.json"))
    },
    **{
        f"spec/{path.name}": path
        for path in sorted((ROOT / "spec").glob("*.md"))
    },
}

ENGINE_FILES = {
    "README.md": PUBLIC / "release" / "ENGINE_QUICKSTART.md",
    "spec/causal_reduction_engine.md": ROOT
    / "spec"
    / "causal_reduction_engine.md",
    "spec/conformance_fixture.md": ROOT / "spec" / "conformance_fixture.md",
    "spec/bundle_and_witness.md": ROOT / "spec" / "bundle_and_witness.md",
    "spec/scoring.md": ROOT / "spec" / "scoring.md",
    "conformance/suite.json": ROOT / "tests" / "conformance" / "suite.json",
    "conformance/matrix.py": ROOT / "tests" / "conformance" / "matrix.py",
    "conformance/full-suite.json": ROOT
    / "tests"
    / "conformance"
    / "full-suite.json",
    "conformance/golden.json": ROOT / "tests" / "conformance" / "golden.json",
    "conformance/check.py": ROOT / "tests" / "conformance" / "check.py",
    "conformance/expected.sha256": ROOT
    / "tests"
    / "conformance"
    / "expected.sha256",
}

FORBIDDEN_PARTS = {
    "author-baselines",
    "authoring.py",
    "build_slice.py",
    "content",
}


class PublicReleaseError(Exception):
    """A stable public-release build or validation failure."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cre.canonical_bytes(value))


def copy_files(destination: Path, mapping: dict[str, Path]) -> None:
    for relative, source in sorted(
        mapping.items(), key=lambda item: item[0].encode("utf-8")
    ):
        if not source.is_file() or source.is_symlink():
            raise PublicReleaseError(f"missing or unsafe source: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def copy_public_tree(destination: Path, include_source: bool = True) -> None:
    for name in ("assets", "brand", "copy", "press", "release", "site"):
        source = PUBLIC / name
        if not source.is_dir() or source.is_symlink():
            raise PublicReleaseError(f"missing public source directory: {source}")
        ignored = None
        if name == "assets" and not include_source:
            ignored = shutil.ignore_patterns("source")
        shutil.copytree(source, destination / name, ignore=ignored)
    shutil.copyfile(PUBLIC / "README.md", destination / "README.md")


def package_manifest(root: Path, kind: str, version: str) -> dict[str, Any]:
    files = []
    for path in sorted(
        (
            item
            for item in root.rglob("*")
            if item.is_file() and item.name != "package.json"
        ),
        key=lambda item: item.relative_to(root).as_posix().encode("utf-8"),
    ):
        if path.is_symlink():
            raise PublicReleaseError(f"package source is a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        validate_public_path(kind, relative)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "format": "afterimage-public-package/0.1",
        "kind": kind,
        "version": version,
        "bundle": EXPECTED_BUNDLE if kind == "player" else None,
        "files": files,
    }


def validate_public_path(kind: str, relative: str) -> None:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PublicReleaseError(f"unsafe package path: {relative}")
    for part in path.parts:
        lowered = part.lower()
        if lowered in FORBIDDEN_PARTS:
            raise PublicReleaseError(f"forbidden public path: {relative}")
        if kind != "engine" and lowered == "golden.json":
            raise PublicReleaseError(f"forbidden public golden: {relative}")
        if kind == "player" and (
            lowered.endswith(".witness.json") or lowered.endswith(".receipt.json")
        ):
            raise PublicReleaseError(f"forbidden solution artifact: {relative}")


def zip_reproducible(source: Path, output: Path) -> None:
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(
            (item for item in source.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(source).as_posix().encode("utf-8"),
        ):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def inspect_archive() -> kit.Bundle:
    if sha256(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise PublicReleaseError("production archive SHA-256 changed")
    try:
        bundle = kit.load_bundle(ARCHIVE)
    except kit.KitError as exc:
        raise PublicReleaseError(
            f"production archive rejected: {exc.code}: {exc.message}"
        ) from exc
    if bundle.bundle != EXPECTED_BUNDLE:
        raise PublicReleaseError("production BundleId changed")
    if bundle.summary()["cases"] != 75:
        raise PublicReleaseError("production archive does not contain 75 cases")
    return bundle


def build(output: Path) -> dict[str, Any]:
    bundle = inspect_archive()
    if output.exists() or output.is_symlink():
        raise PublicReleaseError("output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        stages = temporary / ".stages"
        player = stages / "player"
        engine = stages / "engine"
        media = stages / "media"

        copy_files(player, PLAYER_FILES)
        copy_files(engine, ENGINE_FILES)
        copy_public_tree(media, include_source=True)
        for root, kind, version in (
            (player, "player", VERSION),
            (engine, "engine", "0.1"),
            (media, "media", VERSION),
        ):
            canonical_json(root / "package.json", package_manifest(root, kind, version))

        package_specs = (
            (player, f"afterimage-player-kit-{VERSION}.zip", "player"),
            (engine, "afterimage-engine-kit-0.1.zip", "engine"),
            (media, f"afterimage-media-kit-{VERSION}.zip", "media"),
        )
        packages = []
        for root, name, kind in package_specs:
            target = temporary / name
            zip_reproducible(root, target)
            packages.append(
                {
                    "kind": kind,
                    "path": name,
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                }
            )

        web = temporary / "web"
        copy_public_tree(web, include_source=False)
        downloads = web / "downloads"
        downloads.mkdir()
        for package in packages:
            shutil.copyfile(temporary / package["path"], downloads / package["path"])

        release = {
            "format": "afterimage-public-release/0.1",
            "version": VERSION,
            "bundle": bundle.bundle,
            "archive_sha256": bundle.archive_sha256,
            "cases": bundle.summary()["cases"],
            "points": 10000,
            "locales": ["en", "ja", "zh-Hans", "de"],
            "packages": packages,
            "web_root": "web/site/index.html",
        }
        canonical_json(temporary / "release.json", release)
        canonical_json(
            temporary / "checksums.json",
            {
                "format": "afterimage-public-checksums/0.1",
                "archive_sha256": bundle.archive_sha256,
                "bundle": bundle.bundle,
                "files": [
                    {
                        "path": package["path"],
                        "bytes": package["bytes"],
                        "sha256": package["sha256"],
                    }
                    for package in packages
                ],
            },
        )
        shutil.rmtree(stages)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify(output, smoke=True)
    return release


def validate_zip(path: Path, expected_kind: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != sorted(names, key=lambda name: name.encode("utf-8")):
            raise PublicReleaseError(f"noncanonical ZIP order: {path.name}")
        if "package.json" not in names:
            raise PublicReleaseError(f"package manifest missing: {path.name}")
        for info in infos:
            validate_public_path(expected_kind, info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise PublicReleaseError(f"symlink in package: {info.filename}")
        manifest = json.loads(archive.read("package.json"))
        if manifest.get("format") != "afterimage-public-package/0.1":
            raise PublicReleaseError(f"wrong package format: {path.name}")
        if manifest.get("kind") != expected_kind:
            raise PublicReleaseError(f"wrong package kind: {path.name}")
        expected = []
        for info in infos:
            if info.filename == "package.json":
                continue
            payload = archive.read(info.filename)
            expected.append(
                {
                    "path": info.filename,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        if manifest.get("files") != expected:
            raise PublicReleaseError(f"package manifest mismatch: {path.name}")
        return manifest


def smoke_player(path: Path) -> None:
    temporary = Path(tempfile.mkdtemp(prefix="afterimage-public-smoke-"))
    try:
        with zipfile.ZipFile(path) as archive:
            archive.extractall(temporary)
        environment = dict(os.environ)
        environment["PATH"] = ""
        init = subprocess.run(
            [
                sys.executable,
                "tools/player.py",
                "--json",
                "init",
                "afterimage-2.1.afterimage",
                "desk",
            ],
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if init.returncode != 0:
            raise PublicReleaseError(
                f"public player init failed: {init.stderr or init.stdout}"
            )
        inspect = subprocess.run(
            [
                sys.executable,
                "tools/player.py",
                "--locale",
                "ja",
                "inspect",
                "desk",
                "ORIENT.001",
            ],
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if inspect.returncode != 0 or not any(
            "\u3040" <= char <= "\u9fff" for char in inspect.stdout
        ):
            raise PublicReleaseError(
                f"public player localized inspect failed: {inspect.stderr or inspect.stdout}"
            )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def verify(output: Path, smoke: bool = False) -> dict[str, Any]:
    inspect_archive()
    if not output.is_dir() or output.is_symlink():
        raise PublicReleaseError("public release output is missing or unsafe")
    release = json.loads((output / "release.json").read_text(encoding="utf-8"))
    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    if release.get("format") != "afterimage-public-release/0.1":
        raise PublicReleaseError("wrong public release format")
    if release.get("bundle") != EXPECTED_BUNDLE:
        raise PublicReleaseError("wrong release BundleId")
    if release.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
        raise PublicReleaseError("wrong release archive SHA-256")
    expected_files = checksums.get("files")
    if not isinstance(expected_files, list) or len(expected_files) != 3:
        raise PublicReleaseError("checksums must describe three packages")
    kinds = {
        f"afterimage-player-kit-{VERSION}.zip": "player",
        "afterimage-engine-kit-0.1.zip": "engine",
        f"afterimage-media-kit-{VERSION}.zip": "media",
    }
    for item in expected_files:
        path = output / item["path"]
        if item["path"] not in kinds or not path.is_file() or path.is_symlink():
            raise PublicReleaseError(f"unexpected or missing package: {item['path']}")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise PublicReleaseError(f"package checksum mismatch: {item['path']}")
        validate_zip(path, kinds[item["path"]])
    web_root = output / release.get("web_root", "")
    if not web_root.is_file() or web_root.is_symlink():
        raise PublicReleaseError("deployable web root is missing")
    for package_name in kinds:
        copied = output / "web" / "downloads" / package_name
        if not copied.is_file() or sha256(copied) != sha256(output / package_name):
            raise PublicReleaseError(f"web download drift: {package_name}")
    if smoke:
        smoke_player(output / f"afterimage-player-kit-{VERSION}.zip")
    return {
        "type": "public-release-verified",
        "output": str(output),
        "bundle": EXPECTED_BUNDLE,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "packages": len(kinds),
        "smoke": smoke,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=ROOT / "dist" / "public-release",
    )
    parser.add_argument(
        "--check", action="store_true", help="validate an existing public release"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="run the extracted player smoke test"
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify(args.output, smoke=args.smoke) if args.check else build(args.output)
        print(
            json.dumps(result, indent=2, sort_keys=True)
            if args.pretty
            else cre.canonical_text(result)
        )
        return 0
    except (PublicReleaseError, OSError, ValueError, zipfile.BadZipFile) as exc:
        error = {
            "type": "error",
            "code": "public_release_failed",
            "message": str(exc),
            "context": {},
        }
        print(
            json.dumps(error, indent=2, sort_keys=True)
            if args.pretty
            else cre.canonical_text(error)
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
