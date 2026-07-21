#!/usr/bin/env python3
"""Safe pack, inspect, extract, and adapt Afterimage 0.1 bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "python"))

import cre  # noqa: E402


FORMAT = "afterimage-bundle/0.1"
WORLD_FORMAT = "afterimage-world/0.1"
REQUIRED_FILES = {
    "manifest.json",
    "program/continuity.cre.json",
    "events/base.ndjson",
    "projections/index.json",
    "cases/index.json",
    "fixtures/conformance/index.json",
}
HARD_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
HARD_MAX_MANIFEST_BYTES = 1024 * 1024
HARD_LIMITS = {
    "max_files": 4096,
    "max_total_uncompressed_bytes": 128 * 1024 * 1024,
    "max_single_file_bytes": 32 * 1024 * 1024,
    "max_line_bytes": 1024 * 1024,
}
DEFAULT_LIMITS = {
    "max_files": 512,
    "max_total_uncompressed_bytes": 64 * 1024 * 1024,
    "max_single_file_bytes": 16 * 1024 * 1024,
    "max_line_bytes": 1024 * 1024,
}
ALLOWED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
NESTED_ARCHIVE_SUFFIXES = {".zip", ".jar", ".apk", ".afterimage"}
ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
RAW_HEX = re.compile(r"^[0-9a-f]{64}$")
DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


class KitError(Exception):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}

    def as_value(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


def fail(code: str, message: str, **context: Any) -> None:
    raise KitError(code, message, context)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_path(raw: Any) -> str:
    if not isinstance(raw, str) or not raw:
        fail("invalid_path", "archive path must be non-empty Text")
    normalized = unicodedata.normalize("NFC", raw)
    if raw != normalized:
        fail("invalid_path", "archive path must already be NFC", path=raw)
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise KitError("invalid_path", "archive path contains invalid Unicode") from exc
    if len(encoded) > 240:
        fail("invalid_path", "archive path exceeds 240 UTF-8 bytes", path=raw)
    if "\0" in raw or "\\" in raw or raw.startswith("/") or DRIVE_PREFIX.match(raw):
        fail("invalid_path", "archive path is absolute or contains a forbidden character", path=raw)
    segments = raw.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        fail("invalid_path", "archive path contains an empty, dot, or dot-dot segment", path=raw)
    return raw


def decode_json(data: bytes, location: str) -> Any:
    if data.startswith(b"\xef\xbb\xbf"):
        fail("invalid_json", "UTF-8 BOM is forbidden", path=location)
    try:
        text = data.decode("utf-8", errors="strict")
        parsed = json.loads(text, object_pairs_hook=cre.object_pairs_no_duplicates)
        value = cre.normalize_value(parsed)
    except cre.CREError as exc:
        raise KitError(exc.code, exc.message, {"path": location, **exc.context}) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KitError("invalid_json", f"invalid canonical JSON: {exc}", {"path": location}) from exc
    if cre.canonical_bytes(value) != data:
        fail("noncanonical_json", "JSON bytes are not canonical CRE JSON", path=location)
    return value


def decode_ndjson(data: bytes, location: str, max_line_bytes: int) -> list[Any]:
    if data and not data.endswith(b"\n"):
        fail("noncanonical_ndjson", "NDJSON must end with LF", path=location)
    if b"\r" in data:
        fail("noncanonical_ndjson", "CR bytes are forbidden in NDJSON", path=location)
    if not data:
        return []
    values = []
    for number, line in enumerate(data[:-1].split(b"\n"), 1):
        if not line:
            fail("noncanonical_ndjson", "blank NDJSON lines are forbidden", path=location, line=number)
        if len(line) > max_line_bytes:
            fail("limit_exceeded", "NDJSON line exceeds max_line_bytes", path=location, line=number)
        values.append(decode_json(line, f"{location}:{number}"))
    return values


def require_map(value: Any, keys: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        fail("invalid_schema", "object has wrong fields", path=location, expected=sorted(keys))
    return value


def validate_limits(value: Any) -> dict[str, int]:
    limits = require_map(value, set(HARD_LIMITS), "manifest.json#/limits")
    result: dict[str, int] = {}
    for key, ceiling in HARD_LIMITS.items():
        item = limits[key]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            fail("invalid_limit", "bundle limit must be a positive Int", limit=key)
        if item > ceiling:
            fail("invalid_limit", "bundle limit exceeds kit hard ceiling", limit=key, ceiling=ceiling)
        result[key] = item
    return result


def validate_manifest(value: Any, actual: dict[str, bytes]) -> tuple[dict[str, Any], str]:
    manifest = require_map(
        value,
        {"format", "semantics", "title", "revision", "limits", "files"},
        "manifest.json",
    )
    if manifest["format"] != FORMAT or manifest["semantics"] != "cre/0.1":
        fail("unsupported_format", "manifest format or semantics is unsupported")
    if not isinstance(manifest["title"], str) or not manifest["title"]:
        fail("invalid_manifest", "manifest title must be non-empty Text")
    if not isinstance(manifest["revision"], str) or not manifest["revision"]:
        fail("invalid_manifest", "manifest revision must be non-empty Text")
    limits = validate_limits(manifest["limits"])
    entries = manifest["files"]
    if not isinstance(entries, list):
        fail("invalid_manifest", "manifest files must be a list")
    if len(entries) > limits["max_files"]:
        fail("limit_exceeded", "manifest file count exceeds max_files")
    declared_paths: list[str] = []
    total = 0
    for index, entry_value in enumerate(entries):
        entry = require_map(entry_value, {"path", "bytes", "sha256"}, f"manifest.json#/files/{index}")
        path = canonical_path(entry["path"])
        if path == "manifest.json":
            fail("invalid_manifest", "manifest must not list itself")
        if path == ".afterimage-kit.json":
            fail("invalid_manifest", "manifest uses the reserved WORLD_DIR metadata path")
        size = entry["bytes"]
        digest = entry["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            fail("invalid_manifest", "declared file size must be a non-negative Int", path=path)
        if size > limits["max_single_file_bytes"]:
            fail("limit_exceeded", "declared file exceeds max_single_file_bytes", path=path)
        if not isinstance(digest, str) or not RAW_HEX.fullmatch(digest):
            fail("invalid_manifest", "file sha256 must be 64 lowercase hex", path=path)
        declared_paths.append(path)
        total += size
    if declared_paths != sorted(declared_paths, key=lambda item: item.encode("utf-8")):
        fail("invalid_manifest", "manifest files must be UTF-8 path sorted")
    if len(set(declared_paths)) != len(declared_paths):
        fail("invalid_manifest", "manifest contains duplicate paths")
    if total > limits["max_total_uncompressed_bytes"]:
        fail("limit_exceeded", "declared files exceed max_total_uncompressed_bytes")

    actual_paths = sorted((path for path in actual if path != "manifest.json"), key=lambda item: item.encode("utf-8"))
    if declared_paths != actual_paths:
        fail(
            "manifest_mismatch",
            "manifest file set differs from logical files",
            missing=sorted(set(actual_paths) - set(declared_paths)),
            extra=sorted(set(declared_paths) - set(actual_paths)),
        )
    for entry in entries:
        data = actual[entry["path"]]
        if len(data) != entry["bytes"]:
            fail("manifest_mismatch", "declared file size differs", path=entry["path"])
        if sha256_hex(data) != entry["sha256"]:
            fail("manifest_mismatch", "declared file digest differs", path=entry["path"])
    digest = cre.digest_id(
        "afterimage/bundle/1",
        cre.canonical_bytes(
            {
                "format": manifest["format"],
                "semantics": manifest["semantics"],
                "title": manifest["title"],
                "revision": manifest["revision"],
                "limits": manifest["limits"],
                "files": manifest["files"],
            }
        ),
    )
    return manifest, digest


def validate_sorted_index(
    value: Any,
    *,
    location: str,
    format_name: str,
    collection: str,
    entry_keys: set[str],
    identity_key: str,
    available: set[str],
) -> list[dict[str, Any]]:
    index = require_map(value, {"format", collection}, location)
    if index["format"] != format_name or not isinstance(index[collection], list):
        fail("invalid_index", "index format or collection is invalid", path=location)
    entries = index[collection]
    identities = []
    referenced_paths = []
    for number, entry_value in enumerate(entries):
        entry = require_map(entry_value, entry_keys, f"{location}#/{collection}/{number}")
        identity = entry[identity_key]
        if not isinstance(identity, str) or not identity:
            fail("invalid_index", "index identity must be non-empty Text", path=location)
        identities.append(identity)
        if "path" in entry:
            path = canonical_path(entry["path"])
            if path not in available:
                fail("invalid_index", "index references an unlisted path", path=path)
            referenced_paths.append(path)
    if identities != sorted(identities, key=lambda item: item.encode("utf-8")) or len(set(identities)) != len(identities):
        fail("invalid_index", "index identities must be unique and UTF-8 sorted", path=location)
    if len(set(referenced_paths)) != len(referenced_paths):
        fail("invalid_index", "index paths must be unique", path=location)
    return entries


@dataclass
class ValidatedBundle:
    source: Path
    archive_sha256: str
    bundle: str
    manifest: dict[str, Any]
    files: dict[str, bytes]
    json_values: dict[str, Any]
    ndjson_values: dict[str, list[Any]]

    def summary(self) -> dict[str, Any]:
        return {
            "format": "afterimage-kit-inspection/0.1",
            "bundle": self.bundle,
            "archive_sha256": self.archive_sha256,
            "title": self.manifest["title"],
            "revision": self.manifest["revision"],
            "files": len(self.manifest["files"]),
            "uncompressed_bytes": sum(entry["bytes"] for entry in self.manifest["files"]),
            "base_events": len(self.ndjson_values["events/base.ndjson"]),
            "projections": len(self.json_values["projections/index.json"]["projections"]),
            "cases": len(self.json_values["cases/index.json"]["cases"]),
            "fixtures": len(self.json_values["fixtures/conformance/index.json"]["cases"]),
        }


@dataclass
class LogicalWorld:
    program: dict[str, Any]
    base_events: list[Any]
    projection_index: dict[str, Any]


def validate_logical_files(source: Path, files: dict[str, bytes], archive_sha256: str) -> ValidatedBundle:
    missing = sorted(REQUIRED_FILES - set(files))
    if missing:
        fail("missing_required_file", "bundle is missing required logical files", missing=missing)
    if len(files["manifest.json"]) > HARD_MAX_MANIFEST_BYTES:
        fail("limit_exceeded", "manifest.json exceeds hard limit")
    manifest_value = decode_json(files["manifest.json"], "manifest.json")
    manifest, bundle_digest = validate_manifest(manifest_value, files)
    limits = manifest["limits"]

    json_values: dict[str, Any] = {"manifest.json": manifest_value}
    ndjson_values: dict[str, list[Any]] = {}
    for path, data in files.items():
        if path == "manifest.json":
            continue
        if Path(path).suffix.lower() in NESTED_ARCHIVE_SUFFIXES or data.startswith(ZIP_MAGICS):
            fail("nested_archive", "nested archives are forbidden", path=path)
        if path.endswith(".json"):
            json_values[path] = decode_json(data, path)
        elif path.endswith(".ndjson"):
            ndjson_values[path] = decode_ndjson(data, path, limits["max_line_bytes"])

    program = require_map(json_values["program/continuity.cre.json"], {"semantics", "strata"}, "program/continuity.cre.json")
    if program["semantics"] != "cre/0.1" or not isinstance(program["strata"], list):
        fail("invalid_program", "world program does not use cre/0.1")

    projections = validate_sorted_index(
        json_values["projections/index.json"],
        location="projections/index.json",
        format_name="afterimage-projections/0.1",
        collection="projections",
        entry_keys={"id", "path"},
        identity_key="id",
        available=set(files),
    )
    for entry in projections:
        projection = require_map(json_values.get(entry["path"]), {"id", "rows"}, entry["path"])
        if projection["id"] != entry["id"] or not isinstance(projection["rows"], list):
            fail("invalid_projection", "projection index ID does not match projection", path=entry["path"])

    cases_index = require_map(json_values["cases/index.json"], {"format", "cases"}, "cases/index.json")
    if cases_index["format"] != "afterimage-cases/0.1" or not isinstance(cases_index["cases"], list):
        fail("invalid_index", "case index is invalid", path="cases/index.json")
    case_ids = []
    for number, descriptor in enumerate(cases_index["cases"]):
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("id"), str) or not descriptor["id"]:
            fail("invalid_index", "case descriptor needs a non-empty ID", index=number)
        case_ids.append(descriptor["id"])
    if case_ids != sorted(case_ids, key=lambda item: item.encode("utf-8")) or len(set(case_ids)) != len(case_ids):
        fail("invalid_index", "case IDs must be unique and UTF-8 sorted")

    validate_sorted_index(
        json_values["fixtures/conformance/index.json"],
        location="fixtures/conformance/index.json",
        format_name="afterimage-conformance-index/0.1",
        collection="cases",
        entry_keys={"name", "path"},
        identity_key="name",
        available=set(files),
    )

    for path, value in json_values.items():
        if not isinstance(value, dict) or value.get("format") != "afterimage-case-world/0.1":
            continue
        descriptor = require_map(value, {"format", "program", "events", "projections"}, path)
        for key in ("program", "events", "projections"):
            referenced = canonical_path(descriptor[key])
            if referenced not in files:
                fail("invalid_world", "case world references an unlisted file", path=path, field=key)
        case_program = require_map(json_values.get(descriptor["program"]), {"semantics", "strata"}, descriptor["program"])
        if case_program["semantics"] != "cre/0.1" or not isinstance(case_program["strata"], list):
            fail("invalid_program", "case world program does not use cre/0.1", path=descriptor["program"])
        case_events = ndjson_values.get(descriptor["events"])
        if case_events is None:
            fail("invalid_world", "case world events path must name NDJSON", path=descriptor["events"])
        try:
            loaded = [cre.load_event(item) for item in case_events]
            cre.validate_base_events(loaded, cre.Counters.create(None))
        except cre.CREError as exc:
            raise KitError(exc.code, exc.message, {"path": descriptor["events"], **exc.context}) from exc
        case_projection_index = json_values.get(descriptor["projections"])
        entries = validate_sorted_index(
            case_projection_index,
            location=descriptor["projections"],
            format_name="afterimage-projections/0.1",
            collection="projections",
            entry_keys={"id", "path"},
            identity_key="id",
            available=set(files),
        )
        for entry in entries:
            projection = require_map(json_values.get(entry["path"]), {"id", "rows"}, entry["path"])
            if projection["id"] != entry["id"] or not isinstance(projection["rows"], list):
                fail("invalid_projection", "case projection index ID does not match projection", path=entry["path"])

    base_values = ndjson_values["events/base.ndjson"]
    try:
        events = [cre.load_event(value) for value in base_values]
        cre.validate_base_events(events, cre.Counters.create(None))
    except cre.CREError as exc:
        raise KitError(exc.code, exc.message, {"path": "events/base.ndjson", **exc.context}) from exc

    return ValidatedBundle(
        source=source,
        archive_sha256=archive_sha256,
        bundle=bundle_digest,
        manifest=manifest,
        files=files,
        json_values=json_values,
        ndjson_values=ndjson_values,
    )


def resolve_logical_world(bundle: ValidatedBundle, world_ref: str = "global") -> LogicalWorld:
    if world_ref == "global":
        return LogicalWorld(
            program=bundle.json_values["program/continuity.cre.json"],
            base_events=bundle.ndjson_values["events/base.ndjson"],
            projection_index=bundle.json_values["projections/index.json"],
        )
    path = canonical_path(world_ref)
    descriptor = bundle.json_values.get(path)
    if not isinstance(descriptor, dict) or set(descriptor) != {"format", "program", "events", "projections"} or descriptor.get("format") != "afterimage-case-world/0.1":
        fail("invalid_world", "case world descriptor is missing or invalid", path=path)
    return LogicalWorld(
        program=bundle.json_values[descriptor["program"]],
        base_events=bundle.ndjson_values[descriptor["events"]],
        projection_index=bundle.json_values[descriptor["projections"]],
    )


def zip_member_kind(info: zipfile.ZipInfo) -> int:
    if info.create_system != 3:
        return 0
    return stat.S_IFMT((info.external_attr >> 16) & 0xFFFF)


def read_member_bounded(archive: zipfile.ZipFile, info: zipfile.ZipInfo, ceiling: int) -> bytes:
    chunks = []
    total = 0
    try:
        with archive.open(info, "r") as stream:
            while True:
                chunk = stream.read(min(1024 * 1024, ceiling + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > ceiling:
                    fail("limit_exceeded", "decompressed member exceeds allowed size", path=info.filename)
    except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise KitError("invalid_zip", f"cannot decompress member: {exc}", {"path": info.filename}) from exc
    if total != info.file_size:
        fail("invalid_zip", "member decompressed size differs from central directory", path=info.filename)
    return b"".join(chunks)


def load_bundle(path_value: Path | str) -> ValidatedBundle:
    path = Path(path_value)
    try:
        metadata = path.stat()
    except OSError as exc:
        raise KitError("input_error", f"cannot stat bundle: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        fail("input_error", "bundle path must be a regular file")
    if metadata.st_size > HARD_MAX_ARCHIVE_BYTES:
        fail("limit_exceeded", "archive exceeds 128 MiB hard limit")
    archive_sha = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                archive_sha.update(chunk)
    except OSError as exc:
        raise KitError("input_error", f"cannot read bundle: {exc}") from exc

    files: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if len(infos) > HARD_LIMITS["max_files"] + 1:
                fail("limit_exceeded", "ZIP member count exceeds hard limit")
            normalized_names: set[str] = set()
            total = 0
            for info in infos:
                name = canonical_path(info.filename)
                if name in normalized_names:
                    fail("duplicate_path", "ZIP contains duplicate normalized paths", path=name)
                normalized_names.add(name)
                if info.is_dir() or name.endswith("/"):
                    fail("unsafe_member", "directory entries are forbidden", path=name)
                if info.flag_bits & 0x1:
                    fail("unsafe_member", "encrypted ZIP members are forbidden", path=name)
                kind = zip_member_kind(info)
                if kind not in {0, stat.S_IFREG}:
                    fail("unsafe_member", "non-regular ZIP member is forbidden", path=name)
                if info.compress_type not in ALLOWED_COMPRESSION:
                    fail("unsafe_member", "ZIP compression method is unsupported", path=name)
                if info.file_size < 0 or info.compress_size < 0 or info.file_size > HARD_LIMITS["max_single_file_bytes"]:
                    fail("limit_exceeded", "ZIP member size exceeds hard limit", path=name)
                total += info.file_size
                if total > HARD_LIMITS["max_total_uncompressed_bytes"] + HARD_MAX_MANIFEST_BYTES:
                    fail("limit_exceeded", "ZIP total size exceeds hard limit")
                ceiling = HARD_MAX_MANIFEST_BYTES if name == "manifest.json" else HARD_LIMITS["max_single_file_bytes"]
                files[name] = read_member_bounded(archive, info, ceiling)
    except KitError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise KitError("invalid_zip", f"cannot read ZIP archive: {exc}") from exc
    return validate_logical_files(path, files, archive_sha.hexdigest())


def collect_source_files(source: Path) -> dict[str, bytes]:
    if not source.is_dir() or source.is_symlink():
        fail("input_error", "pack source must be a real directory")
    files: dict[str, bytes] = {}
    for root, directories, filenames in os.walk(source, followlinks=False):
        root_path = Path(root)
        for directory in list(directories):
            candidate = root_path / directory
            if candidate.is_symlink():
                fail("unsafe_source", "source directory contains a symlink", path=str(candidate.relative_to(source)))
        for filename in filenames:
            candidate = root_path / filename
            relative = candidate.relative_to(source).as_posix()
            if relative in {"manifest.json", ".afterimage-kit.json"}:
                continue
            path = canonical_path(relative)
            mode = candidate.lstat().st_mode
            if not stat.S_ISREG(mode):
                fail("unsafe_source", "source contains a non-regular file", path=path)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(candidate, flags)
                with os.fdopen(descriptor, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        fail("unsafe_source", "source changed to a non-regular file", path=path)
                    data = stream.read(HARD_LIMITS["max_single_file_bytes"] + 1)
            except OSError as exc:
                raise KitError("unsafe_source", f"cannot safely read source file: {exc}", {"path": path}) from exc
            if len(data) > HARD_LIMITS["max_single_file_bytes"]:
                fail("limit_exceeded", "source file exceeds hard limit", path=path)
            files[path] = data
    return files


def build_manifest(files: dict[str, bytes], title: str, revision: str, limits: dict[str, int]) -> dict[str, Any]:
    entries = [
        {"path": path, "bytes": len(files[path]), "sha256": sha256_hex(files[path])}
        for path in sorted(files, key=lambda item: item.encode("utf-8"))
    ]
    return {
        "format": FORMAT,
        "semantics": "cre/0.1",
        "title": title,
        "revision": revision,
        "limits": limits,
        "files": entries,
    }


def zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def pack_bundle(source: Path, output: Path, title: str, revision: str, limits: dict[str, int] | None = None) -> ValidatedBundle:
    if output.exists() or output.is_symlink():
        fail("output_exists", "output bundle already exists", path=str(output))
    if not isinstance(title, str) or not title or not isinstance(revision, str) or not revision:
        fail("input_error", "title and revision must be non-empty")
    chosen_limits = validate_limits(limits or dict(DEFAULT_LIMITS))
    files = collect_source_files(source)
    if len(files) > chosen_limits["max_files"]:
        fail("limit_exceeded", "source file count exceeds max_files")
    if sum(map(len, files.values())) > chosen_limits["max_total_uncompressed_bytes"]:
        fail("limit_exceeded", "source size exceeds max_total_uncompressed_bytes")
    for path, data in files.items():
        if len(data) > chosen_limits["max_single_file_bytes"]:
            fail("limit_exceeded", "source file exceeds max_single_file_bytes", path=path)
    manifest = build_manifest(files, title, revision, chosen_limits)
    all_files = {"manifest.json": cre.canonical_bytes(manifest), **files}
    validate_logical_files(source, all_files, "0" * 64)

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True) as archive:
            for path in sorted(all_files, key=lambda item: item.encode("utf-8")):
                archive.writestr(zip_info(path), all_files[path], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise KitError("output_exists", "output bundle appeared during pack", {"path": str(output)}) from exc
        temporary.unlink()
        try:
            return load_bundle(output)
        except Exception:
            output.unlink(missing_ok=True)
            raise
    finally:
        temporary.unlink(missing_ok=True)


def write_file_secure(root: Path, relative: str, data: bytes) -> None:
    destination = root.joinpath(*relative.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with destination.open("xb") as stream:
        os.chmod(destination, 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def extract_bundle(bundle: ValidatedBundle, destination: Path) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        fail("output_exists", "WORLD_DIR destination already exists", path=str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    os.chmod(staging, 0o700)
    metadata = {
        "format": WORLD_FORMAT,
        "bundle": bundle.bundle,
        "archive_sha256": bundle.archive_sha256,
        "files": bundle.manifest["files"],
    }
    try:
        for path in sorted(bundle.files, key=lambda item: item.encode("utf-8")):
            write_file_secure(staging, path, bundle.files[path])
        write_file_secure(staging, ".afterimage-kit.json", cre.canonical_bytes(metadata))
        if destination.exists() or destination.is_symlink():
            fail("output_exists", "WORLD_DIR destination appeared during extraction", path=str(destination))
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata


def walk_world_files(root: Path) -> dict[str, bytes]:
    if not root.is_dir() or root.is_symlink():
        fail("invalid_world", "WORLD_DIR must be a real directory")
    files: dict[str, bytes] = {}
    total = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            candidate = current_path / directory
            if candidate.is_symlink():
                fail("invalid_world", "WORLD_DIR contains a symlink", path=str(candidate.relative_to(root)))
        for filename in filenames:
            candidate = current_path / filename
            relative = canonical_path(candidate.relative_to(root).as_posix())
            mode = candidate.lstat().st_mode
            if not stat.S_ISREG(mode):
                fail("invalid_world", "WORLD_DIR contains a non-regular file", path=relative)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            ceiling = HARD_MAX_MANIFEST_BYTES if relative in {"manifest.json", ".afterimage-kit.json"} else HARD_LIMITS["max_single_file_bytes"]
            try:
                descriptor = os.open(candidate, flags)
                with os.fdopen(descriptor, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        fail("invalid_world", "WORLD_DIR file changed type while reading", path=relative)
                    data = stream.read(ceiling + 1)
            except OSError as exc:
                raise KitError("invalid_world", f"cannot safely read WORLD_DIR file: {exc}", {"path": relative}) from exc
            if len(data) > ceiling:
                fail("limit_exceeded", "WORLD_DIR file exceeds hard limit", path=relative)
            files[relative] = data
            total += len(data)
            if len(files) > HARD_LIMITS["max_files"] + 2:
                fail("limit_exceeded", "WORLD_DIR file count exceeds hard limit")
            if total > HARD_LIMITS["max_total_uncompressed_bytes"] + 2 * HARD_MAX_MANIFEST_BYTES:
                fail("limit_exceeded", "WORLD_DIR total bytes exceed hard limit")
    return files


def verify_world(root: Path) -> ValidatedBundle:
    files = walk_world_files(root)
    if ".afterimage-kit.json" not in files:
        fail("invalid_world", "WORLD_DIR lacks .afterimage-kit.json")
    metadata = require_map(
        decode_json(files.pop(".afterimage-kit.json"), ".afterimage-kit.json"),
        {"format", "bundle", "archive_sha256", "files"},
        ".afterimage-kit.json",
    )
    if metadata["format"] != WORLD_FORMAT:
        fail("invalid_world", "WORLD_DIR metadata format is unsupported")
    if not isinstance(metadata["archive_sha256"], str) or not RAW_HEX.fullmatch(metadata["archive_sha256"]):
        fail("invalid_world", "WORLD_DIR archive hash is invalid")
    bundle = validate_logical_files(root, files, metadata["archive_sha256"])
    if bundle.bundle != metadata["bundle"] or bundle.manifest["files"] != metadata["files"]:
        fail("invalid_world", "WORLD_DIR metadata does not match logical files")
    expected = {"manifest.json", *(entry["path"] for entry in metadata["files"])}
    if set(files) != expected:
        fail("invalid_world", "WORLD_DIR contains unexpected or missing files")
    return bundle


def materialize_case(
    world: ValidatedBundle,
    projection_id: str,
    branch_value: Any | None = None,
    world_ref: str = "global",
) -> dict[str, Any]:
    logical = resolve_logical_world(world, world_ref)
    projections = logical.projection_index["projections"]
    entry = next((item for item in projections if item["id"] == projection_id), None)
    if entry is None:
        fail("unknown_projection", "projection ID is absent from world", projection=projection_id)
    result = {
        "name": f"world:{projection_id}",
        "bundle_digest": world.bundle,
        "program": logical.program,
        "base_events": logical.base_events,
        "projection": world.json_values[entry["path"]],
    }
    if branch_value is not None:
        branch = require_map(branch_value, {"format", "parent_branch", "operations"}, "branch")
        if branch["format"] != "afterimage-branch/0.1" or not isinstance(branch["operations"], list):
            fail("invalid_branch", "branch adapter input is invalid")
        result["parent_branch"] = branch["parent_branch"]
        result["operations"] = branch["operations"]
    return result


def emit(value: Any, pretty: bool = False) -> None:
    if pretty:
        print(json.dumps(cre.json_value(cre.normalize_value(value)), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(cre.canonical_text(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate and summarize a bundle")
    inspect_parser.add_argument("bundle", type=Path)
    inspect_parser.add_argument("--pretty", action="store_true")

    extract_parser = subparsers.add_parser("extract", help="validate and atomically create WORLD_DIR")
    extract_parser.add_argument("bundle", type=Path)
    extract_parser.add_argument("world", type=Path)
    extract_parser.add_argument("--pretty", action="store_true")

    verify_parser = subparsers.add_parser("verify-world", help="revalidate an extracted WORLD_DIR")
    verify_parser.add_argument("world", type=Path)
    verify_parser.add_argument("--pretty", action="store_true")

    pack_parser = subparsers.add_parser("pack", help="build a reproducible bundle from logical files")
    pack_parser.add_argument("source", type=Path)
    pack_parser.add_argument("output", type=Path)
    pack_parser.add_argument("--title", required=True)
    pack_parser.add_argument("--revision", required=True)
    pack_parser.add_argument("--pretty", action="store_true")

    case_parser = subparsers.add_parser("case", help="materialize a CRE case from WORLD_DIR")
    case_parser.add_argument("world", type=Path)
    case_parser.add_argument("--projection", required=True)
    case_parser.add_argument("--case")
    case_parser.add_argument("--branch", type=Path)
    case_parser.add_argument("--output", type=Path)
    case_parser.add_argument("--pretty", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = load_bundle(args.bundle).summary()
        elif args.command == "extract":
            bundle = load_bundle(args.bundle)
            metadata = extract_bundle(bundle, args.world)
            result = {"type": "extracted", "world": str(args.world), **metadata}
        elif args.command == "verify-world":
            result = verify_world(args.world).summary()
        elif args.command == "pack":
            bundle = pack_bundle(args.source, args.output, args.title, args.revision)
            result = {"type": "packed", "path": str(args.output), **bundle.summary()}
        else:
            world = verify_world(args.world)
            branch = decode_json(args.branch.read_bytes(), str(args.branch)) if args.branch else None
            world_ref = "global"
            if args.case:
                descriptor = next(
                    (item for item in world.json_values["cases/index.json"]["cases"] if item.get("id") == args.case),
                    None,
                )
                if descriptor is None:
                    fail("unknown_case", "case ID is absent from world", case=args.case)
                world_ref = descriptor.get("world", "global")
            result = materialize_case(world, args.projection, branch, world_ref)
            encoded = cre.canonical_bytes(result)
            if args.output:
                if args.output.exists() or args.output.is_symlink():
                    fail("output_exists", "case output already exists", path=str(args.output))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("xb") as stream:
                    stream.write(encoded)
                return 0
        emit(result, args.pretty)
        return 0
    except KitError as exc:
        emit({"type": "error", **exc.as_value()})
        return 3
    except cre.CREError as exc:
        emit({"type": "error", **exc.as_value()})
        return 3
    except OSError as exc:
        emit({"type": "error", "code": "io_error", "message": str(exc), "context": {}})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
