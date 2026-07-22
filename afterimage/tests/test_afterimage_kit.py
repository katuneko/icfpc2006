#!/usr/bin/env python3
"""Adversarial and integration tests for afterimage-kit 0.1."""

from __future__ import annotations

import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import cre  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cre.canonical_bytes(value))


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def base_event() -> dict[str, object]:
    return cre.make_event(
        {
            "topic": "sample.input",
            "at": 4,
            "payload": {"value": 7},
            "parents": [],
            "origin": {"kind": "base", "source": "sample", "sequence": 0},
        }
    )


def sample_program() -> dict[str, object]:
    return {
        "semantics": "cre/0.1",
        "strata": [
            {
                "index": 0,
                "rules": [
                    {
                        "id": "sample.echo",
                        "positive": [
                            {"alias": "s", "topic": "sample.input", "where": ["const", True]}
                        ],
                        "negative": [],
                        "aggregate": [],
                        "distinct": [],
                        "guard": ["const", True],
                        "emit": [
                            {
                                "topic": ["const", "sample.output"],
                                "at": ["get", "s", "/at"],
                                "payload": ["get", "s", "/payload"],
                                "parents": [],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def sample_projection() -> dict[str, object]:
    return {
        "id": "sample.public",
        "rows": [
            {
                "positive": [
                    {"alias": "x", "topic": "sample.output", "where": ["const", True]}
                ],
                "negative": [],
                "aggregate": [],
                "distinct": [],
                "guard": ["const", True],
                "value": ["get", "x", "/payload/value"],
                "sort": [],
            }
        ],
    }


def make_source(root: Path) -> Path:
    source = root / "logical"
    event = base_event()
    write_json(source / "program/continuity.cre.json", sample_program())
    write_bytes(source / "events/base.ndjson", cre.canonical_bytes(cre.event_view(event)) + b"\n")
    write_json(
        source / "projections/index.json",
        {
            "format": "afterimage-projections/0.1",
            "projections": [{"id": "sample.public", "path": "projections/sample.public.cre.json"}],
        },
    )
    write_json(source / "projections/sample.public.cre.json", sample_projection())
    write_json(source / "cases/index.json", {"format": "afterimage-cases/0.1", "cases": []})
    write_json(
        source / "fixtures/conformance/index.json",
        {"format": "afterimage-conformance-index/0.1", "cases": []},
    )
    return source


def copy_zip(
    source: Path,
    destination: Path,
    mutate: Callable[[str, bytes], bytes] | None = None,
    extra: list[tuple[zipfile.ZipInfo | str, bytes]] | None = None,
) -> None:
    with zipfile.ZipFile(source, "r") as archive:
        entries = [(info, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(destination, "w") as archive:
        for info, data in entries:
            if mutate is not None:
                data = mutate(info.filename, data)
            clone = kit.zip_info(info.filename)
            archive.writestr(clone, data)
        for name, data in extra or []:
            archive.writestr(name, data)


class AfterimageKitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="afterimage-kit-test-")
        self.root = Path(self.temporary.name)
        self.source = make_source(self.root)
        self.bundle_path = self.root / "sample.afterimage"
        self.bundle = kit.pack_bundle(self.source, self.bundle_path, "Sample City", "test-1")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_kit_error(self, code: str, callback, *args, **kwargs) -> kit.KitError:
        with self.assertRaises(kit.KitError) as captured:
            callback(*args, **kwargs)
        self.assertEqual(captured.exception.code, code)
        return captured.exception

    def test_reproducible_pack_extract_verify_and_dual_engine_protocol(self) -> None:
        second = self.root / "second.afterimage"
        second_bundle = kit.pack_bundle(self.source, second, "Sample City", "test-1")
        self.assertEqual(self.bundle_path.read_bytes(), second.read_bytes())
        self.assertEqual(self.bundle.bundle, second_bundle.bundle)

        world_path = self.root / "world"
        metadata = kit.extract_bundle(self.bundle, world_path)
        verified = kit.verify_world(world_path)
        self.assertEqual(metadata["bundle"], verified.bundle)
        self.assertEqual(verified.summary()["base_events"], 1)

        case = kit.materialize_case(verified, "sample.public")
        result = cre.run_case(case)
        self.assertEqual(result["projection"], [7])
        case_path = self.root / "case.json"
        case_path.write_bytes(cre.canonical_bytes(case))
        commands = [
            [sys.executable, str(ROOT / "reference/python/cre.py"), "protocol", str(case_path)],
            ["node", str(ROOT / "reference/javascript/cre.mjs"), "protocol", str(case_path)],
        ]
        outputs = [subprocess.run(command, check=True, capture_output=True).stdout for command in commands]
        self.assertEqual(outputs[0], outputs[1])
        records = [json.loads(line) for line in outputs[0].splitlines()]
        self.assertEqual([record["type"] for record in records], ["ready", "projection", "done"])
        self.assertEqual(records[1]["value"], 7)

    def test_cli_pack_inspect_extract_verify_and_case(self) -> None:
        tool = [sys.executable, str(ROOT / "tools/afterimage_kit.py")]
        packed = self.root / "cli.afterimage"
        pack = subprocess.run(
            [*tool, "pack", str(self.source), str(packed), "--title", "CLI City", "--revision", "cli-1"],
            check=True,
            capture_output=True,
        )
        self.assertEqual(json.loads(pack.stdout)["type"], "packed")
        inspection = subprocess.run([*tool, "inspect", str(packed)], check=True, capture_output=True)
        self.assertEqual(json.loads(inspection.stdout)["base_events"], 1)

        world = self.root / "cli-world"
        extracted = subprocess.run([*tool, "extract", str(packed), str(world)], check=True, capture_output=True)
        self.assertEqual(json.loads(extracted.stdout)["type"], "extracted")
        verified = subprocess.run([*tool, "verify-world", str(world)], check=True, capture_output=True)
        self.assertEqual(json.loads(verified.stdout)["projections"], 1)

        case_path = self.root / "cli-case.json"
        case_run = subprocess.run(
            [*tool, "case", str(world), "--projection", "sample.public", "--output", str(case_path)],
            check=True,
            capture_output=True,
        )
        self.assertEqual(case_run.stdout, b"")
        self.assertEqual(cre.run_case(cre.load_json(case_path))["projection"], [7])

    def test_extract_refuses_existing_destination(self) -> None:
        destination = self.root / "existing"
        destination.mkdir()
        self.assert_kit_error("output_exists", kit.extract_bundle, self.bundle, destination)

    def test_world_modification_and_unexpected_file_are_detected(self) -> None:
        world = self.root / "world"
        kit.extract_bundle(self.bundle, world)
        program = world / "program/continuity.cre.json"
        program.write_bytes(program.read_bytes() + b" ")
        self.assert_kit_error("manifest_mismatch", kit.verify_world, world)

        world2 = self.root / "world2"
        kit.extract_bundle(self.bundle, world2)
        (world2 / "unexpected.txt").write_text("x", encoding="utf-8")
        self.assert_kit_error("manifest_mismatch", kit.verify_world, world2)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_world_and_source_symlinks_are_rejected(self) -> None:
        world = self.root / "world"
        kit.extract_bundle(self.bundle, world)
        os.symlink("events/base.ndjson", world / "alias")
        self.assert_kit_error("invalid_world", kit.verify_world, world)

        source = make_source(self.root / "symlink-source")
        os.symlink("events/base.ndjson", source / "alias")
        self.assert_kit_error(
            "unsafe_source",
            kit.pack_bundle,
            source,
            self.root / "symlink.afterimage",
            "Bad",
            "bad-1",
        )

    def test_path_attacks_are_rejected_before_manifest_loading(self) -> None:
        attacks = ["../escape", "/absolute", "C:/drive", "a\\b", "e\u0301.txt", "a//b"]
        for index, name in enumerate(attacks):
            path = self.root / f"attack-{index}.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(name, b"x")
            with self.subTest(path=name):
                self.assert_kit_error("invalid_path", kit.load_bundle, path)

    def test_duplicate_path_directory_symlink_and_encryption_are_rejected(self) -> None:
        duplicate = self.root / "duplicate.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zipfile.ZipFile(duplicate, "w") as archive:
                archive.writestr("manifest.json", b"{}")
                archive.writestr("manifest.json", b"{}")
        self.assert_kit_error("duplicate_path", kit.load_bundle, duplicate)

        directory = self.root / "directory.zip"
        with zipfile.ZipFile(directory, "w") as archive:
            archive.writestr("folder/", b"")
        self.assert_kit_error("invalid_path", kit.load_bundle, directory)

        symlink = self.root / "symlink.zip"
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(symlink, "w") as archive:
            archive.writestr(info, b"target")
        self.assert_kit_error("unsafe_member", kit.load_bundle, symlink)

        encrypted = self.root / "encrypted.zip"
        encrypted.write_bytes(self.bundle_path.read_bytes())
        payload = bytearray(encrypted.read_bytes())
        offset = 0
        while True:
            offset = payload.find(b"PK\x03\x04", offset)
            if offset < 0:
                break
            flags = struct.unpack_from("<H", payload, offset + 6)[0] | 1
            struct.pack_into("<H", payload, offset + 6, flags)
            offset += 4
        offset = 0
        while True:
            offset = payload.find(b"PK\x01\x02", offset)
            if offset < 0:
                break
            flags = struct.unpack_from("<H", payload, offset + 8)[0] | 1
            struct.pack_into("<H", payload, offset + 8, flags)
            offset += 4
        encrypted.write_bytes(payload)
        self.assert_kit_error("unsafe_member", kit.load_bundle, encrypted)

    def test_unsupported_compression_and_declared_oversize_are_rejected(self) -> None:
        if hasattr(zipfile, "ZIP_BZIP2"):
            path = self.root / "bzip.zip"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_BZIP2) as archive:
                archive.writestr("manifest.json", b"{}")
            self.assert_kit_error("unsafe_member", kit.load_bundle, path)

        oversized = self.root / "oversized.zip"
        oversized.write_bytes(self.bundle_path.read_bytes())
        payload = bytearray(oversized.read_bytes())
        offset = payload.find(b"PK\x01\x02")
        self.assertGreaterEqual(offset, 0)
        struct.pack_into("<L", payload, offset + 24, kit.HARD_LIMITS["max_single_file_bytes"] + 1)
        oversized.write_bytes(payload)
        self.assert_kit_error("limit_exceeded", kit.load_bundle, oversized)

    def test_crc_corruption_is_rejected(self) -> None:
        path = self.root / "stored.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", b"1234567890")
        payload = bytearray(path.read_bytes())
        name_length = struct.unpack_from("<H", payload, 26)[0]
        extra_length = struct.unpack_from("<H", payload, 28)[0]
        data_offset = 30 + name_length + extra_length
        payload[data_offset] ^= 0x01
        path.write_bytes(payload)
        self.assert_kit_error("invalid_zip", kit.load_bundle, path)

    def test_manifest_noncanonical_mismatch_extra_and_hard_limit_are_rejected(self) -> None:
        pretty = self.root / "pretty.zip"
        copy_zip(
            self.bundle_path,
            pretty,
            lambda name, data: json.dumps(json.loads(data), indent=2).encode() if name == "manifest.json" else data,
        )
        self.assert_kit_error("noncanonical_json", kit.load_bundle, pretty)

        wrong_digest = self.root / "wrong-digest.zip"
        def corrupt_manifest(name: str, data: bytes) -> bytes:
            if name != "manifest.json":
                return data
            value = json.loads(data)
            value["files"][0]["sha256"] = "0" * 64
            return cre.canonical_bytes(value)
        copy_zip(self.bundle_path, wrong_digest, corrupt_manifest)
        self.assert_kit_error("manifest_mismatch", kit.load_bundle, wrong_digest)

        extra = self.root / "extra.zip"
        copy_zip(self.bundle_path, extra, extra=[("extra.txt", b"x")])
        self.assert_kit_error("manifest_mismatch", kit.load_bundle, extra)

        excessive = self.root / "excessive.zip"
        def excessive_manifest(name: str, data: bytes) -> bytes:
            if name != "manifest.json":
                return data
            value = json.loads(data)
            value["limits"]["max_files"] = kit.HARD_LIMITS["max_files"] + 1
            return cre.canonical_bytes(value)
        copy_zip(self.bundle_path, excessive, excessive_manifest)
        self.assert_kit_error("invalid_limit", kit.load_bundle, excessive)

    def test_missing_required_file_and_nested_archive_are_rejected_at_pack(self) -> None:
        missing_source = make_source(self.root / "missing")
        (missing_source / "cases/index.json").unlink()
        self.assert_kit_error(
            "missing_required_file",
            kit.pack_bundle,
            missing_source,
            self.root / "missing.afterimage",
            "Missing",
            "missing-1",
        )

        nested_source = make_source(self.root / "nested")
        write_bytes(nested_source / "assets/payload.zip", b"not even a zip")
        self.assert_kit_error(
            "nested_archive",
            kit.pack_bundle,
            nested_source,
            self.root / "nested.afterimage",
            "Nested",
            "nested-1",
        )

    def test_noncanonical_json_and_ndjson_variants_are_rejected(self) -> None:
        variants = {
            "missing-lf": cre.canonical_bytes(cre.event_view(base_event())),
            "crlf": cre.canonical_bytes(cre.event_view(base_event())) + b"\r\n",
            "blank": cre.canonical_bytes(cre.event_view(base_event())) + b"\n\n",
            "spaced": b'{ "x":1}\n',
        }
        for label, data in variants.items():
            source = make_source(self.root / label)
            write_bytes(source / "events/base.ndjson", data)
            with self.subTest(label=label):
                self.assert_kit_error(
                    "noncanonical_ndjson" if label != "spaced" else "noncanonical_json",
                    kit.pack_bundle,
                    source,
                    self.root / f"{label}.afterimage",
                    "Invalid",
                    label,
                )

        json_source = make_source(self.root / "pretty-json")
        write_bytes(json_source / "cases/index.json", b'{"cases": [], "format": "afterimage-cases/0.1"}')
        self.assert_kit_error(
            "noncanonical_json",
            kit.pack_bundle,
            json_source,
            self.root / "pretty-json.afterimage",
            "Invalid",
            "pretty-json",
        )

    def test_event_id_mismatch_is_rejected(self) -> None:
        source = make_source(self.root / "bad-id")
        event = cre.event_view(base_event())
        event["id"] = "sha256:" + "f" * 64
        write_bytes(source / "events/base.ndjson", cre.canonical_bytes(event) + b"\n")
        self.assert_kit_error(
            "event_id_mismatch",
            kit.pack_bundle,
            source,
            self.root / "bad-id.afterimage",
            "Invalid",
            "bad-id",
        )

    def test_unknown_projection_and_branch_adapter_validation(self) -> None:
        world_path = self.root / "world"
        kit.extract_bundle(self.bundle, world_path)
        world = kit.verify_world(world_path)
        self.assert_kit_error("unknown_projection", kit.materialize_case, world, "missing")
        self.assert_kit_error(
            "invalid_schema",
            kit.materialize_case,
            world,
            "sample.public",
            {"format": "afterimage-branch/0.1"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
