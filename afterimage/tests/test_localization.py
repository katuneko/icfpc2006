#!/usr/bin/env python3
"""Locale coverage, integrity, and semantic-invariance gates."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import afterimage_kit as kit  # noqa: E402
import cre  # noqa: E402
import localization  # noqa: E402
import player  # noqa: E402
import verify_witness as verifier  # noqa: E402


ARCHIVE = ROOT / "dist" / "afterimage-production-2.1-dev.afterimage"
AUTHORS = ROOT / "dist" / "production-2.1-author-baselines"
GOLDEN = json.loads((ROOT / "content" / "production" / "golden.json").read_text(encoding="utf-8"))


class LocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = kit.load_bundle(ARCHIVE)
        cls.case_ids = {item["id"] for item in cls.world.json_values["cases/index.json"]["cases"]}

    def test_all_packs_cover_the_frozen_release_and_preserve_tokens(self) -> None:
        self.assertEqual(len(self.case_ids), 75)
        for locale_code in localization.SUPPORTED:
            pack = localization.load_pack(
                locale_code,
                bundle=self.world.bundle,
                expected_cases=self.case_ids,
                exact_cases=True,
            )
            self.assertEqual(set(pack.cases), self.case_ids)
            self.assertEqual(sum(len(case["hints"]) for case in pack.cases.values()), 225)

    def test_locale_selection_changes_presentation_but_not_bundle_or_receipt(self) -> None:
        archive_sha256 = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
        self.assertEqual(archive_sha256, GOLDEN["archive_sha256"])
        self.assertEqual(self.world.bundle, GOLDEN["bundle"])
        witness = (AUTHORS / "ORIENT.001.witness.json").read_bytes()
        receipts = []
        stories = []
        with tempfile.TemporaryDirectory(prefix="afterimage-locales-") as temporary:
            workspace = Path(temporary) / "desk"
            player.init_workspace(ARCHIVE, workspace, telemetry=False)
            state, world = player.load_workspace(workspace)
            orient = player.descriptors(world)["ORIENT.001"]
            for locale_code in localization.SUPPORTED:
                pack = localization.load_pack(locale_code, bundle=world.bundle)
                view = player.inspect_target(workspace, state, world, "ORIENT.001", locale_code)
                self.assertEqual(view["locale"], locale_code)
                self.assertEqual(view["family"], "ORIENT")
                self.assertEqual(view["answer_schema"], world.json_values[orient["answer_schema"]])
                self.assertEqual(view["story"], pack.story("ORIENT.001"))
                stories.append(view["story"])
                receipt = verifier.verify_witness_bytes(world, witness, "ORIENT.001.witness.json", set())
                receipts.append(cre.canonical_bytes(receipt))
        self.assertEqual(len(set(receipts)), 1)
        self.assertEqual(len(set(stories)), len(localization.SUPPORTED))

    def test_missing_protected_token_is_rejected(self) -> None:
        source = json.loads((ROOT / "locales" / "en.json").read_text(encoding="utf-8"))
        case_id = next(case_id for case_id, case in source["cases"].items() if case["protected"])
        token = source["cases"][case_id]["protected"][0]
        case = source["cases"][case_id]
        for field in ("title", "premise", "submission", "diagnostics"):
            case[field] = case[field].replace(token, "translated-token")
        case["hints"] = [hint.replace(token, "translated-token") for hint in case["hints"]]
        with tempfile.TemporaryDirectory(prefix="afterimage-bad-locale-") as temporary:
            root = Path(temporary)
            (root / "en.json").write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            with mock.patch.object(localization, "LOCALES_ROOT", root):
                with self.assertRaisesRegex(localization.LocalizationError, "protected token is missing"):
                    localization.load_pack("en")


if __name__ == "__main__":
    unittest.main()
