from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_public_release as public_release  # noqa: E402


class PublicReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="afterimage-public-test-")
        root = Path(cls.temporary.name)
        cls.first = root / "first"
        cls.second = root / "second"
        cls.first_result = public_release.build(cls.first)
        cls.second_result = public_release.build(cls.second)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_release_identity_and_public_smoke_are_fixed(self) -> None:
        self.assertEqual(
            self.first_result["bundle"], public_release.EXPECTED_BUNDLE
        )
        self.assertEqual(
            self.first_result["archive_sha256"],
            public_release.EXPECTED_ARCHIVE_SHA256,
        )
        self.assertEqual(self.first_result["cases"], 75)
        self.assertEqual(self.first_result["points"], 10000)
        self.assertEqual(self.first_result["locales"], ["en", "ja", "zh-Hans", "de"])
        checked = public_release.verify(self.first, smoke=True)
        self.assertTrue(checked["smoke"])

    def test_repeated_public_builds_are_byte_identical(self) -> None:
        first = json.loads((self.first / "checksums.json").read_text())
        second = json.loads((self.second / "checksums.json").read_text())
        self.assertEqual(first, second)
        for entry in first["files"]:
            self.assertEqual(
                (self.first / entry["path"]).read_bytes(),
                (self.second / entry["path"]).read_bytes(),
            )

    def test_player_kit_has_no_author_material(self) -> None:
        package = self.first / "afterimage-player-kit-2.1.zip"
        with zipfile.ZipFile(package) as archive:
            names = archive.namelist()
            lowered = [name.lower() for name in names]
            self.assertIn("afterimage-2.1.afterimage", names)
            self.assertIn("locales/ja.json", names)
            self.assertIn("locales/zh-Hans.json", names)
            self.assertIn("locales/de.json", names)
            self.assertFalse(any("author" in name for name in lowered))
            self.assertFalse(any("baseline" in name for name in lowered))
            self.assertFalse(any(name.endswith(".witness.json") for name in lowered))
            self.assertFalse(any(name.endswith(".receipt.json") for name in lowered))
            self.assertEqual(
                archive.read("afterimage-2.1.afterimage"),
                public_release.ARCHIVE.read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
