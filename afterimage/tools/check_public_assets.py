#!/usr/bin/env python3
"""Validate Afterimage publication copy, visual assets, site, and release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
ASSETS = PUBLIC / "assets"
EXPECTED_LOCALES = ["en", "ja", "zh-Hans", "de"]
EXPECTED_BUNDLE = (
    "sha256:517038cdd97cb7d3687f53272e8964a11ffcc1cca82cc69a73668bf56aea0514"
)
EXPECTED_ARCHIVE_SHA256 = (
    "4d2015a522281bddeaa3ec9fedda28715677663926bea924a05494ee78ca57af"
)

REQUIRED_DOCUMENTS = [
    "README.md",
    "brand/BRAND_GUIDE.md",
    "copy/ANNOUNCEMENTS.md",
    "copy/launch-copy.json",
    "press/ALT_TEXT.md",
    "press/ASSET_PROVENANCE.md",
    "press/FACT_SHEET.md",
    "press/FAQ.md",
    "press/PUBLISHING_CHECKLIST.md",
    "press/ONE_SHEET.html",
    "press/SPOILER_GUIDE.md",
    "release/ENGINE_QUICKSTART.md",
    "release/PLAYER_QUICKSTART.md",
    "release/RELEASE_NOTES.md",
    "site/app.js",
    "site/index.html",
    "site/styles.css",
]

EXPECTED_IMAGES = {
    "source/key-art-master-final.png": (1536, 1024),
    "source/poster-art-master.png": (864, 1821),
    "generated/hero-1920x1080.webp": (1920, 1080),
    "generated/hero-2400x1350.jpg": (2400, 1350),
    "generated/key-art-editorial-1536x1024.webp": (1536, 1024),
    "generated/poster-1080x1920.webp": (1080, 1920),
    "generated/poster-1600x2000.webp": (1600, 2000),
    "generated/poster-a4-blank.jpg": (2480, 3508),
    "generated/mark-512.png": (512, 512),
    "generated/wordmark-light-1600.png": (1600, 369),
    "generated/wordmark-dark-1600.png": (1600, 369),
    **{
        f"social/afterimage-og-{locale}.png": (1200, 630)
        for locale in EXPECTED_LOCALES
    },
    **{
        f"social/afterimage-square-{locale}.png": (1080, 1080)
        for locale in EXPECTED_LOCALES
    },
    **{
        f"social/afterimage-story-{locale}.png": (1080, 1920)
        for locale in EXPECTED_LOCALES
    },
}


class AssetError(Exception):
    """A stable public-asset validation failure."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["magick", "identify", "-format", "%w %h", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssetError(f"cannot identify {path}: {result.stderr}")
    width, height = result.stdout.split()
    return int(width), int(height)


def validate_copy() -> dict[str, int]:
    copy = json.loads(
        (PUBLIC / "copy" / "launch-copy.json").read_text(encoding="utf-8")
    )
    if copy.get("format") != "afterimage-launch-copy/0.1":
        raise AssetError("wrong launch-copy format")
    locales = copy.get("locales")
    if not isinstance(locales, dict) or list(locales) != EXPECTED_LOCALES:
        raise AssetError("launch-copy locale order or coverage changed")
    required = {
        "language",
        "htmlLang",
        "editorialTitle",
        "kicker",
        "headline",
        "lead",
        "primaryCta",
        "secondaryCta",
        "sectionWorld",
        "sectionWorldBody",
        "sectionFamilies",
        "families",
        "sectionLoop",
        "loopBody",
        "sectionFacts",
        "facts",
        "footer",
        "downloadLabel",
    }
    for locale, value in locales.items():
        if not isinstance(value, dict) or set(value) != required:
            raise AssetError(f"launch-copy fields changed for {locale}")
        if any("{{" in item for item in value.values() if isinstance(item, str)):
            raise AssetError(f"placeholder leaked into site copy for {locale}")
        families = value["families"]
        if (
            not isinstance(families, list)
            or len(families) != 8
            or sum(int(family[2]) for family in families) != 75
        ):
            raise AssetError(f"family coverage changed for {locale}")
        if not isinstance(value["facts"], list) or len(value["facts"]) != 6:
            raise AssetError(f"public fact coverage changed for {locale}")
    return {"locales": len(locales), "families": 8}


def validate_svg(path: Path) -> None:
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise AssetError(f"invalid SVG {path}: {exc}") from exc
    root = tree.getroot()
    if not root.tag.endswith("svg"):
        raise AssetError(f"SVG root missing: {path}")
    for node in root.iter():
        if node.tag.endswith("script"):
            raise AssetError(f"script in public SVG: {path}")
        for value in node.attrib.values():
            if "http://" in value or "https://" in value or "javascript:" in value:
                raise AssetError(f"external reference in public SVG: {path}")


def validate_assets() -> dict[str, int]:
    manifest_path = ASSETS / "manifest.json"
    if not manifest_path.is_file():
        raise AssetError("asset manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "afterimage-public-assets/0.1":
        raise AssetError("wrong asset manifest format")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise AssetError("asset manifest files missing")
    actual = {
        path.relative_to(ASSETS).as_posix(): path
        for path in ASSETS.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if [entry.get("path") for entry in entries] != sorted(
        actual, key=lambda item: item.encode("utf-8")
    ):
        raise AssetError("asset manifest path coverage drift")
    for entry in entries:
        path = actual[entry["path"]]
        if path.is_symlink():
            raise AssetError(f"asset is a symlink: {entry['path']}")
        if entry.get("bytes") != path.stat().st_size or entry.get("sha256") != sha256(
            path
        ):
            raise AssetError(f"asset manifest checksum drift: {entry['path']}")
    for relative, expected_size in EXPECTED_IMAGES.items():
        path = ASSETS / relative
        if not path.is_file() or path.is_symlink():
            raise AssetError(f"required image missing: {relative}")
        if image_size(path) != expected_size:
            raise AssetError(f"wrong image dimensions: {relative}")
        if path.stat().st_size < 20_000:
            raise AssetError(f"image is suspiciously small: {relative}")
    for path in sorted((ASSETS / "brand").glob("*.svg")):
        validate_svg(path)
    if len(list((ASSETS / "brand").glob("*.svg"))) != 5:
        raise AssetError("brand SVG coverage changed")
    return {"manifest_files": len(entries), "required_images": len(EXPECTED_IMAGES)}


def validate_site() -> dict[str, int]:
    html = (PUBLIC / "site" / "index.html").read_text(encoding="utf-8")
    css = (PUBLIC / "site" / "styles.css").read_text(encoding="utf-8")
    app = (PUBLIC / "site" / "app.js").read_text(encoding="utf-8")
    if "{{" in html or "{{" in css or "{{" in app:
        raise AssetError("publication placeholder leaked into site")
    keys = set(re.findall(r'data-copy="([A-Za-z0-9]+)"', html))
    locale_copy = json.loads(
        (PUBLIC / "copy" / "launch-copy.json").read_text(encoding="utf-8")
    )["locales"]["en"]
    if not keys <= set(locale_copy):
        raise AssetError(f"site copy key missing: {sorted(keys - set(locale_copy))}")
    if "prefers-reduced-motion" not in css or "skip-link" not in html:
        raise AssetError("site accessibility controls are missing")
    if "innerHTML" in app:
        raise AssetError("site localization must not inject HTML")
    for locale in EXPECTED_LOCALES:
        if f'data-locale-button="{locale}"' not in html:
            raise AssetError(f"site locale button missing: {locale}")
    return {"copy_keys": len(keys), "locale_buttons": len(EXPECTED_LOCALES)}


def validate_documents() -> dict[str, int]:
    for relative in REQUIRED_DOCUMENTS:
        path = PUBLIC / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size < 80:
            raise AssetError(f"required public document missing: {relative}")
    facts = (PUBLIC / "press" / "FACT_SHEET.md").read_text(encoding="utf-8")
    if EXPECTED_BUNDLE not in facts or EXPECTED_ARCHIVE_SHA256 not in facts:
        raise AssetError("fact sheet artifact identity drift")
    return {"documents": len(REQUIRED_DOCUMENTS)}


def validate_release(path: Path, smoke: bool) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "tools"))
    import build_public_release  # noqa: PLC0415

    return build_public_release.verify(path, smoke=smoke)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        type=Path,
        default=ROOT / "dist" / "public-release",
        help="public release directory to validate",
    )
    parser.add_argument("--skip-release", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result: dict[str, Any] = {
            "type": "public-assets-verified",
            "copy": validate_copy(),
            "assets": validate_assets(),
            "site": validate_site(),
            "documents": validate_documents(),
        }
        if not args.skip_release:
            result["release"] = validate_release(args.release, smoke=args.smoke)
        print(
            json.dumps(result, indent=2, sort_keys=True)
            if args.pretty
            else json.dumps(result, sort_keys=True, separators=(",", ":"))
        )
        return 0
    except (AssetError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"public-assets: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
