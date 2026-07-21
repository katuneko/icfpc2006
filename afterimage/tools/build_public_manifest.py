#!/usr/bin/env python3
"""Write the canonical manifest for Afterimage public visual assets."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "public" / "assets"
sys.path.insert(0, str(ROOT / "reference" / "python"))

import cre  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_size(path: Path) -> list[int] | None:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return None
    result = subprocess.run(
        ["magick", "identify", "-format", "%w %h", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot identify {path}: {result.stderr}")
    width, height = result.stdout.split()
    return [int(width), int(height)]


def role(relative: str) -> str:
    if relative == "PROMPTS.md":
        return "generation-record"
    if relative.startswith("brand/"):
        return "brand"
    if relative.startswith("source/"):
        return "source-master"
    if relative.startswith("social/"):
        return "social"
    if relative.startswith("generated/contact-sheet"):
        return "review-sheet"
    if relative.startswith("generated/site-preview"):
        return "site-preview"
    if relative.startswith("generated/"):
        return "editorial"
    return "support"


def build() -> dict[str, Any]:
    files = []
    for path in sorted(
        (
            item
            for item in ASSETS.rglob("*")
            if item.is_file() and item.name != "manifest.json"
        ),
        key=lambda item: item.relative_to(ASSETS).as_posix().encode("utf-8"),
    ):
        if path.is_symlink():
            raise RuntimeError(f"asset is a symlink: {path}")
        relative = path.relative_to(ASSETS).as_posix()
        entry: dict[str, Any] = {
            "path": relative,
            "role": role(relative),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        size = image_size(path)
        if size is not None:
            entry["pixels"] = size
        files.append(entry)
    return {
        "format": "afterimage-public-assets/0.1",
        "brand": "Afterimage: The Counterfactual City",
        "license": "publisher-decision-required",
        "source_art_mode": "OpenAI built-in image generation",
        "derivative_builder": "tools/build_public_assets.sh",
        "files": files,
    }


def main() -> int:
    try:
        manifest = build()
        (ASSETS / "manifest.json").write_bytes(cre.canonical_bytes(manifest))
        print(
            json.dumps(
                {
                    "type": "public-asset-manifest",
                    "files": len(manifest["files"]),
                    "output": str(ASSETS / "manifest.json"),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"public-asset-manifest: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
