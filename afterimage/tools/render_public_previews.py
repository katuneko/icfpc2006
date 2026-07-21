#!/usr/bin/env python3
"""Render four localized site previews and the public one-sheet PDF."""

from __future__ import annotations

import functools
import http.server
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
GENERATED = PUBLIC / "assets" / "generated"
LOCALES = ("en", "ja", "zh-Hans", "de")


class SilentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def chromium() -> str:
    for name in ("chromium-browser", "chromium", "google-chrome"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("Chromium is required to render public previews")


def run_chromium(executable: str, arguments: list[str]) -> None:
    profile = Path(tempfile.mkdtemp(prefix="afterimage-chromium-"))
    try:
        completed = subprocess.run(
            [
                executable,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-features=OptimizationGuideModelDownloading",
                f"--user-data-dir={profile}",
                *arguments,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def render() -> None:
    executable = chromium()
    GENERATED.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(SilentHandler, directory=str(PUBLIC))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        for locale in LOCALES:
            output = GENERATED / f"site-preview-{locale}.png"
            run_chromium(
                executable,
                [
                    "--hide-scrollbars",
                    "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=5000",
                    "--window-size=1440,1000",
                    f"--screenshot={output}",
                    f"http://127.0.0.1:{port}/site/?lang={locale}",
                ],
            )
            if not output.is_file() or output.stat().st_size < 100_000:
                raise RuntimeError(f"site preview was not rendered: {locale}")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    contact = GENERATED / "site-preview-contact-sheet.jpg"
    completed = subprocess.run(
        [
            "magick",
            "montage",
            *[str(GENERATED / f"site-preview-{locale}.png") for locale in LOCALES],
            "-thumbnail",
            "720x500",
            "-tile",
            "2x2",
            "-geometry",
            "+12+12",
            "-background",
            "#08131f",
            str(contact),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)

    pdf = GENERATED / "afterimage-one-sheet.pdf"
    run_chromium(
        executable,
        [
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            (PUBLIC / "press" / "ONE_SHEET.html").as_uri(),
        ],
    )
    if not pdf.is_file() or pdf.stat().st_size < 100_000:
        raise RuntimeError("one-sheet PDF was not rendered")


def main() -> int:
    try:
        render()
        print("public-previews: PASS: 4 localized pages and 1 one-sheet PDF")
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"public-previews: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
