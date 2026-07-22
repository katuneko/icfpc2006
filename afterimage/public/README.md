# Afterimage public release assets

This directory is the publication source of truth. Generated outputs are
reproducible from checked-in source art, SVG marks, launch copy, press
documents, and the verified production bundle.

## Deliverables

- `assets/brand/`: full-color, monochrome, light, dark, and favicon SVGs.
- `assets/source/`: generated master key art retained at native resolution.
- `assets/generated/`: web hero, poster, and compressed editorial derivatives.
- `assets/social/`: 1200×630 localized cards and square/portrait variants.
- `brand/BRAND_GUIDE.md`: visual and editorial rules.
- `copy/`: launch and social copy in all four languages.
- `press/`: fact sheet, FAQ, alt text, spoiler policy, provenance, and release
  checklist.
- `site/`: dependency-free four-language landing page.
- `release/`: public quickstart and release notes used by the package builder.

## Build

```bash
bash afterimage/tools/build_public_assets.sh
python3 afterimage/tools/build_public_release.py afterimage/dist/public-release
python3 afterimage/tools/check_public_assets.py
```

The asset build requires ImageMagick, Chromium, and Noto Sans fonts. The
release packages themselves require only Python 3.11+.

Preview the site from the repository root:

```bash
python3 -m http.server 8000
```

Then open `http://127.0.0.1:8000/afterimage/public/site/`.

## Source-image prompts

The exact prompts and built-in image-generation mode are recorded in
`assets/PROMPTS.md`. Derived assets are deterministic; regenerating the source
art is not part of the ordinary build.
