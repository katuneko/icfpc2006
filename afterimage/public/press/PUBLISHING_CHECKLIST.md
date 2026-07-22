# Publishing checklist

## Required decisions

- [ ] Replace `{{DOWNLOAD_URL}}` in copy and fact sheet.
- [ ] Replace `{{CONTACT_EMAIL}}`.
- [ ] Set and publish `{{LICENSE}}` for code, bundle, localization, and media.
- [ ] Choose the public release date and timezone.
- [ ] Decide whether the exact competition format, duration, and scoreboard are
      being announced now or later.
- [ ] Confirm independent legal review of title, marks, third-party references,
      and generated-image policy where applicable.

## Technical release gate

- [ ] Run `python3 afterimage/tools/build_public_release.py --check`.
- [ ] Run `python3 afterimage/tools/check_public_assets.py`.
- [ ] Run `python3 afterimage/tools/check_all.py`.
- [ ] Verify the public archive BundleId and SHA-256 against `FACT_SHEET.md`.
- [ ] Extract every published ZIP into a clean directory and run its smoke test.
- [ ] Confirm no author witness, baseline, case source, golden, or authoring
      tool is present.

## Editorial gate

- [ ] Review all four launch texts with a native speaker.
- [ ] Preview social cards at actual feed size.
- [ ] Check alt text and keyboard navigation.
- [ ] Confirm that screenshots remain in the green spoiler tier.
- [ ] Verify every public URL from a clean browser profile.

