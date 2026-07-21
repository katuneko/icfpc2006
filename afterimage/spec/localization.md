# Afterimage localization contract 0.1

Afterimage has one authoritative semantic world and multiple presentation
languages. A locale must never create a second problem instance.

## Identity boundary

The `.afterimage` archive, its BundleId, case descriptors, events, rules,
projections, schemas, intervention policies, scoring, witnesses, and receipts
are language-independent. Locale packs are ordinary files distributed beside
the archive. They are not inserted into or hashed into the world bundle.

Changing `--locale` may change only titles, premise text, submission guidance,
diagnostic guidance, hints, Markdown headings, and human-readable CLI labels.
It must not change visibility, answer schemas, replay, verification, score,
unlocks, telemetry, or canonical JSON receipts.

The supported locale tags are `en`, `ja`, `zh-Hans`, and `de`. English is the
default. `AFTERIMAGE_LOCALE` selects a default for the process; an explicit
`--locale` takes precedence.

## Pack format

A pack is `locales/LOCALE.json` with format
`afterimage-locale-pack/0.1`. It declares the exact BundleIds it supports, one
complete entry per production case, and the human-readable UI templates.
Every case entry contains:

- `title`, `premise`, `submission`, and `diagnostics` strings;
- exactly three ordered hints, preserving the original hint levels; and
- a generated `protected` list of semantic tokens that must occur verbatim in
  the translation.

Protected tokens include case IDs, public references, JSON pointers, protocol
names, and other code-like values that occur in both semantic data and prose.
They are never translated. JSON keys, enum values, topics, IDs, digests,
error codes, schemas, and payload values remain exactly as specified by the
authoritative bundle even when they are not repeated in prose.

## Loading and failure behavior

The player validates a selected pack before presenting a case or hint. A pack
must have the expected identity, exact fields, valid UI placeholders, support
the current BundleId, contain every case required by that bundle, retain all
protected tokens, and contain three non-empty hints per case. A malformed or
incomplete selected pack fails closed with `invalid_locale`; the player does
not silently mix languages.

The production locale gate additionally requires exact 75-case coverage,
225 hints, identical protected-token inventories, and translated narrative
fields. The twelve-case slice may use the same 75-case pack as a strict
superset.

## Normative diagnostics

Diagnostic `code` values and canonical JSON remain normative and untranslated.
Case-specific diagnostic guidance and the human-readable `VALID`/`INVALID`
shell are localized. An implementation may localize more explanatory error
prose later, but it must retain the raw code and must not alter verifier
behavior.

## Required invariance test

For every supported locale, loading a pack and inspecting a case must leave
the archive SHA-256 and BundleId unchanged. Verifying the same canonical
witness against the same facts must yield byte-identical canonical receipts.
This condition is a release gate, not a documentation convention.
