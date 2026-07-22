# Afterimage Bundle and Witness Format 0.1

Status: normative implementation specification for the vertical slice.

## 1. Goals

The artifact format must be:

- inspectable with ordinary tools;
- independent of host operating system and programming language;
- content-addressed and reproducible;
- usable entirely offline;
- safe to parse under explicit size limits;
- capable of carrying the world, cases, validators, narrative, and fixtures;
- stable enough that a witness identifies exactly what it was checked against.

For the vertical slice, a `.afterimage` file is an ordinary ZIP archive. ZIP
bytes themselves are not canonical; the logical file set is.

## 2. Paths and files

Archive paths MUST:

- be normalized NFC UTF-8;
- use `/` separators;
- be relative and non-empty;
- contain no empty, `.` or `..` segment;
- contain no NUL, backslash, drive prefix, or leading slash;
- be unique after normalization;
- be at most 240 UTF-8 bytes.

Symlinks, device entries, encrypted ZIP members, and nested archives are
forbidden. A verifier MUST reject them before extraction.

Required logical files:

```text
manifest.json
program/continuity.cre.json
events/base.ndjson
projections/index.json
cases/index.json
fixtures/conformance/index.json
```

Optional logical directories:

```text
cases/<CASE-ID>/
projections/
story/
schemas/
assets/
fixtures/
```

The vertical slice uses text and JSON assets only. Later releases MAY include
images or audio with complete text alternatives.

## 3. Manifest and bundle digest

`manifest.json` contains:

```json
{
  "format": "afterimage-bundle/0.1",
  "semantics": "cre/0.1",
  "title": "Afterimage vertical slice",
  "revision": "slice-0.1.0",
  "limits": {
    "max_files": 512,
    "max_total_uncompressed_bytes": 67108864,
    "max_single_file_bytes": 16777216,
    "max_line_bytes": 1048576
  },
  "files": [
    {"path":"events/base.ndjson","bytes":123,"sha256":"..."}
  ]
}
```

The `files` list covers every logical file except `manifest.json`, sorted by
path UTF-8 bytes. Each digest is lowercase raw SHA-256 hex without a prefix.
Declared sizes are uncompressed byte sizes.

The logical bundle digest is:

```text
H("afterimage/bundle/1", canon({
  "format": manifest.format,
  "semantics": manifest.semantics,
  "title": manifest.title,
  "revision": manifest.revision,
  "limits": manifest.limits,
  "files": manifest.files
}))
```

Unknown top-level manifest keys are rejected in 0.1. Thus every recognized
manifest field that can affect loading, limits, or presentation is bound by the
bundle digest.

The four limit keys shown above are required. They count logical files other
than `manifest.json`, decoded uncompressed bytes, bytes in one logical file,
and bytes before LF in one NDJSON line. A declared limit MUST be positive and
MUST NOT exceed the kit hard ceilings of 4096 files, 128 MiB total, 32 MiB per
file, and 1 MiB per line. The archive itself MUST be at most 128 MiB.

`manifest.json` and every `*.json` logical file MUST be UTF-8 canonical JSON
with no BOM, leading/trailing whitespace, or final newline. Duplicate raw or
NFC-normalized keys are invalid. JSON numbers are CRE signed 64-bit integers;
floating point does not exist.

ZIP timestamps, compression level, member order, permissions, and comments do
not affect this digest. Release tooling SHOULD nevertheless produce
byte-reproducible ZIPs with sorted files and a fixed timestamp.

## 4. NDJSON

Each non-empty line is one canonical JSON value followed by LF. CRLF, blank
lines, and a missing final LF are rejected in canonical inputs. No line may
exceed the manifest's declared line-size limit.

`events/base.ndjson` contains event envelopes:

```json
{"id":"sha256:...","topic":"grid.switch","at":120,"payload":{},"parents":[],"origin":{"kind":"base","source":"substation-4","sequence":9}}
```

The loader recomputes every ID.

## 5. Index files

`projections/index.json` is:

```json
{"format":"afterimage-projections/0.1","projections":[{"id":"desk.public","path":"projections/desk.public.cre.json"}]}
```

Projection entries have exactly `id` and `path`, are sorted by normalized ID
UTF-8 bytes, and have unique IDs and paths. Each path names a listed canonical
JSON file containing the compiled projection object from the CRE spec.

`cases/index.json` is:

```json
{"format":"afterimage-cases/0.1","cases":[]}
```

Its case entries use the descriptor shape below, have unique IDs, and are
sorted by normalized ID UTF-8 bytes.

`fixtures/conformance/index.json` is:

```json
{"format":"afterimage-conformance-index/0.1","cases":[{"name":"canonical-null","path":"fixtures/conformance/canonical-null.json"}]}
```

Fixture entries have exactly `name` and `path`, are sorted by normalized name
UTF-8 bytes, and point to listed canonical case JSON files. Empty case and
fixture lists are permitted in development bundles; a released slice declares
its required minimums separately.

## 6. Case descriptors

`cases/index.json` lists case descriptors by ID. Each descriptor includes:

```json
{
  "id": "CASCADE.001",
  "family": "CASCADE",
  "title": "Late Green",
  "points": 80,
  "requires": {"all":["case:ORIENT.001"],"at_least":{"count":2,"of":["case:ORIENT.002","case:ORIENT.003","case:ORIENT.004","case:ORIENT.005"]}},
  "input_branch": "root",
  "world": "worlds/CASCADE.001/world.json",
  "projection": "incident.late-green",
  "answer_schema": "schemas/cascade-answer.json",
  "validator": "cases/CASCADE.001/validator.cre.json",
  "intervention_policy": "cases/CASCADE.001/interventions.json",
  "score": "cases/CASCADE.001/score.json",
  "limits": {}
}
```

Case descriptors have exactly the thirteen keys shown above. `input_branch` is
either the literal `root` or `history:<case-id>`. A concrete BranchId MUST NOT
appear here: BranchId depends on the digest of the bundle containing this
descriptor and would create an unsatisfiable self-reference. `root` resolves
after the bundle digest is known. `history:<case-id>` requires a witness-carried
history whose final step names that case; replay derives the concrete parent.

`world` is either `global` or a listed case-world descriptor:

```json
{"format":"afterimage-case-world/0.1","program":"worlds/CASCADE.001/program.cre.json","events":"worlds/CASCADE.001/base.ndjson","projections":"worlds/CASCADE.001/projections/index.json"}
```

The three referenced paths name a compiled CRE program, canonical event
NDJSON, and a projection index with the same format as the global files.
Case-world isolation is semantic: adding a later case to the bundle MUST NOT
change an earlier case's active events, trace, or projection. `global` selects
the required top-level program/events/projections files and is intended for the
Continuity Desk itself, not as a shortcut for authored case fixtures.

`answer_schema`, `validator`, `intervention_policy`, and `score` are paths to
listed canonical JSON files. The vertical slice uses the following formats.

Case `limits` has at most `max_witness_bytes`, `replay`, and `validator`.
`max_witness_bytes` is a positive integer no greater than 1 MiB. `replay` and
`validator` are CRE resource-limit maps using the counter names from the CRE
specification; omitted counter values use CRE defaults.

```json
{"format":"afterimage-answer-schema/0.1","schema":{"type":"map","required":["event_id"],"properties":{"event_id":{"type":"id"}},"additional":false}}
```

The supported schema nodes are `null`, `bool`, `int`, `text`, `bytes`, `id`,
`list`, and `map`. Nodes MAY add `const` or `enum`; integers MAY add `minimum`
and `maximum`; text/list/map nodes MAY add the corresponding length bounds;
lists use `items`; maps use `required`, `properties`, and `additional`.
An answer schema MAY also contain one `embedded_witness` descriptor with a
non-root JSON Pointer, an allowlist of target cases, and `require_fact`. The
verifier canonicalizes and independently verifies that nested witness against
the same bundle. Nesting is limited to one level; failures expose only a stable
inner error code, not the expected answer or digest.

```json
{"format":"afterimage-validator/0.1","program":{"semantics":"cre/0.1","strata":[]},"decision_projection":{"id":"verify.decision","rows":[]}}
```

The verifier evaluates this CRE program in a separate validation world. It
supplies base events named `verify.answer`, `verify.replay`,
`verify.intervention`, one `verify.active` event per candidate active event,
and one `verify.baseline-active` event per input-baseline active event.
The replay payload contains branch/projection/trace digests, projection
`records`, `active_event_count`, active IDs, resource counters, canonical trace
items, and the trace's EventIds in order. It also contains the root or inherited
input branch, projection, trace, records, active IDs, and the canonical EventId
symmetric difference between that baseline and candidate active sets. Each active-event payload is
the complete original event envelope. The decision projection MUST produce
exactly one map with keys `valid`, `diagnostics`, and `metrics`. Validators
cannot mutate or retroactively affect either replayed world.

```json
{"format":"afterimage-intervention-policy/0.1","required":false,"allowed_kinds":[],"max_operations":0,"weights":{},"topics":[],"pointers":[],"retime":{"minimum":0,"maximum":0}}
```

```json
{"format":"afterimage-score/0.1","family":"ORIENT","reference_scale":16,"metric_bounds":{"witness_units":4096}}
```

Policy operation weights are non-negative integers; allowed operations must
have a declared weight. Empty lists prohibit those capabilities. The score
descriptor's family must match the case family. Family-specific verifier code
derives raw metrics and then applies the public lexicographic packing and score
formula.

When policy `required` is true, `witness.intervention` MUST be an intervention
envelope containing at least one operation. When it is false and
`max_operations` is zero, the witness field MUST be null; a redundant empty
envelope is rejected.

Unlock expressions use only:

- `case:<ID>` for validity facts;
- `cap:<NAME>` for explicit capabilities;
- `all`, `any`, and `at_least` combinators.

Unknown facts are false. Unlock state never changes case validity.

## 7. Intervention envelope

An intervention file is canonical JSON:

```json
{
  "format": "afterimage-intervention/0.1",
  "bundle": "sha256:...",
  "parent_branch": "sha256:...",
  "case": "CASCADE.001",
  "operations": [
    {"kind":"retime","event":"sha256:...","at":118}
  ]
}
```

The loader canonicalizes operations according to the CRE branch specification
and checks the case intervention policy before replay.

## 8. Witness envelope

All families use one outer witness envelope:

```json
{
  "format": "afterimage-witness/0.1",
  "semantics": "cre/0.1",
  "bundle": "sha256:...",
  "case": "CASCADE.001",
  "parent_branch": "sha256:...",
  "history": {
    "format": "afterimage-branch-history/0.1",
    "bundle": "sha256:...",
    "world": "worlds/ACT.001/world.json",
    "steps": [
      {"case":"CASCADE.010","operations":[{"kind":"retime","event":"sha256:...","at":118}]}
    ]
  },
  "intervention": {},
  "answer": {},
  "claimed": {
    "branch": "sha256:...",
    "projection": "sha256:...",
    "trace": "sha256:..."
  },
  "meta": {
    "producer": "team tool name",
    "comment": "optional human note"
  }
}
```

Rules:

- `format`, `semantics`, `bundle`, `case`, `parent_branch`, and `answer` are
  required.
- `intervention` is required for branch cases and null otherwise.
- `history` is forbidden when `input_branch` is `root` and required when it is
  `history:<case-id>`. It has exactly `format`, `bundle`, `world`, and `steps`.
  There are 1 through 32 steps; each has exactly `case` and a non-empty
  `operations` list.
- history starts with a root-input case. Each later step case declares
  `history:<previous-step-case>`, every step is an unlocked fact in the same
  logical world, and the final step case matches the target descriptor.
- the verifier canonicalizes and policy-checks every step against the active
  base/player set produced by the preceding step, discards and recomputes
  derived events, and derives the final parent BranchId. A non-intervening case
  inherits the existing history unchanged and is not added as a history step.
- `parent_branch` and the current intervention envelope MUST name that derived
  final BranchId. History tampering therefore fails before answer validation.
- `claimed` fields are optional cross-checks; the verifier recomputes them.
- `meta` is excluded from all gameplay metrics except total transport size.
- when present, `meta` contains only optional Text `producer` and `comment`
  fields and at least one of them;
- Unknown top-level keys are rejected in 0.1.
- Witness canonical bytes MUST fit the case limit.

The witness digest is:

```text
WitnessDigest = H("afterimage/witness/1", canon(witness))
```

`meta` is therefore transport-identifying even though it never affects
gameplay metrics or validity.

Family answer shapes:

- ORIENT: a small value or digest plus requested explanation tokens;
- CASCADE: intervention plus asserted invariant summary;
- MERGE: accepted IDs, rejected IDs with reason codes and any reason-specific
  proof link, plus the exact active order certificate;
- PULSE: encoded program and optional symbolic invariant certificate;
- MOSAIC: global graph and fragment-placement correspondence;
- LENS: compiled lens program and optional auxiliary-state schema;
- COVENANT: local policies and claimed response bound;
- PARADOX: two branch witnesses and an observational-equivalence certificate.

## 9. Verified world directory

`afterimage-kit extract` writes into a newly created staging directory and
atomically renames it to the requested destination. It never merges into an
existing path. Logical files retain their archive-relative paths and a kit
metadata file is added at the root:

```json
{"format":"afterimage-world/0.1","bundle":"sha256:...","archive_sha256":"...","files":[{"path":"events/base.ndjson","bytes":123,"sha256":"..."}]}
```

The metadata file is canonical JSON, is not part of the bundle digest, and
binds the source archive hash plus the already verified logical file set.
Consumers MUST either call `afterimage-kit verify-world` or independently
recheck every logical file against this metadata before replay. A world
directory containing symlinks, unexpected files, or changed bytes is invalid.

## 10. Verification pipeline

The authoritative verifier performs these steps in order:

1. parse archive and witness within transport limits;
2. canonicalize values and recompute bundle identity;
3. resolve case and unlock eligibility;
4. validate witness outer schema;
5. validate family answer schema and any one-level embedded witness;
6. for a history case, validate the case/world chain and replay every
   policy-checked step from the root base;
7. replay the root or inherited input baseline with the independent evaluator;
8. validate and price current intervention operations, classifying any derived target
   from the baseline and reporting its base ancestors;
9. replay the candidate branch independently;
10. compute the canonical baseline/candidate active-set difference;
11. run the case validator against recomputed results;
12. recompute all raw metrics;
13. compute score using the normative integer formula;
14. derive unlock facts and diagnostics;
15. emit one canonical receipt.

The first failing stage determines the primary error code. A verifier MAY
include secondary diagnostics but MUST NOT replace the primary code.

## 11. Receipt

A successful receipt is:

```json
{
  "format": "afterimage-receipt/0.1",
  "valid": true,
  "bundle": "sha256:...",
  "case": "CASCADE.001",
  "witness": "sha256:...",
  "branch": "sha256:...",
  "projection": "sha256:...",
  "trace": "sha256:...",
  "metrics": {
    "intervention_weight": 1,
    "causal_footprint": 17,
    "witness_units": 9,
    "effective_cost": 12345
  },
  "score": {
    "completion": 52,
    "optimization": 11,
    "total": 63,
    "nominal_max": 80
  },
  "unlocks": ["case:CASCADE.001"],
  "diagnostics": []
}
```

An invalid receipt has `valid: false`, no score or unlocks, and at least one
diagnostic:

```json
{
  "code": "contract_violation",
  "message": "service invariant failed",
  "context": {"at":123,"contract":"ambulance-arrival"}
}
```

`message` is human-facing and may be localized. `code` and `context` are
stable. A diagnostic MUST NOT reveal hidden test inputs that would make the
case a table lookup.

The receipt digest is:

```text
H("afterimage/receipt/1", canon(receipt_without_signature))
```

A live service MAY add a detached signature. Signatures are not required for
offline aggregation.

## 12. Aggregation

The aggregator never trusts receipt scores without either:

- replaying the associated witness; or
- verifying an accepted live-service signature tied to the exact bundle.

For each case, only the highest total score counts. Ties retain the
lexicographically better raw metric tuple; a remaining tie uses the smaller
witness digest for stable display only.

Unlock facts come from any valid witness, not only the highest-scoring one.

## 13. Compatibility

A witness names exact bundle and semantic versions. A verifier MUST NOT
silently migrate it.

If a case is corrected after release, it receives a new bundle revision and
an explicit migration notice. The scoreboard policy decides whether old and
new revisions share a case slot; the file format does not guess.

## 14. Security and robustness checklist

- Check archive limits before allocation or extraction.
- Reject duplicate normalized paths and duplicate JSON keys.
- Never extract outside a disposable directory.
- Do not execute bundle-native code.
- Run validators in CRE with declared deterministic limits.
- Bound diagnostic output.
- Treat all witness strings and meta fields as untrusted.
- Fuzz ZIP, JSON, NDJSON, event, intervention, and witness decoders.
- Maintain corpus fixtures for every previously fixed parser bug.
- Keep live authentication outside witness semantics.
