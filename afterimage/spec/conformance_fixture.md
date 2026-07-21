# CRE 0.1 public conformance fixture and oracle

This document specifies the test-only adapter used by the public CRE suite.
Fixture labels are not CRE values and never occur in a world bundle.

## Frozen files

- `suite.json` is the readable hand-authored baseline.
- `matrix.py` deterministically extends it to 37 cases.
- `full-suite.json` is the canonical, frozen result of that extension.
- `golden.json` contains the exact normative result for all nine canonical
  vectors and 37 cases.
- `expected.sha256` is the raw SHA-256 of `golden.json`.
- `check.py` invokes a participant engine and reports the first differing JSON
  Pointer instead of exposing only an aggregate mismatch.

The oracle deliberately excludes error `message` prose because CRE 0.1 marks
it informative. Error `code` and `context` remain exact and normative.

Canonical vectors use digest domain `afterimage/test-vector/1` unless the
vector explicitly contains a different `domain` field.

## Suite command

A participant executable used with the public checker MUST support:

```text
ENGINE suite FULL_SUITE_JSON
```

It emits exactly one `afterimage-conformance-result/0.1` JSON object on stdout
and exits zero. The object has exact fields `format`, `canonical_vectors`, and
`cases`. Each case entry is exactly `{ok}` or `{name,error}`. An error is
exactly `{code,context}` or `{code,message,context}`; the checker ignores only
`message`. A final LF after the single JSON object is permitted. Successful
suite execution writes nothing to stderr.

Run the localized check as:

```sh
python3 conformance/check.py -- ./my-engine
```

The checker appends `suite` and the absolute `full-suite.json` path.

## Test-only labels

A base-event fixture is an ordinary event body, an unlabeled wrapper exactly
`{"body": EventBody}`, or a labeled wrapper exactly:

```text
{ "label": Text, "body": EventBody }
```

The unlabeled wrapper exists so the readable baseline can keep one uniform
fixture shape; unwrap it without adding anything to the label map. If a
wrapped body contains an optional claimed `id`, validate it after resolving
its parents exactly as for an ordinary event body.

Starting with an empty label map, repeatedly scan pending base entries in
their original order. An entry is ready when every parent string beginning
with `@` names an existing label. Replace those parent references with their
EventIds, validate and identify the event, append it to the resolved base list,
then bind its label. Preserve ready-entry scan order. Duplicate labels are
`invalid_fixture`. If a complete scan makes no progress, fail with
`invalid_fixture` because labels are cyclic or absent.

After all base events resolve, replace `@label` in the `event` and `parents`
fields of branch operations. An absent operation label is `invalid_fixture`.
No other string or field undergoes label substitution.

## Protocol distinction

The suite command above is a batch conformance adapter. It is distinct from
`ENGINE protocol CASE_JSON`: a successful protocol case emits `ready`, zero or
more indexed `projection` records, then `done`, exactly as CRE §10 specifies.
Protocol failure emits one terminal `error` record. There is no one-result-line
protocol variant in CRE 0.1.
