# Continuity Desk engine task

Implement the Causal Reduction Engine protocol described in
`spec/causal_reduction_engine.md`. For onboarding, expose these two commands:

```text
ENGINE suite FULL_SUITE_JSON
ENGINE protocol CASE_JSON
```

Both arguments are filesystem paths to canonical UTF-8 JSON files, not literal
JSON text on the command line.

`suite` emits one conformance-result object. A successful `protocol` case emits
`ready`, zero or more indexed `projection` records, then `done`; failure emits
one terminal `error`. These contracts are intentionally different.

The supplied `conformance/` material is public test data, not a reference
implementation. `suite.json` contains the hand-authored baseline; `matrix.py`
deterministically constructs the extended semantic matrix; `full-suite.json`
freezes that matrix; and `golden.json` gives the normative per-case oracle with
informative error prose removed. `expected.sha256` hashes that oracle. Run:

```sh
python3 conformance/check.py -- ./your-engine
```

The checker appends `suite FULL_SUITE_JSON` and reports the first differing
JSON Pointer. Fixture-only `@label` expansion is specified in
`spec/conformance_fixture.md`. Do not inspect or use another cohort's
implementation.

Record these milestones for the facilitator:

1. the time you first understood the required output;
2. the time of your first bounded evaluator run;
3. every ambiguity that required a documentation question;
4. the first conformance error you could not classify;
5. the time your engine was ready to enter the game session.

The engine must enforce declared resource bounds, reject noncanonical input,
remain deterministic across repeated runs, and work with networking disabled.
You may use any language. At least one cohort should use neither Python nor
Rust.

Stop after five hours even if the matrix is incomplete. Do not receive the
game kit or official runtime until the facilitator ends this session.
