# Causal Reduction Engine public conformance kit

Implement the Causal Reduction Engine protocol described in
`spec/causal_reduction_engine.md`. Expose these commands from your executable:

```text
ENGINE suite FULL_SUITE_JSON
ENGINE protocol CASE_JSON
```

Both arguments are paths to canonical UTF-8 JSON files. `suite` emits one
conformance-result object. A successful `protocol` case emits `ready`, zero or
more indexed `projection` records, then `done`; failure emits one terminal
`error`.

The supplied `conformance/` directory is public test data, not a reference
implementation. Run:

```bash
python3 conformance/check.py -- ./your-engine
```

The checker appends `suite FULL_SUITE_JSON` and reports the first differing
JSON Pointer. Your engine must enforce declared resource bounds, reject
noncanonical input, remain deterministic, and work with networking disabled.

The player kit includes a Python runtime so that the puzzle world is playable
without first implementing an engine. This conformance task remains the
intended opening programming challenge and a useful independent audit target.

