# CRE 0.1 conformance status

Release-kernel gate: **PASS**.

This is the live gate ledger for the semantic kernel. A checked item means the
Python and JavaScript references agree byte-for-byte and their normative result
is covered by the frozen public oracle. It does not mean the full game exists.

## Release matrix complete

- [x] canonical null, booleans, signed integers, i64 endpoints, text escapes,
  NFC text, nested map ordering, and bytes;
- [x] domain-separated SHA-256 IDs and exact 64-bit parsing in both hosts;
- [x] positive joins and positive recursion to a least fixed point;
- [x] absent-event negation and `count`, `sum`, `min`, and `max`, including
  empty `sum` and failing empty `min`;
- [x] arithmetic overflow and expression type failures;
- [x] branch `suppress`, `replace`, `retime`, and `inject`, including canonical
  operation ordering and suppression closure;
- [x] projection ordering/digests, trace digests, and resource counters on the
  baseline cases;
- [x] process exit codes 2 (input), 3 (semantic), and 4 (resource);
- [x] a local trace oracle capped at 128 derived firings.
- [x] missing parents, reversed parent time, same-time causal parents, and
  claimed-ID mismatch;
- [x] negation over a derived and sealed lower stratum;
- [x] every expression opcode and JSON Pointer escape/error boundary;
- [x] duplicate emission suppression through repeated fixed-point rounds and
  distinct-alias rejection;
- [x] `max_base_events`, `max_derived_events`, `max_bindings_tested`,
  `max_value_bytes`, and `max_projection_records` at boundary minus one,
  boundary, and boundary plus one;
- [x] missing player parents, operation conflicts, and invalid replacement
  targets;
- [x] canonical NDJSON `ready`, `projection`, `done`, and single terminal
  `error` records with process exit codes 2, 3, and 4;
- [x] hostile canonical JSON cases: duplicate/NFC-colliding keys, floats,
  invalid Unicode scalars, and non-canonical bytes.
- [x] an empty non-root operation step preserves its supplied parent BranchId
  in both references instead of incorrectly snapping back to root.

Current golden set: 9 canonical vectors, 37 evaluation cases (19 successes and
18 expected failures), seven hostile parser inputs, five successful NDJSON
records, terminal protocol failures, and 35 independent semantic-intent
assertions. Python and JavaScript must agree byte-for-byte before the golden
oracle digest `2cb50ac46b8415ddc4e195e238b7ad73a2f87de8ae106ef8a1006e0ea6bf55ec`
is checked. `full-suite.json` and `golden.json` are shipped with a reference-
free checker that reports the first differing JSON Pointer; informative error
messages are deliberately excluded while codes and tested contexts remain
normative.

## Scope notes

A valid cyclic base fixture cannot normally be constructed: each EventId hashes
its own parent-ID list, so a cycle would require solving a SHA-256 fixed point.
The topological cycle guard remains defense in depth; externally reachable DAG
failures are covered by missing-parent, claimed-ID, and time-order fixtures.

Intervention-policy rejection belongs to the witness verifier, not CRE branch
semantics. Likewise, mapping the tested NDJSON stream onto extracted
`WORLD_DIR` paths belongs to `afterimage-kit`. Both downstream gates now pass,
and ORIENT.001–005 exercise the frozen kernel. The one intentional post-slice
kernel correction is the cross-implemented non-root identity transition above.
