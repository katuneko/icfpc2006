# MERGE 0.1 reconstruction and deduplication certificates

Status: normative for MERGE production cases.

MERGE reconstructs a causal archive from clocked records. Discovery may be
combinatorial, but verification is deterministic: classify every record,
assign every accepted record an integer global time, and supply the exact
active order certificate.

## 1. Records and intervals

Every `merge.record` payload contains non-empty `key` and `source` text plus
integer `local_time`, `offset_min`, `offset_max`, `sequence`, and `weight`.
Offsets satisfy `offset_min <= offset_max`; sequence and weight are
non-negative. Its permitted global interval is:

```text
[local_time - offset_max, local_time - offset_min]
```

Accepted assignments MUST lie in their intervals. Every accepted parent edge
and every adjacent accepted same-source sequence edge imposes a minimum gap of
one tick.

## 2. Answer and exact order certificate

An answer has exactly `accepted`, `rejected`, and `certificate`. Each record
is classified exactly once. Accepted entries contain `event` and `at`.
Ordinary rejected entries contain `event` and `reason`.

The certificate is a duplicate-free list of `before`, `after`, and
`minimum_gap` maps. It MUST equal—not merely contain—the set of active parent
and adjacent same-source constraints. This makes certificate size auditable
and prevents irrelevant edges from disguising a mistaken classification.

`inconsistent` is valid only when the rejected record has no feasible time
after propagating constraints from the submitted accepted schedule.
`conflict_set` requires at least two rejected records whose joint addition is
infeasible.

## 3. Public deduplication contract

A deduplication case publishes this exact contract shape:

```json
{"task":"deduplicate","identity_field":"dedup_key","equivalence_field":"body_key"}
```

The named fields MUST differ and every record MUST contain non-empty Text in
both. Records sharing the identity field are one logical-operation group.
When a group has multiple records, all equivalence-field values MUST match,
exactly one member MUST be accepted, and every other member MUST use:

```json
{"event":"sha256:...","reason":"duplicate","duplicate_of":"sha256:..."}
```

`duplicate_of` MUST name that group's accepted survivor. Thus a temporally
feasible retry cannot be mislabeled `inconsistent`, two copies cannot both
survive, and a record cannot justify suppressing an unrelated operation.
`duplicate` is invalid when the public score descriptor has no deduplication
contract.

Which equivalent member survives is a valid optimization choice. Because
`rejected_weight` is the first MERGE cost component, retaining stronger
evidence is preferred before schedule displacement or certificate size.

## 4. Metrics and diagnostics

Validity is decided before scoring. Valid answers report
`rejected_weight`, `temporal_displacement`, and `certificate_units` as defined
by the scoring specification. Stable failure classes include
`merge_classification`, `merge_interval`, `merge_order`, `merge_certificate`,
`merge_rejection`, and `merge_duplicate`; diagnostics do not reveal a complete
replacement answer.
