# PARADOX paired-history certificate 0.1

Status: normative for `afterimage-paradox-contract/0.1`.

## 1. Claim

PARADOX proves that two materially different, evidence-supported histories can
produce the same published projection while both satisfy every published
safety requirement. The verifier independently replays both branch histories;
the submitted certificate never substitutes for replay.

The answer has exactly:

```json
{
  "left_history":{"format":"afterimage-branch-history/0.1"},
  "right_history":{"format":"afterimage-branch-history/0.1"},
  "equivalence":[{"left":0,"right":0,"digest":"sha256:..."}],
  "safety_evidence":{"left":["sha256:..."],"right":["sha256:..."]},
  "latent_difference":["sha256:..."]
}
```

Each history follows the branch-history rules in `bundle_and_witness.md`, uses
the PARADOX case's logical world, and is policy-checked step by step from root.
Every step case must already be unlocked. The two final BranchIds must differ.

## 2. Contract

A contract has exactly:

```json
{
  "format":"afterimage-paradox-contract/0.1",
  "safety_requirements":[
    {"id":"hospital-safe","topic":"contract.safe","pointer":"/payload/safe","equals":true,"minimum":1}
  ],
  "latent_topics":{"route.private-choice":2},
  "max_public_records":64
}
```

Safety lists are non-empty with unique IDs. For each branch, every active event
of the named topic must have the JSON Pointer value exactly equal to `equals`,
and at least `minimum` matching events must exist. `safety_evidence` is the
deduplicated raw-EventId-sorted union of those events for each side; omissions,
extras, and noncanonical order are invalid.

Latent topics form a non-empty map of positive weights. Material difference is
the raw-EventId-sorted symmetric difference of active events whose topic is in
that map. It must be non-empty and must exactly equal `latent_difference`.
Replacement generally contributes the removed and added event separately.

## 3. Public equivalence

The case's published projection is independently evaluated on both final
states. The complete record lists must be CRE-Value equal and within
`max_public_records`.

`equivalence` is a complete bijection between list indices. Each pair must
name equal records and include:

```text
H("afterimage/paradox-public-record/1", canon(record))
```

The explicit bijection prevents a verifier from accepting equality of an
aggregate digest while silently dropping duplicate records.

## 4. Metrics

After validity, the checker reports:

```text
(paired_witness_units, latent_difference_weight, proof_steps)
```

- `paired_witness_units` is the ceiling of the combined canonical history
  bytes divided by 64;
- `latent_difference_weight` is the sum of published topic weights over the
  exact symmetric difference and must be non-zero;
- `proof_steps` is equivalence pairs plus both safety evidence lists plus
  latent-difference IDs.

The host packs those metrics lexicographically in the listed order before the
ordinary 65% completion / 35% optimization formula. Invalid certificates use
distinct stable codes for same branch, public mismatch, safety failure,
missing material difference, certificate mismatch, and resource limits.
The 0.1 host caps public records at 4,096, safety requirements at 64, and
latent topics at 256; bundle declarations cannot raise those ceilings.
