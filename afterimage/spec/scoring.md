# Afterimage Scoring Specification 0.1

Status: normative integer scoring design for the vertical slice.

## 1. Goals

Scoring must:

- grant meaningful credit for every valid solution;
- leave visible room for optimization after correctness;
- compare heterogeneous families without letting case count dominate;
- be reproducible offline with integer arithmetic;
- avoid hidden relative-to-leader formulas;
- preserve raw metrics when point rounding hides an improvement;
- keep story progression independent of optimization quality.

## 2. Portfolio budgets

The full nominal budget is 10,000 points:

| Family | Cases | Nominal points |
| --- | ---: | ---: |
| ORIENT | 5 | 300 |
| CASCADE | 24 | 2,400 |
| MERGE | 15 | 1,700 |
| PULSE | 14 | 1,700 |
| MOSAIC | 11 | 1,300 |
| LENS | 3 | 1,300 |
| COVENANT | 2 | 1,000 |
| PARADOX | 1 | 300 |
| **Total** | **75** | **10,000** |

These are design budgets, not a promise that every case in a family has equal
weight.

## 3. Case score

Each case declares:

- nominal maximum `P`, a positive integer;
- reference scale `K`, a positive integer;
- a family-specific positive effective cost `C` for a valid witness.

Invalid witnesses score zero and grant no unlock.

For a valid witness:

```text
completion = ceil(65 * P / 100)
pool       = P - completion
quality_ppm = floor(1_000_000 * K / (K + C))
optimization = floor(pool * quality_ppm / 1_000_000)
total = completion + optimization
```

All intermediate arithmetic MUST use checked unsigned integers at least 128
bits wide. The final values fit 64 bits.

Consequences:

- every valid witness earns at least 65% of nominal points;
- a witness with `C = K` earns half of the optimization pool before rounding;
- lower positive cost always weakly improves points;
- cost improvements remain in raw metrics even when total points tie;
- no finite cost reaches the nominal maximum, leaving theoretical headroom;
- zero cost is forbidden by construction to avoid division edge cases.

The nominal maximum is a normalization ceiling. A leaderboard may show both
integer score and raw metrics.

## 4. Lexicographic metric packing

Most families optimize several metrics in priority order. A case descriptor
provides inclusive maxima `M2 ... Mn` for lower-priority metrics and rejects a
validity candidate that exceeds them.

For non-negative metrics `(m1, m2, ..., mn)`, where smaller is better:

```text
C0 = m1
C1 = C0 * (M2 + 1) + m2
...
C  = C(n-1) * (Mn + 1) + mn
C  = C + 1
```

The final `+1` makes cost positive. Packing exactly preserves lexicographic
order while bounds hold. All multiplication is checked.

The bounds are part of the public score descriptor. They MUST be large enough
for reasonable baseline solutions and small enough to keep costs within the
declared integer limit.

## 5. Family metrics

### 5.1 ORIENT

ORIENT cases are completion-focused. Their metric tuple is:

```text
(wrong_or_redundant_claims, witness_units)
```

Validity normally forces the first component to zero. `witness_units` is
`ceil(canonical_answer_bytes / 32)` excluding outer-envelope metadata.

### 5.2 CASCADE

```text
(intervention_weight, causal_footprint, witness_units)
```

- `intervention_weight` is the sum of public operation weights.
- `causal_footprint` is the size of the symmetric difference between active
  event IDs in the parent and candidate fixed points, excluding explicitly
  declared diagnostic topics.
- `witness_units` is `ceil(canonical_answer_bytes / 64)`.

All case safety/service contracts must hold before metrics are considered.

### 5.3 MERGE

```text
(rejected_weight, temporal_displacement, certificate_units)
```

- `rejected_weight` sums public evidence weights of rejected records.
- `temporal_displacement` sums absolute displacement from each accepted
  record's preferred interval center, using the verifier-assigned schedule.
- `certificate_units` is `ceil(canonical_certificate_bytes / 64)`.

### 5.4 PULSE

```text
(program_bytes, worst_latency, live_state_cells)
```

- `program_bytes` is canonical encoded program size, excluding comments.
- `worst_latency` is the maximum output delay over the complete bounded test
  domain or certified model.
- `live_state_cells` is the maximum simultaneously reachable state storage.

Programs exceeding the case step, queue, or state limit are invalid.

### 5.5 MOSAIC

```text
(unexplained_weight, graph_size, certificate_units)
```

`unexplained_weight` is the weight of input fragments declared decoys or left
unplaced. The graph and all claimed correspondences must satisfy the case
contract. `graph_size` counts vertices plus edges after canonicalization.

### 5.6 LENS

```text
(program_nodes, auxiliary_schema_cells, worst_reductions)
```

All round-trip, stability, and provenance laws must pass first. Nodes are
counted in the compiled LENS AST. Auxiliary cells are fields retained solely
to preserve information not representable in one view.

### 5.7 COVENANT

```text
(policy_nodes, worst_response_bound, reachable_states)
```

The verifier explores every permitted schedule within the declared finite
model. A policy failing safety or bounded liveness is invalid.

### 5.8 PARADOX

```text
(paired_witness_units, latent_difference_weight, proof_steps)
```

`latent_difference_weight` rewards a focused countermodel: smaller is better,
but zero is invalid because the histories must materially differ.

## 6. Vertical-slice allocation

The twelve-case slice has a 1,200-point nominal budget:

| Case | Points |
| --- | ---: |
| ORIENT.001 | 40 |
| ORIENT.002 | 50 |
| ORIENT.003 | 60 |
| ORIENT.004 | 70 |
| ORIENT.005 | 80 |
| CASCADE.001 | 80 |
| CASCADE.002 | 100 |
| CASCADE.003 | 140 |
| MERGE.001 | 120 |
| PULSE.001 | 150 |
| MOSAIC.001 | 130 |
| LENS.001 | 180 |
| **Total** | **1,200** |

The slice allocation intentionally overrepresents onboarding. Full-production
budgets replace, rather than add to, these points.

## 7. Story and unlock scoring

Unlock logic sees only validity facts and explicit capabilities. It MUST NOT
read:

- total points;
- optimization points;
- raw cost;
- leaderboard rank;
- whether a solution used hints.

A lower-scoring valid witness grants the same case validity fact as the best
witness.

## 8. Team aggregation

For each case:

1. highest `total` wins;
2. on equal total, lexicographically smaller raw metric tuple wins;
3. on equal metrics, smaller canonical answer size wins;
4. on a complete tie, smaller witness digest provides stable ordering only.

Overall score is the sum of retained case totals. Family dashboards show
earned nominal score, valid cases, and raw bests separately.

## 9. Calibration requirements

Before release, every optimization case needs at least:

- a simple valid baseline with cost `C_baseline`;
- an intended competent solution with cost `C_competent`;
- a strong author solution with cost `C_author`;
- `K` chosen so the competent solution earns approximately 40–65% of the
  optimization pool;
- evidence that small meaningful improvements survive integer rounding often
  enough to remain motivating;
- a score curve plot or table over plausible costs;
- overflow tests for every packed metric boundary.

Family budgets must be simulated over pessimistic, median, and expert team
profiles. No single family should contribute more than 35% of the difference
between the simulated median and winner without an explicit design decision.

## 10. Examples

For a case with `P = 100`, `K = 1,000`, and valid `C = 1,000`:

```text
completion = 65
pool = 35
quality_ppm = 500000
optimization = 17
total = 82
```

For `C = 250`:

```text
quality_ppm = 800000
optimization = 28
total = 93
```

For `C = 4,000`:

```text
quality_ppm = 200000
optimization = 7
total = 72
```

These examples MUST become executable golden tests when scoring code is
implemented.
