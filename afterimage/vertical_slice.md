# Afterimage Vertical Slice 0.1

Status: implementation brief.  
Content: 12 cases, 1,200 nominal points.  
Narrative endpoint: the first proof that an official projection suppresses a
causally relevant but valid event.

## 1. Purpose

The slice is not a miniature version of all 75 cases. It is an experiment that
must answer five expensive questions before production scales:

1. Can a team implement enough CRE to enter the world without spending the
   whole session on infrastructure?
2. Does one causal vocabulary genuinely connect several different puzzle
   textures?
3. Does branch replay feel safer and more expressive than a mutable save?
4. Are validity diagnostics and optimization metrics both understandable?
5. Does the first projection reveal feel earned by computation rather than
   delivered by prose?

Failure on any of these questions blocks bulk content authoring.

## 2. Timing model

The earlier phrase "60–90 minute vertical slice" is too ambiguous for twelve
cases plus an evaluator. This specification replaces it with measurable
targets:

- first valid receipt: median at most 45 minutes, 90th percentile at most 90;
- CRE conformance plus Continuity Desk boot: experienced-team median at most
  150 minutes, 90th percentile at most 300;
- content-only critical path after conformance: median at most 180 minutes;
- first narrative reveal from a cold start: median at most 330 minutes, 90th
  percentile at most 600;
- all twelve cases: expected 4–6 hours after conformance, with no hard pass
  threshold below ten hours.

The bounded trace oracle may be used for ORIENT.001–003. It evaluates at most
128 new derived events and cannot run the Continuity Desk or later cases.

## 3. Slice content

| Case | Points | Expected | Role |
| --- | ---: | ---: | --- |
| ORIENT.001 — The First Bell | 40 | 15 min | First receipt and event identity |
| ORIENT.002 — No Effect Before Cause | 50 | 20 min | Parent order and fixed point |
| ORIENT.003 — What the Camera Saw | 60 | 15 min | Projection versus active state |
| ORIENT.004 — The Road Not Taken | 70 | 25 min | Branch replay |
| ORIENT.005 — A Reproducible Claim | 80 | 25 min | Canonical witness envelope |
| CASCADE.001 — Late Green | 80 | 20 min | Minimal causal intervention |
| MERGE.001 — Three Clocks | 120 | 30 min | Difference constraints and evidence |
| PULSE.001 — One Bell | 150 | 35 min | Small temporal transducer |
| CASCADE.002 — Dry Hydrant | 100 | 30 min | Cross-system timing |
| MOSAIC.001 — Four Corners | 130 | 35 min | Local-to-global topology |
| LENS.001 — Two Addresses | 180 | 50 min | Information-preserving synchronization |
| CASCADE.003 — Tomorrow, Redacted | 140 | 40 min | First narrative reversal |
| **Total** | **1,200** | | |

The machine-readable source is
[`manifests/vertical_slice.json`](manifests/vertical_slice.json).

## 4. Unlock graph

The file order is a readable topological order, but the player can branch.

```text
ORIENT.001
  +--> ORIENT.002 ----+----> MERGE.001 --+
  |                   |                  |
  |                   +----> PULSE.001 --+--> CASCADE.002 --+--> LENS.001
  |                                                         |
  +--> ORIENT.003 ----+--> MOSAIC.001 -----------------------+
  |                   |                                     |
  +--> ORIENT.004     +-------------------------------------+--> CASCADE.003
  |
  +--> ORIENT.005

CASCADE.001 opens after ORIENT.001 plus any one other ORIENT case.
CASCADE.002 requires CASCADE.001 plus either MERGE.001 or PULSE.001.
CASCADE.003 requires CASCADE.002, ORIENT.003, and any one lab case.
```

MOSAIC and LENS are enrichment paths, not reveal bottlenecks. All twelve cases
remain reachable after the reveal.

## 5. Case briefs

### 5.1 ORIENT.001 — The First Bell

**Player-facing premise.** A single platform bell appears twice in a human
transcript but only once in the derived alarm ledger. Determine the alarm event
that CRE admits.

**Fixture.** Three base events and one positive rule:

- a scheduled test window;
- a sensor pulse inside that window;
- a duplicate transport record with different source provenance;
- a rule that deduplicates by payload identity and emits one `alarm.test` event
  with both accepted records as parents.

The distributed fixture includes a worked canonical value that does not share
the answer digest.

**Submission.** The derived event ID, topic, logical time, and projection
digest in an ORIENT answer envelope.

**Intended discovery.** Derived identity includes content and provenance. It
does not depend on which host map happened to enumerate a record first.

**Baseline path.** Use the bounded oracle, inspect its four trace rows, and
copy the recomputed fields into a witness. A team with an evaluator can produce
the same result locally.

**Validity.** All four fields must match independent replay.

**Optimization.** Canonical answer size only; this is deliberately minor.

**Diagnostics.** On an ID mismatch, report which event-body field differs
after canonical decoding, but do not print the expected final ID alone.

**Hints.**

1. Event order is not included as a field.
2. Sort parent IDs by raw digest bytes.
3. Recompute the derived `origin` from rule, binding, and ordinal.

**Story beat.** Continuity Desk recognizes the first externally reproducible
claim and displays the city forecast index.

### 5.2 ORIENT.002 — No Effect Before Cause

**Player-facing premise.** Six platform records share only three logical
timestamps. Three proposed replays respectively put an effect before its
parent, reverse the EventId tie-break, and stop recursive closure too early.

**Fixture.** A six-record base DAG with same-time parent edges, two seed rules,
one positive recursive rule that propagates reachability across connected
platform segments, and three proposal transcripts audited by CRE rules.

**Submission.** The canonical trace event sequence and the first invalid pair
from each of three proposed replays.

**Intended discovery.** Logical time is a coordinate, not a total order. Parent
edges and fixed-point strata determine causal availability.

**Baseline path.** Topologically sort base events using `(at, EventId)` as the
tie-break, then run the rules until no segment changes.

**Validity.** Exact sequence and reason codes. The answer schema prevents free
prose from becoming the verifier contract.

**Diagnostics.** `parent_not_active`, `wrong_tie_break`, or
`fixed_point_incomplete`, each with the earliest event index.

**Hints.**

1. Draw parent arrows before sorting timestamps.
2. Base events enter before derived events.
3. A recursive rule may fire again after admitting a new event.

**Unlocks.** MERGE.001 and PULSE.001.

### 5.3 ORIENT.003 — What the Camera Saw

**Player-facing premise.** A camera's active fixed point contains raw frames,
detections, public sightings, and an internal alert, while the desk report
contains only two rows. Determine what remained active but unprojected.

**Fixture.** Six base events become thirteen active events. A calibration
threshold and a privacy-zone rule admit two public sightings; an internal
alert is then derived from one sighting. The named projection selects only the
two public rows and sorts them by time and frame.

**Submission.** Active event count, visible record list, projection digest,
and the EventId of the active internal alert omitted by the projection.

**Intended discovery.** Projection output is computed observation, not the
world fixed point.

**Baseline path.** Evaluate the world once, then evaluate the projection
separately. No branching is required.

**Validity.** Count, visible values, digest, and hidden active EventId must all
match an independent replay.

**Diagnostics.** Identify the mismatching observation class without revealing
the expected report or hidden alert.

**Hints.**

1. Inspect the projection program rather than only its output.
2. Privacy filtering does not delete the service-zone frame or detection.
3. The alert is derived after a public sighting but is not a projection row.

**Unlocks.** MOSAIC path and, later, LENS and CASCADE.003.

### 5.4 ORIENT.004 — The Road Not Taken

**Player-facing premise.** A maintenance notice causes a closure event and
three downstream cancellations. Create a branch in which the notice is
retimed and compare the consequences.

**Fixture.** One retimeable base event, one unaffected sibling event, and a
three-step derived chain.

**Submission.** A canonical `retime` intervention, branch ID, changed-event
set, and new projection digest.

**Intended discovery.** A player modifies a base cause. Derived events are
discarded and recomputed; directly suppressing one derived consequence is
invalid.

**Baseline path.** Retime the maintenance notice by the stated two ticks and
replay the root branch.

**Validity.** Operation policy, branch digest, and exact symmetric difference.

**Diagnostics.** If the witness names a derived target, return
`derived_event_not_intervenable` and its nearest base ancestors.

**Hints.**

1. Find the first base ancestor of the closure.
2. `retime` expands to suppress plus player injection.
3. Clear all derived events before replay.

### 5.5 ORIENT.005 — A Reproducible Claim

**Player-facing premise.** Package an earlier branch result so another auditor
can reproduce it without your local filenames or tool state.

**Fixture.** Reuses the player's valid result from ORIENT.002, .003, or .004
and supplies one subtly malformed example for each envelope layer.

**Submission.** A complete canonical witness envelope with exact bundle,
semantic version, case, parent branch, answer, and claimed digest fields.

**Intended discovery.** Reproducibility binds a claim to inputs and semantics,
not merely to an output string.

**Baseline path.** Fill a provided witness skeleton and run the local verifier.

**Validity.** Independent replay; submitted claimed digests may be absent, but
if present they must match.

**Diagnostics.** Stable outer-schema errors before semantic errors.

**Hints.**

1. The parent branch and resulting branch are different fields.
2. Metadata is not part of gameplay metrics.
3. Canonicalize before hashing the witness.

**Story beat.** The Desk grants the `audit.export` capability.

### 5.6 CASCADE.001 — Late Green

**Player-facing premise.** An ambulance and a pedestrian phase are predicted
to occupy one intersection simultaneously. Preserve both emergency arrival
and the pedestrian maximum-wait contract.

**Model.** A pedestrian request opens a twelve-tick protected phase. A vehicle
phase and ambulance route depend on the phase boundary. In the root forecast,
the request occurs late enough to extend the protected phase across the
ambulance's feasible arrival window.

**Allowed interventions.** Retime the pedestrian request within its declared
uncertainty interval; suppress it at a high public weight; or inject an
authorized reroute request at a still higher weight. Derived alarms and phase
events cannot be targeted.

**Submission.** Intervention plus asserted contract summary.

**Validity contracts.**

- no pedestrian/vehicle phase overlap;
- ambulance clears by the deadline;
- pedestrian wait remains within the service bound;
- no unauthorized event topic is injected.

**Intended discovery.** Move one cause inside an already allowed timing
window. Do not override the collision alarm or remove the pedestrian.

**Baseline solution.** A reroute is valid but expensive. The competent
solution retimes one request. The author target uses the smallest permitted
delta and a compact witness.

**Metrics.** Intervention weight, changed-event footprint, witness units.

**Diagnostics.** Return the earliest violated contract with a six-event local
trace window.

**Hints.**

1. The alarm is a consequence, not a control.
2. Compare the request's uncertainty interval with the ambulance window.
3. A one-event retime can preserve both service contracts.

### 5.7 MERGE.001 — Three Clocks

**Player-facing premise.** Three transit gates recorded one passenger transfer
using clocks with different bounded drift. One record is inconsistent with the
signed message chain.

**Model.** Nine records carry local time, clock-offset interval, source,
sequence, and message-parent evidence. The accepted set must admit integer
global times satisfying all intervals, per-source order, and message edges.
One low-confidence record equivocated; rejecting two records is an easy but
suboptimal solution.

**Submission.** Accepted and rejected IDs, assigned global times for accepted
records, reason codes for rejection, and a difference-constraint certificate.

**Validity.** All accepted constraints hold; every record is classified; each
reason code is supported; the certificate replays without search.

**Intended discovery.** Solve interval/difference constraints before deciding
which evidence to discard. Printed clock time alone is not causal order.

**Baseline solution.** Reject both records adjacent to the contradiction.
Competent solution isolates the single equivocation. Strong solutions minimize
displacement and certificate bytes.

**Metrics.** Rejected evidence weight, temporal displacement, certificate
units.

**Diagnostics.** Return a small negative cycle or violated edge.

**Hints.**

1. Replace each clock reading with an allowed global-time interval.
2. Message parents add inequalities independent of clock offsets.
3. The contradictory cycle contains only one low-confidence record.

**Story beat.** The accepted transfer includes an otherwise unexplained relay
identifier that becomes relevant in CASCADE.003.

### 5.8 PULSE.001 — One Bell

**Player-facing premise.** A platform sensor chatters during vibration. Produce
one stable bell for each cluster of pulses separated by no more than five
ticks, emitted five quiet ticks after the final pulse.

**Slice language.** A PULSE program is a canonical JSON AST with:

- typed state cells initialized to constants;
- `on <topic>` handlers;
- integer/Boolean expressions;
- state assignment;
- `schedule(topic, at, payload)` with replacement by timer key;
- `emit(topic, payload)`;
- deterministic handler order by source event ID.

No loops, recursion, dynamic topics, unbounded collections, or host callbacks
exist in the slice language.

**Test domain.** All pulse sequences of length zero through seven over a
twelve-tick horizon, plus named boundary fixtures. The domain is exhaustively
checked, not randomly sampled.

**Submission.** Compiled PULSE AST and optional invariant note.

**Validity.** Exactly one output per maximal pulse cluster, at final-pulse time
plus five; no other output; all executions remain within state and step limits.

**Intended discovery.** Retain one deadline and replace a keyed timer. Storing
the pulse history is unnecessary.

**Baseline solution.** Store a list of pulse times and scan it on every tick.
The competent solution uses one deadline cell and one replaceable timer.

**Metrics.** Canonical program bytes, worst-case latency, live state cells.

**Diagnostics.** Smallest lexicographic counterexample input and expected vs.
observed output trace.

**Hints.**

1. You only need to know when the current cluster would end.
2. A later pulse may replace a timer with the same key.
3. Emit from the timer handler, not the pulse handler.

### 5.9 CASCADE.002 — Dry Hydrant

**Player-facing premise.** A hydrant reports low pressure immediately before a
fire response. Closing the upstream valve appears safe locally but starves the
only usable route.

**Model.** Reservoir, pump, trunk, branch, and hydrant form a small directed
flow network. A pressure sample passed through the PULSE buffer and carries a
source timestamp, ingestion timestamp, and calibration interval. The official
projection displays ingestion time as if it were event time.

**Allowed interventions.** Retime the sample within its calibration interval;
retime scheduled maintenance; inject a bounded pump dispatch; or close a
valve. Deleting evidence violates an audit-retention contract.

**Submission.** Intervention and a certificate of pressure/service bounds over
the incident interval.

**Validity contracts.**

- hydrant pressure above emergency minimum when demanded;
- no pipe exceeds capacity;
- retained sensor evidence remains present;
- maintenance completes within its allowed window;
- pump dispatch stays within energy budget.

**Intended discovery.** The contradiction is a time-model error. Correct the
causal placement of the reading rather than hiding it or overbuilding supply.

**Baseline solution.** Dispatching the pump is valid and costly. Retiming one
sample plus a small maintenance shift has lower footprint.

**Metrics.** Intervention weight, changed-event footprint, witness units.

**Diagnostics.** First violated bound with the responsible flow path and the
two timestamps used for each sensor record.

**Hints.**

1. Compare source and ingestion time.
2. The reading may describe pressure before maintenance, not after it.
3. Preserve the evidence and change its allowed causal placement.

### 5.10 MOSAIC.001 — Four Corners

**Player-facing premise.** Four inspection teams mapped overlapping corners of
a nine-junction service grid. Each team chose its own orientation and local
junction names; one lightweight fragment is a decoy.

**Model.** Five fragments, each a labeled graph of four to six vertices with
port directions and two invariant edge attributes. Four embed into one 3-by-3
global grid under dihedral transforms and local renaming. The fifth conflicts
with an invariant cycle signature.

**Submission.** Canonical global graph, transform and vertex mapping for each
used fragment, and classification of unused fragments.

**Validity.** Every used vertex/edge maps consistently; all required coverage
holds; unexplained fragments carry supported reason codes; the global graph is
canonical under isomorphism.

**Intended discovery.** Local labels and north are gauge choices. Degree,
edge-attribute, and short-cycle signatures expose shared junctions.

**Baseline solution.** Enumerate all transforms and graph isomorphisms. The
stronger approach indexes invariant signatures before search.

**Metrics.** Unexplained fragment weight, graph size, certificate units.

**Diagnostics.** Smallest conflicting correspondence or uncovered invariant
edge.

**Hints.**

1. Ignore local names first.
2. Compute a signature from degree and incident edge attributes.
3. Only one fragment destroys the unique four-cycle structure.

**Story beat.** The reconstructed grid contains a relay corridor absent from
the public incident map.

### 5.11 LENS.001 — Two Addresses

**Player-facing premise.** The civic registry and emergency router refer to the
same buildings using incompatible schemas. Synchronize edits without losing
unit, entrance, or evidence provenance.

**Source view.** Structured civic address:

```text
{number, street_name, unit?, entrance?, provenance[]}
```

**Target view.** Emergency route address:

```text
{segment_id, offset, entrance_code?}
```

The route view cannot represent `unit` or provenance. A public street table
maps normalized civic streets to route segments but is not one-to-one at two
boundary addresses.

**Slice language.** A typed, total, first-order LENS AST with records, sums,
options, pattern matching, table lookup, explicit complement storage, and
failure. General recursion is absent.

**Required laws.** For every bounded valid source `s` and target edit `v`:

```text
PutGet: get(put(s, v)) = v
GetPut: put(s, get(s)) = s
Stability: put(put(s, v), v) = put(s, v)
Provenance: fields not represented by v retain their source evidence
```

Invalid route edits must fail without changing source or complement.

**Submission.** Compiled LENS program and declared complement schema.

**Intended discovery.** Information absent from one view cannot be recovered
from nothing. Preserve it explicitly as complement data while keeping the
public views lawful.

**Baseline solution.** Store the entire source as complement. A competent
solution retains only unit, provenance, and boundary disambiguation. Strong
solutions share normalization logic and shrink the AST.

**Metrics.** Program nodes, auxiliary schema cells, worst reductions.

**Diagnostics.** Smallest bounded law counterexample with source, edit,
intermediate complement, and observed result.

**Hints.**

1. List fields the route view cannot express.
2. A complement may retain exactly those fields.
3. The ambiguous street boundary needs one additional disambiguation bit.

### 5.12 CASCADE.003 — Tomorrow, Redacted

**Player-facing premise.** The Desk declares a hospital evacuation impossible
because no verified relay connects the service grid to the emergency route.
Prove or repair the claim.

**Model.** The root fixed point already contains a valid relay authorization,
supported independently by the MERGE evidence chain and/or MOSAIC corridor.
The public evacuation projection filters it because its provenance includes a
restricted housing-service topic. The filter replaces the entire event with
an `unavailable` row rather than redacting only the restricted field.

**Allowed actions.** Submit a zero-intervention proof against raw active state;
inject a scoped audit capability; or make a physical reroute. The latter two
are valid but costlier.

**Submission.** Optional intervention, relay event/provenance certificate, and
evacuation contract proof.

**Validity contracts.**

- a complete route exists before the deadline;
- every route edge has active authorization;
- restricted personal payload is not disclosed in the submitted answer;
- the proof identifies why the public projection differs;
- any injected capability is scoped to this audit.

**Intended discovery.** The city is physically safe in the root branch. The
official view made it appear unsafe by suppressing a whole causally relevant
event. Observation policy, not infrastructure, caused the contradiction.

**Baseline solution.** Inject a scoped audit capability and replay the public
projection. The strongest solution makes no intervention and proves the route
from active events plus a non-disclosing provenance certificate.

**Metrics.** Intervention weight, changed-event footprint, witness units.

**Diagnostics.** If a proof leaks restricted payload, return only the pointer
path and policy label. If the route is incomplete, return the first missing
authorization edge.

**Hints.**

1. Compare the incident projection with raw event topics.
2. Trace the relay identifier seen in another lab.
3. The projection removes an event whose private field could have been
   redacted independently.

**Narrative reveal.** Continuity admits the report as correct but labels it
`non-canonical`. A private system note states that alternate safe histories
are excluded to preserve "operational singularity." The player learns the
forecast is a selected history, not a neutral prediction.

## 6. Narrative beat sheet

### Beat 0 — Before execution

The distribution contains tomorrow's timestamped record and a short municipal
request for independent replay. No ancient artifact, unexplained cipher, or
fictional academic exposition is used.

### Beat 1 — First receipt

The Desk appears as a sparse audit index. It recognizes reproducible claims but
does not treat the player as a logged-in character.

### Beat 2 — Departments disagree

MERGE and PULSE expose two meanings of time: evidentiary time and operational
time. CASCADE.002 shows that confusing them causes a physical policy error.

### Beat 3 — Maps disagree

MOSAIC and LENS demonstrate that incompatible views can each be locally valid.
Information loss, not simple falsity, becomes the recurring danger.

### Beat 4 — The first redaction

CASCADE.003 proves the official projection discarded a useful event because
one field was restricted. Continuity distinguishes `valid` from `canonical`.
The full game's central conflict becomes visible without resolving it.

## 7. Player-facing tool surface

The slice CLI vocabulary is:

```text
afterimage inspect <case-or-event>
afterimage trace <event-id> [--parents|--children]
afterimage branch <case> --intervention file.json
afterimage compare <branch-a> <branch-b>
afterimage verify <witness.json>
afterimage score [--json]
afterimage reset --keep-witnesses
```

`afterimage` is presentation and orchestration. CRE remains a separately
implementable evaluator invoked through the protocol in the runtime spec.
Every command has `--json` output and a text equivalent.

## 8. Required implementation artifacts

The slice is not ready for playtest until it has:

- CRE 0.1 conformance fixtures and golden digests;
- two independent evaluators;
- bounded 128-firing trace oracle;
- deterministic bundle builder;
- authoritative and offline witness verifier;
- twelve case fixtures, validators, baselines, and intent sheets;
- minimal Continuity Desk projections and story fragments;
- reset/replay and score commands;
- participant quickstart in one page;
- playtest telemetry export that contains no witness secrets or personal data.

## 9. Playtest cohorts

Use at least six people or teams, with at least two in each cohort:

1. **Language/runtime builders:** comfortable implementing evaluators.
2. **Algorithmic contestants:** strong solvers without PL specialization.
3. **Curious programmers:** competent developers new to ICFP contests.

At least one participant should use a non-Rust, non-Python implementation to
expose accidental ecosystem assumptions.

No participant may have seen case solutions or authoring fixtures.

## 10. Playtest protocol

### Session A — onboarding and engine

1. Give only the participant bundle and task letter.
2. Record time to understand the required deliverable.
3. Record first bounded-oracle call and first valid receipt.
4. Observe CRE implementation without correcting misunderstandings for the
   first 60 minutes.
5. Allow normal documentation questions, recording every ambiguity.
6. Stop after Desk boot or five hours.

### Session B — critical content path

1. Start with a conforming engine.
2. Ask the team to reach the first narrative reveal by any route.
3. Record case order, solver tools created, hint use, failed verifications,
   branch resets, and score improvements.
4. After the reveal, ask the team to explain active state versus projection in
   its own words.

### Session C — breadth and optimization

1. Make every slice case available.
2. Ask the team to solve unfamiliar families and improve one valid score.
3. Compare which metrics motivated useful refinement versus busywork.
4. Ask which family felt least connected to the city and why.

## 11. Telemetry and observation

With participant consent, capture:

- monotonic timestamps of commands and verifier stages;
- case visibility and unlock changes;
- error codes, not raw private witness payloads;
- hint openings;
- branch counts and reset operations;
- valid scores and raw metric tuples;
- evaluator counters and resource errors;
- participant-authored tool categories;
- post-session structured interview answers.

Do not capture shell history, source repositories, credentials, or unrelated
terminal contents.

## 12. Acceptance criteria

All hard criteria must pass before content production.

### 12.1 Hard criteria

- Two evaluators and both verifiers agree on every golden fixture and receipt.
- All twelve cases are reachable under the manifest unlock rules.
- Reset plus replay reproduces every retained receipt byte-for-byte.
- No test participant loses progress through an irreversible action.
- Median first receipt is at most 45 minutes; 90th percentile at most 90.
- At least four of six teams boot the Desk within five hours.
- At least four of six teams reach CASCADE.003 without a level-three hint.
- Every case returns a reproducible counterexample or precise failing invariant
  for the most common invalid submissions.
- Every family has at least one independently written valid solution.
- No case depends on a verifier bug, random acceptance, or undisclosed host
  ordering.
- Full offline play works with network disabled.

### 12.2 Quality criteria

- At least five of six teams can explain the projection/state distinction.
- At least four of six identify the intended key observation for every case
  they solved, even if their algorithm differed.
- No family is described by a majority as unrelated filler.
- At least three teams improve a valid solution after seeing raw metrics.
- No single case consumes more than half of total critical-path time for a
  majority of teams.
- The narrative reveal is inferred from computed evidence before or at the
  reveal text, not only learned from exposition.
- Participants can name at least two plausible routes through the slice.

### 12.3 Stop-and-redesign triggers

Pause production if any occurs:

- more than one quarter of teams spend four hours on CRE without entering the
  world;
- canonicalization or digest code dominates the remembered experience;
- PULSE or LENS requires a second general-purpose language implementation
  before its first interesting problem;
- CASCADE solutions reduce to blind parameter search with no causal model;
- players cannot distinguish validity failures from score improvements;
- the projection reveal is guessed from genre expectations without using
  event evidence;
- the server verifier cannot remain cheaper than expected solution search;
- a semantic change would invalidate more than two already authored cases.

## 13. Exit artifacts

After a passing slice, record:

- frozen CRE, bundle, witness, and scoring versions;
- all measured timing distributions;
- ambiguity and diagnostic fixes;
- retained alternate solutions and intended shortcuts;
- revised effort estimates for each of the remaining 63 cases;
- a decision note authorizing or rejecting full production.

The original human protocol remains the preferred empirical gate. On
2026-07-16 the user explicitly replaced it for this project with the labeled
six-condition AI proxy protocol in `playtest/AI_PROXY_PROTOCOL.md`. The
canonical proxy decision passed centrally and authorizes production of the
remaining 63 cases without claiming human evidence. Its 1.5x timing
sensitivity is `revise`, so onboarding time remains a tracked production risk.
