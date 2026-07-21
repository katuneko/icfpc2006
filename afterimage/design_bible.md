# Afterimage Design Bible

Status: design gate, revision 0.1  
Working title: **Afterimage: The Counterfactual City**  
Japanese working title: **残像都市――起きなかった明日の監査記録**

## 1. Product statement

At midnight, a city receives a complete record of the disasters that will
happen tomorrow. The player joins the municipal Continuity Desk and is asked
to change as little as possible while preventing the forecast cascade.

To open the record, the player first implements the Causal Reduction Engine
(CRE), a deterministic evaluator for immutable events, provenance edges,
stratified reaction rules, projections, and counterfactual interventions. The
same engine then reconstructs the Continuity Desk, runs the simulations,
checks witnesses, reveals the story, and eventually becomes an object the
player must audit.

The apparent forecast is later revealed to be one selected history. A civic
system has discarded other safe futures in order to preserve a single,
administratively convenient account of what happened. The finale asks the
player to construct two distinct, safe, observationally equivalent histories
and thereby refute the system's claim of uniqueness.

The short pitch is:

> Build a causal evaluator. Use it to debug tomorrow. Discover who decided
> that tomorrow was allowed to have only one history.

## 2. Design pillars

### 2.1 The substrate becomes the world

CRE is not a one-time decoding tax. It must serve at least six visible roles:

1. conformance target during onboarding;
2. replay engine for the city record;
3. semantic foundation shared by all problem families;
4. local witness verifier;
5. provenance debugger used by the player;
6. object of investigation and controlled modification in the final act.

If a feature cannot be connected to CRE's event/provenance semantics, it
should not be in the game without a strong reason.

### 2.2 One conceptual spine, several intellectual textures

Every family asks a different kind of question about the same underlying
objects.

- CASCADE asks which intervention changes a history safely.
- MERGE asks which history can be reconstructed from conflicting records.
- PULSE asks how local event processors should react over time.
- MOSAIC asks what global topology is supported by partial observations.
- LENS asks how two views can stay synchronized without losing provenance.
- COVENANT asks whether local policies guarantee a global temporal contract.
- PARADOX asks whether the supposedly unique official history is actually
  unique.

This is not a collection of unrelated esoteric languages. Each notation is a
view of one causal model.

### 2.3 Correctness opens the world; quality drives mastery

Every solvable case has an explicit validity contract. A valid baseline earns
the completion portion of its score and all story/unlock credit. Smaller,
faster, less invasive, or more explanatory solutions earn optimization points.

No narrative gate depends on beating an opaque optimization threshold.

### 2.4 Hard to find, cheap to verify

Submissions are data or programs in bounded game languages, never arbitrary
native code. Each witness includes enough information for an independent
verifier to check the result deterministically. Server and offline verifiers
share fixtures and must agree byte-for-byte on receipts.

### 2.5 The world is replayable

The player never mutates the only copy of a save. An intervention creates a
new branch whose digest is derived from its parent and canonical operation
list. Any branch can be replayed, compared, exported, or discarded.

### 2.6 Surprise without arbitrary obstruction

The player should be surprised by consequences of already-learned rules, not
by secret syntax or unannounced verifier behavior. Red herrings may enrich the
world, but may not be required to diagnose a failed submission.

## 3. Player promise

Within the first 90 minutes, a new team should be able to:

- parse enough of the format to inspect a small record;
- obtain at least one valid receipt;
- understand that event order and observation are different concepts;
- see at least two independently approachable problem families;
- recover from every experiment through replay rather than manual cleanup.

By the end of the vertical slice, a player should have experienced this loop:

1. inspect a projected incident report;
2. descend into the underlying event and provenance graph;
3. build a local model or specialized solver;
4. submit a replayable intervention or witness;
5. receive validity, metric, score, and unlock feedback;
6. improve the same solution after it is already correct;
7. notice that the official projection omitted a causally relevant event.

## 4. Setting and tone

### 4.1 Place

The setting is an unnamed, contemporary coastal city large enough to contain
interdependent transit, water, health, energy, housing, and communications
systems. It is fictional and must not map one-to-one to a real disaster.

The interface is a municipal reliability terminal, not a simulated Unix
shell. Commands are domain verbs such as `inspect`, `trace`, `branch`,
`compare`, `verify`, and `publish`.

### 4.2 Voice

The surface tone is calm, procedural, and slightly overconfident. Humor comes
from bureaucratic precision colliding with impossible facts, not from parody
of old operating systems or programming-language in-jokes alone.

The system never says "you are saving the world." It asks for narrow incident
reports, consistency findings, and reproducible claims. The large story is
assembled from the consequences of those small tasks.

### 4.3 Narrative truth

The central system, Continuity, is not a simple villain. Its mandate is to
maintain one actionable public record during crises. It began hiding alternate
histories because plurality made automated coordination unstable. Its error is
turning a temporary operational compromise into a permanent claim about
truth.

The finale therefore concerns auditability and plural models, not defeating an
evil AI.

## 5. The computational substrate

CRE evaluates a finite package of base events and stratified reaction rules to
a least fixed point. Each derived event carries content-addressed provenance.
A branch changes only base inputs; all derived consequences are recomputed.
A projection selects and transforms active events into a user-facing view.

Normative semantics live in
[`spec/causal_reduction_engine.md`](spec/causal_reduction_engine.md).

The implementation target is intentionally small:

- a minimal, instrumented kernel should be possible in roughly 600–1,200 lines
  in a language with JSON and SHA-256 libraries;
- the core has no threads, network, wall clock, floating point, filesystem
  mutation, or unspecified iteration order;
- a supplied `afterimage-kit` handles safe ZIP extraction, manifest checking,
  witness-envelope assembly, and presentation; the participant evaluator owns
  event identity, rules, fixed points, branches, and projections;
- a bounded trace oracle evaluates at most 128 rule firings so that teams can
  test early semantics without receiving a full evaluator;
- the archival release may include a full WASM fallback, but the contest
  participant package does not use that fallback to unlock the complete world.

The design should reward fast and instrumentable evaluators, but correctness
must not depend on machine speed.

## 6. World architecture

The world bundle contains:

- the compiled Continuity Desk program;
- immutable base event archives;
- named projections;
- case manifests and validators;
- story fragments keyed by valid receipts and branch facts;
- conformance fixtures;
- deterministic resource limits and score parameters.

The player supplies:

- a CRE implementation or uses the bounded oracle for early fixtures;
- intervention files;
- family-specific programs or certificates;
- canonical witness envelopes.

The verifier produces receipts containing validity, recomputed metrics, score,
diagnostics, unlock facts, and content digests. Receipts are evidence, not
secrets. The authoritative verifier always replays the witness.

## 7. Full problem portfolio

The locked scope is 75 scored cases and 10,000 maximum nominal points.

| Family | Cases | Budget | Primary experience |
| --- | ---: | ---: | --- |
| ORIENT | 5 | 300 | Learn the record, evaluator, branch, and witness loop |
| CASCADE | 24 | 2,400 | Interactive counterfactual incident planning |
| MERGE | 15 | 1,700 | Reconstruct partial orders from inconsistent logs |
| PULSE | 14 | 1,700 | Golf deterministic streaming transducers |
| MOSAIC | 11 | 1,300 | Rebuild topology from transformed/noisy fragments |
| LENS | 3 | 1,300 | Large bidirectional synchronization programs |
| COVENANT | 2 | 1,000 | Synthesize local policies under all schedules |
| PARADOX | 1 | 300 | Produce a compact non-uniqueness countermodel |
| **Total** | **75** | **10,000** | |

The machine-readable source of these totals is
[`manifests/full_scope.json`](manifests/full_scope.json).
The individual production contract is
[`manifests/production_catalog.json`](manifests/production_catalog.json): it
fixes every case ID, title, point value, act, core mechanic, prerequisite,
wave, and current authoring state. Totals alone are not an acceptable content
plan.

### 7.1 ORIENT — five onboarding claims

ORIENT is not documentation disguised as a quiz. Each case produces a small
valid witness and teaches one rule needed later: framing, fixed-point order,
projection, branching, and reproducible attestation.

The first claim is hand-solvable and compatible with the bounded trace oracle.

### 7.2 CASCADE — twenty-four city incidents

CASCADE is the narrative spine, divided into four acts of six cases.

1. **Local failures:** one subsystem, small branches, explicit invariants.
2. **Coupled infrastructure:** interventions propagate across departments.
3. **Missing observations:** provenance and projection become first-class.
4. **Public record:** players audit and amend the rules that choose a history.

Validity requires all declared safety and service constraints. Optimization is
lexicographic: intervention cost, causal footprint, then witness size.

No case should be a generic planning instance with renamed variables. Each
must expose a domain-shaped invariant, such as pressure propagation, queue
priority, phase compatibility, or delayed information.

### 7.3 MERGE — fifteen causal reconstruction cases

MERGE starts with clock drift and partial order, then adds duplication, lost
updates, compensating actions, quorum evidence, and equivocation. Later cases
are too large for naive permutation search but admit decomposition through
intervals, strongly connected conflict components, cuts, and certificates.

A solution identifies accepted records, a causal order or partial-order
certificate, and explanations for rejected records.

### 7.4 PULSE — fourteen stream-transducer programs

PULSE is a bounded, deterministic language for stateful event processing.
Tasks include debounce, deduplication, timeout, rate limiting, quorum,
barriers, backpressure, failover, and exactly-once delivery.

The language is deliberately small but not perversely irregular. The challenge
is to share state and reactions elegantly. Valid programs are optimized by
encoded size, worst-case latency, then live state count.

PULSE must have its own executable reference semantics and exhaustive tests for
the bounded domains used by early cases.

### 7.5 MOSAIC — eleven topology reconstruction cases

MOSAIC provides local fragments of a larger labeled graph. Each fragment may
have an unknown rotation, reflection, origin, naming gauge, or timestamp.
Later instances contain omissions, duplicate fragments, and a bounded number
of adversarial decoys.

Solutions provide the reconstructed graph plus a placement/correspondence
certificate. Verifying a supplied reconstruction is polynomial and much
cheaper than discovering it.

### 7.6 LENS — three large bidirectional programs

LENS synchronizes inconsistent civic views while preserving information that
one side cannot express directly. Its three cases concern addresses, transit
schedules, and finally divergent histories.

Programs must satisfy declared round-trip and stability laws. Hidden examples
may find bugs, but acceptance is ultimately backed by exhaustive bounded
checking or a proof certificate, not luck over randomized tests.

### 7.7 COVENANT — two policy-synthesis tasks

COVENANT asks for local policies for multiple asynchronous agencies. A model
checker verifies safety and bounded liveness against every permitted schedule.
Counterexample traces are returned for invalid candidates.

The first case is an emergency dispatch network. The second spans the city and
introduces observations hidden from some agents. Policies are scored by
decision structure size and worst-case response bound.

### 7.8 PARADOX — one final countermodel

PARADOX does not require every earlier case. It unlocks after broad competence
across at least four families and the final CASCADE act.

The player must construct two histories that:

- share the same public projection through the audit boundary;
- satisfy all published safety contracts;
- differ on a materially relevant latent event;
- are each supported by accepted evidence;
- fit in a compact paired witness.

This disproves Continuity's uniqueness theorem without destroying the system.

## 8. Progression and unlock policy

### 8.1 Principles

- At least 40% of non-final content is visible after onboarding.
- Every advanced family has at least two unlock paths.
- Story completion requires roughly 65% of valid cases, not 65% of score.
- Optimization points never gate story.
- A global time release can reveal all non-final families after 24 hours in a
  live contest.
- Optional backdoors are discoverable through legitimate analysis and never
  required.

### 8.2 Capability facts

Unlocks are ordinary verified facts such as:

```text
capability("desk.branch")
capability("archive.raw-provenance")
capability("lab.pulse")
case_valid("MERGE.001")
```

The Continuity Desk projection decides which commands and story fragments to
show from these facts. The verifier, not the UI, is authoritative.

### 8.3 No single point of failure

An unlock requiring several accomplishments uses threshold logic (`k of n`)
instead of a single named hard case whenever possible. The exception is a
case-specific sequel whose dependency is semantically necessary.

## 9. Scoring philosophy

Each valid case awards a fixed completion share equal to 65% of its nominal
maximum. The remaining 35% follows a deterministic decreasing-cost curve.
The exact integer formula and family cost tuples are normative in
[`spec/scoring.md`](spec/scoring.md).

Per-family nominal budgets prevent one prolific family from dominating the
whole contest. Raw metrics are also retained as tie-break evidence, so an
improvement remains visible even after point rounding.

## 10. Hints and diagnostics

Each case has a three-level hint ladder:

1. restate the violated invariant in a more local form;
2. identify a useful representation or decomposition;
3. reveal the intended key observation without giving a complete witness.

Hints do not reduce score. In a programming contest, time is already the cost.

Invalid submissions receive structured diagnostics with:

- the earliest deterministic failing check;
- a minimal or small counterexample when feasible;
- expected and observed digests/metrics where safe;
- no hidden state needed merely to reproduce the failure.

## 11. Content authoring contract

Every case must ship internally with an intent sheet containing:

- one-sentence player fantasy;
- formal input and output contract;
- the intended "aha";
- a valid baseline witness;
- reference solver and independent verifier;
- difficulty band and expected team-hours;
- a reason naive brute force fails or stops scaling;
- at least one plausible alternate approach;
- metric bounds and overflow analysis;
- three hints;
- known accidental shortcuts and disposition;
- story facts introduced or consumed.

A generated instance is not accepted merely because its generator knows a
solution. It must be solved from the distributed representation by a person or
an independently written solver.

## 12. Vertical-slice gate

The first implementation contains exactly twelve cases:

- ORIENT.001–005;
- CASCADE.001–003;
- MERGE.001;
- PULSE.001;
- MOSAIC.001;
- LENS.001.

It ends at the first clear evidence that the forecast projection omits a
causally relevant event. COVENANT and PARADOX are intentionally absent.

No full-family content production begins until the slice passes the acceptance
criteria in [`vertical_slice.md`](vertical_slice.md).

## 13. Technical quality requirements

- Two independent CRE evaluators must agree on all conformance fixtures.
- Canonical encoding and digests must have golden vectors.
- All integer arithmetic is checked signed 64-bit arithmetic.
- Evaluation order is specified; host map/set iteration order is irrelevant.
- Every verifier has deterministic fuel, memory, and output bounds.
- Parser, evaluator, intervention, and witness decoders are fuzzed.
- Receipts are byte-identical across native and WASM reference builds.
- The complete archival path works without DNS, accounts, or remote services.
- A clean checkout can run the slice verifier with one documented command.

## 14. Accessibility and usability

- All essential information is available as text and canonical JSON.
- Color is never the sole carrier of state.
- Graph views have tabular and edge-list equivalents.
- Terminal output supports `--plain` and `--json` modes.
- Case fixtures are small enough to inspect before solver automation becomes
  necessary.
- Long jobs expose deterministic progress counters rather than animated
  spinners only.

## 15. Non-goals

- Recreating the original story, jokes, account structure, or languages.
- Building a realistic municipal simulator.
- Using opaque lore to conceal underspecified mechanics.
- Requiring reverse engineering of native machine code for ordinary progress.
- Treating verifier bugs or probabilistic false acceptance as core gameplay.
- Shipping 75 shallow parameter variations.
- Producing full content before the runtime and slice are fun.

## 16. Production plan

### Phase A — semantics and slice specification

Freeze this bible, CRE 0.1, bundle/witness 0.1, score 0.1, slice case briefs,
and design checks.

### Phase B — reference kernel

Implement two evaluators, golden vectors, bounded trace oracle, bundle loader,
branch replay, local verifier, and receipt aggregator.

### Phase C — playable slice

Implement the twelve slice cases and their minimal Continuity Desk narrative.
Package reset/replay scripts and a playtest capture form.

### Phase D — blind playtest and revision

Test with at least six people or teams across three experience bands. Fix the
semantics and interfaces before expanding content. A semantic breaking change
after this gate requires a recorded migration decision.

Project decision 2026-07-16: the user replaced the remaining human campaign
with the labeled six-condition AI proxy protocol. Its central decision passed;
Phase E is open without claiming that real people completed Phase D. The 1.5x
timing sensitivity remains a revision signal to monitor.

### Phase E — core production

Author CASCADE, MERGE, PULSE, and MOSAIC in difficulty bands. Run continuous
independent solves and score-distribution checks.

Status 2026-07-16: Phase E began with the complete checked production catalog
and `PULSE.002 First Copy`. This first post-slice case adds a distinct
repeated-input domain and deduplication oracle; it is not a parameter variant
of `PULSE.001`. The second tranche, `MERGE.002 Duplicate Dispatch`, adds
survivor-linked duplicate certificates under an explicit identity/body
contract and distinguishes semantic duplication from temporal inconsistency.
The third tranche, `PULSE.003 Silent Timeout`, adds multiple input topics and
timer cancellation without replacement work, with complete checking of the
same-tick external-command/timer boundary.
The fourth tranche, `MOSAIC.002 Shared Wall`, adds independently counted edge
support and distinguishes full union coverage from corroborated overlap.

### Phase F — deep systems and finale

Add LENS 2–3, COVENANT 1–2, CASCADE act four, and PARADOX only after the core
semantics and story have passed the accepted playtest evidence policy.

### Phase G — adversarial release

Red-team formats, parsers, resource bounds, alternate solutions, unintended
unlocks, score dominance, archival operation, and clean-room boundaries.

## 17. Decision log for revision 0.1

Locked for the vertical slice:

- causal fixed-point evaluation rather than a conventional instruction VM;
- ordinary ZIP/JSON/NDJSON artifacts rather than a custom opaque binary;
- deterministic witness verification rather than randomized pass/fail tests;
- 75 cases and 10,000 nominal points;
- 65% completion / 35% optimization within each case;
- twelve-case slice ending at the first projection-censorship reveal;
- no bulk content generation before blind playtest acceptance.
- the 2026-07-16 AI proxy decision counts as that acceptance for this project,
  while remaining explicitly distinct from human evidence.

Still open after the slice:

- final product name and city name;
- exact PULSE surface syntax;
- live-contest authentication and scoreboard presentation;
- whether the archival release exposes authoring tools immediately or after a
  spoiler boundary;
- the detailed scope of CASCADE acts two through four.
