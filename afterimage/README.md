# Afterimage: The Counterfactual City

`Afterimage` is a clean-room design for a programming-puzzle world with the
scope and structural ambition of ICFPC 2006, but with a different story,
computational substrate, puzzle vocabulary, and set of mechanics.

The player implements a small causal event/dataflow evaluator, uses it to
replay a record of tomorrow's city, and gradually discovers that the record is
not a forecast but a policy-enforced choice among several possible histories.

This directory contains the completed design gate and a release-gated,
cross-implemented CRE 0.1 semantic kernel. It deliberately does not yet
contain the full game or a bulk-generated problem set.

## Documents

- [Design bible](design_bible.md): product vision, player journey, narrative,
  problem families, progression, production gates, and non-goals.
- [Causal Reduction Engine](spec/causal_reduction_engine.md): normative value,
  event, rule, branch, evaluation, projection, and deterministic I/O semantics.
- [PULSE language](spec/pulse_language.md): bounded typed handlers, keyed
  timer replacement, microstep order, exhaustive domains, and resource limits.
- [MERGE certificate](spec/merge_certificate.md): interval reconstruction,
  exact active-order evidence, conflict sets, and public deduplication links.
- [MOSAIC certificate](spec/mosaic_certificate.md): D4 normalization,
  attributed fragment embeddings, full or bounded-completion coverage, and
  supported decoys.
- [LENS language](spec/lens_language.md): typed complement pipelines, finite
  round-trip laws, stable counterexamples, and reduction accounting.
- [COVENANT semantics](spec/covenant_language.md): local observation,
  exhaustive fair schedules, safety, bounded liveness, and stable
  counterexamples.
- [PARADOX certificate](spec/paradox_certificate.md): independently replayed
  paired histories, exact public equivalence, safety evidence, and focused
  latent differences.
- [Bundle and witness format](spec/bundle_and_witness.md): portable archive,
  case, intervention, witness, receipt, and verifier boundaries.
- [Localization contract](spec/localization.md): one immutable semantic
  bundle, external language packs, protected identifiers, and receipt
  invariance across English, Japanese, Simplified Chinese, and German.
- [Scoring](spec/scoring.md): exact integer scoring model and family metrics.
- [Vertical slice](vertical_slice.md): twelve cases, unlock graph, story beats,
  hint policy, playtest script, and acceptance criteria.
- [Full-scope manifest](manifests/full_scope.json): 75-case and 10,000-point
  budget.
- [Production catalog](manifests/production_catalog.json): all 75 case IDs,
  titles, exact points, acts, mechanics, prerequisites, waves, and live status.
- [Vertical-slice manifest](manifests/vertical_slice.json): machine-readable
  case IDs, prerequisites, points, and intended learning goals.
- [Production release manifest](manifests/production_release.json): the
  dependency-composed set of all currently authored cases.
- [Conformance suite](tests/conformance/suite.json): canonical values,
  recursion, stratification, aggregates, branching, and deterministic errors.
- [Conformance fixture contract](spec/conformance_fixture.md): the frozen full
  suite, localized public oracle, test-only labels, and suite/protocol split.
- [Python reference](reference/python/cre.py) and
  [JavaScript reference](reference/javascript/cre.mjs): independent CRE 0.1
  evaluators.
- [Conformance status](conformance_status.md): what is golden now and the exact
  frozen release matrix.
- [Production status](production_status.md): verified implementation coverage,
  current artifact identity, and the remaining vertical-slice gates.
- [Participant quickstart](playtest/PARTICIPANT_QUICKSTART.md): one-page offline
  workspace, submission, replay, reset, and telemetry workflow.
- [Blind-playtest operator protocol](playtest/OPERATOR_PROTOCOL.md): cohort
  separation, consent boundary, observation procedure, and release decision.
- [Observation contract](playtest/OBSERVATION_SCHEMA.md): anonymous exact-field
  campaign draft generation, complete missing-field diagnostics, conservative
  timing aggregation, and auditable pass/revise/stop decisions.
- [Six-agent adversarial simulation](playtest/AI_SIMULATION_2026-07-15.md):
  isolated two-wave implementation/game findings, fixes, and the explicit
  boundary between engineering simulation and human acceptance evidence.
- [AI proxy acceptance protocol](playtest/AI_PROXY_PROTOCOL.md): the explicit
  six-condition reasoning matrix, timing-estimation heuristic, sensitivity
  analysis, and labeling required by the user-authorized proxy gate.
- [AI proxy result](playtest/AI_PROXY_RESULT_2026-07-16.md): the bound
  six-condition evidence, engine anti-gaming check, estimated timing,
  sensitivity result, and production authorization.
- [Public release assets](public/README.md): brand system, generated key art,
  localized launch copy, social cards, press kit, static landing page, and
  reproducible spoiler-safe player, engine, and media packages.

## Current gate

The accepted implementation milestone was a twelve-case vertical slice. All twelve
cases, their private author witnesses, and their reproducible player bundle
are now golden for 1,200/1,200 nominal points. The slice covers identity,
replay, projection, branching, portable embedded claims, optimized causal
intervention, interval reconciliation, temporal programs, graph certificates,
bidirectional transformations, and a zero-intervention active-state proof.
The user-authorized six-condition AI proxy campaign passed the complete slice
criteria centrally on 2026-07-16. Full content production is now open under
that proxy policy; this is explicitly not a claim that a human campaign ran.

Full production is now complete. The complete 75-case/10,000-point catalog is
machine-checked, and `PULSE.002 First Copy` is the first post-slice case. The
current production release contains all 75 runnable cases worth 10,000 nominal
points. Its private baselines prove PULSE repeated-tick and cancelable-timeout
oracles, the MERGE survivor-linked deduplication contract, and MOSAIC
corroborated edge coverage. `CASCADE.004 Cold Platform` begins the local-failure
act by separating a sample's observation time from its delayed ingestion time;
the minimal repair and a costlier emergency-heat alternative are both replayed.
`CASCADE.005 Priority Queue` separates declared priority from non-preemptive
executor order, and `CASCADE.006 Black Ice` closes the act with a unique
treatment/closure rendezvous plus a costlier mobile-barrier solution.
`CASCADE.007 Borrowed Pressure` opens coupled infrastructure by resolving those
retained authorization IDs onto one feeder. `MERGE.003 Lost Acknowledgment`
forbids fabricating an omitted record, while `MOSAIC.003 Rotated Block`
distinguishes legitimate D4 canonicalization from suppressing real histories.
`PULSE.004 Token Window` turns the shared-feeder failure into a bounded
admission controller with exhaustive burst, refill, clamp, and rejection
semantics.
`CASCADE.008 Island Signal` distinguishes daily energy from instantaneous
microgrid capacity at an ambulance crossing, while `MERGE.004 Compensating
Transfer` carries the same incident into the audit layer and proves that an
applied transfer must be corrected by a later event rather than erased.
`CASCADE.009 Last Train Through` collapses six infrastructure windows to one
rail boundary, `MERGE.005 Split Brain` preserves complete competing writer
branches instead of permitting a splice, and `PULSE.005 Two of Three` turns
that audit lesson into a distinct-source one-shot quorum controller.
`CASCADE.010 Cooling Window` begins the hospital-cooling incident with the
first one-field replacement repair and proves that only flow 5 satisfies both
thermal and shared-capacity contracts. `MERGE.006 Quorum Ledger` then separates
a public three-vote winner from a temporally feasible two-vote minority, and
`MOSAIC.004 Missing Tile` recovers the blueprint's bounded central omission
from a public coordinate checksum. Together they make one physical fact appear
as intervention, authorization evidence, and spatial reconstruction.
`CASCADE.011 The Open Bridge` then turns wind and mechanical duration into one
safe command tick. `MERGE.007 Clock Islands` proves that the resulting records
belong to two shared-offset domains, not six independently adjustable clocks.
`MOSAIC.005 Double Exposure` removes a half-turned cached survey before it can
masquerade as independent corroboration, and `PULSE.006 Barrier at Dawn`
requires wind, bridge, and hospital readiness together. The four cases form
one bridge incident from physics through evidence hygiene to live control.
`CASCADE.012 Three Departments` extends that incident into a uniquely forced
three-agency handoff. `MERGE.008 Conflict Component` asks for an
inclusion-minimal explanation of the bad log pair, and `MOSAIC.006 False
Landmark` rejects a geometrically plausible survey only through retained field
evidence. `PULSE.007 Backpressure` makes the repaired handoff operational under
a two-slot queue. `LENS.002 One Timetable` closes the wave by publishing one UTC
view while lawfully preserving distinct operator services, platforms,
calendars, and provenance.
`CASCADE.013 Silent Gauge` starts the missing-observation act by requiring one
repair safe across an entire calibrated uncertainty envelope. `MERGE.009
Minimal Cut` preserves maximum evidence through an exact minimum-weight cut,
and `MOSAIC.007 Timestamp Gauge` turns retained scan ticks into absolute spatial
placement. `PULSE.008 Warm Failover` separates primary availability from
standby readiness. Those four artifacts become the inputs to `COVENANT.001
Dispatch Covenant`, the first full-production policy-synthesis case: two local
agents must meet safety and bounded response under every fair asynchronous
schedule.
`CASCADE.014 Shadow Bus` carries that covenant into a robust two-vehicle
allocation whose only safe integer split is bus 18 / train 52. `MERGE.010
Echoed Update` then proves that the four received dispatches form one rooted
direct-predecessor chain rather than independent commands. `MOSAIC.008 Broken
Ring` uses the published perimeter-cycle contract to recover exactly one
unobserved rail edge, and `PULSE.009 Exactly Once` keeps logical-operation
tombstones across primary failure and recovery. `LENS.003 Two Histories`
closes the wave by exposing that the direct and echoed internal histories were
collapsed into one public record; lawful corrections preserve their sealed
private delta and audit lineage.
`CASCADE.015 Redacted Feeder` follows that disclosure into a feeder incident
whose restricted provenance remains digest-bound while the whole active event
is absent from the public projection. `MERGE.011 Evidence Weight` resolves the
apparent trip cause by independent source weight rather than record count.
`MOSAIC.009 Adversarial Survey` then rejects an anonymous scan that fits the
geometry but loses every conflicting edge by the required support margin.
`PULSE.010 Shared Deadline` turns the surviving evidence into a controller
whose batch fires at the earliest absolute allowance across two input classes.
`CASCADE.016 Witness Gap` closes the act by proving that one injected latent
thermal-overload cause, and neither breaker fault nor manual opening, explains
all retained observations.
`CASCADE.017 Policy Blind Spot` then proves that the same public feeder row can
cover an unsafe underground directive and a safe surface bypass. `MERGE.012
Causal Compression` retains all five investigation records while reducing
seven precedence links to a necessary four-edge skeleton. `MOSAIC.010
Underground Layer` reconstructs connected surface and underground networks
meeting at exactly three shaft portals. `PULSE.011 Burst Budget` enforces a
half-open three-tick switching window over 792 streams. These four results
converge in `COVENANT.002 City Covenant`, where a local sensor must publish the
sealed heat state before Dispatch can safely choose a layer under every fair
schedule.
That covenant's first correction exposes a record attack. `MERGE.013
Equivocation` rejects both incompatible statements from one source, including
the statement that happens to agree with honest witnesses. `MOSAIC.011 Whole
City` joins three overlapping non-square strips into a covered two-layer city
map, while `PULSE.012 Reorder Buffer` drains every newly contiguous correction
fragment across 1,457 streams. `CASCADE.018 The Missing Cause` then requires a
two-event latent explanation whose empty and one-event subsets all fail under
independent replay. `CASCADE.019 Correction Notice` preserves the false prior
notice and publishes an explicit evidence-backed superseding route.
`MERGE.014 Two Consistent Archives` then preserves both feasible amendment
schedules while selecting one canonically. `PULSE.013 Circuit Breaker` protects
the hearing archive with closed/open/half-open semantics over 3,478 streams.
`CASCADE.020 Competing Amendments` uses a public tie-break without declaring
the losing amendment unsafe; `CASCADE.021 Audit Window` separates operational
cooldown from legal time; and `CASCADE.022 Rule of Least Change` proves that
one physical correction dominates a valid but heavier two-symptom repair.
`MERGE.015 The Reconstruction` preserves a genuinely partial citywide order
and all four feasible schedules. `PULSE.014 City Clock` turns the barrier into
a cyclic epoch controller over 27,064 exhaustive streams. `CASCADE.023
Continuity Hearing` binds archive, clock, and policy evidence to one epoch;
`CASCADE.024 The Chosen Tomorrow` applies the public selector without deleting
the unchosen safe route. `PARADOX.001 Two Tomorrows` closes the set by replaying
both complete histories, proving identical public records and independent
safety, and checking the latent route value rather than provenance churn.

Run the design consistency check with:

```bash
python3 afterimage/tools/check_design.py
```

Cross-check both independent evaluators and the frozen golden output with:

```bash
python3 afterimage/tools/run_conformance.py
```

For a small standalone case, inspect the complete normative trace with a hard
cap of 128 derived firings:

```bash
python3 afterimage/tools/trace_oracle.py CASE_OR_SUITE.json --name CASE --pretty
```

Build, validate, extract, and revalidate a portable world bundle with:

```bash
python3 afterimage/tools/afterimage_kit.py pack LOGICAL_DIR world.afterimage \
  --title "Afterimage" --revision slice-0.1.0
python3 afterimage/tools/afterimage_kit.py inspect world.afterimage --pretty
python3 afterimage/tools/afterimage_kit.py extract world.afterimage WORLD_DIR
python3 afterimage/tools/afterimage_kit.py verify-world WORLD_DIR
python3 afterimage/tools/afterimage_kit.py case WORLD_DIR \
  --case ORIENT.001 --projection desk.first-bell --output replay.case.json
```

Run every dependency-free design, CRE, and bundle-security gate with:

```bash
python3 afterimage/tools/check_all.py
```

Build the authored slice bundle and private author baselines with:

```bash
python3 afterimage/tools/build_slice.py afterimage/dist/afterimage-slice.afterimage \
  --author-dir afterimage/dist/author-baselines
python3 afterimage/tools/verify_witness.py WORLD_DIR witness.json --pretty
```

Build the current production release, which composes the frozen slice sources
with post-slice sources, with:

```bash
python3 afterimage/tools/build_slice.py afterimage/dist/afterimage-production-2.1-dev.afterimage \
  --manifest afterimage/manifests/production_release.json \
  --author-dir afterimage/dist/production-2.1-author-baselines \
  --title "Afterimage production release 2.1" \
  --revision production-dev-2.1.0
```

Create an isolated offline player workspace with human-readable commands or
canonical `--json` output:

```bash
python3 afterimage/tools/player.py init \
  afterimage/dist/afterimage-slice-dev.afterimage desk --telemetry
python3 afterimage/tools/player.py status desk
python3 afterimage/tools/player.py inspect desk ORIENT.001
python3 afterimage/tools/player.py verify desk witness.json
python3 afterimage/tools/player.py score desk
python3 afterimage/tools/player.py reset desk --keep-witnesses
```

Select localized case text, headings, hints, and human-readable labels without
changing the bundle or witness format:

```bash
python3 afterimage/tools/player.py --locale ja inspect desk ORIENT.001
python3 afterimage/tools/player.py --locale zh-Hans hint desk ORIENT.001 1
AFTERIMAGE_LOCALE=de python3 afterimage/tools/player.py status desk
```

The four packs live outside the `.afterimage` archive. English, Japanese,
Simplified Chinese, and German therefore share the same BundleId, schemas,
verification, scoring, unlocks, witnesses, and canonical receipts.

Build and verify the public launch assets and separated release packages:

```bash
bash afterimage/tools/build_public_assets.sh
python3 afterimage/tools/build_public_release.py afterimage/dist/public-release
python3 afterimage/tools/check_public_assets.py --smoke
```

The deployable four-language site is at
`afterimage/dist/public-release/web/site/index.html`. The public player kit
contains the exact verified production archive and excludes author witnesses,
private baselines, case sources, golden answers, and authoring tools.

Build three reproducible, physically separated blind-playtest packages. Both
participant kits exclude author witnesses, case sources, author golden data,
and authoring tools; the engine kit intentionally includes the public CRE
conformance oracle:

```bash
python3 afterimage/tools/prepare_playtest.py afterimage/dist/playtest-release --pretty
```

The engine kit contains specifications and public conformance data but no
reference evaluator. The game kit contains the frozen bundle and offline
runtime. The operator kit contains the protocol, interview, exact anonymous
observation contract, and its dependency-closed decision analyzer. Human
timing and comprehension criteria still require actual blind participants;
synthetic analyzer tests and automated author replay do not count as a
playtest.

Case sources live under
[`content/vertical_slice/cases`](content/vertical_slice/cases/); generated
author witnesses are explicitly excluded from the player bundle.

The substrate, bundle, and generic witness gates pass. Content production is
underway: ORIENT.001–005 are authored and golden, including unlock chaining,
causal replay auditing, the active-state/projection boundary, and root-versus-
counterfactual causal differences, plus independently reverified witness
exports. CASCADE.001 adds safety contracts and weighted intervention scoring;
MERGE.001 adds exact difference-constraint certificates, supported single and
conflict-set rejection reasons, and lexicographic reconstruction scoring.
PULSE.001 adds typed scalar state, replaceable keyed timers, deterministic
microsteps, complete 3,304-run checking, stable counterexamples, and temporal
program scoring.
PULSE.002 generalizes the same runtime to task-selected exhaustive oracles,
checks all 1,287 nondecreasing streams with repeated ticks, and requires
canonical-first payload preservation rather than a debounce parameter change.
MERGE.002 adds a public identity/body deduplication contract. A rejected retry
must name its accepted survivor; unrelated links and double survivors fail,
while retaining the weaker copy remains valid but scores worse.
PULSE.003 adds multiple external input topics and a true `cancel` action. Its
zero-cell program is checked over all 545 bounded command traces, including a
cancel exactly at the deadline, repeated starts, and stray cancels.
MOSAIC.002 adds public coverage multiplicity. Its two oppositely oriented
six-junction surveys must independently support both edges of their shared
corridor; union coverage alone is deliberately insufficient.
CASCADE.002 then composes MERGE/PULSE time semantics with a six-contract water
network, contrasting a cheap evidence-preserving time repair with a costly but
valid pump dispatch.
MOSAIC.001 adds D4 graph reconstruction with canonical orientation, complete
attributed-edge coverage, and exhaustive support checks for discarded
fragments.
LENS.001 adds exhaustive bidirectional laws over a bounded civic/route domain
and demonstrates why absent unit, provenance, and boundary information must
live in an explicit complement.
CASCADE.003 closes the slice by proving that a policy-suppressed relay remains
active without intervening on the world or disclosing its private payload. Its
scoped audit capability is valid but deliberately more expensive than the
zero-intervention proof, and non-scoped capability injection is rejected.
The post-slice substrate also implements non-root branch-history replay without
bundle-digest self-reference, exhaustive COVENANT policy checking, and PARADOX
paired-history certificates. The production release now exercises those
mechanisms through the complete 75-case finale. See `production_status.md` for
exact coverage and verification evidence.

## Clean-room boundary

Do not copy the Cult of the Bound Variable, UM/UMIX, its accounts, passwords,
characters, prose, languages, puzzle instances, publication strings, or
Adventure setting. The retained inspiration is limited to high-level product
structure: a participant-built substrate that becomes the world, several
problem families, local verification, optimization after correctness,
multi-path progression, and a finale that turns the substrate into the object
of investigation.
