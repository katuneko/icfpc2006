# Afterimage production status

This is the live, evidence-backed production ledger. Design totals describe
the intended product; only items under **Verified now** exist as runnable
content.

## Verified now

- Product design: 75 cases and 10,000 nominal points remain internally
  consistent.
- Vertical-slice design: 12 cases and 1,200 nominal points; all cases are
  reachable and the reveal has two minimal paths.
- CRE 0.1: Python and JavaScript agree on 37 cases, seven hostile canonical
  inputs, 35 semantic assertions, NDJSON behavior, and public-oracle digest
  `2cb50ac46b8415ddc4e195e238b7ad73a2f87de8ae106ef8a1006e0ea6bf55ec`.
  Three focused tests prove both references pass the shipped checker, error
  prose is non-normative, and a bad engine receives its first JSON-Pointer
  mismatch instead of only an opaque aggregate digest.
- Bundle security: 14 tests cover reproducibility, path and link attacks,
  canonical files, resource limits, corruption, and extracted-world drift.
- Localization: external `en`, `ja`, `zh-Hans`, and `de` packs cover all 75
  cases and all 225 ordered hints without entering the authoritative archive.
  Strict loading checks BundleId support, exact production coverage, UI
  placeholders, and verbatim semantic tokens. Invariance tests hold the
  production archive SHA-256 and BundleId fixed and reproduce one canonical
  witness receipt byte-for-byte under every locale.
- Witness verifier: nine focused tests plus the content suite cover schema,
  policy, branch, replay, claimed digests, ORIENT/CASCADE/MERGE/PULSE scoring,
  unlocks, non-leaking diagnostics, and policy-checked non-root histories that
  avoid bundle/BranchId self-reference.
- PULSE runtime: focused and production-wave tests cover the 3,304-case debounce domain,
  the 1,287-case repeated-tick deduplication domain, same-tick EventId order,
  the 545-case cancelable-timeout domain, the 924-case token-bucket domain,
  the 3,478-case two-of-three quorum domain, and the separate 3,478-case
  all-topic barrier domain, the 10,417-case exactly-once failover domain, and
  the 1,471-case shared-deadline domain, the 792-case sliding-window domain,
  and the 27,064-case cyclic City Clock domain.
  They cover multiple input topics,
  keyed timer replacement and cancellation, deadline-tick external priority,
  canonical-first payload retention, refill-before-admission, capacity clamping,
  rejected-work non-consumption, distinct-source voting, all-topic rendezvous,
  decisive-event payload retention, one-shot release, stable minimal counterexamples, static types,
  bounded resource failure, and standalone helper discovery.
- MOSAIC verifier: twenty-one focused tests cover D4 canonicalization, complete
  attributed-grid coverage, independent shared-edge multiplicity, the
  difference between union and corroborated coverage, rejection of
  unsupported decoy claims, and bounded interior completion under a public
  coordinate checksum, D4/translation-invariant attributed duplicate
  fingerprints, timestamp-gauged placement, one-edge ring recovery, and
  weighted adversarial-noise rejection, and exact connected layer portals.
  Wrong inferred labels, unreported perimeter gaps, double-used
  observations, and false survivor links are rejected independently.
- LENS checker: nine focused tests cover the original 72-by-15 address domain,
  the 96-by-9 timetable domain, and the 48-by-9 divergent-history domain,
  including identity collisions, complement preservation, invalid-edit
  atomicity, and stable law counterexamples.
- COVENANT checker: ten focused tests exhaust every fair schedule in a finite
  asynchronous model, enforce local observation, safety, bounded liveness,
  finite domains and hard resource ceilings, and exercise authoritative host
  scoring. The original reference policy reaches 32 augmented states; the
  production dispatch covenant reaches 80 augmented states with worst response
  bound 5 from both initial availability states. City Covenant adds two hidden
  heat worlds, a local sensor-to-dispatch information channel, 38 reachable
  augmented states, and exact worst response bound 4.
- PARADOX checker: six focused tests and a complete authoritative-verifier
  fixture independently replay two policy-checked histories, prove exact
  public-record equivalence and safety evidence, require a non-zero latent
  semantic latent-value difference rather than provenance-only churn, and
  score all three certificate metrics.
- Authored content: ORIENT.001–005, CASCADE.001–024, MERGE.001–015, PULSE.001–014,
  MOSAIC.001–011, LENS.001–003, COVENANT.001–002, and PARADOX.001, totaling
  10,000 points. The frozen slice suite plus
  the production-release golden test
  cover deterministic builds, author baselines, case-world isolation, label
  resolution, causal replay auditing, projection semantics, root/candidate
  differences, derived-target diagnostics, independently verified exports,
  capability grants, exact order certificates, supported rejection reasons,
  scoped capability enforcement, private-payload non-disclosure, and hostile
  answer mutations. The 75-case production bundle and all private
  baselines rebuild reproducibly and verify from the root unlock chain.
- Player workspace: nine focused tests cover canonicalization of editor JSON,
  raw-cost tie breaking, initial visibility, monotone
  unlocks, all twelve author receipts, 1,121/1,200 aggregate author score,
  case/event inspection, root branch snapshots, comparisons, hints,
  destructive reset, byte-for-byte replay from retained witnesses, and
  rejection of support-directory symlinks before any write. Branch commands
  can carry a verified parent history by BranchId or canonical history file;
  locale selection is available by CLI flag or environment variable.
- Playtest release: reproducible engine, game, and operator zip packages are
  physically separated. The engine package has no reference evaluator; both
  participant packages exclude author witnesses, case sources, author golden
  data, and authoring tools. The engine package intentionally contains only
  the public CRE conformance oracle. The staged game package boots with an empty `PATH` and
  networking is not required. The operator package includes a dependency-
  closed campaign analyzer and exact observation contract.
- Public launch assets: one checked brand system, two generated source-art
  masters with recorded prompts/provenance, five editorial derivatives, three
  raster logo derivatives, twelve localized social cards, four browser-rendered
  locale previews, one A4 one-sheet PDF, four-language static site, launch copy,
  fact sheet, FAQ, alt text, spoiler policy, and publishing checklist are under
  `public/`. The public player, CRE engine, and media ZIPs rebuild byte-for-byte;
  the player boots with an empty `PATH`, displays Japanese ORIENT.001, and
  contains the exact production archive but no author witnesses, baselines,
  case sources, private golden data, or authoring tools. Current package
  SHA-256 values are player `16e5be04b15367e78a55988e10e9d9178ea59d3dcc724aa51a9c7cf87780e4a3`,
  engine `f67bf08a8015062b63a900de975c2d2619450524c7f5e961cb8e196369d52b6f`,
  and media `910c45388060c4009d2ec300f6aa5ab4e9352cc01cb9cd2dfd2a68aa8098b0bf`.
- Playtest decision gate: seven focused tests cover private draft generation,
  random anonymous codes, balanced cohorts, complete missing-field reporting,
  a passing six-team campaign, exact half-minute median and nearest-rank P90,
  censored receipt timing, ratio scaling above six teams, all eight stop-and-
  redesign triggers, strict controlled fields, canonicalization, and immutable
  output. This validates the decision mechanism only; it supplies no human
  evidence.
- Telemetry: disabled by default and consent-gated. Exported events have
  monotone ticks/timestamps and allow only command class, public error code,
  hint level, counts, scores, metrics, and unlocks; tests exclude answer,
  intervention, parent-branch, event-ID, and submission bytes.
- Adversarial simulation: six isolated AI agents in two fresh waves exercised
  independent CRE implementation, localized conformance, offline progression,
  score improvement, and receipt replay. The simulation drove fixes for the
  public oracle and fixture contract, editor JSON, helper launch, raw-cost
  tie-breaking, trace rendering, and complete branch-trace access. This is
  engineering evidence only and supplies no human acceptance or timing data;
  see `playtest/AI_SIMULATION_2026-07-15.md`.
- AI proxy acceptance: six new isolated low/medium/high behavioral-proxy
  conditions produced a canonical `proxy-pass`. All six solved CASCADE.003
  without hints, explained the projection boundary, improved valid metrics,
  and replayed losslessly; independent solutions cover all six slice families.
  Estimated first receipt is median 40 minutes and P90 50. The 1.5x timing
  sensitivity is `revise` because its median becomes 60, so timing remains a
  documented risk. Three independently implemented proxy engines passed a
  private renamed-fixture generalization check; two public-oracle adapters
  were detected and excluded from engine-completion evidence.

The current development bundle is reproducible with:

```text
archive sha256  b161b9b7fed632a519f9b2099e7f17803b5fab8b2f042c8e19664d5192d4ec98
bundle id       sha256:6180c2ae6e0e2e6d024a5713fcf43e7c2f34ef40280dfb56515675cfecc4b11b
logical files   137
base events     54
rules           23
projections     12
```

The current production release is reproducible with:

```text
archive sha256  4d2015a522281bddeaa3ec9fedda28715677663926bea924a05494ee78ca57af
bundle id       sha256:517038cdd97cb7d3687f53272e8964a11ffcc1cca82cc69a73668bf56aea0514
logical files   825
base events     224
rules           72
projections     74
```

Run the complete gate with:

```bash
python3 afterimage/tools/check_all.py
```

## Coverage

| Scope | Runnable | Designed | Runnable points |
| --- | ---: | ---: | ---: |
| Onboarding ORIENT cases | 5 | 5 | 300 / 300 |
| Vertical slice | 12 | 12 | 1,200 / 1,200 |
| Full problem set | 75 | 75 | 10,000 / 10,000 |

ORIENT.001 fixes event identity and provenance. ORIENT.002 checks parent order,
same-time EventId order, and least-fixed-point replay. ORIENT.003 proves that a
projection is an observation and does not delete unreported active events.
ORIENT.004 retimes a base cause and verifies the exact root/counterfactual
active-set difference after all derived events are recomputed.
ORIENT.005 independently replays an embedded prior witness and grants
`cap:audit.export` only after the portable claim verifies.
CASCADE.001 compares retime, suppression, and reroute policies under explicit
safety/service contracts and lexicographic intervention scoring.
MERGE.001 reconstructs eight of nine clock-drift records, proves the active
message/source order with 12 exact certificate edges, and supports both a
two-record conflict-set baseline and the one-record optimal isolation.
PULSE.001 compiles a 485-byte typed controller with one live deadline cell,
replaces keyed timers, and proves exact output over all 3,304 bounded inputs;
its golden author score is 135/150.
PULSE.002 compiles a 388-byte one-cell controller, preserves the canonical
EventId-first payload once per logical tick, and suppresses all retries across
1,287 exhaustive repeated-tick streams. Its author score is 78/80.
MERGE.002 reconstructs six records around one independently clocked gateway
retry. The public dedup contract requires one survivor and an explicit
duplicate link; false links and double survivors are rejected. Keeping the
stronger primary record scores 79/80, while keeping the weight-1 retry is
valid but scores lower.
PULSE.003 adds distinct start/cancel topics and an actual timer invalidation
action. Its 460-byte, zero-cell controller passes all 545 bounded command
traces, including cancellation exactly at the deadline, and scores 86/90.
MOSAIC.002 places two six-junction surveys under opposite orientations. Their
union covers the same canonical 21-element grid, while two central corridor
edges must each have support from both fragments. Dropping one corroborating
edge preserves union coverage but fails `mosaic_overlap`; the author
certificate scores 79/80.
CASCADE.002 verifies pressure, route, capacity, evidence retention,
maintenance-window, and energy contracts. Its two-retime author repair scores
99/100 and beats the separately verified pump-dispatch baseline.
MOSAIC.001 reconstructs a canonical 9-vertex/12-edge grid from four differently
oriented fragments, proves full coverage, and excludes only the independently
non-embeddable weight-1 decoy; its author score is 129/130.
LENS.001 checks 1,080 valid source/edit pairs plus invalid edits against four
round-trip/provenance laws. Its 32-node, 3-cell complement program scores
177/180.
CASCADE.003 proves that relay `v4-v7` remains in the active causal state while
its public projection suppresses the entire restricted event. The author
witness uses no intervention, leaks no private payload, and scores 139/140;
the verifier accepts the scoped audit capability as a costlier alternative
and rejects the same capability outside this case's scope.
CASCADE.004 continues that finding into the first local-failure act. A retained
occupancy sample observed at tick 9 was ingested at tick 18, so the controller
started heating too late and produced an 11-degree platform. The one-retime
repair restores the sample's causal placement, preserves service and evidence,
and scores 69/70. Emergency heat is independently verified as a valid but more
expensive fallback; retiming even one tick late leaves the platform below its
20-degree contract.
CASCADE.005 proves the priority dashboard can be accurate while the executor is
still inverted by an earlier non-preemptive maintenance grant. Moving that
grant beyond the medical arrival scores 69/70; an equal-tick grant still fails,
and an explicit priority boost is valid but costlier.
CASCADE.006 closes the local-failure act with a three-sided timing corridor:
brine must be effective before the closure crew enters, the road must remain
open through ambulance clearance, and closure must precede the school bus.
Exhaustive testing of all 25 authorized retime pairs finds only `(11, 15)`;
only tick 15 works for the separately verified mobile-barrier alternative. The
two retained external authorization references lead into coupled infrastructure.
CASCADE.007 resolves those references as simultaneous road and traffic loads on
one feeder. The fire pump falls to 35 pressure without being defective. Of all
four authorized road-load retimes only tick 11 releases the feeder in time; a
three-unit backup generator is valid but has intervention weight 24. The author
repair scores 79/80.
MERGE.003 reconstructs the missing source-order edge between utility-gateway
sequences 1 and 3 without inventing sequence 2. Omitting the bridge or adding a
phantom edge fails the exact certificate; alternate in-range schedules remain
valid but have worse raw effective cost. The author certificate scores 89/90.
MOSAIC.003 assembles quarter-turned and reflected surveys, then requires the
least D4 attributed-edge encoding. All seven non-identity orientations are
explicitly rejected as `mosaic_noncanonical`, while a wrong local transform is
reported separately. This deliberately distinguishes harmless coordinate
plurality from the materially different histories Continuity suppresses.
PULSE.004 defines the first bounded flow-control controller. Its two-cell,
633-byte program refills and clamps before each request, emits the original
payload immediately, and decrements only on admission. All 924 nondecreasing
streams of up to six requests over six ticks are checked. Focused mutations
prove that over-admission, unclamped idle credit, rejected-work debit, and
ill-typed `min` arithmetic each fail independently. The author score is 95/100.
CASCADE.008 moves the shared-resource argument from one feeder to an islanded
microgrid. The root has enough daily energy but the signal capacitor is still
charging at the ambulance tick, leaving the clinic only two of its required
three power units and the signal ready three ticks late. Exhausting all four
authorized retimes finds only tick 11; a self-powered mobile signal is valid at
the crossing but has intervention weight 22. The author repair scores 89/90.
MERGE.004 records the corresponding seven-credit transfer as committed and
applied, followed by an equal-and-opposite compensation. The competing
rollback claim can only be timestamped before its applied-transfer parent.
Hostile archives that accept the rollback, erase the original transfer, or
erase the compensation are all rejected; a displaced but causal compensation
remains valid with worse raw cost. The author certificate scores 89/90.
CASCADE.009 carries the 180 evacuees from the repaired island crossing to the
last train. Boarding, signed ventilation, traction reserve, clinic load,
travel time, and bridge closure leave exactly `(ventilation 13, departure 17)`
among all 64 authorized retime pairs. Neither retime works alone. A
self-powered bus bridge works only at tick 17 but has intervention weight 40;
the two-retime rail explanation scores 89/90.
MERGE.005 exposes the dispatch cluster's east and west journals as two complete
writer branches. The host contract forbids splicing them into a synthetic
third history and verifies that their combined source order is infeasible.
The evidence-rich east archive scores 99/100; the complete west archive remains
valid at 93/100, while partial, relabeled, and mixed branches are rejected.
PULSE.005 turns that finding into an operational rule: release after two of
three distinct attestations, not two messages. Its four-cell, 1,293-byte
program preserves the canonical payload of the quorum-completing event and
fires once. All 3,478 bounded streams and focused duplicate/three-source/
retrigger/payload mutations pass; the author score is 96/100.
CASCADE.010 opens the hospital-cooling incident with the first production
`replace` repair. Pipe saturation, thermal response, and shared pump capacity
make commanded flow 5 the only safe value among all integers 0–10. A portable
chiller is independently valid only at tick 18 but carries intervention weight
35; the one-field repair scores 99/100.
MERGE.006 reconstructs the incident's public three-of-five authorization
ledger. Three flow-5 attestations reach quorum; two flow-3 cache votes remain
temporally feasible but are correctly rejected as a minority rather than
misreported as inconsistent. Mixed claims, a two-vote winner, a missing quorum
record, and relabeled rejection reasons are all rejected; the author
certificate scores 99/100.
MOSAIC.004 reveals the physical reason for that exact flow: two surveys cover
the perimeter of a 3×3 cooling blueprint while its central vertex and four
incident edges are omitted. The public coordinate checksum determines the
unique bounded completion. A corrupt inferred label or an extra perimeter gap
fails independently; the author certificate scores 99/100.
CASCADE.011 carries the hospital incident to drawbridge B-17. Wind safety
requires opening no earlier than tick 8, while the two-tick mechanism must be
ready for the ambulance at tick 10; exhaustive retimes 6–9 leave only tick 8.
An emergency ferry is valid only at tick 10 and has intervention weight 30.
The author repair scores 109/110.
MERGE.007 reconstructs the same crossing from two clock islands rather than six
independent clocks. Three north records share offset 100 and three south
records share offset -50; alternating cross-domain messages bind their global
order. Per-record drift and dropping a domain member fail `merge_domain`, while
a displaced pair of common offsets remains valid but scores lower. The author
certificate scores 109/110.
MOSAIC.005 finds that the apparent third bridge survey is a half-turned cached
copy of the north survey. A D4/translation-invariant fingerprint retains edge
attributes while ignoring local names, so the cache cannot count as independent
corroboration. False duplicate links and two used copies are rejected; keeping
the low-weight cache instead of the original is valid but worse. The author
certificate scores 99/100.
PULSE.006 closes the incident with an all-topic barrier: wind, reconstructed
bridge readiness, and hospital readiness must all arrive. Its four-cell,
1,378-byte program releases once with the completing event's payload. All
3,478 bounded streams reject duplicate counting, two-of-three release,
refiring, and wrong payloads; the author score is 106/110.

The next incident begins when river gauge RIVER-4 falls silent after the public
timetable launch. CASCADE.013 chooses reserve 4 as the only value safe across
the complete calibrated interval and feeder limit. MERGE.009 then cuts the
shared legacy gateway with weight 3 instead of deleting two weight-2 probes;
the verifier exhaustively recomputes the minimum hitting set. MOSAIC.007 uses
per-vertex scanner ticks to place otherwise uniform two-column conduit scans.
PULSE.008 checks 10,417 command streams for cold takeover, failure, recovery,
duplicate routing, and payload retention. COVENANT.001 composes the incident
into two local policies and exhaustively checks 80 reachable augmented states;
the author policy's exact worst response is 5.

The covenant's first live dispatch then reveals a vehicle absent from the
public fleet. CASCADE.014 exhausts the 70 integer allocations and leaves only
SHADOW-7 capacity 18 plus train capacity 52. MERGE.010 follows the retained
`echo_of` links through four records and requires every rejected echo to name
its direct predecessor. MOSAIC.008 reconstructs the single missing `v7-v8`
edge only because the published perimeter must be one simple ring and every
other edge is observed. PULSE.009 checks 10,417 bounded streams and stores
completion per logical operation across fail/recover transitions. LENS.003
then proves that direct and echoed internal tips collapse to the same EVAC-17
public row while private deltas and audit chains survive all 48-by-9 lawful
synchronizations.

That collapse exposes a second, deliberately redacted grid history.
CASCADE.015 binds the absent feeder-provenance event through a private digest
without disclosing its payload. MERGE.011 selects thermal trip from two
independent weighted sources despite a count tie. MOSAIC.009 rejects a
geometrically valid anonymous scan because signed support wins each conflicting
edge by the published margin. PULSE.010 batches both signal classes at the
minimum absolute deadline, including equality and reopen boundaries.
CASCADE.016 then tests all three single-cause injections and leaves only the
latent thermal-overload event consistent with the observations.

The proved cause then exposes the policy defect behind the incident.
CASCADE.017 keeps the public FEEDER-9 row byte-identical while changing active
safety from an underground route to the surface bypass. MERGE.012 proves all
seven causal constraints with a necessary four-edge transitive reduction.
MOSAIC.010 separates the flattened map into connected surface and underground
layers with exactly three shaft-backed portals. PULSE.011 checks 792 streams
against a capacity-two half-open switching window. COVENANT.002 composes those
facts: Dispatch cannot inspect sealed heat, so a local sensor must publish
clear/hot before the correct layer-specific route is committed. The model
checker explores 38 states and proves exact worst response bound 4.

The covenant's correction trail becomes the next incident. MERGE.013 groups
records by operation and source, rejects both claims from the one equivocating
gateway, and preserves two agreeing honest witnesses. MOSAIC.011 combines
three overlapping 2×3 strips into a canonical 4×3 city with four shaft-backed
portals, connected layers, and four independently repeated seam edges.
PULSE.012 checks 1,457 bounded unique-index streams and drains every newly
contiguous prefix in the decisive arrival tick. CASCADE.018 adds a dedicated
minimal-explanation contract: the complete sensor-report/route-commit pair is
safe, while the empty explanation and both singletons are independently
replayed and rejected. CASCADE.019 closes the wave by retaining notice N-18's
false underground route and publishing N-19 as its explicit surface-route
correction.

The public-record act then confronts ambiguity and institutional timing.
MERGE.014 proves that both middle-record ticks are consistent and selects the
earlier schedule without rejecting the later one. PULSE.013 checks 3,478
streams for consecutive-failure opening, exact cooldown, a singleton
half-open probe, and recovery. CASCADE.020 retains both safe amendments while
binding the selected one to the published canonical order. CASCADE.021 moves
the filing to the inclusive closing tick rather than extending law to match
service delay. CASCADE.022 compares the whole branch: one route replacement
has lower raw effective cost than the valid disclosure-plus-retime repair.

The finale then closes every remaining family. MERGE.015 retains four feasible
citywide schedules and proves a six-edge partial order with a genuine
surface/underground incomparability. PULSE.014 checks a cyclic three-party City
Clock over 27,064 streams and requires two-round reset behavior. CASCADE.023
rejects individually valid evidence from a stale epoch; CASCADE.024 applies a
public history selector while preserving both safe routes. PARADOX.001 replays
surface and underground CASCADE.024 histories from a fixed shared world,
proves byte-identical public records and safety on both sides, and binds the
material difference to `/payload/route`.

## Remaining gates

No content-design cases remain. The user-authorized AI proxy campaign opened
production on 2026-07-16; no human campaign is claimed. Future work is release
polish or optional human playtesting, not completion of the designed set.

The current verifier now scores all eight designed families: ORIENT, CASCADE,
MERGE, PULSE, MOSAIC, LENS, COVENANT, and PARADOX. Non-root histories are
replayed from canonical policy-checked operation chains. All seventy-five
designed cases now have production content and verified author baselines.
