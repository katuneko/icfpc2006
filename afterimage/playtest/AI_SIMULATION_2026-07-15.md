# Six-agent adversarial simulation — 2026-07-15

This report records two isolated waves of three fresh AI agents. It is useful
engineering evidence, but it is **not** an `afterimage-playtest-campaign/0.1`
record and does not satisfy any human timing, comprehension, independence, or
qualitative acceptance criterion in `vertical_slice.md`.

No synthetic observations were entered into the human campaign analyzer.

## Isolation and profiles

Each agent received a new anonymous `T-*` directory, networking was forbidden,
and access to the project source, author baselines, hidden content, other team
directories, and other implementations was forbidden. Session A exposed only
`engine-session.zip`; the official offline `game-session.zip` was exposed only
after Session A ended. The six profiles covered two runtime builders, two
algorithmic contestants, and two curious programmers. Implementations used
Python, JavaScript/Node, and C++17.

The agents' working directories and reports remain outside the release tree
under `/tmp/afterimage-agent-playtest`. They are disposable diagnostic
artifacts, not participant data.

## Wave 1: defects found before hardening

All three agents produced deterministic partial CRE implementations, but none
could pass the original single aggregate-digest gate. All independently
classified the public 37-case split and identified that a mismatch could not
be localized. Their reports converged on the same contract failures:

- the task's one-result description contradicted the normative multi-record
  protocol;
- the aggregate digest exposed no per-case oracle or first mismatch;
- fixture label expansion, base trace null fields, stable error contexts,
  counter shapes, and cross-type ordering were insufficiently explicit.

Using the official offline runtime, all three then reached and solved
`CASCADE.003`. Their routes solved 8, 9, and 8 cases respectively, scoring
630/690, 681/740, and 698/740 on the nominal points they solved. All three
correctly explained active state versus projection and replayed retained
receipts. The wave also exposed three player defects: ordinary editor JSON was
rejected without a conversion path, direct family-helper imports failed, and
equal integer scores selected by witness digest instead of lower raw cost.

Those findings caused the public localized oracle, fixture contract, exact CRE
wording, `player.py canonicalize`, helper import bootstrap, and raw-cost
tie-break to be implemented before Wave 2.

## Wave 2: clean recheck after hardening

Session A was deliberately capped at a short first pass. Two independent
implementations passed all nine canonical vectors and all 37 semantic cases,
including deterministic suite output and successful/error protocol streams.
The third implemented canonical values and hashing, passed all nine vectors,
and stopped honestly at the first unimplemented semantic case. In all three
runs the public checker reported the first differing JSON Pointer, so no agent
was left with an unclassified aggregate mismatch.

All three agents independently found four remaining prose/oracle mismatches:

- `max_value_bytes` is a per-value ceiling, not a returned cumulative counter;
- the frozen fixture also uses an unlabeled `{body}` wrapper;
- canonical test vectors use test-only domain
  `afterimage/test-vector/1`;
- protocol arguments are paths and exit classification is phase-based.

The specifications and engine task now state all four rules explicitly.

In bounded Session B/C passes, all three agents:

- initialized a fresh telemetry-off Desk;
- converted and successfully used ordinary newline-terminated JSON;
- solved six cases and made `CASCADE.003` visible and inspectable;
- ended at score 440 on 480 nominal solved points;
- explained active state versus projection correctly;
- independently replayed every retained receipt.

Each solved `ORIENT.001–003`, `CASCADE.001–002`, and `PULSE.001`. Independent
PULSE programs reached 138/150. One agent reduced program bytes from 370 to
337 while the integer score stayed 138, exercising the lower-raw-cost
same-score condition; another improved from 137 to 138.

The wave found two additional CLI defects. `pulse.py --help` exited silently,
and human-readable `player.py trace` crashed when a derived origin contained
binding bytes even though JSON mode worked. It also showed that auditing a
complete branch trace required internal-runtime composition. The shipped tools
now provide descriptive PULSE help, CRE-safe human rendering, and
`branch --trace-items`; regression tests cover these paths.

## Engineering conclusion

The simulation performed its intended job: fresh implementers localized
contract defects, fresh players exercised progression and replay, and every
reproducible defect found was either fixed or explicitly classified. It raises
confidence in the engine kit and vertical-slice operator surface.

The post-fix release was generated twice byte-for-byte identically. Its package
SHA-256 values are:

```text
engine-session.zip    024d3efb43ef7cd15ed9c571a07d358483025696f6a2c169ebecc8168f7c6861
game-session.zip      68f4e88a564174a11a55fb37b5c62b69b765fb6b53b96e4949ea4e0fd0ee8053
operator-session.zip  616504df8b32f1c326ec4a6f605fa6d00e758d97454018ce4867deab2bab150e
```

It does not answer whether people understand the task within the required
times, whether hints are appropriately paced, whether five of six human teams
can explain the projection/state distinction, or whether the narrative and
family transitions feel satisfying. The remaining release gate is therefore
still the documented six-team human campaign followed by human review of the
analyzer result.

Postscript: on 2026-07-16 the user explicitly replaced that remaining gate
with the separately labeled protocol in `AI_PROXY_PROTOCOL.md`. See
`AI_PROXY_RESULT_2026-07-16.md`; this historical simulation itself was not
retroactively converted into acceptance evidence.
