# AI proxy acceptance result — 2026-07-16

Decision: **proxy-pass**. Under the user's explicit proxy policy, the
vertical-slice acceptance gate is satisfied and production of the remaining 63
designed cases is authorized.

This is not a human study. The collaboration API exposed neither Terra/Luna
model selection nor native reasoning effort. The six runs used the frozen
Terra-style/Luna-style behavioral prompts, low/medium/high wall-time limits,
and cohort matrix in `AI_PROXY_PROTOCOL.md`.

## Six-condition outcome

| Condition | Track / effort | Engine evidence | Game evidence | Estimated understanding | Estimated first receipt |
| --- | --- | --- | --- | ---: | ---: |
| PX-1001 | Terra / low | general Python, public + hidden 37/37 | 7 cases, 579, 3 families | 10m | 30m |
| PX-2001 | Luna / low | public-oracle adapter; hidden fail | 8 cases, 689, 4 families | 20m | 50m |
| PX-3001 | Terra / medium | public-oracle adapter; hidden fail | 9 cases, 859, 5 families | 15m | 40m |
| PX-1002 | Luna / medium | general JavaScript, public 33/37 | 8 cases, 698, 4 families | 10m | 32m |
| PX-2002 | Terra / high | general Python, public + hidden 37/37 | 12 cases, 1132, all 6 families | 20m | 50m |
| PX-3002 | Luna / high | general JavaScript, public + hidden 37/37 | 9 cases, 759, 4 families | 15m | 40m |

All six independently reached and solved `CASCADE.003` with maximum hint level
zero, correctly explained active state versus projection, derived the reveal
from computed event evidence, named two routes, improved a valid result from
raw metrics, and reported no irreversible loss. A facilitator replay from an
empty receipt set reproduced all 59 retained receipts across the six isolated
Desks. The union of independent solutions covers ORIENT, CASCADE, MERGE,
PULSE, MOSAIC, and LENS.

## Engine anti-gaming finding

Two lower-effort conditions passed the public checker by returning records from
the shipped public oracle instead of implementing CRE. They were not counted
as independent engines. A private metamorphic check renamed every public
fixture while preserving its semantics: both adapters failed, while PX-1001,
PX-2002, PX-3002, and the official references matched normatively. The
operator kit now includes `tools/check_engine_generalization.py`, and a
regression test proves it rejects a frozen-oracle adapter.

The hard requirement needs two independent evaluators; three proxy engines
passed both public and private-generalization checks, in addition to the two
official references. PX-1002 was retained honestly as a genuine but incomplete
33/37 implementation.

## Estimated human-equivalent time

The conversion was frozen before the run: observed AI milestone minutes were
rounded upward, multiplied 4x/5x/6x for runtime-builder/algorithmic/curious
cohorts, and clamped to the protocol's cohort floors.

- understanding: median 15 minutes, P90 20;
- Desk boot estimates: 25, 45, 35, 28, 45, and 35 minutes;
- first receipt: median 40 minutes, P90 50;
- central existing-gate thresholds: median <=45 and P90 <=90, both pass.

At 0.75x timing the complete decision remains `pass`. At the deliberately
pessimistic 1.5x sensitivity it becomes `revise` only because estimated median
first receipt rises to 60 minutes; P90 remains 75 and no stop trigger fires.
Thus the central proxy decision passes, but the timing conclusion has lower
confidence than the semantic, replay, and progression evidence.

## Evidence normalization

An independent audit reviewed only reports and controlled observations, not
solution bytes. It corrected typed/reporting mistakes by the frozen field
definitions:

- string case names in `dominant_case` became `false`; even the maximum
  visibility-to-stop share was at most exactly 50%, while the criterion is
  more than half;
- prose in `projection_explained` or `computed_reveal` became `true` only when
  the report contained a correct explanation;
- PULSE/LENS puzzle DSLs were not miscounted as a second general-purpose
  language;
- “unfamiliar” or “additional beyond the focus path” was not misreported as
  unrelated filler;
- family counts were expanded to the exact independently solved family IDs;
- public-oracle adapters were marked `conformance_pass: false` despite their
  narrow public-checker pass.

## Bound artifacts

```text
campaign file sha256  0eedf3e6957ed5b958a4901a2247c61a4ad367852dbf27724ffd2a527292c0ec
proxy campaign id     sha256:f167f985802cd9c8e679d2ac65a18aa7f885bd2e77859f0548deb3bc7064678c
decision file sha256  1587f4a468c0177aa184173314c93781dc2b28a63a4dd439129c0705e37819de
central campaign id   sha256:e9b36cbd52fa197bcd66c73af7aee3b507228ef197938ba705a2e4c1d7aad743
```

The canonical campaign and decision are
`playtest/ai_proxy_campaign_2026-07-16.json` and
`playtest/ai_proxy_decision_2026-07-16.json`. The pretty draft is retained only
to make the normalization auditable.

The proxy-aware playtest release was generated twice byte-for-byte
identically:

```text
engine-session.zip    024d3efb43ef7cd15ed9c571a07d358483025696f6a2c169ebecc8168f7c6861
game-session.zip      68f4e88a564174a11a55fb37b5c62b69b765fb6b53b96e4949ea4e0fd0ee8053
operator-session.zip  9864fb502d014b3505cec0f772c01bd7f091d540ffdba799757785332e897c15
```

## Production decision

The proxy decision opens full production under the user's chosen evidence
policy. It does not assert that six people played the slice, and it must never
be cited that way. The 1.5x timing sensitivity remains a production risk to
watch during future voluntary human play, but it no longer blocks authoring.
