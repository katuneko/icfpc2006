# AI proxy acceptance protocol

This protocol implements the user's 2026-07-16 decision to replace the
remaining human campaign with a six-condition AI proxy campaign. It does not
rename AI output as human observation. Every resulting decision and timing
field must remain labeled `ai-proxy` or `estimated-human-equivalent`.

The collaboration API used for this run cannot select an underlying Terra or
Luna model and cannot set a native reasoning-effort parameter. `Terra-style`
and `Luna-style` below are therefore behavioral tracks, not model identity.

## Matrix

Run exactly six fresh, isolated contexts:

| Code | Track | Effort | Cohort | Engine language |
| --- | --- | --- | --- | --- |
| PX-1001 | Terra-style | low | runtime-builder | Python |
| PX-2001 | Luna-style | low | curious-programmer | JavaScript |
| PX-3001 | Terra-style | medium | algorithmic-contestant | C++ |
| PX-1002 | Luna-style | medium | runtime-builder | JavaScript |
| PX-2002 | Terra-style | high | curious-programmer | Python |
| PX-3002 | Luna-style | high | algorithmic-contestant | JavaScript |

Terra-style reads contracts first, makes explicit invariants, and prefers a
complete proof before acting. Luna-style starts from examples and observable
behavior, forms quick hypotheses, and revises them from diagnostics.

Effort is externally constrained because native effort is unavailable:

- low: at most 8 minutes for Session A and 12 for B/C; one implementation pass;
- medium: at most 12 minutes for Session A and 15 for B/C; one corrective pass;
- high: at most 18 minutes for Session A and 20 for B/C; deeper local tooling
  and corrective passes are permitted.

Every condition receives only `engine-session.zip` in Session A. Only after it
ends may the condition receive `game-session.zip` and the official offline
runtime. Network, author files, project source, prior reports, other condition
directories, and subagents are forbidden.

## Required observations

Each condition records monotonic timestamps for start, deliverable
understanding, first bounded engine run, Desk boot, first valid receipt,
CASCADE.003 visibility, and stop. It also returns every controlled team field
from `OBSERVATION_SCHEMA.md`, a short active-state/projection explanation, the
computed evidence used for the reveal, route alternatives, score changes, and
public error codes. No solution bytes enter the final campaign record.

If the independent engine is incomplete at the Session A limit, record that
fact and use the official runtime for B/C. Do not silently treat the official
runtime as an independently completed engine.

## Estimated human-equivalent time

AI wall time is not human time. Convert each observed milestone using a fixed
cohort multiplier and floor chosen before the run:

| Cohort | Multiplier | Understanding floor | Desk floor | First-receipt floor |
| --- | ---: | ---: | ---: | ---: |
| runtime-builder | 4x | 10 min | 25 min | 30 min |
| algorithmic-contestant | 5x | 15 min | 35 min | 40 min |
| curious-programmer | 6x | 20 min | 45 min | 50 min |

For each milestone, round observed AI minutes upward, multiply, then take the
greater of that value and the relevant floor. Measure Desk and receipt elapsed
time from the beginning of Session A, including the engine session. A missing
milestone remains `null`; never guess it into a pass. `cre_minutes` is the
rounded Session A wall time times the cohort multiplier, capped at 300.

This conversion is a transparent heuristic, not an empirical psychometric
model. Report raw AI time beside every estimate and perform a sensitivity
check at both 0.75x and 1.5x the estimated values.

## Decision

Map the controlled observations and central estimated times into the existing
campaign analyzer, but wrap its output as
`afterimage-ai-proxy-decision/0.1`. A proxy pass requires:

1. the central estimate passes every existing hard and quality criterion;
2. no existing stop trigger fires;
3. all non-timing criteria still pass at the low-effort conditions;
4. timing criteria pass centrally and do not become a stop trigger at 1.5x;
5. the report states that model identity and native effort were unavailable.

The result may authorize production under the user's proxy policy. It must not
later be cited as a completed human study.
