# Blind-playtest observation contract

This contract turns the acceptance rules in `vertical_slice.md` into a
strict, anonymous campaign record. It is evidence for a human playtest, not a
substitute for one. An operator must derive every field from direct session
observation, consented telemetry, or a completed automated gate.

The separately labeled `AI_PROXY_PROTOCOL.md` and `analyze_ai_proxy.py` do not
alter this human schema. They wrap an estimated campaign under the user's
project-specific proxy policy and emit a distinct decision format.

## File and command

Create a private observation draft directly from the release BundleId. Six is
the default; pass `--teams N` for a larger campaign:

```sh
python3 tools/analyze_playtest.py \
  --new campaign.draft.json \
  --bundle sha256:REPLACE_WITH_CHECKSUMS_BUNDLE \
  --teams 6
```

The generator creates the file with mode `0600`, assigns balanced cohorts,
and generates unique eight-digit `T-*` codes with the operating system's
cryptographic randomness. Every value that still needs direct evidence is
`RECORD_REQUIRED`; it never guesses a favorable default. Do not create a
separate code-to-identity mapping.

Edit the exact object in ordinary UTF-8 JSON after each session, replacing
every `RECORD_REQUIRED`. Then validate and canonicalize it without losing
auditability:

```sh
python3 tools/analyze_playtest.py campaign.draft.json \
  --canonicalize campaign.json
```

The command refuses an existing output path. Then analyze the resulting
canonical CRE JSON file with format
`afterimage-playtest-campaign/0.1`. Canonical JSON means UTF-8 without a BOM,
sorted object keys, NFC strings, no duplicate keys, no insignificant
whitespace, and integers only. The maximum input size is 1 MiB.

From `operator-session` run:

```sh
python3 tools/analyze_playtest.py campaign.json --pretty
```

Exit 0 means `pass`, exit 1 means a well-formed `revise` or `stop` decision,
and exit 2 means the campaign record itself is invalid. The non-pretty output
is canonical JSON. Both canonicalization and analysis print the same
domain-separated `campaign` digest, binding a decision to the exact evidence
bytes. A `pass` authorizes a separate production decision note; it does not
silently authorize or generate content.

If any observation remains unrecorded, canonicalization returns one
`campaign_incomplete` error containing the complete count and field-path list,
and does not create the canonical output.

## Envelope

The root object has exactly four fields:

- `format`: `afterimage-playtest-campaign/0.1`.
- `bundle`: the `sha256:*` BundleId printed in `checksums.json`.
- `system`: one exact system-evidence object described below.
- `teams`: 6 through 64 exact team-observation objects.

Unknown fields are errors. Do not add comments, notes, names, email addresses,
handles, paths, source text, witness material, or terminal content.

## System evidence

`system` has exactly these fields. All except the last are booleans.

- `participant_isolation_verified`: every team was blind to author fixtures,
  solutions, and other teams' artifacts.
- `kernel_receipt_agreement`: the two CRE evaluators and authoritative/offline
  verifiers agreed on all golden fixtures and receipts.
- `all_cases_reachable`: all twelve cases were reachable under the shipped
  unlock manifest.
- `all_cases_precise_diagnostics`: common invalid submissions for every case
  produced a reproducible counterexample or precise failing invariant.
- `acceptance_deterministic`: no acceptance depended on a verifier bug,
  randomness, or undisclosed host ordering.
- `reset_replay_lossless`: retained receipts reproduced byte-for-byte.
- `offline_verified`: the complete play flow worked with networking disabled.
- `verifier_cheaper_than_search`: observed verification remained cheaper than
  expected solution search.
- `semantic_invalidated_cases`: integer from 0 through 75 for the number of
  already-authored cases invalidated by a proposed semantic change.

Set a field to true only when its stated evidence was actually checked.

## Team observation

Each object has exactly the following fields:

- `id`: unique `T-` followed by 4 through 12 ASCII digits. Generate it at
  random; never derive it from a participant identity.
- `cohort`: one of `runtime-builder`, `algorithmic-contestant`, or
  `curious-programmer`.
- `engine_language`: one controlled value: `c`, `cpp`, `csharp`, `elixir`,
  `erlang`, `go`, `haskell`, `java`, `javascript`, `kotlin`, `lua`, `ocaml`,
  `other`, `php`, `python`, `ruby`, `rust`, `scala`, `swift`, `typescript`, or
  `zig`. Use `other` rather than writing a name.
- `first_receipt_minutes`: whole minutes from Session A start to first valid
  receipt, or `null` if none appeared within five hours.
- `desk_boot_minutes`: whole minutes to Desk boot, or `null` if it did not
  boot within the session.
- `reached_cascade003`: whether the team reached CASCADE.003.
- `max_hint_level`: greatest hint level opened, 0 through 3.
- `projection_explained`: the post-reveal explanation correctly distinguished
  active state from projection.
- `intended_observations_understood`: for every solved case reviewed, the team
  identified the intended key observation without needing the author's
  algorithm.
- `independent_valid_families`: unique sorted-or-unsorted family identifiers
  for independently written valid solutions. Allowed values are `ORIENT`,
  `CASCADE`, `MERGE`, `PULSE`, `MOSAIC`, and `LENS`.
- `improved_valid_score`: raw metrics led to at least one better valid score.
- `unrelated_families`: unique family identifiers the team described as
  unrelated filler.
- `dominant_case`: one case consumed more than half of this team's total
  critical-path time.
- `computed_reveal`: if the team reached CASCADE.003, computed event evidence
  caused the projection reveal before or at the reveal text.
- `route_count`: distinct plausible slice routes the team could name.
- `cre_minutes`: whole minutes spent on CRE before Desk boot or session end.
- `canonicalization_dominated`: canonicalization or digest mechanics dominated
  the team's remembered experience.
- `pulse_or_lens_required_second_language`: PULSE or LENS forced a second
  general-purpose implementation before its first interesting problem.
- `cascade_blind_search`: a CASCADE solution became blind parameter search
  without a causal model.
- `validity_score_confused`: the team could not distinguish validity failure
  from score improvement.
- `genre_guessed_reveal`: genre expectations, rather than event evidence,
  produced the reveal guess.
- `irreversible_progress_loss`: an irreversible action lost progress.

All unqualified yes/no fields are booleans. Receipt and Desk times are bounded
to the five-hour session, `cre_minutes` to one week, and `route_count` to 1000,
preventing accidental or hostile huge values.

## Aggregation and decisions

The median is the ordinary middle-value median; a half-minute result is
reported exactly as `{numerator, denominator}`. P90 uses the nearest-rank
definition: sorted value at `ceil(0.9 * team_count)`. Criteria written as
4-of-6, 5-of-6, and 3-of-6 scale to larger campaigns as ceiling ratios of
two-thirds, five-sixths, and one-half. This prevents extra weak observations
from diluting the gate.

A `null` receipt is sorted after every observed receipt. If it occupies a
median or P90 rank, that timing criterion fails and its reported value is
`null`; an unobserved receipt is never converted into a favorable duration.

The analyzer returns:

- `pass` only if every hard and quality criterion passes and no stop trigger
  fires;
- `revise` when no stop trigger fires but at least one criterion fails;
- `stop` whenever any stop-and-redesign trigger fires, even if other criteria
  pass.

The output preserves every measured aggregate and threshold, so the decision
can be audited without retaining personal or solution data.
