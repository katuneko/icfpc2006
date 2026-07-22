# Afterimage blind-playtest operator protocol

Use six or more independent people or teams: at least two runtime/language
builders, two algorithmic contestants, and two curious programmers. No
participant may see `content/vertical_slice`, `author-baselines`, `golden.json`,
the authoring tools, or another team's artifacts.

## Before the campaign

1. Hash all three release zips against `checksums.json`.
2. Create exactly one private observation sheet with the exact BundleId:

   ```sh
   python3 tools/analyze_playtest.py \
     --new campaign.draft.json \
     --bundle sha256:REPLACE_WITH_CHECKSUMS_BUNDLE \
     --teams 6
   ```

3. Confirm the generated cohorts have at least two teams each. Never create a
   participant-name mapping; the random `T-*` code is the only identifier.
4. Assign each recruited team one unused generated code and the corresponding
   cohort. Give the code to the team; do not store who received it.

## Before each team session

1. Obtain explicit consent before enabling telemetry. Default to disabled.
2. Give Session A only `engine-session.zip`.
3. Recheck the received zip hash and disable networking.
4. Keep replacing that team's `RECORD_REQUIRED` values from direct
   observations. Do not record terminal content or source code.

## Session A: engine and onboarding

Allow five hours. Do not correct semantic misunderstandings during the first
60 minutes. Record task-understanding time, first bounded run, first valid
receipt, documentation ambiguities, and whether the Desk boots. End Session A
before giving the participant `game-session.zip`.

## Session B: critical path

Start with a conforming engine or the official offline runtime. Ask the team to
reach CASCADE.003 by any route. Record case order, hint level, public error
codes, reset count, and valid scores. After the reveal, ask the participant to
explain active state versus projection without quoting the story text.

## Session C: breadth and optimization

Expose all slice cases. Ask the team to solve unfamiliar families and improve
one valid score. Record which metric caused the change, the least-connected
family, two plausible routes, and whether any case became blind parameter
search.

## Data boundary

Accept only `telemetry-export` output and structured interview answers. Never
collect witnesses, answers, interventions, event payloads, filesystem paths,
shell history, repositories, credentials, screen recordings, or unrelated
terminal output. Delete accidental captures immediately and note only that a
protocol violation occurred.

## Release decision

Apply every hard, quality, and stop-and-redesign criterion in
`vertical_slice.md`. Translate direct observations into the exact anonymous
fields in `OBSERVATION_SCHEMA.md`; do not infer missing successes. Validate a
pretty draft and produce the immutable input with:

```sh
python3 tools/analyze_playtest.py campaign.draft.json \
  --canonicalize campaign.json
python3 tools/analyze_playtest.py campaign.json > decision.json
python3 tools/analyze_playtest.py campaign.json --pretty
```

The second command exits 0 only for `pass`, 1 for `revise` or `stop`, and 2
for invalid evidence. Human timing, comprehension, and independent-solution
criteria cannot be replaced by author baselines or automated tests. Review all
failed criteria and stop triggers, confirm that canonicalization and decision
print the same `campaign` digest, then record that digest in a separate
decision note.
Authorize the remaining 63 cases only after the analyzer says `pass` and a
human reviewer accepts the evidence.

## Project-specific proxy override

The protocol above remains the human-study procedure. For this project, the
user explicitly authorized `AI_PROXY_PROTOCOL.md` as a replacement gate on
2026-07-16. Its output must remain labeled AI proxy and estimated time; it may
authorize production but must never be represented as a completed human
campaign.

Before accepting an independently implemented proxy engine, run the private
renamed-fixture check so a frozen public-oracle adapter cannot count:

```sh
python3 tools/check_engine_generalization.py -- ENGINE
```

Analyze the canonical proxy campaign separately with:

```sh
python3 tools/analyze_ai_proxy.py ai-proxy-campaign.json --pretty
```
