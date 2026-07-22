# Afterimage playtest quickstart

This kit is the offline Continuity Desk for the twelve-case Afterimage slice.
It contains no author witnesses. Keep your own solutions outside the kit or in
the workspace created below.

## 1. Check the kit

Use Python 3.11 or newer. No network access or third-party package is needed.
From the game-kit directory:

```bash
python3 tools/player.py --json init afterimage-slice.afterimage desk
python3 tools/player.py status desk
```

If you consent to the playtest telemetry described by the facilitator, add
`--telemetry` to `init`. Telemetry is off by default and never stores answer,
intervention, event-payload, witness, filesystem-path, shell-history, or source
content.

## 2. Enter the world

`status` lists only currently visible cases. Begin with:

```bash
python3 tools/player.py inspect desk ORIENT.001
```

Choose English, Japanese, Simplified Chinese, or German with `--locale en`,
`--locale ja`, `--locale zh-Hans`, or `--locale de` before the command. You may
also set `AFTERIMAGE_LOCALE` once for the shell:

```bash
python3 tools/player.py --locale ja inspect desk ORIENT.001
AFTERIMAGE_LOCALE=de python3 tools/player.py hint desk ORIENT.001 1
```

Locale selection changes presentation only. Every language uses the same
bundle, answer schema, witness, verifier, score, and canonical receipt.

The case output includes its task letter and answer schema. The normative
formats and semantics are in `spec/`. All commands work without a network.

Create a canonical `afterimage-witness/0.1` JSON file, then submit it:

```bash
python3 tools/player.py canonicalize my-orient-001.draft.json my-orient-001.json
python3 tools/player.py verify desk my-orient-001.json
python3 tools/player.py score desk
python3 tools/player.py status desk
```

A valid receipt is retained and unlocks later cases. Invalid submissions are
not retained. `--json` before the command emits canonical machine-readable
output; the default is human-readable. The `canonicalize` command accepts
ordinary pretty JSON, removes editor whitespace/final newlines, normalizes NFC,
and rejects duplicate keys, floating point, invalid Unicode, and an existing
output path. Use it for witnesses and intervention files, including a draft
whose entire content is `null`.

## 3. Investigate and experiment

```bash
python3 tools/player.py inspect desk EVENT_ID
python3 tools/player.py trace desk EVENT_ID --parents
python3 tools/player.py trace desk EVENT_ID --children
python3 tools/player.py branch desk CASE --intervention intervention.json
python3 tools/player.py branch desk CASE --intervention intervention.json --trace-items
python3 tools/player.py branch desk LATER_CASE --history PARENT_BRANCH_ID \
  --intervention later-intervention.json
python3 tools/player.py compare desk BRANCH_ID_A BRANCH_ID_B
python3 tools/player.py hint desk CASE 1
```

An intervention file is either canonical `null` or an
`afterimage-intervention/0.1` envelope accepted by that case's published
policy. Branch snapshots contain projection records and digests, not private
active-event payloads. `--trace-items` additionally includes the complete
ordered trace in that command's output when a case asks you to audit it; the
stored snapshot remains compact. Family helper modules are normally invoked by
`player.py verify`; `python3 tools/pulse.py --help` documents the standalone
PULSE helper boundary. Hints have three levels and every opening is counted
only when telemetry was enabled with consent.

For a descriptor with `input_branch: history:CASE`, pass the parent BranchId
printed by the earlier `branch` command, a history JSON file, or its player
branch snapshot. The Desk replays and policy-checks the complete history from
root; the parent digest alone is never accepted as proof.

## 4. Protect and reproduce progress

```bash
python3 tools/player.py replay desk
python3 tools/player.py reset desk --keep-witnesses
```

Both commands independently verify retained witnesses from an empty receipt
set. They fail if any receipt cannot be reproduced byte-for-byte. Omitting
`--keep-witnesses` deliberately clears receipts, witnesses, and branches.

If telemetry was enabled, export it after the session to a new file:

```bash
python3 tools/player.py telemetry-export desk playtest-telemetry.json
```

Do not send source code, shell history, credentials, or unrelated files to the
facilitator. Report tool failures with the command, exit code, and public error
code only.
