# Afterimage public player quickstart

This kit contains the complete 75-case Afterimage production world. It contains
no author witnesses, author baselines, case sources, golden answers, or
authoring tools.

## Requirements

- Python 3.11 or newer
- No third-party package
- No network connection

## Start

From the extracted player-kit directory:

```bash
python3 tools/player.py --json init afterimage-2.1.afterimage desk
python3 tools/player.py status desk
python3 tools/player.py inspect desk ORIENT.001
```

Select English, Japanese, Simplified Chinese, or German before the command:

```bash
python3 tools/player.py --locale ja inspect desk ORIENT.001
python3 tools/player.py --locale zh-Hans hint desk ORIENT.001 1
AFTERIMAGE_LOCALE=de python3 tools/player.py status desk
```

Locale selection changes presentation only. The bundle, answer schema,
witness, verifier, score, and canonical receipt remain identical.

## Submit a witness

The `inspect` output gives the case letter and answer schema. Draft ordinary
JSON, canonicalize it, then verify it:

```bash
python3 tools/player.py canonicalize my-answer.draft.json my-answer.json
python3 tools/player.py verify desk my-answer.json
python3 tools/player.py score desk
python3 tools/player.py status desk
```

Invalid submissions are not retained. A valid receipt unlocks later cases.

## Investigate and branch

```bash
python3 tools/player.py inspect desk EVENT_ID
python3 tools/player.py trace desk EVENT_ID --parents
python3 tools/player.py branch desk CASE --intervention intervention.json
python3 tools/player.py compare desk BRANCH_A BRANCH_B
python3 tools/player.py hint desk CASE 1
```

## Reproduce or reset

```bash
python3 tools/player.py replay desk
python3 tools/player.py reset desk --keep-witnesses
```

`replay` independently verifies retained witnesses from an empty receipt set
and requires every receipt to reproduce byte-for-byte.

## Verify the package

Compare the included `package.json` with the release `checksums.json`. The
canonical production archive identity is:

```text
BundleId        sha256:517038cdd97cb7d3687f53272e8964a11ffcc1cca82cc69a73668bf56aea0514
Archive SHA-256 4d2015a522281bddeaa3ec9fedda28715677663926bea924a05494ee78ca57af
```

