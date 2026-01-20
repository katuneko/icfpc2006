# Repository Guidelines

## Project Structure and Module Organization
This repo is a working directory for the ICFPC 2006 puzzle. Most files live in the root.
- `um.rs` is the Rust UM interpreter source; `um` is a built binary.
- `occult.py` is a small O'Cult evaluator used for local checks.
- UM programs and artifacts: `*.um`, `*.umz` (ex: `volume9.um`, `codex.umz`).
- Advice and tests: `*.adv`, `*.tests` (ex: `arith4.adv`, `arith.tests`).
- Inputs and logs: `*_input.txt` for replayable sessions, `volume9_*.txt` for captured output.
- Docs and specs: `problem.md`, `walkthrough.md`, `task.txt`, `um_spec.txt`.

## Build, Test, and Development Commands
- Build the UM interpreter: `rustc um.rs -O -o um`
- Run a UM program with saved input: `./um volume9.um < howie_ls_input.txt > volume9_howie_ls.txt`
- Run O'Cult tests: `python3 occult.py arith4.adv arith.tests`
- No global build system is configured; keep commands self contained.

## Coding Style and Naming Conventions
- Rust: follow rustfmt conventions and 4 space indentation; types `CamelCase`, functions `snake_case`.
- Python: PEP8 style with 4 space indentation; keep functions in `snake_case`.
- Filenames: lowercase with underscores; new logs should keep the `*_input.txt` and `volume9_*` pattern.

## Testing Guidelines
- O'Cult tests live in `*.tests` and are executed with `occult.py`.
- Add new tests next to the advice file they validate (ex: `xml2.adv` with `xml2.tests`).
- There is no coverage target; run local tests before changing advice rules.

## Data and Log Handling
- Treat existing logs as immutable records; do not edit them in place.
- For new runs, create a fresh `*_input.txt` and matching `volume9_*.txt` output so results are reproducible.
- When you make notable progress, update `walkthrough.md` to keep the working notes current.

## Commit and Pull Request Guidelines
- This directory does not contain git history. If you add one, use short imperative messages (ex: `um: fix load program copy`).
- PRs should describe the puzzle step addressed and include any new input or output logs.
