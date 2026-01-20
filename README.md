# icfpc2006

Working directory for the ICFPC 2006 puzzle.

## Highlights
- UM interpreter source: `um.rs` (builds to `um`)
- O'Cult evaluator: `occult.py`
- Programs and artifacts: `*.um`, `*.umz`
- Inputs and logs: `*_input.txt`, `volume9_*.txt`
- Docs and specs: `problem.md`, `walkthrough.md`, `um_spec.txt`

## Build
```bash
rustc um.rs -O -o um
```

## Run
```bash
./um volume9.um < howie_ls_input.txt > volume9_howie_ls.txt
```

## O'Cult tests
```bash
python3 occult.py arith4.adv arith.tests
```
