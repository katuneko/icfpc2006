#!/usr/bin/env python3
"""Fixed-length local search around the known probabilistic multmem program."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random

import balance_solver as balance


PAIRS: list[tuple[int, int]] = []
MAX_STEPS = 0


def evaluate(code: list[int]) -> tuple[int, int, str]:
    ok = halted = 0
    for a, b in PAIRS:
        mem, sr, dr = balance.puzzle_state("multmem", (a, b))
        machine = balance.BalanceMachine(code, mem, sr, dr)
        did_halt, _ = machine.run(MAX_STEPS)
        halted += did_halt
        if did_halt and balance.check_puzzle("multmem", machine, (a, b)):
            ok += 1
    return ok, halted, balance.encode_program(code)


def init_worker(pairs: list[tuple[int, int]], max_steps: int) -> None:
    global PAIRS, MAX_STEPS
    PAIRS = pairs
    MAX_STEPS = max_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=450)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    # Small multipliers give a cheap first-stage signal; survivors must be
    # rechecked separately on the full 1..255 distribution.
    pairs = [(rng.randrange(1, 256), rng.randrange(1, 33)) for _ in range(args.train)]
    known = balance.decode_program(open("multmem.bal", encoding="ascii").read())
    seeds = [known[:index] + known[index + 1 :] for index in (2, 3, 5, 10, 12)]
    operations = list(range(0x80))

    with mp.Pool(args.jobs, initializer=init_worker, initargs=(pairs, args.max_steps)) as pool:
        for round_index in range(args.rounds):
            candidates: dict[tuple[int, ...], list[int]] = {}
            for code in seeds:
                candidates[tuple(code)] = code
                for index in range(len(code)):
                    for byte in operations:
                        if byte != code[index]:
                            changed = code[:]
                            changed[index] = byte
                            candidates[tuple(changed)] = changed
            values = pool.map(evaluate, candidates.values(), chunksize=32)
            values.sort(reverse=True)
            print("round", round_index, *values[:12], sep="\n", flush=True)
            seeds = [balance.decode_program(value[2]) for value in values[: args.beam]]


if __name__ == "__main__":
    main()
