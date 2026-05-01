#!/usr/bin/env python3
"""Local stochastic search for short Balance copyreg programs."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from balance_solver import (
    BalanceMachine,
    check_puzzle,
    decode_program,
    enc_logic,
    enc_math,
    enc_physics,
    enc_science,
    encode_program,
    puzzle_state,
)


def op_pool() -> list[int]:
    physics = [enc_physics(i) for i in range(-16, 16)]
    science = [enc_science(i) for i in range(-16, 16)]
    math = [enc_math(d, s1, s2) for d in (0, 1) for s1 in range(4) for s2 in range(4)]
    logic = [enc_logic(d, s1, s2) for d in (0, 1) for s1 in range(4) for s2 in range(4)]
    return physics + science + math + logic


def load_programs(paths: list[str]) -> list[list[int]]:
    programs = []
    for path in paths:
        text = Path(path).read_text(encoding="ascii").strip()
        if text:
            programs.append(decode_program(text))
    return programs


def normalize(code: list[int], target_len: int, rng: random.Random, pool: list[int]) -> list[int]:
    code = code[:]
    while len(code) > target_len:
        del code[rng.randrange(len(code))]
    while len(code) < target_len:
        code.insert(rng.randrange(len(code) + 1), rng.choice(pool))
    if enc_science(0) not in code:
        code[rng.randrange(len(code))] = enc_science(0)
    return code


def score(code: list[int], cases: list[int], max_steps: int) -> tuple[int, int, int]:
    ok = 0
    halted = 0
    steps_total = 0
    for a in cases:
        mem, s_r, d_r = puzzle_state("copyreg", (a,))
        machine = BalanceMachine(code, mem, s_r, d_r)
        did_halt, steps = machine.run(max_steps=max_steps)
        steps_total += steps
        if did_halt:
            halted += 1
            if check_puzzle("copyreg", machine, (a,)):
                ok += 1
    return ok, halted, -steps_total


def mutate(
    code: list[int],
    rng: random.Random,
    pool: list[int],
    parents: list[list[int]],
    target_len: int,
) -> list[int]:
    child = code[:]
    r = rng.random()
    if r < 0.42:
        child[rng.randrange(target_len)] = rng.choice(pool)
    elif r < 0.62:
        i = rng.randrange(target_len)
        donor = rng.choice(parents)
        child[i] = donor[rng.randrange(len(donor))]
    elif r < 0.76:
        i = rng.randrange(target_len)
        del child[i]
        child.insert(rng.randrange(len(child) + 1), rng.choice(pool))
    elif r < 0.90:
        i = rng.randrange(target_len)
        j = rng.randrange(target_len)
        child[i], child[j] = child[j], child[i]
    else:
        donor = rng.choice(parents)
        if len(donor) >= 4:
            span = rng.randint(2, min(7, len(donor), target_len))
            src = rng.randrange(len(donor) - span + 1)
            dst = rng.randrange(target_len - span + 1)
            child[dst : dst + span] = donor[src : src + span]
    return normalize(child, target_len, rng, pool)


def seed_population(
    parents: list[list[int]],
    target_len: int,
    pop_size: int,
    rng: random.Random,
    pool: list[int],
) -> list[list[int]]:
    pop: list[list[int]] = []
    for parent in parents:
        pop.append(normalize(parent, target_len, rng, pool))
        for i in range(len(parent)):
            if len(parent) > target_len:
                pop.append(normalize(parent[:i] + parent[i + 1 :], target_len, rng, pool))
    while len(pop) < pop_size:
        parent = rng.choice(parents)
        child = normalize(parent, target_len, rng, pool)
        for _ in range(rng.randint(1, 5)):
            child = mutate(child, rng, pool, parents, target_len)
        pop.append(child)
    return pop[:pop_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-len", type=int, default=27)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--pop", type=int, default=96)
    parser.add_argument("--train", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument(
        "--parents",
        nargs="+",
        default=["copyreg28.bal", "copyreg27.bal", "copyreg29.bal", "copyreg30.bal"],
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pool = op_pool()
    parents = load_programs(args.parents)
    parent_ops = sorted({op for p in parents for op in p})
    pool = pool + parent_ops * 5
    full_cases = list(range(1, 256))
    train_cases = rng.sample(full_cases, min(args.train, len(full_cases)))
    cache: dict[tuple[int, ...], tuple[int, int, int]] = {}
    full_cache: dict[tuple[int, ...], tuple[int, int, int]] = {}

    def train_score(code: list[int]) -> tuple[int, int, int]:
        key = tuple(code)
        if key not in cache:
            cache[key] = score(code, train_cases, args.max_steps)
        return cache[key]

    def full_score(code: list[int]) -> tuple[int, int, int]:
        key = tuple(code)
        if key not in full_cache:
            full_cache[key] = score(code, full_cases, args.max_steps)
        return full_cache[key]

    pop = seed_population(parents, args.target_len, args.pop, rng, pool)
    best_full = (-1, -1, -10**18)
    best_hex = ""
    started = time.time()
    generation = 0
    while time.time() - started < args.seconds:
        generation += 1
        ranked = sorted({tuple(c): c for c in pop}.values(), key=train_score, reverse=True)
        for candidate in ranked[:8]:
            fs = full_score(candidate)
            if fs > best_full:
                best_full = fs
                best_hex = encode_program(candidate)
                p = fs[0] / 255.0
                print(
                    "best",
                    "gen",
                    generation,
                    "full",
                    fs,
                    "p",
                    f"{p:.6f}",
                    "pass5",
                    f"{p**5:.6f}",
                    "hex",
                    best_hex,
                    flush=True,
                )
        elites = ranked[: max(8, args.pop // 4)]
        next_pop = elites[:]
        while len(next_pop) < args.pop:
            parent = rng.choice(elites)
            child = parent[:]
            for _ in range(1 + (rng.random() < 0.25) + (rng.random() < 0.08)):
                child = mutate(child, rng, pool, parents, args.target_len)
            next_pop.append(child)
        pop = next_pop
    print("final", best_full, best_hex)


if __name__ == "__main__":
    main()
