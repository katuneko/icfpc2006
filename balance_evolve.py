#!/usr/bin/env python3
"""Stochastic short-program search for Balance puzzle candidates."""

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


def normalize(
    code: list[int],
    target_len: int,
    rng: random.Random,
    pool: list[int],
    require_neg_science: bool,
) -> list[int]:
    code = code[:]
    while len(code) > target_len:
        del code[rng.randrange(len(code))]
    while len(code) < target_len:
        code.insert(rng.randrange(len(code) + 1), rng.choice(pool))
    if enc_science(0) not in code:
        code[rng.randrange(target_len)] = enc_science(0)
    if require_neg_science and not any((b >> 5) == 0 and (b & 0x10) for b in code):
        code[rng.randrange(target_len)] = enc_science(-rng.randint(1, 16))
    return code


def random_cases(puzzle: str, rng: random.Random, count: int) -> list[tuple[int, ...]]:
    if puzzle in ("copymem", "copyreg"):
        values = list(range(1, 256))
        if count >= len(values):
            return [(a,) for a in values]
        return [(a,) for a in rng.sample(values, count)]
    if puzzle == "multmem":
        return [(rng.randint(1, 255), rng.randint(1, 255)) for _ in range(count)]
    if puzzle == "fillmem":
        cases = []
        for _ in range(count):
            i = rng.randint(8, 254)
            j = rng.randint(i + 1, 255)
            cases.append((rng.randint(1, 255), i, j))
        return cases
    raise ValueError(f"unsupported puzzle: {puzzle}")


def score(
    puzzle: str,
    code: list[int],
    cases: list[tuple[int, ...]],
    max_steps: int,
) -> tuple[int, int, int]:
    ok = 0
    halted = 0
    steps_total = 0
    for params in cases:
        mem, s_r, d_r = puzzle_state(puzzle, params)
        machine = BalanceMachine(code, mem, s_r, d_r)
        did_halt, steps = machine.run(max_steps=max_steps)
        steps_total += steps
        if did_halt:
            halted += 1
            if check_puzzle(puzzle, machine, params):
                ok += 1
    return ok, halted, -steps_total


def mutate(
    code: list[int],
    rng: random.Random,
    pool: list[int],
    parents: list[list[int]],
    target_len: int,
    require_neg_science: bool,
) -> list[int]:
    child = code[:]
    r = rng.random()
    if r < 0.36:
        child[rng.randrange(target_len)] = rng.choice(pool)
    elif r < 0.56:
        donor = rng.choice(parents)
        child[rng.randrange(target_len)] = donor[rng.randrange(len(donor))]
    elif r < 0.72:
        del child[rng.randrange(target_len)]
        child.insert(rng.randrange(len(child) + 1), rng.choice(pool))
    elif r < 0.84:
        i = rng.randrange(target_len)
        j = rng.randrange(target_len)
        child[i], child[j] = child[j], child[i]
    elif r < 0.94:
        donor = rng.choice(parents)
        span = rng.randint(2, min(8, len(donor), target_len))
        src = rng.randrange(len(donor) - span + 1)
        dst = rng.randrange(target_len - span + 1)
        child[dst : dst + span] = donor[src : src + span]
    else:
        cut = rng.randrange(target_len)
        child = child[cut:] + child[:cut]
    return normalize(child, target_len, rng, pool, require_neg_science)


def seed_population(
    parents: list[list[int]],
    target_len: int,
    pop_size: int,
    rng: random.Random,
    pool: list[int],
    require_neg_science: bool,
) -> list[list[int]]:
    pop: list[list[int]] = []
    for parent in parents:
        pop.append(normalize(parent, target_len, rng, pool, require_neg_science))
        if len(parent) > target_len:
            for i in range(len(parent)):
                pop.append(normalize(parent[:i] + parent[i + 1 :], target_len, rng, pool, require_neg_science))
    while len(pop) < pop_size:
        child = normalize(rng.choice(parents), target_len, rng, pool, require_neg_science)
        for _ in range(rng.randint(1, 6)):
            child = mutate(child, rng, pool, parents, target_len, require_neg_science)
        pop.append(child)
    return pop[:pop_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("puzzle", choices=("copymem", "copyreg", "multmem", "fillmem"))
    parser.add_argument("--target-len", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--pop", type=int, default=128)
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--probe", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--parents", nargs="+", required=True)
    parser.add_argument("--require-neg-science", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    parents = load_programs(args.parents)
    pool = op_pool()
    parent_ops = sorted({op for parent in parents for op in parent})
    pool = pool + parent_ops * 6

    train_cases = random_cases(args.puzzle, rng, args.train)
    probe_cases = random_cases(args.puzzle, rng, args.probe)
    cache: dict[tuple[int, ...], tuple[int, int, int]] = {}
    probe_cache: dict[tuple[int, ...], tuple[int, int, int]] = {}

    def train_score(code: list[int]) -> tuple[int, int, int]:
        key = tuple(code)
        if key not in cache:
            cache[key] = score(args.puzzle, code, train_cases, args.max_steps)
        return cache[key]

    def probe_score(code: list[int]) -> tuple[int, int, int]:
        key = tuple(code)
        if key not in probe_cache:
            probe_cache[key] = score(args.puzzle, code, probe_cases, args.max_steps)
        return probe_cache[key]

    pop = seed_population(
        parents,
        args.target_len,
        args.pop,
        rng,
        pool,
        args.require_neg_science,
    )
    best_probe = (-1, -1, -10**18)
    best_hex = ""
    started = time.time()
    generation = 0
    while time.time() - started < args.seconds:
        generation += 1
        ranked = sorted({tuple(c): c for c in pop}.values(), key=train_score, reverse=True)
        for candidate in ranked[:10]:
            ps = probe_score(candidate)
            if ps > best_probe:
                best_probe = ps
                best_hex = encode_program(candidate)
                p = ps[0] / len(probe_cases)
                print(
                    "best",
                    "gen",
                    generation,
                    "probe",
                    ps,
                    "p",
                    f"{p:.6f}",
                    "pass5",
                    f"{p**5:.8f}",
                    "hex",
                    best_hex,
                    flush=True,
                )
        elites = ranked[: max(8, args.pop // 4)]
        next_pop = elites[:]
        while len(next_pop) < args.pop:
            child = rng.choice(elites)[:]
            for _ in range(1 + (rng.random() < 0.30) + (rng.random() < 0.10)):
                child = mutate(
                    child,
                    rng,
                    pool,
                    parents,
                    args.target_len,
                    args.require_neg_science,
                )
            next_pop.append(child)
        pop = next_pop
    print("final", best_probe, best_hex)


if __name__ == "__main__":
    main()
