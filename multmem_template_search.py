#!/usr/bin/env python3
"""Constrained search for multmem candidates with loop structure.

The goal is to avoid pure random byte search by forcing at least:
  - one SCIENCE with negative IMM (loop candidate)
  - one SCIENCE 0 (halt candidate)
"""

import argparse
import random

from balance_solver import (
    BalanceMachine,
    check_puzzle,
    enc_logic,
    enc_math,
    enc_physics,
    enc_science,
    encode_program,
    puzzle_state,
)


def score(code: list[int], pairs: list[tuple[int, int]], max_steps: int) -> tuple[int, int]:
    ok = 0
    halted = 0
    for a, b in pairs:
        mem, sR, dR = puzzle_state("multmem", (a, b))
        machine = BalanceMachine(code, mem, sR, dR)
        did_halt, _ = machine.run(max_steps=max_steps)
        if did_halt:
            halted += 1
            if check_puzzle("multmem", machine, (a, b)):
                ok += 1
    return ok, halted


def build_ops() -> tuple[list[int], list[int], list[int], list[int]]:
    physics_ops = [enc_physics(imm) for imm in range(-16, 16)]
    math_ops = [enc_math(d, s1, s2) for d in (0, 1) for s1 in range(4) for s2 in range(4)]
    logic_ops = [enc_logic(d, s1, s2) for d in (0, 1) for s1 in range(4) for s2 in range(4)]
    sci_neg = [enc_science(imm) for imm in range(-16, 0)]
    return physics_ops, math_ops, logic_ops, sci_neg


def random_candidate(
    rng: random.Random,
    physics_ops: list[int],
    ml_ops: list[int],
    sci_neg: list[int],
    max_len: int,
) -> list[int]:
    # Template: preamble + body + loop-jump + epilogue + halt
    pre_len = rng.randint(1, min(10, max_len - 4))
    body_len = rng.randint(1, min(6, max_len - pre_len - 2))
    epi_len = rng.randint(0, min(6, max_len - pre_len - body_len - 2))

    code: list[int] = []
    for _ in range(pre_len):
        code.append(rng.choice(physics_ops + ml_ops))
    for _ in range(body_len):
        code.append(rng.choice(ml_ops + physics_ops))
    code.append(rng.choice(sci_neg))
    for _ in range(epi_len):
        code.append(rng.choice(physics_ops + ml_ops))
    code.append(enc_science(0))
    return code


def mutate(
    rng: random.Random,
    code: list[int],
    physics_ops: list[int],
    ml_ops: list[int],
    sci_neg: list[int],
    max_len: int,
) -> list[int]:
    c = code[:]
    r = rng.random()
    pool = physics_ops + ml_ops + sci_neg + [enc_science(0)]
    if r < 0.55 and c:
        i = rng.randrange(len(c))
        c[i] = rng.choice(pool)
    elif r < 0.78 and len(c) < max_len:
        i = rng.randrange(len(c) + 1)
        c.insert(i, rng.choice(pool))
    elif len(c) > 3:
        i = rng.randrange(len(c))
        del c[i]

    # Keep constraints.
    if not any(((b >> 5) & 0x7) == 0 and (b & 0x1F) >= 0x10 for b in c):
        c.insert(max(0, len(c) // 2), rng.choice(sci_neg))
    if enc_science(0) not in c:
        c.append(enc_science(0))
    if len(c) > max_len:
        c = c[:max_len]
        if enc_science(0) not in c:
            c[-1] = enc_science(0)
    return c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=8000)
    parser.add_argument("--pop", type=int, default=36)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--train", type=int, default=800)
    parser.add_argument("--probe", type=int, default=20000)
    parser.add_argument("--max-steps", type=int, default=620)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    physics_ops, math_ops, logic_ops, sci_neg = build_ops()
    ml_ops = math_ops + logic_ops

    train = [(rng.randint(1, 255), rng.randint(1, 255)) for _ in range(args.train)]
    probe = [(rng.randint(1, 255), rng.randint(1, 255)) for _ in range(args.probe)]
    cache_train: dict[tuple[int, ...], tuple[int, int, int]] = {}
    cache_probe: dict[tuple[int, ...], tuple[int, int, int]] = {}

    def fit_train(code: list[int]) -> tuple[int, int, int]:
        k = tuple(code)
        if k not in cache_train:
            ok, halted = score(code, train, args.max_steps)
            cache_train[k] = (ok, halted, -len(code))
        return cache_train[k]

    def fit_probe(code: list[int]) -> tuple[int, int, int]:
        k = tuple(code)
        if k not in cache_probe:
            ok, halted = score(code, probe, args.max_steps)
            cache_probe[k] = (ok, halted, -len(code))
        return cache_probe[k]

    pop = [
        random_candidate(rng, physics_ops, ml_ops, sci_neg, args.max_len)
        for _ in range(args.pop)
    ]
    scores = {tuple(c): fit_train(c) for c in pop}

    best = max(pop, key=lambda c: scores[tuple(c)])
    best_probe = fit_probe(best)
    print(
        "init",
        scores[tuple(best)],
        best_probe,
        len(best),
        encode_program(best),
        flush=True,
    )

    for it in range(1, args.iters + 1):
        ranked = sorted(
            {tuple(c): c for c in pop}.values(),
            key=lambda c: scores[tuple(c)],
            reverse=True,
        )
        nxt = ranked[: max(4, args.pop // 3)]
        while len(nxt) < args.pop:
            parent = rng.choice(ranked[: max(6, args.pop // 2)])
            child = mutate(rng, parent, physics_ops, ml_ops, sci_neg, args.max_len)
            t = tuple(child)
            if t not in scores:
                scores[t] = fit_train(child)
            nxt.append(child)
        pop = nxt

        if it % 200 == 0:
            top = sorted(
                {tuple(c): c for c in pop}.values(),
                key=lambda c: scores[tuple(c)],
                reverse=True,
            )[:8]
            improved = False
            for c in top:
                pv = fit_probe(c)
                if pv > best_probe:
                    best = c[:]
                    best_probe = pv
                    improved = True
            if improved:
                p = best_probe[0] / args.probe
                print(
                    "iter",
                    it,
                    "probe",
                    best_probe,
                    "p",
                    f"{p:.6f}",
                    "len",
                    len(best),
                    encode_program(best),
                    flush=True,
                )

    p = best_probe[0] / args.probe
    print("final probe:", best_probe, "p=", f"{p:.6f}", "len=", len(best))
    print("best hex:", encode_program(best))


if __name__ == "__main__":
    main()
