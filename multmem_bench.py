#!/usr/bin/env python3
"""Benchmark Balance multmem candidates.

Examples:
  python3 multmem_bench.py --hex 54007d --samples 20000
  python3 multmem_bench.py --file multmem_candidate.bal --samples 100000
"""

import argparse
import random

from balance_solver import (
    BalanceMachine,
    check_puzzle,
    decode_program,
    puzzle_state,
)


def load_code(args: argparse.Namespace) -> list[int]:
    if args.hex:
        return decode_program(args.hex)
    with open(args.file, "r", encoding="ascii") as f:
        return decode_program(f.read().strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hex", help="program hex string")
    parser.add_argument(
        "--file", default="multmem_candidate.bal", help="path to .bal hex file"
    )
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    code = load_code(args)

    random.seed(args.seed)
    ok = 0
    halted = 0
    for _ in range(args.samples):
        a = random.randint(1, 255)
        b = random.randint(1, 255)
        mem, sR, dR = puzzle_state("multmem", (a, b))
        machine = BalanceMachine(code, mem, sR, dR)
        did_halt, _ = machine.run(max_steps=args.max_steps)
        if did_halt:
            halted += 1
            if check_puzzle("multmem", machine, (a, b)):
                ok += 1

    p = ok / args.samples
    pass5 = p**5
    print("hex:", "".join(f"{b:02x}" for b in code))
    print("ok:", ok, "/", args.samples, f"({p:.6f})")
    print("halted:", halted, "/", args.samples)
    print("estimated certify pass rate (5 tests):", f"{pass5:.12f}")


if __name__ == "__main__":
    main()
