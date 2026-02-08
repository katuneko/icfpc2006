#!/usr/bin/env python3
"""Deductive PHYSICS macro search for Balance.

This helper reproduces several structural results used in fillmem/multmem work:
1) strict dR0++ with all other registers preserved has no solution in <= 8 steps
2) dR0++ macro exists in 7 steps if sR1/sR3 are allowed to swap
3) dR1++ macro exists in 8 steps with the same swap relaxation
4) preloop -> main-loop register remap exists in 8 steps (for varying dR0)
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Set, Tuple


IMMS = tuple((i - 32 if i & 0x10 else i) for i in range(32))

# Register order: sR0, sR1, sR2, sR3, dR0, dR1
REG_COUNT = 6


SymbolicReg = Tuple[int, int]  # (source_index 0..5, offset 0..255)
SymbolicState = Tuple[SymbolicReg, SymbolicReg, SymbolicReg, SymbolicReg, SymbolicReg, SymbolicReg]
ConcreteState = Tuple[int, int, int, int, int, int]
ConcreteSampleState = Tuple[ConcreteState, ...]


def step_symbolic(state: SymbolicState, imm: int) -> SymbolicState:
    regs = [list(x) for x in state]
    regs[0][1] = (regs[0][1] + imm) & 0xFF
    mask = imm & 0x1F
    order = [5, 4, 3, 2, 1]  # dR1, dR0, sR3, sR2, sR1
    c = [idx for bit, idx in enumerate(order) if (mask >> bit) & 1]
    if c:
        seq = [0] + c
        old = [tuple(regs[i]) for i in seq]
        for dest, val in zip(seq[1:], old[:-1]):
            regs[dest][0], regs[dest][1] = val
        regs[0][0], regs[0][1] = old[-1]
    return tuple((a, b) for a, b in regs)  # type: ignore[return-value]


def inv_symbolic(state: SymbolicState, imm: int) -> SymbolicState:
    regs = [list(x) for x in state]
    mask = imm & 0x1F
    order = [5, 4, 3, 2, 1]
    c = [idx for bit, idx in enumerate(order) if (mask >> bit) & 1]
    if c:
        sr0_inc = tuple(regs[c[0]])
        old_c: List[Tuple[int, int]] = [(-1, -1)] * len(c)
        for i in range(len(c) - 1):
            old_c[i] = tuple(regs[c[i + 1]])  # type: ignore[assignment]
        old_c[-1] = tuple(regs[0])  # type: ignore[assignment]
        for idx, val in zip(c, old_c):
            regs[idx][0], regs[idx][1] = val
        regs[0][0], regs[0][1] = sr0_inc
    regs[0][1] = (regs[0][1] - imm) & 0xFF
    return tuple((a, b) for a, b in regs)  # type: ignore[return-value]


def step_concrete(state: ConcreteState, imm: int) -> ConcreteState:
    sr0, sr1, sr2, sr3, dr0, dr1 = state
    sr0 = (sr0 + imm) & 0xFF
    regs = [sr0, sr1, sr2, sr3, dr0, dr1]
    mask = imm & 0x1F
    order = [5, 4, 3, 2, 1]
    c = [idx for bit, idx in enumerate(order) if (mask >> bit) & 1]
    if c:
        seq = [0] + c
        old = [regs[i] for i in seq]
        for dest, val in zip(seq[1:], old[:-1]):
            regs[dest] = val
        regs[0] = old[-1]
    return tuple(regs)  # type: ignore[return-value]


def inv_concrete(state: ConcreteState, imm: int) -> ConcreteState:
    regs = list(state)
    mask = imm & 0x1F
    order = [5, 4, 3, 2, 1]
    c = [idx for bit, idx in enumerate(order) if (mask >> bit) & 1]
    if c:
        sr0_inc = regs[c[0]]
        old_c = [0] * len(c)
        for i in range(len(c) - 1):
            old_c[i] = regs[c[i + 1]]
        old_c[-1] = regs[0]
        for idx, val in zip(c, old_c):
            regs[idx] = val
        regs[0] = sr0_inc
    regs[0] = (regs[0] - imm) & 0xFF
    return tuple(regs)  # type: ignore[return-value]


def bidir_search_symbolic(
    start: SymbolicState,
    targets: Set[SymbolicState],
    max_half_depth: int = 4,
) -> Optional[List[int]]:
    f_parent: Dict[SymbolicState, Tuple[Optional[SymbolicState], Optional[int]]] = {
        start: (None, None)
    }
    frontier: Set[SymbolicState] = {start}
    for _ in range(max_half_depth):
        nxt: Set[SymbolicState] = set()
        for st in frontier:
            for imm in IMMS:
                ns = step_symbolic(st, imm)
                if ns in f_parent:
                    continue
                f_parent[ns] = (st, imm)
                nxt.add(ns)
        frontier = nxt

    b_parent: Dict[SymbolicState, Tuple[Optional[SymbolicState], Optional[int]]] = {
        t: (None, None) for t in targets
    }
    back: Set[SymbolicState] = set(targets)
    meet: Optional[SymbolicState] = None
    for _ in range(max_half_depth + 1):
        common = back.intersection(f_parent.keys())
        if common:
            meet = next(iter(common))
            break
        nxt: Set[SymbolicState] = set()
        for st in back:
            for imm in IMMS:
                ps = inv_symbolic(st, imm)
                if ps in b_parent:
                    continue
                b_parent[ps] = (st, imm)
                nxt.add(ps)
        back = nxt

    if meet is None:
        return None

    path1: List[int] = []
    cur = meet
    while cur != start:
        prev, imm = f_parent[cur]
        assert prev is not None and imm is not None
        path1.append(imm)
        cur = prev
    path1.reverse()

    path2: List[int] = []
    cur = meet
    while cur not in targets:
        nxt, imm = b_parent[cur]
        assert nxt is not None and imm is not None
        path2.append(imm)
        cur = nxt

    return path1 + path2


def bidir_search_samples(
    start: ConcreteSampleState,
    target: ConcreteSampleState,
    max_half_depth: int = 4,
) -> Optional[List[int]]:
    f_parent: Dict[ConcreteSampleState, Tuple[Optional[ConcreteSampleState], Optional[int]]] = {
        start: (None, None)
    }
    frontier: Set[ConcreteSampleState] = {start}
    for _ in range(max_half_depth):
        nxt: Set[ConcreteSampleState] = set()
        for st in frontier:
            for imm in IMMS:
                ns = tuple(step_concrete(s, imm) for s in st)
                if ns in f_parent:
                    continue
                f_parent[ns] = (st, imm)
                nxt.add(ns)
        frontier = nxt

    b_parent: Dict[ConcreteSampleState, Tuple[Optional[ConcreteSampleState], Optional[int]]] = {
        target: (None, None)
    }
    back: Set[ConcreteSampleState] = {target}
    meet: Optional[ConcreteSampleState] = None
    for _ in range(max_half_depth + 1):
        common = back.intersection(f_parent.keys())
        if common:
            meet = next(iter(common))
            break
        nxt: Set[ConcreteSampleState] = set()
        for st in back:
            for imm in IMMS:
                ps = tuple(inv_concrete(s, imm) for s in st)
                if ps in b_parent:
                    continue
                b_parent[ps] = (st, imm)
                nxt.add(ps)
        back = nxt

    if meet is None:
        return None

    path1: List[int] = []
    cur = meet
    while cur != start:
        prev, imm = f_parent[cur]
        assert prev is not None and imm is not None
        path1.append(imm)
        cur = prev
    path1.reverse()

    path2: List[int] = []
    cur = meet
    while cur != target:
        nxt, imm = b_parent[cur]
        assert nxt is not None and imm is not None
        path2.append(imm)
        cur = nxt

    return path1 + path2


def apply_symbolic(state: SymbolicState, seq: Sequence[int]) -> SymbolicState:
    cur = state
    for imm in seq:
        cur = step_symbolic(cur, imm)
    return cur


def apply_concrete(state: ConcreteState, seq: Sequence[int]) -> ConcreteState:
    cur = state
    for imm in seq:
        cur = step_concrete(cur, imm)
    return cur


def scenario_strict_dr0(max_half_depth: int) -> None:
    start: SymbolicState = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0))
    target: SymbolicState = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 0))
    seq = bidir_search_symbolic(start, {target}, max_half_depth=max_half_depth)
    print("strict dr0++ <=8:", seq)


def scenario_dr0_swap(max_half_depth: int) -> None:
    # Preserve sR0,sR2,dR1 and increment dR0, allowing sR1/sR3 swap.
    start: SymbolicState = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 0))
    targets: Set[SymbolicState] = {
        ((0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (0, 0)),
        ((0, 0), (3, 0), (2, 0), (1, 0), (4, 1), (0, 0)),
    }
    seq = bidir_search_symbolic(start, targets, max_half_depth=max_half_depth)
    print("dr0++ swap macro:", seq)
    if seq:
        out = apply_symbolic(start, seq)
        print("dr0++ swap out:", out)


def scenario_dr1_swap(max_half_depth: int) -> None:
    # Preserve sR0,sR2,dR0 and increment dR1, allowing sR1/sR3 swap.
    start: SymbolicState = ((0, 0), (1, 0), (2, 0), (3, 0), (0, 0), (4, 0))
    targets: Set[SymbolicState] = {
        ((0, 0), (1, 0), (2, 0), (3, 0), (0, 0), (4, 1)),
        ((0, 0), (3, 0), (2, 0), (1, 0), (0, 0), (4, 1)),
    }
    seq = bidir_search_symbolic(start, targets, max_half_depth=max_half_depth)
    print("dr1++ swap macro:", seq)
    if seq:
        out = apply_symbolic(start, seq)
        print("dr1++ swap out:", out)


def scenario_pre_to_main(max_half_depth: int) -> None:
    # Preloop end -> main-loop start remap, preserving dR0 across sample values.
    # Start sample: (1,3,4,3,dR0,1)
    # Target sample: (5,0,4,3,dR0,5)
    samples = (8, 17, 123, 250)
    start: ConcreteSampleState = tuple((1, 3, 4, 3, dr0, 1) for dr0 in samples)
    target: ConcreteSampleState = tuple((5, 0, 4, 3, dr0, 5) for dr0 in samples)
    seq = bidir_search_samples(start, target, max_half_depth=max_half_depth)
    print("pre->main remap:", seq)
    if seq:
        for dr0 in samples:
            out = apply_concrete((1, 3, 4, 3, dr0, 1), seq)
            print(" sample", dr0, "->", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("strict_dr0", "dr0_swap", "dr1_swap", "pre_to_main"),
        help="run one or more scenarios; default runs dr0_swap/dr1_swap/pre_to_main",
    )
    parser.add_argument(
        "--max-half-depth",
        type=int,
        default=4,
        help="bidirectional half depth (<=8 search when set to 4)",
    )
    args = parser.parse_args()

    scenarios = args.scenario or ["dr0_swap", "dr1_swap", "pre_to_main"]
    for name in scenarios:
        if name == "strict_dr0":
            scenario_strict_dr0(args.max_half_depth)
        elif name == "dr0_swap":
            scenario_dr0_swap(args.max_half_depth)
        elif name == "dr1_swap":
            scenario_dr1_swap(args.max_half_depth)
        elif name == "pre_to_main":
            scenario_pre_to_main(args.max_half_depth)


if __name__ == "__main__":
    main()
