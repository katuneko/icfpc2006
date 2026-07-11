#!/usr/bin/env python3
"""Reproduce local lower-bound checks for short Balance solutions.

This does not prove that every possible copymem program needs 32 bytes.  It
does prove the two smaller claims used when deciding whether to continue the
current shortening approaches:

* clearreg needs at least seven PHYSICS operations to zero its six registers;
* copymem's current loop-register transformation needs at least eight PHYSICS
  operations.
"""

from __future__ import annotations

from collections.abc import Callable

from balance_solver import _physics_step


State = tuple[int, int, int, int, int, int]
MultiState = tuple[State, ...]


IMMEDIATES = tuple(raw - 32 if raw & 16 else raw for raw in range(32))


def inverse_physics(state: State, imm: int) -> State:
    """Invert one PHYSICS operation."""
    new = list(state)
    old = new[:]
    # The rotation order in balance_solver is sR0,dR1,dR0,sR3,sR2,sR1.
    register_indices = (5, 4, 3, 2, 1)
    selected = [
        index
        for bit, index in enumerate(register_indices)
        if (imm & 0x1F) >> bit & 1
    ]
    if not selected:
        old[0] = (new[0] - imm) & 0xFF
    else:
        old[0] = (new[selected[0]] - imm) & 0xFF
        for current, following in zip(selected, selected[1:]):
            old[current] = new[following]
        old[selected[-1]] = new[0]
    return tuple(old)  # type: ignore[return-value]


def transform(states: MultiState, imm: int) -> MultiState:
    return tuple(_physics_step(state, imm) for state in states)


def inverse_transform(states: MultiState, imm: int) -> MultiState:
    return tuple(inverse_physics(state, imm) for state in states)


def reachable(
    start: MultiState,
    depth: int,
    operation: Callable[[MultiState, int], MultiState],
) -> set[MultiState]:
    seen = {start}
    frontier = {start}
    for _ in range(depth):
        following = {
            operation(states, imm)
            for states in frontier
            for imm in IMMEDIATES
        }
        following -= seen
        seen |= following
        frontier = following
    return seen


def has_path(start: MultiState, target: MultiState, forward: int, backward: int) -> bool:
    from_start = reachable(start, forward, transform)
    from_target = reachable(target, backward, inverse_transform)
    return not from_start.isdisjoint(from_target)


def main() -> None:
    clear_start = ((0, 1, 2, 3, 4, 5),)
    clear_target = ((0, 0, 0, 0, 0, 0),)
    clear_under_seven = has_path(clear_start, clear_target, 3, 3)
    print("clearreg PHYSICS length <= 6:", clear_under_seven)
    assert not clear_under_seven

    # Three independent samples prevent accidental agreement for only one
    # iteration of the affine loop-register transformation.
    copy_start = (
        (0, 2, 0, 243, 0, 3),
        (0, 2, 17, 91, 0, 211),
        (0, 2, 255, 7, 0, 250),
    )
    copy_target = tuple(
        (0, 2, (state[2] + 1) & 0xFF, (state[3] + 5) & 0xFF, 0,
         (state[5] + 6) & 0xFF)
        for state in copy_start
    )
    copy_under_eight = has_path(copy_start, copy_target, 4, 3)
    print("copymem loop PHYSICS length <= 7:", copy_under_eight)
    assert not copy_under_eight


if __name__ == "__main__":
    main()
