#!/usr/bin/env python3
"""Inspect local Antomaton candidates with P6-specific event reporting."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List, Tuple

import ant_solver as ant
import ant_local_search as local


def cell_text(value: int) -> str:
    return ant.cell_to_token(local.value_to_cell(value)).strip()


def render_top(state: List[int], width: int, y0: int = 1, y1: int = 4) -> str:
    lines = []
    for y in range(y0, y1):
        cells = [cell_text(state[y * width + x]).rjust(2) for x in range(width)]
        lines.append(f"y={y:02d} " + " ".join(cells))
    return "\n".join(lines)


def p6_events(
    state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    programs: List[List[int]],
    width: int,
) -> List[str]:
    events: List[str] = []
    for pos, value in enumerate(state):
        if not ant.is_ant(value):
            continue
        y, x = divmod(pos, width)
        clan = ant.ant_clan(value)
        direction = ant.ant_dir(value)
        glyph = ant.DIR_TO_ANT_CHAR[direction]
        if direction == 1 and 1 <= y <= 3 and 1 <= x <= 9:
            ahead = neighbors[pos][1]
            ahead_value = state[ahead] if ahead != -1 else ant.CELL_WALL
            events.append(
                f"east ({x},{y}) {clan}{glyph} ahead={cell_text(ahead_value)}"
            )
        if direction == 0 and 2 <= y <= 4 and 1 <= x <= 9:
            ahead = neighbors[pos][0]
            ahead_value = state[ahead] if ahead != -1 else ant.CELL_WALL
            right = neighbors[pos][1]
            right_value = state[right] if right != -1 else ant.CELL_WALL
            if ant.is_ant(ahead_value):
                rel = (ant.ant_dir(ahead_value) - direction) & 3
                turn = ant.DIRS[programs[clan][3 + rel]]
                events.append(
                    f"blocker ({x},{y}) {clan}{glyph} "
                    f"ahead={cell_text(ahead_value)} rel={rel} turn={turn} "
                    f"right={cell_text(right_value)}"
                )
    return events


def simulate(
    puzzle: ant.Puzzle,
    candidate: local.Candidate,
    wilds: List[int],
    max_steps: int,
    trace: bool,
    trace_all: bool,
    row_start: int,
    row_end: int,
    emit: bool = True,
) -> bool:
    neighbors = ant.build_neighbors(puzzle.width, puzzle.height)
    base_state, _wilds, food = ant.build_base_state(puzzle)
    state = local.build_state(puzzle, wilds, candidate)
    seen = set()
    best_events: Tuple[int, List[str], List[int]] | None = None
    for step in range(max_steps + 1):
        if ant.has_success_fast(state, neighbors, food, "facing"):
            if emit:
                print(f"SUCCESS step={step}")
                print(render_top(state, puzzle.width, row_start, row_end))
            return True
        events = p6_events(state, neighbors, candidate.programs, puzzle.width)
        if events and (best_events is None or event_rank(events) < event_rank(best_events[1])):
            best_events = (step, events, state[:])
        if emit and trace and (events or trace_all):
            print(f"STEP {step}")
            for event in events:
                print(f"  {event}")
            print(render_top(state, puzzle.width, row_start, row_end))
        key = bytes(state)
        if key in seen:
            break
        seen.add(key)
        state = ant.step_state(state, neighbors, candidate.programs)
    if emit and best_events is not None:
        step, events, best_state = best_events
        print(f"BEST_EVENT step={step}")
        for event in events:
            print(f"  {event}")
        print(render_top(best_state, puzzle.width, row_start, row_end))
    if emit:
        print("NO_SUCCESS")
    return False


def event_rank(events: Iterable[str]) -> int:
    best = 10000
    for event in events:
        if event.startswith("east ") and ("ahead=-" in event or "ahead=$" in event):
            best = min(best, 0)
        elif "turn=E" in event and ("right=-" in event or "right=$" in event):
            best = min(best, 5)
        elif event.startswith("east "):
            best = min(best, 20)
        elif "right=-" in event or "right=$" in event:
            best = min(best, 25)
        elif "turn=E" in event:
            best = min(best, 30)
        else:
            best = min(best, 40)
    return best


def parse_program_set(text: str) -> Tuple[int, int, int]:
    lhs, sep, rhs = text.partition("=")
    if not sep or rhs not in ant.DIR_TO_IDX:
        raise ValueError(f"invalid program set: {text!r}")
    clan_text, slot_text = lhs.split(":", 1)
    if not slot_text.startswith("p"):
        raise ValueError(f"invalid program slot: {text!r}")
    return int(clan_text), int(slot_text[1:]) - 1, ant.DIR_TO_IDX[rhs]


def try_one_program(
    puzzle: ant.Puzzle,
    candidate: local.Candidate,
    wilds: List[int],
    max_steps: int,
) -> None:
    found = 0
    for clan, row in enumerate(candidate.programs):
        for idx, old in enumerate(row):
            for new in range(4):
                if new == old:
                    continue
                probe = local.clone_candidate(candidate)
                probe.programs[clan][idx] = new
                ok = simulate(
                    puzzle,
                    probe,
                    wilds,
                    max_steps,
                    trace=False,
                    trace_all=False,
                    row_start=1,
                    row_end=4,
                    emit=False,
                )
                if ok:
                    print(f"HIT program {clan}:p{idx + 1}={ant.DIRS[new]}")
                    found += 1
    if found == 0:
        print("NO_ONE_PROGRAM_HIT")


def parse_grid_values(text: str) -> List[int]:
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.extend(ant.parse_grid_wild_values(part))
    out: List[int] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def parse_ant_clans(text: str, program_count: int) -> List[int]:
    if not text:
        return list(range(program_count))
    return [int(part) for part in text.split(",") if part.strip()]


def try_one_grid(
    puzzle: ant.Puzzle,
    candidate: local.Candidate,
    wilds: List[int],
    max_steps: int,
    values: List[int],
) -> None:
    found = 0
    for grid_idx, old in enumerate(candidate.grid):
        for new in values:
            if new == old:
                continue
            probe = local.clone_candidate(candidate)
            probe.grid[grid_idx] = new
            ok = simulate(
                puzzle,
                probe,
                wilds,
                max_steps,
                trace=False,
                trace_all=False,
                row_start=1,
                row_end=4,
                emit=False,
            )
            if ok:
                pos = wilds[grid_idx]
                y, x = divmod(pos, puzzle.width)
                print(f"HIT grid {x},{y}={cell_text(new)}")
                found += 1
    if found == 0:
        print("NO_ONE_GRID_HIT")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--puzzles", default="volume9_gardener_puzzles.txt")
    parser.add_argument("--match", default="Puzzle 6")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-all", action="store_true")
    parser.add_argument("--row-start", type=int, default=1)
    parser.add_argument("--row-end", type=int, default=4)
    parser.add_argument("--set-program", action="append", default=[])
    parser.add_argument("--try-one-program", action="store_true")
    parser.add_argument("--try-one-grid", action="store_true")
    parser.add_argument("--grid-values", default="floor,wall,hole")
    parser.add_argument("--ant-clans", default="")
    args = parser.parse_args()

    puzzles = ant.parse_puzzles(Path(args.puzzles))
    matches = [puzzle for puzzle in puzzles if args.match in puzzle.title]
    if len(matches) != 1:
        raise SystemExit(f"expected one puzzle, found {len(matches)}")
    puzzle = matches[0]
    wilds = ant.find_wildcards(puzzle)
    candidate = local.read_candidate(Path(args.candidate), puzzle, wilds)
    for text in args.set_program:
        clan, idx, value = parse_program_set(text)
        candidate.programs[clan][idx] = value
    if args.try_one_program:
        try_one_program(puzzle, candidate, wilds, args.max_steps)
    elif args.try_one_grid:
        values = parse_grid_values(args.grid_values)
        clans = parse_ant_clans(args.ant_clans, len(candidate.programs))
        values.extend(
            ant.make_ant(clan, direction) for clan in clans for direction in range(4)
        )
        try_one_grid(puzzle, candidate, wilds, args.max_steps, values)
    else:
        simulate(
            puzzle,
            candidate,
            wilds,
            args.max_steps,
            args.trace,
            args.trace_all,
            args.row_start,
            args.row_end,
        )


if __name__ == "__main__":
    main()
