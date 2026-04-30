#!/usr/bin/env python3
"""Local stochastic search for stubborn Antomaton boards.

This is intentionally separate from ant_solver.py's exact/symbolic helpers:
it mutates concrete programs and wildcard contents, scores a full simulation,
and prints complete candidate worlds when it reaches the food.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import random
import re
from time import monotonic
from typing import Iterable, List, Optional, Tuple

import ant_solver as ant


GridChoice = int


@dataclass
class Candidate:
    programs: List[List[int]]
    grid: List[GridChoice]


@dataclass
class EvalResult:
    score: int
    ok: bool
    step: int
    best_detail: str


def parse_program_templates(puzzle: ant.Puzzle) -> List[List[Optional[int]]]:
    templates: List[List[Optional[int]]] = []
    for line in puzzle.programs:
        row: List[Optional[int]] = []
        for ch in line[:7]:
            row.append(None if ch == "*" else ant.DIR_TO_IDX[ch])
        templates.append(row)
    return templates


def apply_fixed_program_option(
    templates: List[List[Optional[int]]],
    option: Optional[str],
) -> None:
    if not option:
        return
    for raw_part in option.split(","):
        part = raw_part.strip()
        if not part:
            continue
        lhs, sep, rhs = part.partition("=")
        if not sep or rhs not in ant.DIR_TO_IDX:
            raise ValueError(f"invalid fixed program option: {part!r}")
        value = ant.DIR_TO_IDX[rhs]
        if lhs.startswith("p") and lhs[1:].isdigit():
            idx = int(lhs[1:]) - 1
            if idx < 0 or idx >= 7:
                raise ValueError(f"program slot out of range: {lhs!r}")
            for row in templates:
                row[idx] = value
            continue
        if ":" in lhs:
            clan_text, slot_text = lhs.split(":", 1)
            if not clan_text.isdigit() or not slot_text.startswith("p"):
                raise ValueError(f"invalid fixed program option: {part!r}")
            clan = int(clan_text)
            idx = int(slot_text[1:]) - 1
            if clan < 0 or clan >= len(templates) or idx < 0 or idx >= 7:
                raise ValueError(f"program slot out of range: {lhs!r}")
            templates[clan][idx] = value
            continue
        raise ValueError(f"invalid fixed program option: {part!r}")


def fixed_program_entries(templates: List[List[Optional[int]]]) -> set[Tuple[int, int]]:
    fixed = set()
    for clan, row in enumerate(templates):
        for idx, value in enumerate(row):
            if value is not None:
                fixed.add((clan, idx))
    return fixed


def render_programs(programs: List[List[int]]) -> List[str]:
    return ["".join(ant.DIRS[v] for v in row) for row in programs]


def value_to_cell(value: int) -> ant.Cell:
    if value == ant.CELL_WALL:
        return ant.Cell(ant.KIND_WALL)
    if value == ant.CELL_HOLE:
        return ant.Cell(ant.KIND_HOLE)
    if value == ant.CELL_FLOOR:
        return ant.Cell(ant.KIND_FLOOR)
    if value == ant.CELL_FOOD:
        return ant.Cell(ant.KIND_FOOD)
    if ant.is_ant(value):
        return ant.Cell(ant.KIND_ANT, ant.ant_clan(value), ant.ant_dir(value))
    raise ValueError(f"unsupported cell value: {value}")


def cell_to_value(cell: ant.Cell) -> int:
    if cell.kind == ant.KIND_WALL:
        return ant.CELL_WALL
    if cell.kind == ant.KIND_HOLE:
        return ant.CELL_HOLE
    if cell.kind == ant.KIND_FLOOR:
        return ant.CELL_FLOOR
    if cell.kind == ant.KIND_FOOD:
        return ant.CELL_FOOD
    if cell.kind == ant.KIND_ANT:
        return ant.make_ant(cell.clan, cell.direction)
    raise ValueError(f"unsupported cell kind: {cell}")


def build_state(
    puzzle: ant.Puzzle,
    wilds: List[int],
    cand: Candidate,
) -> List[int]:
    state, _wilds, _food = ant.build_base_state_with_wildcards(puzzle)
    for idx, value in zip(wilds, cand.grid):
        state[idx] = value
    return state


def target_cells(
    base_state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    success_mode: str,
) -> set[Tuple[int, int]]:
    targets: set[Tuple[int, int]] = set()
    for pos, value in enumerate(base_state):
        if value != ant.CELL_FOOD:
            continue
        for direction, src in enumerate(neighbors[pos]):
            if src == -1:
                continue
            if base_state[src] in (ant.CELL_WALL, ant.CELL_HOLE, ant.CELL_FOOD):
                continue
            if success_mode == "below" and direction != 2:
                continue
            # direction here is from food to source; ant at source must face food.
            face = (direction + 2) & 3
            targets.add((src, face))
    return targets


def optimistic_distances(
    puzzle: ant.Puzzle,
    targets: Iterable[Tuple[int, int]],
) -> List[Optional[int]]:
    base_state, _wilds, _food = ant.build_base_state(puzzle)
    neighbors = ant.build_neighbors(puzzle.width, puzzle.height)
    target_positions = sorted({pos for pos, _direction in targets})
    return ant.compute_distances_to_targets(base_state, neighbors, target_positions)


def state_score(
    state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    programs: List[List[int]],
    targets: set[Tuple[int, int]],
    dist: List[Optional[int]],
    manhattan_weight: int,
    direction_penalty_value: int,
    distance_weight: int,
    support_target_turn: bool,
    support_weight: int,
    support_near_target_turn: bool,
    support_max_distance: int,
    support_program_aware: bool,
    support_entry_turn: bool,
    p6_focus: bool,
) -> Tuple[int, str]:
    best = 10**9
    detail = "no ants"
    live_ants = [(pos, value) for pos, value in enumerate(state) if ant.is_ant(value)]
    live = len(live_ants)
    for pos, value in live_ants:
        y, x = divmod(pos, WIDTH_HINT)
        if p6_focus:
            p6_score = p6_focused_score(state, neighbors, live_ants, programs, pos, value)
            if p6_score is not None and p6_score[0] < best:
                best, detail = p6_score
        d = dist[pos]
        dist_hint = 1000 if d is None else d * distance_weight
        for target_pos, target_dir in targets:
            ty, tx = divmod(target_pos, WIDTH_HINT)
            manhattan = abs(tx - x) + abs(ty - y)
            direction_penalty = (
                0 if ant.ant_dir(value) == target_dir else direction_penalty_value
            )
            score = manhattan * manhattan_weight + dist_hint + direction_penalty
            if score < best:
                best = score
                detail = (
                    f"ant=({x},{y}) {ant.ant_clan(value)}"
                    f"{ant.DIR_TO_ANT_CHAR[ant.ant_dir(value)]} "
                    f"target=({tx},{ty}) {ant.DIRS[target_dir]} "
                    f"dist={d} manhattan={manhattan}"
                )
            if support_target_turn and pos == target_pos and ant.ant_dir(value) != target_dir:
                ahead = neighbors[pos][ant.ant_dir(value)]
                if ahead != -1:
                    ay, ax = divmod(ahead, WIDTH_HINT)
                    good_dirs = good_ahead_dirs(
                        programs, value, target_dir, support_program_aware
                    )
                    support_dist = support_distance(
                        state, live_ants, pos, ahead, good_dirs
                    )
                    support_score = support_dist * support_weight
                    if support_score < best:
                        best = support_score
                        detail = (
                            f"turn-support ant=({x},{y}) {ant.ant_clan(value)}"
                            f"{ant.DIR_TO_ANT_CHAR[ant.ant_dir(value)]} "
                            f"target=({tx},{ty}) {ant.DIRS[target_dir]} "
                            f"ahead=({ax},{ay}) support_dist={support_dist}"
                        )
            if (
                support_near_target_turn
                and ant.ant_dir(value) != target_dir
                and d is not None
                and d <= support_max_distance
            ):
                ahead = neighbors[pos][ant.ant_dir(value)]
                if ahead != -1:
                    ay, ax = divmod(ahead, WIDTH_HINT)
                    good_dirs = good_ahead_dirs(
                        programs, value, target_dir, support_program_aware
                    )
                    support_dist = support_distance(
                        state, live_ants, pos, ahead, good_dirs
                    )
                    near_score = (
                        d * distance_weight
                        + manhattan * manhattan_weight
                        + support_dist * support_weight
                    )
                    if near_score < best:
                        best = near_score
                        detail = (
                            f"near-turn-support ant=({x},{y}) {ant.ant_clan(value)}"
                            f"{ant.DIR_TO_ANT_CHAR[ant.ant_dir(value)]} "
                            f"target=({tx},{ty}) {ant.DIRS[target_dir]} "
                            f"dist={d} ahead=({ax},{ay}) support_dist={support_dist}"
                        )
            if (
                support_entry_turn
                and ant.ant_dir(value) != target_dir
                and d is not None
                and d <= support_max_distance
            ):
                direction = ant.ant_dir(value)
                ahead = neighbors[pos][direction]
                if ahead != -1 and state[ahead] == ant.CELL_FLOOR:
                    rel_turn = (target_dir - direction) & 3
                    clan = ant.ant_clan(value)
                    ay, ax = divmod(ahead, WIDTH_HINT)
                    best_entry: Optional[Tuple[str, int]] = None
                    if programs[clan][1] == rel_turn:
                        left_pos = neighbors[ahead][(direction + 3) & 3]
                        if left_pos != -1:
                            left_dir = (direction + 1) & 3
                            best_entry = (
                                "p2",
                                directed_support_distance(
                                    live_ants, pos, left_pos, left_dir
                                ),
                            )
                    if programs[clan][2] == rel_turn:
                        left_pos = neighbors[ahead][(direction + 3) & 3]
                        right_pos = neighbors[ahead][(direction + 1) & 3]
                        if left_pos != -1 and right_pos != -1:
                            left_dir = (direction + 1) & 3
                            right_dir = (direction + 3) & 3
                            p3_dist = directed_support_distance(
                                live_ants, pos, left_pos, left_dir
                            ) + directed_support_distance(
                                live_ants, pos, right_pos, right_dir
                            )
                            if best_entry is None or p3_dist < best_entry[1]:
                                best_entry = ("p3", p3_dist)
                    if best_entry is not None:
                        entry_kind, entry_dist = best_entry
                        entry_score = (
                            d * distance_weight
                            + manhattan * manhattan_weight
                            + entry_dist * support_weight
                        )
                        if entry_score < best:
                            best = entry_score
                            detail = (
                                f"entry-turn-{entry_kind} ant=({x},{y}) "
                                f"{ant.ant_clan(value)}"
                                f"{ant.DIR_TO_ANT_CHAR[direction]} "
                                f"target=({tx},{ty}) {ant.DIRS[target_dir]} "
                                f"dest=({ax},{ay}) support_dist={entry_dist}"
                            )
    if live == 0:
        return 1000000, detail
    return best - min(live, 20), detail


WIDTH_HINT = 1


def p6_focused_score(
    state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    live_ants: List[Tuple[int, int]],
    programs: List[List[int]],
    pos: int,
    value: int,
) -> Optional[Tuple[int, str]]:
    """Reward P6's real bottleneck: making an east signal in rows 1..3."""
    y, x = divmod(pos, WIDTH_HINT)
    direction = ant.ant_dir(value)
    clan = ant.ant_clan(value)
    # Once an east-facing ant exists in the top corridor, only horizontal progress
    # remains. This should dominate generic "north and close" scores.
    if direction == 1 and 1 <= y <= 3 and 1 <= x <= 9:
        p1_penalty = 0 if programs[clan][0] == 0 else 180
        ahead = neighbors[pos][1]
        ahead_value = state[ahead] if ahead != -1 else ant.CELL_WALL
        wait_penalty = 0
        if ahead_value in (ant.CELL_FLOOR, ant.CELL_FOOD):
            clear_penalty = 0
        elif ant.is_ant(ahead_value):
            rel = (ant.ant_dir(ahead_value) - direction) & 3
            wait_penalty = 0 if programs[clan][3 + rel] == 0 else 90
            clear_penalty = 35 + wait_penalty
        else:
            clear_penalty = 140
        return (
            (9 - x) * 20 + abs(y - 2) * 6 + p1_penalty + clear_penalty,
            f"p6-east-signal ant=({x},{y}) {clan}> "
            f"clear_penalty={clear_penalty} wait_penalty={wait_penalty}",
        )
    # Exact p2 entry-turn setup: north-moving ant enters the cell whose west
    # neighbor is already an east signal, producing a new east signal.
    if direction == 0 and 2 <= y <= 4 and 1 <= x <= 9:
        ahead = neighbors[pos][0]
        if ahead != -1 and state[ahead] == ant.CELL_FLOOR:
            ay, ax = divmod(ahead, WIDTH_HINT)
            left = neighbors[ahead][3]
            if left != -1:
                support = directed_support_distance(live_ants, pos, left, 1)
                p2_penalty = 0 if programs[clan][1] == 1 else 80
                p1_penalty = 0 if programs[clan][0] == 0 else 180
                score = (
                    80
                    + (9 - ax) * 16
                    + abs(ay - 2) * 8
                    + support * 25
                    + p1_penalty
                    + p2_penalty
                )
                return (
                    score,
                    f"p6-p2-prep ant=({x},{y}) {clan}^ dest=({ax},{ay}) "
                    f"left_support_dist={support}",
                )
    # Exact p4..p7 blocker setup: a blocker directly ahead can turn a north ant
    # into an east signal without needing lateral support.
    if direction == 0 and 2 <= y <= 4 and 1 <= x <= 9:
        ahead = neighbors[pos][0]
        if ahead != -1:
            ay, ax = divmod(ahead, WIDTH_HINT)
            if ant.is_ant(state[ahead]):
                rel = (ant.ant_dir(state[ahead]) - direction) & 3
                turn_penalty = 0 if programs[clan][3 + rel] == 1 else 70
                p1_penalty = 0 if programs[clan][0] == 0 else 180
                right = neighbors[pos][1]
                right_value = state[right] if right != -1 else ant.CELL_WALL
                lateral_wait_penalty = 0
                if right_value in (ant.CELL_FLOOR, ant.CELL_FOOD):
                    lateral_penalty = 0
                elif ant.is_ant(right_value):
                    wait_rel = (ant.ant_dir(right_value) - 1) & 3
                    lateral_wait_penalty = (
                        0 if programs[clan][3 + wait_rel] == 0 else 105
                    )
                    lateral_penalty = 45 + lateral_wait_penalty
                else:
                    lateral_penalty = 220
                score = (
                    60
                    + (9 - x) * 10
                    + abs(y - 2) * 6
                    + p1_penalty
                    + turn_penalty
                    + lateral_penalty
                )
                return (
                    score,
                    f"p6-exact-blocker ant=({x},{y}) {clan}^ "
                    f"ahead=({ax},{ay}) rel={rel} turn_penalty={turn_penalty} "
                    f"lateral_penalty={lateral_penalty} "
                    f"lateral_wait_penalty={lateral_wait_penalty}",
                )
            blocker_dist = support_distance(
                state,
                live_ants,
                pos,
                ahead,
                good_ahead_dirs(programs, value, 1, True),
            )
            p1_penalty = 0 if programs[clan][0] == 0 else 180
            score = (
                100
                + (9 - x) * 16
                + abs(y - 2) * 8
                + blocker_dist * 25
                + p1_penalty
            )
            return (
                score,
                f"p6-blocker-prep ant=({x},{y}) {clan}^ "
                f"ahead=({ax},{ay}) blocker_dist={blocker_dist}",
            )
    return None


def good_ahead_dirs(
    programs: List[List[int]],
    center_ant: int,
    target_dir: int,
    enabled: bool,
) -> Optional[set[int]]:
    if not enabled:
        return None
    clan = ant.ant_clan(center_ant)
    direction = ant.ant_dir(center_ant)
    needed_turn = (target_dir - direction) & 3
    return {
        ahead_dir
        for ahead_dir in range(4)
        if programs[clan][3 + ((ahead_dir - direction) & 3)] == needed_turn
    }


def support_distance(
    state: List[int],
    live_ants: List[Tuple[int, int]],
    center_pos: int,
    ahead_pos: int,
    good_dirs: Optional[set[int]],
) -> int:
    if ant.is_ant(state[ahead_pos]):
        if good_dirs is None or ant.ant_dir(state[ahead_pos]) in good_dirs:
            return 0
        return 50
    ay, ax = divmod(ahead_pos, WIDTH_HINT)
    best = 50
    for other_pos, other_value in live_ants:
        if other_pos == center_pos:
            continue
        oy, ox = divmod(other_pos, WIDTH_HINT)
        penalty = (
            0
            if good_dirs is None or ant.ant_dir(other_value) in good_dirs
            else 2
        )
        best = min(best, abs(ox - ax) + abs(oy - ay) + penalty)
    return best


def directed_support_distance(
    live_ants: List[Tuple[int, int]],
    center_pos: int,
    support_pos: int,
    support_dir: int,
) -> int:
    sy, sx = divmod(support_pos, WIDTH_HINT)
    best = 50
    for other_pos, other_value in live_ants:
        if other_pos == center_pos:
            continue
        oy, ox = divmod(other_pos, WIDTH_HINT)
        dir_penalty = 0 if ant.ant_dir(other_value) == support_dir else 2
        best = min(best, abs(ox - sx) + abs(oy - sy) + dir_penalty)
    return best


def evaluate(
    puzzle: ant.Puzzle,
    wilds: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    food_indices: List[int],
    targets: set[Tuple[int, int]],
    dist: List[Optional[int]],
    cand: Candidate,
    max_steps: int,
    success_mode: str,
    manhattan_weight: int,
    direction_penalty: int,
    distance_weight: int,
    support_target_turn: bool,
    support_weight: int,
    support_near_target_turn: bool,
    support_max_distance: int,
    support_program_aware: bool,
    support_entry_turn: bool,
    p6_focus: bool,
) -> EvalResult:
    global WIDTH_HINT
    WIDTH_HINT = puzzle.width
    state = build_state(puzzle, wilds, cand)
    seen = set()
    best_score = 10**9
    best_step = 0
    best_detail = ""
    for step in range(max_steps + 1):
        if ant.has_success_fast(state, neighbors, food_indices, success_mode):
            return EvalResult(-1000000 + step, True, step, "success")
        score, detail = state_score(
            state,
            neighbors,
            cand.programs,
            targets,
            dist,
            manhattan_weight,
            direction_penalty,
            distance_weight,
            support_target_turn,
            support_weight,
            support_near_target_turn,
            support_max_distance,
            support_program_aware,
            support_entry_turn,
            p6_focus,
        )
        score += step
        if score < best_score:
            best_score = score
            best_step = step
            best_detail = detail
        key = bytes(state)
        if key in seen:
            break
        seen.add(key)
        state = ant.step_state(state, neighbors, cand.programs)
    return EvalResult(best_score, False, best_step, best_detail)


def random_candidate(
    rng: random.Random,
    templates: List[List[Optional[int]]],
    wilds: List[int],
    grid_values: List[int],
    ant_values: List[int],
    ant_probability: float,
) -> Candidate:
    programs: List[List[int]] = []
    for row in templates:
        programs.append([
            rng.randrange(4) if value is None else value for value in row
        ])
    grid: List[int] = []
    for _ in wilds:
        if ant_values and rng.random() < ant_probability:
            grid.append(rng.choice(ant_values))
        else:
            grid.append(rng.choice(grid_values))
    return Candidate(programs, grid)


def clone_candidate(cand: Candidate) -> Candidate:
    return Candidate([row[:] for row in cand.programs], cand.grid[:])


def apply_program_templates(cand: Candidate, templates: List[List[Optional[int]]]) -> None:
    for clan, row in enumerate(templates):
        if clan >= len(cand.programs):
            break
        for idx, fixed in enumerate(row):
            if fixed is not None:
                cand.programs[clan][idx] = fixed


def mutate_candidate(
    rng: random.Random,
    cand: Candidate,
    templates: List[List[Optional[int]]],
    grid_values: List[int],
    ant_values: List[int],
    program_mutations: int,
    grid_mutations: int,
) -> Candidate:
    out = clone_candidate(cand)
    mutable_programs = [
        (clan, idx)
        for clan, row in enumerate(templates)
        for idx, fixed in enumerate(row)
        if fixed is None
    ]
    values = grid_values + ant_values
    for _ in range(program_mutations):
        if not mutable_programs:
            break
        clan, idx = rng.choice(mutable_programs)
        old = out.programs[clan][idx]
        new = rng.randrange(3)
        if new >= old:
            new += 1
        out.programs[clan][idx] = new
    for _ in range(grid_mutations):
        if not out.grid:
            break
        idx = rng.randrange(len(out.grid))
        old = out.grid[idx]
        new = rng.choice(values)
        if len(values) > 1:
            while new == old:
                new = rng.choice(values)
        out.grid[idx] = new
    return out


def write_solution(
    path: Path,
    puzzle: ant.Puzzle,
    wilds: List[int],
    cand: Candidate,
) -> None:
    replacements = {idx: value_to_cell(value) for idx, value in zip(wilds, cand.grid)}
    world = ant.build_world(puzzle, replacements)
    ant.write_ant_world(path, puzzle.title, render_programs(cand.programs), world)


def read_candidate(
    path: Path,
    expected: ant.Puzzle,
    wilds: List[int],
) -> Candidate:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"empty candidate file: {path}")
    idx = 1
    programs: List[List[int]] = []
    while idx < len(lines) and not re.fullmatch(r"\s*\d+\s+\d+\s*", lines[idx]):
        line = lines[idx].strip()
        if line:
            programs.append([ant.DIR_TO_IDX[ch] for ch in line[:7]])
        idx += 1
    if idx >= len(lines):
        raise ValueError(f"missing dimensions in candidate file: {path}")
    width, height = map(int, lines[idx].split())
    if width != expected.width or height != expected.height:
        raise ValueError(
            f"candidate dimensions differ: {(width, height)} != "
            f"{(expected.width, expected.height)}"
        )
    idx += 1
    tokens = ant.parse_grid_lines(lines[idx : idx + height], width, height)
    grid: List[int] = []
    for wild_idx in wilds:
        cell = ant.parse_cell(tokens[wild_idx])
        grid.append(cell_to_value(cell))
    return Candidate(programs, grid)


def parse_grid_values(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.extend(ant.parse_grid_wild_values(part))
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzles", default="volume9_gardener_puzzles.txt")
    parser.add_argument("--match", required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid-values", default="floor,wall,hole")
    parser.add_argument("--ant-clans", default="")
    parser.add_argument("--ant-probability", type=float, default=0.75)
    parser.add_argument(
        "--fix-program",
        help="Comma-separated fixed program slots, e.g. p1=N or 0:p4=E.",
    )
    parser.add_argument(
        "--restart-evals",
        type=int,
        default=4000,
        help="Restart annealing after this many child evaluations; <=0 disables.",
    )
    parser.add_argument("--restarts", type=int, default=200)
    parser.add_argument("--output")
    parser.add_argument("--success-below", action="store_true")
    parser.add_argument("--manhattan-weight", type=int, default=100)
    parser.add_argument("--direction-penalty", type=int, default=120)
    parser.add_argument("--distance-weight", type=int, default=5)
    parser.add_argument("--support-target-turn", action="store_true")
    parser.add_argument("--support-weight", type=int, default=20)
    parser.add_argument("--support-near-target-turn", action="store_true")
    parser.add_argument("--support-max-distance", type=int, default=4)
    parser.add_argument("--support-program-aware", action="store_true")
    parser.add_argument("--support-entry-turn", action="store_true")
    parser.add_argument("--p6-focus", action="store_true")
    parser.add_argument("--start-candidate")
    parser.add_argument("--start-every-restart", action="store_true")
    args = parser.parse_args()

    puzzles = ant.parse_puzzles(Path(args.puzzles))
    matches = [puzzle for puzzle in puzzles if args.match in puzzle.title]
    if len(matches) != 1:
        raise SystemExit(f"expected one puzzle, found {len(matches)}")
    puzzle = matches[0]
    templates = parse_program_templates(puzzle)
    apply_fixed_program_option(templates, args.fix_program)
    wilds = ant.find_wildcards(puzzle)
    neighbors = ant.build_neighbors(puzzle.width, puzzle.height)
    base_state, _wilds, food_indices = ant.build_base_state(puzzle)
    success_mode = "below" if args.success_below else "facing"
    targets = target_cells(base_state, neighbors, success_mode)
    dist = optimistic_distances(puzzle, targets)

    grid_values = parse_grid_values(args.grid_values)
    if args.ant_clans:
        clans = [int(part) for part in args.ant_clans.split(",") if part.strip()]
    else:
        clans = list(range(len(puzzle.programs)))
    ant_values = [ant.make_ant(clan, direction) for clan in clans for direction in range(4)]

    rng = random.Random(args.seed)
    start_candidate = (
        read_candidate(Path(args.start_candidate), puzzle, wilds)
        if args.start_candidate
        else None
    )
    if start_candidate is not None:
        apply_program_templates(start_candidate, templates)
    deadline = monotonic() + args.seconds
    best: Optional[Candidate] = None
    best_eval = EvalResult(10**9, False, 0, "")
    evals = 0
    restart = 0

    while monotonic() < deadline:
        restart += 1
        if start_candidate is not None and (restart == 1 or args.start_every_restart):
            cand = clone_candidate(start_candidate)
        else:
            cand = random_candidate(
                rng, templates, wilds, grid_values, ant_values, args.ant_probability
            )
        cur_eval = evaluate(
            puzzle, wilds, neighbors, food_indices, targets, dist, cand,
            args.max_steps, success_mode,
            args.manhattan_weight, args.direction_penalty, args.distance_weight,
            args.support_target_turn, args.support_weight,
            args.support_near_target_turn, args.support_max_distance,
            args.support_program_aware,
            args.support_entry_turn,
            args.p6_focus,
        )
        temp0 = 150.0
        restart_evals = 0
        while monotonic() < deadline and (
            args.restart_evals <= 0 or restart_evals < args.restart_evals
        ):
            evals += 1
            restart_evals += 1
            if cur_eval.ok:
                best = cand
                best_eval = cur_eval
                break
            if cur_eval.score < best_eval.score:
                best = clone_candidate(cand)
                best_eval = cur_eval
                programs = " ".join(render_programs(best.programs))
                print(
                    f"BEST score={best_eval.score} step={best_eval.step} "
                    f"evals={evals} restart={restart} {best_eval.best_detail} "
                    f"programs={programs}",
                    flush=True,
                )
            scale = max(1, len(wilds))
            prog_mut = 1 if rng.random() < 0.8 else rng.randint(2, 4)
            grid_mut = 1 if rng.random() < 0.85 else rng.randint(2, max(2, scale // 3))
            child = mutate_candidate(
                rng, cand, templates, grid_values, ant_values, prog_mut, grid_mut
            )
            child_eval = evaluate(
                puzzle, wilds, neighbors, food_indices, targets, dist, child,
                args.max_steps, success_mode,
                args.manhattan_weight, args.direction_penalty, args.distance_weight,
                args.support_target_turn, args.support_weight,
                args.support_near_target_turn, args.support_max_distance,
                args.support_program_aware,
                args.support_entry_turn,
                args.p6_focus,
            )
            delta = child_eval.score - cur_eval.score
            elapsed_frac = 1.0 - max(0.0, deadline - monotonic()) / max(args.seconds, 1e-6)
            temp = max(0.5, temp0 * (1.0 - elapsed_frac))
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                cand = child
                cur_eval = child_eval
        if best_eval.ok:
            break
        if args.restarts and restart >= args.restarts:
            break

    if best is None:
        print("NO BEST")
        return
    if best_eval.ok:
        print(f"FOUND step={best_eval.step} evals={evals}")
    else:
        print(
            f"NOFOUND best={best_eval.score} step={best_eval.step} "
            f"{best_eval.best_detail} evals={evals}"
        )
    print("PROGRAMS")
    for line in render_programs(best.programs):
        print(line)
    print("GRID")
    for idx, value in zip(wilds, best.grid):
        y, x = divmod(idx, puzzle.width)
        print(f"{x},{y} {ant.cell_to_token(value_to_cell(value))}")
    if args.output:
        write_solution(Path(args.output), puzzle, wilds, best)
        print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
