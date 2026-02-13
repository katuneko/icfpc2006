#!/usr/bin/env python3
"""Antomaton puzzle helper: parse puzzles, simulate, and brute-force small cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import argparse
from collections import Counter, deque
from itertools import combinations, product
import re
import heapq

# Directions: 0=N,1=E,2=S,3=W
DIRS = "NESW"
DIR_TO_IDX = {"N": 0, "E": 1, "S": 2, "W": 3}
IDX_TO_DIR = {v: k for k, v in DIR_TO_IDX.items()}
ANT_DIR_CHARS = "^>v<"
ANT_CHAR_TO_DIR = {"^": 0, ">": 1, "v": 2, "<": 3}
DIR_TO_ANT_CHAR = {v: k for k, v in ANT_CHAR_TO_DIR.items()}

KIND_WALL = "#"
KIND_HOLE = "o"
KIND_FLOOR = "-"
KIND_FOOD = "$"
KIND_ANT = "a"
KIND_WILD = "*"

CELL_WALL = 0
CELL_HOLE = 1
CELL_FLOOR = 2
CELL_FOOD = 3
CELL_WILD = -1
CELL_ANT_BASE = 4
MAX_CLANS = 10
MAX_CELL = CELL_ANT_BASE + MAX_CLANS * 4

ROT_MAPS = (
    (0, 1, 2, 3),
    (3, 0, 1, 2),
    (2, 3, 0, 1),
    (1, 2, 3, 0),
)

ROTATED = [[0] * 4 for _ in range(MAX_CELL)]
for _val in range(MAX_CELL):
    if _val < CELL_ANT_BASE:
        for _rot in range(4):
            ROTATED[_val][_rot] = _val
    else:
        _base = _val - CELL_ANT_BASE
        _clan = _base >> 2
        _dir = _base & 3
        for _rot in range(4):
            ROTATED[_val][_rot] = CELL_ANT_BASE + (_clan << 2) + ((_dir + _rot) & 3)


@dataclass(frozen=True)
class Cell:
    kind: str
    clan: Optional[int] = None
    direction: Optional[int] = None


@dataclass
class Puzzle:
    title: str
    programs: List[str]
    width: int
    height: int
    grid_tokens: List[str]


@dataclass
class World:
    width: int
    height: int
    grid: List[Cell]


@dataclass
class Programs:
    # programs[clan] -> list of 7 direction indices (0..3)
    programs: List[List[int]]


@dataclass(frozen=True)
class CarveSolution:
    start_idx: int
    start_dir: int
    p1: int
    floor_mask: int
    wall_mask: int
    wild_list: Tuple[int, ...]


def rotate_dir(direction: int, steps: int) -> int:
    return (direction + steps) % 4


def rotate_cell(cell: Cell, steps: int) -> Cell:
    if cell.kind != KIND_ANT:
        return cell
    return Cell(KIND_ANT, cell.clan, rotate_dir(cell.direction, steps))


def parse_cell(token: str) -> Cell:
    if len(token) != 2:
        raise ValueError(f"invalid token length: {token!r}")
    first, second = token[0], token[1]
    if second in (KIND_WALL, KIND_HOLE, KIND_FLOOR, KIND_FOOD):
        return Cell(second)
    if second in ANT_CHAR_TO_DIR:
        if not first.isdigit():
            raise ValueError(f"ant without clan digit: {token!r}")
        return Cell(KIND_ANT, int(first), ANT_CHAR_TO_DIR[second])
    if second == KIND_WILD:
        return Cell(KIND_WILD)
    raise ValueError(f"unknown token: {token!r}")


def cell_to_token(cell: Cell) -> str:
    if cell.kind == KIND_ANT:
        return f"{cell.clan}{DIR_TO_ANT_CHAR[cell.direction]}"
    if cell.kind in (KIND_WALL, KIND_HOLE, KIND_FLOOR, KIND_FOOD):
        return f" {cell.kind}"
    raise ValueError(f"cannot render cell kind: {cell}")


def parse_programs(lines: List[str]) -> List[str]:
    programs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not re.fullmatch(r"[NESW*]{7,8}", line):
            raise ValueError(f"invalid program line: {line!r}")
        programs.append(line)
    return programs


def parse_grid_lines(lines: List[str], width: int, height: int) -> List[str]:
    if len(lines) != height:
        raise ValueError("grid line count mismatch")
    grid_tokens = []
    expected_len = width * 2
    for line in lines:
        line = line.rstrip()
        if len(line) != expected_len:
            raise ValueError(f"grid line length mismatch: {line!r}")
        for i in range(0, expected_len, 2):
            grid_tokens.append(line[i : i + 2])
    return grid_tokens


def parse_puzzles(path: Path) -> List[Puzzle]:
    lines = path.read_text().splitlines()
    puzzles: List[Puzzle] = []
    current_title = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_title, current_lines
        if current_title is None:
            return
        # parse current_lines into puzzle data
        idx = 0
        while idx < len(current_lines) and current_lines[idx].strip() == "":
            idx += 1
        program_lines: List[str] = []
        while idx < len(current_lines) and not re.fullmatch(
            r"\s*\d+\s+\d+\s*", current_lines[idx]
        ):
            program_lines.append(current_lines[idx])
            idx += 1
        if idx >= len(current_lines):
            raise ValueError(f"missing width/height in {current_title}")
        width, height = map(int, current_lines[idx].split())
        idx += 1
        grid_lines = current_lines[idx : idx + height]
        programs = parse_programs(program_lines)
        grid_tokens = parse_grid_lines(grid_lines, width, height)
        puzzles.append(Puzzle(current_title, programs, width, height, grid_tokens))
        current_title = None
        current_lines = []

    for line in lines:
        if line.startswith("% "):
            if line.startswith("% UMIX shutdown"):
                break
            flush()
            current_title = line[2:].strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)
    flush()
    return puzzles


def build_programs(programs: List[str], default: Optional[List[int]] = None) -> Programs:
    # programs are per clan index (0..n-1) for lines provided
    built: List[List[int]] = []
    for line in programs:
        if any(ch == "*" for ch in line):
            raise ValueError("wildcards in programs not supported in build_programs")
        if len(line) < 7:
            raise ValueError(f"unsupported program length: {line!r}")
        built.append([DIR_TO_IDX[ch] for ch in line[:7]])
    if default is not None:
        # extend to 10 clans
        while len(built) < 10:
            built.append(default[:])
    return Programs(built)


def build_programs_for_clans(programs: List[str], used_clans: List[int]) -> Programs:
    if not used_clans:
        return Programs([])
    max_clan = max(used_clans)
    built: List[List[int]] = []
    for clan in range(max_clan + 1):
        if clan >= len(programs):
            raise ValueError(f"missing program for clan {clan}")
        line = programs[clan]
        if any(ch == "*" for ch in line):
            raise ValueError(f"wildcards in program for clan {clan}")
        if len(line) < 7:
            raise ValueError(f"unsupported program length for clan {clan}: {line!r}")
        built.append([DIR_TO_IDX[ch] for ch in line[:7]])
    return Programs(built)


def used_clans_in_world(world: World) -> List[int]:
    clans = sorted({cell.clan for cell in world.grid if cell.kind == KIND_ANT})
    return clans


def encode_cell(cell: Cell) -> int:
    if cell.kind == KIND_WALL:
        return CELL_WALL
    if cell.kind == KIND_HOLE:
        return CELL_HOLE
    if cell.kind == KIND_FLOOR:
        return CELL_FLOOR
    if cell.kind == KIND_FOOD:
        return CELL_FOOD
    if cell.kind == KIND_ANT:
        return CELL_ANT_BASE + cell.clan * 4 + cell.direction
    raise ValueError(f"unsupported cell kind: {cell}")


def is_ant(val: int) -> bool:
    return val >= CELL_ANT_BASE


def ant_dir(val: int) -> int:
    return (val - CELL_ANT_BASE) & 3


def ant_clan(val: int) -> int:
    return (val - CELL_ANT_BASE) >> 2


def make_ant(clan: int, direction: int) -> int:
    return CELL_ANT_BASE + (clan * 4) + (direction & 3)


def rotate_ant(val: int, steps: int) -> int:
    if val < CELL_ANT_BASE:
        return val
    base = val - CELL_ANT_BASE
    clan = base >> 2
    direction = base & 3
    return CELL_ANT_BASE + (clan << 2) + ((direction + steps) & 3)


def build_neighbors(width: int, height: int) -> List[Tuple[int, int, int, int]]:
    neighbors = []
    for y in range(height):
        for x in range(width):
            n = (y - 1) * width + x if y > 0 else -1
            e = y * width + (x + 1) if x + 1 < width else -1
            s = (y + 1) * width + x if y + 1 < height else -1
            w = y * width + (x - 1) if x > 0 else -1
            neighbors.append((n, e, s, w))
    return neighbors


def apply_turn_program_int(ant_val: int, programs: List[List[int]], p_idx: int) -> int:
    clan = ant_clan(ant_val)
    turn = programs[clan][p_idx - 1]
    return make_ant(clan, turn)


def step_state(
    state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    programs: List[List[int]],
) -> List[int]:
    count = len(neighbors)
    new_state = [CELL_FLOOR] * count
    rotated = ROTATED
    rot_maps = ROT_MAPS
    for idx, val in enumerate(state):
        if val in (CELL_WALL, CELL_HOLE, CELL_FOOD):
            new_state[idx] = val
            continue
        n_idx, e_idx, s_idx, w_idx = neighbors[idx]
        n_val = state[n_idx] if n_idx != -1 else CELL_WALL
        e_val = state[e_idx] if e_idx != -1 else CELL_WALL
        s_val = state[s_idx] if s_idx != -1 else CELL_WALL
        w_val = state[w_idx] if w_idx != -1 else CELL_WALL
        vals = (n_val, e_val, s_val, w_val)
        if is_ant(val):
            direction = ant_dir(val)
            ahead = vals[direction]
            if ahead in (CELL_FLOOR, CELL_HOLE):
                new_state[idx] = CELL_FLOOR
                continue
            if ahead == CELL_WALL:
                new_state[idx] = make_ant(ant_clan(val), (direction + 1) & 3)
                continue
        if val == CELL_FLOOR:
            # non-orientable cross: all four ants pointing to center
            if (
                is_ant(n_val)
                and ant_dir(n_val) == 2
                and is_ant(e_val)
                and ant_dir(e_val) == 3
                and is_ant(s_val)
                and ant_dir(s_val) == 0
                and is_ant(w_val)
                and ant_dir(w_val) == 1
            ):
                new_state[idx] = CELL_FLOOR
                continue
            # p3: >-< with bottom ant
            for rot in range(4):
                rot_map = rot_maps[rot]
                r_n = rotated[vals[rot_map[0]]][rot]
                r_e = rotated[vals[rot_map[1]]][rot]
                r_s = rotated[vals[rot_map[2]]][rot]
                r_w = rotated[vals[rot_map[3]]][rot]
                if (
                    is_ant(r_w)
                    and ant_dir(r_w) == 1
                    and is_ant(r_e)
                    and ant_dir(r_e) == 3
                    and is_ant(r_s)
                    and ant_dir(r_s) == 0
                ):
                    out = apply_turn_program_int(r_s, programs, 3)
                    new_state[idx] = rotated[out][(-rot) & 3]
                    break
            else:
                # non-orientable pair
                for rot in range(4):
                    rot_map = rot_maps[rot]
                    r_n = rotated[vals[rot_map[0]]][rot]
                    r_s = rotated[vals[rot_map[2]]][rot]
                    if is_ant(r_n) and ant_dir(r_n) == 2 and is_ant(r_s) and ant_dir(r_s) == 0:
                        new_state[idx] = CELL_FLOOR
                        break
                else:
                    # p2
                    for rot in range(4):
                        rot_map = rot_maps[rot]
                        r_s = rotated[vals[rot_map[2]]][rot]
                        r_w = rotated[vals[rot_map[3]]][rot]
                        if is_ant(r_w) and ant_dir(r_w) == 1 and is_ant(r_s) and ant_dir(r_s) == 0:
                            out = apply_turn_program_int(r_s, programs, 2)
                            new_state[idx] = rotated[out][(-rot) & 3]
                            break
                    else:
                        # p1
                        for rot in range(4):
                            rot_map = rot_maps[rot]
                            r_s = rotated[vals[rot_map[2]]][rot]
                            if is_ant(r_s) and ant_dir(r_s) == 0:
                                out = apply_turn_program_int(r_s, programs, 1)
                                new_state[idx] = rotated[out][(-rot) & 3]
                                break
                        else:
                            new_state[idx] = CELL_FLOOR
            continue
        # center ant with ant ahead (p4..p7)
        if is_ant(val):
            direction = ant_dir(val)
            ahead = vals[direction]
            if is_ant(ahead):
                rel = (ant_dir(ahead) - direction) & 3
                clan = ant_clan(val)
                turn = programs[clan][3 + rel]
                new_state[idx] = make_ant(clan, (turn + direction) & 3)
                continue
        new_state[idx] = val
    return new_state


def step_cell_int(
    val: int,
    vals: Tuple[int, int, int, int],
    programs: List[List[int]],
) -> int:
    if val in (CELL_WALL, CELL_HOLE, CELL_FOOD):
        return val
    n_val, e_val, s_val, w_val = vals
    if is_ant(val):
        direction = ant_dir(val)
        ahead = vals[direction]
        if ahead in (CELL_FLOOR, CELL_HOLE):
            return CELL_FLOOR
        if ahead == CELL_WALL:
            return make_ant(ant_clan(val), (direction + 1) & 3)
    if val == CELL_FLOOR:
        if (
            is_ant(n_val)
            and ant_dir(n_val) == 2
            and is_ant(e_val)
            and ant_dir(e_val) == 3
            and is_ant(s_val)
            and ant_dir(s_val) == 0
            and is_ant(w_val)
            and ant_dir(w_val) == 1
        ):
            return CELL_FLOOR
        for rot in range(4):
            rot_map = ROT_MAPS[rot]
            r_n = ROTATED[vals[rot_map[0]]][rot]
            r_e = ROTATED[vals[rot_map[1]]][rot]
            r_s = ROTATED[vals[rot_map[2]]][rot]
            r_w = ROTATED[vals[rot_map[3]]][rot]
            if (
                is_ant(r_w)
                and ant_dir(r_w) == 1
                and is_ant(r_e)
                and ant_dir(r_e) == 3
                and is_ant(r_s)
                and ant_dir(r_s) == 0
            ):
                out = apply_turn_program_int(r_s, programs, 3)
                return ROTATED[out][(-rot) & 3]
        for rot in range(4):
            rot_map = ROT_MAPS[rot]
            r_n = ROTATED[vals[rot_map[0]]][rot]
            r_s = ROTATED[vals[rot_map[2]]][rot]
            if (
                is_ant(r_n)
                and ant_dir(r_n) == 2
                and is_ant(r_s)
                and ant_dir(r_s) == 0
            ):
                return CELL_FLOOR
        for rot in range(4):
            rot_map = ROT_MAPS[rot]
            r_s = ROTATED[vals[rot_map[2]]][rot]
            r_w = ROTATED[vals[rot_map[3]]][rot]
            if is_ant(r_w) and ant_dir(r_w) == 1 and is_ant(r_s) and ant_dir(r_s) == 0:
                out = apply_turn_program_int(r_s, programs, 2)
                return ROTATED[out][(-rot) & 3]
        for rot in range(4):
            rot_map = ROT_MAPS[rot]
            r_s = ROTATED[vals[rot_map[2]]][rot]
            if is_ant(r_s) and ant_dir(r_s) == 0:
                out = apply_turn_program_int(r_s, programs, 1)
                return ROTATED[out][(-rot) & 3]
        return CELL_FLOOR
    if is_ant(val):
        direction = ant_dir(val)
        ahead = vals[direction]
        if is_ant(ahead):
            rel = (ant_dir(ahead) - direction) & 3
            clan = ant_clan(val)
            turn = programs[clan][3 + rel]
            return make_ant(clan, (turn + direction) & 3)
    return val


def simulate_fast(
    state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    programs: List[List[int]],
    food_indices: List[int],
    max_steps: int = 10000,
    success_mode: str = "facing",
) -> Tuple[bool, int]:
    seen = set()
    for step in range(max_steps + 1):
        if has_success_fast(state, neighbors, food_indices, success_mode):
            return True, step
        key = bytes(state)
        if key in seen:
            return False, step
        seen.add(key)
        state = step_state(state, neighbors, programs)
    return False, max_steps


def split_static_ants(state: List[int]) -> Tuple[List[int], Dict[int, int]]:
    base_state = state[:]
    ants: Dict[int, int] = {}
    for idx, val in enumerate(base_state):
        if is_ant(val):
            ants[idx] = val
            base_state[idx] = CELL_FLOOR
    return base_state, ants


def has_success_sparse(
    ants: Dict[int, int],
    neighbors: List[Tuple[int, int, int, int]],
    base_state: List[int],
    mode: str,
) -> bool:
    if mode == "facing":
        for pos, val in ants.items():
            ahead = neighbors[pos][ant_dir(val)]
            if ahead != -1 and base_state[ahead] == CELL_FOOD:
                return True
        return False
    if mode == "below":
        for pos, val in ants.items():
            if ant_dir(val) != 0:
                continue
            ahead = neighbors[pos][0]
            if ahead != -1 and base_state[ahead] == CELL_FOOD:
                return True
        return False
    raise ValueError(f"unknown success mode: {mode!r}")


def step_sparse(
    ants: Dict[int, int],
    neighbors: List[Tuple[int, int, int, int]],
    base_state: List[int],
    programs: List[List[int]],
) -> Dict[int, int]:
    affected = set()
    for pos in ants:
        affected.add(pos)
        n_idx, e_idx, s_idx, w_idx = neighbors[pos]
        if n_idx != -1:
            affected.add(n_idx)
        if e_idx != -1:
            affected.add(e_idx)
        if s_idx != -1:
            affected.add(s_idx)
        if w_idx != -1:
            affected.add(w_idx)
    new_ants: Dict[int, int] = {}
    for idx in affected:
        val = ants.get(idx, base_state[idx])
        n_idx, e_idx, s_idx, w_idx = neighbors[idx]
        n_val = ants.get(n_idx, base_state[n_idx]) if n_idx != -1 else CELL_WALL
        e_val = ants.get(e_idx, base_state[e_idx]) if e_idx != -1 else CELL_WALL
        s_val = ants.get(s_idx, base_state[s_idx]) if s_idx != -1 else CELL_WALL
        w_val = ants.get(w_idx, base_state[w_idx]) if w_idx != -1 else CELL_WALL
        next_val = step_cell_int(val, (n_val, e_val, s_val, w_val), programs)
        if is_ant(next_val):
            new_ants[idx] = next_val
    return new_ants


def simulate_sparse(
    ants: Dict[int, int],
    neighbors: List[Tuple[int, int, int, int]],
    base_state: List[int],
    programs: List[List[int]],
    max_steps: int,
    success_mode: str,
) -> Tuple[bool, int]:
    seen = set()
    for step in range(max_steps + 1):
        if has_success_sparse(ants, neighbors, base_state, success_mode):
            return True, step
        key = tuple(sorted(ants.items()))
        if key in seen:
            return False, step
        seen.add(key)
        ants = step_sparse(ants, neighbors, base_state, programs)
    return False, max_steps


def step_sparse_partial(
    ants: Dict[int, int],
    neighbors: List[Tuple[int, int, int, int]],
    base_state: List[int],
    assignments: List[int],
    program_count: int,
) -> List[Tuple[Dict[int, int], List[int]]]:
    affected = set()
    for pos in ants:
        affected.add(pos)
        n_idx, e_idx, s_idx, w_idx = neighbors[pos]
        if n_idx != -1:
            affected.add(n_idx)
        if e_idx != -1:
            affected.add(e_idx)
        if s_idx != -1:
            affected.add(s_idx)
        if w_idx != -1:
            affected.add(w_idx)

    new_ants: Dict[int, int] = {}
    deps: Dict[int, List[Tuple[int, List[int]]]] = {}

    def add_dep(entry: int, idx: int, outputs: List[int]) -> None:
        deps.setdefault(entry, []).append((idx, outputs))

    for idx in affected:
        val = ants.get(idx, base_state[idx])
        if val in (CELL_WALL, CELL_HOLE, CELL_FOOD):
            continue
        n_idx, e_idx, s_idx, w_idx = neighbors[idx]
        n_val = ants.get(n_idx, base_state[n_idx]) if n_idx != -1 else CELL_WALL
        e_val = ants.get(e_idx, base_state[e_idx]) if e_idx != -1 else CELL_WALL
        s_val = ants.get(s_idx, base_state[s_idx]) if s_idx != -1 else CELL_WALL
        w_val = ants.get(w_idx, base_state[w_idx]) if w_idx != -1 else CELL_WALL
        vals = (n_val, e_val, s_val, w_val)

        if is_ant(val):
            direction = ant_dir(val)
            ahead = vals[direction]
            if ahead in (CELL_FLOOR, CELL_HOLE):
                continue
            if ahead == CELL_WALL:
                new_ants[idx] = make_ant(ant_clan(val), (direction + 1) & 3)
                continue

        if val == CELL_FLOOR:
            if (
                is_ant(n_val)
                and ant_dir(n_val) == 2
                and is_ant(e_val)
                and ant_dir(e_val) == 3
                and is_ant(s_val)
                and ant_dir(s_val) == 0
                and is_ant(w_val)
                and ant_dir(w_val) == 1
            ):
                continue

            for rot in range(4):
                rot_map = ROT_MAPS[rot]
                r_w = ROTATED[vals[rot_map[3]]][rot]
                r_e = ROTATED[vals[rot_map[1]]][rot]
                r_s = ROTATED[vals[rot_map[2]]][rot]
                if (
                    is_ant(r_w)
                    and ant_dir(r_w) == 1
                    and is_ant(r_e)
                    and ant_dir(r_e) == 3
                    and is_ant(r_s)
                    and ant_dir(r_s) == 0
                ):
                    clan = ant_clan(r_s)
                    if clan >= program_count:
                        raise ValueError(f"missing program for clan {clan}")
                    entry = program_entry_index(clan, 3)
                    outputs = [
                        ROTATED[make_ant(clan, turn)][(-rot) & 3]
                        for turn in range(4)
                    ]
                    assigned = assignments[entry]
                    if assigned == -1:
                        add_dep(entry, idx, outputs)
                    else:
                        new_ants[idx] = outputs[assigned]
                    break
            else:
                for rot in range(4):
                    rot_map = ROT_MAPS[rot]
                    r_n = ROTATED[vals[rot_map[0]]][rot]
                    r_s = ROTATED[vals[rot_map[2]]][rot]
                    if (
                        is_ant(r_n)
                        and ant_dir(r_n) == 2
                        and is_ant(r_s)
                        and ant_dir(r_s) == 0
                    ):
                        break
                else:
                    for rot in range(4):
                        rot_map = ROT_MAPS[rot]
                        r_s = ROTATED[vals[rot_map[2]]][rot]
                        r_w = ROTATED[vals[rot_map[3]]][rot]
                        if (
                            is_ant(r_w)
                            and ant_dir(r_w) == 1
                            and is_ant(r_s)
                            and ant_dir(r_s) == 0
                        ):
                            clan = ant_clan(r_s)
                            if clan >= program_count:
                                raise ValueError(f"missing program for clan {clan}")
                            entry = program_entry_index(clan, 2)
                            outputs = [
                                ROTATED[make_ant(clan, turn)][(-rot) & 3]
                                for turn in range(4)
                            ]
                            assigned = assignments[entry]
                            if assigned == -1:
                                add_dep(entry, idx, outputs)
                            else:
                                new_ants[idx] = outputs[assigned]
                            break
                    else:
                        for rot in range(4):
                            rot_map = ROT_MAPS[rot]
                            r_s = ROTATED[vals[rot_map[2]]][rot]
                            if is_ant(r_s) and ant_dir(r_s) == 0:
                                clan = ant_clan(r_s)
                                if clan >= program_count:
                                    raise ValueError(f"missing program for clan {clan}")
                                entry = program_entry_index(clan, 1)
                                outputs = [
                                    ROTATED[make_ant(clan, turn)][(-rot) & 3]
                                    for turn in range(4)
                                ]
                                assigned = assignments[entry]
                                if assigned == -1:
                                    add_dep(entry, idx, outputs)
                                else:
                                    new_ants[idx] = outputs[assigned]
                                break
            continue

        if is_ant(val):
            direction = ant_dir(val)
            ahead = vals[direction]
            if is_ant(ahead):
                rel = (ant_dir(ahead) - direction) & 3
                clan = ant_clan(val)
                if clan >= program_count:
                    raise ValueError(f"missing program for clan {clan}")
                entry = program_entry_index(clan, 4 + rel)
                outputs = [
                    make_ant(clan, (turn + direction) & 3) for turn in range(4)
                ]
                assigned = assignments[entry]
                if assigned == -1:
                    add_dep(entry, idx, outputs)
                else:
                    new_ants[idx] = outputs[assigned]
                continue
            new_ants[idx] = val

    unassigned = [entry for entry in deps if assignments[entry] == -1]
    if not unassigned:
        return [(new_ants, assignments)]

    results: List[Tuple[Dict[int, int], List[int]]] = []

    def recurse(i: int, cur_ants: Dict[int, int], cur_assignments: List[int]) -> None:
        if i == len(unassigned):
            results.append((cur_ants, cur_assignments))
            return
        entry = unassigned[i]
        for val in range(4):
            next_ants = dict(cur_ants)
            next_assignments = cur_assignments[:]
            next_assignments[entry] = val
            for idx, outputs in deps[entry]:
                out = outputs[val]
                if is_ant(out):
                    next_ants[idx] = out
                else:
                    next_ants.pop(idx, None)
            recurse(i + 1, next_ants, next_assignments)

    recurse(0, new_ants, assignments[:])
    return results


def step_sparse_partial_with_grid(
    ants: Dict[int, int],
    neighbors: List[Tuple[int, int, int, int]],
    base_state: List[int],
    assignments: List[int],
    grid_assignments: List[int],
    grid_entry_by_idx: Dict[int, int],
    grid_values: List[int],
    grid_nonfloor_limit: Optional[int],
    program_count: int,
) -> List[Tuple[Dict[int, int], List[int], List[int]]]:
    affected = set()
    for pos in ants:
        affected.add(pos)
        n_idx, e_idx, s_idx, w_idx = neighbors[pos]
        if n_idx != -1:
            affected.add(n_idx)
        if e_idx != -1:
            affected.add(e_idx)
        if s_idx != -1:
            affected.add(s_idx)
        if w_idx != -1:
            affected.add(w_idx)

    positions = set(affected)
    for pos in affected:
        n_idx, e_idx, s_idx, w_idx = neighbors[pos]
        if n_idx != -1:
            positions.add(n_idx)
        if e_idx != -1:
            positions.add(e_idx)
        if s_idx != -1:
            positions.add(s_idx)
        if w_idx != -1:
            positions.add(w_idx)

    unassigned_entries: List[int] = []
    unassigned_set = set()
    for pos in positions:
        if base_state[pos] != CELL_WILD:
            continue
        entry = grid_entry_by_idx.get(pos)
        if entry is None:
            raise ValueError(f"missing grid entry for wildcard at {pos}")
        if grid_assignments[entry] == -1 and entry not in unassigned_set:
            unassigned_set.add(entry)
            unassigned_entries.append(entry)

    results: List[Tuple[Dict[int, int], List[int], List[int]]] = []

    def resolve_step(cur_grid_assignments: List[int]) -> None:
        def get_val(idx: int) -> int:
            ant_val = ants.get(idx)
            if ant_val is not None:
                return ant_val
            base_val = base_state[idx]
            if base_val != CELL_WILD:
                return base_val
            entry = grid_entry_by_idx.get(idx)
            if entry is None:
                raise ValueError(f"missing grid entry for wildcard at {idx}")
            assigned = cur_grid_assignments[entry]
            if assigned == -1:
                raise ValueError("grid wildcard not assigned before use")
            return assigned

        new_ants: Dict[int, int] = {}
        deps: Dict[int, List[Tuple[int, List[int]]]] = {}

        def add_dep(entry: int, idx: int, outputs: List[int]) -> None:
            deps.setdefault(entry, []).append((idx, outputs))

        for idx in affected:
            val = get_val(idx)
            if val in (CELL_WALL, CELL_HOLE, CELL_FOOD):
                continue
            n_idx, e_idx, s_idx, w_idx = neighbors[idx]
            n_val = get_val(n_idx) if n_idx != -1 else CELL_WALL
            e_val = get_val(e_idx) if e_idx != -1 else CELL_WALL
            s_val = get_val(s_idx) if s_idx != -1 else CELL_WALL
            w_val = get_val(w_idx) if w_idx != -1 else CELL_WALL
            vals = (n_val, e_val, s_val, w_val)

            if is_ant(val):
                direction = ant_dir(val)
                ahead = vals[direction]
                if ahead in (CELL_FLOOR, CELL_HOLE):
                    continue
                if ahead == CELL_WALL:
                    new_ants[idx] = make_ant(ant_clan(val), (direction + 1) & 3)
                    continue

            if val == CELL_FLOOR:
                if (
                    is_ant(n_val)
                    and ant_dir(n_val) == 2
                    and is_ant(e_val)
                    and ant_dir(e_val) == 3
                    and is_ant(s_val)
                    and ant_dir(s_val) == 0
                    and is_ant(w_val)
                    and ant_dir(w_val) == 1
                ):
                    continue

                for rot in range(4):
                    rot_map = ROT_MAPS[rot]
                    r_w = ROTATED[vals[rot_map[3]]][rot]
                    r_e = ROTATED[vals[rot_map[1]]][rot]
                    r_s = ROTATED[vals[rot_map[2]]][rot]
                    if (
                        is_ant(r_w)
                        and ant_dir(r_w) == 1
                        and is_ant(r_e)
                        and ant_dir(r_e) == 3
                        and is_ant(r_s)
                        and ant_dir(r_s) == 0
                    ):
                        clan = ant_clan(r_s)
                        if clan >= program_count:
                            raise ValueError(f"missing program for clan {clan}")
                        entry = program_entry_index(clan, 3)
                        outputs = [
                            ROTATED[make_ant(clan, turn)][(-rot) & 3]
                            for turn in range(4)
                        ]
                        assigned = assignments[entry]
                        if assigned == -1:
                            add_dep(entry, idx, outputs)
                        else:
                            new_ants[idx] = outputs[assigned]
                        break
                else:
                    for rot in range(4):
                        rot_map = ROT_MAPS[rot]
                        r_n = ROTATED[vals[rot_map[0]]][rot]
                        r_s = ROTATED[vals[rot_map[2]]][rot]
                        if (
                            is_ant(r_n)
                            and ant_dir(r_n) == 2
                            and is_ant(r_s)
                            and ant_dir(r_s) == 0
                        ):
                            break
                    else:
                        for rot in range(4):
                            rot_map = ROT_MAPS[rot]
                            r_s = ROTATED[vals[rot_map[2]]][rot]
                            r_w = ROTATED[vals[rot_map[3]]][rot]
                            if (
                                is_ant(r_w)
                                and ant_dir(r_w) == 1
                                and is_ant(r_s)
                                and ant_dir(r_s) == 0
                            ):
                                clan = ant_clan(r_s)
                                if clan >= program_count:
                                    raise ValueError(f"missing program for clan {clan}")
                                entry = program_entry_index(clan, 2)
                                outputs = [
                                    ROTATED[make_ant(clan, turn)][(-rot) & 3]
                                    for turn in range(4)
                                ]
                                assigned = assignments[entry]
                                if assigned == -1:
                                    add_dep(entry, idx, outputs)
                                else:
                                    new_ants[idx] = outputs[assigned]
                                break
                        else:
                            for rot in range(4):
                                rot_map = ROT_MAPS[rot]
                                r_s = ROTATED[vals[rot_map[2]]][rot]
                                if is_ant(r_s) and ant_dir(r_s) == 0:
                                    clan = ant_clan(r_s)
                                    if clan >= program_count:
                                        raise ValueError(
                                            f"missing program for clan {clan}"
                                        )
                                    entry = program_entry_index(clan, 1)
                                    outputs = [
                                        ROTATED[make_ant(clan, turn)][(-rot) & 3]
                                        for turn in range(4)
                                    ]
                                    assigned = assignments[entry]
                                    if assigned == -1:
                                        add_dep(entry, idx, outputs)
                                    else:
                                        new_ants[idx] = outputs[assigned]
                                    break
                continue

            if is_ant(val):
                direction = ant_dir(val)
                ahead = vals[direction]
                if is_ant(ahead):
                    rel = (ant_dir(ahead) - direction) & 3
                    clan = ant_clan(val)
                    if clan >= program_count:
                        raise ValueError(f"missing program for clan {clan}")
                    entry = program_entry_index(clan, 4 + rel)
                    outputs = [
                        make_ant(clan, (turn + direction) & 3) for turn in range(4)
                    ]
                    assigned = assignments[entry]
                    if assigned == -1:
                        add_dep(entry, idx, outputs)
                    else:
                        new_ants[idx] = outputs[assigned]
                    continue
                new_ants[idx] = val

        unassigned = [entry for entry in deps if assignments[entry] == -1]
        if not unassigned:
            results.append((new_ants, assignments, cur_grid_assignments))
            return

        def recurse(
            i: int, cur_ants: Dict[int, int], cur_assignments: List[int]
        ) -> None:
            if i == len(unassigned):
                results.append((cur_ants, cur_assignments, cur_grid_assignments))
                return
            entry = unassigned[i]
            for val in range(4):
                next_ants = dict(cur_ants)
                next_assignments = cur_assignments[:]
                next_assignments[entry] = val
                for idx, outputs in deps[entry]:
                    out = outputs[val]
                    if is_ant(out):
                        next_ants[idx] = out
                    else:
                        next_ants.pop(idx, None)
                recurse(i + 1, next_ants, next_assignments)

        recurse(0, new_ants, assignments[:])

    if not unassigned_entries:
        resolve_step(grid_assignments)
        return results

    def recurse_grid(
        i: int, cur_grid_assignments: List[int], nonfloor_count: int
    ) -> None:
        if i == len(unassigned_entries):
            resolve_step(cur_grid_assignments)
            return
        entry = unassigned_entries[i]
        for val in grid_values:
            next_nonfloor_count = nonfloor_count
            if val != CELL_FLOOR:
                next_nonfloor_count += 1
                if (
                    grid_nonfloor_limit is not None
                    and next_nonfloor_count > grid_nonfloor_limit
                ):
                    continue
            next_grid_assignments = cur_grid_assignments[:]
            next_grid_assignments[entry] = val
            recurse_grid(i + 1, next_grid_assignments, next_nonfloor_count)

    initial_nonfloor = 0
    if grid_nonfloor_limit is not None:
        for val in grid_assignments:
            if val not in (-1, CELL_FLOOR):
                initial_nonfloor += 1
        if initial_nonfloor > grid_nonfloor_limit:
            return results
    recurse_grid(0, grid_assignments[:], initial_nonfloor)
    return results


def search_partial_programs_sparse_with_grid(
    ants: Dict[int, int],
    neighbors: List[Tuple[int, int, int, int]],
    base_state: List[int],
    assignments: List[int],
    grid_assignments: List[int],
    grid_entry_by_idx: Dict[int, int],
    grid_values: List[int],
    grid_nonfloor_limit: Optional[int],
    program_count: int,
    max_steps: int,
    success_mode: str,
    dist_map: Optional[List[Optional[int]]] = None,
) -> Optional[Tuple[List[int], List[int], int]]:
    seen_by_ants: Dict[
        Tuple[Tuple[int, int], ...],
        List[Tuple[int, Tuple[int, ...], int, Tuple[int, ...], int]],
    ] = {}
    stack: List[Tuple[Dict[int, int], List[int], List[int], int]] = [
        (ants, assignments, grid_assignments, 0)
    ]
    while stack:
        cur_ants, cur_assignments, cur_grid_assignments, step = stack.pop()
        if not cur_ants:
            continue
        if has_success_sparse(cur_ants, neighbors, base_state, success_mode):
            return cur_assignments, cur_grid_assignments, step
        if step >= max_steps:
            continue
        if dist_map is not None:
            remaining = max_steps - step
            min_dist = compute_min_distance_from_ant_positions(cur_ants, dist_map)
            if min_dist is None or min_dist > remaining:
                continue
        key = tuple(sorted(cur_ants.items()))
        cur_prog_values = tuple(cur_assignments)
        cur_grid_values = tuple(cur_grid_assignments)
        cur_prog_mask = assignment_mask(cur_assignments)
        cur_grid_mask = assignment_mask(cur_grid_assignments)
        records = seen_by_ants.get(key, [])
        records, skip = update_dominance_records_pair(
            records,
            cur_prog_values,
            cur_prog_mask,
            cur_grid_values,
            cur_grid_mask,
            step,
        )
        seen_by_ants[key] = records
        if skip:
            continue
        next_states = step_sparse_partial_with_grid(
            cur_ants,
            neighbors,
            base_state,
            cur_assignments,
            cur_grid_assignments,
            grid_entry_by_idx,
            grid_values,
            grid_nonfloor_limit,
            program_count,
        )
        for next_ants, next_assignments, next_grid_assignments in next_states:
            stack.append(
                (next_ants, next_assignments, next_grid_assignments, step + 1)
            )
    return None


def search_partial_programs_sparse(
    ants: Dict[int, int],
    neighbors: List[Tuple[int, int, int, int]],
    base_state: List[int],
    assignments: List[int],
    program_count: int,
    max_steps: int,
    success_mode: str,
    dist_map: Optional[List[Optional[int]]] = None,
) -> Optional[Tuple[List[int], int]]:
    seen_by_ants: Dict[Tuple[Tuple[int, int], ...], List[Tuple[int, Tuple[int, ...], int]]] = {}
    stack: List[Tuple[Dict[int, int], List[int], int]] = [(ants, assignments, 0)]
    while stack:
        cur_ants, cur_assignments, step = stack.pop()
        if not cur_ants:
            continue
        if has_success_sparse(cur_ants, neighbors, base_state, success_mode):
            return cur_assignments, step
        if step >= max_steps:
            continue
        if dist_map is not None:
            remaining = max_steps - step
            min_dist = compute_min_distance_from_ant_positions(cur_ants, dist_map)
            if min_dist is None or min_dist > remaining:
                continue
        key = tuple(sorted(cur_ants.items()))
        cur_values = tuple(cur_assignments)
        cur_mask = assignment_mask(cur_assignments)
        records = seen_by_ants.get(key, [])
        records, skip = update_dominance_records(
            records,
            cur_values,
            cur_mask,
            step,
        )
        seen_by_ants[key] = records
        if skip:
            continue
        next_states = step_sparse_partial(
            cur_ants,
            neighbors,
            base_state,
            cur_assignments,
            program_count,
        )
        for next_ants, next_assignments in next_states:
            stack.append((next_ants, next_assignments, step + 1))
    return None


def build_world(puzzle: Puzzle, replacements: Dict[int, Cell]) -> World:
    grid: List[Cell] = []
    for idx, token in enumerate(puzzle.grid_tokens):
        cell = parse_cell(token)
        if cell.kind == KIND_WILD:
            if idx not in replacements:
                raise ValueError("missing replacement for wildcard")
            grid.append(replacements[idx])
        else:
            grid.append(cell)
    return World(puzzle.width, puzzle.height, grid)


def world_to_key(world: World) -> Tuple[str, ...]:
    tokens = []
    for cell in world.grid:
        if cell.kind == KIND_ANT:
            tokens.append(f"{cell.clan}{DIR_TO_ANT_CHAR[cell.direction]}")
        else:
            tokens.append(f"{cell.kind}")
    return tuple(tokens)


def get_cell(world: World, x: int, y: int) -> Cell:
    if 0 <= x < world.width and 0 <= y < world.height:
        return world.grid[y * world.width + x]
    return Cell(KIND_WALL)


def set_cell(grid: List[Cell], width: int, x: int, y: int, cell: Cell) -> None:
    grid[y * width + x] = cell


def has_success(world: World, success_mode: str = "facing") -> bool:
    if success_mode == "facing":
        directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
        for y in range(world.height):
            for x in range(world.width):
                cell = get_cell(world, x, y)
                if cell.kind != KIND_ANT:
                    continue
                dx, dy = directions[cell.direction]
                ahead = get_cell(world, x + dx, y + dy)
                if ahead.kind == KIND_FOOD:
                    return True
        return False
    if success_mode == "below":
        for y in range(world.height):
            for x in range(world.width):
                cell = get_cell(world, x, y)
                if cell.kind != KIND_ANT or cell.direction != 0:
                    continue
                ahead = get_cell(world, x, y - 1)
                if ahead.kind == KIND_FOOD:
                    return True
        return False
    raise ValueError(f"unknown success mode: {success_mode!r}")


def apply_turn_program(ant: Cell, programs: Programs, p_idx: int) -> Cell:
    # p_idx is 1..7
    if ant.kind != KIND_ANT:
        raise ValueError("apply_turn_program on non-ant")
    if ant.clan >= len(programs.programs):
        raise ValueError("missing program for clan")
    program = programs.programs[ant.clan]
    turn = program[p_idx - 1]
    # turn is relative to ant's current orientation (north in rotated view)
    return Cell(KIND_ANT, ant.clan, turn)


def rotated_view(
    center: Cell, neighbors: Dict[int, Cell], rot: int
) -> Tuple[Cell, Dict[int, Cell]]:
    rot_center = rotate_cell(center, rot)
    rot_neighbors = {d: rotate_cell(neighbors[(d - rot) % 4], rot) for d in range(4)}
    return rot_center, rot_neighbors


def apply_rules(center: Cell, neighbors: Dict[int, Cell], programs: Programs) -> Cell:
    # base: holes/walls/food stay
    if center.kind in (KIND_HOLE, KIND_WALL, KIND_FOOD):
        return center

    # ant moves into empty or hole
    if center.kind == KIND_ANT:
        ahead = neighbors[center.direction]
        if ahead.kind in (KIND_FLOOR, KIND_HOLE):
            return Cell(KIND_FLOOR)
        if ahead.kind == KIND_WALL:
            return Cell(KIND_ANT, center.clan, rotate_dir(center.direction, 1))

    # patterns involving movement into empty floor (rotation-aware, rule order matters)
    if center.kind == KIND_FLOOR:
        # v / >-< / ^  -> -
        for rot in range(4):
            rot_center, rot_neighbors = rotated_view(center, neighbors, rot)
            if rot_center.kind != KIND_FLOOR:
                continue
            north = rot_neighbors[0]
            east = rot_neighbors[1]
            south = rot_neighbors[2]
            west = rot_neighbors[3]
            if (
                north.kind == KIND_ANT
                and north.direction == 2
                and west.kind == KIND_ANT
                and west.direction == 1
                and east.kind == KIND_ANT
                and east.direction == 3
                and south.kind == KIND_ANT
                and south.direction == 0
            ):
                return Cell(KIND_FLOOR)

        # * / >-< / ^ -> p3 of bottom ant
        for rot in range(4):
            rot_center, rot_neighbors = rotated_view(center, neighbors, rot)
            if rot_center.kind != KIND_FLOOR:
                continue
            east = rot_neighbors[1]
            south = rot_neighbors[2]
            west = rot_neighbors[3]
            if (
                west.kind == KIND_ANT
                and west.direction == 1
                and east.kind == KIND_ANT
                and east.direction == 3
                and south.kind == KIND_ANT
                and south.direction == 0
            ):
                out = apply_turn_program(south, programs, 3)
                return rotate_cell(out, (-rot) % 4)

        # v / *-* / ^ -> -
        for rot in range(4):
            rot_center, rot_neighbors = rotated_view(center, neighbors, rot)
            if rot_center.kind != KIND_FLOOR:
                continue
            north = rot_neighbors[0]
            south = rot_neighbors[2]
            if (
                north.kind == KIND_ANT
                and north.direction == 2
                and south.kind == KIND_ANT
                and south.direction == 0
            ):
                return Cell(KIND_FLOOR)

        # * / >-* / ^ -> p2 of bottom ant
        for rot in range(4):
            rot_center, rot_neighbors = rotated_view(center, neighbors, rot)
            if rot_center.kind != KIND_FLOOR:
                continue
            south = rot_neighbors[2]
            west = rot_neighbors[3]
            if (
                west.kind == KIND_ANT
                and west.direction == 1
                and south.kind == KIND_ANT
                and south.direction == 0
            ):
                out = apply_turn_program(south, programs, 2)
                return rotate_cell(out, (-rot) % 4)

        # * / *-* / ^ -> p1 of bottom ant
        for rot in range(4):
            rot_center, rot_neighbors = rotated_view(center, neighbors, rot)
            if rot_center.kind != KIND_FLOOR:
                continue
            south = rot_neighbors[2]
            if south.kind == KIND_ANT and south.direction == 0:
                out = apply_turn_program(south, programs, 1)
                return rotate_cell(out, (-rot) % 4)

    # patterns for center ant with ant ahead (p4..p7)
    if center.kind == KIND_ANT:
        ahead = neighbors[center.direction]
        if ahead.kind == KIND_ANT:
            rel_dir = rotate_dir(ahead.direction, -center.direction)
            center_rot = rotate_cell(center, -center.direction)
            if rel_dir == 0:
                out = apply_turn_program(center_rot, programs, 4)
                return rotate_cell(out, center.direction)
            if rel_dir == 1:
                out = apply_turn_program(center_rot, programs, 5)
                return rotate_cell(out, center.direction)
            if rel_dir == 2:
                out = apply_turn_program(center_rot, programs, 6)
                return rotate_cell(out, center.direction)
            if rel_dir == 3:
                out = apply_turn_program(center_rot, programs, 7)
                return rotate_cell(out, center.direction)

    # default: keep as-is
    return center


def step_world(world: World, programs: Programs) -> World:
    new_grid: List[Cell] = [Cell(KIND_FLOOR) for _ in range(world.width * world.height)]
    for y in range(world.height):
        for x in range(world.width):
            center = get_cell(world, x, y)
            neighbors = {
                0: get_cell(world, x, y - 1),
                1: get_cell(world, x + 1, y),
                2: get_cell(world, x, y + 1),
                3: get_cell(world, x - 1, y),
            }
            out_cell = apply_rules(center, neighbors, programs)
            set_cell(new_grid, world.width, x, y, out_cell)
    return World(world.width, world.height, new_grid)


def simulate(
    world: World,
    programs: Programs,
    max_steps: int = 10000,
    success_mode: str = "facing",
) -> Tuple[bool, int]:
    seen = set()
    for step in range(max_steps + 1):
        if has_success(world, success_mode):
            return True, step
        key = world_to_key(world)
        if key in seen:
            return False, step
        seen.add(key)
        world = step_world(world, programs)
    return False, max_steps


def has_success_fast(
    state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    food_indices: List[int],
    mode: str = "facing",
) -> bool:
    if mode == "below":
        for idx in food_indices:
            s_idx = neighbors[idx][2]
            if s_idx != -1 and is_ant(state[s_idx]) and ant_dir(state[s_idx]) == 0:
                return True
        return False
    if mode == "facing":
        for idx in food_indices:
            n_idx, e_idx, s_idx, w_idx = neighbors[idx]
            if n_idx != -1 and is_ant(state[n_idx]) and ant_dir(state[n_idx]) == 2:
                return True
            if e_idx != -1 and is_ant(state[e_idx]) and ant_dir(state[e_idx]) == 3:
                return True
            if s_idx != -1 and is_ant(state[s_idx]) and ant_dir(state[s_idx]) == 0:
                return True
            if w_idx != -1 and is_ant(state[w_idx]) and ant_dir(state[w_idx]) == 1:
                return True
        return False
    raise ValueError(f"unknown success mode: {mode!r}")


def find_wildcards(puzzle: Puzzle) -> List[int]:
    wilds = []
    for idx, token in enumerate(puzzle.grid_tokens):
        if parse_cell(token).kind == KIND_WILD:
            wilds.append(idx)
    return wilds


def solve_puzzle1(puzzle: Puzzle, max_steps: int = 5000) -> None:
    wilds = find_wildcards(puzzle)
    if len(wilds) != 1:
        raise ValueError("puzzle1 expected 1 wildcard")
    wild_idx = wilds[0]

    base_state: List[int] = []
    for idx, token in enumerate(puzzle.grid_tokens):
        cell = parse_cell(token)
        if cell.kind == KIND_WILD:
            base_state.append(CELL_FLOOR)
        else:
            base_state.append(encode_cell(cell))

    food_indices = [idx for idx, val in enumerate(base_state) if val == CELL_FOOD]
    neighbors = build_neighbors(puzzle.width, puzzle.height)

    clan0_prog = [DIR_TO_IDX[ch] for ch in puzzle.programs[0]]

    all_prog1: List[Tuple[List[int], str]] = []
    for i in range(4 ** 7):
        temp = i
        prog = []
        for _ in range(7):
            prog.append(temp & 3)
            temp >>= 2
        prog_str = "".join(DIRS[d] for d in prog)
        all_prog1.append((prog, prog_str))

    quick_candidates: List[Tuple[str, int]] = [
        ("floor", CELL_FLOOR),
        ("wall", CELL_WALL),
        ("hole", CELL_HOLE),
    ]
    for d in range(4):
        quick_candidates.append((f"0{DIR_TO_ANT_CHAR[d]}", make_ant(0, d)))

    candidates: List[Tuple[str, int]] = []
    for d in range(4):
        candidates.append((f"1{DIR_TO_ANT_CHAR[d]}", make_ant(1, d)))

    for desc, cell_val in quick_candidates:
        state = base_state[:]
        state[wild_idx] = cell_val
        programs = [clan0_prog]
        ok, steps = simulate_fast(
            state,
            neighbors,
            programs,
            food_indices,
            max_steps=max_steps,
        )
        if ok:
            print(f"Solved with {desc} (clan0 only) steps={steps}")
            return

    for desc, cell_val in candidates:
        state = base_state[:]
        state[wild_idx] = cell_val
        for prog1, prog1_str in all_prog1:
            programs = [clan0_prog, prog1]
            ok, steps = simulate_fast(
                state,
                neighbors,
                programs,
                food_indices,
                max_steps=max_steps,
            )
            if ok:
                print(f"Solved with {desc} prog={prog1_str} steps={steps}")
                return

    print(f"No solution found up to {max_steps} steps")


def summarize_puzzles(puzzles: List[Puzzle], match: Optional[str]) -> None:
    for puzzle in puzzles:
        if match and match not in puzzle.title:
            continue
        cells = [parse_cell(token) for token in puzzle.grid_tokens]
        grid_wilds = sum(1 for cell in cells if cell.kind == KIND_WILD)
        ant_count = sum(1 for cell in cells if cell.kind == KIND_ANT)
        clans = sorted({cell.clan for cell in cells if cell.kind == KIND_ANT})
        program_lengths = Counter(len(line) for line in puzzle.programs)
        program_wilds = sum(line.count("*") for line in puzzle.programs)
        length_desc = ",".join(
            f"{length}x{count}" for length, count in sorted(program_lengths.items())
        )
        clans_desc = ",".join(str(clan) for clan in clans) if clans else "-"
        print(
            f"{puzzle.title} | {puzzle.width}x{puzzle.height} | "
            f"wild={grid_wilds} ants={ant_count} clans={clans_desc} | "
            f"programs={len(puzzle.programs)} len={length_desc} wilds={program_wilds}"
        )


def build_base_state(
    puzzle: Puzzle,
) -> Tuple[List[int], List[int], List[int]]:
    base_state: List[int] = []
    wilds: List[int] = []
    for idx, token in enumerate(puzzle.grid_tokens):
        cell = parse_cell(token)
        if cell.kind == KIND_WILD:
            wilds.append(idx)
            base_state.append(CELL_FLOOR)
        else:
            base_state.append(encode_cell(cell))
    food_indices = [idx for idx, val in enumerate(base_state) if val == CELL_FOOD]
    return base_state, wilds, food_indices


def build_base_state_with_wildcards(
    puzzle: Puzzle,
) -> Tuple[List[int], List[int], List[int]]:
    base_state: List[int] = []
    wilds: List[int] = []
    for idx, token in enumerate(puzzle.grid_tokens):
        cell = parse_cell(token)
        if cell.kind == KIND_WILD:
            wilds.append(idx)
            base_state.append(CELL_WILD)
        else:
            base_state.append(encode_cell(cell))
    food_indices = [idx for idx, val in enumerate(base_state) if val == CELL_FOOD]
    return base_state, wilds, food_indices


def parse_program_assignments(programs: List[str]) -> List[int]:
    assignments: List[int] = []
    for line in programs:
        if len(line) < 7:
            raise ValueError(f"unsupported program length: {line!r}")
        for ch in line[:7]:
            if ch == "*":
                assignments.append(-1)
            elif ch in DIR_TO_IDX:
                assignments.append(DIR_TO_IDX[ch])
            else:
                raise ValueError(f"invalid program char: {ch!r}")
        for ch in line[7:]:
            if ch != "*" and ch not in DIR_TO_IDX:
                raise ValueError(f"invalid extra program char: {ch!r}")
    return assignments


def parse_grid_wild_values(values: str) -> List[int]:
    mapping = {
        "floor": CELL_FLOOR,
        "wall": CELL_WALL,
        "hole": CELL_HOLE,
        "food": CELL_FOOD,
        "-": CELL_FLOOR,
        "#": CELL_WALL,
        "o": CELL_HOLE,
        "$": CELL_FOOD,
    }
    result: List[int] = []
    for part in values.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key in mapping:
            val = mapping[key]
        else:
            m = re.fullmatch(r"([0-9])([\^>v<nesw])", key)
            if not m:
                raise ValueError(f"unknown grid wildcard value: {part!r}")
            clan = int(m.group(1))
            dir_ch = m.group(2)
            if dir_ch in ANT_CHAR_TO_DIR:
                direction = ANT_CHAR_TO_DIR[dir_ch]
            else:
                direction = DIR_TO_IDX[dir_ch.upper()]
            val = make_ant(clan, direction)
        if val not in result:
            result.append(val)
    if not result:
        raise ValueError("grid wildcard values cannot be empty")
    return result


def parse_position_list(value: str, width: int, height: int) -> List[int]:
    if not value:
        return []
    indices = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        if "," not in part:
            raise ValueError(f"invalid position: {part!r}")
        x_str, y_str = (p.strip() for p in part.split(",", 1))
        if not (x_str.isdigit() and y_str.isdigit()):
            raise ValueError(f"invalid position: {part!r}")
        x = int(x_str)
        y = int(y_str)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"position out of bounds: {part!r}")
        indices.append(y * width + x)
    return indices


def parse_rect(value: str, width: int, height: int) -> List[int]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4 or any(not p.isdigit() for p in parts):
        raise ValueError(f"invalid rect: {value!r}")
    x0, y0, x1, y1 = map(int, parts)
    if x0 > x1 or y0 > y1:
        raise ValueError(f"invalid rect: {value!r}")
    if not (0 <= x0 < width and 0 <= x1 < width and 0 <= y0 < height and 0 <= y1 < height):
        raise ValueError(f"rect out of bounds: {value!r}")
    indices = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            indices.append(y * width + x)
    return indices


def parse_dir_value(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    upper = text.upper()
    if upper in DIR_TO_IDX:
        return DIR_TO_IDX[upper]
    if text.isdigit():
        val = int(text)
        if val in (0, 1, 2, 3):
            return val
    raise ValueError(f"invalid direction: {value!r}")


def parse_fixed_ants(
    value: Optional[str],
    width: int,
    height: int,
) -> Dict[int, int]:
    if not value:
        return {}
    entries = [item.strip() for item in value.split(";") if item.strip()]
    if not entries:
        return {}
    fixed: Dict[int, int] = {}
    for entry in entries:
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) != 4:
            raise ValueError(f"invalid fixed ant: {entry!r}")
        x_str, y_str, clan_str, dir_str = parts
        if not (x_str.isdigit() and y_str.isdigit() and clan_str.isdigit()):
            raise ValueError(f"invalid fixed ant: {entry!r}")
        x = int(x_str)
        y = int(y_str)
        clan = int(clan_str)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"fixed ant out of bounds: {entry!r}")
        if not (0 <= clan < MAX_CLANS):
            raise ValueError(f"invalid clan id: {entry!r}")
        dir_key = dir_str.upper()
        if dir_str in ANT_CHAR_TO_DIR:
            direction = ANT_CHAR_TO_DIR[dir_str]
        elif dir_key in DIR_TO_IDX:
            direction = DIR_TO_IDX[dir_key]
        elif dir_str.isdigit():
            direction = int(dir_str)
        else:
            raise ValueError(f"invalid direction: {entry!r}")
        if direction not in (0, 1, 2, 3):
            raise ValueError(f"invalid direction: {entry!r}")
        idx = y * width + x
        if idx in fixed:
            raise ValueError(f"duplicate fixed ant: {entry!r}")
        fixed[idx] = make_ant(clan, direction)
    return fixed


def compute_food_adjacent_positions(
    base_state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
) -> List[int]:
    targets = set()
    for idx, val in enumerate(base_state):
        if val != CELL_FOOD:
            continue
        for n_idx in neighbors[idx]:
            if n_idx == -1:
                continue
            if base_state[n_idx] in (CELL_WALL, CELL_HOLE):
                continue
            targets.add(n_idx)
    return sorted(targets)


def compute_distances_to_targets(
    base_state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    targets: List[int],
) -> List[Optional[int]]:
    dist: List[Optional[int]] = [None] * len(base_state)
    queue = deque(targets)
    for idx in targets:
        dist[idx] = 0
    while queue:
        idx = queue.popleft()
        cur = dist[idx]
        for n_idx in neighbors[idx]:
            if n_idx == -1:
                continue
            if dist[n_idx] is not None:
                continue
            if base_state[n_idx] in (CELL_WALL, CELL_HOLE):
                continue
            dist[n_idx] = cur + 1 if cur is not None else 1
            queue.append(n_idx)
    return dist


def filter_reachable_positions(
    base_state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    positions: List[int],
) -> List[int]:
    targets = compute_food_adjacent_positions(base_state, neighbors)
    if not targets:
        return []
    dist = compute_distances_to_targets(base_state, neighbors, targets)
    return [idx for idx in positions if dist[idx] is not None]


def compute_min_distance_from_positions(
    positions: List[int],
    dist: List[Optional[int]],
) -> Optional[int]:
    best: Optional[int] = None
    for idx in positions:
        d = dist[idx]
        if d is None:
            continue
        best = d if best is None or d < best else best
    return best


def compute_min_distance_from_ant_positions(
    ants: Dict[int, int],
    dist: Optional[List[Optional[int]]],
) -> Optional[int]:
    if dist is None:
        return None
    best: Optional[int] = None
    for pos in ants:
        d = dist[pos]
        if d is None:
            continue
        best = d if best is None or d < best else best
    return best


def build_optimistic_dist_map(
    base_state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
) -> Optional[List[Optional[int]]]:
    optimistic = [
        CELL_FLOOR if val == CELL_WILD else val for val in base_state
    ]
    targets = compute_food_adjacent_positions(optimistic, neighbors)
    if not targets:
        return None
    return compute_distances_to_targets(optimistic, neighbors, targets)


def assignment_mask(values: List[int], wildcard: int = -1) -> int:
    mask = 0
    for idx, val in enumerate(values):
        if val != wildcard:
            mask |= 1 << idx
    return mask


def values_match_mask(mask: int, left: Tuple[int, ...], right: Tuple[int, ...]) -> bool:
    m = mask
    while m:
        bit = m & -m
        idx = bit.bit_length() - 1
        if left[idx] != right[idx]:
            return False
        m ^= bit
    return True


def update_dominance_records(
    records: List[Tuple[int, Tuple[int, ...], int]],
    cur_values: Tuple[int, ...],
    cur_mask: int,
    step: int,
) -> Tuple[List[Tuple[int, Tuple[int, ...], int]], bool]:
    new_records: List[Tuple[int, Tuple[int, ...], int]] = []
    skip = False
    for rec_mask, rec_values, rec_step in records:
        rec_dominates = (
            (rec_mask & ~cur_mask) == 0
            and values_match_mask(rec_mask, rec_values, cur_values)
        )
        if rec_dominates:
            if rec_step <= step:
                skip = True
                new_records.append((rec_mask, rec_values, rec_step))
                continue
            new_records.append((rec_mask, rec_values, step))
            continue
        cur_dominates = (
            (cur_mask & ~rec_mask) == 0
            and values_match_mask(cur_mask, cur_values, rec_values)
            and step <= rec_step
        )
        if cur_dominates:
            continue
        new_records.append((rec_mask, rec_values, rec_step))
    if not skip:
        new_records.append((cur_mask, cur_values, step))
    return new_records, skip


def update_dominance_records_pair(
    records: List[Tuple[int, Tuple[int, ...], int, Tuple[int, ...], int]],
    cur_prog_values: Tuple[int, ...],
    cur_prog_mask: int,
    cur_grid_values: Tuple[int, ...],
    cur_grid_mask: int,
    step: int,
) -> Tuple[List[Tuple[int, Tuple[int, ...], int, Tuple[int, ...], int]], bool]:
    new_records: List[Tuple[int, Tuple[int, ...], int, Tuple[int, ...], int]] = []
    skip = False
    for rec_prog_mask, rec_prog_values, rec_grid_mask, rec_grid_values, rec_step in records:
        rec_dominates = (
            (rec_prog_mask & ~cur_prog_mask) == 0
            and (rec_grid_mask & ~cur_grid_mask) == 0
            and values_match_mask(rec_prog_mask, rec_prog_values, cur_prog_values)
            and values_match_mask(rec_grid_mask, rec_grid_values, cur_grid_values)
        )
        if rec_dominates:
            if rec_step <= step:
                skip = True
                new_records.append(
                    (rec_prog_mask, rec_prog_values, rec_grid_mask, rec_grid_values, rec_step)
                )
                continue
            new_records.append(
                (rec_prog_mask, rec_prog_values, rec_grid_mask, rec_grid_values, step)
            )
            continue
        cur_dominates = (
            (cur_prog_mask & ~rec_prog_mask) == 0
            and (cur_grid_mask & ~rec_grid_mask) == 0
            and values_match_mask(cur_prog_mask, cur_prog_values, rec_prog_values)
            and values_match_mask(cur_grid_mask, cur_grid_values, rec_grid_values)
            and step <= rec_step
        )
        if cur_dominates:
            continue
        new_records.append(
            (rec_prog_mask, rec_prog_values, rec_grid_mask, rec_grid_values, rec_step)
        )
    if not skip:
        new_records.append(
            (cur_prog_mask, cur_prog_values, cur_grid_mask, cur_grid_values, step)
        )
    return new_records, skip


def build_wild_position_filter(
    puzzle: Puzzle,
    positions: Optional[str],
    rect: Optional[str],
    max_dist: Optional[int],
    limit: Optional[int],
) -> Optional[List[int]]:
    if not positions and not rect:
        selected = None
    else:
        selected = []
    wilds = set(find_wildcards(puzzle))
    if positions:
        pos_indices = parse_position_list(positions, puzzle.width, puzzle.height)
        invalid = [idx for idx in pos_indices if idx not in wilds]
        if invalid:
            coords = [divmod(idx, puzzle.width)[::-1] for idx in invalid]
            raise ValueError(f"non-wild positions: {coords}")
        selected.extend(pos_indices)
    if rect:
        rect_indices = parse_rect(rect, puzzle.width, puzzle.height)
        selected.extend([idx for idx in rect_indices if idx in wilds])
    if selected is None:
        selected = list(wilds)
    if max_dist is not None or limit is not None:
        base_state, _wilds, _food_indices = build_base_state(puzzle)
        neighbors = build_neighbors(puzzle.width, puzzle.height)
        targets = compute_food_adjacent_positions(base_state, neighbors)
        if not targets:
            return []
        dist = compute_distances_to_targets(base_state, neighbors, targets)
        filtered = []
        for idx in selected:
            d = dist[idx]
            if d is None:
                continue
            if max_dist is not None and d > max_dist:
                continue
            filtered.append((d, idx))
        if limit is not None:
            filtered.sort()
            filtered = filtered[:limit]
        selected = [idx for _d, idx in filtered]
    if not selected:
        return []
    return sorted(set(selected))


def compute_min_food_distance(
    base_state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
) -> Optional[int]:
    targets = compute_food_adjacent_positions(base_state, neighbors)
    if not targets:
        return None
    dist = compute_distances_to_targets(base_state, neighbors, targets)
    best: Optional[int] = None
    for idx, val in enumerate(base_state):
        if not is_ant(val):
            continue
        d = dist[idx]
        if d is None:
            continue
        best = d if best is None or d < best else best
    return best


def parse_ant_clans(value: Optional[str], program_count: int) -> Optional[List[int]]:
    if not value:
        return None
    clans: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"invalid clan id: {part!r}")
        clan = int(part)
        if clan < 0 or clan >= program_count:
            raise ValueError(f"clan id out of range: {clan}")
        if clan not in clans:
            clans.append(clan)
    if not clans:
        return None
    return clans


def build_ant_specs(program_count: int, allowed_clans: Optional[List[int]]) -> List[int]:
    clans = allowed_clans if allowed_clans is not None else list(range(program_count))
    specs: List[int] = []
    for clan in clans:
        for direction in range(4):
            specs.append((clan * 4) + direction)
    return specs


def program_entry_index(clan: int, p_idx: int) -> int:
    return (clan * 7) + (p_idx - 1)


def step_state_partial(
    state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    assignments: List[int],
    program_count: int,
) -> List[Tuple[List[int], List[int]]]:
    count = len(neighbors)
    new_state = [CELL_FLOOR] * count
    deps: Dict[int, List[Tuple[int, List[int]]]] = {}
    rotated = ROTATED
    rot_maps = ROT_MAPS

    def add_dep(entry: int, idx: int, outputs: List[int]) -> None:
        deps.setdefault(entry, []).append((idx, outputs))

    for idx, val in enumerate(state):
        if val in (CELL_WALL, CELL_HOLE, CELL_FOOD):
            new_state[idx] = val
            continue
        n_idx, e_idx, s_idx, w_idx = neighbors[idx]
        n_val = state[n_idx] if n_idx != -1 else CELL_WALL
        e_val = state[e_idx] if e_idx != -1 else CELL_WALL
        s_val = state[s_idx] if s_idx != -1 else CELL_WALL
        w_val = state[w_idx] if w_idx != -1 else CELL_WALL
        vals = (n_val, e_val, s_val, w_val)

        if is_ant(val):
            direction = ant_dir(val)
            ahead = vals[direction]
            if ahead in (CELL_FLOOR, CELL_HOLE):
                new_state[idx] = CELL_FLOOR
                continue
            if ahead == CELL_WALL:
                new_state[idx] = make_ant(ant_clan(val), (direction + 1) & 3)
                continue

        if val == CELL_FLOOR:
            if (
                is_ant(n_val)
                and ant_dir(n_val) == 2
                and is_ant(e_val)
                and ant_dir(e_val) == 3
                and is_ant(s_val)
                and ant_dir(s_val) == 0
                and is_ant(w_val)
                and ant_dir(w_val) == 1
            ):
                new_state[idx] = CELL_FLOOR
                continue

            for rot in range(4):
                rot_map = rot_maps[rot]
                r_w = rotated[vals[rot_map[3]]][rot]
                r_e = rotated[vals[rot_map[1]]][rot]
                r_s = rotated[vals[rot_map[2]]][rot]
                if (
                    is_ant(r_w)
                    and ant_dir(r_w) == 1
                    and is_ant(r_e)
                    and ant_dir(r_e) == 3
                    and is_ant(r_s)
                    and ant_dir(r_s) == 0
                ):
                    clan = ant_clan(r_s)
                    if clan >= program_count:
                        raise ValueError(f"missing program for clan {clan}")
                    entry = program_entry_index(clan, 3)
                    outputs = [
                        rotated[make_ant(clan, turn)][(-rot) & 3] for turn in range(4)
                    ]
                    add_dep(entry, idx, outputs)
                    break
            else:
                for rot in range(4):
                    rot_map = rot_maps[rot]
                    r_n = rotated[vals[rot_map[0]]][rot]
                    r_s = rotated[vals[rot_map[2]]][rot]
                    if (
                        is_ant(r_n)
                        and ant_dir(r_n) == 2
                        and is_ant(r_s)
                        and ant_dir(r_s) == 0
                    ):
                        new_state[idx] = CELL_FLOOR
                        break
                else:
                    for rot in range(4):
                        rot_map = rot_maps[rot]
                        r_s = rotated[vals[rot_map[2]]][rot]
                        r_w = rotated[vals[rot_map[3]]][rot]
                        if is_ant(r_w) and ant_dir(r_w) == 1 and is_ant(r_s) and ant_dir(r_s) == 0:
                            clan = ant_clan(r_s)
                            if clan >= program_count:
                                raise ValueError(f"missing program for clan {clan}")
                            entry = program_entry_index(clan, 2)
                            outputs = [
                                rotated[make_ant(clan, turn)][(-rot) & 3]
                                for turn in range(4)
                            ]
                            add_dep(entry, idx, outputs)
                            break
                    else:
                        for rot in range(4):
                            rot_map = rot_maps[rot]
                            r_s = rotated[vals[rot_map[2]]][rot]
                            if is_ant(r_s) and ant_dir(r_s) == 0:
                                clan = ant_clan(r_s)
                                if clan >= program_count:
                                    raise ValueError(f"missing program for clan {clan}")
                                entry = program_entry_index(clan, 1)
                                outputs = [
                                    rotated[make_ant(clan, turn)][(-rot) & 3]
                                    for turn in range(4)
                                ]
                                add_dep(entry, idx, outputs)
                                break
                        else:
                            new_state[idx] = CELL_FLOOR
            continue

        if is_ant(val):
            direction = ant_dir(val)
            ahead = vals[direction]
            if is_ant(ahead):
                rel = (ant_dir(ahead) - direction) & 3
                clan = ant_clan(val)
                if clan >= program_count:
                    raise ValueError(f"missing program for clan {clan}")
                entry = program_entry_index(clan, 4 + rel)
                outputs = [make_ant(clan, (turn + direction) & 3) for turn in range(4)]
                add_dep(entry, idx, outputs)
                continue

        new_state[idx] = val

    for entry, cells in deps.items():
        assigned = assignments[entry]
        if assigned != -1:
            for idx, outputs in cells:
                new_state[idx] = outputs[assigned]

    unassigned = [entry for entry in deps if assignments[entry] == -1]
    if not unassigned:
        return [(new_state, assignments)]

    results: List[Tuple[List[int], List[int]]] = []

    def recurse(i: int, cur_state: List[int], cur_assignments: List[int]) -> None:
        if i == len(unassigned):
            results.append((cur_state, cur_assignments))
            return
        entry = unassigned[i]
        for val in range(4):
            next_state = cur_state[:]
            next_assignments = cur_assignments[:]
            next_assignments[entry] = val
            for idx, outputs in deps[entry]:
                next_state[idx] = outputs[val]
            recurse(i + 1, next_state, next_assignments)

    recurse(0, new_state, assignments[:])
    return results


def search_partial_programs(
    state: List[int],
    neighbors: List[Tuple[int, int, int, int]],
    food_indices: List[int],
    assignments: List[int],
    program_count: int,
    max_steps: int,
    success_mode: str,
) -> Optional[Tuple[List[int], int]]:
    seen = set()
    stack: List[Tuple[List[int], List[int], int]] = [(state, assignments, 0)]
    while stack:
        cur_state, cur_assignments, step = stack.pop()
        if has_success_fast(cur_state, neighbors, food_indices, success_mode):
            return cur_assignments, step
        if step >= max_steps:
            continue
        key = (bytes(cur_state), tuple(cur_assignments))
        if key in seen:
            continue
        seen.add(key)
        next_states = step_state_partial(
            cur_state, neighbors, cur_assignments, program_count
        )
        for next_state, next_assignments in next_states:
            stack.append((next_state, next_assignments, step + 1))
    return None


def render_programs(assignments: List[int], template_programs: List[str]) -> List[str]:
    lines = []
    for clan, template in enumerate(template_programs):
        chars = []
        for idx in range(7):
            val = assignments[(clan * 7) + idx]
            chars.append("*" if val == -1 else DIRS[val])
        for ch in template[7:]:
            if ch == "*":
                chars.append("N")
            elif ch in DIR_TO_IDX:
                chars.append(ch)
            else:
                raise ValueError(f"invalid extra program char in template: {template!r}")
        lines.append("".join(chars))
    return lines


def render_grid_assignments(
    grid_assignments: List[int],
    grid_positions: List[int],
    width: int,
) -> str:
    items = []
    for entry, val in enumerate(grid_assignments):
        if val in (-1, CELL_FLOOR):
            continue
        y, x = divmod(grid_positions[entry], width)
        if val == CELL_WALL:
            char = "#"
        elif val == CELL_HOLE:
            char = "o"
        else:
            char = "?"
        items.append(f"{x},{y},{char}")
    return " | ".join(items) if items else "-"


def fill_program_template(
    template: str,
    p1: Optional[int] = None,
    apply_p1: bool = False,
) -> str:
    if len(template) < 7:
        raise ValueError(f"unsupported program length: {template!r}")
    out = []
    for i, ch in enumerate(template):
        if ch == "*":
            if apply_p1 and i == 0:
                if p1 is None:
                    raise ValueError("p1 is required when apply_p1 is true")
                out.append(DIRS[p1])
            else:
                out.append("N")
            continue
        if ch not in DIR_TO_IDX:
            raise ValueError(f"invalid program char: {ch!r}")
        out.append(ch)
    return "".join(out)


def carve_solution_replacements(
    puzzle: Puzzle,
    solution: CarveSolution,
    clan: int = 0,
) -> Dict[int, Cell]:
    all_wilds = find_wildcards(puzzle)
    wild_map = {idx: bit for bit, idx in enumerate(solution.wild_list)}
    replacements: Dict[int, Cell] = {}
    for idx in all_wilds:
        if idx == solution.start_idx:
            replacements[idx] = Cell(KIND_ANT, clan, solution.start_dir)
            continue
        bit = wild_map.get(idx)
        if bit is not None and ((solution.floor_mask >> bit) & 1) == 1:
            replacements[idx] = Cell(KIND_FLOOR)
        else:
            replacements[idx] = Cell(KIND_WALL)
    return replacements


def write_ant_world(
    path: Path,
    title: str,
    programs: List[str],
    world: World,
) -> None:
    lines: List[str] = [title]
    lines.extend(programs)
    lines.append(f"{world.width} {world.height}")
    for y in range(world.height):
        row = []
        for x in range(world.width):
            row.append(cell_to_token(world.grid[y * world.width + x]))
        lines.append("".join(row))
    path.write_text("\n".join(lines) + "\n")


def search_carve_single_ant(
    puzzle: Puzzle,
    max_steps: int,
    success_mode: str,
    p1_dir: Optional[int] = None,
    start_positions: Optional[List[int]] = None,
    allowed_grid_positions: Optional[List[int]] = None,
    greedy: bool = False,
) -> Optional[str]:
    solution = find_carve_single_ant_solution(
        puzzle,
        max_steps=max_steps,
        success_mode=success_mode,
        p1_dir=p1_dir,
        start_positions=start_positions,
        allowed_grid_positions=allowed_grid_positions,
        greedy=greedy,
    )
    if solution is None:
        return None
    y, x = divmod(solution.start_idx, puzzle.width)
    floors = int(solution.floor_mask.bit_count())
    return (
        f"ant=({x},{y}) start={DIRS[solution.start_dir]} "
        f"p1={DIRS[solution.p1]} steps<={max_steps} floors={floors}"
    )


def find_carve_single_ant_solution(
    puzzle: Puzzle,
    max_steps: int,
    success_mode: str,
    p1_dir: Optional[int] = None,
    start_positions: Optional[List[int]] = None,
    allowed_grid_positions: Optional[List[int]] = None,
    greedy: bool = False,
) -> Optional[CarveSolution]:
    base_state, wilds, food_indices = build_base_state_with_wildcards(puzzle)
    if not wilds:
        return None
    neighbors = build_neighbors(puzzle.width, puzzle.height)
    targets = compute_food_adjacent_positions(base_state, neighbors)
    if not targets:
        return None
    optimistic_base, _wilds_floor, _ = build_base_state(puzzle)
    dist = compute_distances_to_targets(optimistic_base, neighbors, targets)

    allowed_wilds = set(allowed_grid_positions) if allowed_grid_positions else set(wilds)
    if start_positions is None:
        start_positions = [idx for idx in wilds if idx in allowed_wilds]
    else:
        start_positions = [idx for idx in start_positions if idx in allowed_wilds]
    if not start_positions:
        return None

    wild_list = [idx for idx in wilds if idx in allowed_wilds]
    wild_map = {idx: bit for bit, idx in enumerate(wild_list)}

    def heuristic(idx: int) -> int:
        d = dist[idx]
        return d if d is not None else 0

    def get_kind(idx: int, floor_mask: int, wall_mask: int) -> int:
        if idx == -1:
            return CELL_WALL
        val = base_state[idx]
        if val != CELL_WILD:
            return val
        bit = wild_map.get(idx)
        if bit is None:
            return CELL_WALL
        mask = 1 << bit
        if wall_mask & mask:
            return CELL_WALL
        if floor_mask & mask:
            return CELL_FLOOR
        return CELL_WILD

    p1_dirs = [p1_dir] if p1_dir is not None else list(range(4))

    for p1 in p1_dirs:
        for start_idx in start_positions:
            if start_idx not in wild_map:
                continue
            start_bit = 1 << wild_map[start_idx]
            for start_dir in range(4):
                # state: (priority, steps, pos, dir, floor_mask, wall_mask)
                heap: List[Tuple[int, int, int, int, int, int]] = []
                seen: Dict[Tuple[int, int, int, int], int] = {}
                heapq.heappush(
                    heap,
                    (
                        heuristic(start_idx),
                        0,
                        start_idx,
                        start_dir,
                        start_bit,
                        0,
                    ),
                )
                while heap:
                    _score, steps, pos, direction, floor_mask, wall_mask = heapq.heappop(
                        heap
                    )
                    if steps > max_steps:
                        continue
                    ahead = neighbors[pos][direction]
                    if ahead != -1 and base_state[ahead] == CELL_FOOD:
                        if success_mode == "facing":
                            success = True
                        elif success_mode == "below":
                            success = direction == 0
                        else:
                            raise ValueError(f"unknown success mode: {success_mode!r}")
                        if success:
                            return CarveSolution(
                                start_idx=start_idx,
                                start_dir=start_dir,
                                p1=p1,
                                floor_mask=floor_mask,
                                wall_mask=wall_mask,
                                wild_list=tuple(wild_list),
                            )
                    key = (pos, direction, floor_mask, wall_mask)
                    prev = seen.get(key)
                    if prev is not None and prev <= steps:
                        continue
                    seen[key] = steps

                    kind = get_kind(ahead, floor_mask, wall_mask)
                    if kind == CELL_WALL:
                        new_dir = (direction + 1) & 3
                        heapq.heappush(
                            heap,
                            (
                                steps + 1 + heuristic(pos),
                                steps + 1,
                                pos,
                                new_dir,
                                floor_mask,
                                wall_mask,
                            ),
                        )
                        continue
                    if kind == CELL_HOLE:
                        continue
                    if kind in (CELL_FLOOR, CELL_FOOD):
                        new_pos = ahead
                        new_dir = (direction + p1) & 3
                        heapq.heappush(
                            heap,
                            (
                                steps + 1 + heuristic(new_pos),
                                steps + 1,
                                new_pos,
                                new_dir,
                                floor_mask,
                                wall_mask,
                            ),
                        )
                        continue
                    # Unknown wildcard: branch
                    bit = wild_map.get(ahead)
                    if bit is None:
                        new_dir = (direction + 1) & 3
                        heapq.heappush(
                            heap,
                            (
                                steps + 1 + heuristic(pos),
                                steps + 1,
                                pos,
                                new_dir,
                                floor_mask,
                                wall_mask,
                            ),
                        )
                        continue
                    mask = 1 << bit
                    move_score = heuristic(ahead)
                    turn_score = heuristic(pos)
                    move_first = move_score <= turn_score
                    if greedy:
                        if move_first:
                            floor_next = floor_mask | mask
                            new_pos = ahead
                            new_dir = (direction + p1) & 3
                            heapq.heappush(
                                heap,
                                (
                                    steps + 1 + move_score,
                                    steps + 1,
                                    new_pos,
                                    new_dir,
                                    floor_next,
                                    wall_mask,
                                ),
                            )
                        else:
                            wall_next = wall_mask | mask
                            new_dir = (direction + 1) & 3
                            heapq.heappush(
                                heap,
                                (
                                    steps + 1 + turn_score,
                                    steps + 1,
                                    pos,
                                    new_dir,
                                    floor_mask,
                                    wall_next,
                                ),
                            )
                        continue
                    # Option 1: open as floor and move forward
                    floor_next = floor_mask | mask
                    new_pos = ahead
                    new_dir = (direction + p1) & 3
                    heapq.heappush(
                        heap,
                        (
                            steps + 1 + move_score,
                            steps + 1,
                            new_pos,
                            new_dir,
                            floor_next,
                            wall_mask,
                        ),
                    )
                    # Option 2: block as wall and turn right
                    wall_next = wall_mask | mask
                    new_dir = (direction + 1) & 3
                    heapq.heappush(
                        heap,
                        (
                            steps + 1 + turn_score,
                            steps + 1,
                            pos,
                            new_dir,
                            floor_mask,
                            wall_next,
                        ),
                    )
    return None


def search_wild_ants(
    puzzle: Puzzle,
    ants: int,
    max_steps: int,
    success_mode: str,
    ant_specs: Optional[List[int]] = None,
    wild_positions: Optional[List[int]] = None,
    fixed_ants: Optional[Dict[int, int]] = None,
    distance_prune: bool = False,
) -> Optional[str]:
    base_state, wilds, food_indices = build_base_state(puzzle)
    if fixed_ants:
        for idx, val in fixed_ants.items():
            if base_state[idx] != CELL_FLOOR:
                raise ValueError("fixed ant overlaps non-floor cell")
            base_state[idx] = val
        wilds = [idx for idx in wilds if idx not in fixed_ants]
    if wild_positions is not None:
        allowed = set(wild_positions)
        wilds = [idx for idx in wilds if idx in allowed]
    neighbors = build_neighbors(puzzle.width, puzzle.height)
    wilds = filter_reachable_positions(base_state, neighbors, wilds)
    if not wilds and ants > 0:
        return None
    dist = None
    fixed_positions: List[int] = []
    if distance_prune:
        targets = compute_food_adjacent_positions(base_state, neighbors)
        dist = compute_distances_to_targets(base_state, neighbors, targets)
        fixed_positions = [idx for idx, val in enumerate(base_state) if is_ant(val)]
    program_count = len(puzzle.programs)
    if program_count == 0:
        raise ValueError("no programs in puzzle")
    assignments = parse_program_assignments(puzzle.programs)
    spec_pool = ant_specs or build_ant_specs(program_count, None)

    for positions in combinations(wilds, ants):
        if distance_prune and dist is not None:
            combined = fixed_positions + list(positions)
            min_dist = compute_min_distance_from_positions(combined, dist)
            if min_dist is None or min_dist > max_steps:
                continue
        for specs in product(spec_pool, repeat=ants):
            state = base_state[:]
            ants_desc = []
            for pos, spec in zip(positions, specs):
                clan = spec // 4
                direction = spec & 3
                state[pos] = make_ant(clan, direction)
                y, x = divmod(pos, puzzle.width)
                ants_desc.append(f"{x},{y},{clan}{DIR_TO_ANT_CHAR[direction]}")
            result = search_partial_programs(
                state,
                neighbors,
                food_indices,
                assignments[:],
                program_count,
                max_steps,
                success_mode,
            )
            if result:
                final_assignments, steps = result
                programs = render_programs(final_assignments, puzzle.programs)
                programs_desc = " ".join(programs)
                ants_joined = " | ".join(ants_desc)
                return (
                    f"ants={ants_joined} programs={programs_desc} steps={steps}"
                )
    return None


def search_wild_ants_sparse(
    puzzle: Puzzle,
    ants: int,
    max_steps: int,
    success_mode: str,
    ant_specs: Optional[List[int]] = None,
    wild_positions: Optional[List[int]] = None,
    fixed_ants: Optional[Dict[int, int]] = None,
    distance_prune: bool = False,
) -> Optional[str]:
    base_state, wilds, _food_indices = build_base_state(puzzle)
    if wild_positions is not None:
        allowed = set(wild_positions)
        wilds = [idx for idx in wilds if idx in allowed]
    neighbors = build_neighbors(puzzle.width, puzzle.height)
    program_count = len(puzzle.programs)
    if program_count == 0:
        raise ValueError("no programs in puzzle")
    assignments = parse_program_assignments(puzzle.programs)
    base_state, fixed_ants_map = split_static_ants(base_state)
    if fixed_ants:
        for idx, val in fixed_ants.items():
            if idx in fixed_ants_map:
                raise ValueError("fixed ant overlaps existing ant")
            if base_state[idx] != CELL_FLOOR:
                raise ValueError("fixed ant overlaps non-floor cell")
            fixed_ants_map[idx] = val
    wilds = [idx for idx in wilds if idx not in fixed_ants_map]
    wilds = filter_reachable_positions(base_state, neighbors, wilds)
    if not wilds and not fixed_ants_map:
        return None
    dist = None
    dist_map = None
    fixed_positions: List[int] = []
    if distance_prune:
        targets = compute_food_adjacent_positions(base_state, neighbors)
        dist = compute_distances_to_targets(base_state, neighbors, targets)
        fixed_positions = list(fixed_ants_map.keys())
        dist_map = build_optimistic_dist_map(base_state, neighbors)
    spec_pool = ant_specs or build_ant_specs(program_count, None)

    for positions in combinations(wilds, ants):
        if distance_prune and dist is not None:
            combined = fixed_positions + list(positions)
            min_dist = compute_min_distance_from_positions(combined, dist)
            if min_dist is None or min_dist > max_steps:
                continue
        for specs in product(spec_pool, repeat=ants):
            ants_state = dict(fixed_ants_map)
            ants_desc = []
            for pos, spec in zip(positions, specs):
                clan = spec // 4
                direction = spec & 3
                ants_state[pos] = make_ant(clan, direction)
                y, x = divmod(pos, puzzle.width)
                ants_desc.append(f"{x},{y},{clan}{DIR_TO_ANT_CHAR[direction]}")
            result = search_partial_programs_sparse(
                ants_state,
                neighbors,
                base_state,
                assignments[:],
                program_count,
                max_steps,
                success_mode,
                dist_map=dist_map,
            )
            if result:
                final_assignments, steps = result
                programs = render_programs(final_assignments, puzzle.programs)
                programs_desc = " ".join(programs)
                ants_joined = " | ".join(ants_desc)
                return (
                    f"ants={ants_joined} programs={programs_desc} steps={steps}"
                )
    return None


def search_wild_ants_grid_sparse(
    puzzle: Puzzle,
    ants: int,
    max_steps: int,
    success_mode: str,
    grid_values: List[int],
    ant_specs: Optional[List[int]] = None,
    grid_nonfloor_limit: Optional[int] = None,
    wild_positions: Optional[List[int]] = None,
    grid_positions: Optional[List[int]] = None,
    fixed_ants: Optional[Dict[int, int]] = None,
    distance_prune: bool = False,
) -> Optional[str]:
    base_state, wilds, _food_indices = build_base_state_with_wildcards(puzzle)
    neighbors = build_neighbors(puzzle.width, puzzle.height)
    program_count = len(puzzle.programs)
    if program_count == 0:
        raise ValueError("no programs in puzzle")
    assignments = parse_program_assignments(puzzle.programs)
    base_state, fixed_ants_map = split_static_ants(base_state)
    if fixed_ants:
        for idx, val in fixed_ants.items():
            if idx in fixed_ants_map:
                raise ValueError("fixed ant overlaps existing ant")
            if base_state[idx] in (CELL_WALL, CELL_HOLE, CELL_FOOD):
                raise ValueError("fixed ant overlaps non-floor cell")
            fixed_ants_map[idx] = val
            base_state[idx] = CELL_FLOOR
    wilds = [idx for idx in wilds if idx not in fixed_ants_map]
    wilds = filter_reachable_positions(base_state, neighbors, wilds)
    if not wilds and not fixed_ants_map:
        return None
    dist = None
    dist_map = None
    fixed_positions: List[int] = []
    if distance_prune:
        targets = compute_food_adjacent_positions(base_state, neighbors)
        dist = compute_distances_to_targets(base_state, neighbors, targets)
        fixed_positions = list(fixed_ants_map.keys())
        dist_map = build_optimistic_dist_map(base_state, neighbors)
    spec_pool = ant_specs or build_ant_specs(program_count, None)

    if ants == 0 and not fixed_ants_map:
        return None

    candidate_positions = wilds
    if wild_positions is not None:
        allowed = set(wild_positions)
        candidate_positions = [idx for idx in wilds if idx in allowed]
    positions_iter = combinations(candidate_positions, ants) if ants > 0 else [()]
    for positions in positions_iter:
        if distance_prune and dist is not None:
            combined = fixed_positions + list(positions)
            min_dist = compute_min_distance_from_positions(combined, dist)
            if min_dist is None or min_dist > max_steps:
                continue
        positions_set = set(positions)
        base_state_cur = base_state[:] if positions_set or grid_positions is not None else base_state
        for pos in positions_set:
            base_state_cur[pos] = CELL_FLOOR
        grid_positions_list = [idx for idx in wilds if idx not in positions_set]
        if grid_positions is not None:
            allowed_grid = set(grid_positions)
            for idx in wilds:
                if idx in positions_set or idx in allowed_grid:
                    continue
                base_state_cur[idx] = CELL_FLOOR
            grid_positions_list = [
                idx for idx in grid_positions_list if idx in allowed_grid
            ]
        grid_entry_by_idx = {
            idx: i for i, idx in enumerate(grid_positions_list)
        }

        for specs in product(spec_pool, repeat=ants):
            ants_state = dict(fixed_ants_map)
            ants_desc = []
            for pos, spec in zip(positions, specs):
                clan = spec // 4
                direction = spec & 3
                ants_state[pos] = make_ant(clan, direction)
                y, x = divmod(pos, puzzle.width)
                ants_desc.append(f"{x},{y},{clan}{DIR_TO_ANT_CHAR[direction]}")
            grid_assignments = [-1] * len(grid_positions_list)
            result = search_partial_programs_sparse_with_grid(
                ants_state,
                neighbors,
                base_state_cur,
                assignments[:],
                grid_assignments,
                grid_entry_by_idx,
                grid_values,
                grid_nonfloor_limit,
                program_count,
                max_steps,
                success_mode,
                dist_map=dist_map,
            )
            if result:
                final_assignments, final_grid_assignments, steps = result
                programs = render_programs(final_assignments, puzzle.programs)
                programs_desc = " ".join(programs)
                ants_joined = " | ".join(ants_desc) if ants_desc else "-"
                grid_desc = render_grid_assignments(
                    final_grid_assignments, grid_positions_list, puzzle.width
                )
                return (
                    f"ants={ants_joined} programs={programs_desc} "
                    f"grid={grid_desc} steps={steps}"
                )
    return None


def search_wild_ants_grid_enum(
    puzzle: Puzzle,
    ants: int,
    max_steps: int,
    success_mode: str,
    grid_values: List[int],
    ant_specs: Optional[List[int]] = None,
    grid_nonfloor_limit: Optional[int] = None,
    wild_positions: Optional[List[int]] = None,
    grid_positions: Optional[List[int]] = None,
    fixed_ants: Optional[Dict[int, int]] = None,
    distance_prune: bool = False,
) -> Optional[str]:
    base_state, wilds, _food_indices = build_base_state(puzzle)
    neighbors = build_neighbors(puzzle.width, puzzle.height)
    program_count = len(puzzle.programs)
    if program_count == 0:
        raise ValueError("no programs in puzzle")
    assignments = parse_program_assignments(puzzle.programs)
    base_state, fixed_ants_map = split_static_ants(base_state)
    if fixed_ants:
        for idx, val in fixed_ants.items():
            if idx in fixed_ants_map:
                raise ValueError("fixed ant overlaps existing ant")
            if base_state[idx] in (CELL_WALL, CELL_HOLE, CELL_FOOD):
                raise ValueError("fixed ant overlaps non-floor cell")
            fixed_ants_map[idx] = val
    wilds = [idx for idx in wilds if idx not in fixed_ants_map]
    if wild_positions is not None:
        allowed = set(wild_positions)
        wilds = [idx for idx in wilds if idx in allowed]
    wilds = filter_reachable_positions(base_state, neighbors, wilds)
    if not wilds and ants > 0:
        return None

    allow_floor = CELL_FLOOR in grid_values
    allowed_nonfloor = [val for val in grid_values if val != CELL_FLOOR]
    allowed_grid = set(grid_positions) if grid_positions is not None else None
    if not allow_floor and allowed_grid is not None:
        if any(idx not in allowed_grid for idx in wilds):
            return None

    dist = None
    dist_map = None
    fixed_positions: List[int] = []
    if distance_prune:
        targets = compute_food_adjacent_positions(base_state, neighbors)
        dist = compute_distances_to_targets(base_state, neighbors, targets)
        fixed_positions = list(fixed_ants_map.keys())
        dist_map = build_optimistic_dist_map(base_state, neighbors)

    spec_pool = ant_specs or build_ant_specs(program_count, None)
    if ants == 0 and not fixed_ants_map:
        return None

    for positions in combinations(wilds, ants):
        if distance_prune and dist is not None:
            combined = fixed_positions + list(positions)
            min_dist = compute_min_distance_from_positions(combined, dist)
            if min_dist is None or min_dist > max_steps:
                continue
        positions_set = set(positions)
        grid_positions_list = [idx for idx in wilds if idx not in positions_set]
        grid_entry_by_idx = {
            idx: i for i, idx in enumerate(grid_positions_list)
        }
        grid_candidates = grid_positions_list
        if allowed_grid is not None:
            grid_candidates = [idx for idx in grid_positions_list if idx in allowed_grid]
            if not allow_floor and len(grid_candidates) < len(grid_positions_list):
                continue

        def iter_grid_assignments() -> Iterable[Dict[int, int]]:
            if not allow_floor:
                if grid_nonfloor_limit is not None and grid_nonfloor_limit < len(grid_candidates):
                    return
                if not allowed_nonfloor:
                    return
                for values in product(allowed_nonfloor, repeat=len(grid_candidates)):
                    yield dict(zip(grid_candidates, values))
                return
            max_k = grid_nonfloor_limit if grid_nonfloor_limit is not None else len(grid_candidates)
            if not allowed_nonfloor:
                yield {}
                return
            yield {}
            for k in range(1, max_k + 1):
                for idxs in combinations(grid_candidates, k):
                    for values in product(allowed_nonfloor, repeat=k):
                        yield dict(zip(idxs, values))

        for grid_assign in iter_grid_assignments():
            base_state_cur = base_state[:]
            grid_assignments = [CELL_FLOOR] * len(grid_positions_list)
            for idx, val in grid_assign.items():
                base_state_cur[idx] = val
                grid_assignments[grid_entry_by_idx[idx]] = val
            for specs in product(spec_pool, repeat=ants):
                ants_state = dict(fixed_ants_map)
                ants_desc = []
                for pos, spec in zip(positions, specs):
                    clan = spec // 4
                    direction = spec & 3
                    ants_state[pos] = make_ant(clan, direction)
                    y, x = divmod(pos, puzzle.width)
                    ants_desc.append(f"{x},{y},{clan}{DIR_TO_ANT_CHAR[direction]}")
                result = search_partial_programs_sparse(
                    ants_state,
                    neighbors,
                    base_state_cur,
                    assignments[:],
                    program_count,
                    max_steps,
                    success_mode,
                    dist_map=dist_map,
                )
                if result:
                    final_assignments, steps = result
                    programs = render_programs(final_assignments, puzzle.programs)
                    programs_desc = " ".join(programs)
                    ants_joined = " | ".join(ants_desc) if ants_desc else "-"
                    grid_desc = render_grid_assignments(
                        grid_assignments, grid_positions_list, puzzle.width
                    )
                    return (
                        f"ants={ants_joined} programs={programs_desc} "
                        f"grid={grid_desc} steps={steps}"
                    )
    return None


def search_single_ant_p1(
    puzzle: Puzzle,
    max_steps: int,
    success_mode: str,
) -> Optional[str]:
    base_state, wilds, food_indices = build_base_state(puzzle)
    neighbors = build_neighbors(puzzle.width, puzzle.height)
    for wild_idx in wilds:
        for p1_dir in range(4):
            programs = [[p1_dir] + [0] * 6]
            for start_dir in range(4):
                state = base_state[:]
                state[wild_idx] = make_ant(0, start_dir)
                ok, steps = simulate_fast(
                    state,
                    neighbors,
                    programs,
                    food_indices,
                    max_steps=max_steps,
                    success_mode=success_mode,
                )
                if ok:
                    y, x = divmod(wild_idx, puzzle.width)
                    return (
                        f"found single ant at ({x},{y}) p1={DIRS[p1_dir]} "
                        f"start={DIRS[start_dir]} steps={steps}"
                    )
    return None


def search_two_ants_p1(
    puzzle: Puzzle,
    max_steps: int,
    success_mode: str,
) -> Optional[str]:
    base_state, wilds, food_indices = build_base_state(puzzle)
    neighbors = build_neighbors(puzzle.width, puzzle.height)
    for p1_dir in range(4):
        programs = [[p1_dir] + [0] * 6]
        for a_idx, b_idx in combinations(wilds, 2):
            for a_dir, b_dir in product(range(4), repeat=2):
                state = base_state[:]
                state[a_idx] = make_ant(0, a_dir)
                state[b_idx] = make_ant(0, b_dir)
                ok, steps = simulate_fast(
                    state,
                    neighbors,
                    programs,
                    food_indices,
                    max_steps=max_steps,
                    success_mode=success_mode,
                )
                if ok:
                    ay, ax = divmod(a_idx, puzzle.width)
                    by, bx = divmod(b_idx, puzzle.width)
                    return (
                        "found two ants "
                        f"a=({ax},{ay}) {DIRS[a_dir]} "
                        f"b=({bx},{by}) {DIRS[b_dir]} "
                        f"p1={DIRS[p1_dir]} steps={steps}"
                    )
    return None


def search_single_ant_fast(
    puzzle: Puzzle,
    max_steps: int,
    success_mode: str,
) -> Optional[str]:
    base_state, wilds, _food_indices = build_base_state(puzzle)
    neighbors = build_neighbors(puzzle.width, puzzle.height)

    def has_success(pos: int, direction: int) -> bool:
        ahead = neighbors[pos][direction]
        if ahead == -1:
            return False
        if base_state[ahead] != CELL_FOOD:
            return False
        if success_mode == "facing":
            return True
        if success_mode == "below":
            return direction == 0
        raise ValueError(f"unknown success mode: {success_mode!r}")

    for wild_idx in wilds:
        y, x = divmod(wild_idx, puzzle.width)
        for p1_dir in range(4):
            for start_dir in range(4):
                pos = wild_idx
                direction = start_dir
                seen = set()
                for step in range(max_steps + 1):
                    if has_success(pos, direction):
                        return (
                            f"ant=({x},{y}) start={DIRS[start_dir]} "
                            f"p1={DIRS[p1_dir]} steps={step}"
                        )
                    state_key = (pos, direction)
                    if state_key in seen:
                        break
                    seen.add(state_key)
                    ahead = neighbors[pos][direction]
                    if ahead == -1 or base_state[ahead] == CELL_WALL:
                        direction = (direction + 1) & 3
                        continue
                    if base_state[ahead] == CELL_HOLE:
                        break
                    if base_state[ahead] == CELL_FOOD:
                        continue
                    pos = ahead
                    direction = (direction + p1_dir) & 3
    return None


def search_single_ant_grid_fast(
    puzzle: Puzzle,
    max_steps: int,
    success_mode: str,
    grid_values: List[int],
    grid_nonfloor_limit: Optional[int],
    start_positions: Optional[List[int]] = None,
    allowed_grid_positions: Optional[List[int]] = None,
) -> Optional[str]:
    base_state, wilds, _food_indices = build_base_state_with_wildcards(puzzle)
    if not wilds:
        return None
    neighbors = build_neighbors(puzzle.width, puzzle.height)
    if allowed_grid_positions is not None:
        allowed = set(allowed_grid_positions)
        for idx in wilds:
            if idx not in allowed:
                base_state[idx] = CELL_FLOOR
        grid_wilds = [idx for idx in wilds if idx in allowed]
    else:
        grid_wilds = wilds
    wild_map = {idx: bit for bit, idx in enumerate(grid_wilds)}
    allowed = set(grid_values)
    if start_positions is None:
        start_positions = wilds
    else:
        start_positions = [idx for idx in start_positions if idx in wilds]
        if not start_positions:
            return None

    def is_success(pos: int, direction: int) -> bool:
        ahead = neighbors[pos][direction]
        if ahead == -1:
            return False
        if base_state[ahead] != CELL_FOOD:
            return False
        if success_mode == "facing":
            return True
        if success_mode == "below":
            return direction == 0
        raise ValueError(f"unknown success mode: {success_mode!r}")

    def cell_kind(
        idx: int, wall_mask: int, hole_mask: int, floor_mask: int
    ) -> Tuple[int, Optional[int]]:
        if idx == -1:
            return CELL_WALL, None
        val = base_state[idx]
        if val != CELL_WILD:
            return val, None
        bit = 1 << wild_map[idx]
        if wall_mask & bit:
            return CELL_WALL, bit
        if hole_mask & bit:
            return CELL_HOLE, bit
        if floor_mask & bit:
            return CELL_FLOOR, bit
        return CELL_WILD, bit

    for start_idx in start_positions:
        start_bit = 0
        bit = wild_map.get(start_idx)
        if bit is not None:
            start_bit = 1 << bit
        for p1_dir in range(4):
            for start_dir in range(4):
                seen = set()
                stack = [(start_idx, start_dir, 0, 0, start_bit, 0)]
                while stack:
                    pos, direction, wall_mask, hole_mask, floor_mask, steps = (
                        stack.pop()
                    )
                    if steps > max_steps:
                        continue
                    key = (pos, direction, wall_mask, hole_mask, floor_mask)
                    if key in seen:
                        continue
                    seen.add(key)
                    if is_success(pos, direction):
                        grid_assignments = [CELL_FLOOR] * len(grid_wilds)
                        for bit, idx in enumerate(grid_wilds):
                            mask = 1 << bit
                            if wall_mask & mask:
                                grid_assignments[bit] = CELL_WALL
                            elif hole_mask & mask:
                                grid_assignments[bit] = CELL_HOLE
                        grid_desc = render_grid_assignments(
                            grid_assignments, grid_wilds, puzzle.width
                        )
                        y, x = divmod(start_idx, puzzle.width)
                        return (
                            f"ant=({x},{y}) start={DIRS[start_dir]} "
                            f"p1={DIRS[p1_dir]} grid={grid_desc} steps={steps}"
                        )
                    ahead = neighbors[pos][direction]
                    kind, bit = cell_kind(ahead, wall_mask, hole_mask, floor_mask)
                    if kind == CELL_WALL:
                        stack.append(
                            (
                                pos,
                                (direction + 1) & 3,
                                wall_mask,
                                hole_mask,
                                floor_mask,
                                steps + 1,
                            )
                        )
                        continue
                    if kind == CELL_HOLE:
                        continue
                    if kind == CELL_FLOOR:
                        next_pos = ahead
                        next_dir = (direction + p1_dir) & 3
                        next_floor = floor_mask
                        if bit is not None:
                            next_floor |= bit
                        stack.append(
                            (next_pos, next_dir, wall_mask, hole_mask, next_floor, steps + 1)
                        )
                        continue
                    if kind == CELL_WILD:
                        nonfloor = (wall_mask | hole_mask).bit_count()
                        if CELL_FLOOR in allowed:
                            next_pos = ahead
                            next_dir = (direction + p1_dir) & 3
                            next_floor = floor_mask | bit
                            stack.append(
                                (
                                    next_pos,
                                    next_dir,
                                    wall_mask,
                                    hole_mask,
                                    next_floor,
                                    steps + 1,
                                )
                            )
                        if (
                            CELL_WALL in allowed
                            and bit is not None
                            and (
                                grid_nonfloor_limit is None
                                or nonfloor + 1 <= grid_nonfloor_limit
                            )
                        ):
                            stack.append(
                                (
                                    pos,
                                    (direction + 1) & 3,
                                    wall_mask | bit,
                                    hole_mask,
                                    floor_mask,
                                    steps + 1,
                                )
                            )
                        if (
                            CELL_HOLE in allowed
                            and bit is not None
                            and (
                                grid_nonfloor_limit is None
                                or nonfloor + 1 <= grid_nonfloor_limit
                            )
                        ):
                            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Antomaton puzzle helper: parse, simulate, and summarize."
    )
    parser.add_argument(
        "--puzzles",
        default="volume9_gardener_puzzles.txt",
        help="Path to the puzzle dump text.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print puzzle summaries instead of solving.",
    )
    parser.add_argument(
        "--match",
        help="Only show puzzles whose title contains this substring.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Max steps for searches.",
    )
    parser.add_argument(
        "--single-ant-p1",
        action="store_true",
        help="Search wildcard placements with one ant; only p1 varies.",
    )
    parser.add_argument(
        "--two-ants-p1",
        action="store_true",
        help="Search wildcard placements with two ants; only p1 varies.",
    )
    parser.add_argument(
        "--single-ant-fast",
        action="store_true",
        help="Fast search for single-ant placements (only p1 matters).",
    )
    parser.add_argument(
        "--single-ant-grid-fast",
        action="store_true",
        help="Fast search for single-ant placements with grid wildcards.",
    )
    parser.add_argument(
        "--carve-single-ant",
        action="store_true",
        help="Carve a single-ant maze by treating wilds as walls by default.",
    )
    parser.add_argument(
        "--carve-p1",
        help="Fix p1 direction for --carve-single-ant (N/E/S/W or 0..3).",
    )
    parser.add_argument(
        "--carve-greedy",
        action="store_true",
        help="Use greedy branching for --carve-single-ant.",
    )
    parser.add_argument(
        "--carve-write",
        help="Write the carved single-ant solution to this .ant file (implies --carve-single-ant).",
    )
    parser.add_argument(
        "--wild-ants",
        action="store_true",
        help="Search wildcard placements with N ants and lazy programs.",
    )
    parser.add_argument(
        "--wild-ants-sparse",
        action="store_true",
        help="Search wildcard placements with N ants using sparse simulation.",
    )
    parser.add_argument(
        "--grid-wilds-sparse",
        action="store_true",
        help="Search wildcard grid contents (floor/wall/hole) using sparse simulation.",
    )
    parser.add_argument(
        "--grid-enum-sparse",
        action="store_true",
        help="Enumerate small grid wildcard assignments and search with sparse simulation.",
    )
    parser.add_argument(
        "--ants",
        type=int,
        help="Number of ants to place for --wild-ants and --grid-wilds-sparse.",
    )
    parser.add_argument(
        "--grid-wild-values",
        default="floor,wall,hole",
        help=(
            "Comma-separated grid wildcard values for --grid-wilds-sparse "
            "(e.g. floor,wall,hole,food,1^,3v)."
        ),
    )
    parser.add_argument(
        "--grid-wild-max-nonfloor",
        type=int,
        help="Max number of non-floor grid wildcard assignments.",
    )
    parser.add_argument(
        "--grid-positions",
        help="Semicolon-separated x,y list to restrict grid wildcard assignments.",
    )
    parser.add_argument(
        "--grid-rect",
        help="Rect x0,y0,x1,y1 to restrict grid wildcard assignments.",
    )
    parser.add_argument(
        "--grid-max-dist",
        type=int,
        help="Max BFS distance from food-adjacent cells for grid wildcard assignments.",
    )
    parser.add_argument(
        "--grid-limit",
        type=int,
        help="Limit grid wildcard assignments to the closest N positions.",
    )
    parser.add_argument(
        "--ant-clans",
        help="Comma-separated clan ids to use when placing ants.",
    )
    parser.add_argument(
        "--ant-positions",
        help="Semicolon-separated x,y list to restrict wildcard ant placement.",
    )
    parser.add_argument(
        "--fixed-ants",
        help="Semicolon-separated x,y,clan,dir list to add fixed ants.",
    )
    parser.add_argument(
        "--ant-rect",
        help="Rect x0,y0,x1,y1 to restrict wildcard ant placement.",
    )
    parser.add_argument(
        "--ant-max-dist",
        type=int,
        help="Max BFS distance from food-adjacent cells for wildcard ant placement.",
    )
    parser.add_argument(
        "--ant-limit",
        type=int,
        help="Limit wildcard ant placement to the closest N positions.",
    )
    parser.add_argument(
        "--small-scan",
        action="store_true",
        help="Scan small-step solutions with varying ants and grid wildcards.",
    )
    parser.add_argument(
        "--scan-ants-max",
        type=int,
        default=3,
        help="Max ants to place for --small-scan.",
    )
    parser.add_argument(
        "--scan-nonfloor-max",
        type=int,
        default=2,
        help="Max non-floor grid assignments for --small-scan.",
    )
    parser.add_argument(
        "--distance-prune",
        action="store_true",
        help="Prune if min ant distance to food exceeds max steps.",
    )
    success_group = parser.add_mutually_exclusive_group()
    success_group.add_argument(
        "--success-facing",
        action="store_true",
        help="Treat success as any ant facing a food cell (default).",
    )
    success_group.add_argument(
        "--success-below",
        action="store_true",
        help="Treat success as an ant facing north below a food cell.",
    )
    args = parser.parse_args()
    success_mode = "below" if args.success_below else "facing"

    puzzles = parse_puzzles(Path(args.puzzles))
    if args.stats:
        summarize_puzzles(puzzles, args.match)
        return
    if args.small_scan:
        if not args.match:
            raise SystemExit("--match is required for --small-scan")
        grid_values = parse_grid_wild_values(args.grid_wild_values)
        for puzzle in puzzles:
            if args.match not in puzzle.title:
                continue
            allowed_clans = parse_ant_clans(args.ant_clans, len(puzzle.programs))
            ant_specs = build_ant_specs(len(puzzle.programs), allowed_clans)
            fixed_ants = parse_fixed_ants(
                args.fixed_ants, puzzle.width, puzzle.height
            )
            wild_positions = build_wild_position_filter(
                puzzle,
                args.ant_positions,
                args.ant_rect,
                args.ant_max_dist,
                args.ant_limit,
            )
            grid_positions = build_wild_position_filter(
                puzzle,
                args.grid_positions,
                args.grid_rect,
                args.grid_max_dist,
                args.grid_limit,
            )
            found = False
            for nonfloor in range(args.scan_nonfloor_max + 1):
                for ants in range(args.scan_ants_max + 1):
                    result = search_wild_ants_grid_sparse(
                        puzzle,
                        ants=ants,
                        max_steps=args.max_steps,
                        success_mode=success_mode,
                        grid_values=grid_values,
                        ant_specs=ant_specs,
                        grid_nonfloor_limit=nonfloor,
                        wild_positions=wild_positions,
                        grid_positions=grid_positions,
                        fixed_ants=fixed_ants,
                        distance_prune=args.distance_prune,
                    )
                    if result:
                        print(f"ants={ants} nonfloor={nonfloor} {result}")
                        found = True
                        break
                if found:
                    break
            if not found:
                print("no solution")
        return
    if args.single_ant_p1 or args.two_ants_p1:
        if not args.match:
            raise SystemExit("--match is required for ant placement searches")
        for puzzle in puzzles:
            if args.match not in puzzle.title:
                continue
            if args.single_ant_p1:
                result = search_single_ant_p1(
                    puzzle,
                    max_steps=args.max_steps,
                    success_mode=success_mode,
                )
                print(result or "no single-ant solution")
            if args.two_ants_p1:
                result = search_two_ants_p1(
                    puzzle,
                    max_steps=args.max_steps,
                    success_mode=success_mode,
                )
                print(result or "no two-ant solution")
        return
    if args.single_ant_fast:
        if not args.match:
            raise SystemExit("--match is required for --single-ant-fast")
        for puzzle in puzzles:
            if args.match not in puzzle.title:
                continue
            result = search_single_ant_fast(
                puzzle,
                max_steps=args.max_steps,
                success_mode=success_mode,
            )
            print(result or "no single-ant solution")
        return
    if args.single_ant_grid_fast:
        if not args.match:
            raise SystemExit("--match is required for --single-ant-grid-fast")
        success_mode = "facing" if args.success_facing else "below"
        grid_values = parse_grid_wild_values(args.grid_wild_values)
        for puzzle in puzzles:
            if args.match not in puzzle.title:
                continue
            start_positions = build_wild_position_filter(
                puzzle,
                args.ant_positions,
                args.ant_rect,
                args.ant_max_dist,
                args.ant_limit,
            )
            allowed_grid_positions = build_wild_position_filter(
                puzzle,
                args.grid_positions,
                args.grid_rect,
                args.grid_max_dist,
                args.grid_limit,
            )
            result = search_single_ant_grid_fast(
                puzzle,
                max_steps=args.max_steps,
                success_mode=success_mode,
                grid_values=grid_values,
                grid_nonfloor_limit=args.grid_wild_max_nonfloor,
                start_positions=start_positions,
                allowed_grid_positions=allowed_grid_positions,
            )
            print(result or "no single-ant grid solution")
        return
    if args.carve_write:
        args.carve_single_ant = True
    if args.carve_single_ant:
        if not args.match:
            raise SystemExit("--match is required for --carve-single-ant/--carve-write")
        p1_dir = parse_dir_value(args.carve_p1)
        for puzzle in puzzles:
            if args.match not in puzzle.title:
                continue
            start_positions = build_wild_position_filter(
                puzzle,
                args.ant_positions,
                args.ant_rect,
                args.ant_max_dist,
                args.ant_limit,
            )
            allowed_grid_positions = build_wild_position_filter(
                puzzle,
                args.grid_positions,
                args.grid_rect,
                args.grid_max_dist,
                args.grid_limit,
            )
            solution = find_carve_single_ant_solution(
                puzzle,
                max_steps=args.max_steps,
                success_mode=success_mode,
                p1_dir=p1_dir,
                start_positions=start_positions,
                allowed_grid_positions=allowed_grid_positions,
                greedy=args.carve_greedy,
            )
            if not solution:
                print("no carved single-ant solution")
                continue
            if args.carve_write:
                programs = [
                    fill_program_template(
                        template,
                        p1=solution.p1,
                        apply_p1=(clan == 0),
                    )
                    for clan, template in enumerate(puzzle.programs)
                ]
                replacements = carve_solution_replacements(puzzle, solution, clan=0)
                world = build_world(puzzle, replacements)
                write_ant_world(Path(args.carve_write), puzzle.title, programs, world)
                print(f"wrote {args.carve_write}")
            else:
                result = search_carve_single_ant(
                    puzzle,
                    max_steps=args.max_steps,
                    success_mode=success_mode,
                    p1_dir=p1_dir,
                    start_positions=start_positions,
                    allowed_grid_positions=allowed_grid_positions,
                    greedy=args.carve_greedy,
                )
                print(result or "no carved single-ant solution")
        return
    if args.wild_ants:
        if not args.match:
            raise SystemExit("--match is required for --wild-ants")
        if args.ants is None:
            raise SystemExit("--ants is required for --wild-ants")
        for puzzle in puzzles:
            if args.match not in puzzle.title:
                continue
            allowed_clans = parse_ant_clans(args.ant_clans, len(puzzle.programs))
            ant_specs = build_ant_specs(len(puzzle.programs), allowed_clans)
            fixed_ants = parse_fixed_ants(
                args.fixed_ants, puzzle.width, puzzle.height
            )
            wild_positions = build_wild_position_filter(
                puzzle,
                args.ant_positions,
                args.ant_rect,
                args.ant_max_dist,
                args.ant_limit,
            )
            result = search_wild_ants(
                puzzle,
                ants=args.ants,
                max_steps=args.max_steps,
                success_mode=success_mode,
                ant_specs=ant_specs,
                wild_positions=wild_positions,
                fixed_ants=fixed_ants,
                distance_prune=args.distance_prune,
            )
            print(result or "no solution")
        return
    if args.wild_ants_sparse:
        if not args.match:
            raise SystemExit("--match is required for --wild-ants-sparse")
        if args.ants is None:
            raise SystemExit("--ants is required for --wild-ants-sparse")
        for puzzle in puzzles:
            if args.match not in puzzle.title:
                continue
            allowed_clans = parse_ant_clans(args.ant_clans, len(puzzle.programs))
            ant_specs = build_ant_specs(len(puzzle.programs), allowed_clans)
            fixed_ants = parse_fixed_ants(
                args.fixed_ants, puzzle.width, puzzle.height
            )
            wild_positions = build_wild_position_filter(
                puzzle,
                args.ant_positions,
                args.ant_rect,
                args.ant_max_dist,
                args.ant_limit,
            )
            result = search_wild_ants_sparse(
                puzzle,
                ants=args.ants,
                max_steps=args.max_steps,
                success_mode=success_mode,
                ant_specs=ant_specs,
                wild_positions=wild_positions,
                fixed_ants=fixed_ants,
                distance_prune=args.distance_prune,
            )
            print(result or "no solution")
        return
    if args.grid_wilds_sparse:
        if not args.match:
            raise SystemExit("--match is required for --grid-wilds-sparse")
        ants = args.ants or 0
        grid_values = parse_grid_wild_values(args.grid_wild_values)
        for puzzle in puzzles:
            if args.match not in puzzle.title:
                continue
            allowed_clans = parse_ant_clans(args.ant_clans, len(puzzle.programs))
            ant_specs = build_ant_specs(len(puzzle.programs), allowed_clans)
            fixed_ants = parse_fixed_ants(
                args.fixed_ants, puzzle.width, puzzle.height
            )
            wild_positions = build_wild_position_filter(
                puzzle,
                args.ant_positions,
                args.ant_rect,
                args.ant_max_dist,
                args.ant_limit,
            )
            grid_positions = build_wild_position_filter(
                puzzle,
                args.grid_positions,
                args.grid_rect,
                args.grid_max_dist,
                args.grid_limit,
            )
            result = search_wild_ants_grid_sparse(
                puzzle,
                ants=ants,
                max_steps=args.max_steps,
                success_mode=success_mode,
                grid_values=grid_values,
                ant_specs=ant_specs,
                grid_nonfloor_limit=args.grid_wild_max_nonfloor,
                wild_positions=wild_positions,
                grid_positions=grid_positions,
                fixed_ants=fixed_ants,
                distance_prune=args.distance_prune,
            )
            print(result or "no solution")
        return
    if args.grid_enum_sparse:
        if not args.match:
            raise SystemExit("--match is required for --grid-enum-sparse")
        if args.grid_wild_max_nonfloor is None:
            raise SystemExit("--grid-wild-max-nonfloor is required for --grid-enum-sparse")
        ants = args.ants or 0
        grid_values = parse_grid_wild_values(args.grid_wild_values)
        for puzzle in puzzles:
            if args.match not in puzzle.title:
                continue
            allowed_clans = parse_ant_clans(args.ant_clans, len(puzzle.programs))
            ant_specs = build_ant_specs(len(puzzle.programs), allowed_clans)
            fixed_ants = parse_fixed_ants(
                args.fixed_ants, puzzle.width, puzzle.height
            )
            wild_positions = build_wild_position_filter(
                puzzle,
                args.ant_positions,
                args.ant_rect,
                args.ant_max_dist,
                args.ant_limit,
            )
            grid_positions = build_wild_position_filter(
                puzzle,
                args.grid_positions,
                args.grid_rect,
                args.grid_max_dist,
                args.grid_limit,
            )
            result = search_wild_ants_grid_enum(
                puzzle,
                ants=ants,
                max_steps=args.max_steps,
                success_mode=success_mode,
                grid_values=grid_values,
                ant_specs=ant_specs,
                grid_nonfloor_limit=args.grid_wild_max_nonfloor,
                wild_positions=wild_positions,
                grid_positions=grid_positions,
                fixed_ants=fixed_ants,
                distance_prune=args.distance_prune,
            )
            print(result or "no solution")
        return

    for puzzle in puzzles:
        if puzzle.title.startswith("Puzzle 1"):
            solve_puzzle1(puzzle, max_steps=args.max_steps)
            return
    raise SystemExit("Puzzle 1 not found")


if __name__ == "__main__":
    main()
