#!/usr/bin/env python3
"""Black Knots solver: find row patterns for a given model spec."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import heapq
import argparse
import random
import re
import sys
import time


Spec = Dict[int, Tuple[int, int]]  # input -> (output, plinks)


@dataclass(frozen=True)
class ModelSpec:
    model: str
    width: int
    outputs: Tuple[int, ...]
    plinks: Tuple[int, ...]


SPEC_LINE_RE = re.compile(r"^\s*(\d+)\s*->\s*\((\d+),\s*(\d+)\)\s*$")
PATH_SPLIT_RE = re.compile(r"[,\s]+")


def parse_specs(log_path: Path, model_order: List[str]) -> Dict[str, Spec]:
    blocks: List[Spec] = []
    current: Spec = {}
    lines = log_path.read_text().splitlines()
    for line in lines:
        if "Which model would you like to see the spec for?" in line:
            if current:
                blocks.append(current)
                current = {}
            continue
        match = SPEC_LINE_RE.match(line)
        if match:
            x, y, z = map(int, match.groups())
            current[x] = (y, z)
    if current:
        blocks.append(current)
    if len(blocks) < len(model_order):
        raise ValueError(
            f"spec blocks ({len(blocks)}) < model_order ({len(model_order)})"
        )
    specs: Dict[str, Spec] = {}
    for model, spec in zip(model_order, blocks):
        specs[model] = spec
    return specs


def build_model_spec(model: str, spec: Spec) -> ModelSpec:
    width = max(spec.keys()) + 1
    outputs = [0] * width
    plinks = [0] * width
    for x, (y, z) in spec.items():
        outputs[x] = y
        plinks[x] = z
    return ModelSpec(model=model, width=width, outputs=tuple(outputs), plinks=tuple(plinks))


def parse_path_positions(path_text: str) -> List[int]:
    parts = [p for p in PATH_SPLIT_RE.split(path_text.strip()) if p]
    if not parts:
        raise ValueError("force path is empty")
    return [int(p) for p in parts]


def enumerate_token_paths(
    start_pos: int,
    right_moves: int,
    left_moves: int,
    rows: int,
    width: int,
) -> Iterable[List[int]]:
    stay_moves = rows - right_moves - left_moves
    if stay_moves < 0:
        return
    path = [start_pos]

    def dfs(step: int, pos: int, r: int, l: int, s: int) -> Iterable[List[int]]:
        if step == rows:
            if r == 0 and l == 0 and s == 0:
                yield path[:]
            return
        if r > 0 and pos + 1 < width:
            path.append(pos + 1)
            yield from dfs(step + 1, pos + 1, r - 1, l, s)
            path.pop()
        if l > 0 and pos - 1 >= 0:
            path.append(pos - 1)
            yield from dfs(step + 1, pos - 1, r, l - 1, s)
            path.pop()
        if s > 0:
            path.append(pos)
            yield from dfs(step + 1, pos, r, l, s - 1)
            path.pop()

    yield from dfs(0, start_pos, right_moves, left_moves, stay_moves)


def enumerate_token_prefixes(
    start_pos: int,
    right_moves: int,
    left_moves: int,
    rows: int,
    prefix_rows: int,
    width: int,
) -> Iterable[List[int]]:
    if prefix_rows > rows:
        return
    path = [start_pos]

    def dfs(step: int, pos: int, r_used: int, l_used: int) -> Iterable[List[int]]:
        if step == prefix_rows:
            remaining = rows - prefix_rows
            r_left = right_moves - r_used
            l_left = left_moves - l_used
            if r_left < 0 or l_left < 0:
                return
            if r_left + l_left > remaining:
                return
            yield path[:]
            return
        if r_used < right_moves and pos + 1 < width:
            path.append(pos + 1)
            yield from dfs(step + 1, pos + 1, r_used + 1, l_used)
            path.pop()
        if l_used < left_moves and pos - 1 >= 0:
            path.append(pos - 1)
            yield from dfs(step + 1, pos - 1, r_used, l_used + 1)
            path.pop()
        path.append(pos)
        yield from dfs(step + 1, pos, r_used, l_used)
        path.pop()

    yield from dfs(0, start_pos, 0, 0)


def enumerate_paths_with_target(
    start_pos: int,
    target_pos: int,
    right_moves: int,
    left_moves: int,
    rows: int,
    width: int,
) -> Iterable[List[int]]:
    stay_moves = rows - right_moves - left_moves
    if stay_moves < 0:
        return
    if start_pos + right_moves - left_moves != target_pos:
        return
    path = [start_pos]

    def dfs(step: int, pos: int, r: int, l: int, s: int) -> Iterable[List[int]]:
        if pos + r - l != target_pos:
            return
        if step == rows:
            if r == 0 and l == 0 and s == 0 and pos == target_pos:
                yield path[:]
            return
        if r > 0 and pos + 1 < width:
            path.append(pos + 1)
            yield from dfs(step + 1, pos + 1, r - 1, l, s)
            path.pop()
        if l > 0 and pos - 1 >= 0:
            path.append(pos - 1)
            yield from dfs(step + 1, pos - 1, r, l - 1, s)
            path.pop()
        if s > 0:
            path.append(pos)
            yield from dfs(step + 1, pos, r, l, s - 1)
            path.pop()

    yield from dfs(0, start_pos, right_moves, left_moves, stay_moves)


class DLXNode:
    def __init__(self) -> None:
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.column: Optional["DLXColumn"] = None
        self.row_id: int = -1


class DLXColumn(DLXNode):
    def __init__(self, name: Union[int, str]) -> None:
        super().__init__()
        self.name = name
        self.size = 0


def dlx_build(
    column_count: int,
    rows: List[Tuple[int, List[int]]],
) -> Tuple[DLXColumn, List[DLXColumn]]:
    root = DLXColumn("root")
    columns: List[DLXColumn] = []
    prev: DLXColumn = root
    for idx in range(column_count):
        col = DLXColumn(idx)
        col.left = prev
        col.right = root
        prev.right = col
        root.left = col
        prev = col
        columns.append(col)

    for row_id, col_ids in rows:
        first: Optional[DLXNode] = None
        prev_node: Optional[DLXNode] = None
        for col_id in col_ids:
            col = columns[col_id]
            node = DLXNode()
            node.column = col
            node.row_id = row_id
            node.down = col
            node.up = col.up
            col.up.down = node
            col.up = node
            col.size += 1
            if first is None:
                first = node
                node.left = node
                node.right = node
            else:
                node.left = prev_node
                node.right = first
                prev_node.right = node
                first.left = node
            prev_node = node
    return root, columns


def dlx_solve(
    root: DLXColumn,
    time_limit: Optional[float] = None,
) -> Tuple[Optional[List[DLXNode]], bool]:
    start_time = time.monotonic()
    deadline = start_time + time_limit if time_limit is not None else None
    solution: List[DLXNode] = []
    timed_out = False

    def choose_column() -> Optional[DLXColumn]:
        c = root.right
        best = None
        while c != root:
            if best is None or c.size < best.size:
                best = c
                if best.size <= 1:
                    break
            c = c.right
        return best

    def cover(col: DLXColumn) -> None:
        col.right.left = col.left
        col.left.right = col.right
        row = col.down
        while row != col:
            node = row.right
            while node != row:
                node.down.up = node.up
                node.up.down = node.down
                node.column.size -= 1
                node = node.right
            row = row.down

    def uncover(col: DLXColumn) -> None:
        row = col.up
        while row != col:
            node = row.left
            while node != row:
                node.column.size += 1
                node.down.up = node
                node.up.down = node
                node = node.left
            row = row.up
        col.right.left = col
        col.left.right = col

    def search() -> Optional[List[DLXNode]]:
        nonlocal timed_out
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            return None
        if root.right == root:
            return list(solution)
        col = choose_column()
        if col is None or col.size == 0:
            return None
        cover(col)
        row = col.down
        while row != col:
            solution.append(row)
            node = row.right
            while node != row:
                cover(node.column)
                node = node.right
            result = search()
            if result is not None:
                return result
            node = row.left
            while node != row:
                uncover(node.column)
                node = node.left
            solution.pop()
            row = row.down
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
        uncover(col)
        return None

    return search(), timed_out

def build_target_order(spec: ModelSpec) -> List[int]:
    pos_to_token = [0] * spec.width
    for token, pos in enumerate(spec.outputs):
        pos_to_token[pos] = token
    return pos_to_token


def build_min_swap_sequence(spec: ModelSpec) -> List[int]:
    target_order = build_target_order(spec)
    order = list(range(spec.width))
    swaps: List[int] = []
    for i in range(spec.width):
        target_tok = target_order[i]
        j = order.index(target_tok)
        while j > i:
            swaps.append(j - 1)
            order[j - 1], order[j] = order[j], order[j - 1]
            j -= 1
    return swaps


def compute_min_right_counts(spec: ModelSpec, swaps: List[int]) -> List[int]:
    order = list(range(spec.width))
    r_min = [0] * spec.width
    for pos in swaps:
        left = order[pos]
        r_min[left] += 1
        order[pos], order[pos + 1] = order[pos + 1], order[pos]
    return r_min


def collect_adjacent_edges(spec: ModelSpec, swaps: List[int]) -> List[Tuple[int, int]]:
    order = list(range(spec.width))
    edges = set()
    for i in range(spec.width - 1):
        a, b = order[i], order[i + 1]
        edges.add((min(a, b), max(a, b)))
    for pos in swaps:
        for i in range(spec.width - 1):
            a, b = order[i], order[i + 1]
            edges.add((min(a, b), max(a, b)))
        order[pos], order[pos + 1] = order[pos + 1], order[pos]
    for i in range(spec.width - 1):
        a, b = order[i], order[i + 1]
        edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def assign_loop_pairs(
    extra: List[int],
    neighbors: List[List[int]],
) -> Optional[Tuple[Tuple[int, int], ...]]:
    extra_tuple = tuple(extra)
    needed_depth = sum(extra) // 2 + 10
    if needed_depth > sys.getrecursionlimit():
        sys.setrecursionlimit(needed_depth + 1000)

    @lru_cache(maxsize=None)
    def dfs(state: Tuple[int, ...]) -> Optional[Tuple[Tuple[int, int], ...]]:
        if all(val == 0 for val in state):
            return ()
        active = [i for i, val in enumerate(state) if val > 0]
        seen: set[int] = set()
        for start in active:
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            total = 0
            while stack:
                v = stack.pop()
                total += state[v]
                for nb in neighbors[v]:
                    if state[nb] > 0 and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            if total & 1:
                return None

        best = None
        best_cand = None
        best_rem = None
        best_list: List[int] = []
        active_neighbor_counts: Dict[int, int] = {}
        for v in active:
            cands = [u for u in neighbors[v] if state[u] > 0]
            if not cands:
                return None
            cnt = len(cands)
            active_neighbor_counts[v] = cnt
            rem = state[v]
            if best is None or cnt < best_cand or (cnt == best_cand and rem > best_rem):
                best = v
                best_cand = cnt
                best_rem = rem
                best_list = cands

        best_list.sort(
            key=lambda u: (state[u], -active_neighbor_counts[u]), reverse=True
        )
        for u in best_list:
            new_state = list(state)
            new_state[best] -= 1
            new_state[u] -= 1
            if new_state[best] < 0 or new_state[u] < 0:
                continue
            res = dfs(tuple(new_state))
            if res is not None:
                return ((best, u),) + res
        return None

    return dfs(extra_tuple)


def assign_loop_pairs_greedy(
    extra: List[int],
    neighbors: List[List[int]],
) -> Optional[Tuple[Tuple[int, int], ...]]:
    remaining = extra[:]
    pairs: List[Tuple[int, int]] = []

    # Parity check per connected component.
    seen: set[int] = set()
    for i in range(len(remaining)):
        if remaining[i] == 0 or i in seen:
            continue
        stack = [i]
        seen.add(i)
        total = 0
        while stack:
            v = stack.pop()
            total += remaining[v]
            for nb in neighbors[v]:
                if remaining[nb] > 0 and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if total & 1:
            return None

    heap = [(-remaining[i], i) for i in range(len(remaining)) if remaining[i] > 0]
    heapq.heapify(heap)

    while heap:
        negd, v = heapq.heappop(heap)
        d = -negd
        if remaining[v] != d:
            continue
        if d == 0:
            continue
        best = None
        best_d = -1
        for u in neighbors[v]:
            if remaining[u] > best_d:
                best_d = remaining[u]
                best = u
        if best is None or best_d == 0:
            return None
        remaining[v] -= 1
        remaining[best] -= 1
        pairs.append((v, best))
        if remaining[v] > 0:
            heapq.heappush(heap, (-remaining[v], v))
        if remaining[best] > 0:
            heapq.heappush(heap, (-remaining[best], best))

    return tuple(pairs)


def construct_serial_solution(spec: ModelSpec) -> Optional[List[str]]:
    swaps = build_min_swap_sequence(spec)
    r_min = compute_min_right_counts(spec, swaps)
    extra = [spec.plinks[t] - r_min[t] for t in range(spec.width)]
    if any(val < 0 for val in extra):
        return None

    edges = collect_adjacent_edges(spec, swaps)
    neighbors: List[List[int]] = [[] for _ in range(spec.width)]
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    extra_sum = sum(extra)
    if extra_sum > 10000:
        pairs = assign_loop_pairs_greedy(extra, neighbors)
        if pairs is None:
            pairs = assign_loop_pairs(extra, neighbors)
    else:
        pairs = assign_loop_pairs(extra, neighbors)
    if pairs is None:
        return None
    loop_counts: Dict[Tuple[int, int], int] = {}
    for a, b in pairs:
        edge = (a, b) if a < b else (b, a)
        loop_counts[edge] = loop_counts.get(edge, 0) + 1

    order = list(range(spec.width))
    row_strings: List[str] = []

    def emit_loops() -> None:
        for i in range(spec.width - 1):
            a, b = order[i], order[i + 1]
            edge = (a, b) if a < b else (b, a)
            loops = loop_counts.get(edge, 0)
            if loops:
                row = ["|"] * spec.width
                row[i] = ">"
                row[i + 1] = "<"
                row_str = "".join(row)
                for _ in range(loops):
                    row_strings.append(row_str)
                    row_strings.append(row_str)
                loop_counts[edge] = 0

    for pos in swaps:
        emit_loops()
        row = ["|"] * spec.width
        row[pos] = ">"
        row[pos + 1] = "<"
        row_strings.append("".join(row))
        order[pos], order[pos + 1] = order[pos + 1], order[pos]
    emit_loops()
    if any(count > 0 for count in loop_counts.values()):
        return None
    return row_strings


def build_rows_from_paths(
    token_paths: List[Tuple[int, ...]],
    width: int,
    rows: int,
) -> List[str]:
    pos_to_token = [[-1] * width for _ in range(rows + 1)]
    for tok, path in enumerate(token_paths):
        for r, pos in enumerate(path):
            if pos_to_token[r][pos] != -1:
                raise ValueError("position conflict in exact cover solution")
            pos_to_token[r][pos] = tok

    row_strings: List[str] = []
    for r in range(rows):
        row = ["|"] * width
        p = 0
        while p < width:
            tok = pos_to_token[r][p]
            if tok < 0:
                raise ValueError("missing token in exact cover solution")
            next_pos = token_paths[tok][r + 1]
            if next_pos == p + 1:
                other = pos_to_token[r][p + 1]
                if other < 0 or token_paths[other][r + 1] != p:
                    raise ValueError("invalid swap in exact cover solution")
                row[p] = ">"
                row[p + 1] = "<"
                p += 2
            elif next_pos == p:
                row[p] = "|"
                p += 1
            elif next_pos == p - 1:
                raise ValueError("unexpected left move in exact cover solution")
            else:
                raise ValueError("unexpected move in exact cover solution")
        row_strings.append("".join(row))
    return row_strings


def solve_model_exact_cover(
    spec: ModelSpec,
    rows: int,
    time_limit: Optional[float] = None,
) -> Tuple[Optional[List[str]], bool]:
    width = spec.width
    token_offset = (rows + 1) * width
    column_count = token_offset + width
    row_infos: List[Tuple[int, Tuple[int, ...]]] = []
    row_entries: List[Tuple[int, List[int]]] = []

    for tok in range(width):
        right_moves = spec.plinks[tok]
        left_moves = right_moves - (spec.outputs[tok] - tok)
        if left_moves < 0:
            return None, False
        paths = list(
            enumerate_paths_with_target(
                tok,
                spec.outputs[tok],
                right_moves,
                left_moves,
                rows,
                width,
            )
        )
        if not paths:
            return None, False
        for path in paths:
            row_id = len(row_infos)
            row_infos.append((tok, tuple(path)))
            cols = [token_offset + tok]
            cols.extend(r * width + pos for r, pos in enumerate(path))
            row_entries.append((row_id, cols))

    root, _columns = dlx_build(column_count, row_entries)
    solution_nodes, timed_out = dlx_solve(root, time_limit=time_limit)
    if solution_nodes is None:
        return None, timed_out

    token_paths: List[Optional[Tuple[int, ...]]] = [None] * width
    for node in solution_nodes:
        tok, path = row_infos[node.row_id]
        token_paths[tok] = path
    if any(path is None for path in token_paths):
        return None, timed_out

    rows_out = build_rows_from_paths(
        [path for path in token_paths if path is not None],
        width,
        rows,
    )
    return rows_out, False


def solve_model_serial_dfs(
    spec: ModelSpec,
    rows: int,
    time_limit: Optional[float] = None,
) -> Optional[List[str]]:
    width = spec.width
    targets = spec.outputs
    max_swaps_row = 1
    start_time = time.monotonic()

    pos_to_token = tuple(range(width))
    r_rem = tuple(spec.plinks)

    def timed_out() -> bool:
        return time_limit is not None and (time.monotonic() - start_time) > time_limit

    def build_token_pos(pos_to_token: Tuple[int, ...]) -> List[int]:
        token_pos = [0] * width
        for p, tok in enumerate(pos_to_token):
            token_pos[tok] = p
        return token_pos

    def feasible_state(
        row_idx: int, pos_to_token: Tuple[int, ...], token_pos: List[int], r_rem: Tuple[int, ...]
    ) -> bool:
        rows_left = rows - row_idx
        total_r = sum(r_rem)
        if total_r != rows_left:
            return False
        n = width
        bit = [0] * (n + 1)

        def bit_add(idx: int, val: int) -> None:
            idx += 1
            while idx <= n:
                bit[idx] += val
                idx += idx & -idx

        def bit_sum(idx: int) -> int:
            idx += 1
            acc = 0
            while idx > 0:
                acc += bit[idx]
                idx -= idx & -idx
            return acc

        inv_right = [0] * width
        sum_inv = 0
        for p in range(n - 1, -1, -1):
            tok = pos_to_token[p]
            tpos = targets[tok]
            inv_right[tok] = bit_sum(tpos - 1) if tpos > 0 else 0
            bit_add(tpos, 1)
            sum_inv += inv_right[tok]
        if (total_r - sum_inv) & 1:
            return False
        for token in range(width):
            p = token_pos[token]
            r = r_rem[token]
            if r < inv_right[token]:
                return False
            l = r - (targets[token] - p)
            if l < 0:
                return False
            if r + l > rows_left:
                return False
        return True

    @lru_cache(maxsize=None)
    def dfs(
        row_idx: int,
        pos_to_token: Tuple[int, ...],
        r_rem: Tuple[int, ...],
    ) -> Optional[Tuple[str, ...]]:
        if timed_out():
            return None
        token_pos = build_token_pos(pos_to_token)
        if not feasible_state(row_idx, pos_to_token, token_pos, r_rem):
            return None
        if row_idx == rows:
            if all(r == 0 for r in r_rem) and all(
                token_pos[t] == targets[t] for t in range(width)
            ):
                return ()
            return None

        candidates = []
        for i in range(width - 1):
            left = pos_to_token[i]
            right = pos_to_token[i + 1]
            if r_rem[left] <= 0:
                continue
            l_right = r_rem[right] - (targets[right] - (i + 1))
            if l_right <= 0:
                continue
            dist_before = abs(targets[left] - i) + abs(targets[right] - (i + 1))
            dist_after = abs(targets[left] - (i + 1)) + abs(targets[right] - i)
            score = dist_after - dist_before
            candidates.append((score, i))

        candidates.sort()
        for _, i in candidates:
            left = pos_to_token[i]
            new_pos = list(pos_to_token)
            new_pos[i], new_pos[i + 1] = new_pos[i + 1], new_pos[i]
            new_r = list(r_rem)
            new_r[left] -= 1
            if new_r[left] < 0:
                continue
            row = ["|"] * width
            row[i] = ">"
            row[i + 1] = "<"
            next_rows = dfs(row_idx + 1, tuple(new_pos), tuple(new_r))
            if next_rows is not None:
                return ("".join(row),) + next_rows
        return None

    result = dfs(0, pos_to_token, r_rem)
    if result is None:
        return None
    return list(result)


def min_rows_needed(spec: ModelSpec) -> int:
    width = spec.width
    max_swaps_row = width // 2
    total_plinks = sum(spec.plinks)
    min_rows_by_swaps = (total_plinks + max_swaps_row - 1) // max_swaps_row
    max_per_token = 0
    for x, (y, z) in enumerate(zip(spec.outputs, spec.plinks)):
        right = z
        left = z - (y - x)
        if left < 0:
            raise ValueError("negative left moves")
        max_per_token = max(max_per_token, right + left)
    return max(min_rows_by_swaps, max_per_token)


def build_forced_edges_from_path(path: List[int]) -> List[Optional[Tuple[int, str]]]:
    rows = len(path) - 1
    edges: List[Optional[Tuple[int, str]]] = [None] * rows
    for row in range(rows):
        delta = path[row + 1] - path[row]
        if delta == 1:
            edges[row] = (path[row], "R")
        elif delta == -1:
            edges[row] = (path[row + 1], "L")
    return edges


def merge_forced_edges(
    base: List[Optional[Tuple[int, str]]],
    overlay: List[Optional[Tuple[int, str]]],
) -> List[Optional[Tuple[int, str]]]:
    if len(base) != len(overlay):
        raise ValueError("forced_edges length mismatch")
    for idx, edge in enumerate(overlay):
        if edge is None:
            continue
        if base[idx] is not None and base[idx] != edge:
            raise ValueError(f"forced edge conflict at row {idx}")
        base[idx] = edge
    return base


def merge_forced_positions(
    base: List[Optional[int]],
    overlay: List[Optional[int]],
) -> List[Optional[int]]:
    if len(base) != len(overlay):
        raise ValueError("forced_positions length mismatch")
    for idx, pos in enumerate(overlay):
        if pos is None:
            continue
        if base[idx] is not None and base[idx] != pos:
            raise ValueError(f"forced position conflict at row {idx}")
        base[idx] = pos
    return base


def parse_forced_edges(
    entries: Optional[List[str]], rows: int, width: int
) -> List[Optional[Tuple[int, str]]]:
    forced_edges: List[Optional[Tuple[int, str]]] = [None] * rows
    if not entries:
        return forced_edges
    for entry in entries:
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid forced edge: {entry}")
        row = int(parts[0])
        pos = int(parts[1])
        direction = parts[2].upper()
        if direction not in ("L", "R"):
            raise ValueError(f"invalid forced edge dir: {entry}")
        if row < 0 or row >= rows:
            raise ValueError(f"forced edge row out of range: {entry}")
        if pos < 0 or pos >= width - 1:
            raise ValueError(f"forced edge pos out of range: {entry}")
        edge = (pos, direction)
        if forced_edges[row] is not None and forced_edges[row] != edge:
            raise ValueError(f"forced edge conflict at row {row}")
        forced_edges[row] = edge
    return forced_edges


def parse_forced_positions(
    entries: Optional[List[str]], rows: int, width: int
) -> List[Optional[int]]:
    forced_positions: List[Optional[int]] = [None] * rows
    if not entries:
        return forced_positions
    for entry in entries:
        parts = entry.split(":")
        if len(parts) != 2:
            raise ValueError(f"invalid forced position: {entry}")
        row = int(parts[0])
        pos = int(parts[1])
        if row < 0 or row >= rows:
            raise ValueError(f"forced position row out of range: {entry}")
        if pos < 0 or pos >= width:
            raise ValueError(f"forced position out of range: {entry}")
        if forced_positions[row] is not None and forced_positions[row] != pos:
            raise ValueError(f"forced position conflict at row {row}")
        forced_positions[row] = pos
    return forced_positions


def solve_model(
    spec: ModelSpec,
    rows: int,
    forced_edges: Optional[List[Optional[Tuple[int, str]]]] = None,
    forced_token: Optional[int] = None,
    forced_positions: Optional[List[Optional[int]]] = None,
    forced_path: Optional[List[int]] = None,
    row_swap_limit: Optional[int] = None,
    time_limit: Optional[float] = None,
) -> Optional[List[str]]:
    width = spec.width
    max_swaps_row = width // 2
    if row_swap_limit is not None:
        if row_swap_limit <= 0:
            raise ValueError("row_swap_limit must be positive")
        if row_swap_limit > max_swaps_row:
            raise ValueError("row_swap_limit exceeds maximum swaps per row")
        max_swaps_row = row_swap_limit
    targets = spec.outputs

    if forced_edges is None:
        forced_edges = [None] * rows
    if len(forced_edges) != rows:
        raise ValueError("forced_edges length mismatch")
    if forced_token is None and any(edge is not None for edge in forced_edges):
        raise ValueError("forced_edges requires forced_token")
    if forced_positions is not None:
        if forced_token is None:
            raise ValueError("forced_positions requires forced_token")
        if len(forced_positions) != rows:
            raise ValueError("forced_positions length mismatch")
        for pos in forced_positions:
            if pos is not None and (pos < 0 or pos >= width):
                raise ValueError("forced_positions out of range")
    if forced_path is not None:
        if forced_token is None:
            raise ValueError("forced_path requires forced_token")
        if len(forced_path) != rows + 1:
            raise ValueError("forced_path length mismatch")
        if forced_positions is not None:
            for idx, pos in enumerate(forced_positions):
                if pos is not None and pos != forced_path[idx]:
                    raise ValueError("forced_positions/forced_path mismatch")

    pos_to_token = tuple(range(width))
    r_rem = tuple(spec.plinks)
    start_time = time.monotonic()

    def build_token_pos(pos_to_token: Tuple[int, ...]) -> List[int]:
        token_pos = [0] * width
        for p, tok in enumerate(pos_to_token):
            token_pos[tok] = p
        return token_pos

    def timed_out() -> bool:
        return time_limit is not None and (time.monotonic() - start_time) > time_limit

    def feasible_state(
        row_idx: int, pos_to_token: Tuple[int, ...], token_pos: List[int], r_rem: Tuple[int, ...]
    ) -> bool:
        rows_left = rows - row_idx
        total_r = sum(r_rem)
        if total_r > rows_left * max_swaps_row:
            return False
        # Each token must still cross all remaining inversions.
        n = width
        bit = [0] * (n + 1)

        def bit_add(idx: int, val: int) -> None:
            idx += 1
            while idx <= n:
                bit[idx] += val
                idx += idx & -idx

        def bit_sum(idx: int) -> int:
            idx += 1
            acc = 0
            while idx > 0:
                acc += bit[idx]
                idx -= idx & -idx
            return acc

        inv_right = [0] * width
        sum_inv = 0
        for p in range(n - 1, -1, -1):
            tok = pos_to_token[p]
            tpos = targets[tok]
            inv_right[tok] = bit_sum(tpos - 1) if tpos > 0 else 0
            bit_add(tpos, 1)
            sum_inv += inv_right[tok]
        if (total_r - sum_inv) & 1:
            return False
        for token in range(width):
            p = token_pos[token]
            r = r_rem[token]
            if r < 0:
                return False
            if r < inv_right[token]:
                return False
            l = r - (targets[token] - p)
            if l < 0:
                return False
            if r + l > rows_left:
                return False
        return True

    @lru_cache(maxsize=None)
    def dfs(
        row_idx: int,
        pos_to_token: Tuple[int, ...],
        r_rem: Tuple[int, ...],
    ) -> Optional[Tuple[str, ...]]:
        if timed_out():
            return None
        token_pos = build_token_pos(pos_to_token)
        if not feasible_state(row_idx, pos_to_token, token_pos, r_rem):
            return None
        if forced_positions is not None:
            forced_pos = forced_positions[row_idx]
            if forced_pos is not None and token_pos[forced_token] != forced_pos:
                return None
        if forced_path is not None:
            if token_pos[forced_token] != forced_path[row_idx]:
                return None
        if row_idx == rows:
            if all(r == 0 for r in r_rem) and all(
                token_pos[t] == targets[t] for t in range(width)
            ):
                return ()
            return None

        rows_left = rows - row_idx
        total_r = sum(r_rem)
        min_swaps_needed = max(0, total_r - (rows_left - 1) * max_swaps_row)
        max_swaps_allowed = min(max_swaps_row, total_r)

        must_left = [False] * width
        must_right = [False] * width
        must_swap = [False] * width
        cannot_left = [False] * width
        cannot_right = [False] * width
        must_any = [False] * width

        for pos in range(width):
            tok = pos_to_token[pos]
            r = r_rem[tok]
            l = r - (targets[tok] - pos)
            total = r + l
            if total == rows_left:
                if r == 0:
                    must_left[pos] = True
                elif l == 0:
                    must_right[pos] = True
                else:
                    must_swap[pos] = True
            if l == 0:
                cannot_left[pos] = True
            if r == 0:
                cannot_right[pos] = True
            must_any[pos] = must_left[pos] or must_right[pos] or must_swap[pos]

        suffix_must = [0] * (width + 1)
        for pos in range(width - 1, -1, -1):
            suffix_must[pos] = suffix_must[pos + 1] + (1 if must_any[pos] else 0)

        forced_edge_pos = None
        forced_edge_dir = None
        forced_edge = forced_edges[row_idx]
        if forced_edge is not None:
            forced_edge_pos, forced_edge_dir = forced_edge

        def build_row(
            i: int,
            cur_pos_to_token: Tuple[int, ...],
            cur_r_rem: Tuple[int, ...],
            row_chars: List[str],
            swaps_used: int,
        ) -> Optional[Tuple[str, ...]]:
            if swaps_used > max_swaps_allowed:
                return None
            remaining_slots = width - i
            max_additional = (remaining_slots // 2)
            if swaps_used + max_additional < min_swaps_needed:
                return None
            min_required_swaps = (suffix_must[i] + 1) // 2
            if swaps_used + min_required_swaps > max_swaps_allowed:
                return None

            if i >= width:
                if swaps_used < min_swaps_needed:
                    return None
                if forced_path is not None:
                    pos_after = None
                    for idx, tok in enumerate(cur_pos_to_token):
                        if tok == forced_token:
                            pos_after = idx
                            break
                    if pos_after is None or pos_after != forced_path[row_idx + 1]:
                        return None
                next_rows = dfs(row_idx + 1, cur_pos_to_token, cur_r_rem)
                if next_rows is None:
                    return None
                return ("".join(row_chars),) + next_rows

            if must_left[i]:
                return None

            force_swap = must_right[i] or must_swap[i]
            if i + 1 < width:
                must_right_j = must_right[i + 1]
                cannot_left_j = cannot_left[i + 1]
                if must_left[i + 1]:
                    force_swap = True
            else:
                must_right_j = False
                cannot_left_j = False
            block_swap = forced_edge_pos is not None and i == forced_edge_pos - 1
            if forced_edge_pos is not None and i == forced_edge_pos:
                force_swap = True

            def try_swap() -> Optional[Tuple[str, ...]]:
                if i + 1 >= width:
                    return None
                if block_swap:
                    return None
                if cannot_right[i]:
                    return None
                if cannot_left_j:
                    return None
                if must_right_j:
                    return None
                if forced_edge_pos is not None and i == forced_edge_pos:
                    if forced_token is None:
                        return None
                    if forced_edge_dir == "R":
                        if cur_pos_to_token[i] != forced_token:
                            return None
                    elif forced_edge_dir == "L":
                        if cur_pos_to_token[i + 1] != forced_token:
                            return None
                    else:
                        raise ValueError("forced edge dir must be L or R")
                left = cur_pos_to_token[i]
                right = cur_pos_to_token[i + 1]
                new_pos_to_token = list(cur_pos_to_token)
                new_pos_to_token[i], new_pos_to_token[i + 1] = (
                    new_pos_to_token[i + 1],
                    new_pos_to_token[i],
                )
                new_r_rem = list(cur_r_rem)
                new_r_rem[left] -= 1
                new_row = row_chars[:]
                new_row[i] = ">"
                new_row[i + 1] = "<"
                return build_row(
                    i + 2,
                    tuple(new_pos_to_token),
                    tuple(new_r_rem),
                    new_row,
                    swaps_used + 1,
                )

            if force_swap:
                res = try_swap()
                if res is not None:
                    return res
                return None

            # try swap / no swap with a simple local heuristic
            swap_first = True
            if i + 1 < width:
                left = cur_pos_to_token[i]
                right = cur_pos_to_token[i + 1]
                dist_before = abs(targets[left] - i) + abs(targets[right] - (i + 1))
                dist_after = abs(targets[left] - (i + 1)) + abs(targets[right] - i)
                swap_first = dist_after <= dist_before

            def try_no_swap() -> Optional[Tuple[str, ...]]:
                if must_right[i] or must_swap[i]:
                    return None
                new_row = row_chars[:]
                new_row[i] = "|"
                return build_row(
                    i + 1, cur_pos_to_token, cur_r_rem, new_row, swaps_used
                )

            if swap_first:
                res = try_swap()
                if res is not None:
                    return res
                res = try_no_swap()
                if res is not None:
                    return res
            else:
                res = try_no_swap()
                if res is not None:
                    return res
                res = try_swap()
                if res is not None:
                    return res
            return None

        row_chars = ["|"] * width
        return build_row(0, pos_to_token, r_rem, row_chars, 0)

    return list(dfs(0, pos_to_token, r_rem) or [])


def solve_model_heuristic(
    spec: ModelSpec,
    rows: int,
    beam_width: int = 200,
    candidates: int = 40,
    seed: Optional[int] = None,
    time_limit: Optional[float] = None,
    forced_edges: Optional[List[Optional[Tuple[int, str]]]] = None,
    forced_token: Optional[int] = None,
    forced_path: Optional[List[int]] = None,
    forced_positions: Optional[List[Optional[int]]] = None,
    row_swap_limit: Optional[int] = None,
) -> Optional[List[str]]:
    width = spec.width
    max_swaps_row = width // 2
    if row_swap_limit is not None:
        if row_swap_limit <= 0:
            raise ValueError("row_swap_limit must be positive")
        if row_swap_limit > max_swaps_row:
            raise ValueError("row_swap_limit exceeds maximum swaps per row")
        max_swaps_row = row_swap_limit
    targets = spec.outputs
    rng = random.Random(seed)
    start_time = time.monotonic()

    if forced_edges is None:
        forced_edges = [None] * rows
    if len(forced_edges) != rows:
        raise ValueError("forced_edges length mismatch")
    if forced_token is None and any(edge is not None for edge in forced_edges):
        raise ValueError("forced_edges requires forced_token")
    if forced_path is not None:
        if forced_token is None:
            raise ValueError("forced_path requires forced_token")
        if len(forced_path) != rows + 1:
            raise ValueError("forced_path length mismatch")
    if forced_positions is not None:
        if forced_token is None:
            raise ValueError("forced_positions requires forced_token")
        if len(forced_positions) != rows:
            raise ValueError("forced_positions length mismatch")
        for pos in forced_positions:
            if pos is not None and (pos < 0 or pos >= width):
                raise ValueError("forced_positions out of range")
        if forced_path is not None:
            for idx, pos in enumerate(forced_positions):
                if pos is not None and pos != forced_path[idx]:
                    raise ValueError("forced_positions/forced_path mismatch")

    def timed_out() -> bool:
        return time_limit is not None and (time.monotonic() - start_time) > time_limit

    def build_token_pos(pos_to_token: Tuple[int, ...]) -> List[int]:
        token_pos = [0] * width
        for p, tok in enumerate(pos_to_token):
            token_pos[tok] = p
        return token_pos

    def feasible_state(
        row_idx: int, pos_to_token: Tuple[int, ...], token_pos: List[int], r_rem: Tuple[int, ...]
    ) -> bool:
        rows_left = rows - row_idx
        total_r = sum(r_rem)
        if total_r > rows_left * max_swaps_row:
            return False
        n = width
        bit = [0] * (n + 1)

        def bit_add(idx: int, val: int) -> None:
            idx += 1
            while idx <= n:
                bit[idx] += val
                idx += idx & -idx

        def bit_sum(idx: int) -> int:
            idx += 1
            acc = 0
            while idx > 0:
                acc += bit[idx]
                idx -= idx & -idx
            return acc

        inv_right = [0] * width
        sum_inv = 0
        for p in range(n - 1, -1, -1):
            tok = pos_to_token[p]
            tpos = targets[tok]
            inv_right[tok] = bit_sum(tpos - 1) if tpos > 0 else 0
            bit_add(tpos, 1)
            sum_inv += inv_right[tok]
        if (total_r - sum_inv) & 1:
            return False
        for token in range(width):
            p = token_pos[token]
            r = r_rem[token]
            if r < 0:
                return False
            if r < inv_right[token]:
                return False
            l = r - (targets[token] - p)
            if l < 0:
                return False
            if r + l > rows_left:
                return False
        return True

    def state_score(pos_to_token: Tuple[int, ...], r_rem: Tuple[int, ...], rows_left: int) -> int:
        token_pos = build_token_pos(pos_to_token)
        dist = sum(abs(targets[t] - token_pos[t]) for t in range(width))
        return dist + 2 * sum(r_rem) + rows_left

    def build_row_random(
        pos_to_token: Tuple[int, ...],
        r_rem: Tuple[int, ...],
        row_idx: int,
        jitter: float = 0.2,
    ) -> Optional[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]:
        rows_left = rows - row_idx
        total_r = sum(r_rem)
        min_swaps_needed = max(0, total_r - (rows_left - 1) * max_swaps_row)
        max_swaps_allowed = min(max_swaps_row, total_r)

        must_left = [False] * width
        must_right = [False] * width
        must_swap = [False] * width
        cannot_left = [False] * width
        cannot_right = [False] * width
        must_any = [False] * width

        for pos in range(width):
            tok = pos_to_token[pos]
            r = r_rem[tok]
            l = r - (targets[tok] - pos)
            total = r + l
            if total == rows_left:
                if r == 0:
                    must_left[pos] = True
                elif l == 0:
                    must_right[pos] = True
                else:
                    must_swap[pos] = True
            if l == 0:
                cannot_left[pos] = True
            if r == 0:
                cannot_right[pos] = True
            must_any[pos] = must_left[pos] or must_right[pos] or must_swap[pos]

        suffix_must = [0] * (width + 1)
        for pos in range(width - 1, -1, -1):
            suffix_must[pos] = suffix_must[pos + 1] + (1 if must_any[pos] else 0)

        forced_edge_pos = None
        forced_edge_dir = None
        forced_edge = forced_edges[row_idx]
        if forced_edge is not None:
            forced_edge_pos, forced_edge_dir = forced_edge

        cur_pos = list(pos_to_token)
        cur_r = list(r_rem)
        row_chars = ["|"] * width
        swaps_used = 0
        i = 0
        while i < width:
            if swaps_used > max_swaps_allowed:
                return None
            remaining_slots = width - i
            max_additional = (remaining_slots // 2)
            if swaps_used + max_additional < min_swaps_needed:
                return None
            min_required_swaps = (suffix_must[i] + 1) // 2
            if swaps_used + min_required_swaps > max_swaps_allowed:
                return None

            if must_left[i]:
                return None

            force_swap = must_right[i] or must_swap[i]
            if i + 1 < width and must_left[i + 1]:
                force_swap = True
            block_swap = forced_edge_pos is not None and i == forced_edge_pos - 1
            if forced_edge_pos is not None and i == forced_edge_pos:
                force_swap = True

            def can_swap() -> bool:
                if i + 1 >= width:
                    return False
                if block_swap:
                    return False
                if cannot_right[i]:
                    return False
                if cannot_left[i + 1]:
                    return False
                if must_right[i + 1]:
                    return False
                return True

            def do_swap() -> bool:
                if forced_edge_pos is not None and i == forced_edge_pos:
                    if forced_token is None:
                        return False
                    if forced_edge_dir == "R":
                        if cur_pos[i] != forced_token:
                            return False
                    elif forced_edge_dir == "L":
                        if cur_pos[i + 1] != forced_token:
                            return False
                    else:
                        return False
                left = cur_pos[i]
                cur_r[left] -= 1
                if cur_r[left] < 0:
                    return False
                cur_pos[i], cur_pos[i + 1] = cur_pos[i + 1], cur_pos[i]
                row_chars[i] = ">"
                row_chars[i + 1] = "<"
                return True

            def do_no_swap() -> bool:
                if must_right[i] or must_swap[i]:
                    return False
                return True

            if force_swap:
                if not can_swap() or not do_swap():
                    return None
                swaps_used += 1
                i += 2
                continue

            swap_preferred = False
            if i + 1 < width:
                left = cur_pos[i]
                right = cur_pos[i + 1]
                dist_before = abs(targets[left] - i) + abs(targets[right] - (i + 1))
                dist_after = abs(targets[left] - (i + 1)) + abs(targets[right] - i)
                swap_preferred = dist_after <= dist_before
            if rng.random() < jitter:
                swap_preferred = not swap_preferred
            choices = ("swap", "noswap") if swap_preferred else ("noswap", "swap")

            moved = False
            for choice in choices:
                if choice == "swap":
                    if can_swap() and do_swap():
                        swaps_used += 1
                        i += 2
                        moved = True
                        break
                else:
                    if do_no_swap():
                        i += 1
                        moved = True
                        break
            if not moved:
                return None

        if swaps_used < min_swaps_needed:
            return None
        if forced_path is not None:
            try:
                pos_after = cur_pos.index(forced_token)
            except ValueError:
                return None
            if pos_after != forced_path[row_idx + 1]:
                return None
        if forced_positions is not None and row_idx + 1 < rows:
            forced_next = forced_positions[row_idx + 1]
            if forced_next is not None:
                try:
                    pos_after = cur_pos.index(forced_token)
                except ValueError:
                    return None
                if pos_after != forced_next:
                    return None
        return "".join(row_chars), tuple(cur_pos), tuple(cur_r)

    start_state = (tuple(range(width)), tuple(spec.plinks), [])
    beam = [start_state]
    for row_idx in range(rows):
        if timed_out():
            return None
        next_states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[int, List[str]]] = {}
        rows_left = rows - row_idx
        for pos_to_token, r_rem, rows_so_far in beam:
            if timed_out():
                return None
            if forced_path is not None or forced_positions is not None:
                token_pos = build_token_pos(pos_to_token)
                if forced_path is not None and token_pos[forced_token] != forced_path[row_idx]:
                    continue
                if forced_positions is not None:
                    forced_pos = forced_positions[row_idx]
                    if forced_pos is not None and token_pos[forced_token] != forced_pos:
                        continue

            if max_swaps_row == 1:
                forced_edge_pos = None
                forced_edge_dir = None
                forced_edge = forced_edges[row_idx]
                if forced_edge is not None:
                    forced_edge_pos, forced_edge_dir = forced_edge
                for i in range(width - 1):
                    if forced_edge_pos is not None and i != forced_edge_pos:
                        continue
                    left = pos_to_token[i]
                    right = pos_to_token[i + 1]
                    if r_rem[left] <= 0:
                        continue
                    l_right = r_rem[right] - (targets[right] - (i + 1))
                    if l_right <= 0:
                        continue
                    if forced_edge_pos is not None:
                        if forced_token is None:
                            continue
                        if forced_edge_dir == "R":
                            if left != forced_token:
                                continue
                        elif forced_edge_dir == "L":
                            if right != forced_token:
                                continue
                        else:
                            continue
                    new_pos = list(pos_to_token)
                    new_pos[i], new_pos[i + 1] = new_pos[i + 1], new_pos[i]
                    new_r = list(r_rem)
                    new_r[left] -= 1
                    if new_r[left] < 0:
                        continue
                    if forced_path is not None or forced_positions is not None:
                        if forced_token is None:
                            continue
                        pos_after = new_pos.index(forced_token)
                        if forced_path is not None and pos_after != forced_path[row_idx + 1]:
                            continue
                        if forced_positions is not None and row_idx + 1 < rows:
                            forced_next = forced_positions[row_idx + 1]
                            if forced_next is not None and pos_after != forced_next:
                                continue
                    row_chars = ["|"] * width
                    row_chars[i] = ">"
                    row_chars[i + 1] = "<"
                    row = "".join(row_chars)
                    new_pos_t = tuple(new_pos)
                    new_r_t = tuple(new_r)
                    token_pos = build_token_pos(new_pos_t)
                    if not feasible_state(row_idx + 1, new_pos_t, token_pos, new_r_t):
                        continue
                    score = state_score(new_pos_t, new_r_t, rows_left - 1)
                    key = (new_pos_t, new_r_t)
                    if key not in next_states or score < next_states[key][0]:
                        next_states[key] = (score, rows_so_far + [row])
            else:
                for _ in range(candidates):
                    res = build_row_random(pos_to_token, r_rem, row_idx)
                    if res is None:
                        continue
                    row, new_pos, new_r = res
                    token_pos = build_token_pos(new_pos)
                    if not feasible_state(row_idx + 1, new_pos, token_pos, new_r):
                        continue
                    score = state_score(new_pos, new_r, rows_left - 1)
                    key = (new_pos, new_r)
                    if key not in next_states or score < next_states[key][0]:
                        next_states[key] = (score, rows_so_far + [row])
        if not next_states:
            return None
        sorted_states = sorted(
            next_states.items(), key=lambda item: item[1][0]
        )
        beam = [
            (key[0], key[1], data[1])
            for key, data in sorted_states[:beam_width]
        ]

    for pos_to_token, r_rem, rows_so_far in beam:
        token_pos = build_token_pos(pos_to_token)
        if all(r == 0 for r in r_rem) and all(
            token_pos[t] == targets[t] for t in range(width)
        ):
            return rows_so_far
    return None


def load_specs_from_logs() -> Dict[str, Spec]:
    model_order = [
        "000",
        "010",
        "020",
        "030",
        "040",
        "050",
        "100",
        "200",
        "300",
        "400",
        "500",
    ]
    log_path = Path("volume9_bbarker_bk_specs_all.txt")
    return parse_specs(log_path, model_order)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Black Knots models")
    parser.add_argument("model", help="model id (ex: 010)")
    parser.add_argument("--rows", type=int, default=None, help="fixed row count")
    parser.add_argument("--max-rows", type=int, default=30, help="max rows to try")
    parser.add_argument("--heuristic", action="store_true", help="use heuristic beam search")
    parser.add_argument("--beam", type=int, default=200, help="beam width for heuristic")
    parser.add_argument(
        "--candidates",
        type=int,
        default=40,
        help="candidate rows per state per row for heuristic",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for heuristic")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="time limit in seconds per row count attempt",
    )
    parser.add_argument("--restarts", type=int, default=1, help="restarts per row count")
    parser.add_argument(
        "--row-swap-limit",
        type=int,
        default=None,
        help="limit swaps per row (1 = serial)",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="force one swap per row and default rows to sum(plinks)",
    )
    parser.add_argument(
        "--serial-dfs",
        action="store_true",
        help="use serial DFS (exactly one swap per row)",
    )
    parser.add_argument(
        "--construct",
        action="store_true",
        help="construct serial solution using swap-sequence augmentation",
    )
    parser.add_argument(
        "--exact-cover",
        action="store_true",
        help="solve using exact cover (DLX) over token paths",
    )
    parser.add_argument(
        "--force-token",
        type=int,
        default=None,
        help="token id to constrain (required for forced paths/edges)",
    )
    parser.add_argument(
        "--force-path",
        type=str,
        default=None,
        help="comma/space-separated positions (rows+1 entries)",
    )
    parser.add_argument(
        "--force-prefix",
        action="store_true",
        help="enumerate prefix paths for forced token",
    )
    parser.add_argument(
        "--prefix-rows",
        type=int,
        default=None,
        help="prefix row count for --force-prefix",
    )
    parser.add_argument(
        "--force-enum",
        action="store_true",
        help="enumerate all paths for forced token",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=0,
        help="limit enumerated paths (0 = all)",
    )
    parser.add_argument(
        "--path-offset",
        type=int,
        default=0,
        help="skip the first N enumerated paths",
    )
    parser.add_argument(
        "--force-edge",
        action="append",
        default=None,
        help="force swap edge at row:pos:dir (dir L/R)",
    )
    parser.add_argument(
        "--force-pos",
        action="append",
        default=None,
        help="force token position at row:pos",
    )
    args = parser.parse_args()

    specs = load_specs_from_logs()
    if args.model not in specs:
        raise SystemExit(f"unknown model: {args.model}")
    model_spec = build_model_spec(args.model, specs[args.model])
    if args.construct:
        if args.rows is not None and args.rows != sum(model_spec.plinks):
            raise SystemExit("--construct requires rows == sum(plinks)")
        solution = construct_serial_solution(model_spec)
        if solution:
            print(f"model {args.model} constructed with {len(solution)} rows")
            for row in solution:
                print(row)
            return
        raise SystemExit("no constructive solution found")
    if args.serial_dfs:
        if args.heuristic:
            raise SystemExit("--serial-dfs conflicts with --heuristic")
        if any(
            [
                args.force_path,
                args.force_enum,
                args.force_prefix,
                args.force_edge,
                args.force_pos,
            ]
        ):
            raise SystemExit("--serial-dfs does not support forced constraints")
        if args.row_swap_limit not in (None, 1):
            raise SystemExit("--serial-dfs conflicts with --row-swap-limit")
        rows_needed = sum(model_spec.plinks)
        if args.rows is not None and args.rows != rows_needed:
            raise SystemExit("--serial-dfs requires rows == sum(plinks)")
        rows = args.rows or rows_needed
        solution = solve_model_serial_dfs(
            model_spec, rows, time_limit=args.time_limit
        )
        if solution:
            print(f"model {args.model} solved with {rows} rows (serial dfs)")
            for row in solution:
                print(row)
            return
        raise SystemExit(f"no serial dfs solution found for {args.model}")
    force_constraints = any(
        [args.force_path, args.force_enum, args.force_prefix, args.force_edge, args.force_pos]
    )
    if args.exact_cover:
        if args.heuristic or args.serial or args.serial_dfs or args.construct:
            raise SystemExit("--exact-cover conflicts with other solver modes")
        if force_constraints:
            raise SystemExit("--exact-cover does not support forced constraints")
        if args.row_swap_limit is not None:
            raise SystemExit("--exact-cover does not support --row-swap-limit")
        start_rows = args.rows or min_rows_needed(model_spec)
        max_rows = args.rows or args.max_rows
        for rows in range(start_rows, max_rows + 1):
            solution, timed_out = solve_model_exact_cover(
                model_spec, rows, time_limit=args.time_limit
            )
            if solution:
                print(f"model {args.model} solved with {rows} rows (exact cover)")
                for row in solution:
                    print(row)
                return
            if timed_out:
                raise SystemExit(
                    f"exact cover timed out for {args.model} at rows {rows}"
                )
        raise SystemExit(f"no exact cover solution found for {args.model}")
    force_mode_count = sum(
        1
        for mode in [args.force_path, args.force_enum, args.force_prefix]
        if mode
    )
    if force_mode_count > 1:
        raise SystemExit("choose only one of --force-path/--force-enum/--force-prefix")
    if args.force_prefix and args.prefix_rows is None:
        raise SystemExit("--prefix-rows is required with --force-prefix")
    if force_constraints and args.rows is None:
        raise SystemExit("--rows is required when using forced constraints")
    if force_constraints and args.force_token is None:
        raise SystemExit("--force-token is required when using forced constraints")

    row_swap_limit = args.row_swap_limit
    if args.serial:
        if row_swap_limit is not None and row_swap_limit != 1:
            raise SystemExit("--serial conflicts with --row-swap-limit")
        row_swap_limit = 1
    if args.serial and args.rows is None:
        start_rows = sum(model_spec.plinks)
        max_rows = start_rows
    else:
        start_rows = args.rows or min_rows_needed(model_spec)
        max_rows = args.rows or args.max_rows

    forced_token = args.force_token
    forced_edges_base: Optional[List[Optional[Tuple[int, str]]]] = None
    forced_path: Optional[List[int]] = None
    forced_positions_base: Optional[List[Optional[int]]] = None
    right_moves = None
    left_moves = None
    if force_constraints:
        if forced_token < 0 or forced_token >= model_spec.width:
            raise SystemExit("force-token out of range")
        right_moves = model_spec.plinks[forced_token]
        left_moves = right_moves - (model_spec.outputs[forced_token] - forced_token)
        if left_moves < 0:
            raise SystemExit("negative left moves for forced token")
        forced_edges_base = parse_forced_edges(
            args.force_edge, start_rows, model_spec.width
        )
        if args.force_pos:
            forced_positions_base = parse_forced_positions(
                args.force_pos, start_rows, model_spec.width
            )

    if args.force_path:
        forced_path = parse_path_positions(args.force_path)
        if len(forced_path) != start_rows + 1:
            raise SystemExit("force-path length mismatch")
        if forced_path[0] != forced_token:
            raise SystemExit("force-path start position mismatch")
        if forced_path[-1] != model_spec.outputs[forced_token]:
            raise SystemExit("force-path end position mismatch")
        right_count = 0
        left_count = 0
        for idx in range(start_rows):
            pos = forced_path[idx]
            next_pos = forced_path[idx + 1]
            if pos < 0 or pos >= model_spec.width:
                raise SystemExit("force-path position out of range")
            if next_pos < 0 or next_pos >= model_spec.width:
                raise SystemExit("force-path position out of range")
            delta = next_pos - pos
            if delta not in (-1, 0, 1):
                raise SystemExit("force-path step out of range")
            if delta == 1:
                right_count += 1
            elif delta == -1:
                left_count += 1
        if forced_path[-1] < 0 or forced_path[-1] >= model_spec.width:
            raise SystemExit("force-path position out of range")
        if right_count != right_moves or left_count != left_moves:
            raise SystemExit("force-path move counts mismatch")
        if forced_positions_base is not None:
            for idx, pos in enumerate(forced_positions_base):
                if pos is not None and pos != forced_path[idx]:
                    raise SystemExit("force-pos conflicts with force-path")

    def run_solver(
        rows: int,
        forced_edges: Optional[List[Optional[Tuple[int, str]]]],
        forced_path_arg: Optional[List[int]],
        forced_positions_arg: Optional[List[Optional[int]]],
    ) -> Optional[List[str]]:
        if args.heuristic:
            for attempt in range(max(1, args.restarts)):
                seed = None if args.seed is None else args.seed + attempt
                solution = solve_model_heuristic(
                    model_spec,
                    rows,
                    beam_width=args.beam,
                    candidates=args.candidates,
                    seed=seed,
                    time_limit=args.time_limit,
                    forced_edges=forced_edges,
                    forced_token=forced_token,
                    forced_path=forced_path_arg,
                    forced_positions=forced_positions_arg,
                    row_swap_limit=row_swap_limit,
                )
                if solution:
                    return solution
            return None
        return solve_model(
            model_spec,
            rows,
            forced_edges=forced_edges,
            forced_token=forced_token,
            forced_path=forced_path_arg,
            forced_positions=forced_positions_arg,
            row_swap_limit=row_swap_limit,
            time_limit=args.time_limit,
        )

    if args.force_prefix:
        rows = start_rows
        prefix_rows = args.prefix_rows
        if prefix_rows < 0 or prefix_rows > rows:
            raise SystemExit("prefix-rows out of range")
        checked = 0
        for idx, prefix in enumerate(
            enumerate_token_prefixes(
                forced_token,
                right_moves,
                left_moves,
                rows,
                prefix_rows,
                model_spec.width,
            )
        ):
            if idx < args.path_offset:
                continue
            if args.max_paths and checked >= args.max_paths:
                break
            checked += 1
            forced_positions = [None] * rows
            if forced_positions_base is not None:
                forced_positions = list(forced_positions_base)
            conflict = False
            for row in range(prefix_rows + 1):
                pos = prefix[row]
                if forced_positions[row] is not None and forced_positions[row] != pos:
                    conflict = True
                    break
                forced_positions[row] = pos
            if conflict:
                continue
            solution = run_solver(rows, forced_edges_base, None, forced_positions)
            if solution:
                print(f"model {args.model} solved with {rows} rows (prefix {idx})")
                for row in solution:
                    print(row)
                return
        raise SystemExit(f"no solution found for {args.model} at rows {rows}")

    if args.force_enum:
        rows = start_rows
        stay_moves = rows - right_moves - left_moves
        if stay_moves < 0:
            raise SystemExit("rows too small for forced token moves")
        checked = 0
        for idx, path in enumerate(
            enumerate_token_paths(
                forced_token, right_moves, left_moves, rows, model_spec.width
            )
        ):
            if idx < args.path_offset:
                continue
            if args.max_paths and checked >= args.max_paths:
                break
            checked += 1
            if path[-1] != model_spec.outputs[forced_token]:
                continue
            forced_edges = list(forced_edges_base)
            try:
                forced_edges = merge_forced_edges(
                    forced_edges, build_forced_edges_from_path(path)
                )
            except ValueError:
                continue
            solution = run_solver(rows, forced_edges, path, forced_positions_base)
            if solution:
                print(f"model {args.model} solved with {rows} rows (path {idx})")
                for row in solution:
                    print(row)
                return
        raise SystemExit(f"no solution found for {args.model} at rows {rows}")

    forced_edges_for_path = None
    if forced_path is not None:
        forced_edges_for_path = merge_forced_edges(
            list(forced_edges_base),
            build_forced_edges_from_path(forced_path),
        )

    for rows in range(start_rows, max_rows + 1):
        if force_constraints:
            forced_edges = forced_edges_for_path or forced_edges_base
            solution = run_solver(rows, forced_edges, forced_path, forced_positions_base)
        else:
            solution = run_solver(rows, None, None, None)
        if solution:
            print(f"model {args.model} solved with {rows} rows")
            for row in solution:
                print(row)
            return
    raise SystemExit(f"no solution found up to {max_rows} rows")


if __name__ == "__main__":
    main()
