#!/usr/bin/env python3
"""Minimal 2D language interpreter for local checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


UNIT = ("unit",)
HASH_CROSSOVER_DEFAULT = True
STRICT_WIRES_DEFAULT = False


def inl(val):
    return ("inl", val)


def inr(val):
    return ("inr", val)


def pair(a, b):
    return ("pair", a, b)


@dataclass
class Box:
    name: str
    cmd: Tuple
    x: int
    y: int
    w: int
    h: int
    n_wire: Optional[int]
    w_wire: Optional[int]
    s_wire: Optional[int]
    e_wire: Optional[int]


@dataclass
class Module:
    name: str
    grid: List[List[str]]
    boxes: List[Box]
    n_input_wire: Optional[int]
    w_input_wire: Optional[int]
    output_wires: List[int]
    wire_count: int


class ParseError(RuntimeError):
    pass


class EvalError(RuntimeError):
    pass


# ---------- Parsing ----------


def parse_modules(
    text: str,
    *,
    hash_crossover: bool = HASH_CROSSOVER_DEFAULT,
    strict_wires: bool = STRICT_WIRES_DEFAULT,
) -> Dict[str, Module]:
    lines = text.splitlines()
    if not lines:
        raise ParseError("empty program")
    width = max(len(line) for line in lines)
    grid = [list(line.ljust(width)) for line in lines]

    used = [[False for _ in range(width)] for _ in range(len(lines))]
    modules = {}

    for y in range(len(lines)):
        for x in range(width):
            if grid[y][x] != "," or used[y][x]:
                continue
            # find top-right comma
            x2 = x + 1
            while x2 < width and grid[y][x2] != ",":
                x2 += 1
            if x2 >= width:
                continue
            # find bottom-left comma
            y2 = y + 1
            while y2 < len(lines) and grid[y2][x] != ",":
                y2 += 1
            if y2 >= len(lines):
                continue
            # verify bottom-right
            if grid[y2][x2] != ",":
                continue
            # mark used
            for yy in range(y, y2 + 1):
                for xx in range(x, x2 + 1):
                    used[yy][xx] = True
            module_grid = [row[x : x2 + 1] for row in grid[y : y2 + 1]]
            module = parse_module_grid(
                module_grid,
                hash_crossover=hash_crossover,
                strict_wires=strict_wires,
            )
            if module.name in modules:
                raise ParseError(f"duplicate module {module.name}")
            modules[module.name] = module

    if not modules:
        raise ParseError("no modules found")
    return modules


def parse_module_grid(
    grid: List[List[str]],
    *,
    hash_crossover: bool = HASH_CROSSOVER_DEFAULT,
    strict_wires: bool = STRICT_WIRES_DEFAULT,
) -> Module:
    height = len(grid)
    width = len(grid[0])
    if height < 3 or width < 3:
        raise ParseError("module too small")

    # Extract name from second row after left border
    name_row = "".join(grid[1][1:-1])
    name = name_row.strip().split(" ")[0]
    if not name:
        raise ParseError("module name not found")

    # Find inputs on borders
    n_input_wire = None
    w_input_wire = None
    output_wires: List[int] = []

    box_bounds = find_box_bounds(grid)
    north_connectors, west_connectors = connector_positions(box_bounds, width, height)

    wire_map, wire_count = build_wire_components(
        grid,
        hash_crossover=hash_crossover,
        north_connectors=north_connectors,
        west_connectors=west_connectors,
    )

    # North input
    n_inputs = [x for x in range(1, width - 1) if grid[0][x] == "|"]
    if len(n_inputs) > 1:
        raise ParseError("multiple north inputs")
    if n_inputs:
        n_input_wire = wire_map.get((n_inputs[0], 0, "S"))

    # West input
    w_inputs = [y for y in range(1, height - 1) if grid[y][0] == "-"]
    if len(w_inputs) > 1:
        raise ParseError("multiple west inputs")
    if w_inputs:
        w_input_wire = wire_map.get((0, w_inputs[0], "E"))

    # East outputs
    for y in range(1, height - 1):
        if grid[y][width - 1] == "-":
            wire_id = wire_map.get((width - 1, y, "W"))
            if wire_id is not None:
                output_wires.append(wire_id)
    output_wires = list(dict.fromkeys(output_wires))

    boxes = parse_boxes(grid, wire_map, box_bounds=box_bounds)
    if strict_wires:
        validate_wire_rules(
            grid,
            boxes,
            wire_map,
            wire_count,
            north_connectors=north_connectors,
            west_connectors=west_connectors,
        )
    return Module(
        name=name,
        grid=grid,
        boxes=boxes,
        n_input_wire=n_input_wire,
        w_input_wire=w_input_wire,
        output_wires=output_wires,
        wire_count=wire_count,
    )


def parse_boxes(
    grid: List[List[str]],
    wire_map: Dict[Tuple[int, int, str], int],
    *,
    box_bounds: Optional[List[Tuple[int, int, int]]] = None,
) -> List[Box]:
    boxes = []
    if box_bounds is None:
        box_bounds = find_box_bounds(grid)
    for x, y, w in box_bounds:
        x2 = x + w - 1
        cmd = "".join(grid[y + 1][x + 1 : x2]).strip()
        box = build_box(cmd, x, y, w, wire_map, grid)
        boxes.append(box)
    return boxes


def find_box_bounds(grid: List[List[str]]) -> List[Tuple[int, int, int]]:
    height = len(grid)
    width = len(grid[0])
    bounds = []
    for y in range(height - 2):
        for x in range(width - 1):
            if grid[y][x] != "*":
                continue
            # find matching top-right
            x2 = x + 1
            while x2 < width and grid[y][x2] != "*":
                if grid[y][x2] != "=":
                    break
                x2 += 1
            if x2 >= width or grid[y][x2] != "*":
                continue
            # verify middle row
            if grid[y + 1][x] != "!" or grid[y + 1][x2] != "!":
                continue
            # verify bottom row
            if grid[y + 2][x] != "*" or grid[y + 2][x2] != "*":
                continue
            for xx in range(x + 1, x2):
                if grid[y + 2][xx] != "=":
                    break
            else:
                bounds.append((x, y, x2 - x + 1))
    return bounds


def connector_positions(
    box_bounds: List[Tuple[int, int, int]],
    width: int,
    height: int,
) -> Tuple[set, set]:
    north_connectors = set()
    west_connectors = set()
    for x, y, w in box_bounds:
        if y - 1 >= 0:
            for xx in range(x + 1, x + w - 1):
                north_connectors.add((xx, y - 1))
        if x - 1 >= 0 and 0 <= y + 1 < height:
            west_connectors.add((x - 1, y + 1))
    return north_connectors, west_connectors


def build_box(cmd: str, x: int, y: int, w: int,
              wire_map: Dict[Tuple[int, int, str], int],
              grid: List[List[str]]) -> Box:
    # Inputs
    n_wire = None
    w_wire = None
    s_wire = None
    e_wire = None

    # North input: find a single 'v' above the top edge with a wire
    n_connectors = []
    for xx in range(x + 1, x + w - 1):
        if y - 1 >= 0 and grid[y - 1][xx] == "v":
            wire_id = wire_map.get((xx, y - 1, "N"))
            if wire_id is not None:
                n_connectors.append(wire_id)
    if len(n_connectors) > 1:
        raise ParseError("multiple north connectors on one box")
    if n_connectors:
        n_wire = n_connectors[0]

    # West input: find a '>' to the left of the middle row
    if x - 1 >= 0 and grid[y + 1][x - 1] == ">":
        w_wire = wire_map.get((x - 1, y + 1, "W"))

    # South output: any wire below bottom edge
    south_wires = set()
    if y + 3 < len(grid):
        for xx in range(x + 1, x + w - 1):
            wire_id = wire_map.get((xx, y + 3, "N"))
            if wire_id is not None:
                south_wires.add(wire_id)
    if len(south_wires) > 1:
        raise ParseError("multiple south outputs on one box")
    if south_wires:
        s_wire = next(iter(south_wires))

    # East output: any wire to the right of middle row
    if x + w < len(grid[0]):
        e_wire = wire_map.get((x + w, y + 1, "W"))

    return Box(
        name=cmd,
        cmd=parse_command(cmd),
        x=x,
        y=y,
        w=w,
        h=3,
        n_wire=n_wire,
        w_wire=w_wire,
        s_wire=s_wire,
        e_wire=e_wire,
    )


# ---------- Wire connectivity ----------


def build_wire_components(
    grid: List[List[str]],
    *,
    hash_crossover: bool = HASH_CROSSOVER_DEFAULT,
    north_connectors: Optional[set] = None,
    west_connectors: Optional[set] = None,
) -> Tuple[Dict[Tuple[int, int, str], int], int]:
    height = len(grid)
    width = len(grid[0])
    parent = {}
    node_opens: Dict[Tuple[int, int, str], set] = {}
    wire_map: Dict[Tuple[int, int, str], Tuple[int, int, str]] = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def is_border_cell(x, y):
        return x == 0 or y == 0 or x == width - 1 or y == height - 1

    def inside_dir(x, y):
        if y == 0:
            return "S"
        if y == height - 1:
            return "N"
        if x == 0:
            return "E"
        if x == width - 1:
            return "W"
        return None

    def add_node(x, y, kind, opens):
        if not opens:
            return
        key = (x, y, kind)
        parent[key] = key
        node_opens[key] = opens
        for d in opens:
            if (x, y, d) in wire_map and wire_map[(x, y, d)] != key:
                raise ParseError(f"ambiguous wire at {x},{y} for {d}")
            wire_map[(x, y, d)] = key

    # initialize nodes
    for y in range(height):
        for x in range(width):
            ch = grid[y][x]
            is_border = is_border_cell(x, y)
            inside = inside_dir(x, y)
            if ch == "|":
                opens = {"N", "S"}
                if is_border:
                    opens = {inside} if inside else set()
                add_node(x, y, "V", opens)
            elif ch == "-":
                opens = {"W", "E"}
                if is_border:
                    opens = {inside} if inside else set()
                add_node(x, y, "H", opens)
            elif ch == "+":
                add_node(x, y, "+", {"N", "S", "W", "E"})
            elif ch == "#":
                if hash_crossover:
                    # Crossover: vertical and horizontal do not connect.
                    add_node(x, y, "V", {"N", "S"})
                    add_node(x, y, "H", {"W", "E"})
                else:
                    add_node(x, y, "#", {"N", "S", "W", "E"})
            elif ch == "v":
                if (
                    north_connectors is not None
                    and (x, y) in north_connectors
                    and y - 1 >= 0
                    and grid[y - 1][x] in ("|", "+", "#", "=")
                ):
                    add_node(x, y, "v", {"N"})
            elif ch == ">":
                if (
                    west_connectors is not None
                    and (x, y) in west_connectors
                    and x - 1 >= 0
                    and grid[y][x - 1] in ("-", "+", "#", "!")
                ):
                    add_node(x, y, ">", {"W"})

    # union adjacent open sides
    for key, opens in node_opens.items():
        x, y, _ = key
        for d in opens:
            if d == "N":
                nx, ny, opp = x, y - 1, "S"
            elif d == "S":
                nx, ny, opp = x, y + 1, "N"
            elif d == "W":
                nx, ny, opp = x - 1, y, "E"
            else:
                nx, ny, opp = x + 1, y, "W"
            neighbor = wire_map.get((nx, ny, opp))
            if neighbor is not None:
                union(key, neighbor)

    # compress
    comp_ids = {}
    wire_id_map: Dict[Tuple[int, int, str], int] = {}
    next_id = 0
    for node in parent.keys():
        root = find(node)
        if root not in comp_ids:
            comp_ids[root] = next_id
            next_id += 1
        wire_id_map[node] = comp_ids[root]

    wire_id_by_dir: Dict[Tuple[int, int, str], int] = {}
    for (x, y, d), node in wire_map.items():
        wire_id_by_dir[(x, y, d)] = wire_id_map[node]
    return wire_id_by_dir, next_id


def validate_wire_rules(
    grid: List[List[str]],
    boxes: List[Box],
    wire_map: Dict[Tuple[int, int, str], int],
    wire_count: int,
    *,
    north_connectors: Optional[set] = None,
    west_connectors: Optional[set] = None,
) -> None:
    height = len(grid)
    width = len(grid[0])

    south_edges = set()
    east_edges = set()
    for box in boxes:
        for xx in range(box.x + 1, box.x + box.w - 1):
            south_edges.add((xx, box.y + 2))
        east_edges.add((box.x + box.w - 1, box.y + 1))

    def is_border_cell(x, y):
        return x == 0 or y == 0 or x == width - 1 or y == height - 1

    def inside_dir(x, y):
        if y == 0:
            return "S"
        if y == height - 1:
            return "N"
        if x == 0:
            return "E"
        if x == width - 1:
            return "W"
        return None

    open_dirs_map: Dict[Tuple[int, int], set] = {}
    for y in range(height):
        for x in range(width):
            ch = grid[y][x]
            dirs = set()
            if ch == "|":
                dirs = {"N", "S"}
                if is_border_cell(x, y):
                    inside = inside_dir(x, y)
                    dirs = {inside} if inside else set()
            elif ch == "-":
                dirs = {"W", "E"}
                if is_border_cell(x, y):
                    inside = inside_dir(x, y)
                    dirs = {inside} if inside else set()
            elif ch in ("+", "#"):
                dirs = {"N", "S", "W", "E"}
            elif ch == "v":
                if (
                    north_connectors is not None
                    and (x, y) in north_connectors
                    and y - 1 >= 0
                    and grid[y - 1][x] in ("|", "+", "#", "=")
                ):
                    dirs = {"N"}
            elif ch == ">":
                if (
                    west_connectors is not None
                    and (x, y) in west_connectors
                    and x - 1 >= 0
                    and grid[y][x - 1] in ("-", "+", "#", "!")
                ):
                    dirs = {"W"}
            if (x, y) in south_edges:
                dirs.add("S")
            if (x, y) in east_edges:
                dirs.add("E")
            open_dirs_map[(x, y)] = dirs

    def neighbor_open(x, y, d):
        if d == "N":
            nx, ny, opp = x, y - 1, "S"
        elif d == "S":
            nx, ny, opp = x, y + 1, "N"
        elif d == "W":
            nx, ny, opp = x - 1, y, "E"
        else:
            nx, ny, opp = x + 1, y, "W"
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            return False
        return opp in open_dirs_map[(nx, ny)]

    for y in range(height):
        for x in range(width):
            ch = grid[y][x]
            is_border = is_border_cell(x, y)
            inside = inside_dir(x, y)
            if is_border and ch in ("+", "#"):
                raise ParseError(f"wire char {ch} on boundary at {x},{y}")
            if ch == "|":
                if is_border:
                    if inside is None or not neighbor_open(x, y, inside):
                        raise ParseError(f"broken | on boundary at {x},{y}")
                else:
                    if not (neighbor_open(x, y, "N") and neighbor_open(x, y, "S")):
                        raise ParseError(f"broken | at {x},{y}")
            elif ch == "-":
                if is_border:
                    if inside is None or not neighbor_open(x, y, inside):
                        raise ParseError(f"broken - on boundary at {x},{y}")
                else:
                    if not (neighbor_open(x, y, "W") and neighbor_open(x, y, "E")):
                        raise ParseError(f"broken - at {x},{y}")
            elif ch == "#":
                if not all(neighbor_open(x, y, d) for d in ("N", "S", "W", "E")):
                    raise ParseError(f"broken # at {x},{y}")
            elif ch == "+":
                count = sum(1 for d in ("N", "S", "W", "E") if neighbor_open(x, y, d))
                if count != 2:
                    raise ParseError(f"broken + at {x},{y}")
            elif ch == "v":
                if (
                    north_connectors is not None
                    and (x, y) in north_connectors
                    and y - 1 >= 0
                    and grid[y - 1][x] in ("|", "+", "#", "=")
                ):
                    if not neighbor_open(x, y, "N"):
                        raise ParseError(f"dangling v at {x},{y}")
            elif ch == ">":
                if (
                    west_connectors is not None
                    and (x, y) in west_connectors
                    and x - 1 >= 0
                    and grid[y][x - 1] in ("-", "+", "#", "!")
                ):
                    if not neighbor_open(x, y, "W"):
                        raise ParseError(f"dangling > at {x},{y}")

    core_components = [False] * wire_count
    for y in range(height):
        for x in range(width):
            ch = grid[y][x]
            if ch == "|":
                dirs = {"N", "S"}
                if is_border_cell(x, y):
                    inside = inside_dir(x, y)
                    dirs = {inside} if inside else set()
            elif ch == "-":
                dirs = {"W", "E"}
                if is_border_cell(x, y):
                    inside = inside_dir(x, y)
                    dirs = {inside} if inside else set()
            elif ch in ("+", "#"):
                dirs = {"N", "S", "W", "E"}
            else:
                continue
            for d in dirs:
                wire_id = wire_map.get((x, y, d))
                if wire_id is not None:
                    core_components[wire_id] = True

    for wire_id in set(wire_map.values()):
        if not core_components[wire_id]:
            raise ParseError(f"wire without core segment id={wire_id}")


# ---------- Command parsing ----------


def tokenize(cmd: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(cmd):
        ch = cmd[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "(),[]":
            tokens.append(ch)
            i += 1
            continue
        if ch == '"':
            j = cmd.find('"', i + 1)
            if j == -1:
                raise ParseError("unterminated quote")
            tokens.append(cmd[i + 1 : j])
            i = j + 1
            continue
        # identifiers
        j = i
        while j < len(cmd) and cmd[j] not in "(),[]" and not cmd[j].isspace():
            j += 1
        tokens.append(cmd[i:j])
        i = j
    return tokens


def parse_command(cmd: str) -> Tuple:
    tokens = tokenize(cmd)
    if not tokens:
        raise ParseError("empty command")
    if tokens[0] == "send":
        return parse_send(tokens[1:])
    if tokens[0] == "case":
        return parse_case(tokens[1:])
    if tokens[0] == "split":
        exp, rest = parse_exp(tokens[1:])
        if rest:
            raise ParseError("extra tokens in split")
        return ("split", exp)
    if tokens[0] == "use":
        if len(tokens) < 2:
            raise ParseError("use missing name")
        return ("use", tokens[1])
    raise ParseError(f"unknown command {tokens[0]}")


def parse_send(tokens: List[str]) -> Tuple:
    if not tokens or tokens[0] != "[":
        raise ParseError("send missing [")
    idx = 1
    if tokens[idx] == "]":
        return ("send", [])
    outputs = []
    while True:
        if tokens[idx] != "(":
            raise ParseError("send expected (")
        idx += 1
        exp, idx = parse_exp_at(tokens, idx)
        if tokens[idx] != ",":
            raise ParseError("send expected ,")
        idx += 1
        outface = tokens[idx]
        if outface not in ("S", "E"):
            raise ParseError("send invalid outface")
        idx += 1
        if tokens[idx] != ")":
            raise ParseError("send expected )")
        idx += 1
        outputs.append((exp, outface))
        if tokens[idx] == "]":
            break
        if tokens[idx] != ",":
            raise ParseError("send expected , between outputs")
        idx += 1
    return ("send", outputs)


def parse_case(tokens: List[str]) -> Tuple:
    exp, rest = parse_exp(tokens)
    if not rest or rest[0] != "of":
        raise ParseError("case missing of")
    if rest[1] not in ("S", "E") or rest[2] != "," or rest[3] not in ("S", "E"):
        raise ParseError("case invalid outfaces")
    if rest[4:]:
        raise ParseError("case extra tokens")
    return ("case", exp, rest[1], rest[3])


def parse_exp_at(tokens: List[str], idx: int) -> Tuple[Tuple, int]:
    if idx >= len(tokens):
        raise ParseError("expected expression")
    tok = tokens[idx]
    if tok == "(":
        if tokens[idx + 1] == ")":
            return (("unit",), idx + 2)
        left, idx = parse_exp_at(tokens, idx + 1)
        if tokens[idx] != ",":
            raise ParseError("pair missing comma")
        right, idx = parse_exp_at(tokens, idx + 1)
        if tokens[idx] != ")":
            raise ParseError("pair missing )")
        return (("pair", left, right), idx + 1)
    if tok == "Inl":
        inner, idx = parse_exp_at(tokens, idx + 1)
        return (("inl", inner), idx)
    if tok == "Inr":
        inner, idx = parse_exp_at(tokens, idx + 1)
        return (("inr", inner), idx)
    if tok in ("N", "W"):
        return (("var", tok), idx + 1)
    raise ParseError(f"unknown expression token {tok}")


def parse_exp(tokens: List[str]) -> Tuple[Tuple, List[str]]:
    exp, idx = parse_exp_at(tokens, 0)
    return exp, tokens[idx:]


# ---------- Evaluation ----------


def eval_module(mod: Module, modules: Dict[str, Module], n_val=None, w_val=None):
    if (mod.n_input_wire is None) != (n_val is None):
        raise EvalError(f"module {mod.name} north input mismatch")
    if (mod.w_input_wire is None) != (w_val is None):
        raise EvalError(f"module {mod.name} west input mismatch")

    wire_values: List[Optional[Tuple]] = [None] * mod.wire_count

    def set_input(wire_id: int, value, label: str):
        if value is None:
            raise EvalError(f"missing {label} input for module {mod.name}")
        existing = wire_values[wire_id]
        if existing is None:
            wire_values[wire_id] = value
        elif existing != value:
            raise EvalError(f"conflicting {label} input for module {mod.name}")

    if mod.n_input_wire is not None:
        set_input(mod.n_input_wire, n_val, "north")
    if mod.w_input_wire is not None:
        set_input(mod.w_input_wire, w_val, "west")

    wire_users: List[List[int]] = [[] for _ in range(mod.wire_count)]
    missing = [0] * len(mod.boxes)
    for idx, box in enumerate(mod.boxes):
        inputs = []
        if box.n_wire is not None:
            inputs.append(box.n_wire)
        if box.w_wire is not None:
            inputs.append(box.w_wire)
        for wire_id in set(inputs):
            wire_users[wire_id].append(idx)
            if wire_values[wire_id] is None:
                missing[idx] += 1

    current_ready = [idx for idx, count in enumerate(missing) if count == 0]
    executed = [False] * len(mod.boxes)

    while current_ready:
        next_ready = []
        next_ready_set = set()
        for idx in current_ready:
            if executed[idx]:
                continue
            new_wires = run_box(mod.boxes[idx], wire_values, modules)
            executed[idx] = True
            for wire_id in new_wires:
                for dep_idx in wire_users[wire_id]:
                    if executed[dep_idx]:
                        continue
                    if missing[dep_idx] > 0:
                        missing[dep_idx] -= 1
                        if missing[dep_idx] == 0 and dep_idx not in next_ready_set:
                            next_ready_set.add(dep_idx)
                            next_ready.append(dep_idx)
        current_ready = next_ready

    outputs = [wire_values[w] for w in mod.output_wires if wire_values[w] is not None]
    if len(outputs) != 1:
        raise EvalError(f"module {mod.name} output count {len(outputs)}")
    return outputs[0]


def run_box(box: Box, wire_values: List[Optional[Tuple]], modules: Dict[str, Module]) -> List[int]:
    cmd = box.cmd
    n_val = wire_values[box.n_wire] if box.n_wire is not None else None
    w_val = wire_values[box.w_wire] if box.w_wire is not None else None
    newly_set = set()

    def eval_exp(exp):
        kind = exp[0]
        if kind == "unit":
            return UNIT
        if kind == "pair":
            return pair(eval_exp(exp[1]), eval_exp(exp[2]))
        if kind == "inl":
            return inl(eval_exp(exp[1]))
        if kind == "inr":
            return inr(eval_exp(exp[1]))
        if kind == "var":
            if exp[1] == "N":
                if n_val is None:
                    raise EvalError("N not available")
                return n_val
            if exp[1] == "W":
                if w_val is None:
                    raise EvalError("W not available")
                return w_val
        raise EvalError("bad exp")

    def send_value(outface: str, value):
        if outface == "S":
            wire_id = box.s_wire
        else:
            wire_id = box.e_wire
        if wire_id is None:
            raise EvalError(f"send to missing outface {outface}")
        existing = wire_values[wire_id]
        if existing is None:
            wire_values[wire_id] = value
            newly_set.add(wire_id)
        else:
            if existing != value:
                raise EvalError("wire conflict")

    if cmd[0] == "send":
        evaluated = [(eval_exp(exp), outface) for exp, outface in cmd[1]]
        outfaces = [outface for _, outface in evaluated]
        if len(outfaces) != len(set(outfaces)):
            raise EvalError("send outfaces must be distinct")
        for value, outface in evaluated:
            send_value(outface, value)
        return sorted(newly_set)
    if cmd[0] == "split":
        val = eval_exp(cmd[1])
        if val[0] != "pair":
            raise EvalError("split on non-pair")
        send_value("S", val[1])
        send_value("E", val[2])
        return sorted(newly_set)
    if cmd[0] == "case":
        val = eval_exp(cmd[1])
        if val[0] == "inl":
            send_value(cmd[2], val[1])
            return sorted(newly_set)
        if val[0] == "inr":
            send_value(cmd[3], val[1])
            return sorted(newly_set)
        raise EvalError("case on non-sum")
    if cmd[0] == "use":
        mod = modules.get(cmd[1])
        if mod is None:
            raise EvalError(f"unknown module {cmd[1]}")
        if (mod.n_input_wire is None) != (box.n_wire is None):
            raise EvalError("use north input mismatch")
        if (mod.w_input_wire is None) != (box.w_wire is None):
            raise EvalError("use west input mismatch")
        if mod.n_input_wire is not None and n_val is None:
            raise EvalError("use missing N input")
        if mod.w_input_wire is not None and w_val is None:
            raise EvalError("use missing W input")
        out = eval_module(
            mod,
            modules,
            n_val if mod.n_input_wire is not None else None,
            w_val if mod.w_input_wire is not None else None,
        )
        send_value("E", out)
        return sorted(newly_set)
    raise EvalError("unknown command")


# ---------- Helpers ----------


def unary(n: int):
    val = inr(UNIT)
    for _ in range(n):
        val = inl(val)
    return val


def value_to_str(val) -> str:
    if val == UNIT:
        return "()"
    if val[0] == "inl":
        return f"Inl {value_to_str(val[1])}"
    if val[0] == "inr":
        return f"Inr {value_to_str(val[1])}"
    if val[0] == "pair":
        return f"({value_to_str(val[1])}, {value_to_str(val[2])})"
    return "?"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("program")
    parser.add_argument("module")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--w", type=int, default=None)
    parser.add_argument(
        "--hash-junction",
        action="store_true",
        help="treat # as a 4-way junction (default is crossover)",
    )
    parser.add_argument(
        "--strict-wires",
        action="store_true",
        help="validate wire connectivity rules",
    )
    args = parser.parse_args()

    text = open(args.program, "r", encoding="utf-8").read()
    modules = parse_modules(
        text,
        hash_crossover=not args.hash_junction,
        strict_wires=args.strict_wires,
    )
    mod = modules.get(args.module)
    if mod is None:
        raise SystemExit(f"module {args.module} not found")

    n_val = unary(args.n) if args.n is not None else None
    w_val = unary(args.w) if args.w is not None else None
    out = eval_module(mod, modules, n_val, w_val)
    print(value_to_str(out))


if __name__ == "__main__":
    main()
