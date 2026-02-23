#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Box:
    x: int
    y: int
    cmd: str
    w_in: bool = False
    n_in: bool = False
    n_conn_x: int | None = None

    @property
    def w(self) -> int:
        return len(self.cmd) + 2

    @property
    def h(self) -> int:
        return 3

    @property
    def x2(self) -> int:
        return self.x + self.w - 1

    @property
    def y2(self) -> int:
        return self.y + self.h - 1

    @property
    def cmd_row(self) -> int:
        return self.y + 1


class Canvas:
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.g = [[" " for _ in range(w)] for _ in range(h)]

    def put(self, x: int, y: int, ch: str) -> None:
        if not (0 <= x < self.w and 0 <= y < self.h):
            raise ValueError(f"put oob {x},{y}")
        prev = self.g[y][x]
        if prev == " " or prev == ch:
            self.g[y][x] = ch
            return

        # Allow controlled "upgrades" when routing wires:
        # - `#` is a crossover (both directions, no connection), so it can replace
        #   a plain vertical or horizontal wire at a planned crossing.
        # - `+` is a junction, so it can replace a plain vertical or horizontal
        #   wire when we intentionally join paths.
        # - If a crossover already exists, don't downgrade it back to `|`/`-`.
        if ch == "#" and prev in ("|", "-", "#"):
            self.g[y][x] = "#"
            return
        if ch == "+" and prev in ("|", "-", "+"):
            self.g[y][x] = "+"
            return
        if prev == "#" and ch == "+":
            # Explicit junction can override a previously-placed crossover.
            self.g[y][x] = "+"
            return
        if prev == "#" and ch in ("|", "-"):
            return
        if prev == "|" and ch == "-":
            # Crossing detected; default to a crossover and allow later override.
            self.g[y][x] = "#"
            return
        if prev == "-" and ch == "|":
            self.g[y][x] = "#"
            return

        raise ValueError(f"clobber at {x},{y}: {prev!r} -> {ch!r}")

    def text(self, x: int, y: int, s: str) -> None:
        for i, ch in enumerate(s):
            self.put(x + i, y, ch)

    def hline(self, x1: int, x2: int, y: int, ch: str = "-") -> None:
        if x2 < x1:
            x1, x2 = x2, x1
        for x in range(x1, x2 + 1):
            self.put(x, y, ch)

    def vline(self, x: int, y1: int, y2: int, ch: str = "|") -> None:
        if y2 < y1:
            y1, y2 = y2, y1
        for y in range(y1, y2 + 1):
            self.put(x, y, ch)

    def add_box(self, b: Box) -> None:
        top = "*" + ("=" * (b.w - 2)) + "*"
        mid = "!" + b.cmd + "!"
        bot = top
        self.text(b.x, b.y + 0, top)
        self.text(b.x, b.y + 1, mid)
        self.text(b.x, b.y + 2, bot)
        if b.w_in:
            self.put(b.x - 1, b.y + 1, ">")
        if b.n_in:
            if b.n_conn_x is None:
                nxc = b.x + 1
            else:
                nxc = b.n_conn_x
            if not (b.x + 1 <= nxc <= b.x + b.w - 2):
                raise ValueError(f"bad n_conn_x {nxc} for box {b.cmd}")
            self.put(nxc, b.y - 1, "v")

    def render(self) -> list[str]:
        return ["".join(row) for row in self.g]


@dataclass
class ModuleSpec:
    name: str
    interior_w: int
    canvas: Canvas
    # body_line_idx (0=name row, 1=blank, 2+=canvas) where west input is placed
    west_input_body_line: int
    # body_line_idx where east output ports exist
    east_output_body_lines: set[int]

    def render(self) -> list[str]:
        w = self.interior_w
        body_lines: list[str] = []

        # name row
        name_row = self.name.ljust(w)
        body_lines.append(name_row)
        # blank row
        body_lines.append(" " * w)
        # canvas rows
        body_lines.extend(self.canvas.render())

        # top/bottom borders
        top = "," + ("." * w) + ","
        bot = top

        out: list[str] = [top]
        for i, core in enumerate(body_lines):
            left = "-" if i == self.west_input_body_line else ":"
            right = "-" if i in self.east_output_body_lines else ":"
            out.append(left + core + right)
        out.append(bot)
        return out


# -------------------- Build modules --------------------


def build_strip_c0() -> ModuleSpec:
    w = 70
    h = 23
    c = Canvas(w, h)

    x_case = 1
    x_send = 30
    x_send_far = 50
    x_pipe = 8
    x_arg = 28
    x_send_conn = x_send - 1

    # Boxes
    case_t = Box(x_case, 0, "case W of S,E", w_in=True)
    send_none1 = Box(x_send, 0, "send[(Inl (),E)]", w_in=True)
    split = Box(x_case, 5, "split N", n_in=True, n_conn_x=x_pipe)
    case_l = Box(x_case, 10, "case N of E,S", n_in=True, n_conn_x=x_pipe)
    send_none2 = Box(x_send, 10, "send[(Inl (),E)]", w_in=True)
    case_nat = Box(x_case, 15, "case N of S,E", n_in=True, n_conn_x=x_pipe)
    # Keep this one out of the way so the success->send_some north wire isn't
    # directly under its (always-open) south output edge.
    send_none3 = Box(x_send_far, 15, "send[(Inl (),E)]", w_in=True)
    send_some = Box(x_send, 20, "send[(Inr W,E)]", w_in=True, n_in=True, n_conn_x=38)

    for b in [case_t, send_none1, split, case_l, send_none2, case_nat, send_none3, send_some]:
        c.add_box(b)

    # case_t E -> send_none1 W
    # Connect directly to send_none1's west connector (one cell before its '>' marker).
    c.hline(case_t.x + case_t.w, x_send - 2, case_t.cmd_row, "-")

    # send_none1 -> output
    c.hline(send_none1.x + send_none1.w, w - 1, send_none1.cmd_row, "-")

    # pipe wire: case_t S (y=3) into split v (y=4)
    c.put(x_pipe, 3, "|")

    # split right arg wire: from split E out (x=10,y=6) to x_arg corner
    c.hline(split.x + split.w, x_arg - 1, split.cmd_row, "-")
    c.put(x_arg, split.cmd_row, "+")
    c.vline(x_arg, split.cmd_row + 1, send_some.cmd_row - 1, "|")

    # split left -> case_l
    c.put(x_pipe, 8, "|")

    # case_l mismatch -> send_none2 (with crossover at arg)
    c.hline(case_l.x + case_l.w, x_arg - 1, case_l.cmd_row, "-")
    c.put(x_arg, case_l.cmd_row, "#")

    # send_none2 -> output
    c.hline(send_none2.x + send_none2.w, w - 1, send_none2.cmd_row, "-")

    # case_l nat down -> case_nat
    c.put(x_pipe, 13, "|")

    # case_nat mismatch -> send_none3 (crossover at arg)
    c.hline(case_nat.x + case_nat.w, x_arg - 1, case_nat.cmd_row, "-")
    c.put(x_arg, case_nat.cmd_row, "#")
    c.hline(x_arg + 1, x_send_far - 2, case_nat.cmd_row, "-")

    # send_none3 -> output
    c.hline(send_none3.x + send_none3.w, w - 1, send_none3.cmd_row, "-")

    # case_nat success wire: start at (x_pipe,18) corner to horizontal, crossover at arg
    c.put(x_pipe, 18, "+")
    # Route to send_some's north connector at x=38 via a junction at (38,18).
    c.hline(x_pipe + 1, 38, 18, "-")
    c.put(38, 18, "+")
    c.put(x_arg, 18, "#")

    # arg vertical crosses case_l/case_nat already, ensure vertical continuity
    c.put(x_arg, case_l.cmd_row, "#")
    c.put(x_arg, case_nat.cmd_row, "#")
    c.put(x_arg, 18, "#")

    # turn arg into send_some west connector
    c.put(x_arg, send_some.cmd_row, "+")

    # send_some -> output
    c.hline(send_some.x + send_some.w, w - 1, send_some.cmd_row, "-")

    # crossover where success wire crosses arg already set

    west_input_body_line = 2 + case_t.cmd_row
    east_output_body_lines = {
        2 + send_none1.cmd_row,
        2 + send_none2.cmd_row,
        2 + send_none3.cmd_row,
        2 + send_some.cmd_row,
    }
    return ModuleSpec(
        name="stripc0",
        interior_w=w,
        canvas=c,
        west_input_body_line=west_input_body_line,
        east_output_body_lines=east_output_body_lines,
    )


def build_strip_c2() -> ModuleSpec:
    w = 70
    h = 33
    c = Canvas(w, h)

    x_case = 1
    x_send = 30
    x_send_far = 50
    x_pipe = 8
    x_arg = 28

    case_t = Box(x_case, 0, "case W of S,E", w_in=True)
    send_none1 = Box(x_send, 0, "send[(Inl (),E)]", w_in=True)
    split = Box(x_case, 5, "split N", n_in=True, n_conn_x=x_pipe)
    case_l = Box(x_case, 10, "case N of E,S", n_in=True, n_conn_x=x_pipe)
    send_none2 = Box(x_send, 10, "send[(Inl (),E)]", w_in=True)

    case_n0 = Box(x_case, 15, "case N of E,S", n_in=True, n_conn_x=x_pipe)
    send_none3 = Box(x_send, 15, "send[(Inl (),E)]", w_in=True)

    case_n1 = Box(x_case, 20, "case N of E,S", n_in=True, n_conn_x=x_pipe)
    send_none4 = Box(x_send, 20, "send[(Inl (),E)]", w_in=True)

    case_n2 = Box(x_case, 25, "case N of S,E", n_in=True, n_conn_x=x_pipe)
    send_none5 = Box(x_send_far, 25, "send[(Inl (),E)]", w_in=True)

    send_some = Box(x_send, 30, "send[(Inr W,E)]", w_in=True, n_in=True, n_conn_x=38)

    for b in [
        case_t,
        send_none1,
        split,
        case_l,
        send_none2,
        case_n0,
        send_none3,
        case_n1,
        send_none4,
        case_n2,
        send_none5,
        send_some,
    ]:
        c.add_box(b)

    # case_t E -> send_none1
    c.hline(case_t.x + case_t.w, x_send - 2, case_t.cmd_row, "-")
    c.hline(send_none1.x + send_none1.w, w - 1, send_none1.cmd_row, "-")

    # pipe case_t S -> split
    c.put(x_pipe, 3, "|")

    # split right arg wire
    c.hline(split.x + split.w, x_arg - 1, split.cmd_row, "-")
    c.put(x_arg, split.cmd_row, "+")
    c.vline(x_arg, split.cmd_row + 1, send_some.cmd_row - 1, "|")

    # split left -> case_l
    c.put(x_pipe, 8, "|")

    # case_l mismatch -> send_none2 with crossover at arg
    c.hline(case_l.x + case_l.w, x_arg - 1, case_l.cmd_row, "-")
    c.put(x_arg, case_l.cmd_row, "#")
    c.hline(send_none2.x + send_none2.w, w - 1, send_none2.cmd_row, "-")

    # nat down from case_l -> case_n0
    c.put(x_pipe, 13, "|")

    # case_n0 mismatch (0) -> send_none3, continue down
    c.hline(case_n0.x + case_n0.w, x_arg - 1, case_n0.cmd_row, "-")
    c.put(x_arg, case_n0.cmd_row, "#")
    c.hline(send_none3.x + send_none3.w, w - 1, send_none3.cmd_row, "-")
    c.put(x_pipe, 18, "|")

    # case_n1 mismatch (1) -> send_none4
    c.hline(case_n1.x + case_n1.w, x_arg - 1, case_n1.cmd_row, "-")
    c.put(x_arg, case_n1.cmd_row, "#")
    c.hline(send_none4.x + send_none4.w, w - 1, send_none4.cmd_row, "-")
    c.put(x_pipe, 23, "|")

    # case_n2 mismatch (>=3) -> send_none5
    c.hline(case_n2.x + case_n2.w, x_arg - 1, case_n2.cmd_row, "-")
    c.put(x_arg, case_n2.cmd_row, "#")
    c.hline(x_arg + 1, x_send_far - 2, case_n2.cmd_row, "-")
    c.hline(send_none5.x + send_none5.w, w - 1, send_none5.cmd_row, "-")

    # success wire from case_n2 S at y=28
    c.put(x_pipe, 28, "+")
    c.hline(x_pipe + 1, 38, 28, "-")
    c.put(38, 28, "+")
    c.put(x_arg, 28, "#")

    # connect arg to send_some
    c.put(x_arg, send_some.cmd_row, "+")

    # send_some -> output
    c.hline(send_some.x + send_some.w, w - 1, send_some.cmd_row, "-")

    # ensure crossovers for arg vertical where used
    for y in [case_l.cmd_row, case_n0.cmd_row, case_n1.cmd_row, case_n2.cmd_row, 28]:
        c.put(x_arg, y, "#")

    west_input_body_line = 2 + case_t.cmd_row
    east_output_body_lines = {
        2 + send_none1.cmd_row,
        2 + send_none2.cmd_row,
        2 + send_none3.cmd_row,
        2 + send_none4.cmd_row,
        2 + send_none5.cmd_row,
        2 + send_some.cmd_row,
    }
    return ModuleSpec(
        name="stripc2",
        interior_w=w,
        canvas=c,
        west_input_body_line=west_input_body_line,
        east_output_body_lines=east_output_body_lines,
    )


def build_out(name: str, const: str) -> ModuleSpec:
    cmd = f"send[({const},E)]"
    bw = len(cmd) + 2
    w = bw + 2  # +1 for output wire and some slack
    h = 3
    c = Canvas(w, h)

    b = Box(1, 0, cmd, w_in=True)
    c.add_box(b)

    # output wire at the box east adjacency is at x = 1+bw
    out_x = b.x + b.w
    c.put(out_x, b.cmd_row, "-")

    west_input_body_line = 2 + b.cmd_row
    east_output_body_lines = {2 + b.cmd_row}

    return ModuleSpec(
        name=name,
        interior_w=w,
        canvas=c,
        west_input_body_line=west_input_body_line,
        east_output_body_lines=east_output_body_lines,
    )


def build_step() -> ModuleSpec:
    w = 180
    h = 26
    c = Canvas(w, h)

    x = 1
    y = 0

    split = Box(x, y, "split W", w_in=True)
    c.add_box(split)

    # sink advice from split S output (wire below split)
    # Place it to the right of the split so the vertical drop doesn't run through the sink box.
    sink = Box(split.x + split.w + 2, y + 3, "send[]", w_in=True)
    c.add_box(sink)
    # connect split S output wire at (x+? , y+3) to sink west input
    # Use a vertical wire at x_pipe_s = x+3 (inside split), then route to sink.
    x_adv = x + 3
    c.put(x_adv, y + 3, "|")
    c.vline(x_adv, y + 3, sink.cmd_row, "|")
    # corner into sink west connector
    c.put(x_adv, sink.cmd_row, "+")
    c.hline(x_adv + 1, sink.x - 2, sink.cmd_row, "-")

    # main chain on split E output (term)
    # wire from split E adjacency to use strip_c2
    x_use0 = split.x + split.w + 4
    use0 = Box(x_use0, y, "use stripc2", w_in=True)
    c.add_box(use0)
    c.hline(split.x + split.w, use0.x - 2, split.cmd_row, "-")

    case0 = Box(use0.x + use0.w + 4, y, "case W of S,E", w_in=True)
    c.add_box(case0)
    c.hline(use0.x + use0.w, case0.x - 2, use0.cmd_row, "-")

    use1 = Box(case0.x + case0.w + 4, y, "use stripc0", w_in=True)
    c.add_box(use1)
    c.hline(case0.x + case0.w, use1.x - 2, case0.cmd_row, "-")

    case1 = Box(use1.x + use1.w + 4, y, "case W of S,E", w_in=True)
    c.add_box(case1)
    c.hline(use1.x + use1.w, case1.x - 2, use1.cmd_row, "-")

    use2 = Box(case1.x + case1.w + 4, y, "use stripc0", w_in=True)
    c.add_box(use2)
    c.hline(case1.x + case1.w, use2.x - 2, case1.cmd_row, "-")

    case2 = Box(use2.x + use2.w + 4, y, "case W of S,E", w_in=True)
    c.add_box(case2)
    c.hline(use2.x + use2.w, case2.x - 2, use2.cmd_row, "-")

    use3 = Box(case2.x + case2.w + 4, y, "use stripc0", w_in=True)
    c.add_box(use3)
    c.hline(case2.x + case2.w, use3.x - 2, case2.cmd_row, "-")

    case3 = Box(use3.x + use3.w + 4, y, "case W of S,E", w_in=True)
    c.add_box(case3)
    c.hline(use3.x + use3.w, case3.x - 2, use3.cmd_row, "-")

    # outputs: use out_* modules, one per branch
    # Place out use boxes below and connect from case S/E outputs.
    outs = [
        ("outnone", case0, "S"),
        ("out1", case1, "S"),
        ("out2", case2, "S"),
        ("out3", case3, "S"),
        ("out4", case3, "E"),
    ]
    out_boxes: list[Box] = []
    for i, (name, src, face) in enumerate(outs):
        ob_y = 6 + i * 4
        cmd = f"use {name}"
        # Align the out box output to the module's east border, and stack
        # vertically so output wires don't run through other boxes/connectors.
        ob_w = len(cmd) + 2
        ob_x = (w - 1) - ob_w
        ob = Box(ob_x, ob_y, cmd, w_in=True)
        out_boxes.append(ob)
        c.add_box(ob)
        # connect from src outface
        if face == "S":
            # src S wire below src bottom at y=src.y+3, pick x=src.x+3
            sx = src.x + 3
            sy = src.y + 3
            c.put(sx, sy, "|")
            c.vline(sx, sy, ob.cmd_row, "|")
            c.put(sx, ob.cmd_row, "+")
            c.hline(sx + 1, ob.x - 2, ob.cmd_row, "-")
        else:
            # E output from src command row at x=src.x+src.w, then route down.
            sx = src.x + src.w
            sy = src.cmd_row
            c.put(sx, sy, "+")
            c.vline(sx, sy + 1, ob.cmd_row, "|")
            c.put(sx, ob.cmd_row, "+")
            c.hline(sx + 1, ob.x - 2, ob.cmd_row, "-")

        # output to module border
        c.put(ob.x + ob.w, ob.cmd_row, "-")

    west_input_body_line = 2 + split.cmd_row
    east_output_body_lines = {2 + ob.cmd_row for ob in out_boxes}

    return ModuleSpec(
        name="step",
        interior_w=w,
        canvas=c,
        west_input_body_line=west_input_body_line,
        east_output_body_lines=east_output_body_lines,
    )


def main() -> None:
    # Constants (Option(term))
    opt1 = "Inr Inl (Inr Inr Inr Inl (),Inl (Inr Inl (),Inl (Inl (Inr Inr Inr Inr Inl (),Inl (Inr Inl (),Inr Inr Inl ())),Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inr Inr Inl ()))))))"
    opt2 = "Inr Inl (Inr Inr Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inl (Inr Inr Inr Inr Inl (),Inr Inr Inl ()),Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inr Inr Inl ())))))))"
    opt3 = "Inr Inl (Inr Inr Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inr Inr Inl ()))))))"
    opt4 = "Inr Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inr Inr Inl ())))))"

    modules = [
        build_strip_c0(),
        build_strip_c2(),
        build_out("outnone", "Inl ()"),
        build_out("out1", opt1),
        build_out("out2", opt2),
        build_out("out3", opt3),
        build_out("out4", opt4),
        build_step(),
    ]

    lines: list[str] = []
    for m in modules:
        lines.extend(m.render())
        lines.append("")

    Path("ocult_step_map.2d").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
