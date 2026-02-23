#!/usr/bin/env python3
from __future__ import annotations

"""
Generate a 2D program implementing one O'Cult "least-heeded" rewrite step.

Target: pass `verify ocult` on UMIX by providing module `step` that takes
W=(advice, term) and outputs Option(term):
  - Inl ()            if no rule applies
  - Inr [[nextTerm]]  if one rewrite step is possible

This follows the reference semantics in `ocult_ref.py` / `hadproblem.md`.
"""

from dataclasses import dataclass
from pathlib import Path


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
        # Controlled upgrades (explicitly placed later).
        if ch == "#" and prev in ("|", "-", "#"):
            self.g[y][x] = "#"
            return
        if ch == "+" and prev in ("|", "-", "+"):
            self.g[y][x] = "+"
            return
        if prev == "#" and ch in ("|", "-"):
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
    west_input_body_line: int
    east_output_body_lines: set[int]

    def render(self) -> list[str]:
        w = self.interior_w
        body_lines: list[str] = []
        body_lines.append(self.name.ljust(w))
        body_lines.append(" " * w)
        body_lines.extend(self.canvas.render())

        top = "," + ("." * w) + ","
        out: list[str] = [top]
        for i, core in enumerate(body_lines):
            left = "-" if i == self.west_input_body_line else ":"
            right = "-" if i in self.east_output_body_lines else ":"
            out.append(left + core + right)
        out.append(top)
        return out


def build_natadd() -> ModuleSpec:
    # W = (a,b) nats -> a+b
    w = 90
    h = 16
    c = Canvas(w, h)

    split = Box(1, 0, "split W", w_in=True)
    c.add_box(split)

    # Route b onto a vertical bus at x_b
    x_b = 18
    c.hline(split.x + split.w, x_b, split.cmd_row, "-")
    c.put(x_b, split.cmd_row, "+")
    c.vline(x_b, split.cmd_row + 1, h - 1, "|")

    # case on a (from split S)
    casea = Box(12, 5, "case W of S,E", w_in=True)
    c.add_box(casea)
    x_a = split.x + 3
    c.put(x_a, split.y + 3, "|")
    c.vline(x_a, split.y + 3, casea.cmd_row, "|")
    c.put(x_a, casea.cmd_row, "+")
    c.hline(x_a + 1, casea.x - 2, casea.cmd_row, "-")

    # Base case: a=0 -> return b (gate with W from casea S (unit))
    sendb = Box(28, 5, "send[(N,E)]", w_in=True, n_in=True, n_conn_x=x_b)
    c.add_box(sendb)
    # control from casea S output (unit) down to sendb W
    x_ctrl = casea.x + 3
    y_ctrl = casea.y + 3
    c.put(x_ctrl, y_ctrl, "|")
    c.vline(x_ctrl, y_ctrl, sendb.cmd_row, "|")
    c.put(x_ctrl, sendb.cmd_row, "+")
    c.hline(x_ctrl + 1, sendb.x - 2, sendb.cmd_row, "-")
    # sendb -> module output (right edge)
    c.hline(sendb.x + sendb.w, w - 1, sendb.cmd_row, "-")

    # Succ case: a=Inr a1 -> recurse on (a1,b)
    mkpair = Box(28, 10, "send[((W,N),E)]", w_in=True, n_in=True, n_conn_x=x_b)
    c.add_box(mkpair)
    # a1 from casea E output to mkpair W
    c.hline(casea.x + casea.w, mkpair.x - 2, casea.cmd_row, "-")
    # mkpair -> use natadd
    use = Box(55, 10, "use natadd", w_in=True)
    c.add_box(use)
    c.hline(mkpair.x + mkpair.w, use.x - 2, mkpair.cmd_row, "-")

    wrap = Box(72, 10, "send[(Inr W,E)]", w_in=True)
    c.add_box(wrap)
    c.hline(use.x + use.w, wrap.x - 2, use.cmd_row, "-")
    c.hline(wrap.x + wrap.w, w - 1, wrap.cmd_row, "-")

    west_input_body_line = 2 + split.cmd_row
    east_output_body_lines = {2 + sendb.cmd_row, 2 + wrap.cmd_row}
    return ModuleSpec(
        name="natadd",
        interior_w=w,
        canvas=c,
        west_input_body_line=west_input_body_line,
        east_output_body_lines=east_output_body_lines,
    )


def main() -> None:
    modules = [
        build_natadd(),
    ]
    lines: list[str] = []
    for m in modules:
        lines.extend(m.render())
        lines.append("")
    Path("ocult_full.2d").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
