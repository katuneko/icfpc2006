#!/usr/bin/env python3
"""Repack unchanged 2D modules at explicit non-overlapping coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

from two_d import parse_modules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("layout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    modules = parse_modules(args.source.read_text(), strict_wires=True)
    placements = []
    width = height = 0
    for line in args.layout.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] == "SIZE":
            continue
        name, x_text, y_text, w_text, h_text = fields
        x, y, w, h = map(int, (x_text, y_text, w_text, h_text))
        module = modules[name]
        assert len(module.grid[0]) == w and len(module.grid) == h
        placements.append((name, x, y, module.grid))
        width = max(width, x + w)
        height = max(height, y + h)

    canvas = [[" "] * width for _ in range(height)]
    occupied = [[False] * width for _ in range(height)]
    for name, x, y, grid in placements:
        for dy, row in enumerate(grid):
            for dx, char in enumerate(row):
                if occupied[y + dy][x + dx]:
                    raise ValueError(f"module overlap at {x + dx},{y + dy}: {name}")
                occupied[y + dy][x + dx] = True
                canvas[y + dy][x + dx] = char

    args.output.write_text("\n".join("".join(row).rstrip() for row in canvas) + "\n")


if __name__ == "__main__":
    main()
