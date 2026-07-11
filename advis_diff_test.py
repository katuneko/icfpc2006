#!/usr/bin/env python3
"""Random differential and semantic checks for compact O'Cult advice."""

import argparse
import random

from occult import App, Atom, parse_rules, run


def app(head, *args):
    term = Atom(head)
    for arg in args:
        term = App(term, arg)
    return term


def numeral(value):
    term = Atom("Z")
    for _ in range(value):
        term = app("S", term)
    return term


def random_arith(rng, depth):
    if depth == 0 or rng.random() < 0.35:
        return numeral(rng.randrange(7))
    operator = rng.choice(("Add", "Mult"))
    return app(operator, random_arith(rng, depth - 1),
               random_arith(rng, depth - 1))


def arith_value(term):
    if term == Atom("Z"):
        return 0
    if isinstance(term, App) and term.left == Atom("S"):
        return 1 + arith_value(term.right)
    raise ValueError("not a numeral")


def random_xml(rng, depth):
    if depth == 0 or rng.random() < 0.25:
        return Atom(rng.choice(("A", "B")))
    if rng.random() < 0.5:
        return app("Seq", random_xml(rng, depth - 1),
                   random_xml(rng, depth - 1))
    return app("Tag", Atom(rng.choice(("Bold", "Emph", "Maj"))),
               random_xml(rng, depth - 1))


def check(candidate, baseline, kind, seed, cases, depth):
    rng = random.Random(seed)
    candidate_rules = parse_rules(open(candidate, encoding="utf-8").read())
    baseline_rules = parse_rules(open(baseline, encoding="utf-8").read())
    for number in range(1, cases + 1):
        source = (random_arith(rng, depth) if kind == "arith"
                  else random_xml(rng, depth))
        initial = app("Compute" if kind == "arith" else "SNF", source)
        got, got_steps = run(candidate_rules, initial)
        expected, expected_steps = run(baseline_rules, initial)
        if got != expected:
            raise SystemExit(
                f"case {number} differs (candidate steps={got_steps}, "
                f"baseline steps={expected_steps})")
        if kind == "arith":
            arith_value(got)
    print(f"passed {cases} {kind} cases (seed={seed}, depth={depth})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("arith", "xml"))
    parser.add_argument("candidate")
    parser.add_argument("baseline")
    parser.add_argument("--seed", type=int, default=2006)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()
    check(args.candidate, args.baseline, args.kind, args.seed, args.cases,
          args.depth)


if __name__ == "__main__":
    main()
