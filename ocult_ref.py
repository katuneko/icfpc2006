#!/usr/bin/env python3
"""Reference implementation for the 2D 'ocult' verify test.

This script evaluates one O'Cult advice-application *step* on the 2D encoding
described in Ohmega's mail (see problems/problems.md).

The 2D verify 'ocult' test expects:
  - input:  W = (advice, term)
  - output: Inl ()           if no rule applies
            Inr [[nextTerm]] if a rule applies once (per O'Cult semantics)

Where [[nextTerm]] is encoded as a *term*:
  App(e1,e2) = Inl ([[e1]], [[e2]])
  Const s    = Inr [[s]]
and names (numbers) are:
  zero = Inl ()
  s(n) = Inr [[n]]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional


# ---------- 2D value AST ----------


@dataclass(frozen=True)
class Unit:
    pass


@dataclass(frozen=True)
class Pair:
    a: "Val"
    b: "Val"


@dataclass(frozen=True)
class Inl:
    v: "Val"


@dataclass(frozen=True)
class Inr:
    v: "Val"


Val = Unit | Pair | Inl | Inr


class ParseError(RuntimeError):
    pass


class Parser:
    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def skip(self) -> None:
        s = self.s
        i = self.i
        n = len(s)
        while i < n and s[i].isspace():
            i += 1
        self.i = i

    def peek(self, lit: str) -> bool:
        return self.s.startswith(lit, self.i)

    def expect(self, lit: str) -> None:
        if not self.peek(lit):
            raise ParseError(f"expected {lit!r} at {self.i}: {self.s[self.i:self.i+40]!r}")
        self.i += len(lit)

    def parse_val(self) -> Val:
        self.skip()
        if self.peek("()"):
            self.expect("()")
            return Unit()
        if self.peek("("):
            self.expect("(")
            a = self.parse_val()
            self.skip()
            self.expect(",")
            b = self.parse_val()
            self.skip()
            self.expect(")")
            return Pair(a, b)
        if self.peek("Inl"):
            self.expect("Inl")
            return Inl(self.parse_val())
        if self.peek("Inr"):
            self.expect("Inr")
            return Inr(self.parse_val())
        raise ParseError(f"unexpected at {self.i}: {self.s[self.i:self.i+40]!r}")


def parse_val(s: str) -> Val:
    p = Parser(s)
    v = p.parse_val()
    p.skip()
    if p.i != len(p.s):
        raise ParseError(f"trailing input at {p.i}: {p.s[p.i:p.i+40]!r}")
    return v


def val_to_str(v: Val) -> str:
    if isinstance(v, Unit):
        return "()"
    if isinstance(v, Pair):
        return f"({val_to_str(v.a)},{val_to_str(v.b)})"
    if isinstance(v, Inl):
        return f"Inl {val_to_str(v.v)}"
    if isinstance(v, Inr):
        return f"Inr {val_to_str(v.v)}"
    raise TypeError(v)


# ---------- O'Cult decoded structures ----------


def dec_nat(v: Val) -> int:
    if isinstance(v, Inl) and isinstance(v.v, Unit):
        return 0
    if isinstance(v, Inr):
        return 1 + dec_nat(v.v)
    raise ValueError("invalid nat encoding")


def enc_nat(n: int) -> Val:
    v: Val = Inl(Unit())
    for _ in range(n):
        v = Inr(v)
    return v


@dataclass(frozen=True)
class Term:
    pass


@dataclass(frozen=True)
class App(Term):
    a: Term
    b: Term


@dataclass(frozen=True)
class Const(Term):
    n: int


@dataclass(frozen=True)
class Pat:
    pass


@dataclass(frozen=True)
class PApp(Pat):
    a: Pat
    b: Pat


@dataclass(frozen=True)
class PConst(Pat):
    n: int


@dataclass(frozen=True)
class PVar(Pat):
    n: int


@dataclass(frozen=True)
class Rule:
    lhs: Pat
    rhs: Pat


def dec_term(v: Val) -> Term:
    if isinstance(v, Inl) and isinstance(v.v, Pair):
        return App(dec_term(v.v.a), dec_term(v.v.b))
    if isinstance(v, Inr):
        return Const(dec_nat(v.v))
    raise ValueError("invalid term encoding")


def enc_term(t: Term) -> Val:
    if isinstance(t, Const):
        return Inr(enc_nat(t.n))
    if isinstance(t, App):
        return Inl(Pair(enc_term(t.a), enc_term(t.b)))
    raise TypeError(t)


def dec_pat(v: Val) -> Pat:
    if isinstance(v, Inl) and isinstance(v.v, Pair):
        return PApp(dec_pat(v.v.a), dec_pat(v.v.b))
    if isinstance(v, Inr) and isinstance(v.v, Inl):
        return PConst(dec_nat(v.v.v))
    if isinstance(v, Inr) and isinstance(v.v, Inr):
        return PVar(dec_nat(v.v.v))
    raise ValueError("invalid pattern encoding")


def dec_rule(v: Val) -> Rule:
    if not isinstance(v, Pair):
        raise ValueError("invalid rule encoding")
    return Rule(dec_pat(v.a), dec_pat(v.b))


def dec_advice(v: Val) -> list[Rule]:
    # nil = Inl (), cons = Inr (h,t)
    if isinstance(v, Inl) and isinstance(v.v, Unit):
        return []
    if isinstance(v, Inr) and isinstance(v.v, Pair):
        return [dec_rule(v.v.a)] + dec_advice(v.v.b)
    raise ValueError("invalid advice encoding")


# ---------- Semantics ----------


Env = Dict[int, Term]  # var_id -> term


def term_eq(a: Term, b: Term) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, Const):
        assert isinstance(b, Const)
        return a.n == b.n
    if isinstance(a, App):
        assert isinstance(b, App)
        return term_eq(a.a, b.a) and term_eq(a.b, b.b)
    raise TypeError(a)


def match_pat(p: Pat, t: Term, env: Env) -> Optional[Env]:
    if isinstance(p, PApp):
        if not isinstance(t, App):
            return None
        env1 = match_pat(p.a, t.a, env)
        if env1 is None:
            return None
        return match_pat(p.b, t.b, env1)
    if isinstance(p, PConst):
        if not isinstance(t, Const):
            return None
        return env if p.n == t.n else None
    if isinstance(p, PVar):
        if p.n in env:
            return env if term_eq(env[p.n], t) else None
        env2 = dict(env)
        env2[p.n] = t
        return env2
    raise TypeError(p)


def inst_pat(p: Pat, env: Env) -> Term:
    if isinstance(p, PApp):
        return App(inst_pat(p.a, env), inst_pat(p.b, env))
    if isinstance(p, PConst):
        return Const(p.n)
    if isinstance(p, PVar):
        return env[p.n]
    raise TypeError(p)


def count_matches(lhs: Pat, t: Term) -> int:
    if match_pat(lhs, t, {}) is not None:
        return 1
    if isinstance(t, App):
        return count_matches(lhs, t.a) + count_matches(lhs, t.b)
    return 0


def apply_rule(rule: Rule, t: Term) -> Optional[Term]:
    env = match_pat(rule.lhs, t, {})
    if env is not None:
        return inst_pat(rule.rhs, env)

    if not isinstance(t, App):
        return None

    c1 = count_matches(rule.lhs, t.a)
    c2 = count_matches(rule.lhs, t.b)

    if c1 == 0 and c2 == 0:
        return None
    if c1 == 0 and c2 > 0:
        b2 = apply_rule(rule, t.b)
        return App(t.a, b2) if b2 is not None else None
    if c1 > 0 and c2 == 0:
        a2 = apply_rule(rule, t.a)
        return App(a2, t.b) if a2 is not None else None

    # both > 0
    if c1 == c2:
        return None
    if c1 > c2:
        b2 = apply_rule(rule, t.b)
        return App(t.a, b2) if b2 is not None else None
    else:
        a2 = apply_rule(rule, t.a)
        return App(a2, t.b) if a2 is not None else None


def step_once(advice: list[Rule], t: Term) -> Optional[Term]:
    for r in advice:
        t2 = apply_rule(r, t)
        if t2 is not None:
            return t2
    return None


def step_2d_value(input_pair: Val) -> Val:
    if not isinstance(input_pair, Pair):
        raise ValueError("expected (advice, term) pair")
    advice_v = input_pair.a
    term_v = input_pair.b
    advice = dec_advice(advice_v)
    term = dec_term(term_v)
    out = step_once(advice, term)
    if out is None:
        return Inl(Unit())
    return Inr(enc_term(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "value",
        nargs="?",
        help="2D value string for (advice,term). You can paste the '(advice,term)' printed by verify when using a debug step.",
    )
    ap.add_argument(
        "--file",
        help="Read the input value string from a file (useful for huge values).",
    )
    ap.add_argument(
        "--pretty",
        action="store_true",
        help="Also print decoded rules and the input term in a compact form.",
    )
    args = ap.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    else:
        if args.value is None or args.value == "-":
            raw = "".join(__import__("sys").stdin.readlines()).strip()
        else:
            raw = args.value

    v_in = parse_val(raw)
    if args.pretty:
        if not isinstance(v_in, Pair):
            raise SystemExit("input is not a pair")
        advice = dec_advice(v_in.a)
        term = dec_term(v_in.b)

        def p_pat(p: Pat) -> str:
            if isinstance(p, PConst):
                return f"C{p.n}"
            if isinstance(p, PVar):
                return f"V{p.n}"
            if isinstance(p, PApp):
                return f"({p_pat(p.a)} {p_pat(p.b)})"
            raise TypeError(p)

        def p_term(t: Term) -> str:
            if isinstance(t, Const):
                return f"C{t.n}"
            if isinstance(t, App):
                return f"({p_term(t.a)} {p_term(t.b)})"
            raise TypeError(t)

        print(f"rules={len(advice)}")
        for i, r in enumerate(advice, 1):
            print(f"{i}: {p_pat(r.lhs)} => {p_pat(r.rhs)}")
        print(f"term: {p_term(term)}")

    v_out = step_2d_value(v_in)
    print(val_to_str(v_out))


if __name__ == "__main__":
    main()
