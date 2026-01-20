#!/usr/bin/env python3
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass(frozen=True)
class Atom:
    name: str

    def is_var(self) -> bool:
        return self.name and self.name[0].islower()

@dataclass(frozen=True)
class App:
    left: 'Term'
    right: 'Term'

Term = Atom | App

@dataclass(frozen=True)
class Rule:
    lhs: Term
    rhs: Term

@dataclass(frozen=True)
class Test:
    initial: Term
    expected: Term

TOKEN_RE = re.compile(r"\s+|\{|\}|\(|\)|=>|->|;|\.|[A-Za-z0-9]+")


def strip_comments(text: str) -> str:
    out = []
    i = 0
    depth = 0
    while i < len(text):
        ch = text[i]
        if ch == '{':
            depth += 1
            i += 1
            # skip until matching }
            while i < len(text) and depth > 0:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def tokenize(text: str) -> List[str]:
    tokens = []
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.isspace() or tok == '{' or tok == '}':
            continue
        tokens.append(tok)
    return tokens


def parse_term(tokens: List[str], i: int, stop: Optional[set] = None) -> Tuple[Term, int]:
    if stop is None:
        stop = set()
    term, i = parse_atom(tokens, i, stop)
    while i < len(tokens) and tokens[i] not in stop:
        if tokens[i] in (')', '=>', '->', ';', '.'):
            break
        nxt, i = parse_atom(tokens, i, stop)
        term = App(term, nxt)
    return term, i


def parse_atom(tokens: List[str], i: int, stop: set) -> Tuple[Term, int]:
    if i >= len(tokens):
        raise ValueError('unexpected EOF')
    tok = tokens[i]
    if tok == '(':
        term, j = parse_term(tokens, i + 1, stop={')'})
        if j >= len(tokens) or tokens[j] != ')':
            raise ValueError('expected )')
        return term, j + 1
    if tok in stop:
        raise ValueError(f'unexpected token {tok}')
    if tok in ('=>', '->', ';', '.'):
        raise ValueError(f'unexpected token {tok}')
    return Atom(tok), i + 1


def parse_rules(text: str) -> List[Rule]:
    text = strip_comments(text)
    tokens = tokenize(text)
    rules: List[Rule] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == '.':
            break
        lhs, i = parse_term(tokens, i, stop={'=>', ';', '.'})
        if i >= len(tokens) or tokens[i] != '=>':
            raise ValueError('expected =>')
        i += 1
        rhs, i = parse_term(tokens, i, stop={';', '.'})
        if i >= len(tokens) or tokens[i] != ';':
            raise ValueError('expected ;')
        i += 1
        rules.append(Rule(lhs, rhs))
    return rules


def parse_tests(text: str) -> List[Test]:
    text = strip_comments(text)
    tokens = tokenize(text)
    tests: List[Test] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == '.':
            break
        initial, i = parse_term(tokens, i, stop={'->', ';', '.'})
        if i >= len(tokens) or tokens[i] != '->':
            raise ValueError('expected ->')
        i += 1
        expected, i = parse_term(tokens, i, stop={';', '.'})
        if i >= len(tokens) or tokens[i] != ';':
            raise ValueError('expected ;')
        i += 1
        tests.append(Test(initial, expected))
    return tests


def term_to_str(term: Term) -> str:
    if isinstance(term, Atom):
        return term.name
    # left-assoc output with parentheses to be unambiguous
    left = term_to_str(term.left)
    right = term_to_str(term.right)
    return f"({left} {right})"


def match(pattern: Term, term: Term, env: Dict[str, Term]) -> Optional[Dict[str, Term]]:
    if isinstance(pattern, Atom):
        if pattern.is_var():
            bound = env.get(pattern.name)
            if bound is None:
                env[pattern.name] = term
                return env
            if bound == term:
                return env
            return None
        if isinstance(term, Atom) and term.name == pattern.name:
            return env
        return None
    if isinstance(term, App):
        env = match(pattern.left, term.left, env)
        if env is None:
            return None
        return match(pattern.right, term.right, env)
    return None


def substitute(term: Term, env: Dict[str, Term]) -> Term:
    if isinstance(term, Atom):
        if term.is_var() and term.name in env:
            return env[term.name]
        return term
    return App(substitute(term.left, env), substitute(term.right, env))


def count_matches(pattern: Term, term: Term) -> int:
    if match(pattern, term, {}) is not None:
        return 1
    if isinstance(term, App):
        return count_matches(pattern, term.left) + count_matches(pattern, term.right)
    return 0


def apply_rule(pattern: Term, replacement: Term, term: Term) -> Tuple[bool, Term]:
    env = match(pattern, term, {})
    if env is not None:
        return True, substitute(replacement, env)
    if isinstance(term, App):
        count_left = count_matches(pattern, term.left)
        count_right = count_matches(pattern, term.right)
        if count_left == 0 and count_right == 0:
            return False, term
        if count_left == 0:
            applied, new_right = apply_rule(pattern, replacement, term.right)
            return applied, App(term.left, new_right)
        if count_right == 0:
            applied, new_left = apply_rule(pattern, replacement, term.left)
            return applied, App(new_left, term.right)
        if count_left == count_right:
            return False, term
        if count_left < count_right:
            applied, new_left = apply_rule(pattern, replacement, term.left)
            return applied, App(new_left, term.right)
        applied, new_right = apply_rule(pattern, replacement, term.right)
        return applied, App(term.left, new_right)
    return False, term


def run(rules: List[Rule], term: Term, max_steps: int = 100000) -> Tuple[Term, int]:
    steps = 0
    while steps < max_steps:
        applied_any = False
        for rule in rules:
            applied, new_term = apply_rule(rule.lhs, rule.rhs, term)
            if applied:
                term = new_term
                steps += 1
                applied_any = True
                break
        if not applied_any:
            break
    return term, steps


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('advice')
    parser.add_argument('tests')
    args = parser.parse_args()

    rules = parse_rules(open(args.advice, 'r', encoding='utf-8').read())
    tests = parse_tests(open(args.tests, 'r', encoding='utf-8').read())

    ok = True
    for t in tests:
        out, steps = run(rules, t.initial)
        if out != t.expected:
            ok = False
            print('FAIL')
            print('Initial:', term_to_str(t.initial))
            print('Got    :', term_to_str(out))
            print('Expected:', term_to_str(t.expected))
            print('Steps:', steps)
            break
    if ok:
        print('All tests passed:', len(tests))

if __name__ == '__main__':
    main()
