# CIRCS.ocult (O'Cult) — Complete Problem Definition + Real Test Vector

This document is meant to be copy-pasted to another solver/model as the full, precise statement of what needs to be implemented to pass `verify ocult` (ICFPC 2006 / UMIX / Ohmega).

The deliverable is a **2D module named `step`**. It must implement **one O'Cult rewrite step** on a 2D-encoded `(advice, term)` input.

## Problem Statement (What You Must Build)

You are solving the **CIRCS.ocult** 2D verification task.

Implement a 2D program that defines a module:

- **Module name**: `step`
- **Input** (West): a single 2D value `W = (advice, term)` encoded as described below
- **Output** (East): the result of performing **exactly one** O'Cult rewrite step using the *least-heeded* semantics (described below)

Passing condition:

- `verify ocult your_program.2d` succeeds (UMIX / verify harness).
- Your score is the program “area” (`verify` prints `Program area: N (smaller is better)`).

The authoritative sources for the task are:

- Ohmega mail: `problems/problems.md` (defines the 2D data encoding for advice/terms/patterns)
- Harmonious Monk mail/manpage: `volume9_hmonk_details.txt` (defines least-heeded rule application)

## What `verify ocult` Actually Expects

Ohmega's mail in `problems/problems.md` says “output the term”, but **the actual `verify ocult` expects an Option**:

- Output `Inl ()` if no rule applies (no rewrite step possible).
- Output `Inr [[nextTerm]]` if one rewrite step is possible, where `[[nextTerm]]` is encoded as a **term**.

Evidence (local logs):
- `volume9_ohmega_verify_ocult_id.txt`
- `volume9_ohmega_verify_ocult_patch_tbl.txt` (`Expected    : Inr ...`)

## 2D Data Encoding (from Ohmega mail)

Source: `problems/problems.md` around the “2D the Ultimate Programming Language?” mail.

2D values are built from:
- `()` (unit)
- `(a,b)` (pair)
- `Inl v` / `Inr v` (sum)

### Natural numbers (names)

```
[[zero]] = Inl ()
[[s(n)]] = Inr [[n]]
```

### Terms

```
[[App(e1,e2)]] = Inl ([[e1]], [[e2]])
[[Const s]]    = Inr [[s]]
```

### Patterns

```
[[App(p1,p2)]] = Inl ([[p1]], [[p2]])
[[Const s]]    = Inr Inl [[s]]
[[Var s]]      = Inr Inr [[s]]
```

### Rules and advice lists

```
[[p1 => p2]]    = ([[p1]], [[p2]])

[[nil]]         = Inl ()
[[cons(h,t)]]   = Inr ([[h]], [[t]])
```

## Semantics to Implement (Least-Heeded Advice)

The semantics are described in `volume9_hmonk_details.txt`, section “Sentences of Advice”.

In short:
- Rules are considered left-to-right.
- A rule may apply at the root term, or inside subterms.
- If a rule can apply, we select where it applies via the “least heeded” rule based on **counts of matches** in each subterm.
- For `verify ocult` we need a **single rewrite step**: apply the first rule (in advice order) that can be applied somewhere according to the least-heeded semantics, and return the rewritten term.

### Required core operations

#### 1) Pattern matching: `match(pat, term) -> Option(env)`

- `Const n` matches only `Const n`.
- `App(p1,p2)` matches only `App(t1,t2)` and matches both sides.
- `Var k` matches any term:
  - If `k` is unbound: bind `k := term`.
  - If `k` is bound: it matches only if the term is structurally equal to the bound term.

The environment `env` is a map: `var_id -> term`.

#### 2) Instantiation: `inst(rhsPat, env) -> term`

- Replace `Var k` with `env[k]`.
- Rebuild `Const` and `App` recursively.

#### 3) Match counting: `count_matches(lhs, term) -> Nat`

Key rule: **do not descend into a subterm that itself matches the rule**.

```
count_matches(lhs, t):
  if match(lhs, t) succeeds: return 1
  else if t is App(t1,t2): return count_matches(lhs,t1) + count_matches(lhs,t2)
  else return 0
```

#### 4) Apply a single rule: `apply_rule(rule, term) -> Option(term)`

Let `rule = lhs => rhs`.

1) If `lhs` matches `term` directly, rewrite at the root:
   - return `Some(inst(rhs, env))`

2) Otherwise, if `term` is not an `App`, rule cannot apply:
   - return `None`

3) Otherwise (`term = App(t1,t2)`), compute:

```
c1 = count_matches(lhs, t1)
c2 = count_matches(lhs, t2)
```

and decide:

- If `c1 == 0` and `c2 == 0`: `None`
- If exactly one side has matches: recursively consider the rule in that side, and rebuild `App`.
- If both sides have matches:
  - If `c1 == c2`: do **not** apply the rule (`None`)
  - If `c1 != c2`: recursively consider the rule in the side with **fewer** matches
    (least heeded where most needed)

If recursion yields `None`, this rule is considered “not applied” at this term.

#### 5) One-step advice application: `step_once(advice, term) -> Option(term)`

Advice is a list of rules.

```
for rule in advice (left-to-right):
  t2 = apply_rule(rule, term)
  if t2 is Some: return Some(t2)
return None
```

#### 6) Output convention for `verify ocult`

```
if step_once(...) == None: output Inl ()
else:                     output Inr [[nextTerm]]
```

(`[[nextTerm]]` is encoded as a term, *not* a pattern.)

## Real Extracted Test Vector (Input and Expected Output)

This is a real `(advice, term)` value extracted from a `verify ocult` run by using a debug `step` that prints `(W,E)` (see `ocult_pair.2d` and `volume9_ohmega_verify_ocult_pair_run_20260214.txt`).

### Input: 2D value `(advice, term)`

File created locally: `ocult_input_pair_val_20260214.txt` (also embedded here).

```text
(Inr ((Inl (Inl (Inr Inl Inr Inr Inr Inl (),Inr Inl Inr Inl ()),Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inl ()),Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inl ()),Inr ((Inl (Inl (Inr Inl Inr Inr Inr Inl (),Inl (Inr Inl Inl (),Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inl ())),Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inl ()),Inl (Inr Inl Inl (),Inl (Inl (Inr Inl Inr Inr Inr Inl (),Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inl ()),Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inl ()))),Inr ((Inl (Inr Inl Inr Inr Inl (),Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inl ()),Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inr Inl ()),Inl ()))),Inl (Inr Inr Inr Inl (),Inl (Inl (Inr Inr Inr Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inr Inr Inl ()))),Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inr Inr Inl ()))))))
```

### Human-readable decoded form

This decode was produced by `python3 ocult_ref.py --file ocult_input_pair_val_20260214.txt --pretty`.

```text
rules=3
1: ((C3 C1) V14) => V14
2: ((C3 (C0 V15)) V14) => (C0 ((C3 V15) V14))
3: (C2 V15) => V15
term: (C2 ((C3 (C0 (C0 C1))) (C0 (C0 (C0 C1)))))
```

Legend:
- `Ck` = `Const k`
- `Vk` = `Var k`
- `(a b)` = `App(a,b)`

### Expected output (Option(term))

Produced by `python3 ocult_ref.py --file ocult_input_pair_val_20260214.txt`, and matches `verify ocult` “Expected”:

```text
Inr Inl (Inr Inr Inr Inl (),Inl (Inr Inl (),Inl (Inl (Inr Inr Inr Inr Inl (),Inl (Inr Inl (),Inr Inr Inl ())),Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inr Inr Inl ()))))))
```

## Worked Example (Manual One-Step Trace for the Real Vector)

This section shows how the expected output is obtained by the semantics above, in a way that can be followed by hand.

Input decoded (from `--pretty`):

```text
rules=3
1: ((C3 C1) V14) => V14
2: ((C3 (C0 V15)) V14) => (C0 ((C3 V15) V14))
3: (C2 V15) => V15
term: (C2 ((C3 (C0 (C0 C1))) (C0 (C0 (C0 C1)))))
```

Let:

```text
T = (C2 t2)
t2 = ((C3 (C0 (C0 C1))) (C0 (C0 (C0 C1))))
```

We evaluate `step_once(rules, T)` by trying rules left-to-right.

### Rule 1: `((C3 C1) V14) => V14`

- Root match on `T` fails because `T` is `(C2 ...)` and the LHS root requires `((C3 C1) ...)`.
- Since `T` is an `App`, we compute counts:
  - `count_matches(lhs1, C2) = 0` (LHS is `App`, term is `Const`)
  - `count_matches(lhs1, t2) = 0` (no subterm matches the required shape)
- Both sides have 0 matches, so rule 1 is not applicable.

### Rule 2: `((C3 (C0 V15)) V14) => (C0 ((C3 V15) V14))`

Root match on `T` fails (same reason: left is `C2`).

Compute counts at `T = App(C2, t2)`:

- `c1 = count_matches(lhs2, C2) = 0`
- `c2 = count_matches(lhs2, t2) = 1` because `t2` matches `lhs2` at the root, so counting returns 1 and does not descend.

Since only the right side has matches, we recurse into `t2` and try to apply rule 2 there.

Now check root match on `t2`:

```text
t2 = (L R)
L = (C3 (C0 (C0 C1)))
R = (C0 (C0 (C0 C1)))
lhs2 = ((C3 (C0 V15)) V14)
```

Matching succeeds with environment:

- `V15 := (C0 C1)`
- `V14 := (C0 (C0 (C0 C1)))`

Instantiate the RHS:

```text
(C0 ((C3 V15) V14))
=> (C0 ((C3 (C0 C1)) (C0 (C0 (C0 C1)))))
```

So:

```text
new_t2 = (C0 ((C3 (C0 C1)) (C0 (C0 (C0 C1)))))
nextTerm = (C2 new_t2)
        = (C2 (C0 ((C3 (C0 C1)) (C0 (C0 (C0 C1))))))
```

At this point rule 2 applied, so `step_once` stops and returns `Some(nextTerm)` (it does not try rule 3).

Therefore the final `step` output is:

```text
Inr [[nextTerm]]
```

which is exactly the expected value shown above:

```text
Inr Inl (Inr Inr Inr Inl (),Inl (Inr Inl (),Inl (Inl (Inr Inr Inr Inr Inl (),Inl (Inr Inl (),Inr Inr Inl ())),Inl (Inr Inl (),Inl (Inr Inl (),Inl (Inr Inl (),Inr Inr Inl ()))))))
```

## Reference Evaluator (Oracle)

Local reference implementation:
- `ocult_ref.py`

It parses 2D value strings (same syntax as `verify` prints) and computes:
- `Inl ()` if no rule applies in one step
- `Inr [[nextTerm]]` otherwise

Useful commands:

```bash
python3 ocult_ref.py --file ocult_input_pair_val_20260214.txt --pretty
python3 ocult_ref.py --file ocult_input_pair_val_20260214.txt
```
