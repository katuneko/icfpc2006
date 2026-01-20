# UM Rules

## Smellular Antomata (Antomaton)

```text
  A New Kind of Fragrance:


     Smellular Antomata



Modern science has come to know that the essence of plants can be
described by the iterative chaos of fractals. But what of the animal
kingdom? From which recreational mathematics shall we draw our
understanding of the zoological critters? I herein propose a simple
model for the behavior of ants, which I call Smellular Antomata.

The world is a finite rectangular grid, composed of cells. Each cell
has some contents: It may contain floor, a wall, a hole, food, or an
ant. Ants can face north, south, west or east.

As we know, ants are blind and so they navigate by smell. An ant can
only smell its immediate neighborhood. It can't smell holes, so it
walks right into them.

Ants have very small brains. They know left from right, but do not
know what grid direction they are facing, so their behavior is
invariant under rotation. Sometimes a bunch of ants can't make a
decision because they can't orient themselves uniquely. In this case
some ants are eaten by an ever-present anteater, which loves the smell
of indecision.

Each ant belongs to one of ten different clans. Each clan has a fixed
behavior when encountering certain situations. These are described
in the model below. An ant knows what clan it is in, but can't tell
what clan the ants it smells are in, because of all the mud.


--------------------------------------------------
  Ant simulation
--------------------------------------------------

The model repeatedly applies a transformation to simulate the ant
world. This section defines the transformation.

The transformation generates a new grid from the old grid. Each cell
in the new grid is a function of the corresponding cell in the old
grid, plus its four immediate neighbors. So that all cells have
neighbors, we imagine the grid being surrounded by walls. This
transformation is invariant under rotation, so I only give one of the
four orientations.

We write # for wall, - for floor, o for hole, and $ for food. An ant
is written ^ if facing north, < if facing west, > if facing east, and
v if facing south. We write * to indicate that the cell is ignored in
the translation. Multiple patterns from this list may apply; in this
case, the earliest appearing one is used.

  *
 *o*     becomes     o
  *

  *
 *#*     becomes     #
  *

  *
 *$*     becomes     $
  *

(That is, holes, walls, and food stay put.)

  -
 *^*     becomes     -
  *

  o
 *^*     becomes     -
  *

(An ant facing empty space or a hole becomes empty space.)

  #
 *^*     becomes     >
  *

(An ant facing a wall always turns to its right.)


Some of the remaining cases depend on the clan of one involved ant.
Each clan has a particular "turning machine": the directions 
(N, E, S, W) for it to turn in the seven different scenarios. We
write these as p1..p7. Because patterns are invariant under rotation,
these directions are always interpreted relative to the ant's current
orientation. For example, an ant facing > turning "E" then faces v.

  v
 >-<     becomes     -
  ^

(The situation is non-orientable, so an anteater comes and eats
the ants.)

  *
 >-<     becomes     p3  of the bottom ant.
  ^
(The middle of the three ants moves into the center cell,
retaining its clan and turning according the third direction
in its program. The other ants are eaten.)

  v
 *-*     becomes     -
  ^
(Also non-orientable.)

  *
 >-*     becomes     p2  of the bottom ant.
  ^

  *
 *-*     becomes     p1  of the bottom ant.
  ^

  ^
 *^*     becomes     p4  of the ant at center.
  *

  >
 *^*     becomes     p5  of the ant at center.
  *

  v
 *^*     becomes     p6  of the ant at center.
  * 

  <
 *^*     becomes     p7  of the ant at center.
  *


An ant simulation terminates in glorious epicurean success if the
following pattern is encountered:

  *
 *$*
  ^

--------------------------------------------------
  Antomaton simulator
--------------------------------------------------

I have developed a high-performance antomaton simulator. It can be run
from the UMIX system on an ant world written in a file world.ant as
follows:

./antomaton world.ant

The input file format is this:

Title                     (title of this world)
NNNWWWW
WEWENNN                   (up to 10 ant turning machines,
SWESWEN                    each as p1..p7)
6 7                       (width and height)
 # # # # # #
 # - $ -2v #
 # - - - - #              (board)
 # - - - - #
 # - -0^ - #
 #1> - - - #
 # # # # # #

Each cell is specified by two characters. The second character is
either #, o, -, $, <, ^, v, or > as above. If the second character is
an ant direction, then the first must be a digit 0-9 representing its
clan. Otherwise, the first character may be anything.

I have supplied an interesting example in the file example.ant.

The simulator can also be run in step-by-step mode by supplying the
flag -i on the command line.

--------------------------------------------------
  Antomaton quest
--------------------------------------------------

I have included a series of patterns. Each corresponds to ant behavior
observed in the wild, and in each case at least one ant reached the
food. A pattern is just like an input file, except that some of the
grid positions, and some of the turning machine instructions, have
been replaced by wildcards *. Grid positions marked with a wildcard
can be filled with any contents (except $) and turning machine
instructions with any direction. The antomaton simulator will
automatically verify inputs against known patterns based on matching
titles.

If these patterns can be satisfied then I will have shown once and for
all that smellular antomata accurately model real ant behavior!
```

### Formalization (derived)

This section restates the Antomaton rules above in algorithmic form for solver use.

#### State and notation
- Grid size W x H. Coordinates (x,y) with x to the east and y to the south.
- Outside the grid is treated as wall (#) for neighbor lookups.
- Cell kinds: wall (#), hole (o), floor (-), food ($), ant (clan c in 0..9, dir d in {N,E,S,W}).
- Each clan has a program p1..p7 with values in {N,E,S,W}.
- Relative turns: N = straight, E = right, S = reverse, W = left.
  If an ant faces d, then turn(d, t) = d + rot(t) mod 4 where rot(N)=0, rot(E)=1,
  rot(S)=2, rot(W)=3.

#### Synchronous update
Given grid G_t, produce G_{t+1}. Each cell depends only on its old value and the
four von Neumann neighbors in G_t. Rules are priority ordered.

1. Static cells: if center is #, o, or $, then it stays unchanged.
2. Center ant with clear forward:
   - Let d be its direction and f be the neighbor in direction d.
   - If f is floor or hole, the center becomes floor (the ant leaves or falls).
   - If f is wall, the center becomes the same ant turned right.
3. Center floor with incoming ants:
   - An incoming ant is an adjacent ant whose direction points toward the center.
   - If incoming from all four directions, or from an opposite pair (N+S or E+W),
     the result is floor (anteater).
   - Otherwise rotate the neighborhood so an incoming ant from the south is the
     "bottom ant" and apply:
     * p3: incoming from south, west, and east -> center becomes
       ant(clan(bottom), turn(bottom_dir, p3)).
     * p2: incoming from south and west -> center becomes
       ant(clan(bottom), turn(bottom_dir, p2)).
     * p1: incoming from south only -> center becomes
       ant(clan(bottom), turn(bottom_dir, p1)).
   - If no incoming ants, the center stays floor.
4. Center ant with ant ahead:
   - If the forward neighbor is an ant with direction d_f, let rel = (d_f - d) mod 4.
   - Use p_{4+rel} to turn: center becomes ant(clan, turn(d, p_{4+rel})).
5. Otherwise the cell stays unchanged (covers an ant facing food).

#### Success
- The simulation halts if any ant is adjacent to food and facing it. Equivalently,
  an ant whose forward neighbor is food ends the run.

## 2D (Two-Dimensional Language)

This section summarizes the 2D language rules as currently understood
from the UMIX "Programming in Two Dimensions" notes and local usage.

### Overview
- Programs are ASCII diagrams: boxes connected by wires.
- Each box has North/West inputs and South/East outputs; at most one
  wire can connect to each face.
- Wires carry immutable values: a wire is empty until a value is sent,
  then it holds that value forever.
- A box executes once when all its input wires (0, 1, or 2) have values.
- Modules are named rectangles that encapsulate boxes and wires; a
  `use` box instantiates a fresh copy of a module (recursion allowed).
- A module instance completes when no more boxes are ready; exactly one
  output wire must have a value, otherwise evaluation fails.

### Values
```text
val ::= () | (val, val) | Inl val | Inr val
```

### Box Syntax
```text
inface  ::= N | W
outface ::= S | E
exp     ::= () | (exp, exp) | Inl exp | Inr exp | inface
command ::= send []
          | send [(exp, outface)]
          | send [(exp, outface), (exp, outface)]
          | case exp of outface, outface
          | split exp
          | use name
```

- Boxes are drawn with `*` corners, `=` on the north/south edges, and
  `!` on the west/east edges.
- No whitespace is allowed between a command and its surrounding box.
- Names are alphanumeric: `0-9a-zA-Z`.
- Extra parentheses are not permitted, and only single spaces may
  appear between tokens (no double spaces).

### Wire Syntax and Connectivity
- Wires are drawn with `| - + #`. A valid wire must include at least
  one of these characters (a bare `v` or `>` is not a wire).
- Openness rules:
  - `|` is open north/south.
  - `-` is open west/east.
  - `#` is open on all four sides.
  - `+` is open on exactly two of the four sides.
  - `v` is open to its north (used to connect to a box's north edge).
  - `>` is open to its west (used to connect to a box's west edge).
  - A box's south `=` is open to the south; a box's east `!` is open
    to the east.
- Connectedness rules inside a module:
  - Each `-` must connect to open neighbors on both west and east.
  - Each `|` must connect to open neighbors on both north and south.
  - Each `#` must connect to open neighbors on all four sides.
  - Each `+` must connect to exactly two open neighbors.
- On a module boundary, only `|` (top/bottom) and `-` (left/right) are
  allowed, and they only need a single open neighbor inside the module.

### Module Syntax
- A module is a rectangle bordered by `.` on the north/south edges, `:`
  on the west/east edges, and `,` at the corners.
- The module name appears in the upper-left corner, followed by a
  single space.
- Optional inputs:
  - North input is a `|` on the top border.
  - West input is a `-` on the left border.
- Outputs are `-` on the east border; a module may have multiple outputs.
- Modules in a source file do not overlap.

### Semantics (Box Evaluation)
- Expressions evaluate by substituting `N` and `W` with the values on
  the north and west input wires (they must be connected).
- `send []` sends nothing.
- `send [(val, outface)]` sends one value on the chosen output.
- `send [(v1, outface1), (v2, outface2)]` sends both values; the
  outfaces must be distinct.
- `split (v1, v2)` sends `v1` south and `v2` east; any other input fails.
- `case Inl v of o1, o2` sends `v` to `o1`; `case Inr v ...` sends to `o2`.
- `use name` evaluates a fresh instance of the named module; its inputs
  are the inputs of the `use` box, and its single output value is sent
  on the `use` box's east face.
- Sending to an unconnected outface, or pattern-matching failure (e.g.,
  `split ()`), causes evaluation to fail.

### Entry Points and Scoring
- `2d prog.2d` runs module `main` with no inputs and prints the result.
- `verify testname prog.2d` runs built-in tests; score is program area
  (smaller is better).
- Built-in tests: `mult`, `rev`, `raytrace`, `ocult`.

## O'Cult (Advice)

This section formalizes the O'Cult advice language and rewrite rules as
implemented by the local evaluator in `occult.py`.

### Syntax

- Comments are enclosed in `{ ... }` and cannot be nested.
- Tokens are identifiers `[A-Za-z0-9]+`, parentheses, `=>`, `->`, `;`, `.`.
- Terms are applications of atoms and are left-associative.

```text
term  ::= atom | "(" term ")" | term term
atom  ::= [A-Za-z0-9]+
rule  ::= term "=>" term ";"
test  ::= term "->" term ";"
file  ::= {rule}+ "."
```

- A variable is any atom whose first character is a lowercase letter.
- A constant is any atom whose first character is not lowercase.
- Parentheses are used only for grouping; application associates to the left.

### Pattern matching

Given a pattern `p` and term `t`:

- If `p` is a variable `v`, then `v` matches any term. Repeated `v` must
  match the same term each time.
- If `p` is a constant atom `c`, then `t` must be the same atom `c`.
- If `p` is an application `p1 p2`, then `t` must be an application
  `t1 t2` and both `p1` matches `t1` and `p2` matches `t2`.

The result of matching is a substitution environment; applying a rule
replaces variables in the RHS using that environment.

### Rewrite selection (O'Cult strategy)

For a single rule `lhs => rhs` and a term `t`:

1. If `lhs` matches `t` at the root, rewrite to `rhs` with the matched
   substitution.
2. Otherwise, if `t` is an application `l r`:
   - Let `count(lhs, l)` be the number of subterms in `l` that match `lhs`.
   - Let `count(lhs, r)` be the number of subterms in `r` that match `lhs`.
   - If both counts are zero, the rule does not apply.
   - If exactly one count is non-zero, descend into that side.
   - If both are non-zero:
     * If the counts are equal, the rule does not apply.
     * Otherwise, descend into the side with fewer matches.
3. If `t` is an atom and the root does not match, the rule does not apply.

`count(lhs, x)` counts matches at every subterm of `x`, including `x` itself.

### Evaluation order

- Rules are tried in the order they appear in the advice file.
- As soon as one rule rewrites the term, evaluation restarts from the
  first rule.
- Evaluation stops when no rule can rewrite the term.

## Black Knots (bbarker)

Black Knots は「隣接 swap を行ごとに並べ、指定の置換と移動回数を同時に満たす」制約充足問題として定式化できる。

設定
- 幅 W。トークン t は初期位置 t (0..W-1) にある。
- 仕様は各トークン t に対して (output O[t], plinks P[t]) を与える。

1 行の機械 (row)
- 長さ W の文字列で、""><"" は位置 i と i+1 の swap、"|" は swap なしを表す。
- 1 行の swap は隣接の非重複ペアの集合 S_r ⊆ {0..W-2}。
  - i ∈ S_r なら i+1 ∉ S_r (同じセルを共有する swap は不可)。

状態遷移
- 行 r の swap を同時に適用し、トークンの順序を更新する (隣接転置の合成)。

経路 (path) による定式化
- x_t[r] をトークン t の r 行目後の位置とする (r=0..R)。
- 端点制約: x_t[0] = t, x_t[R] = O[t]。
- 移動制約: x_t[r+1] - x_t[r] ∈ {-1, 0, +1}。
- 右移動回数: r_t = |{r | x_t[r+1] = x_t[r] + 1}| = P[t]。
- 左移動回数: l_t = r_t - (O[t] - t) >= 0 (正味の移動量と一致)。
- 行整合性: 各行 r で {x_t[r]} は 0..W-1 の置換であり、
  x_t が +1 へ動くとき、隣のトークンは -1 へ動く (swap は必ず 2 個組)。

本質
- Black Knots は「隣接転置の列で指定置換を実現しつつ、各トークンの右移動回数 P[t] を厳密に満たす」問題。
- 行はパスグラフ上のマッチングなので、同一行で独立な swap しかできない。
- 任意の行数 R を選べるが、制約が強いため探索は「置換 + 個別回数制約」の両方を満たす必要がある。

## Balance User's Manual

```text
                          ,-'~~~'-,
                        .~      `. ~.
                       /    8     |  \
                      :         ,'    :
                      |     .--~      |
                      !    /          !
                       \  |    8     /
                        `. ',      .'
                          `-.___.-`

                        B A L A N C E
                        User's Manual


I. Introduction

Night and day. Beauty and truth. Oxygen and phlogiston.  Everywhere we
look there  are perfect opposites.  Balance is a  programming language
based  on  the  concept  of  harmoniously coexisting  duals.  In  this
language,  every operation has  an equal  and opposite  reaction. This
ensures  that  the machine  state  does  not  stray from  equilibrium.
Because it performs two  operations for each instruction, the language
is also twice as fast as single-operation languages.

Balance programs  are run inside  an 8-bit machine with  the following
features:
   * CODE: an arbitrarily long immutable stream of bytes
   * M[0..255]: 256 8-bit bytes of memory
   * IP: the instruction pointer (an arbitrary non-negative value)
   * IS: the instruction speed (ranging from -16 to +15)
   * sR[0..3]: four 8-bit source registers
   * dR[0..1]: two 8-bit destination registers
   * four instructions

Each instruction is specified by a single 8-bit byte. The bits
are numbered from most to least meaningful as follows:

           lmb
   .--------.
   |76543210|
   `--------'
  mmb

   * The bits 7,6,5 specify the opcode
and depending on the instruction:
   * bits 4,3,2,1,0 specify an immediate value IMM
or
   * bit 4 denotes a destination register D
   * bits 3,2 denote the first source register S1
   * bits 1,0 denote the second source register S2

Every problem in computer science can be solved by an additional layer
of  indirection. Balance  thus provides  this  facility automatically:
Each of the source and  destination registers is an indirect register.
For  instance,  most instructions  do  not  work  on the  contents  of
registers S1  and S2 directly,  but use the  contents of S1 and  S2 as
indices into the  memory M. The result is not stored  in D, but rather
stored in the memory location indicated by the current contents of D.

The  machine begins  with IP  = 0  and IS  = 1.  At each  step  of the
machine, an instruction is fetched from CODE[IP] (where CODE[0] is the
first  instruction,  CODE[1] the  second,  etc.).  The instruction  is
executed, and then IP is increased  (or decreased) by the value of IS,
modulo the length of CODE.*

* Note: For the sake of elegance, this calculation is performed modulo
  2^32, so that the instruction pointer computed is
       (((IS + IP) mod 2^32) mod length(CODE)).

II. Instruction Reference

The  four instructions  of  Balance  are  MATH,  LOGIC,  SCIENCE,  and 
PHYSICS.  The following  is  their  specification;  some  examples are  
given in a separate section below.


Opcode  (bits)   Description
 MATH    001
                 MATH  performs addition  and  its dual,  subtraction.
                 These act on different  registers so that the math is
                 not undone.  All operations are  modular with respect
                 to the number of  relevant bits. Source registers are
                 represented  with  two bits,  so  if  S1  is 3,  then
                 sR[S1+1]  is  sR[0].  Similarly,  dR[1+1]  is  dR[0].
                 Quantities in memory are eight bits, so 250 + 20 is
                 14.

                 M[ dR[D+1] ] <- M[ sR[S1+1] ]  -  M[ sR[S2+1] ]
                 M[ dR[D]   ] <- M[ sR[S1]   ]  +  M[ sR[S2]   ]

 LOGIC   010
                 LOGIC performs bitwise 'and' as well as its perfect
                 dual, bitwise 'exclusive or.'

                 M[ dR[D+1] ] <- M[ sR[S1+1] ]  XOR  M[ sR[S2+1] ]
                 M[ dR[D]   ] <- M[ sR[S1]   ]  AND  M[ sR[S2]   ]

 SCIENCE 000
                 SCIENCE tests  a hypothesis and  determines the speed
                 at  which the program  progresses. When  executed, it
                 sets the instruction speed IS to immediate value IMM,
                 as long  as the memory  cell indicated by  sR[0] does
                 not  contain  0.  Because  this  instruction  behaves
                 specially when  the memory  cell contains 0,  it also
                 behaves specially  if IS is set to  zero: the machine
                 then  halts. The  value IMM  is treated  as  a signed
                 five-bit number  in two's complement form,  so it can
                 take on values from -16 to +15.

                 if M[ sR[0] ] = 0 then (nothing)
                 otherwise IS <- IMM

                 if IS = 0 then HALT
                 else (nothing)

 PHYSICS 011
                 PHYSICS changes what the registers reference, in both
                 a linear  and angular  way. The immediate  value IMM,
                 treated as a signed  five-bit number, is added to the
                 register sR[0]  so that it may  reference a different
                 memory cell. The  instruction also rotates the values
                 between some subset of the registers, according  to a 
                 bitmask  derived from IMM.  The source register sR[0]
                 is always  part of  the rotated  set, so  the bitmask
                 used is a 6 bit  number where  the lowest 5  bits are
                 the same as IMM and the sixth bit is always 1.
                 
                 sR[0] <- sR[0] + (IMM as signed 5-bit number)
                 
                 let L=L0,...,L4 be the registers
                     dR[1], dR[0], sR[3], sR[2], sR[1]
                 then let C be the list of n elements Li
                     such that bit i is set in IMM
                     (bit 0 is the least significant,
                      bit 4 is the most significant)
                 then let Cs be the list (sR[0], C0, ..., C(n-1))
                 and  let Cd be the list (C0, ..., C(n-1), sR[0])
                 then, simultaneously
                      Cd0 <- Cs0
                      ...
                      Cdn <- Csn

Any other  opcode stands  for the instruction  BAIL, which  causes the
machine  to terminate  in failure.  Programmers sometimes  insert such
bugs  deliberately in  order to  quickly halt  the  interpreter during
testing.


III. Examples

If M[sR[0]] = 0, IP = 3, IS = 6, length(CODE) = 100,
and CODE[IP] = SCIENCE 12,
then in the next cycle IP = 9 and IS = 6.

If M[sR[0]] = 9, IP = 3, IS = 6, length(CODE) = 100,
and CODE[IP] = SCIENCE 12,
then in the next cycle IP = 15 and IS = 12.

If sR = {0, 1, 2, 3}, dR = {4, 5}, M = {2, 3, 5, 7, 11, 13, 17, ...},
and CODE[IP] = MATH (0, 3, 1)
then in the next cycle M = {2, 3, 5, 7, 10, 253, 17, ...}.

If sR = {0, 1, 2, 3}, dR = {4, 5}, M = {2, 3, 5, 7, 11, 13, 17, ...},
and CODE[IP] = LOGIC (0, 3, 1)
then in the next cycle M = {2, 3, 5, 7, 3, 7, 17, ...}.

If sR = {0, 1, 2, 3}, dR = {4, 5}
and CODE[IP] = PHYSICS -1
then sR[0] is updated with sR[0] + -1 = 255
and in the next cycle sR = {1, 2, 3, 4}, dR = {5, 255}.

If sR = {0, 1, 2, 3}, dR = {4, 5}
and CODE[IP] = PHYSICS -16
Then sR[0] is updated with sR[0] + (-16) = -16.
The bitmask for rotation is   1  1  0  0   0  0
       for the register set {-16, 1, 2, 3} {4, 5},
so in the next cycle sR = {1, -16, 2, 3}, dR = {4, 5}.

If sR = {0, 1, 2, 3}, dR = {4, 5}
and CODE[IP] = PHYSICS 15
then sR[0] is updated with sR[0] + 15 = 15.
The bitmask for rotation is   1  0  1  1   1  1
       for the register set {15, 1, 2, 3} {4, 5}
so in the next cycle sR = {2, 1, 3, 4}, dR = {5, 15}.


IV. Syntax

The Balance language concrete syntax is simply a single line of bytes,
each written  as two hexadecimal  digits, with no whitespace  or other
characters.


V. Balance Certified Professional Program (BCPP)

As  a professional  programmer, you  are invited  to join  one  of the
industry's    leading   programmer   certification    programs.   BCPP
certification  can be obtained  automatically by  solving a  series of
challenge  problems  and  verifying  the results  using  the  supplied
program "certify".

Industry standards demand 99.999% reliability to achieve high customer
satisfaction.  The  "certify"  program  tests that  solutions  to  the
challenge problems  fall within acceptable tolerance  levels. Since no
program can be  truly bug-free, the certification process  may need to
be run several times before the solution is accepted.

Please see  the file  PUZZLES for the  list of challenge  problems. To
certify  a solution,  stored  in  a file  called  "solve.bal", to  the
problem called "prop", run the command

   certify prop solve.bal

Because a  professional programmer knows  that shorter code  is better
code, your certified skill level  depends on the length of the program
that you submit to the certifier.

A challenge problem consists of  a description of the initial register
and  memory values  and  the desired  final  configuration. A  program
solves the challenge if it  achieves the final configuration and halts
gracefully (SCIENCE 0 with M[sR[0]] <> 0).

Accepted  applicants will  receive an  engraved sandstone  diploma and
will  be  added to  the  19101 electronic  edition  of  "Who's Who  in
Computerology." Please allow 4-6 weeks for delivery.
```

## QVICKBASIC (BASIC-like)

```text
QVICKBASIC version VII.0
usage: qbasic file.bas
```

Notes from observed behavior:
- `qbasic` compiles `.bas` into a `.exe` with the same basename (example: `hack.bas` -> `hack.exe`).
- Example compilation sequence (from `hack_fixed.bas` header):
  - `/bin/qbasic hack.bas`
  - `./hack.exe username`

## 2D Programming Language

```text
Subject: Programming in Two Dimensions


Dear cult.cbv.discuss:

I'm pleased to announce a new programming language called 2D. This
language frees the programmer from the shackles of linear programming
by allowing programs to occupy two dimensions. However, unlike 3- and
4- dimensional languages such as CUBOL and Hypercard, it does not
distract the programmer's attention with needless dimensional abandon.

I first present an overview of the language and then delve into a more
careful description of its syntax and semantics.

== 2D Overview ==

2D programs are built from boxes connected together by wires. A box takes
the following form:

    *=======*
    !command!
    *=======*

Wires can connect boxes:


    *========*       *========*
    !command1!------>!command2!
    *========*       *========*

Each box has two input interfaces: its North and West sides. It also
has two output interfaces, its South and East sides. The following box
sends the input that it receives on its North interface to its East
interface:

       |
       v
    *============*
    !send [(N,E)]!----->
    *============*

Wires carry values from one box to another. Each wire starts out with
no value. When a value is sent along a wire, the wire keeps that same
value forever. A box will only activate when all of its inputs (zero,
one, or two) have values.

The values flowing along wires take on the following forms:

val ::= () | (val, val) | Inl val | Inr val

The () value is the single base value. Two values can be paired
together. They can also be stamped with the disjoint constructors Inl
and Inr. Commands manipulate the structure of values and the control
flow of the program by selectively sending along their outputs. For
example, the 'case' command distinguishes between values stamped with
Inl and Inr:

     |
     v
 *=============*
 !case N of E,S!----
 *=============*
     |
     +--------------

If this box is sent Inl () to its North interface, then () is sent
along the wire connecting to the east interface. If it is sent
Inr ((), ()) then ((), ()) is sent along the south interface instead.


2D programs can be organized into modules. A module encapsulates a
collection of boxes and wires and gives them a name. The following
module, called stamp, encapsulates the operation of applying the Inl
and Inr constructors to the first and second components of a pair:

,........|.......................................,
:stamp   |                                       :
:        v                                       :
:     *=======*                                  :
:     !split N!-----+                            :
:     *=======*     v                            :
:        |       *=========================*     :
:        +------>!send [((Inl W, Inr N),E)]!------
:                *=========================*     :
:                                                :
,................................................,

(The split command splits a pair, sending the first component
 south and the second component east.)

A module can be used as a box itself. The following circuit sends
(Inl (), Inr Inl ()) along the wire to the east:

        *========================*
        !send [(((), Inl ()), E)]!---+
        *========================*   |
    +--------------------------------+
    v
  *=========*
  !use stamp!-----------------------------------
  *=========*

Each time a "use" box is executed, a new copy of the referenced module
is made (with wires carrying no values). Recursion is just a
particular use of modules: modules may also "use" themselves. Mutual
recursion between modules is also permitted.

A module is limited to at most one input along each of its north and
west faces. It may have multiple outputs, all along its east face.
When a module is executed, exactly one of its output wires must be
sent a value; this is the value that the "use" box sends along its
interface.

== 2D Syntax ==

=== Box syntax ===

A box's north and south edges are written with the = symbol. Its west
and east edges, which must be exactly one character long, are written
with the ! symbol. The box's corners are written *. No whitespace is
allowed between the command and the box that surrounds it.

The concrete syntax for commands is as follows:

inface ::= N | W

outface ::= S | E

exp ::= () | (exp, exp) | Inl exp | Inr exp | inface

command ::= send []
          | send [(exp, outface)]
          | send [(exp, outface), (exp, outface)]
          | case exp of outface, outface
          | split exp
          | use "name"

Note that extra parentheses are neither required nor permitted.
A space character may be omitted when the character to its left or to
its right is one of ,()[] and two consecutive space characters are
never allowed.

A name consists of one or more characters from the following set:

0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ

If a wire is connected to the north side of a box, the v character
must be used as follows:

    |
    v
  *=======*
  !command!
  *=======*

The wire can connect above any = character. If a wire is connected to
the west side of a box, the > character must be used as follows:

    *=======*
 -->!command!
    *=======*

At most one wire can be connected to each of a box's four faces.

=== Wire syntax ===

Wires are made from the following characters:

|-+#

Every wire must use at least one of these characters.  That is, 
> and v alone are not valid wires.

Each character is "open" on some of its sides. The | character is
open on its north and south sides. The - character is open on its
west and east sides. The + and # characters are both open on all
four sides.

The = character on the south face of a box is open to its south,
and the ! character on the east side of a box is open to its east.
The v character is open to its north, and the > character is open
to its west.

All wire characters within a module must obey the following rules of
connectedness:

  For each - character, its west and east neighbors must both
  be open on their east and west sides, respectively.

  For each | character, its north and south neighbors must
  both be open on their south and north sides, respectively.

  For each # character, its north, south, west, and east neighbors
  must each be open on their south, north, east, and west sides,
  respectively.

  For each + character, exactly two of the following conditions must
  be met:
    a. its north neighbor is open on its south side
    b. its south neighbor is open on its north side
    c. its west neighbor is open on its east side
    d. its east neighbor is open on its west side

Only the | and - wire characters are allowed along module boundaries, and
they only require a single open neighbor on the inside of the module.
(They do not syntactically connect to anything on the outside.)

=== Module syntax ===

The input consists of an arrangement of non-overlapping modules. Each
module is bordered by the . character on its north and south face, the
: character on its west and east face, and the , character in each
corner. Additionally, the north face may optionally have one
occurrence of the | character; this is the north input to the module.
Similarly, the west input (if any) is represented by a - character.
The east side of the module may have any number of occurrences of the
- character; these are its outputs. A module's name must appear in the
upper left corner of the module and be followed by a space.

== 2D Semantics ==

Evaluation of 2D programs revolves around a function for computing the
value of a module instance. A module instance is a collection of
wires, some of which have values, and the boxes that these wires
connect.

A module instance evaluates in a series of evaluation steps. In each
step, the "ready" boxes are identified as those boxes for which all of
their inputs wires have values, and which have not yet executed in
this instance. All ready boxes are evaluated (see below) in an
arbitrary order. If no boxes are ready, then the module instance is
finished. Its output is the value of the single output wire that has a
value. If more than one wire has a value, or if no wire has a value,
then evaluation fails.

=== Box evaluation ===

Boxes only execute when all of their input wires have values. This is
true even if the command does not reference all of the wires.

Commands are executed as follows. First, all expressions in the
command are evaluated. The expressions N and W are replaced with the
values on the North and West wires, respectively. If a value is needed
but no wire is connected, then evaluation fails. Then, commands are
executed as follows:


send []
  nothing happens.

send [(val, outface)]
  val is sent along the specified outface.

send [(val1, outface1), (val2, outface2)]
  val1 is sent to outface1, and val2 is sent to outface2.
  The two outfaces may not be equal.

split (val1, val2)
  val1 is sent south, and val2 is sent east.

case Inl val of outface1, outface2
  val is sent to outface1.

case Inr val of outface1, outface2
  val is sent to outface2.

use mod
  a new instance of the module mod is evaluated. The inputs to 
  the module must match the inputs to this box, and are instantiated
  with the values along those wires. The output along the east
  face is the output of the module instance.


In any other situation (for example, split ()), the machine fails. 
If a value is sent along an outface, then there must be a wire
connected, or the machine fails.



I've developed a prototype interpreter for 2D, which runs on Umix.
Please try it out!

 - Bill
```

## O'Cult Language (Advice)

```text
Subject: O'Cult Version 1.0 Available

Friends,

On my recent journey across the rivers, I was struck with a simply
remarkable idea for a new way to program our Computing Device.  As you
all know well, it is currently difficult for a programmer to correct a
mistake of one of his fellows---but no longer!  Why, when programming in
O'Cult, one programmer needs to have written barely more than a blank
screen before others can begin debugging his code.

I start from a very simple programming language whose terms are
specified as follows:

e ::= c | e e | (e)

where c ranges over constants and we adopt the convention that
juxtaposition associates to the left.  For example,

     Z
     (S Z)
     Add Z (S Z)

are all well-formed terms, and the last parses as (Add Z) (S Z).

Ordinarily, one would enrich this language with more powerful means of
computation.  Instead, I take a different tack: a term can be _advised_
by a set of external computation _rules_.

A rule is a pair of _patterns_, where a pattern extends the language of
terms with variables.  The term (Add Z (S Z)) is quite inert, but if the
term is advised by the following rule,

     Add Z y => y; 

then the program computes (S Z), as expected.

******************
Rules and Matching
******************

More formally, a rule is a pair of patterns separated by '=>' and
terminated with ';'.  A pattern can contain both constants, which are
sequences of letters and numbers beginning with an *uppercase* letter,
and variables, which are sequences of letters and numbers beginning with
a *lowercase* letter.  A well-formed rule is one where the variables in
the right-hand side are a subset of the variables in the left-hand side.

To define how a rule acts on a term, we first define when a pattern
_matches_ a term yielding a set of bindings:

(1) A constant matches only that same constant, yielding the empty set
    of bindings. For example,

    Z matches Z yielding []
    S does not match Z

(2) A variable matches any term, yielding a binding to that term.  For
    example,
    
    x matches (S Z) yielding [x = (S Z)]

(3) A juxtaposition-pattern matches a juxtaposition-term if 
       (a) the pattern's first position matches the term's first position
       (b) the pattern's second position matches the term's second position
       (c) the bindings from the two positions _unify_: for any variable
	   bound in both positions, the term associated with that variable
	   is the same on both positions.  That is, a variable is allowed to
	   appear in a pattern more than once, but it must match the
	   same term in all locations.
    The bindings of the juxtaposition are the union of the bindings from
    each position.

    For example, 

    x y matches S Z yielding [x = S, y = Z]
    x x matches S S yielding [x = S]
    x x does not match S Z

If a rule matches a term, then _applying_ that rule to the term yields
the right-hand component of the rule with the bindings from the match
substituted for the variables.

*******************
Sentences of Advice
*******************
  
This language would be quite boring if a programmer could only specify
one rule.  So, a term may be modified by a _sentence_ of advice, which
is a sequence of rules terminated with the '.' character.

A program consists of a current term and a sentence of advice.  Because
a program is advised by multiple rules, circumstances can arise when
more than one rule in the sentence matches the term.  A good programming
language is based on common sense above all else, and my sand-father was
very fond of the following aphorism:

    "Advice when most needed is least heeded."
                                     - Unknown

Therefore, there is clearly only one correct semantics for applying
advice to a term:

The rules in the sentence are considered left-to-right.

(1) If the current rule matches the current term, the result is the
    application of that rule to the term.

(2) If the current rule does not match the term directly, it may match
    subterms (provided that the term is a juxtaposition).  In this case, 
    whether or not the current rule is applied is determined by:

   (a) Counting the number of matches in each position of the juxtaposition.
       Note that counting does not proceed into subterms that themselves
       match the current rule.

   (b) If the rule does not match in either position, it is not applied.  

       If the rule matches only in one position, it is recursively
       considered for application to that position.

       If the rule matches both positions,
         * if one position has strictly more matches, the rule is
	   recursively considered for application to *other* position.
	   (The rule is least heeded in the position where it is most
	   needed.)

	 * if the rule matches the same number of subterms in both
	   positions, the rule is not applied.

When a rule is not applied, consideration proceeds to the next rule.
When a rule is applied, the process repeats on the new term.  This
process terminates when no rules in the advice apply.

**********
Conclusion
**********

I am sure you can see how easy it is to program in O'Cult.  Now I need
your assistance.  I have included a regression suite as part of the
advise distribution (see the man pages for details), but I need to
collect programs that pass the suite.

I think we might be able to get some good publications out of this work,
but only if we can prove that it is easy to write short programs.  Try
passing the regression suite with as pithy advice as possible.  (The size
of a sentence is the sum of the sizes of the rules in it, where all
variables are constants are considered to have unit size.)

I implore you to hold this idea in confidence; its clean modularization
of crosscutting concerns may prove key in our strife with the Cult of
the LValue.

Please let me know if you have any questions or suggestions,
-Harmonious
```
