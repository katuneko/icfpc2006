# PULSE 0.1 language and exhaustive execution

Status: normative for `afterimage-pulse/0.1` production programs.

PULSE is a bounded event-handler language for small temporal controllers. It
has typed scalar cells, immutable source events, replaceable keyed timers, and
no iteration or host callbacks. A verifier MUST reject programs or executions
that exceed the public contract limits.

## 1. Program document

A program is a canonical CRE JSON value with exactly these fields:

```json
{"format":"afterimage-pulse/0.1","cells":[],"handlers":[]}
```

`cells` is an ordered list of unique `{name,type,initial}` maps. The only cell
types are `int` and `bool`; the initial value MUST have exactly that type.
Cells exist for the entire run, so `live_state_cells` is the declared cell
count.

`handlers` is a non-empty list of unique `{id,on,actions}` maps. `id` is used
for deterministic handler order. `on` MUST name one of the contract input
topics or its timer topic. At one event, all matching handlers run in UTF-8
byte order of their IDs; actions within one handler run in document order.

## 2. Expressions

Expressions are prefix arrays:

| Form | Type and meaning |
| --- | --- |
| `["const", value]` | CRE value; `int` and `bool` retain their scalar type |
| `["cell", name]` | current typed cell value |
| `["event", "at"]` | current event time as `int` |
| `["event", "payload"]` | current event payload |
| `add`, `sub`, `mul`, `min` | two integers to integer |
| `lt`, `le` | two integers to Boolean |
| `eq` | equal-compatible values to Boolean using CRE equality |
| `and`, `or` | two Booleans to Boolean |
| `not` | one Boolean to Boolean |

The verifier MUST type-check every expression and action before running any
fixture. There is no implicit Boolean/integer conversion.

## 3. Actions

The supported action maps are:

```json
{"op":"set","cell":"deadline","value":["const",5]}
{"op":"schedule","key":"bell","topic":"pulse.timer","at":["cell","deadline"],"payload":["const",{}]}
{"op":"cancel","key":"bell"}
{"op":"emit","topic":"rail.stable-bell","payload":["event","payload"]}
{"op":"when","condition":["const",true],"actions":[]}
```

`set` MUST preserve the declared cell type. `schedule` uses a non-empty static
key, the contract timer topic, an integer time not earlier than the current
event, and any CRE payload. Scheduling a key atomically invalidates its older
pending timer. Stale queue entries are ignored and execute no handler.
`cancel` uses a non-empty static key and invalidates that key's pending timer
without scheduling replacement work. Cancelling an inactive key is a no-op.
`emit` MUST use the contract output topic. `when` executes its nested actions
only when its Boolean condition is true. Conditional nesting is capped at
eight levels; it does not introduce a loop.

## 4. Queue and microsteps

The initial queue contains the fixture's external input events. Queue order is
the tuple:

```text
(time, phase, source EventId)
```

External inputs have phase 0 and timers have phase 1. Consequently, every
external pulse at time `t` runs before a timer due at `t`; it can replace that
timer before the timer fires. Within one phase, source EventId order is used.
This microstep rule makes a pulse exactly five ticks after its predecessor
part of the same cluster.

Timer EventIds bind the timer key, due time, payload, replacement generation,
and scheduling source EventId. Emitted outputs are observed in execution
order. Runs begin with fresh cells, queue, generations, and output trace.

## 5. Contract tasks and exhaustive domains

A contract selects one finite input domain and one expected-output oracle. A
contract without `task` is the frozen `debounce` form used by `PULSE.001`.
New contracts MUST carry an explicit `task`. Common fields name the input,
timer, and output topics; bound the horizon and input length; add named
fixtures; and publish execution limits. A verifier MUST reject unknown tasks
and fields rather than silently interpreting a near match.

### 5.1 One Bell: `debounce`

For `PULSE.001`, the generated domain is every subset of times `0..11` having
zero through seven members. It has 3,302 sequences. Named fixtures add
duplicate-time sequences `(0,0)` and `(11,11)`; boundary subsets already in
the generated domain are de-duplicated. The final domain therefore has 3,304
runs and is checked in tuple lexicographic order.

A maximal cluster continues while each adjacent gap is at most five. Its only
expected output occurs at `last_pulse + 5`, on the contract output topic, with
the exact contract payload. The empty input has no output. The first mismatch
or resource failure is the stable counterexample and reports input, expected
trace, observed trace when available, and an inner error code for runtime
failures.

### 5.2 First Copy: `deduplicate-ticks`

For `PULSE.002`, the generated domain is every nondecreasing sequence of zero
through five times drawn from `0..7`, including repetitions. There are 1,287
runs; named fixtures already in that set do not add duplicates. Runs are
checked in tuple lexicographic order.

At each logical tick, source events are ordered by EventId as specified in
section 4. The expected trace contains the first event in that canonical
order, emitted immediately on the contract output topic with its original
payload. Later inputs at the same tick produce no output. Consequently,
`worst_latency` is zero. A checker that generates combinations without
replacement is non-conforming because it omits the task's essential cases.

### 5.3 Silent Timeout: `cancelable-timeout`

This task publishes distinct `start` and `cancel` input topics, a positive
`timeout_ticks`, a fixed output payload, a horizon, and `max_commands`.
The generated domain contains every nondecreasing sequence of zero through
three commands over six ticks, with both kinds independently chosen at every
position. It has 545 traces. Each command retains its authored sequence in
the source EventId, and same-tick commands execute in EventId order.

`start` replaces the one logical deadline with `at + timeout_ticks`; `cancel`
removes it. A still-active deadline emits the exact contract payload. Because
all phase-0 external commands execute before phase-1 timers, a cancel at the
deadline suppresses the output. A start at that tick replaces the old timer
before it can fire. A checker that treats cancel as a far-future replacement
is non-conforming because it creates scheduled work not present in the task.

### 5.4 Token Window: `token-bucket`

This task publishes a positive `capacity`, an `initial_tokens` value between
zero and capacity, a positive integer `refill_per_tick`, a horizon, and
`max_requests`. Its generated domain contains every nondecreasing sequence of
zero through `max_requests` request times, so same-tick bursts are exhaustive
and retain canonical source-EventId order.

Before each request at tick `t`, the bucket is refilled and clamped exactly as
`min(capacity, tokens + (t - last_at) * refill_per_tick)`, then `last_at`
becomes `t`. A positive bucket admits the request immediately with its original
payload and consumes one token. A rejected request consumes nothing. This
ordering means several requests at one tick share one refill but compete in
canonical input order. `worst_latency` is zero.

A checker that omits repeated times, fails to clamp after an idle gap, or
charges rejected work is non-conforming. `mul` and `min` are part of the
language so the submitted controller implements the same published arithmetic
instead of relying on a hidden host primitive.

### 5.5 City Clock: `city-clock`

This task publishes exactly three distinct input roles, `archive`,
`controller`, and `policy`, plus distinct timer and output topics. Its generated
domain is every nondecreasing stream of zero through six labeled signals over
three ticks: 27,064 canonical runs. Same-tick events execute in EventId order.

The oracle remembers the set of roles seen in the current round. A duplicate
role is coalesced. When the third distinct role arrives, the current event is
emitted immediately with its original payload and the remembered set is
cleared atomically. A later event, including one at the same tick, belongs to
the next round. Thus two complete rounds require two outputs; a permanent
one-shot `fired` flag is non-conforming. `worst_latency` is zero.

## 6. Limits and metrics

The public contract bounds canonical program bytes, declared state cells,
executed actions, maximum queue size, and scheduled timers. Replaced timers
still count as scheduled work and remain queue entries until discarded, so a
program cannot evade resource accounting through replacement.

After all domain cases pass, the score tuple is:

```text
(program_bytes, worst_latency, live_state_cells)
```

`program_bytes` is the canonical byte length of the program map only; the
optional invariant note is excluded. `worst_latency` is task-defined: five
for One Bell, zero for First Copy, Token Window, and City Clock, and three for
Silent Timeout. `live_state_cells` is the declared cell count. Scoring is considered
only after exhaustive validity succeeds.
