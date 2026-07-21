# COVENANT finite-policy semantics 0.1

Status: normative for `afterimage-covenant-contract/0.1` and
`afterimage-covenant-policy/0.1`.

## 1. Purpose

COVENANT synthesizes deterministic local policies for asynchronous agents.
Acceptance is exhaustive over a declared finite model; sampling or randomized
schedules cannot establish validity. Every reachable state must satisfy all
safety predicates, and every triggered obligation must reach its goal within
its published bound under every permitted fair schedule.

## 2. Contract

A contract has exactly:

```json
{
  "format":"afterimage-covenant-contract/0.1",
  "domains":{"alarm":[false,true]},
  "initial":[{"alarm":false}],
  "actors":[],
  "actions":[],
  "scheduler":{"fairness_window":2},
  "safety":[],
  "liveness":[],
  "limits":{"max_reachable_states":10000,"max_transitions":100000,"max_expression_nodes":4096}
}
```

`domains` is a non-empty map from state-field Text to a non-empty set of
canonical CRE Values. Every initial and reached state has exactly those fields
and one member of each domain. Initial states are unique.

Actors are UTF-8 ID sorted and have exactly:

```json
{"id":"dispatch","kind":"agent","observes":["alarm"],"actions":["dispatch.wait"]}
```

`kind` is `agent` or `environment`. Agent observations are unique domain-field
names. Environment actors observe no state and represent nondeterministic
external choices. Every actor lists exactly its actions.

Actions are UTF-8 ID sorted and have exactly:

```json
{"id":"dispatch.wait","actor":"dispatch","guard":["const",true],"updates":{}}
```

Guards are Boolean expressions. Updates are simultaneous expressions over the
old state; omitted fields are unchanged. Leaving a declared domain is an
invalid transition, not an implicit domain extension.

Safety entries have `id` and Boolean `predicate`. Liveness entries have `id`,
Boolean `trigger`, Boolean `goal`, and a positive transition `bound`. Property
IDs are unique within their list. Both lists are non-empty.

## 3. Expressions

Expressions are prefix lists. The exact 0.1 opcodes are:

- `const`, `get`;
- `not`, variadic `and`, variadic `or`;
- `eq`, `ne`, `lt`, `le`, `gt`, `ge` using CRE Value equality/order;
- checked signed-i64 `add` and `sub`;
- lazy `if`.

There are no loops, calls, dynamic field names, or host callbacks. Contract
and policy expression nodes are bounded before exploration.

## 4. Policy and locality

A candidate policy has exactly:

```json
{
  "format":"afterimage-covenant-policy/0.1",
  "agents":[
    {"agent":"dispatch","rules":[{"when":["get","alarm"],"action":"dispatch.send"}],"default":"dispatch.wait"}
  ]
}
```

There is exactly one UTF-8-sorted entry per agent actor and none for
environment actors. Rules are checked in listed order; the first true rule
selects its action, otherwise `default` is selected. Every referenced field in
a rule condition must belong to that agent's `observes` list. Selecting a
disabled action in a reachable state invalidates the policy with a shortest
counterexample. Action guards and updates may use global state because they
define the published plant, not the agent's decision procedure.

`policy_nodes` counts one policy root, two nodes per agent, and two nodes plus
the expression-node count per rule.

## 5. Fair asynchronous schedules

At each state an agent contributes its one policy-selected enabled action. An
environment actor contributes every enabled action. Disabled environment
actors are absent from that step.

The scheduler tracks, for each currently enabled actor, how many consecutive
transitions selected another actor. If any enabled actor has age at least
`fairness_window`, the next transition must select an overdue actor; otherwise
any enabled actor may be selected. Selection resets that actor's age to zero;
other enabled ages increment and saturate at the window; disabled actors reset
to zero. Actor and action enumeration is raw UTF-8 ID order, making shortest
counterexamples stable.

## 6. Safety and bounded liveness

The checker performs breadth-first exploration to a fixed point over global
state, fairness ages, and obligation ages.

When a liveness trigger becomes true while its goal is false, an obligation
starts at age zero. Each transition increments its age. Reaching the goal at
age at most `bound` discharges it and contributes to
`worst_response_bound`. Exceeding the bound, or deadlocking with an outstanding
obligation, is invalid. A goal already true does not create an obligation.

Safety is checked in every reached state. Exploration aborts before exceeding
`max_reachable_states` or `max_transitions`. A valid result reports:

```text
(policy_nodes, worst_response_bound, reachable_states)
```

`reachable_states` counts the full augmented exploration states, not only
distinct global maps. Invalid results return one stable primary code and a
shortest public actor/action/state trace; they never expose verifier internals
or unrelated witness data.

The 0.1 host hard ceilings are 64 state fields, 256 values per domain, 256
initial states, 32 actors, 256 actions, 64 properties of each kind, fairness
window 64, 50,000 reached states, 500,000 transitions, 100,000 expression
nodes, and expression depth 128. A bundle cannot raise these ceilings.
