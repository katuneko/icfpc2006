# Causal Reduction Engine 0.1

Status: normative implementation specification for the vertical slice.

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** describe
interoperability requirements. CRE 0.1 is frozen for the two-reference
reference implementations and playable-slice gate. The schema and core
semantics are frozen; section 11's complete release matrix is still a gate.
Any incompatible change requires a new semantic version.

## 1. Scope

The Causal Reduction Engine (CRE) evaluates finite collections of immutable
events and stratified reaction rules. Evaluation derives new events to a least
fixed point. A named projection turns the active fixed point into observable
records. A counterfactual branch changes base events and triggers a complete
replay.

CRE is deterministic. Equal inputs under one semantic version MUST produce the
same active event IDs, projection output, trace digest, and resource counters.

CRE is not:

- a general operating system;
- a native-code sandbox;
- a concurrent or wall-clock-driven runtime;
- a mutable database;
- a probabilistic language;
- a distributed consensus implementation.

Those domains may be modeled as data inside CRE.

## 2. Primitive values

The value grammar is:

```text
Value := null
       | Bool
       | Int
       | Text
       | Bytes
       | List<Value>
       | Map<Text, Value>
```

Requirements:

- `Int` is a signed 64-bit integer in `[-2^63, 2^63-1]`.
- Arithmetic overflow is a deterministic evaluation error. It MUST NOT wrap.
- `Text` is Unicode scalar text normalized to NFC before canonical encoding.
- `Bytes` is an arbitrary finite octet sequence.
- Map keys are unique normalized `Text` values.
- Floating-point numbers do not exist.
- Host-language nullability, object identity, and map order have no semantic
  meaning.

Canonical JSON represents `Bytes` as:

```json
{"$bytes":"base64url-without-padding"}
```

A map whose sole key is `$bytes` is reserved and MUST have exactly one valid
base64url text value. Other maps MUST NOT use `$bytes` as a key.

## 3. Canonical value encoding

`canon(v)` is UTF-8 encoded canonical JSON with these rules:

1. no insignificant whitespace;
2. literals are exactly `null`, `false`, and `true`;
3. integers use the shortest decimal form, with `0` as the only zero form;
4. text uses JSON escaping only where required, with lowercase `u` escapes;
5. map keys are sorted by their normalized UTF-8 byte sequence;
6. list order is preserved;
7. bytes use the reserved representation above;
8. the encoding ends without a newline.

Golden vectors MUST cover all primitive types, escape boundaries, negative
integers, nested maps, normalization, and bytes.

The public conformance fixture hashes its canonical vectors with the
test-only domain `afterimage/test-vector/1` unless a vector explicitly supplies
another domain. This domain is not used by world objects.

`H(domain, parts...)` means SHA-256 over:

```text
UTF8(domain) || 0x00 || uleb128(len(part1)) || part1 || ...
```

and is rendered as lowercase `sha256:` followed by 64 hexadecimal digits.
Different semantic objects MUST use different domain strings.

## 4. Events

An event body has the following logical fields:

```text
EventBody {
  topic: Text,
  at: Int,
  payload: Value,
  parents: List<EventId>,
  origin: Origin
}

Origin := Base { kind: "base", source: Text, sequence: Int }
        | Derived {
            kind: "derived", rule: RuleId, binding: Bytes, ordinal: Int
          }
        | Player {
            kind: "player",
            branch_parent: BranchId,
            ordinal: Int,
            supersedes: EventId | null
          }
```

Requirements:

- `topic` MUST be non-empty and at most 128 UTF-8 bytes.
- `parents` MUST be duplicate-free and sorted by raw digest bytes.
- `sequence` and `ordinal` MUST be non-negative.
- `payload` canonical encoding MUST fit the bundle's declared value-size limit.
- The event body is immutable.

The event ID is:

```text
EventId = H("afterimage/event/1", canon(EventBody))
```

An input event carrying a mismatched ID is invalid. Content-equal events with
different provenance are deliberately different events.

### 4.1 Parent validity

Every parent MUST exist in the same branch's active fixed point. Base events
MAY cite only base parents. A base-event parent graph MUST be acyclic.

For any event with parents:

```text
event.at >= max(parent.at)
```

Equal logical times are permitted. Parent order, not time alone, expresses
causality.

## 5. Programs and rules

A program is a finite sequence of strata. Each stratum contains rules with
unique IDs.

```text
Program {
  semantics: "cre/0.1",
  strata: List<Stratum>
}

Stratum {
  index: Int,
  rules: List<Rule>
}
```

Stratum indices MUST start at zero and be contiguous. Rules are declarative;
their file order does not select a winner.

### 5.1 Rule shape

```text
Rule {
  id: Text,
  positive: List<PositiveClause>,
  negative: List<NegativeClause>,
  aggregate: List<AggregateClause>,
  distinct: List<List<Alias>>,
  guard: Expr,
  emit: List<Emission>
}
```

A positive clause binds an alias to every active event matching a topic and
predicate. A rule MUST have at least one positive clause. Aliases are unique
within the rule.

A negative clause asserts that no matching event exists. An aggregate clause
computes `count`, `sum`, `min`, or `max` over matching events. Negative and
aggregate clauses MAY read only events completed in lower strata.

Positive clauses MAY read lower strata and the current stratum. This permits
positive recursion. An emission belongs to the rule's current stratum.

These restrictions make the program stratified and give it a unique least
fixed point.

### 5.2 Matching and bindings

Candidate events for each positive clause are ordered by raw EventId digest.
The Cartesian binding tuples are enumerated in clause order and then EventId
order. An optional `distinct` declaration requires selected aliases to bind
different IDs.

The binding key is:

```text
binding = H("afterimage/binding/1", canon([
  [alias1, event_id1],
  ...
]))
```

where aliases appear in clause order.

A positive-clause predicate MAY refer only to its own alias and aliases from
earlier positive clauses. A negative or aggregate predicate MAY additionally
refer to all positive aliases. Aggregate aliases are local to their clause and
aggregate results are exposed by name through the `aggregate` expression.

Negative queries and aggregates are evaluated against the completed fixed
points of lower strata plus the current positive binding values. Their own
iteration order is EventId order.

### 5.3 Expression language

Normative compiled expressions are JSON arrays whose first item is an opcode.
The 0.1 opcodes are:

```text
["const", Value]
["get", Alias, JsonPointer]
["list", Expr...]
["map", Text, Expr, ...]
["eq", Expr, Expr]       ["ne", Expr, Expr]
["lt", Expr, Expr]       ["le", Expr, Expr]
["gt", Expr, Expr]       ["ge", Expr, Expr]
["and", Expr...]         ["or", Expr...]
["not", Expr]
["add", Expr, Expr]      ["sub", Expr, Expr]
["mul", Expr, Expr]      ["div", Expr, Expr]
["mod", Expr, Expr]
["min", Expr...]         ["max", Expr...]
["concat", Expr...]
["length", Expr]
["contains", Expr, Expr]
["if", Expr, Expr, Expr]
["hash", Text, Expr]
["aggregate", Name]
```

Rules:

- `get` reads from an event represented as a map with `topic`, `at`,
  `payload`, `parents`, `origin`, and `id` fields.
- JSON Pointer follows RFC 6901 escaping. Missing paths are errors, not null.
- Boolean operators short-circuit left to right.
- `div` truncates toward zero; division or modulo by zero is an error.
- `concat` accepts all-text or all-bytes operands, but not a mixture.
- `length` accepts text in Unicode scalar values, bytes, lists, or maps.
- `contains` is membership for lists/maps and substring for text/bytes.
- `hash` returns a bytes value containing the 32 raw SHA-256 bytes using the
  supplied domain and canonical encoding of its value.
- Ordered comparison and `min`/`max` accept only matching `Int`, matching
  `Text`, or matching `Bytes`. Int uses numeric order, Text compares NFC UTF-8
  bytes, and Bytes compares raw bytes. Bool, mixed types, lists, maps, and null
  yield `type_error`.
- Type errors invalidate the evaluation. There are no implicit conversions.

The authoring language MAY provide nicer syntax, but distributed programs use
only this compiled representation.

### 5.4 Emissions

An emission specifies expressions for `topic`, `at`, `payload`, and additional
parent IDs. Bound positive-clause events are always parents. Explicit parents
are unioned, deduplicated, and sorted.

For emission ordinal `o`, the origin is:

```text
Derived {
  rule: rule.id,
  binding: raw(binding),
  ordinal: o
}
```

If evaluating the same emission yields an existing EventId, it is a no-op.
Two different bodies that accidentally claim one ID constitute a hash
collision and MUST stop evaluation with `digest_collision`.

### 5.5 Compiled JSON schema

The vertical-slice compiled representation fixes the conceptual clauses above
to these JSON shapes:

```json
{
  "id": "relay.propagate",
  "positive": [
    {"alias":"r","topic":"relay.reachable","where":["const",true]},
    {"alias":"e","topic":"relay.edge","where":["eq",["get","r","/payload/node"],["get","e","/payload/from"]]}
  ],
  "negative": [
    {"alias":"x","topic":"relay.blocked","where":["eq",["get","x","/payload/node"],["get","e","/payload/to"]]}
  ],
  "aggregate": [
    {"name":"seen","op":"count","alias":"s","topic":"relay.reachable","where":["const",true]}
  ],
  "distinct": [["r","e"]],
  "guard": ["const",true],
  "emit": [
    {
      "topic": ["const","relay.reachable"],
      "at": ["max",["get","r","/at"],["get","e","/at"]],
      "payload": ["map","node",["get","e","/payload/to"]],
      "parents": []
    }
  ]
}
```

Rules:

- all seven rule keys shown above are required;
- `topic` is an exact non-empty text match; wildcards do not exist in 0.1;
- `where` defaults nowhere and must evaluate to Bool;
- every alias, rule ID, and aggregate name is unique in its scope;
- each `distinct` inner list has at least two positive aliases and requires all
  corresponding event IDs to differ;
- aggregate `op` is `count`, `sum`, `min`, or `max`;
- `count` MUST omit `value`; the other aggregate operations MUST include a
  `value` expression evaluated with the aggregate alias bound;
- `sum` over an empty set is zero; `min` or `max` over an empty set raises
  `empty_aggregate`;
- an emission's `parents` expressions each return one EventId text value;
- bound positive events are parents even when `parents` is empty.

Projection programs reuse the same clause representation:

```json
{
  "id": "public.relays",
  "rows": [
    {
      "positive": [
        {"alias":"r","topic":"relay.reachable","where":["const",true]}
      ],
      "negative": [],
      "aggregate": [],
      "distinct": [],
      "guard": ["const",true],
      "value": ["map","node",["get","r","/payload/node"]],
      "sort": [["get","r","/payload/node"]]
    }
  ]
}
```

Each projection row MUST have at least one positive clause. Projection
negative and aggregate clauses read the complete active fixed point. Explicit
sort values are compared by their canonical byte encoding. Without `sort`, the
canonical output value is the sort key. Equal keys preserve clause/binding
enumeration order.

## 6. Fixed-point evaluation

Given active base events `B` and program `P`:

```text
active := validate_and_sort(B)
for stratum in P.strata:
    repeat:
        additions := empty map EventId -> Event
        for rule in sort_by_utf8(rule.id, stratum.rules):
            for binding in enumerate_bindings(rule, active):
                if negatives_hold(rule, binding, lower_strata(active))
                   and aggregates_bind(rule, binding, lower_strata(active))
                   and eval(rule.guard, binding) == true:
                    for ordinal, emission in enumerate(rule.emit):
                        e := eval_emission(rule, binding, ordinal, emission)
                        validate_event(e, active union additions)
                        additions[e.id] := e
        new := additions minus active.ids
        if new is empty: break
        active := active union new
        charge one firing for each new derived event
    seal current stratum
return active
```

Although the pseudocode defines a stable order for traces and counters, the
semantic result is the least fixed point. An optimized implementation MAY use
indexes, semi-naive evaluation, or parallel computation if its final outputs
and normative counters match.

### 6.1 Resource accounting

The manifest declares:

- `max_base_events`;
- `max_derived_events`;
- `max_bindings_tested`;
- `max_value_bytes`;
- `max_projection_records`.

When a standalone conformance case omits a limit, the exact CRE 0.1 defaults
are `100000`, `1000000`, `10000000`, `1048576`, and `1000000` respectively in
the order above. A bundle case always supplies explicit host-validated limits.
The returned counter object has exactly `base_events`, `derived_events`,
`bindings_tested`, and `projection_records`. `max_value_bytes` is instead a
per-value size ceiling: validate the canonical byte length of each base or
derived event payload before accepting that event. It is not cumulative and
does not add a returned counter field.

Counters are charged in the canonical enumeration above, even if an optimized
implementation does less work internally. Crossing a counted limit yields a
deterministic `resource_exhausted` error naming the first crossed counter.
Exceeding `max_value_bytes` also yields `resource_exhausted`; because this is a
per-value ceiling rather than a counter, its context is empty.

No wall-clock duration is a semantic limit.

## 7. Counterfactual branches

A branch starts from a parent branch and a canonical list of operations.

```text
Operation := Suppress { event: EventId }
           | Inject { topic, at, payload, parents }
           | Replace { event, pointer, value }
           | Retime { event, at }
```

`Replace` and `Retime` are normative sugar:

- suppress the named base event;
- inject a new player event containing the changed body and the old event's
  parent list;
- set `origin.supersedes` to the old event ID.

A plain `Inject` has `origin.supersedes = null`. Player-event ordinals follow
canonical operation order.

Only base or player events may be suppressed or replaced. Derived events MUST
be changed by modifying their causes.

Each case declares allowed operation kinds, topics, JSON pointers, value
bounds, and operation weights. An operation outside that policy makes the
witness invalid before replay.

Operations are canonicalized by `(kind_rank, target_or_body_digest)` and MUST
not conflict. Suppressing the same event twice or replacing an already
suppressed event is invalid.

The ranks are `suppress = 0`, `replace = 1`, `retime = 2`, and `inject = 3`.
The secondary key for the first three operations is the target EventId raw
digest. For `inject` it is
`H("afterimage/inject-operation/1", canon(operation_without_kind))`.
Branch identity hashes the sorted high-level operations before sugar
expansion. A target may appear in at most one high-level operation.

```text
BranchId = H("afterimage/branch/1", canon({
  "parent": parent_branch_id,
  "operations": canonical_operations
}))
```

Branch replay discards all derived events, applies operations to the active
base set, validates it, and evaluates from stratum zero. Suppression is closed
over base-event parent edges: if any parent of a base event becomes inactive,
that base event also becomes inactive, recursively. These implicit
suppressions add no operation weight but do contribute to causal footprint.
An injected replacement does not automatically re-parent old base descendants;
such propagation must be expressed by reaction rules or explicit allowed
operations.

The root branch ID is:

```text
H("afterimage/root-branch/1", canon({"bundle": bundle_digest}))
```

For a non-root branch, `parent_branch_id` in the branch formula is the actual
parent branch ID and the operation list is the canonical high-level list.
The supplied base set MUST already be the active base/player set of that
parent after all earlier branch steps; implementations MUST NOT silently apply
the new operations to the original root base. An empty operation list is an
identity transition and returns the supplied parent BranchId unchanged.

Concrete non-root BranchIds are runtime values and cannot be embedded in the
bundle that creates them because every BranchId depends on that bundle's
digest. The bundle/witness layer therefore uses a non-circular
`history:<case-id>` reference and a witness-carried history. The host replays
that history from root and supplies its resulting base set and BranchId to the
CRE. Branch identity itself remains exactly the formula above.

## 8. Projections

A projection is a named, read-only program evaluated after the world fixed
point. It MAY:

- select active events;
- map them through the expression language;
- group lower-stratum events with the defined aggregates;
- sort records by an explicit tuple;
- redact fields by omission or replacement.

A projection MUST NOT emit world events or modify a branch. Its output is a
list of Values.

If `sort` is empty, the canonical output value bytes form the sole sort key.
Equal keys preserve clause/binding enumeration order.

```text
ProjectionDigest = H("afterimage/projection/1", canon(output_list))
```

The distinction between active state and projected observation is intentional
and becomes part of the story. A projection is inspectable program data, not a
privileged host callback.

## 9. Trace digest

Normative trace mode records one item for every newly admitted event:

```json
{
  "stratum": 0,
  "rule": "water.pressure",
  "binding": "sha256:...",
  "ordinal": 0,
  "event": "sha256:..."
}
```

Base events are recorded first with `rule: null`, ordered by a topological sort
whose tie-break is `(at, EventId)`. A base trace item has all five fields and
uses `stratum: -1`, `rule: null`, `binding: null`, and `ordinal: null`.
Derived events follow canonical evaluation order.

```text
TraceDigest = H("afterimage/trace/1", canon(trace_items))
```

The trace digest is an interoperability check and witness field. Verifiers
MUST derive validity from semantic results, not trust a submitted trace.

## 10. Required command protocol

A participant engine MAY use any executable name. Conformance tests invoke it
through this abstract interface:

```text
ENGINE suite FULL_SUITE_JSON
ENGINE protocol CASE_JSON
ENGINE replay WORLD_DIR --branch BRANCH_JSON --projection NAME --json
```

`FULL_SUITE_JSON`, `CASE_JSON`, `WORLD_DIR`, and `BRANCH_JSON` are filesystem
paths, not literal JSON arguments. Input JSON files are UTF-8 CRE canonical
JSON with no final newline.

`suite` emits one `afterimage-conformance-result/0.1` object and is the batch
adapter defined by `conformance_fixture.md`. `protocol` evaluates one case and
emits the multi-record stream below. It never consumes an NDJSON stream of
cases and has no single-result success envelope.

`WORLD_DIR` is an already validated and extracted logical bundle produced by
the supplied `afterimage-kit`. The participant engine MUST still recompute
event IDs, branch IDs, projection digests, and trace digests. It is not
responsible for ZIP security, live authentication, receipt aggregation, or
terminal presentation.

Exit codes:

- `0`: success;
- `2`: invalid input or canonicalization failure;
- `3`: semantic evaluation error;
- `4`: resource exhaustion;
- `5`: internal implementation failure.

Exit classification is phase-based. Failure while opening, decoding, or
canonicalizing the input file exits `2`, regardless of its stable error code.
After a canonical input value has been accepted, any CRE validation or
evaluation failure exits `3`, except `resource_exhausted`, which exits `4`.
Thus schema codes such as `invalid_program` are semantic exits, while an
`invalid_json` or `duplicate_key` raised during file decoding is an input exit.

JSON mode emits NDJSON:

```json
{"type":"ready","semantics":"cre/0.1","bundle":"sha256:..."}
{"type":"projection","index":0,"value":{}}
{"type":"done","branch":"sha256:...","projection":"sha256:...","trace":"sha256:...","counters":{}}
```

On failure exactly one terminal `error` record is emitted with stable `code`,
`message`, and optional structured `context`. Human prose is informative.
`code` is normative. `context` is normative when a golden fixture requires its
exact value; otherwise it is optional diagnostic detail and MAY be omitted.

The CRE 0.1 error-code vocabulary is:

```text
causal_cycle digest_collision division_by_zero duplicate_event duplicate_key
empty_aggregate event_id_mismatch integer_overflow invalid_base invalid_bytes
invalid_event invalid_expression invalid_fixture invalid_id invalid_json
invalid_length invalid_limit invalid_operation invalid_origin invalid_parent
invalid_pointer invalid_program invalid_projection invalid_rule invalid_text
invalid_time missing_parent missing_path operation_conflict resource_exhausted
type_error unknown_aggregate unknown_alias unknown_opcode
```

The public `full-suite.json`, `golden.json`, and `check.py` localize suite
differences without distributing a reference evaluator. The included reference
evaluators expose the same record contract as `protocol CASE_JSON` before the
bundle extractor exists. `afterimage-kit` adapts extracted `WORLD_DIR` to this
frozen stream; that packaging adapter does not change CRE evaluation semantics.

## 11. Conformance suite

CRE 0.1 is not ready for release freeze until golden fixtures cover:

- canonical values and digest domains;
- Unicode normalization and map ordering;
- base parent validation and same-time parents (the cycle guard is
  defense-in-depth because content-addressed parent IDs make a valid cyclic
  fixture require a SHA-256 fixed point);
- positive joins and duplicate suppression;
- positive recursion to a fixed point;
- stratified negation;
- all aggregate operators, including empty-set behavior;
- arithmetic and type errors;
- branch operation canonicalization and sugar expansion;
- derived-provenance invalidation after suppression;
- projection ordering and digest;
- every resource limit at boundary minus one, boundary, and plus one;
- required NDJSON protocol and exit codes.

At least two independently implemented evaluators MUST pass the same fixtures
before full content authoring begins.

## 12. Deferred 0.2 review

- Alternative empty-set behavior for `min` and `max`. CRE 0.1 raises
  `empty_aggregate` as fixed in section 5.5.
- Allowing player-injected events to cite derived parents. CRE 0.1 accepts only
  parents present in the replay base set, so derived parents are impossible.
- Whether large later bundles require a binary compiled-rule representation.
  JSON remains normative for the slice.
- Alternative counter models. CRE 0.1 charges every complete positive binding
  before negative clauses, aggregates, and the guard are evaluated, including
  bindings later rejected by those checks.
