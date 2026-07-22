# LENS 0.1 bounded law checker

Status: normative for `LENS.001`.

LENS 0.1 is a typed, total, first-order bidirectional pipeline. Its slice
program contains an explicit complement schema, a `get` pipeline from civic
source to route target, and a `put` pipeline from current source plus edited
target back to source. There is no recursion, loop, dynamic table, or host
callback.

The complement cell types in the slice are `option-text`, `text-list`, and
`text`. `Two Addresses` admits `unit`, `provenance`, and `boundary_street`.
The first two preserve fields the route view cannot express. The third retains
which civic street produced the deliberately ambiguous `Q-BOUND/0` route.

The supported get operations are `forward_address(addresses)` and
`encode_entrance(entrances)`. Put uses
`reverse_address(addresses, disambiguation)`,
`decode_entrance(entrances)`, and `restore(fields)`. Tables are immutable
public contract values. Normalized street equality lowercases, trims, and
collapses whitespace. Reverse ambiguity first uses the declared complement;
otherwise it chooses the canonical normalized civic address.

The bounded domain contains 72 civic sources and 15 valid target edits. For
every source and valid edit the verifier checks:

```text
PutGet:    get(put(s, v)) = v
GetPut:    put(s, get(s)) = s
Stability: put(put(s, v), v) = put(s, v)
Provenance: unit and provenance remain unchanged by target edits
```

Get and valid Put must be total. Named invalid targets must return failure and
must not mutate source or complement. Sources and edits are checked in
canonical CRE byte order; the first failed law is the stable diagnostic with
source, edit, expected value, and observation.

Only a lawful program is scored by
`(program_nodes, auxiliary_schema_cells, worst_reductions)`. Nodes recursively
count AST containers and leaves. Auxiliary cells are declared complement
cells. Reductions count public-table rows inspected by one get or put. Static
byte/node/cell limits and a dynamic reduction limit are public and mandatory.
