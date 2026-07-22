# MOSAIC 0.1 graph certificate

Status: normative for the MOSAIC production verifier.

MOSAIC reconstructs one attributed grid from locally named, locally oriented
graph fragments. Local names and compass directions are gauge choices; edge
attribute pairs are invariant.

## Global graph

The submitted graph has `vertices` and `edges`. For the slice contract it is a
complete 3-by-3 orthogonal grid. Vertices are exactly `v0` through `v8` in
row-major coordinate order. Every horizontal and vertical adjacency occurs
once, no diagonal or duplicate edge is allowed, and every edge has two Text
attributes, `material` and `level`.

The eight D4 transforms are `r0`, `r90`, `r180`, `r270`, `mx`, `mxr90`,
`mxr180`, and `mxr270`. For each transform the verifier normalizes coordinates
back to the grid origin, relabels row-major, and canonically encodes sorted
`[a,b,material,level]` edges. The submitted orientation MUST equal the
lexicographically least canonical CRE encoding across all eight transforms.

## Used fragments

A used certificate names the fragment, one D4 transform, and a complete
injective list of local-to-global vertex pairs. After the transform there MUST
be one translation that places every local coordinate on its mapped global
coordinate. Every fragment edge MUST map to a global edge with exactly the
same material/level pair.

The union of used mappings MUST cover all nine global vertices and all twelve
global edges. Coverage prevents a player from submitting a larger explanation
than the fragments justify or omitting an inconvenient global invariant.

An optional public `coverage` contract adds `minimum_shared_edges` and
`shared_edge_multiplicity`. For each mapped global edge, support is counted by
distinct used fragment IDs, never by duplicate local edges. At least the
declared number of global edges MUST reach the declared multiplicity, which is
at least two. This is stricter than union coverage: an edge may remain present
in the reconstructed graph after one corroborating observation is removed,
yet fail the independent-support requirement with `mosaic_overlap`.

`MOSAIC.002` requires two shared edges at multiplicity two. Its north and
south six-junction surveys cover the same middle corridor after different D4
placements. Vertex overlap alone does not satisfy the contract.

## Unused fragments

Every fragment is classified exactly once. The slice reason code is
`invariant_conflict`. It is supported only when exhaustive enumeration of all
eight transforms and all in-grid translations finds no attribute-preserving
embedding into the submitted global graph. A fragment that still embeds
cannot be called a decoy.

## Metrics

Only a valid certificate is scored:

```text
(unexplained_weight, graph_size, certificate_units)
```

`unexplained_weight` sums public weights of unused fragments. `graph_size` is
vertices plus edges. `certificate_units` is the canonical byte length of the
used-certificate list rounded up in 64-byte units. The tuple is packed
lexicographically with public per-case bounds.
