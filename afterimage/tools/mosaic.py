#!/usr/bin/env python3
"""D4 graph-certificate validation for the MOSAIC vertical-slice family."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "python"))

import cre


TRANSFORMS = {
    "r0": lambda x, y: (x, y),
    "r90": lambda x, y: (-y, x),
    "r180": lambda x, y: (-x, -y),
    "r270": lambda x, y: (y, -x),
    "mx": lambda x, y: (-x, y),
    "mxr90": lambda x, y: (-y, -x),
    "mxr180": lambda x, y: (x, -y),
    "mxr270": lambda x, y: (y, x),
}


class MosaicError(Exception):
    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise MosaicError(code, message, context)


def integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a.encode("utf-8") < b.encode("utf-8") else (b, a)


def validate_global(value: Any, width: int, height: int) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"vertices", "edges"}, "mosaic_graph", "global graph fields are invalid")
    expected_count = width * height
    require(isinstance(value["vertices"], list) and len(value["vertices"]) == expected_count, "mosaic_graph", "global graph has the wrong vertex count")
    vertices: dict[str, tuple[int, int]] = {}
    coordinates: set[tuple[int, int]] = set()
    for index, item in enumerate(value["vertices"]):
        require(isinstance(item, dict) and set(item) == {"id", "x", "y"}, "mosaic_graph", "global vertex fields are invalid", index=index)
        expected_id = f"v{index}"
        require(item["id"] == expected_id and integer(item["x"]) and integer(item["y"]), "mosaic_graph", "global vertex identity is not canonical", index=index)
        require((item["x"], item["y"]) == (index % width, index // width), "mosaic_graph", "global vertices must be row-major", index=index)
        vertices[item["id"]] = (item["x"], item["y"])
        coordinates.add((item["x"], item["y"]))
    require(len(coordinates) == expected_count, "mosaic_graph", "global coordinates are duplicated")
    expected_edges = width * (height - 1) + height * (width - 1)
    require(isinstance(value["edges"], list) and len(value["edges"]) == expected_edges, "mosaic_graph", "global graph has the wrong edge count")
    edges: dict[tuple[str, str], tuple[str, str]] = {}
    for index, item in enumerate(value["edges"]):
        require(isinstance(item, dict) and set(item) == {"a", "b", "material", "level"}, "mosaic_graph", "global edge fields are invalid", index=index)
        a, b = item["a"], item["b"]
        require(a in vertices and b in vertices and isinstance(item["material"], str) and isinstance(item["level"], str), "mosaic_graph", "global edge value is invalid", index=index)
        ax, ay = vertices[a]
        bx, by = vertices[b]
        require(abs(ax - bx) + abs(ay - by) == 1, "mosaic_graph", "global edge is not grid-adjacent", index=index)
        key = edge_key(a, b)
        require(key not in edges, "mosaic_graph", "global edge is duplicated", index=index)
        edges[key] = (item["material"], item["level"])
    for y in range(height):
        for x in range(width):
            here = f"v{y * width + x}"
            if x + 1 < width:
                require(edge_key(here, f"v{y * width + x + 1}") in edges, "mosaic_graph", "global horizontal edge is missing")
            if y + 1 < height:
                require(edge_key(here, f"v{(y + 1) * width + x}") in edges, "mosaic_graph", "global vertical edge is missing")
    return {"value": value, "vertices": vertices, "edges": edges}


def require_coordinate_checksum(graph: dict[str, Any], width: int) -> None:
    for (a, b), attributes in graph["edges"].items():
        left, right = sorted((int(a[1:]), int(b[1:])))
        ax, ay = graph["vertices"][a]
        bx, by = graph["vertices"][b]
        axis = "h" if ay == by else "v"
        expected = (f"{axis}-{left:02d}-{right:02d}", str((left + right) % 3))
        require(attributes == expected, "mosaic_checksum", "global edge violates the public coordinate checksum", edge=[a, b], expected=list(expected))


def graph_encoding(graph: dict[str, Any], transform_name: str, width: int, height: int) -> list[list[str]]:
    transform = TRANSFORMS[transform_name]
    transformed = {vertex: transform(*coord) for vertex, coord in graph["vertices"].items()}
    min_x = min(x for x, _y in transformed.values())
    min_y = min(y for _x, y in transformed.values())
    normalized = {vertex: (x - min_x, y - min_y) for vertex, (x, y) in transformed.items()}
    require(set(normalized.values()) == {(x, y) for y in range(height) for x in range(width)}, "mosaic_graph", "transformed graph does not fill the declared grid")
    labels = {vertex: f"v{y * width + x}" for vertex, (x, y) in normalized.items()}
    encoded = []
    for (a, b), (material, level) in graph["edges"].items():
        na, nb = edge_key(labels[a], labels[b])
        encoded.append([na, nb, material, level])
    return sorted(encoded, key=lambda item: tuple(part.encode("utf-8") for part in item))


def require_canonical(graph: dict[str, Any], width: int, height: int) -> None:
    current = graph_encoding(graph, "r0", width, height)
    alternatives = []
    for name in TRANSFORMS:
        try:
            alternatives.append(graph_encoding(graph, name, width, height))
        except MosaicError as exc:
            if exc.code != "mosaic_graph":
                raise
    best = min(alternatives, key=cre.canonical_bytes)
    require(cre.same_value(current, best), "mosaic_noncanonical", "global graph orientation is not the canonical D4 representative")


def validate_fragment(value: Any) -> dict[str, Any]:
    base_keys = {"id", "weight", "vertices", "edges"}
    timestamped = isinstance(value, dict) and set(value) == base_keys | {"time_offset"}
    require(isinstance(value, dict) and set(value) in (base_keys, base_keys | {"time_offset"}), "invalid_case", "MOSAIC fragment fields are invalid")
    require(isinstance(value["id"], str) and value["id"] and integer(value["weight"]) and value["weight"] >= 0, "invalid_case", "MOSAIC fragment identity is invalid")
    if timestamped:
        require(integer(value["time_offset"]), "invalid_case", "MOSAIC fragment time offset is invalid")
    require(isinstance(value["vertices"], list) and 4 <= len(value["vertices"]) <= 6, "invalid_case", "MOSAIC fragment vertex count is invalid")
    vertices: dict[str, tuple[int, int]] = {}
    timestamps: dict[str, int] = {}
    for item in value["vertices"]:
        expected_vertex_fields = {"name", "x", "y", "observed_at"} if timestamped else {"name", "x", "y"}
        require(isinstance(item, dict) and set(item) == expected_vertex_fields, "invalid_case", "MOSAIC local vertex fields are invalid")
        require(isinstance(item["name"], str) and item["name"] not in vertices and integer(item["x"]) and integer(item["y"]), "invalid_case", "MOSAIC local vertex is invalid")
        if timestamped:
            require(integer(item["observed_at"]), "invalid_case", "MOSAIC local timestamp is invalid")
            timestamps[item["name"]] = item["observed_at"]
        vertices[item["name"]] = (item["x"], item["y"])
    require(len(set(vertices.values())) == len(vertices), "invalid_case", "MOSAIC local coordinates are duplicated")
    require(isinstance(value["edges"], list), "invalid_case", "MOSAIC fragment edges must be a list")
    edges: dict[tuple[str, str], tuple[str, str]] = {}
    for item in value["edges"]:
        require(isinstance(item, dict) and set(item) == {"a", "b", "material", "level"}, "invalid_case", "MOSAIC local edge fields are invalid")
        require(item["a"] in vertices and item["b"] in vertices and isinstance(item["material"], str) and isinstance(item["level"], str), "invalid_case", "MOSAIC local edge is invalid")
        key = edge_key(item["a"], item["b"])
        require(key not in edges, "invalid_case", "MOSAIC local edge is duplicated")
        edges[key] = (item["material"], item["level"])
    return {"id": value["id"], "weight": value["weight"], "vertices": vertices, "edges": edges, "time_offset": value.get("time_offset"), "timestamps": timestamps}


def fragment_fingerprint(fragment: dict[str, Any]) -> bytes:
    """Canonical D4/translation-invariant attributed-fragment identity."""

    encodings = []
    for transform in TRANSFORMS.values():
        transformed = {name: transform(*coord) for name, coord in fragment["vertices"].items()}
        min_x = min(x for x, _y in transformed.values())
        min_y = min(y for _x, y in transformed.values())
        normalized = {name: (x - min_x, y - min_y) for name, (x, y) in transformed.items()}
        encoded_edges = []
        for (a, b), (material, level) in fragment["edges"].items():
            left, right = sorted((normalized[a], normalized[b]))
            encoded_edges.append([list(left), list(right), material, level])
        encodings.append({
            "vertices": sorted([list(coord) for coord in normalized.values()]),
            "edges": sorted(encoded_edges, key=cre.canonical_bytes),
        })
    return min((cre.canonical_bytes(value) for value in encodings))


def embeddings(fragment: dict[str, Any], graph: dict[str, Any], width: int, height: int, only_transform: str | None = None) -> list[tuple[str, dict[str, str]]]:
    result = []
    names = [only_transform] if only_transform else list(TRANSFORMS)
    coordinate_to_id = {coord: vertex for vertex, coord in graph["vertices"].items()}
    for transform_name in names:
        transform = TRANSFORMS[transform_name]
        transformed = {name: transform(*coord) for name, coord in fragment["vertices"].items()}
        for tx in range(-3, width + 3):
            for ty in range(-3, height + 3):
                mapping = {name: coordinate_to_id.get((x + tx, y + ty)) for name, (x, y) in transformed.items()}
                if any(vertex is None for vertex in mapping.values()) or len(set(mapping.values())) != len(mapping):
                    continue
                if all(graph["edges"].get(edge_key(mapping[a], mapping[b])) == attributes for (a, b), attributes in fragment["edges"].items()):
                    result.append((transform_name, mapping))
    return result


def geometric_embeddings(fragment: dict[str, Any], graph: dict[str, Any], width: int, height: int) -> list[tuple[str, dict[str, str]]]:
    """Return D4 placements that preserve vertices and adjacency but ignore edge attributes."""

    result = []
    coordinate_to_id = {coord: vertex for vertex, coord in graph["vertices"].items()}
    for transform_name, transform in TRANSFORMS.items():
        transformed = {name: transform(*coord) for name, coord in fragment["vertices"].items()}
        for tx in range(-3, width + 3):
            for ty in range(-3, height + 3):
                mapping = {name: coordinate_to_id.get((x + tx, y + ty)) for name, (x, y) in transformed.items()}
                if any(vertex is None for vertex in mapping.values()) or len(set(mapping.values())) != len(mapping):
                    continue
                if all(edge_key(mapping[a], mapping[b]) in graph["edges"] for a, b in fragment["edges"]):
                    result.append((transform_name, mapping))
    return result


def validate_answer(fragments_value: list[Any], answer: Any, contract: Any) -> dict[str, int]:
    require(
        isinstance(contract, dict) and set(contract) in (
            {"width", "height"}, {"width", "height", "coverage"}, {"width", "height", "completion"},
            {"width", "height", "duplicates"},
            {"width", "height", "landmarks"},
            {"width", "height", "timestamp_gauge"},
            {"width", "height", "ring_recovery"},
            {"width", "height", "weighted_noise"},
            {"width", "height", "layers"},
            {"width", "height", "coverage", "layers"},
        ),
        "invalid_score", "MOSAIC contract fields are invalid",
    )
    width, height = contract["width"], contract["height"]
    require(integer(width) and integer(height) and width > 0 and height > 0, "invalid_score", "MOSAIC dimensions are invalid")
    coverage = contract.get("coverage")
    completion = contract.get("completion")
    duplicates = contract.get("duplicates")
    landmarks = contract.get("landmarks")
    timestamp_gauge = contract.get("timestamp_gauge")
    ring_recovery = contract.get("ring_recovery")
    weighted_noise = contract.get("weighted_noise")
    layers = contract.get("layers")
    if coverage is not None:
        require(isinstance(coverage, dict) and set(coverage) == {"minimum_shared_edges", "shared_edge_multiplicity"}, "invalid_score", "MOSAIC coverage contract fields are invalid")
        require(integer(coverage["minimum_shared_edges"]) and coverage["minimum_shared_edges"] > 0, "invalid_score", "MOSAIC shared-edge minimum is invalid")
        require(integer(coverage["shared_edge_multiplicity"]) and coverage["shared_edge_multiplicity"] >= 2, "invalid_score", "MOSAIC shared-edge multiplicity is invalid")
        maximum_edges = width * (height - 1) + height * (width - 1)
        require(coverage["minimum_shared_edges"] <= maximum_edges, "invalid_score", "MOSAIC shared-edge minimum exceeds the grid")
    if completion is not None:
        require(isinstance(completion, dict) and set(completion) == {"max_missing_vertices", "max_missing_edges", "edge_checksum"}, "invalid_score", "MOSAIC completion contract fields are invalid")
        require(integer(completion["max_missing_vertices"]) and completion["max_missing_vertices"] > 0, "invalid_score", "MOSAIC missing-vertex bound is invalid")
        require(integer(completion["max_missing_edges"]) and completion["max_missing_edges"] > 0, "invalid_score", "MOSAIC missing-edge bound is invalid")
        require(completion["edge_checksum"] == "coordinate-v1", "invalid_score", "MOSAIC completion checksum is unsupported")
    if duplicates is not None:
        require(isinstance(duplicates, dict) and set(duplicates) == {"fingerprint"} and duplicates["fingerprint"] == "d4-attributed-v1", "invalid_score", "MOSAIC duplicate contract is unsupported")
    if landmarks is not None:
        require(isinstance(landmarks, dict) and set(landmarks) == {"anchors"} and isinstance(landmarks["anchors"], list) and landmarks["anchors"], "invalid_score", "MOSAIC landmark contract is invalid")
    if timestamp_gauge is not None:
        require(isinstance(timestamp_gauge, dict) and timestamp_gauge == {"formula": "row-major-v1"}, "invalid_score", "MOSAIC timestamp gauge is unsupported")
    if ring_recovery is not None:
        require(isinstance(ring_recovery, dict) and set(ring_recovery) == {"cycle", "max_missing_edges"}, "invalid_score", "MOSAIC ring-recovery contract fields are invalid")
        cycle = ring_recovery["cycle"]
        require(isinstance(cycle, list) and len(cycle) >= 4 and len(set(cycle)) == len(cycle), "invalid_score", "MOSAIC ring cycle is invalid")
        require(all(isinstance(vertex, str) for vertex in cycle), "invalid_score", "MOSAIC ring vertex is invalid")
        require(integer(ring_recovery["max_missing_edges"]) and ring_recovery["max_missing_edges"] > 0, "invalid_score", "MOSAIC ring missing-edge bound is invalid")
    if weighted_noise is not None:
        require(isinstance(weighted_noise, dict) and set(weighted_noise) == {"minimum_support_margin", "anchors"}, "invalid_score", "MOSAIC weighted-noise contract fields are invalid")
        require(integer(weighted_noise["minimum_support_margin"]) and weighted_noise["minimum_support_margin"] > 0, "invalid_score", "MOSAIC weighted-noise margin is invalid")
        require(isinstance(weighted_noise["anchors"], list) and weighted_noise["anchors"], "invalid_score", "MOSAIC weighted-noise anchors are invalid")
    if layers is not None:
        require(isinstance(layers, dict) and set(layers) == {"values", "portals", "minimum_edges_per_layer", "portal_material"}, "invalid_score", "MOSAIC layer contract fields are invalid")
        require(isinstance(layers["values"], list) and len(layers["values"]) == 2 and len(set(layers["values"])) == 2 and all(isinstance(value, str) and value for value in layers["values"]), "invalid_score", "MOSAIC layer values are invalid")
        require(layers["values"] == sorted(layers["values"], key=lambda value: value.encode("utf-8")), "invalid_score", "MOSAIC layer values must be canonical")
        require(isinstance(layers["portals"], list) and layers["portals"] and len(set(layers["portals"])) == len(layers["portals"]) and all(isinstance(value, str) and value for value in layers["portals"]), "invalid_score", "MOSAIC layer portals are invalid")
        require(integer(layers["minimum_edges_per_layer"]) and layers["minimum_edges_per_layer"] > 0, "invalid_score", "MOSAIC layer edge minimum is invalid")
        require(isinstance(layers["portal_material"], str) and layers["portal_material"], "invalid_score", "MOSAIC portal material is invalid")
    answer_fields = {"global", "used", "unused", "missing"} if completion is not None or ring_recovery is not None else {"global", "used", "unused"}
    require(isinstance(answer, dict) and set(answer) == answer_fields, "mosaic_schema", "MOSAIC answer fields are invalid")
    graph = validate_global(answer["global"], width, height)
    require_canonical(graph, width, height)
    if layers is not None:
        declared_layers = set(layers["values"])
        portals = set(layers["portals"])
        require(portals <= set(graph["vertices"]), "mosaic_layer", "layer contract names an unknown portal")
        by_layer = {value: {edge for edge, attributes in graph["edges"].items() if attributes[1] == value} for value in declared_layers}
        require(all(attributes[1] in declared_layers for attributes in graph["edges"].values()), "mosaic_layer", "global edge uses an undeclared layer")
        require(all(len(edges) >= layers["minimum_edges_per_layer"] for edges in by_layer.values()), "mosaic_layer", "a declared layer has too few edges")
        incident: dict[str, set[str]] = {vertex: set() for vertex in graph["vertices"]}
        materials: dict[str, set[str]] = {vertex: set() for vertex in graph["vertices"]}
        for (a, b), (material, level) in graph["edges"].items():
            incident[a].add(level)
            incident[b].add(level)
            materials[a].add(material)
            materials[b].add(material)
        mixed = {vertex for vertex, values in incident.items() if values == declared_layers}
        require(mixed == portals, "mosaic_layer", "declared portals do not exactly match cross-layer junctions")
        require(all(layers["portal_material"] in materials[vertex] for vertex in portals), "mosaic_layer", "cross-layer junction lacks the portal material")
        for value, edges in by_layer.items():
            vertices = {vertex for edge in edges for vertex in edge}
            seen = {next(iter(vertices))}
            while True:
                expanded = seen | {right for left, right in edges if left in seen} | {left for left, right in edges if right in seen}
                if expanded == seen:
                    break
                seen = expanded
            require(seen == vertices, "mosaic_layer", "a declared layer is disconnected", layer=value)
    if completion is not None:
        require_coordinate_checksum(graph, width)
    if ring_recovery is not None:
        cycle = ring_recovery["cycle"]
        require(all(vertex in graph["vertices"] for vertex in cycle), "invalid_score", "MOSAIC ring names an unknown vertex")
        cycle_edges = {edge_key(cycle[index], cycle[(index + 1) % len(cycle)]) for index in range(len(cycle))}
        require(len(cycle_edges) == len(cycle) and cycle_edges <= set(graph["edges"]), "invalid_score", "MOSAIC ring is not a simple cycle in the global graph")
    fragments = [validate_fragment(value) for value in fragments_value]
    by_id = {fragment["id"]: fragment for fragment in fragments}
    require(len(by_id) == len(fragments), "invalid_case", "MOSAIC fragment IDs are duplicated")
    if timestamp_gauge is not None:
        require(all(fragment["time_offset"] is not None and len(fragment["timestamps"]) >= 2 for fragment in fragments), "invalid_case", "MOSAIC timestamp-gauge fragments need at least two timed vertices")

    def timestamp_consistent(fragment: dict[str, Any], mapping: dict[str, str]) -> bool:
        if timestamp_gauge is None:
            return True
        return all(
            observed_at == fragment["time_offset"] + graph["vertices"][mapping[local]][1] * width + graph["vertices"][mapping[local]][0]
            for local, observed_at in fragment["timestamps"].items()
        )
    anchors_by_fragment: dict[str, dict[str, str]] = {}
    if landmarks is not None:
        for index, anchor in enumerate(landmarks["anchors"]):
            require(isinstance(anchor, dict) and set(anchor) == {"fragment", "local", "global"}, "invalid_score", "MOSAIC landmark anchor fields are invalid", index=index)
            fragment_id = anchor["fragment"]
            require(fragment_id in by_id and anchor["local"] in by_id[fragment_id]["vertices"] and anchor["global"] in graph["vertices"], "invalid_score", "MOSAIC landmark anchor target is invalid", index=index)
            local = anchors_by_fragment.setdefault(fragment_id, {})
            require(anchor["local"] not in local, "invalid_score", "MOSAIC landmark anchor is duplicated", index=index)
            local[anchor["local"]] = anchor["global"]
    if weighted_noise is not None:
        for index, anchor in enumerate(weighted_noise["anchors"]):
            require(isinstance(anchor, dict) and set(anchor) == {"fragment", "local", "global"}, "invalid_score", "MOSAIC weighted-noise anchor fields are invalid", index=index)
            fragment_id = anchor["fragment"]
            require(fragment_id in by_id and anchor["local"] in by_id[fragment_id]["vertices"] and anchor["global"] in graph["vertices"], "invalid_score", "MOSAIC weighted-noise anchor target is invalid", index=index)
            local = anchors_by_fragment.setdefault(fragment_id, {})
            require(anchor["local"] not in local, "invalid_score", "MOSAIC weighted-noise anchor is duplicated", index=index)
            local[anchor["local"]] = anchor["global"]
    require(isinstance(answer["used"], list) and isinstance(answer["unused"], list), "mosaic_schema", "MOSAIC classification must use lists")
    used_ids: set[str] = set()
    covered_vertices: set[str] = set()
    covered_edges: set[tuple[str, str]] = set()
    edge_support: dict[tuple[str, str], set[str]] = {}
    for index, item in enumerate(answer["used"]):
        require(isinstance(item, dict) and set(item) == {"fragment", "transform", "mapping"}, "mosaic_mapping", "used-fragment fields are invalid", index=index)
        fragment_id = item["fragment"]
        require(fragment_id in by_id and fragment_id not in used_ids and item["transform"] in TRANSFORMS, "mosaic_mapping", "used-fragment identity is invalid", index=index)
        fragment = by_id[fragment_id]
        require(isinstance(item["mapping"], list), "mosaic_mapping", "fragment mapping must be a list", index=index)
        mapping: dict[str, str] = {}
        for pair in item["mapping"]:
            require(isinstance(pair, dict) and set(pair) == {"local", "global"} and pair["local"] in fragment["vertices"] and pair["global"] in graph["vertices"] and pair["local"] not in mapping, "mosaic_mapping", "fragment mapping entry is invalid", fragment=fragment_id)
            mapping[pair["local"]] = pair["global"]
        require(set(mapping) == set(fragment["vertices"]) and len(set(mapping.values())) == len(mapping), "mosaic_mapping", "fragment mapping is incomplete or non-injective", fragment=fragment_id)
        valid = any(cre.same_value(candidate, mapping) for _name, candidate in embeddings(fragment, graph, width, height, item["transform"]))
        require(valid, "mosaic_mapping", "fragment transform or edge correspondence is invalid", fragment=fragment_id)
        require(all(mapping[local] == target for local, target in anchors_by_fragment.get(fragment_id, {}).items()), "mosaic_landmark", "used fragment violates a public landmark anchor", fragment=fragment_id)
        require(timestamp_consistent(fragment, mapping), "mosaic_timestamp", "used fragment violates the public timestamp gauge", fragment=fragment_id)
        used_ids.add(fragment_id)
        covered_vertices.update(mapping.values())
        fragment_edges = {edge_key(mapping[a], mapping[b]) for a, b in fragment["edges"]}
        covered_edges.update(fragment_edges)
        for edge in fragment_edges:
            edge_support.setdefault(edge, set()).add(fragment_id)
    fingerprints = {fragment_id: fragment_fingerprint(fragment) for fragment_id, fragment in by_id.items()}
    duplicate_groups: dict[bytes, set[str]] = {}
    for fragment_id, fingerprint in fingerprints.items():
        duplicate_groups.setdefault(fingerprint, set()).add(fragment_id)
    if duplicates is not None:
        require(any(len(group) > 1 for group in duplicate_groups.values()), "invalid_case", "MOSAIC duplicate case has no duplicate fragment group")
    unused_ids: set[str] = set()
    for index, item in enumerate(answer["unused"]):
        duplicate_claim = isinstance(item, dict) and item.get("reason") == "duplicate"
        landmark_claim = isinstance(item, dict) and item.get("reason") == "landmark_conflict"
        timestamp_claim = isinstance(item, dict) and item.get("reason") == "timestamp_conflict"
        outvoted_claim = isinstance(item, dict) and item.get("reason") == "outvoted"
        fields = {"fragment", "reason", "duplicate_of"} if duplicate_claim else {"fragment", "reason"}
        require(isinstance(item, dict) and set(item) == fields, "mosaic_classification", "unused-fragment fields are invalid", index=index)
        fragment_id = item["fragment"]
        require(fragment_id in by_id and fragment_id not in unused_ids, "mosaic_classification", "unused-fragment classification is invalid", index=index)
        if duplicate_claim:
            require(duplicates is not None and item["duplicate_of"] in used_ids and fingerprints[fragment_id] == fingerprints[item["duplicate_of"]], "mosaic_duplicate", "duplicate rejection does not name an equivalent used fragment", fragment=fragment_id)
        elif landmark_claim:
            anchors = anchors_by_fragment.get(fragment_id, {})
            candidates = embeddings(by_id[fragment_id], graph, width, height)
            require(landmarks is not None and anchors and candidates, "mosaic_landmark", "landmark rejection lacks geometric support", fragment=fragment_id)
            require(not any(all(mapping[local] == target for local, target in anchors.items()) for _transform, mapping in candidates), "mosaic_landmark", "unused fragment has a landmark-consistent embedding", fragment=fragment_id)
        elif timestamp_claim:
            candidates = embeddings(by_id[fragment_id], graph, width, height)
            require(timestamp_gauge is not None and candidates, "mosaic_timestamp", "timestamp rejection lacks geometric support", fragment=fragment_id)
            require(not any(timestamp_consistent(by_id[fragment_id], mapping) for _transform, mapping in candidates), "mosaic_timestamp", "unused fragment has a timestamp-consistent embedding", fragment=fragment_id)
        elif outvoted_claim:
            require(weighted_noise is not None, "mosaic_weight", "outvoted fragment requires a weighted-noise contract", fragment=fragment_id)
            supported = False
            anchors = anchors_by_fragment.get(fragment_id, {})
            for _transform, mapping in geometric_embeddings(by_id[fragment_id], graph, width, height):
                if not all(mapping[local] == target for local, target in anchors.items()):
                    continue
                conflicts = []
                for (a, b), attributes in by_id[fragment_id]["edges"].items():
                    global_edge = edge_key(mapping[a], mapping[b])
                    if graph["edges"][global_edge] != attributes:
                        conflicts.append(global_edge)
                if not conflicts:
                    continue
                if all(
                    sum(by_id[supporter]["weight"] for supporter in edge_support.get(edge, set())) - by_id[fragment_id]["weight"] >= weighted_noise["minimum_support_margin"]
                    for edge in conflicts
                ):
                    supported = True
                    break
            require(supported, "mosaic_weight", "unused noisy fragment is not outweighed on every conflicting edge", fragment=fragment_id)
        else:
            require(item["reason"] == "invariant_conflict", "mosaic_classification", "unused-fragment classification is invalid", index=index)
            if weighted_noise is not None:
                anchors = anchors_by_fragment.get(fragment_id, {})
                require(not any(all(mapping[local] == target for local, target in anchors.items()) for _transform, mapping in geometric_embeddings(by_id[fragment_id], graph, width, height)), "mosaic_classification", "geometrically supported noisy fragment must use outvoted classification", fragment=fragment_id)
            require(not embeddings(by_id[fragment_id], graph, width, height), "mosaic_classification", "unused fragment still embeds in submitted graph", fragment=fragment_id)
        unused_ids.add(fragment_id)
    require(not used_ids & unused_ids and used_ids | unused_ids == set(by_id), "mosaic_classification", "every fragment must be classified exactly once")
    if duplicates is not None:
        for group in duplicate_groups.values():
            if len(group) > 1:
                require(len(group & used_ids) == 1, "mosaic_duplicate", "each duplicate group needs exactly one used survivor")
    if completion is None and ring_recovery is None:
        require(covered_vertices == set(graph["vertices"]) and covered_edges == set(graph["edges"]), "mosaic_coverage", "used fragments do not cover the complete global graph")
    elif completion is not None:
        missing = answer["missing"]
        require(isinstance(missing, dict) and set(missing) == {"vertices", "edges"}, "mosaic_omission", "missing-tile certificate fields are invalid")
        require(isinstance(missing["vertices"], list) and all(isinstance(vertex, str) and vertex in graph["vertices"] for vertex in missing["vertices"]), "mosaic_omission", "missing vertex list is invalid")
        require(len(missing["vertices"]) == len(set(missing["vertices"])) and missing["vertices"] == sorted(missing["vertices"], key=lambda value: value.encode("utf-8")), "mosaic_omission", "missing vertices must be unique and canonical")
        declared_edges: list[tuple[str, str]] = []
        require(isinstance(missing["edges"], list), "mosaic_omission", "missing edge list is invalid")
        for index, item in enumerate(missing["edges"]):
            require(isinstance(item, dict) and set(item) == {"a", "b"}, "mosaic_omission", "missing edge fields are invalid", index=index)
            key = edge_key(item["a"], item["b"])
            require(item["a"] == key[0] and item["b"] == key[1] and key in graph["edges"], "mosaic_omission", "missing edge is not canonical or present in the graph", index=index)
            declared_edges.append(key)
        require(len(declared_edges) == len(set(declared_edges)) and declared_edges == sorted(declared_edges, key=lambda edge: (edge[0].encode("utf-8"), edge[1].encode("utf-8"))), "mosaic_omission", "missing edges must be unique and canonical")
        missing_vertices = set(graph["vertices"]) - covered_vertices
        missing_edges = set(graph["edges"]) - covered_edges
        require(set(missing["vertices"]) == missing_vertices and set(declared_edges) == missing_edges, "mosaic_omission", "missing-tile certificate does not match fragment coverage")
        require(0 < len(missing_vertices) <= completion["max_missing_vertices"] and 0 < len(missing_edges) <= completion["max_missing_edges"], "mosaic_omission", "missing-tile certificate exceeds public bounds")
        for vertex in missing_vertices:
            x, y = graph["vertices"][vertex]
            require(0 < x < width - 1 and 0 < y < height - 1, "mosaic_omission", "missing tile must be an interior vertex", vertex=vertex)
        incident = {edge for edge in graph["edges"] if edge[0] in missing_vertices or edge[1] in missing_vertices}
        require(missing_edges == incident, "mosaic_omission", "only edges incident to the missing tile may be inferred")
    else:
        missing = answer["missing"]
        require(isinstance(missing, dict) and set(missing) == {"edges"}, "mosaic_ring", "ring-recovery certificate fields are invalid")
        require(isinstance(missing["edges"], list), "mosaic_ring", "ring-recovery edges must be a list")
        declared_edges: list[tuple[str, str]] = []
        for index, item in enumerate(missing["edges"]):
            require(isinstance(item, dict) and set(item) == {"a", "b"}, "mosaic_ring", "ring-recovery edge fields are invalid", index=index)
            key = edge_key(item["a"], item["b"])
            require(item["a"] == key[0] and item["b"] == key[1] and key in graph["edges"], "mosaic_ring", "ring-recovery edge is invalid", index=index)
            declared_edges.append(key)
        missing_vertices = set(graph["vertices"]) - covered_vertices
        missing_edges = set(graph["edges"]) - covered_edges
        require(not missing_vertices, "mosaic_ring", "ring recovery may not invent a missing vertex")
        require(len(declared_edges) == len(set(declared_edges)) and declared_edges == sorted(declared_edges), "mosaic_ring", "ring-recovery edges must be unique and canonical")
        require(set(declared_edges) == missing_edges, "mosaic_ring", "ring-recovery certificate does not match observed coverage")
        require(0 < len(missing_edges) <= ring_recovery["max_missing_edges"], "mosaic_ring", "ring-recovery omission exceeds its public bound")
        require(missing_edges <= cycle_edges, "mosaic_ring", "only an edge of the published ring may be recovered")
        observed_cycle = cycle_edges - missing_edges
        degrees = {vertex: 0 for vertex in cycle}
        for a, b in observed_cycle:
            degrees[a] += 1
            degrees[b] += 1
        require(sorted(degrees.values()).count(1) == 2 and all(degree in {1, 2} for degree in degrees.values()), "mosaic_ring", "observed ring must be one path closed by the omitted edge")
    if coverage is not None:
        shared = sum(len(supporters) >= coverage["shared_edge_multiplicity"] for supporters in edge_support.values())
        require(shared >= coverage["minimum_shared_edges"], "mosaic_overlap", "used fragments do not provide enough independently shared edges", observed=shared, required=coverage["minimum_shared_edges"])
    certificate_value = {"used": answer["used"], "missing": answer["missing"]} if completion is not None or ring_recovery is not None else answer["used"]
    return {
        "unexplained_weight": sum(by_id[fragment_id]["weight"] for fragment_id in unused_ids),
        "graph_size": len(graph["vertices"]) + len(graph["edges"]),
        "certificate_units": (len(cre.canonical_bytes(certificate_value)) + 63) // 64,
    }
