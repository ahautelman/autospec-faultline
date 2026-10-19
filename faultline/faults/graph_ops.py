"""Small graph and predicate helpers shared by the fault injectors and mutations."""

from __future__ import annotations

import copy

from faultline.graph import AbstractGraph, Edge, Vertex
from faultline.spectrl.predicates import Empty, Or, Predicate, Universe


def union(left: Predicate, right: Predicate) -> Predicate:
    if isinstance(left, Universe) or isinstance(right, Universe):
        return Universe()
    if isinstance(left, Empty):
        return right
    if isinstance(right, Empty):
        return left
    if left == right:
        return left
    return Or(left, right)


def vertex_named(graph: AbstractGraph, name: str) -> Vertex:
    for vertex in graph.vertices:
        if vertex.name == name:
            return vertex
    raise ValueError(f"Graph has no vertex named {name!r}")


def find_edge(graph: AbstractGraph, source_name: str, target_name: str) -> Edge | None:
    for edge in graph.edges:
        if edge.source.name == source_name and edge.target.name == target_name:
            return edge
    return None


def edge_named(graph: AbstractGraph, source_name: str, target_name: str) -> Edge:
    matches = [
        edge
        for edge in graph.edges
        if edge.source.name == source_name and edge.target.name == target_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one edge {source_name!r}->{target_name!r}, found {len(matches)}"
        )
    return matches[0]


def copy_without_vertex(graph: AbstractGraph, removed_name: str) -> AbstractGraph:
    vertex_named(graph, removed_name)
    cloned = AbstractGraph()
    mapping: dict[str, Vertex] = {}
    for vertex in graph.vertices:
        if vertex.name != removed_name:
            mapping[vertex.name] = cloned.add_vertex(
                vertex.name, copy.deepcopy(vertex.region)
            )
    for edge in graph.edges:
        if edge.source.name == removed_name or edge.target.name == removed_name:
            continue
        cloned.add_edge(
            mapping[edge.source.name],
            mapping[edge.target.name],
            copy.deepcopy(edge.safe),
        )
    if graph.source is not None:
        if graph.source.name == removed_name:
            raise ValueError("Cannot remove graph source")
        cloned.source = mapping[graph.source.name]
    if graph.target is not None:
        if graph.target.name == removed_name:
            raise ValueError("Cannot remove graph target")
        cloned.target = mapping[graph.target.name]
    return cloned
