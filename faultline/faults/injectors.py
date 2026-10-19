"""Controlled specification-fault injectors for FaultLine."""

from __future__ import annotations

import copy

from faultline.faults.base import FaultCase, FaultType
from faultline.graph import AbstractGraph, Edge, Vertex
from faultline.spectrl.predicates import Empty, Or, Predicate, Universe


def inject_goal_trap(
    graph: AbstractGraph,
    vertex_name: str,
    trap_region: Predicate,
    *,
    name: str = "goal_trap",
    severity: float | None = None,
) -> FaultCase:
    correct = graph.copy()
    faulty = graph.copy()
    vertex = _vertex_named(faulty, vertex_name)
    vertex.region = _union(vertex.region, copy.deepcopy(trap_region))
    faulty.assert_acyclic()
    return FaultCase(
        name=name,
        fault_type=FaultType.GOAL_TRAP,
        expected_operator="seq",
        correct_graph=correct,
        faulty_graph=faulty,
        description=f"Broadened vertex {vertex_name!r} with an unreachable target region.",
        affected_vertex=vertex_name,
        severity=severity,
    )


def inject_overwide_safety(
    graph: AbstractGraph,
    source_name: str,
    target_name: str,
    extra_safe_region: Predicate,
    *,
    name: str = "overwide_safety",
    severity: float | None = None,
) -> FaultCase:
    correct = graph.copy()
    faulty = graph.copy()
    edge = _edge_named(faulty, source_name, target_name)
    edge.safe = _union(edge.safe, copy.deepcopy(extra_safe_region))
    faulty.assert_acyclic()
    return FaultCase(
        name=name,
        fault_type=FaultType.OVERWIDE_SAFETY,
        expected_operator="seq",
        correct_graph=correct,
        faulty_graph=faulty,
        description=f"Broadened safe region on edge {source_name!r}->{target_name!r}.",
        affected_edge=(source_name, target_name),
        severity=severity,
    )


def inject_unreachable_start(
    graph: AbstractGraph,
    vertex_name: str,
    unreachable_region: Predicate,
    *,
    name: str = "unreachable_start",
    severity: float | None = None,
) -> FaultCase:
    correct = graph.copy()
    faulty = graph.copy()
    vertex = _vertex_named(faulty, vertex_name)
    vertex.region = _union(vertex.region, copy.deepcopy(unreachable_region))
    faulty.assert_acyclic()
    return FaultCase(
        name=name,
        fault_type=FaultType.UNREACHABLE_START,
        expected_operator="past",
        correct_graph=correct,
        faulty_graph=faulty,
        description=f"Broadened vertex {vertex_name!r} with unreachable starting states.",
        affected_vertex=vertex_name,
        severity=severity,
    )


def inject_missing_waypoint(
    graph: AbstractGraph,
    waypoint_name: str,
    *,
    name: str = "missing_waypoint",
    severity: float | None = None,
) -> FaultCase:
    correct = graph.copy()
    waypoint = _vertex_named(graph, waypoint_name)
    incoming = graph.predecessors(waypoint)
    outgoing = graph.successors(waypoint)
    if len(incoming) != 1 or len(outgoing) != 1:
        raise ValueError(
            "Missing-waypoint injection requires exactly one incoming and one outgoing edge "
            f"at {waypoint_name!r}; found {len(incoming)} and {len(outgoing)}"
        )
    before, after = incoming[0], outgoing[0]
    if before.safe != after.safe:
        raise ValueError("Collapsed waypoint edges must have equal safe predicates")
    if graph.source is waypoint or graph.target is waypoint:
        raise ValueError("Cannot remove graph source or target as a waypoint")

    faulty = _copy_without_vertex(graph, waypoint_name)
    source = _vertex_named(faulty, before.source.name)
    target = _vertex_named(faulty, after.target.name)
    if _find_edge(faulty, source.name, target.name) is not None:
        raise ValueError("Direct collapsed edge already exists")
    faulty.add_edge(source, target, copy.deepcopy(before.safe))
    faulty.assert_acyclic()
    return FaultCase(
        name=name,
        fault_type=FaultType.MISSING_WAYPOINT,
        expected_operator="add",
        correct_graph=correct,
        faulty_graph=faulty,
        description=(
            f"Removed waypoint {waypoint_name!r} and collapsed "
            f"{before.source.name!r}->{waypoint_name!r}->{after.target.name!r}."
        ),
        affected_vertex=waypoint_name,
        affected_edge=(before.source.name, after.target.name),
        severity=severity,
    )


def inject_missing_alternative_path(
    graph: AbstractGraph,
    source_name: str,
    target_name: str,
    *,
    name: str = "missing_alternative",
    severity: float | None = None,
) -> FaultCase:
    correct = graph.copy()
    faulty = graph.copy()
    edge = _edge_named(faulty, source_name, target_name)
    faulty.remove_edge(edge)
    faulty.assert_acyclic()
    return FaultCase(
        name=name,
        fault_type=FaultType.MISSING_ALTERNATIVE,
        expected_operator="or",
        correct_graph=correct,
        faulty_graph=faulty,
        description=f"Removed alternative edge {source_name!r}->{target_name!r}.",
        affected_edge=(source_name, target_name),
        severity=severity,
    )


def no_fault_control(graph: AbstractGraph, *, name: str = "no_fault") -> FaultCase:
    return FaultCase(
        name=name,
        fault_type=FaultType.NONE,
        expected_operator=None,
        correct_graph=graph.copy(),
        faulty_graph=graph.copy(),
        description="No fault injected.",
    )


def _union(left: Predicate, right: Predicate) -> Predicate:
    if isinstance(left, Universe) or isinstance(right, Universe):
        return Universe()
    if isinstance(left, Empty):
        return right
    if isinstance(right, Empty):
        return left
    if left == right:
        return left
    return Or(left, right)


def _vertex_named(graph: AbstractGraph, name: str) -> Vertex:
    for vertex in graph.vertices:
        if vertex.name == name:
            return vertex
    raise ValueError(f"Graph has no vertex named {name!r}")


def _find_edge(graph: AbstractGraph, source_name: str, target_name: str) -> Edge | None:
    for edge in graph.edges:
        if edge.source.name == source_name and edge.target.name == target_name:
            return edge
    return None


def _edge_named(graph: AbstractGraph, source_name: str, target_name: str) -> Edge:
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


def _copy_without_vertex(graph: AbstractGraph, removed_name: str) -> AbstractGraph:
    _vertex_named(graph, removed_name)
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
