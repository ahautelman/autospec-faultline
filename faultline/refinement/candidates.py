"""Candidate graphs for the Algorithm 1 commit gate (paper section 3).

The main loop in :mod:`faultline.refinement.loop` tries the four operators in a
fixed order. Each operator returns a new graph, but the loop also has to train
and score the *replacement* it made. This module wraps each operator into a
:class:`Candidate`: the refined graph plus the edge routes that stand in for the
original edge.

Routes exist because the operators replace one edge with different shapes: a
two-hop path for SeqRefine and AddRefine, a single re-homed edge for PastRefine,
and one or more parallel routes for OrRefine. The loop scores the best route,
which matches the best-path product reporting rule of paper section 6.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from faultline.graph.abstract_graph import AbstractGraph, Edge, Vertex
from faultline.learners.base import CannotRefineError, EdgeEstimate
from faultline.refinement.operators import add_refine, or_refine, past_refine, seq_refine


@dataclass(frozen=True)
class Candidate:
    """A refined graph and the routes that replace the original edge.

    Args:
        graph: The graph the operator returned.
        routes: Replacement routes; each route is a tuple of edges forming a
            source-to-target path standing in for the original edge.

    How:
        Bundles the graph edit with the edges the commit gate must train, so the
        loop never has to reverse-engineer an operator's internal structure.
    """

    graph: AbstractGraph
    routes: tuple[tuple[Edge, ...], ...]


def seq_candidate(
    graph: AbstractGraph,
    edge: Edge,
    estimate: EdgeEstimate,
    avoid_tail_length: int,
) -> Candidate:
    """Wrap SeqRefine into a candidate two-hop route.

    Args:
        graph: The graph containing ``edge``.
        edge: The weak edge to refine.
        estimate: The edge's estimate, supplying the rollouts.
        avoid_tail_length: Trailing states AvoidRefine blocks.

    How:
        Calls :func:`~faultline.refinement.operators.seq_refine` and identifies
        the fresh vertex it adds, then reports the two edges through it as one
        route.

    Returns:
        The candidate with the refined graph and the two-hop route.

    Raises:
        ValueError: If the operator did not add the vertex its contract requires.
    """
    refined = seq_refine(graph, edge, estimate, avoid_tail_length)
    return _waypoint_candidate(refined, graph, edge)


def add_candidate(
    graph: AbstractGraph,
    edge: Edge,
    estimate: EdgeEstimate,
    avoid_tail_length: int,
) -> Candidate:
    """Wrap AddRefine into a candidate two-hop route.

    Args:
        graph: The graph containing ``edge``.
        edge: The weak edge to refine.
        estimate: The edge's estimate; successful rollouts are required.
        avoid_tail_length: Unused; present so every builder shares one signature.

    How:
        Calls :func:`~faultline.refinement.operators.add_refine` and reports the
        two edges through the fresh waypoint it adds.

    Returns:
        The candidate with the refined graph and the two-hop route.

    Raises:
        ValueError: If the operator did not add the vertex its contract requires.
    """
    refined = add_refine(graph, edge, estimate)
    return _waypoint_candidate(refined, graph, edge)


def past_candidate(
    graph: AbstractGraph,
    edge: Edge,
    estimate: EdgeEstimate,
    avoid_tail_length: int,
) -> Candidate:
    """Wrap PastRefine into a candidate single-edge route.

    Args:
        graph: The graph containing ``edge``.
        edge: The weak edge to refine.
        estimate: The edge's estimate; successes and failures are required.
        avoid_tail_length: Unused; present so every builder shares one signature.

    How:
        Calls :func:`~faultline.refinement.operators.past_refine`. When it adds a
        fresh entry vertex, the route is the re-homed edge leaving it; when the
        source is the graph source and no vertex is added, the route is the
        original edge with its source region tightened in place.

    Returns:
        The candidate with the refined graph and its single route.
    """
    refined = past_refine(graph, edge, estimate)
    fresh = _optional_fresh_vertex(refined, graph)
    if fresh is None:
        same = _edge_ref(refined, edge.source.name, edge.target.name)
        return Candidate(refined, ((same,),))
    replacement = _edge_ref(refined, fresh.name, edge.target.name)
    return Candidate(refined, ((replacement,),))


def or_candidate(
    graph: AbstractGraph,
    edge: Edge,
    estimate: EdgeEstimate,
    avoid_tail_length: int,
) -> Candidate:
    """Wrap OrRefine into one candidate route per sibling parent.

    Args:
        graph: The graph containing ``edge``.
        edge: The blocked edge to route around.
        estimate: Unused; OrRefine is purely structural.
        avoid_tail_length: Unused; present so every builder shares one signature.

    How:
        Calls :func:`~faultline.refinement.operators.or_refine`, then for every
        other parent of the target reports the two-hop route ``source -> sibling
        -> target``. The untouched original edge is skipped.

    Returns:
        The candidate with the refined graph and one route per sibling.

    Raises:
        CannotRefineError: If the refined graph contains no sibling route.
    """
    refined = or_refine(graph, edge)
    target = _vertex(refined, edge.target.name)
    routes = []
    for sibling in refined.predecessors(target):
        if sibling.source.name == edge.source.name and sibling.target.name == edge.target.name:
            continue
        routes.append((_edge_ref(refined, edge.source.name, sibling.source.name), sibling))
    if not routes:
        raise CannotRefineError("OrRefine produced no alternative route to score")
    return Candidate(refined, tuple(routes))


def _waypoint_candidate(refined: AbstractGraph, graph: AbstractGraph, edge: Edge) -> Candidate:
    """Build a candidate for an operator that added a fresh waypoint vertex.

    Args:
        refined: The graph the operator returned.
        graph: The graph before the edit.
        edge: The edge that was refined.

    How:
        Locates the fresh vertex and reports the route through it, from the
        original source to the original target. A two-hop route is the norm; when
        the refined target was the graph target, SeqRefine makes the fresh vertex
        the target and drops the bridge, so the route is the single new edge.

    Returns:
        The candidate with the route standing in for the original edge.

    Raises:
        ValueError: If the fresh vertex, or the first edge into it, is missing.
    """
    fresh = _optional_fresh_vertex(refined, graph)
    if fresh is None:
        raise ValueError("Operator did not add the waypoint vertex its contract requires")
    first = _edge_ref(refined, edge.source.name, fresh.name)
    bridge = _optional_edge_ref(refined, fresh.name, edge.target.name)
    if bridge is None:
        return Candidate(refined, ((first,),))
    return Candidate(refined, ((first, bridge),))


def _optional_fresh_vertex(refined: AbstractGraph, graph: AbstractGraph) -> Vertex | None:
    """Return the first vertex ``refined`` added, or ``None`` if it added none.

    Args:
        refined: The graph an operator returned.
        graph: The graph before the edit.

    How:
        Compares vertex names: a name absent from the original graph belongs to a
        vertex the operator created. Each operator adds at most one such vertex.

    Returns:
        The fresh vertex, or ``None`` when the edit added no vertex.
    """
    existing = {vertex.name for vertex in graph.vertices}
    for vertex in refined.vertices:
        if vertex.name not in existing:
            return vertex
    return None


def _edge_ref(graph: AbstractGraph, source_name: str, target_name: str) -> Edge:
    """Return the edge of ``graph`` with the given endpoint names.

    Args:
        graph: The graph to search.
        source_name: Source vertex name.
        target_name: Target vertex name.

    How:
        Scans the edge list for the first endpoint-name match and fails fast when
        none exists, which would mean an operator produced an unexpected shape.

    Returns:
        The matching edge.

    Raises:
        ValueError: If no edge has those endpoints.
    """
    edge = _optional_edge_ref(graph, source_name, target_name)
    if edge is None:
        raise ValueError(f"No edge {source_name!r} -> {target_name!r} in the candidate graph")
    return edge


def _optional_edge_ref(
    graph: AbstractGraph, source_name: str, target_name: str
) -> Edge | None:
    """Return the edge with the given endpoint names, or ``None`` if absent.

    Args:
        graph: The graph to search.
        source_name: Source vertex name.
        target_name: Target vertex name.

    How:
        Scans the edge list for the first endpoint-name match and reports the
        absence instead of raising, so a caller can distinguish a missing bridge
        from a malformed operator result.

    Returns:
        The matching edge, or ``None`` when no edge has those endpoints.
    """
    for edge in graph.edges:
        if edge.source.name == source_name and edge.target.name == target_name:
            return edge
    return None


def _vertex(graph: AbstractGraph, name: str) -> Vertex:
    """Return the vertex of ``graph`` with the given name.

    Args:
        graph: The graph to search.
        name: The vertex name to find.

    How:
        Scans the graph's vertices for a name match and fails fast when absent.

    Returns:
        The matching vertex.

    Raises:
        ValueError: If no vertex has that name.
    """
    for vertex in graph.vertices:
        if vertex.name == name:
            return vertex
    raise ValueError(f"No vertex {name!r} in the candidate graph")
