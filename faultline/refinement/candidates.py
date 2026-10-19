"""Candidate graphs for the Algorithm 1 commit gate (paper section 3).

The main loop in :mod:`faultline.refinement.loop` tries the four operators in a
fixed order. Each operator returns a new graph, but the loop also has to train
and score the *replacement* it made. This module wraps each operator into a
:class:`Candidate`: the refined graph plus the edge routes that stand in for the
original edge.

Routes exist because the operators replace one edge with different shapes: the
relabelled edge itself for SeqRefine, a two-hop path for AddRefine, a single
re-homed edge for PastRefine, and one or more parallel routes for OrRefine. The loop scores the best route,
which matches the best-path product reporting rule of paper section 6.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from faultline.graph.abstract_graph import AbstractGraph, Edge, Vertex
from faultline.learners.base import EdgeEstimate
from faultline.refinement.operators import (
    _copied_edge,
    add_refine,
    or_refine,
    past_refine,
    seq_refine,
)


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
    """Wrap SeqRefine into a candidate single-edge route.

    Args:
        graph: The graph containing ``edge``.
        edge: The weak edge to refine.
        estimate: The edge's estimate, supplying the rollouts.
        avoid_tail_length: Trailing states AvoidRefine blocks.

    How:
        Calls :func:`~faultline.refinement.operators.seq_refine`, which relabels
        the edge in place, so the route is that same edge in the refined graph.

    Returns:
        The candidate with the refined graph and the relabelled edge as its route.
    """
    refined = seq_refine(graph, edge, estimate, avoid_tail_length)
    return Candidate(refined, ((_copied_edge(refined, graph, edge),),))


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
        Calls :func:`~faultline.refinement.operators.or_refine`, which appends its
        new ``source -> sibling`` links after the copied edges. Each appended link
        is paired with the sibling edges leaving its target for the original
        target, giving the two-hop route ``source -> sibling -> target``. Reading
        the links by position rather than by endpoint names skips siblings that
        OrRefine left out (they would close a cycle) and never confuses a new link
        with an older parallel edge between the same vertices.

    Returns:
        The candidate with the refined graph and one route per sibling.
    """
    refined = or_refine(graph, edge)
    target = _copied_edge(refined, graph, edge).target
    links = refined.edges[len(graph.edges) :]
    routes = tuple(
        (link, sibling)
        for link in links
        for sibling in refined.predecessors(target)
        if sibling.source is link.target
    )
    return Candidate(refined, routes)


def _waypoint_candidate(refined: AbstractGraph, graph: AbstractGraph, edge: Edge) -> Candidate:
    """Build a candidate for an operator that added a fresh waypoint vertex.

    Args:
        refined: The graph the operator returned.
        graph: The graph before the edit.
        edge: The edge that was refined.

    How:
        Locates the fresh vertex and reports the two-hop route through it, from
        the original source to the original target.

    Returns:
        The candidate with the route standing in for the original edge.

    Raises:
        ValueError: If the fresh vertex, or an edge through it, is missing.
    """
    fresh = _optional_fresh_vertex(refined, graph)
    if fresh is None:
        raise ValueError("Operator did not add the waypoint vertex its contract requires")
    first = _edge_ref(refined, edge.source.name, fresh.name)
    second = _edge_ref(refined, fresh.name, edge.target.name)
    return Candidate(refined, ((first, second),))


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
    for edge in graph.edges:
        if edge.source.name == source_name and edge.target.name == target_name:
            return edge
    raise ValueError(f"No edge {source_name!r} -> {target_name!r} in the candidate graph")
