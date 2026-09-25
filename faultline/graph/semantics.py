"""Semantics over the abstract graph.

Two operations that both the tests and the refinement loop need:

- :func:`models_trajectory` decides whether a state sequence satisfies a graph,
  implementing the ``zeta models G`` relation from paper section 2.3. It is the
  oracle behind the compiler's ``zeta models phi  <=>  zeta models G`` test.
- :func:`best_path_product` computes whole-graph success as the product of edge
  success probabilities along the best source-to-target path, the reporting rule
  from paper section 6.2.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from faultline.graph.abstract_graph import AbstractGraph, Edge

State = np.ndarray


def _edge_traversable(
    edge: Edge, states: Sequence[State], arrival_times: set[int], depart_time: int
) -> bool:
    """Decide whether ``edge`` can be completed by time ``depart_time``.

    Args:
        edge: The reach-avoid edge being attempted.
        states: The full trajectory as a sequence of state vectors.
        arrival_times: Times at which the agent arrived at the edge's source.
        depart_time: The time by which the target must be reached.

    How:
        Tries each arrival time at or before ``depart_time`` and accepts if the
        target region holds at ``depart_time`` while the edge's safe region
        holds at every state from the arrival through ``depart_time``.

    Returns:
        ``True`` if some arrival time makes the edge satisfiable.
    """
    if not edge.target.region.evaluate(states[depart_time]):
        return False
    for start in arrival_times:
        if start > depart_time:
            continue
        if all(edge.safe.evaluate(states[t]) for t in range(start, depart_time + 1)):
            return True
    return False


def _relax_arrivals_at_time(
    graph: AbstractGraph, states: Sequence[State], arrivals: dict[int, set[int]], time: int
) -> None:
    """Propagate arrivals through zero-duration hops at one time step.

    Args:
        graph: The compiled or refined graph.
        states: The full trajectory as a sequence of state vectors.
        arrivals: Per-vertex sets of arrival times, updated in place.
        time: The time step currently being relaxed.

    How:
        Scans the edges repeatedly, adding ``time`` to an edge's target once its
        source can reach that edge by ``time``, until a full pass adds nothing.
        Iterating to a fixpoint makes zero-duration propagation independent of
        the edge order.

    Returns:
        None: ``arrivals`` is mutated in place.
    """
    changed = True
    while changed:
        changed = False
        for edge in graph.edges:
            if time in arrivals[id(edge.target)]:
                continue
            if _edge_traversable(edge, states, arrivals[id(edge.source)], time):
                arrivals[id(edge.target)].add(time)
                changed = True


def models_trajectory(graph: AbstractGraph, states: Sequence[State]) -> bool:
    """Return whether a trajectory satisfies the abstract graph.

    Args:
        graph: The compiled or refined graph, with source and target set.
        states: The trajectory as a sequence of state vectors, in time order.

    How:
        Runs a forward dynamic program over (time, vertex) pairs. Arrival at the
        source is seeded at time zero when the first state lies in the source
        region. For each time step, edges are relaxed to a fixpoint so
        zero-duration hops through universe-labelled vertices propagate without
        depending on edge order. The trajectory satisfies the graph once the
        target can be reached at any time.

    Returns:
        ``True`` if the trajectory induces a path from source to target whose
        every traversed edge satisfies its reach-avoid condition.

    Raises:
        ValueError: If the graph's source or target has not been set.
    """
    if graph.source is None or graph.target is None:
        raise ValueError("Graph source and target must be set before evaluation")
    if not states:
        return False
    arrivals: dict[int, set[int]] = {id(vertex): set() for vertex in graph.vertices}
    if graph.source.region.evaluate(states[0]):
        arrivals[id(graph.source)].add(0)
    for time in range(len(states)):
        _relax_arrivals_at_time(graph, states, arrivals, time)
    return bool(arrivals[id(graph.target)])


def best_path_product(graph: AbstractGraph, probabilities: Mapping[Edge, float]) -> float:
    """Compute whole-graph success as the best path's probability product.

    Args:
        graph: The graph, with source and target set.
        probabilities: Per-edge success probability in ``[0, 1]``; every edge of
            the graph must have an entry.

    How:
        Runs a longest-path dynamic program in topological order, summing
        log-probabilities so the exponential-product comparisons stay stable,
        then exponentiates the best value reaching the target. Missing entries
        raise rather than defaulting, so an untrained edge cannot masquerade as
        a successful one.

    Returns:
        The probability of the most reliable source-to-target path, or ``0.0``
        if no path exists (for example after an operator disconnects the graph).

    Raises:
        ValueError: If the source or target is unset, or an edge has no estimate.
    """
    if graph.source is None or graph.target is None:
        raise ValueError("Graph source and target must be set before evaluation")
    best: dict[int, float] = {id(vertex): float("-inf") for vertex in graph.vertices}
    best[id(graph.source)] = 0.0
    for vertex in graph.topological_order():
        reached = best[id(vertex)]
        if reached == float("-inf"):
            continue
        for edge in graph.successors(vertex):
            if edge not in probabilities:
                raise ValueError(f"No success estimate provided for edge {edge!r}")
            probability = probabilities[edge]
            step = math.log(probability) if probability > 0 else float("-inf")
            candidate = reached + step
            if candidate > best[id(edge.target)]:
                best[id(edge.target)] = candidate
    result = best[id(graph.target)]
    return 0.0 if result == float("-inf") else math.exp(result)
