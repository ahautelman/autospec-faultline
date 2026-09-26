"""The refinement operators that repair a weak edge (paper section 4).

Each operator takes an edge of an abstract graph plus the rollouts the edge's
policy produced, and returns a new graph in which that edge is easier to satisfy
while remaining a *refinement*: every trajectory that satisfies the new graph
also satisfies the old one. The two region sub-procedures the paper uses are
implemented first, then the four graph operators built from them:

- :func:`reach_refine` (ReachRefine) and :func:`avoid_refine` (AvoidRefine)
  tighten one region to the visited or unsafe hull.
- :func:`seq_refine` (SeqRefine) applies both to one edge.
- :func:`add_refine` (AddRefine) inserts a waypoint from successful midpoints.
- :func:`past_refine` (PastRefine) keeps only the start states that succeed.
- :func:`or_refine` (OrRefine) routes around a blocked edge via a sibling.

A sample-starved edge is not an error: the operators raise the paper's
:class:`CannotRefineError`, which Algorithm 1 treats as "no refinement here".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from faultline.geometry import convex_hull, difference, half_plane_partition, intersect
from faultline.graph.abstract_graph import AbstractGraph, Edge, Vertex
from faultline.learners.base import CannotRefineError, EdgeEstimate, Rollout
from faultline.spectrl.predicates import PLANAR_DIMS, Dims, Empty, Predicate

# Paper section 6.2: AvoidRefine blocks the last k states of each unsafe episode.
AVOID_TAIL_LENGTH = 10


def reach_refine(
    region: Predicate, rollouts: Sequence[Rollout], dims: Dims = PLANAR_DIMS
) -> Predicate:
    """Shrink a target region to the hull of the states actually reached in it.

    Args:
        region: The edge's target-region predicate to tighten.
        rollouts: Sampled episodes for the edge, successful or not.
        dims: State coordinates the hull is built in; defaults to ``(x, y)``.

    How:
        Collects every visited state that lies in ``region``, then intersects
        ``region`` with the convex hull of those states. Because the hull is
        intersected with ``region``, the result is always a subset of it.

    Returns:
        The tightened target-region predicate.

    Raises:
        CannotRefineError: If no visited state lies inside ``region``.
    """
    visited = [
        state for rollout in rollouts for state in rollout.states if region.evaluate(state)
    ]
    if not visited:
        raise CannotRefineError("No rollout visited the target region to shrink")
    return intersect(region, convex_hull(visited, dims))


def avoid_refine(
    region: Predicate,
    rollouts: Sequence[Rollout],
    k: int = AVOID_TAIL_LENGTH,
    dims: Dims = PLANAR_DIMS,
) -> Predicate:
    """Shrink a safe region by the hull of the unsafe trajectory tails.

    Args:
        region: The edge's safe-region predicate to tighten.
        rollouts: Sampled episodes for the edge, successful or not.
        k: Number of trailing states to collect from each unsafe episode; the
            paper's hyperparameter (section 6.2).
        dims: State coordinates the hull is built in; defaults to ``(x, y)``.

    How:
        For every episode that left the safe region, takes its last ``k`` states
        and hulls them, then subtracts that hull from ``region``. Subtraction only
        removes states, so the result stays a subset.

    Returns:
        The tightened safe-region predicate.

    Raises:
        ValueError: If ``k`` is not positive.
        CannotRefineError: If no episode left the safe region.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    tails = [
        state
        for rollout in rollouts
        if not rollout.safe
        for state in rollout.states[-k:]
    ]
    if not tails:
        raise CannotRefineError("No rollout left the safe region to avoid")
    return difference(region, convex_hull(tails, dims))


def seq_refine(
    graph: AbstractGraph,
    edge: Edge,
    estimate: EdgeEstimate,
    k: int = AVOID_TAIL_LENGTH,
    dims: Dims = PLANAR_DIMS,
) -> AbstractGraph:
    """Tighten one edge's target and safe regions, replacing it in a new graph.

    Args:
        graph: The graph containing ``edge``.
        edge: The reach-avoid edge to refine.
        estimate: The edge's sampled estimate, supplying the rollouts.
        k: Trailing-state count forwarded to :func:`avoid_refine`.
        dims: State coordinates the hulls are built in.

    How:
        Applies ReachRefine and AvoidRefine independently, skipping a
        sub-procedure that has no data: the paper's hard stop is a sample-starved
        edge, not a region that cannot move. The refined edge runs from the
        original source to a fresh vertex carrying the tightened target, safely
        inside the tightened region. When the refined edge reached the graph
        target, that vertex becomes the new target (the paper's "u' replaced by
        u_r on this path"); otherwise an unconstrained bridge to the original
        target vertex is added so any other path through it survives.

    Returns:
        A new graph with ``edge`` replaced by the refined two-hop path.

    Raises:
        CannotRefineError: If neither region can be tightened.
    """
    refined = graph.copy()
    copied = _copied_edge(refined, graph, edge)
    new_target = _attempt(lambda: reach_refine(copied.target.region, estimate.rollouts, dims))
    new_safe = _attempt(lambda: avoid_refine(copied.safe, estimate.rollouts, k, dims))
    if new_target is None and new_safe is None:
        raise CannotRefineError("SeqRefine has neither a tighter target nor a tighter safe region")
    target = new_target if new_target is not None else copied.target.region
    safe = new_safe if new_safe is not None else copied.safe
    vertex = refined.fresh_vertex("seq", target)
    refined.remove_edge(copied)
    refined.add_edge(copied.source, vertex, safe)
    if copied.target is refined.target:
        refined.target = vertex
    else:
        refined.add_edge(vertex, copied.target)
    return refined


def add_refine(
    graph: AbstractGraph, edge: Edge, estimate: EdgeEstimate, dims: Dims = PLANAR_DIMS
) -> AbstractGraph:
    """Split an edge at a waypoint taken from successful trajectories.

    Args:
        graph: The graph containing ``edge``.
        edge: The reach-avoid edge to refine.
        estimate: The edge's sampled estimate; successful rollouts are required.
        dims: State coordinates the hull is built in.

    How:
        Takes each successful episode's midpoint state, hulls them, and clips the
        hull to the edge's safe region to get the waypoint region. The original
        edge is replaced by two edges sharing the original safe region that meet
        at a fresh waypoint vertex.

    Returns:
        A new graph with ``edge`` replaced by the two-hop route.

    Raises:
        CannotRefineError: If no episode succeeded, or the hull meets no safe
            state to use as a waypoint.
    """
    successes = estimate.require_successful()
    midpoints = [rollout.states[len(rollout.states) // 2] for rollout in successes]
    refined = graph.copy()
    copied = _copied_edge(refined, graph, edge)
    waypoint = intersect(convex_hull(midpoints, dims), copied.safe)
    if isinstance(waypoint, Empty):
        raise CannotRefineError("Waypoint hull does not meet the edge's safe region")
    vertex = refined.fresh_vertex("add", waypoint)
    refined.remove_edge(copied)
    refined.add_edge(copied.source, vertex, copied.safe)
    refined.add_edge(vertex, copied.target, copied.safe)
    return refined


def past_refine(
    graph: AbstractGraph, edge: Edge, estimate: EdgeEstimate, dims: Dims = PLANAR_DIMS
) -> AbstractGraph:
    """Restrict an edge to the start states from which it succeeds.

    Args:
        graph: The graph containing ``edge``.
        edge: The reach-avoid edge to refine.
        estimate: The edge's sampled estimate; successful rollouts are required.
        dims: State coordinates the separating half-plane is fit in.

    How:
        Splits the observed start states into successes and failures, fits a
        half-plane that keeps the successes, and intersects it with the source
        vertex's region. A fresh vertex carries that region; it copies the source
        vertex's incoming edges so it is entered exactly as the original was, and
        the refined edge leaves it. When the source vertex is the graph source it
        has no incoming edges to copy, so its region is tightened in place and the
        seed region shrinks instead.

    Returns:
        A new graph whose ``edge`` accepts only the successful side of its start
        region.

    Raises:
        CannotRefineError: If no episode succeeded, no episode failed, or the
            refined start region is empty.
    """
    successes = estimate.require_successful()
    positive = [rollout.states[0] for rollout in successes]
    negative = [rollout.states[0] for rollout in estimate.rollouts if not rollout.success]
    if not negative:
        raise CannotRefineError("All sampled starts succeeded, so there is nothing to cut")
    refined = graph.copy()
    copied = _copied_edge(refined, graph, edge)
    partition = half_plane_partition(positive, negative, dims)
    if partition is None:
        raise CannotRefineError("Start states admit no separating half-plane")
    region = intersect(copied.source.region, partition)
    if isinstance(region, Empty):
        raise CannotRefineError("Refined start region is empty")
    if copied.source is refined.source:
        refined.source.region = region
        return refined
    vertex = refined.fresh_vertex("past", region)
    for incoming in refined.predecessors(copied.source):
        refined.add_edge(incoming.source, vertex, incoming.safe)
    refined.remove_edge(copied)
    refined.add_edge(vertex, copied.target, copied.safe)
    return refined


def or_refine(graph: AbstractGraph, edge: Edge) -> AbstractGraph:
    """Route around a blocked edge through its target's other parents.

    Args:
        graph: The graph containing ``edge``.
        edge: The blocked reach-avoid edge.

    How:
        For every other edge that already enters ``edge``'s target, adds a
        parallel route ``edge.source -> that parent`` carrying ``edge``'s safe
        region, and tightens the sibling's safe region by the same predicate. The
        original edge is left in place. A sibling that already reaches
        ``edge.source`` is skipped, because the new edge would create a cycle.

    Returns:
        A new graph with the parallel routes added.

    Raises:
        CannotRefineError: If the target has no other parent that can be reached
            from ``edge``'s source without a cycle.
    """
    refined = graph.copy()
    copied = _copied_edge(refined, graph, edge)
    siblings = [
        candidate
        for candidate in refined.predecessors(copied.target)
        if candidate is not copied and not _reaches(refined, candidate.source, copied.source)
    ]
    if not siblings:
        raise CannotRefineError("Target has no alternative parent to route through")
    for sibling in siblings:
        sibling.safe = intersect(sibling.safe, copied.safe)
        refined.add_edge(copied.source, sibling.source, copied.safe)
    return refined


def _attempt(operation: Callable[[], Predicate]) -> Predicate | None:
    """Run a region sub-procedure, treating a sample-starved refusal as no result.

    Args:
        operation: A zero-argument call that returns a refined predicate.

    How:
        Calls the operation and returns its result, or :data:`None` when it raises
        :class:`CannotRefineError` because it had no data to work from. Only that
        specific refusal is caught, so real errors still surface.

    Returns:
        The refined predicate, or ``None`` if the operation refused.
    """
    try:
        return operation()
    except CannotRefineError:
        return None


def _copied_edge(refined: AbstractGraph, graph: AbstractGraph, edge: Edge) -> Edge:
    """Return the copy of ``edge`` inside a deep-copied graph.

    Args:
        refined: The graph returned by ``graph.copy()``.
        graph: The original graph that holds ``edge``.
        edge: The edge to locate by identity in ``graph``.

    How:
        Finds ``edge``'s index in the original graph's edge order and reads the
        same index from the copy. The operator contract is that ``refined`` is an
        untouched ``graph.copy()``, so it preserves both the edge count and the
        edge order; the index therefore names the same edge. The correspondence is
        checked rather than assumed: a copy whose length or position drifted fails
        loudly instead of silently returning the wrong edge.

    Returns:
        The corresponding edge object in ``refined``.

    Raises:
        ValueError: If ``edge`` is not part of ``graph``, or the copied graph does
            not preserve the edge's position and structure.
    """
    if len(refined.edges) != len(graph.edges):
        raise ValueError(
            f"Copied graph has {len(refined.edges)} edges, "
            f"the original has {len(graph.edges)}"
        )
    for index, candidate in enumerate(graph.edges):
        if candidate is edge:
            copied = refined.edges[index]
            if not _same_edge(candidate, copied):
                raise ValueError(
                    f"Copied edge at index {index} does not correspond to {edge!r}"
                )
            return copied
    raise ValueError(f"Edge {edge!r} is not part of the graph")


def _same_edge(edge: Edge, other: Edge) -> bool:
    """Return whether two edges describe the same reach-avoid subgoal.

    Args:
        edge: The first edge to compare.
        other: The second edge to compare.

    How:
        Compares endpoint names and the safe predicate, which implement equality
        for the compiler's predicates. Identity is deliberately not required, so
        an edge and its deep copy compare equal.

    Returns:
        ``True`` when both edges share endpoints and safe region.
    """
    return (
        edge.source.name == other.source.name
        and edge.target.name == other.target.name
        and edge.safe == other.safe
    )


def _reaches(graph: AbstractGraph, start: Vertex, goal: Vertex) -> bool:
    """Return whether ``goal`` is reachable from ``start`` by following edges.

    Args:
        graph: The graph to search.
        start: The vertex to start from.
        goal: The vertex to look for.

    How:
        Runs an iterative depth-first search over successors, comparing vertices
        by identity, and stops as soon as the goal is seen.

    Returns:
        ``True`` when a directed path from ``start`` to ``goal`` exists.
    """
    stack = [start]
    seen = {id(start)}
    while stack:
        vertex = stack.pop()
        if vertex is goal:
            return True
        for candidate in graph.successors(vertex):
            if id(candidate.target) not in seen:
                seen.add(id(candidate.target))
                stack.append(candidate.target)
    return False
