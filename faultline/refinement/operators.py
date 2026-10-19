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
:class:`CannotRefineError`, which Algorithm 1 treats as "no refinement here". So
is a refined region with no area to sample from (a hull of fewer than three
non-collinear points, or an empty intersection): the paper's ``ConvexHull`` cannot
be built there either, and the environment could never draw a start or goal in it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from faultline.geometry import (
    convex_hull,
    difference,
    half_plane_partition,
    intersect,
    region_bounds,
    sample_within,
)
from faultline.graph.abstract_graph import AbstractGraph, Edge, Vertex
from faultline.learners.base import CannotRefineError, EdgeEstimate, Rollout
from faultline.spectrl.predicates import PLANAR_DIMS, Dims, Predicate

# AvoidRefine blocks the last k states of each episode that ended outside the safe
# region. The paper's section 4.3 experiment uses k = 10; FaultLine deliberately
# runs with 2 for now, so report it as a deviation when comparing to the paper.
AVOID_TAIL_LENGTH = 2

# Fixed seed for the sampleability check, so a refusal is reproducible.
_SAMPLEABLE_SEED = 0


def reach_refine(
    region: Predicate, rollouts: Sequence[Rollout], dims: Dims = PLANAR_DIMS
) -> Predicate:
    """Shrink a target region to the hull of the states actually reached in it.

    Args:
        region: The edge's target-region predicate to tighten.
        rollouts: Sampled episodes for the edge, successful or not.
        dims: State coordinates the hull is built in; defaults to ``(x, y)``.

    How:
        Collects the states inside ``region`` that successful episodes visited
        (:func:`reach_witnesses`), then intersects ``region`` with their convex
        hull. Because the hull is intersected with ``region``, the result is
        always a subset of it.

    Returns:
        The tightened target-region predicate.

    Raises:
        CannotRefineError: If no successful episode visited ``region``, or the
            hull has no area to sample a goal from.
    """
    visited = reach_witnesses(region, rollouts)
    if not visited:
        raise CannotRefineError("No successful rollout visited the target region to shrink")
    return _require_sampleable(intersect(region, convex_hull(visited, dims)), dims, "target")


def reach_witnesses(region: Predicate, rollouts: Sequence[Rollout]) -> list[np.ndarray]:
    """Return the states inside a target region that successful episodes visited.

    Args:
        region: The edge's target-region predicate ``b``.
        rollouts: Sampled episodes for the edge, successful or not.

    How:
        Keeps only successful episodes, following the paper's prose (section 3.1:
        "trajectories that successfully reached the goal region"; section 4.3:
        "successful trajectories only reach the accessible portion"). Algorithm 2's
        pseudocode reads every trajectory, but a failed episode can end inside an
        enterable trap within ``b``, which would then stay in the refined target.

    Returns:
        The witness states, in episode then time order.
    """
    return [
        state
        for rollout in rollouts
        if rollout.success
        for state in rollout.states
        if region.evaluate(state)
    ]


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
        k: Number of trailing states to collect from each episode that ended
            unsafe; the paper's hyperparameter (section 6.2).
        dims: State coordinates the hull is built in; defaults to ``(x, y)``.

    How:
        Implements Algorithm 3 exactly: for every episode with ``ζ_i[-1] ∉ c``,
        takes its last ``k`` states and hulls the union of those tails, then
        subtracts that hull from ``region``. Episodes that left ``region`` at some
        point but ended inside it are ignored, as are timeouts that never left it.
        Subtraction only removes states, so the result stays a subset.

    Returns:
        The tightened safe-region predicate.

    Raises:
        ValueError: If ``k`` is not positive.
        CannotRefineError: If no episode ended outside the safe region.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    tails = [
        state
        for rollout in rollouts
        if ended_unsafe(rollout, region)
        for state in rollout.states[-k:]
    ]
    if not tails:
        raise CannotRefineError("No rollout ended outside the safe region to avoid")
    return difference(region, convex_hull(tails, dims))


def ended_unsafe(rollout: Rollout, region: Predicate) -> bool:
    """Return whether an episode ended outside the edge's safe region.

    Args:
        rollout: The sampled episode.
        region: The edge's safe-region predicate ``c``.

    How:
        Tests the final state against ``region``, which is the paper's
        ``ζ_i[-1] ∉ c`` filter. A learner can also flag the final state as unsafe
        because of an environment hazard the specification does not model (the
        hidden trap in ``narrow_door``); ``final_safe`` carries that flag and can
        only mark the episode unsafe. That flag goes beyond Algorithm 3, whose
        unsafe states come from the specification alone (the paper's Figure 8
        writes the avoid region into the specification): with a hidden trap and
        a ``true`` safe region, the paper's filter would never fire. Synthetic or older rollouts without a final
        flag are judged by the region check alone.

    Returns:
        ``True`` when the episode's final state is outside the safe region.
    """
    if not region.evaluate(rollout.states[-1]):
        return True
    return rollout.final_safe is False


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
        edge, not a region that cannot move. As in Algorithm 4, the graph keeps
        its vertex and edge sets: the target vertex is relabelled with the
        tightened target, so every edge entering or leaving it sees the new
        region, and the edge carries the tightened safe region.

    Returns:
        A new graph whose ``edge`` has the tightened target and safe regions.

    Raises:
        CannotRefineError: If neither region can be tightened.
    """
    refined = graph.copy()
    copied = _copied_edge(refined, graph, edge)
    new_target = _attempt(lambda: reach_refine(copied.target.region, estimate.rollouts, dims))
    new_safe = _attempt(lambda: avoid_refine(copied.safe, estimate.rollouts, k, dims))
    if new_target is None and new_safe is None:
        raise CannotRefineError("SeqRefine has neither a tighter target nor a tighter safe region")
    if new_target is not None:
        copied.target.region = new_target
    if new_safe is not None:
        copied.safe = new_safe
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
        CannotRefineError: If no episode succeeded, or the clipped hull has no
            area to sample a waypoint from.
    """
    successes = estimate.require_successful()
    midpoints = [rollout.states[len(rollout.states) // 2] for rollout in successes]
    refined = graph.copy()
    copied = _copied_edge(refined, graph, edge)
    waypoint = _require_sampleable(
        intersect(convex_hull(midpoints, dims), copied.safe), dims, "waypoint"
    )
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
            refined start region has no area to sample a start from.
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
    region = _require_sampleable(intersect(copied.source.region, partition), dims, "start")
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
        The new edges are appended after the copied edges, so a caller can read
        exactly the added links from the tail of the refined edge list.

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


def _require_sampleable(region: Predicate, dims: Dims, role: str) -> Predicate:
    """Return a refined region, refusing one that no start or goal can be drawn from.

    Args:
        region: The refined region an operator is about to label a vertex with.
        dims: State coordinates the region reads.
        role: What the region is used as (``"target"``, ``"waypoint"``,
            ``"start"``), for the refusal message.

    How:
        Rejection-samples the region inside its axis-aligned bounds with the same
        sampler the environment uses. A region with no area there (a degenerate
        hull, an empty intersection) is never hit, so it is refused here instead
        of crashing the environment's ``reset`` later.

    Returns:
        ``region`` unchanged when it can be sampled.

    Raises:
        CannotRefineError: If the region has no samplable area.
    """
    bounds = region_bounds(region, dims)
    if bounds is None:
        # ponytail: an unbounded region is sampled over the whole world by the
        # environment, so it is not checked here; bound it if that ever fails.
        return region
    lower, upper = bounds
    rng = np.random.default_rng(_SAMPLEABLE_SEED)
    if np.any(lower > upper) or sample_within(region, lower, upper, rng) is None:
        raise CannotRefineError(f"Refined {role} region has no area to sample from")
    return region


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
        refined: The graph returned by ``graph.copy()``, possibly with edges
            appended after the copied ones (as OrRefine does).
        graph: The original graph that holds ``edge``.
        edge: The edge to locate by identity in ``graph``.

    How:
        Finds ``edge``'s index in the original graph's edge order and reads the
        same index from the copy. ``graph.copy()`` preserves the edge order and
        ``add_edge`` only appends, so the original edges stay a prefix of the
        copy and the index names the same edge. The correspondence is checked
        rather than assumed: a copy that lost edges or reordered them fails
        loudly instead of silently returning the wrong edge.

    Returns:
        The corresponding edge object in ``refined``.

    Raises:
        ValueError: If ``edge`` is not part of ``graph``, or the copied graph does
            not preserve the edge's position and structure.
    """
    if len(refined.edges) < len(graph.edges):
        raise ValueError(
            f"Copied graph has {len(refined.edges)} edges, "
            f"fewer than the original's {len(graph.edges)}"
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
    """Return whether two edges join the same pair of vertices.

    Args:
        edge: The first edge to compare.
        other: The second edge to compare.

    How:
        Compares endpoint names only. Identity is deliberately not required, so
        an edge and its deep copy compare equal, and the regions are not compared
        because SeqRefine tightens the copied edge's regions in place.

    Returns:
        ``True`` when both edges share endpoint names.
    """
    return edge.source.name == other.source.name and edge.target.name == other.target.name


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
