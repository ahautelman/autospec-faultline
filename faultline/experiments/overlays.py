"""Overlay geometry for the 9-rooms report.

The overlay figure shows what a refinement operator saw and what it produced. This
module derives both from the same inputs the operator consumes - the sampled
trajectories on the bottleneck edge - so the figure never recomputes a hull or a
hyperplane differently from the operator itself. The result is stored as plain
points and line parameters, so the figure layer redraws it from JSON.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from faultline.experiments.records import Overlay
from faultline.geometry import convex_hull, half_plane_partition, intersect
from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.learners import CannotRefineError, EdgeEstimate
from faultline.refinement.operators import or_refine
from faultline.spectrl.predicates import PLANAR_DIMS, HalfPlane, Predicate


def build_overlay(operator: str, graph: AbstractGraph, edge: Edge, estimate: EdgeEstimate) -> Overlay:
    """Capture the geometry a refinement operator consumed and produced.

    Args:
        operator: Operator name (``"seq"``, ``"add"``, ``"past"``, ``"or"``).
        graph: The graph the operator ran on.
        edge: The bottleneck edge it refined.
        estimate: The edge's final estimate, whose rollouts it consumed.

    How:
        Dispatches on the operator. Reach and waypoint refinement store the
        sampled states (target visits or successful midpoints); past refinement
        stores the success and failure starts plus the fitted half-plane; OrRefine
        is structural, so it stores the added edge's vertex names.

    Returns:
        The :class:`~faultline.experiments.records.Overlay` for the figure.

    Raises:
        ValueError: If ``operator`` is not one of the four names.
    """
    if operator == "seq":
        return _reach_overlay(edge, estimate)
    if operator == "add":
        return _waypoint_overlay(estimate)
    if operator == "past":
        return _past_overlay(estimate)
    if operator == "or":
        return _or_overlay(graph, edge)
    raise ValueError(f"Unknown refinement operator {operator!r}")


def refined_region(
    operator: str, edge: Edge, overlay: Overlay
) -> Predicate | None:
    """Rebuild the refined region an operator produced, for shading.

    Args:
        operator: Operator name (``"seq"``, ``"add"``, ``"past"``).
        edge: The bottleneck edge that was refined.
        overlay: The captured overlay inputs.

    How:
        Inverts the operator's geometry from the stored points or half-plane:
        ReachRefine intersects the target with the visited hull, AddRefine
        intersects the midpoint hull with the safe region, and PastRefine
        intersects the source with the fitted half-plane. OrRefine has no region,
        so it returns ``None``.

    Returns:
        The refined region predicate, or ``None`` when the operator is structural
        or the overlay carries no usable geometry.
    """
    if operator == "seq" and overlay.points:
        return intersect(edge.target.region, convex_hull(_to_states(overlay.points), PLANAR_DIMS))
    if operator == "add" and overlay.points:
        return intersect(convex_hull(_to_states(overlay.points), PLANAR_DIMS), edge.safe)
    if operator == "past" and overlay.normal is not None and overlay.offset is not None:
        partition = HalfPlane(np.asarray(overlay.normal), overlay.offset, PLANAR_DIMS)
        return intersect(edge.source.region, partition)
    return None


def _reach_overlay(edge: Edge, estimate: EdgeEstimate) -> Overlay:
    """Capture the target-region visits ReachRefine consumes.

    Args:
        edge: The bottleneck edge whose target region is filtered for.
        estimate: The edge's final estimate, supplying the rollouts.

    How:
        Scans every state of every rollout and keeps those inside the target
        region, which is the same set ReachRefine hulls.

    Returns:
        A ``hull`` overlay carrying those visited states.
    """
    region = edge.target.region
    visited = (
        state
        for rollout in estimate.rollouts
        for state in rollout.states
        if region.evaluate(state)
    )
    return Overlay("hull", _to_pairs(visited), (), None, None, None)


def _waypoint_overlay(estimate: EdgeEstimate) -> Overlay:
    """Capture the successful midpoints AddRefine consumes.

    Args:
        estimate: The edge's final estimate, supplying the rollouts.

    How:
        Takes the midpoint state of each successful rollout, which is what
        AddRefine hulls into the waypoint region; a sample-starved edge yields an
        empty overlay.

    Returns:
        A ``waypoint`` overlay carrying the midpoints.
    """
    try:
        successes = estimate.require_successful()
    except CannotRefineError:
        return Overlay("waypoint", (), (), None, None, None)
    midpoints = [rollout.states[len(rollout.states) // 2] for rollout in successes]
    return Overlay("waypoint", _to_pairs(midpoints), (), None, None, None)


def _past_overlay(estimate: EdgeEstimate) -> Overlay:
    """Capture the split start states and the half-plane PastRefine fits.

    Args:
        estimate: The edge's final estimate, supplying the rollouts.

    How:
        Splits the rollout starts into successes and failures and fits the
        separating half-plane with the same helper PastRefine uses; a
        sample-starved edge or coincident means yields an empty boundary.

    Returns:
        A ``half_plane`` overlay with the starts and the fitted boundary.
    """
    try:
        successes = estimate.require_successful()
    except CannotRefineError:
        return Overlay("half_plane", (), (), None, None, None)
    positive = [rollout.states[0] for rollout in successes]
    negative = [rollout.states[0] for rollout in estimate.rollouts if not rollout.success]
    partition = half_plane_partition(positive, negative, PLANAR_DIMS)
    normal = tuple(float(value) for value in partition.normal) if partition else None
    offset = partition.offset if partition else None
    return Overlay("half_plane", _to_pairs(positive), _to_pairs(negative), normal, offset, None)


def _or_overlay(graph: AbstractGraph, edge: Edge) -> Overlay:
    """Capture the structural edge OrRefine adds between two vertices.

    Args:
        graph: The graph the operator runs on.
        edge: The blocked edge being routed around.

    How:
        Runs OrRefine and finds the edge it added out of the blocked edge's
        source; a refusal yields an empty overlay.

    Returns:
        An ``edge`` overlay naming the added edge's endpoints.
    """
    try:
        refined = or_refine(graph, edge)
    except CannotRefineError:
        return Overlay("edge", (), (), None, None, None)
    added = next(
        (
            candidate
            for candidate in refined.edges
            if candidate.source.name == edge.source.name and candidate.target.name != edge.target.name
        ),
        None,
    )
    names = (added.source.name, added.target.name) if added is not None else None
    return Overlay("edge", (), (), None, None, names)


def _to_pairs(states: Iterable[np.ndarray]) -> tuple[tuple[float, float], ...]:
    """Project states onto the planar coordinates as pairs.

    Args:
        states: State vectors to project.

    How:
        Reads the first two coordinates of each state, the planar coordinates
        every region predicate uses.

    Returns:
        The ``(x, y)`` pairs.
    """
    return tuple((float(state[0]), float(state[1])) for state in states)


def _to_states(points: Sequence[tuple[float, float]]) -> list[np.ndarray]:
    """Turn ``(x, y)`` pairs back into planar state arrays.

    Args:
        points: The pairs to convert.

    How:
        Builds a float array per pair, the shape the convex-hull helper expects.

    Returns:
        The planar state arrays.
    """
    return [np.array(point, dtype=float) for point in points]
