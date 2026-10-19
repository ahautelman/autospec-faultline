"""Capture of what a refinement operator consumed, for the toy 9-rooms report.

The toy report stores the operator's inputs - the sampled trajectories on the bottleneck
edge - as plain points and line parameters, so a figure can be redrawn from JSON. The
figure code never recomputes a hull or a hyperplane: :mod:`faultline.viz.replay` feeds
these points back through the real operator, so the figure shows exactly its result.

``seq`` composes two sub-effects, ReachRefine on the target and AvoidRefine on the
safe region, so its overlay captures both: the target visits and the tails of the
episodes that ended outside the safe region. A scenario whose rollouts never end
unsafe records an empty tail set and its figure is unaffected.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from faultline.experiments.records import Overlay
from faultline.geometry import half_plane_partition
from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.learners import CannotRefineError, EdgeEstimate
from faultline.refinement.operators import (
    AVOID_TAIL_LENGTH,
    ended_unsafe,
    or_refine,
    reach_witnesses,
)
from faultline.spectrl.predicates import PLANAR_DIMS


def build_overlay(
    operator: str,
    graph: AbstractGraph,
    edge: Edge,
    estimate: EdgeEstimate,
    avoid_tail_length: int = AVOID_TAIL_LENGTH,
) -> Overlay:
    """Capture the geometry a refinement operator consumed and produced.

    Args:
        operator: Operator name (``"seq"``, ``"add"``, ``"past"``, ``"or"``).
        graph: The graph the operator ran on.
        edge: The bottleneck edge it refined.
        estimate: The edge's final estimate, whose rollouts it consumed.
        avoid_tail_length: Trailing-state count AvoidRefine blocks per unsafe
            episode; only a ``seq`` overlay captures those tails. Defaults to the
            operator's own ``AVOID_TAIL_LENGTH``.

    How:
        Dispatches on the operator. ``seq`` stores both of its sub-effects'
        inputs - the target visits ReachRefine hulls and the tails of the episodes
        that ended outside the safe region, which AvoidRefine subtracts; waypoint
        refinement stores the successful midpoints; past refinement stores the
        success and failure starts plus the fitted half-plane; OrRefine is
        structural, so it stores the added edge's vertex names.

    Returns:
        The :class:`~faultline.experiments.records.Overlay` for the figure.

    Raises:
        ValueError: If ``operator`` is not one of the four names.
    """
    if operator == "seq":
        return _seq_overlay(edge, estimate, avoid_tail_length)
    if operator == "add":
        return _waypoint_overlay(estimate)
    if operator == "past":
        return _past_overlay(estimate)
    if operator == "or":
        return _or_overlay(graph, edge)
    raise ValueError(f"Unknown refinement operator {operator!r}")


def _seq_overlay(edge: Edge, estimate: EdgeEstimate, tail_length: int) -> Overlay:
    """Capture both SeqRefine sub-effects' inputs.

    Args:
        edge: The bottleneck edge whose target and safe regions are refined.
        estimate: The edge's final estimate, supplying the rollouts.
        tail_length: Trailing-state count AvoidRefine blocks per episode that
            ended outside the safe region.

    How:
        Keeps the target-region states that successful episodes visited, the set
        ReachRefine hulls (:func:`~faultline.refinement.operators.reach_witnesses`),
        and the last ``tail_length`` states of every rollout that ended
        outside the safe region, the set AvoidRefine hulls and subtracts. The
        end-of-episode filter is the paper's ``ζ_i[-1] ∉ c``; an episode that
        merely crossed the boundary and recovered is not a witness. An edge with
        no such rollout yields an empty tail set, so its figure shows only the
        target half.

    Returns:
        A ``hull`` overlay carrying the target visits and the unsafe tails.
    """
    visited = reach_witnesses(edge.target.region, estimate.rollouts)
    tails = (
        state
        for rollout in estimate.rollouts
        if ended_unsafe(rollout, edge.safe)
        for state in rollout.states[-tail_length:]
    )
    return Overlay("hull", _to_pairs(visited), (), None, None, None, _to_pairs(tails))


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
