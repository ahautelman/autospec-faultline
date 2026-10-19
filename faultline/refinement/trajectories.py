"""Bounded samples of an edge's rollouts, kept for figures.

A full set of rollouts is far too large to persist, and a figure only needs a few
representative paths plus the states the operators looked at. These helpers cut an
estimate down to evenly spaced rollouts, strided in time with the endpoints kept, and
collect the witness and unsafe-tail states AvoidRefine and ReachRefine consume.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from faultline.graph.abstract_graph import Edge
from faultline.learners.base import EdgeEstimate, Rollout
from faultline.refinement.operators import ended_unsafe, reach_witnesses

MAX_POINTS = 200
PER_CLASS = 6
MAX_TRAJECTORY_POINTS = 60

Point = tuple[float, float]


@dataclass(frozen=True)
class TrajectorySample:
    """One sampled rollout of an edge, reduced to its planar path.

    Args:
        source: Source vertex name of the edge that was rolled out.
        target: Target vertex name of that edge.
        success: Whether the rollout reached the target safely.
        xy: Strided ``(x, y)`` positions in time order, endpoints included.
    """

    source: str
    target: str
    success: bool
    xy: tuple[Point, ...]


@dataclass(frozen=True)
class StepSamples:
    """The states a refinement step looked at, plus a few sampled paths.

    Args:
        witness_points: Target-region states visited by successful rollouts
            (ReachRefine's witnesses), evenly thinned.
        unsafe_points: Last states of rollouts that ended unsafe (AvoidRefine's
            tails), evenly thinned.
        trajectories: Sampled successful and failed rollouts of the edge.
    """

    witness_points: tuple[Point, ...]
    unsafe_points: tuple[Point, ...]
    trajectories: tuple[TrajectorySample, ...]


def sample_trajectories(
    edge: Edge,
    estimate: EdgeEstimate,
    per_class: int = PER_CLASS,
    max_points: int = MAX_TRAJECTORY_POINTS,
) -> tuple[TrajectorySample, ...]:
    """Sample successful and failed rollouts of an edge.

    Args:
        edge: The edge the estimate belongs to.
        estimate: The edge's estimate, supplying the rollouts.
        per_class: Rollouts to keep from each of the two outcome classes.
        max_points: Most positions kept per rollout.

    How:
        Splits the rollouts by success, picks evenly spaced ones from each class,
        and strides each path in time so it keeps its first and last state.

    Returns:
        The sampled rollouts, successes first.
    """
    successes = [r for r in estimate.rollouts if r.success]
    failures = [r for r in estimate.rollouts if not r.success]
    return tuple(
        TrajectorySample(edge.source.name, edge.target.name, success, _stride_path(rollout, max_points))
        for success, group in ((True, successes), (False, failures))
        for rollout in _evenly(group, per_class)
    )


def collect_step_samples(
    edge: Edge, estimate: EdgeEstimate, tail_length: int, max_points: int = MAX_POINTS
) -> StepSamples:
    """Collect the points and paths a refinement of ``edge`` was based on.

    Args:
        edge: The weak edge being refined.
        estimate: Its estimate, supplying the rollouts.
        tail_length: Trailing states per unsafe rollout, AvoidRefine's ``k``.
        max_points: Most points kept in each point set.

    How:
        Reads the witnesses with the operators' own helpers, so the figure shows
        exactly the states the hulls were built from, then thins each set.

    Returns:
        The step's samples.
    """
    witnesses = reach_witnesses(edge.target.region, estimate.rollouts)
    tails = [
        state
        for rollout in estimate.rollouts
        if ended_unsafe(rollout, edge.safe)
        for state in rollout.states[-tail_length:]
    ]
    return StepSamples(
        witness_points=_points(_evenly(witnesses, max_points)),
        unsafe_points=_points(_evenly(tails, max_points)),
        trajectories=sample_trajectories(edge, estimate),
    )


def thin_points(points: Sequence[Point], count: int = MAX_POINTS) -> tuple[Point, ...]:
    """Thin a point set to at most ``count`` evenly spaced points.

    Args:
        points: ``(x, y)`` points in order.
        count: Most points to keep.

    How:
        Picks evenly spaced points, always keeping the first and last, so the thinned
        set covers the same extent as the original.

    Returns:
        The kept points in their original order.
    """
    return tuple(_evenly(points, count))


def _evenly(items: Sequence[Any], count: int) -> list[Any]:
    """Pick ``count`` evenly spaced items, or all of them when there are fewer."""
    if len(items) <= count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[index] for index in indices]


def _stride_path(rollout: Rollout, max_points: int) -> tuple[Point, ...]:
    """Reduce a rollout to at most ``max_points`` planar positions, ends kept."""
    return _points(_evenly(rollout.states, max_points))


def _points(states: Sequence[np.ndarray]) -> tuple[Point, ...]:
    """Project states onto their ``(x, y)`` coordinates as plain floats."""
    return tuple((float(state[0]), float(state[1])) for state in states)
