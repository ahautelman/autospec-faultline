"""Shared fixtures for the refinement operator tests.

Holds the synthetic regions, rollouts, and hand-built graphs both the behaviour
tests and the soundness tests use, so the two files do not duplicate them.
"""

from __future__ import annotations

import itertools

import numpy as np

from faultline.graph import AbstractGraph, models_trajectory
from faultline.learners import EdgeEstimate, Rollout
from faultline.spectrl.predicates import Box, Universe

# Checkpoint and safety regions for the hand-built test graphs. MID overlaps the
# boundary of START on purpose so a trajectory can cross it in one step.
START = Box((0.0, 0.0), (1.0, 1.0))
MID = Box((1.0, 1.0), (2.0, 2.0))
GOAL = Box((2.0, 2.0), (3.0, 3.0))
ALT = Box((0.4, 0.4), (0.6, 0.6))
SAFE = Box((0.0, 0.0), (2.5, 2.5))

# Enumerated points span the three checkpoint regions plus the wall area between
# them, so the soundness check sees both accepted and rejected trajectories.
POINTS = [
    np.array(point, dtype=float)
    for point in (
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [1.5, 1.5],
        [2.0, 2.0],
        [2.5, 2.5],
        [3.0, 3.0],
    )
]
TRAJECTORY_LENGTH = 3


def rollout(states, reached, safe, final_safe=None):
    """Build a rollout from a plain list of states and its outcomes."""
    return Rollout(
        tuple(np.asarray(state, dtype=float) for state in states),
        reached=reached,
        safe=safe,
        final_safe=final_safe,
    )


def estimate(rollouts):
    """Build an edge estimate whose success rate matches the given rollouts."""
    success = sum(1 for item in rollouts if item.success) / len(rollouts)
    return EdgeEstimate(success, tuple(rollouts))


def chain_graph():
    """Return ``start -> mid -> goal`` and its ``start -> mid`` edge."""
    graph = AbstractGraph()
    start = graph.add_vertex("start", START)
    mid = graph.add_vertex("mid", MID)
    goal = graph.add_vertex("goal", GOAL)
    edge = graph.add_edge(start, mid, SAFE)
    graph.add_edge(mid, goal, Universe())
    graph.source = start
    graph.target = goal
    return graph, edge


def two_hop_graph():
    """Return ``start -> mid -> goal`` and its ``mid -> goal`` edge."""
    graph = AbstractGraph()
    start = graph.add_vertex("start", START)
    mid = graph.add_vertex("mid", START)
    goal = graph.add_vertex("goal", GOAL)
    graph.add_edge(start, mid, SAFE)
    later = graph.add_edge(mid, goal, SAFE)
    graph.source = start
    graph.target = goal
    return graph, later


def two_parent_graph():
    """Return a graph where ``goal`` has two parents, plus the blocked edge."""
    graph = AbstractGraph()
    start = graph.add_vertex("start", START)
    alt = graph.add_vertex("alt", ALT)
    goal = graph.add_vertex("goal", GOAL)
    blocked = graph.add_edge(start, goal, SAFE)
    sibling = graph.add_edge(alt, goal, Universe())
    graph.source = start
    graph.target = goal
    return graph, blocked, sibling


def cyclic_sibling_graph():
    """Return a graph where one parent of ``goal`` reaches the blocked edge's source.

    ``start -> goal`` is a sibling of the blocked ``mid -> goal``, but linking
    ``mid -> start`` would close a cycle, so OrRefine may only route via ``alt``.
    """
    graph = AbstractGraph()
    start = graph.add_vertex("start", START)
    mid = graph.add_vertex("mid", MID)
    alt = graph.add_vertex("alt", ALT)
    goal = graph.add_vertex("goal", GOAL)
    graph.add_edge(start, mid, SAFE)
    blocked = graph.add_edge(mid, goal, SAFE)
    graph.add_edge(start, goal, Universe())
    graph.add_edge(alt, goal, Universe())
    graph.source = start
    graph.target = goal
    return graph, blocked


def existing_link_graph():
    """Return a graph that already links the blocked edge's source to the sibling.

    ``mid -> alt`` exists with the universe as its safe region before OrRefine
    adds its own ``mid -> alt`` edge carrying the blocked edge's ``SAFE``.
    """
    graph = AbstractGraph()
    start = graph.add_vertex("start", START)
    mid = graph.add_vertex("mid", MID)
    alt = graph.add_vertex("alt", ALT)
    goal = graph.add_vertex("goal", GOAL)
    graph.add_edge(start, mid, SAFE)
    graph.add_edge(mid, alt, Universe())
    blocked = graph.add_edge(mid, goal, SAFE)
    graph.add_edge(alt, goal, Universe())
    graph.source = start
    graph.target = goal
    return graph, blocked


def trajectories():
    """Enumerate every short trajectory over the shared sample points."""
    return [list(combo) for combo in itertools.product(POINTS, repeat=TRAJECTORY_LENGTH)]


def assert_soundness(original, refined):
    """Fail if any trajectory satisfies ``refined`` but not ``original``."""
    violations = [
        trajectory
        for trajectory in trajectories()
        if models_trajectory(refined, trajectory) and not models_trajectory(original, trajectory)
    ]
    assert violations == []
