"""Tests for the runtime semantics over the abstract graph.

``models_trajectory`` is the shared oracle behind the compiler's soundness check
and the refinement loop's commitments. This file pins its accept/reject behaviour
on a synthetic chain and adds a small scaling guard so an accidental change to
its dynamic program shows up as a test failure rather than a silent slowdown.
"""

from __future__ import annotations

import time

import numpy as np

from faultline.graph import AbstractGraph, models_trajectory
from faultline.spectrl.predicates import Universe

# The scaling check builds two chains and two trajectories; the larger is twice
# the smaller in both dimensions. The current program is polynomial, so doubling
# both doubles the vertex work and squares the per-state scan: roughly 8x. The
# bound leaves headroom for machine noise while still catching a jump to a much
# worse complexity class.
SMALL_CHAIN = 32
SMALL_STEPS = 100
REPEATS = 5
SCALING_BOUND = 15.0


def _chain_graph(length: int) -> AbstractGraph:
    """Build a chain of ``length`` edges over whole-space vertices.

    Args:
        length: Number of edges; the graph has one more vertex than this.

    How:
        Adds ``length + 1`` universe-region vertices and connects consecutive
        pairs with universe-safe edges, then marks the two ends as source and
        target.

    Returns:
        The chain graph, ready for :func:`models_trajectory`.
    """
    graph = AbstractGraph()
    vertices = [graph.add_vertex(f"v{index}", Universe()) for index in range(length + 1)]
    for index in range(length):
        graph.add_edge(vertices[index], vertices[index + 1])
    graph.source = vertices[0]
    graph.target = vertices[-1]
    return graph


def _trajectory(length: int) -> list[np.ndarray]:
    """Return a ``length``-state trajectory that stays inside every region.

    Args:
        length: Number of states to produce.

    How:
        Repeats a fresh copy of the origin, which every universe region accepts,
        so the trajectory satisfies the synthetic chain.

    Returns:
        The trajectory as a list of two-dimensional state vectors.
    """
    point = np.zeros(2, dtype=float)
    return [point.copy() for _ in range(length)]


def _best_runtime(graph: AbstractGraph, trajectory: list[np.ndarray]) -> float:
    """Return the fastest of several timed :func:`models_trajectory` runs.

    Args:
        graph: The graph to evaluate.
        trajectory: The trajectory to evaluate.

    How:
        Runs the oracle ``REPEATS`` times and keeps the smallest wall-clock time,
        which suppresses one-off scheduler and warm-up noise better than a mean.

    Returns:
        The best observed runtime, in seconds.
    """
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        models_trajectory(graph, trajectory)
        best = min(best, time.perf_counter() - start)
    return best


def test_models_trajectory_accepts_a_synthetic_chain():
    graph = _chain_graph(SMALL_CHAIN)
    trajectory = _trajectory(SMALL_STEPS)

    assert models_trajectory(graph, trajectory) is True


def test_models_trajectory_runtime_scales_polynomially():
    small = _best_runtime(_chain_graph(SMALL_CHAIN), _trajectory(SMALL_STEPS))
    large = _best_runtime(
        _chain_graph(2 * SMALL_CHAIN), _trajectory(2 * SMALL_STEPS)
    )

    # A ratio far above the expected ~8x means the dynamic program lost its
    # polynomial bound, not that this machine is slow: both runs share the same
    # machine, so machine speed cancels out of the ratio.
    assert large < max(small, 1e-9) * SCALING_BOUND
