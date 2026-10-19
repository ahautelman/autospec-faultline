"""Whole-graph learning curves from the per-edge trainings of one loop run.

The toy report plots the bottleneck edge before refinement against the refined route
after it. The full loop has no single bottleneck, so the same idea is applied to the
whole graph: the best source-to-target path's success, step by step.
"""

from __future__ import annotations

from collections.abc import Mapping

from faultline.experiments.records import Checkpoint
from faultline.experiments.training import TrainingRecord, route_product
from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.refinement.loop import _edge_key as edge_key
from faultline.refinement.trajectories import TrajectorySample, sample_trajectories


def best_path(graph: AbstractGraph, probabilities: Mapping[Edge, float]) -> tuple[Edge, ...]:
    """Return the source-to-target path with the largest probability product.

    Args:
        graph: The graph, with source and target set.
        probabilities: Final success probability per edge of the graph.

    How:
        Runs a longest-path pass in topological order, keeping the best product
        reaching each vertex and the edge that achieved it, then walks the
        predecessor edges back from the target.

    Returns:
        The path's edges in order, or ``()`` when the target is unreachable.
    """
    best = {vertex: -1.0 for vertex in graph.vertices}
    arrived_by: dict = {}
    best[graph.source] = 1.0
    for vertex in graph.topological_order():
        if best[vertex] < 0:
            continue
        for edge in graph.successors(vertex):
            product = best[vertex] * probabilities[edge]
            if product > best[edge.target]:
                best[edge.target] = product
                arrived_by[edge.target] = edge
    path: list[Edge] = []
    vertex = graph.target
    while vertex in arrived_by:
        path.append(arrived_by[vertex])
        vertex = arrived_by[vertex].source
    return tuple(reversed(path))


def graph_curve(
    graph: AbstractGraph, trainings: Mapping[tuple, TrainingRecord]
) -> tuple[Checkpoint, ...]:
    """Return the best path's learning curve for a graph.

    Args:
        graph: The graph whose edges all have a training in ``trainings``.
        trainings: Training records keyed by the loop's edge key.

    How:
        Picks the best path by final success, then multiplies that path's edge
        curves step by step. A graph with no path to the target gets an all-zero
        curve on the checkpoint steps of its edges.

    Returns:
        The curve, ordered by step.
    """
    records, path = _records_and_path(graph, trainings)
    if path:
        return route_product([records[edge] for edge in path])
    steps = (checkpoint.step for checkpoint in next(iter(records.values())).checkpoints)
    return tuple(Checkpoint(step, 0.0) for step in steps)


def path_trajectories(
    graph: AbstractGraph, trainings: Mapping[tuple, TrainingRecord]
) -> tuple[TrajectorySample, ...]:
    """Return sampled rollouts of each edge on a graph's best path.

    Args:
        graph: The graph whose edges all have a training in ``trainings``.
        trainings: Training records keyed by the loop's edge key.

    How:
        Finds the best path and samples each of its edges' final rollouts. Each
        edge is trained from its own sampled start states, so the samples are
        per-edge segments, not end-to-end paths.

    Returns:
        The sampled rollouts in path order, empty when the target is unreachable.
    """
    records, path = _records_and_path(graph, trainings)
    return tuple(
        sample for edge in path for sample in sample_trajectories(edge, records[edge].estimate)
    )


def _records_and_path(
    graph: AbstractGraph, trainings: Mapping[tuple, TrainingRecord]
) -> tuple[dict[Edge, TrainingRecord], tuple[Edge, ...]]:
    """Look up each edge's training and find the graph's best path.

    Args:
        graph: The graph whose edges all have a training in ``trainings``.
        trainings: Training records keyed by the loop's edge key.

    How:
        Keys the records by edge and runs :func:`best_path` on their final success.

    Returns:
        The records by edge and the best path's edges.
    """
    records = {edge: trainings[edge_key(edge)] for edge in graph.edges}
    path = best_path(graph, {edge: r.estimate.success_probability for edge, r in records.items()})
    return records, path
