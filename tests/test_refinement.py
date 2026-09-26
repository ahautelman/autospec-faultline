"""Behavioural and guard tests for the refinement operators.

Each operator is pinned on synthetic trajectories: the exact region ReachRefine
and AvoidRefine produce, the structural edit the graph operators make, and the
refusal guard on a sample-starved edge. The soundness of the same operators is
checked separately in ``test_refinement_soundness.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from faultline.learners import CannotRefineError
from faultline.refinement import (
    add_refine,
    avoid_refine,
    or_refine,
    past_refine,
    reach_refine,
    seq_refine,
)
from faultline.refinement.operators import _copied_edge
from tests.refinement_support import (
    MID,
    SAFE,
    chain_graph,
    estimate,
    rollout,
    two_hop_graph,
    two_parent_graph,
)


def test_reach_refine_intersects_the_region_with_the_visited_hull():
    rollouts = (rollout([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]], reached=True, safe=True),)
    refined = reach_refine(MID, rollouts)
    assert refined.evaluate(np.array([1.2, 1.2])) is True
    assert refined.evaluate(np.array([1.9, 1.9])) is False
    assert refined.evaluate(np.array([0.5, 0.5])) is False


def test_reach_refine_raises_when_no_state_visits_the_region():
    rollouts = (rollout([[0.5, 0.5], [0.6, 0.6]], reached=False, safe=True),)
    with pytest.raises(CannotRefineError):
        reach_refine(MID, rollouts)


def test_avoid_refine_subtracts_the_unsafe_tail_hull():
    rollouts = (rollout([[0.5, 0.5], [2.6, 2.0], [2.0, 2.6]], reached=False, safe=False),)
    refined = avoid_refine(SAFE, rollouts)
    assert refined.evaluate(np.array([0.5, 0.5])) is False
    assert refined.evaluate(np.array([2.4, 0.1])) is True


def test_avoid_refine_raises_when_every_rollout_stayed_safe():
    rollouts = (rollout([[0.5, 0.5], [1.0, 1.0]], reached=True, safe=True),)
    with pytest.raises(CannotRefineError):
        avoid_refine(SAFE, rollouts)


def test_avoid_refine_rejects_a_non_positive_tail_length():
    rollouts = (rollout([[0.5, 0.5]], reached=False, safe=False),)
    with pytest.raises(ValueError):
        avoid_refine(SAFE, rollouts, k=0)


def test_seq_refine_replaces_the_edge_with_a_tighter_two_hop_path():
    graph, edge = chain_graph()
    rollouts = (
        rollout([[0.5, 0.5], [1.5, 1.5]], reached=True, safe=True),
        rollout([[0.5, 0.5], [2.6, 2.6]], reached=False, safe=False),
    )
    refined = seq_refine(graph, edge, estimate(rollouts))
    assert len(refined.vertices) == len(graph.vertices) + 1
    assert len(refined.edges) == len(graph.edges) + 1
    assert len(graph.edges) == 2
    vertex = next(candidate for candidate in refined.vertices if candidate.name.startswith("seq"))
    assert vertex.region.evaluate(np.array([1.5, 1.5])) is True
    assert vertex.region.evaluate(np.array([1.2, 1.2])) is False
    new_edge = refined.predecessors(vertex)[0]
    assert new_edge.safe.evaluate(np.array([0.5, 0.5])) is False


def test_seq_refine_raises_when_neither_region_can_move():
    graph, edge = chain_graph()
    rollouts = (rollout([[0.5, 0.5], [0.6, 0.6]], reached=False, safe=True),)
    with pytest.raises(CannotRefineError):
        seq_refine(graph, edge, estimate(rollouts))


def test_add_refine_inserts_a_waypoint_between_the_edges():
    graph, edge = chain_graph()
    rollouts = (
        rollout([[0.2, 0.2], [1.2, 1.2], [1.6, 1.6]], reached=True, safe=True),
        rollout([[0.8, 0.8], [1.4, 1.4], [1.9, 1.9]], reached=True, safe=True),
    )
    refined = add_refine(graph, edge, estimate(rollouts))
    assert len(refined.vertices) == len(graph.vertices) + 1
    assert len(refined.edges) == len(graph.edges) + 1
    vertex = next(candidate for candidate in refined.vertices if candidate.name.startswith("add"))
    assert vertex.region.evaluate(np.array([1.3, 1.3])) is True
    assert vertex.region.evaluate(np.array([0.5, 0.5])) is False


def test_add_refine_raises_without_a_successful_rollout():
    graph, edge = chain_graph()
    rollouts = (rollout([[0.2, 0.2], [1.5, 1.5]], reached=False, safe=True),)
    with pytest.raises(CannotRefineError):
        add_refine(graph, edge, estimate(rollouts))


def test_past_refine_tightens_the_seed_region_when_the_source_is_the_graph_source():
    graph, edge = chain_graph()
    rollouts = (
        rollout([[0.1, 0.1], [1.5, 1.5]], reached=True, safe=True),
        rollout([[0.2, 0.2], [1.5, 1.5]], reached=True, safe=True),
        rollout([[0.9, 0.9], [0.9, 0.9]], reached=False, safe=True),
    )
    refined = past_refine(graph, edge, estimate(rollouts))
    assert refined.source.region.evaluate(np.array([0.1, 0.1])) is True
    assert refined.source.region.evaluate(np.array([0.9, 0.9])) is False
    assert graph.source.region.evaluate(np.array([0.9, 0.9])) is True


def test_past_refine_adds_a_refined_entry_for_a_later_edge():
    graph, later = two_hop_graph()
    rollouts = (
        rollout([[0.1, 0.1], [1.5, 1.5]], reached=True, safe=True),
        rollout([[0.2, 0.2], [1.5, 1.5]], reached=True, safe=True),
        rollout([[0.9, 0.9], [0.9, 0.9]], reached=False, safe=True),
    )
    refined = past_refine(graph, later, estimate(rollouts))
    vertex = next(candidate for candidate in refined.vertices if candidate.name.startswith("past"))
    assert vertex.region.evaluate(np.array([0.1, 0.1])) is True
    assert vertex.region.evaluate(np.array([0.9, 0.9])) is False
    assert [edge.source.name for edge in refined.predecessors(vertex)] == ["start"]


def test_past_refine_raises_without_a_successful_rollout():
    graph, edge = chain_graph()
    rollouts = (rollout([[0.9, 0.9], [0.9, 0.9]], reached=False, safe=True),)
    with pytest.raises(CannotRefineError):
        past_refine(graph, edge, estimate(rollouts))


def test_past_refine_raises_when_every_start_succeeds():
    graph, edge = chain_graph()
    rollouts = (
        rollout([[0.1, 0.1], [1.5, 1.5]], reached=True, safe=True),
        rollout([[0.2, 0.2], [1.5, 1.5]], reached=True, safe=True),
    )
    with pytest.raises(CannotRefineError):
        past_refine(graph, edge, estimate(rollouts))


def test_or_refine_adds_a_route_through_the_sibling_parent():
    graph, blocked, _sibling = two_parent_graph()
    refined = or_refine(graph, blocked)
    assert len(refined.edges) == len(graph.edges) + 1
    new_edges = [
        edge
        for edge in refined.edges
        if edge.source.name == "start" and edge.target.name == "alt"
    ]
    assert len(new_edges) == 1
    assert new_edges[0].safe.evaluate(np.array([2.6, 2.6])) is False
    tightened = [
        edge for edge in refined.edges if edge.source.name == "alt" and edge.target.name == "goal"
    ]
    assert tightened[0].safe.evaluate(np.array([2.6, 2.6])) is False


def test_or_refine_raises_without_an_alternative_parent():
    graph, edge = chain_graph()
    with pytest.raises(CannotRefineError):
        or_refine(graph, edge)


def test_copied_edge_returns_the_corresponding_copied_edge():
    graph, edge = chain_graph()
    refined = graph.copy()

    copied = _copied_edge(refined, graph, edge)

    assert copied is refined.edges[0]
    assert copied is not edge
    assert (copied.source.name, copied.target.name) == (edge.source.name, edge.target.name)
    assert copied.safe == edge.safe


def test_copied_edge_preserves_edge_order_for_every_edge():
    graph, _edge = chain_graph()
    refined = graph.copy()

    assert [_copied_edge(refined, graph, item) for item in graph.edges] == list(refined.edges)


def test_copied_edge_rejects_an_edge_from_another_graph():
    graph, _edge = chain_graph()
    refined = graph.copy()
    _other_graph, foreign = chain_graph()

    with pytest.raises(ValueError):
        _copied_edge(refined, graph, foreign)


def test_copied_edge_rejects_a_copy_that_lost_the_edge_order():
    graph, edge = chain_graph()
    refined = graph.copy()
    moved = refined.edges[0]
    refined.remove_edge(moved)
    refined.add_edge(moved.source, moved.target, moved.safe)

    with pytest.raises(ValueError):
        _copied_edge(refined, graph, edge)
