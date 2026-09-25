"""Soundness tests for the four refinement operators (paper section 5).

Theorem 1 says every trajectory that satisfies a refined graph also satisfies the
original. Each test builds a graph, refines one edge from synthetic rollouts, and
checks the implication over every short trajectory in the shared enumerated state
space. That is the paper's graph-level guarantee, exercised through the real
operator code and the real graph semantics.
"""

from __future__ import annotations

from faultline.refinement import add_refine, or_refine, past_refine, seq_refine
from tests.refinement_support import (
    assert_soundness,
    chain_graph,
    estimate,
    rollout,
    two_parent_graph,
)


def test_seq_refine_is_sound():
    graph, edge = chain_graph()
    rollouts = (
        rollout([[0.5, 0.5], [1.5, 1.5]], reached=True, safe=True),
        rollout([[0.5, 0.5], [2.6, 2.6]], reached=False, safe=False),
    )
    assert_soundness(graph, seq_refine(graph, edge, estimate(rollouts)))


def test_add_refine_is_sound():
    graph, edge = chain_graph()
    rollouts = (
        rollout([[0.2, 0.2], [1.2, 1.2], [1.6, 1.6]], reached=True, safe=True),
        rollout([[0.8, 0.8], [1.4, 1.4], [1.9, 1.9]], reached=True, safe=True),
    )
    assert_soundness(graph, add_refine(graph, edge, estimate(rollouts)))


def test_past_refine_is_sound():
    graph, edge = chain_graph()
    rollouts = (
        rollout([[0.1, 0.1], [1.5, 1.5]], reached=True, safe=True),
        rollout([[0.2, 0.2], [1.5, 1.5]], reached=True, safe=True),
        rollout([[0.9, 0.9], [0.9, 0.9]], reached=False, safe=True),
    )
    assert_soundness(graph, past_refine(graph, edge, estimate(rollouts)))


def test_or_refine_is_sound():
    graph, blocked, _sibling = two_parent_graph()
    assert_soundness(graph, or_refine(graph, blocked))
