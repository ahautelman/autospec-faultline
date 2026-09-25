"""Behavioural tests for the SpectRL-to-graph compiler and graph semantics.

The centrepiece is the equivalence test: over every trajectory in a small
enumerated state space, the compiled graph must accept exactly the trajectories
the reference specification semantics accepts. The remaining tests pin down the
compiler's structural output, the source-region label, the ensuring behaviour,
and whole-graph success.
"""

from __future__ import annotations

import itertools

import pytest

from faultline.graph import best_path_product, compile_spec, models_trajectory
from faultline.spectrl import Achieve, Choice, Ensuring, Seq
from faultline.spectrl.predicates import Box, Universe
from faultline.spectrl.semantics import models_spec

# Single-point regions make "reaches the region" exactly "equals the point".
R_A = Box((0.0, 0.0), (0.0, 0.0))
R_B = Box((1.0, 1.0), (1.0, 1.0))
R_C = Box((2.0, 2.0), (2.0, 2.0))
# A safety region that excludes R_C but admits the other three sample points.
LIMIT = Box((0.0, 0.0), (1.5, 1.5))

POINTS = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.5, 0.5]]
LENGTH = 3

SPECS = [
    Achieve(R_A),
    Seq(Achieve(R_A), Achieve(R_B)),
    Choice(Achieve(R_A), Achieve(R_B)),
    Ensuring(Achieve(R_C), LIMIT),
    Ensuring(Seq(Achieve(R_A), Achieve(R_B)), LIMIT),
    Choice(Seq(Achieve(R_A), Achieve(R_B)), Achieve(R_C)),
    Seq(Achieve(R_A), Choice(Achieve(R_B), Achieve(R_C))),
]


def _all_trajectories():
    """Enumerate every trajectory of length LENGTH over the sample points."""
    return [list(combo) for combo in itertools.product(POINTS, repeat=LENGTH)]


@pytest.mark.parametrize("spec", SPECS)
def test_compiled_graph_agrees_with_spec_semantics(spec):
    """Compiling a spec preserves satisfaction for every enumerated trajectory."""
    graph = compile_spec(spec)
    mismatches = [
        trajectory
        for trajectory in _all_trajectories()
        if models_trajectory(graph, trajectory) != models_spec(spec, trajectory)
    ]
    assert mismatches == []


def test_achieve_compiles_to_source_and_target_with_one_edge():
    """An achieve spec becomes two vertices joined by a single unconstrained edge."""
    graph = compile_spec(Achieve(R_A))
    assert len(graph.vertices) == 2
    assert len(graph.edges) == 1
    assert graph.source.region == Universe()
    assert graph.target.region == R_A
    assert graph.edges[0].source is graph.source
    assert graph.edges[0].target is graph.target


def test_sequence_chains_steps_with_a_bridge_edge():
    """A two-step sequence yields four vertices and three edges."""
    graph = compile_spec(Seq(Achieve(R_A), Achieve(R_B)))
    assert len(graph.vertices) == 4
    assert len(graph.edges) == 3
    assert graph.target.region == R_B


def test_choice_fans_out_and_merges():
    """A two-branch choice yields six vertices and six edges."""
    graph = compile_spec(Choice(Achieve(R_A), Achieve(R_B)))
    # start, merge, and two vertices per branch; each branch contributes its
    # internal edge plus a fan-in and a fan-out edge.
    assert len(graph.vertices) == 6
    assert len(graph.edges) == 6


def test_source_region_labels_the_source_vertex():
    """The source_region argument is written onto the compiled graph's source."""
    graph = compile_spec(Achieve(R_A), source_region=LIMIT)
    assert graph.source.region == LIMIT


def test_ensuring_tightens_every_body_edge():
    """Ensuring intersects its predicate into each edge the body introduced."""
    graph = compile_spec(Ensuring(Seq(Achieve(R_A), Achieve(R_B)), LIMIT))
    mismatches = [
        (edge, point)
        for edge in graph.edges
        for point in POINTS
        if edge.safe.evaluate(point) != LIMIT.evaluate(point)
    ]
    assert mismatches == []


def test_ensuring_can_render_a_spec_unsatisfiable():
    """Reaching R_C while staying inside LIMIT is impossible, in spec and graph."""
    spec = Ensuring(Achieve(R_C), LIMIT)
    graph = compile_spec(spec)
    reaching_trajectory = [[0.0, 0.0], [2.0, 2.0]]
    assert models_spec(spec, reaching_trajectory) is False
    assert models_trajectory(graph, reaching_trajectory) is False


def test_models_trajectory_accepts_a_reaching_trajectory():
    """A trajectory that visits the target region satisfies the graph."""
    assert models_trajectory(compile_spec(Achieve(R_B)), [[0.0, 0.0], [1.0, 1.0]]) is True


def test_models_trajectory_rejects_a_trajectory_that_never_reaches():
    """A trajectory that never visits the target region does not satisfy the graph."""
    assert models_trajectory(compile_spec(Achieve(R_B)), [[0.0, 0.0], [2.0, 2.0]]) is False


def test_best_path_product_multiplies_a_single_path():
    """With one path, whole-graph success is the product of its edge estimates."""
    graph = compile_spec(Seq(Achieve(R_A), Achieve(R_B)))
    probabilities = {
        edge: probability for edge, probability in zip(graph.edges, (0.5, 1.0, 0.8), strict=True)
    }
    assert best_path_product(graph, probabilities) == pytest.approx(0.4)


def test_best_path_product_selects_the_higher_probability_branch():
    """With two branches, whole-graph success is the better branch's product."""
    graph = compile_spec(Choice(Achieve(R_A), Achieve(R_B)))
    # Edges are inserted branch by branch: three for branch A then three for B.
    branch_a = (0.8, 1.0, 0.5)
    branch_b = (0.7, 0.7, 0.7)
    probabilities = {
        edge: probability
        for edge, probability in zip(graph.edges, (*branch_a, *branch_b), strict=True)
    }
    assert best_path_product(graph, probabilities) == pytest.approx(0.4)


def test_best_path_product_requires_every_edge_estimate():
    """A missing edge estimate is an error, not a silent zero."""
    graph = compile_spec(Achieve(R_A))
    with pytest.raises(ValueError, match="No success estimate"):
        best_path_product(graph, {})
