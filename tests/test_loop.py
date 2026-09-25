"""Tests for the Algorithm 1 loop.

The loop's decisions are pinned with a scripted learner that returns controlled
success probabilities and real rollouts, so each test isolates one behaviour:
operator order, first-clearing commit, the no-refinement fallback, topological
scan order, relearn-after-commit, the best-path product, and determinism. A final
smoke test runs the real tabular learner to prove the wiring.
"""

from __future__ import annotations

import pytest

from faultline.envs import Rooms9, default_regions
from faultline.graph import AbstractGraph
from faultline.learners import TabularLearner
from faultline.refinement import AutoSpecConfig, run_autospec
from faultline.spectrl.predicates import Universe
from tests.loop_support import ScriptedLearner, blocked_rollouts, refinable_rollouts
from tests.refinement_support import GOAL, MID, SAFE, START, chain_graph, two_parent_graph

THRESHOLD = 0.9


def test_scans_edges_in_topological_order():
    graph = AbstractGraph()
    start = graph.add_vertex("start", START)
    mid = graph.add_vertex("mid", MID)
    goal = graph.add_vertex("goal", GOAL)
    graph.add_edge(mid, goal, Universe())
    graph.add_edge(start, mid, SAFE)
    graph.source = start
    graph.target = goal

    def probability(edge):
        return 0.5

    learner = ScriptedLearner(probability, blocked_rollouts)
    run = run_autospec(graph, None, learner, AutoSpecConfig(threshold=THRESHOLD))

    assert [(step.source, step.target) for step in run.steps] == [
        ("start", "mid"),
        ("mid", "goal"),
    ]
    assert all(step.operator is None for step in run.steps)


def test_commits_the_first_operator_that_clears():
    graph, _edge = chain_graph()

    def probability(edge):
        if "seq" in edge.source.name or "seq" in edge.target.name:
            return 0.99
        if edge.source.name == "mid" and edge.target.name == "goal":
            return 0.99
        return 0.5

    learner = ScriptedLearner(probability, refinable_rollouts)
    run = run_autospec(graph, None, learner, AutoSpecConfig(threshold=THRESHOLD))

    assert run.steps[0].operator == "seq"
    assert run.steps[0].operators_tried == ("seq",)


def test_commits_add_when_seq_does_not_clear():
    graph, _edge = chain_graph()

    def probability(edge):
        if "seq" in edge.source.name or "seq" in edge.target.name:
            return 0.2
        if "add" in edge.source.name or "add" in edge.target.name:
            return 0.99
        if edge.source.name == "mid" and edge.target.name == "goal":
            return 0.99
        return 0.5

    learner = ScriptedLearner(probability, refinable_rollouts)
    run = run_autospec(graph, None, learner, AutoSpecConfig(threshold=THRESHOLD))

    assert run.steps[0].operators_tried == ("seq", "add")
    assert run.steps[0].operator == "add"


def test_falls_back_when_no_operator_clears():
    graph, _edge = chain_graph()

    def probability(edge):
        return 0.5

    learner = ScriptedLearner(probability, refinable_rollouts)
    run = run_autospec(graph, None, learner, AutoSpecConfig(threshold=THRESHOLD))

    assert [step.operator for step in run.steps] == [None, None]
    assert all(step.before == step.after for step in run.steps)
    assert [(edge.source.name, edge.target.name) for edge in run.graph.edges] == [
        ("start", "mid"),
        ("mid", "goal"),
    ]
    assert len(run.graph.vertices) == 3


def test_relearns_only_edges_the_edit_changed():
    graph, _edge = chain_graph()

    def probability(edge):
        if "seq" in edge.source.name or "seq" in edge.target.name:
            return 0.99
        if edge.source.name == "mid" and edge.target.name == "goal":
            return 0.95
        return 0.5

    learner = ScriptedLearner(probability, refinable_rollouts)
    run = run_autospec(graph, None, learner, AutoSpecConfig(threshold=THRESHOLD))

    assert run.steps[0].operator == "seq"
    assert learner.count("mid", "goal") == 1
    assert learner.count("start", "mid") == 1
    assert learner.count("start", "seq1") == 1
    assert learner.count("seq1", "mid") == 1


def test_commits_or_refine_when_earlier_operators_refuse():
    graph, _blocked, _sibling = two_parent_graph()

    def probability(edge):
        if edge.source.name == "start" and edge.target.name == "goal":
            return 0.4
        return 0.9

    learner = ScriptedLearner(probability, blocked_rollouts)
    run = run_autospec(graph, None, learner, AutoSpecConfig(threshold=0.8))

    assert run.steps[0].operators_tried == ("seq", "add", "past", "or")
    assert run.steps[0].operator == "or"
    assert run.best_path_success == pytest.approx(0.81)


def test_reports_best_path_product():
    graph, _edge = chain_graph()

    def probability(edge):
        if edge.source.name == "start":
            return 0.8
        return 0.9

    learner = ScriptedLearner(probability, refinable_rollouts)
    run = run_autospec(graph, None, learner, AutoSpecConfig(threshold=0.5))

    assert run.steps == ()
    assert run.best_path_success == pytest.approx(0.72)


def test_run_is_deterministic():
    def probability(edge):
        if "seq" in edge.source.name or "seq" in edge.target.name:
            return 0.99
        if edge.source.name == "mid" and edge.target.name == "goal":
            return 0.95
        return 0.5

    config = AutoSpecConfig(threshold=THRESHOLD)
    first_graph, _ = chain_graph()
    second_graph, _ = chain_graph()
    first = run_autospec(first_graph, None, ScriptedLearner(probability, refinable_rollouts), config)
    second = run_autospec(
        second_graph, None, ScriptedLearner(probability, refinable_rollouts), config
    )

    assert first.steps == second.steps
    assert first.edge_success == second.edge_success
    assert first.best_path_success == second.best_path_success


def test_runs_with_the_tabular_learner():
    regions = default_regions()
    graph = AbstractGraph()
    start = graph.add_vertex("start", regions["start"])
    goal = graph.add_vertex("goal", regions["goal"])
    graph.add_edge(start, goal)
    graph.source = start
    graph.target = goal
    env = Rooms9()
    config = AutoSpecConfig(threshold=1.0, n_rollouts=5, seed=0)

    run = run_autospec(graph, env, TabularLearner(), config)

    assert run.steps == ()
    assert run.best_path_success == pytest.approx(1.0)
