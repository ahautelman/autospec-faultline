"""Fast tests for the full-loop run's curves, pooled learner, records, and summary.

Training is replaced by a stub task, so nothing here trains a policy or spawns a
process; the orchestration around it is the real code.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from faultline.experiments.autospec import learner as learner_module
from faultline.experiments.autospec.curves import best_path, graph_curve
from faultline.experiments.autospec.executor import SerialExecutor
from faultline.experiments.autospec.learner import PooledCheckpointLearner
from faultline.experiments.autospec.records import (
    AutoSpecReport,
    report_from_dict,
    report_to_dict,
)
from faultline.experiments.autospec.summary import summary_table
from faultline.experiments.records import Checkpoint
from faultline.experiments.training import ReportSettings, TrainingRecord
from faultline.learners import EdgeEstimate
from faultline.refinement.loop import EdgeSuccess, RefinementStep
from faultline.refinement.loop import _edge_key as edge_key
from faultline.scenarios import SCENARIOS

BUDGET = 100
SETTINGS = ReportSettings(budget=BUDGET, rollouts=10, interval=50, intermediate_rollouts=5)


def _record(edge, curve: tuple[float, float]) -> TrainingRecord:
    """Build a two-checkpoint record whose final estimate is the curve's end."""
    checkpoints = (Checkpoint(50, curve[0]), Checkpoint(BUDGET, curve[1]))
    return TrainingRecord(
        edge.source.name, edge.target.name, checkpoints, EdgeEstimate(curve[1], ()), None
    )


def _two_path_trainings(upper: tuple[float, float], lower: tuple[float, float]):
    """Return the two-path graph and trainings: upper start->mid1->goal, lower via mid2."""
    graph = SCENARIOS["blocked_two_path"].build_graph()
    curves = {
        ("start", "mid1"): upper,
        ("mid1", "goal"): (1.0, 1.0),
        ("start", "mid2"): lower,
        ("mid2", "goal"): (1.0, 1.0),
    }
    trainings = {
        edge_key(edge): _record(edge, curves[(edge.source.name, edge.target.name)])
        for edge in graph.edges
    }
    return graph, trainings


def test_best_path_takes_the_branch_with_the_larger_product():
    graph, trainings = _two_path_trainings(upper=(0.2, 0.3), lower=(0.4, 0.8))
    probabilities = {edge: trainings[edge_key(edge)].estimate.success_probability for edge in graph.edges}

    path = best_path(graph, probabilities)

    assert [(edge.source.name, edge.target.name) for edge in path] == [("start", "mid2"), ("mid2", "goal")]


def test_graph_curve_multiplies_the_best_paths_edge_curves():
    graph, trainings = _two_path_trainings(upper=(0.2, 0.3), lower=(0.4, 0.8))

    curve = graph_curve(graph, trainings)

    assert curve == (Checkpoint(50, 0.4), Checkpoint(BUDGET, 0.8))


def test_graph_curve_is_zero_when_the_target_is_unreachable():
    graph, trainings = _two_path_trainings(upper=(0.2, 0.3), lower=(0.4, 0.8))
    for edge in list(graph.successors(graph.target)):
        graph.remove_edge(edge)
    for edge in graph.predecessors(graph.target):
        graph.remove_edge(edge)

    curve = graph_curve(graph, trainings)

    assert curve == (Checkpoint(50, 0.0), Checkpoint(BUDGET, 0.0))


@pytest.fixture
def stub_task(monkeypatch):
    """Replace the worker task with a stub that counts calls per edge."""
    calls = []

    def task(request):
        calls.append((request.edge.source.name, request.edge.target.name))
        return _record(request.edge, (0.5, 0.9))

    monkeypatch.setattr(learner_module, "train_edge_task", task)
    return calls


def _learner() -> PooledCheckpointLearner:
    return PooledCheckpointLearner("blocked_two_path", 0, SETTINGS, SerialExecutor(), quiet=True)


def test_prefetched_edges_are_trained_once(stub_task):
    graph = SCENARIOS["blocked_two_path"].build_graph()
    learner = _learner()

    learner.prefetch(graph.edges)
    for edge in graph.edges:
        learner.train_edge(None, edge, BUDGET)

    assert sorted(stub_task) == sorted((e.source.name, e.target.name) for e in graph.edges)


def test_estimate_reach_returns_the_stored_final_estimate(stub_task):
    edge = SCENARIOS["blocked_two_path"].build_graph().edges[0]
    learner = _learner()
    handle = learner.train_edge(None, edge, BUDGET)

    estimate = learner.estimate_reach(None, edge, handle, SETTINGS.rollouts, 0)

    assert estimate.success_probability == 0.9


def test_train_edge_rejects_a_budget_that_differs_from_the_settings(stub_task):
    edge = SCENARIOS["blocked_two_path"].build_graph().edges[0]

    with pytest.raises(ValueError, match="budget"):
        _learner().train_edge(None, edge, BUDGET + 1)


def test_estimate_reach_rejects_a_rollout_count_that_differs_from_the_settings(stub_task):
    edge = SCENARIOS["blocked_two_path"].build_graph().edges[0]
    learner = _learner()
    handle = learner.train_edge(None, edge, BUDGET)

    with pytest.raises(ValueError, match="rollouts"):
        learner.estimate_reach(None, edge, handle, SETTINGS.rollouts + 1, 0)


def _report(seed: int, operator: str | None, after: float) -> AutoSpecReport:
    return AutoSpecReport(
        scenario="trap_goal",
        seed=seed,
        expected_operator="seq",
        config={"training": {"budget": BUDGET}},
        steps=(RefinementStep("start", "goal", ("seq",), operator, 0.2, after),),
        initial_edges=(EdgeSuccess("start", "goal", 0.2),),
        final_edges=(EdgeSuccess("start", "goal", after),),
        best_path_before=0.2,
        best_path_after=after,
        reference_before=0.15,
        reference_after=0.85,
        before_curve=(Checkpoint(50, 0.1), Checkpoint(BUDGET, 0.2)),
        after_curve=(Checkpoint(50, 0.3), Checkpoint(BUDGET, after)),
        elapsed_seconds=12.0,
    )


def test_report_round_trips_through_its_dictionary_form():
    report = _report(0, "seq", 0.9)

    assert report_from_dict(report_to_dict(report)) == report


def test_summary_counts_seeds_that_committed_the_expected_operator():
    reports = [_report(0, "seq", 0.9), _report(1, None, 0.2)]

    table = summary_table(reports)

    assert "| trap_goal | seq | 1/2 | 0.200 | 0.550 | 0.15 | 0.85 | 12 |" in table


def test_summary_tolerates_missing_reference_and_unscored_expectations():
    """A 100-rooms report shows per-edge expectations and no match score."""
    report = replace(
        _report(0, "seq", 0.9),
        scenario="fig3",
        expected_operator=None,
        reference_before=None,
        reference_after=None,
        scenario_metadata={"expected_operators": {"M1->M3": ["seq"], "M2->M3": ["seq"]}},
    )

    assert report.matched_expected is False
    table = summary_table([report])

    assert "| fig3 | M1->M3: seq, M2->M3: seq | n/a | 0.200 | 0.900 | — | — | 12 |" in table


def test_report_round_trips_scenario_metadata_and_missing_references():
    """Metadata and absent references survive the JSON mapping unchanged."""
    report = replace(
        _report(0, "seq", 0.9),
        expected_operator=None,
        reference_before=None,
        reference_after=None,
        scenario_metadata={"expected_operators": {"M1->M3": ["seq"]}},
    )

    restored = report_from_dict(json.loads(json.dumps(report_to_dict(report))))

    assert restored == report
    assert restored.reference_before is None
