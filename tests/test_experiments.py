"""Fast tests for the 9-rooms reporting pipeline.

These cover the pieces that need neither PPO nor a display: the checkpoint-series
helpers, JSON round-trips, the overlay inputs captured from the operator, and the route product. Figure rendering
is tested in the ``test_viz_*`` files. The heavy
PPO runs are the CLI's job and the slow suite's.
"""

from __future__ import annotations

import numpy as np
import pytest

from faultline.experiments import (
    Checkpoint,
    Overlay,
    ScenarioReport,
    ScenarioSpread,
    SeedSummary,
    TrainingRecord,
    route_product,
    source_edge,
    success_at,
)
from faultline.experiments.overlays import build_overlay
from faultline.experiments.records import (
    REFERENCE_STEP,
    report_from_dict,
    report_to_dict,
    spread_from_dict,
    spread_to_dict,
)
from faultline.experiments.training import (
    DEFAULT_INTERMEDIATE_ROLLOUTS,
    DEFAULT_ROLLOUTS,
    ReportSettings,
    make_eval_env,
    rollouts_for,
)
from faultline.graph import AbstractGraph
from faultline.learners import EdgeEstimate
from faultline.scenarios import blocked_two_path, narrow_door
from tests.refinement_support import MID, START, chain_graph, estimate, rollout

BUDGET = 200_000


def test_make_eval_env_mirrors_the_training_layout_and_trap():
    """The evaluation env keeps the training env's walls and trap."""
    training = narrow_door().make_env()
    evaluation = make_eval_env(training)
    assert evaluation.layout.doors == training.layout.doors
    assert evaluation.unsafe_region == training.unsafe_region


def _report(name: str = "trap_goal") -> ScenarioReport:
    """Build a small report for the persistence and figure tests."""
    return ScenarioReport(
        scenario=name,
        operator="seq",
        seed=0,
        budget=BUDGET,
        before=(Checkpoint(10_000, 0.2), Checkpoint(BUDGET, 0.4)),
        after=(Checkpoint(10_000, 0.3), Checkpoint(BUDGET, 0.9)),
        overlay=Overlay(
            "hull",
            ((1.1, 1.1), (1.9, 1.1), (1.1, 1.9)),
            (),
            None,
            None,
            None,
            ((0.2, 0.2), (0.3, 0.3)),
        ),
        reference_before=0.15,
        reference_after=0.85,
        operator_refused=True,
    )


def test_success_at_reads_the_exact_step():
    series = (Checkpoint(10, 0.1), Checkpoint(20, 0.5), Checkpoint(30, 0.9))
    assert success_at(series, 20) == 0.5


def test_success_at_falls_back_to_the_nearest_step():
    series = (Checkpoint(10, 0.1), Checkpoint(20, 0.5), Checkpoint(30, 0.9))
    assert success_at(series, 25) == 0.5
    assert success_at(series, 5) == 0.1
    assert success_at(series, 99) == 0.9


def test_report_json_round_trip():
    report = _report()
    assert report_from_dict(report_to_dict(report)) == report


def test_report_from_dict_defaults_the_refusal_flag_for_old_records():
    payload = report_to_dict(_report())
    payload.pop("operator_refused")

    assert report_from_dict(payload).operator_refused is False


def test_spread_json_round_trip():
    spread = ScenarioSpread(
        "trap_goal",
        BUDGET,
        (SeedSummary(0, 0.1, 0.7, 0.2, 0.9), SeedSummary(1, 0.15, 0.75, 0.25, 0.95)),
    )
    assert spread_from_dict(spread_to_dict(spread)) == spread


def test_route_product_multiplies_the_route_edges():
    record = EdgeEstimate(0.5, (rollout([[1.2, 1.2], [1.5, 1.5]], True, True),))
    first = TrainingRecord("a", "b", (Checkpoint(10, 0.5), Checkpoint(20, 0.5)), record, None)
    second = TrainingRecord("b", "c", (Checkpoint(10, 0.8), Checkpoint(20, 0.4)), record, None)

    product = route_product((first, second))

    assert [checkpoint.probability for checkpoint in product] == pytest.approx([0.4, 0.2])


def test_route_product_of_a_single_edge_is_unchanged():
    record = EdgeEstimate(0.5, (rollout([[1.2, 1.2]], True, True),))
    single = TrainingRecord("a", "b", (Checkpoint(10, 0.5), Checkpoint(20, 0.6)), record, None)

    assert route_product((single,)) == single.checkpoints


def test_source_edge_returns_the_bottleneck_edge():
    graph, _edge = chain_graph()
    edge = source_edge(graph)
    assert (edge.source.name, edge.target.name) == ("start", "mid")


def test_source_edge_rejects_a_fan_out():
    graph = AbstractGraph()
    start = graph.add_vertex("start", START)
    first = graph.add_vertex("first", MID)
    second = graph.add_vertex("second", MID)
    graph.add_edge(start, first)
    graph.add_edge(start, second)
    graph.source = start
    graph.target = first

    with pytest.raises(ValueError):
        source_edge(graph)


def test_reach_overlay_collects_the_target_visits():
    graph, edge = chain_graph()
    rollouts = (rollout([[1.2, 1.2], [1.5, 1.5]], True, True),)

    overlay = build_overlay("seq", graph, edge, estimate(rollouts))

    assert overlay.kind == "hull"
    assert overlay.points == ((1.2, 1.2), (1.5, 1.5))
    assert overlay.avoid_points == ()


def test_seq_overlay_collects_the_unsafe_tail():
    graph, edge = chain_graph()
    rollouts = (
        rollout([[1.2, 1.2], [1.5, 1.5]], True, True),
        rollout([[0.5, 0.5], [2.6, 2.6]], False, False),
    )

    overlay = build_overlay("seq", graph, edge, estimate(rollouts))

    assert overlay.points == ((1.2, 1.2), (1.5, 1.5))
    assert overlay.avoid_points == ((0.5, 0.5), (2.6, 2.6))


def test_seq_overlay_ignores_an_episode_that_left_but_ended_safe():
    graph, edge = chain_graph()
    rollouts = (
        rollout([[1.2, 1.2], [1.5, 1.5]], True, True),
        rollout(
            [[0.5, 0.5], [2.6, 2.6], [1.0, 1.0]],
            False,
            False,
            final_safe=True,
        ),
    )

    overlay = build_overlay("seq", graph, edge, estimate(rollouts))

    assert overlay.avoid_points == ()


def test_past_overlay_splits_the_starts_and_fits_a_boundary():
    graph, edge = chain_graph()
    rollouts = (
        rollout([[0.1, 0.1], [1.5, 1.5]], True, True),
        rollout([[0.9, 0.9], [0.9, 0.9]], False, True),
    )

    overlay = build_overlay("past", graph, edge, estimate(rollouts))

    assert overlay.kind == "half_plane"
    assert overlay.normal is not None
    assert len(overlay.points) == 1
    assert len(overlay.counter_points) == 1


def test_or_overlay_records_the_added_edge():
    scenario = blocked_two_path()
    graph = scenario.build_graph()
    edge = scenario.report_edge(graph)
    rollouts = (rollout([[0.5, 0.5]], False, True),)

    overlay = build_overlay("or", graph, edge, estimate(rollouts))

    assert overlay.kind == "edge"
    assert overlay.added_edge == ("mid1", "mid2")
    assert (edge.source.name, edge.target.name) == ("mid1", "goal")
