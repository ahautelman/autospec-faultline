"""Fast tests for the 9-rooms reporting pipeline.

These cover the pieces that need neither PPO nor a display: the checkpoint-series
helpers, JSON round-trips, the overlay geometry derived from the operator's own
inputs, the route product, and figure rendering from persisted records. The heavy
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
    write_figures,
)
from faultline.experiments.overlays import build_overlay, refined_region
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
    rollouts_for,
)
from faultline.graph import AbstractGraph
from faultline.learners import EdgeEstimate
from faultline.scenarios import SCENARIOS
from tests.refinement_support import MID, START, chain_graph, estimate, rollout

BUDGET = 200_000


def _report(name: str = "trap_goal") -> ScenarioReport:
    """Build a small report for the persistence and figure tests."""
    return ScenarioReport(
        scenario=name,
        operator="seq",
        seed=0,
        budget=BUDGET,
        before=(Checkpoint(10_000, 0.2), Checkpoint(BUDGET, 0.4)),
        after=(Checkpoint(10_000, 0.3), Checkpoint(BUDGET, 0.9)),
        overlay=Overlay("hull", ((1.1, 1.1), (1.9, 1.1), (1.1, 1.9)), (), None, None, None),
        reference_before=0.15,
        reference_after=0.85,
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


def test_spread_json_round_trip():
    spread = ScenarioSpread(
        "trap_goal",
        BUDGET,
        (SeedSummary(0, 0.1, 0.7, 0.2, 0.9), SeedSummary(1, 0.15, 0.75, 0.25, 0.95)),
    )
    assert spread_from_dict(spread_to_dict(spread)) == spread


def test_route_product_multiplies_the_route_edges():
    record = EdgeEstimate(0.5, (rollout([[1.2, 1.2], [1.5, 1.5]], True, True),))
    first = TrainingRecord("a", "b", (Checkpoint(10, 0.5), Checkpoint(20, 0.5)), record)
    second = TrainingRecord("b", "c", (Checkpoint(10, 0.8), Checkpoint(20, 0.4)), record)

    product = route_product((first, second))

    assert [checkpoint.probability for checkpoint in product] == pytest.approx([0.4, 0.2])


def test_route_product_of_a_single_edge_is_unchanged():
    record = EdgeEstimate(0.5, (rollout([[1.2, 1.2]], True, True),))
    single = TrainingRecord("a", "b", (Checkpoint(10, 0.5), Checkpoint(20, 0.6)), record)

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
    scenario = SCENARIOS["blocked_two_path"]
    graph = scenario.build_graph()
    edge = source_edge(graph)
    rollouts = (rollout([[0.5, 0.5]], False, True),)

    overlay = build_overlay("or", graph, edge, estimate(rollouts))

    assert overlay.kind == "edge"
    assert overlay.added_edge == ("start", "mid")


def test_refined_region_shrinks_the_target_to_the_visited_hull():
    _graph, edge = chain_graph()
    overlay = Overlay("hull", ((1.1, 1.1), (1.9, 1.1), (1.1, 1.9)), (), None, None, None)

    region = refined_region("seq", edge, overlay)

    assert region is not None
    assert region.evaluate(np.array([1.2, 1.2])) is True
    assert region.evaluate(np.array([1.8, 1.8])) is False


def test_write_figures_renders_from_records(tmp_path):
    pytest.importorskip("matplotlib")
    spread = ScenarioSpread("trap_goal", BUDGET, (SeedSummary(0, 0.1, 0.7, 0.2, 0.9),))

    write_figures({"trap_goal": _report()}, {"trap_goal": spread}, SCENARIOS, tmp_path)

    assert (tmp_path / "reference_comparison.png").exists()
    assert (tmp_path / "reference_comparison_trap_goal.png").exists()
    assert (tmp_path / "learning_curves.png").exists()
    assert (tmp_path / "overlays.png").exists()
    assert (tmp_path / "overlay_trap_goal.png").exists()
    assert (tmp_path / "seed_spread.png").exists()
    assert (tmp_path / "seed_spread_trap_goal.png").exists()


def test_reference_step_is_the_paper_budget():
    assert REFERENCE_STEP == 80_000


def test_rollouts_for_keeps_full_count_at_the_reference_and_budget():
    settings = ReportSettings(budget=BUDGET, rollouts=1000, interval=10_000, intermediate_rollouts=25)

    assert rollouts_for(REFERENCE_STEP, settings) == 1000
    assert rollouts_for(BUDGET, settings) == 1000
    assert rollouts_for(10_000, settings) == 25
    assert rollouts_for(70_000, settings) == 25


def test_rollouts_for_uses_the_last_checkpoint_below_the_reference():
    settings = ReportSettings(budget=BUDGET, rollouts=1000, interval=30_000, intermediate_rollouts=25)

    assert rollouts_for(60_000, settings) == 1000
    assert rollouts_for(90_000, settings) == 25


def test_report_settings_default_intermediate_budget_is_small():
    settings = ReportSettings()

    assert settings.rollouts == DEFAULT_ROLLOUTS
    assert settings.intermediate_rollouts == DEFAULT_INTERMEDIATE_ROLLOUTS
    assert settings.intermediate_rollouts < settings.rollouts
