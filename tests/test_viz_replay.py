"""Tests for replaying a toy report's operator so its overlay can be drawn.

The oracle is the operator itself: running it on the original rollouts must give the same
refined graph as replaying it from the points the report saved.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib", reason="figure tests need the reports extra")

from faultline.experiments.overlays import build_overlay  # noqa: E402
from faultline.experiments.records import Checkpoint, Overlay, ScenarioReport  # noqa: E402
from faultline.graph.snapshot import snapshot_graph  # noqa: E402
from faultline.refinement.operators import AVOID_TAIL_LENGTH, add_refine, or_refine, past_refine, seq_refine  # noqa: E402
from faultline.scenarios import SCENARIOS  # noqa: E402
from faultline.viz.figures.overlay import build_overlay as build_overlay_figure  # noqa: E402
from faultline.viz.replay import replay_refinement  # noqa: E402
from faultline.viz.run import run_view_from_scenario_report  # noqa: E402
from tests.refinement_support import estimate, rollout  # noqa: E402

CURVE = (Checkpoint(80_000, 0.2), Checkpoint(200_000, 0.4))
AFTER = (Checkpoint(80_000, 0.7), Checkpoint(200_000, 0.9))

ROLLOUTS = {
    "seq": [
        rollout([(4, 4), (2, 18), (3, 19)], True, True),
        rollout([(4, 4), (5, 20), (6, 22)], True, True),
        rollout([(4, 4), (6, 18), (2, 21)], True, True),
        rollout([(4, 4), (3, 15), (3, 16)], False, False, final_safe=False),
        rollout([(4, 4), (2, 15), (3.5, 16.5)], False, False, final_safe=False),
    ],
    "add": [
        rollout([(4, 4), (8, 12), (12, 12)], True, True),
        rollout([(4, 4), (9, 15), (12, 12)], True, True),
        rollout([(4, 4), (7, 14), (12, 12)], True, True),
    ],
    "past": [
        rollout([(3, 10), (20, 20)], True, True),
        rollout([(5, 11), (20, 20)], True, True),
        rollout([(2, 12), (20, 20)], True, True),
        rollout([(3, 20), (3, 21)], False, True),
        rollout([(5, 21), (3, 21)], False, True),
        rollout([(2, 22), (3, 21)], False, True),
    ],
    "or": [rollout([(4, 4), (5, 5)], False, True)],
}
SCENARIO_OF = {"seq": "narrow_door", "add": "long_corridor", "past": "mixed_start", "or": "blocked_two_path"}
OPERATORS = {
    "seq": lambda g, e, est: seq_refine(g, e, est, AVOID_TAIL_LENGTH),
    "add": lambda g, e, est: add_refine(g, e, est),
    "past": lambda g, e, est: past_refine(g, e, est),
    "or": lambda g, e, est: or_refine(g, e),
}


def _setup(operator):
    scenario = SCENARIOS[SCENARIO_OF[operator]]
    graph = scenario.build_graph()
    edge = scenario.report_edge(graph)
    return scenario, graph, edge, estimate(ROLLOUTS[operator])


def _report(operator, overlay, **fields) -> ScenarioReport:
    return ScenarioReport(SCENARIO_OF[operator], operator, 0, 200_000, CURVE, AFTER, overlay, 0.2, 0.8, **fields)


@pytest.mark.parametrize("operator", ["seq", "add", "past", "or"])
def test_replay_rebuilds_the_graph_the_operator_produced(operator):
    scenario, graph, edge, est = _setup(operator)
    report = _report(operator, build_overlay(operator, graph, edge, est))

    step, final = replay_refinement(report, scenario)

    assert final == snapshot_graph(OPERATORS[operator](graph, edge, est))
    assert step.operator == operator and step.trace is not None


def test_replayed_step_carries_the_saved_points_and_success_numbers():
    scenario, graph, edge, est = _setup("seq")
    overlay = build_overlay("seq", graph, edge, est)

    step, _final = replay_refinement(_report("seq", overlay), scenario)

    assert step.trace.samples.witness_points == overlay.points
    assert step.trace.samples.unsafe_points == overlay.avoid_points
    assert (step.before, step.after) == (0.4, 0.9)
    assert len(step.trace.vertex_changes) == 1 and len(step.trace.safe_changes) == 1


def test_split_replay_reproduces_the_fitted_boundary():
    scenario, graph, edge, est = _setup("past")
    overlay = build_overlay("past", graph, edge, est)

    step, _final = replay_refinement(_report("past", overlay), scenario)

    (change,) = step.trace.vertex_changes
    assert [part["type"] for part in change.after["predicates"]] == ["box", "half_plane"]
    assert change.after["predicates"][1]["normal"] == pytest.approx(list(overlay.normal))


def test_refused_or_unknown_operator_is_not_replayed():
    scenario, graph, edge, est = _setup("seq")
    overlay = build_overlay("seq", graph, edge, est)

    assert replay_refinement(_report("seq", overlay, operator_refused=True), scenario) is None
    assert replay_refinement(ScenarioReport("narrow_door", "", 0, 10, CURVE, AFTER, overlay, 0.2, 0.8), scenario) is None


@pytest.mark.parametrize("operator", ["seq", "add", "past"])
def test_empty_overlay_is_not_replayed_and_does_not_raise(operator):
    scenario = SCENARIOS[SCENARIO_OF[operator]]
    empty = Overlay("hull", (), (), None, None, None)

    assert replay_refinement(_report(operator, empty), scenario) is None


def test_view_from_a_toy_report_with_saved_points_can_draw_its_overlay():
    scenario, graph, edge, est = _setup("add")
    report = _report("add", build_overlay("add", graph, edge, est))

    view = run_view_from_scenario_report(report, scenario)
    figure, caption = build_overlay_figure(view)

    assert len(view.committed) == 1 and view.final_graph is not None
    assert [ax.get_title(loc="left") for ax in figure.axes[:2]] == ["(a) Before refinement", "(b) After refinement"]
    assert "AddRefine" in caption and "new region" in caption


def test_dense_saved_points_are_thinned_in_the_trace():
    scenario, graph, edge, est = _setup("seq")
    dense = tuple((1.2 + (i % 50) * 0.1, 17.2 + (i // 50) * 0.2) for i in range(1000))
    overlay = Overlay("hull", dense, (), None, None, None)

    step, _final = replay_refinement(_report("seq", overlay), scenario)

    assert len(step.trace.samples.witness_points) == 200
