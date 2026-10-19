"""Tests for refinement traces and the figure data a report persists.

A trace is recovered by diffing the graph before and after a committed operator, so
each test drives the real loop with a scripted learner and checks the shape of the
edit it reports. The persistence tests cover the JSON round trip, the bounded
samples, and loading a report written before traces existed.
"""

from __future__ import annotations

import json
import pytest

from faultline.envs import RoomsLayout, layout_from_dict, layout_to_dict
from faultline.experiments.autospec.records import AutoSpecReport, report_from_dict, report_to_dict
from faultline.experiments.records import Checkpoint
from faultline.geometry import predicate_from_dict
from faultline.graph.snapshot import snapshot_graph
from faultline.refinement import AutoSpecConfig, run_autospec
from faultline.refinement.trace import build_trace
from faultline.refinement.trajectories import (
    MAX_POINTS,
    MAX_TRAJECTORY_POINTS,
    PER_CLASS,
    StepSamples,
    collect_step_samples,
    sample_trajectories,
)
from tests.loop_support import ScriptedLearner, blocked_rollouts, refinable_rollouts, seq_refined
from tests.refinement_support import (
    MID,
    START,
    chain_graph,
    estimate,
    rollout,
    two_parent_graph,
)

NO_SAMPLES = StepSamples((), (), ())
CONFIG = AutoSpecConfig(threshold=0.9)


def _run_seq():
    graph, _edge = chain_graph()

    def probability(edge):
        if seq_refined(edge, MID) or (edge.source.name, edge.target.name) == ("mid", "goal"):
            return 0.99
        return 0.5

    return run_autospec(graph, None, ScriptedLearner(probability, refinable_rollouts), CONFIG)


def _run_or():
    graph, _blocked, _sibling = two_parent_graph()

    def probability(edge):
        return 0.4 if (edge.source.name, edge.target.name) == ("start", "goal") else 0.9

    return run_autospec(graph, None, ScriptedLearner(probability, blocked_rollouts), AutoSpecConfig(threshold=0.8))


def test_seq_trace_reports_the_tightened_target_and_no_structural_edit():
    trace = _run_seq().steps[0].trace

    assert [change.name for change in trace.vertex_changes] == ["mid"]
    assert trace.added_vertices == () and trace.added_edges == () and trace.removed_edges == ()
    assert predicate_from_dict(trace.vertex_changes[0].before) == MID


def test_or_trace_reports_only_added_edges():
    trace = _run_or().steps[0].trace

    assert [(e.source, e.target) for e in trace.added_edges] == [("start", "alt")]
    assert trace.removed_edges == () and trace.vertex_changes == ()


def test_uncommitted_step_has_no_trace():
    graph, _edge = chain_graph()
    learner = ScriptedLearner(lambda edge: 0.5, blocked_rollouts)

    run = run_autospec(graph, None, learner, CONFIG)

    assert all(step.trace is None for step in run.steps)


def test_trace_names_a_new_vertex_and_the_edge_it_replaced():
    graph, edge = chain_graph()
    before = snapshot_graph(graph)
    refined = graph.copy()
    waypoint = refined.fresh_vertex("add", START)
    old = next(e for e in refined.edges if e.source.name == "start" and e.target.name == "mid")
    refined.remove_edge(old)
    refined.add_edge(refined.source, waypoint, old.safe)
    refined.add_edge(waypoint, old.target, old.safe)

    trace = build_trace(before, snapshot_graph(refined), NO_SAMPLES)

    assert [v.name for v in trace.added_vertices] == ["add1"]
    assert trace.removed_edges == (("start", "mid"),)
    assert [(e.source, e.target) for e in trace.added_edges] == [("start", "add1"), ("add1", "mid")]


def test_trace_of_an_unchanged_graph_is_empty():
    graph, _edge = chain_graph()

    trace = build_trace(snapshot_graph(graph), snapshot_graph(graph.copy()), NO_SAMPLES)

    assert (trace.vertex_changes, trace.safe_changes, trace.added_vertices) == ((), (), ())
    assert (trace.added_edges, trace.removed_edges) == ((), ())


def test_trace_keeps_parallel_edges_distinct():
    graph, _edge = chain_graph()
    doubled = graph.copy()
    doubled.add_edge(doubled.source, doubled.vertices[1], doubled.edges[0].safe)

    trace = build_trace(snapshot_graph(graph), snapshot_graph(doubled), NO_SAMPLES)

    assert [(e.source, e.target) for e in trace.added_edges] == [("start", "mid")]
    assert trace.removed_edges == ()


def test_trace_reports_a_tightened_safe_region():
    graph, _edge = chain_graph()
    tightened = graph.copy()
    tightened.edges[0].safe = MID

    trace = build_trace(snapshot_graph(graph), snapshot_graph(tightened), NO_SAMPLES)

    assert [(c.source, c.target) for c in trace.safe_changes] == [("start", "mid")]
    assert predicate_from_dict(trace.safe_changes[0].after) == MID


def _long_rollout(length, success=True):
    states = [(i * 0.01, 0.5) for i in range(length)]
    return rollout(states, reached=success, safe=success)


def test_trajectory_samples_are_bounded_per_class_and_in_length():
    graph, edge = chain_graph()
    many = [_long_rollout(500, i % 2 == 0) for i in range(40)]

    samples = sample_trajectories(edge, estimate(many))

    assert sum(s.success for s in samples) == PER_CLASS
    assert sum(not s.success for s in samples) == PER_CLASS
    assert all(len(s.xy) <= MAX_TRAJECTORY_POINTS for s in samples)


def test_trajectory_sample_keeps_first_and_last_state():
    _graph, edge = chain_graph()

    (sample,) = sample_trajectories(edge, estimate([_long_rollout(500)]))

    assert sample.xy[0] == (0.0, 0.5)
    assert sample.xy[-1] == pytest.approx((4.99, 0.5))


def test_short_rollouts_are_kept_whole():
    _graph, edge = chain_graph()

    (sample,) = sample_trajectories(edge, estimate([_long_rollout(5)]))

    assert len(sample.xy) == 5


def test_step_samples_hold_the_operators_witnesses_and_unsafe_tails():
    graph, edge = chain_graph()
    inside_goal_side = rollout([(0.5, 0.5), (1.5, 1.5)], reached=True, safe=True)
    ends_unsafe = rollout([(0.5, 0.5), (9.0, 9.0), (9.5, 9.5)], reached=False, safe=False)

    samples = collect_step_samples(edge, estimate([inside_goal_side, ends_unsafe]), tail_length=2)

    assert samples.witness_points == ((1.5, 1.5),)
    assert samples.unsafe_points == ((9.0, 9.0), (9.5, 9.5))


def test_point_samples_are_capped():
    _graph, edge = chain_graph()
    wandering = rollout([(1.5, 1.5)] * 1000, reached=True, safe=True)

    samples = collect_step_samples(edge, estimate([wandering]), tail_length=2)

    assert len(samples.witness_points) == MAX_POINTS


def _report_with_figure_data() -> AutoSpecReport:
    run = _run_seq()
    graph, _edge = chain_graph()
    return AutoSpecReport(
        scenario="chain",
        seed=0,
        expected_operator="seq",
        config={"training": {"budget": 100}},
        steps=run.steps,
        initial_edges=run.initial_edge_success,
        final_edges=run.edge_success,
        best_path_before=0.25,
        best_path_after=0.99,
        reference_before=0.2,
        reference_after=0.9,
        before_curve=(Checkpoint(50, 0.1),),
        after_curve=(Checkpoint(50, 0.9),),
        elapsed_seconds=1.0,
        schema_version=2,
        layout=layout_to_dict(RoomsLayout()),
        unsafe_region=None,
        initial_graph=snapshot_graph(graph),
        final_graph=snapshot_graph(run.graph),
        trajectories_before=sample_trajectories(graph.edges[0], estimate([_long_rollout(10)])),
    )


def test_report_with_traces_survives_a_json_text_round_trip():
    report = _report_with_figure_data()

    restored = report_from_dict(json.loads(json.dumps(report_to_dict(report))))

    assert restored == report


def test_report_without_figure_data_still_loads():
    payload = report_to_dict(_report_with_figure_data())
    for name in ("schema_version", "layout", "unsafe_region", "initial_graph", "final_graph",
                 "trajectories_before", "trajectories_after"):
        del payload[name]
    for step in payload["steps"]:
        del step["trace"]

    restored = report_from_dict(json.loads(json.dumps(payload)))

    assert restored.schema_version == 1
    assert restored.initial_graph is None and restored.steps[0].trace is None
    assert restored.trajectories_before == ()


def test_layout_round_trips_with_its_doors():
    layout = RoomsLayout({("v", 0, 1): (9.0, 11.0), ("h", 2, 0): (17.0, 19.0)}, room_size=8.0)

    restored = layout_from_dict(json.loads(json.dumps(layout_to_dict(layout))))

    assert restored.doors == layout.doors
    assert (restored.room_size, restored.rooms_per_side) == (8.0, 3)
