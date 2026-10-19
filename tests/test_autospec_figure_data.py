"""A full-loop report records everything its figures are redrawn from.

The worker task is stubbed with canned rollouts, so the real orchestration in
``run_job`` runs without training a policy.
"""

from __future__ import annotations

import json

import pytest

from faultline.envs import layout_from_dict
from faultline.experiments.autospec import learner as learner_module
from faultline.experiments.autospec.executor import SerialExecutor
from faultline.experiments.autospec.records import SCHEMA_VERSION, report_from_dict, report_to_dict
from faultline.experiments.autospec.runner import RunContext, run_job
from faultline.experiments.records import Checkpoint
from faultline.experiments.training import ReportSettings, TrainingRecord
from faultline.learners import EdgeEstimate
from faultline.refinement import AutoSpecConfig
from faultline.scenarios import SCENARIOS
from tests.refinement_support import rollout

BUDGET = 100
SETTINGS = ReportSettings(budget=BUDGET, rollouts=10, interval=50, intermediate_rollouts=5)
SCENARIO = SCENARIOS["blocked_two_path"]


@pytest.fixture
def stub_task(monkeypatch):
    """Replace the worker task with one returning a successful and a failed rollout."""

    def task(request):
        rollouts = (
            rollout([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], reached=True, safe=True),
            rollout([(1.0, 1.0), (1.5, 1.5)], reached=False, safe=False),
        )
        checkpoints = (Checkpoint(50, 0.4), Checkpoint(BUDGET, 0.5))
        record = TrainingRecord(
            request.edge.source.name,
            request.edge.target.name,
            checkpoints,
            EdgeEstimate(0.5, rollouts),
            None,
        )
        return record

    monkeypatch.setattr(learner_module, "train_edge_task", task)


def _report():
    context = RunContext(
        template=AutoSpecConfig(threshold=0.99, budget=BUDGET, n_rollouts=10),
        settings=SETTINGS,
        executor=SerialExecutor(),
    )
    return run_job(SCENARIO, 0, context)


def test_report_records_the_room_so_it_is_redrawn_without_the_catalogue(stub_task):
    report = _report()

    assert report.schema_version == SCHEMA_VERSION
    assert layout_from_dict(report.layout).doors == SCENARIO.layout.doors


def test_report_snapshots_the_initial_and_final_graphs(stub_task):
    report = _report()

    expected = {vertex.name for vertex in SCENARIO.build_graph().vertices}
    assert {vertex.name for vertex in report.initial_graph.vertices} == expected
    assert report.final_graph is not None


def test_report_samples_trajectories_along_the_best_path(stub_task):
    report = _report()

    segments = {(t.source, t.target) for t in report.trajectories_before}
    assert segments
    assert {t.success for t in report.trajectories_before} == {True, False}
    assert report.initial_graph.source in {source for source, _ in segments}


def test_report_with_figure_data_survives_a_json_text_round_trip(stub_task):
    report = _report()

    restored = report_from_dict(json.loads(json.dumps(report_to_dict(report))))

    assert restored == report


def test_a_real_loop_run_is_drawn_end_to_end(stub_task, tmp_path):
    pytest.importorskip("matplotlib", reason="figure tests need the reports extra")
    from faultline.viz.figures.overlay import write_overlay
    from faultline.viz.run import run_view_from_report

    report = report_from_dict(json.loads(json.dumps(report_to_dict(_report()))))

    paths = write_overlay(run_view_from_report(report), tmp_path, per_step=True)

    assert all(path.stat().st_size > 0 for path in paths)
    caption = paths[-1].read_text(encoding="utf-8")
    assert caption.startswith(("Step 1:", "No edge fell below"))
