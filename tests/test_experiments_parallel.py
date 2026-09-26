"""Fast tests for the parallel orchestration of the 9-rooms pipeline.

These use a stubbed job function and a worker count of one, so they exercise the
dispatch, ordering, and reuse decisions without training a policy or spawning a
process. The thread-pinning check runs the real worker initializer.
"""

from __future__ import annotations

import os

import pytest

from faultline.experiments import Checkpoint, Overlay, ScenarioReport
from faultline.experiments import pipeline
from faultline.experiments.training import ReportSettings
from faultline.scenarios import SCENARIOS

BUDGET = 200_000


def _report(name: str) -> ScenarioReport:
    """Build a small report for the orchestration tests."""
    return ScenarioReport(
        scenario=name,
        operator="seq",
        seed=0,
        budget=BUDGET,
        before=(Checkpoint(10_000, 0.2), Checkpoint(BUDGET, 0.4)),
        after=(Checkpoint(10_000, 0.3), Checkpoint(BUDGET, 0.9)),
        overlay=Overlay("edge", (), (), None, None, None),
        reference_before=0.15,
        reference_after=0.85,
    )


def test_default_workers_prefers_the_environment_override(monkeypatch):
    monkeypatch.setenv(pipeline.WORKERS_ENV, "3")

    assert pipeline.default_workers() == 3


def test_default_workers_leaves_one_core_free(monkeypatch):
    monkeypatch.delenv(pipeline.WORKERS_ENV, raising=False)
    monkeypatch.setattr(pipeline.os, "cpu_count", lambda: 8)

    assert pipeline.default_workers() == 7


def test_default_workers_rejects_a_non_positive_override(monkeypatch):
    monkeypatch.setenv(pipeline.WORKERS_ENV, "0")

    with pytest.raises(ValueError):
        pipeline.default_workers()


def test_run_single_seed_runs_one_job_per_scenario(monkeypatch):
    calls = []

    def fake_run(scenario, seed, settings):
        calls.append((scenario.name, seed))
        return _report(scenario.name)

    monkeypatch.setattr(pipeline, "run_reported_scenario", fake_run)
    scenarios = tuple(SCENARIOS.values())

    reports = pipeline.run_single_seed(scenarios, 3, ReportSettings(), workers=1)

    assert list(reports) == [scenario.name for scenario in scenarios]
    assert calls == [(scenario.name, 3) for scenario in scenarios]


def test_run_multi_seed_skips_known_seeds(monkeypatch):
    calls = []

    def fake_run(scenario, seed, settings):
        calls.append((scenario.name, seed))
        return _report(scenario.name)

    monkeypatch.setattr(pipeline, "run_reported_scenario", fake_run)
    scenario = SCENARIOS["trap_goal"]
    known = {"trap_goal": {0: _report("trap_goal")}}

    spreads = pipeline.run_multi_seed([scenario], [0, 1], ReportSettings(), known, workers=1)

    assert calls == [("trap_goal", 1)]
    assert [summary.seed for summary in spreads["trap_goal"].summaries] == [0, 1]


def test_run_single_seed_rejects_a_zero_worker_count():
    with pytest.raises(ValueError):
        pipeline.run_single_seed((), 0, ReportSettings(), workers=0)


def test_configure_worker_pins_one_thread_and_disables_bars(monkeypatch):
    from faultline.experiments import progress

    monkeypatch.setattr(pipeline.os, "environ", dict(os.environ))
    torch = pytest.importorskip("torch")
    previous_threads = torch.get_num_threads()
    try:
        pipeline.configure_worker()

        assert pipeline.os.environ["OMP_NUM_THREADS"] == "1"
        assert pipeline.os.environ["MKL_NUM_THREADS"] == "1"
        assert torch.get_num_threads() == 1
        items = [1, 2, 3]
        assert progress.track(items) is items
    finally:
        torch.set_num_threads(previous_threads)
        progress.enable()
