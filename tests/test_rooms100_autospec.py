"""Tests for the n-rooms AutoSpec entry point and its slow end-to-end check.

The fast tests cover the argument selection, the data-file naming, and the
resolver-backed job list without training anything. The last test is marked
``slow`` because it runs the tabular planner through the whole refinement loop on
the Fig. 3 world, checking that the loop works at 100 rooms.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from faultline.experiments import rooms100_autospec
from faultline.learners import TabularLearner
from faultline.refinement.loop import AutoSpecConfig
from faultline.scenarios import resolve_scenario, run_scenario


def test_smoke_selection_overrides_worlds_seeds_and_budget():
    """``--smoke`` forces one world, one seed, and a small budget."""
    args = rooms100_autospec.build_parser().parse_args(["--smoke"])

    assert rooms100_autospec.selection(args) == (("fig3",), (0,), 1_000)


def test_default_selection_is_the_two_fixed_worlds_and_five_seeds():
    """Without flags the run covers Fig. 3 and Fig. 6 over the paper's five seeds."""
    args = rooms100_autospec.build_parser().parse_args([])

    worlds, seeds, budget = rooms100_autospec.selection(args)
    assert worlds == ("fig3", "fig6")
    assert seeds == (0, 1, 2, 3, 4)
    assert budget == 100_000


def test_data_path_names_one_world_and_seed():
    """A run's data file is named after its world and seed."""
    path = rooms100_autospec.data_path(Path("out"), "rooms100_random_3", 2)

    assert path == Path("out/data/autospec_rooms100_random_3_seed2.json")


def test_select_jobs_resolves_every_world_and_honours_resume(tmp_path):
    """Jobs resolve through the shared resolver and ``--resume`` skips finished ones."""
    jobs = rooms100_autospec.select_jobs(
        ("fig3", "rooms100_random_0"), (0,), tmp_path, resume=False
    )

    assert [scenario.name for scenario, _seed in jobs] == ["fig3", "rooms100_random_0"]
    assert all(scenario.world is not None for scenario, _seed in jobs)

    finished = rooms100_autospec.data_path(tmp_path, "fig3", 0)
    finished.parent.mkdir(parents=True)
    finished.write_text("{}")
    resumed = rooms100_autospec.select_jobs(
        ("fig3", "rooms100_random_0"), (0,), tmp_path, resume=True
    )

    assert [scenario.name for scenario, _seed in resumed] == ["rooms100_random_0"]


@pytest.mark.slow
def test_tabular_loop_runs_end_to_end_on_the_fig3_world():
    """The refinement loop completes on 100 rooms with the fast planner."""
    scenario = resolve_scenario("fig3")

    result = run_scenario(
        scenario,
        TabularLearner(),
        AutoSpecConfig(threshold=0.8, n_rollouts=10, seed=0),
    )

    assert len(result.edges) == 8
    # The half-sealed M3 is the fault the loop must find: SeqRefine on an edge
    # into M3 lifts the best path from a partial success to a full one.
    assert "seq" in result.operators
    assert result.best_path_after > result.best_path_before
    assert result.best_path_after == pytest.approx(1.0)
