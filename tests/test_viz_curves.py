"""Tests for the learning-curve, per-seed and paper-comparison figures.

Runs are built by hand with simple curves, so each test pins a number a reader would take
off the figure (a mean, a bar height, a dot position) to a value worked out beforehand.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="figure tests need the reports extra")

from faultline.envs import RoomsLayout  # noqa: E402
from faultline.experiments.autospec.records import AutoSpecReport  # noqa: E402
from faultline.experiments.records import Checkpoint, Overlay, ScenarioReport  # noqa: E402
from faultline.graph.snapshot import GraphSnapshot  # noqa: E402
from faultline.refinement.loop import RefinementStep  # noqa: E402
from faultline.scenarios import SCENARIOS  # noqa: E402
from faultline.viz.captions import curves_caption, group_title, step_caption  # noqa: E402
from faultline.viz.figures.curves import build_curves, build_curves_grid, draw_curves, write_curves  # noqa: E402
from faultline.viz.figures.reference import build_reference, write_reference  # noqa: E402
from faultline.viz.figures.seeds import build_seeds, draw_seeds, write_seeds  # noqa: E402
from faultline.viz.legend import legend_handles  # noqa: E402
from faultline.viz.run import (  # noqa: E402
    RunView,
    group_by_scenario,
    run_view_from_report,
    run_view_from_scenario_report,
)
from faultline.viz.style import Role, pyplot  # noqa: E402

EMPTY_GRAPH = GraphSnapshot((), (), None, None)
STEPS = (10_000, 80_000, 200_000)


def _curve(*values):
    return tuple(Checkpoint(step, value) for step, value in zip(STEPS, values, strict=True))


def _run(scenario="narrow_door", seed=0, before=(0.1, 0.2, 0.3), after=(0.5, 0.7, 0.9), **fields) -> RunView:
    fields = {
        "before_curve": _curve(*before), "after_curve": _curve(*after), "budget": 200_000,
        "reference_before": 0.25, "reference_after": 0.75, "threshold": 0.99,
        "steps": (RefinementStep("start", "goal", ("seq",), "seq", 0.3, 0.9),), **fields,
    }
    return RunView(scenario, seed, RoomsLayout(), None, EMPTY_GRAPH, None, **fields)


def _axes_roles(draw, runs):
    ax = pyplot().subplots()[1]
    return ax, draw(ax, runs)


def test_runs_are_grouped_by_scenario_with_seeds_in_order():
    groups = group_by_scenario([_run("a", 2), _run("b", 0), _run("a", 1)])

    assert list(groups) == ["a", "b"]
    assert [run.seed for run in groups["a"]] == [1, 2]


def test_single_seed_curve_is_one_line_per_series_plus_the_threshold():
    ax, roles = _axes_roles(draw_curves, [_run()])

    assert len(ax.lines) == 3
    assert roles == {Role.BEFORE, Role.AFTER, Role.THRESHOLD, Role.REFERENCE}


def test_several_seeds_draw_faint_lines_a_band_and_the_mean():
    runs = [_run(seed=0, after=(0.4, 0.6, 0.8)), _run(seed=1, after=(0.6, 0.8, 1.0))]

    ax, _roles = _axes_roles(draw_curves, runs)

    means = [line for line in ax.lines if np.array_equal(np.round(line.get_ydata(), 6), [0.5, 0.7, 0.9])]
    assert len(means) == 1
    assert len(ax.lines) == 2 * (2 + 1) + 1
    assert len(ax.collections) >= 2


def test_reference_markers_sit_at_the_paper_values_at_80k():
    ax, _roles = _axes_roles(draw_curves, [_run()])

    points = sorted(tuple(offsets[0]) for offsets in (c.get_offsets() for c in ax.collections))
    assert points == [(80_000.0, 0.25), (80_000.0, 0.75)]


def test_reference_markers_are_left_out_when_the_run_stops_before_80k():
    short = _run(before=(0.1, 0.2, 0.3), after=(0.5, 0.7, 0.9))
    short = RunView(**{**short.__dict__, "before_curve": short.before_curve[:1], "after_curve": short.after_curve[:1]})

    _ax, roles = _axes_roles(draw_curves, [short])

    assert Role.REFERENCE not in roles


def test_run_without_threshold_or_reference_draws_only_the_curves():
    bare = _run(threshold=None, reference_before=None, reference_after=None)

    _ax, roles = _axes_roles(draw_curves, [bare])

    assert roles == {Role.BEFORE, Role.AFTER}


def test_grid_has_one_titled_panel_per_scenario_and_blanks_the_rest():
    names = ["trap_goal", "narrow_door", "long_corridor", "mixed_start", "blocked_two_path"]
    groups = group_by_scenario([_run(name) for name in names])

    figure, _caption = build_curves_grid(groups)

    visible = [ax for ax in figure.axes if ax.axison]
    assert len(visible) == 5 and len(figure.axes) == 6
    assert visible[0].get_title() == "trap goal: SeqRefine"
    assert len(figure.legends) == 1


def test_single_scenario_figure_is_single_column_with_the_caption_clauses_it_needs():
    figure, caption = build_curves([_run(seed=0), _run(seed=1)])

    assert figure.get_figwidth() == pytest.approx(3.4)
    assert "mean over 2 seeds" in caption and "threshold" in caption and "80k steps" in caption


def test_write_curves_writes_images_and_a_caption(tmp_path):
    paths = write_curves([_run()], tmp_path)

    assert [p.name for p in paths] == ["curves_narrow_door.pdf", "curves_narrow_door.png", "curves_narrow_door.caption.txt"]


def test_seed_figure_shows_both_checkpoints_when_the_budget_passes_80k():
    ax, roles = _axes_roles(draw_seeds, [_run(seed=0), _run(seed=1, after=(0.5, 0.9, 0.9))])

    assert [t.get_text() for t in ax.get_xticklabels()] == ["Before", "After", "Before", "After"]
    (checkpoints,) = ax.child_axes
    assert [t.get_text() for t in checkpoints.get_xticklabels()] == ["80k steps", "200k steps"]
    assert roles == {Role.BEFORE, Role.AFTER}


def test_seed_figure_shows_only_the_final_checkpoint_for_a_short_budget():
    short = _run(budget=50_000)

    ax, _roles = _axes_roles(draw_seeds, [short])

    assert len(ax.get_xticks()) == 2


def test_seed_dots_sit_at_each_seeds_value_and_the_mean_bar_at_their_average():
    runs = [_run(seed=0, after=(0.5, 0.6, 0.9)), _run(seed=1, after=(0.5, 0.8, 0.9))]

    ax, _roles = _axes_roles(draw_seeds, runs)

    after_80k = next(c for c in ax.collections if np.allclose(c.get_offsets()[:, 1], [0.6, 0.8]))
    assert after_80k.get_offsets()[:, 0].tolist() == [1.0, 1.0]
    segments = [s for c in ax.collections if hasattr(c, "get_segments") for s in c.get_segments()]
    mean_bar_heights = [s[0][1] for s in segments if s[0][1] == s[1][1]]
    assert any(height == pytest.approx(0.7) for height in mean_bar_heights)


def test_single_seed_has_dots_but_no_whiskers():
    ax, _roles = _axes_roles(draw_seeds, [_run()])

    vertical = [s for c in ax.collections if hasattr(c, "get_segments") for s in c.get_segments()
                if s[0][0] == s[1][0]]
    assert vertical == []


def test_seed_caption_counts_the_seeds():
    _figure, caption = build_seeds([_run(seed=i) for i in range(3)])

    assert "one of 3 seeds" in caption


def test_write_seeds_writes_images_and_a_caption(tmp_path):
    assert write_seeds([_run()], tmp_path)[-1].name == "seeds_narrow_door.caption.txt"


def _bar_heights(ax):
    return [patch.get_height() for patch in ax.patches]


def test_reference_bars_are_the_paper_numbers_then_the_runs_mean_at_80k():
    runs = [_run(seed=0, before=(0.1, 0.2, 0.3), after=(0.5, 0.6, 0.9)),
            _run(seed=1, before=(0.1, 0.4, 0.3), after=(0.5, 0.8, 0.9))]

    figure, _caption = build_reference({"narrow_door": runs})

    assert _bar_heights(figure.axes[0]) == pytest.approx([0.25, 0.75, 0.3, 0.7])


def test_scenario_without_paper_numbers_keeps_only_its_own_bars():
    with_numbers = _run("trap_goal")
    without = _run("mine", reference_before=None, reference_after=None)

    figure, _caption = build_reference({"trap_goal": [with_numbers], "mine": [without]})

    assert len(figure.axes[0].patches) == 4 + 2


def test_reference_figure_needs_at_least_one_paper_number():
    bare = _run(reference_before=None, reference_after=None)

    with pytest.raises(ValueError, match="paper's reference"):
        build_reference({"narrow_door": [bare]})


def test_reference_caption_says_single_seed_or_the_mean():
    _f, single = build_reference({"narrow_door": [_run()]})
    _f, several = build_reference({"narrow_door": [_run(seed=0), _run(seed=1)]})

    assert "single seed" in single and "mean over 2 seeds" in several


def test_write_reference_writes_images_and_a_caption(tmp_path):
    assert write_reference({"narrow_door": [_run()]}, tmp_path)[-1].name == "reference_comparison.caption.txt"


def test_group_title_names_the_operators_or_says_they_vary():
    seq = (RefinementStep("a", "b", ("seq",), "seq", 0.1, 0.9),)
    alt = (RefinementStep("a", "b", ("or",), "or", 0.1, 0.9),)

    assert group_title("narrow_door", [seq, seq]) == "narrow door: SeqRefine"
    assert group_title("narrow_door", [seq, alt]) == "narrow door: operators vary by seed"


def test_curves_caption_mentions_only_what_is_drawn():
    caption = curves_caption(1, has_threshold=False, has_reference=False)

    assert "threshold" not in caption and "seeds" not in caption and "Diamonds" not in caption


def test_step_without_a_trace_is_captioned_with_the_operator_only():
    step = RefinementStep("start", "goal", ("seq",), "seq", 0.31, 0.87)

    assert step_caption(0, step, 24.0) == "Step 1: SeqRefine on start → goal. Success 0.31 → 0.87."


def test_threshold_reference_and_bar_roles_have_matching_legend_swatches():
    handles, _labels = legend_handles([Role.THRESHOLD, Role.REFERENCE, Role.REFERENCE_BAR])

    assert handles[0].get_linestyle() == ":"
    assert handles[1].get_marker() == "D"
    assert handles[2].get_hatch()


def _full_report(**overrides) -> AutoSpecReport:
    fields = dict(
        scenario="trap_goal", seed=4, expected_operator="seq",
        config={"autospec": {"threshold": 0.95}, "training": {"budget": 200_000}},
        steps=(), initial_edges=(), final_edges=(), best_path_before=0.3, best_path_after=0.9,
        reference_before=0.15, reference_after=0.85, before_curve=_curve(0.1, 0.2, 0.3),
        after_curve=_curve(0.5, 0.7, 0.9), elapsed_seconds=1.0,
    )
    return AutoSpecReport(**{**fields, **overrides})


def test_view_from_a_report_carries_curves_references_threshold_and_budget():
    view = run_view_from_report(_full_report())

    assert view.before_curve == _curve(0.1, 0.2, 0.3)
    assert (view.reference_before, view.reference_after) == (0.15, 0.85)
    assert (view.threshold, view.budget) == (0.95, 200_000)


def _toy_report(operator="seq", refused=False) -> ScenarioReport:
    return ScenarioReport(
        scenario="trap_goal", operator=operator, seed=0, budget=200_000, before=_curve(0.1, 0.2, 0.3),
        after=_curve(0.5, 0.7, 0.9), overlay=Overlay("edge", (), (), None, None, None),
        reference_before=0.15, reference_after=0.85, operator_refused=refused,
    )


def test_view_from_a_toy_report_without_saved_points_names_the_operator_and_keeps_the_curves():
    view = run_view_from_scenario_report(_toy_report(), SCENARIOS["trap_goal"])

    assert [(s.operator, s.before, s.after) for s in view.steps] == [("seq", 0.3, 0.9)]
    assert view.committed == () and view.final_graph is None
    assert view.after_curve == _curve(0.5, 0.7, 0.9) and view.budget == 200_000


def test_view_from_a_toy_report_whose_operator_refused_applied_nothing():
    view = run_view_from_scenario_report(_toy_report(refused=True), SCENARIOS["trap_goal"])

    assert [s.operator for s in view.steps] == [None]
