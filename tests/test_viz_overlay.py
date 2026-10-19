"""Tests for the run view, the trajectory drawing and the overlay figure.

A run view is built by hand with traces, so each test pins what the figure shows for one
kind of run: refined, refined with per-step panels, unrefined, and an old report.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib", reason="figure tests need the reports extra")

from faultline.envs import RoomsLayout, layout_to_dict  # noqa: E402
from faultline.experiments.autospec.records import AutoSpecReport  # noqa: E402
from faultline.geometry import convex_hull, predicate_to_dict  # noqa: E402
from faultline.graph.snapshot import GraphSnapshot, VertexSnapshot  # noqa: E402
from faultline.refinement.loop import RefinementStep  # noqa: E402
from faultline.refinement.trace import StepTrace, VertexChange  # noqa: E402
from faultline.refinement.trajectories import StepSamples, TrajectorySample  # noqa: E402
from faultline.scenarios import SCENARIOS  # noqa: E402
from faultline.spectrl.predicates import Box  # noqa: E402
from faultline.viz.annotations import draw_step  # noqa: E402
from faultline.viz.figures.overlay import build_overlay, write_overlay  # noqa: E402
from faultline.viz.run import RunView, run_view_from_report  # noqa: E402
from faultline.viz.style import Role, pyplot  # noqa: E402
from faultline.viz.trajectories import draw_trajectories  # noqa: E402

START, GOAL = Box((1.0, 1.0), (7.0, 7.0)), Box((17.0, 17.0), (23.0, 23.0))
GRAPH = GraphSnapshot(
    vertices=(
        VertexSnapshot("start", predicate_to_dict(START)),
        VertexSnapshot("goal", predicate_to_dict(GOAL)),
    ),
    edges=(),
    source="start",
    target="goal",
)
GOOD = TrajectorySample("start", "goal", True, ((4.0, 4.0), (12.0, 12.0), (20.0, 20.0)))
BAD = TrajectorySample("start", "goal", False, ((4.0, 4.0), (5.0, 9.0)))


def _step(operator="seq", traced=True) -> RefinementStep:
    trace = None
    if traced:
        shrunk = VertexChange("goal", predicate_to_dict(GOAL), predicate_to_dict(Box((18.0, 18.0), (22.0, 22.0))))
        trace = StepTrace((shrunk,), (), (), (), (), StepSamples(((19.0, 19.0),), (), ()))
    return RefinementStep("start", "goal", (operator,), operator if traced else None, 0.31, 0.87, trace)


def _run(steps, *, trajectories=True, final=True) -> RunView:
    paths = (GOOD, BAD) if trajectories else ()
    return RunView(
        scenario="trap_goal", seed=3, layout=RoomsLayout(), unsafe_region=None,
        initial_graph=GRAPH, final_graph=GRAPH if final else None, steps=tuple(steps),
        trajectories_before=paths, trajectories_after=paths,
    )


def _titles(figure) -> list[str]:
    return [ax.get_title(loc="left") for ax in figure.axes if ax.get_visible() and ax.axison]


def test_committed_steps_keep_their_scan_index_and_skip_the_rest():
    run = _run([_step(traced=False), _step("or")])

    assert [(index, step.operator) for index, step in run.committed] == [(1, "or")]


def test_refined_run_has_before_and_after_panels():
    figure, _caption = build_overlay(_run([_step()]))

    assert _titles(figure) == ["(a) Before refinement", "(b) After refinement"]
    assert figure._suptitle.get_text() == "trap goal: SeqRefine"


def test_per_step_panels_are_added_and_lettered_in_order():
    figure, _caption = build_overlay(_run([_step("seq"), _step("or")]), per_step=True)

    assert _titles(figure)[2:] == ["(c) Step 1: SeqRefine", "(d) Step 2: OrRefine"]


def test_unrefined_run_is_a_single_panel():
    figure, caption = build_overlay(_run([]))

    assert _titles(figure) == ["(a) Sampled rollouts"]
    assert "no refinement was applied" in caption


def test_legend_names_each_drawn_role_once():
    figure, _caption = build_overlay(_run([_step()]))

    (legend,) = figure.legends
    assert sorted(text.get_text() for text in legend.get_texts()) == sorted([
        "Start region", "Goal region", "Refined region", "Successful rollout", "Failed rollout",
    ])


def test_caption_describes_steps_and_the_rollout_lines():
    _figure, caption = build_overlay(_run([_step()]))

    assert "SeqRefine on start → goal" in caption
    assert "Thin lines are sampled rollouts" in caption


def test_caption_omits_the_rollout_sentence_without_trajectories():
    _figure, caption = build_overlay(_run([_step()], trajectories=False))

    assert "rollouts" not in caption


def test_embedded_caption_does_not_overlap_the_legend():
    figure, _caption = build_overlay(_run([_step()]), embed_caption=True)

    figure.canvas.draw()
    legend_box = figure.legends[0].get_window_extent()
    caption_box = figure.texts[-1].get_window_extent()
    assert caption_box.y1 <= legend_box.y0


def test_write_overlay_writes_images_and_a_caption_file(tmp_path):
    paths = write_overlay(_run([_step()]), tmp_path)

    assert [p.name for p in paths] == [
        "overlay_trap_goal_seed3.pdf",
        "overlay_trap_goal_seed3.png",
        "overlay_trap_goal_seed3.caption.txt",
    ]
    assert "SeqRefine" in paths[-1].read_text(encoding="utf-8")


def test_trajectories_are_coloured_by_outcome_with_failures_underneath():
    ax = pyplot().subplots()[1]

    roles = draw_trajectories(ax, [GOOD, BAD])

    assert roles == {Role.TRAJECTORY_OK, Role.TRAJECTORY_FAIL}
    assert ax.lines[0].get_color() != ax.lines[1].get_color()
    assert ax.lines[1].get_xdata().tolist() == [4.0, 12.0, 20.0]


def test_single_point_path_is_not_drawn():
    ax = pyplot().subplots()[1]

    roles = draw_trajectories(ax, [TrajectorySample("a", "b", True, ((1.0, 1.0),))])

    assert roles == set() and not ax.lines


def test_legend_never_names_a_region_with_no_area():
    line = convex_hull([(19.0, 19.0), (20.0, 20.0), (21.0, 21.0)])
    flat = VertexChange("goal", predicate_to_dict(GOAL), predicate_to_dict(line))
    trace = StepTrace((flat,), (), (), (), (), StepSamples((), (), ()))
    ax = pyplot().subplots()[1]

    assert draw_step(ax, trace, GRAPH, 24.0) == set()


def _report(**overrides) -> AutoSpecReport:
    fields = dict(
        scenario="trap_goal", seed=1, expected_operator="seq", config={"training": {"budget": 10}},
        steps=(_step(),), initial_edges=(), final_edges=(), best_path_before=0.3,
        best_path_after=0.9, reference_before=0.2, reference_after=0.9, before_curve=(),
        after_curve=(), elapsed_seconds=1.0,
    )
    return AutoSpecReport(**{**fields, **overrides})


def test_view_from_a_current_report_reads_the_recorded_room():
    layout = RoomsLayout({("v", 0, 0): (3.0, 5.0)})
    report = _report(schema_version=2, layout=layout_to_dict(layout), initial_graph=GRAPH,
                     final_graph=GRAPH, trajectories_before=(GOOD,))

    view = run_view_from_report(report)

    assert view.layout.doors == layout.doors
    assert view.initial_graph == GRAPH and view.trajectories_before == (GOOD,)


def test_view_from_an_old_report_borrows_the_room_from_the_catalogue():
    view = run_view_from_report(_report())

    assert view.layout is SCENARIOS["trap_goal"].layout
    assert view.final_graph is None
    assert {v.name for v in view.initial_graph.vertices} == {"start", "goal"}


def test_view_from_an_old_report_of_an_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="not in the catalogue"):
        run_view_from_report(_report(scenario="mystery"))


def test_old_report_draws_the_room_and_says_geometry_was_not_recorded():
    view = run_view_from_report(_report(steps=(_step(traced=False),)))

    figure, caption = build_overlay(view)

    assert _titles(figure) == ["(a) Sampled rollouts"]
    assert "was not recorded" in caption
