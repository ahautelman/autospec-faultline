"""Tests for step annotations and the shared legend.

Traces are built by hand so each test pins one shape of edit to the roles drawn for it,
and to what the legend then shows.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib", reason="figure tests need the reports extra")

from faultline.geometry import convex_hull, predicate_to_dict  # noqa: E402
from faultline.graph.snapshot import EdgeSnapshot, GraphSnapshot, VertexSnapshot  # noqa: E402
from faultline.refinement.trace import SafeChange, StepTrace, VertexChange  # noqa: E402
from faultline.refinement.trajectories import StepSamples  # noqa: E402
from faultline.spectrl.predicates import And, Box, HalfPlane, Not, Or, Universe  # noqa: E402
from faultline.viz.annotations import draw_step, half_planes_in, removed_safe_region  # noqa: E402
from faultline.viz.polygons import region_polygons  # noqa: E402
from faultline.viz.legend import add_legend, legend_handles  # noqa: E402
from faultline.viz.style import ROLE_LABELS, Role, pyplot  # noqa: E402

WORLD = 24.0
SQUARE = Box((1.0, 1.0), (7.0, 7.0))
GRAPH = GraphSnapshot(
    vertices=(
        VertexSnapshot("start", predicate_to_dict(SQUARE)),
        VertexSnapshot("goal", predicate_to_dict(Box((17.0, 17.0), (23.0, 23.0)))),
    ),
    edges=(),
    source="start",
    target="goal",
)


def _trace(samples=StepSamples((), (), ()), **edits) -> StepTrace:
    fields = dict(vertex_changes=(), safe_changes=(), added_vertices=(), added_edges=(),
                  removed_edges=())
    return StepTrace(**{**fields, **edits}, samples=samples)


def _draw(trace, **kwargs):
    figure, ax = pyplot().subplots()
    return ax, draw_step(ax, trace, GRAPH, WORLD, **kwargs)


def test_shrunk_region_is_drawn_as_refined():
    change = VertexChange("goal", predicate_to_dict(SQUARE), predicate_to_dict(Box((2.0, 2.0), (4.0, 4.0))))

    _ax, roles = _draw(_trace(vertex_changes=(change,)))

    assert roles == {Role.REFINED}


def test_area_removed_from_a_safe_region_is_drawn_as_unsafe():
    hull = convex_hull([(2.0, 2.0), (4.0, 2.0), (3.0, 4.0)])
    change = SafeChange("start", "goal", predicate_to_dict(Universe()),
                        predicate_to_dict(And(Universe(), Not(hull))))

    _ax, roles = _draw(_trace(safe_changes=(change,)))

    assert roles == {Role.UNSAFE}


def test_new_region_cut_by_a_half_plane_also_draws_the_boundary():
    region = And(SQUARE, HalfPlane((1.0, 0.0), 4.0))
    vertex = VertexSnapshot("past1", predicate_to_dict(region))

    _ax, roles = _draw(_trace(added_vertices=(vertex,)))

    assert roles == {Role.REFINED, Role.BOUNDARY}


def test_added_edge_between_known_vertices_is_drawn_as_an_arrow():
    edge = EdgeSnapshot("start", "goal", predicate_to_dict(Universe()))

    ax, roles = _draw(_trace(added_edges=(edge,)))

    assert roles == {Role.EDGE}
    assert len(ax.texts) == 1


def test_edge_to_an_unknown_vertex_is_skipped():
    edge = EdgeSnapshot("start", "ghost", predicate_to_dict(Universe()))

    _ax, roles = _draw(_trace(added_edges=(edge,)))

    assert roles == set()


def test_samples_are_scattered_unless_switched_off():
    samples = StepSamples(witness_points=((1.0, 1.0),), unsafe_points=((2.0, 2.0),), trajectories=())

    _ax, shown = _draw(_trace(samples))
    _ax, hidden = _draw(_trace(samples), show_samples=False)

    assert shown == {Role.SAMPLE, Role.UNSAFE}
    assert hidden == set()


def test_unchanged_step_draws_nothing():
    ax, roles = _draw(_trace())

    assert roles == set()
    assert not ax.patches and not ax.collections


def test_several_steps_accumulate_on_one_axes():
    ax = pyplot().subplots()[1]
    reach = VertexChange("goal", predicate_to_dict(SQUARE), predicate_to_dict(Box((2.0, 2.0), (4.0, 4.0))))
    edge = EdgeSnapshot("start", "goal", predicate_to_dict(Universe()))

    roles = draw_step(ax, _trace(vertex_changes=(reach,)), GRAPH, WORLD)
    roles |= draw_step(ax, _trace(added_edges=(edge,)), GRAPH, WORLD)

    assert roles == {Role.REFINED, Role.EDGE}
    assert len(ax.patches) == 1


def test_drawing_a_step_registers_no_legend_entries():
    change = VertexChange("goal", predicate_to_dict(SQUARE), predicate_to_dict(Box((2.0, 2.0), (4.0, 4.0))))

    ax, _roles = _draw(_trace(vertex_changes=(change,)))

    assert ax.get_legend_handles_labels() == ([], [])


def test_removed_area_of_an_avoid_step_is_the_safe_region_inside_the_hull():
    safe = Box((0.0, 0.0), (10.0, 10.0))
    hull = convex_hull([(2.0, 2.0), (6.0, 2.0), (4.0, 6.0)])
    change = SafeChange("a", "b", predicate_to_dict(safe), predicate_to_dict(And(safe, Not(hull))))

    removed = removed_safe_region(change)

    assert removed.evaluate([4.0, 3.0]) is True
    assert removed.evaluate([8.0, 8.0]) is False
    assert region_polygons(removed, WORLD) is not None


def test_removed_area_of_an_unfamiliar_change_is_the_general_difference():
    before = Box((0.0, 0.0), (10.0, 10.0))
    after = Box((0.0, 0.0), (5.0, 10.0))
    change = SafeChange("a", "b", predicate_to_dict(before), predicate_to_dict(after))

    removed = removed_safe_region(change)

    assert removed.evaluate([7.0, 5.0]) is True
    assert removed.evaluate([3.0, 5.0]) is False


def test_half_planes_are_found_in_intersections_but_not_unions():
    plane = HalfPlane((1.0, 0.0), 4.0)

    assert half_planes_in(And(SQUARE, plane)) == [plane]
    assert half_planes_in(Or(SQUARE, plane)) == []


def test_legend_lists_each_role_once_in_a_fixed_order():
    handles, labels = legend_handles([Role.EDGE, Role.START, Role.EDGE, Role.UNSAFE, Role.GOAL])

    assert labels == [ROLE_LABELS[Role.START], ROLE_LABELS[Role.GOAL],
                      ROLE_LABELS[Role.UNSAFE], ROLE_LABELS[Role.EDGE]]
    assert len(handles) == 4


def test_unsafe_swatch_is_hatched_like_the_figure():
    (handle,), _labels = legend_handles([Role.UNSAFE])

    assert handle.get_hatch()


def test_legend_is_added_to_the_figure_with_one_entry_per_role():
    figure = pyplot().figure(layout="constrained")

    legend = add_legend(figure, set(Role))

    assert len(legend.get_texts()) == len(Role)


def test_no_legend_is_added_when_nothing_was_drawn():
    assert add_legend(pyplot().figure(), set()) is None
