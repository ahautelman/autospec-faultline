"""Tests for the room drawing and the shared figure style.

Checks what a reader of the figure sees: which roles are drawn, where wall gaps are,
that no per-call legend entries appear, and that figures are written in every format.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib", reason="figure tests need the reports extra")

from faultline.envs import RoomsLayout  # noqa: E402
from faultline.graph.snapshot import snapshot_graph  # noqa: E402
from faultline.scenarios import SCENARIOS  # noqa: E402
from faultline.spectrl.predicates import And, Box, Not  # noqa: E402
from faultline.viz.geometry import draw_region  # noqa: E402
from faultline.viz.room import draw_room  # noqa: E402
from faultline.viz.style import ROLE_COLOURS, ROLE_LABELS, Role, pyplot, save_figure  # noqa: E402


def _room(scenario_name):
    scenario = SCENARIOS[scenario_name]
    figure, ax = pyplot().subplots()
    roles = draw_room(ax, scenario.layout, snapshot_graph(scenario.build_graph()), scenario.unsafe_region)
    return figure, ax, roles


def test_two_vertex_scenario_draws_start_and_goal():
    _figure, _ax, roles = _room("trap_goal")

    assert roles == {Role.START, Role.GOAL}


def test_intermediate_vertices_are_waypoints():
    _figure, _ax, roles = _room("blocked_two_path")

    assert roles == {Role.START, Role.GOAL, Role.WAYPOINT}


def test_hidden_trap_is_drawn_as_unsafe():
    _figure, _ax, roles = _room("narrow_door")

    assert Role.UNSAFE in roles


def test_axes_cover_the_world_with_ticks_on_room_boundaries():
    _figure, ax, _roles = _room("trap_goal")

    assert ax.get_xlim() == (0.0, 24.0) and ax.get_ylim() == (0.0, 24.0)
    assert list(ax.get_xticks()) == [0.0, 8.0, 16.0, 24.0]


def test_room_registers_no_legend_entries():
    _figure, ax, _roles = _room("blocked_two_path")

    assert ax.get_legend_handles_labels() == ([], [])


def test_door_leaves_a_gap_in_its_wall():
    layout = RoomsLayout({("v", 0, 0): (3.0, 5.0)})
    scenario = SCENARIOS["trap_goal"]
    figure, ax = pyplot().subplots()

    draw_room(ax, layout, snapshot_graph(scenario.build_graph()))

    segments = ax.collections[0].get_segments()
    first_room_wall = sorted(
        (min(a[1], b[1]), max(a[1], b[1]))
        for a, b in segments
        if a[0] == b[0] == 8.0 and max(a[1], b[1]) <= 8.0
    )
    assert first_room_wall == [(0.0, 3.0), (5.0, 8.0)]


def test_complement_region_is_rasterised():
    figure, ax = pyplot().subplots()
    region = And(Box((0.0, 0.0), (6.0, 6.0)), Not(Box((2.0, 2.0), (4.0, 4.0))))

    role = draw_region(ax, region, Role.UNSAFE, 24.0)

    assert role is Role.UNSAFE
    assert len(ax.collections) > 0
    assert not ax.patches


def test_every_role_has_a_colour_and_a_label():
    assert set(ROLE_COLOURS) == set(Role) == set(ROLE_LABELS)


def test_save_figure_writes_each_format_and_closes_the_figure(tmp_path):
    plt = pyplot()
    figure, _ax, _roles = _room("trap_goal")

    paths = save_figure(figure, tmp_path / "nested" / "room", ("pdf", "png"))

    assert [path.name for path in paths] == ["room.pdf", "room.png"]
    assert all(path.stat().st_size > 0 for path in paths)
    assert figure.number not in plt.get_fignums()
