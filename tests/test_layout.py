"""Behavioural tests for the 9-rooms wall, door, and region model.

These tests pin the wall-crossing rules with hand-picked coordinates: a move
through a door passes, a move through a solid wall blocks, and a move along a
wall is never blocked by that wall.
"""

from __future__ import annotations

import numpy as np
import pytest

from faultline.envs import (
    RoomsLayout,
    default_doors,
    default_regions,
    open_doors_except_sealed_rooms,
)
from faultline.envs.layout import (
    WORLD_SIZE,
    blocked_two_path_doors,
    layout_from_dict,
    layout_to_dict,
    long_corridor_doors,
    long_mixed_star_doors,
)


def test_default_doors_are_the_four_corridor_walls():
    """The default layout opens exactly the four corridor walls."""
    assert set(default_doors()) == {("v", 0, 0), ("h", 1, 0), ("v", 1, 1), ("h", 2, 1)}


def test_a_door_covers_only_its_centred_span():
    """A corridor door is open at the wall centre and solid near the corner."""
    layout = RoomsLayout(default_doors())
    assert layout.is_open(("v", 0, 0), 4.0) is True
    assert layout.is_open(("v", 0, 0), 1.0) is False


def test_a_wall_without_a_door_is_solid():
    """A wall absent from the door map is solid everywhere."""
    layout = RoomsLayout(default_doors())
    assert layout.is_open(("v", 0, 1), 4.0) is False


def test_move_through_a_door_is_not_blocked():
    """Crossing a wall at its door does not block the move."""
    layout = RoomsLayout(default_doors())
    assert layout.segment_is_blocked([7.0, 4.0], [9.0, 4.0]) is False


def test_move_through_a_solid_wall_is_blocked():
    """Crossing a wall away from its door blocks the move."""
    layout = RoomsLayout(default_doors())
    assert layout.segment_is_blocked([7.0, 1.0], [9.0, 1.0]) is True


def test_move_along_a_wall_is_not_blocked():
    """A move that stays on a wall line crosses no wall."""
    layout = RoomsLayout(default_doors())
    assert layout.segment_is_blocked([8.0, 4.0], [8.0, 6.0]) is False


def test_move_inside_a_room_is_not_blocked():
    """A move that stays inside one room crosses no wall."""
    layout = RoomsLayout(default_doors())
    assert layout.segment_is_blocked([2.0, 2.0], [3.0, 3.0]) is False


@pytest.mark.parametrize(
    "point,room",
    [([2.0, 2.0], (0, 0)), ([10.0, 10.0], (1, 1)), ([22.0, 22.0], (2, 2)), ([23.9, 0.1], (2, 0))],
)
def test_room_of_maps_points_to_grid_rooms(point, room):
    """A point maps to the room that contains it."""
    assert RoomsLayout().room_of(point) == room


def test_clamp_keeps_points_inside_the_world():
    """Clamping pulls an outside point back to the world bounds."""
    assert np.allclose(RoomsLayout().clamp([30.0, -5.0]), [WORLD_SIZE, 0.0])


def test_default_regions_sit_inside_their_rooms():
    """Each named region accepts its room centre and rejects the others."""
    regions = default_regions()
    assert regions["start"].evaluate([4.0, 4.0]) is True
    assert regions["start"].evaluate([12.0, 12.0]) is False
    assert regions["mid"].evaluate([12.0, 12.0]) is True
    assert regions["mid"].evaluate([4.0, 4.0]) is False
    assert regions["goal"].evaluate([20.0, 20.0]) is True
    assert regions["goal"].evaluate([12.0, 12.0]) is False


def test_open_layout_opens_every_wall_away_from_the_sealed_rooms():
    """Only the four walls bounding the two trap rooms stay out of the door map."""
    doors = open_doors_except_sealed_rooms({(0, 2), (1, 2)})
    assert set(doors) == {
        ("v", 0, 0),
        ("v", 0, 1),
        ("v", 1, 0),
        ("v", 1, 1),
        ("h", 0, 0),
        ("h", 1, 0),
        ("h", 2, 0),
        ("h", 2, 1),
    }


def test_open_layout_seals_the_trap_rooms_but_opens_the_goal_room():
    """A move into a sealed room blocks; a move into the reachable goal does not."""
    layout = RoomsLayout(doors=open_doors_except_sealed_rooms({(0, 2), (1, 2)}))
    assert layout.segment_is_blocked([12.0, 15.0], [12.0, 17.0]) is True
    assert layout.segment_is_blocked([20.0, 15.0], [20.0, 17.0]) is False
    assert layout.segment_is_blocked([4.0, 4.0], [20.0, 4.0]) is False


def test_long_corridor_doors_are_the_six_open_walls():
    """The paper's long corridor opens exactly its six unblocked walls."""
    assert set(long_corridor_doors()) == {
        ("h", 0, 0),
        ("h", 0, 1),
        ("h", 1, 0),
        ("h", 1, 1),
        ("v", 0, 2),
        ("v", 1, 0),
    }


def test_long_corridor_route_climbs_the_left_column_and_crosses_the_top():
    """The winding route from START room (0, 0) to GOAL room (1, 1) is open."""
    layout = RoomsLayout(doors=long_corridor_doors())
    # Up the left column: (0, 0) - (0, 1) - (0, 2).
    assert layout.segment_is_blocked([4.0, 4.0], [4.0, 12.0]) is False
    assert layout.segment_is_blocked([4.0, 12.0], [4.0, 20.0]) is False
    # Across the top into room (1, 2).
    assert layout.segment_is_blocked([4.0, 20.0], [12.0, 20.0]) is False
    # Down into the centre GOAL room.
    assert layout.segment_is_blocked([12.0, 20.0], [12.0, 12.0]) is False


def test_long_corridor_walls_off_the_direct_route_and_right_column():
    """The shortcut past the corridor and the centre-right column stay solid."""
    layout = RoomsLayout(doors=long_corridor_doors())
    # No direct move from START room (0, 0) into the middle column.
    assert layout.segment_is_blocked([4.0, 4.0], [12.0, 4.0]) is True
    # The right column is sealed from the centre at the top and middle rows.
    assert layout.segment_is_blocked([12.0, 12.0], [20.0, 12.0]) is True
    assert layout.segment_is_blocked([12.0, 20.0], [20.0, 20.0]) is True


def test_long_mixed_star_doors_are_the_six_open_walls():
    """The paper's long mixed star room opens exactly its six unblocked walls."""
    assert set(long_mixed_star_doors()) == {
        ("v", 0, 0),
        ("v", 1, 0),
        ("h", 0, 0),
        ("v", 0, 1),
        ("v", 1, 1),
        ("h", 2, 1),
    }


def test_long_mixed_star_route_zigzags_to_the_top_right_goal():
    """The star's route climbs, steps east twice, then climbs into the goal."""
    layout = RoomsLayout(doors=long_mixed_star_doors())
    assert layout.segment_is_blocked([4.0, 4.0], [4.0, 12.0]) is False  # (0, 0)-(0, 1)
    assert layout.segment_is_blocked([4.0, 12.0], [12.0, 12.0]) is False  # (0, 1)-(1, 1)
    assert layout.segment_is_blocked([12.0, 12.0], [20.0, 12.0]) is False  # (1, 1)-(2, 1)
    assert layout.segment_is_blocked([20.0, 12.0], [20.0, 20.0]) is False  # (2, 1)-(2, 2)


def test_long_mixed_star_seals_the_top_row_and_opens_the_bottom_arm():
    """The two top-row rooms stay sealed; the bottom row is a dead-end arm."""
    layout = RoomsLayout(doors=long_mixed_star_doors())
    # The sealed top-left room (0, 2) cannot be entered from below or the east.
    assert layout.segment_is_blocked([4.0, 15.0], [4.0, 17.0]) is True
    assert layout.segment_is_blocked([4.0, 20.0], [12.0, 20.0]) is True
    # The bottom row is a second arm: (0, 0) - (1, 0) - (2, 0).
    assert layout.segment_is_blocked([4.0, 4.0], [12.0, 4.0]) is False
    assert layout.segment_is_blocked([12.0, 4.0], [20.0, 4.0]) is False


def test_blocked_two_path_doors_are_the_eight_open_walls():
    """The two-path room opens its two mid merges, two centre walls, four column walls."""
    doors = blocked_two_path_doors()
    assert set(doors) == {
        ("v", 0, 2),
        ("v", 1, 0),
        ("h", 1, 1),
        ("h", 1, 0),
        ("h", 0, 0),
        ("h", 0, 1),
        ("h", 2, 0),
        ("h", 2, 1),
    }
    # The left and right column walls span their whole edge, merging the stacked rooms.
    assert doors[("h", 0, 0)] == (0.0, 8.0)
    assert doors[("h", 0, 1)] == (0.0, 8.0)
    assert doors[("h", 2, 0)] == (16.0, 24.0)
    assert doors[("h", 2, 1)] == (16.0, 24.0)


def test_blocked_two_path_winds_from_start_to_goal():
    """The route climbs the left, drops the centre, and climbs the right column."""
    layout = RoomsLayout(doors=blocked_two_path_doors())
    # Up the left column into MID1: (0, 0) - (0, 1) - (0, 2). The column is open
    # across its whole edge, so an off-centre climb passes too.
    assert layout.segment_is_blocked([4.0, 4.0], [4.0, 12.0]) is False
    assert layout.segment_is_blocked([4.0, 12.0], [4.0, 20.0]) is False
    assert layout.segment_is_blocked([1.0, 4.0], [1.0, 12.0]) is False
    assert layout.segment_is_blocked([7.0, 12.0], [7.0, 20.0]) is False
    # Across MID1 into room (1, 2).
    assert layout.segment_is_blocked([4.0, 20.0], [12.0, 20.0]) is False
    # Down the centre into MID2: (1, 2) - (1, 1) - (1, 0).
    assert layout.segment_is_blocked([12.0, 20.0], [12.0, 12.0]) is False
    assert layout.segment_is_blocked([12.0, 12.0], [12.0, 4.0]) is False
    # Across MID2 into room (2, 0).
    assert layout.segment_is_blocked([12.0, 4.0], [20.0, 4.0]) is False
    # Up the right column into GOAL: (2, 0) - (2, 1) - (2, 2), also open across
    # its whole edge.
    assert layout.segment_is_blocked([20.0, 4.0], [20.0, 12.0]) is False
    assert layout.segment_is_blocked([20.0, 12.0], [20.0, 20.0]) is False
    assert layout.segment_is_blocked([17.0, 4.0], [17.0, 12.0]) is False
    assert layout.segment_is_blocked([23.0, 12.0], [23.0, 20.0]) is False


def test_blocked_two_path_blocks_both_direct_shortcuts():
    """START is walled off from MID2 and MID1 from GOAL, so neither edge is direct."""
    layout = RoomsLayout(doors=blocked_two_path_doors())
    # START cannot step straight into the bottom-middle room.
    assert layout.segment_is_blocked([7.0, 4.0], [9.0, 4.0]) is True
    # MID1 cannot step straight into GOAL.
    assert layout.segment_is_blocked([15.0, 20.0], [17.0, 20.0]) is True


def test_layout_round_trips_a_ten_by_ten_grid():
    """A 10x10 full-edge layout survives the JSON mapping unchanged."""
    doors = open_doors_except_sealed_rooms((), room_size=8.0, rooms_per_side=10)
    layout = RoomsLayout(doors, room_size=8.0, rooms_per_side=10)

    restored = layout_from_dict(layout_to_dict(layout))

    assert restored.room_size == 8.0
    assert restored.rooms_per_side == 10
    assert restored.world_size == 80.0
    assert restored.doors == layout.doors
    assert len(restored.doors) == 180
    assert restored.is_open(("v", 4, 7), 7.5 * 8.0) is True
    assert restored.is_open(("h", 2, 5), 2.5 * 8.0) is True
