"""Behavioural tests for the room-level graph analysis.

The tests use small hand-built grids so every expected component, path and cut
room is obvious from the coordinates. A brute-force cross-check keeps the
cut-room scan honest on random small grids.
"""

from __future__ import annotations

import random
from collections import deque

import pytest

from faultline.envs.rooms.grid import RoomGrid, RoomPath
from faultline.envs.rooms.layout import open_doors_except_sealed_rooms
from faultline.envs.rooms.regions import RegionBlock

ROOM_SIZE = 8.0


def full_doors(rooms_per_side: int) -> dict:
    """Return every internal wall as a full-edge door."""
    return open_doors_except_sealed_rooms(
        (), room_size=ROOM_SIZE, rooms_per_side=rooms_per_side
    )


def grid_without(rooms_per_side: int, missing: set) -> RoomGrid:
    """Build a fully open grid with the named walls left solid."""
    doors = full_doors(rooms_per_side)
    for key in missing:
        doors.pop(key, None)
    return RoomGrid(rooms_per_side, doors, ROOM_SIZE)


def brute_force_cut_rooms(grid: RoomGrid, sources: set, targets: set) -> frozenset:
    """Independently scan every non-endpoint room for disconnection."""

    def reaches(blocked: frozenset) -> bool:
        visited = set(sources)
        queue = deque(sorted(sources))
        while queue:
            room = queue.popleft()
            if room in targets:
                return True
            for neighbour in grid.neighbours(room):
                if neighbour not in visited and neighbour not in blocked:
                    visited.add(neighbour)
                    queue.append(neighbour)
        return False

    if not reaches(frozenset()):
        return frozenset()
    return frozenset(
        room
        for room in grid.rooms
        if room not in sources | targets and not reaches(frozenset((room,)))
    )


def test_neighbours_follow_the_open_door_map():
    """A solid wall removes exactly the neighbour it separates."""
    grid = grid_without(3, {("v", 1, 1)})

    assert grid.neighbours((1, 1)) == ((0, 1), (1, 0), (1, 2))
    assert grid.neighbours((0, 0)) == ((0, 1), (1, 0))


def test_components_merge_open_rooms_and_isolate_a_walled_room():
    """Three rooms join and the fourth stays alone behind two solid walls."""
    grid = grid_without(2, {("v", 0, 1), ("h", 1, 0)})

    assert grid.components() == (
        frozenset({(0, 0), (1, 0), (0, 1)}),
        frozenset({(1, 1)}),
    )
    assert grid.component_of((1, 0)) == frozenset({(0, 0), (1, 0), (0, 1)})


def test_shortest_path_counts_door_crossings_and_is_repeatable():
    """A 3x3 open grid needs four crossings; repeated calls agree."""
    grid = RoomGrid(3, full_doors(3), ROOM_SIZE)

    first = grid.shortest_path({(0, 0)}, {(2, 2)})
    second = grid.shortest_path({(0, 0)}, {(2, 2)})

    assert first is not None
    assert first == second
    assert first.rooms[0] == (0, 0)
    assert first.rooms[-1] == (2, 2)
    assert first.length == 4
    assert RoomPath(((1, 1),)).length == 0


def test_shortest_path_returns_none_when_the_goal_is_sealed():
    """A goal behind solid walls is unreachable."""
    grid = grid_without(3, {("v", 1, 2), ("h", 2, 1)})

    assert grid.shortest_path({(0, 0)}, {(2, 2)}) is None


def test_cut_rooms_match_a_brute_force_scan_on_random_small_grids():
    """The exhaustive cut scan agrees with an independent scan."""
    rng = random.Random(7)
    for rooms_per_side in (2, 3, 4):
        base_keys = sorted(full_doors(rooms_per_side))
        for _ in range(5):
            doors = {
                key: interval
                for key, interval in full_doors(rooms_per_side).items()
                if rng.random() > 0.35
            }
            grid = RoomGrid(rooms_per_side, doors, ROOM_SIZE)
            sources = {grid.rooms[0]}
            targets = {grid.rooms[-1]}
            assert set(base_keys) >= set(doors)
            assert grid.cut_rooms(sources, targets) == brute_force_cut_rooms(
                grid, sources, targets
            )


def test_cut_rooms_finds_the_single_bridge():
    """A one-room corridor has every interior room as a cut room."""
    open_walls = {("v", 0, 0), ("h", 1, 0), ("h", 1, 1)}
    grid = grid_without(3, set(full_doors(3)) - open_walls)

    # The only open route is (0, 0) - (1, 0) - (1, 1) - (1, 2).
    assert grid.shortest_path({(0, 0)}, {(1, 2)}).rooms == (
        (0, 0),
        (1, 0),
        (1, 1),
        (1, 2),
    )
    assert grid.cut_rooms({(0, 0)}, {(1, 2)}) == frozenset({(1, 0), (1, 1)})


def test_sealed_rooms_are_outside_the_reference_component():
    """A room walled off from the reference is reported as sealed."""
    grid = grid_without(3, {("v", 1, 2), ("h", 2, 1)})

    assert grid.sealed_rooms({(0, 0)}) == frozenset({(2, 2)})


def test_partial_door_gap_is_rejected():
    """A centred gap is not a full-edge door, so analysis refuses it."""
    with pytest.raises(ValueError, match="full-edge"):
        RoomGrid(2, {("v", 0, 0): (2.0, 6.0)}, ROOM_SIZE)


def test_region_internal_components_report_a_split_block():
    """Two vertical pairs inside one 2x2 block are separate components."""
    doors = {("v", 0, 0): (0.0, 8.0), ("v", 0, 1): (8.0, 16.0)}
    grid = RoomGrid(2, doors, ROOM_SIZE)
    block = RegionBlock(0, 0)

    assert grid.region_rooms(block) == ((0, 0), (1, 0), (0, 1), (1, 1))
    assert grid.region_internal_components(block) == (
        frozenset({(0, 0), (1, 0)}),
        frozenset({(0, 1), (1, 1)}),
    )


def test_viability_keeps_only_rooms_with_a_safe_route_to_the_target():
    """An isolated room cannot reach the target while staying safe."""
    grid = grid_without(3, {("v", 0, 1), ("v", 1, 1), ("h", 1, 0), ("h", 1, 1)})
    room_count_before = len(grid.rooms)

    viable = grid.viability({(2, 2)}, set(grid.rooms) - {(1, 1)})

    assert (1, 1) not in viable
    assert viable == frozenset(set(grid.rooms) - {(1, 1)})
    assert len(viable) == room_count_before - 1
