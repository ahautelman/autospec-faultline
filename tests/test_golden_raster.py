"""The cell grid agrees with the room-level analysis and the rectangle search is exact."""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from faultline.envs.rooms.grid import RoomGrid
from faultline.envs.rooms.layout import RoomsLayout, full_edge_interval
from faultline.golden.raster import CellGrid, largest_rectangle
from faultline.spectrl.predicates import Box


def random_layout(rng: random.Random, size: int = 3) -> RoomsLayout:
    keys = [("v", c, r) for c in range(size - 1) for r in range(size)]
    keys += [("h", c, r) for c in range(size) for r in range(size - 1)]
    doors = {key: full_edge_interval(key, 8.0) for key in keys if rng.random() < 0.5}
    return RoomsLayout(doors=doors, room_size=8.0, rooms_per_side=size)


def test_resolution_must_divide_the_room_grid():
    with pytest.raises(ValueError):
        CellGrid(RoomsLayout(), 10)


def test_cell_reachability_matches_room_components_on_random_layouts():
    rng = random.Random(21)
    for _ in range(15):
        layout = random_layout(rng)
        cells = CellGrid(layout, resolution=6)
        rooms = RoomGrid(3, layout.doors, 8.0)
        origin = (rng.randrange(3), rng.randrange(3))
        source = np.zeros((6, 6), dtype=bool)
        source[origin[0] * 2, origin[1] * 2] = True
        reached = cells.reachable(source, np.ones((6, 6), dtype=bool))
        component = rooms.component_of(origin)
        for col, row in itertools.product(range(3), range(3)):
            in_room = reached[col * 2 : col * 2 + 2, row * 2 : row * 2 + 2].all()
            assert in_room == ((col, row) in component)


def test_a_solid_wall_blocks_and_a_door_opens_the_same_move():
    solid = CellGrid(RoomsLayout(), resolution=6)
    opened = CellGrid(
        RoomsLayout(doors={("v", 0, 0): full_edge_interval(("v", 0, 0), 8.0)}), resolution=6
    )
    source = np.zeros((6, 6), dtype=bool)
    source[0, 0] = True
    everywhere = np.ones((6, 6), dtype=bool)
    assert not solid.reachable(source, everywhere)[2, 0]
    assert opened.reachable(source, everywhere)[2, 0]


def test_unsafe_cells_block_a_route_but_unsafe_sources_are_kept():
    grid = CellGrid(RoomsLayout(doors={}, room_size=8.0, rooms_per_side=1), resolution=4)
    source = np.zeros((4, 4), dtype=bool)
    source[0, 0] = True
    safe = np.ones((4, 4), dtype=bool)
    safe[1, :] = False  # a full-height trap column
    reached = grid.reachable(source, safe)
    assert reached[0, 0] and not reached[2, 0]


def test_mask_and_box_of_round_trip():
    grid = CellGrid(RoomsLayout(), resolution=24)
    box = Box((2.0, 3.0), (5.0, 4.0))
    mask = grid.mask(box)
    rectangle = largest_rectangle(mask)
    assert grid.box_of(rectangle) == box


def test_largest_rectangle_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        mask = rng.random((int(rng.integers(1, 7)), int(rng.integers(1, 7)))) > 0.35
        x_size, y_size = mask.shape
        best = max(
            (
                (x1 - x0 + 1) * (y1 - y0 + 1)
                for x0, x1 in itertools.combinations_with_replacement(range(x_size), 2)
                for y0, y1 in itertools.combinations_with_replacement(range(y_size), 2)
                if mask[x0 : x1 + 1, y0 : y1 + 1].all()
            ),
            default=0,
        )
        rectangle = largest_rectangle(mask)
        area = 0 if rectangle is None else (rectangle[2] - rectangle[0] + 1) * (rectangle[3] - rectangle[1] + 1)
        assert area == best
        if rectangle is not None:
            assert mask[rectangle[0] : rectangle[2] + 1, rectangle[1] : rectangle[3] + 1].all()


def test_largest_rectangle_of_an_empty_mask_is_none():
    assert largest_rectangle(np.zeros((3, 3), dtype=bool)) is None
