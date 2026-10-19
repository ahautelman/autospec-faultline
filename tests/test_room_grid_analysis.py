"""Brute-force cross-checks of the room-grid queries the gold builder relies on (spec T1).

``tests/test_room_grid.py`` already checks cut rooms. This file checks shortest
path, viability and reachability against independent, deliberately naive
implementations on random small grids.
"""

from __future__ import annotations

import random

from faultline.envs.rooms.grid import RoomGrid
from faultline.envs.rooms.layout import full_edge_interval

SIZE = 4


def random_grid(rng: random.Random) -> RoomGrid:
    keys = [("v", c, r) for c in range(SIZE - 1) for r in range(SIZE)]
    keys += [("h", c, r) for c in range(SIZE) for r in range(SIZE - 1)]
    doors = {key: full_edge_interval(key, 8.0) for key in keys if rng.random() < 0.5}
    return RoomGrid(SIZE, doors, room_size=8.0)


def naive_distances(grid: RoomGrid, sources: set, allowed: set) -> dict:
    """Bellman-Ford style relaxation, independent of the grid's own BFS."""
    dist = {room: 0 for room in sources}
    changed = True
    while changed:
        changed = False
        for room in list(dist):
            for nxt in grid.neighbours(room):
                if nxt in allowed and dist.get(nxt, 10**6) > dist[room] + 1:
                    dist[nxt] = dist[room] + 1
                    changed = True
    return dist


def random_rooms(rng: random.Random, count: int) -> set:
    return {(rng.randrange(SIZE), rng.randrange(SIZE)) for _ in range(count)}


def test_shortest_path_length_matches_naive_relaxation():
    rng = random.Random(11)
    for _ in range(60):
        grid = random_grid(rng)
        sources, targets = random_rooms(rng, 2), random_rooms(rng, 2)
        dist = naive_distances(grid, sources, set(grid.rooms))
        expected = min((dist[t] for t in targets if t in dist), default=None)
        path = grid.shortest_path(sources, targets)
        assert (path.length if path else None) == expected


def test_reachable_matches_naive_relaxation_inside_a_safe_set():
    rng = random.Random(12)
    for _ in range(60):
        grid = random_grid(rng)
        sources = random_rooms(rng, 2)
        safe = {room for room in grid.rooms if rng.random() > 0.3}
        expected = set(naive_distances(grid, sources, safe | sources))
        assert grid.reachable(sources, safe) == expected


def test_viability_matches_per_room_forward_search():
    """A room is viable when it is safe (or a target) and has a route through safe rooms."""
    rng = random.Random(13)
    for _ in range(60):
        grid = random_grid(rng)
        targets = random_rooms(rng, 2)
        safe = {room for room in grid.rooms if rng.random() > 0.3}
        allowed = safe | targets
        expected = {
            room
            for room in allowed
            if set(naive_distances(grid, {room}, allowed)) & targets
        }
        assert grid.viability(targets, safe) == expected


def test_reachable_and_viability_are_mirror_images_when_endpoints_are_safe():
    rng = random.Random(14)
    for _ in range(40):
        grid = random_grid(rng)
        start, goal = random_rooms(rng, 1), random_rooms(rng, 1)
        safe = {room for room in grid.rooms if rng.random() > 0.3} | start | goal
        assert bool(grid.reachable(start, safe) & goal) == bool(grid.viability(goal, safe) & start)
