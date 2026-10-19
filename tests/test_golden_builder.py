"""Gold construction: sealed pockets, in-room traps, waypoints, and Tier 1 failures."""

from __future__ import annotations

import numpy as np

from faultline.envs.rooms.layout import RoomsLayout, full_edge_interval
from faultline.golden.builder import build_gold, chain_graph, cut_room_waypoints, rooms_of_box
from faultline.spectrl.predicates import Box

RESOLUTION = 24  # 3x3 rooms of 8 units, 1-unit cells


def single_edge(start: Box, goal: Box):
    return chain_graph(start, (), goal)


def open_room_layout() -> RoomsLayout:
    return RoomsLayout(doors={})


def test_a_clean_intended_graph_is_kept_exactly():
    start, goal = Box((1.0, 1.0), (3.0, 3.0)), Box((5.0, 1.0), (7.0, 3.0))
    gold = build_gold("clean", open_room_layout(), single_edge(start, goal),
                      resolution=RESOLUTION, max_steps=20)
    assert gold.certificate.tier1_ok
    assert gold.graph.vertices[0].region == start and gold.graph.vertices[1].region == goal
    assert all(value == 0.0 for value in gold.shrinks.values())


def test_a_goal_box_reaching_into_a_sealed_room_is_cut_back():
    start, goal = Box((1.0, 1.0), (3.0, 3.0)), Box((5.0, 1.0), (10.0, 3.0))
    gold = build_gold("pocket", open_room_layout(), single_edge(start, goal),
                      resolution=RESOLUTION, max_steps=20)
    region = gold.graph.target.region
    assert gold.certificate.tier1_ok
    assert region.upper[0] <= 8.0 and region.lower[0] == 5.0
    assert gold.shrinks["goal"] == 1.0 - 3 / 5


def test_a_trap_inside_a_room_is_excluded_from_gold_safe_and_from_the_goal():
    start, goal = Box((1.0, 1.0), (2.0, 2.0)), Box((5.0, 1.0), (7.0, 3.0))
    trap = Box((6.0, 0.0), (7.0, 8.0))
    gold = build_gold("trap", open_room_layout(), single_edge(start, goal), (trap,),
                      resolution=RESOLUTION, max_steps=20)
    edge = gold.graph.edges[0]
    assert gold.certificate.tier1_ok
    assert not edge.safe.evaluate(np.array([6.5, 4.0])) and edge.safe.evaluate(np.array([3.0, 4.0]))
    assert gold.graph.target.region.upper[0] <= 6.0


def test_a_sealed_start_fails_tier_one_with_a_reason():
    start, goal = Box((9.0, 1.0), (11.0, 3.0)), Box((1.0, 1.0), (3.0, 3.0))
    gold = build_gold("sealed", open_room_layout(), single_edge(start, goal),
                      resolution=RESOLUTION, max_steps=20)
    assert not gold.certificate.tier1_ok
    assert any("start" in problem for problem in gold.certificate.problems)


def test_losing_more_than_the_allowed_fraction_fails_tier_one():
    start, goal = Box((1.0, 1.0), (3.0, 3.0)), Box((7.0, 1.0), (12.0, 3.0))
    gold = build_gold("mostly_sealed", open_room_layout(), single_edge(start, goal),
                      resolution=RESOLUTION, max_steps=20)
    assert not gold.certificate.tier1_ok


def test_building_is_deterministic():
    start, goal = Box((1.0, 1.0), (3.0, 3.0)), Box((5.0, 1.0), (10.0, 3.0))
    first = build_gold("a", open_room_layout(), single_edge(start, goal),
                       resolution=RESOLUTION, max_steps=20)
    second = build_gold("a", open_room_layout(), single_edge(start, goal),
                        resolution=RESOLUTION, max_steps=20)
    assert first.graph.target.region == second.graph.target.region


def corridor_layout() -> RoomsLayout:
    keys = [("v", 0, 0), ("v", 1, 0)]
    return RoomsLayout(doors={key: full_edge_interval(key, 8.0) for key in keys})


def test_waypoints_sit_at_the_rooms_every_route_must_cross():
    start, goal = Box((1.0, 1.0), (3.0, 3.0)), Box((18.0, 1.0), (22.0, 3.0))
    waypoints = cut_room_waypoints(corridor_layout(), start, goal)
    assert len(waypoints) == 1
    assert 8.0 <= waypoints[0].lower[0] and waypoints[0].upper[0] <= 16.0
    gold = build_gold("corridor", corridor_layout(), chain_graph(start, waypoints, goal),
                      resolution=RESOLUTION, max_steps=40)
    assert gold.certificate.tier1_ok
    assert [vertex.name for vertex in gold.graph.vertices] == ["start", "w1", "goal"]


def test_no_waypoint_is_needed_when_routes_are_not_forced():
    open_layout = RoomsLayout(
        doors={key: full_edge_interval(key, 8.0)
               for key in [("v", c, r) for c in range(2) for r in range(3)]
               + [("h", c, r) for c in range(3) for r in range(2)]}
    )
    start, goal = Box((1.0, 1.0), (3.0, 3.0)), Box((18.0, 18.0), (22.0, 22.0))
    assert cut_room_waypoints(open_layout, start, goal) == ()


def test_rooms_of_box_treats_the_upper_edge_as_exclusive():
    assert rooms_of_box(Box((1.0, 1.0), (8.0, 8.0)), 8.0) == ((0, 0),)
    assert rooms_of_box(Box((7.0, 1.0), (9.0, 3.0)), 8.0) == ((0, 0), (1, 0))


def test_an_edge_longer_than_the_horizon_fails_tier_one():
    start, goal = Box((1.0, 1.0), (3.0, 3.0)), Box((5.0, 1.0), (7.0, 3.0))
    gold = build_gold("short_horizon", open_room_layout(), single_edge(start, goal),
                      resolution=RESOLUTION, max_steps=1)
    assert not gold.certificate.tier1_ok
    assert any("horizon" in problem for problem in gold.certificate.problems)


def test_the_horizon_check_passes_when_the_edge_fits():
    start, goal = Box((1.0, 1.0), (3.0, 3.0)), Box((5.0, 1.0), (7.0, 3.0))
    gold = build_gold("fits", open_room_layout(), single_edge(start, goal),
                      resolution=RESOLUTION, max_steps=20)
    assert gold.certificate.tier1_ok
