"""Tests for the specification-level satisfaction metric of the 9-rooms report.

The metric chains a refined route's policies from the original edge's start
region and scores the joined trajectory against the original edge. These tests
drive it with a scripted straight-line controller in an open arena, so the
expected outcome follows from the geometry alone.
"""

from __future__ import annotations

import numpy as np

from faultline.envs import Rooms9, RoomsLayout, open_doors_except_sealed_rooms
from faultline.experiments.satisfaction import route_satisfaction
from faultline.scenarios.catalogue import single_edge
from faultline.spectrl.predicates import Box

START = Box((1.0, 1.0), (7.0, 7.0))
WAYPOINT = Box((9.0, 9.0), (15.0, 15.0))
GOAL = Box((17.0, 17.0), (23.0, 23.0))
ROLLOUTS = 20


def open_arena() -> Rooms9:
    return Rooms9(layout=RoomsLayout(doors=open_doors_except_sealed_rooms(())))


def straight_to_goal(observation: np.ndarray) -> np.ndarray:
    # The observation carries the heading and distance to the sampled goal.
    return np.array([min(2.0, float(observation[3])), float(observation[2])])


def test_a_route_that_is_the_original_edge_satisfies_it_from_every_start():
    graph = single_edge(START, GOAL)
    edge = graph.edges[0]

    score = route_satisfaction(open_arena(), edge, ((edge, straight_to_goal),), ROLLOUTS, 0)

    assert score == 1.0


def test_starts_outside_the_refined_route_count_as_failures():
    original = single_edge(START, GOAL).edges[0]
    disjoint = single_edge(Box((9.0, 1.0), (15.0, 7.0)), GOAL).edges[0]

    score = route_satisfaction(
        open_arena(), original, ((disjoint, straight_to_goal),), ROLLOUTS, 0
    )

    assert score == 0.0


def test_a_two_hop_route_is_chained_from_where_the_first_hop_ended():
    original = single_edge(START, GOAL).edges[0]
    first = single_edge(START, WAYPOINT).edges[0]
    second = single_edge(WAYPOINT, GOAL).edges[0]

    score = route_satisfaction(
        open_arena(), original, ((first, straight_to_goal), (second, straight_to_goal)), ROLLOUTS, 0
    )

    assert score == 1.0
