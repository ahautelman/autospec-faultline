"""Tests for the exact polygon form of region predicates.

Expected areas are worked out by hand from the shapes, so a clipping bug shows up as a
wrong area rather than being mirrored by the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from faultline.geometry import convex_hull
from faultline.spectrl.predicates import And, Box, Disc, Empty, HalfPlane, Not, Or, Universe
from faultline.viz.polygons import clip_half_plane, polygon_area, region_polygons

WORLD = 24.0


def _total_area(region):
    return sum(polygon_area(polygon) for polygon in region_polygons(region, WORLD))


def test_box_is_one_rectangle_with_its_area():
    assert _total_area(Box((0.0, 0.0), (4.0, 3.0))) == pytest.approx(12.0)


def test_overlapping_boxes_intersect_to_their_overlap():
    region = And(Box((0.0, 0.0), (4.0, 4.0)), Box((2.0, 2.0), (6.0, 6.0)))

    assert _total_area(region) == pytest.approx(4.0)


def test_disjoint_boxes_intersect_to_nothing():
    region = And(Box((0.0, 0.0), (1.0, 1.0)), Box((2.0, 2.0), (3.0, 3.0)))

    assert region_polygons(region, WORLD) == []


def test_half_plane_is_clipped_to_the_world():
    region = HalfPlane((1.0, 0.0), 6.0)

    assert _total_area(region) == pytest.approx(18.0 * WORLD)


def test_box_cut_by_a_diagonal_half_plane_is_a_triangle():
    region = And(Box((0.0, 0.0), (4.0, 4.0)), HalfPlane((1.0, 1.0), 4.0))

    assert _total_area(region) == pytest.approx(8.0)


def test_convex_hull_is_its_own_polygon():
    hull = convex_hull([(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)])

    assert _total_area(hull) == pytest.approx(6.0)


def test_degenerate_hull_has_no_polygon():
    hull = convex_hull([(0.0, 0.0), (4.0, 0.0)])

    assert region_polygons(hull, WORLD) == []


def test_union_keeps_each_piece():
    region = Or(Box((0.0, 0.0), (1.0, 1.0)), Box((5.0, 5.0), (7.0, 8.0)))

    assert len(region_polygons(region, WORLD)) == 2


def test_disc_area_is_close_to_a_circle():
    region = Disc((10.0, 10.0), 3.0)

    assert _total_area(region) == pytest.approx(math.pi * 9.0, rel=2e-3)


def test_universe_is_the_whole_world_and_empty_is_nothing():
    assert _total_area(Universe()) == pytest.approx(WORLD * WORLD)
    assert region_polygons(Empty(), WORLD) == []


@pytest.mark.parametrize(
    "region",
    [Not(Box((0.0, 0.0), (1.0, 1.0))), And(Box((0.0, 0.0), (4.0, 4.0)), Not(Disc((1.0, 1.0), 1.0)))],
    ids=["not", "and_with_not"],
)
def test_a_complement_needs_a_raster(region):
    assert region_polygons(region, WORLD) is None


def test_other_coordinates_need_a_raster():
    assert region_polygons(Box((0.0,), (1.0,), dims=(2,)), WORLD) is None


def test_clip_keeps_the_inside_and_adds_crossing_points():
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])

    clipped = clip_half_plane(square, np.array([1.0, 0.0]), 1.0)

    assert polygon_area(clipped) == pytest.approx(2.0)
    assert clipped[:, 0].min() == pytest.approx(1.0)


def test_clip_to_a_half_plane_missing_the_polygon_is_empty():
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])

    assert len(clip_half_plane(square, np.array([1.0, 0.0]), 5.0)) == 0
