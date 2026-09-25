"""Tests for the refinement geometry helpers.

The convex hull, the region algebra, and the half-plane partition are the only
non-learning pieces of the operators, so their boundaries are pinned exactly:
where the hull ends, what intersection and subtraction mean, and which side of
the fitted half-plane holds the positive class.
"""

from __future__ import annotations

import numpy as np

from faultline.geometry import convex_hull, difference, half_plane_partition, intersect
from faultline.spectrl.predicates import Box, Empty, Universe


def test_convex_hull_of_no_points_is_empty():
    assert isinstance(convex_hull([]), Empty)


def test_convex_hull_of_one_point_matches_only_that_point():
    hull = convex_hull([[1.0, 1.0]])
    assert hull.evaluate(np.array([1.0, 1.0])) is True
    assert hull.evaluate(np.array([1.5, 1.0])) is False


def test_convex_hull_of_three_points_is_the_triangle():
    hull = convex_hull([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    # A point strictly inside, one on the hypotenuse, and one outside the corner.
    assert hull.evaluate(np.array([0.5, 0.5])) is True
    assert hull.evaluate(np.array([1.0, 1.0])) is True
    assert hull.evaluate(np.array([3.0, 3.0])) is False


def test_convex_hull_of_collinear_points_is_the_segment():
    hull = convex_hull([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert hull.evaluate(np.array([1.0, 0.0])) is True
    assert hull.evaluate(np.array([1.0, 0.5])) is False
    assert hull.evaluate(np.array([3.0, 0.0])) is False


def test_convex_hull_points_are_projected_onto_the_given_dims():
    # A 4-D state whose planar part is the triangle, plus irrelevant coordinates.
    points = [[0.0, 0.0, 9.0, 9.0], [2.0, 0.0, 9.0, 9.0], [0.0, 2.0, 9.0, 9.0]]
    hull = convex_hull(points)
    assert hull.evaluate(np.array([0.5, 0.5, 0.0, 0.0])) is True


def test_intersect_requires_both_regions():
    left = Box((0.0, 0.0), (2.0, 2.0))
    right = Box((1.0, 1.0), (3.0, 3.0))
    region = intersect(left, right)
    assert region.evaluate(np.array([1.5, 1.5])) is True
    assert region.evaluate(np.array([0.5, 0.5])) is False
    assert region.evaluate(np.array([2.5, 2.5])) is False


def test_intersect_shortcuts_the_universe_and_empty():
    region = Box((0.0, 0.0), (2.0, 2.0))
    assert intersect(Universe(), region) == region
    assert intersect(region, Universe()) == region
    assert isinstance(intersect(region, Empty()), Empty)


def test_difference_removes_the_overlap():
    left = Box((0.0, 0.0), (2.0, 2.0))
    right = Box((1.0, 1.0), (3.0, 3.0))
    region = difference(left, right)
    assert region.evaluate(np.array([0.5, 0.5])) is True
    assert region.evaluate(np.array([1.5, 1.5])) is False
    assert region.evaluate(np.array([2.5, 2.5])) is False


def test_difference_shortcuts_the_universe_and_empty():
    region = Box((0.0, 0.0), (2.0, 2.0))
    assert isinstance(difference(region, Universe()), Empty)
    assert difference(region, Empty()) == region
    assert difference(Universe(), Empty()) == Universe()


def test_half_plane_partition_keeps_the_positive_mean():
    partition = half_plane_partition([[0.0, 0.0], [0.2, 0.2]], [[2.0, 2.0], [2.2, 2.2]])
    assert partition is not None
    assert partition.evaluate(np.array([0.1, 0.1])) is True
    assert partition.evaluate(np.array([2.1, 2.1])) is False


def test_half_plane_partition_returns_none_when_means_coincide():
    assert half_plane_partition([[1.0, 1.0]], [[1.0, 1.0]]) is None


def test_half_plane_partition_requires_both_classes():
    assert half_plane_partition([], [[1.0, 1.0]]) is None
    assert half_plane_partition([[1.0, 1.0]], []) is None
