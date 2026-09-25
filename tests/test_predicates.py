"""Behavioural tests for the SpectRL state predicates.

Covers each atomic region's membership including its boundary, the boolean
combinators' truth tables, canonical rendering, and the constructor validation
that makes malformed regions fail fast.
"""

from __future__ import annotations

import pytest

from faultline.spectrl.predicates import (
    And,
    Box,
    Disc,
    Empty,
    HalfPlane,
    Not,
    Or,
    Universe,
)

A = Box((0.0, 0.0), (2.0, 2.0))
B = Box((1.0, 1.0), (3.0, 3.0))
C = Box((5.0, 5.0), (6.0, 6.0))


@pytest.mark.parametrize(
    "point",
    [[1.0, 1.0], [0.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
)
def test_box_accepts_interior_and_boundary_points(point):
    """A box contains its interior and every point on its closed boundary."""
    assert A.evaluate(point) is True


@pytest.mark.parametrize(
    "point",
    [[2.0001, 1.0], [-0.1, 1.0], [1.0, 2.0001], [1.0, -0.001]],
)
def test_box_rejects_points_just_outside(point):
    """A box excludes points beyond any face by even a small margin."""
    assert A.evaluate(point) is False


def test_box_reads_only_its_dims():
    """A planar box ignores state coordinates outside its dims."""
    assert A.evaluate([1.0, 1.0, 999.0, -999.0]) is True


@pytest.mark.parametrize(
    "point,expected",
    [([0.0, 0.0], True), ([1.0, 0.0], True), ([1.1, 0.0], False), ([0.8, 0.8], False)],
)
def test_disc_membership(point, expected):
    """A disc contains exactly the points within its radius of the centre."""
    assert Disc((0.0, 0.0), 1.0).evaluate(point) is expected


@pytest.mark.parametrize(
    "point,expected",
    [([0.5, 99.0], True), ([0.0, 0.0], True), ([-0.5, 0.0], False)],
)
def test_halfplane_membership(point, expected):
    """A half-plane contains points whose projection reaches the offset."""
    assert HalfPlane((1.0, 0.0), 0.0).evaluate(point) is expected


def test_universe_and_empty_are_constant():
    """Universe accepts every state and Empty rejects every state."""
    assert Universe().evaluate([-123.0, 45.6]) is True
    assert Empty().evaluate([0.0, 0.0]) is False


def test_and_is_true_only_when_all_children_hold():
    """And over two overlapping boxes is true exactly on their intersection."""
    both = And(A, B)
    assert both.evaluate([1.5, 1.5]) is True
    assert both.evaluate([0.5, 0.5]) is False
    assert both.evaluate([2.5, 2.5]) is False


def test_or_is_true_when_any_child_holds():
    """Or over two disjoint boxes is true on either box."""
    either = Or(A, C)
    assert either.evaluate([1.0, 1.0]) is True
    assert either.evaluate([5.5, 5.5]) is True
    assert either.evaluate([3.0, 3.0]) is False


def test_not_inverts_the_child():
    """Not negates its child's verdict."""
    assert Not(A).evaluate([9.0, 9.0]) is True
    assert Not(A).evaluate([1.0, 1.0]) is False


def test_box_renders_canonically():
    """A box renders as a Rect literal with compact numbers."""
    assert str(Box((0.0, 0.0), (6.0, 6.0))) == "Rect(0, 0, 6, 6)"


def test_or_inside_and_is_parenthesised():
    """An Or nested under And is parenthesised to preserve precedence."""
    assert str(And(Or(A, C), B)) == "(Rect(0, 0, 2, 2) ∨ Rect(5, 5, 6, 6)) ∧ Rect(1, 1, 3, 3)"


def test_and_inside_or_is_not_parenthesised():
    """An And nested under Or needs no parentheses because And binds tighter."""
    assert str(Or(And(A, B), C)) == "Rect(0, 0, 2, 2) ∧ Rect(1, 1, 3, 3) ∨ Rect(5, 5, 6, 6)"


def test_box_with_inverted_bounds_raises():
    """Declaring a box whose lower bound exceeds its upper bound is rejected."""
    with pytest.raises(ValueError, match="exceeds"):
        Box((3.0, 3.0), (1.0, 1.0))


def test_box_with_mismatched_dims_raises():
    """A box bound count that disagrees with its dims count is rejected."""
    with pytest.raises(ValueError, match="bounds"):
        Box((0.0, 0.0), (1.0, 1.0), dims=(0, 1, 2))


def test_disc_with_negative_radius_raises():
    """A negative disc radius is rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        Disc((0.0, 0.0), -1.0)


def test_halfplane_with_zero_normal_raises():
    """A half-plane with a zero normal is rejected as ambiguous."""
    with pytest.raises(ValueError, match="non-zero"):
        HalfPlane((0.0, 0.0), 1.0)


def test_and_requires_at_least_two_children():
    """An And with fewer than two children is rejected."""
    with pytest.raises(ValueError, match="at least 2"):
        And(A)
