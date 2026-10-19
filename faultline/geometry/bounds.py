"""Axis-aligned bounds for region predicates, used to sample inside a region.

The environment samples a goal point from a vertex's region by rejection sampling.
Refined regions are composite predicates (an intersection of a box with a convex
hull, or a box minus a hull), so sampling them over the whole world is wasteful and
can fail for a small region. This module computes a sound axis-aligned bounding box
for a region when one is known, so the sampler can draw inside the box instead of
the whole world.

A bound is a superset of the region, never a subset, so sampling inside it stays
correct. An unbounded or unrecognised region reports ``None`` and the caller falls
back to sampling the whole world.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from faultline.geometry.convex_hull import ConvexHull
from faultline.spectrl.predicates import And, Box, Dims, Disc, Or, Predicate

Bounds = tuple[np.ndarray, np.ndarray]

# Rejection-sampling budget shared by the environment's start and goal sampler and
# the refinement operators' check that a refined region can be sampled at all.
SAMPLE_ATTEMPTS = 2000


def region_bounds(predicate: Predicate, dims: Dims) -> Bounds | None:
    """Return the axis-aligned bounds of a region along ``dims``.

    Args:
        predicate: The region predicate to bound.
        dims: State coordinates the bounds are expressed in.

    How:
        Dispatches on the predicate shape. A box, disc, or convex hull reports its
        own extent when it reads exactly ``dims``. An intersection intersects its
        children's known bounds, because the intersection is a subset of each
        child. A union needs every child bounded. Anything else (a half-plane, a
        negation, the universe, the empty set) is unbounded along ``dims`` and
        reports ``None``.

    Returns:
        A ``(lower, upper)`` pair of arrays, one entry per coordinate in ``dims``,
        or ``None`` when the region has no finite bound along ``dims``.
    """
    if isinstance(predicate, Box):
        return _box_bounds(predicate, dims)
    if isinstance(predicate, Disc):
        return _disc_bounds(predicate, dims)
    if isinstance(predicate, ConvexHull):
        return _hull_bounds(predicate, dims)
    if isinstance(predicate, And):
        return _intersection_bounds([region_bounds(child, dims) for child in predicate.predicates])
    if isinstance(predicate, Or):
        return _union_bounds([region_bounds(child, dims) for child in predicate.predicates])
    return None


def _box_bounds(predicate: Box, dims: Dims) -> Bounds | None:
    """Return a box's exact extent when it reads exactly ``dims``.

    Args:
        predicate: The box to bound.
        dims: State coordinates the bounds are expressed in.

    How:
        A box stores its extent over its own coordinates, so the bounds transfer
        only when those coordinates are exactly the requested ones; a mismatch
        reports no bound rather than a wrongly indexed one.

    Returns:
        The box's lower and upper corners, or ``None`` when the coordinates differ.
    """
    if predicate.dims != tuple(dims):
        return None
    return predicate.lower.copy(), predicate.upper.copy()


def _disc_bounds(predicate: Disc, dims: Dims) -> Bounds | None:
    """Return the square enclosing a disc when it reads exactly ``dims``.

    Args:
        predicate: The disc to bound.
        dims: State coordinates the bounds are expressed in.

    How:
        The enclosing square is the centre plus or minus the radius on every
        coordinate; it transfers only when the disc reads exactly ``dims``.

    Returns:
        The enclosing square's corners, or ``None`` when the coordinates differ.
    """
    if predicate.dims != tuple(dims):
        return None
    return predicate.center - predicate.radius, predicate.center + predicate.radius


def _hull_bounds(predicate: ConvexHull, dims: Dims) -> Bounds | None:
    """Return the bounding box of a hull's vertices when it reads exactly ``dims``.

    Args:
        predicate: The convex hull to bound.
        dims: State coordinates the bounds are expressed in.

    How:
        The hull's extent is the per-coordinate minimum and maximum of its
        vertices; it transfers only when the hull reads exactly ``dims``.

    Returns:
        The vertex bounding box corners, or ``None`` when the coordinates differ.
    """
    if predicate.dims != tuple(dims):
        return None
    return predicate.vertices.min(axis=0), predicate.vertices.max(axis=0)


def _intersection_bounds(candidates: Sequence[Bounds | None]) -> Bounds | None:
    """Intersect the known child bounds, ignoring unbounded children.

    Args:
        candidates: One bound per child, or ``None`` for an unbounded child.

    How:
        An intersection is contained in each child, so intersecting only the
        bounded children still contains the intersection; an unbounded child is
        skipped rather than making the whole bound unknown.

    Returns:
        The tightest known bounding box, or ``None`` when no child is bounded.
    """
    known = [bound for bound in candidates if bound is not None]
    if not known:
        return None
    lower = np.max([bound[0] for bound in known], axis=0)
    upper = np.min([bound[1] for bound in known], axis=0)
    return lower, upper


def _union_bounds(candidates: Sequence[Bounds | None]) -> Bounds | None:
    """Union the child bounds, requiring every child to be bounded.

    Args:
        candidates: One bound per child, or ``None`` for an unbounded child.

    How:
        A union reaching every child needs each child's extent; a single unbounded
        child makes the union unbounded, so it reports ``None``.

    Returns:
        The enclosing bounding box, or ``None`` when any child is unbounded.
    """
    if any(bound is None for bound in candidates):
        return None
    lower = np.min([bound[0] for bound in candidates], axis=0)
    upper = np.max([bound[1] for bound in candidates], axis=0)
    return lower, upper


def sample_within(
    region: Predicate, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator
) -> np.ndarray | None:
    """Rejection-sample one point of a region inside an axis-aligned box.

    Args:
        region: The predicate the point must satisfy.
        lower: Lower corner of the box to draw from, one entry per coordinate.
        upper: Upper corner of the box to draw from.
        rng: The random generator to draw with, so the caller controls seeding.

    How:
        Draws up to :data:`SAMPLE_ATTEMPTS` uniform points in the box and returns
        the first one inside ``region``. A region with no area inside the box (a
        segment of a degenerate hull, or an empty intersection) is never hit, so
        the caller learns it cannot be sampled instead of looping forever.

    Returns:
        A point inside ``region``, or ``None`` when no draw landed in it.
    """
    for _ in range(SAMPLE_ATTEMPTS):
        point = rng.uniform(lower, upper)
        if region.evaluate(point):
            return point
    return None

