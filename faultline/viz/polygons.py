"""Exact polygon form of region predicates, with no plotting dependency.

Boxes, convex hulls, half-planes and discs are convex, so a region built from them with
``and`` / ``or`` is a union of convex polygons that can be drawn as crisp vector shapes.
A region with a complement (``not``) is not covered here and falls back to a raster in
:mod:`faultline.viz.geometry`. Only the planar ``(x, y)`` coordinates are supported.
"""

from __future__ import annotations

import numpy as np

from faultline.geometry import ConvexHull
from faultline.spectrl.predicates import (
    PLANAR_DIMS,
    And,
    Box,
    Disc,
    Empty,
    HalfPlane,
    Or,
    Predicate,
    Universe,
)

DISC_SIDES = 96
MIN_AREA = 1e-9

Polygons = list[np.ndarray]


def region_polygons(predicate: Predicate, world: float) -> Polygons | None:
    """Return a region as convex polygons, or ``None`` if it needs a raster.

    Args:
        predicate: The region to convert.
        world: Side length of the square world, which bounds unbounded regions.

    How:
        Converts each atom to a counter-clockwise polygon, unions ``or`` children by
        concatenation, and intersects ``and`` children pairwise by clipping. A
        predicate containing a negation, or on other coordinates, gives ``None``.

    Returns:
        Counter-clockwise ``(k, 2)`` polygons whose union is the region; possibly
        empty for an empty region.
    """
    if isinstance(predicate, Empty):
        return []
    if isinstance(predicate, Universe):
        return [_rectangle((0.0, 0.0), (world, world))]
    if isinstance(predicate, (Box, Disc, HalfPlane, ConvexHull)):
        return _atom_polygons(predicate, world)
    if isinstance(predicate, Or):
        return _union([region_polygons(child, world) for child in predicate.predicates])
    if isinstance(predicate, And):
        return _intersection([region_polygons(child, world) for child in predicate.predicates])
    return None


def polygon_area(polygon: np.ndarray) -> float:
    """Return a polygon's area by the shoelace formula.

    Args:
        polygon: ``(k, 2)`` vertices in order.

    How:
        Sums the cross products of consecutive vertices.

    Returns:
        The absolute area.
    """
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clip_half_plane(polygon: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Clip a polygon to the half-plane ``normal . p >= offset``.

    Args:
        polygon: ``(k, 2)`` vertices of a convex polygon.
        normal: Half-plane normal, shape ``(2,)``.
        offset: Half-plane offset along the normal.

    How:
        One Sutherland-Hodgman pass: keeps vertices on the inside and adds the
        crossing point wherever an edge changes side.

    Returns:
        The clipped polygon, with no rows when nothing remains.
    """
    inside = polygon @ normal - offset
    kept: list[np.ndarray] = []
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        here, there = inside[index], inside[(index + 1) % len(polygon)]
        if here >= 0:
            kept.append(current)
        if (here >= 0) != (there >= 0):
            kept.append(current + (following - current) * (here / (here - there)))
    return np.array(kept).reshape(-1, 2)


def _atom_polygons(atom: Predicate, world: float) -> Polygons | None:
    """Convert one convex atom to polygons, or ``None`` off the planar dims."""
    if atom.dims != PLANAR_DIMS:
        return None
    if isinstance(atom, Box):
        return [_rectangle(atom.lower, atom.upper)]
    if isinstance(atom, Disc):
        angles = np.linspace(0.0, 2.0 * np.pi, DISC_SIDES, endpoint=False)
        ring = atom.center + atom.radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return [ring]
    if isinstance(atom, HalfPlane):
        clipped = clip_half_plane(_rectangle((0.0, 0.0), (world, world)), atom.normal, atom.offset)
        return [clipped] if _has_area(clipped) else []
    return [atom.vertices] if len(atom.vertices) >= 3 else []


def _rectangle(lower: tuple[float, float] | np.ndarray, upper: tuple[float, float] | np.ndarray) -> np.ndarray:
    """Return the counter-clockwise rectangle between two corners."""
    (x0, y0), (x1, y1) = lower, upper
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _union(parts: list[Polygons | None]) -> Polygons | None:
    """Concatenate the pieces of a union, or ``None`` if any piece is unsupported."""
    if any(part is None for part in parts):
        return None
    return [polygon for part in parts for polygon in part]


def _intersection(parts: list[Polygons | None]) -> Polygons | None:
    """Intersect unions of convex polygons pairwise, or ``None`` if unsupported."""
    if any(part is None for part in parts):
        return None
    result = parts[0]
    for part in parts[1:]:
        result = [
            clipped
            for left in result
            for right in part
            if _has_area(clipped := _clip_convex(left, right))
        ]
    return result


def _clip_convex(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Clip a convex polygon by another counter-clockwise convex polygon."""
    clipped = subject
    for index, start in enumerate(clipper):
        edge = clipper[(index + 1) % len(clipper)] - start
        normal = np.array([-edge[1], edge[0]])
        clipped = clip_half_plane(clipped, normal, float(normal @ start))
        if len(clipped) == 0:
            break
    return clipped


def _has_area(polygon: np.ndarray) -> bool:
    """Return whether a polygon has a non-degenerate area."""
    return len(polygon) >= 3 and polygon_area(polygon) > MIN_AREA
