"""Convex-hull geometry for the refinement operators (paper section 4).

ReachRefine, AvoidRefine, and AddRefine all shrink a region to the convex hull of
a set of visited states. A hull is a :class:`ConvexHull` predicate, so it composes
with the rest of the predicate algebra (intersection, subtraction) and can label
a graph vertex directly.

Hull points are planar by default (coordinates 0 and 1), matching the 9-rooms
state ``(x, y, theta, d)``, but ``dims`` selects any coordinate subset.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from faultline.spectrl.predicates import PLANAR_DIMS, Box, Dims, Empty, Predicate, State, select

# Visited states are hull vertices, so the hull must accept points exactly on its
# boundary; a small epsilon absorbs float noise from the projection.
_HULL_TOLERANCE = 1e-9

# Two distinct points span a segment, not a polygon; a turn in the chain needs
# the two previous points to compare the new one against.
_SEGMENT_VERTICES = 2
_TURN_POINTS = 2


def convex_hull(points: Sequence[State], dims: Dims = PLANAR_DIMS) -> Predicate:
    """Return the convex hull of a set of states as a predicate.

    Args:
        points: State vectors to enclose; may be empty or contain duplicates.
        dims: State coordinates the hull is built in; defaults to ``(x, y)``.

    How:
        Projects and de-duplicates the points, then reduces them to the hull. No
        points give :class:`Empty` and a single point a degenerate :class:`Box`.
        Two or more points keep the dedicated :class:`ConvexHull` form: a segment
        when they are collinear, otherwise the counter-clockwise polygon from
        Andrew's monotone chain.

    Returns:
        A predicate whose satisfying states lie inside the hull.
    """
    projected = _distinct(points, dims)
    if projected.shape[0] == 0:
        return Empty()
    if projected.shape[0] == 1:
        point = projected[0]
        return Box(point, point, dims)
    if projected.shape[0] == _SEGMENT_VERTICES:
        return ConvexHull(projected, dims)
    return ConvexHull(_monotone_chain(projected), dims)


class ConvexHull(Predicate):
    """A convex point set: the point, segment, or polygon its vertices span.

    Args:
        vertices: Hull vertices in convex position, shape ``(k, len(dims))``.
        dims: State coordinates the hull is drawn in; defaults to ``(x, y)``.

    How:
        Stores the vertices projected onto ``dims``. Membership mirrors the three
        shapes the hull can take. A polygon produced by the monotone chain is
        counter-clockwise, so a point is inside when it lies on the left of every
        directed edge.
    """

    def __init__(self, vertices: Sequence[Sequence[float]], dims: Dims = PLANAR_DIMS) -> None:
        self.vertices = np.asarray(vertices, dtype=float).reshape(-1, len(dims))
        self.dims = tuple(dims)
        if self.vertices.shape[0] == 0:
            raise ValueError("ConvexHull needs at least one vertex")

    def evaluate(self, state: State) -> bool:
        point = select(state, self.dims)
        count = self.vertices.shape[0]
        if count == 1:
            return bool(np.linalg.norm(point - self.vertices[0]) <= _HULL_TOLERANCE)
        if count == _SEGMENT_VERTICES:
            distance = _point_segment_distance(point, self.vertices[0], self.vertices[1])
            return bool(distance <= _HULL_TOLERANCE)
        return all(
            _cross(self.vertices[index], self.vertices[(index + 1) % count], point)
            >= -_HULL_TOLERANCE
            for index in range(count)
        )

    def to_string(self) -> str:
        vertices = ", ".join(str(tuple(vertex.tolist())) for vertex in self.vertices)
        return f"Hull({vertices})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ConvexHull)
            and self.dims == other.dims
            and np.array_equal(self.vertices, other.vertices)
        )

    def __hash__(self) -> int:
        return hash(("ConvexHull", self.dims, tuple(map(tuple, self.vertices))))


def _distinct(points: Sequence[State], dims: Dims) -> np.ndarray:
    """Project points to ``dims`` and drop duplicates.

    Args:
        points: State vectors to project.
        dims: State coordinates to keep.

    How:
        Builds a ``(len(points), len(dims))`` array and removes duplicate rows
        with :func:`numpy.unique`, which also sorts them into a canonical order.

    Returns:
        The distinct projected points, shape ``(n, len(dims))``.
    """
    projected = np.array([select(point, dims) for point in points], dtype=float)
    return projected.reshape(-1, len(dims)) if projected.size else np.zeros((0, len(dims)))


def _cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """Return the 2-D cross product of ``first - origin`` and ``second - origin``.

    Args:
        origin: Shared base point of the two vectors.
        first: End point of the first vector.
        second: End point of the second vector.

    How:
        Computes the scalar ``z`` component of the 3-D cross product, whose sign
        gives the turn direction (positive is counter-clockwise).

    Returns:
        The signed area of the parallelogram, positive when turning left.
    """
    return float(
        (first[0] - origin[0]) * (second[1] - origin[1])
        - (first[1] - origin[1]) * (second[0] - origin[0])
    )


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Return the distance from a point to a closed line segment.

    Args:
        point: The query point.
        start: Segment start.
        end: Segment end.

    How:
        Projects the point onto the segment's infinite line, clamps the
        projection parameter to ``[0, 1]`` so the nearest point stays on the
        segment, and measures the Euclidean distance to it. A zero-length
        segment falls back to the point distance.

    Returns:
        The non-negative distance to the segment.
    """
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))
    parameter = float(np.clip((point - start) @ segment / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + parameter * segment)))


def _monotone_chain(points: np.ndarray) -> np.ndarray:
    """Return the vertices of the convex hull in counter-clockwise order.

    Args:
        points: Distinct projected points, shape ``(n, len(dims))`` with ``n >= 3``.

    How:
        Sorts the points and builds the lower then the upper hull, popping a
        vertex whenever the last turn is not strictly left. Collinear points are
        dropped, so a fully collinear input reduces to two vertices.

    Returns:
        The hull vertices, shape ``(k, len(dims))`` in counter-clockwise order.
    """
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]
    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= _TURN_POINTS and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(ordered):
        while len(upper) >= _TURN_POINTS and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1], dtype=float)
