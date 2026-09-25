"""Half-plane partitioning for PastRefine (paper section 4.3).

PastRefine separates the start states of successful trajectories from those of
failed ones with a single hyperplane, then keeps the successful side. This module
fits that hyperplane. The fit uses the difference of the two class means as the
normal, which is the perpendicular-bisector direction of the means: the normal
points from the failures toward the successes, so the half-plane
``normal . s >= offset`` holds the successful side.

The fit is only exact when the two means differ. When they coincide there is no
separating direction, so the function reports failure rather than returning an
arbitrary cut; PastRefine then declines to refine. The result is restricted to
the original source region by the caller, so even an imperfect cut stays sound.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from faultline.spectrl.predicates import PLANAR_DIMS, Dims, HalfPlane, State, select


def half_plane_partition(
    positive: Sequence[State], negative: Sequence[State], dims: Dims = PLANAR_DIMS
) -> HalfPlane | None:
    """Fit the half-plane that keeps the positive states and cuts the negative ones.

    Args:
        positive: State vectors to keep, normally the successful starts.
        negative: State vectors to cut, normally the failed starts.
        dims: State coordinates the hyperplane is fit in; defaults to ``(x, y)``.

    How:
        Projects both sets, takes the difference of their means as the normal, and
        places the boundary halfway between the means. The sign convention makes
        the positive mean satisfy the returned half-plane, so the successful side
        is always the interior.

    Returns:
        The half-plane containing the positive states, or ``None`` when either set
        is empty or the means coincide, so no separating direction exists.
    """
    positive_points = _project(positive, dims)
    negative_points = _project(negative, dims)
    if positive_points.shape[0] == 0 or negative_points.shape[0] == 0:
        return None
    positive_mean = positive_points.mean(axis=0)
    negative_mean = negative_points.mean(axis=0)
    normal = positive_mean - negative_mean
    if float(np.linalg.norm(normal)) == 0.0:
        return None
    midpoint = (positive_mean + negative_mean) / 2.0
    return HalfPlane(normal, float(normal @ midpoint), dims)


def _project(points: Sequence[State], dims: Dims) -> np.ndarray:
    """Project state vectors onto ``dims``.

    Args:
        points: State vectors to project; may be empty.
        dims: State coordinates to keep.

    How:
        Builds a ``(len(points), len(dims))`` float array from the selected
        coordinates, reshaped so an empty input still has the right shape.

    Returns:
        The projected points, shape ``(len(points), len(dims))``.
    """
    projected = np.array([select(point, dims) for point in points], dtype=float)
    return projected.reshape(-1, len(dims)) if projected.size else np.zeros((0, len(dims)))
