"""Half-plane partitioning for PastRefine (paper section 4.3).

PastRefine separates the start states of successful trajectories from those of
failed ones with a single hyperplane, then keeps the successful side. This module
learns that hyperplane with class-balanced logistic regression, the classifier the
authors' released code fits (``refinement/utils.py``). The successful side is
``normal . s >= offset``.

Logistic regression is fit here with a few Newton steps in numpy, so the core
package does not need scikit-learn. A small ridge penalty keeps the weights
finite when the two classes are linearly separable, where the unpenalised fit
would diverge; the boundary then sits close to the max-margin separator. When the
fit finds no direction at all (the classes coincide) the function reports failure
rather than returning an arbitrary cut, and PastRefine declines to refine. The
caller restricts the result to the original source region, so even an imperfect
cut stays sound.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from faultline.spectrl.predicates import PLANAR_DIMS, Dims, HalfPlane, State, select

# Ridge penalty on the standardised weights: small enough that separable classes
# get a near max-margin boundary, large enough to keep Newton's method finite.
_RIDGE = 1e-4
_NEWTON_STEPS = 100
# Stop once a Newton step moves the weights by less than this.
_STEP_TOLERANCE = 1e-10
# A fitted normal shorter than this has found no separating direction.
_MIN_NORMAL = 1e-9


def half_plane_partition(
    positive: Sequence[State], negative: Sequence[State], dims: Dims = PLANAR_DIMS
) -> HalfPlane | None:
    """Learn the half-plane that keeps the positive states and cuts the negative ones.

    Args:
        positive: State vectors to keep, normally the successful starts.
        negative: State vectors to cut, normally the failed starts.
        dims: State coordinates the hyperplane is fit in; defaults to ``(x, y)``.

    How:
        Projects both sets, standardises the coordinates so the ridge penalty
        treats every axis alike, and fits class-balanced logistic regression
        (:func:`_fit_logistic`). The standardised boundary is mapped back to
        state coordinates, oriented so that the positive class is the interior.

    Returns:
        The half-plane containing the positive side, or ``None`` when either set
        is empty or the fit finds no separating direction.
    """
    positive_points = _project(positive, dims)
    negative_points = _project(negative, dims)
    if positive_points.shape[0] == 0 or negative_points.shape[0] == 0:
        return None
    points = np.vstack([positive_points, negative_points])
    labels = np.concatenate([np.ones(len(positive_points)), np.zeros(len(negative_points))])
    center = points.mean(axis=0)
    scale = points.std(axis=0)
    scale[scale == 0.0] = 1.0
    weights, bias = _fit_logistic((points - center) / scale, labels)
    if float(np.linalg.norm(weights)) < _MIN_NORMAL:
        return None
    # weights . (s - center) / scale + bias >= 0, rewritten as normal . s >= offset.
    normal = weights / scale
    return HalfPlane(normal, float(normal @ center - bias), dims)


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit class-balanced, ridge-penalised logistic regression by Newton's method.

    Args:
        features: Standardised points, shape ``(n, d)``.
        labels: ``1`` for the positive class and ``0`` for the negative, shape ``(n,)``.

    How:
        Weights each sample by ``n / (2 * class size)`` (scikit-learn's
        ``class_weight="balanced"``), so a rare failure class still moves the
        boundary. Minimises the weighted log-loss plus a ridge on the weights (not
        the bias) with Newton steps; the penalty makes the problem strictly convex,
        so the iteration converges.

    Returns:
        The weight vector, shape ``(d,)``, and the bias.
    """
    count, dimension = features.shape
    positives = labels.sum()
    sample_weight = np.where(labels == 1.0, count / (2 * positives), count / (2 * (count - positives)))
    design = np.hstack([features, np.ones((count, 1))])
    penalty = np.diag([_RIDGE] * dimension + [0.0])
    theta = np.zeros(dimension + 1)
    for _ in range(_NEWTON_STEPS):
        probability = 1.0 / (1.0 + np.exp(-design @ theta))
        gradient = design.T @ (sample_weight * (probability - labels)) + penalty @ theta
        curvature = sample_weight * probability * (1.0 - probability)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        theta -= step
        if float(np.linalg.norm(step)) < _STEP_TOLERANCE:
            break
    return theta[:dimension], float(theta[dimension])


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
