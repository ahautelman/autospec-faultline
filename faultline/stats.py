"""Exact Clopper-Pearson binomial bounds, with no SciPy dependency.

The bounds are found by bisection on the exact binomial tail, so they match the
beta-quantile definition to well below the precision the gates need.
"""

from __future__ import annotations

from math import exp, lgamma, log

_ITERATIONS = 80


def _log_pmf(k: int, n: int, p: float) -> float:
    """Return ``log P(X = k)`` for ``X ~ Binomial(n, p)`` with ``0 < p < 1``."""
    return (
        lgamma(n + 1)
        - lgamma(k + 1)
        - lgamma(n - k + 1)
        + k * log(p)
        + (n - k) * log(1.0 - p)
    )


def _upper_tail(k: int, n: int, p: float) -> float:
    """Return ``P(X >= k)``."""
    return min(1.0, sum(exp(_log_pmf(i, n, p)) for i in range(k, n + 1)))


def _lower_tail(k: int, n: int, p: float) -> float:
    """Return ``P(X <= k)``."""
    return min(1.0, sum(exp(_log_pmf(i, n, p)) for i in range(0, k + 1)))


def _check(successes: int, trials: int) -> None:
    """Raise if the counts are not a valid binomial outcome."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError(f"Need 0 <= successes <= trials and trials >= 1, got {successes}/{trials}")


def lower_bound(successes: int, trials: int, alpha: float) -> float:
    """Return the Clopper-Pearson lower bound at tail probability ``alpha``.

    Args:
        successes: Observed successes.
        trials: Number of trials.
        alpha: Probability mass in the lower tail.

    How:
        Finds the ``p`` with ``P(X >= successes; p) = alpha`` by bisection; the
        tail is increasing in ``p``. Zero successes give ``0``.

    Returns:
        The lower bound in ``[0, 1]``.
    """
    _check(successes, trials)
    if successes == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(_ITERATIONS):
        mid = (low + high) / 2.0
        if _upper_tail(successes, trials, mid) < alpha:
            low = mid
        else:
            high = mid
    return high


def upper_bound(successes: int, trials: int, alpha: float) -> float:
    """Return the Clopper-Pearson upper bound at tail probability ``alpha``.

    Args:
        successes: Observed successes.
        trials: Number of trials.
        alpha: Probability mass in the upper tail.

    How:
        Finds the ``p`` with ``P(X <= successes; p) = alpha`` by bisection; the
        tail is decreasing in ``p``. All successes give ``1``.

    Returns:
        The upper bound in ``[0, 1]``.
    """
    _check(successes, trials)
    if successes == trials:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(_ITERATIONS):
        mid = (low + high) / 2.0
        if _lower_tail(successes, trials, mid) > alpha:
            low = mid
        else:
            high = mid
    return low


def one_sided_lower(successes: int, trials: int, confidence: float = 0.95) -> float:
    """Return the one-sided lower confidence bound, used for gating.

    Args:
        successes: Observed successes.
        trials: Number of trials.
        confidence: Confidence level, for example ``0.95``.

    How:
        Puts the whole ``1 - confidence`` in the lower tail.

    Returns:
        The lower bound.
    """
    return lower_bound(successes, trials, 1.0 - confidence)


def two_sided(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the two-sided interval, used for reporting.

    Args:
        successes: Observed successes.
        trials: Number of trials.
        confidence: Confidence level, for example ``0.95``.

    How:
        Splits ``1 - confidence`` equally between the two tails.

    Returns:
        ``(lower, upper)``.
    """
    alpha = (1.0 - confidence) / 2.0
    return lower_bound(successes, trials, alpha), upper_bound(successes, trials, alpha)
