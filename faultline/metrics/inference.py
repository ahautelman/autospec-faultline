"""Paired inference without SciPy: bootstrap intervals, Wilcoxon signed-rank, Holm."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import product

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval of a statistic.

    Args:
        values: Observations (resample over independent instances, not over seeds of one).
        rng: Seeded generator.
        resamples: Number of resamples.
        confidence: Interval level.
        statistic: Function of a resample; the mean by default.

    How:
        Resamples with replacement and takes the percentile bounds.

    Returns:
        ``(low, high)``.

    Raises:
        ValueError: If ``values`` is empty.
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    draws = rng.integers(0, data.size, size=(resamples, data.size))
    stats_ = np.array([statistic(data[row]) for row in draws])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(stats_, alpha)), float(np.quantile(stats_, 1.0 - alpha))


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float], exact_limit: int = 20) -> float:
    """Return the two-sided p-value of the paired Wilcoxon signed-rank test.

    Args:
        a: First condition, one value per instance.
        b: Second condition, paired with ``a``.
        exact_limit: Largest number of non-zero pairs for the exact test.

    How:
        Drops zero differences, ranks the absolute differences (average ranks for ties)
        and takes the smaller signed-rank sum. Up to ``exact_limit`` pairs the p-value is
        exact by enumerating sign assignments; above it, the normal approximation with
        tie and continuity correction is used.

    Returns:
        The p-value; 1.0 when every pair ties.

    Raises:
        ValueError: If the sequences differ in length.
    """
    if len(a) != len(b):
        raise ValueError("Paired samples must have equal length")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[diff != 0.0]
    n = len(diff)
    if n == 0:
        return 1.0
    ranks = _ranks(np.abs(diff))
    w_plus = float(ranks[diff > 0].sum())
    statistic = min(w_plus, float(ranks.sum()) - w_plus)
    if n <= exact_limit:
        total = 0
        hits = 0
        for signs in product((0, 1), repeat=n):
            total += 1
            if min(s := float(np.dot(signs, ranks)), float(ranks.sum()) - s) <= statistic + 1e-9:
                hits += 1
        return min(1.0, hits / total)
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(diff), return_counts=True)
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - float((counts**3 - counts).sum()) / 48.0
    z = (abs(statistic - mean) - 0.5) / math.sqrt(variance)
    return float(min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2.0))))


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm-Bonferroni adjusted p-values.

    Args:
        p_values: Raw p-value per hypothesis name.

    How:
        Sorts ascending, multiplies the i-th smallest by ``m - i`` and enforces
        monotonicity, capping at 1.

    Returns:
        Adjusted p-values by name.
    """
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted
