"""Clopper-Pearson values from the FaultLine spec (section 8)."""

from __future__ import annotations

import pytest

from faultline import stats


def test_two_sided_interval_for_zero_of_ten():
    low, high = stats.two_sided(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.3085, abs=1e-4)


def test_two_sided_interval_for_five_of_ten():
    low, high = stats.two_sided(5, 10)
    assert low == pytest.approx(0.187, abs=1e-3)
    assert high == pytest.approx(0.813, abs=1e-3)


def test_one_sided_lower_bound_for_ten_of_ten_is_the_closed_form():
    assert stats.one_sided_lower(10, 10) == pytest.approx(0.05 ** (1 / 10), abs=1e-9)


def test_bounds_bracket_the_observed_rate_and_tighten_with_more_trials():
    low, high = stats.two_sided(80, 100)
    wide = high - low
    low_big, high_big = stats.two_sided(800, 1000)
    assert low < 0.8 < high
    assert high_big - low_big < wide


def test_invalid_counts_are_rejected():
    with pytest.raises(ValueError):
        stats.one_sided_lower(11, 10)
