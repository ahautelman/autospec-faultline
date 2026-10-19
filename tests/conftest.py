"""Shared pytest fixtures."""

from __future__ import annotations

import sys

import pytest


def pytest_addoption(parser):
    """Register the manual-review switch for the n-rooms figure tests.

    How:
        ``--save-figures`` copies the generated layout figures into
        ``reports/rooms100/layouts/`` so a human can look at them. Without the
        flag the tests only write into pytest's temporary directory.
        ``--figures-count`` limits how many random worlds the contact sheets
        draw; the generator contract is still asserted on every seed.
    """
    parser.addoption(
        "--save-figures",
        action="store_true",
        default=False,
        help="Copy n-rooms layout figures into reports/rooms100/layouts/ for review",
    )
    parser.addoption(
        "--figures-count",
        type=int,
        default=None,
        help="Draw only this many random n-rooms worlds on the contact sheets (default: all)",
    )


@pytest.fixture(autouse=True)
def close_figures():
    """Close every matplotlib figure a test opened, so figures do not pile up.

    How:
        Runs after each test and closes all figures, but only when the test imported
        pyplot, so it costs nothing for tests that never plot.
    """
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")
