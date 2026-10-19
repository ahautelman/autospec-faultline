"""Comparison with the paper's reported success probabilities.

For each scenario, hatched bars give the paper's before and after numbers and solid bars
give this run's before and after at the paper's checkpoint, averaged over seeds. A
scenario without paper numbers keeps its own bars only, so any run can be drawn.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from faultline.experiments.records import REFERENCE_STEP, success_at
from faultline.viz.captions import reference_caption
from faultline.viz.figures.footer import add_footer
from faultline.viz.figures.output import save_with_caption
from faultline.viz.legend import BAR_HATCH
from faultline.viz.run import RunView
from faultline.viz.style import (
    DEFAULT_FORMATS,
    DOUBLE_COLUMN_WIDTH,
    GRID_COLOUR,
    ROLE_COLOURS,
    SINGLE_COLUMN_WIDTH,
    Role,
    pyplot,
)

BAR_WIDTH = 0.19
FIGURE_HEIGHT = 2.7
LABEL_WIDTH = 10
# Offsets in bar widths: paper before, paper after, run before, run after.
OFFSETS = (-1.5, -0.5, 0.5, 1.5)


def build_reference(
    groups: Mapping[str, Sequence[RunView]], *, embed_caption: bool = False
) -> tuple[Any, str]:
    """Build the paper-comparison bar chart.

    Args:
        groups: The runs of each scenario, keyed by scenario name.
        embed_caption: Whether to print the caption inside the figure.

    How:
        Draws four bars per scenario at the paper's reference step: the paper's
        before and after (hatched) and the run's mean before and after (solid), with
        a standard-deviation whisker when there are several seeds.

    Returns:
        The figure and its caption.

    Raises:
        ValueError: If no run carries the paper's numbers, so there is nothing to
            compare against.
    """
    if not any(_has_reference(run) for runs in groups.values() for run in runs):
        raise ValueError("No run carries the paper's reference numbers to compare against")
    plt = pyplot()
    width = SINGLE_COLUMN_WIDTH if len(groups) <= 2 else DOUBLE_COLUMN_WIDTH
    figure, ax = plt.subplots(figsize=(width, FIGURE_HEIGHT), layout="constrained")
    roles = {Role.BEFORE, Role.AFTER}
    for position, runs in enumerate(groups.values()):
        roles |= _draw_group(ax, position, runs)
    _format(ax, list(groups))
    caption = reference_caption(max(len(runs) for runs in groups.values()))
    add_footer(figure, roles, caption if embed_caption else None, width, FIGURE_HEIGHT)
    return figure, caption


def write_reference(
    groups: Mapping[str, Sequence[RunView]],
    output: Path,
    *,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    """Draw the paper comparison and save it with a caption file.

    Args:
        groups: The runs of each scenario, keyed by scenario name.
        output: Directory to write into.
        formats: Image formats to write.

    How:
        Names the files ``reference_comparison`` and writes the caption beside them.

    Returns:
        The image paths, then the caption path.
    """
    figure, caption = build_reference(groups)
    return save_with_caption(figure, caption, output / "reference_comparison", formats)


def _has_reference(run: RunView) -> bool:
    """Return whether a run carries both of the paper's numbers."""
    return run.reference_before is not None and run.reference_after is not None


def _draw_group(ax: Any, position: int, runs: Sequence[RunView]) -> set[Role]:
    """Draw one scenario's paper and run bars; return the roles drawn."""
    roles: set[Role] = set()
    first = runs[0]
    measured = (
        _measure([r.before_curve for r in runs]),
        _measure([r.after_curve for r in runs]),
    )
    colours = (ROLE_COLOURS[Role.BEFORE], ROLE_COLOURS[Role.AFTER])
    if _has_reference(first):
        for offset, value, colour in zip(OFFSETS[:2], (first.reference_before, first.reference_after), colours, strict=True):
            ax.bar(position + offset * BAR_WIDTH, value, BAR_WIDTH, facecolor="white",
                   edgecolor=colour, hatch=BAR_HATCH, linewidth=0.8)
        roles.add(Role.REFERENCE_BAR)
    for offset, (mean, spread), colour in zip(OFFSETS[2:], measured, colours, strict=True):
        # A probability cannot leave [0, 1], so the whisker stops at the bounds.
        whisker = [[min(spread, mean)], [min(spread, 1.0 - mean)]] if len(runs) > 1 else None
        ax.bar(position + offset * BAR_WIDTH, mean, BAR_WIDTH, color=colour,
               yerr=whisker, capsize=2, error_kw={"linewidth": 0.8})
    return roles


def _measure(curves: Sequence[Any]) -> tuple[float, float]:
    """Return the mean and standard deviation of the curves at the reference step."""
    values = [success_at(curve, REFERENCE_STEP) for curve in curves if curve]
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _format(ax: Any, names: list[str]) -> None:
    """Label scenarios, fix the y range and add a light horizontal grid."""
    ax.set_xticks(range(len(names)), [textwrap.fill(n.replace("_", " "), LABEL_WIDTH) for n in names])
    ax.set_ylabel(f"Success probability at {REFERENCE_STEP // 1000}k steps")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
