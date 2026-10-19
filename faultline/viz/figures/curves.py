"""Learning curves: best-path success against training steps, before and after refinement.

One scenario's seeds are drawn on one axes: a thick mean line with the individual seeds
faint behind it and a minimum-to-maximum band, or a single line for one seed. The loop's
success threshold and the paper's reported values are added when the runs carry them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from faultline.experiments.records import REFERENCE_STEP, Checkpoint
from faultline.viz.captions import curves_caption
from faultline.viz.figures.grid import build_panels
from faultline.viz.run import RunView, group_by_scenario
from faultline.viz.figures.output import save_with_caption
from faultline.viz.style import DEFAULT_FORMATS, GRID_COLOUR, ROLE_COLOURS, Role

MEAN_WIDTH = 1.4
SEED_WIDTH = 0.6
SEED_ALPHA = 0.3
BAND_ALPHA = 0.15
REFERENCE_SIZE = 18.0
# Few enough x ticks that the labels stay apart on a narrow grid panel.
X_TICK_BINS = 4


def draw_curves(ax: Any, runs: Sequence[RunView]) -> set[Role]:
    """Draw one scenario's before and after curves on an axes.

    Args:
        ax: Target axes.
        runs: The scenario's runs, one per seed.

    How:
        Draws the before and the after series, then the threshold line and the paper
        reference markers when the runs have them, and formats the axes.

    Returns:
        The roles drawn, for the legend.
    """
    roles: set[Role] = set()
    for role, curves in ((Role.BEFORE, [r.before_curve for r in runs]), (Role.AFTER, [r.after_curve for r in runs])):
        curves = [curve for curve in curves if curve]
        if curves:
            _draw_series(ax, curves, ROLE_COLOURS[role])
            roles.add(role)
    roles |= _draw_threshold(ax, runs)
    roles |= _draw_reference(ax, runs)
    _format(ax)
    return roles


def build_curves(runs: Sequence[RunView], *, embed_caption: bool = False) -> tuple[Any, str]:
    """Build the learning-curve figure for one scenario.

    Args:
        runs: The scenario's runs, one per seed.
        embed_caption: Whether to print the caption inside the figure.

    How:
        Delegates to the shared panel builder with a single group.

    Returns:
        The figure and its caption.
    """
    return build_curves_grid(group_by_scenario(runs), embed_caption=embed_caption)


def build_curves_grid(
    groups: Mapping[str, Sequence[RunView]], *, embed_caption: bool = False
) -> tuple[Any, str]:
    """Build one curve panel per scenario.

    Args:
        groups: The runs of each scenario, keyed by scenario name.
        embed_caption: Whether to print the caption inside the figure.

    How:
        Builds the caption from the most seeds, and whether any run has a threshold or
        reference, then draws each scenario on its own panel.

    Returns:
        The figure and its caption.
    """
    runs = [run for group in groups.values() for run in group]
    caption = curves_caption(
        max(len(group) for group in groups.values()),
        has_threshold=any(run.threshold is not None for run in runs),
        has_reference=any(_reference_visible(run) for run in runs),
    )
    return build_panels(groups, draw_curves, caption, embed_caption=embed_caption), caption


def write_curves(
    runs: Sequence[RunView], output: Path, *, formats: tuple[str, ...] = DEFAULT_FORMATS
) -> list[Path]:
    """Draw one scenario's curves and save them with a caption file.

    Args:
        runs: The scenario's runs, one per seed.
        output: Directory to write into.
        formats: Image formats to write.

    How:
        Names the files ``curves_<scenario>`` and writes the caption beside them.

    Returns:
        The image paths, then the caption path.
    """
    figure, caption = build_curves(runs)
    return save_with_caption(figure, caption, output / f"curves_{runs[0].scenario}", formats)


def _draw_series(ax: Any, curves: list[tuple[Checkpoint, ...]], colour: str) -> None:
    """Draw seed lines, the min-max band and the mean, or one line for one seed."""
    steps, mean, low, high = _statistics(curves)
    if len(curves) == 1:
        ax.plot(steps, mean, color=colour, linewidth=MEAN_WIDTH)
        return
    for curve in curves:
        ax.plot([c.step for c in curve], [c.probability for c in curve], color=colour,
                linewidth=SEED_WIDTH, alpha=SEED_ALPHA)
    ax.fill_between(steps, low, high, color=colour, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(steps, mean, color=colour, linewidth=MEAN_WIDTH)


def _statistics(curves: list[tuple[Checkpoint, ...]]) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Return each step with the mean, minimum and maximum over the seeds that have it."""
    by_step: dict[int, list[float]] = {}
    for curve in curves:
        for point in curve:
            by_step.setdefault(point.step, []).append(point.probability)
    steps = sorted(by_step)
    values = [by_step[step] for step in steps]
    return (
        steps,
        np.array([np.mean(v) for v in values]),
        np.array([np.min(v) for v in values]),
        np.array([np.max(v) for v in values]),
    )


def _draw_threshold(ax: Any, runs: Sequence[RunView]) -> set[Role]:
    """Draw the success threshold line if any run recorded one."""
    thresholds = {run.threshold for run in runs if run.threshold is not None}
    for value in thresholds:
        ax.axhline(value, color=ROLE_COLOURS[Role.THRESHOLD], linestyle=":", linewidth=1.0, zorder=1)
    return {Role.THRESHOLD} if thresholds else set()


def _reference_visible(run: RunView) -> bool:
    """Return whether the paper's reference step lies within the run's curves."""
    steps = [point.step for point in (*run.before_curve, *run.after_curve)]
    has_value = run.reference_before is not None or run.reference_after is not None
    return has_value and bool(steps) and min(steps) <= REFERENCE_STEP <= max(steps)


def _draw_reference(ax: Any, runs: Sequence[RunView]) -> set[Role]:
    """Mark the paper's reported before and after values at the reference step."""
    run = next((r for r in runs if _reference_visible(r)), None)
    if run is None:
        return set()
    for value, role in ((run.reference_before, Role.BEFORE), (run.reference_after, Role.AFTER)):
        if value is not None:
            ax.scatter([REFERENCE_STEP], [value], marker="D", s=REFERENCE_SIZE, facecolor="white",
                       edgecolor=ROLE_COLOURS[role], linewidth=1.0, zorder=5, clip_on=False)
    return {Role.REFERENCE}


def _format(ax: Any) -> None:
    """Label the axes, show steps in thousands and add a light horizontal grid."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    ax.set_xlabel("Training steps per edge")
    ax.set_ylabel("Success probability")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=X_TICK_BINS, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
