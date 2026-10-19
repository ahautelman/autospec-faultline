"""Per-seed spread: each seed's success before and after refinement.

Every seed is a dot, a grey line joins a seed's before and after values, and a bar marks
the mean with a whisker of one standard deviation. Two checkpoints are shown: the paper's
reference step and the final budget (just the budget when it does not pass the reference).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from faultline.experiments.records import REFERENCE_STEP, success_at
from faultline.viz.captions import seeds_caption
from faultline.viz.figures.grid import build_panels
from faultline.viz.run import RunView, group_by_scenario
from faultline.viz.figures.output import save_with_caption
from faultline.viz.style import DEFAULT_FORMATS, GRID_COLOUR, ROLE_COLOURS, Role

GROUP_GAP = 0.8
PAIR_COLOUR = "#b3b3b3"
DOT_SIZE = 10.0
MEAN_HALF_WIDTH = 0.28
# Axes-fraction offset that puts the checkpoint labels below the before/after labels.
CHECKPOINT_AXIS_OFFSET = -0.1


def draw_seeds(ax: Any, runs: Sequence[RunView]) -> set[Role]:
    """Draw each seed's before and after values at the reported checkpoints.

    Args:
        ax: Target axes.
        runs: The scenario's runs, one per seed.

    How:
        For each checkpoint draws a before and an after column: grey pair lines first,
        then the seed dots, then the mean bar with its standard deviation.

    Returns:
        The roles drawn, for the legend.
    """
    ticks, centres, names = [], [], []
    for index, (steps, name) in enumerate(_checkpoints(runs)):
        left = index * (2 + GROUP_GAP)
        before = [_value(run.before_curve, steps) for run in runs]
        after = [_value(run.after_curve, steps) for run in runs]
        _draw_pairs(ax, left, before, after)
        ticks += [left, left + 1]
        centres.append(left + 0.5)
        names.append(name)
    ax.set_xticks(ticks, ["Before", "After"] * len(centres))
    _add_checkpoint_labels(ax, centres, names)
    _format(ax, ticks)
    return {Role.BEFORE, Role.AFTER}


def build_seeds(runs: Sequence[RunView], *, embed_caption: bool = False) -> tuple[Any, str]:
    """Build the per-seed figure for one scenario.

    Args:
        runs: The scenario's runs, one per seed.
        embed_caption: Whether to print the caption inside the figure.

    How:
        Delegates to the shared panel builder with a single group.

    Returns:
        The figure and its caption.
    """
    return build_seeds_grid(group_by_scenario(runs), embed_caption=embed_caption)


def build_seeds_grid(
    groups: Mapping[str, Sequence[RunView]], *, embed_caption: bool = False
) -> tuple[Any, str]:
    """Build one per-seed panel per scenario.

    Args:
        groups: The runs of each scenario, keyed by scenario name.
        embed_caption: Whether to print the caption inside the figure.

    How:
        Builds the caption from the most seeds and the checkpoints of the first run.

    Returns:
        The figure and its caption.
    """
    first = next(iter(groups.values()))
    names = " and ".join(name for _, name in _checkpoints(first))
    caption = seeds_caption(max(len(group) for group in groups.values()), names)
    return build_panels(groups, draw_seeds, caption, embed_caption=embed_caption), caption


def write_seeds(
    runs: Sequence[RunView], output: Path, *, formats: tuple[str, ...] = DEFAULT_FORMATS
) -> list[Path]:
    """Draw one scenario's per-seed figure and save it with a caption file.

    Args:
        runs: The scenario's runs, one per seed.
        output: Directory to write into.
        formats: Image formats to write.

    How:
        Names the files ``seeds_<scenario>`` and writes the caption beside them.

    Returns:
        The image paths, then the caption path.
    """
    figure, caption = build_seeds(runs)
    return save_with_caption(figure, caption, output / f"seeds_{runs[0].scenario}", formats)


def _checkpoints(runs: Sequence[RunView]) -> list[tuple[int, str]]:
    """Return ``(step, label)`` for the reference step and the final budget."""
    budget = max((run.budget or _last_step(run) for run in runs), default=REFERENCE_STEP)
    final = (budget, f"{budget // 1000}k steps")
    if budget > REFERENCE_STEP:
        return [(REFERENCE_STEP, f"{REFERENCE_STEP // 1000}k steps"), final]
    return [final]


def _last_step(run: RunView) -> int:
    """Return the last checkpoint step a run recorded, or the reference step."""
    steps = [point.step for point in (*run.before_curve, *run.after_curve)]
    return max(steps, default=REFERENCE_STEP)


def _value(curve: Any, step: int) -> float:
    """Read a curve at a step, or ``nan`` when the run has no curve."""
    return success_at(curve, step) if curve else float("nan")


def _draw_pairs(ax: Any, left: float, before: list[float], after: list[float]) -> None:
    """Draw one checkpoint's paired dots, mean bars and whiskers."""
    for first, second in zip(before, after, strict=True):
        ax.plot([left, left + 1], [first, second], color=PAIR_COLOUR, linewidth=0.7, zorder=1)
    for position, values, role in ((left, before, Role.BEFORE), (left + 1, after, Role.AFTER)):
        colour = ROLE_COLOURS[role]
        ax.scatter([position] * len(values), values, s=DOT_SIZE, color=colour, zorder=3)
        mean, spread = float(np.nanmean(values)), float(np.nanstd(values))
        ax.hlines(mean, position - MEAN_HALF_WIDTH, position + MEAN_HALF_WIDTH, color=colour,
                  linewidth=2.0, zorder=4)
        if len(values) > 1:
            # A probability cannot leave [0, 1], so the whisker stops at the bounds.
            low, high = max(mean - spread, 0.0), min(mean + spread, 1.0)
            ax.vlines(position, low, high, color=colour, linewidth=1.0, zorder=4)


def _add_checkpoint_labels(ax: Any, centres: list[float], names: list[str]) -> None:
    """Name each checkpoint once, centred under its before and after columns.

    Repeating the checkpoint in every tick label made neighbouring labels collide, so
    the ticks say only before or after and a second axis below carries the checkpoint.
    """
    lower = ax.secondary_xaxis(CHECKPOINT_AXIS_OFFSET)
    lower.set_xticks(centres, names)
    lower.tick_params(length=0)
    lower.spines["bottom"].set_visible(False)


def _format(ax: Any, ticks: list[float]) -> None:
    """Label the y axis, fix its range and add a light horizontal grid."""
    ax.set_ylabel("Success probability")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(ticks[0] - 0.6, ticks[-1] + 0.6)
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
