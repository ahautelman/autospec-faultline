"""Comparison figures for the 9-rooms report.

Three figure types live here, and the geometric overlay lives in
:mod:`faultline.experiments.overlay_figures`:

- the grouped bar chart comparing the paper's reference numbers with this run's
  reference-checkpoint numbers,
- the per-scenario learning curves over the full ``0..budget`` range,
- the per-seed spread across the multi-seed runs.

Each is written per scenario and as a combined grid, all from the persisted
records, so the figures regenerate without retraining.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from faultline.experiments.drawing import pyplot
from faultline.experiments.overlay_figures import write_overlay_figures
from faultline.experiments.records import (
    REFERENCE_STEP,
    ScenarioReport,
    ScenarioSpread,
    success_at,
)
from faultline.scenarios import Scenario

BAR_WIDTH = 0.2
BAR_OFFSETS = (-1.5, -0.5, 0.5, 1.5)
BAR_COLOURS = ("#b0b0b0", "#d9d9d9", "#4c72b0", "#55a868")
REFERENCE_LABELS = ("paper before", "paper after", "yours before", "yours after")
SPREAD_LABELS = ("before 80k", "after 80k", "before final", "after final")
GRID_COLUMNS = 3
PANEL_SIZE = 4.8


def write_figures(
    reports: Mapping[str, ScenarioReport],
    spreads: Mapping[str, ScenarioSpread],
    scenarios: Mapping[str, Scenario],
    output: Path,
) -> None:
    """Write every figure type per scenario and as a combined grid.

    Args:
        reports: Single-seed reports keyed by scenario name.
        spreads: Multi-seed spreads keyed by scenario name; may be empty.
        scenarios: Scenarios keyed by scenario name, for the overlay geometry.
        output: Directory the figures are written to.

    How:
        Writes the reference comparison, learning curves, overlays, and (when
        present) seed spread, each per scenario plus a combined figure.

    Returns:
        None: the figure files are written to ``output``.
    """
    output.mkdir(parents=True, exist_ok=True)
    plot_reference_comparison(reports, output / "reference_comparison.png")
    plot_learning_curves(reports, output / "learning_curves.png")
    write_overlay_figures(reports, scenarios, output)
    for name, report in reports.items():
        plot_reference_comparison_single(report, output / f"reference_comparison_{name}.png")
        plot_learning_curves_single(report, output / f"learning_curves_{name}.png")
    if not spreads:
        return
    plot_seed_spread(spreads, output / "seed_spread.png")
    for name, spread in spreads.items():
        plot_seed_spread_single(spread, output / f"seed_spread_{name}.png")


def plot_reference_comparison(reports: Mapping[str, ScenarioReport], path: Path) -> None:
    """Write the combined reference bar chart.

    Args:
        reports: Single-seed reports keyed by scenario name.
        path: Destination image path.

    How:
        Draws the four grouped bars per scenario, using each report's paper
        reference and its value at the paper's 80k reference step.

    Returns:
        None: the figure is written to ``path``.
    """
    plt = pyplot()
    names = list(reports)
    figure, ax = plt.subplots(figsize=(1.9 * len(names) + 3.0, 4.6))
    _reference_bars(ax, names, reports)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("success probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Paper reference vs this run at the {REFERENCE_STEP // 1000}k checkpoint")
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_reference_comparison_single(report: ScenarioReport, path: Path) -> None:
    """Write one scenario's reference bar chart.

    Args:
        report: The single-seed report to plot.
        path: Destination image path.

    How:
        Draws the same four bars as the combined chart, but for one scenario and
        with a scenario-specific title.

    Returns:
        None: the figure is written to ``path``.
    """
    plt = pyplot()
    figure, ax = plt.subplots(figsize=(4.6, 4.4))
    _reference_bars(ax, [report.scenario], {report.scenario: report})
    ax.set_xticks([0.0])
    ax.set_xticklabels([report.scenario])
    ax.set_ylabel("success probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{report.scenario}: reference vs this run")
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_learning_curves(reports: Mapping[str, ScenarioReport], path: Path) -> None:
    """Write the combined learning-curve grid.

    Args:
        reports: Single-seed reports keyed by scenario name.
        path: Destination image path.

    How:
        Draws one panel per scenario, then blanks any unused panels in the grid.

    Returns:
        None: the figure is written to ``path``.
    """
    plt = pyplot()
    names = list(reports)
    rows = math.ceil(len(names) / GRID_COLUMNS)
    figure, axes = plt.subplots(rows, GRID_COLUMNS, figsize=(PANEL_SIZE * GRID_COLUMNS, 3.8 * rows))
    panels = list(np.ravel(axes))
    for ax, name in zip(panels, names, strict=False):
        _draw_curve(ax, reports[name])
    for ax in panels[len(names) :]:
        ax.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_learning_curves_single(report: ScenarioReport, path: Path) -> None:
    """Write one scenario's learning-curve figure.

    Args:
        report: The single-seed report to plot.
        path: Destination image path.

    How:
        Draws the same before and after curves as the combined grid, but on a
        single larger panel.

    Returns:
        None: the figure is written to ``path``.
    """
    plt = pyplot()
    figure, ax = plt.subplots(figsize=(6.0, 4.2))
    _draw_curve(ax, report)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_seed_spread(spreads: Mapping[str, ScenarioSpread], path: Path) -> None:
    """Write the combined per-seed spread grid.

    Args:
        spreads: Multi-seed spreads keyed by scenario name.
        path: Destination image path.

    How:
        Draws one panel per scenario with the mean and standard deviation across
        seeds for the four reported numbers.

    Returns:
        None: the figure is written to ``path``.
    """
    plt = pyplot()
    names = list(spreads)
    rows = math.ceil(len(names) / GRID_COLUMNS)
    figure, axes = plt.subplots(rows, GRID_COLUMNS, figsize=(PANEL_SIZE * GRID_COLUMNS, 3.8 * rows))
    panels = list(np.ravel(axes))
    for ax, name in zip(panels, names, strict=False):
        _draw_spread(ax, spreads[name])
    for ax in panels[len(names) :]:
        ax.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_seed_spread_single(spread: ScenarioSpread, path: Path) -> None:
    """Write one scenario's per-seed spread figure.

    Args:
        spread: The multi-seed spread to plot.
        path: Destination image path.

    How:
        Draws the same grouped means and standard deviations as the combined grid,
        but on a single larger panel.

    Returns:
        None: the figure is written to ``path``.
    """
    plt = pyplot()
    figure, ax = plt.subplots(figsize=(5.4, 4.2))
    _draw_spread(ax, spread)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _reference_bars(
    ax: Any, names: list[str], reports: Mapping[str, ScenarioReport]
) -> None:
    """Draw the four grouped bars for each named scenario.

    Args:
        ax: Target axes.
        names: Scenario names, in bar-group order.
        reports: The reports to read the values from.

    How:
        Per scenario, draws the paper's before and after reference beside this
        run's before and after at the reference checkpoint, each offset within the
        group so the four bars are comparable across scenarios.

    Returns:
        None: the axes are drawn in place.
    """
    positions = np.arange(len(names), dtype=float)
    for index, name in enumerate(names):
        report = reports[name]
        values = (
            report.reference_before,
            report.reference_after,
            success_at(report.before, REFERENCE_STEP),
            success_at(report.after, REFERENCE_STEP),
        )
        for offset, value, colour, label in zip(
            BAR_OFFSETS, values, BAR_COLOURS, REFERENCE_LABELS, strict=True
        ):
            ax.bar(
                positions[index] + offset * BAR_WIDTH,
                value,
                width=BAR_WIDTH,
                color=colour,
                label=label if index == 0 else None,
            )
    ax.set_xlim(-0.6, len(names) - 0.4)


def _draw_curve(ax: Any, report: ScenarioReport) -> None:
    """Draw one scenario's before and after curves with the reference marker.

    Args:
        ax: Target axes.
        report: The single-seed report to plot.

    How:
        Plots the pre-refinement and refined series over their full step range and
        adds a dotted guide at the reference step, which is the x-value the bar
        chart reads.

    Returns:
        None: the axes are drawn in place.
    """
    before_steps = [checkpoint.step for checkpoint in report.before]
    before_values = [checkpoint.probability for checkpoint in report.before]
    after_steps = [checkpoint.step for checkpoint in report.after]
    after_values = [checkpoint.probability for checkpoint in report.after]
    ax.plot(before_steps, before_values, color="#4c72b0", label="before refinement")
    ax.plot(after_steps, after_values, color="#55a868", label="after refinement")
    ax.axvline(REFERENCE_STEP, color="#999999", linestyle=":", linewidth=1.2, label="reference step")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("success probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{report.scenario} ({report.operator})")
    ax.legend(fontsize=7)


def _draw_spread(ax: Any, spread: ScenarioSpread) -> None:
    """Draw the mean and standard deviation across seeds.

    Args:
        ax: Target axes.
        spread: The multi-seed spread to plot.

    How:
        Groups the four reported numbers (before and after at the reference and
        final checkpoints) and draws each group's mean with its standard deviation
        as the error bar.

    Returns:
        None: the axes are drawn in place.
    """
    before_80k = [summary.before_80k for summary in spread.summaries]
    after_80k = [summary.after_80k for summary in spread.summaries]
    before_final = [summary.before_final for summary in spread.summaries]
    after_final = [summary.after_final for summary in spread.summaries]
    groups = (before_80k, after_80k, before_final, after_final)
    positions = np.arange(len(groups), dtype=float)
    means = [float(np.mean(group)) for group in groups]
    errors = [float(np.std(group)) for group in groups]
    ax.bar(positions, means, yerr=errors, capsize=4, color=BAR_COLOURS)
    ax.set_xticks(positions)
    ax.set_xticklabels(SPREAD_LABELS, rotation=10)
    ax.set_ylabel("success probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{spread.scenario}: {len(spread.summaries)} seeds")
