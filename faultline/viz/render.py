"""Render every figure a data directory supports.

Chooses what to draw from what the runs contain: an overlay needs recorded geometry, a
learning curve needs more than one checkpoint, a per-seed figure needs several seeds, and
the paper comparison needs the paper's numbers. A figure the data cannot support is
skipped with a note instead of failing, so one call works for any run.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from faultline.scenarios import SCENARIOS, Scenario
from faultline.viz.figures.curves import build_curves_grid, write_curves
from faultline.viz.figures.output import save_with_caption
from faultline.viz.figures.overlay import write_overlay
from faultline.viz.figures.reference import write_reference
from faultline.viz.figures.seeds import build_seeds_grid, write_seeds
from faultline.viz.load import RunData, load_data
from faultline.viz.run import RunView, group_by_scenario
from faultline.viz.style import DEFAULT_FORMATS

FIGURE_KINDS = ("overlay", "curves", "seeds", "reference")
MIN_CURVE_POINTS = 2
MIN_SEEDS = 2


@dataclass(frozen=True)
class RenderOptions:
    """Which figures to draw and how.

    Args:
        figures: The kinds to draw, from :data:`FIGURE_KINDS`.
        scenarios: Scenario names to keep, or ``None`` for all.
        seeds: Seeds to keep, or ``None`` for all. The overlay draws the lowest seed
            per scenario unless seeds are given, in which case it draws each one.
        per_step: Add one overlay panel per committed step.
        embed_caption: Also print each caption inside its figure.
        formats: Image formats to write.
    """

    figures: Sequence[str] = FIGURE_KINDS
    scenarios: Collection[str] | None = None
    seeds: Collection[int] | None = None
    per_step: bool = False
    embed_caption: bool = False
    formats: tuple[str, ...] = DEFAULT_FORMATS


@dataclass(frozen=True)
class RenderResult:
    """What a render wrote and what it skipped.

    Args:
        paths: Every file written.
        notes: One line per figure or file that was skipped, with the reason.
    """

    paths: tuple[Path, ...] = ()
    notes: tuple[str, ...] = field(default=())


def render(data: RunData, output: Path, options: RenderOptions = RenderOptions()) -> RenderResult:
    """Draw the requested figures for loaded runs.

    Args:
        data: The runs found in a data directory.
        output: Directory the figures are written to.
        options: Which figures to draw and how.

    How:
        Filters the runs by scenario and seed, then draws each requested kind with its
        own rule for what data it needs, collecting notes for what it skips.

    Returns:
        The files written and the notes.

    Raises:
        ValueError: If an unknown figure kind is requested.
    """
    unknown = [kind for kind in options.figures if kind not in FIGURE_KINDS]
    if unknown:
        raise ValueError(f"Unknown figure kind(s) {unknown}; choose from {list(FIGURE_KINDS)}")
    runs = _select(data.runs, options)
    seed_source = _select(data.seed_runs, options) or runs
    paths: list[Path] = []
    notes = list(data.notes)
    drawers = {
        "overlay": lambda: _overlays(runs, output, options),
        "curves": lambda: _curves(runs, output, options),
        "seeds": lambda: _seeds(seed_source, output, options),
        "reference": lambda: _reference(seed_source, output, options),
    }
    for kind in options.figures:
        written, skipped = drawers[kind]()
        paths += written
        notes += skipped
    return RenderResult(tuple(paths), tuple(notes))


def render_directory(
    data_dir: Path,
    output: Path,
    options: RenderOptions = RenderOptions(),
    catalogue: Mapping[str, Scenario] = SCENARIOS,
) -> RenderResult:
    """Load a data directory and render its figures, printing a short summary.

    Args:
        data_dir: Folder holding the saved runs.
        output: Directory the figures are written to.
        options: Which figures to draw and how.
        catalogue: Scenarios for runs that did not record their own room.

    How:
        Calls :func:`load_data` then :func:`render`, and prints the number of files
        written plus every skip note.

    Returns:
        The render result.
    """
    result = render(load_data(data_dir, catalogue), output, options)
    for note in result.notes:
        print(f"  note: {note}", flush=True)
    print(f"Wrote {len(result.paths)} figure files to {output}", flush=True)
    return result


def _select(runs: Sequence[RunView], options: RenderOptions) -> list[RunView]:
    """Keep the runs of the chosen scenarios and seeds."""
    return [
        run
        for run in runs
        if (options.scenarios is None or run.scenario in options.scenarios)
        and (options.seeds is None or run.seed in options.seeds)
    ]


def _overlays(runs: Sequence[RunView], output: Path, options: RenderOptions) -> tuple[list[Path], list[str]]:
    """Draw an overlay for each run that recorded geometry."""
    paths, notes = [], []
    for name, group in group_by_scenario(runs).items():
        for run in group if options.seeds is not None else group[:1]:
            if run.final_graph is None:
                notes.append(f"overlay for {name} seed {run.seed}: refinement geometry was not recorded")
                continue
            paths += write_overlay(run, output, per_step=options.per_step,
                                   embed_caption=options.embed_caption, formats=options.formats)
    return paths, notes


def _curves(runs: Sequence[RunView], output: Path, options: RenderOptions) -> tuple[list[Path], list[str]]:
    """Draw learning curves for each scenario that recorded more than one checkpoint."""
    paths, notes = [], []
    groups = {}
    for name, group in group_by_scenario(runs).items():
        if any(len(run.before_curve) >= MIN_CURVE_POINTS for run in group):
            groups[name] = group
            paths += write_curves(group, output, formats=options.formats)
        else:
            notes.append(f"curves for {name}: only the final checkpoint was recorded (rerun with --curves)")
    if len(groups) > 1:
        figure, caption = build_curves_grid(groups, embed_caption=options.embed_caption)
        paths += save_with_caption(figure, caption, output / "curves_all", options.formats)
    return paths, notes


def _seeds(runs: Sequence[RunView], output: Path, options: RenderOptions) -> tuple[list[Path], list[str]]:
    """Draw the per-seed figure for each scenario with several seeds."""
    paths, notes = [], []
    groups = {name: group for name, group in group_by_scenario(runs).items() if len(group) >= MIN_SEEDS}
    for name, group in group_by_scenario(runs).items():
        if len(group) < MIN_SEEDS:
            notes.append(f"seeds for {name}: needs at least {MIN_SEEDS} seeds, found {len(group)}")
    for group in groups.values():
        paths += write_seeds(group, output, formats=options.formats)
    if len(groups) > 1:
        figure, caption = build_seeds_grid(groups, embed_caption=options.embed_caption)
        paths += save_with_caption(figure, caption, output / "seeds_all", options.formats)
    return paths, notes


def _reference(runs: Sequence[RunView], output: Path, options: RenderOptions) -> tuple[list[Path], list[str]]:
    """Draw the paper comparison if any run carries the paper's numbers."""
    groups = {
        name: group
        for name, group in group_by_scenario(runs).items()
        if group[0].reference_before is not None and group[0].reference_after is not None
    }
    if not groups:
        return [], ["reference comparison: no run carries the paper's reference numbers"]
    return write_reference(groups, output, formats=options.formats), []
