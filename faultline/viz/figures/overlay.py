"""The overlay figure: the room, what refinement changed, and how the agent moved.

For a run with refinements the figure is ``(a)`` the room with the initial best path's
sampled rollouts and ``(b)`` the same room with every applied edit marked and the
repaired best path's rollouts. ``per_step`` adds one panel per committed step showing
only that step's edit and the states it was based on. A run that applied nothing gets
a single panel. Titles, legend and caption all come from the run, so any scenario or
saved report is drawn the same way.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from faultline.viz.annotations import draw_step
from faultline.viz.captions import operator_name, run_caption, run_title, trajectory_sentence
from faultline.viz.figures.footer import add_footer
from faultline.viz.figures.output import save_with_caption
from faultline.viz.room import draw_room
from faultline.viz.run import RunView
from faultline.viz.style import (
    DEFAULT_FORMATS,
    DOUBLE_COLUMN_WIDTH,
    SINGLE_COLUMN_WIDTH,
    Role,
    pyplot,
)
from faultline.viz.trajectories import draw_trajectories

PANEL_COLUMNS = 2
TITLE_HEIGHT = 0.3
PANEL_HEIGHT = 3.05
NOT_RECORDED = "Refinement geometry was not recorded for this run, so only the room is shown."


def build_overlay(
    run: RunView, *, per_step: bool = False, embed_caption: bool = False
) -> tuple[Any, str]:
    """Build the overlay figure for a run.

    Args:
        run: The run to draw.
        per_step: Also draw one panel per committed step.
        embed_caption: Write the caption under the figure as well as returning it.

    How:
        Chooses the panels from the run: before and after when a step was committed,
        a single panel otherwise, plus step panels on request. Each panel draws the
        room, then the annotations and rollouts that panel shows, and one shared
        legend names the roles drawn anywhere.

    Returns:
        The figure and its caption text.
    """
    plt = pyplot()
    committed = run.committed
    panels = _panel_plan(run, per_step)
    columns = PANEL_COLUMNS if len(panels) > 1 else 1
    rows = math.ceil(len(panels) / columns)
    width = DOUBLE_COLUMN_WIDTH if len(panels) > 1 else SINGLE_COLUMN_WIDTH
    body_height = PANEL_HEIGHT * rows + TITLE_HEIGHT
    figure, axes = plt.subplots(
        rows, columns, figsize=(width, body_height), layout="constrained", squeeze=False
    )
    flat = list(axes.ravel())
    roles: set[Role] = set()
    for ax, (title, draw) in zip(flat, panels, strict=False):
        roles |= draw(ax)
        ax.set_title(title, loc="left")
    for ax in flat[len(panels) :]:
        ax.axis("off")
    figure.suptitle(run_title(run.scenario, run.steps))
    caption = _caption(run, bool(committed))
    add_footer(figure, roles, caption if embed_caption else None, width, body_height)
    return figure, caption


def write_overlay(
    run: RunView,
    output: Path,
    *,
    per_step: bool = False,
    embed_caption: bool = False,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    """Draw a run's overlay and save it with a caption file.

    Args:
        run: The run to draw.
        output: Directory to write into.
        per_step: Also draw one panel per committed step.
        embed_caption: Also write the caption inside the figure.
        formats: Image formats to write.

    How:
        Names the files ``overlay_<scenario>_seed<k>`` and writes the caption beside
        them as ``.caption.txt``, so it can go into a paper's own caption.

    Returns:
        The image paths written, then the caption path.
    """
    figure, caption = build_overlay(run, per_step=per_step, embed_caption=embed_caption)
    stem = output / f"overlay_{run.scenario}_seed{run.seed}"
    return save_with_caption(figure, caption, stem, formats)


def _panel_plan(run: RunView, per_step: bool) -> list[tuple[str, Any]]:
    """Return ``(title, draw function)`` for each panel of the run."""
    graph = run.final_graph or run.initial_graph
    committed = run.committed
    if not committed:
        return [("(a) Sampled rollouts", lambda ax: _room(ax, run) | _paths(ax, run, after=True))]
    panels = [
        ("(a) Before refinement", lambda ax: _room(ax, run) | _paths(ax, run, after=False)),
        ("(b) After refinement", lambda ax: _after(ax, run, graph)),
    ]
    if per_step:
        for index, step in committed:
            title = f"({chr(ord('c') + len(panels) - 2)}) Step {index + 1}: {operator_name(step.operator)}"
            panels.append((title, lambda ax, step=step: _step_panel(ax, run, step, graph)))
    return panels


def _room(ax: Any, run: RunView) -> set[Role]:
    """Draw the run's room from the initial graph."""
    return draw_room(ax, run.layout, run.initial_graph, run.unsafe_region)


def _paths(ax: Any, run: RunView, *, after: bool) -> set[Role]:
    """Draw the sampled rollouts of the initial or the repaired best path."""
    samples = run.trajectories_after if after else run.trajectories_before
    return draw_trajectories(ax, samples)


def _after(ax: Any, run: RunView, graph: Any) -> set[Role]:
    """Draw the room with every committed edit and the repaired best path."""
    roles = _room(ax, run)
    for _index, step in run.committed:
        roles |= draw_step(ax, step.trace, graph, run.world, show_samples=False)
    return roles | _paths(ax, run, after=True)


def _step_panel(ax: Any, run: RunView, step: Any, graph: Any) -> set[Role]:
    """Draw the room with a single step's edit and the states behind it."""
    return _room(ax, run) | draw_step(ax, step.trace, graph, run.world)


def _caption(run: RunView, has_edits: bool) -> str:
    """Compose the caption: step sentences, then the rollout note if any."""
    parts = [run_caption(run.steps, run.world)]
    if run.steps and not has_edits and run.final_graph is None:
        parts.append(NOT_RECORDED)
    if run.trajectories_before or run.trajectories_after:
        parts.append(trajectory_sentence(has_after_panel=has_edits))
    return " ".join(parts)
