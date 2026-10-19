"""One panel per scenario, drawn by a per-axes function, under one shared footer.

The curves and seed figures differ only in what they draw on an axes, so this builds
the grid, titles each panel from its runs, and adds the shared legend and caption. A
single scenario is the same call with one group, which gives a single-column figure.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from faultline.viz.captions import group_title
from faultline.viz.figures.footer import add_footer
from faultline.viz.run import RunView
from faultline.viz.style import DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH, Role, pyplot

GRID_COLUMNS = 3
PANEL_HEIGHT = 2.5
TITLE_HEIGHT = 0.1

DrawPanel = Callable[[Any, Sequence[RunView]], set[Role]]


def build_panels(
    groups: Mapping[str, Sequence[RunView]],
    draw: DrawPanel,
    caption: str,
    *,
    embed_caption: bool = False,
) -> Any:
    """Build a figure with one panel per scenario group.

    Args:
        groups: The runs of each scenario, keyed by scenario name.
        draw: Draws one scenario's runs on an axes and returns the roles it used.
        caption: Caption text, written under the legend when embedded.
        embed_caption: Whether to print the caption inside the figure.

    How:
        One group gives a single-column figure; several give a grid of up to three
        columns at double-column width. Each panel is titled from its runs, unused
        grid cells are blanked, and the roles drawn anywhere share one legend.

    Returns:
        The figure.
    """
    plt = pyplot()
    count = len(groups)
    columns = min(count, GRID_COLUMNS)
    rows = math.ceil(count / columns)
    width = SINGLE_COLUMN_WIDTH if count == 1 else DOUBLE_COLUMN_WIDTH
    body_height = PANEL_HEIGHT * rows + TITLE_HEIGHT
    figure, axes = plt.subplots(
        rows, columns, figsize=(width, body_height), layout="constrained", squeeze=False
    )
    cells = list(axes.ravel())
    roles: set[Role] = set()
    for ax, (name, runs) in zip(cells, groups.items(), strict=False):
        roles |= draw(ax, runs)
        ax.set_title(group_title(name, [run.steps for run in runs]))
    for ax in cells[count:]:
        ax.axis("off")
    add_footer(figure, roles, caption if embed_caption else None, width, body_height)
    return figure
