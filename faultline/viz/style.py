"""The shared look of every figure: fonts, colours by role, sizes, and saving.

Colours are tied to what a thing *is* in a run (a start region, an unsafe area, a failed
rollout), not to the figure it appears in, so the same object looks the same everywhere
and the legend can name each role once. The palette is the colour-blind-safe Okabe-Ito
set. Matplotlib is imported inside the functions that need it.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

SINGLE_COLUMN_WIDTH = 3.4
DOUBLE_COLUMN_WIDTH = 7.0
SAVE_DPI = 300
WALL_COLOUR = "#262626"
WALL_WIDTH = 1.4
TEXT_COLOUR = "#262626"
GRID_COLOUR = "#d9d9d9"
DEFAULT_FORMATS = ("pdf", "png")


class Role(StrEnum):
    """What a drawn element represents in a run.

    How:
        Each member keys one colour and one legend label, so every figure draws and
        names the same kind of element identically.
    """

    START = "start"
    GOAL = "goal"
    WAYPOINT = "waypoint"
    UNSAFE = "unsafe"
    REFINED = "refined"
    BOUNDARY = "boundary"
    EDGE = "edge"
    SAMPLE = "sample"
    TRAJECTORY_OK = "trajectory_ok"
    TRAJECTORY_FAIL = "trajectory_fail"
    BEFORE = "before"
    AFTER = "after"
    THRESHOLD = "threshold"
    REFERENCE = "reference"
    REFERENCE_BAR = "reference_bar"


ROLE_COLOURS: dict[Role, str] = {
    Role.START: "#0072B2",
    Role.GOAL: "#009E73",
    Role.WAYPOINT: "#56B4E9",
    Role.UNSAFE: "#D55E00",
    Role.REFINED: "#E69F00",
    Role.BOUNDARY: "#CC79A7",
    Role.EDGE: "#262626",
    Role.SAMPLE: "#6B4A00",
    Role.TRAJECTORY_OK: "#0072B2",
    Role.TRAJECTORY_FAIL: "#CC79A7",
    Role.BEFORE: "#7f7f7f",
    Role.AFTER: "#0072B2",
    Role.THRESHOLD: "#7f7f7f",
    Role.REFERENCE: "#262626",
    Role.REFERENCE_BAR: "#7f7f7f",
}

ROLE_LABELS: dict[Role, str] = {
    Role.START: "Start region",
    Role.GOAL: "Goal region",
    Role.WAYPOINT: "Waypoint region",
    Role.UNSAFE: "Unsafe area",
    Role.REFINED: "Refined region",
    Role.BOUNDARY: "Split boundary",
    Role.EDGE: "Added edge",
    Role.SAMPLE: "Visited states",
    Role.TRAJECTORY_OK: "Successful rollout",
    Role.TRAJECTORY_FAIL: "Failed rollout",
    Role.BEFORE: "Before refinement",
    Role.AFTER: "After refinement",
    Role.THRESHOLD: "Success threshold",
    Role.REFERENCE: "Paper reference",
    Role.REFERENCE_BAR: "Paper reference",
}

_RC_PARAMS: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.color": TEXT_COLOUR,
    "axes.labelcolor": TEXT_COLOUR,
    "axes.edgecolor": TEXT_COLOUR,
    "xtick.color": TEXT_COLOUR,
    "ytick.color": TEXT_COLOUR,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "hatch.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": SAVE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def pyplot() -> Any:
    """Return ``matplotlib.pyplot`` on the headless Agg backend with the style applied.

    How:
        Imports matplotlib inside the call, selects Agg before pyplot, and updates
        the global rc parameters. The update is idempotent, so every figure entry
        point can call this first.

    Returns:
        The ``matplotlib.pyplot`` module.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(_RC_PARAMS)
    return plt


def save_figure(figure: Any, stem: Path, formats: Sequence[str] = DEFAULT_FORMATS) -> list[Path]:
    """Save a figure once per format and close it.

    Args:
        figure: The matplotlib figure.
        stem: Output path without a suffix; its folder is created if missing.
        formats: File suffixes to write, without the dot.

    How:
        Writes ``<stem>.<format>`` for each format, then closes the figure to free
        its memory.

    Returns:
        The paths written, in ``formats`` order.
    """
    import matplotlib.pyplot as plt

    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_name(f"{stem.name}.{suffix}") for suffix in formats]
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths
