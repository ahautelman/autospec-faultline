"""One shared legend per figure, with an entry per role actually drawn.

Drawing code returns the :class:`Role`s it used instead of labelling its own artists, so
a figure with six panels still gets one short legend with each role named once, in a
fixed order. Proxy artists mimic how each role is drawn (filled patch, hatched patch,
line, dashed line, marker).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from faultline.viz.geometry import DEFAULT_FILL_ALPHA
from faultline.viz.room import TRAP_HATCH
from faultline.viz.style import ROLE_COLOURS, ROLE_LABELS, Role

MAX_COLUMNS = 4
LEGEND_LOCATION = "outside lower center"

# How each role is drawn in the figures, so its legend swatch matches.
_LINE_ROLES = {Role.EDGE, Role.TRAJECTORY_OK, Role.TRAJECTORY_FAIL, Role.BEFORE, Role.AFTER}
_DASHED_ROLES = {Role.BOUNDARY}
_DOTTED_ROLES = {Role.THRESHOLD}
_MARKERS = {Role.SAMPLE: "o", Role.REFERENCE: "D"}
BAR_HATCH = "////"


def legend_handles(roles: Iterable[Role]) -> tuple[list[Any], list[str]]:
    """Build the legend's swatches and labels for a set of roles.

    Args:
        roles: The roles drawn on the figure, in any order, duplicates allowed.

    How:
        Keeps each role once in :class:`Role` declaration order, so every figure
        lists roles consistently, and builds a proxy artist that looks like the role.

    Returns:
        The handles and their labels, in matching order.
    """
    wanted = set(roles)
    ordered = [role for role in Role if role in wanted]
    return [_proxy(role) for role in ordered], [ROLE_LABELS[role] for role in ordered]


def legend_rows(role_count: int) -> int:
    """Return how many rows a legend of ``role_count`` entries takes.

    Args:
        role_count: Number of legend entries.

    How:
        Divides by the column cap, rounding up.

    Returns:
        The row count, zero for no entries.
    """
    return -(-role_count // MAX_COLUMNS)


def add_legend(figure: Any, roles: Iterable[Role], *, bottom: float | None = None) -> Any | None:
    """Add the shared legend below a figure.

    Args:
        figure: The figure to label.
        roles: The roles drawn anywhere on the figure.
        bottom: Figure-fraction height of the legend's lower edge. When ``None`` the
            legend is placed outside the axes by constrained layout, which needs the
            figure to use it; a figure that reserves its own footer passes a value.

    How:
        Places the legend at the bottom with at most four columns.

    Returns:
        The legend, or ``None`` when no role was drawn.
    """
    handles, labels = legend_handles(roles)
    if not handles:
        return None
    columns = min(len(handles), MAX_COLUMNS)
    if bottom is None:
        return figure.legend(handles, labels, loc=LEGEND_LOCATION, ncol=columns)
    return figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, bottom), ncol=columns
    )


def _proxy(role: Role) -> Any:
    """Return a proxy artist that looks like ``role`` is drawn."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    colour = ROLE_COLOURS[role]
    if role in _MARKERS:
        return Line2D([], [], marker=_MARKERS[role], linestyle="", markersize=3.5,
                      markerfacecolor="white" if role is Role.REFERENCE else colour, color=colour)
    if role in _DOTTED_ROLES:
        return Line2D([], [], color=colour, linestyle=":", linewidth=1.2)
    if role is Role.REFERENCE_BAR:
        return Patch(facecolor="white", edgecolor=colour, hatch=BAR_HATCH)
    if role in _DASHED_ROLES:
        return Line2D([], [], color=colour, linestyle="--", linewidth=1.4)
    if role in _LINE_ROLES:
        return Line2D([], [], color=colour, linewidth=1.4)
    hatch = TRAP_HATCH if role is Role.UNSAFE else None
    return Patch(facecolor=to_rgba(colour, DEFAULT_FILL_ALPHA), edgecolor=colour, hatch=hatch)
