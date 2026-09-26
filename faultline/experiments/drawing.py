"""Drawing primitives for the 9-rooms report figures.

The report renders the room grid, region predicates, and the geometry a refinement
operator produced. Matplotlib is imported lazily and pinned to the Agg backend, so
importing the experiments package never requires the reports extra and the figures
render headlessly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from faultline.envs.layout import RoomsLayout
from faultline.geometry import region_bounds
from faultline.experiments.records import Overlay
from faultline.spectrl.predicates import PLANAR_DIMS, Predicate

GRID_RESOLUTION = 220
WALL_WIDTH = 2.4
POINT_SIZE = 8.0
SWATCH_WIDTH = 8.0
EDGE_COLOUR = "#333333"
BOUNDARY_COLOUR = "#8172b3"


def pyplot() -> Any:
    """Return ``matplotlib.pyplot`` on the headless Agg backend.

    How:
        Imports matplotlib inside the call and selects Agg before importing
        pyplot, so a machine without the reports extra only fails when plotting is
        actually requested.

    Returns:
        The ``matplotlib.pyplot`` module.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class RegionStyle:
    """How to render a region or a point set.

    Args:
        color: Matplotlib colour string.
        alpha: Fill or marker opacity.
        label: Legend label.
    """

    color: str
    alpha: float
    label: str


@dataclass(frozen=True)
class WallSpan:
    """One room-length wall run.

    Args:
        line: The grid-line coordinate the wall sits on.
        low: Lower coordinate along the wall.
        high: Upper coordinate along the wall.
        vertical: ``True`` for a wall running along ``y``, else along ``x``.
    """

    line: float
    low: float
    high: float
    vertical: bool


def draw_layout(ax: Any, layout: RoomsLayout) -> None:
    """Draw the room grid's walls and door gaps.

    Args:
        ax: Target axes.
        layout: The scenario's wall and door model.

    How:
        Walks every internal grid line and each room along it, drawing a wall with
        a door as the two solid pieces either side of the door interval and a wall
        without a door whole.

    Returns:
        None: the axes are drawn in place.
    """
    for line_index in range(1, layout.rooms_per_side):
        line = line_index * layout.room_size
        for room in range(layout.rooms_per_side):
            span = WallSpan(line, room * layout.room_size, (room + 1) * layout.room_size, True)
            _segmented_wall(ax, span, layout.doors.get(("v", line_index - 1, room)))
            span = WallSpan(span.line, span.low, span.high, False)
            _segmented_wall(ax, span, layout.doors.get(("h", room, line_index - 1)))


def shade_region(ax: Any, predicate: Predicate, layout: RoomsLayout, style: RegionStyle) -> None:
    """Fill the states a region accepts.

    Args:
        ax: Target axes.
        predicate: The region to shade.
        layout: The scenario layout, for the world size.
        style: Colour, opacity, and legend label.

    How:
        Rasterises the predicate over the world square and fills the accepted
        cells with a contour, which renders any predicate shape without knowing
        its geometry. A legend swatch is added so the fill appears in the legend.

    Returns:
        None: the axes are drawn in place.
    """
    world = layout.world_size
    axis = np.linspace(0.0, world, GRID_RESOLUTION)
    points = np.array([[x, y] for y in axis for x in axis])
    mask = np.array([predicate.evaluate(point) for point in points])
    if not mask.any():
        return
    grid = mask.reshape(GRID_RESOLUTION, GRID_RESOLUTION).astype(float)
    ax.contourf(axis, axis, grid, levels=[0.5, 1.5], colors=[style.color], alpha=style.alpha)
    ax.plot([], [], color=style.color, alpha=style.alpha, linewidth=SWATCH_WIDTH, label=style.label)


def scatter_points(ax: Any, points: tuple[tuple[float, float], ...], style: RegionStyle) -> None:
    """Plot a set of sampled states as small markers.

    Args:
        ax: Target axes.
        points: ``(x, y)`` pairs to plot.
        style: Colour, opacity, and legend label.

    How:
        Scatters the points once; an empty set is skipped so it does not add an
        empty legend entry.

    Returns:
        None: the axes are drawn in place.
    """
    if not points:
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.scatter(xs, ys, s=POINT_SIZE, color=style.color, alpha=style.alpha, label=style.label)


def draw_half_plane(ax: Any, overlay: Overlay, layout: RoomsLayout) -> None:
    """Draw the half-plane boundary PastRefine fitted.

    Args:
        ax: Target axes.
        overlay: The captured overlay, which carries the normal and offset.
        layout: The scenario layout, for the line extent.

    How:
        The boundary is ``normal . p = offset``; this solves for the point closest
        to the origin and extends a line along the normal's perpendicular across
        the world. A missing normal is skipped.

    Returns:
        None: the axes are drawn in place.
    """
    if overlay.normal is None or overlay.offset is None:
        return
    normal = np.asarray(overlay.normal, dtype=float)
    anchor = normal * (overlay.offset / float(normal @ normal))
    direction = np.array([-normal[1], normal[0]])
    extent = layout.world_size
    start = anchor - direction * extent
    stop = anchor + direction * extent
    ax.plot(
        [start[0], stop[0]],
        [start[1], stop[1]],
        color=BOUNDARY_COLOUR,
        linestyle="--",
        linewidth=1.8,
        label="PastRefine boundary",
    )


def draw_arrow(ax: Any, start: np.ndarray, stop: np.ndarray, label: str) -> None:
    """Draw a structural edge as an arrow between two region centres.

    Args:
        ax: Target axes.
        start: Source centre ``(x, y)``.
        stop: Target centre ``(x, y)``.
        label: Legend label for the edge.

    How:
        Draws an annotated arrow, so the added OrRefine edge reads as a graph edit
        rather than a geometric region.

    Returns:
        None: the axes are drawn in place.
    """
    ax.annotate(
        "",
        xy=(stop[0], stop[1]),
        xytext=(start[0], start[1]),
        arrowprops={"arrowstyle": "-|>", "color": EDGE_COLOUR, "linewidth": 2.0},
    )
    ax.plot([], [], color=EDGE_COLOUR, linewidth=2.0, label=label)


def region_centre(predicate: Predicate, layout: RoomsLayout) -> np.ndarray:
    """Return the centre of a region's bounding box.

    Args:
        predicate: The region to centre.
        layout: The scenario layout, used for a fallback centre.

    How:
        Takes the midpoint of the region's axis-aligned bounds; an unbounded
        region falls back to the world centre.

    Returns:
        The centre position ``(x, y)``.
    """
    bounds = region_bounds(predicate, PLANAR_DIMS)
    if bounds is None:
        half = layout.world_size / 2.0
        return np.array([half, half])
    return (bounds[0] + bounds[1]) / 2.0


def configure_axis(ax: Any, layout: RoomsLayout, title: str) -> None:
    """Set the world bounds, a square aspect, and a title on one axes.

    Args:
        ax: Target axes.
        layout: The scenario layout, for the world size.
        title: Axes title.

    How:
        Fixes the limits to the world square, equalises the aspect, hides the
        ticks, and sets a compact title.

    Returns:
        None: the axes are configured in place.
    """
    ax.set_xlim(0.0, layout.world_size)
    ax.set_ylim(0.0, layout.world_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def _segmented_wall(ax: Any, span: WallSpan, door: tuple[float, float] | None) -> None:
    """Draw one room-length wall, leaving a gap where a door opens.

    Args:
        ax: Target axes.
        span: The wall run to draw.
        door: The open interval on the wall, or ``None`` for a solid wall.

    How:
        A solid wall is one segment; a wall with a door is the two segments either
        side of the door interval.

    Returns:
        None: the axes are drawn in place.
    """
    if door is None:
        _segment(ax, span)
        return
    _segment(ax, WallSpan(span.line, span.low, door[0], span.vertical))
    _segment(ax, WallSpan(span.line, door[1], span.high, span.vertical))


def _segment(ax: Any, span: WallSpan) -> None:
    """Draw one solid wall segment.

    Args:
        ax: Target axes.
        span: The segment to draw.

    How:
        Skips a zero-length span; otherwise draws a black line along the span's
        axis.

    Returns:
        None: the axes are drawn in place.
    """
    if span.high <= span.low:
        return
    if span.vertical:
        ax.plot([span.line, span.line], [span.low, span.high], color=EDGE_COLOUR, linewidth=WALL_WIDTH)
    else:
        ax.plot([span.low, span.high], [span.line, span.line], color=EDGE_COLOUR, linewidth=WALL_WIDTH)
