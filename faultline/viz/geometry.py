"""Draw a region predicate on a matplotlib axes.

Regions that reduce to convex polygons are drawn as exact vector shapes; anything
else (a complement) is rasterised over its bounding box. Drawing never registers a
legend entry: the caller gets the :class:`Role` back and builds one shared legend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from faultline.geometry import region_bounds
from faultline.spectrl.predicates import PLANAR_DIMS, Predicate
from faultline.viz.polygons import region_polygons
from faultline.viz.style import ROLE_COLOURS, Role

DEFAULT_FILL_ALPHA = 0.28
OUTLINE_WIDTH = 0.8
RASTER_RESOLUTION = 300


def draw_region(
    ax: Any,
    predicate: Predicate,
    role: Role,
    world: float,
    *,
    alpha: float = DEFAULT_FILL_ALPHA,
    hatch: str | None = None,
    zorder: float = 1.0,
) -> Role | None:
    """Fill and outline the states a region accepts.

    Args:
        ax: Target axes.
        predicate: The region to draw.
        role: What the region is, which picks its colour.
        world: Side length of the square world.
        alpha: Fill opacity; the outline is always opaque.
        hatch: Optional matplotlib hatch pattern, used for unsafe areas.
        zorder: Drawing order on the axes.

    How:
        Draws the exact polygons as one compound path when the region has them, so
        overlapping pieces fill once, and otherwise rasterises the region.

    Returns:
        ``role`` for the caller to collect into the legend, or ``None`` when the
        region has no area and nothing was drawn, so the legend never names an
        invisible element.
    """
    polygons = region_polygons(predicate, world)
    if polygons is None:
        return role if _draw_raster(ax, predicate, role, world, alpha, hatch, zorder) else None
    if not polygons:
        return None
    _draw_polygons(ax, polygons, role, alpha, hatch, zorder)
    return role


def region_centre(predicate: Predicate) -> np.ndarray | None:
    """Return the centre of a region's bounding box.

    Args:
        predicate: The region to centre.

    How:
        Takes the midpoint of the region's planar bounds.

    Returns:
        The centre ``(x, y)``, or ``None`` when the region is unbounded.
    """
    bounds = region_bounds(predicate, PLANAR_DIMS)
    if bounds is None:
        return None
    return (bounds[0] + bounds[1]) / 2.0


def _draw_polygons(
    ax: Any, polygons: list[np.ndarray], role: Role, alpha: float, hatch: str | None, zorder: float
) -> None:
    """Draw convex polygons as a single filled, outlined compound path."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    vertices, codes = [], []
    for polygon in polygons:
        closed = np.vstack([polygon, polygon[:1]])
        vertices.append(closed)
        codes.extend([Path.MOVETO, *([Path.LINETO] * (len(closed) - 2)), Path.CLOSEPOLY])
    colour = ROLE_COLOURS[role]
    patch = PathPatch(
        Path(np.vstack(vertices), codes),
        facecolor=to_rgba(colour, alpha),
        edgecolor=to_rgba(colour, 1.0),
        linewidth=OUTLINE_WIDTH,
        hatch=hatch,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _draw_raster(
    ax: Any, predicate: Predicate, role: Role, world: float, alpha: float, hatch: str | None, zorder: float
) -> bool:
    """Fill a region by evaluating it on a grid over its bounding box.

    Returns ``True`` when any cell was accepted and so something was drawn.
    """
    low, high = np.zeros(2), np.full(2, world)
    bounds = region_bounds(predicate, PLANAR_DIMS)
    if bounds is not None:
        low, high = np.maximum(low, bounds[0]), np.minimum(high, bounds[1])
    xs = np.linspace(low[0], high[0], RASTER_RESOLUTION)
    ys = np.linspace(low[1], high[1], RASTER_RESOLUTION)
    mask = np.array([[predicate.evaluate(np.array([x, y])) for x in xs] for y in ys], dtype=float)
    if not mask.any():
        return False
    colour = ROLE_COLOURS[role]
    ax.contourf(
        xs, ys, mask, levels=[0.5, 1.5], colors=[colour], alpha=alpha, hatches=[hatch], zorder=zorder
    )
    ax.contour(xs, ys, mask, levels=[0.5], colors=[colour], linewidths=OUTLINE_WIDTH, zorder=zorder)
    return True
