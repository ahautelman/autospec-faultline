"""Mark what a refinement step changed on a room.

The room is drawn from the graph the run started with; this module adds the edits on
top. A step is read by the *shape of its edit* (a region that shrank, an unsafe area
cut out of a safe region, a new region, a new edge), not by operator name, so ``seq``
needs no special case, several steps can be drawn on one room, and an operator added
later is drawn with no new code. Each function returns the roles it used, for the
shared legend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from faultline.geometry import intersect, predicate_from_dict
from faultline.graph.snapshot import GraphSnapshot
from faultline.refinement.trace import SafeChange, StepTrace
from faultline.spectrl.predicates import And, HalfPlane, Not, Predicate
from faultline.viz.geometry import draw_region, region_centre
from faultline.viz.room import TRAP_HATCH
from faultline.viz.style import ROLE_COLOURS, Role

REFINED_ZORDER = 1.5
REMOVED_ZORDER = 2.5
SAMPLE_ZORDER = 3.0
BOUNDARY_ZORDER = 3.5
EDGE_ZORDER = 6.0
SAMPLE_SIZE = 3.0
SAMPLE_ALPHA = 0.7
ARROW_WIDTH = 1.1
BOUNDARY_WIDTH = 1.4


def draw_step(
    ax: Any, trace: StepTrace, graph: GraphSnapshot, world: float, *, show_samples: bool = True
) -> set[Role]:
    """Draw one refinement step's edit on a room.

    Args:
        ax: Axes already holding the room.
        trace: The step's trace.
        graph: A graph snapshot that contains every vertex the step mentions,
            normally the final graph, used to place the edge arrows.
        world: Side length of the square world.
        show_samples: Whether to scatter the states the edit was based on.

    How:
        Fills each changed or new region as refined, hatches the area removed from
        a safe region as unsafe, draws the boundary of any half-plane that cut a
        region, adds an arrow for each new edge, and scatters the samples.

    Returns:
        The roles drawn, empty when the step changed nothing visible.
    """
    roles: set[Role] = set()
    refined = [predicate_from_dict(c.after) for c in trace.vertex_changes]
    refined += [predicate_from_dict(v.region) for v in trace.added_vertices]
    for region in refined:
        roles |= _present(draw_region(ax, region, Role.REFINED, world, zorder=REFINED_ZORDER))
        roles |= _draw_boundaries(ax, region)
    for change in trace.safe_changes:
        roles |= _present(_draw_removed(ax, change, world))
    roles |= _draw_edges(ax, trace, graph)
    if show_samples:
        roles |= _draw_samples(ax, trace)
    return roles


def removed_safe_region(change: SafeChange) -> Predicate:
    """Return the area a safe-region change cut out.

    Args:
        change: The edge's before and after safe regions.

    How:
        AvoidRefine builds ``safe and not hull``, so when the new region has that
        form the removed area is exactly ``safe and hull``, which is convex and draws
        as a crisp polygon. Any other change is the general difference.

    Returns:
        The predicate satisfied by states that were safe and no longer are.
    """
    before, after = predicate_from_dict(change.before), predicate_from_dict(change.after)
    if (
        isinstance(after, And)
        and len(after.predicates) == 2
        and after.predicates[0] == before
        and isinstance(after.predicates[1], Not)
    ):
        return intersect(before, after.predicates[1].predicate)
    return And(before, Not(after))


def half_planes_in(region: Predicate) -> list[HalfPlane]:
    """Collect the half-plane atoms a region is cut by.

    Args:
        region: A region predicate.

    How:
        Walks ``and`` children, since a region refined by a split is the original
        intersected with the half-plane; other combinators are not searched.

    Returns:
        The half-planes found, possibly empty.
    """
    if isinstance(region, HalfPlane):
        return [region]
    if isinstance(region, And):
        return [plane for child in region.predicates for plane in half_planes_in(child)]
    return []


def _draw_removed(ax: Any, change: SafeChange, world: float) -> Role | None:
    """Hatch the area an AvoidRefine-style change removed from a safe region."""
    return draw_region(
        ax, removed_safe_region(change), Role.UNSAFE, world, alpha=0.2, hatch=TRAP_HATCH,
        zorder=REMOVED_ZORDER,
    )


def _present(role: Role | None) -> set[Role]:
    """Return the role as a set, empty when nothing was drawn."""
    return set() if role is None else {role}


def _draw_boundaries(ax: Any, region: Predicate) -> set[Role]:
    """Draw the dashed boundary line of every half-plane that cut ``region``."""
    planes = half_planes_in(region)
    for plane in planes:
        normal = plane.normal
        anchor = normal * (plane.offset / float(normal @ normal))
        direction = np.array([-normal[1], normal[0]])
        # Long enough to cross the whole world from any anchor; the axes clip the rest.
        start, stop = anchor - direction * 1e3, anchor + direction * 1e3
        ax.plot([start[0], stop[0]], [start[1], stop[1]], color=ROLE_COLOURS[Role.BOUNDARY],
                linestyle="--", linewidth=BOUNDARY_WIDTH, zorder=BOUNDARY_ZORDER)
    return {Role.BOUNDARY} if planes else set()


def _draw_edges(ax: Any, trace: StepTrace, graph: GraphSnapshot) -> set[Role]:
    """Draw an arrow between region centres for each edge the step added."""
    centres = {v.name: region_centre(predicate_from_dict(v.region)) for v in graph.vertices}
    drawn = False
    for edge in trace.added_edges:
        start, stop = centres.get(edge.source), centres.get(edge.target)
        if start is None or stop is None:
            continue
        ax.annotate(
            "", xy=tuple(stop), xytext=tuple(start), zorder=EDGE_ZORDER,
            arrowprops={"arrowstyle": "-|>", "color": ROLE_COLOURS[Role.EDGE],
                        "linewidth": ARROW_WIDTH, "shrinkA": 4, "shrinkB": 4},
        )
        drawn = True
    return {Role.EDGE} if drawn else set()


def _draw_samples(ax: Any, trace: StepTrace) -> set[Role]:
    """Scatter the visited states and the unsafe tails the edit was based on."""
    roles: set[Role] = set()
    for points, role, marker in (
        (trace.samples.witness_points, Role.SAMPLE, "o"),
        (trace.samples.unsafe_points, Role.UNSAFE, "x"),
    ):
        if points:
            xs, ys = zip(*points, strict=True)
            ax.scatter(xs, ys, s=SAMPLE_SIZE, marker=marker, color=ROLE_COLOURS[role],
                       alpha=SAMPLE_ALPHA, linewidths=0.6, zorder=SAMPLE_ZORDER)
            roles.add(role)
    return roles
