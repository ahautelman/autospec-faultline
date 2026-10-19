"""Draw a 9-rooms scenario from its layout, graph and hidden trap.

Takes only persisted data (a :class:`RoomsLayout`, a :class:`GraphSnapshot` and an
optional trap predicate), so any scenario or saved run is drawn the same way: walls with
door gaps, the graph's regions by role, and the trap if the environment hides one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from faultline.envs.layout import RoomsLayout
from faultline.geometry import predicate_from_dict
from faultline.graph.snapshot import GraphSnapshot, VertexSnapshot, snapshot_graph
from faultline.spectrl.predicates import Predicate
from faultline.viz.geometry import draw_region, region_centre
from faultline.viz.style import TEXT_COLOUR, WALL_COLOUR, WALL_WIDTH, Role

if TYPE_CHECKING:
    from faultline.envs.rooms.verify import VerificationReport
    from faultline.envs.rooms.world import RoomsWorld

REGION_ZORDER = 1.0
TRAP_ZORDER = 2.0
WALL_ZORDER = 4.0
LABEL_ZORDER = 5.0
TRAP_HATCH = "////"
LABEL_SIZE = 6.5
COMPACT_LABEL_SIZE = 11.0
TITLE_SIZE = 7.0
COMPACT_TITLE_SIZE = 12.0
# A faint white plate keeps a label legible where a wall runs through its region.
LABEL_BOX = {"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8}
Segment = tuple[tuple[float, float], tuple[float, float]]


def draw_room(
    ax: Any,
    layout: RoomsLayout,
    graph: GraphSnapshot,
    unsafe_region: Predicate | None = None,
    *,
    compact: bool = False,
) -> set[Role]:
    """Draw the room, the graph's regions and the hidden trap.

    Args:
        ax: Target axes.
        layout: The wall and door model.
        graph: The graph whose vertex regions are shaded.
        unsafe_region: The trap the environment hides, or ``None``.
        compact: Whether to draw a small-multiple panel: short region tags in a
            larger font and no axis ticks or labels. The room grid is kept.

    How:
        Shades each vertex region by its role (the graph source as start, the graph
        target as goal, any other vertex as a waypoint) and labels it with the
        vertex name (or its short tag when compact), hatches the trap, then draws
        the outer wall and the internal walls with their door gaps and sets
        room-aligned axes.

    Returns:
        The roles that were drawn, for the legend.
    """
    drawn = [
        _draw_vertex(ax, vertex, graph, layout.world_size, compact)
        for vertex in graph.vertices
    ]
    if unsafe_region is not None:
        drawn.append(
            draw_region(ax, unsafe_region, Role.UNSAFE, layout.world_size, alpha=0.18,
                        hatch=TRAP_HATCH, zorder=TRAP_ZORDER)
        )
    roles = {role for role in drawn if role is not None}
    _draw_walls(ax, layout)
    _configure_axes(ax, layout, compact)
    return roles


def draw_world(
    ax: Any,
    world: RoomsWorld,
    report: VerificationReport | None = None,
    *,
    compact: bool = False,
) -> set[Role]:
    """Draw an n-rooms world, its regions and a door-count/verdict caption.

    Args:
        ax: Target axes.
        world: The world to draw. Its region names are used as labels, so the
            plan's ``M3``/``M4``/``M5`` appear rather than the paper's
            ``MG``/``MG1``/``MG2``.
        report: A heuristic verification report, or ``None`` to caption only the
            door count. When given, the caption states pass/fail, the rejection
            count stored in ``world.meta``, and how many DAG routes survived.
        compact: Whether to draw a small-multiple panel: short region tags, no
            axis ticks, and a short title (``seed 7 · PASS``) instead of the full
            caption.

    How:
        Builds the specification graph from the world, snapshots it the same way
        a saved run's graph is snapshotted, and delegates the drawing to
        :func:`draw_room`. The caption is set as the axes title, so a contact
        sheet can read it back with ``ax.get_title()``.

    Returns:
        The roles drawn, for a shared legend.
    """
    snapshot = snapshot_graph(world.build_graph())
    roles = draw_room(ax, world.layout, snapshot, compact=compact)
    if compact:
        ax.set_title(compact_caption(world, report), fontsize=COMPACT_TITLE_SIZE)
    else:
        ax.set_title(world_caption(world, report), fontsize=TITLE_SIZE)
    return roles


def world_caption(world: RoomsWorld, report: VerificationReport | None = None) -> str:
    """Return the one-line caption used by :func:`draw_world`.

    Args:
        world: The world being drawn.
        report: A heuristic verification report, or ``None``.

    How:
        Without a report the caption is the world's name and door count. With a
        report it adds a pass/fail verdict, the generator's rejection count when
        present, and the number of feasible Start-to-Goal routes.

    Returns:
        The caption text.
    """
    parts = [world.name, f"{len(world.layout.doors)} doors"]
    if report is not None:
        parts.insert(1, "PASS" if report.solvable else "FAIL")
        attempts = world.meta.get("attempts")
        if attempts is not None:
            parts.append(f"{attempts} attempts")
        parts.append(f"{len(report.routes)} routes")
    return " · ".join(parts)


def compact_caption(world: RoomsWorld, report: VerificationReport | None = None) -> str:
    """Return the short panel title used by compact drawings.

    Args:
        world: The world being drawn.
        report: A heuristic verification report, or ``None``.

    How:
        Names a generated world by its seed (``seed 7``) and any other world by
        its name, then appends the verdict when a report is given. Door and route
        counts are left to the summary table.

    Returns:
        The caption text.
    """
    seed = world.meta.get("seed")
    label = f"seed {seed}" if seed is not None else world.name
    if report is None:
        return label
    return f"{label} · {'PASS' if report.solvable else 'FAIL'}"


def region_tag(name: str) -> str:
    """Return the one-character tag of a region name.

    Args:
        name: A region name such as ``Start``, ``M3`` or ``Goal``.

    How:
        ``Start`` and ``Goal`` become ``S`` and ``G``, a numbered region ``M3``
        becomes ``3``, and any other name is kept whole.

    Returns:
        The short tag.
    """
    if name in ("Start", "Goal"):
        return name[0]
    if len(name) > 1 and name[0] == "M" and name[1:].isdigit():
        return name[1:]
    return name


def _draw_vertex(
    ax: Any, vertex: VertexSnapshot, graph: GraphSnapshot, world: float, compact: bool = False
) -> Role | None:
    """Shade one vertex region by its role and label it with its name or tag."""
    role = _vertex_role(vertex.name, graph)
    region = predicate_from_dict(vertex.region)
    drawn = draw_region(ax, region, role, world, zorder=REGION_ZORDER)
    centre = region_centre(region)
    if centre is not None:
        label = region_tag(vertex.name) if compact else vertex.name
        size = COMPACT_LABEL_SIZE if compact else LABEL_SIZE
        ax.text(*centre, label, ha="center", va="center", fontsize=size,
                color=TEXT_COLOUR, zorder=LABEL_ZORDER, bbox=LABEL_BOX)
    return drawn


def _vertex_role(name: str, graph: GraphSnapshot) -> Role:
    """Return the role of the vertex with this name."""
    if name == graph.source:
        return Role.START
    if name == graph.target:
        return Role.GOAL
    return Role.WAYPOINT


def _draw_walls(ax: Any, layout: RoomsLayout) -> None:
    """Draw the outer boundary and every internal wall with its door gaps."""
    from matplotlib.collections import LineCollection

    ax.add_collection(
        LineCollection(
            [*_outer_segments(layout), *_internal_segments(layout)],
            colors=WALL_COLOUR,
            linewidths=WALL_WIDTH,
            capstyle="projecting",
            zorder=WALL_ZORDER,
            clip_on=False,
        )
    )


def _outer_segments(layout: RoomsLayout) -> list[Segment]:
    """Return the four sides of the world boundary."""
    size = layout.world_size
    corners = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    return [(corners[i], corners[(i + 1) % 4]) for i in range(4)]


def _internal_segments(layout: RoomsLayout) -> list[Segment]:
    """Return the wall pieces on every internal grid line, leaving door gaps."""
    segments: list[Segment] = []
    for line_index in range(1, layout.rooms_per_side):
        line = line_index * layout.room_size
        for room in range(layout.rooms_per_side):
            low, high = room * layout.room_size, (room + 1) * layout.room_size
            door = layout.doors.get(("v", line_index - 1, room))
            segments += [((line, a), (line, b)) for a, b in _solid_pieces(low, high, door)]
            door = layout.doors.get(("h", room, line_index - 1))
            segments += [((a, line), (b, line)) for a, b in _solid_pieces(low, high, door)]
    return segments


def _solid_pieces(low: float, high: float, door: tuple[float, float] | None) -> list[tuple[float, float]]:
    """Split a wall run ``[low, high]`` around a door interval."""
    if door is None:
        return [(low, high)]
    return [piece for piece in ((low, door[0]), (door[1], high)) if piece[1] > piece[0]]


def _configure_axes(ax: Any, layout: RoomsLayout, compact: bool = False) -> None:
    """Fix the world bounds and equal aspect; add ticks and labels unless compact."""
    ax.set_xlim(0.0, layout.world_size)
    ax.set_ylim(0.0, layout.world_size)
    ax.set_aspect("equal")
    for side in ax.spines.values():
        side.set_visible(False)
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ticks = [index * layout.room_size for index in range(layout.rooms_per_side + 1)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.tick_params(length=2.5)
