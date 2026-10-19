"""Gold specs for the benchmark layouts (rooms9 hand layouts, fixed rooms100 worlds)."""

from __future__ import annotations

from faultline.envs.layout import (
    blocked_two_path_doors, long_mixed_star_doors,
)
from faultline.envs.rooms.catalogue import build_world
from faultline.envs.rooms.layout import RoomsLayout
from faultline.envs.rooms9 import default_doors, default_regions
from faultline.golden.builder import build_gold
from faultline.golden.spec import GoldSpec
from faultline.graph import AbstractGraph
from faultline.spectrl.predicates import Box

# ``long_corridor`` is excluded: its intended start/mid/goal chain is infeasible on that layout
# (Tier 1 empties start and mid); it needs waypoints at cut vertices, not a fixed chain.
ROOMS9_LAYOUTS = {
    "default": default_doors,
    "mixed_star": long_mixed_star_doors,
    "blocked_two_path": blocked_two_path_doors,
}
ROOMS9_MAX_STEPS = 30
WAYPOINT_SIDE = 3.0
ROOM_SIZE = 8.0
TRAP_LENGTH = 2.4
TRAP_WIDTH = 0.8


def room_path(layout: RoomsLayout, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Return a shortest room path through open doors (breadth-first, ties broken by room order).

    How:
        A door key ``("v", c, r)`` joins rooms ``(c, r)`` and ``(c + 1, r)``; ``("h", c, r)``
        joins ``(c, r)`` and ``(c, r + 1)``. Partial doors count as open; whether they
        are passable for the agent is Tier 1's question, not this function's.

    Returns:
        The rooms from ``start`` to ``goal``, or an empty list when unreachable.
    """
    from collections import deque

    links: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (kind, c, r) in layout.doors:
        a, b = ((c, r), (c + 1, r)) if kind == "v" else ((c, r), (c, r + 1))
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    parent = {start: None}
    queue = deque([start])
    while queue:
        room = queue.popleft()
        if room == goal:
            break
        for nxt in sorted(links.get(room, [])):
            if nxt not in parent:
                parent[nxt] = room
                queue.append(nxt)
    if goal not in parent:
        return []
    path, room = [], goal
    while room is not None:
        path.append(room)
        room = parent[room]
    return path[::-1]


def rooms9_chain(layout: RoomsLayout | None = None) -> AbstractGraph:
    """Return the intended graph: ``start``, one waypoint per intermediate room, ``goal``.

    How:
        Start and goal use the rooms9 default boxes. Every room strictly between them on
        the shortest door path gets a ``WAYPOINT_SIDE`` box at its centre, so each edge is
        a one-room hop that PPO learns reliably. A layout with no door path gets the
        old ``start -> mid -> goal`` chain.
    """
    regions = default_regions()
    names = ["start"]
    boxes = [regions["start"]]
    rooms = room_path(layout, (0, 0), (2, 2)) if layout is not None else []
    if rooms:
        for index, (c, r) in enumerate(rooms[1:-1], start=1):
            cx, cy = (c + 0.5) * ROOM_SIZE, (r + 0.5) * ROOM_SIZE
            half = WAYPOINT_SIDE / 2
            names.append(f"w{index}")
            boxes.append(Box((cx - half, cy - half), (cx + half, cy + half)))
    else:
        names.append("mid")
        boxes.append(regions["mid"])
    names.append("goal")
    boxes.append(regions["goal"])
    graph = AbstractGraph()
    vertices = [graph.add_vertex(n, b) for n, b in zip(names, boxes)]
    for left, right in zip(vertices, vertices[1:]):
        graph.add_edge(left, right)
    graph.source, graph.target = vertices[0], vertices[-1]
    return graph


def trap_candidates(layout: RoomsLayout, graph: AbstractGraph, resolution: int = 48) -> list[Box]:
    """Return candidate trap strips in rooms that lie on a feasible route but hold no endpoint.

    How:
        Rasterises the layout (partial doors are fine), takes for each edge the cells that
        are both reachable from its source region and viable to its target region, and
        lists the rooms holding such cells but neither region's centre. Each yields one
        strip along ``x`` and one along ``y`` at the room centre. Candidates are only
        proposals, ordered with the room-centre strip last-resort offsets: the gold builder decides whether the world stays feasible.
    """
    import numpy as np

    from faultline.golden.raster import CellGrid

    grid = CellGrid(layout, resolution)
    size = layout.room_size
    n = layout.rooms_per_side
    per_room = resolution // n
    everywhere = np.ones((resolution, resolution), dtype=bool)
    found: list[Box] = []
    seen: set[tuple[int, int]] = set()
    for edge in graph.edges:
        src, dst = grid.mask(edge.source.region), grid.mask(edge.target.region)
        on_route = grid.reachable(src, everywhere) & grid.viability(dst, everywhere)
        ends = {(int(c[0] // per_room), int(c[1] // per_room)) for m in (src, dst) for c in np.argwhere(m)}
        for rx in range(n):
            for ry in range(n):
                block = on_route[rx * per_room:(rx + 1) * per_room, ry * per_room:(ry + 1) * per_room]
                if (rx, ry) in ends or (rx, ry) in seen or not block.any():
                    continue
                seen.add((rx, ry))
                for dx, dy in ((0.0, 0.0), (0.0, 2.4), (0.0, -2.4), (2.4, 0.0), (-2.4, 0.0)):
                    cx, cy = (rx + 0.5) * size + dx, (ry + 0.5) * size + dy
                    found.append(Box((cx - TRAP_LENGTH / 2, cy - TRAP_WIDTH / 2), (cx + TRAP_LENGTH / 2, cy + TRAP_WIDTH / 2)))
                    found.append(Box((cx - TRAP_WIDTH / 2, cy - TRAP_LENGTH / 2), (cx + TRAP_WIDTH / 2, cy + TRAP_LENGTH / 2)))
    return found


def rooms9_gold(layout_name: str, *, resolution: int, protocol_hash: str = "", trap: bool = False) -> GoldSpec | None:
    """Build the gold of one rooms9 layout, optionally with a physical trap.

    Args:
        layout_name: A key of :data:`ROOMS9_LAYOUTS`.
        resolution: Cells per axis.
        protocol_hash: Protocol hash stamped on the certificate.
        trap: Whether to add a trap on the route.

    How:
        With ``trap`` the first candidate strip for which Tier 1 passes is used.

    Returns:
        The gold, or ``None`` when ``trap`` is set and no candidate is feasible.
    """
    layout = RoomsLayout(doors=ROOMS9_LAYOUTS[layout_name]())
    graph = rooms9_chain(layout)
    name = f"rooms9_{layout_name}" + ("_trap" if trap else "")
    if not trap:
        return build_gold(name, layout, graph, resolution=resolution, max_steps=ROOMS9_MAX_STEPS,
                          protocol_hash=protocol_hash)
    for strip in trap_candidates(layout, graph):
        gold = build_gold(name, layout, rooms9_chain(layout), (strip,), resolution=resolution,
                          max_steps=ROOMS9_MAX_STEPS, protocol_hash=protocol_hash)
        if gold.certificate.tier1_ok:
            return gold
    return None


def world_gold(world_name: str, *, resolution: int, protocol_hash: str = "") -> GoldSpec:
    """Build the gold of a fixed rooms100 (or figure) world from its intended topology."""
    world = build_world(world_name)
    return build_gold(world_name, world.layout, world.build_graph(), resolution=resolution,
                      max_steps=world.effective_max_steps(), protocol_hash=protocol_hash)
