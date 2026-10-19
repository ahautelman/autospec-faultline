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
WAYPOINT_SIDE = 0.375      # fraction of the room side
START_GOAL_SIDE = 0.75     # fraction of the room side (6 of 8 on rooms9)
ROOM_SIZE = 8.0
TRAP_LENGTH = 0.3          # fractions of the room side (2.4 and 0.8 on rooms9)
TRAP_WIDTH = 0.1


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


def _room_box(room: tuple[int, int], size: float, side_fraction: float) -> Box:
    cx, cy = (room[0] + 0.5) * size, (room[1] + 0.5) * size
    half = side_fraction * size / 2
    return Box((cx - half, cy - half), (cx + half, cy + half))


def route_chain(
    layout: RoomsLayout,
    start_room: tuple[int, int],
    goal_room: tuple[int, int],
    *,
    hop_rooms: int = 1,
    end_side: float = START_GOAL_SIDE,
) -> AbstractGraph | None:
    """Return the intended chain from ``start_room`` to ``goal_room`` along the door path.

    Args:
        layout: The layout.
        start_room: Room holding the source box.
        goal_room: Room holding the target box.
        hop_rooms: Rooms between consecutive waypoints; 1 gives one-room hops, which PPO
            learns reliably, larger values give harder edges.
        end_side: Side of the source and target boxes as a fraction of the room side.

    How:
        Takes the shortest door path (:func:`room_path`), puts a ``START_GOAL_SIDE`` box
        in the first and last room and a ``WAYPOINT_SIDE`` box (fractions of the room
        side) in every ``hop_rooms``-th intermediate room.

    Returns:
        The graph, or ``None`` when no door path exists.
    """
    rooms = room_path(layout, start_room, goal_room)
    if not rooms:
        return None
    size = layout.room_size
    names, boxes = ["start"], [_room_box(rooms[0], size, end_side)]
    for index, room in enumerate(rooms[1:-1], start=1):
        if index % hop_rooms == 0:
            names.append(f"w{len(names)}")
            boxes.append(_room_box(room, size, WAYPOINT_SIDE))
    names.append("goal")
    boxes.append(_room_box(rooms[-1], size, end_side))
    graph = AbstractGraph()
    vertices = [graph.add_vertex(n, b) for n, b in zip(names, boxes)]
    for left, right in zip(vertices, vertices[1:]):
        graph.add_edge(left, right)
    graph.source, graph.target = vertices[0], vertices[-1]
    return graph


# Per-layout box size where the paper-sized 6x6 boxes made the last hop unlearnable (Tier 2 failed
# on `default`: w3->goal lower bound 0.69-0.76 over 5 seeds). A design choice made after seeing
# certification results, reported in the plan.
ROOMS9_END_SIDE = {"default": 0.5}


def rooms9_chain(layout: RoomsLayout | None = None, end_side: float = START_GOAL_SIDE) -> AbstractGraph:
    """Return the rooms9 intended chain: corner start room to opposite corner goal room."""
    graph = route_chain(layout, (0, 0), (2, 2), end_side=end_side) if layout is not None else None
    if graph is not None:
        return graph
    regions = default_regions()
    fallback = AbstractGraph()
    vs = [fallback.add_vertex(n, regions[n]) for n in ("start", "mid", "goal")]
    fallback.add_edge(vs[0], vs[1]); fallback.add_edge(vs[1], vs[2])
    fallback.source, fallback.target = vs[0], vs[2]
    return fallback


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
                length, width, off = TRAP_LENGTH * size, TRAP_WIDTH * size, 0.3 * size
                for dx, dy in ((0.0, 0.0), (0.0, off), (0.0, -off), (off, 0.0), (-off, 0.0)):
                    cx, cy = (rx + 0.5) * size + dx, (ry + 0.5) * size + dy
                    found.append(Box((cx - length / 2, cy - width / 2), (cx + length / 2, cy + width / 2)))
                    found.append(Box((cx - width / 2, cy - length / 2), (cx + width / 2, cy + length / 2)))
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
    end_side = ROOMS9_END_SIDE.get(layout_name, START_GOAL_SIDE)
    graph = rooms9_chain(layout, end_side)
    name = f"rooms9_{layout_name}" + ("_trap" if trap else "")
    if not trap:
        return build_gold(name, layout, graph, resolution=resolution, max_steps=ROOMS9_MAX_STEPS,
                          protocol_hash=protocol_hash)
    for strip in trap_candidates(layout, graph):
        gold = build_gold(name, layout, rooms9_chain(layout, end_side), (strip,), resolution=resolution,
                          max_steps=ROOMS9_MAX_STEPS, protocol_hash=protocol_hash)
        if gold.certificate.tier1_ok:
            return gold
    return None


ROOMS100_MAX_STEPS = 30


def world_gold(world_name: str, *, resolution: int, protocol_hash: str = "", trap: bool = False,
               hop_rooms: int = 1) -> GoldSpec | None:
    """Build a gold on a fixed world's layout as a one-room-hop chain.

    Args:
        world_name: A fixed world from the catalogue (its layout is used, not its regions:
            the paper's regions deliberately include sealed rooms and are not gold).
        resolution: Cells per axis (a multiple of the rooms per side).
        protocol_hash: Protocol hash.
        trap: Whether to add a trap on the route.
        hop_rooms: Rooms per edge.

    How:
        The start room is the first room of the world's ``Start`` block and the goal room
        the first room of its ``Goal`` block that has a door path; the chain follows the
        shortest door path between them.

    Returns:
        The gold, or ``None`` when no door path or no feasible trap exists.
    """
    world = build_world(world_name)
    layout = world.layout
    starts, goals = world.regions["Start"].rooms(), world.regions["Goal"].rooms()
    pair = next(((a, b) for a in starts for b in goals if room_path(layout, a, b)), None)
    if pair is None:
        return None
    name = f"{world_name}" + ("_trap" if trap else "") + (f"_hop{hop_rooms}" if hop_rooms != 1 else "")
    graph = route_chain(layout, *pair, hop_rooms=hop_rooms)
    if not trap:
        return build_gold(name, layout, graph, resolution=resolution, max_steps=ROOMS100_MAX_STEPS,
                          protocol_hash=protocol_hash)
    for strip in trap_candidates(layout, graph, resolution):
        gold = build_gold(name, layout, route_chain(layout, *pair, hop_rooms=hop_rooms), (strip,),
                          resolution=resolution, max_steps=ROOMS100_MAX_STEPS, protocol_hash=protocol_hash)
        if gold.certificate.tier1_ok:
            return gold
    return None
