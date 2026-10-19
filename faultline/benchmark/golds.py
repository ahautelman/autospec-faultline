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
ROOMS9_MAX_STEPS = 60
TRAP_LENGTH = 2.4
TRAP_WIDTH = 0.8


def rooms9_chain() -> AbstractGraph:
    """Return the intended ``start -> mid -> goal`` graph on the default regions."""
    regions = default_regions()
    graph = AbstractGraph()
    vertices = [graph.add_vertex(name, regions[name]) for name in ("start", "mid", "goal")]
    graph.add_edge(vertices[0], vertices[1])
    graph.add_edge(vertices[1], vertices[2])
    graph.source, graph.target = vertices[0], vertices[2]
    return graph


def trap_candidates(layout: RoomsLayout, graph: AbstractGraph, resolution: int = 48) -> list[Box]:
    """Return candidate trap strips in rooms that lie on a feasible route but hold no endpoint.

    How:
        Rasterises the layout (partial doors are fine), takes for each edge the cells that
        are both reachable from its source region and viable to its target region, and
        lists the rooms holding such cells but neither region's centre. Each yields one
        strip along ``x`` and one along ``y`` at the room centre. Candidates are only
        proposals: the gold builder decides whether the world stays feasible.
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
                cx, cy = (rx + 0.5) * size, (ry + 0.5) * size
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
    graph = rooms9_chain()
    name = f"rooms9_{layout_name}" + ("_trap" if trap else "")
    if not trap:
        return build_gold(name, layout, graph, resolution=resolution, max_steps=ROOMS9_MAX_STEPS,
                          protocol_hash=protocol_hash)
    for strip in trap_candidates(layout, graph):
        gold = build_gold(name, layout, rooms9_chain(), (strip,), resolution=resolution,
                          max_steps=ROOMS9_MAX_STEPS, protocol_hash=protocol_hash)
        if gold.certificate.tier1_ok:
            return gold
    return None


def world_gold(world_name: str, *, resolution: int, protocol_hash: str = "") -> GoldSpec:
    """Build the gold of a fixed rooms100 (or figure) world from its intended topology."""
    world = build_world(world_name)
    return build_gold(world_name, world.layout, world.build_graph(), resolution=resolution,
                      max_steps=world.effective_max_steps(), protocol_hash=protocol_hash)
