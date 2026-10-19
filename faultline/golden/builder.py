"""Build a gold specification from a layout, intended regions, and physical traps.

Gold is constructed from environment ground truth, never searched for. The caller
supplies an *intended* graph (named boxes and edges). The builder removes from each
region the cells that cannot take part in a successful edge, then records how much
it removed so a badly designed intended box is visible instead of silently fixed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from faultline.envs.rooms.env import MAX_SPEED, SUCCESS_RADIUS
from faultline.envs.rooms.grid import RoomGrid
from faultline.envs.rooms.layout import RoomsLayout
from faultline.golden.raster import CellGrid, Mask, largest_rectangle
from faultline.golden.spec import Certificate, GoldSpec, edge_id
from faultline.graph import AbstractGraph
from faultline.spectrl.predicates import And, Box, Not, Predicate, Universe

MAX_PASSES = 50
DEFAULT_MAX_SHRINK = 0.5
WAYPOINT_MARGIN = 0.1


def build_gold(
    name: str,
    layout: RoomsLayout,
    intended: AbstractGraph,
    traps: Sequence[Predicate] = (),
    *,
    resolution: int,
    max_steps: int,
    protocol_hash: str = "",
    max_shrink: float = DEFAULT_MAX_SHRINK,
) -> GoldSpec:
    """Build and Tier-1-check a gold spec.

    Args:
        name: Identifier for the gold spec.
        layout: The physical layout.
        intended: Intended graph; its edge safe sets are kept and combined with
            the trap exclusion. Source and target must be set.
        traps: Physical absorbing traps. Gold labels all of them unsafe.
        resolution: Cells per axis of the analysis grid (protocol value).
        max_steps: Episode horizon recorded on the spec.
        protocol_hash: Hash of the protocol file.
        max_shrink: Largest fraction of an intended region the builder may remove
            before Tier 1 fails; a larger loss means the intended box was mostly
            infeasible.

    How:
        Rasterises every region, then alternates two shrinking passes until nothing
        changes: a source keeps only cells with a safe route to the edge's target,
        a target keeps only cells reachable from the edge's source. After each
        fixpoint every region is cut to its largest inscribed rectangle, because a
        gold label must stay a box. Tier 1 is then checked exactly on the result:
        every source cell is viable for each of its edges, every target cell is
        reachable, and nothing is empty.

    Returns:
        A :class:`GoldSpec` whose certificate has only Tier 1 filled in.

    Raises:
        ValueError: If the intended graph has no source or target.
    """
    if intended.source is None or intended.target is None:
        raise ValueError("Intended graph needs a source and a target")
    grid = CellGrid(layout, resolution)
    trap_mask = _union_mask(grid, traps)
    edges = intended.edges
    safe_preds = {id(edge): _gold_safe(edge.safe, traps) for edge in edges}
    safe_masks = {
        id(edge): (grid.mask(edge.safe) if not isinstance(edge.safe, Universe) else _all(grid))
        & ~trap_mask
        for edge in edges
    }
    original = {vertex.name: grid.mask(vertex.region) for vertex in intended.vertices}
    masks = {key: value.copy() for key, value in original.items()}

    _fixpoint(grid, edges, masks, safe_masks)

    problems: list[str] = []
    shrinks: dict[str, float] = {}
    for vertex in intended.vertices:
        before, after = int(original[vertex.name].sum()), int(masks[vertex.name].sum())
        shrinks[vertex.name] = 0.0 if before == 0 else 1.0 - after / before
        if after == 0:
            problems.append(f"region {vertex.name!r} is empty after removing infeasible cells")
        elif shrinks[vertex.name] > max_shrink:
            problems.append(
                f"region {vertex.name!r} lost {shrinks[vertex.name]:.0%} of its cells "
                f"(limit {max_shrink:.0%})"
            )

    viability = {}
    for edge in edges:
        key = edge_id(edge.source.name, edge.target.name)
        safe = safe_masks[id(edge)]
        viability[key] = grid.viability(masks[edge.target.name] & safe, safe)
        source = masks[edge.source.name]
        if source.any() and not (source <= viability[key]).all():
            problems.append(f"edge {key} has source cells with no safe route to its target")
        reach = grid.reachable(source & safe, safe)
        if masks[edge.target.name].any() and not (masks[edge.target.name] <= reach).all():
            problems.append(f"edge {key} has target cells unreachable from its source")

    for edge in edges:
        needed = _required_steps(grid, masks[edge.source.name], _box_of(edge.target, grid, masks))
        if needed is not None and needed > max_steps:
            problems.append(
                f"edge {edge_id(edge.source.name, edge.target.name)} needs at least {needed} steps "
                f"but the horizon is {max_steps}"
            )

    graph = _gold_graph(grid, intended, masks, original, safe_preds)
    certificate = Certificate(
        tier1_ok=not problems, protocol_hash=protocol_hash, problems=tuple(problems)
    )
    return GoldSpec(
        id=name,
        env_id=name,
        layout=layout,
        traps=tuple(traps),
        graph=graph,
        viability=viability,
        shrinks=shrinks,
        resolution=resolution,
        max_steps=max_steps,
        certificate=certificate,
    )


def _box_of(vertex, grid: CellGrid, masks: dict[str, Mask]) -> Box | None:
    """Return the final box of a vertex, or ``None`` when it has no cells."""
    rectangle = largest_rectangle(masks[vertex.name])
    return None if rectangle is None else grid.box_of(rectangle)


def _required_steps(grid: CellGrid, sources: Mask, target: Box | None) -> int | None:
    """Return a lower bound on the steps the worst start cell needs.

    Args:
        grid: The cell grid.
        sources: The edge's source cells.
        target: The edge's target box.

    How:
        Straight-line distance from the farthest source cell centre to the target
        box, minus the success radius, divided by the top speed. Walls only make a
        real route longer, so this is a necessary condition: an edge it rejects
        cannot succeed from that start, however well it is trained.

    Returns:
        The step lower bound, or ``None`` when there are no sources or no target.
    """
    cells = np.argwhere(sources)
    if len(cells) == 0 or target is None:
        return None
    centres = (cells + 0.5) * grid.cell
    nearest = np.clip(centres, target.lower, target.upper)
    farthest = float(np.linalg.norm(centres - nearest, axis=1).max())
    return int(np.ceil(max(0.0, farthest - SUCCESS_RADIUS) / MAX_SPEED))


def _all(grid: CellGrid) -> Mask:
    """Return a mask covering every cell."""
    return np.ones((grid.resolution, grid.resolution), dtype=bool)


def _union_mask(grid: CellGrid, traps: Sequence[Predicate]) -> Mask:
    """Return the union of the trap masks, or an empty mask for no traps."""
    result = np.zeros((grid.resolution, grid.resolution), dtype=bool)
    for trap in traps:
        result |= grid.mask(trap)
    return result


def _gold_safe(safe: Predicate, traps: Sequence[Predicate]) -> Predicate:
    """Return an edge's safe predicate with every trap excluded.

    Args:
        safe: The intended safe predicate.
        traps: Physical traps gold labels unsafe.

    How:
        A universe safe set becomes ``Not(trap)`` (or an ``And`` of several); any
        other safe set is intersected with the same exclusions.

    Returns:
        The gold safe predicate.
    """
    exclusions = [Not(trap) for trap in traps]
    if not exclusions:
        return safe
    parts = ([] if isinstance(safe, Universe) else [safe]) + exclusions
    return parts[0] if len(parts) == 1 else And(*parts)


def _fixpoint(grid: CellGrid, edges, masks: dict[str, Mask], safe_masks: dict) -> None:
    """Shrink the vertex masks to a joint fixpoint, then to rectangles.

    Args:
        grid: The cell grid.
        edges: The intended edges.
        masks: Vertex name to mask; modified in place.
        safe_masks: Edge identity to safe mask.

    How:
        Repeats edge passes (sources by viability, targets by reachability) and a
        rectangle cut until a full round changes nothing. Masks only shrink, so
        the loop terminates; ``MAX_PASSES`` is a guard, not a tuning knob.
    """
    for _ in range(MAX_PASSES):
        changed = False
        for edge in edges:
            safe = safe_masks[id(edge)]
            source, target = edge.source.name, edge.target.name
            viable = grid.viability(masks[target] & safe, safe)
            reach = grid.reachable(masks[source] & safe, safe)
            new_source, new_target = masks[source] & viable, masks[target] & reach
            changed |= bool((new_source != masks[source]).any() or (new_target != masks[target]).any())
            masks[source], masks[target] = new_source, new_target
        for name, mask in masks.items():
            rectangle = largest_rectangle(mask)
            cut = np.zeros_like(mask)
            if rectangle is not None:
                ix0, iy0, ix1, iy1 = rectangle
                cut[ix0 : ix1 + 1, iy0 : iy1 + 1] = True
            if (cut != mask).any():
                masks[name] = cut
                changed = True
        if not changed:
            return


def _gold_graph(grid, intended, masks, original, safe_preds) -> AbstractGraph:
    """Assemble the gold graph from the shrunk masks.

    Args:
        grid: The cell grid.
        intended: The intended graph.
        masks: Final vertex masks.
        original: Intended vertex masks.
        safe_preds: Gold safe predicate per intended edge identity.

    How:
        A vertex whose mask equals its intended mask keeps the intended predicate
        unchanged, so an unshrunk gold has no rasterisation rounding. Otherwise the
        label is the box of the surviving rectangle.

    Returns:
        The gold graph with source and target set.
    """
    graph = AbstractGraph()
    vertices = {}
    for vertex in intended.vertices:
        mask = masks[vertex.name]
        if (mask == original[vertex.name]).all():
            region = vertex.region
        else:
            rectangle = largest_rectangle(mask)
            region = grid.box_of(rectangle) if rectangle is not None else Box((0, 0), (0, 0))
        vertices[vertex.name] = graph.add_vertex(vertex.name, region)
    for edge in intended.edges:
        graph.add_edge(
            vertices[edge.source.name], vertices[edge.target.name], safe_preds[id(edge)]
        )
    graph.source = vertices[intended.source.name]
    graph.target = vertices[intended.target.name]
    return graph


def rooms_of_box(box: Box, room_size: float) -> tuple[tuple[int, int], ...]:
    """Return the rooms a box overlaps.

    Args:
        box: An axis-aligned box.
        room_size: Width of one room.

    How:
        Maps the box's corners to room indices, treating the upper edge as
        exclusive so a box ending exactly on a wall does not claim the next room.

    Returns:
        The ``(col, row)`` rooms, row-major.
    """
    eps = 1e-9
    lo = np.floor(box.lower / room_size).astype(int)
    hi = np.floor((box.upper - eps) / room_size).astype(int)
    return tuple(
        (col, row)
        for row in range(lo[1], hi[1] + 1)
        for col in range(lo[0], hi[0] + 1)
    )


def cut_room_waypoints(layout: RoomsLayout, start: Box, goal: Box) -> tuple[Box, ...]:
    """Return waypoint boxes at the rooms every start-to-goal route must cross.

    Args:
        layout: The layout, which must use full-edge doors.
        start: The start box.
        goal: The goal box.

    How:
        Uses ``RoomGrid.cut_rooms`` and orders the cut rooms along one shortest
        path. A waypoint at a cut room removes no feasible route, which is why the
        spec places waypoints there. Each box is the room pulled in by a margin.

    Returns:
        Waypoint boxes from start to goal, possibly empty.
    """
    grid = RoomGrid(layout.rooms_per_side, layout.doors, layout.room_size)
    sources = rooms_of_box(start, layout.room_size)
    targets = rooms_of_box(goal, layout.room_size)
    cuts = grid.cut_rooms(sources, targets)
    path = grid.shortest_path(sources, targets)
    if path is None:
        return ()
    margin = WAYPOINT_MARGIN * layout.room_size
    size = layout.room_size
    return tuple(
        Box((col * size + margin, row * size + margin), ((col + 1) * size - margin, (row + 1) * size - margin))
        for col, row in path.rooms
        if (col, row) in cuts
    )


def chain_graph(start: Box, waypoints: Sequence[Box], goal: Box) -> AbstractGraph:
    """Build ``start -> w1 -> ... -> goal`` with a universe safe set on every edge.

    Args:
        start: The start box.
        waypoints: Waypoint boxes, in order.
        goal: The goal box.

    How:
        Names the vertices ``start``, ``w1``, ``w2``, ... and ``goal``.

    Returns:
        The intended chain graph.
    """
    graph = AbstractGraph()
    previous = graph.add_vertex("start", start)
    graph.source = previous
    for index, box in enumerate(waypoints, start=1):
        vertex = graph.add_vertex(f"w{index}", box)
        graph.add_edge(previous, vertex)
        previous = vertex
    end = graph.add_vertex("goal", goal)
    graph.add_edge(previous, end)
    graph.target = end
    return graph


def with_certificate(gold: GoldSpec, certificate: Certificate) -> GoldSpec:
    """Return ``gold`` carrying a new certificate.

    Args:
        gold: The gold spec.
        certificate: The certificate to attach.

    How:
        Replaces only the certificate field.

    Returns:
        A new :class:`GoldSpec`.
    """
    return replace(gold, certificate=certificate)
