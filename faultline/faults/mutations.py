"""The five parametric mutation families (spec 6.1), applied to a certified gold spec.

Every mutation is a pure function ``(gold, params, rng) -> GroundTruthRepair``,
deterministic given the generator's seed, and coarsening only: the faulty spec
accepts everything the gold spec accepts. The validity gate checks that claim
mechanically; nothing here assumes it.
"""

from __future__ import annotations

import copy

import numpy as np

from faultline.faults.graph_ops import copy_without_vertex, edge_named, union, vertex_named
from faultline.faults.ground_truth import (
    EXPECTED_OPERATORS,
    FaultKind,
    GroundTruthRepair,
    Locus,
    MutationNotApplicable,
    MutationParams,
    severity_value,
)
from faultline.faults.placement import find_rectangle
from faultline.geometry import convex_hull, half_plane_partition, region_bounds
from faultline.golden.builder import alternative_paths
from faultline.golden.raster import CellGrid, Mask
from faultline.golden.spec import GoldSpec, edge_id
from faultline.graph import AbstractGraph, Edge, Vertex
from faultline.spectrl.predicates import PLANAR_DIMS, And, Box, Not, Predicate

_HULL_SAMPLE = 600
OPPOSITE_COSINE = 0.95


def mutate_goal_inflation(
    gold: GoldSpec, params: MutationParams, rng: np.random.Generator, protocol=None
) -> GroundTruthRepair:
    """Goal inflation: add a sealed pocket to a target region (ReachRefine).

    Args:
        gold: The certified gold spec.
        params: Mutation parameters; ``vertex`` defaults to the graph target.
        rng: Breaks ties between equally good pocket sites.
        protocol: Optional protocol whose severity grid overrides the defaults.

    How:
        Finds the cells no in-edge can reach under its safe set, outside every
        trap, and places a rectangle of ``rho / (1 - rho)`` times the region's cell
        count as close to the region as possible. The faulty label is the union.
        Goals sampled in the pocket can never be reached, which is the symptom.

    Returns:
        The mutant.

    Raises:
        MutationNotApplicable: If the vertex has no in-edge or no sealed pocket fits.
    """
    graph = gold.graph
    name = params.vertex or graph.target.name
    vertex = vertex_named(graph, name)
    incoming = graph.predecessors(vertex)
    if not incoming:
        raise MutationNotApplicable(f"vertex {name!r} has no incoming edge")
    grid = CellGrid(gold.layout, gold.resolution)
    reached = np.zeros((grid.resolution, grid.resolution), dtype=bool)
    for edge in incoming:
        safe = grid.mask(edge.safe)
        reached |= grid.reachable(grid.mask(edge.source.region) & safe, safe)
    base = grid.mask(vertex.region)
    allowed = ~reached & ~_trap_mask(grid, gold)
    rho = severity_value(FaultKind.GOAL_INFLATION, params, protocol)
    cells = max(1, round(rho / (1.0 - rho) * int(base.sum())))
    anchor = _bounds_box(vertex.region)
    rectangle = find_rectangle(allowed, cells, anchor, grid, rng)
    if rectangle is None:
        raise MutationNotApplicable(f"no sealed pocket of {cells} cells near {name!r}")
    pocket = grid.box_of(rectangle)
    faulty = graph.copy()
    faulty_vertex = vertex_named(faulty, name)
    faulty_vertex.region = union(faulty_vertex.region, pocket)
    pocket_cells = _rectangle_cells(rectangle)
    factors = {
        "pocket_cells": pocket_cells,
        "mass": pocket_cells / (pocket_cells + int(base.sum())),
        "pocket": _box_tuple(pocket),
        "hull_recoverable": _hull_recoverable(grid, base & reached, pocket),
    }
    return _record(
        gold, faulty, FaultKind.GOAL_INFLATION, Locus("vertex", name), params, protocol,
        factors, {Locus("vertex", name): vertex.region},
    )


def mutate_hazard_unlabeled(
    gold: GoldSpec, params: MutationParams, rng: np.random.Generator, protocol=None
) -> GroundTruthRepair:
    """Hazard unlabeled: label only the far part of a physical trap unsafe (AvoidRefine).

    Args:
        gold: The certified gold spec, with at least one box trap.
        params: Mutation parameters; ``edge`` defaults to the edge whose straight
            route passes closest to the trap, ``trap_index`` picks the trap.
        rng: Unused; the placement is deterministic.
        protocol: Optional protocol whose severity grid overrides the defaults.

    How:
        Cuts the trap box along the axis the route travels, on a cell boundary.
        The part facing the edge's source (the approach) is left unlabelled and
        the far part (the core) stays labelled, so the faulty safe set is the gold
        one plus the approach. The trap itself stays in the environment, so the
        agent still dies in it, and AvoidRefine sees rollouts ending there.

    Returns:
        The mutant.

    Raises:
        MutationNotApplicable: If there is no usable box trap or edge.
    """
    if not gold.traps or params.trap_index >= len(gold.traps):
        raise MutationNotApplicable("gold has no such trap")
    trap = gold.traps[params.trap_index]
    if not isinstance(trap, Box):
        raise MutationNotApplicable("only box traps can be partly unlabelled")
    graph = gold.graph
    edge = _trap_edge(graph, trap, params)
    cell = gold.layout.world_size / gold.resolution
    source_c, target_c = _centre(edge.source.region), _centre(edge.target.region)
    direction = target_c - source_c
    axis = int(np.argmax(np.abs(direction)))
    length = int(round((trap.upper[axis] - trap.lower[axis]) / cell))
    if length < 2:
        raise MutationNotApplicable("trap is too thin to split")
    fraction = severity_value(FaultKind.HAZARD_UNLABELED, params, protocol)
    cut = int(min(max(round(fraction * length), 1), length - 1))
    low, high = trap.lower.copy(), trap.upper.copy()
    core_lower, core_upper = low.copy(), high.copy()
    if direction[axis] >= 0:  # source on the low side: approach is the low part
        core_lower[axis] = low[axis] + cut * cell
    else:
        core_upper[axis] = high[axis] - cut * cell
    core = Box(core_lower, core_upper)
    faulty = graph.copy()
    faulty_edge = edge_named(faulty, edge.source.name, edge.target.name)
    relabelled = _relabel(faulty_edge.safe, trap, core)
    if relabelled is None:
        raise MutationNotApplicable("edge safe set does not label the trap")
    faulty_edge.safe = relabelled
    key = edge_id(edge.source.name, edge.target.name)
    factors = {
        "unlabeled_fraction": cut / length,
        "unlabeled_cells": cut * int(round((trap.upper[1 - axis] - trap.lower[1 - axis]) / cell)),
        "axis": "xy"[axis],
        "trap": _box_tuple(trap),
        "core": _box_tuple(core),
        "trap_path_distance": _segment_box_distance(source_c, target_c, trap),
    }
    return _record(
        gold, faulty, FaultKind.HAZARD_UNLABELED, Locus("edge", key), params, protocol,
        factors, {Locus("edge", key): edge.safe},
    )


def mutate_dead_start(
    gold: GoldSpec, params: MutationParams, rng: np.random.Generator, protocol=None
) -> GroundTruthRepair:
    """Dead start: add start cells from which the outgoing edge cannot succeed (PastRefine).

    Args:
        gold: The certified gold spec.
        params: Mutation parameters; ``vertex`` defaults to the graph source and
            ``pockets`` is 1 or 2 (two pockets on opposite sides defeat a single
            half-plane).
        rng: Breaks ties between equally good sites.
        protocol: Optional protocol whose severity grid overrides the defaults.

    How:
        Pocket cells are not viable for any outgoing edge, lie outside every trap,
        and, for a vertex with in-edges, are reachable from upstream. Reachable but
        not viable needs the in- and out-edges to have different safe sets (a
        spec-only hazard on the outgoing edge), which is why this differs from goal inflation.
        The faulty label is the gold region plus the pocket or pockets.

    Returns:
        The mutant.

    Raises:
        MutationNotApplicable: If the vertex has no out-edge or no pocket fits.
    """
    graph = gold.graph
    name = params.vertex or graph.source.name
    vertex = vertex_named(graph, name)
    outgoing = graph.successors(vertex)
    if not outgoing:
        raise MutationNotApplicable(f"vertex {name!r} has no outgoing edge")
    grid = CellGrid(gold.layout, gold.resolution)
    size = (grid.resolution, grid.resolution)
    viable = np.zeros(size, dtype=bool)
    for edge in outgoing:
        viable |= gold.viability[edge_id(edge.source.name, edge.target.name)]
    allowed = ~viable & ~_trap_mask(grid, gold)
    incoming = graph.predecessors(vertex)
    if incoming:
        reached = np.zeros(size, dtype=bool)
        for edge in incoming:
            safe = grid.mask(edge.safe)
            reached |= grid.reachable(grid.mask(edge.source.region) & safe, safe)
        allowed &= reached
    base = grid.mask(vertex.region)
    rho = severity_value(FaultKind.DEAD_START, params, protocol)
    total = max(params.pockets, round(rho / (1.0 - rho) * int(base.sum())))
    anchor = _bounds_box(vertex.region)
    pockets = _place_pockets(grid, allowed, total, params.pockets, anchor, rng)
    faulty = graph.copy()
    faulty_vertex = vertex_named(faulty, name)
    for rectangle in pockets:
        faulty_vertex.region = union(faulty_vertex.region, grid.box_of(rectangle))
    pocket_cells = sum(_rectangle_cells(r) for r in pockets)
    factors = {
        "pocket_cells": pocket_cells,
        "mass": pocket_cells / (pocket_cells + int(base.sum())),
        "pockets": tuple(_box_tuple(grid.box_of(r)) for r in pockets),
        "halfplane_separable": _halfplane_separable(grid, pockets, base),
        "has_upstream": bool(incoming),
    }
    return _record(
        gold, faulty, FaultKind.DEAD_START, Locus("vertex", name), params, protocol,
        factors, {Locus("vertex", name): vertex.region},
    )


def mutate_waypoint_contraction(
    gold: GoldSpec, params: MutationParams, rng: np.random.Generator, protocol=None
) -> GroundTruthRepair:
    """Waypoint contraction: drop one or two waypoints and join their neighbours (AddRefine).

    Args:
        gold: The certified gold spec.
        params: Mutation parameters; ``n_dropped`` is 1 or 2 and the level sets
            the target position of the dropped waypoint along the route.
        rng: Unused; the choice is deterministic.
        protocol: Optional protocol whose severity grid overrides the defaults.

    How:
        A droppable waypoint has exactly one in-edge and one out-edge with equal
        safe sets, so the joined edge keeps the same safety condition. The chain
        whose centre is closest to the target position along its route is dropped.

    Returns:
        The mutant.

    Raises:
        MutationNotApplicable: If no chain of ``n_dropped`` waypoints exists.
    """
    graph = gold.graph
    target_position = severity_value(FaultKind.WAYPOINT_CONTRACTION, params, protocol)
    chains = _waypoint_chains(graph, params.n_dropped)
    if not chains:
        raise MutationNotApplicable(f"no chain of {params.n_dropped} droppable waypoints")
    scored = [(round(abs(_position(graph, chain) - target_position), 9), index) for index, chain in enumerate(chains)]
    _, chosen = min(scored)
    chain = chains[chosen]
    before = graph.predecessors(vertex_named(graph, chain[0]))[0]
    after = graph.successors(vertex_named(graph, chain[-1]))[0]
    faulty = graph
    for dropped in chain:
        faulty = copy_without_vertex(faulty, dropped)
    if _has_edge(faulty, before.source.name, after.target.name):
        raise MutationNotApplicable("joined edge already exists")
    faulty.add_edge(
        vertex_named(faulty, before.source.name),
        vertex_named(faulty, after.target.name),
        copy.deepcopy(before.safe),
    )
    faulty.assert_acyclic()
    position = _position(graph, chain)
    factors = {
        "dropped": chain,
        "position": position,
        "witness_aligned": abs(position - 0.5),
        "route_edges": _spine_edges(graph, chain),
    }
    locus = Locus("vertex", chain[0])
    return _record(
        gold, faulty, FaultKind.WAYPOINT_CONTRACTION, locus, params, protocol, factors,
        {Locus("vertex", name): vertex_named(graph, name).region for name in chain},
    )


def mutate_route_pruning(
    gold: GoldSpec, params: MutationParams, rng: np.random.Generator, protocol=None
) -> GroundTruthRepair:
    """Route pruning: remove the detour edge that makes a weak-by-design edge avoidable (OrRefine).

    Args:
        gold: The certified gold spec, with at least one weak-by-design edge.
        params: Mutation parameters; ``edge`` picks the edge to remove.
        rng: Breaks ties between equally good candidates.
        protocol: Unused.

    How:
        Candidates are edges on an alternative route of a weak edge whose removal
        leaves that weak edge with no alternative route of non-weak edges. The
        faulty graph is the gold graph without the edge. Whether the faulty spec
        still accepts every gold trajectory is the language gate's job: this
        mutant is a learnability fault and is discarded there if it is stricter.

    Returns:
        The mutant.

    Raises:
        MutationNotApplicable: If no single edge removal cuts every alternative.
    """
    graph = gold.graph
    weak = gold.weak_edges
    if not weak:
        raise MutationNotApplicable("gold has no weak-by-design edge")
    candidates = []
    for weak_key in sorted(weak):
        source, target = weak_key.split("->")
        for route in alternative_paths(graph, source, target, weak, weak_key):
            for key in route:
                if params.edge is not None and key != params.edge:
                    continue
                trial = graph.copy()
                trial.remove_edge(edge_named(trial, *key.split("->")))
                if not alternative_paths(trial, source, target, weak, weak_key):
                    candidates.append((key, weak_key))
    candidates = sorted(set(candidates))
    if not candidates:
        raise MutationNotApplicable("no single edge removal cuts every alternative route")
    key, weak_key = candidates[int(rng.integers(len(candidates)))]
    faulty = graph.copy()
    removed = edge_named(faulty, *key.split("->"))
    gold_edge = edge_named(graph, *key.split("->"))
    faulty.remove_edge(removed)
    faulty.assert_acyclic()
    factors = {"removed": key, "weak_edge": weak_key, "alternatives_before": len(
        alternative_paths(graph, *weak_key.split("->"), weak, weak_key))}
    return _record(
        gold, faulty, FaultKind.ROUTE_PRUNING, Locus("edge", key), params, protocol,
        factors, {Locus("edge", key): gold_edge.safe},
    )


def null_mutation(gold: GoldSpec) -> GroundTruthRepair:
    """Return the null instance: the faulty spec equals gold.

    Args:
        gold: The certified gold spec.

    How:
        A copy of the gold graph, used to measure the false-repair rate.

    Returns:
        The null mutant.
    """
    return GroundTruthRepair(
        gold=gold.graph, faulty=gold.graph.copy(), kind=FaultKind.NULL, locus=Locus("none"),
        severity_level="fixed", severity=None, factors={}, target={},
        expected_operators=EXPECTED_OPERATORS[FaultKind.NULL], weak_edges=gold.weak_edges,
    )


MUTATIONS = {
    FaultKind.GOAL_INFLATION: mutate_goal_inflation,
    FaultKind.HAZARD_UNLABELED: mutate_hazard_unlabeled,
    FaultKind.DEAD_START: mutate_dead_start,
    FaultKind.WAYPOINT_CONTRACTION: mutate_waypoint_contraction,
    FaultKind.ROUTE_PRUNING: mutate_route_pruning,
}


def mutate(
    kind: FaultKind, gold: GoldSpec, params: MutationParams, rng: np.random.Generator, protocol=None
) -> GroundTruthRepair:
    """Apply one mutation family.

    Args:
        kind: The family; ``NULL`` returns the unmodified gold.
        gold: The certified gold spec.
        params: Mutation parameters.
        rng: Seeded generator.
        protocol: Optional protocol for the severity grid.

    How:
        Dispatches on the family.

    Returns:
        The mutant.
    """
    if kind is FaultKind.NULL:
        return null_mutation(gold)
    return MUTATIONS[kind](gold, params, rng, protocol)


# ---- helpers -------------------------------------------------------------------------


def _record(gold, faulty, kind, locus, params, protocol, factors, target) -> GroundTruthRepair:
    """Assemble the repair record."""
    severity = severity_value(kind, params, protocol)
    return GroundTruthRepair(
        gold=gold.graph, faulty=faulty, kind=kind, locus=locus,
        severity_level=params.level if severity is not None else "fixed", severity=severity,
        factors=factors, target=target, expected_operators=EXPECTED_OPERATORS[kind],
        weak_edges=gold.weak_edges,
    )


def _trap_mask(grid: CellGrid, gold: GoldSpec) -> Mask:
    """Return the union of the gold trap masks."""
    mask = np.zeros((grid.resolution, grid.resolution), dtype=bool)
    for trap in gold.traps:
        mask |= grid.mask(trap)
    return mask


def _bounds_box(region: Predicate) -> Box | None:
    """Return the bounding box of a region, or ``None`` when it is unbounded."""
    bounds = region_bounds(region, PLANAR_DIMS)
    return None if bounds is None else Box(bounds[0], bounds[1])


def _centre(region: Predicate) -> np.ndarray:
    """Return the centre of a region's bounding box."""
    box = _bounds_box(region)
    if box is None:
        raise MutationNotApplicable("region is unbounded")
    return (box.lower + box.upper) / 2.0


def _box_tuple(box: Box) -> tuple[float, float, float, float]:
    """Return ``(x0, y0, x1, y1)`` of a box."""
    return (float(box.lower[0]), float(box.lower[1]), float(box.upper[0]), float(box.upper[1]))


def _rectangle_cells(rectangle) -> int:
    """Return the number of cells in an inclusive rectangle."""
    ix0, iy0, ix1, iy1 = rectangle
    return (ix1 - ix0 + 1) * (iy1 - iy0 + 1)


def _segment_box_distance(start: np.ndarray, end: np.ndarray, box: Box, samples: int = 101) -> float:
    """Return the smallest distance from the segment ``start``-``end`` to a box."""
    points = start + np.linspace(0.0, 1.0, samples)[:, None] * (end - start)
    nearest = np.clip(points, box.lower, box.upper)
    return float(np.linalg.norm(points - nearest, axis=1).min())


def _trap_edge(graph: AbstractGraph, trap: Box, params: MutationParams) -> Edge:
    """Return the edge to mutate: the given one, else the one closest to the trap."""
    if params.edge is not None:
        source, target = params.edge.split("->")
        return edge_named(graph, source, target)
    best = min(
        graph.edges,
        key=lambda e: _segment_box_distance(_centre(e.source.region), _centre(e.target.region), trap),
    )
    return best


def _relabel(safe: Predicate, trap: Box, core: Box) -> Predicate | None:
    """Replace the exclusion of ``trap`` in a safe predicate with that of ``core``.

    Returns:
        The new predicate, or ``None`` when the safe set never excludes the trap.
    """
    if isinstance(safe, Not) and safe.predicate == trap:
        return Not(core)
    if isinstance(safe, And):
        replaced = [_relabel(part, trap, core) for part in safe.predicates]
        if all(r is None for r in replaced):
            return None
        return And(*[r if r is not None else part for r, part in zip(replaced, safe.predicates)])
    return None


def _place_pockets(grid, allowed, total, count, anchor, rng):
    """Place one pocket, or two on opposite sides of the anchor."""
    if count == 1:
        rectangle = find_rectangle(allowed, total, anchor, grid, rng)
        if rectangle is None:
            raise MutationNotApplicable(f"no pocket of {total} cells fits")
        return [rectangle]
    if count != 2 or anchor is None:
        raise MutationNotApplicable("pockets must be 1, or 2 with a bounded region")
    each = max(1, total // 2)
    first = find_rectangle(allowed, each, anchor, grid, rng)
    if first is None:
        raise MutationNotApplicable(f"no pocket of {each} cells fits")
    centre = (anchor.lower + anchor.upper) / 2.0
    first_centre = (np.array(first[:2]) + np.array(first[2:]) + 1) / 2.0 * grid.cell
    away = first_centre - centre
    xs = (np.arange(grid.resolution) + 0.5) * grid.cell
    offsets = np.stack(np.meshgrid(xs - centre[0], xs - centre[1], indexing="ij"), axis=-1)
    norms = np.linalg.norm(offsets, axis=-1) * (np.linalg.norm(away) or 1.0)
    # A narrow cone opposite the first pocket, so the segment between the pockets
    # crosses the region and no single half-plane can separate both from it.
    opposite = (offsets @ -away) >= OPPOSITE_COSINE * np.maximum(norms, 1e-12)
    remaining = allowed.copy()
    remaining[first[0] : first[2] + 1, first[1] : first[3] + 1] = False
    second = find_rectangle(remaining & opposite, each, anchor, grid, rng)
    if second is None:
        raise MutationNotApplicable("no second pocket on the opposite side")
    return [first, second]


def _cell_points(grid: CellGrid, mask: Mask, limit: int) -> np.ndarray:
    """Return cell centres of a mask, thinned deterministically to about ``limit``."""
    cells = np.argwhere(mask)
    step = max(1, len(cells) // limit)
    return (cells[::step] + 0.5) * grid.cell


def _hull_recoverable(grid: CellGrid, visited: Mask, pocket: Box) -> bool:
    """Return whether the hull of the visited cells excludes the pocket."""
    points = _cell_points(grid, visited, _HULL_SAMPLE)
    if len(points) < 3:
        return True
    try:
        hull = convex_hull(list(points))
    except Exception:  # noqa: BLE001 - degenerate hull means nothing to intersect
        return True
    pocket_points = _cell_points(grid, grid.mask(pocket), _HULL_SAMPLE)
    return not any(hull.evaluate(p) for p in pocket_points)


def _halfplane_separable(grid: CellGrid, pockets, base: Mask) -> bool:
    """Return whether the existing half-plane fit separates the pockets from the region."""
    pocket_mask = np.zeros_like(base)
    for ix0, iy0, ix1, iy1 in pockets:
        pocket_mask[ix0 : ix1 + 1, iy0 : iy1 + 1] = True
    positive = list(_cell_points(grid, base, _HULL_SAMPLE))
    negative = list(_cell_points(grid, pocket_mask, _HULL_SAMPLE))
    plane = half_plane_partition(positive, negative)
    if plane is None:
        return False
    return all(plane.evaluate(p) for p in positive) and not any(plane.evaluate(p) for p in negative)


def _waypoint_chains(graph: AbstractGraph, count: int) -> list[tuple[str, ...]]:
    """Return every run of ``count`` droppable waypoints, as vertex-name tuples."""

    def droppable(vertex: Vertex) -> bool:
        if vertex is graph.source or vertex is graph.target:
            return False
        ins, outs = graph.predecessors(vertex), graph.successors(vertex)
        return len(ins) == 1 and len(outs) == 1 and ins[0].safe == outs[0].safe

    chains = []
    for vertex in graph.vertices:
        run = [vertex]
        while len(run) < count and droppable(run[-1]):
            following = graph.successors(run[-1])[0].target
            if not droppable(following):
                break
            run.append(following)
        if len(run) == count and all(droppable(v) for v in run):
            chains.append(tuple(v.name for v in run))
    return chains


def _spine(graph: AbstractGraph, chain: tuple[str, ...]) -> list[str]:
    """Return the vertex names of the single-route segment containing a chain."""
    names = [chain[0]]
    vertex = vertex_named(graph, chain[0])
    while len(graph.predecessors(vertex)) == 1:
        vertex = graph.predecessors(vertex)[0].source
        names.insert(0, vertex.name)
    vertex = vertex_named(graph, chain[0])
    while len(graph.successors(vertex)) == 1:
        vertex = graph.successors(vertex)[0].target
        if vertex.name not in names:
            names.append(vertex.name)
    return names


def _spine_edges(graph: AbstractGraph, chain: tuple[str, ...]) -> int:
    """Return the number of edges on the route segment around a chain."""
    return len(_spine(graph, chain)) - 1


def _position(graph: AbstractGraph, chain: tuple[str, ...]) -> float:
    """Return a chain's arc-length position (0 = start, 1 = end) on its route segment."""
    names = _spine(graph, chain)
    centres = [_centre(vertex_named(graph, n).region) for n in names]
    legs = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    total = float(legs.sum()) or 1.0
    positions = np.concatenate([[0.0], np.cumsum(legs)]) / total
    indices = [names.index(n) for n in chain]
    return float(np.mean([positions[i] for i in indices]))


def _has_edge(graph: AbstractGraph, source: str, target: str) -> bool:
    """Return whether the graph has an edge between two named vertices."""
    return any(e.source.name == source and e.target.name == target for e in graph.edges)
