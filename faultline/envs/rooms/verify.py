"""Layout verification for n-rooms worlds.

The heuristic verifier is exact about *existence*: because full-edge doors make
continuous reachability equal to room connectivity, a route either exists in the
room graph or it does not. "Heuristic" refers to the specification-level
approximation: it checks that the DAG can be traversed, not that a learner can
learn each edge.

Checks, in order:

1. room-graph components,
2. which rooms of each named region are in the Start component,
3. whether at least one Start-to-Goal DAG route uses only reachable regions,
4. whether every edge of at least one such route fits the environment's step
   horizon.

Dead branches are allowed on purpose: the paper's corridor-switching case is a
world where one branch of the DAG is unusable. They are reported as diagnostics,
not as failures.

The optional learned-policy stage lives in :mod:`faultline.envs.rooms.stochastic`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from math import ceil
from types import MappingProxyType

from faultline.envs.rooms.env import MAX_SPEED
from faultline.envs.rooms.grid import RoomGrid, RoomKey
from faultline.envs.rooms.world import RoomsWorld, SpecTopology

@dataclass(frozen=True)
class VerificationReport:
    """The heuristic verifier's verdict and diagnostics.

    Args:
        solvable: Whether at least one full route exists inside the horizon.
        reasons: Fatal reasons; empty exactly when ``solvable`` is true.
        component_count: Number of room-graph components.
        main_component_size: Number of rooms in the Start component.
        region_reachability: Fraction of each region's four rooms in the Start
            component. A low fraction predicts a ReachRefine symptom.
        region_internal_components: The connected components inside each region
            block. More than one predicts a PastRefine symptom.
        edge_lengths: Shortest room-path length per ``"A->B"`` topology edge, or
            ``None`` when the edge has no path.
        routes: Feasible Start-to-Goal routes, as vertex-name tuples.
        best_route: The preferred feasible route. It is the smallest-total
            route among those whose edges all fit the horizon; when no route
            fits, it falls back to the smallest-total route overall so its
            offending edges can explain the failure.
        dead_edges: Topology edges used by no feasible route. They predict an
            OrRefine or corridor-switching symptom.
        cut_rooms: Rooms whose removal disconnects Start from Goal. A non-empty
            set predicts an AddRefine symptom.
        max_steps: The horizon used for the check, or ``None`` if the world's
            layout is not analysable.
        longest_edge_steps: The longest pure travel time of any connected edge.

    How:
        Frozen and JSON-serialisable. Diagnostics that are not fatal remain in
        their fields; :attr:`solvable` is driven only by the pass rule.
    """

    solvable: bool
    reasons: tuple[str, ...]
    component_count: int
    main_component_size: int
    region_reachability: Mapping[str, float]
    region_internal_components: Mapping[str, tuple[tuple[RoomKey, ...], ...]]
    edge_lengths: Mapping[str, int | None]
    routes: tuple[tuple[str, ...], ...]
    best_route: tuple[str, ...] | None
    dead_edges: tuple[str, ...]
    cut_rooms: tuple[RoomKey, ...]
    max_steps: int | None
    longest_edge_steps: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_reachability", MappingProxyType(dict(self.region_reachability)))
        object.__setattr__(
            self,
            "region_internal_components",
            MappingProxyType(dict(self.region_internal_components)),
        )
        object.__setattr__(self, "edge_lengths", MappingProxyType(dict(self.edge_lengths)))

    def to_dict(self) -> dict:
        """Convert the report to a JSON-serialisable mapping.

        How:
            Converts the mappings and tuples to lists and dictionaries.

        Returns:
            The report as a JSON-serialisable mapping.
        """
        return {
            "solvable": self.solvable,
            "reasons": list(self.reasons),
            "component_count": self.component_count,
            "main_component_size": self.main_component_size,
            "region_reachability": dict(self.region_reachability),
            "region_internal_components": {
                name: [[list(room) for room in component] for component in components]
                for name, components in self.region_internal_components.items()
            },
            "edge_lengths": dict(self.edge_lengths),
            "routes": [list(route) for route in self.routes],
            "best_route": None if self.best_route is None else list(self.best_route),
            "dead_edges": list(self.dead_edges),
            "cut_rooms": [list(room) for room in self.cut_rooms],
            "max_steps": self.max_steps,
            "longest_edge_steps": self.longest_edge_steps,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> VerificationReport:
        """Rebuild a report from :meth:`to_dict`'s output.

        Args:
            payload: The mapping read back from JSON.

        How:
            Restores the tuples and frozen mappings that JSON flattened.

        Returns:
            The report.
        """
        return cls(
            solvable=bool(payload["solvable"]),
            reasons=tuple(payload["reasons"]),
            component_count=int(payload["component_count"]),
            main_component_size=int(payload["main_component_size"]),
            region_reachability={
                name: float(value) for name, value in payload["region_reachability"].items()
            },
            region_internal_components={
                name: tuple(
                    tuple(tuple(room) for room in component) for component in components
                )
                for name, components in payload["region_internal_components"].items()
            },
            edge_lengths={
                name: None if value is None else int(value)
                for name, value in payload["edge_lengths"].items()
            },
            routes=tuple(tuple(route) for route in payload["routes"]),
            best_route=(
                None
                if payload.get("best_route") is None
                else tuple(payload["best_route"])
            ),
            dead_edges=tuple(payload["dead_edges"]),
            cut_rooms=tuple(tuple(room) for room in payload["cut_rooms"]),
            max_steps=payload.get("max_steps"),
            longest_edge_steps=int(payload["longest_edge_steps"]),
        )


def _all_routes(topology: SpecTopology) -> tuple[tuple[str, ...], ...]:
    """Enumerate every simple Start-to-Goal route in the DAG.

    Args:
        topology: The specification DAG, which must contain ``Start`` and ``Goal``.

    How:
        Depth-first search along directed edges, never revisiting a vertex. The
        topology is already known to be acyclic, so the simple-path restriction
        only guards against malformed inputs.

    Returns:
        The routes in depth-first order.

    Raises:
        ValueError: If ``Start`` or ``Goal`` is not a topology vertex.
    """
    if "Start" not in topology.vertices or "Goal" not in topology.vertices:
        raise ValueError("Verification needs topology vertices named 'Start' and 'Goal'")
    routes: list[tuple[str, ...]] = []

    def walk(vertex: str, path: tuple[str, ...]) -> None:
        if vertex == "Goal":
            routes.append(path)
            return
        for successor in topology.successors(vertex):
            if successor not in path:
                walk(successor, path + (successor,))

    walk("Start", ("Start",))
    return tuple(routes)


def _edge_key(source: str, target: str) -> str:
    """Return the stable ``A->B`` key used for edge diagnostics.

    Args:
        source: Source region name.
        target: Target region name.

    How:
        Joins the two names with an arrow.

    Returns:
        The ``A->B`` key.
    """
    return f"{source}->{target}"


def _edge_steps(length: int | None, room_size: float) -> int:
    """Return the pure travel time in steps for a room-path length.

    Args:
        length: Room-path length in door crossings, or ``None``.
        room_size: Width and height of one room.

    How:
        ``None`` (no room path) contributes zero. It only occurs on dead-edge
        diagnostics, never on a feasible route.

    Returns:
        The travel time in environment steps.
    """
    return 0 if length is None else ceil(length * room_size / MAX_SPEED)


def _feasible_routes(
    world: RoomsWorld, component: frozenset[RoomKey], routes: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, ...], ...]:
    """Return the routes whose every region has a room in ``component``.

    Args:
        world: The world whose regions are tested.
        component: The room component to judge against.
        routes: Every Start-to-Goal route of the topology.

    How:
        Keeps a route when each of its regions has at least one room in the
        component.

    Returns:
        The feasible routes.
    """
    return tuple(
        route
        for route in routes
        if all(component & set(world.regions[name].rooms()) for name in route)
    )


def _main_component(
    world: RoomsWorld,
    components: tuple[frozenset[RoomKey], ...],
    routes: tuple[tuple[str, ...], ...],
) -> frozenset[RoomKey]:
    """Pick the component the Start region is judged against.

    Args:
        world: The world whose Start region is used.
        components: Every component of the room graph.
        routes: Every Start-to-Goal route of the topology.

    How:
        A Start block is often split by its own walls, so its rooms can sit in
        several components. Anchoring on one arbitrary Start room would reject
        layouts that a start in another room solves. Instead every component that
        holds a Start room is a candidate, and the one with the most feasible
        routes wins, then the one covering the most Start rooms. Ties keep the
        component with the smallest room, so the choice is deterministic.

    Returns:
        The chosen component.
    """
    start_rooms = set(world.regions["Start"].rooms())
    candidates = [component for component in components if component & start_rooms]
    return max(
        candidates,
        key=lambda component: (
            len(_feasible_routes(world, component, routes)),
            len(component & start_rooms),
        ),
    )


def _edge_lengths(world: RoomsWorld, grid: RoomGrid) -> dict[str, int | None]:
    """Return the shortest room-path length of every topology edge, or ``None``.

    Args:
        world: The world whose topology is measured.
        grid: The room graph of the world.

    How:
        Runs a shortest-path search between the two region room sets of each
        edge.

    Returns:
        The path length per ``A->B`` key, ``None`` where no path exists.
    """
    lengths: dict[str, int | None] = {}
    for source, target in world.topology.edges:
        path = grid.shortest_path(
            world.regions[source].rooms(), world.regions[target].rooms()
        )
        lengths[_edge_key(source, target)] = None if path is None else path.length
    return lengths


def _dead_edges(
    topology: SpecTopology, feasible_routes: tuple[tuple[str, ...], ...]
) -> tuple[str, ...]:
    """Return the topology edges that no feasible route uses.

    Args:
        topology: The specification DAG.
        feasible_routes: The routes that survived the reachability test.

    How:
        Collects the legs of every feasible route and keeps the edges outside
        that set.

    Returns:
        The ``A->B`` keys of the unused edges.
    """
    used = {leg for route in feasible_routes for leg in pairwise(route)}
    return tuple(
        _edge_key(source, target)
        for source, target in topology.edges
        if (source, target) not in used
    )


def _legs(
    route: tuple[str, ...], lengths: Mapping[str, int | None], room_size: float
) -> list[tuple[str, int]]:
    """Return ``(edge key, travel steps)`` for every leg of a route.

    Args:
        route: A route as region names.
        lengths: Room-path length per edge key.
        room_size: Width and height of one room.

    How:
        Converts each consecutive region pair into its edge key and travel time.

    Returns:
        One ``(edge key, steps)`` pair per leg.
    """
    return [
        (key, _edge_steps(lengths[key], room_size))
        for key in (_edge_key(source, target) for source, target in pairwise(route))
    ]


def _choose_route(
    feasible_routes: tuple[tuple[str, ...], ...],
    lengths: Mapping[str, int | None],
    room_size: float,
    max_steps: int,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    """Pick the best route and the reasons the world fails, if it does.

    Args:
        feasible_routes: Routes through reachable regions.
        lengths: Room-path length per edge key.
        room_size: Width and height of one room.
        max_steps: The episode step limit.

    How:
        A route fits when every leg's travel time is within ``max_steps``. The
        best route is the shortest fitting one, so a solvable report names a route
        the environment budget can execute. When none fits, the shortest feasible
        route is kept so its over-long legs explain the failure.

    Returns:
        ``(best_route, reasons)``; ``reasons`` is empty exactly when the world is
        solvable.
    """
    if not feasible_routes:
        return None, ("no feasible Start->Goal route through reachable regions",)
    fitting = tuple(
        route
        for route in feasible_routes
        if all(steps <= max_steps for _, steps in _legs(route, lengths, room_size))
    )
    best = min(
        fitting or feasible_routes,
        key=lambda route: (sum(lengths[key] or 0 for key, _ in _legs(route, lengths, room_size)), route),
    )
    if fitting:
        return best, ()
    reasons = tuple(
        f"{key} needs {steps} steps but the horizon is {max_steps}"
        for key, steps in _legs(best, lengths, room_size)
        if steps > max_steps
    )
    return best, reasons


def verify_heuristic(world: RoomsWorld) -> VerificationReport:
    """Check that a world has at least one full, within-horizon DAG route.

    Args:
        world: The world to check. Its door map must use full-edge doors.

    How:
        Builds the room graph, picks the Start component, records how much of
        each region is reachable, enumerates the Start-to-Goal routes, and keeps
        the feasible ones. A world passes when at least one feasible route has
        every edge within the horizon. Region unreachability and dead edges are
        diagnostics: a world with one dead branch but another working route
        passes.

        The horizon rule only binds when the world sets an explicit
        ``max_steps``. A derived horizon is the longest edge's travel time times
        a slack of at least one, so every edge fits it by construction.

    Returns:
        The report. Its ``solvable`` flag is true exactly when ``reasons`` is
        empty.

    Raises:
        ValueError: If the layout has partial door gaps, if the topology lacks
            ``Start`` or ``Goal``, or if the topology's region blocks are
            malformed.
    """
    grid = world.grid()
    topology = world.topology
    routes_all = _all_routes(topology)
    components = grid.components()
    main = _main_component(world, components, routes_all)
    feasible_routes = _feasible_routes(world, main, routes_all)
    lengths = _edge_lengths(world, grid)
    max_steps = world.effective_max_steps()
    best_route, reasons = _choose_route(feasible_routes, lengths, grid.room_size, max_steps)
    blocks = {name: world.regions[name] for name in topology.vertices}
    return VerificationReport(
        solvable=not reasons,
        reasons=reasons,
        component_count=len(components),
        main_component_size=len(main),
        region_reachability={
            name: len(set(block.rooms()) & main) / len(block.rooms())
            for name, block in blocks.items()
        },
        region_internal_components={
            name: grid.region_internal_components(block) for name, block in blocks.items()
        },
        edge_lengths=lengths,
        routes=feasible_routes,
        best_route=best_route,
        dead_edges=_dead_edges(topology, feasible_routes),
        cut_rooms=tuple(
            sorted(
                grid.cut_rooms(
                    world.regions["Start"].rooms(), world.regions["Goal"].rooms()
                )
            )
        ),
        max_steps=max_steps,
        longest_edge_steps=max(
            (_edge_steps(length, grid.room_size) for length in lengths.values() if length is not None),
            default=0,
        ),
    )


__all__ = ["VerificationReport", "verify_heuristic"]
