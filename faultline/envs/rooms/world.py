"""Specification-level world record for an n-rooms experiment.

A :class:`RoomsWorld` separates three things a fault injector must be able to
change independently:

- the physical layout (walls and doors),
- the specification labels (named 2x2 regions and the DAG joining them),
- and spec-only hazards (regions the specification treats as unsafe even though
  the dynamics do not model them).

Keeping them in one immutable record means a later fault variant can replace
only its target field. The default hazard tuple is empty, which reproduces the
paper's 100-rooms setups: they have no unsafe region.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from faultline.envs.rooms.env import MAX_SPEED, RoomsEnv
from faultline.envs.rooms.grid import RoomGrid
from faultline.envs.rooms.horizon import DEFAULT_SLACK, default_max_steps
from faultline.envs.rooms.layout import RoomsLayout, layout_from_dict, layout_to_dict
from faultline.envs.rooms.regions import RegionBlock
from faultline.geometry.serialize import predicate_from_dict, predicate_to_dict
from faultline.graph.abstract_graph import AbstractGraph
from faultline.spectrl.predicates import Box

# The paper's region names in the order A.3.3 places them. The figure's
# ``MG``, ``MG1`` and ``MG2`` are ``M3``, ``M4`` and ``M5``.
PAPER_REGION_ORDER: tuple[str, ...] = ("Start", "M1", "M2", "M3", "M4", "M5", "Goal")


@dataclass(frozen=True)
class SpecTopology:
    """The specification DAG as data: vertex names and directed edges.

    Args:
        vertices: Region names, unique, in a fixed order.
        edges: Directed ``(source, target)`` name pairs.

    How:
        Converts the inputs to tuples and validates that every edge endpoint is a
        declared vertex, that no edge repeats, and that the graph is acyclic.
        Validation happens once at construction, so a malformed topology cannot
        reach the verifier or the graph builder.

    Raises:
        ValueError: If a name is duplicated, an edge is unknown or repeated, or
            the edges form a cycle.
    """

    vertices: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        vertices = tuple(self.vertices)
        edges = tuple((str(source), str(target)) for source, target in self.edges)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "edges", edges)
        if len(set(vertices)) != len(vertices):
            raise ValueError("SpecTopology vertex names must be unique")
        known = set(vertices)
        for source, target in edges:
            if source not in known or target not in known:
                raise ValueError(f"SpecTopology edge {source!r} -> {target!r} uses an unknown vertex")
        if len(set(edges)) != len(edges):
            raise ValueError("SpecTopology edges must be unique")
        self._check_acyclic()

    def _check_acyclic(self) -> None:
        """Raise if the edges contain a directed cycle.

        How:
            Kahn's algorithm. A cycle leaves some vertices with a non-zero
            in-degree after the queue drains.
        """
        indegree = {name: 0 for name in self.vertices}
        successors: dict[str, list[str]] = {name: [] for name in self.vertices}
        for source, target in self.edges:
            successors[source].append(target)
            indegree[target] += 1
        queue: deque[str] = deque(name for name in self.vertices if indegree[name] == 0)
        visited = 0
        while queue:
            name = queue.popleft()
            visited += 1
            for target in successors[name]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != len(self.vertices):
            raise ValueError("SpecTopology must be acyclic")

    def successors(self, name: str) -> tuple[str, ...]:
        """Return the outgoing neighbours of a vertex, in edge order.

        Args:
            name: The vertex name.

        How:
            Scans the edge list for edges leaving the vertex.

        Returns:
            The target names, in edge order.
        """
        return tuple(target for source, target in self.edges if source == name)

    def predecessors(self, name: str) -> tuple[str, ...]:
        """Return the incoming neighbours of a vertex, in edge order.

        Args:
            name: The vertex name.

        How:
            Scans the edge list for edges entering the vertex.

        Returns:
            The source names, in edge order.
        """
        return tuple(source for source, target in self.edges if target == name)


# The paper's 8-edge DAG (A.3.1): Start branches through M1/M2, merges at M3,
# then branches through M4/M5 and merges at Goal.
PAPER_TOPOLOGY = SpecTopology(
    vertices=PAPER_REGION_ORDER,
    edges=(
        ("Start", "M1"),
        ("Start", "M2"),
        ("M1", "M3"),
        ("M2", "M3"),
        ("M3", "M4"),
        ("M3", "M5"),
        ("M4", "Goal"),
        ("M5", "Goal"),
    ),
)


@dataclass(frozen=True)
class RoomsWorld:
    """An immutable n-rooms world: layout, spec labels, topology, hazards.

    Args:
        name: Identifier used in reports and the scenario registry.
        layout: Physical wall and door model.
        regions: Named 2x2 blocks. Every topology vertex must have one.
        topology: The specification DAG over the named regions.
        hazards: Spec-only hazard boxes. They are physically passable; the
            default empty tuple matches the paper.
        max_steps: Episode step limit, or ``None`` to let the caller derive it.
        region_inset: Distance each region box is pulled in from its block's
            walls. ``0`` is the paper's whole 2x2 block; ``1`` reproduces the
            9-rooms inset.
        meta: Extra JSON-serialisable metadata (seed, orientation, rejected
            attempts, verification report, and so on).

    How:
        Copies and freezes the mapping fields and validates that every region
        block lies inside the layout's grid. Geometry-dependent methods read the
        layout's room size rather than a global constant.
    """

    name: str
    layout: RoomsLayout
    regions: Mapping[str, RegionBlock]
    topology: SpecTopology
    hazards: tuple[Box, ...] = ()
    max_steps: int | None = None
    region_inset: float = 0.0
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        supplied = dict(self.regions)
        # Canonical order: topology vertices first, then any extra regions by
        # name. This makes ``regions`` iteration and serialisation stable.
        regions = {
            name: supplied[name] for name in self.topology.vertices if name in supplied
        }
        regions.update(
            (name, block) for name, block in sorted(supplied.items()) if name not in regions
        )
        object.__setattr__(self, "regions", MappingProxyType(regions))
        object.__setattr__(self, "hazards", tuple(self.hazards))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))
        missing = set(self.topology.vertices) - set(regions)
        if missing:
            raise ValueError(f"RoomsWorld has no region block for {sorted(missing)}")
        n = self.layout.rooms_per_side
        for name, block in regions.items():
            if block.col < 0 or block.row < 0 or block.col + 1 >= n or block.row + 1 >= n:
                raise ValueError(
                    f"Region {name!r} block {block!r} is outside a {n}x{n} grid"
                )
        for hazard in self.hazards:
            if not isinstance(hazard, Box):
                raise TypeError("RoomsWorld hazards must be Box predicates")

    def grid(self) -> RoomGrid:
        """Return the room-level analysis graph for this world.

        How:
            Requires full-edge doors; a layout with partially open walls raises
            through :class:`~faultline.envs.rooms.grid.RoomGrid`.

        Returns:
            The room grid.
        """
        return RoomGrid(self.layout.rooms_per_side, self.layout.doors, self.layout.room_size)

    def region_box(self, name: str) -> Box:
        """Return the continuous box of a named region.

        Args:
            name: The region name.

        How:
            Uses the whole 2x2 block by default. ``region_inset`` pulls each side
            inward, which lets a caller reproduce the 9-rooms inset without
            changing region data.

        Returns:
            The box predicate.

        Raises:
            KeyError: If the name is not a region.
        """
        return self.regions[name].to_box(self.layout.room_size, self.region_inset)

    def build_graph(self) -> AbstractGraph:
        """Build the specification's abstract graph from the topology.

        How:
            Adds one vertex per topology name, labelled with that region's box,
            then one edge per topology edge with the universe as its safe
            region. Sets ``source`` to ``Start`` and ``target`` to ``Goal`` when
            those names exist. The paper's n-rooms spec has no avoid region, so
            the universe is the correct safe predicate.

        Returns:
            A fresh, independent graph.
        """
        graph = AbstractGraph()
        vertices = {
            name: graph.add_vertex(name, self.region_box(name))
            for name in self.topology.vertices
        }
        for source, target in self.topology.edges:
            graph.add_edge(vertices[source], vertices[target])
        if "Start" in vertices:
            graph.source = vertices["Start"]
        if "Goal" in vertices:
            graph.target = vertices["Goal"]
        return graph

    def edge_room_sets(self) -> tuple[tuple[tuple, tuple], ...]:
        """Return each topology edge as its two endpoint room sets.

        How:
            Maps every edge's region names to the blocks' four rooms, preserving
            topology edge order. This is the input
            :func:`~faultline.envs.rooms.grid.default_max_steps` needs.

        Returns:
            One ``(source_rooms, target_rooms)`` pair per edge.
        """
        return tuple(
            (self.regions[source].rooms(), self.regions[target].rooms())
            for source, target in self.topology.edges
        )

    def effective_max_steps(
        self, *, speed: float = MAX_SPEED, slack: float = DEFAULT_SLACK
    ) -> int:
        """Return the horizon this world runs at.

        Args:
            speed: Agent speed used for the travel-time estimate.
            slack: Safety margin over the longest pure travel time.

        How:
            An explicit ``max_steps`` wins. Otherwise the horizon is derived from
            the longest shortest room path among the spec's edges, so a winding
            100-rooms edge is not cut short by a 9-rooms-sized limit.

        Returns:
            The episode step limit.
        """
        if self.max_steps is not None:
            return int(self.max_steps)
        return default_max_steps(
            self.grid(), self.edge_room_sets(), speed=speed, slack=slack
        )

    def make_env(self) -> RoomsEnv:
        """Build a fresh environment over this world's layout.

        How:
            Uses :meth:`effective_max_steps`, which honours an explicit limit or
            derives one from the room graph.

        Returns:
            A new environment instance.
        """
        return RoomsEnv(layout=self.layout, max_steps=self.effective_max_steps())

    def to_scenario(self, name: str | None = None):
        """Wrap this world as a scenario the AutoSpec runner can execute.

        Args:
            name: Registry name to use in reports. Defaults to the world's name,
                so a fixed world keeps its catalogue name.

        How:
            Imports :class:`~faultline.scenarios.catalogue.Scenario` inside the
            call, because the scenario module imports the env package. The
            scenario has no paper reference numbers (the paper gives none for
            100 rooms) and no unsafe region (the n-rooms specification is pure
            reach). Per-edge operator expectations are copied from the world's
            ``expected_operators`` metadata when present, so a designed layout
            can record them without changing the runner.

        Returns:
            A scenario that builds this world's graph and environment.
        """
        from faultline.scenarios.catalogue import Scenario

        scenario_name = name or self.name
        raw_expected = self.meta.get("expected_operators", {})
        expected_operators = (
            {
                str(edge): tuple(str(operator) for operator in operators)
                for edge, operators in raw_expected.items()
            }
            if isinstance(raw_expected, Mapping)
            else {}
        )
        return Scenario(
            name=scenario_name,
            description=f"{self.layout.rooms_per_side}x{self.layout.rooms_per_side} n-rooms world",
            layout=self.layout,
            build_graph=self.build_graph,
            expected_operator=None,
            reference_before=None,
            reference_after=None,
            world=self,
            expected_operators=expected_operators,
            metadata={"world": dict(self.meta)},
        )

    def to_dict(self) -> dict:
        """Convert the world to a JSON-serialisable mapping.

        How:
            Delegates to :func:`world_to_dict`.

        Returns:
            The world as a JSON-serialisable mapping.
        """
        return world_to_dict(self)


def world_to_dict(world: RoomsWorld) -> dict:
    """Convert a world to a JSON-serialisable mapping.

    Args:
        world: The world to convert.

    How:
        Reuses :func:`~faultline.envs.rooms.layout.layout_to_dict` for the
        physical layout. Region blocks become ``[col, row]`` and topology edges
        become two-element lists. Hazards use the predicate encoder. Regions are
        listed in the world's canonical order (topology vertices first, then any
        extras by name), so the output is stable.

    Returns:
        The mapping.
    """
    return {
        "name": world.name,
        "layout": layout_to_dict(world.layout),
        "regions": {name: [block.col, block.row] for name, block in world.regions.items()},
        "topology": {
            "vertices": list(world.topology.vertices),
            "edges": [list(edge) for edge in world.topology.edges],
        },
        "hazards": [predicate_to_dict(hazard) for hazard in world.hazards],
        "max_steps": world.max_steps,
        "region_inset": world.region_inset,
        "meta": dict(world.meta),
    }


def world_from_dict(payload: Mapping) -> RoomsWorld:
    """Rebuild a world from :func:`world_to_dict`'s output.

    Args:
        payload: The mapping read back from JSON.

    How:
        Reverses every field. Region values may be lists or tuples. Hazards are
        decoded through the predicate decoder; they are required to be boxes.

    Returns:
        The reconstructed world.
    """
    topology_payload = payload["topology"]
    topology = SpecTopology(
        tuple(topology_payload["vertices"]),
        tuple(tuple(edge) for edge in topology_payload["edges"]),
    )
    regions = {
        name: RegionBlock(int(col), int(row))
        for name, (col, row) in payload["regions"].items()
    }
    hazards = tuple(predicate_from_dict(item) for item in payload.get("hazards", ()))
    for hazard in hazards:
        if not isinstance(hazard, Box):
            raise TypeError("Serialised RoomsWorld hazards must decode to Box predicates")
    return RoomsWorld(
        name=str(payload["name"]),
        layout=layout_from_dict(payload["layout"]),
        regions=regions,
        topology=topology,
        hazards=hazards,
        max_steps=payload.get("max_steps"),
        region_inset=float(payload.get("region_inset", 0.0)),
        meta=dict(payload.get("meta", {})),
    )


def paper_world_name(seed: int) -> str:
    """Return the registry name for a random paper-style world.

    Args:
        seed: The generator seed.

    How:
        The name is a pure function of the seed, so a worker process can rebuild
        the same world from the name without receiving its data.

    Returns:
        ``rooms100_random_<seed>``.
    """
    return f"rooms100_random_{seed}"


__all__ = [
    "PAPER_REGION_ORDER",
    "PAPER_TOPOLOGY",
    "RoomsWorld",
    "SpecTopology",
    "paper_world_name",
    "world_from_dict",
    "world_to_dict",
]
