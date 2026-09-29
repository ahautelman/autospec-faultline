"""The five 9-rooms validation scenarios (paper sections 8 and 11).

Each builder returns a :class:`Scenario` holding the environment geometry, a
fresh-graph factory, and the operator the section 11 table expects. The graphs are
hand-built (paper section 10) so a scenario isolates one bottleneck edge instead
of the whole compiled specification.

The scenarios map one-to-one onto the table's symptom rows: a goal region that
includes sealed rooms (ReachRefine), a narrow unsafe doorway beside the safe route
(AvoidRefine), one edge spanning the whole corridor (AddRefine), a start region
that straddles a sealed room (PastRefine), and one of two routes to the goal being
unreachable (OrRefine).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from faultline.envs import Rooms9, RoomsLayout, open_doors_except_sealed_rooms
from faultline.envs.layout import (
    ROOMS_PER_SIDE,
    ROOM_SIZE,
    DoorInterval,
    WallKey,
    blocked_two_path_doors,
    long_corridor_doors,
    long_mixed_star_doors,
)
from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.spectrl.predicates import Box, Predicate

START = Box((1.0, 1.0), (7.0, 7.0))
GOAL = Box((17.0, 17.0), (23.0, 23.0))
# The goal box reaches across the top row: only its right third (room (2, 2)) is
# reachable, so a goal sampled in the sealed rooms (0, 2) or (1, 2) cannot be met.
TRAP_GOAL = Box((1.0, 17.0), (23.0, 23.0))
# The two top-row rooms the trap_goal layout seals. Every other internal wall is
# open, so the goal box's only reachable third is room (2, 2).
TRAP_ROOMS: tuple[tuple[int, int], ...] = ((0, 2), (1, 2))
# The start box covers the middle-left room and the sealed top-left room directly
# above it in equal measure. In the long mixed star layout only the lower half
# joins the star's corridor and reaches the goal, so PastRefine's mean-bisector
# half-plane lands on the y = 16 wall.
MIXED_START = Box((1.0, 9.0), (7.0, 23.0))
# narrow_door: the goal sits in the top-left room, reached by going up the
# far-right gap of the dividing wall. The environment hides an unescapable trap in
# the narrow left doorway; the specification does not model it.
NARROW_DOOR_GOAL = Box((1.0, 17.0), (7.0, 23.0))
NARROW_DOOR_TRAP = Box((2.0, 15.0), (4.0, 17.0))
# The two mid regions of the paper's two-path room. MID1 spans the top-left and
# top-middle rooms; MID2 spans the bottom-middle and bottom-right rooms. Both are
# parents of GOAL, and the refinement links MID1 to MID2 so the blocked
# MID1 -> GOAL crossing is rerouted along the winding route.
MID1 = Box((1.0, 17.0), (15.0, 23.0))
MID2 = Box((9.0, 1.0), (23.0, 7.0))
# The long-corridor goal: the paper's figure puts GOAL in the centre room, at the
# far end of the corridor that climbs the left column and crosses the top row.
LONG_CORRIDOR_GOAL = Box((9.0, 9.0), (15.0, 15.0))


@dataclass(frozen=True)
class Scenario:
    """One 9-rooms validation scenario from paper sections 8 and 11.

    Args:
        name: Short identifier used in reports and test ids.
        description: The symptom and the operator the scenario isolates.
        layout: Wall and door model for the environment.
        build_graph: Zero-argument factory returning a fresh abstract graph with
            its source and target set.
        expected_operator: The loop operator the section 11 table expects for the
            symptom, or ``None`` when the scenario cannot isolate one operator.
        reference_before: The table's per-edge success before refinement.
        reference_after: The table's per-edge success after refinement.
        unsafe_region: Optional unescapable trap the environment hides from the
            specification; ``None`` means no trap.
        bottleneck: Optional selector returning the edge the report trains as the
            bottleneck when the graph fans out from its source; ``None`` means the
            single edge leaving the graph source.

    How:
        Holds geometry and metadata only. The graph factory defers construction so
        every run starts from an independent, unrefined graph and no scenario state
        is shared between runs.
    """

    name: str
    description: str
    layout: RoomsLayout
    build_graph: Callable[[], AbstractGraph]
    expected_operator: str | None
    reference_before: float
    reference_after: float
    unsafe_region: Predicate | None = None
    bottleneck: Callable[[AbstractGraph], Edge] | None = None

    def make_env(self) -> Rooms9:
        """Build an environment over this scenario's layout.

        How:
            Creates a fresh :class:`~faultline.envs.rooms9.Rooms9`; the runner sets
            the abstract states for each edge as it trains.

        Returns:
            A new environment instance.
        """
        return Rooms9(layout=self.layout, unsafe_region=self.unsafe_region)

    def report_edge(self, graph: AbstractGraph) -> Edge:
        """Return the bottleneck edge the report trains and refines.

        Args:
            graph: The scenario's freshly built graph.

        How:
            Uses the scenario's ``bottleneck`` selector when it has one, for a
            scenario whose graph fans out from its source; otherwise returns the
            single edge leaving the graph source, the common single-bottleneck
            case.

        Returns:
            The edge the report trains as the before curve and refines.

        Raises:
            ValueError: If the graph has no source, or has no ``bottleneck`` and
                its source does not have exactly one outgoing edge.
        """
        if self.bottleneck is not None:
            return self.bottleneck(graph)
        if graph.source is None:
            raise ValueError("Scenario graph has no source vertex")
        outgoing = graph.successors(graph.source)
        if len(outgoing) != 1:
            raise ValueError(f"Expected one source edge, found {len(outgoing)}")
        return outgoing[0]


def trap_goal() -> Scenario:
    """Return the goal-region-includes-a-trap scenario (ReachRefine).

    How:
        Uses the open trap-goal layout: every internal wall is removed except the
        walls sealing the two top-row trap rooms, so the arena is one open area
        and only room (2, 2) of the top-row goal box can be reached. The
        unreachable third of the goal box is what ReachRefine cuts, over a single
        start-to-goal edge.

    Returns:
        The ``trap_goal`` scenario.
    """
    return Scenario(
        name="trap_goal",
        description="Goal region includes sealed rooms, isolating ReachRefine.",
        layout=RoomsLayout(doors=open_doors_except_sealed_rooms(TRAP_ROOMS)),
        build_graph=lambda: _single_edge(START, TRAP_GOAL),
        expected_operator="seq",
        reference_before=0.15,
        reference_after=0.85,
    )


def narrow_door() -> Scenario:
    """Return the narrow dangerous doorway scenario (AvoidRefine).

    How:
        Builds one open arena split only by the wall along ``y = 16``: a narrow
        doorway on the left, a solid middle, and a gap on the far right. The goal
        is the top-left room and the only safe route is the far-right gap. The
        environment hides an unescapable trap in the narrow doorway, which the
        specification does not model (its safe region is the universe), so a policy
        that takes the short door gets stuck and AvoidRefine has unsafe trajectory
        tails to block.

    Returns:
        The ``narrow_door`` scenario.
    """
    return Scenario(
        name="narrow_door",
        description="A narrow unsafe doorway beside the safe route (AvoidRefine).",
        layout=RoomsLayout(doors=_open_arena_doors()),
        build_graph=lambda: _single_edge(START, NARROW_DOOR_GOAL),
        expected_operator="seq",
        reference_before=0.30,
        reference_after=0.75,
        unsafe_region=NARROW_DOOR_TRAP,
    )


def long_corridor() -> Scenario:
    """Return the long single-edge corridor scenario (AddRefine).

    How:
        Uses the paper's winding corridor from START to GOAL as one edge: from the
        bottom-left room the route climbs the left column, crosses the top, and
        drops into the centre room, five rooms and four doorways long. A policy
        that cannot learn that long horizon leaves AddRefine a waypoint to insert
        at the successful midpoints.

    Returns:
        The ``long_corridor`` scenario.
    """
    return Scenario(
        name="long_corridor",
        description="One edge spans the whole corridor, isolating AddRefine.",
        layout=RoomsLayout(doors=long_corridor_doors()),
        build_graph=lambda: _single_edge(START, LONG_CORRIDOR_GOAL),
        expected_operator="add",
        reference_before=0.20,
        reference_after=0.90,
    )


def mixed_start() -> Scenario:
    """Return the mixed start-region scenario (PastRefine).

    How:
        Uses the paper's long mixed star room. The start box straddles the y = 16
        wall at the top of the left column: the middle-left room below it joins the
        star's corridor and reaches the goal, while the sealed top-left room above
        it cannot. PastRefine therefore has a linearly separable set of failing
        starts to cut on that wall.

    Returns:
        The ``mixed_start`` scenario.
    """
    return Scenario(
        name="mixed_start",
        description="Start region straddles a sealed room, isolating PastRefine.",
        layout=RoomsLayout(doors=long_mixed_star_doors()),
        build_graph=lambda: _single_edge(MIXED_START, GOAL),
        expected_operator="past",
        reference_before=0.40,
        reference_after=0.80,
    )


def blocked_two_path() -> Scenario:
    """Return the blocked two-path scenario (OrRefine).

    How:
        Uses the paper's two-path room: START in room ``(0, 0)``, MID1 spanning
        the top-left and top-middle rooms, MID2 spanning the bottom-middle and
        bottom-right rooms, and GOAL in room ``(2, 2)``. Two interior walls block
        the direct START-to-MID2 and MID1-to-GOAL crossings, so neither nominal
        route of ``(start;mid1;goal) or (start;mid2;goal)`` is realisable on its
        own. The report trains the blocked ``mid1 -> goal`` edge (see
        :meth:`Scenario.report_edge`); OrRefine then reuses the goal's other parent
        ``mid2`` to add ``mid1 -> mid2``, realising ``start -> mid1 -> mid2 ->
        goal``.

    Returns:
        The ``blocked_two_path`` scenario.
    """
    return Scenario(
        name="blocked_two_path",
        description="The direct route to the goal is blocked; a sibling parent exists.",
        layout=RoomsLayout(doors=blocked_two_path_doors()),
        build_graph=_two_path_graph,
        expected_operator="or",
        reference_before=0.0,
        reference_after=0.85,
        bottleneck=_blocked_two_path_bottleneck,
    )


def _single_edge(
    start_region: Predicate, goal_region: Predicate, safe: Predicate | None = None
) -> AbstractGraph:
    """Build a graph of one reach-avoid edge from START-like source to goal.

    Args:
        start_region: Region the edge's source vertex labels.
        goal_region: Region the edge's target vertex labels.
        safe: Optional safe region for the edge; defaults to the universe.

    How:
        Creates the two vertices, joins them with one edge, and sets the graph's
        source and target so the refinement loop can run.

    Returns:
        The one-edge abstract graph.
    """
    graph = AbstractGraph()
    source = graph.add_vertex("start", start_region)
    target = graph.add_vertex("goal", goal_region)
    graph.add_edge(source, target, safe)
    graph.source = source
    graph.target = target
    return graph


def _two_path_graph() -> AbstractGraph:
    """Build the graph for ``(start;mid1;goal) or (start;mid2;goal)``.

    How:
        Adds both branches of the spec's disjunction. The two branch exits meet at
        the goal, so the goal has two parents. The room walls block the direct
        ``start -> mid2`` and ``mid1 -> goal`` crossings, so neither branch
        succeeds on its own; OrRefine links ``mid1`` to the sibling ``mid2`` to
        realise ``start -> mid1 -> mid2 -> goal``.

    Returns:
        The four-vertex, four-edge abstract graph.
    """
    graph = AbstractGraph()
    start = graph.add_vertex("start", START)
    mid1 = graph.add_vertex("mid1", MID1)
    mid2 = graph.add_vertex("mid2", MID2)
    goal = graph.add_vertex("goal", GOAL)
    graph.add_edge(start, mid1)
    graph.add_edge(start, mid2)
    graph.add_edge(mid1, goal)
    graph.add_edge(mid2, goal)
    graph.source = start
    graph.target = goal
    return graph


def _blocked_two_path_bottleneck(graph: AbstractGraph) -> Edge:
    """Return the blocked ``mid1 -> goal`` edge the two-path report trains.

    Args:
        graph: The two-path graph.

    How:
        Names the edge whose target (the goal) already has the ``mid2`` sibling, so
        it is the edge OrRefine reroutes through that sibling.

    Returns:
        The ``mid1 -> goal`` edge.

    Raises:
        ValueError: If the graph has no ``mid1 -> goal`` edge.
    """
    for edge in graph.edges:
        if edge.source.name == "mid1" and edge.target.name == "goal":
            return edge
    raise ValueError("blocked_two_path graph has no mid1 -> goal edge")


def _open_arena_doors() -> dict[WallKey, DoorInterval]:
    """Return the narrow-door door map: one dividing wall over an open arena.

    How:
        Opens every vertical wall and every ``y = 8`` wall across its full edge so
        those rooms merge into a single arena. The only wall left sits on
        ``y = 16``: a narrow doorway in column 0, a solid middle, and a far-right
        gap.

    Returns:
        The wall key to open interval mapping.
    """
    doors: dict[WallKey, DoorInterval] = {}
    for row in range(ROOMS_PER_SIDE):
        low, high = row * ROOM_SIZE, (row + 1) * ROOM_SIZE
        doors[("v", 0, row)] = (low, high)
        doors[("v", 1, row)] = (low, high)
    for col in range(ROOMS_PER_SIDE):
        low, high = col * ROOM_SIZE, (col + 1) * ROOM_SIZE
        doors[("h", col, 0)] = (low, high)
    doors[("h", 0, 1)] = (2.0, 4.0)
    doors[("h", 2, 1)] = (19.0, 23.0)
    return doors


# The five table scenarios, keyed by name and in paper table order.
SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        trap_goal(),
        narrow_door(),
        long_corridor(),
        mixed_start(),
        blocked_two_path(),
    )
}
