"""The five 9-rooms validation scenarios (paper sections 8 and 11).

Each builder returns a :class:`Scenario` holding the environment geometry, a
fresh-graph factory, and the operator the section 11 table expects. The graphs are
hand-built (paper section 10) so a scenario isolates one bottleneck edge instead
of the whole compiled specification.

The scenarios map one-to-one onto the table's symptom rows: a goal region that
includes sealed rooms (ReachRefine), a dangerous second doorway into the goal room
(AvoidRefine), one edge spanning the whole corridor (AddRefine), a start region
that straddles a sealed room (PastRefine), and one of two routes to the goal being
unreachable (OrRefine).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from faultline.envs import Rooms9, RoomsLayout, default_doors
from faultline.envs.layout import DoorInterval, WallKey
from faultline.graph.abstract_graph import AbstractGraph
from faultline.spectrl.predicates import Box, Not, Predicate

START = Box((1.0, 1.0), (7.0, 7.0))
GOAL = Box((17.0, 17.0), (23.0, 23.0))
# The goal box reaches across the top row: only its right third (room (2, 2)) is
# reachable, so a goal sampled in the sealed rooms (0, 2) or (1, 2) cannot be met.
TRAP_GOAL = Box((1.0, 17.0), (23.0, 23.0))
# The start box covers the bottom-left room and the sealed room directly above it
# in equal measure, so only the lower half can reach the goal and PastRefine's
# mean-bisector half-plane lands on the y = 8 wall.
MIXED_START = Box((1.0, 1.0), (7.0, 15.0))
# Room (1, 2), entered only through the extra west doorway and excluded from the
# safe region: the dangerous half of the goal approach.
DANGEROUS_SIDE_ROOM = Box((8.0, 16.0), (16.0, 24.0))
# The centre room, on the default corridor: the sibling source the two-path
# scenario reuses for the alternative route into the goal.
MID = Box((9.0, 9.0), (15.0, 15.0))


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

    def make_env(self) -> Rooms9:
        """Build an environment over this scenario's layout.

        How:
            Creates a fresh :class:`~faultline.envs.rooms9.Rooms9`; the runner sets
            the abstract states for each edge as it trains.

        Returns:
            A new environment instance.
        """
        return Rooms9(layout=self.layout)


def trap_goal() -> Scenario:
    """Return the goal-region-includes-a-trap scenario (ReachRefine).

    How:
        Reuses the default corridor but widens the goal box across the sealed top
        rooms, so only the reachable third of the box can be met and ReachRefine
        has an unreachable part to cut.

    Returns:
        The ``trap_goal`` scenario.
    """
    return Scenario(
        name="trap_goal",
        description="Goal region includes sealed rooms, isolating ReachRefine.",
        layout=RoomsLayout(),
        build_graph=lambda: _single_edge(START, TRAP_GOAL),
        expected_operator="seq",
        reference_before=0.15,
        reference_after=0.85,
    )


def narrow_door() -> Scenario:
    """Return the narrow dangerous doorway scenario (AvoidRefine).

    How:
        Adds a west doorway into the goal room through the side room and excludes
        that side room from the safe region, so a goal approached from the west
        violates safety and AvoidRefine has unsafe trajectory tails to block.

    Returns:
        The ``narrow_door`` scenario.
    """
    doors: dict[WallKey, DoorInterval] = {
        ("h", 1, 1): (11.0, 13.0),
        ("v", 1, 2): (19.0, 21.0),
    }
    return Scenario(
        name="narrow_door",
        description="A dangerous second doorway into the goal room, isolating AvoidRefine.",
        layout=_layout_with(doors),
        build_graph=lambda: _single_edge(START, GOAL, Not(DANGEROUS_SIDE_ROOM)),
        expected_operator="seq",
        reference_before=0.30,
        reference_after=0.75,
    )


def long_corridor() -> Scenario:
    """Return the long single-edge corridor scenario (AddRefine).

    How:
        Uses the default corridor as one edge from START to GOAL, four rooms and
        four doorways long, so a policy that cannot learn the long horizon leaves
        AddRefine a waypoint to insert at the successful midpoints.

    Returns:
        The ``long_corridor`` scenario.
    """
    return Scenario(
        name="long_corridor",
        description="One edge spans the whole corridor, isolating AddRefine.",
        layout=RoomsLayout(),
        build_graph=lambda: _single_edge(START, GOAL),
        expected_operator="add",
        reference_before=0.20,
        reference_after=0.90,
    )


def mixed_start() -> Scenario:
    """Return the mixed start-region scenario (PastRefine).

    How:
        Extends the start box up the sealed left column, so starts below the y = 8
        wall reach the goal while starts above it cannot, and PastRefine has a
        linearly separable set of failing starts to cut.

    Returns:
        The ``mixed_start`` scenario.
    """
    return Scenario(
        name="mixed_start",
        description="Start region straddles a sealed room, isolating PastRefine.",
        layout=RoomsLayout(),
        build_graph=lambda: _single_edge(MIXED_START, GOAL),
        expected_operator="past",
        reference_before=0.40,
        reference_after=0.80,
    )


def blocked_two_path() -> Scenario:
    """Return the blocked two-path scenario (OrRefine).

    How:
        Builds the graph ``start -> goal`` plus ``mid -> goal``. The direct edge to
        the goal is the blocked route while the goal already has the mid edge as a
        second parent, which is exactly the sibling OrRefine reuses to add
        ``start -> mid`` and complete the alternative route.

    Returns:
        The ``blocked_two_path`` scenario.
    """
    return Scenario(
        name="blocked_two_path",
        description="The direct route to the goal is blocked; a sibling parent exists.",
        layout=RoomsLayout(),
        build_graph=_two_path_graph,
        expected_operator="or",
        reference_before=0.0,
        reference_after=0.85,
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
    """Build the two-edge graph ``start -> goal`` plus ``mid -> goal``.

    How:
        Adds a direct edge from the start to the goal and an existing edge from a
        mid corridor region into the same goal. The goal therefore has two parents
        while the start has no route through the sibling, which is what OrRefine
        completes by adding ``start -> mid``.

    Returns:
        The two-edge abstract graph.
    """
    graph = AbstractGraph()
    start = graph.add_vertex("start", START)
    mid = graph.add_vertex("mid", MID)
    goal = graph.add_vertex("goal", GOAL)
    graph.add_edge(start, goal)
    graph.add_edge(mid, goal)
    graph.source = start
    graph.target = goal
    return graph


def _layout_with(extra: dict[WallKey, DoorInterval]) -> RoomsLayout:
    """Build a 9-rooms layout with extra doors over the default corridor.

    Args:
        extra: Wall key to open interval mapping added to the default doors.

    How:
        Copies the default door map, overlays the extra doors, and builds a layout
        that keeps the default room geometry.

    Returns:
        The layout with the extra doors opened.
    """
    doors = dict(default_doors())
    doors.update(extra)
    return RoomsLayout(doors=doors)


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
