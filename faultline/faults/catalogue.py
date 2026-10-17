"""FaultLine benchmark cases designed to isolate each injected fault.

The benchmark cases are deliberately easier than the paper-scale validation
scenarios.  FaultLine's core question is whether AutoSpec recovers the *known
correct repair*, so the underlying PPO task should provide enough trajectory
evidence for the relevant refinement operator to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from faultline.envs import Rooms9, RoomsLayout
from faultline.envs.layout import ROOM_SIZE, ROOMS_PER_SIDE, DoorInterval, WallKey
from faultline.faults.base import FaultCase
from faultline.faults.environments import HiddenHazardRooms9
from faultline.faults.injectors import (
    inject_goal_trap,
    inject_missing_alternative_path,
    inject_missing_waypoint,
    inject_overwide_safety,
    inject_unreachable_start,
)
from faultline.graph import AbstractGraph
from faultline.spectrl.predicates import Box, Not, Predicate


@dataclass(frozen=True)
class FaultBenchmarkCase:
    """One injected fault together with the matching evaluation environment."""

    fault: FaultCase
    make_env: Callable[[], Rooms9]


def _single_edge(
    start_region: Predicate,
    goal_region: Predicate,
    safe: Predicate | None = None,
) -> AbstractGraph:
    graph = AbstractGraph()
    start = graph.add_vertex("start", start_region)
    goal = graph.add_vertex("goal", goal_region)
    graph.add_edge(start, goal, safe)
    graph.source = start
    graph.target = goal
    return graph


def _all_open_doors() -> dict[WallKey, DoorInterval]:
    """Open every internal wall across its full room edge."""

    doors: dict[WallKey, DoorInterval] = {}
    for line_index in range(1, ROOMS_PER_SIDE):
        for room in range(ROOMS_PER_SIDE):
            low = room * ROOM_SIZE
            high = low + ROOM_SIZE
            doors[("v", line_index - 1, room)] = (low, high)
            doors[("h", room, line_index - 1)] = (low, high)
    return doors


def goal_trap_case(severity: float = 1.0) -> FaultBenchmarkCase:
    """ReachRefine: half of a coarse target is in an inaccessible room.

    The correct goal is easy and lies in the start room.  The injected goal adds
    an equally sized box in the adjacent sealed room.  Roughly half the sampled
    goal points are therefore reachable, providing both successes and failures.
    """

    start = Box((1.0, 1.0), (3.5, 3.5))
    correct_goal = Box((5.0, 4.5), (7.0, 6.5))
    trap = Box((9.0, 4.5), (11.0, 6.5))

    graph = _single_edge(start, correct_goal)
    fault = inject_goal_trap(
        graph,
        "goal",
        trap,
        name="goal_trap",
        severity=severity,
    )

    return FaultBenchmarkCase(
        fault=fault,
        make_env=lambda: Rooms9(
            layout=RoomsLayout(doors={}),
            max_steps=40,
        ),
    )


def overwide_safety_case(severity: float = 1.0) -> FaultBenchmarkCase:
    """AvoidRefine: an unmodelled hazard is incorrectly included in ``safe``.

    Everything lies inside one physical room, so wall navigation cannot hide the
    safety failure.  The straight-line route from start to goal passes through a
    central hidden hazard, while a short safe detour exists above or below it.

    The correct specification excludes the hazard.  Fault injection broadens the
    safe predicate so the hazard is logically allowed.  ``HiddenHazardRooms9``
    then exposes unsafe trajectory tails without giving PPO the full logical
    violation penalty until AutoSpec refines the safe predicate.
    """

    start = Box((1.0, 3.25), (2.0, 4.75))
    goal = Box((6.0, 3.25), (7.0, 4.75))

    # A long but vertically narrow strip. Its x-width is deliberately larger
    # than MAX_SPEED (=2), so the point-mass agent cannot jump across it in a
    # single step. Its narrow y-range means a direct policy hits it often, but
    # not always, and a small vertical detour is easy to learn after AvoidRefine.
    hazard = Box((3.0, 3.75), (5.2, 4.25))

    correct_safe = Not(hazard)
    graph = _single_edge(start, goal, correct_safe)

    fault = inject_overwide_safety(
        graph,
        "start",
        "goal",
        hazard,
        name="overwide_safety",
        severity=severity,
    )

    return FaultBenchmarkCase(
        fault=fault,
        make_env=lambda: HiddenHazardRooms9(
            layout=RoomsLayout(doors={}),
            max_steps=20,
            hidden_unsafe_region=hazard,
        ),
    )


def unreachable_start_case(severity: float = 1.0) -> FaultBenchmarkCase:
    """PastRefine: half of the source lies in an adjacent sealed room."""

    correct_start = Box((1.0, 1.0), (3.5, 3.5))
    unreachable = Box((9.0, 1.0), (11.5, 3.5))
    goal = Box((5.0, 4.5), (7.0, 6.5))

    graph = _single_edge(correct_start, goal)
    fault = inject_unreachable_start(
        graph,
        "start",
        unreachable,
        name="unreachable_start",
        severity=severity,
    )

    return FaultBenchmarkCase(
        fault=fault,
        make_env=lambda: Rooms9(
            layout=RoomsLayout(doors={}),
            max_steps=40,
        ),
    )


def missing_waypoint_case(severity: float = 1.0) -> FaultBenchmarkCase:
    """AddRefine: remove a useful midpoint decomposition from an open task."""

    graph = AbstractGraph()
    start = graph.add_vertex("start", Box((1.0, 10.0), (7.0, 14.0)))
    waypoint = graph.add_vertex("waypoint", Box((11.0, 10.0), (14.0, 14.0)))
    goal = graph.add_vertex("goal", Box((21.0, 11.0), (22.5, 13.0)))

    graph.add_edge(start, waypoint)
    graph.add_edge(waypoint, goal)
    graph.source = start
    graph.target = goal

    fault = inject_missing_waypoint(
        graph,
        "waypoint",
        name="missing_waypoint",
        severity=severity,
    )

    return FaultBenchmarkCase(
        fault=fault,
        make_env=lambda: Rooms9(
            layout=RoomsLayout(doors=_all_open_doors()),
            max_steps=9,
        ),
    )


def missing_alternative_case(severity: float = 1.0) -> FaultBenchmarkCase:
    """OrRefine: remove the A->B bridge needed to bypass a blocked direct edge.

    The geometry is tuned so ``start->a``, ``a->b`` and ``b->goal`` are easy
    while ``a->goal`` is too long for the short per-edge horizon.  There is no
    direct ``start->b`` bypass: the correct graph therefore needs ``a->b`` to
    obtain a high-success start-to-goal path.  Fault injection removes that
    bridge, so OrRefine should reconstruct exactly ``a->b->goal`` and improve
    whole-graph success.
    """

    graph = AbstractGraph()
    start = graph.add_vertex("start", Box((1.0, 11.0), (3.0, 13.0)))
    a = graph.add_vertex("a", Box((4.0, 11.0), (6.0, 13.0)))
    b = graph.add_vertex("b", Box((10.0, 11.0), (12.0, 13.0)))
    goal = graph.add_vertex("goal", Box((19.0, 11.0), (21.0, 13.0)))

    graph.add_edge(start, a)
    graph.add_edge(a, goal)
    graph.add_edge(b, goal)
    graph.add_edge(a, b)
    graph.source = start
    graph.target = goal

    fault = inject_missing_alternative_path(
        graph,
        "a",
        "b",
        name="missing_alternative",
        severity=severity,
    )

    return FaultBenchmarkCase(
        fault=fault,
        make_env=lambda: Rooms9(
            layout=RoomsLayout(doors=_all_open_doors()),
            max_steps=6,
        ),
    )


def all_fault_cases() -> tuple[FaultBenchmarkCase, ...]:
    """Return the five RQ1 benchmark fault types."""

    return (
        goal_trap_case(),
        overwide_safety_case(),
        unreachable_start_case(),
        missing_waypoint_case(),
        missing_alternative_case(),
    )
