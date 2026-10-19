"""Behavioural tests for the heuristic verifier and stochastic placeholder.

Each world is a fully open 10x10 grid with hand-picked walls removed, so the
verdict has one visible cause: a disconnected branch, a partly sealed region, or
a horizon that is deliberately too short.
"""

from __future__ import annotations

import json

import pytest

from faultline.envs.rooms.layout import RoomsLayout, open_doors_except_sealed_rooms
from faultline.envs.rooms.regions import RegionBlock
from faultline.envs.rooms9 import default_doors
from faultline.envs.rooms.stochastic import (
    DIRL_NOT_IMPLEMENTED_MESSAGE,
    DiRLVerifier,
    StochasticReport,
)
from faultline.envs.rooms.verify import (
    VerificationReport,
    verify_heuristic,
)
from faultline.envs.rooms.world import PAPER_TOPOLOGY, RoomsWorld, SpecTopology

ROOM_SIZE = 8.0
REGION_BLOCKS = {
    "Start": RegionBlock(0, 0),
    "M1": RegionBlock(3, 0),
    "M2": RegionBlock(0, 3),
    "M3": RegionBlock(3, 3),
    "M4": RegionBlock(6, 4),
    "M5": RegionBlock(3, 6),
    "Goal": RegionBlock(6, 7),
}
# Walls that seal the M2 block from the rest of the grid.
M2_SEAL = {
    ("v", 1, 3),
    ("v", 1, 4),
    ("h", 0, 2),
    ("h", 1, 2),
    ("h", 0, 4),
    ("h", 1, 4),
}
# Walls that seal the M1 block. Together with M2_SEAL the Start branch is gone
# even though the rest of the grid (M3..Goal) stays mutually connected.
M1_SEAL = {
    ("v", 2, 0),
    ("v", 2, 1),
    ("v", 4, 0),
    ("v", 4, 1),
    ("h", 3, 1),
    ("h", 4, 1),
}
# Walls that seal only the room (7, 8) inside the Goal block. The right and
# top walls matter too: without them the agent can go around through row 9.
GOAL_CORNER_SEAL = {("v", 6, 8), ("v", 7, 8), ("h", 7, 7), ("h", 7, 8)}


def full_doors() -> dict:
    """Return every internal wall on a 10x10 grid as a full-edge door."""
    return open_doors_except_sealed_rooms((), room_size=ROOM_SIZE, rooms_per_side=10)


def make_world(missing: set | None = None, **overrides) -> RoomsWorld:
    """Build a 10x10 world with the given walls removed."""
    doors = full_doors()
    for key in missing or set():
        doors.pop(key, None)
    payload = {
        "name": "verify_world",
        "layout": RoomsLayout(doors, ROOM_SIZE, 10),
        "regions": REGION_BLOCKS,
        "topology": PAPER_TOPOLOGY,
        "max_steps": None,
    }
    payload.update(overrides)
    return RoomsWorld(**payload)


def disconnected_world() -> RoomsWorld:
    """Build a 10x10 world with every internal wall solid."""
    return RoomsWorld(
        name="disconnected",
        layout=RoomsLayout({}, ROOM_SIZE, 10),
        regions=REGION_BLOCKS,
        topology=PAPER_TOPOLOGY,
    )


def test_open_world_is_solvable_with_all_four_routes():
    """The fully open paper layout passes every check and has no dead edge."""
    report = verify_heuristic(make_world())

    assert report.solvable is True
    assert report.reasons == ()
    assert report.component_count == 1
    assert report.main_component_size == 100
    assert all(fraction == 1.0 for fraction in report.region_reachability.values())
    assert len(report.routes) == 4
    assert report.best_route is not None
    assert report.dead_edges == ()
    assert report.max_steps == 200
    assert report.longest_edge_steps == 8


def test_a_split_start_block_is_judged_by_its_best_component():
    """Walling off the Start block's first room must not fail a solvable world.

    Room (0, 0) is the smallest Start room. With it isolated, the other three
    Start rooms still reach every region, so the world is solvable from them.
    """
    report = verify_heuristic(make_world({("v", 0, 0), ("h", 0, 0)}))

    assert report.solvable is True
    assert report.component_count == 2
    assert report.main_component_size == 99
    assert report.region_reachability["Start"] == 0.75
    assert len(report.routes) == 4


def test_fully_disconnected_world_fails_with_a_route_reason():
    """Solid walls leave every region unreachable and no route exists."""
    report = verify_heuristic(disconnected_world())

    assert report.solvable is False
    assert report.component_count == 100
    assert report.main_component_size == 1
    assert report.region_reachability["M1"] == 0.0
    assert report.routes == ()
    assert len(report.dead_edges) == 8
    assert "no feasible Start->Goal route" in report.reasons[0]


def test_one_dead_branch_still_passes_through_the_other():
    """Sealing M2 leaves the M1 routes alive; M2 edges are reported as dead."""
    report = verify_heuristic(make_world(M2_SEAL))

    assert report.solvable is True
    assert report.region_reachability["M2"] == 0.0
    assert report.region_reachability["M1"] == 1.0
    assert len(report.routes) == 2
    assert all("M2" not in route for route in report.routes)
    assert set(report.dead_edges) == {"Start->M2", "M2->M3"}
    # The sealed M2 rooms form their own component; everything else is still
    # one component, so the report distinguishes a dead branch from disconnection.
    assert report.component_count == 2
    assert report.main_component_size == 96
    assert len(report.region_internal_components["M2"]) == 1


def test_partly_sealed_region_is_reported_but_still_solvable():
    """One sealed Goal room leaves a 0.75 reachability fraction."""
    report = verify_heuristic(make_world(GOAL_CORNER_SEAL))

    assert report.solvable is True
    assert report.region_reachability["Goal"] == 0.75
    assert len(report.region_internal_components["Goal"]) == 2
    assert (7, 8) in report.region_internal_components["Goal"][1]


def test_a_short_horizon_rejects_an_otherwise_solvable_world():
    """A 14-crossing edge needs 56 steps, so a 10-step horizon fails."""
    topology = SpecTopology(("Start", "Goal"), (("Start", "Goal"),))
    world = RoomsWorld(
        name="short_horizon",
        layout=RoomsLayout(full_doors(), ROOM_SIZE, 10),
        regions={"Start": RegionBlock(0, 0), "Goal": RegionBlock(8, 8)},
        topology=topology,
        max_steps=10,
    )

    report = verify_heuristic(world)

    assert report.solvable is False
    assert report.longest_edge_steps == 56
    assert report.max_steps == 10
    assert any("horizon is 10" in reason for reason in report.reasons)


def test_report_round_trips_through_json():
    """The verdict and every diagnostic survive serialisation."""
    report = verify_heuristic(make_world(M2_SEAL))

    restored = VerificationReport.from_dict(json.loads(json.dumps(report.to_dict())))

    assert restored.to_dict() == report.to_dict()


def test_verifier_refuses_a_partial_door_gap():
    """The 9-rooms narrow-door layout is not room-analysable."""
    world = RoomsWorld(
        name="partial",
        layout=RoomsLayout(default_doors(), ROOM_SIZE, 3),
        regions={"Start": RegionBlock(0, 0), "Goal": RegionBlock(1, 1)},
        topology=SpecTopology(("Start", "Goal"), (("Start", "Goal"),)),
    )

    with pytest.raises(ValueError, match="full-edge"):
        verify_heuristic(world)


def test_dirl_verifier_raises_the_documented_message():
    """The placeholder must not be mistaken for a passing baseline."""
    with pytest.raises(NotImplementedError) as raised:
        DiRLVerifier().verify(make_world(), seed=0)

    assert str(raised.value) == DIRL_NOT_IMPLEMENTED_MESSAGE


def test_stochastic_report_applies_the_caller_threshold():
    """Passing is the caller's threshold applied to the best-path product."""
    report = StochasticReport(
        learner="ppo",
        seed=3,
        budget=80_000,
        threshold=0.99,
        per_edge={"Start->Goal": 0.995, "Goal->Goal": 0.8},
        best_path=0.995,
    )

    assert report.passed is True
    assert report.to_dict()["passed"] is True

    failing = StochasticReport(
        learner="ppo",
        seed=3,
        budget=80_000,
        threshold=0.99,
        per_edge={"Start->Goal": 0.5},
        best_path=0.5,
    )
    assert failing.passed is False


def test_connected_layout_without_a_spec_route_still_fails():
    """Sealing both Start branches leaves no DAG route, not a disconnected grid."""
    report = verify_heuristic(make_world(M1_SEAL | M2_SEAL))

    assert report.solvable is False
    assert report.routes == ()
    assert report.best_route is None
    assert report.region_reachability["M1"] == 0.0
    assert report.region_reachability["M2"] == 0.0
    assert report.component_count == 3
    assert report.main_component_size == 92
    assert set(report.dead_edges) == {
        f"{source}->{target}" for source, target in PAPER_TOPOLOGY.edges
    }
    assert "no feasible Start->Goal route" in report.reasons[0]


def test_a_route_within_horizon_passes_even_when_the_shortest_route_does_not():
    """The pass rule is "at least one route fits", not "the shortest one fits".

    The Start-M1-Goal route is the shortest overall (1 + 5 crossings) but its
    second edge needs 20 steps. The Start-M2-M3-Goal route is tied on total
    length (2 + 2 + 2) and each edge needs only 8 steps, so it must pass.
    """
    topology = SpecTopology(
        ("Start", "M1", "M2", "M3", "Goal"),
        (
            ("Start", "M1"),
            ("M1", "Goal"),
            ("Start", "M2"),
            ("M2", "M3"),
            ("M3", "Goal"),
        ),
    )
    world = RoomsWorld(
        name="alternative_route",
        layout=RoomsLayout(full_doors(), ROOM_SIZE, 10),
        regions={
            "Start": RegionBlock(0, 0),
            "M1": RegionBlock(2, 0),
            "M2": RegionBlock(0, 3),
            "M3": RegionBlock(3, 3),
            "Goal": RegionBlock(2, 6),
        },
        topology=topology,
        max_steps=8,
    )

    report = verify_heuristic(world)

    assert report.edge_lengths["M1->Goal"] == 5
    assert report.edge_lengths["Start->M2"] == 2
    assert report.edge_lengths["M2->M3"] == 2
    assert report.edge_lengths["M3->Goal"] == 2
    assert report.solvable is True
    assert report.reasons == ()
    assert report.best_route == ("Start", "M2", "M3", "Goal")


def test_horizon_boundary_is_inclusive():
    """An edge that needs exactly max_steps fits; only a strict overrun fails."""
    topology = SpecTopology(("Start", "Goal"), (("Start", "Goal"),))
    world = RoomsWorld(
        name="boundary",
        layout=RoomsLayout(full_doors(), ROOM_SIZE, 10),
        regions={"Start": RegionBlock(0, 0), "Goal": RegionBlock(8, 8)},
        topology=topology,
        max_steps=56,
    )

    report = verify_heuristic(world)

    assert report.longest_edge_steps == 56
    assert report.max_steps == 56
    assert report.solvable is True
    assert report.reasons == ()


def test_dangling_topology_branch_is_dead_but_world_passes():
    """A reachable region with no route to Goal is a dead branch, not a failure."""
    topology = SpecTopology(
        PAPER_TOPOLOGY.vertices + ("M6",),
        PAPER_TOPOLOGY.edges + (("M3", "M6"),),
    )
    regions = dict(REGION_BLOCKS)
    regions["M6"] = RegionBlock(0, 6)
    report = verify_heuristic(make_world(topology=topology, regions=regions))

    assert report.solvable is True
    assert report.region_reachability["M6"] == 1.0
    assert report.dead_edges == ("M3->M6",)
    assert len(report.routes) == 4


def test_cut_rooms_between_start_and_goal_are_reported():
    """A one-room-wide gap makes its two rooms the Start->Goal cut."""
    rooms_per_side = 6
    doors = open_doors_except_sealed_rooms(
        (), room_size=ROOM_SIZE, rooms_per_side=rooms_per_side
    )
    # A solid wall between row 2 and row 3, except at column 2.
    for col in range(rooms_per_side):
        if col != 2:
            doors.pop(("h", col, 2), None)
    world = RoomsWorld(
        name="corridor",
        layout=RoomsLayout(doors, ROOM_SIZE, rooms_per_side),
        regions={"Start": RegionBlock(0, 0), "Goal": RegionBlock(4, 4)},
        topology=SpecTopology(("Start", "Goal"), (("Start", "Goal"),)),
    )

    report = verify_heuristic(world)

    assert report.solvable is True
    assert set(report.cut_rooms) == {(2, 2), (2, 3)}


def test_verifier_requires_start_and_goal_vertices():
    """A topology without the two named endpoints cannot be analysed."""
    world = RoomsWorld(
        name="no_start",
        layout=RoomsLayout(full_doors(), ROOM_SIZE, 10),
        regions={"A": RegionBlock(0, 0), "B": RegionBlock(3, 3)},
        topology=SpecTopology(("A", "B"), (("A", "B"),)),
    )

    with pytest.raises(ValueError, match="Start"):
        verify_heuristic(world)


def test_unsolvable_report_round_trips_through_json():
    """A report with no routes and no best route survives serialisation."""
    report = verify_heuristic(disconnected_world())
    assert report.best_route is None
    assert report.routes == ()

    restored = VerificationReport.from_dict(json.loads(json.dumps(report.to_dict())))

    assert restored.to_dict() == report.to_dict()
