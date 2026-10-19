"""Catalogue tests for the frozen random n-rooms worlds.

Every random world is pinned by door count, orientation and all seven region
blocks. A second test regenerates each from its seed, so a deliberate generator
change must regenerate the stored files in the same commit.
"""

from __future__ import annotations

import pytest

from faultline.envs.rooms.catalogue import (
    FIG3_DOORS,
    FIG3_REGIONS,
    FIG6_DOORS,
    FIG6_REGIONS,
    FIG6_WALLS,
    FROZEN_SEEDS,
    build_world,
    fig3_world,
    fig6_world,
    freeze_world,
    frozen_name,
    load_world,
    world_names,
)
from faultline.envs.rooms.generator import generate_world
from faultline.envs.rooms.verify import verify_heuristic
from faultline.envs.rooms.world import world_from_dict, world_to_dict

# seed -> (door count, orientation, region blocks)
EXPECTED = {
    0: (
        93,
        "TR-BL",
        {
            "Start": (8, 8),
            "M1": (1, 6),
            "M2": (5, 6),
            "M3": (6, 1),
            "M4": (6, 8),
            "M5": (1, 8),
            "Goal": (0, 0),
        },
    ),
    1: (
        91,
        "BR-TL",
        {
            "Start": (8, 0),
            "M1": (4, 1),
            "M2": (7, 3),
            "M3": (1, 5),
            "M4": (2, 2),
            "M5": (7, 5),
            "Goal": (0, 8),
        },
    ),
    2: (
        92,
        "BL-TR",
        {
            "Start": (0, 0),
            "M1": (1, 6),
            "M2": (6, 4),
            "M3": (3, 6),
            "M4": (2, 1),
            "M5": (6, 6),
            "Goal": (8, 8),
        },
    ),
    3: (
        90,
        "TR-BL",
        {
            "Start": (8, 8),
            "M1": (0, 4),
            "M2": (0, 8),
            "M3": (6, 1),
            "M4": (8, 1),
            "M5": (2, 3),
            "Goal": (0, 0),
        },
    ),
    4: (
        94,
        "TL-BR",
        {
            "Start": (0, 8),
            "M1": (1, 5),
            "M2": (6, 6),
            "M3": (7, 4),
            "M4": (0, 1),
            "M5": (2, 3),
            "Goal": (8, 0),
        },
    ),
}


@pytest.mark.parametrize("seed", FROZEN_SEEDS)
def test_frozen_world_builds_verifies_and_pins_its_data(seed):
    """Each stored world keeps its door count, orientation and region blocks."""
    doors, orientation, regions = EXPECTED[seed]

    world = load_world(frozen_name(seed))

    assert len(world.layout.doors) == doors
    assert world.meta["orientation"] == orientation
    assert {name: (block.col, block.row) for name, block in world.regions.items()} == regions
    assert verify_heuristic(world).solvable is True


@pytest.mark.parametrize("seed", FROZEN_SEEDS)
def test_frozen_world_round_trips_through_the_serializer(seed):
    """A stored world survives a decode and encode unchanged."""
    world = load_world(frozen_name(seed))

    assert world_to_dict(world_from_dict(world_to_dict(world))) == world_to_dict(world)


@pytest.mark.parametrize("seed", FROZEN_SEEDS)
def test_frozen_seed_matches_a_fresh_generation(seed):
    """The stored file is exactly what the current generator produces."""
    frozen = load_world(frozen_name(seed))
    fresh = generate_world(seed)

    assert world_to_dict(frozen) == world_to_dict(fresh)


def test_registry_lists_every_fixed_world():
    """The registry names are stable and build independent worlds."""
    assert world_names() == tuple(frozen_name(seed) for seed in FROZEN_SEEDS) + (
        "fig3",
        "fig6",
    )
    first = build_world("random_seed0")
    second = build_world("random_seed0")

    assert first is not second
    assert world_to_dict(first) == world_to_dict(second)
    with pytest.raises(KeyError, match="Unknown fixed world"):
        build_world("missing")


def test_freeze_world_writes_a_loadable_file(tmp_path):
    """The freezing helper writes the same data the loader reads back."""
    world = generate_world(0)

    path = freeze_world(world, tmp_path, filename="random_seed0")

    assert path.name == "random_seed0.json"
    assert world_to_dict(load_world("random_seed0", tmp_path)) == world_to_dict(world)


def test_fig3_pins_the_designed_layout_and_its_symptoms():
    """The Fig. 3 replica keeps its door map and predicted refinement symptoms."""
    world = build_world("fig3")
    report = verify_heuristic(world)

    assert len(world.layout.doors) == len(FIG3_DOORS) == 30
    assert {name: (b.col, b.row) for name, b in world.regions.items()} == {
        name: (b.col, b.row) for name, b in FIG3_REGIONS.items()
    }
    assert report.solvable is True
    assert report.region_reachability["M1"] == 0.75
    assert report.region_reachability["M3"] == 0.5
    assert report.region_reachability["M5"] == 0.0
    assert len(report.region_internal_components["M1"]) == 2
    assert len(report.region_internal_components["M3"]) == 2
    assert report.edge_lengths["M1->M3"] > report.edge_lengths["M2->M3"]
    assert report.best_route == ("Start", "M2", "M3", "M4", "Goal")
    assert set(report.dead_edges) == {"M3->M5", "M5->Goal"}
    assert len(report.cut_rooms) == 10


def test_fig3_file_matches_its_builder():
    """The frozen Fig. 3 file is exactly what the builder produces."""
    assert world_to_dict(load_world("fig3")) == world_to_dict(fig3_world())


def test_fig6_pins_the_transcribed_layout_and_its_report():
    """The Fig. 6a transcription keeps its walls, regions and diagnostics."""
    world = build_world("fig6")
    report = verify_heuristic(world)

    assert len(world.layout.doors) == len(FIG6_DOORS) == 123
    assert len(FIG6_WALLS) == 57
    assert set(world.layout.doors).isdisjoint(FIG6_WALLS)
    assert {name: (b.col, b.row) for name, b in world.regions.items()} == {
        name: (b.col, b.row) for name, b in FIG6_REGIONS.items()
    }
    assert report.solvable is True
    assert report.routes == (
        ("Start", "M1", "M3", "M4", "Goal"),
        ("Start", "M1", "M3", "M5", "Goal"),
        ("Start", "M2", "M3", "M4", "Goal"),
        ("Start", "M2", "M3", "M5", "Goal"),
    )
    assert report.best_route == ("Start", "M1", "M3", "M4", "Goal")
    assert report.dead_edges == ()
    assert dict(report.region_reachability) == {
        "Start": 1.0,
        "M1": 1.0,
        "M2": 0.75,
        "M3": 1.0,
        "M4": 1.0,
        "M5": 0.75,
        "Goal": 0.75,
    }
    assert dict(report.edge_lengths) == {
        "Start->M1": 3,
        "Start->M2": 3,
        "M1->M3": 3,
        "M2->M3": 6,
        "M3->M4": 6,
        "M3->M5": 9,
        "M4->Goal": 4,
        "M5->Goal": 4,
    }
    assert report.cut_rooms == ()
    assert report.max_steps == 200


def test_fig6_round_trips_through_the_serializer():
    """The stored Fig. 6 world survives a decode and encode unchanged."""
    world = load_world("fig6")

    assert world_to_dict(world_from_dict(world_to_dict(world))) == world_to_dict(world)
    assert world.meta["source"] == "paper Fig. 6a"


def test_fig6_file_matches_its_builder():
    """The frozen Fig. 6 file is exactly what the builder produces."""
    assert world_to_dict(load_world("fig6")) == world_to_dict(fig6_world())


KNOWN_OPERATORS = {"seq", "add", "past", "or"}


def test_fig3_expectations_match_the_layout_diagnostics():
    """Each expected operator is backed by a symptom the verifier measures.

    ``seq`` needs a partly unreachable target, ``past`` a source split into
    several components, and ``or`` a sibling parent of the target. An expectation
    that no diagnostic supports would be invented, so it fails here.
    """
    world = build_world("fig3")
    report = verify_heuristic(world)
    expected = world.meta["expected_operators"]
    edges = {f"{source}->{target}" for source, target in world.topology.edges}

    assert expected
    for edge, operators in expected.items():
        source, target = edge.split("->")
        assert edge in edges
        assert operators and set(operators) <= KNOWN_OPERATORS
        if "seq" in operators:
            assert report.region_reachability[target] < 1.0
        if "past" in operators:
            assert len(report.region_internal_components[source]) > 1
        if "or" in operators:
            assert len(world.topology.predecessors(target)) > 1


def test_fig3_dead_branch_has_no_expected_operator():
    """The sealed M5 branch has no witness for any operator, so none is expected."""
    world = build_world("fig3")
    report = verify_heuristic(world)

    assert set(report.dead_edges) == {"M3->M5", "M5->Goal"}
    assert not set(report.dead_edges) & set(world.meta["expected_operators"])


def test_fig6_carries_no_designed_expectations():
    """The transcribed Fig. 6a instance has no operator map, so none is recorded."""
    assert "expected_operators" not in build_world("fig6").meta
