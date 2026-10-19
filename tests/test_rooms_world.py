"""Behavioural tests for region blocks, topology and the immutable world record.

The world uses a fully open 10x10 grid so the physical layout cannot mask a
topology or serialisation bug. Region positions are the hand-picked Fig. 3
blocks, read as ``(col, row)`` from the bottom-left.
"""

from __future__ import annotations

import json

import pytest

from faultline.envs.rooms.layout import RoomsLayout, open_doors_except_sealed_rooms
from faultline.envs.rooms.regions import RegionBlock, valid_blocks
from faultline.envs.rooms9 import default_doors
from faultline.envs.rooms.world import (
    PAPER_REGION_ORDER,
    PAPER_TOPOLOGY,
    RoomsWorld,
    SpecTopology,
    paper_world_name,
    world_from_dict,
    world_to_dict,
)
from faultline.spectrl.predicates import Box

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


def full_doors(rooms_per_side: int = 10) -> dict:
    """Return every internal wall as a full-edge door."""
    return open_doors_except_sealed_rooms(
        (), room_size=ROOM_SIZE, rooms_per_side=rooms_per_side
    )


def open_world(**overrides) -> RoomsWorld:
    """Build a fully open 10x10 paper-topology world."""
    payload = {
        "name": "test_world",
        "layout": RoomsLayout(full_doors(), ROOM_SIZE, 10),
        "regions": REGION_BLOCKS,
        "topology": PAPER_TOPOLOGY,
        "max_steps": 1234,
        "meta": {"seed": 1, "orientation": "TL-BR"},
    }
    payload.update(overrides)
    return RoomsWorld(**payload)


def test_region_block_maps_two_rooms_to_a_continuous_box():
    """A block's box spans two rooms from its bottom-left room."""
    block = RegionBlock(3, 4)

    box = block.to_box(ROOM_SIZE)

    assert block.rooms() == ((3, 4), (4, 4), (3, 5), (4, 5))
    assert box.lower.tolist() == [24.0, 32.0]
    assert box.upper.tolist() == [40.0, 48.0]


def test_region_inset_pulls_every_side_inward():
    """The inset parameter reproduces the 9-rooms one-unit margin."""
    box = RegionBlock(0, 0).to_box(ROOM_SIZE, inset=1.0)

    assert box.lower.tolist() == [1.0, 1.0]
    assert box.upper.tolist() == [15.0, 15.0]


def test_region_overlap_allows_touching_blocks_but_not_sharing_rooms():
    """Blocks that share only a wall do not overlap; one shared room does."""
    assert RegionBlock(0, 0).overlaps(RegionBlock(2, 0)) is False
    assert RegionBlock(0, 0).overlaps(RegionBlock(0, 2)) is False
    assert RegionBlock(0, 0).overlaps(RegionBlock(1, 0)) is True
    assert RegionBlock(0, 0).overlaps(RegionBlock(1, 1)) is True


def test_valid_blocks_enumerates_every_2x2_position_in_a_fixed_order():
    """A 10x10 grid has 81 blocks, from (0, 0) to (8, 8)."""
    blocks = valid_blocks(10)

    assert len(blocks) == 81
    assert blocks[0] == RegionBlock(0, 0)
    assert blocks[-1] == RegionBlock(8, 8)
    assert valid_blocks(2) == (RegionBlock(0, 0),)
    with pytest.raises(ValueError, match="at least 2"):
        valid_blocks(1)


def test_paper_topology_is_the_eight_edge_dag():
    """The paper's topology has seven vertices and the eight expected edges."""
    assert PAPER_TOPOLOGY.vertices == PAPER_REGION_ORDER
    assert len(PAPER_TOPOLOGY.edges) == 8
    assert PAPER_TOPOLOGY.successors("Start") == ("M1", "M2")
    assert PAPER_TOPOLOGY.predecessors("M3") == ("M1", "M2")
    assert PAPER_TOPOLOGY.successors("Goal") == ()


def test_topology_rejects_a_cycle():
    """A cyclic specification is not a DAG and is refused."""
    with pytest.raises(ValueError, match="acyclic"):
        SpecTopology(("A", "B"), (("A", "B"), ("B", "A")))


def test_topology_rejects_unknown_vertices_and_duplicates():
    """Malformed topologies fail at construction."""
    with pytest.raises(ValueError, match="unknown vertex"):
        SpecTopology(("A",), (("A", "B"),))
    with pytest.raises(ValueError, match="unique"):
        SpecTopology(("A", "A"), ())


def test_world_builds_the_specification_graph():
    """The graph has one vertex per region and one edge per topology edge."""
    world = open_world()

    graph = world.build_graph()

    assert [vertex.name for vertex in graph.vertices] == list(PAPER_REGION_ORDER)
    assert len(graph.edges) == 8
    assert graph.source.name == "Start"
    assert graph.target.name == "Goal"
    start = next(vertex for vertex in graph.vertices if vertex.name == "Start")
    assert start.region == Box((0.0, 0.0), (16.0, 16.0))


def test_world_makes_an_environment_at_the_requested_horizon():
    """A world over a 10x10 grid has an 80-unit world and keeps its max_steps."""
    world = open_world()

    env = world.make_env()

    assert env.layout.world_size == 80.0
    assert env.max_steps == 1234


def test_world_grid_has_all_180_full_edge_doors():
    """The open 10x10 world exposes 180 room-level door connections."""
    grid = open_world().grid()

    assert len(grid.doors) == 180
    assert len(grid.components()) == 1
    assert grid.shortest_path({(0, 0)}, {(9, 9)}).length == 18


def test_world_round_trips_through_json():
    """Every field survives a JSON encode and decode."""
    world = open_world(hazards=(Box((4.0, 4.0), (8.0, 8.0)),))

    payload = json.loads(json.dumps(world_to_dict(world)))
    restored = world_from_dict(payload)

    assert restored.name == world.name
    assert restored.layout.doors == world.layout.doors
    assert restored.layout.room_size == world.layout.room_size
    assert restored.layout.rooms_per_side == world.layout.rooms_per_side
    assert dict(restored.regions) == dict(world.regions)
    assert restored.topology == world.topology
    assert restored.hazards == world.hazards
    assert restored.max_steps == 1234
    assert dict(restored.meta) == {"seed": 1, "orientation": "TL-BR"}
    assert restored.to_dict() == world_to_dict(world)


def test_world_rejects_a_region_outside_the_grid():
    """A region block must fit inside the layout."""
    with pytest.raises(ValueError, match="outside"):
        open_world(regions={**REGION_BLOCKS, "Goal": RegionBlock(9, 7)})


def test_world_requires_a_block_for_every_topology_vertex():
    """A topology vertex without a region would be unlabelled."""
    regions = dict(REGION_BLOCKS)
    del regions["M5"]

    with pytest.raises(ValueError, match="no region block"):
        open_world(regions=regions)


def test_world_grid_refuses_a_partial_door_gap():
    """The 9-rooms centred-door layout is not analysable at room level."""
    world = RoomsWorld(
        name="partial",
        layout=RoomsLayout(default_doors(), ROOM_SIZE, 3),
        regions={"Start": RegionBlock(0, 0), "Goal": RegionBlock(1, 1)},
        topology=SpecTopology(("Start", "Goal"), (("Start", "Goal"),)),
    )

    with pytest.raises(ValueError, match="full-edge"):
        world.grid()


def test_random_world_names_are_pure_functions_of_the_seed():
    """A worker can rebuild a random world from its registry name."""
    assert paper_world_name(0) == "rooms100_random_0"
    assert paper_world_name(4) == "rooms100_random_4"


def test_region_inset_shrinks_every_region_box_and_round_trips():
    """The world's inset applies to all its regions and survives serialisation."""
    whole = open_world()
    inset = open_world(region_inset=1.0)

    assert whole.region_box("M1").lower.tolist() == [24.0, 0.0]
    assert inset.region_box("M1").lower.tolist() == [25.0, 1.0]
    assert inset.region_box("M1").upper.tolist() == [39.0, 15.0]
    assert world_from_dict(world_to_dict(inset)).region_inset == 1.0
