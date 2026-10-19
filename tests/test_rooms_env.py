"""Behavioural tests for the generic environment and its derived horizon.

The 10x10 layout is fully open where a test needs a door and fully solid where
it needs isolation, so each assertion has one cause.
"""

from __future__ import annotations

import numpy as np

from faultline.envs import Rooms9
from faultline.envs.rooms.env import RoomsEnv
from faultline.envs.rooms.horizon import default_max_steps
from faultline.envs.rooms.layout import RoomsLayout, open_doors_except_sealed_rooms
from faultline.envs.rooms.regions import RegionBlock
from faultline.envs.rooms.world import RoomsWorld, SpecTopology
from faultline.experiments.training import make_eval_env
from faultline.spectrl.predicates import Box, Universe

ROOM_SIZE = 8.0


def full_doors(rooms_per_side: int = 10) -> dict:
    """Return every internal wall as a full-edge door."""
    return open_doors_except_sealed_rooms(
        (), room_size=ROOM_SIZE, rooms_per_side=rooms_per_side
    )


def open_layout() -> RoomsLayout:
    """Return a fully open 10x10 layout."""
    return RoomsLayout(full_doors(), ROOM_SIZE, 10)


def test_generic_env_uses_the_layout_world_size():
    """A 10x10 layout gives an 80-unit world and a 4-D observation."""
    env = RoomsEnv(layout=open_layout(), max_steps=500)

    assert env.layout.world_size == 80.0
    assert env.observation_space.shape == (4,)
    assert env.observation_space.high.tolist()[:2] == [80.0, 80.0]
    assert env.max_steps == 500


def test_full_edge_door_passes_and_solid_wall_blocks():
    """Room-grid motion follows the same door map the analysis uses."""
    open_grid = open_layout()
    assert open_grid.segment_is_blocked([7.0, 4.0], [9.0, 4.0]) is False

    solid = RoomsLayout({}, ROOM_SIZE, 10)
    assert solid.segment_is_blocked([7.0, 4.0], [9.0, 4.0]) is True


def test_generic_env_truncates_at_its_horizon():
    """A static agent truncates at max_steps without success or termination."""
    env = RoomsEnv(layout=RoomsLayout({}, ROOM_SIZE, 10), max_steps=3)
    env.set_abstract_states(
        Box((2.0, 2.0), (2.2, 2.2)), Box((70.0, 70.0), (71.0, 71.0)), Universe()
    )
    env.reset(seed=0)

    for _ in range(3):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0]))

    assert terminated is False
    assert truncated is True
    assert info["is_success"] is False


def test_make_eval_env_preserves_the_horizon_and_class():
    """Evaluation must not silently fall back to a 9-rooms step limit."""
    env = RoomsEnv(layout=open_layout(), max_steps=777)
    evaluation = make_eval_env(env)

    assert type(evaluation) is RoomsEnv
    assert evaluation.max_steps == 777
    assert evaluation.layout is env.layout

    rooms9 = Rooms9(max_steps=321)
    rooms9_evaluation = make_eval_env(rooms9)
    assert type(rooms9_evaluation) is Rooms9
    assert rooms9_evaluation.max_steps == 321


def test_default_max_steps_uses_the_longest_edge_and_the_slack():
    """A block-to-block edge of 14 crossings yields ceil(14*8/2*4) steps."""
    from faultline.envs.rooms.grid import RoomGrid

    room_grid = RoomGrid(10, full_doors(), ROOM_SIZE)
    start = RegionBlock(0, 0)
    goal = RegionBlock(8, 8)

    steps = default_max_steps(
        room_grid, [(start.rooms(), goal.rooms())], speed=2.0, slack=4.0, floor=0
    )

    assert room_grid.shortest_path(start.rooms(), goal.rooms()).length == 14
    assert steps == 224


def test_default_max_steps_never_drops_below_the_floor():
    """A zero-length edge still gets the 200-step minimum."""
    from faultline.envs.rooms.grid import RoomGrid

    room_grid = RoomGrid(10, full_doors(), ROOM_SIZE)
    block = RegionBlock(0, 0)

    assert default_max_steps(room_grid, [(block.rooms(), block.rooms())]) == 200


def test_default_max_steps_skips_a_dead_edge_and_keeps_the_reachable_one():
    """A dead branch does not erase a horizon that a live edge supplies."""
    from faultline.envs.rooms.grid import RoomGrid

    doors = full_doors()
    doors.pop(("v", 8, 9))
    doors.pop(("h", 9, 8))
    room_grid = RoomGrid(10, doors, ROOM_SIZE)
    start = RegionBlock(0, 0)
    goal = RegionBlock(8, 8)

    steps = default_max_steps(
        room_grid,
        [(start.rooms(), goal.rooms()), (((9, 9),), ((0, 9),))],
        speed=2.0,
        slack=4.0,
        floor=0,
    )

    # The live Start -> Goal edge is 14 crossings; the sealed (9, 9) adds nothing.
    assert room_grid.shortest_path({(9, 9)}, {(0, 9)}) is None
    assert steps == 224


def test_default_max_steps_falls_back_to_the_floor_when_nothing_connects():
    """A fully disconnected world reports the minimum rather than raising."""
    from faultline.envs.rooms.grid import RoomGrid

    room_grid = RoomGrid(10, {}, ROOM_SIZE)

    assert default_max_steps(room_grid, [(((0, 0),), ((9, 9),))]) == 200


def _two_region_world(max_steps: int | None) -> RoomsWorld:
    """Build a full-edge world whose only edge crosses 14 rooms."""
    return RoomsWorld(
        name="horizon",
        layout=open_layout(),
        regions={"Start": RegionBlock(0, 0), "Goal": RegionBlock(8, 8)},
        topology=SpecTopology(("Start", "Goal"), (("Start", "Goal"),)),
        max_steps=max_steps,
    )


def test_world_derives_the_horizon_when_none_is_given():
    """The world passes its derived horizon to the environment."""
    world = _two_region_world(None)

    assert world.effective_max_steps() == 224
    assert world.make_env().max_steps == 224


def test_world_keeps_an_explicit_horizon():
    """An explicit limit is copied unchanged."""
    world = _two_region_world(500)

    assert world.effective_max_steps() == 500
    assert world.make_env().max_steps == 500


def test_grid_default_speed_matches_the_environment_speed():
    """The horizon module keeps its own copy of the speed; the two must agree."""
    from faultline.envs.rooms.env import MAX_SPEED
    from faultline.envs.rooms.horizon import DEFAULT_SPEED

    assert DEFAULT_SPEED == MAX_SPEED
