"""Behavioural tests for the 9-rooms environment.

The dynamics, wall collision, safety violation, and success tests use small
regions and exact coordinates, so each expected outcome is derived by hand
rather than from the implementation.
"""

from __future__ import annotations

import math

import numpy as np

from faultline.envs import Rooms9, RoomsLayout
from faultline.spectrl.predicates import And, Box, Disc, Universe

# A layout with no doors: every internal wall is solid. It keeps moves inside
# one room, so the tests isolate the dynamics from the door model.
SOLID_LAYOUT = RoomsLayout(doors={})

FAR_TARGET = Box((20.0, 20.0), (22.0, 22.0))


def test_observation_and_action_spaces_have_the_documented_shape():
    """The state is 4-D and the action is 2-D."""
    env = Rooms9()
    assert env.observation_space.shape == (4,)
    assert env.action_space.shape == (2,)


def test_reset_returns_a_four_component_state_inside_the_source():
    """A reset returns a 4-D state whose position lies in the source region."""
    env = Rooms9()
    source = Box((2.0, 2.0), (4.0, 4.0))
    env.set_abstract_states(source, Universe(), Universe())
    observation, info = env.reset(seed=0)
    assert observation.shape == (4,)
    assert source.evaluate(observation[:2]) is True
    assert info["is_success"] is False


def test_step_applies_point_mass_dynamics():
    """A free step moves the agent by speed times heading."""
    env = Rooms9(layout=SOLID_LAYOUT)
    env.set_abstract_states(Box((2.0, 2.0), (4.0, 4.0)), FAR_TARGET, Universe())
    observation, _ = env.reset(seed=0)
    start = observation[:2].copy()
    speed, heading = 0.5, 0.0
    observation, _reward, _terminated, _truncated, _info = env.step(
        np.array([speed, heading], dtype=np.float32)
    )
    expected = start + np.array([speed * math.cos(heading), speed * math.sin(heading)])
    assert np.allclose(observation[:2], expected, atol=1e-5)


def test_step_into_a_solid_wall_keeps_the_agent_in_place():
    """A move that crosses a solid wall leaves the position unchanged."""
    env = Rooms9(layout=SOLID_LAYOUT)
    env.set_abstract_states(Box((7.0, 3.0), (7.5, 3.5)), FAR_TARGET, Universe())
    observation, _ = env.reset(seed=0)
    start = observation[:2].copy()
    observation, _reward, _terminated, _truncated, _info = env.step(
        np.array([2.0, 0.0], dtype=np.float32)
    )
    assert np.allclose(observation[:2], start)


def test_leaving_the_safe_region_ends_the_episode_without_success():
    """A step outside the safe region terminates without success."""
    env = Rooms9(layout=SOLID_LAYOUT)
    env.set_abstract_states(
        Box((2.5, 2.5), (2.6, 2.6)), FAR_TARGET, Box((2.0, 2.0), (3.0, 3.0))
    )
    _observation, _ = env.reset(seed=1)
    _observation, _reward, terminated, _truncated, info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    assert terminated is True
    assert info["is_success"] is False


def test_reaching_the_sampled_goal_reports_success():
    """Steering to the sampled goal point ends the episode with is_success."""
    env = Rooms9(layout=SOLID_LAYOUT)
    env.set_abstract_states(Box((2.0, 2.0), (2.2, 2.2)), Box((3.0, 3.0), (3.5, 3.5)), Universe())
    observation, _ = env.reset(seed=0)
    info = {"is_success": False}
    for _ in range(50):
        offset = env.current_goal - observation[:2]
        distance = float(np.linalg.norm(offset))
        heading = float(math.atan2(offset[1], offset[0]))
        observation, _reward, terminated, truncated, info = env.step(
            np.array([min(2.0, distance), heading], dtype=np.float32)
        )
        if terminated or truncated:
            break
    assert info["is_success"] is True


def test_episode_truncates_at_the_step_limit():
    """A far, unmoving episode truncates at the step limit without success."""
    env = Rooms9(layout=SOLID_LAYOUT, max_steps=3)
    env.set_abstract_states(Box((2.0, 2.0), (2.1, 2.1)), FAR_TARGET, Universe())
    env.reset(seed=0)
    action = np.array([0.0, 0.0], dtype=np.float32)
    for _ in range(3):
        _observation, _reward, terminated, truncated, info = env.step(action)
    assert terminated is False
    assert truncated is True
    assert info["is_success"] is False


def test_reset_with_the_same_seed_repeats_the_start():
    """Two resets with one seed give the same first state."""
    env = Rooms9(layout=SOLID_LAYOUT)
    env.set_abstract_states(Box((2.0, 2.0), (4.0, 4.0)), FAR_TARGET, Universe())
    first, _ = env.reset(seed=7)
    second, _ = env.reset(seed=7)
    assert np.allclose(first, second)


def test_reset_samples_a_goal_inside_a_small_composite_region():
    """A reset samples the goal inside a small box-and-disc region.

    Rejection sampling over the whole world almost never hits the disc, so this
    pins the bounds-aware sampling path that refined regions rely on.
    """
    env = Rooms9(layout=SOLID_LAYOUT)
    target = And(Box((1.0, 1.0), (3.0, 3.0)), Disc((2.0, 2.0), 0.6))
    env.set_abstract_states(Box((1.0, 1.0), (1.5, 1.5)), target, Universe())

    for seed in range(20):
        env.reset(seed=seed)
        assert target.evaluate(env.current_goal) is True


def test_entering_the_trap_ends_the_episode_as_an_unsafe_violation():
    """A step that lands in the trap terminates the episode without success."""
    trap = Box((2.0, 2.0), (3.0, 3.0))
    env = Rooms9(layout=SOLID_LAYOUT, unsafe_region=trap)
    env.set_abstract_states(Box((1.4, 2.4), (1.6, 2.6)), FAR_TARGET, Universe())
    env.reset(seed=0)
    _observation, _reward, terminated, _truncated, info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    assert terminated is True
    assert info["is_safe"] is False
    assert info["is_success"] is False


def test_without_a_trap_the_same_move_is_free_and_safe():
    """With no trap the agent moves normally and the step stays safe."""
    env = Rooms9(layout=SOLID_LAYOUT)
    env.set_abstract_states(Box((1.4, 2.4), (1.6, 2.6)), FAR_TARGET, Universe())
    observation, _ = env.reset(seed=0)
    before = observation[:2].copy()
    observation, _reward, terminated, _truncated, info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    assert not np.allclose(observation[:2], before)
    assert info["is_safe"] is True
    assert terminated is False
