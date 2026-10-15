from __future__ import annotations

import numpy as np

from faultline.envs import RoomsLayout
from faultline.faults.environments import HiddenHazardRooms9
from faultline.spectrl.predicates import Box, Not, Universe


def _make_env() -> tuple[HiddenHazardRooms9, Box]:
    hazard = Box((3.0, 3.0), (5.0, 5.0))
    env = HiddenHazardRooms9(
        layout=RoomsLayout(doors={}),
        max_steps=20,
        hidden_unsafe_region=hazard,
        hidden_hazard_reward=0.0,
    )
    return env, hazard


def test_hidden_hazard_is_reported_but_not_a_spec_violation():
    env, _hazard = _make_env()
    env.set_abstract_states(
        Universe(),
        Box((6.0, 3.0), (7.0, 5.0)),
        Universe(),
    )
    env.reset(seed=0)

    # Put the agent immediately left of the hidden hazard and move into it.
    env._position = np.array([2.9, 4.0], dtype=float)
    env._goal = np.array([6.5, 4.0], dtype=float)

    _obs, reward, terminated, truncated, info = env.step(
        np.array([0.2, 0.0], dtype=float)
    )

    assert terminated is True
    assert truncated is False
    assert reward == 0.0
    assert info["hidden_hazard"] is True
    assert info["is_safe"] is False
    assert info["is_success"] is False


def test_refined_safe_predicate_turns_same_hazard_into_spec_violation():
    env, hazard = _make_env()
    env.set_abstract_states(
        Universe(),
        Box((6.0, 3.0), (7.0, 5.0)),
        Not(hazard),
    )
    env.reset(seed=0)

    env._position = np.array([2.9, 4.0], dtype=float)
    env._goal = np.array([6.5, 4.0], dtype=float)

    _obs, reward, terminated, truncated, info = env.step(
        np.array([0.2, 0.0], dtype=float)
    )

    assert terminated is True
    assert truncated is False
    assert reward < 0.0
    assert info["is_safe"] is False
    assert info["is_success"] is False
