from __future__ import annotations

import numpy as np

from faultline.envs import RoomsLayout
from faultline.faults.environments import HiddenHazardRooms9
from faultline.graph import AbstractGraph
from faultline.learners.ppo import PPOEdgeLearner
from faultline.spectrl.predicates import Box, Not, Universe


def _make_env() -> tuple[HiddenHazardRooms9, Box]:
    hazard = Box((3.0, 3.0), (5.0, 5.0))
    env = HiddenHazardRooms9(
        layout=RoomsLayout(doors={}),
        max_steps=20,
        hidden_unsafe_region=hazard,
    )
    return env, hazard


def test_hidden_hazard_is_reported_without_training_termination():
    """The faulty specification must not train PPO to avoid the hidden hazard."""

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

    # The benchmark ground truth labels the state unsafe ...
    assert info["hidden_hazard"] is True
    assert info["is_safe"] is False
    assert info["is_success"] is False

    # ... but the faulty training objective is not allowed to learn that merely
    # from termination. Training therefore continues normally.
    assert terminated is False
    assert truncated is False

    # The reward should be an ordinary progress-shaped reward, not the strong
    # Rooms9 violation penalty.
    assert reward > -1.0


def test_refined_safe_predicate_turns_same_hazard_into_spec_violation():
    """After AvoidRefine excludes the hazard, normal safety semantics return."""

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


def test_edge_evaluation_stops_at_first_hidden_hazard_state():
    """AvoidRefine should receive a tail ending at the hidden hazard.

    The environment itself keeps the training episode alive, but
    PPOEdgeLearner._episode stops a reach-avoid evaluation as soon as the
    environment reports that the observed state is unsafe.
    """

    env, hazard = _make_env()

    graph = AbstractGraph()
    start = graph.add_vertex("start", Universe())
    goal = graph.add_vertex(
        "goal",
        Box((6.0, 3.0), (7.0, 5.0)),
    )
    edge = graph.add_edge(
        start,
        goal,
        Universe(),
    )

    env.set_abstract_states(
        edge.source.region,
        edge.target.region,
        edge.safe,
    )
    env.reset(seed=0)

    env._position = np.array([2.9, 4.0], dtype=float)
    env._goal = np.array([6.5, 4.0], dtype=float)
    observation = env._observation()

    learner = PPOEdgeLearner()

    def move_right(_observation: np.ndarray) -> np.ndarray:
        return np.array([0.2, 0.0], dtype=float)

    rollout = learner._episode(
        env,
        edge,
        move_right,
        observation,
    )

    assert rollout.safe is False
    assert rollout.reached is False

    # The recorded rollout must stop on the first hazard state, rather than
    # continuing to the eventual goal. This keeps AvoidRefine's last-k tail
    # localized around the actual safety failure.
    final_position = rollout.states[-1][:2]
    assert hazard.evaluate(final_position)
    assert len(rollout.states) == 2
