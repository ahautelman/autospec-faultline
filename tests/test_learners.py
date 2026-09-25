"""Behavioural tests for the learner protocol and its two implementations.

The protocol contract test runs against both learners. This proves the
operators can consume either backend without change. The PPO case runs only when
Stable-Baselines3 is installed, which is the documented condition for the
optional interop extra.
"""

from __future__ import annotations

import numpy as np
import pytest

from faultline.envs import Rooms9, RoomsLayout, default_regions
from faultline.graph.abstract_graph import Edge, Vertex
from faultline.learners import (
    CannotRefineError,
    EdgeEstimate,
    PPOEdgeLearner,
    Rollout,
    TabularLearner,
)
from faultline.learners.ppo import DEFAULT_RECIPE
from faultline.spectrl.predicates import Universe

SOLID_LAYOUT = RoomsLayout(doors={})
REGIONS = default_regions()
CONTRACT_ROLLOUTS = 2


def _edge(source_region, target_region, safe=None) -> Edge:
    """Build a reach-avoid edge between two freshly named vertices."""
    safe_region = Universe() if safe is None else safe
    return Edge(Vertex("source", source_region), Vertex("target", target_region), safe_region)


def _tabular_estimate(seed: int) -> EdgeEstimate:
    """Run the tabular learner on the default corridor edge."""
    env = Rooms9()
    learner = TabularLearner()
    edge = _edge(REGIONS["start"], REGIONS["goal"])
    policy = learner.train_edge(env, edge, 0)
    return learner.estimate_reach(env, edge, policy, 5, seed=seed)


def _make_tabular_learner() -> TabularLearner:
    """Build the fast planner learner."""
    return TabularLearner()


def _make_ppo_learner() -> PPOEdgeLearner:
    """Build a PPO learner with a small recipe for a fast contract check."""
    pytest.importorskip("stable_baselines3")
    return PPOEdgeLearner(recipe={"n_steps": 64, "n_epochs": 1, "batch_size": 32}, budget=128)


def test_rollout_success_needs_target_and_safety():
    """A rollout succeeds only when it reaches the target and stays safe."""
    assert Rollout((np.zeros(4),), reached=True, safe=False).success is False
    assert Rollout((np.zeros(4),), reached=False, safe=True).success is False
    assert Rollout((np.zeros(4),), reached=True, safe=True).success is True


def test_require_successful_raises_when_no_rollout_succeeded():
    """An estimate with no success refuses to refine."""
    estimate = EdgeEstimate(0.0, (Rollout((np.zeros(4),), reached=False, safe=True),))
    with pytest.raises(CannotRefineError):
        estimate.require_successful()


def test_require_successful_returns_only_successful_rollouts():
    """An estimate returns just the successful rollouts."""
    good = Rollout((np.zeros(4),), reached=True, safe=True)
    bad = Rollout((np.zeros(4),), reached=False, safe=True)
    estimate = EdgeEstimate(0.5, (good, bad))
    assert estimate.require_successful() == (good,)


def test_tabular_learner_succeeds_on_the_default_corridor():
    """The planner routes through the doors and succeeds every episode."""
    assert _tabular_estimate(seed=0).success_probability == 1.0


def test_tabular_learner_fails_when_the_goal_is_unreachable():
    """With every wall solid, the far goal is unreachable, so success is zero."""
    env = Rooms9(layout=SOLID_LAYOUT)
    learner = TabularLearner()
    edge = _edge(REGIONS["start"], REGIONS["goal"])
    policy = learner.train_edge(env, edge, 0)
    assert learner.estimate_reach(env, edge, policy, 5, seed=0).success_probability == 0.0


def test_tabular_learner_is_deterministic():
    """The same seed gives the same estimate and the same first state."""
    first = _tabular_estimate(seed=3)
    second = _tabular_estimate(seed=3)
    assert first.success_probability == second.success_probability
    assert np.allclose(first.rollouts[0].states[0], second.rollouts[0].states[0])


def test_ppo_default_recipe_matches_the_paper_recipe():
    """The default PPO settings match paper section 6.1."""
    assert DEFAULT_RECIPE["learning_rate"] == 0.0003
    assert DEFAULT_RECIPE["n_steps"] == 2048
    assert DEFAULT_RECIPE["batch_size"] == 64
    assert DEFAULT_RECIPE["n_epochs"] == 10
    assert DEFAULT_RECIPE["gamma"] == 0.99
    assert DEFAULT_RECIPE["policy_kwargs"]["net_arch"] == [64, 64]


@pytest.mark.parametrize(
    "make_learner",
    [pytest.param(_make_tabular_learner, id="tabular"), pytest.param(_make_ppo_learner, id="ppo")],
)
def test_learner_contract_returns_well_formed_estimates(make_learner):
    """Both learners return the same well-formed estimate shape.

    The state vectors are 4-D floats and every rollout has at least one state.
    A caller can therefore swap learners without changing its own code.
    """
    env = Rooms9()
    learner = make_learner()
    edge = _edge(REGIONS["start"], REGIONS["goal"])
    policy = learner.train_edge(env, edge, 128)
    estimate = learner.estimate_reach(env, edge, policy, CONTRACT_ROLLOUTS, seed=0)
    assert 0.0 <= estimate.success_probability <= 1.0
    assert len(estimate.rollouts) == CONTRACT_ROLLOUTS
    for rollout in estimate.rollouts:
        assert len(rollout.states) >= 1
        assert rollout.states[0].shape == (4,)
        assert rollout.states[0].dtype == np.dtype(float)
