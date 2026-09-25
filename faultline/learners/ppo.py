"""A Stable-Baselines3 PPO learner with one policy per edge.

This is the real learner from the paper. It trains one small PPO policy for each
reach-avoid edge, using the recipe in section 6.1: a two-layer 64-unit network,
learning rate 3e-4, 2048 rollout steps, batch size 64, 10 epochs per update, and
discount 0.99. The paper evaluates each edge with 1000 rollouts.

Stable-Baselines3 lives in the optional ``interop`` extra. This module imports
it inside :meth:`PPOEdgeLearner.train_edge`, so importing the module never
requires the extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from faultline.graph.abstract_graph import Edge
from faultline.learners.base import EdgeEstimate, Rollout

DEVICE = "cpu"
DEFAULT_BUDGET = 80_000
DEFAULT_ROLLOUTS = 1000

# The paper's recipe from section 6.1: a two-layer 64-unit network, learning
# rate 3e-4, 2048 rollout steps, batch size 64, 10 epochs per update, and
# discount 0.99. A caller may override entries; the tests override them to keep
# a single contract check fast.
DEFAULT_RECIPE: dict[str, Any] = {
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "policy_kwargs": {"net_arch": [64, 64]},
}


class PPOEdgeLearner:
    """A PPO learner that trains one policy per edge.

    Args:
        device: Torch device string for training and inference.
        budget: Default training budget in environment steps, used when the
            caller does not pass one. The paper plots several budgets, so the
            loop calls :meth:`train_edge` with an explicit value.
        recipe: Optional overrides for the Stable-Baselines3 PPO settings,
            merged over :data:`DEFAULT_RECIPE`.

    How:
        Implements the :class:`~faultline.learners.base.Learner` protocol. The
        environment is shared between calls; each call sets its abstract states
        to the edge under test before training or rollouts.
    """

    def __init__(
        self,
        device: str = DEVICE,
        budget: int = DEFAULT_BUDGET,
        recipe: dict[str, Any] | None = None,
    ) -> None:
        self.device = device
        self.budget = int(budget)
        self.recipe = {**DEFAULT_RECIPE, **(recipe or {})}

    def train_edge(self, env: Any, edge: Edge, budget: int | None = None) -> Any:
        """Train a PPO policy for one edge.

        Args:
            env: The environment, which implements ``set_abstract_states`` and
                the gymnasium interface.
            edge: The reach-avoid edge to learn.
            budget: Training budget in environment steps; defaults to the
                learner's budget.

        How:
            Sets the edge as the environment's subgoal, builds a PPO model with
            the section 6.1 recipe, and trains it for the budget.

        Returns:
            The trained Stable-Baselines3 PPO model.

        Raises:
            ImportError: If Stable-Baselines3 is not installed.
        """
        from stable_baselines3 import PPO

        env.set_abstract_states(edge.source.region, edge.target.region, edge.safe)
        model = PPO("MlpPolicy", env, device=self.device, verbose=0, **self.recipe)
        model.learn(total_timesteps=budget if budget is not None else self.budget)
        return model

    def estimate_reach(
        self, env: Any, edge: Edge, policy: Any, n_rollouts: int = DEFAULT_ROLLOUTS, seed: int = 0
    ) -> EdgeEstimate:
        """Roll out a trained policy and estimate its success probability.

        Args:
            env: The environment, as in :meth:`train_edge`.
            edge: The reach-avoid edge under test.
            policy: The trained PPO model from :meth:`train_edge`.
            n_rollouts: Number of episodes to sample; must be positive.
            seed: Base seed for the episode RNG.

        How:
            Sets the subgoal, then runs each episode from a freshly seeded reset
            with deterministic actions. The safe flag tracks whether every
            visited state stayed inside the safe region.

        Returns:
            An :class:`EdgeEstimate` with the success probability and rollouts.

        Raises:
            ValueError: If ``n_rollouts`` is not positive.
        """
        if n_rollouts < 1:
            raise ValueError(f"n_rollouts must be positive, got {n_rollouts}")
        env.set_abstract_states(edge.source.region, edge.target.region, edge.safe)
        rollouts = []
        for episode in range(n_rollouts):
            observation, _info = env.reset(seed=seed + episode)
            rollouts.append(self._episode(env, edge, policy, observation))
        success = sum(1 for rollout in rollouts if rollout.success) / n_rollouts
        return EdgeEstimate(success, tuple(rollouts))

    def _episode(
        self, env: Any, edge: Edge, policy: Any, observation: np.ndarray
    ) -> Rollout:
        """Run one greedy episode and record it.

        Args:
            env: The environment for the episode.
            edge: The reach-avoid edge under test.
            policy: The trained PPO model.
            observation: The first observation from ``reset``.

        How:
            Steps with deterministic policy actions until the episode ends.
            Records every state and tracks the safe region and the success flag.

        Returns:
            The rollout for this episode.
        """
        states = [np.asarray(observation, dtype=float)]
        safe = bool(edge.safe.evaluate(states[0][:2]))
        reached = False
        while True:
            action, _state = policy.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, info = env.step(action)
            states.append(np.asarray(observation, dtype=float))
            safe = safe and bool(edge.safe.evaluate(states[-1][:2]))
            reached = bool(info["is_success"])
            if terminated or truncated:
                break
        return Rollout(tuple(states), reached=reached, safe=safe)
