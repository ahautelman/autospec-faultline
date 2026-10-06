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

from collections.abc import Callable
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

# The rollout fast path copies the body of Stable-Baselines3's ``BasePolicy.predict``
# but hoists its per-step ``set_training_mode(False)`` out of the loop, which the
# profile measured at ~14.4us per step, about a quarter of one evaluation. Because it
# reads two SB3 internals (``policy.obs_to_tensor`` and ``policy._predict``) this is
# guarded by a major-version check and pinned to the public ``predict`` result by a
# bit-for-bit test; bump the prefix only after re-checking that test against the new
# release.
_SUPPORTED_SB3_MAJOR = "2."


def check_sb3_supported() -> None:
    """Fail loudly when Stable-Baselines3 is missing or outside the supported major.

    How:
        Imports Stable-Baselines3 and compares its major-version prefix against
        :data:`_SUPPORTED_SB3_MAJOR`. The fast rollout path mirrors the internals of
        ``BasePolicy.predict``, so an unverified release could silently change every
        evaluation number.

    Returns:
        None: the installed version is supported.

    Raises:
        ImportError: If Stable-Baselines3 is not installed.
        RuntimeError: If its version is outside the supported major.
    """
    try:
        import stable_baselines3
    except ImportError as err:
        raise ImportError("The rollout fast path needs Stable-Baselines3 installed") from err
    if not stable_baselines3.__version__.startswith(_SUPPORTED_SB3_MAJOR):
        raise RuntimeError(
            "The rollout fast path mirrors Stable-Baselines3 "
            f"{_SUPPORTED_SB3_MAJOR}x predict internals; installed "
            f"{stable_baselines3.__version__} is unverified"
        )


def greedy_action_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return a deterministic action function equivalent to ``model.predict``.

    Args:
        model: A trained Stable-Baselines3 model, or its policy object directly.

    How:
        Resolves the inner policy, switches it to evaluation mode once, then
        returns a closure that repeats the ``obs_to_tensor`` -> ``_predict`` ->
        post-process sequence ``BasePolicy.predict`` runs, minus the per-step
        ``set_training_mode``. The result is deliberately identical to
        ``model.predict(observation, deterministic=True)[0]``.

    Returns:
        A function mapping one observation to its greedy action array.
    """
    check_sb3_supported()
    import torch as th
    from gymnasium import spaces

    # A BaseAlgorithm wraps its BasePolicy; accept either so callers can pass the
    # handle train_edge returns without reaching inside it.
    policy = getattr(model, "policy", model)
    policy.set_training_mode(False)

    def action(observation: np.ndarray) -> np.ndarray:
        obs_tensor, vectorized_env = policy.obs_to_tensor(observation)
        with th.no_grad():
            actions = policy._predict(obs_tensor, deterministic=True)
        actions = actions.cpu().numpy().reshape((-1, *policy.action_space.shape))
        if isinstance(policy.action_space, spaces.Box):
            if policy.squash_output:
                actions = policy.unscale_action(actions)
            else:
                actions = np.clip(actions, policy.action_space.low, policy.action_space.high)
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions

    return action


class PPOEdgeLearner:
    """A PPO learner that trains one policy per edge.

    Args:
        device: Torch device string for training and inference.
        budget: Default training budget in environment steps, used when the
            caller does not pass one. The paper plots several budgets, so the
            loop calls :meth:`train_edge` with an explicit value.
        recipe: Optional overrides for the Stable-Baselines3 PPO settings,
            merged over :data:`DEFAULT_RECIPE`.
        seed: Optional RNG seed for training, so a run is reproducible; ``None``
            (the default) leaves Stable-Baselines3 unseeded.

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
        seed: int | None = None,
    ) -> None:
        self.device = device
        self.budget = int(budget)
        self.recipe = {**DEFAULT_RECIPE, **(recipe or {})}
        self.seed = seed

    def train_edge(
        self,
        env: Any,
        edge: Edge,
        budget: int | None = None,
        callback: Any | None = None,
        seed: int | None = None,
    ) -> Any:
        """Train a PPO policy for one edge.

        Args:
            env: The environment, which implements ``set_abstract_states`` and
                the gymnasium interface.
            edge: The reach-avoid edge to learn.
            budget: Training budget in environment steps; defaults to the
                learner's budget.
            callback: Optional Stable-Baselines3 callback, for example a periodic
                checkpoint that evaluates the policy mid-training; ``None`` trains
                without one.
            seed: RNG seed for this training call; overrides the learner's own
                seed. ``None`` falls back to the learner's seed, which itself
                defaults to ``None`` (unseeded). Threading the run seed through
                here is what makes a stochastic learner reproducible.

        How:
            Sets the edge as the environment's subgoal, builds a PPO model with
            the section 6.1 recipe seeded from this call, and trains it for the
            budget, threading any callback through to ``model.learn``.

        Returns:
            The trained Stable-Baselines3 PPO model.

        Raises:
            ImportError: If Stable-Baselines3 is not installed.
        """
        from stable_baselines3 import PPO

        env.set_abstract_states(edge.source.region, edge.target.region, edge.safe)
        model = PPO(
            "MlpPolicy",
            env,
            device=self.device,
            verbose=0,
            seed=seed if seed is not None else self.seed,
            **self.recipe,
        )
        model.learn(total_timesteps=budget if budget is not None else self.budget, callback=callback)
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
            Sets the subgoal, builds the fast greedy-action function once (so the
            per-step training-mode switch is paid a single time, not per rollout
            step), then runs each episode from a freshly seeded reset with
            deterministic actions. The safe flag tracks whether every visited state
            stayed inside the safe region.

        Returns:
            An :class:`EdgeEstimate` with the success probability and rollouts.

        Raises:
            ValueError: If ``n_rollouts`` is not positive.
        """
        if n_rollouts < 1:
            raise ValueError(f"n_rollouts must be positive, got {n_rollouts}")
        env.set_abstract_states(edge.source.region, edge.target.region, edge.safe)
        action = greedy_action_fn(policy)
        rollouts = []
        for episode in range(n_rollouts):
            observation, _info = env.reset(seed=seed + episode)
            rollouts.append(self._episode(env, edge, action, observation))
        success = sum(1 for rollout in rollouts if rollout.success) / n_rollouts
        return EdgeEstimate(success, tuple(rollouts))

    def _episode(
        self,
        env: Any,
        edge: Edge,
        action: Callable[[np.ndarray], np.ndarray],
        observation: np.ndarray,
    ) -> Rollout:
        """Run one greedy episode and record it.

        Args:
            env: The environment for the episode.
            edge: The reach-avoid edge under test.
            action: Deterministic action function, normally from
                :func:`greedy_action_fn`.
            observation: The first observation from ``reset``.

        How:
            Steps with the deterministic action function until the episode ends.
            Records every state and tracks the safe region and the success flag.

        Returns:
            The rollout for this episode.
        """
        states = [np.asarray(observation, dtype=float)]
        safe = bool(edge.safe.evaluate(states[0][:2]))
        reached = False
        while True:
            step_action = action(observation)
            observation, _reward, terminated, truncated, info = env.step(step_action)
            states.append(np.asarray(observation, dtype=float))
            # The edge's safe region is the specification's; the environment may
            # also carry an unmodelled trap (info["is_safe"]), which must count too.
            safe = (
                safe
                and bool(edge.safe.evaluate(states[-1][:2]))
                and bool(info.get("is_safe", True))
            )
            reached = bool(info["is_success"])
            if terminated or truncated:
                break
        return Rollout(tuple(states), reached=reached, safe=safe)


    def continue_edge(
            self,
            env: Any,
            edge: Edge,
            policy: Any,
            budget: int,
            seed: int | None = None,
    ) -> Any:
        """Continue training an existing PPO edge policy.

        LSTS repeatedly assigns small chunks of training to different active
        sub-tasks. This method preserves the PPO parameters between chunks rather
        than constructing a fresh policy.
        """

        print(
            "continuing",
            edge.source.name,
            "->",
            edge.target.name,
            "timesteps:",
            policy.num_timesteps,
        )

        env.set_abstract_states(
            edge.source.region,
            edge.target.region,
            edge.safe,
        )

        policy.set_env(env)

        policy.learn(
            total_timesteps=int(budget),
            reset_num_timesteps=False,
        )

        return policy
