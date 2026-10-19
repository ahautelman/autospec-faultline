"""Environment wrapper that resets from an empirical set of start states."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class StartDistributionWrapper(gym.Wrapper):
    """Reset an environment from an empirical set of start states.

    Args:
        env: The wrapped environment; its ``reset`` must accept
            ``options["start"]``, as :class:`~faultline.envs.rooms.env.RoomsEnv`
            does.
        start_states: Positions to sample a start from, or ``None`` to defer to
            the environment's own start region.
        seed: Seed for the start-state sampler.

    How:
        DiRL uses this to approximate the induced distribution at an abstract
        vertex. The states are the 2-D positions where the preceding edge's
        successful rollouts ended. Each reset draws one uniformly and passes it
        to the environment as ``options["start"]``. Gymnasium 1.x wrappers no
        longer forward unknown attributes, so public attributes and methods of
        the wrapped environment, such as ``set_abstract_states``, are forwarded
        explicitly.
    """

    def __init__(
        self,
        env: Any,
        start_states: tuple[np.ndarray, ...] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(env)
        self.start_states = start_states
        self._rng = np.random.default_rng(seed)

    def __getattr__(self, name: str) -> Any:
        """Forward public lookups that the wrapper does not define to the env."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def set_start_states(self, states: tuple[np.ndarray, ...] | None) -> None:
        """Replace the empirical start distribution; ``None`` clears it."""
        self.start_states = states

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment, from a sampled start state when one is set."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if not self.start_states:
            return self.env.reset(seed=seed, options=options)

        index = int(self._rng.integers(len(self.start_states)))
        position = np.asarray(self.start_states[index], dtype=float)
        merged = {} if options is None else dict(options)
        merged["start"] = position
        return self.env.reset(seed=seed, options=merged)
