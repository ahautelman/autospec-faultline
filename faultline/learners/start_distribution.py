from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class StartDistributionWrapper(gym.Wrapper):
    """Reset an environment from an empirical set of start states.

    DiRL uses this to approximate the induced distribution at an abstract
    vertex. States are 2-D positions reached successfully by the preceding
    edge policy.
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

    def set_start_states(
        self,
        states: tuple[np.ndarray, ...] | None,
    ) -> None:
        self.start_states = states

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if not self.start_states:
            return self.env.reset(seed=seed, options=options)

        index = int(self._rng.integers(len(self.start_states)))
        position = np.asarray(self.start_states[index], dtype=float)

        opts = {} if options is None else dict(options)
        opts["start_position"] = position

        return self.env.reset(seed=seed, options=opts)