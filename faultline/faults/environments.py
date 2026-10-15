"""Benchmark-specific environment variants used by FaultLine fault injection.

These classes intentionally live under ``faultline.faults`` rather than the
core environment package.  They are only needed to construct controlled
benchmark faults and should not affect the normal Rooms9 experiments.
"""

from __future__ import annotations

import numpy as np

from faultline.envs import Rooms9
from faultline.spectrl.predicates import Empty, Predicate


class HiddenHazardRooms9(Rooms9):
    """Rooms9 variant with an unmodelled hazard used by AvoidRefine tests.

    The important distinction is between the *specification* safety predicate
    and the *hidden* hazard:

    * ``safe_region`` is the current edge predicate supplied by AutoSpec.
    * ``hidden_unsafe_region`` is a physical/system hazard that the faulty
      specification initially fails to exclude.

    When the agent enters the hidden hazard while the current specification
    still considers that state safe, the episode terminates and the rollout is
    reported as unsafe through ``info["is_safe"]``.  However, it receives only a
    mild benchmark reward rather than the normal large specification-violation
    penalty.  This prevents PPO from trivially learning to freeze before the
    hazard and ensures that AutoSpec receives unsafe trajectory tails.

    Once AvoidRefine tightens ``safe_region`` around the observed hazard, the
    same state also violates the specification.  It then receives the normal
    Rooms9 violation penalty, so PPO is encouraged to learn the safe detour.

    This class is benchmark-only.  It does not change the semantics of the
    ordinary ``Rooms9`` environment used elsewhere in the project.
    """

    def __init__(
        self,
        *args,
        hidden_unsafe_region: Predicate | None = None,
        hidden_hazard_reward: float = 0.0,
        **kwargs,
    ) -> None:
        # Do not pass the hidden hazard to Rooms9.  The base class would treat it
        # as a fully modelled unsafe region and immediately expose the large
        # violation penalty to PPO, which defeats the purpose of this benchmark.
        kwargs.pop("unsafe_region", None)
        super().__init__(*args, unsafe_region=Empty(), **kwargs)
        self.hidden_unsafe_region: Predicate = (
            hidden_unsafe_region if hidden_unsafe_region is not None else Empty()
        )
        self.hidden_hazard_reward = float(hidden_hazard_reward)

    def step(self, action: np.ndarray):
        """Apply one action while keeping hidden-hazard and spec safety distinct."""

        previous_distance = float(np.linalg.norm(self._goal - self._position))
        self._position = self._advance(action)
        self._steps += 1
        distance = float(np.linalg.norm(self._goal - self._position))

        spec_safe = bool(self.safe_region.evaluate(self._position))
        in_hidden_hazard = bool(self.hidden_unsafe_region.evaluate(self._position))

        # A state excluded by the current specification is a normal Rooms9
        # violation and receives the standard strong penalty through _reward().
        if not spec_safe:
            self._violated = True
            success = False
            reward = self._reward(
                success=False,
                safe=False,
                previous_distance=previous_distance,
                distance=distance,
            )
            terminated = True

        # A hidden hazard is an environmental failure that the faulty logical
        # safety predicate has not yet captured.  Terminating here ensures the
        # final k states used by AvoidRefine are genuinely local to the hazard.
        elif in_hidden_hazard:
            success = False
            reward = self.hidden_hazard_reward
            terminated = True

        else:
            reached = bool(distance <= 1.0)
            success = reached and not self._violated
            reward = self._reward(
                success=success,
                safe=True,
                previous_distance=previous_distance,
                distance=distance,
            )
            terminated = bool(success)

        truncated = bool(self._steps >= self.max_steps and not terminated)

        info = {
            "is_success": bool(success),
            # PPOEdgeLearner records this flag in Rollout.safe.  Therefore a
            # hidden-hazard episode is available to AvoidRefine even though the
            # coarse edge.safe predicate initially says the state is allowed.
            "is_safe": bool(spec_safe and not in_hidden_hazard),
            "hidden_hazard": bool(in_hidden_hazard),
        }

        return self._observation(), reward, terminated, truncated, info
