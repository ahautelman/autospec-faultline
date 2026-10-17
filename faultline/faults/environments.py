"""Benchmark-specific environment variants used by FaultLine fault injection.

These classes intentionally live under ``faultline.faults`` rather than the
core environment package. They are only needed to construct controlled
benchmark faults and should not affect the normal Rooms9 experiments.
"""

from __future__ import annotations

import numpy as np

from faultline.envs import Rooms9
from faultline.spectrl.predicates import Empty, Predicate


class HiddenHazardRooms9(Rooms9):
    """Rooms9 variant for the over-wide-safety FaultLine benchmark.

    The benchmark needs to separate two notions of safety:

    * ``safe_region`` is the safety predicate currently supplied to the RL
      learner by the (possibly faulty) specification.
    * ``hidden_unsafe_region`` is the ground-truth hazard that the faulty
      specification incorrectly permits.

    During PPO *training*, a hidden hazard that is still allowed by
    ``safe_region`` is deliberately not terminal and is not given an extra
    penalty. Training therefore follows the faulty specification rather than
    learning to avoid the benchmark's hidden ground truth by itself.

    The environment still exposes ``info["is_safe"] = False`` whenever the
    hidden hazard is entered. ``PPOEdgeLearner`` uses that flag during
    evaluation and stops the evaluation rollout at the first unsafe state. This
    gives AvoidRefine trajectory tails local to the hazard without leaking the
    hazard into PPO's training reward.

    Once AvoidRefine tightens ``safe_region`` so that the same location is
    excluded by the specification, entering it becomes an ordinary Rooms9
    safety violation: the normal violation penalty is applied and the episode
    terminates. The newly trained policy can then learn the safe detour.

    This class is benchmark-only. The ordinary ``Rooms9`` semantics used by the
    rest of the project are unchanged.
    """

    def __init__(
        self,
        *args,
        hidden_unsafe_region: Predicate | None = None,
        **kwargs,
    ) -> None:
        # Do not pass the hidden hazard to Rooms9. If it were the base
        # ``unsafe_region``, PPO would receive the normal strong violation signal
        # during training and could solve the benchmark without AutoSpec.
        kwargs.pop("unsafe_region", None)
        super().__init__(*args, unsafe_region=Empty(), **kwargs)

        self.hidden_unsafe_region: Predicate = (
            hidden_unsafe_region
            if hidden_unsafe_region is not None
            else Empty()
        )

    def step(self, action: np.ndarray):
        """Apply one action while keeping hidden and specification safety separate.

        If the current specification already excludes the new state, normal
        Rooms9 safety semantics apply: strong violation penalty and termination.

        If the state is only inside the hidden ground-truth hazard, the step is
        *not* terminal during environment training and receives the same progress
        reward as an ordinary safe step. The returned info nevertheless marks the
        state unsafe. The edge evaluator consumes that flag and ends the recorded
        rollout there, which is exactly what AvoidRefine needs.
        """

        previous_distance = float(
            np.linalg.norm(self._goal - self._position)
        )

        self._position = self._advance(action)
        self._steps += 1

        distance = float(
            np.linalg.norm(self._goal - self._position)
        )

        spec_safe = bool(
            self.safe_region.evaluate(self._position)
        )

        in_hidden_hazard = bool(
            self.hidden_unsafe_region.evaluate(self._position)
        )

        # The current specification itself says this state is unsafe. From this
        # point on we use the ordinary Rooms9 violation semantics.
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

        else:
            # A hidden hazard is deliberately invisible to PPO's reward and
            # termination signal while the faulty specification still permits it.
            # This keeps the training objective faithful to the faulty spec.
            reached = bool(distance <= 1.0)
            success = reached and not self._violated and not in_hidden_hazard

            reward = self._reward(
                success=success,
                safe=True,
                previous_distance=previous_distance,
                distance=distance,
            )

            # Reaching the true target remains terminal. Entering only the hidden
            # hazard is *not* terminal here; evaluation terminates its recorded
            # rollout when it sees info["is_safe"] == False.
            terminated = bool(success)

        truncated = bool(
            self._steps >= self.max_steps
            and not terminated
        )

        info = {
            "is_success": bool(success),
            "is_safe": bool(
                spec_safe
                and not in_hidden_hazard
            ),
            "hidden_hazard": bool(
                in_hidden_hazard
            ),
        }

        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            info,
        )
