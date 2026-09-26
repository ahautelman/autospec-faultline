"""The learner interface shared by every spec-guided RL backend.

AutoSpec trains one policy per graph edge. The refinement operators never touch
that policy directly. They consume an :class:`EdgeEstimate`, which holds the
success probability and the sampled rollouts. A learner only has to produce that
estimate. This keeps the operators independent of the backend, so a fast test
learner and a real PPO learner are interchangeable.

Rollouts are plain arrays. The learner reads state vectors from the environment
and never assumes a fixed state dimension, so a change of environment or learner
does not change the operator code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from faultline.graph.abstract_graph import Edge


class CannotRefineError(Exception):
    """Raised when an edge has no successful trajectory to refine from.

    The paper treats this as a hard stop, not a bug. A refinement operator
    raises it when its input rollouts contain no success, because a convex hull
    or a separating hyperplane needs at least one successful sample.
    """


@dataclass(frozen=True)
class Rollout:
    """One episode for an edge subgoal.

    Args:
        states: State vectors in time order.
        reached: Whether the episode reached the target region.
        safe: Whether the episode stayed inside the safe region throughout.

    How:
        Stores the trajectory and the two reach-avoid outcomes. The success
        property combines them.
    """

    states: tuple[np.ndarray, ...]
    reached: bool
    safe: bool

    @property
    def success(self) -> bool:
        """Return whether the episode satisfied the reach-avoid condition.

        How:
            Success needs both the target reached and the safe region respected
            for the whole episode.

        Returns:
            ``True`` when the episode succeeded.
        """
        return self.reached and self.safe


@dataclass(frozen=True)
class EdgeEstimate:
    """The outcome of training and evaluating one edge policy.

    Args:
        success_probability: Fraction of rollouts that succeeded.
        rollouts: The sampled rollouts, in episode order.

    How:
        Stores the estimate and the raw rollouts. The operators read the
        rollouts to build convex hulls and separators.
    """

    success_probability: float
    rollouts: tuple[Rollout, ...]

    def successful(self) -> tuple[Rollout, ...]:
        """Return only the successful rollouts.

        How:
            Keeps rollouts whose reach-avoid condition held.

        Returns:
            The successful rollouts, in episode order.
        """
        return tuple(rollout for rollout in self.rollouts if rollout.success)

    def require_successful(self) -> tuple[Rollout, ...]:
        """Return the successful rollouts, or refuse to refine.

        How:
            Calls :meth:`successful` and raises when the result is empty, which
            is the paper's sample-starved case.

        Returns:
            The successful rollouts.

        Raises:
            CannotRefineError: If no rollout succeeded.
        """
        successes = self.successful()
        if not successes:
            raise CannotRefineError("Edge has no successful trajectory to refine from")
        return successes


@runtime_checkable
class Learner(Protocol):
    """A spec-guided RL backend that trains and evaluates one edge policy."""

    def train_edge(self, env: Any, edge: Edge, budget: int, seed: int | None = None) -> Any:
        """Train a policy for one edge and return a handle for it.

        Args:
            env: The environment, which implements ``set_abstract_states`` and
                the gymnasium ``reset`` and ``step`` methods.
            edge: The reach-avoid edge to learn.
            budget: Training budget, for example environment steps.
            seed: Base RNG seed for training, so a run is reproducible; ``None``
                (the default) leaves the implementation's own default in place.

        How:
            Implementations set the edge as the environment's subgoal, fit their
            policy, and return an opaque handle. The caller passes that handle
            back to :meth:`estimate_reach` without inspecting it. An
            implementation whose training is stochastic must derive that
            randomness from ``seed`` when it is provided, so two calls with the
            same seed and inputs yield the same policy.

        Returns:
            A backend-specific policy handle, passed back to ``estimate_reach``.
        """
        ...

    def estimate_reach(
        self, env: Any, edge: Edge, policy: Any, n_rollouts: int, seed: int
    ) -> EdgeEstimate:
        """Roll out an edge policy and estimate its success probability.

        Args:
            env: The environment, as in ``train_edge``.
            edge: The reach-avoid edge under test.
            policy: The handle returned by ``train_edge``.
            n_rollouts: Number of episodes to sample.
            seed: Seed for the episode RNG, so the estimate is reproducible.

        How:
            Implementations set the edge as the subgoal, run ``n_rollouts``
            episodes from seeded resets, and return the estimate together with
            the raw rollouts that the refinement operators consume.

        Returns:
            An :class:`EdgeEstimate` with the success probability and rollouts.
        """
        ...
