"""Placeholder contract for learned-policy (stochastic) layout verification.

The heuristic verifier in :mod:`faultline.envs.rooms.verify` is exact about
existence. This module holds the second, optional stage: a verifier that trains
a learner on the unrefined graph and reports its success. The protocol and
report are defined now so callers can be written before the DiRL baseline
exists. The placeholder raises rather than returning a passing score, so it
cannot be mistaken for a working baseline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from faultline.envs.rooms.world import RoomsWorld

DIRL_NOT_IMPLEMENTED_MESSAGE = (
    "DiRL baseline is not implemented yet; see plans/rooms100-plan.md §4.3"
)


@dataclass(frozen=True)
class StochasticReport:
    """Placeholder contract for a learned-policy verification result.

    Args:
        learner: Identifier of the learner that produced the estimates.
        seed: Seed of the verification run.
        budget: Training steps per edge.
        threshold: Success threshold the caller requires of the best path.
        per_edge: Estimated success probability per ``"A->B"`` edge.
        best_path: Product of per-edge successes along the best path.

    How:
        Frozen and JSON-serialisable. :attr:`passed` applies the caller's
        threshold to the best-path product, so no caller has to know the rule.
    """

    learner: str
    seed: int
    budget: int
    threshold: float
    per_edge: Mapping[str, float]
    best_path: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_edge", MappingProxyType(dict(self.per_edge)))

    @property
    def passed(self) -> bool:
        """Return whether the best path clears the caller's threshold.

        How:
            Compares the best-path product with the caller's threshold.

        Returns:
            ``True`` when the best path clears the threshold.
        """
        return self.best_path >= self.threshold

    def to_dict(self) -> dict:
        """Convert the report to a JSON-serialisable mapping.

        How:
            Copies every field and adds the derived verdict.

        Returns:
            The report as a JSON-serialisable mapping.
        """
        return {
            "learner": self.learner,
            "seed": self.seed,
            "budget": self.budget,
            "threshold": self.threshold,
            "per_edge": dict(self.per_edge),
            "best_path": self.best_path,
            "passed": self.passed,
        }


class StochasticVerifier(Protocol):
    """Protocol for a learned-policy verifier.

    How:
        A verifier receives a world and a seed and returns a
        :class:`StochasticReport`. The generator accepts one optionally, so a
        world can be required to pass both the heuristic and the learned stage.
    """

    def verify(self, world: RoomsWorld, *, seed: int) -> StochasticReport:
        """Verify a world with a learned policy.

        Args:
            world: The world to verify.
            seed: Seed of the verification run.

        How:
            Trains a learner on the world's unrefined graph and scores the best
            path.
        """
        ...


class DiRLVerifier:
    """Placeholder for the plain-DiRL baseline.

    How:
        Deliberately not implemented. It raises rather than returning a passing
        report, so wiring it in cannot silently turn verification into a no-op.
    """

    def verify(self, world: RoomsWorld, *, seed: int) -> StochasticReport:
        """Raise until the DiRL baseline exists.

        Args:
            world: The world to verify.
            seed: Seed of the verification run.

        How:
            Always raises, so wiring the placeholder in cannot turn verification
            into a no-op.

        Raises:
            NotImplementedError: Always, with the documented message.
        """
        raise NotImplementedError(DIRL_NOT_IMPLEMENTED_MESSAGE)


__all__ = [
    "DIRL_NOT_IMPLEMENTED_MESSAGE",
    "DiRLVerifier",
    "StochasticReport",
    "StochasticVerifier",
]
