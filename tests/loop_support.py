"""Shared helpers for the Algorithm 1 loop tests.

The loop tests must control which operators trigger and which candidates clear
the threshold without training networks. The scripted learner here returns canned
success probabilities together with real rollouts, so the refinement operators
still run their real geometry, and it records every training call for the relearn
assertions.
"""

from __future__ import annotations

from dataclasses import dataclass

from faultline.learners import EdgeEstimate
from tests.refinement_support import rollout


@dataclass(frozen=True)
class ScriptedPolicy:
    """The handle the scripted learner returns for one edge.

    Args:
        source: Source vertex name of the trained edge.
        target: Target vertex name of the trained edge.

    How:
        Names the edge so a test can tell which handle it received; the learner
        itself never inspects the value.
    """

    source: str
    target: str


class ScriptedLearner:
    """A learner whose estimates are set by the test.

    Args:
        probability: Maps an edge to the success probability to report.
        rollouts: Maps an edge to the rollouts to report.

    How:
        Implements the learner protocol without training. ``train_edge`` records
        the call and returns a naming handle; ``estimate_reach`` returns the
        canned estimate. Tests use it to make a chosen operator clear the
        threshold deterministically and to count relearn calls.
    """

    def __init__(self, probability, rollouts):
        self._probability = probability
        self._rollouts = rollouts
        self.train_calls: list[tuple[str, str]] = []

    def train_edge(self, env, edge, budget=0) -> ScriptedPolicy:
        """Record the edge and return a naming handle."""
        self.train_calls.append((edge.source.name, edge.target.name))
        return ScriptedPolicy(edge.source.name, edge.target.name)

    def estimate_reach(self, env, edge, policy, n_rollouts, seed=0) -> EdgeEstimate:
        """Return the canned estimate for the edge."""
        return EdgeEstimate(self._probability(edge), tuple(self._rollouts(edge)))

    def count(self, source: str, target: str) -> int:
        """Return how many times an edge with these endpoints was trained."""
        return self.train_calls.count((source, target))


def refinable_rollouts(edge):
    """Return one successful and one unsafe rollout for an edge.

    Args:
        edge: The edge being estimated; unused.

    How:
        Provides a target-region visit for ReachRefine to shrink and an unsafe
        tail for AvoidRefine to block, so SeqRefine, AddRefine, and PastRefine can
        all fire.

    Returns:
        The canned rollouts.
    """
    return [
        rollout([[0.2, 0.2], [1.5, 1.5]], reached=True, safe=True),
        rollout([[0.8, 0.8], [2.6, 2.6]], reached=False, safe=False),
    ]


def blocked_rollouts(edge):
    """Return one failed, safe rollout for an edge.

    Args:
        edge: The edge being estimated; unused.

    How:
        Supplies no target visit and no success, so ReachRefine, AvoidRefine,
        AddRefine, and PastRefine all refuse and only the structural OrRefine can
        apply.

    Returns:
        The canned rollouts.
    """
    return [rollout([[0.5, 0.5], [0.5, 0.5]], reached=False, safe=True)]
