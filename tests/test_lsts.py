from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faultline.graph.abstract_graph import AbstractGraph
from faultline.learners.base import EdgeEstimate, Rollout
from faultline.learners.lsts import LSTS
from faultline.spectrl.predicates import Box, Universe


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _region(x: float, y: float) -> Box:
    """Create a simple 1 x 1 test region."""
    return Box(
        (x, y),
        (x + 1.0, y + 1.0),
    )


def _simple_chain() -> AbstractGraph:
    """
        start -> A -> goal
    """

    graph = AbstractGraph()

    start = graph.add_vertex(
        "start",
        _region(1, 1),
    )

    a = graph.add_vertex(
        "A",
        _region(5, 5),
    )

    goal = graph.add_vertex(
        "goal",
        _region(10, 10),
    )

    graph.add_edge(
        start,
        a,
        Universe(),
    )

    graph.add_edge(
        a,
        goal,
        Universe(),
    )

    graph.source = start
    graph.target = goal

    return graph


def _branch_graph() -> AbstractGraph:
    r"""
                  A
                /   \
            start   goal
                \   /
                  B

    More precisely:

        start -> A -> goal
        start -> B -> goal
    """

    graph = AbstractGraph()

    start = graph.add_vertex(
        "start",
        _region(1, 1),
    )

    a = graph.add_vertex(
        "A",
        _region(5, 5),
    )

    b = graph.add_vertex(
        "B",
        _region(5, 10),
    )

    goal = graph.add_vertex(
        "goal",
        _region(10, 10),
    )

    graph.add_edge(
        start,
        a,
        Universe(),
    )

    graph.add_edge(
        start,
        b,
        Universe(),
    )

    graph.add_edge(
        a,
        goal,
        Universe(),
    )

    graph.add_edge(
        b,
        goal,
        Universe(),
    )

    graph.source = start
    graph.target = goal

    return graph


# ---------------------------------------------------------------------------
# Fake learner
# ---------------------------------------------------------------------------

@dataclass
class FakePolicy:
    """Tiny stand-in for a PPO policy."""

    edge: tuple[str, str]

    # Number of separate chunks of training received.
    training_calls: int = 1

    # Total artificial environment interaction budget received.
    total_budget: int = 0


class FakeLSTSLearner:
    """Incremental learner used to test LSTS without running PPO.

    ``score_sequences`` specifies what success probability an edge returns
    after each successive evaluation.

    Example:

        {
            ("start", "A"): [0.5, 0.95, 0.95]
        }

    means that the first evaluation returns 0.5, the second 0.95 and all
    later evaluations return 0.95.
    """

    def __init__(
        self,
        score_sequences: dict[
            tuple[str, str],
            list[float],
        ],
    ) -> None:

        self.score_sequences = score_sequences

        # Calls that created a new policy.
        self.train_calls: list[
            tuple[str, str]
        ] = []

        # Calls that continued an existing policy.
        self.continue_calls: list[
            tuple[str, str]
        ] = []

        # IMPORTANT:
        # Chronological sequence containing BOTH train and continue calls.
        #
        # Do not construct chronology by doing:
        #
        #     train_calls + continue_calls
        #
        # because that destroys their actual temporal ordering.
        self.calls: list[
            tuple[str, str]
        ] = []

        self.estimate_counts: dict[
            tuple[str, str],
            int,
        ] = {}

        self.total_training_budget = 0

        self.created_policies: dict[
            tuple[str, str],
            FakePolicy,
        ] = {}

    def train_edge(
        self,
        env,
        edge,
        budget,
        seed=None,
    ) -> FakePolicy:
        """Create a fake policy for an edge."""

        key = (
            edge.source.name,
            edge.target.name,
        )

        self.train_calls.append(key)
        self.calls.append(key)

        self.total_training_budget += budget

        policy = FakePolicy(
            edge=key,
            training_calls=1,
            total_budget=budget,
        )

        self.created_policies[key] = policy

        return policy

    def continue_edge(
        self,
        env,
        edge,
        policy,
        budget,
        seed=None,
    ) -> FakePolicy:
        """Continue training the same fake policy."""

        key = (
            edge.source.name,
            edge.target.name,
        )

        self.continue_calls.append(key)
        self.calls.append(key)

        self.total_training_budget += budget

        policy.training_calls += 1
        policy.total_budget += budget

        return policy

    def estimate_reach(
        self,
        env,
        edge,
        policy,
        n_rollouts,
        seed,
    ) -> EdgeEstimate:
        """Return the next deterministic score for this edge."""

        key = (
            edge.source.name,
            edge.target.name,
        )

        count = self.estimate_counts.get(
            key,
            0,
        )

        self.estimate_counts[key] = count + 1

        sequence = self.score_sequences[key]

        # Once the supplied sequence is exhausted, keep returning the final
        # value.
        index = min(
            count,
            len(sequence) - 1,
        )

        score = sequence[index]

        # Produce ten fake trajectories so that the EdgeEstimate remains
        # realistic enough for other components that might inspect rollouts.
        successes = round(score * 10)

        rollouts = tuple(
            Rollout(
                states=(
                    np.array(
                        [
                            float(i),
                            0.0,
                            0.0,
                            0.0,
                        ],
                        dtype=float,
                    ),
                ),
                reached=(i < successes),
                safe=True,
            )
            for i in range(10)
        )

        return EdgeEstimate(
            success_probability=score,
            rollouts=rollouts,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lsts_reuses_policy_between_training_chunks():
    """The same student policy must be reused between LSTS selections."""

    graph = _simple_chain()

    learner = FakeLSTSLearner(
        {
            ("start", "A"): [
                1.0,
                1.0,
            ],
            ("A", "goal"): [
                1.0,
                1.0,
            ],
        }
    )

    lsts = LSTS(
        epsilon=0.0,
        alpha=1.0,
        threshold=0.9,
        tolerance=0.01,
        chunk_size=10,
        max_interactions=100,
    )

    lsts.learn_graph(
        graph=graph,
        env=object(),
        edge_learner=learner,
        budget=100,
        n_rollouts=10,
        seed=0,
    )

    # A new policy should only be constructed once.
    assert (
        learner.train_calls.count(
            ("start", "A")
        )
        == 1
    )

    # The second training chunk must continue that existing policy.
    assert (
        ("start", "A")
        in learner.continue_calls
    )

    policy = learner.created_policies[
        ("start", "A")
    ]

    assert policy.training_calls >= 2


def test_lsts_activates_successor_only_after_predecessor_converges():
    """A successor must remain inactive until its predecessor is stable."""

    graph = _simple_chain()

    learner = FakeLSTSLearner(
        {
            # Expected sequence:
            #
            # first:
            #   0.50 -> below threshold
            #
            # second:
            #   0.95 -> above threshold but large change
            #
            # third:
            #   0.95 -> above threshold and stable
            #
            # Only after the third evaluation may A->goal become active.
            ("start", "A"): [
                0.50,
                0.95,
                0.95,
            ],

            ("A", "goal"): [
                1.0,
                1.0,
            ],
        }
    )

    lsts = LSTS(
        epsilon=0.0,
        alpha=1.0,
        threshold=0.9,
        tolerance=0.01,
        chunk_size=10,
        max_interactions=100,
    )

    lsts.learn_graph(
        graph=graph,
        env=object(),
        edge_learner=learner,
        budget=100,
        n_rollouts=10,
        seed=0,
    )

    calls = learner.calls

    assert ("A", "goal") in calls

    first_goal_index = calls.index(
        ("A", "goal")
    )

    previous = calls[:first_goal_index]

    # start->A must have received three chunks before its successor becomes
    # available.
    assert previous.count(
        ("start", "A")
    ) == 3

    # Stronger ordering check.
    assert calls[:4] == [
        ("start", "A"),
        ("start", "A"),
        ("start", "A"),
        ("A", "goal"),
    ]


def test_lsts_respects_total_interaction_budget():
    """LSTS must never exceed its maximum interaction budget."""

    graph = _simple_chain()

    learner = FakeLSTSLearner(
        {
            # This task never reaches the required threshold.
            ("start", "A"): [
                0.2,
            ],

            ("A", "goal"): [
                1.0,
            ],
        }
    )

    lsts = LSTS(
        epsilon=0.0,
        alpha=1.0,
        threshold=0.9,
        tolerance=0.01,
        chunk_size=10,
        max_interactions=30,
    )

    lsts.learn_graph(
        graph=graph,
        env=object(),
        edge_learner=learner,
        budget=30,
        n_rollouts=10,
        seed=0,
    )

    # Exactly three chunks of size 10 should have been used.
    assert learner.total_training_budget == 30

    # Since start->A never converged, A->goal must never have become active.
    assert (
        ("A", "goal")
        not in learner.calls
    )


def test_lsts_epsilon_zero_exploits_highest_q_task():
    """After initial exploration, epsilon=0 must choose the best Q task."""

    graph = _branch_graph()

    learner = FakeLSTSLearner(
        {
            # A is clearly better than B.
            ("start", "A"): [
                0.8,
                0.8,
                0.8,
            ],

            ("start", "B"): [
                0.2,
                0.2,
                0.2,
            ],

            ("A", "goal"): [
                1.0,
                1.0,
            ],

            ("B", "goal"): [
                1.0,
                1.0,
            ],
        }
    )

    lsts = LSTS(
        epsilon=0.0,
        alpha=1.0,

        # Neither source edge should converge during this short test.
        threshold=0.99,

        tolerance=0.01,

        chunk_size=10,
        max_interactions=40,
    )

    lsts.learn_graph(
        graph=graph,
        env=object(),
        edge_learner=learner,
        budget=40,
        n_rollouts=10,
        seed=0,
    )

    calls = learner.calls

    # Every newly active task must be tried once.
    assert ("start", "A") in calls
    assert ("start", "B") in calls

    # Deterministic initial exploration:
    #
    # 1. A is tried -> Q(A) = 0.8
    # 2. B is tried -> Q(B) = 0.2
    #
    # After that epsilon=0 means pure exploitation, so A must be selected.
    assert calls[:3] == [
        ("start", "A"),
        ("start", "B"),
        ("start", "A"),
    ]

    # A should receive more attention than B because its Q-value is larger.
    assert calls.count(
        ("start", "A")
    ) > calls.count(
        ("start", "B")
    )