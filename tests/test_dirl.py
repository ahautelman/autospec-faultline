from __future__ import annotations

import numpy as np

from faultline.envs import Rooms9
from faultline.graph.abstract_graph import AbstractGraph
from faultline.learners.base import EdgeEstimate, Rollout
from faultline.learners.dirl import DiRL
from faultline.spectrl.predicates import Box, Universe


def _region(x: float, y: float) -> Box:
    return Box((x, y), (x + 1.0, y + 1.0))


def _build_branch_graph() -> AbstractGraph:
    """
              A
            /   \
        start   goal
            \   /
              B
    """
    graph = AbstractGraph()

    start = graph.add_vertex("start", _region(1, 1))
    a = graph.add_vertex("A", _region(5, 5))
    b = graph.add_vertex("B", _region(5, 10))
    goal = graph.add_vertex("goal", _region(10, 10))

    graph.add_edge(start, a, Universe())
    graph.add_edge(start, b, Universe())
    graph.add_edge(a, goal, Universe())
    graph.add_edge(b, goal, Universe())

    graph.source = start
    graph.target = goal

    return graph


class FakeDiRLLearner:
    """Cheap deterministic edge learner for testing DiRL."""

    def __init__(self, probabilities: dict[tuple[str, str], float]):
        self.probabilities = probabilities
        self.trained: list[tuple[str, str]] = []
        self.received_start_states: dict[
            tuple[str, str], tuple[np.ndarray, ...] | None
        ] = {}

    def train_edge(self, env, edge, budget, seed=None):
        key = (edge.source.name, edge.target.name)
        self.trained.append(key)

        # DiRL wraps the environment with StartDistributionWrapper.
        self.received_start_states[key] = getattr(
            env, "start_states", None
        )

        return key

    def estimate_reach(
            self,
            env,
            edge,
            policy,
            n_rollouts,
            seed,
    ):
        key = (edge.source.name, edge.target.name)
        p = self.probabilities[key]

        successes = round(p * 10)
        rollouts = []

        for i in range(10):
            success = i < successes

            if success:
                low = np.asarray(edge.target.region.lower, dtype=float)
                high = np.asarray(edge.target.region.upper, dtype=float)
                position = (low + high) / 2.0

                terminal = np.array(
                    [position[0], position[1], 0.0, 0.0],
                    dtype=float,
                )
            else:
                terminal = np.array(
                    [0.0, 0.0, 0.0, 0.0],
                    dtype=float,
                )

            rollouts.append(
                Rollout(
                    states=(terminal,),
                    reached=success,
                    safe=True,
                )
            )

        return EdgeEstimate(
            success_probability=p,
            rollouts=tuple(rollouts),
        )


def test_dirl_trains_edges_from_discovered_vertices():
    graph = _build_branch_graph()
    env = Rooms9()

    learner = FakeDiRLLearner(
        {
            ("start", "A"): 0.9,
            ("start", "B"): 0.2,
            ("A", "goal"): 0.9,
            ("B", "goal"): 0.9,
        }
    )

    dirl = DiRL()

    results = dirl.learn_graph(
        graph=graph,
        env=env,
        edge_learner=learner,
        budget=1,
        n_rollouts=10,
        seed=0,
    )

    trained = [
        (result.edge.source.name, result.edge.target.name)
        for result in results
    ]

    assert ("start", "A") in trained
    assert ("start", "B") in trained

    # A is much cheaper to reach than B:
    #
    # -log(0.9) < -log(0.2)
    #
    # so A should be processed before B.
    assert ("A", "goal") in trained

    # Once the goal has been reached through A, Dijkstra terminates.
    # B therefore does not need to be expanded.
    assert ("B", "goal") not in trained


def test_dirl_prefers_higher_probability_path():
    graph = _build_branch_graph()
    env = Rooms9()

    learner = FakeDiRLLearner(
        {
            ("start", "A"): 0.95,
            ("start", "B"): 0.10,
            ("A", "goal"): 0.95,
            ("B", "goal"): 1.0,
        }
    )

    dirl = DiRL()

    dirl.learn_graph(
        graph=graph,
        env=env,
        edge_learner=learner,
        budget=1,
        n_rollouts=10,
        seed=0,
    )

    # start->A and start->B are evaluated when start is expanded.
    #
    # A then has the lowest cumulative cost, so its outgoing edge should
    # be trained before anything leaving B.
    assert learner.trained[:3] == [
        ("start", "A"),
        ("start", "B"),
        ("A", "goal"),
    ]


def test_dirl_passes_successful_terminal_states_forward():
    graph = _build_branch_graph()
    env = Rooms9()

    learner = FakeDiRLLearner(
        {
            ("start", "A"): 1.0,
            ("start", "B"): 0.1,
            ("A", "goal"): 1.0,
            ("B", "goal"): 1.0,
        }
    )

    dirl = DiRL()

    dirl.learn_graph(
        graph=graph,
        env=env,
        edge_learner=learner,
        budget=1,
        n_rollouts=10,
        seed=0,
    )

    received = learner.received_start_states[("A", "goal")]

    assert received is not None
    assert len(received) > 0

    # The successful terminal state of start->A was [5.5, 5.5].
    assert np.allclose(received[0], [5.5, 5.5])


def test_dirl_does_not_expand_unreachable_vertex():
    graph = AbstractGraph()

    start = graph.add_vertex("start", _region(1, 1))
    reachable = graph.add_vertex("reachable", _region(5, 5))
    unreachable = graph.add_vertex("unreachable", _region(15, 15))
    goal = graph.add_vertex("goal", _region(10, 10))

    graph.add_edge(start, reachable, Universe())
    graph.add_edge(reachable, goal, Universe())

    # This vertex exists in the graph, but there is no path from start to it.
    graph.add_edge(unreachable, goal, Universe())

    graph.source = start
    graph.target = goal

    learner = FakeDiRLLearner(
        {
            ("start", "reachable"): 1.0,
            ("reachable", "goal"): 1.0,
            ("unreachable", "goal"): 1.0,
        }
    )

    dirl = DiRL()

    dirl.learn_graph(
        graph=graph,
        env=Rooms9(),
        edge_learner=learner,
        budget=1,
        n_rollouts=10,
        seed=0,
    )

    assert ("unreachable", "goal") not in learner.trained