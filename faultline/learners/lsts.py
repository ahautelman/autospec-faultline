"""LSTS graph-level learner.

Logical Specifications-guided Dynamic Task Sampling.

LSTS sits above an edge learner such as PPOEdgeLearner. It decides which
currently active abstract-graph edge receives the next chunk of training.

Each active task is sampled at least once before epsilon-greedy selection is
used. This avoids starving unseen tasks whose Q-value is still zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.learners.graph import EdgeTraining


@dataclass
class _TaskState:
    """Internal state maintained by the LSTS teacher for one edge."""

    edge: Edge

    # Teacher estimate for how promising this task currently is.
    q_value: float = 0.0

    # Student policy associated with this task.
    policy: Any | None = None

    # Most recent rollout estimate.
    estimate: Any | None = None

    # Previous success score, used to determine convergence.
    previous_score: float | None = None

    # Total number of environment interactions allocated to this task.
    interactions: int = 0

    # Number of times this task has been selected by the teacher.
    #
    # This is used to guarantee that every newly active task is tried
    # at least once before epsilon-greedy exploitation begins.
    selections: int = 0


class LSTS:
    """Logical Specifications-guided Dynamic Task Sampling.

    LSTS maintains a set of active abstract-graph edges. At every teacher
    iteration, one active edge is selected and receives another chunk of
    reinforcement-learning interactions.

    Selection is epsilon-greedy according to a teacher Q-value.

    A task is considered converged when:

        success_probability >= threshold

    and:

        abs(current_score - previous_score) < tolerance

    Once an edge converges, outgoing edges from its target vertex become
    active.

    Parameters
    ----------
    epsilon:
        Probability of selecting a random active task rather than the task
        with the highest Q-value.

    alpha:
        Teacher Q-value update coefficient:

            Q <- alpha * score + (1 - alpha) * Q

    threshold:
        Minimum success probability required before a task can converge.

    tolerance:
        Maximum difference between consecutive task scores for the task to
        be considered stable.

    chunk_size:
        Number of environment interactions assigned to a task each time it
        is selected.

    max_interactions:
        Maximum total number of student interactions across all tasks.
        If None, the budget passed to ``learn_graph`` is used.
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        threshold: float = 0.99,
        tolerance: float = 0.01,
        chunk_size: int = 5_000,
        max_interactions: int | None = None,
    ) -> None:

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")

        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")

        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if max_interactions is not None and max_interactions <= 0:
            raise ValueError("max_interactions must be positive")

        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.tolerance = float(tolerance)
        self.chunk_size = int(chunk_size)
        self.max_interactions = max_interactions

    def learn_graph(
        self,
        graph: AbstractGraph,
        env: Any,
        edge_learner: Any,
        budget: int,
        n_rollouts: int,
        seed: int,
    ) -> tuple[EdgeTraining, ...]:
        """Learn graph-edge policies using dynamic task sampling.

        Parameters
        ----------
        graph:
            Abstract SpectRL graph.

        env:
            Environment used by the underlying edge learner.

        edge_learner:
            Learner for individual reach-avoid tasks.

            For LSTS this learner must provide:

                train_edge(...)
                continue_edge(...)
                estimate_reach(...)

        budget:
            Default total interaction budget when ``max_interactions`` was
            not supplied to the constructor.

        n_rollouts:
            Number of evaluation rollouts after each training chunk.

        seed:
            Random seed used by the teacher and underlying learner.

        Returns
        -------
        tuple[EdgeTraining, ...]
            All edges that received at least one training chunk.
        """

        if graph.source is None:
            raise ValueError("Graph source must be set")

        if graph.target is None:
            raise ValueError("Graph target must be set")

        if not hasattr(edge_learner, "continue_edge"):
            raise TypeError(
                "LSTS requires an edge learner implementing continue_edge(). "
                "PPOEdgeLearner should provide this method."
            )

        rng = np.random.default_rng(seed)

        if self.max_interactions is not None:
            total_budget = self.max_interactions
        else:
            total_budget = int(budget)

        # Only keep edges that can eventually contribute to a path to the
        # target. This avoids spending training effort on dead-end tasks.
        usable_edges = {
            edge
            for edge in graph.edges
            if self._can_reach_target(graph, edge.target)
        }

        task_states = {
            edge: _TaskState(edge=edge)
            for edge in usable_edges
        }

        # Initially only edges leaving the graph source are active.
        active: set[Edge] = {
            edge
            for edge in graph.successors(graph.source)
            if edge in usable_edges
        }

        learned: set[Edge] = set()

        interactions = 0
        teacher_step = 0

        while active and interactions < total_budget:

            edge = self._sample_task(
                active=active,
                tasks=task_states,
                rng=rng,
            )

            task = task_states[edge]

            # Record that this task has now received a teacher selection.
            task.selections += 1

            remaining = total_budget - interactions
            chunk = min(self.chunk_size, remaining)

            if chunk <= 0:
                break

            training_seed = seed + teacher_step

            # First selection creates a new policy.
            if task.policy is None:
                task.policy = edge_learner.train_edge(
                    env,
                    edge,
                    chunk,
                    seed=training_seed,
                )

            # Later selections continue training the existing policy.
            else:
                task.policy = edge_learner.continue_edge(
                    env,
                    edge,
                    task.policy,
                    chunk,
                    seed=training_seed,
                )

            task.interactions += chunk
            interactions += chunk

            # Evaluate the updated student policy.
            estimate = edge_learner.estimate_reach(
                env,
                edge,
                task.policy,
                n_rollouts,
                training_seed,
            )

            task.estimate = estimate

            score = float(estimate.success_probability)

            # Teacher Q update.
            task.q_value = (
                self.alpha * score
                + (1.0 - self.alpha) * task.q_value
            )

            converged = self._converged(
                task=task,
                current_score=score,
            )

            # Save this score for the next stability comparison.
            task.previous_score = score

            if converged:
                active.remove(edge)
                learned.add(edge)

                # Once this edge is learned, tasks leaving its target become
                # available to the teacher.
                self._activate_successors(
                    graph=graph,
                    vertex=edge.target,
                    usable_edges=usable_edges,
                    learned=learned,
                    active=active,
                )

                # If this learned task reaches the graph target, there is now
                # at least one learned chain reaching the final specification
                # node. For the current compact implementation we stop here.
                if edge.target is graph.target:
                    break

            teacher_step += 1

        # Return every task that actually received training.
        return tuple(
            EdgeTraining(
                edge=edge,
                policy=task.policy,
                estimate=task.estimate,
            )
            for edge, task in task_states.items()
            if task.policy is not None
            and task.estimate is not None
        )

    def _sample_task(
        self,
        active: set[Edge],
        tasks: dict[Edge, _TaskState],
        rng: np.random.Generator,
    ) -> Edge:
        """Select one active task.

        Newly activated tasks are sampled once before epsilon-greedy
        exploitation is allowed. This prevents an unseen task with Q=0 from
        being permanently starved after another task receives any positive
        score.
        """

        if not active:
            raise ValueError("Cannot sample from an empty active-task set")

        # Keep ordering deterministic for reproducible tests and experiments.
        ordered = sorted(
            active,
            key=lambda edge: (
                edge.source.name,
                edge.target.name,
            ),
        )

        # ---------------------------------------------------------------
        # Initial exploration
        # ---------------------------------------------------------------
        # Every newly active task receives one training/evaluation attempt.
        untried = [
            edge
            for edge in ordered
            if tasks[edge].selections == 0
        ]

        if untried:
            return untried[0]

        # ---------------------------------------------------------------
        # Epsilon-greedy teacher
        # ---------------------------------------------------------------
        if rng.random() < self.epsilon:
            index = int(rng.integers(len(ordered)))
            return ordered[index]

        best_q = max(
            tasks[edge].q_value
            for edge in ordered
        )

        best = [
            edge
            for edge in ordered
            if tasks[edge].q_value == best_q
        ]

        # Deterministic tie breaking makes runs reproducible.
        return best[0]

    def _converged(
        self,
        task: _TaskState,
        current_score: float,
    ) -> bool:
        """Return whether a task is learned and stable."""

        # Performance must first be sufficiently high.
        if current_score < self.threshold:
            return False

        # One measurement cannot establish stability.
        if task.previous_score is None:
            return False

        improvement = abs(
            current_score - task.previous_score
        )

        return improvement < self.tolerance

    @staticmethod
    def _activate_successors(
        graph: AbstractGraph,
        vertex: Any,
        usable_edges: set[Edge],
        learned: set[Edge],
        active: set[Edge],
    ) -> None:
        """Activate tasks leaving a newly learned vertex."""

        for edge in graph.successors(vertex):

            if edge not in usable_edges:
                continue

            if edge in learned:
                continue

            if edge in active:
                continue

            active.add(edge)

    @staticmethod
    def _can_reach_target(
        graph: AbstractGraph,
        start: Any,
    ) -> bool:
        """Return True when ``start`` lies on some path to graph.target."""

        if graph.target is None:
            return False

        stack = [start]
        visited = set()

        while stack:
            vertex = stack.pop()

            if vertex in visited:
                continue

            visited.add(vertex)

            if vertex is graph.target:
                return True

            for edge in graph.successors(vertex):
                stack.append(edge.target)

        return False