"""DiRL graph-level learner.

Implements the Dijkstra-style lazy edge exploration from:

Jothimurugan et al.,
"Compositional Reinforcement Learning from Logical Specifications", NeurIPS 2021.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from faultline.graph.abstract_graph import AbstractGraph, Edge, Vertex
from faultline.learners.graph import EdgeTraining
from faultline.learners.start_distribution import StartDistributionWrapper


MIN_PROBABILITY = 1e-12


@dataclass
class _VertexRecord:
    """Best discovered path information for one graph vertex."""

    cost: float
    predecessor: Vertex | None = None
    incoming_edge: Edge | None = None
    reached_states: tuple[np.ndarray, ...] | None = None


class DiRL:
    """Dijkstra-style compositional RL over an abstract graph.

    DiRL lazily learns outgoing edge policies only when their source vertex
    becomes the cheapest unprocessed vertex.

    Edge cost:
        c(e) = -log(P(success | induced source distribution))

    The successful terminal states of the chosen incoming edge form an
    empirical approximation of the induced distribution at the target vertex.
    """

    def __init__(self, min_probability: float = MIN_PROBABILITY) -> None:
        self.min_probability = float(min_probability)

    def learn_graph(
        self,
        graph: AbstractGraph,
        env: Any,
        edge_learner: Any,
        budget: int,
        n_rollouts: int,
        seed: int,
    ) -> tuple[EdgeTraining, ...]:

        if graph.source is None or graph.target is None:
            raise ValueError("Graph source and target must be set")

        records = {
            vertex: _VertexRecord(cost=math.inf)
            for vertex in graph.vertices
        }
        records[graph.source].cost = 0.0

        processed: set[Vertex] = set()
        learned: list[EdgeTraining] = []

        while True:
            vertex = self._nearest_unprocessed(records, processed)

            if vertex is None:
                # No reachable/discovered vertices remain.
                break

            if vertex is graph.target:
                break

            processed.add(vertex)

            start_states = records[vertex].reached_states

            wrapped_env = StartDistributionWrapper(
                env,
                start_states=start_states,
                seed=seed,
            )

            for edge_index, edge in enumerate(graph.successors(vertex)):
                edge_seed = seed + 10_000 * len(processed) + edge_index

                policy = edge_learner.train_edge(
                    wrapped_env,
                    edge,
                    budget,
                    seed=edge_seed,
                )

                estimate = edge_learner.estimate_reach(
                    wrapped_env,
                    edge,
                    policy,
                    n_rollouts,
                    edge_seed,
                )

                learned.append(
                    EdgeTraining(
                        edge=edge,
                        policy=policy,
                        estimate=estimate,
                    )
                )

                p = max(
                    estimate.success_probability,
                    self.min_probability,
                )

                edge_cost = -math.log(p)
                candidate_cost = records[vertex].cost + edge_cost

                target = edge.target

                if candidate_cost < records[target].cost:
                    records[target] = _VertexRecord(
                        cost=candidate_cost,
                        predecessor=vertex,
                        incoming_edge=edge,
                        reached_states=self._successful_terminal_positions(
                            edge,
                            estimate,
                        ),
                    )

        return tuple(learned)

    @staticmethod
    def _nearest_unprocessed(
        records: dict[Vertex, _VertexRecord],
        processed: set[Vertex],
    ) -> Vertex | None:
        """Return the discovered unprocessed vertex with minimum path cost."""

        candidates = [
            vertex
            for vertex, record in records.items()
            if vertex not in processed and math.isfinite(record.cost)
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda vertex: (
                records[vertex].cost,
                vertex.name,
            ),
        )

    @staticmethod
    def _successful_terminal_positions(
            edge,
            estimate,
    ) -> tuple[np.ndarray, ...] | None:
        """Return successful rollout endpoints that really lie in edge.target."""

        positions: list[np.ndarray] = []

        for rollout in estimate.successful():
            if not rollout.states:
                continue

            position = np.asarray(
                rollout.states[-1][:2],
                dtype=float,
            )

            if edge.target.region.evaluate(position):
                positions.append(position)

        if not positions:
            return None

        return tuple(positions)