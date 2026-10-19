"""Graph-level specification-guided learning strategies.

PPOEdgeLearner learns one reach-avoid edge. DiRL and LSTS decide which graph
edges to learn, in what order, and with what training budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.learners.base import EdgeEstimate


@dataclass
class EdgeTraining:
    """Policy and estimate learned for one abstract-graph edge."""

    edge: Edge
    policy: Any
    estimate: EdgeEstimate


class GraphLearner(Protocol):
    """High-level algorithm for learning policies over an abstract graph."""

    def learn_graph(
        self,
        graph: AbstractGraph,
        env: Any,
        edge_learner: Any,
        budget: int,
        n_rollouts: int,
        seed: int,
    ) -> tuple[EdgeTraining, ...]:
        """Learn the subset of graph edges selected by the algorithm."""
        ...