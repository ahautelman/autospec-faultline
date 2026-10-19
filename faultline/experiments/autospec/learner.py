"""A learner that farms edge trainings out to a shared executor and records curves.

:func:`~faultline.refinement.loop.run_autospec` trains edges one call at a time. This
learner keeps that interface but submits each training to a pool shared by every
``(scenario, seed)`` run, and :meth:`PooledCheckpointLearner.prefetch` starts the
independent initial edges together. The loop's own ``estimate_reach`` call returns
the final checkpoint's estimate, which is the same rollouts the report evaluates, so
the loop runs unchanged and the learning curves come out as a by-product.
"""

from __future__ import annotations

from concurrent.futures import Executor, Future
from dataclasses import dataclass
from collections.abc import Iterable

from faultline.experiments.autospec.executor import EdgeRequest, train_edge_task
from faultline.experiments.training import ReportSettings, TrainingRecord
from faultline.graph.abstract_graph import Edge
from faultline.learners import EdgeEstimate
from faultline.refinement.loop import _edge_key as edge_key


@dataclass(frozen=True)
class TrainedHandle:
    """The policy handle the loop passes back to ``estimate_reach``.

    Args:
        key: The edge's structural key, which indexes its stored training.

    How:
        Stands in for the model, which never leaves the worker process.
    """

    key: tuple


class PooledCheckpointLearner:
    """The learner protocol backed by a shared executor and checkpointed training.

    Args:
        scenario: Scenario name the workers rebuild their environment from.
        seed: Seed for every training and evaluation of this run.
        settings: Budget, rollout counts, and checkpoint interval.
        executor: The executor shared by all runs.
        quiet: Whether workers swallow their per-checkpoint log lines.

    How:
        Submits one task per distinct edge key and caches the future, so a
        prefetched edge and a later ``train_edge`` call share one training.
        Finished trainings are kept in :attr:`trainings` so the run's curves can
        be read for any edge, committed or discarded.
    """

    def __init__(
        self, scenario: str, seed: int, settings: ReportSettings, executor: Executor, quiet: bool
    ) -> None:
        self.scenario = scenario
        self.seed = seed
        self.settings = settings
        self.executor = executor
        self.quiet = quiet
        self.trainings: dict[tuple, TrainingRecord] = {}
        self._futures: dict[tuple, Future] = {}

    def prefetch(self, edges: Iterable[Edge]) -> None:
        """Start the trainings of edges the loop is about to ask for.

        Args:
            edges: Edges to train ahead of the loop's own requests.

        How:
            Submits any edge without a pending or finished training and returns at
            once, so independent edges train in parallel.
        """
        for edge in edges:
            self._submit(edge)

    def train_edge(self, env: object, edge: Edge, budget: int, seed: int | None = None) -> TrainedHandle:
        """Wait for the edge's training and return its handle.

        Args:
            env: Ignored; workers build their own environment.
            edge: The reach-avoid edge to learn.
            budget: Must equal the settings' budget.
            seed: Ignored; the run's seed is fixed at construction.

        How:
            Submits the edge if it is not already running, waits for the record,
            and stores it under the edge key.

        Returns:
            A handle naming the stored training.

        Raises:
            ValueError: If ``budget`` differs from the settings' budget.
        """
        if budget != self.settings.budget:
            raise ValueError(f"Loop budget {budget} differs from settings {self.settings.budget}")
        key = self._submit(edge)
        record = self._futures[key].result()
        self.trainings[key] = record
        print(
            f"[train] scenario={self.scenario} seed={self.seed} edge={record.source}->{record.target} "
            f"success={record.estimate.success_probability:.3f}",
            flush=True,
        )
        return TrainedHandle(key)

    def estimate_reach(
        self, env: object, edge: Edge, policy: TrainedHandle, n_rollouts: int, seed: int
    ) -> EdgeEstimate:
        """Return the stored final estimate for a trained edge.

        Args:
            env: Ignored.
            edge: Ignored; the handle already names the edge.
            policy: The handle from :meth:`train_edge`.
            n_rollouts: Must equal the settings' rollout count.
            seed: Ignored; the final checkpoint used this run's seed.

        How:
            The worker already evaluated the final policy with the full rollout
            count and seed, so the stored estimate is returned instead of a second
            evaluation.

        Returns:
            The final checkpoint's estimate.

        Raises:
            ValueError: If ``n_rollouts`` differs from the settings' rollouts.
        """
        if n_rollouts != self.settings.rollouts:
            raise ValueError(f"Loop rollouts {n_rollouts} differ from settings {self.settings.rollouts}")
        return self.trainings[policy.key].estimate

    def _submit(self, edge: Edge) -> tuple:
        """Submit an edge's training once and return its key.

        Args:
            edge: The edge to train.

        How:
            Keys the edge structurally and submits a task only the first time
            that key is seen.

        Returns:
            The edge's structural key.
        """
        key = edge_key(edge)
        if key not in self._futures:
            request = EdgeRequest(self.scenario, edge, self.seed, self.settings, self.quiet)
            self._futures[key] = self.executor.submit(train_edge_task, request)
        return key
