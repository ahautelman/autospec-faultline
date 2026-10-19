"""Process-pool plumbing for the full Algorithm 1 run: one task per edge training.

A pool worker trains one edge with the report's checkpointed recipe and returns
the light record, never the model. The serial executor runs the same task in the
calling process, so ``--workers 1`` and a real pool differ only by the process
boundary.
"""

from __future__ import annotations

import contextlib
import io
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from faultline.experiments.pipeline import configure_worker
from faultline.experiments.training import ReportSettings, TrainingRecord, train_with_checkpoints
from faultline.graph.abstract_graph import Edge
from faultline.learners import PPOEdgeLearner
from faultline.scenarios import resolve_scenario


@dataclass(frozen=True)
class EdgeRequest:
    """Everything a worker needs to train one edge.

    Args:
        scenario: Scenario name; only the name crosses the process boundary
            because a scenario's graph factory is not picklable.
        edge: The reach-avoid edge to train.
        seed: Training and evaluation seed.
        settings: Budget, rollout counts, and checkpoint interval.
        quiet: Whether to swallow the per-checkpoint log lines.

    How:
        Bundles the task inputs into one picklable value.
    """

    scenario: str
    edge: Edge
    seed: int
    settings: ReportSettings
    quiet: bool


class SerialExecutor(Executor):
    """An executor that runs each submitted call immediately in this process.

    How:
        Runs the call inside ``submit`` and returns an already-finished future
        carrying the result or the exception, so callers use one code path.
    """

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future:
        """Run ``fn`` now and wrap its outcome in a finished future.

        Args:
            fn: The callable to run.
            *args: Positional arguments for ``fn``.
            **kwargs: Keyword arguments for ``fn``.

        How:
            Calls ``fn`` and stores its result, or the exception it raised.

        Returns:
            A completed :class:`~concurrent.futures.Future`.
        """
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as err:  # noqa: BLE001 - re-raised to the caller by future.result()
            future.set_exception(err)
        return future


def train_edge_task(request: EdgeRequest) -> TrainingRecord:
    """Train one edge with checkpoints and return the record without its model.

    Args:
        request: The edge, seed, and settings to train with.

    How:
        Resolves the scenario name through the shared resolver, which handles
        both the 9-rooms catalogue and the n-rooms worlds (a random world is
        rebuilt from its seed), then builds a fresh environment and a seeded PPO
        learner, runs :func:`train_with_checkpoints`, and drops the policy so the
        record stays small to send back to the parent process.

    Returns:
        The edge's :class:`TrainingRecord` with ``policy`` set to ``None``.
    """
    scenario = resolve_scenario(request.scenario)
    learner = PPOEdgeLearner(seed=request.seed)
    muted = contextlib.redirect_stdout(io.StringIO()) if request.quiet else contextlib.nullcontext()
    with muted:
        record = train_with_checkpoints(
            learner, scenario.make_env(), request.edge, request.seed, request.settings
        )
    return replace(record, policy=None)


def make_executor(workers: int) -> Executor:
    """Build the executor the edge trainings are submitted to.

    Args:
        workers: Worker-process count; ``1`` or less runs serially in-process.

    How:
        Returns a :class:`SerialExecutor` for one worker, otherwise a process pool
        whose workers are pinned to one compute thread by
        :func:`~faultline.experiments.pipeline.configure_worker`.

    Returns:
        The executor; the caller owns it and must shut it down.
    """
    if workers <= 1:
        return SerialExecutor()
    return ProcessPoolExecutor(max_workers=workers, initializer=configure_worker)
