"""Checkpointed PPO training and edge refinement for the 9-rooms report.

The report needs a learning curve per edge, not just a final number. This module
wraps the PPO learner so ``model.learn`` is instrumented with a callback that, at
a fixed step interval, evaluates the current policy over the paper's rollouts and
records ``(step, success_probability)``. The training environment is left
untouched: the callback evaluates on a separate environment built from the same
layout and trap (see :func:`make_eval_env`), so mid-training estimates never
disturb the rollout buffer.

It also holds the per-scenario edge driver, :class:`ScenarioTrainer`, which trains
the bottleneck edge and then the refined route the scenario's operator produces,
and scores a route end to end against the original edge's specification; and
:func:`route_product`, which combines a route's edge curves.

Rollout count per checkpoint is the report's dominant cost. Profiling a 20k-step
``train_edge`` against one 1000-rollout ``estimate_reach`` showed training at
~4s versus ~12s for a single 1000-rollout evaluation, and the default 200k budget
takes 20 such checkpoints. The paper-grade numbers are only read at 80k
(:data:`~faultline.experiments.records.REFERENCE_STEP`) and the final budget, so
every other checkpoint uses a small :data:`DEFAULT_INTERMEDIATE_ROLLOUTS` estimate
instead. That keeps the two compared values faithful while cutting the rest of the
curve by roughly an order of magnitude. The final estimate, whose rollouts feed
the refinement operator and the overlay, always keeps the full count.

The final checkpoint's rollouts are the trajectories the refinement operator
consumes, so the overlay figure reuses them instead of resampling.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from faultline.envs import RoomsEnv
from faultline.experiments.progress import progress_bar
from faultline.experiments.records import REFERENCE_STEP, Checkpoint
from faultline.experiments.satisfaction import route_satisfaction
from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.learners import CannotRefineError, EdgeEstimate, PPOEdgeLearner
from faultline.learners.ppo import greedy_action_fn
from faultline.refinement.candidates import (
    add_candidate,
    or_candidate,
    past_candidate,
    seq_candidate,
)
from faultline.refinement.operators import AVOID_TAIL_LENGTH

CheckpointRecorder = Callable[[int, Any], None]

# The paper's global settings: 200k steps for every scenario, 1000 evaluation
# rollouts, and a checkpoint every 10k steps so the 80k reference point exists.
DEFAULT_BUDGET = 200_000
DEFAULT_ROLLOUTS = 1000
DEFAULT_INTERVAL = 10_000
# Intermediate curve points only need to show the trend. At the default interval
# this turns 18 of the 20 checkpoints into a ~1s estimate instead of a ~12s one.
DEFAULT_INTERMEDIATE_ROLLOUTS = 100

# The candidate builder per operator name, shared with the Algorithm 1 loop so the
# report scores the same replacement route the loop would.
_CANDIDATE_BUILDERS = {
    "seq": seq_candidate,
    "add": add_candidate,
    "past": past_candidate,
    "or": or_candidate,
}


@dataclass(frozen=True)
class ReportSettings:
    """The global training settings for the report.

    Args:
        budget: Training budget in steps for every edge and scenario.
        rollouts: Evaluation rollouts for the two reported checkpoints, the
            reference step and the final budget.
        interval: Steps between checkpoints.
        intermediate_rollouts: Evaluation rollouts for the other checkpoints,
            which only shape the curve and are never compared to the paper.

    How:
        Bundles the global numbers so the pipeline functions stay within a few
        parameters and one settings object drives both report steps.
    """

    budget: int = DEFAULT_BUDGET
    rollouts: int = DEFAULT_ROLLOUTS
    interval: int = DEFAULT_INTERVAL
    intermediate_rollouts: int = DEFAULT_INTERMEDIATE_ROLLOUTS


@dataclass(frozen=True)
class TrainingRecord:
    """One checkpointed edge training.

    Args:
        source: Source vertex name of the trained edge.
        target: Target vertex name.
        checkpoints: The recorded success series, ordered by step.
        estimate: The final success estimate, whose rollouts feed the operators.
        policy: The trained model at the final budget, which the end-to-end
            satisfaction metric rolls out.

    How:
        Pairs the learning curve with the final estimate so the pipeline builds
        the overlay from the same rollouts the refinement operator consumes.
    """

    source: str
    target: str
    checkpoints: tuple[Checkpoint, ...]
    estimate: EdgeEstimate
    policy: Any


@dataclass(frozen=True)
class RefinementOutcome:
    """The result of trying a scenario's operator on the bottleneck edge.

    Args:
        series: The refined route's success series, or ``()`` when nothing was
            produced.
        refused: ``True`` when an operator was attempted but declined for lack of
            a witness, ``False`` when there was no operator or the refinement
            succeeded.
        route: The refined route's edges, ``()`` when nothing was produced.
        records: The route edges' trainings, in route order.

    How:
        Separates "the operator declined to refine" from "the refined route
        scored zero". Both leave the after curve equal to the before curve, so
        without this flag a report cannot tell a refusal from a genuine zero.
    """

    series: tuple[Checkpoint, ...]
    refused: bool
    route: tuple[Edge, ...] = ()
    records: tuple[TrainingRecord, ...] = ()


def make_checkpoint_callback(interval: int, recorder: CheckpointRecorder, progress: Any = None) -> Any:
    """Build a Stable-Baselines3 callback that records every ``interval`` steps.

    Args:
        interval: Steps between evaluations; must be positive.
        recorder: Called as ``recorder(step, model)`` at each checkpoint.
        progress: Optional step progress bar; advanced once per environment step.

    How:
        Defines a small ``BaseCallback`` subclass inside the function, so this
        module never imports Stable-Baselines3 unless checkpointed training runs.
        The callback fires when ``num_timesteps`` first reaches each multiple of
        ``interval``, and advances ``progress`` by the steps taken since its last
        call.

    Returns:
        The callback instance, ready for ``model.learn(callback=...)``.

    Raises:
        ValueError: If ``interval`` is not positive.
    """
    if interval < 1:
        raise ValueError(f"Checkpoint interval must be positive, got {interval}")
    from stable_baselines3.common.callbacks import BaseCallback

    class _CheckpointCallback(BaseCallback):
        """Evaluate, record, and tick the progress bar at each step."""

        def __init__(self) -> None:
            super().__init__()
            self._next_step = interval
            self._last_step = 0

        def _on_step(self) -> bool:
            current = int(self.num_timesteps)
            if progress is not None:
                progress.update(current - self._last_step)
                self._last_step = current
            if current >= self._next_step:
                recorder(current, self.model)
                self._next_step += interval
            return True

    return _CheckpointCallback()


def rollouts_for(step: int, settings: ReportSettings) -> int:
    """Return the evaluation rollout count to use at a checkpoint step.

    Args:
        step: Environment steps completed when the checkpoint fires.
        settings: The global settings, holding the reference and intermediate
            counts.

    How:
        Uses the full rollout count at the final budget and at the checkpoint that
        serves as the 80k reference reading (the last one at or below
        :data:`~faultline.experiments.records.REFERENCE_STEP`); every other
        checkpoint uses the small intermediate count.

    Returns:
        The number of evaluation rollouts for this checkpoint.
    """
    if step >= settings.budget:
        return settings.rollouts
    if REFERENCE_STEP - settings.interval < step <= REFERENCE_STEP:
        return settings.rollouts
    return min(settings.intermediate_rollouts, settings.rollouts)


def make_eval_env(env: RoomsEnv) -> RoomsEnv:
    """Build an evaluation environment that mirrors a training environment's world.

    Args:
        env: The training environment to mirror.

    How:
        Copies the wall-and-door layout, the trap, and the episode step limit, so
        mid-training estimates see exactly the world the policy trains in. Dropping
        the trap here would hide the unsafe rollouts the refinement operators depend
        on, so the estimate would disagree with training. Copying ``max_steps``
        keeps a long-horizon n-rooms run from silently reverting to the default
        limit.

    Returns:
        A fresh environment of the same class over the same layout, trap and horizon.
    """
    return type(env)(
        layout=env.layout, max_steps=env.max_steps, unsafe_region=env.unsafe_region
    )


def train_with_checkpoints(
    learner: PPOEdgeLearner,
    env: RoomsEnv,
    edge: Edge,
    seed: int,
    settings: ReportSettings,
) -> TrainingRecord:
    """Train one edge with periodic success checkpoints.

    Args:
        learner: The PPO learner; its section 6.1 recipe is left unchanged.
        env: The training environment.
        edge: The reach-avoid edge to train.
        seed: Base seed for both training and evaluation.
        settings: Budget, rollout counts, and checkpoint interval.

    How:
        Builds a separate evaluation environment over the same layout and trap,
        runs the learner for the budget with the checkpoint callback attached and
        the seed
        threaded into training. Each checkpoint prints its step, success, rollout
        count, and evaluation wall-clock, under the edge's endpoint names. The
        final estimate is the budget checkpoint's, so the extra evaluation the old
        pipeline ran at the end is skipped. When the interval does not land on the
        budget, a final full-count estimate is taken and appended.

    Returns:
        The :class:`TrainingRecord` with the curve and the final estimate.
    """
    eval_env = make_eval_env(env)
    checkpoints: list[Checkpoint] = []
    final: dict[str, EdgeEstimate] = {}
    tag = f"{edge.source.name}->{edge.target.name}"

    def record(step: int, model: Any) -> None:
        rollouts = rollouts_for(step, settings)
        started = time.perf_counter()
        estimate = learner.estimate_reach(eval_env, edge, model, rollouts, seed)
        checkpoints.append(Checkpoint(step, estimate.success_probability))
        if rollouts == settings.rollouts:
            final["estimate"] = estimate
        elapsed = time.perf_counter() - started
        print(
            f"    [{tag}] checkpoint step={step}/{settings.budget} "
            f"success={estimate.success_probability:.3f} rollouts={rollouts} eval={elapsed:.1f}s",
            flush=True,
        )

    bar = progress_bar(settings.budget, desc=f"train {tag}")
    callback = make_checkpoint_callback(settings.interval, record, progress=bar)
    try:
        model = learner.train_edge(env, edge, settings.budget, callback=callback, seed=seed)
    finally:
        bar.close()
    if "estimate" not in final:
        measured = learner.estimate_reach(env, edge, model, settings.rollouts, seed)
        checkpoints.append(Checkpoint(settings.budget, measured.success_probability))
        final["estimate"] = measured
    return TrainingRecord(
        edge.source.name, edge.target.name, tuple(checkpoints), final["estimate"], model
    )


class ScenarioTrainer:
    """Trains a scenario's edges with checkpoints.

    Args:
        env: The training environment.
        learner: The PPO learner the report drives.
        seed: Training seed for this run.
        settings: The global budget, rollout count, and checkpoint interval.

    How:
        Holds the training context so the pipeline can train any edge, and the
        refined route, without threading the settings through every call.
    """

    def __init__(
        self, env: RoomsEnv, learner: PPOEdgeLearner, seed: int, settings: ReportSettings
    ) -> None:
        self.env = env
        self.learner = learner
        self.seed = seed
        self.settings = settings

    def train(self, edge: Edge) -> TrainingRecord:
        """Train one edge with checkpoints.

        Args:
            edge: The reach-avoid edge to train.

        How:
            Delegates to :func:`~faultline.experiments.training.train_with_checkpoints`
            with this run's seed and settings.

        Returns:
            The edge's :class:`TrainingRecord`.
        """
        return train_with_checkpoints(self.learner, self.env, edge, self.seed, self.settings)

    def refined_series(
        self, graph: AbstractGraph, edge: Edge, estimate: EdgeEstimate, operator: str | None
    ) -> RefinementOutcome:
        """Train the refined route and return its series plus the refusal flag.

        Args:
            graph: The graph the operator runs on.
            edge: The bottleneck edge being refined.
            estimate: The edge's final estimate, supplying the rollouts.
            operator: The operator name, or ``None`` to skip refinement.

        How:
            Builds the operator's candidate through the same builder the loop
            uses. An operator that refuses for lack of a witness is reported
            explicitly, so the caller can tell a refusal apart from a refined
            route that scores zero. Otherwise every route edge is trained with
            checkpoints and the route product is returned.

        Returns:
            The :class:`RefinementOutcome`: the route's success series, empty when
            nothing refined, and whether an operator was refused.
        """
        if operator is None:
            return RefinementOutcome((), False)
        try:
            candidate = _CANDIDATE_BUILDERS[operator](graph, edge, estimate, AVOID_TAIL_LENGTH)
        except CannotRefineError:
            print(
                f"    refinement: {operator} refused "
                "(no witness: no successful trajectory or unsafe tail)",
                flush=True,
            )
            return RefinementOutcome((), True)
        route = candidate.routes[0]
        names = [route[0].source.name, *(route_edge.target.name for route_edge in route)]
        print(f"    refinement: {operator} route {' -> '.join(names)}", flush=True)
        records = tuple(self.train(route_edge) for route_edge in route)
        return RefinementOutcome(route_product(records), False, route, records)

    def satisfaction(
        self, edge: Edge, route: Sequence[Edge], records: Sequence[TrainingRecord]
    ) -> float:
        """Score a trained route end to end against the original edge.

        Args:
            edge: The original bottleneck edge, whose specification is judged.
            route: The edges standing in for ``edge``; ``(edge,)`` for the
                unrefined policy.
            records: The route edges' trainings, in route order.

        How:
            Builds each record's greedy action function and runs
            :func:`~faultline.experiments.satisfaction.route_satisfaction` on a
            fresh evaluation environment with the full rollout count and this
            run's seed.

        Returns:
            The fraction of end-to-end episodes that satisfy ``edge``.
        """
        steps = [
            (hop, greedy_action_fn(record.policy))
            for hop, record in zip(route, records, strict=True)
        ]
        env = make_eval_env(self.env)
        return route_satisfaction(env, edge, steps, self.settings.rollouts, self.seed)


def route_product(records: Sequence[TrainingRecord]) -> tuple[Checkpoint, ...]:
    """Multiply the route edges' series stepwise.

    Args:
        records: The trained route edges, in route order.

    How:
        Aligns the series on their shared checkpoint steps and multiplies the
        per-edge probabilities at each step, matching the loop's best-route
        product for a two-hop route. A single-edge route is returned unchanged.

    Returns:
        The route's success series.
    """
    if len(records) == 1:
        return records[0].checkpoints
    tables = [
        {checkpoint.step: checkpoint.probability for checkpoint in record.checkpoints}
        for record in records
    ]
    common = sorted(set.intersection(*(set(table) for table in tables)))
    return tuple(Checkpoint(step, math.prod(table[step] for table in tables)) for step in common)
