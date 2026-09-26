"""Checkpointed PPO training for the 9-rooms report.

The report needs a learning curve per edge, not just a final number. This module
wraps the PPO learner so ``model.learn`` is instrumented with a callback that, at
a fixed step interval, evaluates the current policy over the paper's rollouts and
records ``(step, success_probability)``. The training environment is left
untouched: the callback evaluates on a separate environment built from the same
layout, so mid-training estimates never disturb the rollout buffer.

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

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from faultline.envs import Rooms9
from faultline.experiments.progress import progress_bar
from faultline.experiments.records import REFERENCE_STEP, Checkpoint
from faultline.graph.abstract_graph import Edge
from faultline.learners import EdgeEstimate, PPOEdgeLearner

CheckpointRecorder = Callable[[int, Any], None]

# The paper's global settings: 200k steps for every scenario, 1000 evaluation
# rollouts, and a checkpoint every 10k steps so the 80k reference point exists.
DEFAULT_BUDGET = 200_000
DEFAULT_ROLLOUTS = 1000
DEFAULT_INTERVAL = 10_000
# Intermediate curve points only need to show the trend. At the default interval
# this turns 18 of the 20 checkpoints into a ~1s estimate instead of a ~12s one.
DEFAULT_INTERMEDIATE_ROLLOUTS = 100


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

    How:
        Pairs the learning curve with the final estimate so the pipeline builds
        the overlay from the same rollouts the refinement operator consumes.
    """

    source: str
    target: str
    checkpoints: tuple[Checkpoint, ...]
    estimate: EdgeEstimate


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


def train_with_checkpoints(
    learner: PPOEdgeLearner,
    env: Rooms9,
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
        Builds a separate evaluation environment over the same layout, runs the
        learner for the budget with the checkpoint callback attached and the seed
        threaded into training. Each checkpoint prints its step, success, rollout
        count, and evaluation wall-clock, under the edge's endpoint names. The
        final estimate is the budget checkpoint's, so the extra evaluation the old
        pipeline ran at the end is skipped. When the interval does not land on the
        budget, a final full-count estimate is taken and appended.

    Returns:
        The :class:`TrainingRecord` with the curve and the final estimate.
    """
    eval_env = Rooms9(layout=env.layout)
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
    return TrainingRecord(edge.source.name, edge.target.name, tuple(checkpoints), final["estimate"])
