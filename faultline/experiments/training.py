"""Checkpointed PPO training for the 9-rooms report.

The report needs a learning curve per edge, not just a final number. This module
wraps the PPO learner so ``model.learn`` is instrumented with a callback that, at
a fixed step interval, evaluates the current policy over the paper's rollouts and
records ``(step, success_probability)``. The training environment is left
untouched: the callback evaluates on a separate environment built from the same
layout, so mid-training estimates never disturb the rollout buffer.

The final checkpoint's rollouts are the trajectories the refinement operator
consumes, so the overlay figure reuses them instead of resampling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from faultline.envs import Rooms9
from faultline.experiments.records import Checkpoint
from faultline.graph.abstract_graph import Edge
from faultline.learners import EdgeEstimate, PPOEdgeLearner

CheckpointRecorder = Callable[[int, Any], None]

# The paper's global settings: 200k steps for every scenario, 1000 evaluation
# rollouts, and a checkpoint every 10k steps so the 80k reference point exists.
DEFAULT_BUDGET = 200_000
DEFAULT_ROLLOUTS = 1000
DEFAULT_INTERVAL = 10_000


@dataclass(frozen=True)
class ReportSettings:
    """The global training settings for the report.

    Args:
        budget: Training budget in steps for every edge and scenario.
        rollouts: Evaluation rollouts per success estimate.
        interval: Steps between checkpoints.

    How:
        Bundles the three global numbers so the pipeline functions stay within a
        few parameters and one settings object drives both report steps.
    """

    budget: int = DEFAULT_BUDGET
    rollouts: int = DEFAULT_ROLLOUTS
    interval: int = DEFAULT_INTERVAL


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


def make_checkpoint_callback(interval: int, recorder: CheckpointRecorder) -> Any:
    """Build a Stable-Baselines3 callback that records every ``interval`` steps.

    Args:
        interval: Steps between evaluations; must be positive.
        recorder: Called as ``recorder(step, model)`` at each checkpoint.

    How:
        Defines a small ``BaseCallback`` subclass inside the function, so this
        module never imports Stable-Baselines3 unless checkpointed training runs.
        The callback fires when ``num_timesteps`` first reaches each multiple of
        ``interval``.

    Returns:
        The callback instance, ready for ``model.learn(callback=...)``.

    Raises:
        ValueError: If ``interval`` is not positive.
    """
    if interval < 1:
        raise ValueError(f"Checkpoint interval must be positive, got {interval}")
    from stable_baselines3.common.callbacks import BaseCallback

    class _CheckpointCallback(BaseCallback):
        """Evaluate and record the policy at a fixed step interval."""

        def __init__(self) -> None:
            super().__init__()
            self._next_step = interval

        def _on_step(self) -> bool:
            if self.num_timesteps >= self._next_step:
                recorder(int(self.num_timesteps), self.model)
                self._next_step += interval
            return True

    return _CheckpointCallback()


def train_with_checkpoints(
    learner: PPOEdgeLearner, env: Rooms9, edge: Edge, seed: int, settings: ReportSettings
) -> TrainingRecord:
    """Train one edge with periodic success checkpoints.

    Args:
        learner: The PPO learner; its section 6.1 recipe is left unchanged.
        env: The training environment.
        edge: The reach-avoid edge to train.
        seed: Base seed for both training and evaluation.
        settings: Budget, rollout count, and checkpoint interval.

    How:
        Builds a separate evaluation environment over the same layout, runs the
        learner for the budget with the checkpoint callback attached and the seed
        threaded into training, then takes one final estimate. The series is the
        callback's checkpoints; a final point at the budget is appended when the
        interval does not divide it. The returned estimate is the last
        checkpoint's, so its rollouts match the trajectories the operator will
        consume.

    Returns:
        The :class:`TrainingRecord` with the curve and the final estimate.
    """
    eval_env = Rooms9(layout=env.layout)
    checkpoints: list[Checkpoint] = []
    final: dict[str, EdgeEstimate] = {}

    def record(step: int, model: Any) -> None:
        estimate = learner.estimate_reach(eval_env, edge, model, settings.rollouts, seed)
        checkpoints.append(Checkpoint(step, estimate.success_probability))
        final["estimate"] = estimate

    callback = make_checkpoint_callback(settings.interval, record)
    model = learner.train_edge(env, edge, settings.budget, callback=callback, seed=seed)
    measured = learner.estimate_reach(env, edge, model, settings.rollouts, seed)
    if not checkpoints or checkpoints[-1].step < settings.budget:
        checkpoints.append(Checkpoint(settings.budget, measured.success_probability))
        final["estimate"] = measured
    return TrainingRecord(edge.source.name, edge.target.name, tuple(checkpoints), final["estimate"])
