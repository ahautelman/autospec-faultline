"""Spec-guided RL backends.

Re-exports the learner protocol and its two implementations. The PPO learner
imports Stable-Baselines3 only when it trains, so this package imports without
the optional interop extra.
"""

from faultline.learners.base import CannotRefineError, EdgeEstimate, Learner, Rollout
from faultline.learners.ppo import PPOEdgeLearner
from faultline.learners.tabular import TabularLearner, TabularPolicy

__all__ = [
    "CannotRefineError",
    "EdgeEstimate",
    "Learner",
    "PPOEdgeLearner",
    "Rollout",
    "TabularLearner",
    "TabularPolicy",
]
