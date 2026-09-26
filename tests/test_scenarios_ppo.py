"""Slow PPO acceptance for the 9-rooms scenarios (paper sections 8 and 11).

Marked ``slow`` and excluded from the default suite, because it trains a PPO policy
per edge. Run it with ``uv run --extra dev --extra interop pytest -m slow``. The
budget and rollout count come from ``FAULTLINE_PPO_BUDGET`` and
``FAULTLINE_PPO_ROLLOUTS``; the paper's full reproduction uses 80000 steps and 1000
rollouts per edge, far slower than these defaults, which only check that the
scenario runner drives the real learner end to end and returns well-formed results.
"""

from __future__ import annotations

import os

import pytest

from faultline.learners import PPOEdgeLearner
from faultline.refinement import AutoSpecConfig
from faultline.scenarios import SCENARIOS, run_scenario

pytestmark = pytest.mark.slow

BUDGET = int(os.environ.get("FAULTLINE_PPO_BUDGET", "4096"))
ROLLOUTS = int(os.environ.get("FAULTLINE_PPO_ROLLOUTS", "20"))
# A small recipe keeps one edge's training to a few seconds; the paper's section
# 6.1 recipe is the learner default and is used for the full run. The learner is
# seeded so a failure reproduces.
RECIPE = {"n_steps": 256, "n_epochs": 2, "batch_size": 64}
SEED = 0


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_ppo_scenario_returns_well_formed_results(name):
    pytest.importorskip("stable_baselines3")
    learner = PPOEdgeLearner(recipe=RECIPE, budget=BUDGET, seed=SEED)
    config = AutoSpecConfig(budget=BUDGET, n_rollouts=ROLLOUTS, seed=SEED)

    result = run_scenario(SCENARIOS[name], learner, config)

    assert result.scenario == name
    assert 0.0 <= result.best_path_before <= 1.0
    assert 0.0 <= result.best_path_after <= 1.0
    assert len(result.edges) == len(SCENARIOS[name].build_graph().edges)
    for edge in result.edges:
        assert 0.0 <= edge.before <= 1.0
        assert 0.0 <= edge.after <= 1.0
