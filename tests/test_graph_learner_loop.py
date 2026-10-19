"""DiRL and the start-state wrapper against the real rooms environment and loop."""

from __future__ import annotations

import numpy as np

from faultline.learners import DiRL, TabularLearner
from faultline.learners.start_distribution import StartDistributionWrapper
from faultline.refinement import AutoSpecConfig
from faultline.scenarios.catalogue import blocked_two_path, trap_goal
from faultline.scenarios.runner import run_scenario

CONFIG = AutoSpecConfig(budget=0, n_rollouts=20, seed=0)


def test_wrapper_resets_from_the_given_start_states_and_forwards_the_env_api():
    scenario = trap_goal()
    graph = scenario.build_graph()
    edge = graph.edges[0]
    wrapped = StartDistributionWrapper(scenario.make_env(), start_states=None, seed=0)
    wrapped.set_abstract_states(edge.source.region, edge.target.region, edge.safe)
    start = (np.asarray(edge.source.region.lower) + np.asarray(edge.source.region.upper)) / 2.0
    wrapped.set_start_states((start,))

    observation, _info = wrapped.reset(seed=1)

    assert np.allclose(observation[:2], start)


def test_run_scenario_accepts_dirl_and_repairs_the_trap_goal_scenario():
    result = run_scenario(trap_goal(), TabularLearner(), CONFIG, DiRL())

    assert result.operators == ("seq",)
    assert result.best_path_after > result.best_path_before


def test_run_scenario_without_a_graph_learner_is_unchanged_by_the_dirl_support():
    plain = run_scenario(blocked_two_path(), TabularLearner(), CONFIG)
    dirl = run_scenario(blocked_two_path(), TabularLearner(), CONFIG, DiRL())

    assert plain.operators == dirl.operators == ()
    assert plain.best_path_after == dirl.best_path_after == 1.0
