"""Fast acceptance tests for the 9-rooms scenarios and the runner.

Two groups. The first pins the catalogue: the five encodings, their expected
operators, and the graph structure each operator needs. The second runs scenarios
end to end. The deterministic tabular planner drives the scenarios whose failure
is unreachable sampled points or an unsafe alternate route (trap goal, narrow door,
mixed start) over five seeds, asserting a stable operator selection. A scripted
learner drives the long-corridor and blocked two-path scenarios, whose failure is a
policy-capacity effect the planner does not have; the real PPO acceptance is the
separate slow suite.
"""

from __future__ import annotations

import pytest

from faultline.learners import TabularLearner
from faultline.refinement import AutoSpecConfig
from faultline.scenarios import SCENARIOS, run_scenario
from tests.loop_support import ScriptedLearner, blocked_rollouts, refinable_rollouts

SEEDS = (0, 1, 2, 3, 4)
# The planner's estimates are deterministic, so a relaxed threshold is enough to
# expose the operator ordering without the paper's strict 0.99 used by the slow
# PPO suite. The rollout count is high enough that PastRefine's mean-bisector
# half-plane lands on the true boundary for every seed.
PLANNER_OPTIONS = {"threshold": 0.9, "n_rollouts": 80}
TABULAR_OPERATORS = (("trap_goal", "seq"), ("narrow_door", "add"), ("mixed_start", "past"))


def _planner_run(name, seed=0):
    """Run one scenario with the tabular planner and the relaxed threshold."""
    config = AutoSpecConfig(seed=seed, **PLANNER_OPTIONS)
    return run_scenario(SCENARIOS[name], TabularLearner(), config)


def test_catalogue_has_the_five_table_scenarios():
    assert list(SCENARIOS) == [
        "trap_goal",
        "narrow_door",
        "long_corridor",
        "mixed_start",
        "blocked_two_path",
    ]
    assert [scenario.expected_operator for scenario in SCENARIOS.values()] == [
        "seq",
        "seq",
        "add",
        "past",
        "or",
    ]


def test_catalogue_records_the_section_11_reference_results():
    references = {
        name: (scenario.reference_before, scenario.reference_after)
        for name, scenario in SCENARIOS.items()
    }
    assert references == {
        "trap_goal": (0.15, 0.85),
        "narrow_door": (0.30, 0.75),
        "long_corridor": (0.20, 0.90),
        "mixed_start": (0.40, 0.80),
        "blocked_two_path": (0.0, 0.85),
    }


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_scenario_graph_is_a_dag_with_endpoints(name):
    graph = SCENARIOS[name].build_graph()
    graph.assert_acyclic()
    assert graph.source is not None
    assert graph.target is not None
    assert len(graph.edges) >= 1


def test_blocked_two_path_goal_has_two_parents():
    graph = SCENARIOS["blocked_two_path"].build_graph()
    parents = {edge.source.name for edge in graph.predecessors(graph.target)}
    assert parents == {"start", "mid"}


def test_trap_goal_selects_reach_refinement():
    result = _planner_run("trap_goal")
    assert result.operators == ("seq",)
    assert result.best_path_before < 0.6
    assert result.best_path_after > 0.9


def test_mixed_start_selects_past_refinement():
    result = _planner_run("mixed_start")
    assert result.operators == ("past",)
    assert result.best_path_before < 0.6
    assert result.best_path_after > 0.9


def test_narrow_door_commits_the_refinement_the_planner_can_exploit():
    # The planner ignores the safe region, so it repairs the dangerous alternate
    # route with AddRefine rather than the paper's AvoidRefine, which the slow PPO
    # suite covers. Either way a refinement must commit and raise the success rate.
    result = _planner_run("narrow_door")
    assert result.operators == ("add",)
    assert result.best_path_after > result.best_path_before


def test_planner_needs_no_repair_on_the_long_corridor():
    result = _planner_run("long_corridor")
    assert result.operators == ()
    assert result.best_path_before == pytest.approx(1.0)


@pytest.mark.parametrize(("name", "operator"), TABULAR_OPERATORS)
def test_tabular_operator_selection_is_stable_over_seeds(name, operator):
    selections = {_planner_run(name, seed).operators for seed in SEEDS}
    assert selections == {(operator,)}


def test_long_corridor_selects_add_with_a_capacity_learner():
    def probability(edge):
        if "seq" in edge.source.name or "seq" in edge.target.name:
            return 0.2
        if "add" in edge.source.name or "add" in edge.target.name:
            return 0.99
        return 0.5

    learner = ScriptedLearner(probability, refinable_rollouts)
    result = run_scenario(SCENARIOS["long_corridor"], learner, AutoSpecConfig(threshold=0.9))

    assert result.operators == ("add",)
    assert result.best_path_after > result.best_path_before


def test_blocked_two_path_selects_or_with_a_capacity_learner():
    def probability(edge):
        if edge.source.name == "start" and edge.target.name == "goal":
            return 0.4
        return 0.9

    learner = ScriptedLearner(probability, blocked_rollouts)
    result = run_scenario(SCENARIOS["blocked_two_path"], learner, AutoSpecConfig(threshold=0.8))

    assert result.operators == ("or",)
    assert result.best_path_before < 0.6
    assert result.best_path_after > result.best_path_before


def test_repeated_tabular_run_is_deterministic():
    first = _planner_run("trap_goal", seed=3)
    second = _planner_run("trap_goal", seed=3)

    assert first.operators == second.operators
    assert first.edges == second.edges
    assert first.best_path_after == second.best_path_after
