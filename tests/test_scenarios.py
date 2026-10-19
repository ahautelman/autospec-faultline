"""Fast acceptance tests for the 9-rooms scenarios and the runner.

Two groups. The first pins the catalogue: the five encodings, their expected
operators, and the graph structure each operator needs. The second runs scenarios
end to end. The deterministic tabular planner drives the scenarios whose failure
is unreachable sampled points (trap goal, mixed start) over five seeds, asserting a
stable operator selection; narrow_door's failure is an unmodelled environment trap
the safety-blind planner cannot repair, so it is checked on its own. A scripted
learner drives the long-corridor and blocked two-path scenarios, whose failure is a
policy-capacity effect the planner does not have; the real PPO acceptance is the
separate slow suite.
"""

from __future__ import annotations

import pytest

from faultline.learners import TabularLearner
from faultline.refinement import AutoSpecConfig
from faultline.scenarios import SCENARIOS, resolve_scenario, run_scenario
from faultline.scenarios.catalogue import LONG_CORRIDOR_GOAL
from tests.loop_support import (
    ScriptedLearner,
    blocked_rollouts,
    refinable_rollouts,
    seq_refined,
)

SEEDS = (0, 1, 2, 3, 4)
# The planner's estimates are deterministic, so a relaxed threshold is enough to
# expose the operator ordering without the paper's strict 0.99 used by the slow
# PPO suite. The rollout count is high enough that PastRefine's mean-bisector
# half-plane lands on the true boundary for every seed.
PLANNER_OPTIONS = {"threshold": 0.9, "n_rollouts": 80}
TABULAR_OPERATORS = (("trap_goal", "seq"), ("mixed_start", "past"))


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
    assert parents == {"mid1", "mid2"}


def test_narrow_door_layout_is_open_arena_with_one_wall():
    """The y=16 wall has the narrow trap door, a fully open middle edge, and a solid right edge."""
    layout = SCENARIOS["narrow_door"].layout
    assert layout.is_open(("v", 0, 1), 12.0) is True
    assert layout.is_open(("v", 1, 1), 12.0) is True
    assert layout.is_open(("h", 0, 0), 4.0) is True
    # The left doorway stays the narrow trap doorway.
    assert layout.is_open(("h", 0, 1), 3.0) is True
    # The middle edge is fully open; the right edge is now solid.
    assert layout.is_open(("h", 1, 1), 9.0) is True
    assert layout.is_open(("h", 1, 1), 15.5) is True
    assert layout.is_open(("h", 2, 1), 21.0) is False
    assert SCENARIOS["narrow_door"].unsafe_region is not None


def test_long_corridor_layout_winds_around_the_sealed_right_column():
    """The corridor climbs the left, crosses the top, and drops into the centre."""
    layout = SCENARIOS["long_corridor"].layout
    assert layout.is_open(("h", 0, 0), 4.0) is True
    assert layout.is_open(("h", 0, 1), 4.0) is True
    assert layout.is_open(("v", 0, 2), 20.0) is True
    assert layout.is_open(("h", 1, 1), 12.0) is True
    # The direct shortcut past the corridor and the right column stay walled off.
    assert layout.is_open(("v", 0, 0), 4.0) is False
    assert layout.is_open(("h", 2, 1), 20.0) is False


def test_mixed_start_layout_is_the_paper_long_mixed_star_room():
    """The star's corridor zig-zags from START to the top-right goal."""
    layout = SCENARIOS["mixed_start"].layout
    assert layout.is_open(("h", 0, 0), 4.0) is True
    assert layout.is_open(("v", 0, 1), 12.0) is True
    assert layout.is_open(("v", 1, 1), 12.0) is True
    assert layout.is_open(("h", 2, 1), 20.0) is True
    # The sealed top-left room and the solid divider across the right column.
    assert layout.is_open(("h", 0, 1), 4.0) is False
    assert layout.is_open(("h", 2, 0), 20.0) is False


def test_blocked_two_path_layout_winds_between_two_blocked_shortcuts():
    """The two-path room blocks both direct edges and only the winding route is open."""
    layout = SCENARIOS["blocked_two_path"].layout
    # The centre column and the two mid merges are open.
    assert layout.is_open(("v", 0, 2), 20.0) is True
    assert layout.is_open(("v", 1, 0), 4.0) is True
    assert layout.is_open(("h", 1, 1), 12.0) is True
    assert layout.is_open(("h", 1, 0), 12.0) is True
    # The left and right column walls are open across their whole edge, not a
    # centred gap, so an off-centre crossing passes.
    assert layout.is_open(("h", 0, 0), 1.0) is True
    assert layout.is_open(("h", 0, 1), 7.0) is True
    assert layout.is_open(("h", 2, 0), 17.0) is True
    assert layout.is_open(("h", 2, 1), 23.0) is True
    # Both direct shortcuts stay solid.
    assert layout.is_open(("v", 0, 0), 4.0) is False
    assert layout.is_open(("v", 1, 2), 20.0) is False


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


def test_narrow_door_trap_is_beyond_the_safety_blind_planner():
    # The trap lives in the environment, not the specification, and the planner
    # never consults the safe region: it walks straight into the door and fails, so
    # no operator raises the measured success. The paper's AvoidRefine acceptance is
    # the separate slow PPO suite.
    result = _planner_run("narrow_door")
    assert result.operators == (None,)
    assert result.best_path_after == result.best_path_before


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
        if seq_refined(edge, LONG_CORRIDOR_GOAL) and edge.target.name == "goal":
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
        blocked = {("mid1", "goal"), ("start", "mid2")}
        if (edge.source.name, edge.target.name) in blocked:
            return 0.4
        return 0.9

    learner = ScriptedLearner(probability, blocked_rollouts)
    result = run_scenario(SCENARIOS["blocked_two_path"], learner, AutoSpecConfig(threshold=0.8))

    assert "or" in result.operators
    assert result.best_path_before < 0.6
    assert result.best_path_after > result.best_path_before


def test_repeated_tabular_run_is_deterministic():
    first = _planner_run("trap_goal", seed=3)
    second = _planner_run("trap_goal", seed=3)

    assert first.operators == second.operators
    assert first.edges == second.edges
    assert first.best_path_after == second.best_path_after


def test_resolver_handles_9_rooms_fixed_and_random_names():
    """The resolver covers the three name forms a worker may receive."""
    assert resolve_scenario("trap_goal") is SCENARIOS["trap_goal"]

    for name in ("fig3", "fig6", "random_seed0"):
        scenario = resolve_scenario(name)
        assert scenario.name == name
        assert scenario.world is not None
        assert scenario.make_env().layout.world_size == 80.0

    generated = resolve_scenario("rooms100_random_0")
    assert generated.name == "rooms100_random_0"
    assert generated.make_env().max_steps >= 200

    with pytest.raises(KeyError, match="Unknown scenario"):
        resolve_scenario("missing")


def test_world_scenario_has_no_paper_reference_and_records_expectations():
    """A 100-rooms scenario is unscored and carries its designed expectations."""
    scenario = resolve_scenario("fig3")

    assert scenario.reference_before is None
    assert scenario.reference_after is None
    assert scenario.expected_operator is None
    assert scenario.expected_operators == {
        "Start->M1": ("seq",),
        "M1->M3": ("seq",),
        "M2->M3": ("seq",),
    }
    assert scenario.metadata["world"]["design"].startswith("hand-designed Fig. 3")
    assert scenario.make_env().max_steps == 200
