"""T7 (baselines order the metrics correctly) and T8 (instrumentation changes nothing)."""

from __future__ import annotations

import numpy as np
import pytest

from faultline.baselines import graph_baselines, random_operator
from faultline.envs.rooms.layout import RoomsLayout
from faultline.faults.ground_truth import FaultKind, MutationParams
from faultline.faults.mutations import mutate
from faultline.golden.builder import build_gold, chain_graph
from faultline.golden.raster import CellGrid
from faultline.metrics.evaluate import evaluate_repair
from faultline.refinement import AutoSpecConfig, run_autospec
from faultline.refinement import loop as loop_module
from faultline.spectrl.predicates import Box
from tests.loop_support import ScriptedLearner, refinable_rollouts, seq_refined
from tests.refinement_support import MID, chain_graph as loop_chain

RES = 24


@pytest.fixture(scope="module")
def setup():
    gold = build_gold("pocket", RoomsLayout(doors={}),
                      chain_graph(Box((1, 1), (3, 3)), (), Box((4, 1), (8, 7))), resolution=RES, max_steps=20)
    repair = mutate(FaultKind.GOAL_INFLATION, gold, MutationParams(), np.random.default_rng(0))
    grid = CellGrid(gold.layout, RES)
    base = graph_baselines(repair, grid)
    ev = {n: evaluate_repair(repair, g, grid, np.random.default_rng(1), audit_samples=150)
          for n, g in [("unrefined", base.unrefined), ("oracle", base.oracle), ("trivializer", base.trivializer),
                       ("gold", repair.gold)]}
    return repair, ev


def locus(ev):
    return next(iter(ev["loci"].values()))


def test_t7_unrefined_scores_zero_and_oracle_scores_perfect(setup):
    _, ev = setup
    assert locus(ev["unrefined"])["f1"] == 0.0
    assert locus(ev["oracle"])["f1"] == 1.0 and locus(ev["oracle"])["attainment"] == 1.0
    assert ev["oracle"]["ged"] == 0.0 < ev["unrefined"]["ged"]


def test_t7_trivializer_wins_removal_but_loses_retention_and_f1(setup):
    _, ev = setup
    triv = locus(ev["trivializer"])
    assert triv["scores"]["removal"] == 1.0
    assert triv["scores"]["retention"] < 0.2
    assert triv["f1"] < 0.3 < locus(ev["oracle"])["f1"]
    assert ev["trivializer"]["ged"] > ev["oracle"]["ged"]
    assert ev["trivializer"]["audit"]["gold_kept"] < 0.5


def test_t7_baselines_are_ordered_by_f1(setup):
    _, ev = setup
    f = {n: locus(ev[n])["f1"] for n in ("unrefined", "trivializer", "oracle")}
    assert f["unrefined"] < f["trivializer"] < f["oracle"]


def test_baselines_do_not_mutate_the_mutant(setup):
    repair, _ = setup
    grid = CellGrid(build_gold("x", RoomsLayout(doors={}),
                    chain_graph(Box((1, 1), (3, 3)), (), Box((4, 1), (8, 7))), resolution=RES, max_steps=20).layout, RES)
    before = repr(repair.faulty.vertices[-1].region)
    graph_baselines(repair, grid)
    assert repr(repair.faulty.vertices[-1].region) == before


# ---- T8: the instrumentation does not change the loop ------------------------------


def _scripted():
    def probability(edge):
        if seq_refined(edge, MID) or (edge.source.name == "mid" and edge.target.name == "goal"):
            return 0.99
        return 0.5
    return ScriptedLearner(probability, refinable_rollouts)


def test_t8_random_operator_restores_the_loop_and_matches_direct_run_for_identity_order():
    config = AutoSpecConfig(threshold=0.9)
    original = loop_module._CANDIDATE_BUILDERS
    graph, _ = loop_chain()
    direct = run_autospec(graph, None, _scripted(), config)
    # a permutation that happens to keep seq first must commit the same operator
    for seed in range(20):
        graph, _ = loop_chain()
        run = random_operator(graph, None, _scripted(), config, np.random.default_rng(seed))
        assert loop_module._CANDIDATE_BUILDERS is original
        names = [n for n, _b in original]
        assert run.steps[0].operator is not None
        # only seq clears in this script, so any order commits seq
        assert run.steps[0].operator == direct.steps[0].operator == "seq"
        assert run.best_path_success == direct.best_path_success


def test_t8_random_operator_changes_order_of_tries():
    config = AutoSpecConfig(threshold=0.9)
    orders = set()
    for seed in range(30):
        graph, _ = loop_chain()
        run = random_operator(graph, None, _scripted(), config, np.random.default_rng(seed))
        orders.add(run.steps[0].operators_tried)
    assert len(orders) > 1


def test_t8_random_operator_is_deterministic_in_its_rng():
    config = AutoSpecConfig(threshold=0.9)
    runs = []
    for _ in range(2):
        graph, _ = loop_chain()
        runs.append(random_operator(graph, None, _scripted(), config, np.random.default_rng(5)))
    assert runs[0].steps[0].operators_tried == runs[1].steps[0].operators_tried


def test_t8_evaluate_is_deterministic(setup):
    repair, _ = setup
    grid = CellGrid(repair_layout(), RES)
    a = evaluate_repair(repair, repair.faulty, grid, np.random.default_rng(3), audit_samples=50)
    b = evaluate_repair(repair, repair.faulty, grid, np.random.default_rng(3), audit_samples=50)
    assert a == b


def repair_layout():
    return RoomsLayout(doors={})
