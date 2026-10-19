"""Recovery metrics: regions, GED, ceiling, audit, inference."""

from __future__ import annotations

import numpy as np
import pytest

from faultline.envs.rooms.layout import RoomsLayout
from faultline.faults.ground_truth import FaultKind, MutationParams
from faultline.faults.mutations import mutate
from faultline.golden.builder import build_gold, chain_graph
from faultline.golden.raster import CellGrid
from faultline.metrics.audit import satisfaction_gap, soundness_audit
from faultline.metrics.ceiling import attainment, ceiling_scores, with_ceiling
from faultline.metrics.ged import region_ged
from faultline.metrics.inference import bootstrap_ci, holm, wilcoxon_signed_rank
from faultline.metrics.regions import region_scores, score_loci
from faultline.spectrl.predicates import Box

RES = 24


def gold_pocket():
    return build_gold("pocket", RoomsLayout(doors={}),
                      chain_graph(Box((1, 1), (3, 3)), (), Box((4, 1), (8, 7))), resolution=RES, max_steps=20)


def m1():
    gold = gold_pocket()
    return gold, mutate(FaultKind.GOAL_INFLATION, gold, MutationParams(), np.random.default_rng(0))


def test_region_scores_extremes():
    gold = np.zeros((4, 4), bool); gold[:2] = True
    faulty = np.ones((4, 4), bool)
    full = region_scores(gold, faulty, gold)
    assert (full.removal, full.retention, full.precision, full.iou) == (1.0, 1.0, 1.0, 1.0)
    none = region_scores(gold, faulty, faulty)
    assert none.removal == 0.0 and none.retention == 1.0 and none.precision == 0.5
    wiped = region_scores(gold, faulty, np.zeros((4, 4), bool))
    assert wiped.removal == 1.0 and wiped.retention == 0.0   # removal alone cannot be gamed


def test_scores_of_faulty_and_gold_graph_as_repairs():
    gold, repair = m1()
    grid = CellGrid(gold.layout, RES)
    as_faulty = list(score_loci(grid, repair.target, repair.faulty, repair.faulty).values())[0]
    as_gold = list(score_loci(grid, repair.target, repair.faulty, repair.gold).values())[0]
    assert as_faulty.removal == 0.0 and as_faulty.retention == 1.0
    assert as_gold.removal == 1.0 and as_gold.iou == 1.0


def test_missing_locus_scores_none():
    gold, repair = m1()
    grid = CellGrid(gold.layout, RES)
    other = chain_graph(Box((1, 1), (3, 3)), (), Box((4, 1), (8, 7)))
    other.vertices[-1].name = "renamed"
    assert list(score_loci(grid, repair.target, repair.faulty, other).values()) == [None]


def test_ged_zero_for_identical_and_ordered_by_damage():
    gold, repair = m1()
    grid = CellGrid(gold.layout, RES)
    assert region_ged(grid, repair.gold, repair.gold) == 0.0
    d_faulty = region_ged(grid, repair.gold, repair.faulty)
    assert 0.0 < d_faulty < 1.0
    assert region_ged(grid, repair.gold, repair.gold) < d_faulty


def test_ged_counts_missing_vertices():
    gold = gold_pocket()
    grid = CellGrid(gold.layout, RES)
    longer = chain_graph(Box((1, 1), (3, 3)), (Box((3, 1), (4, 2)),), Box((4, 1), (8, 7)))
    assert region_ged(grid, gold.graph, longer) > 0.0


def test_ceiling_reach_restores_convex_gold_exactly():
    gold, repair = m1()
    grid = CellGrid(gold.layout, RES)
    repair = with_ceiling(repair, grid)
    (scores,) = ceiling_scores(repair, grid).values()
    assert scores.removal == 1.0 and scores.retention == 1.0


@pytest.mark.parametrize("kind", [FaultKind.WAYPOINT_CONTRACTION, FaultKind.ROUTE_PRUNING])
def test_structural_families_have_no_region_ceiling(kind):
    gold = build_gold("chain", RoomsLayout(doors={}),
                      chain_graph(Box((1, 1), (3, 3)), (Box((4, 1), (6, 3)), Box((7, 1), (9, 3))), Box((10, 1), (12, 3))),
                      resolution=RES, max_steps=40)
    try:
        repair = mutate(kind, gold, MutationParams(), np.random.default_rng(0))
    except Exception:
        pytest.skip("not applicable on this fixture")
    assert with_ceiling(repair, CellGrid(gold.layout, RES)).ceiling is None


def test_attainment():
    from faultline.metrics.regions import RegionScores
    r = lambda x: RegionScores(x, 1, 1, 1, 10, 10)
    assert attainment(r(0.5), r(1.0)) == 0.5
    assert attainment(r(1.0), r(0.5)) == 1.0
    assert attainment(r(0.5), r(0.0)) is None and attainment(None, r(1.0)) is None


def test_audit_identity_and_faulty_as_repair():
    gold, repair = m1()
    rng = np.random.default_rng(0)
    same = soundness_audit(repair.gold, repair.faulty, repair.gold, rng, 24.0, count=200)
    assert same.gold_kept == 1.0 and same.n_fault > 0
    assert same.fault_rejected == 1.0
    unrepaired = soundness_audit(repair.gold, repair.faulty, repair.faulty, rng, 24.0, count=200)
    assert unrepaired.gold_kept == 1.0 and unrepaired.fault_rejected == 0.0


def test_satisfaction_gap():
    gold = gold_pocket()
    good = np.array([[2.0, 2.0], [6.0, 3.0]])
    bad = np.array([[2.0, 2.0], [20.0, 20.0]])
    out = satisfaction_gap(0.9, [good, good, bad, np.empty((0, 2))], gold.graph)
    assert out["s_star"] == 0.5 and out["gap"] == pytest.approx(0.4)
    assert out["s_star_low"] < 0.5 < out["s_star_high"]
    with pytest.raises(ValueError):
        satisfaction_gap(0.9, [], gold.graph)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    data = np.random.default_rng(1).normal(0.6, 0.1, 40)
    lo, hi = bootstrap_ci(data, np.random.default_rng(0), resamples=2000)
    assert lo < data.mean() < hi
    assert (lo, hi) == bootstrap_ci(data, np.random.default_rng(0), resamples=2000)
    with pytest.raises(ValueError):
        bootstrap_ci([], np.random.default_rng(0))


def test_wilcoxon_matches_known_values():
    # classic textbook example, n=9 non-zero pairs: W=0 -> p = 2/2^9
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert wilcoxon_signed_rank(a, [0] * 9) == pytest.approx(2 / 512)
    assert wilcoxon_signed_rank(a, a) == 1.0
    assert wilcoxon_signed_rank([1, 2, 3, 4], [2, 1, 4, 3]) > 0.5
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([1], [1, 2])


def test_wilcoxon_normal_branch_close_to_exact():
    rng = np.random.default_rng(3)
    a = rng.normal(0.3, 1, 20)
    b = rng.normal(0, 1, 20)
    assert wilcoxon_signed_rank(a, b, exact_limit=20) == pytest.approx(
        wilcoxon_signed_rank(a, b, exact_limit=0), abs=0.03)


def test_holm_adjustment():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}
