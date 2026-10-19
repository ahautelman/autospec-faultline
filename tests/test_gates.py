"""Effectiveness and observability gates and the funnel, on scripted baselines."""

from __future__ import annotations

import numpy as np
import pytest

from faultline.faults.gates import (
    Funnel, Stage, best_path, check_effectiveness, check_observability, learn_baseline, screen,
    screen_with_learning, witness_count,
)
from faultline.faults.ground_truth import FaultKind, Locus, MutationParams
from faultline.faults.mutations import mutate, null_mutation
from faultline.golden.builder import build_gold, chain_graph
from faultline.envs.rooms.layout import RoomsLayout
from faultline.learners.base import EdgeEstimate, Rollout
from faultline.spectrl.predicates import Box

ARGS = dict(threshold=0.8, min_gap=0.15, seed_fraction=0.8, min_witnesses=20)


def gold():
    return build_gold("g", RoomsLayout(doors={}), chain_graph(Box((1, 1), (3, 3)), (), Box((4, 1), (8, 7))),
                      resolution=24, max_steps=20)


def mutant(kind=FaultKind.GOAL_INFLATION):
    return mutate(kind, gold(), MutationParams(), np.random.default_rng(0))


def roll(success=True, ended_safe=True):
    state = (np.zeros(4),)
    return Rollout(state, reached=success, safe=success or ended_safe, final_safe=ended_safe)


def estimate(p, n=100, ended_unsafe=0):
    k = round(p * n)
    rolls = [roll(True)] * k + [roll(False, ended_safe=True)] * (n - k - ended_unsafe)
    rolls += [roll(False, ended_safe=False)] * ended_unsafe
    return EdgeEstimate(p, tuple(rolls))


def baselines(graph, p, seeds=(0, 1, 2, 3, 4), **kw):
    return {s: {f"{e.source.name}->{e.target.name}": estimate(p, **kw) for e in graph.edges} for s in seeds}


def test_best_path_uses_probabilities_and_zero_for_unlearned():
    g = mutant().gold
    key = f"{g.edges[0].source.name}->{g.edges[0].target.name}"
    assert best_path(g, {key: estimate(0.9)}) == pytest.approx(0.9)
    assert best_path(g, {key: None}) == 0.0


def test_effective_when_faulty_trails_gold():
    m = mutant()
    r = check_effectiveness(m, baselines(m.gold, 0.95), baselines(m.faulty, 0.5),
                            threshold=0.8, min_gap=0.15, seed_fraction=0.8)
    assert r.effective and r.fraction == 1.0


def test_ineffective_when_fault_is_harmless():
    m = mutant()
    r = check_effectiveness(m, baselines(m.gold, 0.95), baselines(m.faulty, 0.93),
                            threshold=0.8, min_gap=0.15, seed_fraction=0.8)
    assert not r.effective


def test_gap_without_below_threshold_edge_is_not_effective():
    m = mutant()
    # gap 0.17 but the faulty edge (0.81) is above the threshold: loop would not act
    r = check_effectiveness(m, baselines(m.gold, 0.98), baselines(m.faulty, 0.81),
                            threshold=0.8, min_gap=0.15, seed_fraction=0.8)
    assert not r.effective


def test_effectiveness_seed_fraction():
    m = mutant()
    faulty = baselines(m.faulty, 0.5)
    faulty[3] = baselines(m.faulty, 0.97)[3]
    faulty[4] = baselines(m.faulty, 0.97)[4]
    r = check_effectiveness(m, baselines(m.gold, 0.97), faulty, threshold=0.8, min_gap=0.15, seed_fraction=0.8)
    assert r.fraction == pytest.approx(0.6) and not r.effective


def test_effectiveness_rejects_mismatched_seeds():
    m = mutant()
    with pytest.raises(ValueError):
        check_effectiveness(m, baselines(m.gold, 0.9, seeds=(0,)), baselines(m.faulty, 0.5, seeds=(1,)), **{
            "threshold": 0.8, "min_gap": 0.1, "seed_fraction": 0.5})


def test_witness_counts_per_family():
    e = estimate(0.6, n=100, ended_unsafe=10)
    assert witness_count(FaultKind.GOAL_INFLATION, e) == 60
    assert witness_count(FaultKind.WAYPOINT_CONTRACTION, e) == 60
    assert witness_count(FaultKind.HAZARD_UNLABELED, e) == 10
    assert witness_count(FaultKind.DEAD_START, e) == 40
    assert witness_count(FaultKind.ROUTE_PRUNING, e) is None


def test_observability_needs_witnesses_on_every_seed():
    m = mutant()
    ok = check_observability(m, baselines(m.faulty, 0.6), min_witnesses=20)
    assert ok.observable
    poor = baselines(m.faulty, 0.6)
    poor[2] = baselines(m.faulty, 0.1, seeds=(2,))[2]
    assert not check_observability(m, poor, min_witnesses=20).observable


def test_route_pruning_and_null_need_no_samples():
    g = gold()
    assert check_observability(null_mutation(g), baselines(g.graph, 0.9), min_witnesses=20).observable


def test_screen_stages():
    m = mutant()
    gold_b = baselines(m.gold, 0.95)
    assert screen(m, gold_b, baselines(m.faulty, 0.93), **ARGS).stage is Stage.INEFFECTIVE
    assert screen(m, gold_b, baselines(m.faulty, 0.05), **ARGS).stage is Stage.UNOBSERVABLE
    assert screen(m, gold_b, baselines(m.faulty, 0.5), **ARGS).stage is Stage.ACCEPTED
    null = null_mutation(gold())
    assert screen(null, gold_b, gold_b, **ARGS).stage is Stage.ACCEPTED


def test_screen_with_learning_calls_learner_per_seed():
    m = mutant()
    seen = []

    def learn(graph, seed):
        seen.append(seed)
        return baselines(graph, 0.5, seeds=(seed,))[seed]

    result = screen_with_learning(m, baselines(m.gold, 0.95, seeds=(0, 1, 2)), (0, 1, 2), learn, **ARGS)
    assert seen == [0, 1, 2] and result.stage is Stage.ACCEPTED


def test_funnel_counts_and_reasons():
    f = Funnel()
    for _ in range(2):
        f.record(FaultKind.GOAL_INFLATION, Stage.VALID)
    f.record(FaultKind.GOAL_INFLATION, Stage.ACCEPTED)
    f.record(FaultKind.GOAL_INFLATION, Stage.INEFFECTIVE, "harmless", "a")
    f.record(FaultKind.DEAD_START, Stage.NOT_APPLICABLE, "no site", "b")
    out = f.to_dict()
    assert out["counts"]["goal_inflation"] == {"valid": 2, "accepted": 1, "ineffective": 1, "generated": 2}
    assert f.generated(FaultKind.DEAD_START) == 1
    assert [r["reason"] for r in out["rejections"]] == ["harmless", "no site"]


class _FakeLearner:
    def train_edge(self, env, edge, budget, seed):
        return None

    def estimate_reach(self, env, edge, policy, n, seed):
        return estimate(0.7)


def test_learn_baseline_without_graph_learner_trains_every_edge():
    g = gold().graph
    out = learn_baseline(g, None, _FakeLearner(), None, budget=1, n_rollouts=1, seed=0)
    assert set(out) == {f"{e.source.name}->{e.target.name}" for e in g.edges}
