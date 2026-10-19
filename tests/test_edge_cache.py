"""The edge-estimate cache must be exact: a cached run equals an uncached run bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

from faultline.envs.rooms.layout import RoomsLayout
from faultline.faults.ground_truth import FaultKind, MutationParams
from faultline.faults.mutations import mutate
from faultline.benchmark.instances import instance_rng
from faultline.golden.builder import build_gold, chain_graph
from faultline.golden.env import make_env
from faultline.learners import DiRL, PPOEdgeLearner, TabularLearner
from faultline.learners.base import EdgeEstimate, Rollout
from faultline.learners.cache import (
    CachedLearner, EstimateStore, decode_estimate, encode_estimate, estimate_key, fingerprint,
)
from faultline.refinement import AutoSpecConfig, run_autospec
from faultline.spectrl.predicates import Box


def _summary(run) -> tuple:
    """Everything a benchmark record reads from a run, as comparable values."""
    return (
        run.best_path_success,
        tuple((e.source, e.target, e.probability) for e in run.initial_edge_success),
        tuple((e.source, e.target, e.probability) for e in run.edge_success),
        tuple((s.source, s.target, s.operators_tried, s.operator, s.before, s.after) for s in run.steps),
        tuple((v.name, fingerprint(v.region)) for v in run.graph.vertices),
        tuple((e.source.name, e.target.name, fingerprint(e.safe)) for e in run.graph.edges),
    )


@pytest.fixture(scope="module")
def faulty():
    gold = build_gold("tiny", RoomsLayout(doors={}),
                      chain_graph(Box((1, 1), (3, 3)), (Box((4, 1), (6, 3)),), Box((4, 5), (8, 7))),
                      resolution=24, max_steps=20)
    repair = mutate(FaultKind.GOAL_INFLATION, gold, MutationParams(level="high"), instance_rng("cache"))
    return gold, repair.faulty


def test_fingerprint_is_exact():
    assert fingerprint(Box((0, 0), (1, 1))) == fingerprint(Box((0.0, 0.0), (1.0, 1.0)))
    assert fingerprint(Box((0, 0), (1, 1))) != fingerprint(Box((0, 0), (1, np.nextafter(1.0, 2.0))))
    with pytest.raises(TypeError):
        fingerprint(lambda: 0)


def test_encode_decode_round_trip_is_lossless():
    rollouts = (
        Rollout((np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32).astype(float),
                 np.array([1.5, 2.25, 0.1, 2.0])), reached=True, safe=True, final_safe=True),
        Rollout((np.array([0.1, 0.2, 0.3, 0.4]),), reached=False, safe=False, final_safe=None),
    )
    back = decode_estimate(encode_estimate(EdgeEstimate(0.5, rollouts)))
    assert back.success_probability == 0.5
    for a, b in zip(rollouts, back.rollouts):
        assert (a.reached, a.safe, a.final_safe) == (b.reached, b.safe, b.final_safe)
        assert all(np.array_equal(x, y) for x, y in zip(a.states, b.states))


def test_key_depends_on_every_input(faulty):
    gold, graph = faulty
    env, edge = make_env(gold), graph.edges[0]
    base = dict(learner_id=b"x", env=env, edge=edge, budget=10, train_seed=0, n_rollouts=5, estimate_seed=0)
    key = estimate_key(**base)
    assert key == estimate_key(**base)
    for name, value in (("learner_id", b"y"), ("edge", graph.edges[1]), ("budget", 11), ("train_seed", 1),
                        ("n_rollouts", 6), ("estimate_seed", 1)):
        assert estimate_key(**{**base, name: value}) != key, name
    trapped = build_gold("t", gold.layout, gold.graph, (Box((6.5, 0.5), (7, 1)),), resolution=24, max_steps=20)
    assert estimate_key(**{**base, "env": make_env(trapped)}) != key


@pytest.mark.parametrize("make_learner,budget", [
    pytest.param(TabularLearner, 0, id="tabular"),
    pytest.param(lambda: PPOEdgeLearner(device="cpu"), 2048, id="ppo"),
])
def test_cached_run_equals_uncached_run(faulty, tmp_path, make_learner, budget):
    if budget:
        pytest.importorskip("stable_baselines3")
    gold, graph = faulty
    config = AutoSpecConfig(threshold=0.8, budget=budget, n_rollouts=40, seed=3)
    plain = run_autospec(graph, make_env(gold), make_learner(), config, DiRL())
    store = EstimateStore(tmp_path)
    cold = run_autospec(graph, make_env(gold), CachedLearner(make_learner(), store), config, DiRL())
    trained = store.misses
    warm_store = EstimateStore(tmp_path)
    warm = run_autospec(graph, make_env(gold), CachedLearner(make_learner(), warm_store), config, DiRL())
    assert _summary(plain) == _summary(cold) == _summary(warm)
    assert trained > 0 and warm_store.misses == 0 and warm_store.hits == store.hits + store.misses
