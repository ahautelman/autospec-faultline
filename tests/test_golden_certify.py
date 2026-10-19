"""Tier 2 and 3 certification, with a scripted loop and with the real tabular + DiRL stack."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from faultline.envs.rooms.layout import RoomsLayout
from faultline.envs.rooms.env import RoomsEnv
from faultline.golden.builder import build_gold, chain_graph
from faultline.golden.certify import certify_learning
from faultline.learners import DiRL, TabularLearner
from faultline.refinement import AutoSpecConfig
from faultline.spectrl.predicates import Box

CONFIG = AutoSpecConfig(budget=0, n_rollouts=100, threshold=0.8)


def make_gold():
    return build_gold(
        "room", RoomsLayout(doors={}),
        chain_graph(Box((1.0, 1.0), (3.0, 3.0)), (), Box((5.0, 1.0), (7.0, 3.0))),
        resolution=24, max_steps=20,
    )


def scripted(probability, refine_seed=None):
    def run(graph, env, learner, config, graph_learner):
        steps = [SimpleNamespace(source="start", target="goal", operator="seq")] if config.seed == refine_seed else []
        return SimpleNamespace(
            initial_edge_success=[SimpleNamespace(source="start", target="goal", probability=probability)],
            steps=steps,
        )
    return run


def certify(run, **overrides):
    arguments = dict(config=CONFIG, cert_seeds=(100, 101), holdout_seeds=(105,), p_gold_min=0.85, run=run)
    arguments.update(overrides)
    return certify_learning(make_gold(), lambda: None, lambda: None, lambda: None, **arguments)


def test_a_learnable_stable_gold_is_certified():
    certificate = certify(scripted(1.0)).certificate
    assert certificate.certified and certificate.p_gold == 0.85
    assert len(certificate.edge_cp_lower["start->goal"]) == 2


def test_a_weak_edge_fails_tier_two_using_the_lower_bound_not_the_point_estimate():
    # 0.88 observed over 100 rollouts has a lower bound below 0.85.
    certificate = certify(scripted(0.88)).certificate
    assert certificate.tier2_ok is False and not certificate.certified


def test_a_refinement_on_gold_fails_tier_three():
    certificate = certify(scripted(1.0, refine_seed=105)).certificate
    assert certificate.tier3_ok is False
    assert any("hold-out seed 105" in problem for problem in certificate.problems)


def test_an_unlearned_edge_fails_tier_two():
    certificate = certify(scripted(None)).certificate
    assert certificate.tier2_ok is False


def test_overlapping_seed_sets_are_rejected():
    with pytest.raises(ValueError):
        certify(scripted(1.0), holdout_seeds=(101,))


def test_a_spec_that_failed_tier_one_is_returned_unchanged():
    bad = build_gold(
        "sealed", RoomsLayout(doors={}),
        chain_graph(Box((9.0, 1.0), (11.0, 3.0)), (), Box((1.0, 1.0), (3.0, 3.0))),
        resolution=24, max_steps=20,
    )
    result = certify_learning(bad, lambda: None, lambda: None, lambda: None, config=CONFIG,
                              cert_seeds=(1,), holdout_seeds=(2,), p_gold_min=0.85, run=scripted(1.0))
    assert result is bad and result.certificate.tier2_ok is None


def test_the_real_tabular_dirl_stack_certifies_an_open_room_gold():
    gold = make_gold()
    certified = certify_learning(
        gold, lambda: RoomsEnv(layout=gold.layout, max_steps=gold.max_steps),
        TabularLearner, DiRL, config=CONFIG, cert_seeds=(100,), holdout_seeds=(105,), p_gold_min=0.85,
    )
    assert certified.certificate.certified, certified.certificate.problems
