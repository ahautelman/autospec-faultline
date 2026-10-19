"""End-to-end plumbing of the runner on a tiny gold with the tabular learner."""

from __future__ import annotations

import json

import numpy as np
import pytest

from faultline.benchmark.instances import Instance, generate, instance_rng
from faultline.benchmark.runner import Settings, gold_baselines, run_instance, write_record
from faultline.envs.rooms.layout import RoomsLayout
from faultline.faults.gates import Funnel
from faultline.faults.ground_truth import FaultKind, MutationParams
from faultline.faults.mutations import mutate
from faultline.golden.builder import build_gold, chain_graph
from faultline.protocol import load_protocol
from faultline.spectrl.predicates import Box


@pytest.fixture(scope="module")
def gold():
    return build_gold("tiny", RoomsLayout(doors={}),
                      chain_graph(Box((1, 1), (3, 3)), (), Box((4, 1), (8, 7))), resolution=24, max_steps=20)


def test_generate_is_deterministic_and_counts_everything(gold):
    protocol = load_protocol()
    runs = []
    for _ in range(2):
        funnel = Funnel()
        inst = generate([gold], families=(FaultKind.GOAL_INFLATION, FaultKind.HAZARD_UNLABELED),
                        seeds=(0, 1), protocol=protocol, funnel=funnel, audit_samples=50)
        runs.append(([i.id for i in inst], funnel.to_dict()))
    assert runs[0] == runs[1]
    counts = runs[0][1]["counts"]
    assert counts["hazard_unlabeled"]["not_applicable"] == 6          # gold has no trap
    assert counts["goal_inflation"]["valid"] == 6
    nulls = [i for i in inst if i.kind is FaultKind.NULL]
    assert len(nulls) / len(inst) >= 0.2 - 1e-9


def test_run_instance_record_with_tabular_learner(gold, tmp_path):
    protocol = load_protocol()
    settings = Settings(protocol, learner="tabular", method="base", gate_seeds=(0,), budget=2000, n_rollouts=50, audit_samples=40)
    repair = mutate(FaultKind.GOAL_INFLATION, gold, MutationParams(), instance_rng("x"))
    inst = Instance("tiny|goal_inflation|mid|0", gold, FaultKind.GOAL_INFLATION, "mid", 0, repair)
    record = run_instance(inst, settings, gold_baselines(settings, gold), seeds=(0,), skip_gates=True)
    assert record["gate"]["stage"] in {"accepted", "unobservable", "ineffective"}
    assert record["protocol_hash"] == protocol.hash and record["git"]
    assert set(record["baselines"]) == {"unrefined", "oracle", "trivializer"}
    assert len(record["runs"]) == 1 and "autospec_eval" in record["runs"][0]
    path = write_record(record, tmp_path)
    assert json.loads(path.read_text())["id"] == inst.id


def test_hazard_unlabeled_runs_despite_the_gate_and_is_stratified(tmp_path):
    protocol = load_protocol()
    trap = Box((3, 2), (6, 6))
    gold = build_gold("trap", RoomsLayout(doors={}), chain_graph(Box((1, 3), (2, 5)), (), Box((7, 3), (8, 5))), (trap,),
                      resolution=24, max_steps=20)
    settings = Settings(protocol, learner="tabular", method="base", gate_seeds=(0,), budget=2000, n_rollouts=50, audit_samples=40)
    repair = mutate(FaultKind.HAZARD_UNLABELED, gold, MutationParams(), instance_rng("m2"))
    inst = Instance("trap|hazard_unlabeled|mid|0", gold, FaultKind.HAZARD_UNLABELED, "mid", 0, repair)
    record = run_instance(inst, settings, gold_baselines(settings, gold), seeds=(0,))
    assert record["stratum"] == "ineffective_by_construction"
    assert len(record["runs"]) == 1
